"""
MR-SYNTH: Metadata-Conditioned 3D MRI Synthesis Inference Script
================================================================
Generates 3D brain MRI volumes conditioned on acquisition metadata strings
or CSV entries.

Usage Examples:
  # Direct prompt synthesis:
  python -m mr_synth.infer \\
      --metadata-text "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, MAGNETOM_Vida, 3.0), Acquisition (Description, Sequence, Variant): (t1_tse_tra, SE, SK_SP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.009, 0.55, NONE, 90.0)" \\
      --mrclip-ckpt weights/mr_synth_phase2.pt \\
      --clip-checkpoint weights/mr_clip_3d.pt \\
      --cfg-scale 3.0 \\
      --output-dir results/mr_synth_samples

  # From CSV dataset row:
  python -m mr_synth.infer \\
      --csv data/test.csv --csv-idx 0 \\
      --mrclip-ckpt weights/mr_synth_phase2.pt \\
      --clip-checkpoint weights/mr_clip_3d.pt \\
      --output-dir results/mr_synth_samples
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root and open_clip to path
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
for p in [str(SRC_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.utils import set_determinism

from open_clip import create_model_and_transforms, get_tokenizer
from .model import MRCLIPConditionedUNet
from .sample import generate_sample
from .setting import load_config, setup_logging
from .utils import define_instance
from .utils_plot import save_triplanar_figure


def load_clip_text_encoder(
    checkpoint_path: str,
    context_length: int = 98,
    device: str = "cuda",
) -> tuple[torch.nn.Module, Any]:
    """
    Load ViT-B-16 MR-CLIP text encoder and tokenizer.
    """
    model, _, _ = create_model_and_transforms(
        "ViT-B-16",
        device=device,
        textcontextlength=context_length,
        force_image_size=224,
    )
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        if next(iter(sd)).startswith("module."):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        model_sd = model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        model.load_state_dict(filtered, strict=False)

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = get_tokenizer("ViT-B-16", context_length=context_length)
    return model, tokenizer


@torch.no_grad()
def embed_text(text: str, clip_model: torch.nn.Module, tokenizer: Any, device: str = "cuda") -> torch.Tensor:
    """Encode text prompt -> (1, 512) normalized float32 tensor."""
    tokens = tokenizer([text]).to(device)
    feat = clip_model.encode_text(tokens)
    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
    return feat.float()


def load_mr_synth_models(
    args: argparse.Namespace,
    mrclip_ckpt_path: str | None,
    device: torch.device,
    logger: Any = None,
) -> tuple[torch.nn.Module, MRCLIPConditionedUNet, torch.Tensor]:
    """Load autoencoder, MRCLIPConditionedUNet, and latent scale factor."""
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    if hasattr(args, "trained_autoencoder_path") and os.path.exists(args.trained_autoencoder_path):
        ae_ckpt = torch.load(args.trained_autoencoder_path, map_location=device, weights_only=False)
        if "unet_state_dict" in ae_ckpt:
            ae_ckpt = ae_ckpt["unet_state_dict"]
        autoencoder.load_state_dict(ae_ckpt)
        if logger:
            logger.info(f"Loaded Autoencoder from {args.trained_autoencoder_path}")

    base_unet = define_instance(args, "diffusion_unet_def").to(device)
    model = MRCLIPConditionedUNet(unet=base_unet).to(device)

    scale_factor = torch.tensor(1.0)
    if mrclip_ckpt_path and os.path.exists(mrclip_ckpt_path):
        ckpt = torch.load(mrclip_ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("unet_state_dict", ckpt))
        cleaned_state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}

        has_wrapper_keys = any(k.startswith("clip_projection.") or k.startswith("unet.") for k in cleaned_state)
        if has_wrapper_keys:
            model.load_state_dict(cleaned_state, strict=False)
        else:
            model.unet.load_state_dict(cleaned_state, strict=False)

        scale_factor = ckpt.get("scale_factor", torch.tensor(1.0))
        if logger:
            logger.info(f"Loaded MR-SYNTH checkpoint from {mrclip_ckpt_path}")

    return autoencoder, model, scale_factor


def synthesize_from_prompt(
    metadata_text: str,
    mrclip_ckpt: str,
    clip_checkpoint: str,
    output_dir: str = "output/mr_synth",
    output_filename: str = "synthetic_brain.nii.gz",
    cfg_scale: float = 3.0,
    seed: int = 1234,
    output_size: tuple[int, int, int] = (256, 256, 128),
    spacing: tuple[float, float, float] = (0.94, 0.94, 1.36),
    num_inference_steps: int = 30,
    config_dir: str | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """
    High-level Python API to synthesize a 3D MRI volume from metadata string.
    """
    if config_dir is None:
        config_dir = str(PROJECT_ROOT / "configs" / "mr_synth")

    env_cfg = os.path.join(config_dir, "environment_maisi_diff_model_rflow-mr-brain.json")
    model_cfg = os.path.join(config_dir, "config_maisi_diff_model_rflow-mr-brain.json")
    net_cfg = os.path.join(config_dir, "config_network_rflow.json")

    args = load_config(env_cfg, model_cfg, net_cfg)
    logger = setup_logging(output_dir, "synth.log")

    device_t = torch.device(device)
    clip_model, tokenizer = load_clip_text_encoder(clip_checkpoint, device=device)
    clip_emb = embed_text(metadata_text, clip_model, tokenizer, device=device)

    autoencoder, model, scale_factor = load_mr_synth_models(args, mrclip_ckpt, device_t, logger)

    vol_np, _ = generate_sample(
        args=args,
        device=device_t,
        autoencoder=autoencoder,
        model=model,
        scale_factor=scale_factor,
        clip_embedding=clip_emb,
        output_size=output_size,
        spacing=spacing,
        cfg_scale=cfg_scale,
        seed=seed,
        num_inference_steps=num_inference_steps,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_nii_path = os.path.join(output_dir, output_filename)
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    nii = nib.Nifti1Image(vol_np.astype(np.float32), affine=affine)
    nib.save(nii, out_nii_path)

    out_png_path = out_nii_path.replace(".nii.gz", ".png").replace(".nii", ".png")
    save_triplanar_figure(vol_np, out_png_path, title=f"MR-SYNTH | Seed {seed}")
    logger.info(f"Synthesized volume saved: {out_nii_path}")
    logger.info(f"Tri-planar preview saved: {out_png_path}")

    return out_nii_path


def main():
    parser = argparse.ArgumentParser(description="MR-SYNTH 3D MRI Synthesis Inference")
    parser.add_argument("--metadata-text", "--prompt", type=str, default=None,
                        help="Clinical DICOM acquisition metadata string")
    parser.add_argument("--csv", type=str, default=None, help="Path to metadata CSV")
    parser.add_argument("--csv-idx", type=int, default=0, help="Row index in CSV")
    parser.add_argument("--mrclip-ckpt", type=str, default="weights/mr_synth_phase2.pt",
                        help="Path to MR-SYNTH checkpoint")
    parser.add_argument("--clip-checkpoint", type=str, default="weights/mr_clip_3d.pt",
                        help="Path to MR-CLIP text encoder checkpoint")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/mr_synth", help="Output directory")
    parser.add_argument("--output-filename", type=str, default="synth_mri.nii.gz", help="Output NIfTI filename")
    parser.add_argument("--dim", nargs=3, type=int, default=[256, 256, 128], help="Volume dimensions (D, H, W)")
    parser.add_argument("--spacing", nargs=3, type=float, default=[0.94, 0.94, 1.36], help="Voxel spacing (mm)")
    parser.add_argument("--num-steps", type=int, default=30, help="Inference diffusion steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    prompt = args.metadata_text
    if prompt is None:
        if args.csv is not None and os.path.exists(args.csv):
            df = pd.read_csv(args.csv)
            prompt = df.iloc[args.csv_idx]["text"]
        else:
            prompt = (
                "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): "
                "(Siemens, MAGNETOM_Vida, 3.0), Acquisition (Description, Sequence, Variant): "
                "(t1_tse_tra, SE, SK_SP), Imaging Parameters (Echo Time, Repetition Time, "
                "Inversion Time, Flip Angle): (0.009, 0.55, NONE, 90.0)"
            )

    print(f"[*] Synthesizing 3D MRI volume for prompt:\n    {prompt}\n")
    synthesize_from_prompt(
        metadata_text=prompt,
        mrclip_ckpt=args.mrclip_ckpt,
        clip_checkpoint=args.clip_checkpoint,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        output_size=tuple(args.dim),
        spacing=tuple(args.spacing),
        num_inference_steps=args.num_steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
