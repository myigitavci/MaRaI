"""
Dist-CLIP — Inference Script
============================

Two modes:

  single   Easy one-shot inference: source image → harmonised output
  batch    CSV-based evaluation with SSIM / PSNR / LPIPS metrics

Usage examples
--------------
# Single mode — provide a target image for style reference
python -m dist_clip.test single \\
    --source   /data/sub01_t1w.nii.gz \\
    --target   /data/sub01_t2w.nii.gz \\
    --weights  /checkpoints/dist_clip/epoch100_model.pt \\
    --clip-weights /checkpoints/mr_clip_2d/epoch_latest.pt \\
    --out-dir  /results/sub01/

# Single mode — provide target contrast as free text
python -m dist_clip.test single \\
    --source      /data/sub01_t1w.nii.gz \\
    --target-text "T2-weighted MRI, echo time 90ms, repetition time 4000ms" \\
    --weights     /checkpoints/dist_clip/epoch100_model.pt \\
    --clip-weights /checkpoints/mr_clip_2d/epoch_latest.pt \\
    --out-dir     /results/sub01/

# Batch mode — CSV with source/target pairs, full metrics
python -m dist_clip.test batch \\
    --csv      /data/test_pairs.csv \\
    --weights  /checkpoints/dist_clip/ \\
    --clip-weights /checkpoints/mr_clip_2d/epoch_latest.pt \\
    --out-dir  /results/batch_eval/

See docs/DIST_CLIP_TESTING.md for the full guide.
"""

import argparse
import os
import re
import csv

import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom
from torch.utils.data import DataLoader

from dist_clip.modules.model_new import DIST_CLIP
from dist_clip.modules.dataset import (
    PairedImageDataset,
    PairedSliceDataset,
    PairedPNGSliceDataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_volume(path: str):
    """Load a NIfTI file and return (numpy_volume [H,W,S], nibabel_image)."""
    nib_img = nib.load(path)
    vol = nib_img.get_fdata().astype(np.float32)
    return vol, nib_img


def _normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Min-max normalize a 3-D volume to [0, 1]."""
    lo, hi = vol.min(), vol.max()
    if hi > lo:
        return (vol - lo) / (hi - lo)
    return np.zeros_like(vol)


def _vol_to_tensor(vol: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a 3-D numpy volume [H, W, S] to a 5-D tensor [1, S, 1, H, W]
    after min-max normalisation and resize to 224×224.
    """
    import torch.nn.functional as F

    vol = _normalize_volume(vol)
    h, w, s = vol.shape
    # [S, H, W] -> [1, S, H, W] (treat slices as batch for resize)
    t = torch.from_numpy(vol.transpose(2, 0, 1)).unsqueeze(1)  # [S, 1, H, W]
    if h != 224 or w != 224:
        t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    t = t.unsqueeze(0)  # [1, S, 1, 224, 224]
    return t.to(device)


def _tensor_to_vol(t: torch.Tensor) -> np.ndarray:
    """
    Convert model output tensor [S, 1, H, W] (clamped [0,1]) to numpy [H, W, S].
    """
    arr = t.clamp(0, 1).squeeze(1).detach().cpu().numpy()  # [S, H, W]
    return arr.transpose(1, 2, 0)  # [H, W, S]


def _resize_volume_to_shape(vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize a [H, W, S] volume to target shape using trilinear interpolation."""
    if tuple(vol.shape) == tuple(target_shape):
        return vol.astype(np.float32, copy=False)

    factors = tuple(ts / vs for ts, vs in zip(target_shape, vol.shape))
    vol_resized = zoom(vol, zoom=factors, order=1)

    # Guard against minor rounding differences from interpolation.
    if tuple(vol_resized.shape) != tuple(target_shape):
        fixed = np.zeros(target_shape, dtype=np.float32)
        mins = tuple(min(a, b) for a, b in zip(target_shape, vol_resized.shape))
        fixed[:mins[0], :mins[1], :mins[2]] = vol_resized[:mins[0], :mins[1], :mins[2]]
        vol_resized = fixed

    return vol_resized.astype(np.float32, copy=False)


def _build_model(args) -> DIST_CLIP:
    # beta_dim=1 by default: the beta encoder maps the image into a 1-channel latent
    # that the TextConditionedDecoderV2 decoder takes as input.
    mr = DIST_CLIP(
        beta_dim=args.beta_dim,
        pretrained_dist_clip=None,
        gpu_id=args.gpu_id,
        clip_model_path=args.clip_weights,
        use_contrast_feat=args.use_contrast_feat,
        use_beta=True,
        use_patchifier=False,
        base_ch=args.base_ch,
        beta_type="old",
        textcontextlength=args.text_context_length,
    )
    return mr


def _load_checkpoint(mr: DIST_CLIP, ckpt_path: str):
    """Load a single checkpoint file into the model."""
    ckpt = torch.load(ckpt_path, map_location=mr.device)
    if isinstance(ckpt, dict):
        if "decoder" in ckpt:
            missing, unexpected = mr.decoder.load_state_dict(ckpt["decoder"], strict=False)
            if missing or unexpected:
                print(f"[Dist-CLIP] Decoder: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if "beta_encoder" in ckpt and getattr(mr, "beta_encoder", None) is not None:
            missing, unexpected = mr.beta_encoder.load_state_dict(ckpt["beta_encoder"], strict=False)
            if missing or unexpected:
                print(f"[Dist-CLIP] Beta encoder: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if hasattr(mr, "enhanced_style_transfer") and "enhanced_style_transfer" in ckpt:
            mr.enhanced_style_transfer.load_state_dict(ckpt["enhanced_style_transfer"])
    return mr


def _find_latest_checkpoint(root_dir: str) -> str:
    """Walk directory tree and return path of highest-epoch checkpoint."""
    latest_path, latest_epoch = None, -1
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.startswith("epoch") and f.endswith("_model.pt"):
                try:
                    ep = int(f[len("epoch"):].split("_")[0])
                except ValueError:
                    continue
                if ep > latest_epoch:
                    latest_epoch, latest_path = ep, os.path.join(r, f)
    return latest_path


def _resolve_checkpoint(weights_arg: str) -> str:
    """Accept either a direct .pt path or a directory to search."""
    if os.path.isfile(weights_arg):
        return weights_arg
    ckpt = _find_latest_checkpoint(weights_arg)
    if ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint found at: {weights_arg}\n"
            "Pass the path to an epoch*_model.pt file or the directory containing one."
        )
    print(f"[Dist-CLIP] Using checkpoint: {ckpt}")
    return ckpt


# ---------------------------------------------------------------------------
# Single-image inference mode
# ---------------------------------------------------------------------------

def run_single(args):
    """
    Harmonise one source NIfTI to match a target contrast.

    The target contrast can be specified as:
      --target      path to a reference NIfTI (image-guided)
      --target-text free-text description (text-guided)
    Both can be provided; image-guided output is saved with suffix _img.
    """
    if args.target is None and args.target_text is None:
        raise ValueError("Provide at least one of --target or --target-text.")

    os.makedirs(args.out_dir, exist_ok=True)
    mr = _build_model(args)
    ckpt_path = _resolve_checkpoint(args.weights)
    mr = _load_checkpoint(mr, ckpt_path)
    mr.decoder.eval()

    # --- Load source ---
    src_vol, src_nib = _load_volume(args.source)
    src_t = _vol_to_tensor(src_vol, mr.device)           # [1, S, 1, 224, 224]
    b, s, c, h, w = src_t.shape
    src_flat = src_t.view(b * s, c, h, w)               # [S, 1, 224, 224]

    # --- Build mask (simple threshold) ---
    mask = (src_flat > 1e-4).float()

    src_base = os.path.splitext(os.path.basename(args.source))[0].replace(".nii", "")
    out_suffix = args.out_suffix or ""

    with torch.no_grad():

        # ── TEXT-GUIDED ─────────────────────────────────────────────────────
        if args.target_text is not None:
            # Replicate text for every slice
            src_texts = [args.source_text or ""] * (b * s)
            tgt_texts = [args.target_text] * (b * s)

            src_clip_text, tgt_clip_text = mr._encode_clip_features(
                src_texts, tgt_texts, num_slices=1
            )
            output_text, *_ = mr._process_image_chunk(
                src_flat, src_flat, src_clip_text, tgt_clip_text, mask
            )
            vol_rec = _tensor_to_vol(output_text)          # [H, W, S]
            vol_rec = _resize_volume_to_shape(vol_rec, src_vol.shape)
            out_name = f"{src_base}_dist_clip_text{out_suffix}.nii.gz"
            out_path = os.path.join(args.out_dir, out_name)
            nib.save(nib.Nifti1Image(vol_rec, src_nib.affine, src_nib.header), out_path)
            print(f"[Dist-CLIP] Saved text-guided output → {out_path}")

        # ── IMAGE-GUIDED ─────────────────────────────────────────────────────
        if args.target is not None:
            tgt_vol, _ = _load_volume(args.target)
            tgt_t = _vol_to_tensor(tgt_vol, mr.device)     # [1, S, 1, 224, 224]
            # Allow source and target to have different number of slices
            tgt_s = tgt_t.shape[1]
            tgt_flat = tgt_t.view(b if tgt_s == s else 1, tgt_s, c, h, w).view(-1, c, h, w)

            # Compute volume-level target embedding from middle slices
            mid = tgt_s // 2
            half = 5
            start_sl, end_sl = max(0, mid - half), min(tgt_s, mid + half)
            tgt_emb = mr.clip_model.encode_image(tgt_flat[:, :1])   # MR-CLIP expects 1-ch; [tgt_s, D]
            tgt_vol_emb = tgt_emb.view(1, tgt_s, -1)[:, start_sl:end_sl, :].mean(1)  # [1, D]
            tgt_emb_bc = tgt_vol_emb.unsqueeze(1).expand(-1, s, -1).reshape(b * s, -1)

            output_img, *_ = mr._process_image_chunk(
                src_flat, src_flat, None, None, mask,
                contrast_guidance_imgs=None,
                target_contrast_feat_precomputed=tgt_emb_bc,
            )
            vol_rec = _tensor_to_vol(output_img)               # [H, W, S]
            vol_rec = _resize_volume_to_shape(vol_rec, src_vol.shape)
            tgt_base = os.path.splitext(os.path.basename(args.target))[0].replace(".nii", "")
            out_name = f"{src_base}_to_{tgt_base}_dist_clip{out_suffix}.nii.gz"
            out_path = os.path.join(args.out_dir, out_name)
            nib.save(nib.Nifti1Image(vol_rec, src_nib.affine, src_nib.header), out_path)
            print(f"[Dist-CLIP] Saved image-guided output → {out_path}")

    print("[Dist-CLIP] Done.")


# ---------------------------------------------------------------------------
# CSV batch evaluation mode
# ---------------------------------------------------------------------------

def _build_loader(csv_path, dataset_type, batch_size,
                  use_percentile_clip=False, pmin=0.01, pmax=99.9):
    if dataset_type == "image":
        ds = PairedImageDataset(
            csv_path, preload=False,
            use_percentile_clip=use_percentile_clip,
            percentile_min=pmin, percentile_max=pmax,
        )
    elif dataset_type == "slice":
        ds = PairedSliceDataset(csv_path, preload=False)
    elif dataset_type == "pngslice":
        ds = PairedPNGSliceDataset(csv_path, preload=False)
    else:
        raise ValueError(f"Unknown dataset-type: {dataset_type}")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)


def _seq_from_str(s: str) -> str:
    if not isinstance(s, str):
        return "unknown"
    u = s.upper()
    if "FLAIR" in u:
        return "flair"
    if "T2*" in u or "T2STAR" in u or "SWI" in u:
        return "t2star"
    if "T2W" in u or "T2" in u:
        return "t2w"
    if "T1W" in u or "T1" in u:
        return "t1w"
    if "PD" in u:
        return "pdw"
    return "unknown"


def _run_suffix_from_path(path: str) -> str:
    try:
        m = re.search(r"run[-_]?([0-9]+)", str(path), re.IGNORECASE)
        if m:
            return f"_run{m.group(1)}"
    except Exception:
        pass
    return ""


def run_batch(args):
    """Full CSV-based evaluation with SSIM / PSNR / LPIPS metrics."""
    os.makedirs(args.out_dir, exist_ok=True)
    eval_dir = os.path.join(args.out_dir, "eval_outputs")
    nifti_dir = os.path.join(eval_dir, "nifti_recons")
    os.makedirs(nifti_dir, exist_ok=True)

    # Build pair group sizes from CSV for run-suffix logic
    group_size_map = {}
    try:
        with open(args.csv) as f:
            for row in csv.DictReader(f):
                pid = str(row.get("pair", ""))
                if pid:
                    group_size_map[pid] = group_size_map.get(pid, 0) + 1
    except Exception:
        pass

    mr = _build_model(args)
    ckpt_path = _resolve_checkpoint(args.weights)
    mr = _load_checkpoint(mr, ckpt_path)
    mr.decoder.eval()

    loader = _build_loader(
        args.csv, args.dataset_type, args.batch_size,
        args.use_percentile_clip, args.percentile_min, args.percentile_max,
    )

    records = []
    sid = 0

    with torch.no_grad():
        for batch in loader:
            src_items = batch["source"]
            tgt_items = batch["target"]

            src = src_items.img.to(mr.device)
            tgt = tgt_items.img.to(mr.device)

            # Flatten volumes to slices
            if src.dim() == 5:
                b, s, c, h, w = src.shape
                src_flat = src.view(b * s, c, h, w)
                tgt_flat = tgt.view(b * s, c, h, w)
                num_slices = s
            else:
                src_flat, tgt_flat = src, tgt
                b, num_slices = src.shape[0], 1

            mask = mr._check_mask_consistency(
                src_flat, tgt_flat, batch_id=sid, start=0, end=src_flat.shape[0] - 1
            )
            if mask is None:
                continue

            source_texts = src_items.text
            target_texts = tgt_items.text
            if isinstance(source_texts, list) and len(source_texts) == b and num_slices > 1:
                src_rep = [t for t in source_texts for _ in range(num_slices)]
                tgt_rep = [t for t in target_texts for _ in range(num_slices)]
            else:
                src_rep, tgt_rep = source_texts, target_texts

            sc, tc = mr._encode_clip_features(src_rep, tgt_rep, num_slices=1)
            output, *_ = mr._process_image_chunk(src_flat, tgt_flat, sc, tc, mask)

            # Image-guided reconstruction
            if num_slices > 1:
                tgt_emb = mr.clip_model.encode_image(tgt_flat)
                mid = num_slices // 2
                tgt_vol_e = tgt_emb.view(b, num_slices, -1)[:, max(0,mid-5):min(num_slices,mid+5), :].mean(1)
                tgt_bc = tgt_vol_e.unsqueeze(1).expand(-1, num_slices, -1).reshape(b * num_slices, -1)
                output_img, *_ = mr._process_image_chunk(
                    src_flat, tgt_flat, None, None, mask,
                    target_contrast_feat_precomputed=tgt_bc,
                )
            else:
                output_img, *_ = mr._process_image_chunk(
                    src_flat, tgt_flat, None, None, mask,
                    contrast_guidance_imgs=tgt_flat,
                )

            # Save NIfTI outputs for 3-D volumes, collect metrics
            for i in range(b):
                sid += 1
                pair_id = batch.get("pair_id", "unknown")
                if isinstance(pair_id, (list, tuple)):
                    pair_id = pair_id[i]

                src_path = src_items.filepath[i] if isinstance(src_items.filepath, (list, tuple)) else src_items.filepath
                tgt_path = tgt_items.filepath[i] if isinstance(tgt_items.filepath, (list, tuple)) else tgt_items.filepath

                src_seq = _seq_from_str(src_path)
                tgt_seq = _seq_from_str(tgt_path)
                src_run = _run_suffix_from_path(src_path) if group_size_map.get(str(pair_id), 0) > 2 else ""
                tgt_run = _run_suffix_from_path(tgt_path) if group_size_map.get(str(pair_id), 0) > 2 else ""
                src_tag, tgt_tag = f"{src_seq}{src_run}", f"{tgt_seq}{tgt_run}"

                if src.dim() == 5:
                    sl0, sl1 = i * num_slices, (i + 1) * num_slices
                    out_slices = output[sl0:sl1, 0]
                    out_slices_img = output_img[sl0:sl1, 0]
                    tgt_slices = tgt[i, :, 0]

                    vol_rec = out_slices.permute(1, 2, 0).cpu().numpy().astype(np.float32)
                    vol_rec_img = out_slices_img.permute(1, 2, 0).cpu().numpy().astype(np.float32)
                    vol_tgt = tgt_slices.permute(1, 2, 0).cpu().numpy().astype(np.float32)

                    try:
                        src_nib = nib.load(src_path)
                        tgt_nib = nib.load(tgt_path)
                        rec_name = f"sub-{pair_id}_{src_tag}_to_{tgt_tag}_recon.nii.gz"
                        rec_img_name = f"sub-{pair_id}_{src_tag}_to_{tgt_tag}_recon_img.nii.gz"
                        tgt_name = f"sub-{pair_id}_{src_tag}_to_{tgt_tag}_target.nii.gz"
                        nib.save(nib.Nifti1Image(vol_rec, src_nib.affine, src_nib.header), os.path.join(nifti_dir, rec_name))
                        nib.save(nib.Nifti1Image(vol_rec_img, src_nib.affine, src_nib.header), os.path.join(nifti_dir, rec_img_name))
                        nib.save(nib.Nifti1Image(vol_tgt, tgt_nib.affine, tgt_nib.header), os.path.join(nifti_dir, tgt_name))
                        print(f"[Dist-CLIP] Saved {rec_name}")
                    except Exception as e:
                        print(f"[Dist-CLIP] Warning: could not save NIfTI for pair {pair_id}: {e}")

                # Metrics
                if src.dim() == 5:
                    out_i, tgt_i = output[sl0:sl1], tgt_flat[sl0:sl1]
                else:
                    out_i, tgt_i = output[i:i+1], tgt_flat[i:i+1]

                mets = mr.eval_metrics(out_i, tgt_i, mask=mask[sl0:sl1] if src.dim() == 5 else mask[i:i+1])
                records.append({
                    "pair_id": pair_id,
                    "src_seq": src_seq,
                    "tgt_seq": tgt_seq,
                    **{k: round(float(v), 4) for k, v in mets.items()},
                })
                print(f"[Dist-CLIP] pair={pair_id} {src_seq}→{tgt_seq} | " +
                      " ".join(f"{k}={v}" for k, v in mets.items()))

    if records:
        import pandas as pd
        df = pd.DataFrame(records)
        summary_path = os.path.join(eval_dir, "metrics_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"\n[Dist-CLIP] Metrics saved to {summary_path}")
        numeric_cols = df.select_dtypes(include="number").columns
        print(df[numeric_cols].mean().to_string())

    print("\n[Dist-CLIP] Batch evaluation complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _shared_args(p: argparse.ArgumentParser):
    """Arguments common to both modes."""
    p.add_argument("--weights", required=True,
                   help="Path to a .pt checkpoint file, or directory containing epoch*_model.pt files.")
    p.add_argument("--clip-weights", required=True, dest="clip_weights",
                   help="Path to MR-CLIP pretrained weights (.pt).")
    p.add_argument("--out-dir", required=True, dest="out_dir",
                   help="Directory to write outputs.")
    p.add_argument("--gpu-id", type=int, default=0, dest="gpu_id",
                   help="GPU index (default: 0).")
    p.add_argument("--beta-dim", type=int, default=1, dest="beta_dim",
                   help="Beta encoder output channels (default: 1, must match training config).")
    p.add_argument("--use-contrast-feat", default="enhancedv2", dest="use_contrast_feat",
                   choices=["adain", "adapter", "enhanced", "enhancedv2", "multiscale", "enhancedv3"],
                   help="Style conditioning variant (default: enhancedv2).")
    p.add_argument("--base-ch", type=int, default=16, dest="base_ch",
                   help="U-Net base channels (default: 16).")
    p.add_argument("--text-context-length", type=int, default=98, dest="text_context_length")


def main():
    parser = argparse.ArgumentParser(
        description="Dist-CLIP: MRI Harmonization via Contrast-Aware Style Transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── single mode ──────────────────────────────────────────────────────────
    sp = sub.add_parser("single", help="Harmonise a single source volume.")
    _shared_args(sp)
    sp.add_argument("--source", required=True,
                    help="Path to source NIfTI (.nii / .nii.gz).")
    sp.add_argument("--target", default=None,
                    help="Path to reference target NIfTI for image-guided harmonisation.")
    sp.add_argument("--target-text", default=None, dest="target_text",
                    help='Target contrast description, e.g. "T2-weighted MRI, TE 90ms, TR 4000ms".')
    sp.add_argument("--source-text", default=None, dest="source_text",
                    help="Optional: source contrast description (used with --target-text).")
    sp.add_argument("--out-suffix", default="", dest="out_suffix",
                    help="Optional suffix appended to output filename (e.g. _v2).")

    # ── batch mode ───────────────────────────────────────────────────────────
    bp = sub.add_parser("batch", help="CSV-based evaluation with full metrics.")
    _shared_args(bp)
    bp.add_argument("--csv", required=True,
                    help="CSV file with columns: filepath, text, label, pair (and optionally site, orientation).")
    bp.add_argument("--dataset-type", default="image", dest="dataset_type",
                    choices=["image", "slice", "pngslice"])
    bp.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    bp.add_argument("--use-percentile-clip", action="store_true", default=False,
                    dest="use_percentile_clip")
    bp.add_argument("--percentile-min", type=float, default=0.01, dest="percentile_min")
    bp.add_argument("--percentile-max", type=float, default=99.9, dest="percentile_max")

    args = parser.parse_args()

    if args.mode == "single":
        run_single(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
