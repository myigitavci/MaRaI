"""
MR-SYNTH: Sampling Orchestrator and Generation Pipeline
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import numpy as np
import torch
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism
from tqdm import tqdm

from .model import MRCLIPConditionedUNet
from .utils import define_instance
from .utils_infer import ReconModel, initialize_noise_latents


@torch.no_grad()
def generate_sample(
    args: Any,
    device: torch.device,
    autoencoder: torch.nn.Module,
    model: MRCLIPConditionedUNet,
    scale_factor: float | torch.Tensor,
    clip_embedding: torch.Tensor,
    output_size: tuple[int, int, int] = (256, 256, 128),
    spacing: tuple[float, float, float] = (0.94, 0.94, 1.36),
    cfg_scale: float = 2.0,
    seed: int = 1234,
    num_inference_steps: int = 30,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Generate a 3D MRI volume given an MR-CLIP text embedding.

    Args:
        args: Parsed configuration namespace.
        device: Target torch device (e.g. cuda:0).
        autoencoder: Pretrained VAE decoder.
        model: MRCLIPConditionedUNet.
        scale_factor: Latent scaling factor.
        clip_embedding: (1, 512) normalized MR-CLIP text feature.
        output_size: (D, H, W) spatial dimensions.
        spacing: Physical voxel spacing in mm.
        cfg_scale: Classifier-free guidance scale.
        seed: Random generator seed.
        num_inference_steps: Rectified flow steps (default 30).

    Returns:
        vol_np: (D, H, W) numpy array in [0, 1000].
        vol_pt: (1, 1, D, H, W) float tensor.
    """
    set_determinism(seed=seed)
    divisor = 2 ** (len(args.diffusion_unet_def["num_channels"]) - 2)
    latent_shape = (
        1,
        args.latent_channels,
        output_size[0] // divisor,
        output_size[1] // divisor,
        output_size[2] // divisor,
    )

    spacing_raw = np.array(spacing).astype(float) * 1e2
    spacing_tensor = torch.from_numpy(spacing_raw[None, :]).half().to(device)

    noise = torch.randn(latent_shape, device=device)
    image = noise.clone()

    noise_scheduler = define_instance(args, "noise_scheduler")
    if isinstance(noise_scheduler, RFlowScheduler):
        noise_scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
        )
    else:
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    autoencoder.eval()
    model.eval()

    null_emb = model.get_null_embedding(1, device).to(torch.float16 if torch.cuda.is_available() else torch.float32)
    cond_emb = clip_embedding.to(device)

    all_t = noise_scheduler.timesteps
    all_next_t = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype)))

    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        for t, next_t in zip(all_t, all_next_t):
            if cfg_scale == 0.0:
                model_output = model(
                    x=image,
                    timesteps=torch.tensor([t], device=device),
                    clip_embedding=null_emb,
                    spacing_tensor=spacing_tensor,
                )
            elif cfg_scale == 1.0:
                model_output = model(
                    x=image,
                    timesteps=torch.tensor([t], device=device),
                    clip_embedding=cond_emb,
                    spacing_tensor=spacing_tensor,
                )
            else:
                x2 = torch.cat([image, image], dim=0)
                t2 = torch.tensor([t, t], device=device)
                emb2 = torch.cat([cond_emb, null_emb], dim=0)
                sp2 = torch.cat([spacing_tensor, spacing_tensor], dim=0)

                out2 = model(
                    x=x2,
                    timesteps=t2,
                    clip_embedding=emb2,
                    spacing_tensor=sp2,
                )
                pred_cond, pred_uncond = out2.chunk(2, dim=0)
                model_output = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

            step_out = noise_scheduler.step(model_output, t, image, next_t)
            if isinstance(step_out, tuple):
                image = step_out[0]
            elif isinstance(step_out, dict):
                image = step_out.get("prev_sample", step_out)
            else:
                image = step_out

        # Decode latent representation
        val_output = recon_model(image.float())
        val_output = torch.clamp(val_output, 0.0, 1.0) * 1000.0

    vol_np = val_output.squeeze().cpu().numpy()
    return vol_np, val_output
