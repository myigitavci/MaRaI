"""
MR-SYNTH: Two-Phase Diffusion Model Training Pipeline
=====================================================
Fine-tunes the MAISI 3D brain-MRI diffusion UNet conditioned on MR-CLIP
continuous text embeddings.

Two-phase training paradigm:
  Phase 1: Freeze UNet backbone, only train the clip_projection MLP and null embedding.
  Phase 2: Unfreeze UNet backbone, fine-tune jointly with separate learning rates.

Usage:
  # Phase 1:
  python -m mr_synth.train \\
      --phase 1 \\
      --embeddings-pkl data/mrclip_embeddings.pkl \\
      --data-json data/dataset_mrclip.json \\
      --n-epochs 10 \\
      --lr 1e-3

  # Phase 2:
  python -m mr_synth.train \\
      --phase 2 \\
      --embeddings-pkl data/mrclip_embeddings.pkl \\
      --data-json data/dataset_mrclip.json \\
      --existing-ckpt models/mr_synth_phase1_best.pt \\
      --n-epochs 100 \\
      --lr 1e-5 \\
      --proj-lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root and open_clip to path
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
for p in [str(SRC_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import monai
import numpy as np
import torch
import torch.distributed as dist
from monai.networks.schedulers import RFlowScheduler
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from .dataset import EmbeddingStore, load_filenames_and_indices, prepare_data_mrclip
from .model import MRCLIPConditionedUNet
from .setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance


def parse_train_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MR-SYNTH Training Script")
    p.add_argument("-e", "--environment-file", default=str(PROJECT_ROOT / "configs" / "mr_synth" / "environment_maisi_diff_model_rflow-mr-brain.json"))
    p.add_argument("-c", "--config-file", default=str(PROJECT_ROOT / "configs" / "mr_synth" / "config_maisi_diff_model_rflow-mr-brain.json"))
    p.add_argument("-t", "--network-file", default=str(PROJECT_ROOT / "configs" / "mr_synth" / "config_network_rflow.json"))
    p.add_argument("--embeddings-pkl", required=True, help="Path to precomputed MR-CLIP embeddings .pkl")
    p.add_argument("--data-json", required=True, help="Dataset JSON list pointing to cached latents")
    p.add_argument("--phase", type=int, choices=[1, 2], default=1, help="Training phase (1 or 2)")
    p.add_argument("--n-epochs", type=int, default=None, help="Number of epochs to train")
    p.add_argument("--lr", type=float, default=None, help="Learning rate (backbone in phase 2, projection in phase 1)")
    p.add_argument("--proj-lr", type=float, default=1e-4, help="Learning rate for projection head in phase 2")
    p.add_argument("--batch-size", type=int, default=1, help="Per-GPU batch size")
    p.add_argument("--existing-ckpt", type=str, default=None, help="Path to resume/warmstart checkpoint")
    p.add_argument("--output-dir", type=str, default="models/mr_synth", help="Output directory for checkpoints")
    p.add_argument("--save-interval", type=int, default=5, help="Epoch interval to save checkpoints")
    return p.parse_args()


def train():
    cli_args = parse_train_args()
    rank, world_size, local_rank = initialize_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    args = load_config(cli_args.environment_file, cli_args.config_file, cli_args.network_file)

    os.makedirs(cli_args.output_dir, exist_ok=True)
    logger = setup_logging(cli_args.output_dir, f"train_phase{cli_args.phase}.log") if rank == 0 else logging.getLogger("dummy")

    logger.info(f"Starting MR-SYNTH Phase {cli_args.phase} Training on rank {rank}/{world_size}")

    # Load precomputed embedding store
    emb_store = EmbeddingStore(cli_args.embeddings_pkl)
    data_list = load_filenames_and_indices(cli_args.data_json)
    logger.info(f"Loaded {len(data_list)} training latent files")

    train_loader = prepare_data_mrclip(
        train_files=data_list,
        emb_store=emb_store,
        batch_size=cli_args.batch_size,
        num_workers=2,
        shuffle=True,
    )

    # Instantiate base UNet and MRCLIPConditionedUNet
    base_unet = define_instance(args, "diffusion_unet_def").to(device)
    model = MRCLIPConditionedUNet(unet=base_unet).to(device)

    # Load initial weights (MAISI pretrained weights or Phase 1 checkpoint)
    ckpt_path = cli_args.existing_ckpt or getattr(args, "existing_ckpt_filepath", None)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("unet_state_dict", ckpt))
        cleaned = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        if any(k.startswith("clip_projection.") or k.startswith("unet.") for k in cleaned):
            model.load_state_dict(cleaned, strict=False)
        else:
            model.unet.load_state_dict(cleaned, strict=False)
        logger.info(f"Loaded warmstart checkpoint from {ckpt_path}")

    # Set phase freeze / unfreeze
    if cli_args.phase == 1:
        model.freeze_unet()
        lr = cli_args.lr if cli_args.lr is not None else 1e-3
        optimizer = torch.optim.AdamW(
            list(model.clip_projection.parameters()) + [model.null_embedding],
            lr=lr,
            weight_decay=1e-4,
        )
        n_epochs = cli_args.n_epochs if cli_args.n_epochs is not None else 10
    else:
        model.unfreeze_unet()
        unet_lr = cli_args.lr if cli_args.lr is not None else 1e-5
        proj_lr = cli_args.proj_lr
        param_groups = model.trainable_parameter_groups(unet_lr=unet_lr, proj_lr=proj_lr)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        n_epochs = cli_args.n_epochs if cli_args.n_epochs is not None else 100

    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
    noise_scheduler = define_instance(args, "noise_scheduler")

    model.train()
    best_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        step_count = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            clip_embeddings = batch["clip_embedding"].to(device)
            spacing = batch["spacing"].to(device)

            noise = torch.randn_like(images)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            optimizer.zero_grad()
            with autocast("cuda", enabled=torch.cuda.is_available()):
                model_output = model(
                    x=noisy_images,
                    timesteps=timesteps,
                    clip_embedding=clip_embeddings,
                    spacing_tensor=spacing,
                )
                target = noise_scheduler.get_velocity(images, noise, timesteps) if hasattr(noise_scheduler, "get_velocity") else noise
                loss = torch.nn.functional.mse_loss(model_output.float(), target.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            step_count += 1

        avg_loss = epoch_loss / max(step_count, 1)
        if rank == 0:
            logger.info(f"Epoch {epoch}/{n_epochs} | Loss: {avg_loss:.6f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    {"model_state_dict": model.state_dict(), "epoch": epoch, "loss": best_loss},
                    os.path.join(cli_args.output_dir, f"mr_synth_phase{cli_args.phase}_best.pt"),
                )
            if epoch % cli_args.save_interval == 0 or epoch == n_epochs:
                torch.save(
                    {"model_state_dict": model.state_dict(), "epoch": epoch, "loss": avg_loss},
                    os.path.join(cli_args.output_dir, f"mr_synth_phase{cli_args.phase}_epoch_{epoch}.pt"),
                )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
