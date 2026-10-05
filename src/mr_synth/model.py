"""
MR-SYNTH: MR-CLIP Conditioned Diffusion UNet Architecture
==========================================================
Wraps MONAI's DiffusionModelUNetMaisi, replacing discrete integer class-label
conditioning (nn.Embedding) with a continuous 512-dim MR-CLIP text embedding
projected into the UNet's time-embedding space.

Key Features:
- Seamless drop-in on pretrained DiffusionModelUNetMaisi weights.
- 2-layer projection MLP: 512 -> time_embed_dim.
- Learnable null embedding for classifier-free guidance (CFG).
- Dynamic forward hook for injecting projected embeddings into time embeddings.
- Two-phase training helpers (freeze/unfreeze backbone & parameter groups).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)


class MRCLIPConditionedUNet(nn.Module):
    """
    Wraps DiffusionModelUNetMaisi, replacing discrete class-label conditioning
    with continuous MR-CLIP text embeddings.

    Args:
        unet: Pretrained or initialized DiffusionModelUNetMaisi instance.
        clip_embed_dim: Dimension of incoming MR-CLIP text embeddings (default 512).
        dropout_p: Dropout probability applied to the projected embedding during
                   training (enables CFG at inference).
    """

    def __init__(
        self,
        unet: DiffusionModelUNetMaisi,
        clip_embed_dim: int = 512,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.unet = unet

        # Time embedding dimension: num_channels[0] * 4 (e.g., 64 * 4 = 256)
        time_embed_dim = unet.block_out_channels[0] * 4

        # Continuous projection MLP: 512 -> time_embed_dim
        self.clip_projection = nn.Sequential(
            nn.Linear(clip_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Learnable null embedding for CFG unconditional sampling
        self.null_embedding = nn.Parameter(torch.zeros(clip_embed_dim))
        self.dropout_p = dropout_p

        # Expose key attributes from inner UNet
        self.include_top_region_index_input = getattr(unet, "include_top_region_index_input", False)
        self.include_bottom_region_index_input = getattr(unet, "include_bottom_region_index_input", False)
        self.include_spacing_input = getattr(unet, "include_spacing_input", True)

        # Signal that we handle conditioning ourselves
        self.num_class_embeds = None
        self.unet.num_class_embeds = None

    def _project_clip(self, clip_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project (B, 512) CLIP embedding -> (B, time_embed_dim).
        During training with dropout_p > 0, randomly replaces embeddings with the
        learned null embedding to support classifier-free guidance at inference.
        """
        if self.training and self.dropout_p > 0:
            mask = (
                torch.rand(clip_embedding.shape[0], device=clip_embedding.device) < self.dropout_p
            ).unsqueeze(1)
            null = self.null_embedding.to(clip_embedding.dtype).unsqueeze(0)
            clip_embedding = torch.where(mask, null.expand_as(clip_embedding), clip_embedding)
        return self.clip_projection(clip_embedding)

    def get_null_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a batch of null (unconditional) embeddings for CFG."""
        return self.null_embedding.unsqueeze(0).expand(batch_size, -1).to(device)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        clip_embedding: torch.Tensor,
        spacing_tensor: torch.Tensor | None = None,
        top_region_index_tensor: torch.Tensor | None = None,
        bottom_region_index_tensor: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        down_block_additional_residuals: tuple | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with continuous MR-CLIP embedding conditioning.
        """
        clip_proj = self._project_clip(clip_embedding.to(x.dtype))

        def _ts_emb_hook(module, input, output):
            return output + clip_proj

        hook = self.unet.time_embed.register_forward_hook(_ts_emb_hook)

        try:
            out = self.unet(
                x=x,
                timesteps=timesteps,
                context=context,
                class_labels=None,
                spacing_tensor=spacing_tensor,
                top_region_index_tensor=top_region_index_tensor,
                bottom_region_index_tensor=bottom_region_index_tensor,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
            )
        finally:
            hook.remove()

        return out

    def freeze_unet(self) -> None:
        """Phase 1: freeze UNet backbone, only train clip_projection and null_embedding."""
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.clip_projection.parameters():
            p.requires_grad_(True)
        self.null_embedding.requires_grad_(True)

    def unfreeze_unet(self) -> None:
        """Phase 2: unfreeze UNet for joint fine-tuning."""
        for p in self.unet.parameters():
            p.requires_grad_(True)

    def trainable_parameter_groups(self, unet_lr: float, proj_lr: float) -> list[dict]:
        """
        Return parameter groups for differential learning rate training:
          - clip_projection + null_embedding at proj_lr (higher LR)
          - unet backbone at unet_lr (lower LR)
        """
        proj_params = list(self.clip_projection.parameters()) + [self.null_embedding]
        unet_params = [p for p in self.unet.parameters() if p.requires_grad]
        return [
            {"params": proj_params, "lr": proj_lr, "name": "clip_projection"},
            {"params": unet_params, "lr": unet_lr, "name": "unet_backbone"},
        ]
