"""
MR-SYNTH: Inference and Latent Reconstruction Helpers
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.schedulers import RFlowScheduler, DDPMScheduler


class ReconModel(nn.Module):
    """
    Wraps AutoencoderKL to decode latent representations with scale correction.
    """

    def __init__(self, autoencoder: nn.Module, scale_factor: float | torch.Tensor = 1.0) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        if isinstance(scale_factor, torch.Tensor):
            self.scale_factor = scale_factor.item() if scale_factor.numel() == 1 else scale_factor
        else:
            self.scale_factor = scale_factor

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor z -> reconstructed 3D volume."""
        recon = self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)
        return recon


def initialize_noise_latents(
    latent_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate initial random noise latents on device."""
    return torch.randn(latent_shape, device=device, dtype=dtype)
