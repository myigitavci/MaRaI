"""
MR-SYNTH: Metadata-Conditioned 3D MRI Synthesis
================================================
MR-SYNTH synthesizes high-resolution 3D volumetric MRI scans conditioned
on rich DICOM acquisition metadata (pulse sequences, scanner hardware,
field strengths, and imaging parameters such as TE, TR, TI, and Flip Angle).

Built on top of NVIDIA's NV-Generate-CTMR / MAISI-v2 foundation model
and MaRaI's MR-CLIP representation space.
"""

from __future__ import annotations

from .model import MRCLIPConditionedUNet
from .setting import load_config, setup_logging
from .utils_infer import ReconModel, initialize_noise_latents

__all__ = [
    "MRCLIPConditionedUNet",
    "ReconModel",
    "initialize_noise_latents",
    "load_config",
    "setup_logging",
]
