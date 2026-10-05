"""
MR-SYNTH: Medical Image Transforms for Preprocessing and Training
"""

from __future__ import annotations

import warnings
import torch
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandFlipd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    ResizeWithPadOrCropd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)


def define_fixed_intensity_transform(modality: str = "mri", image_keys: list[str] = ["image"]) -> list:
    """Define robust fixed intensity scaling to [0, 1]."""
    return [
        ScaleIntensityRangePercentilesd(
            keys=image_keys,
            lower=0.0,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
    ]


def define_random_intensity_transform(modality: str = "mri", image_keys: list[str] = ["image"]) -> list:
    """Define random augmentations for MRI training."""
    return [
        RandBiasFieldd(keys=image_keys, prob=0.3, coeff_range=(0.0, 0.3)),
        RandGibbsNoised(keys=image_keys, prob=0.3, alpha=(0.5, 1.0)),
        RandAdjustContrastd(keys=image_keys, prob=0.3, gamma=(0.5, 2.0)),
        RandHistogramShiftd(keys=image_keys, prob=0.05, num_control_points=10),
    ]


def get_standard_mri_transforms(target_dim: tuple[int, int, int] = (256, 256, 128)) -> Compose:
    """Standard evaluation/inference transform pipeline for 3D NIfTI."""
    return Compose([
        LoadImaged(keys=["image"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        Orientationd(keys=["image"], axcodes="RAS"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=target_dim),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        ),
    ])
