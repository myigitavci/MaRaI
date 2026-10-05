"""
MR-SYNTH: Spatial and Intensity Augmentations
"""

from __future__ import annotations

from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandFlipd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandRotate90d,
    RandSpatialCropd,
)


def get_training_augmentations(keys: list[str] = ["image"], crop_size: list[int] | None = None) -> Compose:
    """Build composite spatial + intensity augmentations for diffusion training."""
    transforms = [
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandBiasFieldd(keys=keys, prob=0.3, coeff_range=(0.0, 0.3)),
        RandGibbsNoised(keys=keys, prob=0.3, alpha=(0.5, 1.0)),
        RandAdjustContrastd(keys=keys, prob=0.3, gamma=(0.7, 1.5)),
    ]
    if crop_size is not None:
        transforms.append(RandSpatialCropd(keys=keys, roi_size=crop_size, random_size=False))
    return Compose(transforms)
