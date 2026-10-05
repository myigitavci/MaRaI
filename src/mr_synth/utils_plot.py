"""
MR-SYNTH: Visualization and Slicing Utilities
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def normalize_for_display(
    vol: np.ndarray,
    lower_pct: float = 1.0,
    upper_pct: float = 99.5,
) -> np.ndarray:
    """
    Standardize MRI intensity range to [0, 1] using robust percentile clipping.
    """
    v = vol.astype(np.float32)
    bg_thresh = np.percentile(v, 5.0)
    non_bg = v[v > bg_thresh]
    if len(non_bg) > 100:
        p_low = np.percentile(non_bg, lower_pct)
        p_high = np.percentile(non_bg, upper_pct)
    else:
        p_low = np.percentile(v, lower_pct)
        p_high = np.percentile(v, upper_pct)

    if p_high > p_low:
        v_norm = np.clip((v - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        v_norm = np.clip(v, 0.0, 1.0)
    return v_norm


def get_middle_slices(
    vol: np.ndarray,
    axial_height_ratio: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract normalized central axial, coronal, and sagittal 2D slices in [0, 1].
    Uses robust percentile-based brain bounding box and selects the axial slice
    with the maximum brain parenchyma within the mid-cerebral region to guarantee
    zero black or truncated cerebellar slices.
    """
    norm_vol = normalize_for_display(vol)
    sx, sy, sz = norm_vol.shape

    bg_thresh = max(0.08, float(np.percentile(norm_vol, 25.0)))
    brain_pts = np.where(norm_vol > bg_thresh)

    if len(brain_pts[0]) > 500:
        x_min, x_max = int(np.percentile(brain_pts[0], 2.0)), int(np.percentile(brain_pts[0], 98.0))
        y_min, y_max = int(np.percentile(brain_pts[1], 2.0)), int(np.percentile(brain_pts[1], 98.0))
        z_min, z_max = int(np.percentile(brain_pts[2], 2.0)), int(np.percentile(brain_pts[2], 98.0))

        x_mid = int(x_min + (x_max - x_min) * 0.50)
        y_mid = int(y_min + (y_max - y_min) * 0.50)

        # Select axial slice with maximum brain tissue within middle 40%-70% of brain extent
        z_low = int(z_min + (z_max - z_min) * 0.40)
        z_high = int(z_min + (z_max - z_min) * 0.70)
        if z_high > z_low:
            z_slice_vox = [np.sum(norm_vol[:, :, z] > bg_thresh) for z in range(z_low, z_high + 1)]
            z_mid = z_low + int(np.argmax(z_slice_vox))
        else:
            z_mid = int(z_min + (z_max - z_min) * axial_height_ratio)
    else:
        x_mid = sx // 2
        y_mid = sy // 2
        z_mid = int(sz * axial_height_ratio)

    x_mid = int(np.clip(x_mid, 0, sx - 1))
    y_mid = int(np.clip(y_mid, 0, sy - 1))
    z_mid = int(np.clip(z_mid, 0, sz - 1))

    axial = np.rot90(norm_vol[:, :, z_mid])
    coronal = np.rot90(norm_vol[:, y_mid, :])
    sagittal = np.rot90(norm_vol[x_mid, :, :])

    return axial, coronal, sagittal


def save_triplanar_figure(
    vol: np.ndarray,
    output_path: str,
    title: str = "MR-SYNTH Generated Volume",
) -> None:
    """Save tri-planar (Axial, Coronal, Sagittal) montage to PNG."""
    axial, coronal, sagittal = get_middle_slices(vol)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="black")
    axes[0].imshow(axial, cmap="gray")
    axes[0].set_title("Axial", color="white", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(coronal, cmap="gray")
    axes[1].set_title("Coronal", color="white", fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(sagittal, cmap="gray")
    axes[2].set_title("Sagittal", color="white", fontsize=14)
    axes[2].axis("off")

    fig.suptitle(title, color="white", fontsize=16, y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)
