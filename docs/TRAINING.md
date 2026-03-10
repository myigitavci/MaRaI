# Training Guide

This document provides detailed instructions for training MR-CLIP models on both **2D slices** and **3D volumes**.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Data Format](#data-format)
- [2D Training (Slices)](#2d-training-slices)
- [3D Training (Volumes)](#3d-training-volumes)
- [Key Parameters Reference](#key-parameters-reference)
- [Tips & Best Practices](#tips--best-practices)

---

## Prerequisites

Before training, ensure you have:

1. **Environment set up** (see main [README](../README.md#installation))
2. **Preprocessed data** in CSV format

## Data Format

Both 2D and 3D training use CSV files with the following structure:

| Column | Description |
|:-------|:------------|
| `filepath` | Path to image (PNG for 2D) or volume (NIfTI for 3D) |
| `text` | Acquisition metadata as text (e.g., "T1w TE=10ms TR=500ms") |
| `label` | Integer label for grouping similar acquisitions |

**Example CSV:**

```csv
filepath,text,label
/data/sub001_slice_050.png,T1w TE=10 TR=500 TI=900,0
/data/sub001_slice_051.png,T1w TE=10 TR=500 TI=900,0
/data/sub002_slice_030.png,T2w TE=80 TR=4000,1
```

For 3D, each row points to a full volume:

```csv
filepath,text,label
/data/sub001_T1w.nii.gz,T1w TE=10 TR=500 TI=900,0
/data/sub002_T2w.nii.gz,T2w TE=80 TR=4000,1
```

---

## 2D Training (Slices)

### Basic Command

```bash
cd src

python -m open_clip_train.main \
    --train-data=/path/to/train.csv \
    --val-data=/path/to/val.csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key text \
    --dataset-type=csv \
    --model=ViT-B-16 \
    --batch-size=512 \
    --epochs=100 \
    --lr=1e-4 \
    --wd=0.2 \
    --warmup=2000 \
    --workers=8 \
    --device=cuda \
    --logs=/path/to/logs \
    --name=mr_clip_2d \
    --report-to tensorboard \
    --save-frequency 1 \
    --save-most-recent
```

### Full Training Command (Recommended)

For multi-GPU training with all optimizations:

```bash
python -m open_clip_train.main \
    --train-data=/path/to/train.csv \
    --val-data=/path/to/val.csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key text \
    --dataset-type=csv \
    --model=ViT-B-16 \
    --batch-size=512 \
    --epochs=100 \
    --lr=1e-4 \
    --beta1=0.9 \
    --beta2=0.98 \
    --wd=0.2 \
    --warmup=2000 \
    --workers=8 \
    --device=cuda \
    --logs=/path/to/logs \
    --name=mr_clip_2d \
    --report-to tensorboard \
    --save-frequency 1 \
    --save-most-recent \
    --delete-previous-checkpoint \
    --gather-with-grad \
    --local-loss \
    --grad-checkpointing \
    --multipositiveloss
```

### Resume Training

To resume from a checkpoint:

```bash
python -m open_clip_train.main \
    ... \
    --resume=latest
    # or specify path: --resume=/path/to/checkpoint.pt
```

---

## 3D Training (Volumes)

### Overview

3D training uses:
- **Dataset type**: `csv-3d`
- **Vision encoder**: `CLIP_3D` with 3D ViT backbone (enabled via `--vis_3d`)
- **Input**: NIfTI volumes (`.nii` or `.nii.gz`)
- **Preprocessing**: Automatic cropping, resizing, and normalization

### Basic Command

```bash
cd src

python -m open_clip_train.main \
    --train-data=/path/to/train_3d.csv \
    --val-data=/path/to/val_3d.csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key text \
    --dataset-type=csv-3d \
    --model=ViT-B-16 \
    --vis_3d \
    --force-image-size 160 192 160 \
    --batch-size=32 \
    --epochs=125 \
    --lr=1e-4 \
    --wd=0.2 \
    --warmup=2000 \
    --workers=8 \
    --device=cuda \
    --logs=/path/to/logs \
    --name=mr_clip_3d \
    --report-to tensorboard \
    --grad-checkpointing
```

### Full Training Command (Recommended)

```bash
python -m open_clip_train.main \
    --train-data=/path/to/train_3d.csv \
    --val-data=/path/to/val_3d.csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key text \
    --dataset-type=csv-3d \
    --model=ViT-B-16 \
    --vis_3d \
    --force-image-size 160 192 160 \
    --textcontextlength 108 \
    --batch-size=32 \
    --epochs=125 \
    --lr=1e-4 \
    --beta1=0.9 \
    --beta2=0.98 \
    --wd=0.2 \
    --warmup=2000 \
    --workers=32 \
    --device=cuda \
    --logs=/path/to/logs \
    --name=mr_clip_3d \
    --report-to tensorboard \
    --save-frequency 1 \
    --save-most-recent \
    --delete-previous-checkpoint \
    --gather-with-grad \
    --local-loss \
    --grad-checkpointing \
    --multipositiveloss
```

### 3D-Specific Options

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `--vis_3d` | Enable 3D vision encoder (`CLIP_3D`) | `False` |
| `--force-image-size D H W` | Volume dimensions after resize | Model default |
| `--skull` | Use crop region that preserves skull | `False` (skull-stripped crop) |
| `--textcontextlength` | Text token context length | `98` |

### Volume Preprocessing Pipeline

The 3D transform (`Volume3DTransform`) applies:

1. **Fixed crop** — removes background/skull based on registered coordinates
2. **Resize** — trilinear interpolation to target size (e.g., 160×192×160)
3. **Normalization** — robust percentile-based (0.01–99.9%)
4. **Augmentation** (training only) — random affine, flips

> **Note**: Volumes that fail normalization (e.g., corrupted data) are automatically skipped with warnings.

---

## Key Parameters Reference

### Data Parameters

| Parameter | Description |
|:----------|:------------|
| `--train-data` | Path to training CSV |
| `--val-data` | Path to validation CSV |
| `--csv-separator` | CSV delimiter (default: `\t`, use `,` for comma) |
| `--csv-img-key` | Column name for image/volume paths |
| `--csv-caption-key` | Column name for text captions |
| `--dataset-type` | `csv` for 2D, `csv-3d` for 3D |

### Model Parameters

| Parameter | Description |
|:----------|:------------|
| `--model` | Model architecture (e.g., `ViT-B-16`) |
| `--vis_3d` | Use 3D vision encoder |
| `--force-image-size` | Override image/volume size |
| `--textcontextlength` | Text context length |
| `--pretrained` | Path to pretrained weights |

### Training Parameters

| Parameter | Description |
|:----------|:------------|
| `--batch-size` | Batch size per GPU |
| `--epochs` | Number of training epochs |
| `--lr` | Learning rate |
| `--wd` | Weight decay |
| `--warmup` | Warmup steps |
| `--workers` | DataLoader workers |

### Loss & Optimization

| Parameter | Description |
|:----------|:------------|
| `--multipositiveloss` | Multi-positive contrastive loss |
| `--delta` | Weight for image loss (default: 0.5) |
| `--local-loss` | Compute loss locally (for distributed) |
| `--gather-with-grad` | Gather features with gradients |
| `--grad-checkpointing` | Enable gradient checkpointing (saves memory) |

### Checkpointing

| Parameter | Description |
|:----------|:------------|
| `--save-frequency` | Save checkpoint every N epochs |
| `--save-most-recent` | Always save latest checkpoint |
| `--delete-previous-checkpoint` | Remove old checkpoints |
| `--resume` | Resume from checkpoint (`latest` or path) |

---

## Tips & Best Practices

### Memory Management

- **3D volumes are memory-intensive**. Use:
  - `--grad-checkpointing` (essential for 3D)
  - Smaller batch sizes 
  - `--workers` tuned to your CPU cores

### Multi-GPU Training

For distributed training:

```bash
torchrun --nproc_per_node=4 -m open_clip_train.main \
    --gather-with-grad \
    --local-loss \
    ...
```

