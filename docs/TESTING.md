# Testing & Evaluation Guide

This document provides detailed instructions for evaluating MR-CLIP models on both **2D slices** and **3D volumes**.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Pretrained Weights](#pretrained-weights)
- [2D Evaluation (Slices)](#2d-evaluation-slices)
- [3D Evaluation (Volumes)](#3d-evaluation-volumes)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before testing, ensure you have:

1. **Environment set up** (see main [README](../README.md#installation))
2. **Test data** in CSV format (same format as training)
3. **Pretrained weights** downloaded and placed in the correct directory

---

## Pretrained Weights

### Available Models

| Model | Type | Resolution | Description | Download |
|:------|:-----|:-----------|:------------|:---------|
| MR-CLIP 2D | ViT-B/16 | 20×20 bins | 2D slice model | [⬇️ Download](https://drive.google.com/file/d/1jap3aCEPrZwvFMD8LKSBB2oTYz2HgpIG/view?usp=sharing) |
| MR-CLIP 3D | ViT-B/16-3D | 224×224×224 | 3D volume model (no skull) | [⬇️ Download](https://drive.google.com/file/d/11D6sVfHYKR-KADDd16GMt-MtmvXu9SjQ/view?usp=sharing) |

### Weight Placement

Place downloaded weights in:

```
logs/
└── <experiment_name>/
    └── checkpoints/
        └── epoch_latest.pt   # or epoch_N.pt
```

For example:
- 2D weights: `logs/mr_clip_2d/checkpoints/epoch_latest.pt`
- 3D weights: `logs/mr_clip_3d/checkpoints/epoch_latest.pt`

---

## 2D Evaluation (Slices)

### Basic Test Command

```bash
cd src

python -m open_clip_train.main \
    --val-data=/path/to/test.csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key text \
    --dataset-type=csv \
    --model=ViT-B-16 \
    --batch-size=512 \
    --workers=8 \
    --device=cuda \
    --logs=/path/to/logs \
    --name=mr_clip_2d \
    --resume=latest \
    --test
```

### Full Test Command (with all outputs)

```bash
python -m open_clip_train.main \
    --val-data=/path/to/test.csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key text \
    --dataset-type=csv \
    --model=ViT-B-16 \
    --batch-size=1000 \
    --workers=8 \
    --device=cuda \
    --logs=/path/to/logs \
    --name=mr_clip_2d \
    --resume=latest \
    --test \
    --tracepreds \
    --save-embeddings
```

### Validation Only (during training)

To run validation without full test metrics:

```bash
python -m open_clip_train.main \
    --val-data=/path/to/val.csv \
    ... \
    --resume=latest \
    --epochs=0  # Skip training, just validate
```

---

## 3D Evaluation (Volumes)

### Basic Test Command

```bash
cd src

python -m open_clip_train.main \
    --val-data=/path/to/test_3d.csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key text \
    --dataset-type=csv-3d \
    --model=ViT-B-16 \
    --vis_3d \
    --force-image-size 224 \
    --textcontextlength 98 \
    --batch-size=32 \
    --workers=8 \
    --device=cuda \
    --logs=/path/to/logs \
    --name=mr_clip_3d \
    --resume=latest \
    --test \
    --distance
```

### Full Test Command (with all outputs)

```bash
python -m open_clip_train.main \
    --val-data=/path/to/test_3d.csv \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key text \
    --dataset-type=csv-3d \
    --model=ViT-B-16 \
    --vis_3d \
    --force-image-size 224 \
    --textcontextlength 98 \
    --batch-size=32 \
    --workers=8 \
    --device=cuda \
    --logs=/path/to/logs \
    --name=mr_clip_3d \
    --resume=latest \
    --multipositiveloss \
    --distance \
    --test \
    --tracepreds \
    --save-embeddings
```

### 3D-Specific Test Options

| Parameter | Description |
|:----------|:------------|
| `--vis_3d` | **Required** for 3D models |
| `--force-image-size` | Must match training resolution (for public MR-CLIP 3D: `224`) |
| `--distance` | Enable distance-aware loss/evaluation behavior expected by released 3D setup |
| `--skull` | Use if model was trained with skull |

---

## Evaluation Metrics

### Retrieval Metrics

The test phase computes:

| Metric | Description |
|:-------|:------------|
| `R@1` | Recall at 1 (% correct in top-1) |
| `R@5` | Recall at 5 (% correct in top-5) |
| `R@10` | Recall at 10 (% correct in top-10) |
| `Mean Rank` | Average rank of correct match |
| `Median Rank` | Median rank of correct match |

Metrics are computed for both directions:
- **Image → Text**: Given an image, retrieve matching text
- **Text → Image**: Given text, retrieve matching images

### 3D Aggregation Metrics

For 3D evaluation, additional slice-aggregation metrics are computed:

| Metric | Description |
|:-------|:------------|
| `Accuracy (All Votes)` | Majority vote across all top-10 predictions |
| `Accuracy (First Label)` | Majority vote of top-1 predictions only |
| `Accuracy (Top-K Most Voted)` | Whether ground truth is in top-K voted labels |

---

## Output Files

After running `--test`, outputs are saved to `logs/<name>/checkpoints/`:

### Standard Outputs

| File | Description |
|:-----|:------------|
| `results.jsonl` | Per-epoch metrics in JSON Lines format |
| `t2i_ranks.npy` | Text-to-image rank array |
| `i2t_ranks.npy` | Image-to-text rank array |

### With `--tracepreds`

| File | Description |
|:-----|:------------|
| `vocabulary.json` | Top-10 predictions per sample |
| `test_images/` | Saved anchor and retrieved images |

### With `--save-embeddings`

| File | Description |
|:-----|:------------|
| `image_embeddings.pkl` | Image features + labels + paths |
| `text_embeddings.pkl` | Text features + labels + captions |

### 3D-Specific Outputs

| File | Description |
|:-----|:------------|
| `grouped_3d_analysis.json` | Per-volume aggregated predictions |

---

## Test Parameters Reference

### Required Parameters

| Parameter | Description |
|:----------|:------------|
| `--val-data` | Path to test CSV |
| `--model` | Model architecture |
| `--resume` | Checkpoint to load |
| `--test` | Enable test mode |

### Optional Parameters

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `--tracepreds` | Save per-sample predictions | `False` |
| `--save-embeddings` | Save feature embeddings | `False` |
| `--metrics` | Compute metrics on all samples | `False` |
| `--unique` | Compute unique-pair metrics | `False` |

### 3D-Specific Parameters

| Parameter | Description |
|:----------|:------------|
| `--vis_3d` | Enable 3D encoder |
| `--force-image-size` | Volume dimensions |
| `--skull` | Skull-preserving crop |
| `--textcontextlength` | Text context length |

---
