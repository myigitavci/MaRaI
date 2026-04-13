# Dist-CLIP — Testing Guide

Dist-CLIP performs **MRI contrast harmonisation**: given a source MRI volume, it synthesises how that scan would look if acquired with a different contrast protocol, guided by either a reference image or a free-text description.

It builds directly on **MR-CLIP** as its perception backbone.

---

## 1. Installation

```bash
git clone https://github.com/myigitavci/MaRaI.git
cd MaRaI

conda create -n marai python=3.8 -y
conda activate marai
pip install -r requirements.txt
```

---

## 2. Required Pretrained Weights

Dist-CLIP inference needs **two separate checkpoints**:

1. **Dist-CLIP weights** (passed to `--weights`)
2. **MR-CLIP 1-channel weights for Dist-CLIP** (passed to `--clip-weights`)

| Component | Used by CLI arg | Download |
|:----------|:----------------|:---------|
| **Dist-CLIP checkpoint** | `--weights` | [⬇️ Download](https://drive.google.com/file/d/17EisOPCILGgvsmHJXLPHLQRgPsW1ffBk/view?usp=sharing) |
| **MR-CLIP (1-ch) for Dist-CLIP** | `--clip-weights` | [⬇️ Download](https://drive.google.com/file/d/1zBOagX9wUJYV5sSxKZ8M_w42lxrQPBu6/view?usp=sharing) |

> [!IMPORTANT]
> Use the **1-channel MR-CLIP checkpoint** above for Dist-CLIP. Other MR-CLIP checkpoints are not drop-in replacements.

Example:
```bash
DISTCLIP_WEIGHTS=/path/to/epoch026_model.pt
MRCLIP_FOR_DISTCLIP=/path/to/epoch_latest.pt
```

---

## 3. Input Format

All inputs are **NIfTI** volumes (`.nii` or `.nii.gz`), expected to be:
- Mapped to MNI 1mm space and Brain-extracted (skull-stripped)
- Any orientation (axial processing is applied slice-by-slice)

For **batch mode**, a CSV file is required with the following columns:

| Column | Description |
|:-------|:------------|
| `filepath` | Absolute path to the NIfTI file |
| `text` | Acquisition metadata string (e.g. `"T1w, TE 2.5ms, TR 2000ms"`) |
| `label` | Sequence label (e.g. `t1w`, `t2w`, `flair`) |
| `pair` | Integer ID grouping source/target volumes that belong to the same subject |
| `orientation` | *(optional)* slice orientation, default `axial` |
| `site` | *(optional)* scanner site identifier |

Example CSV row:
```
filepath,text,label,pair
/data/sub01_t1w.nii.gz,"T1-weighted MRI, echo time 2.5ms, repetition time 2000ms",t1w,1
/data/sub01_t2w.nii.gz,"T2-weighted MRI, echo time 90ms, repetition time 4000ms",t2w,1
```

---

## 4. Running Inference

### Single Mode — Easy Inference

The **single mode** harmonises one source volume to match a target contrast.
You can provide a **target image** (image-guided) or a **text description** (text-guided), or both.

#### Image-guided (recommended)
```bash
cd src
python -m dist_clip.test single \
    --source      /data/sub01_t1w.nii.gz \
    --target      /data/sub01_t2w.nii.gz \
    --weights     ${DISTCLIP_WEIGHTS} \
    --clip-weights ${MRCLIP_FOR_DISTCLIP} \
    --out-dir     /results/sub01/
```
Output: `/results/sub01/sub01_t1w_to_sub01_t2w_dist_clip.nii.gz`

#### Text-guided
```bash
cd src
python -m dist_clip.test single \
    --source       /data/sub01_t1w.nii.gz \
    --target-text  "T2-weighted MRI, echo time 90ms, repetition time 4000ms" \
    --weights      ${DISTCLIP_WEIGHTS} \
    --clip-weights ${MRCLIP_FOR_DISTCLIP} \
    --out-dir      /results/sub01/
```
Output: `/results/sub01/sub01_t1w_dist_clip_text.nii.gz`

#### Both at once
```bash
python -m dist_clip.test single \
    --source       /data/sub01_t1w.nii.gz \
    --target       /data/sub01_t2w.nii.gz \
    --target-text  "T2-weighted MRI, echo time 90ms, repetition time 4000ms" \
    --weights      ${DISTCLIP_WEIGHTS} \
    --clip-weights ${MRCLIP_FOR_DISTCLIP} \
    --out-dir      /results/sub01/
```
Both outputs are saved (with `_text` and `_img` variants).

---

### Batch Mode — CSV Evaluation

Runs inference on all source/target pairs in a CSV and computes SSIM, PSNR, and LPIPS metrics.

```bash
cd src
python -m dist_clip.test batch \
    --csv          /data/test_pairs.csv \
    --weights      ${DISTCLIP_WEIGHTS} \
    --clip-weights ${MRCLIP_FOR_DISTCLIP} \
    --out-dir      /results/batch_eval/
```

Outputs (written to `--out-dir/eval_outputs/`):
- `nifti_recons/` — reconstructed and target volumes as NIfTI
- `metrics_summary.csv` — per-pair SSIM / PSNR / LPIPS

---

## 5. Key Parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `--weights` | *(required)* | Path to Dist-CLIP checkpoint `.pt` or directory |
| `--clip-weights` | *(required)* | Path to MR-CLIP pretrained weights |
| `--out-dir` | *(required)* | Output directory |
| `--gpu-id` | `0` | GPU index to use |
| `--use-contrast-feat` | `enhancedv2` | Style conditioning variant |
| `--base-ch` | `16` | U-Net base channels (must match training config) |
| `--beta-dim` | `3` | Beta encoder dimension (must match training config) |
| `--text-context-length` | `98` | CLIP text context length (must match MR-CLIP config) |

---

## 6. Output Files

| File | Description |
|:-----|:------------|
| `*_dist_clip.nii.gz` | Image-guided harmonised volume |
| `*_dist_clip_text.nii.gz` | Text-guided harmonised volume |
| `eval_outputs/nifti_recons/*.nii.gz` | Batch mode reconstructions |
| `eval_outputs/metrics_summary.csv` | Per-pair evaluation metrics |

---

## 7. Troubleshooting

**CUDA out of memory** — reduce `--batch-size` (batch mode) or use a smaller volume.

**`No checkpoint found`** — pass the exact path to a `epoch*_model.pt` file instead of a directory.

**Import errors** — make sure you run from `MaRaI/src/` as the working directory and that the conda environment is activated.
