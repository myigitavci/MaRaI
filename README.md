<div align="center">
  <img src="assets/logo.png" alt="MaRaI — Multimodal Acquisition-Aware Radiology AI" width="600"/>

  <br/><br/>

  [![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
  [![arXiv](https://img.shields.io/badge/arXiv-2507.00043-b31b1b.svg)](https://arxiv.org/abs/2507.00043)

  <br/>

  **Building acquisition-aware foundation models for radiology**

</div>

---

## 🧬 About

**MaRaI** (_Multimodal Acquisition-Aware Radiology AI_) is a research initiative focused on developing AI systems that are **aware of how medical images are acquired** — not just what they depict.

Clinical imaging data is governed by complex acquisition protocols (pulse sequences, contrast agents, scanner configurations, etc.). Most existing models treat images as raw pixel grids, ignoring the rich metadata that determines appearance and diagnostic utility. MaRaI closes this gap by building **foundation models that jointly learn from images and their acquisition context**, enabling:

- 🔍 **Contrast-aware retrieval** — find scans by _how_ they were acquired
- 🏷️ **Automated protocol tagging** — label noisy clinical datasets at scale
- 🧠 **Transfer-ready representations** — pretrained embeddings for downstream tasks
- 🔗 **Cross-modal alignment** — bridge imaging data with structured metadata

> New models will be added to this repository as they are developed. All models share a common philosophy: **acquisition metadata is a first-class signal, not an afterthought.**

---

## 🚀 Available Models

### MR-CLIP 2D — Slice-Level Foundation Model

<div align="center">
  <img src="docs/mr-clip-overview.png" alt="MR-CLIP Architecture" width="700"/>
</div>

<br/>

**MR-CLIP** is a multimodal contrastive learning framework that aligns MR images with their DICOM acquisition metadata to learn **contrast-aware representations** — without any manual labels.

**Key highlights:**
- 🧲 Learns from raw acquisition parameters (Echo Time, Repetition Time, etc.)
- 🏥 Trained on diverse multi-scanner, multi-protocol clinical data
- 🧠 Captures contrast variation across _and within_ scans
- 🔬 Anatomy-independent representation learning
- ⚡ Built on [OpenCLIP](https://github.com/mlfoundations/open_clip) with ViT-B/16 backbone

---

### MR-CLIP 3D — Volume-Level Foundation Model

<div align="center">
  <img src="docs/mr-clip-3d.png" alt="3D MR-CLIP" width="700"/>
</div>

<br/>

**MR-CLIP 3D** extends the framework to operate directly on **volumetric MR data** using a 3D Vision Transformer backbone.

**Key highlights:**
- 📦 Processes full 3D volumes (NIfTI format)
- 🧠 3D ViT-B/16-style encoder (`VisionTransformer3D`)
- 🔧 Dedicated 3D preprocessing pipeline with robust normalization
- 🎯 Multi-positive contrastive loss for protocol-level learning
- ⚡ Gradient checkpointing for memory-efficient training

---

### Dist-CLIP — MRI Contrast Harmonisation

**Dist-CLIP** synthesises how an MRI scan would look if acquired with a different contrast protocol. It uses a frozen **MR-CLIP** backbone to understand acquisition metadata, then applies a text-conditioned U-Net decoder to perform style transfer.

**Key highlights:**
- 🎨 Harmonise across T1w, T2w, FLAIR, T2\*, PDw and more
- 🖼️ **Image-guided**: provide a reference scan as the style target
- 📝 **Text-guided**: describe the target contrast in plain language
- 🔗 Built on MR-CLIP representations — no manual labels required
- 📦 Works on full NIfTI volumes (slice-by-slice processing)

---

## 📦 Pretrained Weights

| Model | Type | Input | Config | Download |
|:------|:-----|:------|:-------|:---------|
| MR-CLIP 2D | ViT-B/16 | 2D Slices | 20×20 bins (ET/RT) | [⬇️ Download](https://drive.google.com/file/d/1jap3aCEPrZwvFMD8LKSBB2oTYz2HgpIG/view?usp=drive_link) |
| MR-CLIP (1-ch) | ViT-B/16 | 2D Slices | Dist-CLIP Backbone | [⬇️ Download](#) |
| MR-CLIP 3D | ViT-B/16-3D | 3D Volumes | 20×20 bins (ET/RT), no skull | 🔜 [Coming Soon](https://drive.google.com/file/d/11D6sVfHYKR-KADDd16GMt-MtmvXu9SjQ/view?usp=sharing) |
| Dist-CLIP | U-Net + MR-CLIP | 2D Slices (NIfTI) | enhancedv2, base\_ch=16 | 🔜 [Coming Soon](#) |





---

## ⚡ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/myigitavci/MaRaI.git
cd MaRaI

# Set up environment
conda create -n marai python=3.8 -y
conda activate marai
pip install -r requirements.txt
```

### Data Preprocessing

The preprocessing pipeline converts raw NIfTI + DICOM data into training-ready CSV datasets:

```bash
jupyter notebook preprocessing.ipynb
```


### Training & Testing

<table>
<tr>
<td width="50%">

#### 📖 [Training Guide](docs/TRAINING.md)

Detailed instructions for training MR-CLIP models:
- 2D slice training
- 3D volume training
- Multi-GPU setup
- Hyperparameter reference

</td>
<td width="50%">

#### 📖 [Testing Guide](docs/TESTING.md)

Detailed instructions for evaluation:
- Running inference
- Retrieval metrics
- Saving embeddings
- Output file formats

</td>
</tr>
<tr>
<td width="50%">

#### 📖 [Dist-CLIP Testing Guide](docs/DIST_CLIP_TESTING.md)

How to run MRI harmonisation:
- Single volume inference (image- or text-guided)
- CSV batch evaluation with metrics
- Input / output file formats
- Key parameters

</td>
<td></td>
</tr>
</table>

---


## 🎯 Quick Examples

### 2D Evaluation

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
    --device=cuda \
    --logs=/path/to/logs \
    --name=mr_clip_2d \
    --resume=latest \
    --test
```

### 3D Evaluation

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
    --force-image-size 128 128 128 \
    --batch-size=32 \
    --device=cuda \
    --logs=/path/to/logs \
    --name=mr_clip_3d \
    --resume=latest \
    --test
```

> 📖 See the [Training Guide](docs/TRAINING.md) and [Testing Guide](docs/TESTING.md) for complete documentation.

### Dist-CLIP — Single Volume Harmonisation

```bash
cd src

# Image-guided: harmonise T1w → T2w style
python -m dist_clip.test single \
    --source      /data/sub01_t1w.nii.gz \
    --target      /data/sub01_t2w.nii.gz \
    --weights     /checkpoints/dist_clip/epoch100_model.pt \
    --clip-weights /checkpoints/mr_clip_2d/epoch_latest.pt \
    --out-dir     /results/

# Text-guided
python -m dist_clip.test single \
    --source       /data/sub01_t1w.nii.gz \
    --target-text  "T2-weighted MRI, echo time 90ms, repetition time 4000ms" \
    --weights      /checkpoints/dist_clip/epoch100_model.pt \
    --clip-weights /checkpoints/mr_clip_2d/epoch_latest.pt \
    --out-dir      /results/
```

> 📖 See the [Dist-CLIP Testing Guide](docs/DIST_CLIP_TESTING.md) for full documentation.

---


## 🔑 Key Parameters

| Parameter | 2D | 3D | Description |
|:----------|:--:|:--:|:------------|
| `--dataset-type` | `csv` | `csv-3d` | Dataset format |
| `--vis_3d` | ❌ | ✅ | Enable 3D vision encoder |
| `--force-image-size` | Optional | `D H W` | Input dimensions |
| `--multipositiveloss` | ✅ | ✅ | Multi-positive contrastive loss |
| `--grad-checkpointing` | Optional | Recommended | Memory-efficient training |

---

## 📄 Citation

If you use any code or models from this repository, please cite:

```bibtex
@misc{avci2025mrclipefficientmetadataguidedlearning,
      title={MR-CLIP: Efficient Metadata-Guided Learning of MRI Contrast Representations}, 
      author={Mehmet Yigit Avci and Pedro Borges and Paul Wright and Mehmet Yigitsoy and Sebastien Ourselin and Jorge Cardoso},
      year={2025},
      eprint={2507.00043},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.00043}, 
}
@misc{avci2025metadataaligned3dmrirepresentations,
      title={Metadata-Aligned 3D MRI Representations for Contrast Understanding and Quality Control}, 
      author={Mehmet Yigit Avci and Pedro Borges and Virginia Fernandez and Paul Wright and Mehmet Yigitsoy and Sebastien Ourselin and Jorge Cardoso},
      year={2025},
      eprint={2511.00681},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.00681}, 
}
@misc{avci2025distcliparbitrarymetadataimage,
      title={DIST-CLIP: Arbitrary Metadata and Image Guided MRI Harmonization via Disentangled Anatomy-Contrast Representations}, 
      author={Mehmet Yigit Avci and Pedro Borges and Virginia Fernandez and Paul Wright and Mehmet Yigitsoy and Sebastien Ourselin and Jorge Cardoso},
      year={2025},
      eprint={2512.07674},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.07674}, 
}
```

---

## 📬 Contact

For questions, collaborations, or issues — please [open an issue](https://github.com/myigitavci/MaRaI/issues) in this repository.

---

## 🙏 Acknowledgements

- [OpenCLIP](https://github.com/mlfoundations/open_clip) — Foundation for the MR-CLIP codebase

---
