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

## 🚀 MR-CLIP — Base Foundation Model

<div align="center">
  <img src="docs/mr-clip-overview.png" alt="MR-CLIP Architecture" width="700"/>
</div>

<br/>

<div align="center">

  [![arXiv](https://img.shields.io/badge/arXiv-2507.00043-b31b1b.svg)](https://arxiv.org/abs/2507.00043)
  [![Weights](https://img.shields.io/badge/Download-Pretrained%20Weights-blue.svg)](https://drive.google.com/file/d/1jap3aCEPrZwvFMD8LKSBB2oTYz2HgpIG/view?usp=sharing)

</div>

**MR-CLIP** is a multimodal contrastive learning framework that aligns MR images with their DICOM acquisition metadata to learn **contrast-aware representations** — without any manual labels. It serves as the **base model** that all future extensions in this repository build upon.

**Key highlights:**
- 🧲 Learns from raw acquisition parameters (Echo Time, Repetition Time, etc.)
- 🏥 Trained on diverse multi-scanner, multi-protocol clinical data
- 🧠 Captures contrast variation across _and within_ scans
- 🔬 Anatomy-independent representation learning
- ⚡ Built on [OpenCLIP](https://github.com/mlfoundations/open_clip) with ViT-B/16 backbone

---

## ⚡ Getting Started

### Prerequisites

- Python ≥ 3.8
- CUDA-capable GPU (recommended)
- Conda or virtualenv

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

1. **NIfTI → PNG** — slice extraction, plane detection, normalization
2. **CSV creation** — image paths + simplified DICOM metadata
3. **Labeling** — unique labels from binned acquisition parameters
4. **Splitting** — train / val / test with subject-level separation

```bash
jupyter notebook preprocessing.ipynb
```

### Training & Testing

**Download pre-trained weights:** [⬇️ 20×20 Weights](https://drive.google.com/file/d/1jap3aCEPrZwvFMD8LKSBB2oTYz2HgpIG/view?usp=sharing) → place in `logs/mr_clip/checkpoints/`

```bash
cd src
python -m open_clip_train.main \
    --report-to tensorboard \
    --csv-separator=, \
    --csv-img-key filepath \
    --csv-caption-key text \
    --val-data=/path/to/your/test_data.csv \
    --batch-size=1000 \
    --workers=8 \
    --logs=/path/to/logs \
    --device=cuda \
    --dataset-type=csv \
    --model=ViT-B-16 \
    --name=mr_clip \
    --resume=latest \
    --distance \
    --test \
    --tracepreds
```

| Parameter | Description |
|:---|:---|
| `--val-data` | Path to your test data CSV |
| `--batch-size` | Adjust based on GPU memory |
| `--logs` | Directory for saving logs |
| `--name` | Experiment name (weights go under this folder) |
| `--tracepreds` | Save per-sample retrieval predictions to `mr_clip/checkpoints/` |

---

## 📦 Pretrained Weights

| Model | Resolution | Config | Download |
|:---|:---|:---|:---|
| MR-CLIP (ViT-B/16) | 20×20 bins | ET 20 · RT 20 | [⬇️ Google Drive](https://drive.google.com/file/d/1jap3aCEPrZwvFMD8LKSBB2oTYz2HgpIG/view?usp=sharing) |

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
```

---

## 📬 Contact

For questions, collaborations, or issues — please [open an issue](https://github.com/myigitavci/MaRaI/issues) in this repository.

---

## 🙏 Acknowledgements

- [OpenCLIP](https://github.com/mlfoundations/open_clip) — Foundation for the MR-CLIP codebase

---
