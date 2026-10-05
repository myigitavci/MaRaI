# 🧠 MR-SYNTH: Metadata-Conditioned 3D MRI Synthesis

**MR-SYNTH** is a foundation diffusion model for generating high-resolution, volumetric 3D brain MRI scans conditioned directly on clinical DICOM acquisition metadata.

Built on NVIDIA's [NV-Generate-CTMR](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR) (MAISI-v2 Rectified Flow) framework and conditioned continuously via **MR-CLIP** representations, MR-SYNTH enables **physically consistent, scanner-controllable contrast synthesis**.

---

## 🌟 Key Highlights

- 🧲 **Continuous Acquisition Conditioning**: Controls pulse sequence timing parameters ($TE$, $TR$, $TI$, Flip Angle) with continuous physical decay and relaxation responses adhering to MRI Bloch equations.
- 📦 **Full 3D Volumetric Generation**: Synthesizes whole-brain 3D volumes (e.g. $256 \times 256 \times 128$ or $256 \times 256 \times 256$ NIfTI) in 30 rectified flow steps.
- 🔬 **Multi-Contrast Breadth**: Generates whole-brain and skull-stripped $T_1\text{w}$, $T_2\text{w}$, FLAIR, PD, and arbitrary parametric intermediate contrasts.
- 🎯 **Classifier-Free Guidance (CFG)**: Employs a learnable null embedding for controllable prompt fidelity and variance tuning.

---

## 🏛️ Architecture Overview

```
 DICOM Metadata String
   "A brain MRI, plane axial, Scanner: (Siemens, Vida, 3.0),
    Acquisition: (t2_tse_tra, SE, SK_SP),
    Parameters: (0.08000, 4.500, NONE, 90.0)"
                      │
                      ▼
        ┌───────────────────────────┐
        │  MR-CLIP Text Encoder     │ (ViT-B/16 text branch)
        └─────────────┬─────────────┘
                      │  512-dim Normalized Embedding
                      ▼
        ┌───────────────────────────┐
        │   MLP Projection Head     │ (512 → time_embed_dim)
        └─────────────┬─────────────┘
                      │
                      ▼
         [ Time-Embedding Hook ] ───► [ 3D Diffusion UNet (MAISI-v2 RFlow) ]
                                                       │
                                                       ▼ Latent (4, 32, 32, 16)
                                        ┌──────────────────────────────┐
                                        │ 3D AutoencoderKL (Dec Stage) │
                                        └──────────────┬───────────────┘
                                                       │
                                                       ▼
                                            3D Brain MRI Volume (NIfTI)
```

1. **Text Encoding**: The structured acquisition prompt is encoded into a 512-dimensional continuous latent vector using MaRaI's frozen `MR-CLIP` text encoder.
2. **Time-Space Injection**: The continuous embedding is projected through an MLP (`Linear → SiLU → Linear`) and injected into the UNet's time-embedding block via forward hooks.
3. **Latent Rectified Flow**: The 3D Diffusion UNet denoises a 4-channel latent space in 30 steps using rectified flow trajectory integration.
4. **Volumetric Decoding**: A 3D AutoencoderKL decodes the denoised latents into a standardized $1000$-scale NIfTI volume.

---

## ⚡ Quick Start

### Synthesize 3D MRI from Text Metadata

```bash
python -m mr_synth.infer \
    --metadata-text "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (Siemens, MAGNETOM_Vida, 3.0), Acquisition (Description, Sequence, Variant): (t2_tse_tra, SE, SK_SP), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.08000, 4.500, NONE, 90.0)" \
    --mrsynth-ckpt weights/mr_synth_phase2.pt \
    --clip-checkpoint weights/mr_clip_3d.pt \
    --cfg-scale 3.0 \
    --seed 1234 \
    --output-dir results/mr_synth
```

---

## 🏋️ Training Guide

MR-SYNTH supports two-phase training on cached VAE latents:

### Phase 1: Projection Warmup
Freezes the UNet backbone and only trains the `clip_projection` MLP and `null_embedding`:
```bash
python -m mr_synth.train \
    --phase 1 \
    --embeddings-pkl data/mrclip_embeddings.pkl \
    --data-json data/dataset_mrclip.json \
    --n-epochs 10 \
    --lr 1e-3 \
    --output-dir models/mr_synth
```

### Phase 2: Joint End-to-End Fine-Tuning
Unfreezes the entire UNet and fine-tunes jointly with differential learning rates:
```bash
python -m mr_synth.train \
    --phase 2 \
    --embeddings-pkl data/mrclip_embeddings.pkl \
    --data-json data/dataset_mrclip.json \
    --existing-ckpt models/mr_synth/mr_synth_phase1_best.pt \
    --n-epochs 100 \
    --lr 1e-5 \
    --proj-lr 1e-4 \
    --output-dir models/mr_synth
```

---

## 📚 Related Codebases & Acknowledgements

- **[NV-Generate-CTMR](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR)**: 3D Latent Diffusion Model & MAISI architecture.
- **[MR-CLIP](https://arxiv.org/abs/2507.00043)**: Acquisition-aware multimodal contrastive learning foundation model.
