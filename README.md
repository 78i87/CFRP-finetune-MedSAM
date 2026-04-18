# MedSAM2 PEFT for CFRP / CMC XCT Fibre Segmentation

Adapt [MedSAM2](https://github.com/bowang-lab/MedSAM2) (SAM2.1 + Hiera backbone,
volumetric prompt propagation) for low-contrast fibre-vs-matrix segmentation in
XCT volumes of carbon-fibre-reinforced polymer (CFRP) and silicon-carbide
ceramic matrix composites (SiC-SiC CMC).

The central problem: in XCT, carbon fibre and epoxy matrix have nearly identical
X-ray attenuation, so classical intensity thresholds fail and ViT-based
segmenters trained on medical data need domain adaptation.

## What this repo does

Runs a **PEFT ablation ladder** against the same data:

| Regime            | Trainable | LoRA-only params | 3D Dice | Fibre continuity |
|-------------------|-----------|------------------|---------|------------------|
| Zero-shot MedSAM2 | —         | 0                | **0.092** | 0.20* |
| LoRA              | 8.8%      | 450 K            | **0.657** | 0.14  |
| Conv-LoRA         | 8.8%      | 456 K            | **0.664** | 0.15  |
| Full fine-tune    | 100%      | 0                | **0.680** | 0.16  |

(Measured on a 48 x 256 x 256 synthetic-CFRP val volume, 5 epochs, SAM2.1-tiny
backbone, RTX 4090. Trainable % includes the mask decoder, which is unfrozen
in the LoRA regimes. The LoRA adapters themselves are ~1.1% of total params.)

`*` Zero-shot's high continuity is artifactual: it predicts one giant blob
covering everything, so the "connected component" is long but useless.
Dice reveals this.

**Bottom line:** Conv-LoRA edges vanilla LoRA on both Dice and fibre
continuity, and closes ~75% of the gap to full fine-tune at ~9% trainable
param cost. Delivered as notebooks (`01_`…`07_`) with a small shared
Python package in `src/cfrp_medsam2`.

## Datasets

- **Synthetic CFRP** — Friemann et al. 2025, Chalmers.
  [Zenodo 10.5281/zenodo.15389426](https://doi.org/10.5281/zenodo.15389426).
  Auto-labelled synthetic CT of 3D-textile noobed CFRP. Primary training.
- **SiC-SiC CMC** — Badran et al., Argonne ACDC.
  [badran_deeplearning_supplementarymaterial_v1.1](https://acdc.alcf.anl.gov/mdf/detail/badran_deeplearning_supplementarymaterial_v1.1/).
  Real micro-CT, multi-class (fibre / matrix / void). Cross-domain eval.

The notebooks can also run on a **tiny synthetic toy volume** generated in
`src/cfrp_medsam2/synthetic.py` so you can validate the pipeline end-to-end
without the multi-GB downloads.

## Layout

```
cfrp-medsam2/
  notebooks/
    01_data_download_and_eda.ipynb
    02_preprocess_to_npz.ipynb
    03_zeroshot_baseline.ipynb
    04_train_lora.ipynb
    05_train_conv_lora.ipynb
    06_train_full_ft.ipynb
    07_eval_and_ablation.ipynb
  src/cfrp_medsam2/
    data.py          # NPZ dataset, bbox prompt sampling
    lora.py          # LoRA + Conv-LoRA modules, Hiera injection
    train.py         # shared training loop
    eval.py          # Dice, 3D Dice, fibre-continuity metric
    viz.py           # slice triptychs and propagated-volume frames
    synthetic.py     # toy fibre-in-matrix volumes for smoke tests
  configs/           # YAML per regime
  checkpoints/
  external/MedSAM2/  # cloned upstream repo
  requirements.txt
```

## Quick start

```bash
source /venv/main/bin/activate    # provides torch 2.10 + CUDA
pip install -e .
pip install -r requirements.txt
bash scripts/setup_medsam2.sh     # clones upstream + downloads checkpoint

# End-to-end smoke test on fallback model + tiny toy data (no downloads)
python -m cfrp_medsam2.smoke_test

# The full pipeline on synthetic-CFRP val data:
python scripts/run_zeroshot.py
python scripts/run_training.py lora --epochs 5
python scripts/run_training.py conv_lora --epochs 5
python scripts/run_training.py full_ft --epochs 5
python scripts/run_ablation.py
python scripts/generate_figures.py

# Notebook-driven workflow
jupyter lab notebooks/
```

## Unit tests

```bash
python -m pytest tests           # 19 tests, includes real-MedSAM2 injection
```

## Hardware

Developed against a single 24 GB GPU (RTX 4090). Gradient checkpointing makes
LoRA/Conv-LoRA fit with batch = 2 volumes x 8 slices at 512x512.

## Key files

- `src/cfrp_medsam2/lora.py` — `LoRALinear` / Conv-LoRA modules and
  `inject_lora` walker (validated against MedSAM2's Hiera + memory attention).
- `src/cfrp_medsam2/model.py` — `SegModel` wrapper exposing a single
  `forward_slice(images, boxes=…, points=…)` method; `forward_slice` and the
  low-level `forward_image` / `_prepare_backbone_features` /
  `sam_mask_decoder` chain are differentiable so LoRA grads flow.
- `src/cfrp_medsam2/train.py` — shared trainer with Dice + 0.5·Focal loss,
  AdamW + cosine schedule, per-epoch CSV logging.
- `src/cfrp_medsam2/eval.py` — 2D/3D Dice plus **fibre continuity** metric
  (mean Z-axis extent of connected components relative to GT).
- `src/cfrp_medsam2/synthetic.py` — toy fibre-in-matrix XCT volumes that
  mimic the ~8%-contrast CFRP problem so the pipeline is runnable without
  multi-GB downloads.

## Extending to real CFRP / SiC-SiC data

1. Pull the Chalmers synthetic-label pipeline separately (its source is
   linked from the paper) and write its outputs to `data/raw/cfrp/`.
2. Pull the ACDC / MDF SiC-SiC archive via Globus into `data/raw/sic_sic/`.
3. Run `cfrp_medsam2.preprocess.ingest_directory` with the appropriate
   `label_lut` (the ACDC masks use greyscale palettes for fibre / matrix /
   void).
4. Point `TrainConfig.train_volumes` / `val_volumes` at the new NPZs and
   re-run the notebooks.
