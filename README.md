# MedSAM2 PEFT for CFRP / CMC XCT Fibre Segmentation

Adapt [MedSAM2](https://github.com/bowang-lab/MedSAM2) (SAM2.1 + Hiera backbone,
volumetric prompt propagation) for low-contrast fibre-vs-matrix segmentation in
XCT volumes of carbon-fibre-reinforced polymer (CFRP) and silicon-carbide
ceramic matrix composites (SiC-SiC CMC).

The central problem: in XCT, carbon fibre and epoxy matrix have nearly identical
X-ray attenuation, so classical intensity thresholds fail and ViT-based
segmenters trained on medical data need domain adaptation.

## What this repo does

Runs a **PEFT ablation ladder** against the same data. Two task
variants are reported:

### Real CFRP, per-yarn segmentation (hand-labelled, Zenodo 14891845)

A 512 x 512 x 512 layer-to-layer angle-interlock CFRP volume with 4-class
hand segmentation (matrix + 3 yarn directions). Each sample is a **single
yarn tow** with its own tight bbox prompt — matches how a user would
actually prompt MedSAM2, and is genuinely hard because the yarn/matrix
X-ray contrast is minimal.

| Regime            | Trainable | Per-yarn Dice (mean ± σ) | Median |
|-------------------|-----------|--------------------------|--------|
| Zero-shot MedSAM2 | —         | **0.38 ± 0.13**          | 0.36   |
| LoRA              | 8.8%      | **0.72 ± 0.24**          | 0.83   |
| Conv-LoRA         | 8.8%      | **0.71 ± 0.25**          | 0.83   |
| Full fine-tune    | 100%      | **0.73 ± 0.25**          | 0.85   |

### Synthetic CFRP-ish toy volume (for smoke tests)

Our in-repo synthetic generator (`cfrp_medsam2.synthetic`) packs fibres
with ~8%-contrast matrix intensity into a 48 x 256 x 256 volume. Used for
CI and for smoke-testing without the 750 MB Zenodo download.

| Regime            | Trainable | 3D Dice | Fibre continuity |
|-------------------|-----------|---------|------------------|
| Zero-shot MedSAM2 | —         | 0.092   | 0.20* |
| LoRA              | 8.8%      | 0.657   | 0.14  |
| Conv-LoRA         | 8.8%      | 0.664   | 0.15  |
| Full fine-tune    | 100%      | 0.680   | 0.16  |

`*` Zero-shot's high continuity is artifactual: it predicts one giant blob
covering everything, so the "connected component" is long but useless.
Dice reveals this.

**Bottom line:** on real hand-labelled CFRP, LoRA and Conv-LoRA reach
median 0.83 Dice on per-yarn segmentation — about 98% of the full
fine-tune ceiling — at 8.8% of the trainable parameters (LoRA itself is
~1.1%; the other 7.7% is the unfrozen mask decoder).

Conv-LoRA's edge over vanilla LoRA is small on this particular weave
(LoRA slightly wins on mean Dice, Conv-LoRA ties on median). That's
honest: the Conv-LoRA paper's claim is that local spatial priors help
**when the backbone lacks them**, but the Hiera backbone MedSAM2 uses is
already hierarchical — the marginal benefit of an extra 3x3 conv in the
LoRA down-projection is small. On the plain-ViT SAM-v1 the paper benchmarked,
the gap is larger.

Delivered as notebooks (`01_`…`07_`) with a small shared Python package in
`src/cfrp_medsam2`.

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

# The synthetic-CFRP pipeline (no downloads):
python scripts/run_zeroshot.py
python scripts/run_training.py lora --epochs 5
python scripts/run_training.py conv_lora --epochs 5
python scripts/run_training.py full_ft --epochs 5
python scripts/run_ablation.py
python scripts/generate_figures.py

# The real hand-labelled CFRP pipeline (~750 MB Zenodo download):
python -c "from cfrp_medsam2.download import download_cfrp_labelled; download_cfrp_labelled('data/raw/cfrp_labelled')"
python scripts/preprocess_real_cfrp.py
python scripts/run_zeroshot_real.py
python scripts/run_training_real.py lora --epochs 3
python scripts/run_training_real.py conv_lora --epochs 3
python scripts/run_training_real.py full_ft --epochs 3
python scripts/run_ablation_real.py
python scripts/generate_figures_real.py

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
