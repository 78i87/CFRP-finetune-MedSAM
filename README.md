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

Tiny backbone, 3 epochs (original pipeline):

| Regime            | Trainable | Per-yarn Dice (mean ± σ) | Median |
|-------------------|-----------|--------------------------|--------|
| Zero-shot MedSAM2 | —         | **0.38 ± 0.13**          | 0.36   |
| LoRA              | 8.8%      | **0.72 ± 0.24**          | 0.83   |
| Conv-LoRA         | 8.8%      | **0.71 ± 0.25**          | 0.83   |
| Full fine-tune    | 100%      | **0.73 ± 0.25**          | 0.85   |

**Base+ backbone, 10 epochs** (Tier A upgrade — `run_training_real.py lora
--backbone base_plus --epochs 10`). Evaluated on the held-out Chalmers
`test_00.npz` split by `scripts/run_cross_eval.py`:

| Regime                            | Init                   | Trainable | Per-yarn Dice (mean ± σ) | Median |
|-----------------------------------|------------------------|-----------|--------------------------|--------|
| LoRA (base+, 10 ep)               | SAM2.1 base+           | 4.6%      | **0.786 ± 0.232**        | **0.890** |
| Conv-LoRA (base+, 10 ep)          | SAM2.1 base+           | 4.6%      | **0.787 ± 0.228**        | **0.890** |
| LoRA (tiny, 10 ep)                | MedSAM2_latest (tiny)  | 8.8%      | 0.704 ± 0.223            | 0.805  |
| Conv-LoRA (tiny, 10 ep)           | MedSAM2_latest (tiny)  | 8.8%      | 0.698 ± 0.222            | 0.786  |

Median per-yarn Dice is 0.89 on plain SAM2.1 base+, up from 0.83 on the
original tiny+3ep. Mean improves by 6.6 points. LoRA and Conv-LoRA stay
within noise of each other, consistent with the original observation
that Conv-LoRA's 3x3 conv prior adds little on top of the
already-hierarchical Hiera backbone.

**Starting LoRA from MedSAM2_latest is strictly worse than starting from
plain SAM2.1**, even on the same Chalmers data the LoRA is trained on:
LoRA falls from 0.786 → 0.704 mean (−8 pts) and 0.890 → 0.805 median
(−9 pts); Conv-LoRA falls by a similar margin. Some of this is the
smaller backbone (tiny vs base+), but the effect persists when the
earlier tiny+3ep baseline (also tiny, also started from plain SAM2.1
tiny, not MedSAM2) hit **0.72 mean / 0.83 median** in just 3 epochs -
i.e. plain-SAM2.1-tiny LoRA at 3 epochs already beat MedSAM2-init tiny
LoRA at 10 epochs on median Dice. The medical fine-tune is a worse
foundation for downstream PEFT adaptation, not just a worse zero-shot
starting point. Training trajectory: LoRA from MedSAM2 climbs from val
0.06 zero-shot to 0.74 best (epoch 6), but never catches the 0.78 that
plain-SAM2.1 base+ LoRA reaches at epoch 8.

### Zero-shot baselines across backbones and datasets

Per-component slicewise Dice, 200 samples per dataset, one tight bbox
prompt per connected fibre component. No adapters, no fine-tuning - just
the pretrained SAM2.1 / MedSAM2 backbone at 512x512:

| Backbone                                | Chalmers val | Chalmers test | TU Delft ROI 1 | TU Delft ROI 2 |
|-----------------------------------------|--------------|---------------|----------------|----------------|
| SAM2.1 tiny (`sam2.1_hiera_tiny.pt`)    | 0.38 / 0.36  | 0.38 / 0.35   | 0.38 / 0.38    | **0.50** / 0.52 |
| MedSAM2_latest (medical FT of tiny)     | **0.06** / 0.05 | 0.06 / 0.05 | 0.05 / 0.01    | 0.17 / 0.11    |
| SAM2.1 base+ (`sam2.1_hiera_base_plus.pt`) | 0.38 / 0.41 | 0.38 / 0.42 | 0.32 / 0.35    | 0.46 / 0.48    |

Three things are worth flagging:

1. **MedSAM2's medical fine-tune actively hurts CFRP**. Plain SAM2.1
   tiny gives ~0.38 Dice zero-shot; the `MedSAM2_latest.pt` weight (a
   tiny backbone fine-tuned on medical 3D imagery) collapses to ~0.06
   on the exact same task. Medical-domain priors do not transfer to
   non-medical XCT fibre segmentation. The "Zero-shot MedSAM2 0.38"
   number in earlier versions of this README was silently coming from
   `sam2.1_hiera_tiny.pt`, not from any MedSAM2 weight.

2. **Base+ zero-shot is not meaningfully stronger than tiny zero-shot
   on Chalmers** (0.38 vs 0.38 mean; median is +5pt better on base+).
   The Tier A gain (~+7 points mean Dice after LoRA on base+) is almost
   entirely from the LoRA adaptation, not from the backbone swap.

3. **On TU Delft ROI 2, zero-shot tiny (0.50) beats both the Chalmers-
   adapted LoRA (0.47) and Conv-LoRA (0.42).** The adapter actually
   degrades cross-domain performance on that ROI relative to the
   un-adapted backbone - a stronger "LoRA memorizes Chalmers" signal
   than the raw -35pt drop alone conveys.

### Cross-dataset generalization (Tier B) — trained Chalmers -> evaluated TU Delft

The Chalmers-trained base+ LoRA checkpoints evaluated on the TU Delft
thermoplastic CFRP tape dataset (Boos et al. 2025,
[4TU DOI 10.4121/3a864c60-3023-45ab-a6c6-f36a23d67f56](https://doi.org/10.4121/3a864c60-3023-45ab-a6c6-f36a23d67f56.v1),
2.0 µm voxel, ROIs 1 and 2, fibre/matrix binary). Different scanner
(Zeiss Xradia 520 Versa), different weave (unidirectional tape vs
angle-interlock 3D weave), different label protocol (Trainable Weka
Segmentation vs hand segmentation). Per-component slicewise Dice, one
prompt per connected fibre component:

| Regime                              | Chalmers test (in-domain) | TU Delft ROI 1 | TU Delft ROI 2 |
|-------------------------------------|---------------------------|----------------|----------------|
| Zero-shot SAM2.1 tiny               | 0.383 / 0.346             | 0.377 / 0.379  | **0.497 / 0.520** |
| Zero-shot SAM2.1 base+              | 0.385 / 0.424             | 0.319 / 0.347  | 0.460 / 0.482  |
| Zero-shot MedSAM2_latest (tiny)     | 0.062 / 0.053             | 0.054 / 0.007  | 0.169 / 0.113  |
| LoRA (base+, plain-SAM2.1 init)     | **0.786 / 0.890**        | **0.393 / 0.401** | 0.466 / 0.473  |
| Conv-LoRA (base+, plain-SAM2.1 init) | **0.787 / 0.890**       | 0.367 / 0.373  | 0.422 / 0.438  |
| LoRA (tiny, MedSAM2_latest init)    | 0.704 / 0.805             | 0.261 / 0.255  | 0.220 / 0.213  |
| Conv-LoRA (tiny, MedSAM2_latest init) | 0.698 / 0.786           | 0.268 / 0.279  | 0.219 / 0.212  |

**Finding:** The PEFT adapters do not transfer across scan protocols.
LoRA and Conv-LoRA both collapse from ~0.79 in-domain to ~0.4 out-of-
domain on the same task (fibre vs matrix), despite base+ giving a strong
starting point. Conv-LoRA does *not* close the gap - the extra local
spatial prior the Conv-LoRA paper claims helps here is either not
enough of a lever, or is itself over-fitting to Chalmers.

Four particularly sharp signals in the table above:

- On **TU Delft ROI 2**, the un-adapted SAM2.1 tiny backbone (0.50) beats
  the Chalmers-trained LoRA (0.47) and Conv-LoRA (0.42). The adapter is
  actively *worse* than doing nothing for that scan.
- **Conv-LoRA is strictly worse than LoRA on both TU Delft ROIs** from
  plain-SAM2.1 init, despite being tied on Chalmers. The extra 3x3 conv
  makes the adapter more Chalmers-specific, not more generalizable.
- **MedSAM2_latest-init LoRA halves cross-domain Dice** compared to
  plain-SAM2.1-init LoRA (ROI 2: 0.22 vs 0.47). Starting from a
  wrong-domain fine-tune compounds catastrophically: Chalmers adaptation
  turns the medical features into Chalmers-medical features, which
  transfer to TU Delft even worse than raw medical features did.
- **MedSAM2-init LoRA (0.22 on ROI 2) loses to zero-shot plain SAM2.1
  tiny (0.50 on ROI 2) by 28 points.** The combination of a
  wrong-domain foundation and Chalmers adaptation produces something
  strictly worse than doing nothing with the right foundation. If your
  downstream domain is not medical, plain SAM2.1 is the better base for
  PEFT, full stop.

The in-domain numbers also land essentially tied between LoRA and
Conv-LoRA, reinforcing the original README's observation that Hiera's
hierarchical attention already provides most of the inductive bias.

### Volumetric SAM2 memory propagation (Tier B 2A)

`SegModel.infer_volume` now actually drives `SAM2VideoPredictor.init_state`
+ `propagate_in_video` (previously it silently fell back to slicewise).
Evaluated on Chalmers `test_00.npz` (96 slices) with one mid-slice union
bbox covering all yarn classes:

| Regime            | 3D Dice | Fibre continuity |
|-------------------|---------|------------------|
| LoRA (base+)      | 0.364   | 0.091            |
| Conv-LoRA (base+) | 0.386   | 0.089            |

Single-prompt volumetric Dice is much lower than per-yarn slicewise
because the adapter was trained on per-component prompts (one tight bbox
per yarn tow), not the yarn-union bbox used here. The fibre-continuity
ratio of ~0.09 shows the prediction is highly fragmented along Z - the
expected failure mode of a model adapted only on single-slice
supervision without temporal consistency loss.

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
source /venv/main/bin/activate    # provides torch 2.11 + CUDA (Python 3.14)
pip install -e .
pip install -r requirements.txt
bash scripts/setup_medsam2.sh     # clones upstream + fetches tiny + base+ + MedSAM2_latest

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
python scripts/run_training_real.py lora --backbone base_plus --epochs 10 --save-every-epoch
python scripts/run_training_real.py conv_lora --backbone base_plus --epochs 10 --save-every-epoch
python scripts/run_training_real.py full_ft --epochs 3
python scripts/run_ablation_real.py
python scripts/generate_figures_real.py

# Cross-dataset generalization (Chalmers -> TU Delft, ~15 GB 4TU download):
python -c "from cfrp_medsam2.download import download_tudelft_cfrp; download_tudelft_cfrp('data/raw/tudelft', parts=('cropped', 'readme'))"
unzip -o -j data/raw/tudelft/XrayCT_Cropped_and_Registered_Reconstructed.zip \
    'Cropped_and_Registered Reconstructed/*/[1-4]_2.0um_CRR.tif' \
    'Cropped_and_Registered Reconstructed/*/[1-4]_2.0um_CRR_TWS_F.tif' \
    -d data/raw/tudelft/extracted/
python -c "from cfrp_medsam2.preprocess import ingest_tiff_stack; from pathlib import Path
for i in [1, 2]:  # ROIs 3/4 ship grayscale probability maps, not binary labels
    ingest_tiff_stack(f'data/raw/tudelft/extracted/{i}_2.0um_CRR.tif',
                      f'data/raw/tudelft/extracted/{i}_2.0um_CRR_TWS_F.tif',
                      f'data/processed/tudelft/roi{i}_2p0um.npz',
                      label_lut={0: 0, 255: 1}, resize=512)"
python scripts/run_cross_eval.py checkpoints/lora_real_base_plus_best.pt --backbone base_plus --skip-volumetric
python scripts/run_cross_eval.py checkpoints/conv_lora_real_base_plus_best.pt --backbone base_plus --skip-volumetric

# Notebook-driven workflow
jupyter lab notebooks/
```

## Unit tests

```bash
python -m pytest tests           # 22 tests, includes real-MedSAM2 injection and infer_volume contract
```

## Hardware

Developed against a single 32 GB GPU (RTX 5090, Blackwell, sm_120; earlier
iterations on a 4090). Base+ at 512x512 with batch = 1 uses ~2 GB VRAM during
LoRA training, so there's plenty of room to push batch size or move to 1024
resolution once the dataset grows. Storage-wise the pipeline is built to run
from a large scratch disk (`/workspace` in the development container) because
the TU Delft ROI subset alone is ~15 GB zipped / 46 GB extracted.

## Key files

- `src/cfrp_medsam2/lora.py` — `LoRALinear` / Conv-LoRA modules and
  `inject_lora` walker (validated against MedSAM2's Hiera + memory attention).
- `src/cfrp_medsam2/model.py` — `SegModel` wrapper exposing
  `forward_slice(images, boxes=…, points=…)` for differentiable training
  and `infer_volume(volume, mid_box)` for true SAM2 memory propagation
  (writes JPEG frames to `/workspace/tmp/`, calls `init_state` +
  `add_new_points_or_box` + forward/reverse `propagate_in_video`).
- `src/cfrp_medsam2/train.py` — shared trainer with Dice + 0.5·Focal loss,
  AdamW + cosine schedule, per-epoch CSV logging (flushed row-by-row),
  optional `save_every_epoch` checkpoints and optional TensorBoard writer.
- `src/cfrp_medsam2/eval.py` — 2D/3D Dice plus **fibre continuity** metric
  (mean Z-axis extent of connected components relative to GT).
- `src/cfrp_medsam2/synthetic.py` — toy fibre-in-matrix XCT volumes that
  mimic the ~8%-contrast CFRP problem so the pipeline is runnable without
  multi-GB downloads.
- `scripts/run_cross_eval.py` — loads a LoRA / Conv-LoRA checkpoint, runs
  per-component slicewise Dice and optional volumetric propagation Dice
  on any NPZ under `data/processed/cfrp_real/` and `data/processed/tudelft/`.
- `configs/sam2.1_hiera_b+_512.yaml` — base+ Hiera config clamped to 512x512
  input. Installed into the cloned MedSAM2's hydra search path by
  `scripts/setup_medsam2.sh`.

## Extending to real CFRP / SiC-SiC data

1. Pull the Chalmers synthetic-label pipeline separately (its source is
   linked from the paper) and write its outputs to `data/raw/cfrp/`.
2. Pull the ACDC / MDF SiC-SiC archive via Globus into `data/raw/sic_sic/`.
3. Run `cfrp_medsam2.preprocess.ingest_directory` with the appropriate
   `label_lut` (the ACDC masks use greyscale palettes for fibre / matrix /
   void).
4. Point `TrainConfig.train_volumes` / `val_volumes` at the new NPZs and
   re-run the notebooks.
