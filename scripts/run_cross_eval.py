"""Cross-dataset evaluation: Chalmers-trained LoRA -> TU Delft generalization.

Loads a base+ LoRA (or Conv-LoRA) checkpoint trained on the Chalmers
per-yarn dataset and reports three metrics on each of the configured
evaluation volumes (Chalmers test_00 and any TU Delft NPZ):

  1. Per-component slicewise Dice - one prompt per connected fibre
     component, matching the training-time prompt distribution. This is the
     apples-to-apples number the PEFT ladder cares about.
  2. Volumetric Dice via SAM2 memory propagation - one mid-slice bbox
     prompt around the largest fibre component, propagated forwards and
     backwards through the whole Z stack. This exercises the real MedSAM2
     volumetric inference path fixed in Tier B 2A.
  3. Fibre-continuity ratio (pred mean Z-extent / GT mean Z-extent) on the
     volumetric prediction.

The Chalmers dataset has 4-class labels (0=matrix, 1=warp, 2=weft,
3=binder); TU Delft has binary fibre/matrix. For comparability we collapse
both to a binary fibre foreground in this script without touching the
underlying .npz files.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from cfrp_medsam2.data import (
    NpzVolume,
    SliceDataset,
    SliceDatasetConfig,
    percentile_normalize,
)
from cfrp_medsam2.eval import dice_2d, dice_3d, fibre_continuity
from cfrp_medsam2.lora import LoRAConfig, inject_lora
from cfrp_medsam2.model import ModelConfig, SegModel


BACKBONES: dict[str, tuple[str, str]] = {
    "tiny": ("configs/sam2.1_hiera_t512.yaml", "checkpoints/sam2.1_hiera_tiny.pt"),
    "tiny_medsam2": (
        "configs/sam2.1_hiera_t512.yaml",
        "checkpoints/MedSAM2_latest.pt",
    ),
    "base_plus": (
        "configs/sam2.1_hiera_b+_512.yaml",
        "checkpoints/sam2.1_hiera_base_plus.pt",
    ),
}


# ---------------------------------------------------------------------------
# Dataset specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSpec:
    name: str
    npz: Path
    # target_classes = which raw label ids count as fibre foreground. On
    # Chalmers this is (1, 2, 3) (warp/weft/binder); on TU Delft (binary
    # fibre) it's (1,).
    target_classes: tuple[int, ...]


# ---------------------------------------------------------------------------
# Checkpoint loading (LoRA / Conv-LoRA state dicts carry the adapter weights
# plus the unfrozen mask decoder; the rest of the backbone comes from the
# SAM2.1 base checkpoint via `build_sam2`).
# ---------------------------------------------------------------------------


def load_lora_model(
    backbone: str,
    checkpoint: Path,
    device: str = "cuda",
) -> SegModel:
    sam2_config, base_ckpt = BACKBONES[backbone]
    model = SegModel(
        ModelConfig(
            backend="medsam2",
            sam2_config=sam2_config,
            checkpoint=base_ckpt,
            image_size=512,
            device=device,
        )
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_cfg = payload.get("cfg", {})
    lora_cfg_dict = saved_cfg.get("lora") or {}
    # Drop fields that aren't part of LoRAConfig (older ckpts may store extras).
    lora_cfg = LoRAConfig(**{k: v for k, v in lora_cfg_dict.items() if k in LoRAConfig.__dataclass_fields__})
    inject_lora(model, lora_cfg)
    sd = payload["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The base checkpoint already contributed most keys; only the adapter /
    # decoder keys should come from `sd`. Print a short diagnostic.
    lora_loaded = sum(1 for k in sd if "lora_" in k)
    dec_loaded = sum(1 for k in sd if "mask_decoder" in k)
    print(
        f"[cross_eval] loaded {len(sd)} tensors from {checkpoint.name}  "
        f"(lora={lora_loaded}, mask_decoder={dec_loaded}); "
        f"unexpected={len(unexpected)}  trainable_missing={sum(1 for k in missing if 'lora_' in k)}"
    )
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Metric 1: per-component slicewise Dice
# ---------------------------------------------------------------------------


def per_component_slicewise_dice(
    model: SegModel,
    spec: EvalSpec,
    device: str,
    max_samples: int = 200,
    min_component_voxels: int = 1500,
) -> dict[str, float]:
    ds = SliceDataset(
        SliceDatasetConfig(
            volume_paths=[spec.npz],
            image_size=512,
            per_component=True,
            target_classes=spec.target_classes,
            min_component_voxels=min_component_voxels,
            prompt_mode="bbox",
        )
    )
    n = min(max_samples, len(ds))
    if n == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    stride = max(1, len(ds) // n)
    indices = list(range(0, len(ds), stride))[:n]
    dices: list[float] = []
    with torch.no_grad():
        for i in indices:
            s = ds[i]
            img = s["image"].unsqueeze(0).to(device)
            box = s["prompt"]["box"].to(device)
            logits = model.forward_slice(img, boxes=[box])
            pred = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy().astype(np.uint8)
            dices.append(dice_2d(pred, s["mask"].numpy()))
    arr = np.asarray(dices, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
    }


# ---------------------------------------------------------------------------
# Metric 2 + 3: volumetric Dice + fibre continuity via infer_volume
# ---------------------------------------------------------------------------


def _binary_fibre_volume(vol: NpzVolume, target_classes: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros_like(vol.gts, dtype=np.uint8)
    for c in target_classes:
        mask |= (vol.gts == c).astype(np.uint8)
    return mask


def _mid_slice_box(mask_vol: np.ndarray, jitter: int = 5) -> tuple[int, tuple[int, int, int, int]]:
    """Pick the Z slice with the most fibre pixels and return its tight bbox."""
    areas = mask_vol.reshape(mask_vol.shape[0], -1).sum(axis=1)
    mid = int(np.argmax(areas))
    ys, xs = np.where(mask_vol[mid] > 0)
    if ys.size == 0:
        raise RuntimeError("no fibre pixels in any slice; cannot prompt")
    x0, y0, x1, y1 = int(xs.min()) - jitter, int(ys.min()) - jitter, int(xs.max()) + jitter, int(ys.max()) + jitter
    H, W = mask_vol.shape[-2:]
    x0 = max(x0, 0); y0 = max(y0, 0); x1 = min(x1, W - 1); y1 = min(y1, H - 1)
    return mid, (x0, y0, x1, y1)


def volumetric_propagation_metrics(
    model: SegModel,
    spec: EvalSpec,
    device: str,
) -> dict[str, float]:
    vol = NpzVolume.load(spec.npz)
    binary_gt = _binary_fibre_volume(vol, spec.target_classes)
    mid, box = _mid_slice_box(binary_gt)

    Z, H, W = vol.imgs.shape
    # Build (Z, 3, H, W) 3-slice RGB windows from the uint8 reconstruction.
    imgs = vol.imgs.astype(np.float32) / 255.0
    stacked = np.stack([imgs] * 3, axis=1)  # (Z, 3, H, W)
    vol_t = torch.from_numpy(stacked).to(device)
    box_t = torch.tensor(box, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model.infer_volume(vol_t, box_t)
    pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)

    dice = dice_3d(pred, binary_gt)
    cont = fibre_continuity(pred, binary_gt)
    return {
        "mid_slice": mid,
        "prompt_box_xyxy": list(box),
        "dice_3d": float(dice),
        "fibre_continuity": cont.continuity_ratio,
        "pred_mean_z_extent": cont.pred_mean_length,
        "gt_mean_z_extent": cont.gt_mean_length,
        "pred_fg_voxels": int(pred.sum()),
        "gt_fg_voxels": int(binary_gt.sum()),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_specs(repo: Path) -> list[EvalSpec]:
    specs: list[EvalSpec] = []
    chalmers_test = repo / "data" / "processed" / "cfrp_real" / "test_00.npz"
    if chalmers_test.exists():
        specs.append(EvalSpec("chalmers_test", chalmers_test, target_classes=(1, 2, 3)))
    tudelft_dir = repo / "data" / "processed" / "tudelft"
    if tudelft_dir.is_dir():
        for npz in sorted(tudelft_dir.glob("*.npz")):
            specs.append(EvalSpec(f"tudelft/{npz.stem}", npz, target_classes=(1,)))
    return specs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path, help="Path to a LoRA / Conv-LoRA _best.pt")
    p.add_argument("--backbone", choices=list(BACKBONES.keys()), default="base_plus")
    p.add_argument("--max-slicewise-samples", type=int, default=200)
    p.add_argument(
        "--min-component-voxels",
        type=int,
        default=1500,
        help="Ignore connected components smaller than this when doing per-yarn Dice.",
    )
    p.add_argument(
        "--skip-volumetric",
        action="store_true",
        help="Skip the SAM2 memory-propagation volumetric metric (fast mode).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to logs/cross_eval_<ckpt_stem>.json.",
    )
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_lora_model(args.backbone, args.checkpoint, device=device)
    specs = build_specs(repo)
    if not specs:
        print("[cross_eval] no eval volumes found under data/processed/; nothing to do.")
        return 0

    print(f"[cross_eval] backbone={args.backbone} ckpt={args.checkpoint.name} device={device}")
    print(f"[cross_eval] evaluating on: {', '.join(s.name for s in specs)}")

    results: dict[str, dict] = {
        "checkpoint": str(args.checkpoint),
        "backbone": args.backbone,
        "per_eval": {},
    }

    for spec in specs:
        print(f"\n=== {spec.name}  ({spec.npz.name}, target_classes={spec.target_classes}) ===")
        entry: dict = {}
        sw = per_component_slicewise_dice(
            model,
            spec,
            device=device,
            max_samples=args.max_slicewise_samples,
            min_component_voxels=args.min_component_voxels,
        )
        print(
            f"  slicewise per-component  n={sw['n']}  "
            f"mean={sw['mean']:.4f}  median={sw['median']:.4f}  std={sw['std']:.4f}"
        )
        entry["slicewise_per_component"] = sw

        if not args.skip_volumetric:
            try:
                vol_m = volumetric_propagation_metrics(model, spec, device=device)
                print(
                    f"  volumetric (SAM2 propagation)  dice_3d={vol_m['dice_3d']:.4f}  "
                    f"fibre_continuity={vol_m['fibre_continuity']:.3f}  "
                    f"pred_z_extent={vol_m['pred_mean_z_extent']:.1f}  "
                    f"gt_z_extent={vol_m['gt_mean_z_extent']:.1f}"
                )
                entry["volumetric"] = vol_m
            except Exception as e:
                print(f"  volumetric FAILED: {e.__class__.__name__}: {e}")
                entry["volumetric_error"] = f"{e.__class__.__name__}: {e}"
        results["per_eval"][spec.name] = entry

    out_path = args.out or (repo / "logs" / f"cross_eval_{args.checkpoint.stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[cross_eval] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
