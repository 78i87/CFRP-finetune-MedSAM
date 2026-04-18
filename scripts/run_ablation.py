"""Produce the final ablation table across all regimes.

Loads each trained checkpoint, runs slice-wise inference on the validation
volume with GT-derived bbox prompts, and reports 3D Dice, slice-mean Dice,
and fibre-continuity. Also saves prediction overlays and a qualitative
figure so the notebook can pick them up.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.transform import resize as sk_resize

warnings.filterwarnings("ignore")

from cfrp_medsam2.data import NpzVolume, mask_to_bbox
from cfrp_medsam2.eval import summarize
from cfrp_medsam2.lora import LoRAConfig, inject_lora, trainable_param_summary
from cfrp_medsam2.model import ModelConfig, SegModel


REPO = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = REPO / "checkpoints" / "sam2.1_hiera_tiny.pt"

BASE_TARGETS = ("qkv", "q_proj", "k_proj", "v_proj", "out_proj", "proj")
EXCL = ("mask_decoder.iou_prediction_head", "mlp", "obj_ptr")


def build_model(regime: str, ckpt_name: str | None) -> SegModel:
    model = SegModel(
        ModelConfig(
            backend="medsam2", checkpoint=str(CKPT), image_size=512, device=DEVICE
        )
    )
    if regime in ("lora", "conv_lora"):
        inject_lora(
            model,
            LoRAConfig(
                rank=8,
                alpha=16.0,
                target_substrings=BASE_TARGETS,
                exclude_substrings=EXCL,
                use_conv=(regime == "conv_lora"),
                train_mask_decoder=True,
                include_memory_attention=True,
            ),
        )
    if ckpt_name:
        p = REPO / "checkpoints" / ckpt_name
        if p.exists():
            payload = torch.load(p, map_location=DEVICE, weights_only=False)
            model.load_state_dict(payload["state_dict"], strict=False)
            print(f"[ablation] loaded {p.name} @ epoch {payload.get('epoch')}")
    model.eval()
    return model


def _three_slice_rgb(imgs_s: np.ndarray, z: int) -> np.ndarray:
    """Match the SliceDataset's clip_window=3 RGB stacking (prev, cur, next)."""
    Z = imgs_s.shape[0]
    zm = max(0, z - 1)
    zp = min(Z - 1, z + 1)
    rgb = np.stack([imgs_s[zm], imgs_s[z], imgs_s[zp]]).astype(np.float32) / 255.0
    return rgb


def evaluate(model: SegModel, imgs_s: np.ndarray, gts_s: np.ndarray) -> tuple[np.ndarray, dict]:
    preds = np.zeros_like(gts_s)
    with torch.no_grad():
        for z in range(imgs_s.shape[0]):
            rgb = _three_slice_rgb(imgs_s, z)
            img = torch.from_numpy(rgb).unsqueeze(0).to(DEVICE)
            bb = mask_to_bbox(gts_s[z])
            if bb is None:
                continue
            box = torch.tensor(bb, dtype=torch.float32, device=DEVICE)
            logits = model.forward_slice(img, boxes=[box])
            preds[z] = (torch.sigmoid(logits[0]).cpu().numpy() > 0.5).astype(np.uint8)
    return preds, summarize(preds, gts_s)


def repo_data_path(repo: Path, suffix: str) -> Path:
    """Map ``suffix`` (toy/real) to a validation npz path."""
    if suffix == "real":
        return repo / "data" / "processed" / "cfrp_real" / "val_00.npz"
    return repo / "data" / "processed" / "toy" / "val_00.npz"


def main() -> int:
    import sys

    suffix = sys.argv[1] if len(sys.argv) > 1 else "toy"
    vol = NpzVolume.load(repo_data_path(REPO, suffix))
    S = 512
    imgs_s = np.stack(
        [
            sk_resize(s, (S, S), order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)
            for s in vol.imgs
        ]
    )
    gts_s = np.stack(
        [
            sk_resize(s, (S, S), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            for s in vol.gts
        ]
    )

    runs = [
        ("zero_shot", None),
        ("lora", f"lora_{suffix}_best.pt"),
        ("conv_lora", f"conv_lora_{suffix}_best.pt"),
        ("full_ft", f"full_ft_{suffix}_best.pt"),
    ]
    rows = []
    preds_dict: dict[str, np.ndarray] = {}

    for regime, ckpt_name in runs:
        m = build_model(regime, ckpt_name)
        s = trainable_param_summary(m)
        preds, metrics = evaluate(m, imgs_s, gts_s)
        preds_dict[regime] = preds
        rows.append(
            {
                "regime": regime,
                "trainable": s["trainable"],
                "total": s["total"],
                "trainable_pct": 100.0 * s["trainable"] / s["total"],
                "lora_params": s["lora"],
                **metrics,
            }
        )
        del m
        torch.cuda.empty_cache()
        print(f"[{regime}] {metrics}")

    df = pd.DataFrame(rows)
    out_csv = REPO / "logs" / f"ablation_{suffix}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("\n" + df.to_string(index=False))
    print("\nwrote", out_csv)

    np.savez_compressed(
        REPO / "logs" / f"ablation_preds_{suffix}.npz",
        imgs=imgs_s,
        gts=gts_s,
        **{k: v for k, v in preds_dict.items()},
    )
    with open(REPO / "logs" / f"ablation_{suffix}.json", "w") as f:
        json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
