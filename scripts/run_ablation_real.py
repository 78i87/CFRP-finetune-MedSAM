"""Ablation on real CFRP, per-yarn supervised task.

For each regime (zero_shot / lora / conv_lora / full_ft), evaluate the trained
checkpoint against a fixed sample of yarn components in the validation volume,
using the same SliceDataset the model was trained with (per_component=True).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from cfrp_medsam2.data import SliceDataset, SliceDatasetConfig
from cfrp_medsam2.eval import dice_2d
from cfrp_medsam2.lora import LoRAConfig, inject_lora, trainable_param_summary
from cfrp_medsam2.model import ModelConfig, SegModel


REPO = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = REPO / "checkpoints" / "sam2.1_hiera_tiny.pt"
BASE_TARGETS = ("qkv", "q_proj", "k_proj", "v_proj", "out_proj", "proj")
EXCL = ("mask_decoder.iou_prediction_head", "mlp", "obj_ptr")


def build_model(regime: str, ckpt_name: str | None) -> SegModel:
    model = SegModel(
        ModelConfig(backend="medsam2", checkpoint=str(CKPT), image_size=512, device=DEVICE)
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
            print(f"[ablation_real] loaded {p.name} @ epoch {payload.get('epoch')}")
    model.eval()
    return model


def evaluate_samples(
    model: SegModel, ds: SliceDataset, indices: list[int]
) -> tuple[list[float], list[dict]]:
    dices = []
    examples = []
    with torch.no_grad():
        for i in indices:
            s = ds[i]
            img = s["image"].unsqueeze(0).to(DEVICE)
            box = s["prompt"]["box"].to(DEVICE)
            logits = model.forward_slice(img, boxes=[box])
            pred = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy().astype(np.uint8)
            d = dice_2d(pred, s["mask"].numpy())
            dices.append(d)
            if len(examples) < 4:
                examples.append(
                    {
                        "image": s["image"].cpu().numpy(),
                        "mask": s["mask"].cpu().numpy(),
                        "pred": pred,
                        "dice": d,
                    }
                )
    return dices, examples


def main() -> int:
    ds = SliceDataset(
        SliceDatasetConfig(
            volume_paths=[REPO / "data" / "processed" / "cfrp_real" / "val_00.npz"],
            image_size=512,
            per_component=True,
            target_classes=(1, 2, 3),
            min_component_voxels=1500,
            prompt_mode="bbox",
        )
    )
    n = 200
    stride = max(1, len(ds) // n)
    indices = list(range(0, len(ds), stride))[:n]
    print(f"evaluating on {len(indices)} yarn components from {len(ds)} total")

    runs = [
        ("zero_shot", None),
        ("lora", "lora_real_best.pt"),
        ("conv_lora", "conv_lora_real_best.pt"),
        ("full_ft", "full_ft_real_best.pt"),
    ]
    rows = []
    all_examples = {}
    for regime, ckpt_name in runs:
        m = build_model(regime, ckpt_name)
        s = trainable_param_summary(m)
        dices, examples = evaluate_samples(m, ds, indices)
        all_examples[regime] = examples
        rows.append(
            {
                "regime": regime,
                "trainable": s["trainable"],
                "total": s["total"],
                "trainable_pct": 100.0 * s["trainable"] / s["total"],
                "lora_params": s["lora"],
                "mean_dice": float(np.mean(dices)),
                "std_dice": float(np.std(dices)),
                "median_dice": float(np.median(dices)),
                "p25": float(np.percentile(dices, 25)),
                "p75": float(np.percentile(dices, 75)),
                "n_samples": len(dices),
            }
        )
        del m
        torch.cuda.empty_cache()
        print(f"[{regime}] mean_dice={rows[-1]['mean_dice']:.4f}  std={rows[-1]['std_dice']:.4f}")

    df = pd.DataFrame(rows)
    out_csv = REPO / "logs" / "ablation_real.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("\n" + df.to_string(index=False))
    print("\nwrote", out_csv)
    with open(REPO / "logs" / "ablation_real.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Save example slices for the figure script.
    np.savez_compressed(
        REPO / "logs" / "ablation_real_examples.npz",
        **{
            f"{regime}_image_{i}": ex["image"]
            for regime, exs in all_examples.items()
            for i, ex in enumerate(exs)
        },
        **{
            f"{regime}_mask_{i}": ex["mask"]
            for regime, exs in all_examples.items()
            for i, ex in enumerate(exs)
        },
        **{
            f"{regime}_pred_{i}": ex["pred"]
            for regime, exs in all_examples.items()
            for i, ex in enumerate(exs)
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
