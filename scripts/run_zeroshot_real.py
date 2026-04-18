"""Zero-shot MedSAM2 on the real CFRP validation volume with per-yarn bbox prompts."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from cfrp_medsam2.data import SliceDataset, SliceDatasetConfig
from cfrp_medsam2.eval import dice_2d
from cfrp_medsam2.model import ModelConfig, SegModel


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SegModel(
        ModelConfig(
            backend="medsam2",
            checkpoint=str(repo / "checkpoints" / "sam2.1_hiera_tiny.pt"),
            image_size=512,
            device=device,
        )
    )
    model.eval()

    ds = SliceDataset(
        SliceDatasetConfig(
            volume_paths=[repo / "data" / "processed" / "cfrp_real" / "val_00.npz"],
            image_size=512,
            per_component=True,
            target_classes=(1, 2, 3),
            min_component_voxels=1500,
            prompt_mode="bbox",
        )
    )

    n = min(200, len(ds))
    stride = max(1, len(ds) // n)
    indices = list(range(0, len(ds), stride))[:n]
    dices = []
    with torch.no_grad():
        for i in indices:
            s = ds[i]
            img = s["image"].unsqueeze(0).to(device)
            box = s["prompt"]["box"].to(device)
            logits = model.forward_slice(img, boxes=[box])
            pred = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy().astype(np.uint8)
            dices.append(dice_2d(pred, s["mask"].numpy()))
    mean_d = float(np.mean(dices))
    std_d = float(np.std(dices))
    print(f"zero-shot per-yarn Dice: mean={mean_d:.4f}  std={std_d:.4f}  n={len(dices)}")
    out = repo / "logs" / "zero_shot_real_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"mean_dice": mean_d, "std_dice": std_d, "n": len(dices)}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
