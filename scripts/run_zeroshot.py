"""Compute zero-shot MedSAM2 baseline on the toy validation volume.

Writes logs/zero_shot_metrics.json.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from skimage.transform import resize as sk_resize

warnings.filterwarnings("ignore")

from cfrp_medsam2.data import NpzVolume, mask_to_bbox
from cfrp_medsam2.eval import summarize
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
    vol = NpzVolume.load(repo / "data" / "processed" / "toy" / "val_00.npz")
    Z = vol.imgs.shape[0]
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
    preds = np.zeros_like(gts_s)
    with torch.no_grad():
        for z in range(Z):
            zm = max(0, z - 1)
            zp = min(Z - 1, z + 1)
            rgb = np.stack([imgs_s[zm], imgs_s[z], imgs_s[zp]]).astype(np.float32) / 255.0
            img = torch.from_numpy(rgb).unsqueeze(0).to(device)
            bb = mask_to_bbox(gts_s[z])
            if bb is None:
                continue
            box = torch.tensor(bb, dtype=torch.float32, device=device)
            logits = model.forward_slice(img, boxes=[box])
            preds[z] = (torch.sigmoid(logits[0]).cpu().numpy() > 0.5).astype(np.uint8)
    metrics = summarize(preds, gts_s)
    out = repo / "logs" / "zero_shot_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print("zero-shot metrics:", json.dumps(metrics, indent=2))
    np.save(repo / "logs" / "zero_shot_preds.npy", preds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
