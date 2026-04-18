"""Preprocess the Chalmers labelled CFRP scan (Zenodo 14891845) into MedSAM2 NPZ splits.

The reconstruction is a 512^3 float32 volume; the segmentation has 4 classes
(0 = matrix/air, 1/2/3 = fibre yarns in different orientations). We collapse
to binary fibre/matrix for the headline metric but save the multi-class map
too.

Splits along the Z axis: first 320 slices train, next 96 val, last 96 test.
No leakage because each volume sees an entirely different Z-range.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from cfrp_medsam2.data import percentile_normalize, save_npz_volume


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    raw = repo / "data" / "raw" / "cfrp_labelled"
    out = repo / "data" / "processed" / "cfrp_real"
    out.mkdir(parents=True, exist_ok=True)

    img = tifffile.imread(raw / "real_layer2layer_sample_reconstruction.tiff")
    seg = tifffile.imread(raw / "real_layer2layer_sample_segmentation.tiff").astype(np.uint8)

    img_u8 = percentile_normalize(img)

    # Preserve the 4 original classes so downstream code can pick which yarn
    # direction to segment. Class 0 = matrix/air; 1/2/3 = warp/weft/binder.
    seg_u8 = seg.astype(np.uint8)

    splits = {
        "train_00": (0, 320),
        "val_00": (320, 416),
        "test_00": (416, 512),
    }

    meta = {}
    for name, (z0, z1) in splits.items():
        path = out / f"{name}.npz"
        save_npz_volume(path, img_u8[z0:z1], seg_u8[z0:z1])
        meta[name] = {
            "z_range": [z0, z1],
            "n_slices": z1 - z0,
            "class_fraction": {
                str(c): float((seg_u8[z0:z1] == c).mean()) for c in range(4)
            },
        }
        print(f"wrote {path.name}  shape={img_u8[z0:z1].shape}  classes={meta[name]['class_fraction']}")

    with open(out / "splits.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("splits:", json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
