"""Ingestion helpers to convert raw XCT volumes to MedSAM2-style .npz."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .data import percentile_normalize, save_npz_volume


def _load_tiff_stack(path: Path) -> np.ndarray:
    import tifffile

    arr = tifffile.imread(path)  # (Z, H, W) or (H, W)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr


def ingest_tiff_stack(
    image_path: str | Path,
    label_path: str | Path,
    out_path: str | Path,
    label_lut: dict[int, int] | None = None,
    resize: int | None = None,
) -> Path:
    """Convert a paired (image TIFF stack, label TIFF stack) to the MedSAM2 ``.npz`` format.

    ``label_lut``: optional mapping from raw label id -> compact class id
    (e.g. ``{0: 0, 128: 1, 255: 2}`` to collapse three greys into 3 classes).
    """
    image_path = Path(image_path)
    label_path = Path(label_path)
    out_path = Path(out_path)

    imgs = _load_tiff_stack(image_path)
    gts = _load_tiff_stack(label_path)
    assert imgs.shape == gts.shape, f"shape mismatch: {imgs.shape} vs {gts.shape}"

    imgs = percentile_normalize(imgs)

    if label_lut is not None:
        remapped = np.zeros_like(gts, dtype=np.uint8)
        for raw, new in label_lut.items():
            remapped[gts == raw] = new
        gts = remapped
    else:
        gts = gts.astype(np.uint8)

    if resize is not None and (imgs.shape[-2] != resize or imgs.shape[-1] != resize):
        from skimage.transform import resize as sk_resize

        imgs_r = np.stack(
            [
                sk_resize(s, (resize, resize), order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)
                for s in imgs
            ]
        )
        gts_r = np.stack(
            [
                sk_resize(s, (resize, resize), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                for s in gts
            ]
        )
        imgs, gts = imgs_r, gts_r

    save_npz_volume(out_path, imgs, gts)
    return out_path


def ingest_directory(
    image_dir: str | Path,
    label_dir: str | Path,
    out_dir: str | Path,
    pattern: str = "*.tif*",
    label_lut: dict[int, int] | None = None,
    resize: int | None = None,
) -> list[Path]:
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for img in sorted(image_dir.glob(pattern)):
        lab = label_dir / img.name
        if not lab.exists():
            continue
        out = out_dir / (img.stem + ".npz")
        ingest_tiff_stack(img, lab, out, label_lut=label_lut, resize=resize)
        outputs.append(out)
    return outputs
