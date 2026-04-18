"""Dataset utilities for MedSAM2-style XCT fibre segmentation.

Stores each volume as a single ``.npz`` with keys
``imgs`` (Z, H, W) uint8 and ``gts`` (Z, H, W) uint8, matching the
convention of the bowang-lab/MedSAM2 repo. The dataset yields slice
windows together with a box / point prompt simulated from the ground
truth mask for the central slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SliceWindow = dict[str, torch.Tensor | tuple]


@dataclass
class NpzVolume:
    path: Path
    imgs: np.ndarray  # (Z, H, W) uint8
    gts: np.ndarray   # (Z, H, W) uint8 (binary or class id)

    @classmethod
    def load(cls, path: str | Path) -> "NpzVolume":
        p = Path(path)
        d = np.load(p)
        imgs = d["imgs"]
        gts = d["gts"]
        assert imgs.shape == gts.shape, f"shape mismatch for {p}: {imgs.shape} vs {gts.shape}"
        return cls(path=p, imgs=imgs, gts=gts)


def percentile_normalize(vol: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """Robust min-max to uint8."""
    a, b = np.percentile(vol, [lo, hi])
    if b <= a:
        b = a + 1.0
    out = np.clip((vol - a) / (b - a), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def save_npz_volume(path: str | Path, imgs: np.ndarray, gts: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if imgs.dtype != np.uint8:
        imgs = percentile_normalize(imgs)
    gts = gts.astype(np.uint8)
    np.savez_compressed(p, imgs=imgs, gts=gts)


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) tightly around ``mask > 0`` or None if empty."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def jitter_bbox(
    bbox: tuple[int, int, int, int],
    img_shape: tuple[int, int],
    jitter: int = 5,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, int, int]:
    rng = rng or np.random.default_rng()
    H, W = img_shape
    x0, y0, x1, y1 = bbox
    dx0, dy0, dx1, dy1 = rng.integers(-jitter, jitter + 1, size=4)
    x0 = int(np.clip(x0 + dx0, 0, W - 1))
    y0 = int(np.clip(y0 + dy0, 0, H - 1))
    x1 = int(np.clip(x1 + dx1, x0 + 1, W))
    y1 = int(np.clip(y1 + dy1, y0 + 1, H))
    return x0, y0, x1, y1


def sample_point_prompts(
    mask: np.ndarray,
    n_pos: int = 2,
    n_neg: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (points[N,2] xy, labels[N] with 1=fg, 0=bg)."""
    rng = rng or np.random.default_rng()
    ys_p, xs_p = np.where(mask > 0)
    ys_n, xs_n = np.where(mask == 0)
    pos_idx = rng.choice(ys_p.size, size=min(n_pos, ys_p.size), replace=False) if ys_p.size else []
    neg_idx = rng.choice(ys_n.size, size=min(n_neg, ys_n.size), replace=False) if ys_n.size else []
    pts, lbls = [], []
    for i in pos_idx:
        pts.append([xs_p[i], ys_p[i]])
        lbls.append(1)
    for i in neg_idx:
        pts.append([xs_n[i], ys_n[i]])
        lbls.append(0)
    return np.asarray(pts, dtype=np.float32), np.asarray(lbls, dtype=np.int64)


@dataclass
class SliceDatasetConfig:
    volume_paths: Sequence[Path] = field(default_factory=list)
    image_size: int = 512
    clip_window: int = 3      # number of adjacent slices (must be odd)
    prompt_mode: str = "bbox"  # "bbox" or "point" or "mixed"
    bbox_jitter: int = 5
    positive_only: bool = True  # skip slices whose mask is empty
    binary_class_id: int | None = 1
    # Per-yarn / per-component supervision: if True, each __getitem__ returns
    # a single connected component with its own tight bbox prompt, instead of
    # the union of every component with a single giant bbox.
    per_component: bool = False
    target_classes: tuple[int, ...] | None = None
    min_component_voxels: int = 50


class SliceDataset(Dataset):
    """Yields single-slice supervision with a simulated prompt from the GT mask.

    A 3-slice window (prev, cur, next) is stacked into an RGB image so the
    SAM2 image encoder sees contextual Z-information.
    """

    def __init__(self, cfg: SliceDatasetConfig, seed: int = 0):
        assert cfg.clip_window % 2 == 1, "clip_window must be odd"
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.vols: list[NpzVolume] = [NpzVolume.load(p) for p in cfg.volume_paths]
        # Index entries are (vol_idx, slice_idx, cc_id) when per_component; cc_id=-1 means "union".
        self.index: list[tuple[int, int, int]] = []
        self._component_masks: dict[tuple[int, int, int], np.ndarray] = {}
        self._component_bboxes: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}

        if cfg.per_component:
            from scipy.ndimage import label as cc_label

            classes = cfg.target_classes if cfg.target_classes is not None else (1,)
            for vi, v in enumerate(self.vols):
                for z in range(v.imgs.shape[0]):
                    gt_slice = v.gts[z]
                    # Label CCs per-class so warp vs weft never merge.
                    cc_counter = 0
                    for cls in classes:
                        mask_cls = (gt_slice == cls).astype(np.uint8)
                        if mask_cls.sum() < cfg.min_component_voxels:
                            continue
                        labels, n = cc_label(mask_cls)
                        for k in range(1, n + 1):
                            comp = (labels == k)
                            if comp.sum() < cfg.min_component_voxels:
                                continue
                            cc_id = cc_counter
                            cc_counter += 1
                            key = (vi, z, cc_id)
                            self._component_masks[key] = comp.astype(np.uint8)
                            ys, xs = np.where(comp)
                            self._component_bboxes[key] = (
                                int(xs.min()), int(ys.min()),
                                int(xs.max()), int(ys.max()),
                            )
                            self.index.append(key)
        else:
            for vi, v in enumerate(self.vols):
                for z in range(v.imgs.shape[0]):
                    if cfg.positive_only:
                        gt = v.gts[z]
                        if cfg.binary_class_id is not None:
                            gt = (gt == cfg.binary_class_id).astype(np.uint8)
                        if gt.sum() == 0:
                            continue
                    self.index.append((vi, z, -1))

    def __len__(self) -> int:
        return len(self.index)

    def _slice_window(self, vol: NpzVolume, z: int) -> np.ndarray:
        radius = self.cfg.clip_window // 2
        Z = vol.imgs.shape[0]
        indices = np.clip(np.arange(z - radius, z + radius + 1), 0, Z - 1)
        return vol.imgs[indices]  # (clip_window, H, W)

    def _resize(self, arr: np.ndarray) -> np.ndarray:
        from skimage.transform import resize

        s = self.cfg.image_size
        if arr.ndim == 2:
            out = resize(arr, (s, s), order=0, preserve_range=True, anti_aliasing=False)
            return out.astype(arr.dtype)
        resized = np.stack(
            [resize(a, (s, s), order=1, preserve_range=True, anti_aliasing=True) for a in arr]
        )
        return resized.astype(arr.dtype)

    def __getitem__(self, idx: int) -> SliceWindow:
        vi, z, cc_id = self.index[idx]
        vol = self.vols[vi]
        window = self._slice_window(vol, z).astype(np.float32) / 255.0
        window = self._resize(window)

        if cc_id >= 0:
            gt = self._component_masks[(vi, z, cc_id)]
        else:
            gt = vol.gts[z]
            if self.cfg.binary_class_id is not None:
                gt = (gt == self.cfg.binary_class_id).astype(np.uint8)
        gt_resized = self._resize(gt)

        mode = self.cfg.prompt_mode
        if mode == "mixed":
            mode = "bbox" if self.rng.random() < 0.8 else "point"

        prompt: dict[str, torch.Tensor] = {}
        if mode == "bbox":
            bb = mask_to_bbox(gt_resized)
            if bb is None:
                bb = (0, 0, gt_resized.shape[1] - 1, gt_resized.shape[0] - 1)
            bb = jitter_bbox(bb, gt_resized.shape, self.cfg.bbox_jitter, self.rng)
            prompt["box"] = torch.as_tensor(bb, dtype=torch.float32)
        else:
            pts, lbls = sample_point_prompts(gt_resized, rng=self.rng)
            prompt["points"] = torch.as_tensor(pts, dtype=torch.float32)
            prompt["labels"] = torch.as_tensor(lbls, dtype=torch.int64)

        # Stack the 3-slice window to an RGB-like tensor [3, H, W]
        if window.shape[0] == 1:
            rgb = np.repeat(window, 3, axis=0)
        elif window.shape[0] == 3:
            rgb = window
        else:
            rgb = window[[0, window.shape[0] // 2, -1]]

        return {
            "image": torch.as_tensor(rgb, dtype=torch.float32),
            "mask": torch.as_tensor(gt_resized, dtype=torch.float32),
            "prompt": prompt,
            "meta": (str(vol.path), int(z)),
        }


def collate_slice_batch(batch: Sequence[SliceWindow]) -> dict[str, object]:
    """Custom collate: stack images/masks, keep prompts as list (variable sizes)."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    prompts = [b["prompt"] for b in batch]
    metas = [b["meta"] for b in batch]
    return {"image": images, "mask": masks, "prompt": prompts, "meta": metas}


def discover_npz(root: str | Path) -> list[Path]:
    return sorted(Path(root).rglob("*.npz"))
