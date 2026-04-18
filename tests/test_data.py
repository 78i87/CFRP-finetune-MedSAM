"""Tests for data loading and prompt simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cfrp_medsam2.data import (
    SliceDataset,
    SliceDatasetConfig,
    collate_slice_batch,
    jitter_bbox,
    mask_to_bbox,
    save_npz_volume,
)
from cfrp_medsam2.synthetic import ToyVolumeConfig, make_toy_volume


def _write_toy(tmp_path: Path) -> Path:
    imgs, gts = make_toy_volume(ToyVolumeConfig(shape=(8, 64, 64), num_fibres=5))
    out = tmp_path / "toy.npz"
    save_npz_volume(out, imgs, gts)
    return out


def test_mask_to_bbox_identifies_tight_rectangle() -> None:
    m = np.zeros((32, 32), dtype=np.uint8)
    m[5:10, 20:28] = 1
    bb = mask_to_bbox(m)
    assert bb == (20, 5, 27, 9)


def test_mask_to_bbox_empty_returns_none() -> None:
    assert mask_to_bbox(np.zeros((4, 4), dtype=np.uint8)) is None


def test_jitter_bbox_stays_in_bounds() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        bb = jitter_bbox((5, 5, 15, 15), (32, 32), jitter=5, rng=rng)
        x0, y0, x1, y1 = bb
        assert 0 <= x0 < x1 <= 32
        assert 0 <= y0 < y1 <= 32


def test_slice_dataset_yields_correct_shapes(tmp_path) -> None:
    p = _write_toy(tmp_path)
    ds = SliceDataset(SliceDatasetConfig(volume_paths=[p], image_size=64, prompt_mode="bbox"))
    assert len(ds) > 0
    sample = ds[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["mask"].shape == (64, 64)
    assert sample["prompt"]["box"].shape == (4,)


def test_collate_stacks_images_but_keeps_prompts(tmp_path) -> None:
    p = _write_toy(tmp_path)
    ds = SliceDataset(SliceDatasetConfig(volume_paths=[p], image_size=64, prompt_mode="bbox"))
    batch = collate_slice_batch([ds[0], ds[1]])
    assert batch["image"].shape == (2, 3, 64, 64)
    assert batch["mask"].shape == (2, 64, 64)
    assert isinstance(batch["prompt"], list) and len(batch["prompt"]) == 2


def test_per_component_dataset_yields_disjoint_masks(tmp_path) -> None:
    """per_component mode returns one connected fibre per sample."""
    # Build a volume with 3 tiny square fibres so we know the CC count.
    import numpy as np

    from cfrp_medsam2.data import save_npz_volume

    imgs = np.zeros((2, 32, 32), dtype=np.uint8)
    gts = np.zeros((2, 32, 32), dtype=np.uint8)
    gts[:, 4:10, 4:10] = 1
    gts[:, 20:26, 4:10] = 1
    gts[:, 4:10, 20:26] = 1
    path = tmp_path / "cc.npz"
    save_npz_volume(path, imgs, gts)

    ds = SliceDataset(
        SliceDatasetConfig(
            volume_paths=[path],
            image_size=32,
            per_component=True,
            target_classes=(1,),
            min_component_voxels=4,
            prompt_mode="bbox",
        )
    )
    # 2 slices * 3 components each = 6 samples.
    assert len(ds) == 6
    s = ds[0]
    mask = s["mask"].numpy()
    # Each sample's mask must be a single CC, i.e. no more than 1 blob.
    from scipy.ndimage import label as cc_label
    _, n = cc_label(mask > 0.5)
    assert n == 1
