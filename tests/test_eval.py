"""Tests for eval metrics."""

from __future__ import annotations

import numpy as np

from cfrp_medsam2.eval import dice_2d, dice_3d, fibre_continuity, summarize


def test_dice_2d_perfect() -> None:
    a = np.zeros((8, 8), dtype=np.uint8)
    a[2:6, 2:6] = 1
    assert dice_2d(a, a) > 0.99


def test_dice_2d_disjoint_is_zero() -> None:
    a = np.zeros((8, 8), dtype=np.uint8)
    b = np.zeros((8, 8), dtype=np.uint8)
    a[:4, :4] = 1
    b[4:, 4:] = 1
    assert dice_2d(a, b) < 1e-3


def test_dice_3d_partial_overlap() -> None:
    a = np.zeros((4, 8, 8), dtype=np.uint8)
    a[:, 2:6, 2:6] = 1
    b = a.copy()
    b[:, 4:, :] = 0
    d = dice_3d(a, b)
    assert 0.0 < d < 1.0


def test_fibre_continuity_preserved_for_identical_inputs() -> None:
    # A single cylindrical fibre along axis 0.
    vol = np.zeros((12, 20, 20), dtype=np.uint8)
    vol[:, 8:12, 8:12] = 1
    res = fibre_continuity(vol, vol, axis=0)
    assert abs(res.continuity_ratio - 1.0) < 1e-6


def test_fibre_continuity_penalizes_slicewise_breaks() -> None:
    gt = np.zeros((12, 20, 20), dtype=np.uint8)
    gt[:, 8:12, 8:12] = 1
    pred = gt.copy()
    pred[3:5, :, :] = 0  # a gap breaks the fibre into two shorter components
    res = fibre_continuity(pred, gt, axis=0, min_voxels=1)
    assert res.continuity_ratio < 1.0


def test_summarize_keys() -> None:
    vol = np.zeros((4, 16, 16), dtype=np.uint8)
    vol[:, 4:8, 4:8] = 1
    out = summarize(vol, vol)
    for k in ("dice_3d", "dice_slice_mean", "fibre_continuity"):
        assert k in out
