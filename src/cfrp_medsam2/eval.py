"""Evaluation metrics for fibre segmentation.

Standard Dice / IoU plus a custom **fibre-continuity** metric that measures
how well a predicted mask preserves long, axis-aligned connected components.
This matters for CFRP/CMC because broken fibres in predictions are the
dominant failure mode of slice-wise 2D models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _binarize(x: torch.Tensor | np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return (x > threshold).astype(np.uint8)


def dice_2d(pred: torch.Tensor | np.ndarray, gt: torch.Tensor | np.ndarray, eps: float = 1e-6) -> float:
    p = _binarize(pred)
    g = _binarize(gt)
    inter = float((p & g).sum())
    denom = float(p.sum() + g.sum())
    return (2.0 * inter + eps) / (denom + eps)


def iou_2d(pred: torch.Tensor | np.ndarray, gt: torch.Tensor | np.ndarray, eps: float = 1e-6) -> float:
    p = _binarize(pred)
    g = _binarize(gt)
    inter = float((p & g).sum())
    union = float((p | g).sum())
    return (inter + eps) / (union + eps)


def dice_3d(pred_vol: np.ndarray, gt_vol: np.ndarray, eps: float = 1e-6) -> float:
    p = _binarize(pred_vol)
    g = _binarize(gt_vol)
    inter = float((p & g).sum())
    denom = float(p.sum() + g.sum())
    return (2.0 * inter + eps) / (denom + eps)


def per_slice_dice(pred_vol: np.ndarray, gt_vol: np.ndarray) -> np.ndarray:
    return np.asarray([dice_2d(pred_vol[z], gt_vol[z]) for z in range(pred_vol.shape[0])])


@dataclass
class FibreContinuityResult:
    pred_mean_length: float
    gt_mean_length: float
    continuity_ratio: float  # pred / gt; 1.0 == perfect preservation


def fibre_continuity(
    pred_vol: np.ndarray,
    gt_vol: np.ndarray,
    axis: int = 0,
    min_voxels: int = 50,
) -> FibreContinuityResult:
    """Approximate fibre continuity along ``axis`` using 3D connected components.

    Uses 6-connectivity. Returns the ratio of the mean component extent along
    the fibre axis between predictions and ground truth, ignoring components
    smaller than ``min_voxels``. A ratio near 1 indicates the model preserves
    long, unbroken fibre tracks; a ratio << 1 means the prediction fragments
    fibres slice-by-slice.
    """
    from scipy.ndimage import label as cc_label

    def _mean_axis_extent(vol: np.ndarray) -> float:
        vol_b = _binarize(vol)
        if vol_b.sum() == 0:
            return 0.0
        lab, n = cc_label(vol_b)
        extents = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            if coords.shape[0] < min_voxels:
                continue
            ext = coords[:, axis].max() - coords[:, axis].min() + 1
            extents.append(float(ext))
        return float(np.mean(extents)) if extents else 0.0

    pred_len = _mean_axis_extent(pred_vol)
    gt_len = _mean_axis_extent(gt_vol)
    ratio = pred_len / gt_len if gt_len > 0 else 0.0
    return FibreContinuityResult(pred_len, gt_len, ratio)


def summarize(pred_vol: np.ndarray, gt_vol: np.ndarray) -> dict[str, float]:
    """Headline metrics for a volume."""
    slice_dice = per_slice_dice(pred_vol, gt_vol)
    cont = fibre_continuity(pred_vol, gt_vol)
    return {
        "dice_3d": dice_3d(pred_vol, gt_vol),
        "dice_slice_mean": float(slice_dice.mean()),
        "dice_slice_std": float(slice_dice.std()),
        "fibre_continuity": cont.continuity_ratio,
        "fibre_length_pred": cont.pred_mean_length,
        "fibre_length_gt": cont.gt_mean_length,
    }
