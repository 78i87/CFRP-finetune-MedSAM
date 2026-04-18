"""Visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def slice_triptych(
    image: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_path: str | Path | None = None
):
    """Side-by-side image / GT / prediction plot for one slice."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("image")
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(gt, cmap="Reds", alpha=0.45)
    axes[1].set_title("ground truth")
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(pred, cmap="Blues", alpha=0.45)
    axes[2].set_title("prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def overlay_slice(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Return an RGB overlay image (H, W, 3) in float [0,1]."""
    img = image.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    rgb = np.stack([img, img, img], axis=-1)
    m = mask.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array([1.0, 0.2, 0.2])
    return np.clip(rgb, 0.0, 1.0)


def volume_mid_slices(vol: np.ndarray, n: int = 6) -> list[np.ndarray]:
    """Sample ``n`` slices evenly through the Z axis."""
    Z = vol.shape[0]
    idx = np.linspace(0, Z - 1, n).astype(int)
    return [vol[i] for i in idx]
