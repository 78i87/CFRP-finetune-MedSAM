"""Synthetic toy XCT-like volumes with fibres embedded in a matrix.

This exists purely to let the rest of the pipeline be tested end-to-end
without needing the multi-GB CFRP / SiC-SiC datasets. The toy volumes
mimic the key difficulty of real CFRP XCT: fibre and matrix have almost
identical mean intensity, so the signal is local texture + geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToyVolumeConfig:
    shape: tuple[int, int, int] = (64, 256, 256)  # Z, H, W
    num_fibres: int = 40
    fibre_radius: tuple[float, float] = (3.0, 5.0)
    fibre_intensity: float = 0.52          # mean grey of fibre
    matrix_intensity: float = 0.48         # mean grey of matrix
    noise_sigma: float = 0.06              # Gaussian noise
    ring_artifact_strength: float = 0.02   # faint concentric rings, XCT-ish
    seed: int = 0


def make_toy_volume(cfg: ToyVolumeConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (image[Z,H,W] float32 in [0,1], mask[Z,H,W] uint8 with 1=fibre).

    Fibres are straight cylinders running along the Z axis, so the central
    2D slice of the volume is a near-circular packing of fibres. Their mean
    intensity differs from the matrix by only ~0.04 (8% of dynamic range),
    roughly matching the poor contrast of carbon fibre vs epoxy in XCT.
    """
    cfg = cfg or ToyVolumeConfig()
    rng = np.random.default_rng(cfg.seed)
    Z, H, W = cfg.shape

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # Place non-overlapping fibre centres by rejection sampling.
    centres: list[tuple[float, float, float]] = []
    tries = 0
    while len(centres) < cfg.num_fibres and tries < 5000:
        tries += 1
        cy = rng.uniform(10, H - 10)
        cx = rng.uniform(10, W - 10)
        r = rng.uniform(*cfg.fibre_radius)
        ok = True
        for (cy2, cx2, r2) in centres:
            if (cy - cy2) ** 2 + (cx - cx2) ** 2 < (r + r2 + 1.0) ** 2:
                ok = False
                break
        if ok:
            centres.append((cy, cx, r))

    mask2d = np.zeros((H, W), dtype=np.uint8)
    fibre_noise = np.zeros((H, W), dtype=np.float32)
    for (cy, cx, r) in centres:
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        hit = d2 <= r * r
        mask2d[hit] = 1
        # Slight per-fibre intensity jitter so the model has to learn texture, not a global shift.
        fibre_noise[hit] = rng.normal(0.0, 0.01)

    base2d = np.where(
        mask2d.astype(bool),
        cfg.fibre_intensity + fibre_noise,
        cfg.matrix_intensity,
    ).astype(np.float32)

    # Extend along Z with small inter-slice jitter, add ring + Gaussian noise per slice.
    rings = cfg.ring_artifact_strength * np.sin(
        2 * np.pi * np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / 24.0
    ).astype(np.float32)

    image = np.empty((Z, H, W), dtype=np.float32)
    for z in range(Z):
        noise = rng.normal(0.0, cfg.noise_sigma, size=(H, W)).astype(np.float32)
        image[z] = np.clip(base2d + rings + noise, 0.0, 1.0)

    mask = np.broadcast_to(mask2d, (Z, H, W)).copy()
    return image, mask


def make_toy_dataset(
    n_train: int = 2,
    n_val: int = 1,
    cfg: ToyVolumeConfig | None = None,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Create a tiny train / val set of toy volumes."""
    base = cfg or ToyVolumeConfig()
    train = []
    for i in range(n_train):
        train.append(make_toy_volume(ToyVolumeConfig(**{**base.__dict__, "seed": 1000 + i})))
    val = []
    for i in range(n_val):
        val.append(make_toy_volume(ToyVolumeConfig(**{**base.__dict__, "seed": 9000 + i})))
    return {"train": train, "val": val}
