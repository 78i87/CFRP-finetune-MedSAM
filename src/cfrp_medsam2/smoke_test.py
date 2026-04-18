"""End-to-end smoke test on synthetic toy data.

Run as ``python -m cfrp_medsam2.smoke_test``. Writes two toy volumes,
trains zero-shot / LoRA / Conv-LoRA / full-FT for a few epochs each, and
prints a small ablation table. Requires only the fallback backend, so it
exercises the code path without needing the MedSAM2 checkpoint.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .data import save_npz_volume
from .eval import summarize
from .lora import LoRAConfig
from .model import ModelConfig
from .synthetic import ToyVolumeConfig, make_toy_dataset
from .train import TrainConfig, train


def _write_toys(out_dir: Path) -> tuple[list[Path], list[Path]]:
    cfg = ToyVolumeConfig(shape=(16, 128, 128), num_fibres=15)
    data = make_toy_dataset(n_train=2, n_val=1, cfg=cfg)
    train_paths, val_paths = [], []
    for i, (imgs, gts) in enumerate(data["train"]):
        p = out_dir / f"train_{i:02d}.npz"
        save_npz_volume(p, imgs, gts)
        train_paths.append(p)
    for i, (imgs, gts) in enumerate(data["val"]):
        p = out_dir / f"val_{i:02d}.npz"
        save_npz_volume(p, imgs, gts)
        val_paths.append(p)
    return train_paths, val_paths


def main() -> int:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke_test] device={device}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        train_paths, val_paths = _write_toys(td_path)
        ckpt_dir = td_path / "ckpts"

        base = dict(
            model=ModelConfig(backend="fallback", image_size=128),
            train_volumes=tuple(str(p) for p in train_paths),
            val_volumes=tuple(str(p) for p in val_paths),
            image_size=128,
            batch_size=2,
            epochs=2,
            ckpt_dir=str(ckpt_dir),
            device=device,
        )

        results: dict[str, dict] = {}
        for regime in ("zero_shot", "lora", "conv_lora", "full_ft"):
            cfg = TrainConfig(
                regime=regime,
                log_path=str(td_path / f"log_{regime}.csv"),
                lora=LoRAConfig(rank=4, alpha=8.0, use_conv=(regime == "conv_lora")),
                **base,
            )
            out = train(cfg)
            results[regime] = out

        print("\n=== smoke test summary ===")
        for regime, out in results.items():
            print(f"  {regime:10s}  val_dice={out['best_val_dice']:.4f}")

        # Quick volume-level sanity check with summarize()
        from .data import NpzVolume

        v = NpzVolume.load(val_paths[0])
        pred_vol = (v.imgs > 128).astype(np.uint8)  # dummy prediction
        metrics = summarize(pred_vol, (v.gts == 1).astype(np.uint8))
        print("  volume-level metrics (dummy pred):", json.dumps(metrics, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
