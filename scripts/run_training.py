"""Run LoRA / Conv-LoRA / full-FT training on the toy volumes and write checkpoints + CSVs."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

from cfrp_medsam2.lora import LoRAConfig
from cfrp_medsam2.model import ModelConfig
from cfrp_medsam2.train import TrainConfig, train


def base_cfg(repo: Path, regime: str, epochs: int) -> TrainConfig:
    train_vols = tuple(
        str(p) for p in sorted((repo / "data" / "processed" / "toy").glob("train_*.npz"))
    )
    val_vols = tuple(
        str(p) for p in sorted((repo / "data" / "processed" / "toy").glob("val_*.npz"))
    )
    base = ("qkv", "q_proj", "k_proj", "v_proj", "out_proj", "proj")
    excl = ("mask_decoder.iou_prediction_head", "mlp", "obj_ptr")
    return TrainConfig(
        regime=regime,
        model=ModelConfig(
            backend="medsam2",
            checkpoint=str(repo / "checkpoints" / "sam2.1_hiera_tiny.pt"),
            image_size=512,
        ),
        lora=LoRAConfig(
            rank=8,
            alpha=16.0,
            dropout=0.05,
            target_substrings=base,
            exclude_substrings=excl,
            use_conv=(regime == "conv_lora"),
            train_mask_decoder=True,
            include_memory_attention=True,
        ),
        train_volumes=train_vols,
        val_volumes=val_vols,
        image_size=512,
        batch_size=1,
        epochs=epochs,
        lr=1e-4,
        full_ft_lr=1e-5,
        prompt_mode="mixed" if regime != "full_ft" else "bbox",
        ckpt_dir=str(repo / "checkpoints"),
        log_path=str(repo / "logs" / f"{regime}.csv"),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("regime", choices=["lora", "conv_lora", "full_ft"])
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = base_cfg(repo, args.regime, args.epochs)
    result = train(cfg)
    print("\nRESULT:", result["regime"], "best_val_dice", result["best_val_dice"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
