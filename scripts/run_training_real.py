"""Train LoRA / Conv-LoRA / full-FT on the real Chalmers labelled CFRP volume."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from cfrp_medsam2.lora import LoRAConfig
from cfrp_medsam2.model import ModelConfig
from cfrp_medsam2.train import TrainConfig, train


BACKBONES: dict[str, tuple[str, str]] = {
    "tiny": (
        "configs/sam2.1_hiera_t512.yaml",
        "checkpoints/sam2.1_hiera_tiny.pt",
    ),
    "tiny_medsam2": (
        # Same Hiera-tiny architecture as `tiny`; different init weights
        # (wanglab/MedSAM2 medical fine-tune on top of SAM2.1 tiny).
        "configs/sam2.1_hiera_t512.yaml",
        "checkpoints/MedSAM2_latest.pt",
    ),
    "base_plus": (
        "configs/sam2.1_hiera_b+_512.yaml",
        "checkpoints/sam2.1_hiera_base_plus.pt",
    ),
}


def base_cfg(
    repo: Path,
    regime: str,
    epochs: int,
    backbone: str,
    save_every_epoch: bool,
) -> TrainConfig:
    tr = (str(repo / "data" / "processed" / "cfrp_real" / "train_00.npz"),)
    va = (str(repo / "data" / "processed" / "cfrp_real" / "val_00.npz"),)
    base = ("qkv", "q_proj", "k_proj", "v_proj", "out_proj", "proj")
    excl = ("mask_decoder.iou_prediction_head", "mlp", "obj_ptr")
    sam2_config, ckpt_rel = BACKBONES[backbone]
    log_suffix = f"_{backbone}" if backbone != "tiny" else ""
    return TrainConfig(
        regime=regime,
        model=ModelConfig(
            backend="medsam2",
            sam2_config=sam2_config,
            checkpoint=str(repo / ckpt_rel),
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
        train_volumes=tr,
        val_volumes=va,
        image_size=512,
        batch_size=1,
        epochs=epochs,
        lr=1e-4,
        full_ft_lr=1e-5,
        prompt_mode="mixed" if regime != "full_ft" else "bbox",
        ckpt_dir=str(repo / "checkpoints"),
        log_path=str(repo / "logs" / f"{regime}_real{log_suffix}.csv"),
        save_every_epoch=save_every_epoch,
        tensorboard_dir=str(repo / "logs" / "tb" / f"{regime}_real{log_suffix}"),
        # Per-yarn supervision: one connected yarn tow per sample. We bump
        # the minimum CC size so we only train on substantial tows, and cap
        # samples per epoch so training finishes in minutes, not hours.
        per_component=True,
        target_classes=(1, 2, 3),
        min_component_voxels=1500,
        max_train_samples=600,
        max_val_samples=200,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("regime", choices=["lora", "conv_lora", "full_ft"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument(
        "--backbone",
        choices=list(BACKBONES.keys()),
        default="tiny",
        help="Which SAM2.1 backbone to fine-tune. 'base_plus' needs ~330 MB checkpoint from setup_medsam2.sh.",
    )
    p.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="In addition to the best checkpoint, write one ckpt per epoch (for overfitting debugging).",
    )
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = base_cfg(repo, args.regime, args.epochs, args.backbone, args.save_every_epoch)
    result = train(cfg)
    print("\nRESULT:", result["regime"], args.backbone, "best_val_dice", result["best_val_dice"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
