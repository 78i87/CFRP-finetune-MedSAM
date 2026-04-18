"""Shared training loop for LoRA / Conv-LoRA / full fine-tune regimes.

The loop is deliberately plain — a single file that any of the notebooks
can call with a :class:`TrainConfig`. It tracks per-epoch train / val Dice,
saves the best checkpoint, and writes a small CSV log so the ablation
notebook can tabulate results.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .data import SliceDataset, SliceDatasetConfig, collate_slice_batch
from .eval import dice_2d
from .lora import LoRAConfig, inject_lora, trainable_param_summary
from .model import ModelConfig, SegModel


@dataclass
class TrainConfig:
    regime: str = "lora"  # "zero_shot" | "lora" | "conv_lora" | "full_ft"
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    train_volumes: tuple[str, ...] = ()
    val_volumes: tuple[str, ...] = ()
    image_size: int = 512
    batch_size: int = 2
    num_workers: int = 0
    epochs: int = 5
    lr: float = 1e-4
    full_ft_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    prompt_mode: str = "mixed"
    ckpt_dir: str = "checkpoints"
    log_path: str = "logs/train.csv"
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Per-component supervision (one yarn per sample, one prompt per yarn).
    per_component: bool = False
    target_classes: tuple[int, ...] | None = None
    min_component_voxels: int = 50
    binary_class_id: int | None = 1
    # Cap the number of iterations per epoch (useful when the per-component
    # dataset has tens of thousands of samples).
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    # If true, write a checkpoint at the end of every epoch in addition to
    # the best-so-far. Useful for debugging overfitting and for resuming runs.
    save_every_epoch: bool = False
    # Optional TensorBoard log directory. If None (default) no TB writer is
    # created; if set but `torch.utils.tensorboard` can't be imported, we
    # print a warning and carry on.
    tensorboard_dir: str | None = None


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    num = 2.0 * (pred * target).sum(dim=(-2, -1)) + eps
    den = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + eps
    return 1.0 - (num / den).mean()


def focal_loss(
    logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return dice_loss(logits, target) + 0.5 * focal_loss(logits, target)


def _prepare_model(cfg: TrainConfig) -> tuple[SegModel, list[str]]:
    model = SegModel(cfg.model)
    replaced: list[str] = []
    if cfg.regime in ("lora", "conv_lora"):
        lora_cfg = LoRAConfig(**{**asdict(cfg.lora), "use_conv": cfg.regime == "conv_lora"})
        replaced = inject_lora(model, lora_cfg)
    elif cfg.regime == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
    elif cfg.regime == "zero_shot":
        for p in model.parameters():
            p.requires_grad = False
    else:
        raise ValueError(f"unknown regime {cfg.regime!r}")
    return model, replaced


def _build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    common = dict(
        image_size=cfg.image_size,
        per_component=cfg.per_component,
        target_classes=cfg.target_classes,
        min_component_voxels=cfg.min_component_voxels,
        binary_class_id=cfg.binary_class_id,
    )
    train_ds = SliceDataset(
        SliceDatasetConfig(
            volume_paths=[Path(p) for p in cfg.train_volumes],
            prompt_mode=cfg.prompt_mode,
            **common,
        ),
        seed=cfg.seed,
    )
    val_ds = SliceDataset(
        SliceDatasetConfig(
            volume_paths=[Path(p) for p in cfg.val_volumes],
            prompt_mode="bbox",
            **common,
        ),
        seed=cfg.seed + 1,
    )
    if cfg.max_train_samples is not None and cfg.max_train_samples < len(train_ds):
        train_sampler = RandomSampler(
            train_ds, replacement=False, num_samples=cfg.max_train_samples
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    if cfg.max_val_samples is not None and cfg.max_val_samples < len(val_ds):
        # Take a deterministic stride so the val metric is stable across epochs.
        stride = max(1, len(val_ds) // cfg.max_val_samples)
        val_indices = list(range(0, len(val_ds), stride))[: cfg.max_val_samples]
        val_ds = torch.utils.data.Subset(val_ds, val_indices)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_slice_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_slice_batch,
    )
    return train_loader, val_loader


def _step(
    model: SegModel,
    batch: dict,
    device: str,
) -> tuple[torch.Tensor, float]:
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    prompts = batch["prompt"]
    boxes = []
    for p in prompts:
        if "box" in p:
            boxes.append(p["box"].to(device))
        else:
            boxes.append(torch.tensor([0, 0, images.shape[-1] - 1, images.shape[-2] - 1], device=device, dtype=torch.float32))

    logits = model.forward_slice(images, boxes=boxes)
    loss = combined_loss(logits, masks)
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > 0.5).float()
        # Mean Dice across the batch.
        d = [
            dice_2d(preds[i].cpu().numpy(), masks[i].cpu().numpy())
            for i in range(preds.shape[0])
        ]
        mean_dice = float(np.mean(d))
    return loss, mean_dice


def train(cfg: TrainConfig) -> dict[str, object]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    model, replaced = _prepare_model(cfg)
    model.to(cfg.device)
    summary = trainable_param_summary(model)
    print(
        f"[train:{cfg.regime}] params total={summary['total']:,} "
        f"trainable={summary['trainable']:,} lora={summary['lora']:,} "
        f"({100.0 * summary['trainable'] / max(summary['total'], 1):.2f}% trainable)"
    )
    if replaced:
        print(f"[train:{cfg.regime}] LoRA injected into {len(replaced)} modules; first 3: {replaced[:3]}")

    lr = cfg.full_ft_lr if cfg.regime == "full_ft" else cfg.lr
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        print("[train] nothing to train; running eval only.")
        return _eval_only(model, cfg)
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(cfg.epochs, 1))

    train_loader, val_loader = _build_loaders(cfg)

    tb_writer = None
    if cfg.tensorboard_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            Path(cfg.tensorboard_dir).mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=cfg.tensorboard_dir)
        except Exception as e:
            print(f"[train] tensorboard unavailable ({e.__class__.__name__}: {e}); skipping TB logging.")

    log_stem = Path(cfg.log_path).stem
    suffix = log_stem[len(cfg.regime):].strip("_") or "toy"

    def _save_ckpt(path: Path, epoch: int, va_dice: float) -> None:
        torch.save(
            {
                "state_dict": {
                    n: p.detach().cpu()
                    for n, p in model.state_dict().items()
                    if cfg.regime == "full_ft" or ("lora_" in n or "mask_decoder" in n)
                },
                "cfg": asdict(cfg),
                "epoch": epoch,
                "val_dice": va_dice,
            },
            path,
        )

    best_val = -1.0
    log_rows: list[dict] = []
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_dice", "val_dice", "time_s"])
        writer.writeheader()

        for epoch in range(cfg.epochs):
            t0 = time.time()
            model.train()
            tr_loss = tr_dice = 0.0
            tr_count = 0
            for batch in train_loader:
                optim.zero_grad()
                loss, d = _step(model, batch, cfg.device)
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                optim.step()
                tr_loss += float(loss.item())
                tr_dice += d
                tr_count += 1
            sched.step()
            tr_loss /= max(tr_count, 1)
            tr_dice /= max(tr_count, 1)

            model.eval()
            va_dice = 0.0
            va_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    _, d = _step(model, batch, cfg.device)
                    va_dice += d
                    va_count += 1
            va_dice /= max(va_count, 1)

            dt = time.time() - t0
            row = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_dice": tr_dice,
                "val_dice": va_dice,
                "time_s": dt,
            }
            writer.writerow(row)
            f.flush()
            log_rows.append(row)
            print(
                f"[{cfg.regime}] epoch {epoch:02d}  train_loss={tr_loss:.4f}  "
                f"train_dice={tr_dice:.4f}  val_dice={va_dice:.4f}  ({dt:.1f}s)"
            )

            if tb_writer is not None:
                tb_writer.add_scalar("loss/train", tr_loss, epoch)
                tb_writer.add_scalar("dice/train", tr_dice, epoch)
                tb_writer.add_scalar("dice/val", va_dice, epoch)
                tb_writer.add_scalar("lr", optim.param_groups[0]["lr"], epoch)

            if va_dice > best_val:
                best_val = va_dice
                _save_ckpt(ckpt_dir / f"{cfg.regime}_{suffix}_best.pt", epoch, va_dice)

            if cfg.save_every_epoch:
                _save_ckpt(ckpt_dir / f"{cfg.regime}_{suffix}_epoch{epoch:02d}.pt", epoch, va_dice)

    if tb_writer is not None:
        tb_writer.close()

    return {"regime": cfg.regime, "best_val_dice": best_val, "log": log_rows}


def _eval_only(model: SegModel, cfg: TrainConfig) -> dict[str, object]:
    model.eval()
    _, val_loader = _build_loaders(cfg)
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            _, d = _step(model, batch, cfg.device)
            total += d
            count += 1
    return {"regime": cfg.regime, "best_val_dice": total / max(count, 1), "log": []}
