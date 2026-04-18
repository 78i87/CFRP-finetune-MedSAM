"""LoRA and Conv-LoRA adapters, plus injection helpers for SAM2 / MedSAM2.

Two adapter types are implemented:

- ``LoRALinear``: standard LoRA ``out = W x + (alpha/r) B A x`` on the weight
  matrix of a frozen ``nn.Linear``. Matches Hu et al. 2021.
- ``ConvLoRALinear``: same, but inserts a depth-wise 3x3 conv across a 2D
  spatial grid between the down-projection ``A`` and the up-projection ``B``,
  matching the design idea of Zhong et al. 2024 (Conv-LoRA, ICLR).

SAM2 / MedSAM2 uses a Hiera backbone — a hierarchical ViT. We wrap the QKV
and output projections of every Hiera ``MultiScaleBlock``'s attention module
(and, optionally, the memory-attention blocks). The helper
``inject_lora_into_sam2`` walks the module tree and swaps ``nn.Linear`` layers
by attribute name.

For CI / smoke tests on a machine without the upstream MedSAM2 repo, the same
modules are applied to a tiny toy transformer stub defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a LoRA residual.

    out = W x + dropout(x) @ A^T  -> conv (optional) -> @ B^T  * (alpha / r)
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_conv: bool = False,
        conv_kernel: int = 3,
        grid_hint: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)
        self.use_conv = use_conv
        self.conv_kernel = conv_kernel
        self.grid_hint = grid_hint

        # Freeze original.
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        # Match the device and dtype of the base weights so wrapping works
        # post-``.to(device)`` of the parent model.
        device = base.weight.device
        dtype = base.weight.dtype

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if use_conv:
            pad = conv_kernel // 2
            self.lora_conv = nn.Conv2d(
                rank, rank, kernel_size=conv_kernel, padding=pad, groups=rank, bias=False
            )
            nn.init.dirac_(self.lora_conv.weight)
            self.lora_conv = self.lora_conv.to(device=device, dtype=dtype)
        else:
            self.lora_conv = None

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, r={self.rank}, "
            f"alpha={self.alpha}, conv={self.use_conv}"
        )

    def _spatial_grid_size(self, N: int) -> tuple[int, int]:
        if self.grid_hint is not None:
            return self.grid_hint
        s = int(round(math.sqrt(N)))
        if s * s == N:
            return s, s
        # Give up on conv when the token count isn't a perfect square.
        raise ValueError(f"cannot infer spatial grid from N={N}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        lora_x = self.lora_dropout(x)
        # x: (..., in_features)
        down = F.linear(lora_x, self.lora_A)  # (..., rank)

        if self.lora_conv is not None and down.ndim == 3:
            # Expect shape (B, N, rank); reshape to (B, rank, H, W) then back.
            B, N, R = down.shape
            try:
                H, W = self._spatial_grid_size(N)
            except ValueError:
                H = W = None
            if H is None:
                conv_out = down
            else:
                grid = down.transpose(1, 2).reshape(B, R, H, W)
                grid = self.lora_conv(grid)
                conv_out = grid.reshape(B, R, N).transpose(1, 2)
        else:
            # Non-sequence input (e.g. (B, C) prompts) or conv disabled.
            conv_out = down

        up = F.linear(conv_out, self.lora_B)
        return base_out + self.scaling * up


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_substrings: tuple[str, ...] = ("qkv", "proj", "q_proj", "k_proj", "v_proj", "out_proj")
    exclude_substrings: tuple[str, ...] = ("mask_decoder.iou_prediction_head",)
    use_conv: bool = False
    conv_kernel: int = 3
    train_mask_decoder: bool = False
    train_prompt_encoder: bool = False
    include_memory_attention: bool = True

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def _module_matches(name: str, cfg: LoRAConfig) -> bool:
    if not any(s in name for s in cfg.target_substrings):
        return False
    if any(s in name for s in cfg.exclude_substrings):
        return False
    if not cfg.include_memory_attention and "memory_attention" in name:
        return False
    return True


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    """Walk ``model``, wrap each matching ``nn.Linear`` by a ``LoRALinear``.

    Returns the list of wrapped module paths. Non-matching parameters are
    frozen (``requires_grad = False``) except for LoRA params, and optionally
    the mask decoder / prompt encoder.
    """
    replaced: list[str] = []

    # First pass: freeze everything.
    for p in model.parameters():
        p.requires_grad = False

    # Then inject LoRA.
    def _recursive(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and _module_matches(full, cfg):
                wrapped = LoRALinear(
                    child,
                    rank=cfg.rank,
                    alpha=cfg.alpha,
                    dropout=cfg.dropout,
                    use_conv=cfg.use_conv,
                    conv_kernel=cfg.conv_kernel,
                )
                setattr(module, child_name, wrapped)
                replaced.append(full)
            else:
                _recursive(child, full)

    _recursive(model)

    # Unfreeze newly added LoRA params explicitly in case a parent module
    # has buffers with requires_grad fiddling.
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n or "lora_conv" in n:
            p.requires_grad = True

    # Optionally re-enable mask decoder / prompt encoder.
    if cfg.train_mask_decoder:
        for n, p in model.named_parameters():
            if "mask_decoder" in n and "base.weight" not in n and "base.bias" not in n:
                p.requires_grad = True
    if cfg.train_prompt_encoder:
        for n, p in model.named_parameters():
            if "prompt_encoder" in n and "base.weight" not in n and "base.bias" not in n:
                p.requires_grad = True

    return replaced


def trainable_param_summary(model: nn.Module) -> dict[str, int]:
    total = 0
    trainable = 0
    lora = 0
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
        if "lora_" in n:
            lora += p.numel()
    return {"total": total, "trainable": trainable, "lora": lora}


# ---------------------------------------------------------------------------
# Tiny transformer stub for smoke tests.
# ---------------------------------------------------------------------------


class _ToyAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class ToyTransformer(nn.Module):
    """A tiny stand-in for SAM2's image encoder; used by tests to verify
    adapter injection and gradient flow without requiring the heavy model."""

    def __init__(self, dim: int = 64, depth: int = 2, n_tokens: int = 64) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.dim = dim
        self.tok_embed = nn.Linear(3, dim)
        self.blocks = nn.ModuleList(
            [nn.ModuleDict({"attn": _ToyAttention(dim), "norm": nn.LayerNorm(dim)}) for _ in range(depth)]
        )
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_tokens, 3)
        h = self.tok_embed(x)
        for blk in self.blocks:
            h = h + blk["attn"](blk["norm"](h))
        return self.head(h).squeeze(-1)
