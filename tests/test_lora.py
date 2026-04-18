"""Unit tests for LoRA / Conv-LoRA modules and injection helpers."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from cfrp_medsam2.lora import (
    LoRAConfig,
    LoRALinear,
    ToyTransformer,
    inject_lora,
    trainable_param_summary,
)


def test_loralinear_zero_init_matches_base() -> None:
    """At init, B=0, so LoRALinear must be identical to the base Linear."""
    torch.manual_seed(0)
    base = nn.Linear(16, 32, bias=True)
    wrapped = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(3, 10, 16)
    out_base = base(x)
    out_wrapped = wrapped(x)
    assert torch.allclose(out_base, out_wrapped, atol=1e-6)


def test_conv_lora_changes_output_after_update() -> None:
    """Once LoRA params are non-zero, Conv-LoRA output differs from base."""
    torch.manual_seed(0)
    base = nn.Linear(16, 16)
    wrapped = LoRALinear(base, rank=4, alpha=8.0, use_conv=True)
    with torch.no_grad():
        wrapped.lora_B.add_(0.1 * torch.randn_like(wrapped.lora_B))

    # Use a perfect-square token count so the conv path runs.
    x = torch.randn(2, 64, 16)  # 64 = 8x8 grid
    out_base = base(x)
    out_wrapped = wrapped(x)
    assert not torch.allclose(out_base, out_wrapped, atol=1e-5)


def test_conv_lora_fallback_on_non_sequence_input() -> None:
    """Conv-LoRA should behave like LoRA on (B, C) tensors."""
    torch.manual_seed(0)
    base = nn.Linear(5, 8)
    wrapped = LoRALinear(base, rank=2, alpha=4.0, use_conv=True)
    x = torch.randn(3, 5)  # non-sequence
    # Must not raise even though the conv path can't apply.
    out = wrapped(x)
    assert out.shape == (3, 8)


def test_inject_lora_freezes_base_and_trains_adapters() -> None:
    torch.manual_seed(0)
    model = ToyTransformer(dim=32, depth=2, n_tokens=64)
    cfg = LoRAConfig(rank=4, alpha=8.0, target_substrings=("qkv", "proj"))
    replaced = inject_lora(model, cfg)
    assert replaced, "expected at least one Linear replaced"

    # LoRA params must be trainable, base weights must be frozen.
    for n, p in model.named_parameters():
        if "lora_" in n:
            assert p.requires_grad, f"{n} should be trainable"
        elif "base." in n:
            assert not p.requires_grad, f"{n} should be frozen"

    # A few grad steps on a dummy task must only change LoRA params.
    # NOTE: because LoRA initialises B=0, grads on A are zero at step 0;
    # we do two steps so both A and B can move.
    optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    x = torch.randn(2, 64, 3)
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(2):
        optim.zero_grad()
        loss = model(x).sum()
        loss.backward()
        optim.step()

    any_lora_moved = False
    for n, p in model.named_parameters():
        if "lora_" in n:
            if not torch.equal(p, snapshot[n]):
                any_lora_moved = True
        elif "base." in n:
            assert torch.equal(p, snapshot[n]), f"{n} must remain unchanged"
    assert any_lora_moved, "expected at least one LoRA parameter to update"


def test_trainable_param_summary_counts() -> None:
    model = ToyTransformer(dim=32, depth=2, n_tokens=64)
    cfg = LoRAConfig(rank=4, alpha=8.0, target_substrings=("qkv", "proj"))
    inject_lora(model, cfg)
    summary = trainable_param_summary(model)
    assert summary["lora"] > 0
    assert summary["trainable"] == summary["lora"]
    assert summary["trainable"] < summary["total"]


def test_gradcheck_flow_in_conv_lora() -> None:
    """Backward pass through Conv-LoRA should populate lora_B and conv grads.

    Note: lora_B starts at zero, so grad on lora_A is zero at init (standard
    LoRA property). We seed B slightly before the backward to verify A also
    receives gradient through the full ``A -> conv -> B`` chain.
    """
    torch.manual_seed(0)
    base = nn.Linear(16, 16)
    wrapped = LoRALinear(base, rank=4, alpha=8.0, use_conv=True)
    with torch.no_grad():
        wrapped.lora_B.add_(0.1 * torch.randn_like(wrapped.lora_B))
    x = torch.randn(1, 64, 16)
    y = wrapped(x).sum()
    y.backward()
    assert wrapped.lora_A.grad is not None and wrapped.lora_A.grad.abs().sum() > 0
    assert wrapped.lora_B.grad is not None and wrapped.lora_B.grad.abs().sum() > 0
    assert wrapped.lora_conv.weight.grad is not None
    assert wrapped.lora_conv.weight.grad.abs().sum() > 0
