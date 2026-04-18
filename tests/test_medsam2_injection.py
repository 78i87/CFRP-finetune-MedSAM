"""Integration test: LoRA injection into the real MedSAM2 / SAM2.1 model.

Skipped when either the MedSAM2 repo or its checkpoint is missing so the
unit-test suite still runs on CI without downloads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1] / "external" / "MedSAM2"
CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "sam2.1_hiera_tiny.pt"


pytestmark = pytest.mark.skipif(
    not REPO.exists() or not CKPT.exists(),
    reason="MedSAM2 not installed; run scripts/setup_medsam2.sh",
)


def _build_model(device: str):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from sam2.build_sam import build_sam2  # type: ignore

    return build_sam2("configs/sam2.1_hiera_t512.yaml", str(CKPT), device=device)


def test_lora_injection_into_hiera_and_memory_attention() -> None:
    from cfrp_medsam2.lora import LoRAConfig, inject_lora, trainable_param_summary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(device)

    cfg = LoRAConfig(
        rank=4,
        alpha=8.0,
        target_substrings=("qkv", "q_proj", "k_proj", "v_proj", "out_proj", "proj"),
        exclude_substrings=(
            "mask_decoder.iou_prediction_head",
            "mlp",  # don't touch MLPs
            "obj_ptr",  # don't touch object-ptr projections
        ),
        include_memory_attention=True,
    )
    replaced = inject_lora(model, cfg)
    assert len(replaced) > 40, f"expected many wraps, got {len(replaced)}"
    assert any("image_encoder" in n for n in replaced), "image encoder not adapted"
    assert any("memory_attention" in n for n in replaced), "memory attn not adapted"

    summary = trainable_param_summary(model)
    pct = 100.0 * summary["trainable"] / summary["total"]
    assert summary["lora"] > 0
    assert pct < 2.0, f"PEFT should be <2% params, got {pct:.2f}%"


def test_conv_lora_injection_runs_forward() -> None:
    """Sanity: after Conv-LoRA injection, image_encoder forward still runs."""
    from cfrp_medsam2.lora import LoRAConfig, inject_lora

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(device)

    cfg = LoRAConfig(
        rank=4,
        alpha=8.0,
        target_substrings=("qkv", "proj"),
        exclude_substrings=("mlp", "obj_ptr", "mask_decoder"),
        include_memory_attention=False,
        use_conv=True,
    )
    inject_lora(model, cfg)

    # Run a forward through the trunk with a dummy image.
    x = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        feats = model.image_encoder(x)
    assert isinstance(feats, dict) or hasattr(feats, "shape")
