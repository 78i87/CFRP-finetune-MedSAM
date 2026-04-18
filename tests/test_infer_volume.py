"""Smoke test for :meth:`SegModel.infer_volume`.

On the fallback backend (always available in CI) the method should still
return one slice-wise logit tensor per input slice. The MedSAM2 memory
propagation path is exercised separately in an integration script because
it requires the ~150 MB SAM2.1 checkpoint and a GPU.
"""

from __future__ import annotations

import torch

from cfrp_medsam2.model import ModelConfig, SegModel


def test_infer_volume_fallback_runs_end_to_end():
    model = SegModel(ModelConfig(backend="fallback", device="cpu"))
    Z, H, W = 8, 64, 64
    volume = torch.randn(Z, 3, H, W)
    box = torch.tensor([8, 8, 55, 55], dtype=torch.float32)
    logits = model.infer_volume(volume, box)
    assert logits.shape == (Z, H, W), logits.shape
    assert torch.isfinite(logits).all()


def test_infer_volume_medsam2_api_contract():
    """The medsam2 path should accept `frames_dir` and `offload_video_to_cpu`.

    We don't run the real propagation here (needs the checkpoint); we just
    assert the signature is still what Tier B's cross-eval script expects.
    """
    import inspect

    sig = inspect.signature(SegModel.infer_volume)
    params = sig.parameters
    assert "volume" in params
    assert "mid_box" in params
    assert "frames_dir" in params
    assert "offload_video_to_cpu" in params
    # frames_dir must be keyword-only so callers can pass just the box positionally.
    assert params["frames_dir"].kind is inspect.Parameter.KEYWORD_ONLY
