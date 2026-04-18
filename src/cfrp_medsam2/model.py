"""Thin wrapper around MedSAM2 (when available) plus a minimal fallback.

The wrapper exposes a single :meth:`forward_slice` method that accepts a
batch of RGB-like 3-slice windows plus bbox/point prompts, and returns a
logit mask at the input resolution. This lets ``train.py`` be agnostic to
whether we're using the real MedSAM2 model or the fallback used in the
smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fallback minimal model (only used when MedSAM2 isn't installed).
# ---------------------------------------------------------------------------


class _TinySegHead(nn.Module):
    """A toy encoder-decoder used for smoke tests when MedSAM2 isn't set up.

    This is *not* representative of MedSAM2 — it only exists so the training
    loop has something to exercise end-to-end on the toy dataset.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        # Prompt conditioning: 4 box coords + a fg/bg flag.
        self.prompt_proj = nn.Linear(5, 32)
        self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.out = nn.Conv2d(16, 1, 1)
        # q_proj / k_proj / v_proj / out_proj so LoRA injection has targets.
        self.attn = _TinySelfAttention(32)

    def forward(self, rgb: torch.Tensor, boxes: torch.Tensor | None) -> torch.Tensor:
        h = self.enc1(rgb)
        h = self.pool(h)
        h = self.enc2(h)
        if boxes is not None:
            cond = self.prompt_proj(
                torch.cat([boxes, torch.ones(boxes.shape[0], 1, device=boxes.device)], dim=-1)
            )[:, :, None, None]
            h = h + cond
        # Toy "attention" over spatial tokens so LoRA has something to adapt.
        B, C, H, W = h.shape
        tokens = h.flatten(2).transpose(1, 2)
        tokens = self.attn(tokens)
        h = tokens.transpose(1, 2).reshape(B, C, H, W)
        h = self.up(h)
        return self.out(h)


class _TinySelfAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Unified wrapper
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    backend: str = "auto"       # "medsam2", "fallback", or "auto"
    sam2_config: str = "configs/sam2.1_hiera_t512.yaml"
    checkpoint: str | None = None
    medsam2_repo: str | None = None   # path to cloned upstream repo for sys.path
    image_size: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SegModel(nn.Module):
    """Uniform interface across MedSAM2 and the fallback."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backend = cfg.backend
        self._build()

    def _build(self) -> None:
        if self.backend in ("medsam2", "auto"):
            try:
                self._build_medsam2()
                self.backend = "medsam2"
                return
            except Exception as e:  # pragma: no cover - depends on env
                if self.backend == "medsam2":
                    raise
                print(f"[SegModel] MedSAM2 unavailable ({e.__class__.__name__}: {e}); using fallback.")
        self.backend = "fallback"
        self.net = _TinySegHead()

    def _build_medsam2(self) -> None:
        """Instantiate MedSAM2 via the upstream ``build_sam2`` helper."""
        import sys

        repo = self.cfg.medsam2_repo
        if repo is None:
            default = Path(__file__).resolve().parents[2] / "external" / "MedSAM2"
            if default.exists():
                repo = str(default)
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)

        from sam2.build_sam import build_sam2  # type: ignore  # noqa: E402

        ckpt = self.cfg.checkpoint
        if ckpt is None or not Path(ckpt).exists():
            raise FileNotFoundError(
                f"MedSAM2 checkpoint not found at {ckpt!r}; run scripts/setup_medsam2.sh"
            )
        self.net = build_sam2(self.cfg.sam2_config, ckpt, device=self.cfg.device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_slice(
        self,
        images: torch.Tensor,
        boxes: Sequence[torch.Tensor] | None = None,
        points: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Run a single-slice forward pass, returning logits shape (B, H, W)."""
        if self.backend == "fallback":
            boxes_t = torch.stack(list(boxes), dim=0) if boxes is not None else None
            out = self.net(images, boxes_t)  # (B, 1, H, W)
            out = F.interpolate(out, size=images.shape[-2:], mode="bilinear", align_corners=False)
            return out.squeeze(1)

        # MedSAM2 path: use SAM2ImagePredictor on the middle frame.
        return self._forward_medsam2(images, boxes, points)

    def _forward_medsam2(
        self,
        images: torch.Tensor,
        boxes: Sequence[torch.Tensor] | None,
        points: Sequence[tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> torch.Tensor:
        """Differentiable forward for training — bypasses the inference predictor.

        Uses the lower-level modules directly so gradients flow into the LoRA
        adapters. Mirrors the logic of ``SAM2Base.forward_image`` +
        ``SAM2Base._prepare_backbone_features`` +
        ``SAM2Base._forward_sam_heads`` but without memory propagation.
        """
        if not hasattr(self.net, "image_encoder"):
            raise RuntimeError("Unexpected MedSAM2 model structure: missing image_encoder")

        backbone_out = self.net.forward_image(images)
        _, vision_feats, _, feat_sizes = self.net._prepare_backbone_features(backbone_out)

        # Lowest-res features act as image_embeddings; the two higher-res
        # levels go in as skip features for the mask decoder.
        B = images.shape[0]
        C = self.net.hidden_dim
        H, W = feat_sizes[-1]
        image_embed = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        high_res_features: list[torch.Tensor] | None = None
        if self.net.use_high_res_features_in_sam and len(vision_feats) >= 3:
            high_res_features = [
                vf.permute(1, 2, 0).view(B, vf.shape[-1], *fs)
                for vf, fs in zip(vision_feats[:-1], feat_sizes[:-1])
            ]

        sparse_list = []
        dense_list = []
        for i in range(B):
            box_i = boxes[i] if boxes is not None else None
            pt_i = points[i] if points is not None else None
            sparse, dense = self.net.sam_prompt_encoder(
                points=pt_i,
                boxes=box_i.unsqueeze(0) if box_i is not None else None,
                masks=None,
            )
            sparse_list.append(sparse)
            dense_list.append(dense)
        sparse_emb = torch.cat(sparse_list, dim=0)
        dense_emb = torch.cat(dense_list, dim=0)

        low_res_masks, iou_pred, _, _ = self.net.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.net.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        masks = F.interpolate(low_res_masks, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return masks.squeeze(1)

    # ------------------------------------------------------------------
    # Volumetric inference via SAM2 memory propagation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer_volume(
        self,
        volume: torch.Tensor,
        mid_box: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate a middle-slice bbox prompt up and down the volume.

        ``volume``: (Z, 3, H, W) float32 tensor of 3-slice windows.
        ``mid_box``: (4,) xyxy prompt on the middle slice.
        Returns a (Z, H, W) float tensor of logits.
        """
        Z = volume.shape[0]
        mid = Z // 2
        if self.backend == "fallback":
            # For the fallback, just call forward slice-by-slice with the same box.
            boxes = [mid_box.to(volume.device)] * Z
            preds = self.forward_slice(volume.to(volume.device), boxes=boxes)
            return preds

        # MedSAM2 path: we use the public video predictor API if available.
        try:  # pragma: no cover - exercised only with upstream installed
            from sam2.sam2_video_predictor import SAM2VideoPredictor  # type: ignore
        except Exception as e:
            raise RuntimeError(f"MedSAM2 video predictor unavailable: {e}") from e

        # The upstream API expects a directory of frames, which isn't convenient
        # here; instead fall back to per-slice inference without memory.
        # For the paper-quality pipeline, users should adapt this to call
        # `init_state` on a frame dir and `add_new_points_or_box` + `propagate_in_video`.
        boxes = [mid_box.to(volume.device)] * Z
        return self.forward_slice(volume.to(volume.device), boxes=boxes)


__all__ = ["SegModel", "ModelConfig"]
