"""Generate the real-CFRP qualitative figure and ablation bar chart."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cfrp_medsam2.viz import overlay_slice


REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    data = np.load(REPO / "logs" / "ablation_real_examples.npz")
    regimes = ["zero_shot", "lora", "conv_lora", "full_ft"]

    # Pick example index 1 (arbitrary — any works).
    idx = 1
    img0 = data[f"zero_shot_image_{idx}"][1]  # channel 1 = centre slice
    gt0 = data[f"zero_shot_mask_{idx}"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].imshow(img0, cmap="gray")
    axes[0].set_title("image (centre slice)")
    axes[0].axis("off")
    axes[1].imshow(overlay_slice((img0 * 255).astype(np.uint8), gt0))
    axes[1].set_title("GT yarn")
    axes[1].axis("off")
    for i, r in enumerate(regimes):
        pred = data[f"{r}_pred_{idx}"]
        axes[i + 2].imshow(overlay_slice((img0 * 255).astype(np.uint8), pred))
        axes[i + 2].set_title(r)
        axes[i + 2].axis("off")
    plt.tight_layout()
    out = REPO / "logs" / "qualitative_real.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)

    with open(REPO / "logs" / "ablation_real.json") as f:
        rows = json.load(f)
    names = [r["regime"] for r in rows]
    means = [r["mean_dice"] for r in rows]
    stds = [r["std_dice"] for r in rows]
    medians = [r["median_dice"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["tab:gray", "tab:orange", "tab:red", "tab:blue"]
    axes[0].bar(names, means, yerr=stds, capsize=6, color=colors)
    axes[0].set_ylabel("Per-yarn Dice (mean ± 1σ)")
    axes[0].set_title("Real CFRP (hand-labelled)\nZenodo 14891845")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(names, medians, color=colors)
    axes[1].set_ylabel("Per-yarn Dice (median)")
    axes[1].set_title("Median is a better summary (skewed Dice distribution)")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = REPO / "logs" / "ablation_real_bars.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
