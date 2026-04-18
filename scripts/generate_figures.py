"""Generate the qualitative figures from saved ablation predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cfrp_medsam2.viz import overlay_slice


REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    data = np.load(REPO / "logs" / "ablation_preds.npz")
    imgs = data["imgs"]
    gts = data["gts"]
    zmid = imgs.shape[0] // 2

    regimes = ["zero_shot", "lora", "conv_lora", "full_ft"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()

    axes[0].imshow(imgs[zmid], cmap="gray")
    axes[0].set_title(f"image (z={zmid})")
    axes[0].axis("off")

    axes[1].imshow(overlay_slice(imgs[zmid], gts[zmid]))
    axes[1].set_title("GT fibre mask")
    axes[1].axis("off")

    for i, r in enumerate(regimes):
        if r not in data.files:
            continue
        pred = data[r][zmid]
        axes[i + 2].imshow(overlay_slice(imgs[zmid], pred))
        axes[i + 2].set_title(f"{r}")
        axes[i + 2].axis("off")

    plt.tight_layout()
    out = REPO / "logs" / "qualitative_midslice.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)

    # Fibre continuity bar chart
    import json

    with open(REPO / "logs" / "ablation.json") as f:
        rows = json.load(f)
    names = [r["regime"] for r in rows]
    dice = [r["dice_3d"] for r in rows]
    cont = [r["fibre_continuity"] for r in rows]
    trainp = [r["trainable_pct"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(names, dice, color=["tab:gray", "tab:orange", "tab:red", "tab:blue"])
    axes[0].set_ylabel("3D Dice")
    axes[0].set_title("In-domain Dice")
    axes[0].set_ylim(0, 1)

    ax2 = axes[1]
    ax2.bar(names, cont, color=["tab:gray", "tab:orange", "tab:red", "tab:blue"])
    ax2.set_ylabel("fibre continuity")
    ax2.set_title("Fibre continuity (higher = less fragmentation)")
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    out = REPO / "logs" / "ablation_bars.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
