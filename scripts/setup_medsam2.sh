#!/usr/bin/env bash
# Clone bowang-lab/MedSAM2, install it editable, and fetch a checkpoint.
# Checkpoint size (sam2.1_hiera_tiny.pt) is ~155 MB.
# MedSAM2 fine-tuned weights are larger; we try the tiny SAM2.1 backbone first.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="${HERE}/external"
CKPT="${HERE}/checkpoints"
mkdir -p "${EXT}" "${CKPT}"

if [ ! -d "${EXT}/MedSAM2" ]; then
    git clone --depth 1 https://github.com/bowang-lab/MedSAM2.git "${EXT}/MedSAM2"
fi

# Editable install so we can import sam2 / medsam2 utilities
pip install -e "${EXT}/MedSAM2"[dev] || pip install -e "${EXT}/MedSAM2"

# SAM2.1 tiny backbone (always needed for instantiation even if we load MedSAM2 weights)
if [ ! -f "${CKPT}/sam2.1_hiera_tiny.pt" ]; then
    wget -q --show-progress -O "${CKPT}/sam2.1_hiera_tiny.pt" \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
fi

# MedSAM2 public weights (~155 MB). Falls through to plain SAM2.1 if the URL moves.
if [ ! -f "${CKPT}/MedSAM2_latest.pt" ]; then
    wget -q --show-progress -O "${CKPT}/MedSAM2_latest.pt" \
        https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt || true
fi

echo "Setup complete. Checkpoints in ${CKPT}:"
ls -lh "${CKPT}"
