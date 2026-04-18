#!/usr/bin/env bash
# Clone bowang-lab/MedSAM2, install it editable, and fetch checkpoints.
#
# Downloads:
#   sam2.1_hiera_tiny.pt        ~155 MB  (tiny backbone, needed for the fallback tests)
#   MedSAM2_latest.pt           ~155 MB  (medical-domain fine-tune of the tiny backbone)
#   sam2.1_hiera_base_plus.pt   ~330 MB  (upstream Meta base+ backbone, Tier A target)
#
# Also installs our `configs/sam2.1_hiera_b+_512.yaml` into the cloned
# MedSAM2's `sam2/configs/` directory so `build_sam2` can find it via hydra.
#
# Silent-404 guard: every `wget` is followed by a size check; a truncated
# or zero-byte file is removed so the next run retries cleanly.

set -euo pipefail

# Guard: must run inside /venv/main (python 3.14). The system python3.12 lives
# on the 16 GB overlay and will fill / if we pip install heavy deps there.
# `/venv/main` is a conda env, so check CONDA_PREFIX (VIRTUAL_ENV isn't set).
PY_PREFIX="$(python -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
if [ "${PY_PREFIX}" != "/venv/main" ]; then
    echo "[setup] ERROR: python prefix is '${PY_PREFIX}', expected '/venv/main'."
    echo "[setup]        run 'source /venv/main/bin/activate' first."
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="${HERE}/external"
CKPT="${HERE}/checkpoints"
CFG_SRC="${HERE}/configs"
mkdir -p "${EXT}" "${CKPT}"

# Minimum plausible size for any checkpoint we expect (MB).
MIN_CKPT_MB=50

_fetch() {
    # _fetch <url> <dest-path> <min-size-mb>
    local url="$1" dest="$2" min_mb="$3"
    if [ -f "${dest}" ]; then
        local existing_mb
        existing_mb=$(du -m "${dest}" | cut -f1)
        if [ "${existing_mb}" -ge "${min_mb}" ]; then
            return 0
        fi
        echo "[setup] ${dest} exists but is only ${existing_mb}MB (<${min_mb}MB); refetching."
        rm -f "${dest}"
    fi
    wget -q --show-progress -O "${dest}" "${url}" || {
        echo "[setup] WARN: download failed for ${url}"
        rm -f "${dest}"
        return 1
    }
    local got_mb
    got_mb=$(du -m "${dest}" | cut -f1)
    if [ "${got_mb}" -lt "${min_mb}" ]; then
        echo "[setup] WARN: ${dest} is only ${got_mb}MB (<${min_mb}MB); removing."
        rm -f "${dest}"
        return 1
    fi
}

if [ ! -d "${EXT}/MedSAM2" ]; then
    git clone --depth 1 https://github.com/bowang-lab/MedSAM2.git "${EXT}/MedSAM2"
fi

# Install our base+ config into MedSAM2's hydra search path (sam2/configs/).
# Safe to re-run; `cp -f` overwrites.
cp -f "${CFG_SRC}/sam2.1_hiera_b+_512.yaml" "${EXT}/MedSAM2/sam2/configs/sam2.1_hiera_b+_512.yaml"

# Editable install of upstream so we can import sam2 / medsam2 utilities.
# Skip if already importable - reinstalls re-pull the ~500 MB torch/nvidia
# pyproject requirement set and can exhaust the overlay if anything slips.
if ! python -c "import sam2" >/dev/null 2>&1; then
    pip install -e "${EXT}/MedSAM2"[dev] || pip install -e "${EXT}/MedSAM2"
else
    echo "[setup] sam2 already importable; skipping pip install -e MedSAM2."
fi

# SAM2.1 tiny backbone (always needed for the fallback path + existing tests).
_fetch \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" \
    "${CKPT}/sam2.1_hiera_tiny.pt" \
    "${MIN_CKPT_MB}"

# MedSAM2 fine-tuned weights on the tiny backbone (~155 MB).
_fetch \
    "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt" \
    "${CKPT}/MedSAM2_latest.pt" \
    "${MIN_CKPT_MB}" \
    || echo "[setup] MedSAM2_latest.pt not fetched; the tiny SAM2.1 backbone will be used instead."

# Tier A: upstream SAM2.1 base+ backbone (~330 MB). No MedSAM2 base+ fine-tune
# exists on HuggingFace (all wanglab/MedSAM2 weights are built on tiny), so we
# adapt the raw SAM2.1 base+ directly via LoRA.
_fetch \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt" \
    "${CKPT}/sam2.1_hiera_base_plus.pt" \
    "300"

echo "Setup complete. Checkpoints in ${CKPT}:"
ls -lh "${CKPT}"
