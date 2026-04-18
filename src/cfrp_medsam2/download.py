"""Dataset download helpers.

The full CFRP / SiC-SiC downloads are several GB. Each helper can either
pull the full archive or a sample subset (``sample=True``) for quick
iteration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Zenodo — Chalmers synthetic CFRP (Friemann et al. 2025)
# DOI: 10.5281/zenodo.15389426
# ---------------------------------------------------------------------------

ZENODO_CFRP_RECORD = "15389426"        # real unlabelled "orthogonal noobed" scan
ZENODO_CFRP_LABELLED_RECORD = "14891845"  # real scan + hand segmentation (layer-to-layer angle interlock)


def fetch_zenodo_record(record_id: str) -> dict:
    r = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    r.raise_for_status()
    return r.json()


def list_zenodo_files(record_id: str) -> list[dict]:
    rec = fetch_zenodo_record(record_id)
    return [
        {
            "key": f["key"],
            "size": f["size"],
            "url": f["links"]["self"],
            "checksum": f["checksum"],
        }
        for f in rec["files"]
    ]


def list_cfrp_files() -> list[dict]:
    return list_zenodo_files(ZENODO_CFRP_RECORD)


def list_cfrp_labelled_files() -> list[dict]:
    return list_zenodo_files(ZENODO_CFRP_LABELLED_RECORD)


def download_file(url: str, out_path: Path, chunk: int = 1 << 20) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=out_path.name
        ) as pbar:
            for block in r.iter_content(chunk):
                f.write(block)
                pbar.update(len(block))
        tmp.rename(out_path)


def download_cfrp(
    out_dir: str | Path,
    *,
    only: Iterable[str] | None = None,
    sample: bool = False,
) -> list[Path]:
    """Download all files from the Chalmers unlabelled CFRP record into ``out_dir``."""
    return _download_zenodo(out_dir, list_cfrp_files(), only=only, sample=sample)


def download_cfrp_labelled(
    out_dir: str | Path,
    *,
    only: Iterable[str] | None = None,
) -> list[Path]:
    """Download the Chalmers *labelled* CFRP record (~750 MB).

    This record (10.5281/zenodo.14891845) contains the real layer-to-layer
    angle-interlock scan plus a **hand segmentation** TIFF with fibre / matrix
    labels — the right resource for supervised training on real CFRP.
    """
    return _download_zenodo(out_dir, list_cfrp_labelled_files(), only=only, sample=False)


def _download_zenodo(
    out_dir: str | Path,
    files: list[dict],
    *,
    only: Iterable[str] | None = None,
    sample: bool = False,
) -> list[Path]:
    out_dir = Path(out_dir)
    downloaded: list[Path] = []
    budget = 500 * 1024 * 1024
    for f in files:
        if only is not None and not any(tok in f["key"] for tok in only):
            continue
        if sample and f["size"] > budget:
            continue
        out_path = out_dir / f["key"]
        download_file(f["url"], out_path)
        downloaded.append(out_path)
        if sample:
            budget -= f["size"]
            if budget <= 0:
                break
    return downloaded


# ---------------------------------------------------------------------------
# 4TU.ResearchData — TU Delft / Leibniz thermoplastic CFRP tape (Boos et al. 2025)
# DOI: 10.4121/3a864c60-3023-45ab-a6c6-f36a23d67f56.v1
# Full record is ~122 GB; the segmented (fibre/matrix/pore) ROIs live in
# XrayCT_Cropped_and_Registered_Reconstructed.zip (~15 GB).
# ---------------------------------------------------------------------------

TUDELFT_CFRP_RECORD = "3a864c60-3023-45ab-a6c6-f36a23d67f56"

# Per-file download URLs resolved once via
#   curl https://data.4tu.nl/v2/articles/<record>/files
# The file UUIDs are stable for a published record version, so we pin them
# here to avoid an API roundtrip on every run.
TUDELFT_CFRP_FILES: dict[str, dict] = {
    "cropped": {
        "name": "XrayCT_Cropped_and_Registered_Reconstructed.zip",
        "size": 15_121_013_685,
        "url": "https://data.4tu.nl/file/3a864c60-3023-45ab-a6c6-f36a23d67f56/8e5e0388-36d8-447e-b3b2-e3c969177537",
    },
    "reconstructed": {
        "name": "XrayCT_Reconstructed_Image_Dataset.zip",
        "size": 106_428_467_165,
        "url": "https://data.4tu.nl/file/3a864c60-3023-45ab-a6c6-f36a23d67f56/6808d5b0-4801-49aa-b28a-76a3f846bdaf",
    },
    "microscopy": {
        "name": "Microscopy_images.zip",
        "size": 1_266_383_830,
        "url": "https://data.4tu.nl/file/3a864c60-3023-45ab-a6c6-f36a23d67f56/e42bb7c5-d18f-422a-befd-80dcd3ce0ff3",
    },
    "readme": {
        "name": "ReadMe.txt",
        "size": 412,
        "url": "https://data.4tu.nl/file/3a864c60-3023-45ab-a6c6-f36a23d67f56/1b95f1a3-574a-4c67-8493-3aac4007347f",
    },
}


def download_tudelft_cfrp(
    out_dir: str | Path,
    *,
    parts: Iterable[str] = ("cropped", "readme"),
) -> list[Path]:
    """Download the segmented ROI subset of the TU Delft / Leibniz CFRP tape record.

    ``parts`` picks which named files to pull; valid keys are the entries of
    :data:`TUDELFT_CFRP_FILES`. The default (``"cropped" + "readme"``) grabs
    the ~15 GB cropped/registered reconstruction that carries the
    fibre/matrix/pore segmentations plus the record's README, which is the
    minimum needed for the Chalmers -> TU Delft cross-dataset eval.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for key in parts:
        if key not in TUDELFT_CFRP_FILES:
            raise KeyError(
                f"unknown TU Delft CFRP file {key!r}; "
                f"pick from {sorted(TUDELFT_CFRP_FILES)}"
            )
        meta = TUDELFT_CFRP_FILES[key]
        out_path = out_dir / meta["name"]
        download_file(meta["url"], out_path)
        downloaded.append(out_path)
    return downloaded


# ---------------------------------------------------------------------------
# Argonne ACDC — Badran et al. SiC-SiC CMC
# https://acdc.alcf.anl.gov/mdf/detail/badran_deeplearning_supplementarymaterial_v1.1
# ---------------------------------------------------------------------------


ACDC_MANIFEST_URL = (
    "https://acdc.alcf.anl.gov/mdf/detail/"
    "badran_deeplearning_supplementarymaterial_v1.1/manifest.json"
)


def download_sic_sic_hint(out_dir: str | Path) -> Path:
    """ACDC doesn't expose a stable public API; we write a README with instructions.

    Programmatic download requires the MDF CLI and (sometimes) Globus creds.
    This function drops a ``README.ACDC.md`` into ``out_dir`` so users can
    follow the manual path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readme = out_dir / "README.ACDC.md"
    readme.write_text(
        "# SiC-SiC CMC XCT (Badran et al., Argonne ACDC)\n\n"
        "Landing page: https://acdc.alcf.anl.gov/mdf/detail/"
        "badran_deeplearning_supplementarymaterial_v1.1/\n\n"
        "This dataset is distributed through the Materials Data Facility (MDF).\n"
        "Install the MDF CLI and authenticate with Globus, then pull the\n"
        "`badran_deeplearning_supplementarymaterial_v1.1` collection into this\n"
        "directory. Convert the volume (TIFF stack + mask) using\n"
        "`cfrp_medsam2.preprocess.ingest_tiff_stack`.\n"
    )
    return readme
