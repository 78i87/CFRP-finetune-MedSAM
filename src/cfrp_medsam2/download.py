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
