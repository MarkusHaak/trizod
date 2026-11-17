"""Canonical filesystem paths for the TriZOD repository.

Single source of truth so scripts, tests, and the pipeline never hard-code data
locations. The layout follows the Cookiecutter Data Science convention:

    data/
      raw/        immutable original source inputs (read-only; never edited)
      external/   third-party reference / benchmark datasets
      interim/    regenerable intermediate build artifacts (safe to delete)
      processed/  final canonical dataset(s)
      release/    frozen, versioned Zenodo deposit bundles

Ephemeral, machine-local performance caches live in a separate top-level
``tmp/`` (never a source of truth). Everything under ``data/`` and ``tmp/`` is
gitignored; only ``raw/`` + ``external/`` are irreplaceable, everything else is
regenerable from code + raw + external.

``layout(root)`` builds the whole tree for a given root (used by the
dataset-build chain, which supports an overridable ``--root``); the module-level
constants below are that layout resolved for scripts/tests that just want a
default.

Two environment variables redirect the module-level data constants, read once
at import (a git worktree has no ``data/`` of its own, and a rebuild keeps its
snapshot under its own root):

* ``TRIZOD_DATA_ROOT`` -- a checkout (or a rebuild's ``root/``) holding
  ``data/``. Every ``data/...`` constant (``DATA``, ``RAW``, ``RAW_BMRB``,
  ``EXTERNAL``, ``INTERIM*``, ``PROCESSED*``, ``RELEASE``, ...) resolves under
  it. ``ROOT`` stays the code checkout (scripts, docs, the package), and the
  caches ``TMP`` / ``PKL_DIR`` stay in its ``tmp/``.
* ``TRIZOD_BMRB_DIR`` -- the BMRB snapshot directory (``bmr<ID>/``), e.g.
  ``data/raw/bmrb_entries_2026-11-11``; overrides ``RAW_BMRB`` alone.

Unset (or empty), both fall back to the auto-detected repository root.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

#: Environment variable naming the data root (a directory holding ``data/``).
DATA_ROOT_ENV = "TRIZOD_DATA_ROOT"
#: Environment variable naming the BMRB snapshot directory.
BMRB_DIR_ENV = "TRIZOD_BMRB_DIR"


def repo_root() -> Path:
    """Repository root (this file is ``trizod/paths.py`` -> ``parents[1]``)."""
    return Path(__file__).resolve().parents[1]


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else None


def data_root() -> Path:
    """``$TRIZOD_DATA_ROOT`` when set, else the repository root."""
    return _env_path(DATA_ROOT_ENV) or repo_root()


def bmrb_dir(root=None) -> Path:
    """The BMRB snapshot to read: ``$TRIZOD_BMRB_DIR`` when set, else
    ``data/raw/bmrb_entries`` under ``root`` (default: :func:`data_root`)."""
    override = _env_path(BMRB_DIR_ENV)
    if override is not None:
        return override
    return layout(root if root is not None else data_root()).raw_bmrb


def layout(root=None) -> SimpleNamespace:
    """Resolve the full data/cache layout for ``root`` (default: repo root)."""
    root = Path(root).resolve() if root is not None else repo_root()
    data = root / "data"
    return SimpleNamespace(
        root=root,
        data=data,
        # tiers
        raw=data / "raw",
        external=data / "external",
        interim=data / "interim",
        processed=data / "processed",
        release=data / "release",
        # raw (immutable source)
        raw_bmrb=data / "raw" / "bmrb_entries",
        # external (third-party references / benchmarks)
        ext_chezod=data / "external" / "chezod",
        ext_chezod_1325=data / "external" / "chezod" / "protein_nmr_1325",
        ext_chezod117=data / "external" / "chezod117",
        ext_bmrb_lacs=data / "external" / "bmrb_lacs",
        ext_panav=data / "external" / "panav_offsets.json",
        # interim (regenerable intermediates)
        interim_scored=data / "interim" / "scored",
        interim_build=data / "interim" / "build",
        interim_baseline=data / "interim" / "baseline",
        interim_filter_impact=data / "interim" / "filter_impact",
        interim_chezod_verification=data / "interim" / "chezod_verification",
        lacs_comparison=data / "interim" / "lacs_comparison.npz",
        # processed (final canonical dataset)
        processed_parquet=data / "processed" / "trizod_dataset.parquet",
        processed_deploy=data / "processed" / "deploy",
        # ephemeral cache (separate from data/)
        tmp=root / "tmp",
        pkl_dir=root / "tmp" / "bmrb_entries",
    )


_L = layout(data_root())
_CODE = layout()

# --- module-level constants: code at the repo root, data at data_root() ----
ROOT = _CODE.root
DATA = _L.data
RAW = _L.raw
EXTERNAL = _L.external
INTERIM = _L.interim
PROCESSED = _L.processed
RELEASE = _L.release

RAW_BMRB = bmrb_dir()

EXT_CHEZOD = _L.ext_chezod
EXT_CHEZOD_1325 = _L.ext_chezod_1325
EXT_CHEZOD117 = _L.ext_chezod117
EXT_BMRB_LACS = _L.ext_bmrb_lacs
EXT_PANAV = _L.ext_panav

INTERIM_SCORED = _L.interim_scored
INTERIM_BUILD = _L.interim_build
INTERIM_BASELINE = _L.interim_baseline
INTERIM_FILTER_IMPACT = _L.interim_filter_impact
INTERIM_CHEZOD_VERIFICATION = _L.interim_chezod_verification
LACS_COMPARISON = _L.lacs_comparison

PROCESSED_PARQUET = _L.processed_parquet
PROCESSED_DEPLOY = _L.processed_deploy

TMP = _CODE.tmp
PKL_DIR = _CODE.pkl_dir
