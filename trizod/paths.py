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
constants below are that layout resolved at the auto-detected repo root, for
scripts/tests that just want a default.
"""

from pathlib import Path
from types import SimpleNamespace


def repo_root() -> Path:
    """Repository root (this file is ``trizod/paths.py`` -> ``parents[1]``)."""
    return Path(__file__).resolve().parents[1]


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


_L = layout()

# --- module-level constants at the auto-detected repo root ----------------
ROOT = _L.root
DATA = _L.data
RAW = _L.raw
EXTERNAL = _L.external
INTERIM = _L.interim
PROCESSED = _L.processed
RELEASE = _L.release

RAW_BMRB = _L.raw_bmrb

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

TMP = _L.tmp
PKL_DIR = _L.pkl_dir
