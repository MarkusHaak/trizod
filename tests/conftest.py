"""Shared test configuration.

Data locations come from :mod:`trizod.paths`, so the data-backed tests follow
``TRIZOD_DATA_ROOT`` (a checkout holding ``data/``; a git worktree has none)
and ``TRIZOD_BMRB_DIR`` (the BMRB snapshot, e.g.
``data/raw/bmrb_entries_2026-11-11``)::

    TRIZOD_DATA_ROOT=<checkout> TRIZOD_BMRB_DIR=<snapshot> \\
        uv run --with pyarrow python -m pytest tests/test_chezod_equality.py

On macOS the scoring CLI forks pandarallel workers from a process whose BLAS
(Accelerate / OpenBLAS / OpenMP) already runs threads, and a worker can then
block forever on an inherited lock. The BLAS libraries read their thread count
once, when they load, so it is pinned to 1 here, before any test imports numpy;
every subprocess a test starts inherits it (:data:`BLAS_SINGLE_THREAD`).
"""

import os
import sys
from pathlib import Path

import pytest

from trizod import paths

#: Environment that keeps BLAS single-threaded (macOS fork deadlock, see above).
BLAS_SINGLE_THREAD = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
if sys.platform == "darwin":
    for _var, _value in BLAS_SINGLE_THREAD.items():
        os.environ.setdefault(_var, _value)


def subprocess_env(**extra) -> dict:
    """``os.environ`` plus :data:`BLAS_SINGLE_THREAD` (on every platform) and
    ``extra``: the environment for a test's scoring subprocess."""
    return {**os.environ, **BLAS_SINGLE_THREAD, **extra}


TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = paths.DATA
BMRB_DIR = paths.RAW_BMRB
SUBSET_DIR = TESTS_DIR / "bmrb_subset"
SUBSET_IDS_FILE = TESTS_DIR / "quick_subset_ids.txt"


def has_bmrb_data():
    """Check if BMRB data files are available."""
    return BMRB_DIR.is_dir() and any(BMRB_DIR.iterdir())


def ensure_subset_symlinks():
    """Create (or repair) symlinks for the test subset.

    Rebuilds whenever the on-disk links don't match the expected id set, hold
    dangling links, or point into another BMRB directory than ``BMRB_DIR``
    (``TRIZOD_DATA_ROOT`` / ``TRIZOD_BMRB_DIR`` changed) -- e.g. after ids are
    added to/removed from quick_subset_ids.txt -- instead of silently skipping.
    Only symlinks are considered, so stray files (a macOS ``.DS_Store``) don't
    force a rebuild.
    """
    if not has_bmrb_data() or not SUBSET_IDS_FILE.exists():
        return
    ids = SUBSET_IDS_FILE.read_text().strip().split("\n")
    expected = {f"bmr{i}" for i in ids if (BMRB_DIR / f"bmr{i}").is_dir()}
    existing = (
        {p.name for p in SUBSET_DIR.iterdir() if p.is_symlink()}
        if SUBSET_DIR.is_dir()
        else set()
    )
    target = BMRB_DIR.resolve()
    if existing == expected and all(
        (SUBSET_DIR / n).exists()
        and Path(os.readlink(SUBSET_DIR / n)) == (target / n).resolve()
        for n in existing
    ):
        return  # already set up and valid
    for name in existing:  # clear stale/dangling/removed links before rebuilding
        (SUBSET_DIR / name).unlink()
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    for name in expected:
        (SUBSET_DIR / name).symlink_to((BMRB_DIR / name).resolve())


# Auto-create symlinks on import
ensure_subset_symlinks()


requires_bmrb_data = pytest.mark.skipif(
    not has_bmrb_data(),
    reason=f"BMRB data not available ({BMRB_DIR}; set TRIZOD_DATA_ROOT or "
    "TRIZOD_BMRB_DIR)",
)
