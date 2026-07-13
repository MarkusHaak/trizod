from pathlib import Path

import pytest

from trizod import paths

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

    Rebuilds when the directory is empty or holds dangling links — e.g. after
    the BMRB source dir has moved — instead of silently skipping.
    """
    if not has_bmrb_data() or not SUBSET_IDS_FILE.exists():
        return
    existing = list(SUBSET_DIR.iterdir()) if SUBSET_DIR.is_dir() else []
    if existing and all(p.exists() for p in existing):
        return  # already set up and valid
    for stale in existing:  # clear dangling/stale links before rebuilding
        stale.unlink()
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    ids = SUBSET_IDS_FILE.read_text().strip().split("\n")
    for bmrb_id in ids:
        src = BMRB_DIR / f"bmr{bmrb_id}"
        dst = SUBSET_DIR / f"bmr{bmrb_id}"
        if src.is_dir() and not dst.exists():
            dst.symlink_to(src.resolve())


# Auto-create symlinks on import
ensure_subset_symlinks()


requires_bmrb_data = pytest.mark.skipif(
    not has_bmrb_data(),
    reason="BMRB data not available (data/raw/bmrb_entries/)",
)
