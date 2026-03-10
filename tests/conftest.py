from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
BMRB_DIR = DATA_DIR / "bmrb_entries"
SUBSET_DIR = TESTS_DIR / "bmrb_subset"
SUBSET_IDS_FILE = TESTS_DIR / "quick_subset_ids.txt"


def has_bmrb_data():
    """Check if BMRB data files are available."""
    return BMRB_DIR.is_dir() and any(BMRB_DIR.iterdir())


def ensure_subset_symlinks():
    """Create symlinks for the test subset if they don't exist."""
    if not has_bmrb_data() or not SUBSET_IDS_FILE.exists():
        return
    if SUBSET_DIR.is_dir() and any(SUBSET_DIR.iterdir()):
        return  # already set up
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
    reason="BMRB data not available (data/bmrb_entries/)",
)
