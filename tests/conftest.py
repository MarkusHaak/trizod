import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BMRB_DIR = os.path.join(DATA_DIR, "bmrb_entries")
SUBSET_DIR = os.path.join(TESTS_DIR, "bmrb_subset")
SUBSET_IDS_FILE = os.path.join(TESTS_DIR, "quick_subset_ids.txt")


def has_bmrb_data():
    """Check if BMRB data files are available."""
    return os.path.isdir(BMRB_DIR) and len(os.listdir(BMRB_DIR)) > 0


def ensure_subset_symlinks():
    """Create symlinks for the test subset if they don't exist."""
    if not has_bmrb_data() or not os.path.exists(SUBSET_IDS_FILE):
        return
    if os.path.isdir(SUBSET_DIR) and len(os.listdir(SUBSET_DIR)) > 0:
        return  # already set up
    os.makedirs(SUBSET_DIR, exist_ok=True)
    with open(SUBSET_IDS_FILE) as f:
        ids = f.read().strip().split("\n")
    for bmrb_id in ids:
        src = os.path.join(BMRB_DIR, f"bmr{bmrb_id}")
        dst = os.path.join(SUBSET_DIR, f"bmr{bmrb_id}")
        if os.path.isdir(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)


# Auto-create symlinks on import
ensure_subset_symlinks()


requires_bmrb_data = pytest.mark.skipif(
    not has_bmrb_data(),
    reason="BMRB data not available (data/bmrb_entries/)",
)
