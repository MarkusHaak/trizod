import os
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BMRB_DIR = os.path.join(DATA_DIR, "bmrb_entries")


def has_bmrb_data():
    """Check if BMRB data files are available."""
    return os.path.isdir(BMRB_DIR) and len(os.listdir(BMRB_DIR)) > 0


requires_bmrb_data = pytest.mark.skipif(
    not has_bmrb_data(),
    reason="BMRB data not available (data/bmrb_entries/)",
)
