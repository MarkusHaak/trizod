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

    Rebuilds whenever the on-disk links don't match the expected id set or hold
    dangling links — e.g. after ids are added to/removed from
    quick_subset_ids.txt, or the BMRB source dir has moved — instead of
    silently skipping. Only symlinks are considered, so stray files (a macOS
    ``.DS_Store``) don't force a rebuild.
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
    if existing == expected and all((SUBSET_DIR / n).exists() for n in existing):
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
    reason="BMRB data not available (data/raw/bmrb_entries/)",
)
