"""Entry-level parsing regressions for BmrbEntry."""

import pytest

from tests.conftest import BMRB_DIR, requires_bmrb_data


@requires_bmrb_data
def test_entry_details_uses_details_tag_not_title():
    """Regression for #12: BmrbEntry.details must be sourced from _Entry.Details,
    not _Entry.Title. Otherwise the entry-level details text is never scanned by
    the keyword blacklist (denatur/unfold/misfold/interacti/bound).

    bmr26999 has:
      _Entry.Title   = "Adenylate kinase in Apo form"
      _Entry.Details = "Adenylate kinase in Apo form plus bound to Ap5A, ATP, and AMP"
    so 'bound' appears only in the Details tag.
    """
    import trizod.bmrb.bmrb as bmrb

    entry_dir = BMRB_DIR / "bmr26999"
    if not entry_dir.exists():
        pytest.skip("BMRB 26999 not available in BMRB_DIR")

    entry = bmrb.BmrbEntry("26999", entry_dir)

    assert entry.details is not None
    # 'bound' is present in _Entry.Details but not in _Entry.Title.
    assert "bound" in entry.details.lower()
    assert entry.details.strip() != entry.title.strip()
