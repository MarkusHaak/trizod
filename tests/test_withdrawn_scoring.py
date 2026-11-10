"""Withdrawn BMRB entries are never scored (audit I-06).

The dataset build already drops them from every pool, but a scored entry still
reaches the shipped ``scores.json`` files and the Parquet as an excluded row.
Skipping them where the scoring run enumerates its input files keeps a
withdrawn deposition out of every artefact of the release.
"""

from trizod.dataset.withdrawn import load_withdrawn
from trizod.pipeline import find_bmrb_files


def _touch_entries(root, ids):
    for entry_id in ids:
        (root / f"bmr{entry_id}_3.str").write_text("")


def test_withdrawn_entries_are_not_scored(tmp_path):
    withdrawn = sorted(load_withdrawn())
    assert "36170" in withdrawn
    _touch_entries(tmp_path, ["4205", "15000", *withdrawn])

    found = find_bmrb_files(tmp_path)

    assert set(found) == {"4205", "15000"}


def test_withdrawn_entries_are_skipped_whatever_the_pattern_captures(tmp_path):
    # a custom --BMRB-file-pattern may capture more than the bare entry ID
    _touch_entries(tmp_path, ["4205", "36170"])

    found = find_bmrb_files(tmp_path, pattern=r"(bmr\d+)_3\.str")

    assert set(found) == {"bmr4205"}


def test_withdrawn_entries_can_be_scored_on_request(tmp_path):
    _touch_entries(tmp_path, ["4205", "36170"])

    found = find_bmrb_files(tmp_path, exclude_withdrawn=False)

    assert set(found) == {"4205", "36170"}
