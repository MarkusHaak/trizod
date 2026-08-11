"""Experimental-method prefilter and post-filter index alignment (``trizod.pipeline``).

Two defects are pinned here:

* ``""`` in an ``exp-method-whitelist`` is a SENTINEL meaning "accept a missing
  subtype". Joined into the ``str.contains`` alternation it became an empty
  alternative that matches every string, so the tolerant/moderate whitelists were
  dead code and only the ``solid`` blacklist did any work.
* ``postfilter_dataframe`` built its offset-rejection mask with a fresh
  ``RangeIndex`` and OR-ed it against index-aligned columns, so any caller that
  passes a filtered subset silently got a wrong ``pass_post``.
"""

import numpy as np
import pandas as pd

from trizod.constants import BACKBONE_ATOMS
from trizod.pipeline import postfilter_dataframe, prefilter_dataframe

METHOD_KEY = ("method (sub-)type", "")

# filter_defaults cells, inlined so this test does not move when the tiers do.
UNFILTERED_WHITELIST, UNFILTERED_BLACKLIST = ["", "."], []
TOLERANT_WHITELIST, TOLERANT_BLACKLIST = ["", "solution", "structures"], ["solid"]
STRICT_WHITELIST, STRICT_BLACKLIST = ["solution", "structures"], ["solid"]


def _method_frame(subtypes):
    """A frame that passes every prefilter criterion except, possibly, the method."""
    n = len(subtypes)
    return pd.DataFrame(
        {
            "exp_method": ["NMR"] * n,
            "exp_method_subtype": subtypes,
            "temperature": [298.0] * n,
            "ionic_strength": [0.1] * n,
            "pH": [7.0] * n,
            "seq": ["A" * 20] * n,
            "total_bbshifts": [80] * n,
            "bbshift_types": [4] * n,
            "bbshift_positions": [18] * n,
        }
    )


def _method_selection(subtypes, whitelist, blacklist):
    _df, _missing, sels_pre, *_rest = prefilter_dataframe(
        _method_frame(subtypes),
        method_whitelist=whitelist,
        method_blacklist=blacklist,
        temperature_range=[-np.inf, np.inf],
        ionic_strength_range=[0.0, np.inf],
        pH_range=[-np.inf, np.inf],
        peptide_length_range=[5, np.inf],
        min_backbone_shift_types=1,
        min_backbone_shift_positions=1,
        min_backbone_shift_fraction=0.0,
        max_noncanonical_fraction=1.0,
        max_x_fraction=1.0,
        keywords=[],
        chemical_denaturants=[],
    )
    return [bool(v) for v in sels_pre[METHOD_KEY]]


def test_empty_whitelist_term_is_a_sentinel_not_a_wildcard():
    subtypes = ["X-RAY DIFFRACTION", "solution", None, "SOLID-STATE"]
    assert _method_selection(subtypes, TOLERANT_WHITELIST, TOLERANT_BLACKLIST) == [
        False,  # uninformative subtype: not solution NMR
        True,
        True,  # missing subtype: what the "" sentinel is for
        False,  # blacklisted
    ]


def test_junk_subtypes_rejected_by_the_tolerant_whitelist():
    subtypes = ["THEORETICAL", "STATE", "1H-15N-HSQC", "Magic angle spinning NMR"]
    assert _method_selection(subtypes, TOLERANT_WHITELIST, TOLERANT_BLACKLIST) == [
        False,
        False,
        False,
        False,
    ]


def test_solution_spellings_still_pass():
    subtypes = [
        "solution",
        "SOLUTION",
        "SOLUTION NMR",
        "Solution state",
        "NMR, 20 STRUCTURES",
    ]
    assert all(_method_selection(subtypes, TOLERANT_WHITELIST, TOLERANT_BLACKLIST))


def test_strict_whitelist_still_rejects_a_missing_subtype():
    # No "" sentinel in the strict cell, so a null subtype must not pass.
    subtypes = ["X-RAY DIFFRACTION", "solution", None, "SOLID-STATE"]
    assert _method_selection(subtypes, STRICT_WHITELIST, STRICT_BLACKLIST) == [
        False,
        True,
        False,
        False,
    ]


def test_unfiltered_whitelist_still_accepts_everything():
    # The unfiltered cell is ["", "."]: the "." is the real catch-all regex, the
    # "" only adds the missing-subtype class.
    subtypes = ["X-RAY DIFFRACTION", "solution", None, "SOLID-STATE"]
    assert all(_method_selection(subtypes, UNFILTERED_WHITELIST, UNFILTERED_BLACKLIST))


def _post_frame(n):
    data = {
        "seq": ["A" * 20] * n,
        "zscores": [np.zeros(20)] * n,
        "bbshift_types_post": [4] * n,
        "bbshift_positions_post": [18] * n,
        "pass_pre": [True] * n,
    }
    for atom_type in BACKBONE_ATOMS:
        data[f"off_{atom_type}"] = [0.0] * n
    return pd.DataFrame(data)


def test_postfilter_uses_the_frame_index_not_positions():
    # A caller may hand postfilter_dataframe a filtered subset, whose index is
    # not a RangeIndex; every row here passes every post-filter criterion.
    subset = _post_frame(6).loc[[1, 2, 3, 5]].copy()
    postfilter_dataframe(subset, 1, 1, 0.0, False, ["zscores"])
    assert subset["pass_post"].tolist() == [True, True, True, True]


def test_postfilter_still_rejects_large_offsets_in_a_subset():
    df = _post_frame(6)
    df.loc[3, "off_CA"] = np.nan  # offset rejected during scoring
    subset = df.loc[[1, 2, 3, 5]].copy()
    postfilter_dataframe(subset, 1, 1, 0.0, False, ["zscores"])
    assert subset["pass_post"].tolist() == [True, True, False, True]
