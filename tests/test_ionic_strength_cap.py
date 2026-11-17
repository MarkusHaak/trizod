"""The strict tier caps the ionic strength at 0.5 M (was 3 M), 2026-11.

A convention, not a correction: POTENCI has no ionic-strength term beyond a
Debye-Hueckel screening of side-chain charges, whose leverage is small and
saturates, and same-sequence replicates at 0.5-5 M show no G-score difference
from low-salt ones. The cap keeps the highest-confidence tier in the dilute
buffer regime where that treatment is physically reasonable. Moderate (5 M)
and tolerant (7 M) are unchanged, and deposited values are used as they are.
"""

import numpy as np
import pytest

from tests._stubs import condition_frame, prefilter_tier
from trizod.trizod import filter_defaults


def test_ionic_strength_windows_per_tier():
    assert filter_defaults["ionic-strength-range"].tolist() == [
        [0.0, np.inf],
        [0.0, 7.0],
        [0.0, 5.0],
        [0.0, 0.5],
    ]


IONIC = [0.0, 0.15, 0.5, 0.51, 1.0, 3.0]


def test_strict_keeps_dilute_buffers_and_drops_more_than_half_molar():
    out = prefilter_tier(condition_frame(ionic_strength=IONIC), "strict")
    assert out["pass_pre"].tolist() == [True, True, True, False, False, False]


@pytest.mark.parametrize("tier", ["unfiltered", "tolerant", "moderate"])
def test_looser_tiers_keep_up_to_three_molar(tier):
    out = prefilter_tier(condition_frame(ionic_strength=IONIC), tier)
    assert out["pass_pre"].all()
