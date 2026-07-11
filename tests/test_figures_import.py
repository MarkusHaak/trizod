"""Import-smoke for the trizod.figures subpackage.

Guards against import-time breakage (bad relocations, broken imports) in the
manuscript figure generators, which otherwise have no test coverage because
they need large gitignored inputs (release scores, comparison pickle) to run.
Importing is cheap and catches the most common regression from the Phase 6
figure consolidation. matplotlib is an optional (``[figures]``) dependency, so
the plotting generators are guarded with ``importorskip`` — the lightweight
``style``/``chezod`` helpers are always exercised.
"""

import pytest


def test_style_helpers_import():
    from trizod.figures.style import (
        TIER_COLORS,
        TIERS,
        classify_tier,
        load_tier_sets,
        repo_root,
    )

    assert TIERS[0] == "strict"  # most-stringent-first ordering is load-bearing
    assert set(TIER_COLORS) == set(TIERS)
    assert callable(load_tier_sets)
    assert repo_root().is_dir()

    # classify_tier returns the most stringent tier an entry is in.
    tier_sets = {
        "strict": {"1"},
        "moderate": {"1", "2"},
        "tolerant": set(),
        "unfiltered": set(),
    }
    assert classify_tier("1", tier_sets) == "strict"
    assert classify_tier("2", tier_sets) == "moderate"
    assert classify_tier("999", tier_sets) == "unfiltered"


def test_fig2_lacs_generators_import():
    pytest.importorskip("matplotlib")  # figures extra; skip if not installed
    from trizod.figures.fig2_lacs import plot_lacs_effect
    from trizod.figures.fig2_lacs_case_study import build_case_study_figure

    assert callable(plot_lacs_effect)
    assert callable(build_case_study_figure)


def test_chezod_helpers_import():
    from trizod.figures.chezod import (
        classify,
        load_chezod,
        load_trizod,
        summarize,
    )

    assert callable(load_chezod)
    assert callable(load_trizod)
    assert callable(summarize)
    # classify buckets (pearson, mae) pairs into the four agreement categories.
    assert classify(0.95, 0.5) == "agree"
    assert classify(0.95, 1.5) == "offset_shift"
    assert classify(0.5, 0.5) == "low_variance"
    assert classify(0.5, 2.0) == "genuine"
