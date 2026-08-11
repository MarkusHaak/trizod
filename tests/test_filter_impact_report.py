"""``scripts/filter_impact_report.py`` must reproduce the shipped tier defaults.

The report calls the pipeline directly with its own argument list, so any filter
it forgets to pass silently falls back to a function default instead of the tier
policy — and the report then prints numbers that no shipped tier produces. That
happened twice: ``keyword_search_scope`` defaulted to ``"all"`` (paper-topic
fields searched, unlike every tier), and neither ``physical_state_blacklist``
nor ``method_fallback`` was passed at all, so no physical-state row appeared in
the report and every tier ran without undeclared-method recovery or the solid
veto.

These tests pin the derivation, not the restatement: every ``filter_defaults``
policy must reach the pipeline call, and a new pipeline filter must fail here.
"""

import importlib.util
import inspect
import sys

import numpy as np
import pandas as pd
import pytest

from trizod import paths
from trizod.pipeline import prefilter_dataframe
from trizod.trizod import create_peptide_dataframe, filter_defaults

SCRIPT = paths.ROOT / "scripts" / "filter_impact_report.py"


def _load_report():
    spec = importlib.util.spec_from_file_location("filter_impact_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report = _load_report()

#: ``create_peptide_dataframe`` parameters that are NOT tier policy: the entries
#: themselves, the two output-shape flags, the progress bar, and the two lists
#: the report unions across tiers.
NON_TIER_FRAME_PARAMS = {
    "bmrb_entries",
    "chemical_denaturants",
    "keywords",
    "include_shifts",
    "no_shift_averaging",
    "progress",
}


def _params(func):
    return set(inspect.signature(func).parameters) - {"df"}


# --------------------------------------------------------------------------- #
# the maps must be complete
# --------------------------------------------------------------------------- #


def test_prefilter_args_cover_every_prefilter_parameter():
    """A filter added to ``prefilter_dataframe`` must be wired up here too."""
    assert set(report.PREFILTER_ARGS.values()) == _params(prefilter_dataframe), (
        "PREFILTER_ARGS is out of sync with prefilter_dataframe(); the report "
        "would silently use that parameter's function default instead of the "
        "tier's filter_defaults value."
    )


def test_frame_args_cover_every_tier_driven_frame_parameter():
    tier_driven = _params(create_peptide_dataframe) - NON_TIER_FRAME_PARAMS
    assert set(report.FRAME_ARGS.values()) == tier_driven


@pytest.mark.parametrize("key", sorted({**report.PREFILTER_ARGS, **report.FRAME_ARGS}))
def test_mapped_keys_are_real_filter_defaults_columns(key):
    assert key in filter_defaults.columns


# --------------------------------------------------------------------------- #
# every tier resolves the shipped policy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tier", report.TIERS)
def test_every_tier_resolves_its_shipped_defaults(tier):
    kwargs = report.tier_prefilter_kwargs(tier)
    for key, arg in report.PREFILTER_ARGS.items():
        expected = filter_defaults.loc[tier, key]
        if key == "peptide-length-range":
            continue  # the CLI's inf-append is asserted separately
        if isinstance(expected, list):
            assert list(kwargs[arg]) == expected, key
        else:
            assert kwargs[arg] == expected, key


@pytest.mark.parametrize("tier", report.TIERS)
def test_peptide_length_range_gains_the_cli_upper_bound(tier):
    lo, hi = report.tier_prefilter_kwargs(tier)["peptide_length_range"]
    assert lo == filter_defaults.loc[tier, "peptide-length-range"][0]
    assert hi == np.inf


@pytest.mark.parametrize("tier", report.TIERS)
def test_keyword_search_scope_is_the_shipped_sample_scope(tier):
    """Regression pin: the report used the ``"all"`` function default."""
    scope = report.tier_frame_kwargs(tier)["keyword_search_scope"]
    assert scope == filter_defaults.loc[tier, "keyword-search-scope"] == "sample"


@pytest.mark.parametrize("tier", report.TIERS)
def test_physical_state_and_method_fallback_are_passed(tier):
    """Regression pin: neither reached ``prefilter_dataframe``."""
    kwargs = report.tier_prefilter_kwargs(tier)
    assert kwargs["method_fallback"] == filter_defaults.loc[tier, "method-fallback"]
    assert list(kwargs["physical_state_blacklist"]) == list(
        filter_defaults.loc[tier, "physical-state-blacklist"]
    )


def test_resolving_kwargs_never_mutates_filter_defaults():
    """``peptide_length_range`` is appended to — it must be a copy."""
    before = [list(v) for v in filter_defaults["peptide-length-range"]]
    for tier in report.TIERS:
        report.tier_prefilter_kwargs(tier)
        report.tier_prefilter_kwargs(tier)
    assert [list(v) for v in filter_defaults["peptide-length-range"]] == before


def test_tiers_reading_conditions_differently_get_different_frames():
    """``strict`` reads pH/T/ionic strength differently from ``tolerant``, so a
    single shared DataFrame cannot serve both."""
    assert report.frame_key("strict") != report.frame_key("tolerant")
    assert report.frame_key("unfiltered") == report.frame_key("tolerant")
    assert len({report.frame_key(t) for t in report.TIERS}) == 3


def test_union_of_tier_lists_keeps_every_tier_column():
    keywords = report._union(report.TIERS, "keywords-blacklist")
    for tier in report.TIERS:
        assert set(filter_defaults.loc[tier, "keywords-blacklist"]) <= set(keywords)
    assert len(keywords) == len(set(keywords))


# --------------------------------------------------------------------------- #
# the filters actually fire in the report
# --------------------------------------------------------------------------- #


def _frame(tier, rows):
    """Minimal peptide frame that passes every ``tier`` filter by default.

    Built by hand (not via ``fill_row_data``, which needs parsed BMRB entries)
    but with the same columns, including ``physical_state`` beside
    ``denaturant_evidence`` — ``prefilter_dataframe`` raises if one is present
    without the other.
    """
    defaults = filter_defaults.loc[tier]
    base = {
        "entryID": "12345",
        "stID": "1",
        "entity_assemID": "1",
        "entityID": "1",
        "exp_method": "NMR",
        "exp_method_subtype": "solution structures",
        "temperature": 298.0,
        "ionic_strength": 0.15,
        "pH": 7.0,
        "seq": "A" * 20,
        "total_bbshifts": 100,
        "bbshift_types": 7,
        "bbshift_positions": 20,
        "paramagnetic": False,
        "physical_state": "native",
        "denaturant_evidence": False,
        "sample_state_evidence": "unknown",
    }
    for keyword in defaults["keywords-blacklist"]:
        base[keyword] = False
    for denaturant in defaults["chemical-denaturants"]:
        base[denaturant] = False
    df = pd.DataFrame([{**base, **row} for row in rows])
    # same dtypes create_peptide_dataframe() ends on: an object column of None
    # breaks ~str.contains(), the nullable "string" dtype yields pd.NA
    return df.astype(
        dict.fromkeys(
            [
                "entryID",
                "exp_method",
                "exp_method_subtype",
                "seq",
                "physical_state",
                "sample_state_evidence",
            ],
            "string",
        )
    )


def _labels(results):
    return [r["filter"] for r in results]


def test_baseline_row_passes_every_strict_filter():
    _results, total, passing = report.analyse_tier(_frame("strict", [{}]), "strict")
    assert (total, passing) == (1, 1)


def test_report_applies_the_physical_state_deny_list():
    """A denied state must be filtered AND show up as its own report row."""
    df = _frame(
        "strict",
        [{}, {"physical_state": "molten globule"}, {"physical_state": "denatured"}],
    )
    results, total, passing = report.analyse_tier(df, "strict")

    labels = _labels(results)
    assert any(label.startswith("physical state") for label in labels), labels
    row = next(r for r in results if r["filter"].startswith("physical state"))
    # 'molten globule' is denied outright; 'denatured' is ambiguous and needs
    # independent denaturant evidence, which this row does not carry.
    assert (row["filtered"], row["unique"]) == (1, 1)
    assert (total, passing) == (3, 2)


def test_ambiguous_state_is_denied_only_with_denaturant_evidence():
    df = _frame(
        "strict",
        [
            {"physical_state": "denatured", "denaturant_evidence": False},
            {"physical_state": "denatured", "denaturant_evidence": True},
        ],
    )
    _results, total, passing = report.analyse_tier(df, "strict")
    assert (total, passing) == (2, 1)


def test_report_applies_the_method_fallback():
    """Strict recovers an undeclared-method entry on positive solution evidence
    (``method-fallback='require-solution'``); with the ``"off"`` default the
    report used, the same row is rejected — a whole tier's worth of drift."""
    df = _frame(
        "strict", [{"exp_method_subtype": None, "sample_state_evidence": "solution"}]
    )
    _results, _total, passing = report.analyse_tier(df, "strict")
    assert passing == 1

    as_the_report_used_to = report.tier_prefilter_kwargs("strict")
    as_the_report_used_to["method_fallback"] = "off"
    out = prefilter_dataframe(df.copy(), **as_the_report_used_to)[0]
    assert int(out["pass_pre"].sum()) == 0


def test_report_applies_the_strict_solid_veto():
    """``require-solution`` also vetoes solid evidence whatever the subtype says
    (bmr25289 / bmr27211 in the released strict tier)."""
    df = _frame(
        "strict",
        [{}, {"sample_state_evidence": "solid"}],
    )
    _results, total, passing = report.analyse_tier(df, "strict")
    assert (total, passing) == (2, 1)


def test_unfiltered_tier_denies_no_physical_state():
    df = _frame("unfiltered", [{"physical_state": "molten globule"}])
    results, _total, passing = report.analyse_tier(df, "unfiltered")
    assert passing == 1
    assert not [label for label in _labels(results) if label.startswith("physical")]
