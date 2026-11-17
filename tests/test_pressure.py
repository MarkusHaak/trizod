"""Sample pressure is exported per chain as ``pressure_bar`` and capped at 400 bar.

``_Sample_condition_variable`` rows of type ``pressure`` were parsed into
``SampleConditions.pressure`` and then never read, so a 2,500-bar sample
(ubiquitin bmr18611, the pressure-unfolded L50A mutant bmr52344, the 2,250-bar
melittin shift table of bmr51703) was indistinguishable from an ambient one.

The column follows the condition list of the shift table actually scored, as
``pH`` and ``temperature`` do, and is converted to bar from the deposited unit.
Since 2026-11 the tolerant, moderate and strict tiers exclude a chain whose
pressure is >= 400 bar (``max-pressure-bar``): kilobar pressure populates
partially unfolded states, so its G-scores describe a perturbed ensemble. A
pressure that is not recorded is taken as ambient and passes. One unit repair
exists: a value of 900-1,100 deposited as "bar" is read as hPa (bmr25704,
"1013 bar").
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tests._stubs import (
    Entry,
    Sample,
    condition_frame,
    fill,
    fill_real,
    prefilter_tier,
)
from tests.conftest import requires_bmrb_data
from trizod.bmrb.bmrb import SampleConditions
from trizod.cli.main import app
from trizod.pipeline import print_filter_losses
from trizod.trizod import filter_defaults, output_dataset

ATM = 1.01325


def _conditions(val, unit):
    cond = SampleConditions.__new__(SampleConditions)
    cond.id = "1"
    cond.ionic_strength = (None, None)
    cond.pH = (None, None)
    cond.temperature = (None, None)
    cond.pressure = (val, unit)
    return cond


# --------------------------------------------------------------------------- #
# unit normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("val", "unit", "bar"),
    [
        ("1", "atm", ATM),  # 14,167 deposited rows
        ("1.000", "atm", ATM),
        ("1", "ATM", ATM),
        ("2000", "atm", 2000 * ATM),
        ("2500", "bar", 2500.0),  # bmr18611 / bmr52344
        ("2250", "bar", 2250.0),  # bmr51703
        ("1", "bar", 1.0),
        ("101325", "Pa", ATM),
        ("100000", "Pa", 1.0),
        ("101.325", "kPa", ATM),
        ("250", "MPa", 2500.0),
        ("1013.25", "mbar", ATM),
        ("1013", "hPa", 1.013),
        ("760", "mmHg", ATM),
        ("760", "mm Hg", ATM),
        ("760", "Torr", ATM),
        ("14.6959", "psi", ATM),
        (" 1 ", " atm ", ATM),
    ],
)
def test_pressure_is_converted_to_bar(val, unit, bar):
    assert _conditions(val, unit).get_pressure_bar() == pytest.approx(bar, rel=1e-4)


@pytest.mark.parametrize("val", ["ambient", "Ambient", "atmospheric"])
def test_ambient_pressure_is_one_atmosphere(val):
    # 50 deposited rows say 'ambient' (unit atm): a recorded value, not a gap.
    assert _conditions(val, "atm").get_pressure_bar() == pytest.approx(ATM)


@pytest.mark.parametrize(
    ("val", "unit"),
    [
        (None, None),  # no pressure row at all
        ("", ""),
        (".", "atm"),  # NMR-STAR null value
        ("1", ""),  # value without a unit: 47 rows deposit '1' with unit '.'
        ("1", "."),
        ("1", "1"),
        ("1", "furlongs"),
        ("high", "bar"),
    ],
)
def test_unrecorded_or_unconvertible_pressure_is_nan(val, unit):
    # Null means unknown, never a default: no unit is assumed.
    assert math.isnan(_conditions(val, unit).get_pressure_bar())


def test_literal_values_are_not_repaired():
    # 89 rows deposit '1' in Pa: almost certainly 1 atm, but the column reports
    # the deposition rather than guessing the unit the depositor meant.
    assert _conditions("1", "Pa").get_pressure_bar() == pytest.approx(1e-5)


@pytest.mark.parametrize(
    ("val", "bar"),
    [
        ("1013", 1.013),  # bmr25704: one atmosphere deposited in hPa as "bar"
        ("1013.25", 1.01325),
        ("900", 0.9),  # both edges of the guard are inclusive
        ("1100", 1.1),
    ],
)
def test_about_one_thousand_bar_is_read_as_hpa(val, bar):
    # Nothing is measured at ~1 kbar in the corpus (the deposits are bimodal:
    # ~1 bar, or 2,000-2,500 bar), so a 900-1,100 "bar" value is an
    # atmosphere written in hPa/mbar -- not a high-pressure sample.
    assert _conditions(val, "bar").get_pressure_bar() == pytest.approx(bar)


@pytest.mark.parametrize(
    ("val", "unit", "bar"),
    [
        ("899", "bar", 899.0),
        ("1101", "bar", 1101.0),
        ("2500", "bar", 2500.0),
        ("1013", "mbar", 1.013),  # already a small unit: not scaled twice
        ("1000", "atm", 1000 * ATM),  # the guard is for the unit "bar" only
    ],
)
def test_hpa_guard_is_limited_to_bar_values_near_one_thousand(val, unit, bar):
    assert _conditions(val, unit).get_pressure_bar() == pytest.approx(bar)


# --------------------------------------------------------------------------- #
# per-row: the scored shift table's own condition list
# --------------------------------------------------------------------------- #


def _entry():
    """Two shift tables of one chain, recorded under different condition lists."""
    return Entry(
        {"1": Sample()},
        conditions={
            "1": _conditions("1", "bar"),
            "3": _conditions("2250", "bar"),
            "4": _conditions(None, None),
        },
        peptide_shifts={
            ("1", "1", "1"): (None, "1", "1", ["1"]),
            ("2", "1", "1"): (None, "3", "1", ["1"]),
            ("3", "1", "1"): (None, "4", "1", ["1"]),
        },
    )


def _fill(stID):
    return fill(_entry(), st_id=stID)


def test_pressure_follows_the_shift_tables_condition_list():
    assert _fill("1")["pressure_bar"] == pytest.approx(1.0)
    assert _fill("2")["pressure_bar"] == pytest.approx(2250.0)


def test_absent_pressure_is_nan_not_a_default():
    assert math.isnan(_fill("3")["pressure_bar"])


# --------------------------------------------------------------------------- #
# exported to scores.json
# --------------------------------------------------------------------------- #


def _scored_frame(pressures):
    seq = "AGAGA"
    n = len(seq)
    rows = []
    for i, p in enumerate(pressures):
        rows.append(
            {
                "entryID": str(10000 + i),
                "stID": "1",
                "entity_assemID": "1",
                "entityID": "1",
                "entity_name": "peptide",
                "exp_method": "NMR",
                "exp_method_subtype": "solution",
                "sample_state_evidence": "solution",
                "physical_state": "native",
                "cosolvent_evidence": False,
                "membrane_mimetic": None,
                "membrane_mimetic_sample": False,
                "citation_DOI": None,
                "citation_title": None,
                "ionic_strength": 0.1,
                "pH": 7.0,
                "temperature": 298.0,
                "temperature_placeholder": False,
                "pressure_bar": p,
                "bbshift_positions_post": n,
                "bbshift_types_post": 3,
                "total_bbshifts": 3 * n,
                "seq": seq,
                "k": [3] * n,
                "zscores": [0.5] * n,
                "pass_post": True,
            }
        )
    frame = pd.DataFrame(rows)
    from trizod.offsets import OFFSET_COLUMNS

    for column in OFFSET_COLUMNS:
        frame[column] = 0.0
    return frame


def test_scores_json_carries_pressure_bar(tmp_path):
    output_dataset(
        _scored_frame([2500.0, np.nan]),
        tmp_path / "scores",
        "json",
        ["zscores"],
        4,
        include_shifts=False,
        no_shift_averaging=False,
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "scores.json").read_text().splitlines()
        if line.strip()
    ]
    assert records[0]["pressure_bar"] == pytest.approx(2500.0)
    assert records[1]["pressure_bar"] is None


# --------------------------------------------------------------------------- #
# the max-pressure-bar filter
# --------------------------------------------------------------------------- #

PRESSURES = [np.nan, 1.01325, 399.9, 400.0, 500.0, 2500.0]


def test_max_pressure_bar_defaults_per_tier():
    # tolerant too: it already removes perturbing cosolvents on the same
    # "non-ambient ensemble" logic
    assert filter_defaults["max-pressure-bar"].tolist() == [np.inf, 400.0, 400.0, 400.0]


def test_unfiltered_keeps_every_pressure():
    out = prefilter_tier(condition_frame(pressure_bar=PRESSURES), "unfiltered")
    assert out["pass_pre"].tolist() == [True] * len(PRESSURES)


@pytest.mark.parametrize("tier", ["tolerant", "moderate", "strict"])
def test_pressure_of_400_bar_or_more_is_excluded(tier):
    out = prefilter_tier(condition_frame(pressure_bar=PRESSURES), tier)
    # NaN = not recorded = ambient, and passes; 400 bar itself is excluded
    assert out["pass_pre"].tolist() == [True, True, True, False, False, False]


def test_pressure_filter_is_reported_as_a_prefilter_criterion():
    sels_pre = prefilter_tier(condition_frame(pressure_bar=PRESSURES), "strict", 2)
    keys = [name for name, _crit in sels_pre]
    assert "pressure" in keys
    sel = sels_pre[next(k for k in sels_pre if k[0] == "pressure")]
    assert sel.tolist() == [True, True, True, False, False, False]


def test_print_filter_losses_lists_the_pressure_filter(capsys):
    df = condition_frame(pressure_bar=PRESSURES)
    _, missing, sels_pre, kws, cos, para, sels_all = prefilter_tier(df, "strict", None)
    df["pass_post"] = df["pass_pre"]
    print_filter_losses(df, missing, sels_pre, kws, cos, para, sels_all, {}, {}, {})
    report = capsys.readouterr().out
    line = next(ln for ln in report.splitlines() if ln.startswith("pressure"))
    assert line.split(":")[1].split()[0] == "3"


def test_a_finite_pressure_cap_needs_the_pressure_column():
    df = condition_frame().drop(columns="pressure_bar")
    with pytest.raises(ValueError, match="pressure_bar"):
        prefilter_tier(df, "strict")


def test_max_pressure_bar_is_a_cli_override(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        "trizod.trizod.run_scoring_pipeline",
        lambda args: captured.setdefault("args", args),
    )
    argv = ["--input-dir", str(tmp_path), "--output-prefix", str(tmp_path / "o")]
    argv += ["--cache-dir", str(tmp_path / "c"), "--filter-defaults", "strict"]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, result.output
    assert captured.pop("args").max_pressure_bar == 400.0
    result = CliRunner().invoke(app, [*argv, "--max-pressure-bar", "3000"])
    assert result.exit_code == 0, result.output
    assert captured["args"].max_pressure_bar == 3000.0


# --------------------------------------------------------------------------- #
# real entries
# --------------------------------------------------------------------------- #


def _fill_real(entry_id, stID="1"):
    return fill_real(entry_id, stID)


@requires_bmrb_data
@pytest.mark.parametrize(
    ("entry_id", "stID", "bar"),
    [
        ("18611", "1", 2500.0),  # ubiquitin, 45 % methanol
        ("52344", "1", 2500.0),  # ubiquitin L50A, pressure unfolded
        ("51703", "1", 1.0),  # melittin, ambient shift table ...
        ("51703", "2", 2250.0),  # ... and its 2,250-bar partner (condition list 3)
        ("34568", "1", ATM),
    ],
)
def test_real_entry_pressure(entry_id, stID, bar):
    assert _fill_real(entry_id, stID)["pressure_bar"] == pytest.approx(bar)


@requires_bmrb_data
def test_real_entry_without_a_pressure_row_is_nan():
    # bmr6436 declares no pressure at all
    assert math.isnan(_fill_real("6436")["pressure_bar"])
