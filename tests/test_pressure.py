"""Sample pressure is exported per chain as ``pressure_bar`` -- and filters nothing.

``_Sample_condition_variable`` rows of type ``pressure`` were parsed into
``SampleConditions.pressure`` and then never read, so a 2,500-bar sample
(ubiquitin bmr18611, the pressure-unfolded L50A mutant bmr52344, the 2,250-bar
melittin shift table of bmr51703) was indistinguishable from an ambient one.

The column follows the condition list of the shift table actually scored, as
``pH`` and ``temperature`` do, and is converted to bar from the deposited unit.
No ceiling is applied at any tier: where a high pressure stops being "native"
is a threshold for an NMR expert, not a parser.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

import trizod.bmrb.bmrb as bmrb
from tests.conftest import requires_bmrb_data
from trizod import paths
from trizod.bmrb.bmrb import SampleConditions
from trizod.pipeline import prefilter_dataframe
from trizod.trizod import fill_row_data, filter_defaults, output_dataset

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


# --------------------------------------------------------------------------- #
# per-row: the scored shift table's own condition list
# --------------------------------------------------------------------------- #


class _Named:
    def __init__(self, name=None):
        self.name = name
        self.details = None
        self.framecode = "sample_1"
        self.solvent_system = None
        self.seq = None
        self.paramagnetic = None
        self.components = []
        self.type = "solution"
        self.entities = [("1", "1", "$e", None)]


class _ExperimentList:
    experiments = [("1", "2D 1H-15N HSQC", None, "1", None, "isotropic")]


class _Entry:
    """Two shift tables of one chain, recorded under different condition lists."""

    def __init__(self):
        self.id = "1"
        self.title = None
        self.details = None
        self.citation_title = None
        self.citation_DOI = None
        self.exp_method = "NMR"
        self.exp_method_subtype = "solution"
        self.citation_keywords = []
        self.struct_keywords = []
        self.entities = {"1": _Named(name="entity")}
        self.assemblies = {"1": _Named(name="assembly")}
        self.samples = {"1": _Named()}
        self.conditions = {
            "1": _conditions("1", "bar"),
            "3": _conditions("2250", "bar"),
            "4": _conditions(None, None),
        }
        self.experiment_list = _ExperimentList()

    def get_peptide_shifts(self):
        return {
            ("1", "1", "1"): (None, "1", "1", ["1"]),
            ("2", "1", "1"): (None, "3", "1", ["1"]),
            ("3", "1", "1"): (None, "4", "1", ["1"]),
        }


def _fill(stID):
    row = pd.Series(
        {"entryID": "1", "stID": stID, "entity_assemID": "1", "entityID": "1"}
    )
    bmrb_entries = pd.DataFrame({"entry": [_Entry()]}, index=["1"])
    return fill_row_data(
        row, perturbing_cosolvents=[], keywords=[], bmrb_entries=bmrb_entries
    )


def test_pressure_follows_the_shift_tables_condition_list():
    assert _fill("1")["pressure_bar"] == pytest.approx(1.0)
    assert _fill("2")["pressure_bar"] == pytest.approx(2250.0)


def test_absent_pressure_is_nan_not_a_default():
    assert math.isnan(_fill("3")["pressure_bar"])


# --------------------------------------------------------------------------- #
# exported, never filtered
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
                "citation_DOI": None,
                "citation_title": None,
                "ionic_strength": 0.1,
                "pH": 7.0,
                "temperature": 298.0,
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


@pytest.mark.parametrize("tier", ["unfiltered", "tolerant", "moderate", "strict"])
def test_no_tier_has_a_pressure_filter(tier):
    assert not any("pressure" in key for key in filter_defaults.columns)
    defaults = filter_defaults.loc[tier]
    df = pd.DataFrame(
        {
            "exp_method": ["NMR", "NMR"],
            "exp_method_subtype": ["solution", "solution"],
            "sample_state_evidence": ["solution", "solution"],
            "temperature": [298.0, 298.0],
            "ionic_strength": [0.1, 0.1],
            "pH": [7.0, 7.0],
            "seq": ["A" * 40, "A" * 40],
            "total_bbshifts": [200, 200],
            "bbshift_types": [7, 7],
            "bbshift_positions": [40, 40],
            "pressure_bar": [1.0, 2500.0],
            "paramagnetic": [False, False],
        }
    )
    out = prefilter_dataframe(
        df,
        method_whitelist=defaults["exp-method-whitelist"],
        method_blacklist=defaults["exp-method-blacklist"],
        temperature_range=defaults["temperature-range"],
        ionic_strength_range=defaults["ionic-strength-range"],
        pH_range=defaults["pH-range"],
        peptide_length_range=list(defaults["peptide-length-range"]) + [np.inf],
        min_backbone_shift_types=defaults["min-backbone-shift-types"],
        min_backbone_shift_positions=defaults["min-backbone-shift-positions"],
        min_backbone_shift_fraction=defaults["min-backbone-shift-fraction"],
        max_noncanonical_fraction=defaults["max-noncanonical-fraction"],
        max_x_fraction=defaults["max-x-fraction"],
        keywords=[],
        perturbing_cosolvents=[],
        exclude_paramagnetic=defaults["exclude-paramagnetic"],
        method_fallback=defaults["method-fallback"],
    )[0]
    assert out["pass_pre"].tolist() == [True, True]


# --------------------------------------------------------------------------- #
# real entries
# --------------------------------------------------------------------------- #


def _fill_real(entry_id, stID="1"):
    entry = bmrb.BmrbEntry(entry_id, paths.RAW_BMRB / f"bmr{entry_id}")
    key = next(k for k in entry.get_peptide_shifts() if k[0] == stID)
    row = pd.Series(
        {
            "entryID": entry_id,
            "stID": key[0],
            "entity_assemID": key[1],
            "entityID": key[2],
        }
    )
    bmrb_entries = pd.DataFrame({"entry": [entry]}, index=[entry_id])
    return fill_row_data(
        row, perturbing_cosolvents=[], keywords=[], bmrb_entries=bmrb_entries
    )


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
