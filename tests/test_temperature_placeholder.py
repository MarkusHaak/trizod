"""A deposited temperature of exactly 273 K is a placeholder, not a measurement.

Exactly 273 K appears in 12.9 % of the ADIT-NMR entries submitted from July
2010 to mid-2020 against 0.6 % of all others, and for 237 of 247 checkable
entries the co-deposited PDB entry reports a different temperature (mostly
298 K). Since 2026-11 a temperature row whose raw deposited value is exactly
"273" or "273.0" (unit K or none; not 273.15 or 273.65) is read as NOT
deposited, in every entry -- unless the entry's own text states 0 C (``273 K``
or ``0 °C`` in the condition list's name, framecode or details, the experiment
list details, the entry details or a sample's details). Like any missing
temperature, the strict tier then excludes the chain and the other tiers score
it at the default 298 K; ``temperature_placeholder`` marks it.
"""

import json
import math

import numpy as np
import pandas as pd
import pynmrstar
import pytest

import trizod.bmrb.bmrb as bmrb
from tests._stubs import Entry, Sample, condition_frame, fill, prefilter_tier
from trizod.bmrb.bmrb import SampleConditions, states_zero_celsius
from trizod.trizod import output_dataset

# --------------------------------------------------------------------------- #
# the condition list
# --------------------------------------------------------------------------- #


def _conditions(val, unit="K", stated=False):
    sf = pynmrstar.Saveframe.from_scratch(
        "sample_conditions_1", "_Sample_condition_list"
    )
    sf.add_tag("Sf_category", "sample_conditions")
    sf.add_tag("ID", "1")
    loop = pynmrstar.Loop.from_scratch("_Sample_condition_variable")
    loop.add_tag(["Type", "Val", "Val_err", "Val_units"])
    loop.add_data([["temperature", val, ".", unit or "."]])
    sf.add_loop(loop)
    cond = SampleConditions(sf)
    cond.zero_celsius_stated = stated
    return cond


@pytest.mark.parametrize(("val", "unit"), [("273", "K"), ("273.0", "K"), ("273", "")])
def test_exactly_273_is_not_deposited(val, unit):
    cond = _conditions(val, unit)
    assert cond.is_temperature_placeholder() is True
    assert cond.get_temperature() == 298.0  # the default, where a tier imputes
    assert math.isnan(cond.get_temperature(return_default=False))


def test_unitless_placeholder_is_missing_even_without_unit_assumptions():
    cond = _conditions("273", "")
    assert math.isnan(
        cond.get_temperature(return_default=False, assume_si=False, fix_outliers=False)
    )


@pytest.mark.parametrize(
    ("val", "unit", "kelvin"),
    [
        ("273.15", "K", 273.15),
        ("273.65", "K", 273.65),
        ("298", "K", 298.0),
        ("0", "C", 273.15),
        ("278", "K", 278.0),
    ],
)
def test_other_temperatures_are_kept(val, unit, kelvin):
    cond = _conditions(val, unit)
    assert cond.is_temperature_placeholder() is False
    assert cond.get_temperature(return_default=False) == pytest.approx(kelvin)


def test_corroborated_273_is_kept():
    cond = _conditions("273", "K", stated=True)
    assert cond.is_temperature_placeholder() is False
    assert cond.get_temperature(return_default=False) == 273.0


def test_no_temperature_row_is_not_a_placeholder():
    cond = SampleConditions.__new__(SampleConditions)
    cond.id = "1"
    cond.temperature = (None, None)
    assert cond.is_temperature_placeholder() is False


@pytest.mark.parametrize(
    "text",
    [
        "spectra recorded at 273 K",
        "273K",
        "measured at 0 °C",
        "0°C",
        "0 oC",
        "0oC",
        "0 degC",
        "0 deg C",
        "AT 0 DEG C",
    ],
)
def test_zero_celsius_is_stated(text):
    assert states_zero_celsius(text) is True


@pytest.mark.parametrize(
    "text",
    ["10 °C", "20 oC", "100 °C", "298 K", "1273 K", "", None, "pH 7.0"],
)
def test_zero_celsius_is_not_stated(text):
    assert states_zero_celsius(text) is False


# --------------------------------------------------------------------------- #
# BmrbEntry: the corroboration needs entry-level text
# --------------------------------------------------------------------------- #

ENTRY = """\
data_99999

save_entry_information
   _Entry.Sf_category                 entry_information
   _Entry.Sf_framecode                entry_information
   _Entry.ID                          99999
   _Entry.Title                       '{entry_title}'
   _Entry.Details                     '{entry_details}'
   _Entry.Experimental_method         NMR
   _Entry.Experimental_method_subtype solution
save_

save_assembly
   _Assembly.Sf_category   assembly
   _Assembly.Sf_framecode  assembly
   _Assembly.ID            1
   _Assembly.Name          peptide

   loop_
      _Entity_assembly.ID
      _Entity_assembly.Entity_assembly_name
      _Entity_assembly.Entity_ID
      _Entity_assembly.Entity_label
      _Entity_assembly.Physical_state
      _Entity_assembly.Assembly_ID

      1   peptide   1   $peptide   native   1
   stop_
save_

save_peptide
   _Entity.Sf_category                    entity
   _Entity.Sf_framecode                   peptide
   _Entity.ID                             1
   _Entity.Name                           peptide
   _Entity.Type                           polymer
   _Entity.Polymer_type                   polypeptide(L)
   _Entity.Polymer_seq_one_letter_code    AGAGA
save_

save_sample_1
   _Sample.Sf_category    sample
   _Sample.Sf_framecode   sample_1
   _Sample.ID             1
   _Sample.Type           solution
   _Sample.Details        '{sample_details}'

   loop_
      _Sample_component.ID
      _Sample_component.Mol_common_name
      _Sample_component.Assembly_ID
      _Sample_component.Entity_ID
      _Sample_component.Concentration_val
      _Sample_component.Concentration_val_units
      _Sample_component.Sample_ID

      1   peptide   1   1   1    mM   1
      2   NaCl      .   .   50   mM   1
   stop_
save_

save_{cond_framecode}
   _Sample_condition_list.Sf_category    sample_conditions
   _Sample_condition_list.Sf_framecode   {cond_framecode}
   _Sample_condition_list.ID             1
   _Sample_condition_list.Name           '{cond_name}'
   _Sample_condition_list.Details        '{cond_details}'

   loop_
      _Sample_condition_variable.Type
      _Sample_condition_variable.Val
      _Sample_condition_variable.Val_err
      _Sample_condition_variable.Val_units
      _Sample_condition_variable.Sample_condition_list_ID

      temperature   {temperature}   .   K   1
      pH            7.0             .   pH  1
   stop_
save_

save_experiment_list
   _Experiment_list.Sf_category    experiment_list
   _Experiment_list.Sf_framecode   experiment_list
   _Experiment_list.ID             1
   _Experiment_list.Details        '{experiment_details}'

   loop_
      _Experiment.ID
      _Experiment.Name
      _Experiment.Sample_ID
      _Experiment.Sample_state
      _Experiment.Sample_condition_list_ID
      _Experiment.Experiment_list_ID

      1   '2D 1H-15N HSQC'   1   isotropic   1   1
   stop_
save_

save_assigned_chem_shifts
   _Assigned_chem_shift_list.Sf_category                 assigned_chemical_shifts
   _Assigned_chem_shift_list.Sf_framecode                assigned_chem_shifts
   _Assigned_chem_shift_list.ID                          1
   _Assigned_chem_shift_list.Sample_condition_list_ID    1

   loop_
      _Chem_shift_experiment.Experiment_ID
      _Chem_shift_experiment.Experiment_name
      _Chem_shift_experiment.Sample_ID
      _Chem_shift_experiment.Sample_state
      _Chem_shift_experiment.Assigned_chem_shift_list_ID

      1   '2D 1H-15N HSQC'   1   isotropic   1
   stop_

   loop_
      _Atom_chem_shift.ID
      _Atom_chem_shift.Entity_assembly_ID
      _Atom_chem_shift.Entity_ID
      _Atom_chem_shift.Seq_ID
      _Atom_chem_shift.Comp_ID
      _Atom_chem_shift.Atom_ID
      _Atom_chem_shift.Atom_type
      _Atom_chem_shift.Val
      _Atom_chem_shift.Val_err
      _Atom_chem_shift.Ambiguity_code
      _Atom_chem_shift.Assigned_chem_shift_list_ID

      1   1   1   2   GLY   N   N   109.0   .   1   1
      2   1   1   2   GLY   H   H   8.30    .   1   1
   stop_
save_
"""

TEXT_FIELDS = {
    "entry_title": "a peptide",
    "entry_details": "none",
    "sample_details": "none",
    "cond_framecode": "sample_conditions_1",
    "cond_name": "conditions_1",
    "cond_details": "none",
    "experiment_details": "none",
}


def _entry(tmp_path, temperature="273", **text):
    fields = {**TEXT_FIELDS, **text, "temperature": temperature}
    entry_dir = tmp_path / "bmr99999"
    entry_dir.mkdir()
    (entry_dir / "bmr99999_3.str").write_text(ENTRY.format(**fields))
    return bmrb.BmrbEntry("99999", entry_dir)


def test_entry_placeholder_without_corroboration(tmp_path):
    cond = _entry(tmp_path).conditions["1"]
    assert cond.is_temperature_placeholder() is True
    assert cond.get_temperature() == 298.0


@pytest.mark.parametrize(
    ("field", "text"),
    [
        ("cond_details", "spectra recorded at 0 degC"),
        ("cond_name", "273K conditions"),
        ("cond_framecode", "conditions_273K"),
        ("experiment_details", "all experiments at 0 °C"),
        ("entry_details", "imino protons observed at 273 K"),
        ("sample_details", "kept at 0 oC"),
    ],
)
def test_entry_text_corroborates_zero_celsius(tmp_path, field, text):
    cond = _entry(tmp_path, **{field: text}).conditions["1"]
    assert cond.is_temperature_placeholder() is False
    assert cond.get_temperature(return_default=False) == 273.0


def test_title_is_not_a_corroborating_field(tmp_path):
    entry = _entry(tmp_path, entry_title="cold shock protein at 0 °C")
    assert entry.conditions["1"].is_temperature_placeholder() is True


def test_entry_with_273_15_is_untouched(tmp_path):
    cond = _entry(tmp_path, temperature="273.15").conditions["1"]
    assert cond.is_temperature_placeholder() is False
    assert cond.get_temperature(return_default=False) == pytest.approx(273.15)


# --------------------------------------------------------------------------- #
# per chain: the column and the tiers
# --------------------------------------------------------------------------- #


def _placeholder_entry():
    return Entry({"1": Sample()}, conditions={"1": _conditions("273", "K")})


def test_row_marks_the_imputed_temperature():
    row = fill(_placeholder_entry())
    assert row["temperature_placeholder"] is True
    assert row["temperature"] == 298.0


def test_row_without_placeholder():
    row = fill(Entry({"1": Sample()}))
    assert row["temperature_placeholder"] is False


def test_strict_excludes_the_placeholder_and_moderate_scores_it_at_298():
    strict = fill(_placeholder_entry(), return_default=False)
    moderate = fill(_placeholder_entry(), return_default=True)
    assert math.isnan(strict["temperature"])
    df = condition_frame(temperature=[strict["temperature"]])
    assert prefilter_tier(df, "strict")["pass_pre"].tolist() == [False]
    df = condition_frame(temperature=[moderate["temperature"]])
    assert prefilter_tier(df, "moderate")["pass_pre"].tolist() == [True]


def test_scores_json_carries_the_new_flags(tmp_path):
    seq = "AGAGA"
    frame = pd.DataFrame(
        {
            "entryID": ["1", "2"],
            "stID": ["1", "1"],
            "entity_assemID": ["1", "1"],
            "entityID": ["1", "1"],
            "entity_name": ["p", "p"],
            "exp_method": ["NMR", "NMR"],
            "exp_method_subtype": ["solution", "solution"],
            "sample_state_evidence": ["solution", "solution"],
            "physical_state": ["native", "native"],
            "cosolvent_evidence": [False, False],
            "membrane_mimetic": ["DPC", None],
            "membrane_mimetic_sample": [True, False],
            "citation_DOI": [None, None],
            "citation_title": [None, None],
            "ionic_strength": [0.1, 0.1],
            "pH": [7.0, 7.0],
            "temperature": [298.0, 298.0],
            "temperature_placeholder": [True, False],
            "pressure_bar": [np.nan, np.nan],
            "bbshift_positions_post": [5, 5],
            "bbshift_types_post": [3, 3],
            "total_bbshifts": [15, 15],
            "seq": [seq, seq],
            "k": [[3] * 5, [3] * 5],
            "zscores": [[0.5] * 5, [0.5] * 5],
            "pass_post": [True, True],
        }
    )
    from trizod.offsets import OFFSET_COLUMNS

    for column in OFFSET_COLUMNS:
        frame[column] = 0.0
    output_dataset(frame, tmp_path / "scores", "json", ["zscores"], 4, False, False)
    records = [
        json.loads(line)
        for line in (tmp_path / "scores.json").read_text().splitlines()
        if line.strip()
    ]
    assert [r["temperature_placeholder"] for r in records] == [True, False]
    assert [r["membrane_mimetic_sample"] for r in records] == [True, False]
