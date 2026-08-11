"""Chemical-denaturant tokens, the sample-component matcher, and membrane mimetics.

The matcher reads ``_Sample_component.Mol_common_name`` for components that are
not the studied polymer itself (``_Sample_component.Entity_ID`` unset). Three
things are pinned here:

* the tier token lists -- ``TFE``/``trifluoroethanol`` added, ``DMSO`` added at
  moderate and above, ``TFA`` and ``Potassium Pyrophosphate`` removed;
* two matcher defects: the "no sample referenced" fallback was assigned *inside*
  the loop over denaturants, and ``entry.samples[sID]`` could raise ``KeyError``
  on a whitespace-padded sample ID that the ``try/except Found`` did not catch;
* the ``membrane_mimetic`` column, which annotates the matched token(s) and
  never filters.
"""

import pandas as pd
import pytest

import trizod.bmrb.bmrb as bmrb
from tests.conftest import requires_bmrb_data
from trizod import paths
from trizod.trizod import MEMBRANE_MIMETIC_TOKENS, fill_row_data, filter_defaults

TIERS = ["unfiltered", "tolerant", "moderate", "strict"]


class _Conditions:
    def get_ionic_strength(self, **kwargs):
        return 0.1

    def get_pH(self, **kwargs):
        return 7.0

    def get_temperature(self, **kwargs):
        return 298.0


class _Sample:
    """Stand-in for ``bmrb.Sample``: only ``components`` matters here.

    A component is ``(ID, Assembly_ID, Entity_ID, Mol_common_name, conc, units)``.
    """

    def __init__(self, components=(), name=None):
        self.name = name
        self.details = None
        self.framecode = "sample_1"
        self.type = "solution"
        self.components = list(components)


class _Named:
    def __init__(self, name=None):
        self.name = name
        self.details = None
        self.seq = None
        self.paramagnetic = None
        self.entities = [("1", "1", "$e", None)]


class _ExperimentList:
    experiments = [("1", "2D 1H-15N HSQC", None, "1", None, "isotropic")]


class _Entry:
    def __init__(self, samples, referenced=("1",)):
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
        self.samples = samples
        self.conditions = {"1": _Conditions()}
        self.experiment_list = _ExperimentList()
        self._referenced = list(referenced)

    def get_peptide_shifts(self):
        return {("1", "1", "1"): (None, "1", "1", self._referenced)}


def _component(name, entity_id=""):
    return ("1", "1", entity_id, name, "10", "mM")


def _fill(entry, denaturants):
    row = pd.Series(
        {"entryID": "1", "stID": "1", "entity_assemID": "1", "entityID": "1"}
    )
    bmrb_entries = pd.DataFrame({"entry": [entry]}, index=["1"])
    return fill_row_data(
        row, chemical_denaturants=denaturants, keywords=[], bmrb_entries=bmrb_entries
    )


def _entry_with(*names, referenced=("1",)):
    return _Entry({"1": _Sample([_component(n) for n in names])}, referenced=referenced)


# --------------------------------------------------------------------------- #
# tier token lists
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tier", TIERS)
def test_tfa_is_no_longer_a_denaturant(tier):
    # 57 percent-unit TFA components, median 0.1 %, 56/57 <= 0.2 %: an HPLC
    # counterion whose acidification is already covered by the pH filter.
    assert "TFA" not in filter_defaults.loc[tier, "chemical-denaturants"]


@pytest.mark.parametrize("tier", TIERS)
def test_potassium_pyrophosphate_is_no_longer_a_denaturant(tier):
    assert (
        "Potassium Pyrophosphate"
        not in filter_defaults.loc[tier, "chemical-denaturants"]
    )


def test_tfe_added_from_tolerant_up():
    assert filter_defaults.loc["unfiltered", "chemical-denaturants"] == []
    for tier in ["tolerant", "moderate", "strict"]:
        tokens = filter_defaults.loc[tier, "chemical-denaturants"]
        assert "TFE" in tokens
        assert "trifluoroethanol" in tokens


def test_dmso_added_from_moderate_up_only():
    assert "DMSO" not in filter_defaults.loc["tolerant", "chemical-denaturants"]
    assert "DMSO" in filter_defaults.loc["moderate", "chemical-denaturants"]
    assert "DMSO" in filter_defaults.loc["strict", "chemical-denaturants"]


@pytest.mark.parametrize("tier", TIERS)
def test_membrane_mimetics_never_filter(tier):
    # Decision D3: annotated, not filtered. The rows they would remove are
    # 93.7 % ordered and contain zero disordered chains.
    tokens = [t.lower() for t in filter_defaults.loc[tier, "chemical-denaturants"]]
    for mimetic in MEMBRANE_MIMETIC_TOKENS:
        assert mimetic.lower() not in tokens


# --------------------------------------------------------------------------- #
# token matching
# --------------------------------------------------------------------------- #


def test_tfe_matches_deuterated_spelling():
    assert _fill(_entry_with("TFE-d2"), ["TFE"])["TFE"] is True


def test_trifluoroethanol_is_not_reachable_through_tfe():
    # 'trifluoroethanol'.lower() does not contain 'tfe' -- both tokens are needed.
    row = _fill(_entry_with("trifluoroethanol"), ["TFE", "trifluoroethanol"])
    assert row["TFE"] is False
    assert row["trifluoroethanol"] is True


def test_matching_is_case_insensitive():
    assert _fill(_entry_with("Urea"), ["urea"])["urea"] is True
    assert (
        _fill(_entry_with("potassium pyrophosphate"), ["Potassium Pyrophosphate"])[
            "Potassium Pyrophosphate"
        ]
        is True
    )


def test_entity_id_guard_blocks_polymer_components():
    # bmr31202's '4-guanidinophenyl 4-guanidinobenzoate' is a declared entity,
    # not a bulk denaturant.
    entry = _Entry(
        {
            "1": _Sample(
                [_component("4-guanidinophenyl 4-guanidinobenzoate", entity_id="2")]
            )
        }
    )
    assert _fill(entry, ["guanidin"])["guanidin"] is False


def test_absent_denaturant_stays_false():
    assert _fill(_entry_with("sodium phosphate"), ["urea"])["urea"] is False


# --------------------------------------------------------------------------- #
# matcher defects
# --------------------------------------------------------------------------- #


def test_sample_fallback_applies_to_every_denaturant():
    # No sample is referenced by the shift table, so every token must fall back
    # to all of the entry's samples -- not just the ones after the first
    # iteration of the loop that used to assign the fallback.
    entry = _Entry({"1": _Sample([_component("urea")])}, referenced=())
    row = _fill(entry, ["urea", "guanidin", "DMSO"])
    assert row["urea"] is True
    assert row["guanidin"] is False


def test_sample_fallback_applies_with_an_empty_denaturant_list():
    entry = _Entry({"1": _Sample([_component("SDS")])}, referenced=())
    assert _fill(entry, [])["membrane_mimetic"] == "SDS"


def test_unknown_sample_id_does_not_raise():
    # bmrb.py strips sample IDs only AFTER the membership test, so a
    # whitespace-padded ID can name a sample that is not in entry.samples.
    entry = _Entry({"1": _Sample([_component("urea")])}, referenced=("2",))
    assert _fill(entry, ["urea"])["urea"] is False


# --------------------------------------------------------------------------- #
# membrane_mimetic column
# --------------------------------------------------------------------------- #


def test_membrane_mimetic_column_carries_the_matched_tokens():
    entry = _entry_with("SDS", "DPC micelles")
    assert _fill(entry, [])["membrane_mimetic"] == "SDS;DPC;micelle"


def test_membrane_mimetic_distinguishes_sds_from_maltoside():
    # A bool would collapse the SDS-micelle vs DDM-solubilisation distinction.
    assert _fill(_entry_with("SDS"), [])["membrane_mimetic"] == "SDS"
    assert (
        _fill(_entry_with("n-dodecyl-beta-D-maltoside"), [])["membrane_mimetic"]
        == "dodecyl;maltoside"
    )


def test_membrane_mimetic_is_na_when_nothing_matches():
    assert pd.isna(_fill(_entry_with("sodium phosphate"), [])["membrane_mimetic"])


def test_sodium_dodecyl_sulfate_matches_dodecyl_not_sds():
    assert (
        _fill(_entry_with("sodium dodecyl sulfate"), [])["membrane_mimetic"]
        == "dodecyl"
    )


# --------------------------------------------------------------------------- #
# real entries
# --------------------------------------------------------------------------- #


def _fill_real(entry_id, stID, denaturants):
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
        row, chemical_denaturants=denaturants, keywords=[], bmrb_entries=bmrb_entries
    )


@requires_bmrb_data
@pytest.mark.parametrize(
    ("entry_id", "stID", "token", "expected"),
    [
        ("15259", "1", "TFA", True),  # 0.1 % TFA: matched, no longer a default
        ("15259", "2", "TFE", True),  # the TFE sample is the second shift table
        ("15259", "1", "TFE", False),  # ... and not the first: resolution is per row
        (
            "16596",
            "1",
            "Potassium Pyrophosphate",
            True,
        ),  # a buffer, no longer a default
        ("11051", "1", "DMSO", True),  # DMSO-d6 at 95 %
    ],
)
def test_real_entry_component_flags(entry_id, stID, token, expected):
    assert _fill_real(entry_id, stID, [token])[token] is expected


@requires_bmrb_data
def test_real_membrane_entry_is_annotated_not_filtered():
    strict_tokens = list(filter_defaults.loc["strict", "chemical-denaturants"])
    filled = _fill_real("11002", "1", strict_tokens)
    assert "SDS" in filled["membrane_mimetic"]
    assert not any(filled[token] for token in strict_tokens)
