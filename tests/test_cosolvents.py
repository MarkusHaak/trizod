"""Perturbing-cosolvent tokens, the sample-component matcher, and membrane mimetics.

The filter was called ``chemical-denaturants`` until 2026-08. Only urea and the
guanidinium salts denature; TFE, HFIP and DMSO are helix inducers, so the two
halves of the list bias the score in OPPOSITE directions. What every token
shares is that the sample is no longer aqueous buffer, which is where POTENCI
and the LACS reference tables are parameterised -- see the naming note beside
``COSOLVENT_TOKENS`` in ``trizod/trizod.py``.

The matcher reads ``_Sample_component.Mol_common_name`` for components that are
not the studied polymer itself (``_Sample_component.Entity_ID`` unset). Four
things are pinned here:

* the tier token lists -- ``TFE``/``trifluoroethanol`` added, ``TFA`` and
  ``Potassium Pyrophosphate`` removed, HFIP added at tolerant and above, ``DMSO``
  moved from moderate to tolerant, and the deposited spellings a token misses;
* which tokens are matched on word boundaries (``urea``, ``HFIP``) and which are
  substring matches;
* two matcher defects: the "no sample referenced" fallback was assigned *inside*
  the loop over cosolvents, and ``entry.samples[sID]`` could raise ``KeyError``
  on a whitespace-padded sample ID that the ``try/except Found`` did not catch;
* the ``membrane_mimetic`` column, which annotates the matched token(s) and
  never filters;
* the CLI surface: ``--perturbing-cosolvents`` and its deprecated alias
  ``--chemical-denaturants``.
"""

import logging

import pandas as pd
import pytest
from typer.testing import CliRunner

import trizod.bmrb.bmrb as bmrb
from tests.conftest import requires_bmrb_data
from trizod import paths
from trizod.cli.main import app
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


def _fill(entry, cosolvents):
    row = pd.Series(
        {"entryID": "1", "stID": "1", "entity_assemID": "1", "entityID": "1"}
    )
    bmrb_entries = pd.DataFrame({"entry": [entry]}, index=["1"])
    return fill_row_data(
        row, perturbing_cosolvents=cosolvents, keywords=[], bmrb_entries=bmrb_entries
    )


def _entry_with(*names, referenced=("1",)):
    return _Entry({"1": _Sample([_component(n) for n in names])}, referenced=referenced)


# --------------------------------------------------------------------------- #
# tier token lists
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tier", TIERS)
def test_tfa_is_no_longer_a_cosolvent_token(tier):
    # 57 percent-unit TFA components, median 0.1 %, 56/57 <= 0.2 %: an HPLC
    # counterion whose acidification is already covered by the pH filter.
    assert "TFA" not in filter_defaults.loc[tier, "perturbing-cosolvents"]


@pytest.mark.parametrize("tier", TIERS)
def test_potassium_pyrophosphate_is_no_longer_a_cosolvent_token(tier):
    assert (
        "Potassium Pyrophosphate"
        not in filter_defaults.loc[tier, "perturbing-cosolvents"]
    )


def test_tfe_added_from_tolerant_up():
    assert filter_defaults.loc["unfiltered", "perturbing-cosolvents"] == []
    for tier in ["tolerant", "moderate", "strict"]:
        tokens = filter_defaults.loc[tier, "perturbing-cosolvents"]
        assert "TFE" in tokens
        assert "trifluoroethanol" in tokens


def test_dmso_added_from_tolerant_up():
    # 46 % of the percent-unit DMSO components deposited corpus-wide (66/143)
    # are >= 95 % v/v -- neat DMSO, referenced against DMSO-d6 rather than DSS.
    # 23 such rows were shipping in the tolerant tier.
    assert "DMSO" not in filter_defaults.loc["unfiltered", "perturbing-cosolvents"]
    for tier in ["tolerant", "moderate", "strict"]:
        assert "DMSO" in filter_defaults.loc[tier, "perturbing-cosolvents"]


def test_hfip_added_from_tolerant_up():
    # Hexafluoroisopropanol is a STRONGER helix inducer than TFE (Hirota, Mizuno
    # & Goto 1998, JMB 275:365). 22 entries deposit it, all >= 25 % v/v; six rows
    # were reaching the strict tier while 5 % TFE was excluded.
    assert filter_defaults.loc["unfiltered", "perturbing-cosolvents"] == []
    for tier in ["tolerant", "moderate", "strict"]:
        tokens = filter_defaults.loc[tier, "perturbing-cosolvents"]
        assert "HFIP" in tokens
        assert "hexafluoroisopropanol" in tokens
        assert "hexafluoro-2-propanol" in tokens


@pytest.mark.parametrize("tier", TIERS)
def test_bare_hexafluoro_is_never_a_token(tier):
    # bmr7375 deposits 'tetrakis(acetonitrile)copper(I) hexafluorophosphate' at
    # 1.8/4.2 mM -- a Cu(I) source, not a co-solvent. A bare 'hexafluoro' token
    # removes that row too (verified: +1 row lost at tolerant and at moderate).
    assert "hexafluoro" not in filter_defaults.loc[tier, "perturbing-cosolvents"]


@pytest.mark.parametrize("tier", TIERS)
def test_bare_dimethyl_is_never_a_token(tier):
    # 'dimethyl' matches DSS -- '2,2-dimethyl-2-silapentane-5-sulfonate' -- the
    # chemical-shift reference standard present in a large fraction of samples.
    assert "dimethyl" not in filter_defaults.loc[tier, "perturbing-cosolvents"]


@pytest.mark.parametrize("tier", ["tolerant", "moderate", "strict"])
def test_guanidinium_acronyms_are_kept_as_insurance(tier):
    # Neither token removes a single row that another filter does not already
    # remove (0 exclusive removals at every tier); they are kept against a future
    # deposition that spells the salt without the word 'guanidine'.
    tokens = filter_defaults.loc[tier, "perturbing-cosolvents"]
    assert "GdmCl" in tokens
    assert "Gdn-Hcl" in tokens


@pytest.mark.parametrize("tier", TIERS)
def test_membrane_mimetics_never_filter(tier):
    # Decision D3: annotated, not filtered. The rows they would remove are
    # 93.7 % ordered and contain zero disordered chains.
    tokens = [t.lower() for t in filter_defaults.loc[tier, "perturbing-cosolvents"]]
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


@pytest.mark.parametrize(
    "name",
    [
        "hexafluoroisopropanol",  # 16477, 18227, 19262, 19581, 19582, 34650-3
        "Hexafluoroisopropanol",  # 34650
        "hexafluoroisopropanol-d2",  # 6554
        "1,1,1,3,3,3-hexafluoro-2-propanol",  # 5257
        "HFIP",  # 15145, 16672, 26035, 34132, 34133, 36512, 51531, 6741
        "40% hexafluoroisopropanol (HFIP) aqueous solution",  # 18408-18410
    ],
)
def test_every_deposited_hfip_spelling_is_matched(name):
    tokens = list(filter_defaults.loc["tolerant", "perturbing-cosolvents"])
    row = _fill(_entry_with(name), tokens)
    assert any(row[token] for token in tokens)


def test_hexafluorophosphate_is_not_hfip():
    tokens = list(filter_defaults.loc["strict", "perturbing-cosolvents"])
    row = _fill(
        _entry_with("tetrakis(acetonitrile)copper(I) hexafluorophosphate"), tokens
    )
    assert not any(row[token] for token in tokens)


def test_hfip_is_matched_on_word_boundaries():
    # The acronym is matched with the same boundary discipline as `urea`; the
    # parenthesised and deuterated spellings must still hit. No corpus name
    # embeds 'hfip' in a longer token today (all 14 hits are genuine HFIP), so
    # the negative below is synthetic -- the boundary is insurance.
    assert (
        _fill(
            _entry_with("40% hexafluoroisopropanol (HFIP) aqueous solution"), ["HFIP"]
        )["HFIP"]
        is True
    )
    assert _fill(_entry_with("HFIP-d2"), ["HFIP"])["HFIP"] is True
    assert _fill(_entry_with("xhfipy"), ["HFIP"])["HFIP"] is False


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("urea", True),
        ("8 M urea", True),
        ("Urea-d4", True),  # the only other deposited spelling
        ("urea, 8 M", True),
        ("bis-pyridylurea inhibitor", False),  # bmr26598
        ("palmitate, laureate, and stearate", False),  # bmr50434
        ("2,5-methylenisothiourea-PXY", False),  # bmr11053
        ("urease", False),
    ],
)
def test_urea_is_matched_on_word_boundaries(name, expected):
    assert _fill(_entry_with(name), ["urea"])["urea"] is expected


def test_trifluoro_ethanol_with_a_space_needs_its_own_token():
    # bmr15559/15579/15580 deposit 'Trifluoro Ethanol D2OH' at 50 % v/v, which
    # neither 'TFE' nor 'trifluoroethanol' contains.
    tokens = ["TFE", "trifluoroethanol", "trifluoro ethanol"]
    row = _fill(_entry_with("Trifluoro Ethanol D2OH"), tokens)
    assert row["TFE"] is False
    assert row["trifluoroethanol"] is False
    assert row["trifluoro ethanol"] is True


@pytest.mark.parametrize(
    ("name", "token"),
    [
        ("Dimethyl sulfoxide", "dimethyl sulfoxide"),  # bmr7387 (100 %), 7389 (50 %)
        ("dimethyl sulfoxide", "dimethyl sulfoxide"),  # bmr5627 (5 %)
        ("dimethylsulfoxide", "dimethylsulfoxide"),  # bmr7388 (50 %)
    ],
)
def test_spelled_out_dmso_needs_its_own_tokens(name, token):
    tokens = ["DMSO", "dimethyl sulfoxide", "dimethylsulfoxide"]
    row = _fill(_entry_with(name), tokens)
    assert row["DMSO"] is False
    assert row[token] is True


@pytest.mark.parametrize(
    "name",
    [
        "2,2-dimethyl-2-silapentane-5-sulfonate",
        "4,4-dimethyl-4-silapentane-1-sulfonic acid (DSS)",
        "5,5-dimethylsilapentanesulfonate",
        "sodium 2,2-dimethyl-2-silapentane-5-sulfonate",
    ],
)
@pytest.mark.parametrize("tier", TIERS)
def test_dss_is_never_flagged_by_any_tier_token(tier, name):
    tokens = list(filter_defaults.loc[tier, "perturbing-cosolvents"])
    row = _fill(_entry_with(name), tokens)
    assert not any(row[token] for token in tokens)


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
    # not a bulk cosolvent.
    entry = _Entry(
        {
            "1": _Sample(
                [_component("4-guanidinophenyl 4-guanidinobenzoate", entity_id="2")]
            )
        }
    )
    assert _fill(entry, ["guanidin"])["guanidin"] is False


def test_absent_cosolvent_stays_false():
    assert _fill(_entry_with("sodium phosphate"), ["urea"])["urea"] is False


# --------------------------------------------------------------------------- #
# matcher defects
# --------------------------------------------------------------------------- #


def test_sample_fallback_applies_to_every_cosolvent():
    # No sample is referenced by the shift table, so every token must fall back
    # to all of the entry's samples -- not just the ones after the first
    # iteration of the loop that used to assign the fallback.
    entry = _Entry({"1": _Sample([_component("urea")])}, referenced=())
    row = _fill(entry, ["urea", "guanidin", "DMSO"])
    assert row["urea"] is True
    assert row["guanidin"] is False


def test_sample_fallback_applies_with_an_empty_cosolvent_list():
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


def _fill_real(entry_id, stID, cosolvents):
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
        row, perturbing_cosolvents=cosolvents, keywords=[], bmrb_entries=bmrb_entries
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
        # HFIP, at 25-80 % v/v in every entry that deposits it
        ("19262", "1", "hexafluoroisopropanol", True),  # gp41 MPER, 25 %
        ("6554", "1", "hexafluoroisopropanol", True),  # -d2 spelling, 30 %
        ("5257", "1", "hexafluoro-2-propanol", True),  # 1,1,1,3,3,3- spelling, 40 %
        ("26035", "1", "HFIP", True),  # bare acronym, 25 % v/v
        ("36512", "1", "HFIP", True),  # bare acronym, 40 %
        # ... but not the Cu(I) counterion that shares the 'hexafluoro' prefix
        ("7375", "1", "hexafluoroisopropanol", False),
        ("7375", "1", "HFIP", False),
        # urea, matched on word boundaries
        ("26598", "1", "urea", False),  # 'bis-pyridylurea inhibitor', 1 mM
        ("50434", "1", "urea", False),  # 'palmitate, laureate, and stearate'
        # the spellings a token used to miss
        ("15559", "1", "trifluoro ethanol", True),  # 'Trifluoro Ethanol D2OH', 50 %
        ("7387", "1", "dimethyl sulfoxide", True),  # 'Dimethyl sulfoxide', 100 %
        ("7388", "1", "dimethylsulfoxide", True),  # 'dimethylsulfoxide', 50 %
    ],
)
def test_real_entry_component_flags(entry_id, stID, token, expected):
    assert _fill_real(entry_id, stID, [token])[token] is expected


@requires_bmrb_data
@pytest.mark.parametrize("entry_id", ["19262", "19581", "19582", "26035", "51531"])
def test_gp41_mper_peptides_in_25_percent_hfip_are_excluded_at_strict(entry_id):
    """The five gp41 MPER depositions that reached the strict tier in 25 % HFIP.

    Excluding 5 % TFE while admitting 25 % HFIP was the largest single defect in
    the token list; bmr36512 (40 % HFIP) is the sixth strict row it removes.
    """
    tokens = list(filter_defaults.loc["strict", "perturbing-cosolvents"])
    row = _fill_real(entry_id, "1", tokens)
    assert any(row[token] for token in tokens)


@requires_bmrb_data
@pytest.mark.parametrize("entry_id", ["26598", "50434"])
def test_word_boundary_urea_readmits_the_two_false_positives(entry_id):
    """bmr26598 (a pyridylurea inhibitor) and bmr50434 (laureate) carry no urea."""
    tokens = list(filter_defaults.loc["tolerant", "perturbing-cosolvents"])
    row = _fill_real(entry_id, "1", tokens)
    assert not any(row[token] for token in tokens)


@requires_bmrb_data
def test_real_membrane_entry_is_annotated_not_filtered():
    strict_tokens = list(filter_defaults.loc["strict", "perturbing-cosolvents"])
    filled = _fill_real("11002", "1", strict_tokens)
    assert "SDS" in filled["membrane_mimetic"]
    assert not any(filled[token] for token in strict_tokens)


# --------------------------------------------------------------------------- #
# CLI surface: --perturbing-cosolvents and its deprecated alias
# --------------------------------------------------------------------------- #


def _run_score(monkeypatch, tmp_path, extra_argv):
    """Invoke the bare ``score`` command with the pipeline itself stubbed out.

    Returns ``(result, args)`` where ``args`` is the SimpleNamespace the CLI
    would have handed to ``run_scoring_pipeline``.
    """
    captured = {}

    def _fake_pipeline(args):
        captured["args"] = args

    monkeypatch.setattr("trizod.trizod.run_scoring_pipeline", _fake_pipeline)
    result = CliRunner().invoke(
        app,
        [
            "--input-dir",
            str(tmp_path),
            "--output-prefix",
            str(tmp_path / "out"),
            "--cache-dir",
            str(tmp_path / "cache"),
            *extra_argv,
        ],
    )
    return result, captured.get("args")


def test_perturbing_cosolvents_flag_sets_the_token_list(monkeypatch, tmp_path):
    result, args = _run_score(
        monkeypatch, tmp_path, ["--perturbing-cosolvents", "urea"]
    )
    assert result.exit_code == 0, result.output
    assert args.perturbing_cosolvents == ["urea"]


def test_deprecated_alias_still_works_and_warns(monkeypatch, tmp_path, caplog):
    """`--chemical-denaturants` is kept for one release, and must say so."""
    with caplog.at_level(logging.WARNING, logger="trizod"):
        result, args = _run_score(
            monkeypatch, tmp_path, ["--chemical-denaturants", "urea"]
        )
    assert result.exit_code == 0, result.output
    assert args.perturbing_cosolvents == ["urea"]
    assert "deprecated" in caplog.text
    assert "--perturbing-cosolvents" in caplog.text


def test_new_flag_wins_when_both_are_given(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="trizod"):
        _result, args = _run_score(
            monkeypatch,
            tmp_path,
            ["--chemical-denaturants", "urea", "--perturbing-cosolvents", "DMSO"],
        )
    assert args.perturbing_cosolvents == ["DMSO"]
    assert "deprecated" in caplog.text


def test_no_deprecation_warning_without_the_alias(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="trizod"):
        _result, args = _run_score(monkeypatch, tmp_path, [])
    # falls back to the tier default, which is the tolerant token list
    assert args.perturbing_cosolvents == list(
        filter_defaults.loc["tolerant", "perturbing-cosolvents"]
    )
    assert "deprecated" not in caplog.text


def test_help_advertises_the_new_flag_and_hides_the_alias():
    out = CliRunner().invoke(app, ["--help"]).output
    assert "--perturbing-cosolvents" in out
    assert "--chemical-denaturants" not in out
