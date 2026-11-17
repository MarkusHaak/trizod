"""Where the perturbing-cosolvent filter looks, and the alcohols it was missing.

Until 2026-10 the tier filter read only ``_Sample_component.Mol_common_name`` of
non-polymer components. ``_Sample.Solvent_system`` was not parsed at all and
``_Sample.Details`` fed only the ``cosolvent_evidence`` annotation, so a sample
declared as ``trifluoroethanol/water`` (bmr34568, a pinned test chain), one whose
10 % v/v DMSO is written only in its Details (bmr36154, pinned test), and one
whose component is misspelt ``Triflouroethanol`` (bmr50683, pinned test) all
passed the strict tier. Methanol and ethanol were in no list, so melittin in
100 % methanol (bmr26334) passed every cosolvent check.

Pinned here:

* ``bmrb.Sample`` parses ``_Sample.Solvent_system``;
* the filter searches Mol_common_name + Solvent_system + Details of the samples
  the scored shift table references (all samples only when it references none);
* the spelling-tolerant trifluoroethanol token;
* methanol/ethanol, word-boundary matched and guarded against substituted
  names (2-mercaptoethanol in every deposited spelling, ethanolamine, ...);
* two free-text guards: a denaturant named in a refolding protocol, and a
  negated mention ("no TFE"), are not a sample component;
* ``cosolvent_evidence`` sees the alcohols and the Solvent_system text too.
"""

import pynmrstar
import pytest

import trizod.bmrb.bmrb as bmrb
import trizod.bmrb.sample_state as sample_state
from tests._stubs import Entry as _Entry
from tests._stubs import Sample as _Sample
from tests._stubs import fill, fill_real
from tests.conftest import requires_bmrb_data
from trizod.trizod import (
    COSOLVENT_PATTERN_RES,
    COSOLVENT_TOKENS,
    WORD_BOUNDARY_COSOLVENTS,
    filter_defaults,
)

STRICT_TOKENS = list(filter_defaults.loc["strict", "perturbing-cosolvents"])


# --------------------------------------------------------------------------- #
# _Sample.Solvent_system is parsed
# --------------------------------------------------------------------------- #

_SAMPLE_SF = """\
save_sample_1
   _Sample.Sf_category                      sample
   _Sample.Sf_framecode                     sample_1
   _Sample.Entry_ID                         34568
   _Sample.ID                               1
   _Sample.Name                             .
   _Sample.Type                             solution
   _Sample.Sub_type                         .
   _Sample.Details                          '0.5 mM TNFa_TM_S34P, trifluoroethanol/water'
   _Sample.Aggregate_sample_number          .
   _Sample.Solvent_system                   {solvent}

   loop_
      _Sample_component.ID
      _Sample_component.Mol_common_name
      _Sample_component.Isotopic_labeling
      _Sample_component.Assembly_ID
      _Sample_component.Assembly_label
      _Sample_component.Entity_ID
      _Sample_component.Entity_label
      _Sample_component.Concentration_val
      _Sample_component.Concentration_val_units
      _Sample_component.Entry_ID
      _Sample_component.Sample_ID

      1   TNFa_TM_S34P   'natural abundance'   .   .   1   $entity_1   0.5   mM   34568   1
   stop_
save_
"""


def _parse_sample(solvent):
    return bmrb.Sample(
        pynmrstar.Saveframe.from_string(_SAMPLE_SF.format(solvent=solvent))
    )


def test_sample_parses_solvent_system():
    sample = _parse_sample("trifluoroethanol/water")
    assert sample.solvent_system == "trifluoroethanol/water"
    # ... and keeps Details, which the filter now reads as well
    assert sample.details == "0.5 mM TNFa_TM_S34P, trifluoroethanol/water"


def test_sample_parses_a_quoted_solvent_system():
    assert _parse_sample("'90% H2O/10% DMSO'").solvent_system == "90% H2O/10% DMSO"


def test_null_solvent_system_is_empty():
    # get_tag_vals maps the NMR-STAR null '.' to "" -- never the string '.'
    assert _parse_sample(".").solvent_system == ""


# --------------------------------------------------------------------------- #
# stand-ins for fill_row_data
# --------------------------------------------------------------------------- #


def _component(name, entity_id="", conc="10", units="%"):
    return ("1", "1", entity_id, name, conc, units)


def _fill(entry, cosolvents=STRICT_TOKENS):
    return fill(entry, cosolvents)


def _flagged(row, tokens=STRICT_TOKENS):
    return sorted(token for token in tokens if row[token])


def _one_sample(**kwargs):
    return _Entry({"1": _Sample(**kwargs)})


# --------------------------------------------------------------------------- #
# Solvent_system and Details reach the tier filter
# --------------------------------------------------------------------------- #


def test_solvent_system_is_searched():
    # bmr34568 (pinned test): no non-polymer component at all
    row = _fill(_one_sample(solvent_system="trifluoroethanol/water"))
    assert row["trifluoroethanol"] is True


def test_details_are_searched():
    # bmr36154 (pinned test): the 10 % DMSO is written only in Details
    details = (
        "0.7 mM [U-99% 13C; U-99% 15N] Txn09, 50 mM sodium chloride, 20 mM "
        "sodium phosphate, 10 % v/v Deuteration DMSO, 90% H2O/10% D2O"
    )
    row = _fill(
        _one_sample(
            components=[_component("sodium chloride", conc="50", units="mM")],
            solvent_system="90% H2O/10% D2O",
            details=details,
        )
    )
    assert _flagged(row) == ["DMSO"]


@pytest.mark.parametrize(
    "solvent",
    [
        "50%H2O/50%TFE",  # bmr19859, strict
        "80% TFE-d2, 20% H2O",
        "95% H2O: 5% DMSO",
        "DMSO-d6",
        "hexafluoroisopropanol/water",
        "50% HFIP/50% H2O",
        "8 M urea pH 2.5 10% D2O",
        "90% H2O/10% D2O, 50 mM sodium phosphate pH 6.5, 100 mM NaCl, "
        "6 M guanidine hydrochloride",
        "methanol",  # pick-list value, 26 entries
        "45% Methanol/45% H2O/10% D20",  # bmr18611
        "CD3OH",
        "33% CD3OD/6% D2O/61% H2O",
        "ethanol/water",
        "chloroform-d/ethanol-d5 5:1",
        "50%H2O/50%CF3CD2OH",  # bmr15937, TFE written as its formula
    ],
)
def test_deposited_cosolvent_solvent_systems_are_flagged(solvent):
    assert _flagged(_fill(_one_sample(solvent_system=solvent)))


@pytest.mark.parametrize(
    "solvent",
    [
        "90% H2O/10% D2O",
        "100% D2O",
        "95% H2O/5% D2O",
        "aqueous buffer",
        "10% H2O/ 40%D2O/ 50% d8-glycerol",  # an osmolyte, deliberately kept
        "",
        None,
    ],
)
def test_aqueous_solvent_systems_are_not_flagged(solvent):
    assert _flagged(_fill(_one_sample(solvent_system=solvent))) == []


def test_sample_name_is_still_not_a_cosolvent_field():
    # Only Mol_common_name, Solvent_system and Details are searched; the name is
    # sample-descriptive text for the keyword filter, not a component list.
    assert _flagged(_fill(_one_sample(name="13C 15N CC3 17.5%TFE"))) == []


def test_details_and_solvent_system_feed_cosolvent_evidence_too():
    assert _fill(_one_sample(solvent_system="trifluoroethanol/water"))[
        "cosolvent_evidence"
    ]
    assert _fill(_one_sample(details="dissolved in 100% CD3OH"))["cosolvent_evidence"]
    assert not _fill(_one_sample(solvent_system="90% H2O/10% D2O"))[
        "cosolvent_evidence"
    ]


# --------------------------------------------------------------------------- #
# multi-sample entries: the samples of THIS row's shift table
# --------------------------------------------------------------------------- #


def _two_samples(referenced):
    return _Entry(
        {
            "1": _Sample(solvent_system="90% H2O/10% D2O", details="aqueous"),
            "2": _Sample(solvent_system="trifluoroethanol/water", details="30% TFE"),
        },
        referenced=referenced,
    )


def test_unreferenced_samples_do_not_flag_the_row():
    # bmr15936: ten shift tables, one per TFE concentration; the 0 % one must
    # not inherit its siblings' TFE.
    assert _flagged(_fill(_two_samples(referenced=("1",)))) == []


def test_referenced_sample_flags_the_row():
    assert _flagged(_fill(_two_samples(referenced=("2",)))) == [
        "TFE",
        "trifluoroethanol",
    ]


def test_any_referenced_sample_flags_the_row():
    assert _flagged(_fill(_two_samples(referenced=("1", "2"))))


def test_shift_table_without_sample_references_falls_back_to_every_sample():
    # The entry-level fallback, unchanged: nothing says which sample was used.
    assert _flagged(_fill(_two_samples(referenced=())))


# --------------------------------------------------------------------------- #
# spelling-tolerant trifluoroethanol
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "trifluoroethanol",
        "Triflouroethanol",  # bmr50683 (pinned test), 17.5 % v/v
        "Trifluoro Ethanol D2OH",  # bmr15559/15579/15580, 50 %
        "Trifluoro ethanol D2OH",
        "trifluoro-ethanol",
        "trifluorethanol",  # bmr26679, 50 %
        "trifluroethanol",
        "2,2,2-trifluoroethanol-d3",
        "2.2.2 trifluoroethanol-d2",
        "d3-trifluoroethanol",
        "TRIFLUOROETHANOL-D2",
        "trifluethanol",  # bmr5837, 30 % -- no 'r' at all
        "CF3CD2OH",  # bmr4723/4724, 50 % -- the formula
        "CF3CH2OH",
    ],
)
def test_every_trifluoroethanol_spelling_is_matched(name):
    row = _fill(_one_sample(components=[_component(name)]))
    assert row["trifluoroethanol"] is True


@pytest.mark.parametrize(
    "name",
    [
        "trifluoroacetic acid",
        "TFA",
        "sodium trifluoroacetate",
        "ethanol",
        "CF3COOH",  # TFA again, as its formula
        "2,2,2-trifluoroethylamine",
    ],
)
def test_fuzzy_trifluoroethanol_is_not_overbroad(name):
    row = _fill(_one_sample(components=[_component(name)]))
    assert row["trifluoroethanol"] is False


def test_spaced_trifluoro_ethanol_token_is_subsumed():
    # The literal 'trifluoro ethanol' token existed only because neither other
    # TFE token reached that spelling; the fuzzy token does.
    assert "trifluoro ethanol" not in COSOLVENT_TOKENS
    assert "trifluoroethanol" in COSOLVENT_TOKENS


# --------------------------------------------------------------------------- #
# alcohols
# --------------------------------------------------------------------------- #

ALCOHOL_TOKENS = ["methanol", "ethanol", "CD3OH", "CD3OD", "MeOH", "EtOH"]


def test_alcohols_are_tier_tokens_from_tolerant_up():
    assert filter_defaults.loc["unfiltered", "perturbing-cosolvents"] == []
    for tier in ["tolerant", "moderate", "strict"]:
        tokens = filter_defaults.loc[tier, "perturbing-cosolvents"]
        for token in ALCOHOL_TOKENS:
            assert token in tokens, (tier, token)


def test_alcohols_are_word_boundary_tokens():
    # a whole word, or a guarded pattern (methanol, ethanol), never a substring
    for token in ALCOHOL_TOKENS:
        t = token.lower()
        assert t in WORD_BOUNDARY_COSOLVENTS or t in COSOLVENT_PATTERN_RES


@pytest.mark.parametrize(
    ("name", "token"),
    [
        ("methanol", "methanol"),  # bmr26334 melittin, 100 %
        ("Methanol", "methanol"),
        ("D3-methanol", "methanol"),  # bmr4136
        ("methanol-d4", "methanol"),
        ("methanol (CD3OH)", "methanol"),  # bmr25962
        ("CD3OH", "CD3OH"),
        ("Cd3OH", "CD3OH"),  # bmr17537
        ("CD3OD", "CD3OD"),
        ("CHCl3:CD3OH:H2O (4:4:1)", "CD3OH"),  # bmr6021
        ("MeOH", "MeOH"),  # bmr5580-5582, 37 %
        ("ethanol", "ethanol"),
        ("ethanol-d5", "ethanol"),  # bmr17153
        ("[2H5]-ethanol", "ethanol"),  # bmr6892
        ("ethanol_d6", "ethanol"),
        ("EtOH", "EtOH"),  # bmr51165 samples 2 and 3 -- MeOH's sibling
    ],
)
def test_alcohol_spellings_are_matched(name, token):
    assert _fill(_one_sample(components=[_component(name)]))[token] is True


@pytest.mark.parametrize(
    "name",
    [
        "2-mercaptoethanol",
        "beta-mercaptoethanol",
        "beta-mercapto-ethanol",  # bmr26700
        "B mercapto ethanol",  # bmr27165
        "2-mercapto ethanol",  # bmr30782
        "ethanolamine",
        "triethanolamine",
        "phosphatidylethanolamine",
        "2-bromo-ethanol",
        "2-amino ethanol",
        "2-(2-aminopyridin-1-ium-1-yl)ethanol",
        "(3-methylphenyl)methanol",
        "pyridin-3-ylmethanol",
        "trifluoroethanol",
        "Trifluoro Ethanol D2OH",
    ],
)
def test_substituted_names_are_not_ethanol_or_methanol(name):
    row = _fill(_one_sample(components=[_component(name)]))
    assert row["ethanol"] is False
    assert row["methanol"] is False


def test_etoh_is_a_whole_word():
    # 'acetohydroxamic acid' (bmr15361) contains e-t-o-h
    row = _fill(_one_sample(components=[_component("acetohydroxamic acid")]))
    assert row["EtOH"] is False
    assert _fill(_one_sample(details="5 % EtOH"))["EtOH"] is True


def test_methanol_is_not_ethanol():
    row = _fill(_one_sample(components=[_component("methanol")]))
    assert row["methanol"] is True
    assert row["ethanol"] is False


def test_mercaptoethanol_sample_is_not_flagged_at_all():
    row = _fill(
        _one_sample(
            components=[_component("2-mercapto ethanol", conc="2", units="mM")],
            details="Brd3 ET Domain, 100 mM sodium chloride, 20 mM Na3PO4, "
            "2 mM 2-mercapto ethanol, 90% H2O/10% D2O",
        )
    )
    assert _flagged(row) == []


@pytest.mark.parametrize(
    "texts",
    [
        ["methanol"],
        ["100% CD3OH"],
        ["CD3OD"],
        ["37% MeOH"],
        ["ethanol/water"],
        ["ethanol-d5"],
        ["Triflouroethanol"],
        ["trifluorethanol"],
        ["Trifluethanol"],
        ["50%H2O/50%CF3CD2OH"],
        ["EtOH"],
    ],
)
def test_alcohols_and_tfe_spellings_are_cosolvent_evidence(texts):
    assert sample_state.has_cosolvent_evidence(texts) is True


@pytest.mark.parametrize(
    "texts",
    [
        ["2-mercaptoethanol"],
        ["B mercapto ethanol"],
        ["beta-mercapto-ethanol"],
        ["ethanolamine"],
        ["(3-methylphenyl)methanol"],
        ["trifluoroacetic acid"],
        ["CF3COOH"],
        ["acetohydroxamic acid"],
    ],
)
def test_substituted_alcohols_are_not_cosolvent_evidence(texts):
    assert sample_state.has_cosolvent_evidence(texts) is False


def test_methanol_sample_now_carries_cosolvent_evidence():
    # bmr18611: ubiquitin in 45 % methanol, physical_state 'unfolded'. Without
    # evidence the ambiguous state was kept at moderate.
    row = _fill(
        _one_sample(
            components=[_component("methanol", conc="45")],
            solvent_system="45% Methanol/45% H2O/10% D20",
        )
    )
    assert row["cosolvent_evidence"] is True
    assert row["methanol"] is True


# --------------------------------------------------------------------------- #
# free-text guards
# --------------------------------------------------------------------------- #

_REFOLDING_PROTOCOL = (
    "Collected inclusion bodies were finally dissolved in 50 M Tris-HCl, "
    "pH 7.4, 100 mM NaCl, 6 M Guanidine hydrochloride, 200 mM dithiothreitol. "
    "Prior to refolding dithiothreitol was removed by dialysis against 6 M "
    "guanidine hydrochloride. NMR samples contained 1-2 mM IL-10 in 20 mM "
    "potassium phosphate, pH 6.2, 50 mM NaCl (95% H2O, 5% D2O)."
)


def test_denaturant_in_a_refolding_protocol_is_not_a_sample_component():
    # bmr19377 (pinned test, IL-10): the 6 M GdmCl is the inclusion-body
    # refolding protocol, the NMR sample is phosphate buffer.
    row = _fill(
        _one_sample(solvent_system="95% H2O/5% D2O", details=_REFOLDING_PROTOCOL)
    )
    assert _flagged(row) == []


def test_unfolding_refolding_step_is_a_protocol_too():
    # bmr50471: an H/D back-exchange step, not the NMR sample
    details = (
        "A mild unfolding-refolding step using guanidine hydrochloride was used "
        "to promote back-exchange of amide positions."
    )
    assert _flagged(_fill(_one_sample(details=details))) == []


def test_refolding_guard_never_hides_a_component():
    row = _fill(
        _one_sample(
            components=[_component("guanidine hydrochloride", conc="6", units="M")],
            details="refolded protein, then measured in 6 M guanidine hydrochloride",
        )
    )
    assert row["guanidin"] is True


def test_refolding_guard_is_limited_to_denaturants():
    # A helix inducer is not part of a refolding protocol; a TFE sample whose
    # Details also mention refolding is still a TFE sample.
    row = _fill(_one_sample(details="refolded by dilution; NMR in 30% TFE"))
    assert row["TFE"] is True


def test_denaturant_in_details_without_a_protocol_is_flagged():
    assert _fill(_one_sample(details="Im7 mutant in 6 M UREA"))["urea"] is True


@pytest.mark.parametrize(
    "details",
    [
        "no TFE (2,2,2-trifluoroehtanol)",  # bmr15936 sample 1, the 0 % TFE point
        "control sample without DMSO",
        "no urea",
    ],
)
def test_negated_mention_is_not_a_component(details):
    assert _flagged(_fill(_one_sample(details=details))) == []


def test_negation_only_strips_the_negated_word():
    assert _fill(_one_sample(details="no salt added, 30% TFE"))["TFE"] is True


# --------------------------------------------------------------------------- #
# real entries
# --------------------------------------------------------------------------- #


def _fill_real(entry_id, stID="1", cosolvents=STRICT_TOKENS):
    return fill_real(entry_id, stID, cosolvents)


@requires_bmrb_data
@pytest.mark.parametrize(
    ("entry_id", "stID", "expected"),
    [
        # pinned TriZOD test chains that were admitted in a cosolvent
        ("50683", "1", ["trifluoroethanol"]),  # 'Triflouroethanol', 17.5 % v/v
        ("34568", "1", ["trifluoroethanol"]),  # Solvent_system only
        ("36154", "1", ["DMSO"]),  # Details only, 10 % v/v
        # strict training representatives in TFE, Solvent_system/Details only
        ("19859", "1", ["TFE"]),  # 50 % TFE
        ("30481", "1", ["trifluoroethanol"]),  # melittin, 30 % TFE
        ("34011", "1", ["trifluoroethanol"]),
        ("34322", "1", ["trifluoroethanol"]),  # maximin 3
        # alcohols
        ("26334", "1", ["methanol"]),  # melittin, 100 % methanol
        ("18611", "1", ["methanol"]),  # ubiquitin, 45 % methanol, 2,500 bar
        # spellings the first fix still missed (found in review)
        ("5837", "1", ["trifluoroethanol"]),  # 'trifluethanol', 30 %
        ("15937", "1", ["trifluoroethanol"]),  # Solvent_system '50%H2O/50%CF3CD2OH'
        ("4723", "1", ["trifluoroethanol"]),  # component 'CF3CD2OH', 50 %
        ("51165", "2", ["EtOH"]),  # component 'EtOH'
    ],
)
def test_real_cosolvent_samples_are_flagged(entry_id, stID, expected):
    assert _flagged(_fill_real(entry_id, stID)) == expected


@requires_bmrb_data
@pytest.mark.parametrize(
    ("entry_id", "stID"),
    [
        ("19377", "1"),  # pinned test chain: GdmCl only in the refolding protocol
        ("50471", "1"),  # GdmCl only in an unfolding-refolding back-exchange step
        ("15936", "1"),  # the 'no TFE' point of a TFE titration
        ("26700", "1"),  # beta-mercapto-ethanol, 0.1 %
        ("27165", "1"),  # 'B mercapto ethanol', 2 mM
        ("30782", "1"),  # '2-mercapto ethanol', 2 mM
    ],
)
def test_real_false_positive_spellings_are_not_flagged(entry_id, stID):
    assert _flagged(_fill_real(entry_id, stID)) == []


@requires_bmrb_data
def test_real_titration_resolves_per_shift_table():
    """bmr15936: one shift table per TFE concentration, 0 % to 30 %."""
    assert _flagged(_fill_real("15936", "1")) == []
    assert _flagged(_fill_real("15936", "10")) == ["TFE", "trifluoroethanol"]


@requires_bmrb_data
def test_real_methanol_sample_corroborates_its_unfolded_state():
    row = _fill_real("18611")
    assert row["physical_state"] == "unfolded"
    assert row["cosolvent_evidence"] is True
