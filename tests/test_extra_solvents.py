"""Organic solvents the perturbing-cosolvent filter was missing (NMR-STAR audit I-07).

After the 2026-11 cosolvent fix (alcohols, TFE spellings, Solvent_system/Details)
acetonitrile, chloroform, the propanols and the alkanes still passed every tier:
three pinned TriZOD test chains (50871 and 50900, Solvent_system "50%
Acetonitrile/ 50% H2O"; 36521, 30 % deuterated acetonitrile), the CheZOD117 chain
18578 (30 % acetonitrile) and strict representatives such as 5633 (30 % CD3CN).
POTENCI and the LACS reference tables are aqueous, so none of these samples is in
their domain.

Pinned here, for the tier filter AND the ``cosolvent_evidence`` annotation (one
shared definition, ``sample_state.COSOLVENT_PATTERNS`` plus the word tokens):

* acetonitrile (incl. 'acetonitril', 'perdeuterated acetonitrile',
  'd3-acetonitrile', '2:1acetonitrile-d3'), CD3CN and ACN -- but not the Cu(I)
  ligand in 'tetrakis(acetonitrile)copper(I)', NaCN or 'Propionibacterium acnes';
* chloroform, CDCl3, CHCl3 -- but not a chloroformate;
* isopropanol, 2-propanol, isopropyl alcohol, n-propanol, 1-propanol -- but not
  hexafluoroisopropanol / hexafluoro-2-propanol (HFIP has its own tokens) or a
  mercapto-/amino-propanol;
* hexane and pentane -- but not DSS (2,2-dimethyl-2-SILAPENTANE-5-sulfonate, the
  shift reference), the MPD crystallisation additive (2-methyl-2,4-pentanediol,
  'methyl-2,4-pentane diol'), or the hexanediols;
* dioxane only when written as >= 1 % of the solvent: at 1-13 mM it is the
  1H/13C shift reference (bmr4831, 15447, 15565, 15973), not a cosolvent.
"""

import re

import pytest

import trizod.bmrb.sample_state as sample_state
from tests._stubs import Entry, Sample, fill, fill_real
from tests.conftest import requires_bmrb_data
from trizod.trizod import (
    COSOLVENT_PATTERN_RES,
    COSOLVENT_TOKENS,
    WORD_BOUNDARY_COSOLVENTS,
    _token_matches,
    filter_defaults,
)

STRICT_TOKENS = list(filter_defaults.loc["strict", "perturbing-cosolvents"])

NEW_TOKENS = [
    "acetonitrile",
    "CD3CN",
    "ACN",
    "chloroform",
    "CDCl3",
    "CHCl3",
    "isopropanol",
    "2-propanol",
    "isopropyl alcohol",
    "n-propanol",
    "1-propanol",
    "hexane",
    "pentane",
    "dioxane",
]


# --------------------------------------------------------------------------- #
# stand-ins for fill_row_data
# --------------------------------------------------------------------------- #


def _fill(**sample_kwargs):
    return fill(Entry({"1": Sample(**sample_kwargs)}), STRICT_TOKENS)


def _component(name, conc="30", units="%"):
    return ("1", "1", "", name, conc, units)


def _flagged(row):
    return sorted(token for token in STRICT_TOKENS if row[token])


# --------------------------------------------------------------------------- #
# token lists
# --------------------------------------------------------------------------- #


def test_new_solvents_are_tier_tokens_from_tolerant_up():
    assert filter_defaults.loc["unfiltered", "perturbing-cosolvents"] == []
    for tier in ["tolerant", "moderate", "strict"]:
        tokens = filter_defaults.loc[tier, "perturbing-cosolvents"]
        for token in NEW_TOKENS:
            assert token in tokens, (tier, token)


def test_new_solvents_are_word_boundary_or_pattern_tokens():
    # never a bare substring match: 'acn' is inside NaCN, 'pentane' inside DSS
    for token in NEW_TOKENS:
        assert token in COSOLVENT_TOKENS
        t = token.lower()
        assert t in WORD_BOUNDARY_COSOLVENTS or t in COSOLVENT_PATTERN_RES


# The token lists before they were derived from sample_state's
# ORGANIC_SOLVENT_TOKENS: WORD_BOUNDARY_COSOLVENTS then also listed the nine
# pattern-matched names, and the evidence word list "trifluoroethanol". Pinned so
# the derivation is shown to change no match.
_LEGACY_COSOLVENT_TOKENS = (
    "guanidin", "GdmCl", "Gdn-Hcl", "urea", "TFE", "trifluoroethanol",
    "hexafluoroisopropanol", "hexafluoro-2-propanol", "HFIP", "DMSO",
    "dimethyl sulfoxide", "dimethylsulfoxide", "methanol", "ethanol", "CD3OH",
    "CD3OD", "MeOH", "EtOH", "acetonitrile", "CD3CN", "ACN", "chloroform",
    "CDCl3", "CHCl3", "isopropanol", "2-propanol", "isopropyl alcohol",
    "n-propanol", "1-propanol", "hexane", "pentane", "dioxane",
)  # fmt: skip
_LEGACY_WORD_BOUNDARY_COSOLVENTS = frozenset(
    {
        "urea", "hfip", "methanol", "ethanol", "cd3oh", "cd3od", "meoh", "etoh",
        "acetonitrile", "cd3cn", "acn", "chloroform", "cdcl3", "chcl3",
        "isopropanol", "2-propanol", "isopropyl alcohol", "n-propanol",
        "1-propanol", "hexane", "pentane", "dioxane",
    }
)  # fmt: skip
_LEGACY_EVIDENCE_WORD_TOKENS = (
    "urea", "tfe", "trifluoroethanol", "hfip", "dmso", "sds", "gdn-hcl",
    "gdncl", "cd3oh", "cd3od", "meoh", "etoh", "cd3cn", "acn", "chloroform",
    "cdcl3", "chcl3", "isopropanol", "isopropyl alcohol",
)  # fmt: skip

# Every token, plain and in the contexts its guards exist for, plus decoys.
_MATCH_CORPUS = [
    text
    for token in _LEGACY_COSOLVENT_TOKENS + _LEGACY_EVIDENCE_WORD_TOKENS
    for text in (
        token,
        f"30% {token}",
        f"{token}-d3",
        f"{token}_d6",
        f"x{token}",
        f"{token}x",
        f"2-mercapto {token}",
        f"sila-{token}",
        f"({token})",
        f"tetrakis({token})copper(i)",
        f"5 % {token}",
        f"{token} (10 %)",
        f"no {token}",
    )
] + [
    "2,2-dimethyl-2-silapentane-5-sulfonate",
    "hexafluoroisopropanol-d2",
    "1,1,1,3,3,3-hexafluoro-2-propanol",
    "trifluoro ethanol d2oh",
    "triflouroethanol",
    "cf3cd2oh",
    "2,2,2-trifluoroethanol",
    "1 mm dioxane",
    "8% dioxane-d8",
    "nacn",
    "propionibacterium acnes",
    "urease",
    "palmitate, laureate, and stearate",
    "methyl-2,4-pentane diol",
    "acetohydroxamic acid",
    "50% acetonitrile/ 50% h2o",
]


def _legacy_rule(token):
    if token in COSOLVENT_PATTERN_RES:
        return "pattern"
    return "word" if token in _LEGACY_WORD_BOUNDARY_COSOLVENTS else "substring"


def _rule(token):
    if token in COSOLVENT_PATTERN_RES:
        return "pattern"
    return "word" if token in WORD_BOUNDARY_COSOLVENTS else "substring"


def test_derived_tier_token_list_is_unchanged():
    # same tokens, same spelling (they name the columns), same order
    assert COSOLVENT_TOKENS == _LEGACY_COSOLVENT_TOKENS
    assert set(sample_state.ORGANIC_SOLVENT_WORD_TOKENS) <= {
        t.lower() for t in COSOLVENT_TOKENS
    }


def test_dropped_word_boundary_entries_were_shadowed_by_patterns():
    # The pattern is checked first in _token_matches, so a word-boundary entry
    # for a pattern token was never read.
    dropped = _LEGACY_WORD_BOUNDARY_COSOLVENTS - WORD_BOUNDARY_COSOLVENTS
    assert len(dropped) == 9
    assert dropped <= set(COSOLVENT_PATTERN_RES)
    assert WORD_BOUNDARY_COSOLVENTS <= _LEGACY_WORD_BOUNDARY_COSOLVENTS
    for token in COSOLVENT_TOKENS:
        assert _rule(token.lower()) == _legacy_rule(token.lower()), token


def test_derived_word_boundary_set_matches_every_token_as_before():
    for token in _LEGACY_COSOLVENT_TOKENS:
        t = token.lower()
        for text in _MATCH_CORPUS:
            text = text.lower()
            new = _token_matches(
                t, [text], WORD_BOUNDARY_COSOLVENTS, COSOLVENT_PATTERN_RES
            )
            old = _token_matches(
                t, [text], _LEGACY_WORD_BOUNDARY_COSOLVENTS, COSOLVENT_PATTERN_RES
            )
            assert new == old, (token, text)


def test_cosolvent_evidence_matches_as_before_without_the_trifluoroethanol_word():
    legacy = re.compile(
        "|".join(
            [
                rf"(?<![a-z0-9]){re.escape(token)}"
                for token in sample_state._COSOLVENT_PREFIX_TOKENS
            ]
            + [
                rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])"
                for token in _LEGACY_EVIDENCE_WORD_TOKENS
            ]
            + list(sample_state.COSOLVENT_PATTERNS.values())
        ),
        re.IGNORECASE,
    )
    words = tuple(t for t in _LEGACY_EVIDENCE_WORD_TOKENS if t != "trifluoroethanol")
    assert words == sample_state._COSOLVENT_WORD_TOKENS
    for text in _MATCH_CORPUS + [t.upper() for t in _MATCH_CORPUS]:
        expected = legacy.search(text) is not None
        assert sample_state.has_cosolvent_evidence([text]) is expected, text


def test_patterned_solvents_share_one_definition_with_cosolvent_evidence():
    for token in ["acetonitrile", "2-propanol", "1-propanol", "n-propanol"]:
        assert token in sample_state.COSOLVENT_PATTERNS
    for token in ["hexane", "pentane", "dioxane"]:
        assert token in sample_state.COSOLVENT_PATTERNS
    for token, pattern in sample_state.COSOLVENT_PATTERNS.items():
        assert COSOLVENT_PATTERN_RES[token].pattern == pattern


# --------------------------------------------------------------------------- #
# component names (_Sample_component.Mol_common_name)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "token"),
    [
        ("acetonitrile", "acetonitrile"),  # bmr18578 (CheZOD117), 30 %
        ("Acetonitrile", "acetonitrile"),  # bmr16747, 40 %
        ("acetonitril", "acetonitrile"),  # bmr16452, 50 %
        ("perdeuterated acetonitrile", "acetonitrile"),  # bmr36457, 80 %
        ("d3-acetonitrile", "acetonitrile"),  # bmr25269, 38 %
        ("acetonitrile-d3", "acetonitrile"),
        ("CD3CN", "CD3CN"),  # bmr5633 (strict representative), 30 %
        ("deuterated ACN", "ACN"),  # bmr36521 (pinned test), 30 %
        ("ACN", "ACN"),  # bmr36522
        ("chloroform", "chloroform"),  # bmr21070-21073, 100 %
        ("Chloroform-d", "chloroform"),  # bmr5283
        ("D-chloroform", "chloroform"),  # bmr4136
        ("chloroform-d:methanol-d3 [1:1 (v/v)]", "chloroform"),  # bmr6349
        ("CDCl3", "CDCl3"),  # bmr4146, 19216
        ("CDCL3", "CDCl3"),  # bmr20102
        ("CHCl3:CD3OH:H2O (4:4:1)", "CHCl3"),  # bmr6021
        ("isopropanol", "isopropanol"),  # bmr53330, 12.5 %
        ("Isopropanol", "isopropanol"),  # bmr16873, 30 %
        ("isopropanol-d8", "isopropanol"),
        ("isopropyl alcohol", "isopropyl alcohol"),  # bmr15156
        ("Isopropyl alcohol", "isopropyl alcohol"),  # bmr17810, 25 %
        ("2-propanol", "2-propanol"),  # bmr19216, 50 %
        ("n-propanol", "n-propanol"),
        ("1-propanol", "1-propanol"),
        ("pentane", "pentane"),  # bmr25640, pentane reverse micelle
        ("d-pentane", "pentane"),  # bmr5387, 98 %
        ("n-pentane-d12", "pentane"),
        ("hexane", "hexane"),
        ("n-hexane", "hexane"),
    ],
)
def test_solvent_component_spellings_are_matched(name, token):
    row = _fill(components=[_component(name)])
    assert row[token] is True


@pytest.mark.parametrize(
    "name",
    [
        # the shift references: DSS in every deposited spelling, DSA, TSP, TMS
        "2,2-dimethyl-2-silapentane-5-sulfonate",
        "sodium 2,2-dimethyl-2-silapentane-5-sulfonate",  # bmr30263 (sic, 5 %)
        "2,2-dimethyl-2-silapentane sulfonic acid(DSS)",  # bmr6894
        "2,2-dimethyl-2-silapentane-5-sulfonic_acid_(DSS)",  # bmr5975
        "4,4-dimethyl-4-silapentane-1-sulfonic acid (DSS)",  # bmr19769
        "DSS (4,4-dimethyl-4-silapentane-1-sulfonic acid)",  # bmr31073
        "5,5-dimethylsilapentanesulfonate",  # bmr7231
        "2,2-dimethyl-2-sila-pentane-5-sulfonate",
        "dimethylsila pentane sulfonate",
        "4,4-dimethyl-4-silapentane-1-ammonium trifluoroacetate (DSA)",  # bmr31209
        "sodium 3-(trimethylsilyl)propionate-2,2,3,3-d4",
        "tetramethylsilane",
        # dioxane at reference concentration -- a component name carries no %
        "dioxane",  # bmr15447 (1 mM), bmr15565 (13 mM)
        "1-4 dioxane",  # bmr15973 (1 mM)
        "1,4-dioxane",
        # acetonitrile as a ligand, cyanide, a species name
        "tetrakis(acetonitrile)copper(I) hexafluorophosphate",  # bmr7375, 1.8 mM
        "NaCN",  # bmr19740
        "Propionibacterium acnes extract",
        # chloroformate is a reagent, not chloroform
        "ethyl chloroformate",
        # MPD and the diols are not alkanes
        "2-methyl-2,4-pentanediol",  # bmr53330
        "2-methylpentane-2,4-diol",  # bmr15283
        "(4R)-2-Methylpentane-2,4-diol",  # bmr15156
        "methyl-2,4-pentane diol",  # bmr30088, 45 %
        "Methyl Pentane diol",  # bmr15156, 16873
        "1,5-Pentanediol",  # bmr52913
        "pentane-1,5-diol",
        "1,6-hexanediol",  # bmr51102
        "hexane-1,6-diol",
        "1,2-cyclohexanediol",  # bmr52913
        "1,2-DIHEXANEOYL-SN-GLYCERO-3-PHOSPHOCHOLINE",  # bmr25488
        "N-[4-(p-maleimidomethyl)cyclohexane-carboxamide]",  # bmr25114
        "n-hexadecane-D34",
        # substituted propanols; HFIP is flagged by its own tokens, not these
        "3-mercapto-1-propanol",
        "3-amino-1-propanol",
        "diisopropylfluorophosphatase",
        "isopropyl-beta-D-thiogalactopyranoside",
    ],
)
def test_reference_compounds_and_lookalikes_are_not_new_solvents(name):
    row = _fill(components=[_component(name, conc="1", units="mM")])
    assert [token for token in NEW_TOKENS if row[token]] == []


@pytest.mark.parametrize(
    "name",
    [
        "hexafluoroisopropanol",
        "hexafluoroisopropanol-d2",
        "1,1,1,3,3,3-hexafluoro-2-propanol",
        "hexafluoro-2-propanol",
    ],
)
def test_hfip_is_not_read_as_a_propanol(name):
    row = _fill(components=[_component(name)])
    assert [token for token in NEW_TOKENS if row[token]] == []
    assert _flagged(row)  # ... it is still HFIP


@pytest.mark.parametrize(
    "name",
    [
        "2,2-dimethyl-2-silapentane-5-sulfonate",
        "4,4-dimethyl-4-silapentane-1-sulfonic acid (DSS)",
        "dioxane",
        "tetrakis(acetonitrile)copper(I) hexafluorophosphate",
        "methyl-2,4-pentane diol",
    ],
)
def test_reference_compounds_flag_nothing_at_strict(name):
    assert _flagged(_fill(components=[_component(name, conc="1", units="mM")])) == []


# --------------------------------------------------------------------------- #
# _Sample.Solvent_system and _Sample.Details
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("solvent", "token"),
    [
        ("50% Acetonitrile/ 50% H2O", "acetonitrile"),  # bmr50871, 50900 (test)
        ("60% H2O/10% D2O/30% deuterated ACN", "ACN"),  # bmr36521 (test)
        ("70% D2O/30% deuterated ACN", "ACN"),  # bmr36521 (test)
        ("69.9% H2O / 30% ACN / 0.1% TFA", "ACN"),  # bmr27212
        ("50% H2O/50% ACN-d3", "ACN"),  # bmr31109
        ("70%H2O/30%CD3CN", "CD3CN"),  # bmr25804
        ("71%water/10%D2O/19%CD3CN", "CD3CN"),  # bmr25506
        ("50% H2O / 50% acetonitril (v/v)", "acetonitrile"),  # bmr16452
        ("10% V/V ACETONITRILE/WATER", "acetonitrile"),  # bmr19157
        ("aqueous 30 % acetonitrile, 0.1 % trifluoroacetic acid", "acetonitrile"),
        ("chloroform", "chloroform"),  # pick-list value, 15 entries
        ("44%CDCl3/44%Cd3OH/12%H2O", "CDCl3"),  # bmr17537
        ("CDCl3/2-propanol", "2-propanol"),  # bmr19216
        ("CDCl3/2-propanol", "CDCl3"),
        ("Deuterated pentane", "pentane"),  # bmr25640
        ("95% H2O/5% D2O/5% dioxane-d6", "dioxane"),  # bmr25887
    ],
)
def test_solvent_systems_are_matched(solvent, token):
    assert _fill(solvent_system=solvent)[token] is True


@pytest.mark.parametrize(
    ("details", "token"),
    [
        # bmr4411: dioxane written as a solvent fraction
        ("water mixture of 8% dioxane-d8 and 3% methanol-d3.", "dioxane"),
        ("5 % v/v 1,4-dioxane", "dioxane"),
        ("dioxane-d8 (10 %)", "dioxane"),
        # bmr52500: no separator between the ratio and the solvent
        ("~2 mg in 2:1acetonitrile-d3 and H2O", "acetonitrile"),
        (
            "Oxidized cytochrome c dissolved in 30% acetonitrile-70% aqueous",
            "acetonitrile",
        ),
        ("1 mM drp1, 80% perdeuterated acetonitrile and 20% H2O", "acetonitrile"),
    ],
)
def test_details_are_matched(details, token):
    assert _fill(details=details)[token] is True


@pytest.mark.parametrize(
    "details",
    [
        # bmr4831: dioxane added as the 1H shift reference
        "pH 6.5 (93%/7% H2O/D2O) with a small amount of dioxane for 1H chemical "
        "shift referencing",
        "1 mM dioxane, 10% D2O, 90% H2O",
        "0.1 % dioxane as internal reference",
        "10% D2O, 0.1 mM DSS, 20 mM sodium phosphate",
        "0.01 % w/v DSS, 90% H2O/10% D2O",
        # the 10 % belongs to D2O; the dioxane is the shift reference
        "90% H2O/10% D2O/dioxane",
        # bmr30088: MPD crystallisation, the alcohol is isopropyl alcohol
        "45 % methyl-2,4-pentane diol, 12.5 mM sodium phosphate",
    ],
)
def test_reference_mentions_in_details_are_not_matched(details):
    assert _flagged(_fill(details=details)) == []


def test_negated_mention_is_still_absent():
    # the cosolvent fix's "no X" guard covers the new tokens too
    assert _fill(details="no acetonitrile, 90% H2O/10% D2O")["acetonitrile"] is False


# --------------------------------------------------------------------------- #
# cosolvent_evidence -- the same tokens corroborate a 'denatured' label
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "50% Acetonitrile/ 50% H2O",
        "acetonitril",
        "30% deuterated ACN",
        "sample_13C15N_AcN_pH2",  # bmr51321, 28 % acetonitrile
        "70%H2O/30%CD3CN",
        "chloroform-d",
        "CDCl3",
        "CHCl3",
        "Isopropanol",
        "isopropyl alcohol",
        "CDCl3/2-propanol",
        "n-propanol",
        "1-propanol",
        "Deuterated pentane",
        "n-hexane",
        "8% dioxane-d8",
    ],
)
def test_new_solvents_are_cosolvent_evidence(text):
    assert sample_state.has_cosolvent_evidence([text]) is True


@pytest.mark.parametrize(
    "text",
    [
        "2,2-dimethyl-2-silapentane-5-sulfonate",
        "4,4-dimethyl-4-silapentane-1-sulfonic acid (DSS)",
        "2,2-dimethyl-2-sila-pentane-5-sulfonate",
        "dioxane",
        "a small amount of dioxane for 1H chemical shift referencing",
        "tetrakis(acetonitrile)copper(I) hexafluorophosphate",
        "NaCN",
        "Radical Oxygenase of Propionibacterium acnes",  # bmr34585
        "ACNDV Cp at pH 7.5",  # bmr51506
        "methyl-2,4-pentane diol",
        "1,6-hexanediol",
        "diisopropylfluorophosphatase",
        "isopropyl-beta-D-thiogalactopyranoside (IPTG)",
        "ethyl chloroformate",
    ],
)
def test_lookalikes_are_not_cosolvent_evidence(text):
    assert sample_state.has_cosolvent_evidence([text]) is False


def test_new_solvent_feeds_the_row_annotation():
    assert _fill(solvent_system="50% Acetonitrile/ 50% H2O")["cosolvent_evidence"]


# --------------------------------------------------------------------------- #
# real entries
# --------------------------------------------------------------------------- #


def _fill_real(entry_id, stID="1"):
    return fill_real(entry_id, stID, STRICT_TOKENS)


@requires_bmrb_data
@pytest.mark.parametrize(
    ("entry_id", "token"),
    [
        ("50871", "acetonitrile"),  # pinned TriZOD test chain
        ("50900", "acetonitrile"),  # pinned TriZOD test chain
        ("36521", "ACN"),  # pinned TriZOD test chain, 30 % deuterated ACN
        ("18578", "acetonitrile"),  # CheZOD117 chain, 30 % acetonitrile
        ("5633", "CD3CN"),  # strict representative, 30 % CD3CN
        ("36457", "acetonitrile"),  # 80 % perdeuterated acetonitrile
        ("19216", "2-propanol"),  # CDCl3/2-propanol
        ("19216", "CDCl3"),
        ("25640", "pentane"),  # pentane reverse micelle
        ("4411", "dioxane"),  # 8 % dioxane-d8
        ("25887", "dioxane"),  # 5 % dioxane-d6
    ],
)
def test_real_organic_solvent_samples_are_flagged(entry_id, token):
    row = _fill_real(entry_id)
    assert row[token] is True
    assert row["cosolvent_evidence"]


@requires_bmrb_data
@pytest.mark.parametrize(
    "entry_id",
    [
        "4831",  # "a small amount of dioxane for 1H chemical shift referencing"
        "15447",  # dioxane, 1 mM
        "15973",  # 1-4 dioxane, 1 mM
        "7375",  # tetrakis(acetonitrile)copper(I) hexafluorophosphate
        "19740",  # NaCN
        "5723",  # 2,2-dimethyl-2-silapentane-5-sulfonate
        "19769",  # 4,4-dimethyl-4-silapentane-1-sulfonic acid (DSS)
        "31209",  # DSA
    ],
)
def test_real_reference_compound_samples_are_not_flagged_by_new_tokens(entry_id):
    row = _fill_real(entry_id)
    assert [token for token in NEW_TOKENS if row[token]] == []
