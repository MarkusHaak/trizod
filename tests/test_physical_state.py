"""Row-level state filtering: keyword field scope and ``_Entity_assembly.Physical_state``.

Two mechanisms decide whether a row is dropped for being in a non-native state,
and they must not be confused with each other:

* the free-text ``keywords-blacklist``, which is **field-scoped**: by default it
  searches only the fields that describe the deposited *sample* and never the
  paper-topic fields (citation title/keywords, struct keywords). A study titled
  "... Amyloid Fibrils" that deposits a monomeric IDP is a paper about fibrils,
  not a tube full of them.
* ``_Entity_assembly.Physical_state``, the depositor's own label, matched
  **exactly** (never as a substring) against a per-tier deny list and resolved to
  the entity assembly of the row being filtered.

``intrinsically disordered`` / ``partially disordered`` are the signal the dataset
exists to capture and must never be denied.
"""

import logging

import pandas as pd
import pytest

import trizod.bmrb.bmrb as bmrb
import trizod.bmrb.sample_state as sample_state
from tests.conftest import BMRB_DIR, requires_bmrb_data
from trizod import paths
from trizod.bmrb.sample_state import (
    PHYSICAL_STATE_KEEP,
    is_physical_state_denied,
    unseen_physical_states,
    warn_unseen_physical_states,
)
from trizod.pipeline import prefilter_dataframe
from trizod.trizod import fill_row_data, filter_defaults

TOLERANT_DENY = filter_defaults.loc["tolerant", "physical-state-blacklist"]
MODERATE_DENY = filter_defaults.loc["moderate", "physical-state-blacklist"]
STRICT_DENY = filter_defaults.loc["strict", "physical-state-blacklist"]


class _Conditions:
    """Stand-in for ``bmrb.SampleConditions`` -- constant, valid conditions."""

    def get_ionic_strength(self, **kwargs):
        return 0.1

    def get_pH(self, **kwargs):
        return 7.0

    def get_temperature(self, **kwargs):
        return 298.0


class _Named:
    """Stand-in for Assembly/Entity/Sample: only the free-text fields matter."""

    def __init__(self, name=None, details=None, framecode=None, entities=None):
        self.name = name
        self.details = details
        self.framecode = framecode
        self.entities = entities if entities is not None else [("1", "1", "$e", None)]
        self.seq = None  # skips the backbone-shift block in fill_row_data
        self.paramagnetic = None
        self.components = []
        self.type = "solution"


class _Entry:
    def __init__(
        self,
        title=None,
        details=None,
        citation_title=None,
        citation_keywords=None,
        struct_keywords=None,
        assembly_entities=None,
        sample_name=None,
    ):
        self.id = "1"
        self.title = title
        self.details = details
        self.citation_title = citation_title
        self.citation_DOI = None
        self.exp_method = "NMR"
        self.exp_method_subtype = "solution"
        self.citation_keywords = citation_keywords
        self.struct_keywords = struct_keywords
        self.entities = {"1": _Named(name="entity name")}
        self.assemblies = {
            "1": _Named(name="assembly name", entities=assembly_entities)
        }
        self.samples = {"1": _Named(name=sample_name, framecode="sample_1")}
        self.conditions = {"1": _Conditions()}
        self.experiment_list = _ExperimentList()

    def get_peptide_shifts(self):
        return {
            ("1", "1", "1"): (None, "1", "1", ["1"]),
            ("1", "2", "1"): (None, "1", "1", ["1"]),
        }


class _ExperimentList:
    experiments = [("1", "2D 1H-15N HSQC", None, "1", None, "isotropic")]


def _fill(entry, keywords, entity_assemID="1", **kwargs):
    row = pd.Series(
        {"entryID": "1", "stID": "1", "entity_assemID": entity_assemID, "entityID": "1"}
    )
    bmrb_entries = pd.DataFrame({"entry": [entry]}, index=["1"])
    return fill_row_data(
        row,
        perturbing_cosolvents=[],
        keywords=keywords,
        bmrb_entries=bmrb_entries,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# keyword field scope
# --------------------------------------------------------------------------- #


def test_paper_topic_fields_ignored_under_the_sample_scope():
    entry = _Entry(
        citation_title="Structure of the X:Y complex in the unfolded state",
        citation_keywords=["protein misfolding"],
        struct_keywords=["denatured state"],
    )
    row = _fill(entry, ["unfold", "misfold", "denatur"], keyword_search_scope="sample")
    assert row["unfold"] is False
    assert row["misfold"] is False
    assert row["denatur"] is False


def test_paper_topic_fields_searched_under_the_all_scope():
    entry = _Entry(
        citation_title="Structure of the X:Y complex in the unfolded state",
        citation_keywords=["protein misfolding"],
        struct_keywords=["denatured state"],
    )
    row = _fill(entry, ["unfold", "misfold", "denatur"], keyword_search_scope="all")
    assert row["unfold"] is True
    assert row["misfold"] is True
    assert row["denatur"] is True


@pytest.mark.parametrize("scope", ["sample", "all"])
def test_sample_descriptive_fields_searched_in_both_scopes(scope):
    assert (
        _fill(
            _Entry(title="Denatured ubiquitin"), ["denatur"], keyword_search_scope=scope
        )["denatur"]
        is True
    )
    assert (
        _fill(
            _Entry(details="in 8 M urea, denatured"),
            ["denatur"],
            keyword_search_scope=scope,
        )["denatur"]
        is True
    )
    assert (
        _fill(
            _Entry(sample_name="denatured sample"),
            ["denatur"],
            keyword_search_scope=scope,
        )["denatur"]
        is True
    )


# --------------------------------------------------------------------------- #
# 'bound' is a whole word
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "title",
    [
        "Unbound Med25ACID",
        "Human Pdx1 Homeodomain in the Unbound State",
        "Solution structure human HCN2 CNBD in the cAMP-unbound state",
        "MAGI-1 PDZ1 with Noncanonical Domain Boundaries",
        "not determined disulfide bounded formation",
    ],
)
def test_bound_does_not_match_unbound_or_boundaries(title):
    assert _fill(_Entry(title=title), ["bound"])["bound"] is False


@pytest.mark.parametrize(
    "title",
    [
        "Bound state of the peptide",
        "membrane-bound form of X",
        "DNA-bound homeodomain",
        "X in the ATP bound state",
    ],
)
def test_bound_still_matches_genuine_complexes(title):
    assert _fill(_Entry(title=title), ["bound"])["bound"] is True


def test_prefix_keywords_stay_substring_matches():
    # 'denatur' and 'unfold' are deliberately prefixes, not whole words.
    assert _fill(_Entry(title="Denatured state of X"), ["denatur"])["denatur"] is True
    assert _fill(_Entry(title="the unfolded ensemble"), ["unfold"])["unfold"] is True


@pytest.mark.parametrize(
    ("title", "keyword"),
    [
        ("NMR of Sic1 in complex with Cdc4", "in complex with"),
        ("p53 TAD complexed with MDM2", "complexed with"),
    ],
)
def test_complex_phrases_match_sample_descriptive_text(title, keyword):
    assert _fill(_Entry(title=title), [keyword])[keyword] is True


def test_interacti_is_no_longer_a_strict_default():
    assert "interacti" not in filter_defaults.loc["strict", "keywords-blacklist"]
    assert "in complex with" in filter_defaults.loc["strict", "keywords-blacklist"]
    assert "complexed with" in filter_defaults.loc["strict", "keywords-blacklist"]


# --------------------------------------------------------------------------- #
# _Entity_assembly.Physical_state
# --------------------------------------------------------------------------- #


def test_physical_state_resolved_to_this_rows_entity_assembly():
    entities = [("1", "1", "$e", "native"), ("2", "1", "$e", "denatured")]
    entry = _Entry(assembly_entities=entities)
    assert _fill(entry, [], entity_assemID="1")["physical_state"] == "native"
    assert _fill(entry, [], entity_assemID="2")["physical_state"] == "denatured"


def test_physical_state_absent_is_na():
    entry = _Entry(assembly_entities=[("1", "1", "$e", None)])
    assert pd.isna(_fill(entry, [])["physical_state"])
    entry = _Entry(assembly_entities=[("1", "1", "$e", "")])
    assert pd.isna(_fill(entry, [])["physical_state"])


def test_physical_state_deny_is_exact_not_substring():
    # `denatured` is ambiguous vocabulary, so corroborate to isolate what this
    # test is about: exact matching, never substring matching.
    deny = ["denatured"]

    def denied(value):
        return is_physical_state_denied(value, deny, corroborated=True)

    assert denied("denatured") is True
    assert denied("Denatured") is True
    assert denied("  denatured ") is True
    assert denied("partially denatured") is False
    assert denied("not denatured") is False


def test_unfolded_denied_from_moderate_up_only():
    # `unfolded` is ambiguous vocabulary, so it is denied only where the entry
    # also names a perturbing cosolvent -- above moderate, and never at tolerant.
    assert (
        is_physical_state_denied("unfolded", TOLERANT_DENY, corroborated=True) is False
    )
    assert (
        is_physical_state_denied("unfolded", MODERATE_DENY, corroborated=True) is True
    )
    assert is_physical_state_denied("unfolded", STRICT_DENY, corroborated=True) is True
    # ... and never on the deposited tag alone, at any tier.
    for deny in (TOLERANT_DENY, MODERATE_DENY, STRICT_DENY):
        assert is_physical_state_denied("unfolded", deny) is False


@pytest.mark.parametrize(
    "value",
    [
        "denatured",
        "partially denatured",
        "misfolded",
        "non-native",
        "aggregated",
        "amyloid",
        "amyloid fibril",
        "amyloid fibrils",
        "fibril",
        "fibrils",
        "Fibrillar",
        "molten globule",
    ],
)
def test_tolerant_deny_covers_the_deposited_spelling_variants(value):
    # Ambiguous values need corroborating cosolvent evidence; the rest do not.
    corroborated = value.strip().lower() in sample_state.PHYSICAL_STATE_AMBIGUOUS
    assert (
        is_physical_state_denied(value, TOLERANT_DENY, corroborated=corroborated)
        is True
    )


@pytest.mark.parametrize(
    "value",
    [
        "bound",
        "micelle-bound",
        "SLAS micelle-bound",
        "Reconstituted",
        "Reconstituted in DPC",
    ],
)
def test_bound_states_denied_at_strict_only(value):
    assert is_physical_state_denied(value, MODERATE_DENY) is False
    assert is_physical_state_denied(value, STRICT_DENY) is True


@pytest.mark.parametrize("value", sorted(PHYSICAL_STATE_KEEP))
def test_keep_states_are_denied_by_no_tier(value):
    for deny in (TOLERANT_DENY, MODERATE_DENY, STRICT_DENY):
        assert is_physical_state_denied(value, deny) is False


def test_intrinsically_disordered_is_kept():
    assert "intrinsically disordered" in PHYSICAL_STATE_KEEP
    assert "partially disordered" in PHYSICAL_STATE_KEEP


# --------------------------------------------------------------------------- #
# the row-level guard: a KEEP state must survive a paper-topic-only keyword
# --------------------------------------------------------------------------- #


def test_keep_state_row_not_dropped_by_a_topic_only_keyword():
    # The bmr51322 shape: Abeta(1-42) whose own Physical_state is
    # 'intrinsically disordered', in a paper about amyloid fibrils. A set-level
    # test does not catch this -- only a row-level one does.
    entry = _Entry(
        title="Sequence-specific Backbone Resonance Assignments of Human Amyloid-beta(1-42) at pH 7.0",
        citation_title="Atomic Resolution Insights into pH Shift Induced Deprotonation Events in LS-Shaped Ab(1-42) Amyloid Fibrils",
        assembly_entities=[("1", "1", "$e", "intrinsically disordered")],
    )
    row = _fill(entry, ["amyloid fibril"], keyword_search_scope="sample")
    assert row["physical_state"] == "intrinsically disordered"
    assert row["physical_state"] in PHYSICAL_STATE_KEEP
    assert row["amyloid fibril"] is False
    # ... and the 'all' scope is exactly what used to drop it.
    assert (
        _fill(entry, ["amyloid fibril"], keyword_search_scope="all")["amyloid fibril"]
        is True
    )


# --------------------------------------------------------------------------- #
# vocabulary drift
# --------------------------------------------------------------------------- #


def test_unseen_physical_state_values_are_reported_above_the_threshold():
    values = ["native"] * 10 + ["cryo-trapped"] * 6 + ["one-off"] * 1
    unseen = unseen_physical_states(values, threshold=5)
    assert unseen == {"cryo-trapped": 6}


def test_unseen_physical_state_warns(caplog):
    values = ["quantum foam"] * 7
    with caplog.at_level(logging.WARNING, logger="trizod"):
        warn_unseen_physical_states(values, threshold=5)
    assert "quantum foam" in caplog.text


def test_known_physical_states_do_not_warn(caplog):
    values = ["native"] * 50 + ["denatured"] * 20 + ["na"] * 20
    with caplog.at_level(logging.WARNING, logger="trizod"):
        warn_unseen_physical_states(values, threshold=1)
    assert caplog.text == ""


# --------------------------------------------------------------------------- #
# prefilter integration
# --------------------------------------------------------------------------- #


def _state_frame(states, cosolvent_evidence=None):
    n = len(states)
    return pd.DataFrame(
        {
            "cosolvent_evidence": (
                [False] * n if cosolvent_evidence is None else cosolvent_evidence
            ),
            "exp_method": ["NMR"] * n,
            "exp_method_subtype": ["solution"] * n,
            "physical_state": states,
            "sample_state_evidence": ["solution"] * n,
            "temperature": [298.0] * n,
            "ionic_strength": [0.1] * n,
            "pH": [7.0] * n,
            "seq": ["A" * 20] * n,
            "total_bbshifts": [80] * n,
            "bbshift_types": [4] * n,
            "bbshift_positions": [18] * n,
        }
    )


def _pass_pre(states, deny, cosolvent_evidence=None):
    df, *_rest = prefilter_dataframe(
        _state_frame(states, cosolvent_evidence),
        method_whitelist=["solution"],
        method_blacklist=["solid"],
        temperature_range=[-float("inf"), float("inf")],
        ionic_strength_range=[0.0, float("inf")],
        pH_range=[-float("inf"), float("inf")],
        peptide_length_range=[5, float("inf")],
        min_backbone_shift_types=1,
        min_backbone_shift_positions=1,
        min_backbone_shift_fraction=0.0,
        max_noncanonical_fraction=1.0,
        max_x_fraction=1.0,
        keywords=[],
        perturbing_cosolvents=[],
        physical_state_blacklist=deny,
    )
    return df["pass_pre"].tolist()


def test_prefilter_drops_denied_physical_states():
    states = ["native", "molten globule", "intrinsically disordered", "denatured", None]
    # `molten globule` is denied on the deposited tag alone; `denatured` is
    # ambiguous vocabulary and survives unless the entry names a perturbing cosolvent.
    assert _pass_pre(states, TOLERANT_DENY) == [True, False, True, True, True]
    assert _pass_pre(
        states, TOLERANT_DENY, cosolvent_evidence=[False, False, False, True, False]
    ) == [True, False, True, False, True]


def test_prefilter_keeps_everything_when_the_deny_list_is_empty():
    states = ["native", "molten globule", "denatured", "unfolded"]
    assert _pass_pre(states, []) == [True, True, True, True]


# --------------------------------------------------------------------------- #
# real entries
# --------------------------------------------------------------------------- #


@requires_bmrb_data
@pytest.mark.parametrize(
    ("entry_id", "state"),
    [("5158", "molten globule"), ("5119", "molten globule"), ("16948", "denatured")],
)
def test_real_non_native_entries_are_denied_from_tolerant_up(entry_id, state):
    """5158 apo-myoglobin, 5119 ATP synthase subunit c, 16948 dynamin GED.

    All three are ``split=train, train_tier=moderate`` in the published v0.3.0
    deposit; 5158 scores 52 % of its residues as ordered. No filter has ever
    touched them -- ``Physical_state`` is the only signal that does.
    """
    entry = bmrb.BmrbEntry(entry_id, paths.RAW_BMRB / f"bmr{entry_id}")
    values = {row[3] for row in entry.assemblies["1"].entities}
    assert values == {state}
    # 16948 (dynamin GED in DMSO) carries the ambiguous value `denatured`, so it
    # is denied on its real cosolvent evidence, not on the tag alone.
    texts = [entry.title, entry.details] + [
        c[3] for s_ in entry.samples.values() for c in s_.components
    ]
    corroborated = sample_state.has_cosolvent_evidence(texts)
    assert corroborated is (state in sample_state.PHYSICAL_STATE_AMBIGUOUS)
    for deny in (TOLERANT_DENY, MODERATE_DENY, STRICT_DENY):
        assert is_physical_state_denied(state, deny, corroborated=corroborated) is True

    # ... and the real pipeline drops the row, not just the predicate.
    stID, entity_assemID, entityID = next(iter(entry.get_peptide_shifts()))
    row = pd.Series(
        {
            "entryID": entry_id,
            "stID": stID,
            "entity_assemID": entity_assemID,
            "entityID": entityID,
        }
    )
    df = pd.DataFrame(
        [
            fill_row_data(
                row,
                perturbing_cosolvents=[],
                keywords=[],
                bmrb_entries=pd.DataFrame({"entry": [entry]}, index=[entry_id]),
            )
        ]
    )
    df["exp_method_subtype"] = df["exp_method_subtype"].astype("string")
    out, _missing, sels_pre, *_rest = prefilter_dataframe(
        df,
        method_whitelist=["", "solution", "structures"],
        method_blacklist=["solid"],
        temperature_range=[-float("inf"), float("inf")],
        ionic_strength_range=[0.0, float("inf")],
        pH_range=[-float("inf"), float("inf")],
        peptide_length_range=[5, float("inf")],
        min_backbone_shift_types=1,
        min_backbone_shift_positions=1,
        min_backbone_shift_fraction=0.0,
        max_noncanonical_fraction=1.0,
        max_x_fraction=1.0,
        keywords=[],
        perturbing_cosolvents=[],
        physical_state_blacklist=MODERATE_DENY,
    )
    state_key = ("physical state", f"[{len(MODERATE_DENY)} denied]")
    assert bool(sels_pre[state_key].iloc[0]) is False
    assert bool(out["pass_pre"].iloc[0]) is False


# --------------------------------------------------------------------------- #
# `denatured` / `unfolded` are ambiguous depositor vocabulary
# --------------------------------------------------------------------------- #
#
# Depositors write `denatured` and `unfolded` both for a chemically denatured
# sample and for a natively unfolded IDP. bmr6968 -- alpha-synuclein, titled
# "... of intrinsically disordered alpha-synuclein" -- is deposited as
# `denatured`. Measured over the released tolerant tier: of 101 rows carrying
# one of these four values, 69 have no perturbing cosolvent anywhere in the entry,
# and
# they are dominated by alpha-synuclein (6968/16300/16301), Tau (52309/52401),
# gamma-synuclein, endosulfine alpha and the yeast SNAREs (4286/4287).
# Denying on the tag alone deletes the signal the dataset exists to capture, so
# these four are denied only when the entry independently evidences a perturbing
# cosolvent. The unambiguous states (molten globule, fibril, aggregated, ...)
# are denied on the tag alone.


@pytest.mark.parametrize(
    "value", ["denatured", "partially denatured", "unfolded", "partially unfolded"]
)
def test_ambiguous_states_are_not_denied_without_corroboration(value):
    assert sample_state.denied_physical_states(
        [value], sample_state.PHYSICAL_STATE_MODERATE_DENY, corroborated=[False]
    ) == [False]


@pytest.mark.parametrize(
    "value", ["denatured", "partially denatured", "unfolded", "partially unfolded"]
)
def test_ambiguous_states_are_denied_when_corroborated(value):
    assert sample_state.denied_physical_states(
        [value], sample_state.PHYSICAL_STATE_MODERATE_DENY, corroborated=[True]
    ) == [True]


@pytest.mark.parametrize(
    "value", ["molten globule", "amyloid fibril", "fibrils", "aggregated", "misfolded"]
)
def test_unambiguous_states_are_denied_on_the_tag_alone(value):
    assert sample_state.denied_physical_states(
        [value], sample_state.PHYSICAL_STATE_TOLERANT_DENY, corroborated=[False]
    ) == [True]


def test_corroboration_defaults_to_absent():
    """No corroboration information supplied => the ambiguous states are kept."""
    assert sample_state.denied_physical_states(
        ["denatured"], sample_state.PHYSICAL_STATE_MODERATE_DENY
    ) == [False]


@pytest.mark.parametrize(
    "texts,expected",
    [
        (["8 M urea"], True),
        (["in the presence of 8 M urea"], True),
        (["6 M guanidinium chloride"], True),
        (["GdmCl"], True),
        (["30% TFE"], True),
        # HFIP is a stronger helix inducer than TFE and is exactly the "not
        # aqueous buffer" evidence this rule tests for. Zero corpus rows flip
        # today (no ambiguous-state row names it); the tokens are forward cover.
        (["25 % HFIP"], True),
        (["hexafluoroisopropanol"], True),
        (["hexafluoroisopropanol-d2"], True),
        (["1,1,1,3,3,3-hexafluoro-2-propanol"], True),
        # ... but not the Cu(I) counterion sharing the `hexafluoro` prefix
        (["tetrakis(acetonitrile)copper(I) hexafluorophosphate"], False),
        (["urease"], False),  # `urea` must not fire on `urease`
        (["Urease from jack bean"], False),
        (["bis-pyridylurea inhibitor"], False),  # bmr26598
        (["palmitate, laureate, and stearate"], False),  # bmr50434
        (["phosphate buffer"], False),
        ([None, ""], False),
        # Underscore is a SEPARATOR in deposited names, not a word character:
        # these are real Mol_common_name / Sf_framecode spellings (SDS_d25 in
        # bmr15268, dmso_d6 in bmr18629, Urea_8M in bmr25255) and `\b` would
        # miss every one of them. Zero rows flip today -- all of them are
        # corroborated by another field -- so this is forward cover.
        (["SDS_d25"], True),
        (["dmso_d6"], True),
        (["Urea_8M"], True),
        (["8M_guanidine"], True),
        # ... and the boundary still holds on the other side of the underscore
        (["urease_from_jack_bean"], False),
    ],
)
def test_cosolvent_evidence_in_free_text(texts, expected):
    assert sample_state.has_cosolvent_evidence(texts) is expected


@requires_bmrb_data
def test_alpha_synuclein_survives_the_tolerant_deny_list():
    """bmr6968 is alpha-synuclein deposited as `denatured` with no cosolvent."""
    entry = bmrb.BmrbEntry("6968", str(BMRB_DIR))
    assembly = next(iter(entry.assemblies.values()))
    state = sample_state.resolve_physical_state(assembly, "1", "1")
    assert state == "denatured"
    names = [c[3] for s in entry.samples.values() for c in s.components]
    texts = [entry.title, entry.details] + names
    assert sample_state.has_cosolvent_evidence(texts) is False
    assert sample_state.denied_physical_states(
        [state], sample_state.PHYSICAL_STATE_TOLERANT_DENY, corroborated=[False]
    ) == [False]


@requires_bmrb_data
def test_apo_myoglobin_molten_globule_is_still_denied():
    """bmr5158 is the case this whole mechanism exists for -- it must stay denied."""
    entry = bmrb.BmrbEntry("5158", str(BMRB_DIR))
    assembly = next(iter(entry.assemblies.values()))
    state = sample_state.resolve_physical_state(assembly, "1", "1")
    assert state == "molten globule"
    assert sample_state.denied_physical_states(
        [state], sample_state.PHYSICAL_STATE_TOLERANT_DENY, corroborated=[False]
    ) == [True]
