"""Solution-vs-solid evidence, and the undeclared-method-subtype policy.

4,466 rows / 4,103 entries are dropped from the strict tier for one reason only:
``_Entry.Experimental_method_subtype`` is ABSENT. 97.5 % of them declare
``_Sample.Type == 'solution'``, none carry solid evidence, and 90.7 % were
deposited before 2006, when the tag was not routinely filled in.
:mod:`trizod.bmrb.sample_state` reads what the deposition actually says about the
sample so those rows can be judged on evidence rather than on a missing tag.

Two anchoring traps are pinned here:

* ``bicell_solution``, ``micelles`` and ``liquid crystal`` ARE isotropic solution
  NMR -- a fully anchored ``^solution$`` match wrongly drops bmr5813 and bmr6040;
* the solid veto must be the *refined* one (solid evidence **and** no
  solution-type experiment name). The naive veto has 37.5 % precision and kills
  real solution structures whose ``_Sample.Type`` is mis-typed.
"""

import numpy as np
import pandas as pd
import pytest

import trizod.bmrb.bmrb as bmrb
from tests.conftest import requires_bmrb_data
from trizod import paths
from trizod.bmrb.sample_state import (
    SOLID,
    SOLUTION,
    UNKNOWN,
    classify_sample_state,
    entry_sample_state,
)
from trizod.pipeline import prefilter_dataframe
from trizod.trizod import filter_defaults

SOLUTION_EXPERIMENTS = ["2D 1H-15N HSQC", "3D HNCA", "2D 1H-1H NOESY"]
SOLID_EXPERIMENTS = ["DARR 400 ms", "PAIN 6 ms", "PDSD 4 s", "NCA", "NCO"]


# --------------------------------------------------------------------------- #
# the classifier
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sample_type",
    [
        "solution",
        "Protein solution",
        "gel solution",
        "soultion",  # deposited typo
        "bicell_solution",  # bmr5813
        "micelles",  # bmr6040
        "DPC micelle",
        "liquid crystal",
        "Bi-cell",
        "isotropic",
    ],
)
def test_alignment_media_and_micelles_are_solution(sample_type):
    assert classify_sample_state(sample_types=[sample_type]) == SOLUTION


@pytest.mark.parametrize(
    "sample_type",
    [
        "solid",
        "solid-state",
        "gel solid",
        "hydrated solid",
        "lyophilized powder",
        "polycrystalline powder",
        "microcrystalline",
        "amyloid fibril",
        "fibrils",
        "fiber",
        "filamentous virus",
        "oriented membrane film",
        "liposome",
        "magic angle spinning sample",
    ],
)
def test_solid_sample_types_are_solid(sample_type):
    assert classify_sample_state(sample_types=[sample_type]) == SOLID


def test_no_evidence_at_all_is_unknown():
    assert classify_sample_state() == UNKNOWN
    assert classify_sample_state(sample_types=["cell suspension"]) == UNKNOWN


def test_sample_state_tag_is_read():
    assert classify_sample_state(sample_states=["isotropic"]) == SOLUTION
    assert classify_sample_state(sample_states=["solid"]) == SOLID
    # An aligned sample is neither: it is a solution sample in an alignment
    # medium, and it is also how oriented membrane samples are labelled.
    assert classify_sample_state(sample_states=["anisotropic"]) == UNKNOWN


def test_solid_experiment_names_are_evidence():
    # bmr5815: _Sample.Type says 'solution', the pulse sequences say otherwise.
    assert (
        classify_sample_state(
            sample_types=["solution"],
            experiment_names=[
                "Inversion and Spin Exchange at the Magic Angle",
                "PISEMA-Polarization",
            ],
        )
        == SOLID
    )


def test_refined_veto_spares_a_mistyped_solution_structure():
    # bmr34067 "Solution structure of the RBM5 OCRE domain": _Sample.Type is
    # 'solid', every experiment is a solution experiment. The naive veto (solid
    # evidence alone) drops it; the refined veto must not.
    assert (
        classify_sample_state(
            sample_types=["solid"],
            sample_states=["isotropic"],
            experiment_names=SOLUTION_EXPERIMENTS,
        )
        == SOLUTION
    )


def test_hsqc_alone_does_not_veto_solid_evidence():
    # bmr27211 runs a proton-detected "2D 1H-15N HSQC/HMQC" on an all-solid
    # sample set: HSQC/HMQC are not solution-only experiments any more.
    assert (
        classify_sample_state(
            sample_types=["solid"],
            sample_states=["isotropic"],
            experiment_names=["2D 1H-15N HSQC/HMQC", "3D hCANH", "DARR_600MHz"],
        )
        == SOLID
    )


def test_solid_experiment_tokens_need_word_boundaries():
    # 'mas', 'par' and 'pain' must not match inside an unrelated name.
    assert (
        classify_sample_state(
            sample_types=["solution"], experiment_names=["2D MASSIVE-COMPARISON"]
        )
        == SOLUTION
    )


# --------------------------------------------------------------------------- #
# entry_sample_state
# --------------------------------------------------------------------------- #


class _Sample:
    def __init__(self, type_):
        self.type = type_


class _ExperimentList:
    def __init__(self, names):
        self.experiments = [
            (str(i), name, None, "1", None, "") for i, name in enumerate(names, 1)
        ]


class _Entry:
    def __init__(self, samples, names):
        self.samples = samples
        self.experiment_list = _ExperimentList(names)
        self.shift_tables = {}


def test_entry_sample_state_uses_only_the_referenced_samples():
    entry = _Entry(
        {"1": _Sample("solution"), "2": _Sample("solid")}, SOLUTION_EXPERIMENTS
    )
    assert entry_sample_state(entry, ["1"]) == SOLUTION
    # Sample 2 is solid, but every experiment is a solution experiment, so the
    # refined veto still resolves to solution.
    assert entry_sample_state(entry, ["2"]) == SOLUTION


def test_entry_sample_state_falls_back_to_every_sample():
    entry = _Entry({"1": _Sample("solid")}, SOLID_EXPERIMENTS)
    assert entry_sample_state(entry, []) == SOLID
    assert entry_sample_state(entry, ["nonexistent"]) == SOLID


# --------------------------------------------------------------------------- #
# tier policy
# --------------------------------------------------------------------------- #


def test_method_fallback_tier_defaults():
    assert filter_defaults.loc["unfiltered", "method-fallback"] == "off"
    assert filter_defaults.loc["tolerant", "method-fallback"] == "reject-solid"
    assert filter_defaults.loc["moderate", "method-fallback"] == "reject-solid"
    assert filter_defaults.loc["strict", "method-fallback"] == "require-solution"


def _frame(rows):
    subtypes = [subtype for subtype, _evidence in rows]
    evidence = [ev for _subtype, ev in rows]
    n = len(rows)
    return pd.DataFrame(
        {
            "exp_method": ["NMR"] * n,
            # "string", as create_peptide_dataframe casts it: an all-None object
            # column has no .str accessor semantics to speak of.
            "exp_method_subtype": pd.array(subtypes, dtype="string"),
            "sample_state_evidence": evidence,
            "physical_state": [None] * n,
            "temperature": [298.0] * n,
            "ionic_strength": [0.1] * n,
            "pH": [7.0] * n,
            "seq": ["A" * 20] * n,
            "total_bbshifts": [80] * n,
            "bbshift_types": [4] * n,
            "bbshift_positions": [18] * n,
        }
    )


def _prefilter(rows, tier, frame=None):
    return prefilter_dataframe(
        _frame(rows) if frame is None else frame,
        method_whitelist=list(filter_defaults.loc[tier, "exp-method-whitelist"]),
        method_blacklist=list(filter_defaults.loc[tier, "exp-method-blacklist"]),
        temperature_range=[-np.inf, np.inf],
        ionic_strength_range=[0.0, np.inf],
        pH_range=[-np.inf, np.inf],
        peptide_length_range=[5, np.inf],
        min_backbone_shift_types=1,
        min_backbone_shift_positions=1,
        min_backbone_shift_fraction=0.0,
        max_noncanonical_fraction=1.0,
        max_x_fraction=1.0,
        keywords=[],
        perturbing_cosolvents=[],
        method_fallback=filter_defaults.loc[tier, "method-fallback"],
    )


def _method_selection(rows, tier):
    _df, _missing, sels_pre, *_rest = _prefilter(rows, tier)
    return [bool(v) for v in sels_pre[("method (sub-)type", "")]]


def _pass_pre(rows, tier, frame=None):
    df, *_rest = _prefilter(rows, tier, frame=frame)
    return df["pass_pre"].tolist()


def test_strict_admits_a_null_subtype_on_solution_evidence():
    rows = [(None, SOLUTION), (None, SOLID), (None, UNKNOWN)]
    assert _method_selection(rows, "strict") == [True, False, False]


def test_strict_pass_pre_admits_a_null_subtype_on_solution_evidence():
    # Not just the method selection: a strict-tier null subtype is ALSO counted
    # as a "missing value", which vetoes the row one line after the fallback has
    # admitted it. Without this, the whole recovery is silently worth nothing.
    rows = [(None, SOLUTION), (None, SOLID), (None, UNKNOWN), ("solution", SOLUTION)]
    assert _pass_pre(rows, "strict") == [True, False, False, True]


def test_re_admitted_rows_still_face_every_other_filter():
    rows = [(None, SOLUTION), (None, SOLUTION)]
    frame = _frame(rows)
    frame.loc[1, "temperature"] = None  # a genuinely missing value
    assert _pass_pre(rows, "strict", frame=frame) == [True, False]


def test_strict_fallback_does_not_re_admit_junk_subtypes():
    # The restriction to a NULL subtype is what stops 'X-RAY DIFFRACTION' and
    # 'THEORETICAL' rows from walking into strict on _Sample.Type == 'solution'.
    rows = [
        ("X-RAY DIFFRACTION", SOLUTION),
        ("THEORETICAL", SOLUTION),
        ("STATE", SOLUTION),
        ("solution", SOLUTION),
    ]
    assert _method_selection(rows, "strict") == [False, False, False, True]


def test_strict_vetoes_solid_evidence_whatever_the_subtype_says():
    # bmr25289 declares "NMR, 20 STRUCTURES" and bmr27211 declares "solution";
    # both are solid-state depositions sitting in the released strict tier.
    rows = [("NMR, 20 STRUCTURES", SOLID), ("solution", SOLID), ("solution", SOLUTION)]
    assert _method_selection(rows, "strict") == [False, False, True]


def test_tolerant_rejects_solid_evidence_only_for_a_null_subtype():
    rows = [
        (None, SOLID),  # dropped by the fallback
        (None, SOLUTION),  # admitted by the "" sentinel, as before
        (None, UNKNOWN),  # admitted by the "" sentinel, as before
        ("X-RAY DIFFRACTION", SOLUTION),  # still rejected by the whitelist
        ("NMR, 20 STRUCTURES", SOLID),  # present subtype: fallback does not apply
    ]
    assert _method_selection(rows, "tolerant") == [False, True, True, False, True]


def test_unfiltered_keeps_the_raw_corpus():
    rows = [(None, SOLID), (None, SOLUTION), ("SOLID-STATE", SOLID)]
    assert _method_selection(rows, "unfiltered") == [True, True, True]


def test_unknown_method_fallback_is_rejected():
    with pytest.raises(ValueError, match="method fallback"):
        _frame_rows = [(None, SOLUTION)]
        prefilter_dataframe(
            _frame(_frame_rows),
            method_whitelist=["solution"],
            method_blacklist=["solid"],
            temperature_range=[-np.inf, np.inf],
            ionic_strength_range=[0.0, np.inf],
            pH_range=[-np.inf, np.inf],
            peptide_length_range=[5, np.inf],
            min_backbone_shift_types=1,
            min_backbone_shift_positions=1,
            min_backbone_shift_fraction=0.0,
            max_noncanonical_fraction=1.0,
            max_x_fraction=1.0,
            keywords=[],
            perturbing_cosolvents=[],
            method_fallback="reject-everything",
        )


def test_no_strict_row_keeps_solid_evidence():
    rows = [
        (None, SOLID),
        ("solution", SOLID),
        ("NMR, 20 STRUCTURES", SOLID),
        ("solution", SOLUTION),
        (None, SOLUTION),
    ]
    selection = _method_selection(rows, "strict")
    evidence = [ev for _subtype, ev in rows]
    assert not any(keep and ev == SOLID for keep, ev in zip(selection, evidence))


# --------------------------------------------------------------------------- #
# real entries
# --------------------------------------------------------------------------- #


@requires_bmrb_data
@pytest.mark.parametrize(
    ("entry_id", "expected"),
    [
        ("5815", SOLID),  # fd bacteriophage coat protein: PISEMA, magic angle
        ("25289", SOLID),  # Abeta fibrils: _Sample.Type=solid, DARR/PAIN/PDSD
        ("27211", SOLID),  # P. horikoshii TET2: all five samples _Sample.Type=solid
        ("6191", SOLUTION),
        ("10035", SOLUTION),
        ("5813", SOLUTION),  # 'bicell_solution'
        ("6040", SOLUTION),  # 'micelles'
        ("34067", SOLUTION),  # mis-typed _Sample.Type, real solution structure
        ("36119", SOLUTION),
        ("30293", SOLUTION),
        ("34330", SOLUTION),
        ("16340", SOLUTION),
    ],
)
def test_real_entry_sample_state(entry_id, expected):
    entry = bmrb.BmrbEntry(entry_id, paths.RAW_BMRB / f"bmr{entry_id}")
    stID, entity_assemID, entityID = next(iter(entry.get_peptide_shifts()))
    _shifts, _condID, _assemID, sampleIDs = entry.get_peptide_shifts()[
        (stID, entity_assemID, entityID)
    ]
    assert entry_sample_state(entry, sampleIDs) == expected
