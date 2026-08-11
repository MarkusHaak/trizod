"""Units must be visible in the name of every emitted offset column.

TriZOD carries two per-atom referencing offsets that are *not* in the same unit:

* the POTENCI/AIC residual offset is in **sigma** units -- ``compute_offsets()``
  averages ``diff_arr / REFINED_WEIGHTS`` (``scoring.py``), so it is a multiple
  of the per-atom POTENCI RMSD;
* the LACS offset is in **ppm** -- ``apply_lacs_correction()`` subtracts it
  straight off the raw shift array.

They used to be emitted as ``off_<atom>`` and ``lacs_off_<atom>``, names that
say nothing about either unit. A downstream analysis added the two together and
injected up to 21.2 ppm of pure unit error. These tests pin the fix: every
emitted offset column carries a ``_sigma`` or ``_ppm`` suffix, and the derived
``total_off_<atom>_ppm`` column hands users the ready-made ppm quantity so the
conversion never has to be done by hand again.

The rename also created a *second* defect class, which is pinned here too: a
consumer left spelling a key the emitter no longer writes. ``record.get("off_CA")
or 0.0`` on a renamed column does not raise — it yields a perfect 0.0 for every
chain, and the dataset build ranked exact-sequence duplicates on those zeros
(58 of 12,745 sequences elected a different representative). So no module may
hard-code an offset key at all: the names live in :mod:`trizod.offsets`, and the
readers go through :func:`trizod.offsets.offset_value`, which raises on a key
that is not there.
"""

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trizod import paths
from trizod.constants import BACKBONE_ATOMS, REFINED_WEIGHTS
from trizod.offsets import (
    OFFSET_COLUMNS,
    MissingOffsetColumn,
    lacs_off_ppm_col,
    max_abs_offset,
    off_sigma_col,
    offset_value,
    total_off_ppm_col,
    total_offset_ppm,
)
from trizod.pipeline import postfilter_dataframe
from trizod.trizod import compute_scores_row, output_dataset

#: Any column name mentioning an offset at all.
_OFFSET_NAME = re.compile(r"off_")
#: ... must end in one of the two units TriZOD uses.
_UNIT_SUFFIX = re.compile(r"_(sigma|ppm)$")

LEGACY_NAMES = {f"off_{a}" for a in BACKBONE_ATOMS} | {
    f"lacs_off_{a}" for a in BACKBONE_ATOMS
}


def _scored_frame():
    """One passing row carrying every column ``output_dataset`` reads."""
    seq = "AGAGA"
    n = len(seq)
    row = {
        "entryID": ["12345"],
        "stID": ["1"],
        "entity_assemID": ["1"],
        "entityID": ["1"],
        "entity_name": ["test peptide"],
        "exp_method": ["NMR"],
        "exp_method_subtype": ["solution"],
        "sample_state_evidence": ["solution"],
        "physical_state": ["native"],
        "cosolvent_evidence": [False],
        "membrane_mimetic": ["SDS;micelle"],
        "citation_DOI": ["10.1000/xyz"],
        "citation_title": ["A title"],
        "ionic_strength": [0.15],
        "pH": [7.0],
        "temperature": [298.0],
        "bbshift_positions_post": [n],
        "bbshift_types_post": [3],
        "total_bbshifts": [3 * n],
        "seq": [seq],
        "k": [[3] * n],
        "zscores": [[0.5] * n],
        "pass_post": [True],
    }
    for atom in BACKBONE_ATOMS:
        row[off_sigma_col(atom)] = [2.0]
        row[lacs_off_ppm_col(atom)] = [0.3]
        row[total_off_ppm_col(atom)] = [total_offset_ppm(atom, 0.3, 2.0)]
    return pd.DataFrame(row)


def _emit(tmp_path):
    output_dataset(
        _scored_frame(),
        tmp_path / "scores",
        "json",
        ["zscores"],
        6,
        include_shifts=False,
        no_shift_averaging=False,
    )
    text = (tmp_path / "scores.json").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()][0]


# --------------------------------------------------------------------------- #
# the guard: no unlabelled unit can sneak back in
# --------------------------------------------------------------------------- #


def test_declared_offset_columns_all_carry_a_unit_suffix():
    assert OFFSET_COLUMNS, "no offset columns declared"
    unlabelled = [c for c in OFFSET_COLUMNS if not _UNIT_SUFFIX.search(c)]
    assert not unlabelled, (
        f"offset columns without a unit suffix: {unlabelled}. Every offset "
        "column name must end in _sigma or _ppm -- this is the whole point of "
        "the rename."
    )


def test_emitted_json_has_no_unlabelled_offset_column(tmp_path):
    """Guard the real emission path, not just the declared constant."""
    record = _emit(tmp_path)
    offset_keys = [k for k in record if _OFFSET_NAME.search(k)]
    assert offset_keys, "output_dataset() emitted no offset column at all"
    unlabelled = [k for k in offset_keys if not _UNIT_SUFFIX.search(k)]
    assert not unlabelled, (
        f"output_dataset() emitted offset columns without a unit suffix: {unlabelled}"
    )


def test_emitted_json_does_not_reuse_the_legacy_unitless_names(tmp_path):
    record = _emit(tmp_path)
    assert not (LEGACY_NAMES & set(record)), (
        "the legacy unit-free names are back in the output"
    )


def test_emitted_json_carries_the_three_offset_families(tmp_path):
    record = _emit(tmp_path)
    for atom in BACKBONE_ATOMS:
        assert off_sigma_col(atom) in record
        assert lacs_off_ppm_col(atom) in record
        assert total_off_ppm_col(atom) in record


# --------------------------------------------------------------------------- #
# the arithmetic, pinned to a real chain
# --------------------------------------------------------------------------- #

# Chain 18256_1_1_1 from data/interim/scored/unfiltered/scores.json -- picked
# because four atoms carry a large non-zero POTENCI offset AND a non-zero LACS
# offset, so a sigma/ppm mix-up cannot hide. Values below are the deposited
# off_/lacs_off_ numbers; the totals are hand-computed as
# lacs_ppm + sigma * REFINED_WEIGHTS[atom].
PINNED_CHAIN = "18256_1_1_1"
PINNED_OFFSETS = {
    # atom: (off_sigma, lacs_off_ppm, expected total_off_ppm)
    "C": (9.2934276844, 0.34, 2.0555667505402395),
    "CA": (13.5917784137, 0.45, 3.14389048159534),
    "CB": (-7.7842855874, 0.45, -0.7518936946945602),
    "HA": (-9.4282312504, -0.03, -0.278056764198024),
    "H": (0.0, 0.07, 0.07),
    "N": (0.0, -1.21, -1.21),
    "HB": (0.0, 0.0, 0.0),
}


@pytest.mark.parametrize("atom", list(PINNED_OFFSETS))
def test_total_offset_ppm_pins_a_real_chain(atom):
    off_sigma, lacs_ppm, expected = PINNED_OFFSETS[atom]
    assert total_offset_ppm(atom, lacs_ppm, off_sigma) == pytest.approx(
        expected, abs=1e-12
    )


#: The pinned atoms that actually carry a non-zero sigma offset -- the only ones
#: on which the naive sum and the real conversion can differ at all.
SIGMA_BEARING = [a for a, (off_sigma, _, _) in PINNED_OFFSETS.items() if off_sigma]


@pytest.mark.parametrize("atom", SIGMA_BEARING)
def test_total_offset_ppm_is_not_the_naive_sum(atom):
    """The bug this rename exists to prevent, stated as a test.

    ``total_offset_ppm`` is *called*, and its result is what the naive sum is
    compared against. The previous version of this test compared two literals
    out of ``PINNED_OFFSETS`` and never called the function, so replacing the
    conversion with ``lacs + sigma`` -- the exact bug -- left it green.
    """
    off_sigma, lacs_ppm, _ = PINNED_OFFSETS[atom]
    total = total_offset_ppm(atom, lacs_ppm, off_sigma)
    naive = lacs_ppm + off_sigma
    assert total != pytest.approx(naive), (
        f"total_offset_ppm({atom}) returned the naive sigma+ppm sum; the sigma "
        f"term must be scaled by REFINED_WEIGHTS[{atom}]"
    )


def test_naive_sum_error_on_ca_is_large_enough_to_flip_a_verdict():
    """Magnitude, not just inequality: 2.7 ppm on CA for the pinned chain --
    enough to turn a well-referenced spectrum into a rejected one."""
    off_sigma, lacs_ppm, _ = PINNED_OFFSETS["CA"]
    naive = lacs_ppm + off_sigma
    assert abs(naive - total_offset_ppm("CA", lacs_ppm, off_sigma)) > 2.0


def test_total_offset_ppm_scales_the_sigma_term_by_the_atom_weight():
    for atom in BACKBONE_ATOMS:
        assert total_offset_ppm(atom, 0.0, 1.0) == pytest.approx(REFINED_WEIGHTS[atom])
        assert total_offset_ppm(atom, 1.0, 0.0) == pytest.approx(1.0)


def test_total_offset_ppm_propagates_missing_values():
    assert np.isnan(total_offset_ppm("CA", 0.5, np.nan))
    assert np.isnan(total_offset_ppm("CA", pd.NA, 1.0))
    assert np.isnan(total_offset_ppm("CA", None, 1.0))


# --------------------------------------------------------------------------- #
# the scored values themselves must not move
# --------------------------------------------------------------------------- #


def test_compute_scores_row_stores_the_unconverted_sigma_offset(monkeypatch):
    """Renaming must not convert anything: off_*_sigma stays exactly sigma.

    ``--max-offset`` (3/3/2) is compared against this number inside
    ``pipeline.compute_scores``, so scaling it here would silently change what
    the filter means.
    """
    offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    offsets["CA"] = 2.5
    lacs_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    lacs_offsets["CA"] = 0.3
    n = 5
    cmp_mask = np.ones((n, len(BACKBONE_ATOMS)), dtype=bool)

    def fake_compute_scores(*args, **kwargs):
        return (
            [np.zeros(n)],
            np.full(n, 3),
            cmp_mask,
            offsets,
            [0.0, 0.0, 0.0],
            lacs_offsets,
        )

    monkeypatch.setattr("trizod.trizod.compute_scores", fake_compute_scores)
    row = pd.Series(
        {
            "pass_pre": True,
            "entryID": "12345",
            "stID": "1",
            "entity_assemID": "1",
            "entityID": "1",
            "seq": "AGAGA",
            "ionic_strength": 0.15,
            "pH": 7.0,
            "temperature": 298.0,
        }
    )
    bmrb_entries = pd.DataFrame({"entry": [None]}, index=["12345"])

    out = compute_scores_row(row, bmrb_entries=bmrb_entries)

    assert out[off_sigma_col("CA")] == 2.5
    assert out[lacs_off_ppm_col("CA")] == 0.3
    assert out[total_off_ppm_col("CA")] == pytest.approx(
        0.3 + 2.5 * REFINED_WEIGHTS["CA"]
    )
    for legacy in LEGACY_NAMES:
        assert legacy not in out.index


# --------------------------------------------------------------------------- #
# the post-filter still reads the sigma column
# --------------------------------------------------------------------------- #


def _post_frame(n):
    data = {
        "seq": ["A" * 20] * n,
        "zscores": [np.zeros(20)] * n,
        "bbshift_types_post": [4] * n,
        "bbshift_positions_post": [18] * n,
        "pass_pre": [True] * n,
    }
    for atom_type in BACKBONE_ATOMS:
        data[off_sigma_col(atom_type)] = [0.0] * n
    return pd.DataFrame(data)


def test_postfilter_rejects_on_the_sigma_column():
    """The offset-rejection mask reads off_<atom>_sigma, and only that."""
    df = _post_frame(4)
    df.loc[2, off_sigma_col("CA")] = np.nan  # offset rejected during scoring
    postfilter_dataframe(df, 1, 1, 0.0, False, ["zscores"])
    assert df["pass_post"].tolist() == [True, True, False, True]


def test_postfilter_reports_offsets_under_unit_labelled_keys():
    df = _post_frame(2)
    _, sels_off, _ = postfilter_dataframe(df, 1, 1, 0.0, False, ["zscores"])
    assert set(sels_off) == {off_sigma_col(a) for a in BACKBONE_ATOMS}


# --------------------------------------------------------------------------- #
# no consumer may read an offset key the emitter does not write
# --------------------------------------------------------------------------- #

#: A string that is *shaped like* an offset column key: one of the three
#: prefixes, then either a literal atom (``CA``) or an f-string hole (``{}``),
#: then optionally a unit suffix. Anchored at both ends so prose that merely
#: mentions a column ("lacs_off_CA_ppm, or null.") is not mistaken for one.
_KEY_SHAPED = re.compile(
    r"^(?:off|lacs_off|total_off)_(?:\{\}|[A-Z][A-Z0-9]*)(?:_sigma|_ppm)?$"
)

#: The single module allowed to spell an offset column name. Every other module
#: must call its helpers, which is what makes a rename impossible to half-apply.
NAME_OWNER = Path("trizod/offsets.py")

#: Trees scanned. Tests are excluded on purpose: a test may legitimately pin a
#: literal name (that is what ``LEGACY_NAMES`` above does) precisely so that a
#: rename has to be acknowledged here.
SCANNED_ROOTS = ("trizod", "scripts")


def _offset_key_literals(path):
    """``[(literal, lineno)]`` for every offset-key-shaped string in ``path``.

    f-strings are reduced to their template (``f"off_{a}_sigma"`` ->
    ``off_{}_sigma``) so an interpolated name is caught exactly like a constant
    one -- interpolation is how the stale keys were written in the first place.
    """
    found = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(
                part.value if isinstance(part, ast.Constant) else "{}"
                for part in node.values
            )
        else:
            continue
        if _KEY_SHAPED.match(text):
            found.append((text, node.lineno))
    return found


def _scanned_files():
    for root in SCANNED_ROOTS:
        for path in sorted((paths.ROOT / root).rglob("*.py")):
            if path.relative_to(paths.ROOT) != NAME_OWNER:
                yield path


def test_the_key_shape_detector_actually_detects():
    """Guard the guard: the scan below is only meaningful if it can find a key.

    ``trizod/offsets.py`` is the one module that spells the names, so it must
    light up. If this ever goes quiet, the scan has stopped scanning and every
    assertion built on it is vacuous.
    """
    literals = dict(_offset_key_literals(paths.ROOT / NAME_OWNER))
    assert {"off_{}_sigma", "lacs_off_{}_ppm", "total_off_{}_ppm"} <= set(literals), (
        f"detector found only {sorted(literals)} in {NAME_OWNER}"
    )


def test_no_module_hard_codes_an_offset_column_name():
    """The defect class: a consumer spelling a key the emitter does not write.

    Not "the current stale keys" -- any hard-coded offset key at all, because a
    hard-coded key is what lets a reader and the emitter drift apart silently.
    """
    offenders = [
        f"{path.relative_to(paths.ROOT)}:{lineno}: {literal!r}"
        for path in _scanned_files()
        for literal, lineno in _offset_key_literals(path)
    ]
    assert not offenders, (
        "offset column names are hard-coded outside trizod/offsets.py:\n  "
        + "\n  ".join(offenders)
        + "\nUse off_sigma_col() / lacs_off_ppm_col() / total_off_ppm_col() so a "
        "rename reaches every reader, and offset_value() to read them."
    )


def test_every_name_the_helpers_produce_is_actually_emitted(tmp_path):
    """The other half: the helpers must name columns ``output_dataset`` writes.

    Together with the scan above this closes the loop -- consumers can only name
    a column through the helpers, and every helper name is emitted, so no
    consumer can read a key that is not there.
    """
    record = _emit(tmp_path)
    missing = [c for c in OFFSET_COLUMNS if c not in record]
    assert not missing, (
        f"trizod.offsets names {missing}, which output_dataset() does not emit"
    )


# --------------------------------------------------------------------------- #
# reading an offset back: missing, null and zero are three different answers
# --------------------------------------------------------------------------- #


def test_offset_value_raises_on_a_column_that_is_not_there():
    """The whole bug in one assertion: a stale key must not read as 0.0."""
    record = {off_sigma_col("CA"): 1.5}
    with pytest.raises(MissingOffsetColumn):
        offset_value(record, "off_CA")  # the pre-rename spelling


def test_offset_value_distinguishes_a_measured_zero_from_a_null():
    record = {off_sigma_col("CA"): 0.0, off_sigma_col("CB"): None}
    assert offset_value(record, off_sigma_col("CA")) == 0.0
    assert offset_value(record, off_sigma_col("CB")) is None


def test_offset_value_treats_nan_as_null():
    """``--max-offset`` rejection NaNs the offset; JSON round-trips it as null."""
    assert (
        offset_value({off_sigma_col("CA"): float("nan")}, off_sigma_col("CA")) is None
    )


def test_max_abs_offset_skips_nulls_but_still_raises_on_a_missing_column():
    cols = [off_sigma_col(a) for a in ("CA", "CB")]
    assert max_abs_offset({cols[0]: -2.5, cols[1]: None}, cols) == 2.5
    assert max_abs_offset({cols[0]: None, cols[1]: None}, cols) is None
    with pytest.raises(MissingOffsetColumn):
        max_abs_offset({cols[0]: 1.0}, cols)


def test_max_abs_offset_returns_none_not_zero_when_nothing_is_measurable():
    """``None`` means "no offset measured"; ``0.0`` means "measured, perfect".

    Collapsing the first into the second is the coercion that started all this.
    """
    cols = [off_sigma_col(a) for a in BACKBONE_ATOMS]
    assert max_abs_offset(dict.fromkeys(cols), cols) is None
    assert max_abs_offset(dict.fromkeys(cols, 0.0), cols) == 0.0
