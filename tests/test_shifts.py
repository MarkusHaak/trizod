"""The complete deposited shift table, and the re-referencing attached to it.

``trizod_shifts.parquet`` ships every assigned chemical shift of every released
chain — backbone *and* side chain — as deposited, plus a second, explicitly
labelled column holding the re-referenced value where a trustworthy offset
exists.

Two things are load-bearing here and are pinned by these tests:

1. **Scoring is untouched.** ``get_valid_bbshifts()`` still returns exactly what
   it returned before, so z/g/k and both offset dicts are bit-identical.
2. **Units.** The two offsets TriZOD carries are in *different units* —
   ``lacs_off_<atom>_ppm`` in ppm, ``off_<atom>_sigma`` in multiples of the
   per-atom POTENCI RMSD — and adding them together is a unit error worth up to
   21.2 ppm. ``val_corrected_ppm`` must reproduce the shift the scorer actually
   used, which is the only end-to-end check that catches the mistake.
"""

import copy

import numpy as np
import pytest

from tests.conftest import BMRB_DIR, requires_bmrb_data
from trizod.bmrb.bmrb import (
    BB_ATOM_IDS,
    get_deposited_shifts,
    get_sidechain_shifts,
    get_valid_bbshifts,
)
from trizod.constants import AA3TO1, BACKBONE_ATOMS, REFINED_WEIGHTS
from trizod.offsets import total_offset_ppm
from trizod.shifts import (
    OFFSET_SOURCES,
    SIDECHAIN_MAX_ABS_LACS_PPM,
    annotate_offsets,
    backbone_slot,
    chain_offsets,
)

# 4-residue toy chain (ALA-LEU-PHE-GLY). Tuple layout mirrors ShiftTable.shifts:
# (entity_assembly_ID, entity_ID, Seq_ID, Comp_ID, Atom_ID, Atom_type, Val,
#  Val_err, Ambiguity_code), all strings as parsed from NMR-STAR.
TOY_SEQ = "ALFG"
TOY_SHIFTS = [
    # --- backbone ---------------------------------------------------------
    ("1", "1", "1", "ALA", "CA", "C", "52.30", "0.10", "1"),
    ("1", "1", "1", "ALA", "CB", "C", "19.10", "0.10", "1"),
    ("1", "1", "1", "ALA", "HB1", "H", "1.35", "0.02", "1"),
    ("1", "1", "1", "ALA", "N", "N", "125.10", "0.20", "1"),
    ("1", "1", "2", "LEU", "CA", "C", "55.10", "0.10", "1"),
    ("1", "1", "2", "LEU", "H", "H", "8.21", "0.02", "1"),
    ("1", "1", "2", "LEU", "HB2", "H", "1.60", "0.02", "2"),
    ("1", "1", "2", "LEU", "HB3", "H", "1.70", "0.02", "2"),
    ("1", "1", "2", "LEU", "C", "C", "176.20", "0.10", "1"),
    ("1", "1", "3", "PHE", "N", "N", "120.40", "0.20", "1"),
    ("1", "1", "4", "GLY", "HA2", "H", "3.95", "0.02", "2"),
    ("1", "1", "4", "GLY", "HA3", "H", "4.05", "0.02", "2"),
    # --- side chain -------------------------------------------------------
    ("1", "1", "2", "LEU", "CD1", "C", "24.50", "0.20", "2"),
    ("1", "1", "2", "LEU", "CD2", "C", "23.10", "0.20", "2"),
    ("1", "1", "2", "LEU", "HD11", "H", "0.85", "5.00", "2"),
    ("1", "1", "3", "PHE", "CD1", "C", "131.50", "0.30", "3"),
    ("1", "1", "3", "PHE", "HD1", "H", "7.15", "", "3"),
    ("1", "1", "2", "LEU", "ND2", "N", "112.30", "0.30", "1"),
]

#: A scores.json record's offset block, in the post-rename spelling.
TOY_RECORD = {
    "lacs_off_C_ppm": 0.0,
    "lacs_off_CA_ppm": 0.16,
    "lacs_off_CB_ppm": 0.16,
    "lacs_off_H_ppm": -0.05,
    "lacs_off_HA_ppm": -0.06,
    "lacs_off_HB_ppm": 0.0,
    "lacs_off_N_ppm": 0.87,
    "off_C_sigma": 1.3404011677,
    "off_CA_sigma": 0.0,
    "off_CB_sigma": -1.7826683938,
    "off_H_sigma": 0.0,
    "off_HA_sigma": 0.0,
    "off_HB_sigma": -1.4204271123,
    "off_N_sigma": -0.6955098727,
}


def _toy_table(record=TOY_RECORD):
    df = get_deposited_shifts(TOY_SHIFTS, TOY_SEQ)
    return annotate_offsets(df, chain_offsets(record))


def _row(df, seq_id, atom_id):
    sel = df[(df["seq_id"] == seq_id) & (df["atom_id"] == atom_id)]
    assert len(sel) == 1, f"expected exactly one row for {seq_id}/{atom_id}"
    return sel.iloc[0]


# --------------------------------------------------------------------------- #
# extraction: the table is complete, and it is raw
# --------------------------------------------------------------------------- #


def test_deposited_shifts_cover_backbone_and_side_chain():
    """Every deposited value of a canonical residue appears exactly once, with
    ``is_backbone`` partitioning them on the scoring path's own whitelist."""
    df = get_deposited_shifts(TOY_SHIFTS, TOY_SEQ)

    assert set(zip(df["seq_id"], df["atom_id"])) == {
        (int(t[2]), t[4]) for t in TOY_SHIFTS
    }
    assert set(df.loc[df["is_backbone"], "atom_id"]) <= BB_ATOM_IDS
    assert not set(df.loc[~df["is_backbone"], "atom_id"]) & BB_ATOM_IDS
    # the partition is the whitelist, nothing else
    assert list(df["is_backbone"]) == [a in BB_ATOM_IDS for a in df["atom_id"]]
    for seq_id, comp_id in zip(df["seq_id"], df["comp_id"]):
        assert TOY_SEQ[seq_id - 1] == AA3TO1[comp_id]


def test_backbone_values_are_the_raw_deposited_ones():
    """``val_ppm`` on a backbone row is what ``get_valid_bbshifts`` reads before
    any correction — never a corrected or averaged value."""
    df = get_deposited_shifts(TOY_SHIFTS, TOY_SEQ)
    bbshifts_arr, bbshifts_mask = get_valid_bbshifts(TOY_SHIFTS, TOY_SEQ)

    for pos, atom in zip(*np.where(bbshifts_mask)):
        slot = BACKBONE_ATOMS[atom]
        in_slot = np.array(
            [backbone_slot(c, a) == slot for c, a in zip(df["comp_id"], df["atom_id"])]
        )
        rows = df[(df["seq_id"] == pos + 1).to_numpy() & in_slot]
        assert len(rows) >= 1
        # get_valid_bbshifts averages degenerate partners (HB2/HB3, GLY HA2/HA3)
        assert rows["val_ppm"].mean() == pytest.approx(bbshifts_arr[pos, atom])
    # ... and the deposited partners are kept apart, not pre-averaged
    hb = df[(df["seq_id"] == 2) & df["atom_id"].isin(["HB2", "HB3"])]
    assert sorted(hb["val_ppm"]) == [1.60, 1.70]


def test_backbone_rows_keep_the_values_the_scorer_filters_out():
    """The scoring path drops values with ``Val_err > 1.3`` or an ambiguity code
    outside {1,2}; the published table keeps them, with the code as data. The
    backbone ambiguity whitelist alone would discard 85,831 aromatic
    ring-degenerate values corpus-wide."""
    shifts = [
        ("1", "1", "1", "LEU", "CA", "C", "55.10", "9.90", "1"),  # err > 1.3
        ("1", "1", "1", "LEU", "CB", "C", "42.10", "0.10", "3"),  # code 3
    ]
    df = get_deposited_shifts(shifts, "L")

    assert set(df["atom_id"]) == {"CA", "CB"}
    assert bool(df["is_backbone"].all())
    assert _row(df, 1, "CB")["ambiguity_code"] == 3
    assert _row(df, 1, "CA")["val_err_ppm"] == 9.90
    # the scoring path sees neither
    arr, mask = get_valid_bbshifts(shifts, "L")
    assert not mask.any()


def test_malformed_metadata_survives_extraction():
    """A junk ``Val_err`` or ``Ambiguity_code`` is carried metadata, not a guard.
    Codes outside the BMRB domain (1-9) become null."""
    shifts = [
        ("1", "1", "1", "LEU", "CD1", "C", "24.50", "n/a", "?"),
        ("1", "1", "1", "LEU", "CA", "C", "55.10", ".", "42"),
        ("1", "1", "1", "LEU", "HD11", "H", "0.85", "0.01", "2"),
    ]
    df = get_deposited_shifts(shifts, "L")

    assert df is not None and len(df) == 3
    assert sorted(df["val_ppm"]) == [0.85, 24.50, 55.10]
    assert int(df["val_err_ppm"].isna().sum()) == 2
    assert int(df["ambiguity_code"].isna().sum()) == 2


def test_sequence_mismatch_rejects_the_table():
    """Same sequence-consistency guard as the backbone read, so
    ``sequence[seq_id - 1]`` is a safe join key downstream."""
    assert get_deposited_shifts(TOY_SHIFTS, "AAFG") is None
    assert get_valid_bbshifts(TOY_SHIFTS, "AAFG") is None


def test_get_sidechain_shifts_is_the_complement():
    """The side-chain-only view (used by scripts/sidechain_coverage.py) is
    exactly the ``~is_backbone`` slice of the full table."""
    full = get_deposited_shifts(TOY_SHIFTS, TOY_SEQ)
    sc = get_sidechain_shifts(TOY_SHIFTS, TOY_SEQ)

    assert "is_backbone" not in sc.columns
    assert list(sc["atom_id"]) == list(full.loc[~full["is_backbone"], "atom_id"])
    assert not set(sc["atom_id"]) & BB_ATOM_IDS


# --------------------------------------------------------------------------- #
# the offset policy
# --------------------------------------------------------------------------- #


def test_backbone_slot_mirrors_get_valid_bbshifts():
    """Degenerate partners map onto the slot the scoring path averages them into
    — and only for the residues where it does so."""
    assert [backbone_slot("LEU", a) for a in BACKBONE_ATOMS] == BACKBONE_ATOMS
    assert backbone_slot("LEU", "HB2") == "HB"
    assert backbone_slot("LEU", "HB3") == "HB"
    assert backbone_slot("ALA", "HB1") == "HB"
    assert backbone_slot("GLY", "HA2") == "HA"
    assert backbone_slot("GLY", "HA3") == "HA"
    # get_valid_bbshifts rejects the whole chain for these, so there is no slot
    assert backbone_slot("LEU", "HB1") is None
    assert backbone_slot("LEU", "HA2") is None
    assert backbone_slot("LEU", "CD1") is None


def test_backbone_correction_is_lacs_ppm_plus_potenci_sigma_times_weight():
    """The unit trap, pinned. ``off_<atom>_sigma`` is a multiple of the POTENCI
    RMSD and must be scaled by REFINED_WEIGHTS before it can be subtracted from
    a ppm shift; ``lacs_off_<atom>_ppm`` must not be."""
    df = _toy_table()

    cb = _row(df, 1, "CB")
    expected = total_offset_ppm("CB", 0.16, -1.7826683938)
    assert cb["offset_applied_ppm"] == pytest.approx(expected)
    assert cb["val_corrected_ppm"] == pytest.approx(19.10 - expected)
    # the naive sigma+ppm sum is a different number: that is the whole point
    naive = 0.16 + -1.7826683938
    assert abs(expected - naive) > 1.0
    assert cb["offset_applied_ppm"] != pytest.approx(naive)


def test_backbone_uses_its_own_per_atom_offset():
    """N is corrected by the N offset, CA by the CA offset — never by a shared
    one."""
    df = _toy_table()

    assert _row(df, 3, "N")["offset_applied_ppm"] == pytest.approx(
        total_offset_ppm("N", 0.87, -0.6955098727)
    )
    assert _row(df, 2, "CA")["offset_applied_ppm"] == pytest.approx(
        total_offset_ppm("CA", 0.16, 0.0)
    )
    # degenerate partners inherit the slot's offset, not their own atom_id's
    for atom in ("HB2", "HB3"):
        assert _row(df, 2, atom)["offset_applied_ppm"] == pytest.approx(
            total_offset_ppm("HB", 0.0, -1.4204271123)
        )
    for atom in ("HA2", "HA3"):
        assert _row(df, 4, atom)["offset_applied_ppm"] == pytest.approx(
            total_offset_ppm("HA", -0.06, 0.0)
        )


def test_offset_source_names_the_estimators_that_contributed():
    df = _toy_table()

    assert set(df["offset_source"]) <= set(OFFSET_SOURCES)
    # CB: LACS 0.16 and POTENCI -1.78 sigma both non-zero
    assert _row(df, 1, "CB")["offset_source"] == "lacs+potenci"
    # CA: LACS 0.16, POTENCI rejected (0.0)
    assert _row(df, 2, "CA")["offset_source"] == "lacs_only"
    # HB: LACS does not cover HB (always 0.0), POTENCI -1.42 sigma
    assert _row(df, 2, "HB2")["offset_source"] == "potenci_only"
    # C: both estimators ran and both returned zero -> nothing was subtracted,
    # and the row says so with a MEASURED 0.0, not a null. This is the
    # "measured no-op" case _OFFSET_SOURCE_VALUES documents, and it is the one
    # backbone branch of _backbone_offset that nothing else exercises.
    record = dict(TOY_RECORD, off_C_sigma=0.0)
    zero = _toy_table(record)
    c = _row(zero, 2, "C")
    assert c["offset_source"] == "none"
    assert c["offset_applied_ppm"] == pytest.approx(0.0)
    assert c["val_corrected_ppm"] == pytest.approx(c["val_ppm"])
    assert set(zero.loc[zero["offset_source"] == "none", "offset_applied_ppm"]) <= {0.0}


def test_side_chain_protons_and_nitrogens_are_never_corrected():
    """Measured transfer slopes are 0.079 (1H) and 0.366 (15N), both below the
    0.5 break-even: applying the backbone offset inflates the error (MSE x1.131
    and x1.075). The corrected column is NULL, never a copy of the raw value."""
    df = _toy_table()
    sc = df[~df["is_backbone"]]

    for _, row in sc[sc["atom_type"].isin(["H", "N"])].iterrows():
        assert row["offset_source"] == "not_transferable"
        assert np.isnan(row["val_corrected_ppm"])
        assert np.isnan(row["offset_applied_ppm"])
    assert len(sc[sc["atom_type"].isin(["H", "N"])]) >= 3


def test_side_chain_carbons_use_the_chain_lacs_ca_offset():
    """Transfer slope 0.785 (side-chain MSE x0.79). Single atom, CA: the
    count-weighted and POTENCI-propagated variants all measure worse."""
    df = _toy_table()
    sc = df[~df["is_backbone"] & (df["atom_type"] == "C")]

    assert len(sc) == 3
    for _, row in sc.iterrows():
        assert row["offset_source"] == "lacs_only"
        assert row["offset_applied_ppm"] == pytest.approx(0.16)
        assert row["val_corrected_ppm"] == pytest.approx(row["val_ppm"] - 0.16)
    # explicitly NOT the POTENCI term: that transfers at only 0.093
    assert not any(
        row["offset_applied_ppm"] == pytest.approx(total_offset_ppm("CA", 0.16, 0.0))
        and row["offset_applied_ppm"] != pytest.approx(0.16)
        for _, row in sc.iterrows()
    )


def test_side_chain_carbon_offset_is_withheld_above_the_gate():
    """13 of 16,851 chains carry |lacs_off_CA| > 5 ppm — real gross deposition
    errors, but only 5 of the 9 with side-chain data transfer. Ungated the MSE
    ratio blows up to 7.42, so those rows ship raw with an explicit NULL."""
    record = dict(TOY_RECORD, lacs_off_CA_ppm=SIDECHAIN_MAX_ABS_LACS_PPM + 0.01)
    df = _toy_table(record)
    sc = df[~df["is_backbone"] & (df["atom_type"] == "C")]

    assert len(sc) == 3
    assert set(sc["offset_source"]) == {"not_transferable"}
    assert sc["val_corrected_ppm"].isna().all()
    assert sc["offset_applied_ppm"].isna().all()
    # the backbone rows of the same chain still carry it: the scorer applied it
    assert _row(df, 2, "CA")["offset_applied_ppm"] == pytest.approx(
        total_offset_ppm("CA", SIDECHAIN_MAX_ABS_LACS_PPM + 0.01, 0.0)
    )


def test_a_zero_lacs_offset_is_a_no_op_not_a_null():
    """LACS returning 0.0 is a genuine no-op (it never fires below 20 backbone
    CA observations), so the side-chain carbons ship corrected == raw with
    ``offset_source='none'`` — distinguishable from a withheld value."""
    df = _toy_table(dict(TOY_RECORD, lacs_off_CA_ppm=0.0))
    sc = df[~df["is_backbone"] & (df["atom_type"] == "C")]

    assert set(sc["offset_source"]) == {"none"}
    assert (sc["offset_applied_ppm"] == 0.0).all()
    assert (sc["val_corrected_ppm"] == sc["val_ppm"]).all()


def test_corrected_column_is_never_a_silent_copy():
    """Either ``val_corrected_ppm == val_ppm - offset_applied_ppm`` with the
    offset stated, or both are NULL. There is no third case where a raw value
    is passed off as corrected."""
    for record in (
        TOY_RECORD,
        dict(TOY_RECORD, lacs_off_CA_ppm=9.9),
        dict(TOY_RECORD, lacs_off_CA_ppm=0.0),
    ):
        df = _toy_table(record)
        withheld = df["val_corrected_ppm"].isna()
        assert (df.loc[withheld, "offset_applied_ppm"].isna()).all()
        assert (
            df.loc[withheld, "offset_source"].isin(["not_transferable", "none"])
        ).all()
        stated = df.loc[~withheld]
        np.testing.assert_allclose(
            stated["val_corrected_ppm"],
            stated["val_ppm"] - stated["offset_applied_ppm"],
            rtol=0,
            atol=1e-12,
        )


# --------------------------------------------------------------------------- #
# the Parquet contract
# --------------------------------------------------------------------------- #


def test_shift_parquet_schema_documents_every_unit(tmp_path):
    """A user reading the schema cannot get the units wrong: every numeric field
    carries its unit, and the table states the correction formula."""
    pq = pytest.importorskip("pyarrow.parquet")

    from trizod.shifts import SHIFTS_PARQUET_NAME, write_shift_parquet

    frames = [("1_1_1_1", _toy_table())]
    path, n_rows = write_shift_parquet(frames, tmp_path / SHIFTS_PARQUET_NAME)
    assert n_rows == len(TOY_SHIFTS)

    schema = pq.read_schema(path)
    assert schema.names == [
        "id",
        "seq_id",
        "comp_id",
        "atom_id",
        "atom_type",
        "val_ppm",
        "val_err_ppm",
        "ambiguity_code",
        "val_corrected_ppm",
        "offset_applied_ppm",
        "offset_source",
        "is_backbone",
    ]
    for name in ("val_ppm", "val_err_ppm", "val_corrected_ppm", "offset_applied_ppm"):
        meta = schema.field(name).metadata or {}
        assert meta.get(b"unit") == b"ppm", f"{name} does not declare its unit"
    assert (schema.field("seq_id").metadata or {}).get(b"unit") == b"residue index"
    table_meta = schema.metadata or {}
    formula = table_meta.get(b"correction_formula", b"").decode()
    assert "val_corrected_ppm" in formula and "offset_applied_ppm" in formula
    assert b"offset_source_values" in table_meta


def test_shift_parquet_round_trip_and_join(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")

    from trizod.shifts import write_shift_parquet

    main = {"17665_1_1_1": TOY_SEQ, "17665_2_1_1": TOY_SEQ}
    frames = [(cid, _toy_table()) for cid in main]
    path, n_rows = write_shift_parquet(frames, tmp_path / "trizod_shifts.parquet")

    got = pq.read_table(path).to_pydict()
    assert set(got["id"]) == set(main)
    keys = list(zip(got["id"], got["seq_id"], got["atom_id"]))
    assert keys == sorted(keys)
    for cid, seq_id, comp_id in zip(got["id"], got["seq_id"], got["comp_id"]):
        assert main[cid][seq_id - 1] == AA3TO1[comp_id]
    assert 3 in got["ambiguity_code"]
    assert True in got["is_backbone"] and False in got["is_backbone"]
    assert None in got["val_corrected_ppm"], "withheld corrections must be NULL"


# --------------------------------------------------------------------------- #
# real data: the corrected column reproduces the shift the scorer used
# --------------------------------------------------------------------------- #


def _scored_ppm(seq, shifts, predshiftdct):
    """The per-atom ppm shift the scoring path effectively compared to POTENCI.

    LACS comes off the raw ppm array; the POTENCI residual offset comes off the
    sigma-scaled diffs. Expressed entirely in ppm that is
    ``raw - lacs_off_ppm - off_sigma * REFINED_WEIGHTS[atom]``.
    """
    from trizod.scoring.scoring import get_offset_corrected_shifts

    ret = get_offset_corrected_shifts(seq, shifts, predshiftdct)
    offsets, lacs = ret[4], ret[9]
    arr, mask = get_valid_bbshifts(shifts, seq)
    corrected = arr.copy()
    for j, atom in enumerate(BACKBONE_ATOMS):
        corrected[:, j] -= lacs[atom] + offsets[atom] * REFINED_WEIGHTS[atom]
    return corrected, mask, offsets, lacs


@requires_bmrb_data
@pytest.mark.parametrize("entry_id", ["17665", "16663", "25501"])
def test_corrected_column_reproduces_the_scored_shift(entry_id):
    """THE load-bearing test. For every backbone slot the scorer populated from
    a single deposited row, ``val_corrected_ppm`` must equal the ppm shift the
    scorer compared against POTENCI — which is only true if the sigma offset was
    scaled and the ppm offset was not."""
    import trizod.bmrb.bmrb as bmrb
    import trizod.potenci.potenci as potenci

    entry_dir = BMRB_DIR / f"bmr{entry_id}"
    if not entry_dir.exists():
        pytest.skip(f"BMRB {entry_id} not in BMRB_DIR")
    entry = bmrb.BmrbEntry(entry_id, entry_dir)

    n_checked = 0
    for (_, _, e_id), (shifts, cond_id, _, _) in entry.get_peptide_shifts().items():
        seq = entry.entities[e_id].seq
        if not seq or get_valid_bbshifts(shifts, seq) is None:
            continue
        cond = entry.conditions[cond_id]
        predshiftdct = potenci.get_pred_shifts(
            seq,
            cond.get_temperature(return_default=True),
            cond.get_pH(return_default=True),
            cond.get_ionic_strength(return_default=True),
            cond.get_pH(return_default=True) != 7.0,
        )
        scored = _scored_ppm(seq, shifts, predshiftdct)
        if scored is None:
            continue
        corrected, mask, offsets, lacs = scored

        record = {}
        for atom in BACKBONE_ATOMS:
            record[f"lacs_off_{atom}_ppm"] = lacs[atom]
            record[f"off_{atom}_sigma"] = offsets[atom]
        df = annotate_offsets(get_deposited_shifts(shifts, seq), chain_offsets(record))
        bb = df[df["is_backbone"]].copy()
        bb["slot"] = [backbone_slot(c, a) for c, a in zip(bb["comp_id"], bb["atom_id"])]

        for pos, j in zip(*np.where(mask)):
            cand = bb[(bb["seq_id"] == pos + 1) & (bb["slot"] == BACKBONE_ATOMS[j])]
            if len(cand) != 1:
                continue  # degenerate partners: covered by the mean check below
            assert cand.iloc[0]["val_corrected_ppm"] == pytest.approx(
                corrected[pos, j], abs=1e-9
            ), f"{entry_id} {pos + 1}/{BACKBONE_ATOMS[j]}"
            n_checked += 1
    assert n_checked > 100, f"only {n_checked} backbone slots checked for {entry_id}"


@requires_bmrb_data
def test_scoring_is_bit_identical_with_and_without_the_shift_table():
    """Building the published table must not perturb the scoring path."""
    import trizod.bmrb.bmrb as bmrb
    import trizod.potenci.potenci as potenci
    from trizod.scoring.scoring import (
        compute_gscores,
        compute_zscores,
        convert_to_triplet_data,
        get_offset_corrected_shifts,
    )

    entry_dir = BMRB_DIR / "bmr17665"
    if not entry_dir.exists():
        pytest.skip("BMRB 17665 not in BMRB_DIR")
    entry = bmrb.BmrbEntry("17665", entry_dir)
    (_, _, e_id), (shifts, cond_id, _, _) = next(
        iter(entry.get_peptide_shifts().items())
    )
    seq = entry.entities[e_id].seq
    cond = entry.conditions[cond_id]
    predshiftdct = potenci.get_pred_shifts(
        seq,
        cond.get_temperature(return_default=True),
        cond.get_pH(return_default=True),
        cond.get_ionic_strength(return_default=True),
        cond.get_pH(return_default=True) != 7.0,
    )

    def score():
        ret = get_offset_corrected_shifts(seq, shifts, predshiftdct)
        ashwi, cmp_mask, offsets, lacs = ret[1], ret[2], ret[4], ret[9]
        diffs, dof = convert_to_triplet_data(ashwi, cmp_mask)
        return (
            compute_zscores(diffs, dof, cmp_mask),
            compute_gscores(diffs, dof, cmp_mask),
            dof,
            offsets,
            lacs,
        )

    shifts_before = copy.deepcopy(shifts)
    arr0, mask0 = get_valid_bbshifts(shifts, seq)
    z0, g0, k0, off0, lacs0 = score()

    df = get_deposited_shifts(shifts, seq)
    assert df is not None and df["is_backbone"].any() and (~df["is_backbone"]).any()

    arr1, mask1 = get_valid_bbshifts(shifts, seq)
    z1, g1, k1, off1, lacs1 = score()

    np.testing.assert_array_equal(arr0, arr1)
    np.testing.assert_array_equal(mask0, mask1)
    np.testing.assert_array_equal(z0, z1)
    np.testing.assert_array_equal(g0, g1)
    np.testing.assert_array_equal(k0, k1)
    assert off0 == off1
    assert lacs0 == lacs1
    assert shifts == shifts_before, "get_deposited_shifts mutated the shift tuples"


def test_a_crashing_frame_generator_leaves_no_truncated_parquet(tmp_path):
    """Closing a ParquetWriter mid-stream still writes a valid footer, so an
    in-place write would leave a structurally perfect, silently SHORT table --
    which package_release stages automatically and whose n_records it reads back
    out of the file itself. Nothing may survive at the destination."""
    pytest.importorskip("pyarrow.parquet")

    from trizod.shifts import write_shift_parquet

    def frames():
        yield "17665_1_1_1", _toy_table()
        raise RuntimeError("entry 99999 blew up mid-corpus")

    out = tmp_path / "trizod_shifts.parquet"
    with pytest.raises(RuntimeError, match="blew up"):
        write_shift_parquet(frames(), out)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []
