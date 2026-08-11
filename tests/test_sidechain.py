"""Tests for the discarded side-chain chemical shifts (companion table).

``get_valid_bbshifts()`` keeps 12 atom IDs and drops everything else — ~3.5 M
side-chain values corpus-wide. ``get_sidechain_shifts()`` is the parallel read
that recovers exactly those dropped values, without touching the backbone path:
scoring must stay bit-identical.
"""

import copy

import numpy as np
import pytest

from tests.conftest import BMRB_DIR, requires_bmrb_data
from trizod.bmrb.bmrb import get_sidechain_shifts, get_valid_bbshifts
from trizod.constants import AA3TO1, BACKBONE_ATOMS

# The atom IDs get_valid_bbshifts() consumes; the side-chain table is their
# complement, so the two views must never overlap.
BB_ATOM_IDS = set(BACKBONE_ATOMS) | {"HA2", "HA3", "HB1", "HB2", "HB3"}

# 3-residue toy chain (ALA-LEU-PHE). Tuple layout mirrors ShiftTable.shifts:
# (entity_assembly_ID, entity_ID, Seq_ID, Comp_ID, Atom_ID, Atom_type, Val,
#  Val_err, Ambiguity_code), all strings as parsed from NMR-STAR.
TOY_SEQ = "ALF"
TOY_SHIFTS = [
    ("1", "1", "1", "ALA", "CA", "C", "52.30", "0.10", "1"),
    ("1", "1", "1", "ALA", "CB", "C", "19.10", "0.10", "1"),
    ("1", "1", "1", "ALA", "HB1", "H", "1.35", "0.02", "1"),
    ("1", "1", "2", "LEU", "CA", "C", "55.10", "0.10", "1"),
    ("1", "1", "2", "LEU", "HB2", "H", "1.60", "0.02", "2"),
    ("1", "1", "2", "LEU", "CD1", "C", "24.50", "0.20", "2"),
    ("1", "1", "2", "LEU", "CD2", "C", "23.10", "0.20", "2"),
    ("1", "1", "2", "LEU", "HD11", "H", "0.85", "5.00", "2"),
    ("1", "1", "3", "PHE", "N", "N", "120.40", "0.20", "1"),
    ("1", "1", "3", "PHE", "CD1", "C", "131.50", "0.30", "3"),
    ("1", "1", "3", "PHE", "HD1", "H", "7.15", "", "3"),
]


def test_get_sidechain_shifts_disjoint_from_backbone():
    """The side-chain view is exactly the complement of the backbone view."""
    df = get_sidechain_shifts(TOY_SHIFTS, TOY_SEQ)

    assert df is not None
    assert not set(df["atom_id"]) & BB_ATOM_IDS, (
        f"backbone atoms leaked into the side-chain table: "
        f"{sorted(set(df['atom_id']) & BB_ATOM_IDS)}"
    )
    # every non-backbone atom of the toy table is recovered, none is lost
    assert set(zip(df["seq_id"], df["atom_id"])) == {
        (2, "CD1"),
        (2, "CD2"),
        (2, "HD11"),
        (3, "CD1"),
        (3, "HD1"),
    }
    # Seq_ID is 1-based as deposited and consistent with the polymer sequence
    for seq_id, comp_id in zip(df["seq_id"], df["comp_id"]):
        assert TOY_SEQ[seq_id - 1] == AA3TO1[comp_id]
    # values/errors are numeric, the deposited error is carried through
    assert df.loc[df["atom_id"] == "CD1", "val"].tolist() == [24.50, 131.50]
    assert df.loc[df["atom_id"] == "HD11", "val_err"].tolist() == [5.00]


def test_sidechain_shifts_keep_large_errors_and_all_ambiguity_codes():
    """Neither the ``max_err <= 1.3`` cut nor the backbone ambiguity whitelist
    applies. Aromatic ring-degenerate values (code 3) are 85,466 of the dropped
    set corpus-wide — applying the backbone rule would delete all of them."""
    df = get_sidechain_shifts(TOY_SHIFTS, TOY_SEQ)

    codes = dict(
        zip(df["atom_id"] + "@" + df["seq_id"].astype(str), df["ambiguity_code"])
    )
    assert codes["CD1@3"] == 3, "ambiguity code 3 (aromatic degenerate) was dropped"
    assert codes["HD1@3"] == 3
    assert codes["CD2@2"] == 2
    # Val_err 5.00 ppm is far above the backbone max_err of 1.3 and must survive
    assert (df["val_err"] > 1.3).any()


def test_sidechain_shifts_survive_malformed_metadata():
    """A junk Val_err or Ambiguity_code must not abort the extraction: those two
    columns are carried metadata, not guards. Codes outside the BMRB domain
    (1-9) become null rather than poisoning the int8 column."""
    shifts = [
        ("1", "1", "1", "LEU", "CD1", "C", "24.50", "n/a", "?"),
        ("1", "1", "1", "LEU", "CD2", "C", "23.10", ".", "42"),
        ("1", "1", "1", "LEU", "HD11", "H", "0.85", "0.01", "2"),
    ]
    df = get_sidechain_shifts(shifts, "L")

    assert df is not None and len(df) == 3
    assert df["val"].tolist() == [24.50, 23.10, 0.85]
    assert df["val_err"].isna().tolist() == [True, True, False]
    assert df["ambiguity_code"].isna().tolist() == [True, True, False]


def test_sidechain_shifts_reject_sequence_mismatch():
    """Same sequence-consistency guard as the backbone read: a Comp_ID that
    contradicts the polymer sequence invalidates the whole table, so
    ``sequence[seq_id - 1]`` is a safe join key downstream."""
    assert get_sidechain_shifts(TOY_SHIFTS, "AAF") is None
    assert get_valid_bbshifts(TOY_SHIFTS, "AAF") is None


@requires_bmrb_data
def test_sidechain_disjoint_from_backbone_on_real_entry():
    """Same disjointness on a real deposition, over every chain of the entry."""
    import trizod.bmrb.bmrb as bmrb

    entry_dir = BMRB_DIR / "bmr17665"
    if not entry_dir.exists():
        pytest.skip("BMRB 17665 not in BMRB_DIR")
    entry = bmrb.BmrbEntry("17665", entry_dir)

    n_sidechain = 0
    for (_, _, e_id), (shifts, _, _, _) in entry.get_peptide_shifts().items():
        seq = entry.entities[e_id].seq
        df = get_sidechain_shifts(shifts, seq)
        if df is None:
            continue
        n_sidechain += len(df)
        assert not set(df["atom_id"]) & BB_ATOM_IDS
    assert n_sidechain > 0, "expected side-chain shifts in bmr17665"


@requires_bmrb_data
def test_scoring_unchanged_by_sidechain_extraction():
    """Extracting side chains must not perturb the scoring path: z/g/k and both
    offset dicts are bit-identical, and the raw shift tuples are untouched."""
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
    z0, g0, k0, off0, lacs0 = score()

    df = get_sidechain_shifts(shifts, seq)
    assert df is not None and len(df) > 0

    z1, g1, k1, off1, lacs1 = score()

    np.testing.assert_array_equal(z0, z1)
    np.testing.assert_array_equal(g0, g1)
    np.testing.assert_array_equal(k0, k1)
    assert off0 == off1
    assert lacs0 == lacs1
    assert shifts == shifts_before, "get_sidechain_shifts mutated the shift tuples"


def test_sidechain_parquet_join(tmp_path):
    """The companion Parquet joins 1:1 on ``id`` with the main table, is sorted
    by (id, seq_id, atom_id), and its (seq_id, comp_id) pairs agree with the
    main table's sequence."""
    pq = pytest.importorskip("pyarrow.parquet")

    from trizod.sidechain import write_sidechain_parquet

    # two chains of the same "main table": id -> sequence
    main = {"17665_1_1_1": TOY_SEQ, "17665_2_1_1": TOY_SEQ}
    frames = [(cid, get_sidechain_shifts(TOY_SHIFTS, seq)) for cid, seq in main.items()]

    out = tmp_path / "trizod_sidechain_shifts.parquet"
    path, n_rows = write_sidechain_parquet(frames, out)
    assert path.exists() and n_rows == sum(len(f) for _, f in frames)

    table = pq.read_table(path)
    assert table.column_names == [
        "id",
        "seq_id",
        "comp_id",
        "atom_id",
        "atom_type",
        "val",
        "val_err",
        "ambiguity_code",
    ]
    got = table.to_pydict()

    # 1:1 join: no orphan ids
    assert set(got["id"]) <= set(main)
    assert set(got["id"]) == set(main), "every chain with side chains must be present"
    # sort order
    keys = list(zip(got["id"], got["seq_id"], got["atom_id"]))
    assert keys == sorted(keys)
    # sequence agreement against the main table
    for cid, seq_id, comp_id in zip(got["id"], got["seq_id"], got["comp_id"]):
        assert main[cid][seq_id - 1] == AA3TO1[comp_id]
    # ambiguity code 3 survives the round-trip
    assert 3 in got["ambiguity_code"]
