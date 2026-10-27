"""The far-outlier flag on ``trizod_shifts.parquet`` (Sandro Kuppel's MAD rule).

Sandro found deposition errors (shifts hundreds of ppm off, "flipped" spectra)
skewing his train/validation/test losses and removed them with a median
absolute deviation filter per residue type x atom: k = 8 for C, CA and N, 12 for
CB, nothing for protons. The release ships that rule as two columns rather than
a filter -- the continuous ``mad_z`` and the boolean ``is_far_outlier`` -- so
nothing is dropped and the scores are untouched. The medians and MADs come from
a reference table frozen in the repository, so the flag does not drift with the
next BMRB snapshot.
"""

import pandas as pd
import pytest

from trizod.constants import AA3TO1
from trizod.shifts.outliers import (
    MAD_K,
    annotate_mad,
    compute_mad_reference,
    load_mad_reference,
    write_mad_reference,
)


def _frame(rows):
    """``(seq_id, comp_id, atom_id, val_ppm)`` rows -> a shift-table slice."""
    return pd.DataFrame(rows, columns=["seq_id", "comp_id", "atom_id", "val_ppm"])


def _stats(ref, comp_id, atom_id):
    return ref.set_index(["comp_id", "atom_id"]).loc[(comp_id, atom_id)]


#: Hand-made reference: one MAD of 1 ppm (2 ppm for N) around round medians.
REF = pd.DataFrame(
    {
        "comp_id": ["ALA", "ALA", "ALA", "ALA"],
        "atom_id": ["C", "CA", "CB", "N"],
        "n": [100, 100, 100, 100],
        "median_ppm": [177.0, 52.0, 19.0, 124.0],
        "mad_ppm": [1.0, 1.0, 1.0, 2.0],
    }
)


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #


def test_thresholds_are_the_ones_sandro_chose():
    assert MAD_K == {"C": 8.0, "CA": 8.0, "N": 8.0, "CB": 12.0}


def test_reference_is_the_median_and_raw_mad_per_residue_and_atom():
    df = _frame(
        [(2, "ALA", "CA", v) for v in (50.0, 52.0, 53.0, 54.0, 70.0)]
        + [(2, "GLY", "CA", 45.0), (3, "GLY", "CA", 46.0)]
    )
    ref = compute_mad_reference(df)

    ala = _stats(ref, "ALA", "CA")
    # |deviations| from 53 are 3, 1, 0, 1, 17 -> MAD 1, unscaled (not x1.4826)
    assert (ala.n, ala.median_ppm, ala.mad_ppm) == (5, 53.0, 1.0)
    gly = _stats(ref, "GLY", "CA")
    assert (gly.n, gly.median_ppm, gly.mad_ppm) == (2, 45.5, 0.5)


def test_reference_covers_only_the_flagged_backbone_heavy_atoms():
    df = _frame(
        [
            (2, "ALA", "H", 8.1),
            (2, "ALA", "HA", 4.3),
            (2, "ALA", "HB1", 1.4),
            (2, "LEU", "CD1", 24.5),
            (2, "ALA", "CB", 19.0),
        ]
    )
    ref = compute_mad_reference(df)
    assert set(zip(ref["comp_id"], ref["atom_id"])) == {("ALA", "CB")}


def test_reference_leaves_out_the_n_terminal_amine():
    """Residue 1's N is a free amine (~35-40 ppm), not an amide: real physics,
    not a deposition error, and not part of the amide-N distribution."""
    df = _frame([(1, "MET", "N", 35.0), (5, "MET", "N", 120.0), (9, "MET", "N", 121.0)])
    met = _stats(compute_mad_reference(df), "MET", "N")
    assert (met.n, met.median_ppm) == (2, 120.5)


def test_mad_z_is_the_signed_deviation_in_raw_mad_units():
    out = annotate_mad(_frame([(2, "ALA", "CA", 55.0), (2, "ALA", "N", 120.0)]), REF)
    assert out["mad_z"].tolist() == pytest.approx([3.0, -2.0])


def test_flag_is_beyond_8_mads_for_c_ca_n_and_12_for_cb():
    out = annotate_mad(
        _frame(
            [
                (2, "ALA", "CA", 62.0),  # +10 MADs on CA -> outlier
                (2, "ALA", "CB", 29.0),  # +10 MADs on CB -> inside the wider 12
                (2, "ALA", "CB", 32.0),  # +13 MADs on CB -> outlier
                (2, "ALA", "N", 107.0),  # -8.5 MADs -> outlier in either direction
                (2, "ALA", "C", 185.0),  # exactly 8 MADs -> not beyond the limit
            ]
        ),
        REF,
    )
    assert out["is_far_outlier"].tolist() == [True, False, True, True, False]


def test_mad_z_is_rounded_to_a_hundredth_and_the_flag_agrees_with_it():
    """0.01 MAD is ~0.016 ppm for CA, finer than shifts are deposited, and it
    shrinks the column fourfold. The flag is computed from the ROUNDED value, so
    ``|mad_z| > k`` on the published column reproduces it exactly."""
    out = annotate_mad(
        _frame([(2, "ALA", "CA", 60.004), (2, "ALA", "CA", 60.006)]), REF
    )
    assert out["mad_z"].dtype == "float32"
    assert out["mad_z"].tolist() == pytest.approx([8.0, 8.01])
    assert out["is_far_outlier"].tolist() == [False, True]


def test_protons_side_chains_and_the_n_terminal_amine_are_not_assessed():
    """Not assessed is NULL in both columns -- never False, which would read as
    "checked, and fine"."""
    out = annotate_mad(
        _frame(
            [
                (2, "ALA", "H", 8.2),
                (2, "ALA", "HA", 4.3),
                (2, "ALA", "HB1", 1.4),
                (1, "ALA", "N", 35.0),
                (3, "LEU", "CD1", 24.5),
                (2, "ALA", "CA", 52.5),
            ]
        ),
        REF,
    )
    assert out["mad_z"].isna().tolist() == [True] * 5 + [False]
    assert out["is_far_outlier"].isna().tolist() == [True] * 5 + [False]
    assert out["is_far_outlier"].iloc[-1] == False  # noqa: E712 - pandas NA-aware


def test_a_group_missing_from_the_reference_is_not_assessed():
    out = annotate_mad(_frame([(2, "TRP", "CA", 57.0)]), REF)
    assert pd.isna(out["mad_z"].iloc[0])
    assert pd.isna(out["is_far_outlier"].iloc[0])


def test_annotate_leaves_its_input_alone_and_keeps_row_order():
    df = _frame(
        [(3, "ALA", "N", 124.0), (2, "ALA", "CA", 52.0), (2, "ALA", "C", 177.0)]
    )
    before = df.copy()
    out = annotate_mad(df, REF)
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(out[list(df.columns)], df)
    assert out["mad_z"].tolist() == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# the frozen reference table
# --------------------------------------------------------------------------- #


def test_reference_round_trips_through_its_csv(tmp_path):
    path = tmp_path / "mad_reference.csv"
    write_mad_reference(REF, path, source="unit test")
    back = load_mad_reference(path)
    pd.testing.assert_frame_equal(back, REF, check_dtype=False)
    assert any("unit test" in line for line in path.read_text().splitlines())


def test_the_frozen_reference_covers_every_canonical_residue():
    ref = load_mad_reference()
    expected = {
        (aa, atom) for aa in AA3TO1 for atom in MAD_K if (aa, atom) != ("GLY", "CB")
    }
    assert set(zip(ref["comp_id"], ref["atom_id"])) == expected
    assert (ref["mad_ppm"] > 0).all()
    assert (ref["n"] >= 1000).all(), "a group too small to trust as a reference"


@pytest.mark.parametrize(
    ("comp_id", "atom_id", "low", "high"),
    [
        ("ALA", "CA", 51.5, 54.0),
        ("GLY", "CA", 44.0, 46.5),
        ("ALA", "CB", 18.0, 20.0),
        ("THR", "CB", 68.5, 71.0),
        ("PRO", "C", 175.5, 178.0),
        ("GLY", "N", 107.0, 111.0),
    ],
)
def test_the_frozen_reference_medians_are_textbook_values(comp_id, atom_id, low, high):
    """Guards against swapped columns or a reference built from corrected
    values in the wrong unit: every median must sit where NMR textbooks put it."""
    assert low <= _stats(load_mad_reference(), comp_id, atom_id).median_ppm <= high
