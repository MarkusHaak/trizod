"""Tests for ``output_dataset()`` shift emission (``--include-shifts``)."""

import numpy as np
import pandas as pd
import pytest

from trizod.constants import BACKBONE_ATOMS
from trizod.trizod import output_dataset

SIDECHAIN_SPLIT_ATOMS = ["HA2", "HA3", "HB1", "HB2", "HB3"]


def _make_df(n_cols):
    """Minimal one-row scored DataFrame with an (N, n_cols) bbshifts array."""
    seq = "AGAGA"
    bbshifts = np.arange(len(seq) * n_cols, dtype=float).reshape(len(seq), n_cols)
    return pd.DataFrame(
        {
            "entryID": ["12345"],
            "stID": ["1"],
            "entity_assemID": ["1"],
            "entityID": ["1"],
            "entity_name": ["test peptide"],
            "seq": [seq],
            "k": [[3] * len(seq)],
            "zscores": [[0.5] * len(seq)],
            "pass_post": [True],
            "bbshifts": [bbshifts],
        }
    )


def _read_csv(tmp_path, prefix):
    return pd.read_csv(tmp_path / f"{prefix}.csv")


def test_include_shifts_without_no_averaging(tmp_path):
    """--include-shifts alone must emit the averaged backbone shift columns."""
    df = _make_df(len(BACKBONE_ATOMS))
    bbshifts = df.at[0, "bbshifts"]
    output_dataset(
        df,
        tmp_path / "scores",
        "csv",
        ["zscores"],
        4,
        include_shifts=True,
        no_shift_averaging=False,
    )
    dout = _read_csv(tmp_path, "scores")
    for i, atom_type in enumerate(BACKBONE_ATOMS):
        assert atom_type in dout.columns, f"missing shift column {atom_type}"
        assert dout[atom_type].to_numpy() == pytest.approx(bbshifts[:, i])


def test_include_shifts_with_no_averaging_emits_split_atoms(tmp_path):
    """--no-shift-averaging keeps the 12-column layout, in bbshifts order."""
    expected = list(BACKBONE_ATOMS) + SIDECHAIN_SPLIT_ATOMS
    df = _make_df(len(expected))
    bbshifts = df.at[0, "bbshifts"]
    output_dataset(
        df,
        tmp_path / "scores",
        "csv",
        ["zscores"],
        4,
        include_shifts=True,
        no_shift_averaging=True,
    )
    dout = _read_csv(tmp_path, "scores")
    for i, atom_type in enumerate(expected):
        assert atom_type in dout.columns, f"missing shift column {atom_type}"
        assert dout[atom_type].to_numpy() == pytest.approx(bbshifts[:, i])


def test_no_include_shifts_emits_no_shift_columns(tmp_path):
    """Without --include-shifts no shift column is written."""
    df = _make_df(len(BACKBONE_ATOMS))
    output_dataset(
        df,
        tmp_path / "scores",
        "csv",
        ["zscores"],
        4,
        include_shifts=False,
        no_shift_averaging=False,
    )
    dout = _read_csv(tmp_path, "scores")
    for atom_type in list(BACKBONE_ATOMS) + SIDECHAIN_SPLIT_ATOMS:
        assert atom_type not in dout.columns


def test_include_shifts_does_not_mutate_backbone_atoms(tmp_path):
    """The shift-name list must be a copy: BACKBONE_ATOMS is module-global."""
    before = list(BACKBONE_ATOMS)
    df = _make_df(len(BACKBONE_ATOMS) + len(SIDECHAIN_SPLIT_ATOMS))
    output_dataset(
        df,
        tmp_path / "scores",
        "csv",
        ["zscores"],
        4,
        include_shifts=True,
        no_shift_averaging=True,
    )
    assert list(BACKBONE_ATOMS) == before
