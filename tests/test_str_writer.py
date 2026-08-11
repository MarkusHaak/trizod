"""Tests for the re-referenced .str emitter."""

import numpy as np
import pynmrstar
import pytest

from trizod.constants import BACKBONE_ATOMS, REFINED_WEIGHTS
from trizod.io.str_writer import write_rereferenced_str


def _write_minimal(out_path, lacs_offsets, potenci_offsets):
    """Emit a one-residue file carrying the given offsets, return the parsed entry."""
    write_rereferenced_str(
        out_path,
        entry_id="00001",
        seq="A",
        bbshifts_arr=np.zeros((1, len(BACKBONE_ATOMS))),
        bbshifts_mask=np.zeros((1, len(BACKBONE_ATOMS)), dtype=bool),
        lacs_offsets=lacs_offsets,
        potenci_residual_offsets=potenci_offsets,
        rereference_mode="both",
        pipeline_version="trizod-2026-07-14",
    )
    return pynmrstar.Entry.from_file(str(out_path))


def _loop_value(loop, atom, tag):
    """Value of `tag` for `atom` in a per-atom offset loop, as float."""
    atom_col = loop.tag_index("Atom_ID")
    val_col = loop.tag_index(tag)
    for row in loop.data:
        if row[atom_col] == atom:
            return float(row[val_col])
    raise AssertionError(f"atom {atom} not found in loop")


def test_write_rereferenced_str_round_trip(tmp_path):
    """Emit, then parse back; corrected shifts match within 1e-4 ppm."""
    seq = "AGAGAGAGAG"
    bbshifts_arr = np.zeros((10, len(BACKBONE_ATOMS)))
    bbshifts_mask = np.zeros((10, len(BACKBONE_ATOMS)), dtype=bool)
    ca_idx = BACKBONE_ATOMS.index("CA")
    cb_idx = BACKBONE_ATOMS.index("CB")
    bbshifts_arr[:, ca_idx] = np.linspace(50.0, 56.0, 10)
    bbshifts_arr[:, cb_idx] = np.linspace(18.0, 22.0, 10)
    bbshifts_mask[:, ca_idx] = True
    bbshifts_mask[:, cb_idx] = True

    lacs_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    lacs_offsets["CA"] = 0.5
    lacs_offsets["CB"] = -0.3
    potenci_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)

    out_path = tmp_path / "bmr12345_rereferenced.str"
    write_rereferenced_str(
        out_path,
        entry_id="12345",
        seq=seq,
        bbshifts_arr=bbshifts_arr,
        bbshifts_mask=bbshifts_mask,
        lacs_offsets=lacs_offsets,
        potenci_residual_offsets=potenci_offsets,
        rereference_mode="both",
        pipeline_version="trizod-2026-07-14",
    )

    parsed = pynmrstar.Entry.from_file(str(out_path))
    shift_loops = parsed.get_loops_by_category("Atom_chem_shift")
    assert len(shift_loops) == 1
    loop = shift_loops[0]
    rows = loop.data
    ca_rows = [r for r in rows if r[loop.tag_index("Atom_ID")] == "CA"]
    assert len(ca_rows) == 10
    first_ca_val = float(ca_rows[0][loop.tag_index("Val")])
    assert abs(first_ca_val - bbshifts_arr[0, ca_idx]) < 1e-4


def test_write_rereferenced_str_records_offsets_in_aux(tmp_path):
    """Auxiliary metadata block records LACS + POTENCI residual offsets."""
    out_path = tmp_path / "bmr00001_rereferenced.str"
    bbshifts_arr = np.zeros((1, len(BACKBONE_ATOMS)))
    bbshifts_mask = np.zeros((1, len(BACKBONE_ATOMS)), dtype=bool)
    lacs_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    lacs_offsets["CA"] = 1.5
    potenci_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    potenci_offsets["CA"] = 0.1

    write_rereferenced_str(
        out_path,
        entry_id="00001",
        seq="A",
        bbshifts_arr=bbshifts_arr,
        bbshifts_mask=bbshifts_mask,
        lacs_offsets=lacs_offsets,
        potenci_residual_offsets=potenci_offsets,
        rereference_mode="both",
        pipeline_version="trizod-2026-07-14",
    )
    text = out_path.read_text()
    assert "LACS_offsets" in text
    assert "POTENCI_residual_offsets" in text
    assert "1.5" in text
    assert "trizod-2026-07-14" in text


def test_potenci_residual_offsets_not_labelled_ppm(tmp_path):
    """POTENCI/AIC offsets arrive in sigma units and must not be tagged as ppm.

    `compute_offsets()` averages `diff_arr / REFINED_WEIGHTS`, so `off_<atom>`
    is a multiple of the per-atom POTENCI RMSD, not a ppm offset.
    """
    lacs_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    potenci_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    potenci_offsets["CA"] = 2.0
    parsed = _write_minimal(
        tmp_path / "bmr00001_rereferenced.str", lacs_offsets, potenci_offsets
    )
    loop = parsed.get_loops_by_category("POTENCI_residual_offsets")[0]
    assert "Offset_sigma" in loop.tags
    assert _loop_value(loop, "CA", "Offset_sigma") == pytest.approx(2.0)


def test_potenci_residual_offset_ppm_conversion(tmp_path):
    """The emitted ppm value is the sigma offset times the atom's POTENCI RMSD."""
    lacs_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    potenci_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    potenci_offsets["CA"] = 2.0
    potenci_offsets["HA"] = -3.5
    parsed = _write_minimal(
        tmp_path / "bmr00001_rereferenced.str", lacs_offsets, potenci_offsets
    )
    loop = parsed.get_loops_by_category("POTENCI_residual_offsets")[0]
    assert _loop_value(loop, "CA", "Offset_ppm") == pytest.approx(
        2.0 * REFINED_WEIGHTS["CA"], abs=1e-6
    )
    assert _loop_value(loop, "HA", "Offset_ppm") == pytest.approx(
        -3.5 * REFINED_WEIGHTS["HA"], abs=1e-6
    )
    # sanity: the sigma value alone would be an absurd proton offset
    assert abs(_loop_value(loop, "HA", "Offset_ppm")) < 0.2


def test_lacs_offsets_stay_in_ppm(tmp_path):
    """LACS offsets are subtracted from raw shifts, so they really are ppm."""
    lacs_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    lacs_offsets["CA"] = 1.5
    potenci_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    parsed = _write_minimal(
        tmp_path / "bmr00001_rereferenced.str", lacs_offsets, potenci_offsets
    )
    loop = parsed.get_loops_by_category("LACS_offsets")[0]
    assert "Offset_ppm" in loop.tags
    assert "Offset_sigma" not in loop.tags
    assert _loop_value(loop, "CA", "Offset_ppm") == pytest.approx(1.5)
