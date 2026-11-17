"""Tests for the re-referenced .str emitter."""

import numpy as np
import pynmrstar
import pytest

from trizod.constants import BACKBONE_ATOMS, REFINED_WEIGHTS
from trizod.io.str_writer import write_rereferenced_str
from trizod.offsets import total_offset_ppm


def _write_minimal(out_path, lacs_offsets, potenci_offsets):
    """Emit a one-residue file carrying the given offsets, return the parsed entry."""
    write_rereferenced_str(
        out_path,
        entry_id="00001",
        seq="A",
        bbshifts_arr=np.zeros((1, len(BACKBONE_ATOMS))),
        bbshifts_mask=np.zeros((1, len(BACKBONE_ATOMS)), dtype=bool),
        lacs_offsets_ppm=lacs_offsets,
        potenci_residual_offsets_sigma=potenci_offsets,
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
        lacs_offsets_ppm=lacs_offsets,
        potenci_residual_offsets_sigma=potenci_offsets,
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
        lacs_offsets_ppm=lacs_offsets,
        potenci_residual_offsets_sigma=potenci_offsets,
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


def test_total_offsets_loop_is_the_ready_to_use_ppm_quantity(tmp_path):
    """The Total_offsets loop is what a user subtracts from a deposited shift.

    It is the only loop in the file whose value needs no further arithmetic:
    LACS ppm plus the POTENCI sigma offset converted with the atom's weight.
    """
    lacs_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    lacs_offsets["CA"] = 0.45
    lacs_offsets["HA"] = -0.03
    potenci_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    potenci_offsets["CA"] = 13.5917784137
    potenci_offsets["HA"] = -9.4282312504
    parsed = _write_minimal(
        tmp_path / "bmr00001_rereferenced.str", lacs_offsets, potenci_offsets
    )
    loop = parsed.get_loops_by_category("Total_offsets")[0]
    assert "Offset_ppm" in loop.tags
    assert "Offset_sigma" not in loop.tags
    assert _loop_value(loop, "CA", "Offset_ppm") == pytest.approx(
        3.14389048159534, abs=1e-6
    )
    assert _loop_value(loop, "HA", "Offset_ppm") == pytest.approx(
        -0.278056764198024, abs=1e-6
    )


def test_total_offsets_use_the_shared_helper(tmp_path):
    """Every atom's total matches ``offsets.total_offset_ppm`` exactly."""
    lacs_offsets = {a: 0.1 * (i + 1) for i, a in enumerate(BACKBONE_ATOMS)}
    potenci_offsets = {a: 0.5 * (i - 3) for i, a in enumerate(BACKBONE_ATOMS)}
    parsed = _write_minimal(
        tmp_path / "bmr00001_rereferenced.str", lacs_offsets, potenci_offsets
    )
    loop = parsed.get_loops_by_category("Total_offsets")[0]
    for atom in BACKBONE_ATOMS:
        expected = total_offset_ppm(atom, lacs_offsets[atom], potenci_offsets[atom])
        assert _loop_value(loop, atom, "Offset_ppm") == pytest.approx(
            expected, abs=1e-6
        )


def test_every_emitted_offset_tag_names_its_unit(tmp_path):
    """No offset tag in the .str file may be unit-free."""
    lacs_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    potenci_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    parsed = _write_minimal(
        tmp_path / "bmr00001_rereferenced.str", lacs_offsets, potenci_offsets
    )
    for category in ("LACS_offsets", "POTENCI_residual_offsets", "Total_offsets"):
        loop = parsed.get_loops_by_category(category)[0]
        offset_tags = [t for t in loop.tags if t.lower().startswith("offset")]
        assert offset_tags, f"{category} carries no offset tag"
        for tag in offset_tags:
            assert tag.endswith("_ppm") or tag.endswith("_sigma"), (
                f"{category}.{tag} does not name its unit"
            )


def _raw_loop_value(loop, atom, tag):
    """Value of `tag` for `atom` as the literal string in the file."""
    atom_col = loop.tag_index("Atom_ID")
    val_col = loop.tag_index(tag)
    for row in loop.data:
        if row[atom_col] == atom:
            return row[val_col]
    raise AssertionError(f"atom {atom} not found in loop")


def test_a_rejected_offset_is_null_not_a_measured_zero(tmp_path):
    """`--max-offset` NaNs the offset it rejects, and with
    `--reject-shift-type-only` the chain is still published. Writing 0.000000
    there would assert "measured, and perfectly referenced" for exactly the atom
    TriZOD threw out, which is indistinguishable from a real zero offset."""
    lacs_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    potenci_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    potenci_offsets["CB"] = None  # rejected by --max-offset
    lacs_offsets["N"] = None  # never determined
    parsed = _write_minimal(
        tmp_path / "bmr00001_rereferenced.str", lacs_offsets, potenci_offsets
    )
    potenci = parsed.get_loops_by_category("POTENCI_residual_offsets")[0]
    assert _raw_loop_value(potenci, "CB", "Offset_sigma") == "."
    assert _raw_loop_value(potenci, "CB", "Offset_ppm") == "."
    assert (
        _raw_loop_value(
            parsed.get_loops_by_category("LACS_offsets")[0], "N", "Offset_ppm"
        )
        == "."
    )
    # a null in EITHER term makes the total unknown, not smaller
    total = parsed.get_loops_by_category("Total_offsets")[0]
    assert _raw_loop_value(total, "CB", "Offset_ppm") == "."
    assert _raw_loop_value(total, "N", "Offset_ppm") == "."
    # ... while a genuine zero still reports as a measured zero
    assert _loop_value(total, "CA", "Offset_ppm") == pytest.approx(0.0)
