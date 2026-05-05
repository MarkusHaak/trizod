"""Step 8 — Leu CD1/CD2 and Val CG1/CG2 are rewritten to CDx/CGx for
non-stereospecific (geminal-partner) ambiguity codes."""

import pytest

from tests.conftest import BMRB_DIR, requires_bmrb_data
from trizod.bmrb.bmrb import _maybe_wildcard_methyl


@pytest.mark.parametrize(
    "comp_id,atom_id,ambiguity,expected",
    [
        # Leu CD1/CD2: stereospecific (code "1") preserved
        ("LEU", "CD1", "1", "CD1"),
        ("LEU", "CD2", "1", "CD2"),
        # Leu CD1/CD2: geminal-partner (code "2") wildcarded
        ("LEU", "CD1", "2", "CDx"),
        ("LEU", "CD2", "2", "CDx"),
        # Leu CD1/CD2: missing code → wildcarded conservatively
        ("LEU", "CD1", "", "CDx"),
        ("LEU", "CD2", ".", "CDx"),
        # Val CG1/CG2: stereospecific preserved
        ("VAL", "CG1", "1", "CG1"),
        ("VAL", "CG2", "1", "CG2"),
        # Val CG1/CG2: geminal-partner wildcarded
        ("VAL", "CG1", "2", "CGx"),
        ("VAL", "CG2", "2", "CGx"),
        # Val CG1/CG2: missing code wildcarded
        ("VAL", "CG1", "", "CGx"),
        ("VAL", "CG2", ".", "CGx"),
        # Other residues / atoms: passthrough
        ("ALA", "CB", "1", "CB"),
        ("LEU", "CA", "2", "CA"),
        ("ILE", "CD1", "2", "CD1"),  # Ile, not Leu — unaffected
    ],
)
def test_maybe_wildcard_methyl(comp_id, atom_id, ambiguity, expected):
    assert _maybe_wildcard_methyl(comp_id, atom_id, ambiguity) == expected


@requires_bmrb_data
def test_leu_cdx_appears_in_real_entry():
    """A real entry with non-stereospecifically assigned Leu CD or Val CG
    methyl carbons (BMRB 15000) emits CDx / CGx wildcards after Step 8 rewrite.

    Note: BMRB 17665 (alpha-synuclein) was originally proposed for this test,
    but it only contains Leu/Val proton shifts (HD*, HG*) — no CD1/CD2 or
    CG1/CG2 carbons — so no wildcard rewrite would ever fire. BMRB 15000
    contains LEU CD2 and VAL CG2 carbon shifts with ambiguity code "."
    (unspecified, treated as non-stereospecific), making it a valid fixture.
    """
    import trizod.bmrb.bmrb as bmrb

    entry_dir = BMRB_DIR / "bmr15000"
    if not entry_dir.exists():
        pytest.skip("BMRB 15000 not in BMRB_DIR")

    # BmrbEntry expects bmrb_dir to be the parent directory containing bmr15000/
    entry = bmrb.BmrbEntry("15000", BMRB_DIR)
    found = set()
    for shift_table in entry.shift_tables.values():
        # ShiftTable.shifts is dict {(assembly_id, entity_id): [tuple, ...]}
        for rows in shift_table.shifts.values():
            for row in rows:
                atom_id = row[4]  # _Atom_chem_shift.Atom_ID column
                if atom_id in ("CDx", "CGx"):
                    found.add(atom_id)
    assert found, "expected at least one CDx or CGx wildcard in BMRB 15000 after Step 8"
