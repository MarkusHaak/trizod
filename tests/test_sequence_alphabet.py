"""The sequence alphabet of a parsed entity (release audit D15, 2026-11-17).

A deposited ``_Entity.Polymer_seq_one_letter_code`` may carry characters that
are not residue letters: bmr50015 ends in the translation-stop symbol ``*``,
bmr52376 in an apostrophe (a typo). Neither is a residue, and both reached the
released sequence (and the UdonPred views) verbatim, where a
protein-language-model tokenizer may reject them. The parser now

* strips a TRAILING run of non-letters, before any per-residue array is built,
  so the sequence and every per-residue array lose that terminal position
  together (no shift of either entry references it: bmr50015's last shift is on
  residue 105 of 106, bmr52376's on residue 77 of 78); and
* maps an INTERNAL non-letter to ``X``, one for one, which keeps the residue
  numbering. The 2026-11-11 snapshot holds one, a ``-`` in bmr53341, an entry
  without assigned shifts.
"""

import string

import pynmrstar
import pytest

import trizod.bmrb.bmrb as bmrb
from tests.conftest import BMRB_DIR, requires_bmrb_data

_ENTITY_SF = """\
save_entity_1
   _Entity.Sf_category                 entity
   _Entity.Sf_framecode                entity_1
   _Entity.Entry_ID                    1
   _Entity.ID                          1
   _Entity.Name                        test
   _Entity.Type                        polymer
   _Entity.Polymer_type                polypeptide(L)
   _Entity.Polymer_seq_one_letter_code
;
{seq}
;
save_
"""


def _seq(seq):
    sf = pynmrstar.Saveframe.from_string(_ENTITY_SF.format(seq=seq))
    return bmrb.Entity(sf).seq


@pytest.mark.parametrize(
    ("deposited", "parsed"),
    [
        ("LKGELRSGC*", "LKGELRSGC"),  # bmr50015: a stop codon
        ("HLVLRLRGG'", "HLVLRLRGG"),  # bmr52376: an apostrophe
        ("MKAG*'", "MKAG"),  # a trailing run
        ("MKAG*\n", "MKAG"),  # line breaks are not residues
    ],
)
def test_a_trailing_non_letter_is_stripped(deposited, parsed):
    assert _seq(deposited) == parsed


@pytest.mark.parametrize(
    ("deposited", "parsed"),
    [
        ("RGNQHDVPVIPLDVVD-AAS", "RGNQHDVPVIPLDVVDXAAS"),  # bmr53341
        ("MA*GS", "MAXGS"),
        ("MA1GS", "MAXGS"),
        ("*MAGS", "XMAGS"),  # leading: numbering is kept
        ("MA-*GS*", "MAXXGS"),
    ],
)
def test_an_internal_non_letter_becomes_x_keeping_the_numbering(deposited, parsed):
    seq = _seq(deposited)
    assert seq == parsed
    assert seq.find("GS") == deposited.find("GS")


def test_letters_are_kept_upper_cased():
    # X, U (selenocysteine) and Z (Glx) are deposited letters and stay as they are
    assert _seq("mkXUZa") == "MKXUZA"


def test_line_breaks_and_blanks_are_not_residues():
    assert _seq("MKT AYI\nAKQR") == "MKTAYIAKQR"


def test_a_sequence_of_non_letters_only_is_none():
    assert _seq("*") is None


@requires_bmrb_data
@pytest.mark.parametrize(
    ("entry_id", "length", "tail", "last_shift_seq_id"),
    [("50015", 105, "LKGELRSGC", 105), ("52376", 77, "HLVLRLRGG", 77)],
)
def test_deposited_entries_lose_only_the_unshifted_terminal_position(
    entry_id, length, tail, last_shift_seq_id
):
    entry_dir = BMRB_DIR / f"bmr{entry_id}"
    if not entry_dir.exists():
        pytest.skip(f"BMRB {entry_id} not available in BMRB_DIR")
    entry = bmrb.BmrbEntry(entry_id, entry_dir)
    seq = entry.entities["1"].seq
    assert len(seq) == length and seq.endswith(tail)
    assert set(seq) <= set(string.ascii_uppercase)
    # no shift references the stripped position: the per-residue arrays are
    # built over the stripped sequence and keep every shift
    (key,) = [k for k in entry.get_peptide_shifts() if k[2] == "1"][:1]
    shifts = entry.get_peptide_shifts()[key][0]
    assert max(int(s[2]) for s in shifts) == last_shift_seq_id <= length
    arr, mask = bmrb.get_valid_bbshifts(shifts, seq)
    assert arr.shape[0] == len(seq) and mask.shape[0] == len(seq)
