"""Tests for trizod.io.fasta — unified FASTA helpers (restructure Phase 5).

Guards the parser behaviour the dataset-build/leakage math relies on: first-token
IDs, sequence concatenation, round-trip fidelity, and the order/dedup distinction
between fasta_ids (ordered, keeps dupes) and read_fasta (dict, last wins).
"""

from trizod.io.fasta import count_fasta, fasta_ids, read_fasta, write_fasta


def test_read_fasta_first_token_id_and_concatenated_seq(tmp_path):
    p = tmp_path / "in.fasta"
    p.write_text(">id1 some description here\nACDE\nFGHI\n>id2\nKLMN\n")
    assert read_fasta(p) == {"id1": "ACDEFGHI", "id2": "KLMN"}


def test_write_read_round_trip(tmp_path):
    recs = {"a": "ACDEFG", "b": "KLMNPQ"}
    p = tmp_path / "out.fasta"
    write_fasta(recs, p)
    assert read_fasta(p) == recs


def test_write_fasta_prefix(tmp_path):
    p = tmp_path / "pref.fasta"
    write_fasta({"1": "AAAA"}, p, prefix="bmr")
    assert read_fasta(p) == {"bmr1": "AAAA"}
    assert fasta_ids(p) == ["bmr1"]


def test_fasta_ids_preserves_order_and_duplicates(tmp_path):
    p = tmp_path / "dup.fasta"
    p.write_text(">z\nAA\n>a\nCC\n>z\nGG\n")
    # fasta_ids keeps file order + duplicate headers ...
    assert fasta_ids(p) == ["z", "a", "z"]
    # ... read_fasta collapses to a dict (last occurrence wins).
    assert read_fasta(p) == {"z": "GG", "a": "CC"}
    assert count_fasta(p) == 3
