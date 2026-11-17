"""Labelled residues outside the SIFTS UniProt span, residue level (release
validation 2026-11-17c, G43): :mod:`trizod.dataset.construct_span`."""

import gzip

from trizod.dataset import construct_span as cs


def test_align_map_places_a_tagged_construct_on_its_seqres():
    seqres = "MKTAYIAKQRQISFVKSHFSRQ"
    construct = "GSHM" + seqres[3:]  # a tag in place of the first 3 residues
    mp = cs.align_map(construct, seqres)
    assert list(mp[4:]) == list(range(3, len(seqres)))
    assert all(p < 3 for p in mp[:4])  # the tag aligns (or not) before the span


def _xref(pdb, chain, acc):
    return {"source": "sifts", "db": "UniProt", "accession": acc,
            "details": f"{pdb} chain {chain}"}  # fmt: skip


def test_outside_span_counts_labelled_residues_outside_the_segment(tmp_path):
    seqres = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    with gzip.open(tmp_path / cs.SEQRES, "wt") as fh:
        fh.write(f">1abc_A mol:protein length:{len(seqres)}  X\n{seqres}\n")
    with gzip.open(tmp_path / cs.SIFTS, "wt") as fh:
        fh.write("# header\nPDB\tCHAIN\tSP_PRIMARY\tRES_BEG\tRES_END\n")
        fh.write("1abc\tA\tP12345\t6\t33\n")  # the first five SEQRES residues are not
    seq = "GS" + seqres  # two more residues before the SEQRES start
    mask = [True] * len(seq)
    mask[0] = False  # one of the tag residues unlabelled
    rows = [
        {"id": "1_1_1_1", "split": "train", "sequence": seq, "mask": mask,
         "uniprot_source": "sifts", "uniprot_ids": ["P12345"],
         "xrefs": [_xref("1ABC", "A", "P12345")]},
        {"id": "2_1_1_1", "split": "train", "sequence": seq, "mask": mask,
         "uniprot_source": "author", "uniprot_ids": ["P12345"],
         "xrefs": [_xref("1ABC", "A", "P12345")]},  # not a SIFTS-level chain
        {"id": "3_1_1_1", "split": "excluded", "sequence": seq, "mask": mask,
         "uniprot_source": "sifts", "uniprot_ids": ["P12345"],
         "xrefs": [_xref("1ABC", "A", "P12345")]},  # not a training/test split
    ]  # fmt: skip
    got = cs.outside_span(rows, tmp_path)
    # labelled: 1 tag residue + 5 SEQRES residues before RES_BEG = 6 outside
    assert got == {"1_1_1_1": (6, len(seq) - 1)}
