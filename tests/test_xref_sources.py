"""Dated external sources for the cross-references: BMRB's BMRB<->PDB match list,
SIFTS (PDB chain -> UniProt) and the wwPDB SEQRES sequences.

The NMR-STAR snapshot does not carry every same-deposition PDB entry, and never
the UniProt accession of the PDB chain. These three public flat files fill both,
per chain: a PDB chain is used only if its sequence matches the TriZOD chain,
so the partner of a complex never lends its UniProt accession. Every file is
pinned by a SHA-256 manifest, so a build names exactly what it read.
"""

import gzip
import hashlib
import json

import pytest

from trizod.bmrb import xref_sources as xs
from trizod.bmrb import xrefs as xr

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
PARTNER = "GSHMDEEELLRKAIELSLKESRNSA"  # a different protein in the same structure


def _gz(path, text):
    with gzip.open(path, "wt") as fh:
        fh.write(text)


@pytest.fixture
def sources(tmp_path):
    d = tmp_path / "xref_sources"
    d.mkdir()
    (d / xs.BMRB_PDB_MATCH).write_text(
        "16000,1D3Z\n"  # > 15000: same ADIT-NMR session
        "16000,2ABC\n"
        "4020,1brv\n"  # older: author- or annotator-supplied
        "17000,9ZZZ\n"  # a structure with no SEQRES/SIFTS rows
    )
    _gz(
        d / xs.SEQRES,
        f">1d3z_A mol:protein length:76  UBIQUITIN\n{UBQ}\n"
        f">1d3z_B mol:protein length:25  PARTNER\n{PARTNER}\n"
        ">1d3z_C mol:na length:10  DNA\nACGTACGTAC\n"
        f">2abc_A mol:protein length:80  UBIQUITIN, TAGGED\nGSHM{UBQ}\n"
        f">5xyz_A mol:protein length:76  UBIQUITIN\n{UBQ}\n"
        f">1brv_A mol:protein length:76  UBIQUITIN\n{UBQ}\n",
    )
    _gz(
        d / xs.SIFTS,
        "# 2026/11/08 - 12:37 | PDB: 45.26 | UniProt: 2026.04\n"
        "PDB\tCHAIN\tSP_PRIMARY\tRES_BEG\tRES_END\tPDB_BEG\tPDB_END\tSP_BEG\tSP_END\n"
        "1d3z\tA\tP0CG48\t1\t76\t1\t76\t609\t684\n"
        "1d3z\tB\tQ99999\t1\t25\t1\t25\t1\t25\n"
        "2abc\tA\tP0CG48\t5\t80\t1\t76\t609\t684\n"
        "5xyz\tA\tP62988\t1\t76\t1\t76\t1\t76\n"
        "1brv\tA\tP0CG47\t1\t76\t1\t76\t1\t76\n",
    )
    files = {}
    for name in (xs.BMRB_PDB_MATCH, xs.SEQRES, xs.SIFTS):
        data = (d / name).read_bytes()
        files[name] = {
            "url": f"https://example.org/{name}",
            "retrieved": "2026-11-13T12:00:00Z",
            "last_modified": "Wed, 11 Nov 2026 20:05:04 GMT",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    (d / xs.MANIFEST).write_text(json.dumps({"files": files}))
    return d


def _snapshot_exact(pdb_id):
    return xr.Xref(
        xr.PDB, pdb_id, "exact", None, None, "Related_entries", "PDB", pdb_id
    )


# --------------------------------------------------------------------------- #
# sequence matching
# --------------------------------------------------------------------------- #


def test_identical_sequences_match_fully():
    assert xs.sequence_match(UBQ, UBQ) == (100.0, 100.0)


def test_a_tag_on_the_structure_does_not_lower_the_match():
    assert xs.sequence_match(UBQ, "GSHM" + UBQ) == (100.0, 100.0)


def test_a_point_mutation_lowers_identity_not_coverage_much():
    mutant = UBQ[:30] + ("A" if UBQ[30] != "A" else "G") + UBQ[31:]
    coverage, identity = xs.sequence_match(UBQ, mutant)
    assert 98 < coverage < 100 and 98 < identity < 100


def test_an_unrelated_sequence_does_not_match():
    coverage, identity = xs.sequence_match(UBQ, PARTNER)
    assert coverage < xs.MIN_COVERAGE


def test_empty_sequences_do_not_match():
    assert xs.sequence_match("", UBQ) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# reading the files
# --------------------------------------------------------------------------- #


def test_bmrb_pdb_match_is_read_upper_case(sources):
    got = xs.read_bmrb_pdb_match(sources / xs.BMRB_PDB_MATCH)
    assert got["16000"] == {"1D3Z", "2ABC"}
    assert got["4020"] == {"1BRV"}


def test_seqres_keeps_protein_chains_of_the_wanted_structures_only(sources):
    got = xs.read_seqres(sources / xs.SEQRES, {"1D3Z"})
    assert set(got) == {"1D3Z"}
    assert got["1D3Z"] == {"A": UBQ, "B": PARTNER}


def test_sifts_is_read_per_chain_for_the_wanted_structures_only(sources):
    got = xs.read_sifts(sources / xs.SIFTS, {"1D3Z", "2ABC"})
    assert got == {
        "1D3Z": {"A": ["P0CG48"], "B": ["Q99999"]},
        "2ABC": {"A": ["P0CG48"]},
    }


def test_load_refuses_a_file_that_does_not_match_its_manifest(sources):
    (sources / xs.BMRB_PDB_MATCH).write_text("16000,1D3Z\n")
    with pytest.raises(ValueError, match="sha256"):
        xs.ExternalSources.load(sources, ["16000"])


def test_load_refuses_a_directory_without_a_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        xs.ExternalSources.load(tmp_path, ["16000"])


# --------------------------------------------------------------------------- #
# per-chain links
# --------------------------------------------------------------------------- #


@pytest.fixture
def ext(sources):
    # 5XYZ: an exact PDB entry named only by an entry's NMR-STAR snapshot
    return xs.ExternalSources.load(
        sources, ["16000", "4020", "17000"], extra_pdb_ids=["5XYZ"]
    )


def _keys(links):
    return {(x.db, x.accession, x.source, x.origin) for x in links}


def test_matched_pdb_entries_are_exact_after_15000_and_unspecified_before(ext):
    new = _keys(ext.links("16000", UBQ, []))
    assert (xr.PDB, "1D3Z", "exact", xs.ORIGIN_MATCH) in new
    assert (xr.PDB, "2ABC", "exact", xs.ORIGIN_MATCH) in new
    old = _keys(ext.links("4020", UBQ, []))
    assert (xr.PDB, "1BRV", "unspecified", xs.ORIGIN_MATCH) in old


def test_uniprot_comes_from_the_sequence_matched_chain_only(ext):
    links = ext.links("16000", UBQ, [])
    uni = [x for x in links if x.db == xr.UNIPROT]
    # 1D3Z chain A and 2ABC chain A match; the partner chain B (Q99999) does not
    assert {x.accession for x in uni} == {"P0CG48"}
    assert all(x.source == "sifts" and x.origin == xs.ORIGIN_SIFTS for x in uni)
    best = uni[0]
    assert (best.coverage, best.identity) == (100.0, 100.0)
    assert best.details.startswith("1D3Z chain A")


def test_the_partner_chain_gets_the_partner_accession(ext):
    uni = [x for x in ext.links("16000", PARTNER, []) if x.db == xr.UNIPROT]
    assert [x.accession for x in uni] == ["Q99999"]


def test_snapshot_exact_pdb_entries_are_used_for_sifts_too(ext):
    # entry 18000 is not in the BMRB match file; its snapshot names 5XYZ
    links = ext.links("18000", UBQ, [_snapshot_exact("5XYZ")])
    assert [x.accession for x in links if x.db == xr.UNIPROT] == ["P62988"]
    # the snapshot's own link is not repeated by the external source
    assert not [x for x in links if x.db == xr.PDB]


def test_blast_pdb_hits_are_not_used_for_sifts(ext):
    blast = xr.Xref(
        xr.PDB, "5XYZ", "blast", 100.0, 100.0, "Entity_db_link", "PDB", "5XYZ"
    )
    assert ext.links("18000", UBQ, [blast]) == []


def test_a_structure_without_sequence_or_sifts_rows_adds_only_the_pdb_link(ext):
    links = ext.links("17000", UBQ, [])
    assert _keys(links) == {(xr.PDB, "9ZZZ", "exact", xs.ORIGIN_MATCH)}


def test_manifest_summary_names_every_file(ext):
    summary = ext.provenance()
    assert set(summary["files"]) == {xs.BMRB_PDB_MATCH, xs.SEQRES, xs.SIFTS}
    assert all("sha256" in f and "url" in f for f in summary["files"].values())
    assert summary["min_coverage"] == xs.MIN_COVERAGE


# --------------------------------------------------------------------------- #
# ranking with the snapshot links
# --------------------------------------------------------------------------- #


def test_sifts_ranks_after_author_and_before_blast():
    assert xr.SOURCES.index("author") < xr.SOURCES.index("sifts")
    assert xr.SOURCES.index("sifts") < xr.SOURCES.index("unspecified")


def test_uniprot_ids_prefer_author_then_sifts_then_full_blast_hits(ext):
    blast = xr.Xref(
        xr.UNIPROT, "P0CG47", "blast", 100.0, 100.0, "Entity_db_link", "SP", "P0CG47"
    )
    author = xr.Xref(
        xr.UNIPROT, "P0CG48", "author", None, None, "Entity_db_link", "UNP", "P0CG48"
    )
    sifts = ext.links("16000", UBQ, [])
    assert xr.uniprot_ids(xr.rank_links([blast, *sifts])) == (["P0CG48"], "sifts")
    assert xr.uniprot_ids(xr.rank_links([blast, author, *sifts])) == (
        ["P0CG48"],
        "author",
    )
    assert xr.uniprot_ids(xr.rank_links([blast])) == (["P0CG47"], "blast")
