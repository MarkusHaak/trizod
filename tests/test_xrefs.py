"""UniProt / PDB cross-references per chain, read from the NMR-STAR snapshot.

The BMRB records three kinds of external link, with very different reliability:

* ``_Related_entries`` rows whose relationship is the *BMRB Entry Tracking
  System*: the PDB entry deposited together with this BMRB entry (``exact``);
* ``_Entity_db_link`` / ``_Assembly_db_link`` rows with ``Author_supplied=yes``
  and author-typed ``_Related_entries`` rows (``author``);
* ``_Entity_db_link`` rows with ``Author_supplied=no``: BMRB annotator BLAST
  hits, which include homologs (``blast``), carrying ``Seq_identity`` and the
  query coverage ``Seq_query_to_submitted_percent``.

These tests pin that the three stay apart, that accessions are normalised only
when they really are accessions, and which accessions reach ``uniprot_ids``.
"""

import pynmrstar
import pytest

from trizod.bmrb import xrefs as xr

ENTRY = """\
data_99999

save_entry_information
   _Entry.Sf_category    entry_information
   _Entry.Sf_framecode   entry_information
   _Entry.ID             99999

   loop_
      _Related_entries.Database_name
      _Related_entries.Database_accession_code
      _Related_entries.Relationship
      _Related_entries.Entry_ID

      PDB   1abc   'BMRB Entry Tracking System'   99999
      PDB   2XYZ   'Crystal structure'            99999
      BMRB  12345  .                              99999
      PDB   .      .                              99999
   stop_
save_

save_assembly
   _Assembly.Sf_category    assembly
   _Assembly.Sf_framecode   assembly
   _Assembly.ID             1

   loop_
      _Assembly_db_link.Author_supplied
      _Assembly_db_link.Database_code
      _Assembly_db_link.Accession_code
      _Assembly_db_link.Entry_details
      _Assembly_db_link.Assembly_ID

      yes   PDB   3DEF_A   'complex with partner'   1
      .     PDB   1ABC     .                        1
   stop_
save_

save_entity_1
   _Entity.Sf_category    entity
   _Entity.Sf_framecode   entity_1
   _Entity.ID             1

   loop_
      _Entity_db_link.Ordinal
      _Entity_db_link.Author_supplied
      _Entity_db_link.Database_code
      _Entity_db_link.Accession_code
      _Entity_db_link.Seq_query_to_submitted_percent
      _Entity_db_link.Seq_identity
      _Entity_db_link.Entity_ID

      1   no    SP    Q00001        100.00   98.00    1
      2   no    SP    P01308        100.00   100.00   1
      3   yes   UNP   p04156-2      .        .        1
      4   no    PDB   4ABC          100.00   100.00   1
      5   no    GB    AAA12345      100.00   100.00   1
      6   yes   UNP   B2MG_HUMAN    .        .        1
      7   no    SP    P01308        100.00   100.00   1
   stop_
save_

save_entity_2
   _Entity.Sf_category    entity
   _Entity.Sf_framecode   entity_2
   _Entity.ID             2

   loop_
      _Entity_db_link.Ordinal
      _Entity_db_link.Author_supplied
      _Entity_db_link.Database_code
      _Entity_db_link.Accession_code
      _Entity_db_link.Seq_query_to_submitted_percent
      _Entity_db_link.Seq_identity
      _Entity_db_link.Entity_ID

      1   no   SP           P69905   95.00    100.00   2
      2   .    SWISS-PROT   P69906   100      100      2
      3   no   SP           P69907   100.00   100.00   2
   stop_
save_

save_entity_3
   _Entity.Sf_category    entity
   _Entity.Sf_framecode   entity_3
   _Entity.ID             3
save_
"""


@pytest.fixture(scope="module")
def parsed():
    return xr.parse_xrefs(pynmrstar.Entry.from_string(ENTRY))


def _key(x):
    return (x.db, x.accession, x.source, x.origin)


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "code, db",
    [
        ("UNP", xr.UNIPROT),
        ("UniProt", xr.UNIPROT),
        ("UniProtKB", xr.UNIPROT),
        ("SP", xr.UNIPROT),
        ("SWS", xr.UNIPROT),
        ("SWISS-PROT", xr.UNIPROT),
        ("swissprot", xr.UNIPROT),
        ("Swiss_Prot", xr.UNIPROT),
        ("TrEMBL", xr.UNIPROT),
        ("UniProtKB/Swiss-Prot", xr.UNIPROT),
        ("UNIPROT KB/SWISS PROT", xr.UNIPROT),
        ("PDB", xr.PDB),
        ("pdb", xr.PDB),
        ("RCSB PDB", xr.PDB),
        ("GB", None),
        ("REF", None),
        ("BMRB", None),
        ("PDB_DEV", None),  # integrative structures, not the PDB archive
        (".", None),
        ("", None),
        (None, None),
    ],
)
def test_database_codes_map_to_uniprot_or_pdb_only(code, db):
    assert xr.normalize_db(code) == db


@pytest.mark.parametrize(
    "raw, acc",
    [
        ("P04156", "P04156"),
        ("p04156", "P04156"),
        (" Q8BGM4-1 ", "Q8BGM4"),  # isoform -> base accession
        ("A0A024R161", "A0A024R161"),  # 10-character accession
        ("P01308.2", "P01308"),  # sequence version
        ("B2MG_HUMAN", None),  # an entry NAME, not an accession
        ("NP_000043.3", None),  # RefSeq typed into a UniProt field
        ("301555", None),
        (".", None),
        ("", None),
        (None, None),
    ],
)
def test_uniprot_accessions_are_validated(raw, acc):
    assert xr.normalize_uniprot(raw) == acc


@pytest.mark.parametrize(
    "raw, acc",
    [
        ("1abc", "1ABC"),
        ("2FIF_E", "2FIF"),  # chain suffix dropped
        ("2fif:E", "2FIF"),
        ("pdb_00001abc", "1ABC"),  # extended PDB ID
        ("CKK1", None),
        ("rcsb101202", None),
        ("19291", None),  # a BMRB ID in a PDB field
        (".", None),
        (None, None),
    ],
)
def test_pdb_ids_are_validated(raw, acc):
    assert xr.normalize_pdb(raw) == acc


# --------------------------------------------------------------------------- #
# parsing + provenance
# --------------------------------------------------------------------------- #


def test_tracking_system_pdb_is_exact_and_other_related_entries_are_author(parsed):
    entry = {_key(x) for x in parsed.entry}
    assert (xr.PDB, "1ABC", "exact", "Related_entries") in entry
    assert (xr.PDB, "2XYZ", "author", "Related_entries") in entry
    # BMRB-to-BMRB links and empty accessions are not cross-references
    assert all(x.db in (xr.UNIPROT, xr.PDB) for x in parsed.entry)
    assert all(x.raw_accession not in ("", ".") for x in parsed.entry)


def test_assembly_links_keep_their_author_flag(parsed):
    entry = {_key(x) for x in parsed.entry}
    assert (xr.PDB, "3DEF", "author", "Assembly_db_link") in entry
    assert (xr.PDB, "1ABC", "unspecified", "Assembly_db_link") in entry
    (complex_,) = [x for x in parsed.entry if x.accession == "3DEF"]
    assert complex_.raw_accession == "3DEF_A"
    assert complex_.details == "complex with partner"


def test_entity_links_split_author_and_blast(parsed):
    links = parsed.entity["1"]
    by_acc = {(x.raw_accession, x.source) for x in links}
    assert ("p04156-2", "author") in by_acc
    assert ("P01308", "blast") in by_acc
    assert ("4ABC", "blast") in by_acc
    # GenBank is neither UniProt nor PDB
    assert all(x.raw_accession != "AAA12345" for x in links)


def test_blast_hits_carry_identity_and_coverage(parsed):
    (q,) = [x for x in parsed.entity["1"] if x.accession == "Q00001"]
    assert (q.identity, q.coverage) == (98.0, 100.0)
    (author,) = [x for x in parsed.entity["1"] if x.accession == "P04156"]
    assert (author.identity, author.coverage) == (None, None)


def test_entry_names_are_kept_but_have_no_accession(parsed):
    (name,) = [x for x in parsed.entity["1"] if x.raw_accession == "B2MG_HUMAN"]
    assert name.accession is None
    assert name.db == xr.UNIPROT and name.source == "author"


def test_missing_author_flag_is_unspecified_not_guessed(parsed):
    (x,) = [x for x in parsed.entity["2"] if x.accession == "P69906"]
    assert x.source == "unspecified"
    assert x.raw_db == "SWISS-PROT"


def test_entity_without_links_has_an_empty_list(parsed):
    assert parsed.entity["3"] == []


# --------------------------------------------------------------------------- #
# per chain: entity links + the entry's links, ranked and de-duplicated
# --------------------------------------------------------------------------- #


def test_chain_links_merge_entity_and_entry_links(parsed):
    chain = parsed.for_entity("1")
    keys = [_key(x) for x in chain]
    assert (xr.PDB, "1ABC", "exact", "Related_entries") in keys
    assert (xr.UNIPROT, "P04156", "author", "Entity_db_link") in keys
    # an entity the entry does not declare still gets the entry-level links
    assert {_key(x) for x in parsed.for_entity("42")} == {_key(x) for x in parsed.entry}


def test_chain_links_are_deduplicated(parsed):
    chain = parsed.for_entity("1")
    blast = [x for x in chain if x.accession == "P01308" and x.source == "blast"]
    assert len(blast) == 1


def test_chain_links_are_ordered_uniprot_first_then_by_confidence(parsed):
    chain = parsed.for_entity("1")
    dbs = [x.db for x in chain]
    assert dbs == sorted(dbs, key=[xr.UNIPROT, xr.PDB].index)
    pdb_sources = [x.source for x in chain if x.db == xr.PDB]
    rank = list(xr.SOURCES)
    assert pdb_sources == sorted(pdb_sources, key=rank.index)
    # within one source the deposited order (BMRB Ordinal) is kept
    blast = [x.accession for x in chain if x.db == xr.UNIPROT and x.source == "blast"]
    assert blast == ["Q00001", "P01308"]


# --------------------------------------------------------------------------- #
# convenience: uniprot_ids and the high-confidence PDB IDs
# --------------------------------------------------------------------------- #


def test_uniprot_ids_are_the_author_accessions_when_there_are_any(parsed):
    # entity 1 also has P01308 as a 100 %/100 % BLAST hit, and an author entry
    # NAME (B2MG_HUMAN, no accession): neither joins the author tier
    assert xr.uniprot_ids(parsed.for_entity("1")) == (["P04156"], "author")


def test_every_author_accession_is_listed_in_deposited_order():
    links = [
        xr.Xref(xr.UNIPROT, acc, "author", None, None, "Entity_db_link", "UNP", acc)
        for acc in ["Q9NZQ7", "P07463", "Q9NZQ7"]
    ]
    assert xr.uniprot_ids(links) == (["Q9NZQ7", "P07463"], "author")


def test_without_an_author_link_every_full_identity_full_coverage_hit_is_listed(
    parsed,
):
    """The same sequence in several species (P68246 / P68247 in the release)
    is not ranked: every accession at 100 % identity over 100 % of the chain is
    listed, so a join on any of them hits. P69905 is 100 % identical but covers
    only 95 % of the chain, so it is a fragment match and left out."""
    assert xr.uniprot_ids(parsed.for_entity("2")) == (
        ["P69906", "P69907"],
        "unspecified",
    )
    only_blast = [x for x in parsed.for_entity("2") if x.source == "blast"]
    assert xr.uniprot_ids(only_blast) == (["P69907"], "blast")
    # P01308 is deposited twice but is one accession
    no_author = [x for x in parsed.for_entity("1") if x.source != "author"]
    assert xr.uniprot_ids(no_author) == (["P01308"], "blast")


def test_uniprot_ids_are_empty_without_a_qualifying_link(parsed):
    assert xr.uniprot_ids(parsed.for_entity("3")) == ([], None)
    homolog = [x for x in parsed.for_entity("1") if x.accession == "Q00001"]
    assert xr.uniprot_ids(homolog) == ([], None)


def test_confident_pdb_ids_are_exact_and_author_only(parsed):
    assert xr.confident_pdb_ids(parsed.for_entity("1")) == ["1ABC", "2XYZ", "3DEF"]


def test_records_are_plain_dicts_with_the_declared_fields(parsed):
    recs = xr.xref_records(parsed.for_entity("1"))
    assert all(list(r) == [name for name, _ in xr.XREF_FIELDS] for r in recs)


# --------------------------------------------------------------------------- #
# reading the snapshot
# --------------------------------------------------------------------------- #


def _write_entry(root, entry_id, text):
    d = root / f"bmr{entry_id}"
    d.mkdir(parents=True)
    (d / f"bmr{entry_id}_3.str").write_text(text)


def test_load_xrefs_reads_each_entry_once_and_skips_missing_files(tmp_path):
    _write_entry(tmp_path, "99999", ENTRY)
    got = xr.load_xrefs(tmp_path, ["99999", "99999", "11111"])
    assert set(got) == {"99999"}
    assert got["99999"].for_entity("1")[0].db == xr.UNIPROT
