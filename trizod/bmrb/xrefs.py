"""UniProt and PDB cross-references per chain, read from BMRB NMR-STAR files.

An NMR-STAR v3 entry links out to other databases in three loops, and the three
differ sharply in how much a link can be trusted. This module reads all three,
keeps only the UniProt and PDB links, and labels each with its provenance
(``source``) so a consumer can choose its own trade-off:

``exact``
    ``_Related_entries`` row whose relationship is the *BMRB Entry Tracking
    System*: the PDB entry deposited together with this BMRB entry (one
    ADIT-NMR deposition). Entry-level.
``author``
    Supplied by the depositor: ``_Entity_db_link`` / ``_Assembly_db_link`` with
    ``Author_supplied = yes``, and every other ``_Related_entries`` PDB row
    (free-text relationships such as "Crystal structure"; ``details`` keeps the
    text, because a related structure may be another construct or a complex).
``blast``
    ``_Entity_db_link`` with ``Author_supplied = no``: BMRB annotators' BLAST
    hits. These include homologs -- BMRB 1000 (insulin) maps to five insulin
    accessions -- so each carries ``identity`` (``Seq_identity``, %) and
    ``coverage`` (``Seq_query_to_submitted_percent``, the share of the BMRB
    chain covered by the alignment, %).
``unspecified``
    A db-link row whose ``Author_supplied`` tag is empty (legacy entries). Not
    guessed either way; many of them carry BLAST-style identity values.

Entity-level links (``_Entity_db_link``) belong to one chain; entry-level ones
(``_Related_entries``, ``_Assembly_db_link``) are shared by every chain of the
entry, so a PDB structure of a complex is linked from each partner.

Read from the snapshot's own files rather than the BMRB API, so the links are
reproducible from the same snapshot that produced the labels. This module is
deliberately separate from ``trizod/bmrb/bmrb.py``: that file's hash keys the
entry-pickle and wSCS caches (``provenance.bmrb_parser_version``), and links
that no score depends on should not invalidate them.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import astuple, dataclass, field
from pathlib import Path

import pynmrstar

log = logging.getLogger("trizod.bmrb.xrefs")

UNIPROT = "UniProt"
PDB = "PDB"
#: Databases a link is kept for, in output order.
DATABASES = (UNIPROT, PDB)
#: Provenance values, most to least trustworthy; also the per-chain sort order.
SOURCES = ("exact", "author", "unspecified", "blast")

#: ``Database_code`` spellings, upper-cased with spaces, ``-`` and ``_``
#: removed. Built from every code in the 2026-11-11 snapshot (18,058 entries).
#: ``PDB_DEV`` (integrative structures) is deliberately not the PDB.
_UNIPROT_CODES = frozenset(
    {
        "UNP",
        "UNIPROT",
        "UNIPROTKB",
        "UNIPROTDB",
        "SP",
        "SWS",
        "SWP",
        "SWISSPROT",
        "TR",
        "TREMBL",
        "SPTR",
        "UNIPROTKB/SWISSPROT",
        "UNIPROTKB/TREMBL",
        "UNIPROT/SWISSPROT",
        "UNIPROT/TREMBL",
    }
)
_PDB_CODES = frozenset({"PDB", "RCSB", "RCSBPDB", "PDBE", "PDBJ", "WWPDB"})

#: UniProt's published accession pattern, optionally followed by an isoform
#: (``-2``) or a sequence version (``.2``), both of which are dropped.
_UNIPROT_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
    r"(?:-\d+)?(?:\.\d+)?$"
)
#: Four-character PDB ID, optionally with a chain suffix (``2FIF_E``), or the
#: extended ``pdb_0000xxxx`` form.
_PDB_RE = re.compile(r"^(?:PDB_0000)?([0-9][A-Z0-9]{3})(?:[_:.\-][A-Z0-9]+)?$")
#: ``_Related_entries.Relationship`` of the PDB entry from the same deposition.
_TRACKING_RE = re.compile(r"entry[\s_-]*tracking", re.IGNORECASE)

#: Field order of one exported link, with an Arrow type token (the release
#: Parquet builds its struct type from this, so it stays pyarrow-free here).
XREF_FIELDS = [
    ("db", "string"),
    ("accession", "string"),
    ("source", "string"),
    ("identity", "float32"),
    ("coverage", "float32"),
    ("loop", "string"),
    ("raw_db", "string"),
    ("raw_accession", "string"),
    ("details", "string"),
]


@dataclass(frozen=True)
class Xref:
    """One UniProt or PDB link.

    ``accession`` is the normalised accession (UniProt base accession without
    isoform/version; upper-case four-character PDB ID), or None when the
    deposited value is not one -- e.g. a UniProt entry *name* such as
    ``B2MG_HUMAN``. ``raw_db`` / ``raw_accession`` are verbatim.
    """

    db: str
    accession: str | None
    source: str
    identity: float | None
    coverage: float | None
    loop: str
    raw_db: str
    raw_accession: str
    details: str | None = None

    def as_record(self) -> dict:
        return dict(zip([name for name, _ in XREF_FIELDS], astuple(self)))


assert [f for f, _ in XREF_FIELDS] == list(Xref.__dataclass_fields__)


def _clean(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return None if s in ("", ".", "?") else s


def _float(value) -> float | None:
    s = _clean(value)
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def normalize_db(code) -> str | None:
    """Map a deposited database code to ``UNIPROT``, ``PDB`` or None."""
    s = _clean(code)
    if s is None:
        return None
    key = re.sub(r"[\s_\-]", "", s.upper())
    if key in _UNIPROT_CODES:
        return UNIPROT
    if key in _PDB_CODES:
        return PDB
    return None


def normalize_uniprot(raw) -> str | None:
    s = _clean(raw)
    if s is None:
        return None
    m = _UNIPROT_RE.match(s.upper())
    return m.group(1) if m else None


def normalize_pdb(raw) -> str | None:
    s = _clean(raw)
    if s is None:
        return None
    m = _PDB_RE.match(s.upper())
    return m.group(1) if m else None


def _make(
    db, raw_db, raw_acc, source, loop, identity=None, coverage=None, details=None
):
    norm = normalize_uniprot if db == UNIPROT else normalize_pdb
    return Xref(
        db=db,
        accession=norm(raw_acc),
        source=source,
        identity=identity,
        coverage=coverage,
        loop=loop,
        raw_db=raw_db,
        raw_accession=raw_acc,
        details=details,
    )


def _author_source(flag) -> str:
    s = (_clean(flag) or "").lower()
    if s == "yes":
        return "author"
    if s == "no":
        return "blast"
    return "unspecified"


def _rows(loop: pynmrstar.Loop) -> Iterable[dict]:
    tags = [t.lower() for t in loop.tags]
    for row in loop.data:
        yield dict(zip(tags, row))


def _db_link(row: dict, loop_name: str, details_tag: str | None) -> Xref | None:
    db = normalize_db(row.get("database_code"))
    raw_acc = _clean(row.get("accession_code"))
    if db is None or raw_acc is None:
        return None
    return _make(
        db,
        _clean(row.get("database_code")),
        raw_acc,
        _author_source(row.get("author_supplied")),
        loop_name,
        identity=_float(row.get("seq_identity")),
        coverage=_float(row.get("seq_query_to_submitted_percent")),
        details=_clean(row.get(details_tag)) if details_tag else None,
    )


def _related_entry(row: dict) -> Xref | None:
    db = normalize_db(row.get("database_name"))
    raw_acc = _clean(row.get("database_accession_code"))
    if db is None or raw_acc is None:
        return None
    relationship = _clean(row.get("relationship"))
    exact = db == PDB and relationship is not None and _TRACKING_RE.search(relationship)
    return _make(
        db,
        _clean(row.get("database_name")),
        raw_acc,
        "exact" if exact else "author",
        "Related_entries",
        details=relationship,
    )


def _sort_key(x: Xref) -> tuple[int, int]:
    return DATABASES.index(x.db), SOURCES.index(x.source)


def _dedup(links: Iterable[Xref]) -> list[Xref]:
    seen = set()
    out = []
    for x in links:
        key = (x.db, x.accession or x.raw_accession, x.source, x.loop)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


@dataclass
class EntryXrefs:
    """The links of one entry: per entity ID, and entry-level."""

    entity: dict[str, list[Xref]] = field(default_factory=dict)
    entry: list[Xref] = field(default_factory=list)

    def for_entity(self, entity_id) -> list[Xref]:
        """The chain's links: its entity's own plus the entry-level ones,
        de-duplicated, ordered UniProt before PDB, then by ``SOURCES``, then in
        deposited order (for BLAST hits, the BMRB's ``Ordinal``)."""
        links = self.entity.get(str(entity_id), []) + self.entry
        return sorted(_dedup(links), key=_sort_key)


def parse_xrefs(entry: pynmrstar.Entry) -> EntryXrefs:
    out = EntryXrefs()
    for sf in entry.frame_list:
        category = sf.category
        if category == "entity":
            ids = sf.get_tag("_Entity.ID")
            entity_id = str(ids[0]) if ids else None
            links = out.entity.setdefault(entity_id, [])
        for loop in sf.loops:
            if loop.category == "_Entity_db_link" and category == "entity":
                links.extend(
                    x for r in _rows(loop) if (x := _db_link(r, "Entity_db_link", None))
                )
            elif loop.category == "_Assembly_db_link":
                out.entry.extend(
                    x
                    for r in _rows(loop)
                    if (x := _db_link(r, "Assembly_db_link", "entry_details"))
                )
            elif loop.category == "_Related_entries":
                out.entry.extend(x for r in _rows(loop) if (x := _related_entry(r)))
    return out


def entry_file(bmrb_dir: Path, entry_id: str) -> Path:
    """``bmr<ID>/bmr<ID>_3.str`` under ``bmrb_dir``, or flat, like ``BmrbEntry``."""
    nested = Path(bmrb_dir) / f"bmr{entry_id}" / f"bmr{entry_id}_3.str"
    return nested if nested.exists() else Path(bmrb_dir) / f"bmr{entry_id}_3.str"


def read_entry_xrefs(bmrb_dir: Path, entry_id: str) -> EntryXrefs:
    return parse_xrefs(pynmrstar.Entry.from_file(str(entry_file(bmrb_dir, entry_id))))


def load_xrefs(bmrb_dir: Path, entry_ids: Iterable[str]) -> dict[str, EntryXrefs]:
    """Parse every distinct entry once. An entry whose file is missing or does
    not parse is left out (its chains then get null link columns), with a
    warning -- never an empty, "no links" result."""
    out: dict[str, EntryXrefs] = {}
    failed = []
    for entry_id in sorted({str(e) for e in entry_ids}):
        try:
            out[entry_id] = read_entry_xrefs(bmrb_dir, entry_id)
        except (OSError, ValueError) as exc:
            failed.append(entry_id)
            log.debug("xrefs: cannot read BMRB %s: %s", entry_id, exc)
    if failed:
        log.warning(
            "xrefs: %d entries unreadable under %s (e.g. %s)",
            len(failed),
            bmrb_dir,
            ", ".join(failed[:5]),
        )
    return out


def _is_full_match(x: Xref) -> bool:
    return (x.identity or 0) >= 100 and (x.coverage or 0) >= 100


def uniprot_ids(links: Iterable[Xref]) -> tuple[list[str], str | None]:
    """The chain's own UniProt accessions at the best evidence level it has.

    The author-supplied accessions, if there are any; else every non-author
    accession (``unspecified`` or ``blast``) that is 100 % identical over 100 %
    of the chain. Several such accessions are usually one sequence in several
    species or strains: all are listed, unranked, so a join on any of them
    hits. A hit below that bar is a homolog or a fragment match and is never
    listed (``xrefs`` keeps it). Returns ``(accessions, source)`` with the
    strongest source among them, or ``([], None)``.
    """
    ranked = sorted(
        (x for x in links if x.db == UNIPROT and x.accession is not None),
        key=_sort_key,
    )
    tier = [x for x in ranked if x.source in ("exact", "author")]
    if not tier:
        tier = [x for x in ranked if _is_full_match(x)]
    accessions = list(dict.fromkeys(x.accession for x in tier))
    return accessions, (tier[0].source if tier else None)


def confident_pdb_ids(links: Iterable[Xref]) -> list[str]:
    """Distinct PDB IDs from ``exact`` and ``author`` links, in rank order."""
    out: list[str] = []
    for x in sorted(links, key=_sort_key):
        if (
            x.db == PDB
            and x.source in ("exact", "author")
            and x.accession is not None
            and x.accession not in out
        ):
            out.append(x.accession)
    return out


def xref_records(links: Iterable[Xref]) -> list[dict]:
    return [x.as_record() for x in links]
