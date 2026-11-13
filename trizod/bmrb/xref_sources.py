"""Dated external sources that complete the per-chain cross-references.

The NMR-STAR snapshot (``trizod.bmrb.xrefs``) misses two things BMRB and the
PDB know: part of the BMRB<->PDB same-deposition links (BMRB keeps them in an
internal tracker), and the UniProt accession of the PDB chain itself. Three
public flat files fill both:

``adit_nmr_matched_pdb_bmrb_entry_ids.csv`` (BMRB)
    BMRB entry -> PDB entry. Per BMRB's README, for BMRB IDs above 15000 both
    depositions came from one ADIT-NMR session (``exact``); for older entries
    the match was supplied by the author or a BMRB annotator (``unspecified``).
``pdb_seqres.txt.gz`` (wwPDB)
    The SEQRES sequence of every PDB chain.
``pdb_chain_uniprot.tsv.gz`` (SIFTS, EBI)
    PDB chain -> UniProt accession(s).

A chain's UniProt accession is taken from SIFTS only for a chain of a
high-confidence structure -- the entry's own (``exact`` or BMRB-matched) or one
the depositor names (``author``), never a BLAST PDB hit -- whose sequence
matches the TriZOD chain (``MIN_COVERAGE`` / ``MIN_IDENTITY``). That is what
makes the link per chain: in a complex, each partner gets its own accession.

The files are downloaded once by ``scripts/fetch_xref_sources.py`` into a dated
directory with a ``MANIFEST.json`` (URL, retrieval time, Last-Modified, size,
SHA-256); ``ExternalSources.load`` refuses files that do not match it, and the
release Parquet records the manifest in its metadata.
"""

from __future__ import annotations

import csv
import difflib
import gzip
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from trizod.bmrb import xrefs as xr

BMRB_PDB_MATCH = "adit_nmr_matched_pdb_bmrb_entry_ids.csv"
SEQRES = "pdb_seqres.txt.gz"
SIFTS = "pdb_chain_uniprot.tsv.gz"
MANIFEST = "MANIFEST.json"

#: Where each file is published.
URLS = {
    BMRB_PDB_MATCH: (
        "https://bmrb.io/ftp/pub/bmrb/nmr_pdb_integrated_data/" + BMRB_PDB_MATCH
    ),
    SEQRES: "https://files.wwpdb.org/pub/pdb/derived_data/" + SEQRES,
    SIFTS: "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/" + SIFTS,
}

#: ``origin`` values of the links this module adds.
ORIGIN_MATCH = "bmrb_pdb_match"
ORIGIN_SIFTS = "sifts"

#: BMRB IDs above this were deposited in one ADIT-NMR session with their PDB
#: entry (BMRB README for ``adit_nmr_matched_pdb_bmrb_entry_ids.csv``).
ADIT_SESSION_MIN_ID = 15000

#: A PDB chain lends its SIFTS accession only if it covers this share of the
#: TriZOD chain (%) at this identity (%). Same-deposition structures normally
#: match exactly; the margin absorbs tags and single mutations.
MIN_COVERAGE = 90.0
MIN_IDENTITY = 95.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sequence_match(query: str, subject: str) -> tuple[float, float]:
    """``(coverage, identity)`` of ``query`` (the TriZOD chain) on ``subject``.

    Identical residues M from difflib's matching blocks. Coverage = M / len
    (query); identity = M / the longer of the two aligned spans, so a tag
    outside the aligned region costs nothing and a substitution or indel inside
    it does.
    """
    if not query or not subject:
        return 0.0, 0.0
    sm = difflib.SequenceMatcher(None, query, subject, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    if not blocks:
        return 0.0, 0.0
    m = sum(b.size for b in blocks)
    first, last = blocks[0], blocks[-1]
    span = max(last.a + last.size - first.a, last.b + last.size - first.b)
    return round(100 * m / len(query), 2), round(100 * m / span, 2)


def read_bmrb_pdb_match(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            pdb_id = xr.normalize_pdb(row[1])
            if row[0].strip() and pdb_id:
                out.setdefault(row[0].strip(), set()).add(pdb_id)
    return out


def read_seqres(path: Path, pdb_ids: Iterable[str]) -> dict[str, dict[str, str]]:
    """Protein chains of the wanted structures: ``PDB ID -> chain -> sequence``."""
    wanted = {p.lower() for p in pdb_ids}
    out: dict[str, dict[str, str]] = {}
    key = None
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                fields = line[1:].split()
                pdb_id, _, chain = fields[0].partition("_")
                protein = len(fields) > 1 and fields[1] == "mol:protein"
                key = (pdb_id, chain) if pdb_id in wanted and protein else None
            elif key is not None:
                out.setdefault(key[0].upper(), {})[key[1]] = line.strip()
                key = None
    return out


def read_sifts(path: Path, pdb_ids: Iterable[str]) -> dict[str, dict[str, list[str]]]:
    """``PDB ID -> chain -> UniProt accessions`` (several for a chimera)."""
    wanted = {p.lower() for p in pdb_ids}
    out: dict[str, dict[str, list[str]]] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or fields[0] not in wanted:
                continue
            accs = out.setdefault(fields[0].upper(), {}).setdefault(fields[1], [])
            if fields[2] not in accs:
                accs.append(fields[2])
    return out


def _is_after_adit_cutoff(entry_id: str) -> bool:
    return entry_id.isdigit() and int(entry_id) > ADIT_SESSION_MIN_ID


@dataclass
class ExternalSources:
    bmrb_pdb: dict[str, set[str]] = field(default_factory=dict)
    seqres: dict[str, dict[str, str]] = field(default_factory=dict)
    sifts: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        directory: Path,
        entry_ids: Iterable[str],
        extra_pdb_ids: Iterable[str] = (),
    ) -> ExternalSources:
        """Read the three files for ``entry_ids``, after checking each against
        ``MANIFEST.json``. ``extra_pdb_ids`` are further structures whose chains
        are needed (the snapshot's own ``exact`` and ``author`` PDB entries)."""
        directory = Path(directory)
        manifest = json.loads((directory / MANIFEST).read_text())
        for name in (BMRB_PDB_MATCH, SEQRES, SIFTS):
            expected = manifest.get("files", {}).get(name, {}).get("sha256")
            got = sha256(directory / name)
            if got != expected:
                raise ValueError(
                    f"{directory / name}: sha256 {got} does not match "
                    f"{MANIFEST} ({expected})"
                )
        entries = {str(e) for e in entry_ids}
        match = read_bmrb_pdb_match(directory / BMRB_PDB_MATCH)
        bmrb_pdb = {e: p for e, p in match.items() if e in entries}
        pdb_ids = set(extra_pdb_ids).union(*bmrb_pdb.values())
        return cls(
            bmrb_pdb=bmrb_pdb,
            seqres=read_seqres(directory / SEQRES, pdb_ids),
            sifts=read_sifts(directory / SIFTS, pdb_ids),
            manifest=manifest,
        )

    def provenance(self) -> dict:
        """What the release metadata records about these sources."""
        return {
            "files": self.manifest.get("files", {}),
            "adit_session_min_id": ADIT_SESSION_MIN_ID,
            "min_coverage": MIN_COVERAGE,
            "min_identity": MIN_IDENTITY,
        }

    def links(
        self, entry_id: str, sequence: str, snapshot_links: Iterable[xr.Xref]
    ) -> list[xr.Xref]:
        """The links these sources add to one chain, given its snapshot links."""
        snapshot_links = list(snapshot_links)
        exact = {
            x.accession
            for x in snapshot_links
            if x.db == xr.PDB and x.source == "exact" and x.accession
        }
        author = {
            x.accession
            for x in snapshot_links
            if x.db == xr.PDB and x.source == "author" and x.accession
        }
        matched = self.bmrb_pdb.get(str(entry_id), set())
        source = "exact" if _is_after_adit_cutoff(str(entry_id)) else "unspecified"
        out = [
            xr.Xref(xr.PDB, p, source, None, None, ORIGIN_MATCH, "PDB", p)
            for p in sorted(matched - exact)
        ]
        hits = []
        for pdb_id in sorted(matched | exact | author):
            for chain, subject in self.seqres.get(pdb_id, {}).items():
                coverage, identity = sequence_match(sequence, subject)
                if coverage < MIN_COVERAGE or identity < MIN_IDENTITY:
                    continue
                for acc in self.sifts.get(pdb_id, {}).get(chain, []):
                    hits.append((-coverage, -identity, pdb_id, chain, acc))
        for neg_cov, neg_idt, pdb_id, chain, acc in sorted(hits):
            out.append(
                xr.Xref(
                    xr.UNIPROT,
                    xr.normalize_uniprot(acc),
                    "sifts",
                    -neg_idt,
                    -neg_cov,
                    ORIGIN_SIFTS,
                    "SIFTS",
                    acc,
                    f"{pdb_id} chain {chain}",
                )
            )
        return xr.rank_links(out)


def fetch(directory: Path, retrieved: str) -> dict:
    """Download the three files into ``directory`` and write ``MANIFEST.json``.

    ``retrieved`` is the UTC timestamp to record. Used by
    ``scripts/fetch_xref_sources.py``; never called during a build.
    """
    import urllib.request

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, url in URLS.items():
        request = urllib.request.Request(url, headers={"User-Agent": "trizod"})
        with urllib.request.urlopen(request, timeout=600) as resp:
            last_modified = resp.headers.get("Last-Modified")
            (directory / name).write_bytes(resp.read())
        files[name] = {
            "url": url,
            "retrieved": retrieved,
            "last_modified": last_modified,
            "bytes": (directory / name).stat().st_size,
            "sha256": sha256(directory / name),
        }
    manifest = {"retrieved": retrieved, "files": files}
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest
