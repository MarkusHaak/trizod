"""Labelled residues outside the SIFTS UniProt span, per chain (release
validation 2026-11-17c, G43).

A construct carries residues its UniProt entry does not: tags, linkers, a
fusion partner. The construct-caveat class ``outside_uniprot_span_sifts`` lists
the chains that hold any; its CHANGELOG share must be residue-level (the
labelled residues outside the span over the labelled residues assessed), not
the scored residues of the chains listed (22.8 % in the round-3 deposit against
a residue-level 2.1 %). This module measures it the way the validation's
verifier did, from the release Parquet and the dated cross-reference sources
(``trizod.bmrb.xref_sources``: wwPDB SEQRES and SIFTS):

* a chain is assessed when its ``uniprot_source`` is ``sifts`` and one of its
  ``xrefs`` is a SIFTS UniProt link (``details`` "<pdb> chain <id>") to an
  accession in its ``uniprot_ids``;
* its sequence is aligned to the SEQRES of every such PDB chain (semi-global
  Needleman-Wunsch, match +2, mismatch -1, linear gap -2, free end gaps);
* a residue is inside when it is aligned to a SEQRES position within a SIFTS
  segment (``RES_BEG``-``RES_END``) of the accession.

:func:`outside_span` returns ``{chain ID: (labelled outside, labelled
assessed)}``.
"""

from __future__ import annotations

import contextlib
import gzip
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SEQRES = "pdb_seqres.txt.gz"
SIFTS = "pdb_chain_uniprot.tsv.gz"
MATCH, MISMATCH, GAP = 2, -1, 2
_DETAILS = re.compile(r"(\w{4}) chain (\S+)")


def align_map(a: str, b: str) -> np.ndarray:
    """For each position of ``a`` the aligned position of ``b`` (0-based) or -1:
    semi-global Needleman-Wunsch with free end gaps on both sequences."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.full(n, -1, dtype=np.int64)
    A = np.frombuffer(a.encode(), dtype=np.uint8)
    B = np.frombuffer(b.encode(), dtype=np.uint8)
    H = np.zeros((n + 1, m + 1), dtype=np.int32)
    jj = np.arange(m + 1, dtype=np.int32) * GAP
    for i in range(1, n + 1):
        s = np.where(A[i - 1] == B, MATCH, MISMATCH).astype(np.int32)
        T = np.empty(m + 1, dtype=np.int32)
        T[0] = 0
        T[1:] = np.maximum(H[i - 1, :-1] + s, H[i - 1, 1:] - GAP)
        H[i] = np.maximum.accumulate(T + jj) - jj
    i_end, j_end = n, int(np.argmax(H[n]))
    if H[:, m].max() > H[n, j_end]:
        i_end, j_end = int(np.argmax(H[:, m])), m
    out = np.full(n, -1, dtype=np.int64)
    i, j = i_end, j_end
    while i > 0 and j > 0:
        s = MATCH if a[i - 1] == b[j - 1] else MISMATCH
        if H[i, j] == H[i - 1, j - 1] + s:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif H[i, j] == H[i - 1, j] - GAP:
            i -= 1
        else:
            j -= 1
    return out


def sifts_links(row: dict) -> list[tuple[tuple[str, str], str]]:
    """``((pdb, chain), accession)`` of the chain's SIFTS UniProt links to its
    own accessions (rows whose ``uniprot_source`` is ``sifts``)."""
    if row.get("uniprot_source") != "sifts":
        return []
    accs = set(row.get("uniprot_ids") or ())
    out = []
    for x in row.get("xrefs") or ():
        if (
            x.get("source") == "sifts"
            and x.get("db") == "UniProt"
            and x.get("accession") in accs
            and x.get("details")
        ):
            m = _DETAILS.match(x["details"])
            if m:
                out.append(((m.group(1).lower(), m.group(2)), x["accession"]))
    return out


def _read_seqres(path: Path, wanted: set) -> dict[tuple[str, str], str]:
    out, cur, buf = {}, None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur is not None:
                    out[cur] = "".join(buf)
                pdb, _, chain = line[1:].split()[0].partition("_")
                cur = (pdb.lower(), chain) if (pdb.lower(), chain) in wanted else None
                buf = []
            elif cur is not None:
                buf.append(line.strip())
    if cur is not None:
        out[cur] = "".join(buf)
    return out


def _read_segments(path: Path, wanted: dict) -> dict:
    segs = defaultdict(list)
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 5 or p[0] == "PDB":
                continue
            k = (p[0].lower(), p[1])
            if p[2] in wanted.get(k, ()):
                with contextlib.suppress(ValueError):
                    segs[(k, p[2])].append((int(p[3]), int(p[4])))
    return segs


def outside_span(
    rows: list[dict], xref_dir, splits=("train", "test_trizod")
) -> dict[str, tuple[int, int]]:
    """``{chain ID: (labelled residues outside the SIFTS UniProt span,
    labelled residues assessed)}`` for every assessed chain of ``splits``."""
    xref_dir = Path(xref_dir)
    links = {}
    need: dict[tuple[str, str], set] = defaultdict(set)
    for r in rows:
        if r["split"] not in splits:
            continue
        lk = sifts_links(r)
        if lk:
            links[r["id"]] = (r, lk)
            for k, acc in lk:
                need[k].add(acc)
    if not links:
        return {}
    seqres = _read_seqres(xref_dir / SEQRES, set(need))
    segs = _read_segments(xref_dir / SIFTS, need)
    out = {}
    for cid, (r, lk) in links.items():
        seq = r["sequence"]
        mask = np.asarray([bool(m) for m in r["mask"]], dtype=bool)
        inside = np.zeros(len(seq), dtype=bool)
        assessed = False
        for k, acc in lk:
            sr = seqres.get(k)
            if not sr or (k, acc) not in segs:
                continue
            assessed = True
            allowed = np.zeros(len(sr), dtype=bool)
            for beg, end in segs[(k, acc)]:
                allowed[max(0, beg - 1) : min(len(sr), end)] = True
            mp = align_map(seq, sr)
            hit = mp >= 0
            inside[hit] |= allowed[mp[hit]]
        if assessed:
            out[cid] = (int((mask & ~inside).sum()), int(mask.sum()))
    return out
