"""Step 1-4 of the G-threshold calibration: chain filter, UniProt per chain
(release xref logic), UniProt sequences, residue mapping, reference labels.

Writes out/residues.parquet (one row per scored, mapped, labelled residue),
out/chains_filtered.parquet and out/counts_mapping.json (counts at every step).

Run from the festive-jang-577c3d worktree:
  uv run --with pyarrow --with pandas --with scikit-learn --with scipy \
      --with biopython python <this file>
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

# Data root: the checkout that holds data/ (set TRIZOD_ROOT when running from a worktree).
ROOT = _Path(
    _os.environ.get("TRIZOD_ROOT")
    or next(
        p
        for p in _Path(__file__).resolve().parents
        if (p / "data" / "interim").is_dir()
    )
)

import gzip
import hashlib
import json
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from trizod.bmrb import xref_sources as xs
from trizod.bmrb import xrefs as xr

BASE = Path(f"{ROOT}/data/interim/threshold-calibration-2026-11-16")
DL = BASE / "downloads"
OUT = BASE / "out"
PARQUET = Path(
    _os.environ.get("TRIZOD_RELEASE_PARQUET")
    or f"{ROOT}/data/interim/rebuild-2026-11-11/release/trizod_dataset.parquet"
)
BMRB_DIR = Path(f"{ROOT}/data/raw/bmrb_entries_2026-11-11")
SOURCES_DIR = Path(f"{ROOT}/data/external/xref_sources/2026-11-13")

MIN_IDENTITY = 0.95  # over aligned positions
MIN_COVERAGE = 0.50  # of the chain's scored residues

# DisProt positive terms: IDPO "disorder" and its child terms (current IDPO IDs;
# IDPO:00076/00077/00078 in the old numbering used by CAID).
DISPROT_POS = {"IDPO:0000002", "IDPO:0000003", "IDPO:0000004"}

DENATURED_STATES = {
    "denatured",
    "unfolded",
    "partially denatured",
    "partially unfolded",
}


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------- chain filter
def chain_filter(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts = {"all_chains": len(d)}
    ps = d.physical_state.fillna("").str.strip().str.lower()
    rules = [
        ("is_bound False", ~d.is_bound.astype(bool)),
        (
            "physical_state not in {denatured, unfolded, partially denatured, partially unfolded}",
            ~ps.isin(DENATURED_STATES),
        ),
        ("cosolvent_evidence False", ~d.cosolvent_evidence.astype(bool)),
        ("membrane_mimetic null", d.membrane_mimetic.isna()),
        ("sample_state_evidence == solution", d.sample_state_evidence == "solution"),
        ("n_scored > 0", d.n_scored > 0),
    ]
    keep = np.ones(len(d), bool)
    for name, m in rules:
        keep &= m.to_numpy()
        counts[f"after: {name}"] = int(keep.sum())
    f = d[keep].copy()
    counts["kept_by_split"] = f.split.value_counts().to_dict()
    return f, counts


# ------------------------------------------------------------------ xrefs (mp)
def _read(entry_id):
    try:
        return entry_id, xr.read_entry_xrefs(BMRB_DIR, entry_id)
    except (OSError, ValueError):
        return entry_id, None


def load_xrefs_mp(entry_ids):
    with Pool(12) as p:
        res = p.map(_read, sorted(entry_ids), chunksize=20)
    return {e: x for e, x in res if x is not None}


# ---------------------------------------------------------------- downloads
def record_download(name, url, path, manifest, note=None):
    path = Path(path)
    manifest[name] = {
        "url": url,
        "path": str(path.relative_to(DL)),
        # file mtime = time the download (or last append) finished
        "retrieved": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if note:
        manifest[name]["note"] = note


def parse_fasta_text(txt):
    recs, acc, buf = {}, None, []
    for line in txt.splitlines():
        if line.startswith(">"):
            if acc:
                recs[acc] = "".join(buf)
            acc = line[1:].split("|")[1] if "|" in line else line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if acc:
        recs[acc] = "".join(buf)
    return recs


def _get(url, timeout=120):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode()
        except Exception as exc:  # noqa: BLE001
            log("  retry", attempt, url[:100], exc)
            time.sleep(5)
    return ""


def fetch_uniprot(accs):
    """Canonical sequences via UniProt REST (batched); secondary accessions
    resolved with a sec_acc search. Cached in downloads/uniprot/."""
    fa = DL / "uniprot" / "uniprot_canonical.fasta"
    amap = DL / "uniprot" / "secondary_to_primary.json"
    seqs = parse_fasta_text(fa.read_text()) if fa.exists() else {}
    alias = json.loads(amap.read_text()) if amap.exists() else {}
    todo = [a for a in accs if a not in seqs and a not in alias]
    log(f"UniProt: {len(seqs)} cached, {len(todo)} to fetch")
    for i in range(0, len(todo), 400):
        txt = _get(
            "https://rest.uniprot.org/uniprotkb/accessions?format=fasta&accessions="
            + ",".join(todo[i : i + 400])
        )
        with open(fa, "a") as fh:
            fh.write(txt if txt.endswith("\n") or not txt else txt + "\n")
    seqs = parse_fasta_text(fa.read_text()) if fa.exists() else {}
    for a in [a for a in todo if a not in seqs]:
        txt = _get(
            "https://rest.uniprot.org/uniprotkb/search?format=fasta&query=sec_acc:" + a,
            60,
        )
        recs = parse_fasta_text(txt)
        if len(recs) == 1:
            p, s = next(iter(recs.items()))
            alias[a] = p
            if p not in seqs:
                with open(fa, "a") as fh:
                    fh.write(txt if txt.endswith("\n") else txt + "\n")
                seqs[p] = s
        else:
            alias[a] = None
    amap.write_text(json.dumps(alias, indent=1))
    out = {}
    for a in accs:
        if a in seqs:
            out[a] = seqs[a]
        elif alias.get(a) and alias[a] in seqs:
            out[a] = seqs[alias[a]]
    return out, alias


# ----------------------------------------------------------------- alignment
_ALIGNER = None


def _aligner():
    global _ALIGNER
    if _ALIGNER is None:
        from Bio import Align
        from Bio.Align import substitution_matrices

        a = Align.PairwiseAligner()
        a.mode = "local"
        a.substitution_matrix = substitution_matrices.load("BLOSUM62")
        a.open_gap_score = -10
        a.extend_gap_score = -0.5
        _ALIGNER = a
    return _ALIGNER


AA = re.compile("[^ACDEFGHIKLMNPQRSTVWY]")


def align_map(task):
    """-> (chain_id, acc, identity over aligned columns, coverage of scored
    residues by identical aligned residues, {chain_idx: uniprot_pos 1-based})."""
    cid, acc, cseq, useq, scored = task
    c, u = AA.sub("X", cseq), AA.sub("X", useq)
    try:
        aln = _aligner().align(c, u)[0]
    except Exception:  # noqa: BLE001
        return cid, acc, 0.0, 0.0, {}
    n_al = n_id = 0
    m = {}
    for (s0, e0), (s1, e1) in zip(*aln.aligned):
        for i, j in zip(range(s0, e0), range(s1, e1)):
            n_al += 1
            if c[i] == u[j] and c[i] != "X":
                n_id += 1
                m[i] = j + 1
    ident = n_id / n_al if n_al else 0.0
    ss = set(scored)
    mp = {i: p for i, p in m.items() if i in ss}
    cov = len(mp) / len(ss) if ss else 0.0
    return cid, acc, ident, cov, mp


# ------------------------------------------------------------- references
def load_disprot():
    d = json.load(open(DL / "disprot" / "disprot_current.json"))
    regions = defaultdict(list)
    seq = {}
    n_reg = Counter()
    rel = set()
    for e in d["data"]:
        seq[e["acc"]] = e["sequence"]
        for r in e.get("regions", []):
            if r.get("released"):
                rel.add(r["released"])
            if r.get("term_id") in DISPROT_POS and not r.get("term_is_obsolete"):
                regions[e["acc"]].append((int(r["start"]), int(r["end"])))
                n_reg[r["term_id"]] += 1
    return seq, regions, dict(n_reg), max(rel)


def load_xray_ids():
    xray = set()
    with open(DL / "wwpdb_entry_type" / "pdb_entry_type.txt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[2] == "diffraction" and "prot" in p[1]:
                xray.add(p[0].lower())
    return xray


def _read_sifts(path, accs, xray):
    out = defaultdict(lambda: defaultdict(list))
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("PDB\t"):
                continue
            p = line.rstrip("\n").split("\t")
            if p[2] in accs and p[0].lower() in xray:
                try:
                    out[p[2]][(p[0], p[1])].append((int(p[7]), int(p[8])))
                except ValueError:
                    pass
    return out


def mark(ranges, n):
    a = np.zeros(n + 2, bool)
    for b, e in ranges:
        b, e = max(b, 1), min(e, n)
        if e >= b:
            a[b : e + 1] = True
    return a


def main():
    OUT.mkdir(exist_ok=True)
    counts = {}
    d = pd.read_parquet(
        PARQUET,
        columns=[
            "id",
            "entry_id",
            "entity_id",
            "sequence",
            "gscores",
            "k",
            "mask",
            "n_scored",
            "split",
            "train_tier",
            "pool_tier",
            "cluster_repr",
            "is_bound",
            "physical_state",
            "cosolvent_evidence",
            "sample_state_evidence",
            "membrane_mimetic",
            "ph",
            "temperature",
        ],
    )
    f, c = chain_filter(d)
    counts["chain_filter"] = c
    f["n_scored_masked"] = [int(np.asarray(m, bool).sum()) for m in f["mask"]]
    counts["chain_filter"]["scored_residues_after_filter"] = int(
        f.n_scored_masked.sum()
    )
    log(json.dumps(c, indent=1))

    # ---- 1. UniProt per chain (release logic of build_parquet_dataset.collect_columns)
    entry_ids = set(f.entry_id.astype(str))
    t0 = time.time()
    xrefs = load_xrefs_mp(entry_ids)
    log(
        f"xrefs read for {len(xrefs)}/{len(entry_ids)} entries in {time.time() - t0:.0f}s"
    )
    confident_pdb = {
        p
        for e in xrefs.values()
        for links in [e.entry, *e.entity.values()]
        for p in xr.confident_pdb_ids(links)
    }
    external = xs.ExternalSources.load(
        SOURCES_DIR, entry_ids, extra_pdb_ids=confident_pdb
    )
    acc_col, src_col = [], []
    for eid, ent, seq in zip(
        f.entry_id.astype(str), f.entity_id.astype(str), f.sequence
    ):
        ex = xrefs.get(eid)
        if ex is None:
            acc_col.append([])
            src_col.append(None)
            continue
        links = ex.for_entity(ent)
        links = xr.rank_links(links + external.links(eid, seq, links))
        accs, src = xr.uniprot_ids(links)
        acc_col.append(accs)
        src_col.append(src)
    f["uniprot_ids"] = acc_col
    f["uniprot_source"] = src_col
    has = f.uniprot_ids.map(len) > 0
    counts["uniprot"] = {
        "entries_xrefs_read": len(xrefs),
        "entries": len(entry_ids),
        "chains_with_accession": int(has.sum()),
        "chains_without_accession": int((~has).sum()),
        "source_tier_all_filtered": f.uniprot_source.fillna("none")
        .value_counts()
        .to_dict(),
        "with_accession_by_split": f[has].split.value_counts().to_dict(),
        "scored_residues_in_chains_with_accession": int(f[has].n_scored_masked.sum()),
    }
    log(json.dumps(counts["uniprot"], indent=1))

    # ---- 2. UniProt sequences + residue mapping
    all_accs = sorted({a for L in f.uniprot_ids for a in L})
    useqs, alias = fetch_uniprot(all_accs)
    prim = {a: (alias.get(a) or a) for a in all_accs}
    counts["uniprot_sequences"] = {
        "accessions_requested": len(all_accs),
        "sequences_obtained": len(useqs),
        "resolved_via_secondary_accession": sum(
            1 for a in all_accs if a in useqs and alias.get(a)
        ),
        "n_unresolved": sum(1 for a in all_accs if a not in useqs),
        "unresolved_examples": [a for a in all_accs if a not in useqs][:30],
    }
    tasks = []
    for cid, seq, m, accs in zip(f.id, f.sequence, f["mask"], f.uniprot_ids):
        scored = np.flatnonzero(np.asarray(m, bool)).tolist()
        for a in accs:
            if a in useqs:
                tasks.append((cid, a, seq, useqs[a], scored))
    t0 = time.time()
    with Pool(12) as p:
        res = p.map(align_map, tasks, chunksize=50)
    log(f"aligned {len(tasks)} chain-accession pairs in {time.time() - t0:.0f}s")
    passing = defaultdict(list)
    n_id = n_cov = 0
    for cid, acc, ident, cov, mp in res:
        if ident < MIN_IDENTITY:
            n_id += 1
        elif cov < MIN_COVERAGE:
            n_cov += 1
        else:
            passing[cid].append((cov, ident, prim[acc], mp))
    pf = f[f.id.isin(passing)]
    counts["mapping"] = {
        "chain_accession_pairs_aligned": len(tasks),
        "pairs_failing_identity_lt_95pct": n_id,
        "pairs_failing_coverage_lt_50pct": n_cov,
        "chains_with_sequence_but_no_passing_accession": int(
            f[has & f.uniprot_ids.map(lambda L: any(a in useqs for a in L))].shape[0]
            - len(pf)
        ),
        "chains_mapped": len(pf),
        "chains_mapped_by_split": pf.split.value_counts().to_dict(),
        "scored_residues_in_mapped_chains": int(pf.n_scored_masked.sum()),
        "scored_residues_mapped_best_accession": int(
            sum(max(len(x[3]) for x in L) for L in passing.values())
        ),
    }
    log(json.dumps(counts["mapping"], indent=1))

    # ---- 3. references
    dp_seq, dp_regions, dp_nreg, dp_release = load_disprot()
    xray = load_xray_ids()
    accs_used = {x[2] for L in passing.values() for x in L}
    construct = _read_sifts(
        DL / "sifts_chain_uniprot" / "pdb_chain_uniprot.tsv.gz", accs_used, xray
    )
    observed = _read_sifts(
        DL / "sifts_observed" / "uniprot_segments_observed.tsv.gz", accs_used, xray
    )
    useq_prim = {prim[a]: s for a, s in useqs.items()}
    dp_ok = {a: dp_seq[a] == useq_prim.get(a) for a in accs_used & set(dp_seq)}
    counts["references"] = {
        "disprot_latest_region_release_tag": dp_release,
        "disprot_entries": len(dp_seq),
        "disprot_positive_regions_by_term": dp_nreg,
        "xray_protein_pdb_entries": len(xray),
        "mapped_accessions": len(accs_used),
        "mapped_accessions_in_disprot": len(dp_ok),
        "disprot_sequence_equals_current_uniprot": sum(dp_ok.values()),
        "disprot_sequence_differs_dropped_from_setA": sum(
            not v for v in dp_ok.values()
        ),
        "mapped_accessions_with_xray_chain": sum(
            1 for a in accs_used if construct.get(a)
        ),
        "xray_chains_for_mapped_accessions": sum(len(v) for a, v in construct.items()),
    }
    labA, labB = {}, {}
    for a in accs_used:
        n = len(useq_prim.get(a, ""))
        if not n:
            continue
        obs_any = mark([r for v in observed.get(a, {}).values() for r in v], n)
        if dp_ok.get(a):
            dis = mark(dp_regions.get(a, []), n)
            la = np.full(n + 2, -1, np.int8)
            la[obs_any & ~dis] = 0
            la[dis] = 1
            labA[a] = la
        if construct.get(a):
            cons = mark([r for v in construct[a].values() for r in v], n)
            lb = np.full(n + 2, -1, np.int8)
            lb[cons & obs_any] = 0
            lb[cons & ~obs_any] = 1
            labB[a] = lb

    # ---- 4. residue table; per set the best passing accession with any label
    rows = []
    fi = f.set_index("id")
    for cid, L in passing.items():
        r = fi.loc[cid]
        g = np.asarray(r.gscores, float)
        k = np.asarray(r.k, int)
        for setname, lab in (("A", labA), ("B", labB)):
            best = None
            for cov, ident, a, mp in L:
                if a not in lab:
                    continue
                nlab = sum(1 for p in mp.values() if lab[a][p] >= 0)
                key = (cov, ident, nlab, a)
                if nlab and (best is None or key > best[0]):
                    best = (key, a, mp)
            if best is None:
                continue
            _, a, mp = best
            for i, p in mp.items():
                y = lab[a][p]
                if y >= 0 and np.isfinite(g[i]):
                    rows.append((cid, setname, a, i, p, g[i], k[i], int(y)))
    res_df = pd.DataFrame(
        rows, columns=["id", "set", "acc", "idx", "uniprot_pos", "G", "k", "y"]
    )
    meta = fi[
        [
            "split",
            "train_tier",
            "pool_tier",
            "cluster_repr",
            "entry_id",
            "uniprot_source",
        ]
    ]
    res_df = res_df.join(meta, on="id")
    res_df.to_parquet(OUT / "residues.parquet", index=False)
    summ = {}
    for (s, sp), g in res_df.groupby(["set", "split"]):
        summ[f"set {s} | {sp}"] = {
            "chains": int(g.id.nunique()),
            "clusters": int(g.cluster_repr.fillna(g.id).nunique()),
            "accessions": int(g.acc.nunique()),
            "residues": len(g),
            "positives": int(g.y.sum()),
            "negatives": int((g.y == 0).sum()),
        }
    counts["labelled"] = summ
    f.assign(uniprot_ids=f.uniprot_ids.map(";".join))[
        [
            "id",
            "entry_id",
            "split",
            "cluster_repr",
            "sequence",
            "ph",
            "temperature",
            "uniprot_ids",
            "uniprot_source",
            "n_scored_masked",
        ]
    ].to_parquet(OUT / "chains_filtered.parquet", index=False)
    (OUT / "counts_mapping.json").write_text(json.dumps(counts, indent=1, default=str))
    log(json.dumps(summ, indent=1))

    mpath = DL / "MANIFEST.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {}
    for name, url, path, note in [
        (
            "disprot",
            "https://disprot.org/api/search?release=current&show_ambiguous=false&show_obsolete=false&format=json&namespace=all&get_consensus=false",
            DL / "disprot" / "disprot_current.json",
            f"DisProt 'current' release; latest region release tag in file: {dp_release}",
        ),
        (
            "sifts_uniprot_segments_observed",
            "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/uniprot_segments_observed.tsv.gz",
            DL / "sifts_observed" / "uniprot_segments_observed.tsv.gz",
            "file header: 2026/11/08",
        ),
        (
            "sifts_pdb_chain_uniprot",
            "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz",
            DL / "sifts_chain_uniprot" / "pdb_chain_uniprot.tsv.gz",
            None,
        ),
        (
            "wwpdb_pdb_entry_type",
            "https://files.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt",
            DL / "wwpdb_entry_type" / "pdb_entry_type.txt",
            "Last-Modified Fri, 06 Nov 2026 21:30:34 GMT",
        ),
        (
            "uniprot_canonical_fasta",
            "https://rest.uniprot.org/uniprotkb/accessions?format=fasta&accessions=<400 per request>; secondary accessions via https://rest.uniprot.org/uniprotkb/search?format=fasta&query=sec_acc:<acc>",
            DL / "uniprot" / "uniprot_canonical.fasta",
            f"{len(useqs)} accessions resolved",
        ),
        (
            "uniprot_secondary_to_primary",
            "derived from the sec_acc searches above",
            DL / "uniprot" / "secondary_to_primary.json",
            None,
        ),
    ]:
        record_download(name, url, path, man, note)
    mpath.write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    sys.exit(main())
