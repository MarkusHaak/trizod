"""Residual similarity between the test set and training, computed for the deposit.

The leakage rule (>= 30 % identity over >= 80 % of either chain, >= 27 bits,
every alignment mmseqs reports) separates training from the test sets; it
controls homology at that level and does not eliminate it. ``trizod dataset
finalize`` documents what remains below it, from the release Parquet alone:

* :func:`test_components` -- the homology components of the TriZOD test set
  under the leakage rule: the unit to resample for confidence intervals
  (``test/TriZOD_test_set_components.tsv``);
* :func:`shared_accessions` -- test x training pairs that share a UniProt
  accession (another construct of the same protein), each with its longest
  identical run, best alignment and whether any alignment meets the rule
  (``test/test_training_shared_accessions.tsv``; decision: kept, below the
  rule);
* :func:`remote_homologs` -- test chains with a profile hit among the tolerant
  training representatives (a 3-iteration mmseqs2 profile search, SETH's
  audit), listed in ``known_issues.tsv`` as ``remote_homolog_profile``;
* :func:`shared_construct_issues` -- the reviewed tag-stripped / identical-run
  pairs of :mod:`trizod.dataset.shared_constructs` as known issues;
* :func:`closest_training_chains` -- each test chain's closest tolerant
  training representative at any identity, with identity, coverage, bit score
  and whether it reaches E ≤ 1e-3 or 27 bits
  (``test/test_residual_similarity.tsv``): a description of the split.

Every search runs in a temporary directory; only the tables above ship.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from trizod.dataset import mmseqs as mm
from trizod.dataset import shared_constructs as sc
from trizod.io.fasta import write_fasta

TEST_SPLIT = "test_trizod"
TRAINING_SPLITS = ("train", "redundant")
TOLERANT_UP = ("tolerant", "moderate", "strict")
#: The remote-homology profile search (release audit E25, as SETH's): test
#: chains as queries, 3 iterations, -s 7.5, mmseqs' default E-value 1e-3; a hit
#: counts at > 20 % identity over >= 80 % of either chain.
PROFILE_ITERATIONS = 3
PROFILE_MIN_ID = 20.0
PROFILE_MIN_COV = 0.8
PROFILE_EVALUE = 1e-3
#: An identical run of at least this length marks an overlapping construct in
#: the shared-accession table (the release gate's substring length).
OVERLAP_RUN = mm.SUBSTRING_MIN_LEN
#: UniProt evidence levels that name the chain's own protein (BLAST hits do not).
OWN_ACCESSION_SOURCES = ("exact", "author", "sifts")


def _key(cid: str):
    return tuple(int(p) if p.isdigit() else p for p in str(cid).split("_"))


def _scored(row) -> int:
    return sum(1 for m in row["mask"] if m)


def _require_mmseqs(what: str) -> None:
    if not shutil.which("mmseqs"):
        raise SystemExit(f"mmseqs not found in PATH ({what})")


# --------------------------------------------------------------------------- #
# homology components of the test set (E24)
# --------------------------------------------------------------------------- #


def components(ids, pairs) -> dict[str, str]:
    """``{id: component}``: connected components of ``ids`` under ``pairs``,
    each named after its naturally smallest member."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb, key=_key)] = min(ra, rb, key=_key)
    return {i: find(i) for i in parent}


def test_components(rows: list[dict]) -> dict:
    """The homology components of the test chains under the leakage rule (rule
    ``either``, one all-vs-all search, every reported alignment): ``{"ids":
    {id: component}, "pairs": [...], "n", "singletons", "largest"}``."""
    seqs = {r["id"]: r["sequence"] for r in rows if r["split"] == TEST_SPLIT}
    pairs: list[tuple[str, str]] = []
    if len(seqs) > 1:
        _require_mmseqs("test-set components")
        with tempfile.TemporaryDirectory(prefix="trizod_testcomp_") as td:
            work = Path(td)
            fa = work / "test.fasta"
            write_fasta(seqs, fa)
            hits = mm.find_self_leaks(
                fa, work / "hits.tsv", work / "tmp", work / "search.log"
            )
        pairs = sorted(
            {tuple(sorted((h.train_id, h.test_id), key=_key)) for h in hits},
            key=lambda p: (_key(p[0]), _key(p[1])),
        )
    comp = components(sorted(seqs, key=_key), pairs)
    sizes = Counter(comp.values())
    return {
        "ids": comp,
        "pairs": pairs,
        "n": len(sizes),
        "singletons": sum(1 for n in sizes.values() if n == 1),
        "largest": max(sizes.values(), default=0),
        "sizes": sizes,
    }


def write_components(result: dict, path: Path) -> Path:
    """``id``, ``component``, ``component_size`` per test chain."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comp, sizes = result["ids"], result["sizes"]
    with path.open("w", newline="") as fh:
        fh.write("id\tcomponent\tcomponent_size\n")
        for cid in sorted(comp, key=_key):
            fh.write(f"{cid}\t{comp[cid]}\t{sizes[comp[cid]]}\n")
    return path


def components_text(result: dict, n_test: int) -> str:
    return (
        f"the {n_test:,} test chains form {result['n']:,} homology components "
        f"under the leakage rule ({mm.describe_rule('either')}): "
        f"{result['singletons']:,} singletons, the largest of "
        f"{result['largest']:,} chains (`test/TriZOD_test_set_components.tsv`, "
        "column `component`)"
    )


# --------------------------------------------------------------------------- #
# test x training pairs sharing a UniProt accession (E22)
# --------------------------------------------------------------------------- #

SHARED_ACCESSION_COLUMNS = (
    "test_id",
    "training_id",
    "training_split",
    "accessions",
    "test_source",
    "training_source",
    "test_length",
    "training_length",
    "identical_run",
    "best_pident",
    "best_bits",
    "best_test_cov",
    "best_training_cov",
    "meets_rule",
    "decision",
)


def accession_pairs(rows: list[dict]) -> list[dict]:
    """Test x training (train or redundant) pairs whose ``uniprot_ids``
    share an accession, any evidence level, with the shared accessions."""
    by_acc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["split"] in TRAINING_SPLITS:
            for acc in r.get("uniprot_ids") or ():
                by_acc[acc].append(r)
    pairs: dict[tuple[str, str], dict] = {}
    for t in rows:
        if t["split"] != TEST_SPLIT:
            continue
        for acc in t.get("uniprot_ids") or ():
            for r in by_acc.get(acc, ()):
                rec = pairs.setdefault(
                    (t["id"], r["id"]), {"test": t, "training": r, "accessions": []}
                )
                rec["accessions"].append(acc)
    return [pairs[k] for k in sorted(pairs, key=lambda k: (_key(k[0]), _key(k[1])))]


def _align_pairs(pairs: list[dict]) -> dict[tuple[str, str], list[mm.LeakageHit]]:
    """Every alignment of each pair, from one leakage search (both directions)
    of the paired test chains against the paired training chains."""
    if not pairs:
        return {}
    _require_mmseqs("shared-accession alignments")
    test = {p["test"]["id"]: p["test"]["sequence"] for p in pairs}
    train = {p["training"]["id"]: p["training"]["sequence"] for p in pairs}
    with tempfile.TemporaryDirectory(prefix="trizod_acc_") as td:
        work = Path(td)
        tr_fa, te_fa = work / "train.fasta", work / "test.fasta"
        write_fasta(train, tr_fa)
        write_fasta(test, te_fa)
        hits = mm.search_both_directions(
            tr_fa, te_fa, work / "hits.tsv", work / "tmp", work / "search.log"
        )
    out: dict[tuple[str, str], list] = defaultdict(list)
    for h in hits:
        out[(h.test_id, h.train_id)].append(h)
    return out


def shared_accessions(rows: list[dict]) -> list[dict]:
    """The shared-accession table (``SHARED_ACCESSION_COLUMNS``): per pair the
    longest identical run, the best-scoring alignment and whether ANY
    alignment meets the leakage rule (the release gate guarantees none does;
    a pair that does is reported, never hidden). Decision: kept, below the
    rule; an identical run of >= ``OVERLAP_RUN`` residues is called an
    overlapping construct."""
    pairs = accession_pairs(rows)
    alignments = _align_pairs(pairs)
    out = []
    for p in pairs:
        t, r = p["test"], p["training"]
        hits = alignments.get((t["id"], r["id"]), [])
        best = max(hits, key=lambda h: (h.bits, h.pident), default=None)
        meets = any(mm.is_leak(h, "either") for h in hits)
        run, _, _ = sc.longest_identical_run(t["sequence"], r["sequence"])
        if meets:
            decision = "meets the leakage rule: not removed (gate defect)"
        elif run >= OVERLAP_RUN:
            decision = (
                f"kept: overlapping construct ({run}-residue identical run), "
                "below the leakage rule"
            )
        else:
            decision = "kept: below the leakage rule"
        out.append(
            {
                "test_id": t["id"],
                "training_id": r["id"],
                "training_split": r["split"],
                "accessions": ",".join(sorted(set(p["accessions"]))),
                "test_source": t.get("uniprot_source") or "",
                "training_source": r.get("uniprot_source") or "",
                "test_length": len(t["sequence"]),
                "training_length": len(r["sequence"]),
                "identical_run": run,
                "best_pident": f"{best.pident:.1f}" if best else "",
                "best_bits": f"{best.bits:g}" if best else "",
                "best_test_cov": f"{best.test_cov:.3f}" if best else "",
                "best_training_cov": f"{best.train_cov:.3f}" if best else "",
                "meets_rule": str(meets).lower(),
                "decision": decision,
            }
        )
    return out


def write_shared_accessions(table: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHARED_ACCESSION_COLUMNS, delimiter="\t")
        w.writeheader()
        for rec in table:
            w.writerow(rec)
    return path


def _own(rec: dict) -> bool:
    return (
        rec["test_source"] in OWN_ACCESSION_SOURCES
        and rec["training_source"] in OWN_ACCESSION_SOURCES
    )


def shared_accession_summary(table: list[dict]) -> dict:
    overlapping = [r for r in table if r["identical_run"] >= OVERLAP_RUN]
    return {
        "file": "test/test_training_shared_accessions.tsv",
        "pairs": len(table),
        "test_chains": len({r["test_id"] for r in table}),
        "training_chains": len({r["training_id"] for r in table}),
        "pairs_own_accession_both": sum(1 for r in table if _own(r)),
        "pairs_meeting_rule": sum(1 for r in table if r["meets_rule"] == "true"),
        "overlapping_constructs": [
            f"{r['test_id']}~{r['training_id']} ({r['identical_run']} aa)"
            for r in overlapping
        ],
        "decision": "kept, below the leakage rule",
    }


def shared_accession_text(summary: dict) -> str:
    over = summary["overlapping_constructs"]
    return (
        f"{summary['pairs']:,} test x training pairs ({summary['test_chains']:,} "
        f"test chains, {summary['training_chains']:,} training chains) share a "
        "UniProt accession, i.e. are other constructs of the same protein; "
        f"{summary['pairs_own_accession_both']:,} of them on author, SIFTS or exact "
        "evidence for both chains. "
        + (
            "None meets the leakage rule"
            if not summary["pairs_meeting_rule"]
            else f"{summary['pairs_meeting_rule']:,} meet the leakage rule"
        )
        + ". They are kept, below the rule"
        + (
            f"; {len(over)} are overlapping constructs with an identical run of "
            f"{OVERLAP_RUN} or more residues ({', '.join(over)}), reviewed and kept"
            if over
            else ""
        )
        + " (`test/test_training_shared_accessions.tsv`: identical run, best "
        "alignment, decision)"
    )


def shared_accession_rows(table: list[dict]) -> list[tuple[str, str, str]]:
    per_test: dict[str, list[str]] = defaultdict(list)
    for r in table:
        per_test[r["test_id"]].append(
            f"{r['training_id']} ({r['training_split']}; {r['accessions']}; "
            f"identical run {r['identical_run']}; best alignment "
            f"{r['best_pident'] or '-'} % over {r['best_test_cov'] or '-'} of the "
            f"test chain, {r['best_bits'] or '-'} bits)"
        )
    return [
        (
            cid,
            "shared_uniprot_accession",
            "shares a UniProt accession with training "
            + "; ".join(per_test[cid])
            + "; kept, below the leakage rule "
            "(test/test_training_shared_accessions.tsv)",
        )
        for cid in sorted(per_test, key=_key)
    ]


# --------------------------------------------------------------------------- #
# remote homologues (E25)
# --------------------------------------------------------------------------- #

PROFILE_FORMAT = "query,target,pident,evalue,bits,qstart,qend,qlen,tstart,tend,tlen"


def profile_search_cmds(q_fa, t_fa, work: Path) -> list[list[str]]:
    """The mmseqs commands of the remote-homology profile search."""
    w = Path(work)
    return [
        ["mmseqs", "createdb", str(q_fa), str(w / "qdb"), "--dbtype", "1"],
        ["mmseqs", "createdb", str(t_fa), str(w / "tdb"), "--dbtype", "1"],
        [
            "mmseqs",
            "search",
            str(w / "qdb"),
            str(w / "tdb"),
            str(w / "res"),
            str(w / "tmp"),
            "--num-iterations",
            str(PROFILE_ITERATIONS),
            "-s",
            "7.5",
            "-a",
            "-e",
            f"{PROFILE_EVALUE:g}",
        ],
        [
            "mmseqs",
            "convertalis",
            str(w / "qdb"),
            str(w / "tdb"),
            str(w / "res"),
            str(w / "hits.tsv"),
            "--format-output",
            PROFILE_FORMAT,
        ],
    ]


def parse_profile_hits(path: Path) -> list[dict]:
    cols = PROFILE_FORMAT.split(",")
    out = []
    with Path(path).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = dict(zip(cols, line.rstrip("\n").split("\t")))
            qcov = (int(r["qend"]) - int(r["qstart"]) + 1) / int(r["qlen"])
            tcov = (int(r["tend"]) - int(r["tstart"]) + 1) / int(r["tlen"])
            out.append(
                {
                    "test_id": r["query"],
                    "training_id": r["target"],
                    "pident": float(r["pident"]),
                    "evalue": float(r["evalue"]),
                    "bits": float(r["bits"]),
                    "test_cov": qcov,
                    "training_cov": tcov,
                }
            )
    return out


def is_profile_hit(hit: dict) -> bool:
    return hit["pident"] > PROFILE_MIN_ID and (
        max(hit["test_cov"], hit["training_cov"]) >= PROFILE_MIN_COV
    )


def remote_homologs(rows: list[dict], hits: list[dict] | None = None) -> dict:
    """Test chains with a profile hit among the tolerant training
    representatives: ``{"best": {test id: best hit}, "n_test", "n_scored",
    "hit_scored"}``. ``hits`` (parsed profile hits) skips the search."""
    test = {r["id"]: r for r in rows if r["split"] == TEST_SPLIT}
    db = {
        r["id"]: r["sequence"]
        for r in rows
        if r["split"] == "train" and r.get("train_tier") in TOLERANT_UP
    }
    if hits is None:
        hits = []
        if test and db:
            _require_mmseqs("remote-homology profile search")
            with tempfile.TemporaryDirectory(prefix="trizod_profile_") as td:
                work = Path(td)
                q_fa, t_fa = work / "test.fasta", work / "train.fasta"
                write_fasta({k: r["sequence"] for k, r in test.items()}, q_fa)
                write_fasta(db, t_fa)
                with (work / "log").open("w") as log:
                    for cmd in profile_search_cmds(q_fa, t_fa, work):
                        subprocess.run(
                            cmd, check=True, stdout=log, stderr=subprocess.STDOUT
                        )
                hits = parse_profile_hits(work / "hits.tsv")
    best: dict[str, dict] = {}
    for h in hits:
        if h["test_id"] in test and h["training_id"] in db and is_profile_hit(h):
            cur = best.get(h["test_id"])
            if cur is None or (h["bits"], h["pident"]) > (cur["bits"], cur["pident"]):
                best[h["test_id"]] = h
    return {
        "best": best,
        "n_test": len(test),
        "n_scored": sum(_scored(r) for r in test.values()),
        "hit_scored": sum(_scored(test[c]) for c in best),
        "n_db": len(db),
    }


def remote_homolog_text(res: dict) -> str:
    share = 100 * res["hit_scored"] / res["n_scored"] if res["n_scored"] else 0.0
    return (
        f"{len(res['best']):,} of the {res['n_test']:,} test chains "
        f"({res['hit_scored']:,} of their {res['n_scored']:,} scored residues, "
        f"{share:.1f} %) have a remote homologue among the {res['n_db']:,} "
        "tolerant training representatives: a hit of a "
        f"{PROFILE_ITERATIONS}-iteration mmseqs2 profile search (E ≤ "
        f"{PROFILE_EVALUE:g}) at > {PROFILE_MIN_ID:g} % identity over ≥ "
        f"{PROFILE_MIN_COV * 100:g} % of either chain (`known_issues.tsv`, "
        "`remote_homolog_profile`). Report a model on all test chains and on "
        "those without such a hit"
    )


def remote_homolog_rows(res: dict) -> list[tuple[str, str, str]]:
    return [
        (
            cid,
            "remote_homolog_profile",
            f"profile hit in tolerant training representative {h['training_id']}: "
            f"{h['pident']:.1f} % identity over {h['test_cov']:.2f} of the test "
            f"chain and {h['training_cov']:.2f} of the training chain, "
            f"{h['bits']:g} bits, E {h['evalue']:.2g} ({PROFILE_ITERATIONS}-"
            "iteration mmseqs2 profile search); below the leakage rule",
        )
        for cid, h in sorted(res["best"].items(), key=lambda kv: _key(kv[0]))
    ]


# --------------------------------------------------------------------------- #
# the reviewed shared-construct pairs (E26)
# --------------------------------------------------------------------------- #


def shared_construct_issues(review: list[dict], rows: list[dict]) -> dict:
    """Known-issue rows and the CHANGELOG sentence of the committed review
    (:mod:`trizod.dataset.shared_constructs`); a stale review is an error."""
    problems = sc.check_review(review, rows)
    if problems:
        raise SystemExit(
            "shared-construct review does not match the release: "
            + "; ".join(problems[:5])
        )
    s = sc.summary(review)
    per_test: dict[str, list[dict]] = defaultdict(list)
    for r in review:
        per_test[r["test_id"]].append(r)
    out = []
    for cid in sorted(per_test, key=_key):
        recs = per_test[cid]
        retire = any(r["outcome"] == "retire" for r in recs)
        parts = "; ".join(
            f"{r['training_id']} ({r['source'].replace('_', ' ')}, tags removed: "
            f"test {r['test_tags']}, training {r['training_tags']}; identical run "
            f"{r['run']}, {r['scored_in_run']} of {r['test_scored']} scored "
            f"residues = {float(r['coverage']) * 100:.1f} %)"
            for r in recs
        )
        out.append(
            (
                cid,
                "shared_construct_retired" if retire else "shared_construct_kept",
                f"{'retired' if retire else 'kept'} under the shared-construct "
                f"criterion (identical run >= {sc.RUN_MIN} residues over >= "
                f"{sc.COVERAGE_MIN * 100:g} % of the scored residues after tag "
                f"stripping): {parts} (test/shared_construct_review.tsv)",
            )
        )
    tag = s["tag_stripped_homology"]
    runs = s["identical_run"]
    text = (
        "Test chains that share a construct with training are decided by one "
        "criterion: a test chain is retired when, after the expression tags are "
        "removed (His tags, GB1, thrombin/TEV/HRV-3C sites, RIKEN flanks), an "
        f"identical run of ≥ {sc.RUN_MIN} residues with a training chain covers "
        f"≥ {sc.COVERAGE_MIN * 100:g} % of its scored residues. Reviewed: {s['pairs']:,} pairs on {s['test_chains']:,} test "
        f"chains ({tag['pairs']:,} pairs on {tag['test_chains']:,} chains that meet "
        f"the leakage rule only after tag stripping, {runs['pairs']:,} with an "
        f"identical run of ≥ {sc.RUN_MIN} residues); retired "
        f"{len(s['retired'])} ({', '.join(s['retired']) or 'none'}), kept "
        f"{s['kept_test_chains']:,} (`test/shared_construct_review.tsv`; "
        "`known_issues.tsv`, `shared_construct_kept`)"
    )
    return {"rows": out, "text": text, "summary": s}


# --------------------------------------------------------------------------- #
# each test chain's closest training chain, at any identity
# --------------------------------------------------------------------------- #

#: The residual-similarity search: test chains as queries against the tolerant
#: training representatives, no identity or coverage floor, prefilter off
#: (every pair is aligned), E ≤ 10.
SEARCH_EVALUE = 10.0
SEARCH_FORMAT = "query,target,pident,evalue,bits,qstart,qend,qlen,tstart,tend,tlen"
#: Two significance levels the closest hit is reported at: mmseqs2's default
#: E-value and the leakage rule's bit-score floor.
SIG_EVALUE = 1e-3
SIG_BITS = mm.LEAKAGE_MIN_BITS
HIT_CRITERIA = ("evalue", "bits")
#: Best-hit identity bins (%), lower bound inclusive; the last bin is closed.
IDENTITY_BINS = (0, 20, 30, 40, 50, 60, 70, 80, 90, 100)
SIMILARITY_TABLE = "test/test_residual_similarity.tsv"
SIMILARITY_COLUMNS = (
    "test_id",
    "length",
    "scored_residues",
    "best_training_id",
    "best_pident",
    "best_test_cov",
    "best_training_cov",
    "best_evalue",
    "best_bits",
    "meets_evalue_1e-3",
    "meets_27_bits",
    "remote_homolog_profile",
)


def criterion_label(criterion: str) -> str:
    return {
        "evalue": f"E ≤ {SIG_EVALUE:g}",
        "bits": f"≥ {SIG_BITS:g} bits",
    }[criterion]


def meets(hit: dict, criterion: str | None) -> bool:
    """Whether an alignment counts as a hit under ``criterion`` (None: any
    alignment the search reports)."""
    if criterion is None:
        return True
    if criterion == "evalue":
        return hit["evalue"] <= SIG_EVALUE
    if criterion == "bits":
        return hit["bits"] >= SIG_BITS
    raise ValueError(f"unknown hit criterion {criterion!r}")


def similarity_search_cmd(q_fa, t_fa, out_tsv, tmp) -> list[str]:
    """The mmseqs command of the residual-similarity search."""
    return [
        "mmseqs",
        "easy-search",
        str(q_fa),
        str(t_fa),
        str(out_tsv),
        str(tmp),
        "--min-seq-id",
        "0",
        "-c",
        "0",
        "--alignment-mode",
        "3",
        "-s",
        "7.5",
        "--comp-bias-corr",
        "0",
        "--mask",
        "0",
        "--prefilter-mode",
        str(mm.LEAKAGE_PREFILTER_MODE),
        "-e",
        f"{SEARCH_EVALUE:g}",
        "--max-seqs",
        "100000",
        "--dbtype",
        "1",
        "--format-output",
        SEARCH_FORMAT,
    ]


def describe_similarity_search(n_db: int | None = None) -> str:
    db = f"the {n_db:,} " if n_db is not None else "the "
    return (
        f"mmseqs2 easy-search, test chains as queries against {db}tolerant "
        "training representatives, no identity or coverage floor (--min-seq-id 0 "
        "-c 0), prefilter off (--prefilter-mode 2: every pair aligned), "
        f"--alignment-mode 3 -s 7.5 --comp-bias-corr 0 --mask 0 -e {SEARCH_EVALUE:g}"
    )


def parse_similarity_hits(path: Path) -> list[dict]:
    cols = SEARCH_FORMAT.split(",")
    out = []
    with Path(path).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = dict(zip(cols, line.rstrip("\n").split("\t")))
            out.append(
                {
                    "test_id": r["query"],
                    "training_id": r["target"],
                    "pident": float(r["pident"]),
                    "evalue": float(r["evalue"]),
                    "bits": float(r["bits"]),
                    "test_cov": (int(r["qend"]) - int(r["qstart"]) + 1)
                    / int(r["qlen"]),
                    "training_cov": (int(r["tend"]) - int(r["tstart"]) + 1)
                    / int(r["tlen"]),
                }
            )
    return out


def _better(h: dict, cur: dict | None) -> bool:
    """Highest bit score, then lowest E-value (mmseqs prints whole bits; E
    comes from the raw score, so it separates equal bit scores), then highest
    identity; ties to the naturally smallest training ID, so the choice does
    not depend on mmseqs' output order."""
    if cur is None:
        return True
    a = (h["bits"], -h["evalue"], h["pident"])
    b = (cur["bits"], -cur["evalue"], cur["pident"])
    if a != b:
        return a > b
    return _key(h["training_id"]) < _key(cur["training_id"])


def best_hits(hits: list[dict], criterion: str | None = None) -> dict[str, dict]:
    """``{test id: its best alignment}`` among those meeting ``criterion``."""
    best: dict[str, dict] = {}
    for h in hits:
        if meets(h, criterion) and _better(h, best.get(h["test_id"])):
            best[h["test_id"]] = h
    return best


def _round(obj, digits=4):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, digits) for v in obj]
    return obj


def identity_histogram(values) -> dict[str, int]:
    """Counts per ``IDENTITY_BINS`` bin, keyed ``"[lo, hi)"`` (the last
    ``"[90, 100]"``)."""
    edges = IDENTITY_BINS
    out = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        out[f"[{lo}, {hi}{']' if last else ')'}"] = sum(
            1 for v in values if lo <= v < hi or (last and v == hi)
        )
    return out


def _median(values) -> float | None:
    v = sorted(values)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


def _tolerant_reps(rows: list[dict]) -> dict[str, dict]:
    return {
        r["id"]: r
        for r in rows
        if r["split"] == "train" and r.get("train_tier") in TOLERANT_UP
    }


def similarity_hits(rows: list[dict]) -> list[dict]:
    """Run the residual-similarity search (:func:`similarity_search_cmd`)."""
    test = {r["id"]: r["sequence"] for r in rows if r["split"] == TEST_SPLIT}
    db = {k: r["sequence"] for k, r in _tolerant_reps(rows).items()}
    if not test or not db:
        return []
    _require_mmseqs("residual-similarity search")
    with tempfile.TemporaryDirectory(prefix="trizod_sim_") as td:
        work = Path(td)
        q_fa, t_fa = work / "test.fasta", work / "train.fasta"
        write_fasta(test, q_fa)
        write_fasta(db, t_fa)
        with (work / "log").open("w") as log:
            subprocess.run(
                similarity_search_cmd(q_fa, t_fa, work / "hits.tsv", work / "tmp"),
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return parse_similarity_hits(work / "hits.tsv")


def closest_training_chains(
    rows: list[dict], hits: list[dict] | None = None, remote: set | None = None
) -> dict:
    """Each test chain's closest tolerant training representative
    (``SIMILARITY_COLUMNS``) and a summary: how many test chains align at all,
    the closest hit's identity, and for how many test chains the closest hit
    reaches each significance level (``HIT_CRITERIA``). ``hits`` (parsed search hits) skips the search;
    ``remote`` are the test chains with a remote-homology profile hit
    (:func:`remote_homologs`), marked in the table."""
    remote = set(remote or ())
    test = [r for r in rows if r["split"] == TEST_SPLIT]
    test.sort(key=lambda r: _key(r["id"]))
    reps = _tolerant_reps(rows)
    if hits is None:
        hits = similarity_hits(rows)
    hits = [h for h in hits if h["training_id"] in reps]
    best_any = best_hits(hits)
    table = []
    for r in test:
        cid = r["id"]
        b = best_any.get(cid)
        table.append(
            {
                "test_id": cid,
                "length": len(r["sequence"]),
                "scored_residues": _scored(r),
                "best_training_id": b["training_id"] if b else "",
                "best_pident": f"{b['pident']:.1f}" if b else "",
                "best_test_cov": f"{b['test_cov']:.3f}" if b else "",
                "best_training_cov": f"{b['training_cov']:.3f}" if b else "",
                "best_evalue": f"{b['evalue']:.3g}" if b else "",
                "best_bits": f"{b['bits']:g}" if b else "",
                "meets_evalue_1e-3": str(bool(b) and meets(b, "evalue")).lower(),
                "meets_27_bits": str(bool(b) and meets(b, "bits")).lower(),
                "remote_homolog_profile": str(cid in remote).lower(),
            }
        )

    def hit_block(sel: dict[str, dict]) -> dict:
        ids = list(sel.values())
        pid = [h["pident"] for h in ids]
        return {
            "chains": len(ids),
            "best_identity_median": _median(pid),
            "best_identity_max": max(pid, default=None),
            "best_test_cov_median": _median([h["test_cov"] for h in ids]),
            "best_training_cov_median": _median([h["training_cov"] for h in ids]),
            "histogram_best_identity": identity_histogram(pid),
        }

    summary = {
        "file": SIMILARITY_TABLE,
        "search": describe_similarity_search(len(reps)),
        "best_hit": (
            "per test chain the alignment with the highest bit score (then the "
            "lowest E-value, then the highest identity, then the smallest "
            f"training ID) among all alignments the search reports (E ≤ "
            f"{SEARCH_EVALUE:g})"
        ),
        "test_chains": len(test),
        "training_representatives": len(reps),
        "any_alignment": hit_block(best_any),
        "closest_hit_meets": {
            criterion_label(c): sum(meets(b, c) for b in best_any.values())
            for c in HIT_CRITERIA
        },
        "remote_homolog_chains": len(remote & {r["id"] for r in test}),
    }
    return {"table": table, "summary": _round(summary)}


def write_similarity_table(table: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SIMILARITY_COLUMNS, delimiter="\t")
        w.writeheader()
        for rec in table:
            w.writerow(rec)
    return path


def _hist_text(hist: dict[str, int]) -> str:
    parts = []
    for b, n in hist.items():
        lo, hi = b.strip("[]()").split(", ")
        parts.append(f"{lo}–{hi} % {n:,}")
    return ", ".join(parts)


def similarity_text(summary: dict) -> str:
    """The deposit's sentence on how close the test chains come to training
    below the leakage rule (a description of the split)."""
    anyb = summary["any_alignment"]
    med = anyb["best_identity_median"]
    n = summary["closest_hit_meets"]
    ev, bits = (criterion_label(c) for c in HIT_CRITERIA)
    return (
        f"Searched without an identity or coverage floor, {anyb['chains']:,} of "
        f"the {summary['test_chains']:,} test chains align to one of the "
        f"{summary['training_representatives']:,} tolerant training "
        f"representatives (E ≤ {SEARCH_EVALUE:g}); the closest hit's identity has "
        f"median {'–' if med is None else format(med, '.1f')} % (test chains per "
        f"identity bin: {_hist_text(anyb['histogram_best_identity'])}), and it "
        f"reaches {ev} for {n[ev]:,} test chains and {bits} for {n[bits]:,} "
        f"(`{SIMILARITY_TABLE}`: each test chain's closest training chain with its "
        "identity, coverage and bit score)"
    )
