#!/usr/bin/env python3
"""Test chains that share a construct with training once expression tags are
set aside (release validation 2026-11-17c, check E26).

The leakage rule (>= 30 % identity over >= 80 % of either chain, >= 27 bits)
admits a shared domain by design: two constructs of one protein that overlap
over less than 80 % of both chains stay on both sides. A deposited sequence
also carries what the expression system added -- His tags, a GB1 fusion,
protease sites, the RIKEN GSSGSSG/SGPSSG flanks -- which hides how much of a
test chain is in fact a training construct. One written criterion decides
(decision of 17 November 2026, ``docs/dataset/dataset-construction.md``):

    a pinned test chain is retired as ``shared_construct`` when, after the
    expression tags of both chains are removed (:func:`strip_expression_tags`),
    an identical run of >= ``RUN_MIN`` (30) residues with a training chain
    covers >= ``COVERAGE_MIN`` (50 %) of the test chain's scored residues.
    Otherwise it stays a test chain.

"Training chain" is every training row of the release (``split`` train or
redundant). The run is the longest common substring of the two stripped
sequences; its residues are mapped back to the deposited test sequence, and the
coverage is the share of the test chain's scored residues (``mask``) inside it.

:func:`review` applies the criterion to every test x training pair that shares
an identical run of >= ``RUN_MIN`` residues after stripping (found through a
k-mer index, so the scan is exhaustive), and to any extra candidate pairs
(the tag-stripped homology pairs of the validation). The reviewed outcome of
every pair is committed as ``trizod/dataset/data/shared_construct_review.tsv``;
a ``retire`` outcome must have its row in ``test_retirements.tsv`` (reason
``shared_construct``), and finalize ships the table with the deposit.

Usage
-----
    uv run --with pyarrow python -m trizod.dataset.shared_constructs \\
        --parquet <trizod_dataset.parquet> --out <review.tsv> \\
        [--pairs <TSV with test, train columns>]
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

#: Identical-run length and scored-residue coverage of the criterion.
RUN_MIN = 30
COVERAGE_MIN = 0.5
#: The committed review of every candidate pair.
REVIEW_TSV = Path(__file__).resolve().parent / "data" / "shared_construct_review.tsv"
REVIEW_COLUMNS = (
    "test_id",
    "training_id",
    "source",
    "test_tags",
    "training_tags",
    "run",
    "test_scored",
    "scored_in_run",
    "coverage",
    "outcome",
)
#: Why a pair is in the review: an identical run >= RUN_MIN after stripping
#: (found by :func:`review`'s exhaustive scan), or a homology pair found only
#: after stripping (the validation's tag-stripped search).
SOURCES = ("identical_run", "tag_stripped_homology")

# Expression tags and sites (the patterns of the 2026-11-17 validations).
_HIS = re.compile(r"H{6,}")
_THROMBIN = re.compile(r"LVPRGS")
_TEV = re.compile(r"ENLYFQ[GS]")
_HRV3C = re.compile(r"LEVLFQGP")
#: GB1 (the solubility fusion): its N-terminal strand ("KLILNGK" or the
#: "KLALNGK" variant) to its C-terminal "DDATKTFTVTE".
_GB1_START = re.compile(r"KL[IA]LNGKTLKGETT")
_GB1_END = "DDATKTFTVTE"
#: RIKEN structural-genomics flanks.
RIKEN_N, RIKEN_C = "GSSGSSG", "SGPSSG"
#: How far into the chain an N-terminal site is looked for.
_N_WINDOWS = (("thrombin", _THROMBIN, 45), ("TEV", _TEV, 60), ("3C", _HRV3C, 60))
_C_SITES = (("TEV", _TEV), ("thrombin", _THROMBIN), ("3C", _HRV3C))


@dataclass(frozen=True)
class Stripped:
    """A sequence with its expression tags removed: ``seq``, the original
    0-based index of each of its residues (``positions``) and the tags
    removed, in order (``tags``)."""

    seq: str
    positions: tuple[int, ...]
    tags: tuple[str, ...]


def strip_expression_tags(seq: str) -> Stripped:
    """Remove the expression tags of ``seq``: an N-terminal His tag (within
    the first 30 residues) and everything before it, an N-terminal thrombin,
    TEV or HRV-3C site and everything before it, the RIKEN GSSGSSG / SGPSSG
    flanks, a GB1 fusion (from 4 residues before its first strand to the end
    of its last), and a C-terminal His tag (within the last 20 residues) with
    everything after it, the "LE" linker before it and a protease site in the
    20 residues before that."""
    pos = list(range(len(seq)))
    s = seq
    tags: list[str] = []

    def cut(start: int, end: int) -> None:
        nonlocal s, pos
        s = s[:start] + s[end:]
        pos = pos[:start] + pos[end:]

    m = _HIS.search(s[:30])
    if m:
        cut(0, m.end())
        tags.append("N-His")
    for name, pat, window in _N_WINDOWS:
        m = pat.search(s[:window])
        if m:
            cut(0, m.end())
            tags.append(f"N-{name}")
    if s.startswith(RIKEN_N):
        cut(0, len(RIKEN_N))
        tags.append("RIKEN-N")
    if s.endswith(RIKEN_C):
        cut(len(s) - len(RIKEN_C), len(s))
        tags.append("RIKEN-C")
    mg = _GB1_START.search(s)
    start, end = (mg.start() if mg else -1), s.find(_GB1_END)
    if start >= 0 and end > start:
        cut(max(0, start - 4), end + len(_GB1_END))
        tags.append("GB1")
    tail = min(20, len(s))
    m = _HIS.search(s[len(s) - tail :])
    if m:
        at = len(s) - tail + m.start()
        cut(at, len(s))
        if s.endswith("LE"):
            cut(len(s) - 2, len(s))
        tags.append("C-His")
        for name, pat in _C_SITES:
            tail = min(20, len(s))
            hits = list(pat.finditer(s[len(s) - tail :]))
            if hits:
                cut(len(s) - tail + hits[-1].start(), len(s))
                tags.append(f"C-{name}")
    return Stripped(s, tuple(pos), tuple(tags))


def longest_identical_run(a: str, b: str) -> tuple[int, int, int]:
    """``(length, start in a, start in b)`` of the longest common substring of
    ``a`` and ``b`` (0 when they share no residue; the leftmost run in ``a``
    on a tie)."""
    best = (0, 0, 0)
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                v = prev[j - 1] + 1
                cur[j] = v
                if v > best[0]:
                    best = (v, i - v, j - v)
        prev = cur
    return best


@dataclass(frozen=True)
class PairReview:
    """One reviewed test x training pair (the columns of the review TSV)."""

    test_id: str
    training_id: str
    source: str
    test_tags: str
    training_tags: str
    run: int
    test_scored: int
    scored_in_run: int
    coverage: float
    outcome: str

    def row(self) -> dict[str, str]:
        return {
            "test_id": self.test_id,
            "training_id": self.training_id,
            "source": self.source,
            "test_tags": self.test_tags,
            "training_tags": self.training_tags,
            "run": str(self.run),
            "test_scored": str(self.test_scored),
            "scored_in_run": str(self.scored_in_run),
            "coverage": f"{self.coverage:.3f}",
            "outcome": self.outcome,
        }


def meets_criterion(run: int, coverage: float) -> bool:
    """The E26 criterion: an identical run of >= RUN_MIN residues that covers
    >= COVERAGE_MIN of the test chain's scored residues."""
    return run >= RUN_MIN and coverage >= COVERAGE_MIN


def review_pair(
    test_id: str,
    test_seq: str,
    test_mask,
    training_id: str,
    training_seq: str,
    source: str,
) -> PairReview:
    """Apply the criterion to one pair (sequences as deposited, ``test_mask``
    the test chain's scored residues)."""
    ts, rs = strip_expression_tags(test_seq), strip_expression_tags(training_seq)
    run, start, _ = longest_identical_run(ts.seq, rs.seq)
    mask = [bool(m) for m in test_mask]
    scored = sum(mask)
    in_run = sum(1 for p in ts.positions[start : start + run] if mask[p])
    coverage = in_run / scored if scored else 0.0
    return PairReview(
        test_id=test_id,
        training_id=training_id,
        source=source,
        test_tags=",".join(ts.tags) or "-",
        training_tags=",".join(rs.tags) or "-",
        run=run,
        test_scored=scored,
        scored_in_run=in_run,
        coverage=coverage,
        outcome="retire" if meets_criterion(run, coverage) else "keep",
    )


def _kmers(seq: str, k: int) -> set[str]:
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def run_pairs(test: dict[str, str], training: dict[str, str]) -> set[tuple[str, str]]:
    """Every (test, training) pair whose stripped sequences share an identical
    run of >= RUN_MIN residues: a RUN_MIN-mer index of the stripped training
    sequences, so nothing is missed."""
    index: dict[str, set[str]] = {}
    for rid, seq in training.items():
        for km in _kmers(strip_expression_tags(seq).seq, RUN_MIN):
            index.setdefault(km, set()).add(rid)
    pairs = set()
    for tid, seq in test.items():
        for km in _kmers(strip_expression_tags(seq).seq, RUN_MIN):
            for rid in index.get(km, ()):
                pairs.add((tid, rid))
    return pairs


def _key(cid: str):
    return tuple(int(p) if p.isdigit() else p for p in str(cid).split("_"))


def review(
    test: dict[str, tuple[str, list]],
    training: dict[str, str],
    extra_pairs=(),
) -> list[PairReview]:
    """The reviewed pairs: every pair of :func:`run_pairs` (source
    ``identical_run``) and every ``extra_pairs`` (test ID, training ID) pair
    (source ``tag_stripped_homology`` unless it is also a run pair), each
    with the criterion applied. ``test`` maps a test chain ID to its
    ``(sequence, mask)``; ``training`` a training chain ID to its sequence.
    Extra pairs naming a chain that is not test or not training are skipped."""
    found = run_pairs({t: s for t, (s, _) in test.items()}, training)
    pairs = dict.fromkeys(found, "identical_run")
    for t, r in extra_pairs:
        if t in test and r in training:
            pairs.setdefault((t, r), "tag_stripped_homology")
    out = []
    for (t, r), source in sorted(
        pairs.items(), key=lambda kv: (_key(kv[0][0]), _key(kv[0][1]))
    ):
        seq, mask = test[t]
        out.append(review_pair(t, seq, mask, r, training[r], source))
    return out


def read_review(path=None) -> list[dict[str, str]]:
    """The committed review rows (``#`` lines are comments)."""
    path = Path(REVIEW_TSV if path is None else path)
    with path.open(newline="") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    if rows and tuple(rows[0]) != REVIEW_COLUMNS:
        raise ValueError(f"{path}: columns must be {REVIEW_COLUMNS}")
    for r in rows:
        if r["outcome"] not in ("retire", "keep") or r["source"] not in SOURCES:
            raise ValueError(f"{path}: bad row {r}")
    return rows


def check_review(rows: list[dict[str, str]], release_rows: list[dict]) -> list[str]:
    """Problems of the committed review against a release (empty when it
    holds): every pair is re-derived from the release sequences and masks
    (run, scored residues, outcome); a retired test chain must be
    ``test_retired`` with reason ``shared_construct``, a kept one
    ``test_trizod`` (or retired for another reason)."""
    by_id = {r["id"]: r for r in release_rows}
    problems = []
    for rec in rows:
        t, r = rec["test_id"], rec["training_id"]
        test, train = by_id.get(t), by_id.get(r)
        if test is None or train is None:
            problems.append(f"{t} ~ {r}: not in the release")
            continue
        got = review_pair(
            t, test["sequence"], test["mask"], r, train["sequence"], rec["source"]
        )
        if (str(got.run), str(got.scored_in_run), str(got.test_scored)) != (
            rec["run"],
            rec["scored_in_run"],
            rec["test_scored"],
        ) or got.outcome != rec["outcome"]:
            problems.append(f"{t} ~ {r}: recomputed {got.row()} differs from {rec}")
        if rec["outcome"] == "retire" and not (
            test["split"] == "test_retired"
            and test.get("retired_reason") == "shared_construct"
        ):
            problems.append(f"{t}: reviewed retire, but split {test['split']}")
        if rec["outcome"] == "keep" and test["split"] not in (
            "test_trizod",
            "test_retired",
        ):
            problems.append(f"{t}: reviewed keep, but split {test['split']}")
    return problems


def summary(rows: list[dict[str, str]]) -> dict:
    """Counts of the review: pairs and test chains per source and outcome."""
    out = {"pairs": len(rows), "test_chains": len({r["test_id"] for r in rows})}
    for src in SOURCES:
        sub = [r for r in rows if r["source"] == src]
        out[src] = {"pairs": len(sub), "test_chains": len({r["test_id"] for r in sub})}
    retired = sorted({r["test_id"] for r in rows if r["outcome"] == "retire"}, key=_key)
    out["retired"] = retired
    out["kept_test_chains"] = len(
        {r["test_id"] for r in rows if r["outcome"] == "keep"} - set(retired)
    )
    return out


def write_review(reviews: list[PairReview], path: Path) -> Path:
    path = Path(path)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS, delimiter="\t")
        w.writeheader()
        for rv in reviews:
            w.writerow(rv.row())
    return path


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    with Path(path).open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out = []
    for r in rows:
        t = r.get("test") or r.get("test_id")
        tr = r.get("train") or r.get("training") or r.get("training_id")
        if t and tr:
            out.append((t, tr))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parquet", type=Path, required=True, help="release Parquet")
    ap.add_argument("--out", type=Path, required=True, help="review TSV to write")
    ap.add_argument(
        "--pairs",
        type=Path,
        action="append",
        default=[],
        help="extra candidate pairs (TSV with test and train/training columns), "
        "e.g. the validation's tag-stripped homology pairs (repeatable)",
    )
    ap.add_argument(
        "--test-splits",
        default="test_trizod",
        help="comma-separated splits whose chains are test chains "
        "(default: test_trizod)",
    )
    args = ap.parse_args(argv)
    import pyarrow.parquet as pq

    cols = ["id", "sequence", "mask", "split"]
    rows = pq.read_table(args.parquet, columns=cols).to_pylist()
    test_splits = set(args.test_splits.split(","))
    test = {
        r["id"]: (r["sequence"], r["mask"]) for r in rows if r["split"] in test_splits
    }
    training = {
        r["id"]: r["sequence"] for r in rows if r["split"] in ("train", "redundant")
    }
    extra = [p for f in args.pairs for p in _read_pairs(f)]
    reviews = review(test, training, extra)
    write_review(reviews, args.out)
    retired = sorted({rv.test_id for rv in reviews if rv.outcome == "retire"}, key=_key)
    print(
        f"{len(reviews)} pairs on {len({rv.test_id for rv in reviews})} test chains; "
        f"retire: {retired or 'none'} -> {args.out}"
    )


if __name__ == "__main__":
    main()
