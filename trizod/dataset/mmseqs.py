"""Shared mmseqs2 helpers for the dataset-build chain.

The ``COMMON`` option block is used verbatim by every clustering step and by the
test-set draw (``--alignment-mode 3 --cov-mode 0 -s 7.5 --comp-bias-corr 0
--mask 0``); do not change it without rebuilding + re-verifying the whole funnel.

Test-set leakage is defined separately, by :func:`leakage_search_cmd` and
:func:`is_leak`: a training chain leaks when an alignment to a held-out test
chain reaches >= 30 % identity over >= 80 % of the TEST chain. Coverage is the
mmseqs definition, the aligned span ``(tend - tstart + 1) / tlen``, gaps
included. ``--cov-mode 0`` would demand 80 % of both chains, so a long training
construct that contains a test protein would pass. The leakage search runs
query = training chains against target = test chains (``--cov-mode 1``,
coverage of the target) with E-values computed against the same (test) target
database in both places that use it: the stage-2 removal in
:mod:`trizod.dataset.redundancy` and the release gate in
:mod:`trizod.dataset.package_release`.

The 80 % is NOT passed to mmseqs (``-c 0``): with any ``-c`` above 0, mmseqs
skips a pair by length ratio before aligning it (under ``--cov-mode 1``, when
the training chain is shorter than 80 % of the test chain). Such a chain can
still span >= 80 % of the test chain with a gap, e.g. a test protein minus an
internal loop, so the test-coverage threshold is applied afterwards, to the
unrounded span parsed from the hit table (:func:`is_leak`).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

COMMON = [
    "--alignment-mode",
    "3",
    "--cov-mode",
    "0",
    "-s",
    "7.5",
    "--comp-bias-corr",
    "0",
    "--mask",
    "0",
]

# Test-set leakage criterion: >= 30 % identity over >= 80 % of the test chain.
LEAKAGE_MIN_SEQ_ID = 0.3
LEAKAGE_MIN_TEST_COV = 0.8
# --cov-mode 1 = coverage of the target; the target is always the test set.
LEAKAGE_COV_MODE = 1
# Coverage threshold handed to mmseqs. 0 on purpose: any value above 0 makes
# mmseqs skip pairs by length ratio before alignment (see module docstring);
# LEAKAGE_MIN_TEST_COV is applied to the parsed alignment span instead.
LEAKAGE_MMSEQS_COV = 0
# Start/end columns rather than qcov/tcov, which mmseqs rounds to 3 decimals.
LEAKAGE_FORMAT = "query,target,pident,evalue,bits,qstart,qend,qlen,tstart,tend,tlen"


def with_cov_mode(opts: list[str], cov_mode: int) -> list[str]:
    """Copy of an option block with its ``--cov-mode`` value replaced."""
    out = list(opts)
    out[out.index("--cov-mode") + 1] = str(cov_mode)
    return out


def run(cmd: list[str], log: Path, append: bool = False) -> None:
    """Run an mmseqs command, teeing stdout+stderr to ``log``."""
    print("  $", " ".join(str(c) for c in cmd))
    mode = "ab" if append else "wb"
    with open(log, mode) as h:
        subprocess.run(cmd, check=True, stdout=h, stderr=subprocess.STDOUT)


def cluster_tsv_groups(tsv: Path) -> dict:
    """Parse an mmseqs ``*_cluster.tsv`` into {representative: [members]}."""
    groups: dict[str, list[str]] = {}
    with tsv.open() as f:
        for line in f:
            if line.strip():
                rep, mem = line.rstrip("\n").split("\t")[:2]
                groups.setdefault(rep, []).append(mem)
    return groups


@dataclass(frozen=True)
class LeakageHit:
    """One training/test alignment that meets the leakage criterion."""

    train_id: str
    test_id: str
    pident: float  # percent identity over the alignment
    evalue: float
    bits: float
    train_cov: float  # aligned span / length of the training chain (gaps count)
    test_cov: float  # aligned span / length of the test chain (gaps count)
    train_len: int
    test_len: int


def leakage_search_cmd(
    train_fa: Path, test_fa: Path, out_tsv: Path, tmp: Path
) -> list[str]:
    """``mmseqs easy-search`` for the test-set leakage criterion.

    Query = training chains, target = test chains, ``--min-seq-id 0.3`` and
    ``--cov-mode 1 -c 0``; every other option is the shared ``COMMON`` block.
    mmseqs gets no coverage threshold, so it aligns every prefilter hit
    whatever the length ratio; the >= 80 % test-coverage threshold is applied
    to the parsed hits (:func:`is_leak`).
    """
    return [
        "mmseqs",
        "easy-search",
        str(train_fa),
        str(test_fa),
        str(out_tsv),
        str(tmp),
        "--min-seq-id",
        str(LEAKAGE_MIN_SEQ_ID),
        "-c",
        str(LEAKAGE_MMSEQS_COV),
        *with_cov_mode(COMMON, LEAKAGE_COV_MODE),
        "--format-output",
        LEAKAGE_FORMAT,
    ]


def parse_leakage_hits(tsv: Path) -> list[LeakageHit]:
    """Parse an easy-search TSV written with ``LEAKAGE_FORMAT`` (every reported
    alignment, whatever its coverage). Coverages are computed from the
    alignment start/end, unrounded."""
    cols = LEAKAGE_FORMAT.split(",")
    hits = []
    with tsv.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = dict(zip(cols, line.rstrip("\n").split("\t")))
            qlen, tlen = int(r["qlen"]), int(r["tlen"])
            hits.append(
                LeakageHit(
                    train_id=r["query"],
                    test_id=r["target"],
                    pident=float(r["pident"]),
                    evalue=float(r["evalue"]),
                    bits=float(r["bits"]),
                    train_cov=(int(r["qend"]) - int(r["qstart"]) + 1) / qlen,
                    test_cov=(int(r["tend"]) - int(r["tstart"]) + 1) / tlen,
                    train_len=qlen,
                    test_len=tlen,
                )
            )
    return hits


def is_leak(hit: LeakageHit) -> bool:
    """The leakage criterion on one reported alignment. Identity >= 30 % is
    enforced by mmseqs (``--min-seq-id``); test coverage is checked here."""
    return hit.test_cov >= LEAKAGE_MIN_TEST_COV


def find_test_coverage_leaks(
    train_fa: Path, test_fa: Path, out_tsv: Path, tmp: Path, log: Path
) -> list[LeakageHit]:
    """Every training/test pair with >= 30 % identity over >= 80 % of the test
    chain (one row per pair mmseqs reports). ``out_tsv`` keeps every alignment
    mmseqs reported, including those below the coverage threshold."""
    run(leakage_search_cmd(train_fa, test_fa, out_tsv, tmp), log)
    return [h for h in parse_leakage_hits(out_tsv) if is_leak(h)]
