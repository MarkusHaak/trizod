"""Shared mmseqs2 helpers for the dataset-build chain.

The ``COMMON`` option block is used verbatim by every clustering step and by the
test-set draw (``--alignment-mode 3 --cov-mode 0 -s 7.5 --comp-bias-corr 0
--mask 0``); do not change it without rebuilding + re-verifying the whole funnel.

Test-set leakage is defined separately, by :func:`leakage_search_cmd`: a
training chain leaks when an alignment to a held-out test chain reaches
>= 30 % identity over >= 80 % of the TEST chain. ``--cov-mode 0`` would demand
80 % of both chains, so a long training construct that contains a test protein
(or a pair whose length ratio is below 0.8) would pass. The leakage search
therefore keeps the ``COMMON`` block but swaps in ``--cov-mode 1`` (coverage of
the target) with query = training chains and target = test chains. The same
command backs the stage-2 removal in :mod:`trizod.dataset.redundancy` and the
release gate in :mod:`trizod.dataset.package_release`, so both apply one
criterion with E-values computed against the same (test) target database.

``--cov-mode 1`` makes mmseqs skip a pair before alignment only when the
training chain is shorter than 80 % of the test chain (it could not cover 80 %
of it anyway); longer training chains are always aligned, whatever the length
ratio.
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
LEAKAGE_FORMAT = "query,target,pident,evalue,bits,qcov,tcov,qlen,tlen"


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
    train_cov: float  # fraction of the training chain aligned
    test_cov: float  # fraction of the test chain aligned
    train_len: int
    test_len: int


def leakage_search_cmd(
    train_fa: Path, test_fa: Path, out_tsv: Path, tmp: Path
) -> list[str]:
    """``mmseqs easy-search`` for the test-set leakage criterion.

    Query = training chains, target = test chains, ``--cov-mode 1 -c 0.8``
    (>= 80 % of the TEST chain aligned) and ``--min-seq-id 0.3``; every other
    option is the shared ``COMMON`` block.
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
        str(LEAKAGE_MIN_TEST_COV),
        *with_cov_mode(COMMON, LEAKAGE_COV_MODE),
        "--format-output",
        LEAKAGE_FORMAT,
    ]


def parse_leakage_hits(tsv: Path) -> list[LeakageHit]:
    """Parse an easy-search TSV written with ``LEAKAGE_FORMAT``."""
    hits = []
    with tsv.open() as f:
        for line in f:
            if not line.strip():
                continue
            q, t, pident, evalue, bits, qcov, tcov, qlen, tlen = line.rstrip(
                "\n"
            ).split("\t")[:9]
            hits.append(
                LeakageHit(
                    train_id=q,
                    test_id=t,
                    pident=float(pident),
                    evalue=float(evalue),
                    bits=float(bits),
                    train_cov=float(qcov),
                    test_cov=float(tcov),
                    train_len=int(qlen),
                    test_len=int(tlen),
                )
            )
    return hits


def find_test_coverage_leaks(
    train_fa: Path, test_fa: Path, out_tsv: Path, tmp: Path, log: Path
) -> list[LeakageHit]:
    """Every training/test pair with >= 30 % identity over >= 80 % of the test
    chain (one row per pair mmseqs reports)."""
    run(leakage_search_cmd(train_fa, test_fa, out_tsv, tmp), log)
    return parse_leakage_hits(out_tsv)
