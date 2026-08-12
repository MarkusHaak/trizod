"""Shared mmseqs2 helpers for the dataset-build chain.

The ``COMMON`` option block is used verbatim by every clustering/search step
(``--alignment-mode 3 --cov-mode 0 -s 7.5 --comp-bias-corr 0 --mask 0``); do not
change it without rebuilding + re-verifying the whole funnel.
"""

import subprocess
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
