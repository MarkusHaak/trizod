"""FASTA read/write helpers, unified from the dataset-build scripts.

All four historical copies (build_test_set, package_release, run_mmseqs_pipeline,
build_deploy_fasta) parsed headers identically: a record's ID is the first
whitespace-delimited token of its header line, and its sequence is the
concatenation of the stripped body lines. These functions preserve that exactly.

``read_fasta`` returns an {id: seq} dict (duplicate IDs collapse, last wins);
``fasta_ids`` returns IDs in file order *with* duplicates. The leakage/dedup
math in the dataset pipeline depends on both behaviours, so keep them distinct.
"""

from pathlib import Path


def read_fasta(path: Path) -> dict[str, str]:
    """Parse a FASTA file into an {id: sequence} dict (first-token IDs)."""
    recs: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for line in path.open():
        if line.startswith(">"):
            if cur is not None:
                recs[cur] = "".join(buf)
            cur = line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if cur is not None:
        recs[cur] = "".join(buf)
    return recs


def write_fasta(recs: dict[str, str], path: Path, prefix: str = "") -> None:
    """Write an {id: sequence} dict to FASTA, optionally prefixing each ID."""
    with path.open("w") as fh:
        for rid, seq in recs.items():
            fh.write(f">{prefix}{rid}\n{seq}\n")


def count_fasta(path: Path) -> int:
    """Count records (header lines) in a FASTA file."""
    return sum(1 for ln in path.open() if ln.startswith(">"))


def fasta_ids(path: Path) -> list[str]:
    """Record IDs (first header token) in file order, preserving duplicates."""
    return [ln[1:].split()[0] for ln in path.open() if ln.startswith(">")]
