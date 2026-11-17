"""Per-chain overlap with the CheZOD sets (release audit E23, 2026-11-17).

The release Parquet flags every chain whose sequence meets the leakage
criterion of :mod:`trizod.dataset.mmseqs` -- >= 30 % identity over >= 80 % of
either chain, >= ``LEAKAGE_MIN_BITS`` bits, either search direction -- against
a sequence of

* ``chezod117``: CheZOD117 as released (``data/external/chezod117``) plus the
  117 original sequences SETH published (two of which, "hmbdi" and "6078",
  are no release chain);
* ``chezod1325``: the 1,325 CheZOD sequences (``allseqs1325.txt``);
* ``chezod1174``: CheZOD1174, the SETH training set.

Training chains never overlap CheZOD117 (it is a leakage target); the
CheZOD1325 and CheZOD1174 flags let a benchmark of SETH, ODiNPred or ADOPT use
the test chains clean of the predictor's own training data.
"""

from __future__ import annotations

from pathlib import Path

from trizod.dataset import mmseqs as mm
from trizod.dataset.paths import resolve_paths
from trizod.io.fasta import read_fasta, write_fasta

CHEZOD_SETS = ("chezod117", "chezod1325", "chezod1174")


def overlap_column(name: str) -> str:
    """The Parquet column of a CheZOD set: ``<name>_overlap``."""
    return f"{name}_overlap"


def _chezod1325(path: Path) -> dict[str, str]:
    """``allseqs1325.txt``: ``<BMRB ID> <sequence>`` per line."""
    recs: dict[str, str] = {}
    for line in Path(path).open():
        parts = line.split()
        if len(parts) >= 2:
            recs[parts[0]] = parts[1]
    return recs


def load_chezod_sets(root=None) -> dict[str, dict[str, str]]:
    """``{set name: {id: sequence}}`` for ``CHEZOD_SETS``, read from the
    repository ``root``'s ``data/external``. A missing file is an error."""
    paths = resolve_paths(root=root)
    files = {
        "chezod117": (paths.chezod117, paths.chezod117_original),
        "chezod1325": (paths.chezod1325_txt,),
        "chezod1174": (paths.chezod1174,),
    }
    missing = [str(p) for ps in files.values() for p in ps if not Path(p).exists()]
    if missing:
        raise SystemExit(f"CheZOD sequence files not found: {missing}")
    return {
        "chezod117": {
            **read_fasta(paths.chezod117),
            **read_fasta(paths.chezod117_original),
        },
        "chezod1325": _chezod1325(paths.chezod1325_txt),
        "chezod1174": read_fasta(paths.chezod1174),
    }


def overlapping_chains(
    chains: dict[str, str], targets: dict[str, str], work: Path
) -> set[str]:
    """IDs of the ``chains`` ({id: sequence}) whose sequence meets the leakage
    criterion (rule ``either``) against any of ``targets``. Each distinct
    sequence is searched once; chains sharing it share the verdict."""
    if not chains or not targets:
        return set()
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    by_seq: dict[str, list[str]] = {}
    for cid, seq in chains.items():
        by_seq.setdefault(seq, []).append(cid)
    key = {seq: f"q{i}" for i, seq in enumerate(by_seq)}
    ids_of = {key[seq]: ids for seq, ids in by_seq.items()}
    query_fa, target_fa = work / "chains.fasta", work / "targets.fasta"
    write_fasta({k: seq for seq, k in key.items()}, query_fa)
    write_fasta(targets, target_fa)
    hits = mm.find_leaks(
        query_fa,
        target_fa,
        work / "hits.tsv",
        work / "tmp",
        work / "search.log",
        rule="either",
    )
    return {cid for h in hits for cid in ids_of[h.train_id]}


def chezod_overlaps(
    chains: dict[str, str], sets: dict[str, dict[str, str]], work: Path
) -> dict[str, set[str]]:
    """``{set name: overlapping chain IDs}`` for every set of ``sets``."""
    return {
        name: overlapping_chains(chains, targets, Path(work) / name)
        for name, targets in sets.items()
    }
