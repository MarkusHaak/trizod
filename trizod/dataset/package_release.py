#!/usr/bin/env python3
"""Assemble the TriZOD dataset release bundle for Zenodo.

Gathers the canonical training sets, per-residue score labels, frozen test
sets, and the datasheet into a single staging directory, and writes a
``MANIFEST.json`` with byte sizes, SHA-256 checksums and record/sequence
counts for every file.

Core bundle (~115 MB):
  train/<tier>/   train_<tier>_best.fasta (canonical), train_<tier>.fasta,
                  clusters_best.tsv, clusters.tsv
  scores/<tier>/  scores.json   (per-residue Z/G/k + offsets = the labels)
  test/           CheZOD117_test_set.fasta, TriZOD_test_set.fasta
  README.md       (the datasheet)
  MANIFEST.json

Optional (--include-str, ~1.4 GB): str/<tier>/  re-referenced NMR-STAR files.
Optional (~95 MB): shifts/trizod_shifts.parquet — every assigned chemical
shift of every released chain, backbone and side chain, as deposited, beside the
re-referenced value where one transfers. Staged automatically when it exists;
build it with ``scripts/build_parquet_dataset.py --shifts-out``.

Usage
-----
    uv run python -m trizod.dataset.package_release [--version VER]
        [--include-str] [--shifts-parquet PATH] [--out DIR]
        [--work-dir DIR] [--root DIR]

The output directory defaults under the work dir (gitignored). Nothing is
uploaded; this only stages files locally for a manual Zenodo deposit.

Before the manifest is written, a leakage gate checks the staged bundle and
fails the step when any training record shares an ID or an exact sequence with
a test chain, or covers >= 80 % of a test chain at >= 30 % identity (the
stage-2 criterion of ``trizod.dataset.redundancy``, run with mmseqs, which must
be on PATH); every offending pair is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from trizod import paths as repo_paths
from trizod.dataset.mmseqs import LeakageHit, find_test_coverage_leaks
from trizod.dataset.paths import resolve_paths
from trizod.io.fasta import count_fasta, read_fasta, write_fasta
from trizod.provenance import pipeline_version
from trizod.shifts import SHIFTS_PARQUET_NAME

TIERS = ["unfiltered", "tolerant", "moderate", "strict"]
DEFAULT_VERSION = "2026-07"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(path: Path) -> int:
    return sum(1 for ln in path.open() if ln.strip())


def copy_in(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def find_coverage_leaks(
    train: dict[str, str], test: dict[str, str]
) -> list[tuple[str, LeakageHit]]:
    """Training records that cover >= 80 % of a test chain at >= 30 % identity.

    ``train`` maps ``"<file>:<id>"`` labels to sequences. Each distinct
    training sequence is searched once (query) against the test sequences
    (target) with the stage-2 criterion of ``trizod.dataset.redundancy``
    (:func:`trizod.dataset.mmseqs.leakage_search_cmd`), so the gate re-checks
    exactly what stage 2 enforced, with E-values against the same test
    database. mmseqs scratch lives in a temporary directory, never in the
    bundle. Returns ``(training label, hit)`` pairs.
    """
    if not shutil.which("mmseqs"):
        raise SystemExit(
            "LEAKAGE GATE: mmseqs not found in PATH; the test-coverage check cannot run"
        )
    labels_by_seq: dict[str, list[str]] = {}
    for label, seq in train.items():
        labels_by_seq.setdefault(seq, []).append(label)
    seq_key = {seq: f"q{i}" for i, seq in enumerate(labels_by_seq)}
    labels_by_key = {seq_key[s]: labels for s, labels in labels_by_seq.items()}

    with tempfile.TemporaryDirectory(prefix="trizod_leakgate_") as td:
        work = Path(td)
        train_fa = work / "train.fasta"
        test_fa = work / "test.fasta"
        write_fasta({k: s for s, k in seq_key.items()}, train_fa)
        write_fasta(test, test_fa)
        hits = find_test_coverage_leaks(
            train_fa, test_fa, work / "hits.tsv", work / "tmp", work / "search.log"
        )
    return [
        (label, hit)
        for hit in sorted(hits, key=lambda h: (h.test_id, h.train_id))
        for label in labels_by_key[hit.train_id]
    ]


def assert_no_leakage(bundle: Path) -> None:
    """Release gate: no training sequence may share an ID or an exact
    sequence with any held-out test sequence, and no training sequence may
    cover >= 80 % of a test chain at >= 30 % identity (the stage-2 criterion,
    which also catches a longer training construct that contains a test
    protein). A shared BMRB entry ID (a different shift record of the same
    entry, not a sequence leak) is only reported, not failed."""
    test: dict[str, str] = {}
    for tf in sorted((bundle / "test").glob("*.fasta")):
        test.update(read_fasta(tf))
    test_ids = set(test)
    test_entry = {i.split("_", 1)[0] for i in test_ids}
    test_seqs = set(test.values())

    id_hits: list[tuple[str, str]] = []
    seq_hits: list[tuple[str, str]] = []
    train_records: dict[str, str] = {}
    entry_hits = 0
    for trf in sorted((bundle / "train").rglob("*.fasta")):
        rel = trf.relative_to(bundle / "train").as_posix()
        for tid, seq in read_fasta(trf).items():
            train_records[f"{rel}:{tid}"] = seq
            if tid in test_ids:
                id_hits.append((trf.name, tid))
            if seq in test_seqs:
                seq_hits.append((trf.name, tid))
            if tid.split("_", 1)[0] in test_entry:
                entry_hits += 1
    if id_hits or seq_hits:
        raise SystemExit(
            f"LEAKAGE GATE FAILED: {len(id_hits)} shared IDs + {len(seq_hits)} "
            f"exact-sequence matches between train and test, e.g. "
            f"{(id_hits + seq_hits)[:5]}"
        )

    cov_hits = find_coverage_leaks(train_records, test)
    if cov_hits:
        lines = [
            f"    {label} -> {hit.test_id}: {hit.pident:.1f} % id, "
            f"test cov {hit.test_cov:.3f}, train cov {hit.train_cov:.3f} "
            f"({hit.train_len} vs {hit.test_len} aa, E={hit.evalue:.2g})"
            for label, hit in cov_hits
        ]
        n_pairs = len({(h.train_id, h.test_id) for _, h in cov_hits})
        n_test = len({h.test_id for _, h in cov_hits})
        raise SystemExit(
            f"LEAKAGE GATE FAILED: {len(cov_hits)} training records "
            f"({n_pairs} distinct sequence pairs, {n_test} test chains) cover "
            f">= 80 % of a test chain at >= 30 % identity:\n" + "\n".join(lines)
        )
    msg = (
        f"  leakage gate: OK — 0 shared IDs, 0 exact-sequence matches & 0 "
        f"training records covering >= 80 % of a test chain at >= 30 % "
        f"identity, vs {len(test_ids)} test sequences"
    )
    if entry_hits:
        msg += (
            f"; note: {entry_hits} train records share a BMRB entry ID with a "
            f"test sequence (different shift record, not a sequence leak)"
        )
    print(msg)


def count_parquet_rows(path: Path):
    """Row count of a Parquet file, or None when pyarrow is unavailable.

    pyarrow is not a project dependency (nothing in the pipeline needs it), so
    the manifest records bytes + checksum either way and the row count only
    when the packaging run happens to have it: ``uv run --with pyarrow ...``.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    return pq.ParquetFile(path).metadata.num_rows


def record_entry(rel: str, abs_path: Path) -> dict:
    entry = {"bytes": abs_path.stat().st_size, "sha256": sha256(abs_path)}
    if abs_path.suffix == ".fasta":
        entry["n_sequences"] = count_fasta(abs_path)
    elif abs_path.name == "scores.json":
        entry["n_records"] = count_jsonl(abs_path)
    elif abs_path.suffix == ".parquet":
        n_rows = count_parquet_rows(abs_path)
        if n_rows is not None:
            entry["n_records"] = n_rows
    return {rel: entry}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument(
        "--include-str",
        action="store_true",
        help="also bundle the ~1.4 GB re-referenced .str files",
    )
    ap.add_argument(
        "--shifts-parquet",
        type=Path,
        default=None,
        help=(
            "chemical-shift table to stage under shifts/ "
            "(default: <root>/data/processed/" + SHIFTS_PARQUET_NAME + " if it exists)"
        ),
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="dataset build dir (default: <root>/data/interim/build)",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repository root (default: auto-detected)",
    )
    args = ap.parse_args(argv)

    paths = resolve_paths(args.work_dir, args.root)
    out = args.out or paths.release_bundle

    bundle = out / f"trizod-dataset-{args.version}"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    planned: list[tuple[Path, str]] = []  # (source, relative path in bundle)

    # Compact dataset README -> README.md
    if not paths.bundle_readme.exists():
        raise SystemExit(f"bundle README not found: {paths.bundle_readme}")
    planned.append((paths.bundle_readme, "README.md"))

    trizod_test = paths.testset / "TriZOD_test_set.fasta"
    for tier in TIERS:
        best = paths.mmseqs / f"train_{tier}_best.fasta"
        reps = paths.mmseqs / f"train_{tier}.fasta"
        clu_best = paths.mmseqs / f"train_{tier}_clu_best.tsv"
        clu = paths.mmseqs / f"train_{tier}_clu.tsv"
        scores = paths.scored / tier / "scores.json"
        for src, rel in [
            (best, f"train/{tier}/train_{tier}_best.fasta"),
            (reps, f"train/{tier}/train_{tier}.fasta"),
            (clu_best, f"train/{tier}/clusters_best.tsv"),
            (clu, f"train/{tier}/clusters.tsv"),
            (scores, f"scores/{tier}/scores.json"),
        ]:
            if not src.exists():
                raise SystemExit(f"missing required input: {src}")
            planned.append((src, rel))
        if args.include_str:
            str_dir = paths.scored / tier / "str"
            for sf in sorted(str_dir.glob("*.str")):
                planned.append((sf, f"str/{tier}/{sf.name}"))

    for src in [paths.chezod117, trizod_test]:
        if not src.exists():
            raise SystemExit(f"missing test set: {src}")
        planned.append((src, f"test/{src.name}"))

    # The label table records, per pinned test chain, the strictest tier it still
    # satisfies and any ID substituted since the pin was drawn. Without it an
    # ID-based join against an earlier release is silently wrong, so it ships
    # beside the FASTA rather than staying an intermediate.
    test_labels = paths.testset / "TriZOD_test_set_labels.tsv"
    if not test_labels.exists():
        raise SystemExit(f"missing test-set labels: {test_labels}")
    planned.append((test_labels, f"test/{test_labels.name}"))

    # Chemical-shift table: explicit path is required to exist, the default
    # location is best-effort so a release can be staged before it has been
    # built (scripts/build_parquet_dataset.py --shifts-out).
    shifts = args.shifts_parquet
    if shifts is None:
        default_shifts = repo_paths.layout(args.root).processed / SHIFTS_PARQUET_NAME
        shifts = default_shifts if default_shifts.exists() else None
    elif not shifts.exists():
        raise SystemExit(f"missing chemical-shift table: {shifts}")
    if shifts is not None:
        planned.append((shifts, f"shifts/{SHIFTS_PARQUET_NAME}"))
    else:
        print(
            "  note: no chemical-shift table staged (build it with "
            "scripts/build_parquet_dataset.py --shifts-out)"
        )

    manifest: dict[str, dict] = {}
    total = 0
    print(f"Staging {len(planned)} files into {bundle} ...")
    for src, rel in planned:
        dst = bundle / rel
        copy_in(src, dst)
        manifest.update(record_entry(rel, dst))
        total += dst.stat().st_size

    # Release gate: enforce the train/test leakage guarantee on the staged
    # bundle before it can be deposited.
    assert_no_leakage(bundle)

    summary = {
        "version": args.version,
        "pipeline_version": pipeline_version(),
        "rereference_mode": "both",
        "canonical_training_fasta": "train/<tier>/train_<tier>_best.fasta",
        "tiers": {
            t: {
                "scored_records": manifest[f"scores/{t}/scores.json"]["n_records"],
                "training_reps": manifest[f"train/{t}/train_{t}_best.fasta"][
                    "n_sequences"
                ],
            }
            for t in TIERS
        },
        "test_sets": {
            "CheZOD117": manifest["test/CheZOD117_test_set.fasta"]["n_sequences"],
            "TriZOD_test": manifest["test/TriZOD_test_set.fasta"]["n_sequences"],
        },
        "total_bytes": total,
        "n_files": len(planned),
        "files": dict(sorted(manifest.items())),
    }
    if shifts is not None:
        rel = f"shifts/{SHIFTS_PARQUET_NAME}"
        summary["chemical_shifts"] = {
            "path": rel,
            "n_records": manifest[rel].get("n_records"),
            # stated in the manifest as well as the datasheet: val_ppm is always
            # the deposited value, and val_corrected_ppm is null wherever no
            # trustworthy offset transfers rather than a copy of it
            "raw_column": "val_ppm",
            "rereferenced_column": "val_corrected_ppm",
            "rereferenced_nuclei": "backbone (all), side-chain 13C",
            "raw_only_nuclei": "side-chain 1H, side-chain 15N",
        }
    (bundle / "MANIFEST.json").write_text(json.dumps(summary, indent=2))

    print(f"\nBundle: {bundle}")
    print(f"  files: {len(planned)} (+ MANIFEST.json)")
    print(f"  size:  {total / 1e6:.1f} MB")
    for t in TIERS:
        print(
            f"  {t:>10}: {summary['tiers'][t]['scored_records']:>6} scored, "
            f"{summary['tiers'][t]['training_reps']:>5} training reps"
        )
    print(
        f"  test: CheZOD117={summary['test_sets']['CheZOD117']}, "
        f"TriZOD={summary['test_sets']['TriZOD_test']}"
    )
    if shifts is not None:
        n_shifts = summary["chemical_shifts"]["n_records"]
        print(
            f"  shifts: shifts/{SHIFTS_PARQUET_NAME} "
            f"({n_shifts if n_shifts is not None else 'unknown'} values, "
            "backbone + side chain)"
        )
    print("\nNothing uploaded. Review the bundle, then deposit to Zenodo manually.")


if __name__ == "__main__":
    main()
