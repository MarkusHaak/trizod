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

Usage
-----
    uv run python -m trizod.dataset.package_release [--version VER]
        [--include-str] [--out DIR] [--work-dir DIR] [--root DIR]

The output directory defaults under the work dir (gitignored). Nothing is
uploaded; this only stages files locally for a manual Zenodo deposit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from trizod.dataset.paths import resolve_paths
from trizod.io.fasta import count_fasta, read_fasta

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


def assert_no_leakage(bundle: Path) -> None:
    """Release gate: no training sequence may share an ID or an exact
    sequence with any held-out test sequence. A shared BMRB entry ID (a
    different shift record of the same entry, not a sequence leak) is only
    reported, not failed."""
    test: dict[str, str] = {}
    for tf in sorted((bundle / "test").glob("*.fasta")):
        test.update(read_fasta(tf))
    test_ids = set(test)
    test_entry = {i.split("_", 1)[0] for i in test_ids}
    test_seqs = set(test.values())

    id_hits: list[tuple[str, str]] = []
    seq_hits: list[tuple[str, str]] = []
    entry_hits = 0
    for trf in sorted((bundle / "train").rglob("*.fasta")):
        for tid, seq in read_fasta(trf).items():
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
    msg = (
        f"  leakage gate: OK — 0 shared IDs & 0 exact-sequence matches "
        f"vs {len(test_ids)} test sequences"
    )
    if entry_hits:
        msg += (
            f"; note: {entry_hits} train records share a BMRB entry ID with a "
            f"test sequence (different shift record, not a sequence leak)"
        )
    print(msg)


def record_entry(rel: str, abs_path: Path) -> dict:
    entry = {"bytes": abs_path.stat().st_size, "sha256": sha256(abs_path)}
    if abs_path.suffix == ".fasta":
        entry["n_sequences"] = count_fasta(abs_path)
    elif abs_path.name == "scores.json":
        entry["n_records"] = count_jsonl(abs_path)
    return {rel: entry}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument(
        "--include-str",
        action="store_true",
        help="also bundle the ~1.4 GB re-referenced .str files",
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
        "pipeline_version": "trizod-2026-07-14",
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
    print("\nNothing uploaded. Review the bundle, then deposit to Zenodo manually.")


if __name__ == "__main__":
    main()
