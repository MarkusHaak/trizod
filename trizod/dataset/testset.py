#!/usr/bin/env python3
"""Construct the TriZOD test set from the current strict-tier sequences.

Reproduces the original TriZOD test-set recipe (Senoner & Heinzinger, 2024) on
the *current* dataset snapshot, with a FIXED SEED so it is reproducible (the
original 2024 draw was unseeded and could not be regenerated):

  1. Cluster the strict unique sequences together with all CheZOD sequences
     (CheZOD117 + CheZOD1325) at 30% id / 80% cov.
  2. Keep the clusters that contain NO CheZOD sequence (CheZOD-free) — this is
     what makes the resulting test set disjoint from CheZOD.
  3. Randomly sample 25% of those CheZOD-free clusters (seed=SEED).
  4. Recluster the member sequences of the sampled clusters at 50% id / 80%
     cov; the representatives are the TriZOD test set.

CheZOD117 (the external published benchmark) is NOT rebuilt here — only the
in-distribution TriZOD test set is. The result is written to
``<work-dir>/testset/`` and consumed by ``trizod.dataset.redundancy`` as the
TriZOD-test leakage target.

Outputs (``<work-dir>/testset/``):
  TriZOD_test_set.fasta      one record per 50/80 representative
  TriZOD_test_set_clu.tsv    50/80 cluster membership (representative, member)
  build_test_set_summary.json

Usage
-----
    uv run python -m trizod.dataset.testset [--work-dir DIR] [--root DIR]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from trizod.dataset.mmseqs import COMMON, cluster_tsv_groups, run
from trizod.dataset.paths import resolve_paths
from trizod.io.fasta import read_fasta, write_fasta

SEED = 42
SAMPLE_FRACTION = 0.25
CHEZOD_PREFIX = "CHEZOD__"


def _entry_sort_key(entry_id: str):
    """Deterministic key so the lowest-numbered entry ID sharing a sequence is
    chosen. Splits on '_'; numeric parts sort before non-numeric parts."""
    key = []
    for part in entry_id.split("_"):
        key.append((0, int(part), "") if part.isdigit() else (1, 0, part))
    return tuple(key)


def resolve_pinned_testset(
    pinned: dict[str, str], strict: dict[str, str]
) -> tuple[dict[str, str], dict]:
    """Resolve pinned test sequences against the current strict-tier pool.

    Each pinned sequence maps to its pinned entry ID if that entry is still in
    the strict pool, else to the lowest-numbered current entry sharing the
    identical sequence. A pinned sequence absent from the pool is dropped.
    """
    seq_to_ids: dict[str, list[str]] = {}
    for eid, seq in strict.items():
        seq_to_ids.setdefault(seq, []).append(eid)
    for seq in seq_to_ids:
        seq_to_ids[seq].sort(key=_entry_sort_key)

    test_recs: dict[str, str] = {}
    dropped: list[str] = []
    substitutions: list[list[str]] = []
    for pid, pseq in pinned.items():
        ids = seq_to_ids.get(pseq)
        if not ids:
            dropped.append(pid)
            continue
        chosen = pid if pid in ids else ids[0]
        if chosen != pid:
            substitutions.append([pid, chosen])
        test_recs[chosen] = pseq

    info = {
        "pinned_total": len(pinned),
        "resolved": len(test_recs),
        "dropped": dropped,
        "substitutions": substitutions,
    }
    return test_recs, info


def chezod1325_records(path: Path) -> dict[str, str]:
    """Parse allseqs1325.txt ('<BMRB_ID> <sequence>' per line) into id->seq."""
    recs: dict[str, str] = {}
    for line in path.open():
        parts = line.split()
        if len(parts) >= 2:
            recs[parts[0]] = parts[1]
    return recs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
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
    strict_fasta = paths.final_dataset / "strict" / "strict.fasta"
    out = paths.testset
    tmp = out / "_tmp"

    if tmp.exists():
        shutil.rmtree(tmp)
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    strict = read_fasta(strict_fasta)
    chezod = read_fasta(paths.chezod117)
    chezod.update(chezod1325_records(paths.chezod1325_txt))
    print(f"strict unique seqs: {len(strict)}; CheZOD seqs (117+1325): {len(chezod)}")

    # ---- Step 1: cluster strict + CheZOD @30/80 ----
    step1_in = tmp / "strict_plus_chezod.fasta"
    with step1_in.open("w") as fh:
        for rid, seq in strict.items():
            fh.write(f">{rid}\n{seq}\n")
        for rid, seq in chezod.items():
            fh.write(f">{CHEZOD_PREFIX}{rid}\n{seq}\n")
    pref1 = tmp / "c30"
    run(
        [
            "mmseqs",
            "easy-cluster",
            str(step1_in),
            str(pref1),
            str(tmp / "w30"),
            "--min-seq-id",
            "0.3",
            "-c",
            "0.8",
            *COMMON,
        ],
        out / "build_test_set.log",
    )
    groups30 = cluster_tsv_groups(Path(str(pref1) + "_cluster.tsv"))

    # ---- Step 2: CheZOD-free clusters + their strict members ----
    free_clusters: dict[str, list[str]] = {}
    n_chezod_touch = 0
    for rep, members in groups30.items():
        if any(m.startswith(CHEZOD_PREFIX) for m in members):
            n_chezod_touch += 1
            continue
        strict_members = [m for m in members if not m.startswith(CHEZOD_PREFIX)]
        if strict_members:
            free_clusters[rep] = strict_members
    print(
        f"30/80 clusters: {len(groups30)} "
        f"({n_chezod_touch} CheZOD-touching, {len(free_clusters)} CheZOD-free)"
    )

    # ---- Step 3: random 25% of CheZOD-free clusters (seeded) ----
    rng = random.Random(SEED)
    free_reps = sorted(free_clusters)
    n_sample = round(len(free_reps) * SAMPLE_FRACTION)
    # Keep rng.sample's deterministic (seeded) list order. Wrapping it in a set
    # would make iteration order depend on PYTHONHASHSEED, changing the FASTA
    # written below and hence the order-sensitive mmseqs 50/80 representative
    # pick — defeating the fixed-seed reproducibility this module promises.
    # sampling is already without replacement, so no dedup is needed.
    sampled = rng.sample(free_reps, n_sample)
    sampled_members = [m for rep in sampled for m in free_clusters[rep]]
    print(
        f"sampled {n_sample} / {len(free_reps)} CheZOD-free clusters "
        f"({len(sampled_members)} member sequences) [seed={SEED}]"
    )

    # ---- Step 4: recluster sampled members @50/80 -> test reps ----
    step4_in = tmp / "sampled_members.fasta"
    write_fasta({m: strict[m] for m in sampled_members}, step4_in)
    pref2 = tmp / "c50"
    run(
        [
            "mmseqs",
            "easy-cluster",
            str(step4_in),
            str(pref2),
            str(tmp / "w50"),
            "--min-seq-id",
            "0.5",
            "-c",
            "0.8",
            *COMMON,
        ],
        out / "build_test_set.log",
        append=True,
    )
    rep_fasta = Path(str(pref2) + "_rep_seq.fasta")
    clu_tsv = Path(str(pref2) + "_cluster.tsv")

    test_recs = read_fasta(rep_fasta)
    out_fasta = out / "TriZOD_test_set.fasta"
    write_fasta(test_recs, out_fasta)
    shutil.copy2(clu_tsv, out / "TriZOD_test_set_clu.tsv")

    n_members = sum(1 for ln in clu_tsv.open() if ln.strip())
    summary = {
        "seed": SEED,
        "sample_fraction": SAMPLE_FRACTION,
        "strict_unique_seqs": len(strict),
        "chezod_seqs": len(chezod),
        "clusters_30_80": len(groups30),
        "chezod_touching_clusters": n_chezod_touch,
        "chezod_free_clusters": len(free_reps),
        "sampled_clusters": n_sample,
        "sampled_member_seqs": len(sampled_members),
        "test_set_size": len(test_recs),
        "test_set_clu_members": n_members,
    }
    (out / "build_test_set_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"\nTriZOD test set: {len(test_recs)} representatives "
        f"({n_members} members) -> {out_fasta}"
    )
    print(f"Summary: {out / 'build_test_set_summary.json'}")


if __name__ == "__main__":
    main()
