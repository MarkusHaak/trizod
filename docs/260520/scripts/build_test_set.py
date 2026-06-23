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
docs/260520/data/testset/ and consumed by run_mmseqs_pipeline.py as the
TriZOD-test leakage target.

Outputs (docs/260520/data/testset/):
  TriZOD_test_set.fasta      one record per 50/80 representative
  TriZOD_test_set_clu.tsv    50/80 cluster membership (representative, member)
  build_test_set_summary.json
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STRICT = ROOT / "docs" / "260520" / "data" / "final_dataset" / "strict" / "strict.fasta"
CHEZOD117 = ROOT / "data" / "2024-05-09" / "CheZOD117_test_set.fasta"
CHEZOD1325_TXT = ROOT / "data" / "chezod" / "protein_nmr_1325" / "allseqs1325.txt"
OUT = ROOT / "docs" / "260520" / "data" / "testset"
TMP = OUT / "_tmp"

SEED = 42
SAMPLE_FRACTION = 0.25
CHEZOD_PREFIX = "CHEZOD__"

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
    print("  $", " ".join(str(c) for c in cmd))
    mode = "ab" if append else "wb"
    with open(log, mode) as h:
        subprocess.run(cmd, check=True, stdout=h, stderr=subprocess.STDOUT)


def read_fasta(path: Path) -> dict[str, str]:
    recs: dict[str, str] = {}
    cur: str | None = None
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


def chezod1325_records() -> dict[str, str]:
    """Parse allseqs1325.txt ('<BMRB_ID> <sequence>' per line) into id->seq."""
    recs: dict[str, str] = {}
    for line in CHEZOD1325_TXT.open():
        parts = line.split()
        if len(parts) >= 2:
            recs[parts[0]] = parts[1]
    return recs


def write_fasta(recs: dict[str, str], path: Path, prefix: str = "") -> None:
    with path.open("w") as fh:
        for rid, seq in recs.items():
            fh.write(f">{prefix}{rid}\n{seq}\n")


def cluster_tsv_groups(tsv: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    with tsv.open() as f:
        for line in f:
            if line.strip():
                rep, mem = line.rstrip("\n").split("\t")[:2]
                groups.setdefault(rep, []).append(mem)
    return groups


def main() -> None:
    if TMP.exists():
        shutil.rmtree(TMP)
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    strict = read_fasta(STRICT)
    chezod = read_fasta(CHEZOD117)
    chezod.update(chezod1325_records())
    print(f"strict unique seqs: {len(strict)}; CheZOD seqs (117+1325): {len(chezod)}")

    # ---- Step 1: cluster strict + CheZOD @30/80 ----
    step1_in = TMP / "strict_plus_chezod.fasta"
    with step1_in.open("w") as fh:
        for rid, seq in strict.items():
            fh.write(f">{rid}\n{seq}\n")
        for rid, seq in chezod.items():
            fh.write(f">{CHEZOD_PREFIX}{rid}\n{seq}\n")
    pref1 = TMP / "c30"
    run(
        [
            "mmseqs",
            "easy-cluster",
            str(step1_in),
            str(pref1),
            str(TMP / "w30"),
            "--min-seq-id",
            "0.3",
            "-c",
            "0.8",
            *COMMON,
        ],
        OUT / "build_test_set.log",
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
    sampled = set(rng.sample(free_reps, n_sample))
    sampled_members = [m for rep in sampled for m in free_clusters[rep]]
    print(
        f"sampled {n_sample} / {len(free_reps)} CheZOD-free clusters "
        f"({len(sampled_members)} member sequences) [seed={SEED}]"
    )

    # ---- Step 4: recluster sampled members @50/80 -> test reps ----
    step4_in = TMP / "sampled_members.fasta"
    write_fasta({m: strict[m] for m in sampled_members}, step4_in)
    pref2 = TMP / "c50"
    run(
        [
            "mmseqs",
            "easy-cluster",
            str(step4_in),
            str(pref2),
            str(TMP / "w50"),
            "--min-seq-id",
            "0.5",
            "-c",
            "0.8",
            *COMMON,
        ],
        OUT / "build_test_set.log",
        append=True,
    )
    rep_fasta = Path(str(pref2) + "_rep_seq.fasta")
    clu_tsv = Path(str(pref2) + "_cluster.tsv")

    test_recs = read_fasta(rep_fasta)
    out_fasta = OUT / "TriZOD_test_set.fasta"
    write_fasta(test_recs, out_fasta)
    shutil.copy2(clu_tsv, OUT / "TriZOD_test_set_clu.tsv")

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
    (OUT / "build_test_set_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"\nTriZOD test set: {len(test_recs)} representatives "
        f"({n_members} members) -> {out_fasta}"
    )
    print(f"Summary: {OUT / 'build_test_set_summary.json'}")


if __name__ == "__main__":
    main()
