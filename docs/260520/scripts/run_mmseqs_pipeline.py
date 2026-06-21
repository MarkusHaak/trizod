#!/usr/bin/env python3
"""Run the mmseqs2 redundancy-reduction pipeline (Python wrapper).

Implements, in pure Python via ``subprocess``, the same pipeline described
in the original TriZOD report (Senoner & Heinzinger, 2024):

  1. (Already done in 2024-11): cluster strict @ 30/80, sample 25% of
     non-CheZOD clusters, recluster at 50% to derive the TriZOD test set
     of 348 proteins. Test sets are inputs to this script.
  2. easy-search each tier's deduplicated FASTA against the combined
     test set at 30% identity / 80% coverage and drop any hit. Default
     mmseqs options: --alignment-mode 3 --cov-mode 0 -s 7.5
     --comp-bias-corr 0 --mask 0.
  3. Cluster the strict residual at 50% / 80% to seed the training
     representatives.
  4. clusterupdate to add moderate → tolerant → unfiltered, in this
     order, so the most-stringent member becomes the cluster
     representative for every cluster.

Outputs go to docs/260520/data/mmseqs/.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FINAL = ROOT / "docs" / "260520" / "data" / "final_dataset"
TESTSET_DIR = ROOT / "data" / "2024-05-09"
OUT = ROOT / "docs" / "260520" / "data" / "mmseqs"
TMP = OUT / "_tmp"

CHEZOD = TESTSET_DIR / "CheZOD117_test_set.fasta"
TRIZOD_TEST = TESTSET_DIR / "TriZOD_test_set.fasta"
TIERS = ["strict", "moderate", "tolerant", "unfiltered"]

# Default mmseqs option block from the report.
COMMON = [
    "--alignment-mode", "3",
    "--cov-mode", "0",
    "-s", "7.5",
    "--comp-bias-corr", "0",
    "--mask", "0",
]


def count_fasta(fa: Path) -> int:
    return sum(1 for ln in open(fa) if ln.startswith(">"))


def count_unique_first_column(tsv: Path) -> int:
    seen: set[str] = set()
    with open(tsv) as f:
        for line in f:
            if line.strip():
                seen.add(line.split("\t", 1)[0])
    return len(seen)


def run(cmd: list[str], log: Path, append: bool = False) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    mode = "ab" if append else "wb"
    with open(log, mode) as h:
        subprocess.run(cmd, check=True, stdout=h, stderr=subprocess.STDOUT)


def filter_fasta(fa_in: Path, hits_tsv: Path, fa_out: Path) -> int:
    hit_ids: set[str] = set()
    with open(hits_tsv) as h:
        for line in h:
            if line.strip():
                hit_ids.add(line.split("\t", 1)[0])
    kept = 0
    with open(fa_in) as f, open(fa_out, "w") as o:
        write_record = False
        for line in f:
            if line.startswith(">"):
                header = line[1:].rstrip()
                first_tok = header.split()[0]
                write_record = first_tok not in hit_ids
                if write_record:
                    o.write(line)
                    kept += 1
            elif write_record:
                o.write(line)
    return kept


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    if not shutil.which("mmseqs"):
        raise SystemExit("mmseqs binary not found in PATH")

    print("Inputs:")
    print(f"  CheZOD117:    {CHEZOD}  ({count_fasta(CHEZOD)} sequences)")
    print(f"  TriZOD test:  {TRIZOD_TEST}  ({count_fasta(TRIZOD_TEST)} sequences)")
    for tier in TIERS:
        fa = FINAL / tier / f"{tier}.fasta"
        print(f"  {tier:>10}: {fa}  ({count_fasta(fa)} sequences)")
    print()

    combined = OUT / "combined_testset.fasta"
    with open(combined, "wb") as o:
        for src in (CHEZOD, TRIZOD_TEST):
            o.write(open(src, "rb").read())
    print(f"Combined test set: {combined} ({count_fasta(combined)} sequences)\n")

    # ---------------- Step A: easy-search vs test sets ----------------
    for tier in TIERS:
        fa = FINAL / tier / f"{tier}.fasta"
        hits = OUT / f"{tier}_testset_hits.tsv"
        out_fa = OUT / f"{tier}_no_testset.fasta"
        log = OUT / f"{tier}_easy_search.log"

        print(f"=== Step A ({tier}): mmseqs easy-search vs test sets ===")
        run([
            "mmseqs", "easy-search",
            str(fa), str(combined), str(hits), str(TMP),
            "--min-seq-id", "0.3", "-c", "0.8",
            *COMMON,
            "--format-output", "query,target,pident,evalue,bits",
        ], log)
        kept = filter_fasta(fa, hits, out_fa)
        n_hit_rows = sum(1 for _ in open(hits))
        n_hit_q = len({ln.split("\t", 1)[0] for ln in open(hits) if ln.strip()})
        print(f"  hit rows: {n_hit_rows}, distinct queries with a hit: {n_hit_q}")
        print(f"  kept after test-set removal: {kept}\n")

    # ---------------- Step B: cluster strict @ 50/80 ----------------
    print("=== Step B: cluster strict residual @ 50% / 80% ===")
    strict_fa = OUT / "strict_no_testset.fasta"
    strict_db = TMP / "strict_db"
    strict_clu = TMP / "strict_clu"
    log = OUT / "strict_cluster.log"
    run(["mmseqs", "createdb", str(strict_fa), str(strict_db)], log)
    run([
        "mmseqs", "cluster",
        str(strict_db), str(strict_clu),
        str(TMP / "strict_clu_workdir"),
        "--min-seq-id", "0.5", "-c", "0.8",
        *COMMON,
    ], log, append=True)
    run([
        "mmseqs", "createtsv",
        str(strict_db), str(strict_db), str(strict_clu),
        str(OUT / "train_strict_clu.tsv"),
    ], log, append=True)
    run([
        "mmseqs", "result2repseq",
        str(strict_db), str(strict_clu), str(TMP / "strict_repseq"),
    ], log, append=True)
    run([
        "mmseqs", "result2flat",
        str(strict_db), str(strict_db), str(TMP / "strict_repseq"),
        str(OUT / "train_strict.fasta"),
        "--use-fasta-header",
    ], log, append=True)
    print(
        f"  clusters: {count_unique_first_column(OUT / 'train_strict_clu.tsv')}"
    )
    print(
        f"  reps:     {count_fasta(OUT / 'train_strict.fasta')}\n"
    )

    # ---------------- Step C: clusterupdate ----------------
    prev_db = strict_db
    prev_clu = strict_clu
    for tier in ("moderate", "tolerant", "unfiltered"):
        print(f"=== Step C ({tier}): mmseqs clusterupdate @ 50% / 80% ===")
        fa = OUT / f"{tier}_no_testset.fasta"
        new_db = TMP / f"{tier}_db"
        new_clu = TMP / f"{tier}_clu"
        merged_db = TMP / f"{tier}_newdb"
        workdir = TMP / f"{tier}_workdir"
        log = OUT / f"{tier}_clusterupdate.log"

        run(["mmseqs", "createdb", str(fa), str(new_db)], log)
        run([
            "mmseqs", "clusterupdate",
            str(prev_db), str(new_db), str(prev_clu),
            str(merged_db), str(new_clu), str(workdir),
            "--min-seq-id", "0.5", "-c", "0.8",
            *COMMON,
        ], log, append=True)
        run([
            "mmseqs", "createtsv",
            str(merged_db), str(merged_db), str(new_clu),
            str(OUT / f"train_{tier}_clu.tsv"),
        ], log, append=True)
        run([
            "mmseqs", "result2repseq",
            str(merged_db), str(new_clu),
            str(TMP / f"{tier}_repseq"),
        ], log, append=True)
        run([
            "mmseqs", "result2flat",
            str(merged_db), str(merged_db),
            str(TMP / f"{tier}_repseq"),
            str(OUT / f"train_{tier}.fasta"),
            "--use-fasta-header",
        ], log, append=True)
        print(
            f"  clusters: {count_unique_first_column(OUT / f'train_{tier}_clu.tsv')}"
        )
        print(
            f"  reps:     {count_fasta(OUT / f'train_{tier}.fasta')}\n"
        )
        prev_db = merged_db
        prev_clu = new_clu

    # ---------------- Summary ----------------
    print("=== Pipeline complete ===")
    print(f"{'tier':<12}{'input':>8}{'after_test':>12}{'clusters':>10}{'reps':>8}")
    for tier in TIERS:
        in_fa = FINAL / tier / f"{tier}.fasta"
        nt = OUT / f"{tier}_no_testset.fasta"
        ct = OUT / f"train_{tier}_clu.tsv"
        rt = OUT / f"train_{tier}.fasta"
        print(
            f"{tier:<12}"
            f"{count_fasta(in_fa):>8}"
            f"{count_fasta(nt):>12}"
            f"{count_unique_first_column(ct):>10}"
            f"{count_fasta(rt):>8}"
        )
    print(f"\nAll artefacts under {OUT}")


if __name__ == "__main__":
    main()
