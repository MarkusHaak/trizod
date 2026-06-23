#!/usr/bin/env python3
"""Run the mmseqs2 redundancy-reduction pipeline (Python wrapper).

Implements, in pure Python via ``subprocess``, the same pipeline described
in the original TriZOD report (Senoner & Heinzinger, 2024):

  1. (Already done in 2024-11, FROZEN): cluster strict @ 30/80, sample
     25% of the CheZOD-free clusters, recluster at 50% to derive the
     TriZOD test set of 348 proteins. Test sets are inputs to this
     script. CheZOD1325 was used at this step (to exclude its clusters
     from the random sample) but is NOT a training-leakage target.
  2. Test-set leakage removal of every tier's deduplicated FASTA against
     the combined test set (CheZOD117 + TriZOD-348) at 30% id / 80% cov,
     in TWO stages (matching the manuscript redundancy-reduction figure):
       2a. stage-1 "remove all cluster members": cluster the unfiltered
           superset together with the test sequences at 30/80 and drop
           every training sequence that shares a cluster with a test
           sequence (catches transitive leakage A~B~test that a direct
           search misses).
       2b. stage-2 "search & remove": easy-search the training sequences
           against the test sets at 30/80 and drop any direct hit.
     Default mmseqs options: --alignment-mode 3 --cov-mode 0 -s 7.5
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
OUT = ROOT / "docs" / "260520" / "data" / "mmseqs"
TMP = OUT / "_tmp"

# CheZOD117 is the fixed external benchmark; the TriZOD test set is rebuilt
# from the current snapshot by build_test_set.py (run it first).
CHEZOD = ROOT / "data" / "2024-05-09" / "CheZOD117_test_set.fasta"
TRIZOD_TEST = ROOT / "docs" / "260520" / "data" / "testset" / "TriZOD_test_set.fasta"
TIERS = ["strict", "moderate", "tolerant", "unfiltered"]

# Default mmseqs option block from the report.
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

# Header prefix used to tag test sequences when they are co-clustered with the
# training superset in stage-1, so they never collide with a training header.
TEST_PREFIX = "TESTSET__"


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


def fasta_ids(fa: Path) -> list[str]:
    return [ln[1:].split()[0] for ln in open(fa) if ln.startswith(">")]


def filter_fasta_by_ids(fa_in: Path, remove_ids: set[str], fa_out: Path) -> int:
    """Copy fa_in to fa_out, dropping records whose first header token is in
    remove_ids. Returns the number of records kept."""
    kept = 0
    with open(fa_in) as f, open(fa_out, "w") as o:
        write_record = False
        for line in f:
            if line.startswith(">"):
                first_tok = line[1:].split()[0]
                write_record = first_tok not in remove_ids
                if write_record:
                    o.write(line)
                    kept += 1
            elif write_record:
                o.write(line)
    return kept


def build_combined_testset(sources: list[Path], out: Path) -> dict[str, str]:
    """Concatenate the test-set FASTAs into `out`, de-duplicating by header
    token and asserting no two test sets share a sequence ID. Returns the
    {test_id: source_name} map of every retained test sequence."""
    seen: dict[str, str] = {}
    with open(out, "w") as o:
        for src in sources:
            write_record = False
            for line in open(src):
                if line.startswith(">"):
                    tok = line[1:].split()[0]
                    if tok in seen:
                        raise SystemExit(
                            f"test-set ID collision: {tok} in both "
                            f"{seen[tok]} and {src.name}"
                        )
                    seen[tok] = src.name
                    write_record = True
                    o.write(line)
                elif write_record:
                    o.write(line)
    return seen


def stage1_cluster_member_removal(
    super_fa: Path, combined_test: Path, tmp: Path, log: Path
) -> set[str]:
    """Stage-1 "remove all cluster members".

    Cluster the unfiltered training superset together with the test
    sequences at 30% id / 80% cov and return the set of training IDs that
    land in a cluster which also contains a test sequence. Test headers are
    tagged with TEST_PREFIX so they cannot collide with a training header,
    and so a cluster's test-contact is detectable from the membership alone.
    """
    test_ids = set(fasta_ids(combined_test))
    stage1_in = OUT / "stage1_cluster_input.fasta"
    with open(stage1_in, "w") as o:
        for line in open(super_fa):  # training superset, own headers
            o.write(line)
        for line in open(combined_test):  # test seqs, TESTSET__-prefixed
            o.write((">" + TEST_PREFIX + line[1:]) if line.startswith(">") else line)

    prefix = tmp / "stage1"
    run(
        [
            "mmseqs",
            "easy-cluster",
            str(stage1_in),
            str(prefix),
            str(tmp / "stage1_work"),
            "--min-seq-id",
            "0.3",
            "-c",
            "0.8",
            *COMMON,
        ],
        log,
    )

    # easy-cluster writes <prefix>_cluster.tsv with columns (representative,
    # member); every member (incl. the representative itself) appears once.
    clu_tsv = Path(str(prefix) + "_cluster.tsv")
    members_by_repr: dict[str, list[str]] = {}
    with open(clu_tsv) as f:
        for line in f:
            if not line.strip():
                continue
            rep, mem = line.rstrip("\n").split("\t")[:2]
            members_by_repr.setdefault(rep, []).append(mem)

    leaked: set[str] = set()
    n_test_clusters = 0
    for rep, members in members_by_repr.items():
        group = set(members) | {rep}
        if any(m.startswith(TEST_PREFIX) for m in group):
            n_test_clusters += 1
            leaked |= {m for m in group if not m.startswith(TEST_PREFIX)}
    print(
        f"  stage-1: {len(test_ids)} test seqs; {n_test_clusters} test-touching "
        f"clusters; {len(leaked)} training IDs co-clustered with a test seq"
    )
    return leaked


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # mmseqs refuses to overwrite existing DB outputs, so start from a clean
    # scratch dir on every run (makes the script idempotent / re-runnable).
    if TMP.exists():
        shutil.rmtree(TMP)
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
    test_map = build_combined_testset([CHEZOD, TRIZOD_TEST], combined)
    print(f"Combined test set: {combined} ({len(test_map)} sequences)\n")

    # ---- Step A0: stage-1 cluster-membership removal (vs test sets) ----
    # Cluster the unfiltered superset with the test sequences once; a training
    # ID leaked here is leaked in every (nested) tier, so we apply the same
    # set to all tiers below.
    print("=== Step A0: stage-1 'remove all cluster members' @ 30/80 ===")
    unfiltered_fa = FINAL / "unfiltered" / "unfiltered.fasta"
    leaked = stage1_cluster_member_removal(
        unfiltered_fa, combined, TMP, OUT / "stage1_cluster.log"
    )
    print()

    # ---------------- Step A: stage-2 easy-search vs test sets ----------
    for tier in TIERS:
        fa = FINAL / tier / f"{tier}.fasta"
        hits = OUT / f"{tier}_testset_hits.tsv"
        out_fa = OUT / f"{tier}_no_testset.fasta"
        log = OUT / f"{tier}_easy_search.log"

        print(f"=== Step A ({tier}): stage-2 easy-search vs test sets ===")
        run(
            [
                "mmseqs",
                "easy-search",
                str(fa),
                str(combined),
                str(hits),
                str(TMP),
                "--min-seq-id",
                "0.3",
                "-c",
                "0.8",
                *COMMON,
                "--format-output",
                "query,target,pident,evalue,bits",
            ],
            log,
        )
        hit_ids = {ln.split("\t", 1)[0] for ln in open(hits) if ln.strip()}
        tier_ids = set(fasta_ids(fa))
        leaked_tier = leaked & tier_ids
        removed = leaked_tier | hit_ids
        kept = filter_fasta_by_ids(fa, removed, out_fa)
        print(
            f"  stage-1 removed: {len(leaked_tier)} "
            f"(unique to stage-1: {len(leaked_tier - hit_ids)})\n"
            f"  stage-2 hits:    {len(hit_ids)} "
            f"(unique to stage-2: {len(hit_ids - leaked_tier)})\n"
            f"  total removed:   {len(removed)}; kept after removal: {kept}\n"
        )

    # ---------------- Step B: cluster strict @ 50/80 ----------------
    print("=== Step B: cluster strict residual @ 50% / 80% ===")
    strict_fa = OUT / "strict_no_testset.fasta"
    strict_db = TMP / "strict_db"
    strict_clu = TMP / "strict_clu"
    log = OUT / "strict_cluster.log"
    run(["mmseqs", "createdb", str(strict_fa), str(strict_db)], log)
    run(
        [
            "mmseqs",
            "cluster",
            str(strict_db),
            str(strict_clu),
            str(TMP / "strict_clu_workdir"),
            "--min-seq-id",
            "0.5",
            "-c",
            "0.8",
            *COMMON,
        ],
        log,
        append=True,
    )
    run(
        [
            "mmseqs",
            "createtsv",
            str(strict_db),
            str(strict_db),
            str(strict_clu),
            str(OUT / "train_strict_clu.tsv"),
        ],
        log,
        append=True,
    )
    run(
        [
            "mmseqs",
            "result2repseq",
            str(strict_db),
            str(strict_clu),
            str(TMP / "strict_repseq"),
        ],
        log,
        append=True,
    )
    run(
        [
            "mmseqs",
            "result2flat",
            str(strict_db),
            str(strict_db),
            str(TMP / "strict_repseq"),
            str(OUT / "train_strict.fasta"),
            "--use-fasta-header",
        ],
        log,
        append=True,
    )
    print(f"  clusters: {count_unique_first_column(OUT / 'train_strict_clu.tsv')}")
    print(f"  reps:     {count_fasta(OUT / 'train_strict.fasta')}\n")

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
        run(
            [
                "mmseqs",
                "clusterupdate",
                str(prev_db),
                str(new_db),
                str(prev_clu),
                str(merged_db),
                str(new_clu),
                str(workdir),
                "--min-seq-id",
                "0.5",
                "-c",
                "0.8",
                *COMMON,
            ],
            log,
            append=True,
        )
        run(
            [
                "mmseqs",
                "createtsv",
                str(merged_db),
                str(merged_db),
                str(new_clu),
                str(OUT / f"train_{tier}_clu.tsv"),
            ],
            log,
            append=True,
        )
        run(
            [
                "mmseqs",
                "result2repseq",
                str(merged_db),
                str(new_clu),
                str(TMP / f"{tier}_repseq"),
            ],
            log,
            append=True,
        )
        run(
            [
                "mmseqs",
                "result2flat",
                str(merged_db),
                str(merged_db),
                str(TMP / f"{tier}_repseq"),
                str(OUT / f"train_{tier}.fasta"),
                "--use-fasta-header",
            ],
            log,
            append=True,
        )
        print(f"  clusters: {count_unique_first_column(OUT / f'train_{tier}_clu.tsv')}")
        print(f"  reps:     {count_fasta(OUT / f'train_{tier}.fasta')}\n")
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
