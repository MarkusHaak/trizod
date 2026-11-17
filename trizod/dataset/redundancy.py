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
     the combined test set (CheZOD117 as released, the 117 ORIGINAL CheZOD117
     sequences as SETH published them -- two of which, "hmbdi" and "6078", are
     no release chain, and none of which is identical to its release chain
     (release audit E23) -- the TriZOD test set and the RETIRED TriZOD test
     chains, ``testset/TriZOD_test_retired.fasta``: pinned test chains that
     left the strict tier or were retired for a CheZOD117 overlap, as a
     reviewed shared construct (``trizod/dataset/data/test_retirements.tsv``)
     or as a within-test duplicate, no longer tested on but still kept out of
     training, so training is what it would be with the full pin), in TWO
     stages
     (matching the manuscript redundancy-reduction figure):
       2a. stage-1 "remove all cluster members": cluster the unfiltered
           superset together with the test sequences at 30% id / 80%
           coverage of BOTH sequences (--cov-mode 0) and drop every
           training sequence that shares a cluster with a test sequence
           (catches transitive leakage A~B~test that a direct search
           misses). mmseqs' greedy set cover breaks ties by input order, so
           the input is canonical: the training chains in sequence-ID order,
           then every leakage target, the files merged and sorted by ID
           (``build_combined_testset``, ``write_stage1_input``). Which file
           a target comes from -- test, retired, CheZOD117 -- and the order of
           the files no longer change the removed set (re-audit 2026-11-17b
           dry run: moving four test chains to the retired file flipped
           6514_1_1_1 and eight other chains).
       2b. stage-2 "search & remove": easy-search the training sequences
           against the test sets IN BOTH DIRECTIONS (training as query, and
           test as query; release audit E22) -- once, on the unfiltered
           superset, whose hits restricted to a tier's IDs are that tier's
           hits (the tiers nest, and the criterion is a bit score, which
           depends on neither the database size nor the direction) -- and
           drop every TRAINING chain with an alignment of >= 30% identity and
           >= 27 bits over >= 80% of the TEST chain (``--leakage-rule test``)
           or of the test OR the training chain (``--leakage-rule either``,
           the default, i.e. of the shorter chain; decision D2). Any reported
           alignment counts, the best one or one of up to 10 alternative
           ones (``--alt-ali 10``; release re-audit 2026-11-17b: a repeat or
           tandem construct can meet the rule on its second alignment only,
           e.g. 51068_1_1_1 ~ test 51080_1_1_1), at any identity: the search
           runs with ``--min-seq-id 0`` and the 30 % is applied to each
           alignment, because mmseqs computes no alternative behind a best
           alignment below ``--min-seq-id`` (release validation 2026-11-17c:
           36490_1_1_1 and 50596_1_1_1). A test chain
           is never removed here. Coverage is the aligned span, gaps included
           (--cov-mode 1 -c 0 with the 80 % applied to the parsed spans, see
           ``trizod.dataset.mmseqs.leakage_search_cmd`` / ``is_leak``). This
           is the leakage criterion: a longer training construct that
           contains a test protein, or a pair whose length ratio is below
           0.8 in either direction, is dropped under both rules; a training
           chain contained in a longer test chain only under "either". The
           rule is recorded in
           ``mmseqs/leakage_rule.json``, and the package step's gate checks
           the same one. Stage 1 keeps --cov-mode 0 on purpose: one-sided
           coverage in a clustering lets a long multi-domain construct pull
           fragments of its other domains into the same cluster (single-
           linkage bridging), which would drop chains unrelated to any test
           chain; the direct containment cases are covered by stage 2.
     Shared mmseqs options: --alignment-mode 3 -s 7.5 --comp-bias-corr 0
     --mask 0, with --cov-mode 0 everywhere except the stage-2 search.
  3. Cluster the strict residual at 50% / 80% to seed the training
     representatives.
  4. clusterupdate to add moderate → tolerant → unfiltered, in this
     order, so the most-stringent member becomes the cluster
     representative for every cluster.

Outputs go to ``<work-dir>/mmseqs/``.

Usage
-----
    uv run python -m trizod.dataset.redundancy [--work-dir DIR] [--root DIR]
        [--leakage-rule {test,either}]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from trizod.dataset.mmseqs import (
    COMMON,
    DEFAULT_LEAKAGE_RULE,
    LEAKAGE_RULE_FILE,
    LEAKAGE_RULES,
    LEAKAGE_RULES_HELP,
    check_rule,
    cluster_tsv_groups,
    describe_rule,
    find_leaks,
    reverse_hits_path,
    run,
    write_leakage_rule,
)
from trizod.dataset.paths import resolve_paths
from trizod.dataset.testset import RETIRED_FASTA
from trizod.io.fasta import count_fasta, fasta_ids, read_fasta

TIERS = ["strict", "moderate", "tolerant", "unfiltered"]

#: Per-tier stage-2 search logs of the old layout (one search per tier). Stage 2
#: now searches the unfiltered superset once, so these are only ever left over
#: from an earlier run and would read as the record of the current one.
STALE_SEARCH_LOGS = tuple(f"{t}_easy_search.log" for t in TIERS if t != "unfiltered")


def remove_stale_outputs(out: Path) -> None:
    """Delete the outputs of an earlier run that this run no longer writes."""
    for name in STALE_SEARCH_LOGS:
        (out / name).unlink(missing_ok=True)


# Header prefix used to tag test sequences when they are co-clustered with the
# training superset in stage-1, so they never collide with a training header.
TEST_PREFIX = "TESTSET__"


def count_unique_first_column(tsv: Path) -> int:
    seen: set[str] = set()
    with open(tsv) as f:
        for line in f:
            if line.strip():
                seen.add(line.split("\t", 1)[0])
    return len(seen)


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


def filter_tsv_by_query(
    tsv_in: Path, keep_ids: set[str], tsv_out: Path, column: int = 0
) -> None:
    """Copy the rows of an mmseqs hit TSV whose ``column`` (default the query)
    is in ``keep_ids``, in their original order."""
    with open(tsv_in) as f, open(tsv_out, "w") as o:
        for line in f:
            if line.strip() and line.split("\t")[column] in keep_ids:
                o.write(line)


def leakage_target_sources(
    chezod: Path, testset_dir: Path, chezod_original: Path | None = None
) -> list[Path]:
    """The FASTAs every training chain is checked against: CheZOD117 as
    released, the original CheZOD117 sequences (``chezod_original``, when
    given), the TriZOD test set and, when the test-set step wrote one, the
    retired TriZOD test chains (``trizod.dataset.testset.RETIRED_FASTA``)."""
    sources = [chezod]
    if chezod_original is not None:
        sources.append(chezod_original)
    sources.append(testset_dir / "TriZOD_test_set.fasta")
    retired = testset_dir / RETIRED_FASTA
    if retired.exists():
        sources.append(retired)
    return sources


def canonical_id_key(rec_id: str) -> tuple:
    """Natural sort key of a sequence ID (``9_1_1_1`` < ``100_1_1_1``; a name
    such as the original CheZOD117 ``hmbdi`` sorts after the numbers of its
    position), never comparing an int with a str."""
    return tuple(
        (0, int(p), "") if p.isdigit() else (1, 0, p) for p in rec_id.split("_")
    )


def build_combined_testset(sources: list[Path], out: Path) -> dict[str, str]:
    """Merge the leakage-target FASTAs into `out`, every record (header line
    kept) in :func:`canonical_id_key` order, so the file depends on the
    records alone, not on which source holds which or on the order of the
    sources; no two sources may share a sequence ID. Returns the
    {test_id: source_name} map of every target."""
    seen: dict[str, str] = {}
    records: dict[str, list[str]] = {}
    for src in sources:
        lines = None
        for line in open(src):
            if line.startswith(">"):
                tok = line[1:].split()[0]
                if tok in seen:
                    raise SystemExit(
                        f"test-set ID collision: {tok} in both "
                        f"{seen[tok]} and {src.name}"
                    )
                seen[tok] = src.name
                lines = records[tok] = [line if line.endswith("\n") else line + "\n"]
            elif lines is not None:
                lines.append(line if line.endswith("\n") else line + "\n")
    with open(out, "w") as o:
        for tok in sorted(records, key=canonical_id_key):
            o.writelines(records[tok])
    return seen


def write_stage1_input(super_fa: Path, combined_test: Path, out: Path) -> Path:
    """The stage-1 clustering input: every training chain of ``super_fa``, then
    every target of ``combined_test`` tagged with ``TEST_PREFIX``, each block
    in :func:`canonical_id_key` order and one ``>ID`` line per record, so the
    input -- and with it mmseqs' tie-breaking -- depends on the records alone,
    not on the order or grouping of the files they came from."""
    train = read_fasta(super_fa)
    test = read_fasta(combined_test)
    with open(out, "w") as o:
        for rid in sorted(train, key=canonical_id_key):
            o.write(f">{rid}\n{train[rid]}\n")
        for rid in sorted(test, key=canonical_id_key):
            o.write(f">{TEST_PREFIX}{rid}\n{test[rid]}\n")
    return out


def stage1_cluster_member_removal(
    super_fa: Path, combined_test: Path, tmp: Path, out: Path, log: Path
) -> set[str]:
    """Stage-1 "remove all cluster members".

    Cluster the unfiltered training superset together with the test
    sequences at 30% id / 80% cov and return the set of training IDs that
    land in a cluster which also contains a test sequence. Test headers are
    tagged with TEST_PREFIX so they cannot collide with a training header,
    and so a cluster's test-contact is detectable from the membership alone.
    """
    test_ids = set(fasta_ids(combined_test))
    stage1_in = write_stage1_input(
        super_fa, combined_test, out / "stage1_cluster_input.fasta"
    )

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
    members_by_repr = cluster_tsv_groups(clu_tsv)

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


def stage2_search_removal(
    train_fa: Path,
    combined_test: Path,
    hits: Path,
    tmp: Path,
    log: Path,
    rule: str = DEFAULT_LEAKAGE_RULE,
) -> set[str]:
    """Stage-2 "search & remove".

    Return the training IDs with an alignment to any test chain of >= 30 %
    identity and >= ``LEAKAGE_MIN_BITS`` bits over >= 80 % of the TEST chain
    (rule ``"test"``) or of the test or the training chain (rule
    ``"either"``), found by either search direction (training as query, test
    as query) -- the best alignment of the pair or any alternative one
    (``trizod.dataset.mmseqs.LEAKAGE_ALT_ALI``). The searches run without a
    coverage threshold inside mmseqs
    (which would skip pairs by length ratio) and keep alignments whose span
    covers >= 80 % of the test chain, so a training chain that contains a
    test protein is caught however long it is, and a shorter one whose gapped
    alignment spans the test chain is caught too. Under ``"either"`` an
    alignment that covers >= 80 % of the training chain also counts, which
    catches a training fragment of a longer test chain. Only training IDs are
    returned: the training partner of a hit is removed, never the test chain.
    """
    return {
        h.train_id
        for h in find_leaks(train_fa, combined_test, hits, tmp, log, rule=rule)
    }


def main(argv=None):
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
    ap.add_argument(
        "--leakage-rule",
        choices=LEAKAGE_RULES,
        default=DEFAULT_LEAKAGE_RULE,
        help=(
            "stage-2 leakage rule ("
            + LEAKAGE_RULES_HELP.replace("%", "%%")
            + "; default: %(default)s)"
        ),
    )
    args = ap.parse_args(argv)
    rule = check_rule(args.leakage_rule)
    paths = resolve_paths(args.work_dir, args.root)
    final = paths.final_dataset
    out = paths.mmseqs
    tmp = out / "_tmp"
    chezod = paths.chezod117
    chezod_original = paths.chezod117_original
    trizod_test = paths.testset / "TriZOD_test_set.fasta"

    out.mkdir(parents=True, exist_ok=True)
    # The rule record is written only once the run has finished, so a failed
    # or interrupted run never leaves a record that vouches for stale outputs.
    (out / LEAKAGE_RULE_FILE).unlink(missing_ok=True)
    remove_stale_outputs(out)
    # mmseqs refuses to overwrite existing DB outputs, so start from a clean
    # scratch dir on every run (makes the script idempotent / re-runnable).
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    if not shutil.which("mmseqs"):
        raise SystemExit("mmseqs binary not found in PATH")

    print(f"Leakage rule: {rule} ({describe_rule(rule)})")
    print("Inputs:")
    if not chezod_original.exists():
        raise SystemExit(
            f"original CheZOD117 sequences not found at {chezod_original}: they "
            "are leakage targets (release audit E23)"
        )
    print(f"  CheZOD117:    {chezod}  ({count_fasta(chezod)} sequences)")
    print(
        f"  CheZOD117 (original): {chezod_original}  "
        f"({count_fasta(chezod_original)} sequences)"
    )
    print(f"  TriZOD test:  {trizod_test}  ({count_fasta(trizod_test)} sequences)")
    sources = leakage_target_sources(chezod, paths.testset, chezod_original)
    for src in sources[3:]:
        print(f"  retired:      {src}  ({count_fasta(src)} sequences, leakage only)")
    for tier in TIERS:
        fa = final / tier / f"{tier}.fasta"
        print(f"  {tier:>10}: {fa}  ({count_fasta(fa)} sequences)")
    print()

    combined = out / "combined_testset.fasta"
    test_map = build_combined_testset(sources, combined)
    print(f"Combined test set: {combined} ({len(test_map)} sequences)\n")

    # ---- Step A0: stage-1 cluster-membership removal (vs test sets) ----
    # Cluster the unfiltered superset with the test sequences once; a training
    # ID leaked here is leaked in every (nested) tier, so we apply the same
    # set to all tiers below.
    print("=== Step A0: stage-1 'remove all cluster members' @ 30/80 ===")
    unfiltered_fa = final / "unfiltered" / "unfiltered.fasta"
    leaked = stage1_cluster_member_removal(
        unfiltered_fa, combined, tmp, out, out / "stage1_cluster.log"
    )
    print()

    # ---------------- Step A: stage-2 easy-search vs test sets ----------
    # One search per direction, of the unfiltered superset. Every tier FASTA is
    # a subset of it -- same IDs, same sequences (build names each sequence by
    # its global representative in every tier) -- every training x test pair is
    # aligned independently (no prefilter), and the criterion is a bit score,
    # which no database size enters: a tier's hits are the superset's hits
    # restricted to the tier's IDs, which is what a per-tier search would
    # report.
    print(f"=== Step A: stage-2 easy-search vs test sets ({describe_rule(rule)}) ===")
    unfiltered_hits = out / "unfiltered_testset_hits.tsv"
    all_hit_ids = stage2_search_removal(
        unfiltered_fa,
        combined,
        unfiltered_hits,
        tmp,
        out / "unfiltered_easy_search.log",
        rule=rule,
    )
    superset = read_fasta(unfiltered_fa)
    print()
    for tier in TIERS:
        fa = final / tier / f"{tier}.fasta"
        out_fa = out / f"{tier}_no_testset.fasta"
        print(f"=== Step A ({tier}): stage-2 hits within the tier ===")
        tier_seqs = read_fasta(fa)
        not_nested = [i for i, q in tier_seqs.items() if superset.get(i) != q]
        if not_nested:
            raise SystemExit(
                f"{tier}.fasta is not a subset of unfiltered.fasta: "
                f"{len(not_nested)} IDs absent or with another sequence, e.g. "
                f"{not_nested[:5]}"
            )
        tier_ids = set(tier_seqs)
        if tier != "unfiltered":
            filter_tsv_by_query(
                unfiltered_hits, tier_ids, out / f"{tier}_testset_hits.tsv"
            )
            # the reverse search's training chain is its target (column 2)
            filter_tsv_by_query(
                reverse_hits_path(unfiltered_hits),
                tier_ids,
                reverse_hits_path(out / f"{tier}_testset_hits.tsv"),
                column=1,
            )
        hit_ids = all_hit_ids & tier_ids
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
    strict_fa = out / "strict_no_testset.fasta"
    strict_db = tmp / "strict_db"
    strict_clu = tmp / "strict_clu"
    log = out / "strict_cluster.log"
    run(["mmseqs", "createdb", str(strict_fa), str(strict_db)], log)
    run(
        [
            "mmseqs",
            "cluster",
            str(strict_db),
            str(strict_clu),
            str(tmp / "strict_clu_workdir"),
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
            str(out / "train_strict_clu.tsv"),
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
            str(tmp / "strict_repseq"),
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
            str(tmp / "strict_repseq"),
            str(out / "train_strict.fasta"),
            "--use-fasta-header",
        ],
        log,
        append=True,
    )
    print(f"  clusters: {count_unique_first_column(out / 'train_strict_clu.tsv')}")
    print(f"  reps:     {count_fasta(out / 'train_strict.fasta')}\n")

    # ---------------- Step C: clusterupdate ----------------
    prev_db = strict_db
    prev_clu = strict_clu
    for tier in ("moderate", "tolerant", "unfiltered"):
        print(f"=== Step C ({tier}): mmseqs clusterupdate @ 50% / 80% ===")
        fa = out / f"{tier}_no_testset.fasta"
        new_db = tmp / f"{tier}_db"
        new_clu = tmp / f"{tier}_clu"
        merged_db = tmp / f"{tier}_newdb"
        workdir = tmp / f"{tier}_workdir"
        log = out / f"{tier}_clusterupdate.log"

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
                str(out / f"train_{tier}_clu.tsv"),
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
                str(tmp / f"{tier}_repseq"),
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
                str(tmp / f"{tier}_repseq"),
                str(out / f"train_{tier}.fasta"),
                "--use-fasta-header",
            ],
            log,
            append=True,
        )
        print(f"  clusters: {count_unique_first_column(out / f'train_{tier}_clu.tsv')}")
        print(f"  reps:     {count_fasta(out / f'train_{tier}.fasta')}\n")
        prev_db = merged_db
        prev_clu = new_clu

    # ---------------- Summary ----------------
    print("=== Pipeline complete ===")
    print(f"{'tier':<12}{'input':>8}{'after_test':>12}{'clusters':>10}{'reps':>8}")
    for tier in TIERS:
        in_fa = final / tier / f"{tier}.fasta"
        nt = out / f"{tier}_no_testset.fasta"
        ct = out / f"train_{tier}_clu.tsv"
        rt = out / f"train_{tier}.fasta"
        print(
            f"{tier:<12}"
            f"{count_fasta(in_fa):>8}"
            f"{count_fasta(nt):>12}"
            f"{count_unique_first_column(ct):>10}"
            f"{count_fasta(rt):>8}"
        )
    record = write_leakage_rule(out, rule)
    print(f"\nLeakage rule '{rule}' recorded in {record}")
    print(f"All artefacts under {out}")


if __name__ == "__main__":
    main()
