#!/usr/bin/env python3
"""Construct or emit the TriZOD test set.

DEFAULT mode: emit the committed pin. Loads the pinned test set from
``trizod/dataset/pinned/TriZOD_test_set.fasta`` and resolves it against the
*current* TOLERANT-tier pool (``trizod.dataset.testset.resolve_pinned_testset``,
no mmseqs involved) so entry IDs stay valid as the dataset snapshot evolves.

Why tolerant and not strict: the test set is a set of *sequences*, and its
defining property — CheZOD-disjointness at 30 % id / 80 % cov — belongs to the
sequences and is invariant under any filter change. "Drawn from the strict
tier" describes the 2026-07 construction, not an ongoing invariant, and the
per-residue labels come from ``scores.json`` either way. Resolving against
strict silently shrinks the published set whenever a rescore moves a chain
across the offset thresholds (15 of the 365 as of the post-#20 rescore), which
breaks comparability with the released v0.3.0 numbers. Each resolved chain
therefore carries a ``label_tier`` — the strictest tier its sequence still
satisfies — so consumers can restrict to the strict subset if they want it.

``--redraw`` mode (requires ``--confirm-redraw``): reproduces the original
TriZOD test-set recipe (Senoner & Heinzinger, 2024) on the *current* dataset
snapshot, with a FIXED SEED so it is reproducible (the original 2024 draw was
unseeded and could not be regenerated), and OVERWRITES the committed pin with
the result:

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
  TriZOD_test_set.fasta      one record per representative (pinned or redrawn)
  TriZOD_test_set_labels.tsv per chain: current ID, pinned (v0.3.0) ID,
                             whether the ID was substituted, and label_tier —
                             this file *is* the released-ID mapping table
  build_test_set_summary.json
  (``--redraw`` additionally writes TriZOD_test_set_clu.tsv: 50/80 cluster
  membership, and overwrites the committed pin + its provenance sidecar)

Usage
-----
    uv run python -m trizod.dataset.testset [--work-dir DIR] [--root DIR]
    uv run python -m trizod.dataset.testset --redraw --confirm-redraw ...
"""

from __future__ import annotations

import argparse
import datetime
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

# Strictest first: the first tier whose pool holds a sequence is its label_tier.
TIER_ORDER = ("strict", "moderate", "tolerant", "unfiltered")
# Tier the pin is resolved against — see the module docstring for why.
RESOLVE_TIER = "tolerant"
UNKNOWN_TIER = "unknown"

REDRAW_WARNING = """\
!! --redraw DESTROYS COMPARABILITY WITH EVERY PUBLISHED TriZOD RELEASE !!
The seeded draw is not stable under a pool change: rng.sample() depends on the
number of CheZOD-free clusters, which moves with every rescore or filter
change. A redraw therefore yields a DIFFERENT test set, invalidating the
published test numbers, anything trained against the current split, and every
DisProt/ROC figure in the manuscript. The default (pinned) mode is what the
release chain uses; --redraw exists only to establish a NEW pin on purpose."""


def _entry_sort_key(entry_id: str):
    """Deterministic key so the lowest-numbered entry ID sharing a sequence is
    chosen. Splits on '_'; numeric parts sort before non-numeric parts."""
    key = []
    for part in entry_id.split("_"):
        key.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(key)


def load_tier_pools(final_dataset: Path, tiers=TIER_ORDER) -> dict[str, dict[str, str]]:
    """Read the per-tier deduplicated pools, strictest first.

    Returns {tier: {entry_id: sequence}} for the ``<tier>/<tier>.fasta`` files
    that exist; tiers whose FASTA is missing are skipped (they only affect
    ``label_tier`` granularity, so a partial build still resolves).
    """
    pools: dict[str, dict[str, str]] = {}
    for tier in tiers:
        fa = final_dataset / tier / f"{tier}.fasta"
        if fa.exists():
            pools[tier] = read_fasta(fa)
    return pools


def _label_tier(seq: str, tier_seqs: dict[str, set]) -> str:
    """Strictest tier (per ``TIER_ORDER``) whose pool contains ``seq``."""
    for tier, seqs in tier_seqs.items():
        if seq in seqs:
            return tier
    return UNKNOWN_TIER


def resolve_pinned_testset(
    pinned: dict[str, str],
    pool: dict[str, str],
    tier_pools: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, str], dict]:
    """Resolve pinned test sequences against the current ``pool`` (tolerant).

    Each pinned sequence maps to its pinned entry ID if that entry is still in
    the pool, else to the lowest-numbered current entry sharing the identical
    sequence. A pinned sequence absent from the pool is dropped.

    ``tier_pools`` ({tier: {id: seq}}, strictest first — see
    :func:`load_tier_pools`) supplies each resolved chain's ``label_tier``: the
    strictest tier its sequence still satisfies. It never affects membership.

    ``info["id_map"]`` is the full pinned-ID -> current-ID table (one row per
    resolved chain, ``substituted`` flagging the rows where the ID moved), so a
    holder of an older release can repair ID-based joins. The pin matches by
    *sequence*, so a pool-side representative change would otherwise rename a
    test chain silently.
    """
    seq_to_ids: dict[str, list[str]] = {}
    for eid, seq in pool.items():
        seq_to_ids.setdefault(seq, []).append(eid)
    # Sort by TIER_ORDER so the label is the strictest satisfied tier whatever
    # order the caller assembled the pools in; unknown names rank last.
    tier_seqs = {
        t: set((tier_pools or {})[t].values())
        for t in sorted(
            tier_pools or {},
            key=lambda t: TIER_ORDER.index(t) if t in TIER_ORDER else len(TIER_ORDER),
        )
    }

    test_recs: dict[str, str] = {}
    dropped: list[str] = []
    substitutions: list[list[str]] = []
    label_tiers: dict[str, str] = {}
    id_map: list[dict] = []
    for pid, pseq in pinned.items():
        ids = seq_to_ids.get(pseq)
        if not ids:
            dropped.append(pid)
            continue
        chosen = pid if pid in ids else min(ids, key=_entry_sort_key)
        if chosen != pid:
            substitutions.append([pid, chosen])
        if chosen in test_recs:
            raise ValueError(
                f"two pinned sequences resolved to the same entry {chosen}; "
                f"the pin file must have unique sequences"
            )
        test_recs[chosen] = pseq
        label_tiers[chosen] = _label_tier(pseq, tier_seqs)
        id_map.append(
            {
                "pinned_id": pid,
                "test_id": chosen,
                "substituted": chosen != pid,
                "label_tier": label_tiers[chosen],
            }
        )

    counts: dict[str, int] = {}
    for tier in [*tier_seqs, UNKNOWN_TIER]:
        n = sum(1 for t in label_tiers.values() if t == tier)
        if n:
            counts[tier] = n

    info = {
        "pinned_total": len(pinned),
        "resolved": len(test_recs),
        "dropped": dropped,
        "substitutions": substitutions,
        "label_tiers": label_tiers,
        "label_tier_counts": counts,
        "id_map": id_map,
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


def _easy_cluster(in_fa, pref, work, min_seq_id, log, *, append=False) -> None:
    """Run ``mmseqs easy-cluster`` at ``min_seq_id`` identity and 80% coverage."""
    run(
        [
            "mmseqs",
            "easy-cluster",
            str(in_fa),
            str(pref),
            str(work),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            "0.8",
            *COMMON,
        ],
        log,
        append=append,
    )


def _redraw(paths, out) -> dict[str, str]:
    """Seeded 30/80 -> sample -> 50/80 recipe. Writes the fasta/clu/summary and
    returns {entry_id: sequence} for the redrawn test set."""
    strict_fasta = paths.final_dataset / "strict" / "strict.fasta"
    tmp = out / "_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    strict = read_fasta(strict_fasta)
    chezod = read_fasta(paths.chezod117)
    chezod.update(chezod1325_records(paths.chezod1325_txt))
    print(f"strict unique seqs: {len(strict)}; CheZOD seqs (117+1325): {len(chezod)}")

    # ---- Step 1: cluster strict + CheZOD @30/80 ----
    step1_in = tmp / "strict_plus_chezod.fasta"
    step1_recs = {
        **strict,
        **{f"{CHEZOD_PREFIX}{rid}": seq for rid, seq in chezod.items()},
    }
    write_fasta(step1_recs, step1_in)
    pref1 = tmp / "c30"
    _easy_cluster(step1_in, pref1, tmp / "w30", 0.3, out / "build_test_set.log")
    groups30 = cluster_tsv_groups(Path(str(pref1) + "_cluster.tsv"))

    # ---- Step 2: CheZOD-free clusters + their strict members ----
    free_clusters: dict[str, list[str]] = {}
    n_chezod_touch = 0
    for rep, members in groups30.items():
        # Past this guard no member carries the CheZOD prefix, so every member
        # of the cluster is a strict sequence.
        if any(m.startswith(CHEZOD_PREFIX) for m in members):
            n_chezod_touch += 1
            continue
        free_clusters[rep] = members
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
    _easy_cluster(
        step4_in, pref2, tmp / "w50", 0.5, out / "build_test_set.log", append=True
    )
    rep_fasta = Path(str(pref2) + "_rep_seq.fasta")
    clu_tsv = Path(str(pref2) + "_cluster.tsv")

    test_recs = read_fasta(rep_fasta)
    out_fasta = out / "TriZOD_test_set.fasta"
    write_fasta(test_recs, out_fasta)
    shutil.copy2(clu_tsv, out / "TriZOD_test_set_clu.tsv")

    with clu_tsv.open() as fh:
        n_members = sum(1 for ln in fh if ln.strip())
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

    return test_recs


def _write_pin(pin_path, test_recs: dict[str, str]) -> None:
    """Overwrite the committed pin FASTA + provenance sidecar."""
    pin_path.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(test_recs, pin_path)
    prov = {
        "count": len(test_recs),
        "seed": SEED,
        "sample_fraction": SAMPLE_FRACTION,
        "written": datetime.date.today().isoformat(),
        "mode": "redraw",
    }
    pin_path.with_suffix(".provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n"
    )


def _write_testset(out: Path, test_recs: dict[str, str], info: dict) -> None:
    """Emit the resolved test set: annotated FASTA + the label/ID-map TSV.

    The FASTA header carries ``label_tier`` and the pinned (previous-release)
    ID so the sequence file is self-describing; every consumer in the chain
    keys on the first header token, as the per-tier pool FASTAs already do.
    """
    label_tiers = info["label_tiers"]
    pinned_of = {row["test_id"]: row["pinned_id"] for row in info["id_map"]}
    with (out / "TriZOD_test_set.fasta").open("w") as fh:
        for rid, seq in test_recs.items():
            fh.write(
                f">{rid} label_tier={label_tiers[rid]} "
                f"pinned_id={pinned_of[rid]}\n{seq}\n"
            )
    with (out / "TriZOD_test_set_labels.tsv").open("w") as fh:
        fh.write("test_id\tpinned_id\tsubstituted\tlabel_tier\tlength\n")
        for row in info["id_map"]:
            fh.write(
                f"{row['test_id']}\t{row['pinned_id']}\t{row['substituted']}\t"
                f"{row['label_tier']}\t{len(test_recs[row['test_id']])}\n"
            )


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
    ap.add_argument(
        "--redraw",
        action="store_true",
        help="Redraw the seeded test set and OVERWRITE the committed pin "
        "(default: emit the committed pin). DESTRUCTIVE — also requires "
        "--confirm-redraw.",
    )
    ap.add_argument(
        "--confirm-redraw",
        action="store_true",
        help="Acknowledge that --redraw invalidates every published test "
        "number. Without it, --redraw refuses to run.",
    )
    ap.add_argument(
        "--resolve-tier",
        choices=TIER_ORDER,
        default=RESOLVE_TIER,
        help=f"pool the pinned sequences are resolved against "
        f"(default: {RESOLVE_TIER}; see the module docstring)",
    )
    args = ap.parse_args(argv)
    paths = resolve_paths(args.work_dir, args.root)
    out = paths.testset
    out.mkdir(parents=True, exist_ok=True)

    if args.redraw:
        if not args.confirm_redraw:
            raise SystemExit(
                f"{REDRAW_WARNING}\n\nRefusing to redraw. Re-run with "
                f"--confirm-redraw if you really mean to break comparability."
            )
        print(REDRAW_WARNING)
        # A labels sidecar from an earlier pinned run describes the OLD set.
        (out / "TriZOD_test_set_labels.tsv").unlink(missing_ok=True)
        test_recs = _redraw(paths, out)
        _write_pin(paths.pinned_testset, test_recs)
        print(f"Re-pinned {len(test_recs)} sequences -> {paths.pinned_testset}")
        return

    if not paths.pinned_testset.exists():
        raise SystemExit(
            f"Pinned test set not found at {paths.pinned_testset}. "
            f"Run `trizod dataset test-set --redraw` to establish it."
        )

    tier_pools = load_tier_pools(paths.final_dataset)
    pool = tier_pools.get(args.resolve_tier)
    if pool is None:
        raise SystemExit(
            f"{args.resolve_tier} pool not found at "
            f"{paths.final_dataset / args.resolve_tier / f'{args.resolve_tier}.fasta'}."
            f" Run `trizod dataset build` first."
        )
    pinned = read_fasta(paths.pinned_testset)
    test_recs, info = resolve_pinned_testset(pinned, pool, tier_pools=tier_pools)

    _write_testset(out, test_recs, info)
    summary = {"mode": "pinned", "resolve_tier": args.resolve_tier, **info}
    (out / "build_test_set_summary.json").write_text(json.dumps(summary, indent=2))

    if info["dropped"]:
        print(
            f"WARNING: {len(info['dropped'])} pinned sequences are no longer in "
            f"the {args.resolve_tier} pool and were dropped: {info['dropped']}"
        )
    if info["substitutions"]:
        print(
            f"WARNING: {len(info['substitutions'])} pinned representatives were "
            f"substituted by the lowest-numbered entry with the same sequence. "
            f"The sequence is unchanged but the ID is NOT the released one — "
            f"ID-based joins against an older release must go through "
            f"TriZOD_test_set_labels.tsv: {info['substitutions']}"
        )
    non_strict = {t: n for t, n in info["label_tier_counts"].items() if t != "strict"}
    if non_strict:
        print(
            f"NOTE: {sum(non_strict.values())} test chains no longer meet the "
            f"strict criteria (label_tier: {non_strict}); they are retained so "
            f"the published test set stays comparable across releases."
        )
    print(
        f"TriZOD test set (pinned, resolved against {args.resolve_tier}): "
        f"{len(test_recs)} sequences -> {out / 'TriZOD_test_set.fasta'}"
    )


if __name__ == "__main__":
    main()
