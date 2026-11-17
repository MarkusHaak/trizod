#!/usr/bin/env python3
"""Build a single deployment FASTA (TriZOD train + test, per-residue G-score labels).

Produces a FASTA in the exact format of the SETH/ODiNPred-style
``disorder_trizod.fasta`` (Rostlab/pbc, ``trizod`` branch) so a downstream
disorder predictor can train on it directly. Each record is **two lines**::

    >{ID} SET={train|test} TARGET={g1;g2;...;gN} MASK={b1b2...bN}
    {SEQUENCE}

where

* ``TARGET`` is the per-residue TriZOD **G-score** (``gscores`` in
  ``scores.json``), 0-1 bounded, joined by ``;``. Positions without a score
  (no Z-score / rejected offset; typically termini) use the sentinel
  ``999.0``.
* ``MASK`` is a binary string the same length as the sequence: ``1`` where a
  real G-score is present, ``0`` where it is masked (``TARGET == 999.0``).
* ``len(TARGET) == len(MASK) == len(SEQUENCE)`` for every record.

Membership / split (defaults):

* ``SET=train`` -- the canonical quality-best, redundancy-reduced, CheZOD/
  TriZOD-test-leakage-free **tolerant** training set
  (``<work-dir>/mmseqs/train_tolerant_best.fasta``, 5,684 seqs).
* ``SET=test``  -- the seeded **TriZOD test set**
  (``<work-dir>/testset/TriZOD_test_set.fasta``, 344 seqs).
* ``SET=val``   -- optional. With ``--val-fraction F`` (e.g. 0.15), a uniform
  random subset of the training records (seeded by ``--seed``, default 42) is
  relabelled ``SET=val``; the rest stay ``SET=train``. The val set is carved
  *across the whole train set* (not stratified) and is disjoint from test (it
  is a subset of the already-leakage-free train). Default ``--val-fraction 0``
  emits only train + test.

Which ``scores.json`` supplies the labels: per-residue G-scores are NOT
tier-independent. Tolerant is the only tier that combines a finite
``--max-offset`` (3 sigma) with ``--reject-shift-type-only``, so it masks an
over-limit atom type out of the score (and nulls its offset) where the
unfiltered run applies the offset and scores the type; in release 2026-10 that
makes 1,073 of 15,073 tolerant records differ from the unfiltered ones, 478 of
them training representatives (audit issue I-01). Each record's labels
therefore come from the tier that selected the chain, by
:func:`trizod.dataset.tiers.label_source_tier` -- the same rule
``scripts/build_parquet_dataset.py`` applies to the Parquet's
``label_tier_source``, so the two artefacts agree chain for chain whatever
``--tier`` is (see :data:`trizod.dataset.tiers.LABEL_SOURCE_RULE`). For the
default ``--tier tolerant`` that is the tolerant record for every train and
test chain: a chain held by moderate or strict was never masked at tolerant
(moderate/strict reject an over-limit offset outright), so its moderate/strict
record is the tolerant one.

The script is self-validating: it asserts every ID resolves, that
TARGET/MASK/sequence lengths agree, that train and test are disjoint, and that
G-scores lie in [0, 1]; it aborts otherwise.

Usage
-----
    uv run python -m trizod.dataset.deploy_fasta
        [--tier tolerant] [--out PATH] [--work-dir DIR] [--root DIR]
        [--train-fasta PATH] [--test-fasta PATH]
        [--scored-dir DIR] [--reps-dir DIR]

``--scored-dir`` holds ``<tier>/scores.json`` for all four tiers (default
``<root>/data/interim/scored``) and ``--reps-dir`` the four
``train_<tier>_best.fasta`` files (default ``<work-dir>/mmseqs``); a release
bundle's ``scores/`` and ``train/`` directories work as well.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from trizod.dataset.paths import resolve_paths
from trizod.dataset.tiers import LABEL_SOURCE_RULE, TIER_ORDER, label_source_tier
from trizod.io.fasta import fasta_ids

SENTINEL = "999.0"  # masked / no-score per-residue value (matches reference)


def rep_fasta(reps_dir: Path, tier: str) -> Path:
    """``train_<tier>_best.fasta`` in the work-dir (flat ``mmseqs/``) or the
    release-bundle (``train/<tier>/``) layout."""
    flat = reps_dir / f"train_{tier}_best.fasta"
    return flat if flat.exists() else reps_dir / tier / f"train_{tier}_best.fasta"


def load_scores(path: Path) -> dict[str, dict]:
    """Load a per-tier ``scores.json`` (JSONL) into an ID -> record dict."""
    recs: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            recs[e["ID"]] = e
    return recs


def natural_key(rec_id: str) -> tuple:
    """Sort key for IDs like '15036_1_1_1' -> (15036, 1, 1, 1)."""
    return tuple(int(p) if p.isdigit() else p for p in rec_id.split("_"))


def format_target(gscores: list) -> tuple[str, str]:
    """Return (TARGET string, MASK string) from a per-residue gscore list."""
    target_vals: list[str] = []
    mask_chars: list[str] = []
    for g in gscores:
        if g is None:
            target_vals.append(SENTINEL)
            mask_chars.append("0")
        else:
            # gscores are already 4-dp in scores.json; round defensively and
            # use str() for the shortest round-trip repr (matches reference,
            # which emits 0.0 / 0.1 / 0.12 / 0.1234 rather than fixed .4f).
            target_vals.append(str(round(float(g), 4)))
            mask_chars.append("1")
    return ";".join(target_vals), "".join(mask_chars)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tier",
        default="tolerant",
        help="release tier whose training representatives are emitted as "
        "SET=train (labels follow LABEL_SOURCE_RULE, per chain)",
    )
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
        "--train-fasta",
        type=Path,
        default=None,
        help="training FASTA whose headers list the train IDs "
        "(default: <reps-dir>/train_<tier>_best.fasta)",
    )
    ap.add_argument("--test-fasta", type=Path, default=None)
    ap.add_argument(
        "--scored-dir",
        type=Path,
        default=None,
        help="directory holding <tier>/scores.json for all four tiers "
        "(default: <root>/data/interim/scored)",
    )
    ap.add_argument(
        "--reps-dir",
        type=Path,
        default=None,
        help="directory holding train_<tier>_best.fasta for all four tiers, flat "
        "or as <tier>/train_<tier>_best.fasta (default: <work-dir>/mmseqs)",
    )
    ap.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="fraction of the train set to relabel SET=val (uniform random, "
        "seeded). 0 = no val split (default). The reference uses ~0.15.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the train->val draw (reproducible)",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--min-valid",
        type=int,
        default=3,
        help="warn about records with fewer than this many scored residues",
    )
    args = ap.parse_args(argv)

    if args.tier not in TIER_ORDER:
        raise SystemExit(f"--tier must be one of {TIER_ORDER}: {args.tier!r}")
    paths = resolve_paths(args.work_dir, args.root)
    scored_dir = args.scored_dir or paths.scored
    reps_dir = args.reps_dir or paths.mmseqs
    rep_paths = {t: rep_fasta(reps_dir, t) for t in TIER_ORDER}
    score_paths = {t: scored_dir / t / "scores.json" for t in TIER_ORDER}
    train_fasta = args.train_fasta or rep_paths[args.tier]
    test_fasta = args.test_fasta or (paths.testset / "TriZOD_test_set.fasta")
    out = args.out or paths.deploy_out

    # all four tiers, always: falling back to one tier's records would silently
    # re-introduce the unfiltered-vs-tolerant label split (I-01)
    for p in (train_fasta, test_fasta, *rep_paths.values(), *score_paths.values()):
        if not p.exists():
            raise SystemExit(f"missing required input: {p}")

    if not 0.0 <= args.val_fraction < 1.0:
        raise SystemExit(f"--val-fraction must be in [0, 1): {args.val_fraction}")

    train_ids = fasta_ids(train_fasta)
    test_ids = fasta_ids(test_fasta)

    # split disjointness (a record cannot be both train and test)
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise SystemExit(
            f"train/test ID overlap ({len(overlap)}): {sorted(overlap)[:10]}"
        )

    # Carve a uniform-random validation subset out of train (seeded). Sample
    # from the sorted IDs so the draw is deterministic regardless of FASTA
    # order. val is disjoint from test (subset of the already-disjoint train).
    n_val = round(args.val_fraction * len(train_ids))
    val_ids: set[str] = set()
    if n_val > 0:
        rng = random.Random(args.seed)
        val_ids = set(rng.sample(sorted(train_ids, key=natural_key), n_val))

    scores = {t: load_scores(p) for t, p in score_paths.items()}
    reps = {t: set(fasta_ids(p)) for t, p in rep_paths.items()}

    # (ID, SET) work list; emitted sorted by natural ID.
    tagged = [(i, "val" if i in val_ids else "train") for i in train_ids]
    tagged += [(i, "test") for i in test_ids]
    tagged.sort(key=lambda t: natural_key(t[0]))

    missing: list[str] = []
    few_valid: list[tuple[str, int]] = []
    out_of_range: list[tuple[str, float]] = []
    lines: list[str] = []
    n_by_set = {"train": 0, "val": 0, "test": 0}
    source_counts = dict.fromkeys(TIER_ORDER, 0)
    total_res = 0
    total_sentinel = 0

    for rec_id, set_name in tagged:
        tier = label_source_tier(rec_id, reps, scores)
        if tier is None:
            missing.append(rec_id)
            continue
        e = scores[tier][rec_id]
        source_counts[tier] += 1
        seq = e["seq"]
        gscores = e["gscores"]
        if len(seq) != len(gscores):
            raise SystemExit(
                f"{rec_id}: seq len {len(seq)} != gscores len {len(gscores)}"
            )
        for g in gscores:
            if g is not None and not (-1e-6 <= float(g) <= 1.0 + 1e-6):
                out_of_range.append((rec_id, float(g)))
        target, mask = format_target(gscores)
        n_valid = mask.count("1")
        if n_valid < args.min_valid:
            few_valid.append((rec_id, n_valid))
        total_res += len(seq)
        total_sentinel += mask.count("0")
        n_by_set[set_name] += 1
        lines.append(f">{rec_id} SET={set_name} TARGET={target} MASK={mask}")
        lines.append(seq)

    if missing:
        raise SystemExit(
            f"{len(missing)} IDs absent from every scores.json under "
            f"{scored_dir}: {missing[:10]}"
        )
    if out_of_range:
        raise SystemExit(
            f"{len(out_of_range)} G-scores outside [0,1], e.g. {out_of_range[:5]}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    def _rel(p: Path) -> str:
        # Repo-relative for readability, but fall back to the absolute path when
        # p is outside the repo (e.g. --work-dir elsewhere) so building the
        # summary never raises ValueError after the FASTA is already written.
        return (
            str(p.relative_to(paths.root)) if p.is_relative_to(paths.root) else str(p)
        )

    summary = {
        "tier": args.tier,
        "train_fasta": _rel(train_fasta),
        "test_fasta": _rel(test_fasta),
        "scored_dir": _rel(scored_dir),
        "reps_dir": _rel(reps_dir),
        "label_source": LABEL_SOURCE_RULE,
        "label_tier_source_counts": source_counts,
        "out": str(out),
        "val_fraction": args.val_fraction,
        "seed": args.seed if n_val > 0 else None,
        "n_train": n_by_set["train"],
        "n_val": n_by_set["val"],
        "n_test": n_by_set["test"],
        "n_total": sum(n_by_set.values()),
        "total_residues": total_res,
        "sentinel_residues": total_sentinel,
        "sentinel_fraction": round(total_sentinel / total_res, 4) if total_res else 0,
        "records_below_min_valid": len(few_valid),
        "min_valid_threshold": args.min_valid,
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote {summary['n_total']} records -> {out}")
    print(f"  train (SET=train): {summary['n_train']}")
    if n_val > 0:
        print(
            f"  val   (SET=val)  : {summary['n_val']} "
            f"({args.val_fraction:.0%} of train, seed={args.seed})"
        )
    print(f"  test  (SET=test) : {summary['n_test']}")
    print(f"  labels by selecting tier: {source_counts}")
    print(
        f"  residues: {total_res} total, {total_sentinel} masked "
        f"({summary['sentinel_fraction'] * 100:.1f}% sentinel 999.0)"
    )
    if few_valid:
        print(
            f"  NOTE: {len(few_valid)} records have < {args.min_valid} scored "
            f"residues, e.g. {few_valid[:5]}"
        )
    print(f"  summary -> {out.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
