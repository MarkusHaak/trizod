#!/usr/bin/env python3
"""Analyse duplicate/related BMRB entries sharing the same protein sequence.

Uses pre-computed baseline JSONs (data/interim/baseline/) which already contain
per-tier filtered entries with sequence, conditions, and offsets. Enriches
with entity composition from cached entries to classify bound vs unbound.

Usage:
    uv run python scripts/analyse_duplicate_entries.py --tier tolerant
    uv run python scripts/analyse_duplicate_entries.py --tier moderate --output docs/260422/duplicate-entry-analysis.md
"""

import argparse
import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from trizod.bmrb.bmrb import get_valid_bbshifts
from trizod.constants import BACKBONE_ATOMS

logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
logging.getLogger("trizod.bmrb").setLevel(logging.ERROR)
logger = logging.getLogger("duplicates")


def load_baseline(baseline_path):
    """Load a baseline JSON file into a list of dicts."""
    rows = []
    with open(baseline_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def enrich_with_entity_info(rows, cache_dir):
    """Add entity composition info from cached entries."""
    # Load entity info for each unique entryID
    entry_ids = {r["entryID"] for r in rows}
    entity_info = {}

    for eid in entry_ids:
        pkl = cache_dir / f"{eid}.pkl"
        if not pkl.exists():
            continue
        try:
            with pkl.open("rb") as f:
                entry = pickle.load(f)
            has_non_polymer = any(
                e.type == "non-polymer" for e in entry.entities.values()
            )
            has_nucleic = any(
                e.type == "polymer"
                and e.polymer_type in ("polydeoxyribonucleotide", "polyribonucleotide")
                for e in entry.entities.values()
            )
            n_entities = len(entry.entities)
            entity_info[eid] = {
                "n_entities": n_entities,
                "has_non_polymer": has_non_polymer,
                "has_nucleic": has_nucleic,
                "is_single": n_entities == 1,
            }
        except Exception:
            pass

    for r in rows:
        info = entity_info.get(r["entryID"], {})
        r.update(info)


def build_sequence_groups(rows):
    """Group rows by exact sequence, return groups with 2+ entries."""
    by_seq = defaultdict(list)
    for r in rows:
        if r.get("seq") and len(r["seq"]) >= 10:
            by_seq[r["seq"]].append(r)
    return {seq: entries for seq, entries in by_seq.items() if len(entries) >= 2}


def get_shifts_for_row(row, cache_dir):
    """Load backbone shifts for a specific entry/entity from cache."""
    pkl = cache_dir / f"{row['entryID']}.pkl"
    if not pkl.exists():
        return None, None
    try:
        with pkl.open("rb") as f:
            entry = pickle.load(f)
        peptide_shifts = entry.get_peptide_shifts()
        for (_st, _ea, e_id), (shifts, *_rest) in peptide_shifts.items():
            if e_id == row["entityID"]:
                entity = entry.entities[e_id]
                if entity.seq:
                    return get_valid_bbshifts(shifts, entity.seq)
    except Exception:
        pass
    return None, None


def conditions_similar(a, b, max_temp_diff=10, max_ph_diff=1.0):
    """Check if two entries have similar experimental conditions."""
    T_a, T_b = a.get("temperature"), b.get("temperature")
    pH_a, pH_b = a.get("pH"), b.get("pH")
    if T_a and T_b and abs(T_a - T_b) > max_temp_diff:
        return False
    return not (pH_a and pH_b and abs(pH_a - pH_b) > max_ph_diff)


def compare_shifts(arr_a, mask_a, arr_b, mask_b):
    """Compare backbone shifts for overlapping residues."""
    if arr_a is None or arr_b is None:
        return None
    overlap = mask_a & mask_b
    if overlap.sum() == 0:
        return None

    results = {}
    for j, at in enumerate(BACKBONE_ATOMS):
        sel = overlap[:, j]
        n = sel.sum()
        if n < 3:
            results[at] = {"n": int(n)}
            continue
        diffs = arr_a[sel, j] - arr_b[sel, j]
        ad = np.abs(diffs)
        results[at] = {
            "n": int(n),
            "mean_abs": float(np.mean(ad)),
            "median_abs": float(np.median(ad)),
            "max_abs": float(np.max(ad)),
            "std": float(np.std(diffs)),
            "gt05": float(np.mean(ad > 0.5)),
            "gt1": float(np.mean(ad > 1.0)),
        }
    return results


def run_comparisons(groups, cache_dir, require_similar_conditions=True):
    """Find bound/unbound and independent pairs, compare shifts."""
    bound_unbound = []
    independent = []

    for _seq, entries in groups.items():
        single = [e for e in entries if e.get("is_single", False)]
        bound = [e for e in entries if e.get("has_non_polymer") or e.get("has_nucleic")]

        # Bound vs unbound
        for s in single[:3]:
            for b in bound[:3]:
                if s["entryID"] == b["entryID"]:
                    continue
                if require_similar_conditions and not conditions_similar(s, b):
                    continue
                arr_s, mask_s = get_shifts_for_row(s, cache_dir)
                arr_b, mask_b = get_shifts_for_row(b, cache_dir)
                stats = compare_shifts(arr_s, mask_s, arr_b, mask_b)
                if stats:
                    bound_unbound.append(
                        {
                            "protein": s.get("entity_name") or b.get("entity_name"),
                            "single": s["entryID"],
                            "bound": b["entryID"],
                            "n_entities": b.get("n_entities", "?"),
                            "T_s": s.get("temperature"),
                            "T_b": b.get("temperature"),
                            "pH_s": s.get("pH"),
                            "pH_b": b.get("pH"),
                            "stats": stats,
                        }
                    )

        # Independent (both single)
        for i in range(min(len(single), 4)):
            for k in range(i + 1, min(len(single), 4)):
                s1, s2 = single[i], single[k]
                if require_similar_conditions and not conditions_similar(s1, s2):
                    continue
                arr_1, mask_1 = get_shifts_for_row(s1, cache_dir)
                arr_2, mask_2 = get_shifts_for_row(s2, cache_dir)
                stats = compare_shifts(arr_1, mask_1, arr_2, mask_2)
                if stats:
                    independent.append(
                        {
                            "protein": s1.get("entity_name") or s2.get("entity_name"),
                            "entry_a": s1["entryID"],
                            "entry_b": s2["entryID"],
                            "T_a": s1.get("temperature"),
                            "T_b": s2.get("temperature"),
                            "pH_a": s1.get("pH"),
                            "pH_b": s2.get("pH"),
                            "stats": stats,
                        }
                    )

    return bound_unbound, independent


def agg_table(comparisons, label):
    """Aggregate stats into a markdown table."""
    per_atom = {at: [] for at in BACKBONE_ATOMS}
    for c in comparisons:
        for at in BACKBONE_ATOMS:
            s = c["stats"].get(at, {})
            if "mean_abs" in s and s["n"] >= 3:
                per_atom[at].append(s)

    lines = [f"### {label}", "", f"**{len(comparisons)} pairs compared**", ""]
    lines.append(
        "| Atom | Pairs | Mean abs | Median abs | Max abs | Std | >0.5 ppm | >1.0 ppm |"
    )
    lines.append(
        "| ---- | ----: | -------: | ---------: | ------: | --: | -------: | -------: |"
    )
    for at in BACKBONE_ATOMS:
        e = per_atom[at]
        if not e:
            lines.append(f"| {at} | 0 | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {at} | {len(e)} "
            f"| {np.mean([s['mean_abs'] for s in e]):.3f} "
            f"| {np.mean([s['median_abs'] for s in e]):.3f} "
            f"| {np.mean([s['max_abs'] for s in e]):.3f} "
            f"| {np.mean([s['std'] for s in e]):.3f} "
            f"| {np.mean([s['gt05'] for s in e]):.1%} "
            f"| {np.mean([s['gt1'] for s in e]):.1%} |"
        )
    return lines


def format_report(groups, bound_unbound, independent, tier, cond_match):
    """Format full report."""
    total_seqs = len(groups)
    total_entries = sum(len(v) for v in groups.values())

    lines = [
        "# Duplicate Entry Analysis: Same Protein, Different Experiments",
        "",
        f"**Tier:** {tier} | **Condition matching:** "
        f"{'T within 10K, pH within 1.0' if cond_match else 'none'}",
        f"- **{total_entries}** entries across **{total_seqs}** sequences with 2+ entries",
        "",
    ]

    # Top proteins
    top = sorted(groups.items(), key=lambda x: -len(x[1]))[:10]
    lines.extend(["## Most frequently measured proteins", ""])
    lines.append("| Protein | Len | Entries | Single | Bound |")
    lines.append("| ------- | --: | ------: | -----: | ----: |")
    for seq, entries in top:
        name = next(
            (e.get("entity_name") for e in entries if e.get("entity_name")), "?"
        )
        if len(name) > 35:
            name = name[:32] + "..."
        single = sum(1 for e in entries if e.get("is_single", False))
        bound = sum(
            1 for e in entries if e.get("has_non_polymer") or e.get("has_nucleic")
        )
        lines.append(f"| {name} | {len(seq)} | {len(entries)} | {single} | {bound} |")

    # Bound vs unbound
    lines.extend(["", "## Bound vs Unbound Shift Differences", ""])
    lines.extend(agg_table(bound_unbound, "Bound vs Unbound"))

    # Independent
    lines.extend(["", "## Independent Measurement Reproducibility", ""])
    lines.extend(agg_table(independent, "Independent measurements (noise floor)"))

    # Ratio
    if bound_unbound and independent:
        lines.extend(["", "## Binding effect vs experimental noise", ""])
        lines.append("| Atom | Binding (MAD) | Noise (MAD) | Ratio |")
        lines.append("| ---- | ------------: | ----------: | ----: |")
        for at in BACKBONE_ATOMS:
            bu = [
                c["stats"][at]["mean_abs"]
                for c in bound_unbound
                if "mean_abs" in c["stats"].get(at, {})
            ]
            ind = [
                c["stats"][at]["mean_abs"]
                for c in independent
                if "mean_abs" in c["stats"].get(at, {})
            ]
            if bu and ind:
                bm, im = np.mean(bu), np.mean(ind)
                ratio = bm / im if im > 0 else float("inf")
                lines.append(f"| {at} | {bm:.3f} | {im:.3f} | {ratio:.1f}x |")
            else:
                lines.append(f"| {at} | — | — | — |")
        lines.append("")
        lines.append("Ratio > 1 means binding effect exceeds experimental noise.")

    # Examples
    if bound_unbound:
        lines.extend(["", "## Example bound/unbound pairs (first 15)", ""])
        lines.append(
            "| Protein | Single | Bound | T_s | T_b | pH_s | pH_b | CA MAD | N MAD |"
        )
        lines.append(
            "| ------- | -----: | ----: | --: | --: | ---: | ---: | -----: | ----: |"
        )
        for c in bound_unbound[:15]:
            name = (c["protein"] or "?")[:25]
            ca = c["stats"].get("CA", {}).get("mean_abs")
            n = c["stats"].get("N", {}).get("mean_abs")
            ca_s = f"{ca:.3f}" if ca is not None else "—"
            n_s = f"{n:.3f}" if n is not None else "—"
            t_s = f"{c['T_s']:.0f}" if c["T_s"] else "?"
            t_b = f"{c['T_b']:.0f}" if c["T_b"] else "?"
            ph_s = f"{c['pH_s']:.1f}" if c["pH_s"] else "?"
            ph_b = f"{c['pH_b']:.1f}" if c["pH_b"] else "?"
            lines.append(
                f"| {name} | bmr{c['single']} | bmr{c['bound']} | "
                f"{t_s} | {t_b} | {ph_s} | {ph_b} | {ca_s} | {n_s} |"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse duplicate BMRB entries and shift reproducibility"
    )
    parser.add_argument(
        "--tier",
        choices=["unfiltered", "tolerant", "moderate", "strict"],
        default="tolerant",
        help="Which baseline tier to use (default: tolerant)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("data/interim/baseline"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("tmp/bmrb_entries"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--no-condition-match",
        action="store_true",
        help="Don't require similar T/pH between compared entries",
    )
    args = parser.parse_args()

    baseline_file = args.baseline_dir / f"{args.tier}.json"
    logger.info(f"Loading baseline: {baseline_file}")
    rows = load_baseline(baseline_file)
    logger.info(f"Loaded {len(rows)} entries from {args.tier} tier")

    logger.info("Enriching with entity composition info...")
    enrich_with_entity_info(rows, args.cache_dir)

    logger.info("Grouping by sequence...")
    groups = build_sequence_groups(rows)
    logger.info(f"Found {len(groups)} sequences with 2+ entries")

    cond_match = not args.no_condition_match
    logger.info(f"Running comparisons (condition match: {cond_match})...")
    bound_unbound, independent = run_comparisons(
        groups, args.cache_dir, require_similar_conditions=cond_match
    )
    logger.info(
        f"Found {len(bound_unbound)} bound/unbound pairs, "
        f"{len(independent)} independent pairs"
    )

    report = format_report(groups, bound_unbound, independent, args.tier, cond_match)
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        logger.info(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
