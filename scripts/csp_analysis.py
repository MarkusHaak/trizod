#!/usr/bin/env python3
"""Reid #1 - Chemical Shift Perturbation (CSP) analysis on duplicate pairs.

Computes Reid Alderson's HN/N CSP formula:
    CSP = sqrt(dH^2 + (dN/alpha)^2),  alpha=5

over the bound/unbound pairs identified by analyse_duplicate_entries.py.
Outputs:
  - histogram of CSP values across all pairs (with trimmed-mean+SD threshold)
  - per-residue CSP bar plot for one named pair (binding interface example)

Usage:
    uv run python scripts/csp_analysis.py \\
        --tier tolerant \\
        --baseline-dir data/baseline \\
        --cache-dir tmp/bmrb_entries \\
        --output-histogram docs/260505/figures/csp_histogram.png \\
        --output-example docs/260505/figures/csp_interface_example.png \\
        --example-single 16925 --example-bound 16931
"""

import argparse
import json
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trizod.bmrb.bmrb import get_valid_bbshifts
from trizod.constants import BACKBONE_ATOMS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("csp")

# Reid's down-weighting factor for 15N relative to 1HN.
ALPHA_N = 5.0


def load_baseline(path):
    """Load NDJSON baseline file into a list of row dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def shifts_for(entry_id, cache_dir, entity_id_filter=None):
    """Load the first matching peptide shift table from a cached BmrbEntry.

    Returns (bbshifts_arr, bbshifts_mask, seq) or (None, None, None).
    """
    pkl = cache_dir / f"{entry_id}.pkl"
    entry = None
    if pkl.exists():
        try:
            with pkl.open("rb") as f:
                entry = pickle.load(f)
        except Exception:
            entry = None
    if entry is None:
        # Fall back to re-parsing from raw .str
        from trizod.bmrb.bmrb import BmrbEntry

        raw_dir = Path("data/bmrb_entries") / f"bmr{entry_id}"
        if not raw_dir.exists():
            return None, None, None
        try:
            entry = BmrbEntry(entry_id, raw_dir)
        except Exception:
            return None, None, None
    peptide_shifts = entry.get_peptide_shifts()
    for (_st, _ea, e_id), (shifts, *_rest) in peptide_shifts.items():
        if entity_id_filter is not None and e_id != entity_id_filter:
            continue
        if e_id not in entry.entities:
            continue
        seq = entry.entities[e_id].seq
        if not seq:
            continue
        ret = get_valid_bbshifts(shifts, seq)
        if ret is None:
            continue
        return ret[0], ret[1], seq
    return None, None, None


def compute_pair_csp(seq, arr_a, mask_a, arr_b, mask_b):
    """Per-residue HN/N CSP for one bound/unbound pair.

    Returns a length-N array; positions without both H and N in both entries
    are NaN.
    """
    h_idx = BACKBONE_ATOMS.index("H")
    n_idx = BACKBONE_ATOMS.index("N")
    have_both = (
        mask_a[:, h_idx] & mask_a[:, n_idx] & mask_b[:, h_idx] & mask_b[:, n_idx]
    )
    n = len(seq)
    csp = np.full(n, np.nan)
    for i in range(n):
        if not have_both[i]:
            continue
        dH = arr_a[i, h_idx] - arr_b[i, h_idx]
        dN = arr_a[i, n_idx] - arr_b[i, n_idx]
        csp[i] = float(np.sqrt(dH * dH + (dN / ALPHA_N) ** 2))
    return csp


def trimmed_mean_threshold(values, drop_top_frac=0.10):
    """Reid's trimmed-mean threshold: drop the top 10%, then mean+SD on the rest."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    cutoff = np.quantile(arr, 1.0 - drop_top_frac)
    kept = arr[arr <= cutoff]
    return float(kept.mean()), float(kept.mean() + kept.std(ddof=0))


def find_pairs(rows, cache_dir, max_temp_diff=10.0, max_ph_diff=1.0):
    """Group rows by exact sequence; return (seq, single, bound) triples
    where conditions are similar (T within max_temp_diff, pH within max_ph_diff)."""
    by_seq = defaultdict(list)
    for r in rows:
        if r.get("seq") and len(r["seq"]) >= 10:
            by_seq[r["seq"]].append(r)
    pairs = []
    for seq, entries in by_seq.items():
        if len(entries) < 2:
            continue
        single, bound = [], []
        for r in entries:
            pkl = cache_dir / f"{r['entryID']}.pkl"
            if not pkl.exists():
                continue
            try:
                with pkl.open("rb") as f:
                    entry = pickle.load(f)
            except Exception:
                continue
            n_entities = len(entry.entities)
            has_non_polymer = any(
                e.type == "non-polymer" for e in entry.entities.values()
            )
            has_nucleic = any(
                e.type == "polymer"
                and e.polymer_type
                in ("polydeoxyribonucleotide", "polyribonucleotide")
                for e in entry.entities.values()
            )
            if n_entities == 1:
                single.append(r)
            elif has_non_polymer or has_nucleic:
                bound.append(r)
        # Take cartesian product, restricted to similar conditions
        for s in single[:3]:
            for b in bound[:3]:
                if s["entryID"] == b["entryID"]:
                    continue
                T_a, T_b = s.get("temperature"), b.get("temperature")
                pH_a, pH_b = s.get("pH"), b.get("pH")
                if T_a and T_b and abs(T_a - T_b) > max_temp_diff:
                    continue
                if pH_a and pH_b and abs(pH_a - pH_b) > max_ph_diff:
                    continue
                pairs.append((seq, s, b))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tier",
        choices=["unfiltered", "tolerant", "moderate", "strict"],
        default="tolerant",
    )
    parser.add_argument("--baseline-dir", type=Path, default=Path("data/baseline"))
    parser.add_argument("--cache-dir", type=Path, default=Path("tmp/bmrb_entries"))
    parser.add_argument(
        "--output-histogram",
        type=Path,
        default=Path("docs/260505/figures/csp_histogram.png"),
    )
    parser.add_argument(
        "--output-example",
        type=Path,
        default=Path("docs/260505/figures/csp_interface_example.png"),
    )
    parser.add_argument(
        "--example-single",
        default="16925",
        help="BMRB id of unbound (apo) example entry",
    )
    parser.add_argument(
        "--example-bound",
        default="16931",
        help="BMRB id of bound example entry",
    )
    parser.add_argument(
        "--example-name",
        default="FKBP12",
        help="Protein name for the example title",
    )
    args = parser.parse_args()

    rows = load_baseline(args.baseline_dir / f"{args.tier}.json")
    pairs = find_pairs(rows, args.cache_dir)
    print(f"found {len(pairs)} bound/unbound pairs", file=sys.stderr)

    all_csp = []
    for seq, s, b in pairs:
        arr_s, mask_s, _ = shifts_for(s["entryID"], args.cache_dir, s.get("entityID"))
        arr_b, mask_b, _ = shifts_for(b["entryID"], args.cache_dir, b.get("entityID"))
        if arr_s is None or arr_b is None:
            continue
        if arr_s.shape != arr_b.shape:
            continue
        csp = compute_pair_csp(seq, arr_s, mask_s, arr_b, mask_b)
        all_csp.extend(csp[~np.isnan(csp)].tolist())

    print(f"total CSP values across all pairs: {len(all_csp)}", file=sys.stderr)
    trim_mean, threshold = trimmed_mean_threshold(all_csp)
    print(
        f"trimmed mean = {trim_mean:.3f}, threshold = mean+SD = {threshold:.3f}",
        file=sys.stderr,
    )

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if all_csp:
        upper = np.quantile(all_csp, 0.99)  # clip extreme tail for legibility
        plot_csp = [c for c in all_csp if c <= upper]
        ax.hist(plot_csp, bins=80, color="#4C72B0", alpha=0.85)
    if not np.isnan(threshold):
        ax.axvline(
            threshold,
            color="red",
            ls="--",
            lw=1.2,
            label=f"trimmed mean+SD = {threshold:.3f}",
        )
        ax.legend()
    ax.set_xlabel("CSP (ppm)")
    ax.set_ylabel("count")
    ax.set_title(
        f"HN/N CSP across {len(all_csp):,} residue pairs (alpha = {ALPHA_N}, 99% clip)"
    )
    fig.tight_layout()
    args.output_histogram.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_histogram, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"histogram written to {args.output_histogram}")

    # Interface example
    arr_s, mask_s, seq_s = shifts_for(args.example_single, args.cache_dir)
    arr_b, mask_b, seq_b = shifts_for(args.example_bound, args.cache_dir)
    if arr_s is None or arr_b is None:
        print(
            f"example pair {args.example_single} vs {args.example_bound} unavailable - skipping example plot",
            file=sys.stderr,
        )
        return
    if arr_s.shape != arr_b.shape:
        print(
            f"example pair shape mismatch ({arr_s.shape} vs {arr_b.shape}) - skipping example plot",
            file=sys.stderr,
        )
        return
    csp = compute_pair_csp(seq_s, arr_s, mask_s, arr_b, mask_b)
    residues = np.arange(1, len(seq_s) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    bar_color = np.where(csp > threshold, "#d62728", "#4C72B0")
    ax.bar(residues, np.where(np.isnan(csp), 0, csp), color=bar_color)
    ax.axhline(
        threshold, color="red", ls="--", lw=1.0, label=f"threshold = {threshold:.3f}"
    )
    ax.set_xlabel("residue")
    ax.set_ylabel("CSP (ppm)")
    ax.set_title(
        f"binding interface - {args.example_name}: bmr{args.example_single} (apo) vs bmr{args.example_bound} (bound)"
    )
    ax.legend()
    fig.tight_layout()
    args.output_example.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_example, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"example interface plot written to {args.output_example}")


if __name__ == "__main__":
    main()
