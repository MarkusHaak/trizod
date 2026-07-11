#!/usr/bin/env python3
"""Compare G-scores with and without LACS re-referencing before POTENCI.

Two modes:
  --compute   Process entries and save results to pickle (slow, once)
  --plot-only Re-plot from cached results (fast, iterative)

Produces:
1. Boxplot: G-score difference (lacs - orig) per filter tier
2. Hexbin: original vs LACS-corrected G-scores
3. Violin: distribution of LACS offsets per atom type

Usage:
    uv run python scripts/figures/compare_gscores_lacs.py --compute
    uv run python scripts/figures/compare_gscores_lacs.py --plot-only
    uv run python scripts/figures/compare_gscores_lacs.py --compute --max-entries 10
"""

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import trizod.bmrb.bmrb as bmrb
import trizod.potenci.potenci as potenci
from trizod.cache import load_potenci_cache, save_potenci_cache
from trizod.constants import BACKBONE_ATOMS
from trizod.figures.style import TIERS, classify_tier, load_tier_sets
from trizod.lacs import compute_lacs_offsets
from trizod.scoring.scoring import (
    compare_to_predicted,
    compute_gscores,
    compute_offsets,
    compute_running_offsets,
    compute_weighted_diffs,
    compute_zscores,
    convert_to_triplet_data,
    get_outlier_mask,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Column indices in bbshifts_arr matching BACKBONE_ATOMS
_LACS_ATOMS = ["C", "CA", "CB", "HA", "H", "N"]  # skip HB (index 6)
_ATOM_COL = {atom: i for i, atom in enumerate(BACKBONE_ATOMS)}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def score_from_arrays(bbshifts_arr, bbshifts_mask, predshiftdct):
    """Run the scoring pipeline on a (n, 7) shift array.

    Replicates scoring.get_offset_corrected_shifts logic,
    then computes G-scores.

    Returns (gscores, offsets_dict) or (None, None) on failure.
    """
    # compare_to_predicted modifies bbshifts_arr in-place — caller must pass a copy
    diff_arr, _, cmp_mask = compare_to_predicted(
        predshiftdct, bbshifts_arr, bbshifts_mask
    )
    if np.sum(cmp_mask) == 0:
        return None, None

    offsets_initial = dict.fromkeys(BACKBONE_ATOMS, 0.0)
    weighted_diffs_initial, abs_weighted_diffs_initial = compute_weighted_diffs(
        diff_arr, cmp_mask, offsets_initial
    )
    zscores_initial = compute_zscores(
        abs_weighted_diffs_initial, cmp_mask.sum(axis=1), cmp_mask
    )
    zscores_triplet_initial = compute_zscores(
        *convert_to_triplet_data(abs_weighted_diffs_initial, cmp_mask), cmp_mask
    )
    outlier_mask_initial = get_outlier_mask(
        zscores_triplet_initial,
        zscores_initial,
        abs_weighted_diffs_initial,
        cmp_mask,
        cdf_threshold=6.0,
    )
    new_offsets_initial = compute_offsets(
        weighted_diffs_initial, cmp_mask & ~outlier_mask_initial, min_AIC=6.0
    )
    mean_zscore_initial = np.nanmean(zscores_triplet_initial)
    offsets_final = new_offsets_initial

    offsets_running = compute_running_offsets(diff_arr, cmp_mask, min_AIC=6.0)
    if offsets_running is not None and any(v != 0.0 for v in offsets_running.values()):
        weighted_diffs_corrected, abs_weighted_diffs_corrected = compute_weighted_diffs(
            diff_arr, cmp_mask, offsets_running
        )
        zscores_corrected = compute_zscores(
            abs_weighted_diffs_corrected, cmp_mask.sum(axis=1), cmp_mask
        )
        zscores_triplet_corrected = compute_zscores(
            *convert_to_triplet_data(abs_weighted_diffs_corrected, cmp_mask), cmp_mask
        )
        mean_zscore_corrected = np.nanmean(zscores_triplet_corrected)
        if mean_zscore_initial >= mean_zscore_corrected:
            outlier_mask_corrected = get_outlier_mask(
                zscores_triplet_corrected,
                zscores_corrected,
                abs_weighted_diffs_corrected,
                cmp_mask,
                cdf_threshold=6.0,
            )
            new_offsets_corrected = compute_offsets(
                weighted_diffs_corrected,
                cmp_mask & ~outlier_mask_corrected,
                min_AIC=6.0,
            )
            offsets_final = new_offsets_corrected

    _, abs_weighted_diffs_final = compute_weighted_diffs(
        diff_arr, cmp_mask, offsets_final
    )
    triplet_diffs, triplet_dof = convert_to_triplet_data(
        abs_weighted_diffs_final, cmp_mask
    )
    gscores = compute_gscores(triplet_diffs, triplet_dof, cmp_mask)
    return gscores, offsets_final


def apply_lacs(bbshifts_arr, bbshifts_mask, seq):
    """Compute and apply LACS offsets to a copy of bbshifts_arr."""
    n = len(seq)
    seq_nums = np.arange(1, n + 1)

    obs_shifts = {}
    for atom in _LACS_ATOMS:
        col = _ATOM_COL[atom]
        arr = np.full(n, np.nan)
        valid = bbshifts_mask[:, col]
        arr[valid] = bbshifts_arr[valid, col]
        obs_shifts[atom] = arr

    lacs_offsets = compute_lacs_offsets(seq, seq_nums, obs_shifts)

    corrected = bbshifts_arr.copy()
    for atom, offset in lacs_offsets.items():
        if offset is not None and atom in _ATOM_COL:
            col = _ATOM_COL[atom]
            valid = bbshifts_mask[:, col]
            corrected[valid, col] -= offset

    return corrected, lacs_offsets


def get_potenci(seq, temperature, pH, ion, cache_dir):
    """Get POTENCI predictions, using cache if available."""
    predshiftdct = load_potenci_cache(cache_dir, seq, temperature, pH, ion)
    if predshiftdct is None:
        use_ph_corr = pH != 7.0
        predshiftdct = potenci.get_pred_shifts(seq, temperature, pH, ion, use_ph_corr)
        save_potenci_cache(cache_dir, seq, temperature, pH, ion, predshiftdct)
    return predshiftdct


# ---------------------------------------------------------------------------
# Entry processing
# ---------------------------------------------------------------------------


def process_entry(entry, cache_dir):
    """Process one BMRB entry: compute G-scores with and without LACS.

    Returns (gscore_pairs, lacs_offsets).
    gscore_pairs: list of (entry_id, gscore_orig, gscore_lacs, max_lacs_offset)
    """
    peptide_shifts = entry.get_peptide_shifts()
    gscore_pairs = []
    entry_lacs_offsets = None

    for (_stID, _entity_assemID, entityID), (
        shifts,
        condID,
        _assemID,
        _sampleIDs,
    ) in peptide_shifts.items():
        if condID not in entry.conditions or entityID not in entry.entities:
            continue

        seq = entry.entities[entityID].seq
        if not seq or len(seq) < 20:
            continue

        cond = entry.conditions[condID]
        temperature = cond.get_temperature(return_default=True)
        pH = cond.get_pH(return_default=True)
        ion = cond.get_ionic_strength(return_default=True)

        ret = bmrb.get_valid_bbshifts(shifts, seq)
        if ret is None:
            continue
        bbshifts_arr, bbshifts_mask = ret

        try:
            predshiftdct = get_potenci(seq, temperature, pH, ion, cache_dir)
        except Exception:
            continue

        gscores_orig, _ = score_from_arrays(
            bbshifts_arr.copy(), bbshifts_mask.copy(), predshiftdct
        )
        if gscores_orig is None:
            continue

        corrected_arr, lacs_offsets = apply_lacs(bbshifts_arr, bbshifts_mask, seq)
        entry_lacs_offsets = lacs_offsets

        gscores_lacs, _ = score_from_arrays(
            corrected_arr, bbshifts_mask.copy(), predshiftdct
        )
        if gscores_lacs is None:
            continue

        abs_offsets = [abs(v) for v in lacs_offsets.values() if v is not None]
        max_lacs = max(abs_offsets) if abs_offsets else 0.0

        for i in range(len(seq)):
            go = gscores_orig[i]
            gl = gscores_lacs[i]
            if not np.isnan(go) and not np.isnan(gl):
                gscore_pairs.append((entry.id, go, gl, max_lacs))
        break  # first shift table only

    return gscore_pairs, entry_lacs_offsets


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_gscore_plot(df, output_path):
    """Boxplot of G-score difference per tier + hexbin density."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    text_bbox = {"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8}

    corr = np.corrcoef(df["gscore_orig"], df["gscore_lacs"])[0, 1]
    mae = df["diff"].abs().mean()
    n_entries = df["entry_id"].nunique()
    stats = (
        f"r = {corr:.4f}\nMAE = {mae:.4f}\n"
        f"N = {len(df):,} residues\n{n_entries} entries"
    )

    # Left: boxplot of G-score difference per tier
    box_data = []
    box_labels = []
    for tier in TIERS:
        subset = df.loc[df["tier"] == tier, "diff"]
        if len(subset) == 0:
            continue
        box_data.append(subset.values)
        box_labels.append(f"{tier}\n(n={len(subset):,})")

    bp = ax1.boxplot(
        box_data,
        tick_labels=box_labels,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#cccccc")
        patch.set_alpha(0.8)
    ax1.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax1.set_ylabel("G-score difference (LACS − original)")
    ax1.set_title("Effect of LACS pre-correction by tier")
    ax1.text(
        0.03,
        0.97,
        stats,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=text_bbox,
    )

    # Right: hexbin density
    x = df["gscore_orig"].values
    y = df["gscore_lacs"].values
    from matplotlib.colors import LogNorm

    hb = ax2.hexbin(
        x,
        y,
        gridsize=60,
        cmap="inferno_r",
        mincnt=1,
        extent=(0, 1, 0, 1),
        linewidths=0.3,
        norm=LogNorm(),
    )
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax2.set_xlabel("G-score (original pipeline)")
    ax2.set_ylabel("G-score (with LACS pre-correction)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")
    ax2.set_title("Density")
    plt.colorbar(hb, ax=ax2, label="Count", shrink=0.8)
    ax2.text(
        0.03,
        0.97,
        stats,
        transform=ax2.transAxes,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=text_bbox,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"G-score plot saved to {output_path}")
    plt.close(fig)


def make_lacs_offset_violin(all_offsets, output_path):
    """Violin plot: LACS offset distributions per atom type."""
    atoms = ["CA", "CB", "C", "HA", "H", "N"]
    clip = 10.0
    plot_data = []
    n_clipped = 0
    for atom in atoms:
        vals = [o[atom] for o in all_offsets if o[atom] is not None]
        n_clipped += sum(1 for v in vals if abs(v) > clip)
        vals = [max(-clip, min(clip, v)) for v in vals]
        plot_data.append(vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(
        plot_data, positions=range(len(atoms)), showmedians=True, showextrema=False
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")

    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xticks(range(len(atoms)))
    ax.set_xticklabels(atoms)
    ax.set_xlabel("Atom type")
    ax.set_ylabel("LACS offset (ppm)")
    title = "Distribution of LACS referencing corrections"
    if n_clipped:
        title += f" (clipped {n_clipped} values > {clip} ppm)"
    ax.set_title(title)

    for i, vals in enumerate(plot_data):
        ax.text(
            i,
            ax.get_ylim()[1] * 0.95,
            f"n={len(vals)}",
            ha="center",
            va="top",
            fontsize=8,
            color="gray",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Violin plot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compute", action="store_true", help="Compute and save results")
    mode.add_argument(
        "--plot-only", action="store_true", help="Plot from cached results"
    )

    parser.add_argument("--cache-dir", type=Path, default=Path("tmp"))
    parser.add_argument(
        "--subset-file",
        type=Path,
        default=Path("tests/quick_subset_ids.txt"),
    )
    parser.add_argument(
        "--results-pkl",
        type=Path,
        default=Path("tmp/lacs_comparison_results.pkl"),
        help="Path to save/load computed results",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("data/baseline"),
        help="Directory with {tier}.json baseline files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/archive/260422/gscores_lacs_comparison.png"),
    )
    parser.add_argument(
        "--output-violin",
        type=Path,
        default=Path("docs/archive/260422/lacs_offset_violin.png"),
    )
    parser.add_argument("--max-entries", type=int, default=0)
    args = parser.parse_args()

    if args.compute:
        pkl_dir = args.cache_dir / "bmrb_entries"
        subset_ids = args.subset_file.read_text().splitlines()
        subset_ids = [s.strip() for s in subset_ids if s.strip()]

        all_pairs = []  # list of (entry_id, gscore_orig, gscore_lacs, max_lacs_offset)
        all_lacs_offsets = []
        n_processed = 0
        n_skipped = 0

        for entry_id in subset_ids:
            if args.max_entries and n_processed >= args.max_entries:
                break
            pkl_path = pkl_dir / f"{entry_id}.pkl"
            if not pkl_path.exists():
                n_skipped += 1
                continue
            try:
                with open(pkl_path, "rb") as f:
                    entry = pickle.load(f)
                pairs, lacs_offsets = process_entry(entry, args.cache_dir)
                all_pairs.extend(pairs)
                if lacs_offsets is not None:
                    all_lacs_offsets.append(lacs_offsets)
                n_processed += 1
                if n_processed % 50 == 0:
                    print(
                        f"  processed {n_processed} entries, "
                        f"{len(all_pairs)} residues..."
                    )
            except Exception as e:
                logger.debug(f"Entry {entry_id}: {e}")
                n_skipped += 1

        print(
            f"Done: {n_processed} processed, {n_skipped} skipped, "
            f"{len(all_pairs)} residue pairs, "
            f"{len(all_lacs_offsets)} entries with LACS offsets"
        )

        # Save results
        args.results_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_pkl, "wb") as f:
            pickle.dump({"pairs": all_pairs, "lacs_offsets": all_lacs_offsets}, f)
        print(f"Results saved to {args.results_pkl}")

    # Load results (either just computed or from cache)
    if not args.results_pkl.exists():
        print(f"No results at {args.results_pkl}. Run with --compute first.")
        return

    with open(args.results_pkl, "rb") as f:
        results = pickle.load(f)
    all_pairs = results["pairs"]
    all_lacs_offsets = results["lacs_offsets"]

    if not all_pairs:
        print("No data to plot.")
        return

    # Build DataFrame and assign tiers from pipeline data
    df = pd.DataFrame(
        all_pairs, columns=["entry_id", "gscore_orig", "gscore_lacs", "max_lacs_offset"]
    )
    df["diff"] = df["gscore_lacs"] - df["gscore_orig"]

    tier_sets = load_tier_sets(args.baseline_dir)
    df["tier"] = df["entry_id"].apply(lambda eid: classify_tier(eid, tier_sets))

    for tier in TIERS:
        n = (df["tier"] == tier).sum()
        n_entries = df.loc[df["tier"] == tier, "entry_id"].nunique()
        print(f"  {tier}: {n:,} residues from {n_entries} entries")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_gscore_plot(df, args.output)

    if all_lacs_offsets:
        make_lacs_offset_violin(all_lacs_offsets, args.output_violin)


if __name__ == "__main__":
    main()
