#!/usr/bin/env python3
"""Analyse per-filter impact across all BMRB entries for each stringency tier.

Loads cached BMRB entries, builds the peptide DataFrame, and applies each
filter independently to measure how many entries it removes (total and unique).
Runs only the pre-filter stage — no scoring required.

Usage:
    uv run python scripts/filter_impact_report.py --cache-dir tmp
    uv run python scripts/filter_impact_report.py --cache-dir tmp --output docs/260415/filter-impact.md
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trizod.constants import CANONICAL_AA_MASK
from trizod.trizod import (
    create_peptide_dataframe,
    filter_defaults,
    find_bmrb_files,
    load_bmrb_entries,
    prefilter_dataframe,
)

# Initialize pandarallel (required for parallel_apply in load/create)
from pandarallel import pandarallel

pandarallel.initialize(verbose=0, nb_workers=4)

# Silence noisy sub-loggers
logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
logging.getLogger("trizod.bmrb").setLevel(logging.CRITICAL)
logging.getLogger("trizod.scoring").setLevel(logging.CRITICAL)


def build_dataframe(input_dir, cache_dir, tier="unfiltered"):
    """Load entries and build the peptide DataFrame with unfiltered settings."""
    bmrb_files = find_bmrb_files(input_dir)
    defaults = filter_defaults.loc[tier]

    global_entries, failed = load_bmrb_entries(bmrb_files, cache_dir=cache_dir)

    if failed:
        logging.warning(f"Failed loading {len(failed)} of {len(bmrb_files)} files")

    df = create_peptide_dataframe(
        global_entries,
        chemical_denaturants=filter_defaults.loc["strict", "chemical-denaturants"],
        keywords=filter_defaults.loc["strict", "keywords-blacklist"],
        return_default=True,
        assume_si=True,
        fix_outliers=True,
    )
    return df


def analyse_tier(df, tier):
    """Run prefilter for a given tier and return per-filter impact stats."""
    defaults = filter_defaults.loc[tier]

    # Replicate CLI behaviour: append inf if only min length given
    pep_range = list(defaults["peptide-length-range"])
    if len(pep_range) == 1:
        pep_range.append(np.inf)

    _result = prefilter_dataframe(
        df.copy(),
        method_whitelist=defaults["exp-method-whitelist"],
        method_blacklist=defaults["exp-method-blacklist"],
        temperature_range=defaults["temperature-range"],
        ionic_strength_range=defaults["ionic-strength-range"],
        pH_range=defaults["pH-range"],
        peptide_length_range=pep_range,
        min_backbone_shift_types=defaults["min-backbone-shift-types"],
        min_backbone_shift_positions=defaults["min-backbone-shift-positions"],
        min_backbone_shift_fraction=defaults["min-backbone-shift-fraction"],
        max_noncanonical_fraction=defaults["max-noncanonical-fraction"],
        max_x_fraction=defaults["max-x-fraction"],
        keywords=defaults["keywords-blacklist"],
        chemical_denaturants=defaults["chemical-denaturants"],
        exclude_paramagnetic=defaults["exclude-paramagnetic"],
    )
    (
        df_filtered,
        missing_vals,
        sels_pre,
        sels_kws,
        sels_denat,
        sels_paramag,
        sels_all,
    ) = _result

    # Collect all filter selections into one dict
    all_filters = {}
    for (name, crit), sel in sels_pre.items():
        label = f"{name} {crit}".strip()
        all_filters[label] = sel
    for name, sel in sels_kws.items():
        all_filters[f"keyword: {name}"] = sel
    for name, sel in sels_denat.items():
        all_filters[f"denaturant: {name}"] = sel
    for name, sel in sels_paramag.items():
        all_filters[name] = sel

    # Compute per-filter: total filtered, uniquely filtered
    results = []
    for name, sel in all_filters.items():
        filtered = int((~sel).sum())
        # Unique: entries filtered ONLY by this filter
        others_pass = pd.Series(np.full(len(sel), True))
        for other_name, other_sel in all_filters.items():
            if other_name != name:
                others_pass &= other_sel
        unique = int((~sel & others_pass).sum())
        results.append(
            {
                "filter": name,
                "filtered": filtered,
                "unique": unique,
            }
        )

    total = len(df_filtered)
    passing = int(df_filtered["pass_pre"].sum())
    return results, total, passing


def format_markdown(all_results, total_entries):
    """Format results as a markdown report."""
    lines = [
        "# Filter Impact Report",
        "",
        f"Total entries (rows) in DataFrame: **{total_entries}**",
        "",
        "Each table shows how many entries a filter removes (filtered) and how many",
        "are removed *only* by that filter (unique — would pass if that filter were",
        "disabled).",
        "",
    ]

    for tier, (results, total, passing) in all_results.items():
        filtered_total = total - passing
        pct = 100 * filtered_total / total if total else 0
        lines.append(f"## {tier.capitalize()} tier")
        lines.append("")
        lines.append(
            f"Passing: **{passing}** / {total} (filtered: {filtered_total}, {pct:.1f}%)"
        )
        lines.append("")
        lines.append("| Filter | Filtered | Unique |")
        lines.append("| ------ | -------: | -----: |")
        for r in results:
            if r["filtered"] > 0:
                lines.append(f"| {r['filter']} | {r['filtered']} | {r['unique']} |")
        lines.append("")

    return "\n".join(lines)


def format_terminal(all_results, total_entries):
    """Format results for terminal output."""
    lines = [f"\nTotal entries (rows): {total_entries}\n"]

    for tier, (results, total, passing) in all_results.items():
        filtered_total = total - passing
        pct = 100 * filtered_total / total if total else 0
        lines.append(f"{'=' * 70}")
        lines.append(f" {tier.upper()} TIER")
        lines.append(
            f" Passing: {passing} / {total} (filtered: {filtered_total}, {pct:.1f}%)"
        )
        lines.append(f"{'=' * 70}")
        lines.append(f"{'Filter':<45} {'Filtered':>10} {'Unique':>8}")
        lines.append(f"{'-' * 45} {'-' * 10} {'-' * 8}")
        for r in results:
            if r["filtered"] > 0:
                lines.append(f"{r['filter']:<45} {r['filtered']:>10} {r['unique']:>8}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse per-filter impact across BMRB entries"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/bmrb_entries"),
        help="Directory containing BMRB entry folders (default: data/raw/bmrb_entries)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("tmp"),
        help="Cache directory with pickled entries (default: tmp)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write markdown report to this file (default: terminal only)",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=["unfiltered", "tolerant", "moderate", "strict"],
        default=["unfiltered", "tolerant", "moderate", "strict"],
        help="Which tiers to analyse (default: all)",
    )
    args = parser.parse_args()

    logging.info("Loading BMRB entries and building DataFrame...")
    df = build_dataframe(args.input_dir, args.cache_dir)
    total_entries = len(df)
    logging.info(f"DataFrame has {total_entries} rows")

    all_results = {}
    for tier in args.tiers:
        logging.info(f"Analysing {tier} tier...")
        results, total, passing = analyse_tier(df, tier)
        all_results[tier] = (results, total, passing)

    print(format_terminal(all_results, total_entries))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(format_markdown(all_results, total_entries))
        logging.info(f"Markdown report written to {args.output}")


if __name__ == "__main__":
    main()
