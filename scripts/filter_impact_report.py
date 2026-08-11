#!/usr/bin/env python3
"""Analyse per-filter impact across all BMRB entries for each stringency tier.

Loads cached BMRB entries, builds the peptide DataFrame, and applies each
filter independently to measure how many entries it removes (total and unique).
Runs only the pre-filter stage — no scoring required.

Every argument is resolved from ``filter_defaults`` through the two maps below
rather than restated here. The restatement was the bug: the report silently ran
with ``keyword_search_scope="all"`` (paper-topic fields included, unlike every
shipped tier), no ``physical_state_blacklist`` and ``method_fallback="off"``,
so it reported numbers no shipped tier produces.

Because ``default-conditions`` / ``unit-assumptions`` / ``unit-corrections``
decide how pH, temperature and ionic strength are *read*, a frame built for one
tier cannot be reused for a tier that sets them differently — one frame is built
per distinct configuration (3 for the four shipped tiers).

Usage:
    uv run python scripts/filter_impact_report.py --cache-dir tmp
    uv run python scripts/filter_impact_report.py --cache-dir tmp --output docs/archive/260415/filter-impact.md
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Initialize pandarallel (required for parallel_apply in load/create)
from pandarallel import pandarallel

from trizod.trizod import (
    create_peptide_dataframe,
    filter_defaults,
    find_bmrb_files,
    load_bmrb_entries,
    prefilter_dataframe,
)

pandarallel.initialize(verbose=0, nb_workers=4)

# Silence noisy sub-loggers
logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
logging.getLogger("trizod.bmrb").setLevel(logging.CRITICAL)
logging.getLogger("trizod.scoring").setLevel(logging.CRITICAL)

TIERS = list(filter_defaults.index)

#: ``filter_defaults`` column -> ``prefilter_dataframe`` keyword. Covers every
#: parameter of that function bar ``df``; ``tests/test_filter_impact_report.py``
#: asserts that, so a filter added to the pipeline cannot quietly default here.
PREFILTER_ARGS = {
    "exp-method-whitelist": "method_whitelist",
    "exp-method-blacklist": "method_blacklist",
    "temperature-range": "temperature_range",
    "ionic-strength-range": "ionic_strength_range",
    "pH-range": "pH_range",
    "peptide-length-range": "peptide_length_range",
    "min-backbone-shift-types": "min_backbone_shift_types",
    "min-backbone-shift-positions": "min_backbone_shift_positions",
    "min-backbone-shift-fraction": "min_backbone_shift_fraction",
    "max-noncanonical-fraction": "max_noncanonical_fraction",
    "max-x-fraction": "max_x_fraction",
    "keywords-blacklist": "keywords",
    "perturbing-cosolvents": "perturbing_cosolvents",
    "exclude-paramagnetic": "exclude_paramagnetic",
    "physical-state-blacklist": "physical_state_blacklist",
    "method-fallback": "method_fallback",
}

#: ``filter_defaults`` column -> ``create_peptide_dataframe`` keyword. These are
#: tier policy too: they decide which fields the keyword blacklist searches and
#: how the sample conditions are read, so they partition the tiers into frames.
FRAME_ARGS = {
    "default-conditions": "return_default",
    "unit-assumptions": "assume_si",
    "unit-corrections": "fix_outliers",
    "keyword-search-scope": "keyword_search_scope",
}


def _value(defaults, key):
    """One ``filter_defaults`` cell, list-valued cells copied (never mutated)."""
    value = defaults[key]
    return list(value) if isinstance(value, (list, tuple)) else value


def tier_prefilter_kwargs(tier):
    """Every ``prefilter_dataframe`` argument for ``tier``, from the defaults."""
    defaults = filter_defaults.loc[tier]
    kwargs = {arg: _value(defaults, key) for key, arg in PREFILTER_ARGS.items()}
    # CLI behaviour (`_validate_and_prepare_paths`): a one-element length range
    # means "no upper bound".
    if len(kwargs["peptide_length_range"]) == 1:
        kwargs["peptide_length_range"].append(np.inf)
    return kwargs


def tier_frame_kwargs(tier):
    """Every tier-driven ``create_peptide_dataframe`` argument for ``tier``."""
    defaults = filter_defaults.loc[tier]
    return {arg: _value(defaults, key) for key, arg in FRAME_ARGS.items()}


def frame_key(tier):
    """Tiers sharing this key can share one peptide DataFrame."""
    return tuple(sorted(tier_frame_kwargs(tier).items()))


def _union(tiers, key):
    """Order-preserving union of a list-valued default across ``tiers``."""
    merged = {}
    for tier in tiers:
        merged.update(dict.fromkeys(filter_defaults.loc[tier, key]))
    return list(merged)


def build_dataframes(input_dir, cache_dir, tiers):
    """Load entries once and build one peptide DataFrame per frame configuration.

    Keyword and cosolvent columns are built from the union over ``tiers`` so any
    frame carries the columns every tier's selections index; each tier still
    applies only its own lists.
    """
    bmrb_files = find_bmrb_files(input_dir)
    # the pickle cache dir is created by the CLI, not by load_bmrb_entries
    Path(cache_dir, "bmrb_entries").mkdir(parents=True, exist_ok=True)
    global_entries, failed = load_bmrb_entries(bmrb_files, cache_dir=cache_dir)
    if failed:
        logging.warning(f"Failed loading {len(failed)} of {len(bmrb_files)} files")

    keywords = _union(tiers, "keywords-blacklist")
    cosolvents = _union(tiers, "perturbing-cosolvents")

    frames = {}
    for tier in tiers:
        key = frame_key(tier)
        if key in frames:
            continue
        kwargs = tier_frame_kwargs(tier)
        logging.info(f"Building peptide DataFrame for {dict(kwargs)} ...")
        frames[key] = create_peptide_dataframe(
            global_entries,
            perturbing_cosolvents=cosolvents,
            keywords=keywords,
            **kwargs,
        )
    return frames


def analyse_tier(df, tier):
    """Run prefilter for a given tier and return per-filter impact stats."""
    _result = prefilter_dataframe(df.copy(), **tier_prefilter_kwargs(tier))
    (
        df_filtered,
        missing_vals,
        sels_pre,
        sels_kws,
        sels_cosolvent,
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
    for name, sel in sels_cosolvent.items():
        all_filters[f"cosolvent: {name}"] = sel
    for name, sel in sels_paramag.items():
        all_filters[name] = sel

    # Compute per-filter: total filtered, uniquely filtered
    results = []
    for name, sel in all_filters.items():
        filtered = int((~sel).sum())
        # Unique: entries filtered ONLY by this filter. index=sel.index, else the
        # fresh RangeIndex misaligns against the index-aligned selections.
        others_pass = pd.Series(np.full(len(sel), True), index=sel.index)
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
        choices=TIERS,
        default=list(TIERS),
        help="Which tiers to analyse (default: all)",
    )
    args = parser.parse_args()

    logging.info("Loading BMRB entries and building DataFrames...")
    frames = build_dataframes(args.input_dir, args.cache_dir, args.tiers)
    total_entries = len(next(iter(frames.values())))
    logging.info(
        f"{len(frames)} DataFrame(s) of {total_entries} rows "
        f"for {len(args.tiers)} tier(s)"
    )

    all_results = {}
    for tier in args.tiers:
        logging.info(f"Analysing {tier} tier...")
        results, total, passing = analyse_tier(frames[frame_key(tier)], tier)
        all_results[tier] = (results, total, passing)

    print(format_terminal(all_results, total_entries))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(format_markdown(all_results, total_entries))
        logging.info(f"Markdown report written to {args.output}")


if __name__ == "__main__":
    main()
