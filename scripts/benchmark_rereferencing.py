#!/usr/bin/env python3
"""Benchmark re-referencing methods using synthetic data with known offsets.

Following Reid Alderson's proposed approach (2026-04-01):
1. Generate ground-truth random-coil shifts with POTENCI
2. Corrupt with known per-atom offsets
3. Run each method → compare recovered vs known offsets
4. Add noise for realism

Usage:
    uv run python scripts/benchmark_rereferencing.py
    uv run python scripts/benchmark_rereferencing.py --output docs/260415/benchmark-rereferencing.md
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trizod.potenci.potenci as potenci
from trizod.constants import BACKBONE_ATOMS, REFINED_WEIGHTS
from trizod.lacs import compute_lacs_offsets
from trizod.scoring.scoring import (
    compute_offsets,
    compute_running_offsets,
    compute_weighted_diffs,
    compute_zscores,
    convert_to_triplet_data,
    get_outlier_mask,
)

logging.basicConfig(level=logging.WARNING)

# --- Test sequences (diverse length and composition) ---
SEQUENCES = {
    "ubiquitin_76": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "alpha_synuclein_140": "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA",
    "gb1_56": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    "short_idp_30": "GGGSASGGGSGSAGSGGSGSAGGSGGSAGS",
    "lysozyme_129": "KVFGRCELAAALKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    "collagen_like_50": "GPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGPPGP",
    "charged_50": "EEEEKKKKDDDDRRRRHHHHEEEEKKKKDDDDRRRRHHHHEEEEKKKKDDDD"[:50],
    "aromatic_rich_60": "MFYWFYWFYWFYWFYWGGGGGFYWFYWFYWFYWFYWGGGGGFYWFYWFYWFYWFYWGGGGG",
    "proline_rich_45": "PPPPAGPPPPAGPPPPAGPPPPAGPPPPAGPPPPAGPPPPAGPPPPA",
    "mixed_100": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
}

# --- Offset scenarios ---
OFFSET_SCENARIOS = {
    "small": {
        "CA": 0.5,
        "CB": 0.5,
        "C": 0.3,
        "N": 0.5,
        "H": 0.05,
        "HA": 0.02,
        "HB": 0.02,
    },
    "medium": {
        "CA": 1.5,
        "CB": 1.5,
        "C": 0.8,
        "N": 1.2,
        "H": 0.10,
        "HA": 0.05,
        "HB": 0.05,
    },
    "large": {
        "CA": 3.0,
        "CB": 3.0,
        "C": 1.5,
        "N": 2.0,
        "H": 0.20,
        "HA": 0.10,
        "HB": 0.10,
    },
    "negative_medium": {
        "CA": -1.5,
        "CB": -1.5,
        "C": -0.8,
        "N": -1.2,
        "H": -0.10,
        "HA": -0.05,
        "HB": -0.05,
    },
    "mixed_sign": {
        "CA": 1.5,
        "CB": 1.5,
        "C": -0.8,
        "N": 1.2,
        "H": -0.10,
        "HA": 0.05,
        "HB": 0.05,
    },
}

NOISE_LEVELS = {
    "none": 0.0,
    "low": 0.05,  # 5% of REFINED_WEIGHTS
    "high": 0.15,  # 15% of REFINED_WEIGHTS
}


def generate_potenci_shifts(seq, temperature=298.0, pH=7.0, ion=0.1):
    """Generate POTENCI random-coil predictions and convert to arrays."""
    predshiftdct = potenci.get_pred_shifts(seq, temperature, pH, ion)

    n = len(seq)
    shifts_arr = np.full((n, 7), np.nan)
    mask = np.full((n, 7), False)

    for (res, _aa), atom_shifts in predshiftdct.items():
        i = res - 1
        for j, at in enumerate(BACKBONE_ATOMS):
            if at in atom_shifts and atom_shifts[at] is not None:
                shifts_arr[i, j] = atom_shifts[at]
                mask[i, j] = True

    return shifts_arr, mask, predshiftdct


def corrupt_shifts(shifts_arr, mask, offsets, noise_frac=0.0):
    """Add known offsets and optional Gaussian noise to shifts."""
    corrupted = shifts_arr.copy()
    for j, at in enumerate(BACKBONE_ATOMS):
        if at in offsets:
            corrupted[mask[:, j], j] += offsets[at]

    if noise_frac > 0:
        rng = np.random.default_rng(42)
        for j, at in enumerate(BACKBONE_ATOMS):
            noise_std = REFINED_WEIGHTS[at] * noise_frac
            noise = rng.normal(0, noise_std, size=corrupted.shape[0])
            corrupted[mask[:, j], j] += noise[mask[:, j]]

    return corrupted


def run_lacs(seq, corrupted_arr, mask):
    """Run LACS on corrupted shifts."""
    seq_nums = np.arange(1, len(seq) + 1)
    obs_shifts = {}
    for j, at in enumerate(BACKBONE_ATOMS):
        if at == "HB":
            continue  # LACS doesn't take HB directly (derives from HA)
        arr = np.full(len(seq), np.nan)
        arr[mask[:, j]] = corrupted_arr[mask[:, j], j]
        obs_shifts[at] = arr

    result = compute_lacs_offsets(seq, seq_nums, obs_shifts)
    if result is None:
        return dict.fromkeys(BACKBONE_ATOMS, None)
    return {at: result.get(at) for at in BACKBONE_ATOMS}


def run_trizod_global(corrupted_arr, mask, predshiftdct):
    """Run TriZOD's global offset correction."""
    # Build bbshifts_arr and bbshifts_mask from corrupted data
    diff_arr = corrupted_arr.copy()
    predshift_arr = np.zeros_like(corrupted_arr)
    predshift_mask = np.full_like(mask, False)

    for (res, _aa), atom_shifts in predshiftdct.items():
        i = res - 1
        for j, at in enumerate(BACKBONE_ATOMS):
            if at in atom_shifts and atom_shifts[at] is not None:
                predshift_arr[i, j] = atom_shifts[at]
                predshift_mask[i, j] = True

    cmp_mask = mask & predshift_mask
    np.subtract(corrupted_arr, predshift_arr, where=cmp_mask, out=diff_arr)

    weights = np.array([REFINED_WEIGHTS[at] for at in BACKBONE_ATOMS])
    weighted_diffs = diff_arr / weights

    # Initial Z-scores for outlier detection
    _, abs_wd = compute_weighted_diffs(
        diff_arr, cmp_mask, dict.fromkeys(BACKBONE_ATOMS, 0.0)
    )
    zscores = compute_zscores(abs_wd, cmp_mask.sum(axis=1), cmp_mask)
    zt, kt = convert_to_triplet_data(abs_wd, cmp_mask)
    zscores_t = compute_zscores(zt, kt, cmp_mask)
    outlier_mask = get_outlier_mask(
        zscores_t, zscores, abs_wd, cmp_mask, cdf_threshold=6.0
    )

    offsets = compute_offsets(weighted_diffs, cmp_mask & ~outlier_mask, min_AIC=6.0)
    # Convert back from weighted to ppm
    return {at: offsets[at] * REFINED_WEIGHTS[at] for at in BACKBONE_ATOMS}


def run_trizod_rolling(diff_arr_raw, mask, predshiftdct, corrupted_arr):
    """Run TriZOD's rolling window offset correction."""
    diff_arr = corrupted_arr.copy()
    predshift_arr = np.zeros_like(corrupted_arr)
    predshift_mask = np.full_like(mask, False)

    for (res, _aa), atom_shifts in predshiftdct.items():
        i = res - 1
        for j, at in enumerate(BACKBONE_ATOMS):
            if at in atom_shifts and atom_shifts[at] is not None:
                predshift_arr[i, j] = atom_shifts[at]
                predshift_mask[i, j] = True

    cmp_mask = mask & predshift_mask
    np.subtract(corrupted_arr, predshift_arr, where=cmp_mask, out=diff_arr)

    result = compute_running_offsets(diff_arr, cmp_mask, min_AIC=6.0)
    if result is None:
        return dict.fromkeys(BACKBONE_ATOMS, None)
    # Running offsets are already in weighted space, convert to ppm
    return {at: result.get(at, 0.0) * REFINED_WEIGHTS[at] for at in BACKBONE_ATOMS}


def evaluate_recovery(true_offsets, recovered_offsets):
    """Compare recovered offsets against true offsets."""
    results = {}
    for at in BACKBONE_ATOMS:
        true = true_offsets.get(at, 0.0)
        recovered = recovered_offsets.get(at)
        if recovered is None:
            results[at] = {
                "true": true,
                "recovered": None,
                "error": None,
                "detected": False,
            }
        else:
            error = recovered - true
            detected = abs(error) < max(0.5, abs(true) * 0.3)  # within 0.5 ppm or 30%
            results[at] = {
                "true": true,
                "recovered": recovered,
                "error": error,
                "detected": detected,
            }
    return results


def run_benchmark():
    """Run full benchmark across all sequences, offsets, and noise levels."""
    all_results = []

    for seq_name, seq in SEQUENCES.items():
        logging.warning(f"Processing {seq_name} (len={len(seq)})...")

        # Generate POTENCI predictions
        shifts_arr, mask, predshiftdct = generate_potenci_shifts(seq)

        for offset_name, offsets in OFFSET_SCENARIOS.items():
            for noise_name, noise_frac in NOISE_LEVELS.items():
                corrupted = corrupt_shifts(shifts_arr, mask, offsets, noise_frac)

                # Run each method
                lacs_offsets = run_lacs(seq, corrupted, mask)
                trizod_global = run_trizod_global(corrupted, mask, predshiftdct)
                trizod_rolling = run_trizod_rolling(
                    shifts_arr, mask, predshiftdct, corrupted
                )

                for method_name, method_offsets in [
                    ("LACS", lacs_offsets),
                    ("TriZOD-global", trizod_global),
                    ("TriZOD-rolling", trizod_rolling),
                ]:
                    eval_result = evaluate_recovery(offsets, method_offsets)
                    for at, r in eval_result.items():
                        all_results.append(
                            {
                                "sequence": seq_name,
                                "seq_len": len(seq),
                                "offset_scenario": offset_name,
                                "noise": noise_name,
                                "method": method_name,
                                "atom": at,
                                **r,
                            }
                        )

    return all_results


def format_summary(results):
    """Create summary tables from benchmark results."""
    lines = [
        "# Re-Referencing Benchmark Results",
        "",
        "Synthetic benchmark following Reid Alderson's approach (2026-04-01):",
        "generate ground-truth shifts with POTENCI, corrupt with known offsets,",
        "run each method, compare recovery accuracy.",
        "",
        f"- {len(SEQUENCES)} test sequences (lengths {min(len(s) for s in SEQUENCES.values())}-{max(len(s) for s in SEQUENCES.values())})",
        f"- {len(OFFSET_SCENARIOS)} offset scenarios (small/medium/large/negative/mixed)",
        f"- {len(NOISE_LEVELS)} noise levels (none/low/high)",
        "",
    ]

    # Summary by method and atom type: mean absolute error
    methods = sorted({r["method"] for r in results})
    atoms = BACKBONE_ATOMS

    lines.append("## Mean Absolute Error (ppm) by method and atom type")
    lines.append("")
    lines.append("Averaged across all sequences, offset scenarios, and noise levels.")
    lines.append("Lower is better.")
    lines.append("")
    header = "| Method | " + " | ".join(atoms) + " | Mean |"
    sep = "| ------ | " + " | ".join(["----:"] * len(atoms)) + " | ---: |"
    lines.extend([header, sep])

    for method in methods:
        method_results = [r for r in results if r["method"] == method]
        row = f"| {method} |"
        errors = []
        for at in atoms:
            at_results = [
                r for r in method_results if r["atom"] == at and r["error"] is not None
            ]
            if at_results:
                mae = np.mean([abs(r["error"]) for r in at_results])
                errors.append(mae)
                row += f" {mae:.3f} |"
            else:
                row += " N/A |"
        if errors:
            row += f" {np.mean(errors):.3f} |"
        else:
            row += " N/A |"
        lines.append(row)

    # Detection rate by method and offset magnitude
    lines.extend(["", "## Detection Rate by method and offset magnitude", ""])
    lines.append("Fraction of cases where the recovered offset is within 0.5 ppm")
    lines.append("(or 30%) of the true offset. Higher is better.")
    lines.append("")
    header = "| Method | " + " | ".join(OFFSET_SCENARIOS.keys()) + " | Overall |"
    sep = "| ------ | " + " | ".join(["----:"] * len(OFFSET_SCENARIOS)) + " | ------: |"
    lines.extend([header, sep])

    for method in methods:
        method_results = [r for r in results if r["method"] == method]
        row = f"| {method} |"
        rates = []
        for scenario in OFFSET_SCENARIOS:
            sc_results = [
                r
                for r in method_results
                if r["offset_scenario"] == scenario and r["detected"] is not None
            ]
            if sc_results:
                rate = np.mean([r["detected"] for r in sc_results])
                rates.append(rate)
                row += f" {rate:.1%} |"
            else:
                row += " N/A |"
        if rates:
            row += f" {np.mean(rates):.1%} |"
        else:
            row += " N/A |"
        lines.append(row)

    # Effect of noise
    lines.extend(["", "## Effect of noise on recovery (mean absolute error, ppm)", ""])
    header = "| Method | " + " | ".join(NOISE_LEVELS.keys()) + " |"
    sep = "| ------ | " + " | ".join(["----:"] * len(NOISE_LEVELS)) + " |"
    lines.extend([header, sep])

    for method in methods:
        method_results = [r for r in results if r["method"] == method]
        row = f"| {method} |"
        for noise in NOISE_LEVELS:
            nr = [
                r
                for r in method_results
                if r["noise"] == noise and r["error"] is not None
            ]
            if nr:
                mae = np.mean([abs(r["error"]) for r in nr])
                row += f" {mae:.3f} |"
            else:
                row += " N/A |"
        lines.append(row)

    # Per-sequence performance (mean absolute error across all conditions)
    lines.extend(
        [
            "",
            "## Per-sequence performance (mean absolute error across all conditions)",
            "",
        ]
    )
    header = "| Sequence | Len | " + " | ".join(methods) + " |"
    sep = "| -------- | --: | " + " | ".join(["----:"] * len(methods)) + " |"
    lines.extend([header, sep])

    for seq_name in SEQUENCES:
        seq_len = len(SEQUENCES[seq_name])
        row = f"| {seq_name} | {seq_len} |"
        for method in methods:
            mr = [
                r
                for r in results
                if r["sequence"] == seq_name
                and r["method"] == method
                and r["error"] is not None
            ]
            if mr:
                mae = np.mean([abs(r["error"]) for r in mr])
                row += f" {mae:.3f} |"
            else:
                row += " N/A |"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark re-referencing methods with synthetic data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write markdown report to this file",
    )
    args = parser.parse_args()

    results = run_benchmark()
    report = format_summary(results)
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        logging.warning(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
