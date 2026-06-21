#!/usr/bin/env python3
"""Empirical analysis of the max-offset filter.

The pipeline filter ``--max-offset`` rejects (or partially masks) entries
whose per-atom POTENCI/AIC residual offset exceeds a threshold:
  unfiltered  ∞,  tolerant 3.0 ppm,  moderate 3.0 ppm,  strict 2.0 ppm.

This script loads the released ``data/release/<tier>/scores.json`` files
(written with --rereference-mode both, so LACS pre-correction has already
been applied) and asks:

  1. What does the POTENCI residual offset distribution look like AFTER
     LACS for each backbone atom and each tier?
  2. How many entries WOULD have been rejected (or masked) by the
     current filter thresholds?
  3. How would relaxed thresholds (4 ppm, 5 ppm, ∞) change tier sizes?
  4. Is the residual large enough to matter, or has LACS already
     handled the worst cases?

Outputs (under docs/260520/):
  figures/max_offset_distribution.png — per-atom offset distributions
  figures/max_offset_filter_curve.png — entries removed vs threshold
  data/max_offset_summary.csv         — numeric summary
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RELEASE = ROOT / "data" / "release"
OUTDIR = ROOT / "docs" / "260520" / "figures"
DATAOUT = ROOT / "docs" / "260520" / "data"

TIERS = ["unfiltered", "tolerant", "moderate", "strict"]
COLORS = {
    "strict": "#2ca02c",
    "moderate": "#1f77b4",
    "tolerant": "#ff7f0e",
    "unfiltered": "#d62728",
}
ATOMS = ["C", "CA", "CB", "H", "HA", "HB", "N"]
CURRENT_THRESHOLDS = {
    "unfiltered": float("inf"),
    "tolerant": 3.0,
    "moderate": 3.0,
    "strict": 2.0,
}


def load_scores(tier: str) -> pd.DataFrame:
    path = RELEASE / tier / "scores.json"
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            row = {
                "ID": r["ID"],
                "entryID": r["entryID"],
                "tier": tier,
                "n_bb_types": r["bbshift_types_post"],
                "n_bb_pos": r["bbshift_positions_post"],
            }
            for a in ATOMS:
                v = r.get(f"off_{a}")
                row[f"off_{a}"] = float(v) if v is not None else np.nan
                vl = r.get(f"lacs_off_{a}")
                row[f"lacs_off_{a}"] = float(vl) if vl is not None else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df["max_potenci_off"] = df[[f"off_{a}" for a in ATOMS]].abs().max(axis=1)
    df["max_lacs_off"] = df[[f"lacs_off_{a}" for a in ATOMS]].abs().max(axis=1)
    return df


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    DATAOUT.mkdir(parents=True, exist_ok=True)

    print("Loading per-tier scores.json ...")
    frames = {t: load_scores(t) for t in TIERS}
    for t, df in frames.items():
        print(f"  {t}: {len(df)} entries")

    # ------------------------------------------------------------------
    # Per-atom offset distribution (POTENCI residual) per tier
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    for i, atom in enumerate(ATOMS):
        ax = axes[i // 4, i % 4]
        for tier in TIERS:
            vals = frames[tier][f"off_{atom}"].dropna().values
            vals = vals[np.abs(vals) > 1e-9]
            if len(vals) == 0:
                continue
            ax.hist(
                np.abs(vals),
                bins=np.linspace(0, 5, 60),
                histtype="step",
                lw=1.6,
                color=COLORS[tier],
                label=tier,
                density=True,
            )
        for thr, ls in [(2.0, "-"), (3.0, "--")]:
            ax.axvline(thr, color="gray", lw=0.8, ls=ls, alpha=0.6)
        ax.set_xlabel(f"|POTENCI offset {atom}| [ppm]")
        ax.set_ylabel("density")
        ax.set_title(f"{atom}")
        ax.set_xlim(0, 5)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[1, 3].axis("off")
    fig.suptitle(
        "POTENCI residual offsets (per atom) after LACS pre-correction.  "
        "Vertical lines: 2 ppm (strict), 3 ppm (tolerant/moderate).",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p1 = OUTDIR / "max_offset_distribution.png"
    fig.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p1}")

    # ------------------------------------------------------------------
    # Filter curve: entries removed as a function of threshold
    # ------------------------------------------------------------------
    thresholds = np.concatenate([np.linspace(0.5, 5.0, 46), [10.0, 100.0]])
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
    summary_rows = []
    for tier in TIERS:
        df = frames[tier]
        cur_thr = CURRENT_THRESHOLDS[tier]
        if np.isinf(cur_thr):
            n_passing = len(df)
        else:
            n_passing = (df["max_potenci_off"] <= cur_thr).sum()

        fractions = [(df["max_potenci_off"] <= t).mean() for t in thresholds]
        axA.plot(
            thresholds, fractions, color=COLORS[tier], lw=1.7, label=tier,
        )

        for thr in (2.0, 3.0, 4.0, 5.0, np.inf):
            n_pass = len(df) if np.isinf(thr) else (df["max_potenci_off"] <= thr).sum()
            summary_rows.append({
                "tier": tier,
                "current_threshold": cur_thr,
                "candidate_threshold": thr,
                "n_total": len(df),
                "n_pass": n_pass,
                "frac_pass": n_pass / len(df) if len(df) else 0.0,
                "currently_pass": n_passing,
            })

        # mark current threshold
        if np.isfinite(cur_thr):
            axA.axvline(cur_thr, color=COLORS[tier], lw=0.9, ls=":", alpha=0.7)

    axA.set_xlim(0, 5)
    axA.set_ylim(0, 1.01)
    axA.set_xlabel("max-offset threshold [ppm]")
    axA.set_ylabel("Fraction of entries passing")
    axA.set_title("Fraction passing filter vs threshold (per tier)")
    axA.legend(framealpha=0.9, fontsize=9)
    axA.grid(True, alpha=0.25)

    # Panel B: stacked bar of pass/fail under various thresholds
    pivot = pd.DataFrame(summary_rows)
    cand_set = [2.0, 3.0, 4.0, 5.0, np.inf]
    bar_data = {}
    for cand in cand_set:
        bar_data[cand] = []
        for tier in TIERS:
            row = pivot[(pivot["tier"] == tier) & (pivot["candidate_threshold"] == cand)].iloc[0]
            bar_data[cand].append(row["n_pass"])
    x = np.arange(len(TIERS))
    width = 0.16
    for i, cand in enumerate(cand_set):
        label = "∞" if np.isinf(cand) else f"{cand:.0f} ppm"
        axB.bar(
            x + (i - (len(cand_set) - 1) / 2) * width,
            bar_data[cand], width,
            label=f"max-offset = {label}",
        )
    axB.set_xticks(x)
    axB.set_xticklabels(TIERS)
    axB.set_ylabel("# entries passing")
    axB.set_title("Tier size under candidate max-offset thresholds")
    axB.legend(framealpha=0.9, fontsize=8)
    axB.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    p2 = OUTDIR / "max_offset_filter_curve.png"
    fig.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p2}")

    # Numeric summary
    sum_path = DATAOUT / "max_offset_summary.csv"
    pivot.to_csv(sum_path, index=False)
    print(f"Wrote {sum_path}")

    # ------------------------------------------------------------------
    # Concise numerical summary printed to stdout
    # ------------------------------------------------------------------
    print("\n=== Per-tier POTENCI residual stats ===")
    for tier in TIERS:
        df = frames[tier]
        cur = CURRENT_THRESHOLDS[tier]
        n_total = len(df)
        if np.isinf(cur):
            print(
                f"  {tier:>10}: current max-offset = ∞ (no filter); "
                f"max POTENCI offset distribution: "
                f"median={df['max_potenci_off'].median():.3f}, "
                f"p95={df['max_potenci_off'].quantile(0.95):.3f}, "
                f"p99={df['max_potenci_off'].quantile(0.99):.3f}, "
                f"max={df['max_potenci_off'].max():.3f}"
            )
        else:
            n_pass = (df["max_potenci_off"] <= cur).sum()
            print(
                f"  {tier:>10}: current max-offset = {cur:.1f} ppm; "
                f"{n_pass}/{n_total} pass ({n_pass/n_total:.1%}); "
                f"median={df['max_potenci_off'].median():.3f}, "
                f"p95={df['max_potenci_off'].quantile(0.95):.3f}, "
                f"max={df['max_potenci_off'].max():.3f}"
            )
        # but: entries in this released set already pass — show counts
        # against various thresholds
        for thr in (1.0, 2.0, 3.0, 5.0):
            n = (df["max_potenci_off"] > thr).sum()
            print(f"      |max offset| > {thr:.1f} ppm: {n}/{n_total} ({n/n_total:.1%})")


if __name__ == "__main__":
    main()
