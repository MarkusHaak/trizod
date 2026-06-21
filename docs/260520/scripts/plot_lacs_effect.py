#!/usr/bin/env python3
"""Visualize the effect of LACS pre-correction on TriZOD G-scores.

Uses the cached residue-level comparison pickle produced by
scripts/compare_gscores_lacs.py and the per-tier baseline JSON files
in data/baseline/ to classify each entry by stringency tier.

Outputs (under docs/260520/figures/):
  lacs_effect_gscores.png    — 4-panel summary
  lacs_effect_summary.csv    — per-tier summary table
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[3]
PKL = ROOT / "tmp" / "lacs_comparison_results.pkl"
BASELINE = ROOT / "data" / "baseline"
OUTDIR = ROOT / "docs" / "260520" / "figures"
DATA_OUT = ROOT / "docs" / "260520" / "data"

TIERS = ["strict", "moderate", "tolerant", "unfiltered"]
COLORS = {
    "strict": "#2ca02c",
    "moderate": "#1f77b4",
    "tolerant": "#ff7f0e",
    "unfiltered": "#d62728",
}


def load_tier_sets() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for tier in TIERS:
        ids: set[str] = set()
        path = BASELINE / f"{tier}.json"
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                ids.add(json.loads(line)["entryID"])
        out[tier] = ids
        print(f"  {tier}: {len(ids)} entries")
    return out


def classify_tier(entry_id: str, tier_sets: dict[str, set[str]]) -> str:
    for tier in TIERS:
        if entry_id in tier_sets[tier]:
            return tier
    return "unfiltered"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    print(f"Loading {PKL}...")
    with open(PKL, "rb") as f:
        cache = pickle.load(f)

    df = pd.DataFrame(
        cache["pairs"],
        columns=["entry_id", "g_orig", "g_lacs", "max_lacs_offset"],
    )
    df["diff"] = df["g_lacs"] - df["g_orig"]
    print(f"Loaded {len(df):,} residues from {df['entry_id'].nunique()} entries")

    print("Loading tier classifications...")
    tier_sets = load_tier_sets()
    df["tier"] = df["entry_id"].map(lambda eid: classify_tier(eid, tier_sets))

    # ------------------------------------------------------------------
    # Per-entry summary
    # ------------------------------------------------------------------
    per_entry = (
        df.groupby("entry_id")
        .agg(
            n_res=("g_orig", "size"),
            max_lacs_offset=("max_lacs_offset", "first"),
            mean_diff=("diff", "mean"),
            max_abs_diff=("diff", lambda x: x.abs().max()),
            tier=("tier", "first"),
        )
        .reset_index()
    )
    print(f"Per-entry rows: {len(per_entry)}")

    # ------------------------------------------------------------------
    # Per-tier summary CSV
    # ------------------------------------------------------------------
    summary_rows = []
    for tier in TIERS:
        sub_res = df[df["tier"] == tier]
        sub_ent = per_entry[per_entry["tier"] == tier]
        if len(sub_ent) == 0:
            continue
        n_ent = len(sub_ent)
        summary_rows.append({
            "tier": tier,
            "n_entries": n_ent,
            "n_residues": len(sub_res),
            "frac_offset>0.5ppm": (sub_ent["max_lacs_offset"] > 0.5).mean(),
            "frac_offset>1.0ppm": (sub_ent["max_lacs_offset"] > 1.0).mean(),
            "frac_offset>2.0ppm": (sub_ent["max_lacs_offset"] > 2.0).mean(),
            "frac_offset>3.0ppm": (sub_ent["max_lacs_offset"] > 3.0).mean(),
            "median_mean_gdiff": sub_ent["mean_diff"].median(),
            "median_max_abs_gdiff": sub_ent["max_abs_diff"].median(),
            "p95_max_abs_gdiff": sub_ent["max_abs_diff"].quantile(0.95),
            "frac_residues_|diff|>0.1": (sub_res["diff"].abs() > 0.1).mean(),
            "frac_residues_|diff|>0.2": (sub_res["diff"].abs() > 0.2).mean(),
            "g_score_correlation": sub_res[["g_orig", "g_lacs"]].corr().iloc[0, 1],
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DATA_OUT / "lacs_effect_summary.csv", index=False)
    print(f"Wrote {DATA_OUT / 'lacs_effect_summary.csv'}")
    print(summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 4-panel figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    text_bbox = {"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85}

    # Panel A: hexbin density gscore_orig vs gscore_lacs (all residues)
    ax = axes[0, 0]
    hb = ax.hexbin(
        df["g_orig"],
        df["g_lacs"],
        gridsize=70,
        cmap="inferno_r",
        mincnt=1,
        extent=(0, 1, 0, 1),
        linewidths=0.3,
        norm=LogNorm(),
    )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("G-score (POTENCI-only)")
    ax.set_ylabel("G-score (LACS + POTENCI)")
    ax.set_title("(A) Residue-level G-score: POTENCI-only vs LACS+POTENCI")
    corr = df[["g_orig", "g_lacs"]].corr().iloc[0, 1]
    stats = (
        f"r = {corr:.4f}\n"
        f"MAE = {df['diff'].abs().mean():.4f}\n"
        f"N = {len(df):,} residues"
    )
    ax.text(
        0.03, 0.97, stats, transform=ax.transAxes, fontsize=8.5,
        va="top", ha="left", bbox=text_bbox, family="monospace",
    )
    plt.colorbar(hb, ax=ax, label="Residue count", shrink=0.78)

    # Panel B: ΔG distribution per tier (violin)
    ax = axes[0, 1]
    box_data = []
    box_labels = []
    box_colors = []
    for tier in TIERS:
        vals = df.loc[df["tier"] == tier, "diff"].values
        if len(vals) == 0:
            continue
        box_data.append(vals)
        box_labels.append(f"{tier}\nn={len(vals):,}")
        box_colors.append(COLORS[tier])
    parts = ax.violinplot(
        box_data,
        positions=range(len(box_data)),
        showmedians=True,
        showextrema=False,
        widths=0.85,
    )
    for body, c in zip(parts["bodies"], box_colors):
        body.set_facecolor(c)
        body.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    ax.set_xticks(range(len(box_labels)))
    ax.set_xticklabels(box_labels)
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_ylim(-0.6, 0.6)
    ax.set_ylabel("ΔG-score (LACS+POTENCI − POTENCI-only)")
    ax.set_title("(B) Per-residue ΔG-score distribution by stringency tier")

    # Panel C: |ΔG| vs max LACS offset (per-entry scatter)
    ax = axes[1, 0]
    for tier in TIERS:
        sub = per_entry[per_entry["tier"] == tier]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["max_lacs_offset"],
            sub["max_abs_diff"],
            s=6,
            alpha=0.35,
            c=COLORS[tier],
            label=f"{tier} (n={len(sub)})",
        )
    ax.set_xlabel("Max |LACS offset| per entry [ppm]")
    ax.set_ylabel("Max |ΔG-score| per entry")
    ax.set_title("(C) Larger referencing errors → larger G-score effect")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.axvline(1.0, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.axvline(2.0, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.axvline(3.0, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    # Panel D: cumulative fraction of entries affected (by tier)
    ax = axes[1, 1]
    thresholds = np.linspace(0, 0.5, 200)
    for tier in TIERS:
        sub = per_entry[per_entry["tier"] == tier]
        if len(sub) == 0:
            continue
        fracs = [(sub["max_abs_diff"] >= t).mean() for t in thresholds]
        ax.plot(
            thresholds, fracs,
            color=COLORS[tier], lw=1.8,
            label=f"{tier} (n={len(sub)})",
        )
    ax.set_xlabel("Max |ΔG-score| per entry  (threshold)")
    ax.set_ylabel("Fraction of entries with max |ΔG| ≥ threshold")
    ax.set_title("(D) How many entries does LACS measurably change?")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    fig.suptitle(
        "Effect of LACS pre-correction on TriZOD G-scores "
        f"(N = {df['entry_id'].nunique():,} entries, {len(df):,} residues)",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = OUTDIR / "lacs_effect_gscores.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    # ------------------------------------------------------------------
    # Companion figure: same but only entries where LACS made a difference
    # ------------------------------------------------------------------
    affected = per_entry[per_entry["max_lacs_offset"] >= 0.5]
    fig2, ax2 = plt.subplots(figsize=(7.5, 6))
    for tier in TIERS:
        sub = affected[affected["tier"] == tier]
        if len(sub) == 0:
            continue
        ax2.scatter(
            sub["max_lacs_offset"], sub["max_abs_diff"],
            s=8, alpha=0.45, c=COLORS[tier],
            label=f"{tier} (n={len(sub)})",
        )
    ax2.set_xlabel("Max |LACS offset| per entry [ppm]")
    ax2.set_ylabel("Max |ΔG-score| per entry")
    ax2.set_title(
        "Entries with non-trivial referencing correction\n"
        f"(max LACS offset ≥ 0.5 ppm; N = {len(affected):,} entries)"
    )
    ax2.set_xlim(0.5, 6)
    ax2.set_ylim(0, 1)
    ax2.axvline(1.0, color="gray", ls=":", lw=0.7)
    ax2.axvline(2.0, color="gray", ls=":", lw=0.7)
    ax2.legend(framealpha=0.9, fontsize=9)
    fig2.tight_layout()
    out2 = OUTDIR / "lacs_effect_affected_entries.png"
    fig2.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
