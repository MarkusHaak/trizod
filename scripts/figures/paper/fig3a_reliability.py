#!/usr/bin/env python3
"""Fig. 3a -- stricter tiers agree better with replicate measurements.

A dot plot with one row per CUMULATIVE tier (a chain belongs to every tier at which it
is scored) and two measures side by side, both with 95 % intervals from 1,000
bootstrap resamples of sequences. Only non-bound chains are used, and every chain
carries its own tier's labels (``scores/<tier>/scores.json``). Definitions are in
``fig3_replicates.py``:

* left -- leave-one-entry-out deviation. For each non-bound chain whose sequence has
  non-bound chains in >= 3 BMRB entries: mean |G - consensus| over residues, where the
  consensus is the per-residue median of the replicates from the OTHER entries; the
  median over chains is plotted.
* right -- residue-level Pearson r of G between two non-bound chains of identical
  sequence from different entries, pooled over residues; <= 20 seeded pairs per
  sequence and tier.

The x scales are shared with Fig. 3b (``fig3b_conditions.py``: ``XLIM_DEV``,
``XLIM_R``), so the tier effect and the condition effect can be compared by eye. The
statistics the panel no longer draws (the k strata, the share of chains deviating by
> 0.1, the flip rate, r re-weighted to the unfiltered G distribution, the training
set sizes) are still written to ``fig3_data`` and ``fig3a_numbers.json``; Supplementary
Fig. S7 shows the k strata.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig3a_reliability.py \
        --out-dir <draft>/Figures/paper
Outputs: <out-dir>/fig3a.pdf; <out-dir>/fig3_data/fig3a_*.csv and fig3a_numbers.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig3_replicates as fr  # noqa: E402
import style_paper as sp  # noqa: E402

# --- layout shared with fig3b_conditions.py (keep the two in step) -------------------
# a and b are half-width cells (W_SINGLE) set side by side in Fig. 3 at 1:1.
W, H = sp.W_SINGLE, 38.0  # mm
AX_X = (15.0, 48.5)  # left edge of the two axes, mm
AX_W = 26.0  # width of each axes, mm
AX_Y0, AX_H = 11.0, 22.5  # bottom and height of the axes, mm
XLIM_DEV, XT_DEV = (0.0, 0.09), [0, 0.03, 0.06, 0.09]
XLIM_R, XT_R = (0.4, 1.0), [0.4, 0.6, 0.8, 1.0]
# One x label per measure, identical in a and b (and in Supplementary Fig. S7a, e): in a
# the difference is taken from the consensus of a chain's replicates, in b between the two
# chains of a pair; the caption says which.
XLABEL_DEV = "Difference from\nreplicates (|ΔG|)"
XLABEL_R = "Replicate\ncorrelation ($\\mathit{r}$)"
MS = 5.0  # marker size, pt
EDGE = "#262626"  # dark marker edge keeps the pale unfiltered tier visible
CI_COLOR = "#4D4D4D"
GRID = "#E6E6E6"


def n_reps() -> dict:
    return {
        t: len(sp.read_fasta(sp.BUNDLE / "train" / t / f"train_{t}_best.fasta"))
        for t in sp.TIERS
    }


def compute():
    D = fr.Data()
    L = fr.loo_table(D)
    S = fr.summarise_loo(L)
    P = fr.pair_table(D)
    ref = P[P.tier == "unfiltered"]
    R = []
    for t in sp.TIERS:
        sub = P[P.tier == t]
        b = fr.boot_pooled(sub)
        b["tier"] = t
        b["r_gmatched"] = fr.variance_matched_r(sub, ref)
        b["median_pair_absdg"] = float(sub["pair_mean_absdg"].median())
        R.append(b)
    R = pd.DataFrame(R)
    return D, L, S, P, R


def dot_axes(fig, n_rows: int, labels: list[str]):
    """The two side-by-side dot-plot axes of Fig. 3a/3b: rows top to bottom, shared y."""
    axes = []
    for i, x0 in enumerate(AX_X):
        ax = fig.add_axes(
            [x0 / W, AX_Y0 / H, AX_W / W, AX_H / H], sharey=axes[0] if axes else None
        )
        ax.set_ylim(n_rows - 0.4, -0.6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(range(n_rows))
        if i == 0:
            ax.set_yticklabels(labels, fontsize=sp.FS_BODY)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        axes.append(ax)
    axes[0].set_xlim(*XLIM_DEV)
    axes[0].set_xticks(XT_DEV)
    axes[0].set_xticklabels(["0", "0.03", "0.06", "0.09"])
    axes[1].set_xlim(*XLIM_R)
    axes[1].set_xticks(XT_R)
    return axes


def dot(ax, x, lo, hi, y, color, marker="o"):
    """One estimate with its 95 % interval."""
    ax.plot([lo, hi], [y, y], color=CI_COLOR, lw=0.8, solid_capstyle="butt", zorder=2)
    ax.plot(x, y, ls="", marker=marker, ms=MS, mfc=color, mec=EDGE, mew=0.5, zorder=3)


def draw(S: pd.DataFrame, R: pd.DataFrame):
    fig = sp.figure(W, H)
    ax_d, ax_r = dot_axes(fig, len(sp.TIERS), sp.TIERS)
    s = S[S.kbin == "all"].set_index("tier").loc[sp.TIERS]
    R = R.set_index("tier").loc[sp.TIERS]
    for y, t in enumerate(sp.TIERS):
        c = sp.TIER_COLORS[t]
        dot(
            ax_d,
            s.loc[t, "median_dev"],
            s.loc[t, "median_dev_lo"],
            s.loc[t, "median_dev_hi"],
            y,
            c,
        )
        dot(ax_r, R.loc[t, "r"], R.loc[t, "r_lo"], R.loc[t, "r_hi"], y, c)  # fmt: skip
    ax_d.set_xlabel(XLABEL_DEV)
    ax_r.set_xlabel(XLABEL_R)
    sp.fig_label(fig, "a", 0.0, 0.0)
    return fig


def main(argv=None) -> None:
    sp.parse_args(sp.cli(__doc__), argv)
    reps = n_reps()
    D, L, S, P, R = compute()
    out = sp.out_dir("fig3_data")
    out.mkdir(parents=True, exist_ok=True)
    L.to_csv(out / "fig3a1_loo_chains.csv", index=False)
    S.to_csv(out / "fig3a1_loo_summary.csv", index=False)
    P.drop(columns=["hist"]).to_csv(out / "fig3a2_pairs.csv", index=False)
    R.to_csv(out / "fig3a2_r_summary.csv", index=False)
    fig = draw(S, R.copy())
    pdf = sp.save(fig, "fig3a")

    s_all = S[S.kbin == "all"].set_index("tier")
    Ri = R.set_index("tier")
    nums = {
        "definition": {
            "scope": "non-bound chains only; tier-specific labels from scores/<tier>/scores.json; cumulative tiers",
            "a1": "per chain: mean |G_tier - median(G_unfiltered of non-bound replicates from OTHER entries)| over residues "
            "with >= 2 replicate values; sequences with non-bound chains in >= 3 entries; >= 5 comparable residues; "
            "median over chains; 95 % CI from 1,000 sequence bootstraps (seed 0)",
            "a1_kbins": "chain median k over comparable residues (unfiltered k): <= 12, 13-17, 18-21",
            "a2": "pooled residue-level Pearson r of G between two non-bound chains with identical sequence from different "
            "entries, both scored at the tier (tier labels); <= 20 seeded pairs per sequence and tier; pairs need >= 10 "
            "co-scored residues; 95 % CI from 1,000 sequence bootstraps",
            "r_gmatched": "pooled r after re-weighting pairs so the residue pair-mean-G histogram (10 bins) matches the unfiltered tier",
            "figure": "panel a draws the 'all' rows of a1 (left) and r of a2 (right); the k strata, the >0.1 share, "
            "the flip rate, r_gmatched and training_reps are data only",
        },
        "training_reps": reps,
        "tolerant_label_chains_differing_from_unfiltered": fr.label_differences(D)[
            "tolerant"
        ],
        "a1": {
            t: {
                "n_chains": int(s_all.loc[t, "n_chains"]),
                "n_sequences": int(s_all.loc[t, "n_sequences"]),
                "median_dev": round(s_all.loc[t, "median_dev"], 4),
                "median_dev_ci": [
                    round(s_all.loc[t, "median_dev_lo"], 4),
                    round(s_all.loc[t, "median_dev_hi"], 4),
                ],
                "pct_dev_gt_0.1": round(100 * s_all.loc[t, "share_dev_gt_0.1"], 1),
                "pct_flip": round(100 * s_all.loc[t, "flip"], 1),
            }
            for t in sp.TIERS
        },
        "a1_by_k": {
            kb: {
                t: {
                    "n_chains": int(r.n_chains),
                    "median_dev": round(r.median_dev, 4),
                    "median_dev_ci": [
                        round(r.median_dev_lo, 4),
                        round(r.median_dev_hi, 4),
                    ],
                }
                for t, r in S[S.kbin == kb].set_index("tier").iterrows()
            }
            for kb in fr.K_LABELS
        },
        "a1_relative_reduction_unfiltered_to_strict_pct": round(
            100
            * (
                1
                - s_all.loc["strict", "median_dev"]
                / s_all.loc["unfiltered", "median_dev"]
            ),
            0,
        ),
        "a2": {
            t: {
                "n_pairs": int(Ri.loc[t, "n_pairs"]),
                "n_sequences": int(Ri.loc[t, "n_sequences"]),
                "n_residues": int(Ri.loc[t, "n_residues"]),
                "r": round(Ri.loc[t, "r"], 3),
                "r_ci": [round(Ri.loc[t, "r_lo"], 3), round(Ri.loc[t, "r_hi"], 3)],
                "r_gmatched": round(Ri.loc[t, "r_gmatched"], 3),
                "mae": round(Ri.loc[t, "mae"], 3),
                "mae_ci": [
                    round(Ri.loc[t, "mae_lo"], 3),
                    round(Ri.loc[t, "mae_hi"], 3),
                ],
                "pct_flip": round(100 * Ri.loc[t, "flip"], 1),
                "median_pair_absdg": round(Ri.loc[t, "median_pair_absdg"], 4),
            }
            for t in sp.TIERS
        },
    }
    (out / "fig3a_numbers.json").write_text(
        json.dumps(nums, indent=1, ensure_ascii=False)
    )
    print(json.dumps(nums, indent=1, ensure_ascii=False))
    print(f"wrote {pdf}", file=sys.stderr)


if __name__ == "__main__":
    main()
