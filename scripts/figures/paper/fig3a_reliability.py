#!/usr/bin/env python3
"""Fig. 4c -- quality tiers trade size for agreement between replicates.

Fig. 4c (``fig4c.pdf``, :func:`draw_plane`; the right half of Fig. 3a until 12 Nov 2026): one
point per CUMULATIVE tier (a chain belongs to every tier at which it is scored) on a size x
agreement plane. x, the chains scored at the tier (the bundle's per-tier ``scores.json``,
log axis); y, the residue-level Pearson r of G between two non-bound chains of identical
sequence from different entries, pooled over residues, with its 95 % interval from 1,000
bootstrap resamples of sequences. Only non-bound chains are used, every chain carries its own
tier's labels (``scores/<tier>/scores.json``), <= 20 seeded pairs per sequence and tier, and a
pair needs >= 10 co-scored residues. Definitions are in ``fig3_replicates.py``. Each point is
labelled with its tier and the number of replicate pairs behind its r (18 Nov 2026, figure
review: larger markers, labels clear of every marker and CI bar, no joining line).

The leave-one-entry-out deviation (the left half of the old Fig. 3a) moves to Supplementary
Fig. 7a; :func:`draw` (the old two-measure dot plot) is kept for reuse and is no longer
written. The statistics the panel does not draw (the deviation and its k strata, the share of
chains deviating by > 0.1, the flip rate, r re-weighted to the unfiltered G distribution, the
training set sizes) are still written to ``fig3_data`` and ``fig3a_numbers.json``.
``supp_s7_replicates.py`` imports ``XLABEL_DEV`` and ``XLABEL_R`` from this module.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig3a_reliability.py \\
        --out-dir <draft>/Figures/paper
Outputs: <out-dir>/fig4c.pdf (78 x 55 mm, placed at 1:1); <out-dir>/source_data/fig4c.csv;
<out-dir>/fig3_data/fig3a_*.csv and fig3a_numbers.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig3_replicates as fr  # noqa: E402
import style_paper as sp  # noqa: E402

# --- shared labels and marker style ---------------------------------------------------
# One x label per measure, identical in a and b (and in Supplementary Fig. S7a, e): in a
# the difference is taken from the consensus of a chain's replicates, in b between the two
# chains of a pair; the caption says which.
XLABEL_DEV = "Difference from\nreplicates (|ΔG|)"
XLABEL_R = "Replicate\ncorrelation ($\\mathit{r}$)"
EDGE = "#262626"  # dark marker edge keeps the pale unfiltered tier visible
CI_COLOR = "#4D4D4D"


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


# --- Fig. 4c: size x agreement plane ------------------------------------------------
W_C, H_C = sp.W_SINGLE, 55.0  # mm, one cell of the 2 x 2 Fig. 4 grid
AX_C = (13.0, 9.5, 61.0, 41.5)  # x0, y0, w, h (mm); y0 shared with Fig. 4a, d
XLIM_C = (3000, 30000)  # one decade; leaves room right of the unfiltered point
XT_C = [5000, 10000, 20000]
# y starts at 0.6, not 0: a dot plot with intervals (position, not length, encodes r)
YLIM_C, YT_C = (0.6, 1.0), [0.6, 0.7, 0.8, 0.9, 1.0]
MS_C = 7.5  # marker size, pt (5.5 until 18 Nov 2026; figure review: larger markers)
CI_LW = 1.0
# Two-line direct label per tier: the tier name, and under it the replicate pairs behind r
# (secondary ink). (dx_mm, dy_mm, ha, va of the name) from the point, clear of every marker
# and CI bar: va "bottom" with dy ~0 centres the two lines on the point; tolerant sits lower
# so it stays below the moderate CI bar to its upper left. No line joins the points since
# 18 Nov 2026: it ran through the strict and moderate labels.
LABEL_AT = {
    "unfiltered": (2.6, 0.1, "left", "bottom"),
    "tolerant": (-2.6, 1.0, "right", "top"),
    "moderate": (-2.6, 0.1, "right", "bottom"),
    "strict": (2.6, 0.1, "left", "bottom"),
}


def tier_sizes(D) -> dict:
    """Chains scored at each cumulative tier (the bundle's per-tier scores.json)."""
    return {t: len(D.tier_ids[t]) for t in sp.TIERS}


def draw_plane(R: pd.DataFrame, sizes: dict):
    """Fig. 4c: chains per tier (log x) against replicate r with its 95 % CI (y).

    Each point is labelled with its tier and, under it, the number of replicate pairs behind
    its r (``n_pairs``), so the panel shows both sizes: chains (x) and pairs (label)."""
    from matplotlib import ticker as mticker

    fig = sp.figure(W_C, H_C)
    x0, y0, w, h = AX_C
    ax = fig.add_axes([x0 / W_C, y0 / H_C, w / W_C, h / H_C])
    R = R.set_index("tier").loc[sp.TIERS]
    xs = np.array([sizes[t] for t in sp.TIERS], float)
    for t, x in zip(sp.TIERS, xs):
        r = R.loc[t]
        ax.plot([x, x], [r.r_lo, r.r_hi], color=CI_COLOR, lw=CI_LW, solid_capstyle="butt", zorder=2)  # fmt: skip
        ax.plot(x, r.r, ls="", marker=sp.TIER_MARKERS[t], ms=MS_C, mfc=sp.TIER_COLORS[t],
                mec=EDGE, mew=0.6, zorder=3)  # fmt: skip
        dx, dy, ha, va = LABEL_AT[t]
        # tier name in lower case, the pairs behind r under it
        name = sp.direct_label(ax, x, r.r, t, dx_mm=dx, dy_mm=dy, ha=ha, va=va)
        ax.annotate(f"{int(r.n_pairs):,} pairs", xy=({"left": 0, "right": 1}[ha], 0),
                    xycoords=name, xytext=(0, -0.3), textcoords="offset points", ha=ha,
                    va="top", fontsize=sp.FS_BODY, color=sp.INK_SOFT,
                    annotation_clip=False)  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(*XLIM_C)
    ax.xaxis.set_major_locator(mticker.FixedLocator(XT_C))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(*YLIM_C)
    ax.set_yticks(YT_C)
    ax.set_yticklabels([f"{v:g}" for v in YT_C])
    # no grid: the dots carry their own CIs (figure review item 15)
    ax.set_xlabel("Chains in the tier (log scale)")
    ax.set_ylabel("Agreement between replicates, $\\mathit{r}$")
    sp.fig_label(fig, "c", 0.0, 0.0)
    return fig


def main(argv=None) -> None:
    sp.parse_args(sp.cli(__doc__), argv)
    # italic r in the text face (set here, not at import: supp_s7_replicates imports this module)
    sp.plt.rcParams.update(
        {
            "mathtext.fontset": "custom",
            "mathtext.it": "Arial:italic",
            "mathtext.rm": "Arial",
        }
    )
    reps = n_reps()
    D, L, S, P, R = compute()
    out = sp.out_dir("fig3_data")
    out.mkdir(parents=True, exist_ok=True)
    L.to_csv(out / "fig3a1_loo_chains.csv", index=False)
    S.to_csv(out / "fig3a1_loo_summary.csv", index=False)
    P.drop(columns=["hist"]).to_csv(out / "fig3a2_pairs.csv", index=False)
    R.to_csv(out / "fig3a2_r_summary.csv", index=False)
    sizes = tier_sizes(D)
    src = sp.out_dir("source_data")
    src.mkdir(parents=True, exist_ok=True)
    R.assign(chains_in_tier=R["tier"].map(sizes), sqrt_r=np.sqrt(R["r"]))[
        [
            "tier",
            "chains_in_tier",
            "n_pairs",
            "n_sequences",
            "r",
            "r_lo",
            "r_hi",
            "sqrt_r",
            "r_gmatched",
        ]
    ].to_csv(src / "fig4c.csv", index=False, float_format="%.4f")
    pdf = sp.save(draw_plane(R.copy(), sizes), "fig4c")

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
        "chains_in_tier": sizes,
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
