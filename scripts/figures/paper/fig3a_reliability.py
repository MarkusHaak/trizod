#!/usr/bin/env python3
"""Fig. 3a -- reliability against size across the cumulative tiers.

The x axis of both plots is the number of training representatives a user gets at
each tier (``train/<tier>/train_<tier>_best.fasta``; frozen 2026-10 split, so these
counts move with rebuild fix (a)). Tiers are CUMULATIVE: a chain belongs to every tier
at which it is scored. Only non-bound chains are used, and every chain carries its own
tier's labels (``scores/<tier>/scores.json``). Definitions are in
``fig3_replicates.py``:

(a1) Leave-one-entry-out replicate deviation. For each non-bound chain whose sequence
     has non-bound chains in >= 3 BMRB entries: mean |G - consensus| over residues,
     where the consensus is the per-residue median of the replicates from the OTHER
     entries. Median over chains with a 95 % CI from resampling sequences; the same
     median within strata of the chain's median shift count k (<= 12, 13-17, 18-21).
     The table under the plots gives the share of chains deviating by > 0.1 and the
     mean flip rate at G = 0.65.
(a2) Residue-level Pearson r of G between two non-bound chains of identical sequence
     from different entries, pooled over residues; <= 20 seeded pairs per sequence and
     tier; 95 % CI from resampling sequences. Open markers: the same r after
     re-weighting pairs so that the residue G distribution matches the unfiltered tier
     (a check that a higher r is not a narrower G range).

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

FS_SMALL = 5.0
K_CENTRES = {"k ≤ 12": 6.5, "k 13–17": 15.0, "k 18–21": 19.5}
K_DODGE = {"k ≤ 12": -0.07, "k 13–17": 0.0, "k 18–21": 0.07}  # thousands of reps


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


def draw(S: pd.DataFrame, R: pd.DataFrame, reps: dict):
    from matplotlib.lines import Line2D

    fig = sp.figure(sp.W_SINGLE, 72.0)
    W, H = sp.W_SINGLE, 72.0

    def ax_mm(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    xk = {t: reps[t] / 1000 for t in sp.TIERS}
    xlim = (1.3, 6.4)

    # ------------------------------------------------------------------ a1
    ax = ax_mm(11.5, 27.0, 36.0, 37.0)
    for kb in fr.K_LABELS:
        s = S[S.kbin == kb].set_index("tier").loc[sp.TIERS]
        col = sp.k_color(K_CENTRES[kb])
        xs = [xk[t] + K_DODGE[kb] for t in sp.TIERS]
        ax.plot(xs, s.median_dev, color=col, lw=0.6, ls="--", zorder=2)
        ax.scatter(xs, s.median_dev, s=6, color=col, zorder=3, lw=0)
    s = S[S.kbin == "all"].set_index("tier").loc[sp.TIERS]
    ax.plot(
        [xk[t] for t in sp.TIERS],
        s.median_dev,
        color=sp.CAT["neutral"],
        lw=0.9,
        zorder=4,
    )
    for t in sp.TIERS:
        r = s.loc[t]
        ax.errorbar(
            xk[t], r.median_dev,
            yerr=[[r.median_dev - r.median_dev_lo], [r.median_dev_hi - r.median_dev]],
            fmt=sp.TIER_MARKERS[t], ms=3.6, color=sp.TIER_COLORS[t], mec="white", mew=0.3,
            elinewidth=0.6, capsize=0, zorder=5,
        )  # fmt: skip
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 0.075)
    ax.set_yticks([0, 0.02, 0.04, 0.06])
    ax.set_xlabel("Training representatives (×1,000)")
    ax.set_ylabel("Median deviation from replicates (|ΔG|)")
    h = [Line2D([], [], color=sp.CAT["neutral"], lw=0.9, label="all chains (95 % CI)")]
    h += [
        Line2D([], [], color=sp.k_color(K_CENTRES[kb]), lw=0.6, ls="--", marker="o", ms=2,
               label=f"chains with median {kb}")
        for kb in fr.K_LABELS
    ]  # fmt: skip
    ax.legend(handles=h, loc="lower right", handlelength=1.8, handletextpad=0.4,
              labelspacing=0.25, borderaxespad=0.2, fontsize=FS_SMALL)  # fmt: skip
    ax.text(
        -0.30,
        1.04,
        "a1",
        transform=ax.transAxes,
        fontsize=6.5,
        fontweight="bold",
        va="bottom",
    )
    ax.set_title("Leave-one-entry-out deviation", fontsize=6, loc="left", pad=2)

    # ------------------------------------------------------------------ a2
    ax2 = ax_mm(59.5, 27.0, 26.0, 37.0)
    R = R.set_index("tier").loc[sp.TIERS]
    ax2.plot([xk[t] for t in sp.TIERS], R.r, color=sp.CAT["neutral"], lw=0.9, zorder=4)
    for t in sp.TIERS:
        r = R.loc[t]
        ax2.errorbar(
            xk[t], r.r, yerr=[[r.r - r.r_lo], [r.r_hi - r.r]], fmt=sp.TIER_MARKERS[t], ms=3.6,
            color=sp.TIER_COLORS[t], mec="white", mew=0.3, elinewidth=0.6, capsize=0, zorder=5,
        )  # fmt: skip
        if t == "unfiltered":
            continue  # the reference distribution itself
        ax2.scatter(xk[t] + 0.22, r.r_gmatched, s=9, marker=sp.TIER_MARKERS[t], facecolor="none",
                    edgecolor=sp.TIER_COLORS[t], lw=0.5, zorder=5)  # fmt: skip
    ax2.set_xlim(*xlim)
    ax2.set_ylim(0.6, 1.0)
    ax2.set_xlabel("Training representatives (×1,000)")
    ax2.set_ylabel("Replicate Pearson r (residues)")
    h2 = [
        Line2D([], [], color=sp.CAT["neutral"], lw=0.9, marker="o", ms=2.5, label="pooled r (95 % CI)"),
        Line2D([], [], ls="", marker="o", ms=2.8, mfc="none", mec=sp.CAT["neutral"], mew=0.5,
               label="G range matched\nto unfiltered"),
    ]  # fmt: skip
    ax2.legend(handles=h2, loc="lower left", handlelength=1.4, handletextpad=0.4,
               labelspacing=0.3, borderaxespad=0.2, fontsize=FS_SMALL)  # fmt: skip
    ax2.text(
        -0.40,
        1.04,
        "a2",
        transform=ax2.transAxes,
        fontsize=6.5,
        fontweight="bold",
        va="bottom",
    )
    ax2.set_title("Replicate correlation", fontsize=6, loc="left", pad=2)

    # ------------------------------------------------------------------ table
    s_all = S[S.kbin == "all"].set_index("tier")
    rows = [
        ("Training representatives", [f"{reps[t]:,}" for t in sp.TIERS]),
        ("Chains in a1", [f"{int(s_all.loc[t, 'n_chains']):,}" for t in sp.TIERS]),
        ("Chains deviating > 0.1", [f"{100 * s_all.loc[t, 'share_dev_gt_0.1']:.1f} %" for t in sp.TIERS]),
        ("Flip rate at G = 0.65", [f"{100 * s_all.loc[t, 'flip']:.1f} %" for t in sp.TIERS]),
        ("Chain pairs in a2", [f"{int(R.loc[t, 'n_pairs']):,}" for t in sp.TIERS]),
    ]  # fmt: skip
    x0_lab, x_cols = 2.0, [43.0, 55.0, 67.0, 79.0]
    y_top = 15.6
    dy = 2.55
    renderer = fig.canvas.get_renderer()
    for t, xc in zip(sp.TIERS, x_cols):
        txt = fig.text((xc + 1.0) / W, y_top / H, t, ha="center", va="center", fontsize=FS_SMALL + 0.3,
                       color=sp.TIER_COLORS[t], fontweight="bold")  # fmt: skip
        half_w = txt.get_window_extent(renderer).width / fig.dpi / sp.MM / 2
        fig.add_artist(
            Line2D([(xc + 1.0 - half_w - 1.6) / W], [y_top / H], marker=sp.TIER_MARKERS[t], ms=3.0,
                   color=sp.TIER_COLORS[t], mec="white", mew=0.3, ls="")
        )  # fmt: skip
    fig.add_artist(
        Line2D(
            [x0_lab / W, 86.0 / W], [(y_top - 1.35) / H] * 2, color="#999999", lw=0.4
        )
    )
    for i, (lab, vals) in enumerate(rows):
        y = y_top - (i + 1) * dy - 0.3
        fig.text(x0_lab / W, y / H, lab, ha="left", va="center", fontsize=FS_SMALL)
        for v, xc in zip(vals, x_cols):
            fig.text(xc / W, y / H, v, ha="center", va="center", fontsize=FS_SMALL)

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
    fig = draw(S, R.copy(), reps)
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
