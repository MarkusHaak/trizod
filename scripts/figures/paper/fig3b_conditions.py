#!/usr/bin/env python3
"""Fig. 3b -- replicate pairs measured under different conditions disagree more.

The same two-measure dot plot as Fig. 3a (layout, x scales and helpers imported from
``fig3a_reliability.py``), with one row per condition class instead of one per tier.
Replicate pairs (two non-bound chains of identical sequence from different BMRB
entries, unfiltered labels, >= 10 co-scored residues) measured under the SAME
conditions (|dpH| <= 0.2 and |dT| <= 2 K) or under DIFFERENT conditions
(|dpH| >= 1 or |dT| >= 10 K); pairs in between are dropped, and each class is capped at
20 seeded pairs per sequence. Left, the median over pairs of the per-pair mean |dG|;
right, the pooled residue-level Pearson r. Different-condition pairs are re-weighted so
that their mean |dk| distribution (bins 0, 0.5, 1, 2, 4, 8, inf) matches the
same-condition pairs (shift-count matching). 95 % CIs from 1,000 sequence bootstraps
(the matching weights are re-estimated inside every replicate). The unmatched values
and the residue MAE are written to ``fig3b_numbers.json`` only.

Supplementary Fig. S19 (``S19.pdf``; formerly sub-panel b2 of Fig. 3b): annotations of
the replicates that disagree. Chains of the Fig. 3a deviation analysis that are scored
ONLY at the unfiltered tier, split into deviant (leave-one-entry-out deviation > 0.1) and
consistent (<= 0.1) chains; dots give the share of each group carrying a flag, bars 95 %
sequence-bootstrap CIs. Flags: physical state denied at some tier by the pipeline's own
rule (``trizod.bmrb.sample_state.denied_physical_states`` with the strict deny list and
the cosolvent corroboration rule); a perturbing cosolvent named in the entry
(``cosolvent_evidence``); |POTENCI offset| > 3 sigma for any backbone atom (the tolerant
``max-offset``); solid-state sample (``sample_state_evidence``); pH or temperature
outside the tolerant range (pH 2-12, 263-333 K).

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig3b_conditions.py \
        --out-dir <draft>/Figures/paper
Outputs: <out-dir>/fig3b.pdf and S19.pdf; <out-dir>/fig3_data/fig3b_*.csv and
fig3b_numbers.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig3_replicates as fr  # noqa: E402
import fig3a_reliability as f3a  # noqa: E402
import style_paper as sp  # noqa: E402

from trizod.offsets import off_sigma_col  # noqa: E402

DEV_CUT = 0.1
C_SAME = sp.CAT["neutral"]
TOL_PH = (2.0, 12.0)
TOL_T = (263.0, 333.0)
FLAGS = [
    ("state", "Non-native physical state"),
    ("cosolvent", "Perturbing cosolvent"),
    ("offset", "Offset > 3σ (any atom)"),
    ("solid", "Solid-state sample"),
    ("phT", "pH or T out of range"),
]
COND_ROWS = [("same", "Same\nconditions"), ("different", "Different\nconditions")]
# Supplementary Fig. S19 (was the b2 sub-panel of Fig. 3b): labels of the flags and the
# styles of the two chain groups
FLAGS_S19 = [
    ("state", "Non-native physical state"),
    ("cosolvent", "Perturbing cosolvent"),
    ("offset", "Residual offset > 3σ"),
    ("solid", "Solid-state sample"),
    ("phT", "pH or temperature\nout of range"),
]
FLAG_STYLE = {
    "deviant": {
        "color": "#1A1A1A",
        "marker": "o",
        "mfc": "#1A1A1A",
        "ms": 3.8,
        "dy": -0.15,
    },
    "consistent": {
        "color": "#8C8C8C",
        "marker": "s",
        "mfc": "white",
        "ms": 3.4,
        "dy": 0.15,
    },
}


def flag_table(D: fr.Data, L: pd.DataFrame) -> pd.DataFrame:
    """Flags for the chains scored only at the unfiltered tier (a1 table, unfiltered row)."""
    from trizod.bmrb.sample_state import (
        PHYSICAL_STATE_STRICT_DENY,
        denied_physical_states,
    )

    u = L[(L.tier == "unfiltered") & (L.strictest == "unfiltered")].copy()
    m = D.nb.set_index("id").loc[u["id"]]
    cor = m["cosolvent_evidence"].fillna(False).astype(bool).to_numpy()
    u["state"] = denied_physical_states(
        m["physical_state"], PHYSICAL_STATE_STRICT_DENY, corroborated=cor
    )
    u["cosolvent"] = cor
    offs = np.abs(m[[off_sigma_col(a) for a in sp.BACKBONE_ATOMS]].to_numpy(float))
    u["offset"] = (np.nan_to_num(offs) > 3.0).any(axis=1)
    u["solid"] = (m["sample_state_evidence"].astype(str) == "solid").to_numpy()
    ph, temp = m["ph"].to_numpy(float), m["temperature"].to_numpy(float)
    ph_ok = (ph >= TOL_PH[0]) & (ph <= TOL_PH[1])
    temp_ok = (temp >= TOL_T[0]) & (temp <= TOL_T[1])
    u["phT"] = ~(ph_ok & temp_ok)
    u["physical_state"] = m["physical_state"].to_numpy()
    u["deviant"] = u["dev"] > DEV_CUT
    return u.reset_index(drop=True)


def boot_shares(
    F: pd.DataFrame, n_boot: int = fr.N_BOOT, seed: int = fr.SEED
) -> pd.DataFrame:
    keys = [k for k, _ in FLAGS]
    uniq, inv = np.unique(F["sequence"].to_numpy(), return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    X = F[keys].to_numpy(float)
    dv = F["deviant"].to_numpy()
    rng = np.random.default_rng(seed)
    bs = {True: [], False: []}
    for _ in range(n_boot):
        idx = np.concatenate(
            [members[i] for i in rng.integers(0, len(uniq), len(uniq))]
        )
        for g in (True, False):
            sel = idx[dv[idx] == g]
            bs[g].append(X[sel].mean(0) if sel.size else np.full(len(keys), np.nan))
    rows = []
    for g in (True, False):
        est = X[dv == g].mean(0)
        B = np.vstack(bs[g])
        for j, k in enumerate(keys):
            rows.append({
                "group": "deviant" if g else "consistent", "flag": k, "n_chains": int((dv == g).sum()),
                "n_flagged": int(X[dv == g, j].sum()), "share": float(est[j]),
                "lo": float(np.nanpercentile(B[:, j], 2.5)), "hi": float(np.nanpercentile(B[:, j], 97.5)),
            })  # fmt: skip
    return pd.DataFrame(rows)


def draw(cond_m: dict):
    """Fig. 3b: same- versus different-condition replicate pairs (shift-count matched)."""
    fig = sp.figure(f3a.W, f3a.H)
    ax_d, ax_r = f3a.dot_axes(fig, len(COND_ROWS), [lab for _, lab in COND_ROWS])
    for y, (cond, _) in enumerate(COND_ROWS):
        d = cond_m[cond]
        f3a.dot(
            ax_d,
            d["median_absdg"],
            d["median_absdg_lo"],
            d["median_absdg_hi"],
            y,
            C_SAME,
        )
        f3a.dot(ax_r, d["r"], d["r_lo"], d["r_hi"], y, C_SAME)  # fmt: skip
    ax_d.set_xlabel(f3a.XLABEL_DEV)
    ax_r.set_xlabel(f3a.XLABEL_R)
    sp.fig_label(fig, "b", 0.0, 0.0)
    return fig


def draw_flags(B: pd.DataFrame):
    """Supplementary Fig. S19: annotations of deviant versus consistent unfiltered-only chains.

    Deviant chains carry the message (black filled circles), consistent chains are the
    control (grey open squares), as ordered and disordered residues in Fig. 2b; no category
    hue, because both groups are chains the stricter tiers remove."""
    from matplotlib.lines import Line2D

    W, H = sp.W_SINGLE, 48.0
    fig = sp.figure(W, H)
    ax = fig.add_axes([31.0 / W, 9.5 / H, 44.0 / W, 29.0 / H])
    keys = [k for k, _ in FLAGS_S19]
    y = np.arange(len(keys))
    for g, st in FLAG_STYLE.items():
        sub = B[B.group == g].set_index("flag").loc[keys]
        for j, k in enumerate(keys):
            ax.plot([100 * sub.loc[k, "lo"], 100 * sub.loc[k, "hi"]], [y[j] + st["dy"]] * 2,
                    color=st["color"], lw=0.8, solid_capstyle="butt", zorder=2)  # fmt: skip
        ax.plot(100 * sub.share, y + st["dy"], ls="", marker=st["marker"], ms=st["ms"],
                mfc=st["mfc"], mec=st["color"], mew=0.8, zorder=3)  # fmt: skip
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in FLAGS_S19], fontsize=sp.FS_TICK)
    ax.set_ylim(len(keys) - 0.45, -0.55)
    x_hi = 20 * int(np.ceil(100 * B["hi"].max() / 20 + 1e-9))
    ax.set_xlim(0, x_hi)
    ax.set_xticks(range(0, x_hi + 1, 20))
    ax.set_xlabel("Chains with the annotation (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=f3a.GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    n = {g: int(B[B.group == g].n_chains.iloc[0]) for g in FLAG_STYLE}
    words = {
        "deviant": "Deviate from replicates",
        "consistent": "Agree with replicates",
    }
    h = [
        Line2D([], [], color=st["color"], marker=st["marker"], ms=st["ms"], mfc=st["mfc"],
               mec=st["color"], mew=0.8, lw=0.8, label=f"{words[g]} ({n[g]:,} chains)")
        for g, st in FLAG_STYLE.items()
    ]  # fmt: skip
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(31.0 / W, 1 - 0.5 / H),
               borderaxespad=0.0, handlelength=1.6, handletextpad=0.4, labelspacing=0.3)  # fmt: skip
    return fig


def main(argv=None) -> None:
    sp.parse_args(sp.cli(__doc__), argv)
    D = fr.Data()
    L = fr.loo_table(D)
    C = fr.condition_pairs(D)
    cond_raw = fr.boot_conditions(C, matched=False)
    cond_m = fr.boot_conditions(C, matched=True)
    F = flag_table(D, L)
    B = boot_shares(F)

    out = sp.out_dir("fig3_data")
    out.mkdir(parents=True, exist_ok=True)
    C.to_csv(out / "fig3b1_condition_pairs.csv", index=False)
    F.to_csv(out / "fig3b2_unfiltered_only_chains.csv", index=False)
    B.to_csv(out / "fig3b2_flag_shares.csv", index=False)
    fig = draw(cond_m)
    pdf = sp.save(fig, "fig3b")
    sp.save(draw_flags(B), "S19")

    def rnd(d):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

    top = F.sort_values("dev", ascending=False).head(15)
    nums = {
        "definition": {
            "b1": "non-bound replicate pairs (identical sequence, different entries), unfiltered labels, >= 10 co-scored "
            "residues; same: |dpH| <= 0.2 and |dT| <= 2 K; different: |dpH| >= 1 or |dT| >= 10 K; <= 20 seeded pairs "
            "per sequence and class; shift-count matching re-weights different-condition pairs to the same-condition "
            "mean-|dk| distribution; 95 % CI from 1,000 sequence bootstraps",
            "b2": "a1 chains scored only at the unfiltered tier; deviant = LOO deviation > 0.1; flags as in the script "
            "docstring; 95 % CI from 1,000 sequence bootstraps",
        },
        "b1_unmatched": {k: rnd(v) for k, v in cond_raw.items()},
        "b1_dk_matched": {k: rnd(v) for k, v in cond_m.items()},
        "b2": {
            g: {
                "n_chains": int(B[B.group == g].n_chains.iloc[0]),
                **{
                    r.flag: {
                        "pct": round(100 * r.share, 1),
                        "n": int(r.n_flagged),
                        "ci": [round(100 * r.lo, 1), round(100 * r.hi, 1)],
                    }
                    for r in B[B.group == g].itertuples()
                },  # fmt: skip
            }
            for g in ("deviant", "consistent")
        },
        "b2_any_of_state_cosolvent_offset_pct": {
            "deviant": round(
                100 * F[F.deviant][["state", "cosolvent", "offset"]].any(axis=1).mean(),
                1,
            ),
            "consistent": round(
                100
                * F[~F.deviant][["state", "cosolvent", "offset"]].any(axis=1).mean(),
                1,
            ),
        },
        "b2_top_deviants": top[
            [
                "id",
                "dev",
                "n_res",
                "physical_state",
                "state",
                "cosolvent",
                "offset",
                "solid",
            ]
        ]
        .round(3)
        .to_dict("records"),
    }
    (out / "fig3b_numbers.json").write_text(
        json.dumps(nums, indent=1, ensure_ascii=False, default=str)
    )
    print(json.dumps(nums, indent=1, ensure_ascii=False, default=str))
    print(f"wrote {pdf}", file=sys.stderr)


if __name__ == "__main__":
    main()
