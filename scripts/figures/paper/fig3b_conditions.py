#!/usr/bin/env python3
"""Fig. 3b -- sample conditions change the label, so TriZOD records and grades them.

(b1) Replicate pairs (two non-bound chains of identical sequence from different BMRB
     entries, unfiltered labels, >= 10 co-scored residues) measured under the SAME
     conditions (|dpH| <= 0.2 and |dT| <= 2 K) or under DIFFERENT conditions
     (|dpH| >= 1 or |dT| >= 10 K); pairs in between are dropped, and each class is
     capped at 20 seeded pairs per sequence. Three statistics: the median over pairs of
     the per-pair mean |dG|, the pooled residue-level MAE and the pooled residue-level
     Pearson r. Different-condition pairs are shown raw and after shift-count matching:
     pairs are re-weighted so that their mean |dk| distribution (bins 0, 0.5, 1, 2, 4,
     8, inf) matches the same-condition pairs. 95 % CIs from 1,000 sequence bootstraps
     (the matching weights are re-estimated inside every replicate).
(b2) Which annotations mark the replicates that disagree. Chains of the a1 analysis
     (``fig3a_reliability.py``) that are scored ONLY at the unfiltered tier, split into
     deviant (leave-one-entry-out deviation > 0.1) and consistent (<= 0.1) chains; dots
     give the share of each group carrying a flag, whiskers 95 % sequence-bootstrap CIs.
     Flags: physical state denied at some tier by the pipeline's own rule
     (``trizod.bmrb.sample_state.denied_physical_states`` with the strict deny list and
     the cosolvent corroboration rule); a perturbing cosolvent named in the entry
     (``cosolvent_evidence``); |POTENCI offset| > 3 sigma for any backbone atom (the
     tolerant ``max-offset``); solid-state sample (``sample_state_evidence``); pH or
     temperature outside the tolerant range (pH 2-12, 263-333 K).

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig3b_conditions.py \
        --out-dir <draft>/Figures/paper
Outputs: <out-dir>/fig3b.pdf; <out-dir>/fig3_data/fig3b_*.csv and fig3b_numbers.json.
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

from trizod.offsets import off_sigma_col  # noqa: E402

FS_SMALL = 5.0
DEV_CUT = 0.1
C_SAME = sp.CAT["neutral"]
C_DIFF = sp.CAT["removed"]
TOL_PH = (2.0, 12.0)
TOL_T = (263.0, 333.0)
FLAGS = [
    ("state", "Non-native physical state"),
    ("cosolvent", "Perturbing cosolvent named"),
    ("offset", "|Offset| > 3σ (any atom)"),
    ("solid", "Solid-state sample"),
    ("phT", "pH or T outside tolerant range"),
]


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


def draw(cond_raw: dict, cond_m: dict, C: pd.DataFrame, B: pd.DataFrame):
    from matplotlib.lines import Line2D

    W, H = sp.W_SINGLE, 72.0
    fig = sp.figure(W, H)

    def ax_mm(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    # ------------------------------------------------------------------ b1
    metrics = [
        ("median_absdg", "Median per-pair |ΔG|", (0, 0.10), [0, 0.05, 0.10]),
        ("mae", "Residue MAE", (0, 0.18), [0, 0.05, 0.10, 0.15]),
        ("r", "Residue Pearson r", (0.4, 1.0), [0.4, 0.6, 0.8, 1.0]),
    ]
    x_axes = [10.0, 38.0, 66.0]
    for (key, title, ylim, yt), xa in zip(metrics, x_axes):
        ax = ax_mm(xa, 47.0, 19.0, 18.5)
        same, diff, diffm = cond_raw["same"], cond_raw["different"], cond_m["different"]
        for x, d, col, filled in (
            (0, same, C_SAME, True),
            (1.15, diffm, C_DIFF, True),
            (0.85, diff, C_DIFF, False),
        ):
            ax.errorbar(
                x, d[key], yerr=[[d[key] - d[f"{key}_lo"]], [d[f"{key}_hi"] - d[key]]],
                fmt="o", ms=3.4 if filled else 2.8, color=col, mfc=col if filled else "white",
                mec=col, mew=0.6, elinewidth=0.6, capsize=0, zorder=3,
            )  # fmt: skip
        ax.set_xlim(-0.5, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["same", "different"])
        ax.set_ylim(*ylim)
        ax.set_yticks(yt)
        ax.set_title(title, fontsize=FS_SMALL + 0.5, pad=2)
    ns, nd = cond_raw["same"], cond_raw["different"]
    h = [
        Line2D([], [], ls="", marker="o", ms=3, color=C_SAME,
               label=f"same conditions ({ns['n_pairs']:,} pairs, {ns['n_sequences']} sequences)"),
        Line2D([], [], ls="", marker="o", ms=3, color=C_DIFF,
               label=f"different conditions, shift-count matched ({nd['n_pairs']:,} pairs, {nd['n_sequences']} seq.)"),
        Line2D([], [], ls="", marker="o", ms=2.6, mfc="white", mec=C_DIFF, mew=0.6,
               label="different conditions, unmatched"),
    ]  # fmt: skip
    fig.text(1.0 / W, 65.0 / H, "b1", fontsize=6.5, fontweight="bold", va="bottom")
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(9.0 / W, 42.5 / H), ncol=1,
               fontsize=FS_SMALL, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.0)  # fmt: skip

    # ------------------------------------------------------------------ b2
    ax = ax_mm(40.0, 6.0, 45.0, 21.0)
    keys = [k for k, _ in FLAGS]
    y = np.arange(len(keys))[::-1]
    for g, col, dy in (("deviant", C_DIFF, 0.12), ("consistent", C_SAME, -0.12)):
        sub = B[B.group == g].set_index("flag").loc[keys]
        ax.errorbar(100 * sub.share, y + dy, xerr=[100 * (sub.share - sub.lo), 100 * (sub.hi - sub.share)],
                    fmt="o", ms=3.2, color=col, mec="white", mew=0.3, elinewidth=0.5, capsize=0, zorder=3)  # fmt: skip
    for j, k in enumerate(keys):
        a = B[(B.group == "deviant") & (B.flag == k)].share.iloc[0]
        b = B[(B.group == "consistent") & (B.flag == k)].share.iloc[0]
        ax.plot(
            [100 * a, 100 * b],
            [y[j] + 0.12, y[j] - 0.12],
            color="#BBBBBB",
            lw=0.5,
            zorder=1,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in FLAGS])
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Chains carrying the annotation (%)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color="#EEEEEE", lw=0.4, zorder=0)
    nd_ = int(B[B.group == "deviant"].n_chains.iloc[0])
    nc_ = int(B[B.group == "consistent"].n_chains.iloc[0])
    h2 = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=3,
            color=C_DIFF,
            label=f"deviant (> {DEV_CUT}; n = {nd_})",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=3,
            color=C_SAME,
            label=f"consistent (n = {nc_})",
        ),
    ]
    fig.legend(handles=h2, loc="lower left", bbox_to_anchor=(40.0 / W, 27.6 / H), ncol=2, fontsize=FS_SMALL,
               handletextpad=0.2, columnspacing=1.0, borderaxespad=0.0)  # fmt: skip
    fig.text(1.0 / W, 30.0 / H, "b2", fontsize=6.5, fontweight="bold", va="bottom")
    fig.text(
        7.5 / W,
        30.2 / H,
        "Chains scored only at unfiltered:",
        fontsize=FS_SMALL,
        va="bottom",
    )

    sp.fig_label(fig, "b", 0.0, 0.0)
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
    fig = draw(cond_raw, cond_m, C, B)
    pdf = sp.save(fig, "fig3b")

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
