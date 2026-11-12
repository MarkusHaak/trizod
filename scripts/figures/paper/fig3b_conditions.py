#!/usr/bin/env python3
"""Fig. 4a and 4d -- replicates differ more across pH and temperature; what strict filtering removes.

Fig. 4a (``fig4a.pdf``; was the left half of Fig. 3b until 12 Nov 2026): the median over
replicate pairs of the per-pair mean |dG|, in two blocks. Replicate pairs are two non-bound
chains of identical sequence from different BMRB entries (unfiltered labels, >= 10
co-scored residues, chains that deposit both pH and temperature;
``fig3_replicates.condition_pairs``), measured under the SAME conditions (|dpH| <= 0.2 and
|dT| <= 2 K) or under DIFFERENT conditions (|dpH| >= 1 or |dT| >= 10 K); pairs in between are
dropped, and each class is capped at 20 seeded pairs per sequence. Different-condition pairs
are re-weighted so that their mean |dk| profile (bins 0, 0.5, 1, 2, 4, 8, inf) matches the
same-condition pairs (shift-count matching). Four blocks, one row each, the same-condition
median (black) above the different-condition median (green): all pairs; without pairs whose
two chains differ in sample state (open symbols; solid-state evidence, cosolvent evidence, a
physical state the strict tier denies, or a membrane mimetic; :func:`state_clean_pairs`, the
definition of the ``\\condRatioFigMatchedStateClean`` macros in ``make_numbers.py``); and,
below a rule, the same pairs split by study (review pass 12 Nov 2026,
reports/story/review/science.md): pairs whose two entries share a citation title or DOI
("One study") and pairs from different studies
(``fig3_replicates.study_condition_pairs`` / ``boot_study_conditions(matched="class")``, the
definition of the ``\\condRatioWithinStudy`` / ``\\condRatioAcrossStudy`` macros: each
class's different-condition pairs re-weighted to its own same-condition |dk| profile, the
across-study same-condition pairs to the within-study ones). The ratio different / same is
printed beside each block. 95 % CIs from 1,000 sequence bootstraps; the matching weights are
re-estimated inside every replicate. The residue-level r of the two classes moves to
Supplementary Fig. 7.

Fig. 4d (``fig4d.pdf``; Supplementary Fig. 19 until 12 Nov 2026): annotations of the replicate
chains that only the unfiltered tier keeps. Chains of the leave-one-entry-out analysis
(``fig3_replicates.loo_table``) scored ONLY at the unfiltered tier, split into chains that
deviate from the consensus of their replicates (deviation > 0.1) and chains that agree
(<= 0.1); dots give the share of each group carrying a flag, bars 95 % sequence-bootstrap
CIs. Flags: physical state denied at some tier by the pipeline's own rule
(``trizod.bmrb.sample_state.denied_physical_states`` with the strict deny list and the
cosolvent corroboration rule); a perturbing cosolvent named in the entry
(``cosolvent_evidence``); |POTENCI offset| > 3 sigma for any backbone atom (the tolerant
``max-offset``); solid-state sample (``sample_state_evidence``); pH or temperature outside
the tolerant range (pH 2-12, 263-333 K). ``make_numbers.py`` imports :func:`flag_table` and
:func:`boot_shares` for the ``\\pctDeviantFig*`` / ``\\pctConsistentFig*`` macros.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig3b_conditions.py \\
        --out-dir <draft>/Figures/paper
Outputs: <out-dir>/fig4a.pdf and fig4d.pdf (78 x 55 mm each, placed at 1:1);
<out-dir>/source_data/fig4a.csv and fig4d.csv (the plotted values); <out-dir>/fig3_data/
fig3b1_condition_pairs.csv, fig3b2_*.csv and fig3b_numbers.json (pair and chain tables).
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

DEV_CUT = 0.1
TOL_PH = (2.0, 12.0)
TOL_T = (263.0, 333.0)
# One name per annotation across the paper (figure review item 7; review_burkhard 2.10,
# 2.11): the words of Fig. 2b ("Non-native state", "Cosolvent", "Solid sample") and of the
# Fig. 4 legend ("residual offset", "out of range").
FLAGS = [
    ("state", "Non-native state"),
    ("cosolvent", "Cosolvent"),
    ("offset", "Residual offset > 3σ(n)"),
    ("solid", "Solid sample"),
    ("phT", "pH or temperature out of range"),
]
# Fig. 4d row labels (two lines each, so the 78 mm cell keeps a wide plot)
FLAG_LABELS = {
    "state": "Non-native\nstate",
    "cosolvent": "Cosolvent",
    "offset": "Residual offset\n> 3σ(n)",
    "solid": "Solid\nsample",
    "phT": "pH or temperature\nout of range",
}
# the two chain groups of Fig. 4d: one entity (removed chains), two groups -> sp.CONTRAST
FLAG_STYLE = {
    "deviant": {**sp.CONTRAST["focus"], "ms": 4.0, "dy": -0.17},
    "consistent": {**sp.CONTRAST["control"], "ms": 3.6, "dy": 0.17},
}
FLAG_WORDS = {  # the legend's words (deviate, agree) with the threshold that defines them
    "deviant": "Deviate (|ΔG| > 0.1)",
    "consistent": "Agree (|ΔG| ≤ 0.1)",
}

# --- Fig. 4a ---------------------------------------------------------------------------
W, H = sp.W_SINGLE, 55.0  # mm, one cell of the 2 x 2 Fig. 4 grid
CI_COLOR = sp.INK_SOFT
GRID = "#E6E6E6"  # light grid along the compared axis (as in fig3a_reliability.py)
MS = 5.0  # marker size, pt
AX_Y0 = 9.5  # bottom of the axes, mm: the x labels of all four panels sit on one line
XLABEL_ABSDG = "Difference between replicates, |ΔG|"  # Supplementary figures
# Fig. 4a: the dots are medians of per-pair |dG|
XLABEL_MEDIAN = "Median difference between replicates, |ΔG|"
# Same-condition pairs without a state difference: the "same" entity (black) with the
# open fill that marks "without state differences" in sp.COND["different_state_clean"].
SAME_STATE_CLEAN = {
    **sp.COND["same"],
    "mfc": "white",
    "label": "Same, without state differences",
}
# Fig. 4a blocks, top to bottom: (key, row label, same-condition style, different style).
# One row per block: the same-condition median above, the different-condition median below.
BLOCKS = [
    ("all", "All pairs", sp.COND["same"], sp.COND["different"]),
    ("clean", "Without state\ndifferences", SAME_STATE_CLEAN, sp.COND["different_state_clean"]),
    ("within", "One study", sp.COND["same"], sp.COND["different"]),
    ("across", "Different\nstudies", sp.COND["same"], sp.COND["different"]),
]  # fmt: skip
# row of each block; the light rule sits between "clean" and "within"
BLOCK_Y = {"all": 0.0, "clean": 1.0, "within": 2.3, "across": 3.3}
DODGE = 0.18  # same above (-), different below (+), in row units


def state_signature(D: fr.Data) -> pd.DataFrame:
    """Per non-bound chain (index ``id``): the four sample-state annotations that make two
    chains of a pair differ in state (as ``make_numbers.py``, section (3) of the revision
    numbers): a physical state the strict tier denies, cosolvent evidence, solid-state
    evidence and a membrane mimetic."""
    from trizod.bmrb.sample_state import (
        PHYSICAL_STATE_STRICT_DENY,
        denied_physical_states,
    )

    nb = D.nb.set_index("id")
    cor = nb["cosolvent_evidence"].fillna(False).astype(bool)
    return pd.DataFrame(
        {
            "state": np.asarray(denied_physical_states(nb["physical_state"], PHYSICAL_STATE_STRICT_DENY, corroborated=cor.to_numpy()), bool),
            "cosolvent": cor.to_numpy(),
            "solid": (nb["sample_state_evidence"].astype(str) == "solid").to_numpy(),
            "membrane": nb["membrane_mimetic"].notna().to_numpy(),
        },
        index=nb.index,
    )  # fmt: skip


def state_clean_pairs(D: fr.Data, C: pd.DataFrame) -> pd.DataFrame:
    """The condition pairs ``C`` whose two chains do not differ in sample state."""
    sig = state_signature(D)
    diff = (sig.loc[C.a].to_numpy() != sig.loc[C.b].to_numpy()).any(axis=1)
    return C[~diff].reset_index(drop=True)


def condition_rows(
    C: pd.DataFrame, clean: pd.DataFrame, S: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """The plotted rows of Fig. 4a (shift-count matched) and the four ratios with CIs.

    ``S``: :func:`fig3_replicates.study_condition_pairs` of ``C`` (the same pairs with their
    study relation); the study blocks use ``boot_study_conditions(matched="class")``, the
    definition of the ``\\condAbsDG{Same,Diff}{Within,Across}Study`` and
    ``\\condRatio{Within,Across}Study`` macros."""
    rows, ratios = [], {}
    for block, Cb in (("all", C), ("clean", clean)):
        b = fr.boot_conditions(Cb, matched=True)
        for cond in ("same", "different"):
            d = b[cond]
            rows.append({"block": block, "cond": cond, "n_pairs": d["n_pairs"],
                         "n_sequences": d["n_sequences"], "median_absdg": d["median_absdg"],
                         "lo": d["median_absdg_lo"], "hi": d["median_absdg_hi"],
                         "flip": d["flip"], "r": d["r"],
                         "matching": "different -> all same-condition pairs"})  # fmt: skip
        ratios[block] = b["ratio_median_absdg"]
    bs = fr.boot_study_conditions(S, matched="class")
    for study in ("within", "across"):
        for cond in ("same", "different"):
            d = bs["cells"][(cond, study)]
            m = ("unweighted (reference)" if (cond, study) == ("same", "within") else
                 "same -> within-study same-condition pairs" if cond == "same" else
                 f"different -> {study}-study same-condition pairs")  # fmt: skip
            rows.append({"block": study, "cond": cond, "n_pairs": d["n_pairs"],
                         "n_sequences": d["n_sequences"], "median_absdg": d["median_absdg"],
                         "lo": d["lo"], "hi": d["hi"], "flip": np.nan, "r": np.nan,
                         "matching": m})  # fmt: skip
        ratios[study] = bs["ratios"][f"cond_{study}"]
    ratios["n_unclassified"] = bs["n_unclassified"]
    return pd.DataFrame(rows), ratios


def draw_conditions(R: pd.DataFrame, ratios: dict):
    """Fig. 4a: median per-pair |dG|, same vs different conditions, four blocks, ratios.

    Each block is one row: the same-condition median (black) above the different-condition
    median (green), joined at the right by a bracket that carries their ratio. A light rule
    separates the two state blocks from the two study blocks."""
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    fig = sp.figure(W, H)
    x0, y0, w, h = 23.0, AX_Y0, 34.0, 39.0  # axes, mm
    ax = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
    R = R.set_index(["block", "cond"])
    trb = blended_transform_factory(ax.transAxes, ax.transData)
    for bk, _, st_same, st_diff in BLOCKS:
        yb = BLOCK_Y[bk]
        for cond, st, dy in (("same", st_same, -DODGE), ("different", st_diff, DODGE)):
            d = R.loc[(bk, cond)]
            ax.plot([d.lo, d.hi], [yb + dy] * 2, color=CI_COLOR, lw=0.8,
                    solid_capstyle="butt", zorder=2)  # fmt: skip
            ax.plot(
                d.median_absdg, yb + dy, ms=MS, zorder=3, **sp.marker_kw(st, mew=0.9)
            )
        # ratio bracket, right of the axes
        xb, ya, yz = 1.04, yb - DODGE, yb + DODGE
        ax.plot([xb - 0.02, xb, xb, xb - 0.02], [ya, ya, yz, yz], color=sp.INK, lw=0.6,
                transform=trb, clip_on=False)  # fmt: skip
        rr = ratios[bk]
        ax.text(xb + 0.04, yb, f"×{rr['est']:.2f}\n({rr['lo']:.2f}–{rr['hi']:.2f})",
                transform=trb, ha="left", va="center", fontsize=sp.FS_BODY,
                linespacing=1.15, clip_on=False)  # fmt: skip
    ys = [BLOCK_Y[bk] for bk, *_ in BLOCKS]
    ax.set_ylim(max(ys) + 0.62, min(ys) - 0.62)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab, *_ in BLOCKS], linespacing=1.1)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    # light rule between the state blocks and the study blocks, across label column and axes
    y_rule = (BLOCK_Y["clean"] + BLOCK_Y["within"]) / 2
    trf = blended_transform_factory(fig.transFigure, ax.transData)
    ax.plot([1.0 / W, (x0 + w) / W], [y_rule, y_rule], transform=trf, color=sp.CAT["light"],
            lw=0.5, clip_on=False, zorder=1)  # fmt: skip
    ax.set_xlim(0, 0.1)
    ax.set_xticks([0, 0.03, 0.06, 0.09])
    ax.set_xticklabels(["0", "0.03", "0.06", "0.09"])
    ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(XLABEL_MEDIAN)
    # key above the axes, after the panel letter: the two condition classes (filled; the
    # open symbols are named by their row label)
    y_key = 1 - 2.2 / H
    xk = 5.0
    for st, lab in ((sp.COND["same"], "Same conditions"),
                    (sp.COND["different"], "Different pH or temperature")):  # fmt: skip
        fig.add_artist(Line2D([(xk + 1.0) / W], [y_key], transform=fig.transFigure,
                              ms=MS - 0.6, **sp.marker_kw(st, mew=0.9)))  # fmt: skip
        t = fig.text((xk + 2.6) / W, y_key, lab, ha="left", va="center",
                     fontsize=sp.FS_BODY)  # fmt: skip
        fig.canvas.draw()
        xk = t.get_window_extent().x1 / fig.dpi / sp.MM + 3.0
    # header of the ratio column: different / same, with its 95 % CI
    ax.text(1.02, 1.0, "Ratio", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=sp.FS_BODY, clip_on=False)  # fmt: skip
    sp.fig_label(fig, "a", 0.0, 0.0)
    return fig


# --- Fig. 4d ---------------------------------------------------------------------------
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


def draw_flags(B: pd.DataFrame):
    """Fig. 4d: annotations of deviating versus agreeing chains that only the unfiltered tier
    keeps. One entity (chains the stricter tiers remove), two groups: ink, filled circles
    for the chains that deviate, open squares for those that agree (sp.CONTRAST)."""
    from matplotlib.lines import Line2D

    fig = sp.figure(W, H)
    x0, y0, w, h = 24.0, AX_Y0, 51.0, 41.0  # axes, mm
    ax = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
    keys = [k for k, _ in FLAGS]
    y = np.arange(len(keys))
    for g, st in FLAG_STYLE.items():
        sub = B[B.group == g].set_index("flag").loc[keys]
        for j, k in enumerate(keys):
            ax.plot([100 * sub.loc[k, "lo"], 100 * sub.loc[k, "hi"]], [y[j] + st["dy"]] * 2,
                    color=CI_COLOR, lw=0.8, solid_capstyle="butt", zorder=2)  # fmt: skip
        ax.plot(100 * sub.share, y + st["dy"], ms=st["ms"], zorder=3, **sp.marker_kw(st, mew=0.9))  # fmt: skip
    ax.set_yticks(y)
    ax.set_yticklabels([FLAG_LABELS[k] for k in keys])
    ax.set_ylim(len(keys) - 0.45, -0.55)
    x_hi = 10 * int(np.ceil(100 * B["hi"].max() / 10 + 1e-9))
    ax.set_xlim(0, x_hi)
    ax.set_xticks(range(0, x_hi + 1, 20))
    ax.set_xlabel("Chains with the annotation (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    # key above the axes, after the panel letter, as in Fig. 4a (figure review item 15: it
    # sat inside the plot next to the last row, and no row leaves room for it at the
    # right); chain counts are in the caption
    y_key, xk = 1 - 2.2 / H, 5.0
    for g, st in FLAG_STYLE.items():
        fig.add_artist(Line2D([(xk + 1.0) / W], [y_key], transform=fig.transFigure,
                              ms=st["ms"], **sp.marker_kw(st, mew=0.9)))  # fmt: skip
        t = fig.text((xk + 2.6) / W, y_key, FLAG_WORDS[g], ha="left", va="center",
                     fontsize=sp.FS_BODY)  # fmt: skip
        fig.canvas.draw()
        xk = t.get_window_extent().x1 / fig.dpi / sp.MM + 3.0
    sp.fig_label(fig, "d", 0.0, 0.0)
    return fig


def main(argv=None) -> None:
    sp.parse_args(sp.cli(__doc__), argv)
    D = fr.Data()
    L = fr.loo_table(D)
    C = fr.condition_pairs(D)
    clean = state_clean_pairs(D, C)
    S = fr.study_condition_pairs(D, C)
    cond_raw = fr.boot_conditions(C, matched=False)
    cond_m = fr.boot_conditions(C, matched=True)
    R, ratios = condition_rows(C, clean, S)
    n_unclassified = ratios.pop("n_unclassified")
    F = flag_table(D, L)
    B = boot_shares(F)

    out = sp.out_dir("fig3_data")
    out.mkdir(parents=True, exist_ok=True)
    C.to_csv(out / "fig3b1_condition_pairs.csv", index=False)
    F.to_csv(out / "fig3b2_unfiltered_only_chains.csv", index=False)
    B.to_csv(out / "fig3b2_flag_shares.csv", index=False)
    src = sp.out_dir("source_data")
    src.mkdir(parents=True, exist_ok=True)
    R.assign(
        ratio_block=R["block"].map(lambda b: ratios[b]["est"]),
        ratio_block_lo=R["block"].map(lambda b: ratios[b]["lo"]),
        ratio_block_hi=R["block"].map(lambda b: ratios[b]["hi"]),
    ).to_csv(src / "fig4a.csv", index=False, float_format="%.4f")
    B.to_csv(src / "fig4d.csv", index=False, float_format="%.4f")
    pdf_a = sp.save(draw_conditions(R, ratios), "fig4a")
    pdf_d = sp.save(draw_flags(B), "fig4d")

    def rnd(d):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

    top = F.sort_values("dev", ascending=False).head(15)
    nums = {
        "definition": {
            "fig4a": "non-bound replicate pairs (identical sequence, different entries), unfiltered labels, >= 10 co-scored "
            "residues, both chains depositing pH and temperature; same: |dpH| <= 0.2 and |dT| <= 2 K; different: |dpH| >= 1 or "
            "|dT| >= 10 K; <= 20 seeded pairs per sequence and class; shift-count matching re-weights different-condition pairs "
            "to the same-condition mean-|dk| distribution; 'clean' drops pairs whose chains differ in sample state; 95 % CI "
            "from 1,000 sequence bootstraps",
            "fig4d": "leave-one-entry-out chains scored only at the unfiltered tier; deviant = deviation > 0.1; flags as in the "
            "script docstring; 95 % CI from 1,000 sequence bootstraps",
        },
        "fig4a_rows": R.round(4).to_dict("records"),
        "fig4a_study_unclassified_pairs": n_unclassified,
        "fig4a_ratios": {k: rnd(v) for k, v in ratios.items()},
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
    print(json.dumps({k: nums[k] for k in ("fig4a_rows", "fig4a_ratios", "b2")}, indent=1,
                     ensure_ascii=False, default=str))  # fmt: skip
    print(f"wrote {pdf_a}\nwrote {pdf_d}", file=sys.stderr)


if __name__ == "__main__":
    main()
