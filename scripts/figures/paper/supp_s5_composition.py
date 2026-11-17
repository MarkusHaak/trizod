#!/usr/bin/env python3
"""Supplementary Fig. S5 -- composition of the tiers and the two test sets.

Six chain sets: the four cumulative tiers (scored chains; unfiltered ⊃ tolerant ⊃
moderate ⊃ strict, from the bundle's per-tier ``scores.json``), the TriZOD test set
(``split == test_trizod``) and CheZOD117 (``split == test_chezod117``).

Panels
  a  Chain length (residues), empirical cumulative distribution on a log axis; the
     dotted line marks the median (values in S5_summary.json).
  b  Deposition year of the chain's BMRB entry, from the raw ``_Entry.Submission_date``
     (``paper_raw_scan.py``), in the six bins the text uses. Early BMRB entries carry the
     placeholder date 1995-07-31; they fall in the first bin. Chains whose entry has no
     date ('.') are left out of the shares (one in this build; the caption says so);
     the row labels give each set's full size.
  c  Source organism, a heuristic exact-string parse of the raw
     ``_Entity_natural_src.Organism_name_scientific`` of the chain's entity (approximate:
     strain names and synonyms are not merged). Share of each set's chains; the ten most
     frequent named organisms of the unfiltered tier, then synthetic constructs, all
     other named organisms, and chains with no organism recorded ('.' or no row).

Run:  uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s5_composition.py \
          --out-dir <draft>/Figures/paper
Outputs: <out-dir>/S5.pdf and <out-dir>/suppfigs_a_data/S5_chains.csv / S5_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import paper_raw_scan as rs  # noqa: E402
import style_paper as sp  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TIERS = sp.TIERS
DATA_SUBDIR = "suppfigs_a_data"  # under --out-dir
YEAR_BINS = [
    (None, 2000, "≤2000"),
    (2001, 2005, "2001–05"),
    (2006, 2010, "2006–10"),
    (2011, 2015, "2011–15"),
    (2016, 2020, "2016–20"),
    (2021, 2025, "2021–25"),
]
YEAR_COLORS = ["#F0F0F0", "#D4D4D4", "#ABABAB", "#828282", "#585858", "#2E2E2E"]
SET_COLORS = {
    **sp.TIER_COLORS,
    "TriZOD test set": sp.CAT["test"],
    "CheZOD117": sp.CAT["chezod117"],
}
N_ORG = 10


def fmt(n) -> str:
    return f"{int(n):,}"


def load():
    df = sp.load_main(["id", "entry_id", "entity_id", "length", "split"])
    raw = rs.scan_raw()
    sub = df.entry_id.map(lambda e: raw.get(str(e), {}).get("sub"))
    df["sub"] = sub
    df["year"] = sub.map(lambda s: int(s[:4]) if s and s[:4].isdigit() else np.nan)
    df["org"] = [
        raw.get(str(e), {}).get("org", {}).get(str(en))
        for e, en in zip(df.entry_id, df.entity_id)
    ]
    tids = sp.load_tier_ids()
    sets = {t: df[df.id.isin(tids[t])] for t in TIERS}
    sets["TriZOD test set"] = df[df.split == "test_trizod"]
    sets["CheZOD117"] = df[df.split == "test_chezod117"]
    return df, sets


def year_bin(y):
    for lo, hi, lab in YEAR_BINS:
        if (lo is None or y >= lo) and y <= hi:
            return lab
    return None


def main():
    sp.parse_args(sp.cli(__doc__))
    DATA_OUT = sp.out_dir(DATA_SUBDIR)
    df, sets = load()
    names = list(sets)
    summ: dict = {"sets": {k: len(v) for k, v in sets.items()}}

    # lengths
    summ["length"] = {
        k: {
            q: float(v.length.quantile(p))
            for q, p in (
                ("min", 0),
                ("p25", 0.25),
                ("median", 0.5),
                ("p75", 0.75),
                ("max", 1),
            )
        }
        | {"mean": round(float(v.length.mean()), 1)}
        for k, v in sets.items()
    }
    # years
    yb = {}
    for k, v in sets.items():
        lab = v.year.dropna().map(year_bin)
        yb[k] = {b[2]: int((lab == b[2]).sum()) for b in YEAR_BINS}
    summ["year_bins"] = yb
    summ["year_missing"] = {k: int(v.year.isna().sum()) for k, v in sets.items()}
    summ["year_range"] = {
        k: [int(v.year.min()), int(v.year.max())] for k, v in sets.items()
    }
    summ["placeholder_1995_07_31_chains"] = {
        k: int((v["sub"] == "1995-07-31").sum()) for k, v in sets.items()
    }

    # organisms
    def org_cat(o):
        if o is None or (isinstance(o, float) and np.isnan(o)) or o in (".", "?", ""):
            return "not recorded"
        return o

    df["org_cat"] = df.org.map(org_cat)
    unf = sets["unfiltered"].id
    vc = df.loc[df.id.isin(unf), "org_cat"].value_counts()
    named = [o for o in vc.index if o not in ("not recorded", "synthetic construct")]
    top = named[:N_ORG]
    n_other_names = len(named) - N_ORG
    cols = top + ["synthetic construct", "other named", "not recorded"]

    def oc(o):
        if o in top or o in ("synthetic construct", "not recorded"):
            return o
        return "other named"

    org_tab = (
        pd.DataFrame(
            {
                k: df.loc[df.id.isin(v.id), "org_cat"].map(oc).value_counts()
                for k, v in sets.items()
            }
        )
        .reindex(cols)
        .fillna(0)
        .astype(int)
    )
    summ["organisms"] = org_tab.to_dict()
    summ["organism_distinct_names_unfiltered"] = int(
        len(named) + ("synthetic construct" in vc)
    )
    summ["organism_other_named_distinct"] = n_other_names
    print(
        json.dumps(
            {
                k: summ[k]
                for k in (
                    "sets",
                    "length",
                    "year_bins",
                    "year_missing",
                    "year_range",
                    "placeholder_1995_07_31_chains",
                )
            },
            indent=1,
        )
    )
    print(org_tab)

    # ---------------- figure ----------------------------------------------------------
    W, H = sp.W_DOUBLE, 104.0  # c transposed (organisms as rows) since 12 Nov 2026
    fig = sp.figure(W, H)

    def ax_mm(x, y_top, w, h):
        return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])

    # 160 mm grid: a (ECDF) and b (year bars, legend above in two rows) on top, the
    # organism heat map c below: organisms as rows and the six sets as columns, so every
    # label is horizontal (figure review item 16: no rotated tick labels)
    # a length ECDF
    ax = ax_mm(11, 9, 50, 34)
    handles = []
    for k, v in sets.items():
        x = np.sort(v.length.to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        test = k in ("TriZOD test set", "CheZOD117")
        ax.step(
            x,
            y,
            where="post",
            color=SET_COLORS[k],
            lw=1.0 if test else 0.8,
            ls="-" if not test else (0, (3, 1)) if k == "CheZOD117" else "-",
        )
        handles.append(
            Line2D(
                [],
                [],
                color=SET_COLORS[k],
                lw=1.0,
                ls=(0, (3, 1)) if k == "CheZOD117" else "-",
                label=k,
            )
        )
    ax.set_xscale("log")
    ax.set_xlim(5, 1200)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([5, 10, 20, 50, 100, 200, 500, 1000])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "200", "500", "1,000"])
    ax.minorticks_off()
    # median guide, started right of the legend (the medians lie near 100 residues)
    ax.hlines(0.5, 60, 1200, color=sp.CAT["light"], lw=0.5, ls=":", zorder=0)
    ax.set_xlabel("Chain length (residues)")
    ax.set_ylabel("Cumulative share of chains")
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=sp.FS_TICK,
        handlelength=1.4,
        borderaxespad=0.2,
        labelspacing=0.25,
    )
    sp.fig_label(fig, "a", 0.5, 3)

    # b year bins
    ax = ax_mm(97, 9, 59, 34)
    ys = np.arange(len(names))[::-1]
    for y, k in zip(ys, names):
        tot = sum(yb[k].values())
        left = 0.0
        for (_lo, _hi, lab), c in zip(YEAR_BINS, YEAR_COLORS):
            share = yb[k][lab] / tot
            ax.barh(y, share, left=left, height=0.7, color=c, lw=0.3, edgecolor="white")
            if share >= 0.07:
                ax.text(
                    left + share / 2,
                    y,
                    f"{100 * share:.0f}",
                    ha="center",
                    va="center",
                    fontsize=sp.FS_SMALL,
                    color="white" if c in YEAR_COLORS[3:] else "#222222",
                )
            left += share
        # full set size: the shares leave out chains without a submission date
        ax.text(
            -0.02,
            y,
            f"{k} ({fmt(len(sets[k]))})",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=sp.FS_TICK,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(names) - 0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}"))
    ax.set_xlabel("Chains by deposition year (%)")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=c, edgecolor="#BBBBBB", lw=0.3, label=b[2])
            for b, c in zip(YEAR_BINS, YEAR_COLORS)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=sp.FS_TICK,
        handlelength=0.9,
        columnspacing=1.0,
        handletextpad=0.3,
        labelspacing=0.2,
        borderaxespad=0.2,
    )
    sp.fig_label(fig, "b", 65, 3)

    # c organisms heatmap (rows = organism categories, columns = sets)
    top_c = 52
    n_org = len(cols)
    row_h, col_w = 3.0, 17.0
    ax = ax_mm(36, top_c + 9.5, col_w * len(names), row_h * n_org)
    share = org_tab.div(org_tab.sum(axis=0), axis=1).loc[cols, names]  # cols x sets
    ax.imshow(share.to_numpy() * 100, aspect="auto", cmap="Greys", vmin=0, vmax=60)
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = 100 * share.iat[i, j]
            ax.text(
                j,
                i,
                f"{v:.1f}" if v < 10 else f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=sp.FS_SMALL,
                color="white" if v > 33 else "#222222",
            )
    plain = ("synthetic construct", "other named", "not recorded")
    short = {
        "synthetic construct": "synthetic",
        "other named": "other",
        "not recorded": "not recorded",
    }
    yl = []
    for c in cols:
        if c in plain:
            yl.append(short[c])
        else:  # abbreviated genus
            g, _, rest = c.partition(" ")
            yl.append(f"{g[0]}. {rest}" if rest else g)
    ax.set_yticks(range(n_org))
    ax.set_yticklabels([y.replace("\n", " ") for y in yl], fontsize=sp.FS_TICK)
    for lab, c in zip(ax.get_yticklabels(), cols):
        if c not in plain:
            lab.set_fontstyle("italic")
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("TriZOD test set", "TriZOD\ntest set") for n in names],
                       fontsize=sp.FS_TICK, linespacing=1.05)  # fmt: skip
    ax.tick_params(axis="both", length=0, pad=1.5)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    fig.text(4.5 / W, 1 - (top_c + 0.2) / H, "Chains by source organism (% of the set)",
             ha="left", va="top", fontsize=sp.FS_BODY)  # fmt: skip
    sp.fig_label(fig, "c", 0.5, top_c)

    out = sp.save(fig, "S5")
    print("wrote", out)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, v in sets.items():
        t = v[["id", "entry_id", "length", "sub", "year", "org"]].copy()
        t.insert(0, "set", k)
        rows.append(t)
    pd.concat(rows).to_csv(DATA_OUT / "S5_chains.csv", index=False)
    (DATA_OUT / "S5_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    return summ


if __name__ == "__main__":
    main()
