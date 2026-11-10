#!/usr/bin/env python3
"""Supplementary Fig. S16 -- the MAD far-outlier flag of the shift companion table.

``is_far_outlier = |mad_z| > k`` with ``mad_z = (val_ppm - median) / MAD`` per
(residue type, atom), rounded to 0.01; MAD is the raw median absolute deviation (NOT
scaled by 1.4826), and the median/MAD reference table is frozen in the code repository
(``trizod/shifts/mad_reference.csv``, embedded in the Parquet metadata). k = 8 for C, CA
and N, 12 for CB. Only the raw ``val_ppm`` of C, CA, CB and N is assessed; the N of
residue 1 (the free N-terminal amine) is not. Nothing is filtered.

Panels
  a  |mad_z| per assessed atom (log-log histogram) with the threshold; flagged counts.
  b  Flags per residue type, stacked by atom, with the flag rate (flags per 1,000
     assessed shifts of that residue type).
  c  Per cumulative tier: flags per 1,000 assessed shifts and the share of chains with
     at least one flag.
  d  Flag rate per atom on chains whose assembly contains a metal (``has_metal``) and on
     all other chains; the share of all flags that falls on metal-containing chains is
     printed against their share of the assessed shifts.

Run:  uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s16_mad_flag.py \
          --out-dir <draft>/Figures/paper
Outputs: <out-dir>/S16.pdf and <out-dir>/suppfigs_a_data/S16*_*.csv / S16_summary.json.
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
import pyarrow.parquet as pq  # noqa: E402
import style_paper as sp  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

TIERS = sp.TIERS
DATA_SUBDIR = "suppfigs_a_data"  # under --out-dir
ATOMS = ["C", "CA", "CB", "N"]
K = {"C": 8.0, "CA": 8.0, "CB": 12.0, "N": 8.0}
ATOM_COLOR = {
    "C": sp.OKABE_ITO["skyblue"],
    "CA": sp.OKABE_ITO["blue"],
    "CB": sp.OKABE_ITO["green"],
    "N": sp.OKABE_ITO["vermillion"],
}


def fmt(n) -> str:
    return f"{int(n):,}"


def load():
    meta = pq.read_schema(sp.PQ_SHIFTS).metadata or {}
    rule = meta.get(b"far_outlier_rule", b"").decode()
    t = pq.read_table(
        sp.PQ_SHIFTS,
        columns=["id", "comp_id", "atom_id", "mad_z", "is_far_outlier"],
        filters=[("atom_id", "in", ATOMS)],
    ).to_pandas()
    t = t[t.mad_z.notna()].copy()
    for c in ("id", "comp_id", "atom_id"):
        t[c] = t[c].astype("category")
    t["absz"] = t.mad_z.abs()
    t["flag"] = t.is_far_outlier.astype(bool)
    main = sp.load_main(["id", "has_metal"])
    metal = set(main.id[main.has_metal.fillna(False).astype(bool)])
    t["metal"] = t.id.astype(str).isin(metal)
    return t, rule, main


def main():
    sp.parse_args(sp.cli(__doc__))
    DATA_OUT = sp.out_dir(DATA_SUBDIR)
    t, rule, main_df = load()
    # the published flag is exactly the documented rule
    for a in ATOMS:
        s = t[t.atom_id == a]
        mism = int(((s.absz > K[a]) != s.flag).sum())
        assert mism == 0, (a, mism)
    tids = sp.load_tier_ids()
    summ: dict = {
        "assessed": len(t),
        "flagged": int(t.flag.sum()),
        "pct_flagged": round(100 * t.flag.mean(), 3),
        "by_atom": {
            a: {
                "assessed": int((t.atom_id == a).sum()),
                "flagged": int(t.flag[t.atom_id == a].sum()),
            }
            for a in ATOMS
        },
        "rule": rule,
    }
    byres = (
        t.groupby(["comp_id", "atom_id"], observed=True)
        .flag.sum()
        .unstack(fill_value=0)
    )
    byres = byres.reindex(columns=ATOMS, fill_value=0)
    n_res = t.groupby("comp_id", observed=True).size()
    byres["total"] = byres[ATOMS].sum(axis=1)
    byres["assessed"] = n_res
    byres["per_1000"] = 1000 * byres.total / byres.assessed
    byres = byres.sort_values("total", ascending=False)
    summ["by_residue"] = byres.astype(float).round(3).to_dict(orient="index")
    tier = {}
    for tr in TIERS:
        s = t[t.id.astype(str).isin(tids[tr])]
        tier[tr] = {
            "assessed": len(s),
            "flagged": int(s.flag.sum()),
            "per_1000": round(1000 * s.flag.mean(), 3),
            "chains": len(tids[tr]),
            "chains_flagged": int(s.loc[s.flag, "id"].astype(str).nunique()),
        }
        tier[tr]["pct_chains_flagged"] = round(
            100 * tier[tr]["chains_flagged"] / tier[tr]["chains"], 2
        )
    summ["tiers"] = tier
    met = {
        "flags_on_metal": int(t.flag[t.metal].sum()),
        "pct_flags_on_metal": round(100 * t.flag[t.metal].sum() / t.flag.sum(), 2),
        "pct_assessed_on_metal": round(100 * t.metal.mean(), 2),
        "per_1000_metal": {
            a: round(1000 * t.flag[(t.atom_id == a) & t.metal].mean(), 3) for a in ATOMS
        },
        "per_1000_other": {
            a: round(1000 * t.flag[(t.atom_id == a) & ~t.metal].mean(), 3)
            for a in ATOMS
        },
        "metal_chains": int(main_df.has_metal.fillna(False).sum()),
    }
    summ["metal"] = met
    print(
        json.dumps(
            {k: v for k, v in summ.items() if k not in ("by_residue", "rule")}, indent=1
        )
    )
    print(byres)

    # ---------------- figure ----------------------------------------------------------
    W, H = sp.W_DOUBLE, 101.0
    fig = sp.figure(W, H)

    def ax_mm(x, y_top, w, h):
        return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])

    # a: four histograms
    bins = np.logspace(-1, np.log10(2000), 52)
    for i, a in enumerate(ATOMS):
        ax = ax_mm(12 + i * 42, 6, 36, 28)
        z = np.clip(t.loc[t.atom_id == a, "absz"].to_numpy(), bins[0], bins[-1] * 0.999)
        h, _ = np.histogram(z, bins=bins)
        keep = bins[:-1] < K[a]
        ax.stairs(np.where(keep, h, 0), bins, fill=True, color="#BDBDBD", lw=0)
        ax.stairs(np.where(~keep, h, 0), bins, fill=True, color=ATOM_COLOR[a], lw=0)
        ax.axvline(K[a], color="#222222", lw=0.6, ls=(0, (2, 1.5)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-1, 2000)
        ax.set_ylim(0.7, 3e6)
        ax.set_xticks([0.1, 1, 10, 100, 1000])
        ax.set_xticklabels(["≤0.1", "1", "10", "100", "1,000"])
        ax.minorticks_off()
        ax.set_yticks([1, 1e2, 1e4, 1e6])
        ab = summ["by_atom"][a]
        ax.set_title(a, fontsize=6.5, loc="left", pad=1.5)
        ax.text(
            0.98,
            0.97,
            f"k = {K[a]:g}\n{fmt(ab['flagged'])} of {fmt(ab['assessed'])}\n"
            f"({100 * ab['flagged'] / ab['assessed']:.2f} %)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=4.8,
            color="#333333",
        )
        ax.set_xlabel("|mad_z| (raw-MAD units)")
        if i == 0:
            ax.set_ylabel("Shifts")
    sp.fig_label(fig, "a", 1, 3)

    # b: per residue type
    ax = ax_mm(16, 46, 62, 49)
    res = list(byres.index)
    ys = np.arange(len(res))[::-1]
    for y, r in zip(ys, res):
        left = 0
        for a in ATOMS:
            n = byres.loc[r, a]
            ax.barh(y, n, left=left, height=0.75, color=ATOM_COLOR[a], lw=0)
            left += n
        ax.text(
            left + 15,
            y,
            f"{fmt(left)}  ({byres.loc[r, 'per_1000']:.1f}‰)",
            va="center",
            ha="left",
            fontsize=4.5,
            color="#333333",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([r.capitalize() for r in res], fontsize=5.0)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(res) - 0.4)
    ax.set_xlim(0, byres.total.max() * 1.42)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Flagged shifts (rate per 1,000 assessed)")
    ax.legend(
        handles=[Patch(color=ATOM_COLOR[a], label=a) for a in ATOMS],
        loc="lower right",
        fontsize=4.8,
        handlelength=1.0,
        borderaxespad=0.2,
        ncol=2,
        columnspacing=0.8,
    )
    sp.fig_label(fig, "b", 1, 41)

    # c: per tier
    ax = ax_mm(96, 46, 30, 40)
    x = np.arange(len(TIERS))
    v = [tier[tr]["per_1000"] for tr in TIERS]
    ax.bar(x, v, width=0.66, color=[sp.TIER_COLORS[tr] for tr in TIERS], lw=0)
    for xi, tr in zip(x, TIERS):
        ax.text(
            xi,
            tier[tr]["per_1000"] + 0.03,
            f"{tier[tr]['per_1000']:.2f}",
            ha="center",
            va="bottom",
            fontsize=4.8,
        )
        ax.text(
            xi,
            -0.2,
            f"{tier[tr]['pct_chains_flagged']:.1f} %\n({fmt(tier[tr]['chains_flagged'])})",
            ha="center",
            va="top",
            fontsize=4.5,
            color=sp.TIER_COLORS[tr],
            transform=ax.transData,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, max(v) * 1.25)
    ax.set_ylabel("Flags per 1,000 assessed shifts")
    ax.text(
        0.5,
        -0.27,
        "chains with ≥1 flag (%, n)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=4.8,
        color="#444444",
    )
    for xi, tr in zip(x, TIERS):
        ax.text(
            xi,
            max(v) * 1.2,
            tr[:3] + ".",
            ha="center",
            va="top",
            fontsize=4.8,
            color=sp.TIER_COLORS[tr],
        )
    sp.fig_label(fig, "c", 82, 41)

    # d: metal vs other
    ax = ax_mm(144, 46, 34, 40)
    xx = np.arange(len(ATOMS))
    w = 0.38
    ax.bar(
        xx - w / 2,
        [met["per_1000_metal"][a] for a in ATOMS],
        width=w,
        color=[ATOM_COLOR[a] for a in ATOMS],
        lw=0,
    )
    ax.bar(
        xx + w / 2,
        [met["per_1000_other"][a] for a in ATOMS],
        width=w,
        color="white",
        edgecolor=[ATOM_COLOR[a] for a in ATOMS],
        lw=0.7,
    )
    ax.set_xticks(xx)
    ax.set_xticklabels(ATOMS)
    ax.set_ylabel("Flags per 1,000 assessed shifts")
    ax.legend(
        handles=[
            Patch(facecolor="#777777", label="metal in assembly"),
            Patch(facecolor="white", edgecolor="#777777", lw=0.7, label="other chains"),
        ],
        loc="upper left",
        fontsize=4.8,
        handlelength=1.0,
        borderaxespad=0.2,
    )
    ax.text(
        0.02,
        0.76,
        f"metal-containing chains carry\n{met['pct_flags_on_metal']:.1f} % of flags, "
        f"{met['pct_assessed_on_metal']:.1f} %\nof assessed shifts",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.7,
        color="#333333",
    )
    ymax = max(max(met["per_1000_metal"].values()), max(met["per_1000_other"].values()))
    ax.set_ylim(0, ymax * 1.9)
    sp.fig_label(fig, "d", 132, 41)

    out = sp.save(fig, "S16")
    print("wrote", out)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    byres.to_csv(DATA_OUT / "S16b_flags_by_residue.csv")
    hist = {
        a: np.histogram(
            np.clip(
                t.loc[t.atom_id == a, "absz"].to_numpy(), bins[0], bins[-1] * 0.999
            ),
            bins=bins,
        )[0].tolist()
        for a in ATOMS
    }
    pd.DataFrame({"bin_lo": bins[:-1], "bin_hi": bins[1:], **hist}).to_csv(
        DATA_OUT / "S16a_absz_hist.csv", index=False
    )
    (DATA_OUT / "S16_summary.json").write_text(
        json.dumps(summ, indent=1, default=float)
    )
    return summ


if __name__ == "__main__":
    main()
