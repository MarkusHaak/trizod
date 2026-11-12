#!/usr/bin/env python3
"""Supplementary Fig. S15 -- the chemical-shift companion table (``trizod_shifts.parquet``).

Panels
  a  Every released shift by class (backbone = the 12 atom IDs the scorer reads; side
     chain = everything else) x nucleus, stacked by ``offset_source``: which estimators
     contributed the referencing offset subtracted from the row (``lacs_only``,
     ``lacs+potenci``, ``potenci_only``), a measured zero offset (``none``), or no
     transferable offset, so no corrected value is published (``not_transferable``:
     every side-chain 1H and 15N, and side-chain 13C on the chains with |LACS CA| > 5 ppm).
  b  Side-chain completeness per residue type and cumulative tier: observed side-chain
     H/C/N atoms divided by the expected ones over every canonical residue of the
     sequence (the "all residues, full expected set" definition of
     ``scripts/sidechain_coverage.py``, whose atom lists this script imports). ALA and GLY
     have no side-chain atoms beyond the backbone set and are omitted.
  c  Methyl (ILVMA 13C: ILE CD1, LEU CD1/CD2, VAL CG1/CG2, MET CE, ALA CB; labelled in
     Greek, e.g. Ile Cdelta1) and aromatic ring 13C/15N coverage per tier.
  d  Side-chain shifts per chain, cumulative distribution per tier (chains with none sit
     at the left edge).

All counts are recomputed here from the shifts Parquet (one row per chain x residue x
atom; duplicate statements of one atom count once for coverage). ALA CB is read from the
backbone rows of the Parquet, which keep values the scorer drops (Val_err > 1.3 ppm,
ambiguity codes outside {1, 2}); ``sidechain_coverage.py`` counts it from the scorer's
backbone mask, so the ALA methyl coverage can differ slightly from
``docs/dataset/sidechain-coverage.md``.

Run:  uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s15_shift_companion.py \
          --out-dir <draft>/Figures/paper
Outputs: <out-dir>/S15.pdf and <out-dir>/suppfigs_a_data/S15a_class_nucleus_source.csv / S15_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style_paper as sp  # noqa: E402

sys.path.insert(0, str(sp.REPO / "scripts"))
sys.path.insert(0, str(sp.REPO))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import sidechain_coverage as scc  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from trizod.constants import AA1TO3  # noqa: E402

TIERS = sp.TIERS
DATA_SUBDIR = "suppfigs_a_data"  # under --out-dir
SOURCES = ["lacs_only", "lacs+potenci", "potenci_only", "none", "not_transferable"]
SRC_LABEL = {
    "lacs_only": "LACS only",
    "lacs+potenci": "LACS + POTENCI/AIC",
    "potenci_only": "POTENCI/AIC only",
    "none": "zero offset",
    "not_transferable": "raw only",
}
# Method colours as in Supplementary Fig. S6 (LACS sky blue, POTENCI/AIC dark grey); both
# methods = sky blue hatched in dark grey, so no two corrected classes share a grey. No
# correction: zero offset white (grey outline), raw only light grey. The Okabe-Ito blue,
# orange, green and vermillion keep their fixed meanings (test set, CheZOD, training,
# removed).
SRC_STYLE = {  # facecolor, edgecolor (also the hatch colour), hatch
    "lacs_only": (sp.OKABE_ITO["skyblue"], "white", None),
    "lacs+potenci": (sp.OKABE_ITO["skyblue"], sp.CAT["neutral"], "//////"),
    "potenci_only": (sp.CAT["neutral"], "white", None),
    "none": ("white", "#8C8C8C", None),
    "not_transferable": ("#C4C4C4", "white", None),
}
# plain element symbols: mathtext mass numbers would be lifted to the 6 pt floor and
# inflate the base glyph (the caption names 1H, 13C and 15N)
NUC_LABEL = {"H": "H", "C": "C", "N": "N"}
# side-chain probe atoms in Greek (BMRB names: ALA CB, ILE CD1, LEU CD1/CD2, MET CE,
# VAL CG1/CG2)
METHYL_LABEL = (
    ("ALA", "Ala Cβ"),
    ("ILE", "Ile Cδ1"),
    ("LEU", "Leu Cδ1/2"),
    ("MET", "Met Cε"),
    ("VAL", "Val Cγ1/2"),
)


def fmt(n) -> str:
    return f"{int(n):,}"


def load():
    sh = pq.read_table(
        sp.PQ_SHIFTS,
        columns=[
            "id",
            "seq_id",
            "comp_id",
            "atom_id",
            "atom_type",
            "is_backbone",
            "offset_source",
        ],
    ).to_pandas()
    for c in ("id", "comp_id", "atom_id", "atom_type", "offset_source"):
        sh[c] = sh[c].astype("category")
    main = sp.load_main(["id", "sequence"])
    tids = sp.load_tier_ids()
    return sh, main, tids


def residue_counts(main) -> pd.DataFrame:
    """chains x 20 canonical residue counts (AA3 order of sidechain_coverage)."""
    rows = []
    for seq in main.sequence:
        c = np.zeros(len(scc.AA3), dtype=np.int64)
        for a in seq:
            a3 = AA1TO3.get(a)
            if a3 is not None:
                c[scc.AA_IDX[a3]] += 1
        rows.append(c)
    return pd.DataFrame(np.array(rows), index=main.id.to_numpy(), columns=scc.AA3)


def observed_counts(sh, atom_sets: dict, backbone: bool) -> pd.DataFrame:
    """chains x 20 counts of distinct observed (seq_id, atom_id) within ``atom_sets``."""
    pairs = [(r, a) for r, atoms in atom_sets.items() for a in atoms]
    if not pairs:
        return pd.DataFrame(columns=scc.AA3)
    key = pd.DataFrame(pairs, columns=["comp_id", "atom_id"])
    sub = sh[sh.is_backbone == backbone][["id", "seq_id", "comp_id", "atom_id"]].copy()
    for c in ("id", "comp_id", "atom_id"):
        sub[c] = sub[c].astype(str)
    sub = sub.merge(key, on=["comp_id", "atom_id"], how="inner")
    sub = sub.drop_duplicates(["id", "seq_id", "atom_id"])
    t = sub.groupby(["id", "comp_id"]).size().unstack(fill_value=0)
    return t.reindex(columns=scc.AA3, fill_value=0)


def main():
    sp.parse_args(sp.cli(__doc__))
    DATA_OUT = sp.out_dir(DATA_SUBDIR)
    sh, main_df, tids = load()
    summ: dict = {"rows": len(sh)}

    # ---- a: class x nucleus x offset_source ----------------------------------------
    sh["cls"] = np.where(sh.is_backbone, "backbone", "side chain")
    ct = pd.crosstab([sh.cls, sh.atom_type.astype(str)], sh.offset_source.astype(str))
    ct = ct.reindex(columns=SOURCES, fill_value=0)
    summ["class_nucleus_source"] = {
        f"{a} {b}": r.to_dict() for (a, b), r in ct.iterrows()
    }
    print(ct)

    # ---- b-d: per-chain counters ---------------------------------------------------
    res = residue_counts(main_df)
    sc_rows = sh[~sh.is_backbone]
    obs = observed_counts(sh, scc.EXPECTED_SIDECHAIN, backbone=False)
    methyl_sc = {k: v for k, v in scc.METHYL_CARBONS.items() if k != "ALA"}
    methyl = observed_counts(sh, methyl_sc, backbone=False).add(
        observed_counts(sh, {"ALA": ("CB",)}, backbone=True), fill_value=0
    )
    arom = observed_counts(sh, scc.AROMATIC_RING, backbone=False)
    n_sc = sc_rows.groupby("id", observed=False).size()
    n_sc = n_sc.reindex(main_df.id.to_numpy(), fill_value=0)

    comp = {}
    comp_overall = {}
    meth = {}
    aro = {}
    sc_stats = {}
    for t in TIERS:
        ids = [i for i in main_df.id if i in tids[t]]
        r = res.loc[ids].sum()
        o = obs.reindex(ids, fill_value=0).sum()
        comp[t] = {
            aa: (o[aa] / (r[aa] * scc.EXPECTED_N[i]) if scc.EXPECTED_N[i] else np.nan)
            for i, aa in enumerate(scc.AA3)
        }
        comp_overall[t] = float(o.sum() / (r.to_numpy() @ scc.EXPECTED_N))
        m = methyl.reindex(ids, fill_value=0).sum()
        meth[t] = {
            aa: float(m[aa] / (r[aa] * scc.METHYL_N[scc.AA_IDX[aa]]))
            for aa in sorted(scc.METHYL_CARBONS)
        }
        meth[t]["overall"] = float(m.sum() / (r.to_numpy() @ scc.METHYL_N))
        a = arom.reindex(ids, fill_value=0).sum()
        aro[t] = {
            aa: float(a[aa] / (r[aa] * scc.AROMATIC_N[scc.AA_IDX[aa]]))
            for aa in sorted(scc.AROMATIC_RING)
        }
        aro[t]["overall"] = float(a.sum() / (r.to_numpy() @ scc.AROMATIC_N))
        v = n_sc.loc[ids]
        sc_stats[t] = {
            "chains": len(ids),
            "with_sidechain": int((v > 0).sum()),
            "share_with_sidechain": round(float((v > 0).mean()), 4),
            "total": int(v.sum()),
            "median": float(v.median()),
            "q1": float(v.quantile(0.25)),
            "q3": float(v.quantile(0.75)),
        }
    summ["completeness_overall"] = {t: round(v, 3) for t, v in comp_overall.items()}
    summ["completeness_by_residue"] = {
        t: {k: round(float(x), 3) for k, x in v.items() if np.isfinite(x)}
        for t, v in comp.items()
    }
    summ["methyl_coverage"] = {
        t: {k: round(100 * x, 1) for k, x in v.items()} for t, v in meth.items()
    }
    summ["aromatic_coverage"] = {
        t: {k: round(x, 3) for k, x in v.items()} for t, v in aro.items()
    }
    summ["sidechain_per_chain"] = sc_stats
    print(
        json.dumps(
            {
                k: summ[k]
                for k in (
                    "completeness_overall",
                    "methyl_coverage",
                    "aromatic_coverage",
                    "sidechain_per_chain",
                )
            },
            indent=1,
        )
    )

    # ---------------- figure ----------------------------------------------------------
    W, H = sp.W_DOUBLE, 107.0
    fig = sp.figure(W, H)

    def ax_mm(x, y_top, w, h):
        return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])

    # a stacked bars
    ax = ax_mm(20, 17, 52, 31)
    rows = [
        ("backbone", "H"),
        ("backbone", "C"),
        ("backbone", "N"),
        ("side chain", "H"),
        ("side chain", "C"),
        ("side chain", "N"),
    ]
    ys = np.array([5.3, 4.3, 3.3, 2.0, 1.0, 0.0])
    for y, (cls, nuc) in zip(ys, rows):
        left = 0.0
        tot = int(ct.loc[(cls, nuc)].sum())
        for src in SOURCES:
            n = int(ct.loc[(cls, nuc), src])
            fc, ec, hatch = SRC_STYLE[src]
            ax.barh(y, n / 1e6, left=left, height=0.72, color=fc, edgecolor=ec,
                    hatch=hatch, lw=0.4)  # fmt: skip
            left += n / 1e6
        ax.text(
            left + 0.05,
            y,
            fmt(tot),
            va="center",
            ha="left",
            fontsize=sp.FS_SMALL,
            color="#333333",
        )
        ax.text(
            -0.02,
            y,
            NUC_LABEL[nuc],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=sp.FS_TICK,
        )
    for yc, lab in ((4.3, "backbone"), (1.0, "side chain")):
        ax.text(
            -0.15,
            yc,
            lab,
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="center",
            rotation=90,
            fontsize=sp.FS_TICK,
        )
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, 5.9)
    ax.set_xlim(0, 5.0)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel("Shifts (millions)")
    ax.legend(
        handles=[
            Patch(facecolor=SRC_STYLE[s_][0], edgecolor=SRC_STYLE[s_][1] if s_ in ("none", "lacs+potenci") else SRC_STYLE[s_][0],
                  hatch=SRC_STYLE[s_][2], lw=0.4, label=SRC_LABEL[s_])
            for s_ in SOURCES
        ],
        loc="lower left",
        bbox_to_anchor=(-0.24, 1.03),
        ncol=2,
        handlelength=1.4,
        handleheight=0.9,
        handletextpad=0.4,
        borderaxespad=0.0,
        labelspacing=0.35,
        columnspacing=1.0,
    )  # fmt: skip
    sp.fig_label(fig, "a", 0, 1)

    # b completeness heatmap (residue type x tier)
    aas = [aa for i, aa in enumerate(scc.AA3) if scc.EXPECTED_N[i] > 0]
    mat = np.array([[comp[t][aa] for t in TIERS] for aa in aas])
    mat = np.vstack([mat, [comp_overall[t] for t in TIERS]])
    labels = [f"{aa.capitalize()} ({scc.EXPECTED_N[scc.AA_IDX[aa]]})" for aa in aas] + [
        "all"
    ]
    ax = ax_mm(93, 13, 28, 84)
    ax.imshow(mat, aspect="auto", cmap="Greys", vmin=0, vmax=0.8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=sp.FS_SMALL,
                color="white" if v > 0.45 else "#222222",
                fontweight="bold" if i == mat.shape[0] - 1 else "normal",
            )
    ax.axhline(len(aas) - 0.5, color="white", lw=1.2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=sp.FS_TICK)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels(
        TIERS, fontsize=sp.FS_TICK, rotation=35, ha="left", rotation_mode="anchor"
    )
    ax.tick_params(length=0, pad=1.5)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.text(
        0.5,
        -0.012,
        "Observed / expected\nside-chain atoms",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=sp.FS_BODY,
    )
    sp.fig_label(fig, "b", 78, 1)

    # c methyl + aromatic coverage
    ax = ax_mm(137, 13, 19, 84)
    items = [("Methyls", meth, "overall")] + [
        (lab, meth, aa) for aa, lab in METHYL_LABEL
    ]
    items += [("Aromatics", aro, "overall")] + [
        (aa.capitalize(), aro, aa) for aa in ("HIS", "PHE", "TRP", "TYR")
    ]
    ypos = []
    y = 0.0
    for it in items:
        if it[0] == "Aromatics":
            y += 0.6  # gap between the methyl and the aromatic block
        ypos.append(y)
        y += 1
    ypos = -np.array(ypos)
    for j, t in enumerate(TIERS):
        xs = [100 * src[t][key] for _, src, key in items]
        ax.scatter(
            xs,
            ypos + (1.5 - j) * 0.14,
            s=9,
            marker=sp.TIER_MARKERS[t],
            color=sp.TIER_COLORS[t],
            lw=0,
            zorder=3,
        )
    for yy in ypos:
        ax.axhline(yy, color="#EEEEEE", lw=0.4, zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([it[0] for it in items], fontsize=sp.FS_TICK)
    for lab, it in zip(ax.get_yticklabels(), items):
        if it[0] in ("Methyls", "Aromatics"):
            lab.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Coverage (%)")
    # tier colours and markers: the shared legend in panel d
    sp.fig_label(fig, "c", 122, 1)

    # d side-chain shifts per chain ECDF
    ax = ax_mm(20, 64, 52, 33)
    for t in TIERS:
        ids = [i for i in main_df.id if i in tids[t]]
        v = np.sort(n_sc.loc[ids].to_numpy()) + 1
        y = np.arange(1, len(v) + 1) / len(v)
        ax.step(np.r_[1, v], np.r_[0, y], where="post", color=sp.TIER_COLORS[t], lw=0.9)
    ax.set_xscale("log")
    ax.set_xlim(1, 3000)
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 2, 11, 101, 1001])
    ax.set_xticklabels(["0", "1", "10", "100", "1,000"])
    ax.minorticks_off()
    yt = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(yt, [f"{v:g}" for v in yt])
    ax.set_xlabel("Side-chain shifts per chain")
    ax.set_ylabel("Cumulative share of chains")
    # one tier legend for c (markers) and d (lines); shares and medians per tier are in
    # S15_summary.json
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=sp.TIER_MARKERS[t],
                color=sp.TIER_COLORS[t],
                lw=0.9,
                ms=3.5,
                label=t,
            )
            for t in TIERS
        ],
        loc="upper left",
        borderaxespad=0.1,
        handlelength=1.8,
    )
    sp.fig_label(fig, "d", 0, 59)

    out = sp.save(fig, "S15")
    print("wrote", out)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    ct.to_csv(DATA_OUT / "S15a_class_nucleus_source.csv")
    (DATA_OUT / "S15_summary.json").write_text(
        json.dumps(summ, indent=1, default=float)
    )
    return summ


if __name__ == "__main__":
    main()
