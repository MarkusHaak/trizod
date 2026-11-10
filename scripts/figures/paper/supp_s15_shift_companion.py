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
  c  Methyl (ILVMA 13C: ILE CD1, LEU CD1/CD2, VAL CG1/CG2, MET CE, ALA CB) and aromatic
     ring 13C/15N coverage per tier.
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
    "none": "zero offset (none)",
    "not_transferable": "raw only (not_transferable)",
}
SRC_COLOR = {
    "lacs_only": sp.OKABE_ITO["skyblue"],
    "lacs+potenci": sp.OKABE_ITO["blue"],
    "potenci_only": sp.OKABE_ITO["green"],
    "none": "#D9D9D9",
    "not_transferable": "#8C8C8C",
}
NUC_LABEL = {"H": "$^{1}$H", "C": "$^{13}$C", "N": "$^{15}$N"}


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
    W, H = sp.W_DOUBLE, 104.0
    fig = sp.figure(W, H)

    def ax_mm(x, y_top, w, h):
        return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])

    # a stacked bars
    ax = ax_mm(24, 9, 66, 34)
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
        for s in SOURCES:
            n = int(ct.loc[(cls, nuc), s])
            ax.barh(y, n / 1e6, left=left, height=0.72, color=SRC_COLOR[s], lw=0)
            left += n / 1e6
        ax.text(
            left + 0.04,
            y,
            fmt(tot),
            va="center",
            ha="left",
            fontsize=4.8,
            color="#333333",
        )
        ax.text(
            -0.02,
            y,
            NUC_LABEL[nuc],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=5.5,
        )
    for yc, lab in ((4.3, "backbone"), (1.0, "side chain")):
        ax.text(
            -0.13,
            yc,
            lab,
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="center",
            rotation=90,
            fontsize=5.5,
        )
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, 5.9)
    ax.set_xlim(0, 4.5)
    ax.set_xlabel("Shifts (millions)")
    ax.legend(
        handles=[Patch(color=SRC_COLOR[s], label=SRC_LABEL[s]) for s in SOURCES],
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        fontsize=4.8,
        handlelength=1.0,
        borderaxespad=0.1,
        labelspacing=0.3,
        title="offset_source",
        title_fontsize=4.8,
    )
    sp.fig_label(fig, "a", 1, 3)

    # b completeness heatmap (residue type x tier)
    aas = [aa for i, aa in enumerate(scc.AA3) if scc.EXPECTED_N[i] > 0]
    mat = np.array([[comp[t][aa] for t in TIERS] for aa in aas])
    mat = np.vstack([mat, [comp_overall[t] for t in TIERS]])
    labels = [f"{aa.capitalize()} ({scc.EXPECTED_N[scc.AA_IDX[aa]]})" for aa in aas] + [
        "all"
    ]
    ax = ax_mm(106, 9, 30, 88)
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
                fontsize=4.7,
                color="white" if v > 0.45 else "#222222",
                fontweight="bold" if i == mat.shape[0] - 1 else "normal",
            )
    ax.axhline(len(aas) - 0.5, color="white", lw=1.2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.2)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels(
        TIERS, fontsize=5.0, rotation=35, ha="left", rotation_mode="anchor"
    )
    for lab, t in zip(ax.get_xticklabels(), TIERS):
        lab.set_color(sp.TIER_COLORS[t])
    ax.tick_params(length=0, pad=1.5)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.text(
        0.5,
        -0.012,
        "observed / expected side-chain atoms\n(expected atoms per residue in parentheses)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=4.8,
        color="#444444",
    )
    sp.fig_label(fig, "b", 92, 3)

    # c methyl + aromatic coverage
    ax = ax_mm(153, 9, 25, 86)
    items = [("ILVMA methyls", meth, "overall")] + [
        (lab, meth, aa)
        for aa, lab in (
            ("ALA", "Ala CB"),
            ("ILE", "Ile CD1"),
            ("LEU", "Leu CD1/2"),
            ("MET", "Met CE"),
            ("VAL", "Val CG1/2"),
        )
    ]
    items += [("aromatic rings", aro, "overall")] + [
        (aa.capitalize(), aro, aa) for aa in ("HIS", "PHE", "TRP", "TYR")
    ]
    ypos = []
    y = 0.0
    for it in items:
        if it[0] == "aromatic rings":
            y += 0.6  # gap between the methyl and the aromatic block
        ypos.append(y)
        y += 1
    ypos = -np.array(ypos)
    for j, t in enumerate(TIERS):
        xs = [100 * src[t][key] for _, src, key in items]
        ax.scatter(
            xs,
            ypos + (1.5 - j) * 0.14,
            s=6,
            marker=sp.TIER_MARKERS[t],
            color=sp.TIER_COLORS[t],
            lw=0,
            zorder=3,
        )
    for yy in ypos:
        ax.axhline(yy, color="#EEEEEE", lw=0.4, zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([it[0] for it in items], fontsize=5.0)
    for lab, it in zip(ax.get_yticklabels(), items):
        if it[0] in ("ILVMA methyls", "aromatic rings"):
            lab.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Coverage (%)")
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                ls="",
                marker=sp.TIER_MARKERS[t],
                color=sp.TIER_COLORS[t],
                ms=2.6,
                label=t,
            )
            for t in TIERS
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=4.6,
        handletextpad=0.1,
        borderaxespad=0.2,
        labelspacing=0.2,
        columnspacing=0.5,
    )
    sp.fig_label(fig, "c", 141, 3)

    # d side-chain shifts per chain ECDF
    ax = ax_mm(24, 62, 62, 33)
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
    ax.set_xlabel("Side-chain shifts per chain (log scale, offset by 1)")
    ax.set_ylabel("Cumulative share of chains")
    txt = "\n".join(
        f"{t}: {100 * sc_stats[t]['share_with_sidechain']:.1f} % with side chains, median {sc_stats[t]['median']:g}"
        for t in TIERS
    )
    ax.text(
        0.02,
        0.97,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=4.7,
        color="#333333",
        linespacing=1.3,
    )
    sp.fig_label(fig, "d", 1, 57)

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
