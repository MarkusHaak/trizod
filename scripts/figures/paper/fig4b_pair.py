#!/usr/bin/env python3
"""Fig. 4b -- one protein measured at two temperatures, chosen by rule, and its denominator.

The example is the top-ranked pair of ``fig3_replicates.one_factor_pairs()`` (rule in that
function's docstring and in blueprint 6.5): two non-bound chains of one sequence from one
study, deposited in different BMRB entries, in the same physical state, without cosolvent,
membrane mimetic, solid sample or high pressure, with deposited pH, temperature and ionic
strength, that differ in exactly ONE factor (temperature: |dT| >= 10 K at |dpH| <= 0.2;
pH: |dpH| >= 1 at |dT| <= 2 K; ionic strength equal), with >= 50 co-scored residues. Pairs
rank by their longest run of residues that cross from G > 0.65 in one chain to G < 0.35 in
the other, if the rest of the chain agrees (median |dG| < 0.06) and the median shift counts
k differ by at most 2. Nothing is picked by hand: the generator draws whatever pair ranks
first, and fails loudly if none ranks.

Left: the unfiltered G-score profiles of the two chains (Parquet labels, the labels of the
condition comparison in Fig. 4a), solid for the lower and dashed for the higher value of the
changed factor; every crossing residue is shaded; below, the shift count k of each chain per
residue on the shared k scale, sp.K_CMAP (blank = residue not scored). Right: the denominator --
the per-pair mean |dG| of every qualifying pair (the measure of Fig. 4a), one dot per pair,
with their median and the drawn pair marked, so the example is seen in its distribution.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig4b_pair.py \\
        --out-dir <draft>/Figures/paper
Outputs: ``<out-dir>/fig4b.pdf`` (78 x 55 mm, placed at 1:1);
``<out-dir>/source_data/fig4b_profiles.csv`` (one row per residue) and
``fig4b_pairs.csv`` (every qualifying pair; the Supplementary Table 14 data); prints the
plotted numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig3_replicates as fr  # noqa: E402
import style_paper as sp  # noqa: E402

# italic symbols (G, k) in the text face, as in fig1a_ubiquitin.py
sp.plt.rcParams.update(
    {
        "mathtext.fontset": "custom",
        "mathtext.it": "Arial:italic",
        "mathtext.rm": "Arial",
    }
)

W, H = sp.W_SINGLE, 55.0  # mm; one cell of the 2 x 2 Fig. 4 grid
# profile + k strips (left), denominator strip (right); all in mm from the bottom-left
PROF = (9.7, 16.0, 45.3, 35.5)  # x0, y0, w, h
K_H, K_GAP = 2.4, 0.5  # height of one k strip, gap between strips
K_TOP = PROF[1] - 1.2  # top of the upper k strip
DEN = (62.8, 21.5, 7.0, 30.0)  # the denominator strip
CBAR = (60.5, 10.2, 14.0, 1.6)  # k colour bar, level with the k strips
SHADE = (
    0.0,
    0.0,
    0.0,
    0.09,
)  # crossing residues: ink at 9 % (a highlight, not an entity)
REF = sp.CAT["light"]  # G = 0.35 / 0.65 guides
LINE = {"a": {"ls": "-", "lw": 1.0}, "b": {"ls": (0, (2.6, 1.3)), "lw": 1.0}}
JITTER_SEED = 0


def select(D: fr.Data | None = None):
    """``(D, P, top)``: the qualifying pairs and the ranked top pair (fails if none ranks)."""
    D = D or fr.Data()
    P = fr.one_factor_pairs(D)
    top = fr.top_one_factor_pair(P)
    if top is None:
        raise SystemExit(
            "fig4b: no one-factor pair ranks; draw the distribution instead"
        )
    return D, P, top


def profiles(D: fr.Data, top) -> pd.DataFrame:
    """One row per residue: G and k of both chains, crossing flags (1-based residue)."""
    ga, gb = D.G_unf[top.a], D.G_unf[top.b]
    ka, kb = D.K_unf[top.a], D.K_unf[top.b]
    with np.errstate(invalid="ignore"):
        cross_a = (ga > fr.CROSS_HI) & (gb < fr.CROSS_LO)  # a disordered, b ordered
        cross_b = (gb > fr.CROSS_HI) & (ga < fr.CROSS_LO)
    seq = top.sequence
    return pd.DataFrame(
        {
            "residue": np.arange(1, len(ga) + 1),
            "aa": list(seq),
            f"g_{top.a}": ga,
            f"g_{top.b}": gb,
            f"k_{top.a}": np.where(np.isfinite(ga), ka, np.nan),
            f"k_{top.b}": np.where(np.isfinite(gb), kb, np.nan),
            "crossing": cross_a | cross_b,
            "disordered_in": np.where(cross_a, top.a, np.where(cross_b, top.b, "")),
        }
    )


def _value(top, side: str) -> str:
    """The changed factor's value of chain ``side`` ('a' or 'b') as a label: '278 K'."""
    if top.factor == "temperature":
        return f"{top[f'temperature_{side}']:.0f} K"
    return f"pH {top[f'ph_{side}']:.1f}"


def _ax(fig, x0, y0, w, h, **kw):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H], **kw)


def _shade(ax, res, y0=None, y1=None):
    for r in res:
        ax.axvspan(r - 0.5, r + 0.5, color=SHADE, lw=0, zorder=0)


def _label_run(ax, g, lo, hi, text, room=0.13):
    """Name profile ``g`` at its run of residues lo..hi (1-based): above the run's highest
    point when there is ``room`` (data units) below the top of the axes, else to the right
    of the run's end."""
    i = int(np.nanargmax(g[lo - 1 : hi])) + lo
    if g[i - 1] + room < ax.get_ylim()[1]:
        sp.direct_label(
            ax, i, g[i - 1], text, dx_mm=0, dy_mm=0.8, ha="center", va="bottom"
        )
    else:
        sp.direct_label(
            ax,
            hi + 0.5,
            np.nanmax(g[lo - 1 : hi]),
            text,
            dx_mm=0.8,
            ha="left",
            va="top",
        )


def draw(Pr: pd.DataFrame, P: pd.DataFrame, top):
    fig = sp.figure(W, H)
    x = Pr["residue"].to_numpy()
    n = len(x)
    ga, gb = Pr[f"g_{top.a}"].to_numpy(), Pr[f"g_{top.b}"].to_numpy()
    cross = x[Pr["crossing"].to_numpy()]

    # --- profiles -------------------------------------------------------------------
    ax = _ax(fig, *PROF)
    _shade(ax, cross)
    for yref in (fr.CROSS_LO, fr.CROSS_HI):
        ax.axhline(yref, color=REF, lw=0.5, ls=(0, (1.5, 1.5)), zorder=1)
    ax.plot(x, ga, color=sp.INK, zorder=3, **LINE["a"])
    ax.plot(x, gb, color=sp.INK, zorder=3, **LINE["b"])
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, fr.CROSS_LO, fr.CROSS_HI, 1.0])
    ax.set_yticklabels(["0", f"{fr.CROSS_LO:g}", f"{fr.CROSS_HI:g}", "1"])
    ax.set_ylabel("G-score")
    ax.tick_params(axis="x", labelbottom=False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    # direct labels: each line named above its own highest stretch where it alone is high
    sel_a = Pr["disordered_in"].to_numpy() == top.a
    sel_b = Pr["disordered_in"].to_numpy() == top.b
    for g, sel, side in ((ga, sel_a, "a"), (gb, sel_b, "b")):
        if sel.any():
            runs = fr._runs(sel)
            s, e = max(runs, key=lambda r: r[1] - r[0])
            _label_run(ax, g, s + 1, e + 1, _value(top, side))

    # --- k strips -------------------------------------------------------------------
    cmap = mpl.colormaps[sp.K_CMAP].copy()
    cmap.set_bad("white")
    norm = mpl.colors.Normalize(*sp.K_RANGE)
    for j, side in enumerate(("a", "b")):
        cid = top[side]
        y0 = K_TOP - (j + 1) * K_H - j * K_GAP
        axk = _ax(fig, PROF[0], y0, PROF[2], K_H, sharex=ax)
        k = Pr[f"k_{cid}"].to_numpy(float)[None, :]
        axk.imshow(np.ma.masked_invalid(k), cmap=cmap, norm=norm, aspect="auto",
                   extent=(0.5, n + 0.5, 0, 1), interpolation="nearest", zorder=1)  # fmt: skip
        axk.set_yticks([0.5])
        axk.set_yticklabels([_value(top, side)])
        axk.tick_params(axis="y", length=0)
        for s_ in axk.spines.values():
            s_.set_visible(False)
        if j == 0:
            axk.tick_params(axis="x", labelbottom=False, length=0)
        else:
            axk.set_xlabel("Residue")
            ticks = [1, *range(20, n + 1, 20)]
            axk.set_xticks(ticks)
    # k colour bar
    cax = _ax(fig, *CBAR)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=sp.K_CMAP), cax=cax,
                      orientation="horizontal", ticks=[1, 11, 21])  # fmt: skip
    cb.outline.set_visible(False)
    cax.tick_params(length=1.5, pad=1.0)
    cax.set_title("Shift count $\\mathit{k}$", fontsize=sp.FS_BODY, pad=2.0)

    # --- denominator: every qualifying pair ------------------------------------------
    axd = _ax(fig, *DEN)
    v = P["mean_absdg"].to_numpy(float)
    rng = np.random.default_rng(JITTER_SEED)
    jit = rng.uniform(-0.28, 0.28, len(v))
    is_top = ((P["a"] == top.a) & (P["b"] == top.b)).to_numpy()
    axd.plot(jit[~is_top], v[~is_top], ls="", marker="o", ms=2.6, mfc="white",
             mec=sp.INK, mew=0.6, zorder=3)  # fmt: skip
    axd.plot(jit[is_top], v[is_top], ls="", marker="o", ms=3.6, mfc=sp.INK, mec=sp.INK, zorder=4)  # fmt: skip
    med = float(np.median(v))
    axd.plot([-0.45, 0.45], [med, med], color=sp.INK, lw=1.0, zorder=2)
    sp.direct_label(axd, -0.45, med, "median", dx_mm=-0.6, ha="right")
    sp.direct_label(
        axd, jit[is_top][0] - 0.05, v[is_top][0], "shown", dx_mm=-1.2, ha="right"
    )
    axd.set_xlim(-0.6, 0.6)
    ymax = 0.1 * np.ceil(v.max() / 0.1 + 1e-9)
    axd.set_ylim(0, ymax)
    # no tick label at the top, where the "n pairs" title sits
    yt = np.round(np.arange(0, ymax - 0.05, 0.1), 1)
    axd.set_yticks(yt)
    axd.set_yticklabels([f"{v:g}" for v in yt])
    axd.yaxis.tick_right()
    axd.yaxis.set_label_position("right")
    axd.spines["left"].set_visible(False)
    axd.spines["right"].set_visible(True)
    axd.spines["bottom"].set_visible(False)
    axd.set_xticks([])
    axd.set_ylabel("|ΔG| of the pair")  # the quantity of Fig. 4a, named the same way
    axd.set_title(f"{len(v)} pairs", fontsize=sp.FS_BODY, pad=2.0)

    sp.fig_label(fig, "b", 0.0, 0.0)
    return fig


def main(argv=None) -> None:
    sp.parse_args(sp.cli(__doc__), argv)
    D, P, top = select()
    Pr = profiles(D, top)
    out = sp.out_dir("source_data")
    out.mkdir(parents=True, exist_ok=True)
    Pr.to_csv(out / "fig4b_profiles.csv", index=False, float_format="%.4f")
    P.drop(columns=["sequence"]).to_csv(
        out / "fig4b_pairs.csv", index=False, float_format="%.4f"
    )
    pdf = sp.save(draw(Pr, P, top), "fig4b")
    nums = {
        "rule": "fig3_replicates.one_factor_pairs (blueprint 6.5); top = rank 1",
        "n_pairs": int(len(P)),
        "n_temperature": int((P.factor == "temperature").sum()),
        "n_ph": int((P.factor == "pH").sum()),
        "n_ranked": int(P["rankable"].sum()),
        "median_mean_absdg": round(float(P["mean_absdg"].median()), 4),
        "share_pairs_below_top": round(float((P["mean_absdg"] < top.mean_absdg).mean()), 3),
        "top": {
            "a": top.a, "b": top.b, "factor": top.factor,
            "values": [_value(top, "a"), _value(top, "b")],
            "ph": [top.ph_a, top.ph_b], "ionic_strength": [top.ionic_strength_a, top.ionic_strength_b],
            "n_coscored": int(top.n_coscored), "mean_absdg": round(float(top.mean_absdg), 4),
            "rest_median_absdg": round(float(top.rest_median_absdg), 4),
            "cross_run": [int(top.cross_start), int(top.cross_end)],
            "n_cross_residues": int(top.n_cross_residues),
            "crossing_residues": Pr.loc[Pr.crossing, "residue"].tolist(),
            "medk": [top.medk_a, top.medk_b], "citation_title": top.citation_title,
        },
    }  # fmt: skip
    (out / "fig4b_numbers.json").write_text(
        json.dumps(nums, indent=1, ensure_ascii=False)
    )
    print(json.dumps(nums, indent=1, ensure_ascii=False))
    print(f"wrote {pdf}", file=sys.stderr)


if __name__ == "__main__":
    main()
