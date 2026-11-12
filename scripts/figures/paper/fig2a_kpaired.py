#!/usr/bin/env python3
"""Fig. 3b (Fig. 2b on 11 Nov 2026, Fig. 2a before) -- same residue, different shift count.

For every sequence carried by non-bound chains from at least two BMRB entries,
draw up to 30 chain pairs (seeded), drop same-entry pairs, and compare each
co-scored residue: dZ and dG are the higher-k chain minus the lower-k chain at
that residue, and dk = |k_a - k_b|. Residues are split by the replicate
consensus (median G over all chains with that sequence): ordered < 0.3,
disordered > 0.7. The dk = 0 bin is the built-in null.

This is a faithful port of the scratch analysis ``trust_first/kpaired.py``
(the source of the writing plan's preview numbers and of the ``\\kpair*``
macros in ``numbers.tex``, which ``scripts/manuscript/make_numbers.py``
recomputes with the same code): same pair sampling (``random.Random(0)`` over
``groupby('sequence')`` order), same thresholds, same bins.

The two y axes are scaled to the same fraction of each score's own
ordered-to-disordered difference (median score of ordered- vs disordered-
consensus residues in the pairs), so a shift of equal visual height is an equal
share of the label's dynamic range. Ordered residues (the message) are filled
circles, disordered residues (the control) open squares, both in ink
(``style_paper.CONTRAST``); the two series are labelled directly in the Z-score
axes, not in a legend. The per-bin pair counts are in the Source Data, not in the
artwork.

Story-blueprint redesign (12 Nov 2026): the panel is 106 mm wide (was 100 mm), so
that Fig. 3a (52 mm, ``fig2c_chezod.py``) and this panel fill the 160 mm text
width with a 2 mm gap; the shared y title now sits clear of the tick labels.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig2a_kpaired.py \\
        --out-dir <draft>/Figures/paper
Outputs: <out-dir>/fig3b_kpaired.pdf (106 x 52 mm, placed at 1:1) and
<out-dir>/source_data/fig3b_kpaired.csv; prints the plotted numbers.
``fig2c_chezod.py`` imports the row layout (height, axes top and height, x-label
baseline) from here, so the two panels of the row align.
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402

# italic k in the same sans-serif face as the text (mathtext defaults to DejaVu)
sp.plt.rcParams.update(
    {
        "mathtext.fontset": "custom",
        "mathtext.it": "Arial:italic",
        "mathtext.rm": "Arial",
    }
)

PARQUET = sp.PQ_MAIN

MAX_PAIRS_PER_SEQ = 30
SEED = 0
ORDERED_MAX = 0.3  # replicate-consensus G below this: ordered
DISORDERED_MIN = 0.7  # replicate-consensus G above this: disordered
BIN_EDGES = [-0.5, 0.5, 3, 6, 9, 21]
BIN_LABELS = ["0", "1–3", "4–6", "7–9", "≥10"]
OUT_NAME = "fig3b_kpaired"  # Fig. 3b (fig2b_kpaired.pdf on 11 Nov, fig2a.pdf before)


def residue_pairs(parquet: Path) -> pd.DataFrame:
    """Return one row per co-scored residue of each sampled replicate pair."""
    import pyarrow.parquet as pq

    df = pq.read_table(
        parquet,
        columns=["id", "entry_id", "sequence", "gscores", "zscores", "k", "is_bound"],
    ).to_pandas()
    df = df[~df.is_bound]
    rng = random.Random(SEED)
    out = []
    for _seq, g in df.groupby("sequence"):
        if g.entry_id.nunique() < 2:
            continue
        G = np.vstack([np.asarray(x, float) for x in g.gscores])
        Z = np.vstack([np.asarray(x, float) for x in g.zscores])
        K = np.vstack([np.asarray(x, float) for x in g.k])
        with np.errstate(all="ignore"):
            cons = np.nanmedian(G, axis=0)
        pairs = list(itertools.combinations(range(len(g)), 2))
        if len(pairs) > MAX_PAIRS_PER_SEQ:
            pairs = rng.sample(pairs, MAX_PAIRS_PER_SEQ)
        ent = g.entry_id.values
        for a, b in pairs:
            if ent[a] == ent[b]:
                continue
            m = ~np.isnan(G[a]) & ~np.isnan(G[b]) & ~np.isnan(Z[a]) & ~np.isnan(Z[b])
            if m.sum() == 0:
                continue
            hi = np.where(K[a] >= K[b], a, b)
            lo = np.where(K[a] >= K[b], b, a)
            ix = np.where(m)[0]
            out.append(
                pd.DataFrame(
                    {
                        "dk": np.abs(K[a, ix] - K[b, ix]),
                        "dZ": Z[hi[ix], ix] - Z[lo[ix], ix],
                        "dG": G[hi[ix], ix] - G[lo[ix], ix],
                        "Zmean": 0.5 * (Z[a, ix] + Z[b, ix]),
                        "Gmean": 0.5 * (G[a, ix] + G[b, ix]),
                        "cons": cons[ix],
                    }
                )
            )
    R = pd.concat(out, ignore_index=True)
    R["dkbin"] = pd.cut(R.dk, BIN_EDGES, labels=BIN_LABELS)
    R["state"] = np.where(
        R.cons < ORDERED_MAX,
        "ordered",
        np.where(R.cons > DISORDERED_MIN, "disordered", "intermediate"),
    )
    return R


def summarise(R: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in ("ordered", "disordered"):
        s = R[R.state == state]
        for lab in BIN_LABELS:
            b = s[s.dkbin == lab]
            rows.append(
                {
                    "state": state,
                    "dk_bin": lab,
                    "n_pairs": len(b),
                    "dZ_median": b.dZ.median(),
                    "dZ_q1": b.dZ.quantile(0.25),
                    "dZ_q3": b.dZ.quantile(0.75),
                    "dG_median": b.dG.median(),
                    "dG_q1": b.dG.quantile(0.25),
                    "dG_q3": b.dG.quantile(0.75),
                    "dG_mean": b.dG.mean(),
                }
            )
    return pd.DataFrame(rows)


# Fig. 3 top row: a (fig2c_chezod.py) and b (this file) side by side, together the 160 mm
# text width (W_DOUBLE), placed at 1:1 (a at A_W, b at FIG_W of the 160 mm \textwidth).
A_W = 52.0  # width of Fig. 3a
ROW_GAP = 2.0  # between a and b
FIG_W, FIG_H = sp.W_DOUBLE - A_W - ROW_GAP, 52.0  # 106 x 52 mm
AX_TOP, AX_H = 6.0, 36.5  # shared by a and b, so the axes of the row align
XLABEL_Y_MM = 1.4  # x-label baseline above the bottom edge; same in fig2c_chezod.py
YLABEL_X_MM = (
    2.2  # left edge of the shared, two-line y title (clear of the tick labels)
)
LEFT, RIGHT = 12.0, 1.5  # Z axes start; right margin after the G axes
AX_GAP = 9.0  # between the Z and G axes: holds the G-score tick labels
AX_W = (FIG_W - LEFT - RIGHT - AX_GAP) / 2  # 41.75 mm each
# axes boxes in mm: (left, top, width, height)
BOX = {
    "Z": (LEFT, AX_TOP, AX_W, AX_H),
    "G": (LEFT + AX_W + AX_GAP, AX_TOP, AX_W, AX_H),
}
STYLES = {
    # ordered: the series that carries the message, filled; disordered: the control, open
    # squares in the same ink (style_paper.CONTRAST; grey is CheZOD1325)
    "ordered": {**sp.CONTRAST["focus"], "dx": -0.11},
    "disordered": {**sp.CONTRAST["control"], "dx": 0.11},
}
# Both y axes span the same fraction of each score's gap. LO_FRAC leaves room under the
# disordered whiskers (lowest: dk 1-3 on the Z side, Q1 = -1.89 = -0.19 of the Z gap) for
# the direct label "Disordered residues".
# LO_FRAC -0.41 (was -0.36, 12 Nov 2026 review item 15) brings a negative tick into both
# axes (Z gap 9.96 -> -4.08; G gap 0.778 -> -0.32), so the lower limit carries a tick and
# the shared span is visible.
LO_FRAC, HI_FRAC = -0.41, 0.82
YTICKS = {"Z": [-4, 0, 4, 8], "G": [-0.3, 0, 0.3, 0.6]}
MS = 3.6  # marker size (pt)


def _label_series(ax, S: pd.DataFrame, x: np.ndarray, gap: float) -> None:
    """Direct labels in the Z-score axes: one name beside each series, no legend.

    "Ordered residues" sits left of the last ordered point (dk >= 10), above the rising
    line; "Disordered residues" sits under the flat open-square series, right-aligned to
    the last bin, below the lowest whisker it spans.
    """
    o = S[S.state == "ordered"].set_index("dk_bin").loc[BIN_LABELS]
    d = S[S.state == "disordered"].set_index("dk_bin").loc[BIN_LABELS]
    x_end = x[-1] + STYLES["ordered"]["dx"]
    sp.direct_label(
        ax, x_end, o.dZ_median.iloc[-1], "Ordered residues", dx_mm=-2.4, ha="right"
    )
    # under the disordered whiskers of the last three bins
    y_low = float(d.dZ_q1.iloc[2:].min())
    sp.direct_label(
        ax,
        x[-1] + STYLES["disordered"]["dx"],
        y_low,
        "Disordered residues",
        dx_mm=1.2,
        dy_mm=-1.0,
        ha="right",
        va="top",
    )


def plot(S: pd.DataFrame, gaps: dict, out_pdf: Path) -> None:
    fig = sp.figure(FIG_W, FIG_H)
    x = np.arange(len(BIN_LABELS))
    titles = {"Z": "Z-score", "G": "G-score"}
    for unit in ("Z", "G"):
        left, top, w, h = BOX[unit]
        ax = fig.add_axes([left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])
        score, gap = f"d{unit}", gaps[unit]
        ax.axhline(0, color=sp.CAT["light"], lw=0.5, zorder=0)
        for state, st in STYLES.items():
            s = S[S.state == state].set_index("dk_bin").loc[BIN_LABELS]
            ax.errorbar(
                x + st["dx"],
                s[f"{score}_median"],
                yerr=[
                    s[f"{score}_median"] - s[f"{score}_q1"],
                    s[f"{score}_q3"] - s[f"{score}_median"],
                ],
                color=st["color"],
                marker=st["marker"],
                mfc=st["mfc"],
                mec=st["color"],
                mew=0.8,
                ms=MS,
                lw=1.0,
                elinewidth=0.7,
                capsize=1.6,
                capthick=0.7,
                zorder=3 if state == "ordered" else 2,
            )
        ax.set_ylim(LO_FRAC * gap, HI_FRAC * gap)
        ax.set_xlim(-0.5, len(BIN_LABELS) - 0.5)
        ax.set_yticks(
            YTICKS[unit], [f"{t:g}".replace("-", "\u2212") for t in YTICKS[unit]]
        )
        ax.set_xticks(x, BIN_LABELS)
        ax.set_title(titles[unit], loc="left", fontsize=sp.FS_BODY, pad=3)
        if unit == "Z":
            _label_series(ax, S, x, gap)
    # one y title for both axes, at a fixed distance from the left edge (two lines, so it
    # stays shorter than the axes); one x title, centred under the pair
    fig.text(
        YLABEL_X_MM / FIG_W,
        1 - (AX_TOP + AX_H / 2) / FIG_H,
        "Score difference\n(more \u2212 fewer shifts)",
        rotation=90,
        ha="left",
        va="center",
        ma="center",
        linespacing=1.15,
        fontsize=sp.FS_BODY,
    )
    xc = (BOX["Z"][0] + BOX["G"][0] + BOX["G"][2]) / 2 / FIG_W
    fig.text(
        xc,
        XLABEL_Y_MM / FIG_H,
        r"Difference in shift count $\Delta\mathit{k}$",
        ha="center",
        va="bottom",
        fontsize=sp.FS_BODY,
    )
    sp.fig_label(fig, "b", 0.0, 0.0)
    sp.save(fig, out_pdf.stem)


def main() -> None:
    ap = sp.cli(__doc__)
    ap.add_argument("--parquet", type=Path, default=PARQUET)
    ap.add_argument(
        "--replot",
        action="store_true",
        help="redraw from source_data/fig3b_kpaired.csv without recomputing the pairs",
    )
    args = sp.parse_args(ap)
    if args.replot:
        S = pd.read_csv(
            args.out_dir / "source_data" / f"{OUT_NAME}.csv", dtype={"dk_bin": str}
        )
        plot(
            S,
            {"Z": S.gap_Z.iloc[0], "G": S.gap_G.iloc[0]},
            args.out_dir / f"{OUT_NAME}.pdf",
        )
        return

    R = residue_pairs(args.parquet)
    S = summarise(R)
    o = R[R.state == "ordered"]
    d = R[R.state == "disordered"]
    gaps = {
        "Z": float(o.Zmean.median() - d.Zmean.median()),
        "G": float(d.Gmean.median() - o.Gmean.median()),
    }
    # OLS slope of dZ / dG on dk over ordered residues with dk > 0 (context only)
    oo = o[o.dk > 0]
    slope_z = float(np.polyfit(oo.dk, oo.dZ, 1)[0])
    slope_g = float(np.polyfit(oo.dk, oo.dG, 1)[0])

    print(f"residue pairs: {len(R):,}")
    print(
        f"ordered: {len(o):,}  disordered: {len(d):,}  intermediate: {(R.state == 'intermediate').sum():,}"
    )
    print(f"gaps (ordered vs disordered medians): Z {gaps['Z']:.2f}  G {gaps['G']:.3f}")
    print(
        f"OLS slope (ordered, dk>0): dZ {slope_z:+.3f}/shift, dG {slope_g:+.4f}/shift"
    )
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(S.round(3).to_string(index=False))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "source_data").mkdir(exist_ok=True)
    S.assign(gap_Z=gaps["Z"], gap_G=gaps["G"]).to_csv(
        args.out_dir / "source_data" / f"{OUT_NAME}.csv", index=False
    )
    plot(S, gaps, args.out_dir / f"{OUT_NAME}.pdf")
    print(f"wrote {args.out_dir / (OUT_NAME + '.pdf')}")


if __name__ == "__main__":
    main()
