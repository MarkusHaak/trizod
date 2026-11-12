#!/usr/bin/env python3
"""Fig. 2b (Fig. 2a until 11 Nov 2026) -- same residue, different shift count.

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
share of the label's dynamic range. Ordered residues (the message) are drawn in
ink, disordered residues (the control) in grey; the per-bin pair counts are in
the Source Data, not in the artwork.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig2a_kpaired.py \
        --out-dir <draft>/Figures/paper
Outputs: <out-dir>/fig2b_kpaired.pdf (100 x 52 mm; with Fig. 2c, 58 mm, and a 2 mm gap
it fills the 160 mm text width at 1:1) and <out-dir>/source_data/fig2b_kpaired.csv;
prints the plotted numbers. ``fig2c_chezod.py`` imports the row layout from here.
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
OUT_NAME = "fig2b_kpaired"  # Fig. 2b (was Fig. 2a, fig2a.pdf, until 11 Nov 2026)


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


# Fig. 2 bottom row: b (this file) and c (fig2c_chezod.py) side by side, together the
# 160 mm text width (W_DOUBLE), placed at 1:1 (b at FIG_W / W_DOUBLE of \textwidth).
C_W = 58.0  # width of Fig. 2c
ROW_GAP = 2.0  # between b and c
FIG_W, FIG_H = sp.W_DOUBLE - C_W - ROW_GAP, 52.0  # 100 x 52 mm
AX_TOP, AX_H = 6.0, 36.5  # shared by b and c, so the axes of the row align
AX_W, AX_GAP = 37.0, 11.0  # each score's axes; gap holds the G-score tick labels
# axes boxes in mm: (left, top, width, height)
BOX = {
    "Z": (FIG_W - 1.0 - 2 * AX_W - AX_GAP, AX_TOP, AX_W, AX_H),
    "G": (FIG_W - 1.0 - AX_W, AX_TOP, AX_W, AX_H),
}
STYLES = {
    # ordered: the series that carries the message, in full ink; disordered: the control
    "ordered": {"color": "#1A1A1A", "marker": "o", "mfc": "#1A1A1A", "dx": -0.11},
    "disordered": {"color": "#8C8C8C", "marker": "s", "mfc": "white", "dx": 0.11},
}
# Both y axes span the same fraction of each score's gap. LO_FRAC leaves room for the
# lowest whisker (disordered, dk >= 10, G side: Q1 = -0.165 = -0.215 of the G gap).
LO_FRAC, HI_FRAC = -0.27, 0.82
YTICKS = {"Z": [0, 4, 8], "G": [0, 0.3, 0.6]}
XLABEL_Y_MM = 1.4  # x-label baseline above the bottom edge; same in fig2c_chezod.py


def plot(S: pd.DataFrame, gaps: dict, out_pdf: Path) -> None:
    from matplotlib.lines import Line2D

    fig = sp.figure(FIG_W, FIG_H)
    x = np.arange(len(BIN_LABELS))
    titles = {"Z": "Z-score", "G": "G-score"}
    for unit in ("Z", "G"):
        left, top, w, h = BOX[unit]
        ax = fig.add_axes([left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])
        score, gap = f"d{unit}", gaps[unit]
        ax.axhline(0, color="0.75", lw=0.5, zorder=0)
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
                ms=3.6,
                lw=1.0,
                elinewidth=0.7,
                capsize=1.6,
                capthick=0.7,
                zorder=3 if state == "ordered" else 2,
            )
        ax.set_ylim(LO_FRAC * gap, HI_FRAC * gap)
        ax.set_xlim(-0.5, len(BIN_LABELS) - 0.5)
        ax.set_yticks(YTICKS[unit], [f"{t:g}" for t in YTICKS[unit]])
        ax.set_xticks(x, BIN_LABELS)
        ax.set_title(titles[unit], loc="left", fontsize=sp.FS_BODY, pad=3)
        if unit == "Z":
            ax.set_ylabel(
                "Score difference\n(more \u2212 fewer shifts)", linespacing=1.1
            )
            handles = [
                Line2D(
                    [],
                    [],
                    color=st["color"],
                    marker=st["marker"],
                    mfc=st["mfc"],
                    mec=st["color"],
                    mew=0.8,
                    ms=3.6,
                    lw=1.0,
                    label=f"{state.capitalize()} residues",
                )
                for state, st in STYLES.items()
            ]
            ax.legend(
                handles=handles,
                loc="upper left",
                frameon=False,
                handlelength=1.8,
                borderaxespad=0.2,
                fontsize=sp.FS_TICK,
            )
    # one x label for both axes, centred under the pair
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
        help="redraw from source_data/fig2b_kpaired.csv without recomputing the pairs",
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
