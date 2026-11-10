#!/usr/bin/env python3
"""Fig. 2a -- same residue, different shift count (within-residue k test).

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
share of the label's dynamic range.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig2a_kpaired.py \
        --out-dir <draft>/Figures/paper
Outputs: <out-dir>/fig2a.pdf and <out-dir>/source_data/fig2a.csv; prints the
plotted numbers.
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

PARQUET = sp.PQ_MAIN
FS_SMALL = 5.0

MAX_PAIRS_PER_SEQ = 30
SEED = 0
ORDERED_MAX = 0.3  # replicate-consensus G below this: ordered
DISORDERED_MIN = 0.7  # replicate-consensus G above this: disordered
BIN_EDGES = [-0.5, 0.5, 3, 6, 9, 21]
BIN_LABELS = ["0", "1–3", "4–6", "7–9", "≥10"]


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


def plot(S: pd.DataFrame, gaps: dict, out_pdf: Path) -> None:
    fig = sp.figure(sp.W_DOUBLE, 46.0, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, wspace=0.08)
    axes = fig.subplots(1, 2)
    x = np.arange(len(BIN_LABELS))
    styles = {
        "ordered": {"color": sp.OKABE_ITO["blue"], "marker": "o", "dx": -0.08},
        "disordered": {"color": sp.OKABE_ITO["vermillion"], "marker": "s", "dx": 0.08},
    }
    lo_frac, hi_frac = -0.22, 0.82  # same fraction of each score's gap
    titles = {"Z": "CheZOD Z-score", "G": "TriZOD G-score"}
    for ax, score, unit in ((axes[0], "dZ", "Z"), (axes[1], "dG", "G")):
        gap = gaps[unit]
        for state, st in styles.items():
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
                ms=3.0,
                lw=0.9,
                elinewidth=0.6,
                capsize=1.4,
                label=f"{state} consensus",
            )
            # pair counts per bin, in the series colour, along the top of G
            if unit != "G":
                continue
            ytxt = hi_frac * gap * (0.97 if state == "ordered" else 0.88)
            for i, n in enumerate(s.n_pairs):
                ax.text(
                    i,
                    ytxt,
                    f"{n / 1000:.1f}k",
                    ha="center",
                    va="top",
                    fontsize=FS_SMALL,
                    color=st["color"],
                )
        ax.set_ylim(lo_frac * gap, hi_frac * gap)
        ax.set_xlim(-0.5, len(BIN_LABELS) - 0.5)
        ax.axhline(0, color="0.6", lw=0.5, zorder=0)
        ax.set_xticks(x, BIN_LABELS)
        ax.set_xlabel("Shift-count difference \u0394k at the residue")
        ax.set_ylabel(f"\u0394{unit}, higher-k minus lower-k chain")
        ax.set_title(titles[unit], loc="left")
    axes[1].text(
        -0.45,
        hi_frac * gaps["G"] * 0.78,
        "residue pairs per bin",
        ha="left",
        va="top",
        fontsize=FS_SMALL,
        color="0.35",
    )
    axes[0].legend(loc="upper left", frameon=False, handlelength=1.8)
    sp.fig_label(fig, "a", 0.0, 0.0)
    sp.save(fig, out_pdf.stem)


def main() -> None:
    ap = sp.cli(__doc__)
    ap.add_argument("--parquet", type=Path, default=PARQUET)
    ap.add_argument(
        "--replot",
        action="store_true",
        help="redraw from source_data/fig2a.csv without recomputing the pairs",
    )
    args = sp.parse_args(ap)
    if args.replot:
        S = pd.read_csv(
            args.out_dir / "source_data" / "fig2a.csv", dtype={"dk_bin": str}
        )
        plot(
            S, {"Z": S.gap_Z.iloc[0], "G": S.gap_G.iloc[0]}, args.out_dir / "fig2a.pdf"
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
        args.out_dir / "source_data" / "fig2a.csv", index=False
    )
    plot(S, gaps, args.out_dir / "fig2a.pdf")
    print(f"wrote {args.out_dir / 'fig2a.pdf'}")


if __name__ == "__main__":
    main()
