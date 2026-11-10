#!/usr/bin/env python3
"""Supplementary Fig. S2 -- G-score theory (synthetic Z and G vs k, the G-Z map, empirical G-Z).

Panels
  a  Synthetic Z-score vs shift count k for windows with 0/3 ... 3/3 ordered residues
     (mean and 5-95 % band over simulated residues).
  b  Same for the G-score: its mean is nearly flat in k; only its spread narrows.
  c  The exact G-Z relation: for fixed k both scores are monotone functions of the
     capped residual sum of squares, so each k traces one curve.
  d  Empirical G vs Z over the scored residues of the unfiltered training pool (hexbin),
     with the k = 3 / 9 / 15 / 21 curves of panel c overlaid.

The Z- and G-scores are computed with the scoring code itself
(``trizod.scoring.scoring.compute_zscores`` / ``compute_gscores``), so the curves
match the implementation, not a re-derivation:
    capped deviation  d = min(|Delta_n|, 4),  RSS = sum over the window of d^2
    Z = Wilson-Hilferty normal approximation of the chi^2_k CDF at RSS
    G = (exp(-RSS / (8 k)) - e^-2) / (1 - e^-2)

Synthetic model. A window holds three residues, each contributing k/3 shifts
(k = 3, 6, ..., 21: one to seven backbone atom types measured on all three). Shifts of disordered residues deviate from the random-coil
reference as N(0, 1) (in units of the per-atom random-coil SD sigma(n)); shifts of
ordered residues as N(0, s_o^2). s_o is calibrated so that fully ordered windows at
k = 21 reproduce the median Z of ordered pool residues with k 19-21 (numbers macro
\\zOrderedHighK), which anchors the illustration to the data.

Run: uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s2_gscore_theory.py \
         --out-dir <draft>/Figures/paper
Outputs: <out-dir>/S2.pdf and <out-dir>/suppfigs_b_data/S2_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402

sys.path.insert(0, str(sp.REPO))
from trizod.scoring.scoring import (  # noqa: E402
    chi2_cdf_approx,
    compute_gscores,
    compute_zscores,
)

RNG = np.random.default_rng(20261030)
N_SIM = 20000  # simulated residues per (scenario, k)
KS = np.arange(3, 22, 3)  # k = 3 x (backbone atom types measured on all three residues)
KS_MAP = np.arange(1, 22)  # every k for the analytic G-Z curves
N_ORD = [0, 1, 2, 3]  # ordered residues in the 3-residue window
ORD_LABEL = {0: "0/3 ordered", 1: "1/3 ordered", 2: "2/3 ordered", 3: "3/3 ordered"}
ORD_COLOR = {
    0: sp.OKABE_ITO["skyblue"],
    1: sp.OKABE_ITO["green"],
    2: sp.OKABE_ITO["orange"],
    3: sp.OKABE_ITO["vermillion"],
}
SUMMARY = {}


def scores_from_diffs(absdiffs: np.ndarray, k: int):
    """Z and G for rows of |Delta| with k filled columns, through the scoring code."""
    n = absdiffs.shape[0]
    full = np.zeros((n, 21))
    full[:, :k] = absdiffs
    dof = np.full(n, k)
    mask = np.ones((n, 7), dtype=bool)
    z = compute_zscores(full, dof, mask)
    g = compute_gscores(full, dof, mask)
    return z, g


def simulate(n_ord: int, k: int, s_o: float, n: int = N_SIM):
    owner = np.arange(k) % 3  # shift j belongs to residue j mod 3
    scale = np.where(owner < n_ord, s_o, 1.0)
    d = np.abs(RNG.normal(size=(n, k)) * scale[None, :])
    return scores_from_diffs(d, k)


def calibrate_sigma(target_z: float, k: int = 21) -> float:
    lo, hi = 1.0, 30.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        z, _ = simulate(3, k, mid, n=20000)
        if np.median(z) < target_z:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def analytic_curve(k: int, n: int = 400):
    rss = np.linspace(0.0, 16.0 * k, n)
    z = chi2_cdf_approx(rss, float(k))
    g = (np.exp(-rss / (8.0 * k)) - np.exp(-2.0)) / (1.0 - np.exp(-2.0))
    return z, g


def empirical_pool():
    df = sp.load_main(["id", "pool_tier", "gscores", "zscores", "k", "mask"])
    pool = df[df.pool_tier.notna()]
    G = np.concatenate([sp.arr(g) for g in pool.gscores])
    Z = np.concatenate([sp.arr(z) for z in pool.zscores])
    K = np.concatenate([sp.arr(k) for k in pool.k])
    M = np.concatenate([np.asarray(m, bool) for m in pool["mask"]])
    ok = M & np.isfinite(G) & np.isfinite(Z)
    # all released scored residues, for the record
    Ga = np.concatenate([sp.arr(g) for g in df.gscores])
    Za = np.concatenate([sp.arr(z) for z in df.zscores])
    Ma = np.concatenate([np.asarray(m, bool) for m in df["mask"]])
    oka = Ma & np.isfinite(Ga) & np.isfinite(Za)
    return G[ok], Z[ok], K[ok], len(pool), (Ga[oka], Za[oka])


def main():
    sp.parse_args(sp.cli(__doc__))
    G, Z, K, n_pool_chains, (Ga, Za) = empirical_pool()
    ordered_hi = (G < 0.2) & (K >= 19)
    z_target = float(np.median(Z[ordered_hi]))
    s_o = calibrate_sigma(z_target)
    r_pool = float(np.corrcoef(G, Z)[0, 1])
    r_all = float(np.corrcoef(Ga, Za)[0, 1])
    SUMMARY.update(
        {
            "pool_chains": n_pool_chains,
            "pool_residues": int(len(G)),
            "pearson_G_Z_pool": round(r_pool, 4),
            "released_residues": int(len(Ga)),
            "pearson_G_Z_released": round(r_all, 4),
            "median_Z_ordered_k19_21_pool": round(z_target, 3),
            "sigma_ordered_calibrated": round(s_o, 3),
        }
    )

    # ---- simulation
    sim = {}
    for n_ord in N_ORD:
        rows = []
        for k in KS:
            z, g = simulate(n_ord, int(k), s_o)
            rows.append(
                (
                    k,
                    np.mean(z),
                    *np.percentile(z, [5, 95]),
                    np.mean(g),
                    *np.percentile(g, [5, 95]),
                )
            )
        sim[n_ord] = np.array(rows)
    for n_ord in N_ORD:
        a = sim[n_ord]
        SUMMARY[f"sim_{n_ord}of3"] = {
            "meanZ_k3": round(a[0, 1], 2),
            "meanZ_k21": round(a[-1, 1], 2),
            "meanG_k3": round(a[0, 4], 3),
            "meanG_k21": round(a[-1, 4], 3),
            "meanG_range_k3_21": round(float(a[:, 4].max() - a[:, 4].min()), 3),
            "G_band_width_k3": round(a[0, 6] - a[0, 5], 3),
            "G_band_width_k21": round(a[-1, 6] - a[-1, 5], 3),
        }
    zc, gc = analytic_curve(21)
    SUMMARY["Z_ceiling_k21"] = round(float(zc[-1]), 3)
    SUMMARY["Z_at_RSS0_k21"] = round(float(zc[0]), 3)
    SUMMARY["G_disordered_expected_limit"] = round(
        (np.exp(-1 / 8) - np.exp(-2)) / (1 - np.exp(-2)), 3
    )

    # ---- figure
    fig = sp.figure(sp.W_DOUBLE, 56)
    gs = fig.add_gridspec(
        1, 4, left=0.05, right=0.985, bottom=0.17, top=0.90, wspace=0.42
    )
    axa, axb, axc, axd = (fig.add_subplot(gs[0, i]) for i in range(4))

    for n_ord in N_ORD:
        a = sim[n_ord]
        c = ORD_COLOR[n_ord]
        axa.fill_between(a[:, 0], a[:, 2], a[:, 3], color=c, alpha=0.18, lw=0)
        axa.plot(
            a[:, 0], a[:, 1], color=c, lw=0.9, marker="o", ms=2, label=ORD_LABEL[n_ord]
        )
        axb.fill_between(a[:, 0], a[:, 5], a[:, 6], color=c, alpha=0.18, lw=0)
        axb.plot(a[:, 0], a[:, 4], color=c, lw=0.9, marker="o", ms=2)
    axa.set_xlabel("Shifts in the window, $k$")
    axa.set_ylabel("Z-score (synthetic)")
    axa.set_xlim(2, 22)
    axa.set_xticks(KS)
    axa.axhline(8, color=sp.CAT["light"], lw=0.5, ls=":", zorder=0)
    axa.legend(loc="upper left", handlelength=1.2, borderaxespad=0.2)
    axb.set_xlabel("Shifts in the window, $k$")
    axb.set_ylabel("G-score (synthetic)")
    axb.set_xlim(2, 22)
    axb.set_ylim(0, 1)
    axb.set_xticks(KS)
    axb.axhline(sp.G_DIS, color=sp.CAT["light"], lw=0.5, ls=":", zorder=0)
    axb.text(
        22,
        sp.G_DIS + 0.015,
        "G = 0.65",
        ha="right",
        va="bottom",
        fontsize=5,
        color="#7a7a7a",
    )

    # c: analytic map
    import matplotlib as mpl

    cmap = mpl.colors.ListedColormap(
        mpl.colormaps["viridis"](np.linspace(0.0, 0.82, 256))
    )
    norm = mpl.colors.Normalize(vmin=1, vmax=21)
    for k in KS_MAP:
        z, g = analytic_curve(int(k))
        axc.plot(z, g, color=cmap(norm(k)), lw=0.7)
    axc.set_xlabel("Z-score")
    axc.set_ylabel("G-score")
    axc.set_ylim(-0.02, 1.02)
    axc.set_xlim(-5, 17)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = axc.inset_axes([0.62, 0.62, 0.05, 0.33])
    cb = fig.colorbar(sm, cax=cax, ticks=[1, 11, 21])
    cb.set_label("$k$", fontsize=5.5, labelpad=1)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)

    # d: empirical hexbin
    hb = axd.hexbin(
        Z,
        G,
        gridsize=70,
        extent=(-5, 17, 0, 1),
        bins="log",
        mincnt=1,
        cmap="Greys",
        linewidths=0.0,
        rasterized=False,
    )
    for k, ls in ((3, ":"), (9, "--"), (15, "-."), (21, "-")):
        z, g = analytic_curve(k)
        axd.plot(z, g, color=cmap(norm(k)), lw=0.7, ls=ls, label=f"$k$ = {k}")
    axd.set_xlabel("Z-score")
    axd.set_ylabel("G-score")
    axd.set_xlim(-5, 17)
    axd.set_ylim(-0.02, 1.02)
    axd.legend(loc="lower left", handlelength=1.6, borderaxespad=0.2)
    axd.text(
        0.98,
        0.98,
        f"Pearson r = {r_pool:.3f}\n{len(G):,} residues",
        transform=axd.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
    )
    cax2 = axd.inset_axes([0.62, 0.45, 0.05, 0.28])
    cb2 = fig.colorbar(hb, cax=cax2)
    cb2.set_label("residues", fontsize=5, labelpad=1)
    cb2.ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    cb2.outline.set_linewidth(0.4)

    for ax, l in zip((axa, axb, axc, axd), "abcd"):
        sp.panel_label(ax, l, x=-0.27, y=1.03)
    out = sp.save(fig, "S2")
    data_dir = sp.out_dir("suppfigs_b_data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "S2_summary.json").write_text(json.dumps(SUMMARY, indent=2))
    print(out)
    print(json.dumps(SUMMARY, indent=2))


if __name__ == "__main__":
    main()
