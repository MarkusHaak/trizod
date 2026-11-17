#!/usr/bin/env python3
"""Supplementary Fig. S2 -- G-score theory (synthetic Z and G vs k, the G-Z map, empirical G-Z).

Layout (17 Nov 2026, figure review): a | b above c | d, so the line labels of a and b can
be written out and the residue bands of d are resolved.

Panels
  a  Synthetic Z-score vs shift count k for windows with 0, 1, 2 or 3 ordered residues
     out of three (mean and 5-95 % band over simulated residues); each line is labelled
     at its end ("2 of 3 ordered").
  b  Same for the G-score: its mean is nearly flat in k; only its spread narrows.
  c  The exact G-Z relation: for fixed k both scores are monotone functions of the
     capped residual sum of squares, so each k traces one curve.
  d  Empirical G vs Z over the scored residues of the unfiltered training pool (fine 2-D
     histogram, log counts). No curves are drawn on top: every residue lies on the curve
     of its k (largest |G - G_curve(Z, k)| below 1e-4, the storage precision of G;
     summary key max_abs_dG_from_curve_pool), so the residues form one band per k
     themselves; the overlaid k = 3 / 9 / 15 / 21 curves of the earlier version hid them.
     The curves k = 1 and k = 21 are labelled at the same place in c and d.

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
k = 21 reproduce the median Z of ordered pool residues (G < ``style_paper.G_ORD``; G < 0.2
until 17 Nov 2026) with k 19-21 (numbers macro \\zOrderedHighK, same definition), which
anchors the illustration to the data.

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
# Ordered share is ordinal, so it gets a grey ramp (light = disordered, black = fully
# ordered); the Okabe-Ito hues keep their fixed meanings (CheZOD, training, test, removed).
ORD_COLOR = {
    0: "#A8A8A8",
    1: "#787878",
    2: "#454545",
    3: "#000000",
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


def curve_deviation(G, Z, K) -> float:
    """Largest |G - G_curve(Z, k)| over residues: how far each residue lies from the
    analytic G-Z curve of its k. Z is inverted for the capped RSS by bisection (Z rises
    monotonically with the RSS) and G computed from it. Measured in G, not Z: near G = 1
    Z is steep in the RSS, so the 4-decimal storage of G alone moves Z by up to ~1."""
    lo, hi = np.zeros_like(Z, dtype=float), 16.0 * K
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        up = chi2_cdf_approx(mid, K) < Z
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    rss = 0.5 * (lo + hi)
    g = (np.exp(-rss / (8.0 * K)) - np.exp(-2.0)) / (1.0 - np.exp(-2.0))
    return float(np.max(np.abs(g - G)))


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
    ordered_hi = (G < sp.G_ORD) & (K >= 19)
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
            # panel d draws no curves: each residue's Z equals the curve of its k at its G
            "max_abs_dG_from_curve_pool": float(f"{curve_deviation(G, Z, K):.1g}"),
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

    # ---- figure: a | b (simulation over k) above c | d (the G-Z relation)
    W, H = sp.W_DOUBLE, 100.0  # mm

    def ax_mm(x, top, w, h):
        """Axes at ``x`` mm from the left, ``top`` mm from the top, ``w`` x ``h`` mm."""
        return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H])

    fig = sp.figure(W, H)
    axa = ax_mm(12.0, 5.0, 45.0, 37.0)
    axb = ax_mm(91.0, 5.0, 45.0, 37.0)
    axc = ax_mm(12.0, 55.0, 62.0, 36.0)
    axd = ax_mm(91.0, 55.0, 62.0, 36.0)

    # a and b: each line is labelled at its right end with the number of ordered residues
    # in the three-residue window, written out ("2 of 3 ordered"), so neither a legend nor
    # the caption is needed to read it; the dotted lines are the disorder cut-offs.
    for n_ord in N_ORD:
        a = sim[n_ord]
        c = ORD_COLOR[n_ord]
        lab = f"{n_ord} of 3 ordered"
        axa.fill_between(a[:, 0], a[:, 2], a[:, 3], color=c, alpha=0.12, lw=0)
        axa.plot(a[:, 0], a[:, 1], color=c, lw=0.9, marker="o", ms=2)
        axb.fill_between(a[:, 0], a[:, 5], a[:, 6], color=c, alpha=0.12, lw=0)
        axb.plot(a[:, 0], a[:, 4], color=c, lw=0.9, marker="o", ms=2)
        for ax, y in ((axa, a[-1, 1]), (axb, a[-1, 4])):
            sp.direct_label(ax, KS[-1], y, lab, dx_mm=1.6)
    for ax in (axa, axb):
        ax.set_xlabel("Shifts in the window, $k$")
        ax.set_xlim(2, KS[-1] + 1)
        ax.set_xticks(KS)
        ax.spines["bottom"].set_bounds(KS[0], KS[-1])
    axa.set_ylabel("Z-score (synthetic)")
    axa.set_ylim(-3, 17)
    axa.set_yticks([0, 5, 10, 15])
    # disorder cut-offs (Z = 8; the binary cut G = G_DIS), drawn over the data range only
    axa.hlines(8, 2, KS[-1], color=sp.INK_SOFT, lw=0.5, ls=":", zorder=0)
    axb.set_ylabel("G-score (synthetic)")
    axb.set_ylim(0, 1)
    axb.hlines(sp.G_DIS, 2, KS[-1], color=sp.INK_SOFT, lw=0.5, ls=":", zorder=0)
    sp.direct_label(axa, 2.2, 8, "Z = 8", dx_mm=0, dy_mm=0.4, va="bottom")
    sp.direct_label(
        axb, 2.2, sp.G_DIS, f"G = {sp.G_DIS:.2f}", dx_mm=0, dy_mm=0.4, va="bottom"
    )

    # c: analytic map
    import matplotlib as mpl

    # the paper's k scale (sp.K_CMAP), without its palest top so the lines stay visible
    cmap = mpl.colors.ListedColormap(
        mpl.colormaps[sp.K_CMAP](np.linspace(0.0, 0.85, 256))
    )
    norm = mpl.colors.Normalize(vmin=1, vmax=21)
    for k in KS_MAP:
        z, g = analytic_curve(int(k))
        axc.plot(z, g, color=cmap(norm(k)), lw=0.7)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = axc.inset_axes([0.70, 0.56, 0.035, 0.38])
    cb = fig.colorbar(sm, cax=cax, ticks=[1, 11, 21])
    cb.set_label("$k$", fontsize=sp.FS_TICK, labelpad=2)
    cb.ax.tick_params(labelsize=sp.FS_TICK, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)

    # d: the released residues as a fine 2-D histogram (log counts), WITHOUT the curves of
    # c on top: every residue lies on the curve of its k (max_abs_dG_from_curve_pool), so
    # the residues draw those curves themselves, one grey band per k. The overlay that d
    # carried until 17 Nov 2026 (k = 3 / 9 / 15 / 21, dashed, haloed) hid exactly those
    # bands (figure review, item 7b).
    zr, gr = (-5.0, 17.0), (-0.02, 1.02)
    h2, ze, ge = np.histogram2d(Z, G, bins=(440, 300), range=(zr, gr))
    grey = mpl.colors.ListedColormap(
        mpl.colormaps["Greys"](np.linspace(0.30, 1.0, 256))
    )
    mesh = axd.pcolormesh(
        ze,
        ge,
        np.ma.masked_equal(h2.T, 0),
        cmap=grey,
        norm=mpl.colors.LogNorm(vmin=1, vmax=h2.max()),
        rasterized=True,
    )
    cax2 = axd.inset_axes([0.70, 0.56, 0.035, 0.38])
    cb2 = fig.colorbar(
        mesh, cax=cax2, ticks=[t for t in (1, 10, 100, 1000, 10000) if t <= h2.max()]
    )
    cb2.set_label("Residues", fontsize=sp.FS_TICK, labelpad=2)
    cb2.ax.tick_params(labelsize=sp.FS_TICK, length=1.5, pad=1)
    cb2.outline.set_linewidth(0.4)
    axd.text(
        0.03,
        0.04,
        f"r = {r_pool:.2f}".replace("-", "\N{MINUS SIGN}"),
        transform=axd.transAxes,
        ha="left",
        va="bottom",
        fontsize=sp.FS_TICK,
    )

    # c and d share their axes, and the outermost curves (k = 1, k = 21) are labelled at
    # the same place in both, so a band of d is matched to its curve of c by position
    g_lab = 0.2
    for ax in (axc, axd):
        for k, dx, ha in ((1, -1.0, "right"), (int(KS_MAP[-1]), 1.0, "left")):
            z, g = analytic_curve(k)
            z0 = float(np.interp(g_lab, g[::-1], z[::-1]))  # g falls as z rises
            sp.direct_label(ax, z0, g_lab, f"$k$ = {k}", dx_mm=dx, ha=ha)
        ax.set_xlabel("Z-score")
        ax.set_ylabel("G-score")
        ax.set_xlim(*zr)
        ax.set_ylim(*gr)

    for ax, l in zip((axa, axb, axc, axd), "abcd"):
        sp.panel_label(ax, l, x=-11.0 / (ax.get_position().width * W), y=1.02)
    out = sp.save(fig, "S2")
    data_dir = sp.out_dir("suppfigs_b_data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "S2_summary.json").write_text(json.dumps(SUMMARY, indent=2))
    print(out)
    print(json.dumps(SUMMARY, indent=2))


if __name__ == "__main__":
    main()
