"""Supplementary Fig. S20 -- three G-score cuts from one rule (ordered edge, binary cut, disordered edge).

Every panel has G (or the cut-off t on G) on its x axis, scaled 0-1 alike, and the same
background: four zones bounded by the published cuts G_ORD / G_DIS / G_CONF of paper_config
(0.15 / 0.35 / 0.70 on 17 Nov 2026), shaded on one grey ramp, darker = more disordered:
ordered G < G_ORD (no shade), ambiguous, disordered G >= G_DIS (the binary call), confidently
disordered G >= G_CONF. A thin line marks each cut, and the x ticks sit at the cuts.

Panels
  a  Posterior P(disordered | G) at the reference-consensus prevalence pi, for each reference
     -- A, DisProt disorder against residues observed in X-ray structures; B, residues missing
     from every covering X-ray structure against observed ones; C, same-entry {1H}-15N hetNOE
     < 0.5 -- and for their consensus (band: 95 % interval). The y ticks are the rule's levels
     0.10 / 0.50 / 0.90; open circles, where the consensus crosses them (the estimated cuts).
  b  Bootstrap distribution of the three estimated cuts (share of replicates per 0.01 of G);
     above each, the estimate with its 95 % interval. Each interval holds the published cut.
  c  Agreement of the binary call G >= t between same-condition replicate entries (Cohen's
     kappa) over t: the calibration set (strict-tier unbound chains) and all unbound chains.
  d  G of the scored residues (density, log scale) of the calibration set and of all release
     chains; percentages, share of calibration-set residues per zone.
Curves: primary estimator (log LR interpolated between isotonic block midpoints). Bands and
intervals: 95 %, from 1,000 bootstrap resamples -- in a and b of sequence clusters, jointly
across references; in c of sequences (the replicate groups).

Panels dropped on 17 Nov 2026 (figure review): the likelihood-ratio panel -- at the stated
prevalence the posterior is a monotone transform of the LR (odds = pi / (1 - pi) * LR), so it
repeated panel a on another scale; its curves stay in the source data (S20a_*.csv).

Input: the output directory of scripts/validation/threshold_calibration/framework/compute.py
  $TRIZOD_THRESHOLD_FRAMEWORK_DIR  default <data root>/data/interim/threshold-framework-port-2026-11-17
The framework's paper values (results.json settings.paper_values) must equal the cuts of
paper_config (G_ORD, G_DIS, G_CONF); the script stops otherwise, so the figure never shows
other cuts than the text.

Run: uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s20_threshold.py \
         --out-dir <draft>/Figures/paper
Outputs: <out-dir>/S20.pdf and <out-dir>/suppfigs_d_data/S20*.csv.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_config as pc  # noqa: E402
import style_paper as sp  # noqa: E402

FW = pc.env_path(
    "TRIZOD_THRESHOLD_FRAMEWORK_DIR",
    pc.CHECKOUT / "data" / "interim" / "threshold-framework-port-2026-11-17",
)

W, H = sp.W_DOUBLE, 92.0  # mm
# References: Okabe-Ito hues, validated as a set (dataviz validate_palette.js, all pairs:
# CVD dE >= 11.4, normal-vision dE >= 18.7). Orange is below 3:1 on white, so every
# reference also carries its letter as a direct label next to its curve.
REF = {
    "A": (sp.OKABE_ITO["blue"], "DisProt vs X-ray"),
    "B": (sp.OKABE_ITO["orange"], "missing in X-ray"),
    "C": (sp.OKABE_ITO["green"], "hetNOE < 0.5"),
}
CONS = sp.INK
CUT_LINE = sp.INK_SOFT
GRID = "#C8C8C8"  # hairline guides at the posterior levels
HIST = "#6B6B6B"  # bootstrap histograms
OTHER = "#8A8A8A"  # the comparison set of c and d (all unbound / all chains)
# Zones: one grey ramp, darker = more disordered (ordered = the white surface).
ZONES = (
    ("ordered", "#FFFFFF"),
    ("ambiguous", "#F2F2F2"),
    ("disordered", "#E6E6E6"),
    ("confidently\ndisordered", "#D9D9D9"),
)
LEVELS = ("0.10", "0.50", "0.90")
EDGE_KEYS = ("ordered_edge", "binary_cut", "disordered_edge")


def ax_mm(fig, x: float, top: float, w: float, h: float):
    """Axes at ``x`` mm from the left, ``top`` mm from the top, ``w`` x ``h`` mm."""
    return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H])


def zone_edges(cuts) -> list:
    return [0.0, *cuts, 1.0]


def shade(ax, cuts, line_top: float = 1.0) -> None:
    """The four zones as background bands and a thin line at each cut (up to
    ``line_top``, in axes fraction)."""
    e = zone_edges(cuts)
    for i, (_, col) in enumerate(ZONES):
        if col.upper() != "#FFFFFF":
            ax.axvspan(e[i], e[i + 1], color=col, lw=0, zorder=0)
    for v in cuts:
        ax.axvline(v, ymax=line_top, color=CUT_LINE, lw=0.5, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, *cuts, 1])
    ax.set_xticklabels(["0", *(f"{v:.2f}" for v in cuts), "1"])


def fmt2(v: float) -> str:
    return f"{v:.2f}"


def main() -> None:
    sp.parse_args(sp.cli(__doc__))
    R = json.loads((FW / "results.json").read_text())
    fine = pd.read_csv(FW / "curves_fine.csv")
    boot = pd.read_csv(FW / "boot_thresholds.csv")
    gh = pd.read_csv(FW / "g_histogram.csv")
    kap = pd.read_csv(FW / "replicate_kappa_curve.csv")
    zs = pd.read_csv(FW / "zone_shares.csv")
    C = R["calibration"]
    est = C["primary_estimator"]
    cons = C["estimators"][est]["consensus"]
    pi = cons["pi"][0]
    paper = [float(x) for x in R["settings"]["paper_values"]]
    cuts = (pc.G_ORD, pc.G_DIS, pc.G_CONF)
    if not np.allclose(paper, cuts):
        raise SystemExit(
            f"framework paper values {paper} != paper_config cuts {list(cuts)}: rerun "
            f"framework/compute.py with the paper_config cuts ({FW})"
        )
    x = fine.G

    fig = sp.figure(W, H)
    # left column: a (posterior) over b (bootstrap strip), sharing the G axis;
    # right column: c (replicate kappa) over d (G distribution)
    LX, LW = 13.0, 82.0
    RX, RW = 112.0, 46.5
    TOP = 9.5
    axa = ax_mm(fig, LX, TOP, LW, 50.0)
    axb = ax_mm(fig, LX, TOP + 52.5, LW, 21.0)
    axc = ax_mm(fig, RX, TOP, RW, 29.5)
    axd = ax_mm(fig, RX, TOP + 44.0, RW, 29.5)
    # b: the cut lines stop at the interval row, so they meet the intervals but not
    # the estimate printed above each
    B_YMAX, WHISK_Y = 21.0, 15.5  # % of replicates
    for ax in (axa, axb, axc, axd):
        shade(ax, cuts, line_top=(WHISK_Y + 0.5) / B_YMAX if ax is axb else 1.0)

    # zone names above a (the shading is the same in every panel)
    e = zone_edges(cuts)
    for i, (name, _) in enumerate(ZONES):
        axa.text(
            (e[i] + e[i + 1]) / 2,
            1.015,
            name,
            transform=axa.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=sp.FS_BODY,
            linespacing=1.0,
            clip_on=False,
        )

    # ---- a: posterior at the stated prevalence; the rule's levels are the y ticks
    for lev in LEVELS:
        axa.axhline(float(lev), color=GRID, lw=0.5, zorder=1)
    for r, (col, _) in REF.items():
        axa.plot(x, fine[f"post_{r}"], color=col, lw=0.9, zorder=3)
    axa.fill_between(
        x, fine.post_lo, fine.post_hi, color=CONS, alpha=0.12, lw=0, zorder=2
    )
    axa.plot(x, fine.post, color=CONS, lw=1.6, zorder=4)
    for lev in LEVELS:
        t = cons["cuts"][lev]["t"]
        axa.plot(t, float(lev), "o", ms=4.2, mfc="white", mec=CONS, mew=1.0, zorder=6)

    # letters on the curves where they part (B: its plateau; C: highest; A: lowest)
    def post_at(col: str, g: float) -> float:
        return float(np.interp(g, x, fine[col]))

    sp.direct_label(axa, 1.0, post_at("post_B", 0.995), "B", dx_mm=0.8)
    sp.direct_label(
        axa, 0.43, post_at("post_C", 0.43), "C", dx_mm=-1.0, dy_mm=1.4, ha="right"
    )
    sp.direct_label(
        axa, 0.47, post_at("post_A", 0.47), "A", dx_mm=0.8, dy_mm=-1.6, va="top"
    )
    # key, in the empty lower part of the disordered zone
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=CONS, lw=1.6, label="consensus of A–C")]
    handles += [
        Line2D([], [], color=col, lw=0.9, label=f"{r}  {lab}")
        for r, (col, lab) in REF.items()
    ]
    handles.append(
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=4.2,
            mfc="white",
            mec=CONS,
            mew=1.0,
            label="estimated cut",
        )
    )
    axa.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(cuts[1] + 0.012, 0.0),
        bbox_transform=axa.get_xaxis_transform(),
        handlelength=1.4,
        handletextpad=0.5,
        labelspacing=0.25,
        borderaxespad=0.3,
        borderpad=0.0,
        frameon=False,
    )
    sp.direct_label(
        axa,
        0.985,
        0.03,
        f"$\\pi$ = {pi:.2f}",
        dx_mm=0,
        ha="right",
        va="bottom",
    )
    axa.set_ylim(0, 1)
    axa.set_yticks([0, 0.1, 0.5, 0.9, 1.0])
    axa.set_yticklabels(["0", "0.10", "0.50", "0.90", "1"])
    axa.set_ylabel("P(disordered | G)")
    axa.tick_params(axis="x", labelbottom=False)

    # ---- b: bootstrap distribution of each estimated cut, with estimate and 95 % interval
    bins = np.arange(0, 1.0001, 0.01)
    for lev, key in zip(LEVELS, EDGE_KEYS):
        v = boot[f"{key}_{est}"].dropna()
        h, _ = np.histogram(v, bins=bins)
        axb.stairs(100 * h / len(boot), bins, fill=True, color=HIST, lw=0, zorder=3)
        c = cons["cuts"][lev]
        lo, hi = c["CI"]
        axb.plot([lo, hi], [WHISK_Y, WHISK_Y], color=CONS, lw=0.9, zorder=5)
        axb.plot(c["t"], WHISK_Y, "o", ms=4.2, mfc="white", mec=CONS, mew=1.0, zorder=6)
        sp.direct_label(
            axb,
            c["t"],
            WHISK_Y,
            fmt2(c["t"]),
            dx_mm=0,
            dy_mm=1.3,
            va="bottom",
            ha="center",
        )
    axb.set_ylim(0, B_YMAX)
    axb.set_yticks([0, 10])
    axb.set_ylabel("Bootstrap\nreplicates (%)")
    axb.set_xlabel("G-score")

    # ---- c: replicate agreement of the binary call
    for setname, col, lw, alpha in (
        ("all_unbound_chains", OTHER, 0.9, 0.16),
        ("analysis_set", CONS, 1.3, 0.16),
    ):
        k = kap[kap.set == setname]
        axc.fill_between(k.t, k.lo, k.hi, color=col, alpha=alpha, lw=0, zorder=2)
        axc.plot(k.t, k.kappa, color=col, lw=lw, zorder=3)
    kb = R["replicates"]["analysis_set"]["paper_values"]["kappa_binary"][0]
    axc.plot(cuts[1], kb, "o", ms=3.6, color=CONS, mec="white", mew=0.6, zorder=6)
    sp.direct_label(
        axc,
        cuts[1],
        kb,
        f"$\\kappa$ = {kb:.2f}",
        dx_mm=0.9,
        dy_mm=1.2,
        ha="left",
        va="bottom",
    )
    # direct labels, clear of the cut lines: the calibration set above its band right of
    # the disordered edge, all unbound below its band
    k_cal = kap[kap.set == "analysis_set"]
    k_all = kap[kap.set == "all_unbound_chains"]
    right = (k_cal.t > cuts[2]) & (k_cal.t <= 0.985)
    sp.direct_label(
        axc,
        0.985,
        float(k_cal.hi[right].max()),
        "calibration\nset",
        dx_mm=0,
        dy_mm=0.5,
        ha="right",
        va="bottom",
        linespacing=1.0,
    )
    t_lo = 0.53
    sp.direct_label(
        axc,
        t_lo,
        float(np.interp(t_lo, k_all.t, k_all.lo)),
        "all unbound",
        dx_mm=0,
        dy_mm=-1.2,
        ha="center",
        va="top",
    )
    axc.set_ylim(0.4, 1.04)
    axc.set_yticks([0.4, 0.6, 0.8, 1.0])
    axc.set_ylabel("Replicate agreement ($\\kappa$)")
    axc.set_xlabel("Cut-off $t$ (disordered if $G \\geq t$)")

    # ---- d: G distribution of the scored residues, share per zone
    width = 0.01
    for col_name, col, lw in (
        ("tier_unfiltered+", OTHER, 0.8),
        ("analysis_set", CONS, 1.0),
    ):
        dens = gh[col_name] / (gh[col_name].sum() * width)
        axd.step(
            np.r_[gh.G_lo, 1.0],
            np.r_[dens, dens.iloc[-1]],
            where="post",
            color=col,
            lw=lw,
            zorder=3,
        )
    z = zs[(zs.set == "analysis_set") & (zs.cuts_from == "paper_values")].iloc[0]
    shares = (
        z.ordered,
        1 - z.ordered - z.binary_disordered,
        z.binary_disordered - z.disordered,
        z.disordered,
    )
    for i, s in enumerate(shares):
        axd.text(
            (e[i] + e[i + 1]) / 2,
            0.04,
            f"{100 * s:.0f}%",
            transform=axd.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=sp.FS_BODY,
        )
    # direct labels: all chains above its high-G bump, the calibration set below its trough
    dens_all = gh["tier_unfiltered+"] / (gh["tier_unfiltered+"].sum() * width)
    dens_cal = gh["analysis_set"] / (gh["analysis_set"].sum() * width)
    sp.direct_label(
        axd,
        0.985,
        float(dens_all[gh.G_lo > 0.8].max()),
        "all chains",
        dx_mm=0,
        dy_mm=0.8,
        ha="right",
        va="bottom",
    )
    sp.direct_label(
        axd,
        0.525,
        float(dens_cal[(gh.G_lo > 0.35) & (gh.G_lo < 0.7)].min()),
        "calibration set",
        dx_mm=0,
        dy_mm=-0.8,
        ha="center",
        va="top",
    )
    axd.set_yscale("log")
    axd.set_ylim(0.005, 30)
    axd.set_ylabel("Residues (density)")
    axd.set_xlabel("G-score")

    sp.fig_label(fig, "a", 0.5, 1.0)
    sp.fig_label(fig, "b", 0.5, TOP + 52.5 - 1.0)
    sp.fig_label(fig, "c", RX - 13.0, 1.0)
    sp.fig_label(fig, "d", RX - 13.0, TOP + 44.0 - 3.0)

    out = sp.save(fig, "S20")
    data_dir = sp.out_dir("suppfigs_d_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    for src, name in (
        ("curves_fine.csv", "S20a_posterior_lr_curves.csv"),
        ("boot_thresholds.csv", "S20b_boot_thresholds.csv"),
        ("replicate_kappa_curve.csv", "S20c_replicate_kappa.csv"),
        ("g_histogram.csv", "S20d_g_histogram.csv"),
        ("zone_shares.csv", "S20d_zone_shares.csv"),
    ):
        shutil.copyfile(FW / src, data_dir / name)
    print(out)


if __name__ == "__main__":
    main()
