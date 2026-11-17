"""Supplementary Fig. S20 -- three G-score cuts from one rule (ordered edge, binary cut, disordered edge).

Panels
  a  Likelihood ratio (LR) of disorder over G for each reference -- A, DisProt disorder against
     residues observed in X-ray structures; B, residues missing from every covering X-ray structure
     against observed ones; C, same-entry {1H}-15N hetNOE < 0.5 -- and their consensus
     sqrt(LR_S LR_C), LR_S = sqrt(LR_A LR_B). Horizontal guides: the LR at which the posterior
     reaches 0.10 / 0.50 / 0.90 at the stated prevalence pi.
  b  Posterior P(disordered | G) at pi, per reference and for the consensus. Points with bars: the
     estimated cuts (posterior 0.10 / 0.50 / 0.90) with their 95 % intervals; dashed: the published
     values 0.15 / 0.35 / 0.70.
  c  Bootstrap distribution of the three estimated cuts.
  d  G of the scored residues (log density) of the analysis set and of the whole release, with the
     three zones at the published values (ordered G < 0.15, indeterminate, disordered G >= 0.70).
  e  Agreement of the binary call G >= t between same-condition replicate entries (Cohen's kappa)
     over t: strict-tier unbound chains (the analysis set) and all unbound chains.
Curves: primary estimator (log LR interpolated between isotonic block midpoints). Bands: 95 %
joint cluster-bootstrap intervals (sequence clusters resampled jointly across references).

Input: the output directory of scripts/validation/threshold_calibration/framework/compute.py
  $TRIZOD_THRESHOLD_FRAMEWORK_DIR  default <data root>/data/interim/threshold-framework-port-2026-11-17

Run: uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s20_threshold.py \
         --out-dir <draft>/Figures/paper
Outputs: <out-dir>/S20.pdf and <out-dir>/suppfigs_d_data/S20_*.csv.
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

W, H = sp.W_DOUBLE, 108.0  # mm
REF = {
    "A": (sp.OKABE_ITO["blue"], "A  DisProt / X-ray observed"),
    "B": (sp.OKABE_ITO["orange"], "B  missing in X-ray"),
    "C": (sp.OKABE_ITO["green"], "C  hetNOE < 0.5"),
}
CONS = sp.INK
EDGE = {  # level -> (colour, name, result key)
    "0.10": (sp.OKABE_ITO["skyblue"], "ordered edge", "ordered_edge"),
    "0.50": (sp.INK, "binary cut", "binary_cut"),
    "0.90": (sp.OKABE_ITO["purple"], "disordered edge", "disordered_edge"),
}
ZONE_FILL = (sp.OKABE_ITO["skyblue"], sp.CAT["light"], sp.OKABE_ITO["purple"])


def _paper(ax, paper, ymax=None) -> None:
    for v in paper:
        ax.axvline(v, color=sp.INK_SOFT, lw=0.6, ls="--", zorder=1)


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
    x = fine.G

    fig = sp.figure(W, H)
    gs = fig.add_gridspec(
        2, 6, left=0.075, right=0.985, bottom=0.09, top=0.95, wspace=1.35, hspace=0.5
    )
    axa, axb, axc = (fig.add_subplot(gs[0, 2 * i : 2 * i + 2]) for i in range(3))
    axd, axe = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])

    # a: likelihood ratios
    for r, (col, lab) in REF.items():
        axa.fill_between(
            x,
            np.exp(fine[f"logLR_{r}_lo"]),
            np.exp(fine[f"logLR_{r}_hi"]),
            color=col,
            alpha=0.12,
            lw=0,
        )
        axa.plot(x, np.exp(fine[f"logLR_{r}"]), color=col, lw=0.8, label=lab)
    axa.fill_between(
        x, np.exp(fine.logLR_lo), np.exp(fine.logLR_hi), color=CONS, alpha=0.18, lw=0
    )
    axa.plot(x, np.exp(fine.logLR), color=CONS, lw=1.4, label="consensus")
    for lev, v in cons["lr_levels_at_pi"].items():
        axa.axhline(v, color=sp.CAT["light"], lw=0.6, ls=":", zorder=1)
        axa.text(
            0.99,
            v / 1.12,
            f"P = {float(lev):.2f}",
            fontsize=sp.FS_SMALL,
            color=sp.INK_SOFT,
            va="top",
            ha="right",
        )
    axa.set_yscale("log")
    axa.set_ylim(0.02, 3000)
    axa.set_ylabel("Likelihood ratio of disorder")
    axa.legend(
        loc="upper left", fontsize=sp.FS_SMALL, handlelength=1.2, borderaxespad=0.2
    )

    # b: posterior at the stated prevalence, with the estimated cuts
    for r, (col, _) in REF.items():
        axb.fill_between(
            x, fine[f"post_{r}_lo"], fine[f"post_{r}_hi"], color=col, alpha=0.10, lw=0
        )
        axb.plot(x, fine[f"post_{r}"], color=col, lw=0.8)
    axb.fill_between(x, fine.post_lo, fine.post_hi, color=CONS, alpha=0.18, lw=0)
    axb.plot(x, fine.post, color=CONS, lw=1.4)
    _paper(axb, paper)
    for lev, (col, _name, _) in EDGE.items():
        c = cons["cuts"][lev]
        axb.axhline(float(lev), color=sp.CAT["light"], lw=0.6, ls=":", zorder=1)
        if c["t"] is not None and c["CI"][0] is not None:
            axb.errorbar(
                c["t"],
                float(lev),
                xerr=[[c["t"] - c["CI"][0]], [c["CI"][1] - c["t"]]],
                fmt="o",
                ms=3,
                color=col,
                mec=sp.INK,
                mew=0.5,
                elinewidth=1.0,
                capsize=1.5,
                zorder=5,
            )
    axb.text(
        0.99,
        0.03,
        f"$\\pi$ = {pi:.2f}",
        transform=axb.transAxes,
        ha="right",
        fontsize=sp.FS_SMALL,
    )
    axb.set_ylim(0, 1)
    axb.set_ylabel("P(disordered | G)")

    # c: bootstrap distribution of the cuts
    bins = np.arange(0, 1.0001, 0.01)
    for col, name, key in EDGE.values():
        v = boot[f"{key}_{est}"].dropna()
        axc.hist(
            v,
            bins=bins,
            weights=np.full(len(v), 1 / len(boot)),
            color=col,
            alpha=0.75,
            lw=0,
            label=name,
        )
    _paper(axc, paper)
    axc.set_ylabel("Share of bootstrap replicates")
    axc.set_ylim(0, None)
    axc.legend(
        loc="upper right", fontsize=sp.FS_SMALL, handlelength=1.0, borderaxespad=0.2
    )

    # d: G distribution and the three zones
    width = 0.01
    edges = [0.0, paper[0], paper[2], 1.0]
    for i, col in enumerate(ZONE_FILL):
        axd.axvspan(edges[i], edges[i + 1], color=col, alpha=0.12, lw=0, zorder=0)
    for col_name, col, lab in (
        ("tier_unfiltered+", sp.INK_SOFT, "all release chains"),
        ("analysis_set", sp.INK, "analysis set"),
    ):
        dens = gh[col_name] / (gh[col_name].sum() * width)
        axd.step(
            np.r_[gh.G_lo, 1.0],
            np.r_[dens, dens.iloc[-1]],
            where="post",
            color=col,
            lw=0.9 if col == sp.INK else 0.7,
            ls="-" if col == sp.INK else "--",
            label=lab,
        )
    _paper(axd, paper)
    z = zs[(zs.set == "analysis_set") & (zs.cuts_from == "paper_values")].iloc[0]
    for i, key in enumerate(("ordered", "indeterminate", "disordered")):
        xm = (
            (paper[1] + paper[2]) / 2
            if key == "indeterminate"
            else (edges[i] + edges[i + 1]) / 2
        )
        axd.text(
            xm,
            0.03,
            f"{key}\n{100 * z[key]:.0f} %",
            transform=axd.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=sp.FS_SMALL,
            color=sp.INK_SOFT,
        )
    axd.set_yscale("log")
    axd.set_ylabel("Density of residues")
    axd.set_xlabel("G-score")
    axd.legend(
        loc="upper right", fontsize=sp.FS_SMALL, handlelength=1.6, borderaxespad=0.2
    )

    # e: replicate agreement of the binary call
    for setname, col, lab in (
        ("analysis_set", sp.INK, "analysis set"),
        ("all_unbound_chains", sp.INK_SOFT, "all unbound chains"),
    ):
        k = kap[kap.set == setname]
        axe.fill_between(k.t, k.lo, k.hi, color=col, alpha=0.15, lw=0)
        axe.plot(
            k.t,
            k.kappa,
            color=col,
            lw=1.0,
            ls="-" if col == sp.INK else "--",
            label=lab,
        )
    _paper(axe, paper)
    kb = R["replicates"]["analysis_set"]["paper_values"]["kappa_binary"]
    axe.annotate(
        f"$\\kappa$ = {kb[0]:.2f}",
        xy=(paper[1], kb[0]),
        xytext=(paper[1] + 0.12, kb[0] + 0.02),
        fontsize=sp.FS_SMALL,
        arrowprops={"arrowstyle": "-", "lw": 0.5, "color": sp.INK_SOFT},
    )
    axe.set_ylim(0.4, 1.0)
    axe.set_ylabel("Replicate agreement ($\\kappa$)")
    axe.set_xlabel("Cut-off $t$ (disordered if $G \\geq t$)")
    axe.legend(
        loc="lower center", fontsize=sp.FS_SMALL, handlelength=1.6, borderaxespad=0.2
    )

    for ax in (axa, axb, axc):
        ax.set_xlabel("G-score")
    for ax in (axa, axb, axc, axd, axe):
        ax.set_xlim(0, 1)
    for ax, letter in zip((axa, axb, axc, axd, axe), "abcde"):
        sp.panel_label(ax, letter, x=-0.3 if ax in (axa, axb, axc) else -0.17, y=1.03)

    out = sp.save(fig, "S20")
    data_dir = sp.out_dir("suppfigs_d_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    for src, name in (
        ("curves_fine.csv", "S20ab_lr_posterior_curves.csv"),
        ("boot_thresholds.csv", "S20c_boot_thresholds.csv"),
        ("g_histogram.csv", "S20d_g_histogram.csv"),
        ("zone_shares.csv", "S20d_zone_shares.csv"),
        ("replicate_kappa_curve.csv", "S20e_replicate_kappa.csv"),
    ):
        shutil.copyfile(FW / src, data_dir / name)
    print(out)


if __name__ == "__main__":
    main()
