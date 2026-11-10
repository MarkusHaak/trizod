#!/usr/bin/env python3
"""Supplementary Fig. S6 -- re-referencing in depth.

a  Distribution of the per-chain LACS offsets and of the residual POTENCI/AIC offsets
   (``off_<A>_sigma`` x sigma(n), in ppm), per nucleus, over all released chains. CA and
   CB share one 13C LACS offset, so they form one panel. Zeros (no offset applied) are
   counted, not drawn.
b  Agreement with PANAV per nucleus (the shift lists of Fig. 2b, matched by ``st_id`` with
   ``fig2b_panav.match_st``): Pearson r over all matched lists, Pearson r inside the
   plotting window of Fig. 2b and Spearman rho over all lists, each for the LACS offset
   alone and for the TriZOD total offset (LACS + residual).
c  The LACS reimplementation against the LACS reports distributed by the BMRB
   (``data/external/bmrb_lacs/bmr<ID>_LACS.str``, parsed with
   ``scripts/compare_lacs_bmrb.py:parse_bmrb_lacs_file``), for entries with exactly one
   scored chain. The reports state the MATLAB sign convention (offset = value to ADD),
   so they are negated. The reports give no 15N or 1H(N) offset.
d  Offset recovery on deposited shifts: a seeded sample of released chains is rescored
   with the released scoring function ``trizod.scoring.get_offset_corrected_shifts``
   after a known offset delta has been added to every 13C backbone shift (C, CA, CB) or
   to every 15N shift of the chain's deposited backbone shifts (``trizod_shifts.parquet``,
   ``is_backbone``). The recovered offset is the change in the chain's total offset
   relative to delta = 0, in modes ``lacs``, ``potenci-only`` and ``both`` (released).
   This step runs in a subprocess that never imports pyarrow (it runs scoring code).

Inputs (read-only): release Parquets, PANAV offsets and the Fig. 2b loop cache
(``tmp/paper_figs/panav_loop_to_st_id.json``), BMRB LACS reports.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s6_rereferencing.py \
        --out-dir <draft>/Figures/paper
Outputs: ``<out-dir>/S6.pdf`` and ``<out-dir>/suppfigs_c_data/S6_*.csv``; prints every
plotted statistic. ``--recompute`` reruns the recovery experiment (cached otherwise in
``tmp/paper_figs/s6_recovery.json``).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402

from trizod.offsets import (  # noqa: E402
    lacs_off_ppm_col,
    off_sigma_col,
    total_off_ppm_col,
)

ATOMS = ["C", "CA", "CB", "H", "HA", "HB", "N"]
SIGMA = {  # trizod.constants.REFINED_WEIGHTS (ppm), copied to keep pyarrow-free imports
    "C": 0.1846,
    "CA": 0.1982,
    "CB": 0.1544,
    "HA": 0.02631,
    "H": 0.06708,
    "N": 0.4722,
    "HB": 0.02154,
}
# panel a: nucleus -> (Parquet atom, histogram half-width in ppm, label)
DIST = [
    ("CA", 4.0, "Cα/Cβ"),
    ("C", 4.0, "C′"),
    ("N", 6.0, "N"),
    ("H", 0.6, "Hᴺ"),
    ("HA", 0.6, "Hα"),
]
# panel c: report atom -> (window ppm, label); CA and CB carry one 13C offset in both
LACS_CMP = [("CA", 4.0, "Cα/Cβ"), ("C", 4.0, "C′"), ("HA", 0.8, "Hα")]
BMRB_LACS = sp.BMRB_LACS
# panel d
REC_SEED = 2026
REC_N_CHAINS = 150
REC_MIN_SCORED = 40
REC_DELTAS = [-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
REC_MODES = ["lacs", "potenci-only", "both"]
REC_CACHE = sp.TMP / "paper_figs" / "s6_recovery.json"
MODE_STYLE = {
    "lacs": (sp.OKABE_ITO["blue"], "LACS only (lacs)"),
    "potenci-only": (sp.OKABE_ITO["orange"], "POTENCI/AIC only (potenci-only)"),
    "both": (sp.OKABE_ITO["black"], "both stages (both; released)"),
}
FS_SMALL = 5.3
DATA_SUBDIR = "suppfigs_c_data"  # under --out-dir


# --------------------------------------------------------------------------------
# a: offset distributions
# --------------------------------------------------------------------------------
def offset_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, _, _ in DIST:
        lacs = df[lacs_off_ppm_col(a)].astype(float).to_numpy()
        res = df[off_sigma_col(a)].astype(float).to_numpy() * SIGMA[a]
        for name, v in (("lacs", lacs), ("residual", res)):
            nz = v[np.isfinite(v) & (v != 0)]
            rows.append(
                {
                    "nucleus": a,
                    "stage": name,
                    "n_chains": len(v),
                    "n_nonzero": len(nz),
                    "median_abs_nonzero_ppm": float(np.median(np.abs(nz)))
                    if len(nz)
                    else np.nan,
                    "q90_abs_nonzero_ppm": float(np.quantile(np.abs(nz), 0.9))
                    if len(nz)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------
# b: PANAV statistics
# --------------------------------------------------------------------------------
def panav_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    import fig2b_panav as f2b
    from scipy.stats import spearmanr

    pan = json.loads(sp.PANAV.read_text())
    stmap = f2b.loop_to_st_map(pan, workers=4)
    df = sp.load_main(
        ["id", "entry_id", "st_id", "total_bbshifts"]
        + [total_off_ppm_col(a) for a in f2b.ATOMS]
        + [lacs_off_ppm_col(a) for a in ATOMS]
    )
    df["entry_id"] = df.entry_id.astype(str)
    df["st_id"] = df.st_id.astype(str)
    M, _ = f2b.match_st(pan, stmap, df)
    rows = []
    for a in f2b.ATOMS:
        x = M[f"neg_panav_{a}"].to_numpy(float)
        L = f2b.LIMS[a]
        for which in ("lacs", "total"):
            y = M[f"trizod_{which}_{a}"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            win = m & (np.abs(x) <= L) & (np.abs(y) <= L)
            rows.append(
                {
                    "nucleus": a,
                    "offset": which,
                    "n_lists": int(m.sum()),
                    "n_in_window": int(win.sum()),
                    "window_ppm": L,
                    "pearson_all": f2b.pearson(x[m], y[m]),
                    "pearson_window": f2b.pearson(x[win], y[win]),
                    "spearman_all": float(spearmanr(x[m], y[m]).statistic),
                }
            )
    return pd.DataFrame(rows), M


# --------------------------------------------------------------------------------
# c: LACS reimplementation vs BMRB LACS reports
# --------------------------------------------------------------------------------
def bmrb_lacs_reports() -> pd.DataFrame:
    sys.path.insert(0, str(sp.REPO / "scripts"))
    from compare_lacs_bmrb import parse_bmrb_lacs_file

    rows = []
    for p in sorted(BMRB_LACS.glob("bmr*_LACS.str")):
        m = re.match(r"bmr(\d+)_LACS\.str", p.name)
        if not m:
            continue
        off = parse_bmrb_lacs_file(p.read_text())
        rows.append(
            {"entry_id": m.group(1), **{f"bmrb_{k}": v for k, v in off.items()}}
        )
    return pd.DataFrame(rows)


def lacs_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    B = bmrb_lacs_reports()
    n_chains = df.entry_id.value_counts()
    one = df[df.entry_id.map(n_chains) == 1]
    M = one.merge(B, on="entry_id", how="inner")
    rows = []
    for a, L, _ in LACS_CMP:
        x = -pd.to_numeric(M[f"bmrb_{a}"], errors="coerce").to_numpy(float)
        y = M[lacs_off_ppm_col(a)].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        d = np.abs(x[m] - y[m])
        rows.append(
            {
                "nucleus": a,
                "n_entries": int(m.sum()),
                "pearson": float(np.corrcoef(x[m], y[m])[0, 1]),
                "median_abs_diff_ppm": float(np.median(d)),
                "share_within_0p1_ppm": float((d <= 0.1 + 1e-9).mean()),
                "share_within_0p25_ppm": float((d <= 0.25 + 1e-9).mean()),
                "n_outside_window": int(
                    ((np.abs(x[m]) > L) | (np.abs(y[m]) > L)).sum()
                ),
            }
        )
    info = {
        "n_reports": len(B),
        "n_single_chain_entries_with_report": len(M),
    }
    return pd.DataFrame(rows), M, info


# --------------------------------------------------------------------------------
# d: offset recovery (subprocess without pyarrow)
# --------------------------------------------------------------------------------
def recovery_sample() -> dict:
    """Seeded sample of chains with their deposited backbone shift rows."""
    import pyarrow.parquet as pq

    df = sp.load_main(
        ["id", "sequence", "n_scored", "temperature", "ph", "ionic_strength"]
    )
    pool = df[df.n_scored >= REC_MIN_SCORED].sort_values("id").reset_index(drop=True)
    rng = np.random.default_rng(REC_SEED)
    pick = pool.iloc[np.sort(rng.choice(len(pool), REC_N_CHAINS, replace=False))]
    ids = list(pick.id)
    sh = pq.read_table(
        sp.PQ_SHIFTS,
        columns=[
            "id",
            "seq_id",
            "comp_id",
            "atom_id",
            "atom_type",
            "val_ppm",
            "val_err_ppm",
            "ambiguity_code",
            "is_backbone",
        ],  # fmt: skip
        filters=[("id", "in", ids), ("is_backbone", "==", True)],
    ).to_pandas()
    out = {}
    for r in pick.itertuples():
        s = sh[sh.id == r.id]
        rows = [
            [
                "1",
                "1",
                str(int(q.seq_id)),
                q.comp_id,
                q.atom_id,
                q.atom_type,
                float(q.val_ppm),
                None if pd.isna(q.val_err_ppm) else float(q.val_err_ppm),
                "." if pd.isna(q.ambiguity_code) else str(int(q.ambiguity_code)),
            ]
            for q in s.itertuples()
        ]
        out[r.id] = {
            "seq": r.sequence,
            "T": 298.0 if pd.isna(r.temperature) else float(r.temperature),
            "pH": 7.0 if pd.isna(r.ph) else float(r.ph),
            "ion": 0.1 if pd.isna(r.ionic_strength) else float(r.ionic_strength),
            "rows": rows,
        }
    return {"pool_size": len(pool), "chains": out}


_WORKER = r"""
import json, logging, sys, warnings
import numpy as np
logging.disable(logging.CRITICAL)
warnings.simplefilter("ignore")
from trizod.potenci import potenci
from trizod.scoring.scoring import get_offset_corrected_shifts
from trizod.constants import BACKBONE_ATOMS, REFINED_WEIGHTS

def total(ret):
    off, lacs = ret[4], ret[9]
    return {a: float(lacs.get(a, 0.0)) + float(off.get(a, 0.0)) * REFINED_WEIGHTS[a] for a in BACKBONE_ATOMS}

def perturb(rows, nuclei, d):
    out = []
    for r in rows:
        r = list(r)
        if r[4] in nuclei:
            r[6] = r[6] + d
        out.append(r)
    return out

job = json.load(open(sys.argv[1]))
res = []
for cid, c in job["chains"].items():
    try:
        pred = potenci.get_pred_shifts(c["seq"], c["T"], c["pH"], c["ion"], c["pH"] != 7.0)
    except Exception:
        continue
    for pert, nuclei in (("13C", ["C", "CA", "CB"]), ("15N", ["N"])):
        for mode in job["modes"]:
            base = None
            for d in job["deltas"]:
                ret = get_offset_corrected_shifts(c["seq"], perturb(c["rows"], nuclei, d), pred, rereference_mode=mode)
                if ret is None:
                    break
                t = total(ret)
                if d == 0.0:
                    base = t
                res.append({"id": cid, "perturbed": pert, "mode": mode, "delta": d, **{f"total_{a}": t[a] for a in ("C", "CA", "CB", "N")}})
json.dump(res, open(sys.argv[2], "w"))
"""


def run_recovery(recompute: bool) -> tuple[pd.DataFrame, dict]:
    if REC_CACHE.exists() and not recompute:
        d = json.loads(REC_CACHE.read_text())
        return pd.DataFrame(d["rows"]), d["info"]
    sample = recovery_sample()
    REC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    job = REC_CACHE.with_suffix(".job.json")
    out = REC_CACHE.with_suffix(".out.json")
    job.write_text(
        json.dumps(
            {"chains": sample["chains"], "deltas": REC_DELTAS, "modes": REC_MODES}
        )
    )
    code = REC_CACHE.with_suffix(".worker.py")
    code.write_text(_WORKER)
    # The scoring code runs in the project environment WITHOUT the pyarrow extra (pandas
    # would import pyarrow on its own wherever it is installed), in a fresh interpreter.
    subprocess.run(
        ["uv", "run", "python", str(code), str(job), str(out)], check=True, cwd=sp.REPO
    )
    rows = json.loads(out.read_text())
    info = {
        "seed": REC_SEED,
        "n_sampled": REC_N_CHAINS,
        "pool_size": sample["pool_size"],
        "min_scored": REC_MIN_SCORED,
        "deltas": REC_DELTAS,
    }
    REC_CACHE.write_text(json.dumps({"rows": rows, "info": info}))
    for p in (job, out, code):
        p.unlink()
    return pd.DataFrame(rows), info


def recovery_summary(R: pd.DataFrame) -> pd.DataFrame:
    """Recovered offset = total(delta) - total(0) per chain, summarised per delta."""
    rows = []
    for (pert, mode), g in R.groupby(["perturbed", "mode"]):
        atoms = ["CA", "C"] if pert == "13C" else ["N"]
        base = g[g.delta == 0.0].set_index("id")
        for a in atoms:
            for d, gd in g.groupby("delta"):
                gd = gd.set_index("id")
                common = gd.index.intersection(base.index)
                rec = gd.loc[common, f"total_{a}"] - base.loc[common, f"total_{a}"]
                rows.append(
                    {
                        "perturbed": pert,
                        "nucleus": a,
                        "mode": mode,
                        "delta_ppm": d,
                        "n_chains": len(rec),
                        "median_recovered_ppm": float(np.median(rec)),
                        "q25": float(np.quantile(rec, 0.25)),
                        "q75": float(np.quantile(rec, 0.75)),
                        "share_within_0p1_ppm": float(
                            (np.abs(rec - d) <= 0.1 + 1e-9).mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------------
FIG_W, FIG_H = sp.W_DOUBLE, 150.0


def _ax(fig, left, top, w, h):
    return fig.add_axes([left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])


def plot(df, T_off, PS, LC, M_lacs, S_rec, rec_info) -> None:
    from matplotlib.lines import Line2D

    fig = sp.figure(FIG_W, FIG_H)
    c_lacs, c_res = sp.OKABE_ITO["blue"], sp.OKABE_ITO["vermillion"]

    # ---- a: offset distributions -------------------------------------------
    w, gap, top, h = 28.0, 6.4, 11.0, 24.0
    for j, (a, L, lab) in enumerate(DIST):
        ax = _ax(fig, 12.0 + j * (w + gap), top, w, h)
        bins = np.linspace(-L, L, 41)
        for stage, col in (("lacs", c_lacs), ("residual", c_res)):
            if stage == "lacs":
                v = df[lacs_off_ppm_col(a)].astype(float).to_numpy()
            else:
                v = df[off_sigma_col(a)].astype(float).to_numpy() * SIGMA[a]
            v = v[np.isfinite(v) & (v != 0)]
            ax.hist(np.clip(v, -L, L), bins=bins, histtype="step", color=col, lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.7, 2e4)
        ax.set_xlim(-L, L)
        ax.set_xlabel("offset (ppm)")
        if j == 0:
            ax.set_ylabel("chains")
        else:
            ax.tick_params(axis="y", labelleft=False)
        r = T_off.set_index(["nucleus", "stage"])
        n_l = int(r.loc[(a, "lacs"), "n_nonzero"])
        n_r = int(r.loc[(a, "residual"), "n_nonzero"])
        ax.set_title(lab, fontsize=6, pad=2)
        ax.text(0.03, 0.98, f"{n_l:,}", transform=ax.transAxes, ha="left", va="top",
                fontsize=FS_SMALL, color=c_lacs)  # fmt: skip
        ax.text(0.97, 0.98, f"{n_r:,}", transform=ax.transAxes, ha="right", va="top",
                fontsize=FS_SMALL, color=c_res)  # fmt: skip
    fig.legend(
        handles=[
            Line2D([], [], color=c_lacs, lw=0.9, label="LACS offset"),
            Line2D([], [], color=c_res, lw=0.9, label="residual POTENCI/AIC offset"),
        ],
        loc="upper left",
        bbox_to_anchor=(12.0 / FIG_W, 1 - 0.6 / FIG_H),
        ncol=2,
        handlelength=1.6,
        columnspacing=1.2,
        borderaxespad=0,
    )
    sp.fig_label(fig, "a", 0.5, 1.0)

    # ---- b: PANAV statistics ------------------------------------------------
    ax = _ax(fig, 14.0, 56.0, 52.0, 36.0)
    nuc = ["CA", "CB", "C", "N"]
    lab = {"CA": "Cα", "CB": "Cβ", "C": "C′", "N": "N"}
    stats = [("pearson_all", "Pearson r, all lists", "o"),
             ("pearson_window", "Pearson r, in window", "s"),
             ("spearman_all", "Spearman ρ, all lists", "D")]  # fmt: skip
    P = PS.set_index(["nucleus", "offset"])
    for i, a in enumerate(nuc):
        y0 = len(nuc) - 1 - i
        if i:
            ax.axhline(y0 + 0.5, color="0.9", lw=0.4)
        for s_i, (key, _, mk) in enumerate(stats):
            y = y0 + 0.25 - 0.25 * s_i
            for which, fc in (("total", sp.OKABE_ITO["blue"]), ("lacs", "white")):
                ax.plot(P.loc[(a, which), key], y, marker=mk, ms=2.8, mfc=fc,
                        mec=sp.OKABE_ITO["blue"], mew=0.6, ls="none")  # fmt: skip
    ax.set_yticks(range(len(nuc)), [lab[a] for a in nuc][::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.5, len(nuc) - 0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("correlation with −PANAV correction")
    n_lists = int(P.loc[("CA", "total"), "n_lists"])
    ax.set_title(
        f"{n_lists:,} matched shift lists", fontsize=FS_SMALL + 0.5, loc="left", pad=2
    )
    h1 = [Line2D([], [], marker=mk, ls="none", mfc="0.5", mec="0.5", ms=2.8, label=t)
          for _, t, mk in stats]  # fmt: skip
    h2 = [
        Line2D([], [], marker="o", ls="none", mfc=sp.OKABE_ITO["blue"], mec=sp.OKABE_ITO["blue"], ms=2.8, label="TriZOD total offset"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec=sp.OKABE_ITO["blue"], ms=2.8, label="LACS offset alone"),
    ]  # fmt: skip
    fig.legend(handles=h1 + h2, loc="upper left", bbox_to_anchor=(14.0 / FIG_W, 1 - 101.0 / FIG_H),
               fontsize=FS_SMALL, handletextpad=0.3, labelspacing=0.25, ncol=2,
               columnspacing=0.8, borderaxespad=0)  # fmt: skip
    sp.fig_label(fig, "b", 0.5, 47.0)

    # ---- c: LACS vs BMRB reports -------------------------------------------
    L_info = LC.set_index("nucleus")
    for j, (a, L, labc) in enumerate(LACS_CMP):
        ax = _ax(fig, 86.0 + j * 32.0, 54.0, 27.0, 27.0)
        x = -pd.to_numeric(M_lacs[f"bmrb_{a}"], errors="coerce").to_numpy(float)
        y = M_lacs[lacs_off_ppm_col(a)].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        out = (np.abs(x) > L) | (np.abs(y) > L)
        ax.plot([-L, L], [-L, L], color="0.6", lw=0.5, ls="--", zorder=0)
        ax.scatter(
            x[~out], y[~out], s=1.2, lw=0, color=sp.OKABE_ITO["green"], alpha=0.4
        )
        if out.any():
            ax.scatter(np.clip(x[out], -L, L), np.clip(y[out], -L, L), s=4, marker="o",
                       facecolors="none", edgecolors=sp.OKABE_ITO["vermillion"], lw=0.4)  # fmt: skip
        ax.set_xlim(-L * 1.04, L * 1.04)
        ax.set_ylim(-L * 1.04, L * 1.04)
        ax.set_aspect("equal")
        t = 2 if L >= 2 else 0.4
        ax.set_xticks(np.round(np.arange(-L, L + 1e-9, t), 2))
        ax.set_yticks(np.round(np.arange(-L, L + 1e-9, t), 2))
        r = L_info.loc[a]
        ax.text(0.04, 0.97, labc, transform=ax.transAxes, ha="left", va="top",
                fontsize=FS_SMALL + 1, fontweight="bold")  # fmt: skip
        ax.text(0.04, 0.84,
                f"r = {r.pearson:.3f}\nmedian |Δ| = {r.median_abs_diff_ppm:.2f} ppm\nn = {int(r.n_entries):,}",
                transform=ax.transAxes, ha="left", va="top", fontsize=FS_SMALL, linespacing=1.15)  # fmt: skip
        if out.any():
            ax.text(0.97, 0.03, f"{int(out.sum())} outside", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=FS_SMALL, color=sp.OKABE_ITO["vermillion"])  # fmt: skip
        ax.set_xlabel("−BMRB LACS report (ppm)")
        if j == 0:
            ax.set_ylabel("TriZOD LACS offset (ppm)")
    sp.fig_label(fig, "c", 78.0, 47.0)

    # ---- d: offset recovery -------------------------------------------------
    panels = [("13C", "CA", "Cα (offset added to C′, Cα, Cβ)"),
              ("13C", "C", "C′ (offset added to C′, Cα, Cβ)"),
              ("15N", "N", "N (offset added to N)")]  # fmt: skip
    nz = [d for d in REC_DELTAS if d != 0.0]
    for j, (pert, a, labd) in enumerate(panels):
        ax = _ax(fig, 14.0 + j * 42.0, 118.0, 34.0, 24.0)
        for mode in REC_MODES:
            col, _ = MODE_STYLE[mode]
            g = S_rec[
                (S_rec.perturbed == pert)
                & (S_rec.nucleus == a)
                & (S_rec["mode"] == mode)
            ]
            g = g[g.delta_ppm != 0.0].sort_values("delta_ppm")
            for side in (g[g.delta_ppm < 0], g[g.delta_ppm > 0]):
                ax.plot(side.delta_ppm, 100 * side.share_within_0p1_ppm, color=col, lw=0.9,
                        marker="o", ms=2.0, ls=(0, (3, 1.5)) if mode == "lacs" else "-")  # fmt: skip
        ax.set_xlim(min(nz) - 0.3, max(nz) + 0.3)
        ax.set_ylim(0, 100)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_title(labd, fontsize=6, pad=2)
        ax.set_xlabel("offset added (ppm)")
        if j == 0:
            ax.set_ylabel("chains recovered (%)")
        else:
            ax.tick_params(axis="y", labelleft=False)
    hd = [Line2D([], [], color=MODE_STYLE[m][0], lw=0.9, marker="o", ms=2.0,
                 ls=(0, (3, 1.5)) if m == "lacs" else "-", label=MODE_STYLE[m][1])
          for m in REC_MODES]  # fmt: skip
    fig.legend(handles=hd, loc="upper left", bbox_to_anchor=(142.0 / FIG_W, 1 - 119.0 / FIG_H),
               fontsize=FS_SMALL, handlelength=2.2, labelspacing=0.5, borderaxespad=0,
               title=f"{rec_info['n_sampled']} chains; recovered =\nwithin 0.1 ppm of the offset added",
               title_fontsize=FS_SMALL, alignment="left")  # fmt: skip
    sp.fig_label(fig, "d", 0.5, 111.0)
    sp.save(fig, "S6")


def main() -> None:
    ap = sp.cli(__doc__)
    ap.add_argument(
        "--recompute", action="store_true", help="rerun the recovery experiment"
    )
    args = sp.parse_args(ap)
    DATA_DIR = sp.out_dir(DATA_SUBDIR)

    df = sp.load_main(
        ["id", "entry_id", "st_id"]
        + [lacs_off_ppm_col(a) for a in ATOMS]
        + [off_sigma_col(a) for a in ATOMS]
    )
    df["entry_id"] = df.entry_id.astype(str)
    T_off = offset_table(df)
    print("a: offsets per nucleus (non-zero chains of", len(df), ")")
    print(T_off.round(3).to_string(index=False))

    PS, M_pan = panav_stats()
    print("b: PANAV agreement")
    print(PS.round(3).to_string(index=False))

    LC, M_lacs, lc_info = lacs_comparison(df)
    print("c: LACS reimplementation vs BMRB reports", lc_info)
    print(LC.round(3).to_string(index=False))

    R, rec_info = run_recovery(args.recompute)
    S_rec = recovery_summary(R)
    print("d: offset recovery", rec_info, "chains scored:", R.id.nunique())
    print(S_rec.round(3).to_string(index=False))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    T_off.to_csv(DATA_DIR / "S6a_offsets.csv", index=False)
    PS.to_csv(DATA_DIR / "S6b_panav_stats.csv", index=False)
    LC.to_csv(DATA_DIR / "S6c_lacs_vs_bmrb.csv", index=False)
    cols = (
        ["entry_id", "id"]
        + [lacs_off_ppm_col(a) for a, _, _ in LACS_CMP]
        + [f"bmrb_{a}" for a, _, _ in LACS_CMP]
    )
    M_lacs[cols].to_csv(DATA_DIR / "S6c_lacs_vs_bmrb_entries.csv", index=False)
    S_rec.to_csv(DATA_DIR / "S6d_recovery.csv", index=False)
    R.to_csv(DATA_DIR / "S6d_recovery_chains.csv", index=False)
    (DATA_DIR / "S6_info.json").write_text(
        json.dumps({"lacs_cmp": lc_info, "recovery": rec_info}, indent=1)
    )
    plot(df, T_off, PS, LC, M_lacs, S_rec, rec_info)
    print(f"wrote {sp.out_dir() / 'S6.pdf'}")


if __name__ == "__main__":
    main()
