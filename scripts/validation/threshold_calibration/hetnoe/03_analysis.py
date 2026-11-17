"""Step 3-5: G vs hetNOE (and S2) calibration. Spearman (per chain, pooled with
cluster bootstrap), MCC-optimal threshold per literature hetNOE cut with
cluster-bootstrap CIs, MCC at 0.40 vs 0.65, isotonic reference-free crossing,
caveat analyses (field, termini, filter, cross-entry), figure.

Metric/bootstrap code follows threshold-calibration-2026-11-16/scripts/02_analysis.py
(fine 0.001 G bins, 1000 resamples of whole clusters, percentile CIs).
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

# Data root: the checkout that holds data/ (set TRIZOD_ROOT when running from a worktree).
ROOT = _Path(
    _os.environ.get("TRIZOD_ROOT")
    or next(
        p
        for p in _Path(__file__).resolve().parents
        if (p / "data" / "interim").is_dir()
    )
)

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

BASE = Path(f"{ROOT}/data/interim/threshold-calibration-hetnoe-2026-11-16")
OUT = BASE / "out"
NBOOT = 1000
SEED = 20261116
GRID_J = np.arange(10, 1000, 10)
T = GRID_J / 1000
T_REFS = (0.40, 0.65)
NOE_CUTS = (0.5, 0.6, 0.65)
S2_CUTS = (0.69, 0.8)
MIN_CHAIN_RES = 10


# ------------------------------------------------ shared with 02_analysis.py
def fine_bin(g):
    gi = np.rint(np.asarray(g, float) * 10000).astype(int)
    return np.clip(-(-gi // 10) - 1, 0, 999)


def cluster_counts(df, cluster_col):
    cl, cl_idx = np.unique(df[cluster_col].to_numpy(), return_inverse=True)
    b = fine_bin(df.G.to_numpy())
    y = df.y.to_numpy()
    pos = np.zeros((len(cl), 1000))
    neg = np.zeros((len(cl), 1000))
    np.add.at(pos, (cl_idx[y == 1], b[y == 1]), 1)
    np.add.at(neg, (cl_idx[y == 0], b[y == 0]), 1)
    return pos, neg, cl_idx


def boot_weights(n, rng, nboot=NBOOT):
    W = np.zeros((nboot, n))
    for i in range(nboot):
        W[i] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W


def confusion(pos_hist, neg_hist):
    tp = np.flip(np.cumsum(np.flip(pos_hist, -1), -1), -1)
    fp = np.flip(np.cumsum(np.flip(neg_hist, -1), -1), -1)
    P = pos_hist.sum(-1, keepdims=True)
    N = neg_hist.sum(-1, keepdims=True)
    return tp, fp, P - tp, N - fp


def metrics(tp, fp, fn, tn):
    with np.errstate(divide="ignore", invalid="ignore"):
        mcc = (tp * tn - fp * fn) / np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
    return {"MCC": np.nan_to_num(mcc), "J": tpr + tnr - 1}


def ci(x):
    x = np.asarray(x, float)
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def idx(t):
    return int(np.where(np.isclose(T, t))[0][0])


def calibrate(df, W):
    pos, neg, _ = cluster_counts(df, "cluster")
    tp, fp, fn, tn = confusion(pos.sum(0), neg.sum(0))
    m = {k: v[GRID_J] for k, v in metrics(tp, fp, fn, tn).items()}
    imax = int(np.argmax(m["MCC"]))
    flat = T[m["MCC"] >= m["MCC"].max() - 0.01]
    btp, bfp, bfn, btn = confusion(W @ pos, W @ neg)
    bm = {k: v[:, GRID_J] for k, v in metrics(btp, bfp, bfn, btn).items()}
    b_imax = bm["MCC"].argmax(1)
    b_tstar = T[b_imax]
    b_star = bm["MCC"][np.arange(len(W)), b_imax]
    i40, i65 = idx(0.40), idx(0.65)
    res = {
        "chains": int(df.chain.nunique()),
        "clusters": int(len(pos)),
        "residues": int(len(df)),
        "positives": int(df.y.sum()),
        "prevalence": float(df.y.mean()),
        "ROC_AUC": float(roc_auc_score(df.y, df.G)) if 0 < df.y.mean() < 1 else None,
        "t_star_MCC": float(T[imax]),
        "t_star_MCC_CI": ci(b_tstar),
        "flat_MCC_within_0.01": [float(flat.min()), float(flat.max())],
        "MCC_t_star": float(m["MCC"][imax]),
        "MCC_t_star_CI": ci(b_star),
        "MCC_040": float(m["MCC"][i40]),
        "MCC_040_CI": ci(bm["MCC"][:, i40]),
        "MCC_065": float(m["MCC"][i65]),
        "MCC_065_CI": ci(bm["MCC"][:, i65]),
        "dMCC_040_minus_065": float(m["MCC"][i40] - m["MCC"][i65]),
        "dMCC_040_minus_065_CI": ci(bm["MCC"][:, i40] - bm["MCC"][:, i65]),
        "share_boot_tstar_ge_065": float(np.mean(b_tstar >= 0.65 - 1e-9)),
        "share_boot_tstar_le_040": float(np.mean(b_tstar <= 0.40 + 1e-9)),
        "t_star_J": float(T[int(np.argmax(m["J"]))]),
    }
    curve = pd.DataFrame(
        {
            "t": T,
            "MCC": m["MCC"],
            "MCC_lo": np.nanpercentile(bm["MCC"], 2.5, 0),
            "MCC_hi": np.nanpercentile(bm["MCC"], 97.5, 0),
            "J": m["J"],
        }
    )
    return res, curve


# ------------------------------------------------------------- new pieces
def boot_index(df, W):
    """Per bootstrap: residue weights from cluster weights."""
    cl, cl_idx = np.unique(df.cluster.to_numpy(), return_inverse=True)
    return cl, cl_idx


def spearman_pooled(df, W, cl_idx):
    rho = spearmanr(df.G, df.val).statistic
    b = []
    G, V = df.G.to_numpy(), df.val.to_numpy()
    for w in W:
        rw = w[cl_idx].astype(int)
        sel = np.repeat(np.arange(len(G)), rw)
        b.append(spearmanr(G[sel], V[sel]).statistic)
    return float(rho), ci(b)


def per_chain_spearman(df):
    rows = []
    for ch, g in df.groupby("chain"):
        if len(g) < MIN_CHAIN_RES:
            continue
        r = (
            spearmanr(g.G, g.val).statistic
            if g.G.nunique() > 1 and g.val.nunique() > 1
            else np.nan
        )
        rows.append(
            {
                "chain": ch,
                "cluster": g.cluster.iloc[0],
                "n": len(g),
                "rho": r,
                "G_range": g.G.max() - g.G.min(),
                "noe_range": g.val.max() - g.val.min(),
                "share_noe_lt_0.5": (g.val < 0.5).mean(),
                "share_G_gt_0.4": (g.G > 0.4).mean(),
                "field": g.field.iloc[0],
            }
        )
    return pd.DataFrame(rows)


GRID_G = np.arange(0, 1.0001, 0.001)


def iso_cross(G, V, w, cut):
    ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
    ir.fit(G, V, sample_weight=w)
    p = ir.predict(GRID_G)
    below = np.where(p < cut)[0]
    return (float(GRID_G[below[0]]) if len(below) else np.nan), p


def binned_median_cross(G, V, w, cut, nb=20):
    """Median hetNOE in G-quantile bins; G where the median first drops below cut
    (linear interpolation between bin centres)."""
    order = np.argsort(G)
    Gs, Vs, ws = G[order], V[order], w[order]
    cw = np.cumsum(ws) / ws.sum()
    bins = np.minimum((cw * nb).astype(int), nb - 1)
    xc, med = [], []
    for b in range(nb):
        s = bins == b
        if ws[s].sum() == 0:
            continue
        vv, ww = Vs[s], ws[s]
        o = np.argsort(vv)
        cc = np.cumsum(ww[o]) / ww.sum()
        med.append(vv[o][np.searchsorted(cc, 0.5)])
        xc.append(np.average(Gs[s], weights=ww))
    xc, med = np.array(xc), np.array(med)
    for i in range(1, len(med)):
        if med[i - 1] >= cut > med[i]:
            return (
                float(
                    xc[i - 1]
                    + (cut - med[i - 1]) * (xc[i] - xc[i - 1]) / (med[i] - med[i - 1])
                ),
                xc,
                med,
            )
    return np.nan, xc, med


def reference_free(df, W, cl_idx, cuts):
    out, fits = {}, {}
    G, V = df.G.to_numpy(), df.val.to_numpy()
    one = np.ones(len(G))
    for cut in cuts:
        x0, p0 = iso_cross(G, V, one, cut)
        m0, xc, med = binned_median_cross(G, V, one, cut)
        bx, bm = [], []
        for w in W:
            rw = w[cl_idx]
            s = rw > 0
            bx.append(iso_cross(G[s], V[s], rw[s], cut)[0])
            bm.append(binned_median_cross(G[s], V[s], rw[s], cut)[0])
        out[str(cut)] = {
            "isotonic_mean_crossing": x0,
            "isotonic_CI": ci(bx),
            "binned_median_crossing": m0,
            "binned_median_CI": ci(bm),
            "share_boot_no_crossing_iso": float(np.mean(np.isnan(bx))),
        }
        fits[cut] = (p0, xc, med)
    return out, fits


def pick_one_list(r):
    """One hetNOE list per chain: most residues, then field closest to 600 MHz."""
    s = (
        r.groupby(["chain", "entry_id", "list_id"])
        .agg(n=("val", "size"), field=("field", "first"), dc=("dcond", "first"))
        .reset_index()
    )
    s["df"] = (s.field - 600).abs().fillna(1e9)
    s = s.sort_values(
        ["chain", "dc", "n", "df"], ascending=[True, True, False, True]
    ).drop_duplicates("chain")
    return r.merge(
        s[["chain", "entry_id", "list_id"]], on=["chain", "entry_id", "list_id"]
    )


def prep(kind):
    r = pd.read_parquet(OUT / f"{kind}_residues_all.parquet")
    r["field"] = np.where(r.field > 1e5, r.field / 1e6, r.field)
    r["G"] = r.G.astype(float)
    r = r[r["mask"] & np.isfinite(r.G) & np.isfinite(r.val)].copy()
    r["dcond"] = (r.dpH.fillna(0) / 0.5 + r.dT.fillna(0) / 5.0) if "dpH" in r else 0.0
    r.loc[r.source == "same_entry", "dcond"] = 0.0
    return r


def run_set(name, df, cuts, rng, results, curves, positive_below=True):
    cl, cl_idx = np.unique(df.cluster.to_numpy(), return_inverse=True)
    W = boot_weights(len(cl), rng)
    rho, rho_ci = spearman_pooled(df, W, cl_idx)
    pc = per_chain_spearman(df)
    res = {
        "chains": int(df.chain.nunique()),
        "clusters": int(len(cl)),
        "residues": int(len(df)),
        "spearman_pooled": rho,
        "spearman_pooled_CI": rho_ci,
        "per_chain_n_chains": int(pc.rho.notna().sum()),
        "per_chain_rho_median": float(pc.rho.median()),
        "per_chain_rho_IQR": [
            float(pc.rho.quantile(0.25)),
            float(pc.rho.quantile(0.75)),
        ],
        "per_chain_share_rho_negative": float((pc.rho < 0).mean()),
        "per_chain_share_rho_sig_negative_lt_-0.3": float((pc.rho < -0.3).mean()),
        "cuts": {},
    }
    for cut in cuts:
        d = df.assign(y=(df.val < cut).astype(int))
        if d.y.sum() < 30 or (d.y == 0).sum() < 30:
            res["cuts"][str(cut)] = {"note": "too few", "positives": int(d.y.sum())}
            continue
        rr, cv = calibrate(d, W)
        res["cuts"][str(cut)] = rr
        cv.to_csv(OUT / f"curve_{name}_cut{cut}.csv", index=False, float_format="%.5f")
        curves[(name, cut)] = cv
    rf, fits = reference_free(df, W, cl_idx, cuts)
    res["reference_free"] = rf
    results[name] = res
    return pc, fits


def main():
    rng = np.random.default_rng(SEED)
    results = {
        "settings": {
            "n_boot": NBOOT,
            "seed": SEED,
            "grid": "0.01..0.99",
            "rule": "disordered iff G > t; reference-disordered iff hetNOE < cut (S2 < cut)",
            "bootstrap_unit": "cluster (cluster_repr; else that of a train/redundant chain with the identical sequence; else the sequence)",
        }
    }
    curves = {}
    noe = prep("noe")
    same = noe[noe.source == "same_entry"]
    prim = pick_one_list(same[same.status == "ok"])
    prim.to_parquet(OUT / "noe_residues_primary.parquet")
    pc, fits = run_set("noe_same_entry", prim, NOE_CUTS, rng, results, curves)
    pc.to_csv(
        OUT / "per_chain_spearman_noe_same_entry.csv", index=False, float_format="%.4f"
    )
    # --- sensitivity sets
    sets = {
        "noe_same_entry_no_termini": prim[~prim.terminal],
        "noe_same_entry_train": prim[prim.split == "train"],
        "noe_same_entry_field_le600": prim[prim.field <= 610],
        "noe_same_entry_field_gt600": prim[prim.field > 610],
        "noe_same_entry_k_ge10": prim[prim.k >= 10],
        "noe_same_entry_incl_filtered_out": pick_one_list(
            same[same.status.isin(["ok", "chain removed by chain filter"])]
        ),
    }
    cross = noe[noe.source == "cross_entry"]
    ext = pd.concat([prim, pick_one_list(cross[~cross.chain.isin(prim.chain)])])
    sets["noe_same_plus_cross_entry"] = ext
    sets["noe_cross_entry_only"] = pick_one_list(cross)
    for nm, d in sets.items():
        run_set(nm, d, NOE_CUTS, rng, results, curves)
        print(nm, "done", flush=True)
    # --- field caveat: rigid-limit hetNOE vs field; within-entry multi-field pairs
    rig = prim[prim.G < 0.1]
    fb = pd.cut(
        prim.field,
        [0, 550, 650, 750, 850, 1100],
        labels=["<=550", "600", "700", "800", ">=850"],
    )
    results["field"] = {
        "chains_by_field_bin": prim.drop_duplicates("chain")
        .assign(fb=fb)
        .fb.value_counts()
        .sort_index()
        .astype(int)
        .to_dict(),
        "median_noe_G_lt_0.1_by_field_bin": rig.assign(fb=fb[rig.index])
        .groupby("fb", observed=True)
        .val.median()
        .round(3)
        .to_dict(),
        "median_noe_G_gt_0.6_by_field_bin": prim[prim.G > 0.6]
        .assign(fb=fb[prim.G > 0.6])
        .groupby("fb", observed=True)
        .val.median()
        .round(3)
        .to_dict(),
        "field_missing_chains": int(prim.drop_duplicates("chain").field.isna().sum()),
    }
    ok = same[same.status == "ok"]
    mf = []
    for (ch, eid), g in ok.groupby(["chain", "entry_id"]):
        fl = g.groupby("list_id").field.first().dropna()
        if fl.nunique() < 2:
            continue
        lo, hi = fl.idxmin(), fl.idxmax()
        a = g[g.list_id == lo].set_index("idx0").val
        b = g[g.list_id == hi].set_index("idx0").val
        j = a.index.intersection(b.index)
        if len(j):
            mf.append(
                {
                    "chain": ch,
                    "f_lo": fl.min(),
                    "f_hi": fl.max(),
                    "n": len(j),
                    "median_diff_hi_minus_lo": float((b[j] - a[j]).median()),
                }
            )
    mf = pd.DataFrame(mf)
    mf.to_csv(OUT / "multi_field_pairs.csv", index=False, float_format="%.4f")
    results["field"]["multi_field_chains"] = len(mf)
    if len(mf):
        results["field"]["multi_field_median_diff"] = float(
            mf.median_diff_hi_minus_lo.median()
        )
        results["field"]["multi_field_median_dMHz"] = float(
            (mf.f_hi - mf.f_lo).median()
        )
    # terminal caveat
    results["termini"] = {
        "terminal_residues_primary": int(prim.terminal.sum()),
        "share_noe_lt_0.5_terminal": float((prim[prim.terminal].val < 0.5).mean()),
        "share_noe_lt_0.5_interior": float((prim[~prim.terminal].val < 0.5).mean()),
        "share_G_gt_0.4_terminal": float((prim[prim.terminal].G > 0.4).mean()),
        "share_G_gt_0.4_interior": float((prim[~prim.terminal].G > 0.4).mean()),
    }
    # conditions for primary
    results["conditions_primary"] = {
        "pH_median": float(prim.drop_duplicates("chain").pH.median()),
        "T_median_K": float(prim.drop_duplicates("chain")["T"].median()),
        "field_median_MHz": float(prim.drop_duplicates("chain").field.median()),
    }
    # 2x2 at 0.40 and 0.65 vs cut 0.5/0.65 (counts)
    tab = []
    for cut in NOE_CUTS:
        for t in T_REFS:
            y, p = prim.val < cut, t < prim.G
            tab.append(
                {
                    "noe_cut": cut,
                    "t": t,
                    "TP": int((y & p).sum()),
                    "FP": int((~y & p).sum()),
                    "FN": int((y & ~p).sum()),
                    "TN": int((~y & ~p).sum()),
                }
            )
    pd.DataFrame(tab).to_csv(OUT / "confusion_primary.csv", index=False)
    # --- S2 secondary
    s2 = prep("s2")
    s2p = pick_one_list(s2[(s2.source == "same_entry") & (s2.status == "ok")])
    s2p.to_parquet(OUT / "s2_residues_primary.parquet")
    run_set("s2_same_entry", s2p, S2_CUTS, rng, results, curves)
    s2x = pd.concat(
        [
            s2p,
            pick_one_list(s2[(s2.source == "cross_entry") & ~s2.chain.isin(s2p.chain)]),
        ]
    )
    run_set("s2_same_plus_cross_entry", s2x, S2_CUTS, rng, results, curves)
    # hetNOE vs S2 on shared residues (sanity)
    j = prim.merge(s2p, on=["chain", "idx0"], suffixes=("_noe", "_s2"))
    results["noe_vs_s2_shared"] = {
        "residues": len(j),
        "chains": int(j.chain.nunique()),
        "spearman": float(spearmanr(j.val_noe, j.val_s2).statistic)
        if len(j) > 10
        else None,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))

    # --- summary CSV
    rows = []
    for nm, r in results.items():
        if not isinstance(r, dict) or "cuts" not in r:
            continue
        for cut, c in r["cuts"].items():
            if "t_star_MCC" not in c:
                continue
            rf = r["reference_free"].get(cut, {})
            rows.append(
                {
                    "set": nm,
                    "cut": cut,
                    "chains": c["chains"],
                    "clusters": c["clusters"],
                    "residues": c["residues"],
                    "prevalence": round(c["prevalence"], 3),
                    "spearman": round(r["spearman_pooled"], 3),
                    "spearman_CI": "[%.3f, %.3f]" % tuple(r["spearman_pooled_CI"]),
                    "AUC": round(c["ROC_AUC"], 3),
                    "t_star": c["t_star_MCC"],
                    "t_star_CI": "[%.2f, %.2f]" % tuple(c["t_star_MCC_CI"]),
                    "flat": "%.2f-%.2f" % tuple(c["flat_MCC_within_0.01"]),
                    "MCC_t_star": round(c["MCC_t_star"], 3),
                    "MCC_040": round(c["MCC_040"], 3),
                    "MCC_040_CI": "[%.3f, %.3f]" % tuple(c["MCC_040_CI"]),
                    "MCC_065": round(c["MCC_065"], 3),
                    "MCC_065_CI": "[%.3f, %.3f]" % tuple(c["MCC_065_CI"]),
                    "dMCC_040_065": round(c["dMCC_040_minus_065"], 3),
                    "dMCC_CI": "[%.3f, %.3f]" % tuple(c["dMCC_040_minus_065_CI"]),
                    "boot_tstar_ge_065": c["share_boot_tstar_ge_065"],
                    "iso_cross": rf.get("isotonic_mean_crossing"),
                    "iso_CI": "[%.3f, %.3f]" % tuple(rf["isotonic_CI"]) if rf else "",
                    "median_cross": rf.get("binned_median_crossing"),
                    "median_CI": "[%.3f, %.3f]" % tuple(rf["binned_median_CI"])
                    if rf
                    else "",
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string())

    # --- figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
        }
    )
    GR = "#52514e"
    COL = {0.5: "#2a78d6", 0.6: "#1baf7a", 0.65: "#eb6834"}
    fig, ax = plt.subplots(
        1, 2, figsize=(180 / 25.4, 75 / 25.4), constrained_layout=True
    )
    a = ax[0]
    hb = a.hexbin(
        prim.G,
        prim.val,
        gridsize=40,
        extent=(0, 1, -1.0, 1.05),
        bins="log",
        cmap="Greys",
        mincnt=1,
        lw=0,
    )
    p0, xc, med = fits[0.5]
    a.plot(GRID_G, p0, color="#d62a5c", lw=1.3, label="isotonic fit (mean)")
    a.plot(
        xc,
        med,
        "o",
        ms=2.5,
        color="#6250d6",
        label="binned median (20 G-quantile bins)",
    )
    for cut in NOE_CUTS:
        a.axhline(cut, color=COL[cut], lw=0.7, ls=":")
    for t in T_REFS:
        a.axvline(t, color=GR, lw=0.8, ls="--")
    r0 = results["noe_same_entry"]
    a.set_xlabel("G")
    a.set_ylabel("{1H}-15N hetNOE")
    a.set_ylim(-1.0, 1.05)
    a.legend(frameon=False, loc="lower left", fontsize=6)
    fig.colorbar(hb, ax=a, label="residues", shrink=0.8)
    a.set_title(
        f"a  same-entry hetNOE: {r0['chains']} chains, {r0['residues']:,} residues; "
        f"Spearman {r0['spearman_pooled']:.2f}",
        loc="left",
        fontsize=7,
    )
    a = ax[1]
    for cut in NOE_CUTS:
        cv = curves[("noe_same_entry", cut)]
        c = r0["cuts"][str(cut)]
        a.fill_between(cv.t, cv.MCC_lo, cv.MCC_hi, color=COL[cut], alpha=0.18, lw=0)
        a.plot(
            cv.t,
            cv.MCC,
            color=COL[cut],
            lw=1.2,
            label=f"hetNOE < {cut}: t*={c['t_star_MCC']:.2f} [{c['t_star_MCC_CI'][0]:.2f}, {c['t_star_MCC_CI'][1]:.2f}]",
        )
    for t in T_REFS:
        a.axvline(t, color=GR, lw=0.8, ls="--")
    a.set_xlim(0, 1)
    a.set_xlabel("threshold t (disordered if G > t)")
    a.set_ylabel("MCC")
    a.legend(frameon=False, loc="lower center", fontsize=6)
    a.set_title(
        "b  MCC vs t per hetNOE cut (95% cluster-bootstrap bands)",
        loc="left",
        fontsize=7,
    )
    fig.text(
        0.5,
        -0.02,
        "dashed: t = 0.40 and 0.65; dotted (a): hetNOE cuts 0.5 / 0.6 / 0.65",
        ha="center",
        fontsize=6,
        color=GR,
    )
    fig.savefig(OUT / "fig_threshold_hetnoe.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_threshold_hetnoe.png", dpi=300, bbox_inches="tight")
    print(
        json.dumps(
            {
                k: v
                for k, v in results.items()
                if k in ("field", "termini", "noe_vs_s2_shared", "conditions_primary")
            },
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
