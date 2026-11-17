"""Steps 5-9 of the G-threshold calibration: threshold curves, cluster
bootstrap, strata, internal estimates (valley, logit-GMM, replicate kappa),
sensitivity table and the candidate supplementary figure.

Inputs: out/residues.parquet, out/chains_filtered.parquet (01_map_and_label.py)
and the release Parquet. Outputs: out/*.csv, out/results.json, out/fig_*.{pdf,png}.
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
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path(f"{ROOT}/data/interim/threshold-calibration-2026-11-16")
OUT = BASE / "out"
PARQUET = Path(
    _os.environ.get("TRIZOD_RELEASE_PARQUET")
    or f"{ROOT}/data/interim/rebuild-2026-11-11/release/trizod_dataset.parquet"
)
NBOOT = 1000
SEED = 20261116
T_FINE = np.arange(1, 1000) / 1000  # 0.001..0.999
GRID_J = np.arange(10, 1000, 10)  # indices into T_FINE+1 -> t = 0.01..0.99
T = GRID_J / 1000
T_REF = 0.65
J_REF = 650
SENS_T = np.round(np.arange(0.40, 0.801, 0.05), 2)
TEST_SPLITS = ("test_trizod", "test_chezod117")
K_STRATA = {"k0-9": (0, 9), "k10-15": (10, 15), "k16-21": (16, 21)}


def fine_bin(g):
    """Index b in 0..999 such that G > t_j  <=>  b >= j (t_j = j/1000)."""
    gi = np.rint(np.asarray(g, float) * 10000).astype(int)
    return np.clip(-(-gi // 10) - 1, 0, 999)


def cluster_counts(df, cluster_col):
    """Per cluster: counts of positives / negatives per fine bin."""
    cl, cl_idx = np.unique(df[cluster_col].to_numpy(), return_inverse=True)
    b = fine_bin(df.G.to_numpy())
    y = df.y.to_numpy()
    pos = np.zeros((len(cl), 1000))
    neg = np.zeros((len(cl), 1000))
    np.add.at(pos, (cl_idx[y == 1], b[y == 1]), 1)
    np.add.at(neg, (cl_idx[y == 0], b[y == 0]), 1)
    return pos, neg


def boot_weights(n, rng, nboot=NBOOT):
    W = np.zeros((nboot, n))
    for i in range(nboot):
        W[i] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W


def confusion(pos_hist, neg_hist):
    """hist: (..., 1000) counts per fine bin -> TP/FP/FN/TN at t_j, j=0..999
    where index j means threshold j/1000 (index 0 = t 0 -> everything > 0 except G=0)."""
    tp = np.flip(np.cumsum(np.flip(pos_hist, -1), -1), -1)  # sum_{b>=j}
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
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {
        "MCC": np.nan_to_num(mcc),
        "J": tpr + tnr - 1,
        "F1": f1,
        "BA": (tpr + tnr) / 2,
    }


def binned_auc(pos_hist, neg_hist):
    tp, fp, fn, tn = confusion(pos_hist, neg_hist)
    P = (tp + fn)[..., :1]
    N = (fp + tn)[..., :1]
    tpr = np.concatenate([tp / P, np.zeros_like(P)], -1)
    fpr = np.concatenate([fp / N, np.zeros_like(N)], -1)
    roc = -np.trapezoid(tpr, fpr, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
    rec = tp / P
    ap = (
        np.sum((rec[..., :-1] - rec[..., 1:]) * prec[..., :-1], -1)
        + rec[..., -1] * prec[..., -1]
    )
    return roc, ap


def ci(x):
    x = np.asarray(x, float)
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def calibrate(df, cluster_col, rng, fixed_t=None):
    """Point estimates + cluster bootstrap. fixed_t: if given (a train t*),
    also report MCC at that t (out-of-sample comparison)."""
    pos, neg = cluster_counts(df, cluster_col)
    tp, fp, fn, tn = confusion(pos.sum(0), neg.sum(0))
    m = {k: v[GRID_J] for k, v in metrics(tp, fp, fn, tn).items()}
    imax = int(np.argmax(m["MCC"]))
    jmax = int(np.argmax(m["J"]))
    t_star = float(T[imax])
    flat = T[m["MCC"] >= m["MCC"].max() - 0.01]
    W = boot_weights(len(pos), rng)
    bpos, bneg = W @ pos, W @ neg
    btp, bfp, bfn, btn = confusion(bpos, bneg)
    bm = {k: v[:, GRID_J] for k, v in metrics(btp, bfp, bfn, btn).items()}
    b_imax = bm["MCC"].argmax(1)
    b_tstar = T[b_imax]
    b_mcc_star = bm["MCC"][np.arange(NBOOT), b_imax]
    iref = int(np.where(np.isclose(T, T_REF))[0][0])
    b_mcc_ref = bm["MCC"][:, iref]
    b_tJ = T[bm["J"].argmax(1)]
    roc_b, ap_b = binned_auc(bpos, bneg)
    roc = roc_auc_score(df.y, df.G)
    ap = average_precision_score(df.y, df.G)
    res = {
        "chains": int(df.id.nunique()),
        "clusters": int(len(pos)),
        "residues": int(len(df)),
        "positives": int(df.y.sum()),
        "negatives": int((df.y == 0).sum()),
        "prevalence": float(df.y.mean()),
        "ROC_AUC": float(roc),
        "ROC_AUC_CI": ci(roc_b),
        "PR_AUC": float(ap),
        "PR_AUC_CI": ci(ap_b),
        "t_star_MCC": t_star,
        "t_star_MCC_CI": ci(b_tstar),
        "MCC_t_star": float(m["MCC"][imax]),
        "MCC_t_star_CI": ci(b_mcc_star),
        "MCC_065": float(m["MCC"][iref]),
        "MCC_065_CI": ci(b_mcc_ref),
        "dMCC_tstar_minus_065": float(m["MCC"][imax] - m["MCC"][iref]),
        "dMCC_CI": ci(b_mcc_star - b_mcc_ref),
        "t_star_J": float(T[jmax]),
        "t_star_J_CI": ci(b_tJ),
        "J_t_star": float(m["J"][jmax]),
        "J_065": float(m["J"][iref]),
        "flat_MCC_within_0.01": [float(flat.min()), float(flat.max())],
        "flat_contiguous": bool(
            len(flat) == round((flat.max() - flat.min()) / 0.01) + 1
        ),
        "share_boot_tstar_ge_065": float(np.mean(b_tstar >= T_REF - 1e-9)),
    }
    if fixed_t is not None:
        ifix = int(np.where(np.isclose(T, fixed_t))[0][0])
        res["MCC_at_train_t_star"] = float(m["MCC"][ifix])
        res["MCC_at_train_t_star_CI"] = ci(bm["MCC"][:, ifix])
        res["dMCC_train_tstar_minus_065"] = float(m["MCC"][ifix] - m["MCC"][iref])
        res["dMCC_train_tstar_minus_065_CI"] = ci(bm["MCC"][:, ifix] - b_mcc_ref)
    curves = pd.DataFrame({"t": T, **m})
    for k in ("MCC", "J", "F1", "BA"):
        curves[f"{k}_lo"] = np.nanpercentile(bm[k], 2.5, 0)
        curves[f"{k}_hi"] = np.nanpercentile(bm[k], 97.5, 0)
    return res, curves


# ---------------------------------------------------------- internal estimates
def chain_hists(gs, edges):
    H = np.zeros((len(gs), len(edges) - 1))
    for i, g in enumerate(gs):
        H[i] = np.histogram(g, edges)[0]
    return H


def valley(h, centers, sd=0.03):
    w = centers[1] - centers[0]
    x = np.arange(-4 * sd, 4 * sd + w / 2, w)
    ker = np.exp(-0.5 * (x / sd) ** 2)
    sm = np.convolve(h, ker / ker.sum(), "same")
    sel = (centers > 0.2) & (centers < 0.85)
    return float(centers[sel][np.argmin(sm[sel])]), sm


def gmm_binned(h, x, iters=300):
    """Weighted EM for a 2-component 1-D Gaussian mixture on binned data;
    returns the G where the high component's posterior is 0.5 (between means)."""
    w = h / h.sum()
    mu = np.array([-3.0, 2.0])
    sd = np.array([1.0, 1.0])
    pi = np.array([0.6, 0.4])
    for _ in range(iters):
        p = (
            pi[:, None]
            * np.exp(-0.5 * ((x - mu[:, None]) / sd[:, None]) ** 2)
            / sd[:, None]
        )
        r = p / np.maximum(p.sum(0), 1e-300)
        nk = (r * w).sum(1)
        pi = nk
        mu = (r * w * x).sum(1) / nk
        sd = np.sqrt((r * w * (x - mu[:, None]) ** 2).sum(1) / nk)
    xs = np.linspace(mu.min(), mu.max(), 4001)
    p = (
        pi[:, None]
        * np.exp(-0.5 * ((xs - mu[:, None]) / sd[:, None]) ** 2)
        / sd[:, None]
    )
    hi = int(np.argmax(mu))
    post = p[hi] / p.sum(0)
    xc = xs[np.argmin(np.abs(post - 0.5))]
    return float(1 / (1 + np.exp(-xc))), (pi, mu, sd)


def replicate_pairs(ch, rng, same_cond):
    """Pairs of filtered chains with identical sequence from different entries
    (one chain per entry, at most 20 random pairs per sequence)."""
    groups = []
    for seq, grp in ch.groupby("sequence"):
        rows = grp.drop_duplicates("entry_id").to_dict("records")
        if len(rows) < 2:
            continue
        pairs = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows))]
        if same_cond:
            pairs = [
                (i, j)
                for i, j in pairs
                if np.isfinite(rows[i]["ph"])
                and np.isfinite(rows[j]["ph"])
                and np.isfinite(rows[i]["temperature"])
                and np.isfinite(rows[j]["temperature"])
                and abs(rows[i]["ph"] - rows[j]["ph"]) <= 0.2
                and abs(rows[i]["temperature"] - rows[j]["temperature"]) <= 2
            ]
        if not pairs:
            continue
        if len(pairs) > 20:
            pairs = [pairs[x] for x in rng.choice(len(pairs), 20, replace=False)]
        A, B = [], []
        for i, j in pairs:
            gi = np.asarray(rows[i]["gscores"], float)
            gj = np.asarray(rows[j]["gscores"], float)
            m = (
                np.asarray(rows[i]["mask"], bool)
                & np.asarray(rows[j]["mask"], bool)
                & np.isfinite(gi)
                & np.isfinite(gj)
            )
            A.append(gi[m])
            B.append(gj[m])
        A, B = np.concatenate(A), np.concatenate(B)
        if len(A):
            groups.append((A, B, len(pairs)))
    return groups


def kappa_counts(groups):
    """Per group, arrays over the fine grid: N, nA (A>t), nB, n11."""
    C = np.zeros((len(groups), 4, 1000))
    for gi, (A, B, _) in enumerate(groups):
        ba, bb = fine_bin(A), fine_bin(B)
        bmin = np.minimum(ba, bb)
        for r, b in ((1, ba), (2, bb), (3, bmin)):
            C[gi, r] = np.flip(np.cumsum(np.flip(np.bincount(b, minlength=1000))))
        C[gi, 0] = len(A)
    return C


def kappa(C):
    N, nA, nB, n11 = C[..., 0, :], C[..., 1, :], C[..., 2, :], C[..., 3, :]
    n00 = N - nA - nB + n11
    po = (n11 + n00) / N
    pa, pb = nA / N, nB / N
    pe = pa * pb + (1 - pa) * (1 - pb)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (po - pe) / (1 - pe)


def main():
    rng = np.random.default_rng(SEED)
    res = pd.read_parquet(OUT / "residues.parquet")
    res["G"] = res.G.astype(float)
    res["clu_train"] = res.cluster_repr.fillna(res.id)
    res["clu_chain"] = res.id
    results = {
        "settings": {
            "n_boot": NBOOT,
            "seed": SEED,
            "grid": "0.01..0.99",
            "rule": "disordered iff G > t",
        }
    }
    curves_out = {}

    # ---- 5. main calibration per set
    for s in ("A", "B"):
        r = res[res.set == s]
        tr = r[r.split == "train"]
        te = r[r.split.isin(TEST_SPLITS)]
        rtr, ctr = calibrate(tr, "clu_train", rng)
        rte, cte = calibrate(te, "clu_chain", rng, fixed_t=rtr["t_star_MCC"])
        # pool: every filtered chain; cluster = mapped UniProt accession
        rpo, cpo = calibrate(r, "acc", rng, fixed_t=rtr["t_star_MCC"])
        results[f"set{s}"] = {"train": rtr, "test": rte, "pool": rpo}
        for nm, cv in (("train", ctr), ("test", cte), ("pool", cpo)):
            cv.to_csv(OUT / f"curve_set{s}_{nm}.csv", index=False, float_format="%.5f")
            curves_out[(s, nm)] = cv
        # ---- 6. strata by k (train)
        st = {}
        for name, (lo, hi) in K_STRATA.items():
            sub = tr[(tr.k >= lo) & (tr.k <= hi)]
            if sub.y.sum() >= 30 and (sub.y == 0).sum() >= 30:
                st[name], cv = calibrate(sub, "clu_train", rng)
                cv.to_csv(
                    OUT / f"curve_set{s}_train_{name}.csv",
                    index=False,
                    float_format="%.5f",
                )
            else:
                st[name] = {
                    "residues": len(sub),
                    "positives": int(sub.y.sum()),
                    "note": "too few",
                }
        results[f"set{s}"]["train_by_k"] = st
        print(s, json.dumps(results[f"set{s}"]["train"], indent=0)[:1500], flush=True)
    # union A+B on train (a residue may carry both labels: keep set A's label)
    u = res[res.split == "train"].sort_values("set").drop_duplicates(["id", "idx"])
    results["setAB_union_train"], cu = calibrate(u, "clu_train", rng)
    cu.to_csv(OUT / "curve_setAB_union_train.csv", index=False, float_format="%.5f")
    both = res[res.split == "train"].pivot_table(
        index=["id", "idx"], columns="set", values="y"
    )
    both = both.dropna()
    results["A_B_overlap_train"] = {
        "residues_in_both": int(len(both)),
        "agreement": float((both.A == both.B).mean()) if len(both) else None,
        "crosstab": {
            f"A{a}_B{b}": int(((a == both.A) & (b == both.B)).sum())
            for a in (0, 1)
            for b in (0, 1)
        },
    }

    # ---- 7. internal estimates on filtered train chains
    ch = pd.read_parquet(OUT / "chains_filtered.parquet")
    rel = pd.read_parquet(PARQUET, columns=["id", "gscores", "mask"])
    ch = ch.merge(rel, on="id")
    gs_all = []
    for g, m in zip(ch.gscores, ch["mask"]):
        g = np.asarray(g, float)[np.asarray(m, bool)]
        gs_all.append(g[np.isfinite(g)])
    ch["G_scored"] = gs_all
    trn = ch[ch.split == "train"].reset_index(drop=True)
    edges = np.linspace(0, 1, 201)
    cen = (edges[:-1] + edges[1:]) / 2
    Hg = chain_hists(trn.G_scored, edges)
    ledges = np.linspace(-7, 7, 281)
    lcen = (ledges[:-1] + ledges[1:]) / 2
    Hl = chain_hists(
        [
            np.log(np.clip(g, 1e-3, 1 - 1e-3) / (1 - np.clip(g, 1e-3, 1 - 1e-3)))
            for g in trn.G_scored
        ],
        ledges,
    )
    v0, sm0 = valley(Hg.sum(0), cen)
    m0, (pi, mu, sd) = gmm_binned(Hl.sum(0), lcen)
    W = boot_weights(len(trn), rng)
    bv = np.array([valley(w @ Hg, cen)[0] for w in W])
    bg = np.array([gmm_binned(w @ Hl, lcen, iters=150)[0] for w in W])
    allG = np.concatenate(trn.G_scored.to_list())
    share = np.array([(allG > t).mean() for t in T])
    Hf = np.zeros((len(trn), 1000))
    for i, g in enumerate(trn.G_scored):
        Hf[i] = np.bincount(fine_bin(g), minlength=1000)
    cumf = np.flip(np.cumsum(np.flip(Hf, 1), 1), 1)[:, GRID_J]
    bshare = (W @ cumf) / (W @ Hf.sum(1))[:, None]
    results["internal"] = {
        "train_chains": len(trn),
        "train_residues": int(len(allG)),
        "valley": v0,
        "valley_CI": ci(bv),
        "gmm_logit_crossing": m0,
        "gmm_logit_crossing_CI": ci(bg),
        "gmm_params": {
            "weights": pi.tolist(),
            "means_logit": mu.tolist(),
            "sd_logit": sd.tolist(),
        },
    }
    pd.DataFrame(
        {
            "t": T,
            "share_disordered": share,
            "lo": np.percentile(bshare, 2.5, 0),
            "hi": np.percentile(bshare, 97.5, 0),
        }
    ).to_csv(OUT / "disordered_share_train.csv", index=False, float_format="%.5f")
    pd.DataFrame({"G_bin_center": cen, "count": Hg.sum(0), "smoothed": sm0}).to_csv(
        OUT / "g_histogram_train.csv", index=False, float_format="%.5f"
    )

    # replicate kappa
    kap_curves = {}
    for nm, same in (("all_pairs", False), ("same_condition", True)):
        groups = replicate_pairs(ch, np.random.default_rng(0), same)
        C = kappa_counts(groups)
        k0 = kappa(C.sum(0))[GRID_J]
        Wk = boot_weights(len(groups), rng)
        bk = kappa(np.einsum("bg,gri->bri", Wk, C))[:, GRID_J]
        btk = T[np.nanargmax(bk, 1)]
        i_ref = int(np.where(np.isclose(T, T_REF))[0][0])
        imax = int(np.nanargmax(k0))
        flat = T[k0 >= np.nanmax(k0) - 0.01]
        results[f"replicate_kappa_{nm}"] = {
            "sequence_groups": len(groups),
            "pairs": int(sum(g[2] for g in groups)),
            "residue_pairs": int(sum(len(g[0]) for g in groups)),
            "t_star": float(T[imax]),
            "t_star_CI": ci(btk),
            "kappa_max": float(k0[imax]),
            "kappa_max_CI": ci(np.nanmax(bk, 1)),
            "kappa_065": float(k0[i_ref]),
            "kappa_065_CI": ci(bk[:, i_ref]),
            "dkappa_CI": ci(np.nanmax(bk, 1) - bk[:, i_ref]),
            "flat_within_0.01": [float(flat.min()), float(flat.max())],
        }
        kc = pd.DataFrame(
            {
                "t": T,
                "kappa": k0,
                "lo": np.nanpercentile(bk, 2.5, 0),
                "hi": np.nanpercentile(bk, 97.5, 0),
            }
        )
        kc.to_csv(OUT / f"replicate_kappa_{nm}.csv", index=False, float_format="%.5f")
        kap_curves[nm] = kc

    # ---- 8. sensitivity
    rows = []
    for t in SENS_T:
        i = int(np.where(np.isclose(T, t))[0][0])
        row = {"t": t, "share_disordered_train": share[i]}
        for s in ("A", "B"):
            for nm in ("train", "test"):
                cv = curves_out[(s, nm)]
                row[f"MCC_set{s}_{nm}"] = cv.MCC[i]
        for nm in kap_curves:
            row[f"kappa_{nm}"] = kap_curves[nm].kappa[i]
        rows.append(row)
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / "sensitivity.csv", index=False, float_format="%.4f")
    results["sensitivity"] = sens.round(4).to_dict("records")
    (OUT / "results.json").write_text(json.dumps(results, indent=1))

    # ---- 9. figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
        }
    )
    CA, CB, CK, GR = "#2a78d6", "#eb6834", "#6250d6", "#52514e"
    tA = results["setA"]["train"]["t_star_MCC"]
    tB = results["setB"]["train"]["t_star_MCC"]
    fig, ax = plt.subplots(
        2, 2, figsize=(180 / 25.4, 120 / 25.4), constrained_layout=True
    )

    def refs(a, tstars):
        a.axvline(T_REF, color=GR, ls="--", lw=0.8)
        for t, c in tstars:
            a.axvline(t, color=c, ls=":", lw=0.9)

    a = ax[0, 0]
    a.bar(
        cen,
        Hg.sum(0) / Hg.sum() / (cen[1] - cen[0]),
        width=cen[1] - cen[0],
        color="#c3c2b7",
        lw=0,
    )
    a.plot(cen, sm0 / Hg.sum() / (cen[1] - cen[0]), color="#0b0b0b", lw=1)
    iv = results["internal"]
    for val, lab, c, ls in (
        (iv["valley"], "valley", "#0b0b0b", "-."),
        (iv["gmm_logit_crossing"], "mixture", CK, "-."),
    ):
        a.axvline(val, color=c, ls=ls, lw=0.9)
    a.axvline(T_REF, color=GR, ls="--", lw=0.8)
    a.set_yscale("log")
    a.set_xlabel("G")
    a.set_ylabel("density (train residues)")
    a.set_title(
        f"a  G distribution; valley {iv['valley']:.2f}, mixture {iv['gmm_logit_crossing']:.2f}",
        loc="left",
        fontsize=7,
    )

    a = ax[0, 1]
    for s, c in (("A", CA), ("B", CB)):
        cv = curves_out[(s, "train")]
        a.fill_between(cv.t, cv.MCC_lo, cv.MCC_hi, color=c, alpha=0.2, lw=0)
        r_ = results[f"set{s}"]["train"]
        a.plot(
            cv.t,
            cv.MCC,
            color=c,
            lw=1.2,
            label=f"set {s}: t*={r_['t_star_MCC']:.2f} [{r_['t_star_MCC_CI'][0]:.2f}, {r_['t_star_MCC_CI'][1]:.2f}]",
        )
    refs(a, [(tA, CA), (tB, CB)])
    a.set_xlabel("threshold t")
    a.set_ylabel("MCC (train)")
    a.legend(frameon=False, loc="lower center")
    a.set_title(
        "b  External calibration (DisProt-PDB = A, PDB-missing = B)",
        loc="left",
        fontsize=7,
    )

    a = ax[1, 0]
    for nm, c, ls in (("all_pairs", CK, "-"), ("same_condition", "#1baf7a", "-")):
        kc = kap_curves[nm]
        rk = results[f"replicate_kappa_{nm}"]
        a.fill_between(kc.t, kc.lo, kc.hi, color=c, alpha=0.2, lw=0)
        a.plot(
            kc.t,
            kc.kappa,
            color=c,
            lw=1.2,
            ls=ls,
            label=f"{nm.replace('_', ' ')}: t*={rk['t_star']:.2f} [{rk['t_star_CI'][0]:.2f}, {rk['t_star_CI'][1]:.2f}]",
        )
    refs(a, [(tA, CA), (tB, CB)])
    a.set_xlabel("threshold t")
    a.set_ylabel("Cohen's kappa (replicates)")
    a.legend(frameon=False, loc="lower center")
    a.set_title("c  Replicate agreement", loc="left", fontsize=7)

    a = ax[1, 1]
    a.fill_between(
        T,
        np.percentile(bshare, 2.5, 0),
        np.percentile(bshare, 97.5, 0),
        color=GR,
        alpha=0.25,
        lw=0,
    )
    a.plot(T, share, color="#0b0b0b", lw=1.2)
    refs(a, [(tA, CA), (tB, CB)])
    a.set_xlabel("threshold t")
    a.set_ylabel("share of residues with G > t (train)")
    a.set_title("d  Disordered share", loc="left", fontsize=7)
    for a in ax.ravel():
        a.set_xlim(0, 1)
    fig.text(
        0.5,
        -0.01,
        "dashed grey: t = 0.65; dotted: t* (MCC) for set A (blue) and set B (orange)",
        ha="center",
        fontsize=6,
        color=GR,
    )
    fig.savefig(OUT / "fig_threshold_calibration.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_threshold_calibration.png", dpi=300, bbox_inches="tight")
    print(
        json.dumps({k: v for k, v in results.items() if k != "sensitivity"}, indent=1)[
            :6000
        ]
    )


if __name__ == "__main__":
    main()
