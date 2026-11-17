"""Estimators of the TriZOD G-score threshold framework (numpy only).

Ported from the research code of 2026-11-16:
  threshold-framework-2026-11-16/scripts/01_framework.py   isotonic step LR, geometric pool, prior-shift
                                                           EM, Rogan-Gladen, logit mixture, thresholds,
                                                           replicate validation, held-out check
  threshold-soundness-2026-11-16/review/r03_bootstrap.py   log LR interpolated between block midpoints
                                                           (the primary estimator of the soundness check)

The isotonic fit is a weighted pool-adjacent-violators that reproduces
sklearn.isotonic.IsotonicRegression(increasing=True, y_min=0, y_max=1, out_of_bounds="clip") on the
0.01 bins (fit on the occupied bins, linear interpolation across empty ones), so no sklearn is needed.
"""

from __future__ import annotations

import numpy as np

NB = 100  # 0.01 bins; bin j covers [j/100, (j+1)/100)
EDGES = np.arange(NB) / NB
XC = (np.arange(NB) + 0.5) / NB
NF = 1000  # 0.001 cells of the interpolated estimator
FINE_LO = np.arange(NF) / NF
FINE = FINE_LO + 0.5 / NF
LX_EDGES = np.arange(-7.0, 7.0001, 0.05)  # logit-G histogram of the mixture prior
LX = (LX_EDGES[:-1] + LX_EDGES[1:]) / 2
ZONE_NAMES = ("ordered", "indeterminate", "disordered")


def gbin(g) -> np.ndarray:
    return np.clip(np.floor(np.asarray(g, float) * NB + 1e-9).astype(int), 0, NB - 1)


def ci(x, digits: int = 6) -> list:
    """95 % percentile interval of the finite values (None if fewer than 10)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return [None, None]
    return [
        round(float(np.percentile(x[ok], 2.5)), digits),
        round(float(np.percentile(x[ok], 97.5)), digits),
    ]


def pt_ci(x, digits: int = 6) -> list:
    """[point (row 0), CI low, CI high] of a (nboot + 1,) array whose row 0 is the full-data estimate."""
    x = np.asarray(x, float)
    p = None if not np.isfinite(x[0]) else round(float(x[0]), digits)
    return [p, *ci(x[1:], digits)]


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.asarray(p, float)
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------------- isotonic LR
def pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted increasing isotonic regression (pool adjacent violators)."""
    n = len(y)
    sw = np.empty(n)
    swy = np.empty(n)
    ln = np.empty(n, dtype=np.int64)
    k = 0
    for i in range(n):
        sw[k], swy[k], ln[k] = w[i], w[i] * y[i], 1
        while k > 0 and swy[k - 1] / sw[k - 1] >= swy[k] / sw[k]:
            sw[k - 1] += sw[k]
            swy[k - 1] += swy[k]
            ln[k - 1] += ln[k]
            k -= 1
        k += 1
    return np.repeat(swy[:k] / sw[:k], ln[:k])


def iso_loglr(hp: np.ndarray, hn: np.ndarray) -> np.ndarray:
    """Isotonic P(pos | G) on the 0.01 bins -> block-wise log LR (Haldane 0.5), running maximum.

    Within a reference this is the Bayes identity block odds / reference prior odds; the blocks are
    the runs of equal fitted values (= the segments of the ROC convex hull).
    """
    w = hp + hn
    ok = w > 0
    P, N = hp.sum(), hn.sum()
    if ok.sum() < 3 or P == 0 or N == 0:
        return np.full(NB, np.nan)
    fit = np.clip(pav(hp[ok] / w[ok], w[ok]), 0.0, 1.0)
    p = np.interp(XC, XC[ok], fit)
    brk = np.flatnonzero(np.abs(np.diff(p)) > 1e-12) + 1
    cp, cn = np.r_[0.0, np.cumsum(hp)], np.r_[0.0, np.cumsum(hn)]
    out = np.empty(NB)
    for s, e in zip(np.r_[0, brk], np.r_[brk, NB]):
        out[s:e] = np.log((cp[e] - cp[s] + 0.5) / P) - np.log((cn[e] - cn[s] + 0.5) / N)
    return np.maximum.accumulate(out)


def interp_loglr(step: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Log LR interpolated linearly between the midpoints of the isotonic blocks (constant beyond the
    first and the last midpoint): the smoothed estimator of the soundness check (review r03)."""
    if not np.isfinite(step).all():
        return np.full(len(x), np.nan)
    brk = np.r_[0, np.flatnonzero(np.abs(np.diff(step)) > 1e-12) + 1, NB]
    mids = (XC[brk[:-1]] + XC[brk[1:] - 1]) / 2
    return np.interp(x, mids, step[brk[:-1]])


# ----------------------------------------------------------------------------- joint cluster bootstrap
def boot_weights(n: int, nboot: int, rng: np.random.Generator) -> np.ndarray:
    """(nboot + 1, n) cluster multiplicities; row 0 = the full data."""
    rows = [np.ones(n)]
    rows += [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(nboot)]
    return np.vstack(rows).astype(float)


class Boot:
    """One joint cluster bootstrap over references and the prior histogram.

    comps: name -> (keys, pos, neg) for each reference (pos/neg: clusters x NB label counts) and
    "PI" -> (keys, HG, HL) (clusters x NB G histogram and clusters x len(LX) logit histogram of the
    fit chains). Every replicate resamples the union of the cluster keys of all components together,
    so clusters shared between references (A and B share residues) move together.
    """

    def __init__(
        self, comps: dict, nboot: int, seed: int, cuts=(), curves: bool = True
    ):
        self.comps, self.nboot = comps, nboot
        keys = sorted(set().union(*[set(c[0]) for c in comps.values()]))
        pos_of = {k: i for i, k in enumerate(keys)}
        self.n_union_clusters = len(keys)
        Wg = boot_weights(len(keys), nboot, np.random.default_rng(seed))
        self.W = {nm: Wg[:, [pos_of[k] for k in c[0]]] for nm, c in comps.items()}
        self.refs = [nm for nm in comps if nm != "PI"]
        self.Hp = {nm: self.W[nm] @ comps[nm][1] for nm in self.refs}
        self.Hn = {nm: self.W[nm] @ comps[nm][2] for nm in self.refs}
        HG = self.W["PI"] @ comps["PI"][1]
        self.HG = HG
        self.hG = HG / HG.sum(1, keepdims=True)
        self.HL = self.W["PI"] @ comps["PI"][2]
        self.cuts = np.asarray(cuts, float)
        x = np.r_[XC, FINE, self.cuts]
        self.L, self.Lbin_i, self.Lfine, self.Lcut = {}, {}, {}, {}
        for nm in self.refs if curves else ():
            st = np.vstack(
                [iso_loglr(self.Hp[nm][b], self.Hn[nm][b]) for b in range(nboot + 1)]
            )
            it = np.vstack([interp_loglr(s, x) for s in st])
            self.L[nm] = st
            self.Lbin_i[nm] = it[:, :NB]
            self.Lfine[nm] = it[:, NB : NB + NF]
            self.Lcut[nm] = it[:, NB + NF :]

    def pooled(self, weights: dict, est: str):
        """-> (log LR on the bins (for the prior), log LR on the threshold grid, grid lower edges,
        log LR at the cuts) of the log-linear pool sum_r w_r log LR_r."""
        if est == "step":
            Lb = sum(w * self.L[r] for r, w in weights.items())
            jc = gbin(self.cuts) if len(self.cuts) else np.array([], int)
            return Lb, Lb, EDGES, Lb[:, jc]
        Lb = sum(w * self.Lbin_i[r] for r, w in weights.items())
        Lf = sum(w * self.Lfine[r] for r, w in weights.items())
        Lc = sum(w * self.Lcut[r] for r, w in weights.items())
        return Lb, Lf, FINE_LO, Lc


# ----------------------------------------------------------------------------- prevalence
def em_prior(
    loglr: np.ndarray,
    h: np.ndarray,
    p0: float = 0.2,
    tol: float = 1e-12,
    max_iter: int = 20000,
):
    """Prior-shift MLE (Saerens, Latinne & Decaestecker 2002): pi = mean_i P(dis | G_i; LR, pi) over the
    histogram h, per row. loglr, h: (B, M); returns (B,) (NaN where the LR row is not finite)."""
    loglr = np.atleast_2d(loglr)
    h = np.atleast_2d(h)
    lr = np.exp(loglr)
    p = np.full(loglr.shape[0], p0)
    for _ in range(max_iter):
        pc = p[:, None]
        pn = np.nansum(h * lr * pc / (lr * pc + 1 - pc), 1)
        done = np.nanmax(np.abs(pn - p)) < tol
        p = pn
        if done:
            break
    p[~np.isfinite(loglr).all(1)] = np.nan
    return p


def rogan_gladen(boot: Boot, weights: dict, cut: float) -> np.ndarray:
    """Moment estimator of the same label-shift model: (q - FPR) / (TPR - FPR) at one cut, TPR/FPR
    averaged over the references with the pool weights (a numerical check of the EM, not independent)."""
    j = int(round(cut * NB))
    q = boot.HG[:, j:].sum(1) / boot.HG.sum(1)
    tpr = fpr = 0.0
    for r, w in weights.items():
        tpr = tpr + w * boot.Hp[r][:, j:].sum(1) / boot.Hp[r].sum(1)
        fpr = fpr + w * boot.Hn[r][:, j:].sum(1) / boot.Hn[r].sum(1)
    return np.clip((q - fpr) / (tpr - fpr), 1e-3, 1 - 1e-3)


def mixture_prior(HL: np.ndarray, iters: int = 300) -> np.ndarray:
    """High-component weight of a two-component Gaussian mixture on logit G (the estimator the framework
    first pre-specified; reported only as the disclosed alternative prior)."""
    w = HL / HL.sum(1, keepdims=True)
    B = len(w)
    mu = np.tile([-3.0, 2.0], (B, 1))
    sd = np.ones((B, 2))
    pi = np.tile([0.8, 0.2], (B, 1))
    x = LX[None, None, :]
    for _ in range(iters):
        p = (
            pi[:, :, None]
            * np.exp(-0.5 * ((x - mu[:, :, None]) / sd[:, :, None]) ** 2)
            / sd[:, :, None]
        )
        r = p / np.maximum(p.sum(1, keepdims=True), 1e-300)
        rw = r * w[:, None, :]
        nk = rw.sum(2)
        pi = nk
        mu = (rw * x).sum(2) / nk
        sd = np.sqrt((rw * (x - mu[:, :, None]) ** 2).sum(2) / nk)
    return pi[np.arange(B), np.argmax(mu, 1)]


# ----------------------------------------------------------------------------- thresholds
def posterior(loglr: np.ndarray, pi) -> np.ndarray:
    """P(dis | G) = pi LR / (pi LR + 1 - pi); pi scalar or one value per replicate (row)."""
    loglr = np.asarray(loglr, float)
    lp = logit(pi)
    if np.ndim(lp) and loglr.ndim == 2:
        lp = lp[:, None]
    return expit(loglr + lp)


def first_crossing(post: np.ndarray, level: float, grid_lo: np.ndarray) -> np.ndarray:
    """Lower edge of the first grid cell with posterior >= level (NaN if never or not finite)."""
    c = post >= level
    ok = c.any(-1) & np.isfinite(post).all(-1)
    return np.where(ok, grid_lo[np.argmax(c, -1)], np.nan)


def lr_edges(loglr: np.ndarray, grid_lo: np.ndarray, odds: float):
    """Prior-free evidence edges: first G with LR >= odds; upper edge of the last cell with LR <= 1/odds."""
    up = loglr >= np.log(odds)
    dn = loglr <= -np.log(odds)
    width = grid_lo[1] - grid_lo[0]
    ok = np.isfinite(loglr).all(-1)
    g_up = np.where(up.any(-1) & ok, grid_lo[np.argmax(up, -1)], np.nan)
    last = loglr.shape[-1] - 1 - np.argmax(dn[:, ::-1], -1)
    g_dn = np.where(dn.any(-1) & ok, grid_lo[last] + width, np.nan)
    return g_up, g_dn


def round_to(t, step: float = 0.05):
    return np.round(np.round(np.asarray(t, float) / step) * step, 2)


def rule_value(estimate, lo, hi, paper: float, step: float = 0.05):
    """Frozen reporting rule: keep the paper value if it lies inside the 95 % interval; otherwise the
    multiple of 0.05 inside the interval nearest to the estimate (None if there is none)."""
    if lo is None or hi is None:
        return None, False
    inside = lo - 1e-9 <= paper <= hi + 1e-9
    if inside:
        return paper, True
    cand = [
        round(k * step, 2)
        for k in range(
            int(np.ceil(lo / step - 1e-9)), int(np.floor(hi / step + 1e-9)) + 1
        )
    ]
    if not cand:
        return None, False
    ref = estimate if estimate is not None else (lo + hi) / 2
    return min(cand, key=lambda c: (abs(c - ref), c)), False


def auc_hist(hp: np.ndarray, hn: np.ndarray) -> np.ndarray:
    """ROC AUC of G from binned class counts (ties within a bin count one half); rows = replicates."""
    P, N = hp.sum(-1), hn.sum(-1)
    cn = np.cumsum(hn, -1) - hn
    return (hp * (cn + 0.5 * hn)).sum(-1) / (P * N)


def zone_slices(cuts) -> tuple:
    jo, jd = int(round(cuts[0] * NB)), int(round(cuts[-1] * NB))
    return (slice(0, jo), slice(jo, jd), slice(jd, NB))


# ----------------------------------------------------------------------------- replicate agreement
def kappa(c: np.ndarray) -> float:
    s = c.sum()
    if s <= 0:
        return np.nan
    c = c / s
    po = np.trace(c)
    pe = (c.sum(0) * c.sum(1)).sum()
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def replicate_stats_rows(groups, cuts) -> np.ndarray:
    """Per replicate group: zone counts / binary flips / ordered<->disordered crossings, and the 2x2
    (binary) and 3x3 (zone) agreement tables. groups: list of (G_a, G_b) residue-pair arrays."""
    t_ord, t_bin, t_dis = cuts

    def zone(x):
        return np.where(x < t_ord, 0, np.where(x >= t_dis, 2, 1))

    S = []
    for ga, gb in groups:
        x, z = np.r_[ga, gb], np.r_[gb, ga]
        zx = zone(x)
        fl = (x >= t_bin) != (z >= t_bin)
        cross = (zone(z) == 2 - zx) & (zx != 1)
        row = []
        for zz in range(3):
            row += [
                np.sum(zx == zz),
                np.sum(fl & (zx == zz)),
                np.sum(cross & (zx == zz)),
            ]
        c2 = np.zeros((2, 2))
        np.add.at(c2, ((ga >= t_bin).astype(int), (gb >= t_bin).astype(int)), 1)
        c3 = np.zeros((3, 3))
        np.add.at(c3, (zone(ga), zone(gb)), 1)
        S.append(np.r_[row, c2.ravel(), c3.ravel()])
    return np.array(S, float).reshape(len(S), 9 + 4 + 9)


def replicate_stats(v: np.ndarray) -> dict:
    d = {}
    for zz, nm in enumerate(ZONE_NAMES):
        n_, f_, x_ = v[3 * zz : 3 * zz + 3]
        d[f"{nm}_n"] = n_
        d[f"{nm}_flip"] = f_ / n_ if n_ else np.nan
        d[f"{nm}_cross"] = x_ / n_ if n_ else np.nan
    n_all = v[0] + v[3] + v[6]
    d["binary_flip_all"] = (v[1] + v[4] + v[7]) / n_all if n_all else np.nan
    d["kappa_binary"] = kappa(v[9:13].reshape(2, 2))
    c3 = v[13:22].reshape(3, 3)
    d["kappa_3zone"] = kappa(c3)
    conf = np.ix_([0, 2], [0, 2])
    d["kappa_confident_only"] = kappa(c3[conf])
    d["share_pairs_both_confident"] = c3[conf].sum() / c3.sum() if c3.sum() else np.nan
    return d


def replicate_summary(groups, cuts, nboot: int, rng: np.random.Generator) -> dict:
    """Agreement of the calls between replicate entries, with a bootstrap over replicate groups."""
    S = replicate_stats_rows(groups, cuts)
    out = {
        "cuts": [float(c) for c in cuts],
        "n_groups": len(groups),
        "n_residue_pairs": int(sum(len(a) for a, _ in groups)),
    }
    if not len(S):
        return out
    pt = replicate_stats(S.sum(0))
    W = np.stack(
        [
            np.bincount(rng.integers(0, len(S), len(S)), minlength=len(S))
            for _ in range(nboot)
        ]
    ).astype(float)
    bs = [replicate_stats(w @ S) for w in W]
    for k, v in pt.items():
        if k.endswith("_n"):
            out[k] = int(v)
        else:
            out[k] = [
                None if not np.isfinite(v) else round(float(v), 6),
                *ci([b[k] for b in bs]),
            ]
    return out


def kappa_curve(groups, grid: np.ndarray, nboot: int, rng: np.random.Generator) -> dict:
    """Binary-call kappa between replicates for every cut t of the grid (disordered iff G >= t)."""
    C = np.zeros((len(groups), len(grid), 4))
    for i, (A, B) in enumerate(groups):
        a = A[:, None] >= grid[None, :]
        b = B[:, None] >= grid[None, :]
        C[i, :, 0] = (a & b).sum(0)
        C[i, :, 1] = (a & ~b).sum(0)
        C[i, :, 2] = (~a & b).sum(0)
        C[i, :, 3] = (~a & ~b).sum(0)

    def kap(c):  # c: (..., T, 4); NaN where one call is constant (pe = 1)
        n = c.sum(-1)
        po = (c[..., 0] + c[..., 3]) / n
        pa, pb = (c[..., 0] + c[..., 1]) / n, (c[..., 0] + c[..., 2]) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (po - pe) / (1 - pe)

    flat = C.reshape(len(groups), -1)
    W = np.stack(
        [
            np.bincount(
                rng.integers(0, len(groups), len(groups)), minlength=len(groups)
            )
            for _ in range(nboot)
        ]
    )
    kb = kap((W.astype(float) @ flat).reshape(nboot, len(grid), 4))
    with np.errstate(invalid="ignore"):
        return {
            "t": grid,
            "kappa": kap(C.sum(0)),
            "lo": np.nanpercentile(kb, 2.5, 0),
            "hi": np.nanpercentile(kb, 97.5, 0),
        }
