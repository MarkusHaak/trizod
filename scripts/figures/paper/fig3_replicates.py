"""Replicate analyses shared by Fig. 3a (reliability vs size) and Fig. 3b (conditions).

The BMRB holds many independent measurements of the same protein. A *replicate* of a
chain is another NON-BOUND chain with the identical sequence deposited in a DIFFERENT
BMRB entry. All analyses use only non-bound chains (``is_bound`` False), so a chain is
never compared with a measurement of a complex.

Labels are tier-specific: a chain counted at tier t carries the G-scores of
``scores/<tier>/scores.json`` for that tier (they differ from the unfiltered labels only
at tolerant, where offset rejection is per atom; checked and reported by
:func:`label_differences`). Tiers are CUMULATIVE: a chain belongs to tier t when it is
scored at t, so it also belongs to every more permissive tier.

* :func:`loo_table` -- leave-one-entry-out deviation. For chain i at tier t, the
  per-residue consensus is the median of the unfiltered labels of its replicates from
  the other entries (a fixed reference, identical for every tier); a residue is
  comparable when the chain's own tier-t label and at least two replicate values exist.
  Chains need >= 5 comparable residues; sequences need non-bound chains from >= 3
  entries. Per chain: mean |G - consensus| ("deviation"), flip rate at G = 0.65, and
  the chain's median shift count k over comparable residues (unfiltered labels).
* :func:`pair_table` -- residue-level pairs of replicates for Pearson r / MAE. At each
  tier, candidate pairs (both chains in the tier, different entries) are taken in one
  seeded random order per sequence and capped at ``cap`` pairs per sequence; a pair
  needs >= 10 co-scored residues.
* :func:`condition_pairs` -- the same pairs at the unfiltered tier, split into
  same-condition (|dpH| <= 0.2 and |dT| <= 2 K) and different-condition
  (|dpH| >= 1 or |dT| >= 10 K) pairs, each capped per sequence, with the pair's mean
  |dk| over co-scored residues for shift-count matching.

Bootstrap confidence intervals resample SEQUENCES with replacement (``n_boot``
replicates, fixed seed), because pairs and chains of one sequence are not independent.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402

from trizod.offsets import off_sigma_col  # noqa: E402

SEED = 0
N_BOOT = 1000
MIN_LOO_RES = 5
MIN_PAIR_RES = 10
PAIR_CAP = 20
K_BINS = [0, 12, 17, 21]
K_LABELS = ["k ≤ 12", "k 13–17", "k 18–21"]
SAME = {"dph": 0.2, "dT": 2.0}
DIFF = {"dph": 1.0, "dT": 10.0}

COLUMNS = ["id", "entry_id", "sequence", "is_bound", "gscores", "k", "ph", "temperature",
           "physical_state", "cosolvent_evidence", "sample_state_evidence", "bbshift_types_post",
           *[off_sigma_col(a) for a in sp.BACKBONE_ATOMS]]  # fmt: skip


class Data:
    """Parquet (non-bound subset kept separately) + per-tier labels and memberships."""

    def __init__(self):
        df = sp.load_main(COLUMNS)
        self.df_all = df
        self.tier_ids = sp.load_tier_ids()
        self.strictest = sp.strictest_tier(self.tier_ids)
        df["tier"] = df["id"].map(self.strictest)
        df["trank"] = df["tier"].map(sp.RANK)
        self.labels = {
            t: sp.load_tier_scores(t, fields=("gscores", "k")) for t in sp.TIERS
        }
        nb = df[~df["is_bound"].astype(bool)].reset_index(drop=True)
        self.nb = nb
        self.G_unf = {i: np.asarray(g, float) for i, g in zip(nb["id"], nb["gscores"])}
        self.K_unf = {i: np.asarray(k, float) for i, k in zip(nb["id"], nb["k"])}

    def g(self, tier: str, cid: str) -> np.ndarray:
        return self.labels[tier][cid]["gscores"]


def label_differences(D: Data) -> dict:
    """Chains whose tier label differs from the unfiltered (Parquet) label, per tier."""
    out = {}
    for t in sp.TIERS:
        n_diff = 0
        for cid, rec in D.labels[t].items():
            a = rec["gscores"]
            b = D.labels["unfiltered"][cid]["gscores"]
            if (
                a.shape != b.shape
                or not np.array_equal(np.isnan(a), np.isnan(b))
                or not np.allclose(a[np.isfinite(a)], b[np.isfinite(b)], atol=1e-9)
            ):
                n_diff += 1
        out[t] = n_diff
    return out


# ---------------------------------------------------------------------------- LOO table
def loo_table(D: Data) -> pd.DataFrame:
    nb = D.nb
    rows = []
    for seq, g in nb.groupby("sequence", sort=False):
        if g["entry_id"].nunique() < 3:
            continue
        ids = list(g["id"])
        ents = list(g["entry_id"])
        Gm = np.vstack([D.G_unf[i] for i in ids])
        for j, cid in enumerate(ids):
            others = Gm[[e != ents[j] for e in ents]]
            if others.shape[0] < 2:
                continue
            with np.errstate(all="ignore"):
                n_ok = np.isfinite(others).sum(0)
                cons = np.nanmedian(others, axis=0)
            base = n_ok >= 2
            k_unf = D.K_unf[cid]
            own_unf = D.G_unf[cid]
            m_unf = base & np.isfinite(own_unf)
            medk = float(np.nanmedian(k_unf[m_unf])) if m_unf.sum() else np.nan
            for t in sp.TIERS:
                if cid not in D.tier_ids[t]:
                    continue
                own = D.g(t, cid)
                m = base & np.isfinite(own)
                if m.sum() < MIN_LOO_RES:
                    continue
                d = np.abs(own[m] - cons[m])
                rows.append(
                    {
                        "id": cid,
                        "sequence": seq,
                        "entry_id": ents[j],
                        "tier": t,  # cumulative membership (chain scored at t)
                        "strictest": D.strictest[cid],
                        "dev": float(d.mean()),
                        "flip": float(
                            np.mean((own[m] > sp.G_DIS) != (cons[m] > sp.G_DIS))
                        ),
                        "n_res": int(m.sum()),
                        "medk": medk,
                    }
                )
    L = pd.DataFrame(rows)
    L["kbin"] = pd.cut(L["medk"], K_BINS, labels=K_LABELS)
    return L


def boot_median_by_seq(
    sub: pd.DataFrame, col: str = "dev", n_boot: int = N_BOOT, seed: int = SEED
):
    """Median of ``col`` with a 95 % CI from resampling sequences."""
    if sub.empty:
        return np.nan, np.nan, np.nan
    groups = [v[col].to_numpy() for _, v in sub.groupby("sequence", sort=False)]
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        meds[b] = np.median(np.concatenate([groups[i] for i in pick]))
    return (
        float(sub[col].median()),
        float(np.percentile(meds, 2.5)),
        float(np.percentile(meds, 97.5)),
    )


def boot_mean_by_seq(
    sub: pd.DataFrame, col: str, n_boot: int = N_BOOT, seed: int = SEED
):
    if sub.empty:
        return np.nan, np.nan, np.nan
    agg = sub.groupby("sequence", sort=False)[col].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(agg), (n_boot, len(agg)))
    s = agg[pick, 0].sum(1) / agg[pick, 1].sum(1)
    return (
        float(sub[col].mean()),
        float(np.percentile(s, 2.5)),
        float(np.percentile(s, 97.5)),
    )


def summarise_loo(L: pd.DataFrame) -> pd.DataFrame:
    out = []
    for t in sp.TIERS:
        for kb in [None, *K_LABELS]:
            sub = L[L["tier"] == t]
            if kb is not None:
                sub = sub[sub["kbin"] == kb]
            med, lo, hi = boot_median_by_seq(sub)
            gt, gt_lo, gt_hi = boot_mean_by_seq(
                sub.assign(gt=(sub["dev"] > 0.1).astype(float)), "gt"
            )
            fl, fl_lo, fl_hi = boot_mean_by_seq(sub, "flip")
            out.append(
                {
                    "tier": t,
                    "kbin": kb or "all",
                    "n_chains": int(len(sub)),
                    "n_sequences": int(sub["sequence"].nunique()),
                    "median_dev": med,
                    "median_dev_lo": lo,
                    "median_dev_hi": hi,
                    "share_dev_gt_0.1": gt,
                    "share_dev_gt_0.1_lo": gt_lo,
                    "share_dev_gt_0.1_hi": gt_hi,
                    "flip": fl,
                    "flip_lo": fl_lo,
                    "flip_hi": fl_hi,
                }
            )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- pairs
def _seq_pairs(g: pd.DataFrame, rng: random.Random) -> list:
    rows = g[["id", "entry_id"]].to_numpy().tolist()
    cand = [
        (a[0], b[0])
        for ia, a in enumerate(rows)
        for b in rows[ia + 1 :]
        if a[1] != b[1]
    ]
    rng.shuffle(cand)
    return cand


def _pair_stats(x: np.ndarray, y: np.ndarray) -> dict:
    return {
        "n": x.size,
        "sx": x.sum(),
        "sy": y.sum(),
        "sxx": (x * x).sum(),
        "syy": (y * y).sum(),
        "sxy": (x * y).sum(),
        "sad": np.abs(x - y).sum(),
        "sflip": float(((x > sp.G_DIS) != (y > sp.G_DIS)).sum()),
    }


def pooled(S: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Pooled residue-level r, MAE and flip from per-pair sufficient statistics.

    ``S`` columns: n, sx, sy, sxx, syy, sxy, sad, sflip. ``w`` are per-pair weights."""
    if w is None:
        w = np.ones(len(S))
    T = (S * w[:, None]).sum(0)
    n, sx, sy, sxx, syy, sxy, sad, sflip = T
    cov = sxy / n - (sx / n) * (sy / n)
    vx = sxx / n - (sx / n) ** 2
    vy = syy / n - (sy / n) ** 2
    return {"r": cov / np.sqrt(vx * vy), "mae": sad / n, "flip": sflip / n}


STAT_COLS = ["n", "sx", "sy", "sxx", "syy", "sxy", "sad", "sflip"]


def pair_table(D: Data, cap: int = PAIR_CAP) -> pd.DataFrame:
    """Residue-level replicate pairs per cumulative tier (tier labels for both chains)."""
    nb = D.nb
    rng = random.Random(SEED)
    rows = []
    for seq, g in nb.groupby("sequence", sort=True):
        if g["entry_id"].nunique() < 2:
            continue
        cand = _seq_pairs(g, rng)
        for t in sp.TIERS:
            ids_t = D.tier_ids[t]
            taken = 0
            for a, b in cand:
                if taken >= cap:
                    break
                if a not in ids_t or b not in ids_t:
                    continue
                ga, gb = D.g(t, a), D.g(t, b)
                m = np.isfinite(ga) & np.isfinite(gb)
                if m.sum() < MIN_PAIR_RES:
                    continue
                taken += 1
                x, y = ga[m], gb[m]
                rows.append({"sequence": seq, "tier": t, "a": a, "b": b, **_pair_stats(x, y),
                             "pair_mean_absdg": float(np.abs(x - y).mean()),
                             "hist": np.histogram((x + y) / 2, bins=10, range=(0, 1))[0]})  # fmt: skip
    return pd.DataFrame(rows)


def boot_pooled(
    P: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED, weights=None
) -> dict:
    """Pooled r / MAE / flip with sequence-bootstrap 95 % CIs."""
    S = P[STAT_COLS].to_numpy(float)
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    est = pooled(S, w)
    seqs = P["sequence"].to_numpy()
    uniq, inv = np.unique(seqs, return_inverse=True)
    # per-sequence weighted sums
    SW = np.zeros((len(uniq), S.shape[1]))
    np.add.at(SW, inv, S * w[:, None])
    rng = np.random.default_rng(seed)
    bs = {"r": [], "mae": [], "flip": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        r_ = pooled(SW[pick])
        for k in bs:
            bs[k].append(r_[k])
    out = {
        "n_pairs": int(len(P)),
        "n_sequences": int(len(uniq)),
        "n_residues": int(S[:, 0].sum()),
    }
    for k in ("r", "mae", "flip"):
        out[k] = float(est[k])
        out[f"{k}_lo"] = float(np.percentile(bs[k], 2.5))
        out[f"{k}_hi"] = float(np.percentile(bs[k], 97.5))
    return out


def variance_matched_r(P: pd.DataFrame, ref: pd.DataFrame) -> float:
    """Pooled r after re-weighting pairs so the distribution of residue pair-mean G
    (10 bins) matches ``ref`` -- a check that a higher r is not just a wider G range."""
    h = np.vstack(P["hist"].to_numpy()).sum(0).astype(float)
    href = np.vstack(ref["hist"].to_numpy()).sum(0).astype(float)
    p, pref = h / h.sum(), href / href.sum()
    wbin = np.divide(pref, p, out=np.zeros_like(p), where=p > 0)
    # per-pair weight = mean of its residues' bin weights (pairs keep their residues together)
    H = np.vstack(P["hist"].to_numpy()).astype(float)
    w = (H * wbin).sum(1) / H.sum(1)
    return float(pooled(P[STAT_COLS].to_numpy(float), w)["r"])


def condition_pairs(D: Data, cap: int = PAIR_CAP) -> pd.DataFrame:
    """Unfiltered-tier replicate pairs labelled same / different condition (others dropped)."""
    nb = D.nb
    rng = random.Random(SEED + 1)
    ph = dict(zip(nb["id"], nb["ph"]))
    T = dict(zip(nb["id"], nb["temperature"]))
    rows = []
    for seq, g in nb.groupby("sequence", sort=True):
        if g["entry_id"].nunique() < 2:
            continue
        taken = {"same": 0, "different": 0}
        for a, b in _seq_pairs(g, rng):
            dph, dT = abs(ph[a] - ph[b]), abs(T[a] - T[b])
            if not (np.isfinite(dph) and np.isfinite(dT)):
                continue
            if dph <= SAME["dph"] and dT <= SAME["dT"]:
                cond = "same"
            elif dph >= DIFF["dph"] or dT >= DIFF["dT"]:
                cond = "different"
            else:
                continue
            if taken[cond] >= cap:
                continue
            ga, gb = D.G_unf[a], D.G_unf[b]
            m = np.isfinite(ga) & np.isfinite(gb)
            if m.sum() < MIN_PAIR_RES:
                continue
            taken[cond] += 1
            x, y = ga[m], gb[m]
            dk = np.abs(D.K_unf[a][m] - D.K_unf[b][m])
            rows.append({"sequence": seq, "cond": cond, "a": a, "b": b, "dph": dph, "dT": dT,
                         "mean_absdk": float(np.nanmean(dk)), **_pair_stats(x, y),
                         "pair_mean_absdg": float(np.abs(x - y).mean())})  # fmt: skip
    return pd.DataFrame(rows)


DK_BINS = [0, 0.5, 1, 2, 4, 8, np.inf]


def dk_weights(C: pd.DataFrame) -> np.ndarray:
    """Weights that give the different-condition pairs the same-condition |dk| profile."""
    b = pd.cut(C["mean_absdk"], DK_BINS, right=False, labels=False)
    same = C["cond"] == "same"
    ps = pd.Series(b[same]).value_counts(normalize=True)
    pdf = pd.Series(b[~same]).value_counts(normalize=True)
    w = np.ones(len(C))
    ratio = (ps / pdf).reindex(range(len(DK_BINS) - 1)).fillna(0.0)
    w[~same.to_numpy()] = ratio.reindex(b[~same]).to_numpy()
    return w


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def boot_conditions(
    C: pd.DataFrame, matched: bool, n_boot: int = N_BOOT, seed: int = SEED
) -> dict:
    """Per condition: median per-pair mean |dG|, pooled MAE and r, with sequence-bootstrap CIs.
    With ``matched`` the |dk| weights are recomputed inside every bootstrap replicate."""
    seqs = C["sequence"].to_numpy()
    uniq, inv = np.unique(seqs, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]

    def stats(idx):
        sub = C.iloc[idx].reset_index(drop=True)
        w = dk_weights(sub) if matched else np.ones(len(sub))
        res = {}
        for cond in ("same", "different"):
            m = (sub["cond"] == cond).to_numpy()
            if m.sum() == 0:
                res[cond] = {
                    "median_absdg": np.nan,
                    "mae": np.nan,
                    "r": np.nan,
                    "flip": np.nan,
                }
                continue
            pl = pooled(sub.loc[m, STAT_COLS].to_numpy(float), w[m])
            res[cond] = {
                "median_absdg": weighted_median(
                    sub.loc[m, "pair_mean_absdg"].to_numpy(), w[m]
                ),
                "mae": pl["mae"],
                "r": pl["r"],
                "flip": pl["flip"],
            }
        return res

    est = stats(np.arange(len(C)))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        bs.append(stats(np.concatenate([members[i] for i in pick])))
    out = {}
    for cond in ("same", "different"):
        sub = C[C["cond"] == cond]
        d = {"n_pairs": int(len(sub)), "n_sequences": int(sub["sequence"].nunique())}
        for k in ("median_absdg", "mae", "r", "flip"):
            v = np.array([b[cond][k] for b in bs])
            d[k] = float(est[cond][k])
            d[f"{k}_lo"] = float(np.nanpercentile(v, 2.5))
            d[f"{k}_hi"] = float(np.nanpercentile(v, 97.5))
        out[cond] = d
    ratio = np.array(
        [b["different"]["median_absdg"] / b["same"]["median_absdg"] for b in bs]
    )
    out["ratio_median_absdg"] = {
        "est": est["different"]["median_absdg"] / est["same"]["median_absdg"],
        "lo": float(np.nanpercentile(ratio, 2.5)),
        "hi": float(np.nanpercentile(ratio, 97.5)),
    }
    return out
