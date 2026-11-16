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
  |dk| over co-scored residues for shift-count matching. Only chains whose own condition
  list DEPOSITS both pH and temperature take part (:func:`deposited_conditions`): the
  release fills a missing value with a default and does not flag it, so an imputed value
  could put a pair in the wrong class. An imputed value is treated like a missing one.
* :func:`study_condition_pairs` / :func:`boot_study_conditions` -- the condition pairs
  split by study (one study: a shared normalised citation title or DOI) and by research
  group (a shared depositor, ``_Entry_author``), with medians and ratios in the four cells
  (same / different conditions x within / across studies), shift-count matched, with
  sequence-bootstrap CIs: is the condition effect a study effect? (science review,
  ``reports/story/review/science.md``; macros \\condRatioWithinStudy etc.).
* :func:`one_factor_pairs` -- single-study pairs that differ in exactly one factor (pH or
  temperature), the example of Fig. 4b and the rows of Supplementary Table 14 (rule in
  the function's docstring; deposited values only).

Bootstrap confidence intervals resample SEQUENCES with replacement (``n_boot``
replicates, fixed seed), because pairs and chains of one sequence are not independent.

Run as a script (``--out-dir <draft>/Figures/paper``) it writes the single-study
one-factor pair table to ``<out-dir>/fig3_data/one_factor_pairs.csv``.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_raw_scan as rs  # noqa: E402
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

# Single-study one-factor pairs (blueprint 6.5; Fig. 4b, Supplementary Table 14)
ONE_FACTOR = {
    "temperature": {"dT_min": 10.0, "dph_max": 0.2},  # |dT| >= 10 K, |dpH| <= 0.2
    "pH": {"dph_min": 1.0, "dT_max": 2.0},  # |dpH| >= 1, |dT| <= 2 K
}
ONE_FACTOR_MIN_RES = 50  # co-scored residues
ONE_FACTOR_HIGH_P = 400.0  # bar; both chains must be measured below this
IONIC_ATOL = 0.01  # M; "equal ionic strength" (deposited values agree within 0.01 M)
# a crossing residue: G > CROSS_HI in one chain and G < CROSS_LO in the other
CROSS_HI, CROSS_LO = 0.65, 0.35
RANK_REST_MAX = (
    0.06  # the rest of the chain agrees: median |dG| of non-crossing residues < 0.06
)
RANK_DK_MAX = 2.0  # median shift counts k of the two chains differ by at most 2
ONE_FACTOR_CSV = "one_factor_pairs.csv"

COLUMNS = ["id", "entry_id", "st_id", "entity_name", "sequence", "is_bound", "gscores", "k",
           "ph", "temperature", "ionic_strength", "pressure_bar", "physical_state",
           "cosolvent_evidence", "sample_state_evidence", "membrane_mimetic", "citation_title",
           "citation_doi", "bbshift_types_post",
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


def deposited_conditions(D: Data) -> pd.DataFrame:
    """Per non-bound chain (index ``id``): whether its own sample-condition list deposits
    pH, temperature and ionic strength (bool columns ``ph``, ``temperature``,
    ``ionic_strength``).

    The paper's one definition, ``paper_raw_scan.chain_deposited``: the chain's own
    condition list read with the pipeline's parser, so "not deposited" means exactly that
    the release holds a default (the same flags as Fig. 2a/b, Supplementary Fig. S4 and the
    \\n*NotDeposited* macros). Cached on ``D``."""
    if getattr(D, "_deposited", None) is None:
        nb = D.nb
        dep = rs.chain_deposited(nb["entry_id"], nb["st_id"])
        D._deposited = pd.DataFrame(
            {c: [d[c] for d in dep] for c in rs.DEPOSITED},
            index=pd.Index(nb["id"], name="id"),
        )
    return D._deposited


def condition_pairs(
    D: Data, cap: int = PAIR_CAP, deposited_only: bool = True
) -> pd.DataFrame:
    """Unfiltered-tier replicate pairs labelled same / different condition (others dropped).

    With ``deposited_only`` (the default, and the paper's definition since 12 Nov 2026) a
    pair is skipped, like a pair with a missing value, unless both chains deposit pH and
    temperature (:func:`deposited_conditions`). The candidate order and the seeded draw are
    those of the earlier definition, so ``deposited_only=False`` reproduces it exactly."""
    nb = D.nb
    rng = random.Random(SEED + 1)
    ph = dict(zip(nb["id"], nb["ph"]))
    T = dict(zip(nb["id"], nb["temperature"]))
    if deposited_only:
        dep = deposited_conditions(D)
        ok = dict(zip(dep.index, dep["ph"] & dep["temperature"]))
    rows = []
    for seq, g in nb.groupby("sequence", sort=True):
        if g["entry_id"].nunique() < 2:
            continue
        taken = {"same": 0, "different": 0}
        for a, b in _seq_pairs(g, rng):
            if deposited_only and not (ok[a] and ok[b]):
                continue
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


# ---------------------------------------------------------------------------- studies
# Is the condition effect a study effect? (review, 12 Nov 2026). The pairs of
# condition_pairs() are split by whether the two entries come from ONE STUDY: the same
# normalised citation title (norm_title) or the same citation DOI (norm_doi). Two entries
# are from different studies when they can be compared on a key both carry and no key
# matches; a pair whose entries share no comparable key stays unclassified.
STUDY_CELLS = [(c, s) for c in ("same", "different") for s in ("within", "across")]
STUDY_CSV = "condition_pairs_by_study.csv"


def norm_doi(x) -> str | None:
    """Citation DOI for matching one study: lower case, resolver prefix dropped; None for
    a missing or placeholder value."""
    t = _blank(x).strip().casefold()
    t = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", t)
    return t if t.startswith("10.") else None


def study_relation(ta, tb, da, db) -> str | None:
    """'within' (a shared title or DOI), 'across' (comparable, nothing shared) or None."""
    if (ta is not None and ta == tb) or (da is not None and da == db):
        return "within"
    comparable = (ta is not None and tb is not None) or (
        da is not None and db is not None
    )
    return "across" if comparable else None


def study_condition_pairs(D: Data, C: pd.DataFrame | None = None) -> pd.DataFrame:
    """The condition pairs (``condition_pairs(D)`` unless ``C`` is given) with ``study``
    ('within' / 'across' / None, :func:`study_relation`) and ``shared_author`` (the two
    entries' depositor lists, ``_Entry_author``, share a person: family name + first
    initial). The pairs are exactly those of the condition comparison (Fig. 4a), so the
    four cells partition its same- and different-condition pairs."""
    C = condition_pairs(D) if C is None else C
    nb = D.nb
    title = dict(zip(nb["id"], nb["citation_title"].map(norm_title)))
    doi = dict(zip(nb["id"], nb["citation_doi"].map(norm_doi)))
    ent = dict(zip(nb["id"], nb["entry_id"].astype(str)))
    auth = rs.entry_authors({ent[i] for i in pd.concat([C["a"], C["b"]])})
    C = C.copy()
    C["study"] = [
        study_relation(title[a], title[b], doi[a], doi[b]) for a, b in zip(C.a, C.b)
    ]
    C["shared_author"] = [
        bool(auth[ent[a]] & auth[ent[b]]) if auth[ent[a]] and auth[ent[b]] else np.nan
        for a, b in zip(C.a, C.b)
    ]
    return C


def _ref_weights(sub: pd.DataFrame, cells: dict, ref: np.ndarray) -> np.ndarray:
    """Per-pair weights that give every cell (``cells``: name -> boolean mask) the |dk|
    bin profile of the reference pairs ``ref`` (boolean mask), as :func:`dk_weights`
    does for the different-condition pairs."""
    b = pd.cut(sub["mean_absdk"], DK_BINS, right=False, labels=False).to_numpy()
    nb_ = len(DK_BINS) - 1
    pr = np.bincount(b[ref], minlength=nb_) / max(ref.sum(), 1)
    w = np.zeros(len(sub))
    for m in cells.values():
        if not m.any():
            continue
        pc = np.bincount(b[m], minlength=nb_) / m.sum()
        ratio = np.divide(pr, pc, out=np.zeros(nb_), where=pc > 0)
        w[m] = ratio[b[m]]
    return w


def boot_study_conditions(
    C: pd.DataFrame,
    matched: str | None = "same",
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Median per-pair mean |dG| in the four cells (same / different conditions x within /
    across studies) and their ratios, with sequence-bootstrap 95 % CIs.

    ``C``: :func:`study_condition_pairs` (unclassified pairs are dropped). ``matched``
    selects the shift-count reference that every cell is re-weighted to (recomputed in each
    bootstrap replicate): "same" -- all same-condition pairs (the reference of the main
    condition comparison, ``boot_conditions(matched=True)``); "same_within" -- the
    same-condition pairs within one study; "class" -- each study class separately, its
    different-condition pairs to its own same-condition pairs, and the across-study
    same-condition pairs to the within-study ones (the main analysis's matching repeated
    per class); None -- no re-weighting. Ratios: ``cond_within`` = different / same within
    studies, ``cond_across`` = the same across studies, ``study_same`` = across / within at
    the same conditions, ``study_diff`` = across / within at different conditions,
    ``interaction`` = cond_across / cond_within, ``cond_pooled`` = different / same over
    both classes (cells re-weighted, pooled with their pair counts)."""
    n_all = len(C)
    C = C[C["study"].notna()].reset_index(drop=True)
    seqs = C["sequence"].to_numpy()
    uniq, inv = np.unique(seqs, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]

    def stats(idx):
        sub = C.iloc[idx].reset_index(drop=True)
        cond, study = sub["cond"].to_numpy(), sub["study"].to_numpy()
        cells = {cs: (cond == cs[0]) & (study == cs[1]) for cs in STUDY_CELLS}
        if matched is None:
            w = np.ones(len(sub))
        elif matched == "same":
            w = _ref_weights(sub, cells, cond == "same")
        elif matched == "same_within":
            w = _ref_weights(sub, cells, cells[("same", "within")])
        elif matched == "class":
            w = np.ones(len(sub))
            for s in ("within", "across"):
                ref = cells[("same", s)]
                w_s = _ref_weights(sub, {"d": cells[("different", s)]}, ref)
                w[cells[("different", s)]] = w_s[cells[("different", s)]]
            w_a = _ref_weights(
                sub, {"a": cells[("same", "across")]}, cells[("same", "within")]
            )
            w[cells[("same", "across")]] = w_a[cells[("same", "across")]]
        else:
            raise ValueError(matched)
        x = sub["pair_mean_absdg"].to_numpy()
        med = {
            cs: weighted_median(x[m], w[m]) if m.any() and w[m].sum() > 0 else np.nan
            for cs, m in cells.items()
        }
        pooled_ = {}
        for c in ("same", "different"):
            m = cond == c
            pooled_[c] = weighted_median(x[m], w[m]) if w[m].sum() > 0 else np.nan

        def q(x, y):
            return x / y if np.isfinite(y) and y > 0 else np.nan

        r = {
            "cond_within": q(med[("different", "within")], med[("same", "within")]),
            "cond_across": q(med[("different", "across")], med[("same", "across")]),
            "study_same": q(med[("same", "across")], med[("same", "within")]),
            "study_diff": q(med[("different", "across")], med[("different", "within")]),
            "cond_pooled": q(pooled_["different"], pooled_["same"]),
        }
        r["interaction"] = q(r["cond_across"], r["cond_within"])
        return med, r

    est_m, est_r = stats(np.arange(len(C)))
    rng = np.random.default_rng(seed)
    bm, br = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        m_, r_ = stats(np.concatenate([members[i] for i in pick]))
        bm.append(m_)
        br.append(r_)
    out = {"cells": {}, "ratios": {}, "matched": matched}
    for cs in STUDY_CELLS:
        sub = C[(C["cond"] == cs[0]) & (C["study"] == cs[1])]
        v = np.array([b[cs] for b in bm])
        out["cells"][cs] = {
            "n_pairs": int(len(sub)),
            "n_sequences": int(sub["sequence"].nunique()),
            "median_absdg": float(est_m[cs]),
            "lo": float(np.nanpercentile(v, 2.5)),
            "hi": float(np.nanpercentile(v, 97.5)),
        }
    for k in est_r:
        v = np.array([b[k] for b in br])
        out["ratios"][k] = {
            "est": float(est_r[k]),
            "lo": float(np.nanpercentile(v, 2.5)),
            "hi": float(np.nanpercentile(v, 97.5)),
            "p_le_1": float(np.nanmean(v <= 1.0)),  # bootstrap share of ratios <= 1
        }
    out["n_unclassified"] = int(n_all - len(C))
    return out


# ---------------------------------------------------------------------------- one factor
_PLACEHOLDER_TITLE = re.compile(r"^(\.|\?|na|n/a|none|not available|to be published)?$")


def _blank(x) -> str:
    """A string field as text ('' for None / NaN)."""
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def norm_title(x) -> str | None:
    """Citation title for matching one study: case-folded, whitespace collapsed, a final
    period dropped; None for a missing or placeholder title (which cannot tie two
    entries to one study)."""
    t = re.sub(r"\s+", " ", _blank(x)).strip().rstrip(".").casefold()
    return None if _PLACEHOLDER_TITLE.fullmatch(t) else t


def _runs(mask: np.ndarray) -> list:
    """Inclusive (start, end) index runs of True in a boolean vector."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def crossing_run(ga: np.ndarray, gb: np.ndarray) -> tuple:
    """Longest run of consecutive residues that cross between two G-score profiles: G >
    CROSS_HI in one chain and G < CROSS_LO in the other, in the same direction throughout
    the run. An unscored residue (NaN) breaks a run. Returns ``(length, start, end, side)``
    with 0-based inclusive positions and ``side`` the chain ('a' or 'b') that is disordered
    in the run; ``(0, -1, -1, '')`` when no residue crosses."""
    best = (0, -1, -1, "")
    with np.errstate(invalid="ignore"):
        sides = (
            ("a", (ga > CROSS_HI) & (gb < CROSS_LO)),
            ("b", (gb > CROSS_HI) & (ga < CROSS_LO)),
        )
    for side, m in sides:
        for s, e in _runs(m):
            if e - s + 1 > best[0]:
                best = (int(e - s + 1), int(s), int(e), side)
    return best


def one_factor_pairs(
    D: Data, out_csv: Path | str | None = None, deposited_only: bool = True
) -> pd.DataFrame:
    """Single-study replicate pairs that differ in exactly one factor (blueprint 6.5).

    1. Qualifying chains: non-bound; pH, temperature AND ionic strength deposited, not
       imputed (:func:`deposited_conditions`); no cosolvent evidence, no membrane mimetic, no
       solid-sample evidence; not measured at >= ONE_FACTOR_HIGH_P bar (a missing pressure
       counts as ambient).
    2. Qualifying pairs: identical sequence, one study (same normalised ``citation_title``,
       :func:`norm_title`), different entries, the same ``physical_state``, equal ionic
       strength (within IONIC_ATOL M), and exactly one factor changed:
       temperature pairs |dT| >= 10 K with |dpH| <= 0.2; pH pairs |dpH| >= 1 with |dT| <= 2 K.
       At least ONE_FACTOR_MIN_RES co-scored residues (unfiltered labels of the Parquet).
       Every pair within a sequence and study counts (no cap); chain ``a`` has the lower
       value of the changed factor.
    3. Ranking: by the longest crossing run (:func:`crossing_run`). A pair ranks only if it
       has a crossing run, the rest of the chain agrees (median |dG| over the co-scored
       residues that do not cross < RANK_REST_MAX) and the chains' median shift
       counts k (over co-scored residues) differ by at most RANK_DK_MAX. Ties: lower
       rest-of-chain median |dG|, then the chain IDs.

    Returns one row per qualifying pair (``rank`` 1, 2, ... for ranked pairs, NaN
    otherwise), sorted by rank, then by run length and mean |dG|. ``mean_absdg`` is the
    per-pair mean |dG| over co-scored residues, the measure of the Fig. 4a condition
    comparison. Residue positions (``cross_start``/``cross_end``) are 1-based sequence
    positions. With ``out_csv`` the table is also written there (Supplementary Table 14
    data; the Fig. 4b selection is the row with rank 1). ``deposited_only=False`` drops the
    deposited-value requirement of rule 1 (a diagnostic, not the paper's definition)."""
    nb = D.nb.copy()
    if deposited_only:
        ok = deposited_conditions(D).all(axis=1).reindex(nb["id"]).to_numpy()
    else:
        ok = np.ones(len(nb), bool)
    cos = nb["cosolvent_evidence"].fillna(False).astype(bool).to_numpy()
    memb = nb["membrane_mimetic"].notna().to_numpy()
    solid = (nb["sample_state_evidence"].astype(str) == "solid").to_numpy()
    with np.errstate(invalid="ignore"):
        highp = nb["pressure_bar"].to_numpy(float) >= ONE_FACTOR_HIGH_P
    nb["_title"] = nb["citation_title"].map(norm_title)
    sel = nb[ok & ~cos & ~memb & ~solid & ~highp & nb["_title"].notna().to_numpy()]
    rows = []
    for (_seq, _t), g in sel.groupby(["sequence", "_title"], sort=True):
        if g["entry_id"].nunique() < 2:
            continue
        recs = g.to_dict("records")
        for i, r1 in enumerate(recs):
            for r2 in recs[i + 1 :]:
                if r1["entry_id"] == r2["entry_id"]:
                    continue
                if _blank(r1["physical_state"]) != _blank(r2["physical_state"]):
                    continue
                if not np.isclose(r1["ionic_strength"], r2["ionic_strength"], rtol=0, atol=IONIC_ATOL):  # fmt: skip
                    continue
                dph = abs(r1["ph"] - r2["ph"])
                dT = abs(r1["temperature"] - r2["temperature"])
                tf, pf = ONE_FACTOR["temperature"], ONE_FACTOR["pH"]
                if dT >= tf["dT_min"] and dph <= tf["dph_max"]:
                    factor, key = "temperature", "temperature"
                elif dph >= pf["dph_min"] and dT <= pf["dT_max"]:
                    factor, key = "pH", "ph"
                else:
                    continue
                ra, rb = (r1, r2) if r1[key] <= r2[key] else (r2, r1)
                ga, gb = D.G_unf[ra["id"]], D.G_unf[rb["id"]]
                m = np.isfinite(ga) & np.isfinite(gb)
                if m.sum() < ONE_FACTOR_MIN_RES:
                    continue
                ad = np.abs(ga - gb)
                L, s, e, side = crossing_run(ga, gb)
                with np.errstate(invalid="ignore"):
                    cross = ((ga > CROSS_HI) & (gb < CROSS_LO)) | ((gb > CROSS_HI) & (ga < CROSS_LO))  # fmt: skip
                n_cross = int(cross.sum())
                rest = m & ~cross
                ka, kb = D.K_unf[ra["id"]], D.K_unf[rb["id"]]
                medk_a, medk_b = float(np.nanmedian(ka[m])), float(np.nanmedian(kb[m]))
                rest_med = float(np.median(ad[rest])) if rest.any() else np.nan
                rows.append({
                    "factor": factor, "a": ra["id"], "b": rb["id"],
                    "entry_a": ra["entry_id"], "entry_b": rb["entry_id"],
                    "entity_name": _blank(ra["entity_name"]), "length": len(ra["sequence"]),
                    "temperature_a": float(ra["temperature"]), "temperature_b": float(rb["temperature"]),
                    "ph_a": float(ra["ph"]), "ph_b": float(rb["ph"]),
                    "ionic_strength_a": float(ra["ionic_strength"]), "ionic_strength_b": float(rb["ionic_strength"]),
                    "physical_state": _blank(ra["physical_state"]),
                    "n_coscored": int(m.sum()),
                    "mean_absdg": float(ad[m].mean()), "median_absdg": float(np.median(ad[m])),
                    "cross_run_len": L,
                    "cross_start": s + 1 if L else np.nan, "cross_end": e + 1 if L else np.nan,
                    "disordered_in": (ra["id"] if side == "a" else rb["id"]) if L else "",
                    "n_cross_residues": n_cross, "rest_median_absdg": rest_med,
                    "medk_a": medk_a, "medk_b": medk_b,
                    "rankable": bool(L >= 1 and rest_med < RANK_REST_MAX and abs(medk_a - medk_b) <= RANK_DK_MAX),
                    "citation_title": _blank(ra["citation_title"]),
                    "citation_doi": _blank(ra["citation_doi"]),
                    "sequence": ra["sequence"],
                })  # fmt: skip
    P = pd.DataFrame(rows)
    if P.empty:
        P = pd.DataFrame(columns=["rank", "factor", "a", "b", "rankable"])
    else:
        rk = P[P["rankable"]].sort_values(
            ["cross_run_len", "rest_median_absdg", "a", "b"],
            ascending=[False, True, True, True],
        )
        P["rank"] = pd.Series(np.arange(1, len(rk) + 1), index=rk.index)
        P = P.sort_values(
            ["rank", "cross_run_len", "mean_absdg", "a", "b"],
            ascending=[True, False, False, True, True],
            na_position="last",
        ).reset_index(drop=True)
        P = P[["rank", *[c for c in P.columns if c != "rank"]]]
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rnd = {"temperature_a": 2, "temperature_b": 2, "ph_a": 2, "ph_b": 2,
               "ionic_strength_a": 4, "ionic_strength_b": 4, "mean_absdg": 4,
               "median_absdg": 4, "rest_median_absdg": 4, "medk_a": 1, "medk_b": 1}  # fmt: skip
        Q = P.round({c: n for c, n in rnd.items() if c in P.columns})
        for c in ("rank", "cross_start", "cross_end"):
            if c in Q.columns:
                Q[c] = Q[c].astype("Int64")
        Q.to_csv(out_csv, index=False)
    return P


def top_one_factor_pair(P: pd.DataFrame):
    """The ranked top pair of :func:`one_factor_pairs` (a row), or None when no pair ranks."""
    top = P[P["rank"] == 1]
    return None if top.empty else top.iloc[0]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Write the single-study one-factor pair table."
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="the draft's Figures/paper"
    )
    a = ap.parse_args(argv)
    P = one_factor_pairs(Data(), out_csv=a.out_dir / "fig3_data" / ONE_FACTOR_CSV)
    top = top_one_factor_pair(P)
    print(f"{len(P)} one-factor pairs ({(P.factor == 'temperature').sum()} temperature, "
          f"{(P.factor == 'pH').sum()} pH), {int(P.rankable.sum())} rank; top: "
          f"{None if top is None else (top.a, top.b, top.cross_run_len)}")  # fmt: skip


if __name__ == "__main__":
    main()
