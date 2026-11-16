#!/usr/bin/env python3
"""Supplementary Fig. S10 and Fig. 5c -- reference baselines and headroom.

Fig. 5c (``draw_main``; 160 x 40 mm): pooled residue-level Spearman rho of both baselines,
trained on each tier (rows), on each test set (two facets), against the replicate
ceiling sqrt(rho) of the test chains themselves (``test_ceilings``: each test chain's
label against its non-bound identical-sequence chains from other entries outside the
test sets, <= 20 per test chain, seed 0). Whiskers and the ceiling band are 95 %
percentile bootstrap CIs over test chains (``--n-boot``, default 1,000, seed 0); for the
ceiling a test chain is resampled with all its replicate pairs. Like-for-like check (drawn
since the 12 Nov 2026 review, small open squares below each ridge marker): the ridge rho on
exactly the residue pairs behind each ceiling (each test chain weighted by its number of
replicate pairs; ``ridge_on_replicate_rows``, the definition of the
``\\baseRidgeRhoOnRep*`` macros), with the ceiling's bootstrap draws. All ridge rows are
squares in the tier colour; circles and triangles are kept for the two test sets.
Outputs: <out-dir>/fig5c_baselines.pdf and <out-dir>/fig5_data/fig5c_rho.csv,
fig5c_numbers.json.

Two sequence-only reference models are trained on the training representatives of
each cumulative tier (``train_tier`` at that tier or stricter) and evaluated on BOTH
pinned test sets (TriZOD test, CheZOD117):

* **per-amino-acid mean (floor):** predicts each residue's G as the mean training G of
  its amino-acid type;
* **one-hot ridge:** a 15-residue window (centre +- 7) one-hot encoded over 20 amino
  acids plus "other / outside", with an intercept, fitted by ridge regression
  (lambda = 10) on all scored training residues.

This is the port of ``user_first/ceiling_and_baseline.py`` part 2 that
``scripts/manuscript/make_numbers.py`` (``sec_baselines``) also runs, so the ridge and
floor Pearson r here equal the ``\\base*`` macros. Labels are the release (Parquet)
labels, i.e. what a user loading the Parquet trains on.

Metrics, all at the paper's cut G > 0.65 for AUC:
* per residue: Spearman rho and ROC AUC pooled over all scored test residues;
* per chain: the mean over test chains of the within-chain Spearman rho (chains with
  >= 10 scored residues and non-constant labels) and of the within-chain AUC (chains
  with both disordered and ordered residues).

Supplementary Fig. S10 (``plot``) draws three columns: the per-residue AUC and both
per-chain metrics. Since the 12 Nov 2026 review it uses the encodings of Fig. 5c: one row
per training tier (horizontal labels), the ridge as filled squares in the tier colour, the
amino-acid mean as open diamonds, the test chains' own replicate agreement as a black
vertical line. The pooled per-residue Spearman rho moved to main Fig. 5c
(``draw_main``) on 12 Nov 2026 (story blueprint 6.6); it is still computed and written
to ``S10_baselines.csv`` and Supplementary Table T9.

Headroom (grey band): replicate agreement, i.e. how well ANOTHER measurement of the
same protein predicts a label. Pairs of non-bound identical-sequence chains from
different entries (<= 20 per sequence, seed 0; the residue-level-r sample of
Supplementary Fig. S7), scored in the same metrics with each chain's G used as the
predictor of the other's label (both directions). The band spans the unfiltered to
the strict cumulative tier; the dashed line marks strict. The sqrt(rho_rep) ceiling of
the pooled rho is drawn in Fig. 5c. Replicate agreement of the test chains themselves
(test chain vs its replicates outside the test set; diamonds) is in
``S10_replicate_ceiling_test_sets.csv``.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s10_baselines.py \
        --out-dir <draft>/Figures/paper
Outputs: <out-dir>/S10.pdf and <out-dir>/suppfigs_b_data/S10_*.csv / S10_summary.json
(the full table is Supplementary Table T9).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402
import supp_s7_replicates as s7  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
W = 7
D = (2 * W + 1) * 21 + 1
LAMBDA = 10.0
MIN_CHAIN_RES = 10
TESTS = {"test_trizod": "TriZOD test", "test_chezod117": "CheZOD117"}
DATA_SUBDIR = "suppfigs_b_data"  # under --out-dir
FIG5_SUBDIR = "fig5_data"  # Fig. 5c source data, under --out-dir
COLUMNS = ["id", "entry_id", "sequence", "is_bound", "gscores", "zscores", "k", "ph", "temperature",
           "split", "train_tier"]  # fmt: skip

_LUT = np.full(256, 20, dtype=np.int64)
for _i, _a in enumerate(AA):
    _LUT[ord(_a)] = _i


def feats(seq: str):
    a = _LUT[np.frombuffer(seq.encode(), dtype=np.uint8)]
    L = len(a)
    X = np.zeros((L, D))
    X[:, -1] = 1.0
    for o in range(-W, W + 1):
        i0, i1 = max(0, -o), min(L, L - o)
        if i1 <= i0:
            continue
        rows = np.arange(i0, i1)
        X[rows, (o + W) * 21 + a[rows + o]] = 1.0
    return X, a


def rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy()


def spearman(x, y) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def auc(y, p) -> float:
    pos = y > sp.G_DIS
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    r = rank(p)
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(chains: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """``chains``: list of (label, prediction) per chain (scored residues only)."""
    y = np.concatenate([c[0] for c in chains])
    p = np.concatenate([c[1] for c in chains])
    rho_c = [spearman(yy, pp) for yy, pp in chains if len(yy) >= MIN_CHAIN_RES]
    auc_c = [auc(yy, pp) for yy, pp in chains]
    rho_c = np.array([v for v in rho_c if np.isfinite(v)])
    auc_c = np.array([v for v in auc_c if np.isfinite(v)])
    return {
        "pearson_res": float(np.corrcoef(y, p)[0, 1]),
        "rho_res": spearman(y, p),
        "auc_res": auc(y, p),
        "rho_chain": float(rho_c.mean()) if len(rho_c) else np.nan,
        "n_rho_chain": int(len(rho_c)),
        "auc_chain": float(auc_c.mean()) if len(auc_c) else np.nan,
        "n_auc_chain": int(len(auc_c)),
        "n_res": int(len(y)),
        "n_chains": int(len(chains)),
    }


def predictions(df: pd.DataFrame) -> dict:
    """Per-chain (label, prediction) of both baselines, per training tier and test set.

    Returns ``{(train_tier, model, test_set): {"ids": [...], "chains": [(y, p), ...],
    "n_train_chains": int}}`` over scored residues only; :func:`baselines` scores them.
    """
    tests = {}
    for split, name in TESTS.items():
        recs = []
        sub = df[df.split == split]
        for i, s, gs in zip(sub["id"], sub["sequence"], sub["gscores"]):
            g = np.asarray(gs, float)
            m = ~np.isnan(g)
            X, a = feats(s)
            recs.append((i, X[m], g[m], a[m], np.flatnonzero(m)))
        tests[name] = recs
    out = {}
    tr_rank = df["train_tier"].map(
        lambda v: sp.RANK.get(v, -1) if isinstance(v, str) else -1
    )
    for t in sp.TIERS:
        tr = df[tr_rank >= sp.RANK[t]]
        XtX = np.zeros((D, D))
        Xty = np.zeros(D)
        ysum = np.zeros(21)
        ycnt = np.zeros(21)
        for s, gs in zip(tr.sequence, tr.gscores):
            g = np.asarray(gs, float)
            m = ~np.isnan(g)
            X, a = feats(s)
            X, y, aa = X[m], g[m], a[m]
            XtX += X.T @ X
            Xty += X.T @ y
            np.add.at(ysum, aa, y)
            np.add.at(ycnt, aa, 1)
        ymean = Xty[-1] / XtX[-1, -1]
        aa_mean = np.where(ycnt > 0, ysum / np.maximum(ycnt, 1), ymean)
        w = np.linalg.solve(XtX + LAMBDA * np.eye(D), Xty)
        for name, recs in tests.items():
            # this tier's aa_mean / w, bound explicitly (the lambdas are used right here)
            for model, pred in (("per-amino-acid mean", lambda X, a, mu=aa_mean: mu[a]),
                                ("one-hot ridge", lambda X, a, w=w: X @ w)):  # fmt: skip
                out[(t, model, name)] = {
                    "ids": [r[0] for r in recs],
                    "chains": [(y, pred(X, a)) for _, X, y, a, _ in recs],
                    "scored": [
                        r[4] for r in recs
                    ],  # residue positions of each chain's arrays
                    "n_train_chains": len(tr),
                }
    return out


def baselines(df: pd.DataFrame, preds: dict | None = None) -> pd.DataFrame:
    preds = predictions(df) if preds is None else preds
    rows = []
    for (t, model, name), v in preds.items():
        rows.append(
            {
                "train_tier": t,
                "n_train_chains": v["n_train_chains"],
                "test_set": name,
                "model": model,
                **metrics(v["chains"]),
            }
        )
    return pd.DataFrame(rows)


def pair_metrics(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Replicate agreement: each chain's G predicts the other's label (both directions)."""
    chains = [(x, y) for x, y in pairs] + [(y, x) for x, y in pairs]
    m = metrics(chains)
    m["sqrt_rho_res"] = float(np.sqrt(m["rho_res"]))
    m["sqrt_pearson_res"] = float(np.sqrt(m["pearson_res"]))
    return m


def tier_ceilings(df: pd.DataFrame) -> pd.DataFrame:
    tier_ids = sp.load_tier_ids()
    strictest = sp.strictest_tier(tier_ids)
    d = df.copy()
    d["trank"] = d["id"].map(strictest).map(sp.RANK)
    pairs = s7.replicate_pairs(d, s7.PAIR_CAP)
    rows = []
    for t in sp.TIERS:
        sel = [(p["x"], p["y"]) for p in pairs if p["both"] >= sp.RANK[t]]
        rows.append({"tier": t, "n_pairs": len(sel), **pair_metrics(sel)})
    return pd.DataFrame(rows)


def test_ceiling_pairs(df: pd.DataFrame, cap: int = 20) -> dict:
    """The pairs behind :func:`test_ceilings`, grouped by test chain.

    Returns ``{test_set: [(test_id, replicate_id, y, p, pos), ...]}``: the test chain's
    label ``y`` and one replicate's G ``p`` over their co-scored residues (positions
    ``pos``), for up to ``cap`` non-bound identical-sequence chains of other entries
    outside the test sets (seed 0).
    """
    rng = random.Random(0)
    nb = df[~df["is_bound"].astype(bool) & ~df["split"].isin(list(TESTS))]
    # dict(groupby) would call GroupBy.keys, which is the grouping key, not a method
    by_seq = {s: g for s, g in nb.groupby("sequence")}  # noqa: C416
    out = {}
    for split, name in TESTS.items():
        pairs = []
        for _, r in df[df.split == split].iterrows():
            g = by_seq.get(r.sequence)
            if g is None:
                continue
            reps = [x for x in g.to_dict("records") if x["entry_id"] != r.entry_id]
            rng.shuffle(reps)
            y = np.asarray(r.gscores, float)
            for x in reps[:cap]:
                p = np.asarray(x["gscores"], float)
                m = ~np.isnan(y) & ~np.isnan(p)
                if m.sum() < s7.MIN_PAIR_RES:
                    continue
                pairs.append((r["id"], x["id"], y[m], p[m], np.flatnonzero(m)))
        out[name] = pairs
    return out


def test_ceilings(
    df: pd.DataFrame, cap: int = 20, pairs: dict | None = None
) -> pd.DataFrame:
    """Replicate agreement of the test chains themselves: each test chain's label
    against non-bound identical-sequence chains from other entries (outside the test sets)."""
    pairs = test_ceiling_pairs(df, cap) if pairs is None else pairs
    rows = []
    for name in TESTS.values():
        chains = [(y, p) for _, _, y, p, _ in pairs[name]]
        m_ = metrics(chains) if chains else {}
        rows.append(
            {
                "test_set": name,
                "n_test_chains_with_replicates": len({t for t, *_ in pairs[name]}),
                "n_pairs": len(chains),
                **m_,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# Fig. 5c: Spearman rho of both baselines per training tier against the replicate
# ceiling sqrt(rho_rep) of each test set, with 95 % bootstrap CIs over test chains
# ----------------------------------------------------------------------------------
N_BOOT = 1000
BOOT_SEED = 0


def _pooled_rho(ys: list, ps: list) -> float:
    from scipy.stats import rankdata

    y, p = np.concatenate(ys), np.concatenate(ps)
    if len(y) < 3 or np.ptp(y) == 0 or np.ptp(p) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])


def boot_rho(groups: list[list[tuple]], n_boot: int, seed: int) -> np.ndarray:
    """Pooled residue-level Spearman rho of ``n_boot`` resamples of the groups.

    ``groups``: one list of (y, p) arrays per resampled unit (a test chain; for the
    ceiling, all pairs of one test chain). Units are drawn with replacement.
    """
    rng = np.random.default_rng(seed)
    n = len(groups)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        ys = [y for i in pick for y, _ in groups[i]]
        ps = [p for i in pick for _, p in groups[i]]
        out[b] = _pooled_rho(ys, ps)
    return out


def main_panel_data(preds: dict, pairs: dict, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Rows of Fig. 5c: both baselines per tier and the sqrt(rho) ceiling, per test set."""
    rows = []
    for name in TESTS.values():
        # baselines: resample test chains (one group = one chain), same draws for all models
        for (t, model, nm), v in preds.items():
            if nm != name:
                continue
            groups = [[c] for c in v["chains"]]
            rho = _pooled_rho([y for y, _ in v["chains"]], [p for _, p in v["chains"]])
            bs = boot_rho(groups, n_boot, BOOT_SEED)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            rows.append({"test_set": name, "kind": model, "train_tier": t, "rho": rho,
                         "lo": lo, "hi": hi, "n_chains": len(groups),
                         "n_res": int(sum(len(y) for y, _ in v["chains"])),
                         "n_pairs": np.nan})  # fmt: skip
        # ceiling: resample test chains with all their replicate pairs; sqrt is monotone
        by_test: dict = {}
        for tid, _, y, p, _ in pairs[name]:
            by_test.setdefault(tid, []).append((y, p))
        groups = list(by_test.values())
        rho = _pooled_rho(
            [y for g in groups for y, _ in g], [p for g in groups for _, p in g]
        )
        bs = np.sqrt(np.clip(boot_rho(groups, n_boot, BOOT_SEED), 0, None))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append({"test_set": name, "kind": "replicate ceiling", "train_tier": "",
                     "rho": float(np.sqrt(rho)), "lo": lo, "hi": hi, "n_chains": len(groups),
                     "n_res": int(sum(len(y) for g in groups for y, _ in g)),
                     "n_pairs": len(pairs[name]), "rho_rep": rho})  # fmt: skip
    return pd.DataFrame(rows)


def ridge_on_replicate_chains(preds: dict, pairs: dict) -> dict:
    """A like-for-like check of the ceiling: the ridge rho on exactly the residue pairs
    behind it (each test chain weighted by its number of replicate pairs, co-scored
    residues only), per training tier."""
    out = {}
    for name in TESTS.values():
        for t in sp.TIERS:
            v = preds[(t, "one-hot ridge", name)]
            at = {i: (c, pos) for i, c, pos in zip(v["ids"], v["chains"], v["scored"])}
            ys, ps = [], []
            for tid, _, y, _, pos in pairs[name]:
                (y_all, p_all), scored = at[tid]
                k = np.searchsorted(scored, pos)
                assert np.array_equal(scored[k], pos) and np.allclose(y_all[k], y)
                ys.append(y)
                ps.append(p_all[k])
            out[f"{name} | {t}"] = round(_pooled_rho(ys, ps), 3)
    return out


REP_KIND = "one-hot ridge, ceiling's residue pairs"  # kind of the like-for-like rows


def ridge_on_replicate_rows(
    preds: dict, pairs: dict, n_boot: int = N_BOOT
) -> pd.DataFrame:
    """Fig. 5c like-for-like rows: per test set and tier, the ridge rho on exactly the
    residue pairs behind the ceiling, with a 95 % CI from the ceiling's own bootstrap (test
    chains resampled with all their pairs; same seed, so the same draws). The values and
    CIs are those of the ``\\baseRidgeRhoOnRep*`` macros (make_numbers.sec_review_science)."""
    rows = []
    for name in TESTS.values():
        by_test: dict = {}
        for tid, _, y, _, pos in pairs[name]:
            by_test.setdefault(tid, []).append((y, pos))
        for t in sp.TIERS:
            v = preds[(t, "one-hot ridge", name)]
            at = {i: (c, pos) for i, c, pos in zip(v["ids"], v["chains"], v["scored"])}
            groups = []
            for tid, g in by_test.items():
                (y_all, p_all), scored = at[tid]
                grp = []
                for y, pos in g:
                    k = np.searchsorted(scored, pos)
                    assert np.array_equal(scored[k], pos) and np.allclose(y_all[k], y)
                    grp.append((y, p_all[k]))
                groups.append(grp)
            rho = _pooled_rho([y for g in groups for y, _ in g], [p for g in groups for _, p in g])  # fmt: skip
            lo, hi = np.nanpercentile(boot_rho(groups, n_boot, BOOT_SEED), [2.5, 97.5])
            rows.append({"test_set": name, "kind": REP_KIND, "train_tier": t, "rho": rho,
                         "lo": lo, "hi": hi, "n_chains": len(groups),
                         "n_res": int(sum(len(g) for g in groups)),
                         "n_pairs": len(pairs[name])})  # fmt: skip
    return pd.DataFrame(rows)


# Fig. 5c layout (mm): two facets (TriZOD test | CheZOD117) on one rho axis, one row per
# training tier; placed at 1:1 below Fig. 5a, b. Each row has two sub-rows: the amino-acid
# mean and the ridge on all residues above, the ridge on the ceiling's residue pairs below.
W_C, H_C = sp.W_DOUBLE, 40.0
FACET_X0, FACET_GAP = 13.0, 9.0  # left axes edge = that of Fig. 5a
FACET_W = (W_C - FACET_X0 - FACET_GAP - 1.0) / 2
TITLE_TOP, LABEL_BASE = (
    0.4,
    9.4,
)  # facet titles (top) and the label row (last baseline), mm
ROW_TOP, ROW_H = 12.9, 5.4  # centre of the first row and row pitch, from the figure top
# sub-row offset (row units): all residues above (-), the ceiling's pairs below (+)
SUB = 0.2
# from -0.3: on CheZOD117 the ridge on the ceiling's residue pairs is near 0, with a CI
# down to -0.25 (\baseRidgeRhoOnRepLoCheZOD)
X_LIM_RHO = (-0.3, 1.0)
X_TICKS_RHO = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
AA_STYLE = {"marker": "D", "mfc": "white", "mec": sp.INK_SOFT, "color": sp.INK_SOFT}
BAND = "#E4E4E4"  # the ceiling's CI: a light neutral band (not an entity colour)
MODEL_LABEL = {"per-amino-acid mean": "Amino-acid\nmean", "one-hot ridge": "Ridge"}
# marker sizes (pt): ridge on all residues, ridge on the ceiling's residue pairs
RIDGE_MS, REP_MS = 3.4, 3.0


def draw_main(M: pd.DataFrame):
    """Fig. 5c: baselines per training tier against the replicate ceiling."""
    from matplotlib.lines import Line2D

    W, H = W_C, H_C
    fig = sp.figure(W, H)
    n_rows = len(sp.TIERS)
    y_top = ROW_TOP - ROW_H / 2  # axes top and bottom, mm from the figure top
    y_bot = ROW_TOP + (n_rows - 0.5) * ROW_H
    for j, name in enumerate(TESTS.values()):
        x0 = FACET_X0 + j * (FACET_W + FACET_GAP)
        ax = fig.add_axes([x0 / W, 1 - y_bot / H, FACET_W / W, (y_bot - y_top) / H])
        d = M[M.test_set == name]
        ceil = d[d.kind == "replicate ceiling"].iloc[0]
        ax.axvspan(ceil.lo, ceil.hi, color=BAND, lw=0, zorder=0)
        ax.axvline(ceil.rho, color=sp.INK, lw=1.0, zorder=2)
        ax.axvline(0, color=sp.CAT["light"], lw=0.5, zorder=1)  # rho = 0
        for i, t in enumerate(sp.TIERS):
            # ridge: a square in the tier colour (filled: all residues; open: the ceiling's
            # residue pairs); circles and triangles stay with the two test sets
            ridge = {"marker": "s", "mfc": sp.TIER_COLORS[t], "mec": sp.TIER_COLORS[t],
                     "color": sp.TIER_COLORS[t]}  # fmt: skip
            ridge_rep = {**ridge, "mfc": "white"}
            for kind, style, y, ms, mew in (
                ("per-amino-acid mean", AA_STYLE, i - SUB, RIDGE_MS, 0.7),
                ("one-hot ridge", ridge, i - SUB, RIDGE_MS, 0.7),
                (REP_KIND, ridge_rep, i + SUB, REP_MS, 0.9),
            ):
                r = d[(d.kind == kind) & (d.train_tier == t)].iloc[0]
                ax.plot([r.lo, r.hi], [y, y], color=sp.INK_SOFT, lw=0.6, zorder=3,
                        solid_capstyle="butt")  # fmt: skip
                ax.plot([r.rho], [y], ls="", ms=ms, mew=mew, zorder=4, **style)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_xlim(*X_LIM_RHO)
        ax.set_xticks(
            X_TICKS_RHO, [f"{v:g}".replace("-", "\u2212") for v in X_TICKS_RHO]
        )
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(sp.TIERS if j == 0 else [])
        ax.tick_params(axis="y", length=0, pad=2.0)
        ax.spines["left"].set_visible(False)
        # direct labels in one row above the first tier: the two models, then the ceiling
        y_lab = 1 - LABEL_BASE / H

        def to_fig(xd, x0=x0):  # data rho -> figure fraction
            return (x0 + FACET_W * (xd - X_LIM_RHO[0]) / np.ptp(X_LIM_RHO)) / W

        for kind, lab in MODEL_LABEL.items():
            xm = d[d.kind == kind].rho.mean()
            fig.text(to_fig(xm), y_lab, lab, ha="center", va="baseline", fontsize=sp.FS_BODY,
                     multialignment="center")  # fmt: skip
        # beside the line, on the side with room (right unless the line is near rho = 1)
        room_right = (X_LIM_RHO[1] - ceil.rho) * FACET_W > 20.0
        fig.text(to_fig(ceil.rho) + (0.8 if room_right else -0.8) / W, y_lab,
                 f"Replicate ceiling,\n√ρ ({int(ceil.n_chains)} chains)",
                 ha="left" if room_right else "right", va="baseline", fontsize=sp.FS_BODY,
                 multialignment="left" if room_right else "right")  # fmt: skip
        # facet title: the set's marker, then its name and size
        key = "test" if name == "TriZOD test" else "chezod117"
        n_ch = int(d[d.kind == "one-hot ridge"].n_chains.iloc[0])
        title = ("TriZOD test" if key == "test" else "CheZOD117") + f" ({n_ch} chains)"
        fig.text((x0 + 2.6) / W, 1 - TITLE_TOP / H, title, ha="left", va="top",
                 fontsize=sp.FS_BODY)  # fmt: skip
        fig.add_artist(Line2D([(x0 + 1.0) / W], [1 - (TITLE_TOP + 1.25) / H],
                              transform=fig.transFigure, **sp.marker_kw(key, ms=3.6)))  # fmt: skip
    fig.text(0.6 / W, 1 - LABEL_BASE / H, "Training tier", ha="left", va="baseline",
             fontsize=sp.FS_BODY)  # fmt: skip
    fig.text((FACET_X0 + FACET_W + FACET_GAP / 2) / W, 0.9 / H,
             "Spearman ρ with the test chains' G-scores",
             ha="center", va="bottom", fontsize=sp.FS_BODY)  # fmt: skip
    # key of the open squares, at the left end of the x-title row
    y_k = (0.9 + 1.25) / H
    fig.add_artist(Line2D([1.6 / W], [y_k], transform=fig.transFigure, ls="", marker="s",
                          ms=REP_MS, mew=0.9, mfc="white", mec=sp.INK_SOFT))  # fmt: skip
    fig.text(3.0 / W, y_k, "Ridge on the ceiling's residue pairs", ha="left", va="center",
             fontsize=sp.FS_BODY)  # fmt: skip
    sp.fig_label(fig, "c", 0.0, 0.0)
    return fig


def plot(B: pd.DataFrame, C: pd.DataFrame, TC: pd.DataFrame):
    """Supplementary Fig. S10, in the encodings of Fig. 5c (figure review items 5, 16): one
    row per training tier with horizontal labels, the ridge as a filled square in the tier
    colour, the amino-acid mean as an open diamond, the test chains' agreement with their own
    replicates as a black vertical line; grey band and dashed line, replicate agreement of
    the unfiltered to the strict tier."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    W, H = sp.W_DOUBLE, 92.0
    fig = sp.figure(W, H)

    def ax_mm(left, top, w, h):
        return fig.add_axes([left / W, 1 - (top + h) / H, w / W, h / H])

    # Spearman rho per residue (pooled) is main Fig. 5c (draw_main) since 12 Nov 2026;
    # S10 keeps the other three metrics (story blueprint 6.6)
    cols = [("auc_res", "AUC, G > 0.65\nper residue", "AUC"),
            ("rho_chain", "Spearman ρ\nmean per chain", "ρ"),
            ("auc_chain", "AUC, G > 0.65\nmean per chain", "AUC")]  # fmt: skip
    band, rep = "#E3E3E3", "#7a7a7a"
    chance = "#9A9A9A"  # AUC 0.5: thin solid line, labelled in the panel
    left0, gap, pw, ph = (
        25.0,
        6.5,
        39.5,
        22.0,
    )  # the last tick label stays on the canvas
    tops = (19.0, 58.0)
    y = np.arange(len(sp.TIERS))
    for i, name in enumerate(TESTS.values()):
        for j, (col, lab, xl) in enumerate(cols):
            ax = ax_mm(left0 + j * (pw + gap), tops[i], pw, ph)
            cs = C.set_index("tier")
            lo, hi = cs.loc["unfiltered", col], cs.loc["strict", col]
            ax.axvspan(min(lo, hi), max(lo, hi), color=band, lw=0, zorder=0)
            ax.axvline(hi, color=rep, lw=0.6, ls="--", zorder=1)
            tc = TC.set_index("test_set").loc[name, col]
            ax.axvline(tc, color=sp.INK, lw=1.0, zorder=2)
            for model in ("per-amino-acid mean", "one-hot ridge"):
                d = (
                    B[(B.test_set == name) & (B.model == model)]
                    .set_index("train_tier")
                    .loc[sp.TIERS]
                )
                for r, t in enumerate(sp.TIERS):
                    st = (AA_STYLE if model == "per-amino-acid mean" else
                          {"marker": "s", "mfc": sp.TIER_COLORS[t], "mec": sp.TIER_COLORS[t],
                           "color": sp.TIER_COLORS[t]})  # fmt: skip
                    ax.plot(
                        [d.loc[t, col]], [r], ls="", ms=3.4, mew=0.7, zorder=4, **st
                    )
            ax.set_ylim(len(sp.TIERS) - 0.5, -0.5)
            ax.set_yticks(y)
            ax.set_yticklabels(sp.TIERS if j == 0 else [])
            ax.tick_params(axis="y", length=0, pad=2.0)
            ax.spines["left"].set_visible(False)
            if col.startswith("auc"):
                ax.set_xlim(0.4, 1.0)
                xt = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                ax.set_xticks(xt, [f"{v:g}" for v in xt])
                ax.axvline(0.5, color=chance, lw=0.5, ls="-", zorder=1)
                # in the empty strip left of the line, so no mark is covered
                ax.text(0.485, 1.5, "chance", color=chance, fontsize=sp.FS_SMALL,
                        ha="center", va="center", rotation=90)  # fmt: skip
            else:
                ax.set_xlim(0.0, 1.0)
                ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "0.2", "0.4", "0.6", "0.8", "1"])  # fmt: skip
            if i == 0:
                ax.set_title(lab, fontsize=sp.FS_TICK, linespacing=1.15, pad=3)
            ax.set_xlabel(xl, labelpad=1.5)

    hand = [Line2D([], [], ls="", marker="s", ms=3.4, mew=0.7, color=sp.TIER_COLORS["moderate"],
                   label="Ridge (tier colour)"),
            Line2D([], [], ls="", ms=3.4, mew=0.7, label="Amino-acid mean", **AA_STYLE),
            Line2D([], [], color=sp.INK, lw=1.0, label="Test chains vs their own replicates"),
            Patch(facecolor=band, label="Replicates, unfiltered to strict"),
            Line2D([], [], color=rep, lw=0.6, ls="--", label="Replicates, strict")]  # fmt: skip
    fig.legend(handles=hand, loc="upper left", ncol=3, bbox_to_anchor=(left0 / W, 1.0),
               borderaxespad=0.2, handlelength=1.8, columnspacing=1.6)  # fmt: skip
    # row headers at the far left, rotated: which test set the row evaluates; the tier
    # labels of the first column carry the training tier
    for letter, i, name in (("a", 0, "TriZOD test set"), ("b", 1, "CheZOD117")):
        sp.fig_label(fig, letter, 0.0, tops[i] - (9.0 if i == 0 else 4.0))
        fig.text(4.2 / W, 1 - (tops[i] + ph / 2) / H, name, fontsize=sp.FS_BODY,
                 rotation=90, va="center", ha="center")  # fmt: skip
    fig.text(left0 / W - 1.0 / W, 1 - (tops[0] - 2.2) / H, "Training tier", ha="right",
             va="bottom", fontsize=sp.FS_BODY)  # fmt: skip
    return fig


def main():
    ap = sp.cli(__doc__)
    ap.add_argument(
        "--n-boot", type=int, default=N_BOOT, help="bootstrap resamples (Fig. 5c)"
    )
    args = sp.parse_args(ap)
    DATA_DIR = sp.out_dir(DATA_SUBDIR)
    df = sp.load_main(COLUMNS)
    preds = predictions(df)
    B = baselines(df, preds)
    C = tier_ceilings(df)
    pairs = test_ceiling_pairs(df)
    TC = test_ceilings(df, pairs=pairs)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    B.to_csv(DATA_DIR / "S10_baselines.csv", index=False)
    C.to_csv(DATA_DIR / "S10_replicate_ceiling_by_tier.csv", index=False)
    TC.to_csv(DATA_DIR / "S10_replicate_ceiling_test_sets.csv", index=False)
    summ = {
        "ridge_lambda": LAMBDA,
        "window": 2 * W + 1,
        "ridge_pearson_range": {n: [round(B[(B.test_set == n) & (B.model == "one-hot ridge")].pearson_res.min(), 3),
                                    round(B[(B.test_set == n) & (B.model == "one-hot ridge")].pearson_res.max(), 3)]
                                for n in TESTS.values()},
        "aamean_pearson_range": {n: [round(B[(B.test_set == n) & (B.model == "per-amino-acid mean")].pearson_res.min(), 3),
                                     round(B[(B.test_set == n) & (B.model == "per-amino-acid mean")].pearson_res.max(), 3)]
                                 for n in TESTS.values()},
    }  # fmt: skip
    (DATA_DIR / "S10_summary.json").write_text(json.dumps(summ, indent=2))
    out = sp.save(plot(B, C, TC), "S10")
    # Fig. 5c and its source data
    M = pd.concat([main_panel_data(preds, pairs, args.n_boot),
                   ridge_on_replicate_rows(preds, pairs, args.n_boot)], ignore_index=True)  # fmt: skip
    fig5 = sp.out_dir(FIG5_SUBDIR)
    fig5.mkdir(parents=True, exist_ok=True)
    M.to_csv(fig5 / "fig5c_rho.csv", index=False)
    (fig5 / "fig5c_numbers.json").write_text(json.dumps({
        "metric": "pooled residue-level Spearman rho between prediction and the test chains' "
                  "released G-scores; ceiling = sqrt(rho) of test chain vs its replicates",
        "ci": f"95 % percentile bootstrap over test chains ({args.n_boot} resamples, seed "
              f"{BOOT_SEED}); for the ceiling a test chain is resampled with all its pairs",
        "ridge_rho_on_replicate_test_chains": ridge_on_replicate_chains(preds, pairs),
        "rows": M.round(4).to_dict("records"),
    }, indent=1))  # fmt: skip
    out_c = sp.save(draw_main(M), "fig5c_baselines")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(out)
    print(json.dumps(summ, indent=1))
    print(B.round(3).to_string())
    print(C.round(3).to_string())
    print(TC.round(3).to_string())
    print(M.round(3).to_string())
    print(out_c)


if __name__ == "__main__":
    main()
