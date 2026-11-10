#!/usr/bin/env python3
"""Supplementary Fig. S10 -- reference baselines and headroom.

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

Headroom (grey band): replicate agreement, i.e. how well ANOTHER measurement of the
same protein predicts a label. Pairs of non-bound identical-sequence chains from
different entries (<= 20 per sequence, seed 0; the residue-level-r sample of
Supplementary Fig. S7), scored in the same metrics with each chain's G used as the
predictor of the other's label (both directions). The band spans the unfiltered to
the strict cumulative tier; the line marks strict. For rho the dotted line is
sqrt(rho_rep), the attenuation ceiling for any predictor of a single noisy
measurement. Replicate agreement of the test chains themselves (test chain vs its
replicates outside the test set) is reported in the summary JSON.

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


def baselines(df: pd.DataFrame) -> pd.DataFrame:
    tests = {}
    for split, name in TESTS.items():
        recs = []
        for s, gs in zip(
            df.loc[df.split == split, "sequence"], df.loc[df.split == split, "gscores"]
        ):
            g = np.asarray(gs, float)
            m = ~np.isnan(g)
            X, a = feats(s)
            recs.append((X[m], g[m], a[m]))
        tests[name] = recs
    rows = []
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
                m_ = metrics([(y, pred(X, a)) for X, y, a in recs])
                rows.append(
                    {
                        "train_tier": t,
                        "n_train_chains": len(tr),
                        "test_set": name,
                        "model": model,
                        **m_,
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


def test_ceilings(df: pd.DataFrame, cap: int = 20) -> pd.DataFrame:
    """Replicate agreement of the test chains themselves: each test chain's label
    against non-bound identical-sequence chains from other entries (outside the test sets)."""
    rng = random.Random(0)
    nb = df[~df["is_bound"].astype(bool) & ~df["split"].isin(list(TESTS))]
    # dict(groupby) would call GroupBy.keys, which is the grouping key, not a method
    by_seq = {s: g for s, g in nb.groupby("sequence")}  # noqa: C416
    rows = []
    for split, name in TESTS.items():
        chains, n_with = [], 0
        for _, r in df[df.split == split].iterrows():
            g = by_seq.get(r.sequence)
            if g is None:
                continue
            reps = [x for x in g.to_dict("records") if x["entry_id"] != r.entry_id]
            rng.shuffle(reps)
            got = 0
            y = np.asarray(r.gscores, float)
            for x in reps[:cap]:
                p = np.asarray(x["gscores"], float)
                m = ~np.isnan(y) & ~np.isnan(p)
                if m.sum() < s7.MIN_PAIR_RES:
                    continue
                chains.append((y[m], p[m]))
                got += 1
            n_with += got > 0
        m_ = metrics(chains) if chains else {}
        rows.append(
            {
                "test_set": name,
                "n_test_chains_with_replicates": n_with,
                "n_pairs": len(chains),
                **m_,
            }
        )
    return pd.DataFrame(rows)


def plot(B: pd.DataFrame, C: pd.DataFrame, TC: pd.DataFrame):
    fig = sp.figure(sp.W_DOUBLE, 92)
    cols = [("rho_res", "Spearman ρ, per residue"), ("auc_res", "AUC (G > 0.65), per residue"),
            ("rho_chain", "Spearman ρ, mean per chain"), ("auc_chain", "AUC (G > 0.65), mean per chain")]  # fmt: skip
    x = np.arange(len(sp.TIERS))
    mstyle = {"per-amino-acid mean": {"color": sp.CAT["light"], "marker": "s", "mec": "#7a7a7a"},
              "one-hot ridge": {"color": sp.OKABE_ITO["blue"], "marker": "o", "mec": sp.OKABE_ITO["blue"]}}  # fmt: skip
    axes = {}
    for i, name in enumerate(TESTS.values()):
        for j, (col, lab) in enumerate(cols):
            ax = fig.add_axes([0.075 + j * 0.238, 0.525 - i * 0.445, 0.18, 0.325])
            axes[(i, j)] = ax
            cs = C.set_index("tier")
            lo, hi = cs.loc["unfiltered", col], cs.loc["strict", col]
            ax.axhspan(min(lo, hi), max(lo, hi), color="#E3E3E3", lw=0, zorder=0)
            ax.axhline(hi, color="#7a7a7a", lw=0.6, ls="--", zorder=1)
            if col == "rho_res":
                ax.axhline(
                    cs.loc["strict", "sqrt_rho_res"],
                    color="#7a7a7a",
                    lw=0.6,
                    ls=":",
                    zorder=1,
                )
            for model, st in mstyle.items():
                d = (
                    B[(B.test_set == name) & (B.model == model)]
                    .set_index("train_tier")
                    .loc[sp.TIERS]
                )
                ax.plot(x, d[col], lw=0.8, ms=3, mew=0.5, **st, zorder=3)
            tc = TC.set_index("test_set").loc[name, col]
            ax.plot(
                [3.55],
                [tc],
                marker="D",
                ms=3,
                color="black",
                mfc="white",
                mew=0.6,
                ls="none",
                zorder=4,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(["unf.", "tol.", "mod.", "strict"])
            ax.set_xlim(-0.35, 3.8)
            if col.startswith("auc"):
                ax.set_ylim(0.45, 1.0)
                ax.axhline(0.5, color="#BDBDBD", lw=0.4, ls=":")
            else:
                ax.set_ylim(0.0, 1.0)
            if i == 0:
                ax.set_title(lab, fontsize=6)
            if j == 0:
                ax.set_ylabel(f"{name}\n{'ρ' if 'rho' in col else 'AUC'}")
            else:
                ax.set_ylabel("ρ" if "rho" in col else "AUC")
            if i == 1:
                ax.set_xlabel("Training tier (cumulative)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    hand = [Line2D([], [], lw=0.8, ms=3, mew=0.5, label="one-hot ridge (15-residue window)", **mstyle["one-hot ridge"]),
            Line2D([], [], lw=0.8, ms=3, mew=0.5, label="per-amino-acid mean (floor)", **mstyle["per-amino-acid mean"]),
            Patch(facecolor="#E3E3E3", label="replicate agreement, unfiltered → strict"),
            Line2D([], [], color="#7a7a7a", lw=0.6, ls="--", label="replicate agreement, strict"),
            Line2D([], [], color="#7a7a7a", lw=0.6, ls=":", label="√ρ ceiling, strict"),
            Line2D([], [], marker="D", ms=3, color="black", mfc="white", mew=0.6, ls="none",
                   label="test chains vs their own replicates")]  # fmt: skip
    fig.legend(handles=hand, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.3,
               handlelength=1.8, columnspacing=0.9)  # fmt: skip
    sp.fig_label(fig, "a", 1.5, 14)
    sp.fig_label(fig, "b", 1.5, 55)
    return fig


def main():
    sp.parse_args(sp.cli(__doc__))
    DATA_DIR = sp.out_dir(DATA_SUBDIR)
    df = sp.load_main(COLUMNS)
    B = baselines(df)
    C = tier_ceilings(df)
    TC = test_ceilings(df)
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
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(out)
    print(json.dumps(summ, indent=1))
    print(B.round(3).to_string())
    print(C.round(3).to_string())
    print(TC.round(3).to_string())


if __name__ == "__main__":
    main()
