#!/usr/bin/env python3
"""Supplementary Fig. S7 -- replicate analyses in full.

The BMRB holds many independent measurements of the same protein. This figure gives
every replicate statistic the main text quotes, in both tier definitions, so that no
headline rests on one choice of stratification.

Definitions (identical to ``scripts/manuscript/make_numbers.py``, which writes the
``\\repDev*``, ``\\rRepResidue*`` and ``\\kpair*`` macros, so the figure and the prose
quote the same numbers):

* **Tier membership.** A chain's *strictest tier* is the strictest tier whose
  ``scores/<tier>/scores.json`` scores it. *Cumulative* tier t (what a user selecting t
  gets) holds every chain whose strictest tier is t or stricter. The *exclusive*
  stratum t holds only chains whose strictest tier is t; its "unfiltered" stratum
  therefore holds only chains that FAIL the tolerant filters, and is never a
  "without filtering" baseline.
* **Leave-one-out (LOO) deviation** (port of ``trust_first/rep_tier_k.py``). Sequences
  scored in >= 3 distinct BMRB entries (all chains, bound or not); for each chain the
  per-residue consensus is the median G of the group's other chains; a residue is
  comparable when the chain is scored there and >= 2 other chains are. Chains need
  >= 5 comparable residues. Deviation = mean |G - consensus|; flip = share of residues
  on opposite sides of G = 0.65; chain k = median shift count over comparable residues.
  Labels are the release (unfiltered) labels; a sensitivity check re-reads each
  chain's own tier label (only the tolerant tier differs, writing-plan fix (c)).
* **Residue-level replicate r** (port of ``user_first/ceiling_and_baseline.py``).
  Non-bound chains with identical sequence from different entries; candidate pairs
  shuffled once per sequence (``random.Random(0)``) and capped at 20 per sequence;
  a pair needs >= 10 co-scored residues and counts at every tier that contains both
  chains. r is the pooled Pearson correlation over residues. The *variance-matched*
  r re-weights residues so that the distribution of the pair-mean G (10 bins on
  [0, 1]) equals the strict tier's, a check that a higher r is not just a narrower
  or wider G range. The cap sensitivity re-runs the same shuffled order with caps
  5 / 10 / 20 / 50 / none.
* **Leave-pair-out paired-k test.** The pair sample of Fig. 2a (``fig2a_kpaired.py``:
  non-bound chains, <= 30 pairs per sequence, seed 0, different entries), restricted
  to sequences with >= 3 chains. Each residue's ordered (< 0.3) / disordered (> 0.7)
  state comes either from the consensus over all chains of the sequence (Fig. 2a) or
  from the consensus over all chains EXCEPT the two in the pair (leave-pair-out),
  which removes the weak circularity of a state defined partly by the pair itself.

Bootstrap 95 % confidence intervals resample sequences (1,000 replicates, seed 0),
because chains and pairs of one sequence are not independent.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s7_replicates.py \
        --out-dir <draft>/Figures/paper
Outputs: <out-dir>/S7.pdf and <out-dir>/suppfigs_b_data/S7_*.csv / S7_summary.json.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig3_replicates as fr  # noqa: E402
import style_paper as sp  # noqa: E402

SEED = 0
N_BOOT = 1000
MIN_LOO_RES = 5
MIN_PAIR_RES = 10
PAIR_CAP = 20
CAPS = [5, 10, 20, 50, None]
K_EDGES = [0, 12, 17, 21]
K_LABELS = ["≤12", "13–17", "18–21"]
# paired-k test (Fig. 2a definitions)
KP_MAX_PAIRS = 30
KP_ORD, KP_DIS = 0.3, 0.7
KP_EDGES = [-0.5, 0.5, 3, 6, 9, 21]
KP_LABELS = ["0", "1–3", "4–6", "7–9", "≥10"]

DATA_SUBDIR = "suppfigs_b_data"  # under --out-dir
COLUMNS = [
    "id",
    "entry_id",
    "sequence",
    "is_bound",
    "gscores",
    "zscores",
    "k",
    "ph",
    "temperature",
]


# ------------------------------------------------------------------------------ data
def load():
    df = sp.load_main(COLUMNS)
    tier_ids = sp.load_tier_ids()
    strictest = sp.strictest_tier(tier_ids)
    df["tier"] = df["id"].map(strictest)
    df["trank"] = df["tier"].map(sp.RANK)
    assert df["tier"].notna().all()
    return df, tier_ids


def loo_table(df: pd.DataFrame, tier_labels: dict | None = None) -> pd.DataFrame:
    """Per-chain LOO deviation (release labels); with ``tier_labels`` also the deviation
    of the chain's own tier-t label at every cumulative tier t it belongs to."""
    rows = []
    grp = df.groupby("sequence").filter(lambda g: g.entry_id.nunique() >= 3)
    for seq, g in grp.groupby("sequence"):
        G = np.vstack([np.asarray(x, float) for x in g.gscores])
        K = np.vstack([np.asarray(x, float) for x in g.k])
        ids = list(g["id"])
        for j in range(len(g)):
            others = np.delete(G, j, axis=0)
            with np.errstate(all="ignore"):
                n_ok = (~np.isnan(others)).sum(0)
                cons = np.nanmedian(others, axis=0)
            m = ~np.isnan(G[j]) & (n_ok >= 2)
            if m.sum() < MIN_LOO_RES:
                continue
            d = np.abs(G[j][m] - cons[m])
            row = {
                "id": ids[j],
                "sequence": seq,
                "tier": g["tier"].iloc[j],
                "trank": int(g["trank"].iloc[j]),
                "bound": bool(g["is_bound"].iloc[j]),
                "dev": float(d.mean()),
                "flip": float(np.mean((G[j][m] > sp.G_DIS) != (cons[m] > sp.G_DIS))),
                "medk": float(np.nanmedian(K[j][m])),
                "n_res": int(m.sum()),
            }
            if tier_labels is not None:
                base = n_ok >= 2
                for t in sp.TIERS[: row["trank"] + 1]:
                    own = tier_labels[t][ids[j]]["gscores"]
                    mt = base & ~np.isnan(own)
                    row[f"dev_{t}"] = (
                        float(np.abs(own[mt] - cons[mt]).mean())
                        if mt.sum() >= MIN_LOO_RES
                        else np.nan
                    )
            rows.append(row)
    L = pd.DataFrame(rows)
    L["kbin"] = pd.cut(L["medk"], K_EDGES, labels=K_LABELS)
    return L


def stratum(L: pd.DataFrame, t: str, mode: str) -> pd.DataFrame:
    j = sp.RANK[t]
    return L[L.trank == j] if mode == "exclusive" else L[L.trank >= j]


def boot_median(sub: pd.DataFrame, col="dev"):
    if sub.empty:
        return np.nan, np.nan, np.nan
    groups = [v[col].to_numpy() for _, v in sub.groupby("sequence", sort=False)]
    rng = np.random.default_rng(SEED)
    meds = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        meds[b] = np.median(np.concatenate([groups[i] for i in pick]))
    return (
        float(sub[col].median()),
        float(np.percentile(meds, 2.5)),
        float(np.percentile(meds, 97.5)),
    )


def boot_share(sub: pd.DataFrame, values: np.ndarray):
    """Mean of ``values`` (one per chain) with a sequence-bootstrap CI."""
    if sub.empty:
        return np.nan, np.nan, np.nan
    s = pd.DataFrame({"seq": sub["sequence"].to_numpy(), "v": values})
    agg = s.groupby("seq", sort=False)["v"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(SEED)
    pick = rng.integers(0, len(agg), (N_BOOT, len(agg)))
    bs = agg[pick, 0].sum(1) / agg[pick, 1].sum(1)
    return (
        float(values.mean()),
        float(np.percentile(bs, 2.5)),
        float(np.percentile(bs, 97.5)),
    )


def stratum_fig3(Lf: pd.DataFrame, t: str, mode: str) -> pd.DataFrame:
    """Fig. 3a table (one row per chain and cumulative tier): cumulative = rows of tier t;
    exclusive = rows of tier t whose strictest tier is t."""
    sub = Lf[Lf.tier == t]
    return sub if mode == "cumulative" else sub[sub.strictest == t]


def definitions(L: pd.DataFrame, Lf: pd.DataFrame) -> dict:
    return {"fig3": (Lf, stratum_fig3), "preview": (L, stratum)}


def summarise_loo(defs: dict) -> pd.DataFrame:
    rows = []
    for scope, (S, strat) in defs.items():
        for mode in ("exclusive", "cumulative"):
            for t in sp.TIERS:
                sub = strat(S, t, mode)
                med, lo, hi = boot_median(sub)
                gt, gt_lo, gt_hi = boot_share(sub, (sub.dev > 0.1).to_numpy(float))
                fl, fl_lo, fl_hi = boot_share(sub, sub.flip.to_numpy(float))
                row = {"scope": scope, "mode": mode, "tier": t, "n_chains": len(sub),
                       "n_sequences": sub.sequence.nunique(), "median_dev": med,
                       "median_dev_lo": lo, "median_dev_hi": hi, "share_dev_gt_0.1": gt,
                       "share_lo": gt_lo, "share_hi": gt_hi, "flip": fl, "flip_lo": fl_lo,
                       "flip_hi": fl_hi}  # fmt: skip
                if mode == "cumulative" and f"dev_{t}" in sub:
                    row["median_dev_tier_labels"] = float(sub[f"dev_{t}"].median())
                rows.append(row)
    return pd.DataFrame(rows)


def k_table(defs: dict) -> pd.DataFrame:
    rows = []
    for scope, (S, strat) in defs.items():
        for mode in ("exclusive", "cumulative"):
            for t in sp.TIERS:
                sub = strat(S, t, mode)
                for kb in K_LABELS:
                    s = sub[sub.kbin == kb]
                    rows.append({"scope": scope, "mode": mode, "tier": t, "kbin": kb,
                                 "n_chains": len(s), "median_dev": float(s.dev.median()) if len(s) else np.nan})  # fmt: skip
    return pd.DataFrame(rows)


# -------------------------------------------------------------- residue-level pairs
def replicate_pairs(df: pd.DataFrame, cap: int | None = PAIR_CAP):
    """Pairs as in make_numbers.py (non-bound, different entries, seeded shuffle, cap).

    Returns a list of dicts with the co-scored residues and the strictest tier both
    chains share."""
    rng = random.Random(SEED)
    out = []
    nb = df[~df["is_bound"].astype(bool)]
    for seq, sub in nb.groupby("sequence"):
        if sub["entry_id"].nunique() < 2:
            continue
        rows_ = sub.to_dict("records")
        cand = [
            (a, b)
            for ia, a in enumerate(rows_)
            for b in rows_[ia + 1 :]
            if a["entry_id"] != b["entry_id"]
        ]
        rng.shuffle(cand)
        for a, b in cand if cap is None else cand[:cap]:
            ga, gb = np.asarray(a["gscores"], float), np.asarray(b["gscores"], float)
            m = ~np.isnan(ga) & ~np.isnan(gb)
            if m.sum() < MIN_PAIR_RES:
                continue
            both = min(a["trank"], b["trank"])
            same = (
                abs(a["ph"] - b["ph"]) <= 0.2
                and abs(a["temperature"] - b["temperature"]) <= 2
            )
            diff = (
                abs(a["ph"] - b["ph"]) >= 1
                or abs(a["temperature"] - b["temperature"]) >= 10
            )
            out.append({"sequence": seq, "a": a["id"], "b": b["id"], "both": int(both),
                        "x": ga[m], "y": gb[m], "same": bool(same), "diff": bool(diff)})  # fmt: skip
    return out


N_GBINS = 10


def pair_bin_stats(p) -> np.ndarray:
    """(N_GBINS, 6) sums n, sx, sy, sxx, syy, sxy per pair-mean-G bin."""
    x, y = p["x"], p["y"]
    b = np.clip(((x + y) / 2 * N_GBINS).astype(int), 0, N_GBINS - 1)
    S = np.zeros((N_GBINS, 6))
    for col, v in enumerate((np.ones_like(x), x, y, x * x, y * y, x * y)):
        np.add.at(S[:, col], b, v)
    return S


def r_from(S: np.ndarray, wbin: np.ndarray | None = None) -> float:
    """Pooled Pearson r from per-bin sums (optionally re-weighted per bin)."""
    w = np.ones(N_GBINS) if wbin is None else wbin
    n, sx, sy, sxx, syy, sxy = (S * w[:, None]).sum(0)
    cov = sxy / n - sx * sy / n**2
    return float(cov / np.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2)))


def tier_r_table(pairs, ref_tier="strict") -> tuple[pd.DataFrame, dict]:
    """Raw and variance-matched r per cumulative tier with sequence-bootstrap CIs."""
    seqs = np.array([p["sequence"] for p in pairs])
    both = np.array([p["both"] for p in pairs])
    BS = np.stack([pair_bin_stats(p) for p in pairs])  # (pairs, bins, 6)
    per_tier = {}
    for t in sp.TIERS:
        m = both >= sp.RANK[t]
        uniq, inv = np.unique(seqs[m], return_inverse=True)
        SW = np.zeros((len(uniq), N_GBINS, 6))
        np.add.at(SW, inv, BS[m])
        per_tier[t] = SW
    ref = per_tier[ref_tier].sum(0)[:, 0]
    pref = ref / ref.sum()
    rows = []
    rng = np.random.default_rng(SEED)
    for t in sp.TIERS:
        SW = per_tier[t]
        tot = SW.sum(0)
        p = tot[:, 0] / tot[:, 0].sum()
        wbin = np.divide(pref, p, out=np.zeros_like(p), where=p > 0)
        est, est_vm = r_from(tot), r_from(tot, wbin)
        bs, bs_vm = [], []
        for _ in range(N_BOOT):
            S = SW[rng.integers(0, len(SW), len(SW))].sum(0)
            pb = S[:, 0] / S[:, 0].sum()
            wb = np.divide(pref, pb, out=np.zeros_like(pb), where=pb > 0)
            bs.append(r_from(S))
            bs_vm.append(r_from(S, wb))
        rows.append({"tier": t, "n_pairs": int((both >= sp.RANK[t]).sum()), "n_sequences": len(SW),
                     "n_residues": int(tot[:, 0].sum()), "r": est, "r_lo": np.percentile(bs, 2.5),
                     "r_hi": np.percentile(bs, 97.5), "r_varmatched": est_vm,
                     "r_varmatched_lo": np.percentile(bs_vm, 2.5), "r_varmatched_hi": np.percentile(bs_vm, 97.5),
                     "share_pairmeanG_gt_0.65": float(p[int(0.65 * N_GBINS) + 1 :].sum() + 0.5 * p[int(0.65 * N_GBINS)])})  # fmt: skip
    return pd.DataFrame(rows), {
        "ref_tier": ref_tier,
        "ref_bin_share": pref.round(4).tolist(),
    }


def cap_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cap in CAPS:
        pairs = replicate_pairs(df, cap)
        both = np.array([p["both"] for p in pairs])
        for t in sp.TIERS:
            sel = [p for p, b in zip(pairs, both) if b >= sp.RANK[t]]
            x = np.concatenate([p["x"] for p in sel])
            y = np.concatenate([p["y"] for p in sel])
            rows.append({"cap": "none" if cap is None else cap, "tier": t, "n_pairs": len(sel),
                         "r": float(np.corrcoef(x, y)[0, 1])})  # fmt: skip
    return pd.DataFrame(rows)


def fig3_residue_r(D) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fig. 3a residue-level r (pairs capped per sequence AND tier) with sequence-bootstrap
    CIs and the r variance-matched to the unfiltered tier; plus the cap sensitivity."""
    P = fr.pair_table(D, cap=fr.PAIR_CAP)
    ref = P[P.tier == "unfiltered"]
    rows = []
    for t in sp.TIERS:
        sub = P[P.tier == t]
        b = fr.boot_pooled(sub)
        b.update(
            {"tier": t, "r_varmatched_unfiltered": fr.variance_matched_r(sub, ref)}
        )
        rows.append(b)
    caps = []
    for cap in CAPS:
        Pc = (
            P
            if cap == fr.PAIR_CAP
            else fr.pair_table(D, cap=10**9 if cap is None else cap)
        )
        for t in sp.TIERS:
            sub = Pc[Pc.tier == t]
            caps.append({"cap": "none" if cap is None else cap, "tier": t, "n_pairs": len(sub),
                         "r": float(fr.pooled(sub[fr.STAT_COLS].to_numpy(float))["r"])})  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(caps)


# --------------------------------------------------------- leave-pair-out paired k
def kpaired_lpo(df: pd.DataFrame) -> pd.DataFrame:
    """Fig. 2a pair sample; per residue: dk, dZ, dG, consensus with and without the pair."""
    nb = df[~df["is_bound"].astype(bool)]
    rng = random.Random(SEED)
    out = []
    for _seq, g in nb.groupby("sequence"):
        if g.entry_id.nunique() < 2:
            continue
        G = np.vstack([np.asarray(x, float) for x in g.gscores])
        Z = np.vstack([np.asarray(x, float) for x in g.zscores])
        K = np.vstack([np.asarray(x, float) for x in g.k])
        with np.errstate(all="ignore"):
            cons = np.nanmedian(G, axis=0)
        pairs = list(itertools.combinations(range(len(g)), 2))
        if len(pairs) > KP_MAX_PAIRS:
            pairs = rng.sample(pairs, KP_MAX_PAIRS)
        ent = g.entry_id.values
        for a, b in pairs:
            if ent[a] == ent[b]:
                continue
            m = ~np.isnan(G[a]) & ~np.isnan(G[b]) & ~np.isnan(Z[a]) & ~np.isnan(Z[b])
            if m.sum() == 0:
                continue
            rest = [i for i in range(len(g)) if i not in (a, b)]
            if rest:
                with np.errstate(all="ignore"):
                    cons_lpo = np.nanmedian(G[rest], axis=0)
            else:
                cons_lpo = np.full(G.shape[1], np.nan)
            hi = np.where(K[a] >= K[b], a, b)
            lo = np.where(K[a] >= K[b], b, a)
            ix = np.where(m)[0]
            out.append(pd.DataFrame({"dk": np.abs(K[a, ix] - K[b, ix]), "dZ": Z[hi[ix], ix] - Z[lo[ix], ix],
                                     "dG": G[hi[ix], ix] - G[lo[ix], ix], "cons": cons[ix],
                                     "cons_lpo": cons_lpo[ix]}))  # fmt: skip
    R = pd.concat(out, ignore_index=True)
    R["dkbin"] = pd.cut(R.dk, KP_EDGES, labels=KP_LABELS)
    return R


def state(c: pd.Series) -> np.ndarray:
    return np.where(
        c < KP_ORD, "ordered", np.where(c > KP_DIS, "disordered", "intermediate")
    )


def summarise_kp(R: pd.DataFrame) -> pd.DataFrame:
    rows = []
    has_lpo = R.cons_lpo.notna()
    variants = {
        "all pairs, consensus incl. pair (Fig. 2a)": (R, state(R.cons)),
        "same residues, consensus incl. pair": (R[has_lpo], state(R.cons[has_lpo])),
        "same residues, leave-pair-out consensus": (
            R[has_lpo],
            state(R.cons_lpo[has_lpo]),
        ),
    }
    for name, (S, st) in variants.items():
        for s_ in ("ordered", "disordered"):
            sub = S[st == s_]
            for lab in KP_LABELS:
                b = sub[sub.dkbin == lab]
                rows.append({"variant": name, "state": s_, "dk_bin": lab, "n": len(b),
                             "dZ_median": b.dZ.median(), "dG_median": b.dG.median()})  # fmt: skip
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ plot
def plot(
    S: pd.DataFrame,
    KT: pd.DataFrame,
    RT: pd.DataFrame,
    RF: pd.DataFrame,
    CS: pd.DataFrame,
    KP: pd.DataFrame,
):
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    fig = sp.figure(sp.W_DOUBLE, 150)
    x = np.arange(len(sp.TIERS))
    mode_style = {
        "exclusive": {"mfc": "white", "dx": -0.13, "ls": ":"},
        "cumulative": {"mfc": None, "dx": 0.13, "ls": "-"},
    }
    scope_col = {"preview": sp.CAT["neutral"], "fig3": sp.OKABE_ITO["blue"]}

    # ---- row 1: a median deviation, b share > 0.1, c flip rate
    top = 0.955
    axes1 = [
        fig.add_axes([0.065 + i * 0.325, 0.73, 0.255, top - 0.73]) for i in range(3)
    ]
    metrics = [("median_dev", "median_dev_lo", "median_dev_hi", "Median LOO deviation |G − consensus|", 1),
               ("share_dev_gt_0.1", "share_lo", "share_hi", "Chains deviating > 0.1 (%)", 100),
               ("flip", "flip_lo", "flip_hi", "Residues flipping at G = 0.65 (%)", 100)]  # fmt: skip
    for ax, (col, lo, hi, lab, fac) in zip(axes1, metrics):
        for scope, ddx in (("preview", -0.03), ("fig3", 0.03)):
            for mode, st in mode_style.items():
                d = (
                    S[(S["scope"] == scope) & (S["mode"] == mode)]
                    .set_index("tier")
                    .loc[sp.TIERS]
                )
                xx = x + st["dx"] + ddx
                y = d[col].to_numpy() * fac
                yerr = np.vstack(
                    [y - d[lo].to_numpy() * fac, d[hi].to_numpy() * fac - y]
                )
                ax.errorbar(xx, y, yerr=yerr, color=scope_col[scope], ls=st["ls"], lw=0.7, marker="o",
                            ms=2.6, mfc=st["mfc"] or scope_col[scope], mew=0.6, capsize=1.0, elinewidth=0.5)  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels(sp.TIERS)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylabel(lab)
        ax.set_ylim(bottom=0)
    hand = [Line2D([], [], color=sp.OKABE_ITO["blue"], marker="o", ms=2.6, lw=0.7,
                   label="Fig. 3a: non-bound, leave-one-entry-out"),
            Line2D([], [], color=sp.CAT["neutral"], marker="o", ms=2.6, lw=0.7,
                   label="preview: all chains, leave-one-chain-out"),
            Line2D([], [], color="#7a7a7a", marker="o", ms=2.6, lw=0.7, ls=":", mfc="white", label="exclusive stratum"),
            Line2D([], [], color="#7a7a7a", marker="o", ms=2.6, lw=0.7, label="cumulative tier")]  # fmt: skip
    axes1[0].legend(
        handles=hand, loc="upper right", ncol=1, handlelength=2.0, borderaxespad=0.1
    )
    for ax, l in zip(axes1, "abc"):
        sp.panel_label(ax, l, x=-0.22, y=1.0)

    # ---- row 2: d tier x k heat maps
    cmap = mpl.colormaps["OrRd"]
    norm = mpl.colors.LogNorm(vmin=0.02, vmax=0.25)
    hm_specs = [
        ("fig3", "exclusive"),
        ("fig3", "cumulative"),
        ("preview", "exclusive"),
        ("preview", "cumulative"),
    ]
    hm_title = {"fig3": "Fig. 3a definition", "preview": "preview definition"}
    hm_axes = []
    for i, (scope, mode) in enumerate(hm_specs):
        ax = fig.add_axes([[0.07, 0.255, 0.52, 0.705][i], 0.42, 0.175, 0.215])
        hm_axes.append(ax)
        T = KT[(KT["scope"] == scope) & (KT["mode"] == mode)]
        M = (
            T.pivot(index="tier", columns="kbin", values="median_dev")
            .loc[sp.TIERS, K_LABELS]
            .to_numpy()
        )
        N = (
            T.pivot(index="tier", columns="kbin", values="n_chains")
            .loc[sp.TIERS, K_LABELS]
            .to_numpy()
        )
        ax.imshow(
            np.where(np.isfinite(M), M, np.nan), cmap=cmap, norm=norm, aspect="auto"
        )
        for r_ in range(4):
            for c_ in range(3):
                if N[r_, c_] == 0:
                    ax.text(c_, r_, "–", ha="center", va="center", fontsize=5)
                    continue
                v = M[r_, c_]
                col = "white" if v > 0.09 else "black"
                ax.text(
                    c_,
                    r_ - 0.12,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=col,
                )
                ax.text(
                    c_,
                    r_ + 0.24,
                    f"n = {int(N[r_, c_]):,}",
                    ha="center",
                    va="center",
                    fontsize=4.2,
                    color=col,
                )
        ax.set_xticks(range(3))
        ax.set_xticklabels(K_LABELS)
        ax.set_yticks(range(4))
        ax.set_yticklabels(sp.TIERS if i in (0, 2) else [])
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_xlabel("Chain median k")
        ax.set_title(f"{mode}, {hm_title[scope]}", fontsize=6)
    cax = fig.add_axes([0.905, 0.45, 0.008, 0.16])
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        ticks=[0.02, 0.05, 0.1, 0.2],
    )
    cb.ax.set_yticklabels(["0.02", "0.05", "0.1", "0.2"])
    cb.ax.minorticks_off()
    cb.set_label("Median LOO deviation", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    sp.fig_label(fig, "d", x_mm=1.6, y_mm_from_top=49.5)

    # ---- row 3: e residue-level r, f cap sensitivity, g/h leave-pair-out k test
    y0, h = 0.065, 0.24
    axe = fig.add_axes([0.065, y0, 0.20, h])
    series = [(RF, "r", "r_lo", "r_hi", "Fig. 3a pooled r", "o", None, sp.OKABE_ITO["blue"], -0.12),
              (RF, "r_varmatched_unfiltered", None, None, "Fig. 3a, variance-matched", "s", "white",
               sp.OKABE_ITO["blue"], 0.0),
              (RT, "r", "r_lo", "r_hi", "preview pooled r", "^", "white", sp.CAT["neutral"], 0.12)]  # fmt: skip
    for T, col, lo, hi, lab, mk, mfc, c, dx in series:
        y = T.set_index("tier").loc[sp.TIERS, col].to_numpy()
        yerr = None
        if lo is not None:
            Ti = T.set_index("tier").loc[sp.TIERS]
            yerr = np.vstack([y - Ti[lo].to_numpy(), Ti[hi].to_numpy() - y])
        axe.errorbar(x + dx, y, yerr=yerr, color=c, lw=0.7, marker=mk, ms=2.8, mfc=mfc or c, mew=0.6,
                     capsize=1.0, elinewidth=0.5, label=lab)  # fmt: skip
    RFi = RF.set_index("tier").loc[sp.TIERS]
    for xi, (n, s_) in enumerate(zip(RFi["n_pairs"], RFi["n_sequences"])):
        axe.text(
            xi,
            0.405,
            f"{n:,}\n({s_})",
            ha="center",
            va="bottom",
            fontsize=4.3,
            color="#555555",
        )
    axe.set_xticks(x)
    axe.set_xticklabels(sp.TIERS)
    axe.set_xlim(-0.5, 3.5)
    axe.set_ylim(0.4, 1.0)
    axe.set_ylabel("Residue-level replicate r (G)")
    axe.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.13),
        borderaxespad=0.1,
        handletextpad=0.3,
        handlelength=1.4,
        fontsize=5,
    )  # shorter label + handles keep the legend off the y axis (final pass, 31 Oct)
    axe.set_xlabel("Cumulative tier (Fig. 3a pairs, sequences)")
    sp.panel_label(axe, "e", x=-0.3, y=1.0)

    axf = fig.add_axes([0.335, y0, 0.15, h])
    caps = [str(c) for c in CS["cap"].unique()]
    for t in sp.TIERS:
        d = CS[CS.tier == t]
        axf.plot(
            range(len(caps)),
            d["r"],
            color=sp.TIER_COLORS[t],
            marker=sp.TIER_MARKERS[t],
            ms=2.5,
            lw=0.7,
        )
    axf.set_xticks(range(len(caps)))
    axf.set_xticklabels(caps)
    axf.set_xlabel("Pairs per sequence and tier (cap)")
    axf.set_ylabel("Residue-level replicate r (G)")
    axf.set_ylim(0.5, 1.0)
    axf.legend(
        handles=sp.tier_legend_handles(),
        loc="lower right",
        borderaxespad=0.1,
        handlelength=1.5,
    )
    sp.panel_label(axf, "f", x=-0.42, y=1.0)

    var_style = {"same residues, consensus incl. pair": {"ls": "-", "mfc": None},
                 "same residues, leave-pair-out consensus": {"ls": "--", "mfc": "white"}}  # fmt: skip
    st_col = {"ordered": sp.OKABE_ITO["blue"], "disordered": sp.OKABE_ITO["vermillion"]}
    xk = np.arange(len(KP_LABELS))
    axg = fig.add_axes([0.56, y0, 0.18, h])
    axh = fig.add_axes([0.81, y0, 0.18, h])
    for ax, col, lab in ((axg, "dZ_median", "Median ΔZ (higher-k − lower-k)"),
                         (axh, "dG_median", "Median ΔG (higher-k − lower-k)")):  # fmt: skip
        for var, st in var_style.items():
            for s_ in ("ordered", "disordered"):
                d = (
                    KP[(KP.variant == var) & (KP.state == s_)]
                    .set_index("dk_bin")
                    .loc[KP_LABELS]
                )
                ax.plot(xk, d[col], color=st_col[s_], ls=st["ls"], marker="o", ms=2.5, lw=0.7,
                        mfc=st["mfc"] or st_col[s_], mew=0.6)  # fmt: skip
        ax.axhline(0, color="#9a9a9a", lw=0.4, ls=":")
        ax.set_xticks(xk)
        ax.set_xticklabels(KP_LABELS)
        ax.set_xlabel("Δk at the residue")
        ax.set_ylabel(lab)
    axh.set_ylim(-0.25, 0.25)
    hand = [Line2D([], [], color=st_col["ordered"], marker="o", ms=2.5, lw=0.7, label="ordered (consensus < 0.3)"),
            Line2D([], [], color=st_col["disordered"], marker="o", ms=2.5, lw=0.7, label="disordered (> 0.7)"),
            Line2D([], [], color="#7a7a7a", marker="o", ms=2.5, lw=0.7, label="consensus incl. pair"),
            Line2D([], [], color="#7a7a7a", marker="o", ms=2.5, lw=0.7, ls="--", mfc="white",
                   label="leave-pair-out consensus")]  # fmt: skip
    axh.legend(handles=hand, loc="upper center", borderaxespad=0.1, handlelength=2.0)
    sp.panel_label(axg, "g", x=-0.3, y=1.0)
    sp.panel_label(axh, "h", x=-0.33, y=1.0)
    return fig


def main():
    sp.parse_args(sp.cli(__doc__))
    DATA_DIR = sp.out_dir(DATA_SUBDIR)
    df, tier_ids = load()
    tier_labels = {t: sp.load_tier_scores(t, fields=("gscores",)) for t in sp.TIERS}
    L = loo_table(df, tier_labels)
    D = fr.Data()
    Lf = fr.loo_table(D)
    Lf["kbin"] = Lf["kbin"].map(dict(zip(fr.K_LABELS, K_LABELS)))
    defs = definitions(L, Lf)
    S = summarise_loo(defs)
    KT = k_table(defs)
    pairs = replicate_pairs(df, PAIR_CAP)
    RT, vm_meta = tier_r_table(pairs)
    RF, CS = fig3_residue_r(D)
    CS_preview = cap_sensitivity(df)
    R = kpaired_lpo(df)
    KP = summarise_kp(R)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    L.drop(columns=[c for c in L.columns if c.startswith("dev_")]).to_csv(
        DATA_DIR / "S7_loo_chains.csv", index=False
    )
    S.to_csv(DATA_DIR / "S7_loo_summary.csv", index=False)
    KT.to_csv(DATA_DIR / "S7_tier_by_k.csv", index=False)
    Lf.to_csv(DATA_DIR / "S7_loo_chains_fig3def.csv", index=False)
    RT.to_csv(DATA_DIR / "S7_residue_r_preview.csv", index=False)
    RF.to_csv(DATA_DIR / "S7_residue_r_fig3def.csv", index=False)
    CS.to_csv(DATA_DIR / "S7_cap_sensitivity_fig3def.csv", index=False)
    CS_preview.to_csv(DATA_DIR / "S7_cap_sensitivity_preview.csv", index=False)
    KP.to_csv(DATA_DIR / "S7_kpaired_lpo.csv", index=False)
    summary = {
        "n_loo_chains": int(len(L)),
        "n_loo_sequences": int(L.sequence.nunique()),
        "n_residue_pairs": int(len(pairs)),
        "n_pair_sequences": int(len({p["sequence"] for p in pairs})),
        "variance_matching_preview_to_strict": vm_meta,
        "n_loo_chain_tier_rows_fig3def": int(len(Lf)),
        "n_loo_chains_fig3def": int(Lf.id.nunique()),
        "kpaired_lpo_residue_pairs": int(R.cons_lpo.notna().sum()),
        "kpaired_all_residue_pairs": int(len(R)),
    }
    (DATA_DIR / "S7_summary.json").write_text(json.dumps(summary, indent=2))
    out = sp.save(plot(S, KT, RT, RF, CS, KP), "S7")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(out)
    print(json.dumps(summary, indent=1))
    print(S.round(4).to_string())
    print(KT.round(4).to_string())
    print(RT.round(4).to_string())
    print(RF.round(4).to_string())
    print(CS.round(4).to_string())
    print(CS_preview.round(4).to_string())
    print(KP.round(4).to_string())


if __name__ == "__main__":
    main()
