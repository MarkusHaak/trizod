#!/usr/bin/env python3
"""Supplementary Fig. S7 -- replicate analyses in full.

The BMRB holds many independent measurements of the same protein. This figure gives
every replicate statistic behind Fig. 4 in full: the leave-one-entry-out deviation per
tier (cumulative tiers and exclusive strata; the left half of the old Fig. 3a, now
here only), shift-count strata, the residue-level r per tier with its variance-matched
and no-re-deposition variants (Fig. 4c), the cap on pairs per sequence, the
residue-level r of the condition classes (the r column of the old Fig. 3b; Fig. 4a
shows |dG|) -- and repeats the paired shift-count test of Fig. 3b with a leave-pair-out
consensus.

Panels (12 Nov 2026 layout, story blueprint 6.5 and 6.7):
  a-c  leave-one-entry-out deviation, share of chains deviating by > 0.1, flip rate;
  d    median deviation by tier x chain median k;
  e    residue-level r per tier: pooled (Fig. 4c), variance-matched to the unfiltered
       G distribution, and without identical-data re-depositions (``no_redeposition_r``;
       the definition of the ``\\rRepFigNoRedep*`` macros in make_numbers.py
       ``sec_front_revision``);
  f    pooled r for other caps on pairs per sequence;
  g    residue-level r of replicate pairs under the same or different pH or temperature,
       shift-count matched, for all pairs and for pairs without a state difference
       (``condition_r``: ``fig3_replicates.condition_pairs`` + ``boot_conditions`` and
       ``fig3b_conditions.state_clean_pairs``; the \\condRFig* macros);
  h, i the paired shift-count test with a leave-pair-out consensus.

The figure draws ONE replicate definition, that of Fig. 4 (``fig3_replicates.py``:
non-bound chains, consensus of the replicates from OTHER entries, tier-specific labels;
scope "fig3" below). The all-chains definition (scope "all_chains": all chains, bound
or not, consensus of all other chains of the sequence; the ``\\repDevCum*``,
``\\repDevExcl*`` and ``\\rRepResidue*`` macros of make_numbers.py) is still computed
and written to the data files, but not drawn. Layout: rows of tiers as in Fig. 4c
(a-c, e), tier x shift-count maps (d), condition rows as in Fig. 4a (g), and h/i styled
as Fig. 3b (ordered residues black filled circles, disordered open squares, same y
label and Delta-k bins). ``supp_s10_baselines.py`` imports ``replicate_pairs``,
``PAIR_CAP`` and ``MIN_PAIR_RES``; make_numbers.py imports ``SEED`` (keep the names).

Shift-count classes of panel d: chain median k <= 12, 13-17 and 18-21 (``K_LABELS``),
NOT the classes of Fig. 3c (<= 6, 7-15, 16-21): these are the strata that Methods and
the Supplementary Notes quote (``\\repDevFig*KLow/KMid/KHigh``, from
``fig3_replicates.K_BINS``), and no strict chain has a median k of 6 or less.

Definitions of the all-chains scope (identical to ``scripts/manuscript/make_numbers.py``,
which writes the ``\\repDev*``, ``\\rRepResidue*`` and ``\\kpair*`` macros, so the data
files and the prose quote the same numbers) and of the leave-pair-out test:

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
* **Leave-pair-out paired-k test.** The pair sample of Fig. 3b (``fig2a_kpaired.py``:
  non-bound chains, <= 30 pairs per sequence, seed 0, different entries), restricted
  to sequences with >= 3 chains. Each residue's ordered (< 0.3) / disordered (> 0.7)
  state comes either from the consensus over all chains of the sequence (Fig. 3b) or
  from the consensus over all chains EXCEPT the two in the pair (leave-pair-out),
  which removes the weak circularity of a state defined partly by the pair itself.

Bootstrap 95 % confidence intervals resample sequences (1,000 replicates, seed 0),
because chains and pairs of one sequence are not independent.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s7_replicates.py \
        --out-dir <draft>/Figures/paper
Outputs: <out-dir>/S7.pdf and <out-dir>/suppfigs_b_data/S7_*.csv / S7_summary.json
(``S7_condition_r.csv`` and ``S7_residue_r_no_redeposition.csv`` for panels g and e).
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
# chain median k classes of panel d = fig3_replicates.K_BINS (quoted in Methods and the
# Supplementary Notes); they differ from the Fig. 3c classes (<= 6, 7-15, 16-21)
K_EDGES = [0, 12, 17, 21]
K_LABELS = ["≤12", "13–17", "18–21"]
# paired-k test (Fig. 3b definitions, fig2a_kpaired.py)
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
    """LOO table of fig3_replicates (one row per chain and cumulative tier): cumulative = rows of tier t;
    exclusive = rows of tier t whose strictest tier is t."""
    sub = Lf[Lf.tier == t]
    return sub if mode == "cumulative" else sub[sub.strictest == t]


def definitions(L: pd.DataFrame, Lf: pd.DataFrame) -> dict:
    return {"fig3": (Lf, stratum_fig3), "all_chains": (L, stratum)}


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
    """Fig. 4c residue-level r (pairs capped per sequence AND tier) with sequence-bootstrap
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


# ------------------------------------------- condition classes and re-depositions
def condition_r(D) -> pd.DataFrame:
    """Panel g: residue-level r of replicate pairs under the same or different pH or
    temperature (shift-count matched, sequence bootstrap), for all pairs and for pairs
    without a state difference -- the r column of the old Fig. 3b and the
    ``\\condRFig*`` macros. Fig. 4a draws |dG| of the same pairs."""
    import fig3b_conditions as f3b

    C = fr.condition_pairs(D)
    rows = []
    for block, Cb in (("all", C), ("clean", f3b.state_clean_pairs(D, C))):
        b = fr.boot_conditions(Cb, matched=True)
        for cond in ("same", "different"):
            d = b[cond]
            rows.append({"block": block, "cond": cond, "n_pairs": d["n_pairs"],
                         "n_sequences": d["n_sequences"], "r": d["r"], "r_lo": d["r_lo"],
                         "r_hi": d["r_hi"], "median_absdg": d["median_absdg"]})  # fmt: skip
    return pd.DataFrame(rows)


# identical-data re-depositions: >= REDEP_FRAC of >= REDEP_MIN_SHARED shared backbone
# shifts within REDEP_TOL_PPM (the values of make_numbers.py sec_front_revision)
REDEP_TOL_PPM, REDEP_MIN_SHARED, REDEP_FRAC = 0.005, 20, 0.9


def no_redeposition_r(D) -> tuple[pd.DataFrame, int]:
    """Panel e: the Fig. 4c residue-level r per cumulative tier without identical-data
    re-depositions. Port of make_numbers.py ``sec_front_revision`` (the
    ``\\rRepFigNoRedep*`` macros): ``fig3_replicates.pair_table``'s definition, with
    re-deposition pairs dropped from the candidate list BEFORE the per-sequence cap.
    Returns one row per tier and the number of re-deposition pairs."""
    import pyarrow.parquet as pq

    nb = D.nb
    multi = nb.groupby("sequence")["entry_id"].transform("nunique") >= 2
    ids = set(nb.loc[multi, "id"])
    t = pq.read_table(
        sp.PQ_SHIFTS,
        columns=["id", "seq_id", "atom_id", "val_ppm", "is_backbone"],
        filters=[("is_backbone", "=", True)],
    ).to_pandas()
    t = t[t["id"].isin(ids)].drop_duplicates(["id", "seq_id", "atom_id"])
    SH = {
        i: pd.Series(
            g["val_ppm"].to_numpy(float),
            index=pd.MultiIndex.from_arrays([g["seq_id"], g["atom_id"]]),
        )  # fmt: skip
        for i, g in t.groupby("id", sort=False)
    }

    def redeposit(a, b) -> bool:
        sa, sb = SH.get(a), SH.get(b)
        if sa is None or sb is None:
            return False
        j = sa.index.intersection(sb.index)
        if len(j) < REDEP_MIN_SHARED:
            return False
        d = np.abs(sa.loc[j].to_numpy() - sb.loc[j].to_numpy())
        return bool((d <= REDEP_TOL_PPM).mean() >= REDEP_FRAC)

    rng = random.Random(fr.SEED)
    rows, n_dup = [], 0
    for seq, g in nb.groupby("sequence", sort=True):
        if g["entry_id"].nunique() < 2:
            continue
        cand = []
        for a, b in fr._seq_pairs(g, rng):
            if redeposit(a, b):
                n_dup += 1
            else:
                cand.append((a, b))
        for tier in sp.TIERS:
            ids_t = D.tier_ids[tier]
            taken = 0
            for a, b in cand:
                if taken >= fr.PAIR_CAP:
                    break
                if a not in ids_t or b not in ids_t:
                    continue
                ga, gb = D.g(tier, a), D.g(tier, b)
                m = np.isfinite(ga) & np.isfinite(gb)
                if m.sum() < fr.MIN_PAIR_RES:
                    continue
                taken += 1
                rows.append({"sequence": seq, "tier": tier, "a": a, "b": b,
                             **fr._pair_stats(ga[m], gb[m])})  # fmt: skip
    P = pd.DataFrame(rows)
    out = [{"tier": tier, **fr.boot_pooled(P[P.tier == tier])} for tier in sp.TIERS]
    return pd.DataFrame(out), n_dup


# --------------------------------------------------------- leave-pair-out paired k
def kpaired_lpo(df: pd.DataFrame) -> pd.DataFrame:
    """Fig. 3b pair sample; per residue: dk, dZ, dG, consensus with and without the pair."""
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
        "all pairs, consensus incl. pair (Fig. 3b)": (R, state(R.cons)),
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
# Layout in millimetres (W_DOUBLE wide, placed at 1:1). Rows 1 and 2 are dot plots with one
# row per tier, as Fig. 4c; the tier labels sit left of the first axes of each row.
FIG_H = 143.0  # with its legend, the figure fits one A4 page of the SI
DOT_W, DOT_H = 40.0, 18.0  # dot-plot axes
X_DOT = (16.0, 64.0, 112.0)  # left edges of a, b, c (and e = third column)
TOP1 = 8.0  # top of the row-1 axes
TOP2 = 44.0  # top of the row-2 axes (d maps, e)
TOP3 = 78.0  # top of the row-3 axes (f, g)
TOP4 = 113.0  # top of the row-4 axes (h, i)
H3, H4 = 24.0, 22.0
HM_LEFT, HM_W = (16.0, 52.0), 33.0  # d: the two maps
G_LEFT, G_W = 104.0, 48.0  # g: condition classes (labels to the left of the axes)
EDGE = "#262626"  # marker edge, as Fig. 4c
MS = 4.2
DY = 0.17  # vertical offset of the two markers of one tier
DY3 = 0.25  # the three markers of one tier in e
# ordered / disordered residues exactly as Fig. 3b (fig2a_kpaired.STYLES)
STATE_STYLE = {
    "ordered": dict(sp.CONTRAST["focus"]),
    "disordered": dict(sp.CONTRAST["control"]),  # open ink squares (grey = CheZOD1325)
}
VARIANTS = {  # consensus that defines a residue's state -> line style
    "same residues, consensus incl. pair": ("-", "Consensus of all chains"),
    "same residues, leave-pair-out consensus": ("--", "Consensus without the pair"),
}
# g: the condition classes of Fig. 4a (fig3b_conditions.BLOCKS), drawn with sp.COND;
# "All pairs" (review_burkhard 4): the block holds the pairs with deposited conditions in
# the two classes, not every replicate pair
COND_BLOCKS = [  # (block, header, [(cond, style, label)])
    ("all", "All pairs", [
        ("same", sp.COND["same"], "Same conditions"),
        ("different", sp.COND["different"], "Different pH\nor temperature"),
    ]),
    ("clean", "Without state differences", [
        ("same", {**sp.COND["same"], "mfc": "white"}, "Same conditions"),
        ("different", sp.COND["different_state_clean"], "Different pH\nor temperature"),
    ]),
]  # fmt: skip


def _nice_hi(v: float, step: float) -> float:
    return step * np.ceil(v / step - 1e-9)


def _dot_axes(fig, left: float, top: float, show_labels: bool):
    W, H = sp.W_DOUBLE, FIG_H
    ax = fig.add_axes([left / W, 1 - (top + DOT_H) / H, DOT_W / W, DOT_H / H])
    n = len(sp.TIERS)
    ax.set_ylim(n - 0.45, -0.55)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sp.TIERS if show_labels else [], fontsize=sp.FS_BODY)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E6E6E6", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    return ax


def _dot(ax, x, lo, hi, y, tier, filled=True, marker="o"):
    if lo is not None:
        ax.plot(
            [lo, hi], [y, y], color="#4D4D4D", lw=0.8, solid_capstyle="butt", zorder=2
        )
    ax.plot(x, y, ls="", marker=marker, ms=MS, mfc=sp.TIER_COLORS[tier] if filled else "white",
            mec=EDGE, mew=0.6, zorder=3)  # fmt: skip


def plot(S: pd.DataFrame, KT: pd.DataFrame, RF: pd.DataFrame, CS: pd.DataFrame,
         KP: pd.DataFrame, CR: pd.DataFrame, NR: pd.DataFrame):  # fmt: skip
    """Supplementary Fig. S7, Fig. 4 definition only (scope "fig3" of S and KT)."""
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    W, H = sp.W_DOUBLE, FIG_H
    fig = sp.figure(W, H)
    mid = sp.TIER_COLORS["tolerant"]

    # ---- row 1: a difference from replicates, b share > 0.1, c opposite label
    metrics = [
        ("median_dev", "median_dev_lo", "median_dev_hi", 1, 0.04, 2, f3a_xlabel()),
        ("share_dev_gt_0.1", "share_lo", "share_hi", 100, 20, 0, "Chains with |ΔG| > 0.1 (%)"),
        ("flip", "flip_lo", "flip_hi", 100, 10, 0,
         f"Residues with the opposite\nlabel at G = {sp.G_DIS} (%)"),
    ]  # fmt: skip
    Sf = S[S["scope"] == "fig3"]
    for i, (col, lo, hi, fac, step, nd, lab) in enumerate(metrics):
        ax = _dot_axes(fig, X_DOT[i], TOP1, show_labels=i == 0)
        x_max = 0.0
        for mode, dy, filled in (("cumulative", -DY, True), ("exclusive", DY, False)):
            d = Sf[Sf["mode"] == mode].set_index("tier").loc[sp.TIERS]
            for y, t in enumerate(sp.TIERS):
                _dot(ax, fac * d.loc[t, col], fac * d.loc[t, lo], fac * d.loc[t, hi], y + dy,
                     t, filled)  # fmt: skip
            x_max = max(x_max, fac * d[hi].max())
        x_hi = _nice_hi(x_max, step)
        ticks = np.arange(0, x_hi + step / 2, step)
        ax.set_xlim(0, x_hi)
        ax.set_xticks(ticks, [f"{t:.{nd}f}" if t else "0" for t in ticks])
        ax.set_xlabel(lab)
        sp.fig_label(fig, "abc"[i], X_DOT[i] - 16.0 if i == 0 else X_DOT[i] - 6.0, 4.3)
    hand = [
        Line2D([], [], ls="", marker="o", ms=MS, mfc=mid, mec=EDGE, mew=0.6,
               label="Cumulative tier (every chain the tier scores)"),
        Line2D([], [], ls="", marker="o", ms=MS, mfc="white", mec=EDGE, mew=0.6,
               label="Exclusive stratum (strictest tier only)"),
    ]  # fmt: skip
    fig.legend(handles=hand, loc="upper left", bbox_to_anchor=(X_DOT[0] / W, 1 - 0.3 / H),
               ncol=2, handletextpad=0.2, columnspacing=2.0, borderaxespad=0.0)  # fmt: skip

    # ---- row 2: d tier x k maps (grey ramp: darker = larger difference)
    cmap = mpl.colormaps["Greys"]
    norm = mpl.colors.LogNorm(vmin=0.02, vmax=0.25)
    KTf = KT[KT["scope"] == "fig3"]
    for i, (mode, title) in enumerate((("exclusive", "Exclusive stratum"),
                                       ("cumulative", "Cumulative tier"))):  # fmt: skip
        ax = fig.add_axes([HM_LEFT[i] / W, 1 - (TOP2 + DOT_H + 4.0) / H, HM_W / W,
                           (DOT_H + 4.0) / H])  # fmt: skip
        T = KTf[KTf["mode"] == mode]
        M = T.pivot(index="tier", columns="kbin", values="median_dev").loc[
            sp.TIERS, K_LABELS
        ]
        N = T.pivot(index="tier", columns="kbin", values="n_chains").loc[
            sp.TIERS, K_LABELS
        ]
        M, N = M.to_numpy(float), N.to_numpy(float)
        ax.imshow(np.where(N > 0, M, np.nan), cmap=cmap, norm=norm, aspect="auto")
        for r_ in range(len(sp.TIERS)):
            for c_ in range(len(K_LABELS)):
                if N[r_, c_] == 0:
                    ax.text(c_, r_, "–", ha="center", va="center", fontsize=sp.FS_SMALL)
                    continue
                v = M[r_, c_]
                tc = "white" if norm(v) > 0.55 else "black"
                ax.text(c_, r_ - 0.17, f"{v:.3f}", ha="center", va="center",
                        fontsize=sp.FS_SMALL, color=tc)  # fmt: skip
                ax.text(c_, r_ + 0.22, f"({int(N[r_, c_]):,})", ha="center", va="center",
                        fontsize=sp.FS_SMALL, color=tc)  # fmt: skip
        ax.set_xticks(range(len(K_LABELS)), K_LABELS)
        ax.set_yticks(range(len(sp.TIERS)), sp.TIERS if i == 0 else [])
        ax.tick_params(length=0)
        ax.tick_params(axis="y", labelsize=sp.FS_BODY)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_xlabel(r"Median shift count $\mathit{k}$ of the chain" if i == 0 else "")
        ax.set_title(title, fontsize=sp.FS_BODY, pad=2)
        if i == 0:  # one x label centred under both maps
            ax.xaxis.set_label_coords(
                (HM_LEFT[1] + HM_W - HM_LEFT[0]) / 2 / HM_W, -0.17
            )
    sp.fig_label(fig, "d", 0.0, TOP2 - 5.0)

    # ---- row 2: e residue-level r per tier (Fig. 4c), variance-matched, no re-depositions
    axe = _dot_axes(fig, X_DOT[2], TOP2 + 4.0, show_labels=True)
    Ri = RF.set_index("tier").loc[sp.TIERS]
    Ni = NR.set_index("tier").loc[sp.TIERS] if NR is not None else None
    for y, t in enumerate(sp.TIERS):
        # pooled: filled circle; variance-matched: a cross; without re-depositions: open
        # square (figure review item 16: open diamonds and triangles are the amino-acid
        # mean and CheZOD117 elsewhere; this is the CONTRAST pair plus a cross)
        _dot(axe, Ri.loc[t, "r"], Ri.loc[t, "r_lo"], Ri.loc[t, "r_hi"], y - DY3, t)
        axe.plot(Ri.loc[t, "r_varmatched_unfiltered"], y, ls="", marker="x", ms=MS - 0.6,
                 mec=sp.INK, mew=0.9, zorder=3)  # fmt: skip
        if Ni is not None:
            _dot(axe, Ni.loc[t, "r"], Ni.loc[t, "r_lo"], Ni.loc[t, "r_hi"], y + DY3, t,
                 filled=False, marker="s")  # fmt: skip
    axe.set_ylim(len(sp.TIERS) - 0.4, -0.6)
    axe.set_xlim(0.4, 1.0)
    axe.set_xticks([0.4, 0.6, 0.8, 1.0])
    axe.set_xlabel(f3a_xlabel_r())
    he = [Line2D([], [], ls="", marker="o", ms=MS, mfc=mid, mec=EDGE, mew=0.6,
                 label="Pooled"),
          Line2D([], [], ls="", marker="x", ms=MS - 0.6, mec=sp.INK, mew=0.9,
                 label="Variance-matched")]  # fmt: skip
    if Ni is not None:
        he.append(Line2D([], [], ls="", marker="s", ms=MS, mfc="white", mec=EDGE, mew=0.6,
                         label="Without re-depositions"))  # fmt: skip
    fig.legend(handles=he, loc="lower left", bbox_to_anchor=((X_DOT[2] - 1.0) / W, 1 - (TOP2 + 3.6) / H),
               ncol=1, handletextpad=0.1, borderaxespad=0.0, labelspacing=0.2)  # fmt: skip
    sp.fig_label(fig, "e", X_DOT[2] - 16.0, TOP2 - 5.0)

    # ---- row 3: f cap sensitivity
    axf = fig.add_axes([X_DOT[0] / W, 1 - (TOP3 + H3) / H, 38.0 / W, H3 / H])
    caps = [str(c) for c in CS["cap"].unique()]
    for t in sp.TIERS:
        d = CS[CS.tier == t]
        axf.plot(range(len(caps)), d["r"], color=sp.TIER_COLORS[t], marker=sp.TIER_MARKERS[t],
                 ms=3.0, lw=0.8, mec=EDGE, mew=0.4, label=t)  # fmt: skip
    axf.set_xticks(range(len(caps)), ["all" if c == "none" else c for c in caps])
    axf.set_xlim(-0.4, len(caps) - 0.6)
    axf.set_xlabel("Pairs per sequence (cap)")
    axf.set_ylabel(f3a_xlabel_r().replace("\n", " "))
    axf.set_ylim(0.3, 1.0)  # room below 0.5 for the legend
    axf.set_yticks([0.4, 0.6, 0.8, 1.0])
    axf.legend(loc="lower left", ncol=2, borderaxespad=0.2, handlelength=1.4,
               columnspacing=0.8, handletextpad=0.3, labelspacing=0.2)  # fmt: skip
    sp.fig_label(fig, "f", 0.0, TOP3 - 5.0)

    # ---- row 3: g residue-level r of the condition classes (as Fig. 4a, r instead of |dG|)
    axg = fig.add_axes([G_LEFT / W, 1 - (TOP3 + H3) / H, G_W / W, H3 / H])
    Ci = CR.set_index(["block", "cond"])
    y, ys, labels, heads = 0.0, [], [], {}
    for bk, _, rows in COND_BLOCKS:
        heads[bk] = y - 0.7
        for cond, st, lab in rows:
            d = Ci.loc[(bk, cond)]
            axg.plot([d.r_lo, d.r_hi], [y, y], color="#4D4D4D", lw=0.8,
                     solid_capstyle="butt", zorder=2)  # fmt: skip
            axg.plot(d.r, y, ms=MS, zorder=3, **sp.marker_kw(st, mew=0.8))
            ys.append(y)
            labels.append(lab)
            y += 1.0
        y += 0.9
    axg.set_ylim(max(ys) + 0.55, min(heads.values()) - 0.45)
    axg.set_yticks(ys)
    axg.set_yticklabels(labels, linespacing=1.0)
    axg.tick_params(axis="y", length=0)
    axg.spines["left"].set_visible(False)
    axg.set_xlim(0.4, 1.0)
    axg.set_xticks([0.4, 0.6, 0.8, 1.0])
    axg.grid(axis="x", color="#E6E6E6", lw=0.4, zorder=0)
    axg.set_axisbelow(True)
    axg.set_xlabel(f3a_xlabel_r().replace("\n", " ") + ", unfiltered labels")
    trg = blended_transform_factory(fig.transFigure, axg.transData)
    for bk, head, _ in COND_BLOCKS:
        axg.text((G_LEFT - 30.0) / W, heads[bk], head, transform=trg, ha="left",
                 va="center", fontsize=sp.FS_BODY, clip_on=False)  # fmt: skip
    sp.fig_label(fig, "g", G_LEFT - 34.0, TOP3 - 5.0)

    # ---- row 4: h/i paired shift-count test, drawn as Fig. 3b
    xk = np.arange(len(KP_LABELS))
    boxes = {"Z": 30.0, "G": 98.0}
    hi_w = 52.0
    axes = {}
    for unit, left in boxes.items():
        ax = fig.add_axes([left / W, 1 - (TOP4 + H4) / H, hi_w / W, H4 / H])
        axes[unit] = ax
        ax.axhline(0, color="0.75", lw=0.5, zorder=0)
        for var, (ls, _) in VARIANTS.items():
            for state, st in STATE_STYLE.items():
                d = (
                    KP[(KP.variant == var) & (KP.state == state)]
                    .set_index("dk_bin")
                    .loc[KP_LABELS]
                )
                ax.plot(xk, d[f"d{unit}_median"], color=st["color"], ls=ls, marker=st["marker"],
                        mfc=st["mfc"], mec=st["color"], mew=0.8, ms=3.6, lw=1.0,
                        zorder=3 if state == "ordered" else 2)  # fmt: skip
        ax.set_xlim(-0.5, len(KP_LABELS) - 0.5)
        ax.set_xticks(xk, KP_LABELS)
        ax.set_title(f"{unit}-score", loc="left", fontsize=sp.FS_BODY, pad=3)
        ax.set_xlabel(r"Difference in shift count $\Delta\mathit{k}$")
    axes["Z"].set_ylabel("Score difference\n(more − fewer shifts)", linespacing=1.1)
    axes["Z"].set_ylim(-1.0, 7.0)
    axes["Z"].set_yticks([0, 2, 4, 6])
    axes["G"].set_ylim(-0.12, 0.12)
    axes["G"].set_yticks([-0.1, 0, 0.1], ["−0.1", "0", "0.1"])
    hs = [Line2D([], [], color=st["color"], marker=st["marker"], mfc=st["mfc"], mec=st["color"],
                 mew=0.8, ms=3.6, lw=1.0, label=f"{state.capitalize()} residues")
          for state, st in STATE_STYLE.items()]  # fmt: skip
    axes["Z"].legend(handles=hs, loc="upper left", handlelength=1.8, borderaxespad=0.2)
    hv = [
        Line2D([], [], color="#4D4D4D", ls=ls, lw=1.0, label=lab)
        for ls, lab in VARIANTS.values()
    ]
    axes["G"].legend(handles=hv, loc="upper left", handlelength=2.2, borderaxespad=0.2)
    sp.fig_label(fig, "h", 0.0, TOP4 - 5.0)
    sp.fig_label(fig, "i", boxes["G"] - 8.0, TOP4 - 5.0)
    return fig


def f3a_xlabel() -> str:
    """The x label of the difference from replicates (``fig3a_reliability.XLABEL_DEV``)."""
    import fig3a_reliability as f3a

    return f3a.XLABEL_DEV


def f3a_xlabel_r() -> str:
    import fig3a_reliability as f3a

    return f3a.XLABEL_R


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
    CS_all = cap_sensitivity(df)
    R = kpaired_lpo(df)
    KP = summarise_kp(R)
    CR = condition_r(D)
    NR, n_redep = no_redeposition_r(D)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    L.drop(columns=[c for c in L.columns if c.startswith("dev_")]).to_csv(
        DATA_DIR / "S7_loo_chains_all_chains.csv", index=False
    )
    S.to_csv(DATA_DIR / "S7_loo_summary.csv", index=False)
    KT.to_csv(DATA_DIR / "S7_tier_by_k.csv", index=False)
    Lf.to_csv(DATA_DIR / "S7_loo_chains_fig3def.csv", index=False)
    RT.to_csv(DATA_DIR / "S7_residue_r_all_chains.csv", index=False)
    RF.to_csv(DATA_DIR / "S7_residue_r_fig3def.csv", index=False)
    CS.to_csv(DATA_DIR / "S7_cap_sensitivity_fig3def.csv", index=False)
    CS_all.to_csv(DATA_DIR / "S7_cap_sensitivity_all_chains.csv", index=False)
    KP.to_csv(DATA_DIR / "S7_kpaired_lpo.csv", index=False)
    CR.to_csv(DATA_DIR / "S7_condition_r.csv", index=False)
    NR.to_csv(DATA_DIR / "S7_residue_r_no_redeposition.csv", index=False)
    summary = {
        "drawn": "the Fig. 4 definition (scope 'fig3'); the all-chains definition "
        "(scope 'all_chains') is written to the data files only",
        "k_classes_panel_d": K_LABELS,
        "n_loo_chains_all_chains": int(len(L)),
        "n_loo_sequences_all_chains": int(L.sequence.nunique()),
        "n_residue_pairs_all_chains": int(len(pairs)),
        "n_pair_sequences_all_chains": int(len({p["sequence"] for p in pairs})),
        "variance_matching_all_chains_to_strict": vm_meta,
        "n_loo_chain_tier_rows_fig3def": int(len(Lf)),
        "n_loo_chains_fig3def": int(Lf.id.nunique()),
        "kpaired_lpo_residue_pairs": int(R.cons_lpo.notna().sum()),
        "kpaired_all_residue_pairs": int(len(R)),
        "condition_r_panel_g": CR.round(4).to_dict("records"),
        "n_redeposition_pairs": int(n_redep),
        "residue_r_no_redeposition_panel_e": NR.round(4).to_dict("records"),
    }
    (DATA_DIR / "S7_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    out = sp.save(plot(S, KT, RF, CS, KP, CR, NR), "S7")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(out)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(S.round(4).to_string())
    print(KT.round(4).to_string())
    print(RT.round(4).to_string())
    print(RF.round(4).to_string())
    print(CS.round(4).to_string())
    print(CS_all.round(4).to_string())
    print(KP.round(4).to_string())
    print(CR.round(4).to_string())
    print(f"re-deposition pairs: {n_redep}")
    print(NR.round(4).to_string())


if __name__ == "__main__":
    main()
