"""Inputs of the threshold framework: the build (release Parquet + scored tiers), the per-residue
reference labels re-joined to that build, the analysis sets and the cluster keys.

Labels are joined by chain id AND sequence: a labelled chain enters only if the build holds the same
chain id with the identical sequence; G, k and the scored mask always come from the build, never from
the label files (whose G is that of the build the labels were made on).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from thrlib import LX_EDGES, NB, gbin

TIERS = (
    "unfiltered",
    "tolerant",
    "moderate",
    "strict",
)  # permissive -> strict (nested)
TEST = ("test_trizod", "test_chezod117")
DENATURED = {"denatured", "unfolded", "partially denatured", "partially unfolded"}
REL_COLS = [
    "id",
    "entry_id",
    "sequence",
    "split",
    "cluster_repr",
    "gscores",
    "mask",
    "k",
    "ph",
    "temperature",
    "is_bound",
    "physical_state",
    "cosolvent_evidence",
    "sample_state_evidence",
    "membrane_mimetic",
    "n_scored",
]


def data_root() -> Path:
    """$TRIZOD_ROOT, else the first ancestor of this file that holds data/interim (worktrees live
    inside the main checkout, which holds the data)."""
    env = os.environ.get("TRIZOD_ROOT")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "data" / "interim").is_dir():
            return p
    return here.parents[4]


# ----------------------------------------------------------------------------- build
class Build:
    """Release chains (row order of the Parquet) with per-chain G / k arrays and score tiers."""

    def __init__(self, release: Path, scored_dir: Path):
        rel = pd.read_parquet(release, columns=REL_COLS)
        self.rel = rel
        self.G, self.K = {}, {}
        for cid, g, m, k in zip(rel.id, rel.gscores, rel["mask"], rel.k):
            g = np.asarray(g, float)
            ok = np.asarray(m, bool) & np.isfinite(g)
            self.G[cid] = np.where(ok, g, np.nan)
            self.K[cid] = np.asarray(k, float)
        self.seq = dict(zip(rel.id, rel.sequence))
        self.split = dict(zip(rel.id, rel.split))
        # bootstrap unit: cluster_repr; else that of a train/redundant chain with the identical
        # sequence; else the sequence itself (as in the 2026-11-16 framework)
        s2r = (
            rel[rel.split.isin(["train", "redundant"]) & rel.cluster_repr.notna()]
            .drop_duplicates("sequence")
            .set_index("sequence")
            .cluster_repr.to_dict()
        )
        self.key = {
            c: (r if isinstance(r, str) else s2r.get(s, "seq:" + s))
            for c, r, s in zip(rel.id, rel.cluster_repr, rel.sequence)
        }
        self.tier, self.tier_check = self._tiers(scored_dir)
        self._hg, self._hl = {}, {}

    def _tiers(self, scored_dir: Path):
        """score_tier = strictest tier whose scores.json holds the chain; also checks that the release G
        equals that strictest record (both must come from the same build)."""
        tier, rec = {}, {}
        counts = {}
        for t in TIERS:
            n = 0
            with open(scored_dir / t / "scores.json") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    tier[r["ID"]] = t
                    rec[r["ID"]] = r["gscores"]
                    n += 1
            counts[t] = n
        ndiff_ch = ndiff_res = nmiss = nlen = 0
        for cid, g in self.G.items():
            r = rec.get(cid)
            if r is None:
                nmiss += 1
                continue
            gt = np.array([np.nan if x is None else x for x in r], float)
            if len(gt) != len(g):
                nlen += 1
                continue
            d = int(np.sum(np.isfinite(gt) != np.isfinite(g)))
            both = np.isfinite(gt) & np.isfinite(g)
            d += int(np.sum(np.abs(gt[both] - g[both]) > 1e-3))
            ndiff_res += d
            ndiff_ch += d > 0
        check = {
            "records_per_tier": counts,
            "release_chains": len(self.G),
            "release_chains_without_tier": nmiss,
            "score_tier_counts": pd.Series([tier.get(c) for c in self.G])
            .value_counts(dropna=False)
            .to_dict(),
            "chains_G_differs_from_strictest_record": ndiff_ch,
            "residues_G_differs_from_strictest_record": ndiff_res,
            "chains_length_differs_from_strictest_record": nlen,
        }
        check["score_tier_counts"] = {
            str(k): int(v) for k, v in check["score_tier_counts"].items()
        }
        return {c: tier.get(c) for c in self.G}, check

    def in_tiers(self, tiers) -> set:
        return {c for c, t in self.tier.items() if t in tiers}

    def nested(self, tier: str) -> set:
        """Chains scored at `tier` or stricter (= present in that tier's scores.json)."""
        return self.in_tiers(TIERS[TIERS.index(tier) :])

    def is_test(self, cid) -> bool:
        return self.split[cid] in TEST

    def hist_g(self, cid, krange=None) -> np.ndarray:
        key = (cid, krange)
        if key not in self._hg:
            g = self.G[cid]
            ok = np.isfinite(g)
            if krange is not None:
                ok &= (self.K[cid] >= krange[0]) & (self.K[cid] <= krange[1])
            self._hg[key] = np.bincount(gbin(g[ok]), minlength=NB).astype(float)
        return self._hg[key]

    def hist_logit(self, cid, krange=None) -> np.ndarray:
        key = (cid, krange)
        if key not in self._hl:
            g = self.G[cid]
            ok = np.isfinite(g)
            if krange is not None:
                ok &= (self.K[cid] >= krange[0]) & (self.K[cid] <= krange[1])
            gc = np.clip(g[ok], 1e-3, 1 - 1e-3)
            self._hl[key] = np.histogram(np.log(gc / (1 - gc)), LX_EDGES)[0].astype(
                float
            )
        return self._hl[key]

    def residues(self, ids) -> np.ndarray:
        """All scored G of the chains `ids` (one flat array)."""
        arrs = [self.G[c][np.isfinite(self.G[c])] for c in ids]
        return np.concatenate(arrs) if arrs else np.array([])


# ----------------------------------------------------------------------------- analysis sets
def base_filter(b: Build, exclude_membrane: bool) -> tuple[set, list]:
    """Unbound native solution chains (B0 of the soundness check): is_bound False; physical_state not
    denatured/unfolded; no cosolvent evidence; solution state; n_scored > 0. Optionally also no
    membrane mimetic (the label pipeline's chain filter; the tightened tiers handle mimetics instead).
    Returns the passing ids and the step list [(rule, Series mask)]."""
    r = b.rel
    ps = r.physical_state.fillna("").astype(str).str.strip().str.lower()
    steps = [
        ("is_bound False", ~r.is_bound.fillna(False).astype(bool)),
        ("physical_state not denatured/unfolded", ~ps.isin(DENATURED)),
        ("cosolvent_evidence False", ~r.cosolvent_evidence.fillna(False).astype(bool)),
        ("sample_state_evidence == solution", r.sample_state_evidence == "solution"),
        ("n_scored > 0", r.n_scored > 0),
    ]
    if exclude_membrane:
        steps.append(("membrane_mimetic null", r.membrane_mimetic.isna()))
    keep = np.logical_and.reduce([m.to_numpy() for _, m in steps])
    return set(r.id[keep]), steps


def define_set(
    b: Build, tiers, exclude_membrane: bool, chain_ids=None, apply_base=True
) -> tuple[set, list]:
    """Analysis set = chains whose score_tier is in `tiers`, within the base filter, optionally
    intersected with an explicit id list. Returns (ids, accounting rows)."""
    r = b.rel
    cur = b.in_tiers(tiers)
    acc = [{"step": f"score_tier in {list(tiers)}", "chains": len(cur)}]
    if chain_ids is not None:
        nxt = cur & set(chain_ids)
        acc.append(
            {
                "step": "in the given chain-id list",
                "chains": len(nxt),
                "removed": len(cur) - len(nxt),
            }
        )
        cur = nxt
    if apply_base:
        _, steps = base_filter(b, exclude_membrane)
        for lab, m in steps:
            nxt = cur & set(r.id[m.to_numpy()])
            acc.append(
                {"step": lab, "chains": len(nxt), "removed": len(cur) - len(nxt)}
            )
            cur = nxt
    sp = pd.Series([b.split[c] for c in cur]).value_counts().to_dict()
    acc.append(
        {
            "step": "analysis set",
            "chains": len(cur),
            "by_split": {str(k): int(v) for k, v in sp.items()},
        }
    )
    return cur, acc


def read_ids(spec: str) -> list:
    """`file.json[:key]` (a list, or a dict holding a list under key) or a text file, one id per line."""
    path, key = (
        (spec.split(".json:", 1)[0] + ".json", spec.split(".json:", 1)[1])
        if ".json:" in spec
        else (spec, "")
    )
    p = Path(path).expanduser()
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        return list(d[key] if key else d)
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]


# ----------------------------------------------------------------------------- label re-join
def _join(
    b: Build,
    lab: pd.DataFrame,
    chain_col: str,
    idx_col: str,
    seq_of: dict | None,
    name: str,
):
    """Attach build G / k to labelled residues; drop chains absent from the build or whose sequence
    differs, and residues unscored in the build. Returns (joined frame, counts)."""
    c = {"label_rows": len(lab), "label_chains": int(lab[chain_col].nunique())}
    ids = lab[chain_col].unique()
    in_build = {i for i in ids if i in b.G}
    c["chains_missing_from_build"] = len(set(ids) - in_build)
    if seq_of is None:
        same_seq = in_build
        c["sequence_check"] = "not possible (no label-time sequences)"
    else:
        same_seq = {i for i in in_build if seq_of.get(i) == b.seq[i]}
        c["sequence_check"] = "chain id and sequence"
        c["chains_sequence_differs"] = len(in_build - same_seq)
        c["chains_without_label_sequence"] = int(
            sum(1 for i in in_build if i not in seq_of)
        )
    d = lab[lab[chain_col].isin(same_seq)].copy()
    c["rows_dropped_with_chain"] = len(lab) - len(d)
    idx = d[idx_col].to_numpy()
    cid = d[chain_col].to_numpy()
    ln = np.array([len(b.G[x]) for x in cid])
    inr = idx < ln
    G = np.array([b.G[x][i] if ok else np.nan for x, i, ok in zip(cid, idx, inr)])
    K = np.array([b.K[x][i] if ok else np.nan for x, i, ok in zip(cid, idx, inr)])
    d["G_label_build"] = d.get("G", np.nan)
    d["G"], d["k"] = G, K
    keep = np.isfinite(G)
    c["rows_dropped_unscored_in_build"] = int((~keep).sum())
    d = d[keep].rename(columns={chain_col: "id", idx_col: "idx"})
    c["rows_joined"] = len(d)
    c["chains_joined"] = int(d.id.nunique())
    g0 = d.G_label_build.to_numpy(float)
    both = np.isfinite(g0)
    c["rows_G_changed_vs_label_build"] = int(
        np.sum(np.abs(d.G.to_numpy()[both] - g0[both]) > 1e-3)
    )
    c["duplicate_residue_rows"] = int(
        d.duplicated(["id", "idx"] + (["set"] if "set" in d else [])).sum()
    )
    d["ckey"] = d.id.map(b.key)
    print(f"[join {name}] {json.dumps(c)}", flush=True)
    return d.reset_index(drop=True), c


def join_xray(b: Build, residues: Path, chains: Path | None):
    lab = pd.read_parquet(residues, columns=["id", "set", "idx", "y", "G"])
    seq_of = None
    if chains is not None and Path(chains).exists():
        cf = pd.read_parquet(chains, columns=["id", "sequence"])
        seq_of = dict(zip(cf.id, cf.sequence))
    d, c = _join(b, lab, "id", "idx", seq_of, "xray A/B")
    c["by_set"] = {
        s: {
            "rows": int((d.set == s).sum()),
            "chains": int(d.id[d.set == s].nunique()),
            "positives": int(d.y[d.set == s].sum()),
        }
        for s in ("A", "B")
    }
    return d, c


def join_hetnoe(b: Build, path: Path):
    lab = pd.read_parquet(
        path, columns=["chain", "idx0", "val", "field", "sequence", "G"]
    )
    seq_of = lab.drop_duplicates("chain").set_index("chain").sequence.to_dict()
    lab = lab.drop(columns="sequence")
    d, c = _join(b, lab, "chain", "idx0", seq_of, "hetNOE C")
    d["field"] = np.where(
        d.field > 1e5, d.field / 1e6, d.field
    )  # Hz -> MHz where needed
    return d, c


# ----------------------------------------------------------------------------- components
def cluster_hist(keys, G, y, w=None):
    """-> (sorted unique cluster keys, pos (clusters x NB), neg (clusters x NB))."""
    keys = np.asarray(keys, dtype=object)
    if not len(keys):
        return [], np.zeros((0, NB)), np.zeros((0, NB))
    uk, inv = np.unique(keys.astype(str), return_inverse=True)
    bins = gbin(G)
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    pos = np.zeros((len(uk), NB))
    neg = np.zeros((len(uk), NB))
    np.add.at(pos, (inv, bins), w * y)
    np.add.at(neg, (inv, bins), w * (1 - y))
    return list(uk), pos, neg


def components(
    b: Build,
    chains: set,
    xray: pd.DataFrame,
    noe: pd.DataFrame,
    which="fit",
    krange=None,
    noe_cut=0.5,
    noe_cut_high_field=0.6,
    field_split=600.0,
):
    """Reference histograms (A, B, C, C_field) and the prior histogram (PI) of one chain set.

    which: "fit" = non-test chains, "test" = test-split chains. C: hetNOE < noe_cut at every field
    (primary); C_field: hetNOE < noe_cut up to field_split MHz and < noe_cut_high_field above (the
    field-aware sensitivity of the soundness check).
    """

    def sel(df):
        m = np.array(df.id.isin(chains).to_numpy(), dtype=bool)
        test = df.id.map(b.split).isin(TEST).to_numpy()
        m &= ~test if which == "fit" else test
        if krange is not None:
            m &= (df.k >= krange[0]).to_numpy() & (df.k <= krange[1]).to_numpy()
        return df[m]

    comps, meta = {}, {}
    xs = sel(xray)
    for s in ("A", "B"):
        d = xs[xs.set == s]
        comps[s] = cluster_hist(d.ckey.to_numpy(), d.G.to_numpy(), d.y.to_numpy())
        meta[s] = {
            "chains": int(d.id.nunique()),
            "clusters": len(comps[s][0]),
            "residues": len(d),
            "positives": int(d.y.sum()),
        }
    n = sel(noe)
    yc = (n.val < noe_cut).astype(int).to_numpy()
    comps["C"] = cluster_hist(n.ckey.to_numpy(), n.G.to_numpy(), yc)
    meta["C"] = {
        "chains": int(n.id.nunique()),
        "clusters": len(comps["C"][0]),
        "residues": len(n),
        "positives": int(yc.sum()),
    }
    cut = np.where(n.field.to_numpy() > field_split, noe_cut_high_field, noe_cut)
    yf = (n.val.to_numpy() < cut).astype(int)
    comps["C_field"] = cluster_hist(n.ckey.to_numpy(), n.G.to_numpy(), yf)
    meta["C_field"] = {
        **meta["C"],
        "positives": int(yf.sum()),
        "residues_above_split": int(np.sum(n.field.to_numpy() > field_split)),
    }
    pch = sorted(c for c in chains if (b.split[c] in TEST) == (which == "test"))
    keys = np.array([b.key[c] for c in pch], dtype=object).astype(str)
    uk, inv = (
        np.unique(keys, return_inverse=True)
        if len(keys)
        else (np.array([]), np.array([], int))
    )
    HG = np.zeros((len(uk), NB))
    HL = np.zeros((len(uk), len(LX_EDGES) - 1))
    if len(pch):
        np.add.at(HG, inv, np.stack([b.hist_g(c, krange) for c in pch]))
        np.add.at(HL, inv, np.stack([b.hist_logit(c, krange) for c in pch]))
    comps["PI"] = (list(uk), HG, HL)
    meta["PI"] = {"chains": len(pch), "clusters": len(uk), "residues": int(HG.sum())}
    return comps, meta
