"""TriZOD G-score thresholds: one Bayesian rule for the ordered edge, the binary cut and the disordered
edge (the final recommended scheme of the 2026-11-16 soundness check), computed on any build.

Scheme
  References  A  DisProt disorder vs residues observed in X-ray structures
              B  residues missing from every covering X-ray structure vs observed ones
              C  same-entry 15N{1H} hetNOE < 0.5 at every field (sensitivity: < 0.6 above 600 MHz)
  LR          per reference: isotonic P(pos | G) on 0.01 bins -> block odds / reference prior odds
              (Haldane 0.5), running maximum; PRIMARY = log LR interpolated linearly between block
              midpoints ("interp", thresholds on a 0.001 grid); the step function ("step") is the
              sensitivity
  Pool        consensus LR = sqrt(LR_S LR_C), LR_S = sqrt(LR_A LR_B) (weights 1/4, 1/4, 1/2: equal
              weight to structural and dynamic evidence, a choice)
  Prevalence  prior-shift MLE (Saerens EM) of the analysis set's fit chains = the reference-consensus
              prevalence pi; Rogan-Gladen (numerical check) and the logit mixture (disclosed former
              estimator) as alternatives
  Cuts        smallest G at which P = pi LR / (pi LR + 1 - pi) reaches 0.10 / 0.50 / 0.90, i.e. LR
              reaches (1/9, 1, 9) (1 - pi) / pi; disordered iff G >= t
  Uncertainty 95 % percentile intervals from a joint cluster bootstrap (sequence clusters resampled
              jointly across A, B, C and the prior chains; every step repeated)
  Set         chains whose score tier is --tier or stricter (default: the strict tier), unbound native
              solution chains only (is_bound False, not denatured/unfolded, no cosolvent evidence,
              solution, n_scored > 0); test splits held out of the fit
  Reporting   the paper values (0.15 / 0.35 / 0.70) are kept where each lies inside the 95 % interval
              of its estimate (primary estimator); otherwise flagged, with the nearest multiple of
              0.05 inside the interval as the rule value. Posteriors are quoted only at the stated
              prevalence; the LR at each cut is prevalence-free.

Inputs (CLI flag > environment variable > default under the data root, $TRIZOD_ROOT or the first
ancestor of this file holding data/interim):
  --scored-dir     $TRIZOD_SCORED_DIR       <root>/data/interim/scored  (<tier>/scores.json)
  --release        $TRIZOD_RELEASE_PARQUET  <root>/data/processed/trizod_dataset.parquet
  --xray-labels    $TRIZOD_XRAY_LABELS      <root>/data/interim/threshold-calibration-2026-11-16/out/residues.parquet
  --xray-chains    $TRIZOD_XRAY_CHAINS      chains_filtered.parquet next to --xray-labels (label-time sequences)
  --hetnoe-labels  $TRIZOD_HETNOE_LABELS    <root>/data/interim/threshold-calibration-hetnoe-2026-11-16/out/noe_residues_primary.parquet
  --out            $TRIZOD_THRESHOLD_FRAMEWORK_DIR (required)
The label files come from scripts/validation/threshold_calibration/{xray,hetnoe}/; their residues are
re-joined to the build by chain id and sequence (counts in results.json["join"]).

Outputs (in --out): results.json, curves_bins.csv, curves_fine.csv, thresholds.csv, lr_at_cuts.csv,
zone_shares.csv, transport.csv, replicates.csv, replicate_kappa_curve.csv, kcheck_zone.csv,
kcheck_replicates.csv, boot_thresholds.csv, g_histogram.csv, set_accounting.csv.

Run:
  uv run --extra figures --with pyarrow python \
      scripts/validation/threshold_calibration/framework/compute.py \
      --scored-dir <build>/data/interim/scored --release <build>/release/trizod_dataset.parquet --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inputs as inp  # noqa: E402
import thrlib as tl  # noqa: E402

PAPER = (
    0.15,
    0.35,
    0.70,
)  # published values (ordered edge, binary cut, disordered edge)
LEVELS = (0.10, 0.50, 0.90)
SENS_LEVELS = (0.05, 0.15, 0.85, 0.95)
LEVEL_NAME = {0.10: "ordered_edge", 0.50: "binary_cut", 0.90: "disordered_edge"}
W_CONSENSUS = {"A": 0.25, "B": 0.25, "C": 0.5}
POOLS = {
    "consensus": W_CONSENSUS,
    "structural_only": {"A": 0.5, "B": 0.5},
    "hetnoe_only": {"C": 1.0},
    "A_only": {"A": 1.0},
    "B_only": {"B": 1.0},
    "equal_ABC": {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
    "B_excluded": {"A": 0.5, "C": 0.5},
    "hetnoe_field_aware": {"A": 0.25, "B": 0.25, "C_field": 0.5},
}
ESTIMATORS = ("interp", "step")  # primary first
EVIDENCE = {
    "supporting": 2.08,
    "moderate": 4.33,
    "strong": 18.7,
    "very_strong": 350.0,
}  # Tavtigian 2018 odds
RG_CUT = 0.36  # Rogan-Gladen cut of the 2026-11-16 framework (any cut works; it is a moment check)
K_STRATA = {"k1-4": (1, 4), "k1-9": (1, 9), "k10-21": (10, 21), "all": None}
LOG: list = []


def log(*a) -> None:
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def _path(flag, env: str, default: Path) -> Path:
    if flag:
        return Path(flag).expanduser()
    v = os.environ.get(env)
    return Path(v).expanduser() if v else default


def clean(o):
    """JSON-safe copy: numpy scalars/arrays to Python, NaN/inf to None, sets sorted."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, set):
        return clean(sorted(o))
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return o


def thr_entry(t: np.ndarray, paper: float | None = None) -> dict:
    """Point estimate, CI, NA share and bootstrap SD of one threshold; rounding stability vs paper."""
    b = t[1:]
    fin = b[np.isfinite(b)]
    d = {
        "t": None if not np.isfinite(t[0]) else round(float(t[0]), 4),
        "CI": tl.ci(b, 4),
        "NA_share": round(float(np.mean(~np.isfinite(b))), 4) if len(b) else None,
        "boot_sd": round(float(np.std(fin)), 4) if len(fin) else None,
    }
    if paper is not None and len(fin):
        d["share_rounds_to_paper"] = round(float(np.mean(tl.round_to(fin) == paper)), 4)
        r5 = pd.Series(tl.round_to(fin)).value_counts(normalize=True).sort_index()
        d["rounded_distribution"] = {
            f"{k:.2f}": round(float(v), 4) for k, v in r5.items() if v >= 0.005
        }
    return d


# ----------------------------------------------------------------------------- calibration
def calibrate(name, b, chains, xray, noe, nboot, seed, full=True):
    """Fit the framework on the fit (non-test) chains of `chains`. Returns (summary, arrays)."""
    t0 = time.time()
    comps, meta = inp.components(b, chains, xray, noe, "fit")
    boot = tl.Boot(comps, nboot, seed, cuts=PAPER)
    res = {
        "counts_fit": meta,
        "n_union_clusters": boot.n_union_clusters,
        "nboot": nboot,
        "seed": seed,
    }
    res["auc"] = {
        r: tl.pt_ci(tl.auc_hist(boot.Hp[r], boot.Hn[r]), 4) for r in boot.refs
    }
    pi_rg = tl.rogan_gladen(boot, W_CONSENSUS, RG_CUT)
    pi_mix = tl.mixture_prior(boot.HL) if full else None
    res["prior_alternatives"] = {
        "rogan_gladen": tl.pt_ci(pi_rg, 4),
        "rogan_gladen_cut": RG_CUT,
    }
    if pi_mix is not None:
        res["prior_alternatives"]["logit_mixture"] = tl.pt_ci(pi_mix, 4)
    arrays = {"boot": boot}
    est_out = {}
    for est in ESTIMATORS:
        est_out[est] = {}
        for pool, w in POOLS.items():
            if not full and pool != "consensus":
                continue
            Lb, Lg, glo, Lc = boot.pooled(w, est)
            pi = tl.em_prior(Lb, boot.hG)
            post = tl.posterior(Lg, pi)
            levels = LEVELS + SENS_LEVELS
            th = {lev: tl.first_crossing(post, lev, glo) for lev in levels}
            paper_of = dict(zip(LEVELS, PAPER))
            d = {
                "pi": tl.pt_ci(pi, 4),
                "cuts": {
                    f"{lev:.2f}": thr_entry(th[lev], paper_of.get(lev))
                    for lev in levels
                },
            }
            if pool == "consensus":
                d.update(
                    consensus_details(boot, est, Lb, Lg, glo, Lc, pi, th, pi_rg, pi_mix)
                )
                arrays[est] = {
                    "Lb": Lb,
                    "Lg": Lg,
                    "glo": glo,
                    "Lc": Lc,
                    "pi": pi,
                    "post": post,
                    "th": th,
                }
            est_out[est][pool] = d
    res["estimators"] = est_out
    res["primary_estimator"] = ESTIMATORS[0]
    res["zone_lr"] = zone_lr(
        boot, est_out[ESTIMATORS[0]]["consensus"]["pi"][0], arrays[ESTIMATORS[0]]["pi"]
    )
    res["paper_check"] = paper_check(est_out)
    res["runtime_s"] = round(time.time() - t0, 1)
    c = est_out[ESTIMATORS[0]]["consensus"]
    s = est_out["step"]["consensus"]
    log(
        f"[{name}] chains fit {meta['PI']['chains']} | A {meta['A']['residues']} B {meta['B']['residues']} "
        f"C {meta['C']['residues']} res | pi {c['pi']} | interp "
        + " / ".join(
            f"{c['cuts'][f'{lv:.2f}']['t']} {c['cuts'][f'{lv:.2f}']['CI']}"
            for lv in LEVELS
        )
        + " | step "
        + " / ".join(
            f"{s['cuts'][f'{lv:.2f}']['t']} {s['cuts'][f'{lv:.2f}']['CI']}"
            for lv in LEVELS
        )
        + f" | step pi {s['pi']} [{res['runtime_s']}s]"
    )
    return res, arrays


def consensus_details(boot, est, Lb, Lg, glo, Lc, pi, th, pi_rg, pi_mix) -> dict:
    d = {}
    # LR and posterior (at the stated prevalence = the set's own pi) at the published values
    post_c = tl.posterior(Lc, pi)
    d["at_paper_values"] = {}
    for i, c in enumerate(PAPER):
        refs = {}
        for r in boot.refs:
            lr_r = boot.L[r][:, tl.gbin(c)] if est == "step" else boot.Lcut[r][:, i]
            refs[r] = tl.pt_ci(np.exp(lr_r), 4)
        d["at_paper_values"][f"{c:.2f}"] = {
            "LR": tl.pt_ci(np.exp(Lc[:, i]), 4),
            "posterior": tl.pt_ci(post_c[:, i], 4),
            "LR_per_reference": refs,
        }
    # LR levels that the posterior levels correspond to at pi
    p0 = pi[0]
    d["lr_levels_at_pi"] = {
        f"{lev:.2f}": round(float(lev / (1 - lev) * (1 - p0) / p0), 4) for lev in LEVELS
    }
    # prevalence-free statements: neutral point and evidence edges
    up1, _ = tl.lr_edges(Lg, glo, 1.0)
    d["neutral_point_LR1"] = tl.pt_ci(up1, 4)
    d["evidence_edges"] = {}
    for lab, o in EVIDENCE.items():
        up, dn = tl.lr_edges(Lg, glo, o)
        d["evidence_edges"][lab] = {
            "odds": o,
            "G_ge": tl.pt_ci(up, 4),
            "G_lt": tl.pt_ci(dn, 4),
        }
    # alternative priors (thresholds with the same LR)
    alts = {"rogan_gladen": pi_rg}
    if pi_mix is not None:
        alts["logit_mixture"] = pi_mix
    lo, hi = tl.ci(pi[1:])
    if lo is not None:
        alts["pi_CI_low_fixed"] = np.full(len(pi), lo)
        alts["pi_CI_high_fixed"] = np.full(len(pi), hi)
    d["thresholds_at_alternative_priors"] = {}
    for nm, pv in alts.items():
        po = tl.posterior(Lg, pv)
        d["thresholds_at_alternative_priors"][nm] = {
            "pi": tl.pt_ci(pv, 4),
            **{
                f"{lev:.2f}": thr_entry(tl.first_crossing(po, lev, glo))
                for lev in LEVELS
            },
        }
    # t_dis bootstrap shares (the soundness check's two failure directions)
    td = th[0.90][1:]
    td = td[np.isfinite(td)]
    if len(td):
        d["t_dis_boot_share_le_0.60"] = round(float(np.mean(td <= 0.60 + 1e-9)), 4)
        d["t_dis_boot_share_ge_0.85"] = round(float(np.mean(td >= 0.85 - 1e-9)), 4)
    return d


def zone_lr(boot, pi0, pi) -> dict:
    """Haldane zone LR per reference and pooled (consensus weights) for the published zones, with the
    posterior of each zone at the stated prevalence."""
    out = {}
    logs = {}
    for r in boot.refs:
        Hp, Hn = boot.Hp[r], boot.Hn[r]
        P, N = Hp.sum(1), Hn.sum(1)
        out[r] = {}
        for zn, sl in zip(tl.ZONE_NAMES, tl.zone_slices(PAPER)):
            lr = (Hp[:, sl].sum(1) + 0.5) / P / ((Hn[:, sl].sum(1) + 0.5) / N)
            logs[(r, zn)] = np.log(lr)
            out[r][zn] = tl.pt_ci(lr, 4)
    out["consensus"] = {}
    for zn in tl.ZONE_NAMES:
        lz = sum(w * logs[(r, zn)] for r, w in W_CONSENSUS.items())
        out["consensus"][zn] = {
            "LR": tl.pt_ci(np.exp(lz), 4),
            "posterior_at_pi": tl.pt_ci(tl.posterior(lz, pi), 4),
        }
    out["pi"] = pi0
    return out


def paper_check(est_out) -> dict:
    out = {}
    for est in ESTIMATORS:
        out[est] = {}
        c = est_out[est]["consensus"]["cuts"]
        for lev, paper in zip(LEVELS, PAPER):
            e = c[f"{lev:.2f}"]
            lo, hi = e["CI"]
            val, inside = tl.rule_value(e["t"], lo, hi, paper)
            out[est][LEVEL_NAME[lev]] = {
                "level": lev,
                "paper_value": paper,
                "estimate": e["t"],
                "CI": e["CI"],
                "inside_CI": bool(inside),
                "rule_value": val,
                "flag": None
                if inside
                else f"paper value {paper} outside the 95 % CI {e['CI']} ({est})",
                "share_rounds_to_paper": e.get("share_rounds_to_paper"),
            }
    return out


def calib_summary(res) -> dict:
    """Compact summary of a secondary calibration (precision/continuity checks)."""
    out = {"counts_fit": res["counts_fit"]}
    for est in ESTIMATORS:
        c = res["estimators"][est]["consensus"]
        out[est] = {
            "pi": c["pi"],
            **{LEVEL_NAME[lev]: c["cuts"][f"{lev:.2f}"] for lev in LEVELS},
        }
        out[est]["at_paper_values"] = c["at_paper_values"]
    out["paper_check"] = res["paper_check"]
    return out


# ----------------------------------------------------------------------------- validation
def replicate_groups(b, chains, with_k=False):
    """Same-condition replicate pairs: distinct entries of one sequence, |dpH| <= 0.2, |dT| <= 2 K,
    at most 20 pairs per sequence (as in the 2026-11-16 framework)."""
    ch = b.rel[b.rel.id.isin(chains)]
    sub = np.random.default_rng(0)
    groups, n_pairs = [], 0
    for _s, grp in ch.groupby("sequence", sort=True):
        rows = grp.drop_duplicates("entry_id")[["id", "ph", "temperature"]].to_dict(
            "records"
        )
        if len(rows) < 2:
            continue
        ok = [np.isfinite(r["ph"]) and np.isfinite(r["temperature"]) for r in rows]
        pairs = [
            (i, j)
            for i in range(len(rows))
            for j in range(i + 1, len(rows))
            if ok[i]
            and ok[j]
            and abs(rows[i]["ph"] - rows[j]["ph"]) <= 0.2
            and abs(rows[i]["temperature"] - rows[j]["temperature"]) <= 2
        ]
        if not pairs:
            continue
        if len(pairs) > 20:
            pairs = [pairs[x] for x in sub.choice(len(pairs), 20, replace=False)]
        A, B, KA, KB = [], [], [], []
        for i, j in pairs:
            gi, gj = b.G[rows[i]["id"]], b.G[rows[j]["id"]]
            m = np.isfinite(gi) & np.isfinite(gj)
            A.append(gi[m])
            B.append(gj[m])
            KA.append(b.K[rows[i]["id"]][m])
            KB.append(b.K[rows[j]["id"]][m])
        A, B = np.concatenate(A), np.concatenate(B)
        if len(A):
            groups.append(
                (A, B, np.concatenate(KA), np.concatenate(KB)) if with_k else (A, B)
            )
            n_pairs += len(pairs)
    return groups, n_pairs


def heldout(b, chains, xray, noe, cuts, pi, nboot, rng) -> dict:
    """Per reference on fit and on held-out test chains: zone LR (Haldane 0.5, unlike the 2026-11-16
    framework, so that empty zones in a replicate stay finite), zone posterior at pi, MCC at the binary
    cut, AUC (independent cluster bootstrap per reference)."""
    out = {"cuts": list(cuts), "pi": pi}
    for which in ("fit", "test"):
        comps, _ = inp.components(b, chains, xray, noe, which)
        out[which] = {}
        for nm in ("A", "B", "C"):
            uk, pos, neg = comps[nm]
            if len(uk) < 5:
                out[which][nm] = {"skipped_clusters": len(uk)}
                continue
            W = tl.boot_weights(len(uk), nboot, rng)
            Hp, Hn = W @ pos, W @ neg
            P, N = Hp.sum(1), Hn.sum(1)
            d = {
                "clusters": len(uk),
                "residues": int(pos.sum() + neg.sum()),
                "positives": int(pos.sum()),
            }
            for zn, sl in zip(tl.ZONE_NAMES, tl.zone_slices(cuts)):
                lr = ((Hp[:, sl].sum(1) + 0.5) / P) / ((Hn[:, sl].sum(1) + 0.5) / N)
                d[f"LR_{zn}"] = tl.pt_ci(lr, 4)
                d[f"post_{zn}"] = tl.pt_ci(lr * pi / (lr * pi + 1 - pi), 4)
            j = int(round(cuts[1] * tl.NB))
            tp, fp = Hp[:, j:].sum(1), Hn[:, j:].sum(1)
            fn, tn = P - tp, N - fp
            with np.errstate(invalid="ignore", divide="ignore"):
                mcc = (tp * tn - fp * fn) / np.sqrt(
                    (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
                )
            d["MCC_binary"] = tl.pt_ci(mcc, 4)
            d["AUC"] = tl.pt_ci(tl.auc_hist(Hp, Hn), 4)
            out[which][nm] = d
    return out


def kcheck(
    b, kset, xray, noe, pi_stated, nboot, seed, groups_k
) -> tuple[dict, list, list]:
    """k >= 10 usage check: per stratum of the shift count k, the consensus zone LR / posterior at the
    stated prevalence (joint cluster bootstrap on the k-check set), and replicate agreement by the
    smaller k of the residue pair."""
    zrows, rrows = [], []
    for nm, kr in K_STRATA.items():
        comps, meta = inp.components(b, kset, xray, noe, "fit", krange=kr)
        bt = tl.Boot(comps, nboot, seed, curves=False)
        row = {
            "k": nm,
            "residues_fit": meta["PI"]["residues"],
            **{f"n_{r}": meta[r]["residues"] for r in "ABC"},
        }
        row.update({f"clusters_{r}": meta[r]["clusters"] for r in "ABC"})
        logs = {}
        for r in ("A", "B", "C"):
            Hp, Hn = bt.Hp[r], bt.Hn[r]
            P, N = Hp.sum(1), Hn.sum(1)
            for zn, sl in zip(tl.ZONE_NAMES, tl.zone_slices(PAPER)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    logs[(r, zn)] = np.log(
                        (Hp[:, sl].sum(1) + 0.5) / P / ((Hn[:, sl].sum(1) + 0.5) / N)
                    )
        for zn in tl.ZONE_NAMES:
            lz = sum(w * logs[(r, zn)] for r, w in W_CONSENSUS.items())
            row[f"LR_{zn}"] = tl.pt_ci(np.exp(lz), 4)
            row[f"post_{zn}_at_stated_pi"] = tl.pt_ci(tl.posterior(lz, pi_stated), 4)
            row[f"share_{zn}"] = round(
                float(bt.hG[0][tl.zone_slices(PAPER)[tl.ZONE_NAMES.index(zn)]].sum()), 4
            )
        zrows.append(row)
        lo, hi = kr if kr is not None else (0, 99)
        sub = []
        for A, B, KA, KB in groups_k:
            km = np.minimum(KA, KB)
            s = (km >= lo) & (km <= hi)
            sub.append((A[s], B[s]))
        rs = tl.replicate_summary(sub, PAPER, nboot, np.random.default_rng(seed + 1))
        rrows.append({"k": nm, **rs})
    summ = {"stated_pi": pi_stated, "zone": zrows, "replicates": rrows}
    return summ, zrows, rrows


def spearman_boot(df, nboot, rng) -> dict:
    if len(df) < 10:
        return {"n": len(df)}
    G, V = df.G.to_numpy(), df.val.to_numpy()
    cl, inv = np.unique(df.ckey.astype(str).to_numpy(), return_inverse=True)
    W = tl.boot_weights(len(cl), nboot, rng)[1:].astype(int)
    bs = []
    for w in W:
        sel = np.repeat(np.arange(len(G)), w[inv])
        bs.append(spearmanr(G[sel], V[sel]).statistic)
    return {
        "rho": round(float(spearmanr(G, V).statistic), 4),
        "CI": tl.ci(bs, 4),
        "residues": len(df),
        "chains": int(df.id.nunique()),
        "clusters": len(cl),
    }


# ----------------------------------------------------------------------------- outputs
def curves_tables(arrays, out: Path) -> None:
    boot = arrays["boot"]

    def band(x, prefix):
        with np.errstate(invalid="ignore"):
            return {
                prefix: x[0],
                f"{prefix}_lo": np.nanpercentile(x[1:], 2.5, 0),
                f"{prefix}_hi": np.nanpercentile(x[1:], 97.5, 0),
            }

    st, it = arrays["step"], arrays["interp"]
    cols = {"G_lo": tl.EDGES, "G_mid": tl.XC, "n_fit_residues": boot.HG[0]}
    for r in boot.refs:
        cols[f"n_pos_{r}"] = boot.Hp[r][0]
        cols[f"n_neg_{r}"] = boot.Hn[r][0]
        cols.update(band(boot.L[r], f"logLR_step_{r}"))
        cols.update(band(tl.posterior(boot.L[r], st["pi"]), f"post_step_{r}"))
    cols.update(band(st["Lg"], "logLR_step"))
    cols.update(band(st["post"], "post_step"))
    cols.update(band(it["Lb"], "logLR_interp"))
    pd.DataFrame(cols).to_csv(out / "curves_bins.csv", index=False, float_format="%.6g")
    cols = {"G_lo": tl.FINE_LO, "G": tl.FINE}
    for r in boot.refs:
        cols.update(band(boot.Lfine[r], f"logLR_{r}"))
        cols.update(band(tl.posterior(boot.Lfine[r], it["pi"]), f"post_{r}"))
    cols.update(band(it["Lg"], "logLR"))
    cols.update(band(it["post"], "post"))
    pd.DataFrame(cols).to_csv(out / "curves_fine.csv", index=False, float_format="%.6g")
    rows = {"replicate": np.arange(1, boot.nboot + 1)}
    for est in ESTIMATORS:
        rows[f"pi_{est}"] = arrays[est]["pi"][1:]
        for lev in LEVELS:
            rows[f"{LEVEL_NAME[lev]}_{est}"] = arrays[est]["th"][lev][1:]
    pd.DataFrame(rows).to_csv(
        out / "boot_thresholds.csv", index=False, float_format="%.6g"
    )


def thresholds_table(res, set_name) -> list:
    rows = []
    for est, pools in res["estimators"].items():
        for pool, d in pools.items():
            for lev, e in d["cuts"].items():
                rows.append(
                    {
                        "set": set_name,
                        "estimator": est,
                        "pool": pool,
                        "prior": "pi_em",
                        "pi": d["pi"][0],
                        "pi_lo": d["pi"][1],
                        "pi_hi": d["pi"][2],
                        "level": lev,
                        "t": e["t"],
                        "t_lo": e["CI"][0],
                        "t_hi": e["CI"][1],
                        "NA_share": e["NA_share"],
                        "boot_sd": e["boot_sd"],
                    }
                )
            for prior, alt in d.get("thresholds_at_alternative_priors", {}).items():
                for lev in (f"{x:.2f}" for x in LEVELS):
                    e = alt[lev]
                    rows.append(
                        {
                            "set": set_name,
                            "estimator": est,
                            "pool": pool,
                            "prior": prior,
                            "pi": alt["pi"][0],
                            "pi_lo": alt["pi"][1],
                            "pi_hi": alt["pi"][2],
                            "level": lev,
                            "t": e["t"],
                            "t_lo": e["CI"][0],
                            "t_hi": e["CI"][1],
                            "NA_share": e["NA_share"],
                            "boot_sd": e["boot_sd"],
                        }
                    )
    return rows


def flat(prefix, v) -> dict:
    """[pt, lo, hi] -> three columns."""
    if isinstance(v, list) and len(v) == 3:
        return {prefix: v[0], f"{prefix}_lo": v[1], f"{prefix}_hi": v[2]}
    return {prefix: v}


# ----------------------------------------------------------------------------- main
def sha256(path: Path) -> str | None:
    if not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_digests(a) -> dict:
    """SHA-256 of every input file (provenance of a run)."""
    d = {
        k: sha256(getattr(a, k))
        for k in ("release", "xray_labels", "xray_chains", "hetnoe_labels")
    }
    d.update(
        {f"scores_{t}": sha256(a.scored_dir / t / "scores.json") for t in inp.TIERS}
    )
    return d


def parse_args(argv=None):
    root = inp.data_root()
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    ap.add_argument("--scored-dir", help="build scored dir holding <tier>/scores.json")
    ap.add_argument("--release", help="release Parquet of the same build")
    ap.add_argument(
        "--xray-labels",
        help="residues.parquet of xray/01_map_and_label.py (references A, B)",
    )
    ap.add_argument(
        "--xray-chains",
        help="chains_filtered.parquet of the same run (label-time sequences)",
    )
    ap.add_argument(
        "--hetnoe-labels",
        help="noe_residues_primary.parquet of hetnoe/03_analysis.py (reference C)",
    )
    ap.add_argument("--out", help="output directory")
    ap.add_argument(
        "--tier",
        default="strict",
        choices=inp.TIERS,
        help="calibrate on this tier and stricter (default strict)",
    )
    ap.add_argument(
        "--precision-tier",
        default="moderate",
        help="tier of the precision check ('none' to skip)",
    )
    ap.add_argument(
        "--kcheck-tier",
        default="moderate",
        help="tier of the k >= 10 usage check (strict has few low-k residues)",
    )
    ap.add_argument(
        "--exclude-membrane-mimetic",
        action="store_true",
        help="also drop chains with a membrane mimetic (the label pipeline's filter; reproduces the "
        "soundness check's 'strict_current' set on the 2026-11-11 build)",
    )
    ap.add_argument(
        "--chain-ids",
        help="restrict the analysis set to these ids: file.json[:key] or a text file",
    )
    ap.add_argument(
        "--no-base-filter",
        action="store_true",
        help="do not apply the unbound/native/solution filter",
    )
    ap.add_argument(
        "--nboot", type=int, default=1000, help="bootstrap replicates (default 1000)"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20261116,
        help="RNG seed (default 20261116, the framework's)",
    )
    ap.add_argument(
        "--core-only",
        action="store_true",
        help="calibration only (no precision/k checks, replicates, ...)",
    )
    a = ap.parse_args(argv)
    a.scored_dir = _path(
        a.scored_dir, "TRIZOD_SCORED_DIR", root / "data" / "interim" / "scored"
    )
    a.release = _path(
        a.release,
        "TRIZOD_RELEASE_PARQUET",
        root / "data" / "processed" / "trizod_dataset.parquet",
    )
    lab = root / "data" / "interim"
    a.xray_labels = _path(
        a.xray_labels,
        "TRIZOD_XRAY_LABELS",
        lab / "threshold-calibration-2026-11-16" / "out" / "residues.parquet",
    )
    a.xray_chains = _path(
        a.xray_chains,
        "TRIZOD_XRAY_CHAINS",
        a.xray_labels.parent / "chains_filtered.parquet",
    )
    a.hetnoe_labels = _path(
        a.hetnoe_labels,
        "TRIZOD_HETNOE_LABELS",
        lab
        / "threshold-calibration-hetnoe-2026-11-16"
        / "out"
        / "noe_residues_primary.parquet",
    )
    out = a.out or os.environ.get("TRIZOD_THRESHOLD_FRAMEWORK_DIR")
    if not out:
        ap.error("--out (or $TRIZOD_THRESHOLD_FRAMEWORK_DIR) is required")
    a.out = Path(out).expanduser()
    return a


def main(argv=None) -> None:
    a = parse_args(argv)
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    log(
        f"inputs: scored {a.scored_dir} | release {a.release} | xray {a.xray_labels} | hetnoe {a.hetnoe_labels}"
    )
    b = inp.Build(a.release, a.scored_dir)
    log("tiers:", json.dumps(b.tier_check))
    xray, jx = inp.join_xray(b, a.xray_labels, a.xray_chains)
    noe, jn = inp.join_hetnoe(b, a.hetnoe_labels)
    ids = inp.read_ids(a.chain_ids) if a.chain_ids else None
    tiers = inp.TIERS[inp.TIERS.index(a.tier) :]
    S, acc = inp.define_set(
        b, tiers, a.exclude_membrane_mimetic, ids, apply_base=not a.no_base_filter
    )
    log("analysis set:", json.dumps(acc[-1]))
    R = {
        "protocol": __doc__,
        "settings": {
            "paper_values": PAPER,
            "levels": LEVELS,
            "sensitivity_levels": SENS_LEVELS,
            "pools": POOLS,
            "estimators": ESTIMATORS,
            "primary_estimator": ESTIMATORS[0],
            "nboot": a.nboot,
            "seed": a.seed,
            "hetnoe_cut": 0.5,
            "hetnoe_field_aware": "< 0.5 up to 600 MHz, < 0.6 above",
            "evidence_odds": EVIDENCE,
            "tier": a.tier,
            "analysis_tiers": tiers,
            "exclude_membrane_mimetic": a.exclude_membrane_mimetic,
            "base_filter": not a.no_base_filter,
            "chain_ids": a.chain_ids,
            "rule": "disordered iff G >= t",
        },
        "inputs": {
            k: str(getattr(a, k))
            for k in (
                "scored_dir",
                "release",
                "xray_labels",
                "xray_chains",
                "hetnoe_labels",
            )
        },
        "input_sha256": input_digests(a),
        "tiers": b.tier_check,
        "join": {"xray": jx, "hetnoe": jn},
        "analysis_set": {
            "accounting": acc,
            "chains": len(S),
            "fit_chains": sum(1 for c in S if not b.is_test(c)),
            "test_chains": sum(1 for c in S if b.is_test(c)),
        },
    }
    pd.DataFrame(acc).to_csv(a.out / "set_accounting.csv", index=False)
    cal, arrays = calibrate(f"{a.tier}+", b, S, xray, noe, a.nboot, a.seed, full=True)
    R["calibration"] = cal
    curves_tables(arrays, a.out)
    trows = thresholds_table(cal, "calibration")
    est0 = ESTIMATORS[0]
    pi0 = cal["estimators"][est0]["consensus"]["pi"][0]
    cuts_est = tuple(
        cal["estimators"][est0]["consensus"]["cuts"][f"{lv:.2f}"]["t"] for lv in LEVELS
    )
    # LR / posterior at the published values, per estimator (prevalence-free LR; posterior at stated pi)
    lr_rows = []
    for est in ESTIMATORS:
        for c, v in cal["estimators"][est]["consensus"]["at_paper_values"].items():
            row = {
                "estimator": est,
                "G": c,
                "stated_pi": cal["estimators"][est]["consensus"]["pi"][0],
            }
            row.update(flat("LR", v["LR"]))
            row.update(flat("posterior", v["posterior"]))
            for r, x in v["LR_per_reference"].items():
                row.update(flat(f"LR_{r}", x))
            lr_rows.append(row)
    pd.DataFrame(lr_rows).to_csv(a.out / "lr_at_cuts.csv", index=False)

    if not a.core_only:
        # precision check on a looser tier (its own LR and prevalence)
        if a.precision_tier != "none":
            ptiers = inp.TIERS[inp.TIERS.index(a.precision_tier) :]
            P, pacc = inp.define_set(
                b,
                ptiers,
                a.exclude_membrane_mimetic,
                ids,
                apply_base=not a.no_base_filter,
            )
            pres, _ = calibrate(
                f"{a.precision_tier}+ (precision check)",
                b,
                P,
                xray,
                noe,
                a.nboot,
                a.seed,
                full=False,
            )
            R["precision_check"] = {
                "tier": a.precision_tier,
                "accounting": pacc,
                **calib_summary(pres),
            }
            trows += thresholds_table(pres, f"precision_{a.precision_tier}")
        # replicate agreement (same conditions): strict set and all unbound chains
        base_all, _ = inp.base_filter(b, a.exclude_membrane_mimetic)
        R["replicates"] = {}
        rep_rows, kc_rows = [], []
        grid = np.round(np.arange(1, 100) / 100, 2)
        for setname, ids_ in (("analysis_set", S), ("all_unbound_chains", base_all)):
            groups, n_pairs = replicate_groups(b, ids_)
            R["replicates"][setname] = {"n_pairs": n_pairs}
            for lab, cuts in (("paper_values", PAPER), ("estimated_cuts", cuts_est)):
                if any(c is None for c in cuts):
                    continue
                rs = tl.replicate_summary(
                    groups, cuts, a.nboot, np.random.default_rng(a.seed + 1)
                )
                rs["n_pairs"] = n_pairs
                R["replicates"][setname][lab] = rs
                rep_rows.append(
                    {
                        "set": setname,
                        "cuts_from": lab,
                        **{
                            k: (v if not isinstance(v, list) else v[0])
                            for k, v in rs.items()
                            if k != "cuts"
                        },
                        "cuts": "/".join(f"{c:.3f}" for c in cuts),
                        "kappa_binary_lo": rs["kappa_binary"][1],
                        "kappa_binary_hi": rs["kappa_binary"][2],
                    }
                )
            kc = tl.kappa_curve(
                groups, grid, a.nboot, np.random.default_rng(a.seed + 1)
            )
            kc_rows.append(
                pd.DataFrame(
                    {
                        "set": setname,
                        "t": kc["t"],
                        "kappa": kc["kappa"],
                        "lo": kc["lo"],
                        "hi": kc["hi"],
                    }
                )
            )
        pd.DataFrame(rep_rows).to_csv(a.out / "replicates.csv", index=False)
        pd.concat(kc_rows).to_csv(
            a.out / "replicate_kappa_curve.csv", index=False, float_format="%.5f"
        )
        # held-out test chains (published zones and binary cut)
        R["heldout"] = heldout(
            b, S, xray, noe, PAPER, pi0, a.nboot, np.random.default_rng(a.seed + 2)
        )
        # k >= 10 usage check on a tier with enough low-k residues
        kt = inp.TIERS[inp.TIERS.index(a.kcheck_tier) :]
        K, kacc = inp.define_set(
            b, kt, a.exclude_membrane_mimetic, ids, apply_base=not a.no_base_filter
        )
        gk, _ = replicate_groups(b, K, with_k=True)
        ks, kz, kr = kcheck(b, K, xray, noe, pi0, a.nboot, a.seed + 3, gk)
        allk = np.concatenate([b.K[c][np.isfinite(b.G[c])] for c in b.G])
        setk = np.concatenate([b.K[c][np.isfinite(b.G[c])] for c in S])
        ks["share_k_lt_10"] = {
            "release_all_chains": round(float(np.mean(allk < 10)), 4),
            "analysis_set": round(float(np.mean(setk < 10)), 4),
        }
        ks["tier"] = a.kcheck_tier
        ks["chains"] = len(K)
        R["k_check"] = ks
        pd.DataFrame(
            [
                {kk: (v if not isinstance(v, list) else v[0]) for kk, v in r.items()}
                for r in kz
            ]
        ).to_csv(a.out / "kcheck_zone.csv", index=False)
        pd.DataFrame(
            [
                {
                    kk: (v if not isinstance(v, list) else v[0])
                    for kk, v in r.items()
                    if kk != "cuts"
                }
                for r in kr
            ]
        ).to_csv(a.out / "kcheck_replicates.csv", index=False)
        log(
            "k check:",
            json.dumps(
                {r["k"]: [r["kappa_binary"], r.get("disordered_flip")] for r in kr}
            ),
        )
        # sets for transport and zone shares
        split_ids = {
            s: {c for c, x in b.split.items() if x == s}
            for s in ("train", "test_trizod", "test_chezod117")
        }
        sets = {
            "analysis_set": S,
            "analysis_set_fit": {c for c in S if not b.is_test(c)},
            "analysis_set_test": {c for c in S if b.is_test(c)},
            **{f"tier_{t}+": b.nested(t) for t in inp.TIERS},
            **{f"tier_{t}_only": b.in_tiers((t,)) for t in inp.TIERS[:-1]},
            "all_unbound_chains": base_all,
            **split_ids,
        }
        it = arrays[est0]
        trans = []
        for nm, ids_ in sets.items():
            if not ids_:
                continue
            H = np.sum([b.hist_g(c) for c in ids_], 0)
            h = np.broadcast_to(H / H.sum(), it["Lb"].shape)
            p = tl.em_prior(it["Lb"], h)
            pc = tl.posterior(it["Lc"], p)
            row = {
                "set": nm,
                "chains": len(ids_),
                "residues": int(H.sum()),
                **flat("pi_em", tl.pt_ci(p, 4)),
            }
            for i, c in enumerate(PAPER):
                row.update(flat(f"posterior_at_{c:.2f}", tl.pt_ci(pc[:, i], 4)))
            trans.append(row)
        for pv in (0.05, 0.10, 0.20, 0.30, 0.50):
            pc = tl.posterior(it["Lc"], np.full(len(it["Lc"]), pv))
            row = {"set": f"fixed prevalence {pv:.2f}", "pi_em": pv}
            for i, c in enumerate(PAPER):
                row.update(flat(f"posterior_at_{c:.2f}", tl.pt_ci(pc[:, i], 4)))
            trans.append(row)
        R["transport"] = trans
        pd.DataFrame(trans).to_csv(a.out / "transport.csv", index=False)
        zs = []
        for nm, ids_ in sets.items():
            g = b.residues(ids_)
            if not len(g):
                continue
            for lab, cuts in (("paper_values", PAPER), ("estimated_cuts", cuts_est)):
                if any(c is None for c in cuts):
                    continue
                zs.append(
                    {
                        "set": nm,
                        "cuts_from": lab,
                        "cuts": "/".join(f"{c:.3f}" for c in cuts),
                        "chains": len(ids_),
                        "residues": len(g),
                        "ordered": round(float(np.mean(g < cuts[0])), 5),
                        "indeterminate": round(
                            float(np.mean((g >= cuts[0]) & (g < cuts[2]))), 5
                        ),
                        "disordered": round(float(np.mean(g >= cuts[2])), 5),
                        "binary_disordered": round(float(np.mean(g >= cuts[1])), 5),
                    }
                )
        R["zone_shares"] = zs
        pd.DataFrame(zs).to_csv(a.out / "zone_shares.csv", index=False)
        gh = {"G_lo": tl.EDGES}
        for nm in ("analysis_set", "tier_unfiltered+"):
            gh[nm] = np.sum([b.hist_g(c) for c in sets[nm]], 0).astype(int)
        pd.DataFrame(gh).to_csv(a.out / "g_histogram.csv", index=False)
        # hetNOE vs G (all joined residues; analysis-set fit residues)
        rng = np.random.default_rng(a.seed + 4)
        fit_ids = sets["analysis_set_fit"]
        R["hetnoe_spearman"] = {
            "all_joined": spearman_boot(noe, a.nboot, rng),
            "analysis_set_fit": spearman_boot(noe[noe.id.isin(fit_ids)], a.nboot, rng),
        }
    pd.DataFrame(trows).to_csv(a.out / "thresholds.csv", index=False)
    R["runtime_s"] = round(time.time() - t0, 1)
    (a.out / "results.json").write_text(json.dumps(clean(R), indent=1, allow_nan=False))
    pc = cal["paper_check"][est0]
    log(
        "paper values:",
        json.dumps(
            {
                k: [v["paper_value"], v["estimate"], v["CI"], v["inside_CI"]]
                for k, v in pc.items()
            }
        ),
    )
    log(f"done in {R['runtime_s']} s -> {a.out}")
    (a.out / "log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
