"""Class balance of the release splits (release audit check F33).

Per split -- the training representatives of every tier, the TriZOD test set
and CheZOD117 -- this computes, from the release Parquet:

* the share of scored residues in each of the four G zones the three cuts make
  (G < 0.15, 0.15 <= G < 0.35, 0.35 <= G < 0.70, G >= 0.70);
* the share of scored residues with k < 10;
* the chain-length distribution (median and interquartile range);
* the prevalence pi of disorder and the posterior of disorder at each cut under
  that split's pi, where the threshold framework transported the calibration
  to the split (the Parquet's ``calibration`` metadata key,
  ``prevalence_by_split``). A fixed cut is a fixed likelihood ratio, so its
  posterior moves with pi: a test residue at G = 0.35 is not a 50 % call when
  the test set is more ordered than the calibration set.

Used by :mod:`trizod.dataset.finalize` (``class_balance.tsv`` and the tables in
the README and the datasheet) and by :mod:`trizod.dataset.export_jsonl` (the
UdonPred README).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from trizod.dataset.release_info import (
    CUTS,
    K_MIN_RELIABLE,
    ZONES,
    cut_label,
    posterior,
)
from trizod.dataset.tiers import TIER_ORDER

#: Parquet columns the table is computed from.
READ_COLUMNS = ("split", "train_tier", "gscores", "mask", "k", "length")
TEST_SPLITS = ("test_trizod", "test_chezod117")


def split_names() -> list[str]:
    """Row order of the table: training per tier (loose to strict), then the
    two test sets."""
    return [f"train_{t}" for t in TIER_ORDER] + list(TEST_SPLITS)


def _members(rows, name: str) -> list[dict]:
    if name.startswith("train_"):
        lo = TIER_ORDER.index(name.removeprefix("train_"))
        return [
            r
            for r in rows
            if r["split"] == "train"
            and r.get("train_tier") in TIER_ORDER
            and TIER_ORDER.index(r["train_tier"]) >= lo
        ]
    return [r for r in rows if r["split"] == name]


def _scored(rows) -> tuple[np.ndarray, np.ndarray]:
    """Finite G and their k over ``rows``' scored residues (mask true)."""
    gs, ks = [], []
    for r in rows:
        g = np.asarray(r["gscores"], dtype=float)
        m = np.asarray(r["mask"], dtype=bool) & np.isfinite(g)
        gs.append(g[m])
        ks.append(np.asarray(r["k"], dtype=float)[m])
    if not gs:
        return np.empty(0), np.empty(0)
    return np.concatenate(gs), np.concatenate(ks)


def _transported(calibration: dict | None, name: str) -> dict | None:
    """The framework's transport row for split ``name``: ``train_<tier>`` if it
    has one, ``train`` (every training representative, i.e. the unfiltered
    training set) for ``train_unfiltered``."""
    if not calibration:
        return None
    by_split = calibration.get("prevalence_by_split") or {}
    if name in by_split:
        return by_split[name]
    if name == "train_unfiltered":
        return by_split.get("train")
    return None


def em_prevalence(g: np.ndarray, log_lr, bin_width: float = 0.01) -> float:
    """Prevalence of disorder among residues with G values ``g``, by the
    threshold framework's prior-shift EM on its per-bin likelihood ratios:
    iterate pi <- mean over residues of P(disordered | G, pi)."""
    lr = np.exp(np.asarray(log_lr, dtype=float))
    bins = np.clip(
        np.floor(np.asarray(g) / bin_width + 1e-9).astype(int), 0, len(lr) - 1
    )
    h = np.bincount(bins, minlength=len(lr)) / len(bins)
    p = 0.2
    for _ in range(20000):
        pn = float(np.sum(h * lr * p / (lr * p + 1.0 - p)))
        if abs(pn - p) < 1e-12:
            return pn
        p = pn
    return p


def _lr_at_cuts(calibration: dict | None) -> dict[str, float] | None:
    lr = (calibration or {}).get("likelihood_ratio_at_cuts") or {}
    if not all(cut_label(c) in lr for c in CUTS):
        return None
    return {cut_label(c): lr[cut_label(c)]["value"] for c in CUTS}


def compute(rows, calibration: dict | None = None) -> list[dict]:
    """One record per split (:func:`split_names`); ``rows`` are Parquet rows
    with :data:`READ_COLUMNS`, ``calibration`` the Parquet's ``calibration``
    metadata (or None)."""
    out = []
    for name in split_names():
        members = _members(rows, name)
        g, k = _scored(members)
        n = len(g)
        lengths = np.asarray([r["length"] for r in members], dtype=float)
        rec = {
            "split": name,
            "chains": len(members),
            "scored_residues": n,
        }
        for zone, lo, hi in ZONES:
            inside = (g >= lo) & (g < hi) if hi is not None else g >= lo
            rec[f"share_{zone}"] = float(inside.mean()) if n else None
        rec[f"share_G_ge_{cut_label(CUTS[1])}"] = (
            float((g >= CUTS[1]).mean()) if n else None
        )
        rec[f"share_k_lt_{K_MIN_RELIABLE}"] = (
            float((k < K_MIN_RELIABLE).mean()) if n else None
        )
        if len(lengths):
            q1, med, q3 = np.percentile(lengths, [25, 50, 75])
            rec.update(
                length_median=float(med), length_q1=float(q1), length_q3=float(q3)
            )
        else:
            rec.update(length_median=None, length_q1=None, length_q3=None)
        tr = _transported(calibration, name)
        curve = (calibration or {}).get("lr_curve")
        lr_cuts = _lr_at_cuts(calibration)
        if tr is not None:
            pi = tr.get("pi") or {}
            rec["pi"] = pi.get("value")
            rec["pi_ci95"] = pi.get("ci95")
            post = tr.get("posterior_at_cuts") or {}
            for c in CUTS:
                p = post.get(cut_label(c)) or {}
                rec[f"posterior_at_{cut_label(c)}"] = p.get("value")
                rec[f"posterior_at_{cut_label(c)}_ci95"] = p.get("ci95")
            rec["pi_source"] = "threshold framework (transport, bootstrap CI)"
        elif curve and lr_cuts and n:
            pi = em_prevalence(g, curve["log_lr"], curve.get("bin_width", 0.01))
            rec["pi"] = round(pi, 4)
            rec["pi_ci95"] = None
            for c in CUTS:
                key = cut_label(c)
                rec[f"posterior_at_{key}"] = round(posterior(pi, lr_cuts[key]), 4)
                rec[f"posterior_at_{key}_ci95"] = None
            rec["pi_source"] = "EM on the calibration LR curve (point estimate)"
        else:
            rec["pi"] = rec["pi_ci95"] = None
            for c in CUTS:
                rec[f"posterior_at_{cut_label(c)}"] = None
                rec[f"posterior_at_{cut_label(c)}_ci95"] = None
            rec["pi_source"] = "not calibrated"
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

SPLIT_LABELS = {
    **{f"train_{t}": f"train, {t}" for t in TIER_ORDER},
    "test_trizod": "test_trizod",
    "test_chezod117": "test_chezod117",
}


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f} %"


def _num(x, digits=3) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def _int(x) -> str:
    return "–" if x is None else f"{x:,.0f}" if float(x).is_integer() else f"{x:,.1f}"


def markdown(table: list[dict]) -> str:
    """The table as Markdown (the README, the datasheet, the UdonPred README)."""
    c0, c1, c2 = (cut_label(c) for c in CUTS)
    head = (
        f"| Split | Chains | Scored residues | G < {c0} | {c0}–{c1} | {c1}–{c2} "
        f"| G ≥ {c2} | k < {K_MIN_RELIABLE} | Length, median [IQR] | π [95 % CI] "
        f"| P(disordered) at G = {c0} / {c1} / {c2} |"
    )
    lines = [head, "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in table:
        if r["length_median"] is None:
            length = "–"
        else:
            length = (
                f"{_int(r['length_median'])} [{_int(r['length_q1'])}–"
                f"{_int(r['length_q3'])}]"
            )
        if r["pi"] is None:
            pi = "–"
        elif r["pi_ci95"]:
            lo, hi = r["pi_ci95"]
            pi = f"{_num(r['pi'])} [{_num(lo)}, {_num(hi)}]"
        else:
            pi = f"{_num(r['pi'])}*"
        posts = [r[f"posterior_at_{cut_label(c)}"] for c in CUTS]
        post = "–" if all(p is None for p in posts) else " / ".join(map(_num, posts))
        lines.append(
            f"| {SPLIT_LABELS.get(r['split'], r['split'])} | {r['chains']:,} "
            f"| {r['scored_residues']:,} | {_pct(r['share_ordered'])} "
            f"| {_pct(r['share_indeterminate'])} | {_pct(r['share_disordered'])} "
            f"| {_pct(r['share_confidently_disordered'])} "
            f"| {_pct(r[f'share_k_lt_{K_MIN_RELIABLE}'])} | {length} | {pi} | {post} |"
        )
    return "\n".join(lines)


def notes() -> str:
    """The reading notes printed under the table."""
    return (
        "Training rows are the training representatives at that tier or "
        "stricter (the tiers nest). Shares are over scored residues (mask "
        "true). π is the reference-consensus prevalence of disorder, estimated "
        "for the split by the threshold framework's prior-shift EM: with a "
        "bootstrap 95 % interval where the framework transported the "
        "calibration to the split, as a point estimate (*) from its per-bin "
        "likelihood-ratio curve otherwise. The posterior is P(disordered | G at "
        "the cut) under that π. A fixed cut is a fixed likelihood ratio, so the "
        "same G means less on a more ordered split."
    )


def _flat(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if k.endswith("_ci95"):
            out[f"{k}_lo"], out[f"{k}_hi"] = v if v else (None, None)
        else:
            out[k] = v
    return out


def write_tsv(table: list[dict], path) -> Path:
    """``class_balance.tsv``: one row per split, every value unrounded."""
    path = Path(path)
    flat = [_flat(r) for r in table]
    fields: list[str] = []
    for r in flat:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in flat:
            w.writerow({k: _cell(r.get(k)) for k in fields})
    return path


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if math.isnan(v) else repr(round(v, 6))
    return v


def from_parquet(path) -> tuple[list[dict], dict | None]:
    """The table for a release Parquet, with the calibration read from its
    ``calibration`` metadata key (None when the build recorded none)."""
    from trizod.dataset.export_jsonl import read_release

    rows, meta = read_release(path, columns=READ_COLUMNS)
    calibration = parse_calibration(meta.get("calibration"))
    return compute(rows, calibration), calibration


def parse_calibration(raw: str | None) -> dict | None:
    """The ``calibration`` metadata value as a dict, or None when absent or
    not recorded."""
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None
