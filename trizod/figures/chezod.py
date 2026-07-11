"""CheZOD1325 reproduction helpers (Fig 3 + leakage target).

CheZOD never used LACS. TriZOD's ``--rereference-mode potenci-only`` IS the
CheZOD-equivalent method (POTENCI random-coil + AIC offset correction + CheZOD
Z-score, per Nielsen 2016); LACS is a TriZOD-only improvement and must be
EXCLUDED when reproducing CheZOD.

These helpers load CheZOD's published Z-scores and a TriZOD ``scores.json``,
align them per entry by the best sequence-match offset, and classify the
agreement. The CLI that drives them into a summary report lives at
``scripts/validation/reproduce_chezod.py``.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ATOMS = ["C", "CA", "CB", "H", "HA", "HB", "N"]
NA = 999.0  # CheZOD sentinel for terminal / no-data residues


def load_chezod(chezod_dir):
    """Load CheZOD1325 published sequences + Z-scores from ``chezod_dir``.

    Reads ``allseqs1325.txt`` (``<id> <seq>``) line-aligned with
    ``allscores1325newest.txt`` (comma-separated Z, 999 = NA). Returns
    ``{entry_id: {"seq": str, "z": [float|None]}}``.
    """
    chezod_dir = Path(chezod_dir)
    out = {}
    for sl, scl in zip(
        [x.rstrip() for x in (chezod_dir / "allseqs1325.txt").open() if x.strip()],
        [
            x.rstrip()
            for x in (chezod_dir / "allscores1325newest.txt").open()
            if x.strip()
        ],
    ):
        p = sl.split(None, 1)
        if len(p) < 2 or not p[0].isdigit():
            continue
        seq = p[1].replace(" ", "")
        vals = [float(v) for v in scl.strip().strip("[]").split(",") if v.strip()]
        if len(vals) == len(seq):
            out[p[0]] = {"seq": seq, "z": [None if v == NA else v for v in vals]}
    return out


def load_trizod(path):
    """Load a TriZOD ``scores.json`` (JSONL) keyed by entry ID.

    Returns ``{entry_id: [{"seq", "z", "off"}, ...]}`` where ``off`` is the max
    absolute per-atom POTENCI offset for the record.
    """
    by = {}
    for line in Path(path).open():
        if not line.strip():
            continue
        r = json.loads(line)
        off = max((abs(r.get(f"off_{a}") or 0.0) for a in ATOMS), default=0.0)
        by.setdefault(str(r["entryID"]), []).append(
            {"seq": r["seq"] or "", "z": r["zscores"] or [], "off": off}
        )
    return by


def best(cz, recs):
    """Return (record, offset) whose sequence best matches the CheZOD sequence."""
    bn, bo, br = -1, 0, None
    for r in recs:
        i, j, n = max(
            SequenceMatcher(
                None, cz["seq"], r["seq"], autojunk=False
            ).get_matching_blocks(),
            key=lambda b: b.size,
        )
        if n > bn:
            bn, bo, br = n, j - i, r
    return br, bo


def stats(cz, recs):
    """Per-entry concordance stats between CheZOD and the best TriZOD record."""
    rec, off = best(cz, recs)
    a, b = [], []
    for k, cv in enumerate(cz["z"]):
        m = k + off
        if 0 <= m < len(rec["z"]) and cv is not None and rec["z"][m] is not None:
            a.append(cv)
            b.append(rec["z"][m])
    if len(a) < 5:
        return None
    a, b = np.array(a), np.array(b)
    return {
        "rec": rec,
        "pearson": float(pearsonr(a, b)[0]),
        "mae": float(np.mean(np.abs(a - b))),
        "mean_shift": float(b.mean() - a.mean()),
        "n": len(a),
        "seq_match": cz["seq"] == rec["seq"],
    }


def classify(p, m):
    """Bucket an (pearson, mae) pair into agree / offset_shift / low_variance / genuine."""
    if p >= 0.9 and m <= 1.0:
        return "agree"
    if p >= 0.9 and m > 1.0:
        return "offset_shift"
    if p < 0.9 and m <= 0.8:
        return "low_variance"
    return "genuine"


def summarize(chezod, trizod, label):
    """Aggregate per-entry concordance into a summary dict + sorted genuine list."""
    cats, maes, genuine = {}, [], []
    for eid, cz in chezod.items():
        recs = trizod.get(eid)
        if not recs:
            continue
        s = stats(cz, recs)
        if s is None:
            continue
        c = classify(s["pearson"], s["mae"])
        cats[c] = cats.get(c, 0) + 1
        maes.append(s["mae"])
        if c == "genuine":
            genuine.append(
                {
                    "bmrb_id": eid,
                    "pearson": round(s["pearson"], 3),
                    "mae": round(s["mae"], 3),
                    "mean_shift_tz_minus_cz": round(s["mean_shift"], 2),
                    "max_potenci_off": round(s["rec"]["off"], 2),
                    "seq_match": s["seq_match"],
                    "n": s["n"],
                }
            )
    maes = np.array(maes)
    if maes.size == 0:
        # No CheZOD entry matched a TriZOD record with >=5 comparable residues
        # (empty / mismatched scores.json): return a zeroed summary instead of
        # dividing by len(maes) == 0 or calling np.percentile on an empty array.
        return {
            "label": label,
            "n": 0,
            "categories": cats,
            "mae_median": None,
            "mae_q3": None,
            "mae_p95": None,
            "consistent_frac": None,
        }, []
    return {
        "label": label,
        "n": int(len(maes)),
        "categories": cats,
        "mae_median": round(float(np.median(maes)), 3),
        "mae_q3": round(float(np.percentile(maes, 75)), 3),
        "mae_p95": round(float(np.percentile(maes, 95)), 3),
        "consistent_frac": round(
            (
                cats.get("agree", 0)
                + cats.get("low_variance", 0)
                + cats.get("offset_shift", 0)
            )
            / len(maes),
            3,
        ),
    }, sorted(genuine, key=lambda x: -x["mae"])
