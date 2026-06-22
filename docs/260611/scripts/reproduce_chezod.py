#!/usr/bin/env python3
"""Reproduce CheZOD1325 Z-scores with TriZOD's CheZOD-equivalent pipeline.

KEY POINT: CheZOD never used LACS. TriZOD's `--rereference-mode potenci-only`
IS the CheZOD-equivalent method (POTENCI random-coil + AIC offset correction +
CheZOD Z-score, per Nielsen 2016). LACS is a TriZOD-only improvement and must be
EXCLUDED when reproducing CheZOD. This script compares CheZOD's published
Z-scores against TriZOD potenci-only (the reproduction) and, for contrast,
against the LACS "both" release.

Regenerate the potenci-only scores first (LACS recorded but NOT applied):
    # subset dir of CheZOD BMRB entries already built at tmp/chezod_subset/
    uv run python -m trizod.trizod --input-dir tmp/chezod_subset \
        --filter-defaults unfiltered --rereference-mode potenci-only \
        --output-prefix docs/260611/data/chezod_verification/trizod_potenci_only \
        --output-format json --no-progress --processes 8 --cache-dir tmp

Outputs (gitignored): reproduce_summary.json, reproduce_genuine.csv.
"""

from __future__ import annotations

import csv
import json
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[3]
CHEZOD = ROOT / "data" / "chezod" / "protein_nmr_1325"
PO = (
    ROOT
    / "docs"
    / "260611"
    / "data"
    / "chezod_verification"
    / "trizod_potenci_only.json"
)
BOTH = ROOT / "data" / "release" / "unfiltered" / "scores.json"
OUT = ROOT / "docs" / "260611" / "data" / "chezod_verification"
ATOMS = ["C", "CA", "CB", "H", "HA", "HB", "N"]
NA = 999.0


def load_chezod():
    out = {}
    for sl, scl in zip(
        [x.rstrip() for x in (CHEZOD / "allseqs1325.txt").open() if x.strip()],
        [x.rstrip() for x in (CHEZOD / "allscores1325newest.txt").open() if x.strip()],
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
    by = {}
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        off = max((abs(r.get(f"off_{a}") or 0.0) for a in ATOMS), default=0.0)
        by.setdefault(str(r["entryID"]), []).append(
            {"seq": r["seq"] or "", "z": r["zscores"] or [], "off": off}
        )
    return by


def best(cz, recs):
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
    if p >= 0.9 and m <= 1.0:
        return "agree"
    if p >= 0.9 and m > 1.0:
        return "offset_shift"
    if p < 0.9 and m <= 0.8:
        return "low_variance"
    return "genuine"


def summarize(chezod, trizod, label):
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


def main():
    chezod = load_chezod()
    po = load_trizod(PO)
    both = load_trizod(BOTH)

    po_sum, po_genuine = summarize(chezod, po, "potenci-only (CheZOD reproduction)")
    both_sum, _ = summarize(chezod, both, "both (LACS applied — TriZOD dataset)")

    off_driven = sum(
        1
        for g in po_genuine
        if abs(g["mean_shift_tz_minus_cz"]) > 0.8 or g["max_potenci_off"] >= 2.0
    )
    summary = {
        "potenci_only_reproduction": po_sum,
        "lacs_both_for_contrast": both_sum,
        "n_genuine_potenci_only": len(po_genuine),
        "genuine_offset_correction_driven": off_driven,
        "genuine": po_genuine,
    }
    (OUT / "reproduce_summary.json").write_text(json.dumps(summary, indent=2))
    with (OUT / "reproduce_genuine.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(po_genuine[0].keys()))
        w.writeheader()
        w.writerows(po_genuine)

    print("=== CheZOD reproduction via TriZOD potenci-only (LACS excluded) ===")
    for k, v in po_sum.items():
        print(f"  {k}: {v}")
    print("\n=== Contrast: LACS 'both' mode (the TriZOD dataset) ===")
    for k in ("n", "categories", "mae_median", "consistent_frac"):
        print(f"  {k}: {both_sum[k]}")
    print(
        f"\nGenuine (potenci-only): {len(po_genuine)}; "
        f"offset-correction-driven (|mean shift|>0.8 or POTENCI off>=2 ppm): {off_driven}"
    )
    for g in po_genuine[:12]:
        print(
            f"  bmr{g['bmrb_id']}: r={g['pearson']} mae={g['mae']} "
            f"shift={g['mean_shift_tz_minus_cz']:+.2f} potenci_off={g['max_potenci_off']}"
        )
    print(f"\nWrote {OUT / 'reproduce_summary.json'} and reproduce_genuine.csv")


if __name__ == "__main__":
    main()
