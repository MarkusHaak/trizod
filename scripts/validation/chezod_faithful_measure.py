#!/usr/bin/env python3
"""Measure how faithfully a TriZOD potenci-only scores.json reproduces CheZOD1325.

Reports the legacy bit-equal verdict (per-residue max|diff| <= atol, the
assert_allclose(atol=0.1) criterion from the 2023 test_chezod_equality.py) plus
bit-exact counts and median MAE, after sequence-offset alignment.

Usage: uv run python scripts/validation/chezod_faithful_measure.py <scores.json> [label]
"""

from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[2])
CHEZOD = f"{ROOT}/data/chezod/protein_nmr_1325"
NA = 999.0


def load_chezod():
    out = {}
    for sl, scl in zip(
        [x.rstrip() for x in open(f"{CHEZOD}/allseqs1325.txt") if x.strip()],
        [x.rstrip() for x in open(f"{CHEZOD}/allscores1325newest.txt") if x.strip()],
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
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        by.setdefault(str(r["entryID"]), []).append(
            {"seq": r["seq"] or "", "z": r["zscores"] or []}
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


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    chez, tz = load_chezod(), load_trizod(path)
    maxdiffs, maes, n = [], [], 0
    for eid, cz in chez.items():
        recs = tz.get(eid)
        if not recs:
            continue
        rec, off = best(cz, recs)
        a, b = [], []
        for k, cv in enumerate(cz["z"]):
            m = k + off
            if 0 <= m < len(rec["z"]) and cv is not None and rec["z"][m] is not None:
                a.append(cv)
                b.append(rec["z"][m])
        if len(a) < 5:
            continue
        d = np.abs(np.array(b) - np.array(a))
        maxdiffs.append(float(d.max()))
        maes.append(float(d.mean()))
        n += 1
    maxdiffs, maes = np.array(maxdiffs), np.array(maes)
    print(f"=== {label} ===")
    print(f"  entries compared: {n}")
    print(
        f"  LEGACY VERDICT (all residues within atol): "
        f"<=0.1: {int((maxdiffs <= 0.1).sum())}  "
        f"<=0.01: {int((maxdiffs <= 0.01).sum())}  "
        f"<=1e-4: {int((maxdiffs <= 1e-4).sum())}"
    )
    print(
        f"  per-entry MAE bit-exact (<0.01): {int((maes < 0.01).sum())}  "
        f"| <0.1: {int((maes < 0.1).sum())}"
    )
    print(
        f"  median MAE: {np.median(maes):.4f}  Q3: {np.percentile(maes, 75):.4f}  "
        f"p95: {np.percentile(maes, 95):.4f}"
    )


if __name__ == "__main__":
    main()
