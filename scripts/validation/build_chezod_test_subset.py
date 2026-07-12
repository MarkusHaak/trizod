#!/usr/bin/env python3
"""Build the committed fixture for tests/test_chezod_equality.py.

Selects CheZOD1325 entries that (a) have a TriZOD potenci-only record whose
sequence EXACTLY matches CheZOD's, (b) reproduce the published Z-scores with a
comfortable margin (per-residue max|diff| <= SELECT_TOL over mutually-valid
positions), and (c) have enough comparable residues. Writes a small reference
file with the CheZOD sequence + published Z-scores per entry so the test does
not need the large gitignored allscores file.

Inputs (gitignored): data/external/chezod/protein_nmr_1325/{allseqs1325.txt,
allscores1325newest.txt}; data/interim/chezod_verification/repro_baseline.json
(TriZOD potenci-only scores for the CheZOD subset).
Output (committed): tests/reference/chezod_zscores_subset.json.

Run: uv run python scripts/validation/build_chezod_test_subset.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHEZOD = ROOT / "data" / "external" / "chezod" / "protein_nmr_1325"
BASELINE = ROOT / "data" / "interim" / "chezod_verification" / "repro_baseline.json"
BMRB = ROOT / "data" / "raw" / "bmrb_entries"
OUT = ROOT / "tests" / "reference" / "chezod_zscores_subset.json"

NA = 999.0
SELECT_TOL = 0.03  # margin below the test's atol=0.1
MIN_VALID = 20
TARGET = 30


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
            out[p[0]] = {"seq": seq, "z": vals}  # keep 999 sentinels
    return out


def load_baseline():
    by = {}
    for line in BASELINE.open():
        if not line.strip():
            continue
        r = json.loads(line)
        by.setdefault(str(r["entryID"]), []).append(
            {"seq": r["seq"] or "", "z": r["zscores"] or []}
        )
    return by


def main():
    chez, base = load_chezod(), load_baseline()
    selected = {}
    for eid in sorted(chez, key=int):
        if not (BMRB / f"bmr{eid}").is_dir():
            continue  # test must be able to parse it
        cz = chez[eid]
        exact = [r for r in base.get(eid, []) if r["seq"] == cz["seq"]]
        if not exact:
            continue
        rec = exact[0]
        diffs = [
            abs(rec["z"][i] - cv)
            for i, cv in enumerate(cz["z"])
            if cv != NA and i < len(rec["z"]) and rec["z"][i] is not None
        ]
        if len(diffs) < MIN_VALID or max(diffs) > SELECT_TOL:
            continue
        selected[eid] = {"seq": cz["seq"], "zscores": cz["z"]}
        if len(selected) >= TARGET:
            break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(selected, indent=1))
    print(f"selected {len(selected)} entries -> {OUT}")
    print("ids:", " ".join(selected))


if __name__ == "__main__":
    main()
