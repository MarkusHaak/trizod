#!/usr/bin/env python3
"""Reproduce CheZOD1325 Z-scores with TriZOD's CheZOD-equivalent pipeline.

KEY POINT: CheZOD never used LACS. TriZOD's `--rereference-mode potenci-only`
IS the CheZOD-equivalent method (POTENCI random-coil + AIC offset correction +
CheZOD Z-score, per Nielsen 2016). LACS is a TriZOD-only improvement and must be
EXCLUDED when reproducing CheZOD. This script compares CheZOD's published
Z-scores against TriZOD potenci-only (the reproduction) and, for contrast,
against the LACS "both" release.

The alignment/classification helpers live in ``trizod.figures.chezod``.

Regenerate the potenci-only scores first (LACS recorded but NOT applied):
    # subset dir of CheZOD BMRB entries already built at tmp/chezod_subset/
    uv run python -m trizod.trizod --input-dir tmp/chezod_subset \
        --filter-defaults unfiltered --rereference-mode potenci-only \
        --output-prefix docs/260611/data/chezod_verification/trizod_potenci_only \
        --output-format json --no-progress --processes 8 --cache-dir tmp

Outputs (gitignored): reproduce_summary.json, reproduce_genuine.csv.

Run: uv run python scripts/validation/reproduce_chezod.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from trizod.figures.chezod import load_chezod, load_trizod, summarize

ROOT = Path(__file__).resolve().parents[2]
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


def main():
    chezod = load_chezod(CHEZOD)
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
    OUT.mkdir(parents=True, exist_ok=True)
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
