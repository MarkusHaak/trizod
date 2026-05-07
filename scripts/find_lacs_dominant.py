#!/usr/bin/env python3
"""Rank tolerant-tier entries by 'LACS dominance' — fraction of the total
re-referencing G-score change attributable to LACS vs POTENCI/AIC alone.

LACS dominance = mean_residue |G_lacs - G_raw| /
                 (mean_residue |G_lacs - G_raw| + mean_residue |G_potenci - G_raw|)

Close to 1.0 → LACS catches the offset, POTENCI/AIC alone barely moves the
score.  Close to 0.0 → POTENCI/AIC alone is enough; LACS contributes little.

Also requires the entry to have a *meaningful* total change (mean |ΔG_both -
G_raw| > min_delta) so we don't surface trivial entries where both modes do
nothing.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from case_study_gscore_flips import score_entry  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("data/baseline/tolerant.json"))
    parser.add_argument("--bmrb-cache", type=Path, default=Path("tmp/bmrb_entries"))
    parser.add_argument("--potenci-cache", type=Path, default=Path("tmp"))
    parser.add_argument("--max-scan", type=int, default=400)
    parser.add_argument("--min-delta", type=float, default=0.05,
                        help="Minimum mean |G_both - G_raw| to qualify as a 'real flipper'")
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    (args.potenci_cache / "potenci").mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.baseline.open() if line.strip()]
    if args.max_scan:
        rows = rows[: args.max_scan]

    seen = set()
    results = []
    for r in rows:
        eid = r.get("entryID")
        if not eid or eid in seen:
            continue
        seen.add(eid)
        pkl = args.bmrb_cache / f"{eid}.pkl"
        if not pkl.exists():
            continue
        try:
            with pkl.open("rb") as f:
                entry = pickle.load(f)
        except Exception:
            continue

        try:
            g_none, _ = score_entry(entry, args.potenci_cache, "none")
            g_lacs, _ = score_entry(entry, args.potenci_cache, "lacs")
            g_pot, _ = score_entry(entry, args.potenci_cache, "potenci-only")
            g_both, _ = score_entry(entry, args.potenci_cache, "both")
        except Exception:
            continue
        if any(g is None for g in (g_none, g_lacs, g_pot, g_both)):
            continue

        with np.errstate(invalid="ignore"):
            d_lacs = float(np.nanmean(np.abs(g_lacs - g_none)))
            d_pot = float(np.nanmean(np.abs(g_pot - g_none)))
            d_both = float(np.nanmean(np.abs(g_both - g_none)))

        if not (np.isfinite(d_lacs) and np.isfinite(d_pot) and np.isfinite(d_both)):
            continue
        if d_both < args.min_delta:
            continue
        denom = d_lacs + d_pot
        if denom < 1e-6:
            continue
        dominance = d_lacs / denom

        results.append({
            "entryID": eid,
            "dominance": dominance,
            "d_lacs": d_lacs,
            "d_pot": d_pot,
            "d_both": d_both,
        })

        if len(results) % 50 == 0:
            print(f"  {len(results)} qualifying entries scanned...", file=sys.stderr)

    results.sort(key=lambda r: -r["dominance"])

    print(f"\nTop {args.top_k} by LACS dominance (mean |Δ_lacs| / (mean |Δ_lacs| + mean |Δ_pot|)):")
    print(f"{'entryID':>10}  {'dominance':>10}  {'|Δlacs|':>8}  {'|Δpot|':>8}  {'|Δboth|':>8}")
    for r in results[: args.top_k]:
        print(f"  {r['entryID']:>8}  {r['dominance']:>9.3f}  {r['d_lacs']:>8.3f}  {r['d_pot']:>8.3f}  {r['d_both']:>8.3f}")

    print(f"\nBottom {args.top_k} (POTENCI-dominant):")
    for r in results[-args.top_k:]:
        print(f"  {r['entryID']:>8}  {r['dominance']:>9.3f}  {r['d_lacs']:>8.3f}  {r['d_pot']:>8.3f}  {r['d_both']:>8.3f}")


if __name__ == "__main__":
    main()
