#!/usr/bin/env python3
"""Reid #2 - alpha-synuclein + top-3 G-score flippers case study.

Produces a 4-panel figure showing per-residue G-score before vs after
re-referencing for:
  Panel A: BMRB 17665 (alpha-synuclein, mis-referenced) raw + re-referenced,
           plus BMRB 6968 (alpha-synuclein, ground-truth) as a third trace.
  Panel B/C/D: the three entries (excluding 17665) with the largest mean
           |delta-G| across the dataset, in the tolerant tier.

Note on --max-scan: defaults to 200 to keep wall-clock under ~5 min for
the talk's headline figure. Scaling to the full tolerant tier (~1800
entries) is cleaner but takes ~20-40 min and isn't required to make the
"this isn't cherry-picked" point.

Usage:
    uv run python scripts/case_study_gscore_flips.py \
        --baseline-tolerant data/baseline/tolerant.json \
        --bmrb-cache tmp/bmrb_entries \
        --potenci-cache tmp \
        --output docs/260505/figures/gscore_flips.png \
        --max-scan 200
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trizod.bmrb.bmrb as bmrb  # noqa: E402
import trizod.potenci.potenci as potenci  # noqa: E402
from trizod.scoring.scoring import (  # noqa: E402
    compute_gscores,
    convert_to_triplet_data,
    get_offset_corrected_shifts,
)
from trizod.trizod import load_potenci_cache, save_potenci_cache  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("case_study")


def score_entry(entry, cache_dir, mode):
    """Score the first peptide shift table on `entry` in the given mode.

    Returns (gscores, seq) or (None, None) on any failure (insufficient
    sequence, scoring error, etc.).
    """
    peptide_shifts = entry.get_peptide_shifts()
    for (_st_id, _ea_id, e_id), (shifts, cond_id, _, _) in peptide_shifts.items():
        if cond_id not in entry.conditions or e_id not in entry.entities:
            continue
        seq = entry.entities[e_id].seq
        if not seq or len(seq) < 20:
            continue
        cond = entry.conditions[cond_id]
        temp = cond.get_temperature(return_default=True)
        pH = cond.get_pH(return_default=True)
        ion = cond.get_ionic_strength(return_default=True)

        predshiftdct = load_potenci_cache(cache_dir, seq, temp, pH, ion)
        if predshiftdct is None:
            try:
                predshiftdct = potenci.get_pred_shifts(
                    seq, temp, pH, ion, pH != 7.0
                )
            except Exception:
                return None, None
            save_potenci_cache(cache_dir, seq, temp, pH, ion, predshiftdct)

        try:
            ret = get_offset_corrected_shifts(
                seq, shifts, predshiftdct, rereference_mode=mode
            )
        except Exception:
            return None, None
        if ret is None:
            return None, None
        _wdf, abs_wdf, cmp_mask = ret[0], ret[1], ret[2]
        if not np.any(cmp_mask):
            return None, None
        triplet_diffs, triplet_dof = convert_to_triplet_data(abs_wdf, cmp_mask)
        gscores = compute_gscores(triplet_diffs, triplet_dof, cmp_mask)
        return gscores, seq
    return None, None


def find_top_flippers(
    baseline_path, bmrb_cache, potenci_cache, exclude_ids, k=3, max_scan=0
):
    """Score every tolerant-tier entry with mode='none' and 'both', compute
    mean |delta-G|, return top-k entry ids by that metric.

    If max_scan > 0, only scan the first max_scan entries (for speed).
    """
    rows = []
    with open(baseline_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if max_scan and len(rows) > max_scan:
        rows = rows[:max_scan]

    deltas = []
    seen = set()
    for r in rows:
        eid = r.get("entryID")
        if not eid or eid in exclude_ids or eid in seen:
            continue
        seen.add(eid)
        pkl = bmrb_cache / f"{eid}.pkl"
        if not pkl.exists():
            continue
        try:
            with pkl.open("rb") as f:
                entry = pickle.load(f)
        except Exception:
            continue
        g_none, _ = score_entry(entry, potenci_cache, "none")
        g_both, _ = score_entry(entry, potenci_cache, "both")
        if g_none is None or g_both is None:
            continue
        with np.errstate(invalid="ignore"):
            mean_abs_delta = float(np.nanmean(np.abs(g_both - g_none)))
        if not np.isfinite(mean_abs_delta):
            continue
        deltas.append((eid, mean_abs_delta))
        if len(deltas) % 50 == 0:
            print(f"  scanned {len(deltas)} entries...", file=sys.stderr)

    deltas.sort(key=lambda kv: -kv[1])
    print(
        f"top by mean |delta-G|: {[(e, round(v, 3)) for e, v in deltas[:10]]}",
        file=sys.stderr,
    )
    return [eid for eid, _ in deltas[:k]]


def render(panels, out_path):
    """Render up to 4 panels as a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for ax, panel in zip(axes.flat, panels):
        traces = panel.get("traces", [])
        if not traces:
            ax.set_axis_off()
            continue
        for tr in traces:
            ax.plot(
                panel["residues"],
                tr["gscores"],
                tr["ystyle"],
                label=tr["label"],
                lw=1.4,
            )
        ax.axhline(0.5, color="grey", ls=":", lw=0.8, label="G=0.5 disorder threshold")
        ax.set_xlabel("residue")
        ax.set_ylabel("G-score")
        ax.set_ylim(0, 1)
        ax.set_title(panel["title"], fontsize=10)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure written to {out_path}")


def name_for(entry):
    """Best-effort protein name for the panel title."""
    peptide_shifts = entry.get_peptide_shifts()
    for (_st, _ea, e_id), _ in peptide_shifts.items():
        if e_id in entry.entities:
            n = entry.entities[e_id].name
            if n:
                return n
    return "?"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--baseline-tolerant", type=Path, default=Path("data/baseline/tolerant.json")
    )
    parser.add_argument("--bmrb-cache", type=Path, default=Path("tmp/bmrb_entries"))
    parser.add_argument("--potenci-cache", type=Path, default=Path("tmp"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/260505/figures/gscore_flips.png")
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=200,
        help="Scan only the first N tolerant-tier entries when ranking flippers (0 = scan all)",
    )
    args = parser.parse_args()

    panels = []

    # Panel A: alpha-synuclein 17665 + 6968 ground-truth
    pkl_17665 = args.bmrb_cache / "17665.pkl"
    pkl_6968 = args.bmrb_cache / "6968.pkl"
    if not pkl_17665.exists() or not pkl_6968.exists():
        # Re-parse fresh from .str (post-Step8) when pickles are absent
        bmrb_dir = Path("data/bmrb_entries")
        entry_17665 = bmrb.BmrbEntry("17665", bmrb_dir / "bmr17665")
        entry_6968 = bmrb.BmrbEntry("6968", bmrb_dir / "bmr6968")
    else:
        with pkl_17665.open("rb") as f:
            entry_17665 = pickle.load(f)
        with pkl_6968.open("rb") as f:
            entry_6968 = pickle.load(f)

    g_raw, seq_17665 = score_entry(entry_17665, args.potenci_cache, "none")
    g_ref, _ = score_entry(entry_17665, args.potenci_cache, "both")
    g_truth, seq_6968 = score_entry(entry_6968, args.potenci_cache, "both")

    if g_raw is None or g_ref is None:
        print("ERROR: could not score 17665", file=sys.stderr)
        sys.exit(1)

    residues_a = np.arange(1, len(seq_17665) + 1)
    if g_truth is not None and len(seq_6968) >= len(seq_17665):
        g_truth_aligned = g_truth[: len(seq_17665)]
    else:
        g_truth_aligned = np.full_like(g_raw, np.nan, dtype=float)
        if g_truth is not None:
            g_truth_aligned[: len(g_truth)] = g_truth

    panels.append(
        {
            "title": "alpha-synuclein BMRB 17665 (mis-referenced)",
            "residues": residues_a,
            "traces": [
                {"label": "17665 raw", "ystyle": "-", "gscores": g_raw},
                {"label": "17665 re-referenced", "ystyle": "--", "gscores": g_ref},
                {
                    "label": "6968 ground truth",
                    "ystyle": ":",
                    "gscores": g_truth_aligned,
                },
            ],
        }
    )

    # Panels B/C/D: top-3 flippers
    if args.baseline_tolerant.exists():
        top_ids = find_top_flippers(
            args.baseline_tolerant,
            args.bmrb_cache,
            args.potenci_cache,
            exclude_ids={"17665"},
            k=3,
            max_scan=args.max_scan,
        )
    else:
        print(
            f"WARNING: {args.baseline_tolerant} missing - skipping top-3 flippers",
            file=sys.stderr,
        )
        top_ids = []

    for eid in top_ids:
        pkl = args.bmrb_cache / f"{eid}.pkl"
        if pkl.exists():
            with pkl.open("rb") as f:
                entry = pickle.load(f)
        else:
            try:
                entry = bmrb.BmrbEntry(eid, Path("data/bmrb_entries") / f"bmr{eid}")
            except Exception:
                continue
        g_raw, seq = score_entry(entry, args.potenci_cache, "none")
        g_ref, _ = score_entry(entry, args.potenci_cache, "both")
        if g_raw is None or g_ref is None:
            continue
        residues = np.arange(1, len(seq) + 1)
        title = f"BMRB {eid}: {name_for(entry)[:40]}"
        panels.append(
            {
                "title": title,
                "residues": residues,
                "traces": [
                    {"label": "raw", "ystyle": "-", "gscores": g_raw},
                    {"label": "re-referenced", "ystyle": "--", "gscores": g_ref},
                ],
            }
        )

    while len(panels) < 4:
        panels.append({"title": "(no flipper)", "residues": np.arange(1, 2), "traces": []})

    render(panels, args.output)


if __name__ == "__main__":
    main()
