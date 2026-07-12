#!/usr/bin/env python3
"""Generate one CSP plot per bound/unbound pair for review.

Output:
    docs/260505/csp_pairs/<single>_vs_<bound>.png   (one per pair)
    docs/260505/csp_pairs/index.md                  (sortable index)

Used to (a) sanity-check the CSP analysis and (b) optionally pick a more
compelling headline example than FKBP12 for slide 9 of the talk.

Usage:
    uv run python scripts/csp_per_pair_grid.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from csp_analysis import (  # noqa: E402
    compute_pair_csp,
    find_pairs,
    load_baseline,
    shifts_for,
    trimmed_mean_threshold,
)

OUT_DIR = ROOT / "docs" / "260505" / "csp_pairs"
ALPHA_N = 5.0


def name_for(entry_id, cache_dir):
    """Look up a protein name from the cached BmrbEntry."""
    import pickle

    pkl = cache_dir / f"{entry_id}.pkl"
    if not pkl.exists():
        return None
    try:
        with pkl.open("rb") as f:
            entry = pickle.load(f)
    except Exception:
        return None
    for _e_id, ent in entry.entities.items():
        if ent.name:
            return ent.name
    return None


def render_pair(seq, single_row, bound_row, csp, threshold, cache_dir, out_path):
    n = len(seq)
    residues = np.arange(1, n + 1)
    csp_plot = np.where(np.isnan(csp), 0, csp)
    color = np.where(csp > threshold, "#d62728", "#4C72B0")

    name_s = name_for(single_row["entryID"], cache_dir) or "?"
    name = (name_s or "?")[:60]
    t_s = single_row.get("temperature")
    ph_s = single_row.get("pH")
    cond = []
    if t_s:
        cond.append(f"T={t_s:.0f}K")
    if ph_s:
        cond.append(f"pH={ph_s:.1f}")
    cond_str = " · ".join(cond) if cond else "?"

    n_above = int(np.sum(csp > threshold))
    max_csp = float(np.nanmax(csp)) if not np.all(np.isnan(csp)) else float("nan")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(residues, csp_plot, color=color)
    ax.axhline(
        threshold, color="red", ls="--", lw=1.0, label=f"threshold = {threshold:.3f}"
    )
    ax.set_xlabel("residue")
    ax.set_ylabel("CSP (ppm)")
    title = (
        f"{name}  ·  apo bmr{single_row['entryID']} vs bound bmr{bound_row['entryID']}\n"
        f"{cond_str}  ·  max CSP = {max_csp:.3f}  ·  {n_above} residues above threshold"
    )
    ax.set_title(title, fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return name, max_csp, n_above


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-tolerant",
        type=Path,
        default=Path("data/interim/baseline/tolerant.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("tmp/bmrb_entries"))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_baseline(args.baseline_tolerant)
    pairs = find_pairs(rows, args.cache_dir)
    print(f"found {len(pairs)} bound/unbound pairs", file=sys.stderr)

    # Compute global trimmed-mean threshold once (matches csp_analysis.py output).
    all_csp = []
    csp_by_pair = []
    for seq, s, b in pairs:
        arr_s, mask_s, _ = shifts_for(s["entryID"], args.cache_dir, s.get("entityID"))
        arr_b, mask_b, _ = shifts_for(b["entryID"], args.cache_dir, b.get("entityID"))
        if arr_s is None or arr_b is None or arr_s.shape != arr_b.shape:
            continue
        csp = compute_pair_csp(seq, arr_s, mask_s, arr_b, mask_b)
        all_csp.extend(csp[~np.isnan(csp)].tolist())
        csp_by_pair.append((seq, s, b, csp))
    _, threshold = trimmed_mean_threshold(all_csp)
    print(f"trimmed-mean threshold = {threshold:.3f}", file=sys.stderr)

    rendered = []
    for seq, s, b, csp in csp_by_pair:
        out_path = args.out_dir / f"bmr{s['entryID']}_vs_bmr{b['entryID']}.png"
        name, max_csp, n_above = render_pair(
            seq, s, b, csp, threshold, args.cache_dir, out_path
        )
        rendered.append(
            {
                "single": s["entryID"],
                "bound": b["entryID"],
                "name": name,
                "max_csp": max_csp,
                "n_above": n_above,
                "filename": out_path.name,
            }
        )

    rendered.sort(key=lambda r: -r["max_csp"] if not np.isnan(r["max_csp"]) else 0)

    index_path = args.out_dir / "index.md"
    with index_path.open("w") as f:
        f.write("# CSP per-pair review · sorted by max CSP descending\n\n")
        f.write(
            f"Threshold = **{threshold:.3f} ppm** (trimmed-mean + SD across all pairs)\n\n"
        )
        f.write(f"{len(rendered)} pairs rendered.\n\n")
        f.write(
            "| Rank | Protein | apo | bound | max CSP | residues > threshold | plot |\n"
        )
        f.write("|---:|---|---|---|---:|---:|---|\n")
        for i, r in enumerate(rendered, 1):
            n = (r["name"] or "?")[:50]
            mc = f"{r['max_csp']:.3f}" if not np.isnan(r["max_csp"]) else "?"
            f.write(
                f"| {i} | {n} | bmr{r['single']} | bmr{r['bound']} | "
                f"{mc} | {r['n_above']} | "
                f"![]({r['filename']}) |\n"
            )
    print(f"index written to {index_path}", file=sys.stderr)
    print(f"{len(rendered)} per-pair plots written to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
