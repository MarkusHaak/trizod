#!/usr/bin/env python3
"""Build the NEW (non-22-April) figures for the 6 May talk."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "260505" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def per_tier_deltas():
    """Bar plot of entry counts per tier before/after finalization."""
    tiers = ["unfiltered", "tolerant", "moderate", "strict"]
    pre, post = [], []
    for t in tiers:
        old = ROOT / "data" / "baseline" / f"{t}.json"
        new = ROOT / "data" / "release" / t / "scores.json"
        pre.append(sum(1 for _ in old.open()) if old.exists() else 0)
        post.append(sum(1 for _ in new.open()) if new.exists() else 0)
    x = np.arange(len(tiers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width / 2, pre, width, label="22 April baseline", color="#cccccc")
    ax.bar(x + width / 2, post, width, label="finalized 5 May pipeline", color="#4C72B0")
    for xi, v in zip(x - width / 2, pre):
        ax.text(xi, v + 200, str(v), ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + width / 2, post):
        ax.text(xi, v + 200, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("entries passing")
    ax.set_title("entries per tier — before vs after finalization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "per_tier_deltas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("per_tier_deltas.png written")


def lacs_vs_potenci_overlap():
    """Scatter: LACS offset vs POTENCI residual offset. Strict tier."""
    src = ROOT / "data" / "release" / "strict" / "scores.json"
    if not src.exists():
        # Fallback: empty placeholder so the talk still compiles
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.text(0.5, 0.5, "rerun pending", ha="center", va="center", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(FIG / "lacs_vs_potenci_overlap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    pts_lacs, pts_pot, atoms = [], [], []
    color_map = {"CA": "#1f77b4", "CB": "#ff7f0e", "C": "#2ca02c"}
    with src.open() as f:
        for line in f:
            r = json.loads(line)
            for atom in ("CA", "CB", "C"):
                lacs = r.get(f"lacs_off_{atom}")
                pot = r.get(f"off_{atom}")
                if lacs is None or pot is None:
                    continue
                if isinstance(lacs, float) and (lacs != lacs):  # NaN
                    continue
                if isinstance(pot, float) and (pot != pot):
                    continue
                pts_lacs.append(float(lacs))
                pts_pot.append(float(pot))
                atoms.append(atom)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for atom in ("CA", "CB", "C"):
        sel = [i for i, a in enumerate(atoms) if a == atom]
        if not sel:
            continue
        xs = [pts_lacs[i] for i in sel]
        ys = [pts_pot[i] for i in sel]
        ax.scatter(xs, ys, s=8, alpha=0.4, color=color_map[atom], label=atom)
    if pts_lacs:
        lim = max(abs(min(pts_lacs + pts_pot)), abs(max(pts_lacs + pts_pot)))
        lim = max(lim, 1.0)
        # Clip to 5 ppm to keep the bulk readable
        lim = min(lim, 5.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.plot([-lim, lim], [-lim, lim], ls=":", color="red", lw=0.8, label="y = x")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("LACS offset (ppm)")
    ax.set_ylabel("POTENCI/AIC residual offset (ppm)")
    ax.set_title("LACS vs POTENCI residual — strict tier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "lacs_vs_potenci_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("lacs_vs_potenci_overlap.png written")


def flip_count_by_tier():
    """Stacked bar: entries with material LACS correction (>0.5 ppm on C/CA/CB) per tier."""
    tiers = ["tolerant", "moderate", "strict"]
    flipped = {t: 0 for t in tiers}
    not_flipped = {t: 0 for t in tiers}
    for t in tiers:
        src = ROOT / "data" / "release" / t / "scores.json"
        if not src.exists():
            continue
        with src.open() as f:
            for line in f:
                r = json.loads(line)
                touched = False
                for a in ("C", "CA", "CB"):
                    v = r.get(f"lacs_off_{a}", 0.0) or 0.0
                    try:
                        if abs(float(v)) > 0.5:
                            touched = True
                            break
                    except (TypeError, ValueError):
                        pass
                if touched:
                    flipped[t] += 1
                else:
                    not_flipped[t] += 1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(tiers))
    flip_arr = np.array([flipped[t] for t in tiers])
    nof_arr = np.array([not_flipped[t] for t in tiers])
    ax.bar(x, nof_arr, color="#cccccc", label="negligible LACS correction")
    ax.bar(x, flip_arr, bottom=nof_arr, color="#d62728", label=">0.5 ppm LACS correction on C/CA/CB")
    for xi, top, n in zip(x, nof_arr + flip_arr, flip_arr):
        total = nof_arr[list(x).index(xi)] + n
        pct = 100.0 * n / total if total else 0.0
        ax.text(xi, top + 200, f"{n} ({pct:.0f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("entries")
    ax.set_title("entries materially affected by LACS, by tier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "flip_count_by_tier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("flip_count_by_tier.png written")


def architecture_diagram():
    """Boxes-and-arrows pipeline diagram."""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.5, 1.5, 1.7, 1, "BMRB\nNMR-STAR"),
        (2.7, 1.5, 1.7, 1, "Step 8\nwildcards"),
        (4.9, 1.5, 1.4, 1, "LACS\noffsets"),
        (6.7, 1.5, 1.4, 1, "POTENCI\nresidual"),
        (8.5, 1.5, 1.4, 1, "Z/G\nscores"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)
    for x_start, x_end in [(2.2, 2.7), (4.4, 4.9), (6.3, 6.7), (8.1, 8.5)]:
        ax.annotate(
            "", xy=(x_end, 2), xytext=(x_start, 2), arrowprops=dict(arrowstyle="->", lw=1.5)
        )
    ax.annotate(
        "", xy=(6.0, 0.6), xytext=(6.0, 1.45), arrowprops=dict(arrowstyle="->", lw=1.2, color="grey")
    )
    ax.text(6.0, 0.4, ".str + JSON", ha="center", va="top", color="grey", fontsize=10)
    fig.savefig(FIG / "architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("architecture.png written")


if __name__ == "__main__":
    architecture_diagram()
    per_tier_deltas()
    lacs_vs_potenci_overlap()
    flip_count_by_tier()
