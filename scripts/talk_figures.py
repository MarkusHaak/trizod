#!/usr/bin/env python3
"""Build the figures for the 6 May talk."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "260505" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


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


def workflow_diagram():
    """End-to-end TriZOD workflow with stage numbers + a TODO branch.

    Slide 2 centerpiece. Six stages left to right, with the headline number
    annotated under each. A dashed TODO branch hangs off the output stage.
    """
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(-0.5, 6)
    ax.axis("off")

    # Implemented stages: (x, y, w, h, title, body)
    stages = [
        (0.2,  3.6, 2.0, 1.4, "BMRB NMR-STAR",   "17,388 entries"),
        (2.6,  3.6, 2.2, 1.4, "Parser",          "methyl wildcards\n(Leu CDx, Val CGx)"),
        (5.2,  3.6, 2.2, 1.4, "Filter (4 tiers)","unfilt 16,851\ntol 15,433\nmod 10,107\nstr 3,033"),
        (7.8,  3.6, 2.4, 1.4, "Re-referencing",  "LACS pre-correction\n+ POTENCI/AIC residual"),
        (10.6, 3.6, 1.6, 1.4, "Scoring",         "Z-score · G-score\n3-residue triplet"),
        (12.4, 3.6, 1.0, 1.4, "Output",          ".str + JSON\n+ Zenodo meta"),
    ]
    for x, y, w, h, title, body in stages:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=1.6, ec="#361a54"))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top",
                fontsize=12, fontweight="bold", color="#361a54")
        ax.text(x + w / 2, y + 0.25, body, ha="center", va="bottom",
                fontsize=9, color="#361a54")

    # Forward arrows
    for s1, s2 in zip(stages[:-1], stages[1:]):
        x_start = s1[0] + s1[2]
        x_end = s2[0]
        y = s1[1] + s1[3] / 2
        ax.annotate("", xy=(x_end, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#361a54"))

    # TODO branch off the Output stage
    out_x = stages[-1][0] + stages[-1][2] / 2
    out_y = stages[-1][1]
    todo_x, todo_y, todo_w, todo_h = 5.0, 0.2, 7.5, 2.6
    ax.annotate(
        "",
        xy=(todo_x + todo_w / 2, todo_y + todo_h),
        xytext=(out_x, out_y),
        arrowprops=dict(arrowstyle="->", lw=1.4, color="#888888", ls="dashed"),
    )
    ax.add_patch(plt.Rectangle(
        (todo_x, todo_y), todo_w, todo_h,
        fill=True, facecolor="#fafafa", lw=1.4, ec="#888888", ls="dashed",
    ))
    ax.text(
        todo_x + 0.18, todo_y + todo_h - 0.18, "TODO (post-talk)",
        ha="left", va="top", fontsize=11, fontweight="bold", color="#666666",
    )
    todos = [
        "exclude multi-molecule (bound) entries that distort G-scores",
        "per-sequence representative pick (best conditions → median G-score)",
        "mmseqs2 sequence clustering for ML train/val/test split",
    ]
    for i, t in enumerate(todos):
        ax.text(todo_x + 0.18, todo_y + todo_h - 0.7 - 0.55 * i,
                f"·  {t}", ha="left", va="top", fontsize=10, color="#444444")

    # Title
    ax.text(6.75, 5.55, "TriZOD pipeline — input to output",
            ha="center", va="bottom", fontsize=14, fontweight="bold",
            color="#361a54")

    fig.tight_layout()
    fig.savefig(FIG / "workflow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("workflow.png written")


def _gather_all_csp_pairs():
    """Walk the bound/unbound pairs and return [(seq, single_id, bound_id, csp_array)]."""
    from csp_analysis import (  # noqa: E402
        compute_pair_csp,
        find_pairs,
        load_baseline,
        shifts_for,
    )

    baseline = ROOT / "data" / "baseline" / "tolerant.json"
    cache = ROOT / "tmp" / "bmrb_entries"
    rows = load_baseline(baseline)
    pairs = find_pairs(rows, cache)
    out = []
    for seq, s, b in pairs:
        arr_s, mask_s, _ = shifts_for(s["entryID"], cache, s.get("entityID"))
        arr_b, mask_b, _ = shifts_for(b["entryID"], cache, b.get("entityID"))
        if arr_s is None or arr_b is None or arr_s.shape != arr_b.shape:
            continue
        csp = compute_pair_csp(seq, arr_s, mask_s, arr_b, mask_b)
        if np.all(np.isnan(csp)):
            continue
        out.append((seq, s, b, csp))
    return out


def max_csp_per_pair():
    """One value per pair: the maximum CSP within each pair. 581 points."""
    pairs = _gather_all_csp_pairs()
    max_csp = np.array([float(np.nanmax(csp)) for _, _, _, csp in pairs])
    print(f"  collected {len(pairs)} pairs with max CSP")

    # Use the same trimmed-mean threshold from csp_analysis on the residue-level data
    # (so the threshold matches slide 9). Hard-coded to 0.224 ppm to avoid a slow re-scan.
    threshold = 0.224

    fig, (ax_h, ax_s) = plt.subplots(
        2, 1, figsize=(9, 5.5), gridspec_kw={"height_ratios": [4, 1]}, sharex=True
    )
    upper = float(np.quantile(max_csp, 0.99)) if max_csp.size else 1.0
    bins = np.linspace(0, max(upper, 0.5), 50)
    ax_h.hist(max_csp[max_csp <= upper * 1.5], bins=bins, color="#4C72B0", alpha=0.85)
    ax_h.axvline(threshold, color="red", ls="--", lw=1.4,
                 label=f"residue-level threshold = {threshold} ppm")
    ax_h.set_ylabel("number of pairs")
    ax_h.set_title(
        f"max HN/N CSP per pair · {len(pairs)} bound/unbound pairs · 99% clip"
    )
    ax_h.legend()
    n_above = int(np.sum(max_csp > threshold))
    ax_h.text(
        0.99, 0.92, f"{n_above} / {len(pairs)} pairs above threshold",
        transform=ax_h.transAxes, ha="right", va="top", fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc"},
    )
    # Strip plot below for individual pair visibility
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.4, 0.4, size=max_csp.size)
    ax_s.scatter(max_csp, jitter, s=8, alpha=0.5, color="#4C72B0")
    ax_s.axvline(threshold, color="red", ls="--", lw=1.0)
    ax_s.set_xlim(0, max(upper * 1.5, 0.5))
    ax_s.set_ylim(-1, 1)
    ax_s.set_yticks([])
    ax_s.set_xlabel("max CSP within pair (ppm)")
    fig.tight_layout()
    fig.savefig(FIG / "max_csp_per_pair.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("max_csp_per_pair.png written")


def csp_merged_logy():
    """Overlay per-residue CSPs and max-per-pair CSPs on shared bins, log-y.

    Replaces the slide 10 left histogram by combining both views in one panel.
    """
    from csp_analysis import trimmed_mean_threshold  # noqa: E402

    pairs = _gather_all_csp_pairs()
    per_residue = np.concatenate(
        [csp[~np.isnan(csp)] for _, _, _, csp in pairs]
    )
    per_pair_max = np.array([float(np.nanmax(csp)) for _, _, _, csp in pairs])
    print(f"  per-residue CSPs : {per_residue.size:,}")
    print(f"  max-per-pair CSPs: {per_pair_max.size:,}")

    _, threshold = trimmed_mean_threshold(per_residue.tolist())

    upper_residue = float(np.quantile(per_residue, 0.99))
    upper_max = float(np.quantile(per_pair_max, 0.99))
    x_hi = max(upper_residue, upper_max) * 1.05
    bins = np.linspace(0, x_hi, 60)
    n_clipped_residue = int(np.sum(per_residue > x_hi))
    n_clipped_max = int(np.sum(per_pair_max > x_hi))

    fig, ax = plt.subplots(figsize=(9, 5.0))
    ax.hist(
        np.clip(per_residue, 0, x_hi), bins=bins,
        color="#2a9d8f", alpha=0.65,
        label=f"per-residue CSPs (n = {per_residue.size:,})",
    )
    ax.hist(
        np.clip(per_pair_max, 0, x_hi), bins=bins,
        color="#e76f51", alpha=0.75,
        label=f"max CSP per pair (n = {per_pair_max.size:,})",
    )
    ax.axvline(
        threshold, color="black", ls="--", lw=1.2,
        label=f"residue-level threshold = {threshold:.3f} ppm",
    )
    ax.set_yscale("log")
    ax.set_xlim(0, x_hi)
    ax.set_xlabel("CSP (ppm)")
    ax.set_ylabel("count (log scale)")
    ax.set_title("HN/N CSP — per-residue vs max-per-pair (overlaid, log-y)")
    ax.legend()
    if n_clipped_residue or n_clipped_max:
        ax.text(
            0.99, 0.02,
            f"clipped: {n_clipped_residue} residue · {n_clipped_max} pair > {x_hi:.2f} ppm",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#666666",
        )
    fig.tight_layout()
    fig.savefig(FIG / "csp_merged_logy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("csp_merged_logy.png written")


def valine_structure():
    """Schematic of the valine side chain — the two methyls (CG1, CG2) are
    geminal partners, NMR-equivalent under standard pulse sequences. Used on
    slide 3 to give the audience a quick mental anchor for the methyl-wildcard
    rewrite.
    """
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.2, 3.4)
    ax.axis("off")

    purple = "#361a54"
    accent = "#5a2e8c"
    arrow = "#d62728"
    grey = "#666666"

    # Bonds (drawn first so labels overlay)
    ax.plot([0, 0], [0.35, 1.05], color=purple, lw=1.6)         # Cα — Cβ
    ax.plot([0, -1.0], [1.55, 2.4], color=purple, lw=1.6)        # Cβ — CG1
    ax.plot([0, 1.0], [1.55, 2.4], color=purple, lw=1.6)         # Cβ — CG2
    ax.plot([0, 0], [-0.35, -1.05], color=purple, lw=1.6)        # Cα — backbone
    # implicit Cβ-H stub
    ax.plot([0.35, 0.85], [1.3, 1.0], color=purple, lw=1.0)
    ax.text(0.95, 0.95, "H", ha="left", va="center", fontsize=9, color=grey)

    # Atom labels
    ax.text(0, 0, "Cα", ha="center", va="center", fontsize=15, fontweight="bold", color=purple,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    ax.text(0, 1.3, "Cβ", ha="center", va="center", fontsize=15, fontweight="bold", color=purple,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    ax.text(-1.1, 2.55, "CH₃", ha="center", va="center", fontsize=14, fontweight="bold", color=accent,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    ax.text(-1.1, 2.95, "CG1", ha="center", va="center", fontsize=10, color=accent)
    ax.text(1.1, 2.55, "CH₃", ha="center", va="center", fontsize=14, fontweight="bold", color=accent,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    ax.text(1.1, 2.95, "CG2", ha="center", va="center", fontsize=10, color=accent)

    # Backbone label
    ax.text(0, -1.4, "backbone\n(N · Cα · C=O)", ha="center", va="top", fontsize=9, color=grey)

    # NMR-equivalence arrow between the two methyls
    ax.annotate(
        "", xy=(0.7, 2.55), xytext=(-0.7, 2.55),
        arrowprops=dict(arrowstyle="<->", color=arrow, lw=1.6),
    )
    ax.text(0, 3.15, "NMR-equivalent\n→ CG1 / CG2 indistinguishable",
            ha="center", va="bottom", fontsize=9, color=arrow, fontstyle="italic")

    fig.tight_layout()
    fig.savefig(FIG / "valine.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("valine.png written")


if __name__ == "__main__":
    architecture_diagram()
    workflow_diagram()
    per_tier_deltas()
    lacs_vs_potenci_overlap()
    flip_count_by_tier()
    max_csp_per_pair()
    csp_merged_logy()
    valine_structure()
