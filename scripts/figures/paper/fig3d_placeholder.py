#!/usr/bin/env python3
"""Fig. 3d -- labelled placeholder for the independent-evidence panel.

The cell is filled, in this order of preference (writing plan, Fig. 3 table):
(1) structure agreement by tier: TALOS-N secondary structure from the deposited shifts
    against DSSP on ColabFold models, controlled for shift count k, pLDDT >= 70 and
    n_copies (owner SK; not yet delivered);
(2) PROVISIONAL: DisProt ROC per tier, regenerated with a current DisProt release and
    state-matched;
(3) if neither arrives, Fig. 3 ships with panels a-c.

Run (code worktree root):
    uv run --extra figures python scripts/figures/paper/fig3d_placeholder.py \
        --out-dir <draft>/Figures/paper
Output: <out-dir>/fig3d_placeholder.pdf (88 x 80 mm, the size of cell c).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402


def main() -> None:
    sp.parse_args(sp.cli(__doc__))
    W, H = sp.W_SINGLE, 80.0
    fig = sp.figure(W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(
        mpatches.FancyBboxPatch((4, 3), W - 7, H - 9, boxstyle="round,pad=0,rounding_size=2",
                                facecolor="#F4F4F4", edgecolor="#8C8C8C", lw=0.8, ls=(0, (4, 2)))
    )  # fmt: skip
    ax.text(W / 2, H - 15, "PLACEHOLDER", ha="center", va="center", fontsize=9, fontweight="bold",
            color="#8C8C8C")  # fmt: skip
    ax.text(W / 2, H - 22, "Independent evidence (one slot)", ha="center", va="center", fontsize=7,
            fontweight="bold", color="#4D4D4D")  # fmt: skip
    body = [
        ("1", "Structure agreement by tier (owner SK)",
         "TALOS-N secondary structure from the deposited shifts\nvs DSSP on ColabFold models; controlled for k,\npLDDT ≥ 70 and n_copies"),
        ("2", "PROVISIONAL fallback: DisProt ROC by tier",
         "regenerated with a current DisProt release\nand matched to the sample state"),
        ("3", "If neither arrives", "Fig. 3 ships with panels a–c"),
    ]  # fmt: skip
    y = H - 31
    for num, head, txt in body:
        ax.text(
            10,
            y,
            num,
            ha="left",
            va="top",
            fontsize=6.5,
            fontweight="bold",
            color="#4D4D4D",
        )
        ax.text(
            15,
            y,
            head,
            ha="left",
            va="top",
            fontsize=6.0,
            fontweight="bold",
            color="#4D4D4D",
        )
        ax.text(
            15,
            y - 3.2,
            txt,
            ha="left",
            va="top",
            fontsize=5.5,
            color="#4D4D4D",
            linespacing=1.15,
        )
        y -= 6.0 + 2.6 * (txt.count("\n") + 1)
    sp.fig_label(fig, "d", 0.0, 0.0)
    pdf = sp.save(fig, "fig3d_placeholder")
    print(f"wrote {pdf}", file=sys.stderr)


if __name__ == "__main__":
    main()
