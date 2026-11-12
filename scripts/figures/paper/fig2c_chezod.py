#!/usr/bin/env python3
"""Fig. 3a (Fig. 2c on 11 Nov 2026) -- run like CheZOD, TriZOD reproduces its scores.

ECDF of the per-entry mean absolute error (MAE) between TriZOD Z-scores and the
published CheZOD1325 Z-scores, for two re-referencing modes:

* ``potenci-only`` -- re-referencing by LACS disabled; POTENCI random coil +
  AIC offset correction + CheZOD Z, i.e. the CheZOD-equivalent method.
  Input: ``data/interim/chezod_verification/trizod_potenci_only.json``.
  **PRE-BUILD**: this run dates from 2026-06-22, before the 2026-10 build
  (develop ``5b8147a``). It must be re-run on 2026-10 code
  (``scripts/validation/reproduce_chezod.py`` docstring gives the command)
  before submission; the macros ``\\maeCheZODPotenciOnly*`` carry the same
  pre-build value. The file also predates the offset-column rename, so it is
  read with a lenient loader (only ``seq`` and ``zscores`` are needed here).
  ``--potenci-only PATH`` reads a re-run from elsewhere.

  The artwork carries no provenance mark (11 Nov 2026 redesign): the draft's
  coordinator states the run's vintage outside the figure. ``is_prebuild`` only
  reports it on stdout.
* ``both`` -- LACS then POTENCI/AIC, the mode of the released labels. Input:
  the 2026-10 released unfiltered labels
  (``release_bundle/.../scores/unfiltered/scores.json``), i.e. current code.
  The pre-build ``both`` statistic in ``reproduce_summary.json`` (median 0.389)
  came from the then-current scored tier and is superseded by this curve.

Per-entry alignment and MAE use ``trizod.figures.chezod.stats`` (best
sequence-match offset, residues where both scores exist, >= 5 residues), so the
medians reproduce ``reproduce_summary.json`` for the same inputs. The entries
whose potenci-only Z agrees with CheZOD within 0.1 at every compared residue are
counted on stdout and flagged in the Source Data (no longer drawn as a rug).
Colours follow the paper's fixed meanings (style_paper.ENTITY): grey dashed = the
CheZOD method (without LACS; the CheZOD1325 entity), black = the released labels
(with LACS). The two curves are labelled directly (no legend): "Run like CheZOD
(without LACS)" and "As released (with LACS)", the wording of the Fig. 3 caption and
of the R3 heading. The median of each curve is marked where it crosses 0.5 (open grey
circle, filled black circle) and printed beside the mark with two decimals, the
precision of ``\\maeCheZODPotenciOnlyShort`` and ``\\maeCheZODBothShort``.

Story-blueprint redesign (12 Nov 2026): 52 x 52 mm (was 58 mm), direct labels instead
of the legend, medians marked, x axis ending at 6 (the largest per-entry error is 4.7),
so the last tick is 1 and the decade ticks keep apart at 7 pt.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig2c_chezod.py \\
        --out-dir <draft>/Figures/paper [--potenci-only PATH]
Outputs: <out-dir>/fig3a_chezod.pdf (52 x 52 mm, placed at 1:1 beside Fig. 3b; the row
layout is imported from ``fig2a_kpaired.py``) and <out-dir>/source_data/fig3a_chezod.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402
from fig2a_kpaired import (  # noqa: E402  (row layout)
    A_W,
    AX_H,
    AX_TOP,
    FIG_H,
    XLABEL_Y_MM,
)

from trizod.figures.chezod import load_chezod, stats  # noqa: E402
from trizod.offsets import off_sigma_col, total_off_ppm_col  # noqa: E402

PO = sp.CHEZOD_VER / "trizod_potenci_only.json"  # pre-build (2026-06-22)
BOTH = sp.BUNDLE / "scores" / "unfiltered" / "scores.json"  # 2026-10 release
WITHIN = 0.1  # "within 0.1 Z at every residue"
FLOOR = 1e-3  # errors below this are drawn at the floor (log axis)
X_MAX = 6.0  # right end of the x axis (largest per-entry error: 4.7)
OUT_NAME = "fig3a_chezod"  # Fig. 3a (fig2c_chezod.pdf on 11 Nov, fig2c.pdf before)
# Offset columns as named by 2026-10 code; a pre-build scores.json has only
# ``off_<atom>``/``lacs_off_<atom>``.
CURRENT_SCHEMA_COLS = (off_sigma_col("CA"), total_off_ppm_col("CA"))


def is_prebuild(path: Path) -> bool:
    """True if ``path`` (TriZOD scores.json, JSONL) predates the offset-column rename.

    The first non-empty record decides: 2026-10 code writes every column in
    ``CURRENT_SCHEMA_COLS``; the June 2026 run writes none of them.
    """
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                return not all(c in rec for c in CURRENT_SCHEMA_COLS)
    raise ValueError(f"{path} has no records")


def load_scores_lenient(path: Path) -> dict:
    """``{entry_id: [{"seq", "z"}]}`` from a TriZOD scores.json (JSONL).

    Unlike ``trizod.figures.chezod.load_trizod`` this does not read the offset
    columns, so the pre-rename potenci-only file can be used.
    """
    by: dict = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            by.setdefault(str(r["entryID"]), []).append(
                {"seq": r["seq"] or "", "z": r["zscores"] or [], "off": 0.0}
            )
    return by


def per_entry(chezod: dict, trizod: dict) -> dict:
    """``{entry_id: (mae, max_abs_err, n)}`` for entries with >= 5 compared residues."""
    from trizod.figures.chezod import best

    out = {}
    for eid, cz in chezod.items():
        recs = trizod.get(eid)
        if not recs:
            continue
        s = stats(cz, recs)
        if s is None:
            continue
        rec, off = best(cz, recs)
        d = [
            abs(cv - rec["z"][k + off])
            for k, cv in enumerate(cz["z"])
            if 0 <= k + off < len(rec["z"])
            and cv is not None
            and rec["z"][k + off] is not None
        ]
        out[eid] = (s["mae"], float(max(d)), s["n"])
    return out


def ecdf(v):
    x = np.sort(np.asarray(v, float))
    return x, np.arange(1, len(x) + 1) / len(x)


FIG_W = A_W  # 52 mm; with Fig. 3b and the row gap, the 160 mm text width
LEFT, RIGHT = 12.0, 2.0  # axes left edge (as Fig. 3c) and right margin, mm
YLABEL_X_MM = 4.6  # right edge of the y title, mm from the left edge (as Fig. 3c)
# axes (left, top, width, height) in mm; top, height and x-label height as in Fig. 3b
BOX = (LEFT, AX_TOP, FIG_W - LEFT - RIGHT, AX_H)
MODES = {
    # colour by meaning: grey dashed = the CheZOD method, black = the released labels
    "potenci-only": {
        **sp.line_kw("chezod1325"),
        "entity": "chezod1325",
        "label": "Run like CheZOD\n(without LACS)",
    },
    "both": {
        **sp.line_kw("release"),
        "entity": "release",
        "label": "As released\n(with LACS)",
    },
}


def plot(res: dict) -> None:
    """ECDF of the per-entry MAE on a log axis; errors below ``FLOOR`` sit at the floor.

    About 180 potenci-only entries agree with CheZOD to rounding (MAE < 0.001), so the
    left tick reads "<=0.001" and the curve starts from zero. The light line at 0.5 reads
    off the medians, which are marked on it. The run-like-CheZOD curve is named in the
    empty upper left, above its own curve; the as-released curve in the empty lower
    right, beside its own curve.
    """
    fig = sp.figure(FIG_W, FIG_H)
    left, top, w, h = BOX
    ax = fig.add_axes([left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])
    ax.axhline(0.5, color=sp.CAT["light"], lw=0.5, zorder=0)
    med = {}
    for mode, st in MODES.items():
        maes = [v[0] for v in res[mode].values()]
        med[mode] = float(np.median(maes))
        x, y = ecdf(np.maximum(maes, FLOOR))
        x = np.concatenate([[FLOOR], x])
        y = np.concatenate([[0.0], y])
        ax.step(x, y, where="post", color=st["color"], ls=st["ls"], lw=1.2, zorder=2)
    # medians: the entity's marker where the curve crosses 0.5, value beside it
    for mode, st in MODES.items():
        ax.plot(med[mode], 0.5, zorder=4, **sp.marker_kw(st["entity"], ms=4.2, mew=1.0))
    po, bo = med["potenci-only"], med["both"]
    sp.direct_label(ax, po, 0.5, f"{po:.2f}", dx_mm=-1.6, dy_mm=1.6, ha="right")
    sp.direct_label(ax, bo, 0.5, f"{bo:.2f}", dx_mm=1.6, dy_mm=-1.6, ha="left")
    # direct labels for the two curves
    # run like CheZOD: above the left part of its own curve (which stays below 0.3 there)
    # and above its median value
    sp.direct_label(
        ax,
        FLOOR,
        0.76,
        MODES["potenci-only"]["label"],
        dx_mm=0.6,
        ha="left",
        linespacing=1.1,
    )
    # as released: right of its own curve's foot (the curve reaches 0.3 at an error of
    # 0.29, so the label starts at 0.30 plus 0.8 mm)
    sp.direct_label(
        ax, 0.30, 0.2, MODES["both"]["label"], dx_mm=0.8, ha="left", linespacing=1.1
    )
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.8, X_MAX)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1], ["≤0.001", "0.01", "0.1", "1"])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    # x title at the same height as Fig. 3b's, so the two panels align in the row
    fig.text(
        (left + w / 2) / FIG_W,
        XLABEL_Y_MM / FIG_H,
        # |dZ| against the published CheZOD1325 Z-score, not an "error": CheZOD is the
        # reference here, not the truth; the axis is logarithmic (figure review item 9)
        "Mean |ΔZ| per entry (log scale)",
        ha="center",
        va="bottom",
        fontsize=sp.FS_BODY,
    )
    ax.set_ylabel("Cumulative fraction of entries")
    ax.yaxis.set_label_coords((YLABEL_X_MM - left) / w, 0.5)
    sp.fig_label(fig, "a", 0.0, 0.0)
    sp.save(fig, OUT_NAME)


def main() -> None:
    ap = sp.cli(__doc__)
    ap.add_argument(
        "--potenci-only",
        type=Path,
        default=PO,
        dest="potenci_only",
        help="potenci-only TriZOD scores.json (default: %(default)s)",
    )
    args = sp.parse_args(ap)
    prebuild = is_prebuild(args.potenci_only)
    print(
        f"potenci-only input: {args.potenci_only} "
        f"({'PRE-BUILD run (June 2026 code)' if prebuild else 'current schema'})"
    )
    chezod = load_chezod(sp.CHEZOD_1325)
    res = {
        "potenci-only": per_entry(chezod, load_scores_lenient(args.potenci_only)),
        "both": per_entry(chezod, load_scores_lenient(BOTH)),
    }
    print(f"CheZOD1325 entries with scores: {len(chezod):,}")
    for mode, r in res.items():
        m = np.array([v[0] for v in r.values()])
        mx = np.array([v[1] for v in r.values()])
        print(
            f"{mode:13s} n {len(m):,}  median {np.median(m):.3f}  Q3 "
            f"{np.percentile(m, 75):.3f}  p95 {np.percentile(m, 95):.3f}  "
            f"within {WITHIN} at every residue {(mx <= WITHIN).sum()}  "
            f"within 0.01 {(mx <= 0.01).sum()}  MAE<0.1 {(m < 0.1).sum()}"
        )
    sd = sp.out_dir("source_data")
    sd.mkdir(parents=True, exist_ok=True)
    with open(sd / f"{OUT_NAME}.csv", "w") as fh:
        fh.write("bmrb_id,mode,mae,max_abs_err,n_residues,within_0p1_all\n")
        for mode, r in res.items():
            for e, (m, mx, n) in sorted(r.items()):
                fh.write(f"{e},{mode},{m:.6f},{mx:.6f},{n},{int(mx <= WITHIN)}\n")
    plot(res)
    print(f"wrote {sp.out_dir() / (OUT_NAME + '.pdf')}")


if __name__ == "__main__":
    main()
