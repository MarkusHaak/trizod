#!/usr/bin/env python3
"""Fig. 2c -- fidelity where the methods are comparable: TriZOD vs CheZOD1325.

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

  Provenance mark: while the potenci-only input is a pre-build run, the legend
  entry carries a superscript 1 and the axes carry the footnote "pre-build run
  (June 2026 code)". Both are drawn only for a pre-build input, detected from
  the file content: it lacks the renamed offset columns (``off_<atom>_sigma``,
  ``total_off_<atom>_ppm``) that 2026-10 code writes. A re-run on 2026-10 code
  therefore drops the mark with no edit here, so no draft provenance text
  reaches the submitted artwork.
* ``both`` -- LACS then POTENCI/AIC, the mode of the released labels. Input:
  the 2026-10 released unfiltered labels
  (``release_bundle/.../scores/unfiltered/scores.json``), i.e. current code.
  The pre-build ``both`` statistic in ``reproduce_summary.json`` (median 0.389)
  came from the then-current scored tier and is superseded by this curve.

Per-entry alignment and MAE use ``trizod.figures.chezod.stats`` (best
sequence-match offset, residues where both scores exist, >= 5 residues), so the
medians reproduce ``reproduce_summary.json`` for the same inputs. The entries
whose potenci-only Z agrees with CheZOD within 0.1 at every compared residue
are marked on the potenci-only curve.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig2c_chezod.py \
        --out-dir <draft>/Figures/paper [--potenci-only PATH]
Outputs: <out-dir>/fig2c.pdf and <out-dir>/source_data/fig2c.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402

from trizod.figures.chezod import load_chezod, stats  # noqa: E402
from trizod.offsets import off_sigma_col, total_off_ppm_col  # noqa: E402

PO = sp.CHEZOD_VER / "trizod_potenci_only.json"  # pre-build (2026-06-22)
BOTH = sp.BUNDLE / "scores" / "unfiltered" / "scores.json"  # 2026-10 release
WITHIN = 0.1  # "within 0.1 Z at every residue"
FLOOR = 1e-3  # errors below this are drawn at the floor (log axis)
FS_SMALL = 5.0
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


def plot(res: dict, within_ids: set, prebuild: bool) -> None:
    """ECDF on a log axis; errors below ``FLOOR`` are drawn at the floor.

    About 180 potenci-only entries agree with CheZOD to rounding (MAE < 0.001), so
    the left tick reads "<=10^-3" and the curve starts from zero. The entries whose
    potenci-only Z lies within 0.1 of CheZOD at every compared residue are drawn
    as a rug under the potenci-only curve. ``prebuild`` adds the provenance mark
    (superscript and footnote) for a pre-build potenci-only input.
    """
    fig = sp.figure(62.0, 74.0)
    ax = fig.add_axes([0.2, 0.15, 0.76, 0.64])
    cols = {"potenci-only": sp.OKABE_ITO["blue"], "both": sp.OKABE_ITO["orange"]}
    labels = {
        "potenci-only": "LACS off (potenci-only)" + ("¹" if prebuild else ""),
        "both": "LACS on (both; released labels)",
    }
    for mode in ("potenci-only", "both"):
        maes = {e: v[0] for e, v in res[mode].items()}
        x, y = ecdf(np.maximum(list(maes.values()), FLOOR))
        x = np.concatenate([[FLOOR], x])
        y = np.concatenate([[0.0], y])
        ax.step(x, y, where="post", color=cols[mode], lw=1.0, label=labels[mode])
        med = float(np.median(list(maes.values())))
        ax.plot([med, med], [0, 0.5], color=cols[mode], lw=0.5, ls=":")
        ax.text(
            med / 1.12 if mode == "potenci-only" else med * 1.08,
            0.53 if mode == "potenci-only" else 0.22,
            f"median\n{med:.2f}",
            color=cols[mode],
            fontsize=FS_SMALL,
            ha="left" if mode == "both" else "right",
            va="bottom",
            linespacing=1.0,
        )
        if mode == "potenci-only":
            xs = np.maximum(
                [v[0] for e, v in res[mode].items() if e in within_ids], FLOOR
            )
            ax.plot(
                xs,
                np.full(len(xs), 0.012),
                ls="none",
                marker="|",
                ms=4.0,
                mew=0.3,
                color="0.15",
                label=f"{len(xs)} entries within {WITHIN} Z at every residue",
            )
    ax.axhline(0.5, color="0.8", lw=0.4, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.8, 10)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(
        [1e-3, 1e-2, 1e-1, 1, 10], ["≤10$^{-3}$", "10$^{-2}$", "0.1", "1", "10"]
    )
    ax.set_xlabel("Per-entry mean absolute error vs CheZOD Z")
    ax.set_ylabel("Cumulative fraction of CheZOD1325 entries")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.01),
        frameon=False,
        handlelength=1.5,
        borderaxespad=0.0,
    )
    if prebuild:
        # Draft-only provenance mark; absent once the potenci-only run is redone
        # on 2026-10 code (see module docstring).
        ax.text(
            0.03,
            0.66,
            "¹pre-build run\n(June 2026 code)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=FS_SMALL,
            color="0.35",
        )
    sp.fig_label(fig, "c", 0.0, 0.0)
    sp.save(fig, "fig2c")


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
        f"({'PRE-BUILD: provenance mark drawn' if prebuild else '2026-10 schema'})"
    )
    chezod = load_chezod(sp.CHEZOD_1325)
    res = {
        "potenci-only": per_entry(chezod, load_scores_lenient(args.potenci_only)),
        "both": per_entry(chezod, load_scores_lenient(BOTH)),
    }
    within = {e for e, v in res["potenci-only"].items() if v[1] <= WITHIN}
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
    with open(sd / "fig2c.csv", "w") as fh:
        fh.write("bmrb_id,mode,mae,max_abs_err,n_residues,within_0p1_all\n")
        for mode, r in res.items():
            for e, (m, mx, n) in sorted(r.items()):
                fh.write(f"{e},{mode},{m:.6f},{mx:.6f},{n},{int(mx <= WITHIN)}\n")
    plot(res, within, prebuild)
    print(f"wrote {sp.out_dir() / 'fig2c.pdf'}")


if __name__ == "__main__":
    main()
