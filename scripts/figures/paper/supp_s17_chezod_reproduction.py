#!/usr/bin/env python3
"""Supplementary Fig. S17 -- CheZOD reproduction: outcome categories and discrepancy sources.

Panels
  a  Per-entry Pearson r against mean absolute error (MAE) between TriZOD and the
     published CheZOD1325 Z-scores, re-referencing disabled (``potenci-only``), with
     the four outcome regions of ``trizod.figures.chezod.classify``:
       agree         r >= 0.9 and MAE <= 1.0
       offset_shift  r >= 0.9 and MAE  > 1.0
       low_variance  r <  0.9 and MAE <= 0.8
       genuine       otherwise
  b  Outcome counts with re-referencing disabled vs in the released mode
     (``both``: LACS + POTENCI/AIC).
  c  The genuine discrepancies (re-referencing disabled): mean Z difference
     (TriZOD - CheZOD) against MAE; filled when TriZOD's residual POTENCI/AIC offset
     is >= 2 sigma. The three entries CheZOD's own code scores without carbonyl
     shifts (``skipCO``) are labelled.
  d  Per-residue Z of BMRB 15506, the largest discrepancy: a near-constant upward
     shift. CheZOD's code omits this entry's carbonyl shifts (``skipCO``); TriZOD
     scores them. (All three skipCO entries are CheZOD117 test chains.)

Inputs (read-only): CheZOD1325 published scores (``data/external/chezod/protein_nmr_1325``),
``data/interim/chezod_verification/trizod_potenci_only.json`` (pre-2026-10 code), and the
2026-10 release labels (bundle ``scores/unfiltered/scores.json``) for the released mode.
The per-entry statistics use the same helpers as ``scripts/validation/reproduce_chezod.py``.

Run: uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s17_chezod_reproduction.py \
         --out-dir <draft>/Figures/paper
Outputs: <out-dir>/S17.pdf and <out-dir>/suppfigs_b_data/S17_per_entry.csv / S17_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402

sys.path.insert(0, str(sp.REPO))
from trizod.figures.chezod import (  # noqa: E402
    ATOMS,
    best,
    classify,
    load_chezod,
    stats,
)
from trizod.offsets import legacy_off_col, off_sigma_col  # noqa: E402

PO = sp.CHEZOD_VER / "trizod_potenci_only.json"
BOTH = sp.BUNDLE / "scores" / "unfiltered" / "scores.json"
SKIP_CO = ["15506", "15274", "15719"]
CATS = ["agree", "low_variance", "offset_shift", "genuine"]
CAT_LABEL = {
    "agree": "agree",
    "low_variance": "low variance",
    "offset_shift": "offset shift",
    "genuine": "genuine",
}
CAT_COLOR = {
    "agree": sp.OKABE_ITO["skyblue"],
    "low_variance": sp.OKABE_ITO["green"],
    "offset_shift": sp.OKABE_ITO["purple"],
    "genuine": sp.OKABE_ITO["vermillion"],
}


def load_trizod(path):
    """``trizod.figures.chezod.load_trizod`` that also reads the pre-rename offset names.

    The stored potenci-only run (June 2026) predates the ``off_<atom>_sigma`` rename and
    spells the POTENCI/AIC offset ``off_<atom>`` (same quantity, sigma units; see
    ``trizod.offsets.legacy_off_col``). A record with neither spelling raises.
    """
    by = {}
    for line in Path(path).open():
        if not line.strip():
            continue
        r = json.loads(line)
        vals = []
        for a in ATOMS:
            if off_sigma_col(a) in r:
                v = r[off_sigma_col(a)]
            elif legacy_off_col(a) in r:
                v = r[legacy_off_col(a)]
            else:
                raise KeyError(f"no POTENCI offset column for {a} in {path}")
            if v is not None:
                vals.append(abs(v))
        off = max(vals) if vals else 0.0
        by.setdefault(str(r["entryID"]), []).append(
            {"seq": r["seq"] or "", "z": r["zscores"] or [], "off": off}
        )
    return by


def per_entry(chezod, trizod, mode):
    rows = []
    for eid, cz in chezod.items():
        recs = trizod.get(eid)
        if not recs:
            continue
        s = stats(cz, recs)
        if s is None:
            continue
        rows.append(
            {
                "bmrb_id": eid,
                "mode": mode,
                "pearson": s["pearson"],
                "mae": s["mae"],
                "mean_shift": s["mean_shift"],
                "n": s["n"],
                "seq_match": s["seq_match"],
                "max_potenci_off_sigma": s["rec"]["off"],
                "category": classify(s["pearson"], s["mae"]),
            }
        )
    return pd.DataFrame(rows)


def profile(chezod, trizod, eid):
    cz = chezod[eid]
    rec, off = best(cz, trizod[eid])
    pos, a, b = [], [], []
    for k, cv in enumerate(cz["z"]):
        m = k + off
        if 0 <= m < len(rec["z"]) and cv is not None and rec["z"][m] is not None:
            pos.append(k + 1)
            a.append(cv)
            b.append(rec["z"][m])
    return np.array(pos), np.array(a), np.array(b)


def main():
    sp.parse_args(sp.cli(__doc__))
    chezod = load_chezod(sp.CHEZOD_1325)
    po = load_trizod(PO)
    both = load_trizod(BOTH)
    d_po = per_entry(chezod, po, "potenci-only")
    d_bo = per_entry(chezod, both, "both (2026-10 release)")
    summ = {}
    for name, d in (("potenci_only", d_po), ("both_2026_10", d_bo)):
        summ[name] = {
            "n": int(len(d)),
            "categories": {c: int((d.category == c).sum()) for c in CATS},
            "mae_median": round(float(d.mae.median()), 3),
            "mae_q3": round(float(d.mae.quantile(0.75)), 3),
            "mae_p95": round(float(np.percentile(d.mae, 95)), 3),
            "consistent_frac": round(float((d.category != "genuine").mean()), 3),
        }
    gen = d_po[d_po.category == "genuine"].sort_values("mae", ascending=False)
    summ["genuine_potenci_only"] = int(len(gen))
    summ["genuine_offset_driven"] = int(
        ((gen.mean_shift.abs() > 0.8) | (gen.max_potenci_off_sigma >= 2.0)).sum()
    )
    summ["genuine_top5"] = (
        gen.head(5)[["bmrb_id", "pearson", "mae", "mean_shift"]]
        .round(3)
        .to_dict("records")
    )

    # ---- figure
    fig = sp.figure(sp.W_DOUBLE, 58)
    gs = fig.add_gridspec(
        1,
        4,
        left=0.055,
        right=0.99,
        bottom=0.17,
        top=0.90,
        wspace=0.45,
        width_ratios=[1.15, 0.9, 1.0, 1.15],
    )
    axa, axb, axc, axd = (fig.add_subplot(gs[0, i]) for i in range(4))

    # a: r vs MAE scatter, potenci-only
    for c in CATS:
        s = d_po[d_po.category == c]
        axa.scatter(
            s.mae,
            s.pearson,
            s=3,
            lw=0,
            color=CAT_COLOR[c],
            alpha=0.85,
            label=f"{CAT_LABEL[c]} ({len(s):,})",
            zorder=3 if c == "genuine" else 2,
        )
    axa.set_xscale("log")
    axa.set_xlim(0.008, 8)
    axa.set_ylim(-0.6, 1.02)
    kw = {"color": "#7a7a7a", "lw": 0.5, "ls": "--", "zorder": 1}
    axa.plot([0.008, 8], [0.9, 0.9], **kw)
    axa.plot([1.0, 1.0], [0.9, 1.02], **kw)
    axa.plot([0.8, 0.8], [-0.6, 0.9], **kw)
    axa.set_xlabel("Per-entry mean absolute error (Z units)")
    axa.set_ylabel("Per-entry Pearson r")
    axa.set_title("Re-referencing disabled", fontsize=6)
    axa.legend(loc="lower left", markerscale=2, handletextpad=0.2, borderaxespad=0.2)

    # b: category counts, both modes
    y = np.arange(len(CATS))
    h = 0.38
    for j, (d, lab, hatch) in enumerate(
        ((d_po, "disabled", None), (d_bo, "released (LACS + POTENCI/AIC)", "/////"))
    ):
        cnt = [int((d.category == c).sum()) for c in CATS]
        bars = axb.barh(
            y + (j - 0.5) * h,
            cnt,
            height=h,
            color=[CAT_COLOR[c] for c in CATS],
            edgecolor="#4D4D4D",
            lw=0.4,
            hatch=hatch,
            label=lab,
        )
        for bb, n in zip(bars, cnt):
            axb.text(
                bb.get_width() * 1.15,
                bb.get_y() + bb.get_height() / 2,
                f"{n:,}",
                va="center",
                ha="left",
                fontsize=5,
            )
    axb.set_xscale("log")
    axb.set_xlim(5, 6000)
    axb.set_yticks(y)
    axb.set_yticklabels([CAT_LABEL[c] for c in CATS])
    axb.invert_yaxis()
    axb.set_xlabel("CheZOD1325 entries")
    from matplotlib.patches import Patch

    axb.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor="#4D4D4D",
                lw=0.4,
                label="re-referencing disabled",
            ),
            Patch(
                facecolor="white",
                edgecolor="#4D4D4D",
                lw=0.4,
                hatch="/////",
                label="released mode",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.45, 1.0),
        borderaxespad=0.1,
        handlelength=1.4,
        ncol=1,
    )

    # c: genuine discrepancies
    drv = gen.max_potenci_off_sigma >= 2.0
    axc.scatter(
        gen.mae[~drv],
        gen.mean_shift[~drv],
        s=9,
        facecolor="white",
        edgecolor=CAT_COLOR["genuine"],
        lw=0.6,
        label="residual offset < 2σ",
    )
    axc.scatter(
        gen.mae[drv],
        gen.mean_shift[drv],
        s=9,
        color=CAT_COLOR["genuine"],
        lw=0,
        label="residual offset ≥ 2σ",
    )
    axc.axhline(0, color="#7a7a7a", lw=0.5, ls=":")
    for eid in SKIP_CO:
        r = gen[gen.bmrb_id == eid]
        if len(r):
            axc.annotate(
                eid,
                (r.mae.iloc[0], r.mean_shift.iloc[0]),
                xytext=(-3, 3),
                textcoords="offset points",
                ha="right",
                fontsize=5,
            )
    axc.set_xlabel("Per-entry mean absolute error (Z units)")
    axc.set_ylabel("Mean Z difference, TriZOD − CheZOD")
    axc.legend(loc="lower right", borderaxespad=0.1, handletextpad=0.2)
    axc.set_title(f"Genuine discrepancies ({len(gen)})", fontsize=6)

    # d: 15506 profile
    pos, zc, zt = profile(chezod, po, "15506")
    axd.plot(
        pos, zc, color=sp.OKABE_ITO["orange"], lw=0.8, label="CheZOD (carbonyl omitted)"
    )
    axd.plot(
        pos,
        zt,
        color=sp.CAT["neutral"],
        lw=0.8,
        label="TriZOD, re-referencing disabled",
    )
    axd.axhline(8, color=sp.CAT["light"], lw=0.5, ls=":")
    axd.set_xlabel("Residue (BMRB 15506)")
    axd.set_ylabel("Z-score")
    axd.legend(loc="lower center", borderaxespad=0.1, ncol=1, handlelength=1.4)
    shift = float(np.mean(zt - zc))
    axd.text(
        0.98,
        0.98,
        f"mean difference {shift:+.2f}",
        transform=axd.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
    )
    lo = min(np.nanmin(zc), np.nanmin(zt))
    axd.set_ylim(lo - 4, max(np.nanmax(zc), np.nanmax(zt)) + 2)

    for ax, l in zip((axa, axb, axc, axd), "abcd"):
        sp.panel_label(ax, l, x=-0.24 if l != "b" else -0.42, y=1.03)
    out = sp.save(fig, "S17")

    dd = sp.out_dir("suppfigs_b_data")
    dd.mkdir(exist_ok=True)
    pd.concat([d_po, d_bo]).to_csv(dd / "S17_per_entry.csv", index=False)
    (dd / "S17_summary.json").write_text(json.dumps(summ, indent=2))
    print(out)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
