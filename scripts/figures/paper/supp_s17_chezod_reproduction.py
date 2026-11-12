#!/usr/bin/env python3
"""Supplementary Fig. S17 -- CheZOD reproduction: outcome categories and discrepancy sources.

Panels
  a  Per-entry Pearson r against mean absolute error (MAE) between TriZOD and the
     published CheZOD1325 Z-scores without LACS (CheZOD method; ``potenci-only``), with
     the four outcome regions of ``trizod.figures.chezod.classify``:
       agree         r >= 0.9 and MAE <= 1.0
       offset_shift  r >= 0.9 and MAE  > 1.0
       low_variance  r <  0.9 and MAE <= 0.8
       genuine       otherwise
  b  Outcome counts without LACS (CheZOD method) vs with LACS (released labels;
     ``both``: LACS + POTENCI/AIC).
  c  The genuine discrepancies (without LACS): mean Z difference
     (TriZOD - CheZOD) against MAE; filled when TriZOD's residual POTENCI/AIC offset
     is >= 2 sigma. The three entries CheZOD's own code scores without carbonyl
     shifts (``skipCO``) are labelled.
  d  Per-residue Z of BMRB 15506, the largest discrepancy: a near-constant upward
     shift. CheZOD's code omits this entry's carbonyl shifts (``skipCO``); TriZOD
     scores them. (All three skipCO entries are CheZOD117 test chains.)

Inputs (read-only): CheZOD1325 published scores (``data/external/chezod/protein_nmr_1325``),
``data/interim/chezod_verification/trizod_potenci_only.json`` (a June 2026 run, before the
release code), and the release labels of the selected build (bundle
``scores/unfiltered/scores.json``) for the released mode.
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
    d_bo = per_entry(chezod, both, f"both ({sp.RELEASE} release)")
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
    # Layout in mm, derived from the 160 mm text width (W_DOUBLE; placed at 1:1). The
    # outcome categories are regions of panel a and rows of panel b, so they are labelled
    # in place and need no colour. One key at the top names the two modes, in the paper's
    # wording and with Fig. 2c's encoding: orange = TriZOD without LACS (CheZOD method;
    # a, b), black = with LACS (released labels; b). In c and d, ink = TriZOD and orange =
    # the published CheZOD Z-scores.
    fig_w, fig_h = sp.W_DOUBLE, 58.0
    ax_top, ax_h = 10.0, 40.5
    boxes = {  # left, width (mm)
        "a": (11.0, 30.0),
        "b": (58.0, 22.0),
        "c": (93.5, 24.5),
        "d": (128.0, sp.W_DOUBLE - 3.0 - 128.0),
    }
    fig = sp.figure(fig_w, fig_h)

    def _ax(key):
        left, w = boxes[key]
        return fig.add_axes(
            [left / fig_w, 1 - (ax_top + ax_h) / fig_h, w / fig_w, ax_h / fig_h]
        )

    axa, axb, axc, axd = (_ax(k) for k in "abcd")
    ink, grey = "#333333", "#6E6E6E"
    modes = {
        # same encoding as Fig. 2c: orange = without LACS (CheZOD method), black = released
        "potenci-only": ("Without LACS (CheZOD method)", sp.CAT["chezod"]),
        "both": ("With LACS (released labels)", ink),
    }

    # a: r vs MAE scatter, without LACS
    axa.scatter(
        d_po.mae, d_po.pearson, s=3, lw=0, color=sp.CAT["chezod"], alpha=0.7, zorder=2
    )
    axa.set_xscale("log")
    axa.set_xlim(0.008, 8)
    axa.set_ylim(-0.6, 1.16)
    kw = {"color": "#7a7a7a", "lw": 0.5, "ls": "--", "zorder": 1}
    axa.plot([0.008, 8], [0.9, 0.9], **kw)
    axa.plot([1.0, 1.0], [0.9, 1.16], **kw)
    axa.plot([0.8, 0.8], [-0.6, 0.9], **kw)
    lab = {"fontsize": sp.FS_SMALL, "color": grey}
    axa.text(0.011, 1.09, CAT_LABEL["agree"], ha="left", va="center", **lab)
    axa.text(7.5, 1.09, CAT_LABEL["offset_shift"], ha="right", va="center", **lab)
    axa.text(0.011, -0.5, CAT_LABEL["low_variance"], ha="left", va="center", **lab)
    axa.text(7.5, -0.5, CAT_LABEL["genuine"], ha="right", va="center", **lab)
    axa.set_xticks([0.01, 0.1, 1])
    axa.set_yticks([-0.5, 0, 0.5, 1.0], ["\N{MINUS SIGN}0.5", "0", "0.5", "1"])
    axa.set_xlabel("Mean absolute error per entry")
    axa.set_ylabel("Pearson $r$ per entry")

    # b: category counts, both modes
    y = np.arange(len(CATS))
    h = 0.38
    for j, (d, mode) in enumerate(((d_po, "potenci-only"), (d_bo, "both"))):
        lab_, col = modes[mode]
        cnt = [int((d.category == c).sum()) for c in CATS]
        bars = axb.barh(y + (j - 0.5) * h, cnt, height=h, color=col, lw=0, label=lab_)
        for bb, n in zip(bars, cnt):
            axb.text(
                bb.get_width() * 1.15,
                bb.get_y() + bb.get_height() / 2,
                f"{n:,}",
                va="center",
                ha="left",
                fontsize=sp.FS_SMALL,
            )
    axb.set_xscale("log")
    axb.set_xlim(5, 8000)
    axb.set_xticks([10, 100, 1000])
    axb.set_yticks(y)
    axb.set_yticklabels([CAT_LABEL[c] for c in CATS])
    axb.tick_params(axis="y", length=0)
    axb.invert_yaxis()
    axb.set_xlabel("CheZOD1325 entries")
    # the key for both modes, above a and b
    fig.legend(
        *axb.get_legend_handles_labels(),
        loc="center left",
        bbox_to_anchor=(boxes["a"][0] / fig_w, 1 - 5.2 / fig_h),
        ncol=2,
        handlelength=1.2,
        columnspacing=1.6,
        borderaxespad=0,
        borderpad=0,
        fontsize=sp.FS_TICK,
    )

    # c: genuine discrepancies (without LACS)
    drv = gen.max_potenci_off_sigma >= 2.0
    axc.scatter(gen.mae[~drv], gen.mean_shift[~drv], s=10, facecolor="white",
                edgecolor=ink, lw=0.7, label="Offset < 2σ")  # fmt: skip
    axc.scatter(gen.mae[drv], gen.mean_shift[drv], s=10, color=ink, lw=0,
                label="Offset ≥ 2σ")  # fmt: skip
    axc.axhline(0, color="#7a7a7a", lw=0.5, ls=":")
    for eid in SKIP_CO:
        r = gen[gen.bmrb_id == eid]
        if len(r):
            axc.annotate(
                eid,
                (r.mae.iloc[0], r.mean_shift.iloc[0]),
                xytext=(-3.5, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=sp.FS_SMALL,
            )
    axc.set_xlim(0, float(gen.mae.max()) * 1.12)
    axc.set_xticks([0, 2, 4])
    axc.set_xlabel("Mean absolute error per entry")
    axc.set_ylabel("Mean Z difference\n(TriZOD \N{MINUS SIGN} CheZOD)", linespacing=1.1)
    axc.legend(loc="lower right", borderaxespad=0.1, handletextpad=0.2)
    axc.set_title("Genuine discrepancies", fontsize=sp.FS_TICK, pad=2)

    # d: 15506 profile (TriZOD without LACS)
    pos, zc, zt = profile(chezod, po, "15506")
    axd.plot(pos, zc, color=sp.CAT["chezod"], lw=0.9, label="CheZOD")
    axd.plot(pos, zt, color=ink, lw=0.9, label="TriZOD")
    axd.set_xlabel("Residue (BMRB 15506)")
    axd.set_ylabel("Z-score")
    axd.legend(
        loc="lower left",
        borderaxespad=0.1,
        ncol=2,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    shift = float(np.mean(zt - zc))
    axd.text(
        0.98,
        0.98,
        f"Mean difference {shift:+.2f}".replace("-", "\N{MINUS SIGN}"),
        transform=axd.transAxes,
        ha="right",
        va="top",
        fontsize=sp.FS_SMALL,
    )
    lo = min(np.nanmin(zc), np.nanmin(zt))
    axd.set_ylim(lo - 4, max(np.nanmax(zc), np.nanmax(zt)) + 2.5)
    axd.yaxis.set_major_locator(sp.mticker.MultipleLocator(5))  # integer ticks
    axd.set_title("Largest discrepancy", fontsize=sp.FS_TICK, pad=2)

    # panel letters, one baseline, left of each panel's y label
    for key, dx in (("a", 11.0), ("b", 16.5), ("c", 13.5), ("d", 10.0)):
        sp.fig_label(fig, key, boxes[key][0] - dx, 0.0)
    out = sp.save(fig, "S17")

    dd = sp.out_dir("suppfigs_b_data")
    dd.mkdir(exist_ok=True)
    pd.concat([d_po, d_bo]).to_csv(dd / "S17_per_entry.csv", index=False)
    (dd / "S17_summary.json").write_text(json.dumps(summ, indent=2))
    print(out)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
