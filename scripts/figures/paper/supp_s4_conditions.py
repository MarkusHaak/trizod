#!/usr/bin/env python3
"""Supplementary Fig. S4 -- ionic strength, pressure and physical state per tier.

Every count is a number of chains in the cumulative tier (unfiltered ⊃ tolerant ⊃
moderate ⊃ strict); tier membership comes from the bundle's per-tier ``scores.json``.

Trimmed 12 Nov 2026 (story blueprint 6.3 and 10.3): the pH and temperature histograms
and the annotation shares (old panels a, b and e) are now main Fig. 2a, b, drawn by
``fig2_composition.py``, which imports ``load``, ``RANGES`` and ``range_bars`` from this
module (keep their names and signatures). ``make_numbers.py`` imports ``load`` and
``HIGH_P``. S4 keeps the three panels that Fig. 2 does not show:

Panels
  a    Ionic strength as DEPOSITED for the chain's own shift table. The Parquet column
       ``ionic_strength`` comes from the unfiltered run, which (like tolerant and
       moderate) substitutes 0.1 M when no value is deposited; the strict run does not,
       so a chain without a deposited value cannot reach strict. The histogram therefore
       shows only deposited values; the chains without one are counted in Fig. 2b.
       "Deposited" is the paper's one definition, ``paper_raw_scan.chain_deposited``: the
       chain's own condition list (``_Assigned_chem_shift_list.Sample_condition_list_ID``)
       read with the pipeline's parser, so a chain counts as not deposited exactly when
       the release holds a default. The accepted ranges (0-7, 0-5 and 0-3 M) lie beyond
       the axis; values above 1 M fall in the edge bin.
  b    Sample pressure (bar), which the release exports (``pressure_bar``) but no tier
       filters on. Read from the same raw sample-condition list, units normalised to
       bar. Log-log histogram; the region >= 400 bar is shaded.
  c    Deposited ``_Entity_assembly.Physical_state`` (``physical_state``), counts on a
       log axis; values outside the six common ones are pooled as "other".

The summary JSON keeps every count of the untrimmed figure (pH and temperature not
deposited or out of axis, annotation counts), so earlier numbers stay traceable.

Run:  uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s4_conditions.py \
          --out-dir <draft>/Figures/paper
Outputs: <out-dir>/S4.pdf and <out-dir>/suppfigs_a_data/S4_chains.csv / S4_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import paper_raw_scan as rs  # noqa: E402
import style_paper as sp  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TIERS = sp.TIERS
DATA_SUBDIR = "suppfigs_a_data"  # under --out-dir

# accepted ranges per tier (trizod/trizod.py filter_defaults; numbers.tex \thr* macros)
RANGES = {
    "ph": {"tolerant": (2, 12), "moderate": (4, 10), "strict": (6, 8)},
    "temperature": {
        "tolerant": (263, 333),
        "moderate": (273, 323),
        "strict": (273, 313),
    },
    "ionic_strength": {"tolerant": (0, 7), "moderate": (0, 5), "strict": (0, 3)},
}
DEFAULTS = {"ph": 7.0, "temperature": 298.0, "ionic_strength": 0.1}
RAW_KEY = {"ph": "ph", "temperature": "temp", "ionic_strength": "ionic"}
HIGH_P = 400.0  # bar; the writing plan's "high pressure" cut (Appendix A12)

PHYS_MAIN = [
    "native",
    "intrinsically disordered",
    "denatured",
    "unfolded",
    "partially disordered",
    "molten globule",
]

# (key, row label); the release columns behind the rows (is_bound, has_metal,
# has_nucleic, membrane_mimetic, cosolvent_evidence, n_copies, sample_state_evidence)
# are named in the caption, not in the artwork
ANNOT = [
    ("is_bound", "Bound partner"),
    ("has_metal", "Metal"),
    ("has_nucleic", "Nucleic acid"),
    ("membrane", "Membrane mimetic"),
    ("cosolvent_evidence", "Perturbing cosolvent"),
    ("multi_copy", "Several copies of the chain"),
    ("homo", "Several copies, no partner"),
    ("solid", "Solid-state evidence"),
]


def fmt(n) -> str:
    return f"{int(n):,}"


def load():
    cols = [
        "id",
        "entry_id",
        "st_id",
        "split",
        "ph",
        "temperature",
        "ionic_strength",
        "physical_state",
        "membrane_mimetic",
        "cosolvent_evidence",
        "is_bound",
        "n_copies",
        "has_metal",
        "has_nucleic",
        "sample_state_evidence",
    ]
    df = sp.load_main(cols)
    raw = rs.scan_raw()
    cond = [rs.chain_conditions(raw, e, s) for e, s in zip(df.entry_id, df.st_id)]
    n_unres = sum(c is None for c in cond)
    # "deposited": the pipeline's reading (paper_raw_scan.chain_deposited), the definition
    # every figure and macro uses; ``<col>_deposited_raw_scan`` keeps the scan's float()
    # reading, which misses values such as '298K' or '293,15' (traceability only)
    dep = rs.chain_deposited(df.entry_id, df.st_id)
    for col, key in RAW_KEY.items():
        df[f"{col}_deposited"] = [d[col] for d in dep]
        df[f"{col}_deposited_raw_scan"] = [
            c is not None and c[key] is not None for c in cond
        ]
    df["pressure_raw"] = [None if c is None else c["pressure_raw"] for c in cond]
    df["pressure_bar"] = [
        np.nan if c is None or c["pressure_bar"] is None else c["pressure_bar"]
        for c in cond
    ]
    df["membrane"] = df.membrane_mimetic.notna()
    df["multi_copy"] = df.n_copies > 1
    df["homo"] = (df.n_copies > 1) & ~df.is_bound.astype(bool)
    df["solid"] = df.sample_state_evidence == "solid"
    for c in ("is_bound", "has_metal", "has_nucleic", "cosolvent_evidence"):
        df[c] = df[c].fillna(False).astype(bool)
    tids = sp.load_tier_ids()
    for t in TIERS:
        df[f"in_{t}"] = df.id.isin(tids[t])
    return df, n_unres


LOG_FLOOR = 0.6  # y floor of the log histograms: empty bins drop to it


def nested_hist(ax, df, col, bins, logy=False, mask_col=None):
    """Step outlines of the four cumulative tiers (unfiltered first, strict on top).

    Outlines, not filled bars (figure review item 16): filled layers of nested tiers read
    as a stack whose heights add up."""
    for t in TIERS:
        sel = df[f"in_{t}"]
        if mask_col is not None:
            sel &= df[mask_col]
        v = df.loc[sel, col].to_numpy(float)
        v = v[np.isfinite(v)]
        v = np.clip(v, bins[0], bins[-1] - 1e-9)
        h, _ = np.histogram(v, bins=bins)
        h = np.where(h > 0, h, LOG_FLOOR).astype(float) if logy else h
        ax.stairs(h, bins, fill=False, color=sp.TIER_COLORS[t], lw=0.9,
                  baseline=LOG_FLOOR if logy else 0)  # fmt: skip
    if logy:
        ax.set_yscale("log")


def range_bars(ax, col, ys=(0.95, 0.885, 0.82)):
    """Accepted range per tier as horizontal bars at the top of the axes (axes-fraction y);
    the y-limits leave head-room for them."""
    tr = ax.get_xaxis_transform()
    for y, t in zip(ys, ["tolerant", "moderate", "strict"]):
        lo, hi = RANGES[col][t]
        ax.plot(
            [lo, hi],
            [y, y],
            transform=tr,
            color=sp.TIER_COLORS[t],
            lw=1.4,
            solid_capstyle="butt",
            clip_on=False,
        )


def headroom(ax, f=1.75):
    lo, hi = ax.get_ylim()
    ax.set_ylim(0, hi * f)


def thousands(ax, step):
    """Linear count axis with ticks every ``step`` and thousands separators."""
    from matplotlib import ticker

    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))


def main():
    sp.parse_args(sp.cli(__doc__))
    DATA_OUT = sp.out_dir(DATA_SUBDIR)
    df, n_unres = load()
    summ: dict = {"chains": len(df), "unresolved_condition_lists": n_unres}

    # ---------------- numbers for the notes / legend ---------------------------------
    for col in RAW_KEY:
        summ[f"{col}_not_deposited"] = {
            t: int((df[f"in_{t}"] & ~df[f"{col}_deposited"]).sum()) for t in TIERS
        }
    hp = df[df.pressure_bar >= HIGH_P]
    summ["pressure"] = {
        "chains_with_numeric_pressure": int(df.pressure_bar.notna().sum()),
        "chains_no_pressure_row": int(df.pressure_raw.isna().sum()),
        "chains_unreadable_unit": int(
            (df.pressure_raw.notna() & df.pressure_bar.isna()).sum()
        ),
        "chains_recorded_zero": int((df.pressure_bar == 0).sum()),
        "chains_ambient_0.5_to_1.5_bar": int(df.pressure_bar.between(0.5, 1.5).sum()),
        "high_chains": {t: int(hp[f"in_{t}"].sum()) for t in TIERS},
        "high_entries": {
            t: int(hp.loc[hp[f"in_{t}"], "entry_id"].nunique()) for t in TIERS
        },
        "high_list": [
            {
                "id": r.id,
                "bar": round(float(r.pressure_bar), 1),
                "deposited": r.pressure_raw,
                "strictest_tier": [t for t in TIERS if getattr(r, f"in_{t}")][-1],
                "split": r.split,
            }
            for r in hp.itertuples()
        ],
    }
    summ["annotations"] = {
        key: {t: int((df[f"in_{t}"] & df[key]).sum()) for t in TIERS}
        for key, _ in ANNOT
    }
    ps = df.physical_state.fillna("(not recorded)")
    summ["physical_state"] = {
        t: ps[df[f"in_{t}"]].value_counts().to_dict() for t in TIERS
    }
    print(
        json.dumps({k: v for k, v in summ.items() if k != "physical_state"}, indent=1)
    )

    # ---------------- counts of the untrimmed figure (kept in the summary JSON) -------
    summ["ph_above_14"] = int((df.ph_deposited & (df.ph > 14)).sum())
    summ["temperature_outside_255_345"] = int(
        (df.temperature_deposited & ~df.temperature.between(255, 345)).sum()
    )
    summ["ionic_above_1M"] = int(
        (df.ionic_strength_deposited & (df.ionic_strength > 1)).sum()
    )

    # ---------------- figure ----------------------------------------------------------
    # Three panels that Fig. 2 does not show (Fig. 2a: pH x temperature; Fig. 2b:
    # annotation shares). One message per panel and little text: the counts behind
    # them are in S4_summary.json.
    from matplotlib import ticker

    W, H = sp.W_DOUBLE, 53.0
    fig = sp.figure(W, H)

    def ax_mm(x, y_top, w, h):
        return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])

    top, hh = 9.0, 35.0
    lab_y = top - 3.5
    grey_note = {"fontsize": sp.FS_SMALL, "color": "#555555"}

    # a ionic strength (the accepted ranges, 0-7 / 0-5 / 0-3 M, lie beyond the axis)
    ax = ax_mm(12.5, top, 36.0, hh)
    bins = np.arange(-0.0125, 1.0126, 0.025)
    nested_hist(ax, df, "ionic_strength", bins, mask_col="ionic_strength_deposited")
    headroom(ax, 1.15)
    ax.set_xlim(-0.0125, 1.0125)
    ax.set_xlabel("Deposited ionic strength (M)")
    ax.set_ylabel("Chains")
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "≥1"])
    thousands(ax, 1000)
    sp.fig_label(fig, "a", 0.5, lab_y)

    # b pressure
    ax = ax_mm(60.0, top, 36.0, hh)
    pbins = np.logspace(-2, np.log10(5000), 45)
    df["_p"] = df.pressure_bar.where(df.pressure_bar >= 0.01)
    nested_hist(ax, df, "_p", pbins, logy=True)
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 5000)
    ax.set_ylim(LOG_FLOOR, 4e5)
    ax.xaxis.set_major_locator(ticker.FixedLocator([0.01, 1, 100]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,g}"))
    ax.xaxis.set_minor_locator(ticker.FixedLocator([0.1, 10, 1000]))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_locator(ticker.FixedLocator([1, 100, 1e4]))
    # written out with a thousands separator, like the linear count axis of a
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_locator(ticker.FixedLocator([10, 1000, 1e5]))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.axvspan(HIGH_P, 5000, color="#EDEDED", zorder=-5, lw=0)
    ax.text(
        (HIGH_P * 5000) ** 0.5,
        1.5e5,
        f"≥{HIGH_P:.0f}\nbar",
        ha="center",
        va="top",
        linespacing=1.0,
        **grey_note,
    )
    ax.set_xlabel("Deposited pressure (bar)")
    ax.set_ylabel("Chains")
    ps_ = summ["pressure"]
    ps_["chains_below_0.01_bar"] = int((df.pressure_bar < 0.01).sum())
    summ["pressure_strict_entries"] = [
        r["id"].split("_")[0]
        for r in ps_["high_list"]
        if r["strictest_tier"] == "strict"
    ]
    summ["pressure_train_entries"] = [
        r["id"].split("_")[0] for r in ps_["high_list"] if r["split"] == "train"
    ]
    sp.fig_label(fig, "b", 49.0, lab_y)

    # c physical state (marker shape and colour per tier, as the legend)
    ax = ax_mm(129.0, top, 29.0, hh)
    cats = PHYS_MAIN + ["(not recorded)", "other"]
    cats_lab = PHYS_MAIN + ["not recorded", "other values"]
    for j, t in enumerate(TIERS):
        vc = ps[df[f"in_{t}"]].value_counts()
        other = int(vc[[c for c in vc.index if c not in cats]].sum())
        vals = [int(vc.get(c, 0)) for c in cats[:-1]] + [other]
        ys = np.arange(len(cats))[::-1] + (1.5 - j) * 0.17
        v = np.array(vals, float)
        ok = v > 0  # zero counts are not drawn (log axis)
        ax.scatter(
            v[ok],
            ys[ok],
            s=9,
            marker=sp.TIER_MARKERS[t],
            color=sp.TIER_COLORS[t],
            lw=0,
            zorder=3,
        )
    n_other = ps[~ps.isin(cats)].nunique()
    summ["physical_state_other_distinct"] = int(n_other)
    ax.set_xscale("log")
    ax.set_xlim(0.7, 3e4)
    ax.xaxis.set_major_locator(ticker.FixedLocator([1, 100, 1e4]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_locator(ticker.FixedLocator([10, 1000]))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_yticks(np.arange(len(cats))[::-1])
    ax.set_yticklabels(
        [c if c != "other values" else f"other ({n_other} values)" for c in cats_lab],
        fontsize=sp.FS_TICK,
    )
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(cats) - 0.4)
    for y in np.arange(len(cats)):
        ax.axhline(y, color="#EEEEEE", lw=0.4, zorder=0)
    ax.set_xlabel("Chains")
    sp.fig_label(fig, "c", 97.0, lab_y)
    # a short title beside the panel letter, since the category labels fill the y axis
    fig.text(101.0 / W, 1 - (lab_y + 0.3) / H, "Deposited physical state", fontsize=sp.FS_BODY,
             ha="left", va="top")  # fmt: skip

    # shared tier legend: the outline of a and b and the marker of c, in the tier colour
    handles = [
        Line2D(
            [],
            [],
            ls="-",
            lw=0.9,
            marker=sp.TIER_MARKERS[t],
            color=sp.TIER_COLORS[t],
            ms=4.0,
            label=t,
        )
        for t in TIERS
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(12.5 / W, 1.0),
        ncol=4,
        fontsize=sp.FS_TICK,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=1.4,
        frameon=False,
    )

    out = sp.save(fig, "S4")
    print("wrote", out)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    keep = [
        "id",
        "entry_id",
        "st_id",
        "split",
        "ph",
        "ph_deposited",
        "temperature",
        "temperature_deposited",
        "ionic_strength",
        "ionic_strength_deposited",
        "pressure_bar",
        "physical_state",
        "membrane_mimetic",
        "cosolvent_evidence",
        "is_bound",
        "has_metal",
        "has_nucleic",
        "n_copies",
        "sample_state_evidence",
    ] + [f"in_{t}" for t in TIERS]
    df[keep].to_csv(DATA_OUT / "S4_chains.csv", index=False)
    (DATA_OUT / "S4_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    return summ


if __name__ == "__main__":
    main()
