#!/usr/bin/env python3
"""Supplementary Fig. S4 -- sample conditions and annotations per tier.

Every count is a number of chains in the cumulative tier (unfiltered ⊃ tolerant ⊃
moderate ⊃ strict); tier membership comes from the bundle's per-tier ``scores.json``.

Panels
  a-c  pH, temperature and ionic strength as DEPOSITED for the chain's own shift table.
       The Parquet columns ``ph`` / ``temperature`` / ``ionic_strength`` come from the
       unfiltered run, which (like tolerant and moderate) substitutes pH 7.0, 298 K and
       0.1 M when the value is not deposited; the strict run does not, so a chain
       without a deposited value cannot reach strict. The histograms therefore show only
       deposited values; the chains without a deposited value are drawn in panel e. The
       raw scan (``paper_raw_scan.py``) decides what was deposited, resolved through
       ``_Assigned_chem_shift_list.Sample_condition_list_ID`` exactly as the pipeline
       resolves pH and temperature. Bars above each histogram show the accepted range
       of each tier (``filter_defaults``).
  d    Sample pressure (bar), which the release exports (``pressure_bar``) but no tier
       filters on. Read from the same raw sample-condition list, units normalised to
       bar. Log-log histogram; the region >= 400 bar is shaded.
  e    Binary annotations and missing conditions (share of the tier's chains).
  f    Deposited ``_Entity_assembly.Physical_state`` (``physical_state``), counts on a
       log axis; values outside the seven common ones are pooled as "other".

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
    for col, key in RAW_KEY.items():
        df[f"{col}_deposited"] = [c is not None and c[key] is not None for c in cond]
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


def nested_hist(ax, df, col, bins, logy=False, mask_col=None):
    """Filled histograms of the four cumulative tiers, unfiltered at the back."""
    for t in TIERS:
        sel = df[f"in_{t}"]
        if mask_col is not None:
            sel &= df[mask_col]
        v = df.loc[sel, col].to_numpy(float)
        v = v[np.isfinite(v)]
        v = np.clip(v, bins[0], bins[-1] - 1e-9)
        h, _ = np.histogram(v, bins=bins)
        ax.stairs(h, bins, fill=True, color=sp.TIER_COLORS[t], lw=0)
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

    # ---------------- figure ----------------------------------------------------------
    # One message per panel and little text: the counts behind the old in-panel notes
    # (values not deposited, edge bins, pressure rows) are in S4_summary.json, and the
    # share of chains without a deposited pH, temperature or ionic strength is drawn in
    # panel e with the other per-chain annotations.
    W, H = sp.W_DOUBLE, 120.0
    fig = sp.figure(W, H)

    def ax_mm(x, y_top, w, h):
        return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])

    # 160 mm grid: four 28 mm histograms, each with room on its left for tick labels
    # such as "10,000" (thousands separators); panel letters sit at the left edge of
    # each panel's tick-label column
    top, hh, pw = 9.0, 32.0, 28.0
    xs = [12.5, 51.0, 89.5, 127.0]
    lab_x = [0.5, 40.0, 79.0, 117.0]
    lab_y = top - 3.5
    grey_note = {"fontsize": sp.FS_SMALL, "color": "#555555"}

    # a pH
    ax = ax_mm(xs[0], top, pw, hh)
    bins = np.arange(-0.25, 14.26, 0.5)
    nested_hist(ax, df, "ph", bins, mask_col="ph_deposited")
    headroom(ax)
    ax.set_xlim(0, 14)
    ax.set_xlabel("pH")
    ax.set_ylabel("Chains")
    thousands(ax, 4000)
    range_bars(ax, "ph")
    summ["ph_above_14"] = int((df.ph_deposited & (df.ph > 14)).sum())
    sp.fig_label(fig, "a", lab_x[0], lab_y)

    # b temperature
    ax = ax_mm(xs[1], top, pw, hh)
    bins = np.arange(253.75, 346.26, 2.5)
    nested_hist(ax, df, "temperature", bins, mask_col="temperature_deposited")
    headroom(ax)
    ax.set_xlim(255, 345)
    ax.set_xlabel("Temperature (K)")
    ax.set_xticks([275, 300, 325])
    thousands(ax, 5000)
    range_bars(ax, "temperature")
    summ["temperature_outside_255_345"] = int(
        (df.temperature_deposited & ~df.temperature.between(255, 345)).sum()
    )
    sp.fig_label(fig, "b", lab_x[1], lab_y)

    # c ionic strength (the accepted ranges, 0-7 / 0-5 / 0-3 M, lie beyond the axis)
    ax = ax_mm(xs[2], top, pw, hh)
    bins = np.arange(-0.0125, 1.0126, 0.025)
    nested_hist(ax, df, "ionic_strength", bins, mask_col="ionic_strength_deposited")
    headroom(ax, 1.15)
    ax.set_xlim(-0.0125, 1.0125)
    ax.set_xlabel("Ionic strength (M)")
    ax.set_xticks([0, 0.5, 1.0])
    thousands(ax, 1000)
    summ["ionic_above_1M"] = int(
        (df.ionic_strength_deposited & (df.ionic_strength > 1)).sum()
    )
    sp.fig_label(fig, "c", lab_x[2], lab_y)

    # d pressure
    ax = ax_mm(xs[3], top, pw + 2, hh)
    from matplotlib import ticker

    pbins = np.logspace(-2, np.log10(5000), 45)
    df["_p"] = df.pressure_bar.where(df.pressure_bar >= 0.01)
    nested_hist(ax, df, "_p", pbins, logy=True)
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 5000)
    ax.set_ylim(0.6, 4e5)
    ax.xaxis.set_major_locator(ticker.FixedLocator([0.01, 1, 100]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,g}"))
    ax.xaxis.set_minor_locator(ticker.FixedLocator([0.1, 10, 1000]))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_locator(ticker.FixedLocator([1, 100, 1e4]))
    # written out with a thousands separator, like the linear count axes of a-c
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
    ax.set_xlabel("Pressure (bar)")
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
    sp.fig_label(fig, "d", lab_x[3], lab_y)

    # shared tier legend (colour for every panel, marker shape for f)
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker=sp.TIER_MARKERS[t],
            color=sp.TIER_COLORS[t],
            ms=4.5,
            label=t,
        )
        for t in TIERS
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.0),
        ncol=4,
        fontsize=sp.FS_TICK,
        handletextpad=0.2,
        columnspacing=1.4,
        frameon=False,
    )

    # e per-chain annotations, as shares of the tier's chains
    top2 = top + hh + 15
    h2 = H - top2 - 9
    ax = ax_mm(41, top2, 50, h2)
    n_t = {t: int(df[f"in_{t}"].sum()) for t in TIERS}
    nd_rows = [
        (f"nd:{col}", lab)
        for col, lab in (
            ("ph", "No deposited pH"),
            ("temperature", "No deposited temperature"),
            ("ionic_strength", "No deposited ionic strength"),
        )
    ]
    rows = ANNOT + nd_rows
    share = {}
    for key, _ in ANNOT:
        share[key] = {t: summ["annotations"][key][t] for t in TIERS}
    for key, _ in nd_rows:
        share[key] = summ[f"{key[3:]}_not_deposited"]
    nrow = len(rows)
    bh = 0.19
    for i, (key, label) in enumerate(rows):
        y0 = nrow - 1 - i
        for j, t in enumerate(TIERS):
            pct = 100 * share[key][t] / n_t[t]
            ax.barh(
                y0 + (1.5 - j) * bh, pct, height=bh * 0.9, color=sp.TIER_COLORS[t], lw=0
            )
        ax.text(
            -0.02,
            y0,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=sp.FS_TICK,
        )
    ax.axhline(len(nd_rows) - 0.5, color="#C8C8C8", lw=0.5, zorder=0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.55, nrow - 0.45)
    xmax = max(100 * share[k][t] / n_t[t] for k, _ in rows for t in TIERS)
    ax.set_xlim(0, xmax * 1.05)
    ax.set_xlabel("Chains in the tier (%)")
    sp.fig_label(fig, "e", 0.5, top2 - 5)

    # f physical state
    ax = ax_mm(128, top2, 29, h2)
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
    ax.xaxis.set_minor_locator(ticker.FixedLocator([10, 1000]))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_yticks(np.arange(len(cats))[::-1])
    ax.set_yticklabels(
        [c if c != "other values" else f"other ({n_other} values)" for c in cats_lab],
        fontsize=sp.FS_TICK,
    )
    ax.tick_params(axis="y", length=0)
    for y in np.arange(len(cats)):
        ax.axhline(y, color="#EEEEEE", lw=0.4, zorder=0)
    ax.set_xlabel("Chains")
    ax.set_ylabel("Deposited physical state")
    sp.fig_label(fig, "f", 94, top2 - 5)

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
