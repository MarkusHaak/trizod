#!/usr/bin/env python3
"""Fig. 2 -- Beyond labels, the release keeps shifts, conditions, states and repeated measurements.

One 160 mm figure, 2 x 2 panels (each about 78 x 52 mm), placed at 1:1. Every count is a
chain of the release (``trizod_dataset.parquet``, all 17,403 chains = the unfiltered tier)
unless a panel says otherwise; tier membership comes from the bundle's per-tier
``scores.json`` (cumulative: unfiltered > tolerant > moderate > strict).

Panels
  a  pH against temperature of every chain whose own condition list deposits both, as a
     grey hexagon density (log colour scale, chains per hexagon), with the condition
     windows of the tolerant, moderate and strict tiers (``filter_defaults``; the
     unfiltered tier has none) as nested outlines. Chains without a deposited pH or
     temperature hold a default in the release (pH 7.0, 298 K); they are drawn apart, in
     the row and column marked "none", never inside the density.
  b  Share of each tier's chains with each annotation, as a table: one row per
     annotation, one column per tier (a strip of the tier colour above each column), the
     share printed in every cell and shaded by it (grey, log scale); the chain count of
     the release at the right. Since 13 Nov 2026 (TS: which marker belongs to which
     annotation was hard to see in the dot plot).
  c  The released shifts (``trizod_shifts.parquet``) per atom type: the seven scoring
     atom types (``is_backbone`` rows: H, Hα (HA, HA2, HA3), Hβ (HB, HB1-3), Cα, Cβ, C′,
     N; filled) and all other atoms of each nucleus (open), in millions. Since 13 Nov 2026
     (TS: the caption named seven types, the old bars showed three nuclei).
  d  Repeated measurements: exact sequences by the number of BMRB entries that measured
     them (all chains, bound ones included; ``release_stats.repeat_histogram``), log-log
     dots, with the count of sequences measured in two or more entries and their two
     uses written in the empty corner (no shading since 13 Nov 2026).

"Deposited" follows the pipeline: the paper's one definition,
``paper_raw_scan.chain_deposited`` (the chain's own condition rows read with
``trizod.bmrb.bmrb.SampleConditions``, so a unit inside the value, '298K', a decimal comma,
'293,15', or a range, '298-303', counts as deposited, as in the release). ``s4.load`` sets
``<cond>_deposited`` with it, and :func:`deposited_flags` turns that into
``<cond>_imputed``: a chain counts as imputed exactly when the release holds a default. The
same definition feeds Supplementary Fig. S4, the condition comparison of Fig. 4a and the
``\\n*NotDeposited*`` macros.

Run:  uv run --extra figures --with pyarrow python scripts/figures/paper/fig2_composition.py \
          --out-dir <draft>/Figures/paper [--refresh]
Outputs: <out-dir>/fig2_composition.pdf and <out-dir>/source_data/fig2{a,b,c,d}*.csv
(panel c: fig2c_atom_types.csv, plotted; fig2c_shifts.csv, totals per nucleus),
fig2_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import paper_raw_scan as rs  # noqa: E402
import release_stats as rst  # noqa: E402
import style_paper as sp  # noqa: E402
import supp_s4_conditions as s4  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib import ticker  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

TIERS = sp.TIERS
DATA_SUBDIR = "source_data"  # under --out-dir
NAME = "fig2_composition"
COND = ("ph", "temperature", "ionic_strength")

# panel a: axis ranges (cover every deposited value: pH 1.0-11.1, 252-373 K)
T_LIM = (248.0, 377.0)
PH_LIM = (0.3, 13.4)  # head-room above pH 12 for the window labels
NONSTATE = ["denatured", "unfolded", "molten globule"]  # "non-native state" in b

# panel b rows: (key, label); value order by the release share, the missing condition last
ANNOT_ROWS = [
    ("is_bound", "Second molecule"),
    ("cosolvent_evidence", "Cosolvent"),  # as Fig. 4d and the Fig. 2 legend
    ("homo", "Homo-oligomer"),
    ("membrane", "Membrane mimetic"),
    ("solid", "Solid sample"),
    ("nonnative", "Non-native state"),
]
MISSING_ROWS = [("no_ionic", "No ionic strength")]
# panel c row labels: the measured isotope of each nucleus, as the legend writes them
# (1H, 13C, 15N); the mass number is set as its own 6-pt text, because a mathtext
# superscript of a 7-pt label is 4.9 pt and the 6-pt floor would blow the label up
MASS_NUMBER = {"H": "1", "C": "13", "N": "15"}

# grey density: Greys sampled from light to mid-dark, so the darkest strict outline stays
# visible on the densest hexagon
DENSITY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "density", [mcolors.to_hex(sp.plt.get_cmap("Greys")(v)) for v in (0.12, 0.45, 0.85)]
)
# layout (mm): two rows of 52 mm; in every panel the plot's bottom edge sits AX_BOTTOM
# below the row top, so the x axes of a and b (and of c and d) line up
ROW_H, ROW_GAP, AX_BOTTOM = 52.0, 2.0, 42.0
HALO = [pe.withStroke(linewidth=1.8, foreground="white")]
UBQ_LABEL = rst.LABELS[rst.UBQ]  # "ubiquitin"


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------
def deposited_flags(df: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Add ``<cond>_imputed`` (True: the release holds a default) for pH, temperature and
    ionic strength: the negation of the pipeline-read ``<cond>_deposited``
    (``paper_raw_scan.chain_deposited``, set by ``s4.load``; recomputed here when ``df``
    lacks it or ``refresh`` is set)."""
    df = df.copy()
    if refresh or not all(f"{c}_deposited" in df for c in COND):
        dep = rs.chain_deposited(df.entry_id, df.st_id, refresh=refresh)
        for c in COND:
            df[f"{c}_deposited"] = [d[c] for d in dep]
    for c in COND:
        df[f"{c}_imputed"] = ~df[f"{c}_deposited"].astype(bool)
    return df


def load_chains(refresh: bool = False) -> pd.DataFrame:
    """One row per chain: conditions, imputation flags, annotations, tier membership."""
    df, _ = s4.load()
    df = deposited_flags(df, refresh=refresh)
    df["nonnative"] = df.physical_state.isin(NONSTATE)
    df["no_ionic"] = df.ionic_strength_imputed
    return df


def annotation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Panel b: chains with each annotation per tier, and their share of the tier."""
    rows = []
    n_t = {t: int(df[f"in_{t}"].sum()) for t in TIERS}
    for key, label in ANNOT_ROWS + MISSING_ROWS:
        for t in TIERS:
            n = int((df[f"in_{t}"] & df[key]).sum())
            rows.append(
                {
                    "annotation": label,
                    "column": key,
                    "tier": t,
                    "chains": n,
                    "tier_chains": n_t[t],
                    "share_pct": 100 * n / n_t[t],
                }
            )
    return pd.DataFrame(rows)


def shift_table() -> pd.DataFrame:
    """Panel c: released shifts by nucleus and class (backbone / side chain)."""
    import pyarrow.parquet as pq

    sh = pq.read_table(sp.PQ_SHIFTS, columns=["atom_type", "is_backbone"]).to_pandas()
    ct = pd.crosstab(sh.atom_type.astype(str), sh.is_backbone)
    out = pd.DataFrame(
        {
            "nucleus": ["H", "C", "N"],
            "backbone": [int(ct.loc[n, True]) for n in ("H", "C", "N")],
            "side_chain": [int(ct.loc[n, False]) for n in ("H", "C", "N")],
        }
    )
    out["total"] = out.backbone + out.side_chain
    assert int(out.total.sum()) == len(sh), "nuclei other than H, C, N in the table"
    return out


# ----------------------------------------------------------------------------------
# panels
# ----------------------------------------------------------------------------------
def draw_a(fig, box, df, summ):
    """pH x temperature density with tier windows; imputed values apart."""
    x0, y0, W_, H_ = box  # mm, top-left origin of the panel
    FW, FH = fig.get_size_inches() / sp.MM

    def ax_mm(x, y_top, w, h):
        return fig.add_axes([x / FW, 1 - (y_top + h) / FH, w / FW, h / FH])

    strip = 3.6  # mm, the "none" row / column
    gap = 0.8
    main_w, main_h = 44.0, 33.0
    left, top = x0 + 10.0, y0 + AX_BOTTOM - main_h - gap - strip
    ax = ax_mm(left, top + strip + gap, main_w, main_h)
    ax_top = ax_mm(left, top, main_w, strip)  # pH not deposited: temperature known
    ax_rt = ax_mm(left + main_w + gap, top + strip + gap, strip, main_h)  # T missing
    ax_cn = ax_mm(left + main_w + gap, top, strip, strip)  # both missing
    cax = ax_mm(
        left + main_w + gap + strip + 3.2, top + strip + gap + 6, 1.6, main_h - 12
    )

    both = ~df.ph_imputed & ~df.temperature_imputed
    d = df[both]
    gx = 34  # hexagons across the temperature axis
    norm = mcolors.LogNorm(vmin=1, vmax=None)
    hb = ax.hexbin(
        d.temperature,
        d.ph,
        gridsize=(gx, int(round(gx * main_h / main_w / np.sqrt(3)))),
        extent=(*T_LIM, *PH_LIM),
        mincnt=1,
        cmap=DENSITY_CMAP,
        norm=norm,
        linewidths=0.15,
        edgecolors="face",
        rasterized=True,
    )
    vmax = hb.get_array().max()
    summ["a_hexagon_max"] = int(vmax)
    ax.set_xlim(*T_LIM)
    ax.set_ylim(*PH_LIM)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("pH")
    # no tick at 375 K: its label would run into the "none" of the right column
    ax.xaxis.set_major_locator(ticker.FixedLocator([250, 275, 300, 325, 350]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([2, 4, 6, 8, 10, 12]))

    # tier windows, outermost first; labels above each window's top-left corner
    for t in ("tolerant", "moderate", "strict"):
        (plo, phi), (tlo, thi) = s4.RANGES["ph"][t], s4.RANGES["temperature"][t]
        ax.add_patch(
            Rectangle(
                (tlo, plo),
                thi - tlo,
                phi - plo,
                fill=False,
                ec=sp.TIER_COLORS[t],
                lw=1.7,
                zorder=4,
                path_effects=[pe.withStroke(linewidth=2.7, foreground="white")],
            )
        )
        txt = ax.text(
            tlo + 1.0,
            phi + 0.08,
            t,
            ha="left",
            va="bottom",
            fontsize=sp.FS_BODY,
            color=sp.INK,
            zorder=5,
        )
        txt.set_path_effects(HALO)

    # chains whose pH (top row) or temperature (right column) is a default
    bins_t = np.linspace(*T_LIM, gx + 1)
    bins_p = np.linspace(*PH_LIM, int(round((gx + 1) * main_h / main_w)))
    only_ph = df[df.ph_imputed & ~df.temperature_imputed]
    only_t = df[~df.ph_imputed & df.temperature_imputed]
    n_both = int((df.ph_imputed & df.temperature_imputed).sum())
    h1, _ = np.histogram(only_ph.temperature, bins=bins_t)
    h2, _ = np.histogram(only_t.ph, bins=bins_p)
    m1 = np.ma.masked_less(h1[None, :].astype(float), 1)
    m2 = np.ma.masked_less(h2[:, None].astype(float), 1)
    ax_top.pcolormesh(bins_t, [0, 1], m1, cmap=DENSITY_CMAP, norm=norm, rasterized=True)
    ax_rt.pcolormesh([0, 1], bins_p, m2, cmap=DENSITY_CMAP, norm=norm, rasterized=True)
    if n_both:
        ax_cn.pcolormesh(
            [0, 1], [0, 1], np.array([[n_both]], float), cmap=DENSITY_CMAP, norm=norm
        )
    for a in (ax_top, ax_rt, ax_cn):
        for s_ in a.spines.values():
            s_.set_visible(True)
            s_.set_linewidth(0.5)
            s_.set_color(sp.INK_SOFT)
        a.set_xticks([])
        a.set_yticks([])
    ax_top.set_xlim(*T_LIM)
    ax_top.set_ylim(0, 1)
    ax_rt.set_ylim(*PH_LIM)
    ax_rt.set_xlim(0, 1)
    # the strips are an extra "none" category on each axis
    ax_top.text(
        -0.012, 0.5, "none", transform=ax_top.transAxes, ha="right", va="center",
        fontsize=sp.FS_TICK,
    )  # fmt: skip
    ax_rt.text(
        0.5, -0.012, "none", transform=ax_rt.transAxes, ha="center", va="top",
        fontsize=sp.FS_TICK,
    )  # fmt: skip
    summ["a_imputed"] = {
        "ph_only": len(only_ph),
        "temperature_only": len(only_t),
        "both": n_both,
        "ph_any": int(df.ph_imputed.sum()),
        "temperature_any": int(df.temperature_imputed.sum()),
    }

    cb = fig.colorbar(hb, cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, width=0.4)
    cb.set_ticks([t for t in (1, 10, 100, 1000) if t <= vmax])
    cb.ax.yaxis.set_minor_locator(ticker.NullLocator())
    cb.ax.set_title("Chains", fontsize=sp.FS_BODY, loc="left", pad=3)
    sp.fig_label(fig, "a", x0, y0)
    return ax


def _panel_mm_axes(fig, box):
    """An invisible axes over the whole panel, in mm from its top-left corner."""
    x0, y0, W_, H_ = box
    FW, FH = fig.get_size_inches() / sp.MM
    ax = fig.add_axes([x0 / FW, 1 - (y0 + H_) / FH, W_ / FW, H_ / FH])
    ax.set_xlim(0, W_)
    ax.set_ylim(H_, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    return ax


def share_text(v: float) -> str:
    """Cell text of panel b: the share in per cent, two significant digits at most."""
    if v == 0:
        return "0"
    if v < 0.1:
        return "<0.1"
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


def share_color(v: float) -> str:
    """Cell fill of panel b: light to mid grey with the share (log, 0.1 to 50 %)."""
    if v == 0:
        return "white"
    t = (np.log10(max(v, 0.1)) + 1) / (np.log10(50.0) + 1)
    return mcolors.to_hex(sp.plt.get_cmap("Greys")(0.04 + 0.44 * min(t, 1.0)))


def draw_b(fig, box, tab):
    """Annotation shares per tier as a table: one row per annotation, one column per
    tier (a strip of the tier colour above each column), the share printed in every cell
    and shaded by it; the chain count of the release at the right."""
    x0, y0, W_, H_ = box
    ax = _panel_mm_axes(fig, box)
    lab_x = 22.6  # right edge of the row labels
    c0, cw, cg = 23.6, 11.3, 0.4  # first cell, cell width, gap
    n_x = W_ - 0.3  # right edge of the count column
    top, rh, sep = 11.4, 4.3, 1.6  # first row top, row height, gap before "missing"
    cx = [c0 + j * (cw + cg) + cw / 2 for j in range(len(TIERS))]
    # header: what the cells hold, then the tier names on a strip of their colour
    ax.text((c0 + cx[-1] + cw / 2) / 2, 4.2, "Share of the tier's chains (%)",
            ha="center", va="baseline", fontsize=sp.FS_BODY)  # fmt: skip
    for j, t in enumerate(TIERS):
        ax.text(cx[j], 8.2, t, ha="center", va="baseline", fontsize=sp.FS_BODY)
        ax.add_patch(Rectangle((cx[j] - cw / 2, 9.2), cw, 1.2, color=sp.TIER_COLORS[t],
                               lw=0))  # fmt: skip
    ax.text(n_x, 8.2, "Chains", ha="right", va="baseline", fontsize=sp.FS_BODY)
    rows = ANNOT_ROWS + MISSING_ROWS
    for i, (key, label) in enumerate(rows):
        y = top + i * rh + (sep if i >= len(ANNOT_ROWS) else 0.0)
        yc = y + rh / 2
        sub = tab[tab.column == key].set_index("tier")
        ax.text(lab_x, yc, label, ha="right", va="center", fontsize=sp.FS_BODY)
        for j, t in enumerate(TIERS):
            v = float(sub.loc[t, "share_pct"])
            ax.add_patch(Rectangle((cx[j] - cw / 2, y + 0.2), cw, rh - 0.4,
                                   facecolor=share_color(v), edgecolor="#D9D9D9",
                                   lw=0.4 if v == 0 else 0))  # fmt: skip
            ax.text(cx[j], yc, share_text(v), ha="center", va="center",
                    fontsize=sp.FS_BODY, color=sp.INK if v else sp.INK_SOFT)  # fmt: skip
        n_rel = int(sub.loc["unfiltered", "chains"])
        ax.text(n_x, yc, f"{n_rel:,}", ha="right", va="center", fontsize=sp.FS_BODY)
    ysep = top + len(ANNOT_ROWS) * rh + sep / 2
    ax.plot([1.0, n_x], [ysep, ysep], color=sp.CAT["light"], lw=0.5)
    sp.fig_label(fig, "b", x0, y0)
    return ax


# panel c: the seven scoring atom types (the shift table's is_backbone rows) and all
# other atoms, grouped by nucleus; (row label, atom IDs or None for "all other atoms")
ATOM_ROWS = {
    "H": [("H", ("H",)), ("Hα", ("HA", "HA2", "HA3")),
          ("Hβ", ("HB", "HB1", "HB2", "HB3")), ("other", None)],
    "C": [("Cα", ("CA",)), ("Cβ", ("CB",)), ("C′", ("C",)), ("other", None)],
    "N": [("N", ("N",)), ("other", None)],
}  # fmt: skip


def atom_type_table() -> pd.DataFrame:
    """Panel c: released shifts per scoring atom type, and all other atoms per nucleus."""
    import pyarrow.parquet as pq

    sh = pq.read_table(
        sp.PQ_SHIFTS, columns=["atom_id", "atom_type", "is_backbone"]
    ).to_pandas()
    bb = sh[sh.is_backbone.astype(bool)].atom_id.astype(str).value_counts()
    other = sh[~sh.is_backbone.astype(bool)].atom_type.astype(str).value_counts()
    rows = []
    for nuc, items in ATOM_ROWS.items():
        for label, atoms in items:
            if atoms is None:
                n, scoring = int(other.get(nuc, 0)), False
            else:
                n, scoring = int(sum(bb.get(a, 0) for a in atoms)), True
            rows.append({"nucleus": nuc, "row": label, "scoring_atom_type": scoring,
                         "atom_ids": "" if atoms is None else ";".join(atoms),
                         "shifts": n})  # fmt: skip
    out = pd.DataFrame(rows)
    assert int(out.shifts.sum()) == len(sh), "atom IDs outside the seven types"
    return out


def _isotope_label(ax, y: float, nucleus: str, x: float = -0.02) -> None:
    """Group label of panel c: the element at 7 pt, its mass number raised at 6 pt."""
    from matplotlib.transforms import offset_copy

    tr = ax.get_yaxis_transform()
    t = ax.text(x, y, nucleus, transform=tr, ha="right", va="center",
                fontsize=sp.FS_TICK)  # fmt: skip
    fig = ax.figure
    fig.canvas.draw()
    w_mm = t.get_window_extent().width / fig.dpi / sp.MM
    ax.text(x, y, MASS_NUMBER[nucleus], ha="right", va="bottom", fontsize=sp.FS_SMALL,
            transform=offset_copy(tr, fig=fig, x=-(w_mm + 0.15) * sp.MM, y=0.25 * sp.MM,
                                  units="inches"))  # fmt: skip


def draw_c(fig, box, at):
    """Shifts per atom type: the seven scoring atom types filled, other atoms open."""
    x0, y0, W_, H_ = box
    FW, FH = fig.get_size_inches() / sp.MM
    h = 33.0
    left, top, w = x0 + 17.0, y0 + AX_BOTTOM - h, W_ - 17.0 - 6.0
    ax = fig.add_axes([left / FW, 1 - (top + h) / FH, w / FW, h / FH])
    fill, bh, ggap = sp.INK_SOFT, 0.68, 0.7
    ys, y, groups = [], 0.0, {}
    for nuc, items in ATOM_ROWS.items():
        g = []
        for _ in items:
            ys.append(y)
            g.append(y)
            y += 1.0
        groups[nuc] = float(np.mean(g))
        y += ggap
    for yy, r in zip(ys, at.itertuples()):
        v = r.shifts / 1e6
        kw = ({"color": fill, "ec": fill} if r.scoring_atom_type
              else {"color": "white", "ec": fill})  # fmt: skip
        ax.barh(yy, v, height=bh, lw=0.6, **kw)
        ax.text(v + 0.05, yy, f"{v:.2f}", ha="left", va="center", fontsize=sp.FS_TICK)
    ax.set_yticks(ys)
    ax.set_yticklabels(at.row)
    ax.tick_params(axis="y", length=0, pad=2)
    for nuc, yc in groups.items():
        _isotope_label(ax, yc, nuc, x=-0.17)
    for nuc in list(groups)[1:]:  # thin rules between the nucleus groups
        yb = (
            min(yy for yy, r in zip(ys, at.itertuples()) if r.nucleus == nuc)
            - 0.5
            - ggap / 2
        )
        ax.plot([-0.25, 1.0], [yb, yb], transform=ax.get_yaxis_transform(),
                color="#E6E6E6", lw=0.4, clip_on=False)  # fmt: skip
    ax.set_ylim(ys[-1] + 0.6, ys[0] - 0.6)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 3.0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel("Shifts (millions)")
    # key: filled = scoring atom types, open = other atoms (above the plot)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=fill, edgecolor=fill, lw=0.6,
                  label="scoring atom types"),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=fill, lw=0.6,
                  label="other atoms"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=((x0 + 5.0) / FW, 1 - (y0 + 0.2) / FH), ncol=2,
               handlelength=1.2, handleheight=0.8, handletextpad=0.4, columnspacing=1.2,
               borderaxespad=0, borderpad=0, frameon=False)  # fmt: skip
    sp.fig_label(fig, "c", x0, y0)
    return ax


def draw_d(fig, box, rh: rst.RepeatHistogram, stats: dict):
    """Exact sequences by the number of entries that measured them (log-log)."""
    x0, y0, W_, H_ = box
    FW, FH = fig.get_size_inches() / sp.MM
    h = 37.0
    left, top, w = x0 + 12.0, y0 + AX_BOTTOM - h, W_ - 12.0 - 2.0
    ax = fig.add_axes([left / FW, 1 - (top + h) / FH, w / FW, h / FH])
    hist = rh.hist
    xlo, xhi = 0.75, 62
    ax.plot(hist.n_entries, hist.n_sequences, **sp.marker_kw("release", ms=3.2),
            zorder=3, clip_on=False)  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0.6, 4e4)
    ax.xaxis.set_major_locator(ticker.FixedLocator([1, 2, 5, 10, 20, 50]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_locator(ticker.FixedLocator([1, 10, 100, 1000, 10000]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("BMRB entries that measured the sequence")
    ax.set_ylabel("Sequences")
    # the repeated sequences: one bracket over every dot at 2 or more entries, with
    # their count (the legend gives the chains and the uses; no prose in the panel)
    yb = 3000.0
    ax.plot([2, 2, 50, 50], [yb / 1.35, yb, yb, yb / 1.35], color=sp.INK_SOFT, lw=0.5,
            zorder=2, clip_on=False)  # fmt: skip
    ax.text(50, yb * 1.25, f"2 or more entries: {stats['n_seqs_multi_entry']:,}",
            ha="right", va="bottom", fontsize=sp.FS_BODY)  # fmt: skip
    # the most-measured sequences, named (the two at the maximum, and ubiquitin)
    # (from repeat_stats, so the labels never depend on how many rows rh.top keeps);
    # common names in lower case ("Calmodulin" -> "calmodulin"), as in the text
    nmax = int(stats["n_entries_max_per_seq"])
    at_max = [lab.lower() if lab[1:].islower() else lab
              for lab in stats["labels_at_max"]]  # fmt: skip
    for n_e, text, y_txt in (
        (nmax, ",\n".join(at_max), 14.0),
        (int(stats["n_entries_ubq_exact"]), UBQ_LABEL, 4.0),
    ):
        y = int(hist.loc[hist.n_entries == n_e, "n_sequences"].iloc[0])
        ax.plot([n_e, n_e], [y * 1.35, y_txt * 0.9], color=sp.INK_SOFT, lw=0.4,
                zorder=2)  # fmt: skip
        ax.text(n_e * 1.04, y_txt, text, ha="right", va="bottom", fontsize=sp.FS_BODY,
                linespacing=1.05)  # fmt: skip
    # "measured once": the single largest group, named at its point
    y1 = int(hist.loc[hist.n_entries == 1, "n_sequences"].iloc[0])
    ax.text(1.12, y1, f"measured once: {y1:,}", ha="left", va="center",
            fontsize=sp.FS_BODY)  # fmt: skip
    sp.fig_label(fig, "d", x0, y0)
    return ax


# ----------------------------------------------------------------------------------
def main():
    ap = sp.cli(__doc__)
    ap.add_argument(
        "--refresh", action="store_true", help="re-read the raw condition rows"
    )
    args = sp.parse_args(ap)
    data_out = sp.out_dir(DATA_SUBDIR)
    df = load_chains(refresh=args.refresh)
    summ: dict = {"chains": len(df)}
    for c in COND:
        summ[f"{c}_imputed"] = {
            t: int((df[f"in_{t}"] & df[f"{c}_imputed"]).sum()) for t in TIERS
        }
        summ[f"{c}_not_deposited_raw_scan"] = {
            t: int((df[f"in_{t}"] & ~df[f"{c}_deposited"]).sum()) for t in TIERS
        }
    tab = annotation_table(df)
    st = shift_table()
    at = atom_type_table()
    rh = rst.repeat_histogram()
    stats = rst.repeat_stats()
    summ["repeat_stats"] = stats
    summ["shifts"] = st.set_index("nucleus").to_dict(orient="index")

    W, H = sp.W_DOUBLE, 2 * ROW_H + ROW_GAP
    fig = sp.figure(W, H)
    col2 = W - sp.W_SINGLE
    row2 = ROW_H + ROW_GAP
    draw_a(fig, (0.0, 0.0, sp.W_SINGLE, ROW_H), df, summ)
    draw_b(fig, (col2, 0.0, sp.W_SINGLE, ROW_H), tab)
    draw_c(fig, (0.0, row2, sp.W_SINGLE, ROW_H), at)
    draw_d(fig, (col2, row2, sp.W_SINGLE, ROW_H), rh, stats)
    out = sp.save(fig, NAME)
    print("wrote", out)

    data_out.mkdir(parents=True, exist_ok=True)
    keep = ["id", "entry_id", "st_id", "ph", "ph_imputed", "temperature",
            "temperature_imputed", "ionic_strength", "ionic_strength_imputed"]  # fmt: skip
    df[keep + [f"in_{t}" for t in TIERS]].to_csv(data_out / "fig2a_conditions.csv",
                                                 index=False)  # fmt: skip
    tab.to_csv(data_out / "fig2b_annotations.csv", index=False)
    st.to_csv(data_out / "fig2c_shifts.csv", index=False)
    at.to_csv(data_out / "fig2c_atom_types.csv", index=False)
    rh.hist.to_csv(data_out / "fig2d_repeats.csv", index=False)
    rh.top.to_csv(data_out / "fig2d_top_sequences.csv", index=False)
    (data_out / "fig2_summary.json").write_text(json.dumps(summ, indent=1, default=int))
    print(json.dumps({k: v for k, v in summ.items() if k != "repeat_stats"}, indent=1,
                     default=int))  # fmt: skip
    return summ


if __name__ == "__main__":
    main()
