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
     temperature hold a default in the release (pH 7.0, 298 K; not yet flagged); they are
     drawn apart, in the row and column marked "none", never inside the density.
  b  Share of each tier's chains with each annotation (dot plot, one row per annotation,
     the tier ramp light -> dark, log axis with zero drawn at its own tick); the chain
     count in the release is printed at the right.
  c  The released shifts (``trizod_shifts.parquet``) by nucleus, backbone (the atom IDs
     the scorer reads; filled) against side chain (open), counts in millions printed. No
     offset shading: re-referencing is not this figure's message.
  d  Repeated measurements: exact sequences by the number of BMRB entries that measured
     them (all chains, bound ones included; ``release_stats.repeat_histogram``), log-log
     dots. The region of sequences measured in two or more entries is shaded and labelled
     with both its uses.

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
Outputs: <out-dir>/fig2_composition.pdf and <out-dir>/source_data/fig2{a,b,c,d}*.csv,
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
from matplotlib.lines import Line2D  # noqa: E402
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
SHADE = "#EFEFEF"  # background of the "measured in >= 2 entries" region (not an entity)
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
                lw=1.1,
                zorder=4,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
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


def draw_b(fig, box, tab):
    """Annotation shares per tier: dot plot on a log axis with a zero slot."""
    x0, y0, W_, H_ = box
    FW, FH = fig.get_size_inches() / sp.MM
    lab_w, n_w = 21.0, 9.0
    h = 31.0
    top = y0 + AX_BOTTOM - h
    ax = fig.add_axes(
        [(x0 + lab_w) / FW, 1 - (top + h) / FH, (W_ - lab_w - n_w) / FW, h / FH]
    )
    rows = ANNOT_ROWS + MISSING_ROWS
    zero_x, lo, hi = 0.055, 0.042, 60.0
    nrow = len(rows)
    dy = 0.22  # tier dodge within a row: the four markers no longer overlap (review 12)
    for i, (key, label) in enumerate(rows):
        yc = nrow - 1 - i + (0.0 if i < len(ANNOT_ROWS) else -0.25)
        sub = tab[tab.column == key].set_index("tier")
        for j, t in enumerate(TIERS):
            v = sub.loc[t, "share_pct"]
            ax.plot(
                zero_x if v == 0 else v,
                yc + (1.5 - j) * dy,
                ls="",
                marker=sp.TIER_MARKERS[t],
                color=sp.TIER_COLORS[t],
                mec=sp.TIER_COLORS[t],
                ms=3.0,
                mew=0.6,
                zorder=3,
                clip_on=False,
            )
        ax.text(-0.03, yc, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=sp.FS_TICK)  # fmt: skip
        n_rel = int(sub.loc["unfiltered", "chains"])
        ax.text(1.0 + n_w / (W_ - lab_w - n_w), yc, f"{n_rel:,}",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=sp.FS_TICK)  # fmt: skip
        ax.axhline(yc, color="#EEEEEE", lw=0.4, zorder=0)
    ysep = nrow - len(ANNOT_ROWS) - 0.5 - 0.12
    ax.axhline(ysep, color=sp.CAT["light"], lw=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.25 - 0.8, nrow - 0.45)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(ticker.FixedLocator([zero_x, 0.1, 1, 10]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["0", "0.1", "1", "10"]))
    ax.xaxis.set_minor_locator(
        ticker.FixedLocator([v for d in (0.1, 1, 10) for v in d * np.arange(2, 10)])
    )
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    # axis break between the zero slot and 0.1
    for xb in (0.068, 0.074):
        ax.plot([xb / 1.03, xb * 1.03], [-0.035, 0.035], transform=ax.get_xaxis_transform(),
                color=sp.INK, lw=0.6, clip_on=False)  # fmt: skip
    ax.set_xlabel("Chains in the tier (%)")
    # header of the count column
    ax.text(1.0 + n_w / (W_ - lab_w - n_w), 1.0, "All chains", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=sp.FS_TICK)  # fmt: skip
    # tier key (one entity, four ordered levels), above the plot
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker=sp.TIER_MARKERS[t],
            color=sp.TIER_COLORS[t],
            ms=3.0,
            mew=0.6,
            label=t,
        )  # fmt: skip
        for t in TIERS
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=((x0 + 5.0) / FW, 1 - (y0 + 0.2) / FH),
        ncol=4,
        handletextpad=0.1,
        columnspacing=0.9,
        borderaxespad=0,
        borderpad=0,
        frameon=False,
    )
    sp.fig_label(fig, "b", x0, y0)
    return ax


def _isotope_label(ax, y: float, nucleus: str) -> None:
    """Row label of panel c: the element at 7 pt, its mass number raised at 6 pt."""
    from matplotlib.transforms import offset_copy

    tr = ax.get_yaxis_transform()
    t = ax.text(-0.02, y, nucleus, transform=tr, ha="right", va="center",
                fontsize=sp.FS_TICK)  # fmt: skip
    fig = ax.figure
    fig.canvas.draw()
    w_mm = t.get_window_extent().width / fig.dpi / sp.MM
    ax.text(-0.02, y, MASS_NUMBER[nucleus], ha="right", va="bottom", fontsize=sp.FS_SMALL,
            transform=offset_copy(tr, fig=fig, x=-(w_mm + 0.15) * sp.MM, y=0.25 * sp.MM,
                                  units="inches"))  # fmt: skip


def draw_c(fig, box, st):
    """Shifts by nucleus: backbone (filled) and side chain (open) stacked bars."""
    x0, y0, W_, H_ = box
    FW, FH = fig.get_size_inches() / sp.MM
    h = 34.0
    left, top, w = x0 + 10.0, y0 + AX_BOTTOM - h, W_ - 10.0 - 3.0
    ax = fig.add_axes([left / FW, 1 - (top + h) / FH, w / FW, h / FH])
    ys = np.array([2, 1, 0])
    bh = 0.62
    fill = sp.INK_SOFT
    for y, r in zip(ys, st.itertuples()):
        b, s = r.backbone / 1e6, r.side_chain / 1e6
        ax.barh(y, b, height=bh, color=fill, ec=fill, lw=0.6)
        ax.barh(y, s, left=b, height=bh, color="white", ec=fill, lw=0.6)
        ax.text(b / 2, y, f"{b:.2f}", ha="center", va="center", color="white",
                fontsize=sp.FS_TICK)  # fmt: skip
        if s > 0.6:
            ax.text(b + s / 2, y, f"{s:.2f}", ha="center", va="center", color=sp.INK,
                    fontsize=sp.FS_TICK)  # fmt: skip
        else:
            ax.text(b + s + 0.08, y, f"{s:.2f}", ha="left", va="center", color=sp.INK,
                    fontsize=sp.FS_TICK)  # fmt: skip
        _isotope_label(ax, y, r.nucleus)
    # direct labels of the two classes, over the top bar
    r0 = st.iloc[0]
    # the seven atom types the G-score uses (backbone plus C-beta and H-beta) and all
    # other atoms (review_burkhard 2.2: "backbone" misnamed C-beta and H-beta)
    ax.text(r0.backbone / 2e6, 2 + bh / 2 + 0.08, "scoring atoms", ha="center",
            va="bottom", fontsize=sp.FS_BODY)  # fmt: skip
    ax.text(r0.backbone / 1e6 + r0.side_chain / 2e6, 2 + bh / 2 + 0.08, "other atoms",
            ha="center", va="bottom", fontsize=sp.FS_BODY)  # fmt: skip
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.5, 2.85)
    xmax = float(np.ceil(st.total.max() / 1e6))
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel("Shifts (millions)")
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
    ax.axvspan(1.5, xhi, color=SHADE, lw=0, zorder=0)
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
    # the region and its two uses, in the empty upper right
    ax.text(0.985, 0.97,
            f"Measured in ≥2 entries:\n{stats['n_seqs_multi_entry']:,} sequences, "
            f"{stats['n_chains_multi_entry']:,} chains\n"
            "→ leakage risk (Fig. 5)\n→ replicates (Figs. 3b, 4)",
            transform=ax.transAxes, ha="right", va="top", fontsize=sp.FS_BODY,
            linespacing=1.15)  # fmt: skip
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
    draw_c(fig, (0.0, row2, sp.W_SINGLE, ROW_H), st)
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
    rh.hist.to_csv(data_out / "fig2d_repeats.csv", index=False)
    rh.top.to_csv(data_out / "fig2d_top_sequences.csv", index=False)
    (data_out / "fig2_summary.json").write_text(json.dumps(summ, indent=1, default=int))
    print(json.dumps({k: v for k, v in summ.items() if k != "repeat_stats"}, indent=1,
                     default=int))  # fmt: skip
    return summ


if __name__ == "__main__":
    main()
