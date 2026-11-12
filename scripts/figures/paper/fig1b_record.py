#!/usr/bin/env python3
"""Fig. 1b -- anatomy of one chain record: four residue tracks and a key-value card.

One chain of the release, drawn the way a model sees it. Four tracks share one residue
axis:

1. **Backbone shifts** -- which of the seven backbone atom types (H, N, Cα, Cβ, C′, Hα,
   Hβ) carry a deposited shift at each residue (shift table, ``is_backbone`` rows;
   HA2/HA3 count as Hα and HB/HB1/HB2/HB3 as Hβ);
2. **Side-chain shifts** -- the number of deposited side-chain shifts per residue
   (shift table, the other rows): the release keeps them, although the label does not
   use them;
3. **Shift count k** -- the backbone shifts in the residue's three-residue scoring
   window (chain table ``k``), coloured on the shared k scale (sp.K_CMAP);
4. **G-score** -- the released label (chain table ``gscores``), with the binary cut
   G = 0.65; residues the release does not score (``mask`` false) are marked ×.

A card beside the tracks lists the chain's metadata from the chain table (pH,
temperature, ionic strength, pressure, physical state, bound, cosolvent, membrane
mimetic, tier, split), and a two-table icon shows how the chain table and the shift
table join on ``id``.

**The example is chosen by a fixed rule** (story blueprint 6.1; :func:`select_example`),
never by hand:

* a strict-tier training representative (``split`` = train, ``train_tier`` = strict) of
  60-150 residues;
* at least 95 % of its residues scored (``mask``);
* 15-60 % of its scored residues with G > 0.65, and a run of at least 10 consecutive
  scored residues with G > 0.65 (an unscored residue ends a run);
* physical state native, not bound (``is_bound`` false);
* among these, the most backbone shift types (``bbshift_types_post``), then the longest
  disordered run, then the smallest ``id``.

The strict tier accepts no imputed pH, temperature or ionic strength, so the card shows
deposited values only; the script checks this against the raw sample-condition lists
(``paper_raw_scan.chain_conditions``) when the raw-scan cache exists.

Inputs (read-only): the release Parquet (chain table) and the shift Parquet, filtered on
the chosen ``id``; the raw-scan cache of ``paper_raw_scan.py`` (optional check).

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig1b_record.py \
        --out-dir <draft>/Figures/paper
Outputs: ``<out-dir>/fig1b.pdf`` (vector, ``FIG_W`` x ``FIG_H`` mm, placed at 1:1 in the
second row of Fig. 1 beside ``fig1c.pdf``); ``<out-dir>/source_data/fig1b.csv`` (one row
per residue of the chosen chain, with its ``id``); ``<out-dir>/source_data/fig1b_rule.csv``
(every chain that meets the rule, ranked, the chosen one flagged). The chosen ``id`` and
the number of qualifying chains are printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402

# ----------------------------------------------------------------------------------
# the selection rule (blueprint 6.1)
# ----------------------------------------------------------------------------------
RULE = {
    "tier": "strict",
    "len_min": 60,
    "len_max": 150,
    "min_scored_frac": 0.95,
    "dis_frac_min": 0.15,
    "dis_frac_max": 0.60,
    "min_run": 10,
    "physical_state": "native",
}
RULE_COLUMNS = [
    "id",
    "entity_name",
    "sequence",
    "length",
    "gscores",
    "mask",
    "split",
    "train_tier",
    "physical_state",
    "is_bound",
    "bbshift_types_post",
]

# backbone atom types of the scorer, in the order of the dot matrix (top to bottom)
BB_ROWS = [
    ("H", ("H",)),
    ("N", ("N",)),
    ("Cα", ("CA",)),
    ("Cβ", ("CB",)),
    ("C′", ("C",)),
    ("Hα", ("HA", "HA2", "HA3")),
    ("Hβ", ("HB", "HB1", "HB2", "HB3")),
]

# Plain-language protein names for the card. The Parquet's entity_name is the depositor's
# label (e.g. "MenV_P_267_388"); a chain without an entry here shows its entity_name.
PLAIN_NAMES = {
    "27634_1_1_1": ("Menangle virus phosphoprotein,", "C-terminal region"),
}


def longest_run(g: np.ndarray, m: np.ndarray, cut: float = sp.G_DIS) -> tuple[int, int]:
    """Longest run of consecutive scored residues with G > ``cut``: (length, 0-based start)."""
    best, best_start, cur, start = 0, -1, 0, 0
    for i, (gi, mi) in enumerate(zip(g, m)):
        if mi and np.isfinite(gi) and gi > cut:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, best_start = cur, start
        else:
            cur = 0
    return best, best_start


def select_example(df: pd.DataFrame | None = None) -> tuple[str, pd.DataFrame]:
    """Apply the Fig. 1b rule; return the chosen ``id`` and every qualifying chain, ranked.

    ``make_numbers.py`` can import this to quote the chosen chain and the number of
    qualifying chains (``len(table)``).
    """
    if df is None:
        df = sp.load_main(columns=RULE_COLUMNS)
    c = df[
        (df.split == "train")
        & (df.train_tier == RULE["tier"])
        & df.length.between(RULE["len_min"], RULE["len_max"])
        & (df.physical_state == RULE["physical_state"])
        & (~df.is_bound.astype(bool))
    ]
    rows = []
    for _, r in c.iterrows():
        if r["gscores"] is None or r["mask"] is None:
            continue
        g = sp.arr(r["gscores"])
        m = np.asarray(r["mask"], bool)
        n_scored = int(m.sum())
        if n_scored == 0:
            continue
        dis = float((g[m] > sp.G_DIS).mean())
        run, start = longest_run(g, m)
        rows.append(
            {
                "id": r["id"],
                "entity_name": r["entity_name"],
                "length": int(r["length"]),
                "n_scored": n_scored,
                "scored_frac": n_scored / int(r["length"]),
                "disordered_frac": dis,
                "longest_disordered_run": run,
                "run_first_residue": start + 1,
                "bbshift_types_post": int(r["bbshift_types_post"]),
            }
        )
    t = pd.DataFrame(rows)
    t = t[
        (t.scored_frac >= RULE["min_scored_frac"])
        & t.disordered_frac.between(RULE["dis_frac_min"], RULE["dis_frac_max"])
        & (t.longest_disordered_run >= RULE["min_run"])
    ]
    t = t.sort_values(
        ["bbshift_types_post", "longest_disordered_run", "id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    if t.empty:
        raise SystemExit("no chain meets the Fig. 1b rule")
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    t["chosen"] = t["rank"] == 1
    return str(t.id.iloc[0]), t


# ----------------------------------------------------------------------------------
# the record of one chain
# ----------------------------------------------------------------------------------
def load_record(cid: str) -> tuple[pd.Series, pd.DataFrame]:
    """The chain-table row of ``cid`` and its per-residue table (tracks 1-4)."""
    import pyarrow.parquet as pq

    row = pq.read_table(sp.PQ_MAIN, filters=[("id", "==", cid)]).to_pandas().iloc[0]
    sh = pq.read_table(
        sp.PQ_SHIFTS,
        columns=["id", "seq_id", "atom_id", "is_backbone"],
        filters=[("id", "==", cid)],
    ).to_pandas()
    L = int(row["length"])
    res = pd.DataFrame(
        {
            "id": cid,
            "residue": np.arange(1, L + 1),
            "aa": list(row["sequence"]),
        }
    )
    bb = sh[sh.is_backbone.astype(bool)]
    for _, atoms in BB_ROWS:
        have = set(bb.seq_id[bb.atom_id.isin(atoms)].astype(int))
        res[f"bb_{atoms[0]}"] = [int(i in have) for i in res.residue]
    sc = sh[~sh.is_backbone.astype(bool)].groupby("seq_id").size()
    res["n_sidechain_shifts"] = [int(sc.get(i, 0)) for i in res.residue]
    res["k"] = np.asarray(row["k"], int)
    res["gscore"] = sp.arr(row["gscores"])
    res["mask"] = np.asarray(row["mask"], bool)
    res.loc[~res["mask"], "gscore"] = np.nan  # the release scores only masked residues
    return row, res


def deposited_check(row: pd.Series) -> str:
    """Check that pH, temperature, ionic strength and pressure are in the raw condition list."""
    try:
        import paper_raw_scan as prs
    except ImportError:  # pragma: no cover
        return "raw-scan module not importable; check skipped"
    if not prs.CACHE.exists():
        return f"raw-scan cache {prs.CACHE} missing; check skipped"
    raw = prs.scan_raw()
    c = prs.chain_conditions(raw, row["entry_id"], row["st_id"])
    if c is None:
        return "no raw condition list resolved for this chain"
    missing = [k for k in ("ph", "temp", "ionic", "pressure_raw") if c.get(k) is None]
    return (
        "all deposited"
        if not missing
        else f"NOT deposited (imputed or absent in the release): {missing}"
    ) + (
        f" (raw: pH {c.get('ph')}, T {c.get('temp')}, ionic {c.get('ionic')}, "
        f"pressure {c.get('pressure_raw')})"
    )


def card_rows(row: pd.Series) -> list[tuple]:
    """``(key, value, marker)`` rows of the card; ``None`` separates groups."""

    def num(v, fmt):
        return "not deposited" if v is None or not np.isfinite(v) else fmt.format(v)

    mem = row["membrane_mimetic"]
    state = row["physical_state"]
    return [
        ("pH", num(row["ph"], "{:.1f}"), None),
        ("Temperature", num(row["temperature"], "{:.0f} K"), None),
        ("Ionic strength", num(row["ionic_strength"], "{:.3f} M"), None),
        ("Pressure", num(row["pressure_bar"], "{:.1f} bar"), None),
        None,
        ("Physical state", state if isinstance(state, str) else "not stated", None),
        ("Bound", "yes" if bool(row["is_bound"]) else "no", None),
        ("Cosolvent", "yes" if bool(row["cosolvent_evidence"]) else "none", None),
        ("Membrane mimetic", mem if isinstance(mem, str) and mem else "none", None),
        None,
        ("Tier", str(row["train_tier"]), "tier"),
        ("Split", {"train": "training"}.get(row["split"], str(row["split"])), "split"),
    ]


# ----------------------------------------------------------------------------------
# plot
# ----------------------------------------------------------------------------------
# Fig. 1 layout (story blueprint 10.1): a 160-mm-wide flow (panel a, TikZ) above, and in
# the second row this panel beside panel c (fig1c.pdf), 3 mm apart, both placed at 1:1.
FIG_W, FIG_H = 105.0, 66.0
X0, X1 = 7.0, 61.0  # track area, mm from the left edge
CARD_X0, CARD_X1 = 64.0, 105.0  # the card
# tracks: (top, height) in mm from the top edge; each header's baseline sits 0.9 mm above
# its track, so the first header shares the baseline of the panel letter (2.5 mm)
TRACKS = {
    "bb": (3.4, 17.6),
    "sc": (25.5, 6.2),
    "k": (36.2, 6.2),
    "g": (46.9, 12.2),
}
# the residue axis, 59.1 mm from the top; panel c puts its count axis on the same line
AXIS_BOTTOM = TRACKS["g"][0] + TRACKS["g"][1]
LETTER_BASELINE = 2.5  # mm from the top, as the letters of panels a and c
HEADERS = {
    "bb": "Scoring-atom shifts",  # the seven atom types the G-score uses (Cβ, Hβ too)
    "sc": "Side-chain shifts",
    "k": r"Shift count $\mathit{k}$",
    "g": "G-score",
}
CELL_ON = sp.INK_SOFT  # a deposited backbone shift
CELL_OFF = "#EFEFEF"  # the row background: no shift (a neutral, not an entity colour)
BAR = sp.INK_SOFT
# white with a light outline (a neutral), as the boxes of panel a (fig1a_overview.tex)
CARD_FILL, CARD_LINE = "white", sp.CAT["light"]
PITCH = 3.2  # mm between card lines (7 pt text, 1.3 x leading)
# G-score track: unscored residues get a cross in a strip below G = 0, between the zero
# line and the residue axis, so a cross never reads as a value (figure review item 15)
UNSCORED_STRIP = -0.17  # bottom of the G axes (G units); the y spine starts at 0
UNSCORED_Y = -0.085  # centre of the strip


def _fx(x_mm: float) -> float:
    return x_mm / FIG_W


def _fy(y_mm_from_top: float) -> float:
    return 1 - y_mm_from_top / FIG_H


def _text(fig, x_mm, y_mm, s, color=sp.INK, **kw):
    """7 pt text with its baseline ``y_mm`` below the top edge."""
    kw.setdefault("fontsize", sp.FS_BODY)
    kw.setdefault("va", "baseline")
    return fig.text(_fx(x_mm), _fy(y_mm), s, color=color, **kw)


def _axes(fig, top_mm: float, h_mm: float):
    return fig.add_axes([_fx(X0), _fy(top_mm + h_mm), (X1 - X0) / FIG_W, h_mm / FIG_H])


def _residue_axis(ax, L: int, labels: bool) -> None:
    ax.set_xlim(0.5, L + 0.5)
    ax.set_xticks([t for t in (1, 20, 40, 60, 80, 100, 120, 140) if t <= L])
    if not labels:
        ax.tick_params(axis="x", length=0, labelbottom=False)


def draw_tracks(fig, res: pd.DataFrame) -> None:
    from matplotlib.patches import Rectangle

    L = len(res)
    x = res.residue.to_numpy()
    for key, (top, _) in TRACKS.items():
        _text(fig, X0, top - 0.9, HEADERS[key])

    # 1. backbone shifts: one row per atom type, a dark cell where a shift is deposited
    top, h = TRACKS["bb"]
    ax = _axes(fig, top, h)
    n = len(BB_ROWS)
    ax.set_ylim(n - 0.5, -0.5)
    for j, (_, atoms) in enumerate(BB_ROWS):
        ax.add_patch(
            Rectangle((0.5, j - 0.36), L, 0.72, facecolor=CELL_OFF, lw=0, zorder=1)
        )
        for xi in x[res[f"bb_{atoms[0]}"].to_numpy() == 1]:
            ax.add_patch(
                Rectangle(
                    (xi - 0.42, j - 0.36), 0.84, 0.72, facecolor=CELL_ON, lw=0, zorder=2
                )
            )
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for lab, _ in BB_ROWS])
    ax.tick_params(axis="y", length=0, pad=1.5)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    _residue_axis(ax, L, labels=False)

    # 2. side-chain shifts per residue (bars from zero)
    top, h = TRACKS["sc"]
    ax = _axes(fig, top, h)
    v = res.n_sidechain_shifts.to_numpy()
    ax.bar(x, v, width=0.8, color=BAR, lw=0, zorder=2)
    ax.set_ylim(0, max(10, int(v.max())))
    ax.set_yticks([0, 10])
    _residue_axis(ax, L, labels=False)

    # 3. shift count k: bar height and the shared k scale (sp.K_CMAP)
    top, h = TRACKS["k"]
    ax = _axes(fig, top, h)
    kk = res.k.to_numpy()
    ax.bar(x, kk, width=0.8, color=[sp.k_color(k) for k in kk], lw=0, zorder=2)
    ax.set_ylim(0, sp.K_RANGE[1])
    ax.set_yticks([0, sp.K_RANGE[1]])
    _residue_axis(ax, L, labels=False)

    # 4. G-score with the 0.65 cut; the line breaks at unscored residues, marked x
    top, h = TRACKS["g"]
    ax = _axes(fig, top, h)
    g = res.gscore.to_numpy()
    ax.axhline(sp.G_DIS, color=sp.CAT["light"], lw=0.6, ls=(0, (2.5, 1.5)), zorder=1)
    ax.plot(x, g, color=sp.INK, lw=1.0, zorder=3, solid_joinstyle="round")
    un = x[~res["mask"].to_numpy()]
    if len(un):
        # in the strip below G = 0, outside the data range
        ax.plot(
            un,
            np.full(len(un), UNSCORED_Y),
            ls="",
            marker="x",
            ms=3.0,
            mew=0.8,
            color=sp.INK_SOFT,
            zorder=4,
            clip_on=False,
        )
    ax.set_ylim(UNSCORED_STRIP, 1)
    ax.spines["left"].set_bounds(0, 1)
    ax.set_yticks([0, sp.G_DIS, 1])
    ax.set_yticklabels(["0", f"{sp.G_DIS:g}", "1"])
    _residue_axis(ax, L, labels=True)
    ax.set_xlabel("Residue", labelpad=1.0)
    # the two words go on the side of the cut that the line leaves free at the right end
    right = np.nanmedian(g[-15:])
    y_dis, y_ord = (0.79, 0.51) if right < sp.G_DIS else (0.51, 0.25)
    for y, word in ((y_dis, "disordered"), (y_ord, "ordered")):
        ax.text(L - 1, y, word, ha="right", va="center", color=sp.INK_SOFT)


def draw_card(
    fig, row: pd.Series, top: float = 0.6, bottom: float = AXIS_BOTTOM
) -> None:
    """Key-value card (chain table) and the two-table icon, between ``top`` and ``bottom``."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fig.add_artist(
        FancyBboxPatch(
            (_fx(CARD_X0 + 0.3), _fy(bottom)),
            (CARD_X1 - CARD_X0 - 0.6) / FIG_W,
            (bottom - top) / FIG_H,
            boxstyle="round,pad=0,rounding_size=0.012",
            transform=fig.transFigure,
            facecolor=CARD_FILL,
            edgecolor=CARD_LINE,
            lw=0.4,
            zorder=0,
        )
    )
    kx = CARD_X0 + 2.0  # keys
    vx = CARD_X0 + 25.0  # values ("Membrane mimetic" is 21.3 mm at 7 pt)
    cid = str(row["id"])
    yy = LETTER_BASELINE + 0.4
    _text(fig, kx, yy, "id", color=sp.INK_SOFT)
    _text(fig, kx + 3.6, yy, cid, fontweight="bold")
    for part in PLAIN_NAMES.get(cid, (str(row["entity_name"]),)):
        yy += PITCH
        _text(fig, kx, yy, part)
    yy += PITCH
    n_scored = int(np.asarray(row["mask"], bool).sum())
    _text(
        fig,
        kx,
        yy,
        f"{int(row['length'])} residues, {n_scored} scored",
        color=sp.INK_SOFT,
    )
    yy += 1.3
    fig.add_artist(
        Line2D(
            [_fx(kx), _fx(CARD_X1 - 2.0)],
            [_fy(yy), _fy(yy)],
            transform=fig.transFigure,
            color=CARD_LINE,
            lw=0.5,
        )
    )
    yy += PITCH - 0.2
    for r in card_rows(row):
        if r is None:
            yy += 1.0
            continue
        key, val, mark = r
        _text(fig, kx, yy, key, color=sp.INK_SOFT)
        x = vx
        if mark == "tier":
            fig.add_artist(
                Rectangle(
                    (_fx(x), _fy(yy)),
                    2.0 / FIG_W,
                    2.0 / FIG_H,
                    transform=fig.transFigure,
                    facecolor=sp.TIER_COLORS.get(val, sp.INK),
                    lw=0,
                )
            )
            x += 2.8
        elif mark == "split":
            fig.add_artist(
                Line2D(
                    [_fx(x + 1.0)],
                    [_fy(yy - 0.9)],
                    transform=fig.transFigure,
                    **sp.marker_kw("train", ms=4.0),
                )
            )
            x += 2.8
        _text(fig, x, yy, val)
        yy += PITCH
    draw_tables(fig, kx, bottom - 8.2)


def _table_glyph(fig, x_mm: float, top_mm: float, w: float, h: float) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    head = h * 0.3
    kw = {"transform": fig.transFigure}
    fig.add_artist(
        Rectangle(
            (_fx(x_mm), _fy(top_mm + h)),
            w / FIG_W,
            h / FIG_H,
            facecolor="white",
            edgecolor=sp.INK_SOFT,
            lw=0.5,
            **kw,
        )
    )
    fig.add_artist(
        Rectangle(
            (_fx(x_mm), _fy(top_mm + head)),
            w / FIG_W,
            head / FIG_H,
            facecolor="#BFBFBF",
            edgecolor=sp.INK_SOFT,
            lw=0.5,
            **kw,
        )
    )
    for k in (1, 2):
        yk = top_mm + head + k * (h - head) / 3
        fig.add_artist(
            Line2D(
                [_fx(x_mm), _fx(x_mm + w)],
                [_fy(yk)] * 2,
                color=sp.INK_SOFT,
                lw=0.35,
                **kw,
            )
        )
    fig.add_artist(
        Line2D(
            [_fx(x_mm + w * 0.34)] * 2,
            [_fy(top_mm), _fy(top_mm + h)],
            color=sp.INK_SOFT,
            lw=0.35,
            **kw,
        )
    )


def draw_tables(fig, x_mm: float, top_mm: float) -> None:
    """The release's two tables side by side, joined on ``id``."""
    from matplotlib.lines import Line2D

    w, h, dx = 5.0, 3.8, 20.0
    for i, name in enumerate(("chain table", "shift table")):
        _table_glyph(fig, x_mm + i * dx, top_mm, w, h)
        _text(fig, x_mm + i * dx, top_mm + h + 2.9, name)
    ym = top_mm + h / 2
    fig.add_artist(
        Line2D(
            [_fx(x_mm + w + 0.4), _fx(x_mm + dx - 0.4)],
            [_fy(ym)] * 2,
            transform=fig.transFigure,
            color=sp.INK,
            lw=0.7,
        )
    )
    _text(fig, x_mm + (w + dx) / 2, ym - 0.7, "id", ha="center")


def plot(row: pd.Series, res: pd.DataFrame, out_name: str = "fig1b") -> None:
    fig = sp.figure(FIG_W, FIG_H)
    draw_tracks(fig, res)
    draw_card(fig, row)
    _text(fig, 0.0, LETTER_BASELINE, "b", fontsize=sp.FS_PANEL, fontweight="bold")
    sp.save(fig, out_name)


# ----------------------------------------------------------------------------------
def main() -> None:
    ap = sp.cli(__doc__)
    args = sp.parse_args(ap)

    cid, table = select_example()
    print(f"Fig. 1b rule: {len(table)} chains qualify; chosen {cid}")
    print(table.head(8).to_string(index=False))
    row, res = load_record(cid)
    print("deposited conditions:", deposited_check(row))
    for r in card_rows(row):
        if r is not None:
            print(f"  {r[0]:18s} {r[1]}")

    sd = args.out_dir / "source_data"
    sd.mkdir(parents=True, exist_ok=True)
    res.to_csv(sd / "fig1b.csv", index=False)
    table.to_csv(sd / "fig1b_rule.csv", index=False)
    plot(row, res, "fig1b")
    print(
        f"wrote {args.out_dir / 'fig1b.pdf'}, {sd / 'fig1b.csv'}, {sd / 'fig1b_rule.csv'}"
    )


if __name__ == "__main__":
    main()
