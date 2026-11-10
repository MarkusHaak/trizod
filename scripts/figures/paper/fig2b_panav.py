#!/usr/bin/env python3
"""Fig. 2b -- independent confirmation of the re-referencing offsets by PANAV.

Scatter of the TriZOD total offset (LACS + residual POTENCI/AIC correction, ppm;
``total_off_<atom>_ppm``) against the negated PANAV correction for CA, CB, C and
N, with the Pearson r per nucleus. The C and N panels are shown, not hidden.
A side panel gives the share of all released chains whose LACS offset is at
least 0.5 ppm, per nucleus (CA and CB share one 13C offset in LACS).

Matching PANAV to TriZOD chains (the writing plan asks for ``st_id``, not
entry-level matching):

* ``data/external/panav_offsets.json`` keys each PANAV result by the 0-based
  index of the ``_Atom_chem_shift`` loop in the entry's NMR-STAR file, in
  ``pynmrstar`` order (``scripts/fetch_panav_bmrb.py``: ``enumerate(entry.
  get_loops_by_category("atom_chem_shift"))``). That index is NOT the shift
  list ID: PANAV skips loops it cannot fit, so an entry can carry keys
  ``1, 2, 3`` only.
* This script re-parses each raw file in the same order and maps every loop
  index to the parent saveframe's ``_Assigned_chem_shift_list.ID``, which is
  TriZOD's ``st_id`` (``trizod/bmrb/bmrb.py`` ``ShiftTable.id``). The mapping
  is cached under ``<code checkout>/tmp/paper_figs/`` (``style_paper.TMP``).
* Each PANAV shift list is then matched to the scored chains with the same
  ``(entry_id, st_id)``. PANAV fits one offset per shift list; when a list
  holds several scored chains (multi-chain assemblies), the chain with the most
  backbone shifts (``total_bbshifts``) represents it, so every PANAV list
  contributes exactly one point.

For comparison the script also prints the plan's entry-level preview (entries
with exactly one PANAV list, matched to their first scored chain), which the
``\\rPanav*`` macros in ``numbers.tex`` still carry.

Two Pearson r values are drawn per nucleus: over all matched lists (the plan's
statistic) and over the lists inside the plotted window (|offset| <= 4 ppm for
13C, 6 ppm for 15N). The all-list r is dominated by a handful of gross
deposition errors -- e.g. carbonyl shifts deposited near 0-55 ppm or amide N
near 200 ppm -- that PANAV absorbs into a correction of 70-176 ppm while TriZOD
leaves the offset at zero (LACS finds no fit). They are counted and printed
(``gross``: one method > 20 ppm, the other 0) and drawn pinned to the window
edge as open markers.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig2b_panav.py \
        --out-dir <draft>/Figures/paper
Outputs: <out-dir>/fig2b.pdf and <out-dir>/source_data/fig2b*.csv.
"""

from __future__ import annotations

import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402

from trizod.offsets import lacs_off_ppm_col, total_off_ppm_col  # noqa: E402

ATOMS = ["CA", "CB", "C", "N"]
PANAV_KEY = {"CA": "CA", "CB": "CB", "C": "CO", "N": "N"}
LIMS = {"CA": 4.0, "CB": 4.0, "C": 4.0, "N": 6.0}  # plotted window (ppm, +/-)
LACS_HALF_PPM = 0.5
CACHE = sp.TMP / "paper_figs" / "panav_loop_to_st_id.json"
FS_SMALL = 5.0


# --------------------------------------------------------------------------------
# PANAV loop index -> st_id
# --------------------------------------------------------------------------------
def _loop_st_ids(entry_id: str) -> tuple[str, list | None]:
    """``[st_id of atom_chem_shift loop 0, loop 1, ...]`` in pynmrstar order."""
    import pynmrstar

    d = sp.RAW_BMRB / f"bmr{entry_id}"
    files = sorted(d.glob("*.str"))
    if not files:
        return entry_id, None
    try:
        ent = pynmrstar.Entry.from_file(str(files[0]))
    except Exception:  # noqa: BLE001 -- unreadable file: no mapping
        return entry_id, None
    out = []
    # Same traversal as Entry.get_loops_by_category: frames in order, loops in order.
    for sf in ent.frame_list:
        for lp in sf.loops:
            if lp.category.lower() != "_atom_chem_shift":
                continue
            sid = sf.get_tag("ID")
            col = sorted(set(lp.get_tag("Assigned_chem_shift_list_ID")) - {".", "?"})
            st = sid[0] if sid else (col[0] if col else None)
            # the loop column must agree with the saveframe ID when both exist
            if sid and col and col != [sid[0]]:
                st = f"MISMATCH:{sid[0]}|{','.join(col)}"
            out.append(st)
    return entry_id, out


def loop_to_st_map(pan: dict, workers: int) -> dict:
    """Cached ``{entry_id: [st_id per loop index]}`` for every entry PANAV fitted."""
    want = sorted(e for e, v in pan.items() if v)
    cache = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())
    todo = [e for e in want if e not in cache]
    if todo:
        print(f"mapping PANAV loop indices to st_id for {len(todo):,} entries ...")
        with get_context("spawn").Pool(workers) as pool:
            for i, (e, m) in enumerate(pool.imap_unordered(_loop_st_ids, todo, 32)):
                cache[e] = m
                if (i + 1) % 1000 == 0:
                    print(f"  {i + 1:,}/{len(todo):,}")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, sort_keys=True))
    return {e: cache[e] for e in want}


# --------------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------------
def match_st(pan: dict, stmap: dict, df):
    """One row per PANAV shift list matched to a scored chain by (entry, st_id)."""
    import pandas as pd

    by = {k: g for k, g in df.groupby(["entry_id", "st_id"])}  # noqa: C416 (GroupBy.keys is a list)
    rows, stats = (
        [],
        {"lists": 0, "unmapped": 0, "unscored": 0, "multi": 0, "mismatch": 0},
    )
    for e, v in pan.items():
        if not v:
            continue
        loops = stmap.get(e)
        for idx, sl in v["shift_lists"].items():
            stats["lists"] += 1
            i = int(idx)
            if loops is None or i >= len(loops) or loops[i] is None:
                stats["unmapped"] += 1
                continue
            st = loops[i]
            if st.startswith("MISMATCH"):
                stats["mismatch"] += 1
                continue
            g = by.get((e, st))
            if g is None:
                stats["unscored"] += 1
                continue
            if len(g) > 1:
                stats["multi"] += 1
                g = g.sort_values(["total_bbshifts", "id"], ascending=[False, True])
            r = g.iloc[0]
            row = {
                "entry_id": e,
                "panav_loop": i,
                "st_id": st,
                "chain": r.id,
                "n_chains_in_list": len(g),
            }
            for a in ATOMS:
                p = sl["offsets"].get(PANAV_KEY[a])
                row[f"neg_panav_{a}"] = np.nan if p is None else -float(p)
                row[f"trizod_total_{a}"] = r[total_off_ppm_col(a)]
                row[f"trizod_lacs_{a}"] = r[lacs_off_ppm_col(a)]
            rows.append(row)
    return pd.DataFrame(rows), stats


def match_entry_level(pan: dict, df):
    """The plan's preview: entries with one PANAV list, first scored chain."""
    first = df.drop_duplicates("entry_id", keep="first").set_index("entry_id")
    out = {a: ([], []) for a in ATOMS}
    n = 0
    for e, v in pan.items():
        if not v or len(v["shift_lists"]) != 1 or e not in first.index:
            continue
        n += 1
        off = next(iter(v["shift_lists"].values()))["offsets"]
        r = first.loc[e]
        for a in ATOMS:
            p = off.get(PANAV_KEY[a])
            if p is None:
                continue
            out[a][0].append(r[total_off_ppm_col(a)])
            out[a][1].append(-p)
    r = {}
    for a in ATOMS:
        x, y = np.array(out[a][0], float), np.array(out[a][1], float)
        r[a] = float(np.corrcoef(x, y)[0, 1])
    return n, r


def pearson(x, y) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1])


# --------------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------------
NUC_LABEL = {"CA": "Cα", "CB": "Cβ", "C": "C′", "N": "N"}


def plot(M, share: dict, n_chains: int) -> None:
    import matplotlib.gridspec as gridspec

    fig = sp.figure(112.0, 74.0)
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1, 1, 0.78],
        left=0.105,
        right=0.985,
        bottom=0.115,
        top=0.93,
        wspace=0.42,
        hspace=0.42,
    )
    col = sp.OKABE_ITO["blue"]
    for j, a in enumerate(ATOMS):
        ax = fig.add_subplot(gs[j // 2, j % 2])
        x = M[f"neg_panav_{a}"].to_numpy(float)
        y = M[f"trizod_total_{a}"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        L = LIMS[a]
        out = (np.abs(x) > L) | (np.abs(y) > L)
        ax.plot([-L, L], [-L, L], color="0.6", lw=0.5, ls="--", zorder=0)
        ax.axhline(0, color="0.85", lw=0.4, zorder=0)
        ax.axvline(0, color="0.85", lw=0.4, zorder=0)
        ax.scatter(
            x[~out], y[~out], s=1.2, lw=0, color=col, alpha=0.35, rasterized=False
        )
        # points beyond the window, pinned to its edge as open markers
        ax.scatter(
            np.clip(x[out], -L, L),
            np.clip(y[out], -L, L),
            s=4,
            marker="o",
            facecolors="none",
            edgecolors=sp.OKABE_ITO["vermillion"],
            lw=0.4,
        )
        ax.set_xlim(-L * 1.04, L * 1.04)
        ax.set_ylim(-L * 1.04, L * 1.04)
        ax.set_aspect("equal")
        t = 2 if L <= 4 else 3
        ax.set_xticks(np.arange(-L, L + 0.1, t))
        ax.set_yticks(np.arange(-L, L + 0.1, t))
        r = pearson(x, y)
        r_win = pearson(x[~out], y[~out])
        ax.text(
            0.04,
            0.97,
            NUC_LABEL[a],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FS_SMALL + 1,
            fontweight="bold",
        )
        ax.text(
            0.04,
            0.85,
            f"r = {r:.2f} (all {len(x):,})\nr = {r_win:.2f} (in window)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FS_SMALL,
            linespacing=1.15,
        )
        if out.any():
            ax.text(
                0.97,
                0.03,
                f"{out.sum()} outside",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=FS_SMALL,
                color=sp.OKABE_ITO["vermillion"],
            )
        if j % 2 == 0:
            ax.set_ylabel("TriZOD offset (ppm)")
        if j // 2 == 1:
            ax.set_xlabel("−PANAV correction (ppm)")
    # share of chains with |LACS| >= 0.5 ppm per nucleus
    axb = fig.add_subplot(gs[:, 2])
    labs = ["Cα/Cβ", "C′", "N", "any"]
    keys = ["CA", "C", "N", "any"]
    vals = [100 * share[k] for k in keys]
    yy = np.arange(len(labs))[::-1]
    axb.barh(yy, vals, height=0.62, color=[sp.OKABE_ITO["blue"]] * 3 + ["0.45"])
    for yi, v in zip(yy, vals):
        axb.text(v + 1.5, yi, f"{v:.1f}%", va="center", ha="left", fontsize=FS_SMALL)
    axb.set_yticks(yy, labs)
    axb.set_xlim(0, 60)
    axb.set_xticks([0, 20, 40, 60])
    axb.set_xlabel("Chains with\n|LACS| ≥ 0.5 ppm (%)")
    axb.set_title(f"All {n_chains:,} chains", loc="left", fontsize=FS_SMALL + 0.5)
    axb.set_ylim(-0.6, len(labs) - 0.4)
    # vertically centre the bar panel against the scatter grid
    pos = axb.get_position()
    axb.set_position(
        [pos.x0 + 0.03, pos.y0 + pos.height * 0.25, pos.width - 0.03, pos.height * 0.5]
    )
    sp.fig_label(fig, "b", 0.0, 0.0)
    sp.save(fig, "fig2b")


def main() -> None:
    ap = sp.cli(__doc__)
    ap.add_argument("--workers", type=int, default=6)
    args = sp.parse_args(ap)

    pan = json.loads(sp.PANAV.read_text())
    stmap = loop_to_st_map(pan, args.workers)
    df = sp.load_main(
        ["id", "entry_id", "st_id", "total_bbshifts"]
        + [total_off_ppm_col(a) for a in ATOMS]
        + [lacs_off_ppm_col(a) for a in ("C", "CA", "CB", "H", "HA", "HB", "N")]
    )
    df["entry_id"] = df.entry_id.astype(str)
    df["st_id"] = df.st_id.astype(str)

    M, st = match_st(pan, stmap, df)
    n_el, r_el = match_entry_level(pan, df)
    print(f"PANAV entries with offsets: {sum(1 for v in pan.values() if v):,}")
    print(
        f"PANAV shift lists: {st['lists']:,}; unmapped {st['unmapped']}, "
        f"loop/saveframe ID mismatch {st['mismatch']}, not scored {st['unscored']}, "
        f"matched {len(M):,} (lists holding >1 scored chain: {st['multi']})"
    )
    print(f"matched entries: {M.entry_id.nunique():,}")
    print("Pearson r, st_id matching (all matched lists):")
    for a in ATOMS:
        x = M[f"neg_panav_{a}"].to_numpy(float)
        y = M[f"trizod_total_{a}"].to_numpy(float)
        yl = M[f"trizod_lacs_{a}"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        win = m & (np.abs(x) <= LIMS[a]) & (np.abs(y) <= LIMS[a])
        from scipy.stats import spearmanr

        rho = spearmanr(x[m], y[m]).statistic
        print(
            f"  {a:2s}: r {pearson(x, y):.3f} (n {m.sum():,}); LACS only "
            f"{pearson(x, yl):.3f}; inside +/-{LIMS[a]:.0f} ppm r "
            f"{pearson(x[win], y[win]):.3f} (n {win.sum():,}); Spearman {rho:.3f}"
        )
    for a in ATOMS:
        x = M[f"neg_panav_{a}"].to_numpy(float)
        y = M[f"trizod_total_{a}"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        out = m & ((np.abs(x) > LIMS[a]) | (np.abs(y) > LIMS[a]))
        gross = m & (((np.abs(x) > 20) & (y == 0)) | ((np.abs(y) > 20) & (x == 0)))
        both_big = m & (np.abs(x) > 20) & (np.abs(y) > 20)
        print(
            f"  {a:2s}: outside window {out.sum()}; gross (one > 20 ppm, other 0) "
            f"{gross.sum()}; both > 20 ppm {both_big.sum()}; TriZOD offset == 0 "
            f"{int((m & (y == 0)).sum()):,}; PANAV == 0 {int((m & (x == 0)).sum()):,}"
        )
    print(
        f"entry-level preview (plan): n {n_el:,}; "
        + ", ".join(f"{a} {r_el[a]:.3f}" for a in ATOMS)
    )

    # share of all released chains with |LACS| >= 0.5 ppm
    n_chains = len(df)
    lac = {
        a: df[lacs_off_ppm_col(a)].abs()
        for a in ("C", "CA", "CB", "H", "HA", "HB", "N")
    }
    share = {a: float((lac[a] >= LACS_HALF_PPM).sum()) / n_chains for a in lac}
    anyv = np.zeros(n_chains, bool)
    for a in lac:
        anyv |= (lac[a] >= LACS_HALF_PPM).to_numpy()
    share["any"] = float(anyv.sum()) / n_chains
    print(
        "share of chains with |LACS| >= 0.5 ppm: "
        + ", ".join(
            f"{k} {100 * v:.1f}% ({round(v * n_chains):,})" for k, v in share.items()
        )
    )

    sd = sp.out_dir("source_data")
    sd.mkdir(parents=True, exist_ok=True)
    M.to_csv(sd / "fig2b_panav_matched.csv", index=False)
    with open(sd / "fig2b_lacs_share.csv", "w") as fh:
        fh.write("nucleus,share,n_chains_at_least_0p5ppm,n_chains_total\n")
        for k, v in share.items():
            fh.write(f"{k},{v:.6f},{round(v * n_chains)},{n_chains}\n")
    plot(M, share, n_chains)
    print(f"wrote {sp.out_dir() / 'fig2b.pdf'}")


if __name__ == "__main__":
    main()
