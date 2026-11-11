#!/usr/bin/env python3
"""Fig. 2a (Fig. 1a until 11 Nov 2026) -- ubiquitin: the Z-score grows with k, the G-score does not.

Every released chain whose sequence is exactly the 76-residue human ubiquitin
sequence (``UBQ``) is selected from the release Parquet; bound chains
(``is_bound``) are dropped, because a partner changes the measured state. For
each remaining chain the per-residue Z-score (top) and G-score (bottom) are
drawn over residues 1-76. The C-terminal tail (residues 73-76), disordered in
folded ubiquitin, is shaded.

Chain selection by annotation (``select_chains``, rule in ``NATIVE_RULE``): a
chain is *native* unless its sample is perturbed, i.e. the samples behind its
shift list contain a denaturant or alcohol cosolvent (``PERTURBANTS``), the
condition list of its shift list records a pressure of at least 400 bar, or its
deposited physical state is non-native. Native chains are coloured by their
median per-residue shift count k (classes <= 6, 7-15, 16-21; viridis, as
everywhere in the paper); perturbed chains are drawn thin and light grey
behind, without labels. Pressure is not a release column, so it is read from the
raw NMR-STAR file of each entry, following the chain's own shift list
(``st_id``) to its sample-condition list.

The right-hand strip summarises the folded core (residues 2-71) of the native
chains: the median (bar) and interquartile range (box) of all core residues in
each k class (``core_summary``). ``scripts/manuscript/make_numbers.py`` imports
``select_chains``, ``core_summary`` and ``spearman_core`` from this module for
the ``\\nUbqTeaserChains``, ``\\nUbqClass*``, ``\\zUbqCore*``, ``\\gUbqCore*``,
``\\gUbqTail*`` and ``\\rhoUbq*K`` macros, so text and artwork use one chain set.
The writing plan's earlier selection (mean G-score < 0.5, which keeps the three
high-pressure chains) is printed as a sensitivity check only.

Inputs (read-only): the release Parquet, the per-tier ``scores.json`` IDs of the
release bundle (strictest tier, Source Data only), and ``data/raw/bmrb_entries/``
(pressure and sample components of the ubiquitin entries only).

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig1a_ubiquitin.py \\
        --out-dir <draft>/Figures/paper
Outputs: ``<out-dir>/fig2a_ubiquitin.pdf`` (W_DOUBLE = 160 mm wide, vector, placed at
1:1), ``<out-dir>/source_data/fig2a_ubiquitin.csv`` (one row per chain and residue)
and ``fig2a_ubiquitin_core_summary.csv``; prints the plotted numbers.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402

# italic k in the same sans-serif face as the text (mathtext defaults to DejaVu)
sp.plt.rcParams.update(
    {
        "mathtext.fontset": "custom",
        "mathtext.it": "Arial:italic",
        "mathtext.rm": "Arial",
    }
)

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
CORE = slice(1, 71)  # residues 2-71 (0-based slice), the folded core
TAIL_FIRST, TAIL_LAST = 73, 76  # shaded C-terminal tail (1-based, inclusive)
TEASER_MEAN_G_MAX = 0.5  # writing plan's earlier selection (sensitivity check only)
HIGH_PRESSURE_BAR = 400.0  # same threshold as the writing plan's pressure scan (A12)
# The selection rule in one sentence (Results / caption wording; make_numbers source string).
NATIVE_RULE = (
    "non-bound chains with the exact ubiquitin sequence whose sample contains no denaturant "
    "or alcohol (urea, guanidinium, methanol, ethanol, TFE, HFIP, DMSO), was measured below "
    "400 bar and is deposited in a native physical state"
)
PERTURBANTS = {  # lower-case component-name token -> display name
    "urea": "urea",
    "guanidin": "guanidinium",
    "methanol": "methanol",
    "ethanol": "ethanol",
    "trifluoroethanol": "TFE",
    "tfe": "TFE",
    "hexafluoro": "HFIP",
    "dmso": "DMSO",
    "dimethyl sulfoxide": "DMSO",
}
NON_NATIVE_STATES = {"denatured", "unfolded", "molten globule", "misfolded"}
T_NOTE_RANGE = (
    273.0,
    323.0,
)  # a label names the temperature only outside this range (K)

# plan preview values (writing plan, Fig. 1 table and Appendix A10) for the self-check
PLAN = {
    "chains": 40,
    "entries": 34,
    "bound": 12,
    "nonbound_chains": 28,
    "nonbound_entries": 22,
    "class_n": {"≤6": 11, "7–15": 8, "16–21": 6},
    "core_z": {"≤6": 6.97, "7–15": 10.05, "16–21": 13.91},
    "core_g": {"≤6": 0.087, "7–15": 0.093, "16–21": 0.070},
    "mean_g": {
        "16626_1_1_1": 0.80,
        "15047_1_1_1": 0.79,
        "4375_1_1_1": 0.68,
        "18611_1_1_1": 0.30,
        "18610_1_1_1": 0.14,
        "18610_2_2_1": 0.47,
    },
}


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------
def _to_bar(value: float, unit: str):
    u = unit.lower().strip("'\" ")
    factor = {
        "bar": 1.0,
        "bars": 1.0,
        "atm": 1.01325,
        "mpa": 10.0,
        "kpa": 0.01,
        "pa": 1e-5,
        "mbar": 1e-3,
        "mmhg": 1.01325 / 760,
        "torr": 1.01325 / 760,
        "psi": 0.0689476,
    }.get(u)
    return None if factor is None else value * factor


def sample_record(entry_id: str, st_id: str) -> dict:
    """Pressure (bar) and perturbing components of the sample behind one shift list."""
    import pynmrstar

    path = sp.RAW_BMRB / f"bmr{entry_id}" / f"bmr{entry_id}_3.str"
    ent = pynmrstar.Entry.from_file(str(path))
    shift_lists = [
        s
        for s in ent.get_saveframes_by_category("assigned_chemical_shifts")
        if str(s.get_tag("ID")[0]) == str(st_id)
    ]
    if len(shift_lists) != 1:
        raise RuntimeError(f"{entry_id}: shift list {st_id} not found uniquely")
    sl = shift_lists[0]
    # pressure from the shift list's own sample-condition list
    pressure = None
    cond_id = str(sl.get_tag("Sample_condition_list_ID")[0])
    for cond in ent.get_saveframes_by_category("sample_conditions"):
        if str(cond.get_tag("ID")[0]) != cond_id:
            continue
        loop = cond.get_loop("_Sample_condition_variable")
        for typ, val, unit in loop.get_tag(["Type", "Val", "Val_units"]):
            if "press" not in str(typ).lower():
                continue
            try:
                bar = _to_bar(float(val), str(unit))
            except ValueError:
                bar = None
            if bar is not None:
                pressure = bar if pressure is None else max(pressure, bar)
    # components of the samples measured for this shift list: the shift list's own
    # Sample_IDs, else the samples of its experiments, else every sample of the entry
    sample_ids = set()
    if "_Chem_shift_experiment" in [lp.category for lp in sl.loops]:
        rows = sl.get_loop("_Chem_shift_experiment").get_tag(
            ["Experiment_ID", "Sample_ID"]
        )
        sample_ids = {str(s) for _, s in rows if str(s) not in (".", "?")}
        if not sample_ids:
            exp_ids = {str(e) for e, _ in rows if str(e) not in (".", "?")}
            for el in ent.get_saveframes_by_category("experiment_list"):
                for eid, sid in el.get_loop("_Experiment").get_tag(["ID", "Sample_ID"]):
                    if str(eid) in exp_ids and str(sid) not in (".", "?"):
                        sample_ids.add(str(sid))
    perturbants = []
    for smp in ent.get_saveframes_by_category("sample"):
        if sample_ids and str(smp.get_tag("ID")[0]) not in sample_ids:
            continue
        try:
            loop = smp.get_loop("_Sample_component")
        except KeyError:
            continue
        for name, conc, unit in loop.get_tag(
            ["Mol_common_name", "Concentration_val", "Concentration_val_units"]
        ):
            low = str(name).lower()
            if "mercapto" in low:  # 2-mercaptoethanol is a reductant, not a cosolvent
                continue
            for token, disp in PERTURBANTS.items():
                if token in low:
                    perturbants.append((disp, str(conc), str(unit)))
                    break
    return {"pressure_bar": pressure, "perturbants": sorted(set(perturbants))}


_ID_RE = re.compile(r'^\{"ID":\s*"([^"]+)"')


def tier_ids_fast() -> dict:
    """``{tier: set(IDs)}`` from the bundle's per-tier scores.json, reading only the ID.

    Same result as ``style_paper.load_tier_ids()`` without JSON-parsing ~140 MB of
    per-residue arrays (falls back to ``json`` for a line that does not start with ID).
    """
    ids = {}
    for t in sp.TIERS:
        s = set()
        with open(sp.BUNDLE / "scores" / t / "scores.json") as fh:
            for line in fh:
                if not line.strip():
                    continue
                m = _ID_RE.match(line)
                s.add(m.group(1) if m else json.loads(line)["ID"])
        ids[t] = s
    return ids


COLUMNS = [  # release columns select_chains() and the Source Data need
    "id",
    "entry_id",
    "st_id",
    "sequence",
    "gscores",
    "zscores",
    "k",
    "mask",
    "is_bound",
    "physical_state",
    "temperature",
    "ph",
]


def select_chains(df: pd.DataFrame) -> pd.DataFrame:
    """The chains of Fig. 2a: every non-bound chain of ``df`` with the exact sequence ``UBQ``.

    ``df`` is the release table (one row per chain, e.g. ``style_paper.load_main()`` or
    make_numbers' ``C.df``) with at least the ``COLUMNS`` above. Returns those chains with
    ``median_k`` (over scored residues), ``mean_g``, ``k_class`` (``style_paper.K_CLASSES``),
    ``pressure_bar`` and ``perturbants`` (raw NMR-STAR), ``perturbed`` and ``native``
    (= not perturbed; ``NATIVE_RULE``). ``native`` is the chain set the figure colours and
    summarises, and the one make_numbers.py quotes; both import it from here.
    """
    u = df[(df.sequence == UBQ) & ~df.is_bound.astype(bool)].copy()
    u = u.reset_index(drop=True)
    G = np.vstack([np.asarray(x, float) for x in u.gscores])
    M = np.vstack([np.asarray(x, bool) for x in u["mask"]])
    K = np.vstack([np.asarray(x, float) for x in u.k])
    u["median_k"] = [np.median(k[m]) for k, m in zip(K, M)]
    u["mean_g"] = np.nanmean(G, axis=1)
    bins = [0] + [hi for _, hi in sp.K_CLASSES.values()]  # (0, 6], (6, 15], (15, 21]
    u["k_class"] = pd.cut(u.median_k, bins, labels=list(sp.K_CLASSES)).astype(str)
    rec = [sample_record(str(e), str(s)) for e, s in zip(u.entry_id, u.st_id)]
    u["pressure_bar"] = [r["pressure_bar"] for r in rec]
    u["perturbants"] = [r["perturbants"] for r in rec]
    state = u.physical_state.fillna("").astype(str).str.lower()
    u["perturbed"] = (
        u.perturbants.map(bool)
        | (u.pressure_bar.fillna(0) >= HIGH_PRESSURE_BAR)
        | state.isin(NON_NATIVE_STATES)
    )
    u["native"] = ~u.perturbed
    return u


def load_ubiquitin() -> tuple[pd.DataFrame, dict]:
    df = sp.load_main(columns=COLUMNS)
    all_ubq = df[df.sequence == UBQ]
    counts = {
        "chains": len(all_ubq),
        "entries": all_ubq.entry_id.nunique(),
        "bound": int(all_ubq.is_bound.sum()),
    }
    u = select_chains(df)
    u["tier"] = u.id.map(sp.strictest_tier(tier_ids_fast()))
    counts["nonbound_chains"] = len(u)
    counts["nonbound_entries"] = u.entry_id.nunique()
    u["label"] = [condition_label(r) if r.perturbed else "" for r in u.itertuples()]
    return u, counts


def condition_label(r) -> str:
    parts = []
    for name, conc, unit in r.perturbants:
        unit = unit.strip()
        if unit == "%":
            parts.append(f"{float(conc):g}% {name}")
        elif conc not in (".", "?", ""):
            parts.append(f"{float(conc):g} {unit} {name}")
        else:
            parts.append(name)
    if r.pressure_bar is not None and r.pressure_bar >= HIGH_PRESSURE_BAR:
        parts.append(f"{r.pressure_bar:,.0f} bar")
    if not (T_NOTE_RANGE[0] <= r.temperature <= T_NOTE_RANGE[1]):
        parts.append(f"{r.temperature:.0f} K")
    if r.physical_state and not r.perturbants:
        parts.append(f"{str(r.physical_state).lower()} state")
    return ", ".join(parts)


def core_summary(u: pd.DataFrame, keep: np.ndarray) -> pd.DataFrame:
    """Median and IQR of all core residues per k class over the chains in ``keep``."""
    Z = np.vstack([np.asarray(x, float) for x in u.zscores])
    G = np.vstack([np.asarray(x, float) for x in u.gscores])
    rows = []
    for cl in sp.K_CLASSES:
        m = keep & (u.k_class.values == cl)
        z, g = Z[m][:, CORE].ravel(), G[m][:, CORE].ravel()
        z, g = z[np.isfinite(z)], g[np.isfinite(g)]
        tail = G[m][:, TAIL_FIRST - 1 : TAIL_LAST - 1]  # residues 73-75 (plan)
        rows.append(
            {
                "k_class": cl,
                "n_chains": int(m.sum()),
                "core_z_median": np.median(z),
                "core_z_q1": np.quantile(z, 0.25),
                "core_z_q3": np.quantile(z, 0.75),
                "core_g_median": np.median(g),
                "core_g_q1": np.quantile(g, 0.25),
                "core_g_q3": np.quantile(g, 0.75),
                "tail_g_median": np.nanmedian(tail),
            }
        )
    return pd.DataFrame(rows)


def spearman_core(u: pd.DataFrame) -> tuple[float, float, int]:
    """Spearman rho of a chain's ordered-residue (G < 0.2) median Z / G with its median k."""
    rows = []
    for g, z, k, m in zip(u.gscores, u.zscores, u.k, u["mask"]):
        g, z, k, m = (np.asarray(x, float) for x in (g, z, k, m))
        m = m.astype(bool)
        o = m & (g < 0.2)
        rows.append(
            (
                np.median(k[m]),
                np.median(z[o]) if o.any() else np.nan,
                np.median(g[o]) if o.any() else np.nan,
            )
        )
    r = pd.DataFrame(rows, columns=["k", "z", "g"])
    rz = r[["z", "k"]].corr(method="spearman").iloc[0, 1]
    rg = r[["g", "k"]].corr(method="spearman").iloc[0, 1]
    return float(rz), float(rg), int(r.z.notna().sum())


# ----------------------------------------------------------------------------------
# plot
# ----------------------------------------------------------------------------------
# Native chains are coloured by k class and enter the side summary; perturbed chains are
# drawn thin and light grey behind (select_chains, NATIVE_RULE).
OUT_NAME = "fig2a_ubiquitin"  # Fig. 2a (was Fig. 1a, fig1a.pdf, until 11 Nov 2026)

# Layout in mm, derived from the 160 mm text width (W_DOUBLE; placed at 1:1).
FIG_W, FIG_H = sp.W_DOUBLE, 59.0
LEFT = 11.5  # profile axes start (room for the y label and ticks)
RIGHT_MARGIN = 3.0
STRIP_W = 27.0  # side summary width
STRIP_GAP = 9.0  # between the profiles and the strip (room for the strip's y ticks)
STRIP_LEFT = FIG_W - RIGHT_MARGIN - STRIP_W
PROFILE_W = STRIP_LEFT - STRIP_GAP - LEFT
TOP_Z, TOP_G, AX_H = 6.0, 30.0, 22.0
# axes boxes in mm: (left, top, width, height)
BOX_Z = (LEFT, TOP_Z, PROFILE_W, AX_H)
BOX_G = (LEFT, TOP_G, PROFILE_W, AX_H)
BOX_SZ = (STRIP_LEFT, TOP_Z, STRIP_W, AX_H)
BOX_SG = (STRIP_LEFT, TOP_G, STRIP_W, AX_H)
KEY_Y = 1.7  # key row, mm from the top
Z_LIM = (-4.0, 18.0)
G_LIM = (-0.04, 1.04)
GREY, GREY_LW = "#DCDCDC", 0.45  # perturbed samples: light and thin, behind
NATIVE_LW = 0.7
BOX_HALF = 0.32  # half width of the side-summary boxes (x units = k classes)


def _axes(fig, box):
    left, top, w, h = box
    return fig.add_axes([left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])


def _mm_to_fig(x_mm, y_mm_from_top):
    return x_mm / FIG_W, 1 - y_mm_from_top / FIG_H


def _profiles(ax, u, field, lim, native):
    x = np.arange(1, len(UBQ) + 1)
    ax.axvspan(TAIL_FIRST - 0.5, TAIL_LAST + 0.5, color="0.93", lw=0, zorder=0)
    for i in u.index[~native]:  # perturbed samples first, behind
        ax.plot(x, np.asarray(u.at[i, field], float), color=GREY, lw=GREY_LW, zorder=1)
    # classes from high to low k, so the low-k lines sit on top where they cross
    for cl in list(sp.K_CLASSES)[::-1]:
        for i in u.index[native & (u.k_class.values == cl)]:
            ax.plot(
                x,
                np.asarray(u.at[i, field], float),
                color=sp.K_CLASS_COLORS[cl],
                lw=NATIVE_LW,
                alpha=0.9,
                zorder=2,
                solid_capstyle="round",
            )
    ax.set_xlim(0.3, len(UBQ) + 0.7)
    ax.set_ylim(*lim)
    ax.set_xticks([1, 20, 40, 60, 76])


def _summary(ax, S, field, lim, show_x):
    xs = np.arange(len(S))
    for xi, r in zip(xs, S.itertuples()):
        c = sp.K_CLASS_COLORS[r.k_class]
        med = getattr(r, f"core_{field}_median")
        q1 = getattr(r, f"core_{field}_q1")
        q3 = getattr(r, f"core_{field}_q3")
        # interquartile range as a light box, median as a bar across it
        ax.add_patch(
            mpatches.Rectangle(
                (xi - BOX_HALF, q1), 2 * BOX_HALF, q3 - q1, fc=c, ec="none", alpha=0.35
            )
        )
        ax.plot(
            [xi - BOX_HALF, xi + BOX_HALF],
            [med, med],
            color=c,
            lw=1.6,
            zorder=3,
            solid_capstyle="butt",
        )
    ax.set_xlim(-0.6, len(S) - 0.4)
    ax.set_ylim(*lim)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(S.k_class) if show_x else [])
    if show_x:
        ax.set_xlabel(r"Median $\mathit{k}$ of chain")


def plot(u: pd.DataFrame, S: pd.DataFrame, native: np.ndarray, out_name: str) -> None:
    from matplotlib.lines import Line2D

    fig = sp.figure(FIG_W, FIG_H)
    az, ag = _axes(fig, BOX_Z), _axes(fig, BOX_G)
    _profiles(az, u, "zscores", Z_LIM, native)
    _profiles(ag, u, "gscores", G_LIM, native)
    az.tick_params(axis="x", labelbottom=False)
    az.set_yticks([0, 5, 10, 15])
    ag.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    az.set_ylabel("Z-score")
    ag.set_ylabel("G-score")
    for a in (az, ag):  # same label offset, so the two y labels line up
        a.yaxis.set_label_coords(-7.0 / PROFILE_W, 0.5)
    ag.set_xlabel("Ubiquitin residue")

    # side summary: folded-core median and IQR per k class (same y scale as the profiles)
    sz, sg = _axes(fig, BOX_SZ), _axes(fig, BOX_SG)
    _summary(sz, S, "z", Z_LIM, show_x=False)
    _summary(sg, S, "g", G_LIM, show_x=True)
    sz.set_yticks([0, 5, 10, 15])
    sg.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    sz.tick_params(axis="x", length=0)

    # one-row key above the profiles (k classes, then the grey samples)
    handles = [
        Line2D([], [], color=sp.K_CLASS_COLORS[c], lw=1.4, label=c)
        for c in sp.K_CLASSES
    ]
    handles.append(
        Line2D([], [], color="#C4C4C4", lw=1.0, label="denaturant or high pressure")
    )
    x0, y0 = _mm_to_fig(BOX_Z[0], KEY_Y)
    fig.text(
        x0,
        y0,
        r"Median $\mathit{k}$ of chain:",
        fontsize=sp.FS_TICK,
        va="center",
        ha="left",
    )
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=_mm_to_fig(BOX_Z[0] + 22.0, KEY_Y),
        ncol=len(handles),
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.3,
        borderaxespad=0,
        borderpad=0,
        fontsize=sp.FS_TICK,
    )
    xs, ys = _mm_to_fig(BOX_SZ[0] + BOX_SZ[2] / 2, KEY_Y)
    fig.text(
        xs, ys, "Core (residues 2–71)", fontsize=sp.FS_TICK, va="center", ha="center"
    )
    sp.fig_label(fig, "a", 0.0, 0.0)
    sp.save(fig, out_name)


# ----------------------------------------------------------------------------------
def _check(name, got, plan, tol):
    ok = abs(got - plan) <= tol
    print(f"  {'OK  ' if ok else 'DIFF'} {name}: data {got:g}  plan {plan:g}")
    return ok


def main() -> None:
    args = sp.parse_args(sp.cli(__doc__))

    u, counts = load_ubiquitin()
    native = u.native.values
    S = core_summary(u, native)  # plotted; the macros' definition
    rz, rg, n_rho = spearman_core(u[native])
    teaser = u.mean_g.values < TEASER_MEAN_G_MAX  # writing plan's earlier selection
    S_plan = core_summary(u, teaser)
    rz_plan, rg_plan, n_rho_plan = spearman_core(u[teaser])

    print("ubiquitin (exact 76-aa sequence):", counts)
    print("non-bound chains per k class:", u.k_class.value_counts().to_dict())
    print("selection rule (native):", NATIVE_RULE)
    print("perturbed samples (grey):")
    cols = ["id", "k_class", "median_k", "mean_g", "tier", "physical_state", "label"]
    print(u.loc[u.perturbed, cols].round(3).to_string(index=False))
    print(
        f"native chains (coloured, side summary, macros): {int(native.sum())}; "
        "per k class:",
        u[native].k_class.value_counts().to_dict(),
    )
    print(S.round(3).to_string(index=False))
    print(f"Spearman, native chains (n={n_rho}): Z {rz:.3f}  G {rg:.3f}")
    print("sensitivity: writing plan's selection (non-bound, mean G < 0.5):")
    print(S_plan.round(3).to_string(index=False))
    print(f"  Spearman (n={n_rho_plan}): Z {rz_plan:.3f}  G {rg_plan:.3f}")

    print("check against the writing plan (2026-10 build, mean G < 0.5 selection):")
    for k_ in ("chains", "entries", "bound", "nonbound_chains", "nonbound_entries"):
        _check(k_, counts[k_], PLAN[k_], 0)
    for r in S_plan.itertuples():
        _check(f"n {r.k_class}", r.n_chains, PLAN["class_n"][r.k_class], 0)
        _check(
            f"core Z {r.k_class}",
            round(r.core_z_median, 2),
            PLAN["core_z"][r.k_class],
            0.005,
        )
        _check(
            f"core G {r.k_class}",
            round(r.core_g_median, 3),
            PLAN["core_g"][r.k_class],
            0.0005,
        )
    for cid, pv in PLAN["mean_g"].items():
        _check(
            f"mean G {cid}",
            round(float(u.loc[u.id == cid, "mean_g"].iloc[0]), 2),
            pv,
            0.005,
        )

    # source data: one row per chain and residue
    rows = []
    for r in u.itertuples():
        for i, (z, g, k, m) in enumerate(zip(r.zscores, r.gscores, r.k, r.mask)):
            rows.append(
                {
                    "chain_id": r.id,
                    "entry_id": r.entry_id,
                    "residue": i + 1,
                    "aa": UBQ[i],
                    "scored": bool(m),
                    "z_score": z if m else np.nan,
                    "g_score": g if m else np.nan,
                    "k": int(k) if m else np.nan,
                    "chain_median_k": r.median_k,
                    "k_class": r.k_class,
                    "chain_mean_g": r.mean_g,
                    "native": bool(r.native),
                    "condition": r.label,
                    "pressure_bar": r.pressure_bar,
                    "strictest_tier": r.tier,
                }
            )
    sd = args.out_dir / "source_data"
    sd.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(sd / f"{OUT_NAME}.csv", index=False)
    S.to_csv(sd / f"{OUT_NAME}_core_summary.csv", index=False)
    plot(u, S, native, OUT_NAME)
    print(f"wrote {args.out_dir / (OUT_NAME + '.pdf')}")


if __name__ == "__main__":
    main()
