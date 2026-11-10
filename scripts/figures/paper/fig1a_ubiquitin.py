#!/usr/bin/env python3
"""Fig. 1a -- ubiquitin: one protein, many depositions, two scores.

Every released chain whose sequence is exactly the 76-residue human ubiquitin
sequence (``UBQ``) is selected from the release Parquet; bound chains
(``is_bound``) are dropped, because a partner changes the measured state. For
each remaining chain the per-residue Z-score (top) and G-score (bottom) are
drawn over residues 1-76, coloured by the chain's median per-residue shift count
k (classes <= 6, 7-15, 16-21; viridis, as everywhere in the paper). The
C-terminal tail (residues 73-76), disordered in folded ubiquitin, is shaded.

Samples measured in a non-native or perturbed state are drawn dashed and
labelled with their condition and the strictest tier that keeps them. A sample
is "perturbed" when its own sample (the samples behind the chain's shift list)
contains a denaturant or alcohol cosolvent, when the condition list of the
chain's shift list records a pressure of at least 400 bar, or when the deposited
physical state is non-native. Pressure is not yet a release column (rebuild fix
(e)), so it is read from the raw NMR-STAR file of each entry, following the
chain's own shift list (``st_id``) to its sample-condition list.

The right-hand strip summarises the folded core (residues 2-71): the median
(and interquartile range) of all core residues of the chains in each k class,
over the non-bound chains with mean G-score < 0.5. This is the writing plan's
preview definition (``one_idea/ubq_profiles.py``) and the definition behind the
``\\zUbqCore*`` / ``\\gUbqCore*`` macros in ``numbers.tex``; it keeps the three
high-pressure chains, which are partially unfolded at most. The script also
prints the same medians over the unperturbed chains only, as a sensitivity
check.

Inputs (read-only): the release Parquet, the per-tier ``scores.json`` IDs of the
release bundle (strictest tier), and ``data/raw/bmrb_entries/`` (pressure and
sample components of the ubiquitin entries only).

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig1a_ubiquitin.py \
        --out-dir <draft>/Figures/paper
Outputs: ``<out-dir>/fig1a.pdf`` (180 mm wide, vector) and
``<out-dir>/source_data/fig1a.csv`` (one row per chain and residue); prints the plotted
numbers and compares them with the writing plan's preview values.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

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
TEASER_MEAN_G_MAX = 0.5  # chains entering the k-class core medians (plan definition)
HIGH_PRESSURE_BAR = 400.0  # same threshold as the writing plan's pressure scan (A12)
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
    "rho_z": 0.93,
    "rho_g": -0.03,
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


def load_ubiquitin() -> tuple[pd.DataFrame, dict]:
    df = sp.load_main(
        columns=[
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
    )
    u = df[df.sequence == UBQ].copy().reset_index(drop=True)
    counts = {
        "chains": len(u),
        "entries": u.entry_id.nunique(),
        "bound": int(u.is_bound.sum()),
    }
    strictest = sp.strictest_tier(tier_ids_fast())
    u["tier"] = u.id.map(strictest)
    u = u[~u.is_bound].reset_index(drop=True)
    counts["nonbound_chains"] = len(u)
    counts["nonbound_entries"] = u.entry_id.nunique()
    G = np.vstack([np.asarray(x, float) for x in u.gscores])
    M = np.vstack([np.asarray(x, bool) for x in u["mask"]])
    K = np.vstack([np.asarray(x, float) for x in u.k])
    u["median_k"] = [np.median(k[m]) for k, m in zip(K, M)]
    u["mean_g"] = np.nanmean(G, axis=1)
    bins = [0] + [hi for _, hi in sp.K_CLASSES.values()]  # (0, 6], (6, 15], (15, 21]
    u["k_class"] = pd.cut(u.median_k, bins, labels=list(sp.K_CLASSES)).astype(str)
    rec = [sample_record(e, s) for e, s in zip(u.entry_id, u.st_id)]
    u["pressure_bar"] = [r["pressure_bar"] for r in rec]
    u["perturbants"] = [r["perturbants"] for r in rec]
    state = u.physical_state.fillna("").str.lower()
    u["perturbed"] = (
        u.perturbants.map(bool)
        | (u.pressure_bar.fillna(0) >= HIGH_PRESSURE_BAR)
        | state.isin(NON_NATIVE_STATES)
    )
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
FIG_W, FIG_H = sp.W_DOUBLE, 64.0
# axes boxes in mm: (left, top, width, height)
BOX_Z = (11.0, 3.0, 110.0, 24.0)
BOX_G = (11.0, 31.0, 110.0, 24.0)
LABEL_X = 125.5  # left edge of the label column (mm); labels end before the strip
BOX_SZ = (160.5, 3.0, 18.0, 24.0)
BOX_SG = (160.5, 31.0, 18.0, 24.0)
Z_LIM = (-4.0, 18.0)
G_LIM = (-0.03, 1.03)
FS_SMALL = 5.0
DASH = (0, (3.2, 1.6))


def _axes(fig, box):
    left, top, w, h = box
    return fig.add_axes([left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])


def _mm_to_fig(x_mm, y_mm_from_top):
    return x_mm / FIG_W, 1 - y_mm_from_top / FIG_H


def _profiles(ax, u, field, lim):
    x = np.arange(1, len(UBQ) + 1)
    ax.axvspan(TAIL_FIRST - 0.5, TAIL_LAST + 0.5, color="0.91", lw=0, zorder=0)
    order = list(u.index[~u.perturbed]) + list(u.index[u.perturbed])
    for i in order:
        r = u.loc[i]
        y = np.asarray(r[field], float)
        ax.plot(
            x,
            y,
            color=sp.K_CLASS_COLORS[r.k_class],
            lw=0.85 if r.perturbed else 0.55,
            ls=DASH if r.perturbed else "-",
            alpha=1.0 if r.perturbed else 0.8,
            zorder=3 if r.perturbed else 2,
            solid_capstyle="round",
        )
    ax.set_xlim(0.3, len(UBQ) + 0.7)
    ax.set_ylim(*lim)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 76])


def _summary(ax, S, field, lim, show_x):
    xs = np.arange(len(S))
    for xi, r in zip(xs, S.itertuples()):
        c = sp.K_CLASS_COLORS[r.k_class]
        med = getattr(r, f"core_{field}_median")
        q1 = getattr(r, f"core_{field}_q1")
        q3 = getattr(r, f"core_{field}_q3")
        ax.plot([xi, xi], [q1, q3], color=c, lw=1.0, solid_capstyle="butt")
        ax.plot(xi, med, marker="o", ms=3.4, mfc=c, mec="black", mew=0.4, zorder=3)
    ax.set_xlim(-0.6, len(S) - 0.4)
    ax.set_ylim(*lim)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(S.k_class) if show_x else [])
    # no y tick marks: the scale is shared with the profile panel, and a tick mark next to the
    # right end of the perturbed-sample labels reads as part of the label (final pass, 31 Oct)
    ax.tick_params(axis="y", labelleft=False, length=0)
    if show_x:
        ax.set_xlabel(r"median $\mathit{k}$ of chain")


def plot(u: pd.DataFrame, S: pd.DataFrame, out_name: str) -> None:
    fig = sp.figure(FIG_W, FIG_H)
    az, ag = _axes(fig, BOX_Z), _axes(fig, BOX_G)
    _profiles(az, u, "zscores", Z_LIM)
    _profiles(ag, u, "gscores", G_LIM)
    az.tick_params(axis="x", labelbottom=False)
    az.set_ylabel("Z-score")
    ag.set_ylabel("G-score")
    ag.set_xlabel("Ubiquitin residue")
    # conventional binary cut-offs (Z = 8, G = 0.65), for orientation; named in the legend
    az.axhline(8, color="0.55", lw=0.45, ls=":", zorder=1)
    ag.axhline(sp.G_DIS, color="0.55", lw=0.45, ls=":", zorder=1)
    az.text(
        (TAIL_FIRST + TAIL_LAST) / 2,
        Z_LIM[1] * 0.985,
        "tail",
        ha="center",
        va="top",
        fontsize=FS_SMALL,
        color="0.40",
    )

    # summary strip: core medians per k class
    sz, sg = _axes(fig, BOX_SZ), _axes(fig, BOX_SG)
    _summary(sz, S, "z", Z_LIM, show_x=False)
    _summary(sg, S, "g", G_LIM, show_x=True)
    sz.set_title("core, residues 2–71", fontsize=FS_SMALL, pad=2)
    for a in (sz, sg):
        a.axhline(8 if a is sz else sp.G_DIS, color="0.55", lw=0.45, ls=":", zorder=1)

    # legend for the k classes (label column, beside the Z panel)
    from matplotlib.lines import Line2D

    n_cls = u.k_class.value_counts()
    handles = [
        Line2D(
            [],
            [],
            color=sp.K_CLASS_COLORS[c],
            lw=1.1,
            label=f"{c} ({n_cls.get(c, 0)} chains)",
        )
        for c in sp.K_CLASSES
    ]
    handles.append(
        Line2D([], [], color="0.25", lw=0.9, ls=DASH, label="perturbed sample")
    )
    x0, y0 = _mm_to_fig(LABEL_X - 1.0, BOX_Z[1] + 0.5)
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(x0, y0),
        title=r"median $\mathit{k}$ of chain",
        alignment="left",
        handlelength=2.2,
        handletextpad=0.5,
        labelspacing=0.35,
        borderaxespad=0,
        borderpad=0,
        fontsize=5.5,
        title_fontsize=5.5,
    )

    # labels for the perturbed samples, at their core-median G (beside the G panel)
    groups = []
    for (label, tier), grp in u[u.perturbed].groupby(["label", "tier"], sort=False):
        Gm = np.vstack([np.asarray(x, float) for x in grp.gscores])
        y = float(np.nanmedian(Gm[:, CORE]))
        ids = ", ".join(sorted(grp.entry_id.unique(), key=int))
        groups.append({"label": label, "tier": tier, "ids": ids, "y": y, "n": len(grp)})
    groups.sort(key=lambda d: -d["y"])
    # de-overlap in G units (two-line label block ~ 5.2 mm high)
    min_gap = 5.6 / BOX_G[3] * (G_LIM[1] - G_LIM[0])
    ys = [min(g["y"], G_LIM[1] - 0.06) for g in groups]
    for i in range(1, len(ys)):
        ys[i] = min(ys[i], ys[i - 1] - min_gap)
    shift = G_LIM[0] + 0.07 - ys[-1]
    if shift > 0:
        ys = [y + shift for y in ys]
    for g, ylab in zip(groups, ys):
        # leader from the panel edge (at the group's core-median G) to the label
        xl_fig, _ = _mm_to_fig(LABEL_X, 0)
        y_fig = ag.transData.transform((0, ylab))[1] / fig.bbox.height
        y_edge = ag.transData.transform((0, g["y"]))[1] / fig.bbox.height
        x_edge = ag.transAxes.transform((1, 0))[0] / fig.bbox.width
        fig.add_artist(
            Line2D(
                [x_edge + 0.002, xl_fig - 0.004],
                [y_edge, y_fig],
                transform=fig.transFigure,
                color="0.55",
                lw=0.4,
            )
        )
        cond = g["label"] + (f" ({g['n']} chains)" if g["n"] > 1 else "")
        fig.text(
            xl_fig,
            y_fig + 0.6 / FIG_H,
            cond,
            fontsize=5.5,
            va="bottom",
            ha="left",
        )
        # tier swatch (tier colour) + strictest tier + BMRB IDs
        fig.add_artist(
            Line2D(
                [xl_fig + 0.8 / FIG_W],
                [y_fig - 1.35 / FIG_H],
                transform=fig.transFigure,
                marker="s",
                ms=3.0,
                mfc=sp.TIER_COLORS[g["tier"]],
                mec="0.3",
                mew=0.3,
                ls="none",
            )
        )
        fig.text(
            xl_fig + 2.2 / FIG_W,
            y_fig - 0.4 / FIG_H,
            f"{g['tier']} · BMRB {g['ids']}",
            fontsize=FS_SMALL,
            color="0.30",
            va="top",
            ha="left",
        )
    sp.save(fig, out_name)


# ----------------------------------------------------------------------------------
def _check(name, got, plan, tol):
    ok = abs(got - plan) <= tol
    print(f"  {'OK  ' if ok else 'DIFF'} {name}: data {got:g}  plan {plan:g}")
    return ok


def main() -> None:
    args = sp.parse_args(sp.cli(__doc__))

    u, counts = load_ubiquitin()
    teaser = u.mean_g.values < TEASER_MEAN_G_MAX
    S = core_summary(u, teaser)
    S_native = core_summary(u, ~u.perturbed.values)
    rz, rg, n_rho = spearman_core(u)
    rz_nat, rg_nat, n_rho_nat = spearman_core(u[~u.perturbed])

    print("ubiquitin (exact 76-aa sequence):", counts)
    print("non-bound chains per k class:", u.k_class.value_counts().to_dict())
    print("perturbed samples:")
    cols = ["id", "k_class", "median_k", "mean_g", "tier", "physical_state", "label"]
    print(u.loc[u.perturbed, cols].round(3).to_string(index=False))
    print("core medians, plan definition (non-bound, mean G < 0.5):")
    print(S.round(3).to_string(index=False))
    print("core medians, sensitivity (unperturbed chains only):")
    print(S_native.round(3).to_string(index=False))
    print(f"Spearman (chains with ordered residues, n={n_rho}): Z {rz:.3f}  G {rg:.3f}")
    print(f"  unperturbed only (n={n_rho_nat}): Z {rz_nat:.3f}  G {rg_nat:.3f}")

    print("check against the writing plan:")
    for k_ in ("chains", "entries", "bound", "nonbound_chains", "nonbound_entries"):
        _check(k_, counts[k_], PLAN[k_], 0)
    for r in S.itertuples():
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
    _check("rho Z", round(rz, 2), PLAN["rho_z"], 0.005)
    _check("rho G", round(rg, 2), PLAN["rho_g"], 0.005)
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
                    "in_core_summary": bool(r.mean_g < TEASER_MEAN_G_MAX),
                    "perturbed": bool(r.perturbed),
                    "condition": r.label,
                    "pressure_bar": r.pressure_bar,
                    "strictest_tier": r.tier,
                }
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "source_data").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "source_data" / "fig1a.csv", index=False)
    S.to_csv(args.out_dir / "source_data" / "fig1a_core_summary.csv", index=False)
    plot(u, S, "fig1a")
    print(f"wrote {args.out_dir / 'fig1a.pdf'}")


if __name__ == "__main__":
    main()
