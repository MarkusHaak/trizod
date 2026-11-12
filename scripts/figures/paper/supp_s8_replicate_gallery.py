#!/usr/bin/env python3
"""Supplementary Fig. S8 -- ubiquitin with its non-native chains, two case clusters and a seeded gallery.

a    The ubiquitin chains of Fig. 3c (``fig1a_ubiquitin.load_ubiquitin``: every non-bound
     chain with the exact 76-residue sequence), now WITH the chains that Fig. 3c leaves
     out (since 12 Nov 2026, story blueprint 6.4 and 6.7): native chains coloured by
     their median k as in Fig. 3c, the perturbed ones (denaturant or alcohol, >= 400 bar,
     non-native deposited state; ``fig1a_ubiquitin.NATIVE_RULE``, read from the raw
     NMR-STAR files) light grey, with their condition written beside them. Until 12 Nov
     2026 panel a was the ubiquitin case cluster (seed 50330_1_1_1, the ``\\nCaseUbq*``
     macros); it is no longer drawn.
b, c The case-study clusters of the prion protein (16616_1_1_1) and alpha-synuclein
     (6968_1_1_1). Members are the chains that the
     seed's sequence finds with ``mmseqs easy-search`` (-s 7.5, --max-seqs 20000,
     -e 1e-3) at >= 80 % identity over >= 90 % of both sequences: the definition of the
     ``\\nCase*`` macros in ``numbers.tex`` (make_numbers.py ``sec_ubiquitin``). This
     script reruns the same search with alignment strings and maps every member residue
     onto the seed's residue numbering; residues in insertions relative to the seed are
     not drawn. Per-residue Z-scores (top) and G-scores (bottom) are drawn as in
     Fig. 3c: bound chains are left out; chains measured as native proteins in aqueous
     solution are coloured by their median shift count k (classes <= 6, 7-15, 16-21);
     chains measured in another sample type are light grey (``perturbed``: cosolvent
     evidence, a membrane mimetic, a solid-state sample, a non-native deposited physical
     state, or a pressure of at least 400 bar).
d    A gallery of ``N_GALLERY`` replicate groups drawn with ``GALLERY_SEED`` from all
     sequences whose non-bound chains come from at least ``MIN_ENTRIES`` entries
     (exact sequence identity; sequences sorted lexicographically before the draw), each
     drawn as b and c. Titles name the protein (``GALLERY_NAMES``, from the BMRB entry titles,
     where the deposited entity name is a lab code or generic).

Labels are the release Parquet's (``gscores``, ``zscores``, ``k``, ``mask``). Unlike
panel a (and Fig. 3c), the grey class of b-d reads the Parquet's sample annotations (``cosolvent_evidence``,
``membrane_mimetic``, ``sample_state_evidence``, ``physical_state``, ``pressure_bar``)
and not the raw files. A few alpha-synuclein fibril and micelle samples (31068, 18243,
16302) escape these annotations and stay coloured.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s8_replicate_gallery.py \
        --out-dir <draft>/Figures/paper
Outputs: ``<out-dir>/S8.pdf`` and ``<out-dir>/suppfigs_c_data/S8_*.csv``
(``S8_ubiquitin_chains.csv`` for panel a). The mmseqs case
search is cached in ``tmp/paper_figs/s8_cases`` (``--refresh`` reruns it).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig1a_ubiquitin as f1a  # noqa: E402  (panel a: Fig. 3c's chain set and rule)
import style_paper as sp  # noqa: E402

# italic k in the same sans-serif face as the text (as Fig. 3c; mathtext defaults to DejaVu)
sp.plt.rcParams.update(
    {
        "mathtext.fontset": "custom",
        "mathtext.it": "Arial:italic",
        "mathtext.rm": "Arial",
    }
)

CASES = {  # panel -> (name, seed chain); panel a is the exact ubiquitin sequence of Fig. 3c
    "b": ("Prion protein", "16616_1_1_1"),
    "c": ("α-Synuclein", "6968_1_1_1"),
}
PLAN_COUNTS = {  # numbers.tex \nCase*Chains / \nCase*Entries, for the self-check
    "16616_1_1_1": (36, 34),
    "6968_1_1_1": (108, 71),
}
MIN_FIDENT, MIN_COV = 0.8, 0.9
GALLERY_SEED = 2026
N_GALLERY = 12
MIN_ENTRIES = 3
# A chain is drawn grey ("other sample") when its sample is not a native protein in aqueous
# solution: a cosolvent (denaturant, alcohol, ...), a membrane mimetic, the solid state, a
# non-native deposited physical state, or a pressure of at least HIGH_PRESSURE_BAR.
NON_NATIVE_STATES = {"denatured", "unfolded", "molten globule", "misfolded", "fibril",
                     "fibrils", "amyloid fibril"}  # fmt: skip
HIGH_PRESSURE_BAR = 400.0  # as Fig. 3c
OTHER_LABEL = "cosolvent, micelles, solid state, high pressure or non-native state"
CACHE = sp.TMP / "paper_figs" / "s8_cases"
DATA_SUBDIR = "suppfigs_c_data"  # under --out-dir
Z_LIM = (-4.0, 18.0)
G_LIM = (-0.03, 1.03)
FS_SMALL = sp.FS_SMALL
# deposited entity names that say nothing about the protein; the gallery titles name the
# BMRB entry instead (only for groups that GALLERY_NAMES does not name)
GENERIC_NAME = re.compile(
    r"(entity|entity[ _]?\d+|polymer|protein|peptide|chain[ _]?\w?|\w{0,2})", re.I
)
# Gallery titles: the protein, named from the BMRB entry title where the deposited entity
# name is a lab code or generic (keyed by any entry of the group; checked 11 Nov 2026
# against the entry titles of the 2026-11 draw).
GALLERY_NAMES = {
    "16727": "Culbertcidin",  # antimicrobial peptide
    "1481": "Salmon calcitonin",  # entity "calcitonin C"
    "19128": "Contryphan",  # entity "cIn936", a Conus venom contryphan
    "1072": "Ribonuclease A",
    "16275": "Apelin-17",
    "16435": "Leech tryptase inhibitor",  # entity "LDTI"
    "27317": "HBV core protein",  # entity "Cp149" (hepatitis B virus capsid)
    "27808": "Ubiquitin K63C",  # entity "Ubiquitin_K63C-PA"
    "31115": "Protein G B1 domain",  # entity "entity_1" (streptococcal protein GB1)
    "4233": "ω-atracotoxin-Hv1a",  # entity "omega-atracotoxin-Hv1a"
    "52965": "Trpzip2",  # entity "entity_1" (tryptophan zipper 2)
    "50676": "Thrombin",  # entity "entity_1" (PPACK-thrombin)
    "15047": "Ubiquitin",  # entity "denatured_ubiquitin", but most chains are native (a)
    "5580": "Mersacidin",  # entity "Type B lantibiotics mersacidin" (a type B lantibiotic)
}


# --------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = sp.load_main(
        [
            "id",
            "entry_id",
            "sequence",
            "entity_name",
            "gscores",
            "zscores",
            "k",
            "mask",
            "is_bound",
            "physical_state",
            "cosolvent_evidence",
            "sample_state_evidence",
            "membrane_mimetic",
            "pressure_bar",
            "split",
        ]  # fmt: skip
    )
    df["entry_id"] = df.entry_id.astype(str)
    state = df.physical_state.fillna("").str.lower()
    # "perturbed" = another sample type than a native protein in aqueous solution
    df["perturbed"] = (
        df.cosolvent_evidence.fillna(False).astype(bool)
        | state.isin(NON_NATIVE_STATES)
        | (df.sample_state_evidence.fillna("") == "solid")
        | df.membrane_mimetic.notna()
        | (df.pressure_bar.fillna(0) >= HIGH_PRESSURE_BAR)
    )
    med_k = []
    for k, m in zip(df.k, df["mask"]):
        k, m = np.asarray(k, float), np.asarray(m, bool)
        med_k.append(np.median(k[m]) if m.any() else np.nan)
    df["median_k"] = med_k
    bins = [0] + [hi for _, hi in sp.K_CLASSES.values()]
    df["k_class"] = pd.cut(df.median_k, bins, labels=list(sp.K_CLASSES)).astype(str)
    return df


def case_alignments(df: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    """mmseqs easy-search of the three seed sequences against all chains, with alignments."""
    out = CACHE / "hits.m8"
    if refresh or not out.exists():
        if shutil.which("mmseqs") is None:
            raise SystemExit("mmseqs not on PATH")
        CACHE.mkdir(parents=True, exist_ok=True)
        seqs = df.set_index("id").sequence
        with open(CACHE / "q.fasta", "w") as fh:
            for _, cid in CASES.values():
                fh.write(f">{cid}\n{seqs[cid]}\n")
        with open(CACHE / "t.fasta", "w") as fh:
            for i, s in zip(df.id, df.sequence):
                fh.write(f">{i}\n{s}\n")
        fmt = "query,target,fident,qcov,tcov,evalue,qstart,qend,tstart,tend,qaln,taln"
        cmd = ["mmseqs", "easy-search", str(CACHE / "q.fasta"), str(CACHE / "t.fasta"),
               str(out), str(CACHE / "tmp"), "-s", "7.5", "--max-seqs", "20000", "-e", "1e-3",
               "--format-output", fmt, "--threads", "8"]  # fmt: skip
        with open(CACHE / "mmseqs.log", "w") as h:
            subprocess.run(cmd, check=True, stdout=h, stderr=subprocess.STDOUT)
    cols = [
        "q",
        "t",
        "fid",
        "qcov",
        "tcov",
        "e",
        "qs",
        "qe",
        "ts",
        "te",
        "qaln",
        "taln",
    ]
    H = pd.read_csv(out, sep="\t", names=cols)
    H = H[(H.fid >= MIN_FIDENT) & (H.qcov >= MIN_COV) & (H.tcov >= MIN_COV)]
    return H.drop_duplicates(["q", "t"], keep="first").reset_index(drop=True)


def map_to_seed(h, n_seed: int) -> np.ndarray:
    """Index array: seed position (0-based) -> member position (0-based), -1 if gap."""
    idx = np.full(n_seed, -1, int)
    qi, ti = int(h.qs) - 1, int(h.ts) - 1
    for a, b in zip(h.qaln, h.taln):
        if a != "-" and b != "-":
            idx[qi] = ti
        if a != "-":
            qi += 1
        if b != "-":
            ti += 1
    return idx


def profiles_on_seed(
    df_idx: pd.DataFrame, hits: pd.DataFrame, seed: str
) -> pd.DataFrame:
    n = len(df_idx.loc[seed, "sequence"])
    rows = []
    for h in hits[hits.q == seed].itertuples():
        r = df_idx.loc[h.t]
        idx = map_to_seed(h, n)
        z = np.asarray(r.zscores, float)
        g = np.asarray(r.gscores, float)
        m = np.asarray(r["mask"], bool)
        zz, gg = np.full(n, np.nan), np.full(n, np.nan)
        ok = idx >= 0
        sel = idx[ok]
        good = m[sel]
        pos = np.where(ok)[0][good]
        zz[pos], gg[pos] = z[sel[good]], g[sel[good]]
        rows.append({"id": h.t, "entry_id": r.entry_id, "fident": h.fid, "is_bound": bool(r.is_bound),
                     "perturbed": bool(r.perturbed), "k_class": r.k_class, "split": r.split,
                     "z": zz, "g": gg})  # fmt: skip
    return pd.DataFrame(rows)


def gallery(df: pd.DataFrame) -> tuple[list, int]:
    nb = df[~df.is_bound.astype(bool)]
    n_ent = nb.groupby("sequence").entry_id.nunique()
    eligible = sorted(n_ent[n_ent >= MIN_ENTRIES].index)
    rng = np.random.default_rng(GALLERY_SEED)
    pick = [
        eligible[i] for i in sorted(rng.choice(len(eligible), N_GALLERY, replace=False))
    ]
    return pick, len(eligible)


# --------------------------------------------------------------------------------
# plot (drawn as Fig. 3c: native solution chains coloured by median k, the rest grey)
# --------------------------------------------------------------------------------
FIG_W, FIG_H = sp.W_DOUBLE, 162.0  # with its legend, the figure fits one A4 page
LEFT = 14.0  # left edge of the first column of axes (room for the y labels)
CASE_W, CASE_GAP, CASE_TOP, CASE_H = 45.0, 5.5, 12.5, 18.0
GAL_COLS, GAL_GAP, GAL_TOP, GAL_H, GAL_ROW = 4, 5.0, 69.0, 10.5, 31.0
GAL_W = (FIG_W - LEFT - 1.0 - (GAL_COLS - 1) * GAL_GAP) / GAL_COLS
GREY = "#D2D2D2"  # chains measured in another sample type (as Fig. 3c)


def _ax(fig, left, top, w, h):
    return fig.add_axes([left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])


def _draw_group(az, ag, x, rows, lw):
    """Grey chains (other sample types) behind, coloured chains (median-k class) on top."""
    for r in rows:
        if r["other"]:
            az.plot(x, r["z"], color=GREY, lw=lw * 0.85, zorder=1)
            ag.plot(x, r["g"], color=GREY, lw=lw * 0.85, zorder=1)
    for r in rows:
        if not r["other"]:
            c = sp.K_CLASS_COLORS.get(r["k_class"], "0.5")
            az.plot(x, r["z"], color=c, lw=lw, alpha=0.9, zorder=2)
            ag.plot(x, r["g"], color=c, lw=lw, alpha=0.9, zorder=2)


def _style(az, ag, n, show_xlabel, ylab=True, small=False):
    for a, lim in ((az, Z_LIM), (ag, G_LIM)):
        a.set_xlim(0.3, n + 0.7)
        a.set_ylim(*lim)
        a.xaxis.set_major_locator(
            sp.mticker.MaxNLocator(nbins=4, integer=True, min_n_ticks=2)
        )
    az.tick_params(axis="x", labelbottom=False)
    az.set_yticks([0, 10] if small else [0, 5, 10, 15])  # gallery axes are short
    ag.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    if ylab:
        az.set_ylabel("Z-score")
        ag.set_ylabel("G-score")
    else:
        az.tick_params(axis="y", labelleft=False)
        ag.tick_params(axis="y", labelleft=False)
    if show_xlabel:
        ag.set_xlabel("Residue")


def _case_rows(P: pd.DataFrame) -> list:
    """Non-bound case members, as Fig. 3c (a partner changes the measured state)."""
    P = P[~P.is_bound.astype(bool)]
    return [{"z": r.z, "g": r.g, "k_class": r.k_class, "other": bool(r.perturbed)}
            for r in P.itertuples()]  # fmt: skip


def _gallery_rows(grp: pd.DataFrame) -> list:
    rows = []
    for r in grp.itertuples():
        m = np.asarray(r.mask, bool)
        rows.append({"z": np.where(m, np.asarray(r.zscores, float), np.nan),
                     "g": np.where(m, np.asarray(r.gscores, float), np.nan),
                     "k_class": r.k_class, "other": bool(r.perturbed)})  # fmt: skip
    return rows


def gallery_name(grp: pd.DataFrame) -> str:
    """Protein name of a gallery group: curated (``GALLERY_NAMES``), else the entity name."""
    for e in sorted(grp.entry_id.astype(str), key=int):
        if e in GALLERY_NAMES:
            return GALLERY_NAMES[e]
    name = str(grp.entity_name.iloc[0]) if pd.notna(grp.entity_name.iloc[0]) else ""
    name = name.replace("_", " ").strip()
    if GENERIC_NAME.fullmatch(name):
        return f"BMRB {min(grp.entry_id.astype(str), key=int)}"
    name = re.sub(r"^omega-", "ω-", name, flags=re.I)
    name = (
        name[0].upper() + name[1:] if name[:1].islower() and name[:2] != "ω-" else name
    )
    # never a clipped title (figure review item 16): a long name falls back to the entry
    if len(name) > 26:
        return f"BMRB {min(grp.entry_id.astype(str), key=int)}"
    return name


def ubiquitin_rows(U: pd.DataFrame) -> list:
    """Panel a: the non-bound chains of the exact ubiquitin sequence (Fig. 3c's set), native
    ones by median-k class, the perturbed ones (``fig1a_ubiquitin.NATIVE_RULE``) grey."""
    return [{"z": np.asarray(r.zscores, float), "g": np.asarray(r.gscores, float),
             "k_class": r.k_class, "other": not bool(r.native)}
            for r in U.itertuples()]  # fmt: skip


def condition_groups(U: pd.DataFrame) -> list:
    """Direct labels of panel a: the perturbed chains grouped by what perturbs them (a
    pressure of at least HIGH_PRESSURE_BAR first, else the first denaturant or alcohol
    named in the sample, else the deposited state), with the median G of each group over
    the folded core (residues 2-71), where the label is placed. The full condition of
    every chain is in S8_ubiquitin_chains.csv."""
    P = U[~U.native.astype(bool)]
    groups: dict = {}
    for r in P.itertuples():
        if pd.notna(r.pressure_bar) and r.pressure_bar >= HIGH_PRESSURE_BAR:
            key = f"{r.pressure_bar:,.0f} bar"
        elif r.perturbants:  # (name, concentration, unit) of the first perturbant
            name, conc, unit = r.perturbants[0]
            unit = unit.strip()
            key = f"{float(conc):g}{'' if unit == '%' else ' '}{unit} {name}".strip()
        else:
            key = f"{r.physical_state} state"
        groups.setdefault(key, []).append(
            np.nanmean(np.asarray(r.gscores, float)[1:71])
        )
    return [(k, float(np.median(v)), len(v)) for k, v in groups.items()]


def plot(
    cases: dict,
    gal: list,
    df_idx: pd.DataFrame,
    by_seq: dict,
    n_eligible: int,
    U: pd.DataFrame,
) -> None:
    from matplotlib.lines import Line2D

    fig = sp.figure(FIG_W, FIG_H)
    # ---- a: the ubiquitin chains of Fig. 3c with the perturbed ones ------------
    n = len(f1a.UBQ)
    az = _ax(fig, LEFT, CASE_TOP, CASE_W, CASE_H)
    ag = _ax(fig, LEFT, CASE_TOP + CASE_H + 2.0, CASE_W, CASE_H)
    _draw_group(az, ag, np.arange(1, n + 1), ubiquitin_rows(U), lw=0.55)
    _style(az, ag, n, show_xlabel=True, ylab=True)
    az.set_title("Ubiquitin (Fig. 3c), all unbound chains", fontsize=sp.FS_BODY, pad=2,
                 loc="left")  # fmt: skip
    # condition labels on the grey groups, at residue 38 (inside the folded core, where
    # the native chains lie low), on a light box so the lines stay readable around them
    for label, g_med, _ in condition_groups(U):
        ag.text(38, g_med, label, fontsize=FS_SMALL, color=sp.INK_SOFT, ha="center",
                va="center", zorder=4,
                bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none",
                      "alpha": 0.85})  # fmt: skip
    sp.fig_label(fig, "a", 0.0, 7.5)
    # ---- b, c: case clusters --------------------------------------------------
    for j, (letter, (name, seed)) in enumerate(CASES.items(), start=1):
        n = len(df_idx.loc[seed, "sequence"])
        left = LEFT + j * (CASE_W + CASE_GAP)
        az = _ax(fig, left, CASE_TOP, CASE_W, CASE_H)
        ag = _ax(fig, left, CASE_TOP + CASE_H + 2.0, CASE_W, CASE_H)
        _draw_group(az, ag, np.arange(1, n + 1), _case_rows(cases[seed]), lw=0.55)
        _style(az, ag, n, show_xlabel=True, ylab=False)
        az.set_title(name, fontsize=sp.FS_BODY, pad=2, loc="left")
        sp.fig_label(fig, letter, left - 4.5, 7.5)

    # one-row key above everything, as Fig. 3c
    handles = [Line2D([], [], color=sp.K_CLASS_COLORS[c], lw=1.4, label=c)
               for c in sp.K_CLASSES]  # fmt: skip
    handles.append(Line2D([], [], color=GREY, lw=1.4, label=OTHER_LABEL))
    fig.text(LEFT / FIG_W, 1 - 2.2 / FIG_H, r"Median $\mathit{k}$ of chain:",
             fontsize=sp.FS_TICK, va="center", ha="left")  # fmt: skip
    fig.legend(handles=handles, loc="center left",
               bbox_to_anchor=((LEFT + 23.0) / FIG_W, 1 - 2.2 / FIG_H), ncol=len(handles),
               handlelength=1.6, handletextpad=0.5, columnspacing=1.4, borderaxespad=0,
               borderpad=0, fontsize=sp.FS_TICK)  # fmt: skip

    # ---- d: seeded gallery ----------------------------------------------------
    n_rows = int(np.ceil(len(gal) / GAL_COLS))
    for i, seq in enumerate(gal):
        c, rr = i % GAL_COLS, i // GAL_COLS
        left = LEFT + c * (GAL_W + GAL_GAP)
        top = GAL_TOP + rr * GAL_ROW
        az = _ax(fig, left, top, GAL_W, GAL_H)
        ag = _ax(fig, left, top + GAL_H + 1.5, GAL_W, GAL_H)
        grp = by_seq[seq]
        _draw_group(az, ag, np.arange(1, len(seq) + 1), _gallery_rows(grp), lw=0.5)
        _style(
            az, ag, len(seq), show_xlabel=(rr == n_rows - 1), ylab=(c == 0), small=True
        )
        az.set_title(f"{gallery_name(grp)}\n{len(seq)} aa · {grp.entry_id.nunique()} entries",
                     fontsize=FS_SMALL, pad=1.5, loc="left", linespacing=1.15)  # fmt: skip
    sp.fig_label(fig, "d", 0.0, GAL_TOP - 9.5)
    sp.save(fig, "S8")


def main() -> None:
    ap = sp.cli(__doc__)
    ap.add_argument(
        "--refresh", action="store_true", help="rerun the mmseqs case search"
    )
    args = sp.parse_args(ap)
    DATA_DIR = sp.out_dir(DATA_SUBDIR)

    df = load()
    df_idx = df.set_index("id")
    H = case_alignments(df, args.refresh)
    cases, rows = {}, []
    print("case clusters (chains, entries, bound, excluded) vs plan:")
    for _, seed in CASES.values():
        P = profiles_on_seed(df_idx, H, seed)
        cases[seed] = P
        got = (len(P), P.entry_id.nunique())
        ok = "OK  " if got == PLAN_COUNTS[seed] else "DIFF"
        print(f"  {ok} {seed}: {got} plan {PLAN_COUNTS[seed]}; bound {int(P.is_bound.sum())}; "
              f"excluded {int((P.split == 'excluded').sum())}; min identity {P.fident.min():.3f}")  # fmt: skip
        for r in P.itertuples():
            rows.append({"seed": seed, "chain_id": r.id, "entry_id": r.entry_id, "fident": r.fident,
                         "is_bound": r.is_bound, "perturbed": r.perturbed, "k_class": r.k_class,
                         "split": r.split})  # fmt: skip
    gal, n_eligible = gallery(df)
    nb = df[~df.is_bound.astype(bool)]
    by_seq = {s: nb[nb.sequence == s] for s in gal}
    print(
        f"gallery: {N_GALLERY} of {n_eligible} eligible sequences (seed {GALLERY_SEED})"
    )
    grows = []
    for i, s in enumerate(gal):
        g = by_seq[s]
        print(f"  {i + 1:2d}. {str(g.entity_name.iloc[0])[:40]:40s} len {len(s):4d} chains "
              f"{len(g):3d} entries {g.entry_id.nunique():3d} ids {', '.join(sorted(g.id)[:4])}")  # fmt: skip
        for r in g.itertuples():
            grows.append({"group": i + 1, "sequence_length": len(s), "chain_id": r.id,
                          "entry_id": r.entry_id, "entity_name": r.entity_name,
                          "k_class": r.k_class, "perturbed": r.perturbed})  # fmt: skip
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DATA_DIR / "S8_case_members.csv", index=False)
    pd.DataFrame(grows).to_csv(DATA_DIR / "S8_gallery_members.csv", index=False)
    (DATA_DIR / "S8_info.json").write_text(json.dumps(
        {"gallery_seed": GALLERY_SEED, "n_gallery": N_GALLERY, "min_entries": MIN_ENTRIES,
         "n_eligible_sequences": n_eligible,
         "cases": {seed: {"chains": len(P), "entries": int(P.entry_id.nunique()),
                          "bound": int(P.is_bound.sum())} for seed, P in cases.items()}},
        indent=1))  # fmt: skip
    U, ucounts = f1a.load_ubiquitin()
    print(f"ubiquitin (panel a): {ucounts}; native {int(U.native.sum())}, "
          f"perturbed {int((~U.native).sum())}; groups {condition_groups(U)}")  # fmt: skip
    U.assign(condition=U["label"])[
        ["id", "entry_id", "native", "condition", "pressure_bar", "median_k", "k_class",
         "mean_g", "tier"]
    ].to_csv(DATA_DIR / "S8_ubiquitin_chains.csv", index=False)  # fmt: skip
    plot(cases, gal, df_idx, by_seq, n_eligible, U)
    print(f"wrote {sp.out_dir() / 'S8.pdf'}")


if __name__ == "__main__":
    main()
