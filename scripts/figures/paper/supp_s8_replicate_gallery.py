#!/usr/bin/env python3
"""Supplementary Fig. S8 -- replicate groups beyond ubiquitin: case clusters and a seeded gallery.

a-c  The case-study clusters of ubiquitin (seed chain 50330_1_1_1), the prion protein
     (16616_1_1_1) and alpha-synuclein (6968_1_1_1). Members are the chains that the
     seed's sequence finds with ``mmseqs easy-search`` (-s 7.5, --max-seqs 20000,
     -e 1e-3) at >= 80 % identity over >= 90 % of both sequences: the definition of the
     ``\\nCase*`` macros in ``numbers.tex`` (make_numbers.py ``sec_ubiquitin``). This
     script reruns the same search with alignment strings and maps every member residue
     onto the seed's residue numbering; residues in insertions relative to the seed are
     not drawn. Per-residue Z-scores (top) and G-scores (bottom) are drawn as in
     Fig. 1a: non-bound chains coloured by their median shift count k (classes <= 6,
     7-15, 16-21), chains with cosolvent evidence or a non-native deposited physical
     state dashed. Bound chains, which Fig. 1a leaves out, are drawn thin and grey.
d    A gallery of ``N_GALLERY`` replicate groups drawn with ``GALLERY_SEED`` from all
     sequences whose non-bound chains come from at least ``MIN_ENTRIES`` entries
     (exact sequence identity; sequences sorted lexicographically before the draw), each
     drawn as in Fig. 1a.

Labels are the release Parquet's (``gscores``, ``zscores``, ``k``, ``mask``). Unlike
Fig. 1a, the dashed style reads the Parquet's ``cosolvent_evidence`` and
``physical_state`` and not the raw files, so samples perturbed only by pressure are
not dashed.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s8_replicate_gallery.py \
        --out-dir <draft>/Figures/paper
Outputs: ``<out-dir>/S8.pdf`` and ``<out-dir>/suppfigs_c_data/S8_*.csv``. The mmseqs case
search is cached in ``tmp/paper_figs/s8_cases`` (``--refresh`` reruns it).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402

CASES = {  # panel -> (name, seed chain)
    "a": ("Ubiquitin", "50330_1_1_1"),
    "b": ("Prion protein", "16616_1_1_1"),
    "c": ("α-Synuclein", "6968_1_1_1"),
}
PLAN_COUNTS = {  # writing plan / numbers.tex (chains, entries) for the self-check
    "50330_1_1_1": (119, 103),
    "16616_1_1_1": (36, 34),
    "6968_1_1_1": (108, 71),
}
MIN_FIDENT, MIN_COV = 0.8, 0.9
GALLERY_SEED = 2026
N_GALLERY = 12
MIN_ENTRIES = 3
NON_NATIVE_STATES = {"denatured", "unfolded", "molten globule", "misfolded"}
CACHE = sp.TMP / "paper_figs" / "s8_cases"
DATA_SUBDIR = "suppfigs_c_data"  # under --out-dir
Z_LIM = (-4.0, 18.0)
G_LIM = (-0.03, 1.03)
FS_SMALL = 5.3
DASH = (0, (3.2, 1.6))


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
            "split",
        ]  # fmt: skip
    )
    df["entry_id"] = df.entry_id.astype(str)
    state = df.physical_state.fillna("").str.lower()
    df["perturbed"] = df.cosolvent_evidence.fillna(False).astype(bool) | state.isin(
        NON_NATIVE_STATES
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
# plot
# --------------------------------------------------------------------------------
FIG_W, FIG_H = sp.W_DOUBLE, 200.0


def _ax(fig, left, top, w, h):
    return fig.add_axes([left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])


def _draw(ax, x, y, row_bound, row_k, row_pert, lw=0.5):
    if row_bound:
        ax.plot(x, y, color="0.78", lw=0.35, zorder=1)
    else:
        ax.plot(x, y, color=sp.K_CLASS_COLORS.get(row_k, "0.5"),
                lw=lw * (1.5 if row_pert else 1.0), ls=DASH if row_pert else "-",
                alpha=1.0 if row_pert else 0.8, zorder=3 if row_pert else 2)  # fmt: skip


def _style(az, ag, n, show_xlabel, ylab=True, small=False):
    for a, lim, thr in ((az, Z_LIM, 8.0), (ag, G_LIM, sp.G_DIS)):
        a.set_xlim(0.3, n + 0.7)
        a.set_ylim(*lim)
        a.axhline(thr, color="0.55", lw=0.45, ls=":", zorder=0)
    az.tick_params(axis="x", labelbottom=False)
    if ylab:
        az.set_ylabel("Z", labelpad=1 if small else 2)
        ag.set_ylabel("G", labelpad=1 if small else 2)
    else:
        az.tick_params(axis="y", labelleft=False)
        ag.tick_params(axis="y", labelleft=False)
    az.set_yticks([0, 8, 16])
    ag.set_yticks([0, 0.5, 1])
    if show_xlabel:
        ag.set_xlabel("residue (seed numbering)" if not small else "residue")


def plot(
    cases: dict, gal: list, df_idx: pd.DataFrame, by_seq: dict, n_eligible: int
) -> None:
    from matplotlib.lines import Line2D

    fig = sp.figure(FIG_W, FIG_H)
    # ---- a-c: case clusters ---------------------------------------------------
    w, gap = 49.0, 10.5
    for j, (letter, (name, seed)) in enumerate(CASES.items()):
        P = cases[seed]
        n = len(df_idx.loc[seed, "sequence"])
        left = 12.0 + j * (w + gap)
        az, ag = _ax(fig, left, 12.0, w, 22.0), _ax(fig, left, 37.0, w, 22.0)
        x = np.arange(1, n + 1)
        for bound_first in (True, False):
            for r in P[P.is_bound == bound_first].sort_values("perturbed").itertuples():
                _draw(az, x, r.z, r.is_bound, r.k_class, r.perturbed)
                _draw(ag, x, r.g, r.is_bound, r.k_class, r.perturbed)
        _style(az, ag, n, show_xlabel=True, ylab=(j == 0))
        nb = int((~P.is_bound).sum())
        az.set_title(f"{name} · {len(P)} chains, {P.entry_id.nunique()} entries "
                     f"({nb} non-bound)", fontsize=5.5, pad=2, loc="left")  # fmt: skip
        sp.fig_label(fig, letter, left - 11.0, 6.0)
    handles = [Line2D([], [], color=sp.K_CLASS_COLORS[c], lw=1.0, label=f"median k {c}")
               for c in sp.K_CLASSES]  # fmt: skip
    handles += [
        Line2D(
            [], [], color="0.3", lw=0.9, ls=DASH, label="cosolvent or non-native state"
        ),
        Line2D([], [], color="0.78", lw=0.6, label="bound chain"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(12.0 / FIG_W, 1 - 1.0 / FIG_H),
               ncol=5, fontsize=5.5, handlelength=2.0, columnspacing=1.2, borderaxespad=0)  # fmt: skip

    # ---- d: seeded gallery ----------------------------------------------------
    cols, gw, ggap, top0, gh, rgap = 4, 36.5, 7.0, 82.0, 13.5, 13.0
    for i, seq in enumerate(gal):
        c, rr = i % cols, i // cols
        left = 12.0 + c * (gw + ggap)
        top = top0 + rr * (2 * gh + 1.5 + rgap)
        az, ag = _ax(fig, left, top, gw, gh), _ax(fig, left, top + gh + 1.5, gw, gh)
        grp = by_seq[seq]
        x = np.arange(1, len(seq) + 1)
        for r in grp.sort_values("perturbed").itertuples():
            m = np.asarray(r.mask, bool)
            z = np.where(m, np.asarray(r.zscores, float), np.nan)
            g = np.where(m, np.asarray(r.gscores, float), np.nan)
            _draw(az, x, z, False, r.k_class, r.perturbed, lw=0.45)
            _draw(ag, x, g, False, r.k_class, r.perturbed, lw=0.45)
        _style(az, ag, len(seq), show_xlabel=(rr == 2), ylab=(c == 0), small=True)
        name = str(grp.entity_name.iloc[0]) if pd.notna(grp.entity_name.iloc[0]) else ""
        name = name if len(name) <= 30 else name[:28] + "…"
        az.set_title(f"{name}\n{len(seq)} aa · {len(grp)} chains, {grp.entry_id.nunique()} entries",
                     fontsize=FS_SMALL, pad=1.5, loc="left", linespacing=1.1)  # fmt: skip
    sp.fig_label(fig, "d", 2.0, 72.0)
    fig.text(12.0 / FIG_W, 1 - 72.5 / FIG_H,
             f"{N_GALLERY} of {n_eligible:,} sequences with non-bound chains in ≥ {MIN_ENTRIES} entries, "
             f"drawn with seed {GALLERY_SEED}", fontsize=5.5, va="top")  # fmt: skip
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
    plot(cases, gal, df_idx, by_seq, n_eligible)
    print(f"wrote {sp.out_dir() / 'S8.pdf'}")


if __name__ == "__main__":
    main()
