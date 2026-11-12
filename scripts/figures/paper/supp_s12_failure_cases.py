#!/usr/bin/env python3
"""Supplementary Fig. S12 -- failure cases.

Panels
  a  The 116 chains whose G is 0 at every scored residue. G = 0 exactly when every
     shift in a residue's window sits at or beyond the 4-sigma deviation cap of the
     G-score, which a residue measured by one or two shifts reaches easily. The panel
     compares the per-residue shift count k of their scored residues with that of all
     released scored residues, and lists their splits.
  b  Training representatives with a LACS offset above 5 ppm on a 13C nucleus (nine)
     or on 15N only (three), plus the archive's largest offset (25442_1_2_2, excluded):
     for each affected nucleus, the median deviation of the deposited (open) and the
     re-referenced (filled) shifts from the archive-wide median of that residue type
     and nucleus. LACS fits one 13C offset for CA and CB together, so CA and CB are
     shown together whenever either exceeds 5 ppm. A correction that ends near 0
     repaired the deposition; one that moves a nucleus AWAY from 0 is a deposition
     error that a single shared offset cannot fix.
  c  One apomyoglobin sequence (153 aa), seven chains: acid-unfolded 4676 (pH 2.3,
     ``denatured``; a tolerant training representative), molten-globule 4568
     (pH 4.1, 323 K) and five early depositions whose G is 0 throughout (panel a).
  d  Ubiquitin entry 18611 (45 % methanol, 2,500 bar; ``physical_state`` unfolded)
     and the two chains of entry 18610 (2,500 bar, 258 K; native and denatured), one per
     row, each against the native ubiquitin reference (median and interquartile band of
     non-bound, solution, ``native`` chains of the same 76-residue sequence from other
     entries). The tiers at which each chain is scored are printed and written to
     S12_summary.json (``ubiquitin_tiers``).

Panel b labels each chain by its BMRB entry number; backbone atoms are written in Greek
(C', Calpha, Cbeta, N). G-score axes use the ticks of Fig. 2a (0, 0.5, 1).

Inputs (read-only): release Parquet + shifts Parquet, bundle per-tier scores.json IDs.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s12_failure_cases.py \
        --out-dir <draft>/Figures/paper
Outputs: <out-dir>/S12.pdf and <out-dir>/suppfigs_b_data/S12_*.csv / S12_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402

from trizod.offsets import lacs_off_ppm_col  # noqa: E402

# italic k in the same sans-serif face as the text (as Fig. 2a; mathtext defaults to DejaVu)
sp.plt.rcParams.update(
    {
        "mathtext.fontset": "custom",
        "mathtext.it": "Arial:italic",
        "mathtext.rm": "Arial",
    }
)

DATA_SUBDIR = "suppfigs_b_data"  # under --out-dir
NUCLEI = ["C", "CA", "CB", "N"]
EXTREME_PPM = 5.0
REFERENCE_CHAIN = (
    "25442_1_2_2"  # largest offset in the archive (excluded, not a representative)
)
MYO_CHAIN = "4676_1_1_1"
UBQ_CHAIN = "18611_1_1_1"
UBQ_HP = ["18610_1_1_1", "18610_2_2_1"]
COLUMNS = ["id", "entry_id", "sequence", "is_bound", "gscores", "k", "split", "train_tier",
           "physical_state", "ph", "temperature", "sample_state_evidence", "cosolvent_evidence",
           "bbshift_types_post", "n_scored", *[lacs_off_ppm_col(a) for a in NUCLEI]]  # fmt: skip


# ------------------------------------------------------------------------- panel a
def all_zero(df: pd.DataFrame):
    G = [np.asarray(g, float) for g in df.gscores]
    K = [np.asarray(k, float) for k in df.k]
    az = np.array([np.isfinite(g).any() and np.all(g[np.isfinite(g)] == 0) for g in G])
    k_az = np.concatenate([K[i][np.isfinite(G[i])] for i in np.flatnonzero(az)])
    k_all = np.concatenate([K[i][np.isfinite(G[i])] for i in range(len(df))])
    sub = df[az]
    info = {
        "n_chains": int(az.sum()),
        "split": sub.split.value_counts().to_dict(),
        "train_tier": sub.train_tier.value_counts().to_dict(),
        "k_scored_median": float(np.median(k_az)),
        "k_scored_share_k1": float(np.mean(k_az == 1)),
        "k_scored_share_le2": float(np.mean(k_az <= 2)),
        "k_all_released_median": float(np.median(k_all)),
        "bbshift_types_post": {
            int(k): int(v)
            for k, v in sub.bbshift_types_post.value_counts().sort_index().items()
        },
        "n_scored_median": float(sub.n_scored.median()),
        "share_entry_id_below_5000": float((sub.entry_id.astype(int) < 5000).mean()),
    }
    return az, k_az, k_all, info


# ------------------------------------------------------------------------- panel b
def extreme_cases(df: pd.DataFrame) -> pd.DataFrame:
    import pyarrow.parquet as pq

    reps = df[df.split == "train"]
    off = reps[[lacs_off_ppm_col(a) for a in NUCLEI]].abs()
    ext = reps[(off > EXTREME_PPM).any(axis=1)]
    ids = list(ext.id) + [REFERENCE_CHAIN]
    # archive-wide reference: median re-referenced shift per residue type and nucleus
    t = pq.read_table(sp.PQ_SHIFTS, columns=["comp_id", "atom_id", "val_corrected_ppm"],
                      filters=[("atom_id", "in", NUCLEI)]).to_pandas()  # fmt: skip
    ref = t.groupby(["comp_id", "atom_id"])["val_corrected_ppm"].median()
    s = pq.read_table(
        sp.PQ_SHIFTS, filters=[("id", "in", ids), ("atom_id", "in", NUCLEI)]
    ).to_pandas()
    s["ref"] = [ref.get((c, a), np.nan) for c, a in zip(s.comp_id, s.atom_id)]
    s["dev_dep"] = s.val_ppm - s.ref
    s["dev_cor"] = s.val_corrected_ppm - s.ref
    rows = []
    meta = df.set_index("id")
    for cid in ids:
        r = meta.loc[cid]
        big = [a for a in NUCLEI if abs(r[lacs_off_ppm_col(a)]) > EXTREME_PPM]
        if "CA" in big or "CB" in big:
            big = sorted(set(big) | {"CA", "CB"}, key=NUCLEI.index)
        for a in big:
            v = s[(s.id == cid) & (s.atom_id == a)]
            rows.append({"id": cid, "tier": r.train_tier if isinstance(r.train_tier, str) else r.split,
                         "nucleus": a, "lacs_ppm": float(r[lacs_off_ppm_col(a)]), "n_shifts": len(v),
                         "dev_deposited": float(v.dev_dep.median()), "dev_rereferenced": float(v.dev_cor.median())})  # fmt: skip
    E = pd.DataFrame(rows)
    E["repaired"] = E.dev_rereferenced.abs() < E.dev_deposited.abs()
    return E


# ---------------------------------------------------------------------- panels c, d
def profiles(df: pd.DataFrame):
    myo_seq = df.loc[df.id == MYO_CHAIN, "sequence"].iloc[0]
    myo = df[df.sequence == myo_seq].copy()
    ubq_seq = df.loc[df.id == UBQ_CHAIN, "sequence"].iloc[0]
    ubq = df[df.sequence == ubq_seq]
    ref = ubq[~ubq.is_bound.astype(bool) & (ubq.physical_state == "native")
              & (ubq.sample_state_evidence == "solution") & ~ubq.entry_id.isin(["18610", "18611"])]  # fmt: skip
    R = np.vstack([np.asarray(g, float) for g in ref.gscores])
    with np.errstate(all="ignore"):
        q = np.nanpercentile(R, [25, 50, 75], axis=0)
    return myo, ubq, ref, q


# ------------------------------------------------------------------------------ plot
NUC_LABEL = {"C": "C′", "CA": "Cα", "CB": "Cβ", "N": "N"}  # backbone atoms, Greek
G_TICKS = ([0, 0.5, 1.0], ["0", "0.5", "1"])  # as Fig. 2a


def plot(df, k_az, k_all, info_a, E, myo, ubq, ref, q, tier_ids):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    W, H = sp.W_DOUBLE, 140.0
    fig = sp.figure(W, H)

    def ax_mm(left, top, w, h):
        return fig.add_axes([left / W, 1 - (top + h) / H, w / W, h / H])

    ink = "#333333"

    # a ---------------------------------------------------------------
    axa = ax_mm(13, 8, 44, 44)
    bins = np.arange(0.5, 22.5, 1)
    axa.hist(k_all, bins=bins, density=True, histtype="step", color="#8C8C8C", lw=0.9,
             label="all released residues")  # fmt: skip
    axa.hist(k_az, bins=bins, density=True, color=ink, lw=0,
             label="chains with G = 0 throughout")  # fmt: skip
    axa.set_xlabel(r"Shifts in the residue's window, $\mathit{k}$")
    axa.set_ylabel("Share of scored residues")
    axa.set_xlim(0.5, 21.5)
    axa.set_xticks([1, 5, 10, 15, 21])
    yt = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    axa.set_yticks(yt, [f"{v:g}" for v in yt])
    axa.legend(loc="upper right", handlelength=1.2, borderaxespad=0.1)
    sp.fig_label(fig, "a", 0, 3)

    # b ---------------------------------------------------------------
    # one nucleus per row; the y labels name entry, tier and nucleus, so the marks need
    # no colour: open = deposited, filled = re-referenced, arrow = the LACS correction
    axb = ax_mm(88, 8, 70, 60)
    order = {"strict": 0, "moderate": 1, "tolerant": 2, "unfiltered": 3, "excluded": 4}
    E = E.copy()
    E["o"] = E.tier.map(order)
    E["absoff"] = E.groupby("id").lacs_ppm.transform(lambda v: v.abs().max())
    E["nuc_o"] = E.nucleus.map(NUCLEI.index)
    E = E.sort_values(
        ["o", "absoff", "id", "nuc_o"], ascending=[True, False, True, True]
    )
    y = np.arange(len(E))[::-1]
    labels, prev = [], None
    for yy, (_, r) in zip(y, E.iterrows()):
        axb.annotate("", xy=(r.dev_rereferenced, yy), xytext=(r.dev_deposited, yy),
                     arrowprops={"arrowstyle": "-|>", "color": ink, "lw": 0.6, "mutation_scale": 5, "shrinkA": 1.8, "shrinkB": 1.8})  # fmt: skip
        axb.plot([r.dev_deposited], [yy], marker="o", ms=3.2, mfc="white", mec=ink,
                 mew=0.7, ls="none")  # fmt: skip
        axb.plot([r.dev_rereferenced], [yy], marker="o", ms=3.2, color=ink, ls="none")
        # plain entry numbers (the excluded reference chain is entry 25442's second entity)
        lab = f"{r.id.split('_')[0]} ({r.tier})" if r.id != prev else ""
        nuc = NUC_LABEL[r.nucleus]
        labels.append(f"{lab}  {nuc}" if lab else nuc)
        prev = r.id
    axb.set_yticks(y)
    axb.set_yticklabels(labels, fontsize=sp.FS_SMALL)
    axb.tick_params(axis="y", length=0)
    axb.set_xscale("symlog", linthresh=2)
    axb.axvspan(-2, 2, color="#EFEFEF", lw=0, zorder=0)
    axb.axvline(0, color="#9a9a9a", lw=0.4)
    axb.set_xlim(-250, 450)
    axb.set_xticks([-100, -10, -2, 0, 2, 10, 100])
    axb.set_xticklabels(["−100", "−10", "−2", "0", "2", "10", "100"])
    axb.xaxis.set_minor_locator(sp.mticker.NullLocator())
    axb.set_xlabel("Deviation from the archive median (ppm)")
    axb.set_ylim(-0.7, len(E) - 0.3)
    hand = [Line2D([], [], marker="o", ms=3.2, mfc="white", mec=ink, mew=0.7, ls="none", label="deposited"),
            Line2D([], [], marker="o", ms=3.2, color=ink, ls="none", label="re-referenced")]  # fmt: skip
    axb.legend(handles=hand, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
               borderaxespad=0.2, handletextpad=0.2, columnspacing=1.2)  # fmt: skip
    sp.fig_label(fig, "b", 62, 3)

    # c ---------------------------------------------------------------
    # training chain in bluish green, excluded chain in vermillion (fixed meanings)
    axc = ax_mm(13, 84, 66, 44)
    myo = myo.copy()
    myo["meanG"] = [np.nanmean(np.asarray(g, float)) for g in myo.gscores]
    split_word = {"train": "training", "excluded": "excluded", "redundant": "redundant"}
    for _, r in myo.sort_values("meanG").iterrows():
        g = np.asarray(r.gscores, float)
        x = np.arange(1, len(g) + 1)
        if r.id == MYO_CHAIN:
            col = sp.CAT["train"] if r.split == "train" else sp.CAT["neutral"]
            axc.plot(x, g, color=col, lw=0.9, zorder=3,
                     label=f"{r.entry_id}: {r.physical_state}, pH {r.ph:.1f} ({split_word.get(r.split, r.split)})")  # fmt: skip
        elif r.physical_state == "molten globule":
            col = sp.CAT["removed"] if r.split == "excluded" else sp.CAT["neutral"]
            axc.plot(x, g, color=col, lw=0.9, zorder=2,
                     label=f"{r.entry_id}: molten globule, pH {r.ph:.1f} ({split_word.get(r.split, r.split)})")  # fmt: skip
        else:
            axc.plot(x, g, color=ink, lw=0, marker="|", ms=3, mew=0.6, zorder=1)
    early = myo[(myo.id != MYO_CHAIN) & (myo.physical_state != "molten globule")]
    axc.plot([], [], color=ink, lw=0, marker="|", ms=3, mew=0.6,
             label=f"{len(early)} early holo-myoglobin entries")  # fmt: skip
    axc.axhline(sp.G_DIS, color=sp.CAT["light"], lw=0.5, ls=":")
    axc.set_xlim(0, len(np.asarray(myo.gscores.iloc[0])) + 1)
    axc.set_ylim(-0.03, 1.36)
    axc.set_yticks(*G_TICKS)
    axc.set_xlabel("Residue (myoglobin)")
    axc.set_ylabel("G-score")
    axc.legend(loc="upper left", borderaxespad=0.1, handlelength=1.4, ncol=1)
    sp.fig_label(fig, "c", 0, 79)

    # d ---------------------------------------------------------------
    # small multiples: one perturbed ubiquitin chain per row, each against the native
    # reference (median and interquartile band of native solution chains)
    um = ubq.set_index("id")
    rows = [(UBQ_HP[0], None), (UBQ_HP[1], None), (UBQ_CHAIN, "methanol")]
    x = np.arange(1, q.shape[1] + 1)
    dl, dw, dh, dgap, dtop = 95.0, 63.0, 12.0, 4.0, 84.0
    for i, (cid, what) in enumerate(rows):
        ax = ax_mm(dl, dtop + i * (dh + dgap), dw, dh)
        ax.fill_between(x, q[0], q[2], color="#D9D9D9", lw=0)
        ax.plot(x, q[1], color="#8C8C8C", lw=0.8)
        r = um.loc[cid]
        g = np.asarray(r.gscores, float)
        ax.plot(np.arange(1, len(g) + 1), g, color="black", lw=0.9, zorder=3)
        ax.axhline(sp.G_DIS, color=sp.CAT["light"], lw=0.5, ls=":")
        ax.set_xlim(0, 77)
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks(*G_TICKS)
        lab = (f"{r.entry_id}, in methanol" if what == "methanol"
               else f"{r.entry_id}, {r.physical_state}")  # fmt: skip
        ax.text(0.01, 1.0, lab, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=sp.FS_TICK)  # fmt: skip
        if i < len(rows) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Residue (ubiquitin)")
        if i == 1:
            ax.set_ylabel("G-score")
    # key for the reference, above the first row on the right
    fig.legend(handles=[(Patch(color="#D9D9D9", lw=0), Line2D([], [], color="#8C8C8C", lw=0.8)),
                        Line2D([], [], color="black", lw=0.9)],
               labels=["native reference", "chain"],
               handler_map={tuple: _tuple_handler()}, loc="lower right",
               bbox_to_anchor=((dl + dw) / W, 1 - (dtop - 0.3) / H), ncol=2, borderaxespad=0,
               handlelength=1.6, columnspacing=1.2, fontsize=sp.FS_TICK)  # fmt: skip
    sp.fig_label(fig, "d", 84, 79)
    return fig


def _tuple_handler():
    from matplotlib.legend_handler import HandlerTuple

    return HandlerTuple(ndivide=1, pad=0)


def main():
    sp.parse_args(sp.cli(__doc__))
    DATA_DIR = sp.out_dir(DATA_SUBDIR)
    df = sp.load_main(COLUMNS)
    tier_ids = sp.load_tier_ids()
    az, k_az, k_all, info_a = all_zero(df)
    E = extreme_cases(df)
    myo, ubq, ref, q = profiles(df)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.loc[az, ["id", "entry_id", "split", "train_tier", "n_scored", "bbshift_types_post"]].to_csv(
        DATA_DIR / "S12_all_zero_g_chains.csv", index=False)  # fmt: skip
    E.to_csv(DATA_DIR / "S12_extreme_offsets.csv", index=False)
    summ = {
        "all_zero_g": info_a,
        "extreme_offsets": {
            "n_reps_13C": int(E[(E.id != REFERENCE_CHAIN) & E.nucleus.isin(["C", "CA", "CB"]) & (E.lacs_ppm.abs() > EXTREME_PPM)].id.nunique()),
            "n_reps_any": int(E[E.id != REFERENCE_CHAIN].id.nunique()),
            "nucleus_rows_moved_away_from_archive_median": E.loc[~E.repaired, ["id", "nucleus", "dev_deposited", "dev_rereferenced"]].round(2).to_dict("records"),
        },
        "myoglobin_group": [{"id": r.id, "physical_state": r.physical_state, "ph": float(r.ph), "split": r.split,
                             "train_tier": r.train_tier, "meanG": round(float(np.nanmean(np.asarray(r.gscores, float))), 3),
                             "n_scored": int(r.n_scored)} for _, r in myo.iterrows()],
        "ubiquitin_reference_chains": int(len(ref)),
        "ubiquitin_18611_scored_tiers": [t for t in sp.TIERS if UBQ_CHAIN in tier_ids[t]],
        "ubiquitin_tiers": {c: [t for t in sp.TIERS if c in tier_ids[t]] for c in [*UBQ_HP, UBQ_CHAIN]},
        "ubiquitin_18611_meanG": round(float(np.nanmean(np.asarray(ubq.set_index("id").loc[UBQ_CHAIN, "gscores"], float))), 3),
        "ubiquitin_reference_median_meanG": round(float(np.nanmean(q[1])), 3),
    }  # fmt: skip
    (DATA_DIR / "S12_summary.json").write_text(json.dumps(summ, indent=2, default=str))
    out = sp.save(plot(df, k_az, k_all, info_a, E, myo, ubq, ref, q, tier_ids), "S12")
    pd.set_option("display.width", 220)
    print(out)
    print(json.dumps(summ, indent=1, default=str))
    print(E.round(2).to_string())


if __name__ == "__main__":
    main()
