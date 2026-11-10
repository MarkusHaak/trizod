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
     and entry 18610 (2,500 bar, 258 K) against the native ubiquitin reference
     (median and interquartile band of non-bound, solution, ``native`` chains of the
     same 76-residue sequence from other entries). Below: the tiers at which 18611 is
     scored in the frozen build, and where the planned fixes (b) and (e) would leave it
     (expected from the filter logic, not recomputed). On a later build
     (``style_paper.IS_FROZEN`` false) the second row is that build's own scored tiers
     and the "expected" rows are dropped.

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

DATA_SUBDIR = "suppfigs_b_data"  # under --out-dir
NUCLEI = ["C", "CA", "CB", "N"]
NUC_COLOR = {"C": sp.OKABE_ITO["orange"], "CA": sp.OKABE_ITO["blue"], "CB": sp.OKABE_ITO["skyblue"],
             "N": sp.OKABE_ITO["green"]}  # fmt: skip
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


TIER_REACH = [
    ("frozen 2026-10 build", ["unfiltered", "tolerant", "moderate"]),
    ("fix (b) evidence tokens, or (e) pressure ceiling", ["unfiltered", "tolerant"]),
    ("fix (b) cosolvent-filter tokens", ["unfiltered"]),
]


# ------------------------------------------------------------------------------ plot
def plot(df, k_az, k_all, info_a, E, myo, ubq, ref, q, tier_ids):
    from matplotlib.lines import Line2D

    fig = sp.figure(sp.W_DOUBLE, 128)

    # a ---------------------------------------------------------------
    axa = fig.add_axes([0.065, 0.62, 0.25, 0.33])
    bins = np.arange(0.5, 22.5, 1)
    axa.hist(k_all, bins=bins, density=True, histtype="step", color=sp.CAT["neutral"], lw=0.8,
             label=f"all released residues (median k {np.median(k_all):.0f})")  # fmt: skip
    axa.hist(k_az, bins=bins, density=True, color=sp.CAT["removed"], alpha=0.85, lw=0,
             label=f"{info_a['n_chains']} all-zero-G chains (median k {np.median(k_az):.0f})")  # fmt: skip
    axa.set_xlabel("Shifts in the residue's window, k")
    axa.set_ylabel("Fraction of scored residues")
    axa.set_xlim(0.5, 21.5)
    axa.set_xticks([1, 5, 10, 15, 21])
    axa.legend(loc="upper right", handlelength=1.2, borderaxespad=0.1)
    sp_ = info_a["split"]
    axa.text(0.97, 0.72, f"split of the {info_a['n_chains']} chains\n"
             f"excluded {sp_.get('excluded', 0)}\ntraining {sp_.get('train', 0)}\nredundant {sp_.get('redundant', 0)}",
             transform=axa.transAxes, ha="right", va="top", fontsize=5.5, linespacing=1.3)  # fmt: skip
    sp.panel_label(axa, "a", x=-0.25, y=1.05)

    # b ---------------------------------------------------------------
    axb = fig.add_axes([0.505, 0.56, 0.47, 0.38])
    order = {"strict": 0, "moderate": 1, "tolerant": 2, "unfiltered": 3, "excluded": 4}
    E = E.copy()
    E["o"] = E.tier.map(order)
    E["absoff"] = E.groupby("id").lacs_ppm.transform(lambda v: v.abs().max())
    E = E.sort_values(
        ["o", "absoff", "id", "nucleus"], ascending=[True, False, True, True], key=None
    )
    E["nuc_o"] = E.nucleus.map(NUCLEI.index)
    E = E.sort_values(
        ["o", "absoff", "id", "nuc_o"], ascending=[True, False, True, True]
    )
    y = np.arange(len(E))[::-1]
    labels, prev = [], None
    for yy, (_, r) in zip(y, E.iterrows()):
        c = NUC_COLOR[r.nucleus]
        axb.annotate("", xy=(r.dev_rereferenced, yy), xytext=(r.dev_deposited, yy),
                     arrowprops={"arrowstyle": "-|>", "color": c, "lw": 0.6, "mutation_scale": 5, "shrinkA": 1.5, "shrinkB": 1.5})  # fmt: skip
        axb.plot(
            [r.dev_deposited],
            [yy],
            marker="o",
            ms=3,
            mfc="white",
            mec=c,
            mew=0.7,
            ls="none",
        )
        axb.plot([r.dev_rereferenced], [yy], marker="o", ms=3, color=c, ls="none")
        lab = (
            f"{r.id.rsplit('_', 3)[0] if r.id != REFERENCE_CHAIN else r.id} ({r.tier})"
            if r.id != prev
            else ""
        )
        labels.append(f"{lab}  {r.nucleus}" if lab else r.nucleus)
        prev = r.id
    axb.set_yticks(y)
    axb.set_yticklabels(labels, fontsize=5)
    axb.set_xscale("symlog", linthresh=2)
    axb.axvspan(-2, 2, color="#EFEFEF", lw=0, zorder=0)
    axb.axvline(0, color="#9a9a9a", lw=0.4)
    axb.set_xlim(-250, 450)
    axb.set_xticks([-100, -10, -2, 0, 2, 10, 100])
    axb.set_xticklabels(["−100", "−10", "−2", "0", "2", "10", "100"])
    axb.set_xlabel("Median deviation from the archive median of the residue type (ppm)")
    axb.set_ylim(-0.7, len(E) - 0.3)
    hand = [Line2D([], [], marker="o", ms=3, mfc="white", mec="#555555", ls="none", label="deposited"),
            Line2D([], [], marker="o", ms=3, color="#555555", ls="none", label="re-referenced")] + [
            Line2D([], [], marker="s", ms=3, color=NUC_COLOR[a], ls="none", label=a) for a in NUCLEI]  # fmt: skip
    axb.legend(
        handles=hand,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=6,
        borderaxespad=0.2,
        handletextpad=0.2,
        columnspacing=1.0,
    )
    sp.panel_label(axb, "b", x=-0.36, y=1.04)

    # c ---------------------------------------------------------------
    axc = fig.add_axes([0.065, 0.085, 0.37, 0.36])
    myo = myo.copy()
    myo["meanG"] = [np.nanmean(np.asarray(g, float)) for g in myo.gscores]
    for _, r in myo.sort_values("meanG").iterrows():
        g = np.asarray(r.gscores, float)
        x = np.arange(1, len(g) + 1)
        if r.id == MYO_CHAIN:
            axc.plot(x, g, color=sp.CAT["removed"], lw=0.8, zorder=3,
                     label=f"{r.entry_id}: {r.physical_state}, pH {r.ph:.1f} ({r.split}, {r.train_tier})")  # fmt: skip
        elif r.physical_state == "molten globule":
            axc.plot(x, g, color=sp.OKABE_ITO["orange"], lw=0.8, zorder=2,
                     label=f"{r.entry_id}: molten globule, pH {r.ph:.1f}, {r.temperature:.0f} K ({r.split})")  # fmt: skip
        else:
            axc.plot(
                x, g, color=sp.CAT["neutral"], lw=0, marker="|", ms=3, mew=0.6, zorder=1
            )
    early = myo[(myo.id != MYO_CHAIN) & (myo.physical_state != "molten globule")]
    axc.plot([], [], color=sp.CAT["neutral"], lw=0, marker="|", ms=3, mew=0.6,
             label=f"{len(early)} early holo-myoglobin 1H entries, ≤ {early.n_scored.max()} scored residues "
                   f"({int((early.meanG == 0).sum())} with G = 0 throughout)")  # fmt: skip
    axc.axhline(sp.G_DIS, color=sp.CAT["light"], lw=0.5, ls=":")
    axc.set_xlim(0, len(np.asarray(myo.gscores.iloc[0])) + 1)
    axc.set_ylim(-0.03, 1.25)
    axc.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axc.set_xlabel("Residue (myoglobin, 153 aa)")
    axc.set_ylabel("G-score")
    axc.legend(loc="upper left", borderaxespad=0.1, handlelength=1.4, fontsize=5)
    sp.panel_label(axc, "c", x=-0.17, y=1.03)

    # d ---------------------------------------------------------------
    axd = fig.add_axes([0.55, 0.2, 0.43, 0.25])
    x = np.arange(1, q.shape[1] + 1)
    axd.fill_between(
        x,
        q[0],
        q[2],
        color="#D9D9D9",
        lw=0,
        label=f"native reference, IQR ({len(ref)} chains)",
    )
    axd.plot(x, q[1], color=sp.CAT["neutral"], lw=0.8, label="native reference, median")
    um = ubq.set_index("id")
    for cid, col, ls in ((UBQ_HP[0], sp.OKABE_ITO["blue"], "-"), (UBQ_HP[1], sp.OKABE_ITO["skyblue"], "-"),
                         (UBQ_CHAIN, sp.CAT["removed"], "-")):  # fmt: skip
        r = um.loc[cid]
        g = np.asarray(r.gscores, float)
        lab = (f"18611: 45 % methanol, 2,500 bar ({r.physical_state})" if cid == UBQ_CHAIN
               else f"{cid}: 2,500 bar, {r.temperature:.0f} K ({r.physical_state})")  # fmt: skip
        axd.plot(
            np.arange(1, len(g) + 1), g, color=col, lw=0.8, ls=ls, label=lab, zorder=3
        )
    axd.axhline(sp.G_DIS, color=sp.CAT["light"], lw=0.5, ls=":")
    axd.set_xlim(0, 77)
    axd.set_ylim(-0.03, 1.32)
    axd.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axd.set_xlabel("Residue (ubiquitin, 76 aa)")
    axd.set_ylabel("G-score")
    axd.legend(
        loc="upper left",
        ncol=2,
        borderaxespad=0.1,
        handlelength=1.4,
        fontsize=5,
        columnspacing=0.8,
    )
    sp.panel_label(axd, "d", x=-0.13, y=1.03)
    # tier-reach strip
    axs = fig.add_axes([0.55, 0.012, 0.43, 0.085])
    scored_now = [t for t in sp.TIERS if UBQ_CHAIN in tier_ids[t]]
    if sp.IS_FROZEN:
        assert scored_now == TIER_REACH[0][1], scored_now
        reach_rows, expected = TIER_REACH, True
    else:
        # a later build has the fixes applied: the frozen row (pinned) and the build's own
        reach_rows, expected = (
            [TIER_REACH[0], (f"{sp.RELEASE} build", scored_now)],
            False,
        )
    for i, (name, reach) in enumerate(reach_rows):
        yy = len(TIER_REACH) - 1 - i
        for j, t in enumerate(sp.TIERS):
            on = t in reach
            axs.add_patch(__import__("matplotlib").patches.Rectangle(
                (j * 0.9, yy + 0.1), 0.8, 0.8, facecolor=sp.TIER_COLORS[t] if on else "white",
                edgecolor=sp.TIER_COLORS[t], lw=0.6))  # fmt: skip
        axs.text(
            3.75,
            yy + 0.5,
            name + (" (expected)" if i and expected else ""),
            va="center",
            ha="left",
            fontsize=5,
        )
    for j, t in enumerate(sp.TIERS):
        axs.text(j * 0.9 + 0.4, 3.15, {"unfiltered": "unf.", "tolerant": "tol.", "moderate": "mod.", "strict": "strict"}[t],
                 ha="center", va="bottom", fontsize=4.6)  # fmt: skip
    axs.set_xlim(-0.1, 14)
    axs.set_ylim(0, 3.9)
    axs.axis("off")
    axs.text(-0.15, 1.5, "18611\nscored at", ha="right", va="center", fontsize=5)
    return fig


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
