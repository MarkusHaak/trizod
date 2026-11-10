#!/usr/bin/env python3
"""Fig. 3c -- the held-out sets: split integrity (c1) and disorder content (c2).

(c1) Split integrity of the FROZEN 2026-10 split, i.e. the state BEFORE rebuild fix (a).
For each of the 575 held-out chains (460 TriZOD test + 115 CheZOD117), a permissive
``mmseqs easy-search`` against the training representatives of each tier
(``train/<tier>/train_<tier>_best.fasta``) returns every alignment with identity >= 20 %
and e < 1e-3, with no coverage filter. The search uses the build's own ``COMMON`` block
(``trizod/dataset/mmseqs.py``: ``--alignment-mode 3 -s 7.5 --comp-bias-corr 0 --mask 0``),
so identities and coverages are those the corrected stage-2 criterion will see.

The shaded region is what that corrected criterion excludes: identity >= 0.30 AND >= 80 %
of the TEST chain covered (``qcov``). Each chain is drawn at one alignment, taken from the
union of the four searches:

* if any alignment lies in the region, the in-region one whose representative belongs to
  the strictest tier (ties: highest identity);
* otherwise the alignment with the largest identity x test-chain coverage.

Points are coloured by the strictest tier at which the plotted representative is a
training representative (tiers are nested). Counts inside the region are given per
CUMULATIVE tier, from that tier's own search (e-values depend on the target-set size, so
the per-tier searches find slightly more than the unfiltered search restricted to the
tier). The pipeline's current (bidirectional) definition, a 50 %/50 % and a 90 %/50 %
definition are counted alongside in ``fig3c_numbers.json``. Chains without any alignment
are not drawn.

Colour roles in c1 (revised 31 Oct, format review): orange and blue mean only the two
held-out sets (margin histograms, column heads, c2 row labels); the excluded region and
its bounds are neutral grey (light grey fill + grey hatch, grey dashed lines), so the
region no longer shares a hue with CheZOD117. Every point is filled with its tier colour
and carries a ``EDGE_LW`` dark edge, so the pale unfiltered tier stays visible on white;
the two sets differ by marker shape (circle = TriZOD test, triangle = CheZOD117). Tier
names in the counts table are set in dark text (the tier legend carries the colours).

(c2) One stacked strip of per-chain disorder class -- share of a chain's scored residues
with G > 0.65 in <= 10 % / 10-90 % / >= 90 % -- for CheZOD117, the TriZOD test set and the
training representatives of every tier, each with its own tier's labels
(``scores/<tier>/scores.json``; the test sets use the unfiltered labels, as released).

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig3c_split.py \
        --out-dir <draft>/Figures/paper [--work-dir <scratch>/fig3]
Outputs: <out-dir>/fig3c.pdf; <out-dir>/fig3_data/fig3c1_points.csv,
fig3c2_strip.csv and fig3c_numbers.json. The mmseqs hits are cached in ``--work-dir``
(default ``tmp/fig3c_split`` in the code checkout).
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

FS_SMALL = 5.0
# c1 colour roles: the excluded region is neutral grey, never a category hue
REGION_FILL = "#F2F2F2"  # light grey fill ("shaded" in the legend)
REGION_HATCH = sp.CAT["light"]  # grey hatch over the fill
REGION_LINE = "#7F7F7F"  # grey dashed bounds (scatter and margins)
REGION_TEXT = sp.CAT["neutral"]
EDGE_COLOR = "#262626"  # dark marker edge, keeps the pale unfiltered tier visible
EDGE_LW = 0.3  # pt
ID_MIN, COV_MIN = (
    0.30,
    0.80,
)  # corrected stage-2 criterion: identity, TEST-chain coverage
MMSEQS_ARGS = [
    "--min-seq-id", "0.2", "-c", "0", "-e", "1e-3",
    # the build's COMMON block (trizod/dataset/mmseqs.py), so identities are comparable
    "--alignment-mode", "3", "-s", "7.5", "--comp-bias-corr", "0", "--mask", "0",
    "--max-seqs", "5000",
]  # fmt: skip
FMT = (
    "query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,qcov,tcov"
)
TEST_FASTAS = {
    "test_trizod": sp.BUNDLE / "test" / "TriZOD_test_set.fasta",
    "test_chezod117": sp.BUNDLE / "test" / "CheZOD117_test_set.fasta",
}
RULES = {
    "test_coverage_30_80": lambda h: (h.fident >= ID_MIN) & (h.qcov >= COV_MIN),
    "bidirectional_30_80": lambda h: (h.fident >= ID_MIN) & (h.qcov >= COV_MIN) & (h.tcov >= COV_MIN),
    "id50_testcov50": lambda h: (h.fident >= 0.50) & (h.qcov >= 0.50),
    "id90_testcov50": lambda h: (h.fident >= 0.90) & (h.qcov >= 0.50),
}  # fmt: skip


def mmseqs_version() -> str:
    try:
        return subprocess.run(
            ["mmseqs", "version"], capture_output=True, text=True
        ).stdout.strip()
    except FileNotFoundError:
        return "missing"


def run_search(work: Path, refresh: bool) -> pd.DataFrame:
    """Combined test FASTA vs each tier's representatives; cached per tier in ``work``."""
    work.mkdir(parents=True, exist_ok=True)
    query = work / "test_all.fasta"
    with open(query, "w") as fh:
        for path in TEST_FASTAS.values():
            for k, s in sp.read_fasta(path).items():
                fh.write(f">{k}\n{s}\n")
    frames = []
    for t in sp.TIERS:
        out = work / f"hits_{t}.m8"
        if refresh or not out.exists():
            if shutil.which("mmseqs") is None:
                raise SystemExit("mmseqs not on PATH")
            target = sp.BUNDLE / "train" / t / f"train_{t}_best.fasta"
            tmp = work / f"tmp_{t}"
            cmd = ["mmseqs", "easy-search", str(query), str(target), str(out), str(tmp),
                   *MMSEQS_ARGS, "--format-output", FMT, "--threads", "8", "-v", "1"]  # fmt: skip
            print(" ".join(cmd), file=sys.stderr)
            subprocess.run(cmd, check=True)
            shutil.rmtree(tmp, ignore_errors=True)
        h = pd.read_csv(out, sep="\t", names=FMT.split(","))
        h["search_tier"] = t
        frames.append(h)
    return pd.concat(frames, ignore_index=True)


def analyse(hits: pd.DataFrame, set_of: dict, rep_tier: dict):
    hits = hits.copy()
    hits["set"] = hits["query"].map(set_of)
    hits["in_region"] = RULES["test_coverage_30_80"](hits)
    # union of the four searches: one row per (query, target), best-scoring alignment
    uni = (
        hits.sort_values("bits", ascending=False)
        .drop_duplicates(["query", "target"])
        .reset_index(drop=True)
    )
    uni["ttier"] = uni["target"].map(rep_tier)
    uni["trank"] = uni["ttier"].map(sp.RANK)
    uni["score"] = uni.fident * uni.qcov

    rows = []
    for q, s in set_of.items():
        h = uni[uni["query"] == q]
        if h.empty:
            rows.append({"id": q, "set": s, "has_hit": False, "in_region": False})
            continue
        inr = h[h.in_region]
        if len(inr):
            best = inr.sort_values(["trank", "fident", "qcov"], ascending=False).iloc[0]
        else:
            best = h.sort_values(["score", "fident"], ascending=False).iloc[0]
        rows.append({
            "id": q, "set": s, "has_hit": True, "target": best.target, "fident": best.fident,
            "qcov": best.qcov, "tcov": best.tcov, "qlen": best.qlen, "tlen": best.tlen,
            "evalue": best.evalue, "ttier": best.ttier, "in_region": bool(best.in_region),
        })  # fmt: skip
    P = pd.DataFrame(rows)

    counts = {}
    for t in sp.TIERS:
        own = hits[hits.search_tier == t]
        restricted = uni[uni.trank >= sp.RANK[t]]
        c = {}
        for s in TEST_FASTAS:
            ss = own[own.set == s]
            c[s] = {
                name: int(ss[rule(ss)]["query"].nunique())
                for name, rule in RULES.items()
            }
            c[s]["any_hit"] = int(ss["query"].nunique())
            rr = restricted[restricted.set == s]
            c[s]["test_coverage_30_80_from_union"] = int(
                rr[rr.in_region]["query"].nunique()
            )
        c["reps_in_region_any_test_chain"] = int(own[own.in_region]["target"].nunique())
        c["reps_bidirectional_any_test_chain"] = int(
            own[RULES["bidirectional_30_80"](own)]["target"].nunique()
        )
        counts[t] = c
    return uni, P, counts


def disorder_strip(test_ids: dict, reps: dict):
    df = sp.load_main(["id", "gscores"])
    g_unf = dict(zip(df["id"], df["gscores"]))
    tier_g = {t: sp.load_tier_scores(t, fields=("gscores",)) for t in sp.TIERS}

    def dis_frac(g) -> float:
        g = np.asarray(g, float)
        g = g[np.isfinite(g)]
        return np.nan if g.size == 0 else float((g > sp.G_DIS).mean())

    def res_share(gs) -> float:
        a = np.concatenate([np.asarray(g, float) for g in gs])
        a = a[np.isfinite(a)]
        return float(100 * (a > sp.G_DIS).mean())

    sets = [
        ("CheZOD117", [g_unf[i] for i in test_ids["test_chezod117"]], sp.CAT["chezod"]),
        ("TriZOD test", [g_unf[i] for i in test_ids["test_trizod"]], sp.CAT["test"]),
    ]
    for t in sp.TIERS:
        sets.append(
            (
                f"Training, {t}",
                [tier_g[t][i]["gscores"] for i in sorted(reps[t])],
                sp.CAT["train"],
            )
        )
    strip = []
    for name, gs, col in sets:
        fr = np.array([dis_frac(g) for g in gs])
        fr = fr[np.isfinite(fr)]
        sh = [
            (fr <= 0.10).mean(),
            ((fr > 0.10) & (fr < 0.90)).mean(),
            (fr >= 0.90).mean(),
        ]
        strip.append({
            "set": name, "n": int(fr.size), "shares": [float(x) for x in sh], "color": col,
            "pct_residues_G_gt_065": round(res_share(gs), 2),
            "chains_gt_50pct_disordered": int((fr > 0.5).sum()),
        })  # fmt: skip
    return strip


def draw(P: pd.DataFrame, counts: dict, strip: list, n_total: int):
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    W, H = sp.W_SINGLE, 80.0
    fig = sp.figure(W, H)

    def ax_mm(x, y, w, h, **kw):
        return fig.add_axes([x / W, y / H, w / W, h / H], **kw)

    # ------------------------------------------------------------------ c1 scatter
    X0, Y0, AW, AH = 11.0, 37.0, 42.0, 33.0
    ax = ax_mm(X0, Y0, AW, AH)
    axx = ax_mm(X0, Y0 + AH + 0.6, AW, 5.5, sharex=ax)
    axy = ax_mm(X0 + AW + 0.6, Y0, 5.5, AH, sharey=ax)
    region = mpatches.Rectangle((COV_MIN, ID_MIN), 1.03 - COV_MIN, 1.03 - ID_MIN, facecolor=REGION_FILL,
                                edgecolor=REGION_HATCH, hatch="////", lw=0, zorder=0)  # fmt: skip
    # edgecolor sets the hatch colour; lw=0 hides the outline
    region.set_hatch_linewidth(0.4)
    ax.add_patch(region)
    ax.plot(
        [COV_MIN, COV_MIN, 1.03],
        [1.03, ID_MIN, ID_MIN],
        color=REGION_LINE,
        lw=0.6,
        ls="--",
        zorder=1,
    )
    ax.text(COV_MIN - 0.015, 0.66, "excluded by the\ncorrected criterion", rotation=90, ha="right",
            va="center", fontsize=FS_SMALL, color=REGION_TEXT, linespacing=1.0)  # fmt: skip
    pts = P[P.has_hit].copy()
    rng = np.random.default_rng(0)
    for s, marker, size in (
        ("test_trizod", "o", (4.0, 6.5)),
        ("test_chezod117", "^", (6.0, 9.0)),
    ):
        for t in sp.TIERS:
            for inr in (False, True):
                sub = pts[(pts.set == s) & (pts.ttier == t) & (pts.in_region == inr)]
                if sub.empty:
                    continue
                x = np.clip(
                    sub.qcov.to_numpy() + rng.uniform(-0.004, 0.004, len(sub)), 0, 1.0
                )
                y = np.clip(
                    sub.fident.to_numpy() + rng.uniform(-0.004, 0.004, len(sub)), 0, 1.0
                )
                ax.scatter(
                    x, y, s=size[inr], marker=marker, facecolor=sp.TIER_COLORS[t],
                    edgecolor=EDGE_COLOR, linewidths=EDGE_LW, alpha=0.95 if inr else 0.75,
                    zorder=(3 if inr else 2) + (s == "test_chezod117"),
                )  # fmt: skip
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(0.17, 1.03)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Fraction of the test chain covered")
    ax.set_ylabel("Identity to the closest training representative")
    bins_x = np.linspace(0, 1, 26)
    bins_y = np.linspace(0.2, 1, 21)
    for s, col in (
        ("test_trizod", sp.CAT["test"]),
        ("test_chezod117", sp.CAT["chezod"]),
    ):
        sub = pts[pts.set == s]
        axx.hist(
            sub.qcov, bins=bins_x, histtype="step", color=col, lw=0.6, density=True
        )
        axy.hist(sub.fident, bins=bins_y, histtype="step", color=col, lw=0.6, density=True,
                 orientation="horizontal")  # fmt: skip
    for a in (axx, axy):
        for side in ("top", "right", "left", "bottom"):
            a.spines[side].set_visible(False)
        a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axx.axvline(COV_MIN, color=REGION_LINE, lw=0.5, ls="--")
    axy.axhline(ID_MIN, color=REGION_LINE, lw=0.5, ls="--")

    # right column: legend + counts
    xr = X0 + AW + 8.0
    h_t = [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=2.8,
            mfc=sp.TIER_COLORS[t],
            mec=EDGE_COLOR,
            mew=EDGE_LW,
            label=t,
        )
        for t in sp.TIERS
    ]
    h_s = [
        Line2D(
            [],
            [],
            ls="",
            marker=m,
            ms=ms,
            mfc=sp.CAT["light"],
            mec=EDGE_COLOR,
            mew=EDGE_LW,
            label=lab,
        )
        for m, ms, lab in (("o", 2.8, "TriZOD test"), ("^", 3.2, "CheZOD117"))
    ]
    leg1 = fig.legend(handles=h_t, loc="upper left", bbox_to_anchor=(xr / W, 77.5 / H), fontsize=FS_SMALL,
                      title="Tier of representative", title_fontsize=FS_SMALL, alignment="left",
                      handletextpad=0.2, labelspacing=0.15, borderaxespad=0.0)  # fmt: skip
    fig.add_artist(leg1)
    leg2 = fig.legend(handles=h_s, loc="upper left", bbox_to_anchor=(xr / W, 61.5 / H), fontsize=FS_SMALL,
                      title="Test set (margins)", title_fontsize=FS_SMALL, alignment="left",
                      handletextpad=0.2, labelspacing=0.15, borderaxespad=0.0)  # fmt: skip
    for txt, col in zip(leg2.get_texts(), (sp.CAT["test"], sp.CAT["chezod"])):
        txt.set_color(col)
    n_nohit = int((~P.has_hit).sum())
    fig.text(xr / W, 52.5 / H, f"Not shown: {n_nohit} of {n_total}\nchains without an\nalignment", fontsize=FS_SMALL,
             color=sp.CAT["neutral"], va="top", linespacing=1.05)  # fmt: skip
    # counts table (pre-fix)
    yt = 44.0
    fig.text(xr / W, yt / H, "In shaded region\n(frozen split, pre-fix)", fontsize=FS_SMALL,
             color=REGION_TEXT, va="top", linespacing=1.05)  # fmt: skip
    cx = [xr + 13.5, xr + 25.0]
    yy = yt - 6.0
    fig.text(
        cx[0] / W,
        yy / H,
        "TriZOD",
        fontsize=FS_SMALL,
        ha="right",
        va="center",
        color=sp.CAT["test"],
    )
    fig.text(
        cx[1] / W,
        yy / H,
        "CheZOD117",
        fontsize=FS_SMALL,
        ha="right",
        va="center",
        color=sp.CAT["chezod"],
    )
    for i, t in enumerate(sp.TIERS):
        y_ = yy - 2.3 * (i + 1)
        fig.text(xr / W, y_ / H, t, fontsize=FS_SMALL, va="center", color="black")
        fig.text(cx[0] / W, y_ / H, f"{counts[t]['test_trizod']['test_coverage_30_80']}", fontsize=FS_SMALL,
                 ha="right", va="center")  # fmt: skip
        fig.text(cx[1] / W, y_ / H, f"{counts[t]['test_chezod117']['test_coverage_30_80']}", fontsize=FS_SMALL,
                 ha="right", va="center")  # fmt: skip
    fig.text(1.0 / W, 72.5 / H, "c1", fontsize=6.5, fontweight="bold", va="bottom")

    # ------------------------------------------------------------------ c2 strip
    ax2 = ax_mm(33.0, 6.2, 53.0, 14.6)
    cls_cols = ["#D9D9D9", "#969696", "#252525"]
    for j, s in enumerate(strip):
        left = 0.0
        for c, (v, cc) in enumerate(zip(s["shares"], cls_cols)):
            ax2.barh(j, v, left=left, color=cc, height=0.74, edgecolor="white", lw=0.3)
            if v >= 0.07:
                ax2.text(left + v / 2, j, f"{100 * v:.0f}", ha="center", va="center", fontsize=FS_SMALL,
                         color="black" if c == 0 else "white")  # fmt: skip
            left += v
    ax2.set_yticks(range(len(strip)))
    ax2.set_yticklabels([f"{s['set']} ({s['n']:,})" for s in strip])
    for tl, s in zip(ax2.get_yticklabels(), strip):
        tl.set_color(
            s["color"]
            if s["set"] in ("CheZOD117", "TriZOD test")
            else sp.CAT["neutral"]
        )
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0", "25", "50", "75", "100"])
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(left=False)
    ax2.set_xlabel("Chains (%)", labelpad=1.0)
    h_c = [
        mpatches.Patch(color=c, label=lab)
        for c, lab in zip(cls_cols, ["≤ 10 %", "10–90 %", "≥ 90 %"])
    ]
    fig.legend(handles=h_c, loc="lower left", bbox_to_anchor=(33.0 / W, 21.6 / H), ncol=3, handlelength=1.0,
               handletextpad=0.3, columnspacing=0.9, borderaxespad=0.0, fontsize=FS_SMALL)  # fmt: skip
    fig.text(
        7.0 / W, 22.4 / H, "Residues with G > 0.65:", fontsize=FS_SMALL, va="bottom"
    )
    fig.text(1.0 / W, 22.4 / H, "c2", fontsize=6.5, fontweight="bold", va="bottom")

    sp.fig_label(fig, "c", 0.0, 0.0)
    return fig


def main(argv=None) -> None:
    ap = sp.cli(__doc__)
    ap.add_argument("--work-dir", type=Path, default=sp.TMP / "fig3c_split")
    ap.add_argument("--refresh", action="store_true", help="re-run the mmseqs searches")
    args = sp.parse_args(ap, argv)

    test_ids = {s: list(sp.read_fasta(p)) for s, p in TEST_FASTAS.items()}
    set_of = {i: s for s, ids in test_ids.items() for i in ids}
    reps = {
        t: set(sp.read_fasta(sp.BUNDLE / "train" / t / f"train_{t}_best.fasta"))
        for t in sp.TIERS
    }
    for a, b in zip(sp.TIERS[1:], sp.TIERS[:-1]):
        assert reps[a] <= reps[b], "training representatives are expected to be nested"
    rep_tier = {}
    for t in sp.TIERS:  # strictest tier at which a chain is a representative
        for i in reps[t]:
            rep_tier[i] = t

    hits = run_search(args.work_dir, args.refresh)
    uni, P, counts = analyse(hits, set_of, rep_tier)
    strip = disorder_strip(test_ids, reps)

    fig = draw(P, counts, strip, len(set_of))
    pdf = sp.save(fig, "fig3c")

    plotted = {
        s: {
            "n": int((P.set == s).sum()),
            "no_hit": int(((P.set == s) & ~P.has_hit).sum()),
            "in_region": int(((P.set == s) & P.in_region).sum()),
            "in_region_by_strictest_tier_of_plotted_rep": {
                t: int(((P.set == s) & P.in_region & (P.ttier == t)).sum())
                for t in sp.TIERS
            },
        }
        for s in TEST_FASTAS
    }
    out = {
        "definition": {
            "search": f"mmseqs easy-search {' '.join(MMSEQS_ARGS)} (mmseqs {mmseqs_version()}); one search per tier",
            "query": "575 held-out chains (TriZOD_test_set.fasta + CheZOD117_test_set.fasta)",
            "target": "train/<tier>/train_<tier>_best.fasta (frozen 2026-10 split, BEFORE fix (a))",
            "test_coverage_30_80": "identity >= 0.30 and test-chain coverage (qcov) >= 0.80 (the corrected criterion; shaded)",
            "bidirectional_30_80": "identity >= 0.30 and qcov >= 0.80 and tcov >= 0.80 (the pipeline's current stage-2 definition)",
            "id50_testcov50": "identity >= 0.50 and qcov >= 0.50",
            "id90_testcov50": "identity >= 0.90 and qcov >= 0.50",
            "cumulative": "a chain counts at tier t if an alignment to a representative of tier t satisfies the rule "
            "(that tier's own search)",
            "test_coverage_30_80_from_union": "same, from the union of searches restricted to tier-t representatives",
        },
        "counts_by_tier": counts,
        "plotted_points": plotted,
        "median_identity_of_plotted_alignment": float(P[P.has_hit].fident.median()),
        "verified_containments": {
            q: uni[(uni["query"] == q) & (uni.target == t)][
                ["fident", "qcov", "tcov", "qlen", "tlen"]
            ]
            .round(3)
            .to_dict("records")
            for q, t in [
                ("51709_1_1_1", "53339_1_1_1"),
                ("19221_1_1_1", "19219_1_1_1"),
                ("15813_1_1_1", "30456_1_1_1"),
            ]
        },
        "c2_disorder_class_pct": {
            s["set"]: {
                "n": s["n"],
                **dict(
                    zip(
                        ["<=10", "10-90", ">=90"],
                        [round(100 * v, 1) for v in s["shares"]],
                    )
                ),
                "pct_residues_G_gt_065": s["pct_residues_G_gt_065"],
                "chains_gt_50pct_disordered": s["chains_gt_50pct_disordered"],
            }
            for s in strip
        },
    }
    data_dir = sp.out_dir("fig3_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    P.to_csv(data_dir / "fig3c1_points.csv", index=False)
    pd.DataFrame(
        [
            {
                "set": s["set"],
                "n": s["n"],
                **dict(zip(["le10", "10to90", "ge90"], s["shares"])),
            }
            for s in strip
        ]
    ).to_csv(data_dir / "fig3c2_strip.csv", index=False)
    (data_dir / "fig3c_numbers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"wrote {pdf}", file=sys.stderr)


if __name__ == "__main__":
    main()
