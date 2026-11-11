#!/usr/bin/env python3
"""Fig. 3c and 3d -- the held-out sets against training: similarity (c) and disorder (d).

c. For each held-out chain (TriZOD test + CheZOD117), its closest training
representative: identity against alignment coverage. The search is the build's own
stage-2 leakage search (``trizod.dataset.mmseqs.leakage_search_options``: training
chains as query, the held-out chains as target database, ``--alignment-mode 3
--cov-mode 1 -c 0 -s 7.5 --comp-bias-corr 0 --mask 0 --prefilter-mode 2``, E <= 1e-3),
with one change: ``--min-seq-id`` is lowered from 0.3 to 0.2, so near misses below the
identity threshold are drawn too. Because the held-out chains are the target database,
every E-value is the one the build saw, whatever training set is searched. The query is
the unfiltered tier's representatives (``train/unfiltered/train_unfiltered_best.fasta``);
the representatives of every stricter tier are a subset of them, so the closest
unfiltered representative is the closest representative of any tier.

Coverage is the build's (aligned span / chain length, gaps included). Under the
build's leakage rule (``MANIFEST.json`` ``leakage.rule``; "either" since the 2026-11
build) the x axis is the larger of the two coverages, i.e. the better-covered chain,
and the shaded region is what the rule removes from training: identity >= 30 % over
>= 80 % of the test OR the training chain. Under rule "test" the x axis is the
test-chain coverage. Each held-out chain is drawn at one alignment: an in-region one if
any exists (none should, for a build that passed its gate), otherwise the one with the
largest identity x coverage. Chains without any reported alignment are not drawn; the
key gives drawn and total chains per set. Counts by tier and rule are written to
``fig3c_numbers.json``.

d. Share of chains by disorder class -- the share of a chain's scored residues with
G > 0.65 is <= 10 %, 10-90 % or >= 90 % (the key gives the classes; "mostly disordered"
is avoided because the text uses it for > 50 %) --
for CheZOD117, the TriZOD test set and the training representatives of every tier, each
with its own tier's labels (``scores/<tier>/scores.json``; the test sets use the
unfiltered labels, as released).

Colours: blue = TriZOD test, orange = CheZOD117 (Okabe-Ito roles of ``style_paper``);
the excluded region and the disorder classes are neutral greys.

``MMSEQS_ARGS`` is the search of the earlier version of this panel (held-out chains as
query against each tier's representatives); the panel no longer uses it, but
``scripts/manuscript/make_numbers.py`` imports it for the ``\\nLeak*Common*`` and
near-redundant-representative macros, so it stays unchanged.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig3c_split.py \
        --out-dir <draft>/Figures/paper [--work-dir <scratch>/fig3]
Outputs: <out-dir>/fig3c.pdf and fig3d.pdf (each W_SINGLE wide, set side by side in
Fig. 3); <out-dir>/fig3_data/fig3c_points.csv, fig3d_disorder_classes.csv and
fig3c_numbers.json. The mmseqs hits are cached in ``--work-dir``
(default ``fig3c_split`` in the cache directory), keyed by the inputs and options.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style_paper as sp  # noqa: E402

from trizod.dataset import mmseqs as mm  # noqa: E402

# The earlier panel's search; kept verbatim for scripts/manuscript/make_numbers.py.
MMSEQS_ARGS = [
    "--min-seq-id", "0.2", "-c", "0", "-e", "1e-3",
    "--alignment-mode", "3", "-s", "7.5", "--comp-bias-corr", "0", "--mask", "0",
    "--max-seqs", "5000",
]  # fmt: skip
PANEL_MIN_SEQ_ID = 0.2  # the panel's only change to the stage-2 options
ID_MIN, COV_MIN = mm.LEAKAGE_MIN_SEQ_ID, mm.LEAKAGE_MIN_COV
TEST_FASTAS = {
    "test_trizod": sp.BUNDLE / "test" / "TriZOD_test_set.fasta",
    "test_chezod117": sp.BUNDLE / "test" / "CheZOD117_test_set.fasta",
}
SET_STYLE = {  # label, colour, marker, marker area (pt^2)
    "test_trizod": ("TriZOD test", sp.CAT["test"], "o", 9.0),
    "test_chezod117": ("CheZOD117", sp.CAT["chezod"], "^", 11.0),
}
REGION_FILL = "#EBEBEB"
REGION_LINE = "#7F7F7F"
CLASS_COLORS = ["#DADADA", "#969696", "#3B3B3B"]  # ordered -> disordered
# share of a chain's scored residues with G > G_DIS
CLASS_LABELS = ["≤10%", "10–90%", "≥90%"]
RULES = {  # extra counts for the numbers JSON; x = the coverage the rule tests
    "either_30_80": lambda h: (h.ident >= ID_MIN) & (np.maximum(h.test_cov, h.train_cov) >= COV_MIN),
    "test_30_80": lambda h: (h.ident >= ID_MIN) & (h.test_cov >= COV_MIN),
    "both_30_80": lambda h: (h.ident >= ID_MIN) & (np.minimum(h.test_cov, h.train_cov) >= COV_MIN),
    "test_50_50": lambda h: (h.ident >= 0.50) & (h.test_cov >= 0.50),
    "test_90_50": lambda h: (h.ident >= 0.90) & (h.test_cov >= 0.50),
}  # fmt: skip


def mmseqs_version() -> str:
    return mm.mmseqs_version() or "missing"


def leakage_rule() -> str:
    """The rule the build applied (release MANIFEST), else the pipeline default."""
    try:
        man = json.loads((sp.BUNDLE / "MANIFEST.json").read_text())
        return mm.check_rule(man["leakage"]["rule"])
    except (OSError, ValueError, KeyError, TypeError):
        return mm.DEFAULT_LEAKAGE_RULE


def panel_search_options() -> list[str]:
    """The build's stage-2 options with ``--min-seq-id`` lowered to show near misses."""
    opts = mm.leakage_search_options()
    opts[opts.index("--min-seq-id") + 1] = str(PANEL_MIN_SEQ_ID)
    return opts


def _digest(paths: list[Path], opts: list[str]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    h.update(" ".join(opts).encode())
    return h.hexdigest()[:16]


def run_search(work: Path, refresh: bool) -> pd.DataFrame:
    """Unfiltered representatives (query) vs all held-out chains (target), cached."""
    work.mkdir(parents=True, exist_ok=True)
    target = work / "test_all.fasta"
    with open(target, "w") as fh:
        for path in TEST_FASTAS.values():
            for k, s in sp.read_fasta(path).items():
                fh.write(f">{k}\n{s}\n")
    query = sp.BUNDLE / "train" / "unfiltered" / "train_unfiltered_best.fasta"
    opts = panel_search_options()
    out = work / f"panel_hits_{_digest([query, target], opts)}.tsv"
    if refresh or not out.exists():
        if shutil.which("mmseqs") is None:
            raise SystemExit("mmseqs not on PATH")
        tmp = work / "tmp_panel"
        cmd = ["mmseqs", "easy-search", str(query), str(target), str(out), str(tmp),
               *opts, "--threads", "8", "-v", "1"]  # fmt: skip
        print(" ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, check=True)
        shutil.rmtree(tmp, ignore_errors=True)
    H = pd.DataFrame([dataclasses.asdict(h) for h in mm.parse_leakage_hits(out)])
    H["ident"] = H["pident"] / 100.0
    return H


def analyse(hits: pd.DataFrame, set_of: dict, reps: dict, rep_tier: dict, rule: str):
    h = hits.copy()
    h["set"] = h["test_id"].map(set_of)
    h["ttier"] = h["train_id"].map(rep_tier)
    h["trank"] = h["ttier"].map(sp.RANK)
    h["xcov"] = np.maximum(h.test_cov, h.train_cov) if rule == "either" else h.test_cov
    h["in_region"] = (h.ident >= ID_MIN) & (h.xcov >= COV_MIN)
    h["score"] = h.ident * h.xcov

    rows = []
    for q, s in set_of.items():
        g = h[h["test_id"] == q]
        if g.empty:
            rows.append({"id": q, "set": s, "has_hit": False, "in_region": False})
            continue
        inr = g[g.in_region]
        if len(inr):
            best = inr.sort_values(["trank", "ident", "xcov"], ascending=False).iloc[0]
        else:
            best = g.sort_values(["score", "ident"], ascending=False).iloc[0]
        rows.append({
            "id": q, "set": s, "has_hit": True, "target": best.train_id, "ident": best.ident,
            "xcov": best.xcov, "test_cov": best.test_cov, "train_cov": best.train_cov,
            "test_len": best.test_len, "train_len": best.train_len, "evalue": best.evalue,
            "ttier": best.ttier, "in_region": bool(best.in_region),
        })  # fmt: skip
    P = pd.DataFrame(rows)

    counts = {}
    for t in sp.TIERS:  # cumulative: alignments to that tier's representatives
        own = h[h.train_id.isin(reps[t])]
        c = {}
        for s in TEST_FASTAS:
            ss = own[own.set == s]
            c[s] = {
                name: int(ss[rule_(ss)]["test_id"].nunique())
                for name, rule_ in RULES.items()
            }
            c[s]["any_hit"] = int(ss["test_id"].nunique())
        c["reps_in_region_any_test_chain"] = int(
            own[own.in_region]["train_id"].nunique()
        )
        counts[t] = c
    return h, P, counts


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


# c and d are half-width cells (W_SINGLE) set side by side in Fig. 3 at 1:1, below a and b
H_CD = 50.0  # mm, height of both
AX_Y0_CD, AX_H_CD = 9.5, 34.0  # bottom and height of both axes, mm
Y_LIM_C = (17.5, 102.0)  # identity (%): no marker on an axis
X_LIM_C = (-2.0, 102.0)  # coverage (%)


def draw_c(P: pd.DataFrame, rule: str):
    """Fig. 3c: each held-out chain against its closest training representative."""
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    W, H = sp.W_SINGLE, H_CD
    fig = sp.figure(W, H)
    ax = fig.add_axes([13.0 / W, AX_Y0_CD / H, 62.0 / W, AX_H_CD / H])
    hi = Y_LIM_C[1]
    x0, y0 = 100 * COV_MIN, 100 * ID_MIN
    ax.add_patch(mpatches.Rectangle((x0, y0), X_LIM_C[1] - x0, hi - y0,
                                    facecolor=REGION_FILL, lw=0, zorder=0))  # fmt: skip
    ax.plot(
        [x0, x0, X_LIM_C[1]], [hi, y0, y0], color=REGION_LINE, lw=0.7, ls="--", zorder=1
    )
    ax.text((x0 + X_LIM_C[1]) / 2, (y0 + hi) / 2 + 4, "Excluded", ha="center", va="center",
            color=sp.CAT["neutral"], fontsize=sp.FS_TICK)  # fmt: skip
    pts = P[P.has_hit]
    handles = []
    for s in ("test_trizod", "test_chezod117"):
        lab, col, marker, size = SET_STYLE[s]
        sub = pts[pts.set == s]
        ax.scatter(100 * sub.xcov, 100 * sub.ident, s=size, marker=marker, facecolor=col,
                   edgecolor="white", linewidths=0.25, alpha=0.9, zorder=3)  # fmt: skip
        n_all = int((P.set == s).sum())
        handles.append(
            Line2D([], [], ls="", marker=marker, ms=3.6, mfc=col, mec="white", mew=0.25,
                   label=f"{lab} ({len(sub):,} of {n_all:,})")
        )  # fmt: skip
    ax.set_xlim(*X_LIM_C)
    ax.set_ylim(*Y_LIM_C)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xlabel("Coverage of the better-covered chain (%)" if rule == "either"
                  else "Coverage of the held-out chain (%)")  # fmt: skip
    ax.set_ylabel("Identity to closest\ntraining chain (%)")
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(13.0 / W, (AX_Y0_CD + AX_H_CD + 1.0) / H),
               ncol=2, handletextpad=0.1, columnspacing=1.0, borderaxespad=0.0)  # fmt: skip
    sp.fig_label(fig, "c", 0.0, 0.0)
    return fig


def draw_d(strip: list):
    """Fig. 3d: chains of each set by their share of disordered residues."""
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    W, H = sp.W_SINGLE, H_CD
    fig = sp.figure(W, H)
    ys = [0.0, 1.0, 2.4, 3.4, 4.4, 5.4]
    ax = fig.add_axes([30.0 / W, AX_Y0_CD / H, 45.0 / W, AX_H_CD / H])
    for y, s in zip(ys, strip):
        left = 0.0
        for c, (v, cc) in enumerate(zip(s["shares"], CLASS_COLORS)):
            ax.barh(y, v, left=left, color=cc, height=0.66, edgecolor="white", lw=0.5)
            if v >= 0.09:
                ax.text(left + v / 2, y, f"{100 * v:.0f}", ha="center", va="center",
                        fontsize=sp.FS_TICK, color="black" if c == 0 else "white")  # fmt: skip
            left += v
    labels = [f"{s['set'].replace('Training, ', '')} ({s['n']:,})" for s in strip]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=sp.FS_BODY)
    ax.set_ylim(ys[-1] + 0.55, -0.55)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50", "100"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Chains (%)")
    # group label for the four training rows, left of their tick labels
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    x_left = min(t.get_window_extent(renderer).x0 for t in ax.get_yticklabels()[2:])
    x_fig = inv.transform((x_left, 0))[0] - 1.2 / W
    f0, f1 = (inv.transform(ax.transData.transform((0, y)))[1]
              for y in (ys[2] - 0.33, ys[-1] + 0.33))  # fmt: skip
    fig.add_artist(Line2D([x_fig, x_fig], [f0, f1], color=sp.CAT["neutral"], lw=0.6))
    fig.text(x_fig - 0.8 / W, (f0 + f1) / 2, "Training", rotation=90, ha="right",
             va="center", fontsize=sp.FS_BODY)  # fmt: skip
    # class key above the axes, right-aligned with it, preceded by its title
    h_c = [mpatches.Patch(facecolor=c, edgecolor="none", label=lab)
           for c, lab in zip(CLASS_COLORS, CLASS_LABELS)]  # fmt: skip
    leg = fig.legend(handles=h_c, loc="lower right", bbox_to_anchor=(75.0 / W, (AX_Y0_CD + AX_H_CD + 1.0) / H),
                     ncol=3, handlelength=1.0, handletextpad=0.35, columnspacing=0.9,
                     borderaxespad=0.0)  # fmt: skip
    fig.canvas.draw()
    x0 = inv.transform(leg.get_window_extent(renderer).p0)[0]
    y_mid = inv.transform(leg.get_texts()[0].get_window_extent(renderer).p0)[1]
    fig.text(x0 - 0.8 / W, y_mid, f"Residues with G > {sp.G_DIS}:", ha="right", va="bottom",
             fontsize=sp.FS_TICK)  # fmt: skip
    sp.fig_label(fig, "d", 0.0, 0.0)
    return fig


def main(argv=None) -> None:
    ap = sp.cli(__doc__)
    ap.add_argument("--work-dir", type=Path, default=sp.TMP / "fig3c_split")
    ap.add_argument("--refresh", action="store_true", help="re-run the mmseqs search")
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
    rule = leakage_rule()

    hits = run_search(args.work_dir, args.refresh)
    H, P, counts = analyse(hits, set_of, reps, rep_tier, rule)
    n_in = int(P.in_region.sum())
    if n_in:
        print(
            f"WARNING: {n_in} held-out chains lie in the excluded region",
            file=sys.stderr,
        )
    strip = disorder_strip(test_ids, reps)

    pdf = sp.save(draw_c(P, rule), "fig3c")
    pdf_d = sp.save(draw_d(strip), "fig3d")

    plotted = {
        s: {
            "n": int((P.set == s).sum()),
            "drawn": int(((P.set == s) & P.has_hit).sum()),
            "no_hit": int(((P.set == s) & ~P.has_hit).sum()),
            "in_region": int(((P.set == s) & P.in_region).sum()),
            "max_identity_drawn": round(
                float(P[(P.set == s) & P.has_hit].ident.max()), 3
            ),
        }
        for s in TEST_FASTAS
    }
    out = {
        "definition": {
            "release": sp.RELEASE,
            "pipeline": sp.PIPELINE,
            "search": f"mmseqs easy-search train_unfiltered_best.fasta (query) vs TriZOD_test_set.fasta + "
            f"CheZOD117_test_set.fasta (target) {' '.join(panel_search_options())} (mmseqs {mmseqs_version()}); "
            "the build's stage-2 options with --min-seq-id lowered from 0.3",
            "leakage_rule": rule,
            "x": "max(test-chain, training-chain coverage)"
            if rule == "either"
            else "test-chain coverage",
            "region": f"identity >= {ID_MIN} and x >= {COV_MIN} (removed from training by the build)",
            "plotted_alignment": "per held-out chain: an in-region alignment if any, else the largest identity x x",
            "cumulative": "a chain counts at tier t if an alignment to a representative of tier t satisfies the rule; "
            "one search, E-values against the held-out target database as in the build",
            "rules": {
                "either_30_80": ">= 30 % identity over >= 80 % of the test or the training chain",
                "test_30_80": ">= 30 % identity over >= 80 % of the test chain",
                "both_30_80": ">= 30 % identity over >= 80 % of both chains",
                "test_50_50": ">= 50 % identity over >= 50 % of the test chain",
                "test_90_50": ">= 90 % identity over >= 50 % of the test chain",
            },
        },
        "counts_by_tier": counts,
        "plotted_points": plotted,
        "median_identity_of_plotted_alignment": float(P[P.has_hit].ident.median()),
        "d_disorder_class_pct": {
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
    P.to_csv(data_dir / "fig3c_points.csv", index=False)
    pd.DataFrame(
        [
            {
                "set": s["set"],
                "n": s["n"],
                **dict(zip(["le10", "10to90", "ge90"], s["shares"])),
            }
            for s in strip
        ]
    ).to_csv(data_dir / "fig3d_disorder_classes.csv", index=False)
    (data_dir / "fig3c_numbers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"wrote {pdf} and {pdf_d}", file=sys.stderr)


if __name__ == "__main__":
    main()
