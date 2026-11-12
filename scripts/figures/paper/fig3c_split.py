#!/usr/bin/env python3
"""Fig. 5a and 5b -- the held-out sets against training: similarity (a) and disorder (b).

(The file keeps its old name until the scripts are renumbered at layout freeze; it made
Fig. 3c and 3d of the 11 Nov draft.)

a. For each held-out chain (TriZOD test + CheZOD117), its closest training
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
key ("Shown:") gives drawn and total chains per set. Counts by tier and rule are written
to ``fig5ab_numbers.json``.

b. Share of chains by disorder class -- the share of a chain's scored residues with
G > 0.65 is <= 10 %, 10-90 % or >= 90 % (the key gives the classes; "mostly disordered"
is avoided because the text uses it for > 50 %) --
for the TriZOD test set, CheZOD117 and the training representatives of every tier, each
with its own tier's labels (``scores/<tier>/scores.json``). The test sets use their
released Parquet labels: the unfiltered labels, except for the CheZOD117 chains whose
release label is the tolerant one (\nTolerantLabelDiffTestCheZOD); these are also the
labels Fig. 5c evaluates.

Encodings (``style_paper`` entities): TriZOD test = blue circles, CheZOD117 = vermillion
triangles (row markers in b too); training tiers = the tier ramp and tier markers (b and
c); the excluded region = removed (light grey hatching, dashed bounds); the disorder
classes are an ordinal grey ramp.

``MMSEQS_ARGS`` is the search of the earlier version of this panel (held-out chains as
query against each tier's representatives); the panel no longer uses it, but
``scripts/manuscript/make_numbers.py`` imports it for the ``\\nLeak*Common*`` and
near-redundant-representative macros, so it stays unchanged.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig3c_split.py \
        --out-dir <draft>/Figures/paper [--work-dir <scratch>/fig3]
Outputs: <out-dir>/fig5a_split.pdf and fig5b_disorder.pdf (each W_SINGLE x 58 mm, set
side by side at 1:1 in Fig. 5, sections/fig5.tex); <out-dir>/fig5_data/fig5a_points.csv,
fig5b_disorder_classes.csv and fig5ab_numbers.json. The mmseqs hits are cached in ``--work-dir``
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
from matplotlib.lines import Line2D

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
# marker areas 30 % smaller than on 11 Nov (figure review item 11: about 250 points
# overplot at 30-45 % identity and 70-80 % coverage); drawn with alpha 0.8
SET_STYLE = {  # label, colour, marker, marker area (pt^2)
    "test_trizod": ("TriZOD test", sp.CAT["test"], "o", 6.3),
    "test_chezod117": ("CheZOD117", sp.CAT["chezod117"], "^", 7.7),
}
SCATTER_ALPHA = 0.8
# the excluded region is a removed item: light grey hatching (style_paper.ENTITY)
REGION_FILL = sp.patch_kw("removed")
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

    sets = [  # the order of the rows in Fig. 5b
        ("TriZOD test", [g_unf[i] for i in test_ids["test_trizod"]], sp.CAT["test"]),
        (
            "CheZOD117",
            [g_unf[i] for i in test_ids["test_chezod117"]],
            sp.CAT["chezod117"],
        ),
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


# Fig. 5a and 5b are half-width cells (W_SINGLE) set side by side at 1:1, above Fig. 5c
# (supp_s10_baselines.draw_main). Both share the axes' bottom and top, so they align.
H_AB = 58.0  # mm, height of both
AX_Y0_AB, AX_H_AB = 9.5, 40.5  # bottom and height of both axes, mm
Y_LIM_A = (17.5, 102.0)  # identity (%): no marker on an axis
X_LIM_A = (-2.0, 102.0)  # coverage (%)
EDGE = "white"  # thin white edge separates overlapping markers
MARK_X, LABEL_X = 1.8, 3.6  # Fig. 5b row labels: marker centre and text start, mm


def draw_a(P: pd.DataFrame, rule: str):
    """Fig. 5a: each held-out chain against its closest training representative."""
    import matplotlib.patches as mpatches

    W, H = sp.W_SINGLE, H_AB
    fig = sp.figure(W, H)
    ax = fig.add_axes([13.0 / W, AX_Y0_AB / H, 63.0 / W, AX_H_AB / H])
    hi = Y_LIM_A[1]
    x0, y0 = 100 * COV_MIN, 100 * ID_MIN
    ax.add_patch(mpatches.Rectangle((x0, y0), X_LIM_A[1] - x0, hi - y0,
                                    **REGION_FILL, zorder=0))  # fmt: skip
    ax.plot(
        [x0, x0, X_LIM_A[1]], [hi, y0, y0], **sp.line_kw("removed", lw=0.7), zorder=1
    )
    ax.text((x0 + X_LIM_A[1]) / 2, (y0 + hi) / 2, "Removed\nfrom\ntraining", ha="center",
            va="center", color=sp.INK_SOFT, fontsize=sp.FS_BODY, linespacing=1.1,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})  # fmt: skip
    pts = P[P.has_hit]
    for s in ("test_trizod", "test_chezod117"):
        lab, col, marker, size = SET_STYLE[s]
        sub = pts[pts.set == s]
        ax.scatter(100 * sub.xcov, 100 * sub.ident, s=size, marker=marker, facecolor=col,
                   edgecolor=EDGE, linewidths=0.3, alpha=SCATTER_ALPHA, zorder=3)  # fmt: skip
    ax.set_xlim(*X_LIM_A)
    ax.set_ylim(*Y_LIM_A)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xlabel("Coverage of the better-covered chain (%)" if rule == "either"
                  else "Coverage of the held-out chain (%)")  # fmt: skip
    ax.set_ylabel("Identity to the closest\ntraining representative (%)")
    # key above the axes: marker, set, chains drawn of all chains
    y_key = (AX_Y0_AB + AX_H_AB + 2.6) / H
    t = fig.text(5.0 / W, y_key, "Shown:", ha="left", va="center", fontsize=sp.FS_BODY)
    fig.canvas.draw()
    xk = t.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi / sp.MM + 1.6
    for s in ("test_trizod", "test_chezod117"):
        lab, col, marker, size = SET_STYLE[s]
        n_all = int((P.set == s).sum())
        n_drawn = int(((P.set == s) & P.has_hit).sum())
        fig.add_artist(Line2D([(xk + 1.0) / W], [y_key], transform=fig.transFigure,
                              **sp.marker_kw("test" if s == "test_trizod" else "chezod117",
                                             ms=3.6)))  # fmt: skip
        t = fig.text((xk + 2.4) / W, y_key, f"{lab} ({n_drawn} of {n_all})", ha="left",
                     va="center", fontsize=sp.FS_BODY)  # fmt: skip
        fig.canvas.draw()
        bb = t.get_window_extent(fig.canvas.get_renderer())
        xk = bb.x1 / fig.dpi / sp.MM + 3.5
    sp.fig_label(fig, "a", 0.0, 0.0)
    return fig


def draw_b(strip: list):
    """Fig. 5b: chains of each set by their share of disordered residues."""
    import matplotlib.patches as mpatches

    W, H = sp.W_SINGLE, H_AB
    fig = sp.figure(W, H)
    ys = [0.0, 1.0, 2.5, 3.5, 4.5, 5.5]
    ax_x0, ax_w = 29.0, 46.0
    ax = fig.add_axes([ax_x0 / W, AX_Y0_AB / H, ax_w / W, AX_H_AB / H])
    for y, s in zip(ys, strip):
        left = 0.0
        for c, (v, cc) in enumerate(zip(s["shares"], CLASS_COLORS)):
            ax.barh(y, v, left=left, color=cc, height=0.66, edgecolor="white", lw=0.5)
            if v >= 0.09:
                ax.text(left + v / 2, y, f"{100 * v:.0f}", ha="center", va="center",
                        fontsize=sp.FS_BODY, color="black" if c == 0 else "white")  # fmt: skip
            left += v
    ax.set_yticks(ys)
    ax.set_yticklabels([""] * len(ys))
    ax.set_ylim(ys[-1] + 0.55, -0.55)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Chains (%)")
    # row labels: set marker (test sets) or tier marker (training), then the name and n
    span = ys[-1] + 1.1
    to_fig_y = lambda y: (AX_Y0_AB + AX_H_AB * (ys[-1] + 0.55 - y) / span) / H  # noqa: E731
    keys = ["test", "chezod117"]
    for k, (y, s) in enumerate(zip(ys, strip)):
        fy = to_fig_y(y)
        name = s["set"].replace("Training, ", "")
        fig.text(LABEL_X / W, fy, f"{name} ({s['n']:,})", ha="left", va="center",
                 fontsize=sp.FS_BODY)  # fmt: skip
        if k < 2:
            mk = sp.marker_kw(keys[k], ms=3.6)
        else:
            # tier rows: a square swatch in the tier colour, not the tier marker, so that in
            # Fig. 5 circles and triangles mean only the two test sets (figure review item 5)
            mk = {"ls": "", "marker": "s", "color": sp.TIER_COLORS[name],
                  "mfc": sp.TIER_COLORS[name], "mec": sp.TIER_COLORS[name], "ms": 3.4}  # fmt: skip
        fig.add_artist(Line2D([MARK_X / W], [fy], transform=fig.transFigure, **mk))
    # header above the four training rows
    fig.text(0.6 / W, to_fig_y((ys[1] + ys[2]) / 2 + 0.08), "Training tier", ha="left",
             va="center", fontsize=sp.FS_BODY)  # fmt: skip
    # class key above the axes, one row, preceded by its title
    y_key = (AX_Y0_AB + AX_H_AB + 2.6) / H
    t = fig.text(5.0 / W, y_key, f"Residues with G > {sp.G_DIS}:", ha="left", va="center",
                 fontsize=sp.FS_BODY)  # fmt: skip
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    xk = t.get_window_extent(rnd).x1 / fig.dpi / sp.MM + 2.0
    for c, lab in zip(CLASS_COLORS, CLASS_LABELS):
        fig.add_artist(mpatches.Rectangle((xk / W, y_key - 1.0 / H), 2.6 / W, 2.0 / H,
                                          transform=fig.transFigure, facecolor=c,
                                          edgecolor="none"))  # fmt: skip
        t = fig.text(
            (xk + 3.3) / W, y_key, lab, ha="left", va="center", fontsize=sp.FS_BODY
        )
        fig.canvas.draw()
        xk = t.get_window_extent(rnd).x1 / fig.dpi / sp.MM + 2.4
    sp.fig_label(fig, "b", 0.0, 0.0)
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

    pdf = sp.save(draw_a(P, rule), "fig5a_split")
    pdf_d = sp.save(draw_b(strip), "fig5b_disorder")

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
        "b_disorder_class_pct": {
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
    data_dir = sp.out_dir("fig5_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    P.to_csv(data_dir / "fig5a_points.csv", index=False)
    pd.DataFrame(
        [
            {
                "set": s["set"],
                "n": s["n"],
                **dict(zip(["le10", "10to90", "ge90"], s["shares"])),
                "pct_residues_G_gt_065": s["pct_residues_G_gt_065"],
            }
            for s in strip
        ]
    ).to_csv(data_dir / "fig5b_disorder_classes.csv", index=False)
    (data_dir / "fig5ab_numbers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"wrote {pdf} and {pdf_d}", file=sys.stderr)


if __name__ == "__main__":
    main()
