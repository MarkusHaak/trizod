#!/usr/bin/env python3
"""Supplementary Fig. S3 -- per-criterion filter losses per tier (the build log).

Source: the four ``print_filter_losses()`` reports in the build log
(``style_paper.REBUILD_LOG``: ``$TRIZOD_REBUILD_LOG``, default
``data/interim/build/rebuild_2026-10.log``; one report per tier, in tier order). Every count is
a number of candidate records (entry x shift table x assembly x entity; 17,843 in the
snapshot). "filtered" counts the records a criterion removes; "unique" counts those that
no other criterion of the same stage would have removed.

Panels
  a  what happens to the 17,843 candidate records at each tier: removed before scoring,
     removed after scoring, scored (= the tier's chains);
  b  pre-scoring criteria (sample conditions, keywords, perturbing cosolvents, data
     completeness), one column per tier, shared linear x-axis;
  c  post-scoring losses: shift types rejected for a referencing offset beyond the tier's
     max-offset (tolerant drops the shift type only; moderate and strict drop the record)
     and the post-scoring completeness checks.

Run:  uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s3_filter_losses.py \
          --out-dir <draft>/Figures/paper
Outputs: <out-dir>/S3.pdf and <out-dir>/suppfigs_a_data/S3_filter_losses.csv.
"""

from __future__ import annotations

import re
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import style_paper as sp  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

LOG = sp.REBUILD_LOG
TIERS = sp.TIERS

# ----------------------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------------------
_ROW = re.compile(
    r"^(?P<label>\S.*?)\s*(?P<thr>\[[^\]]*\])?\s*:\s+(?P<a>\d+)\s+(?P<b>\d+)\s*$"
)
_TOT = r"\s*:\s*(\d+) of\s+(\d+) \(\s*([\d.]+) %\)"


def parse_blocks(text: str) -> list[dict]:
    """The four print_filter_losses() reports, each as {pre, rej, post, totals}.

    ``pre``/``post`` map a criterion label to (filtered, unique, threshold-or-None,
    section); ``rej`` maps a backbone atom to (rejected, unique).
    """
    starts = [
        m.start() for m in re.finditer(r"Pre-computation filtering results", text)
    ]
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[s:e]
        end = chunk.find("final dataset entries")
        chunk = chunk[: chunk.find("\n", end) + 1]
        pre, rest = chunk.split("Rejected offsets stats", 1)
        rej, post = rest.split("Post-computation filtering results", 1)
        b = {"pre": OrderedDict(), "rej": OrderedDict(), "post": OrderedDict()}
        for name, part in (("pre", pre), ("rej", rej), ("post", post)):
            section = "criterium"
            for line in part.splitlines():
                st = line.strip()
                if st.endswith("filtered   unique") or st.endswith("rejected   unique"):
                    section = st.split(":")[0].strip()
                    continue
                if st.startswith("total filtered") or st.startswith("remaining"):
                    continue
                m = _ROW.match(st)
                if not m:
                    continue
                label = m.group("label").strip()
                if label.startswith(".*") and label.endswith(".*"):
                    label = label[2:-2]
                thr = m.group("thr")
                if name == "rej":
                    b[name][label] = (int(m.group("a")), int(m.group("b")))
                else:
                    if label == "paramagnetic":
                        section = "paramagnetic"
                    b[name][label] = (
                        int(m.group("a")),
                        int(m.group("b")),
                        thr,
                        section,
                    )
        for key, part, pat in (
            ("pre_total", pre, "total filtered"),
            ("post_total", post, "total filtered"),
            ("remaining", chunk, "remaining for scores computation"),
            ("final", chunk, "final dataset entries"),
        ):
            m = re.search(pat + _TOT, part)
            b[key] = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
        blocks.append(b)
    if len(blocks) != 4:
        raise SystemExit(
            f"expected four print_filter_losses() reports, found {len(blocks)}"
        )
    return blocks


# ----------------------------------------------------------------------------------
# rows of panel b / c: (log label, section key, display label)
# ----------------------------------------------------------------------------------
PRE_ROWS = [
    (
        "Sample and method",
        [
            ("method (sub-)type", "Method subtype (solid-state veto)"),
            ("temperature", "Temperature"),
            ("ionic strength", "Ionic strength"),
            ("pH", "pH"),
            ("physical state", "Physical state (deny list)"),
            ("paramagnetic", "Paramagnetic component"),
        ],
    ),
    (
        "Keyword in the sample description",
        [
            ("denatur", "denatur"),
            ("unfold", "unfold"),
            ("misfold", "misfold"),
            ("bound", "bound"),
            ("in complex with", "in complex with"),
            ("complexed with", "complexed with"),
        ],
    ),
    (
        "Perturbing cosolvent",
        [
            ("guanidin", "guanidin"),
            ("GdmCl", "GdmCl"),
            ("Gdn-Hcl", "Gdn-Hcl"),
            ("urea", "urea"),
            ("TFE", "TFE"),
            ("trifluoroethanol", "trifluoroethanol"),
            ("trifluoro ethanol", "trifluoro ethanol"),
            ("hexafluoroisopropanol", "hexafluoroisopropanol"),
            ("hexafluoro-2-propanol", "hexafluoro-2-propanol"),
            ("HFIP", "HFIP"),
            ("DMSO", "DMSO"),
            ("dimethyl sulfoxide", "dimethyl sulfoxide"),
            ("dimethylsulfoxide", "dimethylsulfoxide"),
        ],
    ),
    (
        "Data completeness",
        [
            ("peptide length", "Chain length"),
            ("bb shift types", "Backbone shift types"),
            ("bb shift positions", "Backbone shift positions"),
            ("bb shift fraction", "Backbone shift fraction"),
            ("non-canonical frac", "Non-canonical residue fraction"),
            ("X fraction", "Unknown residue (X) fraction"),
        ],
    ),
]
REJ_ATOMS = ["C", "CA", "CB", "N", "H", "HA", "HB"]
POST_ROWS = [
    ("bb shift types", "Backbone shift types"),
    ("bb shift positions", "Backbone shift positions"),
    ("bb shift fraction", "Backbone shift fraction"),
    ("error in computation", "Error in computation"),
    ("rejected due to any offset", "Record dropped for an offset"),
]


FILT_GREY = "#D2D2D2"


def lighten(c, f=0.6):
    r, g, b, _ = to_rgba(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f, 1.0)


def fmt(n):
    return f"{n:,}"


def draw_grid(
    fig, rows, blocks_vals, x0_mm, y_top_mm, row_h, col_w, label_w, xmax, header=True
):
    """Small multiples: one axes per tier, rows = criteria. ``blocks_vals[t][label]``
    is (filtered, unique) or None when the criterion is not applied at that tier."""
    W, H = fig.get_size_inches() / sp.MM
    n = len(rows)
    h = n * row_h
    axes = []
    for j, t in enumerate(TIERS):
        left = x0_mm + label_w + j * col_w
        ax = fig.add_axes([left / W, (H - y_top_mm - h) / H, (col_w - 4) / W, h / H])
        axes.append(ax)
        ys = np.arange(n)[::-1]
        for y, row in zip(ys, rows):
            kind, key = row[0], row[1]
            if kind == "section":
                continue
            v = blocks_vals[t].get(key)
            if v is None:
                ax.text(
                    0,
                    y,
                    "not applied",
                    va="center",
                    ha="left",
                    fontsize=5,
                    color="#9A9A9A",
                    style="italic",
                )
                continue
            filt, uniq = v
            ax.barh(y, filt, height=0.72, color=FILT_GREY, lw=0)
            ax.barh(y, uniq, height=0.72, color=sp.TIER_COLORS[t], lw=0)
            txt = f"{fmt(filt)} ({fmt(uniq)})" if filt else "0"
            if filt > 0.6 * xmax:  # long bar: label inside, right-aligned
                dark = uniq > 0.7 * filt and t != "unfiltered"
                ax.text(
                    filt - xmax * 0.015,
                    y,
                    txt,
                    va="center",
                    ha="right",
                    fontsize=5,
                    color="white" if dark else "#222222",
                )
            else:
                ax.text(
                    filt + xmax * 0.015,
                    y,
                    txt,
                    va="center",
                    ha="left",
                    fontsize=5,
                    color="#222222",
                )
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(axis="x", labelsize=5)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        # section separators
        for y, row in zip(ys, rows):
            if row[0] == "section":
                ax.axhline(y, color="#D0D0D0", lw=0.4, zorder=0)
        if header:
            ax.set_title(
                t, fontsize=6, color=sp.TIER_COLORS[t], fontweight="bold", pad=2
            )
    # row labels on the left of the first column
    ax0 = axes[0]
    ys = np.arange(n)[::-1]
    for y, row in zip(ys, rows):
        if row[0] == "section":
            ax0.text(
                -label_w / (col_w - 4),
                y,
                row[2],
                transform=ax0.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=5.5,
                fontweight="bold",
            )
        else:
            ax0.text(
                -0.02,
                y,
                row[2],
                transform=ax0.get_yaxis_transform(),
                va="center",
                ha="right",
                fontsize=5.5,
                style="italic" if row[3] else "normal",
            )
    return axes


def main():
    sp.parse_args(sp.cli(__doc__))
    text = LOG.read_text(errors="ignore")
    B = dict(zip(TIERS, parse_blocks(text)))
    n_cand = B["unfiltered"]["pre_total"][1]
    for t in TIERS:
        assert B[t]["pre_total"][1] == n_cand and B[t]["final"][1] == n_cand

    # ---- report (checked against numbers.tex / writing plan Appendix A3) --------------
    print("candidate records:", n_cand)
    for t in TIERS:
        b = B[t]
        print(
            f"{t:10s} pre {b['pre_total'][0]:6d} ({b['pre_total'][2]} %)  remaining {b['remaining'][0]}"
            f"  post {b['post_total'][0]}  final {b['final'][0]}"
        )

    # ---- figure ---------------------------------------------------------------------
    # Rows follow the log: a criterion no tier reports is dropped (e.g. 'trifluoro ethanol',
    # merged into a misspelling-tolerant pattern after 2026-10), and a keyword or cosolvent
    # criterion that PRE_ROWS does not list (e.g. the alcohols, acetonitrile, chloroform,
    # propanols, alkanes and dioxane added after 2026-10) is appended to its section in log order.
    in_log = {k for t in TIERS for k in B[t]["pre"]}
    listed = {k for _sec, items in PRE_ROWS for k, _lab in items}
    log_sec = {"keyword": "Keyword in the sample description",
               "perturbing cosolvent": "Perturbing cosolvent"}  # fmt: skip
    extra = OrderedDict((s, []) for s in log_sec.values())
    for t in TIERS:
        for key, (_f, _u, _thr, sec) in B[t]["pre"].items():
            s = log_sec.get(sec)
            if s and key not in listed and key not in extra[s]:
                extra[s].append(key)
    for s, keys in extra.items():
        if keys:
            print(f"  {s}: rows added from the log: {', '.join(keys)}")
    dropped = sorted(listed - in_log)
    if dropped:
        print(f"  rows dropped (no tier reports them): {', '.join(dropped)}")
    pre_rows, post_rows = [], []
    for sec, items in PRE_ROWS:
        pre_rows.append(("section", None, sec, False))
        items = [(k, lab) for k, lab in items if k in in_log]
        items += [(k, k) for k in extra.get(sec, [])]
        for key, lab in items:
            pre_rows.append(
                (
                    "row",
                    key,
                    lab,
                    sec != "Sample and method" and sec != "Data completeness",
                )
            )
    post_rows.append(
        ("section", None, "Shift type rejected for an offset beyond max-offset", False)
    )
    for a in REJ_ATOMS:
        post_rows.append(("row", "rej:" + a, a, False))
    post_rows.append(("section", None, "After scoring", False))
    for key, lab in POST_ROWS:
        post_rows.append(("row", "post:" + key, lab, False))

    vals = {}
    for t in TIERS:
        v = {}
        for key, (f, u, _thr, _sec) in B[t]["pre"].items():
            v[key] = (f, u)
        for a, (f, u) in B[t]["rej"].items():
            v["rej:" + a] = (f, u)
        for key, (f, u, _thr, _sec) in B[t]["post"].items():
            v["post:" + key] = (f, u)
        vals[t] = v
    # the unfiltered report omits criteria it never applies; keep them "not applied"
    for k in [r[1] for r in pre_rows if r[0] == "row"]:
        missing = [t for t in TIERS if k not in vals[t]]
        if missing:
            print(f"  criterion '{k}' not applied at: {', '.join(missing)}")

    row_h = 2.35
    label_w = 40.0
    col_w = (sp.W_DOUBLE - 4 - label_w) / 4
    h_a = 19.0
    h_b = len(pre_rows) * row_h
    h_c = len(post_rows) * row_h
    H = 9 + h_a + 9 + h_b + 13 + h_c + 9
    fig = sp.figure(sp.W_DOUBLE, H)
    W = sp.W_DOUBLE

    # ---- a: fate of the candidate records ------------------------------------------------
    ax = fig.add_axes(
        [(label_w + 2) / W, (H - 9 - h_a) / H, (W - label_w - 40) / W, h_a / H]
    )
    ys = np.arange(len(TIERS))[::-1]
    grey_pre, grey_post = "#D2D2D2", "#8F8F8F"
    for y, t in zip(ys, TIERS):
        b = B[t]
        pre, post, fin = b["pre_total"][0], b["post_total"][0], b["final"][0]
        assert pre + post + fin == n_cand
        ax.barh(y, fin, height=0.66, color=sp.TIER_COLORS[t], lw=0)
        ax.barh(y, post, left=fin, height=0.66, color=grey_post, lw=0)
        ax.barh(y, pre, left=fin + post, height=0.66, color=grey_pre, lw=0)
        ax.text(
            fin / 2,
            y,
            f"{fmt(fin)} scored",
            ha="center",
            va="center",
            fontsize=5,
            color="#222222" if t == "unfiltered" else "white",
        )
        ax.text(
            n_cand * 1.01,
            y,
            f"{fmt(pre)} before, {fmt(post)} after scoring",
            ha="left",
            va="center",
            fontsize=5,
            color="#444444",
        )
        ax.text(
            -0.01,
            y,
            t,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=5.5,
            color=sp.TIER_COLORS[t],
            fontweight="bold",
        )
    ax.set_xlim(0, n_cand)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(
        f"Candidate records (entry, shift table, assembly, entity; n = {n_cand:,})"
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=sp.TIER_COLORS["moderate"], label="scored (the tier's chains)"),
            Patch(color=grey_post, label="removed after scoring"),
            Patch(color=grey_pre, label="removed before scoring"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=3,
        handlelength=1.0,
        fontsize=5,
        borderaxespad=0.2,
        columnspacing=1.2,
    )
    sp.fig_label(fig, "a", 1, 3)

    # ---- b: pre-scoring criteria -------------------------------------------------------
    xmax_b = 7000
    y_b = 9 + h_a + 9
    draw_grid(fig, pre_rows, vals, 2, y_b, row_h, col_w, label_w, xmax_b)
    sp.fig_label(fig, "b", 1, y_b - 4)
    fig.text(
        (label_w + 2) / W,
        1 - (y_b + h_b + 6.5) / H,
        "Records removed before scoring: by the criterion (grey bar, first number) and by this criterion alone (coloured bar, in parentheses)",
        fontsize=5.5,
        ha="left",
        va="top",
        color="#444444",
    )

    # ---- c: post-scoring -------------------------------------------------------------
    y_c = y_b + h_b + 13
    xmax_c = 1350
    draw_grid(fig, post_rows, vals, 2, y_c, row_h, col_w, label_w, xmax_c)
    sp.fig_label(fig, "c", 1, y_c - 4)
    fig.text(
        (label_w + 2) / W,
        1 - (y_c + h_c + 6.5) / H,
        "Records affected after scoring (shift-type rows: records whose shift type is rejected); same encoding as b",
        fontsize=5.5,
        ha="left",
        va="top",
        color="#444444",
    )

    out = sp.save(fig, "S3")
    print("wrote", out, f"({W:.0f} x {H:.0f} mm)")

    # source data: every parsed row, long format
    import csv

    dd = sp.out_dir("suppfigs_a_data")
    dd.mkdir(parents=True, exist_ok=True)
    with open(dd / "S3_filter_losses.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tier", "stage", "criterion", "threshold", "filtered", "unique"])
        for t in TIERS:
            b = B[t]
            for key, (f, u, thr, _sec) in b["pre"].items():
                w.writerow([t, "pre-scoring", key, thr or "", f, u])
            for a, (f, u) in b["rej"].items():
                w.writerow([t, "offset rejection (shift type)", a, "", f, u])
            for key, (f, u, thr, _sec) in b["post"].items():
                w.writerow([t, "post-scoring", key, thr or "", f, u])
            for key in ("pre_total", "post_total", "remaining", "final"):
                w.writerow([t, "total", key, "", b[key][0], ""])
    return B


if __name__ == "__main__":
    main()
