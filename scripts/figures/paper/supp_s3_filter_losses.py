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
     completeness), one column per tier, shared linear x-axis. The cosolvent filter
     searches ~30 terms (synonyms, abbreviations, deuterated forms); panel b pools them
     into one row per cosolvent (``COSOLVENT_GROUPS``). The log counts each term on its
     own, and a record can match several terms of one cosolvent, so the pooled counts
     cannot be summed from the log: they are recomputed from the record-level filter
     masks (``cosolvent_group_counts``), and the per-term counts of that recomputation
     must equal the log's, or the script stops. Supplementary Table 3 keeps every term;
  c  post-scoring losses: atom types rejected for a residual offset beyond the tier's
     max-offset (tolerant drops the atom type only; moderate and strict drop the record)
     and the post-scoring completeness checks.

The recomputation rebuilds the tolerant, moderate and strict pre-filter frames from the
build's BmrbEntry pickles (``$TRIZOD_PKL_DIR``, default ``<root>/tmp/bmrb_entries``; read
only, never re-pickled) with ``trizod.trizod.create_peptide_dataframe`` and
``trizod.pipeline.prefilter_dataframe``, in a subprocess that never imports pyarrow
(about 10 min); the result is cached in ``<cache>/paper_figures/S3_cosolvent_groups_<pipeline>.json``.

Run:  uv run --extra figures --with pyarrow python scripts/figures/paper/supp_s3_filter_losses.py \
          --out-dir <draft>/Figures/paper
Outputs: <out-dir>/S3.pdf, <out-dir>/suppfigs_a_data/S3_filter_losses.csv (every term of
the log) and S3_cosolvent_groups.csv (the pooled rows of panel b).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
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
    # the perturbing-cosolvent rows are the pooled COSOLVENT_GROUPS (see below)
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
# atom types of the offset rejection, in the order and with the names of the other figures
REJ_ATOMS = ["CA", "CB", "C", "N", "H", "HA", "HB"]
ATOM_LABEL = {
    "CA": "Cα",
    "CB": "Cβ",
    "C": "C′",
    "N": "N",
    "H": "HN",
    "HA": "Hα",
    "HB": "Hβ",
}

# One row per cosolvent in panel b: the filter's search terms (trizod.trizod.COSOLVENT_TOKENS,
# the keys of the build log) pooled by the chemical they name. A term the log reports that
# no group lists stops the script, so a new term cannot vanish from the figure silently.
COSOLVENT_GROUPS = [
    ("Guanidinium", ["guanidin", "GdmCl", "Gdn-Hcl"]),
    ("Urea", ["urea"]),
    ("TFE", ["TFE", "trifluoroethanol", "trifluoro ethanol"]),
    ("HFIP", ["HFIP", "hexafluoroisopropanol", "hexafluoro-2-propanol"]),
    ("DMSO", ["DMSO", "dimethyl sulfoxide", "dimethylsulfoxide"]),
    ("Methanol", ["methanol", "MeOH", "CD3OH", "CD3OD"]),
    ("Ethanol", ["ethanol", "EtOH"]),
    ("Acetonitrile", ["acetonitrile", "ACN", "CD3CN"]),
    ("Chloroform", ["chloroform", "CHCl3", "CDCl3"]),
    ("Isopropanol", ["isopropanol", "2-propanol", "isopropyl alcohol"]),
    ("1-Propanol", ["n-propanol", "1-propanol"]),
    ("Hexane", ["hexane"]),
    ("Pentane", ["pentane"]),
    ("Dioxane", ["dioxane"]),
]
GROUP_CACHE = sp.TMP / "paper_figures" / f"S3_cosolvent_groups_{sp.PIPELINE}.json"
PKL_DIR = sp._env_path("TRIZOD_PKL_DIR", sp._L.pkl_dir)

# Runs in a fresh interpreter of the project environment (no pyarrow, no figures extra):
# rebuilds each tier's pre-filter frame exactly as trizod.trizod.run_scoring_pipeline does
# and counts, per search term and per group, the records that fail it ("filtered") and
# those that fail it and no other pre-scoring criterion ("unique"; print_filter_losses).
_GROUP_WORKER = r"""
import json, logging, sys, warnings
from pathlib import Path
import pandas as pd
logging.disable(logging.CRITICAL)
warnings.simplefilter("ignore")
from pandarallel import pandarallel
from trizod.dataset.withdrawn import load_withdrawn
from trizod.pipeline import load_cached_entry, prefilter_dataframe
from trizod.trizod import create_peptide_dataframe, filter_defaults

pkl_dir, job_fp, out_fp = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
job = json.loads(Path(job_fp).read_text())
withdrawn = set(load_withdrawn())
entries = {}
for fp in sorted(pkl_dir.glob("*.pkl")):
    if fp.stem in withdrawn:
        continue
    e = load_cached_entry(fp)  # None if stamped by other parsing code
    if e is None:
        raise SystemExit(f"stale or unreadable pickle {fp}: not the build's parser")
    entries[fp.stem] = e
bmrb_entries = pd.DataFrame(
    [(e,) for e in entries.values()], index=list(entries), columns=["entry"]
)
pandarallel.initialize(verbose=0, progress_bar=False)
out = {"n_entries": len(bmrb_entries), "tiers": {}}
for tier in job["tiers"]:
    p = filter_defaults.loc[tier]
    df = create_peptide_dataframe(
        bmrb_entries,
        perturbing_cosolvents=list(p["perturbing-cosolvents"]),
        keywords=list(p["keywords-blacklist"]),
        return_default=bool(p["default-conditions"]),
        assume_si=bool(p["unit-assumptions"]),
        fix_outliers=bool(p["unit-corrections"]),
        keyword_search_scope=str(p["keyword-search-scope"]),
    )
    sels = prefilter_dataframe(
        df,
        method_whitelist=list(p["exp-method-whitelist"]),
        method_blacklist=list(p["exp-method-blacklist"]),
        temperature_range=list(p["temperature-range"]),
        ionic_strength_range=list(p["ionic-strength-range"]),
        pH_range=list(p["pH-range"]),
        peptide_length_range=(list(p["peptide-length-range"]) + [float("inf")])[:2],
        min_backbone_shift_types=int(p["min-backbone-shift-types"]),
        min_backbone_shift_positions=int(p["min-backbone-shift-positions"]),
        min_backbone_shift_fraction=float(p["min-backbone-shift-fraction"]),
        max_noncanonical_fraction=float(p["max-noncanonical-fraction"]),
        max_x_fraction=float(p["max-x-fraction"]),
        keywords=list(p["keywords-blacklist"]),
        perturbing_cosolvents=list(p["perturbing-cosolvents"]),
        exclude_paramagnetic=bool(p["exclude-paramagnetic"]),
        physical_state_blacklist=list(p["physical-state-blacklist"]),
        method_fallback=str(p["method-fallback"]),
    )
    sels_cos, sels_all = sels[4], sels[6]
    fail = {k: (~v).fillna(False).astype(bool).to_numpy() for k, v in sels_all.items()}

    def counts(keys):
        keys = [k for k in keys if k in fail]
        if not keys:
            return None
        f = fail[keys[0]].copy()
        for k in keys[1:]:
            f |= fail[k]
        other = f & False
        for k, v in fail.items():
            if k not in keys:
                other |= v
        return [int(f.sum()), int((f & ~other).sum())]

    out["tiers"][tier] = {
        "n_records": len(df),
        "terms": {k: counts([k]) for k in sels_cos},
        "groups": {lab: counts(keys) for lab, keys in job["groups"]},
    }
    print(f"  S3 cosolvent groups: {tier} done", file=sys.stderr, flush=True)
Path(out_fp).write_text(json.dumps(out))
"""


def cosolvent_group_counts(B: dict) -> dict:
    """``{tier: {group: (filtered, unique) | None}}`` for COSOLVENT_GROUPS, checked against the log.

    None means the tier applies no term of the group. Cached per pipeline version.
    """
    tiers = [t for t in TIERS if any(sec == "perturbing cosolvent"
                                     for *_x, sec in B[t]["pre"].values())]  # fmt: skip
    listed = {k for _lab, keys in COSOLVENT_GROUPS for k in keys}
    for t in tiers:
        terms = [k for k, v in B[t]["pre"].items() if v[3] == "perturbing cosolvent"]
        unknown = [k for k in terms if k not in listed]
        if unknown:
            raise SystemExit(
                f"cosolvent terms without a group in COSOLVENT_GROUPS: {unknown}"
            )
    if GROUP_CACHE.exists():
        res = json.loads(GROUP_CACHE.read_text())
    else:
        print(f"  S3: recomputing the pooled cosolvent counts from {PKL_DIR} (~10 min)")
        GROUP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        job = GROUP_CACHE.with_suffix(".job.json")
        out = GROUP_CACHE.with_suffix(".out.json")
        code = GROUP_CACHE.with_suffix(".worker.py")
        job.write_text(json.dumps({"tiers": tiers, "groups": COSOLVENT_GROUPS}))
        code.write_text(_GROUP_WORKER)
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        subprocess.run(["uv", "run", "python", str(code), str(PKL_DIR), str(job), str(out)],
                       check=True, cwd=sp.REPO, env=env)  # fmt: skip
        res = json.loads(out.read_text())
        GROUP_CACHE.write_text(json.dumps(res, indent=1))
        for f in (job, out, code):
            f.unlink()
    # the recomputation must reproduce the build log term by term
    bad = []
    for t in tiers:
        r = res["tiers"][t]
        if r["n_records"] != B[t]["pre_total"][1]:
            bad.append((t, "records", r["n_records"], B[t]["pre_total"][1]))
        for k, v in B[t]["pre"].items():
            if v[3] == "perturbing cosolvent" and r["terms"].get(k) != [v[0], v[1]]:
                bad.append((t, k, r["terms"].get(k), [v[0], v[1]]))
    if bad:
        raise SystemExit(
            f"pooled cosolvent counts do not reproduce the build log: {bad[:5]} "
            f"(delete {GROUP_CACHE} if the build changed)"
        )
    return {
        t: {lab: (None if t not in tiers or res["tiers"][t]["groups"][lab] is None
                  else tuple(res["tiers"][t]["groups"][lab]))
            for lab, _keys in COSOLVENT_GROUPS}
        for t in TIERS
    }  # fmt: skip


POST_ROWS = [
    ("bb shift types", "Backbone shift types"),
    ("bb shift positions", "Backbone shift positions"),
    ("bb shift fraction", "Backbone shift fraction"),
    ("error in computation", "Error in computation"),
    ("rejected due to any offset", "Chain dropped for an offset"),
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
                    "\N{EN DASH}",
                    va="center",
                    ha="left",
                    fontsize=sp.FS_SMALL,
                    color="#9A9A9A",
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
                    fontsize=sp.FS_SMALL,
                    color="white" if dark else "#222222",
                )
            else:
                ax.text(
                    filt + xmax * 0.015,
                    y,
                    txt,
                    va="center",
                    ha="left",
                    fontsize=sp.FS_SMALL,
                    color="#222222",
                )
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(axis="x", labelsize=sp.FS_SMALL)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        # section separators
        for y, row in zip(ys, rows):
            if row[0] == "section":
                ax.axhline(y, color="#D0D0D0", lw=0.4, zorder=0)
        if header:
            ax.set_title(t, fontsize=sp.FS_BODY, color="black", pad=2)
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
                fontsize=sp.FS_TICK,
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
                fontsize=sp.FS_SMALL,
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
    # Rows follow the log: a criterion no tier reports is dropped, and a keyword criterion
    # that PRE_ROWS does not list is appended to its section in log order. The cosolvent
    # terms are pooled into COSOLVENT_GROUPS (one row per cosolvent).
    groups = cosolvent_group_counts(B)
    in_log = {k for t in TIERS for k in B[t]["pre"]}
    listed = {k for _sec, items in PRE_ROWS for k, _lab in items}
    kw_sec = "Keyword in the sample description"
    extra = []
    for t in TIERS:
        for key, (_f, _u, _thr, sec) in B[t]["pre"].items():
            if sec == "keyword" and key not in listed and key not in extra:
                extra.append(key)
    if extra:
        print(f"  {kw_sec}: rows added from the log: {', '.join(extra)}")
    dropped = sorted(listed - in_log)
    if dropped:
        print(f"  rows dropped (no tier reports them): {', '.join(dropped)}")
    pre_rows, post_rows = [], []
    for sec, items in PRE_ROWS:
        pre_rows.append(("section", None, sec, False))
        items = [(k, lab) for k, lab in items if k in in_log]
        if sec == kw_sec:
            items += [(k, k) for k in extra]
        for key, lab in items:
            pre_rows.append(("row", key, lab, sec == kw_sec))  # italic: searched words
        if sec == kw_sec:
            pre_rows.append(("section", None, "Perturbing cosolvent", False))
            for lab, _keys in COSOLVENT_GROUPS:
                pre_rows.append(("row", "grp:" + lab, lab, False))
    post_rows.append(("section", None, "Atom type over the offset limit", False))
    for a in REJ_ATOMS:
        post_rows.append(("row", "rej:" + a, ATOM_LABEL[a], False))
    post_rows.append(("section", None, "Completeness after rejection", False))
    for key, lab in POST_ROWS:
        post_rows.append(("row", "post:" + key, lab, False))

    vals = {}
    for t in TIERS:
        v = {}
        for key, (f, u, _thr, _sec) in B[t]["pre"].items():
            v[key] = (f, u)
        for lab, fu in groups[t].items():
            if fu is not None:
                v["grp:" + lab] = fu
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

    # rows that remove no record at any tier carry no data; leave them out (the source
    # data keep them, and Supplementary Table 3 lists every criterion and search term)
    def _empty(row):
        return row[0] == "row" and all(
            (vals[t].get(row[1]) or (0, 0))[0] == 0 for t in TIERS
        )

    empty = [r[2] for r in pre_rows + post_rows if _empty(r)]
    if empty:
        print(
            f"  rows without a removed record at any tier (not drawn): {', '.join(empty)}"
        )
    pre_rows = [r for r in pre_rows if not _empty(r)]
    post_rows = [r for r in post_rows if not _empty(r)]

    row_h = 2.2  # 2.2 mm per row keeps figure and legend on one page
    label_w = 40.0
    col_w = (sp.W_DOUBLE - 4 - label_w) / 4
    h_a = 17.0
    h_b = len(pre_rows) * row_h
    h_c = len(post_rows) * row_h
    # vertical budget (mm): a's key and bars, then for b and c a letter line that also
    # holds the bar key (b only), the tier headers, the rows, tick labels and a centred
    # axis title
    top_a, gap_b, gap_c, bottom = 8.0, 17.0, 14.0, 9.0
    H = top_a + h_a + gap_b + h_b + gap_c + h_c + bottom
    fig = sp.figure(sp.W_DOUBLE, H)
    W = sp.W_DOUBLE

    # ---- a: fate of the candidate records ------------------------------------------------
    ax = fig.add_axes(
        [(label_w + 2) / W, (H - top_a - h_a) / H, (W - label_w - 40) / W, h_a / H]
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
            fontsize=sp.FS_SMALL,
            color="#222222" if t == "unfiltered" else "white",
        )
        ax.text(
            n_cand * 1.01,
            y,
            f"{fmt(pre)} before, {fmt(post)} after scoring",
            ha="left",
            va="center",
            fontsize=sp.FS_SMALL,
            color="#444444",
        )
        ax.text(
            -0.01,
            y,
            t,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=sp.FS_TICK,
            color="black",
        )
    ax.set_xlim(0, n_cand)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Candidate chains (n = {n_cand:,})")  # "chain" only (review 2.1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=sp.TIER_COLORS["moderate"], label="Scored"),
            Patch(color=grey_post, label="Removed after scoring"),
            Patch(color=grey_pre, label="Removed before scoring"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=3,
        handlelength=1.0,
        fontsize=sp.FS_TICK,
        borderaxespad=0.2,
        columnspacing=1.2,
    )
    sp.fig_label(fig, "a", 1, 3)

    # ---- b: pre-scoring criteria -------------------------------------------------------
    def axis_title(y_mm, text):
        """Axis title centred under the four tier columns."""
        x_mid = 2 + label_w + (4 * col_w - 4) / 2
        fig.text(
            x_mid / W, 1 - y_mm / H, text, fontsize=sp.FS_BODY, ha="center", va="top"
        )

    xmax_b = 7000
    y_b = top_a + h_a + gap_b
    draw_grid(fig, pre_rows, vals, 2, y_b, row_h, col_w, label_w, xmax_b)
    sp.fig_label(fig, "b", 1, y_b - 9.5)
    axis_title(y_b + h_b + 5.0, "Chains removed before scoring")
    # bar key on the panel-letter line, right-aligned, clear of the tier headers
    fig.legend(
        handles=[
            Patch(color=FILT_GREY, label="Failing the criterion (first number)"),
            Patch(
                color=sp.TIER_COLORS["moderate"],
                label="Failing only this criterion (in parentheses)",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1 - 2 / W, 1 - (y_b - 10.0) / H),
        ncol=2,
        handlelength=1.0,
        fontsize=sp.FS_TICK,
        borderaxespad=0.0,
        columnspacing=1.2,
    )

    # ---- c: post-scoring -------------------------------------------------------------
    y_c = y_b + h_b + gap_c
    xmax_c = 1350
    draw_grid(fig, post_rows, vals, 2, y_c, row_h, col_w, label_w, xmax_c)
    sp.fig_label(fig, "c", 1, y_c - 6.5)
    axis_title(y_c + h_c + 5.0, "Chains affected after scoring (bars as in b)")

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
    with open(dd / "S3_cosolvent_groups.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tier", "cosolvent", "search_terms", "filtered", "unique"])
        for t in TIERS:
            for lab, keys in COSOLVENT_GROUPS:
                fu = groups[t][lab]
                w.writerow([t, lab, "; ".join(keys), *(fu if fu else ("", ""))])
    return B


if __name__ == "__main__":
    main()
