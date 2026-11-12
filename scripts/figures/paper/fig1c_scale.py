#!/usr/bin/env python3
"""Fig. 1b -- scale: TriZOD against CheZOD1325 in chains, residues and sequence clusters.

The panel is a dumbbell plot on one log axis, one row per like-for-like count, with the
fold change written above each row (the axis is logarithmic, so equal folds span equal
distances):

* **chains** -- scored chains in the release (one Parquet row each) against the
  1,325 CheZOD1325 proteins;
* **residues** -- residues with a released G-score (``mask`` true) against the
  residues with a published CheZOD1325 Z-score (value != 999);
* **sequence clusters** -- clusters at 50 % identity / 80 % coverage, computed for
  both sets with *identical* MMseqs2 flags: ``easy-cluster --min-seq-id 0.5 -c 0.8``
  plus the build's ``COMMON`` block (``trizod.dataset.mmseqs.COMMON``:
  ``--alignment-mode 3 --cov-mode 0 -s 7.5 --comp-bias-corr 0 --mask 0``),
  single-threaded so the counts are reproducible. The TriZOD point uses the
  like-for-like scope -- distinct non-bound sequences of at least 20 residues,
  because CheZOD1325 excluded complexes.

The analysis (``compute()``) also counts the clusters over all distinct scored
sequences and the CheZOD117 reference values; they stay in the printed check and are
no longer drawn (Fig. 1 was simplified on 11 Nov 2026), so the Source Data holds only
the six plotted values.

Inputs (read-only): the release Parquet (``sequence``, ``is_bound``, ``length``,
``mask``); ``data/external/chezod/protein_nmr_1325/allseqs1325.txt`` (sequences;
the sibling ``allseqs1325.fasta`` is malformed) and ``allscores1325newest.txt``
(published Z-scores, 999 = missing); ``data/external/chezod/seth_github/
CheZOD117_test_set_sequences.fasta``. Clustering results are cached in
``--work-dir`` (default ``tmp/fig1c_mmseqs`` in the code checkout), keyed by a
hash of the input sequences and the command.

Run (code worktree root):
    uv run --extra figures --with pyarrow python scripts/figures/paper/fig1c_scale.py \
        --out-dir <draft>/Figures/paper
Outputs: ``<out-dir>/fig1c.pdf`` (vector, ``W_DOUBLE`` - 105 - 3 = 52 mm wide and
46.6 mm high, placed at 1:1 beside the 105-mm overview diagram ``fig1a_overview.tex``
with a 3-mm gap, so the two fill the 160-mm text width; the file keeps its old name) and
``<out-dir>/source_data/fig1c.csv``; prints the plotted numbers and compares them
with the writing plan's preview values (Appendix A13).
"""

from __future__ import annotations

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

sys.path.insert(0, str(sp.REPO))
from trizod.dataset.mmseqs import COMMON  # noqa: E402

CZ1325_SEQS = sp.CHEZOD_1325 / "allseqs1325.txt"
CZ1325_SCORES = sp.CHEZOD_1325 / "allscores1325newest.txt"
CZ117_SEQS = sp.CHEZOD / "seth_github" / "CheZOD117_test_set_sequences.fasta"
MIN_LEN = 20  # like-for-like scope: chains of at least 20 aa (the build's assembly cut)
CLUSTER_ARGS = ["--min-seq-id", "0.5", "-c", "0.8", *COMMON, "--threads", "1"]

# writing plan, Fig. 1 table and Appendix A13 (for the self-check only)
PLAN = {
    "trizod_chains": 16851,
    "trizod_residues": 1560162,
    "trizod_clusters_like": 6531,
    "trizod_clusters_all": 8486,
    "trizod_seqs_like": 9482,
    "trizod_seqs_all": 12745,
    "cz1325_chains": 1325,
    "cz1325_residues": 149672,
    "cz1325_clusters": 1316,
}


# ----------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------
def read_chezod1325() -> tuple[list[str], int]:
    """CheZOD1325 sequences (one per protein, duplicates kept) and scored residues."""
    seqs = []
    for line in open(CZ1325_SEQS):
        f = line.split()
        if len(f) >= 2:
            seqs.append(f[1])
    n_res = 0
    for line in open(CZ1325_SCORES):
        line = line.strip()
        if line.startswith("["):
            vals = [float(x) for x in line.strip("[]").split(",") if x.strip()]
            n_res += sum(1 for v in vals if v != 999)
    return seqs, n_res


def mmseqs_clusters(name: str, seqs: list[str], work: Path) -> dict:
    """Number of 50/80 clusters over ``seqs`` (cached by input + command hash)."""
    cmd_tail = list(CLUSTER_ARGS)
    key = hashlib.sha256(
        ("\n".join(seqs) + "|" + " ".join(cmd_tail)).encode()
    ).hexdigest()[:16]
    cache = work / f"clusters_{name}_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    if shutil.which("mmseqs") is None:
        raise SystemExit("mmseqs not on PATH (needed once to fill the cache)")
    wd = work / f"clu_{name}_{key}"
    wd.mkdir(parents=True, exist_ok=True)
    fa = wd / "in.fasta"
    with open(fa, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">s{i}\n{s}\n")
    cmd = [
        "mmseqs",
        "easy-cluster",
        str(fa),
        str(wd / "res"),
        str(wd / "tmp"),
        *cmd_tail,
    ]
    with open(wd / "mmseqs.log", "w") as h:
        subprocess.run(cmd, check=True, stdout=h, stderr=subprocess.STDOUT)
    reps = {ln.split("\t")[0] for ln in open(wd / "res_cluster.tsv") if ln.strip()}
    version = subprocess.run(
        ["mmseqs", "version"], capture_output=True, text=True
    ).stdout.strip()
    out = {
        "name": name,
        "n_seqs": len(seqs),
        "n_clusters": len(reps),
        "mmseqs": version,
        "cmd": "mmseqs easy-cluster " + " ".join(cmd_tail),
    }
    cache.write_text(json.dumps(out, indent=1))
    shutil.rmtree(wd / "tmp", ignore_errors=True)
    return out


def compute(work: Path) -> pd.DataFrame:
    df = sp.load_main(columns=["id", "sequence", "is_bound", "length", "mask"])
    tz_chains = len(df)
    tz_res = int(sum(np.asarray(m, bool).sum() for m in df["mask"]))
    seqs_all = sorted(set(df.sequence))
    seqs_like = sorted(
        set(df.sequence[(~df.is_bound.astype(bool)) & (df.length >= MIN_LEN)])
    )
    cz, cz_res = read_chezod1325()
    cz117 = list(sp.read_fasta(CZ117_SEQS).values())

    c_like = mmseqs_clusters("trizod_nonbound20", seqs_like, work)
    c_all = mmseqs_clusters("trizod_all", seqs_all, work)
    c_cz = mmseqs_clusters("chezod1325", cz, work)
    c_117 = mmseqs_clusters("chezod117", cz117, work)

    rows = [
        ("chains", "TriZOD", tz_chains, "scored chains (Parquet rows)"),
        ("chains", "CheZOD1325", len(cz), "proteins (allseqs1325.txt)"),
        ("chains", "CheZOD117", len(cz117), "proteins (published sequences)"),
        ("residues", "TriZOD", tz_res, "residues with a released G-score (mask)"),
        (
            "residues",
            "CheZOD1325",
            cz_res,
            "residues with a published Z-score (!= 999)",
        ),
        (
            "residues",
            "CheZOD117",
            sum(map(len, cz117)),
            "residues of the published sequences",
        ),
        (
            "clusters",
            "TriZOD",
            c_like["n_clusters"],
            f"50/80 clusters of {c_like['n_seqs']} distinct non-bound sequences >= {MIN_LEN} aa",
        ),
        (
            "clusters",
            "TriZOD (all sequences)",
            c_all["n_clusters"],
            f"50/80 clusters of all {c_all['n_seqs']} distinct scored sequences",
        ),
        (
            "clusters",
            "CheZOD1325",
            c_cz["n_clusters"],
            f"50/80 clusters of {c_cz['n_seqs']} sequences",
        ),
        (
            "clusters",
            "CheZOD117",
            c_117["n_clusters"],
            f"50/80 clusters of {c_117['n_seqs']} sequences",
        ),
    ]
    out = pd.DataFrame(rows, columns=["quantity", "set", "value", "definition"])
    out["mmseqs_command"] = np.where(out.quantity == "clusters", c_like["cmd"], "")
    out["mmseqs_version"] = np.where(out.quantity == "clusters", c_like["mmseqs"], "")
    base = out[out.set == "CheZOD1325"].set_index("quantity").value
    out["ratio_to_chezod1325"] = [v / base[q] for q, v in zip(out.quantity, out.value)]
    out.attrs["n_seqs_like"] = c_like["n_seqs"]
    out.attrs["n_seqs_all"] = c_all["n_seqs"]
    return out


# ----------------------------------------------------------------------------------
# plot
# ----------------------------------------------------------------------------------
# Placed at 1:1 beside the overview diagram of Fig. 1a (105 x 46.6 mm, TikZ): the two
# fill the full text width (W_DOUBLE = 160 mm) with a 3-mm gap, and the heights match.
PANEL_A_W = 105.0  # width of fig1a_overview.tex (three 31-mm boxes, two 6-mm gaps)
GAP_W = 3.0  # \hfill gap between the two panels in sections/fig1.tex
FIG_W, FIG_H = sp.W_DOUBLE - PANEL_A_W - GAP_W, 46.6
AX_BOX = (14.5, 9.6, 34.5, 28.0)  # left, top, width, height (mm)
XLIM = (7e2, 3e6)
ROWS = [  # (quantity, row label), top to bottom
    ("chains", "Chains"),
    ("residues", "Residues"),
    ("clusters", "Sequence\nclusters"),
]
C_TRIZOD = sp.CAT["neutral"]
C_CHEZOD = sp.CAT["chezod"]
C_LINK = "#B3B3B3"
MS = 4.6  # marker diameter (pt)


def plot(d: pd.DataFrame, out_name: str) -> None:
    from matplotlib import ticker as mticker
    from matplotlib.lines import Line2D

    fig = sp.figure(FIG_W, FIG_H)
    left, top, w, h = AX_BOX
    ax = fig.add_axes([left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    n = len(ROWS)
    ax.set_ylim(n - 0.45, -0.75)  # top row first; headroom for the fold labels
    v = d.set_index(["quantity", "set"]).value
    for i, (q, _) in enumerate(ROWS):
        tz, cz = v[(q, "TriZOD")], v[(q, "CheZOD1325")]
        ax.plot([cz, tz], [i, i], color=C_LINK, lw=1.2, solid_capstyle="butt", zorder=2)
        for x, c in ((cz, C_CHEZOD), (tz, C_TRIZOD)):
            ax.plot(
                x, i, "o", ms=MS, color=c, mec="white", mew=0.6, zorder=3, clip_on=False
            )
        ax.text(
            np.sqrt(tz * cz),  # midpoint on the log axis
            i - 0.2,
            f"{tz / cz:.1f}\u00d7",
            ha="center",
            va="bottom",
            fontsize=sp.FS_BODY,
            fontweight="bold",
            color="black",
        )
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in ROWS], fontsize=sp.FS_BODY, linespacing=1.1)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_minor_locator(mticker.NullLocator())  # decades only: fewer marks
    ax.xaxis.grid(True, which="major", color="0.88", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Count (log scale)")
    handles = [
        Line2D([], [], ls="", marker="o", ms=MS, color=C_TRIZOD, label="TriZOD"),
        Line2D([], [], ls="", marker="o", ms=MS, color=C_CHEZOD, label="CheZOD1325"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(left / FIG_W, 1 - 4.6 / FIG_H),
        ncol=2,
        fontsize=sp.FS_TICK,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=1.2,
        borderaxespad=0,
        borderpad=0,
    )
    # panel letter on a baseline 2.5 mm below the top edge, as "a" in fig1a_overview.tex
    fig.text(
        0.0,
        1 - 2.5 / FIG_H,
        "b",
        fontsize=sp.FS_PANEL,
        fontweight="bold",
        ha="left",
        va="baseline",
    )
    sp.save(fig, out_name)


# ----------------------------------------------------------------------------------
def _check(name, got, plan, tol=0):
    ok = abs(got - plan) <= tol
    print(f"  {'OK  ' if ok else 'DIFF'} {name}: data {got:,}  plan {plan:,}")
    return ok


def main() -> None:
    ap = sp.cli(__doc__)
    ap.add_argument("--work-dir", type=Path, default=sp.TMP / "fig1c_mmseqs")
    args = sp.parse_args(ap)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    d = compute(args.work_dir)
    print(d.drop(columns=["mmseqs_command", "mmseqs_version"]).to_string(index=False))
    print("cluster command:", d.mmseqs_command[d.quantity == "clusters"].iloc[0])
    v = d.set_index(["quantity", "set"]).value
    print(
        "check against the writing plan (A13; cluster counts may differ by a few between"
    )
    print("single- and multi-threaded MMseqs2 runs):")
    _check("TriZOD chains", int(v[("chains", "TriZOD")]), PLAN["trizod_chains"])
    _check(
        "TriZOD scored residues",
        int(v[("residues", "TriZOD")]),
        PLAN["trizod_residues"],
    )
    _check(
        "TriZOD like-for-like sequences",
        d.attrs["n_seqs_like"],
        PLAN["trizod_seqs_like"],
    )
    _check("TriZOD all sequences", d.attrs["n_seqs_all"], PLAN["trizod_seqs_all"])
    _check(
        "TriZOD like-for-like clusters",
        int(v[("clusters", "TriZOD")]),
        PLAN["trizod_clusters_like"],
        10,
    )
    _check(
        "TriZOD all-sequence clusters",
        int(v[("clusters", "TriZOD (all sequences)")]),
        PLAN["trizod_clusters_all"],
        10,
    )
    _check(
        "CheZOD1325 proteins", int(v[("chains", "CheZOD1325")]), PLAN["cz1325_chains"]
    )
    _check(
        "CheZOD1325 scored residues",
        int(v[("residues", "CheZOD1325")]),
        PLAN["cz1325_residues"],
    )
    _check(
        "CheZOD1325 clusters",
        int(v[("clusters", "CheZOD1325")]),
        PLAN["cz1325_clusters"],
        10,
    )
    for q in ("chains", "residues", "clusters"):
        print(f"  ratio {q}: {v[(q, 'TriZOD')] / v[(q, 'CheZOD1325')]:.2f}x")
    print(
        f"  ratio clusters (all sequences): {v[('clusters', 'TriZOD (all sequences)')] / v[('clusters', 'CheZOD1325')]:.2f}x"
    )

    (args.out_dir / "source_data").mkdir(parents=True, exist_ok=True)
    plotted = d[d.set.isin(["TriZOD", "CheZOD1325"])]
    plotted.to_csv(args.out_dir / "source_data" / "fig1c.csv", index=False)
    plot(d, "fig1c")
    print(f"wrote {args.out_dir / 'fig1c.pdf'}")


if __name__ == "__main__":
    main()
