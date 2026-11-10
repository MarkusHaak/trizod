#!/usr/bin/env python3
"""Fig. 1c -- what comparability buys: TriZOD against CheZOD1325 in chains, residues, clusters.

Log-scale grouped bars for three like-for-like counts:

* **chains** -- scored chains in the release (one Parquet row each) against the
  1,325 CheZOD1325 proteins;
* **scored residues** -- residues with a released G-score (``mask`` true) against
  the residues with a published CheZOD1325 Z-score (value != 999);
* **sequence clusters** -- clusters at 50 % identity / 80 % coverage, computed for
  both sets with *identical* MMseqs2 flags: ``easy-cluster --min-seq-id 0.5 -c 0.8``
  plus the build's ``COMMON`` block (``trizod.dataset.mmseqs.COMMON``:
  ``--alignment-mode 3 --cov-mode 0 -s 7.5 --comp-bias-corr 0 --mask 0``),
  single-threaded so the counts are reproducible. The TriZOD bar uses the
  like-for-like scope -- distinct non-bound sequences of at least 20 residues,
  because CheZOD1325 excluded complexes -- and a marker shows the count over all
  distinct scored sequences.

CheZOD117 is drawn as a reference tick in each group: its 117 published proteins,
the residues of their published sequences (the per-residue CheZOD117 scores are
not distributed with the local data, so the tick counts sequence residues), and
their clusters under the same flags.

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
Outputs: ``<out-dir>/fig1c.pdf`` (vector, sized for the right-hand
third of a 180-mm figure) and ``<out-dir>/source_data/fig1c.csv``; prints the plotted
numbers and compares them with the writing plan's preview values (Appendix A13).
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
FIG_W, FIG_H = 60.0, 84.0  # height matches the Fig. 1b cartoon beside it
AX_BOX = (11.0, 9.0, 47.0, 66.0)  # left, top, width, height (mm)
YLIM = (2e1, 6e6)
GROUPS = [
    ("chains", "chains"),
    ("residues", "scored\nresidues"),
    ("clusters", "sequence\nclusters (50%)"),
]
C_TRIZOD = sp.CAT["neutral"]
C_CHEZOD = sp.CAT["chezod"]
C_TICK = "#9A6A00"  # darker orange for the CheZOD117 reference tick


def _fmt(v: float) -> str:
    return f"{v:,.0f}"


def plot(d: pd.DataFrame, out_name: str) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig = sp.figure(FIG_W, FIG_H)
    left, top, w, h = AX_BOX
    ax = fig.add_axes([left / FIG_W, 1 - (top + h) / FIG_H, w / FIG_W, h / FIG_H])
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    v = d.set_index(["quantity", "set"]).value
    bw = 0.34
    for gi, (q, _) in enumerate(GROUPS):
        x_tz, x_cz = gi - bw / 2 - 0.01, gi + bw / 2 + 0.01
        tz, cz = v[(q, "TriZOD")], v[(q, "CheZOD1325")]
        ax.bar(
            x_tz, tz - YLIM[0], bottom=YLIM[0], width=bw, color=C_TRIZOD, lw=0, zorder=2
        )
        ax.bar(
            x_cz, cz - YLIM[0], bottom=YLIM[0], width=bw, color=C_CHEZOD, lw=0, zorder=2
        )
        # counts inside the bars (rotated), fold change above the TriZOD bar
        for x, val, col in ((x_tz, tz, "white"), (x_cz, cz, "black")):
            ax.text(
                x,
                YLIM[0] * 1.35,
                _fmt(val),
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=5.0,
                color=col,
                zorder=4,
            )
        top = max(
            tz, v.get((q, "TriZOD (all sequences)"), tz)
        )  # clear the dotted marker
        ax.text(
            x_tz,
            top * 1.25,
            f"{tz / cz:.1f}×",
            ha="center",
            va="bottom",
            fontsize=6.0,
            fontweight="bold",
            color=C_TRIZOD,
        )
        # CheZOD117 reference tick across the CheZOD1325 bar
        t117 = v[(q, "CheZOD117")]
        ax.plot(
            [x_cz - bw * 0.62, x_cz + bw * 0.62],
            [t117, t117],
            color=C_TICK,
            lw=1.1,
            zorder=3,
        )
    # all-sequence clusters: open marker above the like-for-like TriZOD bar
    gi = [q for q, _ in GROUPS].index("clusters")
    x_tz = gi - bw / 2 - 0.01
    all_c = v[("clusters", "TriZOD (all sequences)")]
    ax.plot(
        [x_tz - bw / 2, x_tz + bw / 2],
        [all_c, all_c],
        color=C_TRIZOD,
        lw=0.6,
        ls=(0, (1.5, 1.0)),
        zorder=3,
    )
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([lab for _, lab in GROUPS])
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_xlim(-0.55, len(GROUPS) - 0.45)
    ax.set_ylabel("count")
    ax.yaxis.grid(True, which="major", color="0.88", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        Patch(color=C_TRIZOD, label="TriZOD"),
        Patch(color=C_CHEZOD, label="CheZOD1325"),
        Line2D([], [], color=C_TICK, lw=1.1, label="CheZOD117"),
        Line2D(
            [],
            [],
            color=C_TRIZOD,
            lw=0.6,
            ls=(0, (1.5, 1.0)),
            label="TriZOD, incl. bound and <20 aa",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(left / FIG_W, 1 - 0.6 / FIG_H),
        ncol=2,
        fontsize=5.0,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=0.8,
        labelspacing=0.25,
        borderaxespad=0,
        borderpad=0,
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
    d.to_csv(args.out_dir / "source_data" / "fig1c.csv", index=False)
    plot(d, "fig1c")
    print(f"wrote {args.out_dir / 'fig1c.pdf'}")


if __name__ == "__main__":
    main()
