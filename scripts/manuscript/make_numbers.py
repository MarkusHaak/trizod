#!/usr/bin/env python3
"""Generate the manuscript's numbers layer (``numbers.tex`` + ``numbers_index.md``).

Every dataset number the TriZOD manuscript quotes is computed here from the frozen
2026-10 build and written as one ``\\newcommand`` per number, so the draft never
types a number into prose and a post-rebuild refresh is a single re-run (writing
plan, item (j)).

Inputs (all read-only; resolved under ``--root``, the checkout that holds ``data/``). The
build-specific ones below are the 2026-10 defaults; ``TRIZOD_BUILD_DIR``,
``TRIZOD_RELEASE_DIR``, ``TRIZOD_BUNDLE``, ``TRIZOD_REBUILD_LOG``, ``TRIZOD_PKL_DIR`` and
``TRIZOD_SIDECHAIN_DOC`` select a later build (see :class:`Paths`):

* ``data/interim/build/parquet-2026-10-simplify/{trizod_dataset,trizod_shifts}.parquet``
* ``data/interim/build/release_bundle/trizod-dataset-2026-10/`` (MANIFEST, per-tier
  ``scores/<tier>/scores.json`` labels, ``train/<tier>/*``, ``test/*``)
* ``data/interim/build/{final_dataset,testset,mmseqs}/`` and ``rebuild_2026-10.log``
  (the ``print_filter_losses()`` blocks and the build-step summaries)
* ``data/interim/chezod_verification/`` (CheZOD reproduction, pre-2026-10 code)
* ``data/external/panav_offsets.json``, ``data/external/chezod/protein_nmr_1325/``
* ``data/raw/bmrb_entries/`` (submission dates, sample pressure, organisms, alcohols)
* ``docs/dataset/sidechain-coverage.md`` and ``docs/filtering.md`` (documented values)

Derived intermediates (raw-file scan, mmseqs searches/clusterings) are cached in
``--work-dir`` (default ``tmp/manuscript_numbers`` in the code checkout) so re-runs are fast; pass
``--refresh`` to recompute them. MMseqs2 must be on PATH for the leakage audit, the
like-for-like cluster counts and the case-study clusters; without it those macros
fall back to the writing plan's values and are marked UNREPRODUCED.

Status codes (index column): STABLE (survives the planned rebuild), PENDING-REBUILD
(moves with rebuild fixes (a)-(f)), PREVIEW (exploratory / pre-build analysis to be
recomputed with controls), PROVISIONAL (stale DisProt numbers), UNREPRODUCED (plan
value kept; this script could not recompute it -- the reason is given).

Run (from the code checkout; never in a process that runs the scoring pipeline). Outputs go to
the required ``--out-dir`` (the manuscript-draft directory); ``--write-only`` rewrites them from
the last registry dump in ``--work-dir`` without recomputing:
    uv run --extra figures --with pyarrow python scripts/manuscript/make_numbers.py \
        --out-dir <manuscript-draft> [--root <checkout holding data/>]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from trizod import paths  # noqa: E402
from trizod.offsets import (  # noqa: E402
    lacs_off_ppm_col,
    off_sigma_col,
    total_off_ppm_col,
)

TIERS = ["unfiltered", "tolerant", "moderate", "strict"]
REL = "2026-10"  # release label of the build, reset from its MANIFEST in main()
TCAP = {t: t.capitalize() for t in TIERS}
RANK = {t: i for i, t in enumerate(TIERS)}
ATOMS = ["C", "CA", "CB", "H", "HA", "HB", "N"]
G_DIS = 0.65  # binary disorder cut used throughout the paper
UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

STABLE, PENDING, PREVIEW, PROVISIONAL, UNREPRO = (
    "STABLE",
    "PENDING-REBUILD",
    "PREVIEW",
    "PROVISIONAL",
    "UNREPRODUCED",
)

# Section keys (index grouping = section of first use in the manuscript).
SECTIONS = OrderedDict(
    [
        ("abs", "Abstract, Introduction and contributions (headline values)"),
        ("r1", "R1 / M1 -- BMRB snapshot, parsing and scale (Fig. 1b, 1c; A1, A13)"),
        (
            "r2",
            "R2 / M2 / M3 -- G-score, shift count k and re-referencing (Fig. 1a, 2a, 2b; A8, A10)",
        ),
        ("r3", "R3 / M8 -- CheZOD reproduction (Fig. 2c; A9)"),
        (
            "r4",
            "R4 / M4 / M7 -- tiers, sample conditions and replicate reliability (Fig. 3a, 3b; A2, A7, A10, A12)",
        ),
        (
            "r5",
            "R5 / M5 / M6 / M10 -- dataset assembly, test sets, leakage and baselines (Fig. 3c; A2, A4, A5)",
        ),
        (
            "r6",
            "R6 / Discussion -- DisProt (provisional), failure cases, non-native states, limitations",
        ),
        (
            "m11",
            "M11 / Data availability -- release format, shift companion and MAD flag (A6)",
        ),
        ("par", "Methods parameters (code constants and analysis settings)"),
        ("lit", "Literature and external numbers (A11)"),
        (
            "t3",
            "Supplementary Table T3 / Fig. S3 -- per-filter losses from the rebuild log (A3)",
        ),
        (
            "t7",
            "Supplementary Table T7 -- residue- and chain-level statistics per set (A5)",
        ),
        ("ann", "Supplementary Fig. S4 / Table T5 -- annotations per set (A7)"),
        ("off", "Supplementary Fig. S6 / S12 -- per-atom re-referencing offsets (A8)"),
    ]
)


# ----------------------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------------------
def _neg(s: str) -> str:
    return s.replace("-", r"\ensuremath{-}", 1) if s.startswith("-") else s


def I(n) -> str:  # noqa: E743 -- integer with LaTeX thousands separators
    n = int(round(float(n)))
    s = f"{abs(n):,}".replace(",", "{,}")
    return (r"\ensuremath{-}" if n < 0 else "") + s


def F(x, nd: int = 2, sign: bool = False) -> str:
    """Decimal with fixed digits; text-safe minus; optional explicit plus."""
    x = float(x)
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    if s.startswith("-0") and float(s) == 0:
        s = s[1:]
    if "." in s:
        head, tail = s.split(".")
        neg = head.startswith("-") or head.startswith("+")
        sg, digits = (head[0], head[1:]) if neg else ("", head)
        if len(digits) > 3:
            digits = f"{int(digits):,}".replace(",", "{,}")
        s = f"{sg}{digits}.{tail}"
    return _neg(s)


def P(x, nd: int = 1) -> str:
    """Percentage (x already in percent) WITH the percent sign."""
    return F(x, nd) + r"\%"


def S(s: str) -> str:
    """Plain string, LaTeX-escaped (IDs, dates)."""
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def plainify(v: str) -> str:
    v = (
        v.replace("{,}", ",")
        .replace(r"\%", "%")
        .replace(r"\ensuremath{-}", "-")
        .replace(r"\_", "_")
    )
    v = v.replace(r"\textendash{}", "–").replace("--", "–")
    return v


def _norm(s: str) -> str:
    s = (
        str(s)
        .replace(r"\%", "")
        .replace(r"\ensuremath{-}", "-")
        .replace(r"\_", "_")
        .replace("{,}", ",")
    )
    s = s.replace("−", "-").replace(" ", "").replace(" ", "").replace("%", "")
    return s.lstrip("+")


def plan_matches(value: str, plan: str) -> bool:
    a, b = _norm(value), _norm(plan)
    if a == b:
        return True
    try:
        fa, fb = float(a.replace(",", "")), float(b.replace(",", ""))
    except ValueError:
        return False
    nd = len(b.split(".")[1]) if "." in b else 0
    return round(fa, nd) == round(fb, nd) or abs(fa - fb) < 1e-9


# ----------------------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------------------
@dataclass
class Num:
    name: str
    value: str
    definition: str
    source: str
    status: str
    section: str
    plan: str | None = None
    plan_ok: bool | None = None
    note: str = ""


@dataclass
class Registry:
    items: list = field(default_factory=list)
    names: set = field(default_factory=set)

    def add(self, name, value, definition, source, status, section, plan=None, note=""):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name must be letters only: {name}")
        if name in self.names:
            raise ValueError(f"duplicate macro {name}")
        if section not in SECTIONS:
            raise ValueError(f"unknown section {section}")
        value = str(value)
        if re.search(r"(?<!\\)%|\\\\%", value) or value.count("{") != value.count("}"):
            raise ValueError(f"unsafe LaTeX value for {name}: {value!r}")
        ok = None if plan is None else plan_matches(value, plan)
        self.names.add(name)
        self.items.append(
            Num(name, value, definition, source, status, section, plan, ok, note)
        )
        return value

    def unrepro(
        self,
        name,
        plan_value_latex,
        definition,
        reason,
        section,
        plan_source="writing plan",
    ):
        """Plan value kept verbatim because this script cannot recompute it."""
        return self.add(
            name,
            plan_value_latex,
            definition,
            f"{plan_source}; NOT recomputed: {reason}",
            UNREPRO,
            section,
        )


R = Registry()


# ----------------------------------------------------------------------------------
# paths and loaders
# ----------------------------------------------------------------------------------
def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "interim" / "build").is_dir():
            return p
    raise SystemExit("could not find a checkout with data/interim/build; pass --root")


def env_path(name: str, default: Path) -> Path:
    """``$name`` as a path when set and non-empty, else ``default``."""
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else default


class Paths:
    """Input locations. The defaults are the frozen 2026-10 build under ``--root``; a later
    build is selected with the same environment variables as ``style_paper.py``
    (``scripts/figures/paper/README.md``, "Selecting the build"):

    * ``TRIZOD_BUILD_DIR``: build directory (``final_dataset/``, ``testset/``, ``mmseqs/``);
    * ``TRIZOD_RELEASE_DIR``: ``trizod_dataset.parquet`` + ``trizod_shifts.parquet`` (and the
      deploy FASTA and ``sidechain-coverage.md`` of the post-build steps);
    * ``TRIZOD_BUNDLE``: the release bundle (MANIFEST, ``scores/``, ``train/``, ``test/``);
    * ``TRIZOD_REBUILD_LOG``: the build log (``print_filter_losses()`` blocks, step summaries);
    * ``TRIZOD_PKL_DIR``: the build's BmrbEntry pickle cache (default ``<root>/tmp/bmrb_entries``);
    * ``TRIZOD_DRAW_DIR``: the 2026-08-11 test-set draw (``_tmp/``, ``TriZOD_test_set_clu.tsv``),
      which a rebuild does not redo (default: the build's ``testset/`` when it holds ``_tmp/``,
      else ``<root>/data/interim/build/testset``);
    * ``TRIZOD_SIDECHAIN_DOC``: the side-chain coverage report (default: the release
      directory's ``sidechain-coverage.md`` when ``TRIZOD_RELEASE_DIR`` holds one, else
      ``docs/dataset/sidechain-coverage.md``).

    Release, bundle and log default to fixed 2026-10 names inside the build directory, so
    setting only ``TRIZOD_BUILD_DIR`` fails loudly instead of mixing two builds."""

    def __init__(self, root: Path, work: Path):
        self.root = root
        L = paths.layout(root)
        self.build = env_path("TRIZOD_BUILD_DIR", L.interim_build)
        self.release = env_path(
            "TRIZOD_RELEASE_DIR", self.build / "parquet-2026-10-simplify"
        )
        self.pq = self.release / "trizod_dataset.parquet"
        self.shifts = self.release / "trizod_shifts.parquet"
        self.bundle = env_path(
            "TRIZOD_BUNDLE", self.build / "release_bundle" / "trizod-dataset-2026-10"
        )
        self.manifest = self.bundle / "MANIFEST.json"
        self.fds = self.build / "final_dataset" / "final_dataset_summary.json"
        self.testset = self.build / "testset"
        # the 2026-08-11 test-set draw (testset/_tmp, testset/TriZOD_test_set_clu.tsv); a later
        # build re-resolves the pinned set without redrawing it, so the draw intermediates stay
        # in the build that drew it ($TRIZOD_DRAW_DIR, default: this build's testset/ when it
        # holds them, else <root>/data/interim/build/testset)
        self.draw = env_path(
            "TRIZOD_DRAW_DIR",
            self.testset
            if (self.testset / "_tmp").is_dir()
            else L.interim_build / "testset",
        )
        self.log = env_path("TRIZOD_REBUILD_LOG", self.build / "rebuild_2026-10.log")
        self.pkl = env_path("TRIZOD_PKL_DIR", L.pkl_dir)
        self.chezod_ver = L.interim_chezod_verification
        self.panav = L.ext_panav
        self.cz1325 = L.ext_chezod_1325
        self.raw = L.raw_bmrb
        sc_release = self.release / "sidechain-coverage.md"
        self.sidechain_doc = env_path(
            "TRIZOD_SIDECHAIN_DOC",
            sc_release
            if os.environ.get("TRIZOD_RELEASE_DIR") and sc_release.is_file()
            else REPO / "docs" / "dataset" / "sidechain-coverage.md",
        )
        self.filtering_doc = REPO / "docs" / "filtering.md"
        self.work = work


def rel(p: Path) -> str:
    """Short, checkout-relative source label (code files relative to the code checkout,
    data files relative to ``--root``; a worktree nested in the data checkout is not named)."""
    s = str(p)
    for base in (str(REPO) + "/", str(PATHS.root) + "/"):
        if s.startswith(base):
            return s[len(base) :]
    return s


PAPER_DIR = REPO / "scripts" / "figures" / "paper"


def paper(mod: str):
    """Import an analysis module of the figure generators (``scripts/figures/paper/<mod>.py``).

    Numbers that a figure plots are computed here with the figure's own functions, so text and
    artwork quote one analysis. The modules locate the data through ``TRIZOD_DATA_ROOT``."""
    import importlib

    os.environ["TRIZOD_DATA_ROOT"] = str(PATHS.root)
    if str(PAPER_DIR) not in sys.path:
        sys.path.insert(0, str(PAPER_DIR))
    return importlib.import_module(mod)


def read_fasta(path: Path) -> dict:
    out, key = OrderedDict(), None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            key = line[1:].split()[0]
            out[key] = ""
        elif key is not None:
            out[key] += line
    return out


def arr(v) -> np.ndarray:
    return np.array([np.nan if x is None else x for x in v], dtype=float)


def load_scores(path: Path) -> dict:
    recs = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            recs[r["ID"]] = {
                "g": arr(r["gscores"]),
                "z": arr(r["zscores"]),
                "seq": r["seq"],
                "off": {a: r.get(off_sigma_col(a)) for a in ATOMS},
            }
    return recs


def parse_filter_blocks(log_text: str) -> list[dict]:
    """Parse the four print_filter_losses() reports (one per tier, in tier order)."""
    blocks = []
    starts = [
        m.start() for m in re.finditer(r"Pre-computation filtering results", log_text)
    ]
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(log_text)
        chunk = log_text[s:e]
        end = chunk.find("final dataset entries")
        chunk = chunk[: chunk.find("\n", end) + 1]
        pre, rest = chunk.split("Rejected offsets stats", 1)
        rej, post = rest.split("Post-computation filtering results", 1)
        b = {"pre": OrderedDict(), "rej": OrderedDict(), "post": OrderedDict()}
        row = re.compile(r"^(.+?)\s*:\s+(\d+)\s+(\d+)\s*$", re.M)
        for name, part in (("pre", pre), ("rej", rej), ("post", post)):
            for m in row.finditer(part):
                label = m.group(1).strip()
                b[name][label] = (int(m.group(2)), int(m.group(3)))
        m = re.search(r"total filtered\s*:\s*(\d+) of\s+(\d+) \(\s*([\d.]+) %\)", pre)
        b["pre_total"] = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
        m = re.search(
            r"remaining for scores computation\s*:\s*(\d+) of\s+(\d+) \(\s*([\d.]+) %\)",
            chunk,
        )
        b["remaining"] = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
        m = re.search(r"total filtered\s*:\s*(\d+) of\s+(\d+) \(\s*([\d.]+) %\)", post)
        b["post_total"] = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
        m = re.search(
            r"final dataset entries\s*:\s*(\d+) of\s+(\d+) \(\s*([\d.]+) %\)", chunk
        )
        b["final"] = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
        blocks.append(b)
    return blocks


# ----------------------------------------------------------------------------------
# cached heavy steps
# ----------------------------------------------------------------------------------
def cached_json(name: str, fn, refresh: bool):
    p = PATHS.work / name
    if p.exists() and not refresh:
        return json.loads(p.read_text())
    t = time.time()
    out = fn()
    p.write_text(json.dumps(out))
    print(f"  computed {name} in {time.time() - t:.0f} s", file=sys.stderr)
    return out


_PRESS = re.compile(r"^\s*pressure\s+(\S+)\s+(\S+)\s+(\S+)", re.I)
_SUB = re.compile(r"_Entry\.Submission_date\s+(\S+)")
_ALC = re.compile(r"(?<![A-Za-z0-9])(methanol|ethanol|meoh|etoh)(?![A-Za-z])", re.I)
_ALC_EXCLUDE = re.compile(r"mercapto|fluoro", re.I)
_SC_LOOP = re.compile(
    r"loop_\s*\n((?:\s*_Sample_component\.\S+\s*\n)+)(.*?)\n\s*stop_", re.S
)


def _to_bar(v: float, unit: str):
    u = unit.lower().strip("'\"")
    return {
        "bar": v,
        "bars": v,
        "atm": v * 1.01325,
        "mpa": v * 10,
        "kpa": v / 100,
        "pa": v / 1e5,
        "mbar": v / 1000,
        "psi": v * 0.0689476,
    }.get(u)


def scan_raw() -> dict:
    """One pass over the raw NMR-STAR files: dates, pressure rows, organisms, alcohols."""
    out = {}
    for d in sorted(os.listdir(PATHS.raw)):
        fs = sorted((PATHS.raw / d).glob("*.str"))
        if not fs:
            continue
        txt = fs[0].read_text(errors="ignore")
        eid = d.replace("bmr", "")
        rec = {}
        m = _SUB.search(txt)
        rec["sub"] = m.group(1) if m else None
        press = []
        for line in txt.splitlines():
            m = _PRESS.match(line)
            if not m:
                continue
            val, _err, unit = m.groups()
            try:
                v = float(val)
            except ValueError:
                continue
            press.append([v, unit.lower().strip("'\""), _to_bar(v, unit)])
        rec["press"] = press
        alc = False
        for mm in _SC_LOOP.finditer(txt):
            for line in mm.group(2).splitlines():
                if _ALC.search(line) and not _ALC_EXCLUDE.search(line):
                    alc = True
        rec["alc"] = alc
        org = {}
        i = txt.find("_Entity_natural_src.Organism_name_scientific")
        if i >= 0:
            lp = txt.rfind("loop_", 0, i)
            end = txt.find("stop_", i)
            block = txt[lp:end]
            hdr = re.findall(r"^\s*(_Entity_natural_src\.\S+)", block, re.M)
            body = block.split(hdr[-1], 1)[1]
            try:
                toks = shlex.split(body, posix=True)
            except ValueError:
                toks = []
            n = len(hdr)
            if toks and n and len(toks) % n == 0:
                ci = hdr.index("_Entity_natural_src.Organism_name_scientific")
                ei = (
                    hdr.index("_Entity_natural_src.Entity_ID")
                    if "_Entity_natural_src.Entity_ID" in hdr
                    else None
                )
                for r in range(len(toks) // n):
                    row = toks[r * n : (r + 1) * n]
                    org[row[ei] if ei is not None else "1"] = row[ci]
        rec["org"] = org
        out[eid] = rec
    return out


def have_mmseqs() -> bool:
    return shutil.which("mmseqs") is not None


def write_fasta(path: Path, recs) -> None:
    with open(path, "w") as fh:
        for k, s in recs:
            fh.write(f">{k}\n{s}\n")


def mmseqs_cluster_count(name: str, recs, refresh: bool) -> dict:
    """easy-cluster at 50 % id / 80 % cov with the build's COMMON block; returns counts."""

    def run():
        from trizod.dataset.mmseqs import COMMON

        wd = PATHS.work / f"clu_{name}"
        wd.mkdir(parents=True, exist_ok=True)
        fa = wd / "in.fasta"
        write_fasta(fa, recs)
        cmd = [
            "mmseqs",
            "easy-cluster",
            str(fa),
            str(wd / "res"),
            str(wd / "tmp"),
            "--min-seq-id",
            "0.5",
            "-c",
            "0.8",
            *COMMON,
            "--threads",
            "1",
        ]
        with open(wd / "mmseqs.log", "w") as h:
            subprocess.run(cmd, check=True, stdout=h, stderr=subprocess.STDOUT)
        reps = {
            line.split("\t")[0] for line in open(wd / "res_cluster.tsv") if line.strip()
        }
        ver = subprocess.run(
            ["mmseqs", "version"], capture_output=True, text=True
        ).stdout.strip()
        return {
            "n_seqs": len(recs),
            "n_clusters": len(reps),
            "mmseqs": ver,
            "cmd": " ".join(cmd[1:2] + cmd[5:]),
        }

    return cached_json(f"clusters_{name}.json", run, refresh)


def mmseqs_search(name: str, queries, targets, extra: list, refresh: bool) -> list:
    def run():
        wd = PATHS.work / f"search_{name}"
        wd.mkdir(parents=True, exist_ok=True)
        q, t = wd / "q.fasta", wd / "t.fasta"
        write_fasta(q, queries)
        write_fasta(t, targets)
        fmt = "query,target,fident,qcov,tcov,evalue,alnlen,qlen,tlen"
        cmd = ["mmseqs", "easy-search", str(q), str(t), str(wd / "hits.m8"), str(wd / "tmp"),
               *extra, "--format-output", fmt, "--threads", "8"]  # fmt: skip
        with open(wd / "mmseqs.log", "w") as h:
            subprocess.run(cmd, check=True, stdout=h, stderr=subprocess.STDOUT)
        rows = []
        for line in open(wd / "hits.m8"):
            f = line.rstrip("\n").split("\t")
            rows.append(
                [
                    f[0],
                    f[1],
                    float(f[2]),
                    float(f[3]),
                    float(f[4]),
                    float(f[5]),
                    int(f[6]),
                    int(f[7]),
                    int(f[8]),
                ]
            )
        return rows

    return cached_json(f"search_{name}.json", run, refresh)


# ----------------------------------------------------------------------------------
# context
# ----------------------------------------------------------------------------------
class Ctx:
    pass


def load_context(refresh: bool) -> Ctx:
    C = Ctx()
    t = time.time()
    C.df = pq.read_table(PATHS.pq).to_pandas()
    # pandas >= 3 maps strings to pyarrow-backed columns; element-wise Python access to those
    # (map, iterrows, groupby lambdas) goes through arrow::Array::GetScalar and is ~100x slower.
    for col in C.df.columns:
        if pd.api.types.is_string_dtype(C.df[col]) and C.df[col].dtype != object:
            C.df[col] = C.df[col].astype(object)
    C.pq_meta = {
        k.decode(): v.decode()
        for k, v in (pq.read_schema(PATHS.pq).metadata or {}).items()
    }
    C.n_cols = len(pq.read_schema(PATHS.pq).names)
    C.manifest = json.loads(PATHS.manifest.read_text())
    C.fds = json.loads(PATHS.fds.read_text())
    C.log = PATHS.log.read_text(errors="ignore")
    C.blocks = parse_filter_blocks(C.log)
    assert len(C.blocks) == 4, "expected four print_filter_losses() reports"
    C.scores = {
        t_: load_scores(PATHS.bundle / "scores" / t_ / "scores.json") for t_ in TIERS
    }
    C.tier_ids = {t_: set(C.scores[t_]) for t_ in TIERS}
    C.reps = {
        t_: set(read_fasta(PATHS.bundle / "train" / t_ / f"train_{t_}_best.fasta"))
        for t_ in TIERS
    }
    C.pool = {}
    for t_ in TIERS:
        cb = pd.read_csv(PATHS.bundle / "train" / t_ / "clusters_best.tsv", sep="\t")
        C.pool[t_] = set(cb["member"])
    C.test_tz = read_fasta(PATHS.bundle / "test" / "TriZOD_test_set.fasta")
    C.test_cz = read_fasta(PATHS.bundle / "test" / "CheZOD117_test_set.fasta")
    C.tests = set(C.test_tz) | set(C.test_cz)
    df = C.df
    C.idx = pd.Index(df["id"])
    C.strictest = {}
    for t_ in TIERS:
        for i in C.tier_ids[t_]:
            C.strictest[i] = t_
    df["scored_tier"] = df["id"].map(C.strictest)
    df["srank"] = df["scored_tier"].map(RANK)
    df["meanG"] = [
        np.nanmean(np.asarray(g, float))
        if np.isfinite(np.asarray(g, float)).any()
        else np.nan
        for g in df.gscores
    ]
    # sanity: nesting and parquet == unfiltered
    for a, b in zip(TIERS[1:], TIERS[:-1]):
        assert (
            C.tier_ids[a] <= C.tier_ids[b]
            and C.reps[a] <= C.reps[b]
            and C.pool[a] <= C.pool[b]
        )
    assert C.tier_ids["unfiltered"] == set(df["id"])
    print(f"  loaded parquet + bundle in {time.time() - t:.0f} s", file=sys.stderr)
    C.raw = (
        cached_json("raw_scan.json", scan_raw, refresh) if PATHS.raw.is_dir() else None
    )
    return C


def sel(C, ids) -> pd.Series:
    return C.df["id"].isin(ids)


# ----------------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------------
def sec_snapshot(C):
    df, b = C.df, C.blocks
    src_log = rel(PATHS.log)
    m = re.search(r"Failed loading (\d+) of (\d+) BMRB files", C.log)
    n_fail, n_files = int(m.group(1)), int(m.group(2))
    n_dirs = (
        len([d for d in os.listdir(PATHS.raw) if d.startswith("bmr")])
        if PATHS.raw.is_dir()
        else None
    )
    # builds after 2026-10 skip the entries on withdrawn_entries.tsv before loading
    m_w = re.search(r"skipping (\d+) withdrawn BMRB entries", C.log)
    n_withdrawn = int(m_w.group(1)) if m_w else 0
    assert n_dirs in (None, n_files + n_withdrawn), (n_dirs, n_files, n_withdrawn)
    # a snapshot taken from BMRB's public list holds no withdrawn entry, so the log has no
    # 'skipping' line: both macros are still defined (0 withdrawn)
    R.add(
        "nBMRBWithdrawnSkipped",
        I(n_withdrawn),
        "withdrawn BMRB entries in the snapshot that the build skips before loading",
        f"{src_log} 'skipping N withdrawn BMRB entries' (0 when absent)",
        STABLE,
        "r1",
    )
    R.add(
        "nBMRBDirs",
        I(n_dirs) if n_dirs is not None else I(n_files + n_withdrawn),
        "BMRB entry directories in the raw snapshot (= loaded files + skipped withdrawn entries)",
        "count of data/raw/bmrb_entries",
        STABLE,
        "r1",
    )
    R.add(
        "nBMRBFiles",
        I(n_files),
        f"BMRB NMR-STAR files the {REL} build loads from the snapshot",
        f"{src_log} ('Failed loading ... of N'); count of data/raw/bmrb_entries",
        STABLE,
        "r1",
        "17,388",
    )
    R.add(
        "nFilesFailedLoad",
        I(n_fail),
        "snapshot files that fail to load",
        f"{src_log} 'Failed loading'",
        STABLE,
        "r1",
        "425",
    )
    R.add(
        "pctFilesFailedLoad",
        P(100 * n_fail / n_files, 2),
        "share of snapshot files that fail to load",
        "derived",
        STABLE,
        "r1",
        "2.44",
    )
    n_parsed = C.fds["composition_cache_size"]
    assert n_parsed == n_files - n_fail
    R.add(
        "nEntriesParsed",
        I(n_parsed),
        "BMRB entries parsed (BmrbEntry cached) = files - failed",
        f"{rel(PATHS.fds)} composition_cache_size; {src_log}",
        STABLE,
        "r1",
        "16,963",
    )
    pre_u = b[0]
    R.add(
        "nCandidateRecords",
        I(pre_u["pre_total"][1]),
        "candidate records (entry, shift table, assembly, entity) entering the pre-filter",
        f"{src_log} pre-filter 'of N'",
        STABLE,
        "r1",
        "17,843",
    )
    # Recomputed from the build's own BmrbEntry pickle cache (revision: no longer copied
    # from docs/filtering.md); helper candidate_record_counts() is defined near the end.
    cr = candidate_record_counts(C.refresh)
    if cr and cr["n_records"] == pre_u["pre_total"][1]:
        R.add(
            "nCandidateEntries",
            I(cr["n_entries"]),
            "distinct BMRB entries behind the candidate records",
            f"{rel(Path(cr['cache']))} pickles (the build's --cache-dir): distinct entry IDs "
            f"over BmrbEntry.get_peptide_shifts(), {cr['n_records']} records = log pre-filter "
            f"'of N'; cached in --work-dir candidate_records.json",
            STABLE,
            "r1",
            "15,885",
        )
    else:
        why = (
            "BmrbEntry pickle cache not found"
            if not cr
            else f"cache gives {cr['n_records']} records, log {pre_u['pre_total'][1]}"
        )
        R.unrepro(
            "nCandidateEntries",
            "15{,}885",
            "distinct BMRB entries behind the candidate records",
            why,
            "r1",
            plan_source="docs/filtering.md",
        )
    R.add(
        "nRecordsPassPrefilterUnfiltered",
        I(pre_u["remaining"][0]),
        "candidate records passing the unfiltered pre-filter",
        f"{src_log} 'remaining for scores computation' (unfiltered)",
        STABLE,
        "r1",
        "16,867",
    )
    R.add(
        "pctRecordsPassPrefilterUnfiltered",
        P(pre_u["remaining"][2], 2),
        "share of candidate records passing the unfiltered pre-filter",
        src_log,
        STABLE,
        "r1",
        "94.53",
    )
    R.add(
        "nPostfilterLostUnfiltered",
        I(pre_u["post_total"][0]),
        "records lost after scoring at unfiltered (too few scored positions)",
        f"{src_log} post-filter (unfiltered)",
        STABLE,
        "r1",
        "16",
    )
    n_sc = len(df)
    assert (
        n_sc == C.manifest["tiers"]["unfiltered"]["scored_records"] == pre_u["final"][0]
    )
    R.add(
        "nChainsScored",
        I(n_sc),
        "scored chains in the release (unfiltered tier = Parquet rows)",
        "MANIFEST.json; Parquet rows; rebuild log 'final dataset entries'",
        STABLE,
        "abs",
        "16,851",
    )
    R.add(
        "nChainsScoredApprox",
        I(round(n_sc, -3)),
        "scored chains rounded to the nearest thousand ('about 17,000')",
        "derived from \\nChainsScored",
        STABLE,
        "abs",
        "17,000",
    )
    R.add(
        "pctCandidatesScored",
        P(pre_u["final"][2], 2),
        "scored chains as share of candidate records",
        src_log,
        STABLE,
        "r1",
        "94.44",
    )
    R.add(
        "nEntriesScored",
        I(df.entry_id.nunique()),
        "distinct BMRB entries among the scored chains",
        "Parquet entry_id",
        STABLE,
        "r1",
        "15,101",
    )
    R.add(
        "nSequencesScored",
        I(df.sequence.nunique()),
        "distinct sequences among the scored chains",
        "Parquet sequence",
        STABLE,
        "r1",
        "12,745",
    )
    nb = C.fds["composition_flagged_bound"]
    R.add(
        "nEntriesBoundFlagged",
        I(nb),
        "parsed entries flagged bound or multi-molecule by detect_bound()",
        f"{rel(PATHS.fds)}; {src_log} 'global:'",
        STABLE,
        "r5",
        "4,559",
    )
    R.add(
        "pctEntriesBoundFlagged",
        P(100 * nb / n_parsed),
        "share of parsed entries flagged bound or multi-molecule",
        "derived",
        STABLE,
        "r5",
        "26.9",
    )
    # residues
    res_total = int(sum(len(s) for s in df.sequence))
    res_scored = int(sum(np.asarray(m_, bool).sum() for m_ in df["mask"]))
    R.add(
        "nResiduesScored",
        I(res_scored),
        "scored residues (mask true) over all released chains",
        "Parquet mask",
        STABLE,
        "abs",
        "1,560,162",
    )
    R.add(
        "nResiduesTotal",
        I(res_total),
        "residues over all released chains",
        "Parquet sequence",
        STABLE,
        "r1",
        "1,784,225",
    )
    R.add(
        "pctResiduesUnscored",
        P(100 * (1 - res_scored / res_total)),
        "share of residues of released chains that carry no label",
        "derived (audit: users must not impute them as ordered)",
        STABLE,
        "r6",
        "12.6",
    )
    # submission dates (raw)
    if C.raw:
        subs = {e: r["sub"] for e, r in C.raw.items() if r["sub"] and r["sub"] != "."}
        newest = max(subs.values())
        R.add(
            "dateNewestSubmission",
            S(newest),
            "newest _Entry.Submission_date in the snapshot",
            "raw NMR-STAR scan",
            STABLE,
            "r1",
            "2025-11-06",
        )
        R.add(
            "nEntriesPlaceholderDate",
            I(sum(1 for s in subs.values() if s == "1995-07-31")),
            "snapshot entries carrying the placeholder submission date 1995-07-31",
            "raw NMR-STAR scan (all 17,388 files)",
            STABLE,
            "r1",
            "1,192",
        )
        ent = df.drop_duplicates("entry_id")
        yr = ent.entry_id.map(lambda e: int(subs[e][:4]) if e in subs else np.nan)
        n_old, n_new = int((yr <= 2015).sum()), int((yr >= 2016).sum())
        R.add(
            "nEntriesSubmittedToTwentyFifteen",
            I(n_old),
            "scored entries submitted up to 2015",
            "raw _Entry.Submission_date of scored entries",
            STABLE,
            "r1",
            "9,673",
        )
        R.add(
            "nEntriesSubmittedFromTwentySixteen",
            I(n_new),
            "scored entries submitted from 2016 onward (CheZOD117 has none)",
            "raw _Entry.Submission_date of scored entries",
            STABLE,
            "abs",
            "5,427",
        )
        R.add(
            "nEntriesSubmittedFromTwentySixteenApprox",
            I(round(n_new, -2)),
            "same, rounded to the nearest hundred ('about 5,400')",
            "derived",
            STABLE,
            "r6",
            "5,400",
        )
        C.year = {e: int(s[:4]) for e, s in subs.items()}
    else:
        for n, v, d in [("dateNewestSubmission", "2025-11-06", "newest submission date"), ("nEntriesPlaceholderDate", "1{,}192", "placeholder dates"),
                        ("nEntriesSubmittedToTwentyFifteen", "9{,}673", "entries up to 2015"), ("nEntriesSubmittedFromTwentySixteen", "5{,}427", "entries from 2016"),
                        ("nEntriesSubmittedFromTwentySixteenApprox", "5{,}400", "entries from 2016, rounded")]:  # fmt: skip
            R.unrepro(n, v, d, "raw BMRB directory not available", "r1")
        C.year = {}


def sec_tiers(C):
    df, b, fds = C.df, C.blocks, C.fds["tiers"]
    src_log = rel(PATHS.log)
    plans = {
        "scored": ["16,851", "15,073", "11,155", "4,108"],
        "pct": ["94.44", "84.48", "62.52", "23.02"],
        "entries": ["15,101", "13,604", "10,478", "3,959"],
        "short": ["1,349", "1,005", "426", "28"],
        "bound": ["4,192", "3,825", "2,813", "852"],
        "kept": ["11,310", "10,243", "7,916", "3,228"],
        "uniq": ["9,482", "8,898", "7,219", "3,048"],
        "dup": ["1,828", "1,345", "697", "180"],
        "leak": ["1,381", "1,341", "1,183", "714"],
        "leak2": ["1,222", "1,189", "1,054", "660"],
        "leak1": ["159", "152", "129", "54"],
        "pool": ["8,101", "7,557", "6,036", "2,334"],
        "reps": ["5,803", "5,465", "4,519", "1,895"],
        "red": ["2,298", "2,092", "1,517", "439"],
        "test": ["575", "569", "548", "484"],
        "testcz": ["115", "109", "88", "24"],
        "excl": ["8,175", "6,947", "4,571", "1,290"],
        "ovr": ["479", "463", "407", "175"],
        "ovrpct": ["8.3", "8.5", "9.0", "9.2"],
        "ratio": ["1.40", "1.38", "1.34", "1.23"],
    }
    # stage-1/2 lines from the log
    leak = {}
    for t_ in TIERS:
        m = re.search(
            rf"=== Step A \({t_}\): stage-2.*?stage-1 removed:\s+(\d+) \(unique to stage-1: (\d+)\)\s+stage-2 hits:\s+(\d+) \(unique to stage-2: (\d+)\)\s+total removed:\s+(\d+); kept after removal: (\d+)",
            C.log,
            re.S,
        )
        leak[t_] = tuple(int(x) for x in m.groups())
    ovr = {}
    for t_ in TIERS:
        m = re.search(
            rf"=== {t_} ===\s+clusters: (\d+), overrides \(best != mmseqs pick\): (\d+) \(([\d.]+)%\)",
            C.log,
        )
        ovr[t_] = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
    for j, t_ in enumerate(TIERS):
        T = TCAP[t_]
        st_sc = STABLE if t_ == "unfiltered" else PENDING
        ids = C.tier_ids[t_]
        n = len(ids)
        assert n == C.manifest["tiers"][t_]["scored_records"] == b[j]["final"][0]
        R.add(
            f"nTier{T}",
            I(n),
            f"scored chains in the {t_} tier",
            "MANIFEST.json scored_records; scores/<tier>/scores.json",
            st_sc,
            "abs",
            plans["scored"][j],
        )
        R.add(
            f"pctTier{T}OfCandidates",
            P(b[j]["final"][2], 2),
            f"{t_}-tier chains as share of the candidate records",
            f"{src_log} 'final dataset entries'",
            st_sc,
            "r4",
            plans["pct"][j],
        )
        R.add(
            f"nEntriesTier{T}",
            I(df.loc[sel(C, ids), "entry_id"].nunique()),
            f"distinct entries in the {t_} tier",
            "Parquet + tier IDs",
            st_sc,
            "r4",
            plans["entries"][j],
        )
        f = fds[t_]
        R.add(
            f"nDroppedShort{T}",
            I(f["dropped_short"]),
            f"{t_}: chains < 20 aa dropped before redundancy reduction",
            rel(PATHS.fds),
            st_sc,
            "r5",
            plans["short"][j],
        )
        R.add(
            f"nDroppedBound{T}",
            I(f["dropped_bound"]),
            f"{t_}: bound / multi-molecule chains (>= 20 aa) dropped",
            rel(PATHS.fds),
            st_sc,
            "r5",
            plans["bound"][j],
        )
        R.add(
            f"nKeptRows{T}",
            I(f["kept_rows_after_filters"]),
            f"{t_}: chains kept after length and bound removal",
            rel(PATHS.fds),
            st_sc,
            "r5",
            plans["kept"][j],
        )
        R.add(
            f"nUniqueSeq{T}",
            I(f["unique_sequences_kept"]),
            f"{t_}: unique sequences after exact deduplication",
            rel(PATHS.fds),
            st_sc,
            "r5",
            plans["uniq"][j],
        )
        R.add(
            f"nDuplicatesRemoved{T}",
            I(f["kept_rows_after_filters"] - f["unique_sequences_kept"]),
            f"{t_}: exact-sequence duplicates removed (kept - unique)",
            "derived from FDS",
            st_sc,
            "r5",
            plans["dup"][j],
        )
        s1, s1u, s2, s2u, tot, kept = leak[t_]
        R.add(
            f"nLeakageRemoved{T}",
            I(tot),
            f"{t_}: training sequences removed by test-set leakage control (stage 1 or 2)",
            f"{src_log} Step A ({t_})",
            PENDING,
            "r5",
            plans["leak"][j],
        )
        R.add(
            f"nLeakageStageTwo{T}",
            I(s2),
            f"{t_}: stage-2 direct hits against the 575 test sequences",
            f"{src_log} Step A",
            PENDING,
            "t3",
            plans["leak2"][j],
        )
        R.add(
            f"nLeakageStageOneOnly{T}",
            I(s1u),
            f"{t_}: removed by stage-1 co-clustering only (transitive)",
            f"{src_log} Step A",
            PENDING,
            "t3",
            plans["leak1"][j],
        )
        R.add(
            f"nLeakageStageOne{T}",
            I(s1),
            f"{t_}: removed by stage-1 co-clustering (any)",
            f"{src_log} Step A",
            PENDING,
            "t3",
        )
        assert kept == len(C.pool[t_])
        R.add(
            f"nPool{T}",
            I(kept),
            f"{t_}: training pool after leakage control (input to 50/80 clustering)",
            f"{src_log}; train/{t_}/clusters_best.tsv",
            PENDING,
            "r5",
            plans["pool"][j],
        )
        nrep = len(C.reps[t_])
        assert nrep == C.manifest["tiers"][t_]["training_reps"]
        R.add(
            f"nReps{T}",
            I(nrep),
            f"{t_}: training representatives (50 % id / 80 % cov clusters)",
            "MANIFEST.json training_reps; train_<tier>_best.fasta",
            PENDING,
            "abs",
            plans["reps"][j],
        )
        R.add(
            f"nRedundant{T}",
            I(kept - nrep),
            f"{t_}: redundant pool members (pool - representatives)",
            "derived",
            PENDING,
            "r5",
            plans["red"][j],
        )
        tst = C.tests & ids
        R.add(
            f"nTestInTier{T}",
            I(len(tst)),
            f"{t_}: test chains (TriZOD + CheZOD117) that are scored in this tier",
            "tier IDs x test FASTAs",
            st_sc,
            "r5",
            plans["test"][j],
        )
        R.add(
            f"nTestCheZODInTier{T}",
            I(len(set(C.test_cz) & ids)),
            f"{t_}: CheZOD117 test chains scored in this tier",
            "tier IDs x test FASTA",
            st_sc,
            "r5",
            plans["testcz"][j],
        )
        excl = len(ids - C.pool[t_] - C.tests)
        R.add(
            f"nExcluded{T}",
            I(excl),
            f"{t_}: scored chains in neither pool nor test sets",
            "derived",
            PENDING,
            "r5",
            plans["excl"][j],
        )
        # builds after 2026-10 also drop withdrawn entries and placeholder sequences
        # (> max_unknown_fraction unknown residues) before deduplication
        n_wd, n_ph = f.get("dropped_withdrawn", 0), f.get("dropped_placeholder", 0)
        assert excl == f["dropped_short"] + f["dropped_bound"] + n_wd + n_ph + (
            f["kept_rows_after_filters"] - f["unique_sequences_kept"]
        ) + tot - len(tst), (t_, excl, f, tot, len(tst))
        if "dropped_placeholder" in f:
            R.add(
                f"nDroppedPlaceholder{T}",
                I(n_ph),
                f"{t_}: placeholder sequences (> {C.fds.get('max_unknown_fraction', 0.2):g} "
                "unknown residues) dropped before deduplication",
                rel(PATHS.fds),
                st_sc,
                "r5",
            )
        R.add(
            f"nOverrides{T}",
            I(ovr[t_][1]),
            f"{t_}: clusters whose quality-best member replaces the mmseqs representative",
            f"{src_log} Step 4",
            PENDING,
            "r5",
            plans["ovr"][j],
        )
        R.add(
            f"pctOverrides{T}",
            P(ovr[t_][2]),
            f"{t_}: share of clusters with a quality-best override",
            f"{src_log} Step 4",
            PENDING,
            "r5",
            plans["ovrpct"][j],
        )
        R.add(
            f"ratioPoolReps{T}",
            F(kept / nrep, 2),
            f"{t_}: pool : representatives ratio",
            "derived",
            PENDING,
            "r5",
            plans["ratio"][j],
        )
    for t_, _lost, p_ in [
        ("tolerant", 1778, "1,778"),
        ("moderate", 3918, "3,918"),
        ("strict", 7047, "7,047"),
    ]:
        prev = TIERS[RANK[t_] - 1]
        R.add(
            f"nLostStep{TCAP[t_]}",
            I(len(C.tier_ids[prev]) - len(C.tier_ids[t_])),
            f"chains lost from {prev} to {t_}",
            "derived from tier sizes",
            PENDING,
            "r4",
            p_,
        )
    # split / tier columns
    sp = df.split.value_counts()
    for key, name, p_ in [("train", "nSplitTrain", "5,803"), ("redundant", "nSplitRedundant", "2,298"), ("excluded", "nSplitExcluded", "8,175"),
                          ("test_trizod", "nSplitTestTriZOD", "460"), ("test_chezod117", "nSplitTestCheZOD", "115")]:  # fmt: skip
        R.add(
            name,
            I(sp.get(key, 0)),
            f"Parquet split = {key}",
            "Parquet split column",
            STABLE if key.startswith("test") else PENDING,
            "m11",
            p_,
        )
    tt = df.train_tier.value_counts(dropna=False)
    for t_, p_ in zip(TIERS, ["338", "946", "2,624", "1,895"]):
        R.add(
            f"nTrainTier{TCAP[t_]}",
            I(tt.get(t_, 0)),
            f"chains whose strictest representative tier (train_tier) is {t_}",
            "Parquet train_tier",
            PENDING,
            "r5",
            p_,
        )
    R.add(
        "nTrainTierNull",
        I(df.train_tier.isna().sum()),
        "chains that are no training representative (train_tier null)",
        "Parquet train_tier",
        PENDING,
        "m11",
        "11,048",
    )
    pt = df.pool_tier.value_counts(dropna=False)
    for t_, p_ in zip(TIERS, ["544", "1,521", "3,702", "2,334"]):
        R.add(
            f"nPoolTier{TCAP[t_]}",
            I(pt.get(t_, 0)),
            f"chains whose strictest pool tier (pool_tier) is {t_}",
            "Parquet pool_tier",
            PENDING,
            "m11",
            p_,
        )
    R.add(
        "nPoolTierNull",
        I(df.pool_tier.isna().sum()),
        "chains in no training pool (pool_tier null)",
        "Parquet pool_tier",
        PENDING,
        "m11",
        "8,750",
    )
    # stage-1
    m = re.search(
        r"stage-1: (\d+) test seqs; (\d+) test-touching clusters; (\d+) training IDs co-clustered",
        C.log,
    )
    R.add(
        "nStageOneTestClusters",
        I(m.group(2)),
        "stage-1 30/80 clusters that contain a test sequence",
        f"{src_log} Step A0",
        PENDING,
        "r5",
        "539",
    )
    R.add(
        "nStageOneCoclustered",
        I(m.group(3)),
        "training IDs co-clustered with a test sequence in stage 1",
        f"{src_log} Step A0",
        PENDING,
        "r5",
        "1,261",
    )
    # 2026-10: "leakage gate: OK -- 0 shared IDs & 0 exact-sequence matches vs 575 test ...";
    # later builds: "leakage gate: OK (rule 'either': ...) -- 0 shared IDs, 0 exact-sequence
    # matches, 0 exact-substring containments >= 20 aa (either direction) & 0 training records
    # covering ... vs 568 test sequences"
    m = re.search(
        r"leakage gate: OK .*?(\d+) shared IDs(?:,| &) (\d+) exact-sequence matches.*? vs (\d+) test",
        C.log,
    )
    m_x = re.search(
        r"leakage gate: OK .*?(\d+) exact-substring containments >= (\d+) aa.*?& (\d+) training records covering",
        C.log,
    )
    if m_x:
        R.add(
            "nGateSubstring",
            I(m_x.group(1)),
            f"packaging gate: exact-substring containments >= {m_x.group(2)} aa between training and test (either direction)",
            f"{src_log} Step 5",
            PENDING,
            "r5",
        )
        R.add(
            "nGateCoverageRule",
            I(m_x.group(3)),
            "packaging gate: training records within the leakage rule of a test chain (>= 30 % identity over >= 80 % of either chain)",
            f"{src_log} Step 5",
            PENDING,
            "r5",
        )
    R.add(
        "nGateSharedIDs",
        I(m.group(1)),
        "packaging gate: training/test chains sharing an ID",
        f"{src_log} Step 5",
        PENDING,
        "r5",
        "0",
    )
    R.add(
        "nGateExactSeq",
        I(m.group(2)),
        "packaging gate: exact-sequence matches between training and test",
        f"{src_log} Step 5",
        PENDING,
        "r5",
        "0",
    )
    # exclusion decomposition (chain level, audit definition)
    ex = df[df.split == "excluded"]
    tests_seq = set(df.loc[df.split.str.startswith("test"), "sequence"])
    pool_seq = set(df.loc[df.pool_tier.notna(), "sequence"])
    # builds after 2026-10 drop placeholder sequences (mostly unknown residues) after the
    # length and bound cuts; they get their own reason instead of falling through to "homolog"
    placeholder = set(C.fds.get("placeholder_dropped_ids", []))
    reason = np.where(
        ex.is_bound.fillna(False),
        "bound",
        np.where(
            ex.length < 20,
            "short",
            np.where(
                ex.id.isin(placeholder),
                "placeholder",
                np.where(
                    ex.sequence.isin(tests_seq),
                    "testdup",
                    np.where(ex.sequence.isin(pool_seq), "pooldup", "homolog"),
                ),
            ),
        ),
    )
    rc = Counter(reason)
    if placeholder:
        R.add(
            "nExcludedReasonPlaceholder",
            I(rc["placeholder"]),
            "excluded chains (unfiltered view) whose first exclusion reason is: placeholder sequence "
            f"(> {C.fds.get('max_unknown_fraction', 0.2):g} unknown residues)",
            "Parquet split, is_bound, length, id x final_dataset_summary.json placeholder_dropped_ids",
            PENDING,
            "r5",
        )
    for k_, name, d, p_ in [("bound", "nExcludedReasonBound", "bound / multi-molecule (any length)", "4,488"), ("pooldup", "nExcludedReasonPoolDuplicate", "exact duplicate of a pool sequence", "1,495"),
                            ("short", "nExcludedReasonShort", "shorter than 20 aa (not bound)", "1,050"), ("homolog", "nExcludedReasonTestHomolog", "removed as a test-set homolog", "958"),
                            ("testdup", "nExcludedReasonTestDuplicate", "exact copy of a test sequence", "184")]:  # fmt: skip
        R.add(
            name,
            I(rc[k_]),
            f"excluded chains (unfiltered view) whose first exclusion reason is: {d}",
            "Parquet split, is_bound, length, sequence (audit decomposition, bound > short > test copy > pool copy > homolog)",
            PENDING,
            "r5",
            p_,
        )


def sec_filter_losses(C):
    """Per-criterion filter losses (Table T3) from the four print_filter_losses() blocks."""
    names = {
        "method (sub-)type": "MethodSubtype", "temperature": "Temperature", "ionic strength": "IonicStrength", "pH": "PH",
        "peptide length": "PeptideLength", "bb shift types": "BbShiftTypes", "bb shift positions": "BbShiftPositions",
        "bb shift fraction": "BbShiftFraction", "non-canonical frac": "NoncanonicalFraction", "X fraction": "XFraction",
        "physical state": "PhysicalState", ".*denatur.*": "KwDenatur", ".*unfold.*": "KwUnfold", ".*misfold.*": "KwMisfold",
        ".*bound.*": "KwBound", ".*in complex with.*": "KwInComplexWith", ".*complexed with.*": "KwComplexedWith",
        ".*guanidin.*": "CsGuanidin", ".*GdmCl.*": "CsGdmCl", ".*Gdn-Hcl.*": "CsGdnHcl", ".*urea.*": "CsUrea", ".*TFE.*": "CsTFE",
        ".*trifluoroethanol.*": "CsTrifluoroethanol", ".*trifluoro ethanol.*": "CsTrifluoroEthanolSpaced",
        ".*hexafluoroisopropanol.*": "CsHexafluoroisopropanol", ".*hexafluoro-2-propanol.*": "CsHexafluoroTwoPropanol",
        ".*HFIP.*": "CsHFIP", ".*DMSO.*": "CsDMSO", ".*dimethyl sulfoxide.*": "CsDimethylSulfoxide",
        ".*dimethylsulfoxide.*": "CsDimethylsulfoxide", "paramagnetic": "Paramagnetic",
        # cosolvent tokens added after 2026-10 (audit I-07: alcohols, acetonitrile, chloroform,
        # propanols, alkanes, dioxane)
        ".*methanol.*": "CsMethanol", ".*ethanol.*": "CsEthanol", ".*CD3OH.*": "CsCDThreeOH",
        ".*CD3OD.*": "CsCDThreeOD", ".*MeOH.*": "CsMeOH", ".*EtOH.*": "CsEtOH",
        ".*acetonitrile.*": "CsAcetonitrile", ".*CD3CN.*": "CsCDThreeCN", ".*ACN.*": "CsACN",
        ".*chloroform.*": "CsChloroform", ".*CDCl3.*": "CsCDClThree", ".*CHCl3.*": "CsCHClThree",
        ".*isopropanol.*": "CsIsopropanol", ".*2-propanol.*": "CsTwoPropanol",
        ".*isopropyl alcohol.*": "CsIsopropylAlcohol", ".*n-propanol.*": "CsNPropanol",
        ".*1-propanol.*": "CsOnePropanol", ".*hexane.*": "CsHexane", ".*pentane.*": "CsPentane",
        ".*dioxane.*": "CsDioxane",
        "error in computation": "ErrorInComputation", "rejected due to any offset": "OffsetEntryLevel",
    }  # fmt: skip
    src = rel(PATHS.log) + " print_filter_losses()"
    for j, t_ in enumerate(TIERS):
        T = TCAP[t_]
        st = STABLE if t_ == "unfiltered" else PENDING
        blk = C.blocks[j]
        for part, prefix in (("pre", "Pre"), ("post", "Post")):
            for label, (n, u) in blk[part].items():
                if (
                    label.startswith("total filtered")
                    or label.startswith("criterium")
                    or label.startswith("keyword")
                    or label.startswith("perturbing")
                ):
                    continue
                key = re.sub(r"\s*\[.*$", "", label).strip()
                if key not in names:
                    raise KeyError(f"unknown filter label {label!r}")
                base = f"flt{prefix}{names[key]}{T}"
                thr = re.search(r"\[(.*)\]", label)
                d = (
                    f"{t_}: records removed by the {'pre' if part == 'pre' else 'post'}-scoring criterion '{key}'"
                    + (f" (threshold [{thr.group(1)}])" if thr else "")
                )
                R.add(base, I(n), d + " -- filtered", src, st, "t3")
                R.add(
                    base + "Uniq",
                    I(u),
                    d + " -- uniquely filtered (no other criterion)",
                    src,
                    st,
                    "t3",
                )
        for atom, (n, u) in blk["rej"].items():
            if atom.startswith("backbone"):
                continue
            base = f"fltRejectOffset{atom.capitalize() if len(atom) == 1 else atom.upper()}{T}"
            base = (
                base.replace("CA", "Ca")
                .replace("CB", "Cb")
                .replace("HA", "Ha")
                .replace("HB", "Hb")
            )
            R.add(
                base,
                I(n),
                f"{t_}: records with a rejected {atom} offset (max-offset rule)",
                src,
                st,
                "t3",
            )
            R.add(
                base + "Uniq",
                I(u),
                f"{t_}: records with a rejected {atom} offset only",
                src,
                st,
                "t3",
            )
        R.add(
            f"fltPreTotal{T}",
            I(blk["pre_total"][0]),
            f"{t_}: candidate records removed by the pre-filter",
            src,
            st,
            "r4",
            ["976", "2,562", "5,623", "12,407"][j],
        )
        R.add(
            f"pctPreTotal{T}",
            P(blk["pre_total"][2], 2),
            f"{t_}: share of candidate records removed by the pre-filter",
            src,
            st,
            "t3",
            ["5.47", "14.36", "31.51", "69.53"][j],
        )
        R.add(
            f"fltRemaining{T}",
            I(blk["remaining"][0]),
            f"{t_}: records remaining for scoring",
            src,
            st,
            "t3",
            ["16,867", "15,281", "12,220", "5,436"][j],
        )
        R.add(
            f"fltPostTotal{T}",
            I(blk["post_total"][0]),
            f"{t_}: scored records removed by the post-filter",
            src,
            st,
            "t3",
            ["16", "208", "1,065", "1,328"][j],
        )
        R.add(
            f"pctPostTotal{T}",
            P(blk["post_total"][2], 2),
            f"{t_}: share of scored records removed by the post-filter",
            src,
            st,
            "t3",
            ["0.09", "1.36", "8.72", "24.43"][j],
        )
    # thresholds (Table T1 / threshold strip) from filter_defaults, read in a child process
    code = (
        "import json,math;from trizod.trizod import filter_defaults as f;"
        "c=['temperature-range','ionic-strength-range','pH-range','peptide-length-range','min-backbone-shift-types',"
        "'min-backbone-shift-positions','min-backbone-shift-fraction','max-noncanonical-fraction','max-x-fraction','max-offset',"
        "'physical-state-blacklist','perturbing-cosolvents','keywords-blacklist'];"
        "print(json.dumps({t:{k:(v if not isinstance(v,float) or math.isfinite(v) else 'inf') for k,v in "
        "((k,(list(f.loc[t,k]) if hasattr(f.loc[t,k],'__len__') and not isinstance(f.loc[t,k],str) else f.loc[t,k])) for k in c)} for t in f.index},default=str))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO
    )
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
        raise SystemExit("could not read filter_defaults")
    fd = json.loads(out.stdout)

    def rng(v):
        lo, hi = v
        lo = (
            ""
            if lo in ("-inf", float("-inf"))
            or (isinstance(lo, float) and math.isinf(lo))
            else lo
        )
        return lo, hi

    for t_ in TIERS:
        T = TCAP[t_]
        d = fd[t_]
        st = STABLE
        src_c = "trizod/trizod.py filter_defaults"

        def fmt(x):
            if isinstance(x, str) and x in ("inf", "-inf"):
                return r"\ensuremath{\infty}" if x == "inf" else r"\ensuremath{-\infty}"
            if isinstance(x, (int, float)) and math.isinf(x):
                return r"\ensuremath{\infty}" if x > 0 else r"\ensuremath{-\infty}"
            x = float(x)
            return str(int(x)) if x == int(x) else f"{x:g}"

        tlo, thi = d["temperature-range"]
        R.add(
            f"thrTemperature{T}",
            f"{fmt(tlo)}\\textendash{{}}{fmt(thi)}"
            if not (
                isinstance(tlo, str) or (isinstance(tlo, float) and math.isinf(tlo))
            )
            else "none",
            f"{t_}: accepted temperature range (K)",
            src_c,
            st,
            "par",
        )
        plo, phi = d["pH-range"]
        R.add(
            f"thrPH{T}",
            f"{fmt(plo)}\\textendash{{}}{fmt(phi)}"
            if not (
                isinstance(plo, str) or (isinstance(plo, float) and math.isinf(plo))
            )
            else "none",
            f"{t_}: accepted pH range",
            src_c,
            st,
            "par",
        )
        R.add(
            f"thrIonicStrengthMax{T}",
            fmt(d["ionic-strength-range"][1]),
            f"{t_}: maximum ionic strength (M)",
            src_c,
            st,
            "par",
        )
        R.add(
            f"thrMinLength{T}",
            fmt(d["peptide-length-range"][0]),
            f"{t_}: minimum peptide length (residues) for scoring",
            src_c,
            st,
            "par",
        )
        R.add(
            f"thrMinShiftTypes{T}",
            fmt(d["min-backbone-shift-types"]),
            f"{t_}: minimum number of backbone shift types",
            src_c,
            st,
            "par",
        )
        R.add(
            f"thrMinShiftPositions{T}",
            fmt(d["min-backbone-shift-positions"]),
            f"{t_}: minimum number of residues with backbone shifts",
            src_c,
            st,
            "par",
        )
        R.add(
            f"thrMinShiftFraction{T}",
            fmt(d["min-backbone-shift-fraction"]),
            f"{t_}: minimum fraction of residues with backbone shifts",
            src_c,
            st,
            "par",
        )
        R.add(
            f"thrMaxNoncanonical{T}",
            fmt(d["max-noncanonical-fraction"]),
            f"{t_}: maximum non-canonical residue fraction",
            src_c,
            st,
            "par",
        )
        R.add(
            f"thrMaxXFraction{T}",
            fmt(d["max-x-fraction"]),
            f"{t_}: maximum fraction of X residues",
            src_c,
            st,
            "par",
        )
        R.add(
            f"thrMaxOffset{T}",
            fmt(d["max-offset"]),
            f"{t_}: max-offset (residual POTENCI/AIC offset, in units of the per-atom POTENCI RMSD sigma)",
            src_c,
            st,
            "par",
        )
        R.add(
            f"thrPhysStateDenied{T}",
            I(len(d["physical-state-blacklist"])),
            f"{t_}: number of denied physical-state values",
            src_c,
            st,
            "par",
        )
        R.add(
            f"thrCosolventTokens{T}",
            I(len(d["perturbing-cosolvents"])),
            f"{t_}: number of perturbing-cosolvent tokens",
            src_c,
            st,
            "par",
        )


def sec_test_sets(C):
    df = C.df
    summ = json.loads((PATHS.testset / "build_test_set_summary.json").read_text())
    src = rel(PATHS.testset / "build_test_set_summary.json")
    R.add(
        "nTestTriZOD",
        I(len(C.test_tz)),
        "chains in the pinned TriZOD test set",
        "MANIFEST.json test_sets; test/TriZOD_test_set.fasta",
        STABLE,
        "abs",
        "460",
    )
    R.add(
        "nTestTriZODResolved",
        I(summ["resolved"]),
        "pinned TriZOD test sequences resolved against the tolerant pool",
        src,
        STABLE,
        "r5",
        "460",
    )
    R.add(
        "nTestTriZODDropped",
        I(len(summ["dropped"])),
        "pinned TriZOD test sequences dropped at resolution",
        src,
        STABLE,
        "r5",
        "0",
    )
    R.add(
        "nTestTriZODSubstituted",
        I(len(summ["substitutions"])),
        "pinned TriZOD test sequences substituted at resolution",
        src,
        STABLE,
        "r5",
        "0",
    )
    R.add(
        "nTestTriZODLabelStrict",
        I(summ["label_tier_counts"].get("strict", 0)),
        "TriZOD test chains whose label_tier is strict",
        src + "; Parquet label_tier",
        STABLE,
        "r5",
        "460",
    )
    R.add(
        "nTestCheZOD",
        I(len(C.test_cz)),
        "CheZOD117 proteins mapped to BMRB chains (test_chezod117)",
        "MANIFEST.json; test/CheZOD117_test_set.fasta",
        STABLE,
        "abs",
        "115",
    )
    R.add(
        "nTestCombined",
        I(len(C.tests)),
        "combined held-out test chains (TriZOD + CheZOD117)",
        "test FASTAs",
        STABLE,
        "r5",
        "575",
    )
    R.add(
        "nTestSharedSequences",
        I(len(set(C.test_tz.values()) & set(C.test_cz.values()))),
        "sequences shared by the two test sets",
        "test FASTAs",
        STABLE,
        "r5",
        "0",
    )
    # draw (2026-08-11, seed 42) from testset/_tmp
    tmp = PATHS.draw / "_tmp"
    try:
        allseq = read_fasta(tmp / "strict_plus_chezod.fasta")
        hdr = list(allseq)
        n_in = len(hdr)
        n_cz = sum(1 for h in hdr if h.upper().startswith("CHEZOD"))
        clu = pd.read_csv(
            tmp / "c30_cluster.tsv", sep="\t", header=None, names=["rep", "mem"]
        )
        is_cz = clu.mem.str.upper().str.startswith("CHEZOD")
        g = clu.assign(cz=is_cz).groupby("rep").cz.any()
        n_clu, n_free = len(g), int((~g).sum())
        n_free_mem = int(clu[clu.rep.isin(g.index[~g])].shape[0])
        n_sampled_mem = len(read_fasta(tmp / "sampled_members.fasta"))
        tclu = pd.read_csv(
            PATHS.draw / "TriZOD_test_set_clu.tsv",
            sep="\t",
            header=None,
            names=["rep", "mem"],
        )
        sampled_clusters = clu[
            clu.mem.isin(set(read_fasta(tmp / "sampled_members.fasta")))
        ].rep.nunique()
        src2 = (
            rel(tmp)
            + " (strict_plus_chezod.fasta, c30_cluster.tsv, sampled_members.fasta); testset/TriZOD_test_set_clu.tsv"
        )
        R.add(
            "nDrawInputSeqs",
            I(n_in),
            "test-set draw input: strict unique + CheZOD sequences (2026-08-11)",
            src2,
            STABLE,
            "r5",
            "4,488",
        )
        R.add(
            "nDrawCheZODSeqs",
            I(n_cz),
            "test-set draw input: CheZOD sequences (CheZOD1325 + CheZOD117 + other)",
            src2 + " (header pattern)",
            STABLE,
            "r5",
            "1,440",
        )
        R.add(
            "nDrawStrictSeqs",
            I(n_in - n_cz),
            "test-set draw input: strict unique sequences",
            src2,
            STABLE,
            "r5",
            "3,048",
        )
        R.add(
            "nDrawClusters",
            I(n_clu),
            "test-set draw: clusters at 30 % id / 80 % cov",
            src2,
            STABLE,
            "r5",
            "2,915",
        )
        R.add(
            "nDrawClustersCheZODFree",
            I(n_free),
            "test-set draw: clusters without any CheZOD member",
            src2,
            STABLE,
            "r5",
            "1,646",
        )
        R.add(
            "nDrawClustersCheZODFreeMembers",
            I(n_free_mem),
            "test-set draw: sequences in CheZOD-free clusters",
            src2,
            STABLE,
            "r5",
            "2,190",
        )
        R.add(
            "nDrawClustersSampled",
            I(sampled_clusters),
            "test-set draw: CheZOD-free clusters sampled (25 %, seed 42)",
            src2,
            STABLE,
            "r5",
            "412",
        )
        R.add(
            "nDrawSampledMembers",
            I(n_sampled_mem),
            "test-set draw: member sequences of the sampled clusters",
            src2,
            STABLE,
            "r5",
            "544",
        )
        R.add(
            "nDrawFinalReps",
            I(tclu.rep.nunique()),
            "test-set draw: representatives at 50 % id / 80 % cov (= TriZOD test set)",
            src2,
            STABLE,
            "r5",
            "460",
        )
    except Exception as e:  # pragma: no cover - only when testset/_tmp is gone
        for n, v in [("nDrawInputSeqs", "4{,}488"), ("nDrawCheZODSeqs", "1{,}440"), ("nDrawStrictSeqs", "3{,}048"), ("nDrawClusters", "2{,}915"),
                     ("nDrawClustersCheZODFree", "1{,}646"), ("nDrawClustersCheZODFreeMembers", "2{,}190"), ("nDrawClustersSampled", "412"),
                     ("nDrawSampledMembers", "544"), ("nDrawFinalReps", "460")]:  # fmt: skip
            R.unrepro(
                n, v, "test-set draw count", f"testset/_tmp unreadable ({e})", "r5"
            )
    R.add(
        "pctDrawSampled",
        P(25, 0),
        "share of CheZOD-free clusters sampled for the TriZOD test set",
        "trizod/dataset/testset.py default",
        STABLE,
        "par",
        "25",
    )
    R.add(
        "seedTestDraw",
        "42",
        "random seed of the test-set draw",
        "trizod/dataset/testset.py default; facts sheet Q3",
        STABLE,
        "par",
        "42",
    )
    R.add(
        "dateTestDraw",
        S("2026-08-11"),
        "date of the pinned test-set draw",
        "testset/build_test_set.log mtime / facts sheet Q3",
        STABLE,
        "par",
        "2026-08-11",
    )
    # lengths / residues / MAD per test set
    for name, ids in (("TriZOD", C.test_tz), ("CheZOD", C.test_cz)):
        sub = df[sel(C, ids)]
        L = sub.length.to_numpy()
        pl = {
            "TriZOD": ["20", "110.5", "123.6", "371"],
            "CheZOD": ["46", "99", "119.9", "467"],
        }[name]
        R.add(
            f"lenMinTest{name}",
            I(L.min()),
            f"{name} test set: minimum chain length (aa)",
            "Parquet length",
            STABLE,
            "r5",
            pl[0],
        )
        R.add(
            f"lenMedianTest{name}",
            F(np.median(L), 1) if np.median(L) % 1 else I(np.median(L)),
            f"{name} test set: median chain length (aa)",
            "Parquet length",
            STABLE,
            "r5",
            pl[1],
        )
        R.add(
            f"lenMeanTest{name}",
            F(L.mean(), 1),
            f"{name} test set: mean chain length (aa)",
            "Parquet length",
            STABLE,
            "r5",
            pl[2],
        )
        R.add(
            f"lenMaxTest{name}",
            I(L.max()),
            f"{name} test set: maximum chain length (aa)",
            "Parquet length",
            STABLE,
            "r5",
            pl[3],
        )
        if C.year:
            yrs = sub.entry_id.map(C.year)
            bins = (
                pd.cut(yrs, [1980, 2000, 2005, 2010, 2015, 2020, 2027])
                .value_counts()
                .sort_index()
                .tolist()
            )
            labels = [
                "ToTwoThousand",
                "TwoThousandOneToFive",
                "TwoThousandSixToTen",
                "TwoThousandElevenToFifteen",
                "TwoThousandSixteenToTwenty",
                "TwoThousandTwentyOneToTwentyFive",
            ]
            pv = {
                "TriZOD": ["6", "19", "129", "114", "113", "79"],
                "CheZOD": ["12", "20", "47", "36", "0", "0"],
            }[name]
            for lab, v, p_ in zip(labels, bins, pv):
                R.add(
                    f"nYear{lab}Test{name}",
                    I(v),
                    f"{name} test chains deposited in bin {lab} (submission year)",
                    "raw _Entry.Submission_date",
                    STABLE,
                    "r5",
                    p_,
                )
            R.add(
                f"yearMaxTest{name}",
                str(int(yrs.max())),
                f"{name} test set: latest submission year",
                "raw _Entry.Submission_date",
                STABLE,
                "r5",
                {"TriZOD": "2025", "CheZOD": "2015"}[name],
            )
    R.add(
        "nCheZODTestProteins",
        "117",
        "proteins in the published CheZOD117 test set",
        "Nielsen & Mulder 2016 (literature)",
        STABLE,
        "lit",
        "117",
    )


def residue_stats(recs) -> dict:
    L = S_ = D = D5 = Zlt8 = Zn = 0
    Gs, cf, lens = [], [], []
    for r in recs:
        g, z = r["g"], r["z"]
        m = ~np.isnan(g)
        L += len(g)
        lens.append(len(r["seq"]))
        S_ += int(m.sum())
        gs = g[m]
        D += int((gs > G_DIS).sum())
        D5 += int((gs > 0.5).sum())
        Gs.append(gs)
        zm = ~np.isnan(z)
        Zn += int(zm.sum())
        Zlt8 += int((z[zm] < 8).sum())
        if m.any():
            cf.append(float((gs > G_DIS).mean()))
    G = np.concatenate(Gs)
    lens, cf = np.array(lens), np.array(cf)
    return {
        "chains": len(lens), "residues": L, "scored": S_, "pscored": 100 * S_ / L, "pdis": 100 * D / S_, "pg5": 100 * D5 / S_, "pz8": 100 * Zlt8 / Zn,
        "meanG": G.mean(), "medG": np.median(G), "nodis": int((cf == 0).sum()), "pnodis": 100 * (cf == 0).mean(), "most": int((cf > 0.5).sum()),
        "full": int((cf >= 0.9).sum()), "lmin": lens.min(), "l25": np.percentile(lens, 25), "lmed": np.median(lens), "lmean": lens.mean(),
        "l75": np.percentile(lens, 75), "lmax": lens.max(), "nonscored": len(lens) - len(cf),
    }  # fmt: skip


def sec_residues(C):
    df = C.df
    sets = OrderedDict()
    for t_ in TIERS:
        sets[f"Scored{TCAP[t_]}"] = (
            list(C.scores[t_].values()),
            PENDING if t_ != "unfiltered" else STABLE,
        )
    for t_ in TIERS:
        sets[f"Reps{TCAP[t_]}"] = ([C.scores[t_][i] for i in C.reps[t_]], PENDING)
    for t_ in ("unfiltered", "strict"):
        sets[f"Redundant{TCAP[t_]}"] = (
            [C.scores[t_][i] for i in C.pool[t_] - C.reps[t_]],
            PENDING,
        )
    sets["TestTriZOD"] = ([C.scores["unfiltered"][i] for i in C.test_tz], STABLE)
    sets["TestCheZOD"] = ([C.scores["unfiltered"][i] for i in C.test_cz], STABLE)
    sets["Excluded"] = (
        [C.scores["unfiltered"][i] for i in df.id[df.split == "excluded"]],
        PENDING,
    )
    plan = {
        "ScoredUnfiltered": [
            "16,851",
            "1,784,225",
            "1,560,162",
            "87.4",
            "10.92",
            "13.59",
            "27.26",
            "0.192",
            "0.085",
        ],
        "ScoredTolerant": [
            "15,073",
            "1,633,275",
            "1,453,614",
            "89.0",
            "10.82",
            "13.44",
            "26.33",
            "0.191",
            "0.085",
        ],
        "ScoredModerate": [
            "11,155",
            "1,220,367",
            "1,155,461",
            "94.7",
            "9.88",
            "12.32",
            "19.21",
            "0.181",
            "0.083",
        ],
        "ScoredStrict": [
            "4,108",
            "473,566",
            "453,416",
            "95.8",
            "7.11",
            "9.23",
            "14.45",
            "0.158",
            "0.080",
        ],
        "RepsUnfiltered": [
            "5,803",
            "670,381",
            "607,818",
            "90.7",
            "12.54",
            "15.59",
            "23.88",
            "0.208",
            "0.089",
        ],
        "RepsTolerant": [
            "5,465",
            "638,915",
            "586,778",
            "91.8",
            "12.52",
            "15.54",
            "23.64",
            "0.207",
            "0.088",
        ],
        "RepsModerate": [
            "4,519",
            "527,359",
            "498,542",
            "94.5",
            "11.89",
            "14.68",
            "20.77",
            "0.200",
            "0.086",
        ],
        "RepsStrict": [
            "1,895",
            "228,427",
            "218,620",
            "95.7",
            "8.97",
            "11.25",
            "16.12",
            "0.175",
            "0.083",
        ],
        "RedundantUnfiltered": [
            "2,298",
            "239,580",
            "217,226",
            "90.7",
            "9.68",
            "12.05",
            "27.35",
            "0.181",
            "0.084",
        ],
        "RedundantStrict": [
            "439",
            "51,529",
            "49,446",
            "96.0",
            "4.47",
            "5.92",
            "12.04",
            "0.133",
            "0.077",
        ],
        "TestTriZOD": [
            "460",
            "56,847",
            "54,455",
            "95.8",
            "6.79",
            "8.88",
            "14.41",
            "0.156",
            "0.080",
        ],
        "TestCheZOD": [
            "115",
            "13,794",
            "12,932",
            "93.8",
            "47.80",
            "56.77",
            "66.38",
            "0.521",
            "0.623",
        ],
        "Excluded": [
            "8,175",
            "803,623",
            "667,731",
            "83.1",
            "9.46",
            "11.82",
            "30.60",
            "0.178",
            "0.082",
        ],
    }
    plan_c = {
        "ScoredUnfiltered": [
            "8,497",
            "1,484",
            "526",
            "5",
            "50",
            "97",
            "105.9",
            "139",
            "1,096",
        ],
        "ScoredTolerant": [
            "7,347",
            "1,325",
            "472",
            "5",
            "56",
            "99",
            "108.4",
            "140",
            "1,096",
        ],
        "ScoredModerate": [
            "5,659",
            "823",
            "265",
            "10",
            "64",
            "102",
            "109.4",
            "140",
            "731",
        ],
        "ScoredStrict": [
            "2,147",
            "199",
            "65",
            "15",
            "77",
            "106",
            "115.3",
            "140",
            "490",
        ],
        "RepsUnfiltered": [
            "2,562",
            "552",
            "169",
            "20",
            "68",
            "106",
            "115.5",
            "144",
            "1,096",
        ],
        "RepsStrict": ["928", "117", "43", "20", "86", "111", "120.5", "143", "490"],
        "TestTriZOD": ["239", "24", "9", "20", "79", "110.5", "123.6", "154", "371"],
        "TestCheZOD": ["8", "61", "11", "46", "80", "99", "119.9", "142", "467"],
    }
    src = "scores/<tier>/scores.json labels (unfiltered labels for test/excluded sets) + bundle FASTAs/cluster TSVs"
    C.resstats = {}
    for name, (recs, st) in sets.items():
        s = residue_stats(recs)
        C.resstats[name] = s
        p = plan.get(name, [None] * 9)
        sec = "t7"
        R.add(
            f"nChainsSet{name}",
            I(s["chains"]),
            f"chains in set {name}",
            src,
            st,
            sec,
            p[0],
        )
        R.add(
            f"nResiduesSet{name}",
            I(s["residues"]),
            f"residues in set {name}",
            src,
            st,
            sec,
            p[1],
        )
        R.add(
            f"nResScoredSet{name}",
            I(s["scored"]),
            f"scored residues (non-null G) in set {name}",
            src,
            st,
            sec,
            p[2],
        )
        R.add(
            f"pctResScoredSet{name}",
            P(s["pscored"]),
            f"share of residues scored in set {name}",
            src,
            st,
            sec,
            p[3],
        )
        sec_dis = (
            "r5"
            if name
            in (
                "TestTriZOD",
                "TestCheZOD",
                "RepsUnfiltered",
                "RepsStrict",
                "ScoredUnfiltered",
                "ScoredStrict",
            )
            else sec
        )
        R.add(
            f"pctDisordered{name}",
            P(s["pdis"]),
            f"share of scored residues with G > 0.65 in set {name} (one decimal; checked via the Precise twin)",
            src,
            st,
            sec_dis,
        )
        R.add(
            f"pctDisorderedPrecise{name}",
            P(s["pdis"], 2),
            f"share of scored residues with G > 0.65 in set {name} (two decimals)",
            src,
            st,
            sec,
            p[4],
        )
        R.add(
            f"pctGAboveHalf{name}",
            P(s["pg5"], 2),
            f"share of scored residues with G > 0.5 in set {name}",
            src,
            st,
            sec,
            p[5],
        )
        R.add(
            f"pctZBelowEight{name}",
            P(s["pz8"], 2),
            f"share of scored residues with Z < 8 in set {name}",
            src,
            st,
            sec,
            p[6],
        )
        R.add(
            f"meanG{name}",
            F(s["meanG"], 3),
            f"mean per-residue G in set {name}",
            src,
            st,
            sec,
            p[7],
        )
        R.add(
            f"medianG{name}",
            F(s["medG"], 3),
            f"median per-residue G in set {name}",
            src,
            st,
            sec,
            p[8],
        )
        R.add(
            f"pctResUnscoredSet{name}",
            P(100 - s["pscored"]),
            f"share of residues without a label in set {name}",
            src,
            st,
            sec,
        )
        pc = plan_c.get(name)
        if pc or name in ("ScoredUnfiltered",):
            pc = pc or [None] * 9
            R.add(
                f"nChainsNoDisorder{name}",
                I(s["nodis"]),
                f"chains with no residue at G > 0.65 in set {name}",
                src,
                st,
                sec,
                pc[0],
            )
            R.add(
                f"nChainsMostlyDisordered{name}",
                I(s["most"]),
                f"chains with > 50 % of scored residues at G > 0.65 in set {name}",
                src,
                st,
                "r5" if name.startswith("Test") else sec,
                pc[1],
            )
            R.add(
                f"nChainsFullyDisordered{name}",
                I(s["full"]),
                f"chains with >= 90 % of scored residues at G > 0.65 in set {name}",
                src,
                st,
                sec,
                pc[2],
            )
            R.add(
                f"lenMinSet{name}",
                I(s["lmin"]),
                f"minimum chain length in set {name}",
                src,
                st,
                sec,
                pc[3],
            )
            R.add(
                f"lenQOneSet{name}",
                I(s["l25"]),
                f"25th-percentile chain length in set {name}",
                src,
                st,
                sec,
                pc[4],
            )
            R.add(
                f"lenMedianSet{name}",
                F(s["lmed"], 1) if s["lmed"] % 1 else I(s["lmed"]),
                f"median chain length in set {name}",
                src,
                st,
                sec,
                pc[5],
            )
            R.add(
                f"lenMeanSet{name}",
                F(s["lmean"], 1),
                f"mean chain length in set {name}",
                src,
                st,
                sec,
                pc[6],
            )
            R.add(
                f"lenQThreeSet{name}",
                I(s["l75"]),
                f"75th-percentile chain length in set {name}",
                src,
                st,
                sec,
                pc[7],
            )
            R.add(
                f"lenMaxSet{name}",
                I(s["lmax"]),
                f"maximum chain length in set {name}",
                src,
                st,
                sec,
                pc[8],
            )
    R.add(
        "pctChainsNoDisorderScoredUnfiltered",
        P(C.resstats["ScoredUnfiltered"]["pnodis"]),
        "share of released chains without any residue at G > 0.65",
        src,
        STABLE,
        "t7",
        "50.4",
    )
    assert C.resstats["ScoredUnfiltered"]["nonscored"] == 0
    # G distribution over all released scored residues
    g = np.concatenate(
        [
            np.asarray(x, float)[np.asarray(m_, bool)]
            for x, m_ in zip(df.gscores, df["mask"])
        ]
    )
    R.add(
        "pctResGBelowThirtyFive",
        P(100 * (g < 0.35).mean()),
        "share of released scored residues with G < 0.35",
        "Parquet gscores",
        STABLE,
        "r5",
        "83.6",
    )
    R.add(
        "pctResGIntermediate",
        P(100 * ((g >= 0.35) & (g <= 0.65)).mean()),
        "share of released scored residues with 0.35 <= G <= 0.65",
        "Parquet gscores",
        STABLE,
        "r5",
        "5.5",
    )
    # tolerant residue count under Parquet labels
    tolp = int(
        sum(
            np.asarray(m_, bool).sum()
            for i, m_ in zip(df.id, df["mask"])
            if i in C.tier_ids["tolerant"]
        )
    )
    R.add(
        "nResScoredTolerantParquetLabels",
        I(tolp),
        "scored residues of tolerant chains under the Parquet (unfiltered) labels",
        "Parquet mask x tolerant IDs",
        PENDING,
        "t7",
        "1,453,771",
    )
    R.add(
        "nResUnscoredOnlyAtTolerant",
        I(tolp - C.resstats["ScoredTolerant"]["scored"]),
        "residues scored under unfiltered but not under tolerant labels (issue (c))",
        "derived",
        PENDING,
        "t7",
        "157",
    )
    for t_, p_ in zip(TIERS, ["7,564,166", "7,190,869", "6,094,063", "2,526,062"]):
        R.add(
            f"nBbShiftsUsed{TCAP[t_]}",
            I(df.loc[sel(C, C.tier_ids[t_]), "total_bbshifts"].sum()),
            f"{t_}: backbone shifts used for scoring (sum of total_bbshifts)",
            "Parquet total_bbshifts",
            STABLE if t_ == "unfiltered" else PENDING,
            "r1",
            p_,
        )
    # per-residue k
    ks = np.concatenate(
        [np.asarray(k, float)[np.asarray(m_, bool)] for k, m_ in zip(df.k, df["mask"])]
    )
    R.add(
        "kMedianReleased",
        I(np.median(ks)),
        "median per-residue shift count k over all released scored residues",
        "Parquet k, mask",
        STABLE,
        "r2",
        "15",
    )
    R.add(
        "kQOneReleased",
        I(np.percentile(ks, 25)),
        "25th percentile of k over released scored residues",
        "Parquet k, mask",
        STABLE,
        "r2",
        "11",
    )
    R.add(
        "kQThreeReleased",
        I(np.percentile(ks, 75)),
        "75th percentile of k over released scored residues",
        "Parquet k, mask",
        STABLE,
        "r2",
        "19",
    )
    R.add(
        "kMinReleased",
        I(ks.min()),
        "minimum k of a scored residue",
        "Parquet k, mask",
        STABLE,
        "r2",
        "1",
    )
    R.add(
        "kMaxReleased",
        I(ks.max()),
        "maximum k of a scored residue",
        "Parquet k, mask",
        STABLE,
        "r2",
        "21",
    )
    pool = df[df.pool_tier.notna()]
    pk = np.concatenate(
        [
            np.asarray(k, float)[np.asarray(m_, bool)]
            for k, m_ in zip(pool.k, pool["mask"])
        ]
    )
    R.add(
        "nResiduesPoolUnfiltered",
        I(len(pk)),
        "scored residues of the unfiltered training pool",
        "Parquet pool_tier not null",
        PENDING,
        "r2",
        "825,044",
    )
    R.add(
        "kMedianPool",
        I(np.median(pk)),
        "median k over the unfiltered-pool residues (scope: pool only)",
        "Parquet",
        PENDING,
        "r2",
        "16",
    )
    R.add(
        "kQOnePool",
        I(np.percentile(pk, 25)),
        "25th percentile of k over pool residues",
        "Parquet",
        PENDING,
        "r2",
        "12",
    )
    R.add(
        "kQThreePool",
        I(np.percentile(pk, 75)),
        "75th percentile of k over pool residues",
        "Parquet",
        PENDING,
        "r2",
        "20",
    )
    # KS (audit): test vs training mean G
    from scipy.stats import ks_2samp

    ks_ = ks_2samp(
        df.meanG[df.split == "test_trizod"].dropna(),
        df.meanG[df.split == "train"].dropna(),
    )
    R.add(
        "ksDTestVsTrain",
        F(ks_.statistic, 3),
        "KS statistic D between per-chain mean G of the TriZOD test set and all training representatives",
        "Parquet (audit report section 2)",
        PENDING,
        "r5",
        "0.206",
    )
    e = int(math.floor(math.log10(ks_.pvalue)))
    R.add(
        "ksPTestVsTrain",
        f"\\ensuremath{{{ks_.pvalue / 10**e:.1f}\\times10^{{{e}}}}}",
        "two-sided KS p-value for the same comparison",
        "Parquet",
        PENDING,
        "r5",
        None,
    )


def sec_tolerant_labels(C):
    """Fix (c): tolerant scores.json labels vs the unfiltered (= Parquet) labels."""
    U, T = C.scores["unfiltered"], C.scores["tolerant"]
    diff, changed, cross = [], 0, 0
    for rid, r in T.items():
        gt, gu = r["g"], U[rid]["g"]
        if np.array_equal(gt, gu, equal_nan=True):
            continue
        diff.append(rid)
        both = ~np.isnan(gt) & ~np.isnan(gu)
        d = np.abs(gt[both] - gu[both])
        changed += int((d > 0).sum())
        cross += int(((gt[both] > G_DIS) != (gu[both] > G_DIS)).sum())
    ds = set(diff)
    reps_t = ds & C.reps["tolerant"]
    rep_d, rep_sw = [], 0
    for rid in reps_t:
        gt, gu = T[rid]["g"], U[rid]["g"]
        both = ~np.isnan(gt) & ~np.isnan(gu)
        d = np.abs(gt[both] - gu[both])
        rep_d.append(d)
        rep_sw += int((np.isnan(gt) != np.isnan(gu)).sum())
    rep_d = np.concatenate(rep_d) if rep_d else np.array([0.0])
    src = "scores/tolerant/scores.json vs scores/unfiltered/scores.json (port of facts_tolerant_divergence.py)"
    R.add(
        "nTolerantLabelDiffChains",
        I(len(ds)),
        "tolerant chains whose tolerant labels differ from the released (unfiltered) labels -- fix (c)",
        src,
        PENDING,
        "r4",
        "1,073",
    )
    R.add(
        "nTolerantLabelDiffReps",
        I(len(reps_t)),
        "of them tolerant training representatives",
        src,
        PENDING,
        "r4",
        "478",
    )
    R.add(
        "nTolerantLabelDiffTestCheZOD",
        I(len(ds & set(C.test_cz))),
        "of them CheZOD117 chains",
        src,
        PENDING,
        "r4",
        "16",
    )
    R.add(
        "nTolerantLabelDiffResidues",
        I(changed),
        "residues whose G changes between the two label sets",
        src,
        PENDING,
        "r4",
        "102,463",
    )
    R.add(
        "nTolerantLabelCrossResidues",
        I(cross),
        "of them crossing G = 0.65",
        src,
        PENDING,
        "r4",
        "2,055",
    )
    R.add(
        "meanAbsDGTolerantReps",
        F(rep_d.mean(), 3),
        "mean |dG| over co-scored residues of the differing tolerant representatives",
        src,
        PENDING,
        "r4",
        "0.027",
    )
    R.add(
        "pctTolerantRepResiduesAboveTenth",
        P(100 * (rep_d > 0.1).mean()),
        "share of their residues moving by > 0.1",
        src,
        PENDING,
        "r4",
        "4.2",
    )
    R.add(
        "nTolerantRepResiduesSwitchScored",
        I(rep_sw),
        "their residues switching between scored and unscored",
        src,
        PENDING,
        "r4",
        "74",
    )
    # 18611: methanol concentration from the raw file
    f = PATHS.raw / "bmr18611" / "bmr18611_3.str"
    if f.exists():
        m = re.search(
            r"methanol\s.*?\s(\d+(?:\.\d+)?)\s+\.\s+\.\s+%",
            f.read_text(errors="ignore"),
        )
        if m:
            R.add(
                "pctMethanolEighteenSixEleven",
                P(float(m.group(1)), 0),
                "methanol concentration in the ubiquitin A-state sample of entry 18611",
                "raw bmr18611_3.str _Sample_component",
                STABLE,
                "r2",
                "45",
            )
    # release lineage (Supplementary Table T10)
    rel_dir = PATHS.root / "data" / "release"
    for ver, tag, pv in (("v0.3.0", "VThree", "365"), ("v0.2.0", "VTwo", "344")):
        pqf = rel_dir / f"trizod-dataset-{ver}" / "trizod_dataset.parquet"
        if pqf.exists():
            try:
                old = pq.read_table(pqf, columns=["split"]).to_pandas()
                n = int((old.split == "test_trizod").sum())
                if n:
                    R.add(
                        f"nTestTriZODRelease{tag}",
                        I(n),
                        f"TriZOD test chains in the Zenodo {ver} deposit (superseded)",
                        rel(pqf),
                        STABLE,
                        "m11",
                        pv,
                    )
            except Exception as e:  # pragma: no cover
                print(f"  could not read {pqf}: {e}", file=sys.stderr)


def sec_annotations(C):
    df = C.df
    sets = OrderedDict()
    for t_ in TIERS:
        sets[f"Scored{TCAP[t_]}"] = (
            sel(C, C.tier_ids[t_]),
            STABLE if t_ == "unfiltered" else PENDING,
        )
    for t_ in TIERS:
        sets[f"Reps{TCAP[t_]}"] = (sel(C, C.reps[t_]), PENDING)
    sets["TestTriZOD"] = (sel(C, C.test_tz), STABLE)
    sets["TestCheZOD"] = (sel(C, C.test_cz), STABLE)
    ann = OrderedDict(
        [
            ("Bound", (df.is_bound == True, "is_bound")),  # noqa: E712
            ("Metal", (df.has_metal == True, "has_metal")),  # noqa: E712
            ("Nucleic", (df.has_nucleic == True, "has_nucleic")),  # noqa: E712
            ("Membrane", (df.membrane_mimetic.notna(), "membrane_mimetic not null")),
            ("Cosolvent", (df.cosolvent_evidence == True, "cosolvent_evidence")),  # noqa: E712
            ("MultiCopy", (df.n_copies.fillna(0) > 1, "n_copies > 1")),
            (
                "HomoOligomer",
                (
                    (df.n_copies.fillna(0) > 1) & (df.is_bound == False),  # noqa: E712
                    "n_copies > 1 and not is_bound",
                ),
            ),
        ]
    )
    plan = {
        "Bound": ["4,491", "4,068", "2,927", "860", "0", "0", "0", "0", "0", "3"],
        "Metal": ["1,758", "1,553", "1,267", "474", "0", "0", "0", "0", "0", "1"],
        "Nucleic": ["290", "277", "171", "27", "0", "0", "0", "0", "0", "0"],
        "Membrane": ["766", "686", "448", "101", "260", "241", "180", "46", "12", "1"],
        "Cosolvent": ["822", "273", "150", "23", "217", "82", "51", "4", "5", "2"],
        "MultiCopy": [
            "1,301",
            "1,110",
            "806",
            "260",
            "379",
            "349",
            "293",
            "112",
            "35",
            "6",
        ],
        "HomoOligomer": [
            "858",
            "727",
            "555",
            "215",
            "379",
            "349",
            "293",
            "112",
            "35",
            "6",
        ],
    }
    for a, (mask, d) in ann.items():
        for j, (sname, (sm, st)) in enumerate(sets.items()):
            R.add(
                f"nAnn{a}{sname}",
                I((mask & sm).sum()),
                f"chains with {d} in set {sname}",
                "Parquet annotation columns x tier/rep/test IDs",
                st,
                "ann",
                plan[a][j],
            )
    # homo-oligomer representatives G
    train = df.split == "train"
    selh = (df.n_copies.fillna(0) > 1) & (df.is_bound == False)  # noqa: E712

    def gstats(sub):
        pcm, allg = [], []
        for gg, m_ in zip(sub.gscores, sub["mask"]):
            gg, m_ = np.asarray(gg, float), np.asarray(m_, bool)
            if m_.any():
                pcm.append(np.nanmean(gg[m_]))
                allg.append(gg[m_])
        allg = np.concatenate(allg)
        return (
            len(sub),
            float(np.mean(pcm)),
            float(np.mean(allg)),
            float((allg > G_DIS).mean()),
        )

    n1, a1, r1_, f1 = gstats(df[train & selh])
    n0, a0, r0, f0 = gstats(df[train & ~selh])
    s_ = sel(C, C.reps["strict"])
    _, sa1, _, _ = gstats(df[s_ & selh])
    _, sa0, _, _ = gstats(df[s_ & ~selh])
    src = "Parquet (unfiltered labels), split = train"
    R.add(
        "nHomoOligomerReps",
        I(n1),
        "training representatives annotated as homo-oligomers (n_copies > 1, not bound)",
        src,
        PENDING,
        "r6",
        "379",
    )
    R.add(
        "nHomoOligomerRepsStrict",
        I((s_ & selh).sum()),
        "strict training representatives annotated as homo-oligomers",
        src,
        PENDING,
        "r6",
        "112",
    )
    R.add(
        "meanGHomoOligomerReps",
        F(a1, 3),
        "mean per-chain G of homo-oligomer representatives",
        src,
        PENDING,
        "r6",
        "0.162",
    )
    R.add(
        "meanGOtherReps",
        F(a0, 3),
        "mean per-chain G of the remaining representatives",
        src,
        PENDING,
        "r6",
        "0.219",
    )
    R.add(
        "meanGHomoOligomerRepsStrict",
        F(sa1, 3),
        "same, strict representatives",
        src,
        PENDING,
        "r6",
        "0.130",
    )
    R.add(
        "meanGOtherRepsStrict",
        F(sa0, 3),
        "remaining strict representatives",
        src,
        PENDING,
        "r6",
        "0.181",
    )
    R.add(
        "meanResGHomoOligomerReps",
        F(r1_, 3),
        "per-residue mean G, homo-oligomer representatives",
        src,
        PENDING,
        "r6",
        "0.164",
    )
    R.add(
        "meanResGOtherReps",
        F(r0, 3),
        "per-residue mean G, remaining representatives",
        src,
        PENDING,
        "r6",
        "0.211",
    )
    R.add(
        "pctDisorderedHomoOligomerReps",
        P(100 * f1),
        "share of residues at G > 0.65, homo-oligomer representatives",
        src,
        PENDING,
        "r6",
        "7.5",
    )
    R.add(
        "pctDisorderedOtherReps",
        P(100 * f0),
        "share of residues at G > 0.65, remaining representatives",
        src,
        PENDING,
        "r6",
        "12.9",
    )
    # physical_state
    ps = df.physical_state.fillna("__null__").value_counts()
    pst = df.loc[train, "physical_state"].fillna("__null__").value_counts()
    for key, name, pa, pr in [("native", "Native", "13,350", "4,782"), ("__null__", "Null", "2,881", "872"), ("intrinsically disordered", "IntrinsicallyDisordered", "224", "80"),
                              ("denatured", "Denatured", "135", "13"), ("unfolded", "Unfolded", "101", "17"), ("partially disordered", "PartiallyDisordered", "50", "17"),
                              ("molten globule", "MoltenGlobule", "21", "8")]:  # fmt: skip
        R.add(
            f"nPhysState{name}",
            I(ps.get(key, 0)),
            f"chains with physical_state = {key if key != '__null__' else 'null'}",
            "Parquet physical_state",
            STABLE,
            "ann",
            pa,
        )
        R.add(
            f"nPhysState{name}Reps",
            I(pst.get(key, 0)),
            f"training representatives with physical_state = {key if key != '__null__' else 'null'}",
            "Parquet physical_state, split = train",
            PENDING,
            "ann",
            pr,
        )
    se = df.sample_state_evidence.value_counts()
    R.add(
        "nSampleStateSolution",
        I(se.get("solution", 0)),
        "chains with sample_state_evidence = solution",
        "Parquet",
        STABLE,
        "ann",
        "16,478",
    )
    R.add(
        "nSampleStateSolid",
        I(se.get("solid", 0)),
        "chains with sample_state_evidence = solid",
        "Parquet",
        STABLE,
        "r4",
        "373",
    )
    n_null = int(df.exp_method_subtype.isna().sum())
    R.add(
        "nExpMethodSubtypeNull",
        I(n_null),
        "chains whose _Entry.Experimental_method_subtype is absent",
        "Parquet exp_method_subtype",
        STABLE,
        "r4",
        "3,873",
    )
    R.add(
        "pctExpMethodSubtypeNull",
        P(100 * n_null / len(df)),
        "share of chains with absent method subtype",
        "Parquet",
        STABLE,
        "r4",
        "23.0",
    )


def sec_offsets(C):
    df = C.df
    n = len(df)
    plan = {
        "C": [
            "7,946",
            "47.2",
            "0.28",
            "1.12",
            "3.33",
            "160.73",
            "880",
            "1,796",
            "10.7",
            "1.62",
            "634",
            "268",
        ],
        "CA": [
            "10,560",
            "62.7",
            "0.18",
            "0.84",
            "2.93",
            "300.12",
            "926",
            "1,167",
            "6.9",
            "1.09",
            "256",
            "179",
        ],
        "CB": [
            "10,560",
            "62.7",
            "0.18",
            "0.84",
            "2.93",
            "300.12",
            "926",
            "1,399",
            "8.3",
            "1.25",
            "315",
            "126",
        ],
        "H": [
            "9,991",
            "59.3",
            "0.07",
            "0.18",
            "0.31",
            "1.73",
            "2",
            "1,912",
            "11.3",
            "1.57",
            "672",
            "262",
        ],
        "HA": [
            "7,235",
            "42.9",
            "0.05",
            "0.13",
            "0.25",
            "5.19",
            "1",
            "1,595",
            "9.5",
            "2.22",
            "943",
            "349",
        ],
        "HB": [
            "0",
            "0.0",
            None,
            None,
            None,
            None,
            "0",
            "1,389",
            "8.2",
            "2.63",
            "922",
            "604",
        ],
        "N": [
            "10,449",
            "62.0",
            "0.50",
            "1.26",
            "2.58",
            "300.58",
            "1,861",
            "2,139",
            "12.7",
            "1.73",
            "716",
            "189",
        ],
    }
    src = "Parquet lacs_off_*_ppm, off_*_sigma (unfiltered run; max-offset = inf, nothing rejected)"
    anyl = np.zeros(n, bool)
    anyp = np.zeros(n, bool)
    any2 = np.zeros(n, bool)
    any3 = np.zeros(n, bool)
    any05 = np.zeros(n, bool)
    for a in ATOMS:
        A = a.capitalize() if len(a) == 1 else a[0] + a[1:].lower()
        lac, sig = df[lacs_off_ppm_col(a)], df[off_sigma_col(a)]
        lnz = lac.notna() & (lac != 0)
        snz = sig.notna() & (sig != 0)
        al, asg = lac[lnz].abs(), sig[snz].abs()
        p = plan[a]
        R.add(
            f"nLacsNonzero{A}",
            I(lnz.sum()),
            f"chains with a non-zero LACS offset on {a}",
            src,
            STABLE,
            "off",
            p[0],
        )
        R.add(
            f"pctLacsNonzero{A}",
            P(100 * lnz.mean()),
            f"share of released chains with a non-zero LACS offset on {a}",
            src,
            STABLE,
            "off",
            p[1],
        )
        if len(al):
            R.add(
                f"lacsAbsMedian{A}",
                F(al.median(), 2),
                f"median |LACS offset| on {a} among non-zero (ppm)",
                src,
                STABLE,
                "off",
                p[2],
            )
            R.add(
                f"lacsAbsPNinety{A}",
                F(al.quantile(0.9), 2),
                f"90th percentile |LACS offset| on {a} (ppm)",
                src,
                STABLE,
                "off",
                p[3],
            )
            R.add(
                f"lacsAbsPNinetyNine{A}",
                F(al.quantile(0.99), 2),
                f"99th percentile |LACS offset| on {a} (ppm)",
                src,
                STABLE,
                "off",
                p[4],
            )
            R.add(
                f"lacsAbsMax{A}",
                F(al.max(), 2),
                f"maximum |LACS offset| on {a} (ppm)",
                src,
                STABLE,
                "off",
                p[5],
            )
        R.add(
            f"nLacsAboveOnePpm{A}",
            I((al > 1).sum()),
            f"chains with |LACS offset| > 1 ppm on {a}",
            src,
            STABLE,
            "off",
            p[6],
        )
        R.add(
            f"nPotenciNonzero{A}",
            I(snz.sum()),
            f"chains with a non-zero residual POTENCI/AIC offset on {a}",
            src,
            STABLE,
            "off",
            p[7],
        )
        R.add(
            f"pctPotenciNonzero{A}",
            P(100 * snz.mean()),
            f"share of chains with a non-zero residual POTENCI/AIC offset on {a}",
            src,
            STABLE,
            "off",
            p[8],
        )
        R.add(
            f"potenciAbsMedianSigma{A}",
            F(asg.median(), 2),
            f"median |residual offset| on {a} among non-zero (sigma units)",
            src,
            STABLE,
            "off",
            p[9],
        )
        R.add(
            f"nPotenciAboveTwoSigma{A}",
            I((sig.abs() > 2).sum()),
            f"chains with |residual offset| > 2 sigma on {a}",
            src,
            STABLE,
            "off",
            p[10],
        )
        R.add(
            f"nPotenciAboveThreeSigma{A}",
            I((sig.abs() > 3).sum()),
            f"chains with |residual offset| > 3 sigma on {a}",
            src,
            STABLE,
            "off",
            p[11],
        )
        anyl |= lnz.values
        anyp |= snz.values
        any2 |= (sig.abs() > 2).fillna(False).values
        any3 |= (sig.abs() > 3).fillna(False).values
        any05 |= (lac.abs() >= 0.5).fillna(False).values
    R.add(
        "nLacsNonzeroAny",
        I(anyl.sum()),
        "chains with a non-zero LACS offset on any backbone atom",
        src,
        STABLE,
        "r2",
        "10,743",
    )
    R.add(
        "pctLacsNonzeroAny",
        P(100 * anyl.mean()),
        "share of released chains with a non-zero LACS offset",
        src,
        STABLE,
        "r2",
        "63.8",
    )
    R.add(
        "nPotenciNonzeroAny",
        I(anyp.sum()),
        "chains with a non-zero residual POTENCI/AIC offset on any atom",
        src,
        STABLE,
        "r2",
        "4,855",
    )
    R.add(
        "pctPotenciNonzeroAny",
        P(100 * anyp.mean()),
        "share of chains with a non-zero residual POTENCI/AIC offset",
        src,
        STABLE,
        "r2",
        "28.8",
    )
    R.add(
        "nOffsetAboveTwoSigmaAny",
        I(any2.sum()),
        "chains with any |residual offset| > 2 sigma",
        src,
        STABLE,
        "r2",
        "3,058",
    )
    R.add(
        "nOffsetAboveThreeSigmaAny",
        I(any3.sum()),
        "chains with any |residual offset| > 3 sigma",
        src,
        STABLE,
        "r2",
        "1,433",
    )
    R.add(
        "nLacsAtLeastHalfPpmAny",
        I(any05.sum()),
        "chains with |LACS offset| >= 0.5 ppm on any atom",
        src,
        STABLE,
        "r2",
        "7,004",
    )
    R.add(
        "pctLacsAtLeastHalfPpmAny",
        P(100 * any05.mean()),
        "share of chains with |LACS offset| >= 0.5 ppm on any atom",
        src,
        STABLE,
        "r2",
        "41.6",
    )
    R.add(
        "nLacsAtLeastHalfPpmN",
        I((df[lacs_off_ppm_col("N")].abs() >= 0.5).sum()),
        "chains with |LACS N offset| >= 0.5 ppm",
        src,
        STABLE,
        "r2",
        "5,273",
    )
    # per nucleus, as in the Fig. 2b side panel (LACS fits ONE 13C offset for CA and CB)
    assert (
        (df[lacs_off_ppm_col("CA")].abs() >= 0.5)
        == (df[lacs_off_ppm_col("CB")].abs() >= 0.5)
    ).all()
    for col, tag, d_, pc in (
        (lacs_off_ppm_col("CA"), "CaCb", "CA/CB (one shared 13C offset)", "12.3"),
        (lacs_off_ppm_col("C"), "C", "carbonyl C", "13.4"),
        (lacs_off_ppm_col("N"), "N", "15N", "31.3"),
    ):
        hit = df[col].abs() >= 0.5
        if tag != "N":
            R.add(
                f"nLacsAtLeastHalfPpm{tag}",
                I(hit.sum()),
                f"chains with |LACS {d_} offset| >= 0.5 ppm",
                src + "; Fig. 2b side panel (scripts/figures/paper/fig2b_panav.py)",
                STABLE,
                "r2",
            )
        R.add(
            f"pctLacsAtLeastHalfPpm{tag}",
            P(100 * hit.mean()),
            f"share of all scored chains with |LACS {d_} offset| >= 0.5 ppm",
            src + "; Fig. 2b side panel (scripts/figures/paper/fig2b_panav.py)",
            STABLE,
            "r2",
            pc,
        )
    for t_, pl, pp in zip(
        TIERS, ["63.8", "67.3", "79.3", "93.0"], ["28.8", "28.8", "25.5", "18.3"]
    ):
        ids = sel(C, C.tier_ids[t_]).values
        st = STABLE if t_ == "unfiltered" else PENDING
        R.add(
            f"pctLacsNonzeroAnyTier{TCAP[t_]}",
            P(100 * (anyl & ids).sum() / ids.sum()),
            f"share of {t_}-tier chains with a non-zero LACS offset",
            src,
            st,
            "off",
            pl,
        )
        R.add(
            f"pctPotenciNonzeroAnyTier{TCAP[t_]}",
            P(100 * (anyp & ids).sum() / ids.sum()),
            f"share of {t_}-tier chains with a non-zero residual offset",
            src,
            st,
            "off",
            pp,
        )
    # HB-only rejections (audit): HB the only atom above 3 sigma
    others = np.zeros(n, bool)
    for a in ATOMS:
        if a != "HB":
            others |= (df[off_sigma_col(a)].abs() > 3).fillna(False).values
    hbonly = (df[off_sigma_col("HB")].abs() > 3).fillna(False).values & ~others
    R.add(
        "nOffsetThreeSigmaHBOnly",
        I(hbonly.sum()),
        "chains whose only atom above 3 sigma is HB (never LACS-corrected)",
        src + " (audit report)",
        STABLE,
        "r6",
        "422",
    )
    # LACS CA == CB
    dd = (df[lacs_off_ppm_col("CA")] - df[lacs_off_ppm_col("CB")]).abs()
    R.add(
        "nLacsCaCbDiffer",
        I((dd > 1e-6).sum()),
        "chains whose LACS CA and CB offsets differ (otherwise one shared 13C offset)",
        src,
        STABLE,
        "off",
    )
    # extremes
    big = df[df[lacs_off_ppm_col("CA")].abs() > 5]
    R.add(
        "nChainsLacsCAAboveFive",
        I(len(big)),
        "chains with |LACS CA offset| > 5 ppm",
        src,
        STABLE,
        "r6",
        "13",
    )
    top = df.loc[df[lacs_off_ppm_col("CA")].abs().idxmax()]
    R.add(
        "idLacsMaxChain",
        S(top.id),
        "chain with the largest |LACS CA| offset",
        src,
        STABLE,
        "r6",
        "25442_1_2_2",
    )
    R.add(
        "lacsMaxChainCA",
        F(top[lacs_off_ppm_col("CA")], 2, sign=True),
        "its LACS CA offset (ppm)",
        src,
        STABLE,
        "r6",
        "+300.12",
    )
    R.add(
        "lacsMaxChainN",
        F(top[lacs_off_ppm_col("N")], 2, sign=True),
        "its LACS N offset (ppm)",
        src,
        STABLE,
        "r6",
        "+300.58",
    )
    R.add("splitLacsMaxChain", S(top.split), "its split", src, STABLE, "r6", "excluded")
    # training reps with |LACS| > 5 ppm
    tr = df[df.split == "train"].copy()
    c13 = [lacs_off_ppm_col("C"), lacs_off_ppm_col("CA"), lacs_off_ppm_col("CB")]
    big13 = tr[(tr[c13].abs() > 5).any(axis=1)]
    bigN = tr[(tr[lacs_off_ppm_col("N")].abs() > 5) & ~(tr[c13].abs() > 5).any(axis=1)]
    R.add(
        "nRepsLacsAboveFiveCarbon",
        I(len(big13)),
        "training representatives with |LACS| > 5 ppm on a 13C nucleus (C, CA or CB)",
        "Parquet split = train",
        PENDING,
        "r6",
        "9",
    )
    R.add(
        "nRepsLacsAboveFiveNitrogenOnly",
        I(len(bigN)),
        "training representatives with |LACS| > 5 ppm on 15N only",
        "Parquet split = train",
        PENDING,
        "r6",
        "3",
    )
    R.add(
        "nRepsLacsAboveFive",
        I(len(big13) + len(bigN)),
        "training representatives with |LACS| > 5 ppm on 13C or 15N",
        "Parquet split = train",
        PENDING,
        "r6",
        "12",
    )
    C.extreme_reps = pd.concat([big13, bigN])[
        ["id", "train_tier"] + c13 + [lacs_off_ppm_col("N"), lacs_off_ppm_col("HA")]
    ]
    named = [
        (
            "SeventeenOneOhNine",
            "17109_1_1_1",
            lacs_off_ppm_col("C"),
            "strict",
            "-160.73",
        ),
        (
            "FiveOneSevenNine",
            "5179_1_1_1",
            lacs_off_ppm_col("CA"),
            "moderate",
            "+24.46",
        ),
        (
            "EighteenNineFiftyOne",
            "18951_1_1_1",
            lacs_off_ppm_col("CA"),
            "tolerant",
            "-23.83",
        ),
    ]
    for tag, cid, col, tier, pv in named:
        r = df.loc[df.id == cid].iloc[0]
        R.add(
            f"lacsExtremeRep{tag}",
            F(r[col], 2, sign=True),
            f"{cid} ({r.train_tier} representative): LACS {col.split('_')[2]} offset (ppm)",
            "Parquet",
            PENDING,
            "r6",
            pv,
        )
        assert r.train_tier == tier
    # max-offset thresholds in ppm (audit)
    from trizod.constants import REFINED_WEIGHTS as W

    for a in ("H", "HA", "HB"):
        A = a.capitalize() if len(a) == 1 else a[0] + a[1:].lower()
        R.add(
            f"ppmTwoSigma{A}",
            F(2 * W[a], 3),
            f"strict max-offset (2 sigma) expressed in ppm for {a}",
            "trizod/constants.py REFINED_WEIGHTS x 2",
            STABLE,
            "r6",
            {"H": "0.134", "HA": "0.053", "HB": "0.043"}[a],
        )


def sec_panav(C):
    """PANAV confirmation, matched to scored chains by (entry_id, st_id) exactly as Fig. 2b.

    Replaces the earlier entry-level preview (entries with one PANAV list, first scored chain),
    which differed by at most 0.012 in r. Uses the figure's own matching code."""
    f2b = paper("fig2b_panav")
    pan = json.loads(PATHS.panav.read_text())
    stmap = f2b.loop_to_st_map(pan, 6)
    dfp = f2b.sp.load_main(
        ["id", "entry_id", "st_id", "total_bbshifts"]
        + [total_off_ppm_col(a) for a in f2b.ATOMS]
        + [lacs_off_ppm_col(a) for a in ATOMS]
    )
    dfp["entry_id"] = dfp.entry_id.astype(str)
    dfp["st_id"] = dfp.st_id.astype(str)
    M, st = f2b.match_st(pan, stmap, dfp)
    src = (
        rel(PATHS.panav)
        + " x Parquet total_off_*_ppm, matched by (entry_id, st_id) with fig2b_panav.match_st "
        "(loop index -> _Assigned_chem_shift_list.ID cached in tmp/paper_figs/panav_loop_to_st_id.json)"
    )
    R.add(
        "nPanavEntriesAll",
        I(sum(1 for v in pan.values() if v)),
        "snapshot entries for which PANAV returned offsets",
        rel(PATHS.panav),
        STABLE,
        "r2",
    )
    R.add(
        "nPanavListsAll",
        I(st["lists"]),
        "PANAV shift lists (one PANAV correction per assigned chemical-shift list)",
        rel(PATHS.panav),
        STABLE,
        "r2",
        "7,674",
    )
    R.add(
        "nPanavListsMatched",
        I(len(M)),
        "PANAV shift lists matched to a scored chain by (entry_id, st_id)",
        src,
        STABLE,
        "r2",
        "7,620",
    )
    R.add(
        "nPanavEntriesMatched",
        I(M.entry_id.nunique()),
        "entries contributing at least one matched PANAV shift list",
        src,
        STABLE,
        "r2",
        "7,370",
    )
    for key, name, d_ in (
        ("unscored", "NotScored", "PANAV shift lists without a scored chain"),
        ("mismatch", "IdMismatch", "PANAV shift lists skipped because loop and saveframe IDs disagree"),
        ("unmapped", "Unmapped", "PANAV shift lists whose loop index could not be mapped to a st_id"),
        ("multi", "MultiChain", "matched PANAV shift lists holding more than one scored chain (represented by the chain with most backbone shifts)"),
    ):  # fmt: skip
        R.add(f"nPanavLists{name}", I(st[key]), d_, src, STABLE, "r2")
    R.add(
        "panavWindowCarbonPpm",
        F(f2b.LIMS["CA"], 0),
        "half-width (ppm) of the Fig. 2b plotting window for 13C (CA, CB, C)",
        "scripts/figures/paper/fig2b_panav.py LIMS",
        STABLE,
        "par",
        "4",
    )
    R.add(
        "panavWindowNitrogenPpm",
        F(f2b.LIMS["N"], 0),
        "half-width (ppm) of the Fig. 2b plotting window for 15N",
        "scripts/figures/paper/fig2b_panav.py LIMS",
        STABLE,
        "par",
        "6",
    )
    plan_r = {"CA": "0.80", "CB": "0.68", "C": "0.38", "N": "0.49"}
    plan_w = {"CA": "0.87", "CB": "0.87", "C": "0.90", "N": "0.73"}
    from scipy.stats import spearmanr

    for a in f2b.ATOMS:
        A = a.capitalize() if len(a) == 1 else a[0] + a[1:].lower()
        x = M[f"neg_panav_{a}"].to_numpy(float)
        y = M[f"trizod_total_{a}"].to_numpy(float)
        yl = M[f"trizod_lacs_{a}"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        L = f2b.LIMS[a]
        win = m & (np.abs(x) <= L) & (np.abs(y) <= L)
        R.add(
            f"rPanav{A}",
            F(f2b.pearson(x, y), 2),
            f"Pearson r, TriZOD total {a} offset vs negated PANAV correction, all matched lists (no outlier removal)",
            src,
            STABLE,
            "r2",
            plan_r[a],
        )
        R.add(
            f"nPanavPairs{A}",
            I(m.sum()),
            f"matched lists with both a TriZOD and a PANAV {a} offset",
            src,
            STABLE,
            "r2",
        )
        R.add(
            f"rPanavWindow{A}",
            F(f2b.pearson(x[win], y[win]), 2),
            f"same, lists inside the Fig. 2b window (|offset| <= {L:g} ppm on both axes)",
            src,
            STABLE,
            "r2",
            plan_w[a],
        )
        R.add(
            f"nPanavOutsideWindow{A}",
            I((m & ~win).sum()),
            f"matched lists outside the Fig. 2b {a} window (gross deposition errors among them)",
            src,
            STABLE,
            "r2",
        )
        R.add(
            f"rhoPanav{A}",
            F(spearmanr(x[m], y[m]).statistic, 2),
            f"Spearman rho, TriZOD total {a} offset vs negated PANAV correction, all matched lists",
            src,
            STABLE,
            "off",
        )
        R.add(
            f"rPanavLacsOnly{A}",
            F(f2b.pearson(x, yl), 2),
            f"Pearson r, LACS {a} offset only vs negated PANAV correction, all matched lists",
            src.replace("total_off_*_ppm", "lacs_off_*_ppm"),
            STABLE,
            "off",
        )


def sec_shifts(C):
    df = C.df
    cols = [
        "id",
        "seq_id",
        "comp_id",
        "atom_id",
        "atom_type",
        "is_backbone",
        "val_corrected_ppm",
        "offset_source",
        "is_far_outlier",
        "val_err_ppm",
    ]
    t = time.time()
    sh = pq.read_table(PATHS.shifts, columns=cols).to_pandas(
        strings_to_categorical=True
    )
    meta = {
        k.decode(): v.decode()
        for k, v in (pq.read_schema(PATHS.shifts).metadata or {}).items()
    }
    print(f"  loaded shifts in {time.time() - t:.0f} s", file=sys.stderr)
    src = "trizod_shifts.parquet"
    N = len(sh)
    bb = sh.is_backbone.to_numpy(dtype=bool)
    at_cat = sh.atom_type

    def is_el(el):
        return (at_cat == el).to_numpy(dtype=bool)

    R.add(
        "nShifts",
        I(N),
        "assigned chemical shifts on canonical residues of all released chains",
        src,
        STABLE,
        "abs",
        "11,839,037",
    )
    R.add(
        "nShiftColumns",
        I(len(pq.read_schema(PATHS.shifts).names)),
        "columns of the shift companion table",
        src,
        STABLE,
        "m11",
        "14",
    )
    R.add(
        "nParquetColumns",
        I(C.n_cols),
        "columns of the main Parquet (one row per chain)",
        "trizod_dataset.parquet",
        STABLE,
        "m11",
        "70",
    )
    R.add(
        "nShiftsBackbone",
        I(bb.sum()),
        "backbone shifts (the 12 atom IDs the scorer consumes)",
        src,
        STABLE,
        "m11",
        "8,380,186",
    )
    for el, p_ in (("C", "3,263,875"), ("H", "3,789,493"), ("N", "1,326,818")):
        R.add(
            f"nShiftsBackbone{el}",
            I((bb & is_el(el)).sum()),
            f"backbone shifts on {el} nuclei",
            src,
            STABLE,
            "m11",
            p_,
        )
    R.add(
        "nShiftsSidechain",
        I((~bb).sum()),
        "side-chain shifts",
        src,
        STABLE,
        "m11",
        "3,458,851",
    )
    R.add(
        "pctShiftsSidechain",
        P(100 * (~bb).mean()),
        "share of shifts on side chains",
        src,
        STABLE,
        "m11",
        "29.2",
    )
    for el, p_ in (("H", "2,449,316"), ("C", "946,575"), ("N", "62,960")):
        R.add(
            f"nShiftsSidechain{el}",
            I((~bb & is_el(el)).sum()),
            f"side-chain shifts on {el} nuclei",
            src,
            STABLE,
            "m11",
            p_,
        )
    bbc = sh.loc[sh.is_backbone, "atom_id"].astype(str).value_counts()
    for a, p_ in [("H", "1,385,299"), ("N", "1,326,818"), ("CA", "1,263,924"), ("CB", "1,088,181"), ("C", "911,770"), ("HA", "838,150"),
                  ("HBtwo", "626,468"), ("HBthree", "600,030"), ("HB", "145,332"), ("HAtwo", "66,022"), ("HBone", "65,613"), ("HAthree", "62,579")]:  # fmt: skip
        aid = a.replace("two", "2").replace("three", "3").replace("one", "1")
        name = "nShiftsAtom" + a.replace("CA", "Ca").replace("CB", "Cb").replace(
            "HA", "Ha"
        ).replace("HB", "Hb")
        R.add(
            name,
            I(bbc.get(aid, 0)),
            f"backbone shifts with atom_id {aid}",
            src,
            STABLE,
            "m11",
            p_,
        )
    sc_chains = sh.loc[~sh.is_backbone, "id"].astype(str).unique()
    scs = set(sc_chains)
    R.add(
        "nChainsWithSidechainShifts",
        I(len(scs)),
        "released chains carrying side-chain shifts",
        src,
        STABLE,
        "m11",
        "12,643",
    )
    R.add(
        "pctChainsWithSidechainShifts",
        P(100 * len(scs) / len(df)),
        "share of released chains carrying side-chain shifts",
        src,
        STABLE,
        "m11",
        "75.0",
    )
    for t_, p_ in zip(TIERS[1:], ["11,352", "8,741", "3,300"]):
        R.add(
            f"nChainsWithSidechainShifts{TCAP[t_]}",
            I(len(scs & C.tier_ids[t_])),
            f"{t_}-tier chains carrying side-chain shifts",
            src,
            PENDING,
            "m11",
            p_,
        )
    R.add(
        "pctChainsWithSidechainShiftsStrict",
        P(100 * len(scs & C.tier_ids["strict"]) / len(C.tier_ids["strict"])),
        "share of strict-tier chains carrying side-chain shifts",
        src,
        PENDING,
        "m11",
        "80.3",
    )
    ids = sh["id"]
    for name, tids, pl in (
        ("TriZOD", C.test_tz, ["477,084", "331,111", "145,973"]),
        ("CheZOD", C.test_cz, ["93,132", None, None]),
    ):
        m_ = ids.isin(set(tids)).to_numpy(dtype=bool)
        R.add(
            f"nShiftsTest{name}",
            I(m_.sum()),
            f"shifts on {name} test chains",
            src,
            STABLE,
            "m11",
            pl[0],
        )
        R.add(
            f"nShiftsBackboneTest{name}",
            I((m_ & bb).sum()),
            f"backbone shifts on {name} test chains",
            src,
            STABLE,
            "m11",
            pl[1],
        )
        R.add(
            f"nShiftsSidechainTest{name}",
            I((m_ & ~bb).sum()),
            f"side-chain shifts on {name} test chains",
            src,
            STABLE,
            "m11",
            pl[2],
        )
    corr = sh.val_corrected_ppm.notna().to_numpy(dtype=bool)
    R.add(
        "nShiftsReReferenced",
        I(corr.sum()),
        "shifts carrying a re-referenced value (val_corrected_ppm)",
        src,
        STABLE,
        "m11",
        "9,325,834",
    )
    R.add(
        "pctShiftsReReferenced",
        P(100 * corr.mean()),
        "share of shifts with a re-referenced value",
        src,
        STABLE,
        "m11",
        "78.8",
    )
    R.add(
        "nShiftsReReferencedSidechainCarbon",
        I((corr & ~bb & is_el("C")).sum()),
        "side-chain 13C shifts with a re-referenced value",
        src,
        STABLE,
        "m11",
        "945,648",
    )
    R.add(
        "nShiftsRawOnly",
        I((~corr).sum()),
        "shifts released raw only (no corrected value)",
        src,
        STABLE,
        "m11",
        "2,513,203",
    )
    R.add(
        "pctShiftsRawOnly",
        P(100 * (~corr).mean()),
        "share of shifts released raw only",
        src,
        STABLE,
        "m11",
        "21.2",
    )
    R.add(
        "nShiftsRawOnlySidechainCarbon",
        I((~corr & ~bb & is_el("C")).sum()),
        "side-chain 13C shifts left raw (chains with |LACS CA| > 5 ppm)",
        src,
        STABLE,
        "m11",
        "927",
    )
    osrc = sh.offset_source
    for key, name, p_ in [
        ("lacs_only", "LacsOnly", "5,065,748"),
        ("lacs+potenci", "LacsPotenci", "924,373"),
        ("potenci_only", "PotenciOnly", "306,681"),
        ("none", "None", "2,083,384"),
    ]:
        R.add(
            f"nShiftsBackboneSource{name}",
            I((bb & (osrc == key).to_numpy(dtype=bool)).sum()),
            f"backbone shifts with offset_source = {key}",
            src,
            STABLE,
            "m11",
            p_,
        )
    nz = osrc.isin(["lacs+potenci", "lacs_only", "potenci_only"]).to_numpy(dtype=bool)
    R.add(
        "nShiftsNonzeroOffset",
        I(nz.sum()),
        "shifts from which a non-zero offset was subtracted",
        src,
        STABLE,
        "m11",
        "7,196,245",
    )
    R.add(
        "pctShiftsNonzeroOffset",
        P(100 * nz.mean()),
        "share of shifts from which a non-zero offset was subtracted",
        src,
        STABLE,
        "m11",
        "60.8",
    )
    R.add(
        "nShiftsValErrAboveCut",
        I((sh.val_err_ppm > 1.3).sum()),
        "shift rows with Val_err > 1.3 ppm (kept in the table, dropped by the scorer)",
        src,
        STABLE,
        "m11",
        "6,262",
    )
    # MAD
    assessed = sh.is_far_outlier.notna().to_numpy(dtype=bool)
    flag = sh.is_far_outlier.fillna(False).astype(bool).to_numpy(dtype=bool)
    R.add(
        "nMadAssessed",
        I(assessed.sum()),
        "shifts assessed by the MAD far-outlier rule (C, CA, CB, N)",
        src,
        STABLE,
        "m11",
        "4,589,165",
    )
    R.add(
        "nMadFlagged",
        I(flag.sum()),
        "shifts flagged as far outliers",
        src,
        STABLE,
        "m11",
        "5,807",
    )
    R.add(
        "pctMadFlagged",
        P(100 * flag.sum() / assessed.sum(), 3),
        "share of assessed shifts flagged",
        src,
        STABLE,
        "m11",
        "0.127",
    )
    fa = sh.loc[flag, "atom_id"].astype(str).value_counts()
    for a, p_ in (("CB", "1,786"), ("N", "1,732"), ("CA", "1,322"), ("C", "967")):
        R.add(
            "nMadFlagged" + (a.capitalize() if len(a) == 1 else a[0] + a[1:].lower()),
            I(fa.get(a, 0)),
            f"far-outlier flags on {a}",
            src,
            STABLE,
            "m11",
            p_,
        )
    fids = set(ids[flag].astype(str))
    R.add(
        "nMadChainsFlagged",
        I(len(fids)),
        "released chains with at least one far-outlier flag",
        src,
        STABLE,
        "m11",
        "1,674",
    )
    for t_, p_ in zip(TIERS[1:], ["1,506", "1,280", "512"]):
        R.add(
            f"nMadChainsFlagged{TCAP[t_]}",
            I(len(fids & C.tier_ids[t_])),
            f"{t_}-tier chains with a far-outlier flag",
            src,
            PENDING,
            "m11",
            p_,
        )
    R.add(
        "nMadFlagsStrict",
        I((flag & ids.isin(C.tier_ids["strict"]).to_numpy(dtype=bool)).sum()),
        "far-outlier flags on strict-tier chains",
        src,
        PENDING,
        "m11",
        "1,848",
    )
    for t_, p_ in zip(TIERS, ["673", "654", "570", "244"]):
        R.add(
            f"nMadReps{TCAP[t_]}",
            I(len(fids & C.reps[t_])),
            f"{t_} training representatives with a far-outlier flag",
            src,
            PENDING,
            "m11",
            p_,
        )
    fr = sh.loc[flag, "comp_id"].astype(str).value_counts()
    for aa, name, p_ in (
        ("GLY", "Gly", "1,279"),
        ("PRO", "Pro", "966"),
        ("SER", "Ser", "447"),
        ("ALA", "Ala", "391"),
    ):
        R.add(
            f"nMadFlagged{name}",
            I(fr.get(aa, 0)),
            f"far-outlier flags on {aa} residues",
            src,
            STABLE,
            "m11",
            p_,
        )
    metal = set(df.id[df.has_metal == True])  # noqa: E712
    R.add(
        "pctMadFlagsMetal",
        P(100 * (flag & ids.isin(metal).to_numpy(dtype=bool)).sum() / flag.sum()),
        "share of far-outlier flags on metal-containing chains",
        src + " x Parquet has_metal",
        STABLE,
        "m11",
        "13.1",
    )
    on_metal = ids.isin(metal).to_numpy(dtype=bool)
    R.add(
        "pctMadAssessedMetal",
        P(100 * (assessed & on_metal).sum() / assessed.sum()),
        "share of ASSESSED shifts (C, CA, CB, N) that lie on metal-containing chains (baseline for pctMadFlagsMetal)",
        src + " x Parquet has_metal; = S16_summary.json metal.pct_assessed_on_metal",
        STABLE,
        "m11",
        "10.7",
    )
    atom = sh.atom_id.astype(str).to_numpy()
    for a in ("N", "C"):
        A = a.capitalize()
        for grp, mask_, tag in (
            ("metal", on_metal, "Metal"),
            ("other", ~on_metal, "Other"),
        ):
            sel_ = assessed & mask_ & (atom == a)
            R.add(
                f"madFlagsPerThousand{tag}{A}",
                F(1000 * flag[sel_].sum() / sel_.sum(), 1),
                f"far-outlier flags per 1,000 assessed {a} shifts on {grp} chains (Supplementary Fig. S16d)",
                src + " x Parquet has_metal",
                STABLE,
                "m11",
            )
    for name, tids, pc, ps in (
        ("TriZOD", C.test_tz, "57", "158"),
        ("CheZOD", C.test_cz, "8", "12"),
    ):
        R.add(
            f"nMadChainsTest{name}",
            I(len(fids & set(tids))),
            f"{name} test chains with a far-outlier flag",
            src,
            STABLE,
            "r5",
            pc,
        )
        R.add(
            f"nMadFlagsTest{name}",
            I((flag & ids.isin(set(tids)).to_numpy(dtype=bool)).sum()),
            f"far-outlier flags on {name} test chains",
            src,
            STABLE,
            "r5",
            ps,
        )
    R.add(
        "nMadChainsTest",
        I(len(fids & C.tests)),
        "test chains (both sets) with a far-outlier flag",
        src,
        STABLE,
        "r5",
        "65",
    )
    # metadata-derived (documented in the Parquet itself)
    m = re.search(
        r"([\d,]+) values, ([\d.]+) % of the ([\d,]+) deposited values, on (\d+) distinct Comp_IDs across ([\d,]+) of",
        meta.get("not_covered", ""),
    )
    if m:
        R.add(
            "nShiftsNoncanonical",
            m.group(1).replace(",", "{,}"),
            "deposited values on non-canonical residues (excluded from the table)",
            src + " metadata not_covered",
            STABLE,
            "m11",
            "16,502",
        )
        R.add(
            "pctShiftsNoncanonical",
            m.group(2) + r"\%",
            "share of deposited values on non-canonical residues",
            src + " metadata",
            STABLE,
            "m11",
            "0.139",
        )
        R.add(
            "nShiftsDeposited",
            m.group(3).replace(",", "{,}"),
            "deposited values on released chains (canonical + non-canonical)",
            src + " metadata",
            STABLE,
            "m11",
        )
        R.add(
            "nNoncanonicalCompIDs",
            m.group(4),
            "distinct non-canonical Comp_IDs",
            src + " metadata",
            STABLE,
            "m11",
        )
        R.add(
            "nChainsNoncanonicalShifts",
            m.group(5).replace(",", "{,}"),
            "chains with shifts on non-canonical residues",
            src + " metadata",
            STABLE,
            "m11",
        )
    pol = meta.get("sidechain_transfer_policy", "")
    m1 = re.search(
        r"13C: beta = ([\d.]+) \[([\d.]+), ([\d.]+)\].*?RMSE ([\d.]+) -> ([\d.]+) ppm",
        pol,
    )
    m2 = re.search(r"1H: beta = ([\d.]+)", pol)
    m3 = re.search(r"15N: beta = ([\d.]+) \[([\d.]+), ([\d.]+)\], and ([\d.]+)", pol)
    if m1 and m2 and m3:
        msrc = src + " metadata sidechain_transfer_policy"
        R.add(
            "betaSidechainCarbon",
            m1.group(1),
            "side-chain transfer slope for 13C (offset applied when beta > 0.5)",
            msrc,
            STABLE,
            "m11",
            "0.785",
        )
        R.add(
            "rmseSidechainCarbonBefore",
            m1.group(4),
            "side-chain 13C RMSE before applying the offset (ppm)",
            msrc,
            STABLE,
            "m11",
            "0.919",
        )
        R.add(
            "rmseSidechainCarbonAfter",
            m1.group(5),
            "side-chain 13C RMSE after applying the offset (ppm)",
            msrc,
            STABLE,
            "m11",
            "0.818",
        )
        R.add(
            "betaSidechainProton",
            m2.group(1),
            "side-chain transfer slope for 1H (left raw)",
            msrc,
            STABLE,
            "m11",
            "0.079",
        )
        R.add(
            "betaSidechainNitrogen",
            m3.group(1),
            "side-chain transfer slope for 15N (left raw)",
            msrc,
            STABLE,
            "m11",
            "0.366",
        )
        R.add(
            "betaSidechainNitrogenIV",
            m3.group(4),
            "15N slope after instrumental-variable disattenuation against PANAV",
            msrc,
            STABLE,
            "m11",
        )
    rule = meta.get("far_outlier_rule", "")
    m = re.search(
        r"C: (\d+) \(about ([\d.]+) normal sigma\).*?CB: (\d+) \(about ([\d.]+) normal sigma\)",
        rule,
    )
    if m:
        R.add(
            "madKBackbone",
            m.group(1),
            "MAD far-outlier threshold |mad_z| for C, CA and N (raw MAD units)",
            src + " metadata far_outlier_rule",
            STABLE,
            "par",
            "8",
        )
        R.add(
            "madKBackboneSigma",
            m.group(2),
            "the same threshold in normal-sigma equivalents",
            src + " metadata",
            STABLE,
            "par",
            "5.4",
        )
        R.add(
            "madKCb",
            m.group(3),
            "MAD far-outlier threshold |mad_z| for CB (raw MAD units)",
            src + " metadata",
            STABLE,
            "par",
            "12",
        )
        R.add(
            "madKCbSigma",
            m.group(4),
            "the CB threshold in normal-sigma equivalents",
            src + " metadata",
            STABLE,
            "par",
            "8.1",
        )
    del sh


def sec_sidechain_doc(C):
    p = PATHS.sidechain_doc
    if not p.exists():
        for n, v in [
            ("medianSidechainShiftsPerChainUnfiltered", "109"),
            ("pctMethylCoverageUnfiltered", "46.7\\%"),
            ("pctMethylCoverageStrict", "63.7\\%"),
        ]:
            R.unrepro(
                n,
                v,
                "side-chain coverage",
                "docs/dataset/sidechain-coverage.md missing",
                "m11",
            )
        return
    txt = p.read_text()
    src = rel(p) + " (documented; scripts/sidechain_coverage.py)"

    def table_rows(header_start):
        i = txt.find(header_start)
        rows = []
        for line in txt[i:].splitlines()[2:]:
            if not line.startswith("|"):
                break
            rows.append([c.strip() for c in line.strip("|").split("|")])
        return rows

    bulk = {r[0]: r for r in table_rows("| tier | chains | chains with side chains")}
    comp = {r[0]: r for r in table_rows("| tier | all residues |")}
    meth = {r[0]: r for r in table_rows("| tier | overall | ALA |")}
    aro = {r[0]: r for r in table_rows("| tier | overall | HIS |")}
    pl = {
        "med": ["109", "133", "208", "312"],
        "comp": ["0.308", "0.321", "0.369", "0.418"],
        "ilvma": ["46.7", "48.3", "55.7", "63.7"],
        "aro": ["0.137", None, None, "0.209"],
    }
    for j, t_ in enumerate(TIERS):
        T = TCAP[t_]
        st = STABLE if t_ == "unfiltered" else PENDING
        R.add(
            f"medianSidechainShiftsPerChain{T}",
            bulk[t_][4],
            f"{t_}: median side-chain shifts per chain",
            src,
            st,
            "m11",
            pl["med"][j],
        )
        R.add(
            f"sidechainCompleteness{T}",
            comp[t_][1],
            f"{t_}: per-residue side-chain completeness over all residues",
            src,
            st,
            "m11",
            pl["comp"][j],
        )
        R.add(
            f"pctMethylCoverage{T}",
            meth[t_][1].replace(" %", "").replace("%", "") + r"\%",
            f"{t_}: methyl-TROSY (ILVMA) 13C coverage",
            src,
            st,
            "m11",
            pl["ilvma"][j],
        )
        R.add(
            f"aromaticCoverage{T}",
            aro[t_][1],
            f"{t_}: aromatic ring 13C/15N coverage",
            src,
            st,
            "m11",
            pl["aro"][j],
        )


# ---------------------------------------------------------------- replicate analyses
def rep_loo(C) -> pd.DataFrame:
    """Port of trust_first/rep_tier_k.py: LOO deviation per chain, groups with >= 3 entries."""
    df = C.df
    rows = []
    grp = df.groupby("sequence").filter(lambda g: g.entry_id.nunique() >= 3)
    for _seq, g in grp.groupby("sequence"):
        G = np.vstack([np.asarray(x, float) for x in g.gscores])
        K = np.vstack([np.asarray(x, float) for x in g.k])
        for j, (_idx, r) in enumerate(g.iterrows()):
            others = np.delete(G, j, axis=0)
            n_ok = (~np.isnan(others)).sum(0)
            cons = np.nanmedian(others, axis=0)
            m = ~np.isnan(G[j]) & (n_ok >= 2)
            if m.sum() < 5:
                continue
            d = np.abs(G[j][m] - cons[m])
            flip = (G[j][m] > G_DIS) != (cons[m] > G_DIS)
            rows.append(
                {
                    "id": r.id,
                    "tier": r.scored_tier,
                    "bound": bool(r.is_bound),
                    "dev": d.mean(),
                    "flip": flip.mean(),
                    "medk": np.nanmedian(K[j][m]),
                }
            )
    return pd.DataFrame(rows)


def sec_replicates(C):
    df = C.df
    src_loo = "Parquet + per-tier scores.json IDs; LOO consensus over exact-sequence groups with >= 3 entries (port of trust_first/rep_tier_k.py)"
    c = rep_loo(C)
    C.rep_chain = c
    c["rank"] = c.tier.map(RANK)
    R.add(
        "nRepChains",
        I(len(c)),
        "chains with >= 5 residues comparable to a leave-one-out replicate consensus",
        src_loo,
        PREVIEW,
        "r4",
        "2,543",
    )
    for scope, sub, sfx in (("all", c, ""), ("non-bound", c[~c.bound], "NonBound")):
        plan_ex = {
            "": ["0.059", "0.045", "0.034", "0.029"],
            "NonBound": ["0.070", "0.053", "0.043", "0.032"],
        }[sfx]
        plan_cu = {
            "": ["0.039", "0.037", "0.032", "0.029"],
            "NonBound": ["0.049", "0.044", "0.039", "0.032"],
        }[sfx]
        plan_cs = {
            "": ["16.6", "13.7", "10.9", "6.4"],
            "NonBound": ["23.7", "19.4", "15.8", "5.9"],
        }[sfx]
        plan_cf = {
            "": ["5.6", "4.5", "4.3", "3.2"],
            "NonBound": ["7.9", "6.1", "6.0", "4.0"],
        }[sfx]
        plan_cn = {
            "": ["2,543", "2,057", "1,173", "295"],
            "NonBound": ["1,314", "980", "568", "153"],
        }[sfx]
        plan_en = {"": ["486", "884", "878", "295"], "NonBound": [None] * 4}[sfx]
        plan_es = {"": ["29.2", "17.3", "12.4", "6.4"], "NonBound": [None] * 4}[sfx]
        plan_ef = {"": ["10.4", "4.8", "4.6", "3.2"], "NonBound": [None] * 4}[sfx]
        for j, t_ in enumerate(TIERS):
            T = TCAP[t_]
            ex = sub[sub.tier == t_]
            cu = sub[sub["rank"] >= j]
            R.add(
                f"repDevExcl{T}{sfx}",
                F(ex.dev.median(), 3),
                f"replicate LOO deviation, median, EXCLUSIVE stratum strictest tier = {t_} ({scope}); never quote unfiltered as 'without filtering'",
                src_loo,
                PREVIEW,
                "r4",
                plan_ex[j],
            )
            R.add(
                f"nRepExcl{T}{sfx}",
                I(len(ex)),
                f"chains in the exclusive {t_} stratum ({scope})",
                src_loo,
                PREVIEW,
                "r4",
                plan_en[j],
            )
            R.add(
                f"pctRepDevAboveTenthExcl{T}{sfx}",
                P(100 * (ex.dev > 0.1).mean()),
                f"share of chains deviating > 0.1, exclusive {t_} ({scope})",
                src_loo,
                PREVIEW,
                "r4",
                plan_es[j],
            )
            R.add(
                f"pctRepFlipExcl{T}{sfx}",
                P(100 * ex.flip.mean()),
                f"mean flip rate at G = 0.65, exclusive {t_} ({scope})",
                src_loo,
                PREVIEW,
                "r4",
                plan_ef[j],
            )
            R.add(
                f"repDevCum{T}{sfx}",
                F(cu.dev.median(), 3),
                f"replicate LOO deviation, median, CUMULATIVE tier {t_} (chains scored at {t_} or stricter; {scope})",
                src_loo,
                PREVIEW,
                "abs" if sfx == "" else "r4",
                plan_cu[j],
            )
            R.add(
                f"nRepCum{T}{sfx}",
                I(len(cu)),
                f"chains in the cumulative {t_} tier ({scope})",
                src_loo,
                PREVIEW,
                "r4",
                plan_cn[j],
            )
            R.add(
                f"pctRepDevAboveTenthCum{T}{sfx}",
                P(100 * (cu.dev > 0.1).mean()),
                f"share of chains deviating > 0.1 from their replicates, cumulative {t_} ({scope})",
                src_loo,
                PREVIEW,
                "abs" if sfx == "" else "r4",
                plan_cs[j],
            )
            R.add(
                f"pctRepFlipCum{T}{sfx}",
                P(100 * cu.flip.mean()),
                f"mean flip rate at G = 0.65, cumulative {t_} ({scope})",
                src_loo,
                PREVIEW,
                "r4",
                plan_cf[j],
            )
        red = 100 * (1 - sub[sub["rank"] >= 3].dev.median() / sub.dev.median())
        R.add(
            f"pctRepDevReductionCum{sfx}",
            P(red, 0),
            f"relative reduction of the median LOO deviation from all chains to strict (cumulative; {scope})",
            src_loo,
            PREVIEW,
            "r4",
            {"": "26", "NonBound": "35"}[sfx],
        )
    # k strata, exclusive, non-bound (as in the plan's wording rule)
    nb = c[~c.bound].copy()
    nb["kbin"] = pd.cut(nb.medk, [0, 12, 17, 21], labels=["KLow", "KMid", "KHigh"])
    pk = {("KLow", "unfiltered"): "0.067", ("KLow", "strict"): "0.039", ("KHigh", "unfiltered"): "0.254", ("KHigh", "tolerant"): "0.054",
          ("KHigh", "moderate"): "0.039", ("KHigh", "strict"): "0.028", ("KMid", "moderate"): "0.039", ("KMid", "strict"): "0.043"}  # fmt: skip
    pn = {("KMid", "strict"): "39"}
    for kb, kd in (("KLow", "k <= 12"), ("KMid", "k 13-17"), ("KHigh", "k 18-21")):
        for t_ in TIERS:
            s = nb[(nb.kbin == kb) & (nb.tier == t_)]
            R.add(
                f"repDevExcl{TCAP[t_]}NonBound{kb}",
                F(s.dev.median(), 3),
                f"median LOO deviation, exclusive {t_}, non-bound, chain median {kd}",
                src_loo,
                PREVIEW,
                "r4",
                pk.get((kb, t_)),
            )
            R.add(
                f"nRepExcl{TCAP[t_]}NonBound{kb}",
                I(len(s)),
                "chains in that stratum",
                src_loo,
                PREVIEW,
                "r4",
                pn.get((kb, t_)),
            )
    # cumulative k strata (all chains) for Fig. 3a
    call = c.copy()
    call["kbin"] = pd.cut(call.medk, [0, 12, 17, 21], labels=["KLow", "KMid", "KHigh"])
    for kb in ("KLow", "KMid", "KHigh"):
        for j, t_ in enumerate(TIERS):
            s = call[(call.kbin == kb) & (call["rank"] >= j)]
            R.add(
                f"repDevCum{TCAP[t_]}{kb}",
                F(s.dev.median(), 3),
                f"median LOO deviation, cumulative {t_}, all chains, chain-median-k stratum {kb}",
                src_loo,
                PREVIEW,
                "r4",
            )

    # ---- teaser D: >= 3-chain groups (any entries), exclusive
    G = [np.asarray(g, float) for g in df.gscores]
    Z = [np.asarray(z, float) for z in df.zscores]
    K = [np.asarray(k, float) for k in df.k]
    medk = [
        np.nanmedian(np.where(np.isnan(g), np.nan, k))
        if np.isfinite(g).any()
        else np.nan
        for g, k in zip(G, K)
    ]
    df["medk"] = medk
    rows = []
    for seq, g in df.groupby("sequence"):
        if len(g) < 3:
            continue
        idx = list(g.index)
        Gm = np.vstack([G[i] for i in idx])
        for j, i in enumerate(idx):
            loo = np.nanmedian(np.delete(Gm, j, axis=0), axis=0)
            m = np.isfinite(G[i]) & np.isfinite(loo)
            if m.sum() < 5:
                continue
            rows.append(
                {
                    "seq": seq,
                    "tier": df.srank[i],
                    "medk": df.medk[i],
                    "bound": df.is_bound[i],
                    "dev": np.abs(G[i][m] - loo[m]).mean(),
                }
            )
    D = pd.DataFrame(rows)
    srcD = "Parquet; LOO over exact-sequence groups with >= 3 chains (port of one_idea/teaser.py section D)"
    R.add(
        "nRepGroupsThreeChains",
        I(D.seq.nunique()),
        "exact-sequence groups with >= 3 chains (any entries)",
        srcD,
        PREVIEW,
        "r4",
        "664",
    )
    R.add(
        "nRepChainsThreeChains",
        I(len(D)),
        "chains in those groups with >= 5 comparable residues",
        srcD,
        PREVIEW,
        "r4",
        "3,203",
    )
    D["kbin"] = pd.cut(D.medk, [0, 12, 17, 21], labels=["KLow", "KMid", "KHigh"])
    pd_ = {
        "": ["0.057", "0.049", "0.036", "0.031"],
        "KMid": ["0.079", "0.059", "0.032", "0.033"],
        "KHigh": ["0.088", "0.056", "0.028", "0.028"],
    }
    for j, t_ in enumerate(TIERS):
        R.add(
            f"repDevGroupsExcl{TCAP[t_]}",
            F(D[D.tier == j].dev.median(), 3),
            f">=3-chain groups: median LOO deviation, exclusive {t_}",
            srcD,
            PREVIEW,
            "r4",
            pd_[""][j],
        )
        for kb in ("KMid", "KHigh"):
            R.add(
                f"repDevGroupsExcl{TCAP[t_]}{kb}",
                F(D[(D.tier == j) & (D.kbin == kb)].dev.median(), 3),
                f">=3-chain groups: median LOO deviation, exclusive {t_}, stratum {kb}",
                srcD,
                PREVIEW,
                "r4",
                pd_[kb][j],
            )

    # ---- residue-level replicate r (port of user_first/ceiling_and_baseline.py part 1)
    rng = random.Random(0)
    pairs = []
    for seq, sub in df[~df["is_bound"]].groupby("sequence"):
        if sub["entry_id"].nunique() < 2:
            continue
        rows_ = sub.to_dict("records")
        cand = [
            (a, b_)
            for ia, a in enumerate(rows_)
            for b_ in rows_[ia + 1 :]
            if a["entry_id"] != b_["entry_id"]
        ]
        rng.shuffle(cand)
        for a, b_ in cand[:20]:
            pairs.append((seq, a, b_))
    res = {}
    for seq, a, b_ in pairs:
        ga, gb = np.asarray(a["gscores"], float), np.asarray(b_["gscores"], float)
        m = ~np.isnan(ga) & ~np.isnan(gb)
        if m.sum() < 10:
            continue
        same = (
            abs(a["ph"] - b_["ph"]) <= 0.2
            and abs(a["temperature"] - b_["temperature"]) <= 2
        )
        diff = (
            abs(a["ph"] - b_["ph"]) >= 1
            or abs(a["temperature"] - b_["temperature"]) >= 10
        )
        ta, tb = a["scored_tier"], b_["scored_tier"]
        both = TIERS[min(RANK[ta], RANK[tb])]
        keys = [("tier", t_) for t_ in TIERS if RANK[t_] <= RANK[both]]
        if same:
            keys.append(("cond", "same"))
        if diff:
            keys.append(("cond", "different"))
        for k_ in keys:
            r_ = res.setdefault(k_, {"x": [], "y": [], "n": 0, "seqs": set()})
            r_["x"].append(ga[m])
            r_["y"].append(gb[m])
            r_["n"] += 1
            r_["seqs"].add(seq)
    srcR = "Parquet; pairs of identical-sequence non-bound chains from different entries, <= 20 pairs per sequence, seed 0 (port of user_first/ceiling_and_baseline.py)"
    R.add(
        "nRepPairsResidue",
        I(len(pairs)),
        "replicate chain pairs sampled for the residue-level reliability analysis",
        srcR,
        PREVIEW,
        "r4",
        "2,246",
    )
    R.add(
        "nRepPairSequences",
        I(len({p[0] for p in pairs})),
        "sequences contributing replicate pairs",
        srcR,
        PREVIEW,
        "r4",
        "919",
    )
    plan_r = {"unfiltered": ("0.741", "2,024", "870", "0.861", "0.098", "8.3"), "tolerant": ("0.839", "1,347", "687", "0.916", "0.068", "5.2"),
              "moderate": ("0.848", "668", "428", "0.921", "0.055", "4.4"), "strict": ("0.909", "141", "122", "0.953", "0.027", "1.5")}  # fmt: skip
    for key in [("tier", t_) for t_ in TIERS] + [
        ("cond", "same"),
        ("cond", "different"),
    ]:
        r_ = res[key]
        x, y = np.concatenate(r_["x"]), np.concatenate(r_["y"])
        pr = np.corrcoef(x, y)[0, 1]
        tag = (
            TCAP[key[1]]
            if key[0] == "tier"
            else ("SameCondition" if key[1] == "same" else "DifferentCondition")
        )
        p = (
            plan_r.get(key[1], (None,) * 6)
            if key[0] == "tier"
            else {
                "same": ("0.852", "839", None, None, "0.067", "5.7"),
                "different": ("0.602", "945", None, None, "0.138", "11.9"),
            }[key[1]]
        )
        sec_ = (
            "abs" if key[0] == "tier" and key[1] in ("unfiltered", "strict") else "r4"
        )
        R.add(
            f"rRepResidue{tag}",
            F(pr, 3),
            f"residue-level Pearson r of G between replicate chains, {key[0]} = {key[1]}"
            + (
                " (cumulative: a pair counts at every tier containing both chains)"
                if key[0] == "tier"
                else ""
            ),
            srcR,
            PREVIEW,
            sec_,
            p[0],
        )
        R.add(
            f"nRepPairsResidue{tag}",
            I(r_["n"]),
            f"replicate pairs in stratum {key[1]}",
            srcR,
            PREVIEW,
            "r4",
            p[1],
        )
        R.add(
            f"nRepSeqsResidue{tag}",
            I(len(r_["seqs"])),
            f"sequences in stratum {key[1]}",
            srcR,
            PREVIEW,
            "r4",
            p[2],
        )
        R.add(
            f"rSqrtRepResidue{tag}",
            F(math.sqrt(pr), 3),
            "square root of that r (single-measurement reliability ceiling)",
            srcR,
            PREVIEW,
            "r4",
            p[3],
        )
        R.add(
            f"maeRepResidue{tag}",
            F(np.mean(np.abs(x - y)), 3),
            f"residue-level mean absolute G difference between replicates, stratum {key[1]}",
            srcR,
            PREVIEW,
            "r4",
            p[4],
        )
        R.add(
            f"pctRepResidueFlip{tag}",
            P(100 * np.mean((x > G_DIS) != (y > G_DIS))),
            f"residue-level flip rate at G = 0.65, stratum {key[1]}",
            srcR,
            PREVIEW,
            "r4",
            p[5],
        )
        if key[0] == "tier" and key[1] in ("unfiltered", "strict"):
            R.add(
                f"rRepResidueShort{tag}",
                F(pr, 2),
                "same r rounded to two decimals (abstract)",
                srcR,
                PREVIEW,
                "abs",
                {"unfiltered": "0.74", "strict": "0.91"}[key[1]],
            )
    C.rep_ceiling = {
        k_: np.corrcoef(np.concatenate(v["x"]), np.concatenate(v["y"]))[0, 1]
        for k_, v in res.items()
    }

    # ---- same vs different conditions, non-bound, chain-level |dG| (teaser section E) and all chains (positioning)
    def cond_pairs(sub, cap=50):
        rng_ = random.Random(0)
        out = []
        for seq, g in sub.groupby("sequence"):
            if len(g) < 2:
                continue
            prs = list(itertools.combinations(list(g.index), 2))
            if len(prs) > cap:
                prs = rng_.sample(prs, cap)
            for a, b_ in prs:
                if df.entry_id[a] == df.entry_id[b_]:
                    continue
                m = np.isfinite(G[a]) & np.isfinite(G[b_])
                if m.sum() < 10:
                    continue
                d = np.abs(G[a][m] - G[b_][m])
                out.append(
                    {
                        "seq": seq,
                        "mad": d.mean(),
                        "flip": np.mean((G[a][m] > G_DIS) != (G[b_][m] > G_DIS)),
                        "dph": abs(df.ph[a] - df.ph[b_]),
                        "dT": abs(df.temperature[a] - df.temperature[b_]),
                    }
                )
        E = pd.DataFrame(out)
        return E[(E.dph <= 0.2) & (E.dT <= 2)], E[(E.dph >= 1) | (E.dT >= 10)]

    same, diff = cond_pairs(df[~df.is_bound])
    srcE = "Parquet; non-bound replicate pairs, <= 50 pairs per sequence, seed 0; same: |dpH| <= 0.2 and |dT| <= 2 K; different: |dpH| >= 1 or |dT| >= 10 K (port of one_idea/teaser.py section E)"
    R.add(
        "nCondPairsSame",
        I(len(same)),
        "same-condition replicate pairs (non-bound)",
        srcE,
        PREVIEW,
        "r4",
        "898",
    )
    R.add(
        "nCondPairsDifferent",
        I(len(diff)),
        "different-condition replicate pairs (non-bound)",
        srcE,
        PREVIEW,
        "r4",
        "1,088",
    )
    R.add(
        "condMedianAbsDGSame",
        F(same.mad.median(), 3),
        "median per-pair mean |dG|, same conditions",
        srcE,
        PREVIEW,
        "r4",
        "0.037",
    )
    R.add(
        "condMedianAbsDGDifferent",
        F(diff.mad.median(), 3),
        "median per-pair mean |dG|, different conditions",
        srcE,
        PREVIEW,
        "r4",
        "0.078",
    )
    R.add(
        "pctCondFlipSame",
        P(100 * same.flip.mean()),
        "mean flip rate at G = 0.65, same conditions",
        srcE,
        PREVIEW,
        "r4",
        "7.0",
    )
    R.add(
        "pctCondFlipDifferent",
        P(100 * diff.flip.mean()),
        "mean flip rate at G = 0.65, different conditions",
        srcE,
        PREVIEW,
        "r4",
        "14.9",
    )
    sa, da = cond_pairs(df)
    R.add(
        "condMedianAbsDGSameAll",
        F(sa.mad.median(), 3),
        "median per-pair mean |dG|, same conditions, all chains (exploratory)",
        srcE.replace("non-bound", "all"),
        PREVIEW,
        "r4",
        "0.033",
    )
    R.add(
        "condMedianAbsDGDifferentAll",
        F(da.mad.median(), 3),
        "median per-pair mean |dG|, different conditions, all chains (exploratory)",
        srcE.replace("non-bound", "all"),
        PREVIEW,
        "r4",
        "0.068",
    )
    R.add(
        "ratioCondDisagreement",
        F(diff.mad.median() / same.mad.median(), 1),
        "ratio of the two medians (different / same; 'about twice')",
        srcE,
        PREVIEW,
        "r4",
        None,
    )

    # ---- deviant-flag enrichment (trust_first/cases.py) on unfiltered-only replicate chains
    m = c.merge(df[["id", "physical_state", "cosolvent_evidence", "sample_state_evidence", "ph", "temperature",
                    off_sigma_col("CA"), off_sigma_col("N"), off_sigma_col("C"), off_sigma_col("CB"), off_sigma_col("H"), off_sigma_col("HA")]], on="id")  # fmt: skip
    deny = {
        "denatured",
        "unfolded",
        "amyloid",
        "amyloid fibril",
        "fibril",
        "fibrils",
        "molten globule",
        "misfolded",
        "aggregated",
        "non-native",
        "intermediate",
        "partially denatured",
    }

    def flags(r):
        out = set()
        if str(r.sample_state_evidence) == "solid":
            out.add("solid")
        if r.cosolvent_evidence:
            out.add("cosolvent")
        if str(r.physical_state).lower() in deny:
            out.add("state")
        offs = np.abs(
            np.array(
                [
                    r[off_sigma_col("CA")],
                    r[off_sigma_col("N")],
                    r[off_sigma_col("C")],
                    r[off_sigma_col("CB")],
                    r[off_sigma_col("H")],
                    r[off_sigma_col("HA")],
                ],
                float,
            )
        )
        if (offs > 3).any():
            out.add("offset")
        return out

    u = m[m.tier == "unfiltered"].copy()
    u["fl"] = u.apply(flags, axis=1)
    dv, cs = u[u.dev > 0.1], u[u.dev <= 0.1]
    srcF = "replicate LOO table (unfiltered-only chains) x Parquet flags (port of trust_first/cases.py)"
    R.add(
        "nDeviantChains",
        I(len(dv)),
        "unfiltered-only replicate chains deviating > 0.1 from their replicates",
        srcF,
        PREVIEW,
        "r4",
        "142",
    )
    R.add(
        "nConsistentChains",
        I(len(cs)),
        "unfiltered-only replicate chains deviating <= 0.1",
        srcF,
        PREVIEW,
        "r4",
        "344",
    )
    for fl, name, pd_v, pc_v in (
        ("state", "DeniedState", "36", "7"),
        ("cosolvent", "Cosolvent", "37", "18"),
        ("offset", "OffsetThreeSigma", "25", "8"),
        ("solid", "SolidState", "15", "31"),
    ):
        R.add(
            f"pctDeviant{name}",
            P(100 * dv.fl.map(lambda s, f_=fl: f_ in s).mean(), 0),
            f"share of deviant chains with flag {fl}",
            srcF,
            PREVIEW,
            "r4",
            pd_v,
        )
        R.add(
            f"pctConsistent{name}",
            P(100 * cs.fl.map(lambda s, f_=fl: f_ in s).mean(), 0),
            f"share of consistent chains with flag {fl}",
            srcF,
            PREVIEW,
            "r4",
            pc_v,
        )

    # ---- within-residue dk test (trust_first/kpaired.py)
    nbdf = df[~df.is_bound]
    rng = random.Random(0)
    out = []
    for _seq, g in nbdf.groupby("sequence"):
        if g.entry_id.nunique() < 2:
            continue
        Gm = np.vstack([np.asarray(x, float) for x in g.gscores])
        Zm = np.vstack([np.asarray(x, float) for x in g.zscores])
        Km = np.vstack([np.asarray(x, float) for x in g.k])
        cons = np.nanmedian(Gm, axis=0)
        prs = list(itertools.combinations(range(len(g)), 2))
        if len(prs) > 30:
            prs = rng.sample(prs, 30)
        ent = g.entry_id.values
        for a, b_ in prs:
            if ent[a] == ent[b_]:
                continue
            mm = (
                ~np.isnan(Gm[a])
                & ~np.isnan(Gm[b_])
                & ~np.isnan(Zm[a])
                & ~np.isnan(Zm[b_])
            )
            if mm.sum() == 0:
                continue
            hi = np.where(Km[a] >= Km[b_], a, b_)
            lo = np.where(Km[a] >= Km[b_], b_, a)
            ix = np.where(mm)[0]
            out.append(
                pd.DataFrame(
                    {
                        "dk": np.abs(Km[a, ix] - Km[b_, ix]),
                        "dZ": Zm[hi[ix], ix] - Zm[lo[ix], ix],
                        "dG": Gm[hi[ix], ix] - Gm[lo[ix], ix],
                        "cons": cons[ix],
                    }
                )
            )
    KP = pd.concat(out)
    KP["dkbin"] = pd.cut(
        KP.dk,
        [-0.5, 0.5, 3, 6, 9, 21],
        labels=["Zero", "OneToThree", "FourToSix", "SevenToNine", "TenPlus"],
    )
    srcK = "Parquet; same residue in two non-bound identical-sequence chains from different entries, <= 30 pairs per sequence, seed 0; ordered: replicate-consensus G < 0.3 (port of trust_first/kpaired.py)"
    R.add(
        "nKPairsResidue",
        I(len(KP)),
        "residue pairs in the within-residue shift-count test",
        srcK,
        PREVIEW,
        "r2",
        "145,717",
    )
    o = (
        KP[KP.cons < 0.3]
        .groupby("dkbin", observed=True)
        .agg(n=("dZ", "size"), dZ=("dZ", "median"), dG=("dG", "median"))
    )
    dsd = (
        KP[KP.cons > 0.7]
        .groupby("dkbin", observed=True)
        .agg(dZ=("dZ", "median"), dG=("dG", "median"))
    )
    pz = ["0.00", "0.98", "2.36", "3.84", "5.96"]
    pg = ["0.000", "0.000", "0.000", "0.005", "0.001"]
    pn_ = ["47,329", "29,017", "16,956", "12,321", "10,035"]
    for j, b_ in enumerate(o.index):
        R.add(
            f"kpairDZOrdered{b_}",
            F(o.loc[b_, "dZ"], 2),
            f"median dZ (higher-k minus lower-k), ordered consensus, dk bin {b_}",
            srcK,
            PREVIEW,
            "r2",
            pz[j],
        )
        R.add(
            f"kpairDGOrdered{b_}",
            F(o.loc[b_, "dG"], 3),
            f"median dG, ordered consensus, dk bin {b_}",
            srcK,
            PREVIEW,
            "r2",
            pg[j],
        )
        R.add(
            f"nKPairsOrdered{b_}",
            I(o.loc[b_, "n"]),
            f"residue pairs, ordered consensus, dk bin {b_}",
            srcK,
            PREVIEW,
            "r2",
            pn_[j],
        )
    R.add(
        "kpairDZDisorderedMin",
        F(dsd.dZ.min(), 2),
        "smallest bin median dZ for disordered consensus (G > 0.7)",
        srcK,
        PREVIEW,
        "r2",
        "-0.33",
    )
    R.add(
        "kpairDZDisorderedMax",
        F(dsd.dZ.max(), 2),
        "largest bin median dZ for disordered consensus",
        srcK,
        PREVIEW,
        "r2",
        "0.45",
    )
    R.add(
        "kpairDGDisorderedMin",
        F(dsd.dG.min(), 3),
        "smallest bin median dG for disordered consensus",
        srcK,
        PREVIEW,
        "r2",
        "-0.032",
    )
    R.add(
        "kpairDGDisorderedMax",
        F(dsd.dG.max(), 3),
        "largest bin median dG for disordered consensus",
        srcK,
        PREVIEW,
        "r2",
        "0.003",
    )
    # consensus split of the within-residue pairs (Fig. 2a legend: intermediate pairs not plotted)
    R.add(
        "nKPairsOrderedAll",
        I(int((KP.cons < 0.3).sum())),
        "residue pairs with ordered replicate consensus (G < 0.3), all dk bins (plotted in Fig. 2a)",
        srcK,
        PREVIEW,
        "r2",
        "115,658",
    )
    R.add(
        "nKPairsDisorderedAll",
        I(int((KP.cons > 0.7).sum())),
        "residue pairs with disordered replicate consensus (G > 0.7), all dk bins (plotted in Fig. 2a)",
        srcK,
        PREVIEW,
        "r2",
        "15,588",
    )
    R.add(
        "nKPairsIntermediate",
        I(int(len(KP) - (KP.cons < 0.3).sum() - (KP.cons > 0.7).sum())),
        "residue pairs with intermediate replicate consensus (0.3 <= G <= 0.7), not plotted in Fig. 2a",
        srcK,
        PREVIEW,
        "r2",
        "14,471",
    )

    # ---- binned dk with slopes (teaser section C)
    rng = random.Random(0)
    recs = []
    for _seq, g in nbdf.groupby("sequence"):
        if len(g) < 2:
            continue
        idx = list(g.index)
        prs = list(itertools.combinations(idx, 2))
        if len(prs) > 50:
            prs = rng.sample(prs, 50)
        cons = np.nanmedian(np.vstack([G[i] for i in idx]), axis=0)
        for a, b_ in prs:
            if df.entry_id[a] == df.entry_id[b_]:
                continue
            mm = np.isfinite(G[a]) & np.isfinite(G[b_])
            if mm.sum() < 10:
                continue
            dk = K[a][mm] - K[b_][mm]
            s_ = dk != 0
            if not s_.any():
                continue
            recs.append(
                pd.DataFrame(
                    {
                        "dk": np.abs(dk[s_]),
                        "dZ": np.sign(dk[s_]) * (Z[a][mm] - Z[b_][mm])[s_],
                        "dG": np.sign(dk[s_]) * (G[a][mm] - G[b_][mm])[s_],
                        "cons": cons[mm][s_],
                    }
                )
            )
    CC = pd.concat(recs, ignore_index=True)
    oc = CC[CC.cons < 0.2]
    srcC = "Parquet; non-bound replicate pairs, <= 50 per sequence, seed 0, residues with dk != 0; ordered: consensus G < 0.2 (port of one_idea/teaser.py section C)"
    R.add(
        "nKPairsBinned",
        I(len(CC)),
        "residue pairs with dk != 0 in the binned shift-count test",
        srcC,
        PREVIEW,
        "r2",
        "96,590",
    )
    R.add(
        "nKPairsBinnedOrdered",
        I(len(oc)),
        "of them with ordered consensus",
        srcC,
        PREVIEW,
        "r2",
        "71,785",
    )
    R.add(
        "slopeZPerShift",
        F(np.polyfit(oc.dk, oc.dZ, 1)[0], 2),
        "OLS slope of dZ on dk for ordered residues (Z units per shift)",
        srcC,
        PREVIEW,
        "r2",
        "0.43",
    )
    R.add(
        "slopeGPerShift",
        F(np.polyfit(oc.dk, oc.dG, 1)[0], 4),
        "OLS slope of dG on dk for ordered residues (G units per shift)",
        srcC,
        PREVIEW,
        "r2",
        "0.0015",
    )
    bb_ = (
        oc.assign(bin=pd.cut(oc.dk, [0, 3, 6, 9, 12, 21]))
        .groupby("bin", observed=True)
        .dZ.median()
        .tolist()
    )
    for lab, v, p_ in zip(
        ["OneToThree", "FourToSix", "SevenToNine", "TenToTwelve", "ThirteenPlus"],
        bb_,
        ["0.99", "2.33", "3.82", "5.17", "6.91"],
    ):
        R.add(
            f"kbinDZOrdered{lab}",
            F(v, 2),
            f"median dZ for ordered residues, |dk| bin {lab}",
            srcC,
            PREVIEW,
            "r2",
            p_,
        )

    # ---- Z vs k on pool residues (display candidates P13)
    pool = df.pool_tier.notna()
    gg = np.concatenate([G[i] for i in np.where(pool)[0]])
    zz = np.concatenate([Z[i] for i in np.where(pool)[0]])
    kk = np.concatenate([K[i] for i in np.where(pool)[0]])
    ok = ~np.isnan(gg) & ~np.isnan(zz)
    gg, zz, kk = gg[ok], zz[ok], kk[ok]
    T_ = pd.DataFrame({"g": gg, "z": zz, "kb": pd.cut(kk, [0, 6, 9, 12, 15, 18, 21])})
    o = T_[T_.g < 0.2].groupby("kb", observed=True).z.median()
    dd_ = T_[T_.g > 0.8].groupby("kb", observed=True).z.median()
    gm = T_.groupby("kb", observed=True).g.median()
    srcP = "Parquet, unfiltered training-pool residues (display_candidates P13)"
    R.add(
        "zOrderedLowK",
        F(o.iloc[0], 2),
        "median Z of ordered residues (G < 0.2) with k <= 6, pool residues",
        srcP,
        PENDING,
        "r2",
        "7.13",
    )
    R.add(
        "zOrderedHighK",
        F(o.iloc[-1], 2),
        "median Z of ordered residues with k 19-21, pool residues",
        srcP,
        PENDING,
        "r2",
        "14.13",
    )
    R.add(
        "zDisorderedMin",
        F(dd_.min(), 2),
        "lowest k-bin median Z of disordered residues (G > 0.8)",
        srcP,
        PENDING,
        "r2",
        "-0.76",
    )
    R.add(
        "zDisorderedMax",
        F(dd_.max(), 2),
        "highest k-bin median Z of disordered residues",
        srcP,
        PENDING,
        "r2",
        "-0.34",
    )
    R.add(
        "gMedianLowK",
        F(gm.iloc[0], 3),
        "median G of all pool residues with k <= 6",
        srcP,
        PENDING,
        "r2",
        "0.108",
    )
    R.add(
        "gMedianHighK",
        F(gm.iloc[-1], 3),
        "median G of all pool residues with k 19-21",
        srcP,
        PENDING,
        "r2",
        "0.076",
    )
    R.add(
        "rGZ",
        F(np.corrcoef(gg, zz)[0, 1], 3),
        "Pearson correlation of G and Z over pool residues",
        srcP,
        PENDING,
        "r2",
        "-0.899",
    )


def sec_ubiquitin(C):
    df = C.df
    u = df[df.sequence == UBQ].copy()
    R.add(
        "nUbqChains",
        I(len(u)),
        "chains with the exact 76-aa ubiquitin sequence",
        "Parquet sequence",
        STABLE,
        "abs",
        "40",
    )
    R.add(
        "nUbqEntries",
        I(u.entry_id.nunique()),
        "BMRB entries behind them",
        "Parquet",
        STABLE,
        "abs",
        "34",
    )
    R.add(
        "nUbqBound",
        I(u.is_bound.sum()),
        "of them bound",
        "Parquet is_bound",
        STABLE,
        "r2",
        "12",
    )
    nb = u[~u.is_bound]
    R.add(
        "nUbqNonBoundChains",
        I(len(nb)),
        "non-bound ubiquitin chains (plotted in Fig. 1a)",
        "Parquet",
        STABLE,
        "r2",
        "28",
    )
    R.add(
        "nUbqNonBoundEntries",
        I(nb.entry_id.nunique()),
        "entries behind the non-bound ubiquitin chains",
        "Parquet",
        STABLE,
        "r2",
        "22",
    )
    # Fig. 2a chain set (11 Nov 2026): the native non-bound chains, selected by annotation in
    # scripts/figures/paper/fig1a_ubiquitin.py (select_chains, NATIVE_RULE) and imported from
    # there, so the macros and the figure's coloured chains and core strip are one chain set.
    # (Until 11 Nov the macros used the writing plan's mean G < 0.5 selection, which kept the
    # three 2,500 bar chains.)
    f2a = paper("fig1a_ubiquitin")
    ub = f2a.select_chains(df)
    nat = ub[ub.native].reset_index(drop=True)
    src = (
        "Parquet + raw NMR-STAR sample components, pressure and physical state; "
        "scripts/figures/paper/fig1a_ubiquitin.py select_chains (native: "
        + f2a.NATIVE_RULE
        + "), core_summary, spearman_core (Fig. 2a)"
    )
    rz, rg, _ = f2a.spearman_core(nat)
    R.add(
        "rhoUbqZK",
        F(rz, 2),
        "Spearman correlation of a chain's median Z over its ordered residues (G < 0.2) with "
        "its median k, over the native non-bound ubiquitin chains coloured in Fig. 2a",
        src,
        STABLE,
        "r2",
    )
    R.add(
        "rhoUbqGK",
        F(rg, 2),
        "same for the median G over the ordered residues (native non-bound ubiquitin chains, Fig. 2a)",
        src,
        STABLE,
        "r2",
    )
    R.add(
        "nUbqTeaserChains",
        I(len(nat)),
        "native non-bound ubiquitin chains (no denaturant or alcohol, below 400 bar, native "
        "physical state): the chains coloured in Fig. 2a and summarised in its core strip",
        src,
        STABLE,
        "r2",
    )
    # core strip of Fig. 2a: per median-k class (<= 6 / 7-15 / 16-21), the median of all
    # core residues (2-71) and tail residues (73-75) of the native chains in that class
    cs = f2a.core_summary(ub, ub.native.to_numpy()).set_index("k_class")
    for kc, cl in zip(f2a.sp.K_CLASSES, ("KLow", "KMid", "KHigh")):
        rr_ = cs.loc[kc]  # (not S: S() is the string formatter)
        R.add(
            f"nUbqClass{cl}",
            I(rr_.n_chains),
            f"native non-bound ubiquitin chains (Fig. 2a) in median-k class {kc}",
            src,
            STABLE,
            "r2",
        )
        R.add(
            f"zUbqCore{cl}",
            F(rr_.core_z_median, 2),
            f"median Z over core residues 2-71 of the native chains in median-k class {kc} "
            "(Fig. 2a core strip, bar)",
            src,
            STABLE,
            "r2",
        )
        R.add(
            f"gUbqCore{cl}",
            F(rr_.core_g_median, 3),
            f"median G over core residues 2-71 of the native chains in median-k class {kc} "
            "(Fig. 2a core strip, bar)",
            src,
            STABLE,
            "r2",
        )
        R.add(
            f"gUbqTail{cl}",
            F(rr_.tail_g_median, 2),
            f"median G over tail residues 73-75 of the native chains in median-k class {kc}",
            src,
            STABLE,
            "r2",
        )
    # named perturbed samples
    for eid, tag, pv in (
        ("16626", "SixteenSixTwoSix", "0.80"),
        ("15047", "FifteenOhFortySeven", "0.79"),
        ("4375", "FourThreeSevenFive", "0.68"),
        ("18611", "EighteenSixElevenA", "0.30"),
    ):
        rr = u[u.entry_id == eid].iloc[0]
        R.add(
            f"meanGUbq{tag}",
            F(rr.meanG, 2),
            f"mean G of ubiquitin entry {eid} ({rr.physical_state}, strictest tier {rr.scored_tier})",
            "Parquet",
            STABLE,
            "r2",
            pv,
        )
    for cid, tag, pv in (
        ("18610_1_1_1", "EighteenSixTenNative", "0.14"),
        ("18610_2_2_1", "EighteenSixTenDenatured", "0.47"),
    ):
        rr = u[u.id == cid]
        if len(rr):
            R.add(
                f"meanGUbq{tag}",
                F(rr.iloc[0].meanG, 2),
                f"mean G of ubiquitin chain {cid} ({rr.iloc[0].physical_state})",
                "Parquet",
                STABLE,
                "r2",
                pv,
            )
        else:
            R.unrepro(
                f"meanGUbq{tag}",
                pv,
                f"mean G of {cid}",
                "chain not found in Parquet",
                "r2",
            )
    r18611 = u[u.entry_id == "18611"].iloc[0]
    R.add(
        "tierUbqEighteenSixEleven",
        S(r18611.scored_tier),
        "strictest tier of 18611 (45 % methanol, 2,500 bar)",
        "scores/<tier>/scores.json",
        PENDING,
        "r2",
        "moderate",
    )
    for eid in ("16626", "15047", "4375"):
        assert u[u.entry_id == eid].iloc[0].scored_tier == "unfiltered"
    R.add(
        "tempUbqEighteenSixTen",
        I(u[u.entry_id == "18610"].iloc[0].temperature),
        "temperature (K) of ubiquitin entry 18610",
        "Parquet temperature",
        STABLE,
        "r2",
        "258",
    )
    # case-study clusters (S8): ubiquitin / prion / cytochrome c / alpha-synuclein
    cases = {
        "Ubq": "50330_1_1_1",
        "Prion": "16616_1_1_1",
        "CytC": "25640_1_1_1",
        "ASyn": "6968_1_1_1",
    }
    pl = {
        "Ubq": ("119", "103"),
        "Prion": ("36", "34"),
        "CytC": ("34", "32"),
        "ASyn": ("108", "71"),
    }
    if have_mmseqs():
        seqs = df.set_index("id").sequence
        hits = mmseqs_search(
            "cases",
            [(k, seqs[v]) for k, v in cases.items()],
            list(zip(df.id, df.sequence)),
            ["-s", "7.5", "--max-seqs", "20000", "-e", "1e-3"],
            C.refresh,
        )
        H = pd.DataFrame(
            hits, columns=["q", "t", "fid", "qcov", "tcov", "e", "aln", "ql", "tl"]
        )
        H = H[(H.fid >= 0.8) & (H.qcov >= 0.9) & (H.tcov >= 0.9)]
        srcm = "mmseqs easy-search of the case sequence vs all 16,851 chains, >= 80 % id, >= 90 % query and target coverage"
        for k_, cid in cases.items():
            mem = df[df.id.isin(set(H[H.q == k_].t))]
            R.add(
                f"nCase{k_}Chains",
                I(len(mem)),
                f"chains in the {k_} case-study cluster (seed {cid})",
                srcm,
                STABLE,
                "r6",
                pl[k_][0],
            )
            R.add(
                f"nCase{k_}Entries",
                I(mem.entry_id.nunique()),
                f"entries in the {k_} case-study cluster",
                srcm,
                STABLE,
                "r6",
                pl[k_][1],
            )
            if k_ == "CytC":
                R.add(
                    "nCaseCytCBound",
                    I(mem.is_bound.sum()),
                    "cytochrome-c cluster chains that are bound (heme)",
                    srcm + " x is_bound",
                    STABLE,
                    "r6",
                    "27",
                )
            if k_ == "ASyn":
                R.add(
                    "nCaseASynExcluded",
                    I((mem.split == "excluded").sum()),
                    "alpha-synuclein cluster chains with split = excluded (6968 is in CheZOD117)",
                    srcm,
                    PENDING,
                    "r6",
                    "107",
                )
            if k_ == "Ubq":
                R.add(
                    "nCaseUbqBound",
                    I(mem.is_bound.sum()),
                    "bound chains in the ubiquitin case cluster",
                    srcm,
                    STABLE,
                    "r6",
                    "39",
                )
    else:
        for k_ in cases:
            R.unrepro(
                f"nCase{k_}Chains",
                pl[k_][0],
                f"{k_} case cluster chains",
                "mmseqs not on PATH",
                "r6",
            )


def sec_failures(C):
    df = C.df
    # myoglobin triplet
    for cid, tag, pph, pT, pg, pf in (
        ("5158_1_1_1", "FiveOneFiveEight", "4.1", "323", "0.322", "16.3"),
        ("4676_1_1_1", "FourSixSevenSix", "2.3", "298", "0.721", "64.7"),
        ("4695_1_1_1", "FourSixNineFive", "5.8", None, "0.085", None),
    ):
        r = df[df.id == cid].iloc[0]
        g = np.asarray(r.gscores, float)
        R.add(
            f"phMyo{tag}",
            F(r.ph, 1),
            f"pH of myoglobin chain {cid}",
            "Parquet ph",
            STABLE,
            "r6",
            pph,
        )
        R.add(
            f"tempMyo{tag}",
            I(r.temperature),
            f"temperature (K) of myoglobin chain {cid}",
            "Parquet temperature",
            STABLE,
            "r6",
            pT,
        )
        R.add(
            f"meanGMyo{tag}",
            F(np.nanmean(g), 3),
            f"mean G of myoglobin chain {cid} ({r.physical_state})",
            "Parquet gscores",
            STABLE,
            "r6",
            pg,
        )
        R.add(
            f"pctDisorderedMyo{tag}",
            P(100 * np.nanmean(g[np.isfinite(g)] > G_DIS)),
            f"share of SCORED residues at G > 0.65 in {cid} (the plan's figure divides by all residues)",
            "Parquet gscores",
            STABLE,
            "r6",
            pf,
        )
    # all-zero-G chains
    z = df[df.meanG == 0]
    medk = np.array([np.nanmedian(np.asarray(k, float)) for k in z.k])
    sc = Counter(z.split)
    src = "Parquet gscores, k, split (port of trust_first/gzero.py)"
    R.add(
        "nAllZeroG",
        I(len(z)),
        "chains with G = 0 at every scored residue",
        src,
        PENDING,
        "r6",
        "116",
    )
    R.add(
        "nAllZeroGExcluded",
        I(sc.get("excluded", 0)),
        "of them excluded",
        src,
        PENDING,
        "r6",
        "94",
    )
    R.add(
        "nAllZeroGTrain",
        I(sc.get("train", 0)),
        "of them training representatives",
        src,
        PENDING,
        "r6",
        "17",
    )
    R.add(
        "nAllZeroGRedundant",
        I(sc.get("redundant", 0)),
        "of them redundant",
        src,
        PENDING,
        "r6",
        "5",
    )
    R.add(
        "kMedianAllZeroG",
        I(np.median(medk)),
        "median over these chains of the per-chain median k (all residues)",
        src,
        PENDING,
        "r6",
        "0",
    )
    R.add(
        "pctAllZeroGOldIDs",
        P(100 * (z.entry_id.astype(int) < 5000).mean()),
        "share of these chains with BMRB IDs < 5000",
        src,
        PENDING,
        "r6",
        "62.9",
    )
    ks = np.concatenate(
        [
            np.asarray(k, float)[np.isfinite(np.asarray(g, float))]
            for g, k in zip(z.gscores, z.k)
        ]
    )
    R.add(
        "kMedianAllZeroGScored",
        I(np.median(ks)),
        "median k over the scored residues of these chains",
        src,
        PENDING,
        "r6",
        "1",
    )
    # named chains
    for cid, tag in (
        ("5158_1_1_1", "FiveOneFiveEight"),
        ("4676_1_1_1", "FourSixSevenSix"),
        ("25640_1_1_1", "TwoFiveSixFourZero"),
    ):
        r = df[df.id == cid].iloc[0]
        R.add(
            f"splitChain{tag}", S(r.split), f"split of {cid}", "Parquet", PENDING, "r6"
        )


def sec_cosolvents_pressure(C):
    df = C.df
    if not C.raw:
        R.unrepro(
            "nPressureHighEntries",
            "15",
            "entries with a sample pressure >= 400 bar",
            "raw BMRB directory unavailable",
            "r4",
        )
        return
    # pressure (port of rev2/press.py)
    ents = {e: r["press"] for e, r in C.raw.items() if r["press"]}
    hi = {
        e: max(x[2] for x in p if x[2] is not None)
        for e, p in ents.items()
        if any(x[2] is not None and x[2] >= 400 for x in p)
    }
    src = "raw NMR-STAR 'pressure' sample-condition rows, units normalised to bar (port of rev2/press.py)"
    R.add(
        "nPressureRowEntries",
        I(len(ents)),
        "snapshot entries with a numeric pressure sample-condition row",
        src,
        STABLE,
        "r4",
        "13,166",
    )
    R.add(
        "nPressureHighEntries",
        I(len(hi)),
        "snapshot entries recording a pressure >= 400 bar",
        src,
        STABLE,
        "r4",
        "15",
    )
    scored = set(df.entry_id)
    hs = sorted(set(hi) & scored, key=int)
    R.add(
        "nPressureHighScored",
        I(len(hs)),
        "of them with a scored chain in the release",
        src + " x Parquet",
        STABLE,
        "r4",
        "13",
    )
    reach = {
        e: max((C.strictest[i] for i in df.id[df.entry_id == e]), key=lambda t: RANK[t])
        for e in hs
    }
    strict_e = sorted([e for e, t in reach.items() if t == "strict"], key=int)
    R.add(
        "nPressureHighStrictEntries",
        I(len(strict_e)),
        "high-pressure (>= 400 bar) entries that reach the strict tier",
        src + " x tier IDs",
        PENDING,
        "r4",
        "3",
    )
    R.add(
        "listPressureHighStrictEntries",
        S(", ".join(strict_e)),
        "those entries",
        src,
        PENDING,
        "r4",
        "11547, 25986, 51703",
    )
    trn = df[df.entry_id.isin(hi) & (df.split == "train")]
    R.add(
        "nPressureHighReps",
        I(len(trn)),
        "training representatives measured at >= 400 bar",
        src + " x Parquet split",
        PENDING,
        "r4",
        "2",
    )
    R.add(
        "listPressureHighReps",
        S(", ".join(sorted(trn.id))),
        "those representatives",
        src,
        PENDING,
        "r4",
        "51981_1_1_1, 51982_1_1_1",
    )
    for e, tag, pv in (
        ("18611", "EighteenSixEleven", "2,500"),
        ("18610", "EighteenSixTen", "2,500"),
        ("11547", "OneOneFiveFourSeven", "2,500"),
        ("25986", "TwoFiveNineEightSix", "500"),
        ("51703", "FiveOneSevenOhThree", "2,250"),
    ):
        R.add(
            f"pressureBar{tag}",
            I(hi[e]),
            f"recorded sample pressure of entry {e} (bar)",
            src,
            STABLE,
            "r2" if e.startswith("186") else "r4",
            pv,
        )
    # alcohols (methanol/ethanol as sample components; approximate matcher)
    alc = {e for e, r in C.raw.items() if r["alc"]}
    ch = df[df.entry_id.isin(alc)]
    srcA = "raw _Sample_component loop rows matching methanol/ethanol/MeOH/EtOH (word-bounded; lines with mercapto/fluoro skipped); approximate, recount with the real matcher after fix (b)"
    R.add(
        "nAlcoholEntries",
        I(len(alc & scored)),
        "scored entries with methanol or ethanol as a sample component (plan's 39 came from a looser grep that also hit ethanolamine and missed d3-methanol)",
        srcA,
        PREVIEW,
        "r4",
        "39",
    )
    R.add(
        "nAlcoholChains",
        I(len(ch)),
        "chains of those entries",
        srcA,
        PREVIEW,
        "r4",
        "41",
    )
    R.add(
        "nAlcoholChainsTolerantPlus",
        I(ch.id.isin(C.tier_ids["tolerant"]).sum()),
        "of them scored at tolerant or stricter",
        srcA,
        PENDING,
        "r4",
        "25",
    )
    tr = ch[ch.split == "train"]
    R.add(
        "nAlcoholReps",
        I(len(tr)),
        "of them training representatives",
        srcA,
        PENDING,
        "r4",
        "15",
    )
    for t_, p_ in zip(TIERS, ["6", "2", "6", "1"]):
        R.add(
            f"nAlcoholReps{TCAP[t_]}",
            I((tr.train_tier == t_).sum()),
            f"alcohol-sample training representatives with train_tier {t_}",
            srcA,
            PENDING,
            "r4",
            p_,
        )


def sec_organisms(C):
    df = C.df
    if not C.raw:
        return
    org = [
        C.raw.get(e, {}).get("org", {}).get(en)
        for e, en in zip(df.entry_id, df.entity_id)
    ]
    s = pd.Series(org)
    vc = s.fillna("NA").value_counts()
    src = "raw _Entity_natural_src.Organism_name_scientific (heuristic parse; approximate)"
    R.add(
        "nOrgHomoSapiens",
        I(vc.get("Homo sapiens", 0)),
        "chains from Homo sapiens (heuristic organism parse)",
        src,
        STABLE,
        "r1",
        "5,814",
    )
    R.add(
        "nOrgEColi",
        I(vc.get("Escherichia coli", 0)),
        "chains from Escherichia coli",
        src,
        STABLE,
        "r1",
        "1,244",
    )
    R.add(
        "nOrgNone",
        I(int(s.isna().sum() + s.isin([".", "?", ""]).sum())),
        "chains without a recorded organism ('.' placeholder or no _Entity_natural_src row for the entity; matches \\nOrgNotRecordedChains)",
        src,
        STABLE,
        "r1",
        "1,099",
    )
    R.add(
        "nOrgMusMusculus",
        I(vc.get("Mus musculus", 0)),
        "chains from Mus musculus",
        src,
        STABLE,
        "r1",
        "672",
    )
    R.add(
        "nOrgDistinct",
        I(s[~s.isin([".", "?", ""])].nunique()),
        "distinct organism names (the placeholder '.' excluded)",
        src,
        STABLE,
        "r1",
        "1,951",
    )


def sec_chezod(C):
    rs = json.loads((PATHS.chezod_ver / "reproduce_summary.json").read_text())
    sm = json.loads((PATHS.chezod_ver / "summary.json").read_text())
    src = (
        rel(PATHS.chezod_ver / "reproduce_summary.json")
        + " (pre-2026-10 code; re-run on 2026-10)"
    )
    po = rs["potenci_only_reproduction"]
    # Released mode (both): recomputed from the 2026-10 released labels with the S17 / Fig. 2c
    # code. The reproduce_summary.json contrast (0.389) came from a since-overwritten pre-build
    # scores.json and is not reproducible.
    s17 = paper("supp_s17_chezod_reproduction")
    d_bo = s17.per_entry(
        s17.load_chezod(s17.sp.CHEZOD_1325), s17.load_trizod(s17.BOTH), "both"
    )
    bo = {
        "n": int(len(d_bo)),
        "mae_median": float(d_bo.mae.median()),
        "mae_q3": float(d_bo.mae.quantile(0.75)),
        "mae_p95": float(np.percentile(d_bo.mae, 95)),
        "consistent_frac": float((d_bo.category != "genuine").mean()),
        "categories": {c: int((d_bo.category == c).sum()) for c in s17.CATS},
    }
    srcb = (
        rel(s17.BOTH)
        + f" ({REL} released labels, rereference-mode both) vs CheZOD1325; per-entry alignment "
        "and categories of trizod.figures.chezod via scripts/figures/paper/supp_s17_chezod_reproduction.py"
    )
    R.add(
        "nCheZODReproducedBoth",
        I(bo["n"]),
        f"CheZOD1325 entries scored and aligned in the released mode ({REL} labels)",
        srcb,
        STABLE,
        "r3",
        "1,320",
    )
    R.add(
        "nCheZODBMRBIDs",
        I(sm["chezod_entries"]),
        "CheZOD1325 BMRB IDs",
        rel(PATHS.chezod_ver / "summary.json"),
        STABLE,
        "r3",
        "1,323",
    )
    R.add(
        "nCheZODReproduced",
        I(po["n"]),
        "CheZOD1325 entries scored by TriZOD in the reproduction",
        src,
        PREVIEW,
        "r3",
        "1,320",
    )
    R.add(
        "listCheZODNotScored",
        S(", ".join(sm["not_scored_by_trizod"])),
        "CheZOD1325 entries TriZOD does not score",
        rel(PATHS.chezod_ver / "summary.json"),
        PREVIEW,
        "r3",
        "6078, 6985, 4077",
    )
    R.add(
        "maeCheZODPotenciOnly",
        F(po["mae_median"], 3),
        "median per-entry mean absolute error vs published CheZOD Z-scores, re-referencing disabled (potenci-only)",
        src,
        PREVIEW,
        "r3",
        "0.188",
    )
    R.add(
        "maeCheZODPotenciOnlyShort",
        F(po["mae_median"], 2),
        "same, two decimals (abstract)",
        src,
        PREVIEW,
        "abs",
        "0.19",
    )
    R.add(
        "maeQThreeCheZODPotenciOnly",
        F(po["mae_q3"], 3),
        "third quartile of that error (potenci-only)",
        src,
        PREVIEW,
        "r3",
        "0.362",
    )
    R.add(
        "maePNinetyFiveCheZODPotenciOnly",
        F(po["mae_p95"], 2),
        "95th percentile of that error (potenci-only)",
        src,
        PREVIEW,
        "r3",
        "0.71",
    )
    R.add(
        "pctCheZODConsistentPotenciOnly",
        P(100 * po["consistent_frac"]),
        "share of entries consistent with CheZOD (potenci-only)",
        src,
        PREVIEW,
        "r3",
        "98.3",
    )
    R.add(
        "maeCheZODBoth",
        F(bo["mae_median"], 3),
        f"median per-entry mean absolute error in the released mode (both: LACS + POTENCI/AIC; {REL} labels)",
        srcb,
        STABLE,
        "r3",
        "0.389",
    )
    R.add(
        "maeCheZODBothShort",
        F(bo["mae_median"], 2),
        "same, two decimals",
        srcb,
        STABLE,
        "r3",
        "0.39",
    )
    R.add(
        "maeQThreeCheZODBoth",
        F(bo["mae_q3"], 3),
        "third quartile, released mode",
        srcb,
        STABLE,
        "r3",
        "0.558",
    )
    R.add(
        "maePNinetyFiveCheZODBoth",
        F(bo["mae_p95"], 3),
        "95th percentile, released mode",
        srcb,
        STABLE,
        "r3",
        "0.932",
    )
    R.add(
        "pctCheZODConsistentBoth",
        P(100 * bo["consistent_frac"]),
        "share of consistent entries, released mode",
        srcb,
        STABLE,
        "r3",
        "97.1",
    )
    for k_, name in (
        ("agree", "Agree"),
        ("genuine", "Genuine"),
        ("offset_shift", "OffsetShift"),
        ("low_variance", "LowVariance"),
    ):
        R.add(
            f"nCheZODOutcome{name}PotenciOnly",
            I(po["categories"][k_]),
            f"reproduction outcome '{k_}' (potenci-only)",
            src,
            PREVIEW,
            "r3",
            {
                "agree": "1,263",
                "genuine": "22",
                "offset_shift": "12",
                "low_variance": "23",
            }[k_],
        )
        R.add(
            f"nCheZODOutcome{name}Both",
            I(bo["categories"][k_]),
            f"reproduction outcome '{k_}' (released mode, {REL} labels)",
            srcb,
            STABLE,
            "r3",
            {
                "agree": "1,192",
                "genuine": "38",
                "offset_shift": "34",
                "low_variance": "56",
            }[k_],
        )
    zc = sm["zscore_concordance_on_seq_matched"]
    s2 = rel(PATHS.chezod_ver / "summary.json")
    R.add(
        "rCheZODMedian",
        F(zc["pearson_median"], 3),
        "median per-entry Pearson r with published CheZOD Z (sequence-identical entries)",
        s2,
        PREVIEW,
        "r3",
        "0.992",
    )
    R.add(
        "rCheZODQOne",
        F(zc["pearson_q1"], 3),
        "first quartile of that r",
        s2,
        PREVIEW,
        "r3",
        "0.975",
    )
    R.add(
        "rCheZODQThree",
        F(zc["pearson_q3"], 3),
        "third quartile of that r",
        s2,
        PREVIEW,
        "r3",
        "0.997",
    )
    R.add(
        "nCheZODSequenceIdentical",
        I(zc["n_entries"]),
        "sequence-identical entries in that comparison",
        s2,
        PREVIEW,
        "r3",
        "1,224",
    )
    R.add(
        "nCheZODLengthMismatch",
        I(sm["sequence_match"]["len_mismatch"]),
        "entries whose sequence length differs (tag trimming)",
        s2,
        PREVIEW,
        "r3",
        "96",
    )
    pe = pd.read_csv(PATHS.chezod_ver / "per_entry.csv")
    doc = REPO / "docs" / "archive" / "260611" / "chezod-faithful-excursion.md"
    if doc.exists():
        t = doc.read_text()
        m1 = re.search(r"all residues ≤ 0\.1 \| \*\*(\d+)\*\*", t)
        m2 = re.search(r"all residues ≤ 0\.01 \| (\d+)", t)
        if m1 and m2:
            R.add(
                "nCheZODWithinTenthAll",
                m1.group(1),
                "entries within 0.1 Z of CheZOD at every residue (potenci-only)",
                rel(doc) + " (documented)",
                PREVIEW,
                "r3",
                "272",
            )
            R.add(
                "nCheZODWithinHundredthAll",
                m2.group(1),
                "entries within 0.01 Z at every residue",
                rel(doc) + " (documented)",
                PREVIEW,
                "r3",
                "186",
            )
    else:
        R.unrepro(
            "nCheZODWithinTenthAll",
            "272",
            "entries within 0.1 Z at every residue",
            "docs/archive/260611/chezod-faithful-excursion.md missing",
            "r3",
        )
    R.add(
        "nCheZODMaeBelowTenth",
        I((pe.mae < 0.1).sum()),
        "entries with per-entry mean absolute error < 0.1 (released-mode per_entry.csv)",
        rel(PATHS.chezod_ver / "per_entry.csv"),
        PREVIEW,
        "r3",
    )
    R.add(
        "listCheZODCarbonylSkip",
        S("15506, 15274, 15719"),
        "entries where CheZOD skipped carbonyl shifts (largest genuine discrepancies)",
        src + " genuine list",
        PREVIEW,
        "r3",
        None,
    )
    gen = {g["bmrb_id"] for g in rs["genuine"]}
    assert {"15506", "15274", "15719"} <= gen
    # DisProt (stale)
    srcd = (
        "old Article.tex:217-251, Figures/ROC_curves.png (2023 DisProt release; stale)"
    )
    R.add(
        "accDisProtMean",
        "0.76",
        "mean accuracy against DisProt (stale)",
        srcd,
        PROVISIONAL,
        "r6",
        "0.76",
    )
    R.add(
        "aucDisProtMin",
        "0.777",
        "lowest DisProt ROC AUC across tiers (unfiltered, Z; stale)",
        srcd,
        PROVISIONAL,
        "r6",
        "0.777",
    )
    R.add(
        "aucDisProtMax",
        "0.793",
        "highest DisProt ROC AUC across tiers (tolerant, G; stale)",
        srcd,
        PROVISIONAL,
        "r6",
        "0.793",
    )
    R.add(
        "aucDisProtGainMax",
        "0.01",
        "largest AUC gain of G over Z (stale)",
        srcd,
        PROVISIONAL,
        "r6",
        "0.01",
    )
    R.add(
        "nDisProtCytCEntries",
        "34",
        "aligned cytochrome-c entries in the old DisProt figure (stale)",
        "old Article.tex:242 caption",
        PROVISIONAL,
        "r6",
        "34",
    )
    R.add(
        "nDisProtCytCMatches",
        "891",
        "DisProt matches in the old cytochrome-c figure label (stale)",
        "old Figures/25640_Cytochrome_C.png",
        PROVISIONAL,
        "r6",
        "891",
    )
    R.add(
        "releaseDisProtOld",
        S("2023_06"),
        "DisProt release named in the old draft (replace)",
        "old Article.tex:360, 364",
        PROVISIONAL,
        "r6",
        "2023_06",
    )


def sec_clusters_and_leakage(C):
    df = C.df
    # CheZOD1325 from allseqs1325.txt and residue count from allscores
    cz = []
    for line in open(PATHS.cz1325 / "allseqs1325.txt"):
        f = line.split()
        if len(f) >= 2:
            cz.append((f"cz{f[0]}_{len(cz)}", f[1]))
    n_cz_res = 0
    for line in open(PATHS.cz1325 / "allscores1325newest.txt"):
        line = line.strip()
        if line.startswith("["):
            vals = [float(x) for x in line.strip("[]").split(",") if x.strip()]
            n_cz_res += sum(1 for v in vals if v != 999)
    R.add(
        "nCheZODFullProteins",
        I(len(cz)),
        "proteins in CheZOD1325",
        rel(PATHS.cz1325 / "allseqs1325.txt") + "; Dass et al. 2020",
        STABLE,
        "abs",
        "1,325",
    )
    R.add(
        "nCheZODFullResidues",
        I(n_cz_res),
        "scored residues in CheZOD1325 (non-missing published Z)",
        rel(PATHS.cz1325 / "allscores1325newest.txt"),
        STABLE,
        "r1",
        "149,672",
    )
    n_sc = len(df)
    res_scored = int(sum(np.asarray(m_, bool).sum() for m_ in df["mask"]))
    R.add(
        "ratioChainsVsCheZOD",
        F(n_sc / len(cz), 1),
        "scored chains relative to CheZOD1325 proteins (append $\\times$ or '-fold')",
        "derived",
        STABLE,
        "abs",
        "12.7",
    )
    R.add(
        "ratioResiduesVsCheZOD",
        F(res_scored / n_cz_res, 1),
        "scored residues relative to CheZOD1325",
        "derived",
        STABLE,
        "abs",
        "10.4",
    )
    pl = {"cz": "1,316", "all": "8,486", "nb": "6,531"}
    nb_seqs = sorted(set(df.sequence[(df.is_bound == False) & (df.length >= 20)]))  # noqa: E712
    all_seqs = sorted(set(df.sequence))
    if have_mmseqs():
        cc = mmseqs_cluster_count("chezod1325", cz, C.refresh)
        ca = mmseqs_cluster_count(
            "trizod_all", [(f"s{i}", s) for i, s in enumerate(all_seqs)], C.refresh
        )
        cn = mmseqs_cluster_count(
            "trizod_nonbound20",
            [(f"s{i}", s) for i, s in enumerate(nb_seqs)],
            C.refresh,
        )
        srcc = (
            f"mmseqs {cc['mmseqs']} "
            + "{cmd}; cached in --work-dir (single-threaded: multi-threaded clustering varies by a few clusters between runs)"
        )
        R.add(
            "nClustersCheZODFull",
            I(cc["n_clusters"]),
            "CheZOD1325 sequence clusters at 50 % id / 80 % cov",
            srcc.format(cmd=cc["cmd"]),
            STABLE,
            "r1",
            pl["cz"],
        )
        R.add(
            "nSeqsTriZODNonBoundTwenty",
            I(cn["n_seqs"]),
            "distinct non-bound scored sequences >= 20 aa (like-for-like scope)",
            "Parquet (all scored chains, is_bound False, length >= 20); same set as "
            "final_dataset global_unique_sequences; STABLE: the unfiltered tier and the "
            "bound classifier are untouched by rebuild fixes (a)-(f)",
            STABLE,
            "r1",
            "9,482",
        )
        R.add(
            "nClustersTriZODAll",
            I(ca["n_clusters"]),
            "clusters over all distinct scored sequences",
            srcc.format(cmd=ca["cmd"]),
            STABLE,
            "r1",
            pl["all"],
        )
        R.add(
            "nClustersTriZODNonBound",
            I(cn["n_clusters"]),
            "clusters over non-bound sequences >= 20 aa (like-for-like)",
            srcc.format(cmd=cn["cmd"]),
            STABLE,
            "r1",
            pl["nb"],
        )
        nca, ncn, ncc = ca["n_clusters"], cn["n_clusters"], cc["n_clusters"]
        st_c = STABLE
    else:
        R.unrepro(
            "nClustersCheZODFull",
            "1{,}316",
            "CheZOD1325 clusters",
            "mmseqs not on PATH",
            "r1",
        )
        R.unrepro(
            "nClustersTriZODAll",
            "8{,}486",
            "clusters over all scored sequences",
            "mmseqs not on PATH",
            "r1",
        )
        R.unrepro(
            "nClustersTriZODNonBound",
            "6{,}531",
            "clusters like-for-like",
            "mmseqs not on PATH",
            "r1",
        )
        nca, ncn, ncc = 8486, 6531, 1316
        st_c = UNREPRO
    R.add(
        "ratioClustersAllVsCheZOD",
        F(nca / ncc, 1),
        "TriZOD clusters (all sequences) relative to CheZOD1325 clusters",
        "derived",
        st_c,
        "r1",
        "6.4",
    )
    R.add(
        "ratioClustersLikeForLikeVsCheZOD",
        F(ncn / ncc, 1),
        "like-for-like cluster ratio ('five times as many sequence clusters')",
        "derived",
        st_c,
        "abs",
        "5.0",
    )
    R.add(
        "ratioRepsVsCheZODClusters",
        F(len(C.reps["unfiltered"]) / ncc, 1),
        "unfiltered training representatives relative to CheZOD1325 clusters",
        "derived",
        PENDING,
        "r1",
        "4.4",
    )
    R.add(
        "ratioRepsVsCheZODClustersAfterFix",
        F((len(C.reps["unfiltered"]) - 173) / ncc, 1),
        "same after removing the ~173 representatives the test-coverage rule would drop (estimate)",
        "derived from \\nRepsRemovedTestCovUnfiltered",
        PENDING,
        "r1",
        "4.3",
    )
    # exact-substring containment (audit B4)
    reps = read_fasta(
        PATHS.bundle / "train" / "unfiltered" / "train_unfiltered_best.fasta"
    )
    pairs, chains = [], set()
    for nm, T in (("TriZOD", C.test_tz), ("CheZOD", C.test_cz)):
        for ti, ts in T.items():
            for ri, rs in reps.items():
                if ts in rs or rs in ts:
                    pairs.append((nm, ti, ri))
                    chains.add((nm, ti))
    srcs = "test FASTAs vs train_unfiltered_best.fasta, exact substring either way (audit report B4)"
    R.add(
        "nSubstringPairs",
        I(len(pairs)),
        "training-representative / test-chain pairs where one sequence is an exact substring of the other",
        srcs,
        PENDING,
        "r5",
        "28",
    )
    R.add(
        "nSubstringTestChains",
        I(len(chains)),
        "test chains involved in such exact containment",
        srcs,
        PENDING,
        "r5",
        "22",
    )
    R.add(
        "nSubstringTestChainsTriZOD",
        I(sum(1 for n, _ in chains if n == "TriZOD")),
        "of them TriZOD test chains",
        srcs,
        PENDING,
        "r5",
        "15",
    )
    R.add(
        "nSubstringTestChainsCheZOD",
        I(sum(1 for n, _ in chains if n == "CheZOD")),
        "of them CheZOD117 chains",
        srcs,
        PENDING,
        "r5",
        "7",
    )
    # containment search (user_first A2 definitions)
    pl2 = {
        "bidir": {
            "unfiltered": ("6", "1"),
            "tolerant": ("6", "1"),
            "moderate": ("5", "1"),
            "strict": ("5", "0"),
        },
        "testcov": {
            "unfiltered": ("101", "16"),
            "tolerant": ("98", "15"),
            "moderate": ("77", "9"),
            "strict": ("51", "4"),
        },
        "fifty": {
            "unfiltered": ("79", "24"),
            "tolerant": ("77", "22"),
            "moderate": ("60", "18"),
            "strict": ("39", "7"),
        },
        "ninety": {
            "unfiltered": ("48", "18"),
            "tolerant": ("47", "16"),
            "moderate": ("37", "13"),
            "strict": ("24", "6"),
        },
        "rm_test": {
            "unfiltered": "173",
            "tolerant": "167",
            "moderate": "133",
            "strict": "66",
        },
        "rm_either": {
            "unfiltered": "325",
            "tolerant": "312",
            "moderate": "264",
            "strict": "135",
        },
    }
    defs = {
        "bidir": (
            "Bidir",
            "identity >= 30 % over >= 80 % of BOTH sequences (the pipeline's stage-2 definition)",
        ),
        "testcov": ("TestCov", "identity >= 30 % over >= 80 % of the TEST chain"),
        "fifty": ("FiftyHalf", "identity >= 50 % over >= 50 % of the test chain"),
        "ninety": ("NinetyHalf", "identity >= 90 % over >= 50 % of the test chain"),
    }
    if have_mmseqs():
        test_recs = [(k, v) for k, v in C.test_tz.items()] + [
            (k, v) for k, v in C.test_cz.items()
        ]
        hits = mmseqs_search(
            "containment",
            test_recs,
            list(reps.items()),
            ["-s", "7.5", "-e", "10", "--max-seqs", "1000"],
            C.refresh,
        )
        H = pd.DataFrame(
            hits, columns=["q", "t", "fid", "qcov", "tcov", "e", "aln", "ql", "tl"]
        )
        H = H[H.e < 1e-3]
        tt = {i: t_ for t_ in TIERS for i in C.reps[t_]}  # strictest rep tier
        H["trank"] = H.t.map(lambda x: RANK[tt[x]])
        H["set"] = np.where(H.q.isin(set(C.test_tz)), "TriZOD", "CheZOD")
        srcm = "mmseqs easy-search (-s 7.5, e < 1e-3) of the 575 test chains vs training representatives (port of user_first containment audit)"
        R.add(
            "nContainmentHits",
            I(len(H)),
            "test-vs-representative alignments with e < 1e-3",
            srcm,
            PENDING,
            "r5",
            "2,644",
        )
        crit = {
            "bidir": (H.fid >= 0.3) & (H.qcov >= 0.8) & (H.tcov >= 0.8),
            "testcov": (H.fid >= 0.3) & (H.qcov >= 0.8),
            "fifty": (H.fid >= 0.5) & (H.qcov >= 0.5),
            "ninety": (H.fid >= 0.9) & (H.qcov >= 0.5),
        }
        for key, (tag, d) in defs.items():
            for t_ in TIERS:
                h = H[crit[key] & (H.trank >= RANK[t_])]
                for j, nm in enumerate(("TriZOD", "CheZOD")):
                    R.add(
                        f"nLeak{tag}{nm}{TCAP[t_]}",
                        I(h[h.set == nm].q.nunique()),
                        f"{nm} test chains with a {t_} training representative at {d} (frozen split; must become 0)",
                        srcm,
                        PENDING,
                        "r5",
                        pl2[key][t_][j],
                    )
        either = (H.fid >= 0.3) & ((H.qcov >= 0.8) | (H.tcov >= 0.8))
        for t_ in TIERS:
            R.add(
                f"nRepsRemovedTestCov{TCAP[t_]}",
                I(H[crit["testcov"] & (H.trank >= RANK[t_])].t.nunique()),
                f"{t_} representatives a test-coverage 30/80 rule would remove (estimate, reachable hits only)",
                srcm,
                PENDING,
                "r5",
                pl2["rm_test"][t_],
            )
            R.add(
                f"nRepsRemovedEitherCov{TCAP[t_]}",
                I(H[either & (H.trank >= RANK[t_])].t.nunique()),
                f"{t_} representatives an either-coverage 30/80 rule would remove (estimate)",
                srcm,
                PENDING,
                "r5",
                pl2["rm_either"][t_],
            )
        # bidirectional miss pair
        pr = H[(H.q == "27176_1_1_1") & (H.t == "27177_1_1_1")]
        if len(pr):
            r = pr.iloc[0]
            R.add(
                "fidentMissPair",
                P(100 * r.fid),
                "identity of the bidirectional-rule miss pair 27177_1_1_1 (train, 124 aa) / 27176_1_1_1 (test, 99 aa)",
                srcm,
                PENDING,
                "r5",
                "79.8",
            )
            R.add(
                "lenRatioMissPair",
                F(r.ql / r.tl, 3),
                "length ratio of that pair (test / train)",
                srcm,
                PENDING,
                "r5",
                "0.798",
            )
        # verified containments
        for q, t, tag, pv in (
            ("51709_1_1_1", "53339_1_1_1", "One", ("169", "419")),
            ("19221_1_1_1", "19219_1_1_1", "Two", ("176", "401")),
            ("15813_1_1_1", "30456_1_1_1", "Three", ("117", "443")),
        ):
            r = H[(H.q == q) & (H.t == t)]
            if len(r):
                r = r.iloc[0]
                R.add(
                    f"idContainTest{tag}",
                    S(q),
                    f"verified containment {tag}: test chain",
                    srcm,
                    PENDING,
                    "r5",
                )
                R.add(
                    f"idContainRep{tag}",
                    S(t),
                    f"verified containment {tag}: training representative ({tt[t]})",
                    srcm,
                    PENDING,
                    "r5",
                )
                R.add(
                    f"lenContainTest{tag}",
                    I(r.ql),
                    f"verified containment {tag}: test chain length (aa)",
                    srcm,
                    PENDING,
                    "r5",
                    pv[0],
                )
                R.add(
                    f"lenContainRep{tag}",
                    I(r.tl),
                    f"verified containment {tag}: representative length (aa)",
                    srcm,
                    PENDING,
                    "r5",
                    pv[1],
                )
                R.add(
                    f"pctIdContain{tag}",
                    P(100 * r.fid, 0),
                    f"verified containment {tag}: identity",
                    srcm,
                    PENDING,
                    "r5",
                    "100",
                )
        # --- the same counts with the build's COMMON alignment block (Fig. 3c1) -------------
        # fig3c_split.py searches with --alignment-mode 3 --mask 0 --comp-bias-corr 0 (the block
        # the corrected stage-2 search will use), one search per tier; the counts above use mmseqs
        # defaults (masking, local alignment) and find fewer chains in the region.
        f3c = paper("fig3c_split")
        args_c = list(f3c.MMSEQS_ARGS)
        srcc3 = (
            "mmseqs easy-search " + " ".join(args_c) + " of the 575 test chains vs "
            "train/<tier>/train_<tier>_best.fasta, one search per tier (as scripts/figures/paper/fig3c_split.py)"
        )
        HC = []
        for t_ in TIERS:
            tgt = read_fasta(PATHS.bundle / "train" / t_ / f"train_{t_}_best.fasta")
            for h in mmseqs_search(
                f"leak_common_{t_}", test_recs, list(tgt.items()), args_c, C.refresh
            ):
                HC.append([*h, t_])
        HC = pd.DataFrame(
            HC,
            columns=["q", "t", "fid", "qcov", "tcov", "e", "aln", "ql", "tl", "tier"],
        )
        HC["set"] = np.where(HC.q.isin(set(C.test_tz)), "TriZOD", "CheZOD")
        crit_c = {
            "TestCov": (HC.fid >= 0.3) & (HC.qcov >= 0.8),
            "Bidir": (HC.fid >= 0.3) & (HC.qcov >= 0.8) & (HC.tcov >= 0.8),
            "FiftyHalf": (HC.fid >= 0.5) & (HC.qcov >= 0.5),
            "NinetyHalf": (HC.fid >= 0.9) & (HC.qcov >= 0.5),
        }
        dcrit = {
            "TestCov": "identity >= 30 % over >= 80 % of the TEST chain",
            "Bidir": "identity >= 30 % over >= 80 % of BOTH sequences",
            "FiftyHalf": "identity >= 50 % over >= 50 % of the test chain",
            "NinetyHalf": "identity >= 90 % over >= 50 % of the test chain",
        }
        pl3 = {"TestCov": {"TriZOD": (123, 120, 99, 64), "CheZOD": (22, 21, 16, 10)},
               "Bidir": {"TriZOD": (11, 11, 10, 8), "CheZOD": (1, 1, 1, 0)}}  # fmt: skip
        for tag, cm in crit_c.items():
            for j, t_ in enumerate(TIERS):
                own = HC[cm & (HC.tier == t_)]
                for nm in ("TriZOD", "CheZOD"):
                    pv = pl3.get(tag, {}).get(nm)
                    R.add(
                        f"nLeak{tag}Common{nm}{TCAP[t_]}",
                        I(own[own.set == nm].q.nunique()),
                        f"{nm} test chains with a {t_} training representative at {dcrit[tag]}, "
                        "build COMMON alignment block (Fig. 3c1; frozen split, must become 0)",
                        srcc3,
                        PENDING,
                        "r5",
                        None if pv is None else str(pv[j]),
                    )
        for j, t_ in enumerate(TIERS):
            own = HC[crit_c["TestCov"] & (HC.tier == t_)]
            R.add(
                f"nRepsRemovedTestCovCommon{TCAP[t_]}",
                I(own.t.nunique()),
                f"{t_} representatives the test-coverage 30/80 rule would remove, COMMON block (Fig. 3c1 estimate)",
                srcc3,
                PENDING,
                "r5",
                str((236, 230, 189, 90)[j]),
            )
        hit_q = set(HC.q)
        for nm, T in (("TriZOD", C.test_tz), ("CheZOD", C.test_cz)):
            R.add(
                f"nTestNoHitCommon{nm}",
                I(sum(1 for q in T if q not in hit_q)),
                f"{nm} test chains with no alignment (identity >= 20 %, e < 1e-3) to any training representative (not drawn in Fig. 3c1)",
                srcc3,
                PENDING,
                "r5",
            )
        R.add(
            "nTestNoHitCommon",
            I(sum(1 for q in list(C.test_tz) + list(C.test_cz) if q not in hit_q)),
            "held-out chains (both sets) with no alignment to any training representative",
            srcc3,
            PENDING,
            "r5",
            "245",
        )
        # --- representatives with a near-identical partner in another cluster (audit 6.5) -----
        reps_unf = list(reps.items())
        hs = mmseqs_search("reps_self_common", reps_unf, reps_unf, args_c, C.refresh)
        HS = pd.DataFrame(
            hs, columns=["q", "t", "fid", "qcov", "tcov", "e", "aln", "ql", "tl"]
        )
        HS = HS[HS.q != HS.t]
        cov_short = np.where(HS.ql <= HS.tl, HS.qcov, HS.tcov)
        pr = HS[(HS.fid >= 0.9) & (cov_short >= 0.8)]
        n90 = len(set(pr.q) | set(pr.t))
        srcs3 = (
            "mmseqs easy-search "
            + " ".join(args_c)
            + " of train_unfiltered_best.fasta against itself; "
            "partner = another representative (hence another 50/80 cluster)"
        )
        R.add(
            "nRepsNinetyShorterPartner",
            I(n90),
            "unfiltered training representatives with another representative at >= 90 % identity over >= 80 % of the SHORTER sequence (cluster grouping does not separate them)",
            srcs3,
            PENDING,
            "r5",
            "835",
        )
        R.add(
            "pctRepsNinetyShorterPartner",
            P(100 * n90 / len(reps)),
            "share of unfiltered representatives with such a partner",
            srcs3,
            PENDING,
            "r5",
        )
        # --- TriZOD test chains vs CheZOD1325 (CheZOD-disjointness holds per cluster) ----------
        hz = mmseqs_search(
            "test_vs_chezod1325", list(C.test_tz.items()), cz, args_c, C.refresh
        )
        HZ = pd.DataFrame(
            hz, columns=["q", "t", "fid", "qcov", "tcov", "e", "aln", "ql", "tl"]
        )
        srcz = (
            "mmseqs easy-search "
            + " ".join(args_c)
            + " of TriZOD_test_set.fasta against "
            + rel(PATHS.cz1325 / "allseqs1325.txt")
        )
        both30 = HZ[(HZ.fid >= 0.3) & (HZ.qcov >= 0.8) & (HZ.tcov >= 0.8)]
        one30 = HZ[(HZ.fid >= 0.3) & ((HZ.qcov >= 0.8) | (HZ.tcov >= 0.8))]
        R.add(
            "nTestTriZODCheZODFullHomolog",
            I(both30.q.nunique()),
            "TriZOD test chains with a CheZOD1325 sequence at >= 30 % identity over >= 80 % of both sequences",
            srcz,
            STABLE,
            "r5",
            "18",
        )
        R.add(
            "nTestTriZODCheZODFullHomologOneSided",
            I(one30.q.nunique()),
            "TriZOD test chains with a CheZOD1325 sequence at >= 30 % identity over >= 80 % of either sequence",
            srcz,
            STABLE,
            "r5",
            "85",
        )
        czs = [x for _, x in cz]
        nsub = sum(
            1 for ts in C.test_tz.values() if any(ts in c_ or c_ in ts for c_ in czs)
        )
        R.add(
            "nTestTriZODCheZODFullSubstring",
            I(nsub),
            "TriZOD test chains that are an exact substring of a CheZOD1325 sequence or contain one",
            "test FASTA vs " + rel(PATHS.cz1325 / "allseqs1325.txt"),
            STABLE,
            "r5",
            "2",
        )
    else:
        for key, (tag, d) in defs.items():
            for t_ in TIERS:
                for j, nm in enumerate(("TriZOD", "CheZOD")):
                    R.unrepro(
                        f"nLeak{tag}{nm}{TCAP[t_]}",
                        pl2[key][t_][j],
                        f"{nm} test chains with a {t_} representative at {d}",
                        "mmseqs not on PATH",
                        "r5",
                    )


def sec_baselines(C):
    df = C.df
    AA = "ACDEFGHIKLMNPQRSTVWY"
    lut = np.full(256, 20, dtype=np.int64)
    for i, a in enumerate(AA):
        lut[ord(a)] = i
    W = 7
    D = (2 * W + 1) * 21 + 1

    def feats(seq):
        a = lut[np.frombuffer(seq.encode(), dtype=np.uint8)]
        L = len(a)
        X = np.zeros((L, D), dtype=np.float64)
        X[:, -1] = 1.0
        for o in range(-W, W + 1):
            i0, i1 = max(0, -o), min(L, L - o)
            if i1 <= i0:
                continue
            rows = np.arange(i0, i1)
            X[rows, (o + W) * 21 + a[rows + o]] = 1.0
        return X, a

    def accumulate(sub):
        XtX = np.zeros((D, D))
        Xty = np.zeros(D)
        ysum = np.zeros(21)
        ycnt = np.zeros(21)
        for s, gs in zip(sub.sequence, sub.gscores):
            g = np.asarray(gs, float)
            m = ~np.isnan(g)
            X, a = feats(s)
            X, y, aa = X[m], g[m], a[m]
            XtX += X.T @ X
            Xty += X.T @ y
            np.add.at(ysum, aa, y)
            np.add.at(ycnt, aa, 1)
        return XtX, Xty, ysum, ycnt

    def stack(sub):
        Xs, ys, aas, ch = [], [], [], []
        for i, (s, gs) in enumerate(zip(sub.sequence, sub.gscores)):
            g = np.asarray(gs, float)
            m = ~np.isnan(g)
            X, a = feats(s)
            Xs.append(X[m])
            ys.append(g[m])
            aas.append(a[m])
            ch.append(np.full(int(m.sum()), i))
        return (
            np.vstack(Xs),
            np.concatenate(ys),
            np.concatenate(aas),
            np.concatenate(ch),
        )

    def per_chain(y, p, ch):
        """Supplementary Note N2 protocol: per chain, Spearman rho where G takes more than one
        value and AUC at G > 0.65 where both classes occur; medians over those chains."""
        from scipy.stats import rankdata, spearmanr

        rho, au = [], []
        for c in np.unique(ch):
            m = ch == c
            yy, pp = y[m], p[m]
            if np.unique(yy).size > 1:
                rho.append(spearmanr(yy, pp).statistic)
            pos = yy > G_DIS
            if 0 < pos.sum() < pos.size:
                r_ = rankdata(pp)
                n1, n0 = pos.sum(), (~pos).sum()
                au.append((r_[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        rho, au = np.array(rho, float), np.array(au, float)
        return (
            float(np.nanmedian(rho)),
            int(np.isfinite(rho).sum()),
            float(np.nanmedian(au)),
            int(np.isfinite(au).sum()),
        )

    def spearman(x, y):
        return np.corrcoef(
            pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
        )[0, 1]

    def auc(y, p):
        pos = y > G_DIS
        r = pd.Series(p).rank().to_numpy()
        return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (
            pos.sum() * (~pos).sum()
        )

    tests = {
        "TriZOD": stack(df[df.split == "test_trizod"]),
        "CheZOD": stack(df[df.split == "test_chezod117"]),
    }
    out = {}
    for t_ in TIERS:
        tr = df[
            df.train_tier.map(
                lambda v, t0=t_: (
                    v is not None and v == v and RANK.get(v, -1) >= RANK[t0]
                )
            )
        ]
        XtX, Xty, ysum, ycnt = accumulate(tr)
        ymean = Xty[-1] / XtX[-1, -1]
        aa_mean = np.where(ycnt > 0, ysum / np.maximum(ycnt, 1), ymean)
        w = np.linalg.solve(XtX + 10.0 * np.eye(D), Xty)
        for n, (Xt, yt, aat, cht) in tests.items():
            p1, p2 = aa_mean[aat], Xt @ w
            crho, n_crho, cauc, n_cauc = per_chain(yt, p2, cht)
            out[(t_, n)] = {
                "aa_r": np.corrcoef(yt, p1)[0, 1],
                "aa_auc": auc(yt, p1),
                "r": np.corrcoef(yt, p2)[0, 1],
                "rho": spearman(yt, p2),
                "auc": auc(yt, p2),
                "chain_rho": crho,
                "chain_auc": cauc,
                "n_chain_rho": n_crho,
                "n_chain_auc": n_cauc,
            }
    src = "Parquet (unfiltered labels); per-amino-acid mean and one-hot 15-residue window ridge (lambda = 10) trained on training representatives of each cumulative tier (port of user_first/ceiling_and_baseline.py part 2)"
    for n in ("TriZOD", "CheZOD"):
        vals = [out[(t_, n)] for t_ in TIERS]
        pl = {"TriZOD": {"aa": ("0.144", "0.144"), "r": ("0.457", "0.462"), "rho": ("0.395", "0.406"), "auc": ("0.833", "0.836")},
              "CheZOD": {"aa": ("0.159", "0.164"), "r": ("0.478", "0.495"), "rho": ("0.470", "0.486"), "auc": ("0.724", "0.733")}}[n]  # fmt: skip
        for key, lab in (
            ("aa_r", "AaMeanR"),
            ("r", "RidgeR"),
            ("rho", "RidgeRho"),
            ("auc", "RidgeAuc"),
            ("aa_auc", "AaMeanAuc"),
        ):
            v = [x[key] for x in vals]
            p_ = pl.get(
                {"aa_r": "aa", "r": "r", "rho": "rho", "auc": "auc"}.get(key, ""),
                (None, None),
            )
            R.add(
                f"base{lab}Min{n}",
                F(min(v), 3),
                f"{n} test set: lowest {lab} across the four training tiers",
                src,
                PREVIEW,
                "r5",
                p_[0],
            )
            R.add(
                f"base{lab}Max{n}",
                F(max(v), 3),
                f"{n} test set: highest {lab} across the four training tiers",
                src,
                PREVIEW,
                "r5",
                p_[1],
            )
        srcpc = (
            src
            + "; per-chain metrics by the Supplementary Note N2 protocol (median over chains)"
        )
        for key, lab in (
            ("chain_rho", "RidgeChainRho"),
            ("chain_auc", "RidgeChainAuc"),
        ):
            v = [x[key] for x in vals]
            R.add(f"base{lab}Min{n}", F(min(v), 3), f"{n} test set: lowest per-chain median {key[6:]} of the ridge baseline across the four training tiers", srcpc, PREVIEW, "r5")  # fmt: skip
            R.add(f"base{lab}Max{n}", F(max(v), 3), f"{n} test set: highest per-chain median {key[6:]} of the ridge baseline across the four training tiers", srcpc, PREVIEW, "r5")  # fmt: skip
        R.add(
            f"nBaseChainsRho{n}",
            I(vals[0]["n_chain_rho"]),
            f"{n} test chains with a defined per-chain Spearman rho (G takes more than one value)",
            srcpc,
            STABLE,
            "r5",
        )
        R.add(
            f"nBaseChainsAuc{n}",
            I(vals[0]["n_chain_auc"]),
            f"{n} test chains with a defined per-chain AUC (both G > 0.65 and G <= 0.65 residues)",
            srcpc,
            STABLE,
            "r5",
        )
        for t_ in TIERS:
            R.add(
                f"baseRidgeR{n}{TCAP[t_]}",
                F(out[(t_, n)]["r"], 3),
                f"{n} test set: ridge Pearson r, trained on {t_} representatives",
                src,
                PREVIEW,
                "r5",
            )
    R.add(
        "baseRidgeLambda",
        "10",
        "ridge penalty of the one-hot baseline",
        src,
        STABLE,
        "par",
        "10",
    )
    R.add(
        "baseWindow",
        "15",
        "window length (residues) of the one-hot baseline",
        src,
        STABLE,
        "par",
        "15",
    )


def sec_params(C):
    from trizod.constants import REFINED_WEIGHTS as W

    R.add(
        "thrDisorderG",
        "0.65",
        "binary disorder cut on G used throughout",
        "manuscript convention (old Article.tex:219, 303, 360)",
        STABLE,
        "par",
        "0.65",
    )
    R.add(
        "thrValErr",
        "1.3",
        "deposited Val_err above which the scorer drops a shift (ppm)",
        "trizod/bmrb/bmrb.py get_valid_bbshifts(max_err=1.3)",
        STABLE,
        "par",
        "1.3",
    )
    txt = (REPO / "trizod" / "lacs" / "lacs.py").read_text()
    m = re.search(r"_MIN_RESIDUES = (\d+)", txt)
    R.add(
        "lacsMinResidues",
        m.group(1),
        "minimum CA observations for a LACS fit",
        "trizod/lacs/lacs.py _MIN_RESIDUES",
        STABLE,
        "par",
        "20",
    )
    stxt = (REPO / "trizod" / "scoring" / "scoring.py").read_text()
    aics = {float(x) for x in re.findall(r"min_AIC=([\d.]+)", stxt)} - {999.0}
    assert len(aics) == 1, aics
    R.add(
        "aicThreshold",
        I(aics.pop()),
        "AIC improvement threshold for accepting a POTENCI/AIC offset",
        "trizod/scoring/scoring.py min_AIC=6.0",
        STABLE,
        "par",
        "6",
    )
    m = re.search(r"atom_counts < (\d+)", stxt)
    R.add(
        "aicMinAtoms",
        m.group(1),
        "minimum residues per atom type for an offset estimate",
        "trizod/scoring/scoring.py (atom_counts < 4)",
        STABLE,
        "par",
        "4",
    )
    R.add(
        "offsetWindow",
        "9",
        "residues in the rolling-window offset correction",
        "trizod/scoring/scoring.py compute_running_offsets",
        STABLE,
        "par",
        "9",
    )
    # E[G*] = (1 + 1/(4k))^(-k/2): nearly independent of k
    eg = [(1 + 1 / (4 * k)) ** (-k / 2) for k in range(1, 22)]
    R.add(
        "expGStarKOne",
        F(eg[0], 3),
        "expected G* at k = 1, (1 + 1/(4k))^(-k/2)",
        "closed form; matches trizod/scoring/scoring.py",
        STABLE,
        "par",
        "0.894",
    )
    R.add(
        "expGStarKMax",
        F(eg[-1], 3),
        "expected G* at k = 21",
        "closed form",
        STABLE,
        "par",
        "0.883",
    )
    R.add(
        "kMaxShifts",
        "21",
        "maximum per-residue shift count k (7 atom types x 3 positions)",
        "scoring window definition",
        STABLE,
        "par",
        "21",
    )
    for a in ATOMS:
        A = a.capitalize() if len(a) == 1 else a[0] + a[1:].lower()
        R.add(
            f"weightPotenci{A}",
            F(W[a], 4 if W[a] < 0.1 else 4),
            f"POTENCI RMSD weight (sigma unit) for {a} (ppm; inherited CheZOD constants)",
            "trizod/constants.py REFINED_WEIGHTS",
            STABLE,
            "par",
        )
    R.add(
        "mmseqsLeakIdentity",
        P(30, 0),
        "identity threshold of the leakage search",
        "trizod/dataset/redundancy.py",
        STABLE,
        "par",
        "30",
    )
    R.add(
        "mmseqsLeakCoverage",
        P(80, 0),
        "coverage threshold of the leakage search",
        "trizod/dataset/redundancy.py",
        STABLE,
        "par",
        "80",
    )
    R.add(
        "mmseqsClusterIdentity",
        P(50, 0),
        "identity threshold of the representative clustering",
        "trizod/dataset/redundancy.py",
        STABLE,
        "par",
        "50",
    )
    R.add(
        "mmseqsClusterCoverage",
        P(80, 0),
        "coverage threshold of the representative clustering",
        "trizod/dataset/redundancy.py",
        STABLE,
        "par",
        "80",
    )
    ver = (
        subprocess.run(
            ["mmseqs", "version"], capture_output=True, text=True
        ).stdout.strip()
        if have_mmseqs()
        else "18-8cc5c"
    )
    R.add(
        "mmseqsVersion",
        S(ver),
        "MMseqs2 version used by the build and by this script",
        f"{rel(PATHS.log)}; mmseqs version",
        STABLE,
        "par",
        "18-8cc5c",
    )
    R.add(
        "minChainLength",
        "20",
        "minimum chain length for the training/test assembly (aa)",
        rel(PATHS.fds) + " min_seq_len",
        STABLE,
        "par",
        "20",
    )
    R.add(
        "pipelineVersion",
        S(C.manifest["pipeline_version"]),
        "pipeline version of the frozen build",
        "MANIFEST.json",
        STABLE,
        "par",
        "trizod-5b8147a",
    )
    R.add(
        "releaseVersion",
        S(C.manifest["version"]),
        "release label of the frozen build",
        "MANIFEST.json",
        STABLE,
        "par",
        "2026-10",
    )
    m = re.search(
        r"Written in row groups of (\d+) rows", C.pq_meta.get("sort_order", "")
    )
    R.add(
        "parquetRowGroup",
        m.group(1) if m else "500",
        "rows per Parquet row group",
        "trizod_dataset.parquet metadata sort_order",
        STABLE,
        "par",
        "500",
    )
    R.add(
        "repPairsCapKTest",
        "30",
        "maximum replicate pairs per sequence, within-residue k test",
        "this script (kpaired port)",
        STABLE,
        "par",
        "30",
    )
    R.add(
        "repPairsCapCeiling",
        "20",
        "maximum replicate pairs per sequence, residue-level reliability",
        "this script (ceiling port)",
        STABLE,
        "par",
        "20",
    )
    R.add(
        "repPairsCapCondition",
        "50",
        "maximum replicate pairs per sequence, condition contrast",
        "this script (teaser E port)",
        STABLE,
        "par",
        "50",
    )
    R.add(
        "conceptDOI",
        S("10.5281/zenodo.21309963"),
        "Zenodo concept DOI of the TriZOD dataset",
        "Zenodo / DataCite",
        STABLE,
        "par",
        "10.5281/zenodo.21309963",
    )
    R.add(
        "thrDisorderZ",
        "8",
        "binary disorder cut on the CheZOD Z-score (Z < 8 disordered; ODiNPred/SETH convention), drawn for orientation only",
        "convention of the CheZOD-based predictors; dotted line in Fig. 1a (scripts/figures/paper/fig1a_ubiquitin.py)",
        STABLE,
        "par",
        "8",
    )
    # substituted sample conditions (tiers with default-conditions = True)
    import inspect

    from trizod.bmrb.bmrb import SampleConditions

    for name, fn, d_ in (
        ("defaultPH", "get_pH", "pH substituted when none is deposited (unfiltered, tolerant, moderate)"),
        ("defaultTemperatureK", "get_temperature", "temperature (K) substituted when none is deposited"),
        ("defaultIonicStrengthM", "get_ionic_strength", "ionic strength (M) substituted when none is deposited"),
    ):  # fmt: skip
        # the defaults live in the getters or their _missing_* helpers, so the
        # whole class is searched (the helpers moved in the I-02/I-03 parser fix)
        src_fn = inspect.getsource(SampleConditions)
        val = {
            "get_pH": "7.0",
            "get_temperature": "298.0",
            "get_ionic_strength": "0.1",
        }[fn]
        assert f"return {val}" in src_fn, (fn, "default changed in trizod/bmrb/bmrb.py")
        R.add(
            name,
            val if fn != "get_temperature" else "298",
            d_,
            f"trizod/bmrb/bmrb.py SampleConditions.{fn} (return_default)",
            STABLE,
            "par",
        )
    s4 = paper("supp_s4_conditions")
    R.add(
        "pressureHighBar",
        F(s4.HIGH_P, 0),
        "sample pressure (bar) at or above which a sample counts as high-pressure in the analyses (not a filter yet; fix e sets the ceiling)",
        "writing plan Appendix A12; scripts/figures/paper/supp_s4_conditions.py HIGH_P",
        STABLE,
        "par",
        "400",
    )
    # LACS IRLS iteration caps (issue #22)
    from trizod.lacs import lacs as _lacs

    _sig = inspect.signature(_lacs._robustfit)
    R.add(
        "lacsIterCap",
        str(_sig.parameters["max_iter"].default),
        "iteration cap of the LACS robust fit (coefficient-convergence criterion, issue #22)",
        "trizod/lacs/lacs.py _robustfit(max_iter=...)",
        STABLE,
        "par",
        "200",
    )
    R.add(
        "lacsIterCapOld",
        "50",
        "iteration cap of the LACS robust fit before issue #22 (weight-convergence criterion)",
        "commit 376aad8 (git show 376aad8)",
        STABLE,
        "par",
        "50",
    )
    for name, val, d_ in (
        ("nLacsFitsAtOldCap", "157", "LACS fits that exhausted the old 50-iteration cap"),
        ("nLacsFitsOldCapRun", "23{,}448", "LACS fits in the run that measured the old cap"),
        ("nLacsOldCapRunChains", "2{,}873", "chains in that scoring run (--rereference-mode both)"),
        ("pctLacsFitsAtOldCap", "0.67\\%", "share of LACS fits that exhausted the old cap"),
        ("nLacsOldCapChangedRecords", "4", "records whose offset changed when the fix was applied to that run"),
    ):  # fmt: skip
        R.unrepro(
            name,
            val,
            d_ + " (issue #22)",
            "measured once on a 2,873-chain run with the pre-fix code; recomputing needs the old fit on every chain",
            "par",
            plan_source="commit 376aad8 message",
        )
    # naive side-chain far-outlier rule, PR #24 review (why side chains are not assessed)
    for name, val, d_ in (
        ("pctNaiveFlagsParamagnetic", "20.5\\%", "share of a naive k = 10 robust-z flag list (incl. side chains) on paramagnetic proteins"),
        ("pctNaiveFlagsArgAliased", "9.6\\%", "share of that flag list that are spectrally folded ARG NE shifts"),
        ("pctNaiveFlagsPhysics", "30.1\\%", "share of that flag list that reflects real physics (paramagnetic + ARG NE aliasing)"),
    ):  # fmt: skip
        R.unrepro(
            name,
            val,
            d_,
            "verifier measurement in the PR #24 review on the pre-2026-10 corpus; the naive rule is not shipped",
            "par",
            plan_source="docs/plans/2026-08-09-kuppel-review-response.md:546",
        )
    # release lineage: the 2024 figshare test set (local copy)
    fs = PATHS.root / "data" / "external" / "chezod117" / "TriZOD_test_set.fasta"
    if fs.exists():
        R.add(
            "nTestTriZODFigshare",
            I(len(read_fasta(fs))),
            "TriZOD test chains in the 2024 figshare release (local copy; verify against figshare 10.6084/m9.figshare.25792035)",
            rel(fs)
            + " (directory holds the 2024 release files: <tier>.json, *_rest_set.fasta, test sets)",
            STABLE,
            "par",
            "348",
        )


def sec_literature(C):
    lit = [
        (
            "nCheZODElevenSeventyFour",
            "1{,}174",
            "proteins in CheZOD1174 (Ilzhöfer et al. 2022)",
            "Ilzhöfer et al. 2022, doi:10.3389/fbinf.2022.1019597",
            "1,174",
        ),
        (
            "nResCheZODElevenSeventyFour",
            "132{,}545",
            "residues in CheZOD1174",
            "Ilzhöfer et al. 2022",
            "132,545",
        ),
        (
            "nCheZODPotenciDB",
            "178",
            "POTENCI-database entries seeding CheZOD1325",
            "Dass et al. 2020, Methods",
            "178",
        ),
        (
            "cheZODMinShiftsLow",
            "2.0",
            "CheZOD1325 minimum average shifts per residue, first stage",
            "Dass et al. 2020, Methods",
            "2.0",
        ),
        (
            "cheZODMinShiftsHigh",
            "6.0",
            "CheZOD1325 minimum average shifts per residue, last stage",
            "Dass et al. 2020, Methods",
            "6.0",
        ),
        (
            "pctCheZODIdentityCut",
            "50\\%",
            "CheZOD1325 redundancy cut (identity)",
            "Dass et al. 2020, Methods",
            "50",
        ),
        (
            "nAdoptSequences",
            "1{,}168",
            "ADOPT training sequences (< 50 % identity; different redundancy definition)",
            "Redl et al. 2023, doi:10.1093/nargab/lqad041",
            "1,168",
        ),
        (
            "pctVascoFlagged",
            "40\\%",
            "share of BMRB entries VASCO flags for referencing problems (approximate)",
            "Rieping & Vranken 2010, doi:10.1002/prot.22756 (full text not re-checked)",
            "40",
        ),
        ("nPEDEntries", "461", "PED entries", "positioning note Q1", "461"),
        ("nAtlasChains", "1{,}390", "ATLAS chains", "positioning note Q1", "1,390"),
        ("nMdCATHDomains", "5{,}398", "mdCATH domains", "positioning note Q1", "5,398"),
        (
            "nMBMRBProteins",
            "9{,}381",
            "mBMRB proteins (Dyna-1; exchange labels, a larger shift-derived set of a different label type)",
            "Dyna-1, Nature 2026",
            "9,381",
        ),
        (
            "pctAlphaFoldConditionallyFolded",
            "15\\%",
            "share of human IDRs AF2 predicts confidently in a conditionally folded state (approximate)",
            "Alderson et al. 2023",
            "15",
        ),
    ]
    for name, v, d, s, p in lit:
        R.add(name, v, d, s, STABLE, "lit", p)
    R.unrepro(
        "nInheritedUnparsableEntries",
        "1{,}700",
        "'>1,700 entries CheZOD cannot read' (inherited 2023 claim) -- DO NOT USE unless sourced",
        "no derivation exists (writing plan A11: UNVERIFIED)",
        "lit",
    )
    R.unrepro(
        "nInheritedInvalidEntries",
        "11",
        "'11 invalid' entries (inherited) -- DO NOT USE unless sourced",
        "no derivation (A11 UNVERIFIED)",
        "lit",
    )
    R.unrepro(
        "nInheritedErroredEntries",
        "5",
        "'5 errored' entries (inherited) -- DO NOT USE unless sourced",
        "no derivation (A11 UNVERIFIED)",
        "lit",
    )
    R.unrepro(
        "nDisProtIDRsClaim",
        "10{,}000",
        "DisProt '10,000 IDRs' (old A:87) -- check full text before use",
        "not in the cited abstract (A11 UNVERIFIED)",
        "lit",
    )
    R.unrepro(
        "nDisProtProteinsClaim",
        "2{,}600",
        "DisProt '2,600 proteins' (old A:87) -- check full text",
        "not in the cited abstract (A11 UNVERIFIED)",
        "lit",
    )
    R.unrepro(
        "nBMRBExperimentsClaim",
        "15{,}000",
        "BMRB '>15,000 experiments' (old A:89) -- check before use",
        "not in the cited abstract (A11 UNVERIFIED)",
        "lit",
    )
    R.unrepro(
        "pctIRLSNonConvergence",
        "0.85\\%",
        "entries where the LACS IRLS fit does not converge (issue #22) -- DO NOT USE: unsourced; use the commit-sourced \\pctLacsFitsAtOldCap",
        "GitHub issue #22 / positioning note; not recomputable from the frozen data; contradicted by commit 376aad8 (157 of 23,448 fits, 0.67 %)",
        "r6",
        plan_source="writing plan Discussion 6",
    )


# ----------------------------------------------------------------------------------
# writers
# ----------------------------------------------------------------------------------
MARKERS = r"""% ---------------------------------------------------------------------------
% Status markers (xcolor must be loaded; the zHenriquesLab class loads it).
%   \pending{...}      orange + superscript "rebuild": changes with the planned rebuild
%   \provisional{...}  purple: stale DisProt numbers
%   \placeholder{...}  grey framed box: co-author content that does not exist yet
%   \ownerdraft{RA}{...} blue, prefixed "[RA draft]": owner-drafted text
%   \authornote{...}   red bracketed note: a decision the authors must make
%   \needsnumber{...}  bold red "[NUMBER: ...]": a number this file does not define
% \draftmarkersoff renders all of them as plain text (\draftmarkerson restores).
% ---------------------------------------------------------------------------
\newif\ifdraftmarkers
\draftmarkerstrue
\providecommand{\draftmarkersoff}{\global\draftmarkersfalse}
\providecommand{\draftmarkerson}{\global\draftmarkerstrue}
\DeclareRobustCommand{\pending}[1]{%
  \ifdraftmarkers{\color{orange!85!black}#1\textsuperscript{\tiny\,rebuild}}\else#1\fi}
\DeclareRobustCommand{\provisional}[1]{%
  \ifdraftmarkers{\color{violet}#1}\else#1\fi}
\DeclareRobustCommand{\placeholder}[1]{%
  \ifdraftmarkers
    \ifvmode
      \par\noindent\fcolorbox{gray}{gray!8}{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}{\color{gray!60!black}\textsf{PLACEHOLDER:} #1}}\par
    \else
      \fcolorbox{gray}{gray!8}{\color{gray!60!black}\textsf{PLACEHOLDER:} #1}%
    \fi
  \else#1\fi}
\DeclareRobustCommand{\ownerdraft}[2]{%
  \ifdraftmarkers{\color{blue!75!black}[#1 draft] #2}\else#2\fi}
\DeclareRobustCommand{\authornote}[1]{%
  \ifdraftmarkers{\color{red!80!black}[#1]}\else[#1]\fi}
\DeclareRobustCommand{\needsnumber}[1]{%
  \ifdraftmarkers{\color{red}\textbf{[NUMBER: #1]}}\else[NUMBER: #1]\fi}
% Plain-text versions for PDF bookmarks when hyperref is loaded.
\AtBeginDocument{\ifdefined\pdfstringdefDisableCommands
  \pdfstringdefDisableCommands{%
    \def\pending#1{#1}\def\provisional#1{#1}\def\placeholder#1{#1}%
    \def\ownerdraft#1#2{#2}\def\authornote#1{[#1]}\def\needsnumber#1{[NUMBER: #1]}}%
  \fi}
"""


def sec_fig3_replicates(C):
    """Replicate reliability and condition contrast in the Fig. 3a/3b definition.

    Computed with the figure's own analysis code (scripts/figures/paper/fig3_replicates.py,
    fig3b_conditions.py): non-bound chains only, leave-one-ENTRY-out consensus from the
    unfiltered labels of replicates in other entries, tier-specific labels, cumulative tiers,
    <= 20 seeded pairs per sequence (and tier / condition class), 1,000 sequence bootstraps.
    The older \\repDev*, \\rRepResidue*, \\cond* and \\pctDeviant* previews (sec_replicates)
    pool bound and same-entry chains and stay for Supplementary Fig. S7, which shows both."""
    fr = paper("fig3_replicates")
    f3b = paper("fig3b_conditions")
    D = fr.Data()
    L = fr.loo_table(D)
    S_ = fr.summarise_loo(L)
    Pp = fr.pair_table(D)
    ref = Pp[Pp.tier == "unfiltered"]
    src_a = "scripts/figures/paper/fig3_replicates.py loo_table + summarise_loo (Fig. 3a1 definition)"
    src_r = "scripts/figures/paper/fig3_replicates.py pair_table + boot_pooled (Fig. 3a2 definition)"
    st_of = lambda t_: STABLE if t_ == "unfiltered" else PENDING  # noqa: E731
    sa = S_[S_.kbin == "all"].set_index("tier")
    plan_dev = {
        "unfiltered": "0.055",
        "tolerant": "0.050",
        "moderate": "0.048",
        "strict": "0.032",
    }
    plan_n = {
        "unfiltered": "1,027",
        "tolerant": "723",
        "moderate": "385",
        "strict": "81",
    }
    plan_r = {
        "unfiltered": "0.740",
        "tolerant": "0.845",
        "moderate": "0.834",
        "strict": "0.909",
    }
    for t_ in TIERS:
        T = TCAP[t_]
        r_ = sa.loc[t_]
        R.add(f"nRepChainsFig{T}", I(r_.n_chains), f"non-bound chains with a leave-one-entry-out deviation, cumulative {t_} tier (Fig. 3a1)", src_a, st_of(t_), "r4", plan_n[t_])  # fmt: skip
        R.add(f"nRepSeqsFig{T}", I(r_.n_sequences), f"sequences of those chains, cumulative {t_} tier", src_a, st_of(t_), "r4")  # fmt: skip
        R.add(f"repDevFig{T}", F(r_.median_dev, 3), f"median leave-one-entry-out deviation |G - consensus|, cumulative {t_} tier (Fig. 3a1)", src_a, st_of(t_), "r4", plan_dev[t_])  # fmt: skip
        R.add(f"repDevFig{T}Lo", F(r_.median_dev_lo, 3), "lower 95 % sequence-bootstrap bound of that median", src_a, st_of(t_), "r4")  # fmt: skip
        R.add(f"repDevFig{T}Hi", F(r_.median_dev_hi, 3), "upper 95 % sequence-bootstrap bound of that median", src_a, st_of(t_), "r4")  # fmt: skip
        R.add(f"pctRepDevAboveTenthFig{T}", P(100 * r_["share_dev_gt_0.1"]), f"share of those chains deviating by more than 0.1, cumulative {t_} tier", src_a, st_of(t_), "r4")  # fmt: skip
        R.add(f"pctRepFlipFig{T}", P(100 * r_.flip), f"mean flip rate at G = 0.65 against the consensus, cumulative {t_} tier", src_a, st_of(t_), "r4")  # fmt: skip
        for kb, tag in zip(fr.K_LABELS, ("KLow", "KMid", "KHigh")):
            rk = S_[(S_.kbin == kb) & (S_.tier == t_)].iloc[0]
            R.add(f"repDevFig{T}{tag}", F(rk.median_dev, 3), f"median leave-one-entry-out deviation, cumulative {t_} tier, chain median k stratum {kb}", src_a, st_of(t_), "r4")  # fmt: skip
            R.add(f"nRepChainsFig{T}{tag}", I(rk.n_chains), f"chains in that stratum ({kb}), cumulative {t_} tier", src_a, st_of(t_), "r4")  # fmt: skip
        sub = Pp[Pp.tier == t_]
        b = fr.boot_pooled(sub)
        R.add(f"rRepFig{T}", F(b["r"], 3), f"pooled residue-level Pearson r between replicate chains, cumulative {t_} tier (Fig. 3a2)", src_r, st_of(t_), "r4", plan_r[t_])  # fmt: skip
        R.add(f"rRepFig{T}Short", F(b["r"], 2), "same, two decimals", src_r, st_of(t_), "r4")  # fmt: skip
        R.add(f"rRepFig{T}Lo", F(b["r_lo"], 3), "lower 95 % sequence-bootstrap bound of that r", src_r, st_of(t_), "r4")  # fmt: skip
        R.add(f"rRepFig{T}Hi", F(b["r_hi"], 3), "upper 95 % sequence-bootstrap bound of that r", src_r, st_of(t_), "r4")  # fmt: skip
        R.add(f"rSqrtRepFig{T}", F(math.sqrt(b["r"]), 3), "square root of that r (bound on a predictor's correlation with one measurement)", src_r, st_of(t_), "r4")  # fmt: skip
        R.add(f"rRepFigMatched{T}", F(fr.variance_matched_r(sub, ref), 3), f"same r after re-weighting pairs to the unfiltered tier's pair-mean G distribution, {t_}", src_r, st_of(t_), "r4")  # fmt: skip
        R.add(f"nRepPairsFig{T}", I(b["n_pairs"]), f"replicate pairs behind that r, cumulative {t_} tier", src_r, st_of(t_), "r4")  # fmt: skip
        R.add(f"nRepPairSeqsFig{T}", I(b["n_sequences"]), f"sequences behind those pairs, cumulative {t_} tier", src_r, st_of(t_), "r4")  # fmt: skip
        R.add(f"maeRepFig{T}", F(b["mae"], 3), f"pooled residue-level mean absolute G difference between replicates, cumulative {t_} tier", src_r, st_of(t_), "r4")  # fmt: skip
        R.add(f"pctRepPairFlipFig{T}", P(100 * b["flip"]), f"share of co-scored residues on opposite sides of G = 0.65, cumulative {t_} tier", src_r, st_of(t_), "r4")  # fmt: skip
    R.add(
        "pctRepDevReductionFig",
        P(
            100
            * (1 - sa.loc["strict", "median_dev"] / sa.loc["unfiltered", "median_dev"]),
            0,
        ),
        "relative reduction of the median leave-one-entry-out deviation from the unfiltered to the strict tier (Fig. 3a1)",
        src_a,
        PENDING,
        "r4",
        "42",
    )
    # Fig. 3b1: same vs different conditions
    Cp = fr.condition_pairs(D)
    raw = fr.boot_conditions(Cp, matched=False)
    mat = fr.boot_conditions(Cp, matched=True)
    src_c = "scripts/figures/paper/fig3_replicates.py condition_pairs + boot_conditions (Fig. 3b1; unfiltered labels)"
    for cond, T in (("same", "Same"), ("different", "Diff")):
        R.add(f"nCondPairsFig{T}", I(raw[cond]["n_pairs"]), f"{cond}-condition replicate pairs (Fig. 3b1)", src_c, STABLE, "r4", {"same": "949", "different": "984"}[cond])  # fmt: skip
        R.add(f"nCondSeqsFig{T}", I(raw[cond]["n_sequences"]), f"sequences behind the {cond}-condition pairs", src_c, STABLE, "r4")  # fmt: skip
        R.add(f"condAbsDGFig{T}", F(raw[cond]["median_absdg"], 3), f"median per-pair mean |dG|, {cond} conditions, before shift-count matching", src_c, STABLE, "r4")  # fmt: skip
        R.add(f"condRFig{T}", F(raw[cond]["r"], 3), f"pooled residue-level r, {cond} conditions, before matching", src_c, STABLE, "r4")  # fmt: skip
        R.add(f"condMaeFig{T}", F(raw[cond]["mae"], 3), f"pooled residue-level MAE, {cond} conditions, before matching", src_c, STABLE, "r4")  # fmt: skip
        R.add(f"pctCondFlipFig{T}", P(100 * raw[cond]["flip"]), f"flip rate at G = 0.65, {cond} conditions, before matching", src_c, STABLE, "r4")  # fmt: skip
    for cond, T in (("same", "Same"), ("different", "Diff")):
        R.add(f"condAbsDGFig{T}Lo", F(mat[cond]["median_absdg_lo"], 3), f"lower 95 % bound of the median per-pair |dG|, {cond} conditions (shift-count matched bootstrap)", src_c, STABLE, "r4")  # fmt: skip
        R.add(f"condAbsDGFig{T}Hi", F(mat[cond]["median_absdg_hi"], 3), f"upper 95 % bound of the median per-pair |dG|, {cond} conditions (shift-count matched bootstrap)", src_c, STABLE, "r4")  # fmt: skip
    R.add("condAbsDGFigDiffMatched", F(mat["different"]["median_absdg"], 3), "median per-pair mean |dG|, different conditions, re-weighted to the same-condition |dk| profile", src_c, STABLE, "r4", "0.074")  # fmt: skip
    R.add("condRFigDiffMatched", F(mat["different"]["r"], 3), "pooled residue-level r, different conditions, shift-count matched", src_c, STABLE, "r4", "0.628")  # fmt: skip
    R.add("condMaeFigDiffMatched", F(mat["different"]["mae"], 3), "pooled residue-level MAE, different conditions, shift-count matched", src_c, STABLE, "r4", "0.133")  # fmt: skip
    R.add("pctCondFlipFigDiffMatched", P(100 * mat["different"]["flip"]), "flip rate at G = 0.65, different conditions, shift-count matched", src_c, STABLE, "r4")  # fmt: skip
    for d_, T in ((raw, ""), (mat, "Matched")):
        rr = d_["ratio_median_absdg"]
        R.add(f"condRatioFig{T}", F(rr["est"], 2), f"ratio of median per-pair |dG|, different / same conditions{' (shift-count matched)' if T else ''}", src_c, STABLE, "r4", "1.87" if T else "1.90")  # fmt: skip
        R.add(f"condRatioFig{T}Lo", F(rr["lo"], 2), "lower 95 % sequence-bootstrap bound of that ratio", src_c, STABLE, "r4")  # fmt: skip
        R.add(f"condRatioFig{T}Hi", F(rr["hi"], 2), "upper 95 % sequence-bootstrap bound of that ratio", src_c, STABLE, "r4")  # fmt: skip
    # Fig. 3b2: annotations of deviant vs consistent unfiltered-only chains
    Fl = f3b.flag_table(D, L)
    B = f3b.boot_shares(Fl)
    src_b = "scripts/figures/paper/fig3b_conditions.py flag_table + boot_shares (Fig. 3b2; non-bound chains scored only at the unfiltered tier; deviant = deviation > 0.1)"
    for g, T in (("deviant", "Deviant"), ("consistent", "Consistent")):
        Bg = B[B.group == g].set_index("flag")
        R.add(f"n{T}Fig", I(Bg.n_chains.iloc[0]), f"{g} chains in Fig. 3b2", src_b, PENDING, "r4", {"deviant": "124", "consistent": "180"}[g])  # fmt: skip
        for flag_, tag in (
            ("state", "State"),
            ("cosolvent", "Cosolvent"),
            ("offset", "Offset"),
            ("solid", "Solid"),
            ("phT", "PhT"),
        ):
            R.add(f"pct{T}Fig{tag}", P(100 * Bg.loc[flag_, "share"]), f"share of {g} chains with flag '{flag_}' (Fig. 3b2)", src_b, PENDING, "r4")  # fmt: skip
        sel_ = Fl.deviant if g == "deviant" else ~Fl.deviant
        R.add(f"pct{T}FigAny", P(100 * Fl[sel_][["state", "cosolvent", "offset"]].any(axis=1).mean()), f"share of {g} chains with a non-native state, a cosolvent or an offset > 3 sigma", src_b, PENDING, "r4")  # fmt: skip


def sec_conditions_chain(C):
    """Chain-level sample-condition facts behind Supplementary Figs S4 and S5 (raw NMR-STAR,
    resolved per shift table), computed with the figures' own loaders."""
    s4 = paper("supp_s4_conditions")
    d4, n_unres = s4.load()
    src4 = "raw NMR-STAR condition list of each chain's shift table (scripts/figures/paper/supp_s4_conditions.py load; paper_raw_scan.py cache)"
    st_of = lambda t_: STABLE if t_ == "unfiltered" else PENDING  # noqa: E731
    hp = d4[d4.pressure_bar >= s4.HIGH_P]
    for t_ in TIERS:
        T = TCAP[t_]
        R.add(f"nPressureHighChains{T}", I(hp[f"in_{t_}"].sum()), f"chains of the {t_} tier whose own condition list records >= 400 bar (chain level; 25704's '1013 bar' unit slip included)", src4, st_of(t_), "r4")  # fmt: skip
        for col, nm in (
            ("ph", "Ph"),
            ("temperature", "Temp"),
            ("ionic_strength", "Ionic"),
        ):
            R.add(f"n{nm}NotDeposited{T}", I((d4[f"in_{t_}"] & ~d4[f"{col}_deposited"]).sum()), f"{t_}-tier chains without a deposited {col.replace('_', ' ')} (a default is substituted; strict substitutes none)", src4, st_of(t_), "r4")  # fmt: skip
    R.add("nPressureNoRowChains", I(d4.pressure_raw.isna().sum()), "scored chains whose condition list has no pressure row", src4, STABLE, "r4", "3,711")  # fmt: skip
    R.add("nPressureUnreadableChains", I((d4.pressure_raw.notna() & d4.pressure_bar.isna()).sum()), "scored chains whose pressure row has an unreadable value or unit", src4, STABLE, "r4", "60")  # fmt: skip
    R.add("nPressureBelowHundredthBarChains", I((d4.pressure_bar < 0.01).sum()), "scored chains recording < 0.01 bar (unit slips such as 'Pa' for 1 atm)", src4, STABLE, "r4", "85")  # fmt: skip
    s5 = paper("supp_s5_composition")
    d5, sets5 = s5.load()
    src5 = "raw NMR-STAR _Entry.Submission_date / _Entity_natural_src (scripts/figures/paper/supp_s5_composition.py load)"
    R.add("nChainsPlaceholderDate", I((sets5["unfiltered"]["sub"] == "1995-07-31").sum()), "scored chains whose entry carries the BMRB placeholder submission date 1995-07-31", src5, STABLE, "r1", "647")  # fmt: skip
    org = d5.org
    none_ = org.isna() | org.isin([".", "?", ""])
    R.add("nOrgNotRecordedChains", I(none_.sum()), "scored chains without a recorded source organism ('.' or no _Entity_natural_src row for the entity)", src5, STABLE, "r1", "1,553")  # fmt: skip


def write_tex(path: Path, C) -> None:
    lines = [
        "% numbers.tex -- GENERATED by scripts/manuscript/make_numbers.py; DO NOT EDIT BY HAND.",
        f"% Build: {C.manifest['pipeline_version']} ({C.manifest['version']}), generated {time.strftime('%Y-%m-%d %H:%M')}.",
        "% One \\newcommand per number. Index with definitions, sources and status: numbers_index.md.",
        "% Conventions: integers carry LaTeX thousands separators (16{,}851); pct* macros INCLUDE the",
        "% percent sign (6.8\\%); ratio* macros are bare numbers (append $\\times$ or '-fold'); negative",
        "% values use \\ensuremath{-}. Write \\macro{} (empty group) before a following word.",
        "% Status codes in the comments: STABLE / PENDING-REBUILD / PREVIEW / PROVISIONAL / UNREPRODUCED.",
        "% Wrap PENDING-REBUILD values in \\pending{} and PROVISIONAL values in \\provisional{} in prose.",
        "",
        MARKERS,
    ]
    by_sec = OrderedDict((k, []) for k in SECTIONS)
    for it in R.items:
        by_sec[it.section].append(it)
    for key, title in SECTIONS.items():
        if not by_sec[key]:
            continue
        lines.append("")
        lines.append("% " + "=" * 86)
        lines.append(f"% {title}")
        lines.append("% " + "=" * 86)
        for it in by_sec[key]:
            com = f"{it.status} | {it.definition} | src: {it.source}"
            if it.plan is not None and not it.plan_ok:
                com += f" | PLAN VALUE DIFFERS: {it.plan}"
            com = com.replace("\n", " ")
            lines.append(f"\\newcommand{{\\{it.name}}}{{{it.value}}} % {com}")
    path.write_text("\n".join(lines) + "\n")


def md_escape(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def write_index(path: Path, C) -> None:
    n = len(R.items)
    cnt = Counter(it.status for it in R.items)
    mism = [it for it in R.items if it.plan is not None and not it.plan_ok]
    out = [
        "# Numbers index for the TriZOD draft",
        "",
        f"Generated by `scripts/manuscript/make_numbers.py` on {time.strftime('%Y-%m-%d %H:%M')} from the frozen build "
        f"`{C.manifest['pipeline_version']}` (release label `{C.manifest['version']}`). **Do not edit by hand**: re-run the script after the rebuild.",
        "",
        f"{n} macros: "
        + ", ".join(
            f"{k} {cnt.get(k, 0)}"
            for k in (STABLE, PENDING, PREVIEW, PROVISIONAL, UNREPRO)
        )
        + ".",
        "",
        "**How to use.** Write `\\nChainsScored{}` in prose, never the number. `pct*` macros include the percent sign; `ratio*` macros are bare "
        '(append `$\\times$` or "-fold"); integers carry `{,}` thousands separators; negative values use `\\ensuremath{-}`. '
        "Wrap every PENDING-REBUILD value (and any claim resting on it) in `\\pending{}`, every PROVISIONAL value in `\\provisional{}`. "
        "If a number is missing, write `\\needsnumber{<definition and source>}`.",
        "",
        "**Status codes.** STABLE survives the planned rebuild. PENDING-REBUILD moves with fixes (a)-(f). PREVIEW is an exploratory or "
        "pre-build analysis to recompute with the stated controls (the CheZOD reproduction is pre-build: re-run on 2026-10 code). "
        "PROVISIONAL is a stale DisProt number. UNREPRODUCED keeps the writing plan's value because this script cannot recompute it; the reason is in the source column.",
        "",
        "**Marker macros** (defined at the top of `numbers.tex`): `\\pending{}`, `\\provisional{}`, `\\placeholder{}`, `\\ownerdraft{RA}{}`, "
        "`\\authornote{}`, `\\needsnumber{}`; `\\draftmarkersoff` renders them as plain text. Use `\\placeholder{}` at the start of a "
        "paragraph for anything longer than a few words: in running text it is an unbreakable inline box.",
        "",
    ]
    unrep = [it for it in R.items if it.status == UNREPRO]
    if unrep:
        out += [
            "## UNREPRODUCED values (plan value kept; do not cite without a source)",
            "",
            "| macro | value | definition | reason |",
            "|---|---|---|---|",
        ]
        for it in unrep:
            out.append(
                f"| `\\{it.name}` | {md_escape(plainify(it.value))} | {md_escape(it.definition)} | {md_escape(it.source)} |"
            )
        out.append("")
    if mism:
        out += ["## Values that differ from the writing plan", "", "The macro carries the recomputed value; the plan's figure is shown for reference.", "",
                "| macro | value | plan | definition |", "|---|---|---|---|"]  # fmt: skip
        for it in mism:
            out.append(
                f"| `\\{it.name}` | {md_escape(plainify(it.value))} | {md_escape(it.plan)} | {md_escape(it.definition)} |"
            )
        out.append("")
    by_sec = OrderedDict((k, []) for k in SECTIONS)
    for it in R.items:
        by_sec[it.section].append(it)
    for key, title in SECTIONS.items():
        if not by_sec[key]:
            continue
        out += [
            f"## {title}",
            "",
            "| macro | value | definition | source | status |",
            "|---|---|---|---|---|",
        ]
        for it in by_sec[key]:
            d = it.definition
            if it.plan is not None and not it.plan_ok:
                d += f" **(plan: {it.plan})**"
            out.append(
                f"| `\\{it.name}` | {md_escape(plainify(it.value))} | {md_escape(d)} | {md_escape(it.source)} | {it.status} |"
            )
        out.append("")
    path.write_text("\n".join(out))


# ----------------------------------------------------------------------------------
def main(argv=None):
    global PATHS, REL
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="checkout that holds data/ (default: first parent with data/interim/build)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory for numbers.tex and numbers_index.md (the manuscript-draft directory)",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=(
            Path(os.environ["TRIZOD_CACHE_DIR"]).expanduser()
            if os.environ.get("TRIZOD_CACHE_DIR")
            else paths.TMP
        )
        / "manuscript_numbers",
        help="cache for raw scan and mmseqs runs (default: %(default)s; "
        "$TRIZOD_CACHE_DIR/manuscript_numbers when that is set)",
    )
    ap.add_argument(
        "--refresh", action="store_true", help="recompute cached intermediates"
    )
    ap.add_argument(
        "--skip-baselines",
        action="store_true",
        help="skip the ridge baselines (slowest step)",
    )
    ap.add_argument(
        "--write-only",
        action="store_true",
        help="rewrite numbers.tex / numbers_index.md from the last registry dump in --work-dir",
    )
    a = ap.parse_args(argv)
    root = (a.root or find_root(REPO)).resolve()
    work = a.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    PATHS = Paths(root, work)
    out_dir = a.out_dir.resolve()
    t0 = time.time()
    dump = work / "registry.json"
    if a.write_only:
        d = json.loads(dump.read_text())
        R.items[:] = [Num(**x) for x in d["items"]]
        C = Ctx()
        C.manifest = d["manifest"]
        out_dir.mkdir(parents=True, exist_ok=True)
        write_tex(out_dir / "numbers.tex", C)
        write_index(out_dir / "numbers_index.md", C)
        print(
            f"rewrote {out_dir / 'numbers.tex'} and numbers_index.md from {dump} ({len(R.items)} macros)"
        )
        return
    C = load_context(a.refresh)
    C.refresh = a.refresh
    REL = C.manifest.get("version", REL)
    steps = [sec_snapshot, sec_tiers, sec_filter_losses, sec_test_sets, sec_residues, sec_tolerant_labels, sec_annotations, sec_offsets,
             sec_panav, sec_shifts, sec_sidechain_doc, sec_chezod, sec_clusters_and_leakage, sec_ubiquitin, sec_replicates,
             sec_fig3_replicates, sec_failures, sec_discussion_revision, sec_cosolvents_pressure, sec_conditions_chain, sec_results_b_revision, sec_front_revision, sec_fig1_revision, sec_methods_b_revision, sec_organisms, sec_review_supp_text, sec_supp_figs_revision, sec_params,
             sec_literature]  # fmt: skip
    if not a.skip_baselines:
        steps.insert(-2, sec_baselines)
    steps.append(sec_rev_results_a)  # results_a.tex revision (31 Oct)
    steps.append(sec_methods_a_revision)  # methods_a.tex revision (31 Oct)
    for fn in steps:
        t = time.time()
        fn(C)
        print(
            f"  {fn.__name__}: {time.time() - t:.0f} s ({len(R.items)} macros)",
            file=sys.stderr,
        )
    dump.write_text(
        json.dumps({"manifest": C.manifest, "items": [it.__dict__ for it in R.items]})
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tex(out_dir / "numbers.tex", C)
    write_index(out_dir / "numbers_index.md", C)
    mism = [it for it in R.items if it.plan is not None and not it.plan_ok]
    print(f"wrote {out_dir / 'numbers.tex'} and numbers_index.md: {len(R.items)} macros, {len(mism)} differ from the plan, "
          f"{sum(1 for it in R.items if it.status == UNREPRO)} UNREPRODUCED ({time.time() - t0:.0f} s)")  # fmt: skip
    for it in mism:
        print(f"  DIFF {it.name}: {plainify(it.value)} vs plan {it.plan}")


_CANDIDATE_CODE = r"""
import json, logging, pickle, sys
from pathlib import Path
logging.disable(logging.CRITICAL)
n_rec, ents = 0, set()
for fp in sorted(Path(sys.argv[1]).glob("*.pkl")):
    with fp.open("rb") as f:
        e = pickle.load(f)
    ps = list(e.get_peptide_shifts())
    if ps:
        n_rec += len(ps)
        ents.add(fp.stem)
print(json.dumps({"n_pickles": len(list(Path(sys.argv[1]).glob("*.pkl"))),
                  "n_records": n_rec, "n_entries": len(ents)}))
"""


def candidate_record_counts(refresh: bool):
    """Candidate records and their distinct entries, as create_peptide_dataframe() builds
    them: one record per BmrbEntry.get_peptide_shifts() tuple over the build's pickle
    cache (``<root>/tmp/bmrb_entries``, the ``--cache-dir tmp`` of build_dataset.sh).
    Runs in a child process that never imports pyarrow. Returns None without the cache."""
    cache = PATHS.pkl
    if not cache.is_dir():
        return None

    def run():
        out = subprocess.run(
            [sys.executable, "-c", _CANDIDATE_CODE, str(cache)],
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO)},
            capture_output=True,
            text=True,
            check=True,
        )
        return {**json.loads(out.stdout.strip().splitlines()[-1]), "cache": str(cache)}

    return cached_json("candidate_records.json", run, refresh)


# Identical-data re-deposition rule (audit report, "Replicate depositions"): two replicate
# chains repeat one measurement when >= REDEP_FRAC of >= REDEP_MIN_SHARED shared backbone
# shifts (same seq_id and atom_id) agree within REDEP_TOL_PPM of the deposited value.
REDEP_TOL_PPM, REDEP_MIN_SHARED, REDEP_FRAC = 0.005, 20, 0.9


def sec_front_revision(C):
    """Numbers added in the front.tex revision (31 Oct).

    (1) \\nCandidateRecordsApprox: the archive-size figure of abstract sentence 2 ('the archive
    holds shifts for about N protein chains'), rounded from \\nCandidateRecords; the earlier
    wording used \\nChainsScoredApprox, which counts scored chains, not chains with shifts.
    (2) The Fig. 3a2 residue-level replicate r recomputed WITHOUT identical-data re-depositions:
    fig3_replicates.pair_table's definition (non-bound chains, different entries, tier labels,
    cumulative tiers, <= PAIR_CAP seeded pairs per sequence and tier, >= 10 co-scored residues,
    sequence bootstrap), with re-deposition pairs dropped from the candidate list BEFORE the
    per-sequence cap. Without the rule, this code reproduces \\rRepFig* exactly (checked 31 Oct).
    PREVIEW until fig3_replicates.py and Methods M7 adopt the rule."""
    n_cand = next(int(plainify(it.value).replace(",", "")) for it in R.items if it.name == "nCandidateRecords")  # fmt: skip
    R.add("nCandidateRecordsApprox", I(int(round(n_cand, -3))), "candidate records rounded to the nearest thousand ('the archive holds shifts for about N protein chains', abstract)", "derived from \\nCandidateRecords (rebuild log pre-filter 'of N')", STABLE, "abs")  # fmt: skip
    fr = paper("fig3_replicates")
    sp_ = paper("style_paper")
    D = fr.Data()
    nb = D.nb
    multi = nb.groupby("sequence")["entry_id"].transform("nunique") >= 2
    ids = set(nb.loc[multi, "id"])
    t = pq.read_table(
        PATHS.shifts,
        columns=["id", "seq_id", "atom_id", "val_ppm", "is_backbone"],
        filters=[("is_backbone", "=", True)],
    ).to_pandas()
    t = t[t["id"].isin(ids)].drop_duplicates(["id", "seq_id", "atom_id"])
    SH = {
        i: pd.Series(
            g["val_ppm"].to_numpy(float),
            index=pd.MultiIndex.from_arrays([g["seq_id"], g["atom_id"]]),
        )  # fmt: skip
        for i, g in t.groupby("id", sort=False)
    }

    def redeposit(a, b):
        sa, sb = SH.get(a), SH.get(b)
        if sa is None or sb is None:
            return False
        j = sa.index.intersection(sb.index)
        if len(j) < REDEP_MIN_SHARED:
            return False
        d = np.abs(sa.loc[j].to_numpy() - sb.loc[j].to_numpy())
        return bool((d <= REDEP_TOL_PPM).mean() >= REDEP_FRAC)

    rng = random.Random(fr.SEED)
    rows, dup = [], set()
    for seq, g in nb.groupby("sequence", sort=True):
        if g["entry_id"].nunique() < 2:
            continue
        cand = []
        for a, b in fr._seq_pairs(g, rng):
            (dup.add((a, b)) if redeposit(a, b) else cand.append((a, b)))
        for t_ in sp_.TIERS:
            ids_t = D.tier_ids[t_]
            taken = 0
            for a, b in cand:
                if taken >= fr.PAIR_CAP:
                    break
                if a not in ids_t or b not in ids_t:
                    continue
                ga, gb = D.g(t_, a), D.g(t_, b)
                m = np.isfinite(ga) & np.isfinite(gb)
                if m.sum() < fr.MIN_PAIR_RES:
                    continue
                taken += 1
                x, y = ga[m], gb[m]
                rows.append({"sequence": seq, "tier": t_, "a": a, "b": b, **fr._pair_stats(x, y),
                             "hist": np.histogram((x + y) / 2, bins=10, range=(0, 1))[0]})  # fmt: skip
    Pp = pd.DataFrame(rows)
    ref = Pp[Pp.tier == "unfiltered"]
    src = (f"make_numbers.py sec_front_revision: fig3_replicates.py pair_table definition minus identical-data "
           f"re-depositions (>= {REDEP_FRAC:.0%} of >= {REDEP_MIN_SHARED} shared backbone shifts within "
           f"{REDEP_TOL_PPM} ppm, trizod_shifts.parquet val_ppm), dropped before the per-sequence cap; boot_pooled")  # fmt: skip
    R.add("nRedepositPairsFig", I(len(dup)), "candidate replicate pairs (non-bound, identical sequence, different entries) that are identical-data re-depositions", src, PREVIEW, "r4")  # fmt: skip
    for t_ in TIERS:
        T = TCAP[t_]
        sub = Pp[Pp.tier == t_]
        b = fr.boot_pooled(sub)
        R.add(f"rRepFigNoRedep{T}", F(b["r"], 3), f"pooled residue-level replicate r without identical-data re-depositions, cumulative {t_} tier (Fig. 3a2 definition otherwise)", src, PREVIEW, "r4")  # fmt: skip
        R.add(f"rRepFigNoRedep{T}Short", F(b["r"], 2), "same, two decimals", src, PREVIEW, "r4")  # fmt: skip
        R.add(f"rRepFigNoRedep{T}Lo", F(b["r_lo"], 3), "lower 95 % sequence-bootstrap bound of that r", src, PREVIEW, "r4")  # fmt: skip
        R.add(f"rRepFigNoRedep{T}Hi", F(b["r_hi"], 3), "upper 95 % sequence-bootstrap bound of that r", src, PREVIEW, "r4")  # fmt: skip
        R.add(f"rRepFigMatchedNoRedep{T}", F(fr.variance_matched_r(sub, ref), 3), f"same r re-weighted to the unfiltered tier's pair-mean G distribution, {t_}", src, PREVIEW, "r4")  # fmt: skip
        R.add(f"nRepPairsFigNoRedep{T}", I(b["n_pairs"]), f"replicate pairs behind that r, cumulative {t_} tier", src, PREVIEW, "r4")  # fmt: skip
        R.add(f"nRepPairSeqsFigNoRedep{T}", I(b["n_sequences"]), f"sequences behind those pairs, cumulative {t_} tier", src, PREVIEW, "r4")  # fmt: skip
        R.add(f"maeRepFigNoRedep{T}", F(b["mae"], 3), f"pooled residue-level mean absolute G difference between replicates without re-depositions, cumulative {t_} tier", src, PREVIEW, "r4")  # fmt: skip


def sec_discussion_revision(C):
    """Numbers added in the discussion.tex revision (31 Oct; reviewer 'numbers').

    (1) Extreme LACS offsets over all seven backbone atoms (Discussion limitations): counts of
    chains with |LACS| >= 10 and >= 100 ppm and the largest |LACS| offset in the release.
    (2) The cytochrome-c case cluster (R6; Discussion state-mismatch case): per-chain mean G of
    the chains with more than CYTC_MIN_SCORED scored residues, split into the denatured sample
    and the rest. Membership is sec_ubiquitin's cached case search (search_cases.json: >= 80 %
    id, >= 90 % query and target coverage to 25640_1_1_1), so this step must run after it."""
    df = C.df
    srcl = "Parquet lacs_off_*_ppm, max |offset| over the seven backbone atoms (unfiltered run; max-offset = inf, nothing rejected)"
    mx = df[[lacs_off_ppm_col(a) for a in ATOMS]].abs().max(axis=1)
    R.add("nChainsLacsAboveTen", I(int((mx >= 10).sum())), "chains with |LACS offset| >= 10 ppm on any backbone atom", srcl, STABLE, "r6")  # fmt: skip
    R.add("nChainsLacsAboveHundred", I(int((mx >= 100).sum())), "chains with |LACS offset| >= 100 ppm on any backbone atom", srcl, STABLE, "r6")  # fmt: skip
    R.add("lacsMaxAbs", F(mx.max(), 2), "largest |LACS offset| over all chains and backbone atoms (ppm)", srcl, STABLE, "r6")  # fmt: skip
    CYTC_MIN_SCORED = 50
    R.add("thrCaseCytCMinScored", I(CYTC_MIN_SCORED), "cytochrome-c cluster chains enter the per-chain mean-G range only with more than this many scored residues (the shorter chains carry 7-11)", "analysis setting (make_numbers.py sec_discussion_revision)", STABLE, "r6")  # fmt: skip
    p = PATHS.work / "search_cases.json"
    if not p.exists():
        for nm, d in (("gCaseCytCOrderedMin", "lowest per-chain mean G, non-denatured long cytochrome-c chains"),
                      ("gCaseCytCOrderedMax", "highest per-chain mean G, non-denatured long cytochrome-c chains"),
                      ("gCaseCytCDenatured", "per-chain mean G of the denatured cytochrome-c chain")):  # fmt: skip
            R.unrepro(
                nm, "?", d, "case search cache missing (mmseqs not on PATH)", "r6"
            )
        return
    H = pd.DataFrame(json.loads(p.read_text()), columns=["q", "t", "fid", "qcov", "tcov", "e", "aln", "ql", "tl"])  # fmt: skip
    H = H[(H.q == "CytC") & (H.fid >= 0.8) & (H.qcov >= 0.9) & (H.tcov >= 0.9)]
    mem = df[df.id.isin(set(H.t))]
    srcc = "Parquet gscores (unfiltered labels; per-chain nanmean), n_scored, physical_state x the CytC case cluster (search_cases.json, >= 80 % id, >= 90 % query and target coverage to 25640_1_1_1)"
    long_ = mem[mem.n_scored > CYTC_MIN_SCORED]
    den = long_[long_.physical_state == "denatured"]
    rest = long_[long_.physical_state != "denatured"]
    R.add("nCaseCytCLongChains", I(len(long_)), "cytochrome-c cluster chains with more than \\thrCaseCytCMinScored scored residues", srcc, STABLE, "r6")  # fmt: skip
    R.add("nCaseCytCOrderedChains", I(len(rest)), "of them, chains not annotated as denatured", srcc, STABLE, "r6")  # fmt: skip
    R.add("gCaseCytCOrderedMin", F(rest.meanG.min(), 2), "lowest per-chain mean G among those non-denatured chains", srcc, STABLE, "r6")  # fmt: skip
    R.add("gCaseCytCOrderedMax", F(rest.meanG.max(), 2), "highest per-chain mean G among those non-denatured chains", srcc, STABLE, "r6")  # fmt: skip
    R.add("gCaseCytCOrderedMedian", F(rest.meanG.median(), 2), "median per-chain mean G among those non-denatured chains", srcc, STABLE, "r6")  # fmt: skip
    R.add("nCaseCytCDenatured", I(len(den)), "long cytochrome-c cluster chains annotated as denatured (physical_state)", srcc, STABLE, "r6")  # fmt: skip
    if len(den) == 1:
        R.add("idCaseCytCDenatured", S(den.id.iloc[0]), "the denatured cytochrome-c chain", srcc, STABLE, "r6")  # fmt: skip
        R.add("gCaseCytCDenatured", F(den.meanG.iloc[0], 2), "its per-chain mean G", srcc, STABLE, "r6")  # fmt: skip


def sec_methods_b_revision(C):
    """Numbers added in the methods_b.tex revision (31 Oct).

    (1) Nature-style long dates ('6 November 2025') of \\dateNewestSubmission and \\dateTestDraw,
    for reader-facing text; the ISO macros stay for tables and notes.
    (2) The seeds of the replicate-pair sampling and sequence bootstraps, read from the figure
    generators (fig3_replicates.SEED; its condition_pairs() draws with SEED + 1), checked against
    the k-paired test (fig2a_kpaired.SEED) and Supplementary Fig. S7 (supp_s7_replicates.SEED).
    (3) The minimum length of the planned exact-substring leakage gate (audit report D2: three
    representative/test containment pairs are only 21-22 aa long, so the gate uses 20 aa)."""
    import datetime as _dt

    def _val(name):
        return next(plainify(it.value) for it in R.items if it.name == name)

    def _long(iso):
        d = _dt.date.fromisoformat(iso)
        return f"{d.day} {d.strftime('%B')} {d.year}"

    for name, src in (("dateNewestSubmission", "raw NMR-STAR scan"),
                      ("dateTestDraw", "testset/build_test_set.log mtime / facts sheet Q3")):  # fmt: skip
        R.add(f"{name}Long", S(_long(_val(name))), f"\\{name} written day-month-year (Nature style)", f"derived from \\{name} ({src})", STABLE, "par")  # fmt: skip
    # retrieval date of the BMRB snapshot, from BMRB_SNAPSHOT.json in the snapshot directory
    # (written when the snapshot is downloaded); a snapshot without it is an error, so the
    # paper never states a date the build cannot prove
    snap = PATHS.raw / "BMRB_SNAPSHOT.json"
    sj = json.loads(snap.read_text())
    iso = sj["retrieved"][:10]
    d_ = _dt.date.fromisoformat(iso)
    src_s = f"{snap} ('retrieved': {sj['retrieved']})"
    R.add("dateSnapshot", S(iso), "retrieval date of the BMRB snapshot (ISO)", src_s, STABLE, "par")  # fmt: skip
    R.add("dateSnapshotLong", S(_long(iso)), "retrieval date of the BMRB snapshot, day-month-year (Nature style)", src_s, STABLE, "par")  # fmt: skip
    R.add("dateSnapshotShort", S(f"{d_.day} {d_.strftime('%b')} {d_.year}"), "retrieval date of the BMRB snapshot, abbreviated month (figure text)", src_s, STABLE, "par")  # fmt: skip
    fr = paper("fig3_replicates")
    f2a = paper("fig2a_kpaired")
    s7 = paper("supp_s7_replicates")
    assert fr.SEED == f2a.SEED == s7.SEED, (fr.SEED, f2a.SEED, s7.SEED)
    R.add("seedPairs", I(fr.SEED), "seed of the replicate-pair sampling (Figs. 2a, 3a; Supplementary Fig. S7) and of the sequence bootstraps", "scripts/figures/paper/fig3_replicates.py, fig2a_kpaired.py, supp_s7_replicates.py SEED", STABLE, "par")  # fmt: skip
    R.add("seedConditionPairs", I(fr.SEED + 1), "seed of the same- versus different-condition pair sampling (Fig. 3b)", "scripts/figures/paper/fig3_replicates.py condition_pairs(): random.Random(SEED + 1)", STABLE, "par")  # fmt: skip
    R.add("substringGateMin", I(20), "minimum length (residues) of an exact-substring containment that the planned release gate rejects", "audit report decision D2 (three containment pairs are only 21-22 aa long)", PENDING, "par")  # fmt: skip


def sec_review_supp_text(C):
    """Numbers added by the review of sections/supp_text.tex (31 Oct 2026).

    Supplementary Note N2 rules 3 and 5 (near-redundant representatives; Spearman replicate
    ceiling), N4 (stratum intervals of the Fig. 3a1 deviation), N7 (deferred label-changing
    fixes: placeholder ionic strengths, pH*/pD-only entries, corrected-value outliers) and
    Supplementary Table T12 (share of CheZOD1325 entries with a per-entry error <= 0.5 Z)."""
    df = C.df
    # --- N4: 95 % intervals of the cumulative Fig. 3a1 deviation per chain-median-k stratum ---
    fr = paper("fig3_replicates")
    S_ = fr.summarise_loo(fr.loo_table(fr.Data()))
    src_a = "scripts/figures/paper/fig3_replicates.py loo_table + summarise_loo (Fig. 3a1 definition; sequence bootstrap)"
    for t_ in TIERS:
        st_ = STABLE if t_ == "unfiltered" else PENDING
        for kb, tag in zip(fr.K_LABELS, ("KLow", "KMid", "KHigh")):
            rk = S_[(S_.kbin == kb) & (S_.tier == t_)].iloc[0]
            R.add(f"repDevFig{TCAP[t_]}{tag}Lo", F(rk.median_dev_lo, 3), f"lower 95 % sequence-bootstrap bound of the median leave-one-entry-out deviation, cumulative {t_} tier, chain median k stratum {kb}", src_a, st_, "r4")  # fmt: skip
            R.add(f"repDevFig{TCAP[t_]}{tag}Hi", F(rk.median_dev_hi, 3), f"upper 95 % sequence-bootstrap bound of the median leave-one-entry-out deviation, cumulative {t_} tier, chain median k stratum {kb}", src_a, st_, "r4")  # fmt: skip
    # --- N2 rule 5: replicate ceiling in the protocol's primary metrics (Spearman rho, AUC) ---
    s10 = paper("supp_s10_baselines")
    Cc = s10.tier_ceilings(s10.sp.load_main(s10.COLUMNS)).set_index("tier")
    src_c = "scripts/figures/paper/supp_s10_baselines.py tier_ceilings (pairs of identical-sequence non-bound chains from different entries, <= 20 per sequence, seed 0, the residue-level-r sample of \\rRepResidue*; each chain's G predicts the other's label, both directions; cumulative tiers)"
    for t_ in ("unfiltered", "strict"):
        T = TCAP[t_]
        R.add(f"rhoRepResidue{T}", F(Cc.loc[t_, "rho_res"], 3), f"residue-level Spearman rho of G between replicate chains, cumulative {t_} tier", src_c, PREVIEW, "r5")  # fmt: skip
        R.add(f"rhoSqrtRepResidue{T}", F(math.sqrt(Cc.loc[t_, "rho_res"]), 3), f"square root of that rho (single-measurement ceiling in the protocol's metric), {t_}", src_c, PREVIEW, "r5")  # fmt: skip
        R.add(f"aucRepResidue{T}", F(Cc.loc[t_, "auc_res"], 3), f"residue-level AUC at G > 0.65 of one replicate's G predicting the other's label, cumulative {t_} tier", src_c, PREVIEW, "r5")  # fmt: skip
    # --- N2 rule 3: representatives that stay near-redundant after incremental clustering ---
    reps = read_fasta(
        PATHS.bundle / "train" / "unfiltered" / "train_unfiltered_best.fasta"
    )
    if have_mmseqs() or (PATHS.work / "search_reps_self_common.json").exists():
        args_c = list(paper("fig3c_split").MMSEQS_ARGS)
        hs = mmseqs_search("reps_self_common", list(reps.items()), list(reps.items()), args_c, C.refresh)  # fmt: skip
        HS = pd.DataFrame(hs, columns=["q", "t", "fid", "qcov", "tcov", "e", "aln", "ql", "tl"])  # fmt: skip
        HS = HS[HS.q != HS.t]
        srcs = (
            "mmseqs easy-search " + " ".join(args_c) + " of train_unfiltered_best.fasta against itself "
            "(cache search_reps_self_common.json); unordered pairs of representatives"
        )  # fmt: skip
        for thr, tag in ((0.5, ""), (0.9, "Ninety")):
            p_ = HS[(HS.fid >= thr) & (HS.qcov >= 0.8) & (HS.tcov >= 0.8)]
            pairs = {tuple(sorted(x)) for x in zip(p_.q, p_.t)}
            R.add(f"nRepPairsNearRedundant{tag}", I(len(pairs)), f"pairs of unfiltered training representatives at >= {int(100 * thr)} % identity over >= 80 % of BOTH sequences (survive the 50/80 clustering)", srcs, PENDING, "r5", {"": "59", "Ninety": "4"}[tag])  # fmt: skip
            R.add(f"nRepsNearRedundant{tag}", I(len(set(p_.q) | set(p_.t))), f"unfiltered training representatives in such pairs (>= {int(100 * thr)} % identity)", srcs, PENDING, "r5", {"": "105", "Ninety": "8"}[tag])  # fmt: skip
    # --- T12: per-entry error <= 0.5 Z beside the post hoc outcome classes ---
    s17 = paper("supp_s17_chezod_reproduction")
    cz = s17.load_chezod(s17.sp.CHEZOD_1325)
    for mode, path, T, st_, srcx in (
        ("potenci_only", s17.PO, "PotenciOnly", PREVIEW, rel(s17.PO) + " (pre-2026-10 code; re-run on 2026-10)"),
        ("both", s17.BOTH, "Both", STABLE, rel(s17.BOTH) + f" ({REL} released labels)"),
    ):  # fmt: skip
        d_ = s17.per_entry(cz, s17.load_trizod(path), mode)
        ok = d_.mae <= 0.5
        src_m = (
            srcx
            + " vs CheZOD1325; per-entry alignment of scripts/figures/paper/supp_s17_chezod_reproduction.py"
        )
        R.add(f"nCheZODMaeHalf{T}", I(ok.sum()), f"CheZOD1325 entries with a per-entry mean absolute Z error <= 0.5 ({mode})", src_m, st_, "r3")  # fmt: skip
        R.add(f"pctCheZODMaeHalf{T}", P(100 * ok.mean()), f"share of reproduced CheZOD1325 entries with a per-entry mean absolute Z error <= 0.5 ({mode})", src_m, st_, "r3")  # fmt: skip
    # --- N7: placeholder ionic strengths (exactly 0 M) ---
    z0 = df.ionic_strength == 0
    src_i = "Parquet ionic_strength == 0 (deposited zero; imputation writes 0.1 M) x scores/<tier>/scores.json IDs"
    R.add("nIonicZeroChains", I(z0.sum()), "released chains whose ionic strength is exactly 0 M (placeholder zeros pass as reported values)", src_i, STABLE, "r6", "886")  # fmt: skip
    R.add("nIonicZeroChainsStrict", I(df[z0].id.isin(C.tier_ids["strict"]).sum()), "of them scored at the strict tier", src_i, PENDING, "r6", "132")  # fmt: skip
    R.add("nIonicZeroReps", I((z0 & (df.split == "train")).sum()), "of them training representatives (any tier)", src_i, PENDING, "r6")  # fmt: skip
    # --- N7: entries that record pH only as pH* or pD (the parser reads Type == 'pH' only) ---
    if PATHS.raw.is_dir():

        def scan_ph():
            loop = re.compile(r"loop_\s*\n((?:\s*_Sample_condition_variable\.\S+\s*\n)+)(.*?)\n\s*stop_", re.S)  # fmt: skip
            out = {}
            scored = set(df.entry_id)
            for d in sorted(os.listdir(PATHS.raw)):
                eid = d.replace("bmr", "")
                fs = sorted((PATHS.raw / d).glob("*.str"))
                if eid not in scored or not fs:
                    continue
                txt = fs[0].read_text(errors="ignore")
                has_ph = has_alt = False
                for m in loop.finditer(txt):
                    hdr = re.findall(r"_Sample_condition_variable\.(\S+)", m.group(1))
                    try:
                        toks = shlex.split(m.group(2))
                    except ValueError:
                        continue
                    if not hdr or "Type" not in hdr or len(toks) % len(hdr):
                        continue
                    ti = hdr.index("Type")
                    for r_ in range(len(toks) // len(hdr)):
                        ty = toks[r_ * len(hdr) + ti].lower()
                        if ty == "ph":
                            has_ph = True
                        elif ty.startswith("ph") or ty.startswith("pd"):
                            has_alt = True
                out[eid] = [has_ph, has_alt]
            return out

        ph = cached_json("raw_ph_rows.json", scan_ph, C.refresh)
        only = {e for e, (a, b) in ph.items() if b and not a}
        src_p = "raw NMR-STAR _Sample_condition_variable.Type of scored entries (trizod/bmrb/bmrb.py reads Type == 'pH' only)"
        R.add("nEntriesPhStarOnly", I(len(only)), "scored entries that record the pH only as pH* or pD, so pH 7 is imputed (unfiltered to moderate) or the entry fails strict", src_p, STABLE, "r6", "167")  # fmt: skip
        R.add("nChainsPhStarOnly", I(df.entry_id.isin(only).sum()), "released chains of those entries", src_p, STABLE, "r6")  # fmt: skip
    # --- N7: shifts that become far outliers only after re-referencing ---
    sh = pq.read_table(
        PATHS.shifts,
        columns=["id", "comp_id", "atom_id", "val_corrected_ppm", "mad_z", "is_far_outlier"],
        filters=[("atom_id", "in", ["C", "CA", "CB", "N"])],
    ).to_pandas()  # fmt: skip
    ref = json.loads(pq.read_schema(PATHS.shifts).metadata[b"mad_reference"])
    key = sh.comp_id.astype(str) + " " + sh.atom_id.astype(str)
    med = key.map({k: v["median_ppm"] for k, v in ref.items()})
    mad = key.map({k: v["mad_ppm"] for k, v in ref.items()})
    kthr = np.where(sh.atom_id == "CB", 12.0, 8.0)
    zc = ((sh.val_corrected_ppm - med) / mad).round(2)
    far_c = sh.mad_z.notna() & sh.val_corrected_ppm.notna() & (zc.abs() > kthr)
    only_c = far_c & ~sh.is_far_outlier.fillna(False).astype(bool)
    ch = set(sh.id[only_c])
    split = df.set_index("id").split
    src_o = "trizod_shifts.parquet: assessed C, CA, CB, N rows whose val_corrected_ppm exceeds the far-outlier threshold of the frozen mad_reference (8 MADs; 12 for CB) while val_ppm is not flagged"
    R.add("nShiftsFarOnlyCorrected", I(only_c.sum()), "shifts that are far outliers only after re-referencing (deposited value not flagged)", src_o, STABLE, "m11", "435")  # fmt: skip
    R.add("nChainsFarOnlyCorrected", I(len(ch)), "chains carrying such shifts", src_o, STABLE, "m11")  # fmt: skip
    R.add("nRepsFarOnlyCorrected", I(sum(1 for c in ch if split.get(c) == "train")), "of them training representatives (any tier)", src_o, PENDING, "m11")  # fmt: skip
    # --- T9 / N2 rule 5: the floor baseline in the protocol's primary metric (Spearman rho) ---
    # Same floor as sec_baselines (per-amino-acid mean G of each cumulative tier's training
    # representatives, 21 classes, global-mean fallback); pooled Spearman rho over test residues.
    AA = "ACDEFGHIKLMNPQRSTVWY"
    lut = np.full(256, 20, dtype=np.int64)
    for i_, a_ in enumerate(AA):
        lut[ord(a_)] = i_

    def aa_g(sub):
        aas, gs = [], []
        for s_, g_ in zip(sub.sequence, sub.gscores):
            g_ = np.asarray(g_, float)
            m_ = ~np.isnan(g_)
            aas.append(lut[np.frombuffer(s_.encode(), dtype=np.uint8)][m_])
            gs.append(g_[m_])
        return np.concatenate(aas), np.concatenate(gs)

    tst = {"TriZOD": aa_g(df[df.split == "test_trizod"]), "CheZOD": aa_g(df[df.split == "test_chezod117"])}  # fmt: skip
    rho_floor = {n: [] for n in tst}
    for t_ in TIERS:
        tr = df[df.train_tier.map(lambda v, t0=t_: v is not None and v == v and RANK.get(v, -1) >= RANK[t0])]  # fmt: skip
        a_tr, g_tr = aa_g(tr)
        ysum = np.bincount(a_tr, weights=g_tr, minlength=21)
        ycnt = np.bincount(a_tr, minlength=21)
        aa_mean = np.where(ycnt > 0, ysum / np.maximum(ycnt, 1), g_tr.mean())
        for n, (a_te, g_te) in tst.items():
            p_ = aa_mean[a_te]
            rho_floor[n].append(np.corrcoef(pd.Series(g_te).rank().to_numpy(), pd.Series(p_).rank().to_numpy())[0, 1])  # fmt: skip
    src_f = "Parquet (unfiltered labels); per-amino-acid mean G of the training representatives of each cumulative tier (as sec_baselines), pooled Spearman rho over the scored test residues"
    for n, v in rho_floor.items():
        R.add(f"baseAaMeanRhoMin{n}", F(min(v), 3), f"{n} test set: lowest pooled Spearman rho of the per-amino-acid mean floor across the four training tiers", src_f, PREVIEW, "r5")  # fmt: skip
        R.add(f"baseAaMeanRhoMax{n}", F(max(v), 3), f"{n} test set: highest pooled Spearman rho of the per-amino-acid mean floor across the four training tiers", src_f, PREVIEW, "r5")  # fmt: skip


def sec_rev_results_a(C):
    """Numbers added by the results_a.tex revision (reviewer findings on R2 / R3).

    * CheZOD reproduction: median per-entry Pearson r per mode (the older rCheZODMedian comes
      from per_entry.csv, a pre-build LACS-on run, not from the potenci-only run R3 describes).
    * Within-residue k test (Fig. 2a): noise floor at dk = 0 and the disordered-consensus bins,
      computed with the figure's own functions.
    * LACS offsets in sigma(n) units, and how LACS and the residual offset behave in mostly
      disordered (mean G > 0.65) vs mostly ordered (mean G < 0.2) chains.
    * PANAV: median differences per nucleus, lists from disordered chains, lists outside the
      plotting window.
    """
    from scipy.stats import spearmanr

    from trizod.constants import REFINED_WEIGHTS

    def Fs(x):  # signed decimal; no "+" on a value that rounds to zero
        return F(x, 2, sign=abs(float(x)) >= 0.005)

    df = C.df
    G_ORD = 0.2  # mean-G cut for "mostly ordered" chains in the offset analyses
    R.add(
        "thrOrderedMeanG",
        F(G_ORD, 1),
        "mean-G cut below which a chain counts as mostly ordered in the offset analyses (mostly disordered: mean G > 0.65)",
        "scripts/manuscript/make_numbers.py sec_rev_results_a (analysis convention)",
        STABLE,
        "par",
    )
    dis = (df.meanG > G_DIS).to_numpy()
    ordd = (df.meanG < G_ORD).to_numpy()
    srcG = "Parquet gscores (chain mean over scored residues), lacs_off_*_ppm, off_*_sigma, total_off_*_ppm, temperature"
    R.add(
        "nChainsMeanGDisordered",
        I(dis.sum()),
        "chains with mean G > 0.65 (mostly disordered)",
        srcG,
        STABLE,
        "r2",
    )
    R.add(
        "nChainsMeanGOrdered",
        I(ordd.sum()),
        "chains with mean G < 0.2 (mostly ordered)",
        srcG,
        STABLE,
        "r2",
    )

    # ---- CheZOD reproduction: median per-entry Pearson r, per mode, sequence-identical entries
    s17 = paper("supp_s17_chezod_reproduction")
    cz = s17.load_chezod(s17.sp.CHEZOD_1325)
    for mode, path, status, tag, src_ in (
        ("PotenciOnly", s17.PO, PREVIEW, "re-referencing disabled (potenci-only), pre-2026-10 code",
         rel(s17.PO) + " (pre-2026-10 code; re-run on 2026-10) vs CheZOD1325, via supp_s17_chezod_reproduction.per_entry"),
        ("Both", s17.BOTH, STABLE, f"released mode (both: LACS + POTENCI/AIC), {REL} labels",
         rel(s17.BOTH) + f" ({REL} released labels) vs CheZOD1325, via supp_s17_chezod_reproduction.per_entry"),
    ):  # fmt: skip
        d = s17.per_entry(cz, s17.load_trizod(path), mode)
        e = d[d.seq_match.astype(bool)]
        R.add(
            f"rCheZODMedian{mode}",
            F(e.pearson.median(), 3),
            f"median per-entry Pearson r with published CheZOD Z over sequence-identical entries, {tag}",
            src_,
            status,
            "r3",
        )
        R.add(
            f"rCheZODQOne{mode}",
            F(e.pearson.quantile(0.25), 3),
            f"first quartile of that r, {tag}",
            src_,
            status,
            "r3",
        )
        R.add(
            f"rCheZODQThree{mode}",
            F(e.pearson.quantile(0.75), 3),
            f"third quartile of that r, {tag}",
            src_,
            status,
            "r3",
        )
        R.add(
            f"nCheZODSequenceIdentical{mode}",
            I(len(e)),
            f"sequence-identical entries in that comparison, {tag}",
            src_,
            status,
            "r3",
        )

    # ---- within-residue k test: Fig. 2a's own pairs (noise floor at dk = 0; disordered bins)
    f2a = paper("fig2a_kpaired")
    KP = f2a.residue_pairs(f2a.PARQUET)
    srcK = ("Parquet via scripts/figures/paper/fig2a_kpaired.py residue_pairs (Fig. 2a; ordered: replicate-consensus "
            "G < 0.3, disordered > 0.7, intermediate otherwise; <= 30 pairs per sequence, seed 0)")  # fmt: skip
    lab = {
        "0": "Zero",
        "1–3": "OneToThree",
        "4–6": "FourToSix",
        "7–9": "SevenToNine",
        "≥10": "TenPlus",
    }
    for state, St in (
        ("ordered", "Ordered"),
        ("disordered", "Disordered"),
        ("intermediate", "Intermediate"),
    ):
        s = KP[KP.state == state]
        for b_, B in lab.items():
            x = s[s.dkbin == b_]
            if state == "ordered" and B != "Zero":
                continue  # medians of the other ordered bins are in sec_replicates
            if state == "intermediate" and B not in ("Zero", "TenPlus"):
                continue
            for q, Q in ((0.25, "QOne"), (0.75, "QThree")):
                R.add(f"kpairDZ{St}{B}{Q}", F(x.dZ.quantile(q), 2),
                      f"{'first' if q == 0.25 else 'third'} quartile of dZ (higher-k minus lower-k), {state} consensus, dk bin {b_}"
                      + (" (replicate noise floor: pair orientation arbitrary at dk = 0)" if B == "Zero" else ""),
                      srcK, PREVIEW, "r2")  # fmt: skip
                R.add(f"kpairDG{St}{B}{Q}", F(x.dG.quantile(q), 3),
                      f"{'first' if q == 0.25 else 'third'} quartile of dG, {state} consensus, dk bin {b_}", srcK, PREVIEW, "r2")  # fmt: skip
            if state != "ordered":
                R.add(f"kpairDG{St}{B}", F(x.dG.median(), 3), f"median dG (higher-k minus lower-k), {state} consensus, dk bin {b_}",
                      srcK, PREVIEW, "r2")  # fmt: skip
                R.add(
                    f"kpairDZ{St}{B}",
                    F(x.dZ.median(), 2),
                    f"median dZ, {state} consensus, dk bin {b_}",
                    srcK,
                    PREVIEW,
                    "r2",
                )
                R.add(
                    f"nKPairs{St}{B}Bin",
                    I(len(x)),
                    f"residue pairs, {state} consensus, dk bin {b_}",
                    srcK,
                    PREVIEW,
                    "r2",
                )

    # ---- LACS offsets in sigma(n) units (sigma(n) = REFINED_WEIGHTS, the unit of the tier offset filters)
    srcL = "Parquet lacs_off_*_ppm; sigma(n) = trizod/constants.py REFINED_WEIGHTS (the POTENCI scale of the residual offset)"
    any2 = np.zeros(len(df), bool)
    for a in ["C", "CA", "CB", "H", "HA", "N"]:
        A = a.capitalize() if len(a) == 1 else a[0] + a[1:].lower()
        lac = df[lacs_off_ppm_col(a)].abs()
        m = (lac.notna() & (lac >= 2 * REFINED_WEIGHTS[a] - 1e-9)).to_numpy()
        any2 |= m
        R.add(f"nLacsAtLeastTwoSigma{A}", I(m.sum()), f"chains with |LACS offset| >= 2 sigma(n) on {a} (2 sigma = {2 * REFINED_WEIGHTS[a]:.3f} ppm)",
              srcL, STABLE, "off")  # fmt: skip
        R.add(f"pctLacsAtLeastTwoSigma{A}", P(100 * m.mean()), f"share of all scored chains with |LACS offset| >= 2 sigma(n) on {a}",
              srcL, STABLE, "off")  # fmt: skip
    R.add("nLacsAtLeastTwoSigmaAny", I(any2.sum()), "chains with |LACS offset| >= 2 sigma(n) on any nucleus (C, CA, CB, H, HA, N)",
          srcL, STABLE, "r2")  # fmt: skip
    R.add("pctLacsAtLeastTwoSigmaAny", P(100 * any2.mean()), "share of all scored chains with |LACS offset| >= 2 sigma(n) on any nucleus",
          srcL, STABLE, "r2")  # fmt: skip

    # ---- LACS 15N / 1HN offsets by disorder class (zero = no estimate or |intercept| < 0.005 ppm, so medians over non-zero)
    for a, A in (("N", "N"), ("H", "H")):
        lac = df[lacs_off_ppm_col(a)]
        nz = (lac.notna() & (lac != 0)).to_numpy()
        for cls, mk, Cl in (
            ("mean G > 0.65", dis, "Disordered"),
            ("mean G < 0.2", ordd, "Ordered"),
        ):
            v = lac[mk & nz]
            R.add(f"lacs{A}MedianNonzero{Cl}", Fs(v.median()),
                  f"median non-zero LACS {a} offset (ppm), chains with {cls}", srcG, STABLE, "r2" if a == "N" else "off")  # fmt: skip
            R.add(
                f"nLacs{A}Nonzero{Cl}",
                I(len(v)),
                f"chains with a non-zero LACS {a} offset and {cls}",
                srcG,
                STABLE,
                "off",
            )
            t = df[mk & nz & df.temperature.notna().to_numpy()]
            rho = spearmanr(t.temperature, t[lacs_off_ppm_col(a)]).statistic
            slope = np.polyfit(
                t.temperature.astype(float), t[lacs_off_ppm_col(a)].astype(float), 1
            )[0]
            R.add(f"rhoLacs{A}Temp{Cl}", F(rho, 2), f"Spearman rho, non-zero LACS {a} offset vs sample temperature, chains with {cls}",
                  srcG, STABLE, "off")  # fmt: skip
            R.add(f"slopeLacs{A}Temp{Cl}", F(1000 * slope, 1), f"least-squares slope of that offset on temperature (ppb/K), chains with {cls}",
                  srcG, STABLE, "off")  # fmt: skip
    rnz = np.zeros(len(df), bool)
    for a in ATOMS:
        sg = df[off_sigma_col(a)]
        rnz |= (sg.notna() & (sg != 0)).to_numpy()
    for cls, mk, Cl in (
        ("mean G > 0.65", dis, "Disordered"),
        ("mean G < 0.2", ordd, "Ordered"),
    ):
        R.add(f"pctPotenciNonzeroAny{Cl}", P(100 * rnz[mk].mean()),
              f"share of chains with {cls} that carry a non-zero residual POTENCI/AIC offset on any atom", srcG, STABLE, "r2")  # fmt: skip
    lnzN = (
        df[lacs_off_ppm_col("N")].notna() & (df[lacs_off_ppm_col("N")] != 0)
    ).to_numpy()
    R.add("totalNMedianNonzeroLacsDisordered", Fs(df[total_off_ppm_col("N")][dis & lnzN].median()),
          "median total (LACS + residual) N offset (ppm), chains with mean G > 0.65 and a non-zero LACS N offset", srcG, STABLE, "r2")  # fmt: skip
    sN = df[
        dis
        & lnzN
        & (df[off_sigma_col("N")].notna() & (df[off_sigma_col("N")] != 0)).to_numpy()
    ]
    R.add("rLacsVsResidualNDisordered", F(np.corrcoef(sN[lacs_off_ppm_col("N")], sN[off_sigma_col("N")] * REFINED_WEIGHTS["N"])[0, 1], 2),
          "Pearson r, LACS N offset vs residual N offset (ppm), chains with mean G > 0.65 and both non-zero", srcG, STABLE, "off")  # fmt: skip

    # ---- PANAV by disorder class and median differences (matched exactly as Fig. 2b)
    f2b = paper("fig2b_panav")
    pan = json.loads(PATHS.panav.read_text())
    stmap = f2b.loop_to_st_map(pan, 6)
    dfp = f2b.sp.load_main(
        ["id", "entry_id", "st_id", "total_bbshifts"]
        + [total_off_ppm_col(a) for a in f2b.ATOMS]
        + [lacs_off_ppm_col(a) for a in ATOMS]
    )
    dfp["entry_id"] = dfp.entry_id.astype(str)
    dfp["st_id"] = dfp.st_id.astype(str)
    M, _st = f2b.match_st(pan, stmap, dfp)
    M = M.merge(
        df[["id", "meanG"]].rename(columns={"id": "chain"}), on="chain", how="left"
    )
    srcP = (rel(PATHS.panav) + " x Parquet total_off_*_ppm / lacs_off_*_ppm, matched by (entry_id, st_id) with "
            "fig2b_panav.match_st; chain mean G from Parquet gscores")  # fmt: skip
    mdis = (M.meanG > G_DIS).to_numpy()
    mord = (M.meanG < G_ORD).to_numpy()
    R.add(
        "nPanavListsDisordered",
        I(mdis.sum()),
        "matched PANAV shift lists whose chain has mean G > 0.65",
        srcP,
        STABLE,
        "r2",
    )
    R.add(
        "nPanavListsOrdered",
        I(mord.sum()),
        "matched PANAV shift lists whose chain has mean G < 0.2",
        srcP,
        STABLE,
        "r2",
    )
    out_any = np.zeros(len(M), bool)
    for a in f2b.ATOMS:
        A = a.capitalize() if len(a) == 1 else a[0] + a[1:].lower()
        x = M[f"neg_panav_{a}"].to_numpy(float)
        y = M[f"trizod_total_{a}"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        L = f2b.LIMS[a]
        out_any |= m & ((np.abs(x) > L) | (np.abs(y) > L))
        d = y - x
        R.add(f"panavMedDiff{A}", Fs(np.median(d[m])),
              f"median TriZOD total {a} offset minus negated PANAV correction (ppm), all matched lists", srcP, STABLE, "r2")  # fmt: skip
        R.add(f"panavMedDiff{A}Disordered", Fs(np.median(d[m & mdis])),
              "same, lists from chains with mean G > 0.65", srcP, STABLE, "r2" if a == "N" else "off")  # fmt: skip
        R.add(f"panavMedDiff{A}Ordered", Fs(np.median(d[m & mord])),
              "same, lists from chains with mean G < 0.2", srcP, STABLE, "off")  # fmt: skip
    R.add("nPanavOutsideWindowAny", I(out_any.sum()),
          "matched lists outside the Fig. 2b window on at least one nucleus (|offset| > 4 ppm for 13C, > 6 ppm for 15N, either axis)",
          srcP, STABLE, "r2")  # fmt: skip
    s = M[mdis]
    R.add("panavNMedianDisordered", Fs(s.neg_panav_N.median()),
          "median negated PANAV N correction (ppm), lists from chains with mean G > 0.65", srcP, STABLE, "r2")  # fmt: skip
    R.add("lacsNMedianPanavDisordered", Fs(s.trizod_lacs_N.median()),
          "median LACS N offset (ppm) on the same lists", srcP, STABLE, "r2")  # fmt: skip
    R.add("totalNMedianPanavDisordered", Fs(s.trizod_total_N.median()),
          "median TriZOD total N offset (ppm) on the same lists", srcP, STABLE, "r2")  # fmt: skip
    R.add("rPanavNDisordered", F(f2b.pearson(s.neg_panav_N.to_numpy(float), s.trizod_total_N.to_numpy(float)), 2),
          "Pearson r, TriZOD total N offset vs negated PANAV correction, lists from chains with mean G > 0.65", srcP, STABLE, "off")  # fmt: skip
    R.add("panavMedAbsCaCbDiff", F((M.neg_panav_CA - M.neg_panav_CB).abs().median(), 2),
          "median |PANAV CA correction - PANAV CB correction| (ppm), all matched lists (TriZOD's LACS CA and CB offsets are one shared value)",
          srcP, STABLE, "off")  # fmt: skip
    # CheZOD's own code (github.com/protein-nmr/CheZOD, chezod1_1.py, getCheZODandPCs): minAIC = 6.0 with the
    # same comparison operators as trizod/scoring/scoring.py, but the choice between offset-corrected and
    # uncorrected scores compares means over Z-scores below 20 only (TriZOD: nanmean over all triplet Z).
    R.add("thrCheZODZAverageCap", I(20),
          "CheZOD averages only Z-scores below this value when choosing between offset-corrected and uncorrected scores (TriZOD has no cap)",
          "github.com/protein-nmr/CheZOD chezod1_1.py getCheZODandPCs: av0=average(cdfs30[cdfs30<20.0]), avc=average(cdfs3c[cdfs3c<20.0])",
          STABLE, "par")  # fmt: skip
    R.add("aicThresholdCheZOD", F(6.0, 0),
          "AIC threshold in CheZOD's own code (global: reject if dAIC < minAIC or fewer than 4 shifts; running: accept if dAIC > minAIC), identical to TriZOD's",
          "github.com/protein-nmr/CheZOD chezod1_1.py getCheZODandPCs(minAIC=6.0), visresults", STABLE, "par")  # fmt: skip


# Pre-fix leakage values of the FROZEN 2026-10 split (pipeline trizod-5b8147a, before rebuild fix
# (a)), pinned here so that the paper's account of the containment leak survives the rebuild: the
# live \nLeakTestCovCommon*, \nLeakBidirCommon*, \idContain*, \lenContain* and \nSubstring* macros
# are defined on the CURRENT split and become 0 (or change) once the rebuild lands. While the
# frozen build is current, sec_results_b_revision() checks every pinned value against the live one.
RB_FROZEN_BUILD = "trizod-5b8147a"
RB_PREFIX_PINNED = {
    # live macro: (pinned value, definition of the frozen quantity)
    **{f"nLeakTestCovCommon{s}{T}": (v, f"{s} test chains with a {T.lower()} training representative at identity >= 30 % over >= 80 % of the TEST chain, build alignment block (Fig. 3c1 search), FROZEN 2026-10 split before fix (a)")
       for s, vals in (("TriZOD", ("123", "120", "99", "64")), ("CheZOD", ("22", "21", "16", "10")))
       for T, v in zip(("Unfiltered", "Tolerant", "Moderate", "Strict"), vals)},
    **{f"nLeakBidirCommon{s}{T}": (v, f"{s} test chains with a {T.lower()} training representative at identity >= 30 % over >= 80 % of BOTH sequences (the pipeline's own criterion, which its stage-2 search nonetheless missed), build alignment block, FROZEN 2026-10 split")
       for s, vals in (("TriZOD", ("11", "11", "10", "8")), ("CheZOD", ("1", "1", "1", "0")))
       for T, v in zip(("Unfiltered", "Tolerant", "Moderate", "Strict"), vals)},
    "idContainTestOne": ("51709\\_1\\_1\\_1", "verified containment One (FROZEN 2026-10 split): test chain"),
    "idContainRepOne": ("53339\\_1\\_1\\_1", "verified containment One (FROZEN 2026-10 split): tolerant training representative containing it, 100 % identity"),
    "lenContainTestOne": ("169", "verified containment One (FROZEN 2026-10 split): test chain length (aa)"),
    "lenContainRepOne": ("419", "verified containment One (FROZEN 2026-10 split): representative length (aa)"),
    "idContainTestTwo": ("19221\\_1\\_1\\_1", "verified containment Two (FROZEN 2026-10 split): test chain"),
    "idContainRepTwo": ("19219\\_1\\_1\\_1", "verified containment Two (FROZEN 2026-10 split): moderate training representative"),
    "lenContainTestTwo": ("176", "verified containment Two (FROZEN 2026-10 split): test chain length (aa)"),
    "lenContainRepTwo": ("401", "verified containment Two (FROZEN 2026-10 split): representative length (aa)"),
    "idContainTestThree": ("15813\\_1\\_1\\_1", "verified containment Three (FROZEN 2026-10 split): test chain"),
    "idContainRepThree": ("30456\\_1\\_1\\_1", "verified containment Three (FROZEN 2026-10 split): moderate training representative"),
    "lenContainTestThree": ("117", "verified containment Three (FROZEN 2026-10 split): test chain length (aa)"),
    "lenContainRepThree": ("443", "verified containment Three (FROZEN 2026-10 split): representative length (aa)"),
    "nSubstringPairs": ("28", "training-representative / test-chain pairs where one sequence is an exact substring of the other, FROZEN 2026-10 split"),
    "nSubstringTestChains": ("22", "test chains in such exact containment, FROZEN 2026-10 split"),
    "nSubstringTestChainsTriZOD": ("15", "of them TriZOD test chains, FROZEN 2026-10 split"),
    "nSubstringTestChainsCheZOD": ("7", "of them CheZOD117 chains, FROZEN 2026-10 split"),
    "fidentMissPair": ("79.8\\%", "identity of the bidirectional-rule miss pair 27177_1_1_1 (train, 124 aa) / 27176_1_1_1 (test, 99 aa), FROZEN 2026-10 split"),
    "lenRatioMissPair": ("0.798", "length ratio (test / train) of that pair, FROZEN 2026-10 split"),
}  # fmt: skip


def _rb_prefix_name(live: str) -> str:
    for a, b in (("nLeakTestCovCommon", "nLeakTestCovPreFix"), ("nLeakBidirCommon", "nLeakBidirPreFix"),
                 ("idContain", "idContainPreFix"), ("lenContain", "lenContainPreFix"), ("nSubstring", "nSubstringPreFix"),
                 ("fidentMissPair", "fidentMissPairPreFix"), ("lenRatioMissPair", "lenRatioMissPairPreFix")):  # fmt: skip
        if live.startswith(a):
            return b + live[len(a) :]
    raise KeyError(live)


def sec_results_b_revision(C):
    """Numbers added in the results_b.tex revision (31 Oct; reviewers 'numbers', 'ml_expert',
    'nmr_expert', 'audit_consistency').

    (1) *PreFix: the pre-fix leakage of the frozen 2026-10 split, pinned (RB_PREFIX_PINNED) so
        that R5's account of the containment leak keeps its values after the rebuild. STABLE.
    (2) The alpha-synuclein exclusions split by first exclusion rule (bound > short > exact copy
        of a test sequence > pool copy > homolog, as in Supplementary Table T2). Needs the
        cached case search of sec_ubiquitin.
    (3) The Fig. 3b1 condition contrast restricted to pairs whose chains do not differ in sample
        state (solid-state evidence, cosolvent evidence, a strict-denied physical state or a
        membrane mimetic), computed with the figure's own pair table and bootstrap.
    (4) Fig. 3b2 chain total (deviant + consistent).
    (5) Replicate agreement of the test chains themselves (Supplementary Fig. S10 diamonds).
    (6) Disorder concentration in the TriZOD test set and the disorder share of strict sequences
        in test-draw clusters with and without a CheZOD member.
    (7) High-pressure chains without the 25704 unit slip ('1013 bar' = 1 atm in mbar).
    (8) The old DisProt figure's '891 matches', attributed to the old Fig. 3b (tolerant set)."""
    cur = {it.name: it for it in R.items}
    on_frozen = C.manifest.get("pipeline_version") == RB_FROZEN_BUILD
    for live, (val, d) in RB_PREFIX_PINNED.items():
        if on_frozen and live in cur:
            assert cur[live].value == val, (live, cur[live].value, val)
        R.add(_rb_prefix_name(live), val, d, f"pinned in make_numbers.py RB_PREFIX_PINNED (checked against \\{live} while {RB_FROZEN_BUILD} is the current build)", STABLE, "r5")  # fmt: skip

    df = C.df
    # (2) alpha-synuclein exclusions by first rule
    seqs = df.set_index("id").sequence
    hits = mmseqs_search("cases", [(k, seqs[v]) for k, v in {"Ubq": "50330_1_1_1", "Prion": "16616_1_1_1", "CytC": "25640_1_1_1", "ASyn": "6968_1_1_1"}.items()],
                         list(zip(df.id, df.sequence)), ["-s", "7.5", "--max-seqs", "20000", "-e", "1e-3"], C.refresh)  # fmt: skip
    H = pd.DataFrame(
        hits, columns=["q", "t", "fid", "qcov", "tcov", "e", "aln", "ql", "tl"]
    )
    H = H[(H.fid >= 0.8) & (H.qcov >= 0.9) & (H.tcov >= 0.9) & (H.q == "ASyn")]
    ex = df[df.id.isin(set(H.t)) & (df.split == "excluded")]
    tests_seq = set(df.loc[df.split.str.startswith("test"), "sequence"])
    pool_seq = set(df.loc[df.pool_tier.notna(), "sequence"])
    bound = ex.is_bound.fillna(False).astype(bool)
    short = ~bound & (ex.length < 20)
    tcopy = ~bound & ~short & ex.sequence.isin(tests_seq)
    pcopy = ~bound & ~short & ~tcopy & ex.sequence.isin(pool_seq)
    homol = ~bound & ~short & ~tcopy & ~pcopy
    assert not short.any() and not pcopy.any(), (
        "alpha-synuclein decomposition: unexpected short or pool-copy exclusions"
    )
    srca = "mmseqs case cluster (>= 80 % id, >= 90 % query and target coverage to 6968_1_1_1; sec_ubiquitin) x Parquet split, is_bound, sequence; first exclusion rule as in Supplementary Table T2"
    R.add("nCaseASynExcludedBound", I(bound.sum()), "excluded alpha-synuclein cluster chains whose first exclusion rule is 'bound'", srca, PENDING, "r5")  # fmt: skip
    R.add("nCaseASynExcludedTestCopy", I(tcopy.sum()), "excluded alpha-synuclein cluster chains (not bound) that are exact copies of a test sequence (the CheZOD117 chain 6968_1_1_1)", srca, PENDING, "r5")  # fmt: skip
    R.add("nCaseASynExcludedHomolog", I(homol.sum()), "excluded alpha-synuclein cluster chains removed as test-set homologs (neither bound nor an exact copy)", srca, PENDING, "r5")  # fmt: skip
    R.add("nCaseASynExcludedHomologSeqs", I(ex[homol].sequence.nunique()), "distinct sequences among those homolog exclusions", srca, PENDING, "r5")  # fmt: skip

    # (3) condition contrast without state differences (Fig. 3b1 definition otherwise)
    from trizod.bmrb.sample_state import (
        PHYSICAL_STATE_STRICT_DENY,
        denied_physical_states,
    )

    fr = paper("fig3_replicates")
    D = fr.Data()
    Cp = fr.condition_pairs(D)
    nb = D.nb.set_index("id")
    cor = nb["cosolvent_evidence"].fillna(False).astype(bool)
    memb = df.set_index("id")["membrane_mimetic"].reindex(nb.index).notna()
    sig = pd.DataFrame(
        {
            "state": np.asarray(denied_physical_states(nb["physical_state"], PHYSICAL_STATE_STRICT_DENY, corroborated=cor.to_numpy()), bool),
            "cosolvent": cor.to_numpy(),
            "solid": (nb["sample_state_evidence"].astype(str) == "solid").to_numpy(),
            "membrane": memb.to_numpy(),
        },
        index=nb.index,
    )  # fmt: skip
    diff_state = (sig.loc[Cp.a].to_numpy() != sig.loc[Cp.b].to_numpy()).any(axis=1)
    clean = Cp[~diff_state].reset_index(drop=True)
    raw_a, mat_a = (
        fr.boot_conditions(Cp, matched=False),
        fr.boot_conditions(Cp, matched=True),
    )
    raw_c, mat_c = (
        fr.boot_conditions(clean, matched=False),
        fr.boot_conditions(clean, matched=True),
    )
    srcs = "scripts/figures/paper/fig3_replicates.py condition_pairs + boot_conditions (Fig. 3b1 definition), pairs whose chains differ in solid-state evidence, cosolvent evidence, a strict-denied physical state or a membrane mimetic removed"
    for cond, T in (("same", "Same"), ("different", "Diff")):
        R.add(f"nCondPairsFig{T}StateDiff", I(int((Cp.cond == cond).to_numpy()[diff_state].sum())), f"{cond}-condition pairs (Fig. 3b1) whose chains differ in sample state", srcs, STABLE, "r4")  # fmt: skip
        R.add(f"nCondPairsFig{T}StateClean", I(raw_c[cond]["n_pairs"]), f"{cond}-condition pairs without a sample-state difference", srcs, STABLE, "r4")  # fmt: skip
    R.add("condAbsDGFigSameStateClean", F(raw_c["same"]["median_absdg"], 3), "median per-pair mean |dG|, same conditions, pairs without a state difference", srcs, STABLE, "r4")  # fmt: skip
    R.add("condAbsDGFigDiffMatchedStateClean", F(mat_c["different"]["median_absdg"], 3), "median per-pair mean |dG|, different conditions, shift-count matched, pairs without a state difference", srcs, STABLE, "r4")  # fmt: skip
    rr = mat_c["ratio_median_absdg"]
    R.add("condRatioFigMatchedStateClean", F(rr["est"], 2), "ratio of median per-pair |dG|, different / same conditions, shift-count matched, pairs without a state difference", srcs, STABLE, "r4")  # fmt: skip
    R.add("condRatioFigMatchedStateCleanLo", F(rr["lo"], 2), "lower 95 % sequence-bootstrap bound of that ratio", srcs, STABLE, "r4")  # fmt: skip
    R.add("condRatioFigMatchedStateCleanHi", F(rr["hi"], 2), "upper 95 % sequence-bootstrap bound of that ratio", srcs, STABLE, "r4")  # fmt: skip
    R.add("condMaeRatioFigMatched", F(mat_a["different"]["mae"] / raw_a["same"]["mae"], 2), "ratio of pooled residue-level MAE, different (shift-count matched) / same conditions, all Fig. 3b1 pairs", srcs, STABLE, "r4")  # fmt: skip
    R.add("condMaeRatioFigMatchedStateClean", F(mat_c["different"]["mae"] / raw_c["same"]["mae"], 2), "same ratio of pooled residue-level MAE, pairs without a state difference", srcs, STABLE, "r4")  # fmt: skip
    R.add("condRFigSameStateClean", F(raw_c["same"]["r"], 3), "pooled residue-level r, same conditions, pairs without a state difference", srcs, STABLE, "r4")  # fmt: skip
    R.add("condRFigDiffMatchedStateClean", F(mat_c["different"]["r"], 3), "pooled residue-level r, different conditions, shift-count matched, pairs without a state difference", srcs, STABLE, "r4")  # fmt: skip

    # (4) Fig. 3b2 chain total
    def _int(name):
        return int(plainify(cur[name].value).replace(",", ""))

    R.add("nUnfOnlyFig", I(_int("nDeviantFig") + _int("nConsistentFig")), "non-bound replicate chains scored only at the unfiltered tier with a leave-one-entry-out deviation (Fig. 3b2, deviant + consistent)", "\\nDeviantFig + \\nConsistentFig (fig3b_conditions.py flag_table)", PENDING, "r4")  # fmt: skip

    # (5) replicate agreement of the test chains themselves (Supplementary Fig. S10 diamonds)
    s10 = paper("supp_s10_baselines")
    import style_paper as _sp

    TC = s10.test_ceilings(_sp.load_main(s10.COLUMNS)).set_index("test_set")
    src10 = "scripts/figures/paper/supp_s10_baselines.py test_ceilings (test chain vs non-bound identical-sequence chains of other entries outside the test sets, <= 20 per test chain, seed 0)"
    for name, T in (("TriZOD test", "TriZOD"), ("CheZOD117", "CheZOD")):
        r_ = TC.loc[name]
        R.add(f"rhoRepTest{T}", F(r_.rho_res, 3), f"residue-level Spearman rho between {name} chains and their own replicates (Supplementary Fig. S10)", src10, STABLE, "r5")  # fmt: skip
        R.add(f"rRepTest{T}", F(r_.pearson_res, 3), f"residue-level Pearson r between {name} chains and their own replicates", src10, STABLE, "r5")  # fmt: skip
        R.add(f"nRepChainsTest{T}", I(r_.n_test_chains_with_replicates), f"{name} chains with at least one replicate outside the test sets", src10, STABLE, "r5")  # fmt: skip
        R.add(f"nRepPairsTest{T}", I(r_.n_pairs), f"test-chain / replicate pairs behind those values, {name}", src10, STABLE, "r5")  # fmt: skip

    # (6) disorder concentration in the TriZOD test set; draw clusters with / without CheZOD
    def _dis(sub):
        n_ = d_ = 0
        per = []
        for g, m in zip(sub.gscores, sub["mask"]):
            g = np.asarray(g, float)
            m = np.asarray(m, bool) & np.isfinite(g)
            per.append((int(m.sum()), int((g[m] > G_DIS).sum())))
        per = np.array(per, int).reshape(-1, 2)
        n_, d_ = per.sum(0)
        return n_, d_, per

    tz = df[df.split == "test_trizod"]
    n_, d_, per = _dis(tz)
    mostly = per[:, 1] > 0.5 * per[:, 0]
    srcd = "Parquet gscores x mask, split = test_trizod"
    R.add("nResDisorderedSetTestTriZOD", I(d_), "scored residues with G > 0.65 in the TriZOD test set", srcd, STABLE, "r5")  # fmt: skip
    R.add("pctDisorderedInMostlyDisorderedTestTriZOD", P(100 * per[mostly, 1].sum() / d_, 0), "share of the TriZOD test set's disordered residues (G > 0.65) that lie in its chains more than half disordered (\\nChainsMostlyDisorderedTestTriZOD)", srcd, STABLE, "r5")  # fmt: skip
    R.add("nChainsAnyDisorderedTestTriZOD", I((per[:, 1] > 0).sum()), "TriZOD test chains with at least one residue at G > 0.65", srcd, STABLE, "r5")  # fmt: skip
    clu = pd.read_csv(
        PATHS.draw / "_tmp" / "c30_cluster.tsv",
        sep="\t",
        header=None,
        names=["rep", "mem"],
    )
    clu["cz"] = clu.mem.str.upper().str.startswith("CHEZOD")
    clu["clu_cz"] = clu.rep.map(clu.groupby("rep").cz.any())
    strict_mem = clu[~clu.cz]
    did = df.set_index("id")
    srcc = "testset/_tmp/c30_cluster.tsv (test-set draw, 30 % id / 80 % cov) x Parquet labels (unfiltered = strict labels for these chains)"
    gone = ~strict_mem.mem.isin(did.index)
    if C.manifest.get("pipeline_version") == RB_FROZEN_BUILD:
        assert not gone.any()
    elif gone.any():
        # draw members that a later build no longer releases (withdrawn, placeholder or
        # re-filtered chains) carry no labels; they are left out of the shares below
        R.add("nDrawStrictMembersNotReleased", I(gone.sum()), "strict sequences of the 2026-08-11 test-set draw that this build no longer releases (left out of the draw-cluster disorder shares)", srcc, STABLE, "r5")  # fmt: skip
        strict_mem = strict_mem[~gone]
    for flag, tag, dd in (
        (True, "CheZOD", "with a CheZOD member"),
        (False, "CheZODFree", "without a CheZOD member"),
    ):
        ids = strict_mem[strict_mem.clu_cz == flag].mem
        n2, d2, _ = _dis(did.loc[ids])
        R.add(f"pctDisorderedDrawClusters{tag}", P(100 * d2 / n2), f"share of scored residues with G > 0.65 among the strict sequences of test-draw clusters {dd}", srcc, STABLE, "r5")  # fmt: skip
        R.add(f"nSeqsDrawClusters{tag}", I(len(ids)), f"strict sequences in test-draw clusters {dd}", srcc, STABLE, "r5")  # fmt: skip

    # (7) high-pressure chains without unit slips (S4 loader; 25704 records '1013 bar' = 1 atm in mbar)
    s4 = paper("supp_s4_conditions")
    d4, _ = s4.load()
    slip = d4.pressure_bar.between(
        1000.0, 1030.0
    )  # 1 atm written in mbar but labelled bar
    hp = d4[(d4.pressure_bar >= s4.HIGH_P) & ~slip]
    src4 = "raw NMR-STAR condition list of each chain's shift table (supp_s4_conditions.py load); values of 1,000-1,030 'bar' (25704: 1013) read as 1 atm in mbar"
    for t_ in TIERS:
        R.add(f"nPressureHighChainsNoSlip{TCAP[t_]}", I(hp[f"in_{t_}"].sum()), f"chains of the {t_} tier whose own condition list records >= 400 bar, unit slips excluded", src4, STABLE if t_ == "unfiltered" else PENDING, "r4")  # fmt: skip
    R.add("nPressureHighEntriesNoSlipScored", I(hp.loc[hp.in_unfiltered, "entry_id"].nunique()), "scored entries with a chain measured at >= 400 bar, unit slips excluded", src4, STABLE, "r4")  # fmt: skip

    # (8) old DisProt figure label
    R.add("nDisProtMatchesOld", I(891), "DisProt matches in the title of the old Fig. 3b ('tolerant' set; Article.tex master), the denominator of the old 0.76 accuracy", "old manuscript Fig. 3b title (audit report, refuted claim 13: not the cytochrome-c PNG)", PROVISIONAL, "r6")  # fmt: skip


def sec_fig1_revision(C):
    """Numbers added in the fig1.tex revision (31 Oct; reviewer nmr_expert, Fig. 1a k classes).

    \\nUbqDiUbChains: of the Fig. 1a core-strip chains in the lowest median-k class (non-bound
    ubiquitin, mean G < 0.5, chain median k <= 6; the \\nUbqClassKLow set of sec_ubiquitin),
    those that are subunits of covalently linked diubiquitins (n_copies >= 2) and carry only
    1H and 15N backbone shifts (trizod_shifts.parquet, is_backbone atoms). PREVIEW like the
    other teaser numbers, because the Fig. 1a chain selection is under review."""
    df = C.df
    u = df[(df.sequence == UBQ) & ~df.is_bound]
    medk = np.array(
        [
            np.nanmedian(
                np.where(
                    np.isfinite(np.asarray(g, float)), np.asarray(k, float), np.nan
                )
            )
            for g, k in zip(u.gscores, u.k)
        ]
    )
    low = u[(u.meanG.to_numpy() < 0.5) & (medk > 0) & (medk <= 6)]
    sh = pq.read_table(
        PATHS.shifts,
        columns=["id", "atom_id", "is_backbone"],
        filters=[("id", "in", list(low["id"]))],
    ).to_pandas()
    atoms = sh[sh.is_backbone].groupby("id")["atom_id"].agg(lambda s: set(s))
    hn_only = {i for i, a in atoms.items() if a <= {"H", "N"}}
    diub = low[(low.n_copies >= 2) & low["id"].isin(hn_only)]
    src = ("Parquet sequence, is_bound, gscores, k, n_copies (non-bound ubiquitin, mean G < 0.5, median k <= 6); "
           "trizod_shifts.parquet atom_id of is_backbone shifts (only H and N); entries 30234, 52078, 52108, 52115, 52139")  # fmt: skip
    R.add("nUbqDiUbChains", I(len(diub)), "Fig. 1a core-strip chains with median k <= 6 that are 1H-15N-only subunits of covalently linked diubiquitins (n_copies >= 2)", src, PREVIEW, "r2")  # fmt: skip
    R.add("nUbqDiUbEntries", I(diub.entry_id.nunique()), "diubiquitin entries behind \\nUbqDiUbChains", src, PREVIEW, "r2")  # fmt: skip


def _scan_temperature_units() -> dict:
    """Raw NMR-STAR temperature rows per sample-condition list: {entry: {list_id: [val, unit]}}.

    Mirrors trizod/bmrb/bmrb.py SampleConditions: the last _Sample_condition_variable row whose
    Type contains 'temp' (and is not ionic strength, pH or pressure). Text parse, no pynmrstar."""
    rs = paper("paper_raw_scan")
    out = {}
    for d in sorted(os.listdir(PATHS.raw)):
        fs = sorted((PATHS.raw / d).glob("*.str"))
        if not fs:
            continue
        txt = fs[0].read_text(errors="ignore")
        rec = {}
        for chunk in rs._FRAME_SPLIT.split(txt):
            if "_Sample_condition_list.ID" not in chunk:
                continue
            m = rs._COND_ID.search(chunk)
            if not m:
                continue
            temp = None
            for row in rs._loop_rows(chunk, "Sample_condition_variable"):
                t = row.get("Type", "").lower()
                if "ionic strength" in t or t in ("ph", "pressure"):
                    continue
                if "temp" in t:
                    temp = [row.get("Val"), row.get("Val_units", "")]
            rec[m.group(1)] = temp
        out[d.replace("bmr", "")] = rec
    return out


def sec_methods_a_revision(C):
    """Numbers added in the methods_a.tex revision (31 Oct; reviewers nmr_expert, numbers,
    audit_consistency, format).

    * LACS: the two-line thresholds and the per-line minimum (code constants of
      trizod/lacs/lacs.py); the applied HB offset (\\nLacsCaCbDiffer exists in sec_offsets).
    * Window interpolation: compute_gscores runs on the triplet data, so a residue inside the
      measured span without comparable shifts of its own is scored from i-1 and i+1. The own
      count m(i) is recovered exactly from the released k by k(i) = m(i-1) + m(i) + m(i+1) with
      m(0) = 0 (POTENCI predicts nothing for the first residue); the recurrence is checked
      against the last residue of every chain.
    * Residual (POTENCI/AIC) offsets estimated per atom type: CA and CB of opposite sign
      (helical sense: CA up, CB down) and the share of chains with any non-zero residual
      offset by mean-G class. Unfiltered-run offsets of all scored chains (Parquet).
    * Unit rules of SampleConditions.get_temperature: the 50-100 'not C/F' -> Fahrenheit rule
      and the 1-50 -> Celsius rule, counted on the raw condition list of each scored chain.
    * Test-set containment facts: a CheZOD117 chain inside a TriZOD test chain, a test chain
      that contains shorter training representatives (exact substrings >= 20 aa), and the
      diagnosis of the frozen stage-2 misses under the bidirectional criterion (pairs whose
      length ratio is below 0.8, which mmseqs skips before alignment when -c 0.8 is set)."""
    df = C.df
    srcq = rel(PATHS.pq)
    # --- LACS constants and CA/CB agreement ------------------------------------------------
    lacs_src = (REPO / "trizod" / "lacs" / "lacs.py").read_text()
    thr_cacb = float(
        re.search(r"two_line_threshold: float = ([0-9.]+)", lacs_src).group(1)
    )
    thr_coha = sorted(
        {float(x) for x in re.findall(r"two_line_threshold=([0-9.]+)", lacs_src)}
    )
    assert len(thr_coha) == 1, thr_coha
    n_line = int(re.search(r"^_MIN_POINTS_PER_LINE = (\d+)", lacs_src, re.M).group(1))
    R.add("lacsTwoLineThrCaCb", F(thr_cacb, 0), "LACS two-line split of the regressor for CA and CB (ppm): lines fitted to residues below +t and above -t", "trizod/lacs/lacs.py _compute_13c_offset two_line_threshold default", STABLE, "par")  # fmt: skip
    R.add("lacsTwoLineThrCoHa", F(thr_coha[0], 0), "LACS two-line split of the regressor for C' and HA (ppm)", "trizod/lacs/lacs.py compute_lacs_offsets two_line_threshold=6.0", STABLE, "par")  # fmt: skip
    R.add("lacsMinPointsPerLine", I(n_line), "minimum residues on each side of the split for a two-line LACS fit (else one line)", "trizod/lacs/lacs.py _MIN_POINTS_PER_LINE", STABLE, "par")  # fmt: skip
    R.add("nLacsHbNonzero", I((df[lacs_off_ppm_col("HB")].fillna(0) != 0).sum()), "chains with a non-zero applied LACS HB offset (scoring._LACS_ATOMS omits HB although lacs.py copies the HA offset)", f"{srcq} lacs_off_HB_ppm", STABLE, "r2")  # fmt: skip

    # --- residues scored without a shift of their own ---------------------------------------
    tot = no_own = no_own_pro = bad = 0
    for seq, k, m in zip(df.sequence, df.k, df["mask"]):
        k = np.asarray(k, int)
        m = np.asarray(m, bool)
        n = len(k)
        own = np.zeros(n, int)
        if n > 1:
            own[1] = k[0]
        for i in range(1, n - 1):
            own[i + 1] = k[i] - own[i - 1] - own[i]
        if (
            (own < 0).any()
            or (own > len(ATOMS)).any()
            or (n > 1 and k[n - 1] != own[n - 2] + own[n - 1])
        ):
            bad += 1
            continue
        s = np.array(list(seq))
        tot += m.sum()
        nm = m & (own == 0)
        no_own += nm.sum()
        no_own_pro += (nm & (s == "P")).sum()
    assert bad == 0, f"own-shift recurrence failed for {bad} chains"
    src_k = f"{srcq} k, mask (own count m(i) recovered from k(i) = m(i-1) + m(i) + m(i+1), m(0) = 0; checked at the last residue of every chain)"
    R.add("nResiduesNoOwnShift", I(no_own), "scored residues without a comparable shift of their own (label interpolated from i-1 and i+1)", src_k, STABLE, "r2")  # fmt: skip
    R.add("pctResiduesNoOwnShift", P(100 * no_own / tot), "share of scored residues without a shift of their own", src_k, STABLE, "r2")  # fmt: skip
    R.add("pctNoOwnShiftProline", P(100 * no_own_pro / no_own, 0), "share of those residues that are prolines", src_k, STABLE, "r2")  # fmt: skip

    # --- residual offsets: CA/CB sign and share by mean-G class -------------------------------
    from trizod.constants import REFINED_WEIGHTS

    w = REFINED_WEIGHTS
    ca, cb = df[off_sigma_col("CA")].fillna(0), df[off_sigma_col("CB")].fillna(0)
    both = (ca != 0) & (cb != 0)
    opp = both & (np.sign(ca) != np.sign(cb))
    hel = both & (ca > 0) & (cb < 0)
    src_o = f"{srcq} off_CA_sigma, off_CB_sigma (unfiltered-run residual POTENCI/AIC offsets, all scored chains) x REFINED_WEIGHTS; mean G over scored residues"
    R.add("nResidCaCbBoth", I(both.sum()), "chains with non-zero residual offsets on both CA and CB", src_o, STABLE, "off", "467")  # fmt: skip
    R.add("nResidCaCbOpposite", I(opp.sum()), "of them with opposite signs", src_o, STABLE, "off", "164")  # fmt: skip
    R.add("nResidCaCbHelix", I(hel.sum()), "of them in the helical sense (CA positive, CB negative)", src_o, STABLE, "off", "154")  # fmt: skip
    R.add("ppmResidCaOpposite", F((ca[opp] * w["CA"]).median(), 2, sign=True), "median CA residual offset (ppm) of the opposite-sign chains", src_o, STABLE, "off", "+0.43")  # fmt: skip
    R.add("ppmResidCbOpposite", F((cb[opp] * w["CB"]).median(), 2, sign=True), "median CB residual offset (ppm) of the opposite-sign chains", src_o, STABLE, "off", "-0.38")  # fmt: skip
    R.add("medGResidOpposite", F(df.meanG[opp].median(), 2), "median chain mean G of the opposite-sign chains", src_o, STABLE, "off", "0.34")  # fmt: skip
    R.add("medGResidSame", F(df.meanG[both & ~opp].median(), 2), "median chain mean G of the same-sign chains", src_o, STABLE, "off", "0.70")  # fmt: skip
    R.add("nResidCaCbOppositeReps", I((opp & (df.split == "train")).sum()), "opposite-sign chains that are training representatives (frozen split)", src_o, PENDING, "off")  # fmt: skip
    # Shares for the disordered and ordered classes exist as \\pctPotenciNonzeroAny{Disordered,Ordered}
    # (sec_rev_results_a); only the intermediate class is added here, with the same definition.
    g_ord = 0.2  # = \\thrOrderedMeanG
    rnz = (df[[off_sigma_col(a) for a in ATOMS]].fillna(0) != 0).any(axis=1)
    mid = (df.meanG >= g_ord) & (df.meanG <= G_DIS)
    R.add("nChainsMeanGIntermediate", I(mid.sum()), f"chains with {g_ord} <= mean G <= {G_DIS} (intermediate)", src_o, STABLE, "r4")  # fmt: skip
    R.add("pctPotenciNonzeroAnyIntermediate", P(100 * rnz[mid].mean()), f"share of chains with {g_ord} <= mean G <= {G_DIS} that carry a non-zero residual POTENCI/AIC offset on any atom", src_o, STABLE, "r4", "66")  # fmt: skip

    # --- temperature unit rules ---------------------------------------------------------
    if PATHS.raw.is_dir():
        T = cached_json(
            "temp_units_scan.json",
            _scan_temperature_units,
            getattr(C, "refresh", False),
        )
        rs = paper("paper_raw_scan")
        st = rs.scan_raw()

        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return np.nan

        n_f_lists = 0
        for ent in T.values():
            for t_ in ent.values():
                if (
                    t_
                    and not ("C" in (t_[1] or "") or "F" in (t_[1] or ""))
                    and 50 <= num(t_[0]) <= 100
                ):
                    n_f_lists += 1
        raw_t = []
        for e, s in zip(df.entry_id.astype(str), df.st_id.astype(str)):
            lid = st.get(e, {}).get("st", {}).get(s)
            raw_t.append(T.get(e, {}).get(lid) if lid else None)
        val = np.array([num(t_[0]) if t_ else np.nan for t_ in raw_t])
        cf = np.array(
            [
                bool(t_) and ("C" in (t_[1] or "") or "F" in (t_[1] or ""))
                for t_ in raw_t
            ]
        )
        c_rule = (~cf) & (val >= 1) & (val < 50)
        src_t = "raw NMR-STAR _Sample_condition_variable temperature rows (value, unit) of each chain's condition list; rule of trizod/bmrb/bmrb.py SampleConditions.get_temperature (fix_outliers)"
        R.add("nCondListsFahrenheitRule", I(n_f_lists), "snapshot condition lists with a temperature of 50-100 not marked C or F (read as Fahrenheit up to tolerant)", src_t, STABLE, "r4")  # fmt: skip
        for t_ in ("unfiltered", "tolerant"):
            R.add(f"nChainsCelsiusRule{TCAP[t_]}", I((c_rule & df["id"].isin(C.tier_ids[t_]).to_numpy()).sum()), f"{t_}-tier chains whose temperature (1-50, not marked C or F) is read as Celsius", src_t, STABLE if t_ == "unfiltered" else PENDING, "r4")  # fmt: skip
    else:
        R.unrepro("nCondListsFahrenheitRule", "0", "snapshot condition lists hit by the Fahrenheit rule", "raw BMRB files not found", "r4")  # fmt: skip

    # --- test-set containment -----------------------------------------------------------
    d = df.set_index("id")
    tz = d[d.split == "test_trizod"]
    cz = d[d.split == "test_chezod117"]
    tr = d[d.split == "train"]
    pairs = [
        (a, b)
        for a, sa in cz.sequence.items()
        for b, sb in tz.sequence.items()
        if sa != sb and (sa in sb or sb in sa)
    ]
    assert len(pairs) == 1, pairs
    inner, outer = (
        pairs[0]
        if len(cz.sequence[pairs[0][0]]) < len(tz.sequence[pairs[0][1]])
        else pairs[0][::-1]
    )
    src_c = f"{srcq} sequence, split (exact substring between the two test sets)"
    R.add("nTestSetsContained", I(len(pairs)), "CheZOD117 / TriZOD test chain pairs in which one sequence lies inside the other (no identical sequence)", src_c, STABLE, "r5")  # fmt: skip
    R.add("idTestContainedInner", S(inner), "the CheZOD117 chain that lies verbatim inside a TriZOD test chain", src_c, STABLE, "r5")  # fmt: skip
    R.add("idTestContainedOuter", S(outer), "the TriZOD test chain that contains it", src_c, STABLE, "r5")  # fmt: skip
    gate = 20  # = \\substringGateMin (sec_params); planned gate of fix (a), leakage-fix branch 98db69e
    tests = pd.concat([tz, cz])
    rev = [
        (b, i)
        for b, sb in tests.sequence.items()
        for i, si in tr.sequence.items()
        if len(si) >= gate and len(si) < len(sb) and si in sb
    ]
    src_r = f"{srcq} sequence, split (training representatives of >= {gate} aa that are exact substrings of a longer test chain)"
    on_frozen = C.manifest.get("pipeline_version") == RB_FROZEN_BUILD
    split_d = "frozen split" if on_frozen else f"{C.manifest.get('version')} split"
    R.add("nRevSubstringPairs", I(len(rev)), f"test chain / shorter training representative pairs with exact containment ({split_d})", src_r, PENDING, "r5")  # fmt: skip
    R.add("nRevSubstringTestChains", I(len({b for b, _ in rev})), f"test chains that contain a shorter training representative verbatim ({split_d})", src_r, PENDING, "r5")  # fmt: skip
    # the frozen split's values, pinned so that the account of the reverse containment survives
    # the rebuild (as RB_PREFIX_PINNED for the forward containment)
    pinned_rev = {"nRevSubstringPairs": "16", "nRevSubstringTestChains": "12"}
    live_rev = {
        "nRevSubstringPairs": I(len(rev)),
        "nRevSubstringTestChains": I(len({b for b, _ in rev})),
    }
    for live, val in pinned_rev.items():
        if on_frozen:
            assert live_rev[live] == val, (live, live_rev[live], val)
        R.add(live.replace("nRevSubstring", "nRevSubstringPreFix"), val, f"{live} on the FROZEN 2026-10 split (before fix (a))", f"pinned in make_numbers.py (checked against \\{live} while {RB_FROZEN_BUILD} is the current build)", STABLE, "r5")  # fmt: skip
    ex = "51677_1_1_1"
    reps_ex = sorted({i for b, i in rev if b == ex}, key=lambda x: int(x.split("_")[0]))
    if on_frozen:
        assert len(reps_ex) == 3, reps_ex
        R.add("idRevContainTest", S(ex), "example test chain that contains shorter training representatives verbatim", src_r, PENDING, "r5")  # fmt: skip
        R.add("lenRevContainTest", I(len(d.sequence[ex])), "its length (aa)", src_r, PENDING, "r5")  # fmt: skip
        R.add("idRevContainReps", S(", ".join(reps_ex[:-1]) + " and " + reps_ex[-1]), "the training representatives it contains", src_r, PENDING, "r5")  # fmt: skip
    else:
        # the example is a property of the frozen split; a fixed build removes such
        # representatives, so its values are pinned
        src_p = f"pinned in make_numbers.py (FROZEN 2026-10 split; this build has {len(reps_ex)} such representatives for {ex})"
        R.add("idRevContainTest", S(ex), "example test chain that contained shorter training representatives verbatim (FROZEN 2026-10 split)", src_p, STABLE, "r5")  # fmt: skip
        R.add("lenRevContainTest", "371", "its length (aa)", src_p, STABLE, "r5")  # fmt: skip
        R.add("idRevContainReps", "51674\\_1\\_1\\_1, 51675\\_1\\_1\\_1 and 51676\\_1\\_1\\_1", "the training representatives it contained (FROZEN 2026-10 split)", src_p, STABLE, "r5")  # fmt: skip
    p_ = PATHS.work / "search_leak_common_unfiltered.json"
    if p_.exists():
        rows = json.loads(p_.read_text())
        bid = [
            r
            for r in rows
            if r[1] in C.reps["unfiltered"]
            and r[2] >= 0.3
            and r[3] >= 0.8
            and r[4] >= 0.8
        ]
        short = [r for r in bid if min(r[7], r[8]) / max(r[7], r[8]) < 0.8]
        src_b = "cached build-settings search (search_leak_common_unfiltered.json: test chains vs train_unfiltered_best.fasta) x sequence lengths"
        R.add("nLeakBidirCommonPairsUnfiltered", I(len(bid)), "test chain / unfiltered representative pairs at >= 30 % identity over >= 80 % of both sequences (build settings; frozen split)", src_b, PENDING, "r5")  # fmt: skip
        R.add("nLeakBidirCommonPairsShortUnfiltered", I(len(short)), "of them with a length ratio below 0.8, which mmseqs skips before alignment when -c 0.8 is set", src_b, PENDING, "r5")  # fmt: skip


def sec_supp_figs_revision(C):
    """Numbers of Supplementary Figs. S6 and S8, produced in the Supplementary.tex revision (31 Oct).

    Computed with the figures' own functions (scripts/figures/paper/supp_s6_rereferencing.py and
    supp_s8_replicate_gallery.py), so legend and artwork quote one analysis. S6d reads the cached
    recovery experiment (tmp/paper_figs/s6_recovery.json in the code checkout; rerun with the
    generator's --recompute, about 4 min)."""
    s6 = paper("supp_s6_rereferencing")
    s8 = paper("supp_s8_replicate_gallery")
    # S6c: LACS reimplementation vs the BMRB's LACS reports
    df = s6.sp.load_main(
        ["id", "entry_id", "st_id"] + [lacs_off_ppm_col(a) for a in ATOMS]
    )
    df["entry_id"] = df.entry_id.astype(str)
    LC, _, info = s6.lacs_comparison(df)
    L = LC.set_index("nucleus")
    src6c = "data/external/bmrb_lacs/bmr<ID>_LACS.str (negated, MATLAB sign) x Parquet lacs_off_<A>_ppm, entries with exactly one scored chain (supp_s6_rereferencing.lacs_comparison)"
    R.add("nLacsBmrbReports", I(info["n_reports"]), "LACS reports distributed by the BMRB (local copies parsed)", src6c, STABLE, "off")  # fmt: skip
    R.add("nLacsBmrbEntries", I(info["n_single_chain_entries_with_report"]), "entries with exactly one scored chain and a BMRB LACS report (compared in Supplementary Fig. S6c)", src6c, STABLE, "off")  # fmt: skip
    for a, nm in (("CA", "Ca"), ("C", "C"), ("HA", "Ha")):
        R.add(f"rLacsBmrb{nm}", F(L.loc[a, "pearson"], 3), f"Pearson r, TriZOD LACS {a} offset vs negated BMRB LACS report, single-chain entries", src6c, STABLE, "off")  # fmt: skip
        R.add(f"medAbsDiffLacsBmrb{nm}", F(L.loc[a, "median_abs_diff_ppm"], 2), f"median |TriZOD - BMRB| LACS {a} offset (ppm), single-chain entries", src6c, STABLE, "off")  # fmt: skip
    R.add("pctLacsBmrbWithinQuarterPpmMin", P(100 * L.share_within_0p25_ppm.min()), "lowest share, over CA/CB, C and HA, of compared entries whose TriZOD and BMRB LACS offsets differ by at most 0.25 ppm", src6c, STABLE, "off")  # fmt: skip
    # S6d: offset recovery on deposited shifts
    Rr, rinfo = s6.run_recovery(False)
    S = s6.recovery_summary(Rr)
    S = S[S.delta_ppm != 0.0]
    src6d = "tmp/paper_figs/s6_recovery.json (supp_s6_rereferencing.run_recovery: trizod.scoring.get_offset_corrected_shifts on deposited backbone shifts + delta, modes lacs / potenci-only / both)"
    R.add("seedOffsetRecovery", str(int(rinfo["seed"])), "random seed of the chain sample of the offset-recovery experiment (Supplementary Fig. S6d)", "supp_s6_rereferencing.REC_SEED", STABLE, "par")  # fmt: skip
    R.add("nOffsetRecoveryChains", I(rinfo["n_sampled"]), "chains sampled for the offset-recovery experiment", "supp_s6_rereferencing.REC_N_CHAINS", STABLE, "par")  # fmt: skip
    R.add("minOffsetRecoveryScored", I(rinfo["min_scored"]), "minimum scored residues of a chain eligible for the offset-recovery sample", "supp_s6_rereferencing.REC_MIN_SCORED", STABLE, "par")  # fmt: skip
    R.add("nOffsetRecoveryPool", I(rinfo["pool_size"]), "released chains with at least \\minOffsetRecoveryScored scored residues (the sampling pool)", "Parquet n_scored", STABLE, "off")  # fmt: skip
    R.add("offsetRecoveryMaxPpm", I(max(rinfo["deltas"])), "largest offset (ppm, either sign) added in the offset-recovery experiment", "supp_s6_rereferencing.REC_DELTAS", STABLE, "par")  # fmt: skip
    R.add("thrOffsetRecoveredPpm", F(0.1, 1), "an added offset counts as recovered when the chain's total offset changes by the added value within this tolerance (ppm)", "supp_s6_rereferencing.recovery_summary", STABLE, "par")  # fmt: skip

    def share(mode, nuc, fn):
        g = S[(S["mode"] == mode) & (S.nucleus == nuc)]
        return P(100 * fn(g.share_within_0p1_ppm))

    for nuc, nm in (("CA", "Ca"), ("C", "C"), ("N", "N")):
        R.add(f"pctOffsetRecoveredBoth{nm}Min", share("both", nuc, np.min), f"lowest share, over the added offsets, of sampled chains whose {nuc} offset is recovered in the released mode (both)", src6d, STABLE, "off")  # fmt: skip
        R.add(f"pctOffsetRecoveredLacs{nm}Min", share("lacs", nuc, np.min), "same, LACS alone (mode lacs)", src6d, STABLE, "off")  # fmt: skip
    pot = S[S["mode"] == "potenci-only"]
    R.add("pctOffsetRecoveredPotenciMax", P(100 * pot.share_within_0p1_ppm.max()), "highest share, over CA, C, N and the added offsets, of sampled chains whose offset the POTENCI/AIC stage alone (mode potenci-only) recovers", src6d, STABLE, "off")  # fmt: skip
    # S8d: seeded gallery of replicate groups
    g8 = s8.sp.load_main(["id", "entry_id", "sequence", "is_bound"])
    g8["entry_id"] = g8.entry_id.astype(str)
    _, n_elig = s8.gallery(g8)
    R.add("seedReplicateGallery", str(int(s8.GALLERY_SEED)), "random seed of the replicate-group gallery (Supplementary Fig. S8d)", "supp_s8_replicate_gallery.GALLERY_SEED", STABLE, "par")  # fmt: skip
    R.add("nReplicateGallery", I(s8.N_GALLERY), "replicate groups drawn for the gallery", "supp_s8_replicate_gallery.N_GALLERY", STABLE, "par")  # fmt: skip
    R.add("minReplicateGalleryEntries", I(s8.MIN_ENTRIES), "minimum entries contributing non-bound chains of one exact sequence for the gallery draw", "supp_s8_replicate_gallery.MIN_ENTRIES", STABLE, "par")  # fmt: skip
    R.add("nReplicateGalleryEligible", I(n_elig), "sequences whose non-bound chains come from at least \\minReplicateGalleryEntries entries (the gallery's sampling frame)", "Parquet sequence, entry_id, is_bound (supp_s8_replicate_gallery.gallery)", STABLE, "r2")  # fmt: skip
    R.add("caseClusterIdentity", P(100 * s8.MIN_FIDENT, 0), "minimum sequence identity of a case-study cluster member to the seed chain (mmseqs easy-search; sec_ubiquitin and Supplementary Fig. S8a-c)", "supp_s8_replicate_gallery.MIN_FIDENT = make_numbers sec_ubiquitin H.fid >= 0.8", STABLE, "par")  # fmt: skip
    R.add("caseClusterCoverage", P(100 * s8.MIN_COV, 0), "minimum query and target coverage of a case-study cluster member", "supp_s8_replicate_gallery.MIN_COV = make_numbers sec_ubiquitin qcov, tcov >= 0.9", STABLE, "par")  # fmt: skip


PATHS: Paths

if __name__ == "__main__":
    main()
