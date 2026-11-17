"""Release identity of the TriZOD dataset: version, licences, test-set name,
the G-score cuts and the provenance blocks every release file repeats.

One module, so the Parquet metadata (``scripts/build_parquet_dataset.py``), the
UdonPred view (:mod:`trizod.dataset.export_jsonl`), the deploy FASTA summary
(:mod:`trizod.dataset.deploy_fasta`) and the assembled deposit
(:mod:`trizod.dataset.finalize`) state the same values (release audit
2026-11-17, checks D14, F32, G36 and G38). ``tests/test_release_docs.py``
checks that ``CITATION.cff``, ``.zenodo.json``, ``pyproject.toml`` and the
documentation templates agree with it.

The cuts are metadata, never label columns: the continuous G-score
(``gscores``) is the label, and a recalibration must not change the data.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

#: Dataset release version (semantic versioning of the Zenodo deposit).
DATASET_VERSION = "0.4.0"
#: The previous Zenodo version, the CHANGELOG's baseline.
PREVIOUS_VERSION = "0.3.0"
PREVIOUS_VERSION_DATE = "2026-07-15"
#: Concept DOI of the Zenodo record (all versions).
CONCEPT_DOI = "10.5281/zenodo.21309963"
REPOSITORY = "https://github.com/tsenoner/trizod"

#: Name of the final TriZOD test set. From this release on it may only shrink:
#: a chain that no longer qualifies is retired (``split == "test_retired"``)
#: and stays a leakage target.
TEST_SET_NAME = "TriZOD-test v1.0"

#: Licences: the data (every file of the deposit) and the code that builds it.
DATA_LICENSE = "CC-BY-4.0"
DATA_LICENSE_NAME = "CC BY 4.0"
DATA_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
CODE_LICENSE = "MIT"
#: BMRB asks users to acknowledge it; no BMRB licence is claimed or implied.
BMRB_ACKNOWLEDGEMENT = (
    "Derived from entries of the Biological Magnetic Resonance Data Bank "
    "(BMRB, https://bmrb.io); please acknowledge the BMRB when you use these "
    "data. No BMRB licence is claimed."
)
LICENSE_STATEMENT = (
    f"Data: {DATA_LICENSE_NAME} ({DATA_LICENSE_URL}). Code: {CODE_LICENSE} "
    f"({REPOSITORY}). {BMRB_ACKNOWLEDGEMENT}"
)

#: Where the column dictionary is.
COLUMN_DICTIONARY = (
    "columns.md in the release folder; generated from COLUMNS in "
    "scripts/build_parquet_dataset.py into docs/dataset/columns.md of the code "
    "repository"
)

# --------------------------------------------------------------------------- #
# G-score cuts
# --------------------------------------------------------------------------- #

#: The three cuts on the continuous G-score, loosest first.
ORDERED_CUT = 0.15
DISORDER_CUT = 0.35
CONFIDENT_DISORDER_CUT = 0.70
CUTS = (ORDERED_CUT, DISORDER_CUT, CONFIDENT_DISORDER_CUT)
#: The convention every cut follows.
CUT_RULE = (
    f"disordered iff G >= t; ordered iff G < {ORDERED_CUT:.2f}. Binary call: "
    f"disordered iff G >= {DISORDER_CUT:.2f}. Confidently disordered iff "
    f"G >= {CONFIDENT_DISORDER_CUT:.2f}"
)
#: The four zones the three cuts make, as (name, lower bound, upper bound):
#: lower bound inclusive, upper bound exclusive.
ZONES = (
    ("ordered", 0.0, ORDERED_CUT),
    ("indeterminate", ORDERED_CUT, DISORDER_CUT),
    ("disordered", DISORDER_CUT, CONFIDENT_DISORDER_CUT),
    ("confidently_disordered", CONFIDENT_DISORDER_CUT, None),
)
#: Posterior levels each cut was calibrated to (the smallest G at which the
#: posterior of disorder reaches the level at the calibration prevalence).
CUT_LEVELS = {ORDERED_CUT: 0.10, DISORDER_CUT: 0.50, CONFIDENT_DISORDER_CUT: 0.90}

#: The k >= 10 reliability note that goes with the cuts.
K_MIN_RELIABLE = 10
K_NOTE = (
    f"k (per residue) counts the backbone shifts behind a residue's score. "
    f"k >= {K_MIN_RELIABLE} is a conventional reliability cut, not a filter: "
    "replicate agreement rises gradually with k, and the binary call of a "
    f"residue with k < {K_MIN_RELIABLE} flips between replicate depositions "
    "more often. The release keeps every scored residue and ships k so the "
    "trainer can weight or filter"
)
LABEL_NOTE = (
    "The continuous G-score (gscores) is the label; the cuts are metadata and "
    "never appear as label columns, so a recalibration changes no data"
)


def zone_of(g: float) -> str:
    """The zone of a finite G (:data:`ZONES`)."""
    for name, lo, hi in ZONES:
        if g >= lo and (hi is None or g < hi):
            return name
    raise ValueError(f"G = {g} below 0")


def cut_block() -> dict:
    """The cut block written into the Parquet metadata and every summary."""
    return {
        "values": list(CUTS),
        "ordered": f"G < {ORDERED_CUT:.2f}",
        "disordered": f"G >= {DISORDER_CUT:.2f}",
        "confidently_disordered": f"G >= {CONFIDENT_DISORDER_CUT:.2f}",
        "rule": CUT_RULE,
        "posterior_levels": {f"{c:.2f}": level for c, level in CUT_LEVELS.items()},
        "zones": [{"zone": name, "from": lo, "below": hi} for name, lo, hi in ZONES],
        "label": LABEL_NOTE,
        "k_note": K_NOTE,
    }


def cut_label(c: float) -> str:
    return f"{c:.2f}"


# --------------------------------------------------------------------------- #
# test set
# --------------------------------------------------------------------------- #


def testset_label(n_test: int, n_retired: int) -> str:
    """``TriZOD-test v1.0 (n = 405; 47 retired chains)``."""
    return f"{TEST_SET_NAME} (n = {n_test}; {n_retired} retired chains)"


def testset_block(splits, retired_reasons) -> dict:
    """The test-set identity from the Parquet's ``split`` and
    ``retired_reason`` columns (equal-length sequences)."""
    splits = list(splits)
    reasons: dict[str, int] = {}
    for s, r in zip(splits, retired_reasons):
        if s == "test_retired":
            key = r or "unknown"
            reasons[key] = reasons.get(key, 0) + 1
    n_test = splits.count("test_trizod")
    n_retired = splits.count("test_retired")
    return {
        "name": TEST_SET_NAME,
        "label": testset_label(n_test, n_retired),
        "split": "test_trizod",
        "n_chains": n_test,
        "retired_split": "test_retired",
        "n_retired": n_retired,
        "retired_by_reason": dict(sorted(reasons.items())),
        "chezod117_split": "test_chezod117",
        "n_chezod117": splits.count("test_chezod117"),
        "policy": (
            "the test set may only shrink: a chain that no longer qualifies is "
            "retired (split test_retired, reason in retired_reason), is not "
            "tested on, and stays a leakage target"
        ),
    }


# --------------------------------------------------------------------------- #
# provenance: code commit, BMRB snapshot, calibration
# --------------------------------------------------------------------------- #


def sha256_file(path) -> str:
    """SHA-256 of a file (never cached: a file rewritten in place must re-hash)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(repo, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def code_commit(repo=None, path: str | None = None) -> dict | None:
    """``{"commit": <full sha>, "dirty": bool}`` of the checkout ``repo``
    (default: this code repository); with ``path``, the last commit touching
    that path and whether it has uncommitted changes. None outside git."""
    if repo is None:
        repo = Path(__file__).resolve().parents[2]
    try:
        if path is None:
            commit = _git(repo, "rev-parse", "HEAD")
            dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
        else:
            commit = _git(repo, "log", "-1", "--format=%H", "--", path)
            dirty = _git(repo, "status", "--porcelain", "--", path)
    except (OSError, subprocess.CalledProcessError):
        return None
    if not commit:
        return None
    return {"commit": commit, "dirty": bool(dirty)}


SNAPSHOT_MANIFEST = "BMRB_SNAPSHOT.json"
#: Keys of ``BMRB_SNAPSHOT.json`` copied into the snapshot block.
_SNAPSHOT_KEYS = (
    "retrieved",
    "retrieved_end",
    "n_entries",
    "entry_list_sha256",
    "source",
    "retries",
)


def snapshot_block(bmrb_dir) -> dict | None:
    """The BMRB snapshot the release was built from: its directory name and
    the identity its ``BMRB_SNAPSHOT.json`` records (retrieval window,
    n_entries, entry_list_sha256). None when there is no such file."""
    bmrb_dir = Path(bmrb_dir)
    meta_path = bmrb_dir / SNAPSHOT_MANIFEST
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None
    block = {"name": bmrb_dir.resolve().name}
    block.update({k: meta[k] for k in _SNAPSHOT_KEYS if k in meta})
    return block


FRAMEWORK_PATH = "scripts/validation/threshold_calibration/framework"
RESULTS_JSON = "results.json"
#: The framework's per-bin likelihood-ratio curve (0.01 G bins), beside
#: results.json; its primary-estimator column is carried into the metadata so
#: the prevalence of any split can be estimated (``class_balance.em_prevalence``).
CURVES_CSV = "curves_bins.csv"
LR_BIN_WIDTH = 0.01
#: Calibration-set names of the framework's ``transport`` block that are
#: release splits (plus ``train_<tier>`` if a framework run provides them).
TRANSPORT_SPLITS = ("train", "test_trizod", "test_chezod117")


def _ci(triple) -> dict:
    """``[point, lo, hi]`` -> ``{"value", "ci95"}``."""
    return {"value": triple[0], "ci95": [triple[1], triple[2]]}


def read_calibration(path, framework_commit: dict | None = None) -> dict:
    """Summary of a threshold-framework run for the release metadata.

    ``path`` is the framework's output directory (or its ``results.json``).
    Records the calibration prevalence pi, the consensus likelihood ratio and
    posterior at each cut with their 95 % intervals, the estimated cuts, the
    prevalence of every release split the framework transported to, the
    replicate agreement below and above k = 10, the results.json SHA-256 and
    the framework commit (``framework_commit``, default: the last commit of
    :data:`FRAMEWORK_PATH` in this repository).
    """
    path = Path(path)
    results = path / RESULTS_JSON if path.is_dir() else path
    R = json.loads(results.read_text())
    settings = R["settings"]
    est = settings.get("primary_estimator", "interp")
    cons = R["calibration"]["estimators"][est]["consensus"]
    paper = [float(c) for c in settings["paper_values"]]
    if tuple(round(c, 4) for c in paper) != CUTS:
        raise ValueError(
            f"{results}: the framework calibrated cuts {paper}, the release "
            f"states {list(CUTS)}"
        )
    at_cuts = cons["at_paper_values"]
    lr = {cut_label(c): _ci(at_cuts[cut_label(c)]["LR"]) for c in CUTS}
    post = {cut_label(c): _ci(at_cuts[cut_label(c)]["posterior"]) for c in CUTS}
    check = R["calibration"].get("paper_check", {}).get(est, {})
    estimated = {
        cut_label(v["paper_value"]): {
            "posterior_level": v["level"],
            "estimate": v["estimate"],
            "ci95": v["CI"],
            "inside_ci": v["inside_CI"],
        }
        for v in check.values()
    }
    by_split = {}
    for row in R.get("transport", []):
        name = row.get("set")
        if name in TRANSPORT_SPLITS or str(name).startswith("train_"):
            by_split[name] = {
                "chains": row.get("chains"),
                "residues": row.get("residues"),
                "pi": {
                    "value": row["pi_em"],
                    "ci95": [row["pi_em_lo"], row["pi_em_hi"]],
                },
                "posterior_at_cuts": {
                    cut_label(c): {
                        "value": row[f"posterior_at_{c:.2f}"],
                        "ci95": [
                            row[f"posterior_at_{c:.2f}_lo"],
                            row[f"posterior_at_{c:.2f}_hi"],
                        ],
                    }
                    for c in CUTS
                    if f"posterior_at_{c:.2f}" in row
                },
            }
    replicates = {}
    for row in R.get("k_check", {}).get("replicates", []):
        if row.get("k") in ("k1-9", "k10-21"):
            key = f"k < {K_MIN_RELIABLE}" if row["k"] == "k1-9" else "k >= 10"
            replicates[key] = {
                "kappa_binary": _ci(row["kappa_binary"]),
                "disordered_flip": _ci(row["disordered_flip"]),
                "residue_pairs": row.get("n_residue_pairs"),
            }
    if framework_commit is None:
        framework_commit = code_commit(path=FRAMEWORK_PATH)
    analysis = R.get("analysis_set", {})
    curve = read_lr_curve(results.parent / CURVES_CSV, est)
    return {
        "framework": FRAMEWORK_PATH,
        "framework_commit": framework_commit,
        "results_json": results.parent.name + "/" + results.name,
        "results_json_sha256": sha256_file(results),
        "input_parquet": R.get("inputs", {}).get("release"),
        "input_parquet_sha256": R.get("input_sha256", {}).get("release"),
        "rule": settings.get("rule"),
        "estimator": est,
        "reference_pool": "consensus",
        "reference_weights": settings.get("pools", {}).get("consensus"),
        "calibration_set": {
            "tier": settings.get("tier"),
            "chains": analysis.get("chains"),
            "fit_chains": analysis.get("fit_chains"),
            "held_out_chains": analysis.get("test_chains"),
        },
        "nboot": settings.get("nboot"),
        "seed": settings.get("seed"),
        "prevalence": _ci(cons["pi"]),
        "likelihood_ratio_at_cuts": lr,
        "posterior_at_cuts": post,
        "estimated_cuts": estimated,
        "prevalence_by_split": by_split,
        "replicate_agreement_by_k": replicates,
        "lr_curve": curve,
    }


def read_lr_curve(path, estimator: str = "interp") -> dict | None:
    """The consensus log-likelihood-ratio per 0.01 G bin (``logLR_<estimator>``
    of the framework's ``curves_bins.csv``), or None when the file is absent."""
    import csv

    path = Path(path)
    if not path.is_file():
        return None
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    column = f"logLR_{estimator}"
    if not rows or column not in rows[0]:
        return None
    rows.sort(key=lambda r: float(r["G_lo"]))
    return {
        "source": f"{CURVES_CSV}:{column}",
        "sha256": sha256_file(path),
        "bin_width": LR_BIN_WIDTH,
        "bin": "floor(G / bin_width), clipped to the last bin",
        "log_lr": [round(float(r[column]), 6) for r in rows],
    }


def posterior(pi: float, lr: float) -> float:
    """Posterior probability of disorder at prevalence ``pi`` and likelihood
    ratio ``lr`` (posterior odds = prior odds x LR)."""
    odds = pi / (1.0 - pi) * lr
    return odds / (1.0 + odds)
