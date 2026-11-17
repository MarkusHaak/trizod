#!/usr/bin/env python3
"""Assemble the TriZOD deposit: every hand-over file in one folder, one MANIFEST.

``trizod dataset finalize`` is the last step of a release (release audit
2026-11-17, checks B4, D14, E21, F32, F33, G35, G36, G38 and G39). It takes the
release Parquet built by ``scripts/build_parquet_dataset.py`` (with
``--calibration``), the package bundle it was built from (``trizod dataset
package``), the shift table and the BMRB snapshot, and writes into ``--out``::

    trizod_dataset.parquet          the canonical table (copied)
    trizod_shifts.parquet           the chemical-shift companion (copied)
    disorder_trizod.fasta           deploy FASTA, regenerated from the bundle
    disorder_trizod.summary.json
    udonpred/                       UdonPred v2 JSONL view (export-jsonl)
    crosswalk/                      test-set crosswalk (test-crosswalk):
                                    test_set_crosswalk.{tsv,json},
                                    test_set_overlap.tsv
    test/                           the bundle's test FASTAs (TriZOD-test,
                                    retired chains, CheZOD117 and its original
                                    sequences), the test-set labels table and
                                    the pin provenance
    class_balance.tsv               per-split label distribution (F33)
    known_issues.tsv                per-chain known issues and changes
                                    (chain_id, issue, evidence; G39)
    bmrb_files.sha256               per-file SHA-256 of the BMRB snapshot (B4)
    VALIDATION.json                 the final audit's results (--validation)
    README.md, datasheet.md,        rendered from docs/dataset/ templates with
    CHANGELOG.md                    the counts of this build (G36, G39)
    columns.md                      the column dictionary (D14)
    LICENSE.md, CITATION.cff, zenodo.json
    MANIFEST.json                   every file: SHA-256, bytes, row/chain
                                    counts; version, licence, commits, snapshot
                                    and xref-source hashes, cuts, calibration,
                                    leakage rule (G35)
    SHA256SUMS                      ``sha256sum -c`` over every other file

Two commits are recorded apart (G35, A2): the **scoring commit** scored the
entries and ran the dataset chain into the package bundle (its
``pipeline_version``, or ``--scoring-commit``), the **build/finalize commit**
built the Parquet and assembled the deposit. No shipped file names a path of
the machine it was built on: paths are deposit-, bundle- or repo-relative, or
a file name, and finalize refuses to finish otherwise.

The CHANGELOG's changes and known issues, and ``known_issues.tsv``, are computed
(:mod:`trizod.dataset.known_issues`) from the release, the package bundle and,
when given, the previous build's bundle (``--previous-build``: tier moves,
re-score statistics, sequence changes), the build's BMRB pickle cache
(``--pkl-dir``: pH* / pD, temperature and pressure units), an audit's
snapshot-drift and construct-caveat tables (``--snapshot-drift``,
``--construct-caveats``) and the reviewed table
``docs/dataset/known_issues_reviewed.tsv``. A missing input is stated as not
assessed, never silently left out.

The step is deterministic: the same inputs and output path give byte-identical
files. Nothing is uploaded.

Usage
-----
    uv run --with pyarrow trizod dataset finalize \\
        --parquet <trizod_dataset.parquet> --shifts <trizod_shifts.parquet> \\
        --bundle <work-dir>/release_bundle/trizod-dataset-<version> \\
        --bmrb-dir <data/raw/bmrb_entries_YYYY-MM-DD> --out <release dir> \\
        [--previous-build <previous build's package bundle>] \\
        [--pkl-dir <the build's tmp/bmrb_entries>] \\
        [--snapshot-drift <TSV>] [--construct-caveats <TSV>] \\
        [--scoring-commit <sha>] [--calibration-input <calibrated Parquet>] \\
        [--crosswalk-root <checkout holding data/>] [--set NAME=SPEC ...] \\
        [--validation <VALIDATION.json>] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trizod import paths as repo_paths
from trizod.dataset import class_balance as cb
from trizod.dataset import deploy_fasta, export_jsonl, testset_crosswalk
from trizod.dataset import known_issues as ki
from trizod.dataset import release_info as ri
from trizod.dataset.mmseqs import describe_rule
from trizod.dataset.testset import RETIRED_FASTA
from trizod.dataset.tiers import TIER_ORDER
from trizod.io.fasta import count_fasta, read_fasta

#: Docs templates (``docs/dataset/``) -> deposit file.
TEMPLATES = {
    "bundle-README.md": "README.md",
    "datasheet.md": "datasheet.md",
    "CHANGELOG.md": "CHANGELOG.md",
}
COLUMNS_DOC = "columns.md"
#: The v0.3.0 layout is the first 42 columns (offset names as renamed in 2026-08).
N_COLUMNS_V030 = 42
#: v0.3.0 (Zenodo, 2026-07-15) counts, for the CHANGELOG: scored records and
#: training representatives per tier, test chains. From its Parquet.
V030 = {
    "rows": 16851,
    "columns": 42,
    "tier_records": {
        "unfiltered": 16851,
        "tolerant": 15440,
        "moderate": 10941,
        "strict": 3271,
    },
    "train_reps": {
        "unfiltered": 5900,
        "tolerant": 5660,
        "moderate": 4491,
        "strict": 1381,
    },
    "test_trizod": 365,
    "test_chezod117": 115,
}
#: Release Parquet columns finalize reads.
_READ = (
    "id",
    "entry_id",
    "st_id",
    "entity_assem_id",
    "entity_id",
    "sequence",
    "split",
    "train_tier",
    "length",
    "gscores",
    "mask",
    "k",
    "membrane_mimetic_sample",
    "temperature_placeholder",
    "high_pressure",
    "ionic_strength",
    "retired_reason",
    "chezod1325_overlap",
    "chezod1174_overlap",
    "label_tier",
    "label_tier_source",
    "ph",
    "cosolvent_evidence",
    "is_bound",
    "physical_state",
)
STRICT_IONIC_CAP_M = ki.STRICT_IONIC_CAP_M
#: What ``total_bytes`` of MANIFEST.json covers.
TOTAL_BYTES_COVERS = (
    "the files listed under files, i.e. every file of the deposit except "
    "MANIFEST.json and SHA256SUMS, which describe the others; n_files counts "
    "all of them"
)
_PLACEHOLDER = re.compile(r"\{\{\s*([a-z0-9_]+)\s*\}\}")
#: The maintainer note at the top of a template, dropped when rendering.
_TEMPLATE_NOTE = re.compile(r"<!-- Template:.*?-->\n*", re.S)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _fmt(n) -> str:
    return f"{n:,}"


def _ci(block: dict | None, digits: int = 3) -> str:
    if not block:
        return "not recorded"
    lo, hi = block["ci95"]
    return f"{block['value']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _lr(block: dict) -> str:
    v, (lo, hi) = block["value"], block["ci95"]

    def f(x):
        return f"{x:.2f}" if x < 10 else f"{x:.1f}"

    return f"{f(v)} [{f(lo)}, {f(hi)}]"


def render_template(text: str, values: dict[str, str], name: str = "template") -> str:
    """Fill every ``{{ name }}`` of ``text`` from ``values``; an unknown name is
    an error, so a template can never ship half-rendered."""
    text = _TEMPLATE_NOTE.sub("", text)
    unknown = sorted({m.group(1) for m in _PLACEHOLDER.finditer(text)} - set(values))
    if unknown:
        raise ValueError(f"{name}: no value for {unknown}")
    out = _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), text)
    if "{{" in out:
        raise ValueError(f"{name}: unrendered '{{{{' left after rendering")
    return out


# --------------------------------------------------------------------------- #
# the Parquet
# --------------------------------------------------------------------------- #


def read_parquet(path: Path) -> tuple[list[dict], dict[str, str], list[str]]:
    """Rows (``_READ`` columns), key-value metadata and column names."""
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    rows, meta = export_jsonl.read_release(
        path, columns=tuple(c for c in _READ if c in names)
    )
    return rows, meta, names


def _meta_json(meta: dict[str, str], key: str) -> dict | None:
    try:
        value = json.loads(meta.get(key, ""))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def summarise_rows(rows: list[dict], names: list[str]) -> dict:
    """Counts the docs and the MANIFEST state, all from the Parquet."""
    from collections import Counter

    splits = Counter(r["split"] for r in rows)
    reps = {}
    for i, tier in enumerate(TIER_ORDER):
        reps[tier] = sum(
            1
            for r in rows
            if r["split"] == "train"
            and r.get("train_tier") in TIER_ORDER
            and TIER_ORDER.index(r["train_tier"]) >= i
        )
    test = [r for r in rows if r["split"] == "test_trizod"]

    def flagged(col):
        return sum(1 for r in rows if r.get(col) is True)

    scored = sum(sum(1 for m in r["mask"] if m) for r in rows)
    k_low = sum(
        sum(1 for m, k in zip(r["mask"], r["k"]) if m and k < ri.K_MIN_RELIABLE)
        for r in rows
    )
    return {
        "rows": len(rows),
        "columns": len(names),
        "split_counts": dict(sorted(splits.items())),
        "train_reps": reps,
        "rule_flags": {
            "high_pressure": flagged("high_pressure"),
            "membrane_mimetic_sample": flagged("membrane_mimetic_sample"),
            "ionic_strength_above_strict_cap": sum(
                1
                for r in rows
                if r.get("ionic_strength") is not None
                and r["ionic_strength"] > STRICT_IONIC_CAP_M
            ),
            "temperature_placeholder": flagged("temperature_placeholder"),
        },
        "retired_by_reason": dict(
            sorted(
                Counter(
                    r.get("retired_reason") or "unknown"
                    for r in rows
                    if r["split"] == "test_retired"
                ).items()
            )
        ),
        "test_clean_vs": {
            name: sum(1 for r in test if r.get(f"{name}_overlap") is False)
            for name in ("chezod1325", "chezod1174")
        },
        "scored_residues": scored,
        "share_k_lt_10": k_low / scored if scored else None,
        "new_columns": names[N_COLUMNS_V030:],
    }


# --------------------------------------------------------------------------- #
# the BMRB snapshot (B4)
# --------------------------------------------------------------------------- #

_ENTRY_DIR = re.compile(r"bmr(\d+)")


def snapshot_entry_files(bmrb_dir: Path) -> list[tuple[int, Path]]:
    """``(entry ID, path)`` of every ``bmr<ID>/bmr<ID>_3.str`` of a snapshot,
    sorted by numeric entry ID."""
    files = []
    for d in Path(bmrb_dir).iterdir():
        m = _ENTRY_DIR.fullmatch(d.name)
        if m and d.is_dir():
            f = d / f"{d.name}_3.str"
            if f.is_file():
                files.append((int(m.group(1)), f))
    return sorted(files)


def snapshot_files_table(files, out: Path) -> dict:
    """Write ``sha256sum -c``-format hashes of the snapshot's entry files
    (:func:`snapshot_entry_files`; paths relative to the snapshot root) and
    return their count and the table's own SHA-256."""

    def digest(item):
        return ri.sha256_file(item[1])

    with ThreadPoolExecutor(8) as ex:
        hashes = list(ex.map(digest, files))
    with out.open("w") as fh:
        for (eid, _), sha in zip(files, hashes):
            fh.write(f"{sha}  bmr{eid}/bmr{eid}_3.str\n")
    return {"n_files": len(files), "files_table_sha256": ri.sha256_file(out)}


# --------------------------------------------------------------------------- #
# MANIFEST (G35)
# --------------------------------------------------------------------------- #


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def file_entry(path: Path) -> dict:
    """SHA-256, size and the row / record / sequence count of one file."""
    entry = {"bytes": path.stat().st_size, "sha256": ri.sha256_file(path)}
    suffix = path.suffix
    if suffix == ".parquet":
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        entry["n_rows"] = pf.metadata.num_rows
        if "id" in pf.schema_arrow.names:
            ids = pq.read_table(path, columns=["id"]).column("id")
            entry["n_chains"] = pc.count_distinct(ids).as_py()
    elif suffix == ".fasta":
        entry["n_sequences"] = count_fasta(path)
    elif suffix == ".jsonl":
        entry["n_records"] = _count_lines(path)
    elif suffix == ".tsv":
        entry["n_rows"] = max(_count_lines(path) - 1, 0)
    elif suffix == ".sha256":
        entry["n_files"] = _count_lines(path)
    return entry


def write_checksums(out: Path, skip=("SHA256SUMS",)) -> Path:
    """``SHA256SUMS`` (``sha256sum -c`` format) over every file under ``out``."""
    lines = []
    for p in sorted(out.rglob("*")):
        rel = p.relative_to(out).as_posix()
        if p.is_file() and rel not in skip:
            lines.append(f"{ri.sha256_file(p)}  {rel}")
    path = out / "SHA256SUMS"
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #
# no machine paths in the deposit (G35)
# --------------------------------------------------------------------------- #

#: A JSON string value that is an absolute path, optionally behind a set-spec
#: scheme (``fasta:``, ``parquet:``): the whole value, never a path inside prose.
_ABS_VALUE = re.compile(r"^(?P<scheme>[a-z]+:)?(?P<path>/(?!/)[^\s]+)$")
#: The start of a stray absolute path of a build machine (not inside a
#: relative path or a URL: no path character before it).
_MACHINE_PATH = re.compile(
    r"(?<![\w.~/-])/(?:Users|home|private/var|private/tmp|var/folders|tmp)/"
)


def _anchor_prefixes(anchors) -> list[tuple[str, str]]:
    """``(absolute directory + "/", label)`` of every anchor, deepest first."""
    out = []
    for a in anchors:
        d, label = a if isinstance(a, tuple) else (a, "")
        if d is not None:
            out.append((str(Path(d).resolve()).rstrip("/") + "/", label))
    return sorted(out, key=lambda x: len(x[0]), reverse=True)


def _portable(value: str, anchors) -> str:
    """``value`` with every absolute path relative to the deepest anchor that
    holds it (a whole-value path outside every anchor becomes its file name)."""
    prefixes = _anchor_prefixes(anchors)
    m = _ABS_VALUE.match(value)
    if m:
        deepest = [(Path(prefix), label) for prefix, label in prefixes]
        return (m.group("scheme") or "") + ri.portable_path(m.group("path"), deepest)
    # a path inside prose or an embedded JSON string: relative to its anchor
    for prefix, label in prefixes:
        value = value.replace(prefix, label)
    return value


def _scrub(obj, anchors):
    if isinstance(obj, dict):
        return {k: _scrub(v, anchors) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v, anchors) for v in obj]
    if isinstance(obj, str):
        return _portable(obj, anchors)
    return obj


def scrub_json_paths(path: Path, anchors) -> bool:
    """Rewrite every string value of a JSON file that is an absolute path as a
    path relative to the first of ``anchors`` holding it (the deposit, the
    package bundle, the data root, the checkout), else its file name. Returns
    whether the file changed. Used on files other modules write (the deploy
    summary, the crosswalk, the UdonPred export summary)."""
    text = path.read_text()
    data = json.loads(text)
    clean = _scrub(data, anchors)
    if clean == data:
        return False
    path.write_text(json.dumps(clean, indent=2) + "\n")
    return True


def machine_paths(out: Path, anchors) -> list[str]:
    """``file: path`` of every absolute path of this machine left in a text
    file of ``out`` or the key-value metadata of a Parquet there."""
    needles = sorted(
        {str(Path(a).resolve()) for a in anchors if a is not None} | {str(Path.home())},
        key=len,
        reverse=True,
    )
    hits = []
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(out).as_posix()
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq

            meta = pq.read_schema(p).metadata or {}
            text = "\n".join(v.decode(errors="replace") for v in meta.values())
        elif p.suffix in (".json", ".md", ".tsv", ".cff", ".txt", "") or p.name in (
            "SHA256SUMS",
        ):
            text = p.read_text(errors="replace")
        else:
            continue
        hit = next((n for n in needles if n and n in text), None)
        if hit is None:
            m = _MACHINE_PATH.search(text)
            hit = m.group(0) if m else None
        if hit is not None:
            hits.append(f"{rel}: {hit}")
    return hits


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #


def _tier_table(bundle_manifest: dict, reps: dict) -> str:
    tiers = bundle_manifest.get("tiers", {})
    lines = [
        "| Tier | Scored records | Training representatives |",
        "|---|--:|--:|",
    ]
    for t in TIER_ORDER:
        rec = tiers.get(t, {}).get("scored_records")
        lines.append(
            f"| {t} | {_fmt(rec) if rec is not None else '–'} | {_fmt(reps[t])} |"
        )
    return "\n".join(lines)


def _split_table(split_counts: dict) -> str:
    order = (
        "train",
        "redundant",
        "test_trizod",
        "test_retired",
        "test_chezod117",
        "excluded",
    )
    lines = ["| Split | Chains |", "|---|--:|"]
    lines += [f"| `{s}` | {_fmt(split_counts.get(s, 0))} |" for s in order]
    return "\n".join(lines)


def _changelog_tier_table(bundle_manifest: dict, reps: dict) -> str:
    tiers = bundle_manifest.get("tiers", {})
    lines = [
        f"| Tier | Scored records v{ri.PREVIOUS_VERSION} | Scored records "
        f"v{ri.DATASET_VERSION} | Training representatives v{ri.PREVIOUS_VERSION} "
        f"| Training representatives v{ri.DATASET_VERSION} |",
        "|---|--:|--:|--:|--:|",
    ]
    for t in TIER_ORDER:
        now = tiers.get(t, {}).get("scored_records")
        lines.append(
            f"| {t} | {_fmt(V030['tier_records'][t])} "
            f"| {_fmt(now) if now is not None else '–'} "
            f"| {_fmt(V030['train_reps'][t])} | {_fmt(reps[t])} |"
        )
    return "\n".join(lines)


def _rule_table(flags: dict) -> str:
    rows = [
        (
            "High pressure: `pressure_bar` ≥ 400 bar (`high_pressure`)",
            "tolerant, moderate, strict",
            flags["high_pressure"],
        ),
        (
            "Membrane-mimetic sample, class rule (`membrane_mimetic_sample`)",
            "moderate, strict",
            flags["membrane_mimetic_sample"],
        ),
        (
            f"Ionic strength above {STRICT_IONIC_CAP_M} M (`ionic_strength`)",
            "strict",
            flags["ionic_strength_above_strict_cap"],
        ),
        (
            "Deposited 273 K read as not deposited (`temperature_placeholder`); "
            "re-scored at the 298 K default",
            "strict",
            flags["temperature_placeholder"],
        ),
    ]
    lines = ["| Rule (flag column) | Excluded from | Release chains flagged |"]
    lines.append("|---|---|--:|")
    lines += [f"| {a} | {b} | {_fmt(n)} |" for a, b, n in rows]
    return "\n".join(lines)


def _crosswalk_table(payload: dict) -> str:
    lines = [
        "| Set | Sequences | Still test | Training, tolerant or stricter "
        "| Training, unfiltered only | Redundant | Retired | Excluded or absent |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for rec in payload["overlap"]:
        if rec["set"] == testset_crosswalk.FINAL_SET:
            continue
        lines.append(
            f"| `{rec['set']}` | {_fmt(rec['n_sequences'])} | {_fmt(rec['test'])} "
            f"| {_fmt(rec['in_tolerant_training'])} "
            f"| {_fmt(rec['train_unfiltered'])} | {_fmt(rec['redundant'])} "
            f"| {_fmt(rec['retired'])} "
            f"| {_fmt(rec['excluded'] + rec['absent'])} |"
        )
    return "\n".join(lines)


def _calibration_values(calibration: dict | None) -> dict[str, str]:
    c0, c1, c2 = (ri.cut_label(c) for c in ri.CUTS)
    if not calibration:
        missing = "not recorded (the Parquet was built without --calibration)"
        return {
            "calibration_pi": missing,
            "calibration_set": missing,
            "calibration_lr": missing,
            "calibration_table": missing,
            "framework_commit": "not recorded",
            "results_sha256": "not recorded",
            "k_kappa": "not recorded",
            "k_flip": "not recorded",
        }
    lr = calibration["likelihood_ratio_at_cuts"]
    post = calibration["posterior_at_cuts"]
    est = calibration.get("estimated_cuts", {})
    cs = calibration.get("calibration_set", {})
    lines = [
        "| Cut | Zone it opens | Posterior level | Estimated cut [95 % CI] "
        "| Likelihood ratio [95 % CI] | Posterior at the calibration π [95 % CI] |",
        "|---|---|--:|---|---|---|",
    ]
    names = {
        c0: f"ordered below (G < {c0})",
        c1: f"disordered, binary call (G ≥ {c1})",
        c2: f"confidently disordered (G ≥ {c2})",
    }
    for c in (c0, c1, c2):
        e = est.get(c)
        e_txt = (
            f"{e['estimate']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]"
            if e
            else "–"
        )
        level = ri.CUT_LEVELS[float(c)]
        lines.append(
            f"| {c} | {names[c]} | {level:.2f} | {e_txt} | {_lr(lr[c])} "
            f"| {_ci(post[c])} |"
        )
    fw = calibration.get("framework_commit") or {}
    reps = calibration.get("replicate_agreement_by_k", {})
    hi, lo = reps.get("k >= 10"), reps.get(f"k < {ri.K_MIN_RELIABLE}")
    return {
        "calibration_pi": _ci(calibration["prevalence"]),
        "calibration_set": (
            f"the unbound, solution-state {cs.get('tier', '?')}-tier chains "
            f"without denaturant or cosolvent ({_fmt(cs['chains'])} chains)"
            if cs.get("chains")
            else "not recorded"
        ),
        "calibration_lr": (
            f"{_lr(lr[c0])} at {c0}, {_lr(lr[c1])} at {c1} and {_lr(lr[c2])} at {c2}"
        ),
        "calibration_table": "\n".join(lines),
        "framework_commit": fw.get("commit", "not recorded")
        + (" (uncommitted changes)" if fw.get("dirty") else ""),
        "results_sha256": calibration.get("results_json_sha256", "not recorded"),
        "k_kappa": (
            f"κ {hi['kappa_binary']['value']:.2f} at k ≥ 10 against "
            f"{lo['kappa_binary']['value']:.2f} below"
            + (
                f", over {_fmt(hi['residue_pairs'])} and "
                f"{_fmt(lo['residue_pairs'])} residue pairs of replicate "
                "depositions"
                if hi.get("residue_pairs") and lo.get("residue_pairs")
                else ""
            )
            if hi and lo
            else "not recorded"
        ),
        "k_flip": (
            f"{100 * hi['disordered_flip']['value']:.1f} % at k ≥ 10 against "
            f"{100 * lo['disordered_flip']['value']:.1f} % below"
            if hi and lo
            else "not recorded"
        ),
    }


def straddling_entries(path: Path) -> list[str]:
    """``entry_id`` values whose rows cross a row-group boundary of a Parquet
    sorted by ``entry_id`` (from the row-group statistics)."""
    import pyarrow.parquet as pq

    md = pq.ParquetFile(path).metadata
    names = [md.schema.column(i).name for i in range(md.num_columns)]
    if "entry_id" not in names:
        return []
    col = names.index("entry_id")
    bounds = []
    for i in range(md.num_row_groups):
        st = md.row_group(i).column(col).statistics
        if st is None or not st.has_min_max:
            return []
        bounds.append((st.min, st.max))
    return sorted(
        {
            bounds[i][1]
            for i in range(len(bounds) - 1)
            if bounds[i][1] == bounds[i + 1][0]
        }
    )


def _retired_explained(reasons: dict) -> str:
    if not reasons:
        return "No chain is retired."
    return "; ".join(
        f"`{r}` ({_fmt(n)}): {ri.describe_retired_reason(r)}"
        for r, n in reasons.items()
    )


def changes_and_issues(
    rows: list[dict],
    bundle: Path,
    root: Path,
    *,
    previous_build: Path | None = None,
    previous_label: str | None = None,
    pkl_dir: Path | None = None,
    bmrb_dir: Path | None = None,
    snapshot_drift: Path | None = None,
    construct_caveats: Path | None = None,
    provenance: dict | None = None,
    reviewed: Path | None = None,
    ionic_drivers: Path | None = None,
) -> tuple[dict[str, str], list[tuple[str, str, str]], dict]:
    """The CHANGELOG / README values of the changes and known issues, the rows
    of ``known_issues.tsv`` and a summary for MANIFEST (:mod:`known_issues`)."""
    rows_by_id = {r["id"]: r for r in rows}
    issues: list[tuple[str, str, str]] = []
    assessed = {}
    cur = ki.load_build(bundle)
    if previous_build is not None:
        prev = ki.load_build(previous_build, labels=True)
        label = previous_label or (
            "the previous build "
            + (ri.bundle_pipeline_version(previous_build) or Path(previous_build).name)
        )
        moves = ki.tier_moves(prev, cur, rows_by_id)
        lab = ki.label_changes(prev, rows)
        values = {
            "previous_build": label,
            "tier_moves_table": ki.tier_moves_table(moves),
            "rescore_summary": ki.rescore_text(lab, label),
            "chezod117_placeholders": ki.chezod_placeholder_text(lab, rows),
            "alphabet_changes": ki.alphabet_text(lab),
        }
        issues += (
            ki.tier_move_rows(moves) + ki.rescore_rows(lab) + ki.alphabet_rows(lab)
        )
        assessed["previous_build"] = label
    else:
        missing = "not assessed: finalize ran without --previous-build"
        values = {
            "previous_build": "the previous build (not given)",
            "tier_moves_table": f"Tier moves {missing}.",
            "rescore_summary": f"Re-score statistics {missing}.",
            "chezod117_placeholders": ki.chezod_placeholder_text(
                {"per_chain": {}}, rows
            ),
            "alphabet_changes": f"Sequence changes {missing}.",
        }
    if pkl_dir is not None:
        cond_rows = ki.condition_issues(rows, pkl_dir, bmrb_dir)
        text = ki.condition_text(cond_rows, cur, rows_by_id)
        values.update(
            ph_scale_summary=text["ph"],
            temperature_summary=text["temperature"],
            pressure_unit_summary=text["pressure"],
        )
        issues += cond_rows
        assessed["conditions"] = f"BmrbEntry pickles, {text['unread']} chains unread"
    else:
        missing = "not assessed for this deposit (finalize ran without --pkl-dir)"
        values.update(
            ph_scale_summary=f"pH* and pD chains {missing}.",
            temperature_summary=f"Temperature unit corrections {missing}.",
            pressure_unit_summary=f"Pressures without a unit {missing}.",
        )
    drift = None
    if snapshot_drift is not None:
        drift = ki.snapshot_drift(snapshot_drift, rows_by_id)
        issues += drift["rows"]
        assessed["snapshot_drift"] = Path(snapshot_drift).name
    values["snapshot_drift_summary"] = ki.snapshot_drift_text(drift)
    caveats = None
    if construct_caveats is not None:
        caveats = ki.construct_caveats(construct_caveats, rows)
        issues += caveats["rows"]
        assessed["construct_caveats"] = Path(construct_caveats).name
    values["construct_caveats"] = ki.construct_text(
        caveats, ki.nonstandard_counts(rows)
    )
    reviewed_path = Path(reviewed) if reviewed is not None else root / ki.REVIEWED_TSV
    reviewed_rows = (
        ki.reviewed_issues(reviewed_path, rows_by_id) if reviewed_path.is_file() else []
    )
    issues += reviewed_rows
    values["reviewed_issues"] = ki.reviewed_text(reviewed_rows)
    chezod = ki.chezod117_conditions(rows)
    issues += chezod["rows"]
    values["chezod117_conditions"] = chezod["text"]
    if ionic_drivers is not None:
        drivers = ki.ionic_drivers(ionic_drivers, rows_by_id)
        issues += drivers["rows"]
        values["ionic_drivers"] = drivers["text"]
        assessed["ionic_drivers"] = Path(ionic_drivers).name
    else:
        values["ionic_drivers"] = (
            "Ionic strengths are used as deposited, also where the sample's own "
            "components imply another value (per-chain list not assessed for "
            "this deposit: finalize ran without --ionic-drivers)."
        )
    pin = ki.pin_trimmed(provenance, rows_by_id)
    issues += pin["rows"]
    values["pin_trimmed"] = pin["text"]
    few = ki.few_scored_training(rows)
    issues += few
    values["few_scored_training"] = (
        f"{_fmt(len(few))} training representatives have fewer than "
        f"{ri.K_MIN_RELIABLE} scored residues"
        + (f" ({', '.join(cid for cid, _, _ in few)})" if few else "")
    )
    from collections import Counter

    by_issue = dict(sorted(Counter(i for _, i, _ in set(issues)).items()))
    values["n_known_issues"] = _fmt(len(set(issues)))
    values["known_issue_counts"] = (
        ", ".join(f"`{k}` {_fmt(n)}" for k, n in by_issue.items()) or "none"
    )
    summary = {"file": "known_issues.tsv", "by_issue": by_issue, "inputs": assessed}
    return values, issues, summary


def doc_values(
    meta: dict[str, str],
    summary: dict,
    bundle_manifest: dict,
    class_table: list[dict],
    calibration: dict | None,
    shifts: dict,
    snapshot: dict | None,
    crosswalk: dict,
    *,
    scoring: dict | None = None,
    straddling: list[str] | None = None,
    validation: bool = False,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Every ``{{placeholder}}`` value of the docs templates, from this build."""
    test_set = _meta_json(meta, "test_set") or {}
    commit = _meta_json(meta, "code_commit") or {}
    scoring = scoring or {}
    straddling = straddling or []
    leakage = bundle_manifest.get("leakage", {})
    rule = leakage.get("rule")
    split = summary["split_counts"]
    reasons = summary["retired_by_reason"]
    snap = snapshot or {}
    values = {
        "version": meta.get("dataset_version", "unknown"),
        "previous_version": ri.PREVIOUS_VERSION,
        "previous_version_date": ri.PREVIOUS_VERSION_DATE,
        "concept_doi": ri.CONCEPT_DOI,
        "repository": ri.REPOSITORY,
        "data_license": ri.DATA_LICENSE_NAME,
        "data_license_url": ri.DATA_LICENSE_URL,
        "code_license": ri.CODE_LICENSE,
        "bmrb_acknowledgement": ri.BMRB_ACKNOWLEDGEMENT,
        "test_set_name": ri.TEST_SET_NAME,
        "test_set_label": test_set.get(
            "label",
            ri.testset_label(split.get("test_trizod", 0), split.get("test_retired", 0)),
        ),
        "n_test": _fmt(split.get("test_trizod", 0)),
        "n_retired": _fmt(split.get("test_retired", 0)),
        "retired_by_reason": ", ".join(
            f"{_fmt(n)} {reason}" for reason, n in reasons.items()
        )
        or "none",
        "n_chezod117": _fmt(split.get("test_chezod117", 0)),
        "n_test_clean_chezod1325": _fmt(summary["test_clean_vs"]["chezod1325"]),
        "n_test_clean_chezod1174": _fmt(summary["test_clean_vs"]["chezod1174"]),
        "n_rows": _fmt(summary["rows"]),
        "n_columns": _fmt(summary["columns"]),
        "n_columns_previous": _fmt(V030["columns"]),
        "n_test_previous": _fmt(V030["test_trizod"]),
        "n_new_columns": _fmt(len(summary["new_columns"])),
        "new_columns": ", ".join(f"`{c}`" for c in summary["new_columns"]),
        "n_shifts": _fmt(shifts["n_rows"]),
        "n_shift_chains": _fmt(shifts["n_chains"]),
        "n_train_tolerant": _fmt(summary["train_reps"]["tolerant"]),
        "tier_table": _tier_table(bundle_manifest, summary["train_reps"]),
        "changelog_tier_table": _changelog_tier_table(
            bundle_manifest, summary["train_reps"]
        ),
        "split_table": _split_table(split),
        "rule_table": _rule_table(summary["rule_flags"]),
        "n_temperature_placeholder": _fmt(
            summary["rule_flags"]["temperature_placeholder"]
        ),
        "class_balance_table": cb.markdown(class_table),
        "class_balance_notes": cb.notes(),
        "crosswalk_table": _crosswalk_table(crosswalk),
        "scored_residues": _fmt(summary["scored_residues"]),
        "share_k_lt_10": (
            f"{100 * summary['share_k_lt_10']:.1f} %"
            if summary["share_k_lt_10"] is not None
            else "–"
        ),
        "cut_ordered": ri.cut_label(ri.ORDERED_CUT),
        "cut_disordered": ri.cut_label(ri.DISORDER_CUT),
        "cut_confident": ri.cut_label(ri.CONFIDENT_DISORDER_CUT),
        "cut_rule": ri.CUT_RULE,
        "code_commit": commit.get("commit", "unknown")
        + ("-dirty" if commit.get("dirty") else ""),
        "pipeline_version": commit.get("pipeline_version", "unknown"),
        "scoring_commit": (scoring.get("commit") or "not recorded")
        + ("-dirty" if scoring.get("dirty") else ""),
        "scoring_pipeline_version": scoring.get("pipeline_version") or "not recorded",
        "retired_reasons_explained": _retired_explained(reasons),
        "row_group_note": (
            f"one row group, or two for the {_fmt(len(straddling))} entries whose "
            f"rows cross a group boundary ({', '.join(straddling)})"
            if straddling
            else "one row group"
        ),
        "validation_row": (
            "| `VALIDATION.json` | Results of the release audit this deposit "
            "passed, check by check |"
            if validation
            else ""
        ),
        "snapshot_name": snap.get("name", "not recorded"),
        "snapshot_n_entries": (
            _fmt(snap["n_entries"]) if "n_entries" in snap else "not recorded"
        ),
        "snapshot_retrieved": snap.get("retrieved", "not recorded"),
        "entry_list_sha256": snap.get("entry_list_sha256", "not recorded"),
        "leakage_rule": describe_rule(rule) if rule else "not recorded",
    }
    values.update(_calibration_values(calibration))
    values.update(extra or {})
    return values


def license_text() -> str:
    return (
        f"# Licence\n\n"
        f"**Data** (every file of this deposit): {ri.DATA_LICENSE_NAME}, "
        f"{ri.DATA_LICENSE_URL}. You may share and adapt the data for any "
        "purpose, provided you give appropriate credit (cite the dataset, see "
        "`CITATION.cff`), link to the licence and indicate if changes were "
        "made.\n\n"
        f"**Code** that builds it: {ri.CODE_LICENSE}, {ri.REPOSITORY}.\n\n"
        f"**Source data.** {ri.BMRB_ACKNOWLEDGEMENT}\n"
    )


# --------------------------------------------------------------------------- #
# consistency checks
# --------------------------------------------------------------------------- #


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"finalize: {msg}")


def check_test_fastas(test_dir: Path, rows: list[dict]) -> None:
    """The bundle's test FASTAs list exactly the Parquet's test rows."""
    by_split: dict[str, set[str]] = {}
    for r in rows:
        by_split.setdefault(r["split"], set()).add(r["id"])
    for fasta, split in (
        ("TriZOD_test_set.fasta", "test_trizod"),
        (RETIRED_FASTA, "test_retired"),
        ("CheZOD117_test_set.fasta", "test_chezod117"),
    ):
        path = test_dir / fasta
        ids = set(read_fasta(path)) if path.exists() else set()
        want = by_split.get(split, set())
        _check(
            ids == want,
            f"{path} lists {len(ids)} chains, the Parquet has {len(want)} "
            f"{split} rows ({len(ids ^ want)} differ, e.g. "
            f"{sorted(ids ^ want)[:5]}): the Parquet was not built from this bundle",
        )


def data_fingerprint(path: Path) -> str:
    """SHA-256 of a Parquet's data alone: every column and row serialised as
    one Arrow IPC batch, the key-value metadata dropped. Equal for two builds
    that differ only in metadata or row-group layout (NaN compares by its
    bits, which ``Table.equals`` would not)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path).replace_schema_metadata(None).combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def locate_calibration_input(recorded, explicit=None) -> Path | None:
    """The Parquet the threshold framework calibrated on: ``explicit``
    (``--calibration-input``), else the recorded path -- absolute (older
    builds), or relative to the data root (``$TRIZOD_DATA_ROOT``) or this
    checkout. None when it is not on this machine."""
    if explicit is not None:
        return Path(explicit)
    if not recorded:
        return None
    p = Path(recorded)
    candidates = (
        [p]
        if p.is_absolute()
        else [
            repo_paths.data_root() / p,
            repo_paths.repo_root() / p,
        ]
    )
    return next((c for c in candidates if c.is_file()), None)


def check_calibration_input(
    parquet: Path,
    calibration: dict | None,
    allow_mismatch: bool = False,
    input_path: Path | None = None,
) -> dict | None:
    """The threshold framework calibrates on the release Parquet before its
    metadata is stamped, so the final Parquet's bytes differ from the input it
    records. Check that their data are identical (every column and row; the
    key-value metadata ignored). ``data_identical`` is None when the input is
    not on this machine (pass ``input_path``, ``--calibration-input``)."""
    if not calibration:
        return None
    recorded = calibration.get("input_parquet")
    src = locate_calibration_input(recorded, input_path)
    out = {"input_parquet": recorded, "data_identical": None}
    if src is None or not src.is_file():
        print(
            f"  note: calibration input {recorded} not found (pass "
            "--calibration-input) -- data identity not checked"
        )
        return out
    out["data_identical"] = data_fingerprint(parquet) == data_fingerprint(src)
    _check(
        out["data_identical"] or allow_mismatch,
        f"the calibration was computed on {src}, whose data differ from "
        f"{parquet}: rebuild the Parquet with the inputs of the calibrated build, "
        "or pass --allow-calibration-mismatch for a dry run",
    )
    return out


#: The placeholder CITATION.cff carries until the Zenodo draft reserves the
#: version DOI.
DOI_PLACEHOLDER = "RESERVE-AT-DRAFT"
_CFF_DOI = re.compile(
    r"-\s*type:\s*doi\s*\n\s*value:\s*(?P<doi>\S+)\s*\n\s*description:\s*"
    r"\"?(?P<desc>[^\n\"]*)"
)


def version_doi(cff_text: str, version: str) -> str | None:
    """The version DOI ``CITATION.cff`` lists for ``version``; None while it is
    the placeholder or absent."""
    for m in _CFF_DOI.finditer(cff_text):
        if f"v{version}" in m.group("desc").split():
            doi = m.group("doi")
            return None if DOI_PLACEHOLDER in doi else doi
    return None


def check_repo_metadata(version: str, root: Path) -> None:
    """``CITATION.cff``, ``.zenodo.json`` and ``pyproject.toml`` state the
    release's version and the data licence (G36)."""
    cff = (root / "CITATION.cff").read_text()
    _check(f'version: "{version}"' in cff, f"CITATION.cff does not state {version}")
    _check(
        f"license: {ri.DATA_LICENSE}" in cff,
        f"CITATION.cff does not state the data licence {ri.DATA_LICENSE}",
    )
    zen = json.loads((root / ".zenodo.json").read_text())
    _check(zen.get("version") == version, f".zenodo.json version != {version}")
    _check(
        str(zen.get("license", "")).lower() == ri.DATA_LICENSE.lower(),
        f".zenodo.json licence != {ri.DATA_LICENSE}",
    )
    pyproject = (root / "pyproject.toml").read_text()
    _check(f'version = "{version}"' in pyproject, f"pyproject.toml != {version}")


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def assemble(
    parquet: Path,
    bundle: Path,
    out: Path,
    *,
    shifts: Path,
    bmrb_dir: Path,
    crosswalk_specs: list[tuple[str, str, str]] | None = None,
    crosswalk_root: Path | None = None,
    validation: Path | None = None,
    root: Path | None = None,
    allow_uncalibrated: bool = False,
    allow_calibration_mismatch: bool = False,
    force: bool = False,
    previous_build: Path | None = None,
    previous_label: str | None = None,
    pkl_dir: Path | None = None,
    snapshot_drift: Path | None = None,
    construct_caveats: Path | None = None,
    scoring_commit: str | None = None,
    calibration_input: Path | None = None,
    reviewed: Path | None = None,
    ionic_drivers: Path | None = None,
) -> dict:
    """Assemble the deposit into ``out``; returns the MANIFEST. ``reviewed`` is
    the hand-reviewed known-issue table (default:
    ``docs/dataset/known_issues_reviewed.tsv`` of ``root``)."""
    parquet, bundle, out = Path(parquet), Path(bundle), Path(out)
    root = Path(root) if root is not None else repo_paths.repo_root()
    docs = root / "docs" / "dataset"

    rows, meta, names = read_parquet(parquet)
    version = meta.get("dataset_version")
    _check(
        version is not None,
        f"{parquet} has no dataset_version metadata: rebuild it with "
        "scripts/build_parquet_dataset.py from this checkout",
    )
    calibration = cb.parse_calibration(meta.get("calibration"))
    _check(
        calibration is not None or allow_uncalibrated,
        f"{parquet} carries no calibration (build it with --calibration "
        "<framework dir>), or pass --allow-uncalibrated for a dry run",
    )
    check_repo_metadata(version, root)
    doi = version_doi((root / "CITATION.cff").read_text(), version)
    if doi is None:
        print(
            f"  WARNING: CITATION.cff lists no version DOI for v{version} (still "
            f"{DOI_PLACEHOLDER}): reserve it on the Zenodo draft, write it into "
            "CITATION.cff and .zenodo.json, and re-run finalize"
        )
    calibration_input = check_calibration_input(
        parquet, calibration, allow_calibration_mismatch, calibration_input
    )
    bundle_manifest = json.loads((bundle / "MANIFEST.json").read_text())
    scoring = ri.scoring_block(
        scoring_commit or bundle_manifest.get("pipeline_version"), root
    )
    parquet_scoring = _meta_json(meta, "scoring_commit")
    _check(
        parquet_scoring is None
        or scoring_commit is not None
        or parquet_scoring.get("pipeline_version") == scoring["pipeline_version"],
        f"{parquet} records scoring commit "
        f"{(parquet_scoring or {}).get('pipeline_version')}, the bundle "
        f"{scoring['pipeline_version']}: the Parquet was not built from this bundle",
    )
    check_test_fastas(bundle / "test", rows)
    summary = summarise_rows(rows, names)
    tiers = bundle_manifest.get("tiers", {})
    for t in TIER_ORDER:
        n = tiers.get(t, {}).get("training_reps")
        _check(
            n is None or n == summary["train_reps"][t],
            f"{t}: {n} training representatives in the bundle MANIFEST, "
            f"{summary['train_reps'][t]} in the Parquet",
        )

    # every input that can fail is read before the output folder is touched
    if crosswalk_specs is None:
        crosswalk_specs = testset_crosswalk.default_set_specs(crosswalk_root or root)
    sets = testset_crosswalk.load_sets(crosswalk_specs)
    snapshot = _meta_json(meta, "bmrb_snapshot")
    entry_files = snapshot_entry_files(bmrb_dir)
    if snapshot and "n_entries" in snapshot:
        _check(
            len(entry_files) == snapshot["n_entries"],
            f"{bmrb_dir} holds {len(entry_files)} entry files, the snapshot the "
            f"Parquet names records {snapshot['n_entries']}",
        )

    if out.exists() and any(out.iterdir()):
        _check(force, f"{out} is not empty (pass --force to replace it)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. the tables
    _copy(parquet, out / "trizod_dataset.parquet")
    _copy(shifts, out / "trizod_shifts.parquet")
    shifts_entry = file_entry(out / "trizod_shifts.parquet")

    # 2. test-set files
    test_out = out / "test"
    for src in sorted((bundle / "test").glob("*.fasta")):
        _copy(src, test_out / src.name)
    labels = bundle / "test" / "TriZOD_test_set_labels.tsv"
    if labels.exists():
        _copy(labels, test_out / labels.name)
    provenance = root / "trizod" / "dataset" / "pinned"
    provenance = provenance / "TriZOD_test_set.provenance.json"
    if provenance.exists():
        _copy(provenance, test_out / provenance.name)

    # 3. views, regenerated from this build
    deploy_fasta.main(
        [
            "--root",
            str(root),
            "--scored-dir",
            str(bundle / "scores"),
            "--reps-dir",
            str(bundle / "train"),
            "--test-fasta",
            str(bundle / "test" / "TriZOD_test_set.fasta"),
            "--out",
            str(out / "disorder_trizod.fasta"),
        ]
    )
    deploy = json.loads((out / "disorder_trizod.summary.json").read_text())
    _check(
        deploy["n_train"] == summary["train_reps"]["tolerant"]
        and deploy["n_test"] == summary["split_counts"].get("test_trizod", 0),
        "the deploy FASTA disagrees with the Parquet's tolerant training set "
        "or test set",
    )
    export_jsonl.export(out / "trizod_dataset.parquet", out / "udonpred")
    crosswalk = testset_crosswalk.run_on_parquet(
        sets,
        out / "trizod_dataset.parquet",
        out / "crosswalk",
        release_path="trizod_dataset.parquet",
    )
    # deploy_fasta, export_jsonl and testset_crosswalk record the paths they
    # read and wrote; no path of this machine may ship (G35)
    anchors = [
        out,
        (bundle, "<package bundle>/"),
        crosswalk_root,
        repo_paths.data_root(),
        root,
    ]
    for rel in (
        "disorder_trizod.summary.json",
        "udonpred/export_summary.json",
        "crosswalk/test_set_crosswalk.json",
    ):
        if (out / rel).is_file():
            scrub_json_paths(out / rel, anchors)

    # 4. provenance tables
    files_table = snapshot_files_table(entry_files, out / "bmrb_files.sha256")
    class_table = cb.compute(rows, calibration)
    cb.write_tsv(class_table, out / "class_balance.tsv")
    validation_entry = None
    if validation is not None:
        _copy(Path(validation), out / "VALIDATION.json")
        validation_entry = {
            "file": "VALIDATION.json",
            "source_sha256": ri.sha256_file(validation),
            "paths_rewritten": scrub_json_paths(out / "VALIDATION.json", anchors),
        }
    pin_provenance = test_out / "TriZOD_test_set.provenance.json"
    changes, issue_rows, issues_summary = changes_and_issues(
        rows,
        bundle,
        root,
        previous_build=previous_build,
        previous_label=previous_label,
        pkl_dir=pkl_dir,
        bmrb_dir=bmrb_dir,
        snapshot_drift=snapshot_drift,
        construct_caveats=construct_caveats,
        provenance=(
            json.loads(pin_provenance.read_text()) if pin_provenance.is_file() else None
        ),
        reviewed=reviewed,
        ionic_drivers=ionic_drivers,
    )
    ki.write_tsv(issue_rows, out / "known_issues.tsv")

    # 5. documents
    values = doc_values(
        meta,
        summary,
        bundle_manifest,
        class_table,
        calibration,
        shifts_entry,
        snapshot,
        crosswalk,
        scoring=scoring,
        straddling=straddling_entries(out / "trizod_dataset.parquet"),
        validation=validation is not None,
        extra=changes,
    )
    for template, name in TEMPLATES.items():
        text = (docs / template).read_text()
        (out / name).write_text(render_template(text, values, template))
    columns = (docs / COLUMNS_DOC).read_text()
    missing = [c for c in names if f"`{c}`" not in columns]
    _check(not missing, f"{COLUMNS_DOC} does not document {missing}")
    (out / COLUMNS_DOC).write_text(columns)
    (out / "LICENSE.md").write_text(license_text())
    _copy(root / "CITATION.cff", out / "CITATION.cff")
    _copy(root / ".zenodo.json", out / "zenodo.json")

    # 6. MANIFEST + checksums
    files = {
        p.relative_to(out).as_posix(): file_entry(p)
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    manifest = {
        "dataset": "TriZOD",
        "version": version,
        "license": {
            "data": ri.DATA_LICENSE,
            "code": ri.CODE_LICENSE,
            "statement": ri.LICENSE_STATEMENT,
        },
        "version_doi": doi,
        "concept_doi": ri.CONCEPT_DOI,
        "test_set": _meta_json(meta, "test_set"),
        "cuts": _meta_json(meta, "cuts"),
        "code": {
            "scoring": scoring,
            "build": _meta_json(meta, "code_commit"),
            "package_pipeline_version": bundle_manifest.get("pipeline_version"),
            "finalize": ri.code_commit(root),
            "roles": (
                "scoring: scored the entries and ran the dataset chain into the "
                "package bundle (labels, tiers, splits, shift offsets); build: "
                "built trizod_dataset.parquet from that bundle; finalize: "
                "assembled this deposit and regenerated its views and documents"
            ),
        },
        "threshold_protocol": (
            {
                "framework": calibration.get("framework"),
                "framework_commit": calibration.get("framework_commit"),
                "results_json_sha256": calibration.get("results_json_sha256"),
                "input_parquet_sha256": calibration.get("input_parquet_sha256"),
                "input_data_identical_to_release": (calibration_input or {}).get(
                    "data_identical"
                ),
                "cuts": list(ri.CUTS),
                "rule": calibration.get("rule"),
            }
            if calibration
            else None
        ),
        "calibration": calibration,
        "bmrb_snapshot": {
            **(snapshot or {}),
            "files_table": "bmrb_files.sha256",
            **files_table,
        },
        "xref_sources": _meta_json(meta, "xref_sources"),
        "leakage": bundle_manifest.get("leakage"),
        "parquet": {
            "data_sha256": data_fingerprint(out / "trizod_dataset.parquet"),
            "data_sha256_definition": "SHA-256 of the table's data alone "
            "(one Arrow IPC batch, key-value metadata dropped)",
            "rows": summary["rows"],
            "columns": summary["columns"],
            "split_counts": summary["split_counts"],
            "training_representatives": summary["train_reps"],
            "tier_scored_records": {
                t: tiers.get(t, {}).get("scored_records") for t in TIER_ORDER
            },
            "retired_by_reason": summary["retired_by_reason"],
        },
        "known_issues": issues_summary,
        "validation": validation_entry,
        "n_files": len(files) + 2,  # + MANIFEST.json, SHA256SUMS
        "total_bytes": sum(f["bytes"] for f in files.values()),
        "total_bytes_covers": TOTAL_BYTES_COVERS,
        "files": files,
    }
    manifest = _scrub(manifest, anchors)
    _json(out / "MANIFEST.json", manifest)
    write_checksums(out)
    stray = machine_paths(
        out,
        [
            out,
            bundle,
            crosswalk_root,
            repo_paths.data_root(),
            root,
            parquet.parent,
            Path(shifts).parent,
            Path(bmrb_dir),
        ],
    )
    _check(
        not stray,
        f"absolute paths of this machine in the deposit: {stray[:10]} -- rebuild "
        "the Parquet with this checkout, or make the writer record a relative path",
    )
    return manifest


def _parse_set(item: str) -> tuple[str, str, str]:
    name, sep, spec = item.partition("=")
    if not sep:
        raise SystemExit(f"--set expects NAME=SPEC: {item!r}")
    return name, spec, spec


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parquet", type=Path, required=True, help="release Parquet")
    ap.add_argument("--shifts", type=Path, required=True, help="trizod_shifts.parquet")
    ap.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="package bundle the Parquet was built from (trizod dataset package)",
    )
    ap.add_argument(
        "--bmrb-dir",
        type=Path,
        required=True,
        help="BMRB snapshot directory (bmr<ID>/bmr<ID>_3.str) for bmrb_files.sha256",
    )
    ap.add_argument("--out", type=Path, required=True, help="release folder")
    ap.add_argument(
        "--crosswalk-root",
        type=Path,
        default=None,
        help="checkout holding data/ for the default earlier test sets "
        "(default: this repository)",
    )
    ap.add_argument(
        "--set",
        dest="sets",
        action="append",
        metavar="NAME=SPEC",
        help="replace the crosswalk's earlier test sets; SPEC is fasta:PATH, "
        "parquet:PATH or git:REV:PATH (repeatable)",
    )
    ap.add_argument(
        "--validation",
        type=Path,
        default=None,
        help="VALIDATION.json of the final audit, shipped as is",
    )
    ap.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="assemble a Parquet built without --calibration (dry runs only)",
    )
    ap.add_argument(
        "--allow-calibration-mismatch",
        action="store_true",
        help="accept a Parquet whose data differ from the calibration's input "
        "(dry runs only)",
    )
    ap.add_argument(
        "--previous-build",
        type=Path,
        default=None,
        help="package bundle (or scored dir) of the previous build of record: "
        "tier moves, 273 K re-score statistics and sequence changes against it",
    )
    ap.add_argument(
        "--previous-label",
        default=None,
        help="how the CHANGELOG names --previous-build (default: its pipeline_version)",
    )
    ap.add_argument(
        "--pkl-dir",
        type=Path,
        default=None,
        help="the build's BmrbEntry pickle cache (tmp/bmrb_entries): pH* / pD, "
        "temperature and pressure-unit issues per chain",
    )
    ap.add_argument(
        "--snapshot-drift",
        type=Path,
        default=None,
        help="snapshot-drift table of the release audit (id, released, "
        "label_or_tier_change, per_tier)",
    )
    ap.add_argument(
        "--construct-caveats",
        type=Path,
        default=None,
        help="construct / sample caveat table of the release audit (class, id)",
    )
    ap.add_argument(
        "--ionic-drivers",
        type=Path,
        default=None,
        help="ionic-strength driver log of the release audit (chain, flag, "
        "I_deposited_strict_M, I_implied_M, implied_parts)",
    )
    ap.add_argument(
        "--scoring-commit",
        default=None,
        help="commit that scored the entries and wrote the bundle (default: the "
        "bundle MANIFEST's pipeline_version)",
    )
    ap.add_argument(
        "--calibration-input",
        type=Path,
        default=None,
        help="the Parquet the threshold framework calibrated on, when its "
        "recorded path does not resolve here",
    )
    ap.add_argument("--force", action="store_true", help="replace a non-empty --out")
    args = ap.parse_args(argv)
    specs = [_parse_set(s) for s in args.sets] if args.sets else None
    manifest = assemble(
        args.parquet,
        args.bundle,
        args.out,
        shifts=args.shifts,
        bmrb_dir=args.bmrb_dir,
        crosswalk_specs=specs,
        crosswalk_root=args.crosswalk_root,
        validation=args.validation,
        allow_uncalibrated=args.allow_uncalibrated,
        allow_calibration_mismatch=args.allow_calibration_mismatch,
        force=args.force,
        previous_build=args.previous_build,
        previous_label=args.previous_label,
        pkl_dir=args.pkl_dir,
        snapshot_drift=args.snapshot_drift,
        construct_caveats=args.construct_caveats,
        ionic_drivers=args.ionic_drivers,
        scoring_commit=args.scoring_commit,
        calibration_input=args.calibration_input,
    )
    print(f"\nAssembled TriZOD v{manifest['version']} -> {args.out}")
    print(
        f"  files: {manifest['n_files']}, {manifest['total_bytes'] / 1e6:.1f} MB "
        "(without MANIFEST.json and SHA256SUMS)"
    )
    print(f"  scoring commit: {manifest['code']['scoring']['commit']}")
    print(f"  known issues: {manifest['known_issues']['by_issue']}")
    ts = manifest["test_set"] or {}
    print(f"  test set: {ts.get('label', 'unknown')}")
    print("  check: (cd <out> && sha256sum -c SHA256SUMS)")
    print("Nothing uploaded.")


if __name__ == "__main__":
    main()
