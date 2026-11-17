#!/usr/bin/env python3
"""Export the UdonPred v2 training view (JSONL) from the release Parquet.

The canonical release is ONE Parquet, one row per chain. This module writes the
view UdonPred v2 (github.com/DavidWagemann/UdonPred) trains on, generated from
that file and nothing else, so check F29 can prove the two agree::

    <out>/trizod_<tier>/{train,valid,test}.jsonl   one folder per training tier
    <out>/trizod_<tier>/{train,valid,test}.fasta   same records, for predict.py
    <out>/chezod117/test.{jsonl,fasta}
    <out>/README.md, <out>/export_summary.json

Record schema (UdonPred v2's, as in its figshare training data and read by its
code)::

    {"id": "<chain id>", "x_0": "<sequence>", "y": [g_1, ..., g_N]}

* ``x_0`` -- the loader treats every key starting with ``x_`` as a sequence
  input (``model/data.py`` ``extract_texts``: ``k.startswith("x_")``) and the
  evaluation notebook reads the sequence as ``x["x_0"]`` and the ID as
  ``x["id"]`` (``eval/eval.ipynb``).
* ``y`` -- the per-residue target, the continuous TriZOD G (``gscores``), with
  ``999.0`` wherever the Parquet ``mask`` is False or G is NaN. ``999`` is
  UdonPred's mask value (``model/losses.py`` ``flatten_and_mask(...,
  mask_value=999)``; ``DataCollator(padding=999)``). G values are written as
  the shortest decimal that round-trips the Parquet's float32, so
  ``float32(y) == gscores`` exactly.
* No other keys. The loader would carry an extra string key through, but a
  per-residue list (``k``, say) would be padded with 999 and reach the model's
  ``forward(**kw)``; the trainer only reads the features named in
  ``config/data.yaml``. Matching the deposited schema exactly is the safe choice.

Membership (by the Parquet's ``split`` / ``train_tier`` / ``cluster_repr``):

* train + valid -- the tier's training representatives: ``split == "train"``
  with ``train_tier`` at or stricter than the tier (``TIER_ORDER``). Redundant
  cluster members are never exported.
* valid -- a seeded draw of whole clusters (``cluster_repr``) from those
  representatives: ``round(fraction * n_clusters)`` clusters, drawn with
  ``random.Random(seed).sample`` from the naturally sorted cluster IDs.
  UdonPred v2 ships pre-split files and states no fraction in code (v1 used
  5-fold CV), so the default is 10 %. In the release every training
  representative heads its own 50 %/80 % mmseqs cluster, so the draw is per
  cluster by construction; grouping by ``cluster_repr`` keeps it so if a
  release ever lists several representatives per cluster.
* test -- ``split == "test_trizod"`` only. ``test_retired`` (pinned test chains
  that no longer meet strict) is neither test nor training and is never
  exported.
* chezod117 -- ``split == "test_chezod117"``; ``y`` is TriZOD G for those
  chains, not CheZOD Z.

Labels come from the Parquet row, i.e. from the ``label_tier_source`` tier's
record (``trizod.dataset.tiers.LABEL_SOURCE_RULE``), whichever tier is exported.

Usage
-----
    uv run --with pyarrow trizod dataset export-jsonl \\
        --parquet <trizod_dataset.parquet> [--out DIR] \\
        [--tier tolerant --tier moderate --tier strict] \\
        [--valid-fraction 0.10] [--seed 42]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from trizod import paths
from trizod.dataset.tiers import LABEL_SOURCE_RULE, TIER_ORDER

#: UdonPred's per-residue mask value (``flatten_and_mask(..., mask_value=999)``).
MASK_VALUE = 999.0
#: Keys of every JSONL record, in order.
RECORD_FIELDS = ("id", "x_0", "y")
#: Training tiers exported by default; UdonPred v1 and v2 trained on tolerant.
DEFAULT_TIERS = ("tolerant", "moderate", "strict")
DEFAULT_VALID_FRACTION = 0.10
DEFAULT_SEED = 42

TRAIN_SPLIT = "train"
TEST_SPLIT = "test_trizod"
CHEZOD_SPLIT = "test_chezod117"
RETIRED_SPLIT = "test_retired"
CHEZOD_FOLDER = "chezod117"
SPLIT_FILES = ("train", "valid", "test")

#: Parquet columns the view is built from (``label_tier_source`` is optional).
READ_COLUMNS = (
    "id",
    "sequence",
    "gscores",
    "mask",
    "split",
    "train_tier",
    "cluster_repr",
    "label_tier_source",
)
_REQUIRED = frozenset(READ_COLUMNS) - {"label_tier_source"}

#: Parquet key-value metadata copied into ``export_summary.json``.
_META_KEYS = re.compile(
    r"version|commit|snapshot|split_values|cut|threshold|mask_semantics|label_source"
)

# G range with the same float tolerance as trizod.dataset.deploy_fasta
_G_LO, _G_HI = -1e-6, 1.0 + 1e-6


def natural_key(rec_id: str) -> tuple:
    """Sort key for IDs like ``15036_1_1_1`` that never compares int with str."""
    return tuple(
        (0, int(p), "") if p.isdigit() else (1, 0, p) for p in rec_id.split("_")
    )


def tier_folder(tier: str) -> str:
    return f"trizod_{tier}"


# --------------------------------------------------------------------------- #
# labels and records
# --------------------------------------------------------------------------- #


def label_values(gscores, mask) -> list[float]:
    """Per-residue ``y``: G where labelled, :data:`MASK_VALUE` where not.

    A residue is masked where ``mask`` is False, or G is None/NaN. Labelled
    values are the shortest decimals that round-trip the float32 G.
    """
    if len(gscores) != len(mask):
        raise ValueError(f"gscores ({len(gscores)}) and mask ({len(mask)}) differ")
    g32 = np.array([np.nan if g is None else g for g in gscores], dtype=np.float32)
    out: list[float] = []
    for g, m in zip(g32, mask):
        if not m or np.isnan(g):
            out.append(MASK_VALUE)
            continue
        if not _G_LO <= g <= _G_HI:
            raise ValueError(f"G = {g} outside [0, 1]")
        out.append(float(str(g)))  # numpy's shortest float32 round-trip repr
    return out


def to_record(row: dict) -> dict:
    """One UdonPred JSONL record ``{"id", "x_0", "y"}`` from a Parquet row."""
    seq = row["sequence"]
    y = label_values(row["gscores"], row["mask"])
    if len(y) != len(seq):
        raise ValueError(f"{row['id']}: {len(y)} labels for {len(seq)} residues")
    return {"id": row["id"], "x_0": seq, "y": y}


# --------------------------------------------------------------------------- #
# membership
# --------------------------------------------------------------------------- #


def training_representatives(rows, tier: str) -> list[dict]:
    """Training representatives of ``tier``: ``split == train`` with
    ``train_tier`` at or stricter than ``tier`` (the tier sets nest)."""
    if tier not in TIER_ORDER:
        raise ValueError(f"tier must be one of {TIER_ORDER}: {tier!r}")
    lo = TIER_ORDER.index(tier)
    reps = []
    for r in rows:
        if r["split"] != TRAIN_SPLIT:
            continue
        if r.get("train_tier") not in TIER_ORDER:
            raise ValueError(
                f"{r['id']}: train row with train_tier {r.get('train_tier')!r}"
            )
        if TIER_ORDER.index(r["train_tier"]) >= lo:
            reps.append(r)
    return reps


def cluster_key(row: dict) -> str:
    """The cluster a training representative is drawn by (``cluster_repr``)."""
    return row.get("cluster_repr") or row["id"]


def draw_valid_clusters(reps, fraction: float, seed: int) -> set[str]:
    """Seeded draw of ``round(fraction * n_clusters)`` whole clusters."""
    clusters = sorted({cluster_key(r) for r in reps}, key=natural_key)
    k = round(fraction * len(clusters))
    return set(random.Random(seed).sample(clusters, k))


def _exact_shared(a, b) -> int:
    return len({r["sequence"] for r in a} & {r["sequence"] for r in b})


def build_views(
    rows,
    tiers=DEFAULT_TIERS,
    valid_fraction: float = DEFAULT_VALID_FRACTION,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, list[dict]]]:
    """``{folder: {split: [rows]}}`` for every tier folder plus ``chezod117``.

    Rows within a file are in natural ID order, so the output does not depend
    on the Parquet's row order. Raises ValueError on duplicate IDs or when a
    training sequence also occurs, exactly, in a test file.
    """
    if not 0.0 <= valid_fraction < 1.0:
        raise ValueError(f"valid fraction must be in [0, 1): {valid_fraction}")
    ordered = sorted(rows, key=lambda r: natural_key(r["id"]))
    dup = [i for i, n in Counter(r["id"] for r in ordered).items() if n > 1]
    if dup:
        raise ValueError(f"{len(dup)} duplicate ids, e.g. {dup[:5]}")
    test = [r for r in ordered if r["split"] == TEST_SPLIT]
    chezod = [r for r in ordered if r["split"] == CHEZOD_SPLIT]

    views: dict[str, dict[str, list[dict]]] = {}
    for tier in tiers:
        reps = training_representatives(ordered, tier)
        valid = draw_valid_clusters(reps, valid_fraction, seed)
        views[tier_folder(tier)] = {
            "train": [r for r in reps if cluster_key(r) not in valid],
            "valid": [r for r in reps if cluster_key(r) in valid],
            "test": test,
        }
        for name, held in (("test_trizod", test), ("test_chezod117", chezod)):
            n = _exact_shared(reps, held)
            if n:
                raise ValueError(
                    f"{tier}: {n} training sequence(s) also in {name} (exact "
                    "sequence match) -- the release leakage gate should prevent this"
                )
    views[CHEZOD_FOLDER] = {"test": chezod}
    return views


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_release(path, columns=READ_COLUMNS) -> tuple[list[dict], dict[str, str]]:
    """Rows (as dicts) and key-value metadata of a release Parquet."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "pyarrow is required to read the release Parquet but is not a project "
            "dependency. Re-run with:  uv run --with pyarrow trizod dataset ..."
        ) from exc
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    missing = sorted((set(columns) & _REQUIRED) - names)
    if missing:
        raise SystemExit(f"{path}: missing column(s) {missing}")
    cols = [c for c in columns if c in names]
    rows = pq.read_table(path, columns=cols).to_pylist()
    meta = {k.decode(): v.decode() for k, v in (pf.schema_arrow.metadata or {}).items()}
    return rows, meta


def write_views(views, out) -> dict[str, dict[str, dict]]:
    """Write every view file; returns ``{folder: {split: stats}}``."""
    out = Path(out)
    cache: dict[str, dict] = {}
    stats: dict[str, dict[str, dict]] = {}
    for folder, splits in views.items():
        d = out / folder
        d.mkdir(parents=True, exist_ok=True)
        if (d / "hf").exists():
            print(
                f"  WARNING: {d / 'hf'} exists -- UdonPred loads that cached copy "
                "instead of the new JSONL; delete it before training"
            )
        stats[folder] = {}
        for split, rows in splits.items():
            jl, fa = d / f"{split}.jsonl", d / f"{split}.fasta"
            n_res = n_masked = 0
            with jl.open("w") as fj, fa.open("w") as ff:
                for r in rows:
                    rec = cache.get(r["id"])
                    if rec is None:
                        rec = cache[r["id"]] = to_record(r)
                    fj.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    ff.write(f">{rec['id']}\n{rec['x_0']}\n")
                    n_res += len(rec["y"])
                    n_masked += sum(v == MASK_VALUE for v in rec["y"])
            st = {
                "n_records": len(rows),
                "n_residues": n_res,
                "n_masked": n_masked,
                "n_labelled": n_res - n_masked,
                "sha256": _sha256(jl),
                "bytes": jl.stat().st_size,
                "fasta_sha256": _sha256(fa),
            }
            if split in ("train", "valid"):
                st["n_clusters"] = len({cluster_key(r) for r in rows})
                st["train_tier_counts"] = dict(
                    sorted(Counter(r["train_tier"] for r in rows).items())
                )
            stats[folder][split] = st
    return stats


def verify_views(out, rows, views) -> list[str]:
    """Check F29: the written files agree with the Parquet rows, by ``id``.

    Membership is re-derived from ``rows`` (not from ``views``) for every
    folder; every value is compared position by position: ``y == 999`` exactly
    where the Parquet masks the residue, ``float32(y) == gscores`` elsewhere.
    Returns a list of problems (empty when the view is faithful).
    """
    out = Path(out)
    by_id = {r["id"]: r for r in rows}
    retired = {r["id"] for r in rows if r["split"] == RETIRED_SPLIT}
    want_test = {r["id"] for r in rows if r["split"] == TEST_SPLIT}
    problems: list[str] = []
    for folder, splits in views.items():
        seen: dict[str, set[str]] = {}
        for split in splits:
            rel = f"{folder}/{split}.jsonl"
            path = out / rel
            if not path.is_file():
                problems.append(f"{rel}: missing")
                continue
            ids: list[str] = []
            for n, line in enumerate(path.open(), 1):
                rec = json.loads(line)
                if tuple(rec) != RECORD_FIELDS:
                    problems.append(f"{rel}:{n}: keys {tuple(rec)}")
                    continue
                rid = rec["id"]
                ids.append(rid)
                src = by_id.get(rid)
                if src is None:
                    problems.append(f"{rel}: {rid} not in the Parquet")
                    continue
                if rec["x_0"] != src["sequence"]:
                    problems.append(f"{rel}: {rid}: sequence differs")
                y, g, m = rec["y"], src["gscores"], src["mask"]
                if not len(y) == len(g) == len(m) == len(src["sequence"]):
                    problems.append(f"{rel}: {rid}: length mismatch")
                    continue
                for i, (yi, gi, mi) in enumerate(zip(y, g, m)):
                    masked = (not mi) or gi is None or math.isnan(gi)
                    if masked != (yi == MASK_VALUE) or (
                        not masked and np.float32(yi) != np.float32(gi)
                    ):
                        problems.append(
                            f"{rel}: {rid}: residue {i}: y={yi} G={gi} mask={mi}"
                        )
            if len(ids) != len(set(ids)):
                problems.append(f"{rel}: duplicate ids")
            if retired & set(ids):
                problems.append(f"{rel}: {len(retired & set(ids))} test_retired ids")
            seen[split] = set(ids)
        # membership re-derived from the Parquet columns
        if folder == CHEZOD_FOLDER:
            want = {"test": {r["id"] for r in rows if r["split"] == CHEZOD_SPLIT}}
        else:
            tier = folder.removeprefix("trizod_")
            reps = {r["id"] for r in training_representatives(rows, tier)}
            got = seen.get("train", set()) | seen.get("valid", set())
            if got != reps or seen.get("train", set()) & seen.get("valid", set()):
                problems.append(f"{folder}: train+valid != {tier} representatives")
            want = {"test": want_test}
        if seen.get("test") != want["test"]:
            problems.append(f"{folder}: test membership differs from the Parquet")
    return problems


def code_commit() -> dict | None:
    """Commit of the code repository that ran the export (None outside git)."""
    root = paths.repo_root()
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": head, "dirty": bool(dirty)}


# --------------------------------------------------------------------------- #
# summary + README
# --------------------------------------------------------------------------- #

UDONPRED_SCHEMA_SOURCES = {
    "repository": "https://github.com/DavidWagemann/UdonPred (commit 80c5f0a)",
    "layout": "model/data.py:292-296 (train.jsonl, test.jsonl, valid.jsonl per dataset path)",
    "x_": "model/data.py:159 (x_columns = keys starting with 'x_')",
    "id_x_0_y": "eval/eval.ipynb:98 (x['id'], x['y']) and :507-513 (x['id'], x['x_0'], x['y'])",
    "mask_value": "model/losses.py:16-34 (flatten_and_mask(..., mask_value=999)); "
    "model/data.py:523 (DataCollator padding=999)",
    "dataset_name": "model/data.py:301-302 (dataset name = basename of the path; "
    "must equal its key in config/data.yaml)",
    "training_data": "figshare 10.6084/m9.figshare.31444642 ({train,valid,test}.{jsonl,fasta})",
}


def _source_metadata(meta: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in meta.items() if _META_KEYS.search(k)}


def _leakage_retired(views, rows) -> dict[str, int]:
    retired = [r for r in rows if r["split"] == RETIRED_SPLIT]
    return {
        f: _exact_shared(s["train"] + s["valid"], retired)
        for f, s in views.items()
        if f != CHEZOD_FOLDER
    }


def build_summary(parquet, rows, meta, views, stats, tiers, fraction, seed, problems):
    files = {}
    for folder, splits in stats.items():
        for split, st in splits.items():
            fasta = {"n_records": st["n_records"], "sha256": st["fasta_sha256"]}
            files[f"{folder}/{split}.jsonl"] = {
                k: v for k, v in st.items() if k != "fasta_sha256"
            }
            files[f"{folder}/{split}.fasta"] = fasta
    split_counts = dict(sorted(Counter(r["split"] for r in rows).items()))
    return {
        "view": "UdonPred v2 JSONL",
        "schema": {
            "fields": list(RECORD_FIELDS),
            "id": "Parquet id (BMRB entry_entity_assembly_shift-table)",
            "x_0": "Parquet sequence (one-letter, as scored)",
            "y": "Parquet gscores (TriZOD G, continuous, 0-1) per residue; "
            f"{MASK_VALUE} where mask is False or G is NaN",
            "mask_value": MASK_VALUE,
            "udonpred_sources": UDONPRED_SCHEMA_SOURCES,
        },
        "label_source": LABEL_SOURCE_RULE,
        "source": {
            "path": str(Path(parquet).resolve()),
            "sha256": _sha256(Path(parquet)),
            "num_rows": len(rows),
            "metadata": _source_metadata(meta),
        },
        "code_commit": code_commit(),
        "tiers": list(tiers),
        "seed": seed,
        "valid_fraction": fraction,
        "valid_unit": "cluster_repr; round(fraction * n_clusters) clusters per tier, "
        "random.Random(seed).sample over naturally sorted cluster ids",
        "test_definition": f"split == {TEST_SPLIT} only; {RETIRED_SPLIT} is never exported",
        "split_counts": split_counts,
        "not_exported": {
            s: split_counts.get(s, 0) for s in ("redundant", RETIRED_SPLIT, "excluded")
        },
        "train_equal_to_retired_sequences": _leakage_retired(views, rows),
        "train_reps_head_own_cluster": all(
            cluster_key(r) == r["id"] for r in rows if r["split"] == TRAIN_SPLIT
        ),
        "files": files,
        "valid_ids": {
            f: [r["id"] for r in s["valid"]] for f, s in views.items() if "valid" in s
        },
        "f29_problems": problems[:50],
        "f29_n_problems": len(problems),
    }


def render_readme(summary: dict) -> str:
    n = {rel: st["n_records"] for rel, st in summary["files"].items()}
    n_test = n[f"{tier_folder(summary['tiers'][0])}/test.jsonl"]
    n_chezod = n[f"{CHEZOD_FOLDER}/test.jsonl"]
    pct = round(summary["valid_fraction"] * 100)
    own = (
        "Each representative heads its own 50 %/80 % mmseqs cluster"
        if summary["train_reps_head_own_cluster"]
        else "Representatives sharing a `cluster_repr` stay on one side"
    )

    def scope(tier):
        if tier == TIER_ORDER[-1]:
            return f"{tier} training representatives"
        return f"training representatives at {tier} or stricter"

    rows = []
    for tier in summary["tiers"]:
        d = tier_folder(tier)
        rows.append(
            f"| `{d}/` | {n[d + '/train.jsonl']} | {n[d + '/valid.jsonl']} "
            f"| {n[d + '/test.jsonl']} "
            f"| {scope(tier)}; TriZOD test set |"
        )
    lines = [
        "# TriZOD: UdonPred v2 training view",
        "",
        "Generated by `trizod dataset export-jsonl` from the release Parquet "
        f"`{Path(summary['source']['path']).name}` "
        f"(SHA-256 `{summary['source']['sha256']}`). Every file here is derived "
        "from that one file; `export_summary.json` records counts, checksums, "
        "the seed and the validation IDs.",
        "",
        "## Layout",
        "",
        "| Folder | train | valid | test | Content |",
        "|---|---:|---:|---:|---|",
        *rows,
        f"| `{CHEZOD_FOLDER}/` | | | {n_chezod} "
        "| CheZOD117 chains, labelled with TriZOD G (not CheZOD Z) |",
        "",
        "Each folder also holds `{split}.fasta` with the same records "
        "(`>id`, sequence) for `predict.py`.",
        "",
        "## Record schema",
        "",
        "One JSON object per line, exactly the keys of UdonPred v2's own data:",
        "",
        "```json",
        '{"id": "10028_1_1_1", "x_0": "GSSGSSG...", "y": [999.0, 0.8123, 0.7, ...]}',
        "```",
        "",
        "- `id`: the Parquet `id` (BMRB entry, entity, assembly, shift table).",
        "- `x_0`: the sequence. UdonPred embeds every key starting with `x_`.",
        "- `y`: per-residue continuous G (Parquet `gscores`), `999.0` wherever "
        "the Parquet `mask` is false (NaN in the Parquet). UdonPred drops 999 "
        "from the loss (`flatten_and_mask`). Values are the shortest decimals that "
        "round-trip the Parquet's float32, so `float32(y) == gscores`.",
        "",
        "## Splits",
        "",
        "- **train/valid**: the tier's training representatives (`split == "
        '"train"` and `train_tier` at or stricter than the tier). Redundant '
        "cluster members are not exported. Tolerant is the default UdonPred "
        "trained on; moderate and strict are stricter, smaller subsets.",
        f"- **valid**: {pct} % of the tier's clusters (`cluster_repr`), drawn "
        f"with seed {summary['seed']}; whole clusters go to valid, so no cluster "
        f"straddles train and valid. {own}; homologues below 50 % identity "
        "can still fall on both sides.",
        f'- **test**: the final TriZOD test set, `split == "test_trizod"` '
        f"(n = {n_test}), identical in every tier folder. Retired test chains "
        "(`test_retired`: pinned test chains that no longer meet strict) are "
        "neither test nor training and appear in no file.",
        "",
        "## Using it with UdonPred v2",
        "",
        "Copy a tier folder to `data/<name>/` and give `config/data.yaml` a block "
        "whose key equals `<name>`: UdonPred names a dataset after the basename of "
        "its path and looks its losses up under that key. For example, copy "
        "`trizod_tolerant/` to `data/trizod/` and keep the shipped `trizod:` block. "
        "Delete `data/<name>/hf/` before retraining on a new export: UdonPred "
        "caches the embedded dataset there and reuses it.",
        "",
        "## Caveats",
        "",
        "- Labels are read from the Parquet row, i.e. from the record of the tier "
        "named in `label_tier_source`, whichever tier folder they appear in.",
        "- The per-residue shift count `k` is not exported. The release recommends "
        "k >= 10 for reproducible labels; read `k` from the Parquet by `id` to "
        "weight or filter.",
        "- Test sets have changed between releases; see the test-set crosswalk "
        "before comparing a model trained here with one trained on an earlier "
        "release.",
        "",
    ]
    return "\n".join(lines)


def export(
    parquet,
    out,
    tiers=DEFAULT_TIERS,
    valid_fraction: float = DEFAULT_VALID_FRACTION,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Read the Parquet, write every view, verify it (F29) and summarise."""
    out = Path(out)
    rows, meta = read_release(parquet)
    views = build_views(rows, tiers=tiers, valid_fraction=valid_fraction, seed=seed)
    stats = write_views(views, out)
    problems = verify_views(out, rows, views)
    summary = build_summary(
        parquet, rows, meta, views, stats, tiers, valid_fraction, seed, problems
    )
    (out / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "README.md").write_text(render_readme(summary))
    if problems:
        raise RuntimeError(
            f"F29 failed: {len(problems)} disagreement(s) between the view and "
            f"the Parquet, e.g. {problems[:3]} (see {out / 'export_summary.json'})"
        )
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, required=True, help="release Parquet")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: <parquet dir>/udonpred)",
    )
    ap.add_argument(
        "--tier",
        action="append",
        choices=TIER_ORDER,
        help=f"training tier to export, repeatable (default: {', '.join(DEFAULT_TIERS)})",
    )
    ap.add_argument(
        "--valid-fraction",
        type=float,
        default=DEFAULT_VALID_FRACTION,
        help="fraction of each tier's clusters drawn into valid (default: %(default)s)",
    )
    ap.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="(default: %(default)s)"
    )
    args = ap.parse_args(argv)

    tiers = tuple(dict.fromkeys(args.tier)) if args.tier else DEFAULT_TIERS
    out = args.out or args.parquet.parent / "udonpred"
    summary = export(args.parquet, out, tiers, args.valid_fraction, args.seed)

    print(f"Wrote UdonPred v2 view -> {out}")
    for rel, st in summary["files"].items():
        if rel.endswith(".jsonl"):
            print(
                f"  {rel:32s} {st['n_records']:6d} records "
                f"{st['n_residues']:8d} residues {st['n_masked']:7d} masked"
            )
    print(f"  not exported: {summary['not_exported']}")
    print(f"  seed {summary['seed']}, valid fraction {summary['valid_fraction']}")
    print("  F29: view == Parquet (every value, by id)")


if __name__ == "__main__":
    main()
