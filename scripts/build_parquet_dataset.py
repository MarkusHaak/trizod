"""Build the single-file Parquet release of the TriZOD dataset.

Consolidates the multi-file staged release bundle (per-tier FASTA + per-tier
``scores.json`` JSONL + ``clusters_best.tsv`` + test FASTAs) into ONE
self-contained Parquet table, one row per scored protein chain.

Design (see docs): the per-tier ``scores.json`` files are byte-identical for a
given chain and nest perfectly (strict subset tolerant subset ... unfiltered),
and the per-tier training-representative sets and clustering pools also nest.
So a single row per chain plus a handful of ordinal/categorical columns encodes
every published view with no information loss:

  * ``split``       one of train / redundant / test_chezod117 / test_trizod / excluded
  * ``train_tier``  strictest tier at which the chain is a training representative
  * ``pool_tier``   strictest tier whose redundancy-reduction pool contains the chain

Per-residue labels are stored as equal-length list columns aligned 1:1 to the
sequence; unscored positions are NaN in the score columns and False in the
explicit boolean ``mask`` (no magic sentinel that could leak into a regression
loss).

Usage:
    uv run --with pyarrow python scripts/build_parquet_dataset.py \
        --in-bundle  <path to trizod-dataset-2026-06/> \
        --out        <path to trizod_dataset.parquet>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Tiers ordered loosest -> strictest. "strictest membership" = last match.
TIER_ORDER = ["unfiltered", "tolerant", "moderate", "strict"]

BACKBONE = ["C", "CA", "CB", "H", "HA", "HB", "N"]
OFF_FIELDS = [f"off_{a}" for a in BACKBONE]
LACS_FIELDS = [f"lacs_off_{a}" for a in BACKBONE]


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_fasta_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                ids.add(line[1:].split()[0])
    return ids


def read_cluster_map(path: Path) -> tuple[set[str], dict[str, str], dict[str, float]]:
    """Return (member set, member->best_repr, member->member_quality)."""
    members: set[str] = set()
    best: dict[str, str] = {}
    quality: dict[str, float] = {}
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            m = row["member"]
            members.add(m)
            best[m] = row["best_repr"]
            try:
                quality[m] = float(row["member_quality"])
            except (KeyError, ValueError, TypeError):
                quality[m] = None
    return members, best, quality


def strictest(id_: str, membership: dict[str, set[str]]) -> str | None:
    """Loosest->strictest scan; return the strictest tier whose set has id_."""
    found = None
    for tier in TIER_ORDER:
        if id_ in membership[tier]:
            found = tier
    return found


def as_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (ValueError, TypeError):
        return None
    return f if not math.isnan(f) else None


def as_int(value) -> int | None:
    f = as_float(value)
    return int(f) if f is not None else None


def clean_str(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def build(in_bundle: Path, out_path: Path) -> None:
    scores = read_jsonl(in_bundle / "scores" / "unfiltered" / "scores.json")
    print(f"loaded {len(scores)} scored chains (unfiltered = superset)")

    reps = {
        t: read_fasta_ids(in_bundle / "train" / t / f"train_{t}_best.fasta")
        for t in TIER_ORDER
    }
    pool = {}
    best_repr_map: dict[str, str] = {}
    quality_map: dict[str, float] = {}
    for t in TIER_ORDER:
        members, best, quality = read_cluster_map(
            in_bundle / "train" / t / "clusters_best.tsv"
        )
        pool[t] = members
        # unfiltered pool covers every pooled chain (pools nest); use it as canonical
        if t == "unfiltered":
            best_repr_map, quality_map = best, quality

    test_chezod = read_fasta_ids(in_bundle / "test" / "CheZOD117_test_set.fasta")
    test_trizod = read_fasta_ids(in_bundle / "test" / "TriZOD_test_set.fasta")

    cols: dict[str, list] = {
        name: []
        for name in (
            "id",
            "entry_id",
            "entity_id",
            "entity_assem_id",
            "st_id",
            "entity_name",
            "sequence",
            "length",
            "gscores",
            "zscores",
            "k",
            "mask",
            "n_scored",
            "split",
            "train_tier",
            "pool_tier",
            "cluster_repr",
            "quality",
            "exp_method",
            "exp_method_subtype",
            "ph",
            "temperature",
            "ionic_strength",
            "total_bbshifts",
            "bbshift_positions_post",
            "bbshift_types_post",
            "citation_title",
            "citation_doi",
            *OFF_FIELDS,
            *LACS_FIELDS,
        )
    }

    for rec in scores:
        rid = rec["ID"]
        seq = rec["seq"]
        raw_g = rec.get("gscores") or []
        raw_z = rec.get("zscores") or []
        raw_k = rec.get("k") or []
        mask = [g is not None for g in raw_g]
        gscores = [float(g) if g is not None else float("nan") for g in raw_g]
        zscores = [float(z) if z is not None else float("nan") for z in raw_z]
        k = [int(x) if x is not None else 0 for x in raw_k]

        train_tier = strictest(rid, reps)
        pool_tier = strictest(rid, pool)
        if rid in test_chezod:
            split = "test_chezod117"
        elif rid in test_trizod:
            split = "test_trizod"
        elif train_tier is not None:
            split = "train"
        elif pool_tier is not None:
            split = "redundant"
        else:
            split = "excluded"

        cols["id"].append(rid)
        cols["entry_id"].append(clean_str(rec.get("entryID")))
        cols["entity_id"].append(clean_str(rec.get("entityID")))
        cols["entity_assem_id"].append(clean_str(rec.get("entity_assemID")))
        cols["st_id"].append(clean_str(rec.get("stID")))
        cols["entity_name"].append(clean_str(rec.get("entity_name")))
        cols["sequence"].append(seq)
        cols["length"].append(len(seq))
        cols["gscores"].append(gscores)
        cols["zscores"].append(zscores)
        cols["k"].append(k)
        cols["mask"].append(mask)
        cols["n_scored"].append(sum(mask))
        cols["split"].append(split)
        cols["train_tier"].append(train_tier)
        cols["pool_tier"].append(pool_tier)
        cols["cluster_repr"].append(best_repr_map.get(rid))
        cols["quality"].append(quality_map.get(rid))
        cols["exp_method"].append(clean_str(rec.get("exp_method")))
        cols["exp_method_subtype"].append(clean_str(rec.get("exp_method_subtype")))
        cols["ph"].append(as_float(rec.get("pH")))
        cols["temperature"].append(as_float(rec.get("temperature")))
        cols["ionic_strength"].append(as_float(rec.get("ionic_strength")))
        cols["total_bbshifts"].append(as_int(rec.get("total_bbshifts")))
        cols["bbshift_positions_post"].append(as_int(rec.get("bbshift_positions_post")))
        cols["bbshift_types_post"].append(as_int(rec.get("bbshift_types_post")))
        cols["citation_title"].append(clean_str(rec.get("citation_title")))
        cols["citation_doi"].append(clean_str(rec.get("citation_DOI")))
        for f in OFF_FIELDS + LACS_FIELDS:
            cols[f].append(as_float(rec.get(f)))

    schema = pa.schema(
        [
            ("id", pa.string()),
            ("entry_id", pa.string()),
            ("entity_id", pa.string()),
            ("entity_assem_id", pa.string()),
            ("st_id", pa.string()),
            ("entity_name", pa.string()),
            ("sequence", pa.string()),
            ("length", pa.int32()),
            ("gscores", pa.list_(pa.float32())),
            ("zscores", pa.list_(pa.float32())),
            ("k", pa.list_(pa.int32())),
            ("mask", pa.list_(pa.bool_())),
            ("n_scored", pa.int32()),
            ("split", pa.string()),
            ("train_tier", pa.string()),
            ("pool_tier", pa.string()),
            ("cluster_repr", pa.string()),
            ("quality", pa.float64()),
            ("exp_method", pa.string()),
            ("exp_method_subtype", pa.string()),
            ("ph", pa.float32()),
            ("temperature", pa.float32()),
            ("ionic_strength", pa.float32()),
            ("total_bbshifts", pa.int32()),
            ("bbshift_positions_post", pa.int32()),
            ("bbshift_types_post", pa.int32()),
            ("citation_title", pa.string()),
            ("citation_doi", pa.string()),
            *[(f, pa.float32()) for f in OFF_FIELDS],
            *[(f, pa.float32()) for f in LACS_FIELDS],
        ]
    )

    table = pa.table({name: cols[name] for name in schema.names}, schema=schema)

    metadata = {
        b"dataset": b"TriZOD",
        b"description": b"Per-residue protein-disorder labels from re-referenced BMRB NMR backbone chemical shifts. One row per scored chain; per-residue arrays aligned 1:1 to sequence.",
        b"tier_order_loose_to_strict": b"unfiltered,tolerant,moderate,strict",
        b"split_values": b"train,redundant,test_chezod117,test_trizod,excluded",
        b"mask_semantics": b"mask[i]=true where residue i is scored; gscores/zscores are NaN where mask is false",
    }
    table = table.replace_schema_metadata(metadata)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd", compression_level=19)

    # summary
    from collections import Counter

    split_counts = Counter(cols["split"])
    tier_counts = Counter(t for t in cols["train_tier"] if t is not None)
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"rows: {table.num_rows}   columns: {table.num_columns}")
    print("split counts:", dict(split_counts))
    print("train_tier counts (strictest):", dict(tier_counts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-bundle", type=Path, required=True, help="path to trizod-dataset-2026-06/"
    )
    ap.add_argument("--out", type=Path, required=True, help="output .parquet path")
    args = ap.parse_args()
    build(args.in_bundle, args.out)


if __name__ == "__main__":
    main()
