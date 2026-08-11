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

The companion side-chain table (``--sidechain-out``) is a second, flat long
Parquet holding the ~3.5 M side-chain chemical shifts the scoring path discards
(``bmrb.get_sidechain_shifts``). It joins 1:1 on ``id`` with the main table and
carries values AS DEPOSITED — no re-referencing, no offset correction.

Every column is declared once, in ``COLUMNS`` below, together with the record it
is read from. ``tests/test_parquet_columns.py`` asserts that declaration against
the two upstream producers (``trizod.trizod.output_dataset`` and
``trizod.dataset.composition.detect_bound``), so a column added upstream fails a
test here instead of silently never reaching the deposit — which is exactly how
the C1b/C3/C4 annotations came to stop one step short of the Parquet.

Usage:
    uv run --with pyarrow python scripts/build_parquet_dataset.py \
        --in-bundle     <path to trizod-dataset-2026-06/> \
        --out           <path to trizod_dataset.parquet> \
        [--composition-csv <path to _composition_cache.csv>] \
        [--sidechain-out <path to trizod_sidechain_shifts.parquet>]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from trizod import paths
from trizod.dataset.paths import resolve_paths as resolve_dataset_paths
from trizod.sidechain import iter_sidechain_frames, write_sidechain_parquet

# Tiers ordered loosest -> strictest. "strictest membership" = last match.
TIER_ORDER = ["unfiltered", "tolerant", "moderate", "strict"]

BACKBONE = ["C", "CA", "CB", "H", "HA", "HB", "N"]
OFF_FIELDS = [f"off_{a}" for a in BACKBONE]
LACS_FIELDS = [f"lacs_off_{a}" for a in BACKBONE]

# pyarrow is deliberately NOT a project dependency (nothing in the pipeline or
# the dataset chain needs it), so it is imported lazily — the column contract
# below stays importable, and testable, without it.
COMPOSITION_CACHE_NAME = "_composition_cache.csv"


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


def read_fasta_attrs(path: Path) -> dict[str, dict[str, str]]:
    """Parse the ``key=value`` tokens a FASTA header carries after the ID.

    ``trizod.dataset.testset`` writes ``>ID label_tier=... pinned_id=...``; a
    bundle staged before that existed simply has no tokens, which resolves to an
    empty attribute dict (and therefore a null column), never a made-up value.
    """
    attrs: dict[str, dict[str, str]] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            tokens = line[1:].split()
            if not tokens:
                continue
            attrs[tokens[0]] = dict(
                tok.split("=", 1) for tok in tokens[1:] if "=" in tok
            )
    return attrs


def read_composition_cache(path: Path) -> dict[str, dict[str, str]]:
    """``entryID -> raw composition row`` from ``_composition_cache.csv``.

    Written by ``trizod dataset build`` from
    :func:`trizod.dataset.composition.detect_bound`, keyed on the BMRB entry ID
    (not the chain ID), and covering every parsed entry — including the ones
    that never reach a tier.
    """
    rows: dict[str, dict[str, str]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            entry_id = clean_str(row.get("entryID"))
            if entry_id is not None:
                rows[entry_id] = row
    return rows


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


#: Spellings of "unset" that reach the composition cache as text: NMR-STAR nulls
#: and pandas nulls stringified by ``to_csv``. Never coerced to a value.
_NULL_TEXT = frozenset({"", "nan", "none", "<na>", "null", "."})


def as_bool(value) -> bool | None:
    """Truthy text (CSV) or a real bool (JSON) -> bool; unset -> None.

    Deliberately returns None rather than False for anything unrecognised: a
    silent False here would read downstream as "classified, and negative".
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in _NULL_TEXT:
        return None
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    return None


def as_str_list(value) -> list[str] | None:
    """``"CA;ZN"`` -> ``["CA", "ZN"]``; ``""`` -> ``[]``; unset -> None.

    ``detect_bound`` emits these three fields as ``";".join(sorted(...))``, so
    splitting on ``;`` recovers exactly what it encoded — a list column beats a
    delimiter-joined string for consumers, and the round trip is the classifier's
    own convention rather than a new one invented here. An entry with no ligands
    is an empty list (a measured absence); an entry with no composition record at
    all is null (unknown).
    """
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return []
    if s.lower() in _NULL_TEXT:
        return None
    return [part.strip() for part in s.split(";") if part.strip()]


# --------------------------------------------------------------------------- #
# column contract
# --------------------------------------------------------------------------- #

#: Arrow type per token. Tokens (not ``pa`` objects) so ``COLUMNS`` can be
#: imported without pyarrow installed.
ARROW_TYPES = {
    "string": lambda pa: pa.string(),
    "bool": lambda pa: pa.bool_(),
    "int32": lambda pa: pa.int32(),
    "float32": lambda pa: pa.float32(),
    "float64": lambda pa: pa.float64(),
    "list<bool>": lambda pa: pa.list_(pa.bool_()),
    "list<int32>": lambda pa: pa.list_(pa.int32()),
    "list<float32>": lambda pa: pa.list_(pa.float32()),
    "list<string>": lambda pa: pa.list_(pa.string()),
}

#: Where a column's value comes from. ``scores`` = one ``scores.json`` record,
#: ``composition`` = the ``_composition_cache.csv`` row for the chain's entry ID,
#: ``testset`` = the TriZOD test-set FASTA header, ``derived`` = computed from
#: the bundle's FASTAs / cluster TSVs / other columns.
ORIGINS = ("scores", "composition", "testset", "derived")


@dataclass(frozen=True)
class Column:
    """One Parquet column: its name, its Arrow type, and where its value is read.

    ``convert`` maps the raw value of ``source`` in the ``origin`` record;
    ``derive`` instead computes the value from the per-row context (and takes
    precedence). A column may set both ``source`` and ``derive`` — ``source``
    then only records which upstream key the column consumes, which is what the
    coverage tests check.
    """

    name: str
    arrow: str
    origin: str = "derived"
    source: str | None = None
    convert: Callable | None = None
    derive: Callable | None = None

    def __post_init__(self):
        assert self.arrow in ARROW_TYPES, f"unknown arrow type {self.arrow!r}"
        assert self.origin in ORIGINS, f"unknown origin {self.origin!r}"
        assert self.derive is not None or self.convert is not None, self.name
        assert self.derive is not None or self.source is not None, self.name


def _scores(name, arrow, source, convert):
    return Column(name, arrow, origin="scores", source=source, convert=convert)


def _composition(name, arrow, convert):
    # source == name: the cache column is the classifier's own key
    return Column(name, arrow, origin="composition", source=name, convert=convert)


#: THE column contract. Order is the Parquet column order; the first 42 entries
#: are the published v0.3.0 layout and must not be reordered, renamed or
#: retyped — new columns are appended.
COLUMNS: list[Column] = [
    _scores("id", "string", "ID", clean_str),
    _scores("entry_id", "string", "entryID", clean_str),
    _scores("entity_id", "string", "entityID", clean_str),
    _scores("entity_assem_id", "string", "entity_assemID", clean_str),
    _scores("st_id", "string", "stID", clean_str),
    _scores("entity_name", "string", "entity_name", clean_str),
    Column("sequence", "string", "scores", "seq", derive=lambda c: c["seq"]),
    Column("length", "int32", derive=lambda c: len(c["seq"])),
    Column("gscores", "list<float32>", "scores", "gscores", derive=lambda c: c["g"]),
    Column("zscores", "list<float32>", "scores", "zscores", derive=lambda c: c["z"]),
    Column("k", "list<int32>", "scores", "k", derive=lambda c: c["k"]),
    Column("mask", "list<bool>", derive=lambda c: c["mask"]),
    Column("n_scored", "int32", derive=lambda c: sum(c["mask"])),
    Column("split", "string", derive=lambda c: c["split"]),
    Column("train_tier", "string", derive=lambda c: c["train_tier"]),
    Column("pool_tier", "string", derive=lambda c: c["pool_tier"]),
    Column("cluster_repr", "string", derive=lambda c: c["cluster_repr"]),
    Column("quality", "float64", derive=lambda c: c["quality"]),
    _scores("exp_method", "string", "exp_method", clean_str),
    _scores("exp_method_subtype", "string", "exp_method_subtype", clean_str),
    _scores("ph", "float32", "pH", as_float),
    _scores("temperature", "float32", "temperature", as_float),
    _scores("ionic_strength", "float32", "ionic_strength", as_float),
    _scores("total_bbshifts", "int32", "total_bbshifts", as_int),
    _scores("bbshift_positions_post", "int32", "bbshift_positions_post", as_int),
    _scores("bbshift_types_post", "int32", "bbshift_types_post", as_int),
    _scores("citation_title", "string", "citation_title", clean_str),
    _scores("citation_doi", "string", "citation_DOI", clean_str),
    *[_scores(f, "float32", f, as_float) for f in OFF_FIELDS],
    *[_scores(f, "float32", f, as_float) for f in LACS_FIELDS],
    # --- appended in v0.4.0 --------------------------------------------------
    # Sample-state annotations (plan §3.1/§3.2). Emitted at every tier and
    # filtered by none of them except the per-tier physical_state deny list.
    _scores("physical_state", "string", "physical_state", clean_str),
    _scores("denaturant_evidence", "bool", "denaturant_evidence", as_bool),
    _scores("sample_state_evidence", "string", "sample_state_evidence", clean_str),
    _scores("membrane_mimetic", "string", "membrane_mimetic", clean_str),
    # Composition labels (plan §3.3/§3.4), joined on the BMRB entry ID, so every
    # chain of an entry carries its entry's composition. ``n_entities`` keeps its
    # published meaning: distinct ``_Entity`` records, NOT the copy count — a
    # homodimer is one Entity referenced twice, which is what ``n_copies`` and
    # ``n_entity_assembly_rows`` describe.
    _composition("n_entities", "int32", as_int),
    _composition("has_non_polymer", "bool", as_bool),
    _composition("has_nucleic", "bool", as_bool),
    _composition("has_metal", "bool", as_bool),
    _composition("has_metal_ion", "bool", as_bool),
    _composition("has_metal_cofactor", "bool", as_bool),
    _composition("has_water", "bool", as_bool),
    _composition("has_other_ligand", "bool", as_bool),
    _composition("metal_comp_ids", "list<string>", as_str_list),
    _composition("ligand_comp_ids", "list<string>", as_str_list),
    _composition("ligand_names", "list<string>", as_str_list),
    _composition("multi_protein_assembly", "bool", as_bool),
    _composition("n_entity_assembly_rows", "int32", as_int),
    # NA, never 1, when the classifier errored on the entry: build.py's except
    # branch writes {"is_bound": True, "error": ...} and nothing else.
    _composition("n_copies", "int32", as_int),
    _composition("has_conformational_isomer", "bool", as_bool),
    _composition("is_bound", "bool", as_bool),
    # Strictest tier the test chain's sequence still satisfies. Only test_trizod
    # rows have one; null everywhere else.
    Column(
        "label_tier", "string", origin="testset", source="label_tier", convert=clean_str
    ),
]

#: ``scores.json`` keys carried into the Parquet, derived from ``COLUMNS`` so
#: the two can never disagree.
SCORES_JSON_SOURCES = frozenset(c.source for c in COLUMNS if c.origin == "scores")
#: ``detect_bound()`` keys carried into the Parquet.
COMPOSITION_SOURCES = frozenset(c.source for c in COLUMNS if c.origin == "composition")
#: ``scores.json`` keys deliberately NOT carried: ``--include-shifts`` adds one
#: column per backbone atom holding re-referenced shifts, while the deposit ships
#: raw shifts in the side-chain companion table. The release is generated without
#: that flag, so these keys are normally absent from scores.json entirely.
SCORES_JSON_NOT_CARRIED = frozenset(
    BACKBONE + ["HA2", "HA3", "HB1", "HB2", "HB3"]
)  # fmt: skip


def arrow_schema(pa):
    """The Parquet schema, built from ``COLUMNS`` (needs an imported pyarrow)."""
    return pa.schema([(c.name, ARROW_TYPES[c.arrow](pa)) for c in COLUMNS])


def _import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "pyarrow is required to write Parquet but is not a project "
            "dependency. Re-run with:  uv run --with pyarrow python ..."
        ) from exc
    return pa, pq


def collect_columns(in_bundle: Path, composition_csv: Path | None = None) -> dict:
    """Assemble every ``COLUMNS`` entry into a name -> list-of-values dict.

    Pure Python, no pyarrow: the join/null behaviour is the part worth testing,
    and pyarrow is an optional extra.
    """
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
    trizod_test_attrs = read_fasta_attrs(in_bundle / "test" / "TriZOD_test_set.fasta")
    test_trizod = set(trizod_test_attrs)
    if not any(a.get("label_tier") for a in trizod_test_attrs.values()):
        print(
            "  WARNING: no label_tier= on the TriZOD test FASTA headers "
            "(bundle staged before trizod/dataset/testset.py emitted it) — "
            "the label_tier column will be null"
        )

    composition = {}
    if composition_csv is not None:
        composition = read_composition_cache(composition_csv)
        print(f"loaded composition for {len(composition)} BMRB entries")
        missing_cols = COMPOSITION_SOURCES - set(next(iter(composition.values()), {}))
        if missing_cols:
            print(
                f"  WARNING: {composition_csv} predates "
                f"{sorted(missing_cols)} — those columns will be null. "
                "Re-run `trizod dataset build` to refresh it."
            )
    else:
        print("  WARNING: --skip-composition — every composition column will be null")

    cols: dict[str, list] = {c.name: [] for c in COLUMNS}
    n_comp_hits = 0

    for rec in scores:
        rid = rec["ID"]
        seq = rec["seq"]
        raw_g = rec.get("gscores") or []
        raw_z = rec.get("zscores") or []
        raw_k = rec.get("k") or []
        mask = [g is not None for g in raw_g]

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

        comp_row = composition.get(clean_str(rec.get("entryID")))
        n_comp_hits += comp_row is not None
        records = {
            "scores": rec,
            "composition": comp_row,
            "testset": trizod_test_attrs.get(rid),
        }
        ctx = {
            "seq": seq,
            "g": [float(g) if g is not None else float("nan") for g in raw_g],
            "z": [float(z) if z is not None else float("nan") for z in raw_z],
            "k": [int(x) if x is not None else 0 for x in raw_k],
            "mask": mask,
            "split": split,
            "train_tier": train_tier,
            "pool_tier": pool_tier,
            "cluster_repr": best_repr_map.get(rid),
            "quality": quality_map.get(rid),
        }
        for col in COLUMNS:
            if col.derive is not None:
                cols[col.name].append(col.derive(ctx))
                continue
            # An absent upstream record yields null, never a default: "unknown"
            # and "measured negative" must stay distinguishable downstream.
            record = records[col.origin]
            cols[col.name].append(
                None if record is None else col.convert(record.get(col.source))
            )

    print(
        f"composition joined for {n_comp_hits}/{len(scores)} chains; "
        f"label_tier for {sum(x is not None for x in cols['label_tier'])}"
    )
    return cols


def build(in_bundle: Path, out_path: Path, composition_csv: Path | None = None) -> None:
    cols = collect_columns(in_bundle, composition_csv)

    pa, pq = _import_pyarrow()
    schema = arrow_schema(pa)
    table = pa.table({name: cols[name] for name in schema.names}, schema=schema)

    metadata = {
        b"dataset": b"TriZOD",
        b"description": b"Per-residue protein-disorder labels from re-referenced BMRB NMR backbone chemical shifts. One row per scored chain; per-residue arrays aligned 1:1 to sequence.",
        b"tier_order_loose_to_strict": b"unfiltered,tolerant,moderate,strict",
        b"split_values": b"train,redundant,test_chezod117,test_trizod,excluded",
        b"mask_semantics": b"mask[i]=true where residue i is scored; gscores/zscores are NaN where mask is false",
        b"null_semantics": b"null means unknown/not recorded, never a default: an absent composition record leaves every composition column null, and label_tier is null for every non-test_trizod row",
        b"composition_join": b"composition columns (n_entities .. is_bound) are per BMRB ENTRY, joined on entry_id, from trizod.dataset.composition.detect_bound; n_entities counts distinct _Entity records, so a homodimer is 1 -- see n_copies / n_entity_assembly_rows for stoichiometry",
        b"composition_list_columns": b"metal_comp_ids, ligand_comp_ids and ligand_names are list<string>; an empty list is a measured absence, null is an unclassified entry",
        b"n_copies_semantics": b"conservative lower bound on copies of one entity in an assembly (magnetic-equivalence groups, then physical state, then conformer naming); null where the classifier errored",
        b"membrane_mimetic_semantics": b"';'-joined detergent/lipid tokens matched in the sample components, null when none; annotation only, never filtered",
        b"label_tier_semantics": b"test_trizod rows only: strictest tier whose pool still contains the pinned test sequence",
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


def build_sidechain(in_bundle: Path, out_path: Path, pkl_dir: Path) -> None:
    """Write the companion side-chain shift table for the bundle's chains.

    The ids come from the same unfiltered ``scores.json`` the main table is
    built from (unfiltered is a superset of every tier), so the companion joins
    1:1 on ``id`` with no orphans on either side.
    """
    ids = [
        rec["ID"]
        for rec in read_jsonl(in_bundle / "scores" / "unfiltered" / "scores.json")
    ]
    print(f"\nbuilding side-chain companion for {len(ids)} chains from {pkl_dir} ...")
    if not pkl_dir.is_dir():
        raise SystemExit(f"BMRB pickle cache not found: {pkl_dir}")
    # per-chain rejections (AA mismatch, unparsable Seq_ID) are expected in bulk
    # and would emit thousands of lines; the summary below counts them instead
    logging.getLogger("trizod.bmrb").setLevel(logging.CRITICAL)
    frames = iter_sidechain_frames(ids, pkl_dir)
    path, n_rows = write_sidechain_parquet(frames, out_path)

    _, pq = _import_pyarrow()
    table = pq.read_table(path, columns=["id"])
    n_chains = len(set(table.column("id").to_pylist()))
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")
    print(f"rows: {n_rows}   chains with side-chain shifts: {n_chains} / {len(ids)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-bundle", type=Path, required=True, help="path to trizod-dataset-2026-06/"
    )
    ap.add_argument("--out", type=Path, default=None, help="output .parquet path")
    ap.add_argument(
        "--composition-csv",
        type=Path,
        default=None,
        help=(
            "molecular-composition cache written by `trizod dataset build` "
            "(default: <work-dir>/final_dataset/" + COMPOSITION_CACHE_NAME + ")"
        ),
    )
    ap.add_argument(
        "--skip-composition",
        action="store_true",
        help="write the composition columns as all-null instead of joining them",
    )
    ap.add_argument(
        "--sidechain-out",
        type=Path,
        default=None,
        help="also write the companion side-chain shift table here",
    )
    ap.add_argument(
        "--pkl-dir",
        type=Path,
        default=paths.PKL_DIR,
        help="BMRB pickle cache read by --sidechain-out (default: tmp/bmrb_entries)",
    )
    args = ap.parse_args()
    if not args.out and not args.sidechain_out:
        ap.error("nothing to do: pass --out and/or --sidechain-out")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.out:
        # The composition cache lives in the build work dir, not in the bundle
        # (package_release.py stages only the published artefacts). Missing it is
        # an error rather than a silent all-null block: these annotations are the
        # deliverable, and a Parquet that quietly lacks them looks identical.
        composition_csv = args.composition_csv
        if args.skip_composition:
            composition_csv = None
        elif composition_csv is None:
            composition_csv = (
                resolve_dataset_paths().final_dataset / COMPOSITION_CACHE_NAME
            )
        if composition_csv is not None and not composition_csv.is_file():
            ap.error(
                f"composition cache not found: {composition_csv}\n"
                "run `trizod dataset build` first, pass --composition-csv, or "
                "pass --skip-composition to accept null composition columns"
            )
        build(args.in_bundle, args.out, composition_csv)
    if args.sidechain_out:
        build_sidechain(args.in_bundle, args.sidechain_out, args.pkl_dir)


if __name__ == "__main__":
    main()
