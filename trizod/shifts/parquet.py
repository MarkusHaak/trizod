"""Write the released chemical-shift table as Parquet.

Flat long table, one row per deposited value on a canonical residue — backbone
*and* side chain — joining on ``id`` with ``trizod_dataset.parquet``. Same codec
as the deposit (zstd-19).

Every numeric field declares its unit in **field-level** metadata and the table
declares the correction formula in words, because the one mistake this file
exists to prevent is a unit mistake: ``lacs_off_<atom>_ppm`` is ppm,
``off_<atom>_sigma`` is a multiple of the per-atom POTENCI RMSD, and adding
them is worth up to 21.2 ppm of error. A consumer reading the schema is told
which is which without leaving the file.

pyarrow is deliberately **not** a project dependency (the scoring pipeline and
the dataset chain do not need it); it is imported lazily here, exactly like
``scripts/build_parquet_dataset.py`` expects to be run:

    uv run --with pyarrow python scripts/build_parquet_dataset.py ...
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trizod.shifts.correct import (
    OFFSET_SOURCES,
    SIDECHAIN_CARBON_LACS_ATOM,
    SIDECHAIN_MAX_ABS_LACS_PPM,
)
from trizod.shifts.outliers import MAD_K, MAD_TO_SIGMA, MAD_Z_DECIMALS, MadReference

#: ``0.01`` for ``MAD_Z_DECIMALS = 2``, as the metadata states it.
_MAD_Z_STEP = f"{10**-MAD_Z_DECIMALS:g}"

#: Parquet column order. ``is_backbone`` follows the value columns: it is a
#: filter flag, not a measurement. The far-outlier pair is appended after it.
SHIFT_PARQUET_COLUMNS = [
    "id",
    "seq_id",
    "comp_id",
    "atom_id",
    "atom_type",
    "val_ppm",
    "val_err_ppm",
    "ambiguity_code",
    "val_corrected_ppm",
    "offset_applied_ppm",
    "offset_source",
    "is_backbone",
    "mad_z",
    "is_far_outlier",
]

#: ``(name, arrow type factory, field metadata)``. The metadata is what makes the
#: units unambiguous at the point of use.
_FIELDS = [
    (
        "id",
        lambda pa: pa.string(),
        {
            "description": "chain key, joins with trizod_dataset.parquet.id "
            "(entryID_stID_entity_assemID_entityID)"
        },
    ),
    (
        "seq_id",
        lambda pa: pa.int32(),
        {
            "unit": "residue index",
            "description": "1-based position into trizod_dataset.parquet.sequence: "
            "sequence[seq_id-1] is the residue named by comp_id",
        },
    ),
    ("comp_id", lambda pa: pa.string(), {"description": "residue, 3-letter code"}),
    (
        "atom_id",
        lambda pa: pa.string(),
        {"description": "BMRB/IUPAC atom name as deposited"},
    ),
    (
        "atom_type",
        lambda pa: pa.string(),
        {"description": "element symbol as deposited: C, H, N (rarely P)"},
    ),
    (
        "val_ppm",
        lambda pa: pa.float64(),
        {
            "unit": "ppm",
            "description": "chemical shift EXACTLY AS DEPOSITED in BMRB. Never "
            "re-referenced, never offset-corrected. Always present.",
        },
    ),
    (
        "val_err_ppm",
        lambda pa: pa.float64(),
        {
            "unit": "ppm",
            "description": "deposited Val_err; null when absent or non-numeric. "
            "Carried as data, not applied as a filter.",
        },
    ),
    (
        "ambiguity_code",
        lambda pa: pa.int8(),
        {
            "unit": "BMRB Ambiguity_code (1-9, dimensionless)",
            "description": "as deposited; null when absent or outside 1-9. "
            "2 = stereo-unassigned pair, 3 = aromatic ring degenerate.",
        },
    ),
    (
        "val_corrected_ppm",
        lambda pa: pa.float64(),
        {
            "unit": "ppm",
            "description": "val_ppm - offset_applied_ppm; NULL wherever no "
            "trustworthy offset exists for this row. Never a copy of val_ppm.",
        },
    ),
    (
        "offset_applied_ppm",
        lambda pa: pa.float64(),
        {
            "unit": "ppm",
            "description": "the referencing offset SUBTRACTED from val_ppm, "
            "already converted to ppm; null where none was applied.",
        },
    ),
    (
        "offset_source",
        lambda pa: pa.string(),
        {"description": "provenance of offset_applied_ppm; see table metadata"},
    ),
    (
        "is_backbone",
        lambda pa: pa.bool_(),
        {
            "description": "true for the 12 atom IDs the TriZOD scoring path "
            "consumes (C, CA, CB, H, HA, HB, N, HA2, HA3, HB1, HB2, HB3); "
            "false for every other assigned atom",
        },
    ),
    (
        "mad_z",
        lambda pa: pa.float32(),
        {
            "unit": "raw MAD",
            "description": "(val_ppm - median) / MAD of this comp_id+atom_id in "
            "the frozen reference (table metadata mad_reference), rounded to "
            f"{_MAD_Z_STEP}. MAD is NOT scaled: divide by {MAD_TO_SIGMA} for a "
            "normal-sigma z. NULL where not assessed; see far_outlier_rule.",
        },
    ),
    (
        "is_far_outlier",
        lambda pa: pa.bool_(),
        {
            "description": "|mad_z| beyond the per-atom limit in table metadata "
            "far_outlier_rule; NULL where mad_z is NULL. A flag only: no row is "
            "dropped and the G-scores never read it.",
        },
    ),
]

_CORRECTION_FORMULA = (
    "val_corrected_ppm = val_ppm - offset_applied_ppm, and NULL where "
    "offset_applied_ppm is null. offset_applied_ppm is ALREADY IN PPM. For a "
    "backbone row it is the total offset TriZOD subtracted for that atom, "
    "total_off_<atom>_ppm = lacs_off_<atom>_ppm + off_<atom>_sigma * "
    "REFINED_WEIGHTS[<atom>], where off_<atom>_sigma in trizod_dataset.parquet "
    "is in units of the per-atom POTENCI RMSD and NOT in ppm -- adding the two "
    "raw columns together is a unit error worth up to 21.2 ppm. Degenerate "
    "partners (HB2/HB3, ALA HB1, GLY HA2/HA3) carry the offset of the slot the "
    "scoring path averages them into (HB, HA). For a side-chain row it is "
    "lacs_off_CA_ppm, or null."
)

_SIDECHAIN_POLICY = (
    "Which offset transfers to a side-chain nucleus was measured on this corpus "
    "as a regression slope beta = cov(group-centred side-chain deviation, "
    "offset) / var(offset); subtracting an offset lowers MSE iff beta > 0.5. "
    f"13C: beta = 0.785 [0.767, 0.806] -> lacs_off_{SIDECHAIN_CARBON_LACS_ATOM}"
    "_ppm is applied (side-chain MSE x0.79, RMSE 0.919 -> 0.818 ppm). "
    "1H: beta = 0.079 [0.057, 0.102] -> left raw (applying it inflates MSE "
    "x1.131 and is indistinguishable from applying a random other chain's "
    "offset). 15N: beta = 0.366 [0.330, 0.407], and 0.499 [0.430, 0.555] after "
    "instrumental-variable disattenuation against PANAV -> left raw (applying "
    "it inflates MSE x1.075). The POTENCI residual term is never propagated to "
    "side chains: it transfers at beta 0.05-0.09."
)

_OFFSET_SOURCE_VALUES = (
    "lacs+potenci | lacs_only | potenci_only = a non-zero offset was subtracted, "
    "labelled by the estimators that contributed a non-zero amount. "
    "none = nothing was subtracted; offset_applied_ppm is 0.0 when the "
    "estimators ran and returned zero (val_corrected_ppm == val_ppm, a measured "
    "no-op: LACS never fires below 20 backbone CA observations) and null when no "
    "offset could be determined (val_corrected_ppm null). "
    "not_transferable = the chain has an offset but it does not demonstrably "
    "transfer to this row, so no corrected value is published (both value "
    "columns null): every side-chain 1H and 15N, plus side-chain 13C on the 13 "
    f"chains with |lacs_off_CA_ppm| > {SIDECHAIN_MAX_ABS_LACS_PPM:g} ppm."
)

_FAR_OUTLIER_RULE = (
    "is_far_outlier = |mad_z| > k, mad_z = (val_ppm - median) / MAD rounded to "
    f"{_MAD_Z_STEP} (the flag is evaluated on the rounded, published value), per "
    "(comp_id, atom_id), with k by atom_id: "
    + ", ".join(
        f"{atom}: {k:g} (about {k / MAD_TO_SIGMA:.1f} normal sigma)"
        for atom, k in MAD_K.items()
    )
    + " (Sandro Kuppel's thresholds, read off the corpus histograms). Only the raw "
    f"val_ppm of {', '.join(MAD_K)} is assessed; every other atom (protons, side "
    "chains) is NOT assessed (NULL), and neither is the N of residue 1, the "
    "N-terminal free amine near 35-40 ppm, which is real physics. MAD is the raw "
    f"median absolute deviation, NOT scaled by {MAD_TO_SIGMA}. Median and MAD "
    "come from a reference table frozen in the code repository "
    "(trizod/shifts/mad_reference.csv), embedded here as mad_reference, so the "
    "flag does not drift with BMRB. Nothing is filtered: drop the flagged rows "
    "yourself if your use needs it."
)

PARQUET_METADATA = {
    b"dataset": b"TriZOD chemical shifts",
    b"description": (
        b"Every assigned chemical shift ON A CANONICAL RESIDUE of every released "
        b"chain, backbone and side chain, one row per (chain, residue, atom). "
        b"val_ppm is as deposited; val_corrected_ppm is the TriZOD-re-referenced "
        b"value where one exists. Filter on is_backbone or atom_id for a subset."
    ),
    b"not_covered": (
        b"Shifts assigned to a NON-CANONICAL residue or group are excluded, "
        b"because seq_id would not index the released sequence: the PTMs, the "
        b"non-standard residues and the terminal or lipid groups. Read those "
        b"from the BMRB entry directly. How many values of this release that "
        b"is, on which Comp_IDs and how many chains: the not_covered block of "
        b"the counts key, computed by the build."
    ),
    b"correction_formula": _CORRECTION_FORMULA.encode(),
    b"offset_source_values": _OFFSET_SOURCE_VALUES.encode(),
    b"offset_source_domain": ",".join(OFFSET_SOURCES).encode(),
    b"sidechain_transfer_policy": _SIDECHAIN_POLICY.encode(),
    b"units": (
        b"val_ppm, val_err_ppm, val_corrected_ppm and offset_applied_ppm are all "
        b"ppm. seq_id is a 1-based residue index. ambiguity_code is a BMRB "
        b"dictionary code. Every field repeats its unit in field-level metadata."
    ),
    b"sort_order": b"id,seq_id,atom_id",
    b"seq_id_semantics": (
        b"1-based position into the main table's sequence column: "
        b"sequence[seq_id-1] is the residue named by comp_id"
    ),
    b"raw_values_are_unfiltered": (
        b"The scoring path additionally drops values with Val_err > 1.3 ppm or "
        b"an ambiguity code outside {1,2,null} and averages degenerate partners; "
        b"none of that is applied here. Rows the scorer never used still carry a "
        b"corrected value, because the offset is a per-chain referencing "
        b"constant, not a property of the individual measurement."
    ),
    b"far_outlier_rule": _FAR_OUTLIER_RULE.encode(),
    b"key_uniqueness": (
        b"(id, seq_id, atom_id) is unique except where a deposition states one "
        b"atom twice: the same value under different Ambiguity_codes, or two "
        b"different values (a partial-occupancy or two-state pair, e.g. both "
        b"conformers of a side chain). Both rows are kept as deposited; filter "
        b"or average them yourself. Which keys repeat, by kind and chain: the "
        b"duplicate_keys block of the counts key, computed by the build."
    ),
}

#: Table-metadata keys a release build adds on top of :data:`PARQUET_METADATA`
#: (:func:`restamp_shift_parquet`): the release identity and the counts.
RELEASE_KEYS = ("dataset_version", "license", "code_commit", "bmrb_snapshot")


def import_pyarrow():
    """Import pyarrow, or exit with the invocation that would have it.

    Public within the package (``scripts/build_parquet_dataset.py`` needs the
    same guard) but deliberately not in ``trizod.shifts.__all__``: a build-time
    import shim is not part of the released API surface.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "pyarrow is required to write Parquet but is not a project "
            "dependency. Re-run with:  uv run --with pyarrow python ..."
        ) from exc
    return pa, pq


def shift_schema(mad_reference: MadReference):
    """Arrow schema of ``trizod_shifts.parquet``, units and all.

    ``mad_reference`` is the reference ``mad_z`` was computed against; it is
    embedded in the table metadata.
    """
    pa, _ = import_pyarrow()
    fields = [
        pa.field(
            name,
            arrow(pa),
            nullable=True,
            metadata={k.encode(): v.encode() for k, v in meta.items()},
        )
        for name, arrow, meta in _FIELDS
    ]
    assert [f.name for f in fields] == SHIFT_PARQUET_COLUMNS
    metadata = {**PARQUET_METADATA, **mad_reference.metadata()}
    return pa.schema(fields, metadata=metadata)


def write_shift_parquet(
    frames,
    out_path,
    compression_level: int = 19,
    batch_rows: int = 500_000,
    *,
    mad_reference: MadReference,
    stats: ShiftTableStats | None = None,
):
    """Stream ``(chain_id, DataFrame)`` pairs into the released shift Parquet.

    ``frames`` must be ordered by ``chain_id`` and each frame sorted by
    ``(seq_id, atom_id)`` — which is what
    :func:`trizod.shifts.extract.iter_shift_frames` and
    :func:`trizod.bmrb.bmrb.get_deposited_shifts` produce — so the file is
    globally sorted by ``(id, seq_id, atom_id)`` without buffering ~12 M rows.

    Each frame must already carry the correction columns
    (:func:`trizod.shifts.correct.annotate_offsets`) and the far-outlier columns
    (:func:`trizod.shifts.outliers.annotate_mad`). Pass the same
    ``mad_reference`` the frames were annotated with: it is embedded in the
    table metadata, and a file must describe the reference its ``mad_z`` uses.

    ``stats`` (a :class:`ShiftTableStats`) accumulates the counts of every
    frame written; a release build then restamps the file with them
    (:func:`restamp_shift_parquet`).

    Written to a sibling temp file and promoted only on success. ``frames`` is a
    lazy generator doing unguarded per-chain work, and closing a ParquetWriter
    mid-stream still writes a valid footer — so writing in place would leave a
    structurally perfect, silently truncated table that ``package_release``
    picks up and whose ``n_records`` is read back out of the file itself.

    Returns:
        ``(out_path, n_rows)``.
    """
    pa, pq = import_pyarrow()
    schema = shift_schema(mad_reference)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".partial")

    buffered: list[pd.DataFrame] = []
    buffered_rows = 0
    n_rows = 0
    prev_id = None

    def flush(writer):
        nonlocal buffered, buffered_rows
        if not buffered:
            return
        batch = pd.concat(buffered, ignore_index=True)
        writer.write_table(
            pa.Table.from_pandas(batch, schema=schema, preserve_index=False)
        )
        buffered, buffered_rows = [], 0

    try:
        with pq.ParquetWriter(
            tmp_path, schema, compression="zstd", compression_level=compression_level
        ) as writer:
            for cid, df in frames:
                if prev_id is not None and cid < prev_id:
                    raise ValueError(
                        f"frames must arrive in sorted id order, got {cid!r} after {prev_id!r}"
                    )
                prev_id = cid
                missing = set(SHIFT_PARQUET_COLUMNS) - {"id"} - set(df.columns)
                if missing:
                    raise ValueError(
                        f"frame {cid!r} is missing columns {sorted(missing)}"
                    )
                if stats is not None:
                    stats.add_frame(cid, df)
                df = df.copy()
                df.insert(0, "id", cid)
                buffered.append(df[SHIFT_PARQUET_COLUMNS])
                buffered_rows += len(df)
                n_rows += len(df)
                if buffered_rows >= batch_rows:
                    flush(writer)
            flush(writer)
        tmp_path.replace(out_path)
    finally:
        # Covers KeyboardInterrupt/SystemExit too, and is a no-op once the
        # promote above has moved the file.
        tmp_path.unlink(missing_ok=True)
    return out_path, n_rows


# --------------------------------------------------------------------------- #
# counts computed from the data, and the release restamp
# --------------------------------------------------------------------------- #

#: How many of the most frequent excluded Comp_IDs ``not_covered`` names.
TOP_NOT_COVERED = 12


class ShiftTableStats:
    """Counts of a shift table, accumulated chain by chain while it is written
    (:meth:`add_frame`) and, for the values it excludes, while the raw shifts
    are read (:meth:`add_raw`). :meth:`metadata` renders them as the table's
    ``counts`` key and the computed ``not_covered`` / ``key_uniqueness`` text,
    so no count in the file is typed by hand.
    """

    def __init__(self):
        from collections import Counter

        self.rows = 0
        self.chains = 0
        self.backbone_rows = 0
        self.corrected_rows = 0
        self.offset_source = Counter()
        self.far_assessed = 0
        self.far_flagged = 0
        self.far_chains = 0
        self.dup_same_value = Counter()  # chain -> keys repeating one value
        self.dup_different_value = Counter()  # chain -> keys with two values
        self.raw_chains = 0
        self.raw_values = 0
        self.not_covered_values = 0
        self.not_covered_comp_ids = Counter()
        self.not_covered_chains = 0

    def add_raw(self, cid: str, shifts) -> None:
        """One chain's raw ``ShiftTable.shifts`` tuples (comp_id at index 3),
        for a chain whose table the extraction accepted."""
        from trizod.constants import AA3TO1

        self.raw_chains += 1
        self.raw_values += len(shifts)
        excluded = [row[3] for row in shifts if row[3] not in AA3TO1]
        if excluded:
            self.not_covered_chains += 1
            self.not_covered_values += len(excluded)
            self.not_covered_comp_ids.update(excluded)

    def add_frame(self, cid: str, df: pd.DataFrame) -> None:
        """One chain's annotated frame, as written."""
        if df.empty:
            return
        self.chains += 1
        self.rows += len(df)
        self.backbone_rows += int(df["is_backbone"].sum())
        self.corrected_rows += int(df["val_corrected_ppm"].notna().sum())
        self.offset_source.update(df["offset_source"].astype(str).tolist())
        flags = df["is_far_outlier"]
        assessed = flags.notna()
        self.far_assessed += int(assessed.sum())
        flagged = int(flags[assessed].astype(bool).sum())
        self.far_flagged += flagged
        self.far_chains += flagged > 0
        keys = df[["seq_id", "atom_id"]]
        dup = keys.duplicated(keep=False)
        if dup.any():
            for _, group in df[dup].groupby(["seq_id", "atom_id"], sort=True):
                same = group["val_ppm"].nunique(dropna=False) == 1
                (self.dup_same_value if same else self.dup_different_value)[cid] += 1

    def counts(self) -> dict:
        """Every count, JSON-ready."""
        dep = self.raw_values
        top = self.not_covered_comp_ids.most_common(TOP_NOT_COVERED)
        return {
            "rows": self.rows,
            "chains": self.chains,
            "backbone_rows": self.backbone_rows,
            "sidechain_rows": self.rows - self.backbone_rows,
            "corrected_rows": self.corrected_rows,
            "offset_source": dict(sorted(self.offset_source.items())),
            "far_outlier": {
                "assessed": self.far_assessed,
                "flagged": self.far_flagged,
                "chains": self.far_chains,
            },
            "duplicate_keys": {
                "same_value": sum(self.dup_same_value.values()),
                "different_value": sum(self.dup_different_value.values()),
                "same_value_by_chain": dict(sorted(self.dup_same_value.items())),
                "different_value_by_chain": dict(
                    sorted(self.dup_different_value.items())
                ),
            },
            "not_covered": (
                {
                    "values": self.not_covered_values,
                    "deposited_values": dep,
                    "share": round(self.not_covered_values / dep, 6) if dep else None,
                    "comp_ids": len(self.not_covered_comp_ids),
                    "chains": self.not_covered_chains,
                    "of_chains": self.raw_chains,
                    "most_frequent": dict(top),
                }
                if self.raw_chains
                else "not counted: the raw shifts were not read"
            ),
        }

    def metadata(self) -> dict[str, str]:
        """``counts`` plus the ``not_covered`` and ``key_uniqueness`` text with
        this table's numbers."""
        import json

        c = self.counts()
        meta = {"counts": json.dumps(c, sort_keys=True)}
        nc = c["not_covered"]
        if isinstance(nc, dict):
            pct = f"{100 * nc['share']:.3f} %" if nc["share"] is not None else "–"
            meta["not_covered"] = (
                "Shifts assigned to a NON-CANONICAL residue or group are excluded, "
                "because seq_id would not index the released sequence: "
                f"{nc['values']:,} of the {nc['deposited_values']:,} deposited "
                f"values ({pct}) of the released chains, on {nc['comp_ids']:,} "
                f"distinct Comp_IDs across {nc['chains']:,} of the "
                f"{nc['of_chains']:,} chains, most often "
                + ", ".join(nc["most_frequent"])
                + " (the PTMs, non-standard residues and terminal or lipid "
                "groups). Read those from the BMRB entry directly. Per-Comp_ID "
                "counts: counts.not_covered"
            )
        d = c["duplicate_keys"]
        chains = sorted(
            set(d["same_value_by_chain"]) | set(d["different_value_by_chain"])
        )
        meta["key_uniqueness"] = (
            "(id, seq_id, atom_id) is unique except where a deposition states one "
            f"atom twice: {d['same_value']} keys repeat the same value under "
            f"different Ambiguity_codes and {d['different_value']} keys carry two "
            "different values (a partial-occupancy or two-state pair, e.g. both "
            "conformers of a side chain), in "
            + (", ".join(chains) if chains else "no chain")
            + ". Both rows are kept as deposited; filter or average them "
            "yourself. Per chain: counts.duplicate_keys"
        )
        return meta


def iter_file_frames(path, columns=None):
    """``(chain_id, DataFrame)`` of an existing shift Parquet, chain by chain in
    file order (a chain may span two row groups)."""
    _, pq = import_pyarrow()
    pf = pq.ParquetFile(path)
    carry = None
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=columns).to_pandas()
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
        if df.empty:
            continue
        last = df["id"].iloc[-1]
        tail = df["id"] == last
        carry = df[tail]
        for cid, group in df[~tail].groupby("id", sort=False):
            yield cid, group.reset_index(drop=True)
    if carry is not None and not carry.empty:
        yield carry["id"].iloc[0], carry.reset_index(drop=True)


def data_fingerprint_streamed(path) -> str:
    """SHA-256 of a Parquet's data alone: every row group's record batches as
    Arrow IPC messages, table and field metadata dropped. A metadata-only
    restamp (:func:`restamp_shift_parquet`) leaves it unchanged."""
    import hashlib

    pa, pq = import_pyarrow()
    pf = pq.ParquetFile(path)
    bare = pa.schema([f.remove_metadata() for f in pf.schema_arrow])
    h = hashlib.sha256()
    for i in range(pf.num_row_groups):
        table = pf.read_row_group(i).cast(bare)
        for batch in table.to_batches():
            h.update(batch.serialize().to_pybytes())
    return h.hexdigest()


def restamp_shift_parquet(
    path, metadata: dict, out_path=None, compression_level: int = 19
) -> Path:
    """Rewrite a shift Parquet with its table metadata updated by ``metadata``
    (str -> str; field metadata, data and row-group layout unchanged), into
    ``out_path`` (default: in place, through a sibling temp file).

    The Parquet footer cannot be edited in place, and metadata a writer adds
    after its schema is fixed is not what ``pyarrow.parquet.read_schema``
    returns, so the row groups are re-encoded under the new schema. The data
    are checked unchanged (:func:`data_fingerprint_streamed`) before the file
    is promoted.
    """
    _, pq = import_pyarrow()
    path = Path(path)
    out_path = Path(out_path) if out_path is not None else path
    pf = pq.ParquetFile(path)
    merged = dict(pf.schema_arrow.metadata or {})
    merged.update({k.encode(): str(v).encode() for k, v in metadata.items()})
    schema = pf.schema_arrow.with_metadata(merged)
    tmp_path = out_path.with_name(out_path.name + ".restamp")
    try:
        with pq.ParquetWriter(
            tmp_path, schema, compression="zstd", compression_level=compression_level
        ) as writer:
            for i in range(pf.num_row_groups):
                group = pf.read_row_group(i).replace_schema_metadata(merged)
                writer.write_table(group, row_group_size=max(group.num_rows, 1))
        if data_fingerprint_streamed(tmp_path) != data_fingerprint_streamed(path):
            raise RuntimeError(f"restamping {path} changed its data; not promoted")
        tmp_path.replace(out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return out_path
