"""Write the companion side-chain shift table as Parquet.

Flat long table, one row per deposited side-chain value, joining 1:1 on ``id``
with ``trizod_dataset.parquet``. Same codec as the deposit (zstd-19).

pyarrow is deliberately **not** a project dependency (the scoring pipeline and
the dataset chain do not need it); it is imported lazily here, exactly like
``scripts/build_parquet_dataset.py`` expects to be run:

    uv run --with pyarrow python scripts/build_parquet_dataset.py ...
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trizod.bmrb.bmrb import SIDECHAIN_COLUMNS

# the per-chain frame plus the chain key it is written under
SIDECHAIN_PARQUET_COLUMNS = ["id", *SIDECHAIN_COLUMNS]

PARQUET_METADATA = {
    b"dataset": b"TriZOD side-chain chemical shifts",
    b"description": b"Side-chain chemical shifts discarded by the backbone scoring path, AS DEPOSITED in BMRB: no re-referencing and no offset correction of any kind is applied. One row per value; join 1:1 on id with trizod_dataset.parquet (lacs_off_* there if you want to correct them yourself).",
    b"sort_order": b"id,seq_id,atom_id",
    b"seq_id_semantics": b"1-based position into the main table's sequence column: sequence[seq_id-1] is the residue named by comp_id",
    b"ambiguity_code_semantics": b"BMRB _Atom_chem_shift.Ambiguity_code as deposited, null when absent; 2 = stereo-unassigned pair, 3 = aromatic ring degenerate (NOT filtered out, unlike the backbone read)",
    b"key_uniqueness": b"(id, seq_id, atom_id) is unique except where a deposition states the same value twice under different Ambiguity_codes; those rows are kept as deposited (3 rows in the 2026-07 corpus)",
}


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


def sidechain_schema():
    """Arrow schema of ``trizod_sidechain_shifts.parquet``."""
    pa, _ = _import_pyarrow()
    return pa.schema(
        [
            ("id", pa.string()),
            ("seq_id", pa.int32()),
            ("comp_id", pa.string()),
            ("atom_id", pa.string()),
            ("atom_type", pa.string()),
            # float64: deposited precision round-trips exactly, no re-referencing
            ("val", pa.float64()),
            ("val_err", pa.float64()),
            ("ambiguity_code", pa.int8()),
        ]
    )


def write_sidechain_parquet(
    frames, out_path, compression_level: int = 19, batch_rows: int = 500_000
):
    """Stream ``(chain_id, DataFrame)`` pairs into the companion Parquet.

    ``frames`` must be ordered by ``chain_id`` and each frame sorted by
    ``(seq_id, atom_id)`` — which is what
    :func:`trizod.sidechain.extract.iter_sidechain_frames` and
    :func:`trizod.bmrb.bmrb.get_sidechain_shifts` produce — so the file is
    globally sorted by ``(id, seq_id, atom_id)`` without buffering 3.5 M rows.

    Returns:
        ``(out_path, n_rows)``.
    """
    pa, pq = _import_pyarrow()
    schema = sidechain_schema().with_metadata(PARQUET_METADATA)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    with pq.ParquetWriter(
        out_path, schema, compression="zstd", compression_level=compression_level
    ) as writer:
        for cid, df in frames:
            if prev_id is not None and cid < prev_id:
                raise ValueError(
                    f"frames must arrive in sorted id order, got {cid!r} after {prev_id!r}"
                )
            prev_id = cid
            df = df.copy()
            df.insert(0, "id", cid)
            buffered.append(df[SIDECHAIN_PARQUET_COLUMNS])
            buffered_rows += len(df)
            n_rows += len(df)
            if buffered_rows >= batch_rows:
                flush(writer)
        flush(writer)
    return out_path, n_rows
