"""The deposited chemical-shift table, and its re-referencing.

``trizod.bmrb.get_deposited_shifts()`` reads every assigned shift **on a
canonical residue** of one chain — backbone and side chain, as deposited. Shifts
on non-canonical residues are excluded, because ``seq_id`` would not then index
the released ``sequence``; systematically the PTMs, the non-standard residues
and the terminal or lipid groups. A release build counts them
(:class:`~trizod.shifts.parquet.ShiftTableStats`) and states the numbers in the
table's ``not_covered`` and ``counts`` metadata.

This package attaches TriZOD's re-referencing to the rows that remain
(:mod:`trizod.shifts.correct`, where the per-nucleus transfer policy and the
ppm/sigma unit contract live), flags far-outlier backbone shifts against a
frozen median/MAD reference (:mod:`trizod.shifts.outliers`) and writes the
released table ``trizod_shifts.parquet``.
"""

from .correct import (
    OFFSET_SOURCES,
    SIDECHAIN_CARBON_LACS_ATOM,
    SIDECHAIN_CORRECTED_NUCLEI,
    SIDECHAIN_MAX_ABS_LACS_PPM,
    ChainOffsets,
    annotate_offsets,
    backbone_slot,
    chain_offsets,
)
from .extract import (
    SHIFTS_PARQUET_NAME,
    chain_id,
    entry_chains,
    iter_chains,
    iter_shift_frames,
)
from .outliers import MAD_K, annotate_mad, load_mad_reference
from .parquet import (
    PARQUET_METADATA,
    RELEASE_KEYS,
    SHIFT_PARQUET_COLUMNS,
    ShiftTableStats,
    iter_file_frames,
    restamp_shift_parquet,
    shift_schema,
    write_shift_parquet,
)

__all__ = [
    "MAD_K",
    "OFFSET_SOURCES",
    "PARQUET_METADATA",
    "RELEASE_KEYS",
    "SHIFTS_PARQUET_NAME",
    "SHIFT_PARQUET_COLUMNS",
    "SIDECHAIN_CARBON_LACS_ATOM",
    "SIDECHAIN_CORRECTED_NUCLEI",
    "SIDECHAIN_MAX_ABS_LACS_PPM",
    "ChainOffsets",
    "ShiftTableStats",
    "annotate_mad",
    "annotate_offsets",
    "backbone_slot",
    "chain_id",
    "chain_offsets",
    "entry_chains",
    "iter_chains",
    "iter_file_frames",
    "iter_shift_frames",
    "load_mad_reference",
    "restamp_shift_parquet",
    "shift_schema",
    "write_shift_parquet",
]
