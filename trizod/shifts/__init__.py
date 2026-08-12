"""The deposited chemical-shift table, and its re-referencing.

``trizod.bmrb.get_deposited_shifts()`` reads every assigned shift **on a
canonical residue** of one chain — backbone and side chain, as deposited. Shifts
on non-canonical residues are excluded, because ``seq_id`` would not then index
the released ``sequence``; that is 16,502 values (0.139 % of the 11,855,542
deposited values) on 290 distinct ``Comp_ID``s across 1,050 of the 16,851 chains
of release 2026-08, systematically the PTMs (SEP, TPO, PTR, TYS, HYP, ALY, MLY),
the non-standard residues (ORN, AIB, ABA, NLE, DPR, PCA) and the terminal or
lipid groups (ACE, NH2, MYR).

This package attaches TriZOD's re-referencing to the rows that remain
(:mod:`trizod.shifts.correct`, where the per-nucleus transfer policy and the
ppm/sigma unit contract live) and writes the released table
``trizod_shifts.parquet``.
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
from .parquet import (
    PARQUET_METADATA,
    SHIFT_PARQUET_COLUMNS,
    shift_schema,
    write_shift_parquet,
)

__all__ = [
    "OFFSET_SOURCES",
    "PARQUET_METADATA",
    "SHIFTS_PARQUET_NAME",
    "SHIFT_PARQUET_COLUMNS",
    "SIDECHAIN_CARBON_LACS_ATOM",
    "SIDECHAIN_CORRECTED_NUCLEI",
    "SIDECHAIN_MAX_ABS_LACS_PPM",
    "ChainOffsets",
    "annotate_offsets",
    "backbone_slot",
    "chain_id",
    "chain_offsets",
    "entry_chains",
    "iter_chains",
    "iter_shift_frames",
    "shift_schema",
    "write_shift_parquet",
]
