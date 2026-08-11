"""Side-chain chemical shifts: the values the scoring pipeline discards.

``trizod.bmrb.get_sidechain_shifts()`` recovers them per chain; this package
turns that into the released companion table ``trizod_sidechain_shifts.parquet``
(one row per deposited side-chain value, joining 1:1 on ``id`` with the main
dataset Parquet).
"""

from .extract import (
    SIDECHAIN_PARQUET_NAME,
    chain_id,
    entry_chains,
    entry_sidechain_frames,
    iter_chains,
    iter_sidechain_frames,
)
from .parquet import write_sidechain_parquet

__all__ = [
    "SIDECHAIN_PARQUET_NAME",
    "chain_id",
    "entry_chains",
    "entry_sidechain_frames",
    "iter_chains",
    "iter_sidechain_frames",
    "write_sidechain_parquet",
]
