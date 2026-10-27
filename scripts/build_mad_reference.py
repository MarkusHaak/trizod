"""Freeze the far-outlier reference from a released shift table.

Reads ``val_ppm`` of every C, CA, CB and N row of ``trizod_shifts.parquet``,
computes the median and raw MAD per ``(comp_id, atom_id)``
(:func:`trizod.shifts.outliers.compute_mad_reference`) and writes the CSV that
``build_parquet_dataset.py --shifts-out`` flags against. Run it once per frozen
reference, not once per build: the point of committing the table is that the
flag stays put when BMRB moves.

The input may be a shift table written before the flag existed -- only the
four columns read here are needed, and ``val_ppm`` is the deposited value, so
the reference does not depend on the flag it feeds.

Usage:
    uv run --with pyarrow python scripts/build_mad_reference.py \\
        --shifts data/interim/build/parquet-2026-10/trizod_shifts.parquet \\
        --source "trizod_shifts.parquet of build 2026-10 (trizod-5b8147a)"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trizod.shifts.outliers import (
    MAD_K,
    MAD_REFERENCE_PATH,
    compute_mad_reference,
    write_mad_reference,
)
from trizod.shifts.parquet import import_pyarrow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shifts", type=Path, required=True, help="trizod_shifts.parquet")
    ap.add_argument(
        "--source",
        required=True,
        help="provenance written into the CSV header (which build, which commit)",
    )
    ap.add_argument("--out", type=Path, default=MAD_REFERENCE_PATH)
    args = ap.parse_args()

    _, pq = import_pyarrow()
    table = pq.read_table(
        args.shifts,
        columns=["seq_id", "comp_id", "atom_id", "val_ppm"],
        filters=[("atom_id", "in", list(MAD_K))],
    )
    reference = compute_mad_reference(table.to_pandas())
    write_mad_reference(reference, args.out, source=args.source)
    print(
        f"wrote {args.out}: {len(reference)} groups from {table.num_rows:,} shifts "
        f"(smallest group n={reference['n'].min():,})"
    )


if __name__ == "__main__":
    main()
