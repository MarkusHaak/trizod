"""Download the dated sources for the release cross-references.

Fetches BMRB's BMRB<->PDB match list, the wwPDB SEQRES sequences and SIFTS'
PDB-chain -> UniProt table into one dated directory, with a ``MANIFEST.json``
(URL, retrieval time, Last-Modified, size, SHA-256) that
``scripts/build_parquet_dataset.py --xref-sources`` verifies before use. Run it
once per release, next to the BMRB snapshot; the build itself never downloads.
See ``trizod/bmrb/xref_sources.py``.

Usage:
    uv run python scripts/fetch_xref_sources.py \
        [--out-dir data/external/xref_sources/<YYYY-MM-DD>]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from trizod import paths
from trizod.bmrb import xref_sources as xs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    now = datetime.now(timezone.utc)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=paths.ROOT / "data" / "external" / "xref_sources" / f"{now:%Y-%m-%d}",
        help="target directory (default: data/external/xref_sources/<today>)",
    )
    args = ap.parse_args()
    if (args.out_dir / xs.MANIFEST).exists():
        ap.error(f"{args.out_dir / xs.MANIFEST} exists; sources are never overwritten")
    manifest = xs.fetch(args.out_dir, f"{now:%Y-%m-%dT%H:%M:%SZ}")
    for name, info in manifest["files"].items():
        print(f"{name:45s} {info['bytes']:>11,d} B  {info['last_modified']}")
    print(f"wrote {args.out_dir / xs.MANIFEST}")


if __name__ == "__main__":
    main()
