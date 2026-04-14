"""Bulk-download all BMRB LACS validation reports.

Fetches the directory listing, then downloads all .str files in parallel.
Skips already-cached files.

Usage:
    uv run python scripts/download_lacs_reports.py [--lacs-dir DIR] [--workers N]
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def fetch_lacs_listing() -> list[str]:
    """Fetch all LACS report IDs from the BMRB FTP directory."""
    url = "https://bmrb.io/ftp/pub/bmrb/validation_reports/LACS/"
    with urllib.request.urlopen(url, timeout=120) as resp:
        html = resp.read().decode("utf-8")
    return sorted(set(re.findall(r"bmr(\d+)_LACS\.str", html)), key=int)


def download_one(entry_id: str, lacs_dir: Path) -> tuple[str, bool]:
    """Download a single LACS report. Returns (id, success)."""
    cache_path = lacs_dir / f"bmr{entry_id}_LACS.str"
    if cache_path.exists():
        return entry_id, True

    url = f"https://bmrb.io/ftp/pub/bmrb/validation_reports/LACS/bmr{entry_id}_LACS.str"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            text = resp.read().decode("utf-8")
        cache_path.write_text(text)
        return entry_id, True
    except Exception:
        return entry_id, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lacs-dir", default="data/bmrb_lacs", help="Output directory")
    parser.add_argument("--workers", type=int, default=16, help="Parallel downloads")
    args = parser.parse_args()

    lacs_dir = Path(args.lacs_dir)
    lacs_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching BMRB LACS directory listing...")
    all_ids = fetch_lacs_listing()
    print(f"Total LACS reports on BMRB: {len(all_ids)}")

    already = sum(1 for i in all_ids if (lacs_dir / f"bmr{i}_LACS.str").exists())
    to_download = len(all_ids) - already
    print(f"Already cached: {already}")
    print(f"To download:    {to_download}")

    if to_download == 0:
        print("Nothing to download.")
        # Update index
        index_path = lacs_dir / "_available_ids.txt"
        index_path.write_text("\n".join(all_ids))
        return

    downloaded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, eid, lacs_dir): eid
            for eid in all_ids
            if not (lacs_dir / f"bmr{eid}_LACS.str").exists()
        }
        for future in as_completed(futures):
            eid, success = future.result()
            if success:
                downloaded += 1
            else:
                failed += 1
            if (downloaded + failed) % 500 == 0:
                print(
                    f"  ... {downloaded} downloaded, {failed} failed "
                    f"({downloaded + failed}/{to_download})"
                )

    print(f"\nDone: {downloaded} downloaded, {failed} failed")
    print(f"Total cached: {already + downloaded}")

    # Update index
    index_path = lacs_dir / "_available_ids.txt"
    index_path.write_text("\n".join(all_ids))
    print(f"Index updated: {index_path}")


if __name__ == "__main__":
    main()
