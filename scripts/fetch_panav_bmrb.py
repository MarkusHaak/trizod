#!/usr/bin/env python3
"""Compute PANAV offsets locally for all BMRB entries.

Runs the PANAV JAR directly on local NMR-STAR files (~8 min for 17k entries)
instead of calling the BMRB API (~9 hours). Supports resumption.

Setup:
    curl -L -o tools/panav.jar https://raw.githubusercontent.com/bmrb-io/BMRB-API/master/server/wsgi/bmrbapi/submodules/panav/panav.jar

Usage:
    uv run python scripts/fetch_panav_bmrb.py
    uv run python scripts/fetch_panav_bmrb.py --input-dir data/raw/bmrb_entries --workers 8
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pynmrstar

logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
logger = logging.getLogger("panav_local")

# Flag for graceful shutdown on Ctrl+C
_shutdown = False


def _handle_sigint(signum, frame):
    global _shutdown
    if _shutdown:
        # Second Ctrl+C: force exit
        logger.warning("Force exit")
        os._exit(1)
    _shutdown = True
    logger.info("Shutting down gracefully (Ctrl+C again to force)...")


def run_panav_on_entry(entry_dir: Path, panav_jar: str) -> tuple[str, dict | None]:
    """Run PANAV locally on an NMR-STAR entry directory.

    Returns (entry_id, result_dict) or (entry_id, None) on failure.
    """
    eid = entry_dir.name[3:]  # bmr15000 → 15000
    str_files = sorted(entry_dir.glob("*.str"))
    if not str_files:
        return eid, None

    try:
        entry = pynmrstar.Entry.from_file(str(str_files[0]))
        loops = entry.get_loops_by_category("atom_chem_shift")
    except Exception as e:
        logger.debug(f"bmr{eid}: parse error: {e}")
        return eid, None

    if not loops:
        return eid, None

    result = {"entry_id": eid, "shift_lists": {}}
    for i, loop in enumerate(loops):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".str", delete=False) as tmp:
            tmp.write(str(loop))
            tmp_path = tmp.name

        try:
            proc = subprocess.run(
                [
                    "java",
                    "-cp",
                    panav_jar,
                    "CLI",
                    "-f",
                    "star",
                    "-i",
                    tmp_path,
                    "-j",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                offsets = data.get("Reference offsets", {})
                outliers = data.get("Outliers", [])
                if offsets:
                    result["shift_lists"][str(i)] = {
                        "offsets": offsets,
                        "n_deviants": sum(
                            1 for o in outliers if len(o) >= 4 and o[3] == "D"
                        ),
                        "n_suspicious": sum(
                            1 for o in outliers if len(o) >= 4 and o[3] == "S"
                        ),
                    }
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            logger.debug(f"bmr{eid} loop {i}: {e}")
        finally:
            os.unlink(tmp_path)

    return eid, result if result["shift_lists"] else None


def load_existing(output_path: Path) -> dict:
    """Load previously computed results to enable resumption."""
    if output_path.exists():
        with output_path.open() as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} existing results from {output_path}")
        return data
    return {}


def save_results(results: dict, output_path: Path):
    """Save results atomically (write to .tmp then rename)."""
    tmp = output_path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    tmp.rename(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compute PANAV offsets locally using panav.jar"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/bmrb_entries"),
        help="Directory containing bmr* folders (default: data/raw/bmrb_entries)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/panav_offsets.json"),
        help="Output JSON file (default: data/external/panav_offsets.json)",
    )
    parser.add_argument(
        "--panav-jar",
        type=str,
        default="tools/panav.jar",
        help="Path to panav.jar (default: tools/panav.jar)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save checkpoint every N entries (default: 500)",
    )
    args = parser.parse_args()

    # Verify panav.jar exists
    if not Path(args.panav_jar).exists():
        logger.error(
            f"panav.jar not found at {args.panav_jar}. Download with:\n"
            "  curl -L -o tools/panav.jar https://raw.githubusercontent.com/"
            "bmrb-io/BMRB-API/master/server/wsgi/bmrbapi/submodules/panav/panav.jar"
        )
        return

    entry_dirs = sorted(
        p for p in args.input_dir.iterdir() if p.is_dir() and p.name.startswith("bmr")
    )
    logger.info(f"Found {len(entry_dirs)} entries")

    # Resume from existing results
    results = load_existing(args.output)
    done_ids = set(results.keys())
    remaining = [d for d in entry_dirs if d.name[3:] not in done_ids]
    logger.info(f"Remaining: {len(remaining)}")

    if not remaining:
        logger.info("All entries already processed")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _handle_sigint)

    completed = 0
    with_data = 0
    start_time = time.time()
    last_save = 0

    def _process_result(eid, data):
        nonlocal completed, with_data, last_save
        results[eid] = data
        completed += 1
        if data is not None:
            with_data += 1
        if completed - last_save >= args.save_every:
            save_results(results, args.output)
            last_save = completed
            elapsed = time.time() - start_time
            rate = completed / elapsed
            eta = (len(remaining) - completed) / rate
            logger.info(
                f"[{completed}/{len(remaining)}] "
                f"with_data={with_data} | "
                f"{rate:.1f} entries/s | ETA {eta / 60:.1f}min"
            )

    if args.workers <= 1:
        for entry_dir in remaining:
            if _shutdown:
                break
            eid, data = run_panav_on_entry(entry_dir, args.panav_jar)
            _process_result(eid, data)
    else:
        batch_size = args.workers * 4
        for batch_start in range(0, len(remaining), batch_size):
            if _shutdown:
                break
            batch = remaining[batch_start : batch_start + batch_size]

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(run_panav_on_entry, d, args.panav_jar): d for d in batch
                }
                for future in as_completed(futures):
                    if _shutdown:
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    eid, data = future.result()
                    _process_result(eid, data)

    save_results(results, args.output)
    elapsed = time.time() - start_time
    total_with_data = sum(1 for v in results.values() if v is not None)
    logger.info(
        f"{'Interrupted' if _shutdown else 'Done'} after {elapsed / 60:.1f}min. "
        f"{total_with_data} with PANAV data, "
        f"{len(results) - total_with_data} without, "
        f"of {len(results)} total"
    )


if __name__ == "__main__":
    main()
