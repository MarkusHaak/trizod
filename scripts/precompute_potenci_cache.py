#!/usr/bin/env python3
"""Precompute POTENCI predictions and cache them for faster pipeline runs.

Usage:
    uv run python scripts/precompute_potenci_cache.py --input-dir data/bmrb_entries --cache-dir tmp
    uv run python scripts/precompute_potenci_cache.py --input-dir tests/bmrb_subset --cache-dir tmp

The cache is keyed by (seq, temperature, pH, ionic_strength) so it stays valid
across filter/scoring changes. Only POTENCI input changes invalidate it.

An index file (potenci/_index.tsv) tracks which entries have been processed,
so re-runs skip parsing entirely for known entries.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trizod.bmrb.bmrb as bmrb
import trizod.potenci.potenci as potenci
from trizod.trizod import (
    _potenci_cache_key,
    find_bmrb_files,
    save_potenci_cache,
)


def load_index(potenci_dir):
    """Load the entry→cache_key index. Returns set of processed entry IDs."""
    index_fp = os.path.join(potenci_dir, "_index.tsv")
    processed = set()
    if os.path.exists(index_fp):
        with open(index_fp) as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    processed.add(parts[0])
    return processed


def append_index(potenci_dir, entry_id, keys):
    """Append entry and its cache keys to the index file."""
    index_fp = os.path.join(potenci_dir, "_index.tsv")
    with open(index_fp, "a") as f:
        f.write(f"{entry_id}\t{','.join(keys)}\n")


def main():
    parser = argparse.ArgumentParser(description="Precompute POTENCI prediction cache")
    parser.add_argument(
        "--input-dir", required=True, help="Directory with BMRB NMR-STAR files"
    )
    parser.add_argument(
        "--cache-dir", default="./tmp", help="Cache directory (default: ./tmp)"
    )
    parser.add_argument(
        "--file-pattern", default=r"bmr(\d+)_3\.str", help="BMRB file pattern"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show BMRB parsing warnings and POTENCI debug messages",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Ignore existing index and reprocess all entries",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s : %(message)s")
    logger = logging.getLogger("precompute")
    logger.setLevel(logging.INFO)
    if not args.verbose:
        logging.getLogger("trizod.bmrb").setLevel(logging.CRITICAL)
        logging.getLogger("trizod.potenci").setLevel(logging.CRITICAL)
        logging.getLogger("trizod").setLevel(logging.CRITICAL)

    cache_dir = os.path.abspath(args.cache_dir)
    potenci_dir = os.path.join(cache_dir, "potenci")
    os.makedirs(potenci_dir, exist_ok=True)

    logger.info(f"Finding BMRB files in {args.input_dir}")
    bmrb_files = find_bmrb_files(args.input_dir, args.file_pattern)
    logger.info(f"Found {len(bmrb_files)} BMRB files")

    # Load index of already-processed entries
    if args.reindex:
        processed = set()
        index_fp = os.path.join(potenci_dir, "_index.tsv")
        if os.path.exists(index_fp):
            os.remove(index_fp)
    else:
        processed = load_index(potenci_dir)
    if processed:
        logger.info(f"Skipping {len(processed)} already-indexed entries")

    computed, skipped, failed = 0, 0, 0
    failed_entries = []
    seen_keys = set()
    start = time.time()

    pbar = tqdm(bmrb_files.items(), desc="Precomputing POTENCI", unit="entry")
    for entry_id, filepath in pbar:
        if entry_id in processed:
            skipped += 1
            pbar.set_postfix(new=computed, skip=skipped, fail=failed)
            continue

        try:
            entry = bmrb.BmrbEntry(entry_id, os.path.dirname(filepath))
        except Exception as e:
            failed += 1
            failed_entries.append((entry_id, "parse", str(e)))
            append_index(potenci_dir, entry_id, ["FAILED"])
            continue

        try:
            peptide_shifts = entry.get_peptide_shifts()
        except Exception as e:
            failed += 1
            failed_entries.append((entry_id, "shifts", str(e)))
            append_index(potenci_dir, entry_id, ["FAILED"])
            continue

        entry_keys = []
        for (_stID, _entity_assemID, entityID), (
            _shifts,
            condID,
            _assemID,
            _sampleIDs,
        ) in peptide_shifts.items():
            try:
                entity = entry.entities[entityID]
                seq = entity.seq
                if not seq:
                    continue
                conds = entry.conditions.get(condID, None)
                if conds is None:
                    continue

                temperature = conds.get_temperature()
                pH = conds.get_pH()
                ion = conds.get_ionic_strength()
                if temperature is None or pH is None or ion is None:
                    continue
                if np.isnan(temperature) or np.isnan(pH) or np.isnan(ion):
                    continue

                key = _potenci_cache_key(seq, temperature, pH, ion)
                if key in seen_keys:
                    entry_keys.append(key)
                    continue
                seen_keys.add(key)

                use_ph_corr = pH != 7.0
                predshiftdct = potenci.get_pred_shifts(
                    seq, temperature, pH, ion, use_ph_corr
                )
                save_potenci_cache(cache_dir, seq, temperature, pH, ion, predshiftdct)
                entry_keys.append(key)
                computed += 1
            except Exception as e:
                failed += 1
                failed_entries.append((entry_id, "potenci", str(e)))

        append_index(potenci_dir, entry_id, entry_keys)
        pbar.set_postfix(new=computed, skip=skipped, fail=failed)

    elapsed = time.time() - start
    logger.info(
        f"Done in {elapsed:.1f}s: {computed} computed, {skipped} skipped, {failed} failed"
    )
    if failed_entries:
        log_fp = os.path.join(cache_dir, "potenci_failures.txt")
        with open(log_fp, "w") as f:
            for entry_id, stage, error in failed_entries:
                f.write(f"{entry_id}\t{stage}\t{error}\n")
        logger.info(f"Failed entries written to {log_fp}")


if __name__ == "__main__":
    main()
