"""
Compare our LACS Python reimplementation against BMRB's pre-computed LACS reports.

Downloads LACS validation reports from bmrb.io, parses the offsets, runs our
implementation on the same entries, and produces a comparison summary.

Usage:
    uv run python scripts/compare_lacs_bmrb.py [--max-entries N] [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Backbone atom column order in bbshifts_arr from get_valid_bbshifts()
# Must match trizod.constants.BACKBONE_ATOMS
_BB_COLS = {"C": 0, "CA": 1, "CB": 2, "HA": 3, "H": 4, "N": 5, "HB": 6}


def parse_bmrb_lacs_file(text: str) -> dict[str, float | None]:
    """Extract per-atom offsets from a BMRB LACS .str file.

    Returns dict with keys CA, CB, HA, C (CO) mapping to offset values.
    """
    offsets = {"CA": None, "CB": None, "HA": None, "C": None}

    # Map BMRB saveframe atom names to our atom names
    atom_map = {"CA": "CA", "CB": "CB", "HA": "HA", "CO": "C"}

    for sf_atom, our_atom in atom_map.items():
        # Find the saveframe for this atom
        pattern = rf"save_LACS_CACB_{sf_atom}_output(.*?)save_"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            continue

        block = match.group(1)
        # Extract the offset value
        offset_match = re.search(
            r"_LACS_plot\.Y_axis_chem_shift_offset\s+([-\d.]+)", block
        )
        if offset_match:
            val = offset_match.group(1)
            parsed = float(val) if val not in ("", ".") else None
            if parsed is not None:
                offsets[our_atom] = parsed

    return offsets


def get_available_lacs_ids(lacs_dir: Path) -> set[str]:
    """Get the set of BMRB IDs that have LACS reports.

    Uses the locally cached .str files as the source of truth. If an index
    file exists, reads that instead (faster than globbing thousands of files).
    To refresh: delete _available_ids.txt and re-run.
    """
    index_path = lacs_dir / "_available_ids.txt"
    if index_path.exists():
        return set(index_path.read_text().splitlines())

    # Build index from locally cached files
    ids = set()
    for p in lacs_dir.glob("bmr*_LACS.str"):
        m = re.match(r"bmr(\d+)_LACS\.str", p.name)
        if m:
            ids.add(m.group(1))

    if ids:
        index_path.write_text("\n".join(sorted(ids, key=int)))
        print(f"Built index from {len(ids)} cached LACS reports")
    else:
        # No local files — try fetching the directory listing from BMRB
        print("No cached LACS reports found. Fetching BMRB directory listing...")
        url = "https://bmrb.io/ftp/pub/bmrb/validation_reports/LACS/"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                html = resp.read().decode("utf-8")
            ids = set(re.findall(r"bmr(\d+)_LACS\.str", html))
            index_path.write_text("\n".join(sorted(ids, key=int)))
            print(f"Found {len(ids)} LACS reports on BMRB")
        except Exception as e:
            print(f"Warning: could not fetch LACS listing: {e}")

    return ids


def download_lacs_report(entry_id: str, cache_dir: Path) -> str | None:
    """Download a LACS report from BMRB FTP, with local caching."""
    cache_path = cache_dir / f"bmr{entry_id}_LACS.str"
    if cache_path.exists():
        return cache_path.read_text()

    url = f"https://bmrb.io/ftp/pub/bmrb/validation_reports/LACS/bmr{entry_id}_LACS.str"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            text = resp.read().decode("utf-8")
        cache_path.write_text(text)
        return text
    except Exception:
        return None


def extract_shifts_for_lacs(entry, st_id, ea_id, e_id):
    """Extract per-atom shift arrays from a BmrbEntry for LACS analysis.

    Returns (seq, seq_nums, obs_shifts) or None if extraction fails.
    """
    from trizod.bmrb import bmrb

    peptide_shifts = entry.get_peptide_shifts()
    key = (st_id, ea_id, e_id)
    if key not in peptide_shifts:
        return None

    shifts_data, cond_id, assem_id, sample_ids = peptide_shifts[key]
    entity = entry.entities[e_id]
    seq = entity.seq
    if not seq:
        return None

    result = bmrb.get_valid_bbshifts(shifts_data, seq)
    if result is None:
        return None

    bbshifts_arr, bbshifts_mask = result
    n = len(seq)

    # Build per-atom arrays with NaN for missing
    obs_shifts = {}
    for atom, col_idx in _BB_COLS.items():
        if atom == "HB":
            continue  # LACS doesn't use HB directly
        arr = np.full(n, np.nan)
        for i in range(n):
            if bbshifts_mask[i, col_idx]:
                arr[i] = bbshifts_arr[i, col_idx]
        obs_shifts[atom] = arr

    seq_nums = np.arange(1, n + 1)
    return seq, seq_nums, obs_shifts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-entries",
        type=int,
        default=500,
        help="Maximum number of entries to compare (default: 500)",
    )
    parser.add_argument(
        "--bmrb-dir",
        type=str,
        default="data/bmrb_entries",
        help="Directory with raw BMRB entries",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="tmp",
        help="Cache directory with pickled BmrbEntry objects",
    )
    parser.add_argument(
        "--lacs-dir",
        type=str,
        default="data/bmrb_lacs",
        help="Directory to cache downloaded LACS reports",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    lacs_dir = Path(args.lacs_dir)
    lacs_dir.mkdir(parents=True, exist_ok=True)

    # Find available pickled entries
    pkl_dir = cache_dir / "bmrb_entries"
    if not pkl_dir.exists():
        print(f"Error: pickle cache directory not found: {pkl_dir}")
        sys.exit(1)

    pkl_files = sorted(pkl_dir.glob("*.pkl"))
    available_ids = {p.stem for p in pkl_files}
    print(f"Found {len(available_ids)} cached BMRB entries")

    # Get IDs that actually have LACS reports (avoids thousands of 404s)
    lacs_ids = get_available_lacs_ids(lacs_dir)
    overlap_ids = available_ids & lacs_ids
    print(f"LACS reports available on BMRB: {len(lacs_ids)}")
    print(f"Overlap with our entries:       {len(overlap_ids)}")

    # Import our LACS after setup to avoid slow import at argparse time
    from trizod.lacs import compute_lacs_offsets

    # Process entries
    atoms = ["CA", "CB", "HA", "C"]
    bmrb_offsets_all = {a: [] for a in atoms}
    our_offsets_all = {a: [] for a in atoms}
    entry_ids_compared = []

    n_downloaded = 0
    n_parse_fail = 0
    n_extract_fail = 0
    n_compared = 0

    # Only iterate over IDs that have LACS reports — no wasted 404s
    sorted_ids = sorted(overlap_ids, key=int)

    for entry_id in sorted_ids:
        if n_compared >= args.max_entries:
            break

        # Download BMRB LACS report (cached after first download)
        text = download_lacs_report(entry_id, lacs_dir)
        if text is None:
            continue
        n_downloaded += 1

        # Parse BMRB offsets
        bmrb_offsets = parse_bmrb_lacs_file(text)
        if all(v is None for v in bmrb_offsets.values()):
            n_parse_fail += 1
            continue

        # Load our parsed entry
        pkl_path = pkl_dir / f"{entry_id}.pkl"
        try:
            with open(pkl_path, "rb") as f:
                entry = pickle.load(f)
        except Exception:
            n_parse_fail += 1
            continue

        # Get peptide shifts — try first available (stID, eaID, eID) combo
        try:
            peptide_shifts = entry.get_peptide_shifts()
        except Exception:
            n_extract_fail += 1
            continue

        if not peptide_shifts:
            n_extract_fail += 1
            continue

        # Use the first peptide
        key = next(iter(peptide_shifts))
        result = extract_shifts_for_lacs(entry, *key)
        if result is None:
            n_extract_fail += 1
            continue

        seq, seq_nums, obs_shifts = result

        # Run our LACS
        try:
            our_offsets = compute_lacs_offsets(seq, seq_nums, obs_shifts)
        except Exception as e:
            logger.debug("LACS failed for %s: %s", entry_id, e)
            n_extract_fail += 1
            continue

        # Collect pairs where both have a value
        has_any = False
        for atom in atoms:
            bval = bmrb_offsets.get(atom)
            oval = our_offsets.get(atom)
            if bval is not None and oval is not None:
                # BMRB LACS uses the MATLAB convention: offset = -intercept
                # (negative = add to correct, i.e., obs is too low)
                # Our convention: offset = +intercept (positive = too high, subtract)
                # So BMRB_offset = -our_offset → negate BMRB for comparison
                bmrb_offsets_all[atom].append(-bval)
                our_offsets_all[atom].append(oval)
                has_any = True

        if has_any:
            entry_ids_compared.append(entry_id)
            n_compared += 1

        if n_compared % 100 == 0 and n_compared > 0:
            print(f"  ... compared {n_compared}/{args.max_entries} entries")

    # --- Report ---
    print(f"\n{'=' * 70}")
    print("LACS Comparison Report")
    print(f"{'=' * 70}")
    print(f"Entries with cached pickles:  {len(available_ids)}")
    print(f"Overlapping LACS reports:     {len(overlap_ids)}")
    print(f"LACS reports fetched:         {n_downloaded}")
    print(f"Parse/extract failures:       {n_parse_fail + n_extract_fail}")
    print(f"Entries compared:             {n_compared}")
    print()

    for atom in atoms:
        bmrb_arr = np.array(bmrb_offsets_all[atom])
        our_arr = np.array(our_offsets_all[atom])
        if len(bmrb_arr) == 0:
            print(f"{atom:>3}: no data")
            continue

        diff = our_arr - bmrb_arr
        abs_diff = np.abs(diff)

        # Correlation
        if len(bmrb_arr) > 2 and np.std(bmrb_arr) > 1e-6 and np.std(our_arr) > 1e-6:
            corr = np.corrcoef(bmrb_arr, our_arr)[0, 1]
        else:
            corr = float("nan")

        print(
            f"{atom:>3}:  n={len(bmrb_arr):>5}  "
            f"r={corr:+.4f}  "
            f"mean_diff={np.mean(diff):+.3f}  "
            f"median_|diff|={np.median(abs_diff):.3f}  "
            f"90th_|diff|={np.percentile(abs_diff, 90):.3f}  "
            f"max_|diff|={np.max(abs_diff):.3f}"
        )

    # Agreement breakdown
    print(f"\n{'=' * 70}")
    print("Agreement breakdown (|our - BMRB| thresholds)")
    print(f"{'=' * 70}")
    for threshold in [0.1, 0.25, 0.5, 1.0, 2.0]:
        print(f"\n  Within {threshold} ppm:")
        for atom in atoms:
            bmrb_arr = np.array(bmrb_offsets_all[atom])
            our_arr = np.array(our_offsets_all[atom])
            if len(bmrb_arr) == 0:
                continue
            within = np.sum(np.abs(our_arr - bmrb_arr) <= threshold)
            pct = 100 * within / len(bmrb_arr)
            print(f"    {atom:>3}: {within:>5}/{len(bmrb_arr)} ({pct:.1f}%)")

    # Save raw data for further analysis
    out_path = Path("data/lacs_comparison.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {"entry_ids": np.array(entry_ids_compared)}
    for atom in atoms:
        save_dict[f"bmrb_{atom}"] = np.array(bmrb_offsets_all[atom])
        save_dict[f"ours_{atom}"] = np.array(our_offsets_all[atom])
    np.savez(out_path, **save_dict)
    print(f"\nRaw data saved to {out_path}")


if __name__ == "__main__":
    main()
