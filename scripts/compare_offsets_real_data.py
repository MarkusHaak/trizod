#!/usr/bin/env python3
"""Compare re-referencing offsets across methods on real BMRB data.

Compares LACS (our reimplementation), PANAV (from local JAR), BMRB LACS
(pre-computed), and TriZOD's existing offset correction on real entries.

Usage:
    uv run python scripts/compare_offsets_real_data.py --output docs/archive/260415/real-data-comparison.md
"""

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pynmrstar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trizod.lacs import compute_lacs_offsets

logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
logger = logging.getLogger("compare_offsets")

# Atom types that overlap across methods
# LACS: CA, CB, C, HA, H, N, HB (HB=HA)
# PANAV: CO, CA, CB, N
# BMRB LACS: CA, CB, CO, HA (varies)
COMMON_ATOMS = ["CA", "CB", "C", "N"]  # available in all three methods
ALL_ATOMS = ["CA", "CB", "C", "HA", "H", "N"]


def load_panav_offsets(panav_path):
    """Load PANAV offsets from JSON, return dict[entry_id] -> dict[atom] -> offset."""
    with open(panav_path) as f:
        raw = json.load(f)
    result = {}
    for eid, data in raw.items():
        if data is None:
            continue
        # Take first shift list
        sl = next(iter(data["shift_lists"].values()), None)
        if sl is None:
            continue
        offsets = sl["offsets"]
        # PANAV uses "CO" for carbonyl (we use "C") and opposite sign convention
        # (negative = too high). Flip to match our convention (positive = too high).
        mapped = {}
        for at, val in offsets.items():
            key = "C" if at == "CO" else at
            mapped[key] = -val
        result[eid] = mapped
    return result


def load_bmrb_lacs_offsets(lacs_dir):
    """Load BMRB pre-computed LACS offsets from NMR-STAR files."""
    result = {}
    lacs_path = Path(lacs_dir)
    if not lacs_path.exists():
        return result

    for f in sorted(lacs_path.glob("bmr*_LACS.str")):
        eid = f.name.split("_")[0][3:]  # bmr10002_LACS.str → 10002
        try:
            offsets = _parse_bmrb_lacs_file(f)
            if offsets:
                result[eid] = offsets
        except Exception as e:
            logger.debug(f"bmr{eid}: parse error: {e}")
    return result


def _parse_bmrb_lacs_file(filepath):
    """Parse a BMRB LACS NMR-STAR file to extract offsets."""
    offsets = {}
    with open(filepath) as f:
        content = f.read()

    for section_name, atom_key in [
        ("CACB_CA", "CA"),
        ("CACB_CB", "CB"),
        ("CACB_CO", "C"),
        ("CACB_HA", "HA"),
        ("N_N", "N"),
        ("N_HN", "H"),
    ]:
        marker = f"save_LACS_{section_name}_output"
        if marker not in content:
            continue
        # Find Y_axis_chem_shift_offset in this section
        start = content.index(marker)
        end = (
            content.index("save_", start + 1)
            if "save_" in content[start + 1 :]
            else len(content)
        )
        section = content[start:end]
        for line in section.split("\n"):
            if "_LACS_plot.Y_axis_chem_shift_offset" in line:
                val = line.strip().split()[-1]
                with contextlib.suppress(ValueError):
                    # BMRB LACS uses MATLAB convention (negative = too high),
                    # flip to match our convention (positive = too high)
                    offsets[atom_key] = -float(val)
    return offsets if offsets else None


def compute_our_lacs(entry_dir):
    """Run our LACS reimplementation on a BMRB entry."""
    str_files = sorted(entry_dir.glob("*.str"))
    if not str_files:
        return None

    try:
        entry = pynmrstar.Entry.from_file(str(str_files[0]))
    except Exception:
        return None

    # Extract sequence and shifts
    # Get entity with polymer sequence
    seq = None
    for sf in entry.get_saveframes_by_category("entity"):
        try:
            ptype = sf.get_tag("_Entity.Polymer_type")[0]
            if ptype == "polypeptide(L)":
                raw_seq = sf.get_tag("_Entity.Polymer_seq_one_letter_code")[0]
                seq = raw_seq.replace("\n", "").strip().upper()
                break
        except Exception:
            continue

    if not seq or len(seq) < 20:
        return None

    # Get chemical shifts
    loops = entry.get_loops_by_category("atom_chem_shift")
    if not loops:
        return None

    loop = loops[0]
    try:
        seq_ids = [int(x) for x in loop.get_tag("Seq_ID")]
        atom_ids = loop.get_tag("Atom_ID")
        vals = loop.get_tag("Val")
    except Exception:
        return None

    # Build obs_shifts dict
    n = len(seq)
    seq_nums = np.arange(1, n + 1)
    obs_shifts = {at: np.full(n, np.nan) for at in ["CA", "CB", "C", "HA", "H", "N"]}

    atom_map = {"C'": "C", "CO": "C"}
    for sid, aid, val in zip(seq_ids, atom_ids, vals):
        if sid < 1 or sid > n:
            continue
        try:
            v = float(val)
        except (ValueError, TypeError):
            continue
        mapped = atom_map.get(aid, aid)
        if mapped in obs_shifts:
            obs_shifts[mapped][sid - 1] = v

    result = compute_lacs_offsets(seq, seq_nums, obs_shifts)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare re-referencing offsets on real BMRB data"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/bmrb_entries"),
    )
    parser.add_argument(
        "--panav-file",
        type=Path,
        default=Path("data/external/panav_offsets.json"),
    )
    parser.add_argument(
        "--bmrb-lacs-dir",
        type=Path,
        default=Path("data/external/bmrb_lacs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="Limit number of entries to process (0 = all)",
    )
    args = parser.parse_args()

    # Load PANAV and BMRB LACS
    logger.info("Loading PANAV offsets...")
    panav = load_panav_offsets(args.panav_file)
    logger.info(f"PANAV: {len(panav)} entries with offsets")

    logger.info("Loading BMRB LACS offsets...")
    bmrb_lacs = load_bmrb_lacs_offsets(args.bmrb_lacs_dir)
    logger.info(f"BMRB LACS: {len(bmrb_lacs)} entries with offsets")

    # Find entries available in both PANAV and BMRB LACS
    common_ids = sorted(set(panav.keys()) & set(bmrb_lacs.keys()))
    logger.info(f"Entries in both PANAV and BMRB LACS: {len(common_ids)}")

    # Also run our LACS on a subset
    entry_dirs = {
        p.name[3:]: p
        for p in args.input_dir.iterdir()
        if p.is_dir() and p.name.startswith("bmr")
    }

    # Process entries that have both PANAV and BMRB LACS
    if args.max_entries > 0:
        common_ids = common_ids[: args.max_entries]

    logger.info(f"Running our LACS on {len(common_ids)} entries...")
    comparisons = []
    for i, eid in enumerate(common_ids):
        if (i + 1) % 500 == 0:
            logger.info(f"  [{i + 1}/{len(common_ids)}]")

        entry_dir = entry_dirs.get(eid)
        if entry_dir is None:
            continue

        our_lacs = compute_our_lacs(entry_dir)

        for at in ALL_ATOMS:
            panav_val = panav.get(eid, {}).get(at)
            bmrb_val = bmrb_lacs.get(eid, {}).get(at)
            our_val = our_lacs.get(at) if our_lacs else None

            comparisons.append(
                {
                    "entry_id": eid,
                    "atom": at,
                    "panav": panav_val,
                    "bmrb_lacs": bmrb_val,
                    "our_lacs": our_val,
                }
            )

    report = format_report(comparisons, len(common_ids))
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        logger.info(f"Report written to {args.output}")


def format_report(comparisons, n_entries):
    """Format comparison results as markdown."""
    lines = [
        "# Real-Data Offset Comparison: LACS vs PANAV vs BMRB LACS",
        "",
        f"Compared offsets on **{n_entries}** BMRB entries that have data from all three methods.",
        "",
    ]

    # --- Pairwise agreement ---
    pairs = [
        ("our_lacs", "bmrb_lacs", "Our LACS vs BMRB LACS"),
        ("our_lacs", "panav", "Our LACS vs PANAV"),
        ("bmrb_lacs", "panav", "BMRB LACS vs PANAV"),
    ]

    for key_a, key_b, title in pairs:
        lines.extend(["", f"## {title}", ""])
        lines.append("| Atom | N | Mean diff | MAE | Corr | Agree (<0.5) |")
        lines.append("| ---- | -: | --------: | --: | ---: | -----------: |")

        for at in ALL_ATOMS:
            at_data = [
                c
                for c in comparisons
                if c["atom"] == at and c[key_a] is not None and c[key_b] is not None
            ]
            if len(at_data) < 10:
                lines.append(f"| {at} | {len(at_data)} | — | — | — | — |")
                continue

            vals_a = np.array([c[key_a] for c in at_data])
            vals_b = np.array([c[key_b] for c in at_data])
            diffs = vals_a - vals_b
            n = len(diffs)
            mean_diff = np.mean(diffs)
            mae = np.mean(np.abs(diffs))
            corr = (
                np.corrcoef(vals_a, vals_b)[0, 1]
                if np.std(vals_a) > 0 and np.std(vals_b) > 0
                else 0
            )
            agree = np.mean(np.abs(diffs) < 0.5)

            lines.append(
                f"| {at} | {n} | {mean_diff:+.3f} | {mae:.3f} | {corr:.3f} | {agree:.1%} |"
            )

    # --- Distribution of offsets ---
    lines.extend(["", "## Offset magnitude distribution (our LACS)", ""])
    lines.append("| Atom | N | Mean | Median | Std | |off|>1 ppm | |off|>2 ppm |")
    lines.append("| ---- | -: | ---: | -----: | --: | ---------: | ---------: |")

    for at in ALL_ATOMS:
        at_data = [
            c for c in comparisons if c["atom"] == at and c["our_lacs"] is not None
        ]
        if not at_data:
            continue
        vals = np.array([c["our_lacs"] for c in at_data])
        n = len(vals)
        lines.append(
            f"| {at} | {n} | {np.mean(vals):+.3f} | {np.median(vals):+.3f} | "
            f"{np.std(vals):.3f} | {np.mean(np.abs(vals) > 1):.1%} | "
            f"{np.mean(np.abs(vals) > 2):.1%} |"
        )

    # --- Entries with large offsets ---
    lines.extend(
        ["", "## Entries with large offsets (|offset| > 2 ppm in any atom type)", ""]
    )
    large_entries = set()
    for c in comparisons:
        if c["our_lacs"] is not None and abs(c["our_lacs"]) > 2.0:
            large_entries.add(c["entry_id"])
    lines.append(
        f"**{len(large_entries)}** entries have at least one atom type with |offset| > 2 ppm."
    )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
