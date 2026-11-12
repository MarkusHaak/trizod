#!/usr/bin/env python3
"""Regenerate the full-dataset regression baselines and stamp their provenance.

Usage:
    uv run python scripts/regenerate_baselines.py
    uv run python scripts/regenerate_baselines.py --tiers strict moderate

Writes ``<baseline-dir>/<tier>.json`` for each stringency tier plus a
``PROVENANCE.json`` recording the scoring-code version and git revision that
produced them. ``tests/test_full_dataset_regression.py`` reads that stamp and
fails with a pointer back here when the baselines no longer match the code.

This exists because the baselines are large, uncommitted, hand-regenerated
artifacts read only by a test that is deselected by default: without a
regeneration entry point and a version stamp they rot silently, which is
exactly what happened between March and August 2026 (issue #22 follow-up).

Expect this to take a while on a cold cache -- it scores every BMRB entry once
per tier. The tiers share ``tmp/wSCS/``, so the first tier does most of the
work and later tiers are largely cache hits.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trizod import paths  # noqa: E402
from trizod.provenance import write_baseline_provenance  # noqa: E402

TIERS = ["unfiltered", "tolerant", "moderate", "strict"]


def score_tier(tier, input_dir, baseline_dir, cache_dir):
    """Score one tier into ``baseline_dir/<tier>.json``; return the record count."""
    prefix = baseline_dir / tier
    cmd = [
        sys.executable,
        "-m",
        "trizod.trizod",
        "--input-dir",
        str(input_dir),
        "--filter-defaults",
        tier,
        "--output-prefix",
        str(prefix),
        "--output-format",
        "json",
        "--no-progress",
        "--cache-dir",
        str(cache_dir),
    ]
    print(f"\n=== {tier} ===\n{' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"scoring failed for tier {tier} (exit {result.returncode})")

    out = prefix.with_suffix(".json")
    with open(out) as fh:
        count = sum(1 for line in fh if line.strip())
    print(f"  {tier}: {count} records -> {out}", flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=paths.RAW_BMRB,
        help="Directory of raw BMRB entries",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=paths.INTERIM_BASELINE,
        help="Directory to write the baselines into",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=paths.TMP,
        help="Pipeline cache directory",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=TIERS,
        choices=TIERS,
        help="Stringency tiers to regenerate",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"no BMRB entries at {args.input_dir}")
    args.baseline_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        tier: score_tier(tier, args.input_dir, args.baseline_dir, args.cache_dir)
        for tier in args.tiers
    }

    # Carry forward counts for tiers this run did not touch, so a partial
    # regeneration does not silently drop them from the stamp.
    existing = args.baseline_dir / "PROVENANCE.json"
    if existing.exists():
        try:
            previous = json.loads(existing.read_text()).get("tiers", {})
        except ValueError:
            previous = {}
        for tier, count in previous.items():
            counts.setdefault(tier, count)

    path = write_baseline_provenance(args.baseline_dir, counts)
    print(f"\nwrote {path}")
    for tier, count in sorted(counts.items()):
        print(f"  {tier:<11} {count:>6} records")


if __name__ == "__main__":
    main()
