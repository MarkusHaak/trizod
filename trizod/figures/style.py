"""Shared matplotlib style + data-loading helpers for ``trizod.figures``.

Centralises the per-tier colour map, the stringency-tier ordering, and the
baseline-JSON loaders that the figure modules share, so a single source defines
how the manuscript figures classify and colour entries.
"""

from __future__ import annotations

import json
from pathlib import Path

# Stringency tiers, most-stringent first. The ordering is load-bearing:
# the tiers are nested (strict ⊂ moderate ⊂ tolerant ⊂ unfiltered), so
# ``classify_tier`` returns the FIRST (most stringent) tier an entry is in.
TIERS = ["strict", "moderate", "tolerant", "unfiltered"]

TIER_COLORS = {
    "strict": "#2ca02c",
    "moderate": "#1f77b4",
    "tolerant": "#ff7f0e",
    "unfiltered": "#d62728",
}

# Standard translucent white text box used for in-axes stat annotations.
TEXT_BBOX = {"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85}


def repo_root() -> Path:
    """Repository root, resolved from this file's location in the source tree.

    Used to build default input/output paths when a figure module is run as a
    standalone script from an editable install.
    """
    return Path(__file__).resolve().parents[2]


def load_tier_sets(
    baseline_dir, tiers: list[str] = TIERS, verbose: bool = False
) -> dict[str, set[str]]:
    """Load per-tier entry-ID sets from ``<baseline_dir>/<tier>.json`` (NDJSON).

    Missing tier files yield an empty set rather than raising, so a partial
    baseline directory still classifies what it can.
    """
    baseline_dir = Path(baseline_dir)
    out: dict[str, set[str]] = {}
    for tier in tiers:
        ids: set[str] = set()
        path = baseline_dir / f"{tier}.json"
        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        ids.add(json.loads(line)["entryID"])
        out[tier] = ids
        if verbose:
            print(f"  {tier}: {len(ids)} entries")
    return out


def classify_tier(entry_id, tier_sets, tiers: list[str] = TIERS) -> str:
    """Return the most stringent tier ``entry_id`` belongs to (``unfiltered`` if none)."""
    for tier in tiers:
        if entry_id in tier_sets.get(tier, set()):
            return tier
    return "unfiltered"
