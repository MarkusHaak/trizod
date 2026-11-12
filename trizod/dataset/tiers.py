"""Release tiers and the rule that picks the ``scores.json`` record a chain's
labels come from.

Shared verbatim by the deploy FASTA (:mod:`trizod.dataset.deploy_fasta`) and
the Parquet (``scripts/build_parquet_dataset.py``, its ``label_tier_source``
column and ``label_source`` metadata key), so the two artefacts agree chain for
chain (audit issue I-01).
"""

from __future__ import annotations

#: Release tiers, loosest -> strictest. Their ``scores.json`` files, training
#: representative sets and clustering pools all nest in this order.
TIER_ORDER = ("unfiltered", "tolerant", "moderate", "strict")

#: Which ``scores.json`` record a released chain's labels and offsets are read
#: from. Shared verbatim by the deploy FASTA and the Parquet (its
#: ``label_source`` metadata key), whose ``label_tier_source`` column names it.
LABEL_SOURCE_RULE = (
    "Every per-chain label (gscores, zscores, k, mask, n_scored), offset "
    "(off_*_sigma, lacs_off_*_ppm, total_off_*_ppm) and other scores.json field "
    "of a row is read from ONE record: that of the tier that selected the chain, "
    "named in label_tier_source. A training representative's record is its "
    "train_tier's (the strictest tier whose train_<tier>_best.fasta lists it); "
    "any other chain's is the strictest tier whose scores.json holds it. The "
    "per-tier records are NOT interchangeable: tolerant combines --max-offset 3 "
    "with --reject-shift-type-only, so it masks an over-limit atom type out of "
    "the score and nulls its off_<a>_sigma / total_off_<a>_ppm, where the "
    "unfiltered run applies that offset and scores the type; moderate and strict "
    "reject such a chain outright, so their records equal the unfiltered and "
    "tolerant ones for every chain they hold. A row therefore reproduces its "
    "label_tier_source tier's scores.json, not necessarily a looser tier's."
)


def strictest_tier(chain_id: str, membership) -> str | None:
    """The strictest tier whose ``membership[tier]`` contains ``chain_id``."""
    found = None
    for tier in TIER_ORDER:
        if chain_id in membership.get(tier, ()):
            found = tier
    return found


def label_source_tier(chain_id: str, reps_by_tier, scored_by_tier) -> str | None:
    """The tier whose ``scores.json`` record supplies ``chain_id``'s labels.

    ``train_tier`` (strictest tier listing it as a training representative) if
    it is one, else the strictest tier whose ``scores.json`` holds it; ``None``
    if no tier scores it. See :data:`LABEL_SOURCE_RULE`.

    Args:
        reps_by_tier: tier -> training-representative IDs.
        scored_by_tier: tier -> IDs (or ID-keyed records) in that tier's
            ``scores.json``.
    """
    return strictest_tier(chain_id, reps_by_tier) or strictest_tier(
        chain_id, scored_by_tier
    )
