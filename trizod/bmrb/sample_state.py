"""State signals a depositor records about the sample, and how to read them.

Two independent axes, both parsed by :mod:`trizod.bmrb.bmrb` and neither used by
the pipeline before:

``_Entity_assembly.Physical_state``
    The depositor's own label for the *conformational* state of the studied
    molecule -- ``native``, ``denatured``, ``molten globule``,
    ``intrinsically disordered``, ... It is free text with a long tail (47
    distinct non-null values over 26,885 entity-assembly records), so it is
    matched **exactly** (case-insensitively, whitespace-stripped) against a
    per-tier deny list. Substring matching on free text is precisely what makes
    a keyword like ``unfold`` dangerous, and ``partially denatured`` /
    ``not denatured`` are why an exact match is the only defensible rule here.

``_Sample.Type`` / ``_Experiment.Sample_state`` / experiment names
    The *physical form* of the sample: is this solution NMR or solid-state NMR?
    24 % of BMRB entries never filled in ``_Entry.Experimental_method_subtype``,
    so this evidence is what lets the strict tier admit an undeclared entry
    without also admitting fibrils.

Neither vocabulary is closed. ``warn_unseen_physical_states`` exists so that a
new spelling shows up as a log line instead of silently ceasing to match.
"""

import logging
import re
from collections import Counter

import pandas as pd

# --------------------------------------------------------------------------- #
# _Entity_assembly.Physical_state
# --------------------------------------------------------------------------- #

#: Values that must NEVER be denied by any tier. ``intrinsically disordered``
#: (196 records) and ``partially disordered`` (52) are the signal the dataset
#: exists to capture; the rest are ordinary well-behaved depositions.
PHYSICAL_STATE_KEEP = frozenset(
    {
        "native",
        "intrinsically disordered",
        "partially disordered",
        "natively unstructured",
        "folded",
        "reduced",
        "recombinant",
        "synthetic",
        "mutant",
    }
)

#: Deposited values that carry no state information at all ("na", "Other", ...).
#: Listed only so they do not trip the unseen-vocabulary warning.
PHYSICAL_STATE_NEUTRAL = frozenset(
    {
        "na",
        "n.a.",
        "n/a",
        "other",
        "nonpolymer",
        "liquid",
        "trans",
        "pa",
        "metal-substituted",
        "non-modified",
        "de novo designed",
        "not naturally occurring",
        "native (refolded)",
        "folded in the model system",
    }
)

#: States whose deposited label does NOT by itself mean the sample is unusable.
#: Depositors write ``denatured``/``unfolded`` both for a chemically denatured
#: sample and for a natively unfolded IDP: bmr6968, titled "... of intrinsically
#: disordered alpha-synuclein", is deposited as ``denatured``. Measured over the
#: unfiltered tier -- the complete picture, since no deny list has run there --
#: 238 rows carry one of these four values and **96 of them evidence no
#: denaturant anywhere in the entry**. Those 96 include alpha-synuclein and its
#: disease mutants (6968, 16300, 16342, 17648 A30P, 17649 A53T, 17654 E46K),
#: Tau (19112), gamma-synuclein (7244) and the yeast SNAREs Snc1/Sso1
#: (4286/4287) -- precisely the signal this dataset exists to capture. So these
#: four values are denied only when the entry independently evidences a
#: denaturant (see ``has_denaturant_evidence``); every other denied value stands
#: on the deposited tag alone, which is what still catches bmr5158.
PHYSICAL_STATE_AMBIGUOUS = frozenset(
    {
        "denatured",
        "partially denatured",
        "unfolded",
        "partially unfolded",
    }
)

#: Tokens that evidence an actual chemical denaturant, searched in the sample
#: components and in the sample-descriptive free text. Deliberately
#: tier-independent: this is evidence about what was in the tube, not a filter
#: policy. Matched on word boundaries so ``urea`` cannot fire on ``urease``.
#: Prefixes -- ``guanidin`` must reach ``guanidine`` and ``guanidinium``,
#: ``gdm`` must reach ``GdmCl``.
_DENATURANT_PREFIX_TOKENS = ("guanidin", "gdm")
#: Whole words -- ``urea`` must NOT fire on ``urease``.
_DENATURANT_WORD_TOKENS = (
    "urea",
    "tfe",
    "trifluoroethanol",
    "dmso",
    "sds",
    "gdn-hcl",
    "gdncl",
)

DENATURANT_EVIDENCE_TOKENS = _DENATURANT_PREFIX_TOKENS + _DENATURANT_WORD_TOKENS

_DENATURANT_EVIDENCE_RE = re.compile(
    "|".join(
        [rf"\b{re.escape(token)}" for token in _DENATURANT_PREFIX_TOKENS]
        + [rf"\b{re.escape(token)}\b" for token in _DENATURANT_WORD_TOKENS]
    ),
    re.IGNORECASE,
)

#: Denied from the tolerant tier upwards. ``denatured``/``partially denatured``
#: are in ``PHYSICAL_STATE_AMBIGUOUS`` and therefore require corroboration;
#: ``fibril``/``fibrils``/``Fibrillar``/``amyloid fibril``/``amyloid fibrils``
#: are five separately deposited spellings of the same thing and do not.
PHYSICAL_STATE_TOLERANT_DENY = (
    "denatured",
    "partially denatured",
    "misfolded",
    "non-native",
    "aggregated",
    "amyloid",
    "amyloid fibril",
    "amyloid fibrils",
    "fibril",
    "fibrils",
    "fibrillar",
    "molten globule",
)

#: Adds the states that only a moderate-or-better tier should reject.
PHYSICAL_STATE_MODERATE_DENY = PHYSICAL_STATE_TOLERANT_DENY + (
    "unfolded",
    "partially unfolded",
    "folding intermediate",
    "intermediate",
)

#: Adds the bound / reconstituted states. ``Reconstituted`` and
#: ``Reconstituted in DPC`` are both deposited verbatim.
PHYSICAL_STATE_STRICT_DENY = PHYSICAL_STATE_MODERATE_DENY + (
    "bound",
    "micelle-bound",
    "slas micelle-bound",
    "reconstituted",
    "reconstituted in dpc",
)

#: Every value the deny lists, the keep set or the neutral set know about.
PHYSICAL_STATE_KNOWN = (
    PHYSICAL_STATE_KEEP | PHYSICAL_STATE_NEUTRAL | set(PHYSICAL_STATE_STRICT_DENY)
)


def normalise_physical_state(value):
    """Fold a deposited ``Physical_state`` to its comparison key, or ``None``.

    ``pd.isna`` rather than ``is None``: the column carries ``pd.NA`` for the
    6,500 entity assemblies that declare no state, and ``pd.NA is None`` is
    False -- ``str(pd.NA)`` would yield the string ``'<na>'``, which belongs to
    no vocabulary and so floods the unseen-value warning that exists to surface
    genuinely new spellings.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    value = str(value).strip().lower()
    return value or None


def has_denaturant_evidence(texts):
    """Does any of ``texts`` name a chemical denaturant?

    ``texts`` are sample-descriptive strings: ``_Sample_component.Mol_common_name``
    values plus the entry/assembly/entity/sample free text. Word-boundary matched,
    so ``urea`` does not fire on ``urease``.
    """
    return any(
        text and _DENATURANT_EVIDENCE_RE.search(str(text)) is not None for text in texts
    )


def denied_physical_states(values, deny_list, corroborated=None):
    """Which of ``values`` are exact (case-insensitive, stripped) deny-list hits.

    ``corroborated`` is a parallel sequence of booleans saying whether the entry
    independently evidences a denaturant. The values in
    ``PHYSICAL_STATE_AMBIGUOUS`` are denied only where it is true; every other
    denied value stands on the deposited tag alone. Omitting it means "no
    corroborating evidence", which keeps the ambiguous states.
    """
    denied = {normalise_physical_state(d) for d in deny_list} - {None}
    # Positional, not label-based: `corroborated` may arrive as a Series whose
    # index is not a RangeIndex, and `values` is iterated positionally.
    corroborated = None if corroborated is None else list(corroborated)
    out = []
    for i, value in enumerate(values):
        value = normalise_physical_state(value)
        if value not in denied:
            out.append(False)
        elif value in PHYSICAL_STATE_AMBIGUOUS:
            out.append(bool(corroborated is not None and corroborated[i]))
        else:
            out.append(True)
    return out


def is_physical_state_denied(value, deny_list, corroborated=False):
    """Exact (case-insensitive, stripped) membership of ``value`` in ``deny_list``."""
    return denied_physical_states([value], deny_list, corroborated=[corroborated])[0]


def unseen_physical_states(values, threshold=5):
    """Physical-state values that no vocabulary knows about, above ``threshold``.

    The vocabulary is depositor free text and grows; anything frequent enough to
    matter and unknown to us is reported so the exact match cannot silently stop
    matching a renamed state.
    """
    counts = Counter()
    for value in values:
        value = normalise_physical_state(value)
        if value is not None and value not in PHYSICAL_STATE_KNOWN:
            counts[value] += 1
    return {value: n for value, n in counts.items() if n >= threshold}


def warn_unseen_physical_states(values, threshold=5):
    """Log (and return) the unseen physical-state values above ``threshold``."""
    unseen = unseen_physical_states(values, threshold=threshold)
    if unseen:
        listing = ", ".join(
            f"{value!r} ({n})"
            for value, n in sorted(unseen.items(), key=lambda kv: -kv[1])
        )
        logging.getLogger("trizod").warning(
            f"unrecognised _Entity_assembly.Physical_state values (>= {threshold} rows): {listing}"
        )
    return unseen


def resolve_physical_state(assembly, entity_assemID, entityID):
    """``Physical_state`` of THIS row's entity assembly, or ``None``.

    An entry-wide OR over all entity assemblies over-filters by 10-28 rows per
    tier: a deposition that pairs a ``native`` chain with a ``denatured`` one
    describes two rows, not one.
    """
    for row in getattr(assembly, "entities", []):
        e_assem_ID, e_ID, _label, state = row[0], row[1], row[2], row[3]
        if e_assem_ID == entity_assemID and e_ID in ("", None, entityID):
            return normalise_physical_state(state)
    return None


# --------------------------------------------------------------------------- #
# solution vs solid-state evidence
# --------------------------------------------------------------------------- #

SOLUTION = "solution"
SOLID = "solid"
UNKNOWN = "unknown"

#: ``_Sample.Type`` substrings that mean isotropic solution NMR. Bicelles,
#: micelles and liquid crystals ARE solution NMR -- an anchored ``^solution$``
#: match wrongly rejects ``bicell_solution`` (5813) and ``micelles`` (6040).
_SOLUTION_SAMPLE_TYPES = (
    "solution",
    "soultion",  # deposited typo
    "micelle",
    "bicell",
    "bi-cell",
    "bi_cell",
    "liquid",
    "isotropic",
    "emulsion",
)

#: ``_Sample.Type`` substrings that mean solid-state NMR. Only consulted after
#: the solution vocabulary, so ``liquid crystal`` and ``gel solution`` do not
#: fall through to ``crystal`` / ``gel``.
_SOLID_SAMPLE_TYPES = (
    "solid",
    "crystal",
    "powder",
    "fibril",
    "fiber",
    "fibre",
    "fibrous",
    "filamentous",
    "phage",
    "lyophilized",
    "liposome",
    "membrane",
    "oriented",
    "sediment",
    "frozen",
    "amyloid",
    "magic angle",
    "gel",
)

#: Solid-state pulse sequences. Word-anchored so ``PAR``/``MAS``/``CP`` cannot
#: match inside an unrelated experiment name.
_SOLID_EXPERIMENTS = re.compile(
    r"(?<![a-z0-9])(?:darr|pdsd|pisema|redor|tedor|rfdr|chhc|nhhc|ncacx|ncocx|"
    r"cpmas|cp-mas|spinal|pain|par|mas)(?![a-z0-9])|magic[ -]angle"
)

#: Solution pulse sequences, i.e. the experiments that veto a solid verdict.
#: ``HSQC``/``HMQC`` and the proton-detected triple-resonance names ``hNCO`` /
#: ``hNCA`` are deliberately ABSENT: modern fast-MAS solid-state work runs all of
#: them (bmr27211 deposits "2D 1H-15N HSQC/HMQC", "NCO (hNCO)" and "3D hCANH" on
#: an all-``solid`` sample set), so they cannot be treated as solution evidence.
_SOLUTION_EXPERIMENTS = re.compile(
    r"(?<![a-z0-9])(?:noesy|tocsy|cosy|trosy|hncacb|cbca|cbcanh|hbha|hcch|hnha|"
    r"hnhb|dqf)(?![a-z0-9])"
)


def _sample_type_evidence(value):
    if not value:
        return None
    value = value.lower()
    if any(token in value for token in _SOLUTION_SAMPLE_TYPES):
        return SOLUTION
    if any(token in value for token in _SOLID_SAMPLE_TYPES):
        return SOLID
    return None


def _sample_state_evidence(value):
    if not value:
        return None
    value = value.lower()
    if "anisotropic" in value:
        return None  # aligned media: neither proves solution nor solid
    if "isotropic" in value or "isotoropic" in value or "isotripic" in value:
        return SOLUTION
    if "solution" in value or "bicelle" in value or "micelle" in value:
        return SOLUTION
    if "solid" in value or "solis" in value:  # 'solis' is a deposited typo
        return SOLID
    if "fibril" in value or "sediment" in value or "crystal" in value:
        return SOLID
    if "oriented" in value:
        return SOLID
    return None


def classify_sample_state(sample_types=(), sample_states=(), experiment_names=()):
    """Classify a deposition as ``solution``, ``solid`` or ``unknown``.

    The solid verdict is the *refined* veto: solid evidence **and** no
    solution-type experiment name. The naive veto (solid evidence alone) has 37.5
    % precision -- it kills real solution structures whose ``_Sample.Type`` is
    mis-typed, e.g. 34067 "Solution structure of the RBM5 OCRE domain".
    """
    type_evidence = {_sample_type_evidence(v) for v in sample_types}
    state_evidence = {_sample_state_evidence(v) for v in sample_states}
    names = [str(v).lower() for v in experiment_names if v]
    solid_experiment = any(_SOLID_EXPERIMENTS.search(name) for name in names)
    solution_experiment = any(_SOLUTION_EXPERIMENTS.search(name) for name in names)

    solid = SOLID in type_evidence or SOLID in state_evidence or solid_experiment
    solution = SOLUTION in type_evidence or SOLUTION in state_evidence

    if solid and not solution_experiment:
        return SOLID
    if solution or solution_experiment:
        return SOLUTION
    return UNKNOWN


def entry_sample_state(entry, sampleIDs=()):
    """``classify_sample_state`` over the samples/experiments an entry declares.

    ``sampleIDs`` narrows ``_Sample.Type`` to the samples this row's shift table
    actually references; the experiment list and its ``Sample_state`` values are
    entry-wide, as deposited.
    """
    samples = getattr(entry, "samples", None) or {}
    ids = [sID for sID in sampleIDs if sID in samples] or list(samples)
    # getattr: a pickled entry written before _Sample.Type was parsed still has
    # to classify (as `unknown`) rather than blow up the whole run.
    sample_types = [getattr(samples[sID], "type", None) for sID in ids]
    sample_states, experiment_names = [], []
    experiment_list = getattr(entry, "experiment_list", None)
    for row in getattr(experiment_list, "experiments", []) or []:
        experiment_names.append(row[1])
        if len(row) > 5:
            sample_states.append(row[5])
    for shift_table in (getattr(entry, "shift_tables", None) or {}).values():
        for row in getattr(shift_table, "experiments", []) or []:
            if len(row) > 3:
                sample_states.append(row[3])
    return classify_sample_state(sample_types, sample_states, experiment_names)
