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
#: perturbing cosolvent anywhere in the entry**. Those 96 include alpha-synuclein
#: and its disease mutants (6968, 16300, 16342, 17648 A30P, 17649 A53T, 17654
#: E46K), Tau (19112), gamma-synuclein (7244) and the yeast SNAREs Snc1/Sso1
#: (4286/4287) -- precisely the signal this dataset exists to capture. So these
#: four values are denied only when the entry independently evidences a
#: perturbing cosolvent (see ``has_cosolvent_evidence``); every other denied
#: value stands on the deposited tag alone, which is what still catches bmr5158.
PHYSICAL_STATE_AMBIGUOUS = frozenset(
    {
        "denatured",
        "partially denatured",
        "unfolded",
        "partially unfolded",
    }
)

#: Tokens that evidence a perturbing cosolvent, searched in the sample
#: components and in the sample-descriptive free text. Deliberately
#: tier-independent: this is evidence about what was in the tube, not a filter
#: policy. Matched on word boundaries so ``urea`` cannot fire on ``urease``.
#: Prefixes -- ``guanidin`` must reach ``guanidine`` and ``guanidinium``,
#: ``gdm`` must reach ``GdmCl``, ``hexafluoroisopropanol`` must reach the ``-d2``
#: spelling.
#:
#: "Cosolvent", not "denaturant", because only urea and the guanidinium salts
#: denature: TFE, HFIP and DMSO are helix inducers, so the two halves of this
#: list bias the score in OPPOSITE directions (see the naming note beside
#: ``COSOLVENT_TOKENS`` in ``trizod.trizod``). What every token here shares is
#: that the sample is no longer aqueous buffer.
#:
#: This list is NOT the tier filter's token list (``COSOLVENT_TOKENS`` in
#: ``trizod.trizod``); it answers a narrower question -- does the entry
#: independently evidence that the sample was not aqueous buffer, so that an
#: ambiguous deposited ``denatured``/``unfolded`` label can be believed? HFIP is
#: exactly that evidence and was missing. Adding it flips **zero** rows today
#: (no ambiguous-state row without other evidence names HFIP, measured over the
#: full corpus at all three tiers); it is forward cover, like ``sds``.
_COSOLVENT_PREFIX_TOKENS = (
    "guanidin",
    "gdm",
    "hexafluoroisopropanol",
    "hexafluoro-2-propanol",
)
#: Whole words -- ``urea`` must NOT fire on ``urease``, and ``hfip`` must not be
#: reachable from inside a longer token. ``hexafluoro`` is deliberately absent:
#: it would fire on bmr7375's copper(I) hexafluorophosphate. ``cd3oh`` /
#: ``cd3od`` / ``meoh`` are methanol spellings (bmr15323 'CD3OH' 100 %, bmr25162
#: 'CD3OD' 33 %, bmr5580 'MeOH' 37 %) and ``etoh`` the ethanol one (bmr51165);
#: whole words, so ``etoh`` does not fire inside 'acetohydroxamic acid'. The bare
#: alcohol names are in ``COSOLVENT_PATTERNS`` below because a word boundary
#: alone is not enough. The organic solvents of NMR-STAR audit I-07 follow:
#: ``cd3cn`` (bmr5633, 30 %) and ``acn`` (bmr36521, a pinned test chain, '30%
#: deuterated ACN'; whole word, so not NaCN or 'Propionibacterium acnes'),
#: ``chloroform`` / ``cdcl3`` / ``chcl3`` (not a chloroformate), and
#: ``isopropanol`` (not hexafluoroisopropanol, which is HFIP) / ``isopropyl
#: alcohol`` (bmr17810, 25 %).
_COSOLVENT_WORD_TOKENS = (
    "urea",
    "tfe",
    "trifluoroethanol",
    "hfip",
    "dmso",
    "sds",
    "gdn-hcl",
    "gdncl",
    "cd3oh",
    "cd3od",
    "meoh",
    "etoh",
    "cd3cn",
    "acn",
    "chloroform",
    "cdcl3",
    "chcl3",
    "isopropanol",
    "isopropyl alcohol",
)

#: Trifluoroethanol in every deposited spelling: ``Triflouroethanol`` (bmr50683,
#: 17.5 % v/v), ``trifluorethanol`` (bmr26679), ``trifluroethanol``,
#: ``trifluethanol`` (bmr5837, 30 %), ``Trifluoro Ethanol D2OH`` (bmr15559),
#: ``trifluoro-ethanol``, and the formula ``CF3CD2OH`` / ``CF3CH2OH`` (bmr4723,
#: bmr4724 and bmr15937, 50 %). Lower-case source: compile it IGNORECASE or
#: match it against lower-cased text.
TRIFLUOROETHANOL_PATTERN = (
    r"(?:trifl?(?:uo|ou|u|o)r?[\s-]*o?[\s-]*ethanol|cf3c[hd]2o[hd])"
)

#: Substituents that turn methanol/ethanol into a different compound when they
#: precede it as a separate word: 2-mercapto ethanol (bmr30782), B mercapto
#: ethanol (bmr27165), beta-mercapto-ethanol (bmr26700) are the reducing agent
#: at 2 mM / 0.1 %, and Trifluoro Ethanol is TFE. Spelled without the separator
#: (2-mercaptoethanol, ethanolamine) the word boundary already rejects them.
_ALCOHOL_SUBSTITUENTS = ("mercapto", "fluoro", "chloro", "bromo", "amino")


def alcohol_pattern(name):
    """Word-boundary pattern for a bare alcohol ``name`` that rejects substituted
    names: a substituent word just before it (``2-mercapto ethanol``) or a
    closing bracket (``(3-methylphenyl)methanol``, a ligand)."""
    guards = "".join(
        rf"(?<!{prefix}{sep})" for prefix in _ALCOHOL_SUBSTITUENTS for sep in (" ", "-")
    )
    return rf"(?<![a-z0-9)\]]){guards}{re.escape(name)}(?![a-z0-9])"


#: Acetonitrile (bmr18578, a CheZOD117 chain, 30 %; bmr50871/50900, pinned test
#: chains, 50 %) in every deposited spelling: ``acetonitril`` (bmr16452),
#: ``perdeuterated acetonitrile`` (bmr36457-36462), ``d3-acetonitrile``
#: (bmr25269) and ``2:1acetonitrile-d3`` (bmr52500) -- a digit may sit right
#: before it, so the left boundary is letters only. Not the ligand of bmr7375's
#: 1.8 mM Cu(I) source 'tetrakis(acetonitrile)copper(I) hexafluorophosphate':
#: a closing bracket glued to the next name rejects it.
ACETONITRILE_PATTERN = r"(?<![a-z)\]])acetonitrile?(?![a-z0-9])(?!\)[a-z0-9])"

#: Suffixes that turn an alkane name into a different compound when they follow
#: it as a separate word: the MPD crystallisation additive written
#: 'methyl-2,4-pentane diol' (bmr30088, 45 %) or 'Methyl Pentane diol' (bmr15156,
#: 16873), 'hexane-1,6-diol', a '-1-sulfonate'. Spelled without the separator
#: (2-methyl-2,4-pentanediol, 1,6-hexanediol) the word boundary rejects them.
_ALKANE_SUFFIX_GUARD = (
    r"(?![\s-]*(?:\d+(?:,\d+)*-)?"
    r"(?:(?:di|tri)?ol|(?:di)?one|thiol|amine|sulfon\w*|carbox\w*|(?:di)?oic)"
    r"(?![a-z0-9]))"
)


def alkane_pattern(name):
    """Word-boundary pattern for an alkane solvent ``name`` that rejects DSS --
    2,2-dimethyl-2-SILAPENTANE-5-sulfonate, the shift reference, whose glued
    spelling the boundary already rejects and whose ``sila-pentane`` /
    ``sila pentane`` spellings the guard does -- and substituted names (a diol,
    a sulfonate, see ``_ALKANE_SUFFIX_GUARD``)."""
    return (
        rf"(?<![a-z0-9)\]])(?<!sila-)(?<!sila ){re.escape(name)}(?![a-z0-9])"
        + _ALKANE_SUFFIX_GUARD
    )


#: Dioxane counts only when written as a solvent fraction of at least 1 %:
#: bmr4411 ('8% dioxane-d8') and bmr25887 ('5% dioxane-d6'). At 1-13 mM it is the
#: 1H/13C shift reference -- bmr4831 ("a small amount of dioxane for 1H chemical
#: shift referencing"), bmr15447 and bmr15973 (1 mM), bmr15565 (13 mM) -- and a
#: component name carries no concentration, so a bare 'dioxane' never matches.
#: The percentage comes first ('5 % v/v 1,4-dioxane') or right after the name
#: ('dioxane-d8 (10 %)').
_AT_LEAST_ONE_PERCENT = r"(?<![\d.])[1-9]\d*(?:\.\d+)?\s*%"
DIOXANE_PATTERN = (
    rf"{_AT_LEAST_ONE_PERCENT}\s*(?:\(?v/v\)?\s*)?(?:\[[^\]]*\]\s*)?"
    r"(?:(?:per)?deuterated\s+)?\S*?(?<![a-z0-9])dioxane(?![a-z0-9])"
    r"|(?<![a-z0-9])dioxane(?:-d\d)?\s*[(\[]?\s*[1-9]\d*(?:\.\d+)?\s*%"
)


#: Tokens whose match rule is a pattern rather than a plain substring or word
#: boundary. Shared with the tier filter (``trizod.trizod.COSOLVENT_PATTERN_RES``)
#: so the two lists cannot disagree on what counts as methanol or TFE.
COSOLVENT_PATTERNS = {
    "trifluoroethanol": TRIFLUOROETHANOL_PATTERN,
    "methanol": alcohol_pattern("methanol"),
    "ethanol": alcohol_pattern("ethanol"),
    # NMR-STAR audit I-07: the organic solvents that still passed every tier.
    "acetonitrile": ACETONITRILE_PATTERN,
    # bmr19216 'CDCl3/2-propanol'; the substituent guard keeps
    # 1,1,1,3,3,3-hexafluoro-2-propanol (HFIP) and 3-mercapto-1-propanol out
    "2-propanol": alcohol_pattern("2-propanol"),
    "1-propanol": alcohol_pattern("1-propanol"),
    "n-propanol": alcohol_pattern("n-propanol"),
    # bmr25640 (pentane reverse micelle), bmr5387 ('d-pentane', 98 %)
    "pentane": alkane_pattern("pentane"),
    "hexane": alkane_pattern("hexane"),
    "dioxane": DIOXANE_PATTERN,
}

COSOLVENT_EVIDENCE_TOKENS = (
    _COSOLVENT_PREFIX_TOKENS + _COSOLVENT_WORD_TOKENS + tuple(COSOLVENT_PATTERNS)
)

# Lookarounds rather than `\b`, matching the convention of the sibling matcher
# `trizod.trizod._word_boundary_pattern`: `_` is a word character to `\b`, so
# `\burea\b` does not match the deposited `Urea_8M` and `\bdmso\b` misses the
# `DMSO_d6` / `SDS_d25` spellings that really occur in
# _Sample_component.Mol_common_name -- while `-` must stay a boundary so that
# `urea-d4` and `HFIP-d2` still match. Kept a separate compiled alternation
# rather than shared with trizod.trizod: that module already imports from this
# one, and this pattern is IGNORECASE over raw text where the other is
# per-token over already-lower-cased fields.
_COSOLVENT_EVIDENCE_RE = re.compile(
    "|".join(
        [rf"(?<![a-z0-9]){re.escape(token)}" for token in _COSOLVENT_PREFIX_TOKENS]
        + [
            rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])"
            for token in _COSOLVENT_WORD_TOKENS
        ]
        + list(COSOLVENT_PATTERNS.values())
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


def has_cosolvent_evidence(texts):
    """Does any of ``texts`` name a perturbing cosolvent?

    ``texts`` are sample-descriptive strings: ``_Sample_component.Mol_common_name``
    values, ``_Sample.Solvent_system`` and the entry/assembly/entity/sample free
    text. Word-boundary matched, so ``urea`` does not fire on ``urease`` and
    ``ethanol`` not on 2-mercaptoethanol.
    """
    return any(
        text and _COSOLVENT_EVIDENCE_RE.search(str(text)) is not None for text in texts
    )


def denied_physical_states(values, deny_list, corroborated=None):
    """Which of ``values`` are exact (case-insensitive, stripped) deny-list hits.

    ``corroborated`` is a parallel sequence of booleans saying whether the entry
    independently evidences a perturbing cosolvent. The values in
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
