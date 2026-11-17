"""Build the single-file Parquet release of the TriZOD dataset.

Consolidates the multi-file staged release bundle (per-tier FASTA + per-tier
``scores.json`` JSONL + ``clusters_best.tsv`` + test FASTAs) into ONE
self-contained Parquet table, one row per scored protein chain.

Design (see docs): the per-tier ``scores.json`` files nest (strict subset
moderate subset tolerant subset unfiltered), and so do the per-tier
training-representative sets and clustering pools, so a single row per chain
plus a handful of ordinal/categorical columns encodes every published view's
membership:

  * ``split``       one of train / redundant / test_chezod117 / test_trizod /
                    test_retired / excluded (test_retired: a pinned TriZOD
                    test chain retired for the reason in ``retired_reason``
                    (``release_info.RETIRED_REASONS``) -- not tested
                    on, still a leakage target, so never train or redundant)
  * ``train_tier``  strictest tier at which the chain is a training representative
  * ``pool_tier``   strictest tier whose redundancy-reduction pool contains the chain

``sequence_group`` / ``sequence_group_size`` link every chain, in every split,
to the chains with an identical sequence (:func:`sequence_group_id`, a hash of
the sequence alone, so stable across releases), for overlaying replicates.

A chain's per-tier *records* are NOT interchangeable, though (audit issue I-01).
Tolerant is the only tier that combines a finite ``--max-offset`` (3 sigma) with
``--reject-shift-type-only``: it masks an over-limit atom type out of the score
and nulls that type's ``off_<a>_sigma`` / ``total_off_<a>_ppm``, where the
unfiltered run applies the offset and scores the type -- 1,073 of the 15,073
tolerant records of release 2026-10 differ from the unfiltered run, 478 of them
training representatives. Moderate and strict reject such a chain outright, so
their records equal the unfiltered ones. Each row is therefore read, labels and
offsets alike, from ONE record: that of the tier that selected the chain, named
in ``label_tier_source`` -- its ``train_tier`` if it is a training
representative, otherwise the strictest tier whose ``scores.json`` holds it
(``LABEL_SOURCE_RULE`` in ``trizod.dataset.tiers``, shared with
``trizod.dataset.deploy_fasta`` so the deploy FASTA agrees chain for chain,
and published as the ``label_source`` metadata key).
``tests/test_release_label_consistency.py`` checks a built Parquet against the
bundle's per-tier files.

Per-residue labels are stored as equal-length list columns aligned 1:1 to the
sequence; unscored positions are NaN in the score columns and False in the
explicit boolean ``mask`` (no magic sentinel that could leak into a regression
loss).

The companion shift table (``--shifts-out``) is a second, flat long Parquet
holding every assigned chemical shift **on a canonical residue** of every
released chain — backbone and side chain (``bmrb.get_deposited_shifts``, which
drops the ~0.14 % of deposited values sitting on a non-canonical residue).
It joins on ``id`` with the main table,
ships ``val_ppm`` exactly as deposited, and adds ``val_corrected_ppm`` wherever
a trustworthy referencing offset transfers (``trizod.shifts.annotate_offsets``);
that column is NULL, never a copy of the raw value, where none does. Every C,
CA, CB and N row also carries the far-outlier flag ``mad_z`` / ``is_far_outlier``
against the frozen median/MAD reference (``trizod.shifts.outliers``).

The main table's physical layout (sort order, row groups) is ``SORT_KEYS`` /
``ROW_GROUP_SIZE``; ``sort_order_doc`` is its published explanation.

Every column is declared once, in ``COLUMNS`` below, together with the record it
is read from. ``tests/test_parquet_columns.py`` asserts that declaration against
the two upstream producers (``trizod.trizod.output_dataset`` and
``trizod.dataset.composition.detect_bound``), so a column added upstream fails a
test here instead of silently never reaching the deposit — which is exactly how
the C1b/C3/C4 annotations came to stop one step short of the Parquet.
``COLUMN_DOCS`` documents every column (meaning, null/NaN rule);
``--write-column-dictionary`` renders it to ``docs/dataset/columns.md`` and
``tests/test_column_dictionary.py`` fails when the two disagree.

The table metadata (:func:`table_metadata`) carries the release identity
(``trizod.dataset.release_info``, :func:`release_keys`): ``dataset_version``,
the data and code licences, ``code_commit`` (the commit that built the file),
``scoring_commit`` (the commit that scored the entries and wrote the package
bundle, from its ``pipeline_version`` or ``--scoring-commit``), the
``bmrb_snapshot`` (name, n_entries,
entry_list_sha256), the ``test_set`` name and counts, the ``cuts`` with their
rule and the k >= 10 note, and -- with ``--calibration`` -- the threshold
framework's ``calibration`` summary (prevalence, likelihood ratios at the cuts,
per-split prevalence, framework commit, results.json SHA-256). The shift
table gets the same release keys plus ``counts`` computed from its rows
(:class:`trizod.shifts.ShiftTableStats`); ``--restamp-shifts`` refreshes the
metadata of an existing shift table without rebuilding it.

Usage:
    uv run --with pyarrow python scripts/build_parquet_dataset.py \
        --in-bundle     <path to the package bundle, trizod-dataset-<version>/> \
        --out           <path to trizod_dataset.parquet> \
        [--composition-csv <path to _composition_cache.csv>] \
        [--shifts-out <path to trizod_shifts.parquet>] \
        [--calibration <threshold-framework output dir with results.json>] \
        [--scoring-commit <sha>]

    uv run --with pyarrow python scripts/build_parquet_dataset.py \
        --in-bundle <bundle> --restamp-shifts <trizod_shifts.parquet> \
        --pkl-dir <the build's tmp/bmrb_entries> --bmrb-dir <snapshot> \
        [--shifts-out <restamped copy>]

    uv run python scripts/build_parquet_dataset.py \
        --write-column-dictionary docs/dataset/columns.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from trizod import paths
from trizod.bmrb import xref_sources as xs
from trizod.bmrb import xrefs as xr
from trizod.bmrb.bmrb import (
    BB_EXTRA_ATOM_IDS,
    HIGH_PRESSURE_BAR,
    PD_TO_PH_STAR_OFFSET,
)
from trizod.constants import BACKBONE_ATOMS, REFINED_WEIGHTS
from trizod.dataset import chezod_overlap as co
from trizod.dataset import release_info as ri
from trizod.dataset.mmseqs import describe_rule
from trizod.dataset.paths import resolve_paths as resolve_dataset_paths
from trizod.dataset.testset import RETIRED_FASTA
from trizod.dataset.tiers import (
    LABEL_SOURCE_RULE,
    TIER_ORDER,
    label_source_tier,
    strictest_tier,
)
from trizod.offsets import (
    lacs_off_ppm_col,
    legacy_lacs_off_col,
    legacy_off_col,
    off_sigma_col,
    total_off_ppm_col,
)
from trizod.provenance import pipeline_version
from trizod.shifts import (
    PARQUET_METADATA,
    ShiftTableStats,
    annotate_mad,
    annotate_offsets,
    chain_offsets,
    iter_file_frames,
    iter_shift_frames,
    load_mad_reference,
    restamp_shift_parquet,
    write_shift_parquet,
)
from trizod.shifts.parquet import import_pyarrow

# Publication order of the per-atom offset columns. Deliberately NOT
# ``BACKBONE_ATOMS`` (whose order is the internal shift-array column order):
# these columns are published, so their order is frozen at what v0.3.0 shipped.
BACKBONE = ["C", "CA", "CB", "H", "HA", "HB", "N"]
assert set(BACKBONE) == set(BACKBONE_ATOMS)
# Three offset columns per atom, in three DIFFERENT units -- the names say which:
# off_<a>_sigma is a multiple of the per-atom POTENCI RMSD, lacs_off_<a>_ppm and
# total_off_<a>_ppm are ppm. total_off is the one to subtract from a deposited
# shift; adding the first two together is a unit error worth up to 21.2 ppm.
OFF_FIELDS = [off_sigma_col(a) for a in BACKBONE]
LACS_FIELDS = [lacs_off_ppm_col(a) for a in BACKBONE]
TOTAL_OFF_FIELDS = [total_off_ppm_col(a) for a in BACKBONE]

#: Table-level documentation of the 21 offset columns, mirroring the
#: ``correction_formula`` key of ``trizod_shifts.parquet``. The main table is
#: where the offsets actually ship, and it used to document them nowhere at all
#: while the companion — which carries none of them — explained them
#: exhaustively. Everything a consumer needs to convert is stated here,
#: REFINED_WEIGHTS included, so the conversion never has to be looked up.
OFFSET_COLUMNS_DOC = (
    "Three referencing-offset columns per backbone atom A in "
    + "/".join(BACKBONE)
    + ", in TWO DIFFERENT UNITS. "
    "off_A_sigma: the POTENCI/AIC residual offset, in multiples of the per-atom "
    "POTENCI RMSD REFINED_WEIGHTS[A] -- NOT ppm. --max-offset (3/3/2 per tier) "
    "is compared against this column, which is why it stays in sigma. "
    "lacs_off_A_ppm: the LACS offset, in ppm; it is subtracted straight off the "
    "deposited shift. total_off_A_ppm: derived, in ppm, and the ONLY one to "
    "use downstream -- total_off_A_ppm = lacs_off_A_ppm + off_A_sigma * "
    "REFINED_WEIGHTS[A], so that corrected_ppm = deposited_ppm - "
    "total_off_A_ppm reproduces the shift TriZOD scored. ADDING off_A_sigma "
    "AND lacs_off_A_ppm TOGETHER IS A UNIT ERROR worth up to 21.2 ppm. "
    "REFINED_WEIGHTS (ppm) = "
    + ", ".join(f"{a}:{REFINED_WEIGHTS[a]:g}" for a in BACKBONE)
    + ". ZERO AND NULL: 0.0 means no offset was applied for that atom -- the "
    "estimator returned zero (LACS needs enough observations and does not "
    "correct every atom), or the chain has no shift of that atom at all, in "
    "which case all three of its columns read 0.0 (nothing to correct, not a "
    "measured zero). lacs_off_A_ppm is never null for a scored chain. "
    "off_A_sigma and total_off_A_ppm are null together, with lacs_off_A_ppm "
    "still set, where the row's record (label_tier_source tolerant, which "
    "combines --max-offset with --reject-shift-type-only) rejected atom type A "
    "by --max-offset and masked it out of the score; see the label_source key. "
    "Per-value corrected shifts, including the side-chain transfer policy, are "
    "published in trizod_shifts.parquet (val_ppm / val_corrected_ppm / "
    "offset_applied_ppm / offset_source)."
)

# pyarrow is deliberately NOT a project dependency (nothing in the pipeline or
# the dataset chain needs it), so it is imported lazily — the column contract
# below stays importable, and testable, without it.
COMPOSITION_CACHE_NAME = "_composition_cache.csv"

#: Physical layout of the main table, requested by David Wagemann for his web
#: viewer, which reads the deposit over HTTP range requests: sorted rows in small
#: row groups with statistics, so a filter on ``entry_id`` reads one ~400 kB
#: group instead of the whole file. Explained to consumers by ``sort_order_doc``.
SORT_KEYS = [("entry_id", "ascending"), ("id", "ascending")]
ROW_GROUP_SIZE = 500


def straddling_entries(cols: dict, row_group_size: int) -> list[str]:
    """Entry IDs whose rows cross a row-group boundary of the sorted table
    (``SORT_KEYS``; Python's string order equals Arrow's byte order here)."""
    keys = sorted(zip(*(cols[name] for name, _ in SORT_KEYS)))
    return sorted(
        {
            keys[i][0]
            for i in range(row_group_size, len(keys), row_group_size)
            if keys[i][0] == keys[i - 1][0]
        }
    )


def sort_order_doc(row_group_size: int, straddling: list[str] | None = None) -> str:
    """The ``sort_order`` table metadata: the layout, in the consumer's terms.
    ``straddling`` (:func:`straddling_entries`) names the entries whose rows
    cross a row-group boundary; None when not computed."""
    keys = ",".join(name for name, _ in SORT_KEYS)
    first = SORT_KEYS[0][0]
    if straddling is None:
        groups = "one row group, or two where the entry crosses a group boundary"
    elif straddling:
        groups = (
            f"one row group, except the {len(straddling)} entries whose rows "
            f"cross a group boundary ({', '.join(straddling)}), which touch two"
        )
    else:
        groups = "one row group (no entry crosses a group boundary)"
    return (
        f"{keys} ascending, in string (byte) order: {first} is a string, so "
        f"'10001' sorts before '979'. Written in row groups of {row_group_size} "
        f"rows with min/max statistics on every column, so a filter on {first} "
        f"touches {groups}. id is sorted within each entry but is not "
        f"globally monotone ('1000_1_1_1' follows '10000_1_1_1'), so filter on "
        f"{first} when pruning matters."
    )


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_fasta_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                ids.add(line[1:].split()[0])
    return ids


def read_fasta_attrs(path: Path) -> dict[str, dict[str, str]]:
    """Parse the ``key=value`` tokens a FASTA header carries after the ID.

    ``trizod.dataset.testset`` writes ``>ID label_tier=... pinned_id=...``; a
    bundle staged before that existed simply has no tokens, which resolves to an
    empty attribute dict (and therefore a null column), never a made-up value.
    """
    attrs: dict[str, dict[str, str]] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            tokens = line[1:].split()
            if not tokens:
                continue
            attrs[tokens[0]] = dict(
                tok.split("=", 1) for tok in tokens[1:] if "=" in tok
            )
    return attrs


def read_composition_cache(path: Path) -> dict[str, dict[str, str]]:
    """``entryID -> raw composition row`` from ``_composition_cache.csv``.

    Written by ``trizod dataset build`` from
    :func:`trizod.dataset.composition.detect_bound`, keyed on the BMRB entry ID
    (not the chain ID), and covering every parsed entry — including the ones
    that never reach a tier.
    """
    rows: dict[str, dict[str, str]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            entry_id = clean_str(row.get("entryID"))
            if entry_id is not None:
                rows[entry_id] = row
    return rows


def read_cluster_map(path: Path) -> tuple[set[str], dict[str, str], dict[str, float]]:
    """Return (member set, member->best_repr, member->member_quality)."""
    members: set[str] = set()
    best: dict[str, str] = {}
    quality: dict[str, float] = {}
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            m = row["member"]
            members.add(m)
            best[m] = row["best_repr"]
            try:
                quality[m] = float(row["member_quality"])
            except (KeyError, ValueError, TypeError):
                quality[m] = None
    return members, best, quality


def as_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (ValueError, TypeError):
        return None
    return f if not math.isnan(f) else None


def as_int(value) -> int | None:
    f = as_float(value)
    return int(f) if f is not None else None


def as_high_pressure(value) -> bool | None:
    """``pressure_bar`` -> is it at or above the tiers' ``max-pressure-bar``
    (``HIGH_PRESSURE_BAR``)? Null when the pressure was not recorded."""
    bar = as_float(value)
    return None if bar is None else bar >= HIGH_PRESSURE_BAR


def clean_str(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


#: Spellings of "unset" that reach the composition cache as text: NMR-STAR nulls
#: and pandas nulls stringified by ``to_csv``. Never coerced to a value.
_NULL_TEXT = frozenset({"", "nan", "none", "<na>", "null", "."})


def as_bool(value) -> bool | None:
    """Truthy text (CSV) or a real bool (JSON) -> bool; unset -> None.

    Deliberately returns None rather than False for anything unrecognised: a
    silent False here would read downstream as "classified, and negative".
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in _NULL_TEXT:
        return None
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    return None


def as_str_list(value) -> list[str] | None:
    """``"CA;ZN"`` -> ``["CA", "ZN"]``; ``""`` -> ``[]``; unset -> None.

    ``detect_bound`` emits these three fields as ``";".join(sorted(...))``, so
    splitting on ``;`` recovers exactly what it encoded — a list column beats a
    delimiter-joined string for consumers, and the round trip is the classifier's
    own convention rather than a new one invented here. An entry with no ligands
    is an empty list (a measured absence); an entry with no composition record at
    all is null (unknown).
    """
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return []
    if s.lower() in _NULL_TEXT:
        return None
    return [part.strip() for part in s.split(";") if part.strip()]


# --------------------------------------------------------------------------- #
# column contract
# --------------------------------------------------------------------------- #

#: Arrow type per token. Tokens (not ``pa`` objects) so ``COLUMNS`` can be
#: imported without pyarrow installed.
ARROW_TYPES = {
    "string": lambda pa: pa.string(),
    "bool": lambda pa: pa.bool_(),
    "int32": lambda pa: pa.int32(),
    "float32": lambda pa: pa.float32(),
    "float64": lambda pa: pa.float64(),
    "list<bool>": lambda pa: pa.list_(pa.bool_()),
    "list<int32>": lambda pa: pa.list_(pa.int32()),
    "list<float32>": lambda pa: pa.list_(pa.float32()),
    "list<string>": lambda pa: pa.list_(pa.string()),
    # one UniProt/PDB link per element, fields from trizod.bmrb.xrefs.XREF_FIELDS
    "list<xref>": lambda pa: pa.list_(
        pa.struct([(n, ARROW_TYPES[t](pa)) for n, t in xr.XREF_FIELDS])
    ),
}

#: Where a column's value comes from. ``scores`` = one ``scores.json`` record,
#: ``composition`` = the ``_composition_cache.csv`` row for the chain's entry ID,
#: ``testset`` = the TriZOD test-set FASTA header, ``derived`` = computed from
#: the bundle's FASTAs / cluster TSVs / other columns.
ORIGINS = ("scores", "composition", "testset", "derived")


@dataclass(frozen=True)
class Column:
    """One Parquet column: its name, its Arrow type, and where its value is read.

    ``convert`` maps the raw value of ``source`` in the ``origin`` record;
    ``derive`` instead computes the value from the per-row context (and takes
    precedence). A column may set both ``source`` and ``derive`` — ``source``
    then only records which upstream key the column consumes, which is what the
    coverage tests check.
    """

    name: str
    arrow: str
    origin: str = "derived"
    source: str | None = None
    convert: Callable | None = None
    derive: Callable | None = None

    def __post_init__(self):
        assert self.arrow in ARROW_TYPES, f"unknown arrow type {self.arrow!r}"
        assert self.origin in ORIGINS, f"unknown origin {self.origin!r}"
        assert self.derive is not None or self.convert is not None, self.name
        assert self.derive is not None or self.source is not None, self.name


def _xref_col(ctx: dict, fn: Callable):
    """Apply ``fn`` to the chain's links, or null when the entry was not read."""
    links = ctx["xrefs"]
    return None if links is None else fn(links)


#: Table metadata explaining the cross-reference columns.
XREFS_DOC = (
    "xrefs: list of UniProt/PDB links per chain, each a struct "
    "(db, accession, source, identity, coverage, origin, raw_db, raw_accession, "
    "details). origin is where the link was read: an NMR-STAR loop of the "
    "snapshot (Entity_db_link, Assembly_db_link, Related_entries) or a dated "
    "external file (bmrb_pdb_match = BMRB's adit_nmr_matched_pdb_bmrb_entry_ids"
    ".csv; sifts = SIFTS pdb_chain_uniprot, see xref_sources). source: 'exact' = "
    "the PDB entry deposited together with this BMRB entry (_Related_entries "
    "BMRB Entry Tracking System, or the BMRB match file for BMRB IDs > 15000); "
    "'author' = depositor-supplied (_Entity_db_link / _Assembly_db_link "
    "Author_supplied=yes, other _Related_entries PDB rows); 'sifts' = UniProt "
    "accession of a chain of a high-confidence structure (exact, BMRB-matched "
    "or author PDB entry) whose sequence matches this chain -- identity / coverage are "
    "that sequence match; 'unspecified' = db link with no Author_supplied value "
    "(legacy entries), or a BMRB match-file row for BMRB IDs <= 15000 (author- "
    "or annotator-supplied); 'blast' = BMRB annotator BLAST hit "
    "(Author_supplied=no), which includes homologs -- check identity "
    "(Seq_identity, %) and coverage (Seq_query_to_submitted_percent, share of "
    "this chain aligned, %). _Entity_db_link and sifts links are per chain; "
    "_Related_entries, _Assembly_db_link and bmrb_pdb_match links are per entry "
    "and repeated on every chain of it. accession is the normalised accession "
    "(UniProt base accession without isoform/version, upper-case 4-character "
    "PDB ID) or null when the deposited value is not one (e.g. a UniProt entry "
    "name like B2MG_HUMAN; raw_accession keeps it). uniprot_ids: the chain's "
    "own UniProt accessions at the best evidence level it has -- the "
    "author-supplied ones if any, else the sifts ones, else every remaining "
    "accession with identity = 100 and coverage = 100 (one sequence in several "
    "species is listed unranked); uniprot_source names that tier (author / "
    "sifts / unspecified / blast). pdb_ids: distinct PDB IDs from 'exact' and "
    "'author' links only (other PDB links stay in xrefs). All four columns are "
    "null when the entry was not read and empty/null when it declares no link."
)


#: The same-sequence link (2026-11-17): ``seq-`` + the first
#: ``SEQUENCE_GROUP_HEX`` hex characters of the SHA-256 of the upper-case
#: sequence. A function of the sequence alone, so a sequence keeps its ID in
#: every release; changing either constant renames every group.
SEQUENCE_GROUP_PREFIX = "seq-"
SEQUENCE_GROUP_HEX = 16


def sequence_group_id(seq: str) -> str:
    """The ``sequence_group`` of a sequence (:data:`SEQUENCE_GROUP_PREFIX`)."""
    digest = hashlib.sha256(seq.upper().encode("utf-8")).hexdigest()
    return SEQUENCE_GROUP_PREFIX + digest[:SEQUENCE_GROUP_HEX]


def sequence_groups(
    seqs, group_id: Callable[[str], str] = sequence_group_id
) -> dict[str, tuple[str, int]]:
    """``sequence -> (sequence_group, sequence_group_size)`` for the chain
    sequences ``seqs`` of a release, one element per chain. Fails the build
    when two distinct sequences get the same ID."""
    counts = Counter(seqs)
    ids = {seq: group_id(seq) for seq in counts}
    by_id: dict[str, list[str]] = {}
    for seq, gid in ids.items():
        by_id.setdefault(gid, []).append(seq)
    clashes = {gid: members for gid, members in by_id.items() if len(members) > 1}
    if clashes:
        shown = {gid: [s[:30] for s in m] for gid, m in list(clashes.items())[:5]}
        raise SystemExit(
            f"sequence_group collision: {len(clashes)} IDs are shared by distinct "
            f"sequences ({shown}); lengthen SEQUENCE_GROUP_HEX"
        )
    return {seq: (ids[seq], n) for seq, n in counts.items()}


#: Table metadata explaining the same-sequence link.
SEQUENCE_GROUP_DOC = (
    "sequence_group: '"
    + SEQUENCE_GROUP_PREFIX
    + f"' + the first {SEQUENCE_GROUP_HEX} hex characters of the SHA-256 of the "
    "chain's sequence (upper-case, UTF-8), shared by every chain of the release "
    "whose sequence is identical, in every split. A function of the sequence "
    "alone, so a sequence keeps its ID across releases; the build fails if two "
    "distinct sequences ever get the same ID. sequence_group_size: the number of "
    "chains of this release, across every split, in the group (1 = no other "
    "chain has this sequence). Use: overlay replicate measurements of one "
    "sequence (repeat depositions, other conditions) with "
    "df.groupby('sequence_group'); their per-residue lists align position by "
    "position. The group spans train and test: check split before mixing them. "
    "Identity is exact, so other constructs of the same protein (a tag, a "
    "truncation, a point mutant) usually share uniprot_ids but not "
    "sequence_group; overlaying them needs a residue alignment, which the "
    "release does not provide. Never null"
)


def _scores(name, arrow, source, convert):
    return Column(name, arrow, origin="scores", source=source, convert=convert)


def _composition(name, arrow, convert):
    # source == name: the cache column is the classifier's own key
    return Column(name, arrow, origin="composition", source=name, convert=convert)


#: THE column contract. Order is the Parquet column order; the first 42 entries
#: are the published v0.3.0 layout and must not be reordered, renamed or
#: retyped — new columns are appended.
COLUMNS: list[Column] = [
    _scores("id", "string", "ID", clean_str),
    _scores("entry_id", "string", "entryID", clean_str),
    _scores("entity_id", "string", "entityID", clean_str),
    _scores("entity_assem_id", "string", "entity_assemID", clean_str),
    _scores("st_id", "string", "stID", clean_str),
    _scores("entity_name", "string", "entity_name", clean_str),
    Column("sequence", "string", "scores", "seq", derive=lambda c: c["seq"]),
    Column("length", "int32", derive=lambda c: len(c["seq"])),
    Column("gscores", "list<float32>", "scores", "gscores", derive=lambda c: c["g"]),
    Column("zscores", "list<float32>", "scores", "zscores", derive=lambda c: c["z"]),
    Column("k", "list<int32>", "scores", "k", derive=lambda c: c["k"]),
    Column("mask", "list<bool>", derive=lambda c: c["mask"]),
    Column("n_scored", "int32", derive=lambda c: sum(c["mask"])),
    Column("split", "string", derive=lambda c: c["split"]),
    Column("train_tier", "string", derive=lambda c: c["train_tier"]),
    Column("pool_tier", "string", derive=lambda c: c["pool_tier"]),
    Column("cluster_repr", "string", derive=lambda c: c["cluster_repr"]),
    Column("quality", "float64", derive=lambda c: c["quality"]),
    _scores("exp_method", "string", "exp_method", clean_str),
    _scores("exp_method_subtype", "string", "exp_method_subtype", clean_str),
    _scores("ph", "float32", "pH", as_float),
    _scores("temperature", "float32", "temperature", as_float),
    _scores("ionic_strength", "float32", "ionic_strength", as_float),
    _scores("total_bbshifts", "int32", "total_bbshifts", as_int),
    _scores("bbshift_positions_post", "int32", "bbshift_positions_post", as_int),
    _scores("bbshift_types_post", "int32", "bbshift_types_post", as_int),
    _scores("citation_title", "string", "citation_title", clean_str),
    _scores("citation_doi", "string", "citation_DOI", clean_str),
    *[_scores(f, "float32", f, as_float) for f in OFF_FIELDS],
    *[_scores(f, "float32", f, as_float) for f in LACS_FIELDS],
    # float64: total_off is what a consumer subtracts from a deposited ppm shift,
    # so it must not lose precision relative to the shifts themselves
    *[_scores(f, "float64", f, as_float) for f in TOTAL_OFF_FIELDS],
    # --- appended in v0.4.0 --------------------------------------------------
    # Sample-state annotations (plan §3.1/§3.2). Emitted at every tier; the
    # per-tier physical_state deny list and (since 2026-11) the moderate/strict
    # membrane-mimetic filter, through membrane_mimetic_sample below, read them.
    _scores("physical_state", "string", "physical_state", clean_str),
    _scores("cosolvent_evidence", "bool", "cosolvent_evidence", as_bool),
    _scores("sample_state_evidence", "string", "sample_state_evidence", clean_str),
    _scores("membrane_mimetic", "string", "membrane_mimetic", clean_str),
    # Composition labels (plan §3.3/§3.4), joined on the BMRB entry ID, so every
    # chain of an entry carries its entry's composition. ``n_entities`` keeps its
    # published meaning: distinct ``_Entity`` records, NOT the copy count — a
    # homodimer is one Entity referenced twice, which is what ``n_copies`` and
    # ``n_entity_assembly_rows`` describe.
    _composition("n_entities", "int32", as_int),
    _composition("has_non_polymer", "bool", as_bool),
    _composition("has_nucleic", "bool", as_bool),
    _composition("has_metal", "bool", as_bool),
    _composition("has_metal_ion", "bool", as_bool),
    _composition("has_metal_cofactor", "bool", as_bool),
    _composition("has_water", "bool", as_bool),
    _composition("has_other_ligand", "bool", as_bool),
    _composition("metal_comp_ids", "list<string>", as_str_list),
    _composition("ligand_comp_ids", "list<string>", as_str_list),
    _composition("ligand_names", "list<string>", as_str_list),
    _composition("multi_protein_assembly", "bool", as_bool),
    _composition("n_entity_assembly_rows", "int32", as_int),
    # NA, never 1, when the classifier errored on the entry: build.py's except
    # branch writes {"is_bound": True, "error": ...} and nothing else.
    _composition("n_copies", "int32", as_int),
    _composition("has_conformational_isomer", "bool", as_bool),
    _composition("is_bound", "bool", as_bool),
    # Strictest tier the test chain's sequence still satisfies. Only test_trizod
    # and test_retired rows have one; null everywhere else.
    Column(
        "label_tier", "string", origin="testset", source="label_tier", convert=clean_str
    ),
    # --- appended 2026-10 ----------------------------------------------------
    # Sample pressure in bar from the scored shift table's own condition list,
    # like ph/temperature. Since 2026-11 the tolerant, moderate and strict
    # tiers exclude >= 400 bar (high_pressure, appended below).
    _scores("pressure_bar", "float32", "pressure_bar", as_float),
    # --- appended 2026-11 ----------------------------------------------------
    # Tier whose scores.json record supplied every "scores" column of the row,
    # labels and offsets included (LABEL_SOURCE_RULE; audit I-01).
    Column("label_tier_source", "string", derive=lambda c: c["label_tier_source"]),
    # UniProt / PDB cross-references (2026-11-13), read from the NMR-STAR
    # snapshot by trizod.bmrb.xrefs and joined on (entry_id, entity_id). Null
    # when the entry was not read (no --bmrb-dir / missing file), an empty list
    # when it was and declares no link.
    Column("xrefs", "list<xref>", derive=lambda c: _xref_col(c, xr.xref_records)),
    Column(
        "uniprot_ids",
        "list<string>",
        derive=lambda c: None if c["uniprot"] is None else c["uniprot"][0],
    ),
    Column(
        "uniprot_source",
        "string",
        derive=lambda c: None if c["uniprot"] is None else c["uniprot"][1],
    ),
    Column(
        "pdb_ids", "list<string>", derive=lambda c: _xref_col(c, xr.confident_pdb_ids)
    ),
    # --- appended 2026-11: tier-rule provenance -------------------------------
    # One flag per condition rule, so a consumer sees why a chain sits in a
    # looser tier than its quality would suggest. Null, never False, when the
    # scores.json record predates the rule.
    # The class-rule verdict the moderate/strict membrane filter reads; the
    # descriptive membrane_mimetic string above names the classes found.
    _scores("membrane_mimetic_sample", "bool", "membrane_mimetic_sample", as_bool),
    # The deposited temperature was the 273 K placeholder and was read as not
    # deposited: temperature is then the 298 K default (strict excludes it).
    _scores("temperature_placeholder", "bool", "temperature_placeholder", as_bool),
    # pressure_bar >= 400 bar, the tolerant/moderate/strict ceiling.
    _scores("high_pressure", "bool", "pressure_bar", as_high_pressure),
    # --- appended 2026-11-17: release audit E23/E24 ---------------------------
    # Does the chain's sequence meet the leakage criterion (>= 30 % identity
    # over >= 80 % of either chain, >= 27 bits, either search direction)
    # against CheZOD117 (released + original sequences), CheZOD1325 or
    # CheZOD1174? Null when the overlap was not computed.
    *[
        Column(
            co.overlap_column(name),
            "bool",
            derive=lambda c, name=name: (
                None if c["chezod"] is None else c["id"] in c["chezod"][name]
            ),
        )
        for name in co.CHEZOD_SETS
    ],
    # Why a test_retired chain was retired (release_info.RETIRED_REASONS), from
    # the retired FASTA header; null elsewhere.
    Column(
        "retired_reason", "string", origin="testset", source="reason", convert=clean_str
    ),
    # --- appended 2026-11-17: the same-sequence link --------------------------
    # One ID per distinct sequence (sequence_group_id), shared by every chain of
    # the release whose sequence is identical, in every split, and the number of
    # such chains: replicate measurements a viewer can overlay residue by residue.
    Column("sequence_group", "string", derive=lambda c: c["sequence_group"][0]),
    Column("sequence_group_size", "int32", derive=lambda c: c["sequence_group"][1]),
]

#: ``scores.json`` keys carried into the Parquet, derived from ``COLUMNS`` so
#: the two can never disagree.
SCORES_JSON_SOURCES = frozenset(c.source for c in COLUMNS if c.origin == "scores")
#: ``detect_bound()`` keys carried into the Parquet.
COMPOSITION_SOURCES = frozenset(c.source for c in COLUMNS if c.origin == "composition")
#: ``scores.json`` keys deliberately NOT carried: ``--include-shifts`` adds one
#: column per backbone atom holding re-referenced shifts, while the deposit ships
#: every shift, raw and corrected, in ``trizod_shifts.parquet``. The release is
#: generated without that flag, so these keys are normally absent entirely.
SCORES_JSON_NOT_CARRIED = frozenset(BACKBONE + BB_EXTRA_ATOM_IDS)

# --------------------------------------------------------------------------- #
# column dictionary
# --------------------------------------------------------------------------- #

NEVER_NULL = "never null"
_PER_RESIDUE = "list never null, as long as `sequence`"
_POOL_NULL = "null when the chain is in no tier's redundancy-reduction pool"
_COMP_NULL = "null when the entry has no composition record"
_ATOM_NAMES = {
    "C": "carbonyl C",
    "CA": "Cα",
    "CB": "Cβ",
    "H": "amide H",
    "HA": "Hα",
    "HB": "Hβ",
    "N": "amide N",
}


@dataclass(frozen=True)
class ColumnDoc:
    """What a column means and when it is null (or NaN)."""

    nullable: bool
    meaning: str
    null_rule: str


def _doc(
    meaning: str, null_rule: str = NEVER_NULL, nullable: bool | None = None
) -> ColumnDoc:
    if nullable is None:
        nullable = null_rule != NEVER_NULL
    return ColumnDoc(nullable, meaning, null_rule)


def _offset_docs() -> dict[str, ColumnDoc]:
    docs = {}
    for a in BACKBONE:
        atom = f"{_ATOM_NAMES[a]} ({a})"
        no_shift = (
            f"0.0 where no offset was applied, including every chain without a "
            f"{a} shift (nothing to correct, not a measured zero)"
        )
        rejected = (
            f"null where the row's record (`label_tier_source` tolerant) rejected "
            f"atom type {a} by `--max-offset` and masked it out of the score; "
            f"`lacs_off_{a}_ppm` is still set there"
        )
        docs[off_sigma_col(a)] = _doc(
            f"POTENCI/AIC residual referencing offset of the {atom} shifts, in "
            f"multiples of the per-atom POTENCI RMSD REFINED_WEIGHTS[{a}] = "
            f"{REFINED_WEIGHTS[a]:g} ppm -- NOT ppm. `--max-offset` is compared "
            f"against it. {no_shift}",
            rejected,
        )
        docs[lacs_off_ppm_col(a)] = _doc(
            f"LACS referencing offset of the {atom} shifts, in ppm. 0.0 where "
            "LACS applied none: too few observations, an atom it does not "
            f"correct, or no {a} shift in the chain",
            "never null for a scored chain (null only if the build's scores.json "
            "record lacks the field)",
        )
        docs[total_off_ppm_col(a)] = _doc(
            f"Total referencing offset of the {atom} shifts in ppm: "
            f"`lacs_off_{a}_ppm + off_{a}_sigma × {REFINED_WEIGHTS[a]:g}`. "
            "Subtract it from a deposited shift to get the shift TriZOD scored "
            f"(never add the other two columns: different units). {no_shift}",
            f"null exactly where `off_{a}_sigma` is: {rejected}",
        )
    return docs


def _overlap_docs() -> dict[str, ColumnDoc]:
    what = {
        "chezod117": "CheZOD117 (the released chains plus the 117 original "
        "sequences SETH published)",
        "chezod1325": "CheZOD1325 (allseqs1325, ODiNPred's training set)",
        "chezod1174": "CheZOD1174 (the SETH and ADOPT training set)",
    }
    return {
        co.overlap_column(name): _doc(
            f"The chain meets the leakage criterion ({describe_rule('either')}) "
            f"against at least one sequence of {what[name]}. Benchmark a "
            "predictor trained on that set on the test chains where it is false",
            "null when the overlap was not computed",
        )
        for name in co.CHEZOD_SETS
    }


#: Meaning and null/NaN rule of every column in ``COLUMNS``, keyed by name.
#: ``tests/test_column_dictionary.py`` fails when a column lacks an entry, and
#: when ``docs/dataset/columns.md`` differs from :func:`column_dictionary_markdown`.
COLUMN_DOCS: dict[str, ColumnDoc] = {
    "id": _doc(
        "Chain ID `entryID_stID_entityAssemID_entityID`: BMRB entry, assigned "
        "chemical-shift list, entity assembly and entity. Unique per row; the "
        "join key of every view and of `trizod_shifts.parquet`"
    ),
    "entry_id": _doc("BMRB entry ID, a string (the sort key: '10001' < '979')"),
    "entity_id": _doc("BMRB `_Entity.ID` of the chain"),
    "entity_assem_id": _doc("BMRB `_Entity_assembly.ID` of the chain"),
    "st_id": _doc("BMRB `_Assigned_chem_shift_list.ID` the labels are scored from"),
    "entity_name": _doc("`_Entity.Name`", "null when the entry names no entity"),
    "sequence": _doc(
        "One-letter sequence as scored: the 20 standard residues plus X, U and "
        "Z (metadata `sequence_alphabet`). Every per-residue list is aligned "
        "1:1 to it"
    ),
    "length": _doc("Length of `sequence`"),
    "gscores": _doc(
        "Per-residue TriZOD G-score, 0–1, higher = more disordered: THE label. "
        "Cuts (metadata `cuts`): ordered G < 0.15, disordered G ≥ 0.35, "
        "confidently disordered G ≥ 0.70",
        f"{_PER_RESIDUE}; element NaN exactly where `mask` is false",
        nullable=False,
    ),
    "zscores": _doc(
        "Per-residue CheZOD Z-score (unbounded; grows with the shift count k)",
        f"{_PER_RESIDUE}; element NaN exactly where `mask` is false",
        nullable=False,
    ),
    "k": _doc(
        "Per-residue number of backbone chemical shifts behind the residue's "
        "score (its own and its neighbours'); 0 = none. k > 0 does not imply a "
        "score. k ≥ 10 is a conventional reliability cut (metadata `cuts`)",
        f"{_PER_RESIDUE}; elements never null (0 where no shift)",
        nullable=False,
    ),
    "mask": _doc(
        "Per-residue: true iff the residue is scored, i.e. its G (and Z) is "
        "finite. Where false, G and Z are NaN here and 999 in the UdonPred "
        "view and the deploy FASTA. Mask on this column, not on k",
        f"{_PER_RESIDUE}; elements never null",
        nullable=False,
    ),
    "n_scored": _doc("Number of scored residues, `sum(mask)`"),
    "split": _doc(
        "train / redundant / test_trizod / test_retired / test_chezod117 / "
        "excluded (metadata `split_values`, `test_retired_semantics`)"
    ),
    "train_tier": _doc(
        "Strictest tier at which the chain is a training representative. The "
        "training set of tier T is `split == 'train'` and `train_tier` at T or "
        "stricter (tiers nest: unfiltered ⊃ tolerant ⊃ moderate ⊃ strict)",
        "null unless `split == 'train'`",
    ),
    "pool_tier": _doc(
        "Strictest tier whose redundancy-reduction pool contains the chain", _POOL_NULL
    ),
    "cluster_repr": _doc(
        "Training representative heading the chain's 50 %/80 % mmseqs cluster",
        _POOL_NULL,
    ),
    "quality": _doc(
        "Quality score that picked the cluster representative: tier rank × 10⁶ "
        "+ backbone positions × backbone atom types − max |POTENCI residual|",
        _POOL_NULL,
    ),
    "exp_method": _doc(
        "`_Entry.Experimental_method`", "null when the entry does not declare it"
    ),
    "exp_method_subtype": _doc(
        "`_Entry.Experimental_method_subtype` (e.g. SOLUTION)",
        "null when the entry does not declare it (about a quarter of BMRB)",
    ),
    "ph": _doc(
        "Sample pH from the shift list's sample condition list. A pH* (meter) "
        "reading is used as is; a pD is converted to "
        f"pD − {PD_TO_PH_STAR_OFFSET:g} (`known_issues.tsv` lists the chains). "
        "7.0 imputed where none was deposited, except at strict, which "
        "requires it",
        "null when not deposited and not imputed, or outside the valid range",
    ),
    "temperature": _doc(
        "Sample temperature in K; 298 K imputed where none was deposited or a "
        "273 K placeholder was read as not deposited (`temperature_placeholder`)",
        "null when not deposited and not imputed",
    ),
    "ionic_strength": _doc(
        "Sample ionic strength in M as deposited; 0.1 M imputed where none was "
        "deposited, except at strict",
        "null when not deposited and not imputed, or invalid (negative)",
    ),
    "total_bbshifts": _doc(
        "Backbone shifts (H, HA, HB, C, CA, CB, N) deposited on the chain, "
        "terminal residues not counted",
        "null when the shifts could not be mapped onto the sequence",
    ),
    "bbshift_positions_post": _doc(
        "Residues with at least one backbone shift used in the score",
        "null when the chain was not scored",
    ),
    "bbshift_types_post": _doc(
        "Backbone atom types with shifts used in the score (at most 7)",
        "null when the chain was not scored",
    ),
    "citation_title": _doc(
        "Title of the entry's primary citation", "null when none deposited"
    ),
    "citation_doi": _doc(
        "DOI of the entry's primary citation", "null when none deposited"
    ),
    **_offset_docs(),
    "physical_state": _doc(
        "`_Entity_assembly.Physical_state` of the chain's assembly, lower-cased "
        "(native, intrinsically disordered, denatured, …); the tiers deny "
        "non-native states",
        "null when not deposited",
    ),
    "cosolvent_evidence": _doc(
        "Does the entry name a perturbing cosolvent (urea, guanidinium, TFE, "
        "HFIP, DMSO, methanol, ethanol, SDS, …) in its sample components, "
        "solvent system or sample text?",
        "null only when the scores.json record lacks the field",
    ),
    "sample_state_evidence": _doc(
        "solution / solid / unknown, from `_Sample.Type`, "
        "`_Experiment.Sample_state` and the experiment names",
        "null only when the scores.json record lacks the field",
    ),
    "membrane_mimetic": _doc(
        "';'-joined membrane-mimetic classes recognised in the chain's samples "
        "(phospholipid, bicelle, nanodisc, LPS, SDS, DPC, CHAPS, …), counted by "
        "the class rule or not, or set by a reviewed override (`unnamed "
        "mimetic`: an integral membrane protein whose deposition names no "
        "mimetic). Descriptive; the filter reads `membrane_mimetic_sample`",
        "null when none is recognised",
    ),
    "n_entities": _doc(
        "Distinct `_Entity` records of the entry (a homodimer is 1)", _COMP_NULL
    ),
    "has_non_polymer": _doc("The entry has a non-polymer entity", _COMP_NULL),
    "has_nucleic": _doc("The entry has a DNA/RNA entity", _COMP_NULL),
    "has_metal": _doc("The entry has a metal (ion or cofactor)", _COMP_NULL),
    "has_metal_ion": _doc("The entry has a free metal ion", _COMP_NULL),
    "has_metal_cofactor": _doc("The entry has a metal-containing cofactor", _COMP_NULL),
    "has_water": _doc(
        "The entry lists a water entity (not counted as a member)", _COMP_NULL
    ),
    "has_other_ligand": _doc("The entry has another non-polymer ligand", _COMP_NULL),
    "metal_comp_ids": _doc(
        "Chemical-component IDs of the metals (empty list: none)", _COMP_NULL
    ),
    "ligand_comp_ids": _doc(
        "Chemical-component IDs of the ligands (empty list: none)", _COMP_NULL
    ),
    "ligand_names": _doc("Names of the ligands (empty list: none)", _COMP_NULL),
    "multi_protein_assembly": _doc(
        "An assembly holds more than one distinct protein entity", _COMP_NULL
    ),
    "n_entity_assembly_rows": _doc(
        "`_Entity_assembly` rows of the chain's entity (raw copy count; "
        "inflated by alternative-conformer depositions)",
        _COMP_NULL,
    ),
    "n_copies": _doc(
        "Conservative lower bound on the copies of the entity in an assembly "
        "(metadata `n_copies_semantics`)",
        f"{_COMP_NULL}, or the classifier errored on it",
    ),
    "has_conformational_isomer": _doc(
        "Assembly rows name conformers (cis/trans, major/minor, …), so "
        "`n_entity_assembly_rows` over-counts copies",
        _COMP_NULL,
    ),
    "is_bound": _doc(
        "Bound-complex verdict of `trizod dataset build` (a second protein, a "
        "ligand, nucleic acid or metal in an assembly); bound entries are not "
        "in any training pool",
        _COMP_NULL,
    ),
    "label_tier": _doc(
        "Strictest tier whose pool still contains the pinned test sequence: "
        "strict for every test_trizod row; for a test_retired row any tier, "
        "strict where it was retired for another reason than not_strict",
        "null unless `split` is test_trizod or test_retired",
    ),
    "pressure_bar": _doc(
        "Sample pressure in bar from the shift list's sample condition list, "
        "converted from the deposited unit. A deposited '1 Pa' or '1.013 Pa' "
        "(almost certainly one atmosphere) is stored as deposited, about 1e-5 "
        "bar; '1013 bar' is read as hPa (one atmosphere). `known_issues.tsv` "
        "lists both (`pressure_placeholder_pa`, `pressure_hpa_as_bar`); no tier "
        "is affected",
        "null when not recorded or the unit is unknown (passes as ambient)",
    ),
    "label_tier_source": _doc(
        "Tier whose scores.json record supplied the row's labels and offsets "
        "(metadata `label_source`)"
    ),
    "xrefs": _doc(
        "Every UniProt / PDB link of the chain, a struct (db, accession, "
        "source, identity, coverage, origin, raw_db, raw_accession, details); "
        "metadata `xrefs_semantics`",
        "null when the entry was not read; empty when it declares no link",
    ),
    "uniprot_ids": _doc(
        "The chain's own UniProt accessions at its best evidence level "
        "(author, else sifts, else 100 %/100 % hits)",
        "null when the entry was not read; empty when there is none",
    ),
    "uniprot_source": _doc(
        "Evidence level of `uniprot_ids`: author / sifts / unspecified / blast",
        "null when `uniprot_ids` is empty or the entry was not read",
    ),
    "pdb_ids": _doc(
        "Distinct PDB IDs from `exact` and `author` links",
        "null when the entry was not read; empty when there is none",
    ),
    "membrane_mimetic_sample": _doc(
        "Class-rule verdict: the chain's sample is a membrane mimetic "
        "(metadata `membrane_mimetic_sample_semantics`); excluded from moderate "
        "and strict. Lipopolysaccharide (LPS) micelles and bilayers count as "
        "membrane mimetics. False also when no sample component could be read",
        "null only if the build's scores.json record lacks the field",
    ),
    "temperature_placeholder": _doc(
        "The deposited temperature was exactly 273 K with no stated 0 °C and "
        "was read as not deposited: the chain is scored at the 298 K default "
        "and strict excludes it",
        "null only if the build's scores.json record lacks the field",
    ),
    "high_pressure": _doc(
        "`pressure_bar >= 400`: the ceiling of the tolerant, moderate and strict tiers",
        "null when no pressure was recorded or its unit is missing or unknown "
        "(passes as ambient): filter with `high_pressure IS NOT TRUE`, not "
        "`== False`",
    ),
    **_overlap_docs(),
    "retired_reason": _doc(
        "Why a pinned test chain was retired: "
        + "; ".join(f"{k}: {v}" for k, v in ri.RETIRED_REASONS.items())
        + " (metadata `test_retired_semantics`)",
        "null unless `split == 'test_retired'`",
    ),
    "sequence_group": _doc(
        f"Same-sequence link: `{SEQUENCE_GROUP_PREFIX}` + the first "
        f"{SEQUENCE_GROUP_HEX} hex characters of the SHA-256 of the upper-case "
        "`sequence`, shared by every chain whose `sequence` is identical, in "
        "every split (`split` says which chain is test). A function of the "
        "sequence alone, so stable across releases. Overlay replicate "
        'measurements with `groupby("sequence_group")`: their per-residue lists '
        "align position by position. Other constructs of the same protein (tag, "
        "truncation, mutant) usually share `uniprot_ids` but not `sequence_group`; "
        "overlaying them needs a residue alignment, which the release does not "
        "provide (metadata `sequence_group_semantics`)"
    ),
    "sequence_group_size": _doc(
        "Number of chains of this release, across every split, in the chain's "
        "`sequence_group`; 1 = no other chain has this sequence"
    ),
}


def arrow_type_doc(token: str) -> str:
    """The Arrow type of a column token, spelled out (``list<xref>`` -> its
    struct)."""
    if token == "list<xref>":
        fields = ", ".join(f"{n}: {t}" for n, t in xr.XREF_FIELDS)
        return f"list<struct<{fields}>>"
    return token


def column_dictionary_markdown() -> str:
    """The column dictionary (``docs/dataset/columns.md``), from ``COLUMNS``
    and ``COLUMN_DOCS``. Every Arrow field is declared nullable in the schema;
    the table's ``Nullable`` column says whether the release can hold a null."""
    missing = [c.name for c in COLUMNS if c.name not in COLUMN_DOCS]
    extra = sorted(set(COLUMN_DOCS) - {c.name for c in COLUMNS})
    if missing or extra:
        raise ValueError(f"COLUMN_DOCS out of step: missing {missing}, extra {extra}")
    lines = [
        "# TriZOD dataset: column dictionary",
        "",
        "<!-- Generated by `scripts/build_parquet_dataset.py "
        "--write-column-dictionary docs/dataset/columns.md` from `COLUMNS` and "
        "`COLUMN_DOCS`. Do not edit by hand: "
        "tests/test_column_dictionary.py fails when it differs. -->",
        "",
        f"`trizod_dataset.parquet` has {len(COLUMNS)} columns, one row per "
        "scored protein chain, in this order. Per-residue columns are lists "
        "aligned 1:1 to `sequence`. A residue is scored where `mask` is true; "
        "`gscores` and `zscores` are NaN exactly where it is false (999 in "
        "the UdonPred view and the deploy FASTA). A shift "
        "count k > 0 does not imply a score. Null means unknown or not "
        "recorded, never a default. *Nullable* says whether a null can occur "
        "in the release (every Arrow field is declared nullable). The table "
        "metadata (`pyarrow.parquet.read_schema(path).metadata`) restates the "
        "semantics of the flag columns, the cuts and the provenance. "
        "*scores.json* below is the build-internal per-tier score file each "
        "row was read from (`label_tier_source`); it is not shipped.",
        "",
        "| # | Column | Arrow type | Nullable | Meaning | Null / NaN rule |",
        "|--:|---|---|---|---|---|",
    ]
    for i, col in enumerate(COLUMNS, 1):
        doc = COLUMN_DOCS[col.name]
        cells = [
            str(i),
            f"`{col.name}`",
            f"`{arrow_type_doc(col.arrow)}`",
            "yes" if doc.nullable else "no",
            doc.meaning,
            doc.null_rule,
        ]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(lines) + "\n"


#: Pre-rename offset keys. A ``scores.json`` staged before the offset columns
#: were renamed for unit safety spells them ``off_CA`` / ``lacs_off_CA``; those
#: files must still build, and must produce the same numbers.
LEGACY_OFFSET_KEYS = frozenset(
    [legacy_off_col(a) for a in BACKBONE] + [legacy_lacs_off_col(a) for a in BACKBONE]
)


def normalize_offset_keys(record: dict) -> dict:
    """Return ``record`` with the offset block in the current spelling.

    Reads whichever spelling is present and writes ``off_<a>_sigma`` /
    ``lacs_off_<a>_ppm`` / ``total_off_<a>_ppm``. The sigma->ppm conversion is
    never done here: :func:`trizod.shifts.chain_offsets` defers to
    :func:`trizod.offsets.total_offset_ppm`, the single place it is written.
    """
    if not LEGACY_OFFSET_KEYS & set(record):
        return record
    offsets = chain_offsets(record)
    out = {k: v for k, v in record.items() if k not in LEGACY_OFFSET_KEYS}
    for atom in BACKBONE:
        out[off_sigma_col(atom)] = offsets.sigma[atom]
        out[lacs_off_ppm_col(atom)] = offsets.lacs_ppm[atom]
        out[total_off_ppm_col(atom)] = offsets.total_ppm[atom]
    return out


def load_tier_scores(in_bundle: Path) -> dict[str, dict[str, dict]]:
    """``tier -> {chain ID: record}`` for all four tiers' ``scores.json``.

    All four are required: falling back to the unfiltered record for a missing
    tier would silently re-introduce I-01. Every tier must also nest inside the
    unfiltered superset, which is what the rows are enumerated from.
    """
    scores: dict[str, dict[str, dict]] = {}
    for tier in TIER_ORDER:
        path = in_bundle / "scores" / tier / "scores.json"
        if not path.is_file():
            raise SystemExit(
                f"missing {path}: every tier's scores.json is needed, because a "
                f"chain's labels come from the tier that selected it ({tier} "
                "records are not interchangeable with the unfiltered ones)"
            )
        scores[tier] = {
            rec["ID"]: normalize_offset_keys(rec) for rec in read_jsonl(path)
        }
    for tier in TIER_ORDER[1:]:
        stray = sorted(set(scores[tier]) - set(scores["unfiltered"]))
        if stray:
            raise SystemExit(
                f"{len(stray)} {tier} chains are not in the unfiltered "
                f"scores.json, which must be the superset: {stray[:10]}"
            )
    return scores


def read_train_reps(in_bundle: Path) -> dict[str, set[str]]:
    """``tier -> training-representative IDs`` from ``train_<tier>_best.fasta``."""
    return {
        t: read_fasta_ids(in_bundle / "train" / t / f"train_{t}_best.fasta")
        for t in TIER_ORDER
    }


def label_records(
    in_bundle: Path, reps: dict[str, set[str]] | None = None
) -> dict[str, tuple[str, dict]]:
    """``chain ID -> (label_tier_source, that tier's scores.json record)``.

    One entry per chain of the unfiltered superset, in its file order; the
    tier follows ``LABEL_SOURCE_RULE``. Both the main table and the shift table
    read a chain from here, so their labels and offsets are the same record.
    """
    scores = load_tier_scores(in_bundle)
    if reps is None:
        reps = read_train_reps(in_bundle)
    out: dict[str, tuple[str, dict]] = {}
    absent = []
    for rid in scores["unfiltered"]:
        tier = label_source_tier(rid, reps, scores)
        rec = scores[tier].get(rid)
        if rec is None:
            absent.append(f"{rid} ({tier})")
            continue
        out[rid] = (tier, rec)
    if absent:
        raise SystemExit(
            f"{len(absent)} training representatives are missing from their "
            f"train_tier's scores.json -- the bundle is inconsistent: {absent[:10]}"
        )
    return out


def arrow_schema(pa):
    """The Parquet schema, built from ``COLUMNS`` (needs an imported pyarrow)."""
    return pa.schema([(c.name, ARROW_TYPES[c.arrow](pa)) for c in COLUMNS])


def collect_columns(
    in_bundle: Path,
    composition_csv: Path | None = None,
    labelled: dict[str, tuple[str, dict]] | None = None,
    xrefs: dict[str, xr.EntryXrefs] | None = None,
    external: xs.ExternalSources | None = None,
    chezod_overlaps: dict[str, set[str]] | None = None,
) -> dict:
    """Assemble every ``COLUMNS`` entry into a name -> list-of-values dict.

    Pure Python, no pyarrow: the join/null behaviour is the part worth testing,
    and pyarrow is an optional extra. ``labelled`` is
    ``label_records(in_bundle)``, when already loaded. ``xrefs`` maps a BMRB
    entry ID to its parsed links (``trizod.bmrb.xrefs.load_xrefs``); None
    leaves every cross-reference column null. ``external`` adds the links of
    the dated sources (``trizod.bmrb.xref_sources``) to every chain whose
    entry was read. ``chezod_overlaps`` (``{set name: chain IDs}``, from
    ``trizod.dataset.chezod_overlap.chezod_overlaps``) fills the
    ``<set>_overlap`` flags; None leaves them null.
    """
    reps = read_train_reps(in_bundle)
    if labelled is None:
        labelled = label_records(in_bundle, reps)
    print(f"loaded {len(labelled)} scored chains (unfiltered = superset)")
    pool = {}
    best_repr_map: dict[str, str] = {}
    quality_map: dict[str, float] = {}
    for t in TIER_ORDER:
        members, best, quality = read_cluster_map(
            in_bundle / "train" / t / "clusters_best.tsv"
        )
        pool[t] = members
        # unfiltered pool covers every pooled chain (pools nest); use it as canonical
        if t == "unfiltered":
            best_repr_map, quality_map = best, quality

    test_chezod = read_fasta_ids(in_bundle / "test" / "CheZOD117_test_set.fasta")
    trizod_test_attrs = read_fasta_attrs(in_bundle / "test" / "TriZOD_test_set.fasta")
    test_trizod = set(trizod_test_attrs)
    # Retired TriZOD test chains (trizod.dataset.testset.RETIRED_FASTA): absent
    # from a bundle staged before they existed, which then has none.
    retired_fa = in_bundle / "test" / RETIRED_FASTA
    retired_attrs = read_fasta_attrs(retired_fa) if retired_fa.exists() else {}
    if not any(a.get("label_tier") for a in trizod_test_attrs.values()):
        print(
            "  WARNING: no label_tier= on the TriZOD test FASTA headers "
            "(bundle staged before trizod/dataset/testset.py emitted it) — "
            "the label_tier column will be null"
        )

    composition = {}
    if composition_csv is not None:
        composition = read_composition_cache(composition_csv)
        print(f"loaded composition for {len(composition)} BMRB entries")
        missing_cols = COMPOSITION_SOURCES - set(next(iter(composition.values()), {}))
        if missing_cols:
            print(
                f"  WARNING: {composition_csv} predates "
                f"{sorted(missing_cols)} — those columns will be null. "
                "Re-run `trizod dataset build` to refresh it."
            )
    else:
        print("  WARNING: --skip-composition — every composition column will be null")

    if xrefs is None:
        print("  WARNING: --skip-xrefs — every cross-reference column will be null")

    # every row of the release, every split: the group is the release's own
    groups = sequence_groups([rec["seq"] for _, rec in labelled.values()])

    cols: dict[str, list] = {c.name: [] for c in COLUMNS}
    n_comp_hits = 0

    for rid, (source_tier, rec) in labelled.items():
        seq = rec["seq"]
        raw_g = rec.get("gscores") or []
        raw_z = rec.get("zscores") or []
        raw_k = rec.get("k") or []
        mask = [g is not None for g in raw_g]

        train_tier = strictest_tier(rid, reps)
        pool_tier = strictest_tier(rid, pool)
        if rid in test_chezod:
            split = "test_chezod117"
        elif rid in test_trizod:
            split = "test_trizod"
        elif rid in retired_attrs:
            split = "test_retired"
        elif train_tier is not None:
            split = "train"
        elif pool_tier is not None:
            split = "redundant"
        else:
            split = "excluded"

        entry_id = clean_str(rec.get("entryID"))
        entry_xrefs = (xrefs or {}).get(entry_id)
        chain_links = None
        if entry_xrefs is not None:
            chain_links = entry_xrefs.for_entity(clean_str(rec.get("entityID")))
            if external is not None:
                chain_links = xr.rank_links(
                    chain_links + external.links(entry_id, seq, chain_links)
                )
        comp_row = composition.get(entry_id)
        n_comp_hits += comp_row is not None
        records = {
            "scores": rec,
            "composition": comp_row,
            "testset": trizod_test_attrs.get(rid) or retired_attrs.get(rid),
        }
        ctx = {
            "id": rid,
            "chezod": chezod_overlaps,
            "seq": seq,
            "g": [float(g) if g is not None else float("nan") for g in raw_g],
            "z": [float(z) if z is not None else float("nan") for z in raw_z],
            "k": [int(x) if x is not None else 0 for x in raw_k],
            "mask": mask,
            "split": split,
            "train_tier": train_tier,
            "pool_tier": pool_tier,
            "cluster_repr": best_repr_map.get(rid),
            "quality": quality_map.get(rid),
            "label_tier_source": source_tier,
            "xrefs": chain_links,
            "uniprot": None if chain_links is None else xr.uniprot_ids(chain_links),
            "sequence_group": groups[seq],
        }
        for col in COLUMNS:
            if col.derive is not None:
                cols[col.name].append(col.derive(ctx))
                continue
            # An absent upstream record yields null, never a default: "unknown"
            # and "measured negative" must stay distinguishable downstream.
            record = records[col.origin]
            cols[col.name].append(
                None if record is None else col.convert(record.get(col.source))
            )

    print(
        f"composition joined for {n_comp_hits}/{len(labelled)} chains; "
        f"label_tier for {sum(x is not None for x in cols['label_tier'])}"
    )
    print(
        "labels by selecting tier (label_tier_source):",
        dict(Counter(cols["label_tier_source"])),
    )
    if xrefs is not None:
        print_xref_coverage(cols)
    return cols


def xref_coverage(cols: dict) -> dict:
    """Share of chains with each kind of link (the release coverage summary)."""
    n = len(cols["id"])
    read = [x for x in cols["xrefs"] if x is not None]

    def has(pred):
        return sum(any(pred(link) for link in links) for links in read)

    def uni(link):
        return link["db"] == xr.UNIPROT and link["accession"] is not None

    def pdb(link):
        return link["db"] == xr.PDB and link["accession"] is not None

    return {
        "chains": n,
        "chains_read": len(read),
        "uniprot_author": has(lambda x: uni(x) and x["source"] == "author"),
        "uniprot_sifts": has(lambda x: uni(x) and x["source"] == "sifts"),
        "uniprot_any": has(uni),
        "uniprot_ids": sum(bool(v) for v in cols["uniprot_ids"] if v is not None),
        "uniprot_ids_multiple": sum(
            len(v) > 1 for v in cols["uniprot_ids"] if v is not None
        ),
        "uniprot_source": dict(
            Counter(v for v in cols["uniprot_source"] if v is not None)
        ),
        "pdb_exact": has(lambda x: pdb(x) and x["source"] == "exact"),
        "pdb_exact_or_author": sum(bool(v) for v in cols["pdb_ids"] if v is not None),
        "pdb_any": has(pdb),
        "any_link": has(lambda x: uni(x) or pdb(x)),
    }


def print_xref_coverage(cols: dict) -> None:
    cov = xref_coverage(cols)
    n = cov["chains"] or 1
    print(f"cross-references read for {cov['chains_read']}/{cov['chains']} chains")
    for key, value in cov.items():
        if isinstance(value, int) and key not in ("chains", "chains_read"):
            print(f"  {key:22s} {value:6d}  ({100 * value / n:5.1f} %)")
    print(f"  uniprot_ids by source: {cov['uniprot_source']}")


def _null_note(cols: dict, name: str) -> str:
    """How many rows of ``name`` are null in this build, in words."""
    n = sum(1 for v in cols[name] if v is None)
    if n == 0:
        return "Never null in this release"
    return f"Null on {n} rows, whose scores.json record lacks the field"


def release_keys(
    version: str = ri.DATASET_VERSION,
    snapshot: dict | None = None,
    commit: dict | None = None,
    scoring: dict | None = None,
) -> dict[str, str]:
    """The release identity both Parquet files carry: version, licences, the
    commit that wrote the file (``code_commit``), the commit that scored the
    entries and wrote the package bundle (``scoring_commit``) and the BMRB
    snapshot. ``commit`` defaults to this checkout's."""
    if commit is None:
        commit = ri.code_commit()
    return {
        "dataset_version": version,
        "license": ri.LICENSE_STATEMENT,
        "data_license": ri.DATA_LICENSE,
        "code_license": ri.CODE_LICENSE,
        "code_commit": json.dumps(
            {
                **(commit or {"commit": "unknown", "dirty": None}),
                "pipeline_version": pipeline_version(),
                "repository": ri.REPOSITORY,
                "scope": "built this file from the package bundle "
                "(scripts/build_parquet_dataset.py)",
            },
            sort_keys=True,
        ),
        "scoring_commit": (
            json.dumps(scoring, sort_keys=True)
            if scoring is not None
            else "not recorded: the package bundle states no pipeline_version"
        ),
        "bmrb_snapshot": (
            json.dumps(snapshot, sort_keys=True)
            if snapshot is not None
            else "not recorded: no BMRB_SNAPSHOT.json in the snapshot directory"
        ),
    }


def table_metadata(
    cols: dict,
    row_group_size: int = ROW_GROUP_SIZE,
    xrefs: dict[str, xr.EntryXrefs] | None = None,
    xrefs_snapshot: str | None = None,
    external: xs.ExternalSources | None = None,
    *,
    version: str = ri.DATASET_VERSION,
    snapshot: dict | None = None,
    calibration: dict | None = None,
    commit: dict | None = None,
    scoring: dict | None = None,
) -> dict[bytes, bytes]:
    """The Parquet key-value metadata for the collected ``cols``.

    Pure Python (no pyarrow), so every key is testable without the optional
    extra. ``snapshot`` is :func:`trizod.dataset.release_info.snapshot_block`
    of the BMRB snapshot, ``calibration`` :func:`~trizod.dataset.release_info.
    read_calibration` of a threshold-framework run, ``commit``
    :func:`~trizod.dataset.release_info.code_commit` of this checkout (default:
    looked up) and ``scoring`` :func:`~trizod.dataset.release_info.
    scoring_block` of the commit that scored the entries and wrote the bundle.
    A missing snapshot, calibration or scoring commit is recorded as such, never
    left out silently.
    """
    test_set = ri.testset_block(cols["split"], cols["retired_reason"])
    reasons = sorted({r for r in cols["retired_reason"] if r})
    metadata = {
        b"dataset": b"TriZOD",
        b"description": b"Per-residue protein-disorder labels from re-referenced BMRB NMR backbone chemical shifts. One row per scored chain; per-residue arrays aligned 1:1 to sequence.",
        b"tier_order_loose_to_strict": b"unfiltered,tolerant,moderate,strict",
        b"split_values": b"train,redundant,test_chezod117,test_trizod,test_retired,excluded",
        b"test_retired_semantics": ri.retired_semantics(reasons).encode(),
        b"mask_semantics": b"mask[i] is true iff residue i is scored, i.e. gscores[i] (and zscores[i]) is finite; they are NaN exactly where mask is false (999 in the UdonPred view and the deploy FASTA). A shift count k[i] > 0 does not imply a score: mask on mask, never on k",
        b"null_semantics": b"null means unknown/not recorded, never a default: an absent composition record leaves every composition column null; train_tier is null unless split is train; pool_tier, cluster_repr and quality are null for a chain in no tier's pool; label_tier is null for every row that is neither test_trizod nor test_retired, retired_reason for every row but test_retired, and the <set>_overlap flags when the overlap was not computed. Per-row rules: the column dictionary",
        b"sequence_alphabet": b"sequence holds the 20 standard residues plus X (unknown or non-standard; also an internal non-letter of the deposited sequence), U (selenocysteine) and Z (Glx). A trailing non-letter of the deposited sequence (a translation-stop '*', a typed apostrophe) is no residue and is not part of the chain; the per-residue arrays are aligned 1:1 to sequence",
        b"chezod_overlap_semantics": (
            "chezod117_overlap / chezod1325_overlap / chezod1174_overlap: the "
            "chain's sequence meets the leakage criterion against at least one "
            "sequence of CheZOD117 (the released CheZOD117 chains plus the 117 "
            "original sequences SETH published), CheZOD1325 (allseqs1325) or "
            "CheZOD1174 (the SETH training set): "
            + describe_rule("either")
            + " (mmseqs2, no prefilter). Training and redundant rows never "
            "overlap CheZOD117 (a leakage target); use the CheZOD1325/CheZOD1174 "
            "flags to benchmark a predictor trained on those sets on its clean "
            "subset. Null when not computed"
        ).encode(),
        b"composition_join": b"composition columns (n_entities .. is_bound) are per BMRB ENTRY, joined on entry_id, from trizod.dataset.composition.detect_bound; n_entities counts distinct _Entity records, so a homodimer is 1 -- see n_copies / n_entity_assembly_rows for stoichiometry",
        b"composition_list_columns": b"metal_comp_ids, ligand_comp_ids and ligand_names are list<string>; an empty list is a measured absence, null is an unclassified entry",
        b"n_copies_semantics": b"conservative lower bound on copies of one entity in an assembly (magnetic-equivalence groups, then physical state, then conformer naming); null where the classifier errored",
        b"membrane_mimetic_semantics": b"';'-joined membrane-mimetic classes (trizod.bmrb.membrane: phospholipid, bicelle, nanodisc, liposome, LPS (lipopolysaccharide micelles and bilayers), membrane, SDS, DPC, DHPC, LMPG, LDAO, CHAPS, DDM, ...; 'unnamed mimetic' only from a reviewed override: an integral membrane protein measured by solution NMR whose deposition names no mimetic) recognised in the samples the scored shift table references -- among their components (a non-polymer one by the whole class table, one linked to a polymer entity only by an explicit mimetic name), in _Sample.Type, or as explicit mimetic words in _Sample.Details / Solvent_system / Name -- or set by the reviewed override table trizod/dataset/data/membrane_overrides.tsv; counted or not (a sub-CMC CHAPS is named), null when none. Descriptive; the moderate/strict filter reads membrane_mimetic_sample",
        b"membrane_mimetic_sample_semantics": b"true when the chain's sample is a membrane mimetic by the class rule: a component that is a bilayer lipid or assembly (phospholipid, bicelle, nanodisc, liposome, lipopolysaccharide (LPS) micelles and bilayers) or a dedicated micelle former (SDS, alkylphosphocholines, lysophospholipids, short-chain phosphocholines, LDAO, lauroyl sarcosine, CnEm) at any concentration, or an additive-class detergent (CHAPS, CHAPSO, octyl glucoside/maltoside, DM, DDM, Triton X-100, Tween-20) at or above its CMC in water or with no convertible concentration (digitonin always), CnEm/n-alkanol alignment media not counted; or a sample whose _Sample.Type (micelle, bicelle, liposome, membrane, nanodisc; not reverse micelle) or free text (Details, Solvent_system, Name) names an explicit mimetic, unless it serves only anisotropic (RDC alignment) experiments; or an entry/chain listed, after review of its raw file, in trizod/dataset/data/membrane_overrides.tsv (evidence outside the fields the rule reads, a detergent outside the class table, or an integral membrane protein measured by solution NMR whose deposition names no mimetic: class 'unnamed mimetic'). Lipopolysaccharide (LPS) micelles and bilayers count as membrane mimetics. Excluded from the moderate and strict tiers. "
        + _null_note(cols, "membrane_mimetic_sample").encode(),
        b"temperature_placeholder_semantics": b"true when the deposited temperature was exactly 273 K (the text '273' or '273.0', unit K or none) with no statement of 0 C in the entry's own text, and was read as not deposited: temperature then holds the 298 K default and the strict tier excludes the chain. "
        + _null_note(cols, "temperature_placeholder").encode(),
        b"high_pressure_semantics": b"pressure_bar >= 400 bar, the ceiling of the tolerant, moderate and strict tiers; null when pressure_bar is null: no pressure was recorded, or it was deposited without a unit or in an unknown one (a bare '1' is not read as 1 bar). An unrecorded pressure passes as ambient, so filter with high_pressure IS NOT TRUE, not == False",
        b"label_tier_semantics": b"test_trizod and test_retired rows only: strictest tier whose pool still contains the pinned test sequence. strict for every test_trizod row; for a test_retired row any tier: strict where it was retired for another reason than not_strict (e.g. chezod117_overlap, within_test_duplicate), moderate or tolerant where it no longer meets strict",
        b"label_source": (
            LABEL_SOURCE_RULE + " (scores.json: the build-internal per-tier score "
            "file each row is read from; it is not shipped.)"
        ).encode(),
        b"pressure_bar_semantics": b"sample pressure in bar from the scored shift table's own sample condition list, converted from the deposited unit (bar, atm, Pa, kPa, MPa, hPa, mbar, mmHg, Torr, psi; 'ambient' = 1 atm; 900-1100 deposited as 'bar' is read as hPa, i.e. one atmosphere: known issue pressure_hpa_as_bar); a deposited '1 Pa' or '1.013 Pa' (almost certainly one atmosphere) is stored as deposited, about 1e-5 bar (known issue pressure_placeholder_pa; far below the cap, so no tier is affected); null when not recorded or the unit is missing/unknown. The tolerant, moderate and strict tiers exclude >= 400 bar (high_pressure); an unrecorded pressure passes as ambient",
        b"offset_columns": OFFSET_COLUMNS_DOC.encode(),
        b"sort_order": sort_order_doc(
            row_group_size, straddling_entries(cols, row_group_size)
        ).encode(),
        b"xrefs_semantics": XREFS_DOC.encode(),
        b"sequence_group_semantics": SEQUENCE_GROUP_DOC.encode(),
        b"xrefs_snapshot": (
            (xrefs_snapshot or "unknown") if xrefs is not None else "not read"
        ).encode(),
        b"xref_sources": (
            json.dumps(external.provenance(), sort_keys=True)
            if external is not None
            else "none: NMR-STAR snapshot only"
        ).encode(),
    }
    release = {
        **release_keys(version, snapshot, commit, scoring),
        "test_set": json.dumps(test_set, sort_keys=True),
        "cuts": json.dumps(ri.cut_block(), sort_keys=True),
        "cut_rule": ri.CUT_RULE,
        "k_reliability_note": ri.K_NOTE,
        "calibration": (
            json.dumps(calibration, sort_keys=True)
            if calibration is not None
            else "not recorded: built without --calibration"
        ),
        "column_dictionary": ri.COLUMN_DICTIONARY,
    }
    metadata.update({k.encode(): v.encode() for k, v in release.items()})
    return metadata


def build(
    in_bundle: Path,
    out_path: Path,
    composition_csv: Path | None = None,
    row_group_size: int = ROW_GROUP_SIZE,
    labelled: dict[str, tuple[str, dict]] | None = None,
    xrefs: dict[str, xr.EntryXrefs] | None = None,
    xrefs_snapshot: str | None = None,
    external: xs.ExternalSources | None = None,
    chezod_overlaps: dict[str, set[str]] | None = None,
    *,
    version: str = ri.DATASET_VERSION,
    snapshot: dict | None = None,
    calibration: dict | None = None,
    commit: dict | None = None,
    scoring: dict | None = None,
) -> None:
    if scoring is None:
        scoring = ri.scoring_block(ri.bundle_pipeline_version(in_bundle))
    cols = collect_columns(
        in_bundle,
        composition_csv,
        labelled,
        xrefs=xrefs,
        external=external,
        chezod_overlaps=chezod_overlaps,
    )

    pa, pq = import_pyarrow()
    schema = arrow_schema(pa)
    table = pa.table({name: cols[name] for name in schema.names}, schema=schema)
    table = table.sort_by(SORT_KEYS)

    metadata = table_metadata(
        cols,
        row_group_size,
        xrefs,
        xrefs_snapshot,
        external,
        version=version,
        snapshot=snapshot,
        calibration=calibration,
        commit=commit,
        scoring=scoring,
    )
    table = table.replace_schema_metadata(metadata)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        compression_level=19,
        row_group_size=row_group_size,
        write_statistics=True,
        sorting_columns=pq.SortingColumn.from_ordering(table.schema, SORT_KEYS),
    )

    # summary
    split_counts = Counter(cols["split"])
    tier_counts = Counter(t for t in cols["train_tier"] if t is not None)
    n_groups = math.ceil(table.num_rows / row_group_size)
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"rows: {table.num_rows}   columns: {table.num_columns}")
    sort_keys = ",".join(name for name, _ in SORT_KEYS)
    print(f"sorted by {sort_keys}; {n_groups} row groups of <= {row_group_size}")
    print("split counts:", dict(split_counts))
    print(
        "retired by reason:",
        dict(Counter(r for r in cols["retired_reason"] if r is not None)),
    )
    print("train_tier counts (strictest):", dict(tier_counts))
    print_sequence_groups(cols)


def print_sequence_groups(cols: dict, top: int = 5) -> None:
    """Groups, groups of two or more chains, and the largest groups."""
    size = dict(zip(cols["sequence_group"], cols["sequence_group_size"]))
    name = {}
    for gid, entity in zip(cols["sequence_group"], cols["entity_name"]):
        name.setdefault(gid, entity)
    multi = [g for g, n in size.items() if n > 1]
    print(
        f"sequence groups: {len(size)}, {len(multi)} with >= 2 chains "
        f"({sum(size[g] for g in multi)} chains)"
    )
    for gid in sorted(size, key=lambda g: (-size[g], g))[:top]:
        print(f"  {gid}  {size[gid]:4d} chains  e.g. {name[gid]}")


def shift_release_metadata(stats, release: dict[str, str]) -> dict[str, str]:
    """Every table-metadata key a release shift table gets on top of the
    writer's: the current static text (:data:`trizod.shifts.PARQUET_METADATA`,
    so a restamp also refreshes it), the release identity ``release``
    (:func:`release_keys`) and the counts of ``stats``."""
    static = {k.decode(): v.decode() for k, v in PARQUET_METADATA.items()}
    return {**static, **release, **stats.metadata()}


def build_shifts(
    in_bundle: Path,
    out_path: Path,
    pkl_dir: Path,
    labelled: dict[str, tuple[str, dict]] | None = None,
    release: dict[str, str] | None = None,
) -> None:
    """Write the chemical-shift table for the bundle's chains.

    Every assigned shift on a canonical residue of every released chain,
    backbone and side chain (shifts on non-canonical residues are excluded --
    see :func:`trizod.bmrb.bmrb.get_deposited_shifts`), with ``val_ppm`` as
    deposited and ``val_corrected_ppm`` beside it. Each chain's offsets come
    from the same record as its main-table row (:func:`label_records`, the
    ``label_tier_source`` tier), so an atom type the tolerant tier masked, and
    whose offset it therefore nulled, is not corrected here either: its rows
    get ``offset_source`` ``none`` and a NULL ``val_corrected_ppm``. Every
    chain of the main table is enumerated, so the two join on ``id`` with no
    orphans on either side. ``labelled`` is ``label_records(in_bundle)``, when
    already loaded.

    The table is streamed out with a fast codec while its counts accumulate
    (:class:`trizod.shifts.ShiftTableStats`), then restamped at zstd-19 with the
    release keys ``release`` (:func:`release_keys`; default: this checkout and
    the bundle's scoring commit) and the counts, so no number in its metadata
    is typed by hand.
    """
    if labelled is None:
        labelled = label_records(in_bundle)
    if release is None:
        release = release_keys(
            scoring=ri.scoring_block(ri.bundle_pipeline_version(in_bundle))
        )
    offsets = {rid: chain_offsets(rec) for rid, (_, rec) in labelled.items()}
    print(f"\nbuilding shift table for {len(offsets)} chains from {pkl_dir} ...")
    if not pkl_dir.is_dir():
        raise SystemExit(f"BMRB pickle cache not found: {pkl_dir}")
    # per-chain rejections (AA mismatch, unparsable Seq_ID) are expected in bulk
    # and would emit thousands of lines; the summary below counts them instead
    logging.getLogger("trizod.bmrb").setLevel(logging.CRITICAL)
    reference = load_mad_reference()
    missing = reference.missing_groups()
    if missing:
        raise SystemExit(
            f"far-outlier reference {reference.source!r} lacks {len(missing)} "
            f"residue x atom groups, e.g. {missing[:3]}: their shifts would be "
            "silently unassessed. Regenerate it with scripts/build_mad_reference.py."
        )
    print(f"far-outlier reference: {reference.source}")
    stats = ShiftTableStats()
    frames = (
        (cid, annotate_mad(annotate_offsets(df, offsets.get(cid)), reference))
        for cid, df in iter_shift_frames(list(offsets), pkl_dir, stats=stats)
    )
    path, n_rows = write_shift_parquet(
        frames, out_path, compression_level=1, mad_reference=reference, stats=stats
    )
    restamp_shift_parquet(path, shift_release_metadata(stats, release))
    print_shift_counts(path, stats, len(offsets))


def print_shift_counts(path: Path, stats, n_chains_released: int) -> None:
    c = stats.counts()
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")
    print(f"rows: {c['rows']}   chains: {c['chains']} / {n_chains_released}")
    print(f"  backbone: {c['backbone_rows']}   side chain: {c['sidechain_rows']}")
    print(
        f"  corrected: {c['corrected_rows']}   raw only (NULL): "
        f"{c['rows'] - c['corrected_rows']}"
    )
    print("  offset_source:", c["offset_source"])
    far = c["far_outlier"]
    print(
        f"  far outliers: {far['flagged']} of {far['assessed']} assessed shifts "
        f"in {far['chains']} chains"
    )
    dup = c["duplicate_keys"]
    print(
        f"  repeated keys: {dup['same_value']} same value, "
        f"{dup['different_value']} two values"
    )
    nc = c["not_covered"]
    if isinstance(nc, dict):
        print(
            f"  not covered: {nc['values']} of {nc['deposited_values']} values "
            f"on {nc['comp_ids']} Comp_IDs in {nc['chains']} chains"
        )


def restamp_shifts(
    shifts_in: Path,
    out_path: Path,
    pkl_dir: Path | None,
    release: dict[str, str],
) -> None:
    """Restamp an existing shift table's metadata (release keys, counts, the
    current static text) without rebuilding it: the counts come from the file
    itself, the excluded non-canonical values from the raw shifts of its
    chains in ``pkl_dir`` (counted as not read when None). The data are
    checked unchanged (:func:`trizod.shifts.restamp_shift_parquet`)."""
    from trizod.shifts.extract import iter_chains

    stats = ShiftTableStats()
    ids = []
    for cid, df in iter_file_frames(shifts_in):
        stats.add_frame(cid, df)
        ids.append(cid)
    print(f"read {stats.rows} rows over {stats.chains} chains from {shifts_in}")
    if pkl_dir is not None:
        if not pkl_dir.is_dir():
            raise SystemExit(f"BMRB pickle cache not found: {pkl_dir}")
        logging.getLogger("trizod.bmrb").setLevel(logging.CRITICAL)
        for cid, _seq, shifts in iter_chains(ids, pkl_dir):
            stats.add_raw(cid, shifts)
    restamp_shift_parquet(
        shifts_in, shift_release_metadata(stats, release), out_path=out_path
    )
    print_shift_counts(out_path, stats, len(ids))


def snapshot_label(bmrb_dir: Path) -> str:
    """Name the snapshot the links were read from: the retrieval time recorded in
    its ``BMRB_SNAPSHOT.json`` when present, else the resolved directory."""
    meta = Path(bmrb_dir) / "BMRB_SNAPSHOT.json"
    try:
        retrieved = json.loads(meta.read_text()).get("retrieved")
    except (OSError, ValueError):
        retrieved = None
    resolved = Path(bmrb_dir).resolve().name
    return f"{resolved} (retrieved {retrieved})" if retrieved else resolved


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--in-bundle",
        type=Path,
        default=None,
        help="package bundle written by `trizod dataset package` "
        "(trizod-dataset-<version>/)",
    )
    ap.add_argument("--out", type=Path, default=None, help="output .parquet path")
    ap.add_argument(
        "--composition-csv",
        type=Path,
        default=None,
        help=(
            "molecular-composition cache written by `trizod dataset build` "
            "(default: <work-dir>/final_dataset/" + COMPOSITION_CACHE_NAME + ")"
        ),
    )
    ap.add_argument(
        "--skip-composition",
        action="store_true",
        help="write the composition columns as all-null instead of joining them",
    )
    ap.add_argument(
        "--bmrb-dir",
        type=Path,
        default=paths.RAW_BMRB,
        help=(
            "BMRB NMR-STAR snapshot the UniProt/PDB cross-references are read "
            "from (default: data/raw/bmrb_entries)"
        ),
    )
    ap.add_argument(
        "--skip-xrefs",
        action="store_true",
        help="write the cross-reference columns as all-null instead of reading them",
    )
    ap.add_argument(
        "--xref-sources",
        type=Path,
        default=None,
        help=(
            "dated directory written by scripts/fetch_xref_sources.py (BMRB-PDB "
            "match list, wwPDB SEQRES, SIFTS); adds same-deposition PDB entries "
            "and per-chain SIFTS UniProt accessions. Without it the "
            "cross-references come from the NMR-STAR snapshot only"
        ),
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "repository root whose data/external holds the CheZOD sequence sets "
            "(default: auto-detected)"
        ),
    )
    ap.add_argument(
        "--skip-chezod-overlap",
        action="store_true",
        help="write the <set>_overlap columns as all-null instead of searching",
    )
    ap.add_argument(
        "--shifts-out",
        type=Path,
        default=None,
        help="also write the complete chemical-shift table here",
    )
    ap.add_argument(
        "--pkl-dir",
        type=Path,
        default=paths.PKL_DIR,
        help="BMRB pickle cache read by --shifts-out and --restamp-shifts "
        "(default: tmp/bmrb_entries)",
    )
    ap.add_argument(
        "--restamp-shifts",
        type=Path,
        default=None,
        metavar="SHIFTS_PARQUET",
        help="restamp an existing trizod_shifts.parquet instead of rebuilding it: "
        "release keys, counts (from the file; the excluded non-canonical values "
        "from --pkl-dir) and the current metadata text; written to --shifts-out "
        "(default: in place). The data are checked unchanged",
    )
    ap.add_argument(
        "--scoring-commit",
        default=None,
        help="commit (or trizod-<sha> pipeline_version) that scored the entries "
        "and wrote the package bundle (default: the bundle MANIFEST's "
        "pipeline_version)",
    )
    ap.add_argument(
        "--version",
        default=ri.DATASET_VERSION,
        help="dataset version written into the metadata (default: %(default)s)",
    )
    ap.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "threshold-framework output directory (or its results.json): its "
            "prevalence, likelihood ratios at the cuts, per-split prevalence, "
            "SHA-256 and framework commit go into the `calibration` metadata key"
        ),
    )
    ap.add_argument(
        "--calibration-commit",
        default=None,
        help=(
            "commit of the threshold framework that produced --calibration "
            "(default: the last commit of " + ri.FRAMEWORK_PATH + " here)"
        ),
    )
    ap.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help=(
            "BMRB snapshot directory whose BMRB_SNAPSHOT.json identifies the "
            "snapshot in the metadata (default: --bmrb-dir)"
        ),
    )
    ap.add_argument(
        "--write-column-dictionary",
        type=Path,
        default=None,
        metavar="PATH",
        help="write the column dictionary (Markdown) to PATH, e.g. "
        "docs/dataset/columns.md",
    )
    args = ap.parse_args(argv)
    if args.write_column_dictionary is not None:
        args.write_column_dictionary.write_text(column_dictionary_markdown())
        print(f"wrote {args.write_column_dictionary} ({len(COLUMNS)} columns)")
        if not args.out and not args.shifts_out:
            return
    if args.in_bundle is None:
        ap.error("--in-bundle is required to build")
    if not args.out and not args.shifts_out and not args.restamp_shifts:
        ap.error("nothing to do: pass --out, --shifts-out and/or --restamp-shifts")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    scoring = ri.scoring_block(
        args.scoring_commit or ri.bundle_pipeline_version(args.in_bundle)
    )
    print(f"scoring commit: {scoring['commit']} ({scoring['pipeline_version']})")
    snapshot = ri.snapshot_block(args.snapshot_dir or args.bmrb_dir)
    if snapshot is None:
        print(
            "  WARNING: no BMRB_SNAPSHOT.json in "
            f"{args.snapshot_dir or args.bmrb_dir} -- the bmrb_snapshot "
            "metadata will say 'not recorded'"
        )
    if args.restamp_shifts:
        restamp_shifts(
            args.restamp_shifts,
            args.shifts_out or args.restamp_shifts,
            args.pkl_dir,
            release_keys(args.version, snapshot, scoring=scoring),
        )
        if not args.out:
            return
    composition_csv = None
    if args.out:
        # The composition cache lives in the build work dir, not in the bundle
        # (package_release.py stages only the published artefacts). Missing it is
        # an error rather than a silent all-null block: these annotations are the
        # deliverable, and a Parquet that quietly lacks them looks identical.
        composition_csv = args.composition_csv
        if args.skip_composition:
            composition_csv = None
        elif composition_csv is None:
            composition_csv = (
                resolve_dataset_paths().final_dataset / COMPOSITION_CACHE_NAME
            )
        if composition_csv is not None and not composition_csv.is_file():
            ap.error(
                f"composition cache not found: {composition_csv}\n"
                "run `trizod dataset build` first, pass --composition-csv, or "
                "pass --skip-composition to accept null composition columns"
            )
        if not args.skip_xrefs and not args.bmrb_dir.is_dir():
            ap.error(
                f"BMRB snapshot not found: {args.bmrb_dir}\n"
                "pass --bmrb-dir, or --skip-xrefs to accept null cross-references"
            )
    # the four tiers' scores.json, loaded once for the main and the shift table
    labelled = label_records(args.in_bundle)
    if args.out:
        xrefs = xrefs_snapshot = None
        if not args.skip_xrefs:
            entry_ids = {clean_str(rec.get("entryID")) for _, rec in labelled.values()}
            print(f"reading cross-references for {len(entry_ids)} entries ...")
            xrefs = xr.load_xrefs(args.bmrb_dir, entry_ids)
            xrefs_snapshot = snapshot_label(args.bmrb_dir)
        external = None
        if xrefs is not None and args.xref_sources is not None:
            confident_pdb = {
                p
                for e in xrefs.values()
                for links in [e.entry, *e.entity.values()]
                for p in xr.confident_pdb_ids(links)
            }
            print(f"reading dated cross-reference sources from {args.xref_sources}")
            external = xs.ExternalSources.load(
                args.xref_sources, entry_ids, extra_pdb_ids=confident_pdb
            )
        overlaps = None
        if args.skip_chezod_overlap:
            print("  WARNING: --skip-chezod-overlap — the overlap flags will be null")
        else:
            import tempfile

            chains = {rid: rec["seq"] for rid, (_, rec) in labelled.items()}
            sets = co.load_chezod_sets(root=args.root)
            print(
                "searching CheZOD overlaps: "
                + ", ".join(f"{k} {len(v)}" for k, v in sets.items())
            )
            with tempfile.TemporaryDirectory(prefix="trizod_chezod_") as td:
                overlaps = co.chezod_overlaps(chains, sets, Path(td))
            for name, ids in overlaps.items():
                print(f"  {co.overlap_column(name)}: {len(ids)} chains")
        calibration = None
        if args.calibration is not None:
            fw_commit = (
                {"commit": args.calibration_commit, "dirty": None}
                if args.calibration_commit
                else None
            )
            calibration = ri.read_calibration(args.calibration, fw_commit)
            print(
                f"calibration: pi {calibration['prevalence']['value']} "
                f"{calibration['prevalence']['ci95']}, results.json "
                f"{calibration['results_json_sha256'][:12]}"
            )
        else:
            print("  WARNING: no --calibration -- the calibration metadata is empty")
        build(
            args.in_bundle,
            args.out,
            composition_csv,
            labelled=labelled,
            xrefs=xrefs,
            xrefs_snapshot=xrefs_snapshot,
            external=external,
            chezod_overlaps=overlaps,
            version=args.version,
            snapshot=snapshot,
            calibration=calibration,
            scoring=scoring,
        )
    if args.shifts_out and not args.restamp_shifts:
        build_shifts(
            args.in_bundle,
            args.shifts_out,
            args.pkl_dir,
            labelled,
            release=release_keys(args.version, snapshot, scoring=scoring),
        )


if __name__ == "__main__":
    main()
