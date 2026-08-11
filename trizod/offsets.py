"""Referencing-offset column names, units, and record access (``trizod.offsets``).

TriZOD carries two per-atom referencing offsets that are NOT in the same unit,
and the emitted column names say so:

``off_<atom>_sigma``
    POTENCI/AIC residual offset. ``scoring.compute_offsets`` averages
    ``diff_arr / REFINED_WEIGHTS``, so this is a multiple of the per-atom
    POTENCI RMSD, not ppm. The ``--max-offset`` thresholds (3/3/2) are compared
    against THIS number, so it must stay in sigma.
``lacs_off_<atom>_ppm``
    LACS offset. ``scoring.apply_lacs_correction`` subtracts it straight off the
    raw shift array, so it is ppm.
``total_off_<atom>_ppm``
    Derived. The single number to subtract from a deposited shift to reproduce
    the shift TriZOD scored.

Adding the first two together is a unit error worth up to 21.2 ppm. That is
exactly why ``total_off_<atom>_ppm`` exists: nobody downstream should ever have
to do this arithmetic by hand.

Why this is not in ``trizod.constants``
---------------------------------------
``provenance.scoring_cache_version()`` hashes ``constants.py`` into the
``tmp/wSCS/`` cache key so that a change to the scoring math invalidates stale
cached offsets. None of the helpers below affect a single scored value — they
only name columns and convert an already-computed offset for *reporting*.
Putting them in ``constants.py`` moved the cache key and orphaned ~34k cached
``.npz`` files (984 MB) for nothing. They live here so the scoring-math hash
stays a hash of the scoring math.

Reading offsets back out of a scored record
-------------------------------------------
Use :func:`offset_value` / :func:`max_abs_offset`, never ``record.get(col) or
0.0``. That idiom coerces a *missing key* to a real 0.0, so a renamed column
turns into "this chain is perfectly referenced" instead of an error — which is
how a stale key silently changed which chain won an exact-sequence dedup group.
"""

from trizod.constants import BACKBONE_ATOMS, REFINED_WEIGHTS


def off_sigma_col(atom):
    """Column holding the POTENCI/AIC residual offset for `atom`, in sigma."""
    return f"off_{atom}_sigma"


def lacs_off_ppm_col(atom):
    """Column holding the LACS offset for `atom`, in ppm."""
    return f"lacs_off_{atom}_ppm"


def total_off_ppm_col(atom):
    """Column holding the total applied offset for `atom`, in ppm."""
    return f"total_off_{atom}_ppm"


def legacy_off_col(atom):
    """Pre-rename spelling of :func:`off_sigma_col`. READ-ONLY.

    Scored releases staged before the offset columns gained their unit suffix
    spell them this way. Readers that must still open such a bundle may fall
    back to this name; nothing may ever *emit* it (see
    ``tests/test_offset_units.py``), because the name hides its unit.
    """
    return f"off_{atom}"


def legacy_lacs_off_col(atom):
    """Pre-rename spelling of :func:`lacs_off_ppm_col`. READ-ONLY.

    See :func:`legacy_off_col`.
    """
    return f"lacs_off_{atom}"


#: Every per-atom offset column TriZOD emits, in a stable order.
OFFSET_COLUMNS = (
    [off_sigma_col(a) for a in BACKBONE_ATOMS]
    + [lacs_off_ppm_col(a) for a in BACKBONE_ATOMS]
    + [total_off_ppm_col(a) for a in BACKBONE_ATOMS]
)


def _as_float(value):
    """Coerce to float, mapping None/pd.NA/non-numeric to NaN."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        # pandas.NA raises TypeError on float(); treat it as "not measured".
        return float("nan")


def total_offset_ppm(atom, lacs_off_ppm, off_sigma):
    """Total ppm subtracted from the deposited shift of `atom` during scoring.

    The scoring path applies the two corrections in different domains: the LACS
    offset comes off the raw ppm array (`apply_lacs_correction`), the POTENCI
    residual offset comes off the weighted, sigma-scaled diffs
    (`compute_weighted_diffs`). Expressed entirely in ppm that is

        corrected_ppm = raw_ppm - lacs_off_ppm - off_sigma * REFINED_WEIGHTS[atom]

    so this function returns the bracketed quantity. This is the ONLY place the
    conversion is written down; every emitter reads it from here.

    Missing inputs (None, NaN, pandas.NA) propagate as NaN: an offset that was
    never computed, or was rejected by `--max-offset`, has no ppm equivalent.
    """
    return _as_float(lacs_off_ppm) + _as_float(off_sigma) * REFINED_WEIGHTS[atom]


class MissingOffsetColumn(KeyError):
    """A scored record does not carry an offset column that was asked for.

    Raised instead of returning a default, because the default that used to
    stand here (``record.get(col) or 0.0``) is indistinguishable from a
    genuinely zero offset and silently rewrote every downstream ranking.
    """


def offset_value(record, column):
    """Return one offset column of a scored ``scores.json`` record.

    Three outcomes, all distinct — which is the entire point:

    * column present and numeric -> ``float``
    * column present and ``null``/NaN (offset rejected by ``--max-offset``, or
      never computed for this atom) -> ``None``
    * column absent -> :class:`MissingOffsetColumn`

    A ``scores.json`` that lacks the column is schema drift, not a chain with
    no offset; scoring emits all of :data:`OFFSET_COLUMNS` for every record.
    """
    if column not in record:
        raise MissingOffsetColumn(
            f"{column!r} is not in this scores.json record. Offset columns were "
            "renamed to off_<atom>_sigma / lacs_off_<atom>_ppm / "
            "total_off_<atom>_ppm; a release scored before the rename must be "
            "regenerated before it can be consumed here."
        )
    value = _as_float(record[column])
    return None if value != value else value  # NaN -> None


def max_abs_offset(record, columns):
    """Largest ``abs()`` over `columns` of `record`, or ``None`` if all are null.

    Missing columns raise (see :func:`offset_value`); *null* columns are skipped,
    so a chain whose CB offset was rejected still reports the magnitude of the
    offsets that were computed. ``None`` means "nothing measurable here" and must
    not be conflated with ``0.0``, which means "measured, and perfectly
    referenced".
    """
    values = [offset_value(record, c) for c in columns]
    present = [abs(v) for v in values if v is not None]
    return max(present) if present else None
