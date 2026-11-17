"""Attach TriZOD's re-referencing to a deposited shift table.

The scoring path corrects a backbone shift in two steps, **in two different
units**::

    corrected_ppm = raw_ppm - lacs_off_<atom>_ppm - off_<atom>_sigma * REFINED_WEIGHTS[atom]

``apply_lacs_correction`` subtracts a ppm offset straight off the raw array;
``compute_offsets`` averages ``diff / REFINED_WEIGHTS`` and so returns a
multiple of the per-atom POTENCI RMSD. Adding the two together is a unit error
worth up to 21.2 ppm. Nothing here re-derives that arithmetic — the conversion
lives once, in :func:`trizod.offsets.total_offset_ppm`.

**Which offset applies to which row** is a measured question, not an assumption.
A backbone row carries the offset of the slot the scorer averaged it into. A
side-chain row carries whatever transfers from the backbone estimate, and the
transfer was measured per nucleus on the released corpus, one value per
(chain, ``comp_id``+``atom_id``) group as a deviation from the corpus-wide group
centre. Subtracting ``x`` changes MSE by ``var(x) - 2*cov(dev, x)``, so it helps
iff the transfer slope ``beta = cov(dev, x) / var(x)`` exceeds 0.5:

============ ======================= =========================================
nucleus      beta [95 % CI]          consequence
============ ======================= =========================================
13C          0.785 [0.767, 0.806]    apply ``lacs_off_CA_ppm``; MSE x0.79
                                     (RMSE 0.919 -> 0.818 ppm)
1H           0.079 [0.057, 0.102]    leave raw; applying inflates MSE x1.131
15N          0.366 [0.330, 0.407]    leave raw; applying inflates MSE x1.075
============ ======================= =========================================

For 1H the correction is statistically indistinguishable from applying a random
*other* chain's offset (dispersion test against a permutation null: observed
ΔMADn +23.3 % vs permuted +24.1 %). That is a statement about *transfer*, not
about the side chain: a per-chain side-chain 1H referencing constant genuinely
exists and is reproducible (split-half reliability 0.812) — the backbone amide
offset just explains 1.3 % of its variance. For 15N there is partial transfer
(+12.6 % vs +26.3 %) but it does not reach break-even, and the verdict is
estimator-independent: PANAV, an independent offset estimator, gives beta 0.394
and an instrumental-variable disattenuation gives 0.499 [0.430, 0.555] — at the
break-even, not above it. Neither offset is near zero, so "leave raw" is a
positive finding about transfer, not an absence of signal, and the corrected
column is therefore **NULL** for those rows rather than a copy of the raw value.

Only ``lacs_off_CA_ppm`` is propagated to side-chain carbons, and only it: the
POTENCI residual term transfers at beta 0.093 (CA) / 0.081 (CB) / 0.050 (C), a
residual bias in the backbone-vs-POTENCI comparison rather than a spectrometer
referencing shift, and count-weighting over CA/CB/C measures worse
(beta 0.751, MSE x0.812) than CA alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trizod.constants import BACKBONE_ATOMS
from trizod.offsets import (
    lacs_off_ppm_col,
    legacy_lacs_off_col,
    legacy_off_col,
    off_sigma_col,
    total_off_ppm_col,
    total_offset_ppm,
)

#: Domain of the ``offset_source`` column.
#:
#: ``lacs+potenci`` / ``lacs_only`` / ``potenci_only``
#:     a non-zero offset was subtracted; the label names the estimators that
#:     contributed a non-zero amount to it.
#: ``none``
#:     nothing was subtracted. ``offset_applied_ppm`` is 0.0 when the estimators
#:     ran and returned zero (``val_corrected_ppm == val_ppm``, a measured no-op)
#:     and NULL when no offset could be determined at all
#:     (``val_corrected_ppm`` NULL).
#: ``not_transferable``
#:     the chain has an offset, but it does not demonstrably transfer to this
#:     row, so no corrected value is published: every side-chain 1H and 15N, and
#:     side-chain 13C on the chains whose LACS carbon offset exceeds the gate
#:     below. Both value columns are NULL.
OFFSET_SOURCES = (
    "lacs+potenci",
    "lacs_only",
    "potenci_only",
    "none",
    "not_transferable",
)

#: Backbone atom whose LACS offset is propagated to side-chain carbons.
#: ``lacs_off_CB_ppm`` is identical to it in 16,850 of the 16,851 released chains
#: (LACS emits one carbon offset; the exception is 50331_2_2_2, CA -0.16 vs CB
#: -0.40) and the carbonyl offset transfers at only beta = 0.307.
SIDECHAIN_CARBON_LACS_ATOM = "CA"

#: Above this |lacs_off_CA_ppm| the offset is withheld from side-chain carbons.
#: 13 chains of 16,851 (0.08 %) exceed it, 9 of which carry side-chain carbons —
#: 927 rows in the released table. They are real gross deposition errors that
#: LACS detects correctly, but only 5 of those 9 actually transfer (ratio
#: 0.97-1.02) while 4 do not — a coin flip that cannot be adjudicated per chain,
#: and ungated the MSE ratio blows up to 7.42.
SIDECHAIN_MAX_ABS_LACS_PPM = 5.0

#: Nuclei whose side-chain shifts carry a correction. Everything else (1H, 15N,
#: and the handful of 31P values) is published raw.
SIDECHAIN_CORRECTED_NUCLEI = frozenset({"C"})


def backbone_slot(comp_id, atom_id):
    """The ``BACKBONE_ATOMS`` slot ``get_valid_bbshifts`` reads ``atom_id`` into.

    Mirrors that function's mapping exactly, degeneracy included: ``HB2``/``HB3``
    (any residue) and ALA ``HB1`` average into ``HB``, GLY ``HA2``/``HA3``
    average into ``HA``. Returns ``None`` for an atom it has no slot for — which
    for a canonical residue means the whole chain is rejected there, so such a
    row can never carry an offset either.
    """
    if atom_id in BACKBONE_ATOMS:
        return atom_id
    if atom_id in ("HB2", "HB3"):
        return "HB"
    if atom_id == "HB1" and comp_id == "ALA":
        return "HB"
    if atom_id in ("HA2", "HA3") and comp_id == "GLY":
        return "HA"
    return None


def _finite(value):
    """Coerce to float; None / pandas.NA / non-numeric / NaN all become None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


@dataclass(frozen=True)
class ChainOffsets:
    """One chain's per-atom referencing offsets, in their stated units.

    ``lacs_ppm`` and ``sigma`` are the two raw estimates; ``total_ppm`` is what
    the scorer effectively subtracted from a deposited ppm shift. A value is
    ``None`` where the chain has no estimate for that atom.
    """

    lacs_ppm: dict
    sigma: dict
    total_ppm: dict

    @property
    def sidechain_carbon_ppm(self):
        """The offset propagated to side-chain carbons, or ``None`` if withheld."""
        lacs = self.lacs_ppm.get(SIDECHAIN_CARBON_LACS_ATOM)
        if lacs is None or abs(lacs) > SIDECHAIN_MAX_ABS_LACS_PPM:
            return None
        return lacs


def chain_offsets(record):
    """Read one ``scores.json`` record's offset block into a :class:`ChainOffsets`.

    Accepts both the current column spelling (``off_CA_sigma`` /
    ``lacs_off_CA_ppm`` / ``total_off_CA_ppm``) and the pre-rename one
    (``off_CA`` / ``lacs_off_CA``) that earlier scored releases on disk carry,
    so a bundle staged before the rename still builds. ``total_off_<atom>_ppm``
    is used when present and otherwise computed with the shared helper — never
    by re-deriving the sigma->ppm conversion here.
    """
    record = record or {}
    lacs, sigma, total = {}, {}, {}
    for atom in BACKBONE_ATOMS:
        lacs_ppm = _finite(
            record.get(lacs_off_ppm_col(atom), record.get(legacy_lacs_off_col(atom)))
        )
        off_sigma = _finite(
            record.get(off_sigma_col(atom), record.get(legacy_off_col(atom)))
        )
        total_ppm = _finite(record.get(total_off_ppm_col(atom)))
        if total_ppm is None and lacs_ppm is not None and off_sigma is not None:
            total_ppm = _finite(total_offset_ppm(atom, lacs_ppm, off_sigma))
        lacs[atom], sigma[atom], total[atom] = lacs_ppm, off_sigma, total_ppm
    return ChainOffsets(lacs_ppm=lacs, sigma=sigma, total_ppm=total)


def _backbone_offset(offsets, comp_id, atom_id):
    """``(offset_ppm, offset_source)`` for one backbone row."""
    slot = backbone_slot(comp_id, atom_id)
    if slot is None:
        return None, "none"
    total = offsets.total_ppm.get(slot)
    if total is None:
        return None, "none"
    lacs = offsets.lacs_ppm.get(slot) or 0.0
    sigma = offsets.sigma.get(slot) or 0.0
    if lacs != 0.0 and sigma != 0.0:
        return total, "lacs+potenci"
    if lacs != 0.0:
        return total, "lacs_only"
    if sigma != 0.0:
        return total, "potenci_only"
    return total, "none"


def _sidechain_offset(offsets, atom_type):
    """``(offset_ppm, offset_source)`` for one side-chain row."""
    if atom_type not in SIDECHAIN_CORRECTED_NUCLEI:
        return None, "not_transferable"
    ppm = offsets.sidechain_carbon_ppm
    if ppm is None:
        return None, "not_transferable"
    return ppm, ("lacs_only" if ppm != 0.0 else "none")


def annotate_offsets(df, offsets):
    """Add ``val_corrected_ppm`` / ``offset_applied_ppm`` / ``offset_source``.

    Args:
        df: a :func:`trizod.bmrb.bmrb.get_deposited_shifts` frame for one chain.
        offsets: that chain's :class:`ChainOffsets`.

    Returns:
        A copy of ``df`` with the three columns appended.
        ``val_corrected_ppm = val_ppm - offset_applied_ppm`` wherever an offset
        applies and NULL wherever one does not — never a silent copy of
        ``val_ppm``, which would make "corrected" a lie for exactly the rows a
        consumer most needs to identify.
    """
    df = df.copy()
    if offsets is None:
        offsets = ChainOffsets({}, {}, {})

    applied, source = [], []
    for comp_id, atom_id, atom_type, is_bb in zip(
        df["comp_id"], df["atom_id"], df["atom_type"], df["is_backbone"]
    ):
        if is_bb:
            ppm, src = _backbone_offset(offsets, comp_id, atom_id)
        else:
            ppm, src = _sidechain_offset(offsets, str(atom_type).strip().upper())
        applied.append(np.nan if ppm is None else ppm)
        source.append(src)

    df["val_corrected_ppm"] = df["val_ppm"] - pd.Series(applied, index=df.index)
    df["offset_applied_ppm"] = applied
    df["offset_source"] = source
    return df
