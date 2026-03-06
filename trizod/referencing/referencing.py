"""Chemical shift re-referencing using LACS-inspired analysis.

Detects and corrects systematic referencing errors in NMR chemical shift data
by comparing observed shifts against POTENCI random coil predictions. The LACS
(Linear Analysis of Chemical Shifts) approach exploits the fact that CA and CB
share the same 13C reference, so their offsets are correlated.

This module runs BEFORE the existing per-atom offset correction in scoring.py,
which handles finer-grained local deviations.
"""

import logging
import warnings

import numpy as np

from trizod.constants import BBATNS

logger = logging.getLogger("trizod.referencing")

# Indices into the BBATNS array: ["C", "CA", "CB", "HA", "H", "N", "HB"]
_ATM_IDX = {at: i for i, at in enumerate(BBATNS)}


def estimate_reference_offsets(
    delta: np.ndarray,
    mask: np.ndarray,
    method: str = "lacs",
) -> dict[str, float]:
    """Estimate systematic referencing offsets per atom type.

    Parameters
    ----------
    delta : np.ndarray, shape (n_residues, 7)
        Secondary chemical shifts (observed - predicted).
    mask : np.ndarray, shape (n_residues, 7)
        Boolean mask where both observed and predicted are valid.
    method : str
        "lacs" — LACS-style 13C detection + per-atom-type means for N/H.
        "global" — Simple per-atom-type mean offsets.

    Returns
    -------
    dict[str, float]
        Estimated offset for each atom type (in ppm). Zero if not estimable.
    """
    if method == "global":
        return _global_offsets(delta, mask)
    elif method == "lacs":
        return _lacs_offsets(delta, mask)
    else:
        raise ValueError(f"Unknown re-referencing method: {method}")


def _global_offsets(delta: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Simple per-atom-type mean offset estimation."""
    offsets = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for at in BBATNS:
            i = _ATM_IDX[at]
            n = mask[:, i].sum()
            if n >= 4:
                offsets[at] = float(np.nanmean(np.where(mask[:, i], delta[:, i], np.nan)))
            else:
                offsets[at] = 0.0
    return offsets


def _lacs_offsets(delta: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """LACS-inspired re-referencing.

    For 13C atoms (C, CA, CB): exploits the fact that CA and CB share the
    same 13C spectrometer reference. A systematic offset shifts both equally.

    1. Compute mean(delta_CA) and mean(delta_CB) where both are observed.
    2. If both show a consistent offset (same sign, similar magnitude),
       the 13C offset = (mean_CA + mean_CB) / 2.
    3. Apply the same 13C offset to CO.

    For 15N: simple mean of delta_N.
    For 1H (H, HA, HB): simple mean of each.
    """
    offsets = dict.fromkeys(BBATNS, 0.0)

    i_ca = _ATM_IDX["CA"]
    i_cb = _ATM_IDX["CB"]
    i_co = _ATM_IDX["C"]

    # --- 13C re-referencing (LACS-style) ---
    both_mask = mask[:, i_ca] & mask[:, i_cb]
    n_both = both_mask.sum()

    if n_both >= 10:
        mean_ca = np.mean(delta[both_mask, i_ca])
        mean_cb = np.mean(delta[both_mask, i_cb])

        # LACS criterion: CA and CB offsets should be consistent
        # (same 13C reference error shifts both by the same amount)
        if np.sign(mean_ca) == np.sign(mean_cb) and abs(mean_ca - mean_cb) < 1.0:
            c13_offset = (mean_ca + mean_cb) / 2.0
            offsets["CA"] = c13_offset
            offsets["CB"] = c13_offset
            if mask[:, i_co].sum() >= 4:
                offsets["C"] = c13_offset
            logger.info(
                f"LACS 13C offset: {c13_offset:.3f} ppm "
                f"(CA={mean_ca:.3f}, CB={mean_cb:.3f}, n={n_both})"
            )
        else:
            logger.info(
                f"LACS 13C: CA ({mean_ca:.3f}) and CB ({mean_cb:.3f}) "
                f"disagree, using individual offsets"
            )
            offsets["CA"] = mean_ca
            if mask[:, i_cb].sum() >= 4:
                offsets["CB"] = mean_cb
            if mask[:, i_co].sum() >= 4:
                offsets["C"] = float(np.nanmean(np.where(mask[:, i_co], delta[:, i_co], np.nan)))
    else:
        for at in ("CA", "CB", "C"):
            i = _ATM_IDX[at]
            if mask[:, i].sum() >= 4:
                offsets[at] = float(np.nanmean(np.where(mask[:, i], delta[:, i], np.nan)))

    # --- 15N re-referencing ---
    i_n = _ATM_IDX["N"]
    if mask[:, i_n].sum() >= 10:
        offsets["N"] = float(np.nanmean(np.where(mask[:, i_n], delta[:, i_n], np.nan)))

    # --- 1H re-referencing ---
    for at in ("H", "HA", "HB"):
        i = _ATM_IDX[at]
        if mask[:, i].sum() >= 10:
            offsets[at] = float(np.nanmean(np.where(mask[:, i], delta[:, i], np.nan)))

    return offsets


def validate_offsets(
    offsets: dict[str, float],
    delta: np.ndarray,
    mask: np.ndarray,
    min_observations: int = 10,
    min_aic_improvement: float = 6.0,
    max_offset: float = 5.0,
) -> dict[str, float]:
    """Validate estimated offsets using AIC criterion.

    Only accepts an offset if:
    1. There are at least ``min_observations`` data points for that atom type.
    2. The AIC improvement ``N * ln(sigma_before / sigma_after)`` exceeds threshold.
    3. The absolute offset is within ``max_offset`` ppm.

    Parameters
    ----------
    offsets : dict[str, float]
        Raw estimated offsets from estimate_reference_offsets().
    delta : np.ndarray, shape (n_residues, 7)
        Secondary chemical shifts (observed - predicted).
    mask : np.ndarray, shape (n_residues, 7)
        Boolean mask.
    min_observations : int
        Minimum data points required per atom type.
    min_aic_improvement : float
        Minimum AIC improvement to accept offset.
    max_offset : float
        Maximum acceptable offset magnitude (ppm).

    Returns
    -------
    dict[str, float]
        Validated offsets (rejected offsets set to 0.0).
    """
    validated = dict.fromkeys(BBATNS, 0.0)

    for at in BBATNS:
        offset = offsets.get(at, 0.0)
        if offset == 0.0:
            continue

        i = _ATM_IDX[at]
        col_mask = mask[:, i]
        n = col_mask.sum()

        if n < min_observations:
            logger.info(f"Rejecting {at} offset {offset:.3f}: too few observations ({n})")
            continue

        if abs(offset) > max_offset:
            logger.info(f"Rejecting {at} offset {offset:.3f}: exceeds max ({max_offset})")
            continue

        # AIC test: compare RMS before vs stddev after correction
        residuals = delta[col_mask, i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sigma_before = np.sqrt(np.mean(residuals**2))
            sigma_after = np.std(residuals)

        if sigma_before <= 0 or sigma_after <= 0:
            continue

        daic = n * np.log(sigma_before / sigma_after) - 1
        if daic > min_aic_improvement:
            validated[at] = offset
            logger.info(f"Accepted {at} offset {offset:.3f} (dAIC={daic:.1f}, n={n})")
        else:
            logger.info(f"Rejecting {at} offset {offset:.3f}: low dAIC ({daic:.1f})")

    return validated


def apply_rereferencing(
    cmparr: np.ndarray,
    mask: np.ndarray,
    offsets: dict[str, float],
) -> np.ndarray:
    """Apply re-referencing corrections to secondary chemical shifts.

    Subtracts the validated offsets from cmparr (delta = obs - pred) in place.
    After correction, the systematic referencing error is removed.

    Returns the modified cmparr.
    """
    for at in BBATNS:
        i = _ATM_IDX[at]
        off = offsets.get(at, 0.0)
        if off != 0.0:
            cmparr[mask[:, i], i] -= off
    return cmparr
