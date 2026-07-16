"""
LACS — Linear Analysis of Chemical Shifts (Python reimplementation).

Detects and quantifies chemical shift referencing errors by exploiting
the linear relationship between secondary shifts.  The reference-independent
quantity (δCA − δCB)_obs − (δCA − δCB)_rc is plotted on the X-axis; the
secondary shift of each target atom on the Y-axis.  A non-zero Y-intercept
reveals the referencing offset.

Based on:
  Wang L et al. (2005) J Biomol NMR 32:13-22  (13C, 1HA)
  Wang L & Markley JL (2009) J Biomol NMR 44:95-99  (15N, 1HN)
  MATLAB source: github.com/bmrb-io/LACS
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import mahalanobis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

# Amino acid order matching the MATLAB code (1-indexed codes 1-20).
# Index in these arrays corresponds to the single-letter code via _AA_TO_IDX.
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_ORDER)}

# Random coil shifts at pH ≥ 4 (Wishart et al. 1995, as used in LACS).
# Columns: CA, CB, HA, CO, H, N, (CA-CB)
# fmt: off
_RC_SHIFTS = {
    #       CA     CB     HA     CO      H      N
    "A": (52.5,  19.1,  4.32, 177.8,  8.24, 123.8),
    "C": (58.2,  28.0,  4.55, 174.6,  8.32, 118.8),
    "D": (54.2,  41.1,  4.64, 176.3,  8.34, 120.4),
    "E": (56.6,  29.9,  4.35, 176.6,  8.42, 120.2),
    "F": (57.7,  39.6,  4.62, 175.8,  8.30, 120.3),
    "G": (45.1,   0.0,  3.96, 174.9,  8.33, 108.8),
    "H": (55.0,  29.0,  4.73, 174.1,  8.42, 118.2),
    "I": (61.1,  38.8,  4.17, 176.4,  8.00, 119.9),
    "K": (56.2,  33.1,  4.32, 176.6,  8.29, 120.4),
    "L": (55.1,  42.4,  4.34, 177.6,  8.16, 121.8),
    "M": (55.4,  32.9,  4.48, 176.3,  8.28, 119.6),
    "N": (53.1,  38.9,  4.74, 175.2,  8.40, 118.7),
    "P": (63.3,  32.1,  4.42, 177.3,  0.00,   0.0),
    "Q": (55.7,  29.4,  4.34, 176.0,  8.32, 119.8),
    "R": (56.0,  30.9,  4.34, 176.3,  8.23, 120.5),
    "S": (58.3,  63.8,  4.47, 174.6,  8.31, 115.7),
    "T": (61.8,  69.8,  4.35, 174.7,  8.15, 113.6),
    "V": (62.2,  32.9,  4.12, 176.3,  8.03, 119.2),
    "W": (57.5,  29.6,  4.66, 176.1,  8.25, 121.3),
    "Y": (57.9,  38.8,  4.55, 175.9,  8.12, 120.3),
}
# fmt: on

_RC_CA = 0
_RC_CB = 1
_RC_HA = 2
_RC_CO = 3
_RC_H = 4
_RC_N = 5

# Pre-proline corrections (from LACS ord.m / ordN.m).
# Applied to residue i when residue i+1 is Pro.
# Columns: CA, CB, HA, CO, H, N
# fmt: off
_PRE_PRO = {
    "A": ( 2.0,  1.0, -0.30,  1.9,  0.05, -1.2),
    "C": ( 1.8,  0.9, -0.26,  1.6,  0.02, -1.1),
    "D": ( 2.0,  0.2, -0.29,  1.3,  0.03, -1.0),
    "E": ( 2.4,  0.7, -0.28,  1.7,  0.08, -1.5),
    "F": ( 2.1,  0.5, -0.27,  1.4,  0.17, -0.6),
    "G": ( 0.6,  0.0, -0.17,  0.4,  0.12, -0.3),
    "H": ( 1.7,  0.0, -0.30,  1.5,  0.05,  0.0),
    "I": ( 2.4,  0.1, -0.28,  1.4, -0.06, -1.8),
    "K": ( 2.0,  0.5, -0.29,  1.8,  0.11, -1.2),
    "L": ( 2.0,  0.7, -0.34,  1.9,  0.02, -0.8),
    "M": ( 2.1,  0.5, -0.26,  1.7,  0.03, -1.1),
    "N": ( 1.8,  0.2, -0.31,  1.6,  0.03, -0.3),
    "P": ( 1.8,  1.2, -0.31,  5.9,  0.00,  0.0),
    "Q": ( 2.0,  0.6, -0.31,  1.6,  0.03, -0.8),
    "R": ( 2.0,  0.7, -0.31,  1.8,  0.03, -0.8),
    "S": ( 1.9,  0.5, -0.31,  1.5,  0.05, -0.9),
    "T": ( 2.0,  0.0, -0.26,  1.5,  0.00, -2.4),
    "V": ( 2.4,  0.3, -0.32,  1.4,  0.01, -1.3),
    "W": ( 1.8,  0.7, -0.33,  1.3,  0.16, -0.9),
    "Y": ( 2.1,  0.5, -0.29,  1.1,  0.02, -0.5),
}
# fmt: on

# Preceding-residue correction for 15N / 1HN, derived from the BMRB ordN.m
# `Ncorr` table (github.com/bmrb-io/LACS/ordN.m). ordN.m subtracts a documented
# systematic offset from the raw table before use (Ncorr_N -= 1.486,
# Ncorr_HN -= 0.005); `_NCORR` reproduces that subtraction in code so the raw
# published values are the single source of truth and the offsets are named once
# (this is exactly the hand-transcription that issue #17 got wrong: a prior table
# had Ala N = +1.114 vs -1.486 here). An independent transcription of the raw
# table is machine-checked against `_NCORR` in tests/test_lacs.py::TestNCorrProvenance.
#
# On 600 BMRB entries the ordN.m values roughly halve the residual N-offset bias,
# improve agreement with PANAV (r 0.94->0.96), and remove a composition-dependent
# artifact (see tests/test_lacs.py). This preceding-residue table is separate from
# the post-fit `systematic_corr` constant, which was a DIFFERENT ordN.m addition:
# that constant (issue #20) has since been dropped from _compute_n_offset because it
# is unpublished and empirically induced a ~0.47 ppm N bias vs PANAV -- do NOT
# conflate the two.
_NCORR_SYSTEMATIC_N = 1.486
_NCORR_SYSTEMATIC_HN = 0.005
# Raw ordN.m Ncorr table, AA order ACDEFGHIKLMNPQRSTVWY. Columns: (N_raw, HN_raw).
# fmt: off
_ORDN_NCORR_RAW = {
    "A": (0.0, 0.00), "C": (3.5, 0.17), "D": (1.6, 0.04), "E": (2.0, 0.10),
    "F": (3.2, 0.04), "G": (0.8, -0.04), "H": (2.6, 0.13), "I": (5.0, 0.13),
    "K": (2.4, 0.08), "L": (1.8, 0.02), "M": (1.9, 0.06), "N": (1.5, 0.04),
    "P": (1.2, 0.16), "Q": (2.1, 0.10), "R": (2.2, 0.10), "S": (2.7, 0.08),
    "T": (3.2, 0.09), "V": (4.7, 0.14), "W": (3.6, -0.08), "Y": (3.6, 0.01),
}  # fmt: skip
# fmt: on
# _NCORR columns are (H_corr, N_corr) to match _NCORR_H / _NCORR_N below.
_NCORR = {
    aa: (hn_raw - _NCORR_SYSTEMATIC_HN, n_raw - _NCORR_SYSTEMATIC_N)
    for aa, (n_raw, hn_raw) in _ORDN_NCORR_RAW.items()
}

_NCORR_H = 0
_NCORR_N = 1

# Minimum residues required for LACS analysis
_MIN_RESIDUES = 20
# Minimum points per regression line in two-line fit
_MIN_POINTS_PER_LINE = 10

# Atoms excluded from the 13C analysis (ord.m)
_EXCLUDE_13C = {"C", "G", "P"}
# Atoms excluded from the 15N Y-axis (ordN.m): Pro has no HN/N
# Also exclude Gly and His as preceding residues for N/HN
_EXCLUDE_PRECEDING_N = {"G", "H"}


# ---------------------------------------------------------------------------
# Robust regression (bisquare / Tukey biweight IRLS)
# ---------------------------------------------------------------------------


def _robustfit(x: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-6):
    """Iteratively reweighted least squares with bisquare weight function.

    Mimics MATLAB's ``robustfit(x, y)`` returning (intercept, slope, weights).
    """
    n = len(x)
    if n < 3:
        # Too few points for robust fit; fall back to OLS
        X = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return beta[0], beta[1], np.ones(n)

    X = np.column_stack([np.ones(n), x])

    # Initial OLS fit
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    tune = 4.685  # bisquare tuning constant (MATLAB default)
    reg = 1e-10  # regularisation for near-singular matrices

    weights = np.ones(n)
    for _ in range(max_iter):
        r = y - X @ beta
        # Robust scale estimate (MAD)
        s = np.median(np.abs(r)) / 0.6745
        if s < 1e-12:
            break
        # Leverage via lstsq (robust to near-singular weighted design matrix)
        XtWX = X.T @ (X * weights[:, None]) + reg * np.eye(2)
        try:
            hat = np.sum(X * np.linalg.solve(XtWX, X.T).T, axis=1)
        except np.linalg.LinAlgError:
            break
        hat = np.clip(hat, 0, 1 - 1e-12)
        # Adjusted residuals
        u = r / (tune * s * np.sqrt(1 - hat))
        # Bisquare weights
        new_weights = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        if np.max(np.abs(new_weights - weights)) < tol:
            weights = new_weights
            break
        weights = new_weights
        # Weighted least squares: scale rows by sqrt(w) so lstsq minimizes
        # sum(w * r^2) (as MATLAB robustfit does), not sum(w^2 * r^2).
        sqrt_w = np.sqrt(weights)
        beta, _, _, _ = np.linalg.lstsq(X * sqrt_w[:, None], sqrt_w * y, rcond=None)

    return beta[0], beta[1], weights  # intercept, slope, weights


# ---------------------------------------------------------------------------
# Mahalanobis outlier removal
# ---------------------------------------------------------------------------


def _mahalanobis_outlier_removal(
    X: np.ndarray,
    Y: np.ndarray,
    threshold_frac: float = 0.80,
    max_outlier_frac: float = 0.02,
):
    """Remove outliers based on Mahalanobis distance.

    Parameters
    ----------
    X, Y : arrays of coordinates
    threshold_frac : fraction of max distance above which points are outliers
    max_outlier_frac : stop if total outliers exceed this fraction of inliers

    Returns
    -------
    mask : boolean array, True = inlier
    """
    data = np.column_stack([X, Y])
    n = len(data)
    mask = np.ones(n, dtype=bool)

    if n < 3:
        return mask

    cov = np.cov(data[mask].T)
    # Regularise to avoid singular covariance
    cov += np.eye(2) * 1e-10
    cov_inv = np.linalg.inv(cov)
    mean = np.mean(data[mask], axis=0)

    dists = np.array([mahalanobis(data[i], mean, cov_inv) for i in range(n)])
    dists[~mask] = 0

    sorted_idx = np.argsort(dists)
    max_dist = dists[sorted_idx[-1]]
    if max_dist < 1e-12:
        return mask

    # Find first point exceeding threshold and remove everything from there
    threshold = threshold_frac * max_dist
    total_removed = 0
    for i in reversed(range(n)):
        idx = sorted_idx[i]
        if dists[idx] > threshold:
            mask[idx] = False
            total_removed += 1
        else:
            break

    # Cap at max_outlier_frac
    max_removable = max(1, int(max_outlier_frac * mask.sum()))
    if total_removed > max_removable:
        # Re-add the least extreme outliers
        outlier_indices = sorted_idx[mask == False]  # noqa: E712
        outlier_indices_sorted = outlier_indices[np.argsort(dists[outlier_indices])]
        for idx in outlier_indices_sorted[: total_removed - max_removable]:
            mask[idx] = True

    return mask


# ---------------------------------------------------------------------------
# 13C / 1HA offset: two-line robust regression (ord.m)
# ---------------------------------------------------------------------------


def _compute_13c_offset(
    seq: str,
    seq_nums: np.ndarray,
    obs_ca: np.ndarray,
    obs_cb: np.ndarray,
    obs_target: np.ndarray,
    target_rc_col: int,
    two_line_threshold: float = 1.0,
) -> float | None:
    """Compute referencing offset for a 13C or 1HA atom type.

    Uses the two-line robust regression approach from LACS ord.m.
    The X-axis is (CA-CB)_secondary = (CA-CB)_obs - (CA-CB)_rc.
    The Y-axis is the secondary shift of the target atom.

    Parameters
    ----------
    seq : protein sequence (1-letter codes)
    seq_nums : residue numbers (1-based, sequential)
    obs_ca, obs_cb, obs_target : observed shifts (NaN where missing)
    target_rc_col : index into _RC_SHIFTS tuple (0=CA, 1=CB, 2=HA, 3=CO)

    Returns
    -------
    offset in ppm, or None if insufficient data
    """
    n = len(seq)

    # Work on copies for pre-Pro correction
    ca = obs_ca.copy()
    cb = obs_cb.copy()
    target = obs_target.copy()

    # Apply pre-Pro corrections
    for i in range(n - 1):
        if (
            seq[i] in _PRE_PRO
            and i + 1 < n
            and seq[i + 1] == "P"
            and seq_nums[i + 1] == seq_nums[i] + 1
        ):
            corr = _PRE_PRO[seq[i]]
            if not np.isnan(ca[i]):
                ca[i] += corr[_RC_CA]
            if not np.isnan(cb[i]):
                cb[i] += corr[_RC_CB]
            if not np.isnan(target[i]):
                target[i] += corr[target_rc_col]

    # Filter: valid data, known AA, exclude C/G/P for 13C analysis
    valid = np.ones(n, dtype=bool)
    rc_cacb = np.zeros(n)
    rc_target = np.zeros(n)

    for i in range(n):
        aa = seq[i]
        if aa not in _RC_SHIFTS or aa in _EXCLUDE_13C:
            valid[i] = False
            continue
        if np.isnan(ca[i]) or np.isnan(cb[i]) or np.isnan(target[i]):
            valid[i] = False
            continue
        rc = _RC_SHIFTS[aa]
        rc_cacb[i] = rc[_RC_CA] - rc[_RC_CB]
        rc_target[i] = rc[target_rc_col]

    idx = np.where(valid)[0]
    if len(idx) < _MIN_RESIDUES:
        logger.debug("LACS: fewer than %d valid residues, skipping", _MIN_RESIDUES)
        return None

    # Compute X and Y
    X = (ca[idx] - cb[idx]) - rc_cacb[idx]
    Y = target[idx] - rc_target[idx]

    # Mahalanobis outlier removal
    inlier_mask = _mahalanobis_outlier_removal(X, Y, threshold_frac=0.80)
    X_clean = X[inlier_mask]
    Y_clean = Y[inlier_mask]

    if len(X_clean) < _MIN_RESIDUES:
        return None

    # Two-line robust regression
    # MATLAB LACS uses atom-specific thresholds: ths = [1 1 1 6 6 6 6]
    # indexed via ths(x_original): CA=1, CB=1, HA=6, CO=6
    threshold = two_line_threshold
    sel1 = X_clean < threshold
    sel2 = X_clean > -threshold

    if sel1.sum() < _MIN_POINTS_PER_LINE or sel2.sum() < _MIN_POINTS_PER_LINE:
        # Fall back to single robust fit
        intercept, slope, _ = _robustfit(X_clean, Y_clean)
        offset = round(intercept * 100) / 100
        return offset

    intercept1, slope1, _ = _robustfit(X_clean[sel1], Y_clean[sel1])
    intercept2, slope2, _ = _robustfit(X_clean[sel2], Y_clean[sel2])

    # Offset = mean intercept (positive means obs is too high → subtract to correct)
    offset = round(np.mean([intercept1, intercept2]) * 100) / 100
    return offset


# ---------------------------------------------------------------------------
# 15N / 1HN offset: single robust regression with constraints (ordN.m)
# ---------------------------------------------------------------------------


def _compute_n_offset(
    seq: str,
    seq_nums: np.ndarray,
    obs_ca: np.ndarray,
    obs_cb: np.ndarray,
    obs_target: np.ndarray,
    target_type: str,
) -> float | None:
    """Compute referencing offset for 15N or 1HN.

    Uses the preceding-residue correlation from LACS ordN.m.
    X-axis: (CA-CB)_secondary of residue i-1.
    Y-axis: secondary shift of N or HN at residue i, corrected for
    preceding-residue effect.

    Parameters
    ----------
    seq : protein sequence
    seq_nums : residue numbers (1-based)
    obs_ca, obs_cb : observed CA/CB shifts (for preceding residue)
    obs_target : observed H or N shifts
    target_type : "H" or "N"

    Returns
    -------
    offset in ppm, or None if insufficient data
    """
    n = len(seq)
    target_rc_col = _RC_H if target_type == "H" else _RC_N
    ncorr_col = _NCORR_H if target_type == "H" else _NCORR_N

    # Work on copies for pre-Pro correction
    ca = obs_ca.copy()
    cb = obs_cb.copy()
    target = obs_target.copy()

    # Apply pre-Pro corrections
    for i in range(n - 1):
        if (
            seq[i] in _PRE_PRO
            and i + 1 < n
            and seq[i + 1] == "P"
            and seq_nums[i + 1] == seq_nums[i] + 1
        ):
            corr = _PRE_PRO[seq[i]]
            if not np.isnan(ca[i]):
                ca[i] += corr[_RC_CA]
            if not np.isnan(cb[i]):
                cb[i] += corr[_RC_CB]
            if not np.isnan(target[i]):
                target[i] += corr[_RC_H if target_type == "H" else _RC_N]

    # Build X (preceding residue CA-CB secondary) and Y (target secondary - Ncorr)
    X_list = []
    Y_list = []
    indices = []

    for i in range(1, n):
        aa_curr = seq[i]
        aa_prev = seq[i - 1]

        # Skip if not sequential
        if seq_nums[i] != seq_nums[i - 1] + 1:
            continue
        # Skip if either AA unknown
        if aa_curr not in _RC_SHIFTS or aa_prev not in _RC_SHIFTS:
            continue
        # Skip Pro at current position (no HN/N)
        if aa_curr == "P":
            continue
        # Skip Gly and His as preceding residues (ordN.m filter)
        if aa_prev in _EXCLUDE_PRECEDING_N:
            continue
        # Need CA, CB at i-1 and target at i
        if np.isnan(ca[i - 1]) or np.isnan(cb[i - 1]) or np.isnan(target[i]):
            continue

        rc_prev = _RC_SHIFTS[aa_prev]
        x_val = (ca[i - 1] - cb[i - 1]) - (rc_prev[_RC_CA] - rc_prev[_RC_CB])

        # Filter extreme X values
        if abs(x_val) > 10:
            continue

        rc_curr = _RC_SHIFTS[aa_curr]
        y_raw = target[i] - rc_curr[target_rc_col]

        # Apply preceding-residue correction
        ncorr = _NCORR.get(aa_prev, (0.0, 0.0))[ncorr_col]
        y_val = y_raw - ncorr

        X_list.append(x_val)
        Y_list.append(y_val)
        indices.append(i)

    if len(X_list) < _MIN_RESIDUES:
        logger.debug("LACS N/HN: fewer than %d valid pairs, skipping", _MIN_RESIDUES)
        return None

    X = np.array(X_list)
    Y = np.array(Y_list)

    # Phase A: Mahalanobis outlier removal (stricter threshold: 0.99)
    inlier_mask = _mahalanobis_outlier_removal(X, Y, threshold_frac=0.99)
    X_clean = X[inlier_mask]
    Y_clean = Y[inlier_mask]

    if len(X_clean) < _MIN_RESIDUES:
        return None

    # Phase B: Robust fit with weight-based outlier removal
    intercept, slope, weights = _robustfit(X_clean, Y_clean)

    # Iteratively remove points with weight < 0.2
    mask_b = np.ones(len(X_clean), dtype=bool)
    for _ in range(len(X_clean)):
        if weights[mask_b].size == 0:
            break
        min_w_idx = np.argmin(weights[mask_b])
        if weights[mask_b][min_w_idx] >= 0.2:
            break
        # Find the actual index in X_clean
        active_indices = np.where(mask_b)[0]
        mask_b[active_indices[min_w_idx]] = False
        if mask_b.sum() < _MIN_RESIDUES:
            break
        intercept, slope, weights_new = _robustfit(X_clean[mask_b], Y_clean[mask_b])
        # Map new weights back to full array
        weights = np.zeros(len(X_clean))
        weights[mask_b] = weights_new

    # Constrained slope check
    n_clean = mask_b.sum()
    ec = np.sum(X_clean[mask_b] < -2)  # strand count
    hc = np.sum(X_clean[mask_b] > 2)  # helix count

    if target_type == "N":
        expected_slope = -0.4
        slope_tol = 0.1
        slope_bounds = (-0.45, -0.35)
    else:  # H
        expected_slope = -0.07
        slope_tol = 0.02
        slope_bounds = (-0.08, -0.06)

    needs_constraint = abs(slope - expected_slope) > slope_tol and (
        min(ec, hc) < 0.15 * n_clean or n_clean < 66
    )

    if needs_constraint and n_clean >= _MIN_RESIDUES:
        # Constrained fit: fix slope within bounds, solve for intercept
        # Simplified version of lsqcurvefit: grid search over slope range
        best_rss = np.inf
        best_intercept = intercept
        X_fit = X_clean[mask_b]
        Y_fit = Y_clean[mask_b]
        for test_slope in np.linspace(slope_bounds[0], slope_bounds[1], 100):
            test_intercept = np.mean(Y_fit - test_slope * X_fit)
            rss = np.sum((Y_fit - test_intercept - test_slope * X_fit) ** 2)
            if rss < best_rss:
                best_rss = rss
                best_intercept = test_intercept
        intercept = best_intercept

    # Positive offset = obs too high → subtract to correct.
    #
    # NOTE (issue #20): the reference ordN.m added a fixed post-fit constant
    # here (+0.465 ppm for N, +0.049 for HN). That constant appears in no LACS
    # publication — Wang & Markley 2009 defines the offset as the bare (negative)
    # intercept — and an empirical check on 2000 BMRB entries showed it *induces*
    # a bias: the aligned N offset vs PANAV sits at -0.489 ppm with the constant
    # and collapses to -0.024 ppm without it. It is therefore dropped, matching
    # the published definition.
    offset = intercept
    offset = round(offset * 100) / 100
    return offset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_lacs_offsets(
    seq: str,
    seq_nums: np.ndarray,
    obs_shifts: dict[str, np.ndarray],
) -> dict[str, float | None]:
    """Compute LACS referencing offsets for all available atom types.

    Parameters
    ----------
    seq : str
        Protein sequence (single-letter amino acid codes, length n).
    seq_nums : np.ndarray
        Residue numbers (1-based, shape (n,)).  Must be sequential where
        residues are consecutive in the chain (gaps indicated by non-unit
        increments).
    obs_shifts : dict
        Observed chemical shifts keyed by atom type.  Each value is a
        float array of shape (n,) with NaN for missing assignments.
        Expected keys: any subset of {"CA", "CB", "C", "HA", "H", "N"}.

    Returns
    -------
    dict mapping atom type ("CA", "CB", "C", "HA", "H", "N", "HB") to
    offset in ppm (positive = observed too high), or None if insufficient
    data.  The offset should be **subtracted** from observed shifts to
    correct them.

    Notes
    -----
    CA and CB offsets should be identical (shared 13C reference).
    HB has no direct LACS relationship, but it is a 1H nucleus like HA,
    so any 1H referencing error detected via HA applies equally to HB.
    The HB offset is therefore set equal to the HA offset.

    Examples
    --------
    >>> offsets = compute_lacs_offsets(seq, seq_nums, {
    ...     "CA": ca_arr, "CB": cb_arr, "C": co_arr,
    ...     "HA": ha_arr, "H": h_arr, "N": n_arr,
    ... })
    >>> offsets["CA"]  # e.g., -0.13 means CA shifts are 0.13 ppm too low
    """
    ca = obs_shifts.get("CA")
    cb = obs_shifts.get("CB")

    offsets = {
        "CA": None,
        "CB": None,
        "C": None,
        "HA": None,
        "H": None,
        "N": None,
        "HB": None,
    }

    # 13C and HA analysis requires both CA and CB
    if ca is not None and cb is not None:
        # CA offset
        offsets["CA"] = _compute_13c_offset(seq, seq_nums, ca, cb, ca, _RC_CA)
        # CB offset
        offsets["CB"] = _compute_13c_offset(seq, seq_nums, ca, cb, cb, _RC_CB)

        # CO (C') offset — uses wider threshold (6 ppm) per MATLAB LACS
        co = obs_shifts.get("C")
        if co is not None:
            offsets["C"] = _compute_13c_offset(
                seq, seq_nums, ca, cb, co, _RC_CO, two_line_threshold=6.0
            )

        # HA offset — uses wider threshold (6 ppm) per MATLAB LACS
        ha = obs_shifts.get("HA")
        if ha is not None:
            offsets["HA"] = _compute_13c_offset(
                seq, seq_nums, ca, cb, ha, _RC_HA, two_line_threshold=6.0
            )

        # H (amide) offset — uses preceding residue correlation
        h = obs_shifts.get("H")
        if h is not None:
            offsets["H"] = _compute_n_offset(seq, seq_nums, ca, cb, h, "H")

        # N offset — uses preceding residue correlation
        n_shifts = obs_shifts.get("N")
        if n_shifts is not None:
            offsets["N"] = _compute_n_offset(seq, seq_nums, ca, cb, n_shifts, "N")

        # HB — no direct LACS relationship, but HB is a 1H nucleus like HA,
        # so any 1H referencing error detected via HA applies equally to HB.
        offsets["HB"] = offsets["HA"]
    else:
        logger.debug("LACS: CA and CB both required; skipping entry")

    return offsets
