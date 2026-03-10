import logging
import warnings

import numpy as np
import pandas as pd
import scipy

import trizod.bmrb.bmrb as bmrb
from trizod.constants import BBATNS, REFINED_WEIGHTS  # , Z_CORRECTION


def chi2_cdf_approx(rss, k):
    """Wilson-Hilferty approximation to the chi-squared CDF."""
    with np.errstate(divide="ignore", invalid="ignore"):
        # RuntimeWarnings expected: k can be 0
        result = (
            (
                ((rss / k) ** (1.0 / 6))
                - 0.50 * ((rss / k) ** (1.0 / 3))
                + 1.0 / 3 * ((rss / k) ** (1.0 / 2))
            )
            - (5.0 / 6 - 1.0 / 9 / k - 7.0 / 648 / (k**2) + 25.0 / 2187 / (k**3))
        ) / np.sqrt(1.0 / 18 / k + 1.0 / 162 / (k**2) - 37.0 / 11664 / (k**3))
    return result


def compare_to_predicted(predshiftdct, bbshifts_arr, bbshifts_mask):
    """Compute observed - predicted shift differences."""
    predshift_arr = np.zeros(shape=bbshifts_arr.shape)
    predshift_mask = np.full(shape=bbshifts_mask.shape, fill_value=False)
    for res, aa in predshiftdct:
        i = res - 1
        for j, atom_type in enumerate(BBATNS):
            if (
                atom_type in predshiftdct[(res, aa)]
                and predshiftdct[(res, aa)][atom_type] is not None
            ):
                predshift_arr[i, j] = predshiftdct[(res, aa)][atom_type]
                predshift_mask[i, j] = True
    diff_arr = np.subtract(
        bbshifts_arr,
        predshift_arr,
        where=bbshifts_mask & predshift_mask,
        out=bbshifts_arr,
    )
    return diff_arr, BBATNS, bbshifts_mask & predshift_mask


def compute_running_offsets(diff_arr, mask, min_AIC=999.0):
    weights = np.array([REFINED_WEIGHTS[atom_type] for atom_type in BBATNS])
    weighted_diffs = diff_arr / weights
    df = pd.DataFrame(weighted_diffs).mask(~mask)
    # compute rolling standard deviation over detected shifts
    # (missing values are ignored and stretched by rolling window)
    per_atom_rolling_stds = []
    per_atom_rolling_offsets = []
    per_atom_rolling_stds_raw = []
    for i in range(
        7
    ):  # TODO: only the selected position would suffice for per_atom_rolling_stds
        roll = df[i].dropna().rolling(9, center=True)
        per_atom_rolling_stds.append(roll.std(ddof=0))
        per_atom_rolling_offsets.append(roll.mean())
        per_atom_rolling_stds_raw.append(
            np.sqrt(per_atom_rolling_stds[-1] ** 2 + per_atom_rolling_offsets[-1] ** 2)
        )
    rolling_stds = pd.concat(per_atom_rolling_stds, axis=1).reindex(
        pd.Index(list(range(len(diff_arr))))
    )
    rolling_offsets = pd.concat(per_atom_rolling_offsets, axis=1).reindex(
        pd.Index(list(range(len(diff_arr))))
    )
    rolling_stds_raw = pd.concat(per_atom_rolling_stds_raw, axis=1).reindex(
        pd.Index(list(range(len(diff_arr))))
    )
    # get index with the lowest mean rolling stddev
    # for which all atom types were detected anywhere in this sample
    mean_rolling_stds = (
        rolling_stds[rolling_stds.columns[mask.any(axis=0)]].dropna(axis=0).mean(axis=1)
    )
    try:
        best_idx = mean_rolling_stds.idxmin()
    except ValueError:
        return None  # still not found

    offset_dict = {}
    for col in rolling_stds.dropna(how="all", axis=1).columns:
        atom_type = BBATNS[col]
        rolling_offset = rolling_offsets.loc[best_idx][col]
        std_raw = rolling_stds_raw.loc[best_idx][col]
        std_corrected = rolling_stds.loc[best_idx][col]
        # difference in Akaike's information criterion, 9 is width of window
        delta_AIC = np.log(std_raw / std_corrected) * 9 - 1
        logging.getLogger("trizod.scoring").info(
            f"minimum running average: {atom_type} {rolling_offset} {delta_AIC}"
        )
        if delta_AIC > min_AIC:
            logging.getLogger("trizod.scoring").info(
                f"using offset correction: {atom_type} {rolling_offset} {delta_AIC}"
            )
            offset_dict[atom_type] = rolling_offset
        else:
            logging.getLogger("trizod.scoring").info(
                f"rejecting offset correction due to low delta_AIC: {atom_type} {rolling_offset} {delta_AIC}"
            )

    return offset_dict


def compute_offsets(weighted_diffs, accepted_mask, min_AIC=999.0):
    atom_counts = np.sum(accepted_mask, axis=0)
    # RuntimeWarnings expected: accepted_mask can contain fully-False columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        new_offsets = np.nanmean(weighted_diffs, axis=0, where=accepted_mask)
        std_uncorrected = np.sqrt(
            np.nanmean(weighted_diffs**2, axis=0, where=accepted_mask)
        )
        std_corrected = np.nanstd(weighted_diffs, axis=0, where=accepted_mask)
        with np.errstate(divide="ignore"):
            # for atom_counts == 1, std_corrected is 0, resulting in inf delta_AIC;
            # not problematic since all offsets with atom_counts < 4 are rejected anyway
            delta_AIC = np.log(std_uncorrected / std_corrected) * atom_counts - 1
    reject_mask = (delta_AIC < min_AIC) | (atom_counts < 4)
    std_corrected[reject_mask] = std_uncorrected[reject_mask]
    new_offsets[reject_mask] = 0.0
    new_offsets = dict(zip(BBATNS, new_offsets))
    return new_offsets


def get_outlier_mask(
    zscores_triplet, zscores, abs_weighted_diffs, mask, cdf_threshold=6.0
):
    atom_outliers = abs_weighted_diffs > cdf_threshold
    shift_counts = mask.sum(axis=1)
    final_outliers = (zscores > cdf_threshold) | (
        (zscores_triplet > cdf_threshold) & (zscores > 0.0) & (shift_counts > 0)
    )
    outlier_mask = mask & (
        np.bitwise_or(np.expand_dims(final_outliers, axis=1), atom_outliers)
    )
    return outlier_mask


def compute_weighted_diffs(diff_arr, mask, offset_dict=None):
    if offset_dict is None:
        offset_dict = {}
    weights = np.array([REFINED_WEIGHTS[atom_type] for atom_type in BBATNS])
    offsets = np.array([offset_dict.get(atom_type, 0.0) for atom_type in BBATNS])
    weighted_diffs = diff_arr / weights
    # copy needed: weighted_diffs is reused later, subtract with out= would overwrite it
    abs_weighted_diffs = weighted_diffs.copy()
    abs_weighted_diffs = np.abs(
        np.subtract(weighted_diffs, offsets, where=mask, out=abs_weighted_diffs)
    )
    return weighted_diffs, abs_weighted_diffs


def compute_zscores(diffs, dof, mask, corr=False):
    indices = np.where(np.any(mask, axis=1))
    first_idx, last_idx = indices[0][0], indices[0][-1]
    rss = (np.minimum(diffs, 4.0) ** 2).sum(axis=1)
    zscores = chi2_cdf_approx(rss, dof)
    if corr:
        raise ValueError("Z_CORRECTION is not supported")
    zscores[:first_idx] = np.nan
    zscores[last_idx + 1 :] = np.nan
    return zscores


def compute_pscores(diffs, dof, mask, quotient=2.0, limit=4.0):
    indices = np.where(np.any(mask, axis=1))
    first_idx, last_idx = indices[0][0], indices[0][-1]

    if limit:
        p = np.prod(
            scipy.stats.norm.pdf(np.minimum(diffs, limit) / quotient)
            / scipy.stats.norm.pdf(0.0),
            axis=1,
        )
        with np.errstate(
            divide="ignore"
        ):  # zeros are to be expected; resulting NANs are ok
            p = p ** (1 / dof)
        minimum = scipy.stats.norm.pdf(limit / quotient) / scipy.stats.norm.pdf(0.0)
        p = (p - minimum) / (1.0 - minimum)
    else:
        p = np.prod(
            scipy.stats.norm.pdf(diffs / quotient) / scipy.stats.norm.pdf(0.0), axis=1
        )
        with np.errstate(
            divide="ignore"
        ):  # zeros are to be expected; resulting NANs are ok
            p = p ** (1 / dof)
    p[dof == 0] = np.nan
    p[:first_idx] = np.nan
    p[last_idx + 1 :] = np.nan
    return p


def convert_to_triplet_data(abs_weighted_diffs, mask):
    triplet_diffs = abs_weighted_diffs.copy()
    triplet_diffs[~mask] = 0.0
    triplet_diffs = np.column_stack(
        [
            np.pad(triplet_diffs, ((1, 1), (0, 0)))[2:],
            triplet_diffs,
            np.pad(triplet_diffs, ((1, 1), (0, 0)))[:-2],
        ]
    )
    triplet_dof = np.column_stack(
        [np.pad(mask, ((1, 1), (0, 0)))[2:], mask, np.pad(mask, ((1, 1), (0, 0)))[:-2]]
    ).sum(axis=1)
    return triplet_diffs, triplet_dof


def get_offset_corrected_shifts(seq, shifts, predshiftdct):
    # get polymer sequence and chemical backbone shifts
    ret = bmrb.get_valid_bbshifts(shifts, seq)
    if ret is None:
        logging.getLogger("trizod.scoring").error("retrieving backbone shifts failed")
        return
    bbshifts_arr, bbshifts_mask = ret

    # compare predicted to actual shifts
    diff_arr, _, cmp_mask = compare_to_predicted(
        predshiftdct, bbshifts_arr, bbshifts_mask
    )
    total_backbone_shifts = np.sum(cmp_mask)
    if total_backbone_shifts == 0:
        logging.getLogger("trizod.scoring").error("no comparable backbone shifts")
        return
    logging.getLogger("trizod.scoring").info(
        f"total number of backbone shifts: {total_backbone_shifts}"
    )

    offsets_initial = dict.fromkeys(BBATNS, 0.0)
    weighted_diffs_initial, abs_weighted_diffs_initial = compute_weighted_diffs(
        diff_arr, cmp_mask, offsets_initial
    )
    zscores_initial = compute_zscores(
        abs_weighted_diffs_initial, cmp_mask.sum(axis=1), cmp_mask
    )
    zscores_triplet_initial = compute_zscores(
        *convert_to_triplet_data(abs_weighted_diffs_initial, cmp_mask), cmp_mask
    )
    outlier_mask_initial = get_outlier_mask(
        zscores_triplet_initial,
        zscores_initial,
        abs_weighted_diffs_initial,
        cmp_mask,
        cdf_threshold=6.0,
    )
    new_offsets_initial = compute_offsets(
        weighted_diffs_initial, cmp_mask & ~outlier_mask_initial, min_AIC=6.0
    )
    mean_zscore_initial = np.nanmean(zscores_triplet_initial)
    offsets_final = new_offsets_initial
    outlier_mask_final = outlier_mask_initial

    offsets_running = compute_running_offsets(diff_arr, cmp_mask, min_AIC=6.0)
    if offsets_running is None:
        logging.getLogger("trizod.scoring").warning(
            "no running offset could be estimated"
        )
    elif np.any([v != 0.0 for v in offsets_running.values()]):
        weighted_diffs_corrected, abs_weighted_diffs_corrected = compute_weighted_diffs(
            diff_arr, cmp_mask, offsets_running
        )
        zscores_corrected = compute_zscores(
            abs_weighted_diffs_corrected, cmp_mask.sum(axis=1), cmp_mask
        )
        zscores_triplet_corrected = compute_zscores(
            *convert_to_triplet_data(abs_weighted_diffs_corrected, cmp_mask), cmp_mask
        )
        mean_zscore_corrected = np.nanmean(zscores_triplet_corrected)
        if (
            mean_zscore_initial >= mean_zscore_corrected
        ):  # use offset correction only if it improves accordance with the POTENCI model
            outlier_mask_corrected = get_outlier_mask(
                zscores_triplet_corrected,
                zscores_corrected,
                abs_weighted_diffs_corrected,
                cmp_mask,
                cdf_threshold=6.0,
            )
            new_offsets_corrected = compute_offsets(
                weighted_diffs_corrected,
                cmp_mask & ~outlier_mask_corrected,
                min_AIC=6.0,
            )
            offsets_final = new_offsets_corrected
            outlier_mask_final = outlier_mask_corrected

    weighted_diffs_final, abs_weighted_diffs_final = compute_weighted_diffs(
        diff_arr, cmp_mask, offsets_final
    )
    return (
        weighted_diffs_final,
        abs_weighted_diffs_final,
        cmp_mask,
        outlier_mask_final,
        offsets_final,
        weighted_diffs_initial,
        abs_weighted_diffs_initial,
        outlier_mask_initial,
        offsets_initial,
    )
