"""TriZOD scoring pipeline — importable orchestration logic.

The stateless pipeline functions extracted from ``trizod.trizod`` so the
pipeline can be imported and tested without the CLI. Covers BMRB file discovery
and loading, pre/post filtering, score computation, and the filter-loss report.
``trizod.trizod`` re-exports these names for backward compatibility.

The DataFrame-builder functions that depend on the module-level ``bmrb_entries``
state (``fill_row_data``, ``create_peptide_dataframe``, ``compute_scores_row``)
deliberately remain in ``trizod.trizod`` until that global is removed.
"""

import logging
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import trizod.bmrb.bmrb as bmrb
import trizod.potenci.potenci as potenci
import trizod.scoring.scoring as scoring
from trizod.cache import load_potenci_cache, save_potenci_cache
from trizod.constants import BACKBONE_ATOMS, CANONICAL_AA_MASK
from trizod.io.atomic import atomic_write
from trizod.provenance import scoring_cache_version


class ZscoreComputationError(Exception):
    pass


def find_bmrb_files(input_dir, pattern=r"bmr(\d+)_3\.str"):
    """
    If the given path contains at least one bmr<id>_3.str file, only files in this directory are returned.
    Else, all subdirectories are searched for bmr<id>_3.str files.
    """
    bmrb_files = {}
    input_dir = Path(input_dir)
    for p in input_dir.iterdir():
        match = re.fullmatch(pattern, p.name)
        if match is not None:
            bmrb_files[match.group(1)] = input_dir / match.group(0)
    if not bmrb_files:
        # try finding BMRB files in subdirectories instead
        for d in [p for p in input_dir.iterdir() if p.is_dir()]:
            for p in d.iterdir():
                match = re.fullmatch(pattern, p.name)
                if match is not None:
                    bmrb_files[match.group(1)] = d / match.group(0)
    return bmrb_files


def parse_bmrb_file(row, cache_dir=None):
    try:
        entry = bmrb.BmrbEntry(row.name, row.dir)
    except Exception:
        return None
    if cache_dir:
        with open(row.cache_fp, "wb") as f:
            pickle.dump(entry, f)
    return entry


def load_bmrb_entries(bmrb_files, cache_dir=None):
    entries, failed = {}, []
    columns = ["entry", "dir"]
    # read cached data
    if cache_dir:
        columns.append("cache_fp")
        cache_base = Path(cache_dir) / "bmrb_entries"
        for id_, fp in bmrb_files.items():
            cache_fp = cache_base / f"{id_}.pkl"
            entry = None
            if cache_fp.exists():
                try:
                    with cache_fp.open("rb") as f:
                        entry = pickle.load(f)
                except Exception:
                    logging.getLogger("trizod.bmrb").debug(
                        f"cache file {cache_fp} corrupt or formatted wrong"
                    )
            entries[id_] = (entry, fp.parent, cache_fp)
    else:
        for id_, fp in bmrb_files.items():
            entries[id_] = (None, fp.parent)
    df = pd.DataFrame(entries.values(), index=entries.keys(), columns=columns)
    sel = pd.isna(df.entry)
    if not df.loc[sel].empty:
        df.loc[sel, "entry"] = df.loc[sel].parallel_apply(
            parse_bmrb_file, axis=1, cache_dir=cache_dir
        )
    sel = pd.isna(df.entry)
    failed = df.loc[sel].index.to_list()
    return df.loc[~sel], failed


def prefilter_dataframe(
    df,
    method_whitelist,
    method_blacklist,
    temperature_range,
    ionic_strength_range,
    pH_range,
    peptide_length_range,
    min_backbone_shift_types,
    min_backbone_shift_positions,
    min_backbone_shift_fraction,
    max_noncanonical_fraction,
    max_x_fraction,
    keywords,
    chemical_denaturants,
    exclude_paramagnetic=False,
):
    missing_vals = ~df[
        ["exp_method", "temperature", "ionic_strength", "pH", "seq", "total_bbshifts"]
    ].isna().any(axis=1)
    method_sel = df.exp_method.str.lower().str.contains("nmr")
    whitelist_lower = [entry.lower() for entry in method_whitelist]
    if whitelist_lower:
        method_sel &= df.exp_method_subtype.str.lower().str.contains(
            "|".join(whitelist_lower), regex=True
        )
    else:
        method_sel = False
    blacklist_lower = [entry.lower() for entry in method_blacklist]
    if blacklist_lower:
        method_sel &= ~df.exp_method_subtype.str.lower().str.contains(
            "|".join(blacklist_lower), regex=True
        )
    if "" in whitelist_lower and "" not in blacklist_lower:
        method_sel |= df.exp_method.str.lower().str.contains("nmr") & pd.isna(
            df.exp_method_subtype
        )
    else:
        # method_sel = sels_pre["method (sub-)type"].fillna(False)
        method_sel &= ~pd.isna(df.exp_method_subtype)
        missing_vals &= ~pd.isna(df.exp_method_subtype)
    sels_pre = {
        # "missing values" : ~df[['ionic_strength', 'pH', 'temperature','seq','total_bbshifts', 'bbshift_types']].isna().any(axis=1),
        ("method (sub-)type", ""): method_sel,
        ("temperature", f"{list(temperature_range)}"): (
            df.temperature >= temperature_range[0]
        )
        & (df.temperature <= temperature_range[1]),
        ("ionic strength", f"{list(ionic_strength_range)}"): (
            df.ionic_strength >= ionic_strength_range[0]
        )
        & (df.ionic_strength <= ionic_strength_range[1]),
        ("pH", f"{list(pH_range)}"): (df.pH >= pH_range[0]) & (df.pH <= pH_range[1]),
        ("peptide length", f"{list(peptide_length_range)}"): (
            df.seq.str.len() >= peptide_length_range[0]
        )
        & (df.seq.str.len() <= peptide_length_range[1]),
        ("bb shift types", f"[{min_backbone_shift_types}, inf]"): (
            df.bbshift_types >= min_backbone_shift_types
        ),
        ("bb shift positions", f"[{min_backbone_shift_positions}, inf]"): (
            df.bbshift_positions >= min_backbone_shift_positions
        ),
        ("bb shift fraction", f"[{min_backbone_shift_fraction}, inf]"): (
            (df.bbshift_positions / df.seq.str.len()) >= min_backbone_shift_fraction
        ),
        ("non-canonical frac", f"[0, {max_noncanonical_fraction}]"): (
            (
                1.0
                - df.seq.str.translate(CANONICAL_AA_MASK).str.count("#")
                / df.seq.str.len()
            )
            <= max_noncanonical_fraction
        ),
        ("X fraction", f"[0, {max_x_fraction}]"): (
            df.seq.str.count("X") / df.seq.str.len() <= max_x_fraction
        ),
    }
    sels_kws = {keyword: ~df[keyword] for keyword in keywords}
    sels_denat = {denaturant: ~df[denaturant] for denaturant in chemical_denaturants}
    sels_paramag = {}
    if exclude_paramagnetic:
        sels_paramag = {"paramagnetic": ~df["paramagnetic"].astype(bool)}
    sels_all_pre = (
        {k[0]: v for k, v in sels_pre.items()} | sels_kws | sels_denat | sels_paramag
    )

    passing = missing_vals.copy()
    for _filter, sel in sels_all_pre.items():
        passing &= sel  # passes all filters

    df["pass_pre"] = False
    df.loc[passing, "pass_pre"] = True
    return df, missing_vals, sels_pre, sels_kws, sels_denat, sels_paramag, sels_all_pre


def postfilter_dataframe(
    df,
    min_backbone_shift_types,
    min_backbone_shift_positions,
    min_backbone_shift_fraction,
    reject_shift_type_only,
    score_types,
):
    comp_error = np.full((len(df),), False)
    for score_type in score_types:
        comp_error |= pd.isna(df[score_type])
    sels_post = {
        ("bb shift types", f"[{min_backbone_shift_types}, inf]"): (
            df.bbshift_types_post >= min_backbone_shift_types
        ),
        ("bb shift positions", f"[{min_backbone_shift_positions}, inf]"): (
            df.bbshift_positions_post >= min_backbone_shift_positions
        ),
        ("bb shift fraction", f"[{min_backbone_shift_fraction}, inf]"): (
            (df.bbshift_positions_post / df.seq.str.len())
            >= min_backbone_shift_fraction
        ),
        ("error in computation", ""): (~comp_error),
    }
    if not reject_shift_type_only:
        any_offsets_too_large = pd.Series(np.full((df.shape[0],), False))
        for atom_type in scoring.BACKBONE_ATOMS:
            any_offsets_too_large |= pd.isna(df[f"off_{atom_type}"])
        sels_post.update({("rejected due to any offset", ""): ~any_offsets_too_large})

    sels_off = {
        f"off_{atom_type}": ~pd.isna(df[f"off_{atom_type}"])
        for atom_type in BACKBONE_ATOMS
    }
    sels_all_post = {k[0]: v for k, v in sels_post.items()}  # | sels_off

    passing = df["pass_pre"].copy()
    for _filter, sel in sels_all_post.items():
        passing &= sel  # passes all filters

    df["pass_post"] = False
    df.loc[passing, "pass_post"] = True

    return sels_post, sels_off, sels_all_post


def print_filter_losses(
    df,
    missing_vals,
    sels_pre,
    sels_kws,
    sels_denat,
    sels_paramag,
    sels_all_pre,
    sels_post,
    sels_off,
    sels_all_post,
):
    w_str, w_num = (
        np.max(
            [len(key[0]) + len(key[1]) + 5 for key in sels_all_pre]
            + [len(key[0]) + len(key[1]) + 4 for key in sels_all_post]
            + [40]
        ),
        10,
    )
    total_width = w_str + 2 * w_num + 7 + 8
    print("\nPre-computation filtering results")
    print("=" * total_width)
    print(
        f"{'criterium':>{w_str}} : {'filtered':<{w_num}} {'unique':<{w_num}}"
    )  # {'missing':<{w_num}}")
    for (filter, crit), sel in sels_pre.items():
        uniq = pd.Series(np.full((len(sel),), False))
        for other_filter, other_sel in sels_all_pre.items():
            if other_filter != filter:
                uniq |= ~other_sel
        uniq = ~sel & ~uniq
        s = f"{filter}{'':<{w_str - (len(filter) + len(crit))}}{crit}"
        print(
            f"{s} : {(~sel).sum():>{w_num}} {uniq.sum():>{w_num}}"
        )  # {(uniq & ~missing_vals).sum():>{w_num}}")
    if sels_kws:
        print()
        print(f"{'keyword':>{w_str}} : {'filtered':<{w_num}} {'unique':<{w_num}}")
        for filter, sel in sels_kws.items():
            uniq = pd.Series(np.full((len(sel),), False))
            for other_filter, other_sel in sels_all_pre.items():
                if other_filter != filter:
                    uniq |= ~other_sel
            uniq = ~sel & ~uniq
            print(
                f"{'.*' + filter + '.*':<{w_str}} : {(~sel).sum():>{w_num}} {uniq.sum():>{w_num}}"
            )
    if sels_denat:
        print()
        print(
            f"{'chemical denaturant':>{w_str}} : {'filtered':<{w_num}} {'unique':<{w_num}}"
        )
        for filter, sel in sels_denat.items():
            uniq = pd.Series(np.full((len(sel),), False))
            for other_filter, other_sel in sels_all_pre.items():
                if other_filter != filter:
                    uniq |= ~other_sel
            uniq = ~sel & ~uniq
            print(
                f"{'.*' + filter + '.*':<{w_str}} : {(~sel).sum():>{w_num}} {uniq.sum():>{w_num}}"
            )

    if sels_paramag:
        print()
        for filter, sel in sels_paramag.items():
            uniq = pd.Series(np.full((len(sel),), False))
            for other_filter, other_sel in sels_all_pre.items():
                if other_filter != filter:
                    uniq |= ~other_sel
            uniq = ~sel & ~uniq
            print(f"{filter:<{w_str}} : {(~sel).sum():>{w_num}} {uniq.sum():>{w_num}}")
    print("-" * total_width)
    passing_pre = df["pass_pre"].copy()
    print(
        f"{'total filtered':<{w_str}} : {(~passing_pre).sum():>{w_num}} of {len(df):>{w_num - 3}} ({(~passing_pre).sum() / len(df) * 100.0:>{6}.2f} %)"
    )
    print("=" * total_width)
    print()
    print(
        f"{'remaining for scores computation':<{w_str}} : {(passing_pre).sum():>{w_num}} of {len(df):>{w_num - 3}} ({(passing_pre).sum() / len(df) * 100.0:>{6}.2f} %)"
    )
    print()
    print("\nRejected offsets stats")
    print("=" * total_width)
    print(
        f"{'backbone atom identifier':>{w_str}} : {'rejected':<{w_num}} {'unique':<{w_num}}"
    )
    for filter, sel in sels_off.items():
        uniq = pd.Series(np.full((len(sel),), False))
        for other_filter, other_sel in sels_off.items():
            if other_filter != filter:
                uniq |= ~other_sel
        uniq = ~sel & ~uniq & passing_pre
        print(
            f"{filter[4:]:<{w_str}} : {(~sel & passing_pre).sum():>{w_num}} {uniq.sum():>{w_num}}"
        )
    print("=" * total_width)
    print()
    print("\nPost-computation filtering results")
    print("=" * total_width)
    print(f"{'criterium':>{w_str}} : {'filtered':<{w_num}} {'unique':<{w_num}}")
    for (filter, crit), sel in sels_post.items():
        uniq = pd.Series(np.full((len(sel),), False))
        for other_filter, other_sel in sels_all_post.items():
            if other_filter != filter:
                uniq |= ~other_sel
        uniq = ~sel & ~uniq & passing_pre
        s = f"{filter}{'':<{w_str - (len(filter) + len(crit))}}{crit}"
        print(f"{s} : {(~sel & passing_pre).sum():>{w_num}} {uniq.sum():>{w_num}}")
    print("-" * total_width)
    passing_post = df["pass_post"]
    print(
        f"{'total filtered':<{w_str}} : {(~passing_post & passing_pre).sum():>{w_num}} of {passing_pre.sum():>{w_num - 3}} ({(~passing_post & passing_pre).sum() / passing_pre.sum() * 100.0:>{6}.2f} %)"
    )
    print("=" * total_width)
    print()
    print(
        f"{'final dataset entries':<{w_str}} : {(passing_post).sum():>{w_num}} of {len(df):>{w_num - 3}} ({(passing_post).sum() / len(df) * 100.0:>{6}.2f} %)"
    )


def compute_scores(
    entry,
    stID,
    entity_assemID,
    entityID,
    seq,
    ion,
    pH,
    temperature,
    score_types=None,
    offset_correction=True,
    max_offset=np.inf,
    reject_shift_type_only=False,
    # min_backbone_shift_types=1, min_backbone_shift_positions=1, min_backbone_shift_fraction=0.,
    cache_dir=None,
    rereference_mode="both",
):
    if score_types is None:
        score_types = ["zscores"]
    exe_times = [np.nan, np.nan, np.nan]
    # The scoring-code version is part of the key so that a change to the
    # LACS/scoring math invalidates stale cache entries by construction
    # (issue #20): a key without it silently reused offsets computed by
    # different code.
    shifts_cache_path = (
        cache_dir
        / "wSCS"
        / f"{entry.id}_{stID}_{entity_assemID}_{entityID}_{rereference_mode}"
        f"_v{scoring_cache_version()}.npz"
    )
    if cache_dir and shifts_cache_path.exists():
        try:
            cached = np.load(str(shifts_cache_path))
            weighted_diffs_final = cached["shw"]
            abs_weighted_diffs_final = cached["ashwi"]
            cmp_mask = cached["cmp_mask"]
            outlier_mask_final = cached["olf"]
            offsets_final = cached["offf"]
            weighted_diffs_initial = cached["shw0"]
            abs_weighted_diffs_initial = cached["ashwi0"]
            outlier_mask_initial = cached["ol0"]
            offsets_initial = cached["off0"]
            if "lacs" in cached.files:
                lacs_offsets = dict(zip(BACKBONE_ATOMS, cached["lacs"]))
            else:
                lacs_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
            offsets_final = dict(zip(BACKBONE_ATOMS, offsets_final))
            offsets_initial = dict(zip(BACKBONE_ATOMS, offsets_initial))
        except Exception:
            logging.getLogger("trizod").debug(
                f"cache file {shifts_cache_path} corrupt or formatted wrong, delete and repeat computation"
            )
            shifts_cache_path.unlink()
    if not (cache_dir and shifts_cache_path.exists()):
        peptide_shifts = entry.get_peptide_shifts()
        shifts, condID, assemID, sampleIDs = peptide_shifts[
            (stID, entity_assemID, entityID)
        ]

        try:
            # predict random coil chemical shifts using POTENCI
            use_ph_corr = pH != 7.0
            start_time = time.time()
            predshiftdct = load_potenci_cache(cache_dir, seq, temperature, pH, ion)
            if predshiftdct is None:
                predshiftdct = potenci.get_pred_shifts(
                    seq, temperature, pH, ion, use_ph_corr
                )
                save_potenci_cache(cache_dir, seq, temperature, pH, ion, predshiftdct)
            exe_times[0] = time.time() - start_time
        except Exception as err:
            logging.getLogger("trizod").error(
                f"POTENCI failed for {(entry.id, stID, entity_assemID, entityID)} due to the following error:",
                exc_info=True,
            )
            raise ZscoreComputationError from err
        start_time = time.time()
        ret = scoring.get_offset_corrected_shifts(
            seq, shifts, predshiftdct, rereference_mode=rereference_mode
        )
        if ret is None:
            logging.getLogger("trizod").error(
                f"TriZOD failed for {(entry.id, stID, entity_assemID, entityID)} due to an error in computation of corrected weighted shifts."
            )
            raise ZscoreComputationError
        else:
            exe_times[1] = time.time() - start_time
        (
            weighted_diffs_final,
            abs_weighted_diffs_final,
            cmp_mask,
            outlier_mask_final,
            offsets_final,
            weighted_diffs_initial,
            abs_weighted_diffs_initial,
            outlier_mask_initial,
            offsets_initial,
            lacs_offsets,
        ) = ret
        if cache_dir:
            with atomic_write(shifts_cache_path, "wb") as _cache_fh:
                np.savez(
                    _cache_fh,
                    shw=weighted_diffs_final,
                    ashwi=abs_weighted_diffs_final,
                    cmp_mask=cmp_mask,
                    olf=outlier_mask_final,
                    offf=np.array(
                        [offsets_final[atom_type] for atom_type in BACKBONE_ATOMS]
                    ),
                    shw0=weighted_diffs_initial,
                    ashwi0=abs_weighted_diffs_initial,
                    ol0=outlier_mask_initial,
                    off0=np.array(
                        [offsets_initial[atom_type] for atom_type in BACKBONE_ATOMS]
                    ),
                    lacs=np.array(
                        [lacs_offsets[atom_type] for atom_type in BACKBONE_ATOMS]
                    ),
                )
    offsets = offsets_final
    if not offset_correction:
        abs_weighted_diffs_final = abs_weighted_diffs_initial
        offsets = offsets_initial
    elif not (max_offset is None or np.isinf(max_offset)):
        # check if any offsets are too large
        for i, atom_type in enumerate(BACKBONE_ATOMS):
            if np.abs(offsets_final[atom_type]) > max_offset:
                offsets[atom_type] = np.nan
                if reject_shift_type_only:
                    # mask this backbone shift type, excluding it from scores computation
                    cmp_mask[:, i] = False
    if np.any(cmp_mask):
        start_time = time.time()
        triplet_diffs, triplet_dof = scoring.convert_to_triplet_data(
            abs_weighted_diffs_final, cmp_mask
        )
        scores = []
        for score_type in score_types:
            if score_type == "zscores":
                scores.append(
                    scoring.compute_zscores(triplet_diffs, triplet_dof, cmp_mask)
                )
            elif score_type == "gscores":
                scores.append(
                    scoring.compute_gscores(triplet_diffs, triplet_dof, cmp_mask)
                )
            else:
                raise ValueError
        k = triplet_dof
        exe_times[2] = time.time() - start_time
    else:
        scores, k = (
            [np.full((cmp_mask.shape[0],), np.nan) for i in range(len(score_types))],
            np.full((cmp_mask.shape[0],), np.nan),
        )
    return scores, k, cmp_mask, offsets, exe_times, lacs_offsets
