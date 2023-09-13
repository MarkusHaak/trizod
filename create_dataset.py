import os, sys
import logging
import argparse
import TriZOD.potenci as potenci
import TriZOD.bmrb as bmrb
import TriZOD.trizod as trizod
import numpy as np
import pandas as pd
import traceback
import re
import pickle
from tqdm import tqdm
from pandarallel import pandarallel

class Found(Exception): pass
class ZscoreComputationError(Exception): pass
class OffsetTooLargeException(Exception): pass
class OffsetCausedFilterException(Exception): pass
class FilterException(Exception): pass

def parse_args():
    parser = argparse.ArgumentParser(description='')
    
    io_grp = parser.add_argument_group('Input/Output Options')
    io_grp.add_argument(
        '--input-dir', '-d', default='.',
        help='Directory that is searched recursively for BMRB .str files.')
    io_grp.add_argument(
        '--output-prefix', default='./trizod_dataset',
        help='Prefix (and path) of the created output file.')
    io_grp.add_argument(
        '--output-format', choices=['json', 'csv'], default='csv',
        help='Output file format.')
    io_grp.add_argument(
        '--cache-dir', default='./tmp',
        help='Create and use cache files in the given directory to acelerate repeated execution.')
    io_grp.add_argument(
        '--BMRB-file-pattern', default="bmr(\d+)_3\.str",
        help="regular expression pattern for BMRB files.")
    
    filter_grp = parser.add_argument_group('Filtering Options')
    filter_grp.add_argument(
        '--temperature-range', nargs=2, type=float, default=[273.0, 313.0],
        help='Minimum and maximum temperature in Kelvin.')
    filter_grp.add_argument(
        '--ionic-strength-range', nargs=2, type=float, default=[0., 3.],
        help='Minimum and maximum ionic strength in Mol.')
    filter_grp.add_argument(
        '--pH-range', nargs=2, type=float, default=[3., 11.],
        help='Minimum and maximum pH.')
    filter_grp.add_argument(
        '--no-unit-assumptions', action="store_false",
        help='Do not assume units if they are not given and exclude entries instead.')
    filter_grp.add_argument(
        '--no-unit-corrections', action="store_false",
        help='Do not correct values if units are most likely wrong.')
    filter_grp.add_argument(
        '--no-default-conditions', action="store_false",
        help='Do not assume standard conditions if pH (7), ionic strength (0.1 M) or temperature (298 K) are missing and exclude entries instead.')
    filter_grp.add_argument(
        '--peptide-length-range', nargs='+', type=int, default=[5],
        help='Minimum (and optionally maximum) peptide sequence length.')
    filter_grp.add_argument(
        '--min-backbone-shift-types', type=int, default=3,
        help='Minimum number of different backbone shift types (max 7).')
    filter_grp.add_argument(
        '--min-backbone-shift-positions', type=int, default=5,
        help='Minimum number of positions with at least one backbone shift.')
    filter_grp.add_argument(
        '--min-backbone-shift-fraction', type=float, default=0.,
        help='Minimum fraction of positions with at least one backbone shift.')
    filter_grp.add_argument(
        '--keywords-blacklist', nargs='*', default=['denatur', 'unfold', 'misfold'],#, 'interact', 'bound', 'bind'],
        help='Exclude entries with any of these keywords mentioned anywhere in the BMRB file, case ignored.')
    filter_grp.add_argument(
        '--chemical-denaturants', nargs='*', default=[
            'guanidin', 'GdmCl', 'Gdn-Hcl',
            'urea',
            'BME', '2BME', '2-ME', 'beta-mercaptoethanol'],
        help='Exclude entries with any of these chemicals as substrings of sample components, case ignored.')
    filter_grp.add_argument(
        '--exp-method-whitelist', nargs='*', default=['solution', 'structures'],
        help='Include only entries with any of these keywords as substring of the experiment subtype, case ignored.')
    filter_grp.add_argument(
        '--exp-method-blacklist', nargs='*', default=['state'],
        help='Exclude entries with any of these keywords as substring of the experiment subtype, case ignored.')
    
    scores_grp = parser.add_argument_group('Scoring Options')
    scores_grp.add_argument(
        '--scores-type', choices=['zscores', 'chezod', 'pscores'], default='zscores',
        help='Which type of scores are created: Observation count-independent zscores (zscores), '
        'original CheZOD zscores (chezod) or geometric mean of observation probabilities (pscores).')
    scores_grp.add_argument(
        '--no-offset-correction', action='store_false',
        help='Do not compute correction offsets for random coil chemical shifts')
    scores_grp.add_argument(
        '--max-offset', type=float, default=np.inf,
        help='Maximum valid offset for any random coil chemical shift type.')
    scores_grp.add_argument(
        '--reject-shift-type-only', action='store_true',
        help='Upon exceeding the maximal offset set by <--max-offset>, exclude only the backbone shifts exceeding the offset instead of the whole entry.')


    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()


    if not os.path.exists(args.input_dir):
        logging.error(f"Input directory {args.input_dir} does not exist.")
        exit(1)
    if not os.path.isdir(args.input_dir):
        logging.error(f"Path {args.input_dir} is not a directory.")
        exit(1)
    args.input_dir = os.path.abspath(args.input_dir)

    args.output_prefix = os.path.abspath(args.output_prefix)
    if not os.path.exists(os.path.dirname(args.output_prefix)):
        logging.error(f"Output directory {os.path.dirname(args.output_prefix)} does not exist.")
        exit(1)

    if len(args.peptide_length_range) == 1:
        args.peptide_length_range.append(np.inf)

    args.cache_dir = os.path.abspath(args.cache_dir)
    if not os.path.exists(args.cache_dir):
        logging.debug(f"Directory {args.cache_dir} does not exist and is created.")
        os.makedirs(args.cache_dir)
        os.makedirs(os.path.join(args.cache_dir, 'wSCS'))
    elif not os.path.isdir(args.cache_dir):
        logging.error(f"Path {args.cache_dir} is not a directory.")
        exit(1)

    return args

def find_bmrb_files(input_dir, pattern="bmr(\d+)_3\.str"):
    """
    If the given path contains at least one bmr<id>_3.str file, only files in this directory are returned. 
    Else, all subdirectories are searched for bmr<id>_3.str files.
    """
    bmrb_files = {}
    for p in os.listdir(input_dir):
        m = re.fullmatch(pattern, p)
        if m is not None:
            bmrb_files[m.group(1)] = os.path.join(input_dir, m.group(0))
    if not bmrb_files:
        # try finding BMRB files in subdirectories instead
        for d in [os.path.join(input_dir, p) for p in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, p))]:
            for p in os.listdir(d):
                m = re.fullmatch(pattern, p)
                if m is not None:
                    bmrb_files[m.group(1)] = os.path.join(d, m.group(0))
    return bmrb_files

def load_bmrb_entries(bmrb_files, cache_dir=None):
    bmrb_entries, failed = {}, []
    # read cached data
    if cache_dir:
        bmrb_entries_cache = os.path.join(cache_dir, "bmrb_entries.pkl")
        if os.path.exists(bmrb_entries_cache):
            with open(bmrb_entries_cache, "rb") as f:
                bmrb_entries, failed = pickle.load(f)
    # identify entries missing in the cache and parse them from db files
    missing_bmrb_files = {id_ : fp for id_,fp in bmrb_files.items() if not (id_ in bmrb_entries or id_ in failed)}
    if missing_bmrb_files:
        missing_bmrb_entries, missing_failed = {}, []
        for id_ in tqdm(missing_bmrb_files):
            try:
                entry = bmrb.BmrbEntry(id_, os.path.dirname(missing_bmrb_files[id_]))
            except:
                missing_failed.append(id_)
                continue
            missing_bmrb_entries[id_] = entry
        bmrb_entries.update(missing_bmrb_entries)
        failed.extend(missing_failed)
        # update cache
        if cache_dir:
            with open(bmrb_entries_cache, "wb") as f:
                pickle.dump((bmrb_entries, failed), f)
    # only keep entries of the current file list
    bmrb_entries = {id_:entry for id_,entry in bmrb_entries.items() if id_ in bmrb_files}
    failed = [id_ for id_ in failed if id_ in bmrb_files]
    return bmrb_entries, failed

def prefilter_dataframe(df,
                        method_whitelist, method_blacklist,
                        temperature_range,
                        ionic_strength_range,
                        pH_range,
                        peptide_length_range,
                        min_backbone_shift_types,
                        min_backbone_shift_positions,
                        min_backbone_shift_fraction,
                        keywords,
                        chemical_denaturants,
                        ):
    missing_vals = ~df[['exp_method', 'exp_method_subtype', 'ionic_strength', 'pH', 'temperature', 'seq', 'total_bbshifts']].isna().any(axis=1)
    sels_pre = {
        #"missing values" : ~df[['exp_method', 'exp_method_subtype','ionic_strength', 'pH', 'temperature','seq','total_bbshifts', 'bbshift_types']].isna().any(axis=1),
        "method (sub-)type" : df.exp_method.str.lower().str.contains('nmr') & \
                             (df.exp_method_subtype.str.lower().str.contains("|".join([l.lower() for l in method_whitelist]), regex=True)) & \
                            (~df.exp_method_subtype.str.lower().str.contains("|".join([l.lower() for l in method_blacklist]), regex=True)),
        "temperature" : (df.temperature >= temperature_range[0]) & \
                        (df.temperature <= temperature_range[1]),
        "ionic strength" : (df.ionic_strength >= ionic_strength_range[0]) & \
                           (df.ionic_strength <= ionic_strength_range[1]),
        "pH strength" : (df.pH >= pH_range[0]) & \
                        (df.pH <= pH_range[1]),
        "peptide length" : (df.seq.str.len() >= peptide_length_range[0]) & \
                           (df.seq.str.len() <= peptide_length_range[1]),
        "backbone shift types" : (df.bbshift_types >= min_backbone_shift_types),
        "backbone shift positions" : (df.bbshift_positions >= min_backbone_shift_positions),
        "backbone shift fraction" : ((df.bbshift_positions / df.seq.str.len()) >= min_backbone_shift_fraction),
    }
    sels_kws = {
        kw : ~df[kw] for kw in keywords
    }
    sels_denat = {
        cd : ~df[cd] for cd in chemical_denaturants
    }
    sels_all_pre = sels_pre | sels_kws | sels_denat
    
    passing = missing_vals.copy()
    for filter, sel in sels_all_pre.items():
        passing &= sel # passes all filters
    
    df['pass_pre'] = False
    df.loc[passing, 'pass_pre'] = True
    return df, missing_vals, sels_pre, sels_kws, sels_denat, sels_all_pre

def postfilter_dataframe(df,
                         min_backbone_shift_types,
                         min_backbone_shift_positions,
                         min_backbone_shift_fraction,
                         reject_shift_type_only):
    sels_post = {
        "backbone shift types" : (df.bbshift_types_post >= min_backbone_shift_types),
        "backbone shift positions" : (df.bbshift_positions_post >= min_backbone_shift_positions),
        "backbone shift fraction" : ((df.bbshift_positions_post / df.seq.str.len()) >= min_backbone_shift_fraction),
        "error in computation" : (~pd.isna(df.scores))
    }
    if not reject_shift_type_only:
        any_offsets_too_large = pd.Series(np.full((df.shape[0],), False))
        for at in trizod.BBATNS:
            any_offsets_too_large |= pd.isna(df[f"off_{at}"])
        sels_post.update({"rejected due to any offset" : ~any_offsets_too_large})

    sels_off = {
        f"off_{at}" : ~pd.isna(df[f"off_{at}"]) for at in trizod.BBATNS 
    }
    sels_all_post = sels_post #| sels_off

    passing = df['pass_pre'].copy()
    for filter, sel in sels_all_post.items():
        passing &= sel # passes all filters
    
    df['pass_post'] = False
    df.loc[passing, 'pass_post'] = True

    return sels_post, sels_off, sels_all_post

def print_filter_losses(df, missing_vals, sels_pre, sels_kws, sels_denat, sels_all_pre, sels_post, sels_off, sels_all_post):
    w_str, w_num = np.max([len(key)+4 for key in sels_all_pre] + [len(key)+4 for key in sels_all_post] + [40]), 10
    total_width = (w_str + 3*w_num + 8)
    print("\nPre-computation filtering results")
    print("=" * total_width)
    print(f"{'criterium':>{w_str}} : {'filtered':<{w_num}} {'unique':<{w_num}} {'missing':<{w_num}}")
    for filter, sel in sels_pre.items():
        uniq = pd.Series(np.full((len(sel),), False))
        for other_filter, other_sel in sels_all_pre.items():
            if other_filter != filter:
                uniq |= ~other_sel
        uniq = ~sel & ~uniq
        print(f"{filter:<{w_str}} : {(~sel).sum():>{w_num}} {uniq.sum():>{w_num}} {(uniq & ~missing_vals).sum():>{w_num}}")
    print()
    print(f"{'keyword':>{w_str}} : {'filtered':<{w_num}} {'unique':<{w_num}}")
    for filter, sel in sels_kws.items():
        uniq = pd.Series(np.full((len(sel),), False))
        for other_filter, other_sel in sels_all_pre.items():
            if other_filter != filter:
                uniq |= ~other_sel
        uniq = ~sel & ~uniq
        print(f"{'.*'+filter+'.*':<{w_str}} : {(~sel).sum():>{w_num}} {uniq.sum():>{w_num}}")
    print()
    print(f"{'chemical denaturant':>{w_str}} : {'filtered':<{w_num}} {'unique':<{w_num}}")
    for filter, sel in sels_denat.items():
        uniq = pd.Series(np.full((len(sel),), False))
        for other_filter, other_sel in sels_all_pre.items():
            if other_filter != filter:
                uniq |= ~other_sel
        uniq = ~sel & ~uniq
        print(f"{'.*'+filter+'.*':<{w_str}} : {(~sel).sum():>{w_num}} {uniq.sum():>{w_num}}")
    
    print("-" * total_width)
    passing_pre = df['pass_pre'].copy()
    print(f"{'total filtered':<{w_str}} : {(~passing_pre).sum():>{w_num}} of {len(df):>{w_num-3}} ({(~passing_pre).sum() / len(df) * 100.:>{w_num-1}.2f} %)")
    print("=" * total_width)
    print()
    print(f"{'remaining for scores computation':<{w_str}} : {(passing_pre).sum():>{w_num}} of {len(df):>{w_num-3}} ({(passing_pre).sum() / len(df) * 100.:>{w_num-1}.2f} %)")
    print()
    print("\nRejected offsets stats")
    print("=" * total_width)
    print(f"{'backbone atom identifier':>{w_str}} : {'rejected':<{w_num}} {'unique':<{w_num}}")
    for filter, sel in sels_off.items():
        uniq = pd.Series(np.full((len(sel),), False))
        for other_filter, other_sel in sels_off.items():
            if other_filter != filter:
                uniq |= ~other_sel
        uniq = ~sel & ~uniq & passing_pre
        print(f"{filter[4:]:<{w_str}} : {(~sel & passing_pre).sum():>{w_num}} {uniq.sum():>{w_num}}")
    print("=" * total_width)
    print()
    print("\nPost-computation filtering results")
    print("=" * total_width)
    print(f"{'criterium':>{w_str}} : {'filtered':<{w_num}} {'unique':<{w_num}}")
    for filter, sel in sels_post.items():
        uniq = pd.Series(np.full((len(sel),), False))
        for other_filter, other_sel in sels_all_post.items():
            if other_filter != filter:
                uniq |= ~other_sel
        uniq = ~sel & ~uniq & passing_pre
        print(f"{filter:<{w_str}} : {(~sel & passing_pre).sum():>{w_num}} {uniq.sum():>{w_num}}")
    print("-" * total_width)
    passing_post = df['pass_post']
    print(f"{'total filtered':<{w_str}} : {(~passing_post & passing_pre).sum():>{w_num}} of {passing_pre.sum():>{w_num-3}} ({(~passing_post & passing_pre).sum() / passing_pre.sum() * 100.:>{w_num-1}.2f} %)")
    print("=" * total_width)
    print()
    print(f"{'final dataset entries':<{w_str}} : {(passing_post).sum():>{w_num}} of {len(df):>{w_num-3}} ({(passing_post).sum() / len(df) * 100.:>{w_num-1}.2f} %)")

def fill_row_data(row, chemical_denaturants, keywords,
                  return_default=True, assume_si=True, fix_outliers=True):
    entry = bmrb_entries[row['id']] #row['entry']
    peptide_shifts = entry.get_peptide_shifts()
    shifts, condID, assemID, sampleIDs = peptide_shifts[(row['stID'], row['entity_assemID'], row['entityID'])]
    row['citation_title'] = entry.citation_title
    row['citation_DOI'] = entry.citation_DOI
    row['exp_method'] = entry.exp_method.lower() if entry.exp_method else pd.NA
    row['exp_method_subtype'] = entry.exp_method_subtype.lower() if entry.exp_method_subtype else pd.NA
    row['ionic_strength'] = entry.conditions[condID].get_ionic_strength(return_default=return_default, assume_si=assume_si, fix_outliers=fix_outliers)
    row['pH'] = entry.conditions[condID].get_pH(return_default=return_default)
    row['temperature'] = entry.conditions[condID].get_temperature(return_default=return_default, assume_si=assume_si, fix_outliers=fix_outliers)
    seq = entry.entities[row['entityID']].seq
    row['seq'] = seq
    # retrieve # backbone shifts (H,HA,HB,C,CA,CB,N)
    total_bbshifts, bbshift_types_post, bbshift_positions_post = None, None, None
    if seq:
        ret = bmrb.get_valid_bbshifts(shifts, seq)
        if ret:
            bbshifts, bbshifts_arr, bbshifts_mask = ret
            total_bbshifts = np.sum(bbshifts_mask) # total backbone shifts
            bbshift_types_post = np.any(bbshifts_mask, axis=0).sum() # different backbone shifts
            bbshift_positions_post = np.any(bbshifts_mask, axis=1).sum() # positions with backbone shifts
    row['total_bbshifts'] = total_bbshifts
    row['bbshift_types'] = bbshift_types_post
    row['bbshift_positions'] = bbshift_positions_post
    # check if keywords are present
    fields = [entry.title, entry.details, entry.citation_title,
            entry.assemblies[assemID].name, entry.assemblies[assemID].details, 
            entry.entities[row['entityID']].name, entry.entities[row['entityID']].details]
    if entry.citation_keywords is not None:
        if type(entry.citation_keywords) == list:
            for el in entry.citation_keywords:
                fields.extend(el)
        else:
            fields.extend(entry.citation_keywords)
    if entry.struct_keywords is not None:
        if type(entry.struct_keywords) == list:
            for el in entry.struct_keywords:
                fields.extend(el)
        else:
            fields.extend(entry.struct_keywords)
    for sID in sampleIDs:
        fields.extend([entry.samples[sID].name, entry.samples[sID].details, entry.samples[sID].framecode])
    for keyword in keywords:
        #data[-1].append(False)
        row[keyword] = False
        for field in fields:
            if field: # can be None
                if keyword.lower() in field.lower():
                    row[keyword] = True
                    break
    # check if chemical detergents are present
    for den_comp in chemical_denaturants:
        #data[-1].append(False)
        row[den_comp] = False
        try:
            for sID in sampleIDs:
                for comp in entry.samples[sID].components:
                    if comp[3] and not comp[2]: # if it has a name but no entity entry
                        if comp[3].lower() in den_comp.lower():
                            #data[-1][-1] = True
                            row[den_comp] = True
                            raise Found
        except Found:
            pass
    # add columns that will be filled later
    row['scores'] = None
    row['k'] = None
    row['total_bbshifts_post'] = np.nan
    row['bbshift_types_post'] = np.nan
    row['bbshift_positions_post'] = np.nan
    for at in trizod.BBATNS:
        row[f'off_{at}'] = pd.NA
    return row

def create_peptide_dataframe_parallel(bmrb_entries,
                             chemical_denaturants, keywords,
                             return_default=True, assume_si=True, fix_outliers=True
                             ):
    data = []
    columns = ['id', #'entry', 
               'stID', 'entity_assemID', 'entityID'
            ]

    for id_,entry in tqdm(bmrb_entries.items()):
        peptide_shifts = entry.get_peptide_shifts()
        for (stID, entity_assemID, entityID) in peptide_shifts:
            data.append([])
            data[-1].append(id_)
            #data[-1].append(entry)
            data[-1].extend([stID, entity_assemID, entityID])
    df = pd.DataFrame(data, columns=columns)

    df = df.parallel_apply(fill_row_data, axis=1, args=(chemical_denaturants, keywords), return_default=True, assume_si=True, fix_outliers=True)
    df = df.astype({col : "string" for col in ['id', 'citation_title', 'citation_DOI', 'exp_method', 'exp_method_subtype', 'seq']})
    return df

def create_peptide_dataframe(bmrb_entries,
                             chemical_denaturants, keywords,
                             return_default=True, assume_si=True, fix_outliers=True
                             ):
    data = []
    columns = ['id', 'citation_title', 'citation_DOI',
               'stID', 'entity_assemID', 'entityID',
               'exp_method', 'exp_method_subtype',
               'ionic_strength', 'pH', 'temperature',
               'seq',
               'total_bbshifts', 'bbshift_types', 'bbshift_positions'
            ]
    columns.extend(keywords)
    columns.extend(chemical_denaturants)
    for id_,entry in tqdm(bmrb_entries.items()):
        peptide_shifts = entry.get_peptide_shifts()
        for (stID, entity_assemID, entityID) in peptide_shifts:
            shifts, condID, assemID, sampleIDs = peptide_shifts[(stID, entity_assemID, entityID)]
            data.append([])
            data[-1].append(id_)
            data[-1].append(entry.citation_title)
            data[-1].append(entry.citation_DOI)
            data[-1].extend([stID, entity_assemID, entityID])
            data[-1].append(entry.exp_method.lower() if entry.exp_method else pd.NA)
            data[-1].append(entry.exp_method_subtype.lower() if entry.exp_method_subtype else pd.NA)
            data[-1].append(entry.conditions[condID].get_ionic_strength(return_default=return_default, assume_si=assume_si, fix_outliers=fix_outliers))
            data[-1].append(entry.conditions[condID].get_pH(return_default=return_default))
            data[-1].append(entry.conditions[condID].get_temperature(return_default=return_default, assume_si=assume_si, fix_outliers=fix_outliers))
            seq = entry.entities[entityID].seq
            data[-1].append(seq)
            # retrieve # backbone shifts (H,HA,HB,C,CA,CB,N)
            if seq:
                ret = bmrb.get_valid_bbshifts(shifts, seq)
                if not ret:
                    data[-1].extend([None,None,None])
                else:
                    bbshifts, bbshifts_arr, bbshifts_mask = ret
                    data[-1].append(np.sum(bbshifts_mask)) # total backbone shifts
                    data[-1].append(np.any(bbshifts_mask, axis=0).sum()) # different backbone shifts
                    data[-1].append(np.any(bbshifts_mask, axis=1).sum()) # positions with backbone shifts
            else:
                data[-1].extend([None,None,None])
            # check if keywords are present
            fields = [entry.title, entry.details, entry.citation_title,
                  entry.assemblies[assemID].name, entry.assemblies[assemID].details, 
                  entry.entities[entityID].name, entry.entities[entityID].details]
            if entry.citation_keywords is not None:
                if type(entry.citation_keywords) == list:
                    for el in entry.citation_keywords:
                        fields.extend(el)
                else:
                    fields.extend(entry.citation_keywords)
            if entry.struct_keywords is not None:
                if type(entry.struct_keywords) == list:
                    for el in entry.struct_keywords:
                        fields.extend(el)
                else:
                    fields.extend(entry.struct_keywords)
            for sID in sampleIDs:
                fields.extend([entry.samples[sID].name, entry.samples[sID].details, entry.samples[sID].framecode])
            for keyword in keywords:
                data[-1].append(False)
                for field in fields:
                    if field: # can be None
                        if keyword.lower() in field.lower():
                            data[-1][-1] = True
                            break
            # check if chemical detergents are present
            for den_comp in chemical_denaturants:
                data[-1].append(False)
                try:
                    for sID in sampleIDs:
                        for comp in entry.samples[sID].components:
                            if comp[3] and not comp[2]: # if it has a name but no entity entry
                                if comp[3].lower() in den_comp.lower():
                                    data[-1][-1] = True
                                    raise Found
                except Found:
                    pass

            assert len(data[-1]) == len(columns)
    df = pd.DataFrame(data, columns=columns)
    df = df.astype({col : "string" for col in ['id', 'citation_title', 'citation_DOI', 'exp_method', 'exp_method_subtype', 'seq']})
    # add columns that will be filled later
    df['scores'] = None
    df['k'] = None
    df['total_bbshifts_post'] = np.nan
    df['bbshift_types_post'] = np.nan
    df['bbshift_positions_post'] = np.nan
    for at in trizod.BBATNS:
        df[f'off_{at}'] = pd.NA
    return df

def compute_scores(entry, stID, entity_assemID, entityID,
                   seq, ion, pH, temperature,
                   scores_type='zscores', offset_correction=True, 
                   max_offset=np.inf, reject_shift_type_only=False,
                   #min_backbone_shift_types=1, min_backbone_shift_positions=1, min_backbone_shift_fraction=0.,
                   cache_dir=None):
    wSCS_cache_fp = os.path.join(cache_dir, 'wSCS', f'{entry.id}_{stID}_{entity_assemID}_{entityID}.npz')
    if cache_dir and os.path.exists(wSCS_cache_fp):
        try:
            z = np.load(wSCS_cache_fp)
            shw, ashwi, cmp_mask, olf, offf, shw0, ashwi0, ol0, off0 = z['shw'], z['ashwi'], z['cmp_mask'], z['olf'], z['offf'], z['shw0'], z['ashwi0'], z['ol0'], z['off0']
            offf, off0 = {at:off for at,off in zip(trizod.BBATNS, offf)}, {at:off for at,off in zip(trizod.BBATNS, off0)}
        except:
            logging.debug(f"cache file {wSCS_cache_fp} corrupt or formatted wrong, delete and repeat computation")
            os.remove(wSCS_cache_fp)
    if not (cache_dir and os.path.exists(wSCS_cache_fp)):
        peptide_shifts = entry.get_peptide_shifts()
        shifts, condID, assemID, sampleIDs = peptide_shifts[(stID, entity_assemID, entityID)]
        
        try:
            # predict random coil chemical shifts using POTENCI
            usephcor = (pH != 7.0)
            predshiftdct = potenci.getpredshifts(seq, temperature, pH, ion, usephcor, pkacsvfile=False)
        except:
            logging.error(f"POTENCI failed for {(entry.id, stID, entity_assemID, entityID)} due to the following error:", exc_info=True)
            raise ZscoreComputationError
        ret = trizod.get_offset_corrected_wSCS(seq, shifts, predshiftdct)
        if ret is None:
            logging.error(f'TriZOD failed for {(entry.id, stID, entity_assemID, entityID)} due to an error in computation of corrected wSCSs.')
            raise ZscoreComputationError
        shw, ashwi, cmp_mask, olf, offf, shw0, ashwi0, ol0, off0 = ret
        if cache_dir:
            np.savez(wSCS_cache_fp, 
                     shw=shw, ashwi=ashwi, cmp_mask=cmp_mask, 
                     olf=olf, offf=np.array([offf[at] for at in trizod.BBATNS]),
                     shw0=shw0, ashwi0=ashwi0, ol0=ol0, off0=np.array([off0[at] for at in trizod.BBATNS]))
    offsets = offf
    if offset_correction == False:
        ashwi = ashwi0
        offsets = off0
    elif not (max_offset == None or np.isinf(max_offset)):
        # check if any offsets are too large
        for i,at in enumerate(trizod.BBATNS):
            if np.abs(offf[at]) > max_offset:
                offsets[at] = np.nan
                if reject_shift_type_only:
                    # mask data related to this backbone shift type, excluding it from scores computation
                    cmp_mask[:,i] = False
                #else:
                #    raise OffsetTooLargeException
    if np.any(cmp_mask):
        ashwi3, k3 = trizod.convert_to_triplet_data(ashwi, cmp_mask)
        if scores_type == 'zscores':
            scores = trizod.compute_zscores(ashwi3, k3, cmp_mask, corr=True)
        elif scores_type == 'chezod':
            scores = trizod.compute_zscores(ashwi3, k3, cmp_mask)
        elif scores_type == 'pscores':
            scores = trizod.compute_pscores(ashwi3, k3, cmp_mask)
        else:
            raise ValueError
        # set positions where score is nan to 0 to avoid confusion
        k = k3
        k[np.isnan(scores)] = 0
    else:
        scores, k = np.full((cmp_mask.shape[0],), np.nan), np.full((cmp_mask.shape[0],), np.nan)
    return scores, k, cmp_mask, offsets

def compute_scores_row(row):
    if not row['pass_pre']:
        return row
    try:
        scores, k, cmp_mask, offsets = compute_scores(
            bmrb_entries[row['id']], row['stID'], row['entity_assemID'], row['entityID'],
            row['seq'], row['ionic_strength'], row['pH'], row['temperature'],
            scores_type=args.scores_type, offset_correction=args.no_offset_correction, 
            max_offset=args.max_offset, reject_shift_type_only=args.reject_shift_type_only,
            #min_backbone_shift_types=args.min_backbone_shift_types,
            #min_backbone_shift_positions=args.min_backbone_shift_positions,
            #min_backbone_shift_fraction=args.min_backbone_shift_fraction,
            cache_dir=args.cache_dir)
        row['scores'] = scores
        row['k'] = k
        #row['cmp_mask'] = cmp_mask
        for at in trizod.BBATNS:
            row[f'off_{at}'] = offsets[at]
        row['total_bbshifts_post'] = np.sum(cmp_mask)
        row['bbshift_types_post'] = np.any(cmp_mask, axis=0).sum()
        row['bbshift_positions_post'] = np.any(cmp_mask, axis=1).sum()
    except ZscoreComputationError:
        pass
    return row

def output_dataset(df, output_prefix, output_format):
    if output_format == 'csv':
        df.loc[df.pass_post, 'seq'] = df[df.pass_post].seq.apply(lambda x: list(x))
        dout = df.loc[df.pass_post].reset_index()[['id', 'stID', 'entity_assemID', 'entityID', 'seq', 'scores', 'k']]
        dout['seq'] = dout.seq.apply(lambda x: list(x))
        dout['seq_index'] = dout.seq.apply(lambda x: list(range(1,len(x)+1)))
        dout = dout.explode(['seq_index', 'seq', 'scores', 'k'])
        dout[['id', 'stID', 'entity_assemID', 'entityID', 'seq_index', 'seq', 'scores', 'k']].to_csv(output_prefix + '.csv', float_format='%.3f')
    else:
        raise ValueError(f"Unknown output format: {output_format}")

def main():
    # load all BMRB data
    bmrb_files = find_bmrb_files(args.input_dir, args.BMRB_file_pattern)
    global bmrb_entries
    bmrb_entries, failed = load_bmrb_entries(bmrb_files, args.cache_dir)
    if failed:
        logging.warning(f"Could not load {len(failed)} of {len(bmrb_files)} BMRB files")
    # parse information and filter entries
    import time
    start_time = time.time()    
    df = create_peptide_dataframe_parallel(
        bmrb_entries, 
        chemical_denaturants=args.chemical_denaturants, 
        keywords=args.keywords_blacklist, 
        return_default=args.no_default_conditions, 
        assume_si=args.no_unit_assumptions, 
        fix_outliers=args.no_unit_corrections)
    print("--- %s seconds ---" % (time.time() - start_time))
    df, missing_vals, sels_pre, sels_kws, sels_denat, sels_all_pre = prefilter_dataframe(
        df,
        method_whitelist=args.exp_method_whitelist, 
        method_blacklist=args.exp_method_blacklist,
        temperature_range=args.temperature_range,
        ionic_strength_range=args.ionic_strength_range,
        pH_range=args.pH_range,
        peptide_length_range=args.peptide_length_range,
        min_backbone_shift_types=args.min_backbone_shift_types,
        min_backbone_shift_positions=args.min_backbone_shift_positions,
        min_backbone_shift_fraction=args.min_backbone_shift_fraction,
        keywords=args.keywords_blacklist,
        chemical_denaturants=args.chemical_denaturants,
        )
    # compute zscores for each remaining entry
    np.seterr(invalid='raise', divide='raise')
    df = df.apply(compute_scores_row, axis=1)
    # filter based on offset and scores
    sels_post, sels_off, sels_all_post = postfilter_dataframe(
        df,
        min_backbone_shift_types=args.min_backbone_shift_types,
        min_backbone_shift_positions=args.min_backbone_shift_positions,
        min_backbone_shift_fraction=args.min_backbone_shift_fraction,
        reject_shift_type_only=args.reject_shift_type_only)
    # print filtering results and summary stats
    print_filter_losses(df, missing_vals, sels_pre, sels_kws, sels_denat, sels_all_pre, sels_post, sels_off, sels_all_post)
    # output results
    output_dataset(df, args.output_prefix, args.output_format)

if __name__ == '__main__':
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level=level, format=f'%(levelname)s : %(message)s') #filename='example.log', encoding='utf-8'
    pandarallel.initialize(nb_workers=12, progress_bar=False)
    main()