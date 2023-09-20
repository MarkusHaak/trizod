#!/usr/bin/env python3
import os, sys
import logging
import argparse
import trizod.potenci.potenci as potenci
import trizod.bmrb.bmrb as bmrb
import trizod.scoring.scoring as scoring
from trizod.constants import BBATNS
from trizod.utils import ArgHelpFormatter
import numpy as np
import pandas as pd
import traceback
import re
import pickle
from tqdm import tqdm
from pandarallel import pandarallel
import time

class Found(Exception): pass
class ZscoreComputationError(Exception): pass
class OffsetTooLargeException(Exception): pass
class OffsetCausedFilterException(Exception): pass
class FilterException(Exception): pass

filter_defaults = pd.DataFrame({
    'temperature-range' : [[-np.inf, +np.inf], [263.0, 333.0], [273.0, 313.0], [273.0, 313.0]],
    'ionic-strength-range' : [[0., np.inf], [0., 5.], [0., 3.], [0., 3.]],
    'pH-range' : [[-np.inf, np.inf], [2., 12.], [3., 11.], [4., 9.]],
    'unit-assumptions' : [True, True, True, False],
    'unit-corrections' : [True, True, False, False],
    'default-conditions' : [True, True, True, False],
    'peptide-length-range' : [[5], [5], [10], [15]],
    'min-backbone-shift-types' : [1, 2, 3, 5],
    'min-backbone-shift-positions' : [3, 3, 8, 12],
    'min-backbone-shift-fraction' : [0., 0., 0.6, 0.8],
    'keywords-blacklist' : [[], 
                            ['denatur'], 
                            ['denatur', 'unfold', 'misfold'], 
                            ['denatur', 'unfold', 'misfold', 'interact', 'bound']],#, 'bind'
    'chemical-denaturants' : [[], 
                              ['guanidin', 'GdmCl', 'Gdn-Hcl','urea','BME','2-ME','mercaptoethanol'], 
                              ['guanidin', 'GdmCl', 'Gdn-Hcl','urea','BME','2-ME','mercaptoethanol'], 
                              ['guanidin', 'GdmCl', 'Gdn-Hcl','urea','BME','2-ME','mercaptoethanol', 
                               'TFA', 'trifluoroethanol', 'Potassium Pyrophosphate', 'acetic acid', 'CD3COOH',
                               'DTT', 'dithiothreitol', 'dss', 'deuterated sodium acetate']],
    'exp-method-whitelist' : [['', '.'], ['','solution', 'structures'], ['','solution', 'structures'], ['solution','structures']],
    'exp-method-blacklist' : [[], ['solid', 'state'], ['solid', 'state'], ['solid', 'state']],
    'max-offset' : [np.inf, 3., 3., 2.],
    'reject-shift-type-only' : [True, True, False, False],
}, index=['unfiltered', 'tolerant', 'moderate', 'strict'])


def parse_args():
    init_parser = argparse.ArgumentParser(description='', add_help=False)
    
    io_grp = init_parser.add_argument_group('Input/Output Options')
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
    

    filter_defaults_grp = init_parser.add_argument_group('Filter Default Settings')
    filter_defaults_arg = filter_defaults_grp.add_argument(
        '--filter-defaults', choices=list(filter_defaults.index), default='tolerant',
        help='Sets defaults for all filter options, from unfiltered to strict.')
    args_init, remaining_argv = init_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        parents=[init_parser],
        description=__doc__,
        formatter_class=ArgHelpFormatter,#formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    filter_grp = parser.add_argument_group('Filtering Options')
    filter_grp.add_argument(
        '--temperature-range', nargs=2, type=float, 
        default=filter_defaults.loc[args_init.filter_defaults, 'temperature-range'],
        help='Minimum and maximum temperature in Kelvin.')
    filter_grp.add_argument(
        '--ionic-strength-range', nargs=2, type=float, 
        default=filter_defaults.loc[args_init.filter_defaults, 'ionic-strength-range'],
        help='Minimum and maximum ionic strength in Mol.')
    filter_grp.add_argument(
        '--pH-range', nargs=2, type=float, 
        default=filter_defaults.loc[args_init.filter_defaults, 'pH-range'],
        help='Minimum and maximum pH.')
    filter_grp.add_argument(
        '--unit-assumptions', action=argparse.BooleanOptionalAction, 
        default=filter_defaults.loc[args_init.filter_defaults, 'unit-assumptions'],
        help='Do not assume units if they are not given and exclude entries instead.')
    filter_grp.add_argument(
        '--unit-corrections', action=argparse.BooleanOptionalAction, 
        default=filter_defaults.loc[args_init.filter_defaults, 'unit-corrections'],
        help='Do not correct values if units are most likely wrong.')
    filter_grp.add_argument(
        '--default-conditions', action=argparse.BooleanOptionalAction, 
        default=filter_defaults.loc[args_init.filter_defaults, 'default-conditions'],
        help='Do not assume standard conditions if pH (7), ionic strength (0.1 M) or temperature (298 K) are missing and exclude entries instead.')
    filter_grp.add_argument(
        '--peptide-length-range', nargs='+', type=int, 
        default=filter_defaults.loc[args_init.filter_defaults, 'peptide-length-range'],
        help='Minimum (and optionally maximum) peptide sequence length.')
    filter_grp.add_argument(
        '--min-backbone-shift-types', type=int, 
        default=filter_defaults.loc[args_init.filter_defaults, 'min-backbone-shift-types'],
        help='Minimum number of different backbone shift types (max 7).')
    filter_grp.add_argument(
        '--min-backbone-shift-positions', type=int, 
        default=filter_defaults.loc[args_init.filter_defaults, 'min-backbone-shift-positions'],
        help='Minimum number of positions with at least one backbone shift.')
    filter_grp.add_argument(
        '--min-backbone-shift-fraction', type=float, 
        default=filter_defaults.loc[args_init.filter_defaults, 'min-backbone-shift-fraction'],
        help='Minimum fraction of positions with at least one backbone shift.')
    filter_grp.add_argument(
        '--keywords-blacklist', nargs='*', 
        default=filter_defaults.loc[args_init.filter_defaults, 'keywords-blacklist'],
        help='Exclude entries with any of these keywords mentioned anywhere in the BMRB file, case ignored.')
    filter_grp.add_argument(
        '--chemical-denaturants', nargs='*', 
        default=filter_defaults.loc[args_init.filter_defaults, 'chemical-denaturants'],
        help='Exclude entries with any of these chemicals as substrings of sample components, case ignored.')
    filter_grp.add_argument(
        '--exp-method-whitelist', nargs='*', 
        default=filter_defaults.loc[args_init.filter_defaults, 'exp-method-whitelist'],
        help='Include only entries with any of these keywords as substring of the experiment subtype, case ignored.')
    filter_grp.add_argument(
        '--exp-method-blacklist', nargs='*', 
        default=filter_defaults.loc[args_init.filter_defaults, 'exp-method-blacklist'],
        help='Exclude entries with any of these keywords as substring of the experiment subtype, case ignored.')
    
    scores_grp = parser.add_argument_group('Scoring Options')
    scores_grp.add_argument(
        '--scores-type', choices=['zscores', 'chezod', 'pscores'], 
        default='zscores',
        help='Which type of scores are created: Observation count-independent zscores (zscores), '
        'original CheZOD zscores (chezod) or geometric mean of observation probabilities (pscores).')
    scores_grp.add_argument(
        '--offset-correction', action=argparse.BooleanOptionalAction, default=True,
        help='Do not compute correction offsets for random coil chemical shifts')
    scores_grp.add_argument(
        '--max-offset', type=float, 
        default=filter_defaults.loc[args_init.filter_defaults, 'max-offset'],
        help='Maximum valid offset for any random coil chemical shift type.')
    scores_grp.add_argument(
        '--reject-shift-type-only', action=argparse.BooleanOptionalAction,
        default=filter_defaults.loc[args_init.filter_defaults, 'reject-shift-type-only'],
        help='Upon exceeding the maximal offset set by <--max-offset>, exclude only the backbone shifts exceeding the offset instead of the whole entry.')

    other_grp = parser.add_argument_group('Other Options')
    other_grp.add_argument(
        '--processes', default=8, type=int,
        help='Number of processes to spawn in multiprocessing.')
    other_grp.add_argument(
        '--progress', action=argparse.BooleanOptionalAction,
        default=True,
        help='Show progress bars.')

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    #args = argparse.Namespace(**vars(args_init), **vars(args))


    if not os.path.exists(args.input_dir):
        logging.getLogger('trizod').error(f"Input directory {args.input_dir} does not exist.")
        exit(1)
    if not os.path.isdir(args.input_dir):
        logging.getLogger('trizod').error(f"Path {args.input_dir} is not a directory.")
        exit(1)
    args.input_dir = os.path.abspath(args.input_dir)

    args.output_prefix = os.path.abspath(args.output_prefix)
    if not os.path.exists(os.path.dirname(args.output_prefix)):
        logging.getLogger('trizod').error(f"Output directory {os.path.dirname(args.output_prefix)} does not exist.")
        exit(1)

    if len(args.peptide_length_range) == 1:
        args.peptide_length_range.append(np.inf)

    args.cache_dir = os.path.abspath(args.cache_dir)
    dirs = [args.cache_dir,
            os.path.join(args.cache_dir, 'wSCS'),
            os.path.join(args.cache_dir, 'bmrb_entries')]
    if not np.all([os.path.exists(d) for d in dirs]):
        if not os.path.exists(args.cache_dir):
            logging.getLogger('trizod').debug(f"Directory {args.cache_dir} does not exist and is created.")
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    elif not os.path.isdir(args.cache_dir):
        logging.getLogger('trizod').error(f"Path {args.cache_dir} is not a directory.")
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

'''
def load_bmrb_entries(bmrb_files, cache_dir=None, progress=False):
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
        it = tqdm(missing_bmrb_files) if progress else missing_bmrb_files
        for id_ in it:
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
'''

def parse_bmrb_file(row, cache_dir=None):
    try:
        entry = bmrb.BmrbEntry(row.name, row.dir)
        if cache_dir:
            with open(row.cache_fp, 'wb') as f:
                pickle.dump(entry, f)
        return entry
    except:
        return None

def load_bmrb_entries(bmrb_files, cache_dir=None):
    entries, failed = {}, []
    columns = ['entry', 'dir']
    # read cached data
    if cache_dir:
        columns.append('cache_fp')
        for id_, fp in bmrb_files.items():
            cache_fp = os.path.join(cache_dir, "bmrb_entries", f"{id_}.pkl")
            entry = None
            if os.path.exists(cache_fp):
                try:
                    with open(cache_fp, "rb") as f:
                        entry = pickle.load(f)
                except:
                    logging.getLogger('trizod.bmrb').debug(f"cache file {cache_fp} corrupt or formatted wrong")
            entries[id_] = (entry, os.path.dirname(fp), cache_fp)
    else:
        for id_, fp in bmrb_files.items():
            entries[id_] = (None, os.path.dirname(fp))
    df = pd.DataFrame(entries.values(), index=entries.keys(), columns=columns)
    sel = pd.isna(df.entry)
    df.loc[sel, entry] = df.loc[sel].parallel_apply(parse_bmrb_file, axis=1, cache_dir=cache_dir)
    sel = pd.isna(df.entry)
    failed = df.loc[sel].index.to_list()
    return df.loc[~sel], failed

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
    missing_vals = ~df[['exp_method', 'temperature', 'ionic_strength', 'pH', 'seq', 'total_bbshifts']].isna().any(axis=1)
    method_sel = df.exp_method.str.lower().str.contains('nmr')
    method_whitelist_ = [l.lower() for l in method_whitelist]
    if method_whitelist_:
        method_sel &= (df.exp_method_subtype.str.lower().str.contains("|".join(method_whitelist_), regex=True))
    else:
        method_sel = False
    method_blacklist_ = [l.lower() for l in method_blacklist]
    if method_blacklist_:
        method_sel &= (~df.exp_method_subtype.str.lower().str.contains("|".join(method_blacklist_), regex=True))
    if '' in method_whitelist and '' not in method_blacklist:
        method_sel |= df.exp_method.str.lower().str.contains('nmr') & pd.isna(df.exp_method_subtype)
    else:
        method_sel = sels_pre["method (sub-)type"].fillna(False)
        missing_vals &= ~pd.isna(df.exp_method_subtype)
    sels_pre = {
        #"missing values" : ~df[['ionic_strength', 'pH', 'temperature','seq','total_bbshifts', 'bbshift_types']].isna().any(axis=1),
        "method (sub-)type" : method_sel,
        "temperature" : (df.temperature >= temperature_range[0]) & \
                        (df.temperature <= temperature_range[1]),
        "ionic strength" : (df.ionic_strength >= ionic_strength_range[0]) & \
                           (df.ionic_strength <= ionic_strength_range[1]),
        "pH" : (df.pH >= pH_range[0]) & \
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
        for at in scoring.BBATNS:
            any_offsets_too_large |= pd.isna(df[f"off_{at}"])
        sels_post.update({"rejected due to any offset" : ~any_offsets_too_large})

    sels_off = {
        f"off_{at}" : ~pd.isna(df[f"off_{at}"]) for at in BBATNS 
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
    if sels_kws:
        print()
        print(f"{'keyword':>{w_str}} : {'filtered':<{w_num}} {'unique':<{w_num}}")
        for filter, sel in sels_kws.items():
            uniq = pd.Series(np.full((len(sel),), False))
            for other_filter, other_sel in sels_all_pre.items():
                if other_filter != filter:
                    uniq |= ~other_sel
            uniq = ~sel & ~uniq
            print(f"{'.*'+filter+'.*':<{w_str}} : {(~sel).sum():>{w_num}} {uniq.sum():>{w_num}}")
    if sels_denat:
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
    entry = bmrb_entries.loc[row['id'], 'entry'] #row['entry']
    peptide_shifts = entry.get_peptide_shifts()
    shifts, condID, assemID, sampleIDs = peptide_shifts[(row['stID'], row['entity_assemID'], row['entityID'])]
    row['citation_title'] = entry.citation_title
    row['citation_DOI'] = entry.citation_DOI
    row['exp_method'] = entry.exp_method if entry.exp_method else pd.NA
    row['exp_method_subtype'] = entry.exp_method_subtype if entry.exp_method_subtype else pd.NA
    row['ionic_strength'] = entry.conditions[condID].get_ionic_strength(return_default=return_default, assume_si=assume_si, fix_outliers=fix_outliers)
    row['pH'] = entry.conditions[condID].get_pH(return_default=return_default)
    row['temperature'] = entry.conditions[condID].get_temperature(return_default=return_default, assume_si=assume_si, fix_outliers=fix_outliers)
    seq = entry.entities[row['entityID']].seq
    row['seq'] = seq
    # retrieve # backbone shifts (H,HA,HB,C,CA,CB,N)
    total_bbshifts, bbshift_types, bbshift_positions = None, None, None
    if seq:
        ret = bmrb.get_valid_bbshifts(shifts, seq)
        if ret:
            #bbshifts, bbshifts_arr, bbshifts_mask = ret
            bbshifts_arr, bbshifts_mask = ret
            total_bbshifts = np.sum(bbshifts_mask) # total backbone shifts
            bbshift_types = np.any(bbshifts_mask, axis=0).sum() # different backbone shifts
            bbshift_positions = np.any(bbshifts_mask, axis=1).sum() # positions with backbone shifts
    row['total_bbshifts'] = total_bbshifts
    row['bbshift_types'] = bbshift_types
    row['bbshift_positions'] = bbshift_positions
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
        row[keyword] = False
        for field in fields:
            if field: # can be None
                if keyword.lower() in field.lower():
                    row[keyword] = True
                    break
    # check if chemical detergents are present
    for den_comp in chemical_denaturants:
        row[den_comp] = False
        try:
            for sID in sampleIDs:
                for comp in entry.samples[sID].components:
                    if comp[3] and not comp[2]: # if it has a name but no entity entry
                        if den_comp.lower() in comp[3].lower():
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
    for at in BBATNS:
        row[f'off_{at}'] = pd.NA
    return row

'''
def create_peptide_dataframe(bmrb_entries,
                             chemical_denaturants, keywords,
                             return_default=True, assume_si=True, fix_outliers=True,
                             progress=False
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
    it = tqdm(bmrb_entries.items()) if progress else bmrb_entries.items()
    for id_,entry in it:
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
                    #bbshifts, bbshifts_arr, bbshifts_mask = ret
                    bbshifts_arr, bbshifts_mask = ret
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
    for at in BBATNS:
        df[f'off_{at}'] = pd.NA
    return df
'''

def create_peptide_dataframe(bmrb_entries,
                             chemical_denaturants, keywords,
                             return_default=True, assume_si=True, fix_outliers=True,
                             progress=False
                             ):
    data = []
    columns = ['id', #'entry', 
               'stID', 'entity_assemID', 'entityID'
            ]
    #it = tqdm(bmrb_entries.items()) if progress else bmrb_entries.items()
    #for id_,entry in it:
    #    peptide_shifts = entry.get_peptide_shifts()
    #    for (stID, entity_assemID, entityID) in peptide_shifts:
    #        data.append([])
    #        data[-1].append(id_)
    #        #data[-1].append(entry)
    #        data[-1].extend([stID, entity_assemID, entityID])
    it = tqdm(bmrb_entries.iterrows()) if progress else bmrb_entries.iterrows()
    for id_,row in it:
        peptide_shifts = row.entry.get_peptide_shifts()
        for (stID, entity_assemID, entityID) in peptide_shifts:
            data.append([])
            data[-1].append(id_)
            #data[-1].append(entry)
            data[-1].extend([stID, entity_assemID, entityID])
    df = pd.DataFrame(data, columns=columns)

    df = df.parallel_apply(fill_row_data, axis=1, args=(chemical_denaturants, keywords), 
                           return_default=return_default, assume_si=assume_si, fix_outliers=fix_outliers)
    df = df.astype({col : "string" for col in ['id', 'citation_title', 'citation_DOI', 'exp_method', 'exp_method_subtype', 'seq']})
    return df

def compute_scores(entry, stID, entity_assemID, entityID,
                   seq, ion, pH, temperature,
                   scores_type='zscores', offset_correction=True, 
                   max_offset=np.inf, reject_shift_type_only=False,
                   #min_backbone_shift_types=1, min_backbone_shift_positions=1, min_backbone_shift_fraction=0.,
                   cache_dir=None):
    exe_times = [np.nan, np.nan, np.nan]
    wSCS_cache_fp = os.path.join(cache_dir, 'wSCS', f'{entry.id}_{stID}_{entity_assemID}_{entityID}.npz')
    if cache_dir and os.path.exists(wSCS_cache_fp):
        try:
            z = np.load(wSCS_cache_fp)
            shw, ashwi, cmp_mask, olf, offf, shw0, ashwi0, ol0, off0 = z['shw'], z['ashwi'], z['cmp_mask'], z['olf'], z['offf'], z['shw0'], z['ashwi0'], z['ol0'], z['off0']
            offf, off0 = {at:off for at,off in zip(BBATNS, offf)}, {at:off for at,off in zip(BBATNS, off0)}
        except:
            logging.getLogger('trizod').debug(f"cache file {wSCS_cache_fp} corrupt or formatted wrong, delete and repeat computation")
            os.remove(wSCS_cache_fp)
    if not (cache_dir and os.path.exists(wSCS_cache_fp)):
        peptide_shifts = entry.get_peptide_shifts()
        shifts, condID, assemID, sampleIDs = peptide_shifts[(stID, entity_assemID, entityID)]
        
        try:
            # predict random coil chemical shifts using POTENCI
            usephcor = (pH != 7.0)
            start_time = time.time()
            predshiftdct = potenci.getpredshifts(seq, temperature, pH, ion, usephcor, pkacsvfile=False)
            exe_times[0] = time.time() - start_time
        except:
            logging.getLogger('trizod').error(f"POTENCI failed for {(entry.id, stID, entity_assemID, entityID)} due to the following error:", exc_info=True)
            raise ZscoreComputationError
        start_time = time.time()
        ret = scoring.get_offset_corrected_wSCS(seq, shifts, predshiftdct)
        if ret is None:
            logging.getLogger('trizod').error(f'TriZOD failed for {(entry.id, stID, entity_assemID, entityID)} due to an error in computation of corrected wSCSs.')
            raise ZscoreComputationError
        else:
            exe_times[1] = time.time() - start_time
        shw, ashwi, cmp_mask, olf, offf, shw0, ashwi0, ol0, off0 = ret
        if cache_dir:
            np.savez(wSCS_cache_fp, 
                     shw=shw, ashwi=ashwi, cmp_mask=cmp_mask, 
                     olf=olf, offf=np.array([offf[at] for at in BBATNS]),
                     shw0=shw0, ashwi0=ashwi0, ol0=ol0, off0=np.array([off0[at] for at in BBATNS]))
    offsets = offf
    if offset_correction == False:
        ashwi = ashwi0
        offsets = off0
    elif not (max_offset == None or np.isinf(max_offset)):
        # check if any offsets are too large
        for i,at in enumerate(BBATNS):
            if np.abs(offf[at]) > max_offset:
                offsets[at] = np.nan
                if reject_shift_type_only:
                    # mask data related to this backbone shift type, excluding it from scores computation
                    cmp_mask[:,i] = False
                #else:
                #    raise OffsetTooLargeException
    if np.any(cmp_mask):
        start_time = time.time()
        ashwi3, k3 = scoring.convert_to_triplet_data(ashwi, cmp_mask)
        if scores_type == 'zscores':
            scores = scoring.compute_zscores(ashwi3, k3, cmp_mask, corr=True)
        elif scores_type == 'chezod':
            scores = scoring.compute_zscores(ashwi3, k3, cmp_mask)
        elif scores_type == 'pscores':
            scores = scoring.compute_pscores(ashwi3, k3, cmp_mask)
        else:
            raise ValueError
        # set positions where score is nan to 0 to avoid confusion
        k = k3
        k[np.isnan(scores)] = 0
        exe_times[2] = time.time() - start_time
    else:
        scores, k = np.full((cmp_mask.shape[0],), np.nan), np.full((cmp_mask.shape[0],), np.nan)
    return scores, k, cmp_mask, offsets, exe_times

def compute_scores_row(row, scores_type='zscores', offset_correction=True, 
                       max_offset=np.inf, reject_shift_type_only=False,
                       cache_dir=None):
    if not row['pass_pre']:
        return row
    try:
        start_time = time.time()
        scores, k, cmp_mask, offsets, exe_times = compute_scores(
            bmrb_entries.loc[row['id'], 'entry'], row['stID'], row['entity_assemID'], row['entityID'],
            row['seq'], row['ionic_strength'], row['pH'], row['temperature'],
            scores_type=scores_type, offset_correction=offset_correction, 
            max_offset=max_offset, reject_shift_type_only=reject_shift_type_only,
            #min_backbone_shift_types=args.min_backbone_shift_types,
            #min_backbone_shift_positions=args.min_backbone_shift_positions,
            #min_backbone_shift_fraction=args.min_backbone_shift_fraction,
            cache_dir=cache_dir)
        row['scores'] = scores
        row['k'] = k
        #row['cmp_mask'] = cmp_mask
        for at in BBATNS:
            row[f'off_{at}'] = offsets[at]
        row['total_bbshifts_post'] = np.sum(cmp_mask)
        row['bbshift_types_post'] = np.any(cmp_mask, axis=0).sum()
        row['bbshift_positions_post'] = np.any(cmp_mask, axis=1).sum()
        row['tpotenci'] = exe_times[0]
        row['ttrizod'] = exe_times[1]
        row['tscores'] = exe_times[2]
        row['ttotal'] = time.time() - start_time
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
    args = parse_args()
    if args.processes is None:
        pandarallel.initialize(verbose=0, progress_bar=args.progress)
    else:
        pandarallel.initialize(verbose=0, nb_workers=args.processes, progress_bar=args.progress)
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format=f'%(levelname)s : %(message)s')
    # reject most logging messages for sub-routines like parsing database files:
    logging.getLogger('trizod.bmrb').setLevel(logging.CRITICAL)
    logging.getLogger('trizod.scoring').setLevel(logging.CRITICAL)

    logging.getLogger('trizod').info('Loading BMRB files.')
    bmrb_files = find_bmrb_files(args.input_dir, args.BMRB_file_pattern)
    global bmrb_entries
    bmrb_entries, failed = load_bmrb_entries(bmrb_files, cache_dir=args.cache_dir)
    print()
    if failed:
        logging.getLogger('trizod').warning(f"Could not load {len(failed)} of {len(bmrb_files)} BMRB files")
    logging.getLogger('trizod').info('Parsing and filtering relevant information.')
    df = create_peptide_dataframe(
        bmrb_entries, 
        chemical_denaturants=args.chemical_denaturants, 
        keywords=args.keywords_blacklist, 
        return_default=args.default_conditions, 
        assume_si=args.unit_assumptions, 
        fix_outliers=args.unit_corrections,
        progress=args.progress)
    breakpoint()
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
    print()
    logging.getLogger('trizod').info('Computing scores for each remaining entry.')
    df = df.parallel_apply(
        compute_scores_row, axis=1, 
        scores_type=args.scores_type, 
        offset_correction=args.offset_correction, 
        max_offset=args.max_offset, 
        reject_shift_type_only=args.reject_shift_type_only,
        cache_dir=args.cache_dir)
    print()
    logging.getLogger('trizod').info('Filtering results.')
    sels_post, sels_off, sels_all_post = postfilter_dataframe(
        df,
        min_backbone_shift_types=args.min_backbone_shift_types,
        min_backbone_shift_positions=args.min_backbone_shift_positions,
        min_backbone_shift_fraction=args.min_backbone_shift_fraction,
        reject_shift_type_only=args.reject_shift_type_only)
    logging.getLogger('trizod').info('Output filtering results.')
    print_filter_losses(df, missing_vals, sels_pre, sels_kws, sels_denat, sels_all_pre, sels_post, sels_off, sels_all_post)
    logging.getLogger('trizod').info('Writing dataset to file.')
    output_dataset(df, args.output_prefix, args.output_format)

if __name__ == '__main__':
    main()