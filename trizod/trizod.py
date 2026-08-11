#!/usr/bin/env python3
import logging
import re
import time

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

import trizod.bmrb.bmrb as bmrb
from trizod.bmrb.sample_state import (
    PHYSICAL_STATE_MODERATE_DENY,
    PHYSICAL_STATE_STRICT_DENY,
    PHYSICAL_STATE_TOLERANT_DENY,
    entry_sample_state,
    has_denaturant_evidence,
    resolve_physical_state,
)
from trizod.cache import (
    _potenci_cache_key,  # noqa: F401  (re-exported for backward compatibility)
    load_potenci_cache,  # noqa: F401
    save_potenci_cache,  # noqa: F401
)
from trizod.constants import BACKBONE_ATOMS
from trizod.pipeline import (
    ZscoreComputationError,
    compute_scores,
    find_bmrb_files,
    load_bmrb_entries,
    postfilter_dataframe,
    prefilter_dataframe,
    print_filter_losses,
)
from trizod.provenance import pipeline_version


class OffsetTooLargeException(Exception):
    pass


class OffsetCausedFilterException(Exception):
    pass


class FilterException(Exception):
    pass


#: Keywords that must match as whole words rather than as substrings.
#: ``bound`` is the only one: 22 moderate rows match it exclusively through
#: ``unbound`` / ``boundaries`` ("Unbound Med25ACID", "Human Pdx1 Homeodomain in
#: the Unbound State"), i.e. exactly the free-state depositions the strict tier
#: wants to KEEP. Prefix keywords such as ``denatur`` and ``unfold`` deliberately
#: stay substring matches.
WORD_BOUNDARY_KEYWORDS = frozenset({"bound"})

#: Detergents/lipids that make a sample a membrane mimetic rather than a
#: denaturant. Annotated as the ``membrane_mimetic`` column, never filtered:
#: the rows they would remove are 93.7 % ordered and contain zero disordered
#: chains, so filtering on them would be a one-sided removal of ordered examples.
MEMBRANE_MIMETIC_TOKENS = (
    "SDS",
    "DPC",
    "dodecyl",
    "LPPG",
    "LMPG",
    "DHPC",
    "DMPC",
    "POPC",
    "bicelle",
    "micelle",
    "Triton",
    "CHAPS",
    "octyl",
    "maltoside",
    "digitonin",
    "nanodisc",
)

_KEYWORD_PATTERNS = {}


def _keyword_matches(keyword, fields):
    """Does ``keyword`` occur in any of ``fields`` (all lower-cased already)?"""
    if keyword in WORD_BOUNDARY_KEYWORDS:
        pattern = _KEYWORD_PATTERNS.get(keyword)
        if pattern is None:
            pattern = _KEYWORD_PATTERNS[keyword] = re.compile(
                rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
            )
        return any(pattern.search(field) for field in fields)
    return any(keyword in field for field in fields)


filter_defaults = pd.DataFrame(
    {
        "temperature-range": [
            [-np.inf, +np.inf],
            [263.0, 333.0],
            [273.0, 323.0],
            [273.0, 313.0],
        ],
        "ionic-strength-range": [[0.0, np.inf], [0.0, 7.0], [0.0, 5.0], [0.0, 3.0]],
        "pH-range": [[-np.inf, np.inf], [2.0, 12.0], [4.0, 10.0], [6.0, 8.0]],
        "unit-assumptions": [True, True, True, False],
        "unit-corrections": [True, True, False, False],
        "default-conditions": [True, True, True, False],
        "peptide-length-range": [[5], [5], [10], [15]],
        "min-backbone-shift-types": [1, 2, 3, 4],
        "min-backbone-shift-positions": [3, 3, 8, 12],
        "min-backbone-shift-fraction": [0.0, 0.0, 0.6, 0.8],
        "max-noncanonical-fraction": [1.0, 0.1, 0.025, 0.0],
        "max-x-fraction": [1.0, 0.2, 0.05, 0.0],
        "keywords-blacklist": [
            [],
            ["denatur"],
            ["denatur", "unfold", "misfold"],
            [
                "denatur",
                "unfold",
                "misfold",
                "bound",
                "in complex with",
                "complexed with",
            ],
        ],
        # 'interacti[on/ng]' removed: all 46 of its strict hits came from
        # _Citation_keyword/_Struct_keywords -- paper-topic labels such as
        # "protein-protein interaction" on free monomers -- and only 41 % had an
        # independent detect_bound() signal. The two complex phrases replace it
        # on sample-descriptive fields only (261 strict rows, 94 % corroborated).
        "keyword-search-scope": ["sample", "sample", "sample", "sample"],
        "physical-state-blacklist": [
            [],
            list(PHYSICAL_STATE_TOLERANT_DENY),
            list(PHYSICAL_STATE_MODERATE_DENY),
            list(PHYSICAL_STATE_STRICT_DENY),
        ],
        "chemical-denaturants": [
            [],
            ["guanidin", "GdmCl", "Gdn-Hcl", "urea", "TFE", "trifluoroethanol"],
            ["guanidin", "GdmCl", "Gdn-Hcl", "urea", "TFE", "trifluoroethanol", "DMSO"],
            ["guanidin", "GdmCl", "Gdn-Hcl", "urea", "TFE", "trifluoroethanol", "DMSO"],
        ],
        # 'TFA' removed (57 percent-unit components, median 0.1 %, 56/57 <= 0.2 %:
        # an HPLC counterion, and its acidification pathway is already covered by
        # the pH filter) and 'Potassium Pyrophosphate' removed (a buffer, added in
        # an unreviewed grab-bag commit). Net data impact of both: +1 strict row.
        # 'TFE' is genuinely new -- 'trifluoroethanol' does not contain 'tfe'.
        "exp-method-whitelist": [
            ["", "."],
            ["", "solution", "structures"],
            ["", "solution", "structures"],
            ["solution", "structures"],
        ],
        "exp-method-blacklist": [
            [],
            ["solid"],
            ["solid"],
            ["solid"],
        ],
        # What to do about the 4,103 entries whose _Entry.Experimental_method_subtype
        # tag is simply ABSENT -- 90.8 % of everything the strict method filter
        # rejects, 90.7 % of them deposited before 2006 when the tag was not
        # routinely filled in. 'off' keeps unfiltered = the raw corpus.
        "method-fallback": ["off", "reject-solid", "reject-solid", "require-solution"],
        "exclude-paramagnetic": [False, True, True, True],
        "max-offset": [np.inf, 3.0, 3.0, 2.0],
        "reject-shift-type-only": [True, True, False, False],
    },
    index=["unfiltered", "tolerant", "moderate", "strict"],
)


def fill_row_data(
    row,
    chemical_denaturants,
    keywords,
    return_default=True,
    assume_si=True,
    fix_outliers=True,
    include_shifts=False,
    no_shift_averaging=False,
    keyword_search_scope="all",
    membrane_mimetics=MEMBRANE_MIMETIC_TOKENS,
    bmrb_entries=None,
):
    """Fill in one peptide row's metadata, keyword flags and state columns.

    ``keyword_search_scope`` defaults to ``"all"`` -- the historical behaviour of
    this function -- while the shipped tier defaults resolve it to ``"sample"``
    through ``filter_defaults``, like every other filter policy.
    """
    entry = bmrb_entries.loc[row["entryID"], "entry"]  # row['entry']
    peptide_shifts = entry.get_peptide_shifts()
    shifts, condID, assemID, sampleIDs = peptide_shifts[
        (row["stID"], row["entity_assemID"], row["entityID"])
    ]
    row["citation_title"] = entry.citation_title
    row["citation_DOI"] = entry.citation_DOI
    row["exp_method"] = entry.exp_method if entry.exp_method else pd.NA
    row["exp_method_subtype"] = (
        entry.exp_method_subtype if entry.exp_method_subtype else pd.NA
    )
    row["entity_name"] = entry.entities[row["entityID"]].name
    row["ionic_strength"] = entry.conditions[condID].get_ionic_strength(
        return_default=return_default, assume_si=assume_si, fix_outliers=fix_outliers
    )
    row["pH"] = entry.conditions[condID].get_pH(return_default=return_default)
    row["temperature"] = entry.conditions[condID].get_temperature(
        return_default=return_default, assume_si=assume_si, fix_outliers=fix_outliers
    )
    seq = entry.entities[row["entityID"]].seq
    row["seq"] = seq
    # retrieve # backbone shifts (H,HA,HB,C,CA,CB,N)
    total_bbshifts, bbshift_types, bbshift_positions, bbshifts_arr = (
        None,
        None,
        None,
        None,
    )
    if seq:
        ret = bmrb.get_valid_bbshifts(shifts, seq)
        if ret:
            bbshifts_arr, bbshifts_mask = ret
            if len(bbshifts_mask) >= 2:
                # backbone shift of terminal amino acids are not counted
                total_bbshifts = np.sum(bbshifts_mask[1:-1])  # total backbone shifts
                bbshift_types = np.any(
                    bbshifts_mask, axis=0
                ).sum()  # different backbone shifts
                bbshift_positions = np.any(
                    bbshifts_mask, axis=1
                ).sum()  # positions with backbone shifts
                if include_shifts:
                    if no_shift_averaging:
                        bbshifts_arr, bbshifts_mask = bmrb.get_valid_bbshifts(
                            shifts, seq, averaging=False
                        )
                    bbshifts_arr[~bbshifts_mask] = np.nan
            else:
                total_bbshifts = 0
                bbshifts_arr = None
                bbshift_types = 0
                bbshift_positions = 0
    row["total_bbshifts"] = total_bbshifts
    row["bbshift_types"] = bbshift_types
    row["bbshift_positions"] = bbshift_positions
    if include_shifts:
        row["bbshifts"] = bbshifts_arr
    # check if entry is paramagnetic (assembly or entity level)
    assembly = entry.assemblies[assemID]
    entity = entry.entities[row["entityID"]]
    row["paramagnetic"] = (
        assembly.paramagnetic and assembly.paramagnetic.lower() == "yes"
    ) or (entity.paramagnetic and entity.paramagnetic.lower() == "yes")
    # _Entity_assembly.Physical_state, resolved to THIS row's entity assembly.
    # An entry-wide OR over-filters by 10-28 rows per tier. Emitted at every
    # tier; the per-tier exact-match deny list is applied in prefilter_dataframe.
    physical_state = resolve_physical_state(
        assembly, row["entity_assemID"], row["entityID"]
    )
    row["physical_state"] = physical_state if physical_state else pd.NA
    # A shift table that references no sample is searched against every sample
    # of the entry. Resolved once, here, so the denaturant evidence below and
    # the component scan further down see the same sample set.
    if len(sampleIDs) == 0 and entry.samples:
        sampleIDs = list(entry.samples.keys())
    # `denatured`/`unfolded` are ambiguous depositor vocabulary -- alpha-synuclein
    # (bmr6968) is deposited as `denatured`. Those states are only denied when the
    # entry independently names a denaturant, so record that evidence here, where
    # the sample components are in reach. Tier-independent by design.
    denaturant_texts = [
        entry.title,
        entry.details,
        assembly.name,
        assembly.details,
        entity.name,
        entity.details,
    ]
    for sID in sampleIDs:
        sample = entry.samples.get(sID)
        if sample is None:
            continue
        denaturant_texts.extend([sample.name, sample.details])
        denaturant_texts.extend(comp[3] for comp in sample.components)
    row["denaturant_evidence"] = has_denaturant_evidence(denaturant_texts)
    # Is this solution NMR or solid-state NMR, judging by _Sample.Type,
    # _Experiment.Sample_state and the experiment names?
    row["sample_state_evidence"] = entry_sample_state(entry, sampleIDs)
    # check if keywords are present.
    # Sample-descriptive fields describe what is in the NMR tube; the paper-topic
    # fields (citation title/keywords, struct keywords) describe what the
    # *publication* is about. Matching a state keyword on the latter drops good
    # data -- e.g. bmr51322, an Abeta(1-42) deposition whose own Physical_state
    # is 'intrinsically disordered', on a citation title about amyloid fibrils.
    sample_fields = [
        entry.title,
        entry.details,
        assembly.name,
        assembly.details,
        entity.name,
        entity.details,
    ]
    for sID in sampleIDs:
        sample = entry.samples.get(sID)
        if sample is None:
            continue
        sample_fields.extend([sample.name, sample.details, sample.framecode])
    topic_fields = [entry.citation_title]
    # issue #23: these are lists of STRINGS. The former `fields.extend(el)` on a
    # str exploded it into single characters, so no multi-character keyword could
    # ever match _Citation_keyword.Keyword or _Struct_keywords.
    for kw_field in (entry.citation_keywords, entry.struct_keywords):
        if kw_field is None:
            continue
        if isinstance(kw_field, list):
            topic_fields.extend(kw_field)
        else:
            topic_fields.append(kw_field)
    fields = (
        sample_fields
        if keyword_search_scope == "sample"
        else sample_fields + topic_fields
    )
    fields = [field.lower() for field in fields if field]  # field can be None
    for keyword in keywords:
        row[keyword] = _keyword_matches(keyword.lower(), fields)
    # Sample components, once. _Sample_component.Mol_common_name (comp[3]) is
    # only meaningful for components that are not the studied polymer itself
    # (_Sample_component.Entity_ID, comp[2], unset).
    component_names = []
    for sID in sampleIDs:
        # .get(): bmrb.py strips sample IDs only AFTER the membership test, so a
        # whitespace-padded ID yields a key that is no longer in entry.samples.
        # The KeyError that used to raise here was not caught by the surrounding
        # `except Found`.
        sample = entry.samples.get(sID)
        if sample is None:
            continue
        component_names.extend(
            comp[3].lower() for comp in sample.components if comp[3] and not comp[2]
        )
    # check if chemical detergents are present
    for den_comp in chemical_denaturants:
        row[den_comp] = any(den_comp.lower() in name for name in component_names)
    # Membrane mimetics are annotated, never filtered: the matched token(s), not
    # a bool, so an SDS micelle stays distinguishable from DDM solubilisation.
    matched = [
        token
        for token in membrane_mimetics
        if any(token.lower() in name for name in component_names)
    ]
    row["membrane_mimetic"] = ";".join(matched) if matched else pd.NA
    # add columns that will be filled later
    row["scores"] = None
    row["k"] = None
    row["total_bbshifts_post"] = np.nan
    row["bbshift_types_post"] = np.nan
    row["bbshift_positions_post"] = np.nan
    for atom_type in BACKBONE_ATOMS:
        row[f"off_{atom_type}"] = pd.NA
        row[f"lacs_off_{atom_type}"] = pd.NA
    return row


def create_peptide_dataframe(
    bmrb_entries,
    chemical_denaturants,
    keywords,
    return_default=True,
    assume_si=True,
    fix_outliers=True,
    include_shifts=False,
    no_shift_averaging=False,
    keyword_search_scope="all",
    progress=False,
):
    data = []
    columns = [
        "entryID",  #'entry',
        "stID",
        "entity_assemID",
        "entityID",
    ]
    row_iter = tqdm(bmrb_entries.iterrows()) if progress else bmrb_entries.iterrows()
    for id_, row in row_iter:
        peptide_shifts = row.entry.get_peptide_shifts()
        for stID, entity_assemID, entityID in peptide_shifts:
            data.append([])
            data[-1].append(id_)
            # data[-1].append(entry)
            data[-1].extend([stID, entity_assemID, entityID])
    df = pd.DataFrame(data, columns=columns)
    df = df.parallel_apply(
        fill_row_data,
        axis=1,
        args=(chemical_denaturants, keywords),
        return_default=return_default,
        assume_si=assume_si,
        fix_outliers=fix_outliers,
        include_shifts=include_shifts,
        no_shift_averaging=no_shift_averaging,
        keyword_search_scope=keyword_search_scope,
        bmrb_entries=bmrb_entries,
    )
    df = df.astype(
        dict.fromkeys(
            [
                "entryID",
                "citation_title",
                "citation_DOI",
                "exp_method",
                "exp_method_subtype",
                "entity_name",
                "seq",
                "physical_state",
                "membrane_mimetic",
                "sample_state_evidence",
            ],
            "string",
        )
    )
    return df


def compute_scores_row(
    row,
    score_types=None,
    offset_correction=True,
    max_offset=np.inf,
    reject_shift_type_only=False,
    cache_dir=None,
    rereference_mode="both",
    bmrb_entries=None,
):
    if score_types is None:
        score_types = ["zscores"]
    if not row["pass_pre"]:
        return row
    try:
        start_time = time.time()
        scores, k, cmp_mask, offsets, exe_times, lacs_offsets = compute_scores(
            bmrb_entries.loc[row["entryID"], "entry"],
            row["stID"],
            row["entity_assemID"],
            row["entityID"],
            row["seq"],
            row["ionic_strength"],
            row["pH"],
            row["temperature"],
            score_types=score_types,
            offset_correction=offset_correction,
            max_offset=max_offset,
            reject_shift_type_only=reject_shift_type_only,
            cache_dir=cache_dir,
            rereference_mode=rereference_mode,
        )
        for score_type, score_array in zip(score_types, scores):
            row[score_type] = score_array
        row["k"] = k
        # row['cmp_mask'] = cmp_mask
        for atom_type in BACKBONE_ATOMS:
            row[f"off_{atom_type}"] = offsets[atom_type]
            row[f"lacs_off_{atom_type}"] = lacs_offsets[atom_type]
        row["total_bbshifts_post"] = np.sum(cmp_mask)
        row["bbshift_types_post"] = np.any(cmp_mask, axis=0).sum()
        row["bbshift_positions_post"] = np.any(cmp_mask, axis=1).sum()
        row["tpotenci"] = exe_times[0]
        row["ttrizod"] = exe_times[1]
        row["tscores"] = exe_times[2]
        row["ttotal"] = time.time() - start_time
    except ZscoreComputationError:
        pass
    return row


def output_dataset(
    df,
    output_prefix,
    output_format,
    score_types,
    precision,
    include_shifts,
    no_shift_averaging,
):
    df["ID"] = (
        df["entryID"]
        + "_"
        + df["stID"]
        + "_"
        + df["entity_assemID"]
        + "_"
        + df["entityID"]
    )
    for score_type in score_types:
        df.loc[df.pass_post, score_type] = df.loc[df.pass_post, score_type].apply(
            np.round, args=(precision,)
        )
    shifts = []
    if include_shifts:
        # Column i of bbshifts holds bmrb.get_valid_bbshifts()'s atom i: that is
        # BACKBONE_ATOMS when averaging, plus the split methylene/methyl protons
        # when not. Copy the list -- BACKBONE_ATOMS is a module global.
        shifts = list(BACKBONE_ATOMS)
        if no_shift_averaging:
            shifts += ["HA2", "HA3", "HB1", "HB2", "HB3"]
        for i, atom_type in enumerate(shifts):
            df.loc[df.pass_post, atom_type] = df.loc[df.pass_post, "bbshifts"].apply(
                lambda x, i=i: x[:, i]
            )
            df.loc[df.pass_post, atom_type] = df.loc[df.pass_post, atom_type].apply(
                np.round, args=(precision,)
            )
    if output_format == "csv":
        df["seq"] = df["seq"].astype(object)
        df.loc[df.pass_post, "seq"] = df[df.pass_post].seq.apply(lambda x: list(x))
        dout = df.loc[df.pass_post].reset_index()[
            [
                "ID",
                "entryID",
                "stID",
                "entity_assemID",
                "entityID",
                "entity_name",
                "seq",
                "k",
            ]
            + score_types
            + shifts
        ]
        dout["seq"] = dout.seq.apply(lambda x: list(x))
        dout["seq_index"] = dout.seq.apply(lambda x: list(range(1, len(x) + 1)))
        dout = dout.explode(["seq_index", "seq", "k"] + score_types + shifts)
        dout[
            [
                "ID",
                "entryID",
                "stID",
                "entity_assemID",
                "entityID",
                "entity_name",
                "seq_index",
                "seq",
                "k",
            ]
            + score_types
            + shifts
        ].to_csv(
            output_prefix.parent / f"{output_prefix.name}.csv",
            float_format=f"%.{precision}f",
        )
    elif output_format == "json":
        dout = df.loc[df.pass_post].reset_index()[
            [
                "ID",
                "entryID",
                "stID",
                "entity_assemID",
                "entityID",
                "entity_name",
                "exp_method",
                "exp_method_subtype",
                "sample_state_evidence",
                "physical_state",
                "denaturant_evidence",
                "membrane_mimetic",
                "citation_DOI",
                "citation_title",
                "ionic_strength",
                "pH",
                "temperature",
                "off_C",
                "off_CA",
                "off_CB",
                "off_H",
                "off_HA",
                "off_HB",
                "off_N",
                "lacs_off_C",
                "lacs_off_CA",
                "lacs_off_CB",
                "lacs_off_H",
                "lacs_off_HA",
                "lacs_off_HB",
                "lacs_off_N",
                "bbshift_positions_post",
                "bbshift_types_post",
                "total_bbshifts",
                "seq",
                "k",
            ]
            + score_types
            + shifts
        ]
        dout.to_json(
            output_prefix.parent / f"{output_prefix.name}.json",
            orient="records",
            lines=True,
        )
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def run_scoring_pipeline(args):
    if args.processes is None:
        pandarallel.initialize(verbose=0, progress_bar=args.progress)
    else:
        pandarallel.initialize(
            verbose=0, nb_workers=args.processes, progress_bar=args.progress
        )
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s : %(message)s")
    # reject most logging messages for sub-routines like parsing database files:
    logging.getLogger("trizod.bmrb").setLevel(logging.CRITICAL)
    logging.getLogger("trizod.scoring").setLevel(logging.CRITICAL)

    logging.getLogger("trizod").info("Loading BMRB files.")
    bmrb_files = find_bmrb_files(args.input_dir, args.BMRB_file_pattern)
    bmrb_entries, failed = load_bmrb_entries(bmrb_files, cache_dir=args.cache_dir)
    print()
    if failed:
        logging.getLogger("trizod").warning(
            f"Failed loading {len(failed)} of {len(bmrb_files)} BMRB files"
        )
    logging.getLogger("trizod").info("Parsing and filtering relevant information.")
    df = create_peptide_dataframe(
        bmrb_entries,
        chemical_denaturants=args.chemical_denaturants,
        keywords=args.keywords_blacklist,
        return_default=args.default_conditions,
        assume_si=args.unit_assumptions,
        fix_outliers=args.unit_corrections,
        include_shifts=args.include_shifts,
        no_shift_averaging=args.no_shift_averaging,
        keyword_search_scope=args.keyword_search_scope,
        progress=args.progress,
    )
    df, missing_vals, sels_pre, sels_kws, sels_denat, sels_paramag, sels_all_pre = (
        prefilter_dataframe(
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
            max_noncanonical_fraction=args.max_noncanonical_fraction,
            max_x_fraction=args.max_x_fraction,
            keywords=args.keywords_blacklist,
            chemical_denaturants=args.chemical_denaturants,
            exclude_paramagnetic=args.exclude_paramagnetic,
            physical_state_blacklist=args.physical_state_blacklist,
            method_fallback=args.method_fallback,
        )
    )
    print()
    logging.getLogger("trizod").info("Computing scores for each remaining entry.")
    df = df.parallel_apply(
        compute_scores_row,
        axis=1,
        score_types=args.score_types,
        offset_correction=args.offset_correction,
        max_offset=args.max_offset,
        reject_shift_type_only=args.reject_shift_type_only,
        cache_dir=args.cache_dir,
        rereference_mode=args.rereference_mode,
        bmrb_entries=bmrb_entries,
    )
    if args.progress:
        print()  # prevents overwriting last line of progress bars
    for score_type in args.score_types:
        if score_type not in df:
            # compute_scores_row only creates the score columns for rows that
            # passed the pre-filter, so a run where nothing passes used to reach
            # postfilter_dataframe with no `zscores` column at all and die on a
            # KeyError instead of writing an empty dataset.
            df[score_type] = None
    logging.getLogger("trizod").info("Filtering results.")
    sels_post, sels_off, sels_all_post = postfilter_dataframe(
        df,
        min_backbone_shift_types=args.min_backbone_shift_types,
        min_backbone_shift_positions=args.min_backbone_shift_positions,
        min_backbone_shift_fraction=args.min_backbone_shift_fraction,
        reject_shift_type_only=args.reject_shift_type_only,
        score_types=args.score_types,
    )
    if args.emit_str is not None:
        from trizod.io.str_writer import write_rereferenced_str

        logging.getLogger("trizod").info(
            f"Emitting re-referenced .str files to {args.emit_str}"
        )
        passed = df[df["pass_post"]]
        for _, row in tqdm(
            passed.iterrows(), total=len(passed), disable=not args.progress
        ):
            entry = bmrb_entries.loc[row["entryID"], "entry"]
            peptide_shifts = entry.get_peptide_shifts()
            shifts, _, _, _ = peptide_shifts[
                (row["stID"], row["entity_assemID"], row["entityID"])
            ]
            seq = row["seq"]
            ret = bmrb.get_valid_bbshifts(
                shifts, seq, averaging=not args.no_shift_averaging
            )
            if ret is None:
                continue
            bbshifts_arr, bbshifts_mask = ret
            # The corrected shifts emitted are post-LACS. Subtract per-atom
            # LACS offsets so the file reflects the re-referenced state. The
            # POTENCI residual offsets live on the weighted-diff side and are
            # captured in the aux saveframe rather than subtracted from raw shifts.
            for j, atom in enumerate(BACKBONE_ATOMS):
                lacs_off = row.get(f"lacs_off_{atom}", 0.0)
                if pd.isna(lacs_off):
                    lacs_off = 0.0
                bbshifts_arr[bbshifts_mask[:, j], j] -= float(lacs_off)
            lacs_offsets = {
                atom: (
                    0.0
                    if pd.isna(row.get(f"lacs_off_{atom}", 0.0))
                    else float(row[f"lacs_off_{atom}"])
                )
                for atom in BACKBONE_ATOMS
            }
            potenci_offsets = {
                atom: (
                    0.0
                    if pd.isna(row.get(f"off_{atom}", 0.0))
                    else float(row[f"off_{atom}"])
                )
                for atom in BACKBONE_ATOMS
            }
            out_path = args.emit_str / (
                f"bmr{row['entryID']}_{row['stID']}_{row['entity_assemID']}"
                f"_{row['entityID']}_rereferenced.str"
            )
            write_rereferenced_str(
                out_path,
                entry_id=row["entryID"],
                seq=seq,
                bbshifts_arr=bbshifts_arr,
                bbshifts_mask=bbshifts_mask,
                lacs_offsets=lacs_offsets,
                potenci_residual_offsets=potenci_offsets,
                rereference_mode=args.rereference_mode,
                pipeline_version=pipeline_version(),
            )

    logging.getLogger("trizod").info("Output filtering results.")
    print_filter_losses(
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
    )
    logging.getLogger("trizod").info("Writing dataset to file.")
    output_dataset(
        df,
        args.output_prefix,
        args.output_format,
        args.score_types,
        args.precision,
        args.include_shifts,
        args.no_shift_averaging,
    )


if __name__ == "__main__":
    from trizod.cli.main import app

    app()
