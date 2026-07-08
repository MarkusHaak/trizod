"""TriZOD CLI — Typer entry point.

A single ``score`` command that mirrors the historical argparse flag surface 1:1,
so both ``trizod <flags>`` (Typer's single-command mode) and
``python -m trizod.trizod <flags>`` behave exactly as before. Filter options
default to a sentinel (``None``) and are resolved from the ``--filter-defaults``
tier preset, reproducing the old two-phase argparse behaviour. Subcommands
(``dataset``, ``figures``, …) will be added as that code is promoted into the
package; adding a second command switches the CLI to ``trizod <command> …``.
"""

import logging
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import typer

from trizod.trizod import filter_defaults, run_scoring_pipeline

app = typer.Typer(
    rich_markup_mode=None,
    add_completion=False,
    help="Modern disorder scores for BMRB data.",
)

_LOG = logging.getLogger("trizod")


class OutputFormat(str, Enum):
    json = "json"
    csv = "csv"


class FilterTier(str, Enum):
    unfiltered = "unfiltered"
    tolerant = "tolerant"
    moderate = "moderate"
    strict = "strict"


class RereferenceMode(str, Enum):
    none = "none"
    lacs = "lacs"
    potenci_only = "potenci-only"
    both = "both"


class ScoreType(str, Enum):
    zscores = "zscores"
    gscores = "gscores"


@app.command()
def score(
    input_dir: str = typer.Option(
        ".",
        "--input-dir",
        "-d",
        help="Directory that is searched recursively for BMRB .str files.",
    ),
    output_prefix: str = typer.Option(
        "./trizod_dataset",
        "--output-prefix",
        help="Prefix (and path) of the created output file.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.csv, "--output-format", help="Output file format."
    ),
    cache_dir: str = typer.Option(
        "./tmp",
        "--cache-dir",
        help="Create and use cache files in the given directory to acelerate repeated execution.",
    ),
    bmrb_file_pattern: str = typer.Option(
        r"bmr(\d+)_3\.str",
        "--BMRB-file-pattern",
        help="regular expression pattern for BMRB files.",
    ),
    include_shifts: bool = typer.Option(
        False,
        "--include-shifts",
        help="Add raw backbone atom shift data to the output.",
    ),
    no_shift_averaging: bool = typer.Option(
        False,
        "--no-shift-averaging",
        help="Do not average over Proton groups for HA and HB shifts.",
    ),
    emit_str: Optional[str] = typer.Option(
        None,
        "--emit-str",
        help=(
            "Directory to write re-referenced NMR-STAR (.str) files into. "
            "One file per scored entry: <dir>/bmr<id>_rereferenced.str. Off by default."
        ),
    ),
    filter_defaults_tier: FilterTier = typer.Option(
        FilterTier.tolerant,
        "--filter-defaults",
        help="Sets defaults for all filter options, from unfiltered to strict.",
    ),
    temperature_range: Optional[tuple[float, float]] = typer.Option(
        None, "--temperature-range", help="Minimum and maximum temperature in Kelvin."
    ),
    ionic_strength_range: Optional[tuple[float, float]] = typer.Option(
        None,
        "--ionic-strength-range",
        help="Minimum and maximum ionic strength in Mol.",
    ),
    ph_range: Optional[tuple[float, float]] = typer.Option(
        None, "--pH-range", help="Minimum and maximum pH."
    ),
    unit_assumptions: Optional[bool] = typer.Option(
        None,
        "--unit-assumptions/--no-unit-assumptions",
        help="Assume units for Temp., Ionic str. and pH if they are not given and exclude entries instead.",
    ),
    unit_corrections: Optional[bool] = typer.Option(
        None,
        "--unit-corrections/--no-unit-corrections",
        help="Correct values for Temp., Ionic str. and pH if units are most likely wrong.",
    ),
    default_conditions: Optional[bool] = typer.Option(
        None,
        "--default-conditions/--no-default-conditions",
        help="Assume standard conditions if pH (7), ionic strength (0.1 M) or temperature (298 K) are missing and exclude entries instead.",
    ),
    peptide_length_range: Optional[list[int]] = typer.Option(
        None,
        "--peptide-length-range",
        help="Minimum (and optionally maximum) peptide sequence length.",
    ),
    min_backbone_shift_types: Optional[int] = typer.Option(
        None,
        "--min-backbone-shift-types",
        help="Minimum number of different backbone shift types (max 7).",
    ),
    min_backbone_shift_positions: Optional[int] = typer.Option(
        None,
        "--min-backbone-shift-positions",
        help="Minimum number of positions with at least one backbone shift.",
    ),
    min_backbone_shift_fraction: Optional[float] = typer.Option(
        None,
        "--min-backbone-shift-fraction",
        help="Minimum fraction of positions with at least one backbone shift.",
    ),
    max_noncanonical_fraction: Optional[float] = typer.Option(
        None,
        "--max-noncanonical-fraction",
        help="Maximum fraction of non-canonical amino acids (X count as arbitrary canonical) in the amino acid sequence.",
    ),
    max_x_fraction: Optional[float] = typer.Option(
        None,
        "--max-x-fraction",
        help="Maximum fraction of X letters (arbitrary canonical amino acid) in the amino acid sequence.",
    ),
    keywords_blacklist: Optional[list[str]] = typer.Option(
        None,
        "--keywords-blacklist",
        help="Exclude entries with any of these keywords mentioned anywhere in the BMRB file, case ignored.",
    ),
    chemical_denaturants: Optional[list[str]] = typer.Option(
        None,
        "--chemical-denaturants",
        help="Exclude entries with any of these chemicals as substrings of sample components, case ignored.",
    ),
    exp_method_whitelist: Optional[list[str]] = typer.Option(
        None,
        "--exp-method-whitelist",
        help="Include only entries with any of these keywords as substring of the experiment subtype, case ignored.",
    ),
    exp_method_blacklist: Optional[list[str]] = typer.Option(
        None,
        "--exp-method-blacklist",
        help="Exclude entries with any of these keywords as substring of the experiment subtype, case ignored.",
    ),
    exclude_paramagnetic: Optional[bool] = typer.Option(
        None,
        "--exclude-paramagnetic/--no-exclude-paramagnetic",
        help="Exclude entries flagged as paramagnetic in the BMRB assembly or entity metadata.",
    ),
    score_types: list[ScoreType] = typer.Option(
        [ScoreType.zscores, ScoreType.gscores],
        "--score-types",
        help="Which type of scores are created: observation-count-independent zscores (zscores) or geometric mean of observation probabilities (gscores).",
    ),
    offset_correction: bool = typer.Option(
        True,
        "--offset-correction/--no-offset-correction",
        help="Compute correction offsets for random coil chemical shifts.",
    ),
    rereference_mode: RereferenceMode = typer.Option(
        RereferenceMode.both,
        "--rereference-mode",
        help=(
            "Chemical shift re-referencing strategy. 'none': no correction (raw shifts). "
            "'lacs': LACS pre-correction only. 'potenci-only': legacy POTENCI/AIC offset "
            "detection only. 'both' (default): LACS pre-correction followed by POTENCI/AIC residual."
        ),
    ),
    max_offset: Optional[float] = typer.Option(
        None,
        "--max-offset",
        help="Maximum valid offset correction for any random coil chemical shift type.",
    ),
    reject_shift_type_only: Optional[bool] = typer.Option(
        None,
        "--reject-shift-type-only/--no-reject-shift-type-only",
        help="Upon exceeding the maximal offset set by <--max-offset>, exclude only the backbone shifts exceeding the offset instead of the whole entry.",
    ),
    precision: int = typer.Option(
        4,
        "--precision",
        help="Number of decimal digits that are output to human readable files.",
    ),
    processes: int = typer.Option(
        8, "--processes", help="Number of processes to spawn in multiprocessing."
    ),
    progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Show progress bars."
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
):
    """Score BMRB entries for per-residue disorder (Z-scores / G-scores)."""
    tier = filter_defaults.loc[filter_defaults_tier.value]

    def resolve(value, key):
        return tier[key] if value is None else value

    args = SimpleNamespace(
        input_dir=input_dir,
        output_prefix=output_prefix,
        output_format=output_format.value,
        cache_dir=cache_dir,
        BMRB_file_pattern=bmrb_file_pattern,
        include_shifts=include_shifts,
        no_shift_averaging=no_shift_averaging,
        emit_str=emit_str,
        temperature_range=list(resolve(temperature_range, "temperature-range")),
        ionic_strength_range=list(
            resolve(ionic_strength_range, "ionic-strength-range")
        ),
        pH_range=list(resolve(ph_range, "pH-range")),
        unit_assumptions=bool(resolve(unit_assumptions, "unit-assumptions")),
        unit_corrections=bool(resolve(unit_corrections, "unit-corrections")),
        default_conditions=bool(resolve(default_conditions, "default-conditions")),
        peptide_length_range=list(
            resolve(peptide_length_range, "peptide-length-range")
        ),
        min_backbone_shift_types=int(
            resolve(min_backbone_shift_types, "min-backbone-shift-types")
        ),
        min_backbone_shift_positions=int(
            resolve(min_backbone_shift_positions, "min-backbone-shift-positions")
        ),
        min_backbone_shift_fraction=float(
            resolve(min_backbone_shift_fraction, "min-backbone-shift-fraction")
        ),
        max_noncanonical_fraction=float(
            resolve(max_noncanonical_fraction, "max-noncanonical-fraction")
        ),
        max_x_fraction=float(resolve(max_x_fraction, "max-x-fraction")),
        keywords_blacklist=list(resolve(keywords_blacklist, "keywords-blacklist")),
        chemical_denaturants=list(
            resolve(chemical_denaturants, "chemical-denaturants")
        ),
        exp_method_whitelist=list(
            resolve(exp_method_whitelist, "exp-method-whitelist")
        ),
        exp_method_blacklist=list(
            resolve(exp_method_blacklist, "exp-method-blacklist")
        ),
        exclude_paramagnetic=bool(
            resolve(exclude_paramagnetic, "exclude-paramagnetic")
        ),
        score_types=[s.value for s in score_types],
        offset_correction=offset_correction,
        rereference_mode=rereference_mode.value,
        max_offset=float(resolve(max_offset, "max-offset")),
        reject_shift_type_only=bool(
            resolve(reject_shift_type_only, "reject-shift-type-only")
        ),
        precision=precision,
        processes=processes,
        progress=progress,
        debug=debug,
    )

    _validate_and_prepare_paths(args)
    run_scoring_pipeline(args)


def _validate_and_prepare_paths(args):
    """Port of the argparse post-parse validation: resolve/create I/O paths."""
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        _LOG.error(f"Input directory {input_dir} does not exist.")
        raise typer.Exit(1)
    if not input_dir.is_dir():
        _LOG.error(f"Path {input_dir} is not a directory.")
        raise typer.Exit(1)
    args.input_dir = input_dir.resolve()

    args.output_prefix = Path(args.output_prefix).resolve()
    if not args.output_prefix.parent.exists():
        _LOG.error(f"Output directory {args.output_prefix.parent} does not exist.")
        raise typer.Exit(1)

    if len(args.peptide_length_range) == 1:
        args.peptide_length_range.append(np.inf)

    args.cache_dir = Path(args.cache_dir).resolve()
    subdirs = [
        args.cache_dir,
        args.cache_dir / "wSCS",
        args.cache_dir / "bmrb_entries",
        args.cache_dir / "potenci",
    ]
    if not all(d.exists() for d in subdirs):
        if not args.cache_dir.exists():
            _LOG.debug(f"Directory {args.cache_dir} does not exist and is created.")
        for d in subdirs:
            d.mkdir(parents=True, exist_ok=True)
    elif not args.cache_dir.is_dir():
        _LOG.error(f"Path {args.cache_dir} is not a directory.")
        raise typer.Exit(1)

    if args.emit_str is not None:
        args.emit_str = Path(args.emit_str).resolve()
        args.emit_str.mkdir(parents=True, exist_ok=True)
