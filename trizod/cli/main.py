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

import numpy as np
import typer

from trizod.dataset.mmseqs import (
    DEFAULT_LEAKAGE_RULE,
    LEAKAGE_RULES,
    LEAKAGE_RULES_HELP,
)
from trizod.dataset.release_info import DATASET_VERSION

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


class KeywordSearchScope(str, Enum):
    sample = "sample"
    all = "all"


#: Stage-2 test-set leakage rule: one member per
#: ``trizod.dataset.mmseqs.LEAKAGE_RULES``, so the two cannot drift apart.
LeakageRule = Enum("LeakageRule", {rule: rule for rule in LEAKAGE_RULES}, type=str)


class MethodFallback(str, Enum):
    off = "off"
    reject_solid = "reject-solid"
    require_solution = "require-solution"


@app.callback(invoke_without_command=True)
def score(
    ctx: typer.Context,
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
    emit_str: str | None = typer.Option(
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
    temperature_range: tuple[float, float] | None = typer.Option(
        None, "--temperature-range", help="Minimum and maximum temperature in Kelvin."
    ),
    ionic_strength_range: tuple[float, float] | None = typer.Option(
        None,
        "--ionic-strength-range",
        help="Minimum and maximum ionic strength in Mol.",
    ),
    ph_range: tuple[float, float] | None = typer.Option(
        None, "--pH-range", help="Minimum and maximum pH."
    ),
    unit_assumptions: bool | None = typer.Option(
        None,
        "--unit-assumptions/--no-unit-assumptions",
        help="Assume units for Temp., Ionic str. and pH if they are not given and exclude entries instead.",
    ),
    unit_corrections: bool | None = typer.Option(
        None,
        "--unit-corrections/--no-unit-corrections",
        help="Correct values for Temp., Ionic str. and pH if units are most likely wrong.",
    ),
    default_conditions: bool | None = typer.Option(
        None,
        "--default-conditions/--no-default-conditions",
        help="Assume standard conditions if pH (7), ionic strength (0.1 M) or temperature (298 K) are missing and exclude entries instead.",
    ),
    peptide_length_range: list[int] | None = typer.Option(
        None,
        "--peptide-length-range",
        help="Minimum (and optionally maximum) peptide sequence length.",
    ),
    min_backbone_shift_types: int | None = typer.Option(
        None,
        "--min-backbone-shift-types",
        help="Minimum number of different backbone shift types (max 7).",
    ),
    min_backbone_shift_positions: int | None = typer.Option(
        None,
        "--min-backbone-shift-positions",
        help="Minimum number of positions with at least one backbone shift.",
    ),
    min_backbone_shift_fraction: float | None = typer.Option(
        None,
        "--min-backbone-shift-fraction",
        help="Minimum fraction of positions with at least one backbone shift.",
    ),
    max_noncanonical_fraction: float | None = typer.Option(
        None,
        "--max-noncanonical-fraction",
        help="Maximum fraction of non-canonical amino acids (X count as arbitrary canonical) in the amino acid sequence.",
    ),
    max_x_fraction: float | None = typer.Option(
        None,
        "--max-x-fraction",
        help="Maximum fraction of X letters (arbitrary canonical amino acid) in the amino acid sequence.",
    ),
    keywords_blacklist: list[str] | None = typer.Option(
        None,
        "--keywords-blacklist",
        help="Exclude entries with any of these keywords as a substring of a searched free-text field, case ignored. See --keyword-search-scope.",
    ),
    keyword_search_scope: KeywordSearchScope | None = typer.Option(
        None,
        "--keyword-search-scope",
        help=(
            "Which free-text fields --keywords-blacklist searches. 'sample' (default): "
            "only fields describing the deposited sample (entry title/details, assembly "
            "and entity name/details, sample name/details/framecode). 'all': also the "
            "paper-topic fields (citation title and keywords, struct keywords), which "
            "describe the publication rather than the NMR tube."
        ),
    ),
    physical_state_blacklist: list[str] | None = typer.Option(
        None,
        "--physical-state-blacklist",
        help="Exclude entries whose _Entity_assembly.Physical_state EXACTLY equals one of these values, case ignored.",
    ),
    perturbing_cosolvents: list[str] | None = typer.Option(
        None,
        "--perturbing-cosolvents",
        help=(
            "Exclude entries with any of these chemicals as substrings of sample "
            "components, case ignored. Urea/GdmCl, TFE/HFIP and DMSO all take the "
            "sample out of aqueous buffer, where POTENCI and the LACS reference "
            "tables are parameterised; note the two halves bias in opposite "
            "directions (denaturants toward apparent disorder, the alcohols and "
            "DMSO toward apparent order). Stabilising osmolytes such as TMAO and "
            "glycerol are deliberately not in this family."
        ),
    ),
    chemical_denaturants: list[str] | None = typer.Option(
        None,
        "--chemical-denaturants",
        hidden=True,
        help="DEPRECATED alias of --perturbing-cosolvents. Will be removed after one release.",
    ),
    exp_method_whitelist: list[str] | None = typer.Option(
        None,
        "--exp-method-whitelist",
        help="Include only entries with any of these keywords as substring of the experiment subtype, case ignored.",
    ),
    exp_method_blacklist: list[str] | None = typer.Option(
        None,
        "--exp-method-blacklist",
        help="Exclude entries with any of these keywords as substring of the experiment subtype, case ignored.",
    ),
    method_fallback: MethodFallback | None = typer.Option(
        None,
        "--method-fallback",
        help=(
            "How to use the sample-state evidence (_Sample.Type / "
            "_Experiment.Sample_state / the experiment names). 'off': the "
            "whitelist alone decides. 'reject-solid': reject any row with solid "
            "evidence, whatever its declared subtype says. 'require-solution': "
            "the same veto, and entries with no _Entry.Experimental_method_subtype "
            "are admitted only on positive solution evidence."
        ),
    ),
    exclude_paramagnetic: bool | None = typer.Option(
        None,
        "--exclude-paramagnetic/--no-exclude-paramagnetic",
        help="Exclude entries flagged as paramagnetic in the BMRB assembly or entity metadata.",
    ),
    max_pressure_bar: float | None = typer.Option(
        None,
        "--max-pressure-bar",
        help=(
            "Exclude entries whose scored shift table's sample condition list "
            "records a pressure at or above this value, in bar. A pressure that "
            "is not recorded is taken as ambient and passes."
        ),
    ),
    exclude_membrane_mimetic: bool | None = typer.Option(
        None,
        "--exclude-membrane-mimetic/--no-exclude-membrane-mimetic",
        help=(
            "Exclude entries whose sample is a membrane mimetic: a bilayer "
            "lipid or assembly, or a dedicated micelle former (SDS, DPC, DHPC, "
            "LMPG, LDAO, ...) at any concentration, or an additive-class "
            "detergent (CHAPS, DDM, Triton, ...) at or above its CMC. See "
            "trizod/bmrb/membrane.py."
        ),
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
    max_offset: float | None = typer.Option(
        None,
        "--max-offset",
        help=(
            "Maximum valid offset correction for any random coil chemical shift "
            "type, in SIGMA units (multiples of the per-atom POTENCI RMSD), not "
            "ppm. Compared against the emitted off_<atom>_sigma column."
        ),
    ),
    reject_shift_type_only: bool | None = typer.Option(
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
    """Score BMRB entries for per-residue disorder (Z-scores / G-scores).

    Runs when trizod is invoked with no subcommand (bare ``trizod <flags>``);
    ``trizod dataset ...`` subcommands are handled separately.
    """
    if ctx.invoked_subcommand is not None:
        return
    # Lazy import: keeps `trizod --help` and `trizod dataset ...` from pulling in
    # the scoring stack (pandarallel, potenci, scoring) that they never use.
    from trizod.trizod import filter_defaults, run_scoring_pipeline

    tier = filter_defaults.loc[filter_defaults_tier.value]

    def resolve(value, key):
        return tier[key] if value is None else value

    # `--chemical-denaturants` was renamed to `--perturbing-cosolvents` in
    # 2026-08: the list is not (only) denaturants, and the two halves of it bias
    # the score in opposite directions -- see the naming note in
    # `trizod/trizod.py` beside COSOLVENT_TOKENS. Kept as a deprecated alias for
    # one release. A separate parameter rather than a second option string on
    # `--perturbing-cosolvents`, because Click reports which *parameter* was
    # supplied, not which spelling, and the deprecation warning has to be able
    # to tell.
    if chemical_denaturants is not None:
        _LOG.warning(
            "--chemical-denaturants is deprecated and will be removed after one "
            "release; use --perturbing-cosolvents instead."
        )
        if perturbing_cosolvents is None:
            perturbing_cosolvents = chemical_denaturants

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
        keyword_search_scope=str(
            resolve(
                keyword_search_scope.value if keyword_search_scope else None,
                "keyword-search-scope",
            )
        ),
        physical_state_blacklist=list(
            resolve(physical_state_blacklist, "physical-state-blacklist")
        ),
        perturbing_cosolvents=list(
            resolve(perturbing_cosolvents, "perturbing-cosolvents")
        ),
        exp_method_whitelist=list(
            resolve(exp_method_whitelist, "exp-method-whitelist")
        ),
        exp_method_blacklist=list(
            resolve(exp_method_blacklist, "exp-method-blacklist")
        ),
        method_fallback=str(
            resolve(
                method_fallback.value if method_fallback else None, "method-fallback"
            )
        ),
        exclude_paramagnetic=bool(
            resolve(exclude_paramagnetic, "exclude-paramagnetic")
        ),
        max_pressure_bar=float(resolve(max_pressure_bar, "max-pressure-bar")),
        exclude_membrane_mimetic=bool(
            resolve(exclude_membrane_mimetic, "exclude-membrane-mimetic")
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
    if not args.cache_dir.exists():
        _LOG.debug(f"Directory {args.cache_dir} does not exist and is created.")
    for d in (
        args.cache_dir,
        args.cache_dir / "wSCS",
        args.cache_dir / "bmrb_entries",
        args.cache_dir / "potenci",
    ):
        d.mkdir(parents=True, exist_ok=True)

    if args.emit_str is not None:
        args.emit_str = Path(args.emit_str).resolve()
        args.emit_str.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# `trizod dataset ...` — the dataset-build chain. Thin wrappers that forward to
# the argparse `main(argv)` of each trizod.dataset module, so every option stays
# defined in exactly one place (the module).
# --------------------------------------------------------------------------- #

dataset_app = typer.Typer(
    rich_markup_mode=None,
    help=(
        "Build the redundancy-reduced TriZOD dataset, its training sets "
        "separated from the test sets at 30 % identity over 80 % of either "
        "chain (searched in both directions), and assemble the deposit."
    ),
)
app.add_typer(dataset_app, name="dataset")


def _wd_argv(work_dir, root):
    argv = []
    if work_dir is not None:
        argv += ["--work-dir", work_dir]
    if root is not None:
        argv += ["--root", root]
    return argv


@dataset_app.command("build")
def _dataset_build(
    work_dir: str | None = typer.Option(None, "--work-dir"),
    root: str | None = typer.Option(None, "--root"),
    exclude_homo_oligomers: bool = typer.Option(
        False,
        "--exclude-homo-oligomers",
        help=(
            "Also drop rows whose entity appears on more than one "
            "_Entity_assembly record (n_copies >= 2). Off by default: "
            "oligomeric state is annotated, not filtered."
        ),
    ),
):
    """Bound-removal + exact-seq dedup + quality ranking (-> final_dataset/)."""
    from trizod.dataset import build

    argv = _wd_argv(work_dir, root)
    if exclude_homo_oligomers:
        argv.append("--exclude-homo-oligomers")
    build.main(argv)


@dataset_app.command("test-set")
def _dataset_testset(
    work_dir: str | None = typer.Option(None, "--work-dir"),
    root: str | None = typer.Option(None, "--root"),
    redraw: bool = typer.Option(
        False,
        "--redraw",
        help="Redraw the seeded test set and overwrite the committed pin.",
    ),
    confirm_redraw: bool = typer.Option(
        False,
        "--confirm-redraw",
        help=(
            "Required alongside --redraw. Redrawing breaks comparability with "
            "every released version and with anything already trained on them."
        ),
    ),
    retirements: str | None = typer.Option(
        None,
        "--retirements",
        help=(
            "Reviewed retirement table: pinned test chains retired by decision "
            "(default: the committed trizod/dataset/data/test_retirements.tsv)."
        ),
    ),
):
    """Emit the pinned TriZOD test set (-> testset/); --redraw re-establishes the pin."""
    from trizod.dataset import testset

    argv = _wd_argv(work_dir, root)
    if retirements is not None:
        argv += ["--retirements", retirements]
    if redraw:
        argv.append("--redraw")
    # Without this the guard in testset.main() is unreachable from the supported
    # CLI: --redraw alone always aborts, so a deliberate redraw had to go around
    # the CLI entirely.
    if confirm_redraw:
        argv.append("--confirm-redraw")
    testset.main(argv)


_LEAKAGE_RULE_HELP = f"Stage-2 test-set leakage rule ({LEAKAGE_RULES_HELP})."


@dataset_app.command("redundancy")
def _dataset_redundancy(
    work_dir: str | None = typer.Option(None, "--work-dir"),
    root: str | None = typer.Option(None, "--root"),
    leakage_rule: LeakageRule | None = typer.Option(
        None,
        "--leakage-rule",
        help=(
            f"{_LEAKAGE_RULE_HELP} Default: {DEFAULT_LEAKAGE_RULE}. "
            "Recorded in mmseqs/."
        ),
    ),
):
    """Two-stage test-set leakage removal + mmseqs clustering (-> mmseqs/)."""
    from trizod.dataset import redundancy

    argv = _wd_argv(work_dir, root)
    if leakage_rule is not None:
        argv += ["--leakage-rule", leakage_rule.value]
    redundancy.main(argv)


@dataset_app.command("representatives")
def _dataset_representatives(
    work_dir: str | None = typer.Option(None, "--work-dir"),
    root: str | None = typer.Option(None, "--root"),
):
    """Override mmseqs cluster reps with the quality-best cluster member."""
    from trizod.dataset import representatives

    representatives.main(_wd_argv(work_dir, root))


@dataset_app.command("package")
def _dataset_package(
    version: str = typer.Option(DATASET_VERSION, "--version"),
    include_str: bool = typer.Option(False, "--include-str"),
    out: str | None = typer.Option(None, "--out"),
    work_dir: str | None = typer.Option(None, "--work-dir"),
    root: str | None = typer.Option(None, "--root"),
    leakage_rule: LeakageRule | None = typer.Option(
        None,
        "--leakage-rule",
        help=_LEAKAGE_RULE_HELP
        + " Default: the rule the redundancy step recorded; a different value "
        "is an error.",
    ),
):
    """Stage the package bundle (input of the Parquet) + MANIFEST + leakage gate."""
    from trizod.dataset import package_release

    argv = ["--version", version]
    if include_str:
        argv += ["--include-str"]
    if out is not None:
        argv += ["--out", out]
    if leakage_rule is not None:
        argv += ["--leakage-rule", leakage_rule.value]
    package_release.main(argv + _wd_argv(work_dir, root))


@dataset_app.command("deploy")
def _dataset_deploy(
    tier: str = typer.Option("tolerant", "--tier"),
    val_fraction: float = typer.Option(0.0, "--val-fraction"),
    seed: int = typer.Option(42, "--seed"),
    out: str | None = typer.Option(None, "--out"),
    work_dir: str | None = typer.Option(None, "--work-dir"),
    root: str | None = typer.Option(None, "--root"),
):
    """Build the UdonPred-handoff deployment FASTA (per-residue G-score labels)."""
    from trizod.dataset import deploy_fasta

    argv = ["--tier", tier, "--val-fraction", str(val_fraction), "--seed", str(seed)]
    if out is not None:
        argv += ["--out", out]
    deploy_fasta.main(argv + _wd_argv(work_dir, root))


@dataset_app.command("export-jsonl")
def _dataset_export_jsonl(
    parquet: str = typer.Option(..., "--parquet", help="Release Parquet."),
    out: str | None = typer.Option(
        None, "--out", help="Output directory (default: <parquet dir>/udonpred)."
    ),
    tier: list[str] | None = typer.Option(
        None,
        "--tier",
        help="Training tier to export, repeatable (default: tolerant, moderate, strict).",
    ),
    valid_fraction: float | None = typer.Option(
        None,
        "--valid-fraction",
        help=(
            "Fraction of each tier's homology components drawn into valid "
            "(default: 0.10)."
        ),
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Seed of the valid draw (default: 42)."
    ),
):
    """Write the UdonPred v2 JSONL view ({train,valid,test}.jsonl per tier) of the Parquet."""
    from trizod.dataset import export_jsonl

    argv = ["--parquet", parquet]
    if out is not None:
        argv += ["--out", out]
    for t in tier or []:
        argv += ["--tier", t]
    if valid_fraction is not None:
        argv += ["--valid-fraction", str(valid_fraction)]
    if seed is not None:
        argv += ["--seed", str(seed)]
    export_jsonl.main(argv)


@dataset_app.command("test-crosswalk")
def _dataset_test_crosswalk(
    parquet: str = typer.Option(..., "--parquet", help="Final release Parquet."),
    out: str | None = typer.Option(
        None, "--out", help="Output directory (default: <parquet dir>/crosswalk)."
    ),
    root: str | None = typer.Option(
        None, "--root", help="Directory holding data/ for the default earlier sets."
    ),
    sets: list[str] | None = typer.Option(
        None,
        "--set",
        help="NAME=fasta:PATH|parquet:PATH|git:REV:PATH, repeatable; replaces the defaults.",
    ),
):
    """Crosswalk of every TriZOD test set and its status in the final release."""
    from trizod.dataset import testset_crosswalk

    argv = ["--parquet", parquet]
    if out is not None:
        argv += ["--out", out]
    if root is not None:
        argv += ["--root", root]
    for s in sets or []:
        argv += ["--set", s]
    testset_crosswalk.main(argv)


@dataset_app.command("finalize")
def _dataset_finalize(
    parquet: str = typer.Option(..., "--parquet", help="Release Parquet."),
    shifts: str = typer.Option(..., "--shifts", help="trizod_shifts.parquet."),
    bundle: str = typer.Option(
        ..., "--bundle", help="Package bundle the Parquet was built from."
    ),
    bmrb_dir: str = typer.Option(
        ..., "--bmrb-dir", help="BMRB snapshot directory (per-file SHA-256 table)."
    ),
    out: str = typer.Option(..., "--out", help="Release folder to write."),
    crosswalk_root: str | None = typer.Option(
        None,
        "--crosswalk-root",
        help="Checkout holding data/ for the crosswalk's earlier test sets.",
    ),
    sets: list[str] | None = typer.Option(
        None,
        "--set",
        help="NAME=fasta:PATH|parquet:PATH|git:REV:PATH, repeatable; replaces "
        "the crosswalk's default earlier sets.",
    ),
    validation: str | None = typer.Option(
        None, "--validation", help="VALIDATION.json of the final audit."
    ),
    allow_uncalibrated: bool = typer.Option(
        False,
        "--allow-uncalibrated",
        help="Accept a Parquet built without --calibration (dry runs only).",
    ),
    allow_calibration_mismatch: bool = typer.Option(
        False,
        "--allow-calibration-mismatch",
        help="Accept a Parquet whose data differ from the calibration's input "
        "(dry runs only).",
    ),
    force: bool = typer.Option(False, "--force", help="Replace a non-empty --out."),
    previous_build: str | None = typer.Option(
        None,
        "--previous-build",
        help="Previous build's package bundle: tier moves, re-score statistics "
        "and sequence changes against it.",
    ),
    previous_label: str | None = typer.Option(
        None, "--previous-label", help="How the CHANGELOG names --previous-build."
    ),
    pkl_dir: str | None = typer.Option(
        None,
        "--pkl-dir",
        help="The build's BmrbEntry pickle cache: pH*/pD, temperature and "
        "pressure-unit issues per chain.",
    ),
    snapshot_drift: str | None = typer.Option(
        None, "--snapshot-drift", help="Snapshot-drift table of the release audit."
    ),
    construct_caveats: str | None = typer.Option(
        None,
        "--construct-caveats",
        help="Construct / sample caveat table of the release audit (class, id).",
    ),
    ionic_drivers: str | None = typer.Option(
        None,
        "--ionic-drivers",
        help="Ionic-strength driver log of the release audit.",
    ),
    scoring_commit: str | None = typer.Option(
        None,
        "--scoring-commit",
        help="Commit that scored the entries (default: the bundle's pipeline_version).",
    ),
    calibration_input: str | None = typer.Option(
        None,
        "--calibration-input",
        help="The Parquet the threshold framework calibrated on.",
    ),
):
    """Assemble the deposit: Parquet, views, crosswalk, test FASTAs, rendered
    docs, class balance, known issues, snapshot hashes, MANIFEST.json and
    SHA256SUMS."""
    from trizod.dataset import finalize

    argv = [
        "--parquet",
        parquet,
        "--shifts",
        shifts,
        "--bundle",
        bundle,
        "--bmrb-dir",
        bmrb_dir,
        "--out",
        out,
    ]
    if crosswalk_root is not None:
        argv += ["--crosswalk-root", crosswalk_root]
    for s_ in sets or []:
        argv += ["--set", s_]
    if validation is not None:
        argv += ["--validation", validation]
    if allow_uncalibrated:
        argv.append("--allow-uncalibrated")
    if allow_calibration_mismatch:
        argv.append("--allow-calibration-mismatch")
    if force:
        argv.append("--force")
    for flag, value in (
        ("--previous-build", previous_build),
        ("--previous-label", previous_label),
        ("--pkl-dir", pkl_dir),
        ("--snapshot-drift", snapshot_drift),
        ("--construct-caveats", construct_caveats),
        ("--ionic-drivers", ionic_drivers),
        ("--scoring-commit", scoring_commit),
        ("--calibration-input", calibration_input),
    ):
        if value is not None:
            argv += [flag, value]
    finalize.main(argv)
