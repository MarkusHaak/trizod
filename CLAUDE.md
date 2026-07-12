# TriZOD — Project Instructions

## Quick Start
- Package manager: `uv`
- Install: `uv sync --group dev`
- Run CLI: `uv run trizod --help`
- Run tests: `uv run pytest tests/ -v`
- Lint: `uv run ruff check trizod/` and `uv run ruff format --check trizod/`

## Before Every Commit
1. `uv run ruff check trizod/ tests/`
2. `uv run ruff format --check trizod/ tests/`
3. `uv run pytest tests/ -v`
All three must pass.

## Architecture
- `trizod/cli/main.py` — Typer CLI entry point (`trizod = "trizod.cli.main:app"`): the bare `score` command (mirrors the historical argparse flag surface, tier-preset resolution) plus the `trizod dataset ...` subcommands.
- `trizod/trizod.py` — scoring orchestration: `run_scoring_pipeline()`, the `filter_defaults` DataFrame, `output_dataset()`, and the DataFrame-builders that use the module-global `bmrb_entries` (`fill_row_data()`, `create_peptide_dataframe()`, `compute_scores_row()`). Re-exports `trizod.pipeline`/`trizod.cache` names for backward compatibility; `python -m trizod.trizod` delegates to the CLI.
- `trizod/pipeline.py` — stateless pipeline functions: `find_bmrb_files()`, `load_bmrb_entries()`, `prefilter_dataframe()`, `postfilter_dataframe()`, `print_filter_losses()`, `compute_scores()`.
- `trizod/cache.py` — POTENCI prediction cache (`load_potenci_cache()`/`save_potenci_cache()`, content-addressed by seq+T+pH+ion).
- `trizod/io/` — I/O helpers: `fasta.py` (FASTA read/write) and `str_writer.py` (`write_rereferenced_str()`, re-referenced NMR-STAR emission).
- `trizod/bmrb/bmrb.py` — BMRB NMR-STAR file parsing (Entity, Assembly, SampleConditions, ShiftTable, BmrbEntry)
- `trizod/potenci/potenci.py` — POTENCI random coil shift predictions (public API: `get_pred_shifts()`)
- `trizod/scoring/scoring.py` — Z-score and G-score computation, offset correction (AIC-based global + 9-residue rolling window)
- `trizod/lacs/lacs.py` — LACS re-referencing (detect/correct NMR referencing errors using Wishart random coil tables)
- `trizod/dataset/` — dataset-build chain (see "Dataset Build" below)
- `trizod/figures/` — manuscript figure generators + CheZOD helpers (matplotlib is an opt-in `[figures]` extra)
- `trizod/constants.py` — shared constants (BACKBONE_ATOMS, REFINED_WEIGHTS, AA mappings)

## Pipeline Flow
(orchestrated by `run_scoring_pipeline()` in `trizod/trizod.py`)
1. Parse args in `trizod/cli/main.py` (two-phase: tier preset via `--filter-defaults`, then per-filter overrides)
2. Find & load BMRB files → `pipeline.find_bmrb_files()` / `pipeline.load_bmrb_entries()` (pickle cache in `tmp/bmrb_entries/`)
3. Prefilter entries → `pipeline.prefilter_dataframe()` (applies all filter criteria from `filter_defaults`)
4. Compute scores → `compute_scores_row()` → `pipeline.compute_scores()` per entry (POTENCI predictions cached in `tmp/potenci/`, wSCS cached in `tmp/wSCS/`)
5. Post-filter (offset rejection) → `pipeline.postfilter_dataframe()`; entries exceeding `max-offset` are excluded or masked
6. Output results (JSON + optional CSV) via `output_dataset()` + `pipeline.print_filter_losses()` report

## Caching
- `--cache-dir` defaults to `./tmp`
- `tmp/bmrb_entries/` — pickled BmrbEntry objects
- `tmp/potenci/` — POTENCI predictions as JSON (content-addressed by seq+T+pH+ion hash)
- `tmp/wSCS/` — scored weighted chemical shifts as `.npz` (keyed by entry_id + shift table IDs)
- POTENCI cache is filter-independent (depends only on sequence + conditions), so re-runs with different filter settings are mostly cache hits

## Filtering
- 4 stringency tiers: `unfiltered`, `tolerant`, `moderate`, `strict`
- `filter_defaults` DataFrame in `trizod/trizod.py` defines defaults per tier
- Individual filters overridable via CLI args
- `print_filter_losses()` (in `trizod/pipeline.py`) reports per-filter counts (filtered + uniquely filtered)
- See `docs/filtering.md` for full reference

## Re-Referencing and Offset Correction
- Two complementary systems for correcting systematic NMR referencing errors:
  1. **LACS** (`trizod/lacs/`) — standalone module, uses Wishart 1995 random coil tables as reference. Works on all residues (structured + disordered). Not yet integrated into scoring pipeline.
  2. **POTENCI-based offset correction** (`scoring.py`) — uses POTENCI predictions as reference, AIC-based global offset + 9-residue rolling window. Integrated into scoring.
- LACS is designed to run BEFORE POTENCI comparison (corrects raw observed shifts)
- The POTENCI-based correction handles residual biases AFTER LACS
- `REFINED_WEIGHTS` in `constants.py` are POTENCI RMSD on a 117-entry IDP reference set (Nielsen & Mulder, from CheZOD source code, not published)

## Conventions
- Python >=3.9, ruff for linting/formatting
- Scientific variable names allowed (T, pH, Ion, N, etc.) — see ruff ignore rules
- Always show staged files and proposed commit message, then wait for user approval before committing

## Dataset Build (`trizod dataset ...`)
The redundancy-reduced, leakage-free dataset is built by a chain of subcommands
(modules in `trizod/dataset/`, orchestrated end-to-end by `scripts/build_dataset.sh`):
- `trizod dataset build` (`build.py`) — bound-complex removal + exact-sequence dedup + quality ranking → `final_dataset/`
- `trizod dataset test-set` (`testset.py`) — recreate the seeded, CheZOD-free TriZOD test set → `testset/`
- `trizod dataset redundancy` (`redundancy.py`) — two-stage test-set leakage removal + mmseqs clustering → `mmseqs/`
- `trizod dataset representatives` (`representatives.py`) — override mmseqs cluster reps with the quality-best member
- `trizod dataset package` (`package_release.py`) — assemble the Zenodo release bundle + MANIFEST + leakage gate
- `trizod dataset deploy` (`deploy_fasta.py`) — build the UdonPred-handoff FASTA (per-residue G-score labels)
- Shared helpers: `composition.py` (bound-complex classifier), `mmseqs.py` (mmseqs wrappers), `paths.py` (path resolution), `trizod/io/fasta.py`

## Scripts
- `scripts/build_dataset.sh` — run the full `trizod dataset` chain end-to-end
- `scripts/precompute_potenci_cache.py` — precompute POTENCI predictions for faster pipeline runs
- `scripts/filter_impact_report.py` — per-filter impact analysis across all BMRB entries, outputs markdown
- `scripts/compare_lacs_bmrb.py` — validate LACS reimplementation against BMRB pre-computed LACS reports
- `scripts/fetch_panav_bmrb.py` — compute PANAV offsets locally via panav.jar (~10 min for 17k entries)
- `scripts/benchmark_rereferencing.py` — synthetic benchmark comparing LACS vs TriZOD offset recovery
- `scripts/figures/` — manuscript figure generators (`compare_gscores_lacs.py`, `analyze_max_offset.py`, `regenerate_manuscript_figures.py`)
- `scripts/validation/` — CheZOD reproduction / verification (`reproduce_chezod.py`, `verify_chezod_parsing.py`, `chezod_faithful_measure.py`, `build_chezod_test_subset.py`)

## Testing
- `tests/test_potenci.py` — POTENCI prediction accuracy and edge cases
- `tests/test_smoke.py` — CLI entrypoints, single-entry pipeline integration
- `tests/test_pipeline.py` / `tests/test_pipeline_regression.py` — pipeline integration + subset regression (require `data/`)
- `tests/test_lacs.py` / `tests/test_lacs_integration.py` — LACS module + scoring integration
- `tests/test_rereference_modes.py` — re-referencing mode behaviour
- `tests/test_cli_imports.py` — CLI import surface + backward-compat re-exports
- `tests/test_fasta.py` — `trizod/io/fasta.py` helpers
- `tests/test_figures_import.py` — figure modules import cleanly (matplotlib-optional guard)
- `tests/test_str_writer.py` — re-referenced NMR-STAR emission
- `tests/test_chezod_equality.py` — CheZOD reproduction regression
- `tests/test_full_dataset_regression.py` — full-dataset regression (requires data/)
- Pipeline/regression tests require BMRB data in `data/raw/bmrb_entries/`

## Documentation
- `docs/pipeline.md` / `docs/pipeline-overview.md` — pipeline walkthrough + overview
- `docs/potenci.md` — POTENCI module: origin, API, performance, internals
- `docs/lacs.md` — LACS module: algorithm, differences from MATLAB, API
- `docs/filtering.md` — filter descriptions and default values per stringency level
- `docs/dataset/` — dataset construction notes + datasheet (`dataset-construction.md`, `datasheet.md`)
- `docs/archive/` — superseded / historical material
- Date-stamped working snapshots: `docs/260520/`, `docs/260611/`, `docs/260623/`, `docs/260625/`

## Data
- BMRB entries: `data/raw/bmrb_entries/` (not committed)
- Per-tier scored release: `data/interim/scored/<tier>/scores.json` (not committed) — consumed by `trizod dataset build` and the CheZOD validation scripts
- CheZOD reference: `data/external/chezod/protein_nmr_1325/` (not committed)
- Baselines: `data/interim/baseline/` (not committed)
- BMRB LACS reports: `data/external/bmrb_lacs/` (not committed)
- PANAV offsets: `data/external/panav_offsets.json` (computed locally, not committed)
- Test reference: `tests/reference/` (committed); test subset IDs: `tests/quick_subset_ids.txt` (committed)
- External tools: `tools/panav.jar` (not committed, gitignored)
