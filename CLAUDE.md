# TriZOD — Project Instructions

## Quick Start
- Package manager: `uv`
- Install: `uv sync --group dev`
- Run CLI: `uv run trizod --help`
- Run tests: `uv run --with pyarrow python -m pytest tests/ -v` (keep `python -m`: a bare `uv run --with pyarrow pytest` skipped the Parquet tests silently under uv 0.13.8)
- Lint: `uv run ruff check trizod/` and `uv run ruff format --check trizod/`

## Before Every Commit
1. `uv run ruff check trizod/ tests/`
2. `uv run ruff format --check trizod/ tests/`
3. `uv run --with pyarrow python -m pytest tests/ -v`
All three must pass.

## Architecture
- `trizod/cli/main.py` — Typer CLI entry point (`trizod = "trizod.cli.main:app"`): the bare `score` command (mirrors the historical argparse flag surface, tier-preset resolution) plus the `trizod dataset ...` subcommands.
- `trizod/trizod.py` — scoring orchestration: `run_scoring_pipeline()`, the `filter_defaults` DataFrame, `output_dataset()`, and the DataFrame-builders that use the module-global `bmrb_entries` (`fill_row_data()`, `create_peptide_dataframe()`, `compute_scores_row()`). Re-exports `trizod.pipeline`/`trizod.cache` names for backward compatibility; `python -m trizod.trizod` delegates to the CLI.
- `trizod/pipeline.py` — stateless pipeline functions: `find_bmrb_files()`, `load_bmrb_entries()`, `prefilter_dataframe()`, `postfilter_dataframe()`, `print_filter_losses()`, `compute_scores()`.
- `trizod/cache.py` — POTENCI prediction cache (`load_potenci_cache()`/`save_potenci_cache()`, content-addressed by seq+T+pH+ion).
- `trizod/io/` — I/O helpers: `fasta.py` (FASTA read/write) and `str_writer.py` (`write_rereferenced_str()`, re-referenced NMR-STAR emission).
- `trizod/bmrb/bmrb.py` — BMRB NMR-STAR file parsing (Entity, Assembly, SampleConditions, ShiftTable, BmrbEntry)
- `trizod/bmrb/membrane.py` — membrane-mimetic class table (regex → class, CMC, MW) and the per-sample rule behind `membrane_mimetic_sample`
- `trizod/bmrb/xrefs.py` — per-chain UniProt/PDB cross-references with provenance (`exact`/`author`/`sifts`/`unspecified`/`blast`), read straight from the NMR-STAR files by `scripts/build_parquet_dataset.py --bmrb-dir`; kept out of `bmrb.py` so it does not change `bmrb_parser_version()` (cache key)
- `trizod/bmrb/xref_sources.py` — dated external sources for the cross-references (BMRB–PDB match list, wwPDB SEQRES, SIFTS), SHA-256-verified against their `MANIFEST.json`; used via `build_parquet_dataset.py --xref-sources`
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
- `tmp/bmrb_entries/` — pickled BmrbEntry objects, stamped with `provenance.bmrb_parser_version()` (a hash of `trizod/bmrb/bmrb.py`); a pickle from other parsing code is re-parsed and overwritten on load
- `tmp/potenci/` — POTENCI predictions as JSON (content-addressed by seq+T+pH+ion hash)
- `tmp/wSCS/` — scored weighted chemical shifts as `.npz` (keyed by entry_id + shift table IDs + rereference mode + `scoring_cache_version()` + `bmrb_parser_version()`, so a scoring or parser change misses the cache)
- POTENCI cache is filter-independent (depends only on sequence + conditions), so re-runs with different filter settings are mostly cache hits

## Filtering
- 4 stringency tiers: `unfiltered`, `tolerant`, `moderate`, `strict`
- `filter_defaults` DataFrame in `trizod/trizod.py` defines defaults per tier
- Individual filters overridable via CLI args
- Condition rules (2026-11): `max-pressure-bar` (400 bar from tolerant up; unrecorded = ambient), `exclude-membrane-mimetic` (moderate + strict; class rule in `trizod/bmrb/membrane.py`, verdict in `membrane_mimetic_sample`), strict `ionic-strength-range` [0, 0.5] M, and the 273 K placeholder (`SampleConditions.get_temperature()`: exactly "273"/"273.0" K read as not deposited unless the entry text states 0 °C; flagged `temperature_placeholder`)
- The TriZOD test set is resolved against the strict pool (`trizod/dataset/testset.py`, `RESOLVE_TIER`); pinned chains that left strict are *retired* (`testset/TriZOD_test_retired.fasta`): still leakage targets in `redundancy.py`, staged under `test/` for the package gate, `split = test_retired` in the Parquet, in neither set of the deploy FASTA. Do not trim the pin when chains leave strict
- `print_filter_losses()` (in `trizod/pipeline.py`) reports per-filter counts (filtered + uniquely filtered)
- See `docs/filtering.md` for full reference

## Re-Referencing and Offset Correction
- Two complementary systems for correcting systematic NMR referencing errors:
  1. **LACS** (`trizod/lacs/`) — uses Wishart 1995 random coil tables as reference. Works on all residues (structured + disordered). Integrated into scoring: `scoring.apply_lacs_correction()` is called from `get_offset_corrected_shifts()` and runs by default (`--rereference-mode both`, also active for `lacs`; skipped only for `potenci-only`/`none`).
  2. **POTENCI-based offset correction** (`scoring.py`) — uses POTENCI predictions as reference, AIC-based global offset + 9-residue rolling window. Integrated into scoring.
- LACS is designed to run BEFORE POTENCI comparison (corrects raw observed shifts)
- The POTENCI-based correction handles residual biases AFTER LACS
- `REFINED_WEIGHTS` in `constants.py` are POTENCI RMSD on a 117-entry IDP reference set (Nielsen & Mulder, from CheZOD source code, not published)

## Conventions
- Python >=3.11, ruff for linting/formatting
- Scientific variable names allowed (T, pH, Ion, N, etc.) — see ruff ignore rules
- Always show staged files and proposed commit message, then wait for user approval before committing

## Dataset Build (`trizod dataset ...`)
The redundancy-reduced dataset (training separated from the test sets at the leakage
criterion below; never claim zero leakage) is built by a chain of subcommands
(modules in `trizod/dataset/`, orchestrated end-to-end by `scripts/build_dataset.sh`):
- `trizod dataset build` (`build.py`) — bound-complex removal + exact-sequence dedup + quality ranking → `final_dataset/`
- `trizod dataset test-set` (`testset.py`) — resolve the pinned TriZOD test set (TriZOD-test v1.0; CheZOD-disjoint at cluster level) → `testset/`
- `trizod dataset redundancy` (`redundancy.py`) — two-stage test-set leakage removal + mmseqs clustering → `mmseqs/`
- `trizod dataset representatives` (`representatives.py`) — override mmseqs cluster reps with the quality-best member
- `trizod dataset package` (`package_release.py`) — stage the package bundle (input of the Parquet) + MANIFEST + leakage gate
- `trizod dataset deploy` (`deploy_fasta.py`) — build the UdonPred-handoff FASTA (per-residue G-score labels)
- `trizod dataset export-jsonl` (`export_jsonl.py`) — the UdonPred v2 view of the release Parquet: `trizod_<tier>/{train,valid,test}.jsonl` (`id`, `x_0`, `y`; 999 = masked), cluster-seeded valid split, checked back against the Parquet
- `trizod dataset test-crosswalk` (`testset_crosswalk.py`) — every earlier TriZOD test set (Trizod348, v0.2.0, the pins, v0.3.0) against the final release, by exact sequence
- `trizod dataset finalize` (`finalize.py`) — assemble the deposit into one folder: Parquet + shifts, deploy FASTA, UdonPred view, crosswalk, test FASTAs, `class_balance.tsv`, `bmrb_files.sha256`, README/datasheet/CHANGELOG rendered from the `docs/dataset/` templates, `columns.md`, LICENSE/CITATION/zenodo.json, `MANIFEST.json` over every file + `SHA256SUMS`. Deterministic; refuses an uncalibrated Parquet
- Release identity: `release_info.py` — dataset version (0.4.0), test-set name (TriZOD-test v1.0), licences (data CC BY 4.0, code MIT), the G cuts 0.15 / 0.35 / 0.70 (G >= t; metadata only, never label columns), the k >= 10 note, snapshot/calibration readers. `CITATION.cff`, `.zenodo.json` and `pyproject.toml` must agree with it (`tests/test_release_docs.py`)
- `class_balance.py` — per-split zone shares, k < 10 share, length, pi and posterior at the cuts (F33)
- Parquet metadata + column dictionary: `scripts/build_parquet_dataset.py` (`table_metadata()`, `COLUMN_DOCS`; `--calibration <framework dir>`; `--write-column-dictionary docs/dataset/columns.md`)
- Shared helpers: `composition.py` (bound-complex classifier), `mmseqs.py` (mmseqs wrappers; the leakage criterion: >= 30 % id over >= 80 % of either chain, >= 27 bits, searched in both directions), `chezod_overlap.py` (per-chain CheZOD117/1325/1174 overlap flags of the Parquet), `paths.py` (path resolution), `trizod/io/fasta.py`
- Leakage targets: CheZOD117 (released chains + the 117 original SETH sequences in `data/external/chezod/seth_github/`), the TriZOD test set and the retired test chains. `test-set` retires (never deletes) a pinned chain as `not_strict`, `chezod117_overlap` or `within_test_duplicate`
- Membrane-mimetic rule: `trizod/bmrb/membrane.py`; reviewed title-only entries in `trizod/dataset/data/membrane_overrides.tsv`

## Scripts
- `scripts/build_dataset.sh` — run the full `trizod dataset` chain end-to-end
- `scripts/fetch_xref_sources.py` — download the dated cross-reference sources into `data/external/xref_sources/<date>/` with a SHA-256 manifest (once per release; the build never downloads)
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
- `tests/test_xrefs.py` — UniProt/PDB cross-reference parsing, normalisation and which accessions reach `uniprot_ids`
- `tests/test_xref_sources.py` — dated external sources: manifest check, sequence matching, per-chain SIFTS assignment
- `tests/test_release_metadata.py` / `tests/test_column_dictionary.py` / `tests/test_class_balance.py` / `tests/test_finalize.py` / `tests/test_release_docs.py` — release identity in the Parquet metadata, the generated column dictionary, the class-balance table, the deposit assembly (synthetic bundle in `tests/_release_fixtures.py`) and version/licence/wording consistency (run with `uv run --with pyarrow`)
- `tests/test_membrane_mimetic.py` — membrane-mimetic class rule, entity-linked components, sample-level fallback, override table
- `tests/test_leakage_coverage.py` — leakage criterion, both search directions, bit floor, release gate (every train/redundant row)
- `tests/test_testset_retirement.py` — CheZOD117-overlap and within-test-duplicate retirement
- `tests/test_chezod_overlap.py` — CheZOD overlap flags
- `tests/test_sequence_alphabet.py` — trailing non-letters stripped, internal ones mapped to X
- `tests/test_chezod_equality.py` — CheZOD reproduction regression
- `tests/test_full_dataset_regression.py` — full-dataset regression (requires data/)
- Pipeline/regression tests require BMRB data in `data/raw/bmrb_entries/`, or `TRIZOD_DATA_ROOT=<checkout holding data/>` (a worktree has no `data/`) and `TRIZOD_BMRB_DIR=<snapshot>`; `tests/test_full_dataset_regression.py` (`-m slow`) needs `TRIZOD_REGRESSION_BUILD=<build under data/interim>`

## Documentation
- `docs/pipeline.md` / `docs/pipeline-overview.md` — pipeline walkthrough + overview
- `docs/potenci.md` — POTENCI module: origin, API, performance, internals
- `docs/lacs.md` — LACS module: algorithm, differences from MATLAB, API
- `docs/filtering.md` — filter descriptions and default values per stringency level
- `docs/dataset/` — `dataset-construction.md`; the deposit templates `bundle-README.md`, `datasheet.md`, `CHANGELOG.md` (rendered by `trizod dataset finalize`; type no counts into them); `columns.md` (generated column dictionary)
- `docs/archive/` — superseded / historical material
- Historical working snapshots are archived under `docs/archive/` (dated `260415/` … `260625/`)

## Data
- BMRB entries: `data/raw/bmrb_entries/` (not committed)
- Per-tier scored release: `data/interim/scored/<tier>/scores.json` (not committed) — consumed by `trizod dataset build` and the CheZOD validation scripts
- CheZOD reference: `data/external/chezod/protein_nmr_1325/` (not committed)
- Baselines: `data/interim/baseline/` (not committed)
- BMRB LACS reports: `data/external/bmrb_lacs/` (not committed)
- PANAV offsets: `data/external/panav_offsets.json` (computed locally, not committed)
- Test reference: `tests/reference/` (committed); test subset IDs: `tests/quick_subset_ids.txt` (committed)
- External tools: `tools/panav.jar` (not committed, gitignored)
