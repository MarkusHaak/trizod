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
- `trizod/trizod.py` — pipeline orchestration, CLI, filtering (`filter_defaults` DataFrame, `prefilter_dataframe()`, `print_filter_losses()`, `compute_scores_row()`, `main()`)
- `trizod/bmrb/bmrb.py` — BMRB NMR-STAR file parsing (Entity, Assembly, SampleConditions, ShiftTable, BmrbEntry)
- `trizod/potenci/potenci.py` — POTENCI random coil shift predictions (public API: `get_pred_shifts()`)
- `trizod/scoring/scoring.py` — Z-score and G-score computation, offset correction (AIC-based global + 9-residue rolling window)
- `trizod/lacs/lacs.py` — LACS re-referencing (detect/correct NMR referencing errors using Wishart random coil tables)
- `trizod/constants.py` — shared constants (BACKBONE_ATOMS, REFINED_WEIGHTS, AA mappings)

## Pipeline Flow
1. Parse args (two-phase: preset first, then detailed args)
2. Find & load BMRB files → `load_bmrb_entries()` (with pickle cache in `tmp/bmrb_entries/`)
3. Prefilter entries → `prefilter_dataframe()` (applies all filter criteria from `filter_defaults`)
4. Compute scores → `compute_scores_row()` per entry (POTENCI predictions cached in `tmp/potenci/`, wSCS cached in `tmp/wSCS/`)
5. Post-filter (offset rejection) → entries exceeding `max-offset` are excluded or masked
6. Output results (JSON + optional CSV) + `print_filter_losses()` report

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
- `print_filter_losses()` reports per-filter counts (filtered + uniquely filtered)
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

## Scripts
- `scripts/precompute_potenci_cache.py` — precompute POTENCI predictions for faster pipeline runs
- `scripts/filter_impact_report.py` — per-filter impact analysis across all BMRB entries, outputs markdown
- `scripts/compare_lacs_bmrb.py` — validate LACS reimplementation against BMRB pre-computed LACS reports (6,774 entries)
- `scripts/fetch_panav_bmrb.py` — compute PANAV offsets locally via panav.jar (~10 min for 17k entries)
- `scripts/benchmark_rereferencing.py` — synthetic benchmark comparing LACS vs TriZOD offset recovery

## Testing
- `tests/test_potenci.py` — POTENCI prediction accuracy and edge cases
- `tests/test_smoke.py` — CLI entrypoints, single-entry pipeline integration
- `tests/test_pipeline_regression.py` — 300-entry subset regression (requires data/)
- `tests/test_lacs.py` — LACS module: 11 tests including synthetic benchmark
- Pipeline/regression tests require BMRB data in `data/bmrb_entries/`

## Documentation
- `docs/pipeline.md` — detailed pipeline walkthrough (6 stages)
- `docs/potenci.md` — POTENCI module: origin, API, performance, internals
- `docs/lacs.md` — LACS module: algorithm, differences from MATLAB, API
- `docs/filtering.md` — filter descriptions and default values per stringency level
- `docs/_planning/` — internal planning notes (gitignored)
- `docs/_planning/status-2026-03-25.md` — implementation status and roadmap

## Data
- BMRB entries: `data/bmrb_entries/` (17,388 files, not committed)
- Baselines: `data/baseline/` (not committed)
- Filter impact analysis: `data/filter_impact/` (not committed)
- BMRB LACS reports: `data/bmrb_lacs/` (6,772 files, not committed)
- PANAV offsets: `data/panav_offsets.json` (computed locally, not committed)
- Test reference: `tests/reference/unfiltered.json` (committed)
- Test subset IDs: `tests/quick_subset_ids.txt` (committed)
- External tools: `tools/panav.jar` (103KB, not committed, gitignored)
