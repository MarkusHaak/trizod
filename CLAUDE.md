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
- `trizod/constants.py` — shared constants (BACKBONE_ATOMS, AA mappings, weights)

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

## Offset Correction
- `scoring.py` detects per-atom-type systematic referencing biases between observed and POTENCI-predicted shifts
- Two strategies: global offset (AIC test) and 9-residue rolling window; picks whichever yields lower Z-scores
- Functionally equivalent to re-referencing (LACS/PANAV), but uses POTENCI as the reference instead of BMRB population averages
- Currently only applied internally for scoring — does not output corrected shift files

## Conventions
- Python >=3.9, ruff for linting/formatting
- Scientific variable names allowed (T, pH, Ion, N, etc.) — see ruff ignore rules
- Always show staged files and proposed commit message, then wait for user approval before committing

## Testing
- `tests/test_potenci.py` — POTENCI prediction accuracy and edge cases
- `tests/test_smoke.py` — CLI entrypoints, single-entry pipeline integration
- `tests/test_pipeline_regression.py` — 300-entry subset regression (requires data/)
- Pipeline/regression tests require BMRB data in `data/bmrb_entries/`
- 9 tests total, ~60-80s runtime

## Scripts
- `scripts/precompute_potenci_cache.py` — precompute POTENCI predictions for faster pipeline runs
- `scripts/filter_impact_report.py` — per-filter impact analysis across all BMRB entries, outputs markdown

## Documentation
- `docs/pipeline.md` — detailed pipeline walkthrough (6 stages)
- `docs/potenci.md` — POTENCI module: origin, API, performance, internals
- `docs/filtering.md` — filter descriptions and default values per stringency level
- `docs/_planning/` — internal planning notes (gitignored)
- `docs/_planning/status-2026-03-25.md` — implementation status and roadmap

## Data
- BMRB entries: `data/bmrb_entries/` (17,388 files, not committed)
- Baselines: `data/baseline/` (not committed)
- Filter impact analysis: `data/filter_impact/` (not committed)
- Test reference: `tests/reference/unfiltered.json` (committed)
- Test subset IDs: `tests/quick_subset_ids.txt` (committed)
