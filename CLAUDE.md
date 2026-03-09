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
- `trizod/trizod.py` — pipeline orchestration, CLI, filtering
- `trizod/bmrb/bmrb.py` — BMRB NMR-STAR file parsing
- `trizod/potenci/potenci.py` — POTENCI random coil shift predictions
- `trizod/scoring/scoring.py` — Z-score computation, offset correction
- `trizod/constants.py` — shared constants (BBATNS, AA mappings, weights)

## Conventions
- Python >=3.9, ruff for linting/formatting
- Scientific variable names allowed (T, pH, Ion, N, etc.) — see ruff ignore rules
- Always show staged files and proposed commit message, then wait for user approval before committing

## Testing
- `tests/test_potenci.py` — POTENCI prediction accuracy and edge cases
- `tests/test_smoke.py` — CLI entrypoints, single-entry pipeline integration
- `tests/test_pipeline_regression.py` — 300-entry subset regression (requires data/)
- Pipeline/regression tests require BMRB data in `data/bmrb_entries/`

## Data
- BMRB entries: `data/bmrb_entries/` (17k files, not committed)
- Baselines: `data/baseline/` (not committed)
- Test reference: `tests/reference/unfiltered.json` (committed)
- Test subset IDs: `tests/quick_subset_ids.txt` (committed)
