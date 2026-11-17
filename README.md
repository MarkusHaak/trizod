# TriZOD

Novel, continuous, per-residue disorder scores from protein NMR experiments stored in the BMRB database.

## Description

TriZOD computes per-residue disorder scores from NMR chemical shift data in the Biological Magnetic Resonance Data Bank (BMRB). It extends the CheZOD scoring framework with quantitative statistical descriptors, offering nuanced analysis of intrinsically disordered regions. The CheZOD Z-score measures how much experimentally determined chemical shifts deviate from random coil predictions. The TriZOD G-score normalizes these to [0, 1], independent of the number of available shift types.

## Architecture

TriZOD consists of three core modules:

- **`trizod/bmrb/`** — Parses BMRB NMR-STAR files into structured data (entities, assemblies, sample conditions, shift tables)
- **`trizod/potenci/`** — Predicts random coil chemical shifts using the [POTENCI](https://github.com/protein-nmr/POTENCI) algorithm ([docs](docs/potenci.md))
- **`trizod/scoring/`** — Computes per-residue CheZOD Z-scores and TriZOD G-scores from experimental vs. predicted shifts

The main pipeline (`trizod/trizod.py`) orchestrates these modules. See [docs/pipeline.md](docs/pipeline.md) for a detailed walkthrough.

## Installation

## Usage

## Datasets

The current re-referenced dataset is v0.4.0, published on Zenodo under the concept DOI [10.5281/zenodo.21309963](https://doi.org/10.5281/zenodo.21309963), which always resolves to the latest version (v0.3.0: [10.5281/zenodo.21381698](https://doi.org/10.5281/zenodo.21381698); v0.2.0: [10.5281/zenodo.21309964](https://doi.org/10.5281/zenodo.21309964)). The dataset before the Zenodo releases is at [10.6084/m9.figshare.25792035](https://www.doi.org/10.6084/m9.figshare.25792035). What changed in each version, and the known issues, are in the deposit's `CHANGELOG.md` (rendered from [docs/dataset/CHANGELOG.md](docs/dataset/CHANGELOG.md)) and `known_issues.tsv`.

Four nested datasets of increasing filter stringency are provided: unfiltered, tolerant, moderate, and strict. See [docs/filtering.md](docs/filtering.md) for the complete filter reference and default values.

## Releases & re-referenced dataset

The finalized 2026-05 pipeline emits per-entry re-referenced NMR-STAR (`.str`) files when run with `--emit-str <dir>`. Each emitted file contains the backbone shifts after LACS pre-correction plus an auxiliary saveframe recording the LACS offsets, POTENCI residual offsets, the re-referencing mode, and the pipeline version. See [docs/archive/260505/str-emission.md](docs/archive/260505/str-emission.md) for the file layout.

The full re-referenced TriZOD dataset is published on Zenodo (concept DOI [10.5281/zenodo.21309963](https://doi.org/10.5281/zenodo.21309963); current version v0.4.0), distributed as a single Parquet file with its chemical-shift table and views; `trizod dataset finalize` assembles the deposit ([docs/dataset/dataset-construction.md](docs/dataset/dataset-construction.md)). The repository ships [`.zenodo.json`](.zenodo.json) and [`CITATION.cff`](CITATION.cff) describing the deposit.

Local release artefacts (uncommitted, gitignored) live under `data/interim/scored/<tier>/`.

## Tests

```bash
uv sync --group dev
uv run --with pyarrow python -m pytest tests/          # the gate (keep `python -m`)
uv run ruff check trizod/ tests/ && uv run ruff format --check trizod/ tests/
```

The data-backed tests read the data root from `TRIZOD_DATA_ROOT` (a checkout holding `data/`; a git worktree has none) and the BMRB snapshot from `TRIZOD_BMRB_DIR`; without them they skip. The full-dataset regression (`-m slow`) re-scores a seeded subset of a named build (`TRIZOD_REGRESSION_BUILD`, e.g. `rebuild-2026-11-17b`) and compares every field with that build's `scores.json`.

## Project status

Under active development
