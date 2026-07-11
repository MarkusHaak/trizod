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

The current re-referenced dataset (v0.2.0) is published on Zenodo under the concept DOI [10.5281/zenodo.21309963](https://doi.org/10.5281/zenodo.21309963), which always resolves to the latest version. The previous dataset is at [10.6084/m9.figshare.25792035](https://www.doi.org/10.6084/m9.figshare.25792035).

Four nested datasets of increasing filter stringency are provided: unfiltered, tolerant, moderate, and strict. See [docs/filtering.md](docs/filtering.md) for the complete filter reference and default values.

## Releases & re-referenced dataset

The finalized 2026-05 pipeline emits per-entry re-referenced NMR-STAR (`.str`) files when run with `--emit-str <dir>`. Each emitted file contains the backbone shifts after LACS pre-correction plus an auxiliary saveframe recording the LACS offsets, POTENCI residual offsets, the re-referencing mode, and the pipeline version. See [docs/archive/260505/str-emission.md](docs/archive/260505/str-emission.md) for the file layout.

The full re-referenced TriZOD dataset is published on Zenodo (concept DOI [10.5281/zenodo.21309963](https://doi.org/10.5281/zenodo.21309963); latest version v0.2.0), distributed as a single Parquet file. The repository ships [`.zenodo.json`](.zenodo.json) and [`CITATION.cff`](CITATION.cff) describing the deposit.

Local release artefacts (uncommitted, gitignored) live under `data/release/<tier>/`.

## Project status

Under active development
