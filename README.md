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

The latest dataset is published under the DOI [10.6084/m9.figshare.25792035](https://www.doi.org/10.6084/m9.figshare.25792035).

Four nested datasets of increasing filter stringency are provided: unfiltered, tolerant, moderate, and strict. See [docs/filtering.md](docs/filtering.md) for the complete filter reference and default values.

## Project status

Under active development
