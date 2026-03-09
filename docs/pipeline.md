# TriZOD Pipeline

Detailed walkthrough of the TriZOD data processing pipeline, from raw BMRB
NMR-STAR files to per-residue disorder scores.

## Overview

```
BMRB NMR-STAR files
    │
    ▼
1. Parse entries (bmrb.py)
    │
    ▼
2. Build peptide DataFrame
    │
    ▼
3. Pre-filter
    │
    ▼
4. Compute scores (POTENCI + scoring)
    │
    ▼
5. Post-filter
    │
    ▼
6. Output (CSV / JSON)
```

## Stage 1: Parse BMRB Entries

**Module**: `trizod/bmrb/bmrb.py`

Each BMRB NMR-STAR file is parsed into a `BmrbEntry` object containing:

- **Entity** — molecular identity: name, sequence, polymer type, fragment info
- **Assembly** — oligomeric state, molecular weight, thiol state
- **SampleConditions** — temperature (K), pH, ionic strength (M), pressure
- **ShiftTable(s)** — assigned chemical shifts per atom per residue

The parser extracts structured data from the NMR-STAR tag-value format, handling
missing fields, unit ambiguities (e.g. Celsius vs Kelvin), and multi-entity entries.

## Stage 2: Build Peptide DataFrame

**Module**: `trizod/trizod.py` → `fill_row_data()`

Each parsed entry becomes a row in a pandas DataFrame with columns for:

- Entry ID, sequence, sequence length
- Sample conditions (temperature, pH, ionic strength)
- Backbone chemical shifts (7 atom types: C, CA, CB, HA, H, N, HB)
- Metadata (experiment type, keywords, sample components)

Only backbone atom types defined in `BBATNS` are retained. Non-canonical amino
acids are translated to their canonical equivalents via `CAN_TRANS`.

## Stage 3: Pre-filter

**Module**: `trizod/trizod.py` → `prefilter_dataframe()`

Entries are filtered based on configurable criteria organized into four preset
levels (unfiltered, tolerant, moderate, strict). See [filtering.md](filtering.md)
for the complete filter reference.

Filter categories:
- **Physicochemical**: temperature, pH, ionic strength ranges
- **Data quality**: minimum backbone shift types, positions, coverage fraction
- **Sequence**: peptide length, non-canonical/X-residue fraction limits
- **Content**: keyword blacklists, chemical denaturant detection, experiment method
- **Unit handling**: assumptions, corrections, default condition imputation

A filter loss report tracks how many entries each filter removes.

## Stage 4: Compute Scores

**Modules**: `trizod/potenci/potenci.py`, `trizod/scoring/scoring.py`

This is the core computation, applied per entry via `compute_scores_row()`:

### 4a. POTENCI Random Coil Prediction

`potenci.get_pred_shifts()` predicts what chemical shifts would be expected for a
disordered (random coil) version of the sequence, given the sample conditions
(temperature, pH, ionic strength). This accounts for:

- Sequence-dependent neighbor effects (5-residue sliding window)
- Temperature corrections
- pH-dependent titration of ionizable residues (D, E, H, C, K, R, Y)
- Ionic strength via Debye-Hückel electrostatics

POTENCI dominates pipeline runtime (~92%), primarily due to iterative pKa fitting
with `scipy.optimize.curve_fit`.

### 4b. Weighted Secondary Chemical Shifts

For each residue and atom type, the secondary chemical shift (SCS) is:

```
SCS = observed_shift - predicted_shift
```

These are combined into a weighted sum using empirically derived weights from
`trizod/constants.py`, producing a single per-residue weighted SCS value.

### 4c. Offset Correction

`scoring.get_offset_corrected_wscs()` corrects systematic offsets between
observed and predicted shifts using two strategies:

1. **Global offset** (`compute_offsets`): per-atom-type mean offset, accepted via
   AIC test
2. **Running offset** (`compute_running_offsets`): rolling window (size 9) at the
   position of minimum standard deviation

The method yielding the lower average Z-score is selected.

### 4d. Z-score and G-score

The CheZOD **Z-score** measures how many standard deviations a residue's weighted
SCS deviates from the random coil expectation. Higher Z-scores indicate more
ordered (structured) residues; lower scores indicate disorder.

The TriZOD **G-score** normalizes Z-scores to [0, 1] range, independent of the
number of available shift types, making scores comparable across entries with
different data completeness.

## Stage 5: Post-filter

**Module**: `trizod/trizod.py`

After scoring, entries may be rejected if:

- The offset correction exceeds `--max-offset` (indicating unreliable data)
- With `--reject-shift-type-only`: only the problematic atom type is dropped
  rather than the entire entry

## Stage 6: Output

**Module**: `trizod/trizod.py` → `output_dataset()`

Results are written as JSON (primary) and optionally CSV, containing per-entry:

- Sequence, sample conditions, metadata
- Per-residue Z-scores and G-scores
- Filter status and offset corrections applied
