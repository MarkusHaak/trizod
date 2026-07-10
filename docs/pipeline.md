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
- **Assembly** — composition (number of components, organic ligands, metal
  ions), paramagnetic flag, per-entity physical state (e.g. native, denatured)
- **SampleConditions** — temperature (K), pH, ionic strength (M), pressure
- **ShiftTable(s)** — assigned chemical shifts per atom per residue

The parser extracts structured data from the NMR-STAR tag-value format, handling
missing fields, unit ambiguities (e.g. Celsius vs Kelvin), and multi-entity entries.

## Stage 2: Build Peptide DataFrame

**Module**: `trizod/trizod.py` → `create_peptide_dataframe()` + `fill_row_data()`

Each BMRB entry can contain multiple shift tables, entity assemblies, and
entities (chains). The DataFrame gets one row per unique
`(shift_table, entity_assembly, entity)` combination — most entries produce a
single row, but multi-chain complexes or entries with multiple shift tables
produce several.

Columns populated at this stage:

- **IDs**: `entryID`, `stID`, `entity_assemID`, `entityID`
- **Conditions**: `temperature` (K), `pH`, `ionic_strength` (M)
- **Sequence**: `seq` (one-letter code), used later for POTENCI predictions
- **Shift statistics**: `total_bbshifts`, `bbshift_types` (how many of the 7
  atom types are present), `bbshift_positions` (residues with at least one shift)
- **Metadata**: `entity_name`, `exp_method`, `exp_method_subtype`,
  `citation_title`, `citation_DOI`
- **Flags**: `paramagnetic`, keyword booleans, denaturant booleans
- **Placeholders** (filled in Stage 4): `scores`, `k`, `off_C`..`off_HB`,
  `total_bbshifts_post`, `bbshift_types_post`, `bbshift_positions_post`

The backbone shifts array (`seq_len x 7` float matrix + boolean mask) is not
stored in the DataFrame — it is computed on the fly during Stage 4 scoring.

Only backbone atom types defined in `BACKBONE_ATOMS` are retained. Pro-chiral
methylene/methyl protons — glycine `HA2/HA3`, non-Ala `HB2/HB3`, and Ala
`HB1/HB2/HB3` — are always collapsed into `HA`/`HB` via averaging (POTENCI only predicts
the mean value, so stereospecific vs non-stereospecific makes no difference). Non-canonical residues are dropped from the shift table entirely (they remain as `X` placeholders in the sequence and
contribute no shifts).

## Stage 3: Pre-filter

**Module**: `trizod/trizod.py` → `prefilter_dataframe()`

Entries are filtered based on configurable criteria organized into four preset
levels (`unfiltered`, `tolerant`, `moderate`, `strict`). Each filter can be
overridden individually via CLI. See [filtering.md](filtering.md) for the
complete filter reference with default values per tier.

Before named filters run, rows missing any required value (`exp_method`,
`temperature`, `ionic_strength`, `pH`, `seq`, or `total_bbshifts`) are
silently rejected.

Filter groups:

- **Experiment method**: two-layer whitelist/blacklist on `exp_method_subtype`.
  The blacklist (`"solid"` from tolerant upward) excludes solid-state NMR. The
  whitelist (`"solution"`, `"structures"`) controls what's allowed — entries
  with missing subtype (25% of BMRB) pass in unfiltered/tolerant/moderate but
  are rejected in strict.
- **Physicochemical ranges**: temperature, pH, ionic strength must fall within
  the tier's bounds (e.g. strict: T ∈ [273, 313] K, pH ∈ [6, 8]).
- **Data quality**: minimum backbone shift types (of the 7 atom types), minimum
  positions with shifts, minimum fraction of sequence covered by shifts.
- **Sequence**: minimum peptide length, maximum fraction of non-canonical
  residues (`CANONICAL_AA_MASK` counts canonical AAs in the sequence), maximum
  fraction of `X` residues.
- **Content blacklists**: keyword blacklist (searched across title, entity name,
  assembly name/details, citation keywords, sample names), chemical denaturant
  detection (searched in sample component names), paramagnetic flag.

Note: the `unit-assumptions`, `unit-corrections`, and `default-conditions`
settings in `filter_defaults` are **not filters** — they control how sample
condition values are parsed in Stage 2 (whether to assume SI units for missing
unit annotations, fix outlier temperatures, impute defaults for missing pH /
temperature / ionic strength). They affect which values the physicochemical
filters see, but are not filter criteria themselves.

`print_filter_losses()` reports per-filter counts: how many entries each filter
removed, and how many were *uniquely* removed (would have passed if only that
filter were disabled).

## Stage 4: Compute Scores

**Modules**: `trizod/potenci/potenci.py`, `trizod/scoring/scoring.py`

This is the core computation, applied per entry via `compute_scores_row()`:

### Re-referencing (`--rereference-mode`)

NMR chemical shifts can carry systematic **referencing errors** (a constant
per-atom offset from a mis-set spectral reference). TriZOD corrects these before
scoring; the strategy is chosen with `--rereference-mode`:

- **`none`** — raw shifts, no correction.
- **`lacs`** — LACS pre-correction only (`trizod/lacs/lacs.py`). LACS detects
  offsets by regressing observed secondary shifts against the **Wishart (1995)
  random-coil** reference, independent of POTENCI, and works on structured and
  disordered residues alike. It is applied to the observed backbone shifts
  **before** the POTENCI comparison below.
- **`potenci-only`** — the POTENCI/AIC offset correction of §4c only (this is the
  CheZOD-equivalent method; see [lacs.md](lacs.md) and the CheZOD reproduction).
- **`both`** (default) — LACS pre-correction, then the POTENCI/AIC offset
  correction of §4c on the residual. LACS handles the bulk referencing error;
  the POTENCI/AIC step mops up residual per-atom biases POTENCI still sees.

See [lacs.md](lacs.md) for the LACS algorithm and its differences from the
original MATLAB implementation.

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

### 4b. Secondary Chemical Shifts and Weighting

For each residue and atom type, the secondary chemical shift (SCS) is:

```
SCS = observed_shift - predicted_shift
```

The result is a `(seq_len, 7)` array of differences — one column per backbone
atom type. Each SCS is then divided by its atom-type-specific weight from
`REFINED_WEIGHTS` in `trizod/constants.py`. The weights act as expected standard
deviations, normalising each atom type so they contribute proportionally
(e.g. N shifts vary over ~20 ppm while H shifts vary over ~3 ppm — without
normalisation, N would dominate). The output is still a 2D array, not a single
value per residue.

### 4c. POTENCI/AIC Offset Correction

`scoring.get_offset_corrected_shifts()` detects and corrects systematic biases
between observed and POTENCI-predicted shifts. This addresses referencing
errors or consistent prediction biases for individual atom types. Under the
default `--rereference-mode both`, it operates on the LACS-precorrected shifts
and so captures only the residual offset; under `potenci-only` it is the sole
correction.

The procedure:

1. **Initial Z-scores** are computed from the raw weighted SCS (no offset).
2. **Outlier detection** (`get_outlier_mask()`): residues with Z-scores > 6.0 are
   flagged and excluded from offset estimation, preventing extreme values from
   biasing the correction.
3. **Two offset strategies** are computed independently:
   - **Global offset** (`compute_offsets`): per-atom-type mean SCS across all
     non-outlier residues, accepted only if the AIC improvement exceeds the
     threshold (delta AIC > 6.0) and at least 4 data points exist for that
     atom type.
   - **Running offset** (`compute_running_offsets`): 9-residue rolling window,
     selecting the window position with the lowest mean standard deviation
     across atom types. Also subject to the AIC test.
4. **Strategy selection**: the running offset is adopted only if it yields a
   lower mean Z-score (= better agreement with random coil) than the global
   offset. Otherwise the global offset is used.
5. The final weighted SCS are recomputed with the selected offsets applied.

### 4d. Z-score and G-score

The CheZOD **Z-score** uses a chi-squared CDF approximation (Wilson-Hilferty) to
measure how much a residue's weighted SCS deviates from random coil. For each
residue, the residual sum of squares (RSS) of the weighted, offset-corrected
SCS is computed and mapped through the chi-squared distribution. The degrees
of freedom equal the number of comparable atom types at that residue.

Scores are computed over a **3-residue sliding window** (`convert_to_triplet_data()`):
the weighted SCS from residues i-1, i, and i+1 are concatenated, and degrees
of freedom are summed across the triplet. This smooths scores and incorporates
neighbour context. Higher Z-scores indicate more ordered (structured) residues;
lower scores indicate disorder. Terminal residues receive `NaN`.

The TriZOD **G-score** is a separate scoring function, not a normalisation of
the Z-score. It computes the geometric mean of per-atom-type Gaussian
observation probabilities from the same weighted SCS. The result falls in
[0, 1] and is independent of the number of available shift types, making
scores comparable across entries with different data completeness.

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
