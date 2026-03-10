# POTENCI — Random Coil Chemical Shift Prediction

POTENCI (**P**rediction **O**f **T**otal **E**ffects on **N**MR **C**hemical shifts
from **I**ntrinsically disordered proteins) predicts the **random coil chemical shifts**
of backbone atoms in a protein sequence — i.e., the shifts you'd expect if the protein
had no structure, just a flexible disordered chain.

In the TriZOD pipeline, these predicted shifts are compared against experimentally
measured shifts. Large deviations indicate structure; small deviations indicate disorder.

## Origin

Based on the POTENCI method published in:

> Nielsen JT, Mulder FAA. *POTENCI: prediction of temperature, neighbor and pH-corrected
> chemical shifts for intrinsically disordered proteins.* J Biomol NMR. 2018;70:141-165.
> [doi:10.1007/s10858-018-0175-y](https://doi.org/10.1007/s10858-018-0175-y)

Adapted from [protein-nmr/POTENCI](https://github.com/protein-nmr/POTENCI)
(commit `17dd2e6`, file `pytenci1_3.py`), originally by Frans Mulder
(fmulder@chem.au.dk). Adapted by Markus Haak (markus.haak@tum.de) and
Tobias Senoner (tobias.senoner@tum.de).

## Differences from upstream

Our implementation is functionally equivalent (identical data tables, identical
algorithm) but improved for integration into the TriZOD pipeline:

| Aspect              | Upstream (`potenci.py3`)               | Ours                                                    |
| ------------------- | -------------------------------------- | ------------------------------------------------------- |
| Data tables         | Hardcoded multi-line strings           | CSV files in `data/`                                    |
| pKa inner loop      | Nested Python loops                    | NumPy vectorized over pH values                         |
| `eval()` usage      | Throughout (`eval(lin[4])`)            | Replaced with `float()`                                 |
| Data caching        | None (re-parses strings each call)     | Module-level constants                                  |
| Logging             | `VERB` flag + `print()`                | Python `logging` module                                 |
| Numerical stability | `exp(x²) / (e*r)` can overflow         | Log-space: `exp(x² - log(e*r))`                         |
| Naming              | camelCase / smooshed (`getpredshifts`) | snake_case (`get_pred_shifts`)                          |
| Constants           | Single-letter names (`e`, `a`, `b`)    | Descriptive (`DIELECTRIC_WATER`, `MIN_CHARGE_DISTANCE`) |

## Predicted atoms

7 backbone atom types per residue:

| Atom | Description | Notes |
|------|-------------|-------|
| `C`  | Carbonyl carbon (C') | |
| `CA` | Alpha carbon | |
| `CB` | Beta carbon | Absent for Glycine |
| `N`  | Amide nitrogen | |
| `H`  | Amide proton | Absent for Proline |
| `HA` | Alpha proton | |
| `HB` | Beta proton | |

Terminal residues (first and last) are excluded from predictions because they
lack the full 5-residue context window.

## The prediction model (4 additive terms)

For each interior residue, the predicted shift is:

```
shift = center_shift + neighbor_corrections + temperature_correction + pH_correction
```

### 1. Center shifts (`centshifts.csv`, 20 amino acids)

Base random coil chemical shift for each amino acid x atom type. These are empirically
determined reference values (e.g., Alanine CA = 52.53 ppm).

### 2. Neighbor corrections (`neicorrs.csv` + `termcorrs.csv`)

The identity of the 4 neighboring residues (i-2, i-1, i+1, i+2) perturbs the shift.
Computed in `pred_pent_shift()` using a **5-residue sliding window (pentamer)**:

```
pentamer = [i-2, i-1, center, i+1, i+2]
```

For terminal residues, out-of-bounds positions are replaced by virtual `"n"` and `"c"`
residues with their own correction coefficients (`termcorrs.csv`).

On top of simple neighbor corrections, there are **combination corrections**
(`combdevs.csv`, 247 entries) that capture non-additive effects. Amino acids are
grouped into 6 classes (G, P, aromatic, aliphatic, positive, negative) and specific
center+neighbor group pairs get extra correction terms.

### 3. Temperature correction (`tempcoeffs.csv`)

Linear scaling from a reference temperature of 298K:

```
correction = coeff / 1000 * (T - 298)
```

Coefficients are per amino acid x atom type. Not applied to HB.

### 4. pH correction (the expensive part)

Only applied when `use_ph_corr=True` (i.e., pH != 7.0). This accounts for the
protonation state of titratable residues (D, E, H, C, K, R, Y, and N/C-termini)
at non-neutral pH. This is where ~92% of compute time goes.

#### Step 1: Predict pKa values (`calc_pkas_from_seq`)

The algorithm iteratively predicts effective pKa values for all titratable sites:

1. **Identify titratable residues** in the sequence (D, E, H, C, K, R, Y, n-term,
   c-term) with their reference pKa values from `constants.py`.

2. **Compute pairwise electrostatic interactions** using a Debye-Hückel model:
   - Distance between sites estimated from sequence separation:
     `d = 5.0 + sqrt(|i-j|) * 7.5` Angstrom
   - Ionic strength dampens interactions (higher salt = weaker coupling)
   - `_debye_huckel_W()` computes the screened Coulomb energy using the complementary
     error function (`erfc`)

3. **Iteratively fit pKa values** (5 cycles):
   - For each titratable site, enumerate all 2^W protonation microstates in a sliding
     window of W residues (W = min(5, n_sites))
   - Compute the Boltzmann weight of each microstate from a free energy matrix G that
     combines: intrinsic pKa, pH, pairwise interactions, and the current titration
     state of sites outside the window
   - Sum Boltzmann weights to get the titration fraction at each pH value (54 pH points
     from 2.0 to 10.0)
   - Fit the resulting titration curve to a Henderson-Hasselbalch equation using
     `scipy.curve_fit` to extract effective pKa and Hill coefficient (nH)
   - Use the new titration fractions as input for the next cycle

**Why the sliding window?** With N titratable sites, the full partition function has 2^N
microstates — exponential and infeasible for large proteins. The sliding window (size 5)
approximates this by enumerating microstates locally, treating distant sites as fixed at
their current titration state. The iterative cycles let information propagate across the
sequence.

#### Step 2: Compute shift corrections (`_get_ph_corrs`)

For each titratable residue with predicted pKa:

```
frac     = titration_fraction(pH, predicted_pKa, nH)    # at actual pH
frac_ref = titration_fraction(7.0, reference_pKa, nH)   # at reference pH
delta    = (frac - frac_ref) * ph_shift_delta            # from phshifts.csv
```

This correction is applied to the titratable residue itself AND its immediate neighbors
(preceding/succeeding), since protonation state changes propagate to nearby residues.

## Output format

`get_pred_shifts("AAGDTFKISELVK", 298.0, 7.0, 0.1)` returns:

```python
{
    (2, 'A'): {'C': 177.5, 'CA': 52.7, 'CB': 19.2, 'H': 8.21, 'HA': 4.26, 'N': 123.5, 'HB': 1.32},
    (3, 'G'): {'C': 173.7, 'CA': 45.3, 'H': 8.28, 'HA': 3.95, 'N': 109.7},  # no CB/HB
    (4, 'D'): { ... },
    ...
    (12, 'V'): { ... },
    # (1, 'A') and (13, 'K') excluded — terminal residues
}
```

- **Keys**: `(residue_number, amino_acid)` — 1-based, terminals excluded
- **Values**: dict of atom type to predicted shift in **ppm**
- Glycine has no CB/HB; Proline has no H

In the TriZOD pipeline, these predicted shifts are subtracted from experimental shifts
to get **secondary chemical shifts (SCS)**, which are then converted to Z-scores
indicating disorder.

## Public API

```python
get_pred_shifts(seq, temperature, pH, ion, use_ph_corr=True)
```

**Parameters:**

- `seq` — protein sequence (single-letter amino acids)
- `temperature` — in Kelvin (e.g. 298.0)
- `pH` — sample pH (e.g. 7.0)
- `ion` — ionic strength in M (e.g. 0.1)
- `use_ph_corr` — apply pH-dependent corrections (set `False` for pH=7.0)

## Caching

### Module-level caching (at import time)

All CSV data tables are parsed once when the module is first imported and stored as
module-level constants:

| Constant | Source | Content |
|----------|--------|---------|
| `CENTER_SHIFTS` | `centshifts.csv` | Base shift per (amino acid, atom) |
| `NEIGHBOR_CORRS` | `neicorrs.csv` + `termcorrs.csv` | Neighbor effect coefficients |
| `TEMP_CORRS` | `tempcoeffs.csv` | Temperature coefficients |
| `COMB_CORRS` | `combdevs.csv` | Combination correction terms |
| `PH_SHIFTS` | `phshifts.csv` | pH shift deltas for titratable residues |
| `_AA_GROUP` | computed | Amino acid to group mapping (6 groups) |
| `_OUTER_MATRICES` / `_ALL_TUPLES` | computed | Pre-computed binary enumeration matrices for pKa fitting |

### Pipeline-level caching (disk)

Since POTENCI output depends only on `(seq, temperature, pH, ionic_strength)`, the
pipeline caches results to disk via `--cache-dir`:

- **Key**: SHA-256 hash of `"{seq}|{T}|{pH}|{ion}"` (truncated to 16 chars)
- **Storage**: `{cache_dir}/potenci/{hash}.json`
- **Precompute**: `scripts/precompute_potenci_cache.py` can batch-compute the cache
  for the entire dataset upfront, with an index file (`_index.tsv`) to skip
  already-processed BMRB entries on re-runs

## Performance profile

POTENCI dominates the TriZOD pipeline (~92% of total time), almost entirely
in `calc_pkas_from_seq()` which iteratively fits pKa values for titratable
residues using `scipy.optimize.curve_fit`.

On the 300-entry test subset:

- **With pH correction** (78% of entries): ~200ms/entry average
- **Without pH correction** (pH=7.0): ~1ms/entry

Sequence length and number of titratable residues (D, E, H, C, K, R, Y) are
the main cost drivers.

## Module structure

```
trizod/potenci/
├── __init__.py          # Re-exports get_pred_shifts and constants
├── constants.py         # Physical constants, reference pKa values
├── potenci.py           # Core prediction code
├── data/
│   ├── centshifts.csv   # Center (random coil) chemical shifts per amino acid
│   ├── neicorrs.csv     # Neighbor correction coefficients
│   ├── termcorrs.csv    # N/C-terminal corrections
│   ├── tempcoeffs.csv   # Temperature correction coefficients
│   ├── combdevs.csv     # Combination deviation corrections
│   └── phshifts.csv     # pH-dependent shift changes
└── README.md
```

## Key internal functions

| Function                 | Purpose                                                        |
| ------------------------ | -------------------------------------------------------------- |
| `calc_pkas_from_seq()`   | Iterative pKa prediction (the bottleneck)                      |
| `_get_ph_corrs()`        | pH-dependent shift corrections using predicted pKas            |
| `pred_pent_shift()`      | Predict shift for a 5-residue window                           |
| `_build_pentamer()`      | Build 5-residue context window with terminal handling          |
| `_get_temp_corr()`       | Temperature correction                                         |
| `_titration_fraction()`  | Henderson-Hasselbalch titration fraction (used by `curve_fit`) |
| `_debye_huckel_W()`      | Electrostatic interaction energy (Debye-Huckel)                |
| `_w_to_logp()`           | Convert interaction energy to log-probability                  |
| `_small_matrix_limits()` | Sliding window bounds for pKa fitting                          |
| `_small_matrix_pos()`    | Position within sliding window                                 |

## Testing

Dedicated tests in `tests/test_potenci.py`:

- **Reference value matching** — predictions compared against known values (commit `8905a85`)
- **pH correction** — verifies pH-sensitive residues shift at non-neutral pH
- **Terminal exclusion** — first/last residues are excluded from predictions
- **Glycine constraints** — no CB/HB predictions for glycine
