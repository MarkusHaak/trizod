# POTENCI — Random Coil Chemical Shift Prediction

This module predicts random coil NMR chemical shifts for proteins using the
POTENCI algorithm. It is used by the TriZOD pipeline to compute expected shifts,
which are then compared to experimental shifts to derive disorder Z-scores.

## Origin

Adapted from [protein-nmr/POTENCI](https://github.com/protein-nmr/POTENCI)
(commit `17dd2e6`, file `pytenci1_3.py`), originally by Frans Mulder
(fmulder@chem.au.dk).

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

## Public API

```python
get_pred_shifts(seq, temperature, pH, ion, use_ph_corr=True, pka_csv_path=None, identifier="")
```

Returns `dict[(residue_num, aa_letter)] -> dict[atom_type -> shift_value]`.

**Parameters:**

- `seq` — protein sequence (single-letter amino acids)
- `temperature` — in Kelvin (e.g. 298.0)
- `pH` — sample pH (e.g. 7.0)
- `ion` — ionic strength in M (e.g. 0.1)
- `use_ph_corr` — apply pH-dependent corrections (set `False` for pH=7.0)
- `pka_csv_path` — path to pre-computed pKa CSV, or `False` to compute on the fly

## Performance profile

POTENCI dominates the TriZOD pipeline (~92% of total time), almost entirely
in `calc_pkas_from_seq()` which iteratively fits pKa values for titratable
residues using `scipy.optimize.curve_fit`.

On the 300-entry test subset:

- **With pH correction** (78% of entries): ~200ms/entry average
- **Without pH correction** (pH=7.0): ~1ms/entry

Sequence length and number of titratable residues (D, E, H, C, K, R, Y) are
the main cost drivers. For full-dataset runs, the pipeline's `--cache-dir`
option caches the downstream weighted SCS results, but POTENCI predictions
themselves could also be cached since they depend only on (seq, T, pH, ion).

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

## Testing

Dedicated tests in `tests/test_potenci.py`:

- **Reference value matching** — predictions compared against known values (commit `8905a85`)
- **pH correction** — verifies pH-sensitive residues shift at non-neutral pH
- **Terminal exclusion** — first/last residues are excluded from predictions
- **Glycine constraints** — no CB/HB predictions for glycine

## Key internal functions

| Function                 | Purpose                                                        |
| ------------------------ | -------------------------------------------------------------- |
| `calc_pkas_from_seq()`   | Iterative pKa prediction (the bottleneck)                      |
| `get_ph_corrs()`         | pH-dependent shift corrections using predicted pKas            |
| `pred_pent_shift()`      | Predict shift for a 5-residue window                           |
| `_get_temp_corr()`       | Temperature correction                                         |
| `_titration_fraction()`  | Henderson-Hasselbalch titration fraction (used by `curve_fit`) |
| `_debye_huckel_W()`      | Electrostatic interaction energy (Debye-Hückel)                |
| `_w_to_logp()`           | Convert interaction energy to log-probability                  |
| `_small_matrix_limits()` | Sliding window bounds for pKa fitting                          |
| `_small_matrix_pos()`    | Position within sliding window                                 |
