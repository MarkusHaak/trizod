# POTENCI Data Files

This directory contains the chemical shift prediction data tables used by the POTENCI algorithm for predicting random coil NMR chemical shifts in intrinsically disordered proteins.

## Overview

POTENCI (Prediction Of Temperature, Neighbor and pH-corrected Chemical shifts for Intrinsically disordered proteins) is an empirical method that predicts backbone and side-chain chemical shifts based on:

- Amino acid type and local sequence context
- Temperature effects
- pH-dependent ionization states
- Electrostatic interactions

## File Descriptions

### tablecent.csv

**Central residue chemical shift values** - Base chemical shift values for each amino acid type at reference conditions.

- **Columns**: `aa`, `C`, `CA`, `CB`, `N`, `H`, `HA`, `HB`
- **Rows**: 20 standard amino acids (one-letter codes)
- **Units**: ppm (parts per million)
- **Reference conditions**: 298K, pH 7.0, ionic strength 0.1M
- **Note**: Empty cells indicate unavailable data (e.g., Glycine lacks CB/HB atoms, Proline lacks H)

### tablenei.csv

**Neighbor residue corrections** - Chemical shift perturbations caused by neighboring amino acids.

- **Columns**: `atom`, `aa`, `corr1`, `corr2`, `corr3`, `corr4`
- **Rows**: One entry per (atom type, amino acid) pair
- **Correction positions**:
  - `corr1`: Effect from residue at position i-2
  - `corr2`: Effect from residue at position i-1
  - `corr3`: Effect from residue at position i+1
  - `corr4`: Effect from residue at position i+2
- **Units**: ppm

### tabletermcorrs.csv

**Terminal corrections** - Additional corrections for N-terminal and C-terminal residues.

- **Columns**: `atom`, `term`, `corr`
- **Terms**:
  - `n`: N-terminus corrections
  - `c`: C-terminus corrections
- **Units**: ppm
- **Application**: Added to residues at protein termini to account for charge effects

### tabletempk.csv

**Temperature coefficients** - Linear temperature dependence coefficients for chemical shifts.

- **Columns**: `aa`, `CA`, `CB`, `C`, `N`, `H`, `HA`
- **Rows**: 20 standard amino acids
- **Units**: ppb/K (parts per billion per Kelvin)
- **Usage**: Δδ = coefficient × (T - 298K) / 1000
- **Reference**: 298K (25°C)

### tablecombdevs.csv

**Combinatorial deviations** - Higher-order corrections for specific sequence patterns.

- **Columns**: `atom`, `neipos`, `centgroup`, `neigroup`, `segment`, `val1`, `val2`
- **Parameters**:
  - `neipos`: Neighbor position relative to center (-2, -1, 1, 2)
  - `centgroup`: Central residue group classification
  - `neigroup`: Neighbor residue group classification
  - `segment`: 5-residue pattern identifier (e.g., "xxGPx")
  - `val1`: Correction value (ppm)
  - `val2`: Standard error estimate (ppm)
- **Groups**:
  - `G`: Glycine
  - `P`: Proline
  - `r`: Aromatic (F, Y, W)
  - `a`: Aliphatic (L, I, V, M, C, A)
  - `+`: Positive (K, R)
  - `-`: Negative (D, E)
  - `p`: Polar (N, Q, S, T, H)

### tablephshifts.csv

**pH-dependent shifts** - Chemical shift changes upon protonation/deprotonation of titratable groups.

- **Columns**: `residue`, `atom`, `val1`, `val2`, `val3`, `val4`, `val5`
- **Residues**: Titratable amino acids (D, E, H, C, Y, K, R)
- **Column interpretation**:
  - `val1`: Shift at low pH (protonated state)
  - `val2`: Shift at high pH (deprotonated state)
  - `val3`: Δδ = val2 - val1 (shift difference, main value used)
  - `val4`: Effect on previous residue (i-1)
  - `val5`: Effect on next residue (i+1)
- **Units**: ppm
- **Note**: Not all atoms/residues have neighbor effects (val4/val5 may be empty)

## Data Format

All CSV files use:

- **Delimiter**: Comma (`,`)
- **Header row**: Column names in first row
- **Missing data**: Empty cells or "na" for unavailable values
- **Encoding**: UTF-8

## Usage

These data files are automatically loaded by the `trizod.potenci.constants` module:

```python
from trizod.potenci.constants import (
    load_central_shifts,
    load_neighbor_corrections,
    load_temperature_coefficients,
    load_combinatorial_deviations,
    load_ph_shifts,
)

# Data is cached after first load
central_shifts = load_central_shifts()
```

## References

**Original POTENCI publication:**
Nielsen, J. T., & Mulder, F. A. (2018). POTENCI: prediction of temperature, neighbor and pH-corrected chemical shifts for intrinsically disordered proteins. _Journal of Biomolecular NMR_, 70(3), 141-165.
DOI: [10.1007/s10858-018-0166-5](https://doi.org/10.1007/s10858-018-0166-5)

**Original implementation:**
https://github.com/protein-nmr/POTENCI

**Data source:**
The data tables are derived from statistical analysis of NMR chemical shift databases and empirical fitting to experimental measurements.

## License

The data files maintain the licensing terms of the original POTENCI implementation.
