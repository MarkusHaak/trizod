# LACS — Linear Analysis of Chemical Shifts

Python reimplementation of [LACS](https://github.com/bmrb-io/LACS) (Wang et al. 2005, 2009), a method to detect and quantify NMR chemical shift referencing errors. Located in `trizod/lacs/`.

## The problem

Every NMR chemical shift is measured relative to a reference standard (typically DSS or TMS). If the experimenter's reference is wrong, **every shift of that nucleus type is offset by the same constant amount**. For example, if the 13C reference is off by +2 ppm, then every CA, CB, and C' shift in the entry is 2 ppm too high. Roughly 20% of 13C and 35% of 15N assignments in the BMRB have referencing errors.

## The core insight

The **difference** between two shifts of the same nucleus type is immune to referencing errors because the constant offset cancels:

```
(CA_obs - CB_obs) = (CA_true + k) - (CB_true + k) = CA_true - CB_true
```

LACS exploits this by using `(CA - CB)_secondary` as a **reference-independent X-axis**. The secondary shift of any individual atom (Y-axis) retains the offset. A non-zero Y-intercept of the X-Y regression reveals the referencing error.

## Algorithm: 13C and HA atoms (CA, CB, C', HA)

### Step 1: Pre-proline correction

Proline's cyclic side chain perturbs the chemical shifts of the **preceding** residue. Before computing secondary shifts, LACS adds amino-acid-specific corrections to residues immediately before a proline. These corrections (`_PRE_PRO` table, 20 amino acids x 6 atom types) remove this known systematic effect so it doesn't contaminate the offset estimate.

### Step 2: Compute secondary shifts

For each residue `i` with amino acid type `aa`:

- **X-axis** (reference-independent):
  ```
  X_i = (CA_obs,i - CB_obs,i) - (CA_rc,aa - CB_rc,aa)
  ```
  This is the secondary shift of the CA-CB *difference*. Any 13C referencing error cancels because it affects both CA and CB equally.

- **Y-axis** (contains the offset):
  ```
  Y_i = target_obs,i - target_rc,aa
  ```
  where `target` is whichever atom we're analysing (CA, CB, HA, or C'). A constant referencing error `k` shifts all Y values up by `k`.

The random coil reference values (`_RC_SHIFTS`) come from Wishart et al. (1995), the same table used in the original MATLAB implementation.

**Residue filters:** Cys, Gly, and Pro are excluded from the 13C analysis. Cys has variable shifts due to disulfide bonds, Gly lacks CB, and Pro has unusual backbone geometry. At least 20 valid residues are required.

### Step 3: Outlier removal (Mahalanobis distance)

Before regression, statistical outliers are removed. The Mahalanobis distance of each (X, Y) point from the centroid is computed using the sample covariance matrix. Points whose distance exceeds 80% of the maximum distance are flagged as outliers, capped at 2% of the dataset.

This removes residues with unusual chemical shifts (e.g., near paramagnetic centres, ligand-binding sites, or assignment errors) that would distort the regression.

### Step 4: Two-line robust regression

The secondary shift data has a characteristic "V-shape" because helical and strand residues pull in opposite directions:
- **Helix residues** have large positive CA-CB secondary shifts (X >> 0)
- **Strand residues** have large negative CA-CB secondary shifts (X << 0)
- **Coil residues** cluster near X = 0

LACS fits **two overlapping regression lines** using an atom-specific threshold:

| Line | Points used | Captures |
|------|------------|----------|
| Line 1 (strand side) | X < +threshold | Coil + strand residues |
| Line 2 (helix side) | X > -threshold | Coil + helix residues |

The threshold differs by atom type (matching the MATLAB `ths` array):

| Atom type | Threshold | Rationale |
|-----------|-----------|-----------|
| CA, CB | 1 ppm | Strong correlation with X; narrow overlap suffices |
| HA, C' (CO) | 6 ppm | Weaker X correlation; wider overlap needed to include enough helix/strand points and avoid intercept bias |

Each line is fitted with **robust regression** (iteratively reweighted least squares with Tukey bisquare weights) to resist remaining outliers. This is equivalent to MATLAB's `robustfit`.

Each line needs at least 10 points. If either side has too few, a single robust regression on all data is used as fallback.

### Step 5: Extract offset

The referencing offset is the **mean of the two Y-intercepts**:

```
offset = mean(intercept_strand, intercept_helix)
```

Positive offset means the observed shifts are systematically too high. Subtract the offset to correct.

The two-intercept averaging reduces sensitivity to the slope, which varies with protein composition. It also provides an implicit quality check: if the two intercepts disagree substantially, the data may be problematic.

## Algorithm: 15N and HN atoms

The 15N and HN analysis uses a **different linear relationship**: the secondary shift of nitrogen at residue `i` correlates with the CA-CB secondary shift of the **preceding residue** `i-1` (not the same residue). This inter-residue correlation arises from backbone conformational coupling.

### Differences from the 13C method

**X-axis uses the preceding residue:**
```
X_i = (CA_obs,i-1 - CB_obs,i-1) - (CA_rc,aa(i-1) - CB_rc,aa(i-1))
```

**Preceding-residue correction (Ncorr):** The identity of residue `i-1` systematically influences the N/HN shift of residue `i`. The `_NCORR` table provides empirical corrections (20 amino acid types x 2 atom types) that are subtracted from Y before fitting.

**Additional filters:**
- Residue `i` must not be Pro (no amide H/N)
- Residue `i-1` must not be Gly or His (their CA-CB differences are poor predictors)
- Residues must be sequential (no chain gaps)
- |X| must be ≤ 10 ppm

**Two-phase outlier removal:**
1. **Phase A — Mahalanobis distance** with a stricter threshold (0.99 vs 0.80 for 13C)
2. **Phase B — Weight-based trimming:** After robust fitting, any point with bisquare weight < 0.2 is iteratively removed and the fit is recomputed

**Constrained slope:** The expected slopes are well-characterised:
- N vs (CA-CB)_{i-1}: slope ~ -0.4 (constrained to [-0.45, -0.35])
- HN vs (CA-CB)_{i-1}: slope ~ -0.07 (constrained to [-0.08, -0.06])

If the slope deviates from the expected range **and** the dataset is small (< 66 points) or unbalanced (< 15% of points on either the helix or strand side), the slope is constrained and the intercept is re-estimated.

**Systematic correction:** An empirical offset is added after the regression:
- N: +0.465 ppm
- HN: +0.049 ppm

These constants account for residual systematic bias in the Ncorr correction table.

## Robust regression implementation

Our `_robustfit` reimplements MATLAB's `robustfit` using iteratively reweighted least squares (IRLS) with the bisquare (Tukey biweight) weight function:

1. **Initialise** with ordinary least squares (OLS)
2. **Compute residuals** `r = y - X @ beta`
3. **Robust scale** `s = median(|r|) / 0.6745` (MAD estimator)
4. **Leverage** `h_i` from the hat matrix of the weighted design matrix
5. **Standardised residuals** `u_i = r_i / (4.685 * s * sqrt(1 - h_i))`
6. **Bisquare weights** `w_i = (1 - u_i^2)^2` if `|u_i| < 1`, else 0
7. **Re-fit** weighted least squares and repeat until convergence

The tuning constant 4.685 gives 95% asymptotic efficiency at the normal distribution while providing strong resistance to outliers.

## Data tables

| Table | Source | Purpose |
|-------|--------|---------|
| `_RC_SHIFTS` | Wishart et al. 1995 | Random coil shifts for 20 amino acids, 6 atom types |
| `_PRE_PRO` | LACS `ord.m` / `ordN.m` | Pre-proline corrections (20 AA x 6 atoms) |
| `_NCORR` | LACS `ordN.m` | Preceding-residue corrections for N/HN (20 AA x 2 atoms) |

## Public API

```python
from trizod.lacs import compute_lacs_offsets

offsets = compute_lacs_offsets(
    seq="MQIFVKTLTG...",           # protein sequence (1-letter codes)
    seq_nums=np.arange(1, n+1),    # residue numbers (1-based)
    obs_shifts={                    # observed shifts (NaN for missing)
        "CA": ca_array,
        "CB": cb_array,
        "C":  co_array,
        "HA": ha_array,
        "H":  h_array,
        "N":  n_array,
    },
)
# Returns: {"CA": 1.58, "CB": 1.58, "C": 1.46, "HA": -0.05, "H": 0.02, "N": -0.31}
#          None for atom types with insufficient data
```

**Sign convention:** Positive offset = observed shifts are too high by that amount. Subtract the offset from observed shifts to correct them. This matches TriZOD's existing offset correction convention in `scoring.py`.

**Atom coverage:** CA, CB, C' (CO), HA, H (amide), N, HB. HB has no direct LACS relationship but is a 1H nucleus like HA, so its offset is set equal to the HA offset.

**Requirements:**
- Both CA and CB shifts must be present (the X-axis needs them)
- At least 20 valid residues after filtering
- At least 10 points per regression line for the two-line fit

## Limitations

- **Requires secondary shift spread.** LACS exploits the variation between helical, strand, and coil residues. Intrinsically disordered proteins with near-zero secondary shifts provide minimal X-axis spread, making the regression poorly determined. This is less of a concern for TriZOD since most BMRB entries are structured proteins.

- **CA and CB offsets are coupled.** Both use the same X-axis and both are 13C nuclei, so they always produce (near-)identical offsets. This is correct for referencing errors (which affect all 13C equally) but means LACS cannot detect atom-type-specific errors for CA vs CB.

- **HB is inferred, not measured.** LACS has no direct relationship for HB, so its offset is propagated from HA (both are 1H nuclei sharing the same reference). This is correct for referencing errors but would not catch HB-specific systematic biases.

- **Assumes constant offset.** LACS assumes the referencing error is a single constant per nucleus type. Position-dependent errors (rare) are not detected.

## Differences from the MATLAB original

| Aspect | MATLAB (bmrb-io/LACS) | Python (trizod/lacs) |
|--------|----------------------|---------------------|
| Language | MATLAB | Python / NumPy |
| Robust regression | `robustfit` (built-in) | Custom bisquare IRLS |
| Constrained fit | `lsqcurvefit` | Grid search over slope range |
| Dependencies | MATLAB license | numpy, scipy only |
| Sign convention | offset = -intercept (add to correct) | offset = +intercept (subtract to correct) |
| Output | Text + NMR-STAR files | Python dict |

## References

- Wang L, Eghbalnia H, Bahrami A, Markley JL (2005) "Linear analysis of carbon-13 chemical shift differences and its application to the detection and correction of errors in referencing and spin system identifications." *J Biomol NMR* 32:13-22. [doi:10.1007/s10858-005-1717-0](https://doi.org/10.1007/s10858-005-1717-0)
- Wang L, Markley JL (2009) "Empirical correlation between protein backbone 15N and 13C secondary chemical shifts and its application to nitrogen chemical shift re-referencing." *J Biomol NMR* 44:95-99. [doi:10.1007/s10858-009-9324-0](https://doi.org/10.1007/s10858-009-9324-0)
- Wishart DS, Bigam CG, Holm A, Hodges RS, Sykes BD (1995) "1H, 13C and 15N random coil NMR chemical shifts of the common amino acids." *J Biomol NMR* 5:67-81.
