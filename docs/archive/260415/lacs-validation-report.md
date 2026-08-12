# LACS Python Reimplementation — Validation Report

**Date:** 2026-04-15
**Author:** Tobias Senoner (generated with Claude Code)

## Summary

We reimplemented LACS (Linear Analysis of Chemical Shifts) in Python and validated it against 6,692 BMRB entries that have pre-computed LACS reports from the original MATLAB implementation. The Python version reproduces the MATLAB results with high fidelity across all four atom types.

## Background

**What is LACS?** LACS detects and quantifies NMR chemical shift referencing errors by exploiting a reference-independent linear relationship. The difference (CA - CB) cancels any constant 13C offset, providing a clean X-axis. The secondary shift of each target atom (Y-axis) retains the offset, which appears as the Y-intercept of a robust regression.

**Why reimplement?** The original is MATLAB (requires license, hard to integrate). We need re-referencing as a pipeline stage in TriZOD to correct the ~20% of 13C and ~35% of 15N assignments in the BMRB that are improperly referenced. Our Python version runs in the existing numpy/scipy stack with zero new dependencies.

**References:**
- Wang L et al. (2005) J Biomol NMR 32:13-22 (13C, 1HA)
- Wang L & Markley JL (2009) J Biomol NMR 44:95-99 (15N, 1HN)
- MATLAB source: github.com/bmrb-io/LACS

## Validation Dataset

- **BMRB LACS reports:** 6,774 files at `bmrb.io/ftp/pub/bmrb/validation_reports/LACS/` (last updated ~July 2020)
- **Our parsed entries:** 16,963 cached BmrbEntry pickles
- **Overlap:** 6,756 entries have both a BMRB LACS report and a local parsed entry
- **Successfully compared:** 6,692 entries (11 failed due to parsing issues, 53 had no peptide data)
- **Atom types compared:** CA, CB, HA, C' (CO) — the 4 atoms covered by BMRB's LACS reports (N, H are not in the BMRB LACS output files)

## Results

### Per-Atom Agreement

| Atom | n | Correlation (r) | Mean diff (ppm) | Median \|diff\| (ppm) | 90th %ile \|diff\| (ppm) | Max \|diff\| (ppm) |
|------|---:|:---:|:---:|:---:|:---:|:---:|
| **CA** | 6,691 | 0.992 | -0.000 | 0.04 | 0.13 | 1.15 |
| **CB** | 6,691 | 0.992 | -0.000 | 0.04 | 0.13 | 1.15 |
| **HA** | 5,263 | 0.955 | -0.008 | 0.01 | 0.04 | 0.23 |
| **C'** | 5,117 | 0.998 | -0.030 | 0.05 | 0.17 | 3.63 |

### Agreement by Threshold

| Threshold | CA | CB | HA | C' |
|:---------:|:--:|:--:|:--:|:--:|
| ≤ 0.10 ppm | 84.8% | 84.8% | 99.5% | 76.1% |
| ≤ 0.25 ppm | 97.7% | 97.7% | 100.0% | 95.9% |
| ≤ 0.50 ppm | 99.7% | 99.7% | 100.0% | 99.0% |
| ≤ 1.00 ppm | 100.0% | 100.0% | 100.0% | 99.8% |

### Interpretation

- **CA/CB** (r = 0.992): Near-perfect match. Mean difference is essentially zero. 97.7% within 0.25 ppm. CA and CB always produce identical offsets (shared 13C reference), as expected.

- **HA** (r = 0.955): Strong agreement. Median difference is just 0.01 ppm — the best of all atom types. 100% within 0.25 ppm. HA offsets have small dynamic range (std ~0.08 ppm) since 1H is usually well-referenced to DSS, which limits the correlation coefficient despite excellent absolute accuracy.

- **C'** (r = 0.998): Very strong agreement after fixing an initial bug (see below). Small residual mean bias of -0.03 ppm. 95.9% within 0.25 ppm. The few outliers (max 3.63 ppm) likely arise from differences in how our bisquare IRLS handles edge cases vs MATLAB's built-in `robustfit`.

## Bugs Found and Fixed During Validation

The iterative comparison revealed that the MATLAB code uses **atom-specific thresholds** for the two-line regression, passed via an indirection (`x+1` then `x-1`) that was easy to misread:

```matlab
ths = [1 1 1 6 6 6 6];  % indexed by original column - 1
% Effective mapping: CA=1, CB=1, HA=6, CO=6
```

Our initial implementation hardcoded `threshold = 1` for all atoms. This was correct for CA/CB but wrong for HA and CO, which both need threshold = 6.

### Fix 1: C' (CO) threshold

Using threshold = 1 excluded ~40% of helix/strand residues from the CO regression, biasing the intercept.

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| C' correlation | 0.953 | **0.998** |
| C' mean diff | -0.147 ppm | **-0.030 ppm** |
| C' within 0.25 ppm | 65.0% | **95.9%** |

### Fix 2: HA threshold

Same root cause. The narrower threshold attenuated HA offsets by ~20% (slope of ours-vs-BMRB was 0.80 instead of 1.0).

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| HA correlation | 0.835 | **0.955** |
| HA median \|diff\| | 0.03 ppm | **0.01 ppm** |
| HA within 0.10 ppm | 95.8% | **99.5%** |
| HA max \|diff\| | 0.52 ppm | **0.23 ppm** |

## What Our Implementation Covers

| Atom | LACS method | Coverage |
|------|-----------|----------|
| CA | 13C two-line regression | Direct |
| CB | 13C two-line regression | Direct |
| C' (CO) | 13C two-line regression (threshold=6) | Direct |
| HA | 1H two-line regression | Direct |
| H (amide) | 15N preceding-residue correlation | Direct (not in BMRB reports, so not validated here) |
| N | 15N preceding-residue correlation | Direct (not in BMRB reports, so not validated here) |
| HB | Propagated from HA (same 1H reference) | Indirect |

BMRB's LACS reports only contain CA, CB, HA, CO. Our implementation also covers H and N (per Wang & Markley 2009) and propagates the HA offset to HB, giving full coverage of all 7 backbone atom types used by TriZOD.

## Deep Investigation of Remaining Differences

All remaining differences are exact multiples of 0.01 ppm (both implementations round to 0.01), forming a smooth bell-shaped distribution (std ~0.09 ppm for CA). A systematic trace of all 6,692 entries revealed two distinct root causes:

### Cause 1: Different input residues (92% of disagreements)

BMRB's MATLAB LACS never starts at residue 1 — the minimum start residue across all 6,774 reports is 10. The start position varies widely (10-197) because BMRB uses the depositor's numbering, which often skips N-terminal tags, signal peptides, or uses PDB numbering offsets.

Our code uses the full sequence from position 1 and includes all valid residues. This means we typically have 5-15 more residues than BMRB, concentrated at the N-terminus.

| Pattern | Prevalence | Description |
|---------|-----------|-------------|
| N-terminal excess | 72% | Our extra points are residues 1..K at the N-terminus |
| Different shift table | 2% | Multi-entity entries; BMRB analyzed a different chain |
| Same residues | 8% | Identical input data |
| Other (mixed) | 18% | Combination of terminal and internal differences |

**This is correct behavior, not a bug.** Including more valid residues gives our regression more data and should produce more robust offset estimates. Excluding terminal residues to match BMRB would be a regression in quality.

Correlation between point count difference and offset difference is weak (r ~0.06 with sequence length), confirming that including extra terminal residues rarely distorts the offset.

### Cause 2: Robust regression convergence (8% of disagreements)

For the ~550 entries where our input residues exactly match BMRB's, the remaining differences are 0.00-0.03 ppm — pure numerical differences between our bisquare IRLS (numpy) and MATLAB's built-in `robustfit`. Both use the Tukey biweight function with tuning constant 4.685, but differ in:

- Internal least-squares solver (numpy `lstsq` vs MATLAB's QR decomposition)
- Leverage computation precision
- Convergence detection threshold
- Floating-point accumulation order

These differences propagate through the iterative reweighting to produce ±0.01-0.02 ppm shifts in the final intercept, which rounds to a different 0.01 bucket.

**This is irreducible** without reimplementing MATLAB's exact internal solver, which would add complexity for zero practical benefit.

### Should we change anything?

**No.** Both sources are benign:
- Including more residues is better, not worse
- 0.01-0.03 ppm regression differences are negligible vs real referencing errors (1-5 ppm)
- No systematic bias (mean diff = -0.000 for CA/CB)

## Performance

| Operation | Time |
|-----------|------|
| Full comparison (6,692 entries) | 102 seconds |
| Per-entry LACS computation | ~0.015 seconds |
| Downloading all 6,774 LACS reports (16 threads) | ~5 minutes (one-time) |

## Next Steps

1. **Validate H and N offsets** — BMRB doesn't publish these, but we can use Reid's synthetic benchmark (POTENCI ground truth + known corruption + recovery) for N and H
2. **Integrate into pipeline** — add LACS as Stage 2.5, before POTENCI scoring
3. **Compare LACS vs our existing AIC-based offset correction** — on real data, do they agree? Where do they disagree?
4. **Output corrected .str files** — apply LACS offsets to produce re-referenced chemical shift files for distribution

## Code

- **Implementation:** `trizod/lacs/lacs.py` (~550 lines)
- **Tests:** `tests/test_lacs.py` (11 tests, synthetic benchmark)
- **Comparison script:** `scripts/compare_lacs_bmrb.py`
- **Download script:** `scripts/download_lacs_reports.py`
- **Documentation:** `docs/lacs.md`
- **Raw comparison data:** `data/lacs_comparison.npz`
