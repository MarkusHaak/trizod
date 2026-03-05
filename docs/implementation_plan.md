# Plan: Implement BMRB Expert Suggestions

## Context

The TriZOD project processes NMR chemical shift data from the BMRB database to compute per-residue disorder scores. NMR experts (Reid Alderson, Iva Pritisanac) reviewed the filtering pipeline and provided 12 suggestions. The current `refactor/modernize-codebase` branch has modernization commits ready to merge. This plan implements all expert feedback as focused, reviewable PRs.

## PR Strategy

```
PR0: Merge refactor/modernize-codebase → main (prerequisite)
PR1: fix/blacklist-cleanup               — Fix denaturant + exp-method blacklists
PR2: feat/paramagnetic-filter             — Add paramagnetic sample filtering
PR3: fix/relax-strict-thresholds          — Relax min-backbone-shift-types (5→4)
PR4: fix/temperature-heuristic            — Improve Celsius detection range
PR5: feat/methyl-wildcard-convention      — Implement CD*/CG* wildcard labeling
PR6: feat/chemical-shift-rereferencing    — Full re-referencing module (LACS-inspired)
```

PRs 1-5 are independent and can be developed in parallel after PR0. PR6 is a larger feature.

---

## PR0: Merge Refactoring Branch

**Status**: PR created (https://github.com/MarkusHaak/trizod/pull/3) — PENDING AUDIT & MERGE

Open PR from `refactor/modernize-codebase` → `main` and merge. This is the prerequisite for all subsequent PRs.

### Audit checklist before merge:
- [ ] Ruff passes cleanly (lint + format)
- [ ] Code runs and produces same results as main branch
- [ ] POTENCI produces identical output
- [ ] pyproject.toml is correct and complete
- [ ] No regressions in test suite

---

## PR1: Fix Blacklists (`fix/blacklist-cleanup`)

**Addresses**: Suggestions #3 (keyword blacklist) and #5 (exp-method blacklist)

### 1a. Clean chemical-denaturants list in strict tier

**File**: `trizod/trizod.py` lines 67-88

**Remove 9 items** from strict `chemical-denaturants` (they are NOT denaturants):
- Reducing agents: `"BME"`, `"2-ME"`, `"mercaptoethanol"`, `"DTT"`, `"dithiothreitol"`
- Reference standard: `"dss"`
- NMR buffers: `"acetic acid"`, `"CD3COOH"`, `"deuterated sodium acetate"`

**Keep**: `"guanidin"`, `"GdmCl"`, `"Gdn-Hcl"`, `"urea"`, `"TFA"`, `"trifluoroethanol"`, `"Potassium Pyrophosphate"`

### 1b. Fix exp-method-blacklist substring matching

**File**: `trizod/trizod.py` lines 96-101

**Problem**: `["solid", "state"]` items are joined with `|` for regex matching (line 453 in `prefilter_dataframe`). `"state"` matches `"solution-state"` and `"liquid-state"`.

**Fix**: Change `["solid", "state"]` → `["solid"]` at all tiers.

### Verification
- Run `--filter-defaults strict`, compare filter loss report before/after
- Expect more entries passing denaturant filter (DTT/DSS/acetate entries no longer excluded)
- Confirm no "solution-state" entries incorrectly excluded

---

## PR2: Add Paramagnetic Filter (`feat/paramagnetic-filter`)

**Addresses**: Suggestion #2 (paramagnetic filtering) — currently **not implemented**

### 2a. Parse `_Entity.Paramagnetic` in Entity class

**File**: `trizod/bmrb/bmrb.py`, `Entity.__init__` (after line 47, `self.fragment`)

```python
self.paramagnetic = get_tag_vals(sf, "_Entity.Paramagnetic", indices=0)
```

### 2b. Parse paramagnetic chemical components + add method to BmrbEntry

**File**: `trizod/bmrb/bmrb.py`, `BmrbEntry.__init__` (after entity parsing block, ~line 437)

Parse `chem_comp` saveframes for `_Chem_comp.Paramagnetic` tag. Add `is_paramagnetic()` method that checks both entities and chem_comp for `"yes"`.

### 2c. Wire into pipeline

**File**: `trizod/trizod.py`

1. Add `"exclude-paramagnetic": [False, True, True, True]` to `filter_defaults`
2. Add `--exclude-paramagnetic` CLI argument in `parse_args()`
3. Add `row["is_paramagnetic"] = entry.is_paramagnetic()` in `fill_row_data()` (~line 670)
4. Add `~df["is_paramagnetic"]` check in `prefilter_dataframe()` when enabled
5. Pass param through `main()` call chain

### Verification
- Test with BMRB 4837 (cytochrome C) and BMRB 18991 — should be excluded
- Filter loss report shows new "paramagnetic" row

---

## PR3: Relax Strict Thresholds (`fix/relax-strict-thresholds`)

**Addresses**: Suggestion #9 (min-backbone-shift-types)

### Change min-backbone-shift-types from 5 to 4

**File**: `trizod/trizod.py` line 56

```python
"min-backbone-shift-types": [1, 2, 3, 4],  # was 5
```

Modern NMR experiments commonly yield HN, N, CO, CA (4 types) without HA. CB is technically side-chain. Requiring 5 excludes many valid datasets.

### Ionic strength — no code change needed

Current strict range `[0, 3]` is in Molar — already encompasses 150 mM physiological concentration. Add clarifying comment only.

### Verification
- Compare filter loss report with `--filter-defaults strict` before/after

---

## PR4: Improve Temperature Heuristic (`fix/temperature-heuristic`)

**Addresses**: Suggestion #6 (temperature Celsius/Kelvin correction)

### Extend Celsius detection range

**File**: `trizod/bmrb/bmrb.py` line 168

```python
if 1 <= val < 50:   # was: 15 <= val < 50
```

Values 1-14 Kelvin are physically impossible for liquid-state protein NMR. They are almost certainly Celsius (e.g., 5°C = 278 K). Liquid-state NMR runs down to ~257 K (-16°C), but 1-14 K is never valid.

### Verification
- BMRB entries with temperature 5, 10, 25, 37 → should convert to 278, 283, 298, 310 K
- No regression on entries correctly in Kelvin (298, 310)

---

## PR5: Methyl Wildcard Convention (`feat/methyl-wildcard-convention`)

**Addresses**: Suggestion #1 (Iva's stereospecific methyl labeling) and #12 (ambiguity codes)

### Overview

For Leucine (CD1/CD2) and Valine (CG1/CG2) methyl groups, when stereospecific assignment is NOT explicitly stated, use `CD*`/`CG*` wildcard labels. This helps downstream users working on automatic resonance assignment protocols.

**Key insight**: The current pipeline only processes backbone atoms (`BBATNS = ["C", "CA", "CB", "HA", "H", "N", "HB"]`). Methyl side-chain atoms are filtered out in `get_valid_bbshifts()` at line 706. The wildcard convention affects how shift data is labeled in the **output**, not the Z-score scoring.

### 5a. Add function to process side-chain methyl shifts

**File**: `trizod/bmrb/bmrb.py` (new function, after `get_valid_bbshifts`)

Create `get_methyl_shifts(shifts, seq, stereospecific=False)`:
1. Filter shifts to Leu CD1/CD2 and Val CG1/CG2 atom types
2. Check ambiguity code for each assignment:
   - Code `1` = unique/stereospecific → keep CD1/CD2 or CG1/CG2 labels
   - Code `2` or higher = ambiguous → relabel as CD*/CG*
3. If `stereospecific=False` (default): always use wildcard labels unless the entry is known to have stereospecific assignments
4. Return a DataFrame with columns: `pos`, `aa3`, `atm_id` (CD1/CD2/CD*/CG1/CG2/CG*), `val`

### 5b. Check for stereospecific annotation

**File**: `trizod/bmrb/bmrb.py`

In `BmrbEntry.__init__`, scan entry title/details/citation for keywords like "stereospecific" to set a `self.has_stereospecific_methyls` flag. Example: BMRB 18414 explicitly states stereospecific assignments.

### 5c. Include methyl shifts in output

**File**: `trizod/trizod.py`

In `fill_row_data()`, call `get_methyl_shifts()` and store the results in a new column `row["methyl_shifts"]`. In `output_dataset()`, include methyl shift data in the JSON output.

### 5d. Add CLI option

**File**: `trizod/trizod.py`

Add `--include-methyl-shifts` flag (default False) to control whether methyl data is included in output.

### Verification
- Process BMRB 18414 (known stereospecific) → should preserve CD1/CD2 labels
- Process a random entry without stereospecific annotation → should use CD*/CG*
- Verify output JSON contains methyl shift data when `--include-methyl-shifts` is used

---

## PR6: Chemical Shift Re-Referencing (`feat/chemical-shift-rereferencing`)

**Addresses**: Suggestion #10 (chemical shift re-referencing, endorsed by both Reid and Iva)

### Overview

The project already has a basic offset correction in `scoring.py` (the CheZOD approach from Mulder 2016):
- `compute_offsets()` (line 96): Global per-atom-type mean offset with AIC test
- `compute_running_offsets()` (line 47): Rolling window (size 9) offset at position of min stddev
- `get_offset_corrected_wscs()` (line 192): Selects whichever offset method yields lower avg Z-score

What's missing: a proper **re-referencing** step that can detect and correct larger systematic errors (>2-3 ppm) before scoring, using the established LACS approach. ~25% of BMRB entries have referencing errors.

### Approach: LACS-Inspired Re-Referencing Using POTENCI

Implement a structure-independent re-referencing module that combines:
1. **LACS-style CA-CB analysis** for 13C offset detection (the CA-CB difference is referencing-independent)
2. **POTENCI-based per-atom analysis** for all atom types (H, N, HA, CA, CB, C)
3. Grid search over candidate offsets to find optimal correction

### 6a. Create new module `trizod/referencing/`

**New files**:
- `trizod/referencing/__init__.py` — exports public API
- `trizod/referencing/referencing.py` — main re-referencing logic

### 6b. Core algorithm: `estimate_reference_offsets()`

```python
def estimate_reference_offsets(
    observed_shifts: np.ndarray,    # (n_residues, 7) array
    shifts_mask: np.ndarray,        # boolean mask
    predicted_shifts: np.ndarray,   # POTENCI predictions (n_residues, 7)
    seq: str,
    method: str = "lacs",           # "lacs", "global", "combined"
) -> dict[str, float]:
```

**LACS-style 13C detection** (for CA, CB, C):
1. Compute secondary shifts: `Δα = obs_CA - pred_CA`, `Δβ = obs_CB - pred_CB`
2. The difference `Δα - Δβ` is **referencing-independent** (both CA and CB have the same 13C offset)
3. If `mean(Δα)` and `mean(Δβ)` are both shifted by the same amount, this indicates a 13C referencing error
4. The 13C offset = `mean(Δα + Δβ) / 2` (average of CA and CB offsets)
5. For CO: correlate secondary CO shifts against CA-CB; if CO shifts show a systematic offset consistent with a 13C error, include it

**N offset detection** (for 15N):
1. Use the known correlation between secondary N shifts and secondary CA/CB shifts of the preceding residue
2. Compute: `N_offset = mean(obs_N - pred_N)` after excluding outliers
3. Validate using AIC criterion

**1H offset detection** (for H, HA):
1. `H_offset = mean(obs_H - pred_H)` for amide protons
2. `HA_offset = mean(obs_HA - pred_HA)` for alpha protons
3. These are typically smaller offsets but can still be significant

**Grid search refinement**:
1. Initial estimate from per-atom-type means
2. Refine by testing candidate offsets in [-5, +5] ppm at 0.1 ppm increments
3. Score each candidate using sum of squared deviations from POTENCI predictions
4. Select offset that minimizes total deviation

### 6c. Validation: `validate_offsets()`

```python
def validate_offsets(
    offsets: dict[str, float],
    observed_shifts: np.ndarray,
    predicted_shifts: np.ndarray,
    shifts_mask: np.ndarray,
    min_observations: int = 10,
    min_aic_improvement: float = 6.0,
) -> dict[str, float]:
```

- Reject offsets for atom types with fewer than `min_observations` data points
- Apply AIC test: accept offset only if `N * ln(σ_before / σ_after) > min_aic_improvement`
- Return validated offsets (unaccepted offsets set to 0.0)

### 6d. Integration with pipeline

**File**: `trizod/trizod.py`

Add re-referencing as an optional pre-processing step before the scoring pipeline:

1. Add CLI arguments:
   - `--rereferencing` / `--no-rereferencing` (default: True for tolerant/moderate/strict)
   - `--rereferencing-method` (choices: `"lacs"`, `"global"`, `"combined"`)
   - `--max-reref-offset` (max acceptable re-referencing correction, default 5.0 ppm)

2. In `compute_scores_row()` (~line 836): after POTENCI predictions, before offset correction:
   ```python
   if rereferencing:
       reref_offsets = estimate_reference_offsets(...)
       validated = validate_offsets(...)
       # Apply re-referencing corrections to observed shifts
       observed_shifts -= validated_offsets
   # Then proceed with existing offset correction pipeline
   ```

3. Store re-referencing offsets in output (per-atom-type corrections applied)

### 6e. Add to filter_defaults

```python
"rereferencing": [False, True, True, True],
"rereferencing-method": ["global", "lacs", "lacs", "lacs"],
```

### Relationship to existing offset correction

The re-referencing step runs **before** the existing offset correction:
1. **Re-referencing** (new): Corrects systematic spectrometer/referencing errors (typically 0.5-3 ppm for 13C, 1-2 ppm for 15N)
2. **Offset correction** (existing `scoring.py`): Fine-tunes remaining per-atom offsets using AIC-based approach (typically <1 ppm)

The existing `compute_offsets()` and `compute_running_offsets()` in `scoring.py` remain unchanged and operate on the re-referenced data.

### Verification
- Run on known mis-referenced entries from RefDB (if available)
- Compare re-referencing offsets against published PANAV corrections for well-characterized entries
- Verify that correctly-referenced entries get near-zero corrections
- Compare Z-score distributions before/after re-referencing
- Run full pipeline and verify overall dataset quality improves

---

## Items Noted But Not Requiring Code Changes

| Suggestion | Status |
|-----------|--------|
| Disulfide bond states (#4) | Useful metadata but not critical for backbone scoring. Can be added later as a metadata column by parsing `_Assembly.Thiol_state` |
| Max-offset clarification (#7) | Already well-documented in code — it's the max per-atom-type offset before rejection (in ppm of weighted SCS) |
| Deuterium isotope effect (#11) | Non-trivial correction depending on torsion angles. Requires separate research effort |

---

## End-to-End Verification

After all PRs merged, run full pipeline on BMRB data:
```bash
trizod --input-dir data/bmrb_entries --filter-defaults strict --output strict_new.json
trizod --input-dir data/bmrb_entries --filter-defaults tolerant --output tolerant_new.json
```

Compare against `data/2024-05-09/strict.json` and `data/2024-05-09/tolerant.json` to quantify impact.

---

## Key Files Reference

| File | Changes |
|------|---------|
| `trizod/trizod.py` | PRs 1,2,3,5,6 — filter_defaults, CLI args, fill_row_data, prefilter, compute_scores |
| `trizod/bmrb/bmrb.py` | PRs 2,4,5 — Entity parsing, temperature heuristic, methyl shifts |
| `trizod/scoring/scoring.py` | Unchanged (existing offset correction continues to work) |
| `trizod/referencing/` (new) | PR6 — new re-referencing module |
| `trizod/constants.py` | Potentially PR5 for methyl atom type definitions |
