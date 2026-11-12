# Comprehensive Summary of Suggestions from Email Thread

## 1. **Stereospecific Chemical Shift Assignment Convention (Iva)**

**Suggestion**: Implement a "wildcard" convention for methyl resonance labeling:

- Label **Leucine side-chain methyls as CD\*** (instead of CD1/CD2)
- Label **Valine methyls as CG\*** (instead of CG1/CG2)
- Apply this convention in **all cases where stereospecific assignment is not explicitly stated**

**Rationale**:

- Very few datasets have true stereospecific assignments
- BMRB entry 18414 is given as an example where stereospecific assignments ARE explicitly stated
- This convention would help people working on **automatic resonance assignment protocols** and is already standard practice in that field

---

## 2. **Paramagnetic vs. Diamagnetic Filtering (Reid)**

**Suggestion**: Exclude all paramagnetic proteins and ligands from the dataset

**Details**:

- Paramagnetic species cause very large chemical shift outliers
- Example: cytochrome C (BMRB 4837)
- Paramagnets are usually on ligands, so may already be filtered by existing ligand blacklist

**Implementation**:

- Use BMRB advanced search syntax:
  - For proteins: `_Entity.Paramagnetic ~* 'yes'`
  - For ligands: `_Chem_comp.Paramagnetic ~* 'yes'`
- Searching for paramagnetic proteins alone will also return proteins bound to paramagnetic ligands
- Additional example: BMRB 18991

---

## 3. **Keyword Filtering Blacklist Refinement (Reid)**

**Suggestion**: Remove or reconsider several keywords from the blacklist that unnecessarily exclude high-quality datasets

**Keywords to REMOVE from blacklist**:

### a) **Reducing Agents** (should NOT be excluded):

- DTT (dithiothreitol)
- BME (2-mercaptoethanol)
- 2-ME
- TCEP (another reducing agent)

**Rationale**: These are commonly added to keep cysteine residues reduced (-SH) and prevent time-dependent oxidation and disulfide bond formation. Most Cys-containing proteins studied by NMR include these agents. They don't denature proteins at typical concentrations.

### b) **Chemical Shift Reference Standard** (should NOT be excluded):

- DSS (2,2-dimethyl-2-silapentane-5-sulfonate)

**Rationale**: DSS is a chemical shift referencing agent added at very low concentrations. Its ¹H NMR signal serves as an internal reference to correct the ¹H chemical shift scale. It's inert and doesn't bind to proteins or affect their spectra.

### c) **Acetate Buffers** (should NOT be excluded):

- Acetic acid
- Deuterated sodium acetate
- Acetate

**Rationale**: Acetate is a common NMR buffer for pH 4-6 range. At typical NMR buffer concentrations (25-50 mM), it's only 0.14% of pure acetic acid concentration (17.4 M) and serves as a buffer without denaturing proteins.

**Action Items**:

- Check how many datasets are currently excluded by the [DTT, BME, 2-ME, etc.] filters
- Verify how many of these involve native proteins that should be retained

---

## 4. **Disulfide Bond Considerations (Reid)**

**Suggestion**: Account for reduced vs. oxidized cysteine states

**Details**:

- Cysteine has very different chemical shifts (especially Cβ) when:
  - Reduced: -SH
  - Oxidized: -S-S- (disulfide bond)
- BMRB has searchable field: Molecular Assembly / Assembly / Thiol state (fully reduced; fully oxidized)
- Cβ chemical shift can help discriminate between states even if not properly annotated
- Note: Some proteins require disulfide bonds for stability and will unfold in presence of reducing agents

---

## 5. **Experimental Method Filtering Clarification (Reid)**

**Suggestion**: Clarify the 'state' keyword in exp-method-blacklist

**Current filter**: `['solid', 'state']`

**Question**: Does this mean:

- solid **AND** state, or
- solid **OR** state?

**Concern**: Liquid-NMR practitioners often use terms like 'solution-state' or 'liquid-state', so filtering by 'state' alone could incorrectly exclude liquid-state NMR data.

**Note**: Need to verify how BMRB distinguishes liquid from solid-state NMR (may use 'solid' alone).

---

## 6. **Temperature Data Correction (Reid)**

**Suggestion**: Correct for temperature reporting in Celsius vs. Kelvin

**Issue**: Current temperature plot shows artifacts from people reporting in Celsius instead of Kelvin

**Context**:

- Most liquid-state protein NMR: **278-313 K** (5-40°C)
- Extreme low-end: ~257 K (-16°C) using special small-tube techniques (very uncommon)
- Values of 5-40 in the data are **most likely Celsius**, not Kelvin

**Action**: Implement conversion or standardization to ensure all temperatures are in consistent units.

---

## 7. **Max-Offset Clarification Needed (Reid)**

**Question**: What does "max-offset" refer to?

**Observation**: Values appear to be 2 or 3

**Speculation**: Could this be standard deviations from the mean?

**Action**: Clarify the meaning and calculation of this parameter.

---

## 8. **Ionic Strength Threshold Relaxation (Reid)**

**Suggestion**: Relax ionic strength threshold to physiological concentration

**Current threshold**: Unknown (needs checking)

**Recommended threshold**: Up to **~150 mM** (physiological concentration)

**Rationale**:

- Distribution of ionic strength in BMRB is likely centered around 75 mM
- Few datasets are likely lost by including up to 150 mM
- Worth checking the actual distribution to confirm

---

## 9. **Min-Backbone-Shift Types Threshold (Reid)**

**Suggestion**: Relax minimum from 5 to 4 backbone shift types for strict dataset

**Current requirement**: 5 backbone shift types

**Proposed definition of backbone shifts**: N, HN, CO, CA, HA

**Rationale**:

- Requirement of 5 is **very strict**
- CB is technically classified as side chain, not backbone
- Modern NMR assignment experiments commonly provide:
  - **HN, N, CO, CA** (4 types), or
  - **HN, N, CO, CA, CB** (4 backbone + 1 side chain)
- HA, while useful, is increasingly omitted from routine assignments

**Recommendation**: Set minimum to 4 to include any combination of [N, HN, CO, CA, HA]

---

## 10. **Chemical Shift Re-Referencing (Reid, endorsed by Iva)**

**Suggestion**: Implement systematic chemical shift re-referencing to correct for systematic offsets

**Problem**:

- Various sources of error create systematic offsets in chemical shifts
- Offsets can reach **1 ppm or higher** (sometimes >2.5 ppm for ¹³C)
- Caused by incorrect spectrometer settings or improper referencing techniques
- Impacts secondary chemical shift and structural propensity analyses

**Prevalence**: As of 2003, ~25% of BMRB entries needed re-referencing

**Available Tools**:

### a) **RefDB** (Wishart Lab)

- Website: https://refdb.wishartlab.com/
- Originally published 2003 with ~500 proteins
- Reference: https://pubmed.ncbi.nlm.nih.gov/12652131/
- Uses internal correlations between shifts for re-referencing

### b) **PANAV** (Wishart Group)

- Java software (old but functional)
- Reference: https://link.springer.com/article/10.1007/s10858-010-9407-y

### c) **Z-score approach** (Frans Mulder group)

- In-house method
- Reference: https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2016.00004/full

**Iva's suggestion**: Implement this functionality directly in the bmrb.py script to "obliterate" the old Java software and do everything cleanly in Python.

---

## 11. **Deuterium Isotope Effect Correction (Reid)**

**Suggestion**: Account for systematic offsets in ²H-labeled samples

**Issue**:

- Samples isotope-enriched with ²H (e.g., ²H,¹³C,¹⁵N-labeled protein) show systematic chemical shift offsets
- Atoms directly bonded to ²H are most affected
- Effect scales **non-linearly** with number of bonds to ²H atom
- Depends on torsion angles, making full correction non-trivial

**Current BMRB handling**: Likely only annotates if sample is ²H-labeled, without correcting for the isotope effect

**Reference**: https://pmc.ncbi.nlm.nih.gov/articles/PMC3457063/

---

## 12. **Ambiguity Code Considerations (Reid)**

**Suggestion**: Include ambiguity code as a filtering flag for stereospecific assignments

**Context**:

- Related to stereospecific chemical shift assignments
- Example: Leucine CD1 vs. CD2 methyls
- Some researchers assign arbitrarily without knowing which is which
- Others can confidently assign the exact atom

**BMRB encoding**: Ambiguity code indicates confidence level of assignment

**Scope**: Applies to all stereocenters:

- Leucine (CD1, CD2)
- Valine (CG1, CG2)
- Glycine (HA2, HA3)
- Other stereocenters

**Recommendation**: Use as a useful flag for subsequent filtering steps

---

## 13. **Prioritization and Implementation Strategy (Iva)**

**Suggestions**:

1. **Prioritize Reid's list**: Distinguish between:

   - Essential "fixes" that must be addressed
   - "Would be nice to also consider" items

2. **Implementation approach**: Either:

   - Have a meeting to discuss priorities, or
   - Have Reid fork the repository and directly flag/note suggestions in the bmrb.py script

3. **Collaborative approach**: Reid could help with chemical shift re-referencing implementation (if Tobias and Michael agree)

---

## Summary of Priority Actions

### **High Priority**:

1. Remove inappropriate keywords from blacklist (DTT, BME, DSS, acetate)
2. Implement chemical shift re-referencing
3. Relax min-backbone-shift types from 5 to 4
4. Correct temperature units (Celsius vs. Kelvin)
5. Filter paramagnetic samples

### **Medium Priority**:

1. Implement wildcard convention for methyl groups (CD*, CG*)
2. Relax ionic strength threshold to 150 mM
3. Account for disulfide bond states
4. Clarify experimental method filtering ('state' keyword)

### **For Investigation**:

1. Determine impact of current keyword filters on dataset size
2. Clarify max-offset parameter definition
3. Consider deuterium isotope effect corrections
4. Evaluate use of ambiguity codes for filtering
