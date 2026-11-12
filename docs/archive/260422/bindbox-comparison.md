# BindBox: Deep-Dive & Comparison with TriZOD

Date: 2026-04-22

## What is BindBox?

[BindBox](https://github.com/bindresearch/BindBox) is a **Streamlit web application** developed by [Bind Research](https://github.com/bindresearch) (UK) that provides interactive NMR chemical shift analysis tools focused on **intrinsically disordered proteins (IDPs)**.

Tech stack: Streamlit, Plotly, Polars, SciPy, Matplotlib, pynmrstar.

## BindBox Tools

### 1. BMRB Chemical Shifts Dashboard (~1568 lines)
Interactive exploration of BMRB shift distributions:
- **1D histograms** per residue/atom with Gaussian fitting and POTENCI reference overlay
- **2D contour plots** (e.g., H-N, HA-CA) and **3D volume plots**
- Filtering by: residue type, atom, preceding/following residue, pH (0-14), temperature (273-373K), pressure, ionic strength, organism (eukaryote/prokaryote), species
- Dataset toggle: disordered vs. structured vs. all proteins
- CSV export of filtered data

### 2. SpinForecast: Probabilistic Peak Assignment (~928 lines)
Bayesian assignment of experimental NMR peaks to residues **without sequential connectivities**:
- Input: protein sequence + peak list (CSV, TAB, or CCPN NEF format)
- Loads BMRB shift distributions for disordered proteins, corrected to user's experimental conditions via POTENCI
- Fits Gaussian KDEs per residue/atom
- Bayes' theorem with uniform priors: P(residue | shifts)
- Output: posterior probabilities, confidence tiers (>99%, >75%, >50%, <50%), out-of-distribution warnings
- Leave-one-out atom contribution analysis (heatmap)

### 3. POTENCI: Random Coil Shift Predictor
- Same algorithm as TriZOD (Nielsen & Mulder 2018), adapted from Frans Mulder lab
- Input: protein sequence + conditions (pH, temperature, ionic strength)
- Output: predicted shifts for N, C, CA, CB, H, HA, HB
- Generates simulated 2D spectra (H-N, HA-CA, etc.)

### 4. Temperature/pH Adjustment (~550 lines)
- Input: existing peak list + reference/target conditions
- Calculates how shifts change with pH/temperature changes
- Supports single adjustment, temperature ramps, and pH titrations
- Visualization with color gradients across condition series

### 5. SpinExplorer (download page only)
- External standalone desktop GUI for NMR data processing (Bruker/Varian FIDs)
- Not part of BindBox codebase; BindBox hosts download links and tutorials

## BindBox Data

### BMRB Datasets (pre-computed Parquet files)
| Directory | Content |
|-----------|---------|
| `Shifts_Disordered/` | Raw BMRB shifts from disordered regions |
| `Shifts_Structured/` | Shifts from structured regions |
| `Shifts_All/` | All BMRB shifts |
| `Shifts_Disordered_Corrected/` | pH/temp normalized to pH=7.0, T=298K |
| `Shifts_Disordered_Corrected_Referenced/` | Corrected + referencing offsets |

- **3,320 unique BMRB entries** in the disordered dataset
- **480,599 total shift rows** across 20 amino acids
- BMRB September 2025 release
- Disorder definition: AlphaFold2 pLDDT < 70 for 30+ continuous residues
- Per-residue pLDDT scores are **not stored** (used as binary filter only)

### Columns per Parquet file
`BMRB entry ID`, `number of entities`, `entity id`, `chemical shifts (ppm)`, `residue number`, `preceding residue type`, `following residue type`, `i-2 residue type`, `i+2 residue type`, `temperature (K)`, `pH`, `ionic strength (M)`, `pressure (atm)`, `physical state`, `sample state`, `organism common name`, `organism superkingdom`, `sequence`, `atom`, `residue`

## Comparison: BindBox vs. TriZOD

### Similarities

| Aspect | BindBox | TriZOD |
|--------|---------|--------|
| Domain | NMR chemical shifts, disordered proteins | NMR chemical shifts, disordered proteins |
| POTENCI | Adapted from Frans Mulder lab | Adapted from Frans Mulder lab (same origin) |
| BMRB data | Sept 2025 release | Full BMRB archive |
| Backbone atoms | C, CA, CB, HA, H, N, HB | C, CA, CB, HA, H, N, HB |
| pH/temp corrections | Yes (via POTENCI) | Yes (via POTENCI) |
| pynmrstar | Used for NEF parsing | Used for NMR-STAR parsing |

### Key Differences

| Aspect | BindBox | TriZOD |
|--------|---------|--------|
| Interface | Web GUI (Streamlit) | CLI + Python library |
| Mode | Interactive, single-protein | Batch, database-scale (17K entries) |
| Primary output | Visualizations, probabilities | Per-residue disorder scores (Z/G) |
| Data storage | Pre-aggregated Parquet | Raw NMR-STAR + multi-level caches |
| Filtering | User-driven UI controls | Systematic 4-tier framework (16 criteria) |
| Referencing | User-supplied offsets | LACS automatic detection |
| Peak assignment | SpinForecast (Bayesian) | Not in scope |
| Disorder scoring | Not in scope | Core purpose (Z-scores, G-scores) |
| Disorder definition | AlphaFold2 pLDDT-based | CheZOD Z/G-score-based |
| Scale | One protein at a time | Full BMRB in ~24h |

### Competition

1. **POTENCI implementation** -- both maintain separate forks of the same algorithm, risk of divergence
2. **BMRB data processing** -- both parse BMRB data but via different pipelines (Parquet vs. NMR-STAR), potentially producing different "clean" datasets
3. **Audience overlap** -- NMR spectroscopists studying IDPs might use either tool
4. **pH/temperature correction** -- both apply same corrections but potentially with subtly different implementations

### Collaboration Opportunities

1. **Shared POTENCI package** -- extract into a single shared Python package, eliminate divergence, halve maintenance. TriZOD's version is more thoroughly tested/refactored; converge on that codebase.
2. **TriZOD scores in BindBox dashboard** -- they have the frontend, we have the scores. Display Z/G-scores as filterable dimension in their BMRB dashboard.
3. **BindBox as TriZOD GUI** -- wrap single-protein TriZOD scoring in their Streamlit app, making Z/G-scores accessible without CLI.
4. **LACS referencing in BindBox** -- auto-detect referencing errors instead of relying on user-supplied offsets.
5. **Unified BMRB filtering** -- use TriZOD's 16-criteria framework to generate BindBox's Parquet datasets, ensuring both tools work from identically curated data.
6. **SpinForecast + TriZOD validation** -- test whether high-confidence SpinForecast assignments correlate with specific disorder score ranges.

## Their Preprint Plans

As of April 2026, Bind Research is preparing a preprint (target: ~May 2026) focused on leveraging BMRB chemical shift distributions for probabilistic IDP peak assignment (SpinForecast). They have expressed willingness to collaborate and accept pull requests on a permissive license.
