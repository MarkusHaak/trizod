# Reid #1 — Chemical Shift Perturbation (CSP) analysis

## What was asked

Reid Alderson asked us to compute, for the bound-vs-unbound duplicate pairs we already have in BMRB, the standard *Chemical Shift Perturbation* used in NMR titration analysis:

```
CSP = sqrt( dH^2 + (dN/alpha)^2 ),  alpha = 5
dH = 1HN_bound - 1HN_unbound
dN = 15N_bound - 15N_unbound
```

The 15N shift is divided by α=5 because it resonates over a ~5× broader frequency range (105-130 ppm) than 1HN (5-10 ppm), so it's effectively down-weighted to the same dynamic range.

**Threshold for "significantly perturbed" residues** (Reid's trimmed-mean recipe, to avoid bias from the largest CSPs at the binding interface):
1. Drop the top 10% of CSPs.
2. Compute mean + SD on the remaining 90%.
3. Residues with CSP > threshold mark the binding interface.

## What we ran on

The same 581 bound/unbound pairs identified by `analyse_duplicate_entries.py`:
- Exact-sequence match (so we're comparing the same protein in two states).
- Conditions similar (T within 10 K, pH within 1.0) so chemical shift differences reflect binding, not different experimental setups.
- "Bound" = entry contains a non-polymer ligand or a nucleic-acid binding partner; "unbound" = single-entity entry.

61,063 residue-level CSP values across all pairs.

## Numbers from the run

| Statistic | Value |
|---|---|
| pairs analysed | 581 |
| CSP values total | 61,063 |
| trimmed mean (after dropping top 10%) | 0.112 ppm |
| threshold = trimmed mean + SD | **0.224 ppm** |

The 0.224 ppm threshold falls squarely in Reid's expected range (HN/N CSPs are small in absolute ppm because 15N is divided by 5).

## Figures

### `docs/260505/figures/csp_histogram.png`

CSP distribution across all 61,063 residue pairs, x-axis clipped at the 99th percentile for legibility, with the trimmed-mean+SD threshold drawn as a vertical red dashed line. Most residues sit well below the threshold (consistent with most bound/unbound pairs being mild perturbations across most of the sequence); the tail past the threshold is the binding-interface signal aggregated across all 581 pairs.

### `docs/260505/figures/csp_interface_example.png`

Per-residue CSP bar plot for **FKBP12 apo (bmr16925) vs FKBP12 bound (bmr16931)**. Bars exceeding the threshold are coloured red (binding interface); bars below are blue. Two large CSPs at residues 55 and 58 (~4.7 ppm and ~5.8 ppm) sit far above the threshold — the FKBP12 binding pocket signature, well-documented in the FKBP/FK506 literature.

## Talking points for the slide

1. **Method:** the standard NMR titration formula, weighted to put 1HN and 15N on equal footing.
2. **Coverage:** 581 bound/unbound pairs were already in our duplicate-analysis pipeline; CSP is a 50-line addition on top.
3. **Threshold:** Reid's trimmed-mean recipe gives 0.224 ppm — defensible, robust to outliers.
4. **Headline figure:** FKBP12 example. Two red bars at residues 55, 58 — the binding pocket. Reproduces the published interface signature without needing a 3D structure.
5. **Caveat:** HN/N alone misses the binding-effect signal that lives on 13C atoms (CA/CB). The Schumann-Williamson Δω_RMS extension (multi-atom CSP weighted by per-atom BMRB σ) is the natural next step.

## Code surface

| File | Purpose |
|---|---|
| `scripts/csp_analysis.py` | Loads tier baseline → groups by sequence → finds bound/unbound pairs with similar conditions → computes per-pair CSP → renders histogram + interface example. |

## Commits

- `ee85672` — `feat(scripts): chemical shift perturbation (CSP) analysis (Reid #1)`
