# Figure caption — TriZOD redundancy-reduction pipeline

Caption text for `publication/manuscript/Figures/2026-06-22_TriZOD_redundancy_reduction.drawio(.png)`.
The dataset construction it depicts is documented in
[`dataset-construction.md`](dataset-construction.md).

## Short caption (figure-only, ~2 sentences)

> **Construction of the TriZOD dataset.** An automated pipeline converts
> 17,388 BMRB entries into per-residue disorder labels across four nested
> filter tiers (*strict ⊂ moderate ⊂ tolerant ⊂ unfiltered*) and, after
> bound-complex removal, exact-sequence de-duplication, held-out-test-set
> leakage removal, and mmseqs clustering/clusterupdate, into
> redundancy-reduced training sets of 1,254–5,927 proteins; a seeded
> side-branch builds the CheZOD-disjoint TriZOD test set (344) that, with the
> external CheZOD117 (115), forms the held-out evaluation data.

## Full caption (with symbol key)

> **Construction of the TriZOD dataset and its leakage-free test split.**
> *(right) Training-set construction.* Each column is one of four **nested**
> filter-stringency tiers (*strict ⊂ moderate ⊂ tolerant ⊂ unfiltered*); each
> row gives the number of sequences remaining after a step. The 17,388 BMRB
> entries (gray cylinder = database) are scored per tier (*scored*), reduced to
> *unique* sequences by removing bound/multi-molecule complexes and sequences
> < 20 aa and de-duplicating identical sequences, then reduced to
> *non-redundant* sets by removing test-set leakage — an mmseqs `search`
> (30 % identity / 80 % coverage) against the held-out test sets, applied to
> every tier (red box). The non-redundant sets are turned into the per-tier
> *training sets* (green) by mmseqs `cluster` on *strict* followed by iterative
> `clusterupdate` *strict → moderate → tolerant → unfiltered* (50 % identity /
> 80 % coverage), so each tier extends the previous and shared sequences keep
> the strictest-tier representative.
> *(left, shaded) Test-set construction.* The *strict* tier's unique sequences
> (dashed arrow: provenance) are clustered together with the external CheZOD117
> and CheZOD1325 sets (orange parallelograms) at 30 % identity / 80 % coverage;
> clusters containing any CheZOD sequence are dropped (so the test set is
> CheZOD-disjoint); a fixed-seed random 25 % of the remaining clusters is
> reclustered at 50 % identity / 80 % coverage (diamond = random sampling) to
> give the **TriZOD test set** (344, blue). Together with CheZOD117 (115) it
> forms the **held-out test sets** used for the leakage removal above.
> CheZOD1325 enters only this construction and is not itself a training-leakage
> target. **Colours:** gray = database; orange = external CheZOD reference set;
> blue = TriZOD test set; green = training set; white = intermediate count. All
> mmseqs steps use `--alignment-mode 3 --cov-mode 0 -s 7.5 --comp-bias-corr 0
> --mask 0`.

## Numbers shown (for cross-checking the figure)

| row | strict | moderate | tolerant | unfiltered |
|---|--:|--:|--:|--:|
| scored | 3,033 | 10,107 | 15,433 | 16,851 |
| unique | 2,039 | 6,346 | 9,035 | 9,480 |
| non-redundant | 1,547 | 5,425 | 7,867 | 8,290 |
| training sets | 1,254 | 4,063 | 5,684 | 5,927 |

Test-set side-branch: strict 2,039 + CheZOD117 115 + CheZOD1325 1,323 →
2,511 clusters → drop 1,274 CheZOD-touching → 1,237 CheZOD-free → random 25 % →
309 clusters (397 seq) → **TriZOD test 344**.
