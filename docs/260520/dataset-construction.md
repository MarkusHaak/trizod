# TriZOD dataset — construction note

A single end-to-end reference for how the TriZOD disorder dataset
(release `2026-06`) is built, with the exact per-tier numbers. The figure
`publication/manuscript/Figures/2026-06-22_TriZOD_redundancy_reduction.drawio`
visualises this pipeline; its caption text lives in
[`figure-caption.md`](figure-caption.md).

Reproduce everything with `scripts/build_dataset.sh`.

## 0. Inputs

| Input | What | Source |
|---|---|---|
| BMRB entries | 17,388 NMR-STAR v3 files | `data/bmrb_entries/` |
| CheZOD117 | 115 BMRB-mapped sequences (117 proteins, 2 unmapped) — the external hand-balanced benchmark | `data/2024-05-09/CheZOD117_test_set.fasta` |
| CheZOD1325 | 1,323 BMRB IDs (the larger CheZOD set) | `data/chezod/protein_nmr_1325/allseqs1325.txt` |

The four **nested** filter tiers are `strict ⊂ moderate ⊂ tolerant ⊂
unfiltered`.

## 1. Score (all tiers)

Each BMRB chemical-shift record is parsed with integrity checks, filtered
per tier (temperature / pH / ionic-strength windows, backbone-shift coverage,
keyword & denaturant blacklists, etc.), **re-referenced** (LACS pre-correction
then POTENCI/AIC residual correction, `--rereference-mode both`), and scored
(per-residue Z-score and the 0–1 **G-score**). One record per
`ID = entryID_stID_entityAssemID_entityID`.

| | strict | moderate | tolerant | unfiltered |
|---|--:|--:|--:|--:|
| **scored records** | 3,033 | 10,107 | 15,433 | 16,851 |

→ `data/release/<tier>/scores.json`. (`build_final_dataset.py` reads these.)

## 2. Clean: bound/multi-molecule removal + length filter + dedup

`build_final_dataset.py`:

- **Bound/multi-molecule removal** — drop entries with >1 distinct entity in
  any assembly, or any non-polymer (ligand) / nucleic-acid / metal entity
  (homo-oligomers kept). ~27 % of cached entries (4,558 / 16,963) are flagged.
- **Length filter** — drop sequences < 20 aa.
- **Exact-sequence dedup** — keep the highest-`quality_score` record per
  identical sequence (`quality_score = tier_rank·10⁶ + (n_bb_pos × n_bb_types) −
  max|POTENCI residual|`); a `global_repr_ID` is carried so identical sequences
  share one header across tiers (needed for `clusterupdate`).

| dropped per tier | strict | moderate | tolerant | unfiltered |
|---|--:|--:|--:|--:|
| bound / multi-molecule | 829 | 2,646 | 3,858 | 4,195 |
| length < 20 aa | 30 | 475 | 1,108 | 1,349 |
| **unique sequences** | **2,039** | **6,346** | **9,035** | **9,480** |

→ `docs/260520/data/final_dataset/<tier>/<tier>.fasta`.

## 3. Build the TriZOD test set (seeded, from the strict tier)

`build_test_set.py` reconstructs the test set on the **current snapshot**
(fixed seed = 42, so it is reproducible):

1. Cluster the strict unique sequences **together with** CheZOD117 + CheZOD1325
   at **30 % identity / 80 % coverage** → 2,511 clusters.
2. Drop every cluster containing a CheZOD sequence (1,274 CheZOD-touching) →
   **1,237 CheZOD-free** clusters. This is what makes the test set disjoint
   from CheZOD.
3. Randomly sample **25 %** of the CheZOD-free clusters → 309 clusters
   (397 member sequences).
4. Recluster those members at **50 % identity / 80 % coverage** → the **TriZOD
   test set = 344** representatives.

→ `docs/260520/data/testset/TriZOD_test_set.fasta`. CheZOD117 (the external
benchmark) is **not** rebuilt; it stays fixed.

## 4. Redundancy reduction (two-stage leakage removal + clustering)

`run_mmseqs_pipeline.py`, against the held-out test sets **CheZOD117 (115) +
TriZOD test (344)** — CheZOD1325 is *not* a training-leakage target. Common
mmseqs options throughout: `--alignment-mode 3 --cov-mode 0 -s 7.5
--comp-bias-corr 0 --mask 0`.

- **Stage 1 — remove all cluster members** (A0): cluster the unfiltered
  superset with the test sequences at 30/80 and drop every training sequence
  sharing a cluster with a test sequence (catches transitive leaks).
- **Stage 2 — search & remove** (A): `easy-search` each tier vs the test sets
  at 30/80 and drop direct hits.
- **Cluster** the strict residual at **50 % id / 80 % cov** (B), then iterative
  **`clusterupdate`** strict → moderate → tolerant → unfiltered (C) so each
  tier extends the previous and shared sequences keep the strict-tier
  representative.

| | strict | moderate | tolerant | unfiltered |
|---|--:|--:|--:|--:|
| leakage dropped (total) | 492 | 921 | 1,168 | 1,190 |
| — of which stage-1-only (transitive) | 31 | 112 | 143 | 148 |
| **non-redundant** (after leakage) | 1,547 | 5,425 | 7,867 | 8,290 |
| **training reps** (cluster representatives) | **1,254** | **4,063** | **5,684** | **5,927** |

## 5. Quality-best representative override

`cluster_best_repr.py` replaces each mmseqs cluster representative with the
highest-`quality_score` member (differs in 7.5–8.8 % of clusters) →
`train_<tier>_best.fasta` (the **canonical** training FASTA).

## 6. Package + leakage gate

`package_release.py --version 2026-06` stages the bundle and runs a hard
**leakage gate**: it fails if any test ID or exact test sequence appears in any
training set (passes: 0 collisions vs 459 test sequences).

→ `docs/260520/data/release_bundle/trizod-dataset-2026-06/` (train/ + scores/ +
test/ + README + MANIFEST).

## 7. The leakage guarantee

Every `train_<tier>` set is redundancy-reduced (two stages, 30/80) against
**CheZOD117 + TriZOD-344**, which are themselves disjoint (the TriZOD set
excludes CheZOD clusters). So a model can train on any `train_<tier>` set and be
evaluated on **CheZOD117** (the established hand-balanced benchmark, comparable
to ODiNPred/SETH) and on **TriZOD-344** (the larger, natural-BMRB-distribution
held-out set) with no train/test leakage — the basis for the UdonPred benchmark.

## Scripts (run order)

```bash
scripts/build_dataset.sh         # the whole release end to end
# = build_final_dataset → build_test_set → run_mmseqs_pipeline
#   → cluster_best_repr → package_release
```
