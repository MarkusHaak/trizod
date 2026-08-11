# TriZOD dataset — construction note

A single end-to-end reference for how the TriZOD disorder dataset
(release `2026-08`) is built, with the exact per-tier numbers. The figure
`publication/manuscript/Figures/2026-06-22_TriZOD_redundancy_reduction.drawio`
visualises this pipeline; its caption text lives in
[`figure-caption.md`](figure-caption.md).

Reproduce everything with `scripts/build_dataset.sh`.

> Every number on this page is from the `2026-08` build under
> `data/interim/build/`. They differ substantially from `2026-07` (v0.3.0),
> which predates the filtering corrections summarised in
> [`bundle-README.md`](bundle-README.md#changelog--release-2026-08).

## 0. Inputs

| Input | What | Source |
|---|---|---|
| BMRB entries | 17,388 NMR-STAR v3 files, 16,963 parsing into usable entries | `data/raw/bmrb_entries/` |
| CheZOD117 | 115 BMRB-mapped sequences (117 proteins, 2 unmapped) — the external hand-balanced benchmark | `data/external/chezod117/CheZOD117_test_set.fasta` |
| CheZOD1325 | 1,323 BMRB IDs (the larger CheZOD set) | `data/external/chezod/protein_nmr_1325/allseqs1325.txt` |

The four **nested** filter tiers are `strict ⊂ moderate ⊂ tolerant ⊂
unfiltered`.

## 1. Score (all tiers)

Each BMRB chemical-shift record is parsed with integrity checks, filtered
per tier (temperature / pH / ionic-strength windows, backbone-shift coverage,
keyword & perturbing-cosolvent blacklists, the `_Entity_assembly.Physical_state` deny
list, the experiment-method whitelist and its `sample_state_evidence` fallback,
etc.), **re-referenced** (LACS pre-correction then POTENCI/AIC residual
correction, `--rereference-mode both`), and scored (per-residue Z-score and the
0–1 **G-score**). One record per `ID = entryID_stID_entityAssemID_entityID`.

| | strict | moderate | tolerant | unfiltered |
|---|--:|--:|--:|--:|
| **scored records** | 4,113 | 11,175 | 15,193 | 16,851 |

→ `data/interim/scored/<tier>/scores.json`. (`trizod dataset build` reads these.)

Compared record-for-record against the `2026-07` (v0.3.0) release, by `ID`:

| | strict | moderate | tolerant | unfiltered |
|---|--:|--:|--:|--:|
| v0.3.0 | 3,514 | 11,306 | 15,446 | 16,851 |
| in both | 3,229 | 11,108 | 15,173 | 16,851 |
| removed | 285 | 198 | 273 | 0 |
| added | 884 | 67 | 20 | 0 |
| **net** | **+599** | **−131** | **−253** | **0** |

`unfiltered` is the identical record set: every new filter policy is empty or
`off` there, so it is unchanged by construction. Strict grows because the
experiment-method fallback re-admits entries whose method-subtype tag is merely
absent, and because `interacti` no longer fires on paper-topic metadata.

## 2. Clean: bound/multi-molecule removal + length filter + dedup

`trizod dataset build`:

- **Bound/multi-molecule removal** — drop entries with >1 distinct protein
  entity in any assembly, or any non-polymer (ligand) / nucleic-acid / metal
  entity (homo-oligomers kept; a `water` entity is **not** an assembly member).
  4,559 of 16,963 cached entries (26.9 %) are flagged.
- **Length filter** — drop sequences < 20 aa.
- **Exact-sequence dedup** — keep the highest-`quality_score` record per
  identical sequence (`quality_score = tier_rank·10⁶ + (n_bb_pos × n_bb_types) −
  max|POTENCI residual|`); a `global_repr_ID` is carried so identical sequences
  share one header across tiers (needed for `clusterupdate`).

| dropped per tier | strict | moderate | tolerant | unfiltered |
|---|--:|--:|--:|--:|
| length < 20 aa | 28 | 428 | 1,043 | 1,349 |
| bound / multi-molecule (of those ≥ 20 aa) | 851 | 2,813 | 3,843 | 4,192 |
| kept rows | 3,234 | 7,934 | 10,307 | 11,310 |
| **unique sequences** | **3,050** | **7,225** | **8,934** | **9,482** |

→ `data/interim/build/final_dataset/<tier>/<tier>.fasta`, with the per-tier and
overall counts in `final_dataset_summary.json`.

## 3. Build the TriZOD test set (pinned; seeded draw behind it)

The seeded recipe (fixed seed = 42) is:

1. Cluster the strict unique sequences **together with** CheZOD117 + CheZOD1325
   at **30 % identity / 80 % coverage**.
2. Drop every cluster containing a CheZOD sequence. This is what makes the test
   set disjoint from CheZOD.
3. Randomly sample **25 %** of the CheZOD-free clusters.
4. Recluster those members at **50 % identity / 80 % coverage** → the TriZOD
   test-set representatives.

**The test set is pinned**, so that recipe only runs on a deliberate redraw. The
pin holds **365** sequences at `trizod/dataset/pinned/TriZOD_test_set.fasta`;
`trizod dataset test-set` resolves each pinned sequence against the **current
tolerant pool** and writes `TriZOD_test_set.fasta` plus
`TriZOD_test_set_labels.tsv` (`test_id`, `pinned_id`, `substituted`,
`label_tier`, `length`). Resolving against *tolerant* rather than *strict* is
what keeps the benchmark stable when a filter change moves the strict boundary.

For the `2026-08` build (`build_test_set_summary.json`):

| | |
|---|--:|
| pinned | 365 |
| **resolved** | **364** |
| dropped | 1 — `19342_1_1_1` |
| ID-substituted | 1 — `50998_1_1_1` → `5599_1_1_1` |
| `label_tier` = strict / moderate / tolerant | 347 / 10 / 7 |

`19342_1_1_1` ("Transmembrane-cytosolic part of Trop2") is dropped because it
lists a sample component `TFE` at **70 %** (`_Sample.Solvent_system` reads
`30%H2O/70% trifluoroethanol`), now matched by the new TFE cosolvent token —
tolerant already carries it, so the chain leaves the resolve pool. At that
concentration the shifts report a solvent-forced helical conformation, not the
aqueous state, so this is a correction rather than collateral.
`50998_1_1_1` → `5599_1_1_1` is a byte-identical 199-residue sequence under a
lower entry number; **ID-based joins against v0.3.0 must go through
`TriZOD_test_set_labels.tsv`.** The 17 chains whose `label_tier` is no longer
`strict` are retained on purpose — test numbers must stay comparable across
releases even while training sets move.

→ `data/interim/build/testset/`. CheZOD117 (the external benchmark) is **not**
rebuilt; it stays fixed at 115.

Run `trizod dataset test-set --redraw` to re-run the seeded recipe from scratch
and overwrite the pin — only when deliberately regenerating the test set (e.g.
after a BMRB refresh). The draw is **not** stable under a pool change even with
the same seed, so a redraw invalidates comparison with every previous release.

## 4. Redundancy reduction (two-stage leakage removal + clustering)

`trizod dataset redundancy`, against the held-out test sets **CheZOD117 (115) +
TriZOD test (364)**, 479 sequences in total — CheZOD1325 is *not* a
training-leakage target. Common mmseqs options throughout:
`--alignment-mode 3 --cov-mode 0 -s 7.5 --comp-bias-corr 0 --mask 0`.

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
| input (unique sequences, §2) | 3,050 | 7,225 | 8,934 | 9,482 |
| leakage dropped (total) | 618 | 1,067 | 1,214 | 1,241 |
| — of which stage-2 `easy-search` hits | 567 | 919 | 1,038 | 1,062 |
| — of which stage-1-only (transitive) | 51 | 148 | 176 | 179 |
| **non-redundant** (after leakage) | **2,432** | **6,158** | **7,720** | **8,241** |
| **training reps** (cluster representatives) | **1,998** | **4,625** | **5,590** | **5,907** |

→ `data/interim/build/mmseqs/`.

## 5. Quality-best representative override

`trizod dataset representatives` replaces each mmseqs cluster representative
with the highest-`quality_score` member → `train_<tier>_best.fasta` (the
**canonical** training FASTA). The cluster count is unchanged (1,998 / 4,625 /
5,590 / 5,907); only which member represents each cluster moves — 171 / 401 /
457 / 472 clusters (8.6 / 8.7 / 8.2 / 8.0 %), listed in
`cluster_repr_overrides_<tier>.tsv`.

## 6. Package + leakage gate

`trizod dataset package --version 2026-08` stages the bundle and runs a hard
**leakage gate**: it fails if any test ID or exact test sequence appears in any
training set. It passes with **0 shared IDs and 0 exact-sequence matches**
against the 479 test sequences.

→ `data/interim/build/release_bundle/trizod-dataset-2026-08/` (train/ + scores/
+ test/ + README + MANIFEST — 24 files, 139,790,629 bytes). The single-table
`trizod_dataset.parquet` (16,851 rows × 70 columns) and the chemical-shift
companion `trizod_shifts.parquet` (11,839,037 assigned shifts on canonical
residues over all 16,851 chains — 8,380,186 backbone, 3,458,851 side chain, of
which 12,643 chains carry side-chain assignments) are then built from that
bundle by `scripts/build_parquet_dataset.py`.

## 7. The leakage guarantee

Every `train_<tier>` set is redundancy-reduced (two stages, 30/80) against
**CheZOD117 + the 364-sequence TriZOD test set**, which are themselves disjoint
(the TriZOD set excludes CheZOD clusters). So a model can train on any
`train_<tier>` set and be evaluated on **CheZOD117** (the established
hand-balanced benchmark, comparable to ODiNPred/SETH) and on the TriZOD test set
(the larger, natural-BMRB-distribution held-out set) with no train/test leakage
— the basis for the UdonPred benchmark.

## Scripts (run order)

```bash
scripts/build_dataset.sh         # the whole release end to end
# = trizod dataset build → test-set → redundancy → representatives → package
```
