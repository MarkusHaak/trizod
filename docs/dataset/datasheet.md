# TriZOD dataset — datasheet

A continuous, per-residue protein-disorder dataset derived from BMRB NMR
backbone chemical shifts, re-referenced and redundancy-reduced. This datasheet
describes the dataset that is packaged for release (Zenodo); the code that
produces it is the TriZOD pipeline in this repository.

> Companion: the *trained disorder predictor* benchmarked on this dataset is
> reported separately (**UdonPred**). This dataset is built with a leakage-free
> split so that predictor can train here and test on CheZOD117 fairly.

## 1. What this dataset is

For every selected BMRB chemical-shift record, TriZOD provides a **per-residue
continuous disorder score** — the CheZOD-style **Z-score** and the new
**TriZOD G-score** (0–1, expected value independent of the number of measured
shifts) — computed from the deviation of observed backbone shifts from
POTENCI random-coil predictions, after **LACS + POTENCI/AIC re-referencing**.

It is released in **four nested stringency tiers** (`unfiltered ⊃ tolerant ⊃
moderate ⊃ strict`) so users can trade quantity for quality.

| tier | scored records | unique-seq training reps |
|---|---:|---:|
| unfiltered | 16,851 | 5,907 |
| tolerant | 15,193 | 5,590 |
| moderate | 11,175 | 4,625 |
| strict | 4,113 | 1,998 |

Pipeline version: `trizod-28be333`, `--rereference-mode both`. Dataset
release: `2026-08`.

> Tier membership is **not** comparable to release `2026-07` (v0.3.0)
> record-for-record — that release predates the filtering corrections listed in
> the bundle changelog (`bundle-README.md`). The `unfiltered` tier is the
> identical record set; `tolerant`/`moderate` lose a net 253/131 records and
> `strict` gains a net 599. The **test sets are deliberately kept comparable**
> (§4).

## 2. Composition

Four layers are shipped:

1. **Per-residue scores (the labels)** — `scores/<tier>/scores.json` (JSONL).
   One record per `ID = entryID_stID_entity_assemID_entityID`. Key fields:
   - `seq` — amino-acid sequence (1-letter).
   - `zscores`, `gscores`, `k` — per-residue lists **aligned 1:1 to `seq`**
     (`null` where no shift is available; `k` = number of weighted secondary
     shifts contributing to that residue).
   - `off_<atom>` / `lacs_off_<atom>` — POTENCI/AIC residual and LACS offsets
     per backbone atom (C, CA, CB, H, HA, HB, N). **Different units**:
     `lacs_off_<atom>` is ppm, `off_<atom>` is in sigma units (multiples of the
     per-atom POTENCI RMSD `REFINED_WEIGHTS[atom]`) — see §6.
   - conditions (`pH`, `temperature`, `ionic_strength`), counts, citation.
   - **sample-state annotation** (new): `physical_state` (the depositor's
     `_Entity_assembly.Physical_state`), `denaturant_evidence` (does the entry
     independently name a chemical denaturant?), `sample_state_evidence`
     (`solution` / `solid` / `unknown`) and `membrane_mimetic` (matched
     detergent/lipid tokens). Only `physical_state` participates in a filter;
     the rest are annotation. See `docs/filtering.md`.
2. **Redundancy-reduced training sets** — `train/<tier>/`:
   - `train_<tier>_best.fasta` — **canonical**: one sequence per mmseqs cluster,
     represented by the highest-quality member.
   - `train_<tier>.fasta` — paper-faithful mmseqs cluster representatives.
   - `clusters_best.tsv` / `clusters.tsv` — cluster membership (`repr`,`member`).
3. **Single-table Parquet** — `trizod_dataset.parquet`, one row per chain
   (16,851 rows × 63 columns), encoding every published view through the
   ordinal/categorical columns `split` (train 5,907 · excluded 8,131 ·
   redundant 2,334 · test_trizod 364 · test_chezod117 115), `train_tier`
   (strict 1,998 · moderate 2,627 · tolerant 965 · unfiltered 317),
   `pool_tier`, `cluster_repr` and `label_tier`, plus the sample-state and
   assembly-composition annotation columns.
4. **Side-chain companion** — `trizod_sidechain_shifts.parquet`, 3,458,851
   side-chain shifts (2,449,316 ¹H, 946,575 ¹³C, 62,960 ¹⁵N) over 12,643 of the
   16,851 chains, joining on `id`. **As deposited** — not re-referenced: the
   backbone offsets are not transferable to side-chain nuclei.

Held-out **test sets** (`test/`): `CheZOD117_test_set.fasta` (115 seq) and
`TriZOD_test_set.fasta` (364 seq), with per-residue targets obtainable from the
corresponding score records by `ID`.

## 3. How it was built (provenance)

1. **Source**: BMRB NMR-STAR v3 entries (17,388), parsed with integrity checks;
   16,963 parse into usable entries.
2. **Filtering** (4 tiers): temperature / pH / ionic-strength windows, peptide
   length, backbone-shift coverage minima, non-canonical/X caps, experiment
   method (solution NMR only at strict, with a `sample_state_evidence` fallback
   for the ~24 % of entries that declare no method subtype), paramagnetic-sample
   exclusion, keyword & chemical-denaturant blacklists, and an exact-match
   `_Entity_assembly.Physical_state` deny list. Keyword matching is scoped to
   sample-descriptive fields, not the paper-topic metadata. See
   `docs/filtering.md` for the per-tier defaults and the measured per-filter
   losses.
3. **Re-referencing**: LACS pre-correction (validated against 6,692 BMRB LACS
   reports) followed by POTENCI/AIC residual offset correction.
4. **Scoring**: per-residue Z-score and G-score.
5. **Bound-complex removal**: drop entries with >1 distinct protein entity in
   any assembly, or any non-polymer (ligand) / nucleic-acid / metal entity
   (homo-oligomers kept; a `water` entity does **not** count as an assembly
   member) — 4,559 of 16,963 cached entries (26.9 %) are flagged.
6. **Length filter**: drop sequences < 20 residues.
7. **Exact-sequence dedup**: keep the highest-`quality_score` record per
   identical sequence, where
   `quality_score = tier_rank·10⁶ + (n_bb_pos × n_bb_types) − max|POTENCI residual|`.
8. **Redundancy reduction (mmseqs2)**, verbatim from the original TriZOD report
   (common options `--alignment-mode 3 --cov-mode 0 -s 7.5 --comp-bias-corr 0
   --mask 0`): test-set leakage removal against CheZOD117 + TriZOD-364 in **two
   stages** — stage-1 cluster-membership removal (cluster the superset with the
   test sequences and drop any training sequence sharing a cluster with a test
   sequence) then stage-2 `easy-search`, both at 30% id / 80% cov — followed by
   `cluster` strict @ 50/80 and iterative `clusterupdate`
   moderate→tolerant→unfiltered so shared sequences keep the strict-tier
   representative.
9. **Quality-best override** of cluster representatives (`train_<tier>_best`).

The TriZOD test set is a **pinned** set re-resolved against the current tolerant
pool by `trizod dataset test-set` (`--redraw` re-runs the seeded draw and
overwrites the pin). Reproduce the whole release with `scripts/build_dataset.sh`
(chains `trizod dataset build` → `test-set` → `redundancy` →
`representatives` → `package`).

## 4. Splits and the leakage guarantee

Every `train_<tier>` set is **redundancy-reduced against CheZOD117 + the
TriZOD-364 test set** at 30% identity / 80% coverage, in two stages
(cluster-membership removal + `easy-search`); 1,241 / 1,214 / 1,067 / 618
training sequences are dropped per tier (of which 179 / 176 / 148 / 51 are
transitive leaks caught only by stage-1). The TriZOD test set is itself
constructed (with a fixed seed) from strict-tier clusters containing no CheZOD
sequence (CheZOD117 and CheZOD1325). Therefore a model trained on any
`train_<tier>` set can be evaluated on CheZOD117 (or the TriZOD test set)
**with no train/test leakage** — the basis for the UdonPred benchmark. The
training sets are *not* reduced against the larger CheZOD1325 set, which enters
only the test-set selection; evaluating on CheZOD1325 is therefore not
leakage-free. A release-time gate in `package_release.py` asserts that no test
ID or exact test sequence appears in any training set; it passes with **0 shared
IDs and 0 exact-sequence matches** against 479 test sequences.

**Test-set stability across releases.** The TriZOD test set is a pin of 365
chains re-resolved against the current tolerant pool, so that test numbers stay
comparable while training sets move with the filters. In this release 364 of the
365 resolve:

- `19342_1_1_1` ("Transmembrane-cytosolic part of Trop2") is **dropped**: it
  lists a sample component `TFE` at 70 % (`_Sample.Solvent_system` reads
  `30%H2O/70% trifluoroethanol`), now matched by the TFE denaturant token. At
  that concentration the shifts report a solvent-forced helical conformation
  rather than the aqueous state, so removing it is a correction, not collateral.
- `50998_1_1_1` is **ID-substituted** to `5599_1_1_1`, a byte-identical
  199-residue sequence under a lower entry number. ID-based joins against
  v0.3.0 must go through `TriZOD_test_set_labels.tsv` (`test_id`, `pinned_id`,
  `substituted`, `label_tier`, `length`), written next to the test FASTA.
- **17 retained chains no longer meet strict criteria** — 10 `moderate`, 7
  `tolerant` — recorded in the `label_tier` column. They are kept deliberately:
  a benchmark that shrinks whenever a filter changes cannot be compared across
  releases.

## 5. Recommended use

- **Training a sequence→disorder predictor**: use `train_<tier>_best.fasta` as
  inputs and the matching `scores/<tier>/scores.json` records (joined by `ID`)
  for per-residue targets. Use the **G-score** as the target (k-independent,
  0–1); mask residues where `k == 0` / the target is `null`.
- **Tier choice**: `moderate` balances size and quality; `strict` is cleanest;
  `unfiltered` is largest/noisiest. Tiers are nested.
- **Cluster-aware CV**: group by cluster from `clusters_best.tsv`.

## 6. Limitations & caveats

- Z-scores are unbounded below and their magnitude grows with shift count `k`
  (encodes NMR experiment coverage); prefer G-scores as a training target.
- CheZOD117 here is 115 BMRB-mapped sequences (2 of 117 did not map cleanly at
  95% id / 90% cov).
- Homo-oligomers are retained and may carry inter-chain shift perturbations.
  `n_entity_assembly_rows` and `n_copies` let you filter them, but `n_copies` is
  **both an under- and an over-count**: under-annotated homodimers list a single
  `_Entity_assembly` row, while alternative-conformer depositions (cis/trans,
  major/minor, folded/unfolded) list several rows for one molecule. It is not a
  stoichiometry guarantee in either direction; `has_conformational_isomer` marks
  the rows where the ambiguity is known to apply.
- `physical_state` is depositor free text with a long tail (47 distinct non-null
  values). The deny list is an **exact** match, so a new spelling silently stops
  matching; the pipeline logs a warning for any unrecognised value occurring on
  ≥ 5 rows.
- Membrane mimetics (SDS, DPC, bicelles, nanodiscs, …) are **annotated, not
  filtered**, at every tier. Records under a membrane mimetic are 93.7 % ordered
  and contain no disordered chains, so removing them would bias the label
  distribution toward disorder.
- Side-chain shifts in the companion Parquet are **as deposited**: the LACS and
  POTENCI offsets are backbone-derived and are not applied to them.
- `max-offset` filtering removes large post-LACS residuals (3/3/2 per tier).
  The threshold is compared against `off_<atom>`, which is in **sigma units**
  (multiples of the per-atom POTENCI RMSD), **not ppm**: `off_<atom>` is the
  mean of `(observed − POTENCI) / REFINED_WEIGHTS[atom]`. A `max-offset` of 3
  is therefore ~0.59 ppm for CA and ~0.08 ppm for HA. 1,433 of the 16,851
  unfiltered records (8.5 %) exceed 3 sigma on some atom after LACS and are
  flagged.
  `lacs_off_<atom>`, by contrast, **is** raw ppm — it is subtracted directly
  from the deposited shifts. Do not add the two columns together.
- Scores reflect the deposited shifts and conditions; re-referencing corrects
  systematic referencing errors but cannot fix fundamentally wrong depositions.

## 7. Licensing & citation

License: MIT (see repository). Derived from BMRB (https://bmrb.io/).
Cite via `CITATION.cff`; dataset DOI: *pending first Zenodo deposit*.

## 8. Files

See `MANIFEST.json` in the bundle for the exact file list with byte sizes,
SHA-256 checksums, and record/sequence counts. Layout:

```
trizod-dataset-<version>/
├── README.md                     (the shipped datasheet + changelog)
├── MANIFEST.json
├── train/<tier>/                 train_<tier>_best.fasta, train_<tier>.fasta,
│                                 clusters_best.tsv, clusters.tsv
├── scores/<tier>/scores.json     per-residue Z/G/k + offsets (the labels)
├── test/                         CheZOD117_test_set.fasta, TriZOD_test_set.fasta
├── trizod_dataset.parquet        the whole release as one table (63 columns)
└── trizod_sidechain_shifts.parquet   side-chain shifts as deposited
```

`MANIFEST.json` covers the FASTA/JSON layer (23 files, 139,765,222 bytes); the
two Parquet files are built from it afterwards by
`scripts/build_parquet_dataset.py`.

The re-referenced backbone-shift NMR-STAR (`.str`) files (~1.4 GB) are an
optional separate component for NMR users (`--include-str`).

## 9. Changelog

Release `2026-08` follows an external review of the filtering code by **Sandro
Kuppel (Helmholtz Munich)**, with input from **Reid Alderson** and **Iva
Pritisanac**. The full changelog — six filter-behaviour corrections, the policy
changes, the new columns, the side-chain companion and the corrected `off_*`
units — is in `bundle-README.md`, which ships as the bundle `README.md` and
doubles as the Zenodo description.
