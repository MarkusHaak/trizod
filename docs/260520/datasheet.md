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
| unfiltered | 16,851 | 5,927 |
| tolerant | 15,433 | 5,684 |
| moderate | 10,107 | 4,063 |
| strict | 3,033 | 1,254 |

Pipeline version: `trizod-2026-05-05`, `--rereference-mode both`. Dataset
release: `2026-06`.

## 2. Composition

Two layers are shipped:

1. **Per-residue scores (the labels)** — `scores/<tier>/scores.json` (JSONL).
   One record per `ID = entryID_stID_entity_assemID_entityID`. Key fields:
   - `seq` — amino-acid sequence (1-letter).
   - `zscores`, `gscores`, `k` — per-residue lists **aligned 1:1 to `seq`**
     (`null` where no shift is available; `k` = number of weighted secondary
     shifts contributing to that residue).
   - `off_<atom>` / `lacs_off_<atom>` — POTENCI/AIC residual and LACS offsets
     per backbone atom (C, CA, CB, H, HA, HB, N).
   - conditions (`pH`, `temperature`, `ionic_strength`), counts, citation.
2. **Redundancy-reduced training sets** — `train/<tier>/`:
   - `train_<tier>_best.fasta` — **canonical**: one sequence per mmseqs cluster,
     represented by the highest-quality member.
   - `train_<tier>.fasta` — paper-faithful mmseqs cluster representatives.
   - `clusters_best.tsv` / `clusters.tsv` — cluster membership (`repr`,`member`).

Held-out **test sets** (`test/`): `CheZOD117_test_set.fasta` (115 seq) and
`TriZOD_test_set.fasta` (344 seq), with per-residue targets obtainable from the
corresponding score records by `ID`.

## 3. How it was built (provenance)

1. **Source**: BMRB NMR-STAR v3 entries (17,388), parsed with integrity checks.
2. **Filtering** (4 tiers): temperature / pH / ionic-strength windows, peptide
   length, backbone-shift coverage minima, non-canonical/X caps, experiment
   method (excludes solid-state), paramagnetic-sample exclusion, keyword &
   chemical-denaturant blacklists. See `docs/filtering.md` and Table 1.
3. **Re-referencing**: LACS pre-correction (validated against 6,692 BMRB LACS
   reports) followed by POTENCI/AIC residual offset correction.
4. **Scoring**: per-residue Z-score and G-score.
5. **Bound-complex removal**: drop entries with >1 distinct entity in any
   assembly, or any non-polymer/nucleic-acid/metal entity (homo-oligomers kept)
   — ~27% of cached entries.
6. **Length filter**: drop sequences < 20 residues.
7. **Exact-sequence dedup**: keep the highest-`quality_score` record per
   identical sequence, where
   `quality_score = tier_rank·10⁶ + (n_bb_pos × n_bb_types) − max|POTENCI residual|`.
8. **Redundancy reduction (mmseqs2)**, verbatim from the original TriZOD report
   (common options `--alignment-mode 3 --cov-mode 0 -s 7.5 --comp-bias-corr 0
   --mask 0`): test-set leakage removal against CheZOD117 + TriZOD-344 in **two
   stages** — stage-1 cluster-membership removal (cluster the superset with the
   test sequences and drop any training sequence sharing a cluster with a test
   sequence) then stage-2 `easy-search`, both at 30% id / 80% cov — followed by
   `cluster` strict @ 50/80 and iterative `clusterupdate`
   moderate→tolerant→unfiltered so shared sequences keep the strict-tier
   representative.
9. **Quality-best override** of cluster representatives (`train_<tier>_best`).

The TriZOD test set is reconstructed from the current snapshot by
`build_test_set.py` (seeded). Reproduce the whole release with
`docs/260520/scripts/run_all.sh` (chains `build_final_dataset` →
`build_test_set` → `run_mmseqs_pipeline` → `cluster_best_repr` →
`package_release`).

## 4. Splits and the leakage guarantee

Every `train_<tier>` set is **redundancy-reduced against CheZOD117 + the
TriZOD-344 test set** at 30% identity / 80% coverage, in two stages
(cluster-membership removal + `easy-search`); 1,190 / 1,168 / 921 / 492 training
sequences are dropped per tier (of which 148 / 143 / 112 / 31 are transitive
leaks caught only by stage-1). The TriZOD-344 set is itself constructed (with a
fixed seed) from strict-tier clusters containing no CheZOD sequence (CheZOD117
and CheZOD1325). Therefore a model trained on any `train_<tier>` set can be
evaluated on CheZOD117 (or TriZOD-344) **with no train/test leakage** — the
basis for the UdonPred benchmark. The training sets are *not* reduced against the larger CheZOD1325
set, which enters only the test-set selection; evaluating on CheZOD1325 is
therefore not leakage-free. A release-time gate in `package_release.py` asserts
that no test ID or exact test sequence appears in any training set.

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
- `max-offset` filtering removes large post-LACS residuals (3/3/2 ppm per tier);
  ~16.7% of unfiltered entries exceed 3 ppm after LACS and are flagged.
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
├── README.md                     (this datasheet)
├── MANIFEST.json
├── train/<tier>/                 train_<tier>_best.fasta, train_<tier>.fasta,
│                                 clusters_best.tsv, clusters.tsv
├── scores/<tier>/scores.json     per-residue Z/G/k + offsets (the labels)
└── test/                         CheZOD117_test_set.fasta, TriZOD_test_set.fasta
```

The re-referenced backbone-shift NMR-STAR (`.str`) files (~1.4 GB) are an
optional separate component for NMR users (`--include-str`).
