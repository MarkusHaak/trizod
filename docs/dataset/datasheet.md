# TriZOD dataset v{{version}}: datasheet

<!-- Template: `trizod dataset finalize` renders it into the deposit's
datasheet.md with this build's counts. Every placeholder in double braces is
filled from the release; do not type counts into this file. -->

Structured after the datasheet questions of Gebru et al. (2021). The deposit's
`README.md` is the short version; `columns.md` documents every column.

## 1. Motivation

**Why was it created?** To give sequence-based protein-disorder predictors a
large, continuous, residue-level training target measured in solution, with
explicit quality tiers and test sets held out from training at a stated
homology level. NMR chemical shifts report the conformational ensemble per
residue; their deviation from random-coil values measures order.

**Who created it?** The TriZOD authors (`CITATION.cff`), Technical University of
Munich and Helmholtz Munich. The code is at {{repository}}.

## 2. Composition

**Instances.** One row per scored protein chain of a BMRB chemical-shift list
({{n_rows}} rows, {{n_columns}} columns): its sequence and, per residue, the
TriZOD **G-score** (0–1, higher = more disordered; the label), the CheZOD
Z-score, the shift count `k` and a boolean `mask`. Rows also carry sample
conditions, rule flags, referencing offsets, assembly composition, UniProt/PDB
cross-references and the split columns. The companion
`trizod_shifts.parquet` holds {{n_shifts}} assigned shifts over
{{n_shift_chains}} chains.

**Tiers.** Four nested stringency tiers, `unfiltered ⊃ tolerant ⊃ moderate ⊃
strict`, trade quantity for quality (filters: `docs/filtering.md` of the code
repository):

{{tier_table}}

**Splits.**

{{split_table}}

**Test sets.** {{test_set_label}} (`split == "test_trizod"`), drawn from the
strict tier with a fixed seed and pinned; retired chains ({{retired_by_reason}})
stay leakage targets and are not tested on. CheZOD117 ({{n_chezod117}}
BMRB-mapped chains of the 117) is the external benchmark. CheZOD117 chains are
not filtered on sample conditions (known issues). The test chains are not
independent: {{test_components}}; these components are the resampling unit.

**Labels and missing values.** A residue is scored where `mask` is true; G and
Z are NaN where it is false (999 in the UdonPred view and the deploy FASTA). Terminal residues, residues without enough backbone shifts,
and residues whose atom types an offset rejected are unscored. A shift count
`k > 0` does not imply a score. {{share_k_lt_10}} of the {{scored_residues}}
scored residues have k < 10.

**Class balance.**

{{class_balance_table}}

{{class_balance_notes}}

**Is anything confidential or about people?** No. The data describe proteins;
no personal data are involved. Annotator demographics: not applicable.

## 3. Collection

**Source.** BMRB NMR-STAR v3 entries, snapshot `{{snapshot_name}}`
({{snapshot_n_entries}} entries, retrieved {{snapshot_retrieved}}, entry-list
SHA-256 `{{entry_list_sha256}}`). BMRB offers no versioned access to single
entries, so `bmrb_files.sha256` records the SHA-256 of every entry file read.
UniProt/PDB links also come from three dated public files (BMRB's BMRB–PDB match
list, wwPDB SEQRES, SIFTS), pinned by SHA-256 in the Parquet metadata
`xref_sources` and in `MANIFEST.json`.

**Who measured the data?** The depositors of each BMRB entry; citations are in
`citation_title` and `citation_doi`.

## 4. Preprocessing and labelling

1. **Parse** every entry with integrity checks; read the sample conditions of
   the shift list's own condition list. A trailing non-letter of a deposited
   sequence is dropped, an internal one becomes X.
2. **Filter** per tier: temperature, pH and ionic-strength windows, backbone-shift
   coverage, non-canonical residue caps, experiment method, paramagnetic
   samples, keyword and perturbing-cosolvent lists, the `Physical_state` deny
   list, and the 2026-11 condition rules (pressure ≥ 400 bar out of tolerant and
   up; membrane-mimetic samples out of moderate and strict; strict ionic
   strength ≤ 0.5 M; a deposited 273 K read as not deposited).
3. **Re-reference** with LACS, then a POTENCI/AIC residual offset
   (`--rereference-mode both`).
4. **Score** each residue: Z-score and G-score against POTENCI random-coil
   predictions (machine annotation: POTENCI, LACS, the G-score).
5. **Build the training sets**: remove bound complexes and chains under 20
   residues, keep the best record per exact sequence, separate every training
   chain from the test sets (stage 1: no shared 30 %/80 % cluster; stage 2: no
   alignment that mmseqs2 reports meets {{leakage_rule}}: the best alignment of
   a pair and up to 10 alternative ones, at any identity), cluster at 50 %/80 %
   and keep the quality-best member per cluster.
6. **Gate** the packaged bundle: no training or redundant row shares an ID, a
   sequence, an exact substring of ≥ 20 residues, or an alignment under the same
   rule with any test sequence.

**Cuts on G** (metadata, not label columns): ordered G < {{cut_ordered}},
disordered G ≥ {{cut_disordered}} (the binary call), confidently disordered
G ≥ {{cut_confident}}; {{cut_rule}}. Each is the smallest G at which the
posterior of disorder reaches 0.10, 0.50 or 0.90 on {{calibration_set}}, where
π = {{calibration_pi}}:

{{calibration_table}}

Threshold framework commit `{{framework_commit}}`, results.json SHA-256
`{{results_sha256}}`. {{threshold_fit_retired}}. Because a fixed cut is a fixed likelihood ratio, the
posterior at a cut depends on the prevalence of the set it is applied to: see
the per-split π and posteriors in the class-balance table.

**Raw data.** The deposited shifts ship in `trizod_shifts.parquet` (`val_ppm`,
exactly as deposited) beside the re-referenced value.

## 5. Uses

**Intended.** Training and evaluating sequence-based disorder predictors on the
continuous G-score; selecting cleaner subsets by tier; benchmarking on
{{test_set_name}} and CheZOD117; NMR-side analyses of re-referenced shifts.

**Recommended practice.**

- Train on `gscores` with `mask`; default tier tolerant
  ({{n_train_tolerant}} training representatives); group cross-validation by
  `cluster_repr`.
- Weight or filter by `k` if label reproducibility matters: replicate agreement
  {{k_kappa}}; disordered-zone flip rate {{k_flip}}.
- Report pooled and per-chain Spearman, AUC at G ≥ {{cut_disordered}} on all,
  confident-only and k ≥ 10 residues, resampling the {{n_test_components}} test
  components (`test/TriZOD_test_set_components.tsv`) for confidence intervals;
  report also on the test chains without a remote homologue in training, and
  next to the nearest-neighbour baseline (`test/nearest_neighbour_baseline.json`).
- Compare with an earlier model only on the clean subset (README, "Earlier test
  sets are not comparable"); CheZOD-trained predictors on the test chains whose
  `chezod1325_overlap` / `chezod1174_overlap` is false.

**Not intended.** As a structural ground truth for single residues: the G-score
summarises the solution ensemble under the deposited conditions. Strict-tier
performance estimates performance under native conditions, not under
denaturants, high pressure or in membrane mimetics.

## 6. Limitations

- Labels depend on the deposited sample conditions and metadata, which carry
  errors (CHANGELOG known issues; `known_issues.tsv` lists the chains): pH*
  readings are used as is and pD readings converted to pD − 0.4, Celsius
  entered as kelvin is corrected where detected, pressures without a unit read
  as not recorded, and ionic strengths are used as deposited even where the
  sample's own components contradict them.
- Z-scores grow with `k`; prefer G.
- Homo-oligomers are retained; `n_copies` is a conservative bound, not a
  stoichiometry.
- `membrane_mimetic_sample` is false also when no sample component could be
  read; recall is limited by what the deposition states. Lipopolysaccharide
  (LPS) micelles and bilayers count as membrane mimetics. Reviewed overrides
  (the code repository's `membrane_overrides.tsv`) cover a mimetic named only
  outside the sample, a detergent outside the class table or named only in
  text or a chain name, and integral membrane proteins measured by solution NMR
  whose deposition names no mimetic (class `unnamed mimetic`); reviewed chains
  left unflagged and the known false positive are in `known_issues.tsv`.
- Construct and sample caveats: uniformly ²H-labelled protein (isotope shifts
  not corrected), labelled residues outside the UniProt-aligned span (tags),
  paramagnetic samples and non-standard residues. {{construct_caveats}}
- Homology between training and test is controlled at {{leakage_rule}}, as
  searched (every alignment mmseqs2 reports), not eliminated: {{shared_accessions}}.
  {{remote_homologs}}. {{shared_constructs}}. {{nearest_neighbour}}. Nearly
  every sequence is in the
  corpora protein language models are pretrained on.
- **Offsets come in three units.** `off_<atom>_sigma` is in multiples of the
  per-atom POTENCI RMSD (`--max-offset` is compared against it),
  `lacs_off_<atom>_ppm` is ppm, and `total_off_<atom>_ppm =
  lacs_off_<atom>_ppm + off_<atom>_sigma × REFINED_WEIGHTS[atom]` is the ppm
  total to subtract from a deposited shift. Never add the first two.
- **Side-chain shifts are re-referenced in three regimes** (`offset_source`):
  backbone shifts get the full offset the scorer used; side-chain ¹³C only the
  LACS offset, which transfers to side chains (measured on the 2026-08 build:
  slope β ≈ 0.79, lowering side-chain ¹³C error), while the POTENCI residual does
  not (β < 0.1); side-chain ¹H and ¹⁵N stay raw, because applying the backbone
  offset increased their error (β ≈ 0.08 and 0.37, below the 0.5 break-even).
  Side-chain ¹³C is withheld on chains with |`lacs_off_CA_ppm`| > 5 ppm.
- **Gross deposition errors are flagged, not removed**: C, CA, CB and N shifts
  beyond 8 (12 for CB) raw MADs of their residue × atom median carry
  `is_far_outlier`; the reference medians are frozen in the code repository.

## 7. Distribution and licence

Zenodo, concept DOI {{concept_doi}} (cite the version DOI of the record you
downloaded; `CITATION.cff`). Data: {{data_license}} ({{data_license_url}}).
Code: {{code_license}}, {{repository}}. {{bmrb_acknowledgement}}

Integrity: `MANIFEST.json` lists every file with SHA-256, size and counts;
`sha256sum -c SHA256SUMS` checks a download. `croissant.json` describes the deposit in
MLCommons Croissant 1.0 with the Responsible-AI extension (files with SHA-256,
the Parquet's columns, the splits, and sections 3–8 of this datasheet). Scored by TriZOD
`{{scoring_commit}}`; Parquet built and deposit assembled by TriZOD
`{{code_commit}}`.

## 8. Maintenance

The test set may only shrink: a chain that no longer qualifies is retired and
stays a leakage target. A recalibration of the cuts changes the metadata and
the documentation, not the labels. Every version is a separate Zenodo record
with its own DOI and a `CHANGELOG.md` entry; issues go to {{repository}}.
