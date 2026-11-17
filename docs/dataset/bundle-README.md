# TriZOD dataset v{{version}}

<!-- Template: `trizod dataset finalize` renders it into the deposit's README.md
with this build's counts. Every placeholder in double braces is filled from the
release Parquet, its metadata, the package MANIFEST and the shift table; do not
type counts into this file. -->

Per-residue **protein-disorder** labels from BMRB NMR backbone chemical shifts.
Observed shifts are re-referenced (LACS, then a POTENCI/AIC residual offset) and
scored against POTENCI random-coil predictions, giving a CheZOD-style Z-score and
the continuous TriZOD **G-score** (0–1, higher = more disordered) per residue.

- **Rows:** {{n_rows}} scored protein chains · **Columns:** {{n_columns}}
- **Test set:** {{test_set_label}}, plus CheZOD117 ({{n_chezod117}} chains)
- **Licence:** data {{data_license}}; code {{code_license}} ({{repository}})
- **Source:** BMRB snapshot `{{snapshot_name}}` ({{snapshot_n_entries}} entries,
  retrieved {{snapshot_retrieved}})
- **Code:** scored by `{{scoring_commit}}` ({{scoring_pipeline_version}}: every
  entry scored, the dataset chain run into the package bundle); Parquet built
  and deposit assembled by `{{code_commit}}` ({{pipeline_version}})
- **Cite:** `CITATION.cff`; concept DOI {{concept_doi}} (all versions)

## Files

| File | Content |
|---|---|
| `trizod_dataset.parquet` | **The dataset.** One row per scored chain; per-residue `gscores`, `zscores`, `k` and `mask` lists aligned 1:1 to `sequence`; split, tier and cluster columns; sample conditions, rule flags, cross-references. Key-value metadata: version, licence, code commit, snapshot, test-set name, cuts, calibration |
| `trizod_shifts.parquet` | Every assigned chemical shift on a canonical residue, backbone and side chain, raw and re-referenced: {{n_shifts}} rows over {{n_shift_chains}} chains. Joins on `id` |
| `udonpred/` | UdonPred v2 training view generated from the Parquet: `trizod_<tier>/{train,valid,test}.jsonl` (`y` = G, 999 where masked) and `chezod117/test.jsonl`; its own README |
| `disorder_trizod.fasta` (+ `.summary.json`) | SETH/ODiNPred-style view: tolerant training set and the test set, `SET=`, `TARGET=` (G, 999.0 where masked), `MASK=` |
| `test/` | The test sets as FASTA: `TriZOD_test_set.fasta` ({{test_set_name}}), `TriZOD_test_retired.fasta` (retired chains, leakage targets only), `CheZOD117_test_set.fasta`, `CheZOD117_original_sequences.fasta` (the 117 sequences as SETH published them); `TriZOD_test_set_labels.tsv`; `TriZOD_test_set_components.tsv` (each test chain's homology component, the unit to resample); `test_training_shared_accessions.tsv` (test and training chains that share a UniProt accession, with the decision); `shared_construct_review.tsv` (the reviewed shared constructs); the pin provenance |
| `crosswalk/` | Every TriZOD test set since 2024 and the status of its sequences in this release (`test_set_crosswalk.tsv`/`.json`, `test_set_overlap.tsv`) |
| `class_balance.tsv` | Label distribution per split (table below, unrounded) |
| `known_issues.tsv` | Every chain with a known issue or a change named in the CHANGELOG: `chain_id`, `issue`, `evidence` ({{n_known_issues}} rows) |
{{known_issues_inputs_row}}
| `columns.md` | Column dictionary: name, Arrow type, nullability, meaning, null/NaN rule |
| `datasheet.md` | Datasheet: motivation, composition, collection, preprocessing, uses, limitations |
| `CHANGELOG.md` | Changes since v{{previous_version}}, and the known issues |
| `bmrb_files.sha256` | SHA-256 of every BMRB entry file of the snapshot (`sha256sum -c` format) |
| `MANIFEST.json`, `SHA256SUMS` | SHA-256, size and row/chain counts of every file; version, licence, commits, snapshot and cross-reference-source hashes, cuts, calibration, leakage rule |
| `LICENSE.md`, `CITATION.cff`, `zenodo.json` | Licence, citation, deposit metadata |
{{validation_row}}

Check the download with `sha256sum -c SHA256SUMS`.

## Quick start

```python
import pandas as pd

df = pd.read_parquet("trizod_dataset.parquet")

TIERS = ["unfiltered", "tolerant", "moderate", "strict"]   # loose -> strict, nested
def training_set(df, tier):
    keep = set(TIERS[TIERS.index(tier):])                  # this tier and stricter
    return df[(df.split == "train") & df.train_tier.isin(keep)]

train = training_set(df, "tolerant")         # the default; UdonPred trained on it
test = df[df.split == "test_trizod"]          # {{test_set_name}}
chezod = df[df.split == "test_chezod117"]

row = train.iloc[0]
seq, g, mask, k = row["sequence"], row["gscores"], row["mask"], row["k"]
# loss = mse(pred[mask], g[mask])     # g is NaN exactly where mask is False
```

To overlay replicate measurements of one sequence (repeat depositions of
ubiquitin, say), group on the same-sequence link,
`df[df.sequence_group_size > 1].groupby("sequence_group")`: the chains of a
group have identical sequences, so their per-residue lists align position by
position, and the group spans every split (check `split` before mixing test and
training chains). Other constructs of the same protein (a tag, a truncation, a
mutant) usually share `uniprot_ids` but not `sequence_group`; overlaying them
needs a residue alignment, which the release does not provide.

## Labels, mask and cuts

- **The label is the continuous G-score** (`gscores`). Train on it as a
  regression target; the cuts below are for evaluation and are recorded as
  metadata (`cuts`, `cut_rule`), never as label columns.
- **Mask on `mask`, not on `k`.** `mask[i]` is true exactly where residue *i* is
  scored, i.e. where G is finite. G and Z are NaN where it is false in the
  Parquet, and 999 in the UdonPred view and the deploy FASTA. A residue can
  have shifts (`k > 0`) and still no score.
- **Cuts:** ordered G < {{cut_ordered}}; disordered (the binary call)
  G ≥ {{cut_disordered}}; confidently disordered G ≥ {{cut_confident}}. Rule:
  {{cut_rule}}.
- **Calibration.** Each cut is the smallest G at which the posterior of disorder
  reaches 0.10, 0.50 and 0.90 on {{calibration_set}}, against a consensus of
  X-ray, DisProt and hetNOE references. There the prevalence of disorder is
  π = {{calibration_pi}}; the likelihood ratios are {{calibration_lr}}.
  Framework commit `{{framework_commit}}`, results.json SHA-256
  `{{results_sha256}}` (metadata `calibration`).

{{calibration_table}}

- **A fixed cut is a fixed likelihood ratio**, so its posterior moves with the
  prevalence of the set it is applied to; the class-balance table gives each
  split's π and posterior.
- **k ≥ 10.** `k` counts the backbone shifts behind a residue's score. k ≥ 10 is
  a conventional reliability cut, not a filter: replicate agreement of the
  binary call rises gradually with k ({{k_kappa}}; disordered-zone flip rate
  {{k_flip}}). {{share_k_lt_10}} of all scored residues have k < 10. Weight or
  filter by `k` yourself.

## Class balance

{{class_balance_table}}

{{class_balance_notes}}

## Splits and the test set

Four **nested** stringency tiers (`unfiltered ⊃ tolerant ⊃ moderate ⊃ strict`):

{{tier_table}}

{{split_table}}

- `train`: training representatives, one per 50 %/80 % mmseqs cluster;
  `train_tier` is the strictest tier at which the chain is one. `redundant`:
  members of a training cluster headed by another chain (`cluster_repr`).
  `excluded`: scored, in no training pool (bound complexes, chains under 20
  residues, chains in no tier's pool).
- **{{test_set_name}}** (`split == "test_trizod"`, n = {{n_test}}) is drawn from
  the strict tier and pinned; from this release on it may only shrink. A pinned
  chain that no longer qualifies is retired (`split == "test_retired"`,
  {{n_retired}} chains: {{retired_by_reason}}): it is not tested on, and it and
  its homologues stay out of every training set.
- **How training is separated from the test sets.** Every training and redundant
  chain is separated from {{test_set_name}}, the retired chains and CheZOD117
  (the released chains and the 117 original sequences) by a searched rule: no
  alignment that mmseqs2 reports between a training chain and a test chain
  meets {{leakage_rule}}; none shares a sequence or an exact substring of 20 or
  more residues with one, and none shares a 30 %/80 % cluster with one. mmseqs2
  reports the best alignment of a pair and up to 10 alternative ones, at any
  identity (`--min-seq-id 0`), and the rule is applied to each. The release
  gate re-checks every training and redundant row against every test sequence
  before packaging. This controls homology at that rule, as searched; it does
  not eliminate it. Below it: {{shared_accessions}}. {{remote_homologs}}.
  {{shared_constructs}}. The overlap with the sequence corpora protein language
  models are pretrained on remains as well.
- **CheZOD.** The test set is disjoint from CheZOD117 under the leakage rule
  (overlapping chains are retired). Per-chain flags `chezod1325_overlap` and
  `chezod1174_overlap` mark the {{n_test_overlap_chezod1325}} and
  {{n_test_overlap_chezod1174}} test chains with relatives in ODiNPred's
  (CheZOD1325) and SETH's or ADOPT's (CheZOD1174) training sets; benchmark such
  predictors on the clean subsets ({{n_test_clean_chezod1325}} and
  {{n_test_clean_chezod1174}} of the {{n_test}} test chains).
- **Resampling unit.** Test chains are not independent: {{test_components}}.
  Bootstrap or jackknife over these components, not over chains or residues.

### Earlier test sets are not comparable

The TriZOD test set has been redrawn or trimmed several times, and each change
turned former test chains into training data. Comparisons against
**UdonPred's published Trizod348 results**, against the **v0.2.0** and
**v{{previous_version}}** test sets, and on the **chains trimmed from the
pin of August 2026** (`known_issues.tsv`, issue `pin_trimmed`, gives each
one's split now) are therefore not valid on this release. `crosswalk/` lists every earlier set and
where its sequences are now (exact sequence match, a lower bound on overlap):

{{crosswalk_table}}

To compare a model trained here with one trained on an earlier release, use the
clean subset: the {{test_set_name}} chains absent from both models' training
data, by exact sequence and preferably under the same 30 %/80 % rule. UdonPred's
published training data are on figshare (10.6084/m9.figshare.31444642).

## Evaluation guidance

- Report pooled residue-level Spearman (comparable with the CheZOD117
  literature) and per-chain Spearman; for confidence intervals resample the
  {{n_test_components}} test components (`test/TriZOD_test_set_components.tsv`),
  not chains or residues.
- Report a model also on the test chains without a remote homologue in training
  (`known_issues.tsv`, issue `remote_homolog_profile`).
- Report AUC at G ≥ {{cut_disordered}} on all scored residues, on confident
  residues only (G < {{cut_ordered}} or G ≥ {{cut_confident}}), and on k ≥ 10.
- State whether predictions were smoothed before scoring.
- The strict-only test set measures performance under native, deposited
  conditions.

## Columns

`columns.md` documents all {{n_columns}} columns. The table metadata
(`pyarrow.parquet.read_schema("trizod_dataset.parquet").metadata`) restates the
semantics of the flag columns, `label_source`, the offsets' units, the sort
order and every provenance key.

- **Labels come from one record per chain** (`label_tier_source`): the tier that
  selected the chain. Tolerant masks an over-limit atom type out of the score
  where moderate and strict reject the chain, so per-tier records are not
  interchangeable.
- **Offsets come in three units.** `off_<atom>_sigma` is in multiples of the
  per-atom POTENCI RMSD, `lacs_off_<atom>_ppm` in ppm; subtract
  `total_off_<atom>_ppm` from a deposited shift to reproduce the shift TriZOD
  scored, and never add the first two.
- **Rule flags.** `high_pressure` (≥ 400 bar, excluded from tolerant and up;
  null = not recorded, so filter with `high_pressure IS NOT TRUE`),
  `membrane_mimetic_sample` (excluded from moderate and strict; lipopolysaccharide
  (LPS) micelles and bilayers count as membrane mimetics, as do the reviewed
  overrides of the code repository's `membrane_overrides.tsv`),
  `temperature_placeholder` (deposited 273 K read as not deposited; strict
  excludes it), and `ionic_strength` (strict caps it at 0.5 M).
- **Cross-references.** `xrefs`, `uniprot_ids`, `uniprot_source` and `pdb_ids`
  link every chain to UniProt and the PDB, each link labelled by its evidence
  (`exact`, `author`, `sifts`, `unspecified`, `blast`).
- The table is sorted by `(entry_id, id)` in string order in 500-row groups, so
  a filter on `entry_id` reads {{row_group_note}}.

## Chemical shifts (`trizod_shifts.parquet`)

One row per `(chain, residue, atom)`; `seq_id` is a 1-based index into the main
table's `sequence`. `val_ppm` is always as deposited. `val_corrected_ppm` is
`val_ppm − offset_applied_ppm` where an offset demonstrably transfers, and null
otherwise (never a copy): backbone shifts get the full offset the scorer used,
side-chain ¹³C the LACS offset only, side-chain ¹H and ¹⁵N none
(`offset_source = not_transferable`). C, CA, CB and N shifts carry a
far-outlier flag (`mad_z`, `is_far_outlier`; a flag, not a filter). Shifts on
non-canonical residues are not included. Units and the correction formula are
in the field and table metadata; the datasheet explains the three regimes.

## Provenance and licence

Built from BMRB snapshot `{{snapshot_name}}` (entry-list SHA-256
`{{entry_list_sha256}}`; per-file hashes in `bmrb_files.sha256`). TriZOD
`{{scoring_commit}}` scored the entries and ran the tier, redundancy and
test-set chain; TriZOD `{{code_commit}}` built the Parquet and assembled this
deposit (MANIFEST `code`). Data: {{data_license}} ({{data_license_url}}). Code:
{{code_license}}. {{bmrb_acknowledgement}}

What changed since v{{previous_version}} ({{previous_version_date}}), and the
known issues, are in `CHANGELOG.md`; `known_issues.tsv` lists the chains.
