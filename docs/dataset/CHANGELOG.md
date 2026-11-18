# TriZOD dataset changelog

<!-- Template: `trizod dataset finalize` renders it with this build's counts
into the deposit's CHANGELOG.md, which is also pasted into the Zenodo version
notes. Every placeholder in double braces is filled from the release; type no
counts of the current release into this file. The only other version it
compares with is the previous published one (its counts come from
finalize.V030). The known-issue paragraphs come from trizod.dataset.known_issues
(finalize --pkl-dir, --construct-caveats with --xref-sources, --ionic-drivers,
and docs/dataset/known_issues_reviewed.tsv) and
trizod.dataset.residual_similarity; a missing input renders as "not
assessed". -->

## v{{version}}

Against **v{{previous_version}}** (Zenodo, {{previous_version_date}}), the
previous published version. Labels, tier membership and the test set all
changed, so no number computed on v{{previous_version}} carries over;
regenerate it.

### Test set: {{test_set_label}}

- **Named and frozen.** The TriZOD test set is now {{test_set_name}}, n =
  {{n_test}}. From this release on it may only shrink: a chain that no longer
  qualifies is retired (`split == "test_retired"`, reason in `retired_reason`),
  is not tested on, and stays a leakage target with its homologues.
- **Redrawn.** v{{previous_version}} tested on {{n_test_previous}} chains. The
  set was redrawn from the corrected strict pool by the original seeded recipe
  and is resolved against the strict tier of this release.
- **Retired:** {{n_retired}} chains ({{retired_by_reason}}).
  {{retired_reasons_explained}}.
- **Shared constructs.** {{shared_constructs}}.
- **Resampling unit.** Test chains are not independent: {{test_components}};
  resample components for confidence intervals.
- **Earlier test sets are not comparable**: UdonPred's published Trizod348
  results, v0.2.0 and v{{previous_version}}. The new `crosswalk/` lists every
  earlier set and where its sequences are now; the README gives the overlap
  table and the clean-subset rule.

### Tiers

{{changelog_tier_table}}

### Four new condition rules

{{rule_table}}

- Chains a rule removes stay in the release at a looser tier; the flag column
  says why.
- **Re-scored chains.** A deposited temperature of exactly 273 K is read as not
  deposited ({{n_temperature_placeholder}} chains carry
  `temperature_placeholder`). Strict excludes them; the other tiers re-score them
  at the 298 K default, which changes their labels. `known_issues.tsv` lists
  every such chain (`temperature_placeholder_rescore`).
- **CheZOD117 placeholder chains.** {{chezod117_placeholders}} CheZOD117 is not
  filtered on sample conditions, so these stay test chains with their new
  labels.
- Pressure ≥ 400 bar describes a deliberately perturbed ensemble; membrane
  mimetics change the ensemble a disorder label describes; the strict ionic cap
  is a convention that keeps the cleanest tier in dilute buffer.

### From one cut to three

v{{previous_version}} stated no cut; the TriZOD analyses until 2026-11 called
a residue disordered at G > 0.40. This release replaces that single cut by three
calibrated cuts with the convention G ≥ t: ordered G < {{cut_ordered}}, disordered (binary call)
G ≥ {{cut_disordered}}, confidently disordered G ≥ {{cut_confident}}. They are
the smallest G at which the posterior of disorder reaches 0.10, 0.50 and 0.90 at
the calibration prevalence π = {{calibration_pi}}; likelihood ratios
{{calibration_lr}}. The cuts live in the Parquet metadata (`cuts`, `cut_rule`,
`calibration`), in the README and in the datasheet, never in a label column:
the continuous G stays the label. k ≥ 10 is documented as a conventional
reliability cut.

### Columns: {{n_columns_previous}} → {{n_columns}}

- **Renamed (breaking):** `off_<atom>` → `off_<atom>_sigma` (multiples of the
  per-atom POTENCI RMSD, not ppm) and `lacs_off_<atom>` →
  `lacs_off_<atom>_ppm`; v{{previous_version}} documented `off_<atom>` as ppm,
  which it never was. Migration is mechanical.
- **New ({{n_new_columns}}):** {{new_columns}}.
- Among them: the ppm total `total_off_<atom>_ppm` to subtract from a deposited
  shift; sample-state annotation (`physical_state`, `cosolvent_evidence`,
  `sample_state_evidence`, `membrane_mimetic`); assembly composition (`n_entities`
  … `is_bound`); `label_tier` and `label_tier_source`; `pressure_bar`; the rule
  flags `membrane_mimetic_sample`, `temperature_placeholder` and `high_pressure`;
  UniProt/PDB cross-references (`xrefs`, `uniprot_ids`, `uniprot_source`,
  `pdb_ids`); the CheZOD overlap flags `chezod117_overlap`, `chezod1325_overlap`
  and `chezod1174_overlap`; `retired_reason`; and the same-sequence link
  `sequence_group` (an ID hashed from the sequence alone, so stable across
  releases, shared by every chain with an identical sequence in every split)
  with `sequence_group_size`, for overlaying replicate measurements.
- Rows are sorted by `(entry_id, id)` in 500-row groups for selective reads.
- `columns.md` documents every column; the Parquet metadata now records the
  version, licences, code commit, scoring commit, BMRB snapshot, test-set
  name, cuts and calibration.

### New files

- `trizod_shifts.parquet`: every assigned shift on a canonical residue, backbone
  and side chain, raw and re-referenced, with a far-outlier flag.
- `udonpred/`: the UdonPred v2 JSONL view (train/valid/test per tier, `y` = G,
  999 where masked), generated from the Parquet and verified against it.
- `disorder_trizod.fasta`: the SETH/ODiNPred-style view.
- `crosswalk/`: the test-set crosswalk across every earlier test set.
- `test/`: the test FASTAs, including `TriZOD_test_retired.fasta` and
  `CheZOD117_original_sequences.fasta`.
- `known_issues.tsv`: every chain with a known issue, with its evidence.
- `datasheet.md`, `columns.md`, `class_balance.tsv`, `bmrb_files.sha256`,
  `MANIFEST.json` (now over every file, with the scoring commit beside the
  build commit) and `SHA256SUMS`.
- `test/test_residual_similarity.tsv`: each test chain's closest tolerant
  training representative at any identity (identity, coverage, bit score), a
  description of how close the test set comes to training below the leakage
  rule.
- `croissant.json`: MLCommons Croissant 1.0 metadata with the Responsible-AI
  extension (every file with SHA-256, the Parquet's columns, the splits).

### Fixes

- **Leakage search in both directions.** Stage 2 and the release gate search
  with training and with test chains as queries, under a bit-score floor of 27
  bits that does not depend on database size; the gate reads every training
  and redundant row. Before, a one-directional search missed pairs that only
  the reverse search finds.
- **Every alignment of a pair.** The leakage searches (stage 2, the gate, the
  CheZOD-overlap flags, the within-test check, the UdonPred validation split)
  report up to 10 alternative alignments per pair besides the best
  (`--alt-ali 10`) and run with `--min-seq-id 0`: mmseqs2 computes the
  alternatives of a pair only behind a best alignment that passes
  `--min-seq-id`, so with 0.3 a pair whose best alignment was below 30 %
  identity was dropped with all its alternatives. The rule (30 % identity over
  80 % of either chain, 27 bits) is applied to every reported alignment, so a
  training chain that meets the rule only on the second alignment of a repeat
  or tandem construct, or only behind a best alignment below 30 % identity, is
  removed.
- **CheZOD117 as leakage targets**: all 117 original sequences, including the
  two that map to no release chain; test chains overlapping CheZOD117 are
  retired.
- **Membrane-mimetic recall**: entity-linked detergents, sample types and sample
  text are read; reviewed cases the rule cannot see are listed in the code
  repository (`trizod/dataset/data/membrane_overrides.tsv`): a mimetic named
  only in the title, assembly name or citation; a detergent outside the class
  table (sulfobetaines, LMNG, alkyl betaines) or named only in sample text or a
  chain name; and integral membrane proteins measured by solution NMR whose
  deposition names no mimetic (class `unnamed mimetic`, probable).
  Lipopolysaccharide (LPS) micelles and bilayers count as membrane mimetics,
  everywhere and by rule (class `LPS`).
- **Cosolvent spellings**: the isopropanol abbreviations `isoPrOH` and `IPA`
  (and `iPrOH`, `i-PrOH`) and the hyphenated TFE formula `CF3-CD2-OH` are read
  as perturbing cosolvents; in v{{previous_version}}, 50 % isopropanol
  (50767_1_1_1) and 50 % TFE (31073_1_1_1) passed the moderate tier.
- **Sequence alphabet**: a trailing stop codon or apostrophe of a deposited
  sequence is dropped, an internal non-letter becomes X. Sequences hold the 20
  standard residues plus X, U and Z.
- **Filter corrections** after an external review of the filter code (Sandro
  Kuppel): keyword fields appended character by character, an empty regex
  alternative that matched everything, a `has_metal` flag that never fired,
  missed nucleic-acid hybrids, water counted as an assembly member; keyword
  search scoped to sample-descriptive fields; perturbing cosolvents (TFE, HFIP,
  DMSO, alcohols) found where declared; `Physical_state` used; undeclared
  experimental methods judged on the sample-state evidence.
- **Scores**: v{{previous_version}}'s tiers came from a superseded scoring run
  (issue #20); this release is built end to end on corrected scores.

### Snapshot and licence

- BMRB snapshot `{{snapshot_name}}` ({{snapshot_n_entries}} entries) replaces
  the snapshot of 2025-12-01. BMRB serves only the current file, so
  `bmrb_files.sha256` now records every file read.
- The data are now licensed {{data_license}} (v{{previous_version}}: MIT); the
  code stays {{code_license}}. {{bmrb_acknowledgement}}

### Known issues

`known_issues.tsv` lists every affected chain, one row per chain and issue
(`chain_id`, `issue`, `evidence`; {{n_known_issues}} rows: {{known_issue_counts}}).
In short:

- **Sample metadata.** {{ph_scale_summary}} {{temperature_summary}}
  {{pressure_unit_summary}} {{ionic_drivers}} The unfiltered tier has no
  pressure ceiling.
- **Flags**: `membrane_mimetic_sample` is false also where no sample component
  could be read; `high_pressure` is null, not false, where no pressure was
  recorded (filter with `high_pressure IS NOT TRUE`). Membrane-mimetic recall
  is limited to what the deposition states: the gaps the rule cannot see
  (detergents outside the class table, a detergent named only in text or a
  chain name, integral membrane proteins with no mimetic deposited) are
  reviewed overrides; chains reviewed and left unflagged are
  `membrane_mimetic_reviewed_not_listed`, the known false positive 34151_1_1_1
  is `membrane_class_false_positive`. LPS micelles and bilayers count as
  membrane mimetics.
- **CheZOD117.** {{chezod117_conditions}}
- **Labels**: a residue with shifts can be unscored (mask on `mask`, not on
  `k`); a residue with k < 10 is less reproducible; {{few_scored_training}}.
- **Sample and construct.** {{construct_caveats}}
- **Reviewed chains.** {{reviewed_issues}}
- **Residual similarity**: homology is controlled at the searched rule, not
  eliminated. {{remote_homologs}}. {{shared_accessions}}. Resampling unit:
  {{test_components}}. {{residual_similarity}}. {{threshold_fit_retired}}.
- **Shift table**: a few shift keys repeat, the same value under two
  ambiguity codes or a partial-occupancy pair with two values; the
  `trizod_shifts.parquet` metadata (`key_uniqueness`, `counts`) names them.
