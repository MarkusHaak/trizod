# TriZOD dataset changelog

<!-- Template: `trizod dataset finalize` renders it with this build's counts
into the deposit's CHANGELOG.md, which is also pasted into the Zenodo version
notes. Every placeholder in double braces is filled from the release; type no
counts of the current release into this file. Counts of earlier releases and
of dated audit builds are historical and stated as such. -->

## v{{version}}

Against **v{{previous_version}}** (Zenodo, {{previous_version_date}}), the last
deposited version. Labels, tier membership and the test set all changed, so no
number computed on v{{previous_version}} carries over; regenerate it. The
development builds between the two (2026-08, the 2026-11-11 build of record and
the 2026-11-17 audit rebuild) were never deposited; their changes are folded in
below.

### Test set: {{test_set_label}}

- **Named and frozen.** The TriZOD test set is now {{test_set_name}}, n =
  {{n_test}}. From this release on it may only shrink: a chain that no longer
  qualifies is retired (`split == "test_retired"`, reason in `retired_reason`),
  is not tested on, and stays a leakage target with its homologues.
- **History.** v{{previous_version}} tested on {{n_test_previous}} chains. The
  set was redrawn in August 2026 from the corrected strict pool by the original
  seeded recipe (460 chains), trimmed in November 2026 to the 452 chains that
  still resolved, and is now resolved against the strict tier of this build.
  Retired: {{n_retired}} chains ({{retired_by_reason}}). `not_strict`: the chain
  no longer meets strict (mostly the 2026-11 condition rules);
  `chezod117_overlap`: it meets the leakage criterion against a CheZOD117
  sequence; `within_test_duplicate`: the shorter of two test chains at ≥ 50 %
  identity over ≥ 80 % of either chain.
- **Earlier test sets are not comparable**: UdonPred's published Trizod348
  results, v0.2.0, v{{previous_version}}, and the chains trimmed from the
  460-chain pin (some are now unfiltered-tier training data). The new
  `crosswalk/` lists every earlier set and where its sequences are now; the
  README gives the overlap table and the clean-subset rule.

### Tiers

{{changelog_tier_table}}

### Four condition rules (2026-11)

{{rule_table}}

- Chains a rule removes stay in the release at a looser tier; the flag column
  says why.
- **Re-scored chains.** A deposited temperature of exactly 273 K is read as not
  deposited ({{n_temperature_placeholder}} chains carry
  `temperature_placeholder`). Strict excludes them; the other tiers re-score them
  at the 298 K default, which changes their labels. On the 2026-11-17 audit
  build, 718 of the 720 such chains changed, with a per-chain maximum |ΔG| of
  median 0.127, 90th percentile 0.305 and maximum 0.732, and 1.9 % of their
  residues crossed G = 0.35.
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
  and `chezod1174_overlap`; and `retired_reason`.
- Rows are sorted by `(entry_id, id)` in 500-row groups for selective reads.
- `columns.md` documents every column; the Parquet metadata now records the
  version, licences, code commit, BMRB snapshot, test-set name, cuts and
  calibration.

### New files

- `trizod_shifts.parquet`: every assigned shift on a canonical residue, backbone
  and side chain, raw and re-referenced, with a far-outlier flag.
- `udonpred/`: the UdonPred v2 JSONL view (train/valid/test per tier, `y` = G,
  999 where masked), generated from the Parquet and verified against it.
- `disorder_trizod.fasta`: the SETH/ODiNPred-style view.
- `crosswalk/`: the test-set crosswalk across every earlier test set.
- `test/`: the test FASTAs, including `TriZOD_test_retired.fasta` and
  `CheZOD117_original_sequences.fasta`.
- `datasheet.md`, `columns.md`, `class_balance.tsv`, `bmrb_files.sha256`,
  `MANIFEST.json` (now over every file) and `SHA256SUMS`.

### Fixes

- **Leakage search in both directions.** Stage 2 and the release gate search
  with training and with test chains as queries, under a bit-score floor of 27
  bits that does not depend on database size; the gate reads every training
  and redundant row. Before, a one-directional search missed pairs that only
  the reverse search finds.
- **CheZOD117 as leakage targets**: all 117 original sequences, including the
  two that map to no release chain; test chains overlapping CheZOD117 are
  retired.
- **Membrane-mimetic recall**: entity-linked detergents, sample types and sample
  text are read; reviewed title-only cases are listed in the code repository.
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
  the snapshot of 2025-12-01. Measured between the two: 680 entries added, 10
  withdrawn, and a few deposited shift lists changed in place; BMRB serves only
  the current file, so `bmrb_files.sha256` now records every file read.
- The data are now licensed {{data_license}} (v{{previous_version}}: MIT); the
  code stays {{code_license}}. {{bmrb_acknowledgement}}

### Known issues

The full list, with chain IDs, is in the Supplementary Information of the
TriZOD paper ("Known issues"). In short:

- **Sample metadata**: pH* or pD stored as pH; Celsius values entered as kelvin
  (corrected where detected); ionic strengths that contradict the sample's own
  components (used as deposited); pressures deposited without a unit (read as
  not recorded). The unfiltered tier has no pressure ceiling.
- **Flags**: `membrane_mimetic_sample` is false also where no sample component
  could be read; `high_pressure` is null, not false, where no pressure was
  recorded (filter with `high_pressure IS NOT TRUE`). A few membrane-mimetic
  depositions named only in free text remain undetected.
- **CheZOD117 is not filtered on conditions**: it contains cosolvent, bound and
  273 K placeholder chains.
- **Labels**: a residue with shifts can be unscored (mask on `mask`, not on
  `k`); a residue with k < 10 is less reproducible; some training
  representatives have fewer than 10 scored residues.
- **Sample and construct**: uniformly ²H-labelled protein (no isotope-shift
  correction), residues outside the UniProt-aligned span (tags), a few
  paramagnetic samples; no flag column yet.
- **Residual similarity**: homology is controlled at the 30 %/80 % rule, not
  eliminated; a profile search finds remote homologues of some test chains in
  training, and some test chains share a UniProt accession with a training
  chain (another construct).
- **Shift table**: a few shift keys repeat for partial-occupancy pairs.
