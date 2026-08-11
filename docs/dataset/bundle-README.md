# TriZOD dataset (release 2026-08)

Per-residue **protein-disorder** labels derived from BMRB NMR backbone chemical
shifts, in four **nested** stringency tiers (`strict ⊂ moderate ⊂ tolerant ⊂
unfiltered`). Observed shifts are re-referenced (LACS + POTENCI/AIC) and scored
against POTENCI random-coil predictions; the sequences are redundancy-reduced
and held-out test sets are removed.

## Contents

| tier | training proteins | scored records |
|---|--:|--:|
| unfiltered | 5,907 | 16,851 |
| tolerant | 5,590 | 15,193 |
| moderate | 4,625 | 11,175 |
| strict | 1,998 | 4,113 |

Held-out **test sets** (disjoint from every training set): CheZOD117 (115) and
TriZOD (364).

## Layout

```
trizod-dataset-2026-08/
├── train/<tier>/
│   ├── train_<tier>_best.fasta   one sequence per cluster (canonical)
│   ├── train_<tier>.fasta        mmseqs cluster representatives
│   └── clusters_best.tsv, clusters.tsv   cluster membership (repr, member)
├── scores/<tier>/scores.json     per-residue labels (one JSON object per line)
├── test/                         CheZOD117 + TriZOD test FASTAs
├── trizod_dataset.parquet        the same release as ONE table (63 columns,
│                                 16,851 rows, one row per chain)
├── trizod_sidechain_shifts.parquet   side-chain shifts as deposited (3,458,851)
└── MANIFEST.json                 file list, sizes, SHA-256, counts
```

`MANIFEST.json` covers the FASTA/JSON layer; the two Parquet files are built
from it by `scripts/build_parquet_dataset.py` and carry the same records.

## `scores.json` fields (one record per line; `ID` matches the FASTA header)

| field | meaning |
|---|---|
| `ID` | `entryID_stID_entityAssemID_entityID` |
| `seq` | amino-acid sequence (1-letter) |
| `gscores` | per-residue **G-score** (0–1), aligned 1:1 to `seq` (`null` = no data) |
| `zscores` | per-residue CheZOD Z-score (same alignment) |
| `k` | number of weighted secondary shifts behind each residue (`0` = no shift) |
| `off_<atom>`, `lacs_off_<atom>` | per-atom referencing offsets (C, CA, CB, H, HA, HB, N). **Different units**: `lacs_off_<atom>` is ppm; `off_<atom>` is in sigma units — mean of `(observed − POTENCI) / REFINED_WEIGHTS[atom]`. Multiply by `REFINED_WEIGHTS[atom]` for ppm; never add the two |
| `pH`, `temperature`, `ionic_strength`, `citation` | sample metadata |
| `physical_state`, `denaturant_evidence`, `sample_state_evidence`, `membrane_mimetic` | sample-state annotation (see below) |

### Sample-state and composition annotation

New in this release, emitted for **every** record at **every** tier. These
describe the deposition; only `physical_state` participates in any filter.

| field | meaning |
|---|---|
| `physical_state` | `_Entity_assembly.Physical_state` for this row's entity assembly, verbatim and lower-cased (`native`, `intrinsically disordered`, `denatured`, `molten globule`, …), or null |
| `denaturant_evidence` | bool — does the entry independently name a chemical denaturant (guanidin*, urea, TFE, trifluoroethanol, DMSO, SDS, Gdn-HCl) anywhere in its sample components or sample-descriptive text? |
| `sample_state_evidence` | `solution` / `solid` / `unknown`, derived from `_Sample.Type`, `_Experiment.Sample_state` and the experiment names |
| `membrane_mimetic` | matched membrane-mimetic token(s), `;`-joined, or null. **Annotation only — never filtered** |

`trizod_dataset.parquet` additionally carries the assembly-composition columns
`n_entities`, `n_entity_assembly_rows`, `n_copies`, `has_conformational_isomer`,
`multi_protein_assembly`, `has_non_polymer`, `has_nucleic`, `has_metal`,
`has_metal_ion`, `has_metal_cofactor`, `has_water`, `has_other_ligand`,
`metal_comp_ids`, `ligand_comp_ids`, `ligand_names`, `is_bound`, and the split
columns `split` / `train_tier` / `pool_tier` / `cluster_repr` / `label_tier`.

## Side-chain shifts (`trizod_sidechain_shifts.parquet`)

3,458,851 side-chain chemical shifts (2,449,316 ¹H, 946,575 ¹³C, 62,960 ¹⁵N)
over 12,643 of the 16,851 chains — the values the backbone scoring path
discards. Columns: `id`, `seq_id`, `comp_id`, `atom_id`, `atom_type`, `val`,
`val_err`, `ambiguity_code`. Joins on `id` with the main table.

**Values are AS DEPOSITED** — no re-referencing, no offset correction. The
backbone re-referencing offsets are not transferable to side-chain nuclei, so
they are deliberately not applied here.

## Use for ML

- **Inputs**: `train/<tier>/train_<tier>_best.fasta`.
- **Target**: the **G-score** (0–1, independent of shift count) from the matching
  `scores/<tier>/scores.json`, joined by `ID`. Mask residues where `k == 0`
  (G-score `null`).
- **Tier**: `moderate` balances size and quality; `strict` is cleanest;
  `unfiltered` is largest. Tiers are nested.
- **Cluster-aware CV**: group by `clusters_best.tsv`.
- Train on any tier and evaluate on **CheZOD117** and/or the **TriZOD** test set
  with no train/test leakage.

## Changelog — release 2026-08

This release follows an external review of the filtering code by **Sandro Kuppel
(Helmholtz Munich)**, with input from **Reid Alderson** and **Iva Pritisanac**.
Six independent claims were audited; all six were at least partly confirmed, and
the audit surfaced several further defects. Every filter-behaviour item below
changes which records are in which tier, so **tier membership is not comparable
to release 2026-07 (v0.3.0) record-for-record**. The held-out test sets are
deliberately kept comparable (see below).

### 1. Filter-behaviour corrections (bugs)

| # | what was wrong | effect |
|---|---|---|
| a | `_Citation_keyword` and `_Struct_keywords` were appended one **character at a time** (`fields.extend(el)` on a string), so no multi-character keyword could ever match them (issue #23) | the keyword blacklist silently never saw either field |
| b | `""` in the experiment-method whitelist was joined into the regex alternation, where an empty alternative matches **every** string | the tolerant and moderate method whitelists were no-ops; `""` is now a sentinel meaning "accept a *missing* subtype" and never reaches the matcher |
| c | `has_metal` tested `_Entity.Type.startswith("metal")`; **no** BMRB entity record is typed `metal` | the flag was dead code, constant `False`. Metals are now derived from `_Chem_comp.Formula` via `_Entity.Nonpolymer_comp_ID` |
| d | `has_nucleic` matched only two polymer types and missed `polydeoxyribonucleotide/polyribonucleotide hybrid` | 16 entity records in 14 entries misclassified |
| e | a `water` entity counted as an assembly member, making three protein-only depositions look bound (25640 cytochrome c, 34125 cytotoxin-1, 34240 engrailed homeodomain) | those chains are now eligible again |
| f | `--include-shifts` assigned `shifts` only inside the `--no-shift-averaging` branch | the flag emitted nothing |

### 2. Filter-policy changes

- **Keyword search is field-scoped** (`--keyword-search-scope`, default
  `sample`). The blacklist now searches only sample-descriptive fields — entry
  title/details, assembly name/details, entity name/details, sample
  name/details/framecode. The paper-topic fields (`citation_title`,
  `_Citation_keyword`, `_Struct_keywords`) describe what the *publication* is
  about, not what is in the tube, and are no longer searched by default.
- **`interacti` removed** from the strict blacklist and replaced by the phrases
  `in complex with` / `complexed with`. All 46 of its strict hits came from
  paper-topic fields ("protein–protein interaction" on free monomers).
- **`bound` is now matched as a whole word**, so "Unbound …" and "… Domain
  Boundaries" no longer fire. `-bound` compounds still match.
- **Denaturant list corrected**: `TFE` and `trifluoroethanol` added at tolerant
  and above, `DMSO` at moderate and above; **`TFA`** (an HPLC counterion, median
  0.1 % of the sample) and **`Potassium Pyrophosphate`** (a buffer) removed as
  non-denaturants.
- **Undeclared experimental method** (`--method-fallback`). About 24 % of BMRB
  entries never filled in `_Entry.Experimental_method_subtype`; they used to be
  dropped from the strict tier for that alone. They are now judged on
  `sample_state_evidence`: `require-solution` at strict, `reject-solid` at
  tolerant/moderate, `off` at unfiltered. A refined solid veto (solid evidence
  **and** no solution-type experiment name) additionally removes genuine
  solid-state depositions — 25289 (×2) and 27211 — that were in the released
  strict tier.
- **`_Entity_assembly.Physical_state` is now used.** Non-native states
  (`misfolded`, `aggregated`, `amyloid*`, `fibril*`, `molten globule`, …) are
  denied from tolerant upwards; `bound` / `reconstituted` states additionally at
  strict. This is what removes 5158 (apo-myoglobin molten globule), 5119 and
  16948, which no filter had ever touched.
  `denatured` / `partially denatured` / `unfolded` / `partially unfolded` are
  **ambiguous depositor vocabulary** — used both for a chemically denatured
  sample and for a natively unfolded IDP — and are denied **only when the entry
  independently names a denaturant**. α-synuclein (6968), Tau and the other
  natively unfolded depositions therefore stay in.
- **Membrane mimetics are annotated, never filtered**, at any tier: the records
  they would remove are 93.7 % ordered and contain zero disordered chains, so
  filtering on them would be a one-sided removal of well-determined ordered
  examples.

### 3. New metadata columns

Additive — no existing column changes meaning. `physical_state`,
`denaturant_evidence`, `sample_state_evidence`, `membrane_mimetic`,
`n_entity_assembly_rows`, `n_copies`, `has_conformational_isomer`, `has_metal`,
`has_metal_ion`, `has_metal_cofactor`, `has_water`, `has_other_ligand`,
`metal_comp_ids`, `ligand_comp_ids`, `ligand_names`, `label_tier`. The release
Parquet grows from 42 to **63** columns.

`n_copies` is a conservative copy count: the largest
`Magnetic_equivalence_group_code` group when every row carries one; else 1 if the
rows span more than one `Physical_state`; else 1 if a row name or Details carries
a conformer word (cis/trans, major/minor, form A/B, folded/unfolded); else the
raw row count. It is **not** a stoichiometry guarantee in either direction:
under-annotated homodimers declare a single row, and alternative-conformer
depositions inflate the raw `n_entity_assembly_rows`.

### 4. New companion file

`trizod_sidechain_shifts.parquet` — 3,458,851 side-chain shifts that previous
releases discarded entirely. As deposited, unreferenced (see above).

### 5. What moved

| tier | 2026-07 (v0.3.0) | 2026-08 | removed | added |
|---|--:|--:|--:|--:|
| unfiltered | 16,851 | 16,851 | 0 | 0 |
| tolerant | 15,446 | 15,193 | 273 | 20 |
| moderate | 11,306 | 11,175 | 198 | 67 |
| strict | 3,514 | 4,113 | 285 | 884 |

The unfiltered tier is the **identical record set**, ID for ID. Training
representatives move with the tiers: 5,907 / 5,590 / 4,625 / 1,998
(unfiltered / tolerant / moderate / strict).

### 6. The test sets stay comparable

The TriZOD test set is a **pin**, not a fresh draw, and is re-resolved against
the tolerant pool at every rebuild. Of the 365 pinned chains:

- **364 resolve.** The one drop is `19342_1_1_1` ("Transmembrane-cytosolic part
  of Trop2"), which lists a sample component `TFE` at **70 %**
  (`_Sample.Solvent_system` reads `30%H2O/70% trifluoroethanol`) and is now
  caught by the new TFE denaturant token. At that concentration the shifts
  report a solvent-forced helical conformation rather than the aqueous state,
  so its removal is a correction, not collateral.
- `50998_1_1_1` was **ID-substituted** to `5599_1_1_1` — a byte-identical
  199-residue sequence under a lower entry number. ID-based joins against v0.3.0
  must therefore go through `TriZOD_test_set_labels.tsv`
  (`test_id`, `pinned_id`, `substituted`, `label_tier`, `length`), which
  `trizod dataset test-set` writes next to the test FASTA.
- **17 retained chains no longer meet strict criteria** (10 `moderate`, 7
  `tolerant`); the new `label_tier` column records the strictest tier whose
  criteria each test chain still passes. They were kept deliberately: the test
  set is a fixed benchmark, and shrinking it on every filter change would make
  results incomparable across releases.

CheZOD117 is unchanged at 115 BMRB-mapped sequences. The release-time leakage
gate passes with **0 shared IDs and 0 exact-sequence matches** between any
training set and either test set.

### 7. Documentation corrections

`off_<atom>` is in **sigma units** — the mean of
`(observed − POTENCI) / REFINED_WEIGHTS[atom]` — and was previously documented
as ppm in the v0.3.0 README and datasheet. `lacs_off_<atom>` **is** ppm. The two
must never be added. `--max-offset` (3 / 3 / 2) is likewise a sigma threshold.

### 8. Known carry-over

v0.3.0's tier counts already reflected a superseded scoring run (issue #20).
This release is the first built end to end on the corrected scores *and* the
corrected filters.

Derived from BMRB (https://bmrb.io/). License: MIT.
