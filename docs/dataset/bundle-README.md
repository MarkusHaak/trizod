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
├── trizod_dataset.parquet        the same release as ONE table (70 columns,
│                                 16,851 rows, one row per chain)
├── trizod_shifts.parquet         every assigned shift on a canonical residue,
│                                 backbone AND side chain (11,839,037 rows),
│                                 raw and re-referenced
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
| `off_<atom>_sigma`, `lacs_off_<atom>_ppm`, `total_off_<atom>_ppm` | per-atom referencing offsets (C, CA, CB, H, HA, HB, N) in **three different units** — see §7. To reproduce the shift TriZOD scored, subtract `total_off_<atom>_ppm` from the deposited shift; never add the other two together |
| `pH`, `temperature`, `ionic_strength`, `citation` | sample metadata |
| `physical_state`, `cosolvent_evidence`, `sample_state_evidence`, `membrane_mimetic` | sample-state annotation (see below) |

### Sample-state and composition annotation

New in this release, emitted for **every** record at **every** tier. These
describe the deposition; only `physical_state` participates in any filter.

| field | meaning |
|---|---|
| `physical_state` | `_Entity_assembly.Physical_state` for this row's entity assembly, verbatim and lower-cased (`native`, `intrinsically disordered`, `denatured`, `molten globule`, …), or null |
| `cosolvent_evidence` | bool — does the entry independently name a perturbing cosolvent (guanidin*, gdm*, hexafluoroisopropanol*, hexafluoro-2-propanol*, urea, TFE, trifluoroethanol, HFIP, DMSO, SDS, Gdn-HCl, GdnCl) anywhere in its sample components or sample-descriptive text? Renamed from `denaturant_evidence` — only urea and the guanidinium salts denature, while TFE/HFIP/DMSO are helix inducers, so the family is named for what it shares (the sample is not aqueous buffer) rather than for half of it |
| `sample_state_evidence` | `solution` / `solid` / `unknown`, derived from `_Sample.Type`, `_Experiment.Sample_state` and the experiment names |
| `membrane_mimetic` | matched membrane-mimetic token(s), `;`-joined, or null. **Annotation only — never filtered** |

`trizod_dataset.parquet` additionally carries the assembly-composition columns
`n_entities`, `n_entity_assembly_rows`, `n_copies`, `has_conformational_isomer`,
`multi_protein_assembly`, `has_non_polymer`, `has_nucleic`, `has_metal`,
`has_metal_ion`, `has_metal_cofactor`, `has_water`, `has_other_ligand`,
`metal_comp_ids`, `ligand_comp_ids`, `ligand_names`, `is_bound`, and the split
columns `split` / `train_tier` / `pool_tier` / `cluster_repr` / `label_tier`.

## Chemical shifts (`trizod_shifts.parquet`)

Every assigned chemical shift **on a canonical residue** — backbone *and* side
chain — of every released chain: **11,839,037** rows over all 16,851 chains
(12,643 of which carry side-chain assignments), one row per
`(chain, residue, atom)`, joining on `id` with the main table.

| | ¹H | ¹³C | ¹⁵N | total |
|---|--:|--:|--:|--:|
| backbone (`is_backbone = true`) | 3,789,493 | 3,263,875 | 1,326,818 | 8,380,186 |
| side chain | 2,449,316 | 946,575 | 62,960 | 3,458,851 |

Columns: `id`, `seq_id`, `comp_id`, `atom_id`, `atom_type`, `val_ppm`,
`val_err_ppm`, `ambiguity_code`, `val_corrected_ppm`, `offset_applied_ppm`,
`offset_source`, `is_backbone`. Every numeric field declares its unit in the
Parquet **field** metadata and the table declares the correction formula, so a
consumer reading the schema cannot get the units wrong.

`seq_id` is a **1-based index into the main table's `sequence`**:
`sequence[seq_id-1]` is the residue named by `comp_id`. Values are unfiltered —
the quality cuts the scoring path applies (`Val_err > 1.3 ppm`, ambiguity codes
outside {1,2,null}, degenerate-partner averaging) are **not** applied here;
`val_err_ppm` and `ambiguity_code` ship as data instead.

**What is not in it.** Shifts assigned to a **non-canonical residue or group**
are dropped, because `seq_id` would not then index the released `sequence`. That
is 16,502 values — **0.139 %** of the 11,855,542 deposited values — on 290
distinct `Comp_ID`s across 1,050 of the 16,851 chains: post-translational
modifications (HYP, SEP, TPO, PTR, TYS, ALY, MLY, M3L), non-standard residues
(ORN, AIB, ABA, NLE, DPR, PCA, MLE, DAL) and terminal or lipid groups (ACE,
NH2, MYR). Read those from the BMRB entry directly if you need them.

### Re-referencing: three regimes, and `offset_source` tells you which

`val_ppm` is **always exactly as deposited** — never re-referenced.
`val_corrected_ppm` is `val_ppm − offset_applied_ppm`, and is **NULL, never a
silent copy of `val_ppm`**, wherever no offset demonstrably transfers.

| regime | rows | what is subtracted | `offset_source` | `val_corrected_ppm` |
|---|--:|---|---|---|
| **backbone** | 8,380,186 | the full offset the scorer used, `total_off_<atom>_ppm` | `lacs+potenci` 924,335 · `lacs_only` 5,065,786 · `potenci_only` 306,681 · `none` 2,083,384 | present on every row |
| **side-chain ¹³C** | 946,575 | **LACS only** — `val_ppm − lacs_off_CA_ppm` | `lacs_only` 899,443 · `none` 46,205 · `not_transferable` 927 | present on 945,648 rows |
| **side-chain ¹H and ¹⁵N** | 2,512,276 | **nothing — the value is left RAW** | `not_transferable` | NULL |

`offset_applied_ppm` is the per-row materialisation of the per-chain offsets in
`trizod_dataset.parquet`: for a backbone row it equals that chain's
`total_off_<atom>_ppm` = `lacs_off_<atom>_ppm + off_<atom>_sigma ×
REFINED_WEIGHTS[atom]` (§7), with degenerate partners — HB2/HB3, ALA HB1, GLY
HA2/HA3 — carrying the offset of the slot the scorer averages them into (HB,
HA); for a side-chain ¹³C row it equals `lacs_off_CA_ppm`. It is **already in
ppm**, so `val_corrected_ppm = val_ppm − offset_applied_ppm` needs no conversion.

`none` is a *measured* zero, not a missing value: the estimators ran and
returned 0.0 (LACS does not fire below 20 backbone CA observations), so
`offset_applied_ppm` is `0.0` and `val_corrected_ppm == val_ppm`.
`not_transferable` means the chain **has** an offset that does not transfer to
this row; both value columns are NULL there.

**Why side-chain ¹³C gets the LACS term and not the POTENCI term.** Which offset
transfers was measured on this corpus as a regression slope
`β = cov(group-centred side-chain deviation, offset) / var(offset)`; subtracting
an offset lowers MSE iff `β > 0.5`. The LACS carbon offset transfers at
β = 0.785 [0.767, 0.806] and applying it lowers side-chain ¹³C MSE by ×0.79
(RMSE 0.919 → 0.818 ppm). The POTENCI residual (`off_<atom>_sigma`) does not: it
transfers at β = 0.093 (CA) / 0.081 (CB) / 0.050 (C), because it is a residual
bias in the backbone-vs-POTENCI comparison, not a spectrometer referencing
error. So side-chain carbons carry `lacs_off_CA_ppm` and nothing else.

**Why ¹H and ¹⁵N are left raw.** β = 0.079 [0.057, 0.102] for side-chain ¹H and
0.366 [0.330, 0.407] for side-chain ¹⁵N — both below the 0.5 break-even — and
applying the backbone offset **inflates** the error (MSE ×1.131 and ×1.075).
This is *not* "there was nothing to correct": a real per-chain side-chain ¹H
referencing constant does exist (split-half reliability 0.812), but the backbone
amide offset explains only 1.3 % of its variance. The ¹⁵N verdict is
estimator-independent — PANAV, an independent offset estimator, gives β = 0.394,
and an instrumental-variable disattenuation gives 0.499 [0.430, 0.555], at the
break-even rather than above it.

**The ¹³C gate.** Side-chain ¹³C is additionally withheld
(`not_transferable`, both value columns NULL) on the **13 chains** with
|`lacs_off_CA_ppm`| > 5 ppm — 9 of which carry side-chain carbons, 927 rows.
Those are genuine gross deposition errors that LACS detects correctly, but only
5 of the 9 demonstrably transfer to the side chain and the difference cannot be
adjudicated one chain at a time.

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
- **Cosolvent list corrected** (the filter was renamed from
  `chemical-denaturants` to `perturbing-cosolvents`; see below): `TFE`,
  `trifluoroethanol`, `trifluoro ethanol`, `HFIP`/`hexafluoroisopropanol`/
  `hexafluoro-2-propanol` and `DMSO`/`dimethyl sulfoxide`/`dimethylsulfoxide`
  added at tolerant and above; **`TFA`** (an HPLC counterion, median 0.1 % of
  the sample) and **`Potassium Pyrophosphate`** (a buffer) removed. The rename
  is not cosmetic: urea and GdmCl inflate apparent **disorder**, while TFE,
  HFIP and DMSO induce helix and so inflate apparent **order** — the worse
  error for a disorder dataset — and what all five share is that the sample is
  no longer aqueous buffer, which is where POTENCI and the LACS reference
  tables are parameterised. Stabilising osmolytes (TMAO, glycerol) are
  deliberately not excluded.
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
  independently names a perturbing cosolvent**. α-synuclein (6968), Tau and
  the other natively unfolded depositions therefore stay in.
- **Membrane mimetics are annotated, never filtered**, at any tier: the records
  they would remove are 93.7 % ordered and contain zero disordered chains, so
  filtering on them would be a one-sided removal of well-determined ordered
  examples.

### 3. New metadata columns

Additive apart from the offset rename of §7 — no retained column changes
meaning. The 28 new columns are `physical_state`, `cosolvent_evidence`,
`sample_state_evidence`, `membrane_mimetic`, `n_entities`,
`multi_protein_assembly`, `n_entity_assembly_rows`, `n_copies`,
`has_conformational_isomer`, `has_non_polymer`, `has_nucleic`, `has_metal`,
`has_metal_ion`, `has_metal_cofactor`, `has_water`, `has_other_ligand`,
`metal_comp_ids`, `ligand_comp_ids`, `ligand_names`, `is_bound`, `label_tier`,
and the seven `total_off_<atom>_ppm` columns of §7. The release Parquet grows
from 42 to **70** columns.

`n_copies` is a conservative copy count: the largest
`Magnetic_equivalence_group_code` group when every row carries one; else 1 if the
rows span more than one `Physical_state`; else 1 if a row name or Details carries
a conformer word (cis/trans, major/minor, form A/B, folded/unfolded); else the
raw row count. It is **not** a stoichiometry guarantee in either direction:
under-annotated homodimers declare a single row, and alternative-conformer
depositions inflate the raw `n_entity_assembly_rows`.

### 4. New companion file

`trizod_shifts.parquet` — **11,839,037** chemical shifts, backbone and side
chain, raw (`val_ppm`) beside the re-referenced value (`val_corrected_ppm`)
wherever an offset demonstrably transfers. 3,458,851 of them are side-chain
values that previous releases discarded entirely. See the section above for the
three re-referencing regimes and for what the table does not cover.

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
  caught by the new TFE cosolvent token. At that concentration the shifts
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

### 7. Offset columns renamed, and a ppm total added

v0.3.0 shipped `off_<atom>` and `lacs_off_<atom>`, which are **not in the same
unit** — and nothing in the names said so. They are renamed, and a derived ppm
total is added:

| column | unit | meaning |
|---|---|---|
| `off_<atom>_sigma` (was `off_<atom>`) | multiples of the per-atom POTENCI RMSD `REFINED_WEIGHTS[atom]` — **not ppm** | mean of `(observed − POTENCI) / REFINED_WEIGHTS[atom]` after LACS. `--max-offset` (3 / 3 / 2) is compared against **this**, so it stays in sigma |
| `lacs_off_<atom>_ppm` (was `lacs_off_<atom>`) | ppm | LACS offset, subtracted straight off the deposited shift |
| `total_off_<atom>_ppm` (**new**) | ppm | `lacs_off_<atom>_ppm + off_<atom>_sigma × REFINED_WEIGHTS[atom]` — the single number to subtract from a deposited shift to reproduce the shift TriZOD scored |

Adding the first two together is a unit error worth up to **21.2 ppm**; that is
exactly why the third column exists. `off_<atom>` was documented as ppm in the
v0.3.0 README and datasheet; it never was. Migration is mechanical:
`off_<a>` → `off_<a>_sigma`, `lacs_off_<a>` → `lacs_off_<a>_ppm`.

### 8. Known carry-over

v0.3.0's tier counts already reflected a superseded scoring run (issue #20).
This release is the first built end to end on the corrected scores *and* the
corrected filters.

Derived from BMRB (https://bmrb.io/). License: MIT.
