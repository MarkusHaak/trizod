# TriZOD Filtering Reference

TriZOD filters peptide shift data entries from the BMRB database using
configurable criteria. The `--filter-defaults` CLI argument sets default values
for all filters at one of four stringency levels: `unfiltered`, `tolerant`,
`moderate`, and `strict`. Individual filters can be overridden with their
respective CLI arguments.

## Filter Descriptions

| Filter                       | Description                                                                                                                                                                                 |
| :--------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| temperature-range            | Minimum and maximum temperature in Kelvin.                                                                                                                                                  |
| ionic-strength-range         | Minimum and maximum ionic strength in Mol.                                                                                                                                                  |
| pH-range                     | Minimum and maximum pH.                                                                                                                                                                     |
| unit-assumptions             | Assume units for Temp., Ionic str. and pH if they are not given and exclude entries instead.                                                                                                |
| unit-corrections             | Correct values for Temp., Ionic str. and pH if units are most likely wrong.                                                                                                                 |
| default-conditions           | Assume standard conditions if pH (7), ionic strength (0.1 M) or temperature (298 K) are missing and exclude entries instead.                                                                |
| peptide-length-range         | Minimum (and optionally maximum) peptide sequence length.                                                                                                                                   |
| min-backbone-shift-types     | Minimum number of different backbone shift types (max 7).                                                                                                                                   |
| min-backbone-shift-positions | Minimum number of positions with at least one backbone shift.                                                                                                                               |
| min-backbone-shift-fraction  | Minimum fraction of positions with at least one backbone shift.                                                                                                                             |
| max-noncanonical-fraction    | Maximum fraction of non-canonical amino acids (X count as arbitrary canonical) in the amino acid sequence.                                                                                  |
| max-x-fraction               | Maximum fraction of X letters (arbitrary canonical amino acid) in the amino acid sequence.                                                                                                  |
| keywords-blacklist           | Exclude entries with any of these keywords as a substring of one of the searched free-text fields, case ignored. See "Keyword search scope" below — the search is field-scoped, not a scan of the whole BMRB file. |
| keyword-search-scope         | Which free-text fields `keywords-blacklist` searches: `sample` (default) or `all`. See "Keyword search scope".                                                                              |
| physical-state-blacklist     | Exclude entries whose `_Entity_assembly.Physical_state` **exactly equals** one of these values (case ignored, whitespace stripped), resolved to the entity assembly of the row being filtered. Four ambiguous values are denied only where the entry independently names a denaturant. See "Physical state". |
| chemical-denaturants         | Exclude entries with any of these chemicals as substrings of `_Sample_component.Mol_common_name`, case ignored, for components that are not the studied polymer itself (no `Entity_ID`). Distinct from `denaturant_evidence`, which is broader and never filters. |
| exp-method-whitelist         | Include only entries with any of these keywords as substring of the experiment subtype, case ignored. `""` is a sentinel meaning "accept a *missing* subtype", never a search term.          |
| exp-method-blacklist         | Exclude entries with any of these keywords as substring of the experiment subtype, case ignored.                                                                                            |
| method-fallback              | What to do about entries with no `_Entry.Experimental_method_subtype` at all: `off`, `reject-solid` or `require-solution`. See "Undeclared experimental method".                            |
| exclude-paramagnetic         | Exclude entries flagged as paramagnetic in the BMRB assembly or entity metadata. Paramagnetic samples cause massive chemical shift perturbations that make Z-score computation meaningless. |
| max-offset                   | Maximum valid offset correction for any random coil chemical shift type, in **sigma units** — the offsets are means of `(observed - POTENCI) / REFINED_WEIGHTS[atom]`, not ppm. Multiply by `REFINED_WEIGHTS[atom]` for ppm (e.g. 3 sigma is ~0.59 ppm for CA, ~0.08 ppm for HA). |
| reject-shift-type-only       | Upon exceeding the maximal offset set by `--max-offset`, exclude only the backbone shifts exceeding the offset instead of the whole entry.                                                  |

## Keyword Search Scope

`keywords-blacklist` does **not** scan the whole NMR-STAR file. It substring-matches
(case-insensitively) against a fixed list of free-text fields, resolved for the
entity assembly of the row being filtered:

| group             | fields                                                                                     |
| :---------------- | :----------------------------------------------------------------------------------------- |
| sample-descriptive | entry title, entry details, assembly name, assembly details, entity name, entity details, and for every referenced sample: sample name, sample details, sample framecode |
| paper-topic       | citation title, citation keywords (`_Citation_keyword`), structure keywords (`_Struct_keywords`) |

Everything else — the shift tables, sample conditions, experiment lists, software
records, comments — is never searched.

The distinction between the two groups matters: the paper-topic fields describe what
the *publication* is about, not what is in the NMR tube. A study titled "protein-protein
interactions of X" that deposits free, unbound X will match a keyword such as
`interacti` on its citation metadata alone.

`--keyword-search-scope` selects the group:

| value              | fields searched                                    |
| :----------------- | :------------------------------------------------- |
| `sample` (default) | sample-descriptive only                            |
| `all`              | sample-descriptive **and** paper-topic              |

All four tiers default to `sample`. Measured justification: all 46 strict hits of the
former `interacti` keyword came from `_Citation_keyword`/`_Struct_keywords`, none from
any sample-descriptive field, and only 41 % had an independent bound-complex signal.
Topic matching also demonstrably drops good data — BMRB 51322 (Aβ(1-42), whose own
`Physical_state` is `intrinsically disordered`) is dropped purely on a citation title
about amyloid fibrils.

### Whole-word keywords

Keywords are substrings by default, which is what makes prefixes such as `denatur` and
`unfold` match `denatured`/`unfolded`. `bound` is the one exception: it is matched as a
whole word, because 22 moderate rows match it *exclusively* through `unbound` or
`boundaries` — "Unbound Med25ACID", "Human Pdx1 Homeodomain in the Unbound State",
"Noncanonical Domain Boundaries" — i.e. exactly the free-state depositions the strict
tier wants to keep. `-bound` compounds such as "membrane-bound" still match.

`interacti` was **removed** from the strict tier and replaced by the phrases
`in complex with` / `complexed with`, matched on sample-descriptive fields only.

## Physical State

`_Entity_assembly.Physical_state` is the depositor's own label for the conformational
state of the studied molecule. It is emitted as the `physical_state` column at **all
four tiers**, resolved to the entity assembly of the row being filtered (an entry-wide
OR over-filters by 10–28 rows per tier), and matched **exactly** — never as a
substring, because `partially denatured` and `not denatured` are both deposited values
that a substring rule on `denatured` would get wrong.

| tier       | additionally denied                                                                                                                  |
| :--------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| unfiltered | — (column emitted, nothing denied)                                                                                                   |
| tolerant   | `misfolded`, `non-native`, `aggregated`, `amyloid`, `amyloid fibril(s)`, `fibril(s)`, `fibrillar`, `molten globule`; **plus** `denatured`, `partially denatured` *only where corroborated* |
| moderate   | + `folding intermediate`, `intermediate`; **plus** `unfolded`, `partially unfolded` *only where corroborated*                        |
| strict     | + `bound`, `micelle-bound`, `SLAS micelle-bound`, `reconstituted`, `reconstituted in DPC`                                            |

`unfolded` / `partially unfolded` are absent from the **tolerant** list entirely — not
even a corroborated hit removes them there. The tolerant tier is meant to be permissive,
and `unfolded` is the label canonical IDPs are most often deposited under.

### Ambiguous states need corroboration

Four of the deposited values are **ambiguous depositor vocabulary**
(`PHYSICAL_STATE_AMBIGUOUS` in `trizod/bmrb/sample_state.py`):

    denatured   partially denatured   unfolded   partially unfolded

Depositors use them both for a chemically denatured sample *and* for a natively
unfolded IDP. BMRB **6968**, whose entity is literally named *"intrinsically disordered
alpha-synuclein"* and whose buffer is KPi/EDTA/NaCl, is deposited as `denatured`.
`tests/test_physical_state.py::test_alpha_synuclein_survives_the_tolerant_deny_list`
pins that it must not be dropped.

So these four are denied **only when the entry independently names a denaturant**.
Everything else in the deny list stands on the tag alone — which is what still removes
BMRB **5158** (apo-myoglobin molten globule, 52 % of its residues scored ordered and a
`train_tier=moderate` chain in v0.3.0), **5119** and **16948**.

The corroborating signal is `has_denaturant_evidence()`, surfaced as the
tier-independent **`denaturant_evidence`** column. It is deliberately not a filter
policy but a statement about what was in the tube:

| | |
| :-- | :-- |
| tokens | `guanidin*`, `gdm*` (prefixes, so `guanidine`/`guanidinium`/`GdmCl` all match) and the whole words `urea`, `TFE`, `trifluoroethanol`, `DMSO`, `SDS`, `Gdn-HCl`, `GdnCl` |
| matching | word-boundary anchored and case-insensitive, so `urea` cannot fire on `urease` |
| searched | entry title & details, assembly name & details, entity name & details, and for every referenced sample its name, details and **all** `_Sample_component.Mol_common_name` values — including components that carry an `Entity_ID`, unlike the `chemical-denaturants` filter |

**What corroboration costs and buys.** Three candidate rules, measured on the 17,843-row
pre-filter frame as `filtered (unique)` — same definitions as in "Measured impact":

| rule                                        | tolerant  | moderate  | strict    |
| :------------------------------------------ | :-------- | :-------- | :-------- |
| deny on the tag alone                        | 214 (65)  | 323 (46)  | 331 (14)  |
| **deny only where corroborated** (shipped)   | 180 (39)  | 221 (12)  | 229 (4)   |
| never deny an ambiguous value                | 41 (20)   | 45 (11)   | 53 (4)    |

The shipped rule therefore costs **39 / 12 / 4** rows that nothing else removes, against
**65 / 46 / 14** for the tag-alone rule. What it still catches is the case the mechanism
was built for: the rows it uniquely removes are dominated by `molten globule` (18
tolerant, 10 moderate), i.e. 5158 and its siblings.

**What it retains, in the released tiers** (`data/interim/scored/<tier>/scores.json`):

| | unfiltered | tolerant | moderate | strict |
| :------------------------------------------ | ---------: | -------: | -------: | -----: |
| records carrying one of the four values      |        238 |       77 |       28 |      5 |
| — of them, with no denaturant anywhere       |         96 |       72 |       28 |      5 |
| records retained that a tag-alone rule would delete | n/a | **24** | **28** | **5** |

The `unfiltered` tier applies no deny list, so its 238 rows are the raw population: 142
of them do name a denaturant and are the cases the rule is meant to catch. The retained
records are the ones the dataset exists for — at tolerant they include α-synuclein
(6968, 16342), the yeast SNAREs Snc1/Sso1 (4286/4287) and the Myc bHLHZip domain
(27704); at moderate they add endosulfine α (15136), NS5A D2 (15225), ACTR/CBP
(15397/15398), the T-cell-receptor ζ chain (15409), γ-synuclein (7244), Tau
(52309/52401), HMGA1 (7279) and α-synuclein again (25227/25228).

### Values that can never be denied

`PHYSICAL_STATE_KEEP` lists the values that no tier may ever deny;
`intrinsically disordered` (224 records in the released unfiltered tier) and
`partially disordered` (50) are the signal the dataset exists to capture. The vocabulary
is depositor free text with a long tail (47 distinct non-null values), so any value that
is unknown to all three vocabularies and occurs on five or more rows is logged as a
warning rather than silently ignored. On the current corpus nothing warns: only
`microcrystalline` (3 records) and `solid state` (1) are unaccounted for, and both
describe the *physical form* of the sample, which `sample_state_evidence` handles
instead.

## Undeclared Experimental Method

24 % of BMRB entries never filled in `_Entry.Experimental_method_subtype`; 4,466 rows /
4,103 entries used to be dropped from the strict tier for that reason alone — 90.8 % of
everything the strict method filter rejects, and 90.7 % of them deposited before 2006.
`--method-fallback` decides those rows on what the deposition actually says about the
sample, via the `sample_state_evidence` column (`solution` / `solid` / `unknown`), which
is derived from `_Sample.Type`, `_Experiment.Sample_state` and the experiment names.

| value                          | effect                                                                                                     |
| :----------------------------- | :---------------------------------------------------------------------------------------------------------- |
| `off` (unfiltered)             | the whitelist alone decides — "unfiltered" stays the raw corpus                                            |
| `reject-solid` (tolerant, moderate) | rows with **no** subtype are dropped when the evidence is `solid`                                     |
| `require-solution` (strict)    | rows with **no** subtype are admitted only on `solution` evidence, **and** any row with `solid` evidence is rejected whatever its declared subtype says |

The fallback never applies to a subtype that is present but uninformative: doing so would
walk `X-RAY DIFFRACTION` and `THEORETICAL` rows into the tiers on `_Sample.Type ==
'solution'`.

Bicelles, micelles and liquid crystals count as **solution** — they are isotropic
solution NMR. The solid verdict is the *refined* veto: solid evidence **and** no
solution-type experiment name, so a real solution structure with a mis-typed
`_Sample.Type` (e.g. BMRB 34067, "Solution structure of the RBM5 OCRE domain") is not
lost. `HSQC`/`HMQC` and the proton-detected `hNCO`/`hCANH` names are not treated as
solution evidence, because fast-MAS solid-state work runs them too.

Over the 17,843-row candidate frame, `sample_state_evidence` resolves to **solution** on
17,449 rows and **solid** on 394. The refined veto is what makes that split conservative:
the naive veto (solid evidence alone, ignoring the experiment names) has 37.5 % measured
precision. With the fallback in place, the strict method criterion rejects 461 rows in
total and only 49 of them are rejected by nothing else — down from 4,466 rows dropped for
an absent tag before.

## Filter Defaults by Stringency Level

| Filter                       | unfiltered  | tolerant                                | moderate                                | strict                                                                                         |
| :--------------------------- | ----------- | --------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------- |
| temperature-range            | [-inf,+inf] | [263,333]                               | [273,323]                               | [273,313]                                                                                      |
| ionic-strength-range         | [0,+inf]    | [0,7]                                   | [0,5]                                   | [0,3]                                                                                          |
| pH-range                     | [-inf,+inf] | [2,12]                                  | [4,10]                                  | [6,8]                                                                                          |
| unit-assumptions             | Yes         | Yes                                     | Yes                                     | No                                                                                             |
| unit-corrections             | Yes         | Yes                                     | No                                      | No                                                                                             |
| default-conditions           | Yes         | Yes                                     | Yes                                     | No                                                                                             |
| peptide-length-range         | [5,+inf]    | [5,+inf]                                | [10,+inf]                               | [15,+inf]                                                                                      |
| min-backbone-shift-types     | 1           | 2                                       | 3                                       | 4                                                                                              |
| min-backbone-shift-positions | 3           | 3                                       | 8                                       | 12                                                                                             |
| min-backbone-shift-fraction  | 0.0         | 0.0                                     | 0.6                                     | 0.8                                                                                            |
| max-noncanonical-fraction    | 1.0         | 0.1                                     | 0.025                                   | 0.0                                                                                            |
| max-x-fraction               | 1.0         | 0.2                                     | 0.05                                    | 0.0                                                                                            |
| keywords-blacklist           | []          | ['denatur']                             | ['denatur', 'unfold', 'misfold']        | ['denatur', 'unfold', 'misfold', 'bound', 'in complex with', 'complexed with']                 |
| keyword-search-scope         | sample      | sample                                  | sample                                  | sample                                                                                         |
| physical-state-blacklist     | []          | tolerant deny list (see "Physical state") | + unfolded / intermediate             | + bound / reconstituted                                                                        |
| ↳ ambiguous values           | —           | corroborated only                       | corroborated only                       | corroborated only                                                                              |
| chemical-denaturants         | []          | ['guanidin', 'GdmCl', 'Gdn-Hcl', 'urea', 'TFE', 'trifluoroethanol'] | ['guanidin', 'GdmCl', 'Gdn-Hcl', 'urea', 'TFE', 'trifluoroethanol', 'DMSO'] | ['guanidin', 'GdmCl', 'Gdn-Hcl', 'urea', 'TFE', 'trifluoroethanol', 'DMSO']                    |
| exp-method-whitelist         | ['', '.']   | ['','solution', 'structures']           | ['','solution', 'structures']           | ['solution', 'structures']                                                                     |
| exp-method-blacklist         | []          | ['solid']                               | ['solid']                               | ['solid']                                                                                      |
| method-fallback              | off         | reject-solid                            | reject-solid                            | require-solution                                                                               |
| exclude-paramagnetic         | No          | Yes                                     | Yes                                     | Yes                                                                                            |
| max-offset                   | +inf        | 3                                       | 3                                       | 2                                                                                              |
| reject-shift-type-only       | Yes         | Yes                                     | No                                      | No                                                                                             |

`TFA` and `Potassium Pyrophosphate` were **removed** from the strict denaturant list:
TFA's 57 percent-unit components have a median of 0.1 % (56/57 ≤ 0.2 %), i.e. it is an
HPLC counterion whose acidification pathway the pH filter already covers, and potassium
pyrophosphate is a buffer. Net data impact of both removals: +1 strict row. `TFE` is a
genuinely new token — `'trifluoroethanol'` does not contain `'tfe'`, so both are needed.
DMSO is applied **ungated**: its concentration boundary is undefined at the only value
that matters (bmr36172 is exactly 5.0 % v/v).

Each filter can be set individually with the respective CLI option, which takes
precedence over `--filter-defaults`.

## Measured Impact

Two different populations are counted below; mixing them up is the commonest way to
misread these numbers.

**Per released record — what actually shipped.** Comparing the `2026-08` release against
`2026-07` (v0.3.0) by record `ID`, over `data/interim/scored/<tier>/scores.json`:

| tier       | v0.3.0 | 2026-08 | removed | added |     net |
| :--------- | -----: | ------: | ------: | ----: | ------: |
| unfiltered | 16,851 |  16,851 |       0 |     0 |       0 |
| tolerant   | 15,446 |  15,193 |     273 |    20 |    −253 |
| moderate   | 11,306 |  11,175 |     198 |    67 |    −131 |
| strict     |  3,514 |   4,113 |     285 |   884 |    +599 |

`unfiltered` is the identical record set, ID for ID: every new policy is empty or `off`
there. Strict grows because `--method-fallback require-solution` re-admits entries whose
method-subtype tag is merely absent, and because `interacti` no longer fires on
paper-topic metadata; it also loses the three genuine solid-state rows (25289 ×2, 27211)
that were in the v0.3.0 strict tier.

**Per pre-filter row — the pipeline's own filter-loss report.** The table below is the
`Pre-computation filtering results` block that `print_filter_losses()` prints, for all
four tiers of the current defaults. The frame is **17,843 candidate rows over 15,885
entries** — one row per `(shift table, entity assembly, entity)`, before scoring.
Each cell is `filtered (unique)`:

- **filtered** — rows in the whole frame that this criterion rejects, *regardless of
  whether anything else also rejects them*. Criteria overlap heavily, so these do not sum.
- **unique** — rows that **only** this criterion rejects. This is the marginal cost of
  the criterion: drop it and the tier gains exactly this many rows.

| criterion                       | unfiltered  | tolerant    | moderate      | strict        |
| :------------------------------ | :---------- | :---------- | :------------ | :------------ |
| method (sub-)type               | 0 (0)       | 453 (351)   | 453 (189)     | 461 (49)      |
| temperature                     | 0 (0)       | 111 (31)    | 243 (78)      | 711 (127)     |
| ionic strength                  | 0 (0)       | 0 (0)       | 30 (20)       | 6,812 (1,807) |
| pH                              | 0 (0)       | 74 (59)     | 1,088 (636)   | 4,813 (908)   |
| peptide length                  | 39 (8)      | 39 (3)      | 309 (33)      | 819 (10)      |
| bb shift types                  | 649 (0)     | 956 (96)    | 2,365 (788)   | 5,014 (446)   |
| bb shift positions              | 968 (299)   | 968 (85)    | 1,452 (0)     | 1,883 (0)     |
| bb shift fraction               | 0 (0)       | 0 (0)       | 2,241 (388)   | 3,439 (609)   |
| non-canonical fraction          | 0 (0)       | 0 (0)       | 0 (0)         | 3 (1)         |
| X fraction                      | 0 (0)       | 543 (228)   | 1,222 (490)   | 1,886 (129)   |
| **physical state**              | —           | 180 (39)    | 221 (12)      | 229 (4)       |
| paramagnetic                    | —           | 173 (140)   | 173 (106)     | 173 (27)      |
| keyword `denatur`               | —           | 93 (15)     | 93 (6)        | 93 (3)        |
| keyword `unfold`                | —           | —           | 96 (40)       | 96 (13)       |
| keyword `misfold`               | —           | —           | 8 (8)         | 8 (0)         |
| keyword `bound` (whole word)    | —           | —           | —             | 1,151 (375)   |
| keyword `in complex with`       | —           | —           | —             | 880 (338)     |
| keyword `complexed with`        | —           | —           | —             | 173 (58)      |
| denaturant `guanidin`           | —           | 19 (2)      | 19 (2)        | 19 (2)        |
| denaturant `GdmCl`              | —           | 3 (0)       | 3 (0)         | 3 (0)         |
| denaturant `Gdn-Hcl`            | —           | 2 (0)       | 2 (0)         | 2 (0)         |
| denaturant `urea`               | —           | 133 (28)    | 133 (19)      | 133 (9)       |
| denaturant `TFE`                | —           | 175 (168)   | 175 (98)      | 175 (12)      |
| denaturant `trifluoroethanol`   | —           | 75 (69)     | 75 (42)       | 75 (9)        |
| denaturant `DMSO`               | —           | —           | 130 (67)      | 130 (14)      |
| missing required values         | 30          | 30          | 30            | 6,895         |
| **rows passing the pre-filter** | **16,867**  | **15,409**  | **12,246**    | **5,442**     |

A dash means the criterion is not configured at that tier. `filtered` is identical across
tiers for the keyword and denaturant rows because the underlying boolean column is
tier-independent — only `unique`, the marginal cost, is tier-specific.

Two reading notes:

- The pre-filter is not the release. Scoring and the offset post-filter reduce these
  further: 16,867 → 16,851 at unfiltered, and 5,442 → 4,113 at strict, where
  `--max-offset 2` is the dominant additional loss.
- `missing required values` jumps to 6,895 at strict only because strict drops the `""`
  whitelist sentinel, so a row with no method subtype is no longer treated as having
  complete metadata. The `--method-fallback require-solution` re-admission is applied
  before this count, which is why the strict method criterion itself rejects only 461.

The single largest strict-only cost is the **ionic-strength window** (1,807 rows removed
by nothing else), followed by **pH** (908) and the **backbone-shift-fraction** minimum
(609) — all three larger than every state, keyword and denaturant policy combined.

## Annotation Columns

The scoring pipeline emits these on **every** row at **every** tier. Only
`physical_state` participates in a filter (via `physical-state-blacklist`); the rest
never remove anything on their own.

| column                  | meaning                                                                                                      |
| :---------------------- | :------------------------------------------------------------------------------------------------------------ |
| `physical_state`        | `_Entity_assembly.Physical_state` for this row's entity assembly, verbatim (lower-cased, stripped), or null   |
| `denaturant_evidence`   | bool — does the entry independently name a chemical denaturant? Gates the four ambiguous physical states; see "Physical state" |
| `sample_state_evidence` | `solution` / `solid` / `unknown` — see "Undeclared experimental method"                                       |
| `membrane_mimetic`      | the matched membrane-mimetic token(s), `;`-joined, or null. A string, not a bool, so an SDS micelle stays distinguishable from DDM solubilisation |

### Membrane mimetics are annotated, never filtered

The vocabulary is `SDS, DPC, dodecyl, LPPG, LMPG, DHPC, DMPC, POPC, bicelle, micelle,
Triton, CHAPS, octyl, maltoside, digitonin, nanodisc`, matched against
`_Sample_component.Mol_common_name` for components that are not the studied polymer.

It is deliberately **not** a filter at any tier, for a measurable reason: the records it
would remove are 93.7 % ordered and contain **zero** disordered chains, against a
baseline disordered fraction of 4.1 %. Filtering on it would be a one-sided deletion of
well-determined ordered examples — it would shift the label distribution toward disorder
rather than improve data quality. Micelles and bicelles are also isotropic solution NMR,
so they are not a solid-state signal either (see `sample_state_evidence`). Users who
want a detergent-free subset can filter on the column themselves; the release does not
make that choice for them.

### Assembly-composition columns

These come from `trizod dataset build` (`trizod/dataset/composition.py`), not from the
scoring pre-filter, and reach the release Parquet. They are annotation except for
`is_bound`, which drives bound-complex removal at build time.

| column                                          | meaning                                                                              |
| :---------------------------------------------- | :------------------------------------------------------------------------------------ |
| `n_entities`, `multi_protein_assembly`          | distinct entities in the assembly, and whether more than one *protein* entity is present |
| `n_entity_assembly_rows`                        | raw count of `_Entity_assembly` rows referencing this entity — no interpretation      |
| `n_copies`                                      | conservative copy count, in priority order: the largest `Magnetic_equivalence_group_code` group when every row has one; else 1 if the rows span more than one `Physical_state` (alternative states, not copies); else 1 if a row name or Details carries a conformer word (cis/trans, major/minor, form A/B, folded/unfolded); else the raw row count |
| `has_conformational_isomer`                     | the deposited `_Entity_assembly.Conformational_isomer` tag is `yes`. Reported, but deliberately **not** a demotion trigger for `n_copies` on its own — genuine oligomers (bmr7091 GroES heptamer, bmr53193 PF4 tetramer) set it too |
| `has_metal`, `has_metal_ion`, `has_metal_cofactor`, `metal_comp_ids` | metal content, derived from `_Chem_comp.Formula` via `_Entity.Nonpolymer_comp_ID`. A bare metal formula is an *ion*; a metal-bearing compound (HEM, FES, SF4, …) is a *cofactor* |
| `has_non_polymer`, `has_other_ligand`, `ligand_comp_ids`, `ligand_names` | non-polymer content and its PDB chem-comp identities                                   |
| `has_nucleic`                                   | DNA, RNA **or** the `polydeoxyribonucleotide/polyribonucleotide hybrid` polymer type   |
| `has_water`                                     | a `water` entity is present. Water is **excluded** from the assembly member set: three protein-only depositions (25640, 34125, 34240) were previously discarded from every tier because a WATER entity made them look bound |
| `is_bound`                                      | the build-time verdict that drives bound-complex removal                               |
