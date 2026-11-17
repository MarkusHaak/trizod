# TriZOD dataset — construction note

A single end-to-end reference for how the TriZOD disorder dataset is built:
dataset release **v0.4.0**, test set **TriZOD-test v1.0**, data licensed
**CC BY 4.0**, code **MIT** (BMRB acknowledged, no BMRB licence claimed). The
figure
`publication/manuscript/Figures/2026-06-22_TriZOD_redundancy_reduction.drawio`
visualises this pipeline; its caption text lives in
[`figure-caption.md`](figure-caption.md).

Reproduce everything with `scripts/build_dataset.sh`.

> The per-stage numbers on this page are a **worked example from the
> `2026-08` development build** (under `data/interim/build/`), kept to show the
> size of each step; they are not the counts of any deposited release. The
> counts of a release are rendered from its own files by `trizod dataset
> finalize` into the deposit's README, datasheet, CHANGELOG and MANIFEST
> ([`bundle-README.md`](bundle-README.md), [`datasheet.md`](datasheet.md) and
> [`CHANGELOG.md`](CHANGELOG.md) are those templates).

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
| **scored records** | 4,108 | 11,159 | 15,080 | 16,851 |

→ `data/interim/scored/<tier>/scores.json`. (`trizod dataset build` reads these.)

Compared record-for-record against the `2026-07` (v0.3.0) release, by `ID`:

| | strict | moderate | tolerant | unfiltered |
|---|--:|--:|--:|--:|
| v0.3.0 | 3,271 | 10,941 | 15,440 | 16,851 |
| in both | 2,891 | 10,624 | 15,050 | 16,851 |
| removed | 380 | 317 | 390 | 0 |
| added | 1,217 | 535 | 30 | 0 |
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
3. Randomly sample **25 %** of the clusters without a CheZOD sequence.
4. Recluster those members at **50 % identity / 80 % coverage** → the TriZOD
   test-set representatives.

**The test set is pinned**, so that recipe only runs on a deliberate redraw;
what the pin resolves to in a release is named **TriZOD-test v1.0** (its size
and retired count are in the release's `test_set` metadata). The
pin holds **460** sequences at `trizod/dataset/pinned/TriZOD_test_set.fasta`
(redrawn 2026-08-11, seed 42, sample fraction 0.25, from the final strict pool);
`trizod dataset test-set` resolves each pinned sequence against the **current
strict pool** and writes `TriZOD_test_set.fasta` plus
`TriZOD_test_set_labels.tsv` (`test_id`, `pinned_id`, `substituted`,
`label_tier`, `length`, `status`, `reason`). The TriZOD test set is defined as
strict-tier sequences (since 2026-11; from 2026-08 it was resolved against
tolerant to keep the benchmark whole). A pinned sequence that no longer meets
strict after a rule change or rescore is **retired**: it leaves the test set but
is written to `TriZOD_test_retired.fasta` (header `label_tier=… pinned_id=…
reason=not_strict`, `status = retired` in the labels table), which the
redundancy step adds to its leakage targets. Retired chains and their
homologues therefore stay out of every training set exactly as with the full
pin; only the test set shrinks. The release stages the file under `test/` (the
leakage gate checks it), the Parquet gives such chains `split = test_retired`,
and the deploy FASTA puts them in neither set. The pin is consequently **not**
trimmed when chains leave strict — it is what defines the retired set; only a
pinned sequence absent from the tolerant pool too is dropped outright. Test
metrics are comparable across releases only on the chains both test sets
share.

Two more reasons retire a chain that is still strict (release audit
2026-11-17), checked in this order after the resolution, both with mmseqs:

| `reason` | when |
|---|---|
| `not_strict` | the pinned sequence is no longer in the strict pool (above) |
| `chezod117_overlap` | the chain meets the leakage criterion (§4: ≥ 30 % identity over ≥ 80 % of either chain, ≥ 27 bits, either search direction) against a CheZOD117 sequence — a released CheZOD117 chain or one of the 117 original sequences (§4). `19481_1_1_1` contains CheZOD117 `19482` as an exact 91-residue substring, so the two test sets overlapped |
| `within_test_duplicate` | the remaining test chains are searched against each other at ≥ 50 % identity over ≥ 80 % of **either** chain (the coverage the release uses everywhere else; until 2026-11-17 within-test redundancy was controlled only at both-chain coverage by the 50 %/80 % recluster of the draw). Of every such pair the shorter chain is retired — the contained one for an exact substring (`6439_1_1_1` inside `6440_1_1_1`), the lower entry ID kept on a tie — greedily from the longest chain down, so no two remaining test chains form a pair and no chain is retired for a partner that is itself retired |
| `shared_construct` | a reviewed decision, not a rule (re-audit 2026-11-17b, E26): the chain meets no leakage criterion on its deposited sequence but is a near-copy of training chains once its expression tag or fusion partner is set aside, so its labels are largely training labels. The chains, their training partners and the evidence are listed in `trizod/dataset/data/test_retirements.tsv` |

Every retired chain, whatever the reason, stays a leakage target; the test set
only shrinks. The per-chain hits are recorded in `build_test_set_summary.json`
(`chezod117_overlaps`, `within_test_duplicates`, `retired_by_reason`), and the
Parquet carries the reason as `retired_reason`. The test set is therefore
**disjoint from CheZOD117 at the leakage criterion** and CheZOD-disjoint at
cluster level from the draw; it is not "CheZOD-free" — 88 test chains overlap
CheZOD1325 (flag `chezod1325_overlap`, §6).

For the `2026-08` build (`build_test_set_summary.json`):

| | |
|---|--:|
| pinned | 460 |
| **resolved** | **460** |
| dropped | 0 |
| ID-substituted | 0 |
| `label_tier` = strict / moderate / tolerant | 460 / 0 / 0 |

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

`trizod dataset redundancy`, against the held-out test sets **CheZOD117 (115
release chains) + the 117 original CheZOD117 sequences + the TriZOD test set +
the retired TriZOD test chains** — CheZOD1325 is *not* a training-leakage
target. The original CheZOD117 sequences are the ones SETH published
(`data/external/chezod/seth_github/CheZOD117_test_set_sequences.fasta`); none is
identical to its release chain, and two ("hmbdi" and "6078") map to no release
chain at all, so until 2026-11-17 a training chain could hold one of them
(`34812_1_1_1` is "hmbdi" with one extra N-terminal Gly; release audit E23).
Common mmseqs options throughout:
`--alignment-mode 3 -s 7.5 --comp-bias-corr 0 --mask 0`, with `--cov-mode 0`
(80 % coverage of **both** sequences) everywhere except the stage-2 search.

**Leakage criterion.** A training chain leaks when an alignment to a test chain
reaches **≥ 30 % identity and ≥ 27 bits over ≥ 80 %** of the chain(s) named by
the **leakage rule** (`--leakage-rule`, decision D2), found by **either search
direction**:

| rule | ≥ 80 % of | also catches |
|---|---|---|
| `test` | the test chain | a longer construct that contains a test protein; a shorter chain whose gapped alignment spans the test chain (e.g. the test protein minus an internal loop) |
| `either` (default since 2026-11) | the test chain **or** the training chain (the shorter one) | everything `test` catches, plus a training chain contained in a longer test chain (e.g. test `51677_1_1_1`, 371 aa, which contains representatives `51674`–`51676`) |

Coverage is the mmseqs aligned span (gaps included). The search is the same
under both rules; only the coverage test on its hits differs. The redundancy
step records the rule in `mmseqs/leakage_rule.json`, and the package step checks
the same rule.

**The significance condition (since 2026-11-17).** Identity and coverage alone
would count chance alignments of short chains (a 20–30 aa pair at 30 % identity
is expected by chance), so an alignment must also be significant. Until
2026-11-17 that was the `easy-search` default E ≤ 10⁻³ with the test chains as
the target database — but an E-value scales with the query length and the size
of the target database, so the same pair passed or failed depending on which
side was the query and how many chains the target set held. Searching with the
test chains as queries found three more pairs at 30 %/80 % (`7350_1_1_1` ~
`52238_1_1_1`, `30761_1_1_1` ~ `26610_1_1_1`, `34286_1_1_1` ~ `52584_1_1_1`, all at
30–39 % identity and 32–33 bits), and searching per target set found more
(`34481_1_1_1` ~ `15008_1_1_1`, `15844_1_1_1` ~ `17736_1_1_1`; release audit E22).
The condition is now a **bit-score floor of 27 bits**, a property of the
alignment alone. *Why a bit score and why 27:* mmseqs 18 has no option to fix
the database size of its E-values, and a bit score makes forward and reverse
searches, a per-tier search and the release gate agree on every pair whatever
their database. 27 bits is the most permissive score the former cut admitted in
the 2026-11-17 build (its shortest queries against the ~570-chain test
database), so every pair it removed is still removed, while the dependence on
query length and database size is gone; for two 100-residue chains a chance
local alignment of 27 bits has a pairwise E-value of about 10⁻⁴, and the
identity and coverage thresholds carry the rest. mmseqs runs with `-e 100` only
so that it reports every alignment above the floor; the search fails if the cap
could have hidden one (`check_evalue_headroom`). **Both directions are
searched** — training chains as queries against the test chains and test chains
as queries against the training chains — and a pair counts when either meets
the criterion: the alignment score is symmetric but the aligned span is not
always (`26357_1_1_1` ~ CheZOD117 `19135_1_1_1` spans 76 % of the training chain
one way and 80 % the other). The **training** partner of a hit is removed, never
the test chain. The release gate's exact-substring check (below) needs no
score.

- **Stage 1 — remove all cluster members** (A0): cluster the unfiltered
  superset with the test sequences at 30 % id / 80 % coverage of both
  sequences (`--cov-mode 0`) and drop every training sequence sharing a
  cluster with a test sequence (catches transitive leaks). Stage 1 keeps
  `--cov-mode 0` on purpose: target coverage in a clustering would let a long
  multi-domain construct pull fragments of its other domains into one
  cluster, dropping chains unrelated to any test chain. The containment case
  is handled by stage 2.
- **Stage 2 — search & remove** (A): `easy-search` the training chains against
  the test sets in both directions (since 2026-11 one search per direction of
  the unfiltered superset, whose hits each tier restricts to its own IDs;
  `<tier>_testset_hits.tsv` and `<tier>_testset_hits_reverse.tsv`) with
  `--min-seq-id 0.3 --cov-mode 1 -c 0 --prefilter-mode 2 -e 100 --dbtype 1`
  (every pair aligned, no k-mer prefilter; the 27-bit floor applied to the
  parsed hits) and drop
  every training chain with an alignment that spans ≥ 80 % of the test chain,
  `(tend − tstart + 1) / tlen ≥ 0.8`, computed from the hit table (rule
  `either`: or ≥ 80 % of the training chain, `(qend − qstart + 1) / qlen`).
  This enforces the leakage criterion. The 80 % is not passed to mmseqs: with any
  `-c` above 0, mmseqs skips a pair by length ratio before aligning it (under
  `--cov-mode 1`, when the training chain is shorter than 80 % of the test
  chain; under `--cov-mode 0`, when either chain is). Stage 2 catches a
  training construct that contains a test chain (e.g. test `51709_1_1_1`,
  169 aa, inside `53339_1_1_1`, 419 aa) and gapped pairs whose length ratio is
  below 0.8 in either direction (`27177_1_1_1`, 124 aa, vs test
  `27176_1_1_1`, 99 aa, 79.8 % identity; `19243_1_1_1`, 148 aa, spanning test
  `16205_1_1_1`, 238 aa, at 59.6 %). Builds up to `2026-10` ran this stage
  with `--cov-mode 0 -c 0.8` (80 % of both chains) and missed all of these.
- **Cluster** the strict residual at **50 % id / 80 % cov** (B), then iterative
  **`clusterupdate`** strict → moderate → tolerant → unfiltered (C) so each
  tier extends the previous and shared sequences keep the strict-tier
  representative.

| | strict | moderate | tolerant | unfiltered |
|---|--:|--:|--:|--:|
| input (unique sequences, §2) | 3,048 | 7,221 | 8,901 | 9,482 |
| leakage dropped (total) | 714 | 1,183 | 1,341 | 1,381 |
| — of which stage-2 `easy-search` hits | 660 | 1,054 | 1,189 | 1,222 |
| — of which stage-1-only (transitive) | 54 | 129 | 152 | 159 |
| **non-redundant** (after leakage) | **2,334** | **6,038** | **7,560** | **8,101** |
| **training reps** (cluster representatives) | **1,895** | **4,520** | **5,466** | **5,803** |

These counts are from the `2026-08` build, whose stage 2 still used
`--cov-mode 0`. With the test-chain criterion, stage 2 removes more, and the
counts change at the next rebuild.

→ `data/interim/build/mmseqs/`.

## 5. Quality-best representative override

`trizod dataset representatives` replaces each mmseqs cluster representative
with the highest-`quality_score` member → `train_<tier>_best.fasta` (the
**canonical** training FASTA). The cluster count is unchanged (1,895 / 4,520 /
5,466 / 5,803); only which member represents each cluster moves — 175 / 408 /
464 / 480 clusters (9.2 / 9.0 / 8.5 / 8.3 %), listed in
`cluster_repr_overrides_<tier>.tsv`.

## 6. Package + leakage gate

`trizod dataset package --version <version>` stages the package bundle (the
input of the Parquet; the deposit itself is assembled in §6a) and runs a hard
**leakage gate**. The gate fails if

- any test ID or exact test sequence appears in any training set;
- any staged training record is an **exact substring of a test chain, or
  contains one, over ≥ 20 aa** (whatever the rule; no aligner involved); or
- any staged training record meets the stage-2 criterion **under the rule the
  build recorded** (`mmseqs/leakage_rule.json`): ≥ 30 % identity over ≥ 80 % of
  the test chain (`test`) or of the test or the training chain (`either`), at
  ≥ 27 bits, searched in both directions (until 2026-11-17: E ≤ 10⁻³, training
  chains as queries only).
  `--leakage-rule` on the package step may only restate the recorded rule.

The coverage check runs the stage-2 search (both directions, coverage from the
alignment span), so mmseqs must be on PATH, and every
offending pair is listed. The rule and its parameters are copied into
`MANIFEST.json` under `leakage`. The `2026-08` bundle passed the ID and
exact-sequence checks with **0 shared IDs and 0 exact-sequence matches**
against the 575 test sequences. The coverage and substring checks were added
later, and the `2026-10` bundle fails them. Under the `test` rule, a training
chain contained in a longer test chain stays in training, so the substring
check fails such a build whenever one is ≥ 20 aa.

→ `<work-dir>/release_bundle/trizod-dataset-<version>/` (train/ + scores/ +
test/ + MANIFEST). The single-table `trizod_dataset.parquet` and the
chemical-shift companion `trizod_shifts.parquet` (every assigned shift on a
canonical residue, backbone and side chain) are then built from that bundle by
`scripts/build_parquet_dataset.py`, with `--calibration <threshold-framework
output>` so the table metadata carries the calibration of the cuts. The main
table is sorted by `(entry_id, id)` in 500-row groups; every C, CA, CB and N row
of the shift table is flagged against the frozen median/MAD reference
`trizod/shifts/mad_reference.csv`. The table metadata records the dataset
version, the licences, the code commit, the BMRB snapshot (name, n_entries,
entry_list_sha256), the test-set name and counts, the cuts with their rule and
the k ≥ 10 note, and the calibration summary; `docs/dataset/columns.md` is the
column dictionary, generated from the column contract.

## 6a. Assemble the deposit

`trizod dataset finalize` collects the hand-over into one folder: the Parquet
and the shift table; the deploy FASTA, regenerated from the bundle; the UdonPred
v2 JSONL view and the test-set crosswalk, generated from the Parquet; the test
FASTAs (including the retired chains and the 117 original CheZOD117 sequences)
with the labels table and the pin provenance; `class_balance.tsv`;
`known_issues.tsv` (one row per chain and issue: `chain_id`, `issue`,
`evidence`); `bmrb_files.sha256` (the SHA-256 of every snapshot entry file);
`VALIDATION.json` of the final audit when given (`--validation`); the README,
datasheet and CHANGELOG rendered from the templates in `docs/dataset/` with the
build's own counts; `columns.md`, `LICENSE.md`, `CITATION.cff` and
`zenodo.json`; and one `MANIFEST.json` over every file (SHA-256, size, row and
chain counts, version and version DOI, licence, the scoring commit apart from
the build and finalize commits, snapshot and cross-reference-source hashes,
cuts, calibration and threshold-framework commit, leakage rule, known-issue
counts; `total_bytes` covers every file but `MANIFEST.json` and `SHA256SUMS`)
plus `SHA256SUMS`.

The CHANGELOG's changes and known issues and `known_issues.tsv` are computed
(`trizod.dataset.known_issues`), never typed: tier moves by rule, the 273 K
re-score statistics and sequence changes against the previous build
(`--previous-build <its package bundle>`); pH* / pD readings, temperatures
read as Celsius or Fahrenheit and pressures without a unit from the build's
`BmrbEntry` pickles (`--pkl-dir`); snapshot drift and construct caveats from
the audit's tables (`--snapshot-drift`, `--construct-caveats`); the pin's
trimmed chains from its provenance; the reviewed chains of
`docs/dataset/known_issues_reviewed.tsv` (each checked to be in the release).
A missing input renders as "not assessed".

Finalize refuses a Parquet without calibration, test FASTAs that disagree with
the Parquet, a Parquet whose scoring commit differs from the bundle's, a
snapshot whose file count differs from the one the Parquet names, and any
shipped file that still names an absolute path of the build machine (the deploy
summary, crosswalk and UdonPred export summary record deposit-, bundle- or
repo-relative paths after finalize rewrites them). The same inputs give
byte-identical files in any output folder.

## 7. How training is separated from the test sets

Every `train_<tier>` set is redundancy-reduced in two stages against
**CheZOD117 (release chains and original sequences), the TriZOD test set and
the retired test chains**. The stage-2 mmseqs search, in both directions, finds
no training chain that aligns to a test chain at ≥ 30 % identity and ≥ 27 bits
over ≥ 80 % of the test chain (rule `test`) or of the test or the training chain
(rule `either`), whatever the length ratio (builds since 2026-11 align every
pair, `--prefilter-mode 2`, so no k-mer prefilter can miss one; see "The
significance condition"), and the release gate finds none either, nor an
exact-substring containment of ≥ 20 aa in either direction. Since 2026-11-17 the
gate reads **every training and redundant row of every tier** — the training
FASTAs and every member of every `clusters*.tsv`, with sequences from
`scores/<tier>/scores.json` — where it used to read the training FASTAs only and
so missed 1,521 redundant rows. The two test sets are themselves disjoint at the
leakage criterion: the TriZOD set was drawn from clusters without a CheZOD
sequence, and a TriZOD test chain that still meets the criterion against
CheZOD117 is retired (§3). So a model can train on any `train_<tier>` set and be
evaluated on **CheZOD117** (the established hand-balanced benchmark, comparable
to ODiNPred/SETH) and on **TriZOD-test v1.0** (the larger,
natural-BMRB-distribution held-out set) with homology controlled at that
criterion — the basis for the UdonPred benchmark. What is controlled is stated
precisely because it is not zero: remote homologues below the criterion, test
chains that share a UniProt accession with a training chain (another
construct), and the overlap with protein-language-model pretraining corpora
remain, and earlier TriZOD test sets are not comparable (the deposit's
crosswalk).

## 8. Sequence alphabet

A released `sequence` holds the **20 standard residues plus X, U and Z**: X for
an unknown or non-standard residue, U for selenocysteine, Z for Glx, each as
deposited in `_Entity.Polymer_seq_one_letter_code` (upper-cased, line breaks
removed). Characters that are not letters are no residues (release audit D15):

- A **trailing** run of non-letters is stripped at parse time
  (`trizod.bmrb.bmrb.normalize_sequence`), before any per-residue array is
  built, so the sequence and every per-residue array (`gscores`, `zscores`,
  `k`, `mask`) lose that terminal position together. The 2026-11-11 snapshot
  has two: `50015_1_1_1` ends in the translation-stop symbol `*` (its poly-seq
  loop lists residue 106 as `X`; the last shift is on residue 105) and
  `52376_1_1_1` in a typed apostrophe (the poly-seq loop ends at 77; the last
  shift is on 77). No shift references either stripped position.
- An **internal** non-letter becomes `X`, one for one, which keeps the residue
  numbering of the shifts. The snapshot holds one, a `-` in bmr53341, an entry
  without assigned chemical shifts, so no released chain has one.

A protein-language-model tokenizer that knows the 20 residues and X therefore
needs only U and Z mapped (the 2026-11-17 build: U at 2 positions of 1 chain, Z
at 1 position).

## 9. CheZOD overlap flags

The Parquet flags every chain whose sequence meets the leakage criterion (§4,
rule `either`, both directions, ≥ 27 bits) against at least one sequence of a
CheZOD set: `chezod117_overlap` (release chains plus the 117 original
sequences), `chezod1325_overlap` (`allseqs1325.txt`) and `chezod1174_overlap`
(the SETH training set, `CheZOD1174_training_set_sequences.fasta`)
(`trizod/dataset/chezod_overlap.py`). Training and redundant rows never overlap
CheZOD117; the CheZOD1325 and CheZOD1174 flags pick the test chains clean of a
predictor's own training data (SETH and ADOPT train on CheZOD1174, ODiNPred on
CheZOD1325) for a fair comparison.

## Scripts (run order)

```bash
scripts/build_dataset.sh         # the whole chain end to end
# = trizod dataset build → test-set → redundancy → representatives → package
uv run --with pyarrow python scripts/build_parquet_dataset.py \
    --in-bundle <bundle> --out <trizod_dataset.parquet> --shifts-out <trizod_shifts.parquet> \
    --composition-csv <work-dir>/final_dataset/_composition_cache.csv \
    --bmrb-dir <snapshot> --xref-sources <xref dir> --calibration <framework dir>
uv run --with pyarrow trizod dataset finalize --parquet <trizod_dataset.parquet> \
    --shifts <trizod_shifts.parquet> --bundle <bundle> --bmrb-dir <snapshot> \
    --out <release dir> --crosswalk-root <checkout holding data/> \
    --previous-build <previous build's bundle> --pkl-dir <the build's tmp/bmrb_entries> \
    [--snapshot-drift <TSV>] [--construct-caveats <TSV>] [--validation <VALIDATION.json>] \
    [--calibration-input <the Parquet the framework calibrated on>]
# refresh an existing shift table's metadata (release keys, counts) without rebuilding it
uv run --with pyarrow python scripts/build_parquet_dataset.py --in-bundle <bundle> \
    --restamp-shifts <trizod_shifts.parquet> --pkl-dir <the build's tmp/bmrb_entries> \
    --bmrb-dir <snapshot>
```

The test gate is `uv run --with pyarrow python -m pytest tests/`. Keep the
`python -m`: with uv 0.13.8 a bare `uv run --with pyarrow pytest` skipped the
Parquet tests silently (release audit 2026-11-17b, A2). The
data-backed tests read `TRIZOD_DATA_ROOT` (a checkout holding `data/`) and
`TRIZOD_BMRB_DIR` (the snapshot); `tests/test_full_dataset_regression.py`
(`-m slow`, `TRIZOD_REGRESSION_BUILD=<a build under data/interim>`) re-scores a
seeded, rule-aware subset of that build's snapshot and compares every field
with the build's `scores.json`.

`scripts/build_mad_reference.py` regenerates the far-outlier reference from a
shift table. It is **not** part of a routine build: the committed reference is
frozen on purpose, so rerun it only to move the flag deliberately, and commit
the result.
