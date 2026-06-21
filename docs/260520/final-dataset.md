# Building the ML-ready dataset

This is the bridge between the 2026-05-05 score release and the mmseqs
clustering step. It takes the four per-tier `scores.json` files,
attaches assembly composition from the BMRB pkls, drops bound /
multi-molecule entries, deduplicates by exact sequence with a
quality-based ranking, and writes per-tier FASTAs + ranked TSVs.

## Inputs

| Path | Purpose |
|---|---|
| `data/release/<tier>/scores.json` | Per-tier scored entries (16,851 / 15,433 / 10,107 / 3,033). |
| `tmp/bmrb_entries/*.pkl` | Cached `BmrbEntry` objects (16,963 files); used to read assembly composition. |

## What "bound / multi-molecule" means here

An entry is flagged **bound** if *any* of the following holds:

| signal | source |
|---|---|
| more than one **distinct** `Entity` referenced inside any of its assemblies | `Assembly.entities` |
| has a `non-polymer` entity (ligand, drug, cofactor) | `Entity.type == 'non-polymer'` |
| has a `polydeoxyribonucleotide` or `polyribonucleotide` entity | `Entity.polymer_type` |
| has a `metal` entity | `Entity.type` startswith "metal" |

Homo-oligomers — same Entity record referenced multiple times in one
assembly — are **kept** (they show up as a single distinct entity ID
in `Assembly.entities`).

Across the 16,963 cached BMRB entries, 4,558 (26.9%) are flagged
multi-molecule by these criteria.

## Quality score for ranking

Each row gets a flat composite quality score so we can pick the best
representative for a sequence (and later, the best member of an mmseqs
cluster):

```
quality_score = tier_rank × 1,000,000
              + (bbshift_positions_post × bbshift_types_post)
              − max(|POTENCI residual offset|)
```

* `tier_rank` is 4 / 3 / 2 / 1 for strict / moderate / tolerant /
  unfiltered. The × 1,000,000 weight ensures any entry from a stricter
  tier outranks every entry from a looser tier.
* The next term rewards entries with more residues × atom types
  measured (more shift evidence ⇒ more reliable G-scores).
* The last term penalises larger residual offsets — a tie-breaker, not
  a primary signal.

The exact values are not important — only the ordering matters.

## Pipeline (`scripts/build_final_dataset.py`)

1. Build a composition cache by loading every BMRB pkl once
   (`_composition_cache.csv`).
2. Load `scores.json` for every tier, concatenate, attach composition,
   compute `quality_score`.
3. Drop rows whose sequence is < 20 residues or whose entry is bound.
4. **Global dedup**: for every sequence, pick the highest-quality row
   as the canonical representative; record its ID as `global_repr_ID`.
   Carry this ID through every tier so that the *same sequence* gets
   the *same FASTA header* in every tier file (essential for
   `mmseqs clusterupdate`, which requires consistent IDs between old
   and new databases).
5. **Per-tier FASTA**: each tier's FASTA contains every sequence whose
   strictest-passing tier is ≥ this tier, headered with the global
   representative ID. Tiers are therefore nested by sequence
   (strict ⊂ moderate ⊂ tolerant ⊂ unfiltered).
6. **Per-tier ranked TSV**: every entry that passes the tier, with
   columns for the quality score, the seq-rank inside the tier, the
   global representative ID, and the composition flags.

## Output counts

(From `data/final_dataset/final_dataset_summary.json`.)

| tier | initial rows | dropped (<20 aa) | dropped (bound) | kept rows | unique sequences (FASTA) | dedup ratio |
|---|---:|---:|---:|---:|---:|---:|
| unfiltered | 16,851 | 1,349 | 4,195 | 11,307 | **9,480** | 16.2% |
| tolerant | 15,433 | 1,108 | 3,858 | 10,467 | **9,035** | 13.7% |
| moderate | 10,107 | 475 | 2,646 | 6,986 | **6,346** | 9.2% |
| strict | 3,033 | 30 | 829 | 2,174 | **2,039** | 6.2% |

Roughly half of the dataset shrinkage comes from the bound-complex
filter, the rest from exact-sequence collapse and length filtering.
The stricter the tier, the smaller the dedup ratio — strict-tier
entries are already mostly unique sequences (people only deposit
multiple copies of an interesting sequence after the first deposit
passes quality controls).

## Reading the ranked TSV

```
$ head -n1 docs/260520/data/final_dataset/strict/strict_all_ranked.tsv
ID  entryID  stID  entity_assemID  entityID  tier  tier_rank  len  n_bb_pos  n_bb_types  total_bbshifts  shift_volume  max_potenci_off  max_lacs_off  quality_score  seq_rank_tier  is_seq_repr_tier  global_repr_ID  global_repr_tier  is_bound  has_non_polymer  has_nucleic  multi_protein_assembly  n_entities  entity_name  ionic_strength  pH  temperature  seq
```

Useful filters:

* `seq_rank_tier == 1` → the FASTA representative for this sequence
  in this tier.
* `is_seq_repr_tier == True` → same as above.
* `global_repr_ID` → the canonical ID used in every tier's FASTA
  (matches the FASTA header).

## Reproducing

```bash
uv run python docs/260520/scripts/build_final_dataset.py
```

Outputs go to `docs/260520/data/final_dataset/{<tier>,_composition_cache.csv,final_dataset_summary.json}`.
