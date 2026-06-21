# mmseqs2 redundancy reduction

## Source for the parameters

The pipeline and parameters are taken **verbatim** from the original
TriZOD report
(`trizod_publication/source-material/sharelatex-export-2026-05-08/report.tex`),
section *"Redundancy reduction"*:

> Redundancy reduction was performed in a multi-step procedure using
> mmseqs2 (Steinegger 2017). First, all sequences in the strict set
> were clustered using `mmseqs cluster` at 30% sequence identity and
> 80% coverage with options
> `--alignment-mode 3 --cov-mode 0 -s 7.5 --comp-bias-corr 0 --mask 0`.
> If not specified otherwise, all other mmseqs calls were performed
> with these options. A subset of 25% of all clusters that contained
> no sequence of any CheZOD dataset was chosen at random and their
> member sequences clustered again but at 50% sequence identity to
> yield the TriZOD test set of 348 proteins.
>
> Next, all sequences of the unfiltered set […] was redundancy reduced
> against test set entries by […] `mmseqs easy-search` at 30% sequence
> identity and 80% coverage. The remaining sequences were used to
> generate filter preset specific, redundancy reduced training sets.
> To this end, the strict set […] were clustered using `mmseqs cluster`
> at 50% sequence identity and 80% coverage. Then, the resulting
> clusters were iteratively updated with sequences from the other
> filter sets […] using `mmseqs clusterupdate` at 50% sequence
> identity and 80% coverage to yield all remaining training sets that
> all share the same cluster representatives for common clusters.

The CheZOD117 set (115 sequences) and the TriZOD test set (348
sequences) were derived in November 2025 and live in
`data/2024-05-09/`. We treat both as **frozen inputs** — they are not
re-derived in this folder. Everything downstream of the test sets is
re-run on the May 2026 dataset.

## Pipeline (`scripts/run_mmseqs_pipeline.py`)

For each tier:

1. **Step A** — `mmseqs easy-search <tier>.fasta combined_testset.fasta
   --min-seq-id 0.3 -c 0.8 --alignment-mode 3 --cov-mode 0 -s 7.5
   --comp-bias-corr 0 --mask 0`.
   Drop every sequence that hits either CheZOD117 or the TriZOD test
   set.
   Output: `<tier>_no_testset.fasta`.

2. **Step B** — `mmseqs createdb` then `mmseqs cluster` on
   `strict_no_testset.fasta` at `--min-seq-id 0.5 -c 0.8 [common
   options]`. Produce:
   * `train_strict_clu.tsv` — two-column cluster table.
   * `train_strict.fasta` — one record per cluster representative.

3. **Step C** — for each next-most-stringent tier (moderate → tolerant
   → unfiltered):
   `mmseqs clusterupdate <prev_db> <new_db> <prev_clu>
   <new_db_merged> <new_clu> <workdir> --min-seq-id 0.5 -c 0.8 [common
   options]`. Because `clusterupdate` reuses old cluster IDs whenever
   a sequence is unchanged, the cluster representative of every shared
   sequence stays the strict-tier (or earlier-tier) member, mirroring
   the paper's intent.

   This requires that the FASTAs across tiers carry identical IDs for
   identical sequences — implemented in `build_final_dataset.py` via
   the `global_repr_ID` column (see [final-dataset.md](final-dataset.md)).

## Output counts

(From the pipeline `run_mmseqs_pipeline.py` stdout.)

| tier | input seqs | after test-set filter | clusters | reps |
|---|---:|---:|---:|---:|
| strict | 2,039 | 1,657 | 1,388 | 1,388 |
| moderate | 6,346 | 5,604 | 4,205 | 4,205 |
| tolerant | 9,035 | 8,062 | 5,828 | 5,828 |
| unfiltered | 9,480 | 8,473 | 6,071 | 6,071 |

Hits against the combined test set (CheZOD117 + TriZOD test):

| tier | distinct queries with a hit |
|---|---:|
| strict | 382 |
| moderate | 742 |
| tolerant | 973 |
| unfiltered | 1,007 |

**Comparison to the paper's numbers**: the paper reports ~7,324
unfiltered cluster representatives; we get 6,071. The drop is
attributable to (a) the bound-complex filter (~27% of pkl entries
flagged multi-molecule, more aggressive than the paper's pipeline),
and (b) the May 2026 BMRB snapshot causing some new sequences to
collapse into existing clusters.

## Quality-best cluster representative override

The paper's pipeline leaves the representative choice to mmseqs'
internal logic. We additionally provide a **quality-based override**
via `scripts/cluster_best_repr.py`:

* For every mmseqs cluster, find the member with the highest
  `quality_score` (defined in [final-dataset.md](final-dataset.md)).
* If that member is not the mmseqs-picked representative, record the
  override in `cluster_repr_overrides_<tier>.tsv`.
* Write `train_<tier>_best.fasta` with one record per cluster using
  the score-best member.

Override stats:

| tier | clusters | overrides | % |
|---|---:|---:|---:|
| strict | 1,388 | 109 | 7.9% |
| moderate | 4,205 | 341 | 8.1% |
| tolerant | 5,828 | 434 | 7.4% |
| unfiltered | 6,071 | 444 | 7.3% |

Top-3 quality-difference examples (strict): the mmseqs-pick was
within-tier, but the override gains 400–600 points of quality
(equivalent to a few hundred extra backbone shift positions). For
ML training where the goal is *learning a good predictor of disorder*,
training on the best-quality member of each cluster is the more
defensible choice.

## Recommended ML dataset

Pick **one** of:

* `train_<tier>.fasta` — official mmseqs representatives (matches the
  paper's recipe).
* `train_<tier>_best.fasta` — score-best representatives (preserves
  cluster structure but trains on better-quality data).

We ship both. The metadata to choose between them sits in
`train_<tier>_clu_best.tsv`, which has columns:

```
cluster_repr  member  best_repr  member_quality  best_repr_quality  quality_diff_vs_mmseqs  member_tier  member_ID  n_bb_pos  n_bb_types  max_potenci_off  max_lacs_off
```

## Reproducing

```bash
# 1. Cleanly re-cluster (paths are derived from script location).
uv run python docs/260520/scripts/run_mmseqs_pipeline.py

# 2. Add quality-best override columns + FASTA.
uv run python docs/260520/scripts/cluster_best_repr.py
```

All artefacts land in `docs/260520/data/mmseqs/`.
