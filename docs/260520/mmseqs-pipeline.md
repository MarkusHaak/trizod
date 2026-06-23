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

CheZOD117 (115 sequences, the fixed external benchmark) lives in
`data/2024-05-09/`. The TriZOD test set (344 sequences) is **recreated
from the current snapshot** by `scripts/build_test_set.py` (seeded; see
the recipe quoted above) and written to
`docs/260520/data/testset/TriZOD_test_set.fasta`. Run `build_test_set.py`
before `run_mmseqs_pipeline.py`.

## Pipeline (`scripts/run_mmseqs_pipeline.py`)

For each tier:

1. **Step A — test-set leakage removal (two stages)** against the
   combined test set (CheZOD117 + TriZOD test), both at
   `--min-seq-id 0.3 -c 0.8 [common options]`:
   * *Stage 1 ("remove all cluster members", Step A0)* — `mmseqs
     easy-cluster` the unfiltered superset together with the test
     sequences and drop every training sequence that shares a cluster
     with a test sequence (catches transitive leaks A~B~test). The
     leaked IDs are removed from every nested tier.
   * *Stage 2 ("search & remove")* — `mmseqs easy-search <tier>.fasta
     combined_testset.fasta` and drop every direct hit.
   Output: `<tier>_no_testset.fasta` (tier minus stage-1 ∪ stage-2).
   CheZOD1325 is **not** a leakage target here (it only shapes the
   TriZOD test set during its construction).

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
| strict | 2,039 | 1,547 | 1,254 | 1,254 |
| moderate | 6,346 | 5,425 | 4,063 | 4,063 |
| tolerant | 9,035 | 7,867 | 5,684 | 5,684 |
| unfiltered | 9,480 | 8,290 | 5,927 | 5,927 |

Per-tier removal against the combined test set (CheZOD117 + TriZOD
test), split by stage (stage-2 = direct easy-search hits; stage-1
only = additional transitive leaks):

| tier | stage-2 hits | stage-1 only | total removed |
|---|---:|---:|---:|
| strict | 461 | 31 | 492 |
| moderate | 809 | 112 | 921 |
| tolerant | 1,025 | 143 | 1,168 |
| unfiltered | 1,042 | 148 | 1,190 |

**Comparison to the paper's numbers**: the paper reports ~7,324
unfiltered cluster representatives; we get 5,927. The drop is
attributable to (a) the bound-complex filter (~27% of pkl entries
flagged multi-molecule, more aggressive than the paper's pipeline),
(b) exact-sequence dedup, and (c) the May 2026 BMRB snapshot causing
some new sequences to collapse into existing clusters.

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
| strict | 1,254 | 110 | 8.8% |
| moderate | 4,063 | 339 | 8.3% |
| tolerant | 5,684 | 433 | 7.6% |
| unfiltered | 5,927 | 443 | 7.5% |

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
# 1. Recreate the TriZOD test set from the current snapshot (seeded).
uv run python docs/260520/scripts/build_test_set.py

# 2. Cleanly re-cluster (paths are derived from script location).
uv run python docs/260520/scripts/run_mmseqs_pipeline.py

# 3. Add quality-best override columns + FASTA.
uv run python docs/260520/scripts/cluster_best_repr.py
```

Or run the whole release end to end: `docs/260520/scripts/run_all.sh`.

All artefacts land in `docs/260520/data/mmseqs/`.
