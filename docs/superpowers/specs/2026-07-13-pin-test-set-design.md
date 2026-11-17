# Pin the TriZOD test set — design

- **Date:** 2026-07-13
- **Status:** approved (pending spec review)
- **Module:** `trizod/dataset/testset.py` (+ `paths.py`, new committed pin file, tests)

## Problem

`trizod dataset test-set` redraws the TriZOD test set from scratch on every run:
it clusters the strict-tier sequences, seed-samples 25 % of the CheZOD-free
clusters, and reclusters the sampled members at 50/80 to get representatives.
The fixed seed only makes the draw reproducible for a **byte-identical input**.

In practice the input is not stable: the quality-best representative of each
duplicate-sequence cluster shifts whenever scores change (e.g. the LACS
`robustfit` fix), which reshuffles the cluster representative IDs, which changes
the `sorted(free_clusters)` list the seeded sampler indexes into. A tiny,
otherwise-benign score change therefore cascades into a wholesale test-set
turnover — the 2026-07-13 bug-fix rebuild changed **262 of 344** test sequences
(only 82 shared). The same fragility applies when the dataset grows.

A benchmark test set must be stable across reruns, bug fixes, and dataset growth.

## Goal

Freeze the test set. After a one-time adoption of the corrected (post-fix)
draw, future `test-set` runs reproduce exactly the same test sequences, whether
the dataset grew or scores changed. Redrawing is an explicit, opt-in action.

Non-goals: changing the CheZOD117 external benchmark; changing the sampling
recipe itself; regenerating the Zenodo Parquet / deploy FASTA (separate steps).

## Decision (Approach A: pin the final representatives)

Pin the **final representative sequences** — the content of
`TriZOD_test_set.fasta` — as a committed reference file, and make `test-set`
load it by default instead of redrawing. Rejected alternatives: (B) stabilising
the clustering inputs — still mmseqs-dependent and still drifts as clusters
shift; (C) hashing each sequence to test/train — deterministic but produces a
*different* set than the current draw, which we explicitly want to preserve.

**Baseline:** the post-fix draw from the 2026-07-13 rebuild
(`data/interim/build/testset/TriZOD_test_set.fasta`, 342 sequences). Prior users
will be handed the new set; everything is correct and stable from here.

## Design

### Pinned reference file (committed)

- `trizod/dataset/pinned/TriZOD_test_set.fasta` — the 342 post-fix representative
  sequences, `>entry_id` headers (same format as the emitted file).
- `trizod/dataset/pinned/TriZOD_test_set.provenance.json` — sidecar recording:
  source build/version, date, upstream seed + sample fraction, sequence count,
  and a note that it was adopted from the 2026-07-13 post-fix rebuild. Purely
  informational.

Committed under the package (not the git-ignored `data/`) so it ships with the
package and is version-controlled. `paths.resolve_paths()` gains a
`pinned_testset` entry resolving to this repo-root-relative path.

### `test-set` modes

`test-set` (module `testset.py`) gains a `--redraw` flag:

- **Default (pinned, no flag):** load the pinned sequences; build a
  `sequence -> [entry_ids]` map from the current strict-tier fasta
  (`final_dataset/strict/strict.fasta`); for each pinned sequence emit one record
  labelled by the **pinned entry ID if still present**, else the
  **lowest-numbered entry ID** sharing that identical sequence. A pinned sequence
  with no entry in the current strict pool is **dropped with a WARNING** and
  recorded in the summary. Writes `TriZOD_test_set.fasta` and
  `build_test_set_summary.json`. No mmseqs, no clustering.
- **`--redraw`:** run today's seeded 30/80 → sample → 50/80 recipe (unchanged),
  write the outputs, **and overwrite the committed pin file + provenance**. This
  is the only operation that changes the test set. If the pin file is missing,
  `test-set` errors with a message telling the user to run `--redraw` to
  establish it.

### Data flow

```
committed pin fasta ──default──▶ testset.py ──▶ <work-dir>/testset/TriZOD_test_set.fasta
current strict fasta ─(resolve entry IDs)                    │
                                                             ├─▶ redundancy.py (leakage removal, re-clusters fresh)
                                                             └─▶ deploy_fasta.py (per-residue G-score labels)
```

Downstream is unchanged — both consumers read `TriZOD_test_set.fasta`. Leakage
removal still re-clusters test+train fresh against the pinned sequences, so
train stays disjoint from the pinned test set as data grows.

### Error handling

- Missing pin file → explicit error: run `test-set --redraw` to establish it.
- Pinned sequence absent from current strict pool → drop + WARNING; count and IDs
  listed in `build_test_set_summary.json` under `dropped`.
- Pinned entry ID gone but sequence present → substitute lowest-numbered entry
  with the identical sequence; count listed under `resolved_substitutions`.

### Testing (TDD)

Unit tests (no mmseqs; default path is pure sequence matching):

1. Pinned load reproduces exactly the pinned sequences given a matching strict
   pool.
2. Entry-ID fallback: pinned ID absent → lowest-numbered entry with same seq.
3. Dropped sequence: pinned seq absent from strict pool → excluded + warned +
   recorded in summary.
4. Stability: perturbing the strict pool's representative IDs (same sequences)
   leaves the emitted test set identical.
5. Missing pin file → raises the documented error.

The `--redraw` seeded path keeps its existing mmseqs-gated smoke coverage.

### Bootstrap

Copy the current `data/interim/build/testset/TriZOD_test_set.fasta` (342 seqs)
to the committed pin location and write the provenance sidecar, as part of
implementing this change.

## Open assumptions

- The post-fix local build is the set we distribute; no separate canonical
  release FASTA needs to be pinned instead.
- The strict-tier fasta is the correct resolution source (the test set is
  defined over CheZOD-free strict sequences).
