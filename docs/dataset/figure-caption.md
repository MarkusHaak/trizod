# Figure caption — TriZOD dataset-construction figure

The figure is maintained as **TikZ/LaTeX** (canonical):

- `publication/manuscript/Figures/2026-06-22_TriZOD_redundancy_reduction.tex` (+ `.pdf`)
- Superseded checkpoints live in `Figures/archive/` (see its `README.md` for the
  drawio → precolor → colorblind → twinpanel → mirrored → current progression).

The **authoritative caption** is the `\caption{}` for `fig:redundancy_reduction` in
`Article.tex`; the text below mirrors it for reference and cross-checking. The
construction is documented in [`dataset-construction.md`](dataset-construction.md).

> **2026-06 redesign note.** The figure now uses a strict *shape = function,
> colour = data category* grammar. The row after leakage removal is labelled
> **leakage-free** (was "non-redundant"): those sequences are free of test-set
> leakage but still internally redundant — the redundancy reduction is the 50/80
> clustering that produces the training sets. The former decision-diamond
> (random sampling) and trapezium (external CheZOD) shapes were removed.

## Short caption (figure-only, ~2 sentences)

> **Construction of the TriZOD dataset.** An automated pipeline parses 17,388 BMRB
> entries, removes duplicate sequences, bound/multi-molecule complexes and sequences
> < 20 aa to obtain the *unfiltered* set, and derives four **nested** filter tiers by
> applying progressively stricter quality filters (unfiltered → tolerant → moderate →
> strict); each tier is made leakage-free by removing held-out-test-set leakage and is
> then redundancy-reduced by mmseqs clustering into training sets of 1,254–5,927
> proteins. A seeded side-branch removes CheZOD redundancy from the strict set and
> reclusters a random subset into the CheZOD-disjoint TriZOD test set (344) which, with
> the external CheZOD117 (115), forms the held-out evaluation data.

## Full caption (with symbol key)

> **Construction of the TriZOD dataset and its leakage-free test split.** Read left→right.
> *(left) Training-set construction.* The 17,388 BMRB entries (cylinder) are parsed and
> reduced to the **unfiltered** set (9,480) by removing duplicate sequences,
> bound/multi-molecule complexes and sequences < 20 aa; progressively stricter quality
> filters derive the four **nested** tiers (unfiltered 9,480 ⊃ tolerant 9,035 ⊃ moderate
> 6,346 ⊃ strict 2,039). Each tier is first made **leakage-free** by two-stage test-set-
> leakage removal (vermillion box) at 30 % identity / 80 % coverage against the held-out
> test sets — stage 1 drops training sequences that co-cluster with a test sequence,
> stage 2 drops direct `easy-search` hits — yielding the leakage-free sets
> (8,290 / 7,867 / 5,425 / 1,547), which are still internally redundant. These are then
> redundancy-reduced into the per-tier **training sets** (green; 5,927 / 5,684 / 4,063 /
> 1,254): `cluster` builds the strict set, then iterative `clusterupdate`
> *strict → moderate → tolerant → unfiltered* combines the previous tier's training set
> with the current leakage-free set, so each tier extends the previous. `cluster`,
> `clusterupdate` and the test-set `recluster` all use 50 % identity / 80 % coverage.
> *(right) Test-set construction.* The *strict* tier's unique sequences (grey dashed
> provenance arrow) have their CheZOD redundancy removed (vermillion box: clustered with
> the external CheZOD117 and CheZOD1325 sets at 30 % / 80 %, dropping every cluster that
> contains a CheZOD sequence, so the test set is CheZOD-disjoint), leaving 1,237
> CheZOD-free clusters; a fixed-seed random 25 % of these is reclustered at 50 % / 80 %
> to give the **TriZOD test set** (344, blue). The two **held-out test sets** searched
> against in the leakage step (merged vermillion dotted arrow) are CheZOD117 (115; 2 of
> 117 did not map) and the TriZOD test set; CheZOD1325 enters only the test-set
> construction and is *not* a training-leakage target.
> **Visual encoding (dual, so the figure reads in grayscale and under colour-vision
> deficiency):** *shape* = function — cylinder, source database (BMRB); parallelogram,
> external input data (CheZOD); grey box, operation (parse, cluster, clusterupdate,
> random sample, recluster); vermillion box, sequence-removal step; white box,
> intermediate sequence set; rounded coloured box, output dataset. *colour*
> (colourblind-safe Okabe-Ito) = data category — orange, external CheZOD; green,
> training set; blue, TriZOD test set; vermillion, removed/held-out sequences.
> *Arrows:* solid black, data flow; grey dashed, provenance; vermillion dotted, held-out
> feedback into the leakage step; open arrowheads mark the nested-tier ordering and the
> clustering cascade. All mmseqs steps use
> `--alignment-mode 3 --cov-mode 0 -s 7.5 --comp-bias-corr 0 --mask 0`.

## Numbers shown (for cross-checking the figure)

| row (figure order, left→right) | unfiltered | tolerant | moderate | strict |
|---|--:|--:|--:|--:|
| filter tiers (unique) | 9,480 | 9,035 | 6,346 | 2,039 |
| leakage-free | 8,290 | 7,867 | 5,425 | 1,547 |
| training sets | 5,927 | 5,684 | 4,063 | 1,254 |

The figure shows the cleaned **unfiltered** set (9,480, leftmost column) as the entry
point and derives the stricter tiers rightward from it; the upstream per-tier *scored*
record counts (16,851 / 15,433 / 10,107 / 3,033, before sequence de-duplication) and the within-step
cluster counts (e.g. 2,511 total / 1,274 CheZOD-touching) are no longer drawn but remain
documented in [`dataset-construction.md`](dataset-construction.md).

Test-set side-branch: strict 2,039 + CheZOD117 (115) + CheZOD1325 (1,323) →
remove CheZOD redundancy → 1,237 CheZOD-free → random 25 % → 309 clusters (397 seq) →
recluster → **TriZOD test 344**.
