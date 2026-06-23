# TriZOD Manuscript — Master Plan (rev. 2026-06-18)

**Scope: this is a DATASET manuscript only.** The trained disorder predictor
and its CheZOD/SETH benchmarking are handled by the separate **UdonPred** paper
(bioRxiv). Nothing in this manuscript trains or evaluates an ML model. TriZOD's
job is to deliver and validate the *dataset* (and the pipeline + G-score that
produce it), with a leakage-free split so UdonPred can train on it and test on
CheZOD117 fairly.

A deep-investigation snapshot of where the project stands and a structured plan
to a submittable dataset paper. Cross-checked against the old manuscript
(`trizod_publication/manuscript/Article.tex`), the original report
(`trizod_publication/source-material/.../report.tex`), the codebase, the latest
talk (`docs/260505/talk.typ`), and the decision trail in `docs/260505/` +
`docs/260520/`.

---

## 1. Where we stand

The pipeline, the four-tier scored release, and the redundancy-reduced
per-tier training sets are **all built**. For a dataset paper there is **no
missing code** on the critical path — the remaining work is (a) packaging +
releasing the data, (b) a few finalization decisions, and (c) rewriting the
report-style manuscript into a dataset paper.

| Track | State |
|---|---|
| BMRB parser, 4-tier filtering, POTENCI, Z/G scoring | ✅ shipped, tested |
| LACS re-referencing + integration (`--rereference-mode both`) | ✅ shipped, tested, validated vs 6,692 BMRB LACS reports |
| 4-tier scored release (`data/release/<tier>/scores.json`) | ✅ 16,851 / 15,433 / 10,107 / 3,033 entries |
| Bound-complex removal + exact-seq dedup + quality ranking | ✅ `docs/260520/scripts/build_final_dataset.py` |
| mmseqs2 redundancy reduction + CheZOD/TriZOD-test leakage removal | ✅ `docs/260520/scripts/run_mmseqs_pipeline.py` |
| Quality-best representative override | ✅ `docs/260520/scripts/cluster_best_repr.py` |
| Leakage-free test sets (CheZOD117=115, TriZOD=344, seeded rebuild) | ✅ `build_test_set.py` |
| **Data committed / released (Zenodo DOI)** | ❌ **datasets are UNTRACKED + ungitted; nothing deposited** |
| Manuscript | ⚠️ old report-style draft, mid-LaTeX-migration; reframe to dataset paper |

---

## 2. Dataset locations (verified 2026-06-18)

| What | Path |
|---|---|
| **Per-tier training sets (deliverable)** | `docs/260520/data/mmseqs/train_<tier>.fasta` (mmseqs reps) and `train_<tier>_best.fasta` (quality-best) |
| Cluster membership | `docs/260520/data/mmseqs/train_<tier>_clu.tsv` / `_clu_best.tsv` |
| Post-leakage, pre-cluster sets | `docs/260520/data/mmseqs/<tier>_no_testset.fasta` |
| Deduplicated sets + full metadata | `docs/260520/data/final_dataset/<tier>/<tier>.fasta` + `<tier>_all_ranked.tsv` |
| Per-residue scores (Z/G/k + POTENCI & LACS offsets) | `data/release/<tier>/scores.json` |
| Test sets | CheZOD117: `data/2024-05-09/CheZOD117_test_set.fasta`; TriZOD: `docs/260520/data/testset/TriZOD_test_set.fasta` (rebuilt) |
| Combined test set used for leakage removal | `docs/260520/data/mmseqs/combined_testset.fasta` |

**Tracking status:** `data/` is gitignored; `docs/260520/` is entirely
untracked (0 tracked files). The finished datasets exist only on this machine —
**packaging + release is the #1 priority for a dataset paper.**

---

## 3. How the dataset was built (the recipe to write up)

1. **Score** all BMRB entries with the finalized pipeline (`--rereference-mode
   both`) → `data/release/<tier>/scores.json` (16,851 / 15,433 / 10,107 / 3,033).
2. **Flag bound/multi-molecule** (`build_final_dataset.py:detect_bound`): >1
   distinct Entity in any assembly OR a non-polymer (ligand) OR nucleic-acid OR
   metal entity. Homo-oligomers are kept. ~27% (4,558/16,963) flagged.
3. **Drop** bound entries + sequences < 20 aa.
4. **Quality score** per row:
   `quality_score = tier_rank·10⁶ + (n_bb_pos × n_bb_types) − max|POTENCI residual|`.
5. **Exact-sequence dedup**: keep the highest-quality member per identical
   sequence; carry a `global_repr_ID` stable across tiers (needed for
   `clusterupdate`). Unique seqs: 9,480 / 9,035 / 6,346 / 2,039.
6. **Leakage removal** (`run_mmseqs_pipeline.py` Steps A0+A) vs `combined_testset`
   = CheZOD117 + TriZOD-344 at **30% id / 80% cov**, in **two stages**: stage-1
   "remove all cluster members" (cluster the superset with the test seqs, drop
   any training seq sharing a cluster with a test seq) then stage-2
   `easy-search`. Dropped: 1,190 / 1,168 / 921 / 492 (of which 148 / 143 / 112 /
   31 are transitive leaks unique to stage-1) → kept 8,290 / 7,867 / 5,425 /
   1,547. CheZOD1325 is test-set-selection only (not a training-leakage target).
7. **Cluster** strict residual at **50% id / 80% cov** (Step B), then iterative
   **`clusterupdate`** moderate→tolerant→unfiltered (Step C) so shared sequences
   keep the strict-tier representative. Final cluster reps: 5,927 / 5,684 /
   4,063 / 1,254.
8. **Optional quality-best override** (`cluster_best_repr.py`): replace mmseqs'
   internal representative with the highest-quality cluster member (differs in
   7.3–8.1% of clusters) → `train_<tier>_best.fasta`.

mmseqs common options throughout: `--alignment-mode 3 --cov-mode 0 -s 7.5
--comp-bias-corr 0 --mask 0` (verbatim from the original report).

Full funnel:

| tier | scored | unique seqs | dropped (test leakage) | kept | cluster reps |
|---|---:|---:|---:|---:|---:|
| unfiltered | 16,851 | 9,480 | 1,190 | 8,290 | 5,927 |
| tolerant | 15,433 | 9,035 | 1,168 | 7,867 | 5,684 |
| moderate | 10,107 | 6,346 | 921 | 5,425 | 4,063 |
| strict | 3,033 | 2,039 | 492 | 1,547 | 1,254 |

---

## 4. Data-leakage exclusion (the TriZOD principle) — implemented

The training sets are **redundancy-reduced against CheZOD117 + the TriZOD-344
test set** at 30% identity / 80% coverage (Step 6 above), the same protocol the
original TriZOD report used. Additionally, the TriZOD-344 test set is rebuilt
from the current snapshot (`build_test_set.py`, seeded) from clusters containing
**no** CheZOD sequence. Both test sets are therefore disjoint from every
`train_<tier>` set, so UdonPred can train on TriZOD and evaluate on CheZOD117
with no leakage. This guarantee is the
load-bearing link between this dataset paper and the UdonPred paper.

---

## 5. The CheZOD dataset — what we have vs. fetch

- **Have:** CheZOD117 as TriZOD-scored BMRB entries (composite IDs, **115 seq**,
  the 117 CheZOD proteins mapped to BMRB at 95% id / 90% cov; 2 unmapped) in
  `data/2024-05-09/`. Sufficient for leakage removal + distribution comparison.
- **Do NOT have:** the larger **CheZOD-1325** set anywhere in the suite. Needed
  only if the manuscript keeps the original report's parsing-superiority claim
  (11 mis-parsed + 5 errored entries in the 1325 set). Fetch from Nielsen &
  Mulder 2016 / ODiNPred (mirrored in the SETH repo) if that claim is retained.
- Methods must state the CheZOD117→BMRB mapping and the 115-vs-117 discrepancy.

---

## 6. The three main display items (dataset paper)

Each leads with a **bold finding statement**, then description.

### Figure 1 — Dataset & construction pipeline
**An automated, integrity-checked pipeline converts 17,388 raw BMRB entries into
a re-referenced, redundancy-reduced, CheZOD-leakage-free continuous-disorder
dataset of 1,254–5,927 non-redundant proteins per stringency tier — an order of
magnitude larger than the CheZOD datasets it is benchmarked against.**
Panels: (A) end-to-end workflow (parse → 4-tier filtering → LACS re-referencing
→ Z/G scoring → bound removal → exact-seq dedup → mmseqs redundancy reduction);
(B) per-tier funnel (Section 3 table); (C) leakage-removal step explicit
(1,190/1,168/921/492 training sequences dropped for matching CheZOD117 +
TriZOD-344). Supported by **Table 1** (4-tier filter matrix). *Status: numbers +
figure redrawn (`2026-06-22_TriZOD_redundancy_reduction.drawio`).*

### Figure 2 — LACS re-referencing (data-quality advance)
**LACS pre-correction removes systematic NMR referencing errors that shift at
least one residue's G-score by >0.1 in 55% of strict-tier entries, most
strikingly rescuing the mis-referenced α-synuclein deposit (BMRB 17665) from a
structured-looking to a correctly disordered profile.**
Panels: the existing 4-panel `docs/260520/figures/lacs_effect_gscores.png`
(hexbin POTENCI-only vs LACS+POTENCI r≈0.97; per-tier ΔG violins; max|LACS
offset| vs max|ΔG|; cumulative affected-entry fraction) + α-synuclein 17665 vs
ground-truth 6968 (LACS recovers CA/CB +2.82, N +1.76 ppm). *Status: exists;
cosmetic regen + add the α-synuclein panel.* This is the biggest novelty since
the original report (which had no LACS at all).

### Figure 3 — The G-score label & validation
**The k-independent, 0–1-bounded TriZOD G-score produces BMRB-representative
disorder distributions that agree with DisProt (ROC-AUC up to 0.79) while
exposing annotation errors — e.g. fully-disordered-labeled Cytochrome C scores
as ordered — establishing it as a faithful, experiment-bias-free
continuous-disorder label.**
Panels: (A) G-score vs Z-score (synthetic k-dependence + correlation), showing
the G-score's expected value is k-independent; (B) per-residue/per-protein
G-score PDFs across tiers vs CheZOD117 (TriZOD = natural BMRB distribution,
CheZOD117 = hand-balanced); (C) DisProt ROC (4 tiers × Z/G) + mean-accuracy
vs disorder-fraction scatter, with the Prion/Ubiquitin/Cytochrome C case studies.
*Status: all sub-figures exist in the old manuscript; recompose.*

Arc: **how it's built (Fig 1) → why the labels are correct (Fig 2) → what the
label is + external validation (Fig 3)**, with the DisProt-discrepancy argument
as the discussion hook. The aligned-cluster per-protein examples and the
max-offset distribution become supplementary figures.

---

## 7. Result subsection headers (bold findings)

1. **An integrity-checked NMR-STAR pipeline scores 16,851 BMRB entries across
   four nested stringency tiers, ~10× the legacy CheZOD datasets.**
2. **LACS pre-correction is not redundant with POTENCI/AIC: it moves ≥1
   residue's G-score by >0.10 in 55% of strict-tier entries and recovers
   multi-ppm referencing errors the residual correction alone misses.**
3. **Bound-complex removal, exact-sequence dedup, and mmseqs redundancy
   reduction yield CheZOD-leakage-free training sets of 1,254–5,927 proteins per
   tier without arbitrary cutoffs.**
4. **The k-independent, 0–1-bounded TriZOD G-score is a faithful continuous
   disorder label whose expected value does not inherit NMR experiment-count
   bias, unlike the CheZOD Z-score.**
5. **Continuous BMRB-derived disorder agrees with DisProt (AUC up to 0.79) yet
   exposes annotation errors, arguing binary X-ray-dominated consensus is an
   imperfect ground truth.**

---

## 8. Master execution plan (dataset paper)

**Phase A — Commit & package the dataset (PRIORITY 1, ≈1–2 days)**
- Commit the methyl-wildcard revert (uncommitted working tree) + run the 3-gate
  check (ruff check, ruff format, pytest).
- Decide what to commit vs. release: move the build/mmseqs scripts into the
  tracked repo (they currently sit in untracked `docs/260520/`); decide the
  canonical training FASTA (mmseqs-rep vs quality-best).
- Package the per-tier training sets + scores + test sets + a README/datasheet;
  push a tagged release and mint the **Zenodo DOI** (currently only metadata
  exists).

**Phase B — Finalization decisions (≈0.5 day, can parallel A)**
- Canonical training set: `train_<tier>_best.fasta` (recommended) vs
  `train_<tier>.fasta`.
- Keep homo-oligomers? (currently kept).
- Keep the CheZOD-1325 parsing-superiority claim? If yes, fetch CheZOD-1325.
- Re-emit `.str` files without methyl wildcards (cosmetic).

**Phase C — Figures (≈2–3 days)**
- Draw Fig 1 workflow + funnel; add α-synuclein panel to Fig 2; recompose Fig 3.

**Phase D — Manuscript rewrite (≈1 week)**
- Reframe to the dataset arc (Section 6); rewrite Methods for LACS + the
  build/mmseqs recipe + CheZOD117 mapping; consolidated dataset-size table; add
  a "leakage-free split for downstream predictor training (see UdonPred)"
  paragraph; LaTeX hygiene (author footnotes, emails, Acknowledgements,
  supplementary, drop orphaned Figure_1.png, fix typos).

**Critical path:** Phase A (commit + Zenodo) → Phase D (manuscript). Figures and
finalization decisions run in parallel.

---

## 9. Loose ends

- Commit the methyl-wildcard revert (uncommitted).
- Move `docs/260520/scripts/*` and the dataset into tracked/released locations.
- Mint Zenodo DOI (needs maintainer to authorise GitHub–Zenodo + a tag).
- ~~Reconcile cluster counts vs the paper (6,071 vs ~7,324 unfiltered).~~
  **Resolved (2026-06):** (a) stage-1 cluster-membership removal added to
  `run_mmseqs_pipeline.py`; (b) the TriZOD test set recreated from the current
  snapshot with a fixed seed (`build_test_set.py`, 344 seq) instead of the
  frozen 2024 set; dataset re-run → reps 5,927 / 5,684 / 4,063 / 1,254; figure
  (`2026-06-22_TriZOD_redundancy_reduction.drawio`) + datasheet + `Article.tex`
  (1,547/1,254; 5,425/4,063; 243/5,927) refreshed to match. The larger
  unfiltered/strict ratio vs the old 7,324 is the newer BMRB snapshot + the more
  aggressive bound filter + exact-seq dedup.
- Deferred expert items (non-blocking): disulfide/thiol-state CB handling, ²H
  isotope correction, ionic-strength relaxation, ambiguity-code filtering.
