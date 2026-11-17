# TriZOD Repository — Master Restructuring Plan

**Dateline: 2026-06-25**

**This plan SUPERSEDES `docs/260611/master-plan.md`** and carries forward its
still-valid content (the dataset funnel numbers, the build recipe in Section 3,
the leakage-removal protocol, the figure findings, and the Phase A–D execution
arc). **This remains a DATASET-only paper** — the trained predictor and its
CheZOD/SETH benchmarking belong to the separate **UdonPred** paper. Nothing here
trains or evaluates an ML model. TriZOD's job is to deliver and validate the
*dataset* (plus the pipeline + G-score that produce it) with a leakage-free
split.

Where the 2026-06-11 plan said "move `docs/260520/scripts/*` into tracked
locations" as a loose end, this plan makes that concrete and extends it to the
whole repository: the dataset-build chain, the figure generators, and the CLI
all become tracked, importable, reproducible package code.

> One stale claim from the prior plan is corrected throughout: `docs/260520/`
> and `docs/260611/` scripts **are already git-tracked** (committed in
> `fae181d`/`6bf0b2e` and later). The restructure is therefore a *relocation*,
> not a rescue from untracked oblivion. What remains untracked is the generated
> *data* under `docs/*/data/` (gitignored, regenerable) and the released bundle.

---

## 2. Executive summary

1. **Lift the library out of the CLI god-module.** `trizod/trizod.py` (1,450 L)
   is split into `trizod/pipeline.py` (orchestration), `trizod/cache.py`
   (POTENCI cache), and a thin `trizod/cli/` layer — with backward-compat
   re-export shims so the 5 scripts that import from `trizod.trizod` keep working.
2. **Kill the hidden `global bmrb_entries`** state (read by three pandarallel
   workers) by passing entries explicitly; this is the single riskiest mechanical
   change and gets a dedicated before/after test.
3. **Promote the dataset-build chain into `trizod/dataset/`** — the
   `build_final_dataset → build_test_set → run_mmseqs_pipeline →
   cluster_best_repr → package_release → build_deploy_fasta` chain becomes
   importable, parameterized (no `parents[3]` ROOT hacks), and driven by a CLI
   subcommand. `detect_bound` becomes a shared classifier.
4. **Consolidate figures into `trizod/figures/`** with a shared style/IO module
   and one module per manuscript figure, plus a single
   `regenerate_manuscript_figures` entrypoint that writes straight into the
   manuscript `Figures/` dir.
5. **Adopt Typer + Rich for a multi-subcommand `trizod` CLI** (`score`,
   `dataset`, `figures`, `package`, `cache`), keeping the bare `trizod <flags>`
   and `python -m trizod.trizod` paths working 1:1 so the subprocess-driven tests
   stay green.
6. **Packaging hygiene:** move `matplotlib` to a `[figures]` extra, delete the
   stale `trizod.egg-info/`, reconcile the AGPL/MIT license + version + test-set
   count + author-email metadata mismatches before any Zenodo deposit, and fix
   the `.DS_Store`/gitignore inconsistencies.
7. **Docs consolidation:** archive the 260415/260422/260505/`_planning`/
   `superpowers` notebook iterations under `docs/archive/`, promote the canonical
   dataset spec to `docs/dataset/`, fix `pipeline.md` to document LACS Stage-4c,
   delete the empty `docs/260625/` data side and the stale `pipeline-overview.md`.
8. **Settle the figure set and the binding-dataset scope:** the manuscript gets
   a dedicated **LACS Figure 2** (currently missing entirely); the
   **binding-partner/CSP dataset is a SEPARATE future paper**, with its CSP
   primitives promoted to `trizod/binding/` and one supplementary paragraph in
   this paper. Delete the orphan baby-photo `Figure_1.png`.

---

## 3. Target repository structure (after the restructure)

```
trizod/                                  # repo root
├── pyproject.toml                       # hatchling; matplotlib → [figures] extra; typer added
├── uv.lock
├── LICENSE                              # one license, reconciled (recommend AGPL-3.0-only)
├── README.md
├── CITATION.cff                         # version + license + email aligned
├── .zenodo.json                         # version + license + "TriZOD-344" aligned
├── .gitignore                           # +.DS_Store consistent; docs/260415 rule fixed
│
├── trizod/                              # the importable package
│   ├── __init__.py                      # __version__, META_DATA
│   ├── constants.py                     # BACKBONE_ATOMS, REFINED_WEIGHTS, per-atom sigma, AA maps
│   ├── utils.py                         # (ArgHelpFormatter removed once argparse gone)
│   ├── pipeline.py                      # NEW: find/load BMRB, build df, pre/post filter, compute_scores
│   ├── cache.py                         # NEW: potenci cache key + load/save (re-exported by trizod.py)
│   ├── trizod.py                        # THIN: run_scoring_pipeline() + back-compat re-export shims
│   │
│   ├── bmrb/         __init__.py, bmrb.py            # parser (cleaned: no #breakpoint, get_pressure)
│   ├── potenci/      __init__.py, potenci.py, data/  # random-coil prediction (unchanged)
│   ├── scoring/      __init__.py, scoring.py         # Z/G scoring (dead 'corrected' path removed)
│   ├── lacs/         __init__.py, lacs.py            # LACS re-referencing
│   ├── io/           __init__.py, str_writer.py, fasta.py  # __init__ re-exports; NEW fasta.py
│   │
│   ├── dataset/                         # NEW subpackage (was docs/260520/scripts + docs/260623)
│   │   ├── __init__.py                  # public API: build, test_set, redundancy, package, deploy
│   │   ├── composition.py               # detect_bound + bound/exact-seq dedup + quality_score
│   │   ├── build.py                     # build_final_dataset
│   │   ├── testset.py                   # build_test_set (seeded)
│   │   ├── redundancy.py                # run_mmseqs_pipeline (two-stage leakage removal)
│   │   ├── mmseqs.py                    # COMMON option block, run() wrapper, cluster_tsv_groups
│   │   ├── representatives.py           # cluster_best_repr (quality-best override)
│   │   ├── package_release.py           # Zenodo bundle + MANIFEST + leakage gate
│   │   └── deploy_fasta.py              # UdonPred-handoff FASTA builder
│   │
│   ├── figures/                         # NEW subpackage (figure generators, [figures] extra)
│   │   ├── __init__.py
│   │   ├── style.py                     # shared matplotlib style + data-loading helpers
│   │   ├── fig1_pipeline.py             # dataset funnel + disorder PDFs
│   │   ├── fig2_lacs.py                 # 4-panel LACS effect (was plot_lacs_effect.py)
│   │   ├── fig2_lacs_case_study.py      # αSyn 17665/6968 panel (was case_study_gscore_flips.py)
│   │   ├── fig3_gscore.py              # G-vs-Z, PDFs, DisProt ROC, case studies
│   │   ├── chezod.py                    # CheZOD reproduction + 117→BMRB mapping helpers
│   │   └── supplementary.py             # max_offset_* and aligned-cluster panels
│   │
│   ├── binding/                         # NEW subpackage — SEPARATE future paper (seed only)
│   │   ├── __init__.py
│   │   ├── composition.py               # re-uses dataset.composition.detect_bound classifier
│   │   ├── pairing.py                   # apo/bound pairing, conditions_similar, signal/noise
│   │   └── csp.py                       # per-residue CSP (HN/N + multi-atom), threshold, labels
│   │
│   └── cli/                             # NEW: Typer multi-subcommand CLI
│       ├── __init__.py                  # exposes `app`
│       ├── main.py                      # Typer root app + callback; default → score
│       ├── score.py                     # mirrors today's flat flags 1:1
│       ├── dataset.py                   # build / test-set / redundancy / package / deploy
│       ├── figures.py                   # regenerate manuscript figures
│       └── cache.py                     # potenci cache precompute
│
├── scripts/                            # thin CLIs + validation utilities (kept tracked)
│   ├── precompute_potenci_cache.py     # → relies on editable install (no sys.path hack)
│   ├── filter_impact_report.py
│   ├── benchmark_rereferencing.py      # backs tests/test_lacs.py
│   ├── compare_lacs_bmrb.py            # LACS validation (CLAUDE.md-documented)
│   ├── find_lacs_dominant.py
│   ├── build_dataset.sh                # thin wrapper around `trizod dataset build` (was run_all.sh)
│   ├── validation/                     # CheZOD reproduction (load-bearing for tests)
│   │   ├── build_chezod_test_subset.py # generates tests/reference/chezod_zscores_subset.json
│   │   ├── reproduce_chezod.py
│   │   ├── verify_chezod_parsing.py
│   │   └── chezod_faithful_measure.py
│   ├── figures/                        # thin callers of trizod.figures (manuscript regen)
│   │   ├── regenerate_manuscript_figures.py
│   │   ├── analyze_max_offset.py
│   │   └── compare_gscores_lacs.py
│   └── binding/                        # binding-dataset workstream (future paper)
│       ├── analyse_duplicate_entries.py
│       ├── csp_analysis.py             # thin CLI over trizod.binding.csp
│       └── csp_per_pair_grid.py
│
├── tests/
│   ├── conftest.py
│   ├── reference/                      # committed fixtures (unfiltered.json, chezod subset, …)
│   ├── quick_subset_ids.txt
│   ├── test_potenci.py  test_lacs.py  test_lacs_integration.py  test_str_writer.py
│   ├── test_rereference_modes.py  test_chezod_equality.py
│   ├── test_smoke.py  test_pipeline_regression.py  test_full_dataset_regression.py
│   ├── test_pipeline.py                # NEW: prefilter/postfilter/compute_scores (no data dep)
│   ├── test_filtering.py               # NEW: synthetic-df filter unit tests
│   ├── test_fasta.py                   # NEW: fasta round-trip (dedup-safety)
│   ├── test_cli_imports.py             # NEW: import-smoke for the 5 back-compat symbols
│   └── test_binding.py                 # NEW: FKBP12 interface (res 55,58) + 0.224 threshold
│
└── docs/
    ├── pipeline.md                     # UPDATED: LACS Stage-4c + --rereference-mode
    ├── potenci.md  lacs.md  filtering.md
    ├── dataset/                        # PROMOTED canonical spec (living, manuscript-facing)
    │   ├── dataset-construction.md     # single source of truth for the funnel
    │   ├── datasheet.md                # release-bundle README (self-contained)
    │   └── figure-caption.md           # Fig 1 caption
    ├── binding-dataset.md              # NEW: consolidated binding/CSP concept note
    ├── 260520/                         # KEPT: dataset hub minus promoted spec/scripts/data
    ├── 260611/                         # KEPT: this-plan-superseded master + CheZOD provenance
    ├── 260623/                         # KEPT: deploy-FASTA iteration notes
    ├── 260625/                         # this plan
    └── archive/                        # superseded notebook history
        ├── 260415/  260422/  260505/
        ├── _planning/                  # keep suggested_improvements.md as expert-review record
        ├── superpowers/
        └── 260611/investigate_chezod_mismatches.py
```

---

## 4. Migration plan — ordered, independently-shippable phases

Every phase ends with the **3-gate + CLI smoke**:

```
uv run ruff check trizod/ tests/
uv run ruff format --check trizod/ tests/
uv run pytest tests/ -v
uv run trizod --help        # and: python -m trizod.trizod --help
```

The sequence keeps the repo runnable and tests green at every step.

### Phase 0 — Hygiene + metadata (no code risk; ship first)
**Goal:** clear cruft and fix release-blocking metadata before touching code.
- Delete `trizod.egg-info/` (untracked, stale pre-hatchling artifact).
- Delete on-disk cruft: `scripts/__pycache__/`, `docs/260520/scripts/__pycache__/`,
  `docs/260520/data/mmseqs/_tmp/` (28 MB), `docs/260520/data/release_bundle/trizod-dataset-2026-05/` (122 MB),
  empty `docs/260625/` data artifacts; remove committed `docs/260505/.DS_Store`.
- `.gitignore`: add/normalize `.DS_Store` and `*.egg-info/`; resolve the
  `docs/260415/` rule (recommend `git rm --cached docs/260415/lacs-validation-report.md`
  after confirming its content is in `docs/lacs.md`).
- Reconcile metadata: **license** (recommend AGPL-3.0-only everywhere),
  **version** (single string, recommend `0.2.0`), `.zenodo.json` "TriZOD-348"→"344",
  author email, and the repo URL mismatch (`MarkusHaak`→`tsenoner`).
- **Verify:** `git status` clean of cruft; grep license/version returns one value
  each; 3-gate + CLI smoke pass.

### Phase 1 — Extract the library out of the CLI module
**Goal:** make pipeline logic importable without the CLI; zero behavior change.
- Create `trizod/cache.py` (move `load_potenci_cache`, `save_potenci_cache`,
  `_potenci_cache_key`). Re-export from `trizod.trizod` (`from trizod.cache import *`).
- Create `trizod/pipeline.py` (move `find_bmrb_files`, `load_bmrb_entries`,
  `parse_bmrb_file`, `create_peptide_dataframe`, `fill_row_data`,
  `prefilter_dataframe`, `postfilter_dataframe`, `compute_scores`,
  `compute_scores_row`, `print_filter_losses`). Re-export from `trizod.trizod`.
- Keep `filter_defaults`, `parse_args`, `main`, `output_dataset` in `trizod.trizod`.
- **Verify:** `test_cli_imports.py` confirms the 5 script-import symbols resolve;
  3-gate + CLI smoke pass.

### Phase 2 — Eliminate `global bmrb_entries` (riskiest change)
**Goal:** remove hidden module state; make workers testable.
- Pass the entries explicitly into `fill_row_data`/`create_peptide_dataframe`/
  `compute_scores_row` via `functools.partial` bound before `parallel_apply`.
- Add `test_pipeline.py` running a 2-entry DataFrame before/after to prove the
  closure pickles under pandarallel's fork model.
- Replace bare `except Exception` in `parse_bmrb_file` and the silent
  `ZscoreComputationError` pass in `compute_scores_row` with INFO-level logging.
- **Verify:** new before/after worker test green; full pytest + CLI smoke pass.

### Phase 3 — Dead-code + subpackage hygiene
**Goal:** remove traps and normalize subpackages (no new features).
- Remove the dead `corrected`/`Z_CORRECTION` path: `compute_zscores` `corr`
  branch, the `corrected` choice in `--score-types`, and delete
  `trizod/scoring/pscore_to_zscore_polynomial_coeffs.npz`.
- Clean `bmrb.py`: remove `#breakpoint()` block (L565–574), the no-op
  `get_pressure` stub.
- Populate `trizod/io/__init__.py` to re-export `write_rereferenced_str`.
- Add `trizod/io/fasta.py` (`read_fasta`, `write_fasta`, `count_fasta`,
  `fasta_ids`) + `test_fasta.py` round-trip.
- **Verify:** grep confirms no caller of removed symbols; 3-gate + CLI smoke pass.

### Phase 4 — Typer CLI (keep entrypoint identical)
**Goal:** multi-subcommand CLI with a 1:1-compatible `score` default.
- Add `typer` dep; create `trizod/cli/` with `app = typer.Typer(rich_markup_mode=None)`.
- `score.py` mirrors every current flag (BooleanOptionalAction → `--flag/--no-flag`,
  nargs ranges → `Tuple`/`List`, choices → Enum). The `--filter-defaults` tier
  preset resolves in a callback.
- Bare `trizod <flags>` routes to `score` via `invoke_without_command`.
- Point `[project.scripts] trizod = "trizod.cli.main:app"`; keep
  `python -m trizod.trizod` delegating to the same `run_scoring_pipeline`.
- **Verify (critical):** smoke tests grep `input-dir`, `--rereference-mode`,
  `{none,lacs,potenci-only,both}` — keep `rich_markup_mode=None` so the choices
  render as plain text; `test_emit_str_smoke` flat invocation must pass.

### Phase 5 — Promote the dataset-build chain to `trizod/dataset/`
**Goal:** tracked, importable, reproducible build; surfaced via `trizod dataset`.
- Move the 6 scripts (Section 5 table); replace `ROOT = parents[3]` with a shared
  `resolve_root()` / CLI `--root`; route FASTA/mmseqs helpers through
  `trizod.io.fasta` + `trizod.dataset.mmseqs` (do **not** change the mmseqs option
  string `--alignment-mode 3 --cov-mode 0 -s 7.5 --comp-bias-corr 0 --mask 0`).
- Extract `detect_bound` into `dataset/composition.py`.
- Add `scripts/build_dataset.sh` thin wrapper around `trizod dataset build`.
- **Verify:** run end-to-end and confirm the funnel reproduces (reps
  5,927/5,684/4,063/1,254) before committing; import-smoke test added to pytest;
  3-gate + CLI smoke pass.

### Phase 6 — Consolidate figures + CheZOD validation
**Goal:** one reproducible figure entrypoint; validation scripts tracked + findable.
- Move figure generators to `trizod/figures/` with `style.py`; add
  `scripts/figures/regenerate_manuscript_figures.py` emitting into the manuscript
  `Figures/` dir (documenting the `data/release/<tier>/scores.json` prerequisite).
- Move CheZOD scripts to `scripts/validation/` (keep `build_chezod_test_subset.py`
  tracked — it builds the committed test fixture).
- **Verify:** figures regenerate; `test_chezod_equality.py` still green; 3-gate pass.

### Phase 7 — Docs consolidation + binding seed
**Goal:** settle living docs; seed the binding workstream without scope creep.
- Move 260415/260422/260505/`_planning`/`superpowers` → `docs/archive/`;
  promote canonical spec → `docs/dataset/`; update `pipeline.md` LACS section;
  delete `pipeline-overview.md` after harvesting its mermaid + tier table.
- Write `docs/binding-dataset.md`; promote CSP primitives to `trizod/binding/`
  with `test_binding.py` (FKBP12 res 55,58; 0.224 threshold). Mark it
  out-of-scope for THIS paper.
- Update every reproduce-block path in docs after script relocations.
- **Verify:** grep `docs/2604`/`docs/2605` references fixed; 3-gate + CLI smoke pass.

---

## 5. Consolidated MOVES table (deduped across all reports)

| From | To | Why |
|---|---|---|
| `trizod/trizod.py` `load/save_potenci_cache`, `_potenci_cache_key` | `trizod/cache.py` (re-exported) | Library logic imported by 5 scripts from the CLI module |
| `trizod/trizod.py` `find_bmrb_files`, `load_bmrb_entries`, `parse_bmrb_file`, `create_peptide_dataframe`, `fill_row_data`, `prefilter/postfilter_dataframe`, `compute_scores`, `compute_scores_row`, `print_filter_losses` | `trizod/pipeline.py` (re-exported) | Make orchestration importable + testable independent of CLI |
| `trizod/trizod.py` `parse_args`, `filter_defaults`, `main` body | `trizod/cli/main.py` + `score.py`; `main()`→`run_scoring_pipeline()` | Isolate CLI; keep `python -m trizod.trizod` + entrypoint working |
| `[project.scripts] trizod = "trizod.trizod:main"` | `trizod = "trizod.cli.main:app"` | Route console script through Typer app; subcommands reachable |
| `docs/260520/scripts/build_final_dataset.py` | `trizod/dataset/build.py` (+ `detect_bound`→`composition.py`) | Core build step; tracked + importable; classifier shared with binding |
| `docs/260520/scripts/build_test_set.py` | `trizod/dataset/testset.py` | Seeded leakage-free split; the UdonPred link |
| `docs/260520/scripts/run_mmseqs_pipeline.py` | `trizod/dataset/redundancy.py` (+ `mmseqs.py`) | Two-stage leakage removal; dedup mmseqs helpers |
| `docs/260520/scripts/cluster_best_repr.py` | `trizod/dataset/representatives.py` | Quality-best representative override |
| `docs/260520/scripts/package_release.py` | `trizod/dataset/package_release.py` | Zenodo bundle + MANIFEST + leakage gate |
| `docs/260623/scripts/build_deploy_fasta.py` | `trizod/dataset/deploy_fasta.py` | UdonPred-handoff FASTA |
| `docs/260520/scripts/run_all.sh` | `scripts/build_dataset.sh` + `trizod dataset build` | Promote orchestration; thin wrapper during transition |
| `docs/260520/scripts/plot_lacs_effect.py` | `trizod/figures/fig2_lacs.py` | Manuscript Fig 2 (LACS) generator |
| `scripts/case_study_gscore_flips.py` | `trizod/figures/fig2_lacs_case_study.py` | αSyn 17665/6968 Fig 2 panel; repoint cache import to `trizod.cache` |
| `docs/260611/scripts/reproduce_chezod.py` | `trizod/figures/chezod.py` helpers + `scripts/validation/reproduce_chezod.py` | CheZOD reproduction + 117→BMRB mapping (Fig 3 + leakage target) |
| `docs/260520/scripts/analyze_max_offset.py` | `scripts/figures/analyze_max_offset.py` | Supplementary max-offset figures |
| `scripts/compare_gscores_lacs.py` | `scripts/figures/compare_gscores_lacs.py` | Produces the `tmp/lacs_comparison_results.pkl` Fig 2 consumes |
| `scripts/talk_figures.py` | `docs/archive/260505/talk_figures.py` | Obsolete 6-May talk figures |
| `docs/260415/create_pptx.py` | `docs/archive/260415/create_pptx.py` (or delete) | Obsolete 16-Apr PPTX; untracked |
| `docs/260611/scripts/build_chezod_test_subset.py` | `scripts/validation/build_chezod_test_subset.py` | Builds committed `tests/reference/chezod_zscores_subset.json` |
| `docs/260611/scripts/verify_chezod_parsing.py`, `chezod_faithful_measure.py` | `scripts/validation/` | CheZOD parsing/faithfulness validation |
| `docs/260611/scripts/investigate_chezod_mismatches.py` | `docs/archive/260611/` | One-off deep-dive; archive |
| `scripts/csp_analysis.py` | `trizod/binding/csp.py` core + `scripts/binding/csp_analysis.py` CLI | Seed binding-site dataset (future paper) |
| `scripts/analyse_duplicate_entries.py` | `trizod/binding/pairing.py` core + `scripts/binding/` CLI | apo/bound pairing primitives |
| `scripts/csp_per_pair_grid.py` | `scripts/binding/csp_per_pair_grid.py` | Figure caller; import from `trizod.binding` not sys.path hack |
| `docs/260520/dataset-construction.md`, `datasheet.md`, `figure-caption.md` | `docs/dataset/` | Promote canonical spec out of dated folder (living reference) |
| `docs/260415/`, `260422/`, `260505/`, `_planning/`, `superpowers/` | `docs/archive/…` | Superseded notebook history |
| `docs/pipeline-overview.md` | delete (harvest mermaid+tier table into `pipeline.md`/`filtering.md`) | Stale pre-LACS duplicate, gitignored |
| `trizod/scoring/pscore_to_zscore_polynomial_coeffs.npz` | delete | Dead data for the always-raising `corr=True` path |
| `trizod.egg-info/` | delete | Stale pre-hatchling setuptools artifact |
| `matplotlib` in `[project].dependencies` | `[project.optional-dependencies].figures` | Imported by zero core modules; figure-only |
| `docs/260520/data/release_bundle/trizod-dataset-2026-05/` (122 MB) | delete | Superseded by 2026-06 bundle (cited everywhere) |
| `docs/260520/data/mmseqs/_tmp/` (28 MB) | delete | mmseqs scratch; regenerated each run |
| `publication/manuscript/Figures/Figure_1.png` | delete (manuscript repo) | Orphan baby photo, referenced nowhere in `Article.tex` |

> `publication/` files are **NOT moved** by this repo's restructure (separate
> private repo). The two manuscript deletions above are recorded as manuscript-repo
> commits done in the publication workspace, not here.

---

## 6. Code-stability plan

**Refactors (Phases 1–4):**
- `trizod.trizod` → `pipeline.py` + `cache.py` + `cli/` with re-export shims.
- `global bmrb_entries` → explicit-argument passing (dedicated before/after test).
- `main()` → `run_scoring_pipeline(config)` callable, shared by Typer `score` and
  `python -m trizod.trizod`.
- `print_filter_losses` (~120 L) → split formatting from "unique-loss"
  recomputation; cover with synthetic-df tests.
- `fill_row_data` (~130 L) → reduce the `Found`-exception control flow; unit test.

**Dead-code removal:**
- `corrected`/`Z_CORRECTION` path + `pscore_to_zscore_polynomial_coeffs.npz`.
- `bmrb.py` `#breakpoint()` block + no-op `get_pressure`.
- Commented dead line `args = argparse.Namespace(...)` (trizod.py:349).
- `ArgHelpFormatter` once argparse is gone (Typer shows defaults natively).

**Exception visibility:**
- `parse_bmrb_file` and `compute_scores_row` log the failing file/entry at INFO
  (keep `trizod.bmrb`/`trizod.scoring` at CRITICAL so only orchestration drops
  surface) — needed to audit the dataset funnel.

**Test gaps to fill (so the restructure verifies without the untracked `data/`):**
- `test_pipeline.py` — `prefilter/postfilter/compute_scores` on a synthetic df.
- `test_filtering.py` — filter-column logic on a synthetic df.
- `test_fasta.py` — round-trip (guards the dedup helper unification).
- `test_cli_imports.py` — import-smoke for the 5 back-compat symbols.
- `test_binding.py` — FKBP12 interface (res 55,58) + 0.224 ppm threshold.
- worker before/after test for the `global`-removal (Phase 2).

---

## 7. CLI plan

**Chosen framework: Typer (built on Click) + Rich.** Rationale: it removes the
brittle hand-rolled two-phase `init_parser → parse_known_args →
filter_defaults.loc → full parser` machinery (a tier preset becomes a
callback-resolved default), gives type-hint-driven subcommands and free
`--help`/colors/completion, and the console-script entry is one line. Click would
be the same result with a heavier decorator-per-flag port; status-quo argparse
keeps the fragile two-phase code. Resolving the prior open question: **swap to
Typer**, but only after the library extraction (Phases 1–3) so the swap touches
no pipeline logic.

**Subcommand surface (flat naming):**
```
trizod <flags>            # bare = score (invoke_without_command), back-compat
trizod score <flags>      # mirrors every current flag 1:1
trizod dataset build      # the run_all.sh chain (score→build→mmseqs→best→package)
trizod dataset test-set
trizod dataset redundancy
trizod dataset package
trizod dataset deploy
trizod figures            # regenerate manuscript figures
trizod cache potenci      # precompute POTENCI cache
trizod package            # Zenodo bundle
```

**Migration that keeps the entrypoint working:**
- `[project.scripts] trizod = "trizod.cli.main:app"`.
- Bare `trizod <flags>` routes to `score` via `invoke_without_command` (preserves
  `scripts/build_dataset.sh` / muscle memory).
- `python -m trizod.trizod` keeps delegating to the same `run_scoring_pipeline`.
- `rich_markup_mode=None` so smoke-test substring greps (`input-dir`,
  `--rereference-mode`, `{none,lacs,potenci-only,both}`) still match.
- Use `typing.Optional/List/Tuple` (not PEP-604 unions) to keep the 3.9 floor.

---

## 8. Packaging + hygiene cleanup

| Item | Action | How to verify |
|---|---|---|
| matplotlib | move to `[project.optional-dependencies].figures` (core = numpy, pandas, pynmrstar, scipy, tqdm, pandarallel) | `uv sync` then `uv run trizod --help` (works w/o matplotlib); `uv run pytest` 57 pass; wheel `Requires-Dist` shows matplotlib only under `extra == 'figures'` |
| typer | add `typer>=0.12` to core deps | `uv sync`; `uv run trizod --help` |
| python-pptx | declare in a `slides`/`dataset` extra only IF pptx code is promoted (currently archived) | `uv sync --extra …` resolves |
| `trizod.egg-info/` | delete (untracked, gitignored) | `uv build --wheel` still succeeds; `uv run trizod --help` works |
| license | unify to AGPL-3.0-only across `pyproject`, `LICENSE`, `CITATION.cff`, `.zenodo.json` (recommend AGPL — matches LICENSE+pyproject) | grep all four → one license string |
| version | single string (recommend `0.2.0`) in `pyproject`/`CITATION.cff`/`.zenodo.json` | `uv build` wheel name; `importlib.metadata.version('trizod')` |
| test-set count | `.zenodo.json` "TriZOD-348" → "344" | re-read file |
| author email | align `pyproject` (tum.de) ↔ `CITATION.cff` (gmail) | re-read |
| repo URL | align `pyproject [project.urls]` (MarkusHaak) ↔ `CITATION.cff` (tsenoner) | re-read |
| `.DS_Store` | ensure ignored everywhere; remove committed `docs/260505/.DS_Store` | `git ls-files` shows none tracked |
| `docs/260415/` gitignore | `git rm --cached lacs-validation-report.md` (content already in `docs/lacs.md`) OR un-ignore siblings | `git ls-files docs/260415` matches intent |
| build backend | **leave hatchling flat-layout as-is** (auto-discovers all subpackages + data) | `uv build --wheel` lists all subpackages |
| tmp/ (2.5 GB) | leave gitignored; ensure Zenodo software deposit uses `git archive`/sdist, not the working tree | inspect sdist contents |

---

## 9. Figure set (settled)

**Decision: 4 main figures + Table 1.** This reconciles the master-plan's
3-figure arc with the live `Article.tex` 4-figure set by promoting **LACS to its
own main figure** (the biggest novelty since the original report; currently
entirely absent from the manuscript). The aligned-cluster per-protein examples
move to supplementary.

| Fig | Content | Readiness | Regen script | Source |
|---|---|---|---|---|
| **Table 1** | 4-tier filter matrix | done in `Article.tex` | n/a (from `filter_defaults`) | `trizod/cli/score.py` |
| **Fig 1** | dataset-construction TikZ + funnel + disorder PDFs | TikZ reproducible; PDF panel needs regen vs 2026-06 build | `trizod/figures/fig1_pipeline.py` | `2026-06-22_TriZOD_redundancy_reduction.tex` |
| **Fig 2 (NEW)** | LACS: 4-panel effect + αSyn 17665/6968 panel | both halves reproducible in-repo; **must be composited + wired into Article.tex** | `trizod/figures/fig2_lacs.py` + `fig2_lacs_case_study.py` | `plot_lacs_effect.py` + `case_study_gscore_flips.py` |
| **Fig 3** | G-vs-Z synthetic + correlation | PNG exists, **no in-repo generator** (legacy `trizod_publication` repo) | `trizod/figures/fig3_gscore.py` | re-derive |
| **Fig 4** | DisProt validation: ROC + mean-acc | PNG exists, **no in-repo generator** | `trizod/figures/fig3_gscore.py` | re-derive |
| **Supp** | per-protein examples (Prion/Ubiquitin/Cytochrome C), max-offset, PDFs | PNGs exist, generators partly missing | `trizod/figures/supplementary.py` | re-derive |

**Reproducible figure entrypoint:** `scripts/figures/regenerate_manuscript_figures.py`
(or a `trizod figures` subcommand) emits every main figure straight into
`publication/manuscript/Figures/` with the manuscript filenames, failing loudly
if `data/release/<tier>/scores.json` is absent. The 8 May-8 sharelatex bitmaps
(ROC, disorder PDFs, 3 case studies, G-vs-Z, synthetic, mean_acc) have **no
in-repo generator** and may be stale vs the 5,927/5,684/4,063/1,254 reps — they
must be re-derived **after** the methyl-wildcard revert is committed to avoid a
third snapshot mismatch.

**Verdict on `publication/manuscript/Figures/Figure_1.png`:** it is a **1.0 MB
photograph of a baby**, dated May 8, referenced nowhere in `Article.tex` — an
accidental co-author Overleaf upload. **Delete it** from the manuscript repo
(confirm with the co-author first; Overleaf round-trips could re-add it). It must
not ship with the submission.

---

## 10. Docs consolidation plan

**Living docs going forward** (the active layer):
- `docs/dataset/{dataset-construction.md (canonical funnel), datasheet.md,
  figure-caption.md}` — single source of truth for the dataset; `datasheet.md`
  stays self-contained for the Zenodo bundle but cites construction.md as upstream.
- `docs/pipeline.md` (UPDATED with LACS Stage-4c + `--rereference-mode`),
  `potenci.md`, `lacs.md`, `filtering.md`.
- `docs/260611/master-plan.md` (kept as superseded-by-this-plan history) +
  the 3 CheZOD provenance docs.
- `docs/binding-dataset.md` (NEW concept note for the future paper).
- `docs/260625/restructure-master-plan.md` (this plan).

**Archive** → `docs/archive/`: `260415/`, `260422/`, `260505/`, `_planning/`
(keep `suggested_improvements.md` as the expert-review record), `superpowers/`,
`260611/investigate_chezod_mismatches.py`.

**Delete:** `pipeline-overview.md` (after harvesting mermaid + tier table), the
empty `docs/260625/` artifacts, the 2026-05 bundle, mmseqs `_tmp/`.

**Experiments+results → manuscript mapping:**
| Experiment / doc | Manuscript slot |
|---|---|
| `docs/dataset/dataset-construction.md` funnel | Fig 1 + Table 1 + Methods (build recipe) |
| `lacs-effect-on-gscores.md` + LACS validation | Fig 2 + Results §2 (LACS not redundant) |
| `max-offset-analysis.md` | Supplementary (offset distribution) |
| CheZOD reproduction (`260611/chezod-*.md`) | Methods (CheZOD reproduction + 117→BMRB mapping) |
| `alphasyn-case-study.md` (archived) | Fig 2 αSyn panel |
| G-score vs Z-score, DisProt | Fig 3 + Fig 4 + Results §4/§5 |
| `duplicate-entry-analysis.md`, `csp-analysis.md` (archived) | `docs/binding-dataset.md` (future paper) + one supplementary paragraph |

---

## 11. Binding-partner / interaction-site dataset track

**The idea:** the disorder pipeline already DETECTS bound complexes
(`detect_bound`: >1 distinct entity, OR non-polymer/ligand, OR nucleic, OR metal)
only to DROP ~27% (4,558/16,963) of entries. Repurpose that signal: pair apo vs
bound BMRB entries of the same sequence under matched conditions and compute
per-residue chemical-shift perturbation (CSP) to label binding-interface residues.

**What already exists (committed):** `scripts/analyse_duplicate_entries.py` +
`scripts/csp_analysis.py` + `scripts/csp_per_pair_grid.py` produced **581 apo/bound
pairs, 61,063 residue CSPs, 528 per-pair interface plots**, a validated **FKBP12**
interface example (res 55,58 recovered without a structure), Reid Alderson's
CSP = √(dH² + (dN/5)²) with a 0.224 ppm trimmed-mean+SD threshold, and a **noise
floor** (1,301 same-state pairs: CA 1.6×, CB 1.7×, N 2.1×, H 2.0× over noise).

**Feasibility verdict: HIGHLY FEASIBLE — roughly two-thirds built — but OUT OF
SCOPE for this dataset paper.** Keep it a **separate future paper** (or short data
note); at most one supplementary paragraph here noting the discarded bound entries
are a reusable interaction-mapping resource. This protects the priority-1 Zenodo
release and the disorder-only narrative.

**Method sketch (for the future paper):**
1. Promote CSP primitives to `trizod/binding/` (`composition.py` re-uses
   `detect_bound`; `pairing.py`; `csp.py`) with `test_binding.py`.
2. Generalize pairing to **protein-protein** complexes (currently only
   ligand/nucleic) — the largest, most valuable, currently-discarded class.
3. Apply **LACS re-referencing to both apo and bound** shifts before CSP (the
   scripts use raw shifts today, so some "binding" CSP is a referencing artefact).
4. Implement the **Schumann-Williamson multi-atom Δω-RMS** CSP (the 13C signal is
   as strong as HN/N) with per-atom σ in `constants.py`.
5. Emit per-residue binding labels (binary + continuous CSP + partner identity
   from `Entity.name`/`db_links`/`Assembly.organic_ligands`) mirroring the
   disorder dataset's shape; add a datasheet; validate vs DisProt binding regions.

**Where it belongs:** **future paper** (coordinate authorship with Reid Alderson).
Seed code lives in `trizod/binding/`; concept note in `docs/binding-dataset.md`;
one supplementary paragraph in this manuscript.

---

## 12. Outstanding ANALYSIS TODOs (prioritized)

1. **[P1]** Run the full 3-gate + CLI smoke after each migration phase; never
   leave the repo red between phases.
2. **[P1]** Verify the `global bmrb_entries` removal survives pandarallel (Phase 2
   before/after worker test) — the single riskiest mechanical change.
3. **[P1]** Re-run the dataset funnel after Phase 5 and confirm reps
   5,927/5,684/4,063/1,254 reproduce before committing the moves.
4. **[P1]** Commit the methyl-wildcard revert (uncommitted working tree per
   master-plan §8) **before** regenerating any `.str`-derived figure.
5. **[P2]** Composite Fig 2 (LACS 4-panel + αSyn) and wire into `Article.tex`
   (new `\label`, fix downstream `\ref`).
6. **[P2]** Re-derive the 8 non-reproducible Fig 3/4/Supp PNGs against the 2026-06
   build; update captions/numbers in lockstep if values shift.
7. **[P2]** Confirm CheZOD-1325 presence (`data/chezod/protein_nmr_1325/allseqs1325.txt`)
   — `build_test_set.py` + the validation scripts need it; master-plan §5 says the
   suite lacks it. Resolve the hidden dependency or document it.
8. **[P2]** Reconcile the CheZOD parsing-superiority claim status (chezod-data.md
   "not yet executed" vs master-plan Phase B "decision").
9. **[P3]** Unify the divergent `read_fasta`/`count_fasta`/`fasta_ids` copies via
   `trizod/io/fasta.py` + round-trip test (guards retained-ID drift).
10. **[P3]** Quantify the realised binding-dataset size across all tiers + all
    bound classes (future paper scoping).
11. **[P3]** Mint the Zenodo DOI (needs maintainer to authorise GitHub–Zenodo + tag)
    after metadata reconciliation.

---

## 13. Open questions for the user (with recommended defaults)

1. **License: AGPL-3.0-only or MIT?** — *Recommend AGPL-3.0-only* (matches the
   `LICENSE` file + `pyproject`; flip `CITATION.cff`/`.zenodo.json` to match).
2. **Release version string?** — *Recommend `0.2.0`* across all three metadata files.
3. **Canonical training FASTA: `train_<tier>_best.fasta` or `train_<tier>.fasta`?**
   — *Recommend `_best`* (quality-best override; `package_release.py`/`deploy_fasta.py`
   already default to it; datasheet calls it canonical).
4. **CLI framework: Typer or stay argparse?** — *Recommend Typer* (after library
   extraction; preserves the entrypoint + flag surface 1:1).
5. **Subcommand naming: flat (`trizod dataset build`) vs deeper nesting?** —
   *Recommend flat-with-one-group* (`trizod dataset <step>`, everything else flat).
6. **Does LACS get its own main figure?** — *Recommend YES, Fig 2* (it is the
   biggest novelty and is currently missing entirely from the manuscript).
7. **Keep `fig:bmrb_examples` (Prion/Ubiquitin/Cytochrome) as main or supplementary?**
   — *Recommend supplementary* (per master-plan §6).
8. **Re-derive the stale May-8 PNGs against the 2026-06 build, or accept them?** —
   *Recommend re-derive* after the methyl-wildcard revert (internal consistency).
9. **Keep the CheZOD-1325 parsing-superiority claim?** — *Recommend keep only if
   CheZOD-1325 is fetched + reproducible*; otherwise soften to the CheZOD-117 mapping.
10. **Binding-dataset scope?** — *Recommend separate future paper* + one
    supplementary paragraph; seed code in `trizod/binding/` now.
11. **Does `python -m trizod.trizod` stay a forever-public entry?** — *Recommend
    keep the shim* until tests migrate to invoke the Typer app directly.
12. **Bump Python floor to 3.10?** — *Recommend keep 3.9* (lock resolves it; use
    `typing.Optional/List` in CLI signatures).
13. **mmseqs2 as declared optional dep vs documented external binary?** —
    *Recommend documented external binary* (not pip-installable; document an
    acquisition path alongside `panav.jar`).

---

## 14. Risk-verification corrections (adversarial pass — these OVERRIDE the above)

Sections 1–13 were synthesized *before* the adversarial risk check. The check ran
against the live tree; the following corrections take precedence where they
conflict. **`safe_to_proceed: true`** — the phased shape is sound; apply these
fixes inside the named phases.

### Factual corrections to claims above
- **License is ALREADY `AGPL-3.0-only`** in `pyproject.toml` and `LICENSE` — there
  is *no* AGPL-vs-MIT mismatch to reconcile *there*. Only `CITATION.cff` /
  `.zenodo.json` may differ; check just those two. (Open Q1 stands, but the scope
  is narrower than Section 8 implies.)
- **Version is a consistent `0.0.1`**, not a cross-file mismatch. Bumping to
  `0.2.0` is a *decision* (Open Q2), not a reconciliation.
- **Repo URL in `pyproject` IS `MarkusHaak`** (real). Whether to repoint to
  `tsenoner` is a decision, not a typo fix.
- **CheZOD-1325 IS present on disk** (`data/chezod/protein_nmr_1325/allseqs1325.txt`)
  — TODO P2 #7's "suite lacks it" worry is moot on this machine; just document the
  dependency.
- **Dataset-build data IS present** (`docs/260520/data/{final_dataset,mmseqs,testset,
  release_bundle}`) — so the Phase-5 "reproduce-and-diff the funnel" verification is
  actually executable, not aspirational.

### Phase 0 — targets that don't exist as described (de-scope these)
- **No tracked `.DS_Store` anywhere** (`git ls-files` is clean) — drop "remove
  committed `docs/260505/.DS_Store`".
- **`.gitignore` already has `*.egg-info/` (L4) and `.DS_Store` (L47)** — the
  "add/normalize" step is mostly a no-op; verify, don't re-add.
- **`docs/pipeline-overview.md` is untracked** — its removal is a plain `rm`, not
  `git rm`.
- **`docs/260415/lacs-validation-report.md` IS tracked** despite `docs/260415/`
  being gitignored — pick ONE fate (archive-as-tracked **or** `git rm --cached`),
  not both.

### Phase 2 — MISSING from the plan (blocking for correctness)
- **`scripts/filter_impact_report.py:44-53` mutates the global being removed**
  (`import trizod.trizod as _trizod_mod; _trizod_mod.bmrb_entries = global_entries`).
  Removing `global bmrb_entries` makes this a silent no-op → wrong/empty output.
  **Must** be refactored to the new explicit-entries API in Phase 2 and added to
  its checklist + verified end-to-end.

### Phase 4 — the Typer mitigation in §7 is TECHNICALLY WRONG
- `tests/test_smoke.py:38` asserts the **exact** substring
  `{none,lacs,potenci-only,both}` (argparse's brace syntax). Typer/Click render
  Enum choices as `[none|lacs|potenci-only|both]` — **`rich_markup_mode=None` does
  NOT make Click emit argparse's `{...}` form** (context7-confirmed). **Rewrite
  that assertion** (assert each token, or the `[a|b|...]` form) in the SAME commit
  that repoints the entrypoint, so the gate never goes red.
- **`typer>=0.12` uncapped breaks the Python 3.9 floor**: Typer 0.24.0 (2026-02)
  dropped 3.9. Either pin **`typer>=0.12,<0.24`** (keep 3.9) or bump
  `requires-python>=3.10` + ruff target. Run `uv sync` on a 3.9 env before
  committing Phase 4.
- **`python -m trizod.trizod` is load-bearing for 3 subprocess tests**
  (`test_smoke.test_emit_str_smoke`, `test_pipeline_regression`,
  `test_full_dataset_regression`) via the `if __name__=="__main__": main()` guard
  (trizod.py:1449-1450) with the full flag set (`--input-dir --output-prefix
  --filter-defaults --cache-dir --emit-str --processes --no-progress`). Keep that
  guard + flag surface intact; add a test asserting `-m` still parses every flag
  before repointing the console script.

### Phase 5 — define the data-root (UNSPECIFIED, will break)
- The six dataset scripts compute `ROOT = Path(__file__).resolve().parents[3]`
  (`build_final_dataset.py:47`, `build_test_set.py:35`, `run_mmseqs_pipeline.py:39`,
  `cluster_best_repr.py:35`, `package_release.py:36`, `build_deploy_fasta.py:58`)
  and both **read and write** `ROOT/docs/260520/data/...`. Moving them into
  `trizod/dataset/` changes `parents[3]` depth (resolves *above* the repo) **and**
  the plan keeps `docs/260520/data/` while emptying it of scripts. **Decide the
  canonical work-dir** (explicit `--work-dir`/`resolve_root()` param, e.g.
  `data/dataset/`) and thread it through all six — do NOT rely on relative depth.
  Then re-run and **diff outputs against the existing `docs/260520/data` artifacts**
  (IDs + order) before deleting the old copies.

### Phase 3 / Phase 6 — extra precision
- **Dead-code removal must NOT touch the LIVE AIC offset path** in `scoring.py`
  (`std_corrected`/`corrected_arr`/`weighted_diffs_corrected` are real). ONLY the
  `corr=True` branch (`scoring.py:162-168`, raises) and the `--score-types
  corrected` choice (`trizod.py:289`, L1089-1092) are dead. Easy to over-delete.
- **FASTA unification needs an ID-preservation/order test**, not just round-trip:
  `build_test_set.read_fasta` returns `dict`, `run_mmseqs.fasta_ids`/`count_fasta`
  return `list`/`int`, `deploy.read_fasta_ids` returns `list`. The leakage math
  depends on exact ID retention + order — a subtly different parser changes which
  sequences get dropped.
- **`scripts/csp_per_pair_grid.py:26` imports 5 symbols** (`compute_pair_csp`,
  `find_pairs`, `load_baseline`, `shifts_for`, `trimmed_mean_threshold`) from
  `csp_analysis` — these split across BOTH `trizod/binding/csp.py` AND
  `pairing.py`. Map each symbol to its target module before the move (the §5 MOVES
  row implies a single source).

### Phase 7 — tracking-status flip
- **Moving gitignored `docs/_planning/` + `docs/260415/` into the un-ignored
  `docs/archive/` makes them committable.** The plan *wants*
  `suggested_improvements.md` tracked — good — but this could sweep other
  `_planning` notes into git. Either mirror the ignore rules under `docs/archive/`
  or `git add` only the intended record files; review `git status` before commit.

### Scripts the plan never mentions (assign a fate)
- `scripts/compare_offsets_real_data.py`, `scripts/download_lacs_reports.py`,
  `scripts/fetch_panav_bmrb.py` — none appear in §5 MOVES or §3's `scripts/` tree.
  They import `trizod.*` / reference `panav.jar` / LACS reports. Decide keep /
  archive per script (default: keep under `scripts/` as LACS/PANAV utilities).

### Docs that document the old layout (update in lockstep)
- `README.md` + the checked-in `CLAUDE.md` (Scripts + Caching sections) list
  script paths and `tmp/` cache dirs explicitly; **update both** after relocations.
- ~10 doc reproduce-blocks reference `docs/260520/scripts/...` / `run_all.sh`
  (incl. the bundle-facing `datasheet.md`). Make Phase 7 verification a grep-gate:
  `grep -rn 'docs/260520/scripts\|docs/260611/scripts\|docs/260623/scripts\|run_all.sh' docs/`
  returns only intentional historical mentions.

### One real blocker to gate on (not a stop-start blocker)
- **No test guards the dataset-build chain or the figure generators today.** So
  Phases 5 and 6 must **not be committed** until the funnel is re-run end-to-end and
  outputs are diffed against the present `docs/260520/data` artifacts — otherwise a
  silent path/parser regression ships into the Zenodo dataset.
