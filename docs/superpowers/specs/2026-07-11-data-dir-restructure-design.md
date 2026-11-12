# Data directory restructure (Phase 1) — design

- **Date:** 2026-07-11
- **Status:** proposed (awaiting review)
- **Scope:** Consolidate all dataset data under a Cookiecutter-Data-Science–style
  top-level `data/`, remove all *generated* data from `docs/`, and centralize
  data-path resolution in a single module.

## Goal

Today, build **inputs** live under top-level `data/` while build **outputs** are
scattered under `docs/260520|260611|260623|260625/data/`, and ~40 path strings are
hard-coded across the code/scripts/tests. This makes `docs/` a data dump and every
move a multi-file edit.

Establish the field-standard layout (raw / external / interim / processed, per the
deep-research verdict) so that: data flows one direction, `raw/` is immutable,
everything else is regenerable, the Zenodo deposit is a frozen versioned bundle, and
`docs/` holds no dataset data.

## Non-goals (explicitly deferred)

- **Phase 2 docs cleanup** — relocating/promoting the leftover `.md` files in
  `docs/260520|260611|260623|260625/` and deleting the emptied dirs. Only their
  `data/` subdirs move here.
- Moving `tmp/` (ephemeral caches) — it already matches the recommended top-level,
  gitignored cache location; unchanged.
- Re-uploading Zenodo. The published v0.2.0 is frozen; we only relocate the local
  copy of the deposit bundle.

## Target layout

```
data/                          # gitignored (/data/)
  raw/
    bmrb_entries/              # raw BMRB NMR-STAR source (~3.1 GB) — set read-only
  external/                    # third-party reference / benchmark sets
    chezod/                    # CheZOD-1325 reference
    chezod117/                 # CheZOD117 benchmark  (was data/2024-05-09)
    bmrb_lacs/                 # BMRB precomputed LACS reports
    panav_offsets.json         # PANAV-tool comparison offsets
  interim/                     # regenerable intermediates (safe to delete + rebuild)
    scored/                    # per-entry scored release, per tier  (was data/release)
    build/                     # dataset-build work_dir  (was docs/260520/data)
    chezod_verification/       # (was docs/260611/data/chezod_verification)
    filter_impact/             # per-filter analysis
    baseline/                  # per-tier baselines
    lacs_comparison.npz
  processed/                   # final canonical dataset
    trizod_dataset.parquet     # canonical single-file dataset (build output)
    deploy/                    # UdonPred handoff FASTA  (was docs/260623/data/deploy)
  release/                     # frozen, versioned Zenodo deposit bundles
    trizod-dataset-v0.2.0/     # parquet + README + LICENSE + CITATION + zenodo.json
                               #   + CHECKSUMS + UPLOAD  (was docs/260625/data/zenodo)
  README.md                    # describes this layout
tmp/                           # unchanged — ephemeral caches (bmrb pkl, potenci, wSCS)
```

## Migration map (from → to)

| Current | New | Class |
|---|---|---|
| `data/bmrb_entries/` | `data/raw/bmrb_entries/` | raw (then read-only) |
| `data/chezod/` | `data/external/chezod/` | external |
| `data/2024-05-09/` | `data/external/chezod117/` | external |
| `data/bmrb_lacs/` | `data/external/bmrb_lacs/` | external |
| `data/panav_offsets.json` | `data/external/panav_offsets.json` | external |
| `data/release/` | `data/interim/scored/` | interim (**name flip**) |
| `docs/260520/data/` (final_dataset, mmseqs, testset, release_bundle) | `data/interim/build/` | interim |
| `docs/260611/data/chezod_verification/` | `data/interim/chezod_verification/` | interim |
| `data/filter_impact/` | `data/interim/filter_impact/` | interim |
| `data/baseline/` | `data/interim/baseline/` | interim |
| `data/lacs_comparison.npz` | `data/interim/lacs_comparison.npz` | interim |
| `docs/260623/data/deploy/` | `data/processed/deploy/` | processed |
| `docs/260625/data/zenodo/` | `data/release/trizod-dataset-v0.2.0/` | release |
| *(build output)* | `data/processed/trizod_dataset.parquet` | processed |
| `tmp/**` | *(unchanged)* | cache |

**Name flip:** `data/release/` currently means "per-entry scored inputs"; it becomes
`data/interim/scored/`, and `data/release/` is repurposed to mean the published
Zenodo bundle. This is deliberate.

## Path centralization — `trizod/paths.py`

New module: the single source of truth for repo + data paths. Exposes the roots
(`RAW`, `EXTERNAL`, `INTERIM`, `PROCESSED`, `RELEASE`) and well-known files
(`RAW_BMRB`, `EXTERNAL_CHEZOD`, `EXTERNAL_CHEZOD117`, `EXTERNAL_BMRB_LACS`,
`EXTERNAL_PANAV`, `INTERIM_SCORED`, `INTERIM_BASELINE`, …), all resolved from
`repo_root()`.

- `trizod/dataset/paths.py` keeps its `resolve_paths(work_dir=…)` contract but sources
  its defaults from `trizod/paths.py` (`work_dir` default → `INTERIM/"build"`, `scored`
  input → `INTERIM/"scored"`, `chezod117` → `EXTERNAL/"chezod117"/…`, `chezod1325_txt`
  → `EXTERNAL/"chezod"/…`, `deploy_out` → `PROCESSED/"deploy"/…`). `pkl_dir` (tmp) and
  `bundle_readme` (a `.md` doc under `docs/260520/`, Phase-2 territory) unchanged.
- Scripts/figures/tests replace hard-coded `default=Path("data/…")` with references to
  `trizod.paths` constants, so future moves are one-file edits.

## Files to update (blast radius ≈ 40 refs)

- **1 central:** `trizod/dataset/paths.py` (repoint via `trizod/paths.py`).
- **`trizod/` code:** `dataset/{build,deploy_fasta,redundancy,representatives,package_release,testset}.py`
  (help strings), `figures/{fig2_lacs,fig2_lacs_case_study}.py`, `dataset/__init__.py`
  (docstring).
- **`scripts/`:** `build_dataset.sh`, `compare_lacs_bmrb.py`, `compare_offsets_real_data.py`,
  `fetch_panav_bmrb.py`, `download_lacs_reports.py`, `filter_impact_report.py`,
  `precompute_potenci_cache.py`, `find_lacs_dominant.py`, `analyse_duplicate_entries.py`,
  `csp_analysis.py`, `csp_per_pair_grid.py`, `figures/*`, `validation/*`.
- **`tests/`:** `conftest.py`, `test_full_dataset_regression.py`, `test_chezod_equality.py`
  (skip-reason strings + any hard-coded fixture paths).
- **docs/config:** `README.md`, `CLAUDE.md` (Data section), `.gitignore`, `data/README.md`.
- **`.gitignore`:** drop the four `docs/260*/data/` rules (now covered by `/data/`); keep `/data/`.

## Execution order

1. Add `trizod/paths.py`; wire `trizod/dataset/paths.py` to it.
2. `mkdir` new tiers; `mv` each dir/file per the migration map (all gitignored → git sees nothing).
3. Update every reference (grouped by file); update `.gitignore`, `README.md`, `CLAUDE.md`, `data/README.md`.
4. `chmod -R a-w data/raw/bmrb_entries` (enforce raw immutability).
5. Verify (below). Commit.

## Verification

- `uv run ruff check` + `ruff format --check`.
- `uv run pytest tests/ -v` (pipeline/regression/chezod tests exercise the moved
  `data/raw/bmrb_entries`, `data/interim/baseline`, `data/external/chezod` paths).
- Dataset-build smoke: `trizod dataset build --help` etc. resolve; a dry `resolve_paths()`
  points at the new locations.
- Re-run `scripts/build_parquet_dataset.py --in-bundle data/interim/build/release_bundle/… --out data/processed/trizod_dataset.parquet` and diff SHA-256 against the published parquet (`9ee7191a…`) to prove the move + repoint is lossless.
- `git status` shows only tracked code/doc edits (no data), confirming ignores are correct.

## Risks & rollback

- **Missed reference** → a script/test breaks. Mitigation: the grep-derived list above +
  full `pytest` run + a repo-wide `grep -rn "data/\(bmrb_entries\|release\|…\)"` after edits
  to confirm zero stragglers.
- **Moves are `mv` on one filesystem** (instant, reversible); data is gitignored so no
  history rewrite. Rollback = `mv` back + `git checkout` the code edits.
- The published Zenodo v0.2.0 is untouched (frozen remotely); only the local copy moves.
