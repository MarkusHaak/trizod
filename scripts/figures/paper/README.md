# Paper figures and numbers

These scripts make the artwork and the numbers of the TriZOD paper:

- `scripts/manuscript/make_numbers.py` writes `numbers.tex` (one `\newcommand` per
  quoted number) and `numbers_index.md`.
- The generators in this folder write the figure PDFs and their data tables.

Artwork is drawn at the draft's text width and placed at 1:1: 160 mm for full width
(`style_paper.W_DOUBLE`) and 78 mm for half width (`W_SINGLE`), with all text at 7 pt
(tick labels included), nothing below 6 pt and 8 pt bold panel letters. Colours follow one
meaning per entity in every figure (`style_paper.ENTITY`, `CAT`, `COND`, tier ramp; story
blueprint section 6.8). `make_numbers.py` reads the snapshot's
retrieval date from `BMRB_SNAPSHOT.json` in the raw snapshot directory
(`<root>/data/raw/bmrb_entries/`) and fails without it, so the paper never states a date the
snapshot cannot prove; the file is written when the snapshot is downloaded.

All of them **read one finished build** and **never run the scoring pipeline**. By default
that is the build of record, the 2026-11-11 snapshot rebuild at `trizod-a481315` in
`data/interim/rebuild-2026-11-11/` of the checkout that holds `data/`. Environment variables
select another build (see [Selecting the build](#selecting-the-build)); both the generators
and `make_numbers.py` take the choice from `paper_config.py`.

There is one exception, in Supplementary Fig. S6d. It re-scores 150 sampled chains with the
released `get_offset_corrected_shifts()` in a pyarrow-free subprocess. That happens only
when its cache is missing or `--recompute` is given.

## Running them

Run everything from the code checkout with the `figures` extra and pyarrow:

```bash
DRAFT=/path/to/manuscript-draft          # the LaTeX draft (numbers.tex lives here)
OUT="$DRAFT/Figures/paper"               # figure PDFs + data sub-folders
RUN="uv run --extra figures --with pyarrow python"

# 1. numbers first
$RUN scripts/manuscript/make_numbers.py --out-dir "$DRAFT"

# 2. then the figures (any order): main Figs. 1-5, then the Supplementary Figures
for g in fig1b_record fig1c_scale fig2_composition fig2c_chezod fig2a_kpaired \
         fig1a_ubiquitin fig3b_conditions fig3a_reliability fig4b_pair fig3c_split \
         supp_s10_baselines \
         supp_s2_gscore_theory supp_s3_filter_losses supp_s4_conditions \
         supp_s5_composition supp_s6_rereferencing supp_s7_replicates \
         supp_s8_replicate_gallery supp_s12_failure_cases supp_s15_shift_companion \
         supp_s16_mad_flag supp_s17_chezod_reproduction fig2b_panav; do
    $RUN scripts/figures/paper/$g.py --out-dir "$OUT"
done

# 3. numbers again (some macros read the figures' data tables), then the TikZ pieces
#    (Fig. 1a: fig1a_overview.tex is \input by the draft; Suppl. Fig. S1: compile S1.tex)
```

On the build of record (2026-11-11 snapshot) the whole chain, LaTeX passes included, is
scripted in `reports/story/regen_v2.sh` of the code worktree.

Run the numbers first. `make_numbers.py` imports the figure modules and calls their
analysis functions, so the prose and the artwork quote the same analysis. It also fills
three caches that the figures reuse: the raw scan, the PANAV loop map and the S6d recovery.

### Arguments

- **`--out-dir` (required).** Every generator takes it; nothing has a default output
  location. PDFs go into it, and data tables go into its `source_data/`, `fig3_data/` and
  `suppfigs_{a,b,c}_data/` sub-folders. Point it at a scratch directory to check a change
  without touching the draft.
- **Data root.** By default this is `data/interim/rebuild-2026-11-11/root` of the checkout
  that holds `data/` (the code checkout, or the first parent that has `data/interim/build`,
  so a git worktree uses the main checkout's `data/`). To override it:
  - generators: set `TRIZOD_DATA_ROOT`;
  - `make_numbers.py`: pass `--root` (or set `TRIZOD_DATA_ROOT`).
- **Previews.** Set `TRIZOD_FIG_PREVIEW_DIR` to also write PNG previews there.

### Selecting the build

`paper_config.resolve()` decides the build for the generators (`style_paper.py`) and for
`make_numbers.py`, so the artwork and the prose always describe one build. Every variable is
optional; unset, the build of record is read (`RB` = `<checkout>/data/interim/rebuild-2026-11-11`).

| Variable | What it points at | Default |
|---|---|---|
| `TRIZOD_DATA_ROOT` | root holding `data/raw`, `data/external`, `data/interim/chezod_verification` and `tmp/` (`make_numbers.py`: `--root`) | `$RB/root` |
| `TRIZOD_BUILD_DIR` | build directory: `final_dataset/`, `testset/`, `mmseqs/` | `<root>/data/interim/build` |
| `TRIZOD_RELEASE_DIR` | `trizod_dataset.parquet`, `trizod_shifts.parquet` (and the post-build `disorder_trizod.fasta`, `sidechain-coverage.md`) | `$RB/release` |
| `TRIZOD_BUNDLE` | release bundle: `MANIFEST.json`, `scores/`, `train/`, `test/` | `$TRIZOD_BUILD_DIR/release_bundle/trizod-dataset-2026-11` |
| `TRIZOD_REBUILD_LOG` | build log with the four `print_filter_losses()` reports | `$RB/rebuild_2026-11-11.log` |
| `TRIZOD_PKL_DIR` | the build's BmrbEntry pickle cache (`make_numbers.py` and `supp_s3_filter_losses.py`) | `<root>/tmp/bmrb_entries` |
| `TRIZOD_SIDECHAIN_DOC` | side-chain coverage report (`make_numbers.py` only) | `$TRIZOD_RELEASE_DIR/sidechain-coverage.md` when it exists, else `docs/dataset/sidechain-coverage.md` |
| `TRIZOD_DRAW_DIR` | the 2026-08-11 test-set draw (`_tmp/`, `TriZOD_test_set_clu.tsv`; `make_numbers.py` only). A rebuild re-resolves the pinned set without redrawing it, so the draw stays in the build that drew it | `$TRIZOD_BUILD_DIR/testset` when it holds `_tmp/`, else `<checkout>/data/interim/build/testset` |
| `TRIZOD_CACHE_DIR` | where the caches below go (and `make_numbers.py --work-dir`) | `tmp/` of the code checkout |

The four build variables (`TRIZOD_BUILD_DIR`, `TRIZOD_RELEASE_DIR`, `TRIZOD_BUNDLE`,
`TRIZOD_REBUILD_LOG`) go together: setting some but not all of them stops with an error, so
a partial override never mixes two builds. The release label and pipeline version come
from the bundle's `MANIFEST.json` (`style_paper.RELEASE`, `style_paper.PIPELINE`);
`style_paper.IS_FROZEN` is true only on the frozen 2026-10 build, whose split predates the
rebuild fixes.

### G-score cuts and split values

The three G-score cuts are set in `paper_config.py` only: `G_ORD` (ordered below, 0.15),
`G_DIS` (the binary cut, 0.35) and `G_CONF` (confidently disordered from, 0.70), with the
convention "disordered at G >= cut". `style_paper` re-exports them for the generators;
`make_numbers.py` and `threshold_numbers.py` import them, and `threshold_numbers.py` and
`supp_s20_threshold.py` stop when the calibration framework was run with other values. A
recalibration changes the one line there. The same module names the release's split values:
`test_retired` (from the 2026-11-17 build on, a pinned TriZOD test chain that left the
strict tier) is in `HELD_OUT_SPLITS`, never training and never tested on. Releases from the
2026-11-17 build on also carry `membrane_mimetic_sample`, `temperature_placeholder` and
`high_pressure`; `style_paper.load_main` adds them as all-null to an older release, and
`style_paper.membrane_sample` is the one "membrane-mimetic sample" rule (`membrane_mimetic`
now names classes, a sub-CMC additive included, so `membrane_mimetic.notna()` no longer means
a membrane sample).

Example, the frozen 2026-10 build of the first draft:

```bash
T=/path/to/trizod                                  # the checkout holding data/
export TRIZOD_DATA_ROOT=$T
export TRIZOD_BUILD_DIR=$T/data/interim/build
export TRIZOD_RELEASE_DIR=$TRIZOD_BUILD_DIR/parquet-2026-10-simplify
export TRIZOD_BUNDLE=$TRIZOD_BUILD_DIR/release_bundle/trizod-dataset-2026-10
export TRIZOD_REBUILD_LOG=$TRIZOD_BUILD_DIR/rebuild_2026-10.log
# then the loop above, unchanged
```

### Caches and tools

- **Caches.** Derived intermediates are cached in the git-ignored `tmp/` of the code
  checkout (or `$TRIZOD_CACHE_DIR`):
  - `tmp/manuscript_numbers/`
  - `tmp/paper_figures/` (raw scan, deposited conditions, entry authors)
  - `tmp/paper_figs/` (PANAV loop map, S6d recovery, S8 case search)
  - `tmp/fig1c_mmseqs/`
  - `tmp/fig3c_split/`

  The raw-file scans, the PANAV loop map, the entry authors and every `make_numbers` cache
  are keyed by their inputs (`paper_config.keyed_json`): the snapshot's `BMRB_SNAPSHOT.json`,
  the build's `MANIFEST.json`, the code that computes them and the mmseqs inputs and
  arguments. A cache written for another snapshot, build or code is recomputed, never
  reused. The S6d recovery, the S8 case search, the S3 cosolvent groups, `fig1c_mmseqs/` and
  `fig3c_split/` keep their own keys, so give a new build its own `$TRIZOD_CACHE_DIR` or
  refresh them. To force a recompute, pass `--refresh` (`make_numbers`, `fig2_composition`,
  `fig3c`, `S8`) or `--recompute` (`S6`), or delete the cache.
- **MMseqs2** must be on `PATH` when one of these caches is empty: `fig1c`, `fig3c`, `S8`,
  and the leakage and cluster numbers of `make_numbers`.

## What each script makes

Outputs are relative to `--out-dir`. Inputs are relative to the data root. "Parquet" means
the release `trizod_dataset.parquet`, and "tiers" means the bundle's per-tier
`scores/<tier>/scores.json`.

Figure map of the draft (story blueprint, 12 Nov 2026): five main figures and Table 1.
The script names keep the earlier numbering until the layout is frozen (then one rename
commit); the output names already follow the new numbering.

| Display item | Panels | Script (output) |
|---|---|---|
| Fig. 1 | a pipeline overview | TikZ `fig1a_overview.tex` in the draft (macros from `numbers.tex`) |
| | b one chain record | `fig1b_record.py` (`fig1b.pdf`) |
| | c scale against CheZOD1325 | `fig1c_scale.py` (`fig1c.pdf`) |
| Table 1 | obstacle, remedy, columns | LaTeX only (`sections/table1.tex`) |
| Fig. 2 | a pH x temperature, b annotations per tier, c shifts by atom type (scoring atoms vs other atoms), d repeated measurements | `fig2_composition.py` (`fig2_composition.pdf`) |
| Fig. 3 | a ubiquitin | `fig1a_ubiquitin.py` (`fig3a_ubiquitin.pdf`) |
| | b same residue, different shift count | `fig2a_kpaired.py` (`fig3b_kpaired.pdf`) |
| | c CheZOD reproduction | `fig2c_chezod.py` (`fig3c_chezod.pdf`) |
| Fig. 4 | a conditions, d what strict filtering removes | `fig3b_conditions.py` (`fig4a.pdf`, `fig4d.pdf`) |
| | b one single-study pair | `fig4b_pair.py` (`fig4b.pdf`) |
| | c size against agreement per tier | `fig3a_reliability.py` (`fig4c.pdf`) |
| Fig. 5 | a held-out chains vs training, b disorder per set | `fig3c_split.py` (`fig5a_split.pdf`, `fig5b_disorder.pdf`) |
| | c baselines against the replicate ceiling | `supp_s10_baselines.py` (`fig5c_baselines.pdf`) |

Retired on 12 Nov 2026: Suppl. Fig. S19 (now Fig. 4d) and Suppl. Table T13 (now Table 1).

| Script | Makes | Main inputs | Data written |
|---|---|---|---|
| `scripts/manuscript/make_numbers.py` | `numbers.tex`, `numbers_index.md` (in `--out-dir`) | the whole build, CheZOD, PANAV, raw BMRB, the build's `tmp/bmrb_entries`, `sidechain-coverage.md`, `docs/filtering.md` | registry dump in `--work-dir` |
| `fig1b_record.py` | Fig. 1b (`fig1b.pdf`), anatomy of one chain record: shift, side-chain, k and G tracks plus a key-value card; the chain is chosen by a fixed rule (`select_example`) | Parquet, shifts Parquet, raw-scan cache (optional check that the card's conditions were deposited) | `source_data/fig1b.csv`, `fig1b_rule.csv` |
| `fig1c_scale.py` | Fig. 1c (`fig1c.pdf`), TriZOD vs CheZOD1325 in chains, residues and clusters | Parquet, CheZOD1325 + CheZOD117 sequences, MMseqs2 | `source_data/fig1c.csv` |
| `fig2_composition.py` | Fig. 2a-d (`fig2_composition.pdf`): pH x temperature with tier windows, annotations per tier, shifts by atom type, repeated measurements (`--refresh` re-reads the raw condition rows; a new snapshot rescans by itself, the scan cache is keyed by `BMRB_SNAPSHOT.json`) | Parquet, shifts Parquet, tiers, raw scan | `source_data/fig2{a,b,c,d}*.csv`, `fig2_summary.json` |
| `fig2c_chezod.py` | Fig. 3c (`fig3c_chezod.pdf`), MAE vs CheZOD1325 without and with LACS (`--potenci-only PATH`) | CheZOD1325, a LACS-off run (`--potenci-only`; the paper uses `interim/chezod_verification_2026-11-12/trizod_potenci_only.json`, run on the release code), unfiltered tier | `source_data/fig3c_chezod.csv` |
| `fig2a_kpaired.py` | Fig. 3b (`fig3b_kpaired.pdf`), within-residue shift-count test (`--replot` redraws from the CSV) | Parquet | `source_data/fig3b_kpaired.csv` |
| `fig1a_ubiquitin.py` | Fig. 3a (`fig3a_ubiquitin.pdf`), ubiquitin Z/G profiles of the native-condition chains | Parquet, tiers, raw BMRB (ubiquitin entries) | `source_data/fig3a_ubiquitin*.csv` |
| `fig3b_conditions.py` | Fig. 4a (`fig4a.pdf`), same vs different conditions; Fig. 4d (`fig4d.pdf`), annotations of the chains strict filtering removes | Parquet, tiers, raw scan | `source_data/fig4a.csv`, `fig4d.csv`; `fig3_data/fig3b*` |
| `fig4b_pair.py` | Fig. 4b (`fig4b.pdf`), the top single-study one-factor pair (`fig3_replicates.one_factor_pairs`) | Parquet, raw scan (deposited conditions, through `fig3_replicates`) | `source_data/fig4b_profiles.csv`, `fig4b_pairs.csv` (Suppl. Table T14) |
| `fig3a_reliability.py` | Fig. 4c (`fig4c.pdf`), size against replicate agreement per tier | Parquet, tiers, train FASTAs | `source_data/fig4c.csv`; `fig3_data/fig3a*` |
| `fig3c_split.py` | Fig. 5a (`fig5a_split.pdf`, held-out chains vs training) and Fig. 5b (`fig5b_disorder.pdf`, disorder content per set) | bundle test/train FASTAs, tiers, MMseqs2 | `fig5_data/fig5a_points.csv`, `fig5b_disorder_classes.csv`, `fig5ab_numbers.json` |
| `supp_s10_baselines.py` | Fig. 5c (`fig5c_baselines.pdf`, `draw_main`) and Suppl. Fig. S10 (`S10.pdf`), baselines and headroom | Parquet, tiers | `suppfigs_b_data/S10_*` |
| `fig2b_panav.py` | Suppl. Fig. S18 (`S18.pdf`), offsets vs PANAV, LACS shares | Parquet, `external/panav_offsets.json`, raw BMRB | `source_data/S18_*.csv` |
| `fig3d_placeholder.py` | unused since 11 Nov 2026 | none | none |
| `supp_s2_gscore_theory.py` | Suppl. Fig. S2, G-score theory | Parquet, `trizod.scoring` functions | `suppfigs_b_data/S2_summary.json` |
| `supp_s3_filter_losses.py` | Suppl. Fig. S3, filter losses per tier (cosolvent synonyms pooled) | the build log, the build's pickles (`TRIZOD_PKL_DIR`) | `suppfigs_a_data/S3_filter_losses.csv`, `S3_cosolvent_groups.csv` |
| `supp_s4_conditions.py` | Suppl. Fig. S4, sample conditions (ionic strength, pressure, physical state; pH, temperature and annotations moved to Fig. 2) | Parquet, tiers, raw scan | `suppfigs_a_data/S4_*` |
| `supp_s5_composition.py` | Suppl. Fig. S5, composition | Parquet, tiers, raw scan | `suppfigs_a_data/S5_*` |
| `supp_s6_rereferencing.py` | Suppl. Fig. S6, re-referencing (`--recompute`) | Parquet + shifts Parquet, PANAV, `external/bmrb_lacs/` | `suppfigs_c_data/S6*` |
| `supp_s7_replicates.py` | Suppl. Fig. S7, replicate analyses | Parquet, tiers | `suppfigs_b_data/S7_*` |
| `supp_s8_replicate_gallery.py` | Suppl. Fig. S8, replicate gallery (`--refresh`) | Parquet, MMseqs2 | `suppfigs_c_data/S8_*` |
| `supp_s12_failure_cases.py` | Suppl. Fig. S12, failure cases | Parquet + shifts Parquet, tiers | `suppfigs_b_data/S12_*` |
| `supp_s15_shift_companion.py` | Suppl. Fig. S15, shift companion table | shifts Parquet, Parquet, tiers | `suppfigs_a_data/S15*` |
| `supp_s16_mad_flag.py` | Suppl. Fig. S16, MAD far-outlier flag | shifts Parquet, Parquet, tiers | `suppfigs_a_data/S16*` |
| `supp_s17_chezod_reproduction.py` | Suppl. Fig. S17, CheZOD reproduction | CheZOD1325, potenci-only run, unfiltered tier | `suppfigs_b_data/S17_*` |

### Modules that are not generators

- `style_paper.py` holds the shared style, paths and loaders, plus `--out-dir` handling.
- `fig3_replicates.py` holds the replicate analyses behind Figs. 4a-d, S7 and the
  single-study pairs (`one_factor_pairs`, Fig. 4b and Suppl. Table T14).
- `release_stats.py` counts repeated measurements of exact sequences (Fig. 2d; the R2 and R7
  macros) and checks the ubiquitin selection rule of Fig. 3c.
- `paper_raw_scan.py` is the raw NMR-STAR scan behind Fig. 2a, S4 and S5 and the deposited
  condition flags. Run it alone to rebuild its cache (`--refresh`).

These panels are not made here: Fig. 1a and Suppl. Fig. S1 are TikZ (`fig1a_overview.tex`,
`S1.tex` in the draft's `Figures/paper/`). Suppl. Figs. S9, S11, S13 and S14 are still
placeholders in the draft.

## Manuscript utilities (`scripts/manuscript/`)

These only read the draft:

- `lint_draft.py <draft>`
- `list_markers.py --draft <draft> [--md out.md]`
- `supp_citation_order.py --draft <draft>`
- `wordcount.py <draft> <scratch-out>`, which needs `texcount`.

All of them can run with plain `python3`.
