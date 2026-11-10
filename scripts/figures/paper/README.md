# Paper figures and numbers

These scripts make the artwork and the numbers of the TriZOD paper:

- `scripts/manuscript/make_numbers.py` writes `numbers.tex` (one `\newcommand` per
  quoted number) and `numbers_index.md`.
- The generators in this folder write the figure PDFs and their data tables.

All of them **read the frozen 2026-10 build** and **never run the scoring pipeline**.
The build is `data/interim/build/`: `parquet-2026-10-simplify/`,
`release_bundle/trizod-dataset-2026-10/` and `rebuild_2026-10.log`.

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

# 2. then the figures (any order)
for g in fig1a_ubiquitin fig1c_scale fig2a_kpaired fig2b_panav fig2c_chezod \
         fig3a_reliability fig3b_conditions fig3c_split fig3d_placeholder \
         supp_s2_gscore_theory supp_s3_filter_losses supp_s4_conditions \
         supp_s5_composition supp_s6_rereferencing supp_s7_replicates \
         supp_s8_replicate_gallery supp_s10_baselines supp_s12_failure_cases \
         supp_s15_shift_companion supp_s16_mad_flag supp_s17_chezod_reproduction; do
    $RUN scripts/figures/paper/$g.py --out-dir "$OUT"
done
```

Run the numbers first. `make_numbers.py` imports the figure modules and calls their
analysis functions, so the prose and the artwork quote the same analysis. It also fills
three caches that the figures reuse: the raw scan, the PANAV loop map and the S6d recovery.

### Arguments

- **`--out-dir` (required).** Every generator takes it; nothing has a default output
  location. PDFs go into it, and data tables go into its `source_data/`, `fig3_data/` and
  `suppfigs_{a,b,c}_data/` sub-folders. Point it at a scratch directory to check a change
  without touching the draft.
- **Data root.** By default this is the checkout that holds `data/interim/build`, found
  through `trizod.paths`: the code checkout, or the first parent that has it, so a git
  worktree uses the main checkout's `data/`. To override it:
  - generators: set `TRIZOD_DATA_ROOT`;
  - `make_numbers.py`: pass `--root`.
- **Previews.** Set `TRIZOD_FIG_PREVIEW_DIR` to also write PNG previews there.

### Caches and tools

- **Caches.** Derived intermediates are cached in the git-ignored `tmp/` of the code
  checkout:
  - `tmp/manuscript_numbers/`
  - `tmp/paper_figures/raw_scan.json`
  - `tmp/paper_figs/` (PANAV loop map, S6d recovery, S8 case search)
  - `tmp/fig1c_mmseqs/`
  - `tmp/fig3c_split/`

  To recompute, delete them or pass `--refresh` (`make_numbers`, `fig3c`, `S8`) or
  `--recompute` (`S6`).
- **MMseqs2** must be on `PATH` when one of these caches is empty: `fig1c`, `fig3c`, `S8`,
  and the leakage and cluster numbers of `make_numbers`.

## What each script makes

Outputs are relative to `--out-dir`. Inputs are relative to the data root. "Parquet" means
the release `trizod_dataset.parquet`, and "tiers" means the bundle's per-tier
`scores/<tier>/scores.json`.

| Script | Makes | Main inputs | Data written |
|---|---|---|---|
| `scripts/manuscript/make_numbers.py` | `numbers.tex`, `numbers_index.md` (in `--out-dir`) | the whole build, CheZOD, PANAV, raw BMRB, `tmp/bmrb_entries`, `docs/dataset/sidechain-coverage.md`, `docs/filtering.md` | registry dump in `--work-dir` |
| `fig1a_ubiquitin.py` | Fig. 1a, ubiquitin Z/G profiles | Parquet, tiers, raw BMRB (ubiquitin entries) | `source_data/fig1a*.csv` |
| `fig1c_scale.py` | Fig. 1c, TriZOD vs CheZOD1325 scale | Parquet, CheZOD1325 + CheZOD117 sequences, MMseqs2 | `source_data/fig1c.csv` |
| `fig2a_kpaired.py` | Fig. 2a, within-residue k test (`--replot` redraws from the CSV) | Parquet | `source_data/fig2a.csv` |
| `fig2b_panav.py` | Fig. 2b, offsets vs PANAV, LACS shares | Parquet, `external/panav_offsets.json`, raw BMRB | `source_data/fig2b_*.csv` |
| `fig2c_chezod.py` | Fig. 2c, MAE vs CheZOD1325 (`--potenci-only PATH`) | CheZOD1325, `interim/chezod_verification/trizod_potenci_only.json`, unfiltered tier | `source_data/fig2c.csv` |
| `fig3a_reliability.py` | Fig. 3a, reliability vs size | Parquet, tiers, train FASTAs | `fig3_data/fig3a*` |
| `fig3b_conditions.py` | Fig. 3b, conditions and flags | Parquet, tiers | `fig3_data/fig3b*` |
| `fig3c_split.py` | Fig. 3c, split integrity and disorder strip | bundle test/train FASTAs, tiers, MMseqs2 | `fig3_data/fig3c*` |
| `fig3d_placeholder.py` | Fig. 3d placeholder cell | none | none |
| `supp_s2_gscore_theory.py` | Suppl. Fig. S2, G-score theory | Parquet, `trizod.scoring` functions | `suppfigs_b_data/S2_summary.json` |
| `supp_s3_filter_losses.py` | Suppl. Fig. S3, filter losses per tier | `rebuild_2026-10.log` | `suppfigs_a_data/S3_filter_losses.csv` |
| `supp_s4_conditions.py` | Suppl. Fig. S4, sample conditions | Parquet, tiers, raw scan | `suppfigs_a_data/S4_*` |
| `supp_s5_composition.py` | Suppl. Fig. S5, composition | Parquet, tiers, raw scan | `suppfigs_a_data/S5_*` |
| `supp_s6_rereferencing.py` | Suppl. Fig. S6, re-referencing (`--recompute`) | Parquet + shifts Parquet, PANAV, `external/bmrb_lacs/` | `suppfigs_c_data/S6*` |
| `supp_s7_replicates.py` | Suppl. Fig. S7, replicate analyses | Parquet, tiers | `suppfigs_b_data/S7_*` |
| `supp_s8_replicate_gallery.py` | Suppl. Fig. S8, replicate gallery (`--refresh`) | Parquet, MMseqs2 | `suppfigs_c_data/S8_*` |
| `supp_s10_baselines.py` | Suppl. Fig. S10, baselines and headroom | Parquet, tiers | `suppfigs_b_data/S10_*` |
| `supp_s12_failure_cases.py` | Suppl. Fig. S12, failure cases | Parquet + shifts Parquet, tiers | `suppfigs_b_data/S12_*` |
| `supp_s15_shift_companion.py` | Suppl. Fig. S15, shift companion table | shifts Parquet, Parquet, tiers | `suppfigs_a_data/S15*` |
| `supp_s16_mad_flag.py` | Suppl. Fig. S16, MAD far-outlier flag | shifts Parquet, Parquet, tiers | `suppfigs_a_data/S16*` |
| `supp_s17_chezod_reproduction.py` | Suppl. Fig. S17, CheZOD reproduction | CheZOD1325, potenci-only run, unfiltered tier | `suppfigs_b_data/S17_*` |

### Modules that are not generators

- `style_paper.py` holds the shared style, paths and loaders, plus `--out-dir` handling.
- `fig3_replicates.py` holds the replicate analyses behind Fig. 3a/b and S7.
- `paper_raw_scan.py` is the raw NMR-STAR scan behind S4/S5. Run it alone to rebuild its
  cache (`--refresh`).

These panels are not made here: Fig. 1b and Suppl. Fig. S1 are TikZ (`fig1b_cartoon.tex`,
`S1.tex` in the draft). Suppl. Figs. S9, S11, S13 and S14 are still placeholders in the
draft.

## Manuscript utilities (`scripts/manuscript/`)

These only read the draft:

- `lint_draft.py <draft>`
- `list_markers.py --draft <draft> [--md out.md]`
- `supp_citation_order.py --draft <draft>`
- `wordcount.py <draft> <scratch-out>`, which needs `texcount`.

All of them can run with plain `python3`.
