# TriZOD Final Pipeline + 2026-05-06 Status Talk — Design

**Date:** 2026-05-05
**Author:** Tobias Senoner
**Talk:** 2026-05-06, 15 min + discussion
**Audience:** TriZOD project meeting (supervisor + collaborators incl. Reid Alderson, Iva Pritisanac)

---

## 1. Goals

1. **Finalize the TriZOD pipeline** by landing the last two outstanding modernization steps:
   - **Step 8** — Leu/Val ambiguous methyl wildcards (CDx/CGx instead of guessed CD1/CD2).
   - **Step 9** — LACS re-referencing integrated as a pre-step in the scoring pipeline, with re-referenced NMR-STAR (`.str`) files emitted as a release artifact.
2. **Regenerate the full TriZOD dataset** with the finalized pipeline (warm caches → ~1-2 h compute).
3. **Answer Reid's two follow-up questions** with publishable figures:
   - **Reid #1 (CSP):** chemical shift perturbations between bound/unbound duplicate pairs.
   - **Reid #2 (αSyn):** G-score before/after re-referencing for BMRB 17665 vs ground-truth 6968.
4. **Commit all uncommitted work** (260422 artifacts, two analysis scripts) and prepare release artifacts (`.zenodo.json`, `CITATION.cff`).
5. **Deliver a 15-minute Typst presentation** covering the finalized pipeline end-to-end plus Reid's two analyses.

---

## 2. Non-goals (explicitly out of scope)

- Full PANAV/CheSPI/BaMORC head-to-head benchmark beyond what's in `scripts/benchmark_rereferencing.py` already.
- A round-trip-perfect NMR-STAR rewrite. We emit a backbone-shifts-only NMR-STAR subset.
- Publication-grade BindBox collaboration plan. (Optional single mention slide only.)
- Touching the POTENCI module beyond what's required by Step 8.
- Releasing the dataset to Zenodo *today*. We add the metadata files and document the workflow; actual upload happens after the talk.

---

## 3. Architecture: where things land

### 3.1 Step 8 — methyl wildcards (parser layer)

- **File:** `trizod/bmrb/bmrb.py` (peptide-shift extraction).
- **Behavior:** When a Leu CD1/CD2 or Val CG1/CG2 chemical shift carries a BMRB ambiguity code indicating non-stereospecific (geminal-partner) assignment, or no ambiguity code is provided, output the wildcard atom name `CDx` / `CGx` instead of the guessed CD1 / CG1. The exact code values are settled in the implementation plan against `tests/reference/` snippets (likely BMRB code 2; we sample real entries to confirm before coding).
- **Stereospecific assignments are preserved** when the ambiguity field explicitly states unique stereospecific assignment (e.g., BMRB 18414).
- **Cache impact:** `tmp/bmrb_entries/*.pkl` is invalidated for entries containing Leu or Val. Re-parse is single-pass, ~30-60 min for 17K entries.
- **Scoring impact:** None for backbone-only scoring (the wildcards live on side-chain methyls; Z-/G-scores use the seven backbone atoms). Wildcards appear in emitted `.str` files for downstream automatic-assignment tools.
- **Test:** Add a unit test using a small synthetic BMRB-style snippet with both ambiguous and stereospecific Leu CD assignments; assert that ambiguous → CDx, stereospecific → CD1/CD2.

### 3.2 Step 9 — LACS pre-correction in scoring (scoring layer)

- **File:** `trizod/scoring/scoring.py`.
- **Behavior:** Before the existing POTENCI-based offset detection, run `lacs.compute_lacs_offsets` on the raw observed shifts and subtract the per-atom LACS offset from the observed array. The existing AIC-based POTENCI/residual offset correction then runs as a second pass on the LACS-corrected shifts (it picks up residual bias not captured by LACS — chiefly the atom types LACS doesn't cover, and any disagreement between LACS's Wishart reference and POTENCI's reference).
- **Atoms:** LACS covers CA, CB, C', HA, H, N. HB is not corrected by LACS (POTENCI/AIC handles HB residual bias).
- **CLI:** Add `--rereference-mode {none,lacs,potenci-only,both}` with **default `both`** (LACS pre-correction → POTENCI/AIC residual). `none` = raw observed shifts, no correction (used in the αSyn case study to produce the "raw" trace). `lacs` = LACS-only. `potenci-only` = current behaviour (legacy / regression). `both` = the new default pipeline.
- **Output bookkeeping:** `compute_scores_row()` returns LACS offsets in addition to the existing POTENCI offsets, both stored per-entry in the JSON output (keys: `lacs_offsets`, `potenci_residual_offsets`).
- **Cache impact:** `tmp/wSCS/*.npz` is invalidated. POTENCI prediction cache (`tmp/potenci/`) is unaffected.
- **Test:** Extend `tests/test_pipeline_regression.py` so it asserts the per-entry JSON has both offset keys; spot-check that for a known mis-referenced entry (17665) the LACS C-atom offset is non-trivial.

### 3.3 `.str` emission (output layer)

- **File:** `trizod/trizod.py` (CLI flag) + a new `trizod/io/str_writer.py` (small module; pynmrstar-based).
- **CLI:** `--emit-str <dir>` flag. When set, after scoring each entry, write a corrected NMR-STAR file containing the **backbone shift table only** (not the full BMRB record).
- **Content of emitted `.str`:**
  - One `_Atom_chem_shift` saveframe with corrected backbone shifts.
  - `_Auxiliary_info` block listing: `LACS_offsets` (per atom), `POTENCI_residual_offsets` (per atom), `re_referenced: yes`, `source_BMRB_id`, `pipeline_version`, `re_referencing_software`.
  - Filename: `<dir>/bmr<id>_rereferenced.str`.
- **Methyl wildcards:** if Step 8 emitted CDx/CGx, those names propagate to the `.str` file unchanged.
- **Test:** Round-trip emit-then-parse with `pynmrstar`; assert the emitted file is valid NMR-STAR and that pulled-back shifts match in-memory shifts to ≤ 1e-6 ppm.

### 3.4 Release metadata (repository layer)

- **`.zenodo.json`** (root): title, creators, keywords, license (MIT to match repo), related identifiers (BMRB), upload type "dataset". Not yet activated for actual deposit.
- **`CITATION.cff`** (root): formal citation block; `version: TriZOD-0.1.0-pipeline`.
- **`README.md` update:** new "Releases & Re-referenced Dataset" section pointing to `data/release/` (local) and "Zenodo DOI: pending" placeholder.
- **`data/release/`** (gitignored): destination for the rerun output (`.str` files + tier JSONs + `manifest.json`).
- **No** GitHub Releases or Zenodo upload today. We *prepare* the metadata so a follow-up commit can flip the switch.

### 3.5 Full pipeline rerun (compute layer)

- **Trigger:** `uv run trizod --input-dir data/bmrb_entries/ --filter-defaults <tier> --emit-str data/release/<tier>/ --output-prefix data/release/<tier>` for each of `unfiltered`, `tolerant`, `moderate`, `strict`.
- **Order:** start with `strict` (smallest output, validates plumbing) before kicking off the larger tiers.
- **Caches:** POTENCI cache is preserved (filter-independent). BMRB pickle cache is invalidated by Step 8; wSCS cache is invalidated by Step 9.
- **Expected wall-clock (warm POTENCI cache, cold BMRB + wSCS):** ~30-60 min reparse, ~30-60 min scoring, ~10-15 min `.str` emit per tier. Aim for `strict` done by 6 PM, the rest unattended overnight.

---

## 4. Reid's analyses

### 4.1 Reid #2 — αSyn case study + top-3 flippers, single small-multiples figure (W6)

A four-panel figure showing per-residue G-score before vs after re-referencing for the αSyn case study and the three other entries with the largest mean |ΔG| in the dataset. This combines what was originally separate ("αSyn case study" + "dataset-wide flip screen") into one named, concrete figure.

- **Layout:** 2×2 grid (or 1×4 row) of panels. Each panel: x-axis = residue position, y-axis (the value axis) = G-score (0-1), two paired traces per panel:
  - "raw" (LACS off) — solid line.
  - "re-referenced" (LACS on, default pipeline) — dashed line.
  - The αSyn panel adds a third trace: BMRB 6968 ground-truth (dotted line).
- **Panel A — αSyn (BMRB 17665, mis-referenced).** Inputs: BMRB 17665 + 6968 (both confirmed present in `data/bmrb_entries/`).
- **Panels B, C, D — top 3 flippers.** Identified post-rerun as the three entries with the largest **mean |ΔG|** across all residues, restricted to entries that pass the `tolerant` filter tier (avoids junk entries dominating). Each top-3 panel shows raw + re-referenced traces only — no ground-truth (none exists). The BMRB ID + protein name (from BMRB metadata) is the panel title.
- **Threshold annotation:** a horizontal line at G = 0.5 in every panel — the boundary between "looks disordered" (≥ 0.5) and "looks structured" (< 0.5) under the standard CheZOD interpretation. Shaded regions where raw and re-referenced disagree on which side of 0.5 they sit.
- **Script:** `scripts/case_study_gscore_flips.py`.
- **Procedure:**
  1. Run scoring on 17665 with `--rereference-mode none` → raw G-scores; with `--rereference-mode both` (default) → re-referenced G-scores. Record LACS offsets; sanity-check against Reid's quoted 2.9 ppm and `data/bmrb_lacs/bmr17665_LACS.str`.
  2. Run scoring on 6968 with default re-ref → ground-truth G-scores. Expected LACS offset ~0.35 ppm.
  3. From the rerun output (or, if rerun not yet done, from `tmp/lacs_comparison_results.pkl` produced by `compare_gscores_lacs.py`), compute per-entry **mean |ΔG|**. Restrict to entries in the `tolerant` baseline JSON. Take the top 3 by mean |ΔG|, excluding 17665 (it goes in panel A).
  4. For each top-3 entry: re-run scoring with `--rereference-mode none` and with `--rereference-mode both` to get the two per-residue G-score traces.
  5. Render the four-panel figure.
- **Output:** `docs/260505/figures/gscore_flips.png`.
- **Headline:** "BMRB 17665 raw → looks helical (consistent with the original helical-tetramer interpretation of the 17665 paper); re-referenced → disordered, matches αSyn ground truth (BMRB 6968). Three other named entries show the same flip pattern, demonstrating αSyn isn't cherry-picked — re-referencing systematically rescues mis-classified disorder calls across the BMRB."

### 4.2 (deleted — merged into 4.1)

The standalone histogram of mean |ΔG| across the dataset is dropped. The top-3 panels in §4.1 give the same "this is systematic, not anecdotal" message in a more concrete, audience-friendly form.

### 4.3 Reid #1 — CSP analysis (W7)

- **Source:** the 581 bound/unbound pairs and 1301 independent pairs already aggregated by `analyse_duplicate_entries.py` (saved per-pair shift differences for backbone atoms).
- **Method 1 (HN/N CSP, Reid's formula):** For each bound/unbound pair and each residue with both H and N, compute `CSP = sqrt(dH^2 + (dN/5)^2)`.
- **Method 2 (multi-atom Δω_RMS, Schumann/Williamson 2008):** For each pair and each residue with ≥2 backbone atoms in common, compute `Δω_RMS = sqrt( mean_atom( (Δω / σ_atom)^2 ) )` where `σ_atom` is the per-atom BMRB standard deviation (from `bmrb.io/ref_info/csstats.php` or hard-coded constants).
- **Threshold (Reid's trimmed-mean recipe):** drop the top 10% of CSP values to avoid outlier bias, compute `mean + SD` on the rest, use that as the "significantly perturbed" threshold.
- **Outputs:**
  - `docs/260505/figures/csp_histogram.png` — histogram of HN/N CSP across all 581 pairs, with the trimmed-mean threshold marked.
  - `docs/260505/figures/csp_interface_example.png` — per-residue CSP bar plot for one named binding pair (candidate: FKBP12 `bmr16925` vs `bmr16931` per the duplicate analysis; sub in if a cleaner pair appears). Shows the threshold line and binding-interface residues highlighted.
- **Script:** extension of `scripts/analyse_duplicate_entries.py` with new `--mode csp` (or new sibling script `scripts/csp_analysis.py` if cleaner — decided during writing-plans).

---

## 5. Talk outline (Typst, 15 min, ~13 slides)

Template: `/Users/tsenoner/Downloads/presentation_template.typ` (touying + metropolis).
File: `docs/260505/talk.typ` (with figures in `docs/260505/figures/`).

Slide budget: ~70 s/slide. Reid's two analyses are slides 9-10, the most expensive in time.

**Hard constraint — no overlap with the 22 April talk.** The 22 April presentation already covered: the LACS-offset violin plot, the boxplot+hexbin G-score change figure (`gscores_lacs_comparison_full.png`), the duplicate-entry table, and the BindBox comparison. **None of those figures or findings are reused** in this talk. Any "re-referencing motivation" or "dataset-wide effect" content must come from fresh figures generated against the finalized rerun (different angle, different visualization, or different summary statistic — not a re-skin of the April plots). BindBox is explicitly out of scope.

```
1.  Title — TriZOD final pipeline + dataset release
2.  TriZOD in one slide — purpose, scale (17,388 BMRB entries), output
3.  What's new since 22 April — bullets: Step 8 wildcards, LACS in pipeline, .str emission, Zenodo workflow
4.  Filter improvements (Steps 4-7) — entry-count deltas across 4 tiers, filter-loss table
5.  Step 8 — methyl wildcards, why it matters downstream (count of Leu/Val ambiguous methyls relabeled)
6.  Re-referencing in the pipeline — architecture diagram (raw → LACS → POTENCI residual → scores), small inset comparing LACS vs POTENCI/AIC offset capture (NEW figure, not the April violin)
7.  Per-tier dataset deltas after re-referencing — table or grouped bar of entry-count change + mean G-score change per tier (NEW; April talk only showed pooled hexbin)
8.  How many entries flip side of the G=0.5 disorder threshold — single-number headline + stacked bar by tier (NEW)
9.  αSyn + top-3 flippers (Reid #2) — 4-panel residue-vs-G-score figure (gscore_flips.png)
10. CSP analysis (Reid #1) — histogram + one named interface example (csp_histogram.png + csp_interface_example.png)
11. Final pipeline architecture — one end-to-end diagram
12. Released artifacts — .str format, tier JSONs, .zenodo.json + CITATION.cff, Zenodo deposit workflow (DOI placeholder)
13. Next steps & open questions — downstream of Step 8, dataset deposit timeline, follow-ups from Reid/Iva
```

Cut order if running over 15 min: slide 8 (flip headline) merges into slide 7 → slide 5 (Step 8 detail) collapses into slide 4 → slides 11+12 merge.

---

## 6. Workstream graph & schedule

```
W1 (parser, ~2h)       ┐
W2 (scoring, ~2h)       ├─→ W5 (rerun, ~2h compute) ─→ updated figures for slides 4,8
W3 (.str emit, ~2h)     ┘
W4 (release meta, ~30m)

W6 (αSyn case study, ~2h)    — can run on existing data once W2 lands
W7 (CSP, ~1.5h)              — can run on existing duplicates output now
W8 (talk in Typst, ~3h)      — skeleton against existing 260422/, swap figures as W5/W6/W7 land
W9 (commits, rolling)        — see §7
```

Dependency order: **W1 || W2 || W3 || W4 → W5 → final figures**. **W6 || W7 || W8** can run in parallel with W5 against existing baselines and be re-rendered post-rerun. Long-pole start time matters more than total hours — kick W1 + W2 + W3 + W7 in parallel first.

Soft stop: **9 PM today**. Hard stop: **11 PM today**. Pipeline rerun runs unattended overnight if it isn't done by 9 PM. Talk content must be locked by 11 PM at the latest.

---

## 7. Commits

```
C1 (now):   chore(docs): commit 260422 presentation, plots, and analysis scripts
            - scripts/analyse_duplicate_entries.py
            - scripts/compare_gscores_lacs.py
            - docs/260422/  (entire directory)
            DELETE docs/template.tex + docs/template.pdf  (stale; Typst is now canonical)

C2 (W1):    feat(bmrb): wildcard naming (CDx/CGx) for ambiguous Leu/Val methyls
C3 (W2):    feat(scoring): integrate LACS pre-correction into scoring pipeline
C4 (W3):    feat(output): emit re-referenced NMR-STAR files via --emit-str
C5 (W4):    chore(release): add .zenodo.json, CITATION.cff, README release section
C6 (W5):    data: regenerate baselines and release artifacts with finalized pipeline
            (note: data/ is gitignored; commit message records the run command + version)
C7 (W6,W7): feat(scripts): αSyn case study + CSP analysis
            + docs/260505/figures/* committed
C8 (W8):    docs(talk): 2026-05-06 status & final-pipeline talk (Typst)
```

Each commit goes through the standard pre-commit gate (`uv run ruff check`, `uv run ruff format --check`, `uv run pytest tests/ -v`).

---

## 8. Risks & cut-list

| ID | Risk | Mitigation / cut |
|----|------|------------------|
| R1 | Step 8 ambiguity-code parsing logic lands incorrect → Leu/Val miscoded | Defer Step 8: cut from C2 + slide 5; LACS-only release. Step 8 lands post-talk. |
| R2 | `pynmrstar`-emitted `.str` rejected by BMRB validators | Emit a backbone-shifts-only NMR-STAR subset documented as such; no full round-trip claim. |
| R3 | αSyn entries missing | **Mitigated**: bmr17665 and bmr6968 confirmed in `data/bmrb_entries/`. |
| R4 | CSP threshold method behaves badly on noisy pairs | Fall back to fixed threshold (mean + 2σ) on all data; document choice on slide. |
| R5 | Rerun exceeds 4 h wall-clock | Run unattended overnight; talk uses pre-rerun numbers + a "fresh data on Zenodo this week" disclaimer. |
| R6 | Talk overruns 15 min | Pre-arranged cut order (see §5). Practice once at 9-10 PM. |

Hard cut-list (in order of removal if compressed): slide 8 (flip headline) merges into slide 7 → Step 8 detail slide 5 → Step 8 implementation itself → release-metadata commit (lands post-talk).

---

## 9. Acceptance criteria

A successful end-of-day looks like:

1. `git log` shows commits C1-C8 (or C1-C8 minus deferred items).
2. `uv run pytest tests/ -v` passes; `ruff` clean.
3. `data/release/strict/` contains `.str` files and a `manifest.json`.
4. `docs/260505/figures/` contains `gscore_flips.png` (αSyn + top-3 flippers, 4 panels), `csp_histogram.png`, `csp_interface_example.png`.
5. `docs/260505/talk.pdf` (or `.typ`-rendered output) compiles cleanly to a 13-slide deck.
6. One end-to-end dry run of the talk completed under 15 min.

---

## 10. Open questions (deferred to writing-plans)

1. ~~CLI verb resolved: `--rereference-mode {none,lacs,potenci-only,both}` with default `both`.~~
2. Should `compute_scores_row()` return a structured offsets dict keyed by atom, or two flat arrays? (Affects JSON schema.)
3. CSP analysis: extend `analyse_duplicate_entries.py` (`--mode csp`), or new sibling script `scripts/csp_analysis.py`? Lean toward sibling for SRP.
4. CSP per-atom σ values: pull live from `bmrb.io/ref_info/csstats.php`, or hard-code Schumann's published values? Hard-code — the values are stable and avoid network at run time.
