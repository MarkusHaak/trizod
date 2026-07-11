# Step 9 — LACS pre-correction integrated into the scoring pipeline

## What changed

LACS re-referencing — previously a standalone module (`trizod/lacs/lacs.py`, validated against 6,692 BMRB-published LACS reports) — is now applied as a **first-pass offset correction** inside the scoring pipeline, before the existing POTENCI/AIC residual offset detection.

A new CLI flag `--rereference-mode {none, lacs, potenci-only, both}` (default **`both`**) selects the strategy:

| mode | behaviour |
|---|---|
| `none` | no correction; raw observed shifts → POTENCI difference → scores |
| `lacs` | LACS pre-correction only |
| `potenci-only` | legacy behaviour; skip LACS, run POTENCI/AIC offset detection |
| `both` (default) | LACS pre-correction → POTENCI/AIC residual on the LACS-corrected shifts |

LACS covers **CA, CB, C′, HA, H, N**. HB is intentionally not corrected by LACS (Wishart random-coil tables don't cover HB; POTENCI/AIC handles HB residual bias).

## Output additions

For every scored entry, the JSON output now includes:

- `off_<atom>` — POTENCI residual offset (existing, atoms = C, CA, CB, H, HA, HB, N).
- `lacs_off_<atom>` — LACS offset (new, same atom set; HB always 0.0).

Both are persisted in the wSCS cache (`tmp/wSCS/<id>_<st>_<ea>_<e>_<mode>.npz`). The cache filename is now keyed on the re-referencing mode so different modes can coexist without contaminating each other.

## Code surface

| File | Change |
|---|---|
| `trizod/scoring/scoring.py` | New `apply_lacs_correction(bbshifts_arr, bbshifts_mask, seq) → (corrected_arr, offsets_dict)` helper; `get_offset_corrected_shifts` rewritten to honour all four modes and return a 10-tuple ending in `lacs_offsets`. |
| `trizod/trizod.py` | New `--rereference-mode` flag plumbed through `compute_scores_row → compute_scores → get_offset_corrected_shifts`. New `lacs_off_<atom>` columns initialised in `fill_row_data` and emitted in JSON output. wSCS cache extended with a `lacs` array. |
| `tests/test_lacs_integration.py` | Helper unit tests (synthetic biased input → recovers ~2 ppm CA offset; empty mask → all-zero offsets). |
| `tests/test_rereference_modes.py` | αSyn 17665 regression: `mode=none` → all-zero LACS offsets; `mode=both` → at least one C/CA/CB offset > 1 ppm (matches Reid's ~2.9 ppm TALOS-N reading). |
| `tests/test_pipeline_regression.py` | Now passes `--rereference-mode potenci-only` so the legacy reference (pre-LACS) stays valid. |

## Talking points for the slide

- **What:** LACS is now a default first-pass step in scoring; users get re-referenced disorder scores out of the box.
- **Why:** ~25% of BMRB entries have systematic referencing errors (Wishart 2003). Without correction, all derived disorder scores for those entries are wrong. The previous TriZOD approach (POTENCI/AIC offset) detects only the *residual* bias after referencing; it can't replace LACS for the largest systematic errors.
- **How verified:**
  - Synthetic test: a 2 ppm CA bias on 75 random-coil residues → LACS recovers ~2 ppm.
  - Real-data test: BMRB 17665 (mis-referenced αSyn) → LACS detects ~2.8 ppm CA/CB offset, matching Reid Alderson's TALOS-N reading of 2.9 ppm.
- **Backwards compatibility:** `--rereference-mode potenci-only` reproduces the old TriZOD behaviour. The legacy regression baseline (`tests/reference/unfiltered.json`) is preserved by passing this flag in CI.

## Commits

- `dced8b1` — `feat(cli): add --rereference-mode flag scaffolding`
- `d5c95f0` — `fixup(scoring): include rereference_mode in wSCS cache key + restyle test`
- `3d6ba85` — `feat(scoring): integrate LACS pre-correction into the scoring pipeline`
- `451ff5b` — `fixup(scoring): exclude HB from LACS offset application`
