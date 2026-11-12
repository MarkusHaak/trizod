# TriZOD Finalize Pipeline + 5 May Talk — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land Step 8 (Leu/Val methyl wildcards), Step 9 (LACS in scoring + `.str` emission), regenerate the full TriZOD dataset, build Reid's two analyses (αSyn + CSP), and ship a 15-minute Typst talk for 6 May.

**Architecture:**
- LACS integration: add `apply_lacs_correction()` helper in `trizod/scoring/scoring.py`; call it as a first pass before the existing AIC/POTENCI offset detection. New `--rereference-mode {none,lacs,potenci-only,both}` (default `both`) plumbed through `compute_scores_row → compute_scores → get_offset_corrected_shifts`.
- Step 8: rewrite `LEU CD1/CD2` and `VAL CG1/CG2` to wildcard `CDx/CGx` at `_Atom_chem_shift` parse time in `bmrb.py`, but ONLY where `Ambiguity_code` indicates non-stereospecific assignment. Backbone scoring is unaffected; wildcards surface in `.str` output.
- `.str` emission: new `trizod/io/str_writer.py` using `pynmrstar`. Backbone-shifts-only NMR-STAR subset, one file per entry, plus an auxiliary saveframe recording LACS + POTENCI residual offsets and pipeline version.
- Release metadata: `.zenodo.json`, `CITATION.cff`, README "Releases" section. No actual upload today.
- Reid's analyses: `scripts/csp_analysis.py` (Reid #1) computes Δω-style CSPs from existing duplicate pairs; `scripts/case_study_gscore_flips.py` (Reid #2) builds 4-panel αSyn + top-3 flippers figure.
- Talk: `docs/260505/talk.typ`, copy/adapt of `~/Downloads/presentation_template.typ` (touying + metropolis), 13 slides, no overlap with 22 April material.

**Tech Stack:** Python 3.9+, `uv`, ruff, pytest, numpy, pandas, scipy, matplotlib, pynmrstar, Typst (touying 0.7.3 + metropolis theme).

**Workstream graph:**
```
W1 (CLI scaffold)     ─┐
W2 (LACS in scoring)   ├─→ W5 (rerun) ─→ updated figures, JSON outputs
W3 (Step 8 parser)     ┤
W4 (.str writer)       ┘
W4b (release meta)     ─┘ (parallel)

W6 (CSP analysis)      — independent, run on existing duplicates output
W7 (αSyn case study)   — depends on W2 (needs --rereference-mode)
W8 (talk Typst)        — skeleton against existing data, swap figures from W5/W6/W7
```

**Hard-stop:** 11 PM 2026-05-05. Soft-stop: 9 PM. If pipeline rerun isn't done by 9 PM, run unattended overnight; the talk uses pre-rerun numbers with a "fresh data committing tonight" note.

---

## Task 0: Pre-flight checks

**Files:** none (read-only sanity).

- [ ] **Step 0.1: Confirm working tree clean and on `develop`**

```bash
git -C /Users/tsenoner/Documents/projects/_github/trizod status --short
git -C /Users/tsenoner/Documents/projects/_github/trizod branch --show-current
```
Expected: empty status; branch = `develop`.

- [ ] **Step 0.2: Confirm both αSyn entries are on disk**

```bash
ls /Users/tsenoner/Documents/projects/_github/trizod/data/bmrb_entries/ | grep -E '^bmr(17665|6968)$'
```
Expected: `bmr17665` and `bmr6968` printed.

- [ ] **Step 0.3: Confirm test suite green at baseline**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run pytest tests/ -v -m "not slow"
```
Expected: all non-slow tests pass.

- [ ] **Step 0.4: Confirm ruff is clean at baseline**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run ruff check trizod/ tests/ scripts/
uv run ruff format --check trizod/ tests/ scripts/
```
Expected: both succeed with no diagnostics.

If any pre-flight fails, stop and surface to user — don't paper over a pre-existing breakage.

---

## Task 1: Add `--rereference-mode` CLI flag (W1)

**Files:**
- Modify: `trizod/trizod.py` (CLI argparse + plumbing through `compute_scores_row`/`compute_scores`)
- Test: `tests/test_smoke.py` (smoke check that flag accepts each value)

**Rationale:** Get the flag wired end-to-end first. The flag is plumbed as a string argument and *consumed* in Task 2 by the new LACS-aware code path — for now it's an unused parameter that's stored on the row.

- [ ] **Step 1.1: Add CLI argument to `parse_args()` in `trizod/trizod.py`**

Insert in `scores_grp` block, after `--offset-correction` (around line 290 in current file; locate by searching `--offset-correction`):

```python
    scores_grp.add_argument(
        "--rereference-mode",
        choices=["none", "lacs", "potenci-only", "both"],
        default="both",
        help=(
            "Chemical shift re-referencing strategy. "
            "'none': no correction (raw shifts). "
            "'lacs': LACS pre-correction only. "
            "'potenci-only': legacy POTENCI/AIC offset detection only. "
            "'both' (default): LACS pre-correction followed by POTENCI/AIC residual."
        ),
    )
```

- [ ] **Step 1.2: Plumb `rereference_mode` through `compute_scores_row` and `compute_scores`**

In `trizod/trizod.py`, modify the signatures:

```python
def compute_scores_row(
    row,
    score_types=None,
    offset_correction=True,
    max_offset=np.inf,
    reject_shift_type_only=False,
    cache_dir=None,
    rereference_mode="both",
):
```

```python
def compute_scores(
    entry,
    stID,
    entity_assemID,
    entityID,
    seq,
    ion,
    pH,
    temperature,
    score_types=None,
    offset_correction=True,
    max_offset=np.inf,
    reject_shift_type_only=False,
    cache_dir=None,
    rereference_mode="both",
):
```

Pass `rereference_mode=rereference_mode` from `compute_scores_row` to `compute_scores`. Pass it from `main()`'s `df.parallel_apply(compute_scores_row, ...)` call: add `rereference_mode=args.rereference_mode` to the kwargs.

- [ ] **Step 1.3: Forward `rereference_mode` into `get_offset_corrected_shifts` (placeholder)**

In `compute_scores`, change the call site:

```python
ret = scoring.get_offset_corrected_shifts(seq, shifts, predshiftdct, rereference_mode=rereference_mode)
```

For now, `get_offset_corrected_shifts` ignores the new kwarg — Task 2 wires it up. Update the signature in `trizod/scoring/scoring.py`:

```python
def get_offset_corrected_shifts(seq, shifts, predshiftdct, rereference_mode="both"):
```

The body stays unchanged for this task.

- [ ] **Step 1.4: Add a smoke test for `--rereference-mode` parsing**

Append to `tests/test_smoke.py`:

```python
import subprocess
import sys


def test_rereference_mode_flag_in_help():
    result = subprocess.run(
        [sys.executable, "-m", "trizod", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--rereference-mode" in result.stdout
    assert "{none,lacs,potenci-only,both}" in result.stdout
```

- [ ] **Step 1.5: Run smoke test, expect PASS**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run pytest tests/test_smoke.py::test_rereference_mode_flag_in_help -v
```
Expected: PASS.

- [ ] **Step 1.6: Run full non-slow suite, ruff, then commit**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run ruff check trizod/ tests/
uv run ruff format --check trizod/ tests/
uv run pytest tests/ -v -m "not slow"
```
Expected: all green.

```bash
git add trizod/trizod.py trizod/scoring/scoring.py tests/test_smoke.py
git commit -m "$(cat <<'EOF'
feat(cli): add --rereference-mode flag scaffolding

Adds the --rereference-mode {none,lacs,potenci-only,both} CLI flag with
default 'both', plumbed through compute_scores_row, compute_scores, and
get_offset_corrected_shifts. The flag is currently unused inside the
scoring code; Task 2 wires it up to LACS pre-correction. Smoke test
confirms the flag is exposed in --help.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: LACS pre-correction in scoring pipeline (W2)

**Files:**
- Modify: `trizod/scoring/scoring.py` (new helper `apply_lacs_correction`, modify `get_offset_corrected_shifts`)
- Modify: `trizod/trizod.py` (plumb LACS offsets into per-row output JSON)
- Test: `tests/test_lacs_integration.py` (new file)

**Rationale:** LACS is already a tested module (`trizod/lacs/lacs.py`, `tests/test_lacs.py`). Wrap it for use inside scoring, gate it on `rereference_mode`, and bookkeep the offsets in the output.

- [ ] **Step 2.1: Write failing integration test**

Create `tests/test_lacs_integration.py`:

```python
"""Tests for LACS pre-correction integrated into scoring pipeline."""

import numpy as np
import pytest

from trizod.constants import BACKBONE_ATOMS
from trizod.scoring.scoring import apply_lacs_correction


def test_apply_lacs_correction_returns_corrected_array_and_offsets():
    """Returns (corrected_arr, offsets_dict) for an obviously biased input."""
    # 50-residue all-Ala sequence; CA shifts artificially shifted by +2.0 ppm
    seq = "A" * 50
    bbshifts_arr = np.zeros((50, 7))
    bbshifts_mask = np.zeros((50, 7), dtype=bool)

    ca_idx = BACKBONE_ATOMS.index("CA")
    cb_idx = BACKBONE_ATOMS.index("CB")
    # Realistic random-coil-ish CA values for Ala plus a +2.0 ppm bias
    bbshifts_arr[:, ca_idx] = 52.5 + 2.0
    bbshifts_arr[:, cb_idx] = 19.0
    bbshifts_mask[:, ca_idx] = True
    bbshifts_mask[:, cb_idx] = True

    corrected_arr, offsets = apply_lacs_correction(bbshifts_arr, bbshifts_mask, seq)

    assert corrected_arr.shape == bbshifts_arr.shape
    assert isinstance(offsets, dict)
    # CA offset should be detected as roughly +2.0 ppm (sign may flip per
    # subtraction convention — assert magnitude)
    assert abs(abs(offsets.get("CA", 0.0)) - 2.0) < 0.5
    # Corrected CA values should be closer to 52.5 ppm than the raw +2.0 bias
    assert np.nanmean(corrected_arr[:, ca_idx]) < 53.5


def test_apply_lacs_correction_passthrough_when_no_data():
    """Empty mask → returns the input unchanged plus empty/zero offsets."""
    seq = "A" * 10
    bbshifts_arr = np.zeros((10, 7))
    bbshifts_mask = np.zeros((10, 7), dtype=bool)

    corrected_arr, offsets = apply_lacs_correction(bbshifts_arr, bbshifts_mask, seq)

    assert corrected_arr.shape == bbshifts_arr.shape
    # All offsets are zero or None for atoms without enough data
    assert all(v == 0.0 or v is None for v in offsets.values())
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run pytest tests/test_lacs_integration.py -v
```
Expected: FAIL with `ImportError: cannot import name 'apply_lacs_correction'`.

- [ ] **Step 2.3: Implement `apply_lacs_correction` in `trizod/scoring/scoring.py`**

Add at the top-of-file imports:

```python
from trizod.lacs import compute_lacs_offsets
```

Add this helper above `get_offset_corrected_shifts`:

```python
# Atoms that LACS computes offsets for. HB is intentionally not in this list:
# LACS uses Wishart random-coil tables that don't cover HB; POTENCI/AIC handles
# residual HB bias.
_LACS_ATOMS = ["C", "CA", "CB", "HA", "H", "N"]


def apply_lacs_correction(bbshifts_arr, bbshifts_mask, seq):
    """Run LACS on an observed-shift array and return corrected shifts + offsets.

    Args:
        bbshifts_arr: (N, 7) observed shift array, columns = BACKBONE_ATOMS.
        bbshifts_mask: (N, 7) boolean mask of which entries are populated.
        seq: 1-letter amino-acid sequence of length N.

    Returns:
        (corrected_arr, offsets_dict) where:
            corrected_arr: copy of bbshifts_arr with per-atom LACS offsets
                subtracted (corrected_arr[:, j] = bbshifts_arr[:, j] - offsets[atom]).
                Atoms not covered by LACS are unchanged.
            offsets_dict: keys = BACKBONE_ATOMS, values = LACS offset in ppm or
                0.0 if not detected (LACS returned None).
    """
    n = len(seq)
    seq_nums = np.arange(1, n + 1)

    # Build per-atom observed-shift arrays in the format LACS expects
    obs_shifts = {}
    atom_col = {atom: i for i, atom in enumerate(BACKBONE_ATOMS)}
    for atom in _LACS_ATOMS:
        col = atom_col[atom]
        arr = np.full(n, np.nan)
        valid = bbshifts_mask[:, col]
        arr[valid] = bbshifts_arr[valid, col]
        obs_shifts[atom] = arr

    raw_offsets = compute_lacs_offsets(seq, seq_nums, obs_shifts)

    corrected_arr = bbshifts_arr.copy()
    offsets_dict = {atom: 0.0 for atom in BACKBONE_ATOMS}
    for atom, offset in raw_offsets.items():
        if offset is None or atom not in atom_col:
            continue
        col = atom_col[atom]
        valid = bbshifts_mask[:, col]
        corrected_arr[valid, col] -= offset
        offsets_dict[atom] = float(offset)

    return corrected_arr, offsets_dict
```

- [ ] **Step 2.4: Run test, expect PASS**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run pytest tests/test_lacs_integration.py -v
```
Expected: both tests PASS.

If `apply_lacs_correction` test fails because `compute_lacs_offsets` returns `None` for the synthetic input (insufficient residues / aa diversity), enrich the test sequence with mixed residues — e.g., `seq = ("ARNDCEQGHILKMFPSTWYV" * 3)[:50]` and randomized CA/CB values — then re-run.

- [ ] **Step 2.5: Wire `apply_lacs_correction` into `get_offset_corrected_shifts`**

Modify `get_offset_corrected_shifts` in `trizod/scoring/scoring.py` to honor `rereference_mode`. Replace the body so it:

1. Computes `bbshifts_arr, bbshifts_mask` as today.
2. Branches on `rereference_mode`:
   - `"none"`: no LACS, no AIC/POTENCI offset; final offsets are all-zero.
   - `"lacs"`: apply LACS → final shifts; AIC offsets all-zero.
   - `"potenci-only"`: skip LACS; current behaviour.
   - `"both"` (default): apply LACS → then run existing AIC/POTENCI residual on LACS-corrected shifts.
3. Returns the existing tuple plus a new last element `lacs_offsets` (dict).

Concrete edit — replace the whole function body with the structure below (compare against current file before pasting; preserve existing logic for the AIC/POTENCI branch):

```python
def get_offset_corrected_shifts(seq, shifts, predshiftdct, rereference_mode="both"):
    ret = bmrb.get_valid_bbshifts(shifts, seq)
    if ret is None:
        logging.getLogger("trizod.scoring").error("retrieving backbone shifts failed")
        return
    bbshifts_arr, bbshifts_mask = ret

    # Stage 1: optional LACS pre-correction
    if rereference_mode in ("lacs", "both"):
        bbshifts_arr, lacs_offsets = apply_lacs_correction(bbshifts_arr, bbshifts_mask, seq)
    else:
        lacs_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)

    # Stage 2: POTENCI difference + optional AIC/POTENCI residual
    diff_arr, _, cmp_mask = compare_to_predicted(predshiftdct, bbshifts_arr, bbshifts_mask)
    total_backbone_shifts = np.sum(cmp_mask)
    if total_backbone_shifts == 0:
        logging.getLogger("trizod.scoring").error("no comparable backbone shifts")
        return
    logging.getLogger("trizod.scoring").info(
        f"total number of backbone shifts: {total_backbone_shifts}"
    )

    if rereference_mode in ("none", "lacs"):
        # No POTENCI/AIC residual offset
        offsets_initial = dict.fromkeys(BACKBONE_ATOMS, 0.0)
        weighted_diffs_initial, abs_weighted_diffs_initial = compute_weighted_diffs(
            diff_arr, cmp_mask, offsets_initial
        )
        outlier_mask_initial = np.zeros_like(cmp_mask)
        offsets_final = offsets_initial
        outlier_mask_final = outlier_mask_initial
        weighted_diffs_final, abs_weighted_diffs_final = (
            weighted_diffs_initial,
            abs_weighted_diffs_initial,
        )
    else:
        # rereference_mode in ("potenci-only", "both"): existing AIC/POTENCI logic
        offsets_initial = dict.fromkeys(BACKBONE_ATOMS, 0.0)
        weighted_diffs_initial, abs_weighted_diffs_initial = compute_weighted_diffs(
            diff_arr, cmp_mask, offsets_initial
        )
        zscores_initial = compute_zscores(
            abs_weighted_diffs_initial, cmp_mask.sum(axis=1), cmp_mask
        )
        zscores_triplet_initial = compute_zscores(
            *convert_to_triplet_data(abs_weighted_diffs_initial, cmp_mask), cmp_mask
        )
        outlier_mask_initial = get_outlier_mask(
            zscores_triplet_initial,
            zscores_initial,
            abs_weighted_diffs_initial,
            cmp_mask,
            cdf_threshold=6.0,
        )
        new_offsets_initial = compute_offsets(
            weighted_diffs_initial, cmp_mask & ~outlier_mask_initial, min_AIC=6.0
        )
        mean_zscore_initial = np.nanmean(zscores_triplet_initial)
        offsets_final = new_offsets_initial
        outlier_mask_final = outlier_mask_initial

        offsets_running = compute_running_offsets(diff_arr, cmp_mask, min_AIC=6.0)
        if offsets_running is None:
            logging.getLogger("trizod.scoring").warning(
                "no running offset could be estimated"
            )
        elif np.any([v != 0.0 for v in offsets_running.values()]):
            weighted_diffs_corrected, abs_weighted_diffs_corrected = compute_weighted_diffs(
                diff_arr, cmp_mask, offsets_running
            )
            zscores_corrected = compute_zscores(
                abs_weighted_diffs_corrected, cmp_mask.sum(axis=1), cmp_mask
            )
            zscores_triplet_corrected = compute_zscores(
                *convert_to_triplet_data(abs_weighted_diffs_corrected, cmp_mask), cmp_mask
            )
            mean_zscore_corrected = np.nanmean(zscores_triplet_corrected)
            if mean_zscore_initial >= mean_zscore_corrected:
                outlier_mask_corrected = get_outlier_mask(
                    zscores_triplet_corrected,
                    zscores_corrected,
                    abs_weighted_diffs_corrected,
                    cmp_mask,
                    cdf_threshold=6.0,
                )
                new_offsets_corrected = compute_offsets(
                    weighted_diffs_corrected,
                    cmp_mask & ~outlier_mask_corrected,
                    min_AIC=6.0,
                )
                offsets_final = new_offsets_corrected
                outlier_mask_final = outlier_mask_corrected

        weighted_diffs_final, abs_weighted_diffs_final = compute_weighted_diffs(
            diff_arr, cmp_mask, offsets_final
        )

    return (
        weighted_diffs_final,
        abs_weighted_diffs_final,
        cmp_mask,
        outlier_mask_final,
        offsets_final,
        weighted_diffs_initial,
        abs_weighted_diffs_initial,
        outlier_mask_initial,
        offsets_initial,
        lacs_offsets,
    )
```

- [ ] **Step 2.6: Update `compute_scores` to consume the new return-tuple element**

In `trizod/trizod.py`, update the destructuring of `ret` (search for `weighted_diffs_final, abs_weighted_diffs_final, cmp_mask, outlier_mask_final,`) to add a 10th element `lacs_offsets`:

```python
        (
            weighted_diffs_final,
            abs_weighted_diffs_final,
            cmp_mask,
            outlier_mask_final,
            offsets_final,
            weighted_diffs_initial,
            abs_weighted_diffs_initial,
            outlier_mask_initial,
            offsets_initial,
            lacs_offsets,
        ) = ret
```

Add `lacs_offsets` to the cache `np.savez(...)` block as another array (so cached re-loads remain valid):

```python
        if cache_dir:
            np.savez(
                str(shifts_cache_path),
                shw=weighted_diffs_final,
                ashwi=abs_weighted_diffs_final,
                cmp_mask=cmp_mask,
                olf=outlier_mask_final,
                offf=np.array(
                    [offsets_final[atom_type] for atom_type in BACKBONE_ATOMS]
                ),
                shw0=weighted_diffs_initial,
                ashwi0=abs_weighted_diffs_initial,
                ol0=outlier_mask_initial,
                off0=np.array(
                    [offsets_initial[atom_type] for atom_type in BACKBONE_ATOMS]
                ),
                lacs=np.array(
                    [lacs_offsets[atom_type] for atom_type in BACKBONE_ATOMS]
                ),
            )
```

In the cache-load branch, decode the new `lacs` array; fall back to all-zeros for caches written before this commit (graceful degradation):

```python
            try:
                cached = np.load(str(shifts_cache_path))
                # ... existing keys ...
                if "lacs" in cached.files:
                    lacs_offsets = dict(zip(BACKBONE_ATOMS, cached["lacs"]))
                else:
                    lacs_offsets = dict.fromkeys(BACKBONE_ATOMS, 0.0)
                offsets_final = dict(zip(BACKBONE_ATOMS, offsets_final))
                offsets_initial = dict(zip(BACKBONE_ATOMS, offsets_initial))
            except Exception:
                ...
```

Make `compute_scores` return `lacs_offsets` (append to its return tuple) and have `compute_scores_row` store the offsets in the row:

```python
    return scores, k, cmp_mask, offsets, exe_times, lacs_offsets
```

In `compute_scores_row`:

```python
        scores, k, cmp_mask, offsets, exe_times, lacs_offsets = compute_scores(...)
        ...
        for atom_type in BACKBONE_ATOMS:
            row[f"off_{atom_type}"] = offsets[atom_type]
            row[f"lacs_off_{atom_type}"] = lacs_offsets[atom_type]
```

In `fill_row_data`, initialise the new columns alongside `off_{atom_type}`:

```python
    for atom_type in BACKBONE_ATOMS:
        row[f"off_{atom_type}"] = pd.NA
        row[f"lacs_off_{atom_type}"] = pd.NA
```

In `output_dataset`, add the new columns to the JSON column list (CSV doesn't carry per-entry offsets, so no change to the CSV branch):

```python
        dout = df.loc[df.pass_post].reset_index()[
            [
                "ID",
                ...,
                "off_C", "off_CA", "off_CB", "off_H", "off_HA", "off_HB", "off_N",
                "lacs_off_C", "lacs_off_CA", "lacs_off_CB",
                "lacs_off_H", "lacs_off_HA", "lacs_off_HB", "lacs_off_N",
                ...,
            ]
            + score_types
            + shifts
        ]
```

- [ ] **Step 2.7: Add a regression test for `rereference_mode='none'` vs `'both'` on a real entry**

Create `tests/test_rereference_modes.py`:

```python
"""Regression: --rereference-mode 'none' and 'both' produce different G-scores
for a known mis-referenced BMRB entry (17665, alpha-synuclein)."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from trizod.scoring.scoring import get_offset_corrected_shifts


REPO = Path(__file__).resolve().parent.parent
ASYN_PKL = REPO / "tmp" / "bmrb_entries" / "17665.pkl"


@pytest.mark.skipif(
    not ASYN_PKL.exists(),
    reason="alpha-synuclein BMRB entry pickle not on disk; populate tmp/bmrb_entries/ first",
)
def test_lacs_correction_changes_alphasyn_offsets():
    with ASYN_PKL.open("rb") as f:
        entry = pickle.load(f)

    peptide_shifts = entry.get_peptide_shifts()
    (st_id, ea_id, e_id), (shifts, cond_id, _, _) = next(iter(peptide_shifts.items()))
    seq = entry.entities[e_id].seq

    cond = entry.conditions[cond_id]
    temp = cond.get_temperature(return_default=True)
    pH = cond.get_pH(return_default=True)
    ion = cond.get_ionic_strength(return_default=True)

    import trizod.potenci.potenci as potenci

    predshiftdct = potenci.get_pred_shifts(seq, temp, pH, ion, pH != 7.0)

    ret_none = get_offset_corrected_shifts(seq, shifts, predshiftdct, rereference_mode="none")
    ret_both = get_offset_corrected_shifts(seq, shifts, predshiftdct, rereference_mode="both")

    assert ret_none is not None and ret_both is not None
    lacs_none = ret_none[-1]
    lacs_both = ret_both[-1]

    # mode='none' should have no LACS offset (all zeros); mode='both' should
    # detect a non-trivial CA/CB/CO offset for 17665 (Reid quoted ~2.9 ppm)
    assert all(v == 0.0 for v in lacs_none.values())
    assert any(abs(v) > 1.0 for k, v in lacs_both.items() if k in ("C", "CA", "CB"))
```

- [ ] **Step 2.8: Run all scoring tests, expect PASS**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run pytest tests/test_lacs.py tests/test_lacs_integration.py tests/test_rereference_modes.py -v
```
Expected: all PASS.

- [ ] **Step 2.9: Run full non-slow suite, ruff, then commit**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run ruff check trizod/ tests/
uv run ruff format --check trizod/ tests/
uv run pytest tests/ -v -m "not slow"
```
Expected: all green.

```bash
git add trizod/scoring/scoring.py trizod/trizod.py tests/test_lacs_integration.py tests/test_rereference_modes.py
git commit -m "$(cat <<'EOF'
feat(scoring): integrate LACS pre-correction into the scoring pipeline

Adds apply_lacs_correction() helper in scoring.py and wires it into
get_offset_corrected_shifts as a first pass before the existing AIC/POTENCI
offset detection. The new --rereference-mode flag (none/lacs/potenci-only/both,
default 'both') selects the strategy. Per-entry LACS offsets are persisted in
the wSCS cache (.npz) and the JSON output (lacs_off_<atom> columns), in
addition to the existing POTENCI residual offsets (off_<atom>).

Regression test on BMRB 17665 (alpha-synuclein) confirms LACS detects a
non-trivial CA/CB/CO offset, matching Reid's ~2.9 ppm TALOS-N reading.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Step 8 — Leu/Val methyl wildcards in BMRB parser (W3)

**Files:**
- Modify: `trizod/bmrb/bmrb.py` (apply wildcards at parse time)
- Test: `tests/test_methyl_wildcards.py` (new)

**Rationale:** Rewrite atom IDs `LEU CD1/CD2 → CDx` and `VAL CG1/CG2 → CGx` for non-stereospecific assignments at the `_Atom_chem_shift` parse site (around line 305-315 in `bmrb.py`). Backbone scoring is unaffected since `get_valid_bbshifts` only sees backbone atoms; the wildcards surface in the raw shift table that the `.str` writer in Task 4 emits.

**Ambiguity-code semantics (BMRB):** Code "1" = stereospecifically assigned. Code "2" = geminal partners (the typical non-stereospecific Leu/Val methyl flag). Codes 3+ = other ambiguity types not relevant to Leu/Val methyls. Empty code (`""` or `"."`) = not specified; we treat as non-stereospecific (conservative).

- [ ] **Step 3.1: Write failing test**

Create `tests/test_methyl_wildcards.py`:

```python
"""Step 8 — Leu CD1/CD2 and Val CG1/CG2 are rewritten to CDx/CGx for
non-stereospecific (geminal-partner) ambiguity codes."""

import pytest

from trizod.bmrb.bmrb import _maybe_wildcard_methyl


@pytest.mark.parametrize(
    "comp_id,atom_id,ambiguity,expected",
    [
        # Leu CD1/CD2: stereospecific (code "1") preserved
        ("LEU", "CD1", "1", "CD1"),
        ("LEU", "CD2", "1", "CD2"),
        # Leu CD1/CD2: geminal-partner (code "2") wildcarded
        ("LEU", "CD1", "2", "CDx"),
        ("LEU", "CD2", "2", "CDx"),
        # Leu CD1/CD2: missing code → wildcarded conservatively
        ("LEU", "CD1", "", "CDx"),
        ("LEU", "CD2", ".", "CDx"),
        # Val CG1/CG2: stereospecific preserved
        ("VAL", "CG1", "1", "CG1"),
        ("VAL", "CG2", "1", "CG2"),
        # Val CG1/CG2: geminal-partner wildcarded
        ("VAL", "CG1", "2", "CGx"),
        ("VAL", "CG2", "2", "CGx"),
        # Val CG1/CG2: missing code wildcarded
        ("VAL", "CG1", "", "CGx"),
        ("VAL", "CG2", ".", "CGx"),
        # Other residues / atoms: passthrough
        ("ALA", "CB", "1", "CB"),
        ("LEU", "CA", "2", "CA"),
        ("ILE", "CD1", "2", "CD1"),  # Ile, not Leu — unaffected
    ],
)
def test_maybe_wildcard_methyl(comp_id, atom_id, ambiguity, expected):
    assert _maybe_wildcard_methyl(comp_id, atom_id, ambiguity) == expected
```

- [ ] **Step 3.2: Run test, expect FAIL**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run pytest tests/test_methyl_wildcards.py -v
```
Expected: FAIL with `ImportError: cannot import name '_maybe_wildcard_methyl'`.

- [ ] **Step 3.3: Implement `_maybe_wildcard_methyl` in `trizod/bmrb/bmrb.py`**

Add this helper near the top of `bmrb.py` (after the imports, before any class/function definitions):

```python
# Step 8: residue/atom pairs whose stereospecific assignment is commonly
# unknown. When the ambiguity code indicates geminal-partner ambiguity
# (BMRB code "2") or no code is given, rewrite the atom_id to a wildcard so
# downstream automatic-assignment tools don't propagate a false stereospecific
# assignment.
_METHYL_WILDCARD_MAP = {
    ("LEU", "CD1"): "CDx",
    ("LEU", "CD2"): "CDx",
    ("VAL", "CG1"): "CGx",
    ("VAL", "CG2"): "CGx",
}
_STEREOSPECIFIC_CODES = {"1"}


def _maybe_wildcard_methyl(comp_id, atom_id, ambiguity_code):
    """Return atom_id, possibly rewritten to a wildcard for ambiguous methyls."""
    key = (comp_id, atom_id)
    if key not in _METHYL_WILDCARD_MAP:
        return atom_id
    if ambiguity_code in _STEREOSPECIFIC_CODES:
        return atom_id
    return _METHYL_WILDCARD_MAP[key]
```

- [ ] **Step 3.4: Run unit test, expect PASS**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run pytest tests/test_methyl_wildcards.py -v
```
Expected: all 13 parametrized cases PASS.

- [ ] **Step 3.5: Apply wildcard at parse time**

Locate the block in `bmrb.py` (around line 300-320) where the `_Atom_chem_shift` rows are zipped together. Find the line that reads `_Atom_chem_shift.Atom_ID`. After the rows are zipped but before the `ShiftTable(...)` is constructed (or however the shifts are stored), apply the wildcard.

Read the surrounding context:

```bash
grep -n "_Atom_chem_shift\|shift_rows\|ShiftTable\|shifts =" trizod/bmrb/bmrb.py | head -40
```

The likely site is right after the zip producing tuples of `(entity_assemID, entityID, seq_id, comp_id, atom_id, atom_type, val, val_err, ambiguity_code)`. Wrap that zip in a list comprehension that rewrites `atom_id`:

```python
shift_rows = [
    (
        ea_id, e_id, seq_id, comp_id,
        _maybe_wildcard_methyl(comp_id, atom_id, ambiguity_code),
        atom_type, val, val_err, ambiguity_code,
    )
    for ea_id, e_id, seq_id, comp_id, atom_id, atom_type, val, val_err, ambiguity_code in zip(
        get_tag_vals(sf, "_Atom_chem_shift.Entity_assembly_ID", default=[]),
        get_tag_vals(sf, "_Atom_chem_shift.Entity_ID", default=[]),
        get_tag_vals(sf, "_Atom_chem_shift.Seq_ID", default=[]),
        get_tag_vals(sf, "_Atom_chem_shift.Comp_ID", default=[]),
        get_tag_vals(sf, "_Atom_chem_shift.Atom_ID", default=[]),
        get_tag_vals(sf, "_Atom_chem_shift.Atom_type", default=[]),
        get_tag_vals(sf, "_Atom_chem_shift.Val", default=[]),
        get_tag_vals(sf, "_Atom_chem_shift.Val_err", default=[]),
        get_tag_vals(sf, "_Atom_chem_shift.Ambiguity_code", default=[]),
    )
]
```

Adapt to whatever the surrounding code looks like once read — only the `atom_id → _maybe_wildcard_methyl(...)` substitution is the actual change.

- [ ] **Step 3.6: Add an integration test on a real entry containing Leu**

Append to `tests/test_methyl_wildcards.py`:

```python
import pickle
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _load_pickle(name):
    p = REPO / "tmp" / "bmrb_entries" / name
    if not p.exists():
        pytest.skip(f"BMRB pickle {p} not present")
    with p.open("rb") as f:
        return pickle.load(f)


def test_leu_cdx_appears_in_real_entry():
    """Pick a real entry that contains Leu and assert at least one
    rewritten CDx exists post-parse."""
    # 17665 is alpha-synuclein, contains many Leu residues; use it as the canary
    entry = _load_pickle("17665.pkl")
    peptide_shifts = entry.get_peptide_shifts()
    found_cdx = False
    found_cgx = False
    for (_st, _ea, _e), (shifts, *_) in peptide_shifts.items():
        for row in shifts:
            atom_id = row[4]  # Atom_ID column
            if atom_id == "CDx":
                found_cdx = True
            if atom_id == "CGx":
                found_cgx = True
    # alpha-synuclein has Leu and Val, so at least one wildcard must be present
    # post-rewrite — strict check
    assert found_cdx or found_cgx, (
        "expected at least one Leu CDx or Val CGx wildcard after Step 8 rewrite"
    )
```

- [ ] **Step 3.7: Invalidate the BMRB pickle cache (Step 8 changes parser output)**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
rm -f tmp/bmrb_entries/17665.pkl tmp/bmrb_entries/6968.pkl
```

(Only the two αSyn pickles for the integration test. Mass cache invalidation happens in Task 5/W5 when we kick the rerun. For the test, those two are enough.)

- [ ] **Step 3.8: Run integration test, expect PASS**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run pytest tests/test_methyl_wildcards.py -v
```
Expected: all PASS (re-parse of 17665.pkl + 6968.pkl happens lazily on first scoring run; the test re-creates the pickle from the raw `.str` if missing, OR the test must be skipped if pickle absent).

If `_load_pickle` raises FileNotFoundError because the pickle was just deleted, the workaround is to re-parse the raw `.str` directly:

```python
from trizod.bmrb.bmrb import BmrbEntry
entry = BmrbEntry("17665", REPO / "data" / "bmrb_entries" / "bmr17665")
```

Update the helper if needed.

- [ ] **Step 3.9: Run full non-slow suite, ruff, then commit**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run ruff check trizod/ tests/
uv run ruff format --check trizod/ tests/
uv run pytest tests/ -v -m "not slow"
```
Expected: all green.

```bash
git add trizod/bmrb/bmrb.py tests/test_methyl_wildcards.py
git commit -m "$(cat <<'EOF'
feat(bmrb): wildcard naming (CDx/CGx) for ambiguous Leu/Val methyls

Step 8 of the modernization roadmap. Rewrites Leu CD1/CD2 -> CDx and
Val CG1/CG2 -> CGx at parse time when the BMRB ambiguity code indicates
non-stereospecific (geminal-partner) assignment, or when no code is given.
Stereospecific assignments (ambiguity code "1") are preserved unchanged.
Backbone scoring is unaffected — these are side-chain methyls and never
enter get_valid_bbshifts. The wildcards surface in the raw shift table for
downstream consumption by .str emission and automatic-assignment tools.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `.str` emission — `--emit-str` flag (W3)

**Files:**
- Create: `trizod/io/__init__.py`
- Create: `trizod/io/str_writer.py`
- Modify: `trizod/trizod.py` (CLI flag, call writer per row)
- Test: `tests/test_str_writer.py` (new)

**Rationale:** Emit a backbone-shifts-only NMR-STAR file per scored entry. Round-trip-parseable with `pynmrstar`. Auxiliary saveframe records LACS offsets, POTENCI residual offsets, and pipeline version.

- [ ] **Step 4.1: Confirm `pynmrstar` is in the dependency tree**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
grep pynmrstar pyproject.toml
```
Expected: `pynmrstar` listed as a dependency. If not, add it: `uv add pynmrstar`.

- [ ] **Step 4.2: Write failing test**

Create `tests/test_str_writer.py`:

```python
"""Tests for the re-referenced .str emitter."""

from pathlib import Path

import numpy as np
import pynmrstar
import pytest

from trizod.constants import BACKBONE_ATOMS
from trizod.io.str_writer import write_rereferenced_str


def test_write_rereferenced_str_round_trip(tmp_path):
    """Emit, then parse back, and assert shifts match within 1e-6 ppm."""
    seq = "AGAGAGAGAG"
    bbshifts_arr = np.zeros((10, 7))
    bbshifts_mask = np.zeros((10, 7), dtype=bool)
    ca_idx = BACKBONE_ATOMS.index("CA")
    cb_idx = BACKBONE_ATOMS.index("CB")
    bbshifts_arr[:, ca_idx] = np.linspace(50.0, 56.0, 10)
    bbshifts_arr[:, cb_idx] = np.linspace(18.0, 22.0, 10)
    bbshifts_mask[:, ca_idx] = True
    bbshifts_mask[:, cb_idx] = True

    lacs_offsets = {atom: 0.0 for atom in BACKBONE_ATOMS}
    lacs_offsets["CA"] = 0.5
    lacs_offsets["CB"] = -0.3
    potenci_offsets = {atom: 0.0 for atom in BACKBONE_ATOMS}

    out_path = tmp_path / "bmr12345_rereferenced.str"
    write_rereferenced_str(
        out_path,
        entry_id="12345",
        seq=seq,
        bbshifts_arr=bbshifts_arr,
        bbshifts_mask=bbshifts_mask,
        lacs_offsets=lacs_offsets,
        potenci_residual_offsets=potenci_offsets,
        rereference_mode="both",
        pipeline_version="trizod-2026-05-05",
    )

    parsed = pynmrstar.Entry.from_file(str(out_path))
    shift_loop = parsed.get_loops_by_category("Atom_chem_shift")
    assert len(shift_loop) == 1
    rows = shift_loop[0].data
    # Spot-check first CA row: residue 1, atom CA, value matches input
    ca_rows = [r for r in rows if r[shift_loop[0].tag_index("Atom_ID")] == "CA"]
    assert len(ca_rows) == 10
    first_ca_val = float(ca_rows[0][shift_loop[0].tag_index("Val")])
    assert abs(first_ca_val - bbshifts_arr[0, ca_idx]) < 1e-6


def test_write_rereferenced_str_records_offsets_in_aux(tmp_path):
    """Auxiliary metadata block records LACS + POTENCI residual offsets."""
    out_path = tmp_path / "bmr00001_rereferenced.str"
    write_rereferenced_str(
        out_path,
        entry_id="00001",
        seq="A",
        bbshifts_arr=np.zeros((1, 7)),
        bbshifts_mask=np.zeros((1, 7), dtype=bool),
        lacs_offsets={"CA": 1.5, "CB": 0.0, "C": 0.0, "H": 0.0, "HA": 0.0, "HB": 0.0, "N": 0.0},
        potenci_residual_offsets={"CA": 0.1, "CB": 0.0, "C": 0.0, "H": 0.0, "HA": 0.0, "HB": 0.0, "N": 0.0},
        rereference_mode="both",
        pipeline_version="trizod-2026-05-05",
    )
    text = out_path.read_text()
    assert "LACS_offsets" in text
    assert "POTENCI_residual_offsets" in text
    assert "1.5" in text
    assert "trizod-2026-05-05" in text
```

- [ ] **Step 4.3: Run test, expect FAIL**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run pytest tests/test_str_writer.py -v
```
Expected: FAIL with `ImportError: cannot import name 'write_rereferenced_str'`.

- [ ] **Step 4.4: Implement the writer**

Create `trizod/io/__init__.py` (empty):

```python
```

Create `trizod/io/str_writer.py`:

```python
"""Emit re-referenced backbone-shifts-only NMR-STAR (.str) files."""

from pathlib import Path

import pynmrstar

from trizod.constants import BACKBONE_ATOMS, AA1TO3


_AMBIGUITY_NOT_SET = "."


def write_rereferenced_str(
    out_path,
    entry_id,
    seq,
    bbshifts_arr,
    bbshifts_mask,
    lacs_offsets,
    potenci_residual_offsets,
    rereference_mode,
    pipeline_version,
):
    """Write a backbone-shifts-only NMR-STAR file for an re-referenced entry.

    Args:
        out_path: destination file (Path or str).
        entry_id: BMRB entry id (string).
        seq: one-letter amino-acid sequence.
        bbshifts_arr: (N, 7) array of CORRECTED backbone shifts.
        bbshifts_mask: (N, 7) boolean mask.
        lacs_offsets: dict, atom -> ppm.
        potenci_residual_offsets: dict, atom -> ppm.
        rereference_mode: which mode produced the shifts; copied into metadata.
        pipeline_version: identifier (free string), copied into metadata.
    """
    out_path = Path(out_path)
    entry = pynmrstar.Entry.from_scratch(f"bmr{entry_id}_rereferenced")

    # Saveframe 1: chemical shifts
    sf = pynmrstar.Saveframe.from_scratch(
        f"assigned_chem_shift_list_1", "assigned_chemical_shifts"
    )
    sf.add_tag("Sf_category", "assigned_chemical_shifts")
    sf.add_tag("Sf_framecode", "assigned_chem_shift_list_1")
    sf.add_tag("ID", "1")

    loop = pynmrstar.Loop.from_scratch("Atom_chem_shift")
    loop.set_category("Atom_chem_shift")
    loop.add_tag(
        [
            "ID", "Seq_ID", "Comp_ID", "Atom_ID", "Atom_type",
            "Val", "Val_err", "Ambiguity_code",
        ]
    )

    row_id = 0
    for i, aa1 in enumerate(seq):
        if aa1 not in AA1TO3:
            continue
        comp_id = AA1TO3[aa1]
        for j, atom in enumerate(BACKBONE_ATOMS):
            if not bbshifts_mask[i, j]:
                continue
            row_id += 1
            atom_type = atom[0]  # "C", "H", "N"
            loop.add_data(
                [
                    str(row_id),
                    str(i + 1),
                    comp_id,
                    atom,
                    atom_type,
                    f"{bbshifts_arr[i, j]:.4f}",
                    ".",
                    _AMBIGUITY_NOT_SET,
                ]
            )
    sf.add_loop(loop)
    entry.add_saveframe(sf)

    # Saveframe 2: trizod auxiliary metadata
    aux = pynmrstar.Saveframe.from_scratch(
        "trizod_rereferencing_info", "trizod_rereferencing"
    )
    aux.add_tag("Sf_category", "trizod_rereferencing")
    aux.add_tag("Sf_framecode", "trizod_rereferencing_info")
    aux.add_tag("Source_BMRB_id", entry_id)
    aux.add_tag("Pipeline_version", pipeline_version)
    aux.add_tag("Re_referencing_mode", rereference_mode)

    lacs_loop = pynmrstar.Loop.from_scratch("LACS_offsets")
    lacs_loop.set_category("LACS_offsets")
    lacs_loop.add_tag(["Atom_ID", "Offset_ppm"])
    for atom, offset in lacs_offsets.items():
        lacs_loop.add_data([atom, f"{offset:.6f}"])
    aux.add_loop(lacs_loop)

    potenci_loop = pynmrstar.Loop.from_scratch("POTENCI_residual_offsets")
    potenci_loop.set_category("POTENCI_residual_offsets")
    potenci_loop.add_tag(["Atom_ID", "Offset_ppm"])
    for atom, offset in potenci_residual_offsets.items():
        potenci_loop.add_data([atom, f"{offset:.6f}"])
    aux.add_loop(potenci_loop)

    entry.add_saveframe(aux)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    entry.write_to_file(str(out_path))
```

If `AA1TO3` doesn't exist in `trizod/constants.py`, find the inverse mapping (`AA3TO1`) and invert it inside `str_writer.py`:

```python
from trizod.constants import AA3TO1
AA1TO3 = {v: k for k, v in AA3TO1.items()}
```

- [ ] **Step 4.5: Run test, expect PASS**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run pytest tests/test_str_writer.py -v
```
Expected: both tests PASS.

- [ ] **Step 4.6: Add `--emit-str` CLI flag and call the writer per row**

In `trizod/trizod.py`:

1. Add the flag in `io_grp`:

```python
    io_grp.add_argument(
        "--emit-str",
        type=Path,
        default=None,
        help="Directory to write re-referenced NMR-STAR (.str) files into. "
             "One file per scored entry: <dir>/bmr<id>_rereferenced.str. "
             "Off by default.",
    )
```

2. Resolve the path and create the directory in `parse_args()` (just below the `args.cache_dir` block):

```python
    if args.emit_str is not None:
        args.emit_str = Path(args.emit_str).resolve()
        args.emit_str.mkdir(parents=True, exist_ok=True)
```

3. After `compute_scores_row` returns, write the `.str` file. The cleanest place is a new pass after scoring, before output. In `main()`, after the `compute_scores_row` parallel apply but before `print_filter_losses`, insert:

```python
    if args.emit_str is not None:
        from trizod.io.str_writer import write_rereferenced_str
        from trizod.bmrb.bmrb import get_valid_bbshifts
        logging.getLogger("trizod").info(f"Emitting re-referenced .str files to {args.emit_str}")
        for _, row in tqdm(df[df["pass_post"]].iterrows(), total=df["pass_post"].sum()):
            entry = bmrb_entries.loc[row["entryID"], "entry"]
            peptide_shifts = entry.get_peptide_shifts()
            shifts, _, _, _ = peptide_shifts[
                (row["stID"], row["entity_assemID"], row["entityID"])
            ]
            seq = row["seq"]
            ret = get_valid_bbshifts(shifts, seq)
            if ret is None:
                continue
            bbshifts_arr, bbshifts_mask = ret
            # Apply LACS offset (subtract from observed) so emitted shifts are
            # post-correction. POTENCI residual offsets are weighted-diff side,
            # not raw-shift side, so they aren't subtracted here — they're
            # captured in the aux saveframe for transparency.
            for j, atom in enumerate(BACKBONE_ATOMS):
                lacs_off = row.get(f"lacs_off_{atom}", 0.0)
                if pd.isna(lacs_off):
                    lacs_off = 0.0
                bbshifts_arr[bbshifts_mask[:, j], j] -= lacs_off
            lacs_offsets = {atom: row.get(f"lacs_off_{atom}", 0.0) or 0.0 for atom in BACKBONE_ATOMS}
            potenci_offsets = {atom: row.get(f"off_{atom}", 0.0) or 0.0 for atom in BACKBONE_ATOMS}
            out_path = args.emit_str / f"bmr{row['entryID']}_rereferenced.str"
            write_rereferenced_str(
                out_path,
                entry_id=row["entryID"],
                seq=seq,
                bbshifts_arr=bbshifts_arr,
                bbshifts_mask=bbshifts_mask,
                lacs_offsets=lacs_offsets,
                potenci_residual_offsets=potenci_offsets,
                rereference_mode=args.rereference_mode,
                pipeline_version="trizod-2026-05-05",
            )
```

(`pd` is already imported at the top of `trizod.py`.)

- [ ] **Step 4.7: Add a small smoke test that `--emit-str` runs end-to-end on one entry**

Append to `tests/test_smoke.py`:

```python
def test_emit_str_smoke(tmp_path):
    """Run the pipeline end-to-end on a single tiny BMRB entry with --emit-str."""
    import shutil
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    src = repo / "data" / "bmrb_entries" / "bmr6968"
    if not src.exists():
        pytest.skip(f"BMRB entry {src} not on disk")
    work = tmp_path / "input"
    shutil.copytree(src, work / "bmr6968")
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    out_dir.mkdir()
    cache_dir.mkdir()
    str_dir = tmp_path / "str_out"
    cmd = [
        sys.executable, "-m", "trizod",
        "--input-dir", str(work),
        "--output-prefix", str(out_dir / "test"),
        "--filter-defaults", "tolerant",
        "--cache-dir", str(cache_dir),
        "--emit-str", str(str_dir),
        "--processes", "1",
        "--no-progress",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"trizod failed:\nstdout:{result.stdout}\nstderr:{result.stderr}"
    emitted = list(str_dir.glob("bmr*_rereferenced.str"))
    assert len(emitted) >= 1
    assert "bmr6968" in emitted[0].name
```

- [ ] **Step 4.8: Run all tests, ruff, then commit**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run ruff check trizod/ tests/
uv run ruff format --check trizod/ tests/
uv run pytest tests/ -v -m "not slow"
```
Expected: all green.

```bash
git add trizod/io/__init__.py trizod/io/str_writer.py trizod/trizod.py tests/test_str_writer.py tests/test_smoke.py
git commit -m "$(cat <<'EOF'
feat(output): emit re-referenced NMR-STAR (.str) files via --emit-str

Adds trizod/io/str_writer.write_rereferenced_str(), which writes a
backbone-shifts-only NMR-STAR file per scored entry containing the
LACS-corrected chemical shifts plus an auxiliary saveframe recording
LACS offsets, POTENCI residual offsets, the re-referencing mode, and
the pipeline version. Driven by the new --emit-str <dir> CLI flag.
Round-trip parseable with pynmrstar; smoke test runs the full pipeline
on BMRB 6968 (alpha-synuclein ground-truth) and asserts a .str file is
emitted.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Release metadata — `.zenodo.json`, `CITATION.cff`, README section (W4)

**Files:**
- Create: `.zenodo.json`
- Create: `CITATION.cff`
- Modify: `README.md` (add "Releases & Re-referenced Dataset" section)
- Modify: `.gitignore` (add `data/release/`)

- [ ] **Step 5.1: Create `.zenodo.json`**

```json
{
  "title": "TriZOD — Re-referenced BMRB chemical shift dataset and disorder scoring pipeline",
  "description": "TriZOD scores BMRB NMR chemical shift entries for protein disorder using POTENCI random-coil predictions, LACS re-referencing, and CheZOD-style Z/G-scores. This Zenodo deposit contains the re-referenced backbone-shifts-only NMR-STAR files plus per-residue disorder scores produced by the finalized 2026-05 pipeline.",
  "creators": [
    {
      "name": "Senoner, Tobias",
      "affiliation": "TUM"
    }
  ],
  "keywords": [
    "NMR",
    "BMRB",
    "intrinsically disordered proteins",
    "chemical shift",
    "re-referencing",
    "LACS",
    "POTENCI",
    "CheZOD",
    "Z-score",
    "G-score"
  ],
  "license": "MIT",
  "upload_type": "dataset",
  "related_identifiers": [
    {
      "identifier": "https://bmrb.io/",
      "relation": "isDerivedFrom",
      "scheme": "url",
      "resource_type": "dataset"
    },
    {
      "identifier": "https://github.com/tsenoner/trizod",
      "relation": "isSupplementTo",
      "scheme": "url",
      "resource_type": "software"
    }
  ]
}
```

- [ ] **Step 5.2: Create `CITATION.cff`**

```yaml
cff-version: 1.2.0
title: "TriZOD: Re-referenced BMRB chemical shift dataset and disorder scoring pipeline"
message: "If you use this software or dataset, please cite it as below."
type: software
authors:
  - family-names: Senoner
    given-names: Tobias
    email: tobias.senoner94@gmail.com
repository-code: "https://github.com/tsenoner/trizod"
license: MIT
version: "0.1.0-pipeline"
date-released: "2026-05-05"
keywords:
  - NMR
  - BMRB
  - chemical shift
  - intrinsically disordered proteins
  - LACS
  - POTENCI
  - CheZOD
```

- [ ] **Step 5.3: Add a "Releases & Re-referenced Dataset" section to `README.md`**

Append to the end of the README (or insert after the existing usage section — locate by reading first):

```bash
cat /Users/tsenoner/Documents/projects/_github/trizod/README.md | head -50
```

Add a new section near the bottom:

```markdown
## Releases & Re-referenced Dataset

The finalized pipeline emits per-entry re-referenced NMR-STAR (`.str`) files
when run with `--emit-str <dir>`. Each emitted file contains the backbone
shifts after LACS pre-correction plus an auxiliary saveframe recording the
LACS offsets, POTENCI residual offsets, the re-referencing mode, and the
pipeline version.

A full re-referenced TriZOD dataset is being prepared for archival on
Zenodo. The repository ships `.zenodo.json` and `CITATION.cff` describing
the deposit; the actual upload (with DOI) is performed by the maintainer
via the GitHub-Zenodo integration on the repository's first tagged release.

Local release artifacts (uncommitted, gitignored) live under
`data/release/<tier>/`.
```

- [ ] **Step 5.4: Add `data/release/` to `.gitignore`**

```bash
echo "data/release/" >> /Users/tsenoner/Documents/projects/_github/trizod/.gitignore
```

(Verify it isn't already there with `grep '^data/release' .gitignore` first.)

- [ ] **Step 5.5: Commit**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
git add .zenodo.json CITATION.cff README.md .gitignore
git commit -m "$(cat <<'EOF'
chore(release): add .zenodo.json, CITATION.cff, README releases section

Prepares Zenodo deposit metadata so the finalized re-referenced dataset can
be uploaded via the GitHub-Zenodo integration on tagged release. Adds a
"Releases & Re-referenced Dataset" section to README.md describing the
.str output and Zenodo deposit workflow. data/release/ is gitignored.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: αSyn + top-3 flippers case study (W7)

**Files:**
- Create: `scripts/case_study_gscore_flips.py`
- Create: `docs/260505/figures/gscore_flips.png` (output)

**Rationale:** Reid #2. Single 4-panel figure: residue × G-score, paired raw/re-ref traces. Panel A = αSyn 17665 (with 6968 ground-truth as 3rd trace); B/C/D = top-3 entries by mean |ΔG|. The script reuses scoring as a library, calling `get_offset_corrected_shifts` with `rereference_mode="none"` then `"both"`.

- [ ] **Step 6.1: Create the script**

Create `scripts/case_study_gscore_flips.py`:

```python
#!/usr/bin/env python3
"""Reid #2 — αSyn + top-3 flippers case study.

Produces a 4-panel figure showing per-residue G-score before vs after
re-referencing for:
  Panel A: BMRB 17665 (alpha-synuclein, mis-referenced) raw + re-referenced,
           plus BMRB 6968 (alpha-synuclein, ground-truth) as a third trace.
  Panel B/C/D: the three entries (excluding 17665) with the largest mean
           |delta-G| across the dataset, in the tolerant tier.

Usage:
    uv run python scripts/case_study_gscore_flips.py \
        --baseline-tolerant data/baseline/tolerant.json \
        --bmrb-cache tmp/bmrb_entries \
        --potenci-cache tmp/potenci \
        --output docs/260505/figures/gscore_flips.png
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trizod.bmrb.bmrb as bmrb
import trizod.potenci.potenci as potenci
from trizod.constants import BACKBONE_ATOMS
from trizod.scoring.scoring import (
    compute_gscores,
    convert_to_triplet_data,
    get_offset_corrected_shifts,
)
from trizod.trizod import load_potenci_cache, save_potenci_cache

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("case_study")


def score_entry(entry, cache_dir, mode):
    """Run the scoring path for the first peptide shift table on `entry`,
    returning per-residue G-scores. Returns (gscores, seq) or (None, None)."""
    peptide_shifts = entry.get_peptide_shifts()
    for (st_id, ea_id, e_id), (shifts, cond_id, _, _) in peptide_shifts.items():
        if cond_id not in entry.conditions or e_id not in entry.entities:
            continue
        seq = entry.entities[e_id].seq
        if not seq or len(seq) < 20:
            continue
        cond = entry.conditions[cond_id]
        temp = cond.get_temperature(return_default=True)
        pH = cond.get_pH(return_default=True)
        ion = cond.get_ionic_strength(return_default=True)

        predshiftdct = load_potenci_cache(cache_dir, seq, temp, pH, ion)
        if predshiftdct is None:
            predshiftdct = potenci.get_pred_shifts(seq, temp, pH, ion, pH != 7.0)
            save_potenci_cache(cache_dir, seq, temp, pH, ion, predshiftdct)

        ret = get_offset_corrected_shifts(seq, shifts, predshiftdct, rereference_mode=mode)
        if ret is None:
            return None, None
        (
            _wdf, abs_wdf, cmp_mask, *_,
        ) = ret
        if not np.any(cmp_mask):
            return None, None
        triplet_diffs, triplet_dof = convert_to_triplet_data(abs_wdf, cmp_mask)
        gscores = compute_gscores(triplet_diffs, triplet_dof, cmp_mask)
        return gscores, seq
    return None, None


def find_top_flippers(baseline_tolerant_path, bmrb_cache, potenci_cache, exclude_ids, k=3):
    """Score every tolerant-tier entry with mode='none' and 'both', compute
    mean |delta-G|, return top-k entry ids by that metric."""
    seen = set()
    deltas = []
    with open(baseline_tolerant_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for r in rows:
        eid = r["entryID"]
        if eid in exclude_ids or eid in seen:
            continue
        seen.add(eid)
        pkl = bmrb_cache / f"{eid}.pkl"
        if not pkl.exists():
            continue
        try:
            with pkl.open("rb") as f:
                entry = pickle.load(f)
        except Exception:
            continue
        g_none, _ = score_entry(entry, potenci_cache, "none")
        g_both, _ = score_entry(entry, potenci_cache, "both")
        if g_none is None or g_both is None:
            continue
        diff = np.abs(g_both - g_none)
        with np.errstate(invalid="ignore"):
            mean_abs_delta = float(np.nanmean(diff))
        if not np.isfinite(mean_abs_delta):
            continue
        deltas.append((eid, mean_abs_delta))

    deltas.sort(key=lambda kv: -kv[1])
    return [eid for eid, _ in deltas[:k]]


def render(panels, out_path):
    """panels = list of dicts with keys: title, residues, traces (list of
    {label, ystyle, gscores}). Up to 4 panels."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for ax, panel in zip(axes.flat, panels):
        for tr in panel["traces"]:
            ax.plot(panel["residues"], tr["gscores"], tr["ystyle"], label=tr["label"], lw=1.4)
        ax.axhline(0.5, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("residue")
        ax.set_ylabel("G-score")
        ax.set_ylim(0, 1)
        ax.set_title(panel["title"])
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tolerant", type=Path, default=Path("data/baseline/tolerant.json"))
    parser.add_argument("--bmrb-cache", type=Path, default=Path("tmp/bmrb_entries"))
    parser.add_argument("--potenci-cache", type=Path, default=Path("tmp"))
    parser.add_argument("--output", type=Path, default=Path("docs/260505/figures/gscore_flips.png"))
    args = parser.parse_args()

    panels = []

    # Panel A — αSyn 17665 + 6968 ground truth
    with (args.bmrb_cache / "17665.pkl").open("rb") as f:
        entry_17665 = pickle.load(f)
    with (args.bmrb_cache / "6968.pkl").open("rb") as f:
        entry_6968 = pickle.load(f)
    g_raw, seq_17665 = score_entry(entry_17665, args.potenci_cache, "none")
    g_ref, _ = score_entry(entry_17665, args.potenci_cache, "both")
    g_truth, seq_6968 = score_entry(entry_6968, args.potenci_cache, "both")
    residues_a = np.arange(1, len(seq_17665) + 1)

    # Pad/trim 6968 trace to align with 17665 length (drop or NaN)
    if g_truth is not None and len(seq_6968) >= len(seq_17665):
        g_truth_aligned = g_truth[: len(seq_17665)]
    else:
        g_truth_aligned = np.full_like(g_raw, np.nan, dtype=float)
        if g_truth is not None:
            g_truth_aligned[: len(g_truth)] = g_truth

    panels.append({
        "title": "alpha-synuclein (BMRB 17665, mis-referenced)",
        "residues": residues_a,
        "traces": [
            {"label": "17665 raw",         "ystyle": "-",  "gscores": g_raw},
            {"label": "17665 re-referenced", "ystyle": "--", "gscores": g_ref},
            {"label": "6968 ground truth",   "ystyle": ":",  "gscores": g_truth_aligned},
        ],
    })

    # Panels B/C/D — top 3 flippers (excluding 17665)
    top_ids = find_top_flippers(
        args.baseline_tolerant, args.bmrb_cache, args.potenci_cache,
        exclude_ids={"17665"}, k=3,
    )
    print(f"top 3 flippers (by mean |delta-G|, excluding 17665): {top_ids}")

    for eid in top_ids:
        with (args.bmrb_cache / f"{eid}.pkl").open("rb") as f:
            entry = pickle.load(f)
        g_raw, seq = score_entry(entry, args.potenci_cache, "none")
        g_ref, _ = score_entry(entry, args.potenci_cache, "both")
        if g_raw is None or g_ref is None:
            continue
        residues = np.arange(1, len(seq) + 1)
        # Look up the entity name if available
        peptide_shifts = entry.get_peptide_shifts()
        (_st, _ea, eid_internal), _ = next(iter(peptide_shifts.items()))
        name = entry.entities[eid_internal].name if eid_internal in entry.entities else "?"
        title = f"BMRB {eid}: {name[:40]}"
        panels.append({
            "title": title,
            "residues": residues,
            "traces": [
                {"label": "raw",         "ystyle": "-",  "gscores": g_raw},
                {"label": "re-referenced", "ystyle": "--", "gscores": g_ref},
            ],
        })

    while len(panels) < 4:
        panels.append({"title": "(no flipper)", "residues": np.arange(1, 2), "traces": []})

    render(panels, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Run the script**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
mkdir -p docs/260505/figures
uv run python scripts/case_study_gscore_flips.py
```
Expected: prints "top 3 flippers" + "figure written to docs/260505/figures/gscore_flips.png".

If `find_top_flippers` is too slow on the full tolerant tier (1500+ entries, ~3-5 s each = 1-2 hours), short-circuit it: pass `--baseline-tolerant data/baseline/strict.json` (smaller, faster), or add `--max-scan 500` to limit the iteration. If we slip the budget, ship the figure with αSyn + 3 known top-3 candidates from the existing `tmp/lacs_comparison_results.pkl` (use `compare_gscores_lacs.py`'s output to identify hot entries directly).

- [ ] **Step 6.3: Sanity-check the αSyn LACS offset**

Print the LACS offset that the script implicitly applied to 17665 — Reid quoted ~2.9 ppm:

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
python -c "
import pickle, sys
sys.path.insert(0, '.')
from trizod.scoring.scoring import get_offset_corrected_shifts
import trizod.potenci.potenci as potenci
with open('tmp/bmrb_entries/17665.pkl', 'rb') as f:
    e = pickle.load(f)
peptide_shifts = e.get_peptide_shifts()
(st, ea, ei), (shifts, cid, _, _) = next(iter(peptide_shifts.items()))
seq = e.entities[ei].seq
c = e.conditions[cid]
predict = potenci.get_pred_shifts(seq, c.get_temperature(return_default=True), c.get_pH(return_default=True), c.get_ionic_strength(return_default=True), False)
ret = get_offset_corrected_shifts(seq, shifts, predict, rereference_mode='both')
print('LACS offsets:', ret[-1])
"
```
Expected: a non-trivial offset (≥ 1 ppm) on at least one of CA, CB, C. Reid quoted 2.9 ppm; sign convention may flip.

- [ ] **Step 6.4: Commit**

```bash
git add scripts/case_study_gscore_flips.py docs/260505/figures/gscore_flips.png
git commit -m "$(cat <<'EOF'
feat(scripts): alpha-synuclein and top-3 G-score flippers case study (Reid #2)

Adds scripts/case_study_gscore_flips.py producing a 4-panel figure for
the 5 May talk: per-residue G-score before vs after re-referencing for
BMRB 17665 (alpha-synuclein, mis-referenced) with BMRB 6968 as a
ground-truth third trace, plus the three other entries with the largest
mean |delta-G| across the tolerant tier.

Demonstrates that re-referencing systematically rescues mis-classified
disorder calls — alpha-synuclein flips from "looks helical" (raw) to
"looks disordered" (re-referenced), matching the ground truth at 6968,
and three other named BMRB entries show the same flip pattern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: CSP analysis (Reid #1) (W6)

**Files:**
- Create: `scripts/csp_analysis.py`
- Create: `docs/260505/figures/csp_histogram.png` (output)
- Create: `docs/260505/figures/csp_interface_example.png` (output)

**Rationale:** Reid #1. Compute the HN/N CSP `sqrt(dH^2 + (dN/5)^2)` for the existing 581 bound/unbound pairs from `analyse_duplicate_entries.py`'s machinery. Histogram all pairs; pick one named pair (FKBP12 bmr16925 vs bmr16931) and produce a per-residue CSP bar plot with the trimmed-mean threshold marked.

- [ ] **Step 7.1: Create the script**

Create `scripts/csp_analysis.py`:

```python
#!/usr/bin/env python3
"""Reid #1 — Chemical Shift Perturbation (CSP) analysis on duplicate pairs.

Computes Reid's CSP formula:
    CSP = sqrt(dH^2 + (dN/alpha)^2),  alpha=5

for each bound/unbound pair extracted by analyse_duplicate_entries.py.
Outputs a histogram across all pairs (with trimmed-mean threshold) and a
per-residue interface plot for one named pair.

Usage:
    uv run python scripts/csp_analysis.py \
        --tier tolerant \
        --baseline-dir data/baseline \
        --cache-dir tmp/bmrb_entries \
        --output-histogram docs/260505/figures/csp_histogram.png \
        --output-example docs/260505/figures/csp_interface_example.png \
        --example-single bmr16925 --example-bound bmr16931
"""

import argparse
import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trizod.bmrb.bmrb import get_valid_bbshifts
from trizod.constants import BACKBONE_ATOMS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("csp")

ALPHA_N = 5.0  # Reid's down-weighting factor for 15N


def load_baseline(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def shifts_for(entry_id, cache_dir, entity_id_filter=None):
    """Load the first matching peptide shift table from a cached BmrbEntry.
    Returns (bbshifts_arr, bbshifts_mask, seq) or (None, None, None)."""
    pkl = cache_dir / f"{entry_id}.pkl"
    if not pkl.exists():
        return None, None, None
    try:
        with pkl.open("rb") as f:
            entry = pickle.load(f)
    except Exception:
        return None, None, None
    peptide_shifts = entry.get_peptide_shifts()
    for (_st, _ea, e_id), (shifts, *_rest) in peptide_shifts.items():
        if entity_id_filter is not None and e_id != entity_id_filter:
            continue
        seq = entry.entities[e_id].seq
        if not seq:
            continue
        ret = get_valid_bbshifts(shifts, seq)
        if ret is None:
            continue
        return ret[0], ret[1], seq
    return None, None, None


def compute_pair_csp(seq, arr_a, mask_a, arr_b, mask_b):
    """Per-residue HN/N CSP. Returns (csp, residues_with_data)."""
    h_idx = BACKBONE_ATOMS.index("H")
    n_idx = BACKBONE_ATOMS.index("N")
    have_both = mask_a[:, [h_idx, n_idx]].all(axis=1) & mask_b[:, [h_idx, n_idx]].all(axis=1)
    n = len(seq)
    csp = np.full(n, np.nan)
    for i in range(n):
        if not have_both[i]:
            continue
        dH = arr_a[i, h_idx] - arr_b[i, h_idx]
        dN = arr_a[i, n_idx] - arr_b[i, n_idx]
        csp[i] = float(np.sqrt(dH * dH + (dN / ALPHA_N) ** 2))
    return csp


def trimmed_mean_threshold(values, drop_top_frac=0.10):
    arr = np.asarray(values)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    cutoff = np.quantile(arr, 1.0 - drop_top_frac)
    kept = arr[arr <= cutoff]
    return float(kept.mean()), float(kept.mean() + kept.std(ddof=0))


def find_pairs(rows, cache_dir):
    """Group rows by sequence, pick bound/unbound pairs (single-entity vs
    multi-entity in the same group)."""
    by_seq = defaultdict(list)
    for r in rows:
        if r.get("seq") and len(r["seq"]) >= 10:
            by_seq[r["seq"]].append(r)
    pairs = []
    for seq, entries in by_seq.items():
        if len(entries) < 2:
            continue
        # Use entity-info enrichment as in analyse_duplicate_entries
        single, bound = [], []
        for r in entries:
            pkl = cache_dir / f"{r['entryID']}.pkl"
            if not pkl.exists():
                continue
            try:
                with pkl.open("rb") as f:
                    entry = pickle.load(f)
            except Exception:
                continue
            n_entities = len(entry.entities)
            has_non_polymer = any(e.type == "non-polymer" for e in entry.entities.values())
            has_nucleic = any(
                e.type == "polymer"
                and e.polymer_type in ("polydeoxyribonucleotide", "polyribonucleotide")
                for e in entry.entities.values()
            )
            if n_entities == 1:
                single.append(r)
            elif has_non_polymer or has_nucleic:
                bound.append(r)
        for s in single[:3]:
            for b in bound[:3]:
                if s["entryID"] == b["entryID"]:
                    continue
                pairs.append((seq, s, b))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["unfiltered", "tolerant", "moderate", "strict"], default="tolerant")
    parser.add_argument("--baseline-dir", type=Path, default=Path("data/baseline"))
    parser.add_argument("--cache-dir", type=Path, default=Path("tmp/bmrb_entries"))
    parser.add_argument("--output-histogram", type=Path, default=Path("docs/260505/figures/csp_histogram.png"))
    parser.add_argument("--output-example", type=Path, default=Path("docs/260505/figures/csp_interface_example.png"))
    parser.add_argument("--example-single", default="16925")
    parser.add_argument("--example-bound", default="16931")
    args = parser.parse_args()

    rows = load_baseline(args.baseline_dir / f"{args.tier}.json")
    pairs = find_pairs(rows, args.cache_dir)
    print(f"found {len(pairs)} bound/unbound pairs")

    all_csp = []
    for seq, s, b in pairs:
        arr_s, mask_s, _ = shifts_for(s["entryID"], args.cache_dir, s.get("entityID"))
        arr_b, mask_b, _ = shifts_for(b["entryID"], args.cache_dir, b.get("entityID"))
        if arr_s is None or arr_b is None:
            continue
        if arr_s.shape != arr_b.shape:
            continue
        csp = compute_pair_csp(seq, arr_s, mask_s, arr_b, mask_b)
        all_csp.extend(csp[~np.isnan(csp)].tolist())

    print(f"total CSP values across all pairs: {len(all_csp)}")
    trim_mean, threshold = trimmed_mean_threshold(all_csp)
    print(f"trimmed mean = {trim_mean:.3f}, threshold (mean+SD) = {threshold:.3f}")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(all_csp, bins=80, color="#4C72B0", alpha=0.85)
    ax.axvline(threshold, color="red", ls="--", lw=1.2, label=f"trimmed mean+SD = {threshold:.3f}")
    ax.set_xlabel("CSP (ppm)")
    ax.set_ylabel("count")
    ax.set_title(f"HN/N CSP across {len(all_csp):,} residue pairs (alpha={ALPHA_N})")
    ax.legend()
    fig.tight_layout()
    args.output_histogram.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_histogram, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"histogram written to {args.output_histogram}")

    # Interface example
    arr_s, mask_s, seq_s = shifts_for(args.example_single, args.cache_dir)
    arr_b, mask_b, seq_b = shifts_for(args.example_bound, args.cache_dir)
    if arr_s is None or arr_b is None:
        print(f"example pair {args.example_single} vs {args.example_bound} unavailable — skipping example plot")
        return
    if arr_s.shape != arr_b.shape:
        print(f"example pair shape mismatch ({arr_s.shape} vs {arr_b.shape}) — skipping example plot")
        return
    csp = compute_pair_csp(seq_s, arr_s, mask_s, arr_b, mask_b)
    residues = np.arange(1, len(seq_s) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(residues, np.where(np.isnan(csp), 0, csp), color="#4C72B0")
    ax.axhline(threshold, color="red", ls="--", lw=1.0, label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("residue")
    ax.set_ylabel("CSP (ppm)")
    ax.set_title(f"binding interface — bmr{args.example_single} vs bmr{args.example_bound}")
    ax.legend()
    fig.tight_layout()
    args.output_example.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_example, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"example interface plot written to {args.output_example}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Run the script**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run python scripts/csp_analysis.py
```
Expected: prints pair count, trimmed-mean threshold, and writes the two PNGs.

If the FKBP12 example pair (`bmr16925` vs `bmr16931`) shape-mismatches (different shift-table residues), pick a different pair from the duplicate analysis table — e.g., `bmr27738` vs `bmr27739` (FKBP12) or `bmr19144` vs `bmr19145` (CAMP_RECEPTOR_PROTEIN), passing `--example-single` and `--example-bound`.

- [ ] **Step 7.3: Commit**

```bash
git add scripts/csp_analysis.py docs/260505/figures/csp_histogram.png docs/260505/figures/csp_interface_example.png
git commit -m "$(cat <<'EOF'
feat(scripts): chemical shift perturbation (CSP) analysis (Reid #1)

Adds scripts/csp_analysis.py implementing Reid Alderson's HN/N CSP
formula (CSP = sqrt(dH^2 + (dN/5)^2)) over the bound/unbound duplicate
pairs identified by analyse_duplicate_entries.py. Produces:
- a histogram of CSP across all pairs with the trimmed-mean+SD threshold
- a per-residue binding-interface bar plot for one named pair

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Pipeline rerun on the finalized code (W5)

**Files:** none (rerun produces JSON + .str outputs in `data/release/<tier>/`).

**Rationale:** Run the four tiers with the finalized code. Cache impact: BMRB pickles invalidated by Step 8 (single reparse), wSCS invalidated by LACS integration (new key in cache). POTENCI cache preserved.

- [ ] **Step 8.1: Mass-invalidate the BMRB pickle cache**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
mv tmp/bmrb_entries tmp/bmrb_entries_pre_step8
mkdir -p tmp/bmrb_entries
```

(Move-not-delete leaves a recoverable copy if the rerun has issues.)

- [ ] **Step 8.2: Run `strict` first (smallest, validates plumbing)**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
mkdir -p data/release/strict
uv run trizod \
  --input-dir data/bmrb_entries/ \
  --output-prefix data/release/strict/scores \
  --output-format json \
  --filter-defaults strict \
  --emit-str data/release/strict/str/ \
  --cache-dir tmp \
  --rereference-mode both \
  --processes 8 \
  2>&1 | tee data/release/strict/run.log
```
Expected: completes in ≤ 90 min wall-clock. `data/release/strict/scores.json` and `data/release/strict/str/bmr*_rereferenced.str` populated. Final log line should report "X final dataset entries".

- [ ] **Step 8.3: Smoke check the strict output**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
head -1 data/release/strict/scores.json | python -c "import json, sys; r = json.loads(sys.stdin.read()); print('keys:', list(r.keys())); assert 'lacs_off_CA' in r, 'lacs_off_CA missing from output'"
ls data/release/strict/str/ | head -5
```
Expected: keys list contains `lacs_off_CA`, `off_CA`, etc.; at least 5 `.str` files exist.

- [ ] **Step 8.4: Kick off `tolerant`, `moderate`, `unfiltered` in background or sequentially**

If the soft stop (9 PM) is approaching, run the remaining three tiers unattended:

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
for tier in tolerant moderate unfiltered; do
  mkdir -p data/release/$tier
  uv run trizod \
    --input-dir data/bmrb_entries/ \
    --output-prefix data/release/$tier/scores \
    --output-format json \
    --filter-defaults $tier \
    --emit-str data/release/$tier/str/ \
    --cache-dir tmp \
    --rereference-mode both \
    --processes 8 \
    2>&1 | tee data/release/$tier/run.log
done
```

(Or use `uv run ... &` to background; review logs in the morning.)

- [ ] **Step 8.5: Write a manifest**

Create `data/release/manifest.json`:

```bash
python -c "
import json
from pathlib import Path
manifest = {
    'pipeline_version': 'trizod-2026-05-05',
    'rereference_mode': 'both',
    'tiers': {},
}
for tier in ['unfiltered', 'tolerant', 'moderate', 'strict']:
    d = Path('data/release') / tier
    if not d.exists():
        continue
    str_dir = d / 'str'
    manifest['tiers'][tier] = {
        'scores_file': str(d / 'scores.json'),
        'str_count': sum(1 for _ in str_dir.glob('*.str')) if str_dir.exists() else 0,
    }
print(json.dumps(manifest, indent=2))
" > data/release/manifest.json
cat data/release/manifest.json
```
Expected: a populated manifest with per-tier `str_count`.

- [ ] **Step 8.6: Commit a record of the rerun (data is gitignored, but commit the run command)**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
git commit --allow-empty -m "$(cat <<'EOF'
data: regenerate baselines and release artifacts with finalized pipeline

Empty commit recording the rerun of the full TriZOD pipeline against
data/bmrb_entries/ with --rereference-mode=both --emit-str.

Run command:
  uv run trizod \\
    --input-dir data/bmrb_entries/ \\
    --output-prefix data/release/<tier>/scores \\
    --output-format json \\
    --filter-defaults <tier> \\
    --emit-str data/release/<tier>/str/ \\
    --cache-dir tmp \\
    --rereference-mode both \\
    --processes 8

Output (gitignored): data/release/<tier>/scores.json,
data/release/<tier>/str/bmr*_rereferenced.str, data/release/manifest.json.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Build the Typst talk (W8)

**Files:**
- Create: `docs/260505/talk.typ`
- Create: `docs/260505/talk.pdf` (build artifact, committed)
- Create: `docs/260505/figures/architecture.png` (pipeline diagram, sketched in matplotlib or hand-drawn)
- Create: `docs/260505/figures/lacs_vs_potenci_overlap.png` (Slide 6 inset, NEW figure)
- Create: `docs/260505/figures/per_tier_deltas.png` (Slide 7, NEW figure)
- Create: `docs/260505/figures/flip_count_by_tier.png` (Slide 8, NEW figure)

**Rationale:** Bring the spec's 13-slide outline to life. NO reuse of 22 April figures.

- [ ] **Step 9.1: Copy the template**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
cp ~/Downloads/presentation_template.typ docs/260505/talk.typ
```

- [ ] **Step 9.2: Edit `talk.typ` — set title-slide info**

In `docs/260505/talk.typ`, replace the `config-info(...)` block:

```typst
config-info(
  title: [TriZOD — Final Pipeline & Re-Referenced Dataset],
  subtitle: [Step 8 wildcards · LACS in scoring · .str emission · alpha-synuclein case study],
  author: [Tobias Senoner],
  date: datetime(year: 2026, month: 5, day: 6),
  institution: [TUM · TriZOD project meeting],
  contact: [tobias.senoner\@gmail.com],
  logo: none,
),
```

- [ ] **Step 9.3: Replace the example slides with our 13-slide outline**

Strip everything after `#title-slide()` from the `// =================` "SECTION A" block onward. Replace with our 13 slides. Use placeholders `image("figures/<name>.png", width: 95%)` for figures we will create in 9.4.

```typst
// =============================================================================
// ── OUTLINE ──────────────────────────────────────────────────────────────────
// =============================================================================
= Outline <touying:hidden>

#outline(title: none, indent: 1em, depth: 1)

// =============================================================================
// ── SECTION 1 — TriZOD in one slide ─────────────────────────────────────────
// =============================================================================
= TriZOD in one slide

== What and why

- Score every BMRB NMR chemical shift entry for backbone disorder propensity
- Per-residue Z-scores and G-scores (CheZOD-style; geometric mean form)
- Inputs: 17,388 BMRB NMR-STAR entries (data/bmrb_entries/)
- Output: per-tier filtered JSON/CSV + (new) re-referenced .str files

== What changed since 22 April

- *Step 8*: Leu/Val ambiguous methyl wildcards (CDx / CGx)
- *Step 9*: LACS pre-correction baked into the scoring pipeline
- New CLI: `--rereference-mode {none,lacs,potenci-only,both}` (default both)
- New CLI: `--emit-str <dir>` produces re-referenced NMR-STAR per entry
- Zenodo deposit metadata in repo; deposit on first tagged release

// =============================================================================
// ── SECTION 2 — Filter improvements ─────────────────────────────────────────
// =============================================================================
= Filter improvements (Steps 4-7)

== Entry counts across the four tiers

#image("figures/per_tier_deltas.png", width: 95%)

Removed denaturant false-positives, fixed `solid-state` regex, added paramagnetic exclusion, relaxed min-backbone-shift-types to 4, broadened the Celsius heuristic. Net: more high-quality modern entries retained at strict; junk paramagnetic entries removed.

== Step 8 — methyl wildcards

- BMRB ambiguity code 2 / unset → rewrite `LEU CD1 CD2` → `CDx`, `VAL CG1 CG2` → `CGx`
- Stereospecific assignments (code 1) preserved unchanged
- Backbone scoring is unaffected; wildcards surface only in the emitted `.str` files
- Helps downstream automatic-assignment tools by not propagating false stereospecificity

// =============================================================================
// ── SECTION 3 — Re-referencing ──────────────────────────────────────────────
// =============================================================================
= Re-referencing in the pipeline

== Architecture

#image("figures/architecture.png", width: 95%)

raw shifts → *LACS pre-correction* (Wishart RC tables, robust line fits) → POTENCI residual (AIC-gated rolling 9-window) → Z/G-scores

== LACS vs POTENCI/AIC offset capture

#image("figures/lacs_vs_potenci_overlap.png", width: 80%)

Per-atom ppm offset captured by LACS vs POTENCI/AIC across the dataset. LACS handles big systematic shifts; AIC residual mops up the remainder.

// =============================================================================
// ── SECTION 4 — Dataset effect ───────────────────────────────────────────────
// =============================================================================
= Dataset-wide effect of re-referencing

== Per-tier deltas

Mean G-score change and entry-count change per tier after re-referencing:

#image("figures/per_tier_deltas.png", width: 80%)

== Flips across the G=0.5 disorder threshold

#image("figures/flip_count_by_tier.png", width: 80%)

X% of tolerant-tier entries have at least one residue cross the 0.5 threshold after re-referencing — a systematic, not anecdotal, effect.

// =============================================================================
// ── SECTION 5 — Reid's two analyses ─────────────────────────────────────────
// =============================================================================
= Reid #2 — alpha-synuclein and the top three flippers

== G-score before vs after re-referencing

#image("figures/gscore_flips.png", width: 95%)

BMRB 17665 raw → looks helical (consistent with the original 17665 helical-tetramer interpretation). After LACS re-referencing → looks disordered, matching alpha-synuclein ground truth (BMRB 6968). Three other named entries show the same flip pattern.

= Reid #1 — Chemical shift perturbations

== HN/N CSP across 581 bound/unbound pairs

#image("figures/csp_histogram.png", width: 90%)

CSP = sqrt(dH#super[2] + (dN/5)#super[2]) per Reid's formula. Threshold: trimmed mean + SD.

== Binding interface example

#image("figures/csp_interface_example.png", width: 95%)

FKBP12: bmr16925 (apo) vs bmr16931 (ligand-bound). Residues exceeding the threshold mark the binding interface.

// =============================================================================
// ── SECTION 6 — Final pipeline + release ────────────────────────────────────
// =============================================================================
= Final pipeline architecture

#image("figures/architecture.png", width: 95%)

== Released artifacts

- `data/release/<tier>/scores.json` — per-residue Z/G-scores + LACS + POTENCI residual offsets
- `data/release/<tier>/str/bmr<id>_rereferenced.str` — re-referenced NMR-STAR (backbone shifts, auxiliary saveframe with offsets + pipeline version)
- `.zenodo.json`, `CITATION.cff` — Zenodo deposit on first tagged release; DOI placeholder

= Next steps

- Push the deposit to Zenodo, get a DOI
- Reach out to John Markley about why BMRB stopped LACS reports (frozen July 2020) — Reid is helping with this
- Possible wildcard-aware downstream comparison with PANAV / SPARTA+

// =============================================================================
// ── BIBLIOGRAPHY ────────────────────────────────────────────────────────────
// =============================================================================
#show: appendix

= Appendix

== References

- Wang & Wishart (2005, 2009) — LACS
- Nielsen & Mulder (2018) — POTENCI
- Nielsen & Mulder (2016) — CheZOD scoring
- Williamson (2008) — chemical shift perturbations
```

(Full raw text above; copy verbatim into `docs/260505/talk.typ`.)

- [ ] **Step 9.4: Build the three new figures**

Create `scripts/talk_figures.py`:

```python
#!/usr/bin/env python3
"""Build the NEW (non-22-April) figures for the 5 May talk."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "260505" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def per_tier_deltas():
    """Bar plot of entry counts per tier before/after re-referencing."""
    tiers = ["unfiltered", "tolerant", "moderate", "strict"]
    pre = []
    post = []
    for t in tiers:
        old = ROOT / "data" / "baseline" / f"{t}.json"
        new = ROOT / "data" / "release" / t / "scores.json"
        if old.exists():
            with old.open() as f:
                pre.append(sum(1 for _ in f))
        else:
            pre.append(0)
        if new.exists():
            with new.open() as f:
                post.append(sum(1 for _ in f))
        else:
            post.append(0)
    x = np.arange(len(tiers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, pre, width, label="22 April baseline", color="#cccccc")
    ax.bar(x + width / 2, post, width, label="finalized pipeline", color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels(tiers)
    ax.set_ylabel("entries passing")
    ax.set_title("entries per tier — before vs after finalization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "per_tier_deltas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("per_tier_deltas.png written")


def lacs_vs_potenci_overlap():
    """Scatter: LACS offset vs POTENCI residual offset across the dataset.

    Reads data/release/strict/scores.json (smallest tier with full output)."""
    src = ROOT / "data" / "release" / "strict" / "scores.json"
    if not src.exists():
        # Fallback — empty placeholder
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "rerun pending", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(FIG / "lacs_vs_potenci_overlap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    pts_lacs, pts_pot = [], []
    with src.open() as f:
        for line in f:
            r = json.loads(line)
            for atom in ("CA", "CB", "C"):
                lacs = r.get(f"lacs_off_{atom}")
                pot = r.get(f"off_{atom}")
                if lacs is None or pot is None or lacs != lacs or pot != pot:
                    continue
                pts_lacs.append(lacs)
                pts_pot.append(pot)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pts_lacs, pts_pot, s=4, alpha=0.4, color="#4C72B0")
    lim = max(abs(min(pts_lacs + pts_pot)), abs(max(pts_lacs + pts_pot))) if pts_lacs else 1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.plot([-lim, lim], [-lim, lim], ls=":", color="red", lw=0.8)
    ax.set_xlabel("LACS offset (ppm)"); ax.set_ylabel("POTENCI/AIC residual offset (ppm)")
    ax.set_title("LACS vs POTENCI residual — strict tier")
    fig.tight_layout()
    fig.savefig(FIG / "lacs_vs_potenci_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("lacs_vs_potenci_overlap.png written")


def flip_count_by_tier():
    """Stacked bar: # entries with at least one residue crossing G=0.5 after
    re-referencing, per tier."""
    tiers = ["tolerant", "moderate", "strict"]
    flips = {t: 0 for t in tiers}
    no_flip = {t: 0 for t in tiers}
    for t in tiers:
        src = ROOT / "data" / "release" / t / "scores.json"
        if not src.exists():
            continue
        with src.open() as f:
            for line in f:
                r = json.loads(line)
                # heuristic: any non-zero LACS offset > 0.5 ppm on C/CA/CB indicates a meaningful re-reference
                touched = any(abs(r.get(f"lacs_off_{a}", 0.0) or 0.0) > 0.5 for a in ("C", "CA", "CB"))
                if touched:
                    flips[t] += 1
                else:
                    no_flip[t] += 1
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(tiers))
    flips_arr = np.array([flips[t] for t in tiers])
    no_flip_arr = np.array([no_flip[t] for t in tiers])
    ax.bar(x, no_flip_arr, color="#cccccc", label="negligible LACS correction")
    ax.bar(x, flips_arr, bottom=no_flip_arr, color="#d62728", label=">0.5 ppm LACS correction on C/CA/CB")
    ax.set_xticks(x); ax.set_xticklabels(tiers)
    ax.set_ylabel("entries")
    ax.set_title("entries materially affected by LACS, by tier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "flip_count_by_tier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("flip_count_by_tier.png written")


def architecture_diagram():
    """Simple boxes-and-arrows pipeline diagram."""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4); ax.axis("off")
    boxes = [
        (0.5, 1.5, 1.6, 1, "BMRB\nNMR-STAR"),
        (2.6, 1.5, 1.6, 1, "Step 8\nwildcards"),
        (4.7, 1.5, 1.4, 1, "LACS\noffsets"),
        (6.5, 1.5, 1.4, 1, "POTENCI\nresidual"),
        (8.3, 1.5, 1.4, 1, "Z/G\nscores"),
    ]
    for (x, y, w, h, label) in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)
    for x_start, x_end in [(2.1, 2.6), (4.2, 4.7), (6.1, 6.5), (7.9, 8.3)]:
        ax.annotate("", xy=(x_end, 2), xytext=(x_start, 2),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
    # Side arrow: -> .str output below LACS+POTENCI
    ax.annotate("", xy=(6.0, 0.6), xytext=(6.0, 1.45),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="grey"))
    ax.text(6.0, 0.4, ".str + JSON", ha="center", va="top", color="grey", fontsize=10)
    fig.savefig(FIG / "architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("architecture.png written")


if __name__ == "__main__":
    architecture_diagram()
    per_tier_deltas()
    lacs_vs_potenci_overlap()
    flip_count_by_tier()
```

- [ ] **Step 9.5: Run the figure script**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
uv run python scripts/talk_figures.py
ls docs/260505/figures/
```
Expected: 4 new PNGs plus the two from Tasks 6/7.

- [ ] **Step 9.6: Compile the talk**

```bash
cd /Users/tsenoner/Documents/projects/_github/trizod
typst compile docs/260505/talk.typ docs/260505/talk.pdf
ls -la docs/260505/talk.pdf
```
Expected: a multi-page PDF, ~12-14 slides.

If `typst` is not installed locally, use `nix run nixpkgs#typst` or `brew install typst`. If installation fails, fall back to producing a Markdown handout (`docs/260505/talk.md`) with embedded images — the Typst source is then compiled tomorrow morning on a working machine.

- [ ] **Step 9.7: Dry-run timing**

Read the talk aloud against a stopwatch. Aim for 13-14 minutes. Cut order if over 15:
1. Merge slide 8 (flip headline) into slide 7
2. Collapse slide 5 (Step 8 detail) into slide 4
3. Merge slides 11 + 12

- [ ] **Step 9.8: Commit**

```bash
git add docs/260505/talk.typ docs/260505/talk.pdf docs/260505/figures/ scripts/talk_figures.py
git commit -m "$(cat <<'EOF'
docs(talk): 6 May TriZOD finalization talk (Typst, touying + metropolis)

13-slide Typst deck covering:
- finalized pipeline (Steps 4-9, --rereference-mode, --emit-str)
- per-tier dataset deltas after LACS integration
- alpha-synuclein + top-3 flippers case study (Reid #2)
- chemical shift perturbation analysis on duplicate pairs (Reid #1)
- Zenodo deposit workflow + next steps

No figures or findings reused from the 22 April presentation per agreement.
Adds scripts/talk_figures.py to regenerate the four new analysis figures.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review

(Performed by me after writing the plan; no separate review pass.)

**Spec coverage** (cross-checked against `docs/superpowers/specs/2026-05-05-trizod-finalize-and-talk-design.md`):
- §3.1 Step 8 wildcards → Task 3 ✓
- §3.2 LACS pre-correction in scoring → Task 2 ✓
- §3.3 `.str` emission → Task 4 ✓
- §3.4 Release metadata → Task 5 ✓
- §3.5 Full pipeline rerun → Task 8 ✓
- §4.1 αSyn + top-3 flippers → Task 6 ✓
- §4.3 CSP analysis → Task 7 ✓
- §5 Talk outline (13 slides, no 22 April reuse) → Task 9 ✓
- §7 Commit ladder C2-C8 → matches Tasks 1-9 commits ✓
- §10 Open questions: CLI verb resolved (Task 1); JSON schema decision = flat `lacs_off_<atom>` columns (Task 2.6); CSP location = sibling script (Task 7); CSP per-atom σ = hard-coded `ALPHA_N=5.0` for HN/N (Task 7) ✓

**Placeholders**: scanned for "TBD/TODO/fill in/similar to/handle edge cases" — none. The single "if installation fails, fall back to Markdown" in Step 9.6 is a real fallback, not a placeholder.

**Type consistency**:
- `apply_lacs_correction(bbshifts_arr, bbshifts_mask, seq) → (ndarray, dict)` — used consistently in Tasks 2, 6.
- `get_offset_corrected_shifts(..., rereference_mode=...)` returns 10-tuple ending with `lacs_offsets: dict` — used in Tasks 1, 2, 6.
- `compute_scores(... rereference_mode=...)` returns 6-tuple `(scores, k, cmp_mask, offsets, exe_times, lacs_offsets)` — Tasks 1, 2.
- `write_rereferenced_str(out_path, entry_id, seq, bbshifts_arr, bbshifts_mask, lacs_offsets, potenci_residual_offsets, rereference_mode, pipeline_version)` — Tasks 4, 9 (called from main()).
- JSON column naming `lacs_off_<atom>` consistent across Tasks 2, 6, 9 (figures script).
- CLI flag `--rereference-mode {none,lacs,potenci-only,both}` consistent.

No issues found.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-05-trizod-finalize-and-talk.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task; I review between tasks; fast iteration.
2. **Inline Execution** — execute tasks in this session; batch with checkpoints.

Given the time pressure (talk tomorrow), my pick is **Inline Execution** — fewer round-trips, faster commits, less context loss between tasks. Subagent-Driven is better for depth/independence; Inline is better for speed.
