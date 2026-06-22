# Excursion: can we reproduce CheZOD1325 more faithfully?

An isolated experiment (no change to the release pipeline) to see how close
TriZOD's CheZOD-equivalent scoring (`--rereference-mode potenci-only`, no LACS)
can get to CheZOD's published per-residue Z-scores, driven by deep online +
codebase research (5-agent workflow). **Result: it is already as faithful as the
algorithm allows; no code change gets meaningfully closer — the residual is
input/data, not the scoring math.**

## Method

- Research workflow (4 parallel agents → 1 synthesis): exact CheZOD algorithm
  (Nielsen & Mulder 2016 + the `protein-nmr/CheZOD` `chezod1_1.py` source),
  POTENCI/conditions, the recovered 2023 "bit-equal" TriZOD code (git), and the
  current knobs.
- Experiment: re-score the 1320 CheZOD1325 entries (`tmp/chezod_subset`) in an
  isolated cache, measure with `docs/260611/scripts/chezod_faithful_measure.py`
  (legacy verdict = every residue within atol after sequence alignment).

## Baseline (current code, potenci-only)

| metric | value |
|---|---|
| entries compared | 1320 |
| legacy verdict, all residues ≤ 0.1 | **272** |
| all residues ≤ 0.01 | 186 |
| all residues ≤ 1e-4 | 0 |
| per-entry MAE bit-exact (<0.01) | 271 |
| median MAE | 0.188 Z |

## What the research found (and the experiment confirmed)

1. **The algorithm is already a faithful reimplementation.** POTENCI random-coil
   + per-nucleus weights + the |Δ|-capped (4.0) χ² over residue triplets + the
   Wilson–Hilferty transform are byte-identical to CheZOD's `chezod1_1.py`
   (`convChi2CDF`), and 186 entries match on *every* residue to ≤ 0.01.

2. **The suspected "drifted knobs" are red herrings.** Recovering the 2023
   bit-equal snapshot (commit `a9df3ac`, validated at `assert_allclose(atol=0.1)`
   against CheZOD's own reference files via the now-stubbed
   `test_chezod_equality.py`) shows the *active* generator (`new_main`) used:
   `use_ph_corr = pH != 7.0` (the `6.99<pH<7.01` band was dead `main()` code);
   `min_AIC = 6.0` with `N·ln(σ₀/σc) − 1` (the "AIC > 20 / disordered-only" is the
   2016 *paper*, not the POTENCI-based generator); the 9-residue rolling window;
   and the log-space Debye–Hückel `nan_to_num`. **All identical to current TriZOD.**

3. **The one real code regression since 2023 is a no-op here.** Commit `bb19cf0`
   replaced the running-offset raw-std `roll.apply(sqrt(mean(x²)))` with the
   algebraic identity `sqrt(std²+mean²)`. Reverting it and re-scoring changed
   **0 / 1320 entries** (max change 0.000).

4. **CheZOD's own quirks explain the biggest outliers.** `chezod1_1.py`
   hard-codes `skipCO = ['15719','15274','15506']` (drops the carbonyl for those
   3 buggy entries) — exactly the largest TriZOD↔CheZOD shifts (+2.4/+3.7/+4.6 Z).
   TriZOD does not replicate that bug. Separately, the reference is rounded to
   ~3 decimals (18,066 of ~20k sampled values), so true bit-exactness is
   unmeasurable.

5. **The residual is input-level.** It is per-residue, zero-centered, and
   uncorrelated with TriZOD's applied offset magnitude (Spearman 0.03). Only
   19.5% of residues match to 3 decimals; median per-entry max-diff ≈ 1 Z. Well-
   reproduced entries (all residues ≤ 0.1) are more often offset-free (64% vs 43%)
   and at pH 7 (32% vs 25%), but no condition cleanly separates them. The most
   likely dominant cause is **BMRB data version**: re-deposited/re-versioned
   entries (edited shift values, changed ambiguity codes or sample-condition rows)
   in `data/bmrb_entries/` vs CheZOD's era. The 2023 validation sidestepped this
   by only certifying entries whose parsed conditions matched CheZOD's at
   atol=0.01 — it never claimed full-set bit-exactness.

## Conclusion

TriZOD already reimplements the CheZOD algorithm faithfully (matching its own
2023 atol=0.1-validated generator); **no code knob improves the match** — the
suspected divergences are red herrings or no-ops. To certify reproduction going
forward, re-establish a regression test at atol=0.1 against
`allscores1325newest.txt`. The remaining gap is per-entry inputs (conditions +
BMRB data version) and CheZOD's own `skipCO`/rounding quirks, none of which are
worth changing in the release pipeline.

Reproduce: re-score `tmp/chezod_subset` with `--rereference-mode potenci-only
--filter-defaults unfiltered`, then
`uv run python docs/260611/scripts/chezod_faithful_measure.py <scores.json>`.
