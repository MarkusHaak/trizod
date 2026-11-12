# Effect of LACS pre-correction on TriZOD G-scores

## Question

The 2026-05-05 release runs the pipeline with `--rereference-mode both`:
LACS pre-correction first, then the POTENCI/AIC residual on top.
A natural question follows: **how much does LACS actually move the
final G-scores compared to POTENCI/AIC alone?**

This is not an "independent effect" measurement — both modes share the
POTENCI/AIC step, so the residual differences attributable to LACS
will be smaller than LACS' raw offset magnitudes. That is the point:
**we want to know how much the LACS step changes the numbers users
finally consume**.

## Data

Source: `tmp/lacs_comparison_results.pkl`, produced by
`scripts/figures/compare_gscores_lacs.py`. The cache covers **14,217 BMRB
entries** (every entry where both pipelines successfully scored at
least one residue), totalling **1,439,864 residue-level G-score
pairs** (G-score_POTENCI-only, G-score_LACS+POTENCI).

Entry-to-tier mapping comes from
`data/baseline/{strict,moderate,tolerant,unfiltered}.json`; each entry
is assigned to the strictest tier it passes. Tier counts in the
cached comparison:

| tier | entries | residues |
|---|---:|---:|
| strict | 1,981 | 221,744 |
| moderate | 7,993 | 835,587 |
| tolerant | 3,158 | 315,346 |
| unfiltered | 1,085 | 67,187 |

## Figures

### `figures/lacs_effect_gscores.png`

Four-panel summary:

* **A** — Residue-level hexbin of G-score (POTENCI-only) vs
  G-score (LACS + POTENCI). Most density sits on the diagonal — for
  the majority of residues, LACS does not change anything. The visible
  off-diagonal density shows that, for a non-trivial fraction of
  residues, LACS shifts the G-score by tens of percentage points.
  Overall correlation r ≈ 0.96, MAE ≈ 0.02 per residue.
* **B** — Per-residue ΔG-score (LACS minus POTENCI-only) distribution
  per stringency tier (violin plot, clipped to [-0.6, 0.6]). The
  bulk of every distribution sits at 0 (median = 0 for all tiers).
  The tails extend furthest in *strict* — counter-intuitive at first
  but explained by the fact that strict-tier entries have enough
  chemical shifts for LACS to compute a confident, non-zero offset.
* **C** — Per-entry scatter: max |LACS offset| (ppm) on the x-axis vs
  max |ΔG-score| on the y-axis. Clear monotonic relationship: bigger
  referencing errors → bigger G-score effects. Vertical dotted lines
  mark 1, 2 and 3 ppm thresholds.
* **D** — Cumulative fraction of entries with max |ΔG-score| ≥ threshold,
  per tier. At a threshold of 0.10 (a 10-percentage-point G-score
  change for at least one residue), roughly **55% of strict entries**,
  46% of moderate, 22% of tolerant and 17% of unfiltered are affected.

### `figures/lacs_effect_affected_entries.png`

The same per-entry scatter restricted to entries where the LACS
offset is at least 0.5 ppm (8,021 / 14,217 entries). This is the
"there was actually something to correct" subset; the relationship
becomes crisper without the bulk of zero-effect entries.

## Numbers (per tier)

(From `data/lacs_effect_summary.csv`.)

| tier | entries | frac >0.5 ppm LACS | frac >1 ppm | frac >2 ppm | median max\|ΔG\| | p95 max\|ΔG\| | residue\|ΔG\|>0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict | 1,981 | 75.5% | 46.7% | 13.0% | 0.175 | 0.539 | 5.2% |
| moderate | 7,993 | 60.4% | 36.3% | 9.6% | 0.135 | 0.494 | 4.6% |
| tolerant | 3,158 | 42.5% | 31.1% | 13.1% | 0.000 | 0.425 | 3.8% |
| unfiltered | 1,085 | 32.6% | 23.8% | 11.2% | 0.000 | 0.374 | 4.0% |

Reading guide:

* "frac >X ppm LACS" — fraction of entries in this tier whose largest
  per-atom LACS offset exceeds X ppm.
* "median max|ΔG|" — for the median entry in this tier, how big is the
  *largest* G-score change introduced by LACS on any residue.
* "residue|ΔG|>0.1" — fraction of residues in this tier whose
  G-score changes by more than 0.10 when LACS is enabled.

## What this means

* **LACS rarely changes most residues**, but **frequently changes
  some residues in most entries**. Median per-entry maximum ΔG ≈ 0.17
  for strict and 0.13 for moderate: half of all entries in those
  tiers have at least one residue whose G-score shifts by a tenth
  to a fifth of the full [0, 1] range.
* The tier ordering is **not** monotone: strict-tier entries are more
  often affected than tolerant or unfiltered ones. This reflects
  LACS' requirement of ≥ 20 valid residues for a confident offset —
  strict entries are exactly the ones with enough data to detect the
  small biases LACS catches.
* **LACS pre-correction is not redundant with POTENCI/AIC offsets.**
  The G-score correlation between the two modes is high (~0.97) but
  not 1, and the tails are the very entries that downstream users
  most care about (e.g. αSyn 17665, where LACS recovers a ~2.8 ppm
  CA/CB offset that POTENCI/AIC alone misses).

## Reproducing

```bash
# 1. (long: a few hours) generate the cached comparison if you don't
#    have one already.  Reads BMRB pkls and runs both scoring branches.
uv run python scripts/figures/compare_gscores_lacs.py --compute

# 2. plot
uv run python -m trizod.figures.fig2_lacs
```
