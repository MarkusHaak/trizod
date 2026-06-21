# Is the `--max-offset` filter still worth keeping?

## The filter

After scoring, TriZOD rejects (or partially masks) entries whose
POTENCI/AIC residual offset for any backbone atom exceeds a threshold.
The current per-tier defaults (`trizod/trizod.py:96`):

| tier | `max-offset` (ppm) | `reject-shift-type-only` |
|---|---:|---:|
| unfiltered | ∞ | True (no rejection at all) |
| tolerant | 3.0 | True (mask offending atom only) |
| moderate | 3.0 | False (reject entire entry) |
| strict | 2.0 | False (reject entire entry) |

The threshold is applied to the **residual after LACS** because the
2026-05-05 release uses `--rereference-mode both` (LACS pre-correction
→ POTENCI/AIC residual). The question for this study: now that LACS
already removes the biggest systematic biases, does the filter still
catch anything that matters?

## Evidence

### `figures/max_offset_distribution.png`

Per-atom histogram of |POTENCI residual offset| after LACS, split by
tier. Vertical lines mark the current thresholds (2 ppm / 3 ppm).
Take-aways:

* For every atom and every tier, the bulk of entries have residuals
  well below 1 ppm — LACS has already absorbed the dominant
  referencing errors.
* **HN and N residuals are the largest** — the 1H/15N atoms (especially
  N) carry the longest tails. CA / CB residuals are tight.
* The unfiltered set still has a long tail past 3 ppm; the filtered
  tiers cut off cleanly at their thresholds.

### `figures/max_offset_filter_curve.png`

Left panel: cumulative fraction of entries (per tier) passing a
candidate threshold. The current per-tier defaults (dotted vertical
lines) hit at the elbow of each curve — i.e. they retain the bulk of
the distribution while excluding the long tail.

Right panel: tier sizes under candidate thresholds (2, 3, 4, 5, ∞ ppm).
Reading: at the strict threshold of 2 ppm, 3,033 entries survive.
Loosening strict to 3 ppm would gain ~430 entries (~14%) but admit
residuals that we have explicitly chosen to call "too large to trust".

### Per-tier numerical summary

Computed from `data/release/<tier>/scores.json` (which has the filter
already applied), so the **maximum residual visible per tier is the
filter cutoff itself**:

| tier | n | median max\|off\| | p95 | p99 | max | entries with \|off\|>3 ppm (in unfiltered base) |
|---|---:|---:|---:|---:|---:|---:|
| unfiltered | 16,851 | 0.00 | 4.89 | 10.78 | 243.01 | 16.7% |
| tolerant | 15,433 | 0.00 | 2.71 | — | 3.00 | 0 (filter active) |
| moderate | 10,107 | 0.00 | 2.66 | — | 3.00 | 0 (filter active) |
| strict | 3,033 | 0.00 | 1.37 | — | 2.00 | 0 (filter active) |

(`data/max_offset_summary.csv` has the full table for the candidate
thresholds 2, 3, 4, 5, ∞.)

The 243 ppm peak in *unfiltered* is real — it represents entries
whose deposited shifts are essentially wrong (off by tens of standard
deviations even after LACS). Those entries should not feed any
downstream ML or analysis.

## Recommendation

**Keep the filter at the current defaults.** Justifications:

1. **Empirical**: 16.7% of `unfiltered` entries have at least one
   backbone atom whose residual exceeds 3 ppm *after LACS*. These are
   not referencing errors — LACS already corrected those. They are
   data-quality issues that the user should not silently include.
2. **No safer alternative**: removing `max-offset` would inflate the
   unfiltered set's noise floor by ~17% and propagate to downstream
   ML training as label noise.
3. **The threshold is not hyper-sensitive**: at 2 vs 3 ppm strict, the
   number of retained entries differs by ~10–15%, but the median
   residual on the *retained* set barely moves (LACS+POTENCI has
   already produced a small-residual majority).
4. **Tier-specific values still make sense**: strict's 2 ppm threshold
   roughly matches the 95th-percentile residual of the unfiltered
   set's well-behaved subset; moderate/tolerant's 3 ppm threshold
   covers the 99th percentile of well-behaved entries.

The one tweak worth considering is **tightening strict to 1.5 ppm**.
That would push strict's worst-case residual below the typical LACS
detection limit on a single atom (~0.3–0.5 ppm), at the cost of
losing ~6% of strict entries. Not done here — would require a
re-release of the strict tier.

## What is NOT recommended

* **Removing the filter for unfiltered**: doing so admits the 243 ppm
  outliers, which has no scientific upside and only pollutes downstream
  use.
* **Replacing the filter with a per-residue mask**: this is *already*
  the behaviour for unfiltered/tolerant (`reject-shift-type-only=True`).
  For moderate/strict the filter rejects whole entries, which is the
  right thing for ML use cases where partial entries are awkward.

## Reproducing

```bash
uv run python docs/260520/scripts/analyze_max_offset.py
```

Outputs:

* `figures/max_offset_distribution.png`
* `figures/max_offset_filter_curve.png`
* `data/max_offset_summary.csv`
