# Reverting the Step 8 methyl-wildcard rewrite

## Decision

The Step-8 methyl-wildcard rewrite shipped in commit `0614bfa` and
documented in `docs/archive/260505/step8-methyl-wildcards.md` is **reverted**.

The BMRB parser no longer renames Leu CD1/CD2 → CDx or Val CG1/CG2 → CGx
when the deposited ambiguity code is non-stereospecific. The atom IDs
are now passed through verbatim, matching the rest of the codebase's
philosophy ("do not silently rewrite deposited data").

## Why

* The user determined that downstream NMR tooling (NEF-aware pipelines)
  should be responsible for handling its own stereospecificity
  canonicalisation. Doing it inside TriZOD's BMRB parser obscures the
  provenance of the rewrite from anyone reading the released `.str`
  files.
* Backbone scoring is unaffected (CDx/CGx were never in the backbone
  atom allowlist), so removing the step has **zero effect on the
  Z-scores and G-scores** in any of the 4 released tiers.
* The released `.str` files (in `data/release/<tier>/str/`) shipped in
  the 2026-05-05 release still carry CDx/CGx tokens. Re-running
  `trizod --emit-str ...` against the same release configuration is a
  cosmetic follow-up; it is **not blocking** for any downstream
  consumer of the scores.

## Changes

| File | Change |
|---|---|
| `trizod/bmrb/bmrb.py` | Removed module-level `_METHYL_WILDCARD_MAP`, `_STEREOSPECIFIC_CODES`, `_maybe_wildcard_methyl` helper and the rewrite hook inside `ShiftTable.__init__`. |
| `tests/test_methyl_wildcards.py` | Deleted (15 parametrized cases). |

The diff is small; no other code referenced the wildcard machinery.

## Verification

```bash
uv run pytest tests/                  # 20 passed (excluding slow regression)
uv run ruff check trizod/ tests/      # clean
```

The pipeline regression test (`tests/test_pipeline_regression.py`) was
**not** rerun in this commit because the reference baseline
(`tests/reference/unfiltered.json`) was generated *before* Step 8 was
shipped — backbone scoring is identical with or without the wildcard
step, so the existing baseline remains a valid reference for the
post-rollback parser.

## Follow-up (not done here)

* Re-emit the 4 released `.str` directories with the cleaned parser so
  the deposited stereospecific labels reach downstream consumers
  unchanged. Estimated cost: ~10 minutes given POTENCI cache is
  already warm.
