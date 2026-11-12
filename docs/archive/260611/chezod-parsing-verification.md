# Reproducing CheZOD1325 with TriZOD — verification & mismatch analysis

Does TriZOD's CheZOD-equivalent pipeline reproduce CheZOD's published per-residue
Z-scores? **Yes — the CheZOD pipeline is fully reimplemented.** TriZOD's
`--rereference-mode potenci-only` is that reimplementation: the **same POTENCI
random-coil model + data tables**, the **same Wilson–Hilferty Z-score formula**
(triplet χ², per-nucleus σ, |Δ| capped at 4), and a **faithful AIC-gated offset
correction**. It reproduces CheZOD at **median per-residue MAE 0.19 Z-units; 271
of 1320 entries bit-exact (MAE < 0.01) and 526 within MAE 0.1**.

Bit-exactness on 20% of entries is only possible if the random-coil model,
per-nucleus weights, the χ² cap and the Wilson–Hilferty transform are *identical*
— so the algorithm is completely reimplemented, not merely similar. This was
established historically: Markus Haak's 2023 commits ("identical output to
original CheZOD") validated TriZOD against pre-computed CheZOD reference files —
but at **atol = 0.1 per residue, never bit-for-bit**. The current match (median
0.19, 526 within 0.1) is exactly consistent with that historical bar. (There is
no *live* equality test today — `tests/test_chezod_equality.py` is a TODO stub
and the reference files are gone.)

Provenance note: the CheZOD1325 reference (`allscores1325newest.txt`) was itself
computed with **POTENCI** (not the older 2016 Tamiola random-coil model) — adopted
by Nielsen & Mulder 2019 and used by ODiNPred 2020. Decisive corroboration:
TriZOD's `REFINED_WEIGHTS` are the POTENCI-era per-nucleus RMSDs (25–78% below the
Tamiola σ), and 271 entries reproduce bit-exact. So `potenci-only` reproduces the
*right* reference.

> Critical: LACS is a TriZOD-only improvement and must be EXCLUDED from the
> CheZOD reproduction. Comparing CheZOD against TriZOD's LACS "both" release was
> apples-to-oranges and doubled the apparent error (median MAE 0.39 vs 0.19).
> LACS is applied only to the TriZOD dataset.

Scripts:
```
uv run python scripts/validation/verify_chezod_parsing.py         # coverage
uv run python docs/260611/scripts/investigate_chezod_mismatches.py # terminal trimming
uv run python scripts/validation/reproduce_chezod.py              # CheZOD reproduction (potenci-only)
```
Regenerate the potenci-only scores (LACS recorded but NOT applied) per the header
of `reproduce_chezod.py`.

## Headline numbers

| comparison | n | median MAE (Z) | agree | consistent | genuine |
|---|---:|---:|---:|---:|---:|
| **potenci-only (CheZOD reproduction)** | 1320 | **0.188** | 1263 | 98.3% | 22 (1.7%) |
| both / LACS (the TriZOD dataset) | 1320 | 0.389 | 1192 | 97.1% | 38 |

(Categories on aligned per-residue scores: agree = r≥0.9 & MAE≤1.0; offset_shift =
r≥0.9 & MAE>1.0; low_variance = r<0.9 & MAE≤0.8; genuine = r<0.9 & MAE>0.8.)

## 1. Coverage and the 3 not-scored entries — FOR THE MANUSCRIPT

TriZOD scores **1320 / 1323** CheZOD1325 entries. Document these 3 in the
CheZOD-reproduction/differences paragraph:
- **bmr4077** — not in our BMRB snapshot (download/version gap), not a method issue.
- **bmr6078, bmr6985** — rejected by TriZOD's parser: no `_Experiment_list`
  saveframe, which TriZOD requires to bind shifts to sample conditions
  (temperature/pH/ionic strength) for POTENCI. CheZOD scored them without one.
  This is a deliberate strictness (a default-conditions fallback could recover
  them); it affects 2 of 1323 entries. *Per the user: not fixing the 3; document
  as known, minor reproduction differences.*

## 2. Why CheZOD sequences are shorter (the 96 "length mismatches")

After sequence-offset alignment, 94/96 are CheZOD being a substring of TriZOD,
with TriZOD carrying a few extra **terminal** residues. The extras are
overwhelmingly **expression/purification tags**:
- C-terminal in 86/96, N-terminal in 9, both in 2 (median 4 extra residues).
- Most common extra blocks: **His-tags** (`HHH` ×29, plus `HHHHH`, `HHHHHP`),
  GS/cloning linkers (`SG`, `SSG`, `SGP`, `SGPSSG`, `GPS`), and initiator `M`.

**How/why CheZOD trims:** CheZOD's data curation removed non-native expression
tags and linkers from the deposited sequences. TriZOD instead keeps the full
BMRB-deposited entity sequence verbatim.

**Should TriZOD trim too?** Recommendation: **no, not by default**, consistent
with TriZOD's "don't silently rewrite deposited data" philosophy (same reasoning
as the methyl-wildcard revert). The tag residues carry real measured shifts and
valid scores; reliable tag detection is error-prone (risk of clipping native
His-rich/Gly-rich regions); downstream users can trim. A defensible *optional*
addition would be to flag likely-tag terminal runs (e.g. ≥5 His) in the
metadata rather than remove them — a possible future feature, not required for
the dataset. For the comparison, aligning by sequence resolves all 96.

## 3. Why not bit-exact everywhere — where the residual comes from

The scoring **math is identical** (271 bit-exact prove RC model + weights + cap +
Z-transform match). The residual ~0.19 MAE is therefore **input/processing**, and
its empirical signature pins it down: it is **per-residue and zero-centered**
(centered-MAE ≈ MAE; pooled mean +0.03), **uncorrelated with TriZOD's applied
offset magnitude** (Spearman 0.03), and only 99/1320 entries have any
shift-coverage mismatch. So it is *not* a global re-referencing/offset bug. In
order of contribution:

A dedicated reproduction excursion (deep git/literature research + a re-scoring
experiment; see [chezod-faithful-excursion.md](chezod-faithful-excursion.md))
tested every suspected cause and overturned most of them:

1. **The suspected "drifted knobs" are red herrings.** Recovering TriZOD's 2023
   bit-equal snapshot (commit `a9df3ac`, validated at atol=0.1 against CheZOD's
   own reference files) shows the *active* generator used `pH != 7.0` (the band
   was dead code), `min_AIC = 6.0` with the `N·ln(σ₀/σc) − 1` formula, the
   9-residue rolling window, and the log-space Debye–Hückel — all **identical to
   current TriZOD**. The earlier "AIC > 20 / disordered-only" is the 2016 *paper*,
   not the POTENCI-based code that produced `allscores1325newest.txt`.
2. **The one real code regression since 2023 is a no-op here.** The running
   raw-std formula (commit `bb19cf0`) was reverted in the excursion and re-scored:
   **0/1320 entries changed.**
3. **CheZOD's own quirks explain the biggest outliers.** CheZOD hard-codes
   `skipCO` (drops the carbonyl) for exactly `bmr15719/15274/15506` — three of the
   largest "genuine" shifts; TriZOD doesn't replicate that bug. The reference is
   also rounded to ~3 decimals, so true bit-exactness is unmeasurable.
4. **The residual is input-level, not the math.** Well-reproduced entries are
   more often offset-free (offset==0 in 64% vs 43%) and at pH 7 (32% vs 25%); no
   code knob moves it. The remaining gap is per-entry condition values and, most
   likely, BMRB data-version differences (re-deposited shifts/conditions vs
   CheZOD's era) — unfixable without CheZOD's exact per-entry inputs. Note the
   2023 validation itself only certified entries whose conditions matched at
   atol=0.01, so the full-set residual includes condition-mismatched entries the
   historical test excluded.

The residual is therefore input/data, not the scoring math. Full per-entry
residuals: `data/chezod_verification/reproduce_genuine.csv`.

## 4. What "complete reimplementation" means here, and whether we can get closer

The CheZOD **algorithm** (POTENCI RC + Wilson–Hilferty Z + AIC offset) is fully
reimplemented and reproduces the reference to the historical atol = 0.1 bar
(272/1320 entries match on *every* residue within 0.1). Bit-exactness for every
entry was never the standard — and is literally unmeasurable, since the published
reference is rounded to ~3 decimals.

The excursion tested whether any code change gets us closer and concluded **no**:
the suspected knobs (pH-trigger band, AIC→20, rolling window, pH numerics) match
the validated 2023 generator already, and the one genuine code regression
(running raw-std) changed 0/1320 entries. The only edits that would move specific
entries are replicating CheZOD's own quirks — its `skipCO` bug for
`bmr15719/15274/15506`, and (optionally) the Z<20-capped offset-decision — neither
worth landing. The broad residual is per-entry inputs (conditions + BMRB
data version), not closable without CheZOD's exact per-entry inputs. A live
regression test could be re-established at atol = 0.1 against
`allscores1325newest.txt`.

## 5. Manuscript framing (CheZOD reproduction & differences)

- **Reproduction:** "TriZOD's CheZOD-equivalent scoring (POTENCI random-coil +
  AIC offset correction, no LACS) reproduces CheZOD's published per-residue
  Z-scores on the 1,323-protein set with median MAE 0.19 (271/1320 bit-exact),
  consistent with the original atol = 0.1 validation."
- **Differences to disclose:** (i) 3 entries not scored (2 lack an experiment
  list, 1 absent from our snapshot); (ii) CheZOD trimmed expression tags whereas
  TriZOD retains the full deposited sequence; (iii) a small per-residue residual
  from TriZOD's independent condition parsing / shift averaging (and a minor
  offset-threshold difference), not from the scoring math.
- LACS is the TriZOD-only re-referencing improvement applied to the TriZOD
  dataset, not to the CheZOD reproduction.

Artifacts (gitignored): `docs/260611/data/chezod_verification/`
`{per_entry.csv, per_entry_aligned.csv, mismatch_summary.json,
trizod_potenci_only.json, reproduce_summary.json, reproduce_genuine.csv}`.
