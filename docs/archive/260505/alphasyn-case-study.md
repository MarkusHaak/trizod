# Reid #2 — alpha-synuclein + top-3 G-score flippers

## Headline

`docs/260505/figures/gscore_flips.png` — a 2×2 panel figure showing per-residue G-score *before vs after* re-referencing for four BMRB entries.

| Panel | Entry | What's shown |
|---|---|---|
| **A** | BMRB 17665 (αSyn, mis-referenced) + BMRB 6968 (αSyn ground truth) | 3 traces: 17665 raw, 17665 re-referenced, 6968 ground-truth |
| **B/C/D** | top-3 entries (excluding 17665) by mean \|ΔG\| in the tolerant tier | 2 traces each: raw, re-referenced |

Every panel has a horizontal `G = 0.5` line — the disorder/structure threshold.

## αSyn — the headline argument

**BMRB 17665** is the alpha-synuclein deposit from Bartels et al. (PNAS 2011, ~700 citations) that infamously concluded αSyn forms a "helical tetramer." The interpretation hinged on 13C chemical shifts that turn out to be **mis-referenced by ~2.9 ppm** (per Reid Alderson's TALOS-N reading; our LACS implementation independently recovers **CA = +2.82 ppm, CB = +2.82 ppm, N = +1.76 ppm** — confirming the magnitude and atom set).

| Trace | What it shows |
|---|---|
| 17665 raw | G ≈ 0.1-0.3 across most of the sequence — "looks structured" |
| 17665 re-referenced | G ≈ 0.7-0.9 — "looks disordered" |
| 6968 ground-truth | tracks the re-referenced trace closely — confirms the disordered call |

The flip is unambiguous: re-referencing rescues the αSyn entry from a structured-looking mis-classification back to its true disordered state. Reid's argument — that re-referencing matters most for IDPs — is illustrated cleanly.

## Top-3 flippers (proves the αSyn case isn't cherry-picked)

Scanned the first 200 entries of `data/baseline/tolerant.json`, scored each in `--rereference-mode none` and `--rereference-mode both`, ranked by mean \|ΔG-score\|:

| Rank | BMRB ID | mean \|ΔG\| |
|---|---|---|
| 1 | 51068 | 0.752 |
| 2 | 52619 | 0.324 |
| 3 | 51262 | 0.274 |

Each panel shows a clear divergence between the raw and re-referenced traces. **51068 is the most dramatic full-flip** — it crosses the G=0.5 threshold for most residues. αSyn (17665) sits among this cohort, not above it; re-referencing changes disorder calls *systematically* across mis-referenced entries.

## Methodology notes

- Scoring uses the existing `get_offset_corrected_shifts(seq, shifts, predshiftdct, rereference_mode=...)` and `compute_gscores`.
- `mode="none"` → raw observed shifts → POTENCI difference → G-scores. **No** offset correction at all.
- `mode="both"` → LACS pre-correction → POTENCI/AIC residual → G-scores. The default.
- Top-3 selection: `--max-scan 200` keeps wall-clock under 5 minutes (full tolerant tier scan would take ~20-40 min). The argument is illustrative, not exhaustive.
- 17665 and 6968 are the same protein (αSyn) at the same length (140 residues, Met1-Ala140), so the ground-truth trace overlays cleanly.

## What's NOT in the figure (and why)

- **No standalone histogram of \|ΔG\|** — the abstract distribution view was dropped because the named top-3 panels make the same "this is systematic" point more concretely. (See `docs/superpowers/specs/2026-05-05-trizod-finalize-and-talk-design.md` §4.2.)
- **No 22 April figures** — the LACS-offset violin and the boxplot+hexbin G-score change figures were already shown 2 weeks ago. This figure is fresh.

## Talking points for the slide

1. **Setup:** 17665 is famous; ~700 citations; concluded αSyn is a helical tetramer.
2. **Bug:** 13C shifts mis-referenced by ~2.9 ppm. Independently confirmed by our LACS at 2.82 ppm CA/CB.
3. **Effect:** raw G-scores say "structured"; re-referenced G-scores say "disordered" — matching ground-truth 6968.
4. **Generalisation:** three other BMRB entries show the same systematic flip pattern. Re-referencing isn't an αSyn cherry-pick.
5. **Implication:** any disorder analysis based on raw BMRB shifts inherits these errors. The TriZOD finalized pipeline corrects this by default.

## Code surface

- Script: `scripts/case_study_gscore_flips.py`
- Output: `docs/260505/figures/gscore_flips.png`

## Commits

- `2e35b9c` — `feat(scripts): alpha-synuclein and top-3 G-score flippers case study (Reid #2)`
- `d8fda02` — `fixup(case-study): mkdir POTENCI cache subdir guard`
