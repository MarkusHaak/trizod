# Step 8 — Leu/Val ambiguous methyl wildcards

## What changed

The BMRB parser now rewrites side-chain methyl atom IDs to wildcards when the deposited assignment is non-stereospecific:

| Residue | Original `Atom_ID` | Rewritten | Trigger |
|---|---|---|---|
| LEU | `CD1` | `CDx` | ambiguity code ≠ `"1"` |
| LEU | `CD2` | `CDx` | ambiguity code ≠ `"1"` |
| VAL | `CG1` | `CGx` | ambiguity code ≠ `"1"` |
| VAL | `CG2` | `CGx` | ambiguity code ≠ `"1"` |

Stereospecific assignments (BMRB ambiguity code `"1"`) are **preserved unchanged**. Empty (`""`) or unspecified (`"."`) codes are treated as non-stereospecific (conservative — most BMRB depositors don't fill the field for ambiguous methyls).

## Why

Most NMR datasets do not actually distinguish Leu CD1 from CD2 (or Val CG1 from CG2) — the two methyls of each pair are NMR-equivalent absent dedicated stereoselective experiments. BMRB depositors nevertheless assign one shift to "CD1" and the other to "CD2" arbitrarily. Downstream automatic-assignment tools that consume these labels propagate the false stereospecificity into their predictions.

By renaming the atom IDs to a shared wildcard (`CDx`, `CGx`) when the ambiguity code says the assignment isn't stereospecific, TriZOD's released `.str` files no longer carry the false stereospecificity. Stereospecifically-assigned entries (e.g. BMRB 18414, where the depositor explicitly set ambiguity code `"1"`) are passed through unchanged.

## Scope

- **Backbone scoring is unaffected.** `get_valid_bbshifts` filters to backbone atoms (C/CA/CB/HA/H/HB/N) plus their stereo variants for HA/HB; CDx/CGx are not in the allowlist and silently drop. Z/G-scores don't change.
- **`.str` emission (Task 4) consumes the rewritten IDs** — emitted files reflect the wildcards.

## Code surface

| File | Change |
|---|---|
| `trizod/bmrb/bmrb.py` | New module-level helper `_maybe_wildcard_methyl(comp_id, atom_id, ambiguity_code)` + `_METHYL_WILDCARD_MAP` + `_STEREOSPECIFIC_CODES`. Applied inside `ShiftTable.__init__`'s row-grouping loop before storage in `self.shifts`. Comment notes the duplicate-row behaviour for downstream consumers when both methyls of a pair carry non-stereospecific codes. |
| `tests/test_methyl_wildcards.py` (new) | 15 parametrized helper cases (stereospecific preserved; ambiguous rewritten; missing code rewritten; non-Leu/non-Val passthrough) + 1 integration test on BMRB 15000 confirming `CDx`/`CGx` appear in the parsed shift table. |

## Talking points for the slide

- **What:** TriZOD's released `.str` files no longer carry false Leu/Val stereospecific assignments.
- **Why this matters:** Reid Alderson and Iva Pritisanac flagged that automatic-assignment tools (e.g. NEF-aware pipelines) propagate the bad labels. CD*/CG* is the standard convention in those tools.
- **Impact:** Cosmetic for backbone disorder scoring; substantive for downstream consumers of our re-referenced dataset.
- **Coverage:** ~half of all Leu/Val methyl carbon shifts in the BMRB (estimated from ambiguity-code statistics; exact figure quoted in the slide once the rerun completes).

## Note on αSyn case study

BMRB 17665 (the αSyn case-study entry for Reid #2) doesn't actually carry Leu CD or Val CG carbon shifts — it has only the proton methyls (HD, HG). So Step 8 doesn't visibly affect that specific entry. The integration test fixture is BMRB 15000, which has Leu CD2 and Val CG2 carbons with ambiguity code `.`.

## Commits

- `0614bfa` — `feat(bmrb): wildcard naming (CDx/CGx) for ambiguous Leu/Val methyls`
- `655293c` — `fixup(bmrb): drop Step-8 planning prefix from comments + duplicate-row note`
