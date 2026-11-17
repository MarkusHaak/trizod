# TriZOD dataset v{{version}}: validation decisions

<!-- Template: `trizod dataset finalize --validation <VALIDATION.json>` renders
it into the deposit's VALIDATION.md, beside VALIDATION.json (release
validation 2026-11-17c, "What you must do at deposit time", item 1). It records
the human decisions the release checklist asks for; VALIDATION.json holds the
check-by-check results. Placeholders in double braces are filled from this
build; dated counts are those of the audited builds and say so. Update the G37
section when the maintainer has signed off, and re-run finalize. -->

`VALIDATION.json` holds the result of every check of the final release audit;
this file records the decisions a person made where the checklist asks for one.
The test set is {{test_set_label}}; the leakage rule is {{leakage_rule}}.

## A3: ionic-strength drivers

**Kept as deposited.** The audit's driver log lists the 40 chains whose
deposited ionic strength decides their strict membership; all 40 keep the
deposited value. The strict cap is 0.5 M, applied to the value as deposited;
where a chain's own sample components imply a value at or below the cap, it is
listed in `known_issues.tsv` (`ionic_strength_contradicts_components`) and
stays out of strict. Decision of 17 November 2026.

## E21: overlap with earlier test sets

**No per-row overlap column.** Earlier TriZOD test sets are not comparable
with this one, and a per-row column would have to be redefined at every
release. The deposit ships `crosswalk/` instead: every earlier test set, and
where each of its sequences is in this release (exact match, a lower bound),
with the clean-subset rule in the README. The overlap with
UdonPred's published training data (figshare 10.6084/m9.figshare.31444642) is
computed after the freeze, once the download is approved.

## E20 / E22: alternative alignments and shared accessions

**Option (a), rebuild.** mmseqs2 computes a pair's alternative alignments only
behind a best alignment that passes `--min-seq-id`; with 0.3, two training
representatives met the rule on an alternative alignment the search never
computed (36490_1_1_1 against retired 53299_1_1_1, 50596_1_1_1 against test
chain 30007_1_1_1). Every leakage search (stage 2, the release gate, the
CheZOD-overlap flags, the within-test check, the homology components of the
UdonPred validation split and of the test set) now runs with
`--min-seq-id 0` and applies the rule to every reported alignment. On the
2026-11-17c build this removes exactly the two chains (2,568 training chains
removed by stage 2 instead of 2,566); the stage-2 searches take 8.9 s instead
of 8.2 s.

**Shared accessions: kept.** {{shared_accessions}}. The three overlapping
constructs the audit named below the rule (52143_1_1_1 ~ 53627_1_1_1, a 37-aa
identical run; 50077_1_1_1 ~ 50079_1_1_1, 29 aa; 25975_1_1_1 ~ 6946_1_1_1,
21 aa) are kept: none meets the rule, and the shared-construct criterion (E26)
does not retire them.

## E26: shared constructs

**One written criterion.** {{shared_constructs}}. The criterion, the tag
patterns and every pair's outcome are in the code repository
(`trizod/dataset/data/shared_construct_review.tsv`,
`docs/dataset/dataset-construction.md`). 7070_1_1_1 (Prp24 RRM1-2; an
83-residue run identical to training 16243_1_1_1 holds 51.9 % of its scored
residues) is retired as `shared_construct`.

## E24: the resampling unit

Test chains are not independent: {{test_components}}. Resample components,
not chains or residues.

## C8: membrane mimetics

**LPS policy.** Lipopolysaccharide micelles and bilayers count as membrane
mimetics, everywhere and by rule (class `LPS`); reviewed overrides only where
the LPS is named outside the sample (title, assembly name, citation), as in
transferred-NOE depositions on one sample.

**Reviewed residuals.** The 28 moderate and strict chains the class rule left
unflagged although they look like membrane-mimetic samples were reviewed
against their raw NMR-STAR files and added to the code repository's
`trizod/dataset/data/membrane_overrides.tsv`: 7 definite entries (the mimetic
is in the file but outside the class table or the fields the rule reads:
sulfobetaines, LMNG, a C14 betaine, octyl glucoside in `_Sample.Details`, a
chain named "ctrdpc") and 15 integral membrane proteins measured by solution
NMR whose deposition names no mimetic (class `unnamed mimetic`, probable).
Three are not listed (5376: the free peptide over native disk membranes; 4151
and 6056: probably an organic solvent, not a mimetic) and appear in
`known_issues.tsv` (`membrane_mimetic_reviewed_not_listed`). 34151_1_1_1 is
the known false positive (`membrane_class_false_positive`). The overrides and
the new cosolvent spellings (isopropanol `isoPrOH` and `IPA`, the hyphenated
TFE formula) take effect at scoring: they need a rebuild from the scoring
step.

## C12: the re-baselined projection

The reference is the 2026-11-16 projection with the decided membrane policy:
moderate 10,779 projected against 10,690 rebuilt in 2026-11-17c (−0.83 %).
With the deposit's own membrane flag in place of the projection's rule it is
10,661 (+0.27 %); the remaining 39 chains are all 273 K re-scores. The strict
per-rule losses are 3/125/43/146 against the projected 3/105/43/146. The test
set size is the decided v1.0 size, the pin's 452 chains less the retired ones.
The C8 overrides and the cosolvent spellings move moderate and strict further
(this build's counts are in the CHANGELOG); the projection is re-baselined on
each rebuild, not carried over. Audit record only, no defect.

## G37: maintainer sign-off

**Pending.** After the hand-over the UdonPred maintainer runs
`python run.py train` with ProstT5 embeddings on the shipped `udonpred/`
`trizod_tolerant/` view and confirms the `{id, x_0, y}` / 999 schema in
writing. Record the sign-off (name, date, the deposit's MANIFEST SHA-256) here
and re-run finalize; publication waits for it.
