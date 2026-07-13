# CheZOD reference data — provenance & re-verification plan

Fetched 2026-06 to support the manuscript's parsing-superiority claim and as
the external reference distribution. Stored under `data/chezod/` (gitignored).

## What was fetched

| File | Source | Contents |
|---|---|---|
| `protein_nmr_1325/allseqs1325.txt` | ODiNPred (Dass, Mulder & Nielsen 2020), via Wayback snapshot `20230406054806` of `http://www.protein-nmr.org/CheZOD1325.tar.gz` | **Canonical CheZOD1325**: 1325 lines `<BMRB_ID> <sequence>` (1323 numeric BMRB IDs + 2 non-BMRB: `dss1`, `hmbdi`) |
| `protein_nmr_1325/allscores1325newest.txt` | same tarball | Per-residue CheZOD Z-scores, **line-aligned** to `allseqs1325.txt` (`999` = no data) |
| `seth_github/CheZOD1174_training_set_sequences.fasta` | SETH repo `github.com/DagmarIlz/SETH` | SETH training subset, 1174 seqs, FASTA headers = BMRB IDs |
| `seth_github/CheZOD1174_training_set_CheZOD_scores.txt` | same | `<BMRB_ID>:\t<comma-sep per-residue Z-scores>` (`999` = no data) |
| `seth_github/CheZOD117_test_set_sequences.fasta` | same | SETH test set, 117 seqs |
| `chezod1325_bmrb_ids.txt`, `chezod1174_bmrb_ids.txt`, `chezod117_bmrb_ids.txt`, `chezod_union_bmrb_ids.txt` | derived | de-duplicated BMRB-ID lists |

The original `http://www.protein-nmr.org/CheZOD1325.tar.gz` and `allseqs1325.fasta`
URLs now 404 (link rot); the JKU "protein-nmr.org" page still lists them but the
files are gone. Recovered via the Internet Archive.

## Key facts (verified)

- Canonical CheZOD1325: **1323 unique numeric BMRB IDs**; **1322/1323 present in
  our local `data/bmrb_entries/` cache** (1 absent) → TriZOD can score ~all of
  them.
- SETH split: CheZOD1174 (train) + CheZOD117 (test) = 1291 BMRB IDs, 0 overlap.
- Canonical-1325 vs SETH-1291: 1257 shared; 66 only in 1325; 34 only in SETH
  (SETH is a later redundancy-reduced/updated variant).
- Our frozen test set `data/2024-05-09/CheZOD117_test_set.fasta` = 115 BMRB IDs,
  all 115 ⊂ SETH's 117 (2 of 117 unmapped at 95% id / 90% cov).

## Re-verifying the parsing-superiority claim (manuscript)

The original report claims CheZOD's parser mis-associated 11 and errored on 5 of
the 1325-protein set, and that >1700 BMRB entries are inaccessible to it. To
reproduce on the current data:

1. Take the 1323 canonical CheZOD1325 BMRB IDs (`data/chezod/chezod1325_bmrb_ids.txt`).
2. Run TriZOD's parser over those entries (1322 are cached) and record successes.
3. Obtain CheZOD's own parser (the "CheZOD source code" referenced in
   `trizod/constants.py` for `REFINED_WEIGHTS`) and run it over the same entries.
4. Diff: entries TriZOD parses correctly that CheZOD mis-parses (target: ~11) or
   errors on (~5); cross-check residue-level Z-scores where both succeed.

Step 3 (locating/running CheZOD's parser) is the open dependency; steps 1–2 are
runnable now. Not yet executed.
