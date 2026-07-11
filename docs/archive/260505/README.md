# 2026-05-06 — TriZOD final pipeline & dataset release

This directory holds the source material for the **6 May 2026** TriZOD project meeting talk: per-section markdown summaries that distill the work done on each component, plus the figures used in the deck.

The talk itself lives in `talk.typ` (Typst, touying + metropolis theme); compile with `typst compile talk.typ talk.pdf`.

## Layout

| File | Content |
|---|---|
| `README.md` | This index. |
| `talk.typ` | Typst source of the 13-slide presentation. |
| `talk.pdf` | Compiled deck. |
| `figures/` | All PNGs referenced from `talk.typ`. |
| `step8-methyl-wildcards.md` | Step 8: Leu/Val ambiguous methyl renaming (CDx / CGx). |
| `step9-lacs-pipeline-integration.md` | Step 9: LACS pre-correction integrated into scoring; `--rereference-mode` flag. |
| `str-emission.md` | `--emit-str` flag and the re-referenced NMR-STAR output format. |
| `release-metadata.md` | `.zenodo.json`, `CITATION.cff`, README releases section. |
| `pipeline-rerun.md` | Full rerun against `data/bmrb_entries/` with `--rereference-mode=both --emit-str`. |
| `alphasyn-case-study.md` | Reid #2 — αSyn 17665 + top-3 flippers analysis. |
| `csp-analysis.md` | Reid #1 — chemical shift perturbations on bound/unbound duplicates. |
| `talk-outline.md` | Slide-by-slide outline + speaker notes. |

## Constraint

**No content in this directory may reuse figures or findings from the 22 April 2026 talk** (`docs/260422/`). The April deck already covered: LACS-offset violins, the boxplot+hexbin G-score change figure, the duplicate-entry table, and the BindBox comparison. This deck stands on the *finalized* pipeline plus Reid's two new analyses (αSyn case study and CSP).
