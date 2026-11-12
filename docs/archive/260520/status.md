# Project status & open questions — 2026-05-20

## TL;DR

The TriZOD chemical-shift pipeline is **production-ready**. The May 2026
release (`data/release/{unfiltered,tolerant,moderate,strict}/scores.json`)
contains 16,851 / 15,433 / 10,107 / 3,033 entries respectively, scored
with `--rereference-mode both` (LACS pre-correction → POTENCI/AIC
residual). All upstream subsystems (BMRB parsing, POTENCI cleanup,
LACS, paramagnetic / blacklist / temperature-unit filters,
re-referenced `.str` emission) are merged and tested.

What this folder finishes:

* removed the methyl-wildcard rewrite step (now deemed unwanted),
* quantified what LACS actually buys us at the G-score level,
* asked whether the `--max-offset` filter still earns its place after
  LACS,
* built the first end-to-end **ML-ready dataset**: bound-complex
  removal → sequence-level dedup → mmseqs2 redundancy reduction at
  the paper's exact parameters.

## What's done (2026-05-20)

| Track | State | Pointer |
|---|---|---|
| Pipeline orchestration & CLI | ✅ shipped | `trizod/trizod.py` |
| BMRB parser | ✅ shipped | `trizod/bmrb/bmrb.py` |
| POTENCI random-coil predictions | ✅ shipped, cached | `trizod/potenci/` |
| LACS re-referencing, integrated | ✅ shipped | `trizod/lacs/`, `trizod/scoring/scoring.py` |
| LACS validated vs BMRB LACS reports | ✅ 6,692 entries cross-checked | `scripts/compare_lacs_bmrb.py` |
| αSyn (17665) case study | ✅ documented | `docs/260505/alphasyn-case-study.md` |
| CSP analysis on bound/unbound pairs | ✅ shipped | `docs/260505/csp-analysis.md` |
| Per-tier release (4 tiers) | ✅ released | `data/release/manifest.json` |
| `.str` emission with LACS offsets | ✅ shipped | `--emit-str` |
| Methyl wildcard step | ✅ **REVERTED** today | [methyl-wildcards-removal.md](methyl-wildcards-removal.md) |
| LACS effect on G-scores quantified | ✅ today | [lacs-effect-on-gscores.md](lacs-effect-on-gscores.md) |
| `max-offset` filter rationale | ✅ today | [max-offset-analysis.md](max-offset-analysis.md) |
| Multi-molecule (bound) filter | ✅ today | [final-dataset.md](final-dataset.md) |
| Sequence dedup with quality ranking | ✅ today | [final-dataset.md](final-dataset.md) |
| mmseqs2 redundancy reduction (per paper) | ✅ today | [mmseqs-pipeline.md](mmseqs-pipeline.md) |

## Outstanding open questions

1. **CD/CG side-chain ambiguity, downstream consumers.** Step 8 wildcards
   (CDx / CGx) were rolled back because backbone scoring is unaffected
   and downstream NEF tools are willing to do their own wildcarding.
   The flip-side is that the released `.str` files now carry the
   BMRB-deposited (potentially false) stereospecific labels.
   *If* a downstream consumer cares about this, they should run NEF /
   PDB-style canonicalisation themselves, not rely on TriZOD.
   See [methyl-wildcards-removal.md](methyl-wildcards-removal.md).

2. **`max-offset` threshold values.** The empirical analysis
   ([max-offset-analysis.md](max-offset-analysis.md)) shows that LACS
   already deals with the dominant referencing errors and the residual
   POTENCI/AIC offsets are usually small. The current tier defaults
   (3.0 / 3.0 / 2.0 ppm) remove ~16.7% of unfiltered entries that
   still carry > 3 ppm of residual after LACS — these are
   pathological cases (max observed = 243 ppm) and worth excluding.
   **Recommendation: keep the filter at the current values.**
   The most defensible change would be tightening strict to 1.5 ppm,
   which loses ~6% of strict entries but produces a tier where no
   atom carries > 1.5 ppm of unexplained residual.

3. **Final dataset is conservative on "bound"**. The bound-complex
   filter today flags an entry as multi-molecule if *any* of the
   following holds: more than one distinct entity in some assembly,
   has a non-polymer entity (ligand), nucleic-acid entity, or metal
   entity. This drops ~27% of the BMRB pkl cache; for tier-specific
   counts see [final-dataset.md](final-dataset.md). Open question:
   for downstream ML, do we want to additionally drop **homo-oligomeric**
   entries that may have inter-chain shift perturbations? Today we
   keep them.

4. **mmseqs cluster representative override**. We compute a per-entry
   `quality_score` so that, after clustering, the highest-quality
   member of each cluster can be selected (overriding mmseqs' internal
   choice). 7-8% of clusters per tier would change if this override is
   applied. We ship **both** files: the unmodified mmseqs cluster
   representative FASTA (`train_<tier>.fasta`) and the
   quality-best FASTA (`train_<tier>_best.fasta`).
   **Open question for the user:** which should be the "official"
   training set? My recommendation is `_best.fasta` — it preserves
   the paper's cluster structure but trains on better-quality data.

5. **Re-emit `.str` files without methyl wildcards?** The 2026-05-05
   release embedded CDx / CGx labels into the released NMR-STAR files.
   The score JSONs are unaffected. If we want the released `.str`
   files to reflect the post-rollback parser behaviour, a re-run of
   `trizod --emit-str ...` on the same 4 tiers is needed.
   **Not done in this folder** — flagged as a follow-up commit.

6. **mmseqs cluster sizes are smaller than the paper.** Paper reports
   ~7,324 unfiltered cluster reps; we get 5,927 (2026-06 build, with
   stage-1 cluster-membership leakage removal added and the TriZOD test
   set recreated from the current snapshot). Likely causes: (a)
   bound-complex removal (~27%) is more aggressive here than in the
   paper's pipeline, (b) the May 2026 BMRB snapshot adds new
   chains that further consolidate clusters. **Not blocking.**

## File map (recent commits, top-of-tree)

```
trizod/
├── docs/_planning/                  internal notes (gitignored content)
├── docs/260415/                     LACS rollout / validation
├── docs/260422/                     LACS-vs-POTENCI per-tier scatter +
│                                     CSP analysis baseline
├── docs/260505/                     2026-05-05 release docs (Step 1–9)
└── docs/260520/                     ← THIS FOLDER
    └── (see README.md)
```
