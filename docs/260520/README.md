# 2026-05-20 — Investigation, dataset finalisation & mmseqs run

This folder collects the work done after the 2026-05-05 release to:

1. Take stock of where TriZOD stands and what is still open
   ([`status.md`](status.md)).
2. Quantify the **practical effect of LACS pre-correction on G-scores**
   ([`lacs-effect-on-gscores.md`](lacs-effect-on-gscores.md)).
3. **Revert** the Step-8 methyl-wildcard rewrite that has now been
   deemed unwanted
   ([`methyl-wildcards-removal.md`](methyl-wildcards-removal.md)).
4. Decide whether the `--max-offset` filter is still worth keeping per
   tier now that LACS pre-corrects most referencing errors
   ([`max-offset-analysis.md`](max-offset-analysis.md)).
5. Build the **final, machine-learning–ready dataset** on top of the
   2026-05-05 release: drop multi-molecule (bound) BMRB entries,
   pick the highest-quality representative for every duplicate sequence
   and assign a ranking that survives downstream mmseqs clustering
   ([`final-dataset.md`](final-dataset.md)).
6. Run the **mmseqs2 redundancy-reduction pipeline** with the exact
   parameters used in the original TriZOD report (Senoner & Heinzinger
   2024) and produce per-tier training sets that share consistent
   cluster representatives across tiers
   ([`mmseqs-pipeline.md`](mmseqs-pipeline.md)).

## Folder layout

```
docs/260520/
├── README.md                      this file
├── status.md                      project status & open questions
├── methyl-wildcards-removal.md    Step-8 reversal
├── lacs-effect-on-gscores.md      LACS impact on G-scores (figures)
├── max-offset-analysis.md         max-offset filter empirical study
├── final-dataset.md               bound-complex filter & sequence dedup
├── mmseqs-pipeline.md             clustering recipe + output stats
├── scripts/                       all analysis & build scripts
│   ├── plot_lacs_effect.py
│   ├── analyze_max_offset.py
│   ├── build_final_dataset.py
│   ├── run_mmseqs_pipeline.py
│   └── cluster_best_repr.py
├── figures/                       PNGs referenced by the markdown
│   ├── lacs_effect_gscores.png
│   ├── lacs_effect_affected_entries.png
│   ├── max_offset_distribution.png
│   └── max_offset_filter_curve.png
└── data/                          numeric outputs & generated artefacts
    ├── lacs_effect_summary.csv
    ├── max_offset_summary.csv
    ├── final_dataset/             per-tier dedup FASTAs + ranked TSVs
    │   ├── _composition_cache.csv
    │   ├── final_dataset_summary.json
    │   └── <tier>/
    │       ├── <tier>.fasta
    │       ├── <tier>_all_ranked.tsv
    │       └── <tier>_summary.json
    └── mmseqs/                    test-set-cleaned FASTAs + cluster
        ├── train_<tier>.fasta     mmseqs cluster representatives
        ├── train_<tier>_clu.tsv   2-col cluster file
        ├── train_<tier>_best.fasta             quality-best repr override
        ├── train_<tier>_clu_best.tsv           enriched cluster table
        └── cluster_repr_overrides_<tier>.tsv   clusters where best ≠ mmseqs pick
```

## Reproducing the work

```bash
# 1. (one-off, ~5 min)   build the deduplicated per-tier FASTAs
uv run python -m trizod.dataset.build

# 2. (~30 s) run the mmseqs redundancy-reduction recipe from the paper
uv run python -m trizod.dataset.redundancy

# 3. (~1 s)  attach quality-score-based "best member" annotation
uv run python -m trizod.dataset.representatives

# Plots
uv run python docs/260520/scripts/plot_lacs_effect.py
uv run python docs/260520/scripts/analyze_max_offset.py
```

Inputs each script consumes are documented at the top of the script.
