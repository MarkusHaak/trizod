"""Dataset-build chain for TriZOD.

Promotes the dataset-construction scripts (formerly under docs/260520/scripts/
and docs/260623/scripts/) into importable, work-dir-parameterized modules:

    composition   -- bound-complex detection + quality/dedup helpers
    build         -- build_final_dataset (bound removal, dedup, ranking)
    testset       -- build_test_set (seeded, leakage-free)
    redundancy    -- run_mmseqs_pipeline (two-stage leakage removal + clustering)
    mmseqs        -- shared mmseqs option block + run() wrapper
    representatives-- cluster_best_repr (quality-best representative override)
    package_release-- Zenodo bundle assembly
    deploy_fasta  -- UdonPred-handoff FASTA builder
    paths         -- resolve_paths(): the repo-root / work-dir path contract
"""
