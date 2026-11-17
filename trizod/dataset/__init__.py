"""Dataset-build chain for TriZOD.

Promotes the dataset-construction scripts (formerly under docs/260520/scripts/
and docs/260623/scripts/) into importable, work-dir-parameterized modules:

    composition   -- bound-complex detection + quality/dedup helpers
    build         -- build_final_dataset (bound removal, dedup, ranking)
    testset       -- resolve the pinned TriZOD test set (TriZOD-test v1.0)
    redundancy    -- run_mmseqs_pipeline (two-stage leakage removal + clustering)
    mmseqs        -- shared mmseqs option block + run() wrapper
    representatives-- cluster_best_repr (quality-best representative override)
    package_release-- package bundle + leakage gate (input of the Parquet)
    finalize      -- assemble the deposit: views, docs, MANIFEST over every file
    release_info  -- version, licences, test-set name, G cuts, provenance blocks
    class_balance -- per-split label distribution (zones, k < 10, pi)
    deploy_fasta  -- UdonPred-handoff FASTA builder
    paths         -- resolve_paths(): the repo-root / work-dir path contract
"""
