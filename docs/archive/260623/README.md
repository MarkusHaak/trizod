# 260623 — Deployment FASTA (per-residue G-score labels)

Builds a single self-contained FASTA carrying per-residue TriZOD **G-score**
disorder labels for the **tolerant** training set and the **TriZOD test set**,
in the exact format of the SETH/ODiNPred-style `disorder_trizod.fasta`
(Rostlab/pbc, `trizod` branch), for handing off to a collaborator deploying a
disorder predictor.

## Script

`trizod.dataset.deploy_fasta` — self-validating builder (promoted from
`docs/260623/scripts/build_deploy_fasta.py` into the package in restructure
Phase 5).

```bash
# train + test (tolerant tier, default)
uv run python -m trizod.dataset.deploy_fasta

# train + val + test (15% of train randomly held out as val, seeded)
uv run python -m trizod.dataset.deploy_fasta \
    --val-fraction 0.15 \
    --out docs/260623/data/deploy/disorder_trizod_with_val.fasta
```

Inputs (read from the 260520 dataset build):
`docs/260520/data/mmseqs/train_tolerant_best.fasta` (train IDs),
`docs/260520/data/testset/TriZOD_test_set.fasta` (test IDs),
`data/release/tolerant/scores.json` (per-residue G-score labels).

## Output (gitignored — `docs/260623/data/`, regenerable)

| File | Split | Records |
|------|-------|--------:|
| `data/deploy/disorder_trizod.fasta` | train, test | 5,684 + 344 |
| `data/deploy/disorder_trizod_with_val.fasta` | train, val, test | 4,831 + 853 + 344 |

See `data/deploy/README.md` for the per-record format (`>{ID} SET=... TARGET=...
MASK=...` then sequence; `TARGET` = G-score, `999.0` sentinel, `MASK` 1/0).

## Notes

- `TARGET` is the **G-score** (0–1, k-independent), not the raw CheZOD Z-score.
- Train/test are leakage-free (no shared ID or exact sequence); train is
  redundancy-reduced (mmseqs2 30/80) against the TriZOD test set + CheZOD.
- `val` is a uniform random subset of train (seed 42), disjoint from test, not
  redundancy-reduced against the remaining train (standard for a val hold-out).
- Output FASTAs are gitignored like the other dated `data/` dirs; regenerate
  from the script. The intended distribution channel is the Zenodo deposit.
