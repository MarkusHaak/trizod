# TriZOD dataset (release 2026-06)

Per-residue **protein-disorder** labels derived from BMRB NMR backbone chemical
shifts, in four **nested** stringency tiers (`strict ⊂ moderate ⊂ tolerant ⊂
unfiltered`). Observed shifts are re-referenced (LACS + POTENCI/AIC) and scored
against POTENCI random-coil predictions; the sequences are redundancy-reduced
and held-out test sets are removed.

## Contents

| tier | training proteins | scored records |
|---|--:|--:|
| unfiltered | 5,927 | 16,851 |
| tolerant | 5,684 | 15,433 |
| moderate | 4,063 | 10,107 |
| strict | 1,254 | 3,033 |

Held-out **test sets** (disjoint from every training set): CheZOD117 (115) and
TriZOD (344).

## Layout

```
trizod-dataset-2026-06/
├── train/<tier>/
│   ├── train_<tier>_best.fasta   one sequence per cluster (canonical)
│   ├── train_<tier>.fasta        mmseqs cluster representatives
│   └── clusters_best.tsv, clusters.tsv   cluster membership (repr, member)
├── scores/<tier>/scores.json     per-residue labels (one JSON object per line)
├── test/                         CheZOD117 + TriZOD test FASTAs
└── MANIFEST.json                 file list, sizes, SHA-256, counts
```

## `scores.json` fields (one record per line; `ID` matches the FASTA header)

| field | meaning |
|---|---|
| `ID` | `entryID_stID_entityAssemID_entityID` |
| `seq` | amino-acid sequence (1-letter) |
| `gscores` | per-residue **G-score** (0–1), aligned 1:1 to `seq` (`null` = no data) |
| `zscores` | per-residue CheZOD Z-score (same alignment) |
| `k` | number of weighted secondary shifts behind each residue (`0` = no shift) |
| `off_<atom>`, `lacs_off_<atom>` | per-atom referencing offsets (C, CA, CB, H, HA, HB, N) |
| `pH`, `temperature`, `ionic_strength`, `citation` | sample metadata |

## Use for ML

- **Inputs**: `train/<tier>/train_<tier>_best.fasta`.
- **Target**: the **G-score** (0–1, independent of shift count) from the matching
  `scores/<tier>/scores.json`, joined by `ID`. Mask residues where `k == 0`
  (G-score `null`).
- **Tier**: `moderate` balances size and quality; `strict` is cleanest;
  `unfiltered` is largest. Tiers are nested.
- **Cluster-aware CV**: group by `clusters_best.tsv`.
- Train on any tier and evaluate on **CheZOD117** and/or the **TriZOD** test set
  with no train/test leakage.

Derived from BMRB (https://bmrb.io/). License: MIT.
