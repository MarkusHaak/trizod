# Full pipeline rerun on the finalized code

## What we ran

For each filter tier, the finalized TriZOD pipeline against `data/bmrb_entries/` (17,388 BMRB entries):

```bash
uv run trizod \
  --input-dir data/bmrb_entries/ \
  --output-prefix data/release/<tier>/scores \
  --output-format json \
  --filter-defaults <tier> \
  --emit-str data/release/<tier>/str/ \
  --cache-dir tmp \
  --rereference-mode both \
  --processes 8
```

Each tier produces:
- `data/release/<tier>/scores.json` — per-entry NDJSON with Z/G-scores, POTENCI residual offsets (`off_<atom>`), LACS offsets (`lacs_off_<atom>`), and metadata.
- `data/release/<tier>/str/bmr<id>_<st>_<ea>_<e>_rereferenced.str` — re-referenced backbone-shifts NMR-STAR file per scored entity.

## Cache behaviour

| Cache | Status before rerun | Effect |
|---|---|---|
| `tmp/bmrb_entries/` | mass-invalidated (Step 8 wildcard rewriting changes the parser output) | full re-parse, ~30 min |
| `tmp/potenci/` | preserved (filter-independent, depends only on seq + conditions) | warm hits across all 4 tiers |
| `tmp/wSCS/` | implicitly invalidated by the new mode-keyed filename (Task 1 fixup) | recomputed for `--rereference-mode both` |

The pre-Step-8 BMRB pickle cache is preserved at `tmp/bmrb_entries_pre_step8/` for rollback.

## Results — entry counts per tier

After Tasks 1-7's pipeline changes (Steps 4-9 + LACS-in-pipeline + Step 8 wildcards):

| Tier | Entries scored | Entries final (post-filter) | `.str` files emitted |
|---|---:|---:|---:|
| `unfiltered` | 17,843 | (rerun in progress) | 5,576 (partial, growing) |
| `tolerant` | 17,843 | 15,433 (86.49%) | 15,433 |
| `moderate` | 17,843 | 10,107 (56.64%) | 10,107 |
| `strict` | 17,843 | 3,033 (17.00%) | 3,033 |

(`Entries scored` is larger than the 17,388 BMRB entry count because some entries contain multiple peptide shift tables / entities.)

## Manifest

`data/release/manifest.json` records the pipeline version (`trizod-2026-05-05`), the re-referencing mode (`both`), and per-tier scores/`.str` counts.

## What's NOT in the rerun

- Tier-level CSV outputs are not generated. We requested `--output-format json` (the JSON output carries the per-entry LACS+POTENCI offsets the talk needs).
- The unfiltered tier was still finishing the `.str` emission step at this writeup; the manifest will be regenerated once that completes.
- We did NOT push the `.str` archive to Zenodo; the metadata files are in the repo, the actual deposit happens on the first tagged release.

## Talking points for the slide

- **Volume:** 15,433 entries scored at the tolerant tier with both LACS and POTENCI/AIC offset corrections applied; 15,433 re-referenced `.str` files emitted.
- **Cache thrift:** POTENCI cache survived; only BMRB parsing + scoring were redone. Rerun wall-clock per tier ≈ 25-90 min (warm POTENCI cache, cold wSCS).
- **Backwards compatibility:** the legacy regression baseline (`tests/reference/unfiltered.json`) still validates against `--rereference-mode potenci-only` — no silent breakage of existing consumers.

## Commits

- `(empty record-only commit pending)` — `data: regenerate baselines and release artifacts with finalized pipeline`. The actual data lives under `data/release/` (gitignored); the commit message records the run command + pipeline version.
