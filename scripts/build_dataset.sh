#!/usr/bin/env bash
# Reproduce the TriZOD dataset release end to end.
#
# Order:
#   0. score   — per-tier per-residue Z/G scores (heavy; ~17k BMRB entries).
#                Produces data/release/<tier>/scores.json. SKIPPED if those
#                already exist (pass --rescore to force).
#   1. build   — bound/multi-molecule removal + length<20 drop + exact-seq dedup
#                (trizod.dataset.build -> <work-dir>/final_dataset/<tier>/<tier>.fasta)
#   2. testset — recreate the seeded TriZOD test set (trizod.dataset.testset)
#   3. mmseqs  — two-stage test-set leakage removal (stage-1 cluster-membership
#                + stage-2 easy-search) then cluster @50/80 + clusterupdate
#                (trizod.dataset.redundancy)
#   4. best    — quality-best cluster-representative override
#                (trizod.dataset.representatives)
#   5. package — stage the release bundle + MANIFEST + leakage gate
#                (trizod.dataset.package_release)
#
# Usage:
#   scripts/build_dataset.sh [VERSION] [--rescore]
#   VERSION defaults to 2026-06.
#
# Requirements: uv, mmseqs in PATH. Run from anywhere (paths auto-resolve to the
# repo root; artefacts land under docs/260520/data by default).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="2026-06"
RESCORE=0
for arg in "$@"; do
  case "$arg" in
    --rescore) RESCORE=1 ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) VERSION="$arg" ;;
  esac
done

TIERS=(unfiltered tolerant moderate strict)

command -v uv     >/dev/null || { echo "uv not found in PATH" >&2; exit 1; }
command -v mmseqs >/dev/null || { echo "mmseqs not found in PATH" >&2; exit 1; }
echo "mmseqs: $(mmseqs version 2>/dev/null | head -1)"

# ---- Step 0: scoring (guarded) ----
need_score=0
for t in "${TIERS[@]}"; do
  [ -s "data/release/$t/scores.json" ] || need_score=1
done
if [ "$RESCORE" = 1 ] || [ "$need_score" = 1 ]; then
  echo "== Step 0: scoring per tier (--rereference-mode both) =="
  for t in "${TIERS[@]}"; do
    mkdir -p "data/release/$t"
    uv run trizod \
      --input-dir data/bmrb_entries \
      --filter-defaults "$t" \
      --rereference-mode both \
      --output-format json \
      --output-prefix "data/release/$t/scores" \
      --cache-dir tmp
  done
else
  echo "== Step 0: scoring SKIPPED (data/release/<tier>/scores.json present; --rescore to force) =="
fi

# ---- Steps 1-5: deterministic, re-runnable ----
echo "== Step 1: build (bound removal + dedup) =="
uv run python -m trizod.dataset.build

echo "== Step 2: testset (recreate TriZOD test set, seeded) =="
uv run python -m trizod.dataset.testset

echo "== Step 3: redundancy (two-stage leakage removal + clustering) =="
uv run python -m trizod.dataset.redundancy

echo "== Step 4: representatives (quality-best override) =="
uv run python -m trizod.dataset.representatives

echo "== Step 5: package_release --version $VERSION (with leakage gate) =="
uv run python -m trizod.dataset.package_release --version "$VERSION"

echo "== Done. Bundle: docs/260520/data/release_bundle/trizod-dataset-$VERSION =="
