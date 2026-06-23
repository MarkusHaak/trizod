#!/usr/bin/env bash
# Reproduce the TriZOD dataset release end to end.
#
# Order:
#   0. score   — per-tier per-residue Z/G scores (heavy; ~17k BMRB entries).
#                Produces data/release/<tier>/scores.json. SKIPPED if those
#                already exist (pass --rescore to force).
#   1. build   — bound/multi-molecule removal + length<20 drop + exact-seq dedup
#                (docs/260520/data/final_dataset/<tier>/<tier>.fasta)
#   2. mmseqs  — two-stage test-set leakage removal (stage-1 cluster-membership
#                + stage-2 easy-search) then cluster @50/80 + clusterupdate
#   3. best    — quality-best cluster-representative override
#   4. package — stage the release bundle + MANIFEST + leakage gate
#
# Usage:
#   docs/260520/scripts/run_all.sh [VERSION] [--rescore]
#   VERSION defaults to 2026-06.
#
# Requirements: uv, mmseqs in PATH. Run from the repository root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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

SCRIPTS="docs/260520/scripts"
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

# ---- Steps 1-4: deterministic, re-runnable ----
echo "== Step 1: build_final_dataset (bound removal + dedup) =="
uv run python "$SCRIPTS/build_final_dataset.py"

echo "== Step 2: build_test_set (recreate TriZOD test set, seeded) =="
uv run python "$SCRIPTS/build_test_set.py"

echo "== Step 3: run_mmseqs_pipeline (two-stage leakage removal + clustering) =="
uv run python "$SCRIPTS/run_mmseqs_pipeline.py"

echo "== Step 4: cluster_best_repr (quality-best override) =="
uv run python "$SCRIPTS/cluster_best_repr.py"

echo "== Step 5: package_release --version $VERSION (with leakage gate) =="
uv run python "$SCRIPTS/package_release.py" --version "$VERSION"

echo "== Done. Bundle: docs/260520/data/release_bundle/trizod-dataset-$VERSION =="
