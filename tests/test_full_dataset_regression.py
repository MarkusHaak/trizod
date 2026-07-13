"""Full-dataset regression: compare pipeline output against baselines in data/interim/baseline/.

Run with:
    uv run pytest tests/test_full_dataset_regression.py -v

Requires:
    - BMRB data in data/raw/bmrb_entries/
    - Baseline files in data/interim/baseline/ (unfiltered.json, tolerant.json, etc.)
    - Precomputed POTENCI cache in tmp/potenci/ (optional but recommended for speed)
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import requires_bmrb_data
from trizod import paths

BASELINE_DIR = paths.INTERIM_BASELINE
BMRB_DIR = paths.RAW_BMRB

FILTER_LEVELS = ["unfiltered", "tolerant", "moderate", "strict"]


def load_jsonl(path):
    entries = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            entries[entry["ID"]] = entry
    return entries


@pytest.mark.slow
@requires_bmrb_data
@pytest.mark.skipif(
    not BASELINE_DIR.is_dir(),
    reason="Baseline files not available (data/interim/baseline/)",
)
class TestFullDatasetRegression:
    @pytest.mark.parametrize("filter_level", FILTER_LEVELS)
    def test_matches_baseline(self, tmp_path, filter_level):
        baseline_file = BASELINE_DIR / f"{filter_level}.json"
        if not baseline_file.exists():
            pytest.skip(f"Baseline {baseline_file} not found")

        output_prefix = str(tmp_path / filter_level)

        # Use existing cache if available (speeds up from hours to minutes)
        cache_dir = str(Path(__file__).resolve().parent.parent / "tmp")

        cmd = [
            sys.executable,
            "-m",
            "trizod.trizod",
            "--input-dir",
            str(BMRB_DIR),
            "--filter-defaults",
            filter_level,
            "--output-prefix",
            output_prefix,
            "--output-format",
            "json",
            "--no-progress",
            "--cache-dir",
            cache_dir,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours max for full dataset
        )
        assert result.returncode == 0, f"Pipeline failed:\n{result.stderr}"

        baseline = load_jsonl(baseline_file)
        actual = load_jsonl(output_prefix + ".json")

        assert len(actual) == len(baseline), (
            f"Entry count mismatch for {filter_level}: "
            f"{len(actual)} actual vs {len(baseline)} baseline"
        )

        zscore_mismatches = []
        for id_ in baseline:
            assert id_ in actual, f"Missing entry: {id_}"

            ref = baseline[id_]
            act = actual[id_]

            assert act["seq"] == ref["seq"], f"{id_}: sequence mismatch"
            assert act["temperature"] == pytest.approx(ref["temperature"]), (
                f"{id_}: temperature mismatch"
            )

            ref_z = ref.get("zscores", [])
            act_z = act.get("zscores", [])
            assert len(act_z) == len(ref_z), f"{id_}: zscore length mismatch"

            for i, (a, r) in enumerate(zip(act_z, ref_z)):
                if a is None and r is None:
                    continue
                if a is None or r is None or abs(a - r) > 1e-6:
                    zscore_mismatches.append(f"{id_} pos {i}: {a} != {r}")
                    break

        assert not zscore_mismatches, (
            f"{len(zscore_mismatches)} entries with Z-score mismatches "
            f"(showing first 10):\n" + "\n".join(zscore_mismatches[:10])
        )
