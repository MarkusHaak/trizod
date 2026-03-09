"""Regression test: run pipeline on 100-entry subset and compare against reference."""

import json
import os
import subprocess
import sys
import tempfile

import pytest

from tests.conftest import requires_bmrb_data

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_FILE = os.path.join(TESTS_DIR, "reference", "unfiltered.json")
SUBSET_DIR = os.path.join(TESTS_DIR, "bmrb_subset")


def load_jsonl(path):
    entries = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            entries[entry["ID"]] = entry
    return entries


@requires_bmrb_data
@pytest.mark.skipif(
    not os.path.exists(REFERENCE_FILE),
    reason="Reference file not generated yet",
)
class TestPipelineRegression:
    def test_unfiltered_matches_reference(self, tmp_path):
        output_prefix = str(tmp_path / "unfiltered")
        result = subprocess.run(
            [
                sys.executable, "-m", "trizod.trizod",
                "--input-dir", SUBSET_DIR,
                "--filter-defaults", "unfiltered",
                "--output-prefix", output_prefix,
                "--output-format", "json",
                "--no-progress",
                "--cache-dir", str(tmp_path / "cache"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Pipeline failed:\n{result.stderr}"

        reference = load_jsonl(REFERENCE_FILE)
        actual = load_jsonl(output_prefix + ".json")

        assert len(actual) == len(reference), (
            f"Entry count mismatch: {len(actual)} vs {len(reference)}"
        )

        for id_ in reference:
            assert id_ in actual, f"Missing entry: {id_}"
            ref = reference[id_]
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
                assert a == pytest.approx(r, abs=1e-6), (
                    f"{id_} pos {i}: zscore {a} != {r}"
                )
