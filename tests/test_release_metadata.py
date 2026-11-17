"""The release Parquet states its own identity (release audit 2026-11-17, D14,
F32, G38 and B4): version, licences, code commit, BMRB snapshot, test-set name,
the G-score cuts with their rule and the k >= 10 note, and the threshold
framework's calibration summary. Every key is checked on the pure-Python
metadata builder, and once through a real Parquet."""

import json
import os
from pathlib import Path

import pytest

from trizod.dataset import release_info as ri

from ._release_fixtures import (
    BUILD_COMMIT,
    FRAMEWORK_COMMIT,
    build_release_parquet,
    load_builder,
    write_bundle,
    write_results,
    write_snapshot,
)

builder = load_builder()
#: The threshold-framework run of 2026-11-17, under ``$TRIZOD_DATA_ROOT`` (a
#: checkout holding data/; a worktree has none) or this repository.
REAL_FRAMEWORK = (
    Path(os.environ.get("TRIZOD_DATA_ROOT") or Path(__file__).resolve().parents[1])
    / "data"
    / "interim"
    / "threshold-framework-2026-11-17"
)


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    root = tmp_path_factory.mktemp("release")
    bundle = write_bundle(root / "bundle")
    snapshot = write_snapshot(root / "bmrb")
    results = write_results(root / "framework")
    cols = builder.collect_columns(bundle)
    meta = builder.table_metadata(
        cols,
        snapshot=ri.snapshot_block(snapshot),
        calibration=ri.read_calibration(results, FRAMEWORK_COMMIT),
        commit=BUILD_COMMIT,
    )
    return {
        "root": root,
        "bundle": bundle,
        "snapshot": snapshot,
        "results": results,
        "meta": {k.decode(): v.decode() for k, v in meta.items()},
    }


def _json(meta, key):
    return json.loads(meta[key])


def test_version_and_licences_are_stated(release):
    meta = release["meta"]
    assert meta["dataset_version"] == ri.DATASET_VERSION == "0.4.0"
    assert meta["data_license"] == "CC-BY-4.0"
    assert meta["code_license"] == "MIT"
    assert "CC BY 4.0" in meta["license"] and "MIT" in meta["license"]
    assert "BMRB" in meta["license"] and "No BMRB licence is claimed" in meta["license"]


def test_code_commit_is_the_full_hash_with_its_dirty_flag(release):
    commit = _json(release["meta"], "code_commit")
    assert commit["commit"] == "c" * 40
    assert commit["dirty"] is False
    assert commit["pipeline_version"].startswith("trizod-")
    assert commit["repository"] == ri.REPOSITORY


def test_snapshot_name_count_and_entry_list_hash(release):
    snap = _json(release["meta"], "bmrb_snapshot")
    assert snap["name"] == release["snapshot"].name
    assert snap["n_entries"] == 3
    assert snap["entry_list_sha256"] == "a" * 64
    assert snap["retrieved"] == "2026-11-11T12:46:31Z"


def test_test_set_is_named_with_its_size_and_retired_count(release):
    ts = _json(release["meta"], "test_set")
    assert ts["name"] == "TriZOD-test v1.0"
    assert ts["n_chains"] == 2
    assert ts["n_retired"] == 1
    assert ts["retired_by_reason"] == {"not_strict": 1}
    assert ts["label"] == "TriZOD-test v1.0 (n = 2; 1 retired chains)"
    assert ts["n_chezod117"] == 1


def test_cut_block_states_the_three_cuts_and_the_rule(release):
    cuts = _json(release["meta"], "cuts")
    assert cuts["values"] == [0.15, 0.35, 0.70]
    assert cuts["ordered"] == "G < 0.15"
    assert cuts["disordered"] == "G >= 0.35"
    assert cuts["confidently_disordered"] == "G >= 0.70"
    assert "disordered iff G >= t" in cuts["rule"]
    assert "ordered iff G < 0.15" in cuts["rule"]
    assert "k >= 10" in cuts["k_note"]
    assert "never appear as label columns" in cuts["label"]
    assert release["meta"]["cut_rule"] == ri.CUT_RULE
    assert "k >= 10" in release["meta"]["k_reliability_note"]


def test_calibration_summary_is_recorded(release):
    cal = _json(release["meta"], "calibration")
    assert cal["prevalence"] == {"value": 0.186, "ci95": [0.141, 0.244]}
    lr = cal["likelihood_ratio_at_cuts"]
    assert lr["0.15"] == {"value": 0.56, "ci95": [0.32, 0.83]}
    assert lr["0.35"]["value"] == 5.2
    assert lr["0.70"]["ci95"] == [23.2, 157.6]
    assert cal["framework_commit"] == FRAMEWORK_COMMIT
    assert cal["results_json_sha256"] == ri.sha256_file(
        release["results"] / "results.json"
    )
    assert cal["rule"] == "disordered iff G >= t"
    assert cal["input_parquet_sha256"] == "e" * 64
    assert set(cal["prevalence_by_split"]) == {"train", "test_trizod", "test_chezod117"}
    assert cal["prevalence_by_split"]["test_trizod"]["pi"]["value"] == 0.15
    assert cal["replicate_agreement_by_k"]["k >= 10"]["kappa_binary"]["value"] == 0.94


def test_column_dictionary_is_pointed_to(release):
    assert "columns.md" in release["meta"]["column_dictionary"]


def test_mask_semantics_is_the_finite_g_rule_not_k(release):
    mask = release["meta"]["mask_semantics"]
    assert "finite" in mask
    assert "k == 0" not in mask
    assert "999" in mask


def test_null_semantics_covers_test_retired_label_tier(release):
    null = release["meta"]["null_semantics"]
    assert "neither test_trizod nor test_retired" in null


def test_missing_snapshot_and_calibration_are_recorded_as_such(release):
    cols = builder.collect_columns(release["bundle"])
    meta = builder.table_metadata(cols, commit=BUILD_COMMIT)
    assert meta[b"calibration"].startswith(b"not recorded")
    assert meta[b"bmrb_snapshot"].startswith(b"not recorded")


def test_read_calibration_rejects_other_cuts(tmp_path):
    results = write_results(tmp_path / "fw", cuts=(0.15, 0.40, 0.70))
    with pytest.raises(ValueError, match="calibrated cuts"):
        ri.read_calibration(results, FRAMEWORK_COMMIT)


def test_framework_commit_defaults_to_the_framework_directory(tmp_path):
    results = write_results(tmp_path / "fw")
    cal = ri.read_calibration(results)
    commit = cal["framework_commit"]
    assert commit is None or len(commit["commit"]) == 40


@pytest.mark.skipif(
    not (REAL_FRAMEWORK / "results.json").is_file(),
    reason="threshold-framework output of 2026-11-17 not on this machine",
)
def test_read_calibration_on_the_real_framework_run():
    cal = ri.read_calibration(REAL_FRAMEWORK, FRAMEWORK_COMMIT)
    assert cal["prevalence"] == {"value": 0.186, "ci95": [0.1414, 0.244]}
    lr = cal["likelihood_ratio_at_cuts"]
    assert lr["0.15"]["value"] == pytest.approx(0.5645)
    assert lr["0.35"]["value"] == pytest.approx(5.1981)
    assert lr["0.70"]["value"] == pytest.approx(36.6919)
    assert cal["prevalence_by_split"]["test_trizod"]["pi"]["value"] == 0.1534
    reps = cal["replicate_agreement_by_k"]
    assert reps["k >= 10"]["kappa_binary"]["value"] == pytest.approx(0.9419, abs=1e-4)
    assert reps["k < 10"]["kappa_binary"]["value"] == pytest.approx(0.8028, abs=1e-4)


def test_metadata_round_trips_through_the_parquet(release):
    pq = pytest.importorskip("pyarrow.parquet")
    out = build_release_parquet(
        release["bundle"],
        release["root"] / "trizod_dataset.parquet",
        snapshot_dir=release["snapshot"],
        results_dir=release["results"],
    )
    meta = {k.decode(): v.decode() for k, v in pq.read_schema(out).metadata.items()}
    for key in (
        "dataset_version",
        "license",
        "data_license",
        "code_license",
        "code_commit",
        "bmrb_snapshot",
        "test_set",
        "cuts",
        "cut_rule",
        "k_reliability_note",
        "calibration",
        "column_dictionary",
    ):
        assert key in meta, key
    assert json.loads(meta["test_set"])["n_chains"] == 2
    assert json.loads(meta["calibration"])["prevalence"]["value"] == 0.186
