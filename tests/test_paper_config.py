"""Build selection, input-keyed caches and the shared "perturbed" rule of the paper scripts
(``scripts/figures/paper/paper_config.py``, ``style_paper.is_perturbed``)."""

import json
import sys
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "scripts" / "figures" / "paper"
sys.path.insert(0, str(PAPER))

import paper_config as pc  # noqa: E402

BUILD_VARS = pc.BUILD_VARS


def _clear_env(monkeypatch):
    for v in (*BUILD_VARS, "TRIZOD_DATA_ROOT", "TRIZOD_PKL_DIR", "TRIZOD_DRAW_DIR"):
        monkeypatch.delenv(v, raising=False)


def test_default_is_build_of_record(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    b = pc.resolve(tmp_path)
    assert b.release == pc.RECORD_DIR / "release"
    assert b.bundle == b.build / "release_bundle" / pc.RECORD_BUNDLE
    assert b.log == pc.RECORD_DIR / pc.RECORD_LOG
    assert b.pkl == tmp_path / "tmp" / "bmrb_entries"
    assert b.pq_main.name == "trizod_dataset.parquet"


def test_env_overrides(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    for v in BUILD_VARS:
        monkeypatch.setenv(v, str(tmp_path / v))
    monkeypatch.setenv("TRIZOD_PKL_DIR", str(tmp_path / "pkl"))
    b = pc.resolve(tmp_path)
    assert b.build == tmp_path / "TRIZOD_BUILD_DIR"
    assert b.release == tmp_path / "TRIZOD_RELEASE_DIR"
    assert b.bundle == tmp_path / "TRIZOD_BUNDLE"
    assert b.log == tmp_path / "TRIZOD_REBUILD_LOG"
    assert b.pkl == tmp_path / "pkl"


def test_partial_build_override_fails(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TRIZOD_BUILD_DIR", str(tmp_path))
    with pytest.raises(SystemExit, match="TRIZOD_RELEASE_DIR"):
        pc.resolve(tmp_path)


def test_keyed_json_recomputes_on_new_inputs(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"n": len(calls)}

    p = tmp_path / "c.json"
    assert pc.keyed_json(p, {"snapshot": "a"}, compute) == {"n": 1}
    assert pc.keyed_json(p, {"snapshot": "a"}, compute) == {"n": 1}  # hit
    assert pc.keyed_json(p, {"snapshot": "b"}, compute) == {"n": 2}  # new input
    assert pc.keyed_json(p, {"snapshot": "b"}, compute, refresh=True) == {"n": 3}
    assert json.loads(p.read_text())["_inputs"] == {"snapshot": "b"}


def test_unkeyed_legacy_cache_is_ignored(tmp_path):
    p = tmp_path / "raw_scan.json"
    p.write_text(json.dumps({"_version": 3, "entries": {"1": {}}}))
    assert pc.read_keyed(p, {"snapshot": "a"}) is None


def test_snapshot_digest_follows_snapshot_record(tmp_path):
    (tmp_path / "bmr1").mkdir()
    d0 = pc.snapshot_digest(tmp_path)
    raw2 = tmp_path / "other"
    (raw2 / "bmr1").mkdir(parents=True)
    (raw2 / "BMRB_SNAPSHOT.json").write_text('{"retrieved": "2026-11-11"}')
    assert pc.snapshot_digest(raw2) != d0


def test_code_digest_tracks_source():
    def f():
        return 1

    def g():
        return 2

    assert pc.code_digest(f) == pc.code_digest(f)
    assert pc.code_digest(f) != pc.code_digest(g)


def test_is_perturbed_rule():
    pytest.importorskip("matplotlib")
    pd = pytest.importorskip("pandas")
    import style_paper as sp

    df = pd.DataFrame(
        {
            "physical_state": ["native", "Denatured", None, "native", "native", "native", "native"],
            "sample_state_evidence": ["solution", "solution", "solution", "solid", "solution", "solution", None],
            "membrane_mimetic": [None, None, None, None, "DPC", None, None],
            "cosolvent_evidence": [False, False, None, False, False, True, False],
            "pressure_bar": [1.0, None, None, None, None, None, 400.0],
        }
    )  # fmt: skip
    assert sp.is_perturbed(df).tolist() == [False, True, False, True, True, True, True]
    # Fig. 3a passes the raw-file evidence for cosolvent and pressure
    raw = sp.is_perturbed(df, cosolvent=[False] * 7, pressure_bar=[1.0] * 7).tolist()
    assert raw == [False, True, False, True, True, False, False]
    assert sp.HIGH_PRESSURE_BAR == 400.0
