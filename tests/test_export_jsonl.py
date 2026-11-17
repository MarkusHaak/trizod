"""UdonPred v2 JSONL view of the release Parquet (checks F29, G37).

The view is ``<out>/trizod_<tier>/{train,valid,test}.jsonl`` plus
``<out>/chezod117/test.jsonl``, one ``{"id", "x_0", "y"}`` record per chain,
``y`` the per-residue G with 999 where the Parquet masks the residue.

Most tests run on in-memory rows shaped like the Parquet's (pyarrow is not a
project dependency); the round trip through a real Parquet file needs
``uv run --with pyarrow pytest tests/test_export_jsonl.py``. A built release
can be checked end to end with::

    TRIZOD_EXPORT_PARQUET=<trizod_dataset.parquet> \\
        uv run --with pyarrow pytest tests/test_export_jsonl.py -rs
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from trizod.dataset import export_jsonl as ex

NAN = float("nan")


def _row(rid, split, *, train_tier=None, cluster_repr=None, seq=None, g=None, m=None):
    seq = seq or "MKTAYIAK"
    if g is None:
        g = [NAN, 0.1234, 0.5, 0.9, 1.0, 0.0, 0.3333, NAN][: len(seq)]
    if m is None:
        m = [not (isinstance(x, float) and math.isnan(x)) for x in g]
    return {
        "id": rid,
        "sequence": seq,
        "gscores": g,
        "mask": m,
        "split": split,
        "train_tier": train_tier,
        "cluster_repr": cluster_repr,
        "label_tier_source": train_tier or "strict",
    }


@pytest.fixture
def rows():
    """A miniature release: 40 training reps over four tiers, their redundant
    members, TriZOD test, retired test, CheZOD117 and excluded chains."""
    out = []
    tiers = ["unfiltered", "tolerant", "moderate", "strict"]
    for i in range(40):
        rid = f"{1000 + i}_1_1_1"
        out.append(_row(rid, "train", train_tier=tiers[i % 4], cluster_repr=rid))
        # every fourth representative has a redundant cluster member
        if i % 4 == 0:
            out.append(
                _row(f"{5000 + i}_1_1_1", "redundant", cluster_repr=rid, seq="MKTAYIAR")
            )
    out.append(_row("9001_1_1_1", "test_trizod", seq="GSHMLE"))
    out.append(_row("9002_1_1_1", "test_trizod", seq="GSHMLEKK"))
    out.append(_row("9003_1_1_1", "test_retired", seq="PPPPGG"))
    out.append(_row("9004_1_1_1", "test_retired", seq="PPPPGA"))
    out.append(_row("8001_1_1_1", "test_chezod117", seq="AAAAKK"))
    out.append(_row("7001_1_1_1", "excluded", seq="WWWWW"))
    return out


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _f32(x) -> np.float32:
    return np.float32(x)


# --------------------------------------------------------------------------- #
# record schema and labels
# --------------------------------------------------------------------------- #


def test_record_has_exactly_udonpred_fields():
    rec = ex.to_record(_row("1_1_1_1", "train", train_tier="strict"))
    assert tuple(rec) == ex.RECORD_FIELDS == ("id", "x_0", "y")
    assert rec["id"] == "1_1_1_1"
    assert rec["x_0"] == "MKTAYIAK"
    assert len(rec["y"]) == len(rec["x_0"])
    assert all(isinstance(v, float) for v in rec["y"])


def test_999_exactly_where_mask_false_or_nan():
    g = [NAN, 0.25, 0.5, 0.75, None, 0.1]
    m = [False, True, False, True, True, True]  # pos 2: mask wins over a value
    y = ex.label_values(g, m)
    assert y[0] == ex.MASK_VALUE == 999.0
    assert y[2] == 999.0  # mask False although a finite G is present
    assert y[4] == 999.0  # None (null) with mask True still masks
    assert [i for i, v in enumerate(y) if v == 999.0] == [0, 2, 4]
    assert y[1] == 0.25 and y[3] == 0.75 and y[5] == pytest.approx(0.1)


def test_labels_are_float32_identical_to_parquet():
    # float32 values as pyarrow hands them back (double-widened float32)
    raw = [float(np.float32(v)) for v in (0.1234, 0.3333, 0.7, 1e-05, 0.0, 1.0)]
    y = ex.label_values(raw, [True] * len(raw))
    assert [_f32(a) == _f32(b) for a, b in zip(y, raw)] == [True] * len(raw)
    # and printed short: the shortest repr that round-trips the float32
    assert json.dumps(y[0]) == "0.1234"


def test_label_length_mismatch_raises():
    with pytest.raises(ValueError):
        ex.label_values([0.1, 0.2], [True])


def test_out_of_range_label_raises():
    with pytest.raises(ValueError):
        ex.label_values([0.1, 1.5], [True, True])


# --------------------------------------------------------------------------- #
# split membership
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("tier", "allowed"),
    [
        ("tolerant", {"tolerant", "moderate", "strict"}),
        ("moderate", {"moderate", "strict"}),
        ("strict", {"strict"}),
    ],
)
def test_training_representatives_follow_train_tier(rows, tier, allowed):
    reps = ex.training_representatives(rows, tier)
    want = {
        r["id"] for r in rows if r["split"] == "train" and r["train_tier"] in allowed
    }
    assert {r["id"] for r in reps} == want
    assert all(r["split"] == "train" for r in reps)  # never redundant members


def test_unknown_tier_rejected(rows):
    with pytest.raises(ValueError):
        ex.training_representatives(rows, "lenient")


def test_views_layout(rows):
    views = ex.build_views(rows, tiers=("tolerant", "strict"), seed=1)
    assert set(views) == {"trizod_tolerant", "trizod_strict", "chezod117"}
    assert set(views["trizod_tolerant"]) == {"train", "valid", "test"}
    assert set(views["chezod117"]) == {"test"}
    assert [r["id"] for r in views["chezod117"]["test"]] == ["8001_1_1_1"]


def test_train_plus_valid_is_the_tier(rows):
    views = ex.build_views(rows, seed=3)
    for tier in ex.DEFAULT_TIERS:
        v = views[ex.tier_folder(tier)]
        got = {r["id"] for r in v["train"]} | {r["id"] for r in v["valid"]}
        assert got == {r["id"] for r in ex.training_representatives(rows, tier)}


def test_valid_and_train_disjoint_by_cluster(rows):
    # two reps share a cluster here, so a per-row draw could split them
    rows = [dict(r) for r in rows]
    for r in rows:
        if r["id"] == "1005_1_1_1":
            r["cluster_repr"] = "1001_1_1_1"
    for seed in range(20):
        v = ex.build_views(rows, tiers=("tolerant",), valid_fraction=0.3, seed=seed)
        v = v["trizod_tolerant"]
        tr = {ex.cluster_key(r) for r in v["train"]}
        va = {ex.cluster_key(r) for r in v["valid"]}
        assert tr.isdisjoint(va)
        assert {r["id"] for r in v["train"]}.isdisjoint({r["id"] for r in v["valid"]})


def test_valid_fraction_counts_clusters(rows):
    v = ex.build_views(rows, tiers=("tolerant",), valid_fraction=0.10, seed=0)
    v = v["trizod_tolerant"]
    n_clusters = len({ex.cluster_key(r) for r in v["train"] + v["valid"]})
    assert len({ex.cluster_key(r) for r in v["valid"]}) == round(0.10 * n_clusters)


def test_test_is_test_trizod_only_never_retired(rows):
    views = ex.build_views(rows, seed=0)
    retired = {r["id"] for r in rows if r["split"] == "test_retired"}
    assert retired
    for folder, splits in views.items():
        for split, recs in splits.items():
            assert retired.isdisjoint(r["id"] for r in recs), (folder, split)
        if folder.startswith("trizod_"):
            assert {r["id"] for r in splits["test"]} == {"9001_1_1_1", "9002_1_1_1"}


def test_deterministic_with_seed(rows, tmp_path):
    a = ex.write_views(ex.build_views(rows, seed=7), tmp_path / "a")
    b = ex.write_views(ex.build_views(list(reversed(rows)), seed=7), tmp_path / "b")
    files = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*"))
    assert files
    for rel in files:
        if (tmp_path / "a" / rel).is_file():
            assert (tmp_path / "a" / rel).read_bytes() == (
                tmp_path / "b" / rel
            ).read_bytes()
    assert a == b
    c = ex.build_views(rows, seed=8)["trizod_tolerant"]["valid"]
    d = ex.build_views(rows, seed=7)["trizod_tolerant"]["valid"]
    assert {r["id"] for r in c} != {r["id"] for r in d}


def test_invalid_fraction_rejected(rows):
    with pytest.raises(ValueError):
        ex.build_views(rows, valid_fraction=1.0)


def test_train_test_sequence_overlap_fails(rows):
    rows = [dict(r) for r in rows]
    rows.append(_row("9100_1_1_1", "test_trizod", seq="MKTAYIAK"))  # = a train seq
    with pytest.raises(ValueError, match="sequence"):
        ex.build_views(rows)


# --------------------------------------------------------------------------- #
# writing + F29 verification
# --------------------------------------------------------------------------- #


def test_written_files_and_f29(rows, tmp_path):
    views = ex.build_views(rows, seed=42)
    stats = ex.write_views(views, tmp_path)
    for folder in ("trizod_tolerant", "trizod_moderate", "trizod_strict"):
        for split in ("train", "valid", "test"):
            assert (tmp_path / folder / f"{split}.jsonl").is_file()
            assert (tmp_path / folder / f"{split}.fasta").is_file()
    assert (tmp_path / "chezod117" / "test.jsonl").is_file()

    by_id = {r["id"]: r for r in rows}
    for path in tmp_path.rglob("*.jsonl"):
        for rec in _read_jsonl(path):
            assert tuple(rec) == ("id", "x_0", "y")
            src = by_id[rec["id"]]
            assert rec["x_0"] == src["sequence"]
            for y, g, m in zip(rec["y"], src["gscores"], src["mask"], strict=True):
                if not m:
                    assert y == 999.0
                else:
                    assert _f32(y) == _f32(g)

    # stats agree with the files
    st = stats["trizod_strict"]["test"]
    recs = _read_jsonl(tmp_path / "trizod_strict" / "test.jsonl")
    assert st["n_records"] == len(recs)
    assert st["n_residues"] == sum(len(r["y"]) for r in recs)
    assert st["n_masked"] == sum(v == 999.0 for r in recs for v in r["y"])
    assert (
        st["sha256"]
        == hashlib.sha256(
            (tmp_path / "trizod_strict" / "test.jsonl").read_bytes()
        ).hexdigest()
    )

    assert ex.verify_views(tmp_path, rows, views) == []


def test_verify_catches_a_changed_value(rows, tmp_path):
    views = ex.build_views(rows, seed=42)
    ex.write_views(views, tmp_path)
    p = tmp_path / "trizod_strict" / "test.jsonl"
    recs = _read_jsonl(p)
    recs[0]["y"][1] = 0.5001
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    problems = ex.verify_views(tmp_path, rows, views)
    assert problems and any("9001_1_1_1" in s or "9002_1_1_1" in s for s in problems)


def test_verify_catches_unmasked_999(rows, tmp_path):
    views = ex.build_views(rows, seed=42)
    ex.write_views(views, tmp_path)
    p = tmp_path / "chezod117" / "test.jsonl"
    recs = _read_jsonl(p)
    recs[0]["y"][1] = 999.0  # position 1 is labelled in the Parquet
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert ex.verify_views(tmp_path, rows, views)


# --------------------------------------------------------------------------- #
# Parquet round trip (needs pyarrow)
# --------------------------------------------------------------------------- #


def _write_parquet(rows, path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("sequence", pa.string()),
            ("gscores", pa.list_(pa.float32())),
            ("mask", pa.list_(pa.bool_())),
            ("split", pa.string()),
            ("train_tier", pa.string()),
            ("cluster_repr", pa.string()),
            ("label_tier_source", pa.string()),
        ],
        metadata={"dataset_version": "test-0.0.0", "mask_semantics": "mask"},
    )
    cols = {n: [r[n] for r in rows] for n in schema.names}
    pq.write_table(pa.Table.from_pydict(cols, schema=schema), path)


def test_export_end_to_end_from_parquet(rows, tmp_path):
    src = tmp_path / "release.parquet"
    _write_parquet(rows, src)
    out = tmp_path / "out"
    summary = ex.export(src, out, seed=42)
    assert (out / "README.md").is_file()
    on_disk = json.loads((out / "export_summary.json").read_text())
    assert on_disk == summary
    assert summary["seed"] == 42
    assert summary["valid_fraction"] == ex.DEFAULT_VALID_FRACTION
    assert summary["source"]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert summary["source"]["metadata"]["dataset_version"] == "test-0.0.0"
    assert summary["f29_problems"] == []
    assert summary["split_counts"]["test_retired"] == 2
    assert summary["files"]["trizod_tolerant/test.jsonl"]["n_records"] == 2
    for rel, st in summary["files"].items():
        assert st["sha256"] == hashlib.sha256((out / rel).read_bytes()).hexdigest()
    assert set(summary["valid_ids"]) == {
        "trizod_tolerant",
        "trizod_moderate",
        "trizod_strict",
    }
    assert "code_commit" in summary


@pytest.mark.skipif(
    not os.environ.get("TRIZOD_EXPORT_PARQUET"),
    reason="set TRIZOD_EXPORT_PARQUET=<trizod_dataset.parquet> to check a release",
)
def test_export_real_release(tmp_path):
    pytest.importorskip("pyarrow")
    src = Path(os.environ["TRIZOD_EXPORT_PARQUET"])
    summary = ex.export(src, tmp_path / "out")
    assert summary["f29_problems"] == []
    rows, _ = ex.read_release(src)
    n_test = sum(r["split"] == "test_trizod" for r in rows)
    assert summary["files"]["trizod_tolerant/test.jsonl"]["n_records"] == n_test


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_registered():
    from typer.testing import CliRunner

    from trizod.cli.main import app

    res = CliRunner().invoke(app, ["dataset", "export-jsonl", "--help"])
    assert res.exit_code == 0, res.output
    for opt in ("--parquet", "--out", "--tier", "--valid-fraction", "--seed"):
        assert opt in res.output
