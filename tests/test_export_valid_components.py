"""The UdonPred validation split is homology-separated (re-audit 2026-11-17b,
item 3 / G37).

Every training representative heads its own 50 %/80 % cluster, so drawing
``valid`` by whole clusters was a random 10 % of representatives: in the
tolerant view 235 of 488 valid chains had a training homolog at the leakage
rule, 46 at >= 90 % identity. ``valid`` is now drawn over the connected
components of the tier's training representatives under the leakage rule
(>= 30 % identity over >= 80 % of either chain, >= 27 bits, any reported
alignment, both directions; an all-vs-all mmseqs search), about 10 % of the
components, seed 42, so no training chain meets the rule against a valid chain.
"""

from __future__ import annotations

import json
import math
import random
import re
import shutil

import pytest

from trizod.dataset import export_jsonl as ex
from trizod.dataset import mmseqs as mm
from trizod.io.fasta import write_fasta

requires_mmseqs = pytest.mark.skipif(
    shutil.which("mmseqs") is None, reason="mmseqs binary not found in PATH"
)

AA = "ACDEFGHIKLMNPQRSTVWY"
NAN = float("nan")


def _row(rid, seq, split="train", train_tier="tolerant", cluster_repr=None):
    g = [0.5] * len(seq)
    return {
        "id": rid,
        "sequence": seq,
        "gscores": g,
        "mask": [True] * len(seq),
        "split": split,
        "train_tier": train_tier if split == "train" else None,
        "cluster_repr": cluster_repr or (rid if split == "train" else None),
        "label_tier_source": train_tier,
    }


# --------------------------------------------------------------------------- #
# components and the draw (no mmseqs)
# --------------------------------------------------------------------------- #


def test_components_are_the_connected_components_of_the_pairs():
    ids = [f"{i}_1_1_1" for i in range(1, 8)]
    pairs = [("1_1_1_1", "2_1_1_1"), ("2_1_1_1", "3_1_1_1"), ("5_1_1_1", "6_1_1_1")]
    comp = ex.leakage_components(ids, pairs)
    assert comp["1_1_1_1"] == comp["2_1_1_1"] == comp["3_1_1_1"] == "1_1_1_1"
    assert comp["5_1_1_1"] == comp["6_1_1_1"] == "5_1_1_1"
    assert {comp["4_1_1_1"], comp["7_1_1_1"]} == {"4_1_1_1", "7_1_1_1"}
    # a component's ID is its naturally smallest member (10 > 9 numerically)
    comp = ex.leakage_components(["10_1_1_1", "9_1_1_1"], [("10_1_1_1", "9_1_1_1")])
    assert set(comp.values()) == {"9_1_1_1"}


def test_pairs_outside_the_tier_are_ignored():
    # 1 ~ 3 only through 2, which is not in this tier: no edge
    comp = ex.leakage_components(
        ["1_1_1_1", "3_1_1_1"], [("1_1_1_1", "2_1_1_1"), ("2_1_1_1", "3_1_1_1")]
    )
    assert comp["1_1_1_1"] != comp["3_1_1_1"]


def test_the_draw_takes_whole_components(tmp_path):
    rng = random.Random(1)
    rows = [
        _row(f"{i}_1_1_1", "".join(rng.choice(AA) for _ in range(40)))
        for i in range(1, 41)
    ]
    # four families of five chains and 20 singletons: 24 components
    pairs = [
        (f"{f * 5 + a}_1_1_1", f"{f * 5 + a + 1}_1_1_1")
        for f in range(4)
        for a in range(1, 5)
    ]
    for seed in range(15):
        v = ex.build_views(
            rows, tiers=("tolerant",), valid_fraction=0.25, seed=seed, pairs=pairs
        )
        v = v["trizod_tolerant"]
        train = {r["id"] for r in v["train"]}
        valid = {r["id"] for r in v["valid"]}
        assert train | valid == {r["id"] for r in rows} and not train & valid
        assert not any((a in train) != (b in train) for a, b in pairs)
        comp = ex.leakage_components([r["id"] for r in rows], pairs)
        assert len({comp[i] for i in valid}) == round(0.25 * 24) == 6


def test_cluster_mates_share_a_component():
    rows = [
        _row("1_1_1_1", "MKV" * 10),
        _row("2_1_1_1", "WWG" * 10, cluster_repr="1_1_1_1"),
        _row("3_1_1_1", "PPA" * 10),
    ]
    comp = ex.representative_components(rows, pairs=())
    assert comp["1_1_1_1"] == comp["2_1_1_1"] != comp["3_1_1_1"]


def test_the_split_is_deterministic_and_seeded():
    rng = random.Random(2)
    rows = [
        _row(f"{i}_1_1_1", "".join(rng.choice(AA) for _ in range(30)))
        for i in range(1, 61)
    ]
    pairs = [(f"{i}_1_1_1", f"{i + 1}_1_1_1") for i in range(1, 60, 3)]
    a = ex.build_views(rows, seed=42, pairs=pairs)["trizod_tolerant"]["valid"]
    b = ex.build_views(list(reversed(rows)), seed=42, pairs=list(reversed(pairs)))
    assert [r["id"] for r in a] == [r["id"] for r in b["trizod_tolerant"]["valid"]]
    c = ex.build_views(rows, seed=43, pairs=pairs)["trizod_tolerant"]["valid"]
    assert {r["id"] for r in a} != {r["id"] for r in c}


def test_verify_flags_a_pair_across_train_and_valid(tmp_path):
    rows = [_row("1_1_1_1", "MKV" * 10), _row("2_1_1_1", "WWG" * 10)]
    views = {
        "trizod_tolerant": {"train": [rows[0]], "valid": [rows[1]], "test": []},
    }
    ex.write_views(views, tmp_path)
    problems = ex.verify_views(tmp_path, rows, views, pairs=[("1_1_1_1", "2_1_1_1")])
    assert any("leakage rule" in p for p in problems)
    assert ex.verify_views(tmp_path, rows, views, pairs=()) == []


# --------------------------------------------------------------------------- #
# the all-vs-all search (mmseqs)
# --------------------------------------------------------------------------- #


def _mutate(seq: str, keep_every: int, rng: random.Random) -> str:
    return "".join(
        c if i % keep_every else rng.choice(AA.replace(c, ""))
        for i, c in enumerate(seq)
    )


def _families(seed: int = 20261117):
    """Ten families of three homologues (~66 % identity to their founder,
    100 aa) and twenty unrelated singletons: 30 components."""
    rng = random.Random(seed)
    seqs = {}
    n = 1
    for _ in range(10):
        founder = "".join(rng.choice(AA) for _ in range(100))
        for k in range(3):
            seqs[f"{n}_1_1_1"] = founder if k == 0 else _mutate(founder, 3, rng)
            n += 1
    for _ in range(20):
        seqs[f"{n}_1_1_1"] = "".join(rng.choice(AA) for _ in range(100))
        n += 1
    return seqs


@requires_mmseqs
def test_the_all_vs_all_search_finds_the_families(tmp_path):
    seqs = _families()
    fa = tmp_path / "reps.fasta"
    write_fasta(seqs, fa)
    hits = mm.find_self_leaks(
        fa, tmp_path / "h.tsv", tmp_path / "tmp", tmp_path / "log"
    )
    pairs = {frozenset((h.train_id, h.test_id)) for h in hits}
    family = {f"{i}_1_1_1": (i - 1) // 3 for i in range(1, 31)}
    assert pairs and all(len(p) == 2 for p in pairs)
    for p in pairs:
        a, b = sorted(p)
        assert a in family and b in family and family[a] == family[b]
    assert all(mm.is_leak(h, "either") for h in hits)


@requires_mmseqs
def test_no_training_chain_meets_the_rule_against_a_valid_chain(tmp_path):
    seqs = _families()
    rows = [_row(i, s) for i, s in seqs.items()]
    pairs = ex.training_leakage_pairs(rows)
    comp = ex.leakage_components(list(seqs), pairs)
    assert len(set(comp.values())) == 30
    views = ex.build_views(
        rows, tiers=("tolerant",), valid_fraction=0.2, seed=42, pairs=pairs
    )
    v = views["trizod_tolerant"]
    assert len({comp[r["id"]] for r in v["valid"]}) == round(0.2 * 30)
    # an independent search: the training chains against the valid chains
    train_fa, valid_fa = tmp_path / "train.fasta", tmp_path / "valid.fasta"
    write_fasta({r["id"]: r["sequence"] for r in v["train"]}, train_fa)
    write_fasta({r["id"]: r["sequence"] for r in v["valid"]}, valid_fa)
    leaks = mm.find_leaks(
        train_fa, valid_fa, tmp_path / "h.tsv", tmp_path / "tmp", tmp_path / "log"
    )
    assert leaks == []


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
        metadata={"dataset_version": "test-0.0.0"},
    )
    cols = {n: [r[n] for r in rows] for n in schema.names}
    pq.write_table(pa.Table.from_pydict(cols, schema=schema), path)


@requires_mmseqs
def test_export_records_the_components_and_the_rule(tmp_path):
    seqs = _families()
    rows = [_row(i, s) for i, s in seqs.items()]
    rows.append(_row("9001_1_1_1", "GSHM" * 20, split="test_trizod"))
    rows.append(_row("8001_1_1_1", "AAKK" * 20, split="test_chezod117"))
    src = tmp_path / "release.parquet"
    _write_parquet(rows, src)
    summary = ex.export(src, tmp_path / "out", tiers=("tolerant",))
    assert summary["f29_problems"] == []
    comps = summary["valid_components"]["trizod_tolerant"]
    valid = set(summary["valid_ids"]["trizod_tolerant"])
    assert valid == {m for members in comps.values() for m in members}
    assert len(comps) == round(ex.DEFAULT_VALID_FRACTION * 30) == 3
    stats = summary["components"]["trizod_tolerant"]
    assert stats["n_components"] == 30 and stats["n_train_valid_pairs"] == 0
    assert stats["n_valid_components"] == 3 and stats["largest"] == 3
    assert "every reported alignment" in summary["valid_rule"]
    assert "component" in summary["valid_unit"]
    readme = (tmp_path / "out" / "README.md").read_text()
    assert "below 50 % identity" not in readme
    assert "no training chain meets the leakage rule against a valid chain" in readme
    on_disk = json.loads((tmp_path / "out" / "export_summary.json").read_text())
    assert on_disk == summary
    assert math.isclose(summary["valid_fraction"], 0.10)


def test_the_cli_help_names_homology_components():
    from typer.testing import CliRunner

    from trizod.cli.main import app

    res = CliRunner().invoke(app, ["dataset", "export-jsonl", "--help"])
    assert res.exit_code == 0, res.output
    # rich draws boxes around the options and wraps them
    text = " ".join(re.sub(r"[\u2500-\u257f]", " ", res.output).split())
    assert "homology components" in text and "clusters drawn" not in text
