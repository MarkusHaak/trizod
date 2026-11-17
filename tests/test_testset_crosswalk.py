"""Test-set crosswalk and overlap table (checks G38, E21).

One row per sequence that was ever in a TriZOD test set, its membership in each
named set and its status in the final release (exact-sequence match).
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest

from trizod.dataset import testset_crosswalk as cw


def _rel(rid, seq, split, train_tier=None, cluster_repr=None):
    return {
        "id": rid,
        "sequence": seq,
        "split": split,
        "train_tier": train_tier,
        "cluster_repr": cluster_repr,
    }


@pytest.fixture
def release():
    return [
        _rel("1_1_1_1", "AAAA", "train", "strict", "1_1_1_1"),
        _rel("1_2_2_2", "AAAA", "excluded"),  # duplicate deposition, excluded
        _rel("2_1_1_1", "CCCC", "test_trizod"),
        _rel("4_1_1_1", "GGGG", "test_retired"),
        _rel("5_1_1_1", "TTTT", "test_trizod"),  # new in the final set
        _rel("6_1_1_1", "KKKK", "redundant", None, "7_1_1_1"),
        _rel("7_1_1_1", "KKKR", "train", "tolerant", "7_1_1_1"),
        _rel("8_1_1_1", "MMMM", "test_chezod117"),
    ]


@pytest.fixture
def sets():
    return [
        cw.HeldOutSet(
            "old",
            "oldest set",
            {"spec": "fasta:x"},
            {"a1": "AAAA", "a2": "CCCC", "a3": "WWWW"},
        ),
        cw.HeldOutSet(
            "mid",
            "middle set",
            {"spec": "fasta:y"},
            {"b1": "CCCC", "b2": "GGGG", "b9": "KKKK"},
        ),
    ]


def test_crosswalk_one_row_per_sequence(sets, release):
    rows = cw.build_crosswalk(sets, release)
    seqs = [r["sequence"] for r in rows]
    # every sequence of every set plus the final test set, each once
    assert sorted(seqs) == sorted({"AAAA", "CCCC", "WWWW", "GGGG", "KKKK", "TTTT"})
    by_seq = {r["sequence"]: r for r in rows}
    assert by_seq["CCCC"]["in_old"] == 1 and by_seq["CCCC"]["in_mid"] == 1
    assert by_seq["CCCC"]["old_ids"] == "a2" and by_seq["CCCC"]["mid_ids"] == "b1"
    assert by_seq["CCCC"]["in_final"] == 1 and by_seq["CCCC"]["final_ids"] == "2_1_1_1"
    assert by_seq["TTTT"]["in_old"] == 0 and by_seq["TTTT"]["in_final"] == 1
    assert by_seq["CCCC"]["n_sets"] == 3


def test_release_status(sets, release):
    by_seq = {r["sequence"]: r for r in cw.build_crosswalk(sets, release)}
    assert by_seq["AAAA"]["release_status"] == "train"  # train beats excluded
    assert by_seq["AAAA"]["release_splits"] == "excluded;train"
    assert by_seq["AAAA"]["release_ids"] == "1_1_1_1;1_2_2_2"
    assert by_seq["AAAA"]["release_train_tier"] == "strict"
    assert by_seq["CCCC"]["release_status"] == "test"
    assert by_seq["GGGG"]["release_status"] == "retired"
    assert by_seq["WWWW"]["release_status"] == "absent"
    assert by_seq["WWWW"]["release_ids"] == ""
    assert by_seq["KKKK"]["release_status"] == "redundant"
    assert by_seq["KKKK"]["release_cluster_repr"] == "7_1_1_1"


def test_status_priority_is_total():
    assert cw.release_status([]) == "absent"
    assert cw.release_status(["excluded", "test_retired"]) == "retired"
    assert cw.release_status(["redundant", "train"]) == "train"
    assert cw.release_status(["test_chezod117"]) == "chezod117"
    assert cw.release_status(["train", "test_trizod"]) == "test"
    with pytest.raises(ValueError):
        cw.release_status(["mystery"])


def test_overlap_summary(sets, release):
    rows = cw.build_crosswalk(sets, release)
    summ = {s["set"]: s for s in cw.overlap_summary(rows, sets)}
    old = summ["old"]
    assert old["n_ids"] == 3 and old["n_sequences"] == 3
    assert old["train"] == 1 and old["train_strict"] == 1
    assert old["in_tolerant_training"] == 1
    assert old["test"] == 1 and old["absent"] == 1
    mid = summ["mid"]
    assert (mid["test"], mid["retired"], mid["redundant"], mid["train"]) == (1, 1, 1, 0)
    fin = summ["final"]
    assert fin["n_sequences"] == 2 and fin["test"] == 2


def test_pairwise_shared(sets, release):
    rows = cw.build_crosswalk(sets, release)
    pw = cw.pairwise_shared(rows, ["old", "mid", "final"])
    assert pw["old"]["mid"] == 1 and pw["mid"]["old"] == 1
    assert pw["old"]["final"] == 1
    assert pw["old"]["old"] == 3


def test_write_tsv_roundtrip(sets, release, tmp_path):
    rows = cw.build_crosswalk(sets, release)
    p = tmp_path / "x.tsv"
    cw.write_tsv(rows, p)
    with p.open() as fh:
        back = list(csv.DictReader(fh, delimiter="\t"))
    assert len(back) == len(rows)
    assert list(back[0]) == list(rows[0])
    assert {r["sequence"] for r in back} == {r["sequence"] for r in rows}


def test_parse_fasta_text():
    text = ">x1 extra\nAAA\nCC\n>x2\nGG\n"
    assert cw.parse_fasta_text(text) == {"x1": "AAACC", "x2": "GG"}


def test_load_fasta_and_git_sources(tmp_path):
    fa = tmp_path / "s.fasta"
    fa.write_text(">q1\nMKV\n>q2\nLLL\n")
    assert cw.load_source(f"fasta:{fa}") == {"q1": "MKV", "q2": "LLL"}

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*a):
        subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)

    git("init", "-q")
    (repo / "pin.fasta").write_text(">p1\nAAA\n")
    git("add", "pin.fasta")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "one")
    first = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    (repo / "pin.fasta").write_text(">p1\nAAA\n>p2\nCCC\n")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "two")
    assert cw.load_source(f"git:{first}:pin.fasta", repo=repo) == {"p1": "AAA"}
    assert cw.load_source("git:HEAD:pin.fasta", repo=repo) == {"p1": "AAA", "p2": "CCC"}


def test_bad_source_spec():
    with pytest.raises(ValueError):
        cw.load_source("ftp:somewhere")


def test_parquet_source(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    p = tmp_path / "r.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["a", "b", "c"],
                "sequence": ["AA", "CC", "GG"],
                "split": ["test_trizod", "train", "test_trizod"],
            }
        ),
        p,
    )
    assert cw.load_source(f"parquet:{p}") == {"a": "AA", "c": "GG"}


def test_default_specs_name_every_earlier_set(tmp_path):
    names = [s[0] for s in cw.default_set_specs(tmp_path)]
    assert names == [
        "trizod2024_348",
        "v0.2.0_344",
        "pin_2026-07-13_342",
        "v0.3.0_365",
        "pin_2026-08-11_460",
        "pin_2026-11-10_452",
    ]


def test_run_writes_outputs(sets, release, tmp_path):
    out = tmp_path / "cw"
    res = cw.run(sets, release, out, release_source={"path": "x", "sha256": "0"})
    assert (out / "test_set_crosswalk.tsv").is_file()
    assert (out / "test_set_overlap.tsv").is_file()
    assert (out / "test_set_crosswalk.json").is_file()
    assert res["n_sequences"] == 6
    assert "exact" in res["match"]


def test_cli_registered():
    from typer.testing import CliRunner

    from trizod.cli.main import app

    res = CliRunner().invoke(app, ["dataset", "test-crosswalk", "--help"])
    assert res.exit_code == 0, res.output
    for opt in ("--parquet", "--out", "--root", "--set"):
        assert opt in res.output


def test_real_pins_present_in_git():
    """The pin revisions the defaults name exist in this repository."""
    repo = Path(cw.__file__).resolve().parents[2]
    probe = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", "cc5b006^{commit}"],
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip("pin history not available (no git checkout)")
    for name, spec, _ in cw.default_set_specs(repo):
        if spec.startswith("git:"):
            members = cw.load_source(spec, repo=repo)
            n = int(name.rsplit("_", 1)[1])
            assert len(members) == n, name
