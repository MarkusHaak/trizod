import pytest

from trizod.dataset import testset
from trizod.dataset.paths import resolve_paths
from trizod.dataset.testset import resolve_pinned_testset
from trizod.io.fasta import count_fasta, read_fasta


def test_pinned_testset_path_and_file_present():
    paths = resolve_paths()
    assert paths.pinned_testset.name == "TriZOD_test_set.fasta"
    assert paths.pinned_testset.parent.name == "pinned"
    assert paths.pinned_testset.exists(), "committed pinned test set is missing"
    assert count_fasta(paths.pinned_testset) == 342


def test_pinned_exact_match():
    pinned = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}
    strict = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC", "300_1_1_1": "GGGG"}
    recs, info = resolve_pinned_testset(pinned, strict)
    assert recs == {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}
    assert info["dropped"] == []
    assert info["substitutions"] == []
    assert info["resolved"] == 2 and info["pinned_total"] == 2


def test_pinned_entry_id_fallback_lowest_numbered():
    pinned = {"999_1_1_1": "AAAA"}
    strict = {"100_1_1_1": "AAAA", "50_1_1_1": "AAAA"}
    recs, info = resolve_pinned_testset(pinned, strict)
    assert recs == {"50_1_1_1": "AAAA"}
    assert info["substitutions"] == [["999_1_1_1", "50_1_1_1"]]


def test_pinned_dropped_when_sequence_absent():
    pinned = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}
    strict = {"100_1_1_1": "AAAA"}
    recs, info = resolve_pinned_testset(pinned, strict)
    assert recs == {"100_1_1_1": "AAAA"}
    assert info["dropped"] == ["200_1_1_1"]


def test_pinned_stable_under_representative_reshuffle():
    pinned = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}
    strict_v1 = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}
    strict_v2 = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC", "900_1_1_1": "AAAA"}
    r1, _ = resolve_pinned_testset(pinned, strict_v1)
    r2, _ = resolve_pinned_testset(pinned, strict_v2)
    assert r1 == r2 == {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}


def _setup_root_and_wd(tmp_path, pin_records, strict_records):
    root = tmp_path / "root"
    wd = tmp_path / "wd"
    pin_dir = root / "trizod" / "dataset" / "pinned"
    pin_dir.mkdir(parents=True)
    with (pin_dir / "TriZOD_test_set.fasta").open("w") as fh:
        for rid, seq in pin_records.items():
            fh.write(f">{rid}\n{seq}\n")
    strict_dir = wd / "final_dataset" / "strict"
    strict_dir.mkdir(parents=True)
    with (strict_dir / "strict.fasta").open("w") as fh:
        for rid, seq in strict_records.items():
            fh.write(f">{rid} tier=strict\n{seq}\n")
    return root, wd


def test_main_pinned_mode_emits_pinned_set(tmp_path):
    root, wd = _setup_root_and_wd(
        tmp_path,
        pin_records={"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"},
        strict_records={"100_1_1_1": "AAAA", "200_1_1_1": "CCCC", "300_1_1_1": "GG"},
    )
    testset.main(["--work-dir", str(wd), "--root", str(root)])
    out = read_fasta(wd / "testset" / "TriZOD_test_set.fasta")
    assert out == {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}


def test_main_missing_pin_errors(tmp_path):
    root = tmp_path / "root"
    wd = tmp_path / "wd"
    (wd / "final_dataset" / "strict").mkdir(parents=True)
    (wd / "final_dataset" / "strict" / "strict.fasta").write_text(">1_1_1_1\nAA\n")
    with pytest.raises(SystemExit):
        testset.main(["--work-dir", str(wd), "--root", str(root)])
