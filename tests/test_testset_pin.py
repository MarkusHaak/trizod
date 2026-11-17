import json
import shutil

import pytest

from trizod.dataset import testset
from trizod.dataset.paths import resolve_paths
from trizod.dataset.testset import resolve_pinned_testset
from trizod.io.fasta import count_fasta, read_fasta, write_fasta

requires_mmseqs = pytest.mark.skipif(
    shutil.which("mmseqs") is None, reason="mmseqs binary not found in PATH"
)


def test_pinned_testset_path_and_file_present():
    paths = resolve_paths()
    assert paths.pinned_testset.name == "TriZOD_test_set.fasta"
    assert paths.pinned_testset.parent.name == "pinned"
    assert paths.pinned_testset.exists(), "committed pinned test set is missing"
    # Count must match the committed provenance (FASTA + provenance are written
    # together by _write_pin), so an intentional redraw can't leave this stale.
    prov = json.loads(paths.pinned_testset.with_suffix(".provenance.json").read_text())
    assert count_fasta(paths.pinned_testset) == prov["count"]  # currently 365


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


def test_pinned_id_preferred_over_lower_numbered_duplicate():
    # The pinned representative (100) is still present, but a LOWER-numbered
    # entry (50) now shares the same sequence. The pinned ID must win, so the
    # emitted record is 100 (not the lowest-numbered 50) and no substitution.
    pinned = {"100_1_1_1": "AAAA"}
    strict = {"50_1_1_1": "AAAA", "100_1_1_1": "AAAA"}
    recs, info = resolve_pinned_testset(pinned, strict)
    assert recs == {"100_1_1_1": "AAAA"}
    assert info["substitutions"] == []


def _setup_root_and_wd(tmp_path, pin_records, tier_records):
    """Lay out a fake repo root (pin) + work dir (per-tier pools).

    ``tier_records`` maps a tier name to its {id: seq} pool; every named tier
    gets a ``final_dataset/<tier>/<tier>.fasta`` with build.py-style headers.
    """
    root = tmp_path / "root"
    wd = tmp_path / "wd"
    pin_dir = root / "trizod" / "dataset" / "pinned"
    pin_dir.mkdir(parents=True)
    write_fasta(pin_records, pin_dir / "TriZOD_test_set.fasta")
    for tier, records in tier_records.items():
        tier_dir = wd / "final_dataset" / tier
        tier_dir.mkdir(parents=True)
        with (tier_dir / f"{tier}.fasta").open("w") as fh:
            for rid, seq in records.items():
                fh.write(f">{rid} tier={tier}\n{seq}\n")
    # CheZOD117, released and original: the test chains are checked against
    # them (release audit E23); unrelated to every test chain here
    ext = root / "data" / "external"
    (ext / "chezod117").mkdir(parents=True)
    write_fasta(
        {"9_1_1_1": "MKTAYIAKQRQISFVKSHFSRQ"},
        ext / "chezod117" / "CheZOD117_test_set.fasta",
    )
    (ext / "chezod" / "seth_github").mkdir(parents=True)
    write_fasta(
        {"hmbdi": "KHDHHQGHTHSGKGHACHHEHN"},
        ext / "chezod" / "seth_github" / "CheZOD117_test_set_sequences.fasta",
    )
    return root, wd


@requires_mmseqs
def test_main_pinned_mode_emits_pinned_set(tmp_path):
    pool = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC", "300_1_1_1": "GG"}
    root, wd = _setup_root_and_wd(
        tmp_path,
        pin_records={"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"},
        tier_records={"strict": pool, "moderate": pool, "tolerant": pool},
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


def test_duplicate_sequence_pin_raises():
    # Two distinct pinned IDs carry the same sequence but only one strict entry
    # has it, so both resolve to the same entry. This must fail loudly (a real
    # exception, not a bare assert that -O would strip) rather than silently
    # collapsing the two records into one.
    pinned = {"999_1_1_1": "AAAA", "888_1_1_1": "AAAA"}
    strict = {"100_1_1_1": "AAAA"}
    with pytest.raises(ValueError, match="unique sequences"):
        resolve_pinned_testset(pinned, strict)


def test_label_tier_marks_strictest_tier_satisfied():
    # Same three sequences, differing in how far up the tier ladder they get.
    tier_pools = {
        "strict": {"100_1_1_1": "AAAA"},
        "moderate": {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"},
        "tolerant": {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC", "300_1_1_1": "GGGG"},
    }
    pinned = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC", "300_1_1_1": "GGGG"}
    recs, info = resolve_pinned_testset(
        pinned, tier_pools["tolerant"], tier_pools=tier_pools
    )
    assert len(recs) == 3
    assert info["label_tiers"] == {
        "100_1_1_1": "strict",
        "200_1_1_1": "moderate",
        "300_1_1_1": "tolerant",
    }
    assert info["label_tier_counts"] == {"strict": 1, "moderate": 1, "tolerant": 1}


def test_label_tier_independent_of_tier_pool_argument_order():
    # The label must be the STRICTEST satisfied tier regardless of the order
    # the pools are handed in (a caller-built dict need not be sorted).
    pools = {
        "tolerant": {"100_1_1_1": "AAAA"},
        "strict": {"100_1_1_1": "AAAA"},
    }
    _, info = resolve_pinned_testset(
        {"100_1_1_1": "AAAA"}, pools["tolerant"], tier_pools=pools
    )
    assert info["label_tiers"] == {"100_1_1_1": "strict"}


def test_the_test_set_is_resolved_against_strict():
    # The TriZOD test set is defined as strict-tier sequences (2026-11).
    assert testset.RESOLVE_TIER == "strict"


def test_resolve_against_strict_drops_chain_that_left_strict():
    # 200 left strict after a rule change but is still tolerant: it is no longer
    # a test chain, and the pin is trimmed after the rebuild. A looser pool
    # would keep it, labelled non-strict -- that is what label_tier is for.
    strict = {"100_1_1_1": "AAAA"}
    tolerant = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}
    pinned = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}
    recs, info = resolve_pinned_testset(
        pinned, strict, tier_pools={"strict": strict, "tolerant": tolerant}
    )
    assert set(recs) == {"100_1_1_1"}
    assert info["dropped"] == ["200_1_1_1"]
    assert info["label_tier_counts"] == {"strict": 1}
    recs_tol, info_tol = resolve_pinned_testset(
        pinned, tolerant, tier_pools={"strict": strict, "tolerant": tolerant}
    )
    assert set(recs_tol) == {"100_1_1_1", "200_1_1_1"}
    assert info_tol["label_tiers"]["200_1_1_1"] == "tolerant"


def test_id_map_covers_every_resolved_chain_and_flags_substitution():
    # 999 is gone from the pool and its sequence now sits under 50, so the ID
    # changes; the map must record the v0.3.0 -> current pair explicitly.
    pinned = {"100_1_1_1": "AAAA", "999_1_1_1": "CCCC"}
    pool = {"100_1_1_1": "AAAA", "50_1_1_1": "CCCC"}
    recs, info = resolve_pinned_testset(pinned, pool, tier_pools={"tolerant": pool})
    assert set(recs) == {"100_1_1_1", "50_1_1_1"}
    by_pin = {row["pinned_id"]: row for row in info["id_map"]}
    assert set(by_pin) == {"100_1_1_1", "999_1_1_1"}
    assert by_pin["100_1_1_1"] == {
        "pinned_id": "100_1_1_1",
        "test_id": "100_1_1_1",
        "substituted": False,
        "label_tier": "tolerant",
    }
    assert by_pin["999_1_1_1"]["test_id"] == "50_1_1_1"
    assert by_pin["999_1_1_1"]["substituted"] is True


def _three_tier_pin(tmp_path):
    # 200 is only in tolerant, 300 only in moderate+tolerant.
    strict = {"100_1_1_1": "AAAA"}
    moderate = {"100_1_1_1": "AAAA", "300_1_1_1": "GGGG"}
    tolerant = {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC", "300_1_1_1": "GGGG"}
    return _setup_root_and_wd(
        tmp_path,
        pin_records={"100_1_1_1": "AAAA", "200_1_1_1": "CCCC", "300_1_1_1": "GGGG"},
        tier_records={"strict": strict, "moderate": moderate, "tolerant": tolerant},
    )


@requires_mmseqs
def test_main_resolves_against_strict_and_retires_the_rest(tmp_path, capsys):
    # 200 (tolerant only) and 300 (moderate) left strict: they are no longer
    # test chains, but they are RETIRED, not forgotten -- written to their own
    # FASTA so the redundancy step keeps them (and their homologues) out of
    # training exactly as if they were still test chains.
    root, wd = _three_tier_pin(tmp_path)
    testset.main(["--work-dir", str(wd), "--root", str(root)])

    out_dir = wd / "testset"
    assert read_fasta(out_dir / "TriZOD_test_set.fasta") == {"100_1_1_1": "AAAA"}
    assert read_fasta(out_dir / testset.RETIRED_FASTA) == {
        "200_1_1_1": "CCCC",
        "300_1_1_1": "GGGG",
    }
    headers = [
        ln.strip() for ln in (out_dir / testset.RETIRED_FASTA).open() if ln[0] == ">"
    ]
    assert headers == [
        ">200_1_1_1 label_tier=tolerant pinned_id=200_1_1_1 reason=not_strict",
        ">300_1_1_1 label_tier=moderate pinned_id=300_1_1_1 reason=not_strict",
    ]
    summary = json.loads((out_dir / "build_test_set_summary.json").read_text())
    assert summary["resolve_tier"] == "strict"
    assert summary["retire_tier"] == "tolerant"
    assert summary["retired"] == ["200_1_1_1", "300_1_1_1"]
    assert summary["dropped"] == []
    assert summary["label_tier_counts"] == {"strict": 1}
    assert "retired" in capsys.readouterr().out


@requires_mmseqs
def test_retired_chains_are_listed_in_the_labels_table(tmp_path):
    root, wd = _three_tier_pin(tmp_path)
    testset.main(["--work-dir", str(wd), "--root", str(root)])
    rows = [
        ln.rstrip("\n").split("\t")
        for ln in (wd / "testset" / "TriZOD_test_set_labels.tsv").open()
    ]
    assert rows[0][-2:] == ["status", "reason"]
    status = {r[0]: (r[3], r[5]) for r in rows[1:]}
    assert status == {
        "100_1_1_1": ("strict", "test"),
        "200_1_1_1": ("tolerant", "retired"),
        "300_1_1_1": ("moderate", "retired"),
    }


@requires_mmseqs
def test_a_pinned_sequence_in_no_pool_is_dropped_not_retired(tmp_path):
    # e.g. a withdrawn entry with no live twin: nothing to keep out of training
    pool = {"100_1_1_1": "AAAA"}
    root, wd = _setup_root_and_wd(
        tmp_path,
        pin_records={"100_1_1_1": "AAAA", "999_1_1_1": "WWWW"},
        tier_records={"strict": pool, "tolerant": pool},
    )
    testset.main(["--work-dir", str(wd), "--root", str(root)])
    summary = json.loads((wd / "testset" / "build_test_set_summary.json").read_text())
    assert summary["dropped"] == ["999_1_1_1"]
    assert summary["retired"] == []
    assert read_fasta(wd / "testset" / testset.RETIRED_FASTA) == {}


def test_a_redraw_removes_a_stale_retired_file(tmp_path, monkeypatch):
    root, wd = _setup_root_and_wd(
        tmp_path,
        pin_records={"100_1_1_1": "AAAA"},
        tier_records={"strict": {"100_1_1_1": "AAAA"}},
    )
    monkeypatch.setattr(testset, "_redraw", lambda paths, out: {"7_1_1_1": "TTTT"})
    stale = wd / "testset"
    stale.mkdir(parents=True)
    (stale / testset.RETIRED_FASTA).write_text(">1_1_1_1\nAA\n")
    testset.main(
        ["--work-dir", str(wd), "--root", str(root), "--redraw", "--confirm-redraw"]
    )
    assert not (stale / testset.RETIRED_FASTA).exists()


@requires_mmseqs
def test_main_with_a_looser_resolve_tier_writes_labels(tmp_path):
    root, wd = _three_tier_pin(tmp_path)
    testset.main(
        ["--work-dir", str(wd), "--root", str(root), "--resolve-tier", "tolerant"]
    )

    out_dir = wd / "testset"
    assert read_fasta(out_dir / "TriZOD_test_set.fasta") == {
        "100_1_1_1": "AAAA",
        "200_1_1_1": "CCCC",
        "300_1_1_1": "GGGG",
    }
    # label_tier is carried on the FASTA header too, so the sequence file is
    # self-describing without the sidecar.
    headers = [
        ln.strip()
        for ln in (out_dir / "TriZOD_test_set.fasta").open()
        if ln.startswith(">")
    ]
    assert ">100_1_1_1 label_tier=strict pinned_id=100_1_1_1" in headers
    assert ">200_1_1_1 label_tier=tolerant pinned_id=200_1_1_1" in headers

    rows = [
        ln.rstrip("\n").split("\t")
        for ln in (out_dir / "TriZOD_test_set_labels.tsv").open()
    ]
    assert rows[0] == [
        "test_id",
        "pinned_id",
        "substituted",
        "label_tier",
        "length",
        "status",
        "reason",
    ]
    labels = {r[0]: r[3] for r in rows[1:]}
    assert labels == {
        "100_1_1_1": "strict",
        "200_1_1_1": "tolerant",
        "300_1_1_1": "moderate",
    }

    summary = json.loads((out_dir / "build_test_set_summary.json").read_text())
    assert summary["resolve_tier"] == "tolerant"
    assert summary["resolved"] == 3
    assert summary["dropped"] == []
    assert summary["label_tier_counts"] == {"strict": 1, "moderate": 1, "tolerant": 1}


@requires_mmseqs
def test_main_records_id_substitution_in_label_map(tmp_path):
    # The pinned representative 999 lost its pool ID to the lower-numbered 50.
    strict = {"50_1_1_1": "AAAA"}
    root, wd = _setup_root_and_wd(
        tmp_path,
        pin_records={"999_1_1_1": "AAAA"},
        tier_records={"strict": strict},
    )
    testset.main(["--work-dir", str(wd), "--root", str(root)])
    rows = [
        ln.rstrip("\n").split("\t")
        for ln in (wd / "testset" / "TriZOD_test_set_labels.tsv").open()
    ]
    assert rows[1][:3] == ["50_1_1_1", "999_1_1_1", "True"]


def test_redraw_requires_explicit_confirmation(tmp_path, monkeypatch):
    root, wd = _setup_root_and_wd(
        tmp_path,
        pin_records={"100_1_1_1": "AAAA"},
        tier_records={"tolerant": {"100_1_1_1": "AAAA"}},
    )
    called = []
    monkeypatch.setattr(testset, "_redraw", lambda *a, **k: called.append(1) or {})
    with pytest.raises(SystemExit, match="comparability"):
        testset.main(["--work-dir", str(wd), "--root", str(root), "--redraw"])
    assert called == [], "redraw must not run without confirmation"
    # the committed pin is untouched
    assert read_fasta(root / "trizod" / "dataset" / "pinned" / "TriZOD_test_set.fasta")


def test_redraw_runs_when_confirmed(tmp_path, monkeypatch):
    root, wd = _setup_root_and_wd(
        tmp_path,
        pin_records={"100_1_1_1": "AAAA"},
        tier_records={"tolerant": {"100_1_1_1": "AAAA"}},
    )
    monkeypatch.setattr(testset, "_redraw", lambda paths, out: {"7_1_1_1": "TTTT"})
    # A labels sidecar left behind by an earlier pinned run describes the OLD
    # test set; a redraw must not leave it next to the new FASTA.
    stale = wd / "testset"
    stale.mkdir(parents=True)
    (stale / "TriZOD_test_set_labels.tsv").write_text("test_id\n100_1_1_1\n")
    testset.main(
        ["--work-dir", str(wd), "--root", str(root), "--redraw", "--confirm-redraw"]
    )
    pin = root / "trizod" / "dataset" / "pinned" / "TriZOD_test_set.fasta"
    assert read_fasta(pin) == {"7_1_1_1": "TTTT"}
    assert not (stale / "TriZOD_test_set_labels.tsv").exists()


def test_every_pinned_sequence_is_a_test_or_a_retired_chain():
    """Guard the pin on whatever pools are actually on disk.

    The ``data/interim/build`` artefacts can be stale relative to
    ``data/interim/scored`` (they are rebuilt only by ``trizod dataset
    build``); this reads what is there and skips if the build has never run.
    """
    paths = resolve_paths()
    strict = paths.final_dataset / "strict" / "strict.fasta"
    if not strict.exists():
        pytest.skip(f"no built strict pool at {strict}")
    pinned = read_fasta(paths.pinned_testset)
    tier_pools = testset.load_tier_pools(paths.final_dataset)
    recs, info = resolve_pinned_testset(
        pinned, tier_pools["strict"], tier_pools=tier_pools
    )
    retired, rinfo = testset.retire_dropped(pinned, info["dropped"], tier_pools)
    # The invariant, not the count: every pinned sequence is either a test
    # chain (strict) or a retired one (still a leakage target). A rule change
    # moving test chains out of strict is expected and handled by retiring
    # them; a sequence in no pool at all means the pool moved under the pin.
    assert rinfo["dropped"] == [], (
        f"pinned sequences in neither the strict nor the tolerant pool: {rinfo}"
    )
    assert len(recs) + len(retired) == len(pinned)


def test_write_pin_writes_fasta_and_provenance(tmp_path):
    pin_path = tmp_path / "pinned" / "TriZOD_test_set.fasta"
    testset._write_pin(pin_path, {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"})
    assert read_fasta(pin_path) == {"100_1_1_1": "AAAA", "200_1_1_1": "CCCC"}
    prov = json.loads(pin_path.with_suffix(".provenance.json").read_text())
    assert prov["count"] == 2
    assert prov["mode"] == "redraw"
    assert prov["seed"] == testset.SEED
