"""Test-set leakage criterion: coverage of the TEST chain, not of both chains.

A training chain leaks when an alignment to a held-out test chain reaches
>= 30 % identity over >= 80 % of the *test* chain. The original stage-2 search
used ``--cov-mode 0`` (>= 80 % of BOTH chains), so a longer training construct
that contains a test protein survived, and so did a pair whose length ratio is
below 0.8 (27177_1_1_1, 124 aa, vs test 27176_1_1_1, 99 aa, 79.8 % identity).

The fixtures below rebuild both cases from seeded random sequences:

* containment   - a 120-aa test chain embedded in a 420-aa training construct
* length ratio  - a 99-aa test chain vs a 124-aa training chain at 79.8 %
                  identity (99/124 = 0.798 < 0.8)
* deletion      - the mirror case: a 99-aa training chain that is a 124-aa test
                  chain minus a 25-residue internal loop. The alignment spans the
                  whole test chain (mmseqs coverage counts the gap) at 79.8 %
                  identity, but the training chain is shorter than 80 % of the
                  test chain, so any mmseqs coverage threshold above 0 skips the
                  pair before alignment.
* fragment      - a training chain that is exactly the first half of a test
                  chain: the test chain CONTAINS a shorter training chain. Not a
                  leak under the "test" rule (50 % of the test chain), a leak
                  under the "either" rule (100 % of the training chain), and an
                  exact-substring containment the release gate always rejects
* mutated fragment - the second half of the same test chain with a
                  substitution every 8th residue: a leak only under "either",
                  and no shared 20-mer, so the substring check passes it
* unrelated     - a random training chain (no leak)

Leakage rules (decision D2, ``trizod.dataset.mmseqs.LEAKAGE_RULES``):
"test" = >= 80 % of the test chain; "either" (default) = >= 80 % of the test
chain or of the training chain, i.e. of the shorter one. Whatever the rule, the
release gate also fails on exact-substring containment of >= 20 aa in either
direction, and it checks the rule the redundancy step recorded.

mmseqs-dependent tests skip cleanly when the binary is not on PATH.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from trizod.dataset import mmseqs as mm
from trizod.dataset import package_release, redundancy
from trizod.io.fasta import read_fasta, write_fasta

requires_mmseqs = pytest.mark.skipif(
    shutil.which("mmseqs") is None, reason="mmseqs binary not found in PATH"
)

AA = "ACDEFGHIKLMNPQRSTVWY"

CONTAIN_TEST = "100_1_1_1"
CONTAIN_TRAIN = "900_1_1_1"
RATIO_TEST = "101_1_1_1"
RATIO_TRAIN = "901_1_1_1"
FRAG_TEST = "102_1_1_1"
FRAG_TRAIN = "902_1_1_1"
UNRELATED_TRAIN = "903_1_1_1"
DELETION_TEST = "103_1_1_1"
DELETION_TRAIN = "904_1_1_1"
MUTFRAG_TRAIN = "906_1_1_1"


def _rand(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(AA) for _ in range(n))


def _substitute(seq: str, positions: list[int]) -> str:
    """Replace each listed position with a different residue."""
    s = list(seq)
    for p in positions:
        s[p] = AA[(AA.index(s[p]) + 7) % len(AA)]
    return "".join(s)


def _shares_kmer(a: str, b: str, k: int) -> bool:
    return any(a[i : i + k] in b for i in range(len(a) - k + 1))


def _fixture_seqs() -> tuple[dict[str, str], dict[str, str]]:
    rng = random.Random(20261030)
    contain_test = _rand(rng, 120)
    ratio_test = _rand(rng, 99)
    frag_test = _rand(rng, 100)
    test = {
        CONTAIN_TEST: contain_test,
        RATIO_TEST: ratio_test,
        FRAG_TEST: frag_test,
    }
    # 20 of 99 positions mutated, spread over the chain so the local alignment
    # spans the whole test chain: 79/99 = 79.8 % identity.
    ratio_train = _substitute(ratio_test, list(range(2, 99, 5))[:20]) + _rand(rng, 25)
    train = {
        CONTAIN_TRAIN: _rand(rng, 150) + contain_test + _rand(rng, 150),
        RATIO_TRAIN: ratio_train,
        FRAG_TRAIN: frag_test[:50],
        UNRELATED_TRAIN: _rand(rng, 200),
    }
    # drawn last so the sequences above stay as they were
    deletion_test = _rand(rng, 124)
    test[DELETION_TEST] = deletion_test
    train[DELETION_TRAIN] = deletion_test[:50] + deletion_test[75:]
    # no further random draws: derived from frag_test so the above stay fixed
    train[MUTFRAG_TRAIN] = _substitute(frag_test[50:], list(range(3, 50, 8)))
    assert not _shares_kmer(train[MUTFRAG_TRAIN], frag_test, 20)
    assert train[FRAG_TRAIN] in frag_test
    assert len(train[RATIO_TRAIN]) == 124 and len(test[RATIO_TEST]) == 99
    assert len(train[DELETION_TRAIN]) == 99 and len(test[DELETION_TEST]) == 124
    assert contain_test in train[CONTAIN_TRAIN]
    return train, test


@pytest.fixture
def fastas(tmp_path: Path) -> tuple[Path, Path]:
    train, test = _fixture_seqs()
    train_fa = tmp_path / "train.fasta"
    test_fa = tmp_path / "test.fasta"
    write_fasta(train, train_fa)
    write_fasta(test, test_fa)
    return train_fa, test_fa


def _make_bundle(root: Path, train: dict[str, str], test: dict[str, str]) -> Path:
    """Minimal staged-bundle layout as assert_no_leakage reads it."""
    bundle = root / "bundle"
    chezod = dict(list(test.items())[:1])
    trizod = dict(list(test.items())[1:])
    (bundle / "test").mkdir(parents=True)
    write_fasta(chezod, bundle / "test" / "CheZOD117_test_set.fasta")
    write_fasta(trizod, bundle / "test" / "TriZOD_test_set.fasta")
    for tier in ("strict", "tolerant"):
        (bundle / "train" / tier).mkdir(parents=True)
        write_fasta(train, bundle / "train" / tier / f"train_{tier}_best.fasta")
        write_fasta(train, bundle / "train" / tier / f"train_{tier}.fasta")
    return bundle


# ---------------------------------------------------------------- command


def test_leakage_search_cmd_requires_test_chain_coverage(tmp_path):
    cmd = mm.leakage_search_cmd(
        tmp_path / "train.fa", tmp_path / "test.fa", tmp_path / "hits.tsv", tmp_path
    )
    assert cmd[:2] == ["mmseqs", "easy-search"]
    # orientation: query = training chains, target = test chains ...
    assert cmd[2] == str(tmp_path / "train.fa")
    assert cmd[3] == str(tmp_path / "test.fa")
    # ... so --cov-mode 1 (coverage of the target) is coverage of the TEST chain
    assert cmd.count("--cov-mode") == 1
    assert cmd[cmd.index("--cov-mode") + 1] == "1"
    # no coverage threshold inside mmseqs: any -c > 0 also skips a pair by
    # length ratio before alignment; the 80 % is applied to the parsed span
    assert cmd[cmd.index("-c") + 1] == "0"
    assert cmd[cmd.index("--min-seq-id") + 1] == "0.3"
    # the rest of the shared option block is unchanged
    for flag, val in [
        ("--alignment-mode", "3"),
        ("-s", "7.5"),
        ("--comp-bias-corr", "0"),
        ("--mask", "0"),
    ]:
        assert cmd[cmd.index(flag) + 1] == val
    fmt = cmd[cmd.index("--format-output") + 1].split(",")
    assert {"tstart", "tend", "tlen"} <= set(fmt)


def _tsv_row(**fields) -> str:
    return "\t".join(str(fields[c]) for c in mm.LEAKAGE_FORMAT.split(",")) + "\n"


def _row(train_id, test_id, tstart, tend, tlen, pident=50.0) -> str:
    return _tsv_row(
        query=train_id,
        target=test_id,
        pident=pident,
        evalue=1e-10,
        bits=100,
        qstart=1,
        qend=tend - tstart + 1,
        qlen=tend - tstart + 1,
        tstart=tstart,
        tend=tend,
        tlen=tlen,
    )


def test_test_coverage_is_the_exact_aligned_span(tmp_path):
    """Test coverage is (tend - tstart + 1) / tlen, compared unrounded: mmseqs
    prints tcov with 3 decimals, so 2000/2501 = 0.79968 would read 0.800."""
    tsv = tmp_path / "hits.tsv"
    tsv.write_text(
        _row("t_edge", "T1", 1, 2000, 2501)  # 0.79968: not a leak
        + _row("t_exact", "T2", 3, 10, 10)  # 8/10 = 0.8: a leak
        + _row("t_half", "T3", 1, 50, 100)  # 0.5: not a leak
    )
    hits = {h.train_id: h for h in mm.parse_leakage_hits(tsv)}
    assert hits["t_edge"].test_cov == pytest.approx(2000 / 2501)
    assert hits["t_exact"].test_cov == pytest.approx(0.8)
    leaks = {h.train_id for h in hits.values() if mm.is_leak(h, rule="test")}
    assert leaks == {"t_exact"}


def test_common_block_still_bidirectional():
    """Stage 1 and the 50/80 clustering keep --cov-mode 0 (unchanged)."""
    assert mm.COMMON[mm.COMMON.index("--cov-mode") + 1] == "0"


# ------------------------------------------------- old criterion (control)


@requires_mmseqs
def test_bidirectional_coverage_misses_containment_and_length_ratio(fastas, tmp_path):
    """Control: the pre-fix stage-2 flags (--cov-mode 0) miss both leak types,
    so the fixtures really reproduce the bug the fix targets."""
    train_fa, test_fa = fastas
    hits = tmp_path / "old_hits.tsv"
    subprocess.run(
        [
            "mmseqs",
            "easy-search",
            str(train_fa),
            str(test_fa),
            str(hits),
            str(tmp_path / "old_tmp"),
            "--min-seq-id",
            "0.3",
            "-c",
            "0.8",
            *mm.COMMON,
            "--format-output",
            "query,target",
        ],
        check=True,
        capture_output=True,
    )
    found = {ln.split("\t", 1)[0] for ln in hits.read_text().splitlines() if ln}
    assert CONTAIN_TRAIN not in found
    assert RATIO_TRAIN not in found


@requires_mmseqs
def test_target_coverage_threshold_skips_shorter_training_chain(fastas, tmp_path):
    """Control: with --cov-mode 1 -c 0.8, mmseqs skips a training chain shorter
    than 80 % of the test chain before alignment, although its alignment spans
    the whole test chain. This is why the leakage search runs with -c 0."""
    train_fa, test_fa = fastas
    hits = tmp_path / "c08_hits.tsv"
    subprocess.run(
        [
            "mmseqs",
            "easy-search",
            str(train_fa),
            str(test_fa),
            str(hits),
            str(tmp_path / "c08_tmp"),
            "--min-seq-id",
            "0.3",
            "-c",
            "0.8",
            *mm.with_cov_mode(mm.COMMON, 1),
            "--format-output",
            "query,target",
        ],
        check=True,
        capture_output=True,
    )
    found = {ln.split("\t", 1)[0] for ln in hits.read_text().splitlines() if ln}
    assert {CONTAIN_TRAIN, RATIO_TRAIN} <= found
    assert DELETION_TRAIN not in found


# --------------------------------------------------------------- stage 2


@requires_mmseqs
def test_leak_search_reports_test_chain_coverage(fastas, tmp_path):
    train_fa, test_fa = fastas
    raw = tmp_path / "hits.tsv"
    hits = mm.find_test_coverage_leaks(
        train_fa, test_fa, raw, tmp_path / "tmp", tmp_path / "log"
    )
    pairs = {(h.train_id, h.test_id): h for h in hits}
    assert set(pairs) == {
        (CONTAIN_TRAIN, CONTAIN_TEST),
        (RATIO_TRAIN, RATIO_TEST),
        (DELETION_TRAIN, DELETION_TEST),
    }
    # mmseqs reports the half-length fragment too (no coverage threshold
    # inside mmseqs); the test-coverage filter is what drops it
    raw_ids = {ln.split("\t", 1)[0] for ln in raw.read_text().splitlines() if ln}
    assert FRAG_TRAIN in raw_ids
    contain = pairs[(CONTAIN_TRAIN, CONTAIN_TEST)]
    assert contain.test_cov >= 0.99 and contain.train_cov < 0.5
    assert contain.pident == pytest.approx(100.0)
    ratio = pairs[(RATIO_TRAIN, RATIO_TEST)]
    assert ratio.test_cov >= 0.8
    assert 30.0 <= ratio.pident < 80.0
    assert (ratio.train_len, ratio.test_len) == (124, 99)
    deletion = pairs[(DELETION_TRAIN, DELETION_TEST)]
    assert deletion.test_cov == pytest.approx(1.0)
    assert deletion.pident == pytest.approx(79.8, abs=0.1)
    assert (deletion.train_len, deletion.test_len) == (99, 124)


@requires_mmseqs
def test_stage2_drops_containment_and_length_ratio_leaks(fastas, tmp_path):
    train_fa, test_fa = fastas
    removed = redundancy.stage2_search_removal(
        train_fa,
        test_fa,
        tmp_path / "hits.tsv",
        tmp_path / "tmp",
        tmp_path / "log",
        rule="test",
    )
    assert removed == {CONTAIN_TRAIN, RATIO_TRAIN, DELETION_TRAIN}
    # the "test" criterion is coverage of the TEST chain: a training fragment
    # that covers half of a test chain is not a leak, nor is an unrelated chain
    assert FRAG_TRAIN not in removed
    assert MUTFRAG_TRAIN not in removed
    assert UNRELATED_TRAIN not in removed


# ------------------------------------------------------- packaging gate


@requires_mmseqs
def test_gate_fails_on_containment_and_reports_pairs(tmp_path):
    train, test = _fixture_seqs()
    bundle = _make_bundle(tmp_path, train, test)
    with pytest.raises(SystemExit) as exc:
        package_release.assert_no_leakage(bundle, rule="test")
    msg = str(exc.value)
    assert "LEAKAGE GATE FAILED" in msg
    for tid in (
        CONTAIN_TRAIN,
        CONTAIN_TEST,
        RATIO_TRAIN,
        RATIO_TEST,
        DELETION_TRAIN,
        DELETION_TEST,
    ):
        assert tid in msg
    # the exact half-length fragment is reported by the substring check, not
    # by the (default, test-coverage) rule; the mutated one by neither
    assert FRAG_TRAIN in msg
    assert MUTFRAG_TRAIN not in msg and UNRELATED_TRAIN not in msg


@requires_mmseqs
def test_gate_fails_on_length_ratio_pair_alone(tmp_path):
    train, test = _fixture_seqs()
    train = {k: train[k] for k in (RATIO_TRAIN, MUTFRAG_TRAIN, UNRELATED_TRAIN)}
    bundle = _make_bundle(tmp_path, train, test)
    with pytest.raises(SystemExit) as exc:
        package_release.assert_no_leakage(bundle, rule="test")
    assert RATIO_TRAIN in str(exc.value) and RATIO_TEST in str(exc.value)


@requires_mmseqs
def test_gate_fails_on_shorter_training_chain_spanning_test_chain(tmp_path):
    train, test = _fixture_seqs()
    train = {k: train[k] for k in (DELETION_TRAIN, MUTFRAG_TRAIN, UNRELATED_TRAIN)}
    bundle = _make_bundle(tmp_path, train, test)
    with pytest.raises(SystemExit) as exc:
        package_release.assert_no_leakage(bundle, rule="test")
    msg = str(exc.value)
    assert DELETION_TRAIN in msg and DELETION_TEST in msg
    assert MUTFRAG_TRAIN not in msg


@requires_mmseqs
def test_gate_passes_clean_bundle_and_leaves_it_untouched(tmp_path, capsys):
    train, test = _fixture_seqs()
    clean = {k: train[k] for k in (MUTFRAG_TRAIN, UNRELATED_TRAIN)}
    bundle = _make_bundle(tmp_path, clean, test)
    before = sorted(p.relative_to(bundle) for p in bundle.rglob("*"))
    package_release.assert_no_leakage(bundle, rule="test")
    after = sorted(p.relative_to(bundle) for p in bundle.rglob("*"))
    assert before == after  # no mmseqs scratch left inside the bundle
    out = capsys.readouterr().out
    assert "leakage gate: OK" in out
    assert "80 %" in out  # the coverage criterion is stated in the report
    assert "rule 'test'" in out  # ... with the rule it was checked under
    assert "exact-substring" in out and "20 aa" in out


def test_gate_still_fails_on_exact_sequence_match(tmp_path):
    train, test = _fixture_seqs()
    leaked = {"905_1_1_1": test[FRAG_TEST]}
    bundle = _make_bundle(tmp_path, leaked, test)
    with pytest.raises(SystemExit, match="exact-sequence"):
        package_release.assert_no_leakage(bundle, rule="test")


# ===================================================== leakage rule (D2)


def _hit(test_cov: float, train_cov: float) -> mm.LeakageHit:
    return mm.LeakageHit(
        train_id="tr",
        test_id="te",
        pident=50.0,
        evalue=1e-10,
        bits=100.0,
        train_cov=train_cov,
        test_cov=test_cov,
        train_len=100,
        test_len=100,
    )


def test_leakage_rules_and_default():
    """Two rules; decision D2 makes "either" the default, because the "test" rule
    leaves training chains that sit inside longer test chains (and so fails the
    release's own substring gate)."""
    assert mm.LEAKAGE_RULES == ("test", "either")
    assert mm.DEFAULT_LEAKAGE_RULE == "either"
    assert mm.SUBSTRING_MIN_LEN == 20


def test_leakage_search_aligns_every_pair_without_the_kmer_prefilter(tmp_path):
    """The k-mer prefilter at -s 7.5 misses weak homologs (30-50 % identity) that
    meet the criterion; --prefilter-mode 2 aligns every training x test pair,
    which costs seconds at this database size."""
    cmd = mm.leakage_search_cmd(
        tmp_path / "train.fa", tmp_path / "test.fa", tmp_path / "hits.tsv", tmp_path
    )
    assert cmd.count("--prefilter-mode") == 1
    assert cmd[cmd.index("--prefilter-mode") + 1] == "2"


@pytest.mark.parametrize(
    "test_cov, train_cov, as_test, as_either",
    [
        (1.0, 0.3, True, True),  # training construct contains the test chain
        (0.5, 1.0, False, True),  # test chain contains the training chain
        (0.8, 0.8, True, True),  # boundary, both sides
        (0.79, 0.79, False, False),  # below on both sides
        (0.79, 0.8, False, True),  # boundary on the training side only
    ],
)
def test_is_leak_per_rule(test_cov, train_cov, as_test, as_either):
    hit = _hit(test_cov, train_cov)
    assert mm.is_leak(hit) is as_either  # default rule ("either", decision D2)
    assert mm.is_leak(hit, rule="test") is as_test
    assert mm.is_leak(hit, rule="either") is as_either


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="leakage rule"):
        mm.is_leak(_hit(1.0, 1.0), rule="both")


def test_leakage_rule_record_roundtrip(tmp_path):
    assert mm.read_leakage_rule(tmp_path) is None
    path = mm.write_leakage_rule(tmp_path, "either")
    assert path == tmp_path / mm.LEAKAGE_RULE_FILE
    rec = json.loads(path.read_text())
    assert rec["rule"] == "either"
    assert rec["min_seq_id"] == 0.3 and rec["min_coverage"] == 0.8
    assert "training chain" in rec["coverage_of"]
    assert mm.read_leakage_rule(tmp_path) == "either"
    with pytest.raises(ValueError, match="leakage rule"):
        mm.write_leakage_rule(tmp_path, "both")


@requires_mmseqs
def test_stage2_either_rule_drops_training_chain_inside_test_chain(fastas, tmp_path):
    """A test chain that CONTAINS a shorter training chain (51677 with
    51674-51676 in the audit): missed by "test", caught by "either"."""
    train_fa, test_fa = fastas
    by_rule = {
        rule: redundancy.stage2_search_removal(
            train_fa,
            test_fa,
            tmp_path / f"{rule}_hits.tsv",
            tmp_path / f"{rule}_tmp",
            tmp_path / f"{rule}.log",
            rule=rule,
        )
        for rule in ("test", "either")
    }
    by_test, by_either = by_rule["test"], by_rule["either"]
    assert FRAG_TRAIN not in by_test and MUTFRAG_TRAIN not in by_test
    assert by_either == by_test | {FRAG_TRAIN, MUTFRAG_TRAIN}
    assert UNRELATED_TRAIN not in by_either


@requires_mmseqs
def test_find_leaks_reports_training_chain_coverage(fastas, tmp_path):
    train_fa, test_fa = fastas
    hits = mm.find_leaks(
        train_fa,
        test_fa,
        tmp_path / "h.tsv",
        tmp_path / "tmp",
        tmp_path / "log",
        rule="either",
    )
    pairs = {(h.train_id, h.test_id): h for h in hits}
    frag = pairs[(FRAG_TRAIN, FRAG_TEST)]
    assert frag.train_cov == pytest.approx(1.0) and frag.test_cov < 0.8
    assert frag.pident == pytest.approx(100.0)
    mut = pairs[(MUTFRAG_TRAIN, FRAG_TEST)]
    assert mut.train_cov >= 0.8 and mut.test_cov < 0.8


# ------------------------------------------------- substring containment


def test_substring_containments_both_directions():
    test = {"T1": "M" + "ACDEFGHIKLMNPQRSTVWYACDEF" + "W" * 10, "T2": "Q" * 40}
    train = {
        "a.fasta:inside_test": "ACDEFGHIKLMNPQRSTVWY",  # 20 aa inside T1
        "a.fasta:too_short": "CDEFGHIKLMNPQRSTVWY",  # 19 aa inside T1
        "a.fasta:contains_test": "GG" + test["T1"] + "GG",  # T1 inside it
        "a.fasta:unrelated": "HHHHHHHHHHHHHHHHHHHHHHHHH",
    }
    found = package_release.find_substring_containments(train, test)
    got = {(c.train_label, c.test_id, c.direction, c.length) for c in found}
    assert got == {
        ("a.fasta:inside_test", "T1", "train in test", 20),
        ("a.fasta:contains_test", "T1", "test in train", len(test["T1"])),
    }


def test_substring_check_threshold_is_configurable():
    test = {"T1": "ACDEFGHIKLMNPQRSTVWYW"}
    train = {"x:short": "ACDEFGHIKL"}  # 10 aa inside T1
    assert package_release.find_substring_containments(train, test) == []
    found = package_release.find_substring_containments(train, test, min_len=10)
    assert [(c.train_label, c.direction) for c in found] == [
        ("x:short", "train in test")
    ]


@requires_mmseqs
def test_gate_fails_on_substring_containment_under_test_rule(tmp_path):
    """The exact half-length fragment passes the test-coverage rule (50 % of
    the test chain) but the gate still fails it: substring containment of
    >= 20 aa is rejected whatever the rule."""
    train, test = _fixture_seqs()
    train = {k: train[k] for k in (FRAG_TRAIN, UNRELATED_TRAIN)}
    bundle = _make_bundle(tmp_path, train, test)
    with pytest.raises(SystemExit) as exc:
        package_release.assert_no_leakage(bundle, rule="test")
    msg = str(exc.value)
    assert "exact-substring containment" in msg
    assert FRAG_TRAIN in msg and FRAG_TEST in msg
    assert "train in test" in msg
    assert "0 (training record, test chain) pairs" in msg  # coverage rule: clean


@requires_mmseqs
def test_gate_rule_either_rejects_what_rule_test_passes(tmp_path, capsys):
    train, test = _fixture_seqs()
    train = {k: train[k] for k in (MUTFRAG_TRAIN, UNRELATED_TRAIN)}
    bundle = _make_bundle(tmp_path, train, test)
    package_release.assert_no_leakage(bundle, rule="test")
    assert "leakage gate: OK" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        package_release.assert_no_leakage(bundle, rule="either")
    msg = str(exc.value)
    assert MUTFRAG_TRAIN in msg and FRAG_TEST in msg
    assert "rule 'either'" in msg
    assert UNRELATED_TRAIN not in msg


# ------------------------------------- package step: the recorded rule


def _make_release_inputs(
    root: Path, train: dict[str, str], test: dict[str, str], rule: str | None
) -> Path:
    """A minimal repo root + work dir that ``package_release.main`` accepts."""
    work = root / "work"
    (root / "docs" / "dataset").mkdir(parents=True)
    (root / "docs" / "dataset" / "bundle-README.md").write_text("# bundle\n")
    chezod_dir = root / "data" / "external" / "chezod117"
    chezod_dir.mkdir(parents=True)
    items = list(test.items())
    write_fasta(dict(items[:1]), chezod_dir / "CheZOD117_test_set.fasta")
    (work / "testset").mkdir(parents=True)
    write_fasta(dict(items[1:]), work / "testset" / "TriZOD_test_set.fasta")
    (work / "testset" / "TriZOD_test_set_labels.tsv").write_text("id\tlabel\n")
    mm_dir = work / "mmseqs"
    mm_dir.mkdir(parents=True)
    clusters = "".join(f"{k}\t{k}\n" for k in train)
    for tier in package_release.TIERS:
        write_fasta(train, mm_dir / f"train_{tier}_best.fasta")
        write_fasta(train, mm_dir / f"train_{tier}.fasta")
        (mm_dir / f"train_{tier}_clu_best.tsv").write_text(clusters)
        (mm_dir / f"train_{tier}_clu.tsv").write_text(clusters)
        scored = root / "data" / "interim" / "scored" / tier
        scored.mkdir(parents=True)
        (scored / "scores.json").write_text('{"ID": "x"}\n')
    if rule is not None:
        mm.write_leakage_rule(mm_dir, rule)
    return work


def _package(root: Path, work: Path, *extra: str) -> Path:
    package_release.main(
        ["--version", "t", "--work-dir", str(work), "--root", str(root), *extra]
    )
    return work / "release_bundle" / "trizod-dataset-t"


@requires_mmseqs
def test_package_gate_checks_the_rule_the_build_recorded(tmp_path):
    train, test = _fixture_seqs()
    clean_for_test = {k: train[k] for k in (MUTFRAG_TRAIN, UNRELATED_TRAIN)}

    root_t = tmp_path / "rule_test"
    work_t = _make_release_inputs(root_t, clean_for_test, test, "test")
    manifest = json.loads((_package(root_t, work_t) / "MANIFEST.json").read_text())
    leak = manifest["leakage"]
    assert leak["rule"] == "test"
    assert leak["recorded_by_build"] is True
    assert leak["substring_gate_min_len"] == 20
    assert leak["gate"] == "passed"

    root_e = tmp_path / "rule_either"
    work_e = _make_release_inputs(root_e, clean_for_test, test, "either")
    with pytest.raises(SystemExit) as exc:
        _package(root_e, work_e)
    assert MUTFRAG_TRAIN in str(exc.value)
    bundle = work_e / "release_bundle" / "trizod-dataset-t"
    assert not (bundle / "MANIFEST.json").exists()  # no manifest for a failed gate


def test_package_refuses_a_rule_other_than_the_recorded_one(tmp_path):
    train, test = _fixture_seqs()
    work = _make_release_inputs(
        tmp_path, {UNRELATED_TRAIN: train[UNRELATED_TRAIN]}, test, "test"
    )
    with pytest.raises(SystemExit, match="recorded"):
        _package(tmp_path, work, "--leakage-rule", "either")


def test_package_requires_a_recorded_rule_or_an_explicit_one(tmp_path):
    train, test = _fixture_seqs()
    work = _make_release_inputs(
        tmp_path, {UNRELATED_TRAIN: train[UNRELATED_TRAIN]}, test, None
    )
    with pytest.raises(SystemExit, match="leakage rule"):
        _package(tmp_path, work)


@requires_mmseqs
def test_package_explicit_rule_without_record_is_marked_unrecorded(tmp_path):
    train, test = _fixture_seqs()
    work = _make_release_inputs(
        tmp_path, {UNRELATED_TRAIN: train[UNRELATED_TRAIN]}, test, None
    )
    bundle = _package(tmp_path, work, "--leakage-rule", "either")
    leak = json.loads((bundle / "MANIFEST.json").read_text())["leakage"]
    assert leak["rule"] == "either" and leak["recorded_by_build"] is False


# ------------------------------------------ redundancy step end to end


def _make_redundancy_inputs(root: Path) -> Path:
    train, test = _fixture_seqs()
    rng = random.Random(7)
    pool = dict(train)
    pool.update({f"95{i}_1_1_1": _rand(rng, 80 + 7 * i) for i in range(6)})
    work = root / "work"
    for tier in redundancy.TIERS:  # every tier holds the whole pool (nested)
        d = work / "final_dataset" / tier
        d.mkdir(parents=True)
        write_fasta(pool, d / f"{tier}.fasta")
    chezod_dir = root / "data" / "external" / "chezod117"
    chezod_dir.mkdir(parents=True)
    items = list(test.items())
    write_fasta(dict(items[:1]), chezod_dir / "CheZOD117_test_set.fasta")
    (work / "testset").mkdir(parents=True)
    write_fasta(dict(items[1:]), work / "testset" / "TriZOD_test_set.fasta")
    return work


@requires_mmseqs
@pytest.mark.parametrize("rule", ["test", "either"])
def test_redundancy_applies_and_records_the_rule(tmp_path, rule):
    work = _make_redundancy_inputs(tmp_path)
    argv = ["--work-dir", str(work), "--root", str(tmp_path)]
    if rule != mm.DEFAULT_LEAKAGE_RULE:
        argv += ["--leakage-rule", rule]
    redundancy.main(argv)
    out = work / "mmseqs"
    assert mm.read_leakage_rule(out) == rule
    kept = set(read_fasta(out / "unfiltered_no_testset.fasta"))
    for leak in (CONTAIN_TRAIN, RATIO_TRAIN, DELETION_TRAIN):
        assert leak not in kept
    inside_test = {FRAG_TRAIN, MUTFRAG_TRAIN}
    if rule == "either":
        assert not inside_test & kept
    else:
        assert inside_test <= kept
    assert UNRELATED_TRAIN in kept


def _hit_lines(tsv: Path) -> set[str]:
    return {ln for ln in tsv.read_text().splitlines() if ln.strip()}


@requires_mmseqs
def test_redundancy_stage2_searches_once_and_matches_per_tier_searches(
    tmp_path, monkeypatch
):
    """Stage 2 runs one easy-search, on the unfiltered superset; each tier's
    hits are that search restricted to the tier's IDs. Checked against a
    separate search of every (nested, strictly smaller) tier FASTA -- what the
    step ran before. The search is the same under both rules (only the parsed
    hits are filtered differently), so one rule suffices."""
    rule = "test"
    work = _make_redundancy_inputs(tmp_path)
    final = work / "final_dataset"
    pool = read_fasta(final / "unfiltered" / "unfiltered.fasta")
    ids = sorted(pool)
    # nested tiers, each leak in a different one
    keep = {
        "tolerant": [i for i in ids if i != CONTAIN_TRAIN],
        "moderate": [i for i in ids if i not in (CONTAIN_TRAIN, FRAG_TRAIN)],
        "strict": [RATIO_TRAIN, MUTFRAG_TRAIN, UNRELATED_TRAIN, ids[-1]],
    }
    for tier, members in keep.items():
        write_fasta({i: pool[i] for i in members}, final / tier / f"{tier}.fasta")

    searches = []
    real_find_leaks = redundancy.find_leaks

    def counting_find_leaks(train_fa, *args, **kwargs):
        searches.append(Path(train_fa).name)
        return real_find_leaks(train_fa, *args, **kwargs)

    monkeypatch.setattr(redundancy, "find_leaks", counting_find_leaks)
    argv = ["--work-dir", str(work), "--root", str(tmp_path), "--leakage-rule", rule]
    redundancy.main(argv)
    monkeypatch.undo()
    assert searches == ["unfiltered.fasta"]

    out = work / "mmseqs"
    combined = out / "combined_testset.fasta"
    unfiltered_removed = set(pool) - set(
        read_fasta(out / "unfiltered_no_testset.fasta")
    )
    for tier in redundancy.TIERS:
        fa = final / tier / f"{tier}.fasta"
        tier_ids = set(read_fasta(fa))
        own = tmp_path / f"own_{tier}_hits.tsv"
        own_hits = {
            h.train_id
            for h in mm.find_leaks(
                fa, combined, own, tmp_path / f"own_{tier}", tmp_path / "log", rule
            )
        }
        assert _hit_lines(out / f"{tier}_testset_hits.tsv") == _hit_lines(own), tier
        kept = set(read_fasta(out / f"{tier}_no_testset.fasta"))
        assert not kept & own_hits, tier
        # removed = (stage-1 | stage-2) restricted to the tier, as before
        assert kept == tier_ids - unfiltered_removed, tier


@requires_mmseqs
def test_redundancy_refuses_a_tier_that_is_not_nested(tmp_path):
    work = _make_redundancy_inputs(tmp_path)
    strict_fa = work / "final_dataset" / "strict" / "strict.fasta"
    write_fasta({"999_1_1_1": AA * 3}, strict_fa)
    with pytest.raises(SystemExit, match="not a subset"):
        redundancy.main(["--work-dir", str(work), "--root", str(tmp_path)])


# ------------------------------------------------------- CLI + script


def test_cli_forwards_leakage_rule(monkeypatch):
    from typer.testing import CliRunner

    from trizod.cli.main import LeakageRule, app

    assert tuple(r.value for r in LeakageRule) == mm.LEAKAGE_RULES
    seen = {}
    monkeypatch.setattr(redundancy, "main", lambda argv: seen.update(red=argv))
    monkeypatch.setattr(package_release, "main", lambda argv: seen.update(pkg=argv))
    runner = CliRunner()
    r = runner.invoke(app, ["dataset", "redundancy", "--leakage-rule", "either"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["dataset", "package", "--leakage-rule", "either"])
    assert r.exit_code == 0, r.output
    for argv in (seen["red"], seen["pkg"]):
        assert argv[argv.index("--leakage-rule") + 1] == "either"
    # the default is not forwarded: redundancy falls back to "test", package
    # to the rule the build recorded
    seen.clear()
    assert runner.invoke(app, ["dataset", "redundancy"]).exit_code == 0
    assert runner.invoke(app, ["dataset", "package"]).exit_code == 0
    assert "--leakage-rule" not in seen["red"] + seen["pkg"]
    r = runner.invoke(app, ["dataset", "redundancy", "--leakage-rule", "both"])
    assert r.exit_code != 0


def _run_build_script(tmp_path: Path, *args: str) -> list[str]:
    """Run a copy of scripts/build_dataset.sh with stub uv/mmseqs that log
    their arguments; returns the logged ``uv`` calls."""
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(repo / "scripts" / "build_dataset.sh", root / "scripts")
    for tier in ("unfiltered", "tolerant", "moderate", "strict"):
        d = root / "data" / "interim" / "scored" / tier
        d.mkdir(parents=True)
        (d / "scores.json").write_text("{}\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "uv.log"
    (bindir / "uv").write_text(f'#!/bin/sh\necho "$*" >> "{log}"\n')
    (bindir / "mmseqs").write_text("#!/bin/sh\necho 18-test\n")
    for f in bindir.iterdir():
        f.chmod(0o755)
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path)}
    subprocess.run(
        ["bash", str(root / "scripts" / "build_dataset.sh"), *args],
        check=True,
        env=env,
        capture_output=True,
    )
    return log.read_text().splitlines()


def test_build_script_passes_leakage_rule_to_redundancy_and_package(tmp_path):
    calls = _run_build_script(tmp_path, "9.9", "--leakage-rule", "either")
    red = [c for c in calls if "dataset redundancy" in c]
    pkg = [c for c in calls if "dataset package" in c]
    assert red == ["run trizod dataset redundancy --leakage-rule either"]
    assert pkg == ["run trizod dataset package --version 9.9 --leakage-rule either"]


def test_build_script_default_rule_is_either(tmp_path):
    calls = _run_build_script(tmp_path)
    red = [c for c in calls if "dataset redundancy" in c]
    assert red == ["run trizod dataset redundancy --leakage-rule either"]


def test_redundancy_removes_search_logs_of_the_old_per_tier_layout(tmp_path):
    # stage 2 now writes one search log; older runs left one per tier
    from trizod.dataset.redundancy import STALE_SEARCH_LOGS, remove_stale_outputs

    for name in STALE_SEARCH_LOGS:
        (tmp_path / name).write_text("old run\n")
    kept = tmp_path / "unfiltered_easy_search.log"
    kept.write_text("current run\n")

    remove_stale_outputs(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [kept.name]
