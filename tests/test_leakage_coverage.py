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
* fragment      - a training chain covering only half of a test chain (no leak)
* unrelated     - a random training chain (no leak)

mmseqs-dependent tests skip cleanly when the binary is not on PATH.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import pytest

from trizod.dataset import mmseqs as mm
from trizod.dataset import package_release, redundancy
from trizod.io.fasta import write_fasta

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


def _rand(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(AA) for _ in range(n))


def _substitute(seq: str, positions: list[int]) -> str:
    """Replace each listed position with a different residue."""
    s = list(seq)
    for p in positions:
        s[p] = AA[(AA.index(s[p]) + 7) % len(AA)]
    return "".join(s)


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
    assert len(train[RATIO_TRAIN]) == 124 and len(test[RATIO_TEST]) == 99
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
    assert cmd[cmd.index("-c") + 1] == "0.8"
    assert cmd[cmd.index("--min-seq-id") + 1] == "0.3"
    # the rest of the shared option block is unchanged
    for flag, val in [
        ("--alignment-mode", "3"),
        ("-s", "7.5"),
        ("--comp-bias-corr", "0"),
        ("--mask", "0"),
    ]:
        assert cmd[cmd.index(flag) + 1] == val
    assert "tcov" in cmd[cmd.index("--format-output") + 1].split(",")


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


# --------------------------------------------------------------- stage 2


@requires_mmseqs
def test_leak_search_reports_test_chain_coverage(fastas, tmp_path):
    train_fa, test_fa = fastas
    hits = mm.find_test_coverage_leaks(
        train_fa, test_fa, tmp_path / "hits.tsv", tmp_path / "tmp", tmp_path / "log"
    )
    pairs = {(h.train_id, h.test_id): h for h in hits}
    assert set(pairs) == {(CONTAIN_TRAIN, CONTAIN_TEST), (RATIO_TRAIN, RATIO_TEST)}
    contain = pairs[(CONTAIN_TRAIN, CONTAIN_TEST)]
    assert contain.test_cov >= 0.99 and contain.train_cov < 0.5
    assert contain.pident == pytest.approx(100.0)
    ratio = pairs[(RATIO_TRAIN, RATIO_TEST)]
    assert ratio.test_cov >= 0.8
    assert 30.0 <= ratio.pident < 80.0
    assert (ratio.train_len, ratio.test_len) == (124, 99)


@requires_mmseqs
def test_stage2_drops_containment_and_length_ratio_leaks(fastas, tmp_path):
    train_fa, test_fa = fastas
    removed = redundancy.stage2_search_removal(
        train_fa, test_fa, tmp_path / "hits.tsv", tmp_path / "tmp", tmp_path / "log"
    )
    assert removed == {CONTAIN_TRAIN, RATIO_TRAIN}
    # the criterion is coverage of the TEST chain: a training fragment that
    # covers half of a test chain is not a leak, nor is an unrelated chain
    assert FRAG_TRAIN not in removed
    assert UNRELATED_TRAIN not in removed


# ------------------------------------------------------- packaging gate


@requires_mmseqs
def test_gate_fails_on_containment_and_reports_pairs(tmp_path):
    train, test = _fixture_seqs()
    bundle = _make_bundle(tmp_path, train, test)
    with pytest.raises(SystemExit) as exc:
        package_release.assert_no_leakage(bundle)
    msg = str(exc.value)
    assert "LEAKAGE GATE FAILED" in msg
    for tid in (CONTAIN_TRAIN, CONTAIN_TEST, RATIO_TRAIN, RATIO_TEST):
        assert tid in msg
    assert FRAG_TRAIN not in msg and UNRELATED_TRAIN not in msg


@requires_mmseqs
def test_gate_fails_on_length_ratio_pair_alone(tmp_path):
    train, test = _fixture_seqs()
    train = {k: v for k, v in train.items() if k != CONTAIN_TRAIN}
    bundle = _make_bundle(tmp_path, train, test)
    with pytest.raises(SystemExit) as exc:
        package_release.assert_no_leakage(bundle)
    assert RATIO_TRAIN in str(exc.value) and RATIO_TEST in str(exc.value)


@requires_mmseqs
def test_gate_passes_clean_bundle_and_leaves_it_untouched(tmp_path, capsys):
    train, test = _fixture_seqs()
    clean = {k: train[k] for k in (FRAG_TRAIN, UNRELATED_TRAIN)}
    bundle = _make_bundle(tmp_path, clean, test)
    before = sorted(p.relative_to(bundle) for p in bundle.rglob("*"))
    package_release.assert_no_leakage(bundle)
    after = sorted(p.relative_to(bundle) for p in bundle.rglob("*"))
    assert before == after  # no mmseqs scratch left inside the bundle
    out = capsys.readouterr().out
    assert "leakage gate: OK" in out
    assert "80 %" in out  # the coverage criterion is stated in the report


def test_gate_still_fails_on_exact_sequence_match(tmp_path):
    train, test = _fixture_seqs()
    leaked = {"904_1_1_1": test[FRAG_TEST]}
    bundle = _make_bundle(tmp_path, leaked, test)
    with pytest.raises(SystemExit, match="exact-sequence"):
        package_release.assert_no_leakage(bundle)
