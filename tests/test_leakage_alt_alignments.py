"""The leakage rule is applied to EVERY reported alignment of a pair.

Release re-audit 2026-11-17b (items E19, E20, E22): the stage-2 search and the
release gate kept mmseqs' best-scoring alignment per pair only, so a second
alignment that meets >= 30 % identity over >= 80 % of either chain at >= 27
bits was never seen. Three training representatives leaked that way
(6280_1_1_1 ~ retired 34124_1_1_1, 4029_1_1_1 ~ test 19298_1_1_1, 51068_1_1_1
~ test 51080_1_1_1; a repeat or a tandem construct). Decision (option a): the
search runs with ``--alt-ali 10`` and ``is_leak`` is applied to every reported
alignment.

The synthetic pair below rebuilds the 51068 ~ 51080 case: an 80-residue
training chain whose middle 30 residues occur verbatim in a 250-residue test
chain (the best alignment, ~75 bits over 38 % of the training chain: no leak),
and elsewhere in the same test chain a full-length copy at 40 % identity (~47
bits over 98 % of the training chain: a leak, but only the second alignment).

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
from trizod.io.fasta import write_fasta

requires_mmseqs = pytest.mark.skipif(
    shutil.which("mmseqs") is None, reason="mmseqs binary not found in PATH"
)

AA = "ACDEFGHIKLMNPQRSTVWY"
TRAIN_ID = "51068_9_9_9"
TEST_ID = "51080_9_9_9"
OTHER_TEST = "1_1_1_1"


def _second_alignment_pair(seed: int = 20261120) -> tuple[str, str]:
    rng = random.Random(seed)

    def rand(n: int) -> str:
        return "".join(rng.choice(AA) for _ in range(n))

    train = rand(80)
    exact = train[25:55]  # 30 aa verbatim: the best alignment, 38 % of train
    # the whole training chain at 40 % identity (positions 0 and 2 of every 5)
    mutated = "".join(
        c if i % 5 in (0, 2) else AA[(AA.index(c) + 7) % len(AA)]
        for i, c in enumerate(train)
    )
    test = rand(40) + exact + rand(60) + mutated + rand(40)
    return train, test


def _row(**kw) -> str:
    return "\t".join(str(kw[c]) for c in mm.LEAKAGE_FORMAT.split(",")) + "\n"


# --------------------------------------------------------------------------- #
# the search options and the rule record
# --------------------------------------------------------------------------- #


def test_the_leakage_search_reports_alternative_alignments(tmp_path):
    assert mm.LEAKAGE_ALT_ALI == 10
    cmd = mm.leakage_search_cmd(
        tmp_path / "a.fa", tmp_path / "b.fa", tmp_path / "h.tsv", tmp_path
    )
    assert cmd[cmd.index("--alt-ali") + 1] == "10"
    # the stage-1 clustering and the test-set draw are unchanged
    assert "--alt-ali" not in mm.COMMON


def test_alternative_alignments_can_be_switched_off_for_a_control():
    assert "--alt-ali" not in mm.leakage_search_options(alt_ali=0)


def test_the_rule_record_says_every_alignment_counts(tmp_path):
    rec = json.loads(mm.write_leakage_rule(tmp_path, "either").read_text())
    assert rec["alt_alignments"] == mm.LEAKAGE_ALT_ALI
    assert "every reported alignment" in rec["alignments"]
    assert "--alt-ali 10" in rec["stage2_search"]
    assert "every reported alignment" in mm.describe_rule("either")


# --------------------------------------------------------------------------- #
# the criterion on canned hits (no mmseqs)
# --------------------------------------------------------------------------- #


def _fake_run(forward_rows: str, reverse_rows: str):
    def run(cmd, log, append=False):
        out = Path(cmd[4])
        out.write_text(forward_rows if "train" in Path(cmd[2]).name else reverse_rows)

    return run


def test_find_leaks_applies_the_rule_to_a_second_alignment(tmp_path, monkeypatch):
    # 51068_1_1_1 ~ 51080_1_1_1 as mmseqs reported it with --alt-ali 10: the
    # best alignment covers 49 % of the training chain, the second 84 %
    best = _row(
        query="R",
        target="T",
        pident=61.5,
        evalue=4.1e-8,
        bits=34,
        qstart=19,
        qend=43,
        qlen=51,
        tstart=212,
        tend=237,
        tlen=252,
    )
    second = _row(
        query="R",
        target="T",
        pident=43.1,
        evalue=1.5e-7,
        bits=32,
        qstart=6,
        qend=48,
        qlen=51,
        tstart=17,
        tend=72,
        tlen=252,
    )
    monkeypatch.setattr(mm, "run", _fake_run(best + second, ""))
    (tmp_path / "train.fa").write_text(">R\n" + "A" * 51 + "\n")
    (tmp_path / "test.fa").write_text(">T\n" + "A" * 252 + "\n")
    hits = mm.find_leaks(
        tmp_path / "train.fa",
        tmp_path / "test.fa",
        tmp_path / "hits.tsv",
        tmp_path / "tmp",
        tmp_path / "log",
    )
    assert [(h.train_id, h.test_id) for h in hits] == [("R", "T")]
    assert hits[0].pident == 43.1 and hits[0].train_cov >= 0.8
    first, *_ = mm.parse_leakage_hits(tmp_path / "hits.tsv")
    assert not mm.is_leak(first)  # the best alignment alone is no leak


# --------------------------------------------------------------------------- #
# a real search (mmseqs)
# --------------------------------------------------------------------------- #


@pytest.fixture
def pair_fastas(tmp_path: Path) -> tuple[Path, Path]:
    train, test = _second_alignment_pair()
    train_fa, test_fa = tmp_path / "train.fasta", tmp_path / "test.fasta"
    write_fasta({TRAIN_ID: train}, train_fa)
    write_fasta({TEST_ID: test}, test_fa)
    return train_fa, test_fa


def _search(query_fa, target_fa, out_tsv, tmp, alt_ali):
    cmd = [
        "mmseqs",
        "easy-search",
        str(query_fa),
        str(target_fa),
        str(out_tsv),
        str(tmp),
        *mm.leakage_search_options(alt_ali=alt_ali),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return mm.parse_leakage_hits(out_tsv)


@requires_mmseqs
def test_the_best_alignment_alone_misses_the_synthetic_pair(pair_fastas, tmp_path):
    """Control: without --alt-ali mmseqs reports one alignment per pair, which
    does not meet the rule in either direction."""
    train_fa, test_fa = pair_fastas
    fwd = _search(train_fa, test_fa, tmp_path / "f.tsv", tmp_path / "tf", 0)
    rev = _search(test_fa, train_fa, tmp_path / "r.tsv", tmp_path / "tr", 0)
    assert len(fwd) == 1 and len(rev) == 1
    assert not any(mm.is_leak(h, "either") for h in fwd + rev)


@requires_mmseqs
def test_only_a_second_alignment_meets_the_rule(pair_fastas, tmp_path):
    train_fa, test_fa = pair_fastas
    hits = mm.find_leaks(
        train_fa, test_fa, tmp_path / "h.tsv", tmp_path / "tmp", tmp_path / "log"
    )
    assert [(h.train_id, h.test_id) for h in hits] == [(TRAIN_ID, TEST_ID)]
    (hit,) = hits
    assert hit.train_cov >= 0.8 and 30 <= hit.pident < 50
    assert hit.bits >= mm.LEAKAGE_MIN_BITS
    # it is not the best alignment mmseqs reports for the pair
    reported = mm.parse_leakage_hits(tmp_path / "h.tsv")
    assert len(reported) > 1
    assert max(h.bits for h in reported) > hit.bits
    assert not mm.is_leak(max(reported, key=lambda h: h.bits))


@requires_mmseqs
def test_stage2_removes_a_training_chain_leaking_on_a_second_alignment(
    pair_fastas, tmp_path
):
    train_fa, test_fa = pair_fastas
    removed = redundancy.stage2_search_removal(
        train_fa, test_fa, tmp_path / "h.tsv", tmp_path / "tmp", tmp_path / "log"
    )
    assert removed == {TRAIN_ID}


@requires_mmseqs
def test_the_gate_fails_on_a_pair_only_a_second_alignment_finds(tmp_path):
    train, test = _second_alignment_pair()
    bundle = tmp_path / "bundle"
    (bundle / "test").mkdir(parents=True)
    write_fasta({OTHER_TEST: AA * 5}, bundle / "test" / "CheZOD117_test_set.fasta")
    write_fasta({TEST_ID: test}, bundle / "test" / "TriZOD_test_set.fasta")
    for tier in ("strict", "tolerant"):
        (bundle / "train" / tier).mkdir(parents=True)
        for name in (f"train_{tier}_best.fasta", f"train_{tier}.fasta"):
            write_fasta({TRAIN_ID: train}, bundle / "train" / tier / name)
    with pytest.raises(SystemExit) as exc:
        package_release.assert_no_leakage(bundle, rule="either")
    assert TRAIN_ID in str(exc.value) and TEST_ID in str(exc.value)
