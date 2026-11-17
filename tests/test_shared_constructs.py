"""The E26 criterion: a test chain that shares a construct with training once
expression tags are set aside (release validation 2026-11-17c).

A pinned test chain is retired as ``shared_construct`` when, after removing the
expression tags of both chains, an identical run of >= 30 residues with a
training chain covers >= 50 % of the test chain's scored residues.
"""

from __future__ import annotations

import random

import pytest

from trizod.dataset import shared_constructs as sc
from trizod.dataset import testset

AA = "ACDEFGHIKLMNPQRSTVWY"


def _rand(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ADEFIKLMNQRSTVWY") for _ in range(n))


# --------------------------------------------------------------------------- #
# tag stripping
# --------------------------------------------------------------------------- #


def test_riken_flanks_are_stripped_and_positions_kept():
    core = _rand(40, 1)
    s = sc.strip_expression_tags(sc.RIKEN_N + core + sc.RIKEN_C)
    assert s.seq == core
    assert s.tags == ("RIKEN-N", "RIKEN-C")
    assert s.positions == tuple(range(7, 47))


def test_an_n_terminal_his_tag_and_thrombin_site_are_stripped():
    core = _rand(50, 2)
    s = sc.strip_expression_tags("MGSSHHHHHHSSGLVPRGSH" + core)
    assert s.seq == "H" + core
    assert s.tags == ("N-His", "N-thrombin")


def test_a_c_terminal_his_tag_and_its_linker_are_stripped():
    core = _rand(50, 3)
    s = sc.strip_expression_tags(core + "LEHHHHHH")
    assert s.seq == core and s.tags == ("C-His",)


def test_a_gb1_fusion_is_stripped():
    # pinned test chain 52047_1_1_1: GB1 ("KLALNGK" variant) + N-myc TAD
    seq = (
        "GAMAQYKLALNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTETSDLEFDSLQPSF"
        "YPDEDDFYFGGPDSTPPGEDIWKKFELLPTP"
    )
    s = sc.strip_expression_tags(seq)
    assert "GB1" in s.tags
    # from 4 residues before the first GB1 strand to the end of its last
    assert s.seq == "GA" + "TSDLEFDSLQPSFYPDEDDFYFGGPDSTPPGEDIWKKFELLPTP"
    assert all(seq[p] == c for p, c in zip(s.positions, s.seq))


def test_an_untagged_sequence_is_unchanged():
    core = _rand(60, 4)
    s = sc.strip_expression_tags(core)
    assert s.seq == core and s.tags == () and s.positions == tuple(range(60))


def test_longest_identical_run():
    assert sc.longest_identical_run("XXABCDEYY", "QABCDEQ") == (5, 2, 1)
    assert sc.longest_identical_run("AAAA", "CCCC") == (0, 0, 0)


# --------------------------------------------------------------------------- #
# the criterion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("run", "coverage", "retire"),
    [(30, 0.5, True), (83, 0.519, True), (29, 0.9, False), (37, 0.137, False)],
)
def test_the_criterion(run, coverage, retire):
    assert sc.RUN_MIN == 30 and sc.COVERAGE_MIN == 0.5
    assert sc.meets_criterion(run, coverage) is retire


def test_coverage_counts_scored_residues_of_the_deposited_test_chain():
    shared = _rand(40, 5)
    test_seq = sc.RIKEN_N + shared + _rand(30, 6) + sc.RIKEN_C
    train_seq = "MHHHHHHLVPRGS" + shared + _rand(20, 7)
    # 60 scored residues, 30 of them inside the shared 40-residue run
    mask = [False] * len(test_seq)
    for i in range(7, 37):
        mask[i] = True
    for i in range(47, 77):
        mask[i] = True
    rv = sc.review_pair("T", test_seq, mask, "R", train_seq, "identical_run")
    assert rv.run >= 40 and rv.test_scored == 60 and rv.scored_in_run == 30
    assert rv.coverage == pytest.approx(0.5) and rv.outcome == "retire"
    assert rv.test_tags == "RIKEN-N,RIKEN-C" and rv.training_tags == "N-His,N-thrombin"
    mask[7] = False  # 29 of 59: below half
    assert sc.review_pair("T", test_seq, mask, "R", train_seq, "x").outcome == "keep"


def test_the_scan_finds_every_run_hidden_behind_tags():
    shared = _rand(35, 8)
    test = {
        "T1": (sc.RIKEN_N + shared + sc.RIKEN_C, [True] * 48),
        "T2": (_rand(80, 9), [True] * 80),
    }
    training = {"R1": "MHHHHHHENLYFQG" + shared, "R2": _rand(90, 10)}
    rows = sc.review(test, training, extra_pairs=[("T2", "R2"), ("T9", "R1")])
    got = {(r.test_id, r.training_id): (r.source, r.outcome) for r in rows}
    assert got == {
        ("T1", "R1"): ("identical_run", "retire"),
        ("T2", "R2"): ("tag_stripped_homology", "keep"),
    }


# --------------------------------------------------------------------------- #
# the committed review
# --------------------------------------------------------------------------- #


def test_the_committed_review_is_consistent_with_the_retirements():
    rows = sc.read_review()
    assert rows, "the committed review is empty"
    retired = {
        r.id: r.reason
        for r in testset.load_test_retirements(testset.TEST_RETIREMENTS_TSV)
    }
    for r in rows:
        retire = sc.meets_criterion(int(r["run"]), float(r["coverage"]))
        assert (r["outcome"] == "retire") is retire, r
        if retire:
            assert retired.get(r["test_id"]) == "shared_construct", r["test_id"]
    s = sc.summary(rows)
    # decision 2026-11-17: 7070_1_1_1 (83-residue run with 16243_1_1_1, 51.9 %
    # of its scored residues) is retired; no other reviewed pair meets it
    assert s["retired"] == ["7070_1_1_1"]
    pair = {(r["test_id"], r["training_id"]): r for r in rows}
    assert pair[("7070_1_1_1", "16243_1_1_1")]["run"] == "83"
    assert s["tag_stripped_homology"]["pairs"] >= 30


def _release_row(cid, seq, split, mask=None, reason=None):
    return {
        "id": cid,
        "sequence": seq,
        "mask": mask or [True] * len(seq),
        "split": split,
        "retired_reason": reason,
    }


def test_check_review_flags_a_stale_row():
    shared = _rand(40, 11)
    test_seq, train_seq = shared + _rand(10, 12), "MHHHHHH" + shared
    rv = sc.review_pair("T", test_seq, [True] * 50, "R", train_seq, "identical_run")
    rows = [rv.row()]
    good = [
        _release_row("T", test_seq, "test_retired", reason="shared_construct"),
        _release_row("R", train_seq, "train"),
    ]
    assert sc.check_review(rows, good) == []
    still_test = [_release_row("T", test_seq, "test_trizod"), good[1]]
    assert any("reviewed retire" in p for p in sc.check_review(rows, still_test))
    other_seq = [good[0], _release_row("R", _rand(60, 13), "train")]
    assert any("differs" in p for p in sc.check_review(rows, other_seq))
