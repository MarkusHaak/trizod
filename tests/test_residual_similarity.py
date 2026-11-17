"""What remains below the leakage rule, documented in the deposit (release
validation 2026-11-17c, E22, E24, E25, E26): :mod:`trizod.dataset.residual_similarity`."""

from __future__ import annotations

import random
import shutil

import pytest

from trizod.dataset import mmseqs as mm
from trizod.dataset import residual_similarity as rs
from trizod.dataset import shared_constructs as sc

requires_mmseqs = pytest.mark.skipif(
    shutil.which("mmseqs") is None, reason="mmseqs binary not found in PATH"
)
AA = "ACDEFGHIKLMNPQRSTVWY"


def _row(cid, split, seq, *, mask=None, tier=None, acc=(), src=None, reason=None):
    return {
        "id": cid,
        "split": split,
        "sequence": seq,
        "mask": mask if mask is not None else [True] * len(seq),
        "train_tier": tier,
        "uniprot_ids": list(acc),
        "uniprot_source": src,
        "retired_reason": reason,
    }


def _rand(n, rng):
    return "".join(rng.choice(AA) for _ in range(n))


# --------------------------------------------------------------------------- #
# test components (E24)
# --------------------------------------------------------------------------- #


def test_components_are_named_after_their_smallest_member():
    comp = rs.components(
        ["10_1_1_1", "2_1_1_1", "3_1_1_1", "4_1_1_1"],
        [("10_1_1_1", "3_1_1_1"), ("3_1_1_1", "2_1_1_1"), ("9_1_1_1", "4_1_1_1")],
    )
    assert comp == {
        "10_1_1_1": "2_1_1_1",
        "2_1_1_1": "2_1_1_1",
        "3_1_1_1": "2_1_1_1",
        "4_1_1_1": "4_1_1_1",
    }


@requires_mmseqs
def test_the_test_set_components_follow_the_leakage_rule(tmp_path):
    rng = random.Random(7)
    founder = _rand(90, rng)
    variant = "".join(
        c if i % 3 else AA[(AA.index(c) + 5) % 20] for i, c in enumerate(founder)
    )  # 67 % identical over the whole chain: one component
    rows = [
        _row("1_1_1_1", "test_trizod", founder),
        _row("2_1_1_1", "test_trizod", variant),
        _row("3_1_1_1", "test_trizod", _rand(90, rng)),
        _row("4_1_1_1", "train", founder),  # training: not a test chain
    ]
    res = rs.test_components(rows)
    assert res["ids"] == {
        "1_1_1_1": "1_1_1_1",
        "2_1_1_1": "1_1_1_1",
        "3_1_1_1": "3_1_1_1",
    }
    assert (res["n"], res["singletons"], res["largest"]) == (2, 1, 2)
    path = rs.write_components(res, tmp_path / "c.tsv")
    assert path.read_text().splitlines() == [
        "id\tcomponent\tcomponent_size",
        "1_1_1_1\t1_1_1_1\t2",
        "2_1_1_1\t1_1_1_1\t2",
        "3_1_1_1\t3_1_1_1\t1",
    ]
    assert "2 homology components" in rs.components_text(res, 3)


# --------------------------------------------------------------------------- #
# shared accessions (E22)
# --------------------------------------------------------------------------- #


def _hit(test, train, pident, bits, test_cov, train_cov):
    return mm.LeakageHit(
        train_id=train,
        test_id=test,
        pident=pident,
        evalue=1e-3,
        bits=bits,
        train_cov=train_cov,
        test_cov=test_cov,
        train_len=100,
        test_len=100,
    )


def test_shared_accessions_record_run_alignment_and_decision(monkeypatch):
    rng = random.Random(3)
    shared = _rand(37, rng)
    rows = [
        _row("52143_1_1_1", "test_trizod", shared + _rand(200, rng),
             acc=["O00571"], src="sifts"),
        _row("53627_1_1_1", "train", _rand(130, rng) + shared, acc=["O00571"],
             src="author"),
        _row("1_1_1_1", "test_trizod", _rand(80, rng), acc=["P1"], src="sifts"),
        _row("2_1_1_1", "redundant", _rand(80, rng), acc=["P1", "P2"], src="blast"),
        _row("3_1_1_1", "excluded", _rand(80, rng), acc=["P1"], src="sifts"),
    ]  # fmt: skip
    canned = {
        ("52143_1_1_1", "53627_1_1_1"): [_hit("52143_1_1_1", "53627_1_1_1",
                                              100.0, 83.2, 0.133, 0.22)],
        ("1_1_1_1", "2_1_1_1"): [_hit("1_1_1_1", "2_1_1_1", 28.0, 30, 0.9, 0.9)],
    }  # fmt: skip
    monkeypatch.setattr(rs, "_align_pairs", lambda pairs: canned)
    table = rs.shared_accessions(rows)
    got = {(r["test_id"], r["training_id"]): r for r in table}
    assert set(got) == {("1_1_1_1", "2_1_1_1"), ("52143_1_1_1", "53627_1_1_1")}
    over = got[("52143_1_1_1", "53627_1_1_1")]
    assert over["identical_run"] >= 37 and over["meets_rule"] == "false"
    assert over["decision"].startswith("kept: overlapping construct")
    below = got[("1_1_1_1", "2_1_1_1")]
    assert below["decision"] == "kept: below the leakage rule"  # 28 % < 30 %
    s = rs.shared_accession_summary(table)
    assert s["pairs"] == 2 and s["pairs_own_accession_both"] == 1
    assert s["pairs_meeting_rule"] == 0
    assert "52143_1_1_1~53627_1_1_1" in s["overlapping_constructs"][0]
    rows_ki = rs.shared_accession_rows(table)
    assert [c for c, i, _ in rows_ki] == ["1_1_1_1", "52143_1_1_1"]
    assert all(i == "shared_uniprot_accession" for _, i, _ in rows_ki)


def test_a_shared_accession_pair_meeting_the_rule_is_reported(monkeypatch):
    rows = [
        _row("1_1_1_1", "test_trizod", "A" * 50, acc=["P1"], src="sifts"),
        _row("2_1_1_1", "train", "C" * 50, acc=["P1"], src="sifts"),
    ]
    canned = {("1_1_1_1", "2_1_1_1"): [_hit("1_1_1_1", "2_1_1_1", 40, 40, 0.9, 0.9)]}
    monkeypatch.setattr(rs, "_align_pairs", lambda pairs: canned)
    (rec,) = rs.shared_accessions(rows)
    assert rec["meets_rule"] == "true" and "gate defect" in rec["decision"]


# --------------------------------------------------------------------------- #
# remote homologues (E25)
# --------------------------------------------------------------------------- #


def _phit(test, train, pident, test_cov, train_cov, bits=40):
    return {"test_id": test, "training_id": train, "pident": pident,
            "evalue": 1e-5, "bits": bits, "test_cov": test_cov,
            "training_cov": train_cov}  # fmt: skip


def test_remote_homologs_count_test_chains_and_scored_residues():
    rows = [
        _row("1_1_1_1", "test_trizod", "A" * 10, mask=[True] * 8 + [False] * 2),
        _row("2_1_1_1", "test_trizod", "A" * 10),
        _row("3_1_1_1", "test_trizod", "A" * 10),
        _row("10_1_1_1", "train", "A" * 10, tier="tolerant"),
        _row("11_1_1_1", "train", "A" * 10, tier="unfiltered"),
    ]
    hits = [
        _phit("1_1_1_1", "10_1_1_1", 25.0, 0.85, 0.5),  # a hit
        _phit("1_1_1_1", "10_1_1_1", 30.0, 0.9, 0.9, bits=60),  # the better one
        _phit("2_1_1_1", "10_1_1_1", 20.0, 0.9, 0.9),  # not > 20 %
        _phit("3_1_1_1", "11_1_1_1", 40.0, 0.9, 0.9),  # unfiltered rep: not in db
    ]
    res = rs.remote_homologs(rows, hits)
    assert list(res["best"]) == ["1_1_1_1"]
    assert res["best"]["1_1_1_1"]["bits"] == 60
    assert (res["n_test"], res["n_scored"], res["hit_scored"]) == (3, 28, 8)
    assert "1 of the 3 test chains (8 of their 28 scored residues, 28.6 %)" in (
        rs.remote_homolog_text(res)
    )
    ((cid, issue, ev),) = rs.remote_homolog_rows(res)
    assert (cid, issue) == ("1_1_1_1", "remote_homolog_profile")
    assert "10_1_1_1" in ev


def test_the_profile_search_is_seths_three_iteration_search(tmp_path):
    cmds = rs.profile_search_cmds(tmp_path / "q.fa", tmp_path / "t.fa", tmp_path)
    search = next(c for c in cmds if c[1] == "search")
    assert search[search.index("--num-iterations") + 1] == "3"
    assert search[search.index("-s") + 1] == "7.5" and "-a" in search
    assert float(search[search.index("-e") + 1]) == 1e-3


# --------------------------------------------------------------------------- #
# the shared-construct review as known issues (E26)
# --------------------------------------------------------------------------- #


def test_shared_construct_issues_list_every_reviewed_test_chain():
    rng = random.Random(5)
    shared = _rand(40, rng)
    t1, r1 = shared + _rand(10, rng), "MHHHHHH" + shared
    t2, r2 = _rand(60, rng), _rand(60, rng)
    rows = [
        _row("1_1_1_1", "test_retired", t1, reason="shared_construct"),
        _row("2_1_1_1", "train", r1),
        _row("3_1_1_1", "test_trizod", t2),
        _row("4_1_1_1", "train", r2),
    ]
    review = [
        sc.review_pair("1_1_1_1", t1, [True] * 50, "2_1_1_1", r1, "identical_run").row(),
        sc.review_pair("3_1_1_1", t2, [True] * 60, "4_1_1_1", r2,
                       "tag_stripped_homology").row(),
    ]  # fmt: skip
    got = rs.shared_construct_issues(review, rows)
    assert [(c, i) for c, i, _ in got["rows"]] == [
        ("1_1_1_1", "shared_construct_retired"),
        ("3_1_1_1", "shared_construct_kept"),
    ]
    assert "retired 1 (1_1_1_1), kept 1" in got["text"]
    rows[0]["split"] = "test_trizod"  # the release did not apply the retirement
    with pytest.raises(SystemExit, match="does not match the release"):
        rs.shared_construct_issues(review, rows)
