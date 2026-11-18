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


# --------------------------------------------------------------------------- #
# best training hit per test chain and the nearest-neighbour baseline (E27)
# --------------------------------------------------------------------------- #


def _nrow(cid, split, g, *, tier=None):
    """A row with G-scores ``g`` (None = unscored)."""
    return {
        **_row(cid, split, "A" * len(g), mask=[x is not None for x in g], tier=tier),
        "gscores": [float("nan") if x is None else x for x in g],
    }


def _nhit(test, train, *, bits, evalue, qaln, taln, pident=40.0, qs=1, ts=1):
    return {"test_id": test, "training_id": train, "pident": pident,
            "evalue": evalue, "bits": bits, "qstart": qs, "tstart": ts,
            "test_cov": 0.9, "training_cov": 0.9, "qaln": qaln, "taln": taln}  # fmt: skip


def test_the_label_transfer_follows_the_alignment_through_gaps():
    hit = _nhit("t", "r", bits=30, evalue=1e-4, qaln="AA-AA", taln="AAAA-", qs=2)
    train_g = [0.1, 0.2, None, 0.4]
    # query 2..5 (0-based 1..4), target 1..4: q1-t0, q2-t1, (gap in q: t2),
    # q3-t3, q4-(gap in t)
    assert rs.transfer(hit, train_g, 6) == [None, 0.1, 0.2, 0.4, None, None]


def test_best_hits_take_the_highest_bit_score_and_break_ties_by_id():
    hits = [
        _nhit("1_1_1_1", "20_1_1_1", bits=30, evalue=1e-2, qaln="A", taln="A"),
        _nhit("1_1_1_1", "3_1_1_1", bits=30, evalue=1e-2, qaln="A", taln="A"),
        _nhit("1_1_1_1", "4_1_1_1", bits=25, evalue=1e-4, qaln="A", taln="A"),
    ]
    assert rs.best_hits(hits)["1_1_1_1"]["training_id"] == "3_1_1_1"
    assert rs.best_hits(list(reversed(hits)))["1_1_1_1"]["training_id"] == "3_1_1_1"
    assert rs.best_hits(hits, "evalue")["1_1_1_1"]["training_id"] == "4_1_1_1"
    assert rs.best_hits(hits, "bits")["1_1_1_1"]["training_id"] == "3_1_1_1"
    assert rs.meets({"bits": 27.0, "evalue": 1.0}, "bits")
    assert not rs.meets({"bits": 26.9, "evalue": 1e-3}, "bits")
    assert rs.meets({"bits": 1.0, "evalue": 1e-3}, "evalue")


def test_auc_counts_ties_as_half():
    assert rs.auc([0.1, 0.2, 0.3, 0.4], [False, False, True, True]) == 1.0
    assert rs.auc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5
    assert rs.auc([0.1, 0.2], [True, True]) is None
    assert rs.identity_histogram([10, 20, 29.9, 100])["[20, 30)"] == 2
    assert rs.identity_histogram([100])["[90, 100]"] == 1


def test_nearest_neighbour_table_and_baseline():
    rows = [
        _nrow("1_1_1_1", "test_trizod", [0.9, 0.8, 0.1, None]),
        _nrow("2_1_1_1", "test_trizod", [0.1, 0.2]),
        _nrow("10_1_1_1", "train", [0.7, 0.9, 0.0, 0.0], tier="tolerant"),
        _nrow("11_1_1_1", "train", [0.2, 0.2], tier="moderate"),
        _nrow("12_1_1_1", "train", [0.5, 0.5], tier="unfiltered"),  # not in db
    ]
    hits = [
        # 1: a significant hit to 10 (both criteria)
        _nhit("1_1_1_1", "10_1_1_1", bits=40, evalue=1e-6, qaln="AAAA", taln="AAAA"),
        # 2: a weak hit to 11 (>= 27 bits, but E > 1e-3)
        _nhit("2_1_1_1", "11_1_1_1", bits=28, evalue=0.01, qaln="AA", taln="AA"),
        # an unfiltered-only training chain is not a representative searched
        _nhit("2_1_1_1", "12_1_1_1", bits=90, evalue=1e-20, qaln="AA", taln="AA"),
    ]
    res = rs.nearest_neighbour(rows, hits, remote={"1_1_1_1"})
    table = {r["test_id"]: r for r in res["table"]}
    assert list(table) == ["1_1_1_1", "2_1_1_1"]
    assert table["1_1_1_1"]["best_training_id"] == "10_1_1_1"
    assert table["2_1_1_1"]["best_training_id"] == "11_1_1_1"
    assert table["1_1_1_1"]["meets_evalue_1e-3"] == "true"
    assert table["2_1_1_1"]["meets_evalue_1e-3"] == "false"
    assert table["2_1_1_1"]["meets_27_bits"] == "true"
    assert table["1_1_1_1"]["transferred_residues_evalue_1e-3"] == 3
    assert table["2_1_1_1"]["transferred_residues_evalue_1e-3"] == 0
    assert table["2_1_1_1"]["transferred_residues_27_bits"] == 2
    assert table["1_1_1_1"]["remote_homolog_profile"] == "true"
    s = res["summary"]
    assert (s["test_chains"], s["training_representatives"]) == (2, 2)
    mean = (0.7 + 0.9 + 0.0 + 0.0 + 0.2 + 0.2) / 6
    assert s["baseline"]["train_mean_G"] == round(mean, 4)
    ev = s["baseline"]["evalue"]
    assert ev["hit_chains"] == 1 and s["baseline"]["bits"]["hit_chains"] == 2
    assert ev["all_residues"]["residues"] == 5
    assert ev["transferred_residues"]["residues"] == 3
    assert ev["transferred_share"] == 0.6
    # E <= 1e-3: 1 gets 0.7, 0.9, 0.0 (truth 0.9, 0.8, 0.1), 2 gets the mean
    # 0.333 (truth 0.1, 0.2): the two disordered residues rank highest
    assert ev["all_residues"]["auc_G>=0.35"] == 1.0
    assert ev["without_remote_homolog_chains"]["chains"] == 1
    text = rs.nn_text(s)
    assert "2 of the 2 test chains align" in text
    assert "1 chains with a hit" in text and "≥ 27 bits (2 chains" in text
    assert "`test/nearest_neighbour_baseline.json`" in text


def test_the_residual_similarity_search_has_no_floor(tmp_path):
    cmd = rs.nn_search_cmd("q.fa", "t.fa", "hits.tsv", "tmp")
    assert cmd[cmd.index("--min-seq-id") + 1] == "0"
    assert cmd[cmd.index("-c") + 1] == "0"
    assert cmd[cmd.index("--prefilter-mode") + 1] == "2"
    assert float(cmd[cmd.index("-e") + 1]) == 10
    fmt = cmd[cmd.index("--format-output") + 1].split(",")
    assert fmt[-2:] == ["qaln", "taln"]
    tsv = tmp_path / "hits.tsv"
    tsv.write_text("1_1_1_1\t10_1_1_1\t50.0\t1e-05\t40\t1\t10\t20\t3\t12\t10\tAC\tAC\n")
    (h,) = rs.parse_nn_hits(tsv)
    assert (h["test_cov"], h["training_cov"]) == (0.5, 1.0)
    assert (h["qaln"], h["taln"], h["qstart"], h["tstart"]) == ("AC", "AC", 1, 3)


@requires_mmseqs
def test_the_residual_similarity_search_runs(tmp_path):
    rng = random.Random(3)
    a = _rand(60, rng)
    rows = [
        _nrow("1_1_1_1", "test_trizod", [0.5] * 60),
        _nrow("10_1_1_1", "train", [0.2] * 60, tier="tolerant"),
    ]
    rows[0]["sequence"] = a
    rows[1]["sequence"] = a[:30] + _rand(30, rng)
    hits = rs.nn_hits(rows)
    assert hits and all(h["training_id"] == "10_1_1_1" for h in hits)
    assert all(len(h["qaln"]) == len(h["taln"]) for h in hits)
