"""Test chains retired for a CheZOD117 overlap or as a within-test duplicate
(release audit E23/E24, 2026-11-17).

The TriZOD test set only shrinks: a pinned chain that must not be tested on is
RETIRED -- written to ``TriZOD_test_retired.fasta`` with a ``reason``, still a
leakage target, never deleted. Besides ``not_strict`` (left the strict tier)
there are now two reasons:

* ``chezod117_overlap`` -- the chain meets the leakage criterion (>= 30 %
  identity over >= 80 % of either chain, >= 27 bits, either search direction)
  against a CheZOD117 sequence, released or original: 19481_1_1_1 contains
  CheZOD117 19482 as an exact 91-residue substring, so the two test sets
  overlapped.
* ``within_test_duplicate`` -- after that, the test set is checked against
  itself at 50 % identity over 80 % of EITHER chain (the release's coverage
  everywhere else); of each such pair the shorter chain (the contained one, for
  6439_1_1_1 inside 6440_1_1_1) is retired, the lower entry ID kept on a tie.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest

from trizod.dataset import testset
from trizod.io.fasta import read_fasta, write_fasta

requires_mmseqs = pytest.mark.skipif(
    shutil.which("mmseqs") is None, reason="mmseqs binary not found in PATH"
)

AA = "ACDEFGHIKLMNPQRSTVWY"


def _rand(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(AA) for _ in range(n))


def _mutate(seq: str, every: int) -> str:
    """Substitute every ``every``-th residue (identity 1 - 1/every)."""
    s = list(seq)
    for i in range(0, len(s), every):
        s[i] = AA[(AA.index(s[i]) + 7) % len(AA)]
    return "".join(s)


# --------------------------------------------------------------------------- #
# which chain of a near-duplicate pair is retired
# --------------------------------------------------------------------------- #


def test_the_shorter_chain_of_a_pair_is_retired():
    recs = {"6440_1_1_1": "M" * 258, "6439_1_1_1": "M" * 117}
    retired = testset.select_within_test_duplicates(
        recs, [("6439_1_1_1", "6440_1_1_1")]
    )
    assert retired == {"6439_1_1_1": "6440_1_1_1"}


def test_on_equal_length_the_higher_entry_id_is_retired():
    recs = {"51949_1_1_1": "M" * 136, "15603_1_1_1": "M" * 136}
    retired = testset.select_within_test_duplicates(
        recs, [("51949_1_1_1", "15603_1_1_1")]
    )
    assert retired == {"51949_1_1_1": "15603_1_1_1"}


def test_a_chain_is_retired_only_against_a_kept_partner():
    # A ~ B and B ~ C, but A !~ C: retiring B leaves A and C both test chains
    recs = {"1_1_1_1": "M" * 200, "2_1_1_1": "M" * 150, "3_1_1_1": "M" * 100}
    pairs = [("2_1_1_1", "1_1_1_1"), ("3_1_1_1", "2_1_1_1")]
    assert testset.select_within_test_duplicates(recs, pairs) == {"2_1_1_1": "1_1_1_1"}


def test_the_selection_does_not_depend_on_input_order():
    rng = random.Random(24)
    ids = [f"{n}_1_1_1" for n in range(1, 13)]
    recs = {i: "M" * rng.choice([90, 100, 110]) for i in ids}
    pairs = [(a, b) for a in ids for b in ids if a < b and rng.random() < 0.3]
    expected = testset.select_within_test_duplicates(recs, pairs)
    for _ in range(5):
        shuffled = dict(rng.sample(list(recs.items()), len(recs)))
        flipped = [(b, a) if rng.random() < 0.5 else (a, b) for a, b in pairs]
        rng.shuffle(flipped)
        assert testset.select_within_test_duplicates(shuffled, flipped) == expected
    kept = set(recs) - set(expected)
    assert not any(a in kept and b in kept for a, b in pairs)


def test_retire_chains_moves_rows_and_records_the_reason():
    pinned = {"1_1_1_1": "AAAA", "2_1_1_1": "CCCC"}
    pool = dict(pinned)
    recs, info = testset.resolve_pinned_testset(
        pinned, pool, tier_pools={"strict": pool}
    )
    retired_recs, retired_info = {}, {"id_map": [], "dropped": []}
    testset.retire_chains(
        recs,
        info,
        retired_recs,
        retired_info,
        {"2_1_1_1"},
        testset.CHEZOD117_OVERLAP,
    )
    assert recs == {"1_1_1_1": "AAAA"} and retired_recs == {"2_1_1_1": "CCCC"}
    assert [r["test_id"] for r in info["id_map"]] == ["1_1_1_1"]
    (row,) = retired_info["id_map"]
    assert row["reason"] == "chezod117_overlap" and row["label_tier"] == "strict"
    assert info["label_tier_counts"] == {"strict": 1}


# --------------------------------------------------------------------------- #
# the test-set step end to end (mmseqs)
# --------------------------------------------------------------------------- #

OVERLAP_RELEASED = "19481_1_1_1"  # contains a released CheZOD117 chain
OVERLAP_ORIGINAL = "34813_1_1_1"  # a homologue of an original-only sequence
LONG, SHORT = "6440_1_1_1", "6439_1_1_1"  # SHORT is a substring of LONG
NEAR_A, NEAR_B = "15603_1_1_1", "51949_1_1_1"  # ~88 % identity, 136 vs 168 aa
DISTANT = "7000_1_1_1"  # ~45 % identity to NEAR_A: no duplicate at 50 %
KEPT = "100_1_1_1"


def _fixture():
    rng = random.Random(20261117)
    chezod_released = _rand(rng, 91)
    hmbdi = _rand(rng, 82)
    long_chain = _rand(rng, 258)
    near = _rand(rng, 136)
    test = {
        KEPT: _rand(rng, 120),
        OVERLAP_RELEASED: _rand(rng, 30) + chezod_released + _rand(rng, 38),
        OVERLAP_ORIGINAL: "G" + _mutate(hmbdi, 10),
        LONG: long_chain,
        SHORT: long_chain[60:177],
        NEAR_A: near,
        NEAR_B: _mutate(near, 8) + _rand(rng, 32),
        DISTANT: "".join(
            c if i % 9 in (0, 1, 2, 3) else AA[(AA.index(c) + 5) % 20]
            for i, c in enumerate(near)
        ),
    }
    released = {"19482_1_1_1": chezod_released}
    original = {"hmbdi": hmbdi, "19482": chezod_released[2:]}
    return test, released, original


def _setup(tmp_path: Path):
    test, released, original = _fixture()
    root, wd = tmp_path / "root", tmp_path / "wd"
    pin = root / "trizod" / "dataset" / "pinned" / "TriZOD_test_set.fasta"
    pin.parent.mkdir(parents=True)
    write_fasta(test, pin)
    for tier in ("strict", "moderate", "tolerant"):
        d = wd / "final_dataset" / tier
        d.mkdir(parents=True)
        write_fasta(test, d / f"{tier}.fasta")
    ext = root / "data" / "external"
    (ext / "chezod117").mkdir(parents=True)
    write_fasta(released, ext / "chezod117" / "CheZOD117_test_set.fasta")
    (ext / "chezod" / "seth_github").mkdir(parents=True)
    write_fasta(
        original, ext / "chezod" / "seth_github" / "CheZOD117_test_set_sequences.fasta"
    )
    return root, wd, test


@pytest.fixture
def built(tmp_path):
    root, wd, test = _setup(tmp_path)
    testset.main(["--work-dir", str(wd), "--root", str(root)])
    return wd / "testset", test


def _reasons(out: Path) -> dict[str, str]:
    reasons = {}
    for line in (out / testset.RETIRED_FASTA).open():
        if line.startswith(">"):
            tokens = line[1:].split()
            attrs = dict(t.split("=", 1) for t in tokens[1:])
            reasons[tokens[0]] = attrs["reason"]
    return reasons


@requires_mmseqs
def test_chains_overlapping_chezod117_are_retired(built):
    out, test = built
    reasons = _reasons(out)
    assert reasons[OVERLAP_RELEASED] == testset.CHEZOD117_OVERLAP
    # an ORIGINAL CheZOD117 sequence that is no release chain counts too
    assert reasons[OVERLAP_ORIGINAL] == testset.CHEZOD117_OVERLAP


@requires_mmseqs
def test_within_test_duplicates_are_retired_shorter_first(built):
    out, test = built
    reasons = _reasons(out)
    assert reasons[SHORT] == testset.WITHIN_TEST_DUPLICATE
    assert reasons[NEAR_A] == testset.WITHIN_TEST_DUPLICATE  # 136 < 168 aa
    kept = read_fasta(out / "TriZOD_test_set.fasta")
    assert set(kept) == {KEPT, LONG, NEAR_B, DISTANT}


@requires_mmseqs
def test_retired_chains_keep_their_sequence_and_stay_leakage_targets(built):
    out, test = built
    retired = read_fasta(out / testset.RETIRED_FASTA)
    assert retired == {
        k: test[k] for k in (OVERLAP_RELEASED, OVERLAP_ORIGINAL, SHORT, NEAR_A)
    }
    # the test set only shrinks: every pinned chain is a test or a retired chain
    kept = read_fasta(out / "TriZOD_test_set.fasta")
    assert set(kept) | set(retired) == set(test) and not set(kept) & set(retired)


@requires_mmseqs
def test_the_labels_table_has_a_reason_column(built):
    out, _ = built
    rows = [
        ln.rstrip("\n").split("\t")
        for ln in (out / "TriZOD_test_set_labels.tsv").open()
    ]
    header = rows[0]
    assert header[-2:] == ["status", "reason"]
    by_id = {r[0]: dict(zip(header, r)) for r in rows[1:]}
    assert by_id[KEPT]["status"] == "test" and by_id[KEPT]["reason"] == ""
    assert by_id[SHORT]["status"] == "retired"
    assert by_id[SHORT]["reason"] == testset.WITHIN_TEST_DUPLICATE
    assert by_id[OVERLAP_RELEASED]["reason"] == testset.CHEZOD117_OVERLAP


@requires_mmseqs
def test_the_summary_records_the_retirements(built):
    out, _ = built
    summary = json.loads((out / "build_test_set_summary.json").read_text())
    by_reason = summary["retired_by_reason"]
    assert sorted(by_reason[testset.CHEZOD117_OVERLAP]) == sorted(
        [OVERLAP_RELEASED, OVERLAP_ORIGINAL]
    )
    assert sorted(by_reason[testset.WITHIN_TEST_DUPLICATE]) == sorted([SHORT, NEAR_A])
    pairs = {tuple(p[:2]) for p in summary["within_test_duplicates"]}
    assert pairs == {(SHORT, LONG), (NEAR_A, NEAR_B)}
    hits = {h["test_id"]: h["chezod117_id"] for h in summary["chezod117_overlaps"]}
    assert hits[OVERLAP_RELEASED] in {"19482_1_1_1", "19482"}
    assert hits[OVERLAP_ORIGINAL] == "hmbdi"
    assert summary["test_set_size"] == 4


@requires_mmseqs
def test_the_checks_need_the_original_chezod117_sequences(tmp_path):
    root, wd, _ = _setup(tmp_path)
    (
        root
        / "data"
        / "external"
        / "chezod"
        / "seth_github"
        / "CheZOD117_test_set_sequences.fasta"
    ).unlink()
    with pytest.raises(SystemExit, match="CheZOD117"):
        testset.main(["--work-dir", str(wd), "--root", str(root)])
