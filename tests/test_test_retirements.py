"""Reviewed test-chain retirements: ``trizod/dataset/data/test_retirements.tsv``.

Release re-audit 2026-11-17b, item 9 (E26): four TriZOD test chains meet no
leakage rule on their deposited sequences, but are near-copies of training
chains once the expression tags are set aside (52047_1_1_1 GB1 + N-myc TAD,
27176_1_1_1 GB1 fusion, 10214_1_1_1 MAGI-1 WW, 53299_1_1_1 RBM39). The author
retired them as ``test_retired`` with the reason ``shared_construct``: not
tested on, still leakage targets. The decision lives in a committed, reviewed
table read by :mod:`trizod.dataset.testset`; the pin itself is not edited.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path

import pytest

from trizod.dataset import testset
from trizod.io.fasta import read_fasta, write_fasta

requires_mmseqs = pytest.mark.skipif(
    shutil.which("mmseqs") is None, reason="mmseqs binary not found in PATH"
)

DECIDED = {"52047_1_1_1", "27176_1_1_1", "10214_1_1_1", "53299_1_1_1"}
HEADER = "\t".join(testset.RETIREMENT_COLUMNS) + "\n"
AA = "ACDEFGHIKLMNPQRSTVWY"


# --------------------------------------------------------------------------- #
# the committed table
# --------------------------------------------------------------------------- #


def test_the_committed_table_retires_the_four_decided_chains():
    rows = testset.load_test_retirements()
    assert {r.id for r in rows} == DECIDED
    assert testset.SHARED_CONSTRUCT == "shared_construct"
    assert {r.reason for r in rows} == {testset.SHARED_CONSTRUCT}
    for r in rows:
        assert r.evidence.strip() and r.partners


def test_every_committed_row_is_a_chain_of_the_committed_pin():
    pin = read_fasta(Path(testset.__file__).parent / "pinned" / "TriZOD_test_set.fasta")
    for row in testset.load_test_retirements():
        assert pin.get(row.id) == row.sequence, row.id


def test_the_committed_table_records_the_pairs_kept_after_review():
    text = testset.TEST_RETIREMENTS_TSV.read_text()
    comments = "".join(ln for ln in text.splitlines(True) if ln.startswith("#"))
    for chain in (
        "50148_1_1_1",
        "15057_1_1_1",
        "17306_1_1_1",
        "4224_1_1_1",
        "27747_1_1_1",
        "53200_1_1_1",
        "19642_1_1_1",
    ):
        assert chain in comments, chain
    assert "kept" in comments.lower()


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("id\tsequence\treason\tevidence\n", "header"),
        (HEADER + "1_1_1_1\tMKV\tsoap\tP\twhy\n", "reason"),
        (HEADER + "1_1_1_1\tMKV\tshared_construct\tP\t\n", "evidence"),
        (HEADER + "1_1_1_1\tMKV\tshared_construct\t\twhy\n", "partner"),
        (HEADER + "bmr1\tMKV\tshared_construct\tP\twhy\n", "ID"),
        (HEADER + "1_1_1_1\tmkv1\tshared_construct\tP\twhy\n", "sequence"),
        (
            HEADER
            + "1_1_1_1\tMKV\tshared_construct\tP\twhy\n"
            + "1_1_1_1\tMKV\tshared_construct\tP\twhy\n",
            "duplicate",
        ),
        (HEADER + "1_1_1_1\tMKV\tshared_construct\tP\n", "fields"),
    ],
)
def test_a_malformed_table_is_rejected(tmp_path, text, error):
    path = tmp_path / "t.tsv"
    path.write_text(text)
    with pytest.raises(ValueError, match=error):
        testset.load_test_retirements(path)


def test_comment_and_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text(
        "# a comment\n\n" + HEADER + "# kept: 2_1_1_1\n"
        "1_1_1_1\tMKV\tshared_construct\t5_1_1_1,6_1_1_1\tGB1 tag\n"
    )
    (row,) = testset.load_test_retirements(path)
    assert (row.id, row.sequence, row.reason) == ("1_1_1_1", "MKV", "shared_construct")
    assert row.partners == ("5_1_1_1", "6_1_1_1") and row.evidence == "GB1 tag"


# --------------------------------------------------------------------------- #
# matching the table against the resolved pin
# --------------------------------------------------------------------------- #


def _row(rid, seq, reason="shared_construct"):
    return testset.ReviewedRetirement(rid, seq, reason, ("9_1_1_1",), "evidence")


def _resolved(pinned, pool=None):
    pool = dict(pinned) if pool is None else pool
    recs, info = testset.resolve_pinned_testset(
        pinned, pool, tier_pools={"strict": pool}
    )
    return recs, info, {}, {"id_map": [], "dropped": []}


def test_a_row_retires_the_test_chain_with_its_sequence():
    pinned = {"1_1_1_1": "AAAA", "2_1_1_1": "CCCC"}
    recs, info, rrecs, rinfo = _resolved(pinned)
    apply, report = testset.match_test_retirements(
        pinned, info, rinfo, [_row("2_1_1_1", "CCCC")]
    )
    assert set(apply) == {"2_1_1_1"} and apply["2_1_1_1"].reason == "shared_construct"
    assert [r["test_id"] for r in report["applied"]] == ["2_1_1_1"]


def test_a_row_follows_a_substituted_id_by_its_sequence():
    # the pinned entry left the pool; its identical-sequence twin stands in
    pinned = {"1_1_1_1": "AAAA", "2_1_1_1": "CCCC"}
    recs, info, rrecs, rinfo = _resolved(
        pinned, pool={"1_1_1_1": "AAAA", "3_1_1_1": "CCCC"}
    )
    apply, _ = testset.match_test_retirements(
        pinned, info, rinfo, [_row("2_1_1_1", "CCCC")]
    )
    assert set(apply) == {"3_1_1_1"}


def test_a_row_whose_sequence_disagrees_with_the_pin_is_an_error():
    pinned = {"1_1_1_1": "AAAA", "2_1_1_1": "CCCC"}
    recs, info, rrecs, rinfo = _resolved(pinned)
    with pytest.raises(ValueError, match="2_1_1_1"):
        testset.match_test_retirements(pinned, info, rinfo, [_row("2_1_1_1", "CCCD")])


def test_a_row_of_an_already_retired_chain_keeps_the_earlier_reason():
    pinned = {"1_1_1_1": "AAAA", "2_1_1_1": "CCCC"}
    recs, info, rrecs, rinfo = _resolved(pinned)
    testset.retire_chains(
        recs, info, rrecs, rinfo, {"2_1_1_1"}, testset.CHEZOD117_OVERLAP
    )
    apply, report = testset.match_test_retirements(
        pinned, info, rinfo, [_row("2_1_1_1", "CCCC")]
    )
    assert apply == {}
    (row,) = report["already_retired"]
    assert row["test_id"] == "2_1_1_1" and row["reason"] == testset.CHEZOD117_OVERLAP


def test_a_row_for_another_pin_is_reported_not_applied():
    pinned = {"1_1_1_1": "AAAA"}
    recs, info, rrecs, rinfo = _resolved(pinned)
    apply, report = testset.match_test_retirements(
        pinned, info, rinfo, [_row("7_1_1_1", "WWWW")]
    )
    assert apply == {} and report["not_in_pin"] == ["7_1_1_1"]


# --------------------------------------------------------------------------- #
# the test-set step end to end (mmseqs)
# --------------------------------------------------------------------------- #

KEPT, SHARED = "100_1_1_1", "52047_1_1_1"


def _setup(tmp_path: Path):
    rng = random.Random(4711)
    test = {
        KEPT: "".join(rng.choice(AA) for _ in range(120)),
        SHARED: "".join(rng.choice(AA) for _ in range(103)),
        "200_1_1_1": "".join(rng.choice(AA) for _ in range(90)),
    }
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
    unrelated = {"19482_1_1_1": "".join(rng.choice(AA) for _ in range(91))}
    write_fasta(unrelated, ext / "chezod117" / "CheZOD117_test_set.fasta")
    (ext / "chezod" / "seth_github").mkdir(parents=True)
    write_fasta(
        unrelated, ext / "chezod" / "seth_github" / "CheZOD117_test_set_sequences.fasta"
    )
    table = tmp_path / "retirements.tsv"
    table.write_text(
        HEADER + f"{SHARED}\t{test[SHARED]}\tshared_construct\t52066_1_1_1\t"
        "N-myc TAD identical to training after the GB1 tag\n"
    )
    return root, wd, test, table


@requires_mmseqs
def test_the_step_retires_the_reviewed_chain_as_shared_construct(tmp_path):
    root, wd, test, table = _setup(tmp_path)
    testset.main(
        ["--work-dir", str(wd), "--root", str(root), "--retirements", str(table)]
    )
    out = wd / "testset"
    assert set(read_fasta(out / "TriZOD_test_set.fasta")) == {KEPT, "200_1_1_1"}
    assert read_fasta(out / testset.RETIRED_FASTA) == {SHARED: test[SHARED]}
    header = next(
        ln for ln in (out / testset.RETIRED_FASTA).open() if ln.startswith(">")
    )
    assert "reason=shared_construct" in header
    labels = [
        ln.rstrip("\n").split("\t")
        for ln in (out / "TriZOD_test_set_labels.tsv").open()
    ]
    by_id = {r[0]: dict(zip(labels[0], r)) for r in labels[1:]}
    assert by_id[SHARED]["status"] == "retired"
    assert by_id[SHARED]["reason"] == testset.SHARED_CONSTRUCT
    summary = json.loads((out / "build_test_set_summary.json").read_text())
    assert summary["retired_by_reason"][testset.SHARED_CONSTRUCT] == [SHARED]
    assert summary["test_set_size"] == 2
    reviewed = summary["reviewed_retirements"]
    assert reviewed["table"] == str(table)
    assert [r["test_id"] for r in reviewed["applied"]] == [SHARED]
    assert "N-myc" in reviewed["applied"][0]["evidence"]


@requires_mmseqs
def test_the_step_reads_the_committed_table_by_default(tmp_path, monkeypatch):
    root, wd, test, table = _setup(tmp_path)
    monkeypatch.setattr(testset, "TEST_RETIREMENTS_TSV", table)
    testset.main(["--work-dir", str(wd), "--root", str(root)])
    summary = json.loads((wd / "testset" / "build_test_set_summary.json").read_text())
    assert summary["retired_by_reason"] == {testset.SHARED_CONSTRUCT: [SHARED]}


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #


def test_the_cli_forwards_the_retirement_table(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from trizod.cli.main import app

    seen = {}
    monkeypatch.setattr(testset, "main", lambda argv: seen.update(argv=argv))
    runner = CliRunner()
    table = tmp_path / "t.tsv"
    r = runner.invoke(app, ["dataset", "test-set", "--retirements", str(table)])
    assert r.exit_code == 0, r.output
    assert seen["argv"][seen["argv"].index("--retirements") + 1] == str(table)
    # the default is not forwarded: testset.main reads the committed table
    seen.clear()
    assert runner.invoke(app, ["dataset", "test-set"]).exit_code == 0
    assert "--retirements" not in seen["argv"]
    out = runner.invoke(app, ["dataset", "test-set", "--help"]).output
    help_text = "".join(re.sub(r"[\u2500-\u257f]", " ", out).split())
    assert "--retirements" in help_text and "test_retirements.tsv" in help_text
