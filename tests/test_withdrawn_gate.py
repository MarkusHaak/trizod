"""Withdrawn / unlisted BMRB entries never reach a dataset pool (audit I-06).

Ten entries in the 2025-11 snapshot are no longer listed by BMRB
(``/v2/list_entries``): six return HTTP 404 and four are still served but
delisted (instant search labels nine "withdrawn by the author" and 36392 "on
hold"). Two of them, 36170_1_1_1 and 36288_1_1_1, sit in the pinned TriZOD test
set; 36392_1_1_1 is an unfiltered-tier training representative.

``trizod dataset build`` drops every entry on the committed list before
deduplication, so a withdrawn chain lands in no tier pool, hence in no training
set and no test set. These tests run the real ``build.main`` on tiny synthetic
``scores.json`` + pickle inputs and never touch the network.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from tests._stubs import run_build as _run_build
from tests._stubs import scored_record
from trizod.dataset import build, withdrawn
from trizod.dataset.testset import load_tier_pools, resolve_pinned_testset
from trizod.io.fasta import read_fasta

TIERS = ("unfiltered", "tolerant", "moderate", "strict")

# --------------------------------------------------------------------------- #
# synthetic build harness (tests/_stubs.py, shared with the placeholder gate)
# --------------------------------------------------------------------------- #


def _record(rid: str, seq: str, n_bb_pos: int = 30) -> dict:
    return scored_record(rid, seq, n_bb_pos)


def _nested(records_by_top_tier: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Tiers nest (strict ⊂ moderate ⊂ tolerant ⊂ unfiltered): a record whose
    strictest tier is T is written to T and every looser tier."""
    out: dict[str, list[dict]] = {t: [] for t in TIERS}
    for top, recs in records_by_top_tier.items():
        for tier in TIERS[: TIERS.index(top) + 1]:
            out[tier].extend(recs)
    return out


def _all_pool_ids(final: Path) -> set[str]:
    ids: set[str] = set()
    for tier in TIERS:
        ids |= set(read_fasta(final / tier / f"{tier}.fasta"))
        with (final / tier / f"{tier}_all_ranked.tsv").open() as fh:
            next(fh)
            ids |= {line.split("\t", 1)[0] for line in fh if line.strip()}
    return ids


SEQ_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"  # 33 aa
SEQ_B = "GSHMSEQDLRALLEEAAKRGNPEEVRKLLEKG"  # 32 aa
SEQ_C = "MDEKLNQAIQEALRKGDKEGALKLLEEALK"  # 30 aa


# --------------------------------------------------------------------------- #
# the committed list
# --------------------------------------------------------------------------- #

EXPECTED_STATUS = {
    "30167": "withdrawn",
    "30168": "withdrawn",
    "30794": "withdrawn",
    "34337": "withdrawn",
    "50748": "withdrawn",
    "50838": "withdrawn",
    "36144": "withdrawn",
    "36170": "withdrawn",
    "36288": "withdrawn",
    "36392": "on_hold",
}


def test_committed_list_records_the_ten_unlisted_entries():
    entries = withdrawn.load_withdrawn()
    assert {eid: e.status for eid, e in entries.items()} == EXPECTED_STATUS
    for e in entries.values():
        assert e.date_checked == "2026-11-06"
        assert "list_entries" in e.source
        assert "instant" in e.source


def test_committed_list_lives_inside_the_package():
    """Shipped with the code (versioned in git), not under the ignored data/."""
    path = withdrawn.WITHDRAWN_TSV
    assert path.exists()
    assert Path(withdrawn.__file__).parent in path.parents


def test_withdrawn_list_round_trips(tmp_path):
    entries = [
        withdrawn.UnlistedEntry("9", "on_hold", "2030-01-02", "src", "held"),
        withdrawn.UnlistedEntry("10", "withdrawn", "2030-01-02", "src", ""),
    ]
    path = tmp_path / "list.tsv"
    withdrawn.write_withdrawn(entries, path)
    assert withdrawn.load_withdrawn(path) == {e.entry_id: e for e in entries}


def test_committed_list_is_in_canonical_writer_format(tmp_path):
    """A refresh rewrites the file with ``write_withdrawn``; the committed copy
    must already be in that exact format so the refresh diff shows only real
    status changes."""
    path = tmp_path / "rewritten.tsv"
    withdrawn.write_withdrawn(withdrawn.load_withdrawn().values(), path)
    assert path.read_text() == withdrawn.WITHDRAWN_TSV.read_text()


def test_unknown_status_is_rejected(tmp_path):
    path = tmp_path / "list.tsv"
    path.write_text(
        "entry_id\tstatus\tdate_checked\tsource\tdetail\n123\tgone\t2026-11-06\tsrc\t\n"
    )
    with pytest.raises(ValueError, match="gone"):
        withdrawn.load_withdrawn(path)


# --------------------------------------------------------------------------- #
# the build gate
# --------------------------------------------------------------------------- #


def test_build_drops_withdrawn_entries_from_every_tier_pool(tmp_path):
    final = _run_build(
        tmp_path,
        _nested(
            {
                # pinned test chain, strict -> present in every tier's scores.json
                "strict": [_record("36170_1_1_1", SEQ_A), _record("100_1_1_1", SEQ_C)],
                # unfiltered-tier training representative
                "unfiltered": [_record("36392_1_1_1", SEQ_B)],
            }
        ),
    )
    ids = _all_pool_ids(final)
    assert "36170_1_1_1" not in ids
    assert "36392_1_1_1" not in ids
    assert "100_1_1_1" in ids
    for tier in TIERS:
        seqs = set(read_fasta(final / tier / f"{tier}.fasta").values())
        assert SEQ_A not in seqs and SEQ_B not in seqs


def test_withdrawn_entries_are_dropped_before_dedup(tmp_path):
    """36288 (withdrawn) and 36385 (its live re-deposition) share a sequence.
    36288 outranks 36385, so a post-dedup filter would delete the sequence's
    only representative; dropping first makes 36385 the representative."""
    final = _run_build(
        tmp_path,
        _nested(
            {
                "strict": [
                    _record("36288_1_1_1", SEQ_A, n_bb_pos=33),
                    _record("36385_1_1_1", SEQ_A, n_bb_pos=20),
                ]
            }
        ),
    )
    for tier in TIERS:
        assert read_fasta(final / tier / f"{tier}.fasta") == {"36385_1_1_1": SEQ_A}


def test_build_logs_and_summarises_the_withdrawn_drop(tmp_path, capsys):
    final = _run_build(
        tmp_path,
        _nested(
            {
                "strict": [_record("36170_1_1_1", SEQ_A), _record("100_1_1_1", SEQ_C)],
                "unfiltered": [_record("36392_1_1_1", SEQ_B)],
            }
        ),
    )
    out = capsys.readouterr().out
    summary = json.loads((final / "final_dataset_summary.json").read_text())
    # 36170 appears in 4 tier files, 36392 in 1 -> 5 rows, 2 chains.
    assert summary["global_dropped_withdrawn"] == 5
    assert summary["withdrawn_dropped_ids"] == ["36170_1_1_1", "36392_1_1_1"]
    assert summary["withdrawn_list"]["n_entries"] == 10
    assert summary["withdrawn_list"]["date_checked"] == "2026-11-06"
    assert summary["tiers"]["unfiltered"]["dropped_withdrawn"] == 2
    assert summary["tiers"]["strict"]["dropped_withdrawn"] == 1
    assert "withdrawn" in out and "5 rows" in out


def test_build_accepts_an_alternative_withdrawn_list(tmp_path):
    custom = tmp_path / "custom.tsv"
    withdrawn.write_withdrawn(
        [withdrawn.UnlistedEntry("100", "withdrawn", "2030-01-01", "test", "")],
        custom,
    )
    final = _run_build(
        tmp_path,
        _nested(
            {"strict": [_record("36170_1_1_1", SEQ_A), _record("100_1_1_1", SEQ_C)]}
        ),
        "--withdrawn-list",
        str(custom),
    )
    ids = _all_pool_ids(final)
    assert "100_1_1_1" not in ids
    assert "36170_1_1_1" in ids  # not on the custom list


def test_gate_is_on_by_default():
    args = build.make_parser().parse_args([])
    assert args.withdrawn_list is None  # None -> the committed list
    assert build.resolve_withdrawn_list(args.withdrawn_list) == withdrawn.WITHDRAWN_TSV


def test_default_build_never_touches_the_network(tmp_path, monkeypatch):
    def _no_network(*a, **k):
        raise AssertionError("the default build must not call the network")

    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    _run_build(tmp_path, _nested({"strict": [_record("100_1_1_1", SEQ_C)]}))


# --------------------------------------------------------------------------- #
# what the pinned test set does with the result (testset.py, unchanged)
# --------------------------------------------------------------------------- #


def test_pinned_test_set_drops_a_withdrawn_chain_or_moves_to_a_live_twin(tmp_path):
    """The pin is resolved by SEQUENCE against the tolerant pool. A withdrawn
    chain with no live identical-sequence twin (36170) is dropped; one whose
    sequence survives in another entry (36288 -> its re-deposition 36385) is
    substituted, not dropped."""
    final = _run_build(
        tmp_path,
        _nested(
            {
                "strict": [
                    _record("36170_1_1_1", SEQ_A),
                    _record("36288_1_1_1", SEQ_B, n_bb_pos=33),
                    _record("36385_1_1_1", SEQ_B, n_bb_pos=20),
                    _record("100_1_1_1", SEQ_C),
                ]
            }
        ),
    )
    pinned = {"36170_1_1_1": SEQ_A, "36288_1_1_1": SEQ_B, "100_1_1_1": SEQ_C}
    pools = load_tier_pools(final)
    recs, info = resolve_pinned_testset(pinned, pools["tolerant"], tier_pools=pools)
    assert info["dropped"] == ["36170_1_1_1"]
    assert info["substitutions"] == [["36288_1_1_1", "36385_1_1_1"]]
    assert recs == {"36385_1_1_1": SEQ_B, "100_1_1_1": SEQ_C}


# --------------------------------------------------------------------------- #
# refreshing the list from the BMRB API (stubbed: no network in tests)
# --------------------------------------------------------------------------- #


def _fake_bmrb(listed, entry_codes, instant_labels):
    """A ``get(url) -> (http_status, json)`` stand-in for the BMRB API."""
    calls = []

    def get(url):
        calls.append(url)
        if "/list_entries" in url:
            return 200, sorted(listed)
        if "/entry/" in url:
            eid = url.split("/entry/")[1].split("?")[0]
            return entry_codes[eid], None
        if "/instant" in url:
            eid = url.split("term=")[1]
            return 200, [
                {"value": "999999", "label": "unrelated hit"},
                {"value": eid, "label": instant_labels[eid]},
            ]
        raise AssertionError(url)

    get.calls = calls
    return get


def test_fetch_unlisted_entries_classifies_without_network():
    get = _fake_bmrb(
        listed={"1", "2", "36385"},
        entry_codes={"30167": 404, "36170": 200, "36392": 200, "777": 200},
        instant_labels={
            "30167": "Entry has been withdrawn by the author.",
            "36170": "Entry has been withdrawn by the author.",
            "36392": "Entry is on hold. Release: Released",
            "777": "Some released entry title",
        },
    )
    entries = withdrawn.fetch_unlisted_entries(
        ["1", "2", "30167", "36170", "36392", "777"], get=get, today="2030-05-06"
    )
    by_id = {e.entry_id: e for e in entries}
    assert [e.entry_id for e in entries] == ["777", "30167", "36170", "36392"]
    assert by_id["30167"].status == "withdrawn"
    assert "404" in by_id["30167"].detail
    assert by_id["36170"].status == "withdrawn"
    assert "200" in by_id["36170"].detail
    assert by_id["36392"].status == "on_hold"
    assert by_id["777"].status == "unlisted"  # missing from the list, no label
    assert {e.date_checked for e in entries} == {"2030-05-06"}
    # Only the unlisted entries are probed one by one.
    assert not any(u.endswith("/entry/1?tag=Entry.ID") for u in get.calls)


def test_local_entry_ids_reads_a_bmrb_snapshot_dir(tmp_path):
    for name in ("bmr36170", "bmr4", "notes"):
        (tmp_path / name).mkdir()
    for eid in ("36170", "4"):
        (tmp_path / f"bmr{eid}" / f"bmr{eid}_3.str").write_text("")
    (tmp_path / "notes" / "readme.txt").write_text("")
    (tmp_path / "bmr5.str").write_text("")
    assert withdrawn.local_entry_ids(tmp_path) == {"36170", "4"}
