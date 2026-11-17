"""Per-chain changes and known issues of a release (release audit 2026-11-17b,
item 7): every number the CHANGELOG states and every row of
``known_issues.tsv`` is computed by :mod:`trizod.dataset.known_issues` from the
release, the builds or a table finalize is given."""

import json
import pickle
import subprocess
import sys

import pynmrstar
import pytest

from trizod import paths
from trizod.bmrb.bmrb import SampleConditions
from trizod.dataset import known_issues as ki


def _conditions(*rows):
    """``SampleConditions`` parsed from (Type, Val, Val_units) rows."""
    sf = pynmrstar.Saveframe.from_scratch(
        "sample_conditions_1", "_Sample_condition_list"
    )
    sf.add_tag("Sf_category", "sample_conditions")
    sf.add_tag("ID", "1")
    loop = pynmrstar.Loop.from_scratch("_Sample_condition_variable")
    loop.add_tag(["Type", "Val", "Val_err", "Val_units"])
    loop.add_data([[t, v or ".", ".", u or "."] for t, v, u in rows])
    sf.add_loop(loop)
    return SampleConditions(sf)


T298 = ("temperature", "298", "K")


def test_a_pd_reading_is_reported_with_its_conversion():
    got = ki.classify_conditions(_conditions(("pD", "7.5", "pD"), T298))
    assert got == [
        (
            "ph_scale_pd",
            "pD 7.5 deposited (unit pD), converted to pD − 0.4: ph 7.1",
        )
    ]


def test_a_ph_star_reading_is_used_as_is():
    got = ki.classify_conditions(_conditions(("pH*", "6.5", "pH*"), T298))
    assert got == [
        ("ph_scale_ph_star", "pH* 6.5 deposited (unit pH*), used as is: ph 6.5")
    ]


def test_plain_ph_and_kelvin_are_no_issue():
    assert ki.classify_conditions(_conditions(("pH", "6.5", "pH"), T298)) == []


@pytest.mark.parametrize(
    ("value", "issue"),
    [
        ("25", "temperature_celsius_as_kelvin"),
        ("77", "temperature_fahrenheit_as_kelvin"),
    ],
)
def test_a_temperature_below_100_kelvin_is_read_as_celsius_or_fahrenheit(value, issue):
    got = ki.classify_conditions(_conditions(("temperature", value, "K")))
    assert [i for i, _ in got] == [issue]
    assert f"temperature {value} deposited with unit K" in got[0][1]


def test_a_pressure_without_a_unit_is_reported():
    got = ki.classify_conditions(_conditions(T298, ("pressure", "1", "")))
    assert got[0][0] == "pressure_unit_missing"
    assert "without a unit" in got[0][1]
    assert ki.classify_conditions(_conditions(T298, ("pressure", "1", "atm"))) == []


class FakeEntry:
    """What :func:`known_issues.condition_issues` reads from a pickled entry."""

    def __init__(self, cond):
        self.conditions = {"1": cond}

    def get_peptide_shifts(self):
        return {("1", "1", "1"): ([], "1", "1", ["1"])}


def test_condition_issues_read_the_build_pickles(tmp_path):
    with open(tmp_path / "9001.pkl", "wb") as fh:
        pickle.dump(FakeEntry(_conditions(("pD", "7.0", ""), T298)), fh)
    rows = [
        {"id": "9001_1_1_1", "entry_id": "9001", "st_id": "1",
         "entity_assem_id": "1", "entity_id": "1"},
        {"id": "9001_2_1_1", "entry_id": "9001", "st_id": "2",
         "entity_assem_id": "1", "entity_id": "1"},
        {"id": "9002_1_1_1", "entry_id": "9002", "st_id": "1",
         "entity_assem_id": "1", "entity_id": "1"},
    ]  # fmt: skip
    got = ki.condition_issues(rows, tmp_path)
    assert ("9001_1_1_1", "ph_scale_pd") in {(c, i) for c, i, _ in got}
    assert ("9001_2_1_1", "conditions_unread", "no such shift table") in got
    assert [i for c, i, _ in got if c == "9002_1_1_1"] == ["conditions_unread"]
    cur = {"ids": {"strict": {"9001_1_1_1"}, "tolerant": {"9001_1_1_1"}}}
    text = ki.condition_text(got, cur, {r["id"]: r for r in rows})
    assert text["ph"].startswith("1 released chains were measured in pH* or pD")
    assert "(1 of them strict: 9001_1_1_1)" in text["ph"]
    assert text["unread"] == 2


def test_compare_labels_counts_crossings_and_mask_flips():
    old = [None, 0.30, 0.40, 0.80, 0.10]
    new = [None, 0.36, 0.40, 0.60, None]
    got = ki.compare_labels(old, new)
    assert got["scored_in_both"] == 3
    assert got["max_abs_dg"] == pytest.approx(0.20)
    assert got["crossings"] == {0.35: 1, 0.70: 1}
    assert got["mask_flips"] == 1
    assert got["changed_positions"] == [2, 4, 5]


def test_tier_move_causes_follow_the_rule_flags():
    row = {
        "high_pressure": True,
        "membrane_mimetic_sample": True,
        "ionic_strength": 1.0,
        "temperature_placeholder": True,
    }
    assert ki._move_causes("unfiltered", row, False) == "273 K re-score"
    assert ki._move_causes("tolerant", row, False) == "pressure + 273 K re-score"
    assert (
        ki._move_causes("strict", row, True)
        == "pressure + membrane + ionic + 273 K + sequence alphabet"
    )
    assert ki._move_causes("moderate", {}, False) == "other"
    assert ki._move_causes("moderate", None, False) == "not in this release"


def test_snapshot_drift_lists_changed_chains_with_their_split(tmp_path):
    path = tmp_path / "drift.tsv"
    per_tier = json.dumps(
        {"unfiltered": {"gscores": {"max_abs_dG": 0.33, "mask_flips": 2}},
         "moderate": "leaves_tier_with_new_file"}
    )  # fmt: skip
    path.write_text(
        "id\treleased\tsplit\tlabel_tier_source\tlabel_or_tier_change\tper_tier\n"
        f"50751_1_1_1\tTrue\ttrain\ttolerant\tTrue\t{per_tier}\n"
        "11348_1_1_1\tTrue\ttest_trizod\tstrict\tFalse\t{}\n"
        "36002_1_1_1\tFalse\tNone\tNone\tTrue\t{}\n"
    )
    rows = {"50751_1_1_1": {"split": "train"}, "11348_1_1_1": {"split": "test_trizod"}}
    got = ki.snapshot_drift(path, rows)
    assert got["changed"] == ["36002_1_1_1", "50751_1_1_1"]
    assert got["test_affected"] == []
    ev = {c: e for c, _, e in got["rows"]}
    assert "unfiltered: max |ΔG| 0.33, 2 mask flips" in ev["50751_1_1_1"]
    assert "moderate: leaves tier with new file" in ev["50751_1_1_1"]
    assert ev["36002_1_1_1"].endswith("now no longer released")
    assert "no test chain is affected" in ki.snapshot_drift_text(got)


def _row(cid, split, seq, mask, train_tier=None):
    return {"id": cid, "split": split, "sequence": seq, "mask": mask,
            "train_tier": train_tier}  # fmt: skip


def test_construct_caveats_and_nonstandard_counts(tmp_path):
    rows = [
        _row("1_1_1_1", "train", "AXAA", [True] * 4, "tolerant"),
        _row("2_1_1_1", "train", "AAAA", [True, True, False, False], "unfiltered"),
        _row("3_1_1_1", "test_trizod", "AAAU", [True] * 4),
    ]
    tsv = tmp_path / "caveats.tsv"
    tsv.write_text("class\tid\nuniform_2H\t1_1_1_1\nuniform_2H\t9_1_1_1\n")
    got = ki.construct_caveats(tsv, rows)
    d = got["summary"]["uniform_2H"]
    assert d["training"] == {"chains": 1, "share": pytest.approx(4 / 6)}
    assert d["tolerant training"] == {"chains": 1, "share": 1.0}
    assert d["test"]["chains"] == 0
    assert got["unknown"] == ["9_1_1_1"]
    nonstd = ki.nonstandard_counts(rows)
    assert nonstd == {
        "train_chains": 1,
        "train_positions": 1,
        "train_residues": 8,
        "test_chains": 1,
    }
    text = ki.construct_text(got, nonstd)
    assert "`uniform_2H`: training 1 chains (66.7 % of scored residues)" in text
    assert "flag columns are deferred" in text
    assert "not counted" in ki.construct_text(None, nonstd)


def test_few_scored_training_and_pin_trimmed():
    rows = [
        _row("1_1_1_1", "train", "A" * 20, [True] * 9 + [False] * 11, "moderate"),
        _row("2_1_1_1", "train", "A" * 20, [True] * 10 + [False] * 10, "moderate"),
    ]
    few = ki.few_scored_training(rows)
    assert [c for c, _, _ in few] == ["1_1_1_1"]
    prov = {"count": 452, "written": "2026-11-10", "dropped": ["1_1_1_1", "9_1_1_1"],
            "trimmed_from": {"count": 460, "written": "2026-08-11"}}  # fmt: skip
    got = ki.pin_trimmed(prov, {r["id"]: r for r in rows})
    assert got["n"] == 2
    assert "2 chains were trimmed from the 460-chain pin" in got["text"]
    assert "1 not released, 1 train" in got["text"]


def test_reviewed_table_refuses_a_chain_not_in_the_release(tmp_path):
    tsv = tmp_path / "r.tsv"
    tsv.write_text("chain_id\tissue\tevidence\n1_1_1_1\tx\ty\n")
    assert ki.reviewed_issues(tsv, {"1_1_1_1": {"split": "train"}}) == [
        ("1_1_1_1", "x", "y (split train)")
    ]
    with pytest.raises(SystemExit, match="not in the release"):
        ki.reviewed_issues(tsv, {})


def test_the_committed_reviewed_table_is_well_formed():
    path = paths.ROOT / ki.REVIEWED_TSV
    lines = path.read_text().splitlines()
    assert lines[0].split("\t") == list(ki.KNOWN_ISSUES_COLUMNS)
    for line in lines[1:]:
        cid, issue, evidence = line.split("\t")
        assert cid.count("_") == 3 and issue and evidence


def test_write_tsv_sorts_dedups_and_flattens(tmp_path):
    out = ki.write_tsv(
        [("10_1_1_1", "b", "x\ty"), ("9_1_1_1", "a", "z"), ("9_1_1_1", "a", "z")],
        tmp_path / "k.tsv",
    )
    assert out.read_text().splitlines() == [
        "chain_id\tissue\tevidence",
        "9_1_1_1\ta\tz",
        "10_1_1_1\tb\tx y",
    ]


def test_paths_follow_the_data_root_and_bmrb_dir_overrides(tmp_path):
    """``TRIZOD_DATA_ROOT`` moves every data/ constant, ``TRIZOD_BMRB_DIR`` the
    snapshot alone; the code root and the caches stay in the checkout."""
    code = (
        "import json; from trizod import paths as p; print(json.dumps("
        "[str(p.ROOT), str(p.DATA), str(p.RAW_BMRB), str(p.INTERIM_SCORED), "
        "str(p.TMP)]))"
    )
    snap = tmp_path / "snap"
    env = {
        "TRIZOD_DATA_ROOT": str(tmp_path),
        "TRIZOD_BMRB_DIR": str(snap),
        "PATH": "/usr/bin:/bin",
    }
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    root, data, bmrb, scored, tmp = json.loads(out)
    base = tmp_path.resolve()
    assert root == str(paths.repo_root()) and tmp == str(paths.repo_root() / "tmp")
    assert data == str(base / "data")
    assert scored == str(base / "data" / "interim" / "scored")
    assert bmrb == str(snap.resolve())
    env.pop("TRIZOD_BMRB_DIR")
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert json.loads(out)[2] == str(base / "data" / "raw" / "bmrb_entries")


def test_chezod117_conditions_name_the_chains_a_tier_would_reject():
    rows = [
        {"id": "15768_1_1_1", "split": "test_chezod117", "cosolvent_evidence": True},
        {"id": "4985_1_1_1", "split": "test_chezod117", "is_bound": True},
        {"id": "5_1_1_1", "split": "test_chezod117", "physical_state": "denatured"},
        {"id": "6_1_1_1", "split": "test_chezod117", "physical_state": "native"},
        {"id": "7_1_1_1", "split": "train", "cosolvent_evidence": True},
    ]
    got = ki.chezod117_conditions(rows)
    assert [c for c, _, _ in got["rows"]] == ["5_1_1_1", "4985_1_1_1", "15768_1_1_1"]
    assert "1 chains name a perturbing cosolvent (15768_1_1_1)" in got["text"]
    assert (
        "1 are deposited as denatured or (partially) unfolded (5_1_1_1)" in got["text"]
    )


def test_ionic_drivers_list_the_inconsistent_chains(tmp_path):
    log = tmp_path / "drivers.tsv"
    log.write_text(
        "chain\tflag\tI_deposited_strict_M\tI_implied_M\timplied_parts\n"
        "5365_1_1_1\tINCONSISTENT: components imply <= 0.5 M\t1.5\t0.128\tNaCl 100 mM\n"
        "4881_1_1_1\toverstated (both > 0.5 M)\t3.0\t0.54\tKPPi\n"
    )
    got = ki.ionic_drivers(log, {"5365_1_1_1": {"split": "redundant"}})
    assert got["rows"] == [
        (
            "5365_1_1_1",
            "ionic_strength_contradicts_components",
            "ionic strength 1.5 M deposited, its sample components imply 0.128 M "
            "(NaCl 100 mM); used as deposited, so the chain is out of strict "
            "(split redundant)",
        )
    ]
    assert got["text"].startswith("Ionic strengths are used as deposited: 1 chains")
