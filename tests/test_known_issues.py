"""Per-chain known issues of a release: every number the CHANGELOG states and
every row of ``known_issues.tsv`` is computed by
:mod:`trizod.dataset.known_issues` from the release, the build or a table
finalize is given, and describes a released chain, never a change against
another build."""

import json
import pickle
import re
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
        "train_scored_positions": 1,
        "train_chain_residues": 4,
        "train_residues": 8,
        "test_chains": 1,
    }
    text = ki.construct_text(got, nonstd)
    assert "`uniform_2H` (whole chains): training 1 chains (66.7 % of scored" in text
    assert "flag columns are deferred" in text
    assert "not counted" in ki.construct_text(None, nonstd)


def test_few_scored_training():
    rows = [
        _row("1_1_1_1", "train", "A" * 20, [True] * 9 + [False] * 11, "moderate"),
        _row("2_1_1_1", "train", "A" * 20, [True] * 10 + [False] * 10, "moderate"),
    ]
    few = ki.few_scored_training(rows)
    assert [c for c, _, _ in few] == ["1_1_1_1"]


def test_every_placeholder_chain_gets_one_row_without_a_previous_build():
    rows = [
        {"id": "20_1_1_1", "split": "train", "temperature_placeholder": True},
        {"id": "3_1_1_1", "split": "test_chezod117", "temperature_placeholder": True},
        {"id": "4_1_1_1", "split": "test_retired", "temperature_placeholder": True,
         "retired_reason": "not_strict"},
        {"id": "5_1_1_1", "split": "train", "temperature_placeholder": False},
        {"id": "6_1_1_1", "split": "train"},
    ]  # fmt: skip
    got = ki.temperature_placeholder_rows(rows)
    assert got == [
        (cid, "temperature_placeholder_rescore",
         "deposited 273 K read as not deposited; scored at the 298 K default")
        for cid in ("3_1_1_1", "4_1_1_1", "20_1_1_1")
    ]  # fmt: skip
    text = ki.chezod_placeholder_text(rows)
    assert text.startswith("1 CheZOD117 chains carry `temperature_placeholder`")
    assert "(3_1_1_1)" in text
    assert "1 retired test chains carry it (1 not_strict)" in text
    assert "ΔG" not in text and "previous" not in text


def test_no_issue_compares_with_another_build():
    for name in ("tier_moves", "label_changes", "snapshot_drift", "pin_trimmed",
                 "alphabet_rows", "rescore_rows", "changed_entries"):  # fmt: skip
        assert not hasattr(ki, name), name


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
        # facts only: no internal audit or validation label
        assert not re.search(r"audit|validation|2026-\d\d-\d\d|\b[A-G]\d{1,2}\b",
                             evidence), (cid, evidence)  # fmt: skip


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


# --------------------------------------------------------------------------- #
# pressure classes and construct caveats
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", ["1", "1.013"])
def test_a_pressure_of_about_one_pascal_is_a_placeholder(value):
    got = ki.classify_conditions(_conditions(T298, ("pressure", value, "Pa")))
    assert [i for i, _ in got] == ["pressure_placeholder_pa"]
    assert f"pressure {value} Pa deposited" in got[0][1]
    assert "pressure_bar" in got[0][1] and "no tier" in got[0][1]
    # one atmosphere in Pa is no issue
    assert ki.classify_conditions(_conditions(T298, ("pressure", "101325", "Pa"))) == []


def test_a_bar_pressure_of_one_atmosphere_in_hpa_is_reported():
    got = ki.classify_conditions(_conditions(T298, ("pressure", "1013", "bar")))
    assert [i for i, _ in got] == ["pressure_hpa_as_bar"]
    assert "read as hPa" in got[0][1] and "1.013" in got[0][1]


def test_the_pressure_sentence_names_the_placeholder_classes():
    issues = [
        ("1_1_1_1", "pressure_placeholder_pa", "x"),
        ("2_1_1_1", "pressure_placeholder_pa", "x"),
        ("3_1_1_1", "pressure_hpa_as_bar", "x"),
        ("4_1_1_1", "pressure_unit_missing", "x"),
    ]
    cur = {"ids": {t: set() for t in ("unfiltered", "tolerant", "moderate", "strict")}}
    text = ki.condition_text(issues, cur, {})["pressure"]
    assert "2 released chains deposited a pressure of about 1 Pa" in text
    assert "1 deposited 1013 bar" in text and "3_1_1_1" in text
    assert "1 released chains deposited a pressure without a unit" in text


def test_construct_caveats_report_residue_classes_at_residue_level(tmp_path):
    rows = [
        _row("1_1_1_1", "train", "AAAAAAAAAA", [True] * 10, "tolerant"),
        _row("2_1_1_1", "train", "AAAAAAAAAA", [True] * 10, "tolerant"),
        _row("3_1_1_1", "test_trizod", "AAAAAAAAAA", [True] * 10),
    ]
    tsv = tmp_path / "caveats.tsv"
    tsv.write_text(
        "class\tid\noutside_uniprot_span_sifts\t1_1_1_1\ndeut_uniform_2H\t2_1_1_1\n"
    )
    span = {"1_1_1_1": (2, 10), "2_1_1_1": (0, 10), "3_1_1_1": (1, 10)}
    got = ki.construct_caveats(
        tsv, rows, residue_counts={"outside_uniprot_span_sifts": span}
    )
    text = ki.construct_text(got, ki.nonstandard_counts(rows))
    assert (
        "`outside_uniprot_span_sifts` (residue level): training 2 of 20 "
        "labelled residues (10.0 %) of the 2 assessed chains"
    ) in text
    assert "`deut_uniform_2H` (whole chains): training 1 chains" in text
    ev = {(c, i): e for c, i, e in got["rows"]}
    assert (
        "2 of 10 labelled residues"
        in ev[("1_1_1_1", "construct_outside_uniprot_span_sifts")]
    )
    assert (
        "known_issues_inputs/construct_caveats.tsv"
        in ev[("2_1_1_1", "construct_deut_uniform_2H")]
    )
    no_counts = ki.construct_text(
        ki.construct_caveats(tsv, rows), ki.nonstandard_counts(rows)
    )
    assert "residue-level share not measured" in no_counts


def test_nonstandard_letters_are_counted_in_the_residues_of_their_chains():
    rows = [
        _row("1_1_1_1", "train", "AXAAX", [True, True, True, True, False], "tolerant"),
        _row("2_1_1_1", "train", "AAAA", [True] * 4, "tolerant"),
        _row("3_1_1_1", "test_trizod", "AAAA", [True] * 4),
    ]
    nonstd = ki.nonstandard_counts(rows)
    assert nonstd["train_positions"] == 2 and nonstd["train_scored_positions"] == 1
    assert nonstd["train_chain_residues"] == 5
    text = ki.construct_text(None, nonstd)
    assert "2 X/U/Z positions (1 scored) in the 5 residues of these chains" in text


def test_a_listed_chain_without_labelled_residues_outside_the_span_gets_no_row(
    tmp_path,
):
    rows = [
        _row("1_1_1_1", "train", "AAAAAAAAAA", [True] * 10, "tolerant"),
        _row("2_1_1_1", "train", "AAAAAAAAAA", [True] * 10, "tolerant"),
        _row("3_1_1_1", "redundant", "AAAAAAAAAA", [True] * 10),
    ]
    tsv = tmp_path / "caveats.tsv"
    tsv.write_text(
        "class\tid\n"
        "outside_uniprot_span_sifts\t1_1_1_1\n"
        "outside_uniprot_span_sifts\t2_1_1_1\n"
        "outside_uniprot_span_sifts\t3_1_1_1\n"
    )
    span = {"1_1_1_1": (2, 10), "2_1_1_1": (0, 10)}  # 3_1_1_1 not assessed
    got = ki.construct_caveats(
        tsv, rows, residue_counts={"outside_uniprot_span_sifts": span}
    )
    listed = {c for c, _, _ in got["rows"]}
    assert listed == {"1_1_1_1", "3_1_1_1"}
    assert got["dropped"] == {"outside_uniprot_span_sifts": ["2_1_1_1"]}
    summary = got["summary"]["outside_uniprot_span_sifts"]
    assert summary["listed_without_labelled_residue"] == 1
    assert summary["training"]["affected_chains"] == 1
    text = ki.construct_text(got, ki.nonstandard_counts(rows))
    assert "1 listed chains have no labelled residue there" in text
    ev = {c: e for c, _, e in got["rows"]}
    assert "not assessed at residue level" in ev["3_1_1_1"]
