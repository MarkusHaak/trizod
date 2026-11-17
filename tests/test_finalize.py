"""``trizod dataset finalize`` assembles the deposit (release audit 2026-11-17,
B4, E21, F33, G35, G36, G38, G39) -- tested end to end on a synthetic bundle,
with the real docs templates and repository metadata."""

import hashlib
import json
import re
import shutil

import pytest

from trizod.dataset import finalize
from trizod.dataset import release_info as ri

from ._release_fixtures import (
    CHAINS,
    TIERS,
    _seq,
    build_release_parquet,
    record,
    write_bundle,
    write_results,
    write_shifts,
    write_snapshot,
)

pytest.importorskip("pyarrow")

EXPECTED_FILES = {
    "trizod_dataset.parquet",
    "trizod_shifts.parquet",
    "disorder_trizod.fasta",
    "disorder_trizod.summary.json",
    "udonpred/README.md",
    "udonpred/export_summary.json",
    "udonpred/trizod_tolerant/train.jsonl",
    "udonpred/trizod_tolerant/test.jsonl",
    "udonpred/chezod117/test.jsonl",
    "crosswalk/test_set_crosswalk.tsv",
    "crosswalk/test_set_crosswalk.json",
    "crosswalk/test_set_overlap.tsv",
    "test/TriZOD_test_set.fasta",
    "test/TriZOD_test_retired.fasta",
    "test/CheZOD117_test_set.fasta",
    "test/CheZOD117_original_sequences.fasta",
    "test/TriZOD_test_set_labels.tsv",
    "README.md",
    "datasheet.md",
    "CHANGELOG.md",
    "columns.md",
    "class_balance.tsv",
    "known_issues.tsv",
    "VALIDATION.json",
    "bmrb_files.sha256",
    "LICENSE.md",
    "CITATION.cff",
    "zenodo.json",
    "MANIFEST.json",
    "SHA256SUMS",
}


def write_previous_bundle(bundle, prev):
    """The synthetic bundle as an earlier build saw it: 1003_1_1_1 was still
    moderate (the membrane rule moved it), 2003_1_1_1 was strict with other
    labels (the 273 K rule re-scored it), and 1001_1_1_1's deposited sequence
    still carried a trailing stop character."""
    shutil.copytree(bundle, prev)
    for tier in TIERS:
        path = prev / "scores" / tier / "scores.json"
        recs = [json.loads(line) for line in path.read_text().splitlines() if line]
        ids = {r["ID"] for r in recs}
        if tier == "moderate" and "1003_1_1_1" not in ids:
            recs.append(record("1003_1_1_1", CHAINS["1003_1_1_1"][1]))
        if tier in ("moderate", "strict") and "2003_1_1_1" not in ids:
            recs.append(record("2003_1_1_1", CHAINS["2003_1_1_1"][1]))
        for r in recs:
            if r["ID"] == "2003_1_1_1":
                r["gscores"] = [
                    None if g is None else round(abs(g - 0.4), 4) for g in r["gscores"]
                ]
            if r["ID"] == "1001_1_1_1":
                r["seq"] += "*"
                r["gscores"] = r["gscores"][:-1] + [0.5, None]
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return prev


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("inputs")
    bundle = write_bundle(root / "bundle")
    snapshot = write_snapshot(root / "bmrb_entries_2026-11-11")
    results = write_results(root / "framework")
    parquet = build_release_parquet(
        bundle,
        root / "trizod_dataset.parquet",
        snapshot_dir=snapshot,
        results_dir=results,
        overlaps={
            "chezod117": set(),
            "chezod1325": {"2001_1_1_1"},
            "chezod1174": set(),
        },
    )
    shifts = write_shifts(root / "trizod_shifts.parquet")
    earlier = root / "earlier.fasta"
    # an earlier test set: one sequence still in the test set, one now training
    earlier.write_text(
        f">old1\n{_seq(2001)}\n>old2\n{_seq(1003)}\n>old3\nMMMMMMMMMMMM\n"
    )
    reviewed = root / "reviewed.tsv"
    reviewed.write_text(
        "chain_id\tissue\tevidence\n2001_1_1_1\treviewed_issue\ta reviewed note\n"
    )
    validation = root / "VALIDATION.json"
    validation.write_text('{"audit": "synthetic"}\n')
    return {
        "root": root,
        "bundle": bundle,
        "previous": write_previous_bundle(bundle, root / "previous"),
        "snapshot": snapshot,
        "results": results,
        "parquet": parquet,
        "shifts": shifts,
        "reviewed": reviewed,
        "validation": validation,
        "specs": [("pin_old_3", f"fasta:{earlier}", "an earlier pin")],
    }


def _assemble(inputs, out, **kw):
    kw.setdefault("reviewed", inputs["reviewed"])
    kw.setdefault("validation", inputs["validation"])
    kw.setdefault("previous_build", inputs["previous"])
    kw.setdefault("previous_label", "the synthetic earlier build")
    return finalize.assemble(
        inputs["parquet"],
        inputs["bundle"],
        out,
        shifts=inputs["shifts"],
        bmrb_dir=inputs["snapshot"],
        crosswalk_specs=inputs["specs"],
        **kw,
    )


@pytest.fixture(scope="module")
def release(inputs):
    out = inputs["root"] / "release"
    manifest = _assemble(inputs, out)
    return {"out": out, "manifest": manifest}


def _files(out):
    return {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}


def test_every_hand_over_file_is_assembled(release):
    files = _files(release["out"])
    assert files >= EXPECTED_FILES, sorted(EXPECTED_FILES - files)


def test_manifest_lists_every_file_with_checksum_size_and_counts(release):
    out, manifest = release["out"], release["manifest"]
    listed = set(manifest["files"])
    assert listed == _files(out) - {"MANIFEST.json", "SHA256SUMS"}
    assert manifest["n_files"] == len(_files(out))
    for rel, entry in manifest["files"].items():
        data = (out / rel).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest(), rel
        assert entry["bytes"] == len(data), rel
    files = manifest["files"]
    assert files["trizod_dataset.parquet"]["n_rows"] == len(CHAINS)
    assert files["trizod_dataset.parquet"]["n_chains"] == len(CHAINS)
    assert files["trizod_shifts.parquet"]["n_rows"] == 2 * len(CHAINS)
    assert files["test/TriZOD_test_set.fasta"]["n_sequences"] == 2
    assert files["disorder_trizod.fasta"]["n_sequences"] == 5
    assert files["udonpred/trizod_tolerant/test.jsonl"]["n_records"] == 2
    assert files["crosswalk/test_set_crosswalk.tsv"]["n_rows"] == 4
    assert files["class_balance.tsv"]["n_rows"] == 6
    assert files["bmrb_files.sha256"]["n_files"] == 3
    assert manifest["total_bytes"] == sum(f["bytes"] for f in files.values())
    assert "MANIFEST.json and SHA256SUMS" in manifest["total_bytes_covers"]
    assert manifest["validation"]["file"] == "VALIDATION.json"
    assert manifest["validation"]["paths_rewritten"] is False


def test_manifest_records_the_release_identity(release, inputs):
    m = release["manifest"]
    assert m["version"] == ri.DATASET_VERSION
    assert m["license"]["data"] == "CC-BY-4.0" and m["license"]["code"] == "MIT"
    assert m["test_set"]["name"] == "TriZOD-test v1.0"
    assert m["test_set"]["n_chains"] == 2 and m["test_set"]["n_retired"] == 1
    assert m["cuts"]["values"] == [0.15, 0.35, 0.70]
    assert m["code"]["build"]["commit"] == "c" * 40
    assert m["code"]["package_pipeline_version"] == "trizod-cccccc"
    # the scoring commit is recorded apart from the build/finalize commits
    scoring = m["code"]["scoring"]
    assert scoring["pipeline_version"] == "trizod-cccccc"
    assert scoring["commit"] == "cccccc" and "package bundle" in scoring["scope"]
    assert m["code"]["finalize"] is None or "commit" in m["code"]["finalize"]
    tp = m["threshold_protocol"]
    assert tp["framework_commit"]["commit"] == "f" * 40
    assert tp["results_json_sha256"] == ri.sha256_file(
        inputs["results"] / "results.json"
    )
    assert tp["rule"] == "disordered iff G >= t"
    snap = m["bmrb_snapshot"]
    assert snap["name"] == "bmrb_entries_2026-11-11"
    assert snap["n_entries"] == 3 and snap["n_files"] == 3
    assert snap["entry_list_sha256"] == "a" * 64
    assert snap["files_table_sha256"] == m["files"]["bmrb_files.sha256"]["sha256"]
    assert "xref_sources" in m
    assert m["leakage"]["rule"] == "either"
    assert m["parquet"]["split_counts"]["test_trizod"] == 2
    assert m["parquet"]["training_representatives"]["tolerant"] == 3
    assert m["parquet"]["retired_by_reason"] == {"not_strict": 1}


def test_sha256sums_checks_every_file(release):
    out = release["out"]
    lines = (out / "SHA256SUMS").read_text().splitlines()
    listed = {}
    for line in lines:
        sha, rel = line.split("  ", 1)
        listed[rel] = sha
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == sha, rel
    assert set(listed) == _files(out) - {"SHA256SUMS"}


def test_bmrb_files_table_is_sha256sum_format_in_numeric_order(release, inputs):
    lines = (release["out"] / "bmrb_files.sha256").read_text().splitlines()
    paths = [line.split("  ", 1)[1] for line in lines]
    assert paths == [
        "bmr1001/bmr1001_3.str",
        "bmr2001/bmr2001_3.str",
        "bmr3001/bmr3001_3.str",
    ]
    for line in lines:
        sha, rel = line.split("  ", 1)
        assert sha == ri.sha256_file(inputs["snapshot"] / rel)


def test_assembly_is_deterministic(inputs, release):
    """No file names its own output path any more, so two output folders hold
    byte-identical deposits."""
    out = inputs["root"] / "release_again"
    _assemble(inputs, out)
    first = release["out"]
    assert _files(out) == _files(first)
    for rel in sorted(_files(first)):
        assert (out / rel).read_bytes() == (first / rel).read_bytes(), rel
    # into the same path again: byte-identical everywhere
    again = _assemble(inputs, first, force=True)
    assert again == release["manifest"]


def test_rendered_docs_carry_this_builds_identity(release):
    out = release["out"]
    for name in ("README.md", "datasheet.md", "CHANGELOG.md"):
        text = (out / name).read_text()
        assert "{{" not in text and "<!-- Template" not in text, name
        assert "0.4.0" in text, name
        assert "TriZOD-test v1.0" in text, name
        assert "CC BY 4.0" in text, name
        assert not re.search(r"leakage[- ]free|no train/test leakage", text, re.I), name
    readme = (out / "README.md").read_text()
    assert "TriZOD-test v1.0 (n = 2; 1 retired chains)" in readme
    assert "scored by `cccccc` (trizod-cccccc" in " ".join(readme.split())
    assert "`known_issues.tsv`" in readme and "`VALIDATION.json`" in readme
    assert "scores.json" not in readme
    assert "a filter on `entry_id` reads one row group" in " ".join(readme.split())
    assert "**Rows:** 10 scored protein chains" in readme
    assert "π = 0.186 [0.141, 0.244]" in readme
    assert "| train, tolerant | 3 |" in readme  # class-balance table
    assert "`pin_old_3` | 3 | 1 | 1 |" in readme  # crosswalk overlap table
    assert "Trizod348" in readme and "v0.3.0" in readme
    flat = " ".join(readme.split())
    assert "(1 and 2 of the 2 test chains)" in flat  # CheZOD clean subsets
    changelog = (out / "CHANGELOG.md").read_text()
    assert "| strict | 3,271 | 5 | 1,381 | 1 |" in changelog
    assert "not assessed" in changelog  # no --pkl-dir, --snapshot-drift, ...
    assert "`not_strict` (1): it no longer meets the strict tier" in changelog
    assert "`membrane_mimetic_sample`) | moderate, strict | 1 |" in changelog
    assert "1 not_strict" in changelog


def test_views_carry_the_cuts_and_the_test_set_name(release):
    out = release["out"]
    deploy = json.loads((out / "disorder_trizod.summary.json").read_text())
    assert deploy["cuts"]["values"] == [0.15, 0.35, 0.70]
    assert deploy["dataset_version"] == ri.DATASET_VERSION
    assert deploy["test_set"].startswith("TriZOD-test v1.0")
    readme = (out / "udonpred" / "README.md").read_text()
    assert "TriZOD-test v1.0 (n = 2; 1 retired chains)" in readme
    assert "G >= 0.35" in readme and "k >= 10" in readme
    assert "## Class balance" in readme
    summary = json.loads((out / "udonpred" / "export_summary.json").read_text())
    assert summary["test_set"]["name"] == "TriZOD-test v1.0"
    assert "calibration" in summary["source"]["metadata"]


def test_the_crosswalk_names_the_release_relatively(release):
    payload = json.loads(
        (release["out"] / "crosswalk" / "test_set_crosswalk.json").read_text()
    )
    assert payload["release"]["path"] == "trizod_dataset.parquet"


def test_refuses_a_parquet_without_calibration(inputs, tmp_path):
    parquet = build_release_parquet(
        inputs["bundle"], tmp_path / "p.parquet", snapshot_dir=inputs["snapshot"]
    )
    kw = {
        "shifts": inputs["shifts"],
        "bmrb_dir": inputs["snapshot"],
        "crosswalk_specs": inputs["specs"],
        "reviewed": inputs["reviewed"],
    }
    with pytest.raises(SystemExit, match="calibration"):
        finalize.assemble(parquet, inputs["bundle"], tmp_path / "out", **kw)
    manifest = finalize.assemble(
        parquet, inputs["bundle"], tmp_path / "out", allow_uncalibrated=True, **kw
    )
    assert manifest["threshold_protocol"] is None
    readme = (tmp_path / "out" / "README.md").read_text()
    assert "not recorded" in readme


def test_refuses_a_bundle_whose_test_set_differs(inputs, tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(inputs["bundle"], bundle)
    (bundle / "test" / "TriZOD_test_set.fasta").write_text(">9999_1_1_1\nAAAA\n")
    with pytest.raises(SystemExit, match="not built from this bundle"):
        finalize.assemble(
            inputs["parquet"],
            bundle,
            tmp_path / "out",
            shifts=inputs["shifts"],
            bmrb_dir=inputs["snapshot"],
            crosswalk_specs=inputs["specs"],
        )


def test_refuses_a_snapshot_with_another_entry_count(inputs, tmp_path):
    snap = write_snapshot(tmp_path / "bmrb", entry_ids=("1001",))
    with pytest.raises(SystemExit, match="entry files"):
        finalize.assemble(
            inputs["parquet"],
            inputs["bundle"],
            tmp_path / "out",
            shifts=inputs["shifts"],
            bmrb_dir=snap,
            crosswalk_specs=inputs["specs"],
        )
    # refused before the output folder was touched
    assert not (tmp_path / "out").exists()


def test_refuses_a_non_empty_output_without_force(inputs, release):
    with pytest.raises(SystemExit, match="not empty"):
        _assemble(inputs, release["out"])


def test_render_template_refuses_unknown_placeholders():
    with pytest.raises(ValueError, match="no value"):
        finalize.render_template("v{{version}} {{nope}}", {"version": "1"})
    text = "<!-- Template: note -->\nv{{ version }}"
    assert finalize.render_template(text, {"version": "1"}) == "v1"


def test_calibration_input_with_the_same_data_is_accepted(inputs, tmp_path):
    """The framework calibrates on the Parquet before its metadata is stamped:
    the same data under other metadata is the expected case."""
    unstamped = build_release_parquet(inputs["bundle"], tmp_path / "input.parquet")
    unstamped_overlaps = build_release_parquet(
        inputs["bundle"],
        tmp_path / "input2.parquet",
        overlaps={
            "chezod117": set(),
            "chezod1325": {"2001_1_1_1"},
            "chezod1174": set(),
        },
    )
    results = write_results(tmp_path / "fw", release=unstamped_overlaps)
    parquet = build_release_parquet(
        inputs["bundle"],
        tmp_path / "final.parquet",
        snapshot_dir=inputs["snapshot"],
        results_dir=results,
        overlaps={
            "chezod117": set(),
            "chezod1325": {"2001_1_1_1"},
            "chezod1174": set(),
        },
    )
    kw = {
        "shifts": inputs["shifts"],
        "bmrb_dir": inputs["snapshot"],
        "crosswalk_specs": inputs["specs"],
        "reviewed": inputs["reviewed"],
    }
    # the Parquet records the input by file name only (no machine path) ...
    assert (
        finalize.cb.parse_calibration(finalize.read_parquet(parquet)[1]["calibration"])[
            "input_parquet"
        ]
        == "input2.parquet"
    )
    manifest = finalize.assemble(parquet, inputs["bundle"], tmp_path / "out0", **kw)
    assert manifest["threshold_protocol"]["input_data_identical_to_release"] is None
    # ... so the input is named with --calibration-input
    manifest = finalize.assemble(
        parquet,
        inputs["bundle"],
        tmp_path / "out",
        calibration_input=unstamped_overlaps,
        **kw,
    )
    assert manifest["threshold_protocol"]["input_data_identical_to_release"] is True

    # other data (the overlap flags were not computed): refused
    results = write_results(tmp_path / "fw_other", release=unstamped)
    other = build_release_parquet(
        inputs["bundle"],
        tmp_path / "final_other.parquet",
        snapshot_dir=inputs["snapshot"],
        results_dir=results,
        overlaps={
            "chezod117": set(),
            "chezod1325": {"2001_1_1_1"},
            "chezod1174": set(),
        },
    )
    kw["calibration_input"] = unstamped
    with pytest.raises(SystemExit, match="data differ"):
        finalize.assemble(other, inputs["bundle"], tmp_path / "out2", **kw)
    manifest = finalize.assemble(
        other,
        inputs["bundle"],
        tmp_path / "out2",
        allow_calibration_mismatch=True,
        force=True,
        **kw,
    )
    assert manifest["threshold_protocol"]["input_data_identical_to_release"] is False


def test_calibration_input_not_on_this_machine_is_recorded_unchecked(release):
    tp = release["manifest"]["threshold_protocol"]
    assert tp["input_data_identical_to_release"] is None


# --------------------------------------------------------------------------- #
# provenance: no machine paths; changes and known issues (audit 2026-11-17b)
# --------------------------------------------------------------------------- #


def test_no_shipped_file_names_a_path_of_this_machine(release, inputs):
    out = release["out"]
    root = str(inputs["root"].resolve())
    for p in out.rglob("*"):
        if p.is_file() and p.suffix in (".json", ".md", ".tsv", ".cff", ""):
            assert root not in p.read_text(), p
    assert finalize.machine_paths(out, [inputs["root"]]) == []
    cw = json.loads((out / "crosswalk" / "test_set_crosswalk.json").read_text())
    source = {s["name"]: s["source"] for s in cw["sets"]}["pin_old_3"]
    assert source["spec"] == "fasta:earlier.fasta" and source["path"] == "earlier.fasta"
    deploy = json.loads((out / "disorder_trizod.summary.json").read_text())
    assert deploy["out"] == "disorder_trizod.fasta"
    assert deploy["scored_dir"] == "<package bundle>/scores"
    udon = json.loads((out / "udonpred" / "export_summary.json").read_text())
    assert udon["source"]["path"] == "trizod_dataset.parquet"


def test_machine_paths_finds_an_absolute_path(tmp_path):
    (tmp_path / "a.json").write_text('{"p": "/Users/someone/data/x.parquet"}')
    (tmp_path / "b.md").write_text(
        "relative data/raw/bmrb_entries and https://x.org/home/y"
    )
    hits = finalize.machine_paths(tmp_path, [])
    assert hits == ["a.json: /Users/"]


def test_scrub_json_paths_keeps_prose_and_relative_paths(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(
        json.dumps(
            {
                "path": str(tmp_path / "out" / "trizod_dataset.parquet"),
                "spec": f"fasta:{tmp_path / 'data' / 'external' / 't.fasta'}",
                "elsewhere": "/opt/other/thing.fasta",
                "url": "https://github.com/tsenoner/trizod",
                "embedded": f"calibrated on {tmp_path / 'data' / 'x.parquet'} today",
                "prose": "train/valid/test, see data/raw",
                "nested": [{"p": str(tmp_path / "bundle" / "scores")}],
            }
        )
    )
    assert finalize.scrub_json_paths(
        f, [tmp_path / "out", (tmp_path / "bundle", "<package bundle>/"), tmp_path]
    )
    got = json.loads(f.read_text())
    assert got == {
        "path": "trizod_dataset.parquet",
        "spec": "fasta:data/external/t.fasta",
        "elsewhere": "thing.fasta",
        "url": "https://github.com/tsenoner/trizod",
        "embedded": "calibrated on data/x.parquet today",
        "prose": "train/valid/test, see data/raw",
        "nested": [{"p": "<package bundle>/scores"}],
    }


def _known_issues(out):
    lines = (out / "known_issues.tsv").read_text().splitlines()
    assert lines[0] == "chain_id\tissue\tevidence"
    return [tuple(line.split("\t")) for line in lines[1:]]


def test_known_issues_lists_tier_moves_rescores_and_sequence_changes(release):
    out = release["out"]
    rows = _known_issues(out)
    by = {(cid, issue): ev for cid, issue, ev in rows}
    assert by[("1003_1_1_1", "tier_move")] == "left moderate (membrane)"
    assert "left strict (273 K)" in by[("2003_1_1_1", "tier_move")]
    assert "left moderate (273 K re-score)" in by[("2003_1_1_1", "tier_move")]
    assert by[("2003_1_1_1", "temperature_placeholder_rescore")].startswith(
        "deposited 273 K read as not deposited, re-scored at 298 K: max |ΔG| 0.400"
    )
    assert by[("1001_1_1_1", "sequence_alphabet_change")].startswith(
        "sequence 31 → 30 residues"
    )
    assert by[("2001_1_1_1", "reviewed_issue")] == "a reviewed note (split test_trizod)"
    # the pin provenance's trimmed chains, with their status in this release
    trimmed = [ev for cid, issue, ev in rows if issue == "pin_trimmed"]
    assert trimmed and all("now not released" in ev for ev in trimmed)
    # sorted, unique, and listed in MANIFEST
    assert rows == sorted(set(rows), key=lambda r: (ki_key(r[0]), r[1], r[2]))
    m = release["manifest"]
    assert m["known_issues"]["by_issue"]["tier_move"] == 2
    assert m["files"]["known_issues.tsv"]["n_rows"] == len(rows)


def ki_key(cid):
    from trizod.dataset.known_issues import _id_key

    return _id_key(cid)


def test_changelog_states_the_tier_moves_and_rescore_statistics(release):
    text = " ".join((release["out"] / "CHANGELOG.md").read_text().split())
    assert "Tier moves against the synthetic earlier build" in text
    assert (
        "| moderate | 8 → 6 | 2: 273 K re-score 1 (2003_1_1_1), membrane 1 "
        "(1003_1_1_1) | 0 |"
    ) in text
    assert "| strict | 6 → 5 | 1: 273 K 1 (2003_1_1_1) | 0 |" in text
    assert "1 of the 1 such chains in both builds changed their labels" in text
    assert "1001_1_1_1 (sequence 31 → 30 residues" in text
    assert "2001_1_1_1: a reviewed note (split test_trizod)" in text


def test_retired_reasons_are_explained_generically():
    from trizod.dataset import release_info as ri

    text = ri.retired_semantics(["not_strict", "a_new_reason"])
    for reason in ri.RETIRED_REASONS:
        assert reason in text
    assert "a_new_reason (no description recorded" in text
    assert finalize._retired_explained({"a_new_reason": 2}).startswith(
        "`a_new_reason` (2): no description recorded"
    )
