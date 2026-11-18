"""``croissant.json``:
:mod:`trizod.dataset.croissant` on its own; the assembled deposit's file is
tested in ``tests/test_finalize.py``."""

from __future__ import annotations

import json

import pytest

from trizod.dataset import croissant as cr
from trizod.dataset import release_info as ri

COLUMNS_MD = """# column dictionary

| # | Column | Arrow type | Nullable | Meaning | Null / NaN rule |
|--:|---|---|---|---|---|
| 1 | `id` | `string` | no | Chain ID | never null |
| 2 | `gscores` | `list<float32>` | no | Per-residue G; `a | b` stays | NaN where masked |
| 3 | `xrefs` | `list<struct<db: string, identity: float32>>` | yes | Links | null when unread |
"""

DATASHEET = """# datasheet

## 2. Composition

**Is anything confidential or about people?** No. Proteins only.

## 3. Collection

**Source.** BMRB entries, snapshot `x`.

## 4. Preprocessing and labelling

1. **Parse** every entry.
2. **Score** each residue: the G-score.

**Cuts on G** (metadata): ordered G < 0.15.

| a | b |
|---|---|

**Raw data.** The deposited shifts ship.

## 5. Uses

**Intended.** Training disorder predictors.

- Train on `gscores`.

## 6. Limitations

- Labels depend on
  the conditions.
- Z grows with k.

## 8. Maintenance

The test set may only shrink.
"""


def test_the_column_dictionary_is_read_with_types_and_rules():
    cols = cr.read_columns(COLUMNS_MD)
    assert [c["name"] for c in cols] == ["id", "gscores", "xrefs"]
    assert cols[1]["meaning"] == "Per-residue G; `a | b` stays"
    assert cols[2]["nullable"] and not cols[0]["nullable"]
    assert cr.data_type("string") == ("sc:Text", False)
    assert cr.data_type("list<int32>") == ("sc:Integer", True)
    assert cr.data_type("list<bool>") == ("sc:Boolean", True)
    assert cr.data_type(cols[2]["arrow_type"]) == (None, True)
    assert cr.struct_fields(cols[2]["arrow_type"]) == [
        ("db", "sc:Text"),
        ("identity", "sc:Float"),
    ]
    with pytest.raises(ValueError):
        cr.data_type("decimal128")


def test_the_chains_record_set_maps_every_column():
    rs = cr.chains_record_set(cr.read_columns(COLUMNS_MD))
    by = {f["name"]: f for f in rs["field"]}
    assert rs["key"] == {"@id": "chains/id"}
    assert by["gscores"]["repeated"] is True and "repeated" not in by["id"]
    assert "Null / NaN: NaN where masked" in by["gscores"]["description"]
    sub = by["xrefs"]["subField"]
    assert [s["@id"] for s in sub] == ["chains/xrefs/db", "chains/xrefs/identity"]
    assert sub[0]["source"]["transform"] == {"jsonPath": "db"}
    assert sub[0]["source"]["extract"] == {"column": "xrefs"}


def test_the_datasheet_answers_the_rai_fields():
    rai = cr.rai_fields(DATASHEET)
    assert rai["rai:dataPreprocessingProtocol"] == [
        "Parse every entry.",
        "Score each residue: the G-score.",
    ]
    assert rai["rai:dataAnnotationProtocol"].startswith("Score each residue")
    assert "Cuts on G" in rai["rai:dataAnnotationProtocol"]
    assert rai["rai:dataLimitations"] == [
        "Labels depend on the conditions.",
        "Z grows with k.",
    ]
    assert rai["rai:dataUseCases"][0] == "Intended. Training disorder predictors."
    assert rai["rai:dataReleaseMaintenancePlan"] == "The test set may only shrink."
    assert rai["rai:dataCollectionRawData"].startswith("Source. BMRB")
    assert "Raw data." in rai["rai:dataCollectionRawData"]
    assert rai["rai:personalSensitiveInformation"][0].startswith("Is anything")
    with pytest.raises(ValueError, match="does not answer"):
        cr.rai_fields("# empty\n")


def test_the_release_date_comes_from_citation_cff():
    assert cr.date_released('cff-version: 1.2.0\ndate-released: "2026-12-01"\n') == (
        "2026-12-01"
    )
    assert cr.date_released("cff-version: 1.2.0\n") is None


def _deposit(tmp_path):
    out = tmp_path / "dep"
    (out / "test").mkdir(parents=True)
    (out / "columns.md").write_text(COLUMNS_MD)
    (out / "datasheet.md").write_text(DATASHEET)
    (out / "zenodo.json").write_text(
        json.dumps(
            {
                "title": "TriZOD",
                "description": "d",
                "creators": [{"name": "Doe, Jane", "orcid": "0000-0000-0000-0001"}],
                "keywords": ["NMR"],
            }
        )
    )
    (out / "test" / "a.fasta").write_text(">a\nAC\n")
    (out / "MANIFEST.json").write_text("{}")
    (out / "SHA256SUMS").write_text("")
    return out


def test_the_distribution_lists_every_file_but_the_self_describing_ones(tmp_path):
    out = _deposit(tmp_path)
    path = cr.write(out, version="9.9.9")
    doc = json.loads(path.read_text())
    listed = {d["contentUrl"]: d for d in doc["distribution"]}
    assert set(listed) == {"columns.md", "datasheet.md", "zenodo.json", "test/a.fasta"}
    assert listed["test/a.fasta"]["sha256"] == ri.sha256_file(out / "test" / "a.fasta")
    assert listed["test/a.fasta"]["encodingFormat"] == "text/x-fasta"
    assert doc["creator"] == [
        {
            "@type": "sc:Person",
            "name": "Jane Doe",
            "affiliation": "",
            "identifier": "https://orcid.org/0000-0000-0000-0001",
        }
    ]
    assert doc["url"] == f"https://doi.org/{ri.CONCEPT_DOI}"
    assert "datePublished" not in doc
    assert cr.check(path) == []
    (out / "test" / "a.fasta").write_text(">a\nACD\n")
    (out / "test" / "b.fasta").write_text(">b\nAC\n")
    problems = cr.check(path)
    assert "sha256 differs: test/a.fasta" in problems
    assert "not listed: test/b.fasta" in problems
    # deterministic
    first = path.read_bytes()
    assert cr.write(out, version="9.9.9").read_bytes() != first  # a.fasta changed
    again = cr.write(out, version="9.9.9").read_bytes()
    assert cr.write(out, version="9.9.9").read_bytes() == again


def test_file_descriptions_state_the_release_as_it_is():
    """known_issues.tsv lists known issues, not changes against another build;
    the residual-similarity table describes the split."""
    known = cr.describe_file("known_issues.tsv")
    assert "known issue" in known and "change" not in known
    sim = cr.describe_file("test/test_residual_similarity.tsv")
    assert "closest tolerant training representative" in sim
    assert "baseline" not in sim and "nearest" not in sim
    assert cr.describe_file("known_issues_inputs/construct_caveats.tsv").endswith(
        "construct_caveats.tsv"
    )
