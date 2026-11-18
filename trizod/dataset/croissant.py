"""``croissant.json``: the deposit described in MLCommons Croissant 1.0 with the
Responsible-AI extension (release validation 2026-11-17, G41).

:func:`build` writes it from what ``trizod dataset finalize`` has already put
into the deposit, so it says nothing the shipped files do not:

* the dataset: name, description, keywords, creators (with ORCID iDs) and
  licence from ``zenodo.json``; version and DOI from the release;
* ``distribution``: one ``cr:FileObject`` per file of the deposit with its
  SHA-256 and size, except ``croissant.json``, ``MANIFEST.json`` and
  ``SHA256SUMS``, which describe the others (``MANIFEST.json`` and
  ``SHA256SUMS`` in turn cover ``croissant.json``);
* ``recordSet``: ``splits`` (a ``cr:Split`` enumeration of the values of the
  ``split`` column) and ``chains``, one field per column of
  ``trizod_dataset.parquet`` with the column dictionary's meaning, null rule
  and type (``columns.md``); per-residue lists are ``repeated`` fields, and
  ``chains/split`` references ``splits/name``;
* the RAI fields the datasheet answers (``datasheet.md``, sections 2-8): data
  collection and raw data, preprocessing, the computed labels, intended uses,
  limitations, personal or sensitive information, maintenance.

Deterministic: the same deposit gives byte-identical output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from trizod.dataset import release_info as ri

CROISSANT = "croissant.json"
#: Files the distribution leaves out: this file, and the two checksum lists
#: that cover it.
SELF_DESCRIBING = (CROISSANT, "MANIFEST.json", "SHA256SUMS")
MAIN_PARQUET = "trizod_dataset.parquet"
CONFORMS_TO = (
    "http://mlcommons.org/croissant/1.0",
    "http://mlcommons.org/croissant/RAI/1.0",
)
#: The standard Croissant 1.0 ``@context`` (as mlcroissant writes it).
CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}
#: Arrow scalar type (as ``columns.md`` writes it) -> Croissant data type.
ARROW_TYPES = {
    "string": "sc:Text",
    "bool": "sc:Boolean",
    "int8": "sc:Integer",
    "int16": "sc:Integer",
    "int32": "sc:Integer",
    "int64": "sc:Integer",
    "float32": "sc:Float",
    "float64": "sc:Float",
    "double": "sc:Float",
}
#: File suffix -> encoding format (MIME type).
ENCODING = {
    ".parquet": "application/x-parquet",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".jsonl": "application/jsonlines",
    ".md": "text/markdown",
    ".fasta": "text/x-fasta",
    ".cff": "application/x-yaml",
    ".sha256": "text/plain",
}
#: What each split value means (the ``split`` column; metadata ``split_values``).
SPLITS = (
    (
        "train",
        "cr:TrainingSplit",
        "Training representatives: the training set of tier T is split == "
        "'train' with train_tier at T or stricter (tiers nest: unfiltered ⊃ "
        "tolerant ⊃ moderate ⊃ strict); the default tier is tolerant",
    ),
    (
        "redundant",
        None,
        "Members of a training cluster that are not its representative; not "
        "for training next to their representative, never for testing",
    ),
    (
        "test_trizod",
        "cr:TestSplit",
        f"{ri.TEST_SET_NAME}: the TriZOD test set, drawn from the strict tier "
        "and held out from training under the leakage rule",
    ),
    (
        "test_retired",
        None,
        "Former test chains that no longer qualify (retired_reason); not "
        "tested on, kept out of training as leakage targets",
    ),
    (
        "test_chezod117",
        "cr:TestSplit",
        "CheZOD117, the external benchmark (BMRB-mapped chains), held out "
        "from training under the leakage rule",
    ),
    (
        "excluded",
        None,
        "Scored chains outside every training and test set (bound complexes, "
        "chains under 20 residues, duplicates of a better record, leakage "
        "with a test chain)",
    ),
)

_TABLE_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|")


# --------------------------------------------------------------------------- #
# the column dictionary
# --------------------------------------------------------------------------- #


def _cells(line: str) -> list[str]:
    """The cells of a markdown table row (``|`` inside backticks kept)."""
    cells, cur, tick = [], "", False
    for ch in line.strip().strip("|"):
        if ch == "`":
            tick = not tick
        if ch == "|" and not tick:
            cells.append(cur.strip())
            cur = ""
        else:
            cur += ch
    cells.append(cur.strip())
    return cells


def read_columns(columns_md: str) -> list[dict]:
    """``[{"name", "arrow_type", "nullable", "meaning", "null_rule"}]`` from
    the column dictionary (``columns.md``), in column order."""
    out = []
    for line in columns_md.splitlines():
        if not _TABLE_ROW.match(line):
            continue
        _, name, arrow, nullable, meaning, null_rule = _cells(line)[:6]
        out.append(
            {
                "name": name.strip("`"),
                "arrow_type": arrow.strip("`"),
                "nullable": nullable == "yes",
                "meaning": meaning,
                "null_rule": null_rule,
            }
        )
    return out


def data_type(arrow_type: str) -> tuple[str | None, bool]:
    """``(Croissant data type, repeated)`` of an Arrow type of ``columns.md``;
    the data type is None for a struct (its keys are sub-fields,
    :func:`struct_fields`)."""
    t = arrow_type.strip()
    repeated = t.startswith("list<")
    if repeated:
        t = t[len("list<") : -1]
    if t.startswith("struct<"):
        return None, repeated
    if t not in ARROW_TYPES:
        raise ValueError(f"no Croissant data type for Arrow type {arrow_type!r}")
    return ARROW_TYPES[t], repeated


def struct_fields(arrow_type: str) -> list[tuple[str, str]]:
    """``[(key, Croissant data type)]`` of a (list of) struct type."""
    m = re.search(r"struct<([^<>]*)>", arrow_type)
    if not m:
        return []
    out = []
    for part in m.group(1).split(","):
        key, _, t = part.strip().partition(":")
        dtype, _ = data_type(t.strip())
        out.append((key.strip(), dtype))
    return out


# --------------------------------------------------------------------------- #
# the datasheet as RAI fields
# --------------------------------------------------------------------------- #


def _plain(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    return " ".join(text.split())


def datasheet_sections(datasheet_md: str) -> dict[int, str]:
    """``{section number: body}`` of the ``## N. Title`` sections."""
    out, cur, buf = {}, None, []
    for line in datasheet_md.splitlines():
        m = re.match(r"^## (\d+)\. ", line)
        if m:
            if cur is not None:
                out[cur] = "\n".join(buf)
            cur, buf = int(m.group(1)), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf)
    return out


def blocks(body: str) -> list[str]:
    """Paragraphs and list items of a markdown section as plain one-line
    strings; tables are dropped."""
    out, cur = [], []

    def flush():
        if cur:
            text = _plain(" ".join(cur))
            if text:
                out.append(text)
            cur.clear()

    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith("|"):
            flush()
            continue
        if re.match(r"^(- |\d+\. )", s):
            flush()
            s = re.sub(r"^(- |\d+\. )", "", s)
        cur.append(s)
    flush()
    return out


def _starting(items: list[str], prefix: str) -> list[str]:
    return [i for i in items if i.startswith(prefix)]


def rai_fields(datasheet_md: str) -> dict:
    """The RAI properties the datasheet answers."""
    sec = datasheet_sections(datasheet_md)
    collection = blocks(sec.get(3, ""))
    prep = blocks(sec.get(4, ""))
    uses = blocks(sec.get(5, ""))
    composition = blocks(sec.get(2, ""))
    steps = [b for b in prep if not b.startswith(("Cuts on G", "Threshold", "Raw"))]
    labels = [b for b in steps if b.startswith("Score")] + _starting(prep, "Cuts on G")
    fields = {
        "rai:dataCollection": " ".join(collection),
        "rai:dataCollectionType": ["Secondary Data analysis"],
        "rai:dataCollectionRawData": " ".join(
            _starting(collection, "Source.") + _starting(prep, "Raw data.")
        ),
        "rai:dataPreprocessingProtocol": steps,
        "rai:dataAnnotationProtocol": " ".join(labels),
        "rai:dataUseCases": uses,
        "rai:dataLimitations": blocks(sec.get(6, "")),
        "rai:personalSensitiveInformation": _starting(
            composition, "Is anything confidential"
        ),
        "rai:dataReleaseMaintenancePlan": " ".join(blocks(sec.get(8, ""))),
    }
    missing = [k for k, v in fields.items() if not v]
    if missing:
        raise ValueError(f"datasheet.md does not answer {missing}")
    return fields


# --------------------------------------------------------------------------- #
# dataset, distribution and record sets
# --------------------------------------------------------------------------- #


def _person(creator: dict) -> dict:
    family, _, given = creator["name"].partition(", ")
    out = {
        "@type": "sc:Person",
        "name": f"{given} {family}".strip(),
        "affiliation": creator.get("affiliation", ""),
    }
    if creator.get("orcid"):
        out["identifier"] = f"https://orcid.org/{creator['orcid']}"
    return out


def cite_as(zenodo: dict, version: str, doi: str) -> str:
    authors = " and ".join(c["name"] for c in zenodo.get("creators", ()))
    key = "trizod_" + version.replace(".", "_")
    return (
        f"@dataset{{{key},\n  author = {{{authors}}},\n"
        f"  title = {{{zenodo.get('title', 'TriZOD')}}},\n"
        f"  version = {{{version}}},\n  publisher = {{Zenodo}},\n"
        f"  doi = {{{doi}}},\n  url = {{https://doi.org/{doi}}}\n}}"
    )


def describe_file(rel: str) -> str:
    """One line on what a deposit file holds."""
    name = Path(rel).name
    fixed = {
        MAIN_PARQUET: "The dataset: one row per scored protein chain, per-residue "
        "G-score (the label), Z-score, shift count k and mask lists aligned to "
        "the sequence; split, tier, condition, flag and cross-reference columns "
        "(columns.md)",
        "trizod_shifts.parquet": "Every assigned chemical shift on a canonical "
        "residue, raw and re-referenced; joins on id",
        "disorder_trizod.fasta": "SETH/ODiNPred-style FASTA view: tolerant "
        "training set and the test set, G-score targets (999.0 where masked)",
        "disorder_trizod.summary.json": "Counts of disorder_trizod.fasta",
        "known_issues.tsv": "Every chain with a known issue or a change named "
        "in the CHANGELOG, with its evidence",
        "class_balance.tsv": "Per-split label distribution",
        "bmrb_files.sha256": "SHA-256 of every BMRB entry file of the snapshot",
        "columns.md": "The column dictionary of trizod_dataset.parquet",
        "datasheet.md": "The datasheet",
        "README.md": "The deposit's README",
        "CHANGELOG.md": "Changes since the previous version and the known issues",
        "LICENSE.md": "Licence of the data and the code",
        "CITATION.cff": "How to cite the dataset",
        "zenodo.json": "Zenodo deposit metadata",
        "VALIDATION.json": "Results of the release audit, check by check",
        "VALIDATION.md": "The human decisions the release audit records",
    }
    if rel in fixed:
        return fixed[rel]
    top = rel.split("/")[0]
    if top == "udonpred":
        return f"UdonPred v2 view generated from the Parquet: {rel}"
    if top == "test":
        return f"Test-set file: {name}"
    if top == "crosswalk":
        return f"Test-set crosswalk across earlier TriZOD test sets: {name}"
    if top == "known_issues_inputs":
        return f"Per-chain input table cited by known_issues.tsv: {name}"
    return name


def distribution(out: Path) -> list[dict]:
    files = []
    for p in sorted(Path(out).rglob("*")):
        rel = p.relative_to(out).as_posix()
        if not p.is_file() or rel in SELF_DESCRIBING:
            continue
        files.append(
            {
                "@type": "cr:FileObject",
                "@id": rel,
                "name": rel,
                "description": describe_file(rel),
                "contentUrl": rel,
                "contentSize": f"{p.stat().st_size} B",
                "encodingFormat": ENCODING.get(p.suffix, "text/plain"),
                "sha256": ri.sha256_file(p),
            }
        )
    return files


def splits_record_set() -> dict:
    data = []
    for name, url, desc in SPLITS:
        rec = {"splits/name": name, "splits/description": desc}
        rec["splits/url"] = url or ""
        data.append(rec)
    return {
        "@type": "cr:RecordSet",
        "@id": "splits",
        "name": "splits",
        "description": "The values of the split column of trizod_dataset.parquet. "
        "url names the Croissant split for the two test sets and the training "
        "split; the other three values are not for training or testing",
        "dataType": "cr:Split",
        "key": {"@id": "splits/name"},
        "field": [
            {
                "@type": "cr:Field",
                "@id": "splits/name",
                "name": "name",
                "description": "The split value",
                "dataType": "sc:Text",
            },
            {
                "@type": "cr:Field",
                "@id": "splits/url",
                "name": "url",
                "description": "The Croissant split it is (empty: none)",
                "dataType": "sc:Text",
            },
            {
                "@type": "cr:Field",
                "@id": "splits/description",
                "name": "description",
                "description": "What the split value means",
                "dataType": "sc:Text",
            },
        ],
        "data": data,
    }


def chains_record_set(columns: list[dict]) -> dict:
    fields = []
    for col in columns:
        dtype, repeated = data_type(col["arrow_type"])
        desc = col["meaning"]
        if col["null_rule"]:
            desc += f". Null / NaN: {col['null_rule']}"
        desc += f" (Arrow {col['arrow_type']}; nullable: " + (
            "yes)" if col["nullable"] else "no)"
        )
        source = {
            "fileObject": {"@id": MAIN_PARQUET},
            "extract": {"column": col["name"]},
        }
        field = {
            "@type": "cr:Field",
            "@id": f"chains/{col['name']}",
            "name": col["name"],
            "description": desc,
        }
        if dtype is not None:
            field["dataType"] = dtype
            field["source"] = source
        else:
            field["subField"] = [
                {
                    "@type": "cr:Field",
                    "@id": f"chains/{col['name']}/{key}",
                    "name": key,
                    "description": f"Key {key} of each {col['name']} element",
                    "dataType": sub_type,
                    "source": {**source, "transform": {"jsonPath": key}},
                }
                for key, sub_type in struct_fields(col["arrow_type"])
            ]
        if repeated:
            field["repeated"] = True
        if col["name"] == "split":
            field["references"] = {"field": {"@id": "splits/name"}}
        fields.append(field)
    return {
        "@type": "cr:RecordSet",
        "@id": "chains",
        "name": "chains",
        "description": "trizod_dataset.parquet, one record per scored protein "
        "chain; per-residue fields are lists aligned 1:1 to sequence. Select a "
        "split on chains/split (splits) and a tier's training set on "
        "chains/train_tier",
        "key": {"@id": "chains/id"},
        "field": fields,
    }


def build(
    out: Path,
    *,
    version: str,
    doi: str | None = None,
) -> dict:
    """The Croissant document of the deposit in ``out`` (every file but
    :data:`SELF_DESCRIBING` already written)."""
    out = Path(out)
    zenodo = json.loads((out / "zenodo.json").read_text())
    columns = read_columns((out / "columns.md").read_text())
    if not columns:
        raise ValueError("columns.md holds no column table")
    doi = doi or ri.CONCEPT_DOI
    doc = {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": list(CONFORMS_TO),
        "name": "TriZOD",
        "alternateName": zenodo.get("title", ""),
        "description": zenodo.get("description", ""),
        "version": version,
        "url": f"https://doi.org/{doi}",
        "sameAs": f"https://doi.org/{ri.CONCEPT_DOI}",
        "citeAs": cite_as(zenodo, version, doi),
        "creator": [_person(c) for c in zenodo.get("creators", ())],
        "keywords": list(zenodo.get("keywords", ())),
        "license": ri.DATA_LICENSE_URL,
        "isBasedOn": "https://bmrb.io/",
        "isAccessibleForFree": True,
        "isLiveDataset": False,
        "distribution": distribution(out),
        "recordSet": [splits_record_set(), chains_record_set(columns)],
    }
    cff = out / "CITATION.cff"
    released = date_released(cff.read_text()) if cff.is_file() else None
    if released:
        doc["datePublished"] = released
    doc.update(rai_fields((out / "datasheet.md").read_text()))
    return doc


def date_released(cff_text: str) -> str | None:
    """``date-released`` of a ``CITATION.cff`` (YYYY-MM-DD), else None: the
    deposit date is set there when the Zenodo record is published."""
    m = re.search(
        r"^date-released:\s*[\"']?(\d{4}-\d{2}-\d{2})", cff_text, re.MULTILINE
    )
    return m.group(1) if m else None


def write(out: Path, *, version: str, doi: str | None = None) -> Path:
    path = Path(out) / CROISSANT
    doc = build(out, version=version, doi=doi)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return path


def check(path: Path) -> list[str]:
    """Problems of a written ``croissant.json`` against the files beside it:
    every file but :data:`SELF_DESCRIBING` listed once with its SHA-256 and
    size, no listed file missing."""
    path = Path(path)
    out = path.parent
    doc = json.loads(path.read_text())
    listed = {d["contentUrl"]: d for d in doc.get("distribution", ())}
    on_disk = {
        p.relative_to(out).as_posix()
        for p in out.rglob("*")
        if p.is_file() and p.relative_to(out).as_posix() not in SELF_DESCRIBING
    }
    problems = [f"not listed: {f}" for f in sorted(on_disk - set(listed))]
    problems += [f"listed but absent: {f}" for f in sorted(set(listed) - on_disk)]
    for rel in sorted(on_disk & set(listed)):
        d, p = listed[rel], out / rel
        if d.get("sha256") != ri.sha256_file(p):
            problems.append(f"sha256 differs: {rel}")
        if d.get("contentSize") != f"{p.stat().st_size} B":
            problems.append(f"size differs: {rel}")
    return problems
