"""The Parquet deliverable must carry every column its producers emit.

The Zenodo deposit *is* ``trizod_dataset.parquet``, but until now its column
list was hand-maintained in ``scripts/build_parquet_dataset.py`` while the
columns were produced somewhere else entirely — ``output_dataset()`` for the
per-chain metadata, ``detect_bound()`` for the composition labels. Nothing tied
the two together, so the C1b/C3/C4 annotations were emitted into
``scores.json`` / ``_composition_cache.csv`` and then quietly dropped one step
before the artefact anyone downloads.

These tests close that gap: they read the column list from the *producers* and
assert the builder carries it. A new annotation upstream now fails here instead
of silently never shipping.
"""

import importlib.util
import json
import sys

import pandas as pd
import pytest

from trizod import paths
from trizod.bmrb.bmrb import EntityAssemblyRow
from trizod.dataset import composition as comp
from trizod.trizod import output_dataset

SCRIPT = paths.ROOT / "scripts" / "build_parquet_dataset.py"


def _load_builder():
    """Import ``scripts/build_parquet_dataset.py`` (not an installed package).

    Registered in ``sys.modules`` before execution because ``@dataclass``
    resolves its string annotations through the module's own namespace.
    """
    spec = importlib.util.spec_from_file_location("build_parquet_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()

SCORE_TYPES = ["zscores", "gscores"]

#: The 42 columns published in v0.3.0, in order, with their Arrow types. Pinned
#: literally: consumers of the deposit index on these, so they may be appended
#: to but never reordered, renamed or retyped.
PUBLISHED_V030 = [
    ("id", "string"),
    ("entry_id", "string"),
    ("entity_id", "string"),
    ("entity_assem_id", "string"),
    ("st_id", "string"),
    ("entity_name", "string"),
    ("sequence", "string"),
    ("length", "int32"),
    ("gscores", "list<float32>"),
    ("zscores", "list<float32>"),
    ("k", "list<int32>"),
    ("mask", "list<bool>"),
    ("n_scored", "int32"),
    ("split", "string"),
    ("train_tier", "string"),
    ("pool_tier", "string"),
    ("cluster_repr", "string"),
    ("quality", "float64"),
    ("exp_method", "string"),
    ("exp_method_subtype", "string"),
    ("ph", "float32"),
    ("temperature", "float32"),
    ("ionic_strength", "float32"),
    ("total_bbshifts", "int32"),
    ("bbshift_positions_post", "int32"),
    ("bbshift_types_post", "int32"),
    ("citation_title", "string"),
    ("citation_doi", "string"),
    *[(f"off_{a}", "float32") for a in builder.BACKBONE],
    *[(f"lacs_off_{a}", "float32") for a in builder.BACKBONE],
]


# --------------------------------------------------------------------------- #
# scores.json -> Parquet
# --------------------------------------------------------------------------- #


def _scored_frame():
    """One passing row carrying every column ``output_dataset`` reads.

    Deliberately hand-written rather than derived from the builder: if
    ``output_dataset`` starts reading a column this frame lacks, the selection
    raises and the test says so, which is the same failure the Parquet builder
    would otherwise absorb in silence.
    """
    seq = "AGAGA"
    n = len(seq)
    row = {
        "entryID": ["12345"],
        "stID": ["1"],
        "entity_assemID": ["1"],
        "entityID": ["1"],
        "entity_name": ["test peptide"],
        "exp_method": ["NMR"],
        "exp_method_subtype": ["solution"],
        "sample_state_evidence": ["solution"],
        "physical_state": ["native"],
        "denaturant_evidence": [False],
        "membrane_mimetic": ["SDS;micelle"],
        "citation_DOI": ["10.1000/xyz"],
        "citation_title": ["A title"],
        "ionic_strength": [0.15],
        "pH": [7.0],
        "temperature": [298.0],
        "bbshift_positions_post": [n],
        "bbshift_types_post": [3],
        "total_bbshifts": [3 * n],
        "seq": [seq],
        "k": [[3] * n],
        "zscores": [[0.5] * n],
        "gscores": [[0.25] * n],
        "pass_post": [True],
    }
    for atom in builder.BACKBONE:
        row[f"off_{atom}"] = [0.1]
        row[f"lacs_off_{atom}"] = [0.2]
    return pd.DataFrame(row)


def _emit_scores_json(tmp_path):
    try:
        output_dataset(
            _scored_frame(),
            tmp_path / "scores",
            "json",
            SCORE_TYPES,
            4,
            include_shifts=False,
            no_shift_averaging=False,
        )
    except KeyError as exc:
        pytest.fail(
            f"output_dataset() reads a column this test does not stub: {exc}. "
            "Add it to _scored_frame() here AND to COLUMNS in "
            "scripts/build_parquet_dataset.py — an emitted column that the "
            "Parquet builder does not carry never reaches the deposit."
        )
    text = (tmp_path / "scores.json").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_parquet_carries_every_scores_json_column(tmp_path):
    """Every key ``output_dataset`` writes must map to a Parquet column."""
    records = _emit_scores_json(tmp_path)
    assert len(records) == 1
    uncarried = (
        set(records[0]) - builder.SCORES_JSON_SOURCES - builder.SCORES_JSON_NOT_CARRIED
    )
    assert not uncarried, (
        f"scores.json emits {sorted(uncarried)}, which scripts/"
        "build_parquet_dataset.py does not carry into the Parquet deposit. "
        "Add a Column(..., origin='scores') for each, or list it in "
        "SCORES_JSON_NOT_CARRIED with a reason."
    )


def test_declared_scores_sources_are_actually_emitted(tmp_path):
    """...and nothing declared may be a phantom key that is never written."""
    emitted = set(_emit_scores_json(tmp_path)[0])
    assert emitted >= builder.SCORES_JSON_SOURCES, sorted(
        builder.SCORES_JSON_SOURCES - emitted
    )


@pytest.mark.parametrize(
    "key",
    [
        "physical_state",
        "denaturant_evidence",
        "sample_state_evidence",
        "membrane_mimetic",
    ],
)
def test_sample_state_annotations_reach_the_parquet(key):
    """Regression pin for the defect: these four stopped at scores.json."""
    assert key in builder.SCORES_JSON_SOURCES


# --------------------------------------------------------------------------- #
# detect_bound() -> Parquet
# --------------------------------------------------------------------------- #


class _StubEntity:
    id = "1"
    name = "stub"
    type = "polymer"
    polymer_type = "polypeptide(L)"
    nonpolymer_comp_id = ""
    nonpolymer_comp_label = ""


class _StubAssembly:
    entity_assembly_rows = [
        EntityAssemblyRow(
            id="1",
            name="stub",
            entity_id="1",
            entity_label="$stub",
            physical_state="native",
            conformational_isomer="no",
            magnetic_equivalence_group="",
            details="",
        )
    ]


class _StubEntry:
    entities = {"1": _StubEntity()}
    assemblies = {"1": _StubAssembly()}
    chem_comps = {}


def test_parquet_carries_every_composition_column():
    """Every ``detect_bound()`` label must map to a Parquet column, and no
    Parquet column may claim a composition key the classifier never emits."""
    assert set(comp.detect_bound(_StubEntry())) == builder.COMPOSITION_SOURCES


# --------------------------------------------------------------------------- #
# the published layout may only be appended to
# --------------------------------------------------------------------------- #


def test_published_v030_columns_are_unchanged():
    published = [(c.name, c.arrow) for c in builder.COLUMNS[: len(PUBLISHED_V030)]]
    assert published == PUBLISHED_V030


def test_new_columns_are_appended_after_the_published_ones():
    names = [c.name for c in builder.COLUMNS]
    assert len(names) == len(set(names)), "duplicate Parquet column name"
    assert len(names) > len(PUBLISHED_V030)


def test_every_column_declares_a_known_arrow_type():
    """Checked without pyarrow (an optional extra): the token must be known, and
    ``Column.__post_init__`` rejects an unknown one at declaration time."""
    for column in builder.COLUMNS:
        assert column.arrow in builder.ARROW_TYPES, column.name


def test_arrow_schema_matches_the_column_contract():
    pa = pytest.importorskip("pyarrow")
    schema = builder.arrow_schema(pa)
    assert schema.names == [c.name for c in builder.COLUMNS]


# --------------------------------------------------------------------------- #
# joining: composition on entry ID, label_tier on chain ID, nulls for misses
# --------------------------------------------------------------------------- #

COMPOSITION_HEADER = ["entryID", *sorted(builder.COMPOSITION_SOURCES)]


def _write_bundle(root, *, records, test_trizod_header=None):
    """Minimal bundle: scores + per-tier train/cluster files + test FASTAs."""
    scores = root / "scores" / "unfiltered"
    scores.mkdir(parents=True)
    (scores / "scores.json").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    for tier in builder.TIER_ORDER:
        tier_dir = root / "train" / tier
        tier_dir.mkdir(parents=True)
        (tier_dir / f"train_{tier}_best.fasta").write_text("")
        (tier_dir / "clusters_best.tsv").write_text(
            "cluster\tmember\tbest_repr\tmember_quality\n"
        )
    test = root / "test"
    test.mkdir(parents=True)
    (test / "CheZOD117_test_set.fasta").write_text("")
    (test / "TriZOD_test_set.fasta").write_text(
        f"{test_trizod_header}\nAAAA\n" if test_trizod_header else ""
    )
    return root


def _record(entry_id, chain_id, **extra):
    rec = {
        "ID": chain_id,
        "entryID": entry_id,
        "stID": "1",
        "entity_assemID": "1",
        "entityID": "1",
        "seq": "AAAA",
        "k": [3, 3, 3, 3],
        "zscores": [1.0, 1.0, None, 1.0],
        "gscores": [0.5, 0.5, None, 0.5],
    }
    rec.update(extra)
    return rec


def _write_composition(path, rows):
    lines = [",".join(COMPOSITION_HEADER)]
    for row in rows:
        lines.append(",".join(str(row.get(k, "")) for k in COMPOSITION_HEADER))
    path.write_text("\n".join(lines) + "\n")
    return path


def _one(cols, name, i=0):
    return cols[name][i]


def test_composition_joins_on_entry_id_and_missing_entries_are_null(tmp_path):
    bundle = _write_bundle(
        tmp_path / "bundle",
        records=[
            _record("12345", "12345_1_1_1"),
            _record("12345", "12345_2_1_1"),  # 2nd chain of the SAME entry
            _record("99999", "99999_1_1_1"),  # no composition record at all
        ],
    )
    csv_path = _write_composition(
        tmp_path / "_composition_cache.csv",
        [
            {
                "entryID": "12345",
                "n_entities": "2",
                "has_non_polymer": "True",
                "has_nucleic": "False",
                "has_metal": "True",
                "has_metal_ion": "True",
                "has_metal_cofactor": "False",
                "has_water": "False",
                "has_other_ligand": "False",
                "metal_comp_ids": "CA;ZN",
                "ligand_comp_ids": "CA;ZN",
                "ligand_names": "",
                "multi_protein_assembly": "False",
                "n_entity_assembly_rows": "2",
                "n_copies": "2",
                "has_conformational_isomer": "False",
                "is_bound": "True",
            }
        ],
    )
    cols = builder.collect_columns(bundle, csv_path)

    # both chains of entry 12345 carry the entry's composition
    for i in (0, 1):
        assert _one(cols, "has_metal", i) is True
        assert _one(cols, "has_nucleic", i) is False
        assert _one(cols, "n_copies", i) == 2
        assert _one(cols, "metal_comp_ids", i) == ["CA", "ZN"]
        # measured absence -> empty list, not null
        assert _one(cols, "ligand_names", i) == []
    # an entry with no composition row is null everywhere, never False/0
    for name in builder.COMPOSITION_SOURCES:
        assert _one(cols, name, 2) is None, name


def test_stale_composition_cache_yields_nulls_not_a_crash(tmp_path):
    """The cache on disk predates C3/C4; absent columns must read as null."""
    bundle = _write_bundle(
        tmp_path / "bundle", records=[_record("12345", "12345_1_1_1")]
    )
    csv_path = tmp_path / "old_cache.csv"
    csv_path.write_text("entryID,n_entities,is_bound\n12345,1,False\n")

    cols = builder.collect_columns(bundle, csv_path)
    assert _one(cols, "n_entities") == 1
    assert _one(cols, "is_bound") is False
    assert _one(cols, "n_copies") is None
    assert _one(cols, "ligand_comp_ids") is None
    assert _one(cols, "has_metal") is None


def test_skip_composition_leaves_every_composition_column_null(tmp_path):
    bundle = _write_bundle(
        tmp_path / "bundle", records=[_record("12345", "12345_1_1_1")]
    )
    cols = builder.collect_columns(bundle, None)
    for name in builder.COMPOSITION_SOURCES:
        assert _one(cols, name) is None, name


def test_label_tier_comes_from_the_test_fasta_header(tmp_path):
    bundle = _write_bundle(
        tmp_path / "bundle",
        records=[_record("12345", "12345_1_1_1"), _record("99999", "99999_1_1_1")],
        test_trizod_header=">12345_1_1_1 label_tier=tolerant pinned_id=12345_1_1_1",
    )
    cols = builder.collect_columns(bundle, None)
    assert _one(cols, "label_tier", 0) == "tolerant"
    assert _one(cols, "split", 0) == "test_trizod"
    # every non-test row: no label_tier, and no invented one
    assert _one(cols, "label_tier", 1) is None


def test_label_tier_null_when_the_bundle_predates_the_header(tmp_path):
    bundle = _write_bundle(
        tmp_path / "bundle",
        records=[_record("12345", "12345_1_1_1")],
        test_trizod_header=">12345_1_1_1",
    )
    cols = builder.collect_columns(bundle, None)
    assert _one(cols, "split") == "test_trizod"
    assert _one(cols, "label_tier") is None


def test_sample_state_columns_are_copied_from_scores_json(tmp_path):
    bundle = _write_bundle(
        tmp_path / "bundle",
        records=[
            _record(
                "12345",
                "12345_1_1_1",
                physical_state="molten globule",
                denaturant_evidence=True,
                sample_state_evidence="solution",
                membrane_mimetic="SDS;micelle",
            ),
            _record("99999", "99999_1_1_1"),  # keys absent entirely
        ],
    )
    cols = builder.collect_columns(bundle, None)
    assert _one(cols, "physical_state") == "molten globule"
    assert _one(cols, "denaturant_evidence") is True
    assert _one(cols, "sample_state_evidence") == "solution"
    assert _one(cols, "membrane_mimetic") == "SDS;micelle"
    for name in (
        "physical_state",
        "denaturant_evidence",
        "sample_state_evidence",
        "membrane_mimetic",
    ):
        assert _one(cols, name, 1) is None, name


def test_published_columns_still_populate(tmp_path):
    """The additive change must not disturb the 42 existing columns."""
    bundle = _write_bundle(
        tmp_path / "bundle",
        records=[
            _record(
                "12345",
                "12345_1_1_1",
                pH=7.4,
                temperature=298.0,
                ionic_strength=0.15,
                citation_DOI="10.1000/xyz",
                entity_name="test peptide",
                total_bbshifts=12,
            )
        ],
    )
    cols = builder.collect_columns(bundle, None)
    assert _one(cols, "id") == "12345_1_1_1"
    assert _one(cols, "entry_id") == "12345"
    assert _one(cols, "sequence") == "AAAA"
    assert _one(cols, "length") == 4
    assert _one(cols, "mask") == [True, True, False, True]
    assert _one(cols, "n_scored") == 3
    assert _one(cols, "split") == "excluded"
    assert _one(cols, "ph") == pytest.approx(7.4)
    assert _one(cols, "citation_doi") == "10.1000/xyz"
    assert _one(cols, "k") == [3, 3, 3, 3]


# --------------------------------------------------------------------------- #
# scalar coercion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        (True, True),
        (False, False),
        ("True", True),
        ("False", False),
        ("true", True),
        (None, None),
        ("", None),
        ("nan", None),
        ("<NA>", None),
        ("maybe", None),
    ],
)
def test_as_bool_never_invents_false(raw, expected):
    assert builder.as_bool(raw) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CA;ZN", ["CA", "ZN"]),
        ("ZN", ["ZN"]),
        ("", []),  # classified, nothing found
        (None, None),  # not classified
        ("nan", None),
        (["CA"], ["CA"]),
    ],
)
def test_as_str_list_round_trips_the_classifier_join(raw, expected):
    assert builder.as_str_list(raw) == expected
