"""The column dictionary is generated from the column contract and cannot go
stale (release audit 2026-11-17, D14 and D16).

``docs/dataset/columns.md`` is rendered from ``COLUMNS`` and ``COLUMN_DOCS`` in
``scripts/build_parquet_dataset.py``. Regenerate it with::

    uv run python scripts/build_parquet_dataset.py \\
        --write-column-dictionary docs/dataset/columns.md
"""

import re

import pytest

from trizod import paths

from ._release_fixtures import build_release_parquet, load_builder, write_bundle

builder = load_builder()
COLUMNS_MD = paths.ROOT / "docs" / "dataset" / "columns.md"


def _table_rows(text: str) -> dict[str, list[str]]:
    rows = {}
    for line in text.splitlines():
        m = re.match(r"\| \d+ \| `([^`]+)` \|", line)
        if m:
            rows[m.group(1)] = [c.strip() for c in line.strip("|").split(" | ")]
    return rows


def test_every_column_has_a_meaning_and_a_null_rule():
    names = [c.name for c in builder.COLUMNS]
    assert sorted(builder.COLUMN_DOCS) == sorted(names)
    for name, doc in builder.COLUMN_DOCS.items():
        assert doc.meaning.strip(), name
        assert doc.null_rule.strip(), name


def test_committed_dictionary_is_the_generated_one():
    assert COLUMNS_MD.read_text() == builder.column_dictionary_markdown(), (
        "docs/dataset/columns.md is stale: run scripts/build_parquet_dataset.py "
        "--write-column-dictionary docs/dataset/columns.md"
    )


def test_dictionary_lists_every_column_in_order_with_its_arrow_type():
    rows = _table_rows(COLUMNS_MD.read_text())
    assert list(rows) == [c.name for c in builder.COLUMNS]
    for col in builder.COLUMNS:
        assert rows[col.name][2] == f"`{builder.arrow_type_doc(col.arrow)}`"


def test_a_parquet_column_missing_from_the_dictionary_fails(tmp_path):
    """The check finalize runs: every column of a built Parquet is documented."""
    pq = pytest.importorskip("pyarrow.parquet")
    bundle = write_bundle(tmp_path / "bundle")
    out = build_release_parquet(bundle, tmp_path / "trizod_dataset.parquet")
    text = COLUMNS_MD.read_text()
    names = pq.read_schema(out).names
    assert [n for n in names if f"`{n}`" not in text] == []
    assert len(names) == len(_table_rows(text))


def test_the_mask_rule_is_finite_g_not_k_zero():
    rows = _table_rows(COLUMNS_MD.read_text())
    mask = " ".join(rows["mask"])
    assert "finite" in mask
    assert "k == 0" not in COLUMNS_MD.read_text()
    assert "not on k" in mask
    assert "999" in mask


def test_nullability_matches_the_build(tmp_path):
    """A column documented as never null holds no null in a built table."""
    bundle = write_bundle(tmp_path / "bundle")
    cols = builder.collect_columns(bundle)
    for name, doc in builder.COLUMN_DOCS.items():
        if not doc.nullable:
            assert all(v is not None for v in cols[name]), name


def test_the_cli_writes_the_dictionary(tmp_path):
    out = tmp_path / "columns.md"
    builder.main(["--write-column-dictionary", str(out)])
    assert out.read_text() == builder.column_dictionary_markdown()


def test_the_sequence_group_columns_are_documented():
    """The same-sequence link: never null, and the dictionary says that a
    different construct of the same protein is not in the group."""
    rows = _table_rows(COLUMNS_MD.read_text())
    for name, arrow in (("sequence_group", "string"), ("sequence_group_size", "int32")):
        assert rows[name][2] == f"`{arrow}`", name
        assert rows[name][3] == "no", name
        assert builder.COLUMN_DOCS[name].null_rule == builder.NEVER_NULL, name
    group = " ".join(rows["sequence_group"])
    for word in ("SHA-256", "identical", "split", "uniprot_ids", "alignment"):
        assert word in group, word
