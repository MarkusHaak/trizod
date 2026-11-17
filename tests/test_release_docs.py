"""Version, licence and test-set name agree across the release documents, and
the documents carry no stale counts or overclaims (release audit 2026-11-17,
G36 and D16).

The README, datasheet and CHANGELOG are templates rendered by ``trizod dataset
finalize`` with the build's own counts (``tests/test_finalize.py`` renders
them); here the committed sources are checked."""

import json
import re
import tomllib

import pytest

from trizod import paths
from trizod.cli.main import app
from trizod.dataset import finalize
from trizod.dataset import release_info as ri

ROOT = paths.ROOT
DOCS = ROOT / "docs" / "dataset"
TEMPLATES = [DOCS / name for name in finalize.TEMPLATES]

#: Every file that states the release identity or describes the split.
WORDING_SCOPE = [
    *TEMPLATES,
    DOCS / "dataset-construction.md",
    DOCS / "columns.md",
    DOCS / "figure-caption.md",
    ROOT / "CITATION.cff",
    ROOT / ".zenodo.json",
    ROOT / "CLAUDE.md",
    ROOT / "README.md",
    ROOT / "scripts" / "build_parquet_dataset.py",
    *sorted((ROOT / "trizod").rglob("*.py")),
]
OVERCLAIM = re.compile(
    r"leakage[- ]free|leak-free|no train/test leakage|with no (train/test )?leakage"
    r"|free of (test-set )?leakage",
    re.I,
)


def test_pyproject_states_the_dataset_version():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["project"]["version"] == ri.DATASET_VERSION


def test_citation_cff_is_the_dataset_with_its_licence_and_test_set():
    text = (ROOT / "CITATION.cff").read_text()
    assert "type: dataset" in text
    assert f'version: "{ri.DATASET_VERSION}"' in text
    assert f"license: {ri.DATA_LICENSE}" in text
    assert "MIT" in text  # the code licence is stated beside it
    assert ri.TEST_SET_NAME in text


def test_zenodo_json_states_version_licence_and_test_set():
    data = json.loads((ROOT / ".zenodo.json").read_text())
    assert data["version"] == ri.DATASET_VERSION
    assert data["license"] == "cc-by-4.0"
    assert ri.TEST_SET_NAME in data["description"]
    assert "MIT" in data["notes"] and "BMRB" in data["notes"]
    assert data["upload_type"] == "dataset"


def test_finalize_accepts_the_repository_metadata():
    finalize.check_repo_metadata(ri.DATASET_VERSION, ROOT)


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_templates_take_version_licence_and_test_set_from_the_build(path):
    text = path.read_text()
    for placeholder in ("version", "test_set_name", "data_license"):
        assert "{{" + placeholder + "}}" in text, placeholder
    if path.name != "CHANGELOG.md":
        assert "{{test_set_label}}" in text or "{{n_test}}" in text


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_templates_type_no_counts(path):
    """Counts come from the build. A thousands-separated number typed into a
    template is a count that will go stale (the v0.3.0 column of the CHANGELOG
    is rendered from finalize.V030, not typed)."""
    text = finalize._TEMPLATE_NOTE.sub("", path.read_text())
    assert not re.findall(r"\b\d{1,3},\d{3}\b", text)


def test_construction_note_states_version_licence_and_test_set():
    text = (DOCS / "dataset-construction.md").read_text()
    assert f"v{ri.DATASET_VERSION}" in text
    assert ri.TEST_SET_NAME in text
    assert "CC BY 4.0" in text and "MIT" in text


@pytest.mark.parametrize(
    "path", WORDING_SCOPE, ids=lambda p: p.relative_to(ROOT).as_posix()
)
def test_no_leakage_overclaim(path):
    hits = OVERCLAIM.findall(path.read_text())
    assert not hits, f"{path}: {hits}"


def test_cli_help_states_what_is_controlled():
    from typer.testing import CliRunner

    out = CliRunner().invoke(app, ["dataset", "--help"]).output
    assert not OVERCLAIM.search(out)
    assert "30 % identity" in out


def test_no_stale_mask_rule_or_membrane_claims():
    for path in [*TEMPLATES, DOCS / "dataset-construction.md", DOCS / "columns.md"]:
        text = path.read_text()
        assert "k == 0" not in text, path
        assert not re.search(r"annotation only|never filtered", text), path


def test_package_and_cli_default_to_the_dataset_version():
    from trizod.dataset import package_release

    assert package_release.DEFAULT_VERSION == ri.DATASET_VERSION
    src = (ROOT / "trizod" / "cli" / "main.py").read_text()
    assert '"2026-07"' not in src


def test_citation_cff_matches_zenodo_and_lists_every_version_doi():
    """Re-audit 2026-11-17b, G36: CITATION.cff describes the dataset, points to
    the deposit's LICENSE.md, lists the version DOIs and agrees with
    .zenodo.json on the title and the affiliations."""
    text = (ROOT / "CITATION.cff").read_text()
    zen = json.loads((ROOT / ".zenodo.json").read_text())
    assert f'title: "{zen["title"]}"' in text
    assert "LICENSE.md" in text and "see LICENSE)" not in text
    assert "the code in this repository" not in text
    for creator in zen["creators"]:
        assert f'affiliation: "{creator["affiliation"]}"' in text, creator["name"]
    assert finalize.version_doi(text, "0.2.0") == "10.5281/zenodo.21309964"
    assert finalize.version_doi(text, "0.3.0") == "10.5281/zenodo.21381698"
    # v0.4.0: the placeholder until the Zenodo draft reserves the DOI
    doi = finalize.version_doi(text, ri.DATASET_VERSION)
    assert doi is None or doi.startswith("10.5281/zenodo.")


def test_readme_states_the_current_version_and_the_test_gate():
    text = (ROOT / "README.md").read_text()
    assert f"v{ri.DATASET_VERSION}" in text
    assert "is v0.3.0" not in text and "latest version v0.3.0" not in text
    assert "uv run --with pyarrow python -m pytest tests/" in text


def test_docs_type_no_tier_counts_of_a_build():
    """docs/filtering.md states no tier sizes (they live in each release's
    CHANGELOG), and nothing says the build-internal scores.json holds nulls."""
    filtering = (ROOT / "docs" / "filtering.md").read_text()
    assert "11,305 → 10,811" not in filtering and "4,245 → 3,957" not in filtering
    for path in [*TEMPLATES, DOCS / "columns.md"]:
        assert "null in scores.json" not in path.read_text(), path
        assert "null in `scores.json`" not in path.read_text(), path


def test_the_readme_and_changelog_introduce_the_sequence_group():
    """How to overlay replicate measurements of one sequence, and what the
    group does not cover (other constructs of the same protein)."""
    readme = (DOCS / "bundle-README.md").read_text()
    assert 'groupby("sequence_group")' in readme
    assert "`uniprot_ids`" in readme and "alignment" in readme
    changelog = (DOCS / "CHANGELOG.md").read_text()
    assert "`sequence_group`" in changelog and "`sequence_group_size`" in changelog
