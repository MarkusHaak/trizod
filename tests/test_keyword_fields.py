"""Keyword-blacklist field collection in ``fill_row_data`` (GitHub issue #23).

``_Citation_keyword.Keyword`` and ``_Struct_keywords`` were collected with
``fields.extend(el)`` where ``el`` is a *string*, so the string was exploded into
its characters and no multi-character blacklist keyword could ever match either
field. These tests pin whole-string matching for both.

Which fields are searched at all is deliberately *not* changed here -- the guard
tests below only pin the status quo so the field-scoping change stays a separate,
bisectable diff.
"""

import pandas as pd

from trizod.trizod import fill_row_data


class _Conditions:
    """Stand-in for ``bmrb.SampleConditions`` -- constant, valid conditions."""

    def get_ionic_strength(self, **kwargs):
        return 0.1

    def get_pH(self, **kwargs):
        return 7.0

    def get_temperature(self, **kwargs):
        return 298.0


class _Named:
    """Stand-in for Assembly/Entity/Sample: only the free-text fields matter."""

    def __init__(self, name=None, details=None, framecode=None):
        self.name = name
        self.details = details
        self.framecode = framecode
        self.seq = None  # skips the backbone-shift block in fill_row_data
        self.paramagnetic = None
        self.components = []


class _Entry:
    def __init__(
        self,
        title=None,
        details=None,
        citation_keywords=None,
        struct_keywords=None,
    ):
        self.id = "1"
        self.title = title
        self.details = details
        self.citation_title = None
        self.citation_DOI = None
        self.exp_method = "NMR"
        self.exp_method_subtype = "solution"
        self.citation_keywords = citation_keywords
        self.struct_keywords = struct_keywords
        self.entities = {1: _Named(name="entity name")}
        self.assemblies = {1: _Named(name="assembly name")}
        self.samples = {1: _Named(name="sample name", framecode="sample_1")}
        self.conditions = {1: _Conditions()}

    def get_peptide_shifts(self):
        return {(1, 1, 1): (None, 1, 1, [1])}


def _fill(entry, keywords):
    row = pd.Series({"entryID": "1", "stID": 1, "entity_assemID": 1, "entityID": 1})
    bmrb_entries = pd.DataFrame({"entry": [entry]}, index=["1"])
    return fill_row_data(
        row,
        chemical_denaturants=[],
        keywords=keywords,
        bmrb_entries=bmrb_entries,
    )


def test_citation_keywords_matched_whole():
    entry = _Entry(citation_keywords=["Protein misfolding"])
    assert _fill(entry, ["misfold"])["misfold"] is True


def test_struct_keywords_matched_whole():
    entry = _Entry(struct_keywords=["protein-protein interaction"])
    assert _fill(entry, ["interacti"])["interacti"] is True


def test_scalar_keyword_string_appended_not_exploded():
    # Every corpus entry carries lists today, but the scalar branch must not
    # explode its string either.
    entry = _Entry(citation_keywords="Protein misfolding")
    assert _fill(entry, ["misfold"])["misfold"] is True
    entry = _Entry(struct_keywords="protein-protein interaction")
    assert _fill(entry, ["interacti"])["interacti"] is True


def test_single_character_keyword_still_matches():
    # The pre-fix character explosion made 1-character keywords work by accident;
    # whole-string matching must keep them working.
    entry = _Entry(citation_keywords=["Z"], struct_keywords=["E"])
    row = _fill(entry, ["z", "e"])
    assert row["z"] is True
    assert row["e"] is True


def test_absent_keyword_stays_false():
    entry = _Entry(citation_keywords=["Protein misfolding"])
    assert _fill(entry, ["denatur"])["denatur"] is False


def test_none_keyword_fields_are_tolerated():
    entry = _Entry(citation_keywords=None, struct_keywords=None)
    assert _fill(entry, ["misfold"])["misfold"] is False


def test_sample_descriptive_fields_still_searched():
    # Guard: this change must not alter WHICH fields are searched.
    assert _fill(_Entry(title="Denatured ubiquitin"), ["denatur"])["denatur"] is True
    assert (
        _fill(_Entry(details="in 8 M urea, denatured"), ["denatur"])["denatur"] is True
    )
