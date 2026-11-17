"""Lipopolysaccharide (LPS) samples are membrane mimetics, by rule.

Release re-audit 2026-11-17b, item 4 (C8): the class table had no LPS class, so
of the LPS-sample chains in the moderate pool 25 were unflagged while 10 were
flagged, through per-entry overrides (19334, 20059, 20060, 26036, 26037, 36451)
or a "micelle"/"bicelle" token beside the LPS (18102, 21011, 34476, 36724);
26036 and 26038 deposit identical samples, one flagged and one not. Decision:
LPS micelles and bilayers count as membrane mimetics, everywhere and
consistently. An LPS component, sample type or sample text is classified by the
rule (class ``LPS``, any concentration); overrides remain only for entries whose
LPS is named outside the sample (title, assembly name), with class ``LPS``.

Proteins named after LPS are no evidence: anti-lipopolysaccharide factors,
LPS-binding protein, the Lpt LPS transport/assembly machinery.
"""

from __future__ import annotations

import pytest

from tests._stubs import Entry, Named, Sample, fill
from trizod.bmrb import membrane


def _component(name, value="100", unit="mM", entity_id=""):
    return ("1", "1", entity_id, name, value, unit)


def _label(name):
    cls = membrane.classify_component(name)
    return None if cls is None else cls.label


# --------------------------------------------------------------------------- #
# classification by name
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("LPS", "LPS"),
        ("Lipopolysaccharide", "LPS"),
        ("lipopolysaccharides", "LPS"),
        ("LPS from S. typhosa", "LPS"),  # bmr36044
        ("E. coli O111:B4 LPS", "LPS"),
        ("LPS micelles", "micelle"),  # a micelle by name; counts either way
        ("LIPOPOLYSACCHARIDE BICELLE", "bicelle"),  # bmr36724
    ],
)
def test_an_lps_component_is_classified(name, label):
    assert _label(name) == label


@pytest.mark.parametrize(
    "name",
    [
        "anti-lipopolysaccharide factor",
        "Anti-LPS factor ALFPm3",
        "LPS-binding protein",
        "lipopolysaccharide binding protein",
        "Lipopolysaccharide export system protein LptA",
        "LPS assembly protein LptD",
        "lipopolysaccharide transport protein LptC",
        "ALPS",
        "LPS1",
    ],
)
def test_a_protein_named_after_lps_is_not_classified(name):
    assert _label(name) is None


def test_lps_counts_at_any_concentration():
    (cls,) = membrane.classify_name("Lipopolysaccharide")
    assert cls.kind != membrane.ADDITIVE
    assert membrane.component_counts(cls, "14.5", "microM")  # bmr21008
    assert membrane.component_counts(cls, "0.02", "mM")  # bmr36038


# --------------------------------------------------------------------------- #
# samples
# --------------------------------------------------------------------------- #


def _fill(sample, entity_type="polymer"):
    entry = Entry({"1": sample})
    entry.entities = {"1": Named(name="peptide", entity_type=entity_type)}
    return fill(entry)


@pytest.mark.parametrize(
    "name",
    ["Lipopolysaccharide", "LPS", "LPS from S. typhosa"],
)
def test_a_non_polymer_lps_component_makes_a_mimetic(name):
    row = _fill(Sample([_component(name, "10", "uM")]))
    assert row["membrane_mimetic_sample"] is True
    assert "LPS" in row["membrane_mimetic"].split(";")


@pytest.mark.parametrize(
    "name",
    ["YI12FF+LPS", "WR17_LPS", "KR8_LPS", "GG8WF+LPS", "CDT-LPS"],
)
def test_a_peptide_component_named_with_lps_makes_a_mimetic(name):
    # bmr20060, 19468, 19470, 20061, 18102: the peptide entity's component
    # name records the LPS it was measured in
    row = _fill(Sample([_component(name, "0.5", "mM", entity_id="1")]))
    assert row["membrane_mimetic_sample"] is True
    assert "LPS" in row["membrane_mimetic"].split(";")


@pytest.mark.parametrize(
    "name",
    ["LPS-binding protein", "anti-lipopolysaccharide factor", "LptA"],
)
def test_a_protein_named_after_lps_is_no_evidence(name):
    row = _fill(Sample([_component(name, "0.5", "mM", entity_id="1")]))
    assert row["membrane_mimetic_sample"] is False


@pytest.mark.parametrize(
    ("field", "text"),
    [
        ("details", "Peptide in LPS micelles"),
        ("details", "1 mM peptide, 10 uM lipopolysaccharide"),
        ("name", "VG16KRKP in LPS"),
        ("solvent_system", "90% H2O/10% D2O, LPS"),
    ],
)
def test_lps_in_the_sample_text_makes_a_mimetic(field, text):
    sample = Sample([_component("peptide", entity_id="1")], **{field: text})
    row = _fill(sample)
    assert row["membrane_mimetic_sample"] is True


@pytest.mark.parametrize(
    "text",
    ["free peptide in the absence of LPS", "LPS-free buffer"],
)
def test_a_negated_lps_mention_is_no_evidence(text):
    row = _fill(Sample([_component("peptide", entity_id="1")], details=text))
    assert row["membrane_mimetic_sample"] is False


@pytest.mark.parametrize("sample_type", ["lipopolysaccharide", "LPS micelle"])
def test_an_lps_sample_type_makes_a_mimetic(sample_type):
    row = _fill(Sample([_component("peptide", entity_id="1")], sample_type=sample_type))
    assert row["membrane_mimetic_sample"] is True


# --------------------------------------------------------------------------- #
# the committed override table
# --------------------------------------------------------------------------- #

#: LPS entries whose LPS is named only outside the sample (title, assembly
#: name): reviewed overrides of class LPS.
TITLE_ONLY_LPS = ("19334", "20059", "26036", "26037", "26038", "25877", "36451")


def test_lps_overrides_have_class_lps():
    overrides = membrane.load_overrides()
    for key in TITLE_ONLY_LPS:
        assert overrides[key].label == "LPS", key


def test_overrides_the_lps_rule_covers_are_gone():
    # bmr20060 deposits its sample component as "YI12FF+LPS": the rule sees it
    assert "20060" not in membrane.load_overrides()


def test_no_override_names_lps_under_another_class():
    for key, override in membrane.load_overrides().items():
        reason = override.reason.lower()
        if "lps" in reason or "lipopolysacchar" in reason:
            assert override.label == "LPS", key
