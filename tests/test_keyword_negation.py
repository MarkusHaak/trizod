"""Negated and IDP phrases do not trip the ``unfold``/``denatur`` keywords.

The moderate/strict keyword lists match ``unfold`` and ``denatur`` as substrings
of the sample-descriptive free text (tolerant has ``denatur`` too). That is
right for "pressure unfolded ubiquitin" or "denatured in 8 M urea", and wrong
for the two phrase families that say the opposite:

* "intrinsically unfolded" / "natively unfolded" -- the IDP vocabulary, i.e.
  exactly the signal the dataset exists to capture (bmr6436 Xc4149, the
  alpha/beta-synuclein depositions 15298/16300/16301);
* "non-denaturing" -- a negation (bmr27021, medin "in non-denaturing
  conditions").

Those phrases are stripped before the two keywords are matched. The keywords
are deliberately NOT gated on cosolvent evidence: in free text "unfolded" far
more often describes a genuinely non-native sample with no cosolvent at all
(pressure-, cold- or acid-unfolded), and gating re-admitted 33 such moderate
chains in the audit's what-if.
"""

import pytest

from tests._stubs import Entry, Sample, fill, fill_real
from tests.conftest import requires_bmrb_data
from trizod.trizod import filter_defaults


def _Entry(title=None, sample_details=None, citation_title=None):
    """One sample whose Details are ``sample_details``."""
    return Entry(
        {"1": Sample(details=sample_details)},
        title=title,
        citation_title=citation_title,
    )


KEYWORDS = ["denatur", "unfold", "misfold"]


def _fill(entry, keywords=KEYWORDS, scope="sample"):
    return fill(entry, keywords=keywords, keyword_search_scope=scope)


@pytest.mark.parametrize(
    "title",
    [
        "Natively unfolded Xc4149",  # bmr6436
        "intrinsically unfolded protein alpha-synuclein",
        "Intrinsically-unfolded domain",
        "the natively-unfolded state of beta-synuclein",
        "an intrinsically  unfolded linker",
    ],
)
def test_idp_phrases_do_not_match_unfold(title):
    assert _fill(_Entry(title=title))["unfold"] is False


@pytest.mark.parametrize(
    "title",
    [
        "medin in non-denaturing conditions",  # bmr27021
        "nondenaturing buffer",
        "non denaturing conditions",
        "the non-denatured protein",
        # a denaturant named as absent (found in review)
        "unfolded states as present in solution. No denaturants are added",  # 5076
        "populated at equilibrium in the abscence of denaturant",  # 17513
        "in the absence of denaturant",
        "measured without denaturant",
    ],
)
def test_negated_denaturation_does_not_match_denatur(title):
    assert _fill(_Entry(title=title))["denatur"] is False


@pytest.mark.parametrize(
    ("title", "keyword"),
    [
        ("ubiquitin L50A mutant pressure unfolded", "unfold"),  # bmr52344
        ("folded/unfolded drkN SH3", "unfold"),
        ("unfolded bovine pancreatic ribonuclease A", "unfold"),
        ("designed cold unfolding four helix bundle", "unfold"),
        ("partially unfolded intermediate", "unfold"),
        ("Denatured ubiquitin", "denatur"),
        ("HIV1 protease folded and cold-denatured", "denatur"),
        ("misfolded prion", "misfold"),
    ],
)
def test_genuine_non_native_text_still_matches(title, keyword):
    assert _fill(_Entry(title=title))[keyword] is True


def test_a_second_genuine_mention_still_matches():
    title = "natively unfolded in buffer, thermally unfolded at 60 C"
    assert _fill(_Entry(title=title))["unfold"] is True
    title = "non-denaturing purification; NMR of the urea-denatured state"
    assert _fill(_Entry(title=title))["denatur"] is True
    title = "in the absence of denaturant and in ambient and denaturing conditions"
    assert _fill(_Entry(title=title))["denatur"] is True
    title = "studied in the absence and presence of denaturant"
    assert _fill(_Entry(title=title))["denatur"] is True


def test_negation_applies_to_sample_details_too():
    entry = _Entry(sample_details="natively unfolded protein in 20 mM phosphate")
    assert _fill(entry)["unfold"] is False


@pytest.mark.parametrize("scope", ["sample", "all"])
def test_negation_applies_in_both_search_scopes(scope):
    entry = _Entry(citation_title="An intrinsically unfolded protein")
    assert _fill(entry, scope=scope)["unfold"] is False


def test_keywords_are_not_gated_on_cosolvent_evidence():
    # No cosolvent anywhere, and the row is still caught: a pressure-unfolded
    # sample is non-native whatever the solvent.
    row = _fill(_Entry(title="ubiquitin pressure unfolded at 2500 bar"))
    assert row["cosolvent_evidence"] is False
    assert row["unfold"] is True


def test_negation_does_not_touch_other_keywords():
    row = _fill(
        _Entry(title="natively unfolded peptide bound to DNA"),
        keywords=["unfold", "bound"],
    )
    assert row["unfold"] is False
    assert row["bound"] is True


def test_tier_keyword_lists_are_unchanged():
    assert filter_defaults.loc["tolerant", "keywords-blacklist"] == ["denatur"]
    assert filter_defaults.loc["moderate", "keywords-blacklist"] == KEYWORDS


# --------------------------------------------------------------------------- #
# real entries
# --------------------------------------------------------------------------- #


def _fill_real(entry_id, keywords=KEYWORDS):
    return fill_real(entry_id, keywords=keywords, keyword_search_scope="sample")


@requires_bmrb_data
@pytest.mark.parametrize("entry_id", ["6436", "15298", "16300", "16301", "27021"])
def test_real_idp_and_negated_entries_pass_the_keywords(entry_id):
    row = _fill_real(entry_id)
    assert not row["unfold"]
    assert not row["denatur"]


@requires_bmrb_data
def test_real_unfolded_state_without_denaturant_passes_denatur():
    # bmr5076, YB1 cold shock domain: "No denaturants are added to the
    # samples" was the only 'denatur' hit, so the tolerant tier (keyword list
    # [denatur]) dropped an unfolded-in-buffer sample. 'unfold' still applies.
    row = _fill_real("5076")
    assert not row["denatur"]
    assert row["unfold"]


@requires_bmrb_data
def test_real_absence_of_denaturant_keeps_the_genuine_mention():
    # bmr17513: "in the abscence of denaturant" AND "ambient and denaturing
    # conditions" (0 M and 6 M urea) -- the second still trips the keyword.
    assert _fill_real("17513")["denatur"]


@requires_bmrb_data
def test_real_pressure_unfolded_ubiquitin_is_still_caught():
    assert _fill_real("52344")["unfold"] is True
