"""Membrane mimetics: a class rule, filtered at moderate and strict (2026-11).

Until 2026-11 a substring list (``MEMBRANE_MIMETIC_TOKENS``) only annotated the
``membrane_mimetic`` column. Its recall was the weaker side -- micelles and
bicelles written as IUPAC lipid names, LDAO, Fos-choline or lauroyl sarcosine
went unflagged -- and it flagged soluble proteins solubilised by a sub-CMC
additive (1 mM CHAPS) and two RDC alignment media. A sample is now a membrane
mimetic (``membrane_mimetic_sample``) when any non-polymer component is

  (a) a bilayer-forming phospholipid or bilayer assembly, at any concentration;
  (b) a dedicated micelle former (SDS, alkylphosphocholines, lysophospholipids,
      short-chain phosphocholines, LDAO, lauroyl sarcosine, CnEm), at any
      concentration -- sub-CMC SDS already induces helix, and a classic
      mimetic deposited at "10 uM" is a unit error, not an aqueous sample;
  (c) an additive-class detergent at or above its CMC in water, or with no
      convertible concentration;

except a CnEm alignment medium (CnEm with an n-alkanol, or a ``liquid
crystal`` sample without phospholipid). ``membrane_mimetic`` keeps naming every
recognised class, counted or not.
"""

import math

import pandas as pd
import pytest

from tests._stubs import Entry, Sample, condition_frame, fill, prefilter_tier
from trizod.bmrb import membrane
from trizod.trizod import filter_defaults

TIERS = ["unfiltered", "tolerant", "moderate", "strict"]


def _label(name):
    cls = membrane.classify_component(name)
    return None if cls is None else cls.label


# --------------------------------------------------------------------------- #
# classification by name
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "label"),
    [
        # (b) micelle formers
        ("SDS", "SDS"),
        ("SDS-d25", "SDS"),
        ("sodium dodecyl sulfate", "SDS"),
        ("sodium lauryl sulphate", "SDS"),
        ("sodium dodecyl sulfphate", "SDS"),  # bmr34388, sic
        ("DPC", "DPC"),
        ("d38-DPC", "DPC"),
        ("dodecylphosphocholine", "DPC"),
        ("n-dodecyl phosphocholine-d38", "DPC"),
        ("Dodecyl Phosphcholine", "DPC"),  # deposited misspellings
        ("dodecyphosphocholine", "DPC"),
        ("dodecylphosphocoline", "DPC"),
        ("Fos-choline-12", "DPC"),
        ("decylphosphocholine", "FC-10"),
        ("Fos-Choline-10", "FC-10"),
        ("Foscholine-14", "FC-14"),
        ("tetradecylphosphocholine", "FC-14"),
        ("Hexadecylphosphocholine", "FC-16"),
        ("LMPG", "LMPG"),
        ("1-myristoyl-2-hydroxy-sn-glycero-3-phospho-(1'-rac-glycerol)", "LMPG"),
        ("LPPG", "LPPG"),
        ("1-palmitoyl-2-hydroxy-sn-glycero-3-phospho-(1'-rac-glycerol)", "LPPG"),
        ("16:0 Lyso PG", "LPPG"),
        ("DHPC", "DHPC"),
        ("d22-DHPC", "DHPC"),
        ("1,2-dihexanoyl-sn-glycero-3-phosphocholine", "DHPC"),
        ("1,2-diheptanoyl-sn-glycero-3-phosphocholine", "DH7PC"),
        ("Dioctanoyl phosphatidyl glycerol", "short-chain phospholipid"),
        ("LDAO", "LDAO"),
        ("lauryldimethylamine-N-oxide", "LDAO"),
        ("N-Lauroyl-sarcosine", "lauroyl sarcosine"),
        ("sodium lauroyl sarcosinate", "lauroyl sarcosine"),
        ("pentaethylene glycol monododecyl ether (C12E5)", "CnEm"),
        ("N-octylpentaoxyethylene", "CnEm"),
        ("C8E5", "CnEm"),
        ("dodecyltrimethylammonium chloride", "DTAC"),
        ("micelles", "micelle"),
        # (a) bilayer lipids and assemblies
        ("DMPC", "phospholipid"),
        ("DMPC (D54)", "phospholipid"),
        ("POPG", "phospholipid"),
        ("3:1 POPE/POPG vesicles", "phospholipid"),
        ("1,2-dimyristoyl-sn-glycero-3-phosphocholine", "phospholipid"),
        ("1,2-dimyristoyl-d54-sn-glycero-3-phosphocholine", "phospholipid"),
        ("1-palmitoyl-2-oleoyl-sn-glycero-3-phosphocholine", "phospholipid"),
        ("1,2-diphytanoyl-sn-glycero-3-phosphocholine", "phospholipid"),
        ("1,2-di-O-tetradecyl-sn-glycero-3-phosphocholine", "phospholipid"),
        ("d54-14:0 PC (DMPC lipid)", "phospholipid"),
        ("E. coli total lipid extract", "phospholipid"),
        ("asolectin lipids", "phospholipid"),
        ("cardiolipin", "phospholipid"),
        ("bicelles", "bicelle"),
        ("LIPOPOLYSACCHARIDE BICELLE", "bicelle"),
        ("MSP1D1", "nanodisc"),
        ("nanodisc", "nanodisc"),
        ("liposome", "liposome"),
        ("Small Unilamellar Vesicle", "liposome"),
        # (c) additive-class detergents
        ("CHAPS", "CHAPS"),
        ("CHAPSO", "CHAPSO"),
        ("n-octyl-beta-D-glucopyranoside", "OG"),
        ("n-Octyl-β-D-glucopyranoside", "OG"),
        ("n-octyl-beta-D-maltoside", "OM"),
        ("n-dodecyl-beta-D-maltoside", "DDM"),
        ("DDM", "DDM"),
        ("n-Decyl-b-D-maltoside", "DM"),
        ("DM", "DM"),
        ("Triton X-100", "Triton X-100"),
        ("Tween 20", "Tween-20"),
        ("polysorbate 20", "Tween-20"),
        ("digitonin", "digitonin"),
    ],
)
def test_component_is_classified(name, label):
    assert _label(name) == label


@pytest.mark.parametrize(
    "name",
    [
        "sodium phosphate",
        "DSS",
        "D2O",
        "2-mercaptoethanol",
        "n-hexanol",
        # bile-acid ligands, acyl-CoA and lysozyme are not mimetics
        "Glycochenodeoxycholic acid",
        "sodium cholate",
        "Myristoyl coenzyme A",
        "palmitoyl-CoA",
        "lysozyme",
        # a peptidoglycan precursor (ligand), not a bilayer lipid
        "lipid II",
        # L-DOPA, not dioleoyl phosphatidic acid
        "L-DOPA",
        # sarcosine alone is an osmolyte
        "sarcosine",
    ],
)
def test_non_mimetics_are_not_classified(name):
    assert _label(name) is None


@pytest.mark.parametrize(
    ("label", "kind"),
    [
        ("phospholipid", "bilayer"),
        ("bicelle", "bilayer"),
        ("SDS", "micelle"),
        ("DPC", "micelle"),
        ("DHPC", "micelle"),
        ("CnEm", "micelle"),
        ("CHAPS", "additive"),
        ("DDM", "additive"),
        ("digitonin", "additive"),
    ],
)
def test_class_kinds(label, kind):
    assert membrane.MIMETIC_CLASSES_BY_LABEL[label].kind == kind


def test_additive_cmcs_in_water():
    cmc = {
        label: cls.cmc_millimolar
        for label, cls in membrane.MIMETIC_CLASSES_BY_LABEL.items()
        if cls.kind == "additive"
    }
    assert cmc == {
        "CHAPS": 8.0,
        "CHAPSO": 8.0,
        "OG": 18.0,
        "OM": 19.5,
        "DM": 1.8,
        "DDM": 0.17,
        "Triton X-100": 0.24,
        "Tween-20": 0.06,
        "digitonin": None,  # counted whenever present
    }


# --------------------------------------------------------------------------- #
# concentration -> mM
# --------------------------------------------------------------------------- #

MW = 614.9  # CHAPS


@pytest.mark.parametrize(
    ("value", "unit", "mM"),
    [
        ("100", "mM", 100.0),
        ("0.1", "M", 100.0),
        ("10", "uM", 0.01),
        ("10", "µM", 0.01),
        ("10", "μM", 0.01),
        ("10", "microM", 0.01),
        ("1", "%", 10000 / MW),
        ("1", "% w/v", 10000 / MW),
        ("1", "mg/mL", 1000 / MW),
        ("500", "ug/mL", 500 / MW),
        ("500", "µg/ml", 500 / MW),
    ],
)
def test_concentration_is_converted_to_mM(value, unit, mM):
    assert membrane.concentration_mM(value, unit, MW) == pytest.approx(mM)


@pytest.mark.parametrize(
    ("value", "unit", "mw"),
    [
        ("5", "mg", MW),  # an amount
        ("2", "umol", MW),
        ("3", "w/w", MW),  # a bare ratio
        ("1", "eq", MW),
        ("", "mM", MW),
        (None, None, MW),
        ("1", "%", None),  # needs a molar mass
        ("1", "", MW),
    ],
)
def test_amounts_ratios_and_gaps_are_not_concentrations(value, unit, mw):
    assert math.isnan(membrane.concentration_mM(value, unit, mw))


# --------------------------------------------------------------------------- #
# the counting rule, per sample
# --------------------------------------------------------------------------- #


def _sample(*components, sample_type="solution"):
    """Components as (name, value, unit); non-polymer (no Entity_ID)."""
    return Sample(
        [(str(i), "1", "", n, v, u) for i, (n, v, u) in enumerate(components, 1)],
        sample_type=sample_type,
    )


def _is_mimetic(*components, sample_type="solution"):
    return membrane.classify_samples([_sample(*components, sample_type=sample_type)])[1]


@pytest.mark.parametrize(
    "component",
    [
        ("POPC", "10", "uM"),  # (a) at any concentration
        ("DMPC", "", ""),
        ("DPC", "10", "uM"),  # (b): bmr21013 "DPC micelle bound", unit error
        ("DPC-d38", "50", "uM"),  # bmr30509
        ("DHPC", "4.5", "mM"),  # bmr50241/50261, below the 15 mM CMC
        ("SDS", "0.1", "mM"),  # sub-CMC SDS induces helix
        ("CHAPS", "8", "mM"),  # (c) at the CMC ...
        ("CHAPS", "20", "mM"),  # ... and above it
        ("CHAPS", "", ""),  # (c) without a concentration
        ("CHAPS", "5", "mg"),  # (c) with an amount, not a concentration
        ("CHAPS", "0.5", "%"),  # 8.1 mM
        ("DDM", "0.5", "mM"),
        ("Tween 20", "0.01", "%"),  # 0.08 mM >= 0.06 mM
        ("digitonin", "0.001", "mM"),  # digitonin counts whenever present
    ],
)
def test_component_makes_the_sample_a_membrane_mimetic(component):
    assert _is_mimetic(component) is True


@pytest.mark.parametrize(
    "component",
    [
        ("CHAPS", "1", "mM"),  # NALP1 PYD (16670): a solubility additive
        ("CHAPS", "0.05", "mM"),  # 25219
        ("CHAPSO", "0.1", "mM"),  # 51677
        ("n-octyl-beta-D-glucopyranoside", "5", "mM"),
        ("Triton X-100", "0.1", "mM"),
        ("DDM", "0.1", "mM"),
        ("sodium phosphate", "20", "mM"),
    ],
)
def test_sub_cmc_additive_is_not_a_membrane_mimetic(component):
    assert _is_mimetic(component) is False


def test_the_description_still_names_a_sub_cmc_additive():
    labels, is_mimetic = membrane.classify_samples([_sample(("CHAPS", "1", "mM"))])
    assert labels == ["CHAPS"] and is_mimetic is False


@pytest.mark.parametrize(
    "components",
    [
        # RDC alignment media: 31134 (C12E5/hexanol), 25324 (C8E5/octanol)
        [("C12E5", "5", "%"), ("n-hexanol", "", "")],
        [("N-octylpentaoxyethylene", "", ""), ("1-octanol", "", "")],
    ],
)
def test_cnem_alkanol_alignment_medium_is_not_counted(components):
    assert _is_mimetic(*components) is False


def test_liquid_crystal_cnem_without_phospholipid_is_not_counted():
    assert _is_mimetic(("C12E5", "5", "%"), sample_type="liquid crystal") is False


def test_cnem_alone_is_a_micelle_former():
    assert _is_mimetic(("C12E5", "5", "%")) is True


def test_alignment_exception_covers_only_the_cnem():
    # the phospholipid of a liquid-crystalline sample, or a DPC micelle beside a
    # CnEm/alkanol medium, still counts
    assert (
        _is_mimetic(("C12E5", "", ""), ("DMPC", "", ""), sample_type="liquid crystal")
        is True
    )
    assert _is_mimetic(("C12E5", "", ""), ("hexanol", "", ""), ("DPC", "1", "mM"))


def test_descriptive_labels_follow_the_class_table_order():
    labels, _ = membrane.classify_samples(
        [_sample(("DMPC", "", ""), ("DHPC", "", ""), ("CHAPS", "1", "mM"))]
    )
    order = list(membrane.MIMETIC_CLASSES_BY_LABEL)
    assert labels == sorted(["phospholipid", "DHPC", "CHAPS"], key=order.index)


@pytest.mark.parametrize(
    ("name", "labels"),
    [
        ("DMPC/DHPC", ["DHPC", "phospholipid"]),  # bmr30931's bicelle
        ("DMPC+CHAPS", ["phospholipid", "CHAPS"]),  # bmr20004
        ("POPC/POPG lipids", ["phospholipid"]),
        ("16:0 Lyso PG", ["LPPG"]),  # ':' is not a mixture separator
        ("1,2-dihexanoyl-sn-glycero-3-phosphocholine", ["DHPC"]),
    ],
)
def test_a_mixture_name_names_each_of_its_parts(name, labels):
    assert membrane.classify_samples([_sample((name, "4", "% w/v"))])[0] == labels


def test_an_excluded_part_does_not_hide_a_mimetic_in_the_mixture():
    # the bile acid is a ligand, the DPC still a micelle former
    assert _is_mimetic(("DPC/sodium cholate", "100", "mM")) is True


# --------------------------------------------------------------------------- #
# fill_row_data: the chain's own samples
# --------------------------------------------------------------------------- #


def _component(name, value="100", unit="mM", entity_id=""):
    return ("1", "1", entity_id, name, value, unit)


def test_row_carries_the_bool_and_the_description():
    row = fill(Entry({"1": Sample([_component("DPC")])}))
    assert row["membrane_mimetic_sample"] is True
    assert row["membrane_mimetic"] == "DPC"


def test_row_without_a_mimetic():
    row = fill(Entry({"1": Sample([_component("sodium phosphate")])}))
    assert row["membrane_mimetic_sample"] is False
    assert pd.isna(row["membrane_mimetic"])


def test_only_the_referenced_sample_is_judged():
    entry = Entry(
        {"1": Sample([_component("NaCl")]), "2": Sample([_component("DPC")])},
        referenced=("1",),
    )
    assert fill(entry)["membrane_mimetic_sample"] is False


def test_no_referenced_sample_falls_back_to_every_sample():
    entry = Entry(
        {"1": Sample([_component("NaCl")]), "2": Sample([_component("DPC")])},
        referenced=(),
    )
    assert fill(entry)["membrane_mimetic_sample"] is True


def test_the_polymer_itself_is_not_a_component():
    # _Sample_component.Entity_ID set: the studied molecule, whatever its name
    row = fill(Entry({"1": Sample([_component("DPC", entity_id="1")])}))
    assert row["membrane_mimetic_sample"] is False


def test_sub_cmc_chaps_chain_is_annotated_but_not_a_mimetic():
    row = fill(Entry({"1": Sample([_component("CHAPS", "1", "mM")])}))
    assert row["membrane_mimetic"] == "CHAPS"
    assert row["membrane_mimetic_sample"] is False


# --------------------------------------------------------------------------- #
# the exclude-membrane-mimetic filter
# --------------------------------------------------------------------------- #


def test_exclude_membrane_mimetic_defaults_per_tier():
    assert filter_defaults["exclude-membrane-mimetic"].tolist() == [
        False,
        False,
        True,
        True,
    ]


@pytest.mark.parametrize("tier", TIERS)
def test_membrane_mimetic_filter(tier):
    df = condition_frame(membrane_mimetic_sample=[False, True])
    expected = [True, tier in ("unfiltered", "tolerant")]
    assert prefilter_tier(df, tier)["pass_pre"].tolist() == expected


def test_membrane_mimetic_filter_is_reported_as_a_prefilter_criterion():
    df = condition_frame(membrane_mimetic_sample=[False, True])
    sels_pre = prefilter_tier(df, "moderate", 2)
    key = next(k for k in sels_pre if k[0] == "membrane mimetic")
    assert sels_pre[key].tolist() == [True, False]


def test_the_filter_needs_the_bool_column():
    df = condition_frame().drop(columns="membrane_mimetic_sample")
    with pytest.raises(ValueError, match="membrane_mimetic_sample"):
        prefilter_tier(df, "strict")
