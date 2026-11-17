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

from tests._stubs import (
    Entry,
    Named,
    Sample,
    condition_frame,
    fill,
    fill_real,
    prefilter_tier,
)
from tests.conftest import requires_bmrb_data
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
    # _Sample_component.Entity_ID set to the studied polymer: its own name, even
    # one the full class table would read as a lipid, says nothing
    row = fill(
        Entry({"1": Sample([_component("lipid transfer protein", entity_id="1")])})
    )
    assert row["membrane_mimetic_sample"] is False


def test_a_mimetic_mislinked_to_the_polymer_is_classified():
    # bmr26040: "DPC", 200 mM, deposited with Entity_ID 1 (the peptide)
    row = fill(Entry({"1": Sample([_component("DPC", entity_id="1")])}))
    assert row["membrane_mimetic_sample"] is True
    assert row["membrane_mimetic"] == "DPC"


def test_a_non_polymer_entity_component_is_classified():
    # bmr16853: DPC deposited as its own non-polymer entity 2
    entry = Entry({"1": Sample([_component("DPC", entity_id="2")])})
    entry.entities["2"] = Named(name="DPC", entity_type="non-polymer")
    row = fill(entry)
    assert row["membrane_mimetic_sample"] is True
    assert row["membrane_mimetic"] == "DPC"


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


# --------------------------------------------------------------------------- #
# release audit C8 (2026-11-17): entity-linked components, sample-level
# evidence and the reviewed title-only entries
# --------------------------------------------------------------------------- #


def _linked(name, entity_id, value="100", unit="mM"):
    """A component carrying an ``_Sample_component.Entity_ID``."""
    return ("1", "1", entity_id, name, value, unit)


def _entities(**types):
    """``{entity ID: entity}``; entity 1 is the studied polymer."""
    ents = {"1": Named(name="protein", entity_type="polymer")}
    ents.update({eid: Named(name=f"e{eid}", entity_type=t) for eid, t in types.items()})
    return ents


def _classify(*samples, entities=None, alignment_samples=frozenset()):
    return membrane.classify_samples(
        list(samples),
        entities=_entities() if entities is None else entities,
        alignment_samples=alignment_samples,
    )


@pytest.mark.parametrize(
    ("component", "label"),
    [
        (("DPC", "200", "mM"), "DPC"),  # bmr16853
        (("Dodecylphosphocholine (DPC) Micelle", "14", "mg"), "DPC"),  # bmr17558
        (
            ("1,2-DIOCTANOYL-SN-GLYCERO-3-PHOSPHOCHOLINE", "", ""),
            "short-chain phospholipid",
        ),
        (("SDS", "40", "mM"), "SDS"),  # bmr51632
        (("CHAPS", "1", "%"), "CHAPS"),  # bmr26756: 16 mM, above the CMC
    ],
)
def test_a_non_polymer_entity_component_is_judged_by_the_full_table(component, label):
    name, value, unit = component
    labels, is_mimetic = _classify(
        Sample([_linked(name, "2", value, unit)]),
        entities=_entities(**{"2": "non-polymer"}),
    )
    assert labels == [label] and is_mimetic is True


def test_a_sub_cmc_non_polymer_additive_is_still_not_counted():
    labels, is_mimetic = _classify(
        Sample([_linked("CHAPS", "2", "1", "mM")]),
        entities=_entities(**{"2": "non-polymer"}),
    )
    assert labels == ["CHAPS"] and is_mimetic is False


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("DPC", "DPC"),  # bmr26040, 200 mM
        ("DPC-d38", "DPC"),  # bmr30059/30060, Hylin analogs
        ("YW12D_SDS", "SDS"),  # bmr15544
        ("HSP12_SDS", "SDS"),
        ("Spike Fusion Peptide, Bicelles", "bicelle"),  # bmr30909
        ("OmpX in MSPdH5 Nanodisc", "nanodisc"),  # bmr27315
        ("tBid in LPPG micelle", "LPPG"),  # bmr19054
        ("DMPC:DHPC(3:1)", "phospholipid"),  # bmr6457
        ("pentameric CorA in DMPC bilayers", "phospholipid"),
        ("Membrane Scaffold Protein", "nanodisc"),
        ("MSP1D1", "nanodisc"),
        ("Capra hircus Cathelicidin-1 (dodecylphosphocholine)", "DPC"),
    ],
)
def test_an_explicit_mimetic_linked_to_a_polymer_is_classified(name, label):
    labels, is_mimetic = _classify(Sample([_linked(name, "1")]))
    assert label in labels and is_mimetic is True


@pytest.mark.parametrize(
    "name",
    [
        # merozoite surface proteins, not the nanodisc scaffold protein
        "MSP1-19",
        "MSP2",
        "1-25MSP2",
        "C-terminal domain of MSP2",
        "MSP-P56S DOMAIN/VAPB",  # a major-sperm-protein domain
        "MSP",  # bare: scaffold or antigen, the name cannot tell
        # Sds3 (a Sin3 corepressor subunit), not SDS
        "Sds3 CTD",
        "Sds3SID",
        # lipid-transfer and lipid-binding proteins
        "Plant Lipid Transfer Protein-2",
        "non-specific lipid transfer protein 1",
        "ileal lipid binding protein",
        # VAP-A, named after the vesicle it binds
        "vesicle-associated membrane protein-associated protein-A",
        "DPCD",  # deleted in primary ciliary dyskinesia
        "DHPCN",  # bmr16401, a dehaloperoxidase construct
        "Dm EDC3",  # Drosophila, not decyl maltoside
        "Dm_DCP1_EVH1_domain",
        "Salmonella SopE2",  # not SOPE
    ],
)
def test_a_protein_name_linked_to_a_polymer_is_not_a_mimetic(name):
    labels, is_mimetic = _classify(Sample([_linked(name, "1")]))
    assert labels == [] and is_mimetic is False


@pytest.mark.parametrize(
    "name",
    [
        "Plant Lipid Transfer Protein-2",
        "merozoite surface protein 2",
        "vesicle-associated membrane protein-associated protein-A",
        "Sds3",
    ],
)
def test_a_protein_name_is_not_a_mimetic_even_without_an_entity_id(name):
    assert membrane.classify_name(name) == []


def test_a_dangling_entity_id_is_judged_like_a_polymer_link():
    # Entity_ID 7 names no entity: only an explicit mimetic name counts
    assert _classify(Sample([_linked("DPC", "7")]))[1] is True
    assert _classify(Sample([_linked("lipid transfer protein", "7")]))[1] is False


@pytest.mark.parametrize(
    "sample_type",
    [
        "micelle",
        "micelles",
        "b_micelles",
        "DPC micelle",
        "bicelle",
        "bi-cell",
        "bicell_solution",
        "liposome",
        "proteoliposome",
        "membrane",
        "oriented membrane film",
        "micelle, lipid bilayers",
        "nanodisc",
    ],
)
def test_the_sample_type_alone_makes_a_membrane_mimetic(sample_type):
    # bmr30170, 11049, 36080 ...: the protein is the only component listed
    sample = Sample([_linked("peptide", "1", "1", "mM")], sample_type=sample_type)
    labels, is_mimetic = _classify(sample)
    assert labels and is_mimetic is True


@pytest.mark.parametrize(
    "sample_type",
    [
        "solution",
        "solid",
        "emulsion",
        "liquid crystal",
        "gel solution",
        "filamentous virus",
        "membrane protein",
        # 34477/34490: a PEG/hexanol alignment medium deposited as a reverse
        # micelle; 36005: calmodulin in the water pool of a CTAB reverse micelle
        "reverse micelle",
    ],
)
def test_other_sample_types_are_not_membrane_mimetics(sample_type):
    sample = Sample([_linked("peptide", "1", "1", "mM")], sample_type=sample_type)
    assert _classify(sample) == ([], False)


@pytest.mark.parametrize(
    ("details", "label"),
    [
        ("in DPC micelles", "DPC"),  # bmr11049
        ("16% deuterated DMPC/DHPC (q=0.3)", "phospholipid"),  # bmr25268
        ("Bacteriorhodopsin incorporated into nanodiscs", "nanodisc"),  # bmr50009
        ("1 mM peptide, 20 mM sodium acetate, 100 mM [U-2H] SDS", "SDS"),  # 36361
        ("1 mM mo1853, 5 mM d-36 Dodecylphosphorylcholine", "DPC"),  # bmr36584
        ("sodium dodecyl sulfate : peptide = 100 : 1 (Molar ratio)", "SDS"),
        ("15N, 13C labelled mini-NS2B in LMPG micelles", "LMPG"),  # bmr51929
        ("micelle-solubilized cytosolic domain", "micelle"),  # bmr30200
    ],
)
def test_explicit_mimetic_words_in_the_sample_details_count(details, label):
    sample = Sample([_linked("peptide", "1", "1", "mM")], details=details)
    labels, is_mimetic = _classify(sample)
    assert label in labels and is_mimetic is True


@pytest.mark.parametrize(
    ("field", "text", "label"),
    [
        ("solvent_system", "SDS", "SDS"),  # bmr15853
        ("solvent_system", "90% H2O/10% D2O + dodecylphosphocholine", "DPC"),
        ("name", "Ost4 reconstituted in POPC and POPE", "phospholipid"),  # 52090
    ],
)
def test_explicit_mimetic_words_in_the_solvent_system_or_name_count(field, text, label):
    sample = Sample([_linked("peptide", "1", "1", "mM")], **{field: text})
    labels, is_mimetic = _classify(sample)
    assert label in labels and is_mimetic is True


@pytest.mark.parametrize(
    "details",
    [
        # bmr50217: a non-detergent sulfobetaine is not a detergent
        "0.5 M Non-detergent sulfobetaine (NDSB)-195 buffer",
        "in the absence of SDS",
        "without DPC",
        "SDS-free buffer",
        "1 mM CHAPS",  # an additive detergent needs a concentration: components only
        "lipid-free apoA-I",  # 'lipid' is not an explicit mimetic in free text
        "Sds3 C-terminal domain",
        "protein in CTAB reverse micelles",
        "20 mM sodium phosphate, 50 mM NaCl, 90% H2O/10% D2O",
    ],
)
def test_other_sample_details_are_not_evidence(details):
    sample = Sample([_linked("peptide", "1", "1", "mM")], details=details)
    assert _classify(sample) == ([], False)


def test_an_alignment_sample_is_not_judged_by_its_type_or_details():
    # bmr30410: samples 3 and 4 are typed "bicelle" but serve only the
    # anisotropic IPAP-HSQC (RDC) experiments; the shifts are isotropic
    rdc = Sample(
        [_linked("Mpp11", "1", "200", "uM")], sample_type="bicelle", sample_id="3"
    )
    assert _classify(rdc, alignment_samples=frozenset({"3"})) == ([], False)
    assert _classify(rdc)[1] is True


def test_an_alignment_sample_component_still_counts():
    # the component rule is unchanged: a bicelle listed as a component counts
    rdc = Sample([_component("DMPC")], sample_type="bicelle", sample_id="3")
    assert _classify(rdc, alignment_samples=frozenset({"3"}))[1] is True


@pytest.mark.parametrize(
    ("experiments", "aligned"),
    [
        # (ID, name, raw-data flag, Sample_ID, label, Sample_state)
        ([("1", "HSQC", "no", "1", "", "isotropic")], set()),
        (
            [
                ("1", "HNCA", "no", "1", "", "isotropic"),
                ("2", "IPAP", "no", "2", "", "anisotropic"),
                ("3", "IPAP", "no", " 3 ", "", "Anisotropic"),
            ],
            {"2", "3"},
        ),
        (
            [
                ("1", "HSQC", "no", "2", "", "isotropic"),
                ("2", "IPAP", "no", "2", "", "anisotropic"),
            ],
            set(),
        ),
        ([("1", "HSQC", "no", "2", "", "")], set()),
        # bmr34738: every experiment marked anisotropic -- the shifts were
        # measured in those samples, none is an alignment sample
        (
            [
                ("1", "HSQC", "no", "1", "", "anisotropic"),
                ("2", "HNCO", "no", "2", "", "anisotropic"),
            ],
            set(),
        ),
    ],
)
def test_alignment_samples_are_used_only_by_anisotropic_experiments(
    experiments, aligned
):
    assert membrane.alignment_sample_ids(experiments) == aligned


def test_fill_skips_the_sample_level_evidence_of_an_alignment_sample():
    # bmr30410: the shift table lists the isotropic assignment sample 1 and,
    # through its RDC experiment, the bicelle-typed alignment sample 2
    iso = Sample([_component("Mpp11", entity_id="1")], sample_id="1")
    rdc = Sample(
        [_component("Mpp11", entity_id="1")], sample_type="bicelle", sample_id="2"
    )
    entry = Entry({"1": iso, "2": rdc}, referenced=("1", "2"))
    entry.experiment_list.experiments = [
        ("1", "3D HNCA", "no", "1", "", "isotropic"),
        ("2", "1H-15N IPAP HSQC", "no", "2", "", "anisotropic"),
    ]
    assert fill(entry)["membrane_mimetic_sample"] is False


# --------------------------------------------------------------------------- #
# reviewed title-only entries: the committed override table
# --------------------------------------------------------------------------- #


def test_the_committed_override_table_is_well_formed():
    overrides = membrane.load_overrides()
    assert overrides, "the reviewed title-only entries must not be empty"
    for key, override in overrides.items():
        assert key == override.id and key.split("_")[0].isdigit()
        assert len(key.split("_")) in (1, 4)
        assert override.label in membrane.MIMETIC_CLASSES_BY_LABEL
        assert override.reason.strip()


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("id\tclass\n1\tDPC\treason\n", "header"),
        ("id\tclass\treason\n1\tsoap\ttitle\n", "class"),
        ("id\tclass\treason\n1\tDPC\ttitle\n1\tSDS\ttitle\n", "duplicate"),
        ("id\tclass\treason\n1\tDPC\t\n", "reason"),
        ("id\tclass\treason\nbmr1\tDPC\ttitle\n", "ID"),
        ("id\tclass\treason\n1_1\tDPC\ttitle\n", "ID"),
    ],
)
def test_a_malformed_override_table_is_rejected(tmp_path, text, error):
    path = tmp_path / "overrides.tsv"
    path.write_text(text)
    with pytest.raises(ValueError, match=error):
        membrane.load_overrides(path)


def test_an_overridden_entry_is_a_membrane_mimetic(monkeypatch):
    override = membrane.MembraneOverride("1", "DPC", "title: 'in DPC micelles'")
    monkeypatch.setattr(membrane, "load_overrides", lambda path=None: {"1": override})
    row = fill(Entry({"1": Sample([_component("sodium phosphate")])}))
    assert row["membrane_mimetic_sample"] is True
    assert row["membrane_mimetic"] == "DPC"


def test_a_chain_override_marks_only_that_chain(monkeypatch):
    # bmr6857: shift list 3 is the DPC one (conditions "FF2 in DPC"), though
    # its experiment cites the aqueous sample 1 that shift list 1 also uses
    override = membrane.MembraneOverride("1_3_1_1", "DPC", "conditions_3")
    monkeypatch.setattr(
        membrane, "load_overrides", lambda path=None: {"1_3_1_1": override}
    )
    entry = Entry(
        {"1": Sample([_component("KCl")])},
        peptide_shifts={
            ("1", "1", "1"): (None, "1", "1", ["1"]),
            ("3", "1", "1"): (None, "1", "1", ["1"]),
        },
    )
    assert fill(entry, st_id="3")["membrane_mimetic_sample"] is True
    assert fill(entry, st_id="1")["membrane_mimetic_sample"] is False


def test_an_entry_not_in_the_table_is_untouched(monkeypatch):
    monkeypatch.setattr(membrane, "load_overrides", lambda path=None: {})
    row = fill(Entry({"1": Sample([_component("sodium phosphate")])}))
    assert row["membrane_mimetic_sample"] is False


# --------------------------------------------------------------------------- #
# deposited entries (release audit C8)
# --------------------------------------------------------------------------- #


@requires_bmrb_data
@pytest.mark.parametrize(
    ("entry_id", "expected", "label"),
    [
        ("26040", True, "DPC"),  # DPC 200 mM linked to the peptide entity
        ("16853", True, "DPC"),  # DPC as its own non-polymer entity
        ("11049", True, "DPC"),  # _Sample.Type micelle, Details "in DPC micelles"
        ("15853", True, "SDS"),  # _Sample.Solvent_system "SDS" (shift table 3)
        ("34738", True, "micelle"),  # micelle samples, every experiment anisotropic
        ("30761", True, "DPC"),  # title only: reviewed override
        ("30410", False, None),  # bicelle samples serve only the RDC experiments
        ("30969", False, None),  # "Sds3 CTD"
        ("5325", False, None),  # "Plant Lipid Transfer Protein-2"
    ],
)
def test_deposited_entries(entry_id, expected, label):
    st_id = "3" if entry_id == "15853" else None
    row = fill_real(entry_id, st_id)
    assert row["membrane_mimetic_sample"] is expected
    if label is not None:
        assert label in row["membrane_mimetic"].split(";")


# --------------------------------------------------------------------------- #
# release validation 2026-11-17c, C8: the 28 undetected moderate/strict chains
# --------------------------------------------------------------------------- #

#: Reviewed against the raw NMR-STAR files of the 2026-11-11 snapshot: the
#: mimetic is named in the file but outside the class table or the fields the
#: rule reads (detergents the table lacks, a detergent in _Sample.Details, a
#: chain name).
C8_DEFINITE = {
    "27093": "micelle",  # sulfobetaine-12 30-50 mM
    "18026": "OG",  # 50 mM octyl glucoside in _Sample.Details
    "51492": "micelle",  # 3 % SB3-10
    "26936": "micelle",  # LMNG 0.01 % + CHS
    "26937": "micelle",
    "27502": "micelle",  # C14 betaine 100 mM
    "19603": "DPC",  # 'ctrdpc', twin of 19597 (SDS)
}
#: Integral membrane proteins measured by solution NMR whose deposition names
#: no mimetic (probable): class "unnamed mimetic".
C8_PROBABLE = {
    "19637",
    "51063",
    "18648",
    "15019",
    "7340",
    "19126",
    "19994",
    "27438",
    "18494",
    "19379",
    "26617",
    "51026",
    "18706",
    "16497",
    "26624",
}
#: Reviewed and not listed: no mimetic evidence that attaches to the scored
#: sample (5376: free peptide over native disk membranes, the shift list cites
#: no sample; 4151 and 6056: probably organic solvent, not a mimetic).
C8_REJECTED = {"5376", "4151", "6056"}


def test_the_c8_definite_residuals_are_overridden_with_their_class():
    overrides = membrane.load_overrides()
    for key, label in C8_DEFINITE.items():
        assert overrides[key].label == label, key


def test_the_c8_probable_membrane_proteins_carry_the_unnamed_class():
    overrides = membrane.load_overrides()
    for key in C8_PROBABLE:
        assert overrides[key].label == "unnamed mimetic", key
        assert "probable" in overrides[key].reason, key


def test_the_rejected_c8_residuals_are_documented_not_listed():
    overrides = membrane.load_overrides()
    text = membrane.MEMBRANE_OVERRIDES_TSV.read_text()
    comments = "".join(ln for ln in text.splitlines(True) if ln.startswith("#"))
    for key in C8_REJECTED | {"7176"}:
        assert key not in overrides, key
        assert key in comments, key


def test_the_unnamed_class_matches_no_component_or_text():
    cls = membrane.MIMETIC_CLASSES_BY_LABEL["unnamed mimetic"]
    for name in ("unnamed mimetic", "membrane", "micelle", "DPC", "", "protein"):
        assert membrane.classify_component(name) is not cls
    assert cls not in membrane.explicit_classes("in an unnamed mimetic")


def test_an_unnamed_override_flags_the_chain(monkeypatch):
    override = membrane.MembraneOverride("1", "unnamed mimetic", "probable")
    monkeypatch.setattr(membrane, "load_overrides", lambda path=None: {"1": override})
    row = fill(Entry({"1": Sample([_component("sodium phosphate")])}))
    assert row["membrane_mimetic_sample"] is True
    assert row["membrane_mimetic"] == "unnamed mimetic"
