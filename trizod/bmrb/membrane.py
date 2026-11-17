"""Membrane-mimetic samples: which detergents and lipids count, and when.

A chain's sample is a membrane mimetic when any non-polymer
``_Sample_component`` of the samples its shift table references (every sample
of the entry if it references none) meets one of three conditions:

(a) **bilayer** -- a bilayer-forming phospholipid or bilayer assembly, at any
    concentration: diacyl or ether PC, PG, PE, PS or PA with chains >= C10,
    cardiolipin, lipid extracts, bicelles, nanodiscs/MSP, liposomes/vesicles,
    and lipopolysaccharide (LPS) micelles and bilayers, the outer-membrane
    mimetics of the antimicrobial-peptide depositions (re-audit 2026-11-17b,
    C8: one LPS policy, by rule; overrides only where the LPS is named outside
    the sample);
(b) **micelle** -- a dedicated micelle former, at any concentration: SDS,
    alkylphosphocholines (DPC, FC-10/12/14/16), lysophospholipids (LMPG, LPPG,
    lyso-PC), short-chain phospholipids (DHPC 06:0, 07:0, 08:0), LDAO, lauroyl
    sarcosine, CnEm, the cationic DTAC/CTAB, or a component named "micelles";
(c) **additive** -- a detergent that is also used as a solubility additive,
    only at a deposited concentration >= its CMC in water, or when no
    concentration can be converted (an amount, a bare ratio, nothing).
    Digitonin counts whenever present.

Why three classes and not one CMC test: CMCs span three orders of magnitude
(LPPG 0.018 mM to octyl glucoside 18 mM), and sub-CMC SDS already induces
helix in disordered chains (dehydrin K2, alpha-synuclein, XAO). A CMC test on a
classic mimetic also turns a deposition unit error into a false "aqueous"
label: bmr21013 is titled "DPC micelle bound RI23" but lists DPC at "10 uM".
The additive class exists for soluble proteins solubilised by 0.05-10 mM
CHAPS, CHAPSO, octyl glucoside or Triton (NALP1 PYD with 1 mM CHAPS, 16670).

**Exception** -- CnEm/n-alkanol liquid-crystal alignment media (31134:
C12E5/hexanol, 25324: C8E5/octanol): a CnEm component is not counted when its
sample also holds an n-alkanol, or is of ``_Sample.Type`` "liquid crystal"
without any phospholipid.

Not mimetics, whatever the name suggests: bile-acid ligands (cholate,
glycochenodeoxycholate), acyl-CoA, lysozyme, lipid II, sarcosine itself,
anti-lipopolysaccharide factors and proteins of LPS binding, transport,
export or assembly (LBP, LptA, LptD). A
mixture deposited as one component name ("DMPC/DHPC", "DMPC+CHAPS") is
classified part by part; each part is assigned the first class of
``MIMETIC_CLASSES`` whose pattern it matches.

Water CMCs are upper bounds for ionic detergents (they fall in salt), and the
CHAPSO CMC is taken equal to CHAPS's. Recall depends on a name regex that
future depositions will outgrow; ``MIMETIC_CLASSES`` is the one place to
extend it.

Which components and which sample fields are read (release audit C8,
2026-11-17):

* A component without ``_Sample_component.Entity_ID``, or linked to an entity
  whose ``_Entity.Type`` is not ``polymer`` (a detergent deposited as its own
  non-polymer entity, bmr16853/26756), is classified by the whole table.
* A component linked to a polymer entity is normally the studied protein,
  named however the depositor liked ("Plant Lipid Transfer Protein-2",
  "MSP1-19", "Sds3 CTD"). It counts only when its name holds an *explicit*
  mimetic token (``EXPLICIT_MIMETIC``: micelle, bicelle, nanodisc, liposome,
  SDS, DPC, DHPC, LPPG, LMPG, LDAO, a phospholipid acronym such as DMPC, the
  nanodisc scaffolds MSP1D1/MSPdH5 ...), bounded so that Sds3, DPCD, DHPCN or
  SopE2 do not match: bmr26040 lists "DPC" (200 mM) with Entity_ID 1, bmr30909
  "Spike Fusion Peptide, Bicelles". An Entity_ID that names no entity is
  judged the same way.
* Sample-level fallback: ``_Sample.Type`` (micelle, bicelle, liposome,
  membrane, nanodisc ..., but not "reverse micelle" or "liquid crystal") and
  explicit mimetic tokens in ``_Sample.Details``, ``_Sample.Solvent_system``
  or ``_Sample.Name`` (not negated: "absence of", "without", "-free") make the
  sample a mimetic when no component says so (bmr11049: "in DPC micelles", the
  peptide the only component; bmr15853: solvent system "SDS"). A sample
  every experiment of which is anisotropic (an RDC alignment sample,
  bmr30410's bicelle samples) is not judged by this fallback: its type
  describes the alignment medium, not the medium the shifts were measured in.
* Entries whose only evidence is the title were reviewed against the raw
  files; the genuine ones are listed in the committed
  ``trizod/dataset/data/membrane_overrides.tsv`` (:func:`load_overrides`).
  Titles are never matched automatically.

Known recall limits: a mimetic named only in free text the rule does not read
(the citation, a bare "detergent"), or only in a title not yet reviewed; and a
bare "MSP" linked to a polymer (scaffold protein or merozoite antigen, the
name cannot tell -- bmr30639). Known false positive: bmr34151.
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass
from pathlib import Path

from trizod.bmrb.bmrb import parse_condition_value

BILAYER, MICELLE, ADDITIVE = "bilayer", "micelle", "additive"


@dataclass(frozen=True)
class MimeticClass:
    """One class of the membrane-mimetic table.

    ``cmc_millimolar`` is the critical micelle concentration in water (mM). It decides
    only for ``kind == "additive"``; for those, None means "counts whenever
    present" (digitonin). ``mw`` (g/mol) converts %, mg/mL and ug/mL to mM.
    """

    label: str
    kind: str
    cmc_millimolar: float | None
    mw: float | None
    pattern: str

    def __post_init__(self):
        assert self.kind in (BILAYER, MICELLE, ADDITIVE), self.kind


# Shared fragments. Names are matched lower-cased; "not preceded/followed by a
# letter" is used instead of \b, because deposited names glue labels to
# isotope tags and units ("DPC-d38", "d54-DMPC", "DMPC(D54)").
_NOT_AFTER = r"(?<![a-z])"
_NOT_BEFORE = r"(?![a-z])"
_PHOSPHOCHOLINE = r"[\s-]*phos(?:pho-?ryl|pho|ph)?-?c(?:h)?olin"
#: Lipopolysaccharide, also as the acronym glued to a peptide name ("WR17_LPS",
#: "YI12FF+LPS", "CDT-LPS"), but not an anti-LPS factor or a protein of LPS
#: binding, transport, export or assembly (LBP, the Lpt machinery).
_LPS = (
    r"(?<![a-z])(?<!anti-)(?<!anti )(?:lps|lipopolysacch?arides?)(?![a-z0-9])"
    r"(?![\s_-]*(?:binding|transport|export|assembly|sensing))"
)

#: THE class table: regex -> (label, kind, CMC in water, molar mass). Order is
#: precedence -- the first pattern that matches a component name decides its
#: class -- and the order labels are reported in. Lyso and short-chain lipids
#: come before the bilayer lipids, ``dodecyl`` detergents before each other's
#: generic forms, and the bare word "micelle" last among the micelle formers.
MIMETIC_CLASSES: tuple[MimeticClass, ...] = (
    # --- (b) micelle formers: any concentration ------------------------------
    MimeticClass(
        "SDS",
        MICELLE,
        8.2,
        288.4,
        # also the deposited "sodium dodecyl sulfphate" (bmr34388)
        # not "Sds3" (a Sin3 corepressor subunit, bmr30969)
        rf"{_NOT_AFTER}sds(?![a-z0-9])|(?:dodecyl|lauryl)[\s-]*sul[fp]",
    ),
    MimeticClass(
        "LPPG",
        MICELLE,
        0.018,
        507.0,
        r"lppg|lyso\w*[\s-]*palmit|palmit\w*[\s-]*lyso|16:0[\s-]*lyso[\s-]*pg"
        r"|palmit\w*-2-hydroxy.*glycerol",
    ),
    MimeticClass(
        "LMPG",
        MICELLE,
        0.16,
        478.5,
        r"lmpg|lyso\w*[\s-]*myrist|myrist\w*[\s-]*lyso|14:0[\s-]*lyso[\s-]*pg"
        r"|myrist\w*-2-hydroxy.*glycerol",
    ),
    MimeticClass(
        "LPC",
        MICELLE,
        None,
        None,
        rf"lyso\w*[\s-]*pc{_NOT_BEFORE}|lyso\w*[\s-]*phosphatidyl[\s-]*cholin"
        r"|(?:lauroyl|myristoyl|palmitoyl|oleoyl|stearoyl)-2-hydroxy-sn-glycero"
        r"-3-phosphocholin",
    ),
    MimeticClass(
        "FC-16",
        MICELLE,
        0.013,
        407.6,
        rf"hexadecyl{_PHOSPHOCHOLINE}|fos-?choline-?16|{_NOT_AFTER}fc-?16(?!\d)",
    ),
    MimeticClass(
        "FC-14",
        MICELLE,
        0.12,
        379.5,
        rf"tetradecyl{_PHOSPHOCHOLINE}|fos-?choline-?14|{_NOT_AFTER}fc-?14(?!\d)",
    ),
    MimeticClass(
        "DPC",
        MICELLE,
        1.5,
        351.5,
        # dodecylphosphocholine and the deposited misspellings "Dodecyl
        # Phosphcholine", "dodecyphosphocholine", "dodecylphosphocoline"
        rf"{_NOT_AFTER}dpc{_NOT_BEFORE}|dodec[a-z]*{_PHOSPHOCHOLINE}"
        rf"|fos-?choline-?12|{_NOT_AFTER}fc-?12(?!\d)",
    ),
    MimeticClass(
        "FC-10",
        MICELLE,
        11.0,
        323.4,
        rf"{_NOT_AFTER}decyl{_PHOSPHOCHOLINE}|fos-?choline-?10"
        rf"|{_NOT_AFTER}fc-?10(?!\d)",
    ),
    MimeticClass(
        "DH7PC",
        MICELLE,
        1.4,
        481.6,
        rf"dihept|{_NOT_AFTER}dh7pc|07:0[\s-]*pc{_NOT_BEFORE}",
    ),
    MimeticClass(
        "DHPC",
        MICELLE,
        15.0,
        453.5,
        rf"dhpc|dihexan|06:0[\s-]*pc{_NOT_BEFORE}",
    ),
    MimeticClass(
        "short-chain phospholipid",
        MICELLE,
        None,
        480.0,
        rf"dioctanoyl|08:0[\s-]*p[cg]{_NOT_BEFORE}",
    ),
    MimeticClass(
        "LDAO",
        MICELLE,
        1.0,
        229.4,
        r"ldao|ddao|(?:lauryl|dodecyl)[\s-]*dimethyl[\s-]*amine"
        r"|dimethyl[\s-]*dodecyl[\s-]*amine[\s-]*(?:n-)?oxide",
    ),
    MimeticClass(
        "lauroyl sarcosine",
        MICELLE,
        None,
        293.4,
        r"lauroyl[\s-]*sarcosin|sarcosyl|sarkosyl",
    ),
    MimeticClass(
        "CnEm",
        MICELLE,
        None,
        None,
        # C12E5, C8E5, C12E8 ... and their written-out names
        r"(?<![a-z0-9])c(?:[6-9]|1[0-8])e\d{1,2}(?!\d)"
        r"|(?:tetra|penta|hexa|octa)ethylene[\s-]*glycol[\s-]*mono[\s-]*"
        r"(?:octyl|decyl|dodecyl)[\s-]*ether|octyl[\s-]*pentaoxyethylene",
    ),
    MimeticClass(
        "DTAC",
        MICELLE,
        None,
        280.0,
        rf"dodecyl[\s-]*trimethyl[\s-]*ammonium|{_NOT_AFTER}dtac{_NOT_BEFORE}",
    ),
    MimeticClass(
        "CTAB",
        MICELLE,
        None,
        364.5,
        rf"(?:cetyl|hexadecyl)[\s-]*trimethyl[\s-]*ammonium"
        rf"|{_NOT_AFTER}cta[bc]{_NOT_BEFORE}",
    ),
    MimeticClass("micelle", MICELLE, None, None, r"micell"),
    # --- (a) bilayer lipids and assemblies: any concentration ----------------
    MimeticClass(
        "phospholipid",
        BILAYER,
        None,
        700.0,
        # acronyms (DMPC, DPPC, DOPC, POPC, DPhPC, ... -PG/-PE/-PS; DMPA/POPA;
        # not DOPA, which is also L-DOPA), "14:0 PC", the acyl names of the
        # IUPAC spellings, phosphatidyl-X, cardiolipin, lipid extracts
        r"(?<![a-z0-9])(?:d(?:m|p|l|o|s|ph)|po|so)p[cges](?![a-z])"
        r"|(?<![a-z0-9])(?:dm|dp|po)pa(?![a-z])"
        r"|(?<![\d:])(?:1\d|2[0-4]):\d+(?:[\s-]+(?:1\d|2[0-4]):\d+)?[\s-]*p[cgesa]"
        rf"{_NOT_BEFORE}"
        r"|dimyrist|dipalmit|diphytan|dioleoyl|dilauroyl|distearoyl"
        r"|di-?o-?(?:tetra|hexa)decyl|palmitoyl-?(?:2-)?oleoyl"
        r"|phosphatidyl[\s-]*(?:cholin|glycerol|ethanolamin|serin)"
        r"|phosphatidic acid|cardiolipin|asolectin|phospholipid"
        rf"|{_NOT_AFTER}lipids?(?!\s*i+{_NOT_BEFORE}){_NOT_BEFORE}",
    ),
    MimeticClass("bicelle", BILAYER, None, None, r"bi-?_?cell"),
    MimeticClass(
        "nanodisc",
        BILAYER,
        None,
        None,
        # MSP1D1, MSP1E3D1, MSP2N2, MSPdH5 (bmr27315) ... the scaffold proteins
        rf"nano-?disc|{_NOT_AFTER}msp(?:\d|{_NOT_BEFORE}|dh\d)|membrane scaffold protein",
    ),
    MimeticClass(
        "liposome",
        BILAYER,
        None,
        None,
        rf"liposom|vesicle|{_NOT_AFTER}[sl]uvs?{_NOT_BEFORE}",
    ),
    MimeticClass("amphipol", BILAYER, None, None, r"amphipol"),
    # Lipopolysaccharide micelles and bilayers, the outer-membrane mimetics of
    # the antimicrobial-peptide depositions (re-audit 2026-11-17b, C8): any
    # concentration, like every bilayer lipid. After the assemblies, so "LPS
    # micelles" and "lipopolysaccharide bicelle" keep their assembly label.
    MimeticClass("LPS", BILAYER, None, None, _LPS),
    # A name (or a _Sample.Type) that is just a membrane: "membrane", "lipid
    # membranes", "oriented membrane film". Anchored, so "outer membrane
    # protein A" or a "membrane protein" sample type are not one.
    MimeticClass(
        "membrane",
        BILAYER,
        None,
        None,
        r"^\s*(?:(?:lipid|oriented|native)\s+)?membranes?(?:\s+film)?\s*$",
    ),
    # --- (c) additive-class detergents: >= CMC or no concentration ----------
    MimeticClass("CHAPSO", ADDITIVE, 8.0, 630.9, r"chapso"),
    MimeticClass("CHAPS", ADDITIVE, 8.0, 614.9, r"chaps|cholamidopropyl"),
    MimeticClass(
        "DDM",
        ADDITIVE,
        0.17,
        510.6,
        rf"(?:dodecyl|lauryl)[^,;]*malto|{_NOT_AFTER}ddm{_NOT_BEFORE}",
    ),
    MimeticClass(
        "DM",
        ADDITIVE,
        1.8,
        482.6,
        rf"{_NOT_AFTER}decyl[^,;]*malto|(?<![a-z0-9])dm(?![a-z0-9])",
    ),
    MimeticClass("OM", ADDITIVE, 19.5, 454.5, r"octyl[^,;]*malto"),
    MimeticClass(
        "OG",
        ADDITIVE,
        18.0,
        292.4,
        r"octyl[^,;]*gluc|(?<![a-z0-9])og(?![a-z0-9])",
    ),
    MimeticClass("Triton X-100", ADDITIVE, 0.24, 625.0, r"triton"),
    MimeticClass("Tween-20", ADDITIVE, 0.06, 1228.0, r"tween|polysorbate"),
    MimeticClass("digitonin", ADDITIVE, None, 1229.3, r"digitonin"),
)

MIMETIC_CLASSES_BY_LABEL = {cls.label: cls for cls in MIMETIC_CLASSES}
_COMPILED = tuple((re.compile(cls.pattern), cls) for cls in MIMETIC_CLASSES)

#: Names that contain a class pattern but are not mimetics: bile-acid ligands
#: (glycochenodeoxycholate alone is 34 component rows), acyl-CoA, lysozyme, and
#: proteins named after a lipid or a vesicle (lipid-transfer and lipid-binding
#: proteins, VAP-A "vesicle-associated membrane protein-associated protein").
NOT_A_MIMETIC = re.compile(
    r"cholic|cholan|cholate|coenzyme|(?<![a-z])coa(?![a-z])|lysozyme"
    r"|lipid[\s-]*(?:transfer|binding)|vesicle[\s-]*associated"
)

#: An EXPLICIT membrane-mimetic token: a word that names a micelle, bicelle,
#: nanodisc or liposome, or a dedicated micelle former or bilayer lipid by its
#: unambiguous name or acronym. Every alternative is classified by
#: ``classify_component`` on its own. The acronyms are bounded on both sides
#: by anything but a letter or digit, so a protein named "Sds3", "DPCD",
#: "DHPCN" or "SopE2" is no match; a bare "lipid", "vesicle", "MSP" or "DM" is
#: not explicit (lipid-transfer protein, VAP-A, merozoite surface protein,
#: Drosophila). Read where a name may be a protein's (a component linked to a
#: polymer entity) and in free text (``_Sample.Details``).
EXPLICIT_MIMETIC = re.compile(
    r"micell|bi-?_?cell|nano-?disc|liposom|membrane scaffold protein"
    r"|(?<![a-z])msp\d*(?:d\d|e\d|dh\d|n\d)"
    r"|(?<![a-z])(?:sds|dpc|dhpc|lppg|lmpg|ldao)(?![a-z0-9])"
    r"|(?:dodecyl|lauryl)[\s-]*sul[fp]"
    rf"|dodec[a-z]*{_PHOSPHOCHOLINE}"
    r"|(?<![a-z0-9])(?:d(?:m|p|l|o|s|ph)|po|so)p[cges](?![a-z0-9])"
    rf"|{_LPS}"
)

#: A reverse micelle encapsulates a protein in its water pool (calmodulin in
#: CTAB, bmr36005), and two depositions call a PEG/hexanol alignment medium one
#: (bmr34477/34490): never sample-level evidence.
_REVERSE_MICELLE = re.compile(r"reverse[\s_-]*micell\w*")

#: A mention in free text that is negated: "in the absence of SDS", "without
#: DPC" (the words just before), "SDS-free" (just after).
_NEGATED_BEFORE = re.compile(r"(?:absence\s+of|without|free\s+of)\W+(?:\S+\W+){0,3}$")
_NEGATED_AFTER = re.compile(r"^\S*?-?free(?![a-z])")

#: Sample types that are never sample-level evidence, whatever they contain.
_NOT_A_MIMETIC_SAMPLE_TYPE = re.compile(r"reverse[\s_-]*micell|liquid crystal")

#: An n-alkanol co-surfactant of a CnEm alignment medium (hexanol, octanol,
#: n-hexanol, 1-octanol, hexan-1-ol ...).
N_ALKANOL = re.compile(
    r"(?<![a-z])(?:n-|1-)?(?:pent|hex|hept|oct|non|dec|dodec)an(?:-1-)?ol(?![a-z])"
)

#: Concentration unit -> factor to mM (molar units) ...
_MOLAR_TO_MM = {"m": 1e3, "mm": 1.0, "um": 1e-3, "nm": 1e-6}
#: ... or factor to mg/mL, divided by the molar mass for mM (mass units).
#: A percentage is taken as g/100 mL, w/v and w/w alike for dilute samples.
_MASS_TO_MG_PER_ML = {
    "%": 10.0,
    "%w/v": 10.0,
    "%(w/v)": 10.0,
    "w/v%": 10.0,
    "%w/w": 10.0,
    "%(w/w)": 10.0,
    "%v/v": 10.0,
    "mg/ml": 1.0,
    "g/l": 1.0,
    "ug/ml": 1e-3,
    "mg/l": 1e-3,
    "mg/ul": 1e3,  # implausible, but a concentration: converted literally
}


def _unit_key(unit) -> str:
    """'uM'/'µM'/'μM'/'microM' -> 'um'; whitespace removed, lower case."""
    key = "".join(str(unit or "").split()).lower()
    key = key.replace("µ", "u").replace("μ", "u")
    return "u" + key[len("micro") :] if key.startswith("micro") else key


def concentration_mM(value, unit, mw=None) -> float:
    """A deposited component concentration in mM, NaN when it is none.

    M, mM, uM (and nM) convert directly; %, mg/mL and ug/mL through the molar
    mass ``mw``. Amounts (mg, umol), bare ratios (w/w, eq), an unknown unit or
    an unparseable value are not concentrations: NaN.
    """
    number, unit_in_value = parse_condition_value(value)
    if math.isnan(number):
        return math.nan
    key = _unit_key(unit if unit not in (None, "", ".") else unit_in_value)
    if key in _MOLAR_TO_MM:
        return number * _MOLAR_TO_MM[key]
    if key in _MASS_TO_MG_PER_ML and mw:
        return number * _MASS_TO_MG_PER_ML[key] * 1000.0 / mw
    return math.nan


def classify_component(name) -> MimeticClass | None:
    """The membrane-mimetic class of a ``_Sample_component.Mol_common_name``."""
    if not name:
        return None
    text = str(name).lower()
    if NOT_A_MIMETIC.search(text):
        return None
    for pattern, cls in _COMPILED:
        if pattern.search(text):
            return cls
    return None


def component_counts(cls: MimeticClass, value, unit) -> bool:
    """Does a component of class ``cls`` make its sample a membrane mimetic?"""
    if cls.kind != ADDITIVE or cls.cmc_millimolar is None:
        return True
    conc = concentration_mM(value, unit, cls.mw)
    return math.isnan(conc) or conc >= cls.cmc_millimolar


def _entity_type(entities, entity_id) -> str | None:
    """``_Entity.Type`` (lower case) of the entity a component links to, None
    when the ID names no entity (or no entity table was given)."""
    entity = (entities or {}).get(str(entity_id).strip())
    etype = getattr(entity, "type", None)
    return str(etype).strip().lower() if etype else None


def _components(sample, entities=None):
    """``(name, value, unit, explicit_only)`` of every named component of a
    sample that may be a mimetic.

    ``explicit_only`` is False for a component without ``Entity_ID`` or linked
    to a non-polymer entity (the whole class table applies) and True for one
    linked to a polymer entity, or to an ID that names no entity: that is
    normally the studied protein, so only ``EXPLICIT_MIMETIC`` tokens count.
    """
    for comp in sample.components:
        name, entity_id = comp[3], comp[2]
        if not name:
            continue
        if not str(entity_id or "").strip():
            yield name, comp[4], comp[5], False
            continue
        etype = _entity_type(entities, entity_id)
        yield name, comp[4], comp[5], etype in (None, "polymer")


def explicit_classes(text) -> list[MimeticClass]:
    """The classes of the ``EXPLICIT_MIMETIC`` tokens in ``text`` (a name or
    free text), in order of appearance; a reverse micelle and a negated
    mention ("absence of", "without", "-free") are no evidence."""
    text = _REVERSE_MICELLE.sub(" ", str(text or "").lower())
    found = []
    for match in EXPLICIT_MIMETIC.finditer(text):
        if _NEGATED_BEFORE.search(text[: match.start()]) or _NEGATED_AFTER.match(
            text[match.end() :]
        ):
            continue
        cls = classify_component(match.group(0))
        if cls is not None:
            found.append(cls)
    return found


#: Separators of a mixture written as one component name: "DMPC/DHPC" (a
#: bicelle), "DMPC+CHAPS". Not ':' (``14:0 PC``, ``16:0 Lyso PG``) and not ','
#: (``1,2-dimyristoyl-...``).
_MIXTURE_SEPARATOR = re.compile(r"\s*[/+;]\s*")


def classify_name(name) -> list[MimeticClass]:
    """The classes of a component name, one per part of a mixture name, in
    the order of the parts (a part may yield none)."""
    parts = _MIXTURE_SEPARATOR.split(str(name or ""))
    return [cls for cls in map(classify_component, parts) if cls is not None]


def classify_sample(sample, entities=None) -> list[tuple[MimeticClass, bool]]:
    """``(class, counts)`` for every recognised mimetic component of a sample
    (every part of a mixture name). A mixture's parts share its concentration,
    which only matters for an additive-only mixture. ``entities`` (``{entity
    ID: entity}``) tells a non-polymer entity's component, judged by the whole
    table, from one linked to a polymer, judged by its explicit tokens only."""
    found, names = [], []
    for name, value, unit, explicit_only in _components(sample, entities):
        names.append(str(name).lower())
        classes = explicit_classes(name) if explicit_only else classify_name(name)
        for cls in classes:
            found.append((cls, component_counts(cls, value, unit)))
    has_phospholipid = any(cls.kind == BILAYER for cls, _counts in found)
    alignment_medium = any(N_ALKANOL.search(n) for n in names) or (
        str(getattr(sample, "type", "") or "").strip().lower() == "liquid crystal"
        and not has_phospholipid
    )
    return [
        (cls, counts and not (alignment_medium and cls.label == "CnEm"))
        for cls, counts in found
    ]


#: The free-text fields of a sample searched for explicit mimetic tokens:
#: ``_Sample.Details`` ("in DPC micelles", bmr11049), ``_Sample.Solvent_system``
#: ("SDS", bmr15853) and ``_Sample.Name`` ("Ost4 reconstituted in POPC and
#: POPE", bmr52090).
SAMPLE_TEXT_FIELDS = ("details", "solvent_system", "name")


def sample_level_classes(sample) -> list[MimeticClass]:
    """The sample-level fallback: the class ``_Sample.Type`` names (micelle,
    bicelle, liposome, membrane, nanodisc ...; never a reverse micelle or a
    liquid crystal) and those of the explicit mimetic tokens of the sample's
    free text, ``SAMPLE_TEXT_FIELDS``. Each counts as a mimetic, whatever its
    concentration -- the sample says it *is* one."""
    found = []
    sample_type = str(getattr(sample, "type", "") or "").strip().lower()
    if sample_type and not _NOT_A_MIMETIC_SAMPLE_TYPE.search(sample_type):
        found += classify_name(sample_type)
    for field in SAMPLE_TEXT_FIELDS:
        found += explicit_classes(getattr(sample, field, None))
    return found


def alignment_sample_ids(experiments) -> set[str]:
    """IDs of the samples every experiment of which is anisotropic, beside
    another sample that is not: RDC alignment samples (bmr30410's bicelle
    samples 3 and 4, used only by ``1H-15N IPAP HSQC`` next to the isotropic
    sample 1 the shifts were assigned in). When every sample is anisotropic
    (bmr34738 marks all its experiments so), none is set apart: the shifts
    were measured in them. ``experiments`` are ``ExperimentList.experiments``
    tuples (``Sample_ID`` at index 3, ``Sample_state`` at index 5)."""
    states: dict[str, set[str]] = {}
    for exp in experiments or ():
        sample_id = str(exp[3] or "").strip()
        if sample_id:
            states.setdefault(sample_id, set()).add(str(exp[5] or "").strip().lower())
    aligned = {sid for sid, st in states.items() if st == {"anisotropic"}}
    return aligned if aligned != set(states) else set()


def classify_samples(
    samples, entities=None, alignment_samples=frozenset()
) -> tuple[list[str], bool]:
    """``(labels, is_membrane_mimetic)`` over a chain's samples.

    ``labels`` names every recognised class once, in ``MIMETIC_CLASSES`` order,
    whether or not it counts (a sub-CMC CHAPS is still named); the bool is the
    membrane-mimetic rule. ``entities`` is the entry's ``{entity ID: entity}``
    (see :func:`classify_sample`). The sample-level fallback
    (:func:`sample_level_classes`) is read for every sample whose ID is not in
    ``alignment_samples`` (:func:`alignment_sample_ids`).
    """
    seen, is_mimetic = set(), False
    for sample in samples:
        for cls, counts in classify_sample(sample, entities):
            seen.add(cls.label)
            is_mimetic |= counts
        if str(getattr(sample, "id", None) or "").strip() in alignment_samples:
            continue
        for cls in sample_level_classes(sample):
            seen.add(cls.label)
            is_mimetic = True
    labels = [label for label in MIMETIC_CLASSES_BY_LABEL if label in seen]
    return labels, is_mimetic


# --------------------------------------------------------------------------- #
# reviewed title-only entries
# --------------------------------------------------------------------------- #

#: The committed table of entries whose only membrane-mimetic evidence is the
#: entry title, each reviewed against its raw NMR-STAR file (release audit C8).
MEMBRANE_OVERRIDES_TSV = (
    Path(__file__).resolve().parents[1] / "dataset" / "data" / "membrane_overrides.tsv"
)
OVERRIDE_COLUMNS = ("id", "class", "reason")
#: An override key: a BMRB entry ID (every chain of the entry) or a chain ID
#: ``<entry>_<shift table>_<entity assembly>_<entity>`` (that chain only, for an
#: entry whose other shift tables were measured outside the mimetic).
_OVERRIDE_ID = re.compile(r"^\d+(?:_\d+_\d+_\d+)?$")


@dataclass(frozen=True)
class MembraneOverride:
    """A reviewed entry or chain (``id``, see ``_OVERRIDE_ID``): a
    membrane-mimetic sample of class ``label`` (a ``MIMETIC_CLASSES`` label),
    for ``reason``."""

    id: str
    label: str
    reason: str


@functools.cache
def _load_overrides(path: Path) -> dict[str, MembraneOverride]:
    overrides: dict[str, MembraneOverride] = {}
    header = None
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if header is None:
            if tuple(fields) != OVERRIDE_COLUMNS:
                raise ValueError(
                    f"{path}:{lineno}: header must be {OVERRIDE_COLUMNS}, got {fields}"
                )
            header = fields
            continue
        if len(fields) != len(OVERRIDE_COLUMNS):
            raise ValueError(f"{path}:{lineno}: expected 3 fields, got {fields}")
        key, label, reason = (f.strip() for f in fields)
        if not _OVERRIDE_ID.match(key):
            raise ValueError(f"{path}:{lineno}: {key!r} is no entry or chain ID")
        if label not in MIMETIC_CLASSES_BY_LABEL:
            raise ValueError(f"{path}:{lineno}: unknown class {label!r}")
        if not reason:
            raise ValueError(f"{path}:{lineno}: {key} has no reason")
        if key in overrides:
            raise ValueError(f"{path}:{lineno}: duplicate id {key}")
        overrides[key] = MembraneOverride(key, label, reason)
    if header is None:
        raise ValueError(f"{path}: no header {OVERRIDE_COLUMNS}")
    return overrides


def load_overrides(path=None) -> dict[str, MembraneOverride]:
    """``{entry or chain ID: MembraneOverride}`` from the committed table (or
    ``path``). Raises ``ValueError`` on a malformed header or ID, an unknown
    class, a missing reason or a duplicated ID."""
    return _load_overrides(Path(MEMBRANE_OVERRIDES_TSV if path is None else path))


def classify_chain(entry, samples, chain_id=None) -> tuple[list[str], bool]:
    """``(labels, is_membrane_mimetic)`` of a chain of ``entry`` whose shift
    table references ``samples``: :func:`classify_samples` with the entry's
    entities and alignment samples, then the reviewed override of the chain
    (``chain_id``) or of its entry, if any (:func:`load_overrides`)."""
    experiments = getattr(getattr(entry, "experiment_list", None), "experiments", ())
    labels, is_mimetic = classify_samples(
        samples,
        entities=getattr(entry, "entities", None),
        alignment_samples=alignment_sample_ids(experiments),
    )
    overrides = load_overrides()
    override = overrides.get(str(chain_id)) or overrides.get(
        str(getattr(entry, "id", "")).strip()
    )
    if override is not None:
        seen = set(labels) | {override.label}
        labels = [label for label in MIMETIC_CLASSES_BY_LABEL if label in seen]
        is_mimetic = True
    return labels, is_mimetic
