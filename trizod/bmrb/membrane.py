"""Membrane-mimetic samples: which detergents and lipids count, and when.

A chain's sample is a membrane mimetic when any non-polymer
``_Sample_component`` of the samples its shift table references (every sample
of the entry if it references none) meets one of three conditions:

(a) **bilayer** -- a bilayer-forming phospholipid or bilayer assembly, at any
    concentration: diacyl or ether PC, PG, PE, PS or PA with chains >= C10,
    cardiolipin, lipid extracts, bicelles, nanodiscs/MSP, liposomes/vesicles;
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
glycochenodeoxycholate), acyl-CoA, lysozyme, lipid II, sarcosine itself. A
mixture deposited as one component name ("DMPC/DHPC", "DMPC+CHAPS") is
classified part by part; each part is assigned the first class of
``MIMETIC_CLASSES`` whose pattern it matches.

Water CMCs are upper bounds for ionic detergents (they fall in salt), and the
CHAPSO CMC is taken equal to CHAPS's. Recall depends on a name regex that
future depositions will outgrow; ``MIMETIC_CLASSES`` is the one place to
extend it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

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
        rf"{_NOT_AFTER}sds{_NOT_BEFORE}|(?:dodecyl|lauryl)[\s-]*sul[fp]",
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
        rf"nano-?disc|{_NOT_AFTER}msp(?:\d|{_NOT_BEFORE})|membrane scaffold protein",
    ),
    MimeticClass(
        "liposome",
        BILAYER,
        None,
        None,
        rf"liposom|vesicle|{_NOT_AFTER}[sl]uvs?{_NOT_BEFORE}",
    ),
    MimeticClass("amphipol", BILAYER, None, None, r"amphipol"),
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
#: (glycochenodeoxycholate alone is 34 component rows), acyl-CoA, lysozyme.
NOT_A_MIMETIC = re.compile(
    r"cholic|cholan|cholate|coenzyme|(?<![a-z])coa(?![a-z])|lysozyme"
)

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


def _non_polymer_components(sample):
    """``(name, value, unit)`` of the components that are not the studied
    polymer (``_Sample_component.Entity_ID`` unset)."""
    for comp in sample.components:
        if comp[3] and not comp[2]:
            yield comp[3], comp[4], comp[5]


#: Separators of a mixture written as one component name: "DMPC/DHPC" (a
#: bicelle), "DMPC+CHAPS". Not ':' (``14:0 PC``, ``16:0 Lyso PG``) and not ','
#: (``1,2-dimyristoyl-...``).
_MIXTURE_SEPARATOR = re.compile(r"\s*[/+;]\s*")


def classify_name(name) -> list[MimeticClass]:
    """The classes of a component name, one per part of a mixture name, in
    the order of the parts (a part may yield none)."""
    parts = _MIXTURE_SEPARATOR.split(str(name or ""))
    return [cls for cls in map(classify_component, parts) if cls is not None]


def classify_sample(sample) -> list[tuple[MimeticClass, bool]]:
    """``(class, counts)`` for every recognised mimetic component of a sample
    (every part of a mixture name). A mixture's parts share its concentration,
    which only matters for an additive-only mixture."""
    found, names = [], []
    for name, value, unit in _non_polymer_components(sample):
        names.append(str(name).lower())
        for cls in classify_name(name):
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


def classify_samples(samples) -> tuple[list[str], bool]:
    """``(labels, is_membrane_mimetic)`` over a chain's samples.

    ``labels`` names every recognised class once, in ``MIMETIC_CLASSES`` order,
    whether or not it counts (a sub-CMC CHAPS is still named); the bool is the
    membrane-mimetic rule.
    """
    seen, is_mimetic = set(), False
    for sample in samples:
        for cls, counts in classify_sample(sample):
            seen.add(cls.label)
            is_mimetic |= counts
    labels = [label for label in MIMETIC_CLASSES_BY_LABEL if label in seen]
    return labels, is_mimetic
