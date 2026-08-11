"""Molecular-composition classifier for BMRB entries (bound-complex detection).

Extracted from ``build_final_dataset`` so both the disorder dataset build (which
DROPS bound complexes) and the future binding-site dataset (which will KEEP and
mine them) share one classifier.

``detect_bound()`` returns two kinds of field:

* **membership** — ``is_bound`` and the four flags it is built from. A row that
  is ``is_bound`` is dropped from every tier of the dataset.
* **labels** — metal / ligand / oligomer metadata. These are emitted for every
  entry and filter nothing by default (decision D3: oligomeric state is an
  orthogonal biological axis, not an NMR-data-quality axis).
"""

import re
from collections import Counter

# --------------------------------------------------------------------------- #
# vocabularies
# --------------------------------------------------------------------------- #

# Every ``_Entity.Polymer_type`` value that is a nucleic acid. The hybrid was
# missed by the previous two-value exact match: 16 entity records in 14 entries
# carry it, 8 of which flip ``has_nucleic`` and 4 ``is_bound`` (17351, 19226,
# 30184, 34228). Non-nucleic values in the corpus: polypeptide(L)/(D),
# polysaccharide(D), cyclic-pseudo-peptide, other.
NUCLEIC_POLYMER_TYPES = frozenset(
    {
        "polydeoxyribonucleotide",
        "polyribonucleotide",
        "polydeoxyribonucleotide/polyribonucleotide hybrid",
    }
)

# ``_Entity.Type`` values, lower-cased. The full corpus histogram over 21,842
# entity records is polymer 18,897 / non-polymer 2,926 / water 16 / null 1 /
# D-SACCHARIDE 1 / SACCHARIDE 1 -- there is no ``metal`` type, which is why the
# old ``startswith("metal")`` rule was dead code.
POLYMER_TYPE = "polymer"
NON_POLYMER_TYPE = "non-polymer"
WATER_TYPE = "water"

METAL_ION = "ion"
METAL_COFACTOR = "cofactor"

_ELEMENTS = frozenset(
    """
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni
    Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
    Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au
    Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr
    """.split()  # noqa: SIM905
)
# Non-metals and metalloids. Everything else in the table counts as a metal.
_NON_METALS = frozenset(
    "H He B C N O F Ne Si P S Cl Ar As Se Br Kr Te I Xe At Rn".split()  # noqa: SIM905
)
METAL_ELEMENTS = _ELEMENTS - _NON_METALS

# Tie-break only: comp IDs whose ``_Chem_comp.Formula`` is missing or unusable.
# Both lists were derived from the formulas that DO resolve in the corpus, so
# they can never disagree with the primary rule. Bare-numeric comp IDs are
# deliberately excluded -- ``3`` is MAGNESIUM ION in bmr17610 but a meaningless
# key anywhere else, and its formula resolves anyway.
METAL_ION_COMP_IDS = frozenset(
    """
    AG CA CA_2+ CD CE CO CS CU CU1 DY FE FE2 GA HG K LA LU MG MN NA NI RH3 SM
    SR TB TM YB YT3 ZN
    """.split()  # noqa: SIM905
)
METAL_COFACTOR_COMP_IDS = frozenset(
    """
    2FJ 3UQ 7BU 9F0 ALF AMPPCP AUZ BCL BEF BF2 CGO DHE F3S FES GIX H9C HEA HEB
    HEC HEC_ox HEM HEM_ox HEM_red LN8 MGF MOO PHF REO RUL Re1O1 SF4 VO4 ZEM ZNH
    [2Fe-2S]
    """.split()  # noqa: SIM905
)
# Last resort: 193 non-polymer entity records carry no comp ID at all and are
# identifiable only by their free-text ``_Entity.Name``. Keys are lower-cased.
METAL_ENTITY_NAMES = {
    "zinc ion": (METAL_ION, "ZN"),
    "zinc (ii) ion": (METAL_ION, "ZN"),
    "zn": (METAL_ION, "ZN"),
    "calcium ion": (METAL_ION, "CA"),
    "ca": (METAL_ION, "CA"),
    "ca_2+": (METAL_ION, "CA"),
    "magnesium ion": (METAL_ION, "MG"),
    "magesium ion": (METAL_ION, "MG"),  # depositor typo, bmr-observed
    "mg": (METAL_ION, "MG"),
    "potassium ion": (METAL_ION, "K"),
    "sodium ion": (METAL_ION, "NA"),
    "protoporphyrin ix containing fe": (METAL_COFACTOR, "HEM"),
    "heme": (METAL_COFACTOR, "HEM"),
    "heme b/c": (METAL_COFACTOR, "HEM"),
    "hem": (METAL_COFACTOR, "HEM"),
    "perrhenate": (METAL_COFACTOR, "REO"),
}

# Words that mark repeated ``_Entity_assembly`` rows as alternative conformers
# of ONE molecule rather than copies of it. Deliberately narrow and
# word-anchored: "chain A"/"unit B"/"subunit 3" must not match, or genuine
# oligomers (bmr6851 XPF dimer, bmr53193 PF4 tetramer) would be demoted.
_CONFORMER_RE = re.compile(
    r"\b(cis|trans|major|minor|conformer|conformers|conformation|conformational"
    r"|isomer|isomers|rotamer|form|forms|folded|unfolded)\b",
    re.IGNORECASE,
)

_FORMULA_STD = re.compile(r"([A-Z][a-z]?)(\d*)")
_FORMULA_TOKEN = re.compile(r"^([A-Za-z]{1,2})(\d*)$")
_FORMULA_TOKEN_PREFIXED = re.compile(r"^(\d+)([A-Za-z]{1,2})$")
_FORMULA_CHARGE = re.compile(r"^\d*[+-]?$")
# NMR-STAR null spellings. "na" is deliberately absent: it is a real
# _Entity_assembly.Physical_state value (148 records) *and* the sodium formula.
_NULL_VALUES = frozenset({"", ".", "-", "?"})


# --------------------------------------------------------------------------- #
# _Chem_comp.Formula parsing
# --------------------------------------------------------------------------- #


def parse_formula_elements(formula):
    """Return the set of element symbols in ``formula``, or None if unparseable.

    Standard casing is tried first so ``C O`` stays carbon + oxygen rather than
    becoming cobalt, and ``N3`` stays azide rather than becoming nobelium. Only
    when a token fails that pass is it re-read as an all-caps symbol, which is
    what recovers depositor spellings like ``CA1``, ``DY`` and ``TM``.
    """
    if not formula:
        return None
    text = formula.strip().strip("'\"").strip()
    if not text or text.lower() in _NULL_VALUES:
        return None
    tokens = [t for t in text.split() if not _FORMULA_CHARGE.match(t)]
    if not tokens:
        return None

    elements = set()
    for token in tokens:
        symbols = _parse_formula_token(token)
        if symbols is None:
            return None
        elements.update(symbols)
    return frozenset(elements) or None


def _parse_formula_token(token):
    pos, symbols = 0, []
    while pos < len(token):
        match = _FORMULA_STD.match(token, pos)
        if match is None or match.start() != pos or match.group(1) not in _ELEMENTS:
            symbols = None
            break
        symbols.append(match.group(1))
        pos = match.end()
    if symbols and pos == len(token):
        return symbols
    match = _FORMULA_TOKEN.match(token) or _FORMULA_TOKEN_PREFIXED.match(token)
    if match:
        symbol = next(g for g in match.groups() if g and not g.isdigit())
        if symbol.capitalize() in _ELEMENTS:
            return [symbol.capitalize()]
    return None


def classify_metal(formula):
    """``METAL_ION`` / ``METAL_COFACTOR`` / None, from a formula alone.

    A formula that is a single metal element (optionally with a charge) is a
    bare metal ion: ``Zn``, ``Zn 2``, ``CA1``. A formula that merely *contains*
    a metal is a cofactor: HEM (``C34 H32 Fe N4 O4``), FES (``Fe2 S2``), ALF
    (``Al F4``). ``has_metal`` is the union of the two.
    """
    elements = parse_formula_elements(formula)
    if not elements:
        return None
    if not elements & METAL_ELEMENTS:
        return None
    return METAL_ION if len(elements) == 1 else METAL_COFACTOR


# --------------------------------------------------------------------------- #
# ligand identity
# --------------------------------------------------------------------------- #


def _clean(value):
    """Normalise a raw NMR-STAR scalar to a plain string ("" when unset)."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in _NULL_VALUES else text


def comp_id_from_label(label):
    """``$chem_comp_GTPgS`` -> ``GTPgS``; the fallback when Nonpolymer_comp_ID
    is unset (bmr6123 CTO, bmr7114 BCD, bmr16669 GTPgS)."""
    text = _clean(label).lstrip("$")
    prefix = "chem_comp_"
    if text.lower().startswith(prefix):
        text = text[len(prefix) :]
    return text


def entity_comp_id(entity):
    return _clean(getattr(entity, "nonpolymer_comp_id", "")) or comp_id_from_label(
        getattr(entity, "nonpolymer_comp_label", "")
    )


def classify_entity_metal(entity, chem_comps):
    """Metal status of one entity: (kind, comp_id) or (None, comp_id).

    Priority: ``_Chem_comp.Formula`` (primary, resolves 93 % of non-polymer
    entities) -> comp-ID allow-list (tie-break for a missing/unusable formula)
    -> ``_Entity.Name`` (the 193 records with no comp ID at all).
    """
    comp_id = entity_comp_id(entity)
    comp = chem_comps.get(comp_id) if comp_id else None
    kind = classify_metal(getattr(comp, "formula", None)) if comp is not None else None
    if kind is not None:
        return kind, comp_id
    if comp_id:
        if comp_id in METAL_ION_COMP_IDS:
            return METAL_ION, comp_id
        if comp_id in METAL_COFACTOR_COMP_IDS:
            return METAL_COFACTOR, comp_id
        return None, comp_id
    named = METAL_ENTITY_NAMES.get(_clean(getattr(entity, "name", "")).lower())
    if named is not None:
        return named
    return None, comp_id


# --------------------------------------------------------------------------- #
# oligomeric state
# --------------------------------------------------------------------------- #


def derive_copy_count(rows):
    """Conservative number of copies of one entity inside one assembly.

    Priority, per plan §3.3:

    (a) ``Magnetic_equivalence_group_code`` populated on every row -> the size
        of the largest group (bmr19377 IL-10 3 rows -> 2; bmr27605 4 -> 2;
        bmr7091 GroES 7 -> 7).
    (b) otherwise, rows spanning more than one ``Physical_state`` describe
        alternative states, not copies (bmr19104, native/unfolded/native -> 1).
    (c) otherwise, a conformer word in a row name or Details demotes to 1
        (bmr15711 apoSOD1 cis/trans, bmr51492 "TDP-43, form A"/"form B").
    (d) otherwise, the raw row count.

    ``Conformational_isomer`` is parsed and reported but is deliberately NOT a
    demotion trigger on its own: bmr53193 (asymmetric PF4 tetramer) and bmr7091
    (GroES heptamer) both set it while being genuine oligomers.

    The result is a lower bound in both directions -- under-annotated homodimers
    deposited as a single row (bmr15021) read as 1.
    """
    if len(rows) <= 1:
        return len(rows)

    groups = [_clean(r.magnetic_equivalence_group) for r in rows]
    if all(groups):
        return max(Counter(groups).values())

    if len({_clean(r.physical_state).lower() for r in rows}) > 1:
        return 1

    for row in rows:
        if _CONFORMER_RE.search(f"{_clean(row.name)} {_clean(row.details)}"):
            return 1
    return len(rows)


# --------------------------------------------------------------------------- #
# the classifier
# --------------------------------------------------------------------------- #


def detect_bound(entry) -> dict:
    """Classify a BmrbEntry's molecular composition.

    ``is_bound`` = any of: a non-polymer entity (ligand/drug), a nucleic-acid
    entity (DNA/RNA), or an assembly containing more than one distinct
    non-water Entity. Homo-oligomers reference one Entity multiple times and
    count as single-molecule; see ``n_copies`` for that axis.

    Metal status is a LABEL, never a membership criterion: every metal-bearing
    entity is a ``non-polymer`` entity, so ``has_metal`` implies
    ``has_non_polymer`` and reviving it moves nothing in or out of the dataset.
    """
    entities = list(entry.entities.values())
    entity_types = {e.id: _clean(e.type).lower() for e in entities}

    has_non_polymer = any(t == NON_POLYMER_TYPE for t in entity_types.values())
    has_water = any(t == WATER_TYPE for t in entity_types.values())
    # No ``type == polymer`` guard: an absent Type tag must not hide a nucleic
    # acid whose Polymer_type says exactly what it is.
    has_nucleic = any(_clean(e.polymer_type) in NUCLEIC_POLYMER_TYPES for e in entities)

    metal_ids, cofactor_ids, ligand_ids, ligand_names = [], [], [], []
    has_other_ligand = False
    for e in entities:
        etype = entity_types[e.id]
        if etype in (POLYMER_TYPE, WATER_TYPE):
            continue
        if etype != NON_POLYMER_TYPE:
            # SACCHARIDE / D-SACCHARIDE / unset Type: a ligand only when the
            # deposition declares one (bmr6123 CTO, bmr7114 BCD, bmr16669
            # GTPgS). Labelled, not folded into ``is_bound`` -- all three are
            # already bound via the assembly branch.
            if not entity_comp_id(e):
                continue
            has_other_ligand = True
        kind, comp_id = classify_entity_metal(e, entry.chem_comps)
        if kind == METAL_ION:
            metal_ids.append(comp_id or _clean(e.name))
        elif kind == METAL_COFACTOR:
            cofactor_ids.append(comp_id or _clean(e.name))
        if comp_id:
            ligand_ids.append(comp_id)
        name = _clean(e.name)
        if name:
            ligand_names.append(" ".join(name.split()))

    # Assembly composition. ``water`` is excluded from the member set: three
    # protein-only depositions (bmr25640, bmr34125, bmr34240) were dropped from
    # every tier solely because a WATER entity is listed alongside the protein.
    # ``n_entity_assembly_rows`` stays raw (no exclusions, no interpretation);
    # ``n_copies`` counts polymer entities only, so a nanodisc deposition
    # reports the protein's stoichiometry and not its 250 lipid records.
    water_ids = {eid for eid, t in entity_types.items() if t == WATER_TYPE}
    polymer_ids = {eid for eid, t in entity_types.items() if t == POLYMER_TYPE}
    multi_protein_assembly = False
    n_entity_assembly_rows = 0
    n_copies = 1
    for asm in entry.assemblies.values():
        n_entity_assembly_rows += len(asm.entity_assembly_rows)
        member_ids = {
            _clean(r.entity_id)
            for r in asm.entity_assembly_rows
            if _clean(r.entity_id) and _clean(r.entity_id) not in water_ids
        }
        if len(member_ids) > 1:
            multi_protein_assembly = True
        by_entity = {}
        for row in asm.entity_assembly_rows:
            eid = _clean(row.entity_id)
            if eid in polymer_ids:
                by_entity.setdefault(eid, []).append(row)
        for entity_rows in by_entity.values():
            n_copies = max(n_copies, derive_copy_count(entity_rows))

    has_conformational_isomer = any(
        _clean(r.conformational_isomer).lower() == "yes"
        for asm in entry.assemblies.values()
        for r in asm.entity_assembly_rows
    )

    # n_entities: number of distinct entities in the entry (homo-oligomers
    # show up as a single Entity that is referenced multiple times in an
    # assembly).  Unchanged meaning -- nothing published shifts under it.
    n_entities = len(entities)
    entity_ids = [e.id for e in entities]
    assert len(set(entity_ids)) == n_entities, "non-unique entity IDs"

    bound = has_non_polymer or has_nucleic or multi_protein_assembly
    return {
        "n_entities": n_entities,
        "has_non_polymer": has_non_polymer,
        "has_nucleic": has_nucleic,
        "has_metal": bool(metal_ids or cofactor_ids),
        "has_metal_ion": bool(metal_ids),
        "has_metal_cofactor": bool(cofactor_ids),
        "has_water": has_water,
        "has_other_ligand": has_other_ligand,
        "metal_comp_ids": ";".join(sorted(set(metal_ids + cofactor_ids))),
        "ligand_comp_ids": ";".join(sorted(set(ligand_ids))),
        "ligand_names": ";".join(sorted(set(ligand_names))),
        "multi_protein_assembly": multi_protein_assembly,
        "n_entity_assembly_rows": n_entity_assembly_rows,
        "n_copies": n_copies,
        "has_conformational_isomer": has_conformational_isomer,
        "is_bound": bound,
    }
