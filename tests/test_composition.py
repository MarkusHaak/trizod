"""Molecular-composition classifier regressions (``trizod.dataset.composition``).

Covers the four defects audited in ``docs/plans/2026-08-09-kuppel-review-response.md``
§3.3/§3.4:

* ``has_metal`` was dead code — no ``_Entity.Type`` ever starts with ``metal``
  (histogram over 21,842 entity records: polymer 18,897 / non-polymer 2,926 /
  water 16 / null 1 / D-SACCHARIDE 1 / SACCHARIDE 1).
* ``has_nucleic`` missed ``polydeoxyribonucleotide/polyribonucleotide hybrid``.
* a ``water`` entity in the assembly made protein-only depositions look bound.
* homo-oligomers were invisible, and the naive "one entity on N assembly rows"
  rule mislabels conformer depositions (bmr15711 apoSOD1, bmr19104, bmr51492).

Real entries are used wherever a real entry exists; every assertion below was
checked against the raw ``bmr<id>_3.str`` tags first.
"""

import pickle

import pytest

from tests.conftest import BMRB_DIR, requires_bmrb_data
from trizod.bmrb.bmrb import EntityAssemblyRow
from trizod.dataset import composition as comp


def load_entry(entry_id):
    """Parse a raw BMRB entry, skipping if that entry is not on disk."""
    import trizod.bmrb.bmrb as bmrb

    entry_dir = BMRB_DIR / f"bmr{entry_id}"
    if not entry_dir.exists():
        pytest.skip(f"BMRB {entry_id} not available in BMRB_DIR")
    return bmrb.BmrbEntry(str(entry_id), entry_dir)


# --------------------------------------------------------------------------- #
# _Chem_comp.Formula parsing — the primary metal rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "formula,elements",
    [
        ("Zn", {"Zn"}),
        ("Zn 2", {"Zn"}),  # trailing charge token, bmr-observed for ZN
        ("'Zn 2'", {"Zn"}),  # raw NMR-STAR quoting
        ("CA1", {"Ca"}),  # all-caps + count, comp CA_2+
        ("DY", {"Dy"}),  # all-caps two-letter symbol
        ("TM", {"Tm"}),
        ("C N", {"C", "N"}),  # cyanide — NOT cobalt+nitrogen
        ("C O", {"C", "O"}),  # carbon monoxide — NOT cobalt
        ("N3", {"N"}),  # azide — NOT nobelium
        ("Cl", {"Cl"}),
        ("H2 O", {"H", "O"}),
        ("Fe2 S2", {"Fe", "S"}),
        ("C34 H32 Fe N4 O4", {"C", "H", "Fe", "N", "O"}),  # HEM
        ("C10 H16 N5 O13 P3", {"C", "H", "N", "O", "P"}),  # ATP
        ("C42 H70 O35", {"C", "H", "O"}),  # BCD
    ],
)
def test_parse_formula_elements(formula, elements):
    assert comp.parse_formula_elements(formula) == frozenset(elements)


@pytest.mark.parametrize("formula", [None, "", ".", "-", "X", "C16PN?"])
def test_parse_formula_elements_unresolvable(formula):
    assert comp.parse_formula_elements(formula) is None


@pytest.mark.parametrize("formula", ["Zn", "Zn 2", "CA1", "Mg", "Fe", "K", "Na", "DY"])
def test_bare_metal_formula_is_a_metal_ion(formula):
    assert comp.classify_metal(formula) == comp.METAL_ION


@pytest.mark.parametrize(
    "formula",
    ["C34 H32 Fe N4 O4", "Fe2 S2", "Fe4 S4", "Al F4", "Be F3", "O4 Re", "Mo O4"],
)
def test_metal_bearing_compound_is_a_cofactor_not_an_ion(formula):
    """HEM/HEC/FES/ALF contain a metal but are not bare metal ions.

    Documented split: ``has_metal_ion`` means "the ligand IS a metal", and
    ``has_metal_cofactor`` means "the ligand CONTAINS a metal". ``has_metal``
    is the union, so HEM counts as metal-bearing but not as a metal ion.
    """
    assert comp.classify_metal(formula) == comp.METAL_COFACTOR


@pytest.mark.parametrize("formula", ["C N", "C O", "N3", "Cl", "H2 O", "C42 H70 O35"])
def test_non_metal_formula_is_not_a_metal(formula):
    assert comp.classify_metal(formula) is None


def test_nucleic_polymer_types_cover_every_corpus_value():
    """Enumerated over bmrb_meta.jsonl (16,963 entries, 21,842 entity records).

    Observed ``_Entity.Polymer_type`` values: polypeptide(L) 17,282 · None 2,949 ·
    polydeoxyribonucleotide 845 · polyribonucleotide 672 · polypeptide(D) 27 ·
    polysaccharide(D) 25 · cyclic-pseudo-peptide 17 ·
    polydeoxyribonucleotide/polyribonucleotide hybrid 16 · other 9.
    """
    assert len(comp.NUCLEIC_POLYMER_TYPES) == 3
    for nucleic in (
        "polydeoxyribonucleotide",
        "polyribonucleotide",
        "polydeoxyribonucleotide/polyribonucleotide hybrid",
    ):
        assert nucleic in comp.NUCLEIC_POLYMER_TYPES
    for non_nucleic in (
        "polypeptide(L)",
        "polypeptide(D)",
        "polysaccharide(D)",
        "cyclic-pseudo-peptide",
        "other",
        "",
    ):
        assert non_nucleic not in comp.NUCLEIC_POLYMER_TYPES


# --------------------------------------------------------------------------- #
# copy-count derivation — unit level
# --------------------------------------------------------------------------- #


def _row(rid, name="", state="native", isomer="no", meq="", details=""):
    return EntityAssemblyRow(
        id=str(rid),
        name=name,
        entity_id="1",
        entity_label="$e",
        physical_state=state,
        conformational_isomer=isomer,
        magnetic_equivalence_group=meq,
        details=details,
    )


def test_single_row_is_one_copy():
    assert comp.derive_copy_count([_row(1)]) == 1


def test_repeated_rows_in_one_state_are_copies():
    assert comp.derive_copy_count([_row(1, "chain A"), _row(2, "chain B")]) == 2


def test_magnetic_equivalence_group_wins_over_row_count():
    rows = [_row(i, meq=m) for i, m in enumerate(["1", "1", "2", "2"], start=1)]
    assert comp.derive_copy_count(rows) == 2


def test_partial_magnetic_equivalence_group_is_ignored():
    """Rule (a) needs the code on EVERY row; a half-filled column proves nothing."""
    rows = [_row(1, "chain A", meq="1"), _row(2, "chain B", meq="")]
    assert comp.derive_copy_count(rows) == 2


def test_rows_spanning_two_physical_states_are_not_copies():
    rows = [_row(1, state="native"), _row(2, state="unfolded")]
    assert comp.derive_copy_count(rows) == 1


def test_conformer_named_rows_are_not_copies():
    rows = [_row(1, "monomer, cis conformer"), _row(2, "monomer, trans conformer")]
    assert comp.derive_copy_count(rows) == 1


def test_conformer_token_in_details_demotes():
    rows = [_row(1, "chain A", details="major form"), _row(2, "chain B")]
    assert comp.derive_copy_count(rows) == 1


# --------------------------------------------------------------------------- #
# oligomer metadata — real entries
# --------------------------------------------------------------------------- #


@requires_bmrb_data
def test_apo_sod1_conformers_are_not_oligomeric():
    """bmr15711 — "Backbone chemical shift assignements for monomeric apoSOD1".

    Two ``_Entity_assembly`` rows, both ``native``, named "chain 1, proline cis
    conformer" / "chain 2, proline trans conformer"; ``Conformational_isomer``
    is ``no`` on both, so the name tokens are the only usable signal. This chain
    is in the published ``test_chezod117`` split — a naive rule labels it a dimer.
    """
    out = comp.detect_bound(load_entry(15711))
    assert out["n_entity_assembly_rows"] == 2
    assert out["n_copies"] == 1
    assert out["is_bound"] is False


@requires_bmrb_data
def test_two_native_one_unfolded_rows_are_not_oligomeric():
    """bmr19104 — 3 rows, states native/unfolded/native (published train row)."""
    out = comp.detect_bound(load_entry(19104))
    assert out["n_entity_assembly_rows"] == 3
    assert out["n_copies"] == 1
    assert out["has_conformational_isomer"] is True


@requires_bmrb_data
def test_named_forms_are_not_oligomeric():
    """bmr51492 — "TDP-43, form A" / "form B", both ``partially disordered``."""
    out = comp.detect_bound(load_entry(51492))
    assert out["n_entity_assembly_rows"] == 2
    assert out["n_copies"] == 1


@requires_bmrb_data
def test_genuine_homodimer_reports_two_copies():
    """bmr6851 — "Solution Structure of the human homodimeric DNA repair protein
    XPF": rows "XPF dimer protein unit A"/"unit B", both native, no meq code."""
    out = comp.detect_bound(load_entry(6851))
    assert out["n_entities"] == 1  # unchanged meaning: distinct Entity records
    assert out["n_entity_assembly_rows"] == 2
    assert out["n_copies"] == 2


@requires_bmrb_data
def test_conformational_isomer_tag_alone_does_not_demote():
    """bmr53193 — asymmetric PF4 tetramer: 4 native rows, all with
    ``Conformational_isomer = yes``, named "PF4 tetramer, chain A".."chain D".
    The tag is set on genuine oligomers too, so it must not demote on its own."""
    out = comp.detect_bound(load_entry(53193))
    assert out["n_entity_assembly_rows"] == 4
    assert out["n_copies"] == 4
    assert out["has_conformational_isomer"] is True


@requires_bmrb_data
def test_magnetic_equivalence_group_beats_row_count_on_il10():
    """bmr19377 — IL-10 dimer deposited as 3 rows with meq codes 1/1/2; the third
    row is a second signal of the N-terminus, not a third protomer. Published
    ``test_trizod`` row, so the naive count of 3 is visible in the release."""
    out = comp.detect_bound(load_entry(19377))
    assert out["n_entity_assembly_rows"] == 3
    assert out["n_copies"] == 2


@requires_bmrb_data
def test_groes_heptamer_reports_seven_copies():
    """bmr7091 — 7 GroES subunits, meq code 1 on all of them."""
    out = comp.detect_bound(load_entry(7091))
    assert out["n_entity_assembly_rows"] == 7
    assert out["n_copies"] == 7


# --------------------------------------------------------------------------- #
# nucleic acids, metals, ligands, water — real entries
# --------------------------------------------------------------------------- #


@requires_bmrb_data
def test_hybrid_polymer_type_counts_as_nucleic():
    """bmr17351 — ``polydeoxyribonucleotide/polyribonucleotide hybrid``.

    16 entity records in 14 entries carry the hybrid type; 8 flip
    ``has_nucleic`` and 4 flip ``is_bound`` (17351, 19226, 30184, 34228).
    """
    out = comp.detect_bound(load_entry(17351))
    assert out["has_nucleic"] is True
    assert out["is_bound"] is True


@requires_bmrb_data
def test_metal_ion_resolved_from_chem_comp_formula():
    """bmr6191 — SecA + a ZN non-polymer entity; ``_Chem_comp.Formula`` is ``Zn``.

    ``_Entity.Type`` is ``non-polymer``, never ``metal``, which is why the old
    ``startswith("metal")`` rule was False on all 16,963 entries.
    """
    out = comp.detect_bound(load_entry(6191))
    assert out["has_metal"] is True
    assert out["has_metal_ion"] is True
    assert out["has_metal_cofactor"] is False
    assert out["metal_comp_ids"] == "ZN"
    assert out["ligand_comp_ids"] == "ZN"


@requires_bmrb_data
def test_heme_is_a_metal_cofactor_not_a_metal_ion():
    """bmr16307 — GlbN + HEM, ``_Chem_comp.Formula = 'C34 H32 Fe N4 O4'``."""
    out = comp.detect_bound(load_entry(16307))
    assert out["has_metal"] is True
    assert out["has_metal_ion"] is False
    assert out["has_metal_cofactor"] is True
    assert out["metal_comp_ids"] == "HEM"


@requires_bmrb_data
def test_non_metal_ligands_are_listed_but_not_metal():
    """bmr17210 — DnaK NBD + ATP + AGS + MG. Only MG is a metal."""
    out = comp.detect_bound(load_entry(17210))
    assert out["has_metal"] is True
    assert out["metal_comp_ids"] == "MG"
    assert out["ligand_comp_ids"] == "AGS;ATP;MG"
    assert "ATP" in out["ligand_names"]


@requires_bmrb_data
@pytest.mark.parametrize("entry_id", [25640, 34125, 34240])
def test_water_only_extra_entity_is_not_a_bound_complex(entry_id):
    """bmr25640 (cytochrome c, 105 aa), bmr34125 (cytotoxin-1, 60 aa),
    bmr34240 (engrailed homeodomain, 64 aa): protein-only depositions that were
    discarded from every tier solely because a WATER entity is listed."""
    out = comp.detect_bound(load_entry(entry_id))
    assert out["has_water"] is True
    assert out["multi_protein_assembly"] is False
    assert out["is_bound"] is False
    # the WATER record still shows up in the raw, uninterpreted row count
    assert out["n_entity_assembly_rows"] == 2


@requires_bmrb_data
@pytest.mark.parametrize("entry_id,comp_id", [(6123, "CTO"), (7114, "BCD")])
def test_saccharide_typed_ligands_are_labelled(entry_id, comp_id):
    """bmr6123 (D-SACCHARIDE CTO) / bmr7114 (SACCHARIDE BCD) match neither
    ``polymer`` nor ``non-polymer``. They are still ``is_bound`` via the
    assembly branch, but were invisible to every ligand label."""
    out = comp.detect_bound(load_entry(entry_id))
    assert out["has_non_polymer"] is False
    assert out["has_other_ligand"] is True
    assert out["ligand_comp_ids"] == comp_id
    assert out["is_bound"] is True


@requires_bmrb_data
def test_untyped_entity_with_a_comp_label_is_a_ligand():
    """bmr16669 — RhoA + GTPgS, ``_Entity.Type = '.'`` and the only ligand
    evidence is ``_Entity.Nonpolymer_comp_label = $chem_comp_GTPgS``."""
    out = comp.detect_bound(load_entry(16669))
    assert out["has_other_ligand"] is True
    assert out["ligand_comp_ids"] == "GTPgS"


@requires_bmrb_data
def test_metal_never_makes_an_entry_bound_on_its_own():
    """Labelling-only claim: reviving ``has_metal`` must not move any entry into
    ``is_bound`` that was not already there via ``has_non_polymer``."""
    for entry_id in (6191, 16307, 17210):
        out = comp.detect_bound(load_entry(entry_id))
        assert out["has_metal"] is True
        assert out["has_non_polymer"] is True


# --------------------------------------------------------------------------- #
# PR0 — the build must not swallow classifier errors
# --------------------------------------------------------------------------- #


def test_build_surfaces_classifier_errors(tmp_path):
    """A stale pickle (an object without the attributes the classifier reads)
    used to be caught by a bare ``except Exception`` that wrote
    ``{"is_bound": True, "error": ...}`` and let the build complete silently
    with NaN composition metadata on thousands of entries."""
    from trizod.dataset import build

    with (tmp_path / "12345.pkl").open("wb") as fh:
        pickle.dump({"not": "a BmrbEntry"}, fh)

    with pytest.raises(RuntimeError) as excinfo:
        build.build_composition_cache(tmp_path)
    assert "12345" in str(excinfo.value)


def test_build_composition_cache_accepts_a_valid_entry(tmp_path):
    from trizod.dataset import build

    entry = _StubEntry()
    with (tmp_path / "99999.pkl").open("wb") as fh:
        pickle.dump(entry, fh)

    cache = build.build_composition_cache(tmp_path)
    assert cache["99999"]["is_bound"] is False
    assert cache["99999"]["n_copies"] == 1


class _StubEntity:
    def __init__(self):
        self.id = "1"
        self.name = "stub"
        self.type = "polymer"
        self.polymer_type = "polypeptide(L)"
        self.nonpolymer_comp_id = ""
        self.nonpolymer_comp_label = ""


class _StubAssembly:
    def __init__(self):
        self.id = "1"
        self.entities = [("1", "1", "$stub", "native")]
        self.entity_assembly_rows = [_row(1, "stub")]


class _StubEntry:
    def __init__(self):
        self.entities = {"1": _StubEntity()}
        self.assemblies = {"1": _StubAssembly()}
        self.chem_comps = {}


def test_exclude_homo_oligomers_flag_is_opt_in():
    """Decision D3: oligomeric state is annotated everywhere, never filtered by
    default. The opt-in flag must exist and must default to off."""
    from trizod.dataset import build

    parser = build.make_parser()
    assert parser.parse_args([]).exclude_homo_oligomers is False
    assert parser.parse_args(["--exclude-homo-oligomers"]).exclude_homo_oligomers
