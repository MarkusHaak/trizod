"""Molecular-composition classifier for BMRB entries (bound-complex detection).

Extracted from ``build_final_dataset`` so both the disorder dataset build (which
DROPS bound complexes) and the future binding-site dataset (which will KEEP and
mine them) share one classifier.
"""


def detect_bound(entry) -> dict:
    """Classify a BmrbEntry's molecular composition.

    ``is_bound`` = any of: a non-polymer entity (ligand/drug), a nucleic-acid
    entity (DNA/RNA), a metal entity, or an assembly containing more than one
    distinct Entity (homo-oligomers, which reference one Entity multiple times,
    count as single-molecule).
    """
    entities = list(entry.entities.values())
    has_non_polymer = any(e.type == "non-polymer" for e in entities)
    has_nucleic = any(
        e.type == "polymer"
        and (e.polymer_type or "") in ("polydeoxyribonucleotide", "polyribonucleotide")
        for e in entities
    )
    has_metal = any((e.type or "").lower().startswith("metal") for e in entities)
    # n_entities: number of distinct entities in the entry (homo-oligomers
    # show up as a single Entity that is referenced multiple times in an
    # assembly).
    n_entities = len(entities)
    entity_ids = [e.id for e in entities]
    assert len(set(entity_ids)) == n_entities, "non-unique entity IDs"

    # Check assemblies for multi-entity composition.  Some entries have
    # one Entity record but multiple assemblies/entity_assemblies if it's
    # a homo-oligomer; we treat that as single-molecule.
    multi_protein_assembly = False
    for asm in entry.assemblies.values():
        distinct_entity_ids_in_asm = {e[1] for e in asm.entities if e[1]}
        if len(distinct_entity_ids_in_asm) > 1:
            multi_protein_assembly = True
            break

    bound = has_non_polymer or has_nucleic or has_metal or multi_protein_assembly
    return {
        "n_entities": n_entities,
        "has_non_polymer": has_non_polymer,
        "has_nucleic": has_nucleic,
        "has_metal": has_metal,
        "multi_protein_assembly": multi_protein_assembly,
        "is_bound": bound,
    }
