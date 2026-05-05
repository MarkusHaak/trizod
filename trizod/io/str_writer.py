"""Emit re-referenced backbone-shifts-only NMR-STAR (.str) files."""

from pathlib import Path

import pynmrstar

from trizod.constants import AA1TO3, BACKBONE_ATOMS

_AMBIGUITY_NOT_SET = "."


def _atom_type_for(atom_id):
    """Return the NMR-STAR Atom_type prefix for a given backbone atom_id."""
    if atom_id.startswith("H"):
        return "H"
    if atom_id.startswith("N"):
        return "N"
    return "C"


def write_rereferenced_str(
    out_path,
    entry_id,
    seq,
    bbshifts_arr,
    bbshifts_mask,
    lacs_offsets,
    potenci_residual_offsets,
    rereference_mode,
    pipeline_version,
):
    """Write a backbone-shifts-only NMR-STAR file for a re-referenced entry.

    Args:
        out_path: destination file (Path or str). Parent directory is created.
        entry_id: BMRB entry id, used in the saveframe identifier and metadata.
        seq: one-letter amino-acid sequence (length N).
        bbshifts_arr: (N, len(BACKBONE_ATOMS)) corrected shifts (already
            LACS-corrected if rereference_mode applied LACS).
        bbshifts_mask: (N, len(BACKBONE_ATOMS)) boolean mask.
        lacs_offsets: dict atom -> ppm.
        potenci_residual_offsets: dict atom -> ppm.
        rereference_mode: which mode produced the shifts; copied to metadata.
        pipeline_version: free-form string copied to metadata.
    """
    out_path = Path(out_path)
    entry = pynmrstar.Entry.from_scratch(f"bmr{entry_id}_rereferenced")

    # Saveframe 1: chemical shifts
    sf = pynmrstar.Saveframe.from_scratch(
        "assigned_chem_shift_list_1", "assigned_chemical_shifts"
    )
    sf.add_tag("Sf_category", "assigned_chemical_shifts")
    sf.add_tag("Sf_framecode", "assigned_chem_shift_list_1")
    sf.add_tag("ID", "1")

    loop = pynmrstar.Loop.from_scratch("Atom_chem_shift")
    loop.set_category("Atom_chem_shift")
    loop.add_tag(
        [
            "ID",
            "Seq_ID",
            "Comp_ID",
            "Atom_ID",
            "Atom_type",
            "Val",
            "Val_err",
            "Ambiguity_code",
        ]
    )

    row_id = 0
    for i, aa1 in enumerate(seq):
        if aa1 not in AA1TO3:
            continue
        comp_id = AA1TO3[aa1]
        for j, atom in enumerate(BACKBONE_ATOMS):
            if not bbshifts_mask[i, j]:
                continue
            row_id += 1
            loop.add_data(
                [
                    str(row_id),
                    str(i + 1),
                    comp_id,
                    atom,
                    _atom_type_for(atom),
                    f"{bbshifts_arr[i, j]:.4f}",
                    ".",
                    _AMBIGUITY_NOT_SET,
                ]
            )
    sf.add_loop(loop)
    entry.add_saveframe(sf)

    # Saveframe 2: trizod re-referencing metadata + per-atom offsets
    aux = pynmrstar.Saveframe.from_scratch(
        "trizod_rereferencing_info", "trizod_rereferencing"
    )
    aux.add_tag("Sf_category", "trizod_rereferencing")
    aux.add_tag("Sf_framecode", "trizod_rereferencing_info")
    aux.add_tag("Source_BMRB_id", entry_id)
    aux.add_tag("Pipeline_version", pipeline_version)
    aux.add_tag("Re_referencing_mode", rereference_mode)

    lacs_loop = pynmrstar.Loop.from_scratch("LACS_offsets")
    lacs_loop.set_category("LACS_offsets")
    lacs_loop.add_tag(["Atom_ID", "Offset_ppm"])
    for atom in BACKBONE_ATOMS:
        lacs_loop.add_data([atom, f"{lacs_offsets.get(atom, 0.0):.6f}"])
    aux.add_loop(lacs_loop)

    potenci_loop = pynmrstar.Loop.from_scratch("POTENCI_residual_offsets")
    potenci_loop.set_category("POTENCI_residual_offsets")
    potenci_loop.add_tag(["Atom_ID", "Offset_ppm"])
    for atom in BACKBONE_ATOMS:
        potenci_loop.add_data([atom, f"{potenci_residual_offsets.get(atom, 0.0):.6f}"])
    aux.add_loop(potenci_loop)

    entry.add_saveframe(aux)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    entry.write_to_file(str(out_path))
