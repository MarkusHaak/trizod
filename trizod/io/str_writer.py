"""Emit re-referenced backbone-shifts-only NMR-STAR (.str) files."""

from pathlib import Path

import pynmrstar

from trizod.constants import AA1TO3, BACKBONE_ATOMS, REFINED_WEIGHTS
from trizod.offsets import total_offset_ppm

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
    lacs_offsets_ppm,
    potenci_residual_offsets_sigma,
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
        lacs_offsets_ppm: dict atom -> ppm, i.e. the `lacs_off_<atom>_ppm`
            columns. LACS offsets are subtracted from the raw shift array, so
            they are genuine ppm.
        potenci_residual_offsets_sigma: dict atom -> sigma units, i.e. the
            `off_<atom>_sigma` columns. `scoring.compute_offsets()` averages
            `diff_arr / REFINED_WEIGHTS`, so these are multiples of the
            per-atom POTENCI RMSD, NOT ppm.
        rereference_mode: which mode produced the shifts; copied to metadata.
        pipeline_version: free-form string copied to metadata.

    Three per-atom offset loops are written, and every tag names its unit:
    `LACS_offsets` (ppm), `POTENCI_residual_offsets` (sigma plus its ppm
    equivalent), and `Total_offsets` -- the ready-to-use ppm quantity from
    `offsets.total_offset_ppm()`, which is what a reader subtracts from a
    deposited shift to reproduce the shift TriZOD scored.
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

    # `Offset_ppm` is the right tag name here and only here: LACS offsets are
    # subtracted from the raw ppm shift array, so the number really is ppm.
    lacs_loop = pynmrstar.Loop.from_scratch("LACS_offsets")
    lacs_loop.set_category("LACS_offsets")
    lacs_loop.add_tag(["Atom_ID", "Offset_ppm"])
    for atom in BACKBONE_ATOMS:
        lacs_loop.add_data([atom, f"{lacs_offsets_ppm.get(atom, 0.0):.6f}"])
    aux.add_loop(lacs_loop)

    # The POTENCI/AIC offsets come in as sigma units (see docstring). Feeding
    # them to a bare `Offset_ppm` tag -- as this loop once did -- is the unit
    # error the whole column rename exists to prevent, so the sigma value gets
    # a tag that says sigma and the ppm equivalent is written alongside.
    potenci_loop = pynmrstar.Loop.from_scratch("POTENCI_residual_offsets")
    potenci_loop.set_category("POTENCI_residual_offsets")
    potenci_loop.add_tag(["Atom_ID", "Offset_sigma", "Offset_ppm"])
    for atom in BACKBONE_ATOMS:
        offset_sigma = potenci_residual_offsets_sigma.get(atom, 0.0)
        offset_ppm = offset_sigma * REFINED_WEIGHTS[atom]
        potenci_loop.add_data([atom, f"{offset_sigma:.6f}", f"{offset_ppm:.6f}"])
    aux.add_loop(potenci_loop)

    # The one loop a reader needs: total ppm to subtract from a deposited shift.
    total_loop = pynmrstar.Loop.from_scratch("Total_offsets")
    total_loop.set_category("Total_offsets")
    total_loop.add_tag(["Atom_ID", "Offset_ppm"])
    for atom in BACKBONE_ATOMS:
        total_ppm = total_offset_ppm(
            atom,
            lacs_offsets_ppm.get(atom, 0.0),
            potenci_residual_offsets_sigma.get(atom, 0.0),
        )
        total_loop.add_data([atom, f"{total_ppm:.6f}"])
    aux.add_loop(total_loop)

    entry.add_saveframe(aux)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    entry.write_to_file(str(out_path))
