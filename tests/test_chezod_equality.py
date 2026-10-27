"""Regression test: TriZOD reproduces published CheZOD Z-scores.

TriZOD's `potenci-only` scoring (POTENCI random-coil + AIC offset correction +
CheZOD-style Z-score, LACS excluded) is a reimplementation of the CheZOD method.
This test locks that in: for a curated subset of the CheZOD1325 set whose TriZOD
entity sequence exactly matches CheZOD's, it recomputes the per-residue Z-scores
from the BMRB data and asserts agreement with the published values within
atol=0.1 on every comparable residue — the tolerance at which the implementation
was historically validated (commit a9df3ac, 2023).

The reference subset lives in tests/reference/chezod_zscores_subset.json (built
by scripts/validation/build_chezod_test_subset.py from
data/external/chezod/protein_nmr_1325/allscores1325newest.txt). The test needs the BMRB
entry files (data/raw/bmrb_entries/) like the other pipeline tests.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import BMRB_DIR, requires_bmrb_data

REFERENCE = Path(__file__).resolve().parent / "reference" / "chezod_zscores_subset.json"
NA = 999.0  # CheZOD sentinel for terminal / no-data residues
ATOL = 0.1
MIN_COMPARABLE = 10

_REF = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else {}


def _potenci_only_triplet_zscores(entry, shifts, condID, seq):
    """Per-residue triplet Z-scores for one peptide-shift table, LACS excluded."""
    import trizod.potenci.potenci as potenci
    import trizod.scoring.scoring as scoring

    conds = entry.conditions[condID]
    temperature = conds.get_temperature()
    pH = conds.get_pH()
    ion = conds.get_ionic_strength()
    use_ph_corr = pH != 7.0
    predshiftdct = potenci.get_pred_shifts(seq, temperature, pH, ion, use_ph_corr)
    ret = scoring.get_offset_corrected_shifts(
        seq, shifts, predshiftdct, rereference_mode="potenci-only"
    )
    if ret is None:
        return None
    abs_weighted_diffs_final, cmp_mask = ret[1], ret[2]
    return scoring.compute_zscores(
        *scoring.convert_to_triplet_data(abs_weighted_diffs_final, cmp_mask), cmp_mask
    )


@requires_bmrb_data
@pytest.mark.skipif(not _REF, reason="CheZOD reference subset not available")
@pytest.mark.parametrize("bmrb_id", sorted(_REF))
def test_reproduces_chezod_zscores(bmrb_id):
    import trizod.bmrb.bmrb as bmrb

    ref = _REF[bmrb_id]
    ref_seq, ref_z = ref["seq"], ref["zscores"]
    entry = bmrb.BmrbEntry(bmrb_id, BMRB_DIR / f"bmr{bmrb_id}")

    # CheZOD accepts an entry if ANY peptide-shift table reproduces it; mirror that.
    best_max_diff = None
    for (_stID, _entity_assemID, entityID), (
        shifts,
        condID,
        _assemID,
        _sampleIDs,
    ) in entry.get_peptide_shifts().items():
        seq = entry.entities[entityID].seq
        if seq != ref_seq:
            continue  # only compare the entity/sequence CheZOD scored
        zscores = _potenci_only_triplet_zscores(entry, shifts, condID, seq)
        if zscores is None:
            continue
        diffs = [
            abs(zscores[i] - cv)
            for i, cv in enumerate(ref_z)
            if cv != NA and i < len(zscores) and not np.isnan(zscores[i])
        ]
        if len(diffs) < MIN_COMPARABLE:
            continue
        max_diff = max(diffs)
        if best_max_diff is None or max_diff < best_max_diff:
            best_max_diff = max_diff

    assert best_max_diff is not None, (
        f"bmr{bmrb_id}: no peptide-shift table matched the CheZOD sequence "
        f"with >= {MIN_COMPARABLE} comparable residues"
    )
    assert best_max_diff <= ATOL, (
        f"bmr{bmrb_id}: max per-residue |TriZOD - CheZOD| = {best_max_diff:.4f} "
        f"exceeds atol={ATOL}"
    )
