"""Regression: --rereference-mode 'none' and 'both' produce different LACS
offsets for a known mis-referenced BMRB entry (17665, alpha-synuclein)."""

from pathlib import Path

import pytest

from tests.conftest import BMRB_DIR, requires_bmrb_data

REPO = Path(__file__).resolve().parent.parent


@requires_bmrb_data
def test_lacs_correction_changes_alphasyn_offsets():
    import trizod.bmrb.bmrb as bmrb
    import trizod.potenci.potenci as potenci
    from trizod.scoring.scoring import get_offset_corrected_shifts

    asyn_dir = BMRB_DIR / "bmr17665"
    if not asyn_dir.exists():
        pytest.skip("BMRB 17665 not in BMRB_DIR")

    entry = bmrb.BmrbEntry("17665", asyn_dir)
    peptide_shifts = entry.get_peptide_shifts()
    (st_id, ea_id, e_id), (shifts, cond_id, _, _) = next(iter(peptide_shifts.items()))
    seq = entry.entities[e_id].seq

    cond = entry.conditions[cond_id]
    temp = cond.get_temperature(return_default=True)
    pH = cond.get_pH(return_default=True)
    ion = cond.get_ionic_strength(return_default=True)

    predshiftdct = potenci.get_pred_shifts(seq, temp, pH, ion, pH != 7.0)

    ret_none = get_offset_corrected_shifts(
        seq, shifts, predshiftdct, rereference_mode="none"
    )
    ret_both = get_offset_corrected_shifts(
        seq, shifts, predshiftdct, rereference_mode="both"
    )

    assert ret_none is not None and ret_both is not None
    lacs_none = ret_none[-1]
    lacs_both = ret_both[-1]

    # mode='none' should have no LACS offset (all zeros).
    assert all(v == 0.0 for v in lacs_none.values()), (
        f"expected zero LACS offsets in 'none' mode, got {lacs_none}"
    )
    # mode='both' should detect a non-trivial CA/CB/CO offset for 17665
    # (Reid quoted ~2.9 ppm via TALOS-N).
    big_offsets = {
        k: v for k, v in lacs_both.items() if k in ("C", "CA", "CB") and abs(v) > 1.0
    }
    assert big_offsets, (
        f"expected at least one large LACS offset on C/CA/CB for BMRB 17665; got {lacs_both}"
    )
