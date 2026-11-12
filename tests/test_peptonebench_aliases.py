"""Pre-2026-04 names that PeptoneBench imports (NMR-STAR audit I-16).

PeptoneBench (PeptoneLtd/peptonebench, ``src/peptonebench/trizod_gscores.py``)
builds PeptoneDB-CS on TriZOD and imports::

    from trizod.constants import AA1TO3, BBATNS
    from trizod.potenci.potenci import getpredshifts
    from trizod.scoring.scoring import (
        compute_pscores, convert_to_triplet_data, get_offset_corrected_wSCS)

(``nmrcs.py`` imports ``getpredshifts`` alone.) Develop renamed four of them --
``BBATNS`` -> ``BACKBONE_ATOMS``, ``getpredshifts`` -> ``get_pred_shifts``,
``compute_pscores`` -> ``compute_gscores``, ``get_offset_corrected_wSCS`` ->
``get_offset_corrected_shifts`` -- so they break the day develop reaches main.
Three are plain aliases. ``get_offset_corrected_wSCS`` cannot be: the new
function runs LACS by default and returns a tenth value (the LACS offsets), and
PeptoneBench unpacks nine. It is a wrapper that keeps the old contract.
"""

import numpy as np
import pytest

SEQ = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQ"


def test_peptonebench_import_lines_resolve():
    from trizod.constants import AA1TO3, BBATNS  # noqa: F401
    from trizod.potenci.potenci import getpredshifts  # noqa: F401
    from trizod.scoring.scoring import (  # noqa: F401
        compute_pscores,
        convert_to_triplet_data,
        get_offset_corrected_wSCS,
    )


def test_bbatns_is_backbone_atoms():
    from trizod.constants import BACKBONE_ATOMS, BBATNS

    assert BBATNS is BACKBONE_ATOMS


def test_getpredshifts_is_get_pred_shifts():
    from trizod.potenci.potenci import get_pred_shifts, getpredshifts

    assert getpredshifts is get_pred_shifts


def test_compute_pscores_is_compute_gscores():
    from trizod.scoring.scoring import compute_gscores, compute_pscores

    assert compute_pscores is compute_gscores


# --------------------------------------------------------------------------- #
# get_offset_corrected_wSCS keeps the old nine-value, POTENCI-only contract
# --------------------------------------------------------------------------- #


def _peptonebench_shifts(seq, predshifts, seed=0):
    """Shift tuples built the way PeptoneBench builds them: random-coil shifts
    plus noise, with a uniform +1 ppm CA miscalibration for the offset step."""
    from trizod.constants import AA1TO3

    rng = np.random.default_rng(seed)
    shifts = []
    for (res, aa), atoms in sorted(predshifts.items()):
        for atom, val in atoms.items():
            if atom not in {"C", "CA", "CB", "N", "H"}:
                continue
            val = val + rng.normal(0.0, 0.1) + (1.0 if atom == "CA" else 0.0)
            shifts.append(
                ("None", "None", res, AA1TO3[aa], atom, atom[0], val, 0.0, "")
            )
    return shifts


@pytest.fixture(scope="module")
def inputs():
    from trizod.potenci.potenci import getpredshifts

    # PeptoneBench's exact keywords, including the obsolete `pkacsvfile`
    predshifts = getpredshifts(
        seq=SEQ, temperature=298.0, pH=7.0, ion=0.1, pkacsvfile=False
    )
    return predshifts, _peptonebench_shifts(SEQ, predshifts)


def test_wscs_wrapper_returns_nine_values(inputs):
    from trizod.scoring.scoring import get_offset_corrected_wSCS

    predshifts, shifts = inputs
    ret = get_offset_corrected_wSCS(seq=SEQ, shifts=shifts, predshiftdct=predshifts)
    assert len(ret) == 9


def test_wscs_wrapper_is_the_potenci_only_mode_of_the_new_function(inputs):
    # The old function had no LACS step: rereference_mode="potenci-only" is it.
    from trizod.scoring.scoring import (
        get_offset_corrected_shifts,
        get_offset_corrected_wSCS,
    )

    predshifts, shifts = inputs
    old = get_offset_corrected_wSCS(seq=SEQ, shifts=shifts, predshiftdct=predshifts)
    new = get_offset_corrected_shifts(
        SEQ, shifts, predshifts, rereference_mode="potenci-only"
    )
    assert len(new) == 10
    for got, want in zip(old, new[:9]):
        if isinstance(want, dict):
            assert got == want
        else:
            np.testing.assert_array_equal(got, want)
    # ... and it did correct the +1 ppm CA miscalibration (offsets are in units
    # of the per-atom weight, so back to ppm)
    from trizod.constants import REFINED_WEIGHTS

    assert old[4]["CA"] * REFINED_WEIGHTS["CA"] == pytest.approx(1.0, abs=0.2)


def test_wscs_wrapper_can_opt_into_the_current_default(inputs):
    from trizod.scoring.scoring import (
        get_offset_corrected_shifts,
        get_offset_corrected_wSCS,
    )

    predshifts, shifts = inputs
    ret = get_offset_corrected_wSCS(
        seq=SEQ, shifts=shifts, predshiftdct=predshifts, rereference_mode="both"
    )
    new = get_offset_corrected_shifts(SEQ, shifts, predshifts)
    assert len(ret) == 9
    assert ret[4] == new[4]


def test_wscs_wrapper_passes_failure_through():
    from trizod.scoring.scoring import get_offset_corrected_wSCS

    # no comparable shifts -> None, as before (not a TypeError on None[:9])
    assert get_offset_corrected_wSCS(seq=SEQ, shifts=[], predshiftdct={}) is None


def test_peptonebench_gscore_recipe_runs(inputs):
    """PeptoneBench's compute_gscores, step for step, on the old names."""
    from trizod.constants import BBATNS
    from trizod.scoring.scoring import (
        compute_pscores,
        convert_to_triplet_data,
        get_offset_corrected_wSCS,
    )

    predshifts, shifts = inputs
    ret = get_offset_corrected_wSCS(seq=SEQ, shifts=shifts, predshiftdct=predshifts)
    shw, ashwi, cmp_mask, olf, offf, shw0, ashwi0, ol0, off0 = ret
    assert set(offf) == set(BBATNS)
    for i, at in enumerate(BBATNS):
        assert np.isfinite(offf[at]), (i, at)
    ashwi3, k3 = convert_to_triplet_data(ashwi, cmp_mask)
    scores = compute_pscores(ashwi3, k3, cmp_mask)
    assert scores.shape == (len(SEQ),)
    # random-coil shifts read as disordered
    assert np.nanmean(scores) > 0.5
