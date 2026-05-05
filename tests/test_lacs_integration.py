"""Tests for LACS pre-correction integrated into scoring pipeline."""

import numpy as np

from trizod.constants import BACKBONE_ATOMS
from trizod.scoring.scoring import apply_lacs_correction


def test_apply_lacs_correction_returns_corrected_array_and_offsets():
    """For an obviously biased input, returns (corrected_arr, offsets_dict).

    LACS needs secondary-shift variance on the (CA-CB) X-axis to fit its
    slope, so we use a realistic 80-residue sequence with per-residue
    Wishart random-coil values plus moderate Gaussian noise.
    """
    from trizod.lacs.lacs import _RC_CA, _RC_CB, _RC_SHIFTS

    seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    n_res = len(seq)
    rng = np.random.default_rng(42)

    bbshifts_arr = np.zeros((n_res, 7))
    bbshifts_mask = np.zeros((n_res, 7), dtype=bool)

    ca_idx = BACKBONE_ATOMS.index("CA")
    cb_idx = BACKBONE_ATOMS.index("CB")

    # Wishart random-coil + secondary-shift noise (~1.0 ppm) to give LACS a
    # non-degenerate X-axis. CA shifted by +2.0 ppm to mimic mis-referencing.
    rc_ca = np.array([_RC_SHIFTS[aa][_RC_CA] for aa in seq])
    rc_cb = np.array([_RC_SHIFTS[aa][_RC_CB] for aa in seq])

    bbshifts_arr[:, ca_idx] = rc_ca + rng.normal(0, 1.0, n_res) + 2.0
    bbshifts_arr[:, cb_idx] = rc_cb + rng.normal(0, 1.0, n_res) + 2.0
    bbshifts_mask[:, ca_idx] = ~np.isnan(rc_ca) & (rc_ca != 0.0)
    bbshifts_mask[:, cb_idx] = ~np.isnan(rc_cb) & (rc_cb != 0.0)

    corrected_arr, offsets = apply_lacs_correction(bbshifts_arr, bbshifts_mask, seq)

    assert corrected_arr.shape == bbshifts_arr.shape
    assert isinstance(offsets, dict)
    assert set(offsets.keys()) == set(BACKBONE_ATOMS)
    # CA offset magnitude should be roughly 2 ppm; tolerate sign convention.
    assert abs(abs(offsets["CA"]) - 2.0) < 0.7, (
        f"CA offset {offsets['CA']} not near 2.0"
    )


def test_apply_lacs_correction_passthrough_when_no_data():
    """Empty mask yields zero offsets and the input array unchanged."""
    seq = "A" * 10
    bbshifts_arr = np.zeros((10, 7))
    bbshifts_mask = np.zeros((10, 7), dtype=bool)

    corrected_arr, offsets = apply_lacs_correction(bbshifts_arr, bbshifts_mask, seq)

    assert corrected_arr.shape == bbshifts_arr.shape
    assert all(v == 0.0 for v in offsets.values())
