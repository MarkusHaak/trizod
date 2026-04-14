"""Tests for the LACS (Linear Analysis of Chemical Shifts) module.

Validates offset recovery using POTENCI-generated ground truth shifts
corrupted with known referencing offsets (Reid Alderson's benchmark idea).
"""

import numpy as np
import pytest

from trizod.lacs import compute_lacs_offsets

# A realistic 80-residue sequence (ubiquitin-like)
_SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
_SEQ_NUMS = np.arange(1, len(_SEQ) + 1)

# Random coil shifts from LACS reference table (subset for building test data)
_RC = {
    "A": (52.5, 19.1, 4.32, 177.8, 8.24, 123.8),
    "C": (58.2, 28.0, 4.55, 174.6, 8.32, 118.8),
    "D": (54.2, 41.1, 4.64, 176.3, 8.34, 120.4),
    "E": (56.6, 29.9, 4.35, 176.6, 8.42, 120.2),
    "F": (57.7, 39.6, 4.62, 175.8, 8.30, 120.3),
    "G": (45.1, 0.0, 3.96, 174.9, 8.33, 108.8),
    "H": (55.0, 29.0, 4.73, 174.1, 8.42, 118.2),
    "I": (61.1, 38.8, 4.17, 176.4, 8.00, 119.9),
    "K": (56.2, 33.1, 4.32, 176.6, 8.29, 120.4),
    "L": (55.1, 42.4, 4.34, 177.6, 8.16, 121.8),
    "M": (55.4, 32.9, 4.48, 176.3, 8.28, 119.6),
    "N": (53.1, 38.9, 4.74, 175.2, 8.40, 118.7),
    "P": (63.3, 32.1, 4.42, 177.3, 0.00, 0.0),
    "Q": (55.7, 29.4, 4.34, 176.0, 8.32, 119.8),
    "R": (56.0, 30.9, 4.34, 176.3, 8.23, 120.5),
    "S": (58.3, 63.8, 4.47, 174.6, 8.31, 115.7),
    "T": (61.8, 69.8, 4.35, 174.7, 8.15, 113.6),
    "V": (62.2, 32.9, 4.12, 176.3, 8.03, 119.2),
    "W": (57.5, 29.6, 4.66, 176.1, 8.25, 121.3),
    "Y": (57.9, 38.8, 4.55, 175.9, 8.12, 120.3),
}


def _generate_shifts(seq, secondary_noise_std=0.0, rng=None):
    """Generate synthetic shifts from random coil + optional Gaussian noise.

    Returns dict of arrays keyed by atom type.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(seq)
    ca = np.full(n, np.nan)
    cb = np.full(n, np.nan)
    ha = np.full(n, np.nan)
    co = np.full(n, np.nan)
    h = np.full(n, np.nan)
    n_arr = np.full(n, np.nan)

    for i, aa in enumerate(seq):
        if aa not in _RC:
            continue
        rc = _RC[aa]
        ca[i] = (
            rc[0] + rng.normal(0, secondary_noise_std) if secondary_noise_std else rc[0]
        )
        if aa != "G":  # Gly has no CB
            cb[i] = (
                rc[1] + rng.normal(0, secondary_noise_std)
                if secondary_noise_std
                else rc[1]
            )
        ha[i] = (
            rc[2] + rng.normal(0, secondary_noise_std * 0.1)
            if secondary_noise_std
            else rc[2]
        )
        co[i] = (
            rc[3] + rng.normal(0, secondary_noise_std) if secondary_noise_std else rc[3]
        )
        if aa != "P":  # Pro has no HN
            h[i] = (
                rc[4] + rng.normal(0, secondary_noise_std * 0.05)
                if secondary_noise_std
                else rc[4]
            )
            n_arr[i] = (
                rc[5] + rng.normal(0, secondary_noise_std)
                if secondary_noise_std
                else rc[5]
            )

    return {"CA": ca, "CB": cb, "HA": ha, "C": co, "H": h, "N": n_arr}


def _corrupt_shifts(shifts, offsets):
    """Add known offsets to shifts (simulating referencing error)."""
    corrupted = {}
    for atom, arr in shifts.items():
        if atom in offsets:
            corrupted[atom] = arr + offsets[atom]
        else:
            corrupted[atom] = arr.copy()
    return corrupted


class TestLacsBasic:
    """Basic sanity tests."""

    def test_correctly_referenced_returns_near_zero(self):
        """Correctly referenced shifts with realistic spread → offsets near zero."""
        # LACS needs secondary shift variance to work; pure random coil has
        # zero spread on the X-axis (CA-CB secondary).  Use small noise to
        # simulate a typical protein's secondary shift distribution.
        shifts = _generate_shifts(
            _SEQ, secondary_noise_std=1.5, rng=np.random.default_rng(42)
        )
        offsets = compute_lacs_offsets(_SEQ, _SEQ_NUMS, shifts)

        for atom in ["CA", "CB", "C", "HA"]:
            assert offsets[atom] is not None, f"Expected offset for {atom}"
            assert abs(offsets[atom]) < 1.0, (
                f"{atom} offset {offsets[atom]} too large for correct reference"
            )

    def test_too_few_residues_returns_none(self):
        """Short sequence should return None for all offsets."""
        short_seq = "ACDEF"
        short_nums = np.arange(1, 6)
        shifts = _generate_shifts(short_seq)
        offsets = compute_lacs_offsets(short_seq, short_nums, shifts)

        for atom in ["CA", "CB", "C", "HA", "H", "N"]:
            assert offsets[atom] is None

    def test_missing_ca_cb_returns_none(self):
        """If CA or CB missing, all offsets should be None."""
        shifts = _generate_shifts(_SEQ)
        del shifts["CB"]
        offsets = compute_lacs_offsets(_SEQ, _SEQ_NUMS, shifts)

        for atom in offsets:
            assert offsets[atom] is None


class TestLacsSyntheticBenchmark:
    """Reid Alderson's synthetic benchmark: corrupt and recover offsets."""

    @pytest.mark.parametrize(
        "corrupt_offsets",
        [
            {"CA": 2.0, "CB": 2.0, "C": 2.0, "HA": 0.0},
            {"CA": -1.8, "CB": -1.8, "C": 0.6, "HA": -0.1},
            {"CA": 4.5, "CB": 4.5, "C": 1.5, "HA": 0.3},
        ],
        ids=["moderate-13C", "mixed-offsets", "large-13C"],
    )
    def test_13c_offset_recovery_clean(self, corrupt_offsets):
        """Recover known 13C offsets from shifts with mild secondary spread."""
        shifts = _generate_shifts(
            _SEQ, secondary_noise_std=1.0, rng=np.random.default_rng(42)
        )
        corrupted = _corrupt_shifts(shifts, corrupt_offsets)
        recovered = compute_lacs_offsets(_SEQ, _SEQ_NUMS, corrupted)

        for atom in ["CA", "CB", "C", "HA"]:
            if atom in corrupt_offsets:
                expected = corrupt_offsets[atom]
                assert recovered[atom] is not None, f"No offset recovered for {atom}"
                assert abs(recovered[atom] - expected) < 0.5, (
                    f"{atom}: expected ~{expected}, got {recovered[atom]}"
                )

    @pytest.mark.parametrize(
        "corrupt_offsets",
        [
            {"CA": 2.0, "CB": 2.0, "C": 2.0, "HA": 0.0},
            {"CA": -1.8, "CB": -1.8, "C": 0.6, "HA": -0.1},
        ],
        ids=["moderate-13C", "mixed-offsets"],
    )
    def test_13c_offset_recovery_noisy(self, corrupt_offsets):
        """Recover offsets with Gaussian noise simulating secondary structure."""
        shifts = _generate_shifts(
            _SEQ, secondary_noise_std=2.0, rng=np.random.default_rng(123)
        )
        corrupted = _corrupt_shifts(shifts, corrupt_offsets)
        recovered = compute_lacs_offsets(_SEQ, _SEQ_NUMS, corrupted)

        for atom in ["CA", "CB"]:
            if atom in corrupt_offsets:
                expected = corrupt_offsets[atom]
                assert recovered[atom] is not None, f"No offset recovered for {atom}"
                # Wider tolerance for noisy data
                assert abs(recovered[atom] - expected) < 1.5, (
                    f"{atom}: expected ~{expected}, got {recovered[atom]}"
                )

    def test_ca_cb_offsets_equal(self):
        """CA and CB offsets should be identical (shared 13C reference)."""
        shifts = _generate_shifts(
            _SEQ, secondary_noise_std=1.0, rng=np.random.default_rng(42)
        )
        corrupted = _corrupt_shifts(shifts, {"CA": 1.5, "CB": 1.5})
        recovered = compute_lacs_offsets(_SEQ, _SEQ_NUMS, corrupted)

        assert recovered["CA"] is not None
        assert recovered["CB"] is not None
        assert abs(recovered["CA"] - recovered["CB"]) < 0.3, (
            f"CA ({recovered['CA']}) and CB ({recovered['CB']}) should be near-equal"
        )

    def test_hb_equals_ha(self):
        """HB offset should be propagated from HA (both are 1H nuclei)."""
        shifts = _generate_shifts(
            _SEQ, secondary_noise_std=1.0, rng=np.random.default_rng(42)
        )
        recovered = compute_lacs_offsets(_SEQ, _SEQ_NUMS, shifts)

        assert recovered["HB"] == recovered["HA"]

    def test_zero_offset_detected(self):
        """No corruption → offset near zero (false positive check)."""
        shifts = _generate_shifts(
            _SEQ, secondary_noise_std=1.5, rng=np.random.default_rng(99)
        )
        recovered = compute_lacs_offsets(_SEQ, _SEQ_NUMS, shifts)

        for atom in ["CA", "CB"]:
            assert recovered[atom] is not None
            assert abs(recovered[atom]) < 1.0, (
                f"{atom} offset {recovered[atom]} too large for zero corruption"
            )
