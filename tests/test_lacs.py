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


# ordN.m preceding-residue correction ``Ncorr`` table, exactly as published in
# the BMRB LACS source (github.com/bmrb-io/LACS/ordN.m), AA order
# ACDEFGHIKLMNPQRSTVWY.  Columns are the RAW (N, HN) values; ordN.m then applies
# ``Ncorr_N -= 1.486`` and ``Ncorr_HN -= 0.005`` before use.
_ORDN_RAW = {
    "A": (0.0, 0.00), "C": (3.5, 0.17), "D": (1.6, 0.04), "E": (2.0, 0.10),
    "F": (3.2, 0.04), "G": (0.8, -0.04), "H": (2.6, 0.13), "I": (5.0, 0.13),
    "K": (2.4, 0.08), "L": (1.8, 0.02), "M": (1.9, 0.06), "N": (1.5, 0.04),
    "P": (1.2, 0.16), "Q": (2.1, 0.10), "R": (2.2, 0.10), "S": (2.7, 0.08),
    "T": (3.2, 0.09), "V": (4.7, 0.14), "W": (3.6, -0.08), "Y": (3.6, 0.01),
}  # fmt: skip
_ORDN_NCORR_N = {aa: raw_n - 1.486 for aa, (raw_n, _) in _ORDN_RAW.items()}
_ORDN_NCORR_HN = {aa: raw_hn - 0.005 for aa, (_, raw_hn) in _ORDN_RAW.items()}


def _inject_ncorr_effect(shifts, seq, table_n, table_hn):
    """Add a preceding-residue (i-1) effect to N/HN, simulating the real effect
    that ordN.m's Ncorr correction is meant to remove."""
    out = {atom: arr.copy() for atom, arr in shifts.items()}
    for i in range(1, len(seq)):
        prev = seq[i - 1]
        if not np.isnan(out["N"][i]):
            out["N"][i] += table_n.get(prev, 0.0)
        if not np.isnan(out["H"][i]):
            out["H"][i] += table_hn.get(prev, 0.0)
    return out


class TestNCorrProvenance:
    """Issue #17: the _NCORR preceding-residue table must match its cited
    source, BMRB ordN.m (raw Ncorr minus the documented 1.486 / 0.005)."""

    def test_ncorr_matches_ordn_source(self):
        from trizod.lacs import lacs as lacs_mod

        for aa in _ORDN_RAW:
            got_hn, got_n = lacs_mod._NCORR[aa]
            assert got_n == pytest.approx(_ORDN_NCORR_N[aa], abs=1e-6), (
                f"{aa} N: {got_n} != ordN.m {_ORDN_NCORR_N[aa]}"
            )
            assert got_hn == pytest.approx(_ORDN_NCORR_HN[aa], abs=1e-6), (
                f"{aa} HN: {got_hn} != ordN.m {_ORDN_NCORR_HN[aa]}"
            )


class TestNOffsetCompositionIndependence:
    """The recovered 15N offset must not depend on amino-acid composition once
    the (ordN.m) preceding-residue effect has been correctly subtracted.  A
    wrong Ncorr table leaves a composition-dependent residual (issue #17)."""

    def _n_offset(self, seq, rng_seed):
        seq_nums = np.arange(1, len(seq) + 1)
        base = _generate_shifts(
            seq, secondary_noise_std=1.0, rng=np.random.default_rng(rng_seed)
        )
        shifts = _inject_ncorr_effect(base, seq, _ORDN_NCORR_N, _ORDN_NCORR_HN)
        return compute_lacs_offsets(seq, seq_nums, shifts)["N"]

    def test_n_offset_independent_of_preceding_composition(self):
        # Two correctly-referenced proteins carrying the true ordN.m preceding
        # effect but with opposite-extreme preceding residues (Ala vs Ile).
        off_ala = self._n_offset("A" * 50, rng_seed=1)
        off_ile = self._n_offset("I" * 50, rng_seed=2)

        assert off_ala is not None and off_ile is not None
        # With the faithful table the effect cancels and both recover the same
        # offset; the wrong table gives a >4 ppm composition-dependent split.
        assert abs(off_ala - off_ile) < 0.5, (
            f"N offset depends on composition: Ala-rich={off_ala}, "
            f"Ile-rich={off_ile} (Δ={off_ala - off_ile:.2f})"
        )
