"""Tests for the POTENCI random coil chemical shift prediction module."""

import pytest

import trizod.potenci.potenci as potenci

# Reference values generated on main branch (commit 8905a85)
# Sequence: AAGDTFKISELVK, T=298K, pH=7.0, ion=0.1M, no pH correction
REFERENCE_SEQUENCE = "AAGDTFKISELVK"
REFERENCE_TEMPERATURE = 298.0
REFERENCE_PH = 7.0
REFERENCE_ION = 0.1

# Subset of reference values: (residue_num, aa, atom_type, value)
REFERENCE_VALUES = [
    (2, "A", "CA", 52.705700),
    (2, "A", "N", 123.529426),
    (3, "G", "CA", 45.270900),
    (5, "T", "CB", 69.767200),
    (7, "K", "H", 8.115990),
    (10, "E", "C", 176.293150),
    (12, "V", "HA", 4.098429),
]


class TestPredictions:
    """Test POTENCI predictions against known reference values."""

    def test_predictions_match_reference(self):
        result = potenci.get_pred_shifts(
            REFERENCE_SEQUENCE,
            REFERENCE_TEMPERATURE,
            REFERENCE_PH,
            REFERENCE_ION,
            use_ph_corr=False,
            pka_csv_path=False,
        )

        for res_num, aa, atom_type, expected in REFERENCE_VALUES:
            actual = result[(res_num, aa)][atom_type]
            assert actual == pytest.approx(expected, abs=1e-6), (
                f"Mismatch for residue {res_num}{aa} {atom_type}: "
                f"expected {expected}, got {actual}"
            )

    def test_predictions_with_ph_correction(self):
        """POTENCI with pH correction should produce different results from pH=7."""
        result_no_ph = potenci.get_pred_shifts(
            REFERENCE_SEQUENCE,
            REFERENCE_TEMPERATURE,
            REFERENCE_PH,
            REFERENCE_ION,
            use_ph_corr=False,
            pka_csv_path=False,
        )
        result_with_ph = potenci.get_pred_shifts(
            REFERENCE_SEQUENCE,
            REFERENCE_TEMPERATURE,
            5.5,  # different pH
            REFERENCE_ION,
            use_ph_corr=True,
            pka_csv_path=False,
        )
        # D (Asp) is pH-sensitive; its shifts should differ at pH 5.5 vs 7.0
        d_ca_no_ph = result_no_ph[(4, "D")]["CA"]
        d_ca_with_ph = result_with_ph[(4, "D")]["CA"]
        assert d_ca_no_ph != pytest.approx(d_ca_with_ph, abs=0.01)


class TestEdgeCases:
    """Test POTENCI edge cases and structural constraints."""

    def test_terminal_residues_excluded(self):
        """POTENCI should not predict shifts for terminal residues."""
        result = potenci.get_pred_shifts(
            REFERENCE_SEQUENCE,
            REFERENCE_TEMPERATURE,
            REFERENCE_PH,
            REFERENCE_ION,
            use_ph_corr=False,
            pka_csv_path=False,
        )
        # First residue (1, 'A') and last residue (13, 'K') should be absent
        assert (1, "A") not in result
        residue_nums = [k[0] for k in result]
        assert max(residue_nums) < len(REFERENCE_SEQUENCE)

    def test_glycine_has_no_cb(self):
        """Glycine should not have CB or HB predictions."""
        result = potenci.get_pred_shifts(
            REFERENCE_SEQUENCE,
            REFERENCE_TEMPERATURE,
            REFERENCE_PH,
            REFERENCE_ION,
            use_ph_corr=False,
            pka_csv_path=False,
        )
        gly_shifts = result[(3, "G")]
        assert "CB" not in gly_shifts
        assert "HB" not in gly_shifts
