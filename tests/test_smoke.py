"""Smoke tests to verify basic functionality before any refactoring."""

import subprocess
import sys
import numpy as np
import pytest
from tests.conftest import requires_bmrb_data, BMRB_DIR


class TestCLI:
    def test_help_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "trizod.trizod", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "input-dir" in result.stdout

    def test_trizod_entrypoint(self):
        result = subprocess.run(
            ["trizod", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


class TestPOTENCIPredictions:
    """Test POTENCI random coil predictions against known reference values."""

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

    def test_predictions_match_reference(self):
        import trizod.potenci.potenci as potenci

        result = potenci.getpredshifts(
            self.REFERENCE_SEQUENCE,
            self.REFERENCE_TEMPERATURE,
            self.REFERENCE_PH,
            self.REFERENCE_ION,
            usephcor=False,
            pkacsvfile=False,
        )

        for res_num, aa, atom_type, expected in self.REFERENCE_VALUES:
            actual = result[(res_num, aa)][atom_type]
            assert actual == pytest.approx(expected, abs=1e-6), (
                f"Mismatch for residue {res_num}{aa} {atom_type}: "
                f"expected {expected}, got {actual}"
            )

    def test_predictions_with_ph_correction(self):
        """POTENCI with pH correction should produce different results from pH=7."""
        import trizod.potenci.potenci as potenci

        result_no_ph = potenci.getpredshifts(
            self.REFERENCE_SEQUENCE,
            self.REFERENCE_TEMPERATURE,
            self.REFERENCE_PH,
            self.REFERENCE_ION,
            usephcor=False,
            pkacsvfile=False,
        )
        result_with_ph = potenci.getpredshifts(
            self.REFERENCE_SEQUENCE,
            self.REFERENCE_TEMPERATURE,
            5.5,  # different pH
            self.REFERENCE_ION,
            usephcor=True,
            pkacsvfile=False,
        )
        # D (Asp) is pH-sensitive; its shifts should differ at pH 5.5 vs 7.0
        d_ca_no_ph = result_no_ph[(4, "D")]["CA"]
        d_ca_with_ph = result_with_ph[(4, "D")]["CA"]
        assert d_ca_no_ph != pytest.approx(d_ca_with_ph, abs=0.01)

    def test_terminal_residues_excluded(self):
        """POTENCI should not predict shifts for terminal residues."""
        import trizod.potenci.potenci as potenci

        result = potenci.getpredshifts(
            self.REFERENCE_SEQUENCE,
            self.REFERENCE_TEMPERATURE,
            self.REFERENCE_PH,
            self.REFERENCE_ION,
            usephcor=False,
            pkacsvfile=False,
        )
        # First residue (1, 'A') and last residue (13, 'K') should be absent
        assert (1, "A") not in result
        residue_nums = [k[0] for k in result.keys()]
        assert max(residue_nums) < len(self.REFERENCE_SEQUENCE)

    def test_glycine_has_no_cb(self):
        """Glycine should not have CB or HB predictions."""
        import trizod.potenci.potenci as potenci

        result = potenci.getpredshifts(
            self.REFERENCE_SEQUENCE,
            self.REFERENCE_TEMPERATURE,
            self.REFERENCE_PH,
            self.REFERENCE_ION,
            usephcor=False,
            pkacsvfile=False,
        )
        gly_shifts = result[(3, "G")]
        assert "CB" not in gly_shifts
        assert "HB" not in gly_shifts


@requires_bmrb_data
class TestPipelineSingleEntry:
    """Test pipeline on a single BMRB entry."""

    def test_parse_bmrb_entry(self):
        import trizod.bmrb.bmrb as bmrb

        entry = bmrb.BmrbEntry("4493", BMRB_DIR + "/bmr4493")
        assert entry.id == "4493"
        assert len(entry.entities) > 0
        assert len(entry.shift_tables) > 0

    def test_compute_scores_single_entry(self):
        import trizod.bmrb.bmrb as bmrb
        import trizod.potenci.potenci as potenci
        import trizod.scoring.scoring as scoring

        entry = bmrb.BmrbEntry("4493", BMRB_DIR + "/bmr4493")
        peptide_shifts = entry.get_peptide_shifts()
        assert len(peptide_shifts) > 0

        for (stID, entity_assemID, entityID), (shifts, condID, assemID, sampleIDs) in peptide_shifts.items():
            entity = entry.entities[entityID]
            seq = entity.seq
            conds = entry.conditions[condID]
            temperature = conds.get_temperature()
            pH = conds.get_pH()
            ion = conds.get_ionic_strength()

            usephcor = pH != 7.0
            predshiftdct = potenci.getpredshifts(
                seq, temperature, pH, ion, usephcor, pkacsvfile=False
            )
            ret = scoring.get_offset_corrected_wSCS(seq, shifts, predshiftdct)
            assert ret is not None, "Score computation returned None"
            shw, ashwi, cmp_mask, olf, offf, shw0, ashwi0, ol0, off0 = ret
            assert cmp_mask.any(), "No comparable backbone shifts found"
            break  # only test first peptide
