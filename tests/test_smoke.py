"""Smoke tests to verify basic functionality before any refactoring."""

import subprocess
import sys

from tests.conftest import BMRB_DIR, requires_bmrb_data


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

        for (_stID, _entity_assemID, entityID), (
            shifts,
            condID,
            _assemID,
            _sampleIDs,
        ) in peptide_shifts.items():
            entity = entry.entities[entityID]
            seq = entity.seq
            conds = entry.conditions[condID]
            temperature = conds.get_temperature()
            pH = conds.get_pH()
            ion = conds.get_ionic_strength()

            use_ph_corr = pH != 7.0
            predshiftdct = potenci.get_pred_shifts(
                seq, temperature, pH, ion, use_ph_corr, pka_csv_path=False
            )
            ret = scoring.get_offset_corrected_wSCS(seq, shifts, predshiftdct)
            assert ret is not None, "Score computation returned None"
            shw, ashwi, cmp_mask, olf, offf, shw0, ashwi0, ol0, off0 = ret
            assert cmp_mask.any(), "No comparable backbone shifts found"
            break  # only test first peptide
