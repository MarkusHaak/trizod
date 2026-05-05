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

    def test_rereference_mode_flag_in_help(self):
        """The --rereference-mode flag is exposed in trizod --help."""
        result = subprocess.run(
            [sys.executable, "-m", "trizod.trizod", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "--rereference-mode" in result.stdout
        assert "{none,lacs,potenci-only,both}" in result.stdout


@requires_bmrb_data
class TestPipelineSingleEntry:
    """Test pipeline on a single BMRB entry."""

    def test_parse_bmrb_entry(self):
        import trizod.bmrb.bmrb as bmrb

        entry = bmrb.BmrbEntry("4493", BMRB_DIR / "bmr4493")
        assert entry.id == "4493"
        assert len(entry.entities) > 0
        assert len(entry.shift_tables) > 0

    def test_compute_scores_single_entry(self):
        import trizod.bmrb.bmrb as bmrb
        import trizod.potenci.potenci as potenci
        import trizod.scoring.scoring as scoring

        entry = bmrb.BmrbEntry("4493", BMRB_DIR / "bmr4493")
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
                seq, temperature, pH, ion, use_ph_corr
            )
            ret = scoring.get_offset_corrected_shifts(seq, shifts, predshiftdct)
            assert ret is not None, "Score computation returned None"
            (
                weighted_diffs_final,
                abs_weighted_diffs_final,
                cmp_mask,
                outlier_mask_final,
                offsets_final,
                weighted_diffs_initial,
                abs_weighted_diffs_initial,
                outlier_mask_initial,
                offsets_initial,
                lacs_offsets,
            ) = ret
            assert cmp_mask.any(), "No comparable backbone shifts found"
            break  # only test first peptide
