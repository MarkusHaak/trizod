"""Placeholder ("mostly X") sequences never reach a dataset pool (audit I-08).

Many 1990s depositions give the polymer sequence as literal "X" everywhere
except the assigned residues (bmr1712: 250 X then SRDVHSKVISEE). The
unfiltered tier allows any X fraction and ``MIN_SEQ_LEN`` counts the X's, so
41-43 such chains became unfiltered-tier training representatives — and MMseqs2
cannot align X, so 505_1_1_1 (Cu/Zn SOD, 111 of 118 X) escaped leakage control
against the CheZOD117 test chain 15711_1_1_1.

``trizod dataset build`` drops a chain whose literal-"X" fraction exceeds
``MAX_UNKNOWN_FRACTION`` (0.2, the tolerant tier's ``max-x-fraction``) at the
``MIN_SEQ_LEN`` screen. These tests run the real ``build.main`` on tiny
synthetic inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._stubs import run_build as _run_build
from tests._stubs import scored_record
from trizod.dataset import build
from trizod.io.fasta import read_fasta

TIERS = ("unfiltered", "tolerant", "moderate", "strict")

# --------------------------------------------------------------------------- #
# synthetic build harness (tests/_stubs.py, shared with the withdrawn gate)
# --------------------------------------------------------------------------- #


def _record(rid: str, seq: str, n_bb_pos: int = 15) -> dict:
    return scored_record(rid, seq, n_bb_pos)


def _ranked_ids(final: Path, tier: str) -> set[str]:
    with (final / tier / f"{tier}_all_ranked.tsv").open() as fh:
        next(fh)
        return {line.split("\t", 1)[0] for line in fh if line.strip()}


# 505_1_1_1-like: 118 residues, 111 X, the 7 named ones being metal ligands.
SOD_PLACEHOLDER = "X" * 40 + "H" + "X" * 20 + "HH" + "X" * 30 + "HDH" + "X" * 21 + "H"
REAL_50 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG"
AT_LIMIT = "XXXX" + "ACDEFGHIKLMNPQRS"  # 4 / 20 = 0.20 -> kept
OVER_LIMIT = "XXXXX" + "ACDEFGHIKLMNPQRS"  # 5 / 21 = 0.238 -> dropped


# --------------------------------------------------------------------------- #
# the predicate
# --------------------------------------------------------------------------- #


def test_sod_fixture_mirrors_505():
    assert len(SOD_PLACEHOLDER) == 118
    assert SOD_PLACEHOLDER.count("X") == 111


def test_threshold_is_point_two():
    assert build.MAX_UNKNOWN_FRACTION == 0.2


def test_threshold_is_the_tolerant_tiers_max_x_fraction():
    # the gate is documented as the tolerant max-x-fraction applied to the
    # unfiltered tier; a change to either must be a change to both
    from trizod.trizod import filter_defaults

    tolerant_max_x = filter_defaults.loc["tolerant", "max-x-fraction"]
    assert tolerant_max_x == build.MAX_UNKNOWN_FRACTION


@pytest.mark.parametrize(
    ("seq", "expected"),
    [
        ("XXAA", 0.5),
        ("AAAA", 0.0),
        ("", 0.0),
        # literal X only: '*', 'Z', 'B' and lowercase are other questions
        ("AB*Z", 0.0),
        ("xxAA", 0.0),
    ],
)
def test_unknown_fraction_counts_literal_x(seq, expected):
    assert build.unknown_fraction(seq) == pytest.approx(expected)


def test_placeholder_predicate_keeps_the_limit_itself():
    assert not build.is_placeholder(AT_LIMIT)
    assert build.is_placeholder(OVER_LIMIT)
    assert build.is_placeholder(SOD_PLACEHOLDER)
    assert not build.is_placeholder(REAL_50)


# --------------------------------------------------------------------------- #
# the build gate
# --------------------------------------------------------------------------- #


def _placeholder_build(tmp_path, *extra):
    unfiltered = [
        _record("505_1_1_1", SOD_PLACEHOLDER),
        _record("1001_1_1_1", REAL_50),
        _record("1002_1_1_1", AT_LIMIT),
        _record("1003_1_1_1", OVER_LIMIT),
    ]
    by_tier = {
        "unfiltered": unfiltered,
        "tolerant": [_record("1001_1_1_1", REAL_50)],
        "moderate": [_record("1001_1_1_1", REAL_50)],
        "strict": [_record("1001_1_1_1", REAL_50)],
    }
    return _run_build(tmp_path, by_tier, *extra)


def test_build_drops_placeholder_sequences_from_the_unfiltered_pool(tmp_path):
    final = _placeholder_build(tmp_path)
    pool = read_fasta(final / "unfiltered" / "unfiltered.fasta")
    assert set(pool) == {"1001_1_1_1", "1002_1_1_1"}
    assert SOD_PLACEHOLDER not in pool.values()
    ranked = _ranked_ids(final, "unfiltered")
    assert "505_1_1_1" not in ranked and "1003_1_1_1" not in ranked


def test_build_logs_and_summarises_the_placeholder_drop(tmp_path, capsys):
    final = _placeholder_build(tmp_path)
    out = capsys.readouterr().out
    summary = json.loads((final / "final_dataset_summary.json").read_text())
    assert summary["max_unknown_fraction"] == 0.2
    assert summary["global_dropped_placeholder"] == 2
    assert summary["placeholder_dropped_ids"] == ["1003_1_1_1", "505_1_1_1"]
    assert summary["tiers"]["unfiltered"]["dropped_placeholder"] == 2
    assert summary["tiers"]["strict"]["dropped_placeholder"] == 0
    assert summary["tiers"]["unfiltered"]["max_unknown_fraction"] == 0.2
    assert "placeholder" in out and "2 rows" in out


def test_placeholder_count_is_marginal_to_short_and_bound(tmp_path):
    """A row already removed as too short or as a bound complex is not counted
    again: the placeholder count is what this gate adds."""
    by_tier = {
        "unfiltered": [
            _record("2001_1_1_1", "X" * 15 + "AAA"),  # 18 aa: short
            _record("2002_1_1_1", SOD_PLACEHOLDER),  # bound complex
            _record("2003_1_1_1", OVER_LIMIT),  # the gate's own
            _record("2004_1_1_1", REAL_50),
        ]
    }
    final = _run_build(tmp_path, by_tier, bound_entries=frozenset({"2002"}))
    summary = json.loads((final / "final_dataset_summary.json").read_text())
    tier = summary["tiers"]["unfiltered"]
    assert tier["dropped_short"] == 1
    assert tier["dropped_bound"] == 1
    assert tier["dropped_placeholder"] == 1
    assert tier["kept_rows_after_filters"] == 1


def test_max_unknown_fraction_is_overridable(tmp_path):
    final = _placeholder_build(tmp_path, "--max-unknown-fraction", "1.0")
    pool = read_fasta(final / "unfiltered" / "unfiltered.fasta")
    assert {"505_1_1_1", "1003_1_1_1"} <= set(pool)


def test_gate_is_on_by_default():
    assert build.make_parser().parse_args([]).max_unknown_fraction == 0.2
