"""Regression: the ionic-strength 'unit unknown' log names Molar, not Kelvin."""

import logging

import pytest

from trizod.bmrb.bmrb import SampleConditions


def _make_conditions():
    cond = SampleConditions.__new__(SampleConditions)
    cond.id = "test"
    cond.ionic_strength = (None, None)
    cond.pH = (None, None)
    cond.pressure = (None, None)
    cond.temperature = (None, None)
    return cond


def test_ionic_strength_unknown_unit_logs_molar_not_kelvin(caplog):
    """An unrecognized ionic-strength unit is assumed to be Molar; the INFO log
    must say so (previously copied 'assuming K' from the temperature branch)."""
    cond = _make_conditions()
    cond.ionic_strength = ("0.15", "weird-unit")
    with caplog.at_level(logging.INFO, logger="trizod.bmrb"):
        val = cond.get_ionic_strength()
    assert val == pytest.approx(0.15)  # value is unaffected either way
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "assuming M" in msgs
    assert "assuming K" not in msgs
