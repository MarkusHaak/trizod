"""Unit tests for SampleConditions unit conversions (temperature, pH, ionic
strength). These exercise the conversion arithmetic directly without needing
real BMRB data by constructing a bare SampleConditions instance."""

import pytest

from trizod.bmrb.bmrb import SampleConditions


def _make_conditions():
    """A SampleConditions with no saveframe; fields set directly by tests."""
    cond = SampleConditions.__new__(SampleConditions)
    cond.id = "test"
    cond.ionic_strength = (None, None)
    cond.pH = (None, None)
    cond.pressure = (None, None)
    cond.temperature = (None, None)
    return cond


@pytest.mark.parametrize(
    "fahrenheit,expected_kelvin",
    [
        (32.0, 273.15),  # freezing point
        (77.0, 298.15),  # 25 C, common NMR temperature
        (98.6, 310.15),  # body temperature, 37 C
        (68.0, 293.15),  # 20 C
    ],
)
def test_fahrenheit_to_kelvin(fahrenheit, expected_kelvin):
    """K = (F - 32) * 5/9 + 273.15 (not * 1.8)."""
    cond = _make_conditions()
    cond.temperature = (str(fahrenheit), "F")
    assert cond.get_temperature() == pytest.approx(expected_kelvin, abs=1e-6)


def test_fahrenheit_outlier_heuristic():
    """A unit-less value in [50, 100] is assumed to be Fahrenheit and must use
    the same (F - 32) * 5/9 + 273.15 conversion."""
    cond = _make_conditions()
    cond.temperature = ("77", "")  # no explicit unit -> outlier heuristic
    assert cond.get_temperature() == pytest.approx(298.15, abs=1e-6)


def test_celsius_to_kelvin():
    cond = _make_conditions()
    cond.temperature = ("25", "C")
    assert cond.get_temperature() == pytest.approx(298.15, abs=1e-6)


def test_kelvin_passthrough():
    cond = _make_conditions()
    cond.temperature = ("298.15", "K")
    assert cond.get_temperature() == pytest.approx(298.15, abs=1e-6)


def test_celsius_outlier_heuristic():
    """A unit-less value in [1, 50) is assumed to be Celsius."""
    cond = _make_conditions()
    cond.temperature = ("25", "")
    assert cond.get_temperature() == pytest.approx(298.15, abs=1e-6)
