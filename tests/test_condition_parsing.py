"""Sample-condition parsing: the NMR-STAR audit findings I-02 and I-03.

I-02 -- rows typed ``pH*`` or ``pD`` were ignored (only ``Type == "pH"`` was
read), so 167 entries fell back to the pH 7.0 default, which also switches
POTENCI's pH correction off. Eight chains are typed ``pH`` but carry the unit
``pD``. A ``pH*`` reading is used as pH; a ``pD`` is brought onto that scale as
pD - 0.4.

I-03 -- four smaller defects in the same lines:
  (a) any Type containing "temp" was read and the last row kept, so a
      ``temperature controller setting`` row overrode the calibrated
      ``temperature`` (50428_1_1_1 scored at 265 K instead of 283 K);
  (b) the ionic-strength unit ``mu`` (the symbol mu, already molar) was scaled
      as micromolar (4065_1_1_1 at 1.5e-7 M instead of 0.15 M);
  (c) ``convert_val`` was a bare ``float()``, so "281.5K", "293,15" and
      "298-303" fell back to the default (52233_1_1_1 16.5 K off);
  (d) physically impossible values (18863: pH 0.000) reached POTENCI; a
      recorded impossible value is now invalid (NaN at every tier), and
      only an absent value gets the tier's default.

Synthetic condition lists go through a real pynmrstar saveframe, so the Type
matching in ``SampleConditions.__init__`` is exercised as well as the getters.
The regressions on real entries skip when the BMRB mirror is absent.
"""

import math

import pynmrstar
import pytest

from tests.conftest import requires_bmrb_data
from trizod import paths
from trizod.bmrb.bmrb import SampleConditions

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _conditions(*rows):
    """``SampleConditions`` parsed from (Type, Val, Val_units) rows.

    An empty string is written as the NMR-STAR null ``.``.
    """
    sf = pynmrstar.Saveframe.from_scratch(
        "sample_conditions_1", "_Sample_condition_list"
    )
    sf.add_tag("Sf_category", "sample_conditions")
    sf.add_tag("ID", "1")
    if rows:
        loop = pynmrstar.Loop.from_scratch("_Sample_condition_variable")
        loop.add_tag(["Type", "Val", "Val_err", "Val_units"])
        loop.add_data(
            [[t, v or ".", ".", u or "."] for t, v, u in rows],
        )
        sf.add_loop(loop)
    return SampleConditions(sf)


_TEMP = ("temperature", "298", "K")


# --------------------------------------------------------------------------- #
# I-02: pH* and pD
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("type_", ["pH*", "PH*", "ph*"])
def test_ph_star_row_is_read_as_ph(type_):
    # bmr4741: 'pH* 2.50 n/a' -- the uncorrected meter reading in D2O.
    cond = _conditions((type_, "2.50", "n/a"), _TEMP)
    assert cond.get_pH() == pytest.approx(2.5)
    assert cond.get_pH(return_default=False) == pytest.approx(2.5)
    assert cond.get_pH_scale() == "pH*"


@pytest.mark.parametrize("unit", ["pH", "pD", "n/a"])
def test_pd_row_is_shifted_onto_the_ph_star_scale(unit):
    # 89 rows are typed pD (72 of them with the unit 'pH'): pD = pH* + 0.4.
    cond = _conditions(("pD", "7.4", unit), _TEMP)
    assert cond.get_pH() == pytest.approx(7.0)
    assert cond.get_pH_scale() == "pD"


def test_ph_typed_row_with_pd_unit_is_a_pd():
    # Eight chains are typed 'pH' but carry the unit 'pD'.
    cond = _conditions(("pH", "4.4", "pD"), _TEMP)
    assert cond.get_pH() == pytest.approx(4.0)
    assert cond.get_pH_scale() == "pD"


def test_ph_typed_row_with_ph_star_unit_is_used_unchanged():
    cond = _conditions(("pH", "4.4", "pH*"), _TEMP)
    assert cond.get_pH() == pytest.approx(4.4)
    assert cond.get_pH_scale() == "pH*"


@pytest.mark.parametrize("other", [("pH*", "5.5", "n/a"), ("pD", "5.9", "pH")])
@pytest.mark.parametrize("plain_first", [True, False])
def test_plain_ph_row_wins_over_ph_star_and_pd(other, plain_first):
    # bmr4052/4053 deposit both 'pH 5.1' and 'pH* 5.5' in one list.
    plain = ("pH", "5.1", "n/a")
    rows = (plain, other) if plain_first else (other, plain)
    cond = _conditions(*rows, _TEMP)
    assert cond.get_pH() == pytest.approx(5.1)
    assert cond.get_pH_scale() == "pH"


def test_pd_unit_on_a_bare_instance():
    # Instances built without a saveframe (as older tests do) carry no
    # pH_scale attribute; the scale is then read from the unit.
    cond = SampleConditions.__new__(SampleConditions)
    cond.id = "bare"
    cond.ionic_strength = cond.pressure = cond.temperature = (None, None)
    cond.pH = ("7.4", "pD")
    assert cond.get_pH() == pytest.approx(7.0)
    assert cond.get_pH_scale() == "pD"


def test_plain_ph_is_unchanged():
    cond = _conditions(("pH", "6.60", "n/a"), _TEMP)
    assert cond.get_pH() == pytest.approx(6.6)
    assert cond.get_pH_scale() == "pH"


def test_missing_ph_keeps_default_and_nan():
    cond = _conditions(_TEMP)
    assert cond.get_pH() == 7.0
    assert math.isnan(cond.get_pH(return_default=False))
    assert cond.get_pH_scale() is None


def test_null_ph_star_keeps_default_and_nan():
    cond = _conditions(("pH*", "", "n/a"), _TEMP)
    assert cond.get_pH() == 7.0
    assert math.isnan(cond.get_pH(return_default=False))


# --------------------------------------------------------------------------- #
# I-03 (a): a temperature controller setting never overrides the temperature
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("plain_first", [True, False])
def test_controller_setting_does_not_override_temperature(plain_first):
    # bmr50428 list 2: temperature 283 K, controller setting 265 K.
    plain = ("temperature", "283", "K")
    ctrl = ("temperature controller setting", "265", "K")
    rows = (plain, ctrl) if plain_first else (ctrl, plain)
    cond = _conditions(*rows)
    assert cond.get_temperature() == pytest.approx(283.0)


def test_controller_setting_is_used_when_it_is_the_only_temperature():
    # bmr15380, 16484, 18809 ... deposit only the controller setting.
    cond = _conditions(("temperature controller setting", "280", "K"))
    assert cond.get_temperature() == pytest.approx(280.0)


# --------------------------------------------------------------------------- #
# I-03 (b): ionic-strength unit 'mu' is the symbol mu, i.e. molar
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("unit", ["mu", "MU", "μ", "µ"])  # Greek mu, micro sign
def test_ionic_strength_unit_mu_is_molar(unit):
    cond = _conditions(("ionic strength", "0.150", unit))
    assert cond.get_ionic_strength() == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("val", "unit", "molar"), [("0.15", "M", 0.15), ("150", "mM", 0.15)]
)
def test_ionic_strength_m_and_mm_unchanged(val, unit, molar):
    cond = _conditions(("ionic strength", val, unit))
    assert cond.get_ionic_strength() == pytest.approx(molar)


def test_ionic_strength_type_with_underscore_is_read():
    # bmr4345 list 2 types it 'ionic_strength'.
    cond = _conditions(("ionic_strength", "0.25", "M"))
    assert cond.get_ionic_strength() == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# I-03 (c): units in the value, decimal commas and ranges
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("298", 298.0),
        (" 7.0 ", 7.0),
        ("1e-06", 1e-06),
        ("-5", -5.0),  # a sign, not a range
        ("281.5K", 281.5),  # bmr52233
        ("293K", 293.0),
        ("298k", 298.0),
        ("298 K", 298.0),
        ("293,15", 293.15),  # decimal comma
        ("0,092", 0.092),  # decimal comma (ionic strength, M)
        ("298-303", 300.5),  # range -> midpoint
        ("230 to 283", 256.5),
        ("7.5-8", 7.75),
    ],
)
def test_convert_val_parses_deposited_spellings(raw, expected):
    assert _conditions().convert_val(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw", [None, "", ".", "n/a", "na", "acid", "no salts", "<0.1", "< 30", "1,000"]
)
def test_convert_val_unparseable_is_nan(raw):
    # '<0.1' is a bound, not a value; '1,000' is ambiguous (thousands or decimal).
    assert math.isnan(_conditions().convert_val(raw))


@pytest.mark.parametrize(
    ("val", "unit", "kelvin"),
    [
        ("281.5K", "K", 281.5),  # bmr52233
        ("293,15", "K", 293.15),
        ("298-303", "K", 300.5),
        # The unit in the value is used: '45' alone would be read as Celsius.
        ("45F", ".", 280.3722222),
        ("20-30", "C", 298.15),
    ],
)
def test_temperature_spellings(val, unit, kelvin):
    cond = _conditions(("temperature", val, unit))
    assert cond.get_temperature() == pytest.approx(kelvin)


def test_ph_range_uses_midpoint():
    cond = _conditions(("pH", "7.5-8", "pH"))
    assert cond.get_pH() == pytest.approx(7.75)


def test_ionic_strength_decimal_comma():
    cond = _conditions(("ionic strength", "0,092", "M"))
    assert cond.get_ionic_strength() == pytest.approx(0.092)


# --------------------------------------------------------------------------- #
# I-03 (d): a recorded but impossible value is invalid (NaN at every tier);
# only an absent value gets the default
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("val", ["0.000", "0.4", "14.5", "70"])
def test_impossible_ph_is_invalid_not_defaulted(val):
    # bmr18863 deposits pH 0.000: NaN even where the tier imputes defaults, so
    # the pH window rejects the entry instead of scoring it at pH 7.0.
    cond = _conditions(("pH", val, "pH"), _TEMP)
    assert math.isnan(cond.get_pH())
    assert math.isnan(cond.get_pH(return_default=False))


@pytest.mark.parametrize("val", ["0.5", "14"])
def test_ph_window_edges_are_kept(val):
    cond = _conditions(("pH", val, "pH"), _TEMP)
    assert cond.get_pH() == pytest.approx(float(val))


def test_pd_is_range_checked_after_conversion():
    # pD 0.8 -> pH* 0.4: below the window, so invalid.
    cond = _conditions(("pD", "0.8", "pH"), _TEMP)
    assert math.isnan(cond.get_pH())


@pytest.mark.parametrize(
    ("val", "unit"),
    [("0", "K"), ("105", "K"), ("100-110", "K"), ("400", "K"), ("150", "C")],
)
def test_impossible_temperature_is_invalid_not_defaulted(val, unit):
    cond = _conditions(("temperature", val, unit))
    assert math.isnan(cond.get_temperature())
    assert math.isnan(cond.get_temperature(return_default=False))


@pytest.mark.parametrize(
    ("val", "unit", "kelvin"),
    [("250", "K", 250.0), ("373", "K", 373.0), ("-5", "C", 268.15)],
)
def test_temperature_window_edges_are_kept(val, unit, kelvin):
    cond = _conditions(("temperature", val, unit))
    assert cond.get_temperature() == pytest.approx(kelvin)


def test_negative_ionic_strength_is_invalid_not_defaulted():
    cond = _conditions(("ionic strength", "-0.1", "M"))
    assert math.isnan(cond.get_ionic_strength())
    assert math.isnan(cond.get_ionic_strength(return_default=False))


def test_zero_ionic_strength_is_kept():
    # bmr18863 deposits 0.000 M: implausible as a buffer, but not impossible.
    cond = _conditions(("ionic strength", "0.000", "M"))
    assert cond.get_ionic_strength() == 0.0


@pytest.mark.parametrize("val", ["", "n/a", "acid"])
def test_absent_conditions_keep_default_where_the_tier_imputes(val):
    # Absent (null) or not a number at all: the historical default when the
    # tier imputes (return_default=True), NaN/None when it does not.
    cond = _conditions(("pH", val, "pH"), ("temperature", val, "K"))
    cond.ionic_strength = (val, "M")
    assert cond.get_pH() == 7.0
    assert cond.get_temperature() == 298.0
    assert cond.get_ionic_strength() == 0.1
    assert math.isnan(cond.get_pH(return_default=False))
    assert math.isnan(cond.get_temperature(return_default=False))
    assert cond.get_ionic_strength(return_default=False) is None


def test_absent_rows_keep_default_where_the_tier_imputes():
    cond = _conditions()  # no condition rows at all
    assert cond.get_pH() == 7.0
    assert cond.get_temperature() == 298.0
    assert cond.get_ionic_strength() == 0.1


# --------------------------------------------------------------------------- #
# real entries (skip without the BMRB mirror)
# --------------------------------------------------------------------------- #


def _entry_conditions(entry_id):
    """{condition-list ID: SampleConditions} for one deposited entry."""
    files = sorted((paths.RAW_BMRB / f"bmr{entry_id}").glob("*.str"))
    if not files:
        pytest.skip(f"BMRB entry {entry_id} not available")
    entry = pynmrstar.Entry.from_file(str(files[0]))
    conds = [
        SampleConditions(sf)
        for sf in entry.get_saveframes_by_category("sample_conditions")
    ]
    return {c.id: c for c in conds}


@requires_bmrb_data
@pytest.mark.parametrize(("entry_id", "ph"), [("4741", 2.5), ("16028", 3.0)])
def test_real_ph_star_entries(entry_id, ph):
    cond = _entry_conditions(entry_id)["1"]
    assert cond.get_pH_scale() == "pH*"
    assert cond.get_pH(return_default=False) == pytest.approx(ph)


@requires_bmrb_data
def test_real_controller_setting_entry():
    conds = _entry_conditions("50428")
    assert conds["1"].get_temperature() == pytest.approx(283.0)  # not 273
    assert conds["2"].get_temperature() == pytest.approx(283.0)  # not 265


@requires_bmrb_data
def test_real_mu_ionic_strength_entry():
    cond = _entry_conditions("4065")["1"]
    assert cond.get_ionic_strength(return_default=False) == pytest.approx(0.15)


@requires_bmrb_data
def test_real_unit_in_value_entry():
    cond = _entry_conditions("52233")["1"]
    assert cond.get_temperature(return_default=False) == pytest.approx(281.5)


@requires_bmrb_data
def test_real_ph_zero_entry():
    # pH 0.000 is recorded, so invalid: NaN even with defaults on.
    cond = _entry_conditions("18863")["1"]
    assert math.isnan(cond.get_pH(return_default=False))
    assert math.isnan(cond.get_pH())


@requires_bmrb_data
def test_real_impossible_temperature_entry():
    # bmr2606 records 218 K: invalid, not the 298 K default.
    cond = _entry_conditions("2606")["1"]
    assert math.isnan(cond.get_temperature())
