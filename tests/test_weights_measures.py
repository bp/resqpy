import math as maths

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.weights_and_measures as wam
from resqpy.olio.exceptions import InvalidUnitError, IncompatibleUnitsError

# ------- Test uoms ------


def test_valid_uoms():

   # Good units
   assert "0.001 gal[US]/gal[US]" in wam.valid_uoms()
   assert "Btu[IT]/min" in wam.valid_uoms()
   assert "%" in wam.valid_uoms()

   # Bad units
   assert "foo barr" not in wam.valid_uoms()
   assert "" not in wam.valid_uoms()
   assert None not in wam.valid_uoms()

   # With attributes, optionally with subset
   for quantity in [None, "length"]:
      uom_dict = wam.valid_uoms(quantity = quantity, return_attributes = True)
      assert "m" in uom_dict.keys()
      assert uom_dict["m"]["name"] == "metre"
      assert uom_dict["m"]["dimension"] == "L"


def test_uom_aliases():
   for uom, aliases in wam.UOM_ALIASES.items():
      assert uom in wam.valid_uoms(), f"Bad uom {uom}"
      for alias in aliases:
         if alias != uom:
            assert alias not in wam.valid_uoms(), f"Bad alias {alias}"

   for alias, uom in wam.UOM_ALIAS_MAP.items():
      assert uom in wam.valid_uoms(), f"Bad uom {uom}"
      if alias != uom:
         assert alias not in wam.valid_uoms(), f"Bad alias {alias}"

   for uom in wam.CASE_INSENSITIVE_UOMS:
      assert uom in wam.valid_uoms(), f"Bad uom {uom}"


def test_aliases_are_unique():
   n_aliases = sum(map(len, wam.UOM_ALIASES.values()))
   all_aliases = set.union(*wam.UOM_ALIASES.values())
   assert n_aliases == len(all_aliases)


# ------- Test parsing uoms -------


@pytest.mark.parametrize("input_uom, expected_uom", [('gapi', 'gAPI'), ('m', 'm'), ('M', 'm'),
                                                     ('gal[UK]/mi', 'gal[UK]/mi'),
                                                     ("0.001 gal[US]/gal[US]", "0.001 gal[US]/gal[US]"),
                                                     ('1E6 m3/day', '1E6 m3/d')])
def test_uom_from_string(input_uom, expected_uom):
   validated_uom = wam.rq_uom(input_uom)
   assert expected_uom == validated_uom


@pytest.mark.parametrize("input_uom", ["foobar", None, "", "qwertyuiol", "bifröst"])
def test_bad_unit_raises_error(input_uom):
   with pytest.raises(InvalidUnitError):
      wam.rq_uom(input_uom)


# ----- Test quantities -------


def test_quantities():

   # Set of all quantities
   quantities = wam.valid_quantities()
   assert "length" in quantities
   assert len(quantities) > 10
   for q in quantities:
      uoms = wam.valid_uoms(quantity = q)
      assert len(uoms) > 0
      for uom in uoms:
         assert uom in wam.valid_uoms(), f"Bad uom {uom}"

   # With attributes
   quantities_dict = wam.valid_quantities(return_attributes = True)
   assert "length" in quantities_dict.keys()
   assert quantities_dict["length"]["dimension"] == "L"
   assert "m" in quantities_dict["length"]["members"]


# ---- Test unit conversions -------


@pytest.mark.parametrize(
   "unit_from, unit_to, value, expected",
   [
      # Straightforward conversions
      ("m", "m", 1, 1),
      ("m", "km", 1, 0.001),
      ("ft", "m", 1, 0.3048),
      ("ft", "ft[US]", 1, 0.999998),

      # Fractional units with aliases
      ("bbl/d", "stb/day", 1, 1),
      ("mmstb/day", "1E6 bbl/d", 1, 1),
      ("mmscf/day", "1E6 ft3/d", 1, 1),
      ("scf/stb", "ft3/bbl", 1, 1),

      # Aliases of common units
      ("metres", "m", 1, 1),
      ("meters", "m", 1, 1),
      ("pu", "%", 1, 1),
      ("p.u.", "%", 1, 1),

      # Different base units!
      ("%", "v/v", 1, 0.01),
      ("pu", "v/v", 1, 0.01),
      ("m3/m3", "%", 1, 100),
      ("D", "ft2", 10, 1.062315e-10),
   ])
@pytest.mark.filterwarnings("ignore:Assuming base units")
def test_unit_conversion(unit_from, unit_to, value, expected):
   result = wam.convert(value, unit_from, unit_to)
   assert maths.isclose(result, expected, rel_tol = 1e-4)


def test_convert_array():
   # Duck typing should work
   value = np.array([1, 2, 3])
   expected = np.array([1000, 2000, 3000])
   result = wam.convert(value, unit_from = "km", unit_to = "m")
   assert_array_almost_equal(result, expected)


def test_conversion_factors_are_numeric():
   for uom in wam.valid_uoms():
      base_unit, dimension, factors = wam.get_conversion_factors(uom)
      assert base_unit in wam.valid_uoms()
      assert len(dimension) > 0
      assert len(factors) == 4, f"Issue with {uom}"
      assert all(isinstance(f, (int, float)) for f in factors), f"Issue with {uom}"


# Test incompatible units raise an Error


@pytest.mark.parametrize("unit_from, unit_to", [
   ("m", "gAPI"),
   ("%", "bbl"),
   ("%", "ft"),
   ("m", "m3"),
])
def test_incompatible_units(unit_from, unit_to):
   with pytest.raises(IncompatibleUnitsError):
      wam.convert(1, unit_from, unit_to)


def test_length_conversion():
   assert maths.isclose(wam.convert_lengths(1.0, 'feet', 'metres'), 0.3048)
   assert maths.isclose(wam.convert_lengths(10.0, 'ft', 'm'), 3.048)
   a = np.random.random((4, 5))
   a_m = 0.3048 * a
   wam.convert_lengths(a, 'ft', 'm')
   assert np.all(np.isclose(a_m, a))
   a = np.random.random((4, 5))
   a_ft = a / 0.3048
   assert np.all(np.isclose(a_ft, wam.convert_lengths(a, 'm', 'ft')))
   b = a.copy()
   wam.convert_lengths(a, 'ft', 'm')
   wam.convert_lengths(a, 'm', 'ft')
   assert np.all(np.isclose(a, b))


def test_time_conversion():
   for period in (1.0, 0.0238, 67.824, 0.0):
      assert maths.isclose(24 * period, wam.convert_times(period, 'days', 'hours'))
      assert maths.isclose(24 * 60 * period, wam.convert_times(period, 'd', 'mins'))
      assert maths.isclose(24 * 60 * 60 * period, wam.convert_times(period, 'day', 'sec'))
      assert maths.isclose(24 * 60 * 60 * 1000 * period, wam.convert_times(period, 'day', 'ms'))
      assert maths.isclose(60 * period, wam.convert_times(period, 'hr', 'min'))
      assert maths.isclose(60 * period, wam.convert_times(period, 'min', 's'))
      assert maths.isclose(period / 24, wam.convert_times(period, 'days', 'hours', invert = True))
      assert maths.isclose(period / (24 * 60), wam.convert_times(period, 'd', 'mins', invert = True))
      assert maths.isclose(period / (24 * 60 * 60), wam.convert_times(period, 'day', 'sec', invert = True))
      assert maths.isclose(period / (24 * 60 * 60 * 1000), wam.convert_times(period, 'day', 'ms', invert = True))
      assert maths.isclose(period / 60, wam.convert_times(period, 'hr', 'min', invert = True))
      assert maths.isclose(period / 60, wam.convert_times(period, 'min', 's', invert = True))
      assert maths.isclose(period / 24, wam.convert_times(period, 'hours', 'days'))
      assert maths.isclose(period / (24 * 60), wam.convert_times(period, 'mins', 'd'))
      assert maths.isclose(period / (24 * 60 * 60), wam.convert_times(period, 'sec', 'day'))
      assert maths.isclose(period / (24 * 60 * 60 * 1000), wam.convert_times(period, 'ms', 'day'))
      assert maths.isclose(period / 60, wam.convert_times(period, 'min', 'hr'))
      assert maths.isclose(period / 60, wam.convert_times(period, 's', 'min'))
   a = np.random.random((2, 3, 4))
   b = a.copy()
   wam.convert_times(a, 'd', 's')
   assert np.all(np.isclose(a, 24.0 * 60.0 * 60.0 * b))
   wam.convert_times(a, 'd', 's', invert = True)
   assert np.all(np.isclose(a, b))


def test_pressure_conversion():
   assert maths.isclose(20684.28, wam.convert_pressures(3000.0, 'psi', 'kPa'), rel_tol = 1.0e-4)
   a = np.random.random(10)
   b = a.copy()
   assert np.all(np.isclose(100.0 * b, wam.convert_pressures(a, 'bar', 'kPa')))
   wam.convert_pressures(a, 'kPa', 'bar')
   assert np.all(np.isclose(b, a))


def test_volume_conversion():
   assert maths.isclose(wam.convert_volumes(1.0, 'm3', 'ft3'), 35.3147, rel_tol = 1.0e-4)
   assert maths.isclose(wam.convert_volumes(1000.0, 'stb', 'm3'), 158.987, rel_tol = 1.0e-4)
   units = ('m3', 'ft3', 'bbl')
   for rate in (2100.0, 0.0, -312.45):
      for from_units in units:
         for to_units in units:
            assert maths.isclose(
               rate, wam.convert_volumes(wam.convert_volumes(rate, from_units, to_units), to_units, from_units))


def test_flow_rate_conversion():
   assert maths.isclose(wam.convert_flow_rates(1000.0, 'm3/d', 'ft3/d'), 35314.6667, rel_tol = 1.0e-5)
   assert maths.isclose(wam.convert_flow_rates(1000.0, 'm3/d', 'bbl/d'), 1000.0 / 0.158987, rel_tol = 1.0e-4)
   units = ('m3/d', 'ft3/s', 'bbl/d', 'stb/hr', '1000 bbl/d', '1E6 ft3/d')
   for rate in (2100.0, 0.0, -312.45):
      for from_units in units:
         for to_units in units:
            assert maths.isclose(
               rate, wam.convert_flow_rates(wam.convert_flow_rates(rate, from_units, to_units), to_units, from_units))
