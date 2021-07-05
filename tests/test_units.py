import math as maths

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.olio.weights_and_measures as bwam
from resqpy.olio.exceptions import InvalidUnitError, IncompatibleUnitsError


# ------- Test uoms ------


def test_valid_uoms():

   # Good units
   assert "0.001 gal[US]/gal[US]" in bwam.valid_uoms()
   assert "Btu[IT]/min" in bwam.valid_uoms()
   assert "%" in bwam.valid_uoms()

   # Bad units
   assert "foo barr" not in bwam.valid_uoms()
   assert "" not in bwam.valid_uoms()
   assert None not in bwam.valid_uoms()


def test_uom_aliases():
   for uom, aliases in bwam.ALIASES.items():
      assert uom in bwam.valid_uoms(), f"Bad uom {uom}"
      for alias in aliases:
         if alias != uom:
            assert alias not in bwam.valid_uoms(), f"Bad alias {alias}"

   for alias, uom in bwam.ALIAS_MAP.items():
      assert uom in bwam.valid_uoms(), f"Bad uom {uom}"
      if alias != uom:
         assert alias not in bwam.valid_uoms(), f"Bad alias {alias}"


# ------- Test parsing uoms -------


@pytest.mark.parametrize("input_uom, expected_uom", [
   ('gapi', 'gAPI'),
   ('m', 'm'),
   ('M', 'm'),
   ('gal[UK]/mi', 'gal[UK]/mi'),
   ("0.001 gal[US]/gal[US]", "0.001 gal[US]/gal[US]"),
   ('1E6 m3/day', '1E6 m3/d')
])
def test_uom_from_string(input_uom, expected_uom):
   validated_uom = bwam.rq_uom(input_uom)
   assert expected_uom == validated_uom


@pytest.mark.parametrize("input_uom", ["foobar", None, "", "qwertyuiol", "bifröst"])
def test_bad_unit_raises_error(input_uom):
   with pytest.raises(InvalidUnitError):
      bwam.rq_uom(input_uom)



# ---- Test unit conversions -------

@pytest.mark.parametrize("unit_from, unit_to, value, expected", [
   # Straightforward conversions
   ("m", "m", 1, 1),
   ("m", "km", 1, 0.001),
   ("ft", "m", 1, 0.3048),
   ("ft", "ft[US]", 1, 0.999998),

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
   result = bwam.convert(value, unit_from, unit_to)
   assert maths.isclose(result, expected, rel_tol=1e-4)


def test_convert_array():
   # Duck typing should work
   value = np.array([1,2,3])
   expected = np.array([1000, 2000, 3000])
   result = bwam.convert(value, unit_from="km", unit_to="m")
   assert_array_almost_equal(result, expected)


def test_conversion_factors_are_numeric():
   for uom in bwam.valid_uoms():
      base_unit, dimension, factors = bwam.get_conversion_factors(uom)
      assert base_unit in bwam.valid_uoms()
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
      bwam.convert(1, unit_from, unit_to)