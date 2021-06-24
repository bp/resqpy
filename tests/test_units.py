"""Unit tests for pint unit registry"""

import math as maths

import pint
import pytest

import resqpy.olio.weights_and_measures as bwam
from resqpy.olio.weights_and_measures import Q_, ureg


@pytest.mark.parametrize("unit", [
    "m",
    "metres",
    "ft",
    "ft[US]",
])
def test_units_are_understood(unit):
    ureg.parse_units(unit)


@pytest.mark.skip("Not ready yet")
def test_all_uoms_understood():
    all_uoms = bwam.properties_data()['uoms']
    assert len(all_uoms) > 100

    for uom in all_uoms:
        try:
            ureg.parse_units(uom)
        except Exception as e:
            raise ValueError(f"Cannot handle uom '{uom}'") from e


def test_undefined_units_raise_error():
    with pytest.raises(pint.UndefinedUnitError):
        Q_(1, 'foobar')


@pytest.mark.parametrize("unit_from, unit_to, factor", [
    ("m", "ft", 3.28084),
    ("ft", "ft[US]", 0.999998),
    ("pu", "v/v", 0.01),
    ("%", "v/v", 0.01),
    ("pu", "%", 1),
    ("p.u.", "%", 1),
])
def test_unit_conversion(unit_from, unit_to, factor):
    old = Q_(1, unit_from)
    
    new = old.to(ureg.parse_units(unit_to))
    assert maths.isclose(new.magnitude, factor, rel_tol=1e-5)

    # NB this does not work:    new = old.to(unit_to)    
    # Looks like we should always first use ureg.parse_units()
