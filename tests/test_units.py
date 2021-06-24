"""Unit tests for pint unit registry"""

import math as maths

import pint
import pytest

from resqpy.olio.weights_and_measures import Q_, ureg


@pytest.mark.parametrize("unit", [
    "m",
    "metres",
    "ft",
    "ft[US]",
])
def test_units_are_understood(unit):
    Q_(1, unit)


def test_undefined_units_raise_error():
    with pytest.raises(pint.UndefinedUnitError):
        Q_(1, 'foobar')


@pytest.mark.parametrize("unit_from, unit_to, factor", [
    ("m", "ft", 3.28084),
    ("ft", "ft[US]", 0.999998),
])
def test_unit_conversion(unit_from, unit_to, factor):
    old = Q_(1, unit_from)
    new = old.to(unit_to)
    assert maths.isclose(new.magnitude, factor, rel_tol=1e-5)
