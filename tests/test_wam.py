import pytest

import math as maths
import numpy as np

import resqpy.olio.weights_and_measures as wam

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
            assert maths.isclose(rate, wam.convert_volumes(wam.convert_volumes(rate, from_units, to_units),
                                                           to_units, from_units))

def test_flow_rate_conversion():
   assert maths.isclose(wam.convert_flow_rates(1000.0, 'm3/d', 'ft3/d'), 35314.6667, rel_tol = 1.0e-5)
   assert maths.isclose(wam.convert_flow_rates(1000.0, 'm3/d', 'bbl/d'), 1000.0 / 0.158987, rel_tol = 1.0e-4)
   units = ('m3/d', 'ft3/s', 'bbl/d', 'stb/hr', 'm3/ms', '1000 bbl/d', '1E6 ft3/d')
   for rate in (2100.0, 0.0, -312.45):
      for from_units in units:
         for to_units in units:
            assert maths.isclose(rate, wam.convert_flow_rates(wam.convert_flow_rates(rate, from_units, to_units),
                                                              to_units, from_units))
