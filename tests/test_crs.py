# test crs creation and conversion from one crs to another

import pytest
import numpy as np
import resqpy.model as rq
import resqpy.crs as rqc


def test_crs():

   # create some coordinate reference systems
   model = rq.Model(new_epc = True, create_basics = True)
   crs_default = rqc.Crs(model)
   crs_m = rqc.Crs(model, xy_units = 'm', z_units = 'm')
   crs_ft = rqc.Crs(model, xy_units = 'ft', z_units = 'ft')
   crs_mixed = rqc.Crs(model, xy_units = 'm', z_units = 'ft')
   crs_offset = rqc.Crs(model, xy_units = 'm', z_units = 'm', x_offset = 100.0, y_offset = -100.0, z_offset = -50.0)
   crs_elevation = rqc.Crs(model, z_inc_down = 'False')

   # check that distincitveness is recognised
   assert crs_default.is_equivalent(crs_m)
   assert not crs_m.is_equivalent(crs_ft)
   assert not crs_mixed.is_equivalent(crs_m)
   assert not crs_m.is_equivalent(crs_offset)
   assert not crs_m.is_equivalent(crs_elevation)

   # create some xml
   crs_m.create_xml()
   crs_offset.create_xml()

   # test conversion
   ft_to_m = 0.3048

   a = np.empty((10, 3))
   a[:, 0] = np.random.random(10) *  5.0e5
   a[:, 1] = np.random.random(10) * 10.0e5
   a[:, 2] = np.random.random(10) *  4.0e3

   b = a.copy()
   crs_m.convert_array_from(crs_default, a)
   assert np.max(np.abs(b - a)) < 1.0e-6
   a[:] = b
   crs_m.convert_array_to(crs_m, a)
   assert np.all(a == b)
   crs_ft.convert_array_from(crs_m, a)
   assert np.max(np.abs(b - a * ft_to_m)) < 1.0e-6
   crs_ft.convert_array_to(crs_m, a)
   assert np.max(np.abs(b - a)) < 1.0e-6
   a[:] = b
   crs_m.local_to_global_array(a)
   assert np.max(np.abs(b - a)) < 1.0e-6
   a[:] = b
   crs_offset.global_to_local_array(a)
   a[:, 0] += 100.0
   a[:, 1] -= 100.0
   a[:, 2] -= 50.0
   assert np.max(np.abs(b - a)) < 1.0e-6
