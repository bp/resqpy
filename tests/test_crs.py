# test crs creation and conversion from one crs to another

import pytest
import numpy as np
import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.olio.uuid as bu


def test_crs():

   # create some coordinate reference systems
   model = rq.Model(new_epc = True, create_basics = True)
   crs_default = rqc.Crs(model)
   assert crs_default.null_transform
   crs_m = rqc.Crs(model, xy_units = 'm', z_units = 'm')
   crs_ft = rqc.Crs(model, xy_units = 'ft', z_units = 'ft')
   crs_mixed = rqc.Crs(model, xy_units = 'm', z_units = 'ft')
   crs_offset = rqc.Crs(model, xy_units = 'm', z_units = 'm', x_offset = 100.0, y_offset = -100.0, z_offset = -50.0)
   assert not crs_offset.null_transform
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
   a[:, 0] = np.random.random(10) * 5.0e5
   a[:, 1] = np.random.random(10) * 10.0e5
   a[:, 2] = np.random.random(10) * 4.0e3

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

   # todo: test rotation


def test_crs_reuse():
   model = rq.Model(new_epc = True, create_basics = True)
   crs_a = rqc.Crs(model)
   crs_a.create_xml()
   crs_b = rqc.Crs(model)
   crs_b.create_xml()
   assert len(model.parts(obj_type = 'LocalDepth3dCrs')) == 1
   assert crs_a == crs_b
   assert bu.matching_uuids(crs_a.uuid, crs_b.uuid)
   crs_c = rqc.Crs(model, z_inc_down = False)
   crs_c.create_xml()
   assert len(model.parts(obj_type = 'LocalDepth3dCrs')) == 2
   assert crs_c != crs_a
   assert not bu.matching_uuids(crs_c.uuid, crs_a.uuid)
   crs_d = rqc.Crs(model, z_units = 'ft')
   crs_d.create_xml()
   assert len(model.parts(obj_type = 'LocalDepth3dCrs')) == 3
   crs_e = rqc.Crs(model, z_inc_down = False)
   crs_e.create_xml()
   assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 3
   assert crs_e == crs_c
   assert bu.matching_uuids(crs_e.uuid, crs_c.uuid)
   crs_f = rqc.Crs(model)
   crs_f.create_xml(reuse = False)
   assert len(model.parts(obj_type = 'LocalDepth3dCrs')) == 4
   assert crs_f == crs_a
   assert not bu.matching_uuids(crs_f.uuid, crs_a.uuid)

   # todo: test parent epsg code equivalence
