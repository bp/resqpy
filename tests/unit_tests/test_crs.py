# test crs creation and conversion from one crs to another

import os
import math as maths
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import resqpy.crs as rqc
import resqpy.model as rq
import resqpy.olio.uuid as bu


def test_switch_axes():
    xy = rqc.switch_axes("easting northing", "easting northing", 2.0, 3.0)
    assert_array_almost_equal(xy, (2.0, 3.0))
    xy = rqc.switch_axes("northing easting", "northing easting", 2.0, 3.0)
    assert_array_almost_equal(xy, (2.0, 3.0))
    xy = rqc.switch_axes("easting northing", "northing easting", 2.0, 3.0)
    assert_array_almost_equal(xy, (3.0, 2.0))
    xy = rqc.switch_axes("northing easting", "easting northing", -2.0, 3.0)
    assert_array_almost_equal(xy, (3.0, -2.0))
    xy = rqc.switch_axes("easting northing", "westing southing", 2.0, -3.0)
    assert_array_almost_equal(xy, (-2.0, 3.0))
    xy = rqc.switch_axes("westing southing", "easting northing", 2.0, 3.0)
    assert_array_almost_equal(xy, (-2.0, -3.0))
    xy = rqc.switch_axes("northing easting", "westing southing", -2.0, -3.0)
    assert_array_almost_equal(xy, (3.0, 2.0))
    xy = rqc.switch_axes("westing southing", "northing easting", 2.0, 3.0)
    assert_array_almost_equal(xy, (-3.0, -2.0))
    xy = rqc.switch_axes("westing southing", "northing easting", 0.0, 0.0)
    assert_array_almost_equal(xy, (0.0, 0.0))
    xy = rqc.switch_axes("westing southing", "westing southing", 2.0, 3.0)
    assert_array_almost_equal(xy, (2.0, 3.0))
    xy = rqc.switch_axes("easting northing", "southing westing", 2.0, 3.0)
    assert_array_almost_equal(xy, (-3.0, -2.0))
    xy = rqc.switch_axes("southing westing", "northing easting", 2.0, 3.0)
    assert_array_almost_equal(xy, (-2.0, -3.0))
    xy = rqc.switch_axes("easting northing", "northing westing", 2.0, 3.0)
    assert_array_almost_equal(xy, (3.0, -2.0))
    xy = rqc.switch_axes("northing westing", "easting northing", 2.0, 3.0)
    assert_array_almost_equal(xy, (-3.0, 2.0))
    xy = rqc.switch_axes("northing westing", "westing southing", 2.0, 3.0)
    assert_array_almost_equal(xy, (3.0, -2.0))
    xy = rqc.switch_axes("westing southing", "northing westing", 2.0, 3.0)
    assert_array_almost_equal(xy, (-3.0, 2.0))
    xy = rqc.switch_axes("westing northing", "westing southing", 2.0, 3.0)
    assert_array_almost_equal(xy, (2.0, -3.0))
    xy = rqc.switch_axes("westing southing", "westing northing", 2.0, 3.0)
    assert_array_almost_equal(xy, (2.0, -3.0))


def test_switch_axes_array():
    a = np.array([(2.0, 3.0, 5.0), (7.0, 11.0, 13.0), (17.0, 19.0, 23.0), (29.0, 31.0, 37.0)], dtype = float)
    xyz = a.copy()
    rqc.switch_axes_array("easting northing", "easting northing", xyz)
    assert_array_almost_equal(xyz, a)
    xyz = a.copy()
    rqc.switch_axes_array("northing easting", "northing easting", xyz)
    assert_array_almost_equal(xyz, a)
    xyz = a.copy()
    rqc.switch_axes_array("easting northing", "northing easting", xyz)
    assert_array_almost_equal(xyz[:, 0], a[:, 1])
    assert_array_almost_equal(xyz[:, 1], a[:, 0])
    assert_array_almost_equal(xyz[:, 2], a[:, 2])
    xyz = a.copy()
    rqc.switch_axes_array("northing easting", "easting northing", xyz)
    assert_array_almost_equal(xyz[:, 0], a[:, 1])
    assert_array_almost_equal(xyz[:, 1], a[:, 0])
    assert_array_almost_equal(xyz[:, 2], a[:, 2])
    xyz = a.copy()
    rqc.switch_axes_array("easting northing", "westing southing", xyz)
    assert_array_almost_equal(xyz[:, 0], -a[:, 0])
    assert_array_almost_equal(xyz[:, 1], -a[:, 1])
    assert_array_almost_equal(xyz[:, 2], a[:, 2])
    xyz = a.copy()
    rqc.switch_axes_array("westing southing", "easting northing", xyz)
    assert_array_almost_equal(xyz[:, 0], -a[:, 0])
    assert_array_almost_equal(xyz[:, 1], -a[:, 1])
    xyz = a.copy()
    rqc.switch_axes_array("northing easting", "westing southing", xyz)
    assert_array_almost_equal(xyz[:, 0], -a[:, 1])
    assert_array_almost_equal(xyz[:, 1], -a[:, 0])
    xy = a[:, :2].copy()  # check working for xy data only
    rqc.switch_axes_array("westing southing", "northing easting", xy)
    assert_array_almost_equal(xy[:, 0], -a[:, 1])
    assert_array_almost_equal(xy[:, 1], -a[:, 0])
    xyz = a.copy()
    rqc.switch_axes_array("westing southing", "northing easting", xyz)
    assert_array_almost_equal(xyz[:, 0], -a[:, 1])
    assert_array_almost_equal(xyz[:, 1], -a[:, 0])
    xyz = a.copy()
    rqc.switch_axes_array("westing southing", "westing southing", xyz)
    assert_array_almost_equal(xyz, a)
    xyz = a.copy()
    rqc.switch_axes_array("easting northing", "southing westing", xyz)
    assert_array_almost_equal(xyz[:, 0], -a[:, 1])
    assert_array_almost_equal(xyz[:, 1], -a[:, 0])
    xyz = a.copy()
    rqc.switch_axes_array("southing westing", "northing easting", xyz)
    assert_array_almost_equal(xyz[:, 0], -a[:, 0])
    assert_array_almost_equal(xyz[:, 1], -a[:, 1])
    xyz = a.copy()
    rqc.switch_axes_array("easting northing", "northing westing", xyz)
    assert_array_almost_equal(xyz[:, 0], a[:, 1])
    assert_array_almost_equal(xyz[:, 1], -a[:, 0])
    xyz = a.copy()
    rqc.switch_axes_array("northing westing", "easting northing", xyz)
    assert_array_almost_equal(xyz[:, 0], -a[:, 1])
    assert_array_almost_equal(xyz[:, 1], a[:, 0])
    xyz = a.copy()
    rqc.switch_axes_array("northing westing", "westing southing", xyz)
    assert_array_almost_equal(xyz[:, 0], a[:, 1])
    assert_array_almost_equal(xyz[:, 1], -a[:, 0])
    xyz = a.copy()
    rqc.switch_axes_array("westing southing", "northing westing", xyz)
    assert_array_almost_equal(xyz[:, 0], -a[:, 1])
    assert_array_almost_equal(xyz[:, 1], a[:, 0])
    xyz = a.copy()
    rqc.switch_axes_array("westing northing", "westing southing", xyz)
    assert_array_almost_equal(xyz[:, 0], a[:, 0])
    assert_array_almost_equal(xyz[:, 1], -a[:, 1])
    xyz = a.copy()
    rqc.switch_axes_array("westing southing", "westing northing", xyz)
    assert_array_almost_equal(xyz[:, 0], a[:, 0])
    assert_array_almost_equal(xyz[:, 1], -a[:, 1])
    # check also working for a single point
    xyz = np.array((3.0, -5.0, 7.0), dtype = float)
    rqc.switch_axes_array("northing westing", "westing southing", xyz)
    assert_array_almost_equal(xyz, (-5.0, -3.0, 7.0))


def test_crs(tmp_path):
    # create some coordinate reference systems
    model = rq.new_model(os.path.join(tmp_path, 'crs_test.epc'))
    crs_default = rqc.Crs(model)
    assert crs_default.null_transform
    crs_m = rqc.Crs(model, xy_units = 'm', z_units = 'm')
    crs_em_a = rqc.Crs(model, xy_units = 'm', z_units = 'm', extra_metadata = {'ab': 'a'})
    crs_em_b = rqc.Crs(model, xy_units = 'm', z_units = 'm', extra_metadata = {'ab': 'b'})
    crs_em_c = rqc.Crs(model, xy_units = 'm', z_units = 'm', extra_metadata = {'c': 'a'})
    crs_ft = rqc.Crs(model, xy_units = 'ft', z_units = 'ft')
    crs_mixed = rqc.Crs(model, xy_units = 'm', z_units = 'ft')
    crs_offset = rqc.Crs(model, xy_units = 'm', z_units = 'm', x_offset = 100.0, y_offset = -100.0, z_offset = -50.0)
    assert not crs_offset.null_transform
    crs_elevation = rqc.Crs(model, z_inc_down = False)
    crs_rotate = rqc.Crs(model, rotation = maths.pi / 2.0, rotation_units = 'rad')
    crs_north_rotate = rqc.Crs(model,
                               axis_order = 'northing easting',
                               rotation = maths.pi / 2.0,
                               rotation_units = 'rad')
    crs_offset_rotate = rqc.Crs(model,
                                x_offset = 10.0,
                                y_offset = -10.0,
                                z_offset = -5.0,
                                rotation = maths.pi / 2.0,
                                rotation_units = 'rad')
    crs_south = rqc.Crs(model, axis_order = 'southing westing')
    crs_south_elevation = rqc.Crs(model, axis_order = 'southing westing', z_inc_down = False)
    crs_time_s = rqc.Crs(model, xy_units = 'm', time_units = 's')
    crs_time_ms = rqc.Crs(model, xy_units = 'm', time_units = 'ms')
    for crs_time in [crs_time_s, crs_time_ms]:
        assert crs_time.resqml_type == 'LocalTime3dCrs'

    # check that distincitveness is recognised
    assert crs_default.is_equivalent(crs_m)
    assert not crs_em_a == crs_m
    assert not crs_em_b == crs_em_a
    assert not crs_em_c == crs_em_a
    assert not crs_m.is_equivalent(crs_ft)
    assert not crs_mixed.is_equivalent(crs_m)
    assert not crs_m.is_equivalent(crs_offset)
    assert not crs_m.is_equivalent(crs_offset_rotate)
    assert not crs_m.is_equivalent(crs_elevation)
    assert not crs_m.is_equivalent(crs_rotate)
    assert not crs_m.is_equivalent(crs_north_rotate)
    assert not crs_m.is_equivalent(crs_south)
    assert not crs_time_s.is_equivalent(crs_time_ms)
    assert not crs_south_elevation.z_inc_down
    for depth_crs in [crs_default, crs_m, crs_ft, crs_mixed, crs_offset, crs_elevation, crs_rotate, crs_south]:
        assert depth_crs.resqml_type == 'LocalDepth3dCrs'
        assert not crs_time_s == depth_crs
        assert not crs_time_ms == depth_crs

    # check handedness
    assert not crs_m.is_right_handed_xy()
    assert not crs_m.is_right_handed_xyz()
    assert not crs_elevation.is_right_handed_xy()
    assert not crs_offset_rotate.is_right_handed_xy()
    assert crs_elevation.is_right_handed_xyz()
    assert crs_north_rotate.is_right_handed_xy()
    assert crs_south.is_right_handed_xy()
    assert crs_south.is_right_handed_xyz()
    assert crs_south_elevation.is_right_handed_xy()
    assert not crs_south_elevation.is_right_handed_xyz()

    # create some xml
    for crs in [
            crs_default, crs_m, crs_ft, crs_mixed, crs_offset, crs_elevation, crs_rotate, crs_south, crs_time_s,
            crs_time_ms, crs_offset_rotate, crs_em_a
    ]:
        crs.create_xml()
    model.store_epc()
    # check re-use of equivalent crs'es
    assert bu.matching_uuids(crs_default.uuid, crs_m.uuid)

    # test conversion
    ft_to_m = 0.3048

    a = np.empty((10, 3))
    a[:, 0] = np.random.random(10) * 5.0e5
    a[:, 1] = np.random.random(10) * 10.0e5
    a[:, 2] = np.random.random(10) * 4.0e3

    b = a.copy()
    crs_m.convert_array_from(crs_default, a)
    assert_array_almost_equal(a, b)
    a[:] = b
    crs_m.convert_array_to(crs_m, a)
    assert np.all(a == b)
    crs_ft.convert_array_from(crs_m, a)
    assert_array_almost_equal(b, a * ft_to_m)
    crs_ft.convert_array_to(crs_m, a)
    assert_array_almost_equal(a, b)
    a[:] = b
    crs_m.local_to_global_array(a)
    assert_array_almost_equal(a, b)
    a[:] = b
    crs_offset.global_to_local_array(a)
    a[:, 0] += 100.0
    a[:, 1] -= 100.0
    a[:, 2] -= 50.0
    assert_array_almost_equal(a, b)
    a[:] = b
    crs_mixed.convert_array_from(crs_m, a)
    assert_array_almost_equal(b[..., :2], a[..., :2])
    assert_array_almost_equal(b[..., 2], a[..., 2] * ft_to_m)
    a[:] = b
    crs_south.convert_array_from(crs_m, a)
    assert_array_almost_equal(a[..., 0], -b[..., 1])
    assert_array_almost_equal(a[..., 1], -b[..., 0])
    assert_array_almost_equal(a[..., 2], b[..., 2])
    a[:] = b
    crs_south_elevation.convert_array_from(crs_m, a)
    assert_array_almost_equal(a[..., 0], -b[..., 1])
    assert_array_almost_equal(a[..., 1], -b[..., 0])
    assert_array_almost_equal(a[..., 2], -b[..., 2])

    # test single point conversion
    p = (456.78, 678.90, -1234.56)
    assert_array_almost_equal(p, crs_offset.global_to_local(crs_offset.local_to_global(p)))
    p_ft = crs_m.convert_to(crs_ft, np.array(p))
    assert_array_almost_equal(p, crs_m.convert_from(crs_ft, p_ft))

    #  test time conversion
    pt = (123456.0, 234567.0, 1983.0)
    pt_s = np.array(crs_time_ms.convert_to(crs_time_s, pt))
    pt_s[2] *= 1000.0  # convert from seconds back to milliseconds
    assert_array_almost_equal(pt, pt_s)

    # todo: test rotation
    p = (0.00, 234.00, 5678.90)
    pr = crs_rotate.local_to_global(p)
    assert_array_almost_equal(pr, (234.00, 0.00, 5678.90))
    assert_array_almost_equal(crs_rotate.global_to_local(pr), p)


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
