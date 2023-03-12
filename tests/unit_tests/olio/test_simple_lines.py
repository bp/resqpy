import math as maths
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.model as rq
import resqpy.grid as grr
import resqpy.olio.simple_lines as sl


def test_polygon_line():
    l1 = np.array([(123.0, 456.0, 0.0), (987.0, 456.0, 0.0), (500.0, 800.0, 0.0)], dtype = float)
    l2 = sl.polygon_line(l1, tolerance = 0.001)
    assert_array_almost_equal(l2, l1)
    l1c = np.concatenate((l1, np.expand_dims(l1[0], axis = 0)), axis = 0)
    l2 = sl.polygon_line(l1c, tolerance = 0.1)
    assert len(l2) == len(l1c) - 1
    assert_array_almost_equal(l2, l1)
    l1 = np.array([(123.0, 456.0, 0.0), (987.0, 456.0, 10.0), (500.0, 800.0, 20.0), (123.0, 456.0, 30.0)],
                  dtype = float)
    l2 = sl.polygon_line(l1, tolerance = 0.1)
    assert_array_almost_equal(l2, l1)
    l1 = np.array([(123.0, 456.0, 0.0), (987.0, 456.0, 10.0), (500.0, 800.0, 20.0), (123.3, 456.3, 0.3)], dtype = float)
    l2 = sl.polygon_line(l1, tolerance = 1.0)
    assert_array_almost_equal(l2, l1[:-1])


def test_duplicate_vertices_removed():
    l1 = np.array([(123.0, 456.0, 0.0), (987.0, 456.0, 0.0), (500.0, 800.0, 0.0)], dtype = float)
    l2 = sl.duplicate_vertices_removed(l1)
    assert_array_almost_equal(l2, l1)
    l1 = np.array([(123.0, 456.0, 0.0), (987.0003, 456.0003, 0.0003), (987.0, 456.0, 0.0), (500.0, 800.0, 0.0),
                   (500.0003, 800.0003, 0.0003)],
                  dtype = float)
    l2 = sl.duplicate_vertices_removed(l1)
    assert len(l2) == len(l1) - 2
    l2 = sl.duplicate_vertices_removed(l1, tolerance = 0.0005)
    assert_array_almost_equal(l2, l1)


# nearest_rods(line_list, projection, grid, axis, ref_slice0 = 0, plus_face = False)
def test_nearest_rods(example_model_and_crs):
    model, crs = example_model_and_crs
    grid = grr.RegularGrid(model,
                           extent_kji = (10, 10, 10),
                           dxyz = (1.0, 2.0, 0.5),
                           origin = (100.0, 200.0, 300.0),
                           crs_uuid = crs.uuid,
                           title = 'grid for rods',
                           as_irregular_grid = True)
    grid.write_hdf5_from_caches()
    grid.create_xml()
    model.store_epc()
    model = rq.Model(model.epc_file)
    grid = model.grid()
    lines = [
        np.array([(101.9, 210.3, 303.1), (107.2, 207.8, 301.6)], dtype = float),
        np.array([(100.0, 200.0, 300.0), (102.0, 204.0, 301.0), (104.0, 208.0, 302.0), (106.0, 212.0, 303.0)],
                 dtype = float)
    ]
    e_ki_0 = np.array([(6, 2), (3, 7)], dtype = int)
    e_kj_0 = np.array([(6, 5), (3, 4)], dtype = int)
    e_1 = np.array([(0, 0), (2, 2), (4, 4), (6, 6)], dtype = int)
    for ref_slice0 in (0, 5):
        for plus_face in (False, True):
            rod_list = sl.nearest_rods(lines, 'xz', grid, 'J', ref_slice0 = ref_slice0, plus_face = plus_face)
            assert len(rod_list) == len(lines)
            for i in range(len(lines)):
                assert len(rod_list[i]) == len(lines[i]) and rod_list[i].ndim == 2 and rod_list[i].shape[1] == 2
            assert np.all(rod_list[0] == e_ki_0)
            assert np.all(rod_list[1] == e_1)
            rod_list = sl.nearest_rods(lines, 'yz', grid, 'I', ref_slice0 = ref_slice0, plus_face = plus_face)
            assert len(rod_list) == len(lines)
            for i in range(len(lines)):
                assert len(rod_list[i]) == len(lines[i]) and rod_list[i].ndim == 2 and rod_list[i].shape[1] == 2
            assert np.all(rod_list[0] == e_kj_0)
            assert np.all(rod_list[1] == e_1)
