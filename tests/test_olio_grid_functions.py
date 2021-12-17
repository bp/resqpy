import os
import numpy as np
import pytest

from resqpy.olio.grid_functions import left_right_foursome, infill_block_geometry, resequence_nexus_corp, random_cell, determine_corp_ijk_handedness, determine_corp_extent, translate_corp, triangles_for_cell_faces, actual_pillar_shape
import resqpy.model as rq
import resqpy.grid as grr


def test_infill_block_geometry():

    # --------- Arrange ----------
    x_ = np.array([2, 4, 6, 8])
    y_ = np.array([1, 3, 5, 7])
    depth_ = np.array([10.0, 30.0, 50.0, 70.0])
    x, y, depth = np.meshgrid(x_, y_, depth_)
    extent = (4, 4, 4)
    thickness = np.ones(extent) * 15

    # Set two inactive cells
    depth[1, 2, 1] = 0.001  # below the default depth_zero_tolerance so should be set to 0 by function
    depth[1, 2, 2] = 0.001
    # Assertions to check that depth does have 2 values == 0.001
    assert len(depth[depth == 0.001]) == 2
    assert np.unique(thickness) == np.array([15.])
    # --------- Act ----------
    infill_block_geometry(extent = extent, depth = depth.transpose(), thickness = thickness, x = x, y = y)
    # --------- Assert ----------
    assert len(depth[depth == 0]) == 0
    assert len(thickness[thickness == 22.5]) == 2


def test_infill_block_geometry_log_warnings(caplog):

    # --------- Arrange ----------
    x_ = np.array([2, 4, 6, 8])
    y_ = np.array([1, 3, 5, 7])
    depth_ = np.array([10.0, 30.0, 50.0, 70.0])
    x, y, depth = np.meshgrid(x_, y_, depth_)
    extent = (4, 4, 4)
    thickness = np.ones(extent) * 15

    # Set two inactive cells
    depth[1, 2, 1] = 0.001  # below the default depth_zero_tolerance so should be set to 0 by function
    depth[1, 2, 2] = 0.001
    # Assertions to check that depth does have 2 values == 0.001
    assert len(depth[depth == 0.001]) == 2
    assert np.unique(thickness) == np.array([15.])
    # --------- Act ----------
    infill_block_geometry(extent = extent, depth = depth, thickness = thickness, x = x, y = y)
    # --------- Assert ----------
    assert 'Check k_increase_direction and tolerances' in caplog.text
    assert 'Cells [3, 3, 1] and [3, 3, 3] overlap ...' in caplog.text
    assert 'Skipping rest of i,j column' in caplog.text


@pytest.mark.parametrize('previous, next, expected_result', [
    ((1, 0), (0, 1), np.array([[False, True], [True, True]], dtype = bool)),
    ((1, 0), (1, 2), np.array([[False, False], [True, True]], dtype = bool)),
    ((1, 0), (2, 1), np.array([[False, False], [True, False]], dtype = bool)),
    ((1, 2), (0, 1), np.array([[False, True], [False, False]], dtype = bool)),
    ((1, 2), (1, 0), np.array([[True, True], [False, False]], dtype = bool)),
    ((1, 2), (2, 1), np.array([[True, True], [True, False]], dtype = bool)),
    ((0, 1), (1, 0), np.array([[True, False], [False, False]], dtype = bool)),
    ((0, 1), (2, 1), np.array([[True, False], [True, False]], dtype = bool)),
    ((0, 1), (1, 2), np.array([[True, False], [True, True]], dtype = bool)),
    ((2, 1), (1, 0), np.array([[True, True], [False, True]], dtype = bool)),
    ((2, 1), (0, 1), np.array([[False, True], [False, True]], dtype = bool)),
    ((2, 1), (1, 2), np.array([[False, False], [False, True]], dtype = bool)),
])
def test_left_right_foursome(previous, next, expected_result):

    # --------- Arrange ----------
    full_p_list = [previous, (1, 1), next]
    p_index = 1
    # --------- Act ----------
    result = left_right_foursome(full_pillar_list = full_p_list, p_index = p_index)
    # --------- Assert ----------
    np.testing.assert_equal(result, expected_result)


def test_left_right_foursome_error_handling():

    # --------- Arrange ----------
    p_index = 1
    # arrange for exceptions
    full_p_list_entry_error = [(1, 0), (1, 2), (0, 1)]
    full_p_list_exit_error = [(1, 0), (1, 1), (2, 2)]

    # --------- Act and Assert----------
    with pytest.raises(Exception) as excinfo:
        result_entry_error = left_right_foursome(full_pillar_list = full_p_list_entry_error, p_index = p_index)
        assert "code failure whilst taking entry sides from dubious full pillar list" in str(excinfo.value)
    with pytest.raises(Exception) as excinfo2:
        result_exit_error = left_right_foursome(full_pillar_list = full_p_list_exit_error, p_index = p_index)
        assert "code failure whilst taking exit sides from dubious full pillar list" in str(excinfo.value)


def test_resequence_nexus_corp():

    # TODO: confirm you used the correct cornerpoint ordering from Nexus (pg 1424 Nexus Keyword file)
    # --------- Arrange ----------
    corner_points = np.array([[[[[[[0., 0., 0.], [10., 0., 0.]], [[10., -10., 0.], [0., -10., 0.]]],
                                 [[[0., 0., 10.], [10., 0., 10.]], [[10., -10., 10.], [0., -10., 10.]]]]]]])

    expected_result = np.array([[[[[[[0., 0., 0.], [10., 0., 0.]], [[0., -10., 0.], [10., -10., 0.]]],
                                   [[[0., 0., 10.], [10., 0., 10.]], [[0., -10., 10.], [10., -10., 10.]]]]]]])

    # --------- Act ----------
    resequence_nexus_corp(corner_points = corner_points)

    # --------- Assert ----------
    np.testing.assert_almost_equal(corner_points, expected_result)


# TODO: Find out if grid is too simple
def test_random_cell(tmp_path):

    # --------- Arrange----------
    epc = os.path.join(tmp_path, 'grid.epc')
    model = rq.new_model(epc)

    # create a basic block grid
    dxyz = (55.0, 65.0, 27.0)
    grid = grr.RegularGrid(model, extent_kji = (4, 3, 2), title = 'concrete', origin = (0.0, 0.0, 1000.0), dxyz = dxyz)
    grid.create_xml(add_cell_length_properties = True)
    corner_points = grid.corner_points()

    # --------- Act----------
    (k, j, i) = random_cell(corner_points = corner_points)

    # --------- Assert----------
    assert k <= 4
    assert j <= 3
    assert i <= 2


def test_triangles_for_cell_faces():
    # --------- Arrange----------
    corner_points = np.array([[[[0., 0., 0.], [10., 0., 0.]], [[0., -10., 0.], [10., -10., 0.]]],
                              [[[0., 0., 10.], [10., 0., 10.]], [[0., -10., 10.], [10., -10., 10.]]]])
    # --------- Act----------
    tri = triangles_for_cell_faces(cp = corner_points)
    # face centre points
    k_face_centre_points = [np.array([5., -5., 0.]), np.array([5., -5., 10])]
    j_face_centre_points = [np.array([5., 0., 5.]), np.array([5., -10., 5.])]
    i_face_centre_points = [np.array([0., -5., 5.]), np.array([10., -5., 5.])]

    # --------- Assert----------
    np.testing.assert_almost_equal(tri[0, :, :, 0][0][0], k_face_centre_points[0])
    np.testing.assert_almost_equal(tri[0, :, :, 0][1][0], k_face_centre_points[1])
    np.testing.assert_almost_equal(tri[1, :, :, 0][0][0], j_face_centre_points[0])
    np.testing.assert_almost_equal(tri[1, :, :, 0][1][0], j_face_centre_points[1])
    np.testing.assert_almost_equal(tri[2, :, :, 0][0][0], i_face_centre_points[0])
    np.testing.assert_almost_equal(tri[2, :, :, 0][1][0], i_face_centre_points[1])


def test_actual_pillar_shape(tmp_path):  # TODO: test with more complicated grid geometries

    # --------- Arrange----------
    epc = os.path.join(tmp_path, 'grid.epc')
    model = rq.new_model(epc)

    # create a basic block grid
    dxyz = (10.0, 10.0, 10.0)
    grid = grr.RegularGrid(model, extent_kji = (3, 3, 2), title = 'grid1', origin = (0.0, 0.0, 1000.0), dxyz = dxyz)

    # --------- Act----------
    pillar_shape = actual_pillar_shape(pillar_points = grid.corner_points())

    # --------- Assert----------
    assert pillar_shape == 'vertical'


def test_determine_corp_ijk_handedness():

    # --------- Arrange----------
    corner_points_right = np.array([[[[[[[0., 0., 0.], [10., 0., 0.]], [[0., -10., 0.], [10., -10., 0.]]],
                                       [[[0., 0., 10.], [10., 0., 10.]], [[0., -10., 10.], [10., -10., 10.]]]]]]])

    corner_points_left = np.array([[[[[[[0., 0., 0.], [10., 0., 0.]], [[0., 10., 0.], [10., 10., 0.]]],
                                      [[[0., 0., 10.], [10., 0., 10.]], [[0., 10., 10.], [10., 10., 10.]]]]]]])

    # --------- Act----------
    handedness_right = determine_corp_ijk_handedness(corner_points = corner_points_right)
    handedness_left = determine_corp_ijk_handedness(corner_points = corner_points_left)

    # --------- Assert----------
    assert handedness_right == 'right'
    assert handedness_left == 'left'


# TODO: Find out if using a grid is too simple
def test_determine_corp_extent(tmp_path):

    # --------- Arrange----------
    epc = os.path.join(tmp_path, 'grid.epc')
    model = rq.new_model(epc)

    # create a basic block grid
    dxyz = (55.0, 65.0, 27.0)
    grid = grr.RegularGrid(model, extent_kji = (4, 3, 2), title = 'concrete', origin = (0.0, 0.0, 1000.0), dxyz = dxyz)
    grid.create_xml(add_cell_length_properties = True)
    corner_points = grid.corner_points()
    corner_points_reshaped = corner_points.reshape(1, 1, 24, 2, 2, 2, 3)
    # --------- Act----------
    [nk, nj, ni] = determine_corp_extent(corner_points = corner_points_reshaped)

    # --------- Assert----------
    assert [nk, nj, ni] == [4, 3, 2]


@pytest.mark.parametrize(
    'x_shift, y_shift, expected_result',
    [(2, 0,
      np.array([[[[[[[2.001, 0.001, 0.001], [12.001, 0.001, 0.001]], [[2.001, -10.001, 0.001], [12.001, -10.001, 0.001]]
                   ],
                   [[[2.001, 0.001, 10.001], [12.001, 0.001, 10.001]],
                    [[2.001, -10.001, 10.001], [12.001, -10.001, 10.001]]]]]]])),
     (0, 2,
      np.array([[[[[[[0.001, 2.001, 0.001], [10.001, 2.001, 0.001]], [[0.001, -8.001, 0.001], [10.001, -8.001, 0.001]]],
                   [[[0.001, 2.001, 10.001], [10.001, 2.001, 10.001]],
                    [[0.001, -8.001, 10.001], [10.001, -8.001, 10.001]]]]]]]))])
def test_translate_corp(x_shift, y_shift, expected_result):

    # --------- Arrange----------
    corner_points = np.array([[[[[[[0.001, 0.001, 0.001], [10.001, 0.001, 0.001]],
                                  [[0.001, -10.001, 0.001], [10.001, -10.001, 0.001]]],
                                 [[[0.001, 0.001, 10.001], [10.001, 0.001, 10.001]],
                                  [[0.001, -10.001, 10.001], [10.001, -10.001, 10.001]]]]]]])

    # --------- Act----------
    translate_corp(corner_points = corner_points, x_shift = x_shift, y_shift = y_shift, preserve_digits = -3)
    # print(corner_points)
    # --------- Assert----------
    np.testing.assert_almost_equal(expected_result, corner_points)
