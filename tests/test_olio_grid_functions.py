import os
import numpy as np
import pytest

from resqpy.olio.grid_functions import left_right_foursome, infill_block_geometry, resequence_nexus_corp, random_cell, determine_corp_ijk_handedness, determine_corp_extent, translate_corp, triangles_for_cell_faces, actual_pillar_shape, columns_to_nearest_split_face
import resqpy.model as rq
import resqpy.grid as grr
from resqpy.lines import Polyline
from resqpy.derived_model import add_faults
from resqpy.olio.random_seed import seed


def test_infill_block_geometry():

    # --------- Arrange ----------
    x_ = np.array([2, 4, 6, 8])
    y_ = np.array([1, 3, 5, 7])
    depth_ = np.array([10.0, 30.0, 50.0, 70.0])
    x, y, depth = np.meshgrid(x_, y_, depth_)
    extent = (4, 4, 4)
    thickness = np.full(extent, 15)

    # Set two inactive cells
    depth[1, 2, 1] = 0.001  # below the default depth_zero_tolerance so should be set to 0 by function
    depth[1, 2, 2] = 0.001
    # Assertions to check that depth does have 2 values == 0.001
    assert np.isclose(depth[1, 2, 1], 0.001)
    assert np.isclose(depth[1, 2, 2], 0.001)
    np.testing.assert_almost_equal(np.unique(thickness), np.array([15.]))
    # --------- Act ----------
    infill_block_geometry(extent = extent, depth = depth.transpose(), thickness = thickness, x = x, y = y)
    # --------- Assert ----------
    assert len(depth[np.isclose(depth, 0.)]) == 0
    assert len(thickness[np.isclose(thickness, 22.)]) == 2


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
    assert np.isclose(depth[1, 2, 1], 0.001)
    assert np.isclose(depth[1, 2, 2], 0.001)
    np.testing.assert_almost_equal(np.unique(thickness), np.array([15.]))
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

    # --------- Arrange ----------
    corner_points = np.array([[[[[[[0., 0., 0.], [10., 0., 0.]], [[10., -10., 0.], [0., -10., 0.]]],
                                 [[[0., 0., 10.], [10., 0., 10.]], [[10., -10., 10.], [0., -10., 10.]]]]]]])

    expected_result = np.array([[[[[[[0., 0., 0.], [10., 0., 0.]], [[0., -10., 0.], [10., -10., 0.]]],
                                   [[[0., 0., 10.], [10., 0., 10.]], [[0., -10., 10.], [10., -10., 10.]]]]]]])

    # --------- Act ----------
    resequence_nexus_corp(corner_points = corner_points)

    # --------- Assert ----------
    np.testing.assert_almost_equal(corner_points, expected_result)


def test_random_cell(tmp_path):

    # --------- Arrange----------
    seed(1923877)
    epc = os.path.join(tmp_path, 'grid.epc')
    model = rq.new_model(epc)

    # create a basic block grid
    dxyz = (10.0, 10.0, 100.0)
    grid = grr.RegularGrid(model,
                           extent_kji = (4, 2, 2),
                           title = 'grid_1',
                           origin = (0.0, 10.0, 1000.0),
                           dxyz = dxyz,
                           as_irregular_grid = True)

    grid_points = grid.points_ref(masked = False)

    # pinch out cells in the k == 2 layer
    grid_points[3] = grid_points[2]

    # collapse cell kji0 == 0,0,0 in the j direction
    grid_points[0:2, 1, 0:2, 1] = 10.  # same as origin y value

    # store grid
    grid.write_hdf5()
    grid.create_xml(add_cell_length_properties = True)
    grid_uuid = grid.uuid
    model.store_epc()

    # check that the grid can be read
    model = rq.Model(epc)
    grid_reloaded = grr.any_grid(model, uuid = grid_uuid)
    corner_points = grid_reloaded.corner_points()

    # --------- Act----------
    # call random_cell function 50 times
    trial_number = 0
    while trial_number < 50:
        (k, j, i) = random_cell(corner_points = corner_points, border = 0.0)
        # --------- Assert----------
        assert 0 <= k < 4
        assert 0 <= j < 2
        assert 0 <= i < 2
        # check that none of the k,j,i combinations that correspond to pinched cells are chosen by the random_cell function
        assert (k, j, i) not in [(0, 0, 0), (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1)]
        trial_number += 1

    # reshape corner get the extent of the new grid
    corner_points_reshaped = corner_points.reshape(1, 1, 16, 2, 2, 2, 3)
    new_extent = determine_corp_extent(corner_points = corner_points_reshaped)
    assert np.all(new_extent == np.array([4, 2, 2], dtype = int))


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
    assert tri.shape == (3, 2, 4, 3, 3)


def test_actual_pillar_shape(tmp_path):

    # --------- Arrange----------
    epc = os.path.join(tmp_path, 'grid.epc')
    model = rq.new_model(epc)

    # create a basic block grid
    dxyz = (10.0, 10.0, 10.0)
    vertical_grid = grr.RegularGrid(model,
                                    extent_kji = (2, 2, 2),
                                    title = 'vert_grid',
                                    origin = (0.0, 0.0, 0.0),
                                    dxyz = dxyz)
    straight_grid = grr.RegularGrid(model,
                                    extent_kji = (2, 2, 2),
                                    title = 'straight_grid',
                                    origin = (10.0, 10.0, 0.0),
                                    dxyz = dxyz,
                                    as_irregular_grid = True)
    curved_grid = grr.RegularGrid(model,
                                  extent_kji = (3, 2, 2),
                                  title = 'curved_grid',
                                  origin = (10.0, 10.0, 0.0),
                                  dxyz = dxyz,
                                  as_irregular_grid = True)

    # shift the corner points of cellkji0 == (0, 0, 0) by - 10 x and -10 y units
    straight_grid.corner_points()[0][0][0][0][0][0] = np.array([0., 0., 0])
    straight_grid.corner_points()[0][0][0][0][0][1] = np.array([10., 0., 0])

    # shift 2 corner points of cellkji0 == (1, 0, 0) by -5 x units
    curved_grid.corner_points()[0][0][0][1][0][0] = np.array([5., 10., 10])
    curved_grid.corner_points()[0][0][0][1][0][1] = np.array([15., 10., 10])

    # --------- Act----------
    pillar_shape_vertical = actual_pillar_shape(pillar_points = vertical_grid.corner_points())
    pillar_shape_straight = actual_pillar_shape(pillar_points = straight_grid.corner_points())
    pillar_shape_curved = actual_pillar_shape(pillar_points = curved_grid.corner_points())

    # --------- Assert----------
    assert pillar_shape_vertical == 'vertical'
    assert pillar_shape_straight == 'straight'
    assert pillar_shape_curved == 'curved'


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
    # yapf: disable
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
# yapf: enable
def test_translate_corp(x_shift, y_shift, expected_result):

    # --------- Arrange----------
    # yapf: disable
    corner_points = np.array([[[[[[[0.001,   0.001,  0.001], [10.001,   0.001,  0.001]],
                                  [[0.001, -10.001,  0.001], [10.001, -10.001,  0.001]]],
                                 [[[0.001,   0.001, 10.001], [10.001,   0.001, 10.001]],
                                  [[0.001, -10.001, 10.001], [10.001, -10.001, 10.001]]]]]]])
    # yapf: enable

    # --------- Act----------
    translate_corp(corner_points = corner_points, x_shift = x_shift, y_shift = y_shift, preserve_digits = -3)

    # --------- Assert----------
    np.testing.assert_almost_equal(expected_result, corner_points)
