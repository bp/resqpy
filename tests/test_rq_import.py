import resqpy.rq_import as rqi

import numpy as np
from numpy.testing import assert_array_almost_equal


## TODO: move to a general utilities area once complete
def simple_grid_corns(k_gap = False, righthanded = True):
    """Returns corner points for simple 2x2x2 grid"""
    origin_cell = np.array([
        [
            [
                [0, 2, 0],  # top, back, left, K+, J+, I-
                [1, 2, 0]
            ],  # top, back, right, K+, J+, I+
            [
                [0, 1, 0],  # top, front, left, K+, J-, I-
                [1, 1, 0]
            ]
        ],  # top, front, right, K+, J-, I+
        [
            [
                [0, 2, 1],  # bottom, back, left, K-, J+, I-
                [1, 2, 1]
            ],  # bottom, back, right, K-, J+, I+
            [
                [0, 1, 1],  # bottom, front, left, K-, J-, I-
                [1, 1, 1]
            ]
        ]
    ])  # bottom, front, right, K-, J-, I+

    if not righthanded:
        origin_cell[:, :, :, 1] = origin_cell[:, :, :, 1] * -1
        origin_cell[:, :, :, 1] += 2

    c000 = origin_cell.copy()
    c100 = origin_cell.copy()
    c010 = origin_cell.copy()
    c110 = origin_cell.copy()
    c001 = origin_cell.copy()
    c101 = origin_cell.copy()
    c011 = origin_cell.copy()
    c111 = origin_cell.copy()

    c100[:, :, :, 0] += 1,  # k0, j0, i1
    c110[:, :, :, 0] += 1  # k0, j1, i1
    c101[:, :, :, 0] += 1  # k1, j0, i1
    c111[:, :, :, 0] += 1  # k1, j1, i1

    if righthanded:
        c010[:, :, :, 1] -= 1  # k0, j1, i0
        c110[:, :, :, 1] -= 1  # k0, j1, i1
        c011[:, :, :, 1] -= 1  # k1, j1, i0
        c111[:, :, :, 1] -= 1  # k1, j1, i1
    else:
        c010[:, :, :, 1] += 1  # k0, j1, i0
        c110[:, :, :, 1] += 1  # k0, j1, i1
        c011[:, :, :, 1] += 1  # k1, j1, i0
        c111[:, :, :, 1] += 1  # k1, j1, i1

    c001[:, :, :, 2] += 1  # k1, j0, i0
    c101[:, :, :, 2] += 1  # k1, j0, i1
    c011[:, :, :, 2] += 1  # k1, j1, i0
    c111[:, :, :, 2] += 1  # k1, j1, i1

    corns = np.array(
        [
            [
                [
                    c000,  # k0, j0, i0
                    c100
                ],  # k0, j0, i1
                [
                    c010,  # k0, j1, i0
                    c110
                ]
            ],  # k0, j1, i1
            [
                [
                    c001,  # k1, j0, i0
                    c101
                ],  # k1, j0, i1
                [
                    c011,  # k1, j1, i0
                    c111
                ]
            ]
        ],
        dtype = 'float')  # k1, j1, i1

    if k_gap:
        corns[1, :, :, :, :, :, 2] += 1
    return corns


def test_grid_from_cp_simple(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    corns = simple_grid_corns()

    # Act
    grid = rqi.grid_from_cp(model, cp_array = corns, crs_uuid = crs.uuid, ijk_handedness = None)

    # Assert
    assert grid is not None
    assert_array_almost_equal(grid.extent_kji, (2, 2, 2))
    assert grid.grid_is_right_handed
    assert grid.k_gaps is None
    assert grid.k_direction_is_down
    assert grid.pillar_shape == 'curved'
    assert grid.crs_uuid == crs.uuid


def test_grid_from_cp_simple_left(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    corns = simple_grid_corns(righthanded = False)

    # Act
    grid = rqi.grid_from_cp(model, cp_array = corns, crs_uuid = crs.uuid, ijk_handedness = None)

    # Assert
    assert grid is not None
    assert_array_almost_equal(grid.extent_kji, (2, 2, 2))
    assert not grid.grid_is_right_handed
    assert grid.k_gaps is None
    assert grid.k_direction_is_down
    assert grid.pillar_shape == 'curved'
    assert grid.crs_uuid == crs.uuid


def test_grid_from_cp_simple_straight(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    corns = simple_grid_corns()

    # Act
    grid = rqi.grid_from_cp(model, cp_array = corns, crs_uuid = crs.uuid, known_to_be_straight = True)

    # Assert
    assert grid is not None
    assert_array_almost_equal(grid.extent_kji, (2, 2, 2))
    assert grid.grid_is_right_handed
    assert grid.k_gaps is None
    assert grid.k_direction_is_down
    assert grid.pillar_shape == 'straight'


def test_grid_from_cp_kgap(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    corns = simple_grid_corns(k_gap = True)

    # Act
    grid = rqi.grid_from_cp(model, cp_array = corns, crs_uuid = crs.uuid)

    # Assert
    assert grid is not None
    assert_array_almost_equal(grid.extent_kji, (2, 2, 2))
    assert grid.grid_is_right_handed
    assert grid.k_gaps == 1
    assert grid.k_direction_is_down


def test_grid_from_cp_kgap_zvoid(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    corns = simple_grid_corns()

    # Add a k-gap
    corns[1, :, :, :, :, :, 2] += 0.5

    # Act
    grid = rqi.grid_from_cp(model, cp_array = corns, crs_uuid = crs.uuid, max_z_void = 1)

    # Assert
    assert grid is not None
    assert_array_almost_equal(grid.extent_kji, (2, 2, 2))
    assert grid.grid_is_right_handed
    assert grid.k_gaps is None
    assert grid.k_direction_is_down
