import resqpy.rq_import as rqi
import resqpy.model as rq
import os

import numpy as np
from numpy.testing import assert_array_almost_equal

import pytest


## TODO: move to a general utilities area once complete
def simple_grid_corns(k_gap = False, righthanded = True, undefined = False):
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
    if undefined:
        corns[0, 0, 0, :, :, :, :] = np.nan
    return corns


def test_grid_from_cp_simple(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    corns = simple_grid_corns()
    expected_points = np.array([[[[0, 2, 0], [1, 2, 0], [2, 2, 0]], [[0, 1, 0], [1, 1, 0], [2, 1, 0]],
                                 [[0, 0, 0], [1, 0, 0], [2, 0, 0]]],
                                [[[0, 2, 1], [1, 2, 1], [2, 2, 1]], [[0, 1, 1], [1, 1, 1], [2, 1, 1]],
                                 [[0, 0, 1], [1, 0, 1], [2, 0, 1]]],
                                [[[0, 2, 2], [1, 2, 2], [2, 2, 2]], [[0, 1, 2], [1, 1, 2], [2, 1, 2]],
                                 [[0, 0, 2], [1, 0, 2], [2, 0, 2]]]])

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
    assert grid.geometry_defined_for_all_cells_cached
    assert_array_almost_equal(grid.points_cached, expected_points)


def test_grid_from_cp_simple_inactive(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    corns = simple_grid_corns()
    active_mask = np.ones((2, 2, 2))
    active_mask[1, 1, 1] = 0
    expected_inactive = ~active_mask.astype(bool)

    # Act
    grid = rqi.grid_from_cp(model,
                            cp_array = corns,
                            crs_uuid = crs.uuid,
                            ijk_handedness = None,
                            active_mask = active_mask)

    # Assert
    assert grid is not None
    assert_array_almost_equal(grid.extent_kji, (2, 2, 2))
    assert grid.grid_is_right_handed
    assert grid.k_gaps is None
    assert grid.k_direction_is_down
    assert grid.pillar_shape == 'curved'
    assert grid.crs_uuid == crs.uuid
    assert grid.geometry_defined_for_all_cells_cached
    assert grid.array_cell_geometry_is_defined is None
    assert_array_almost_equal(grid.inactive, expected_inactive)


def test_grid_from_cp_simple_nogeom(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    corns = simple_grid_corns(undefined = True)
    expected_bool = np.ones((2, 2, 2))
    expected_bool[0, 0, 0] = 0

    # Act
    grid = rqi.grid_from_cp(model,
                            cp_array = corns,
                            crs_uuid = crs.uuid,
                            ijk_handedness = None,
                            geometry_defined_everywhere = False)

    # Assert
    assert grid is not None
    assert_array_almost_equal(grid.extent_kji, (2, 2, 2))
    assert grid.grid_is_right_handed
    assert grid.k_gaps is None
    assert grid.k_direction_is_down
    assert grid.pillar_shape == 'curved'
    assert grid.crs_uuid == crs.uuid
    assert not grid.geometry_defined_for_all_cells_cached
    assert_array_almost_equal(grid.array_cell_geometry_is_defined, expected_bool)


def test_grid_from_cp_simple_inactive_nogeom(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    corns = simple_grid_corns(undefined = True)
    expected_bool = np.ones((2, 2, 2))
    expected_bool[0, 0, 0] = 0
    active_mask = np.ones((2, 2, 2))
    active_mask[1, 1, 1] = 0
    expected_active = active_mask.copy()
    expected_active[0, 0, 0] = 0
    expected_inactive = ~expected_active.astype(bool)

    # Act
    grid = rqi.grid_from_cp(model,
                            cp_array = corns,
                            crs_uuid = crs.uuid,
                            ijk_handedness = None,
                            geometry_defined_everywhere = False,
                            active_mask = active_mask)

    # Assert
    assert grid is not None
    assert_array_almost_equal(grid.extent_kji, (2, 2, 2))
    assert grid.grid_is_right_handed
    assert grid.k_gaps is None
    assert grid.k_direction_is_down
    assert grid.pillar_shape == 'curved'
    assert grid.crs_uuid == crs.uuid
    assert not grid.geometry_defined_for_all_cells_cached
    assert_array_almost_equal(grid.array_cell_geometry_is_defined, expected_active)
    assert_array_almost_equal(grid.inactive, expected_inactive)


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


@pytest.mark.parametrize('surfaces,format,interp_and_feat,role,rqclass,newparts',
                         [(['Surface_zmap.dat'], 'zmap', False, 'map', 'surface', 2),
                          (['Surface_zmap.dat'], 'zmap', True, 'map', 'surface', 4),
                          (['Surface_roxartext.txt'], 'roxar', False, 'map', 'surface', 2),
                          (['Surface_roxartext.txt'], 'roxar', True, 'map', 'surface', 4),
                          (['Surface_roxartext.txt'], 'rms', False, 'map', 'surface', 2),
                          (['Surface_roxartext.txt'], 'rms', True, 'map', 'surface', 4),
                          (['Surface_tsurf.txt'], 'GOCAD-Tsurf', False, 'map', 'surface', 2),
                          (['Surface_tsurf.txt'], 'GOCAD-Tsurf', True, 'map', 'surface', 4),
                          (['Surface_zmap.dat', 'Surface_zmap.dat'], 'zmap', False, 'map', 'surface', 3),
                          (['Surface_zmap.dat', 'Surface_zmap.dat'], 'zmap', True, 'map', 'surface', 6),
                          (['Surface_zmap.dat'], 'zmap', False, 'pick', 'surface', 2),
                          (['Surface_zmap.dat'], 'zmap', False, 'pick', 'TriangulatedSet', 2)])
# (['Surface_zmap.dat'],'zmap',False,'pick','Grid2d',2)]) # TODO: Fails due to bug, Mesh.create_xml does not have an argument for crs_uuid
def test_add_surfaces(example_model_and_crs, test_data_path, surfaces, format, rqclass, interp_and_feat, role,
                      newparts):
    model, crs = example_model_and_crs
    model.store_epc()

    surface_paths = [os.path.join(test_data_path, surf) for surf in surfaces]

    rqi.add_surfaces(epc_file = model.epc_file,
                     surface_file_format = format,
                     surface_file_list = surface_paths,
                     surface_role = role,
                     rq_class = rqclass,
                     make_horizon_interpretations_and_features = interp_and_feat)

    model = rq.Model(model.epc_file)
    assert len(model.parts()) == newparts
    if rqclass in ['surface', 'TriangulatedSet']:
        assert len(model.parts_list_of_type('obj_TriangulatedSetRepresentation')) == len(surfaces)
    else:
        assert len(model.parts_list_of_type('obj_Grid2dRepresentation')) == len(surfaces)

    if interp_and_feat:
        assert len(model.parts_list_of_type('obj_HorizonInterpretation')) == len(surfaces)


def test_add_ab_properties(example_model_with_properties, test_data_path):
    # Arrange
    model = example_model_with_properties
    ab_facies = os.path.join(test_data_path, 'facies.ib')
    ab_ntg = os.path.join(test_data_path, 'ntg_355.db')

    ab_list = [(ab_facies, 'facies_ab', 'discrete', None, None, None, None, None, True, None),
               (ab_ntg, 'ntg_ab', 'net to gross ratio', None, None, None, None, None, False, None)]

    rqi.add_ab_properties(model.epc_file, ab_property_list = ab_list)

    reload = rq.Model(model.epc_file)
    pc = reload.grid().property_collection

    property_names = [pc.citation_title_for_part(part) for part in pc.parts()]
    assert 'facies_ab' in property_names
    assert 'ntg_ab' in property_names
    for part in pc.parts():
        if pc.citation_title_for_part(part) == 'facies_ab':
            assert not pc.continuous_for_part(part)
            farray = pc.cached_part_array_ref(part)
            assert np.min(farray) == 0
            assert np.max(farray) == 5
        elif pc.citation_title_for_part(part) == 'ntg_ab':
            assert pc.continuous_for_part(part)
            ntgarray = pc.cached_part_array_ref(part)
            assert np.min(ntgarray) > 0.4
            assert np.max(ntgarray) < 0.7
