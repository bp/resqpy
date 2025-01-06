import math as maths
import numpy as np
from numpy.testing import assert_array_almost_equal
import os
import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.surface as rqs
import resqpy.olio.uuid as bu

import pytest


@pytest.fixture
def triple_patch_model_crs_surface(example_model_and_crs):
    """Example resqpy model with a 3 patch surface"""

    p0 = np.array([(-10.0, -10.0, 0.0), (0.0, -10.0, 0.1), (10.0, -10.0, -0.1), (-10.0, 0.0, 0.2), (0.0, 0.0, -0.2),
                   (-10.0, 10.0, 0.3)],
                  dtype = float)
    p1 = -p0
    p2 = p0 + np.array((30.0, 0.0, 0.5), dtype = float)
    t = np.array([(0, 1, 3), (1, 4, 3), (1, 2, 4), (3, 4, 5)], dtype = int)
    t_and_p_list = [(t, p0), (t, p1), (t, p2)]

    model, crs = example_model_and_crs
    surf = rqs.Surface(model, title = 'triple patch', crs_uuid = crs.uuid)
    surf.set_multi_patch_from_triangles_and_points(t_and_p_list)
    surf.write_hdf5()
    surf.create_xml()
    model.store_epc
    return (model, crs, surf)


def test_tripatch_set_to_triangle(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    corners = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]])

    # Act
    tripatch = rqs.TriangulatedPatch(parent_model = model)
    tripatch.set_to_triangle(corners)

    # Assert
    assert tripatch is not None
    assert_array_almost_equal(tripatch.triangles, np.array([[0, 1, 2]]))
    assert_array_almost_equal(tripatch.points, corners)


def test_tripatch_verticalscale(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    corners = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]])

    # Act
    tripatch = rqs.TriangulatedPatch(parent_model = model)
    tripatch.set_to_triangle(corners)

    # Assert
    assert tripatch is not None
    # Scale without a reference depth initially
    tripatch.vertical_rescale_points(0, 10)
    assert_array_almost_equal(tripatch.points[:, 2], np.array([0, 10, 10]))
    # Scale with a reference depth
    tripatch.vertical_rescale_points(5, 2)
    assert_array_almost_equal(tripatch.points[:, 2], np.array([-5, 15, 15]))


def test_tripatch_set_from_irregularmesh(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    mesh_xyz = np.array([[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 1]]])

    # Act
    tripatch = rqs.TriangulatedPatch(parent_model = model)
    tripatch.set_from_irregular_mesh(mesh_xyz = mesh_xyz, quad_triangles = False)

    # Assert
    assert tripatch is not None
    assert tripatch.node_count == 4
    assert tripatch.points.shape == (4, 3)
    assert_array_almost_equal(tripatch.points[0], mesh_xyz[0, 0])
    assert_array_almost_equal(tripatch.points[3], mesh_xyz[1, 1])
    assert_array_almost_equal(tripatch.triangles, np.array([[0, 1, 2], [3, 1, 2]]))


def test_tripatch_columnfromindex(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    mesh_xyz = np.array([[[0, 0, 1], [1, 0, 1], [2, 0, 2]], [[0, 1, 1], [1, 1, 1], [2, 1, 1]]])

    # Act
    tripatch = rqs.TriangulatedPatch(parent_model = model)
    tripatch.set_from_irregular_mesh(mesh_xyz = mesh_xyz, quad_triangles = False)
    assert tripatch is not None
    j0, i0 = tripatch.column_from_triangle_index(0)
    j1, i1 = tripatch.column_from_triangle_index(np.array([0, 3]))

    # Assert
    assert j0 == 0
    assert i0 == 0
    assert_array_almost_equal(j1, np.array([0, 0]))
    assert_array_almost_equal(i1, np.array([0, 1]))


def test_tripatch_columnfromindex_quad(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    mesh_xyz = np.array([[[0, 0, 1], [1, 0, 1], [2, 0, 2]], [[0, 1, 1], [1, 1, 1], [2, 1, 1]]])

    # Act
    tripatch = rqs.TriangulatedPatch(parent_model = model)
    tripatch.set_from_irregular_mesh(mesh_xyz = mesh_xyz, quad_triangles = True)
    assert tripatch is not None
    j0, i0 = tripatch.column_from_triangle_index(0)
    j1, i1 = tripatch.column_from_triangle_index(np.array([0, 7]))
    j2, i2 = tripatch.column_from_triangle_index(8)
    j3, i3 = tripatch.column_from_triangle_index(np.array([0, 8]))

    # Assert
    assert j0 == 0
    assert i0 == 0
    assert_array_almost_equal(j1, np.array([0, 0]))
    assert_array_almost_equal(i1, np.array([0, 1]))
    assert j2 is None
    assert i2 is None
    assert j3 is None
    assert i3 is None


def test_tripatch_set_from_irregularmesh_quad(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    mesh_xyz = np.array([[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 1]]])

    # Act
    tripatch = rqs.TriangulatedPatch(parent_model = model)
    tripatch.set_from_irregular_mesh(mesh_xyz = mesh_xyz, quad_triangles = True)

    # Assert
    assert tripatch is not None
    assert tripatch.node_count == 5
    assert tripatch.points.shape == (5, 3)
    assert_array_almost_equal(tripatch.points[0], mesh_xyz[0, 0])
    assert_array_almost_equal(tripatch.points[3], mesh_xyz[1, 1])
    assert_array_almost_equal(tripatch.triangles, np.array([[4, 0, 1], [4, 1, 3], [4, 3, 2], [4, 2, 0]]))


def test_tripatch_set_from_sparse(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    mesh_xyz = np.array([[[0, 0, 1], [1, 0, 1], [2, 0, 2]], [[0, 1, 1], [1, 1, np.nan], [2, 1, 1]],
                         [[0, 2, 1], [1, 2, 1], [2, 2, 1]]])

    # Act
    tripatch = rqs.TriangulatedPatch(parent_model = model)
    tripatch.set_from_sparse_mesh(mesh_xyz = mesh_xyz)

    # Assert
    assert tripatch is not None
    assert tripatch.node_count == 8
    assert tripatch.points.shape == (8, 3)
    assert_array_almost_equal(tripatch.points[0], mesh_xyz[0, 0])
    assert_array_almost_equal(tripatch.points[7], mesh_xyz[2, 2])
    assert_array_almost_equal(tripatch.triangles[0], np.array([1, 3, 0]))
    assert_array_almost_equal(tripatch.triangles[3], np.array([4, 6, 7]))


def test_tripatch_set_from_torn(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    mesh_xyz = np.array([[[[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, np.nan]]],
                          [[[1, 0, 1], [2, 0, 1]], [[1, 1, np.nan], [2, 1, 1]]]]])

    # Act
    tripatch = rqs.TriangulatedPatch(parent_model = model)
    tripatch.set_from_torn_mesh(mesh_xyz = mesh_xyz)

    # Assert
    assert tripatch is not None
    assert tripatch.node_count == 8
    assert tripatch.points.shape == (8, 3)
    assert tripatch.ni == 2
    assert_array_almost_equal(tripatch.points[0], mesh_xyz[0, 0, 0, 0])
    assert_array_almost_equal(tripatch.points[2], mesh_xyz[0, 0, 1, 0])
    assert_array_almost_equal(tripatch.triangles, np.array([[0, 1, 2], [3, 1, 2], [4, 5, 6], [7, 5, 6]]))


def test_tripatch_set_from_torn_quad(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    mesh_xyz = np.array([[[[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, np.nan]]],
                          [[[1, 0, 1], [2, 0, 1]], [[1, 1, np.nan], [2, 1, 1]]]]])

    # Act
    tripatch = rqs.TriangulatedPatch(parent_model = model)
    tripatch.set_from_torn_mesh(mesh_xyz = mesh_xyz, quad_triangles = True)

    # Assert
    assert tripatch is not None
    assert tripatch.node_count == 10
    assert tripatch.points.shape == (10, 3)
    assert tripatch.ni == 2
    assert_array_almost_equal(tripatch.points[0], mesh_xyz[0, 0, 0, 0])
    assert_array_almost_equal(tripatch.points[2], mesh_xyz[0, 0, 1, 0])
    assert_array_almost_equal(
        tripatch.triangles,
        np.array([[8, 0, 1], [8, 1, 3], [8, 3, 2], [8, 2, 0], [9, 4, 5], [9, 5, 7], [9, 7, 6], [9, 6, 4]]))


def test_tripatch_set_cellface_corp(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    cp = np.array([[[[0, 0, 0], [0, 1, 0]], [[1, 1, 0], [1, 0, 0]]], [[[0, 0, 1], [0, 1, 1]], [[1, 1, 1], [1, 0, 1]]]])

    # Act
    tripatch = rqs.TriangulatedPatch(parent_model = model)
    tripatch.set_to_cell_faces_from_corner_points(cp = cp)

    # Assert
    assert tripatch is not None
    assert tripatch.node_count == 14
    assert tripatch.points.shape == (14, 3)
    assert_array_almost_equal(tripatch.points[0], cp[0, 0, 0])
    assert_array_almost_equal(tripatch.points[2], cp[0, 1, 0])
    assert_array_almost_equal(tripatch.triangles[0], np.array([8, 0, 1]))
    assert_array_almost_equal(tripatch.triangles[-1], np.array([13, 3, 7]))


def test_tripatch_set_cellface_corp_quadfalse(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    cp = np.array([[[[0, 0, 0], [0, 1, 0]], [[1, 1, 0], [1, 0, 0]]], [[[0, 0, 1], [0, 1, 1]], [[1, 1, 1], [1, 0, 1]]]])

    # Act
    tripatch = rqs.TriangulatedPatch(parent_model = model)
    tripatch.set_to_cell_faces_from_corner_points(cp = cp, quad_triangles = False)

    # Assert
    assert tripatch is not None
    assert tripatch.node_count == 8
    assert tripatch.points.shape == (8, 3)
    assert_array_almost_equal(tripatch.points[0], cp[0, 0, 0])
    assert_array_almost_equal(tripatch.points[2], cp[0, 1, 0])
    assert_array_almost_equal(tripatch.triangles[0], np.array([0, 3, 1]))
    assert_array_almost_equal(tripatch.triangles[-1], np.array([7, 1, 3]))


def test_set_multi_patch_from_triangles_and_points(triple_patch_model_crs_surface):

    def check_surface(surf):
        assert surf is not None
        assert surf.number_of_patches() == 3
        previous_t = None
        for patch in range(3):
            assert surf.triangle_count(patch) == 4
            assert surf.node_count(patch) == 6
            t, p = surf.triangles_and_points(patch)
            assert t is not None and p is not None
            assert t.shape == (4, 3)
            assert p.shape == (6, 3)
            if patch:
                assert np.all(t == previous_t)
            else:
                previous_t = t
            ct, cp = surf.triangles_and_points(patch, copy = True)
            assert np.all(ct == t)
            assert_array_almost_equal(cp, p)
            assert id(ct) != id(t) and id(cp) != id(p)

    model, crs, surf = triple_patch_model_crs_surface
    check_surface(surf)
    # reload and check again
    surf = rqs.Surface(model, uuid = surf.uuid)
    assert surf is not None
    check_surface(surf)


def test_patch_edges(triple_patch_model_crs_surface):

    def check_edges(surf):
        expected_edges = np.array([(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 5), (4, 5)], dtype = int)
        expected_counts = np.array((1, 1, 1, 2, 2, 1, 2, 1, 1), dtype = int)
        expected_lengths = np.full((4, 3), 10.0, dtype = float)
        root_2 = maths.sqrt(2.0)
        expected_lengths[0, 1] *= root_2
        expected_lengths[1, 2] *= root_2
        expected_lengths[2, 1] *= root_2
        expected_lengths[3, 1] *= root_2
        assert surf.number_of_patches() == 3
        for patch in range(3):
            edges = surf.distinct_edges(patch)
            assert edges is not None
            assert edges.shape == (9, 2)
            assert np.all(edges == expected_edges)
            edges, counts = surf.distinct_edges_and_counts(patch)
            assert np.all(edges == expected_edges)
            assert np.all(counts == expected_counts)
            edge_lengths = surf.edge_lengths(patch = patch)
            assert_array_almost_equal(edge_lengths, expected_lengths, decimal = 1)  # low tolerance due to z offsets
            edge_lengths = surf.edge_lengths(required_uom = 'ft', patch = patch)
            assert_array_almost_equal(edge_lengths, expected_lengths / 0.3048, decimal = 1)

    model, crs, surf = triple_patch_model_crs_surface
    check_edges(surf)
    # reload and check again
    surf = rqs.Surface(model, uuid = surf.uuid)
    assert surf is not None
    check_edges(surf)


def test_patch_line_intersection(triple_patch_model_crs_surface):
    model, crs, surf = triple_patch_model_crs_surface
    xyz = surf.line_intersection((15.0, 5.0, -2.0), (0.0, 0.0, 1.0), line_segment = False, patch = None)
    assert xyz is None
    xyz = surf.line_intersection((15.0, 5.0, -2.0), (0.0, 0.0, 1.0), line_segment = True, patch = None)
    assert xyz is None
    xyz = surf.line_intersection((5.0, 5.0, -2.0), (0.0, 0.0, 1.0), line_segment = False, patch = None)
    assert xyz is not None
    assert_array_almost_equal(xyz, (5.0, 5.0, -0.15))
    xyz = surf.line_intersection((5.0, 5.0, -2.0), (0.0, 0.0, 1.0), line_segment = True, patch = None)
    assert xyz is None
    xyz = surf.line_intersection((5.0, 5.0, -2.0), (0.0, 0.0, 1.0), line_segment = False, patch = 0)
    assert xyz is None
    xyz = surf.line_intersection((5.0, 5.0, -2.0), (0.0, 0.0, 1.0), line_segment = False, patch = 1)
    assert xyz is not None
    assert_array_almost_equal(xyz, (5.0, 5.0, -0.15))
    xyz = surf.line_intersection((5.0, 5.0, -2.0), (0.0, 0.0, 1.0), line_segment = False, patch = 2)
    assert xyz is None
    # try a line at the join of two patches; might be prone to fail due to precision
    xyz = surf.line_intersection((5.0, -5.0, -2.0), (0.0, 0.0, 1.0), line_segment = False, patch = None)
    assert xyz is not None
    assert_array_almost_equal(xyz, (5.0, -5.0, -0.15))  # z result arbitrarily from patch 0
    xyz = surf.line_intersection((5.0, -5.0, -2.0), (0.0, 0.0, 1.0), line_segment = False, patch = 0)
    assert xyz is not None
    assert_array_almost_equal(xyz, (5.0, -5.0, -0.15))
    xyz = surf.line_intersection((5.0, -5.0, -2.0), (0.0, 0.0, 1.0), line_segment = False, patch = 1)
    assert xyz is not None
    assert_array_almost_equal(xyz, (5.0, -5.0, -0.05))
    xyz = surf.line_intersection((5.0, -5.0, -2.0), (0.0, 0.0, 1.0), line_segment = False, patch = 2)
    assert xyz is None


def test_surface_from_list_of_patches_of_triangles_and_points(example_model_and_crs):
    p0 = np.array([(-10.0, -10.0, 0.0), (0.0, -10.0, 0.1), (10.0, -10.0, -0.1), (-10.0, 0.0, 0.2), (0.0, 0.0, -0.2),
                   (-10.0, 10.0, 0.3)],
                  dtype = float)
    p1 = -p0
    p2 = p0 + np.array((30.0, 0.0, 0.5), dtype = float)
    t = np.array([(0, 1, 3), (1, 4, 3), (1, 2, 4), (3, 4, 5)], dtype = int)
    t_and_p_list = [(t, p0), (t, p1), (t, p2)]
    model, crs = example_model_and_crs
    cm_surf = rqs.Surface.from_list_of_patches_of_triangles_and_points(model, t_and_p_list, 'class method', crs.uuid)
    cm_surf.write_hdf5()
    cm_surf.create_xml()
    surf = rqs.Surface(model, title = 'triple patch', crs_uuid = crs.uuid)
    surf.set_multi_patch_from_triangles_and_points(t_and_p_list)
    surf.write_hdf5()
    surf.create_xml()
    model.store_epc()
    model = rq.Model(model.epc_file)
    cm_surf = rqs.Surface(model, uuid = cm_surf.uuid)
    surf = rqs.Surface(model, uuid = surf.uuid)
    cm_t, cm_p = cm_surf.triangles_and_points()
    t, p = surf.triangles_and_points()
    assert np.all(t == cm_t)
    assert_array_almost_equal(p, cm_p)
    p_i = cm_surf.patch_indices_for_triangle_indices(np.arange(12, dtype = int))
    assert np.all(p_i == (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2))
