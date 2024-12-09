import math as maths
import numpy as np
from numpy.testing import assert_array_almost_equal
import os
import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.grid
import resqpy.grid_surface as rqgs
import resqpy.lines as rql
import resqpy.organize as rqo
import resqpy.surface as rqs
import resqpy.olio.uuid as bu
import resqpy.olio.triangulation as tri

import pytest

# Unit tests for surface classes


def test_surface(tmp_model):
    # Set up a Surface
    title = 'Mountbatten'
    model = tmp_model
    surf = rqs.Surface(parent_model = model, title = title)
    surf.create_xml()

    # Add a interpretation
    assert surf.represented_interpretation_root is None
    surf.create_interpretation_and_feature(kind = 'fault')
    assert surf.represented_interpretation_root is not None

    # Check fault can be loaded in again
    model.store_epc()
    fault_interp = rqo.FaultInterpretation(model, uuid = surf.represented_interpretation_uuid)
    fault_feature = rqo.TectonicBoundaryFeature(model, uuid = fault_interp.tectonic_boundary_feature.uuid)

    # Check title matches expected title
    assert fault_feature.feature_name == title


def test_faces_for_surface(tmp_model):
    crs = resqpy.crs.Crs(tmp_model)
    crs.create_xml()
    grid = resqpy.grid.RegularGrid(tmp_model, extent_kji = (3, 3, 3), crs_uuid = crs.uuid, set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    # todo: create sloping planar surface
    # call find faces for each of 3 different methods
    points = np.zeros((2, 2, 3))
    points[1, :, 1] = 3.0
    points[:, 1, 0] = 3.0
    points[:, 1, 2] = 3.0
    points[:, :, 2] += 0.25
    triangles = np.zeros((2, 3), dtype = int)
    triangles[0] = (0, 1, 2)
    triangles[1] = (3, 1, 2)
    surf = rqs.Surface(tmp_model, crs_uuid = crs.uuid)
    surf.set_from_triangles_and_points(triangles, points.reshape((-1, 3)))
    assert surf is not None
    assert surf.node_count() == 4
    assert surf.triangle_count() == 2
    assert_array_almost_equal(surf.normal(), (-0.707107, 0.0, 0.707107))
    surf.write_hdf5()
    surf.create_xml()
    surf = rqs.Surface(tmp_model, uuid = surf.uuid)
    assert surf is not None
    assert surf.node_count() == 4
    assert surf.triangle_count() == 2
    assert surf.normal_vector is not None
    assert_array_almost_equal(surf.normal_vector, (-0.707107, 0.0, 0.707107))
    for mode in ['staffa', 'regular', 'auto']:
        gcs = rqgs.find_faces_to_represent_surface(grid, surf, name = mode, mode = mode)
        assert gcs is not None
        assert gcs.count == 12
        cip = set([tuple(pair) for pair in gcs.cell_index_pairs])
        expected_cip = grid.natural_cell_indices(
            np.array([[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 0]], [[0, 2, 0], [1, 2, 0]], [[1, 0, 0], [1, 0, 1]],
                      [[1, 1, 0], [1, 1, 1]], [[1, 2, 0], [1, 2, 1]], [[1, 0, 1], [2, 0, 1]], [[1, 1, 1], [2, 1, 1]],
                      [[1, 2, 1], [2, 2, 1]], [[2, 0, 1], [2, 0, 2]], [[2, 1, 1], [2, 1, 2]], [[2, 2, 1], [2, 2, 2]]],
                     dtype = int))
        e_cip = set([tuple(pair) for pair in expected_cip])
        assert cip == e_cip  # note: this assumes lower cell index is first, which happens to be true
        # todo: check face indices
        gcs.write_hdf5()
        gcs.create_xml()
        assert bu.matching_uuids(
            tmp_model.uuid(obj_type = 'GridConnectionSetRepresentation', multiple_handling = 'newest'), gcs.uuid)


def test_delaunay_triangulation(example_model_and_crs):
    model, crs = example_model_and_crs

    # number of random points to use
    n = 20

    # create a set of random points
    x = np.random.random(n) * 1000.0
    y = np.random.random(n) * 1000.0
    z = np.random.random(n)  #  note: triangulation does not use z values
    p = np.stack((x, y, z), axis = -1)

    # make a PointSet object
    ps = rqs.PointSet(model, crs_uuid = crs.uuid, points_array = p, title = 'random points in square')

    # make another PointSet as random points within a closed polyline
    vertices = np.array(
        ((50.0, 99.0, 13.0), (85.0, 60.0, 17.5), (62.7, 11.0, 10.0), (33.3, 15.3, 19.2), (12.8, 57.8, 15.0)))
    polygon = rql.Polyline(model, set_crs = crs.uuid, is_closed = True, set_coord = vertices, title = 'the pentagon')
    polygon.write_hdf5()
    polygon.create_xml()
    ps2 = rqs.PointSet(model,
                       crs_uuid = crs.uuid,
                       polyline = polygon,
                       random_point_count = n,
                       title = 'random points in polygon')

    # process the point sets into triangulated surfaces
    for point_set in (ps, ps2):
        point_set.write_hdf5()
        point_set.create_xml()
        surf = rqs.Surface(model, point_set = point_set, title = 'surface from ' + str(point_set.title))
        assert surf is not None
        surf.write_hdf5()
        surf.create_xml()
        # check that coordinate range of points looks okay
        triangles, points = surf.triangles_and_points()
        assert len(points) == n
        original_points = point_set.full_array_ref()
        assert_array_almost_equal(np.nanmin(original_points, axis = 0), np.nanmin(points, axis = 0))
        assert_array_almost_equal(np.nanmax(original_points, axis = 0), np.nanmax(points, axis = 0))


def test_surface_from_point_set_with_flange_extension(example_model_and_crs):
    model, crs = example_model_and_crs

    # number of random points to use
    n = 20

    # create a set of random points
    x = np.random.random(n) * 1000.0 - 500.0
    y = np.random.random(n) * 1000.0 - 500.0
    z = np.random.random(n)  #  note: triangulation does not use z values
    p = np.stack((x, y, z), axis = -1)
    centre = np.mean(p, axis = 0)

    # make a PointSet object
    ps = rqs.PointSet(model, crs_uuid = crs.uuid, points_array = p, title = 'random points in square')
    ps.write_hdf5()
    ps.create_xml()

    # try out flange extension
    surf = rqs.Surface(model, crs_uuid = ps.crs_uuid, title = 'surface from ' + str(ps.title))
    assert surf is not None
    surf.set_from_point_set(ps,
                            reorient = True,
                            reorient_max_dip = None,
                            extend_with_flange = True,
                            flange_point_count = 15,
                            flange_radial_factor = 10.0,
                            flange_radial_distance = None,
                            flange_inner_ring = False,
                            saucer_parameter = None,
                            make_clockwise = False)
    surf.write_hdf5()
    surf.create_xml()
    assert surf.node_count() == 35

    # flange extension with simple saucer
    surf = rqs.Surface(model, crs_uuid = ps.crs_uuid, title = 'surface from ' + str(ps.title))
    assert surf is not None
    surf.set_from_point_set(ps,
                            reorient = True,
                            reorient_max_dip = None,
                            extend_with_flange = True,
                            flange_point_count = 12,
                            flange_radial_factor = 5.0,
                            flange_radial_distance = None,
                            flange_inner_ring = False,
                            saucer_parameter = 45.0,
                            make_clockwise = False)
    surf.write_hdf5()
    surf.create_xml()
    assert surf.node_count() == 32

    # flange extension with explicit radial distance, inner ring, and simple saucer
    surf = rqs.Surface(model, crs_uuid = ps.crs_uuid, title = 'surface from ' + str(ps.title))
    assert surf is not None
    surf.set_from_point_set(ps,
                            reorient = True,
                            reorient_max_dip = None,
                            extend_with_flange = True,
                            flange_point_count = 12,
                            flange_radial_factor = 2.0,
                            flange_radial_distance = 2000.0,
                            flange_inner_ring = True,
                            saucer_parameter = -60.0,
                            make_clockwise = False)
    surf.write_hdf5()
    surf.create_xml()
    assert surf.node_count() == 56
    _, p = surf.triangles_and_points()
    min_p = np.min(p, axis = 0)
    max_p = np.max(p, axis = 0)
    assert -2010.0 <= min_p[0] - centre[0] < -1900.0
    assert -2010.0 <= min_p[1] - centre[1] < -1900.0
    assert 1900.0 < max_p[0] - centre[0] <= 2010.0
    assert 1900.0 < max_p[1] - centre[1] <= 2010.0
    assert 0.0 <= min_p[2] <= 1.0
    assert 3464.0 < max_p[2] < 3465.5


def test_surface_from_point_set_with_nan_removal(example_model_and_crs):
    model, crs = example_model_and_crs

    # number of random points to use
    n = 20

    # create a set of random points
    x = np.random.random(n) * 1000.0 - 500.0
    y = np.random.random(n) * 1000.0 - 500.0
    z = np.random.random(n)  #  note: triangulation does not use z values
    p = np.stack((x, y, z), axis = -1)

    # set a few nans
    p[5, 1] = np.nan
    p[7, :] = np.nan
    p[13, 0] = np.nan

    # make a PointSet object
    ps = rqs.PointSet(model, crs_uuid = crs.uuid, points_array = p, title = 'random points in square')
    ps.write_hdf5()
    ps.create_xml()

    # try out flange extension
    surf = rqs.Surface(model, crs_uuid = ps.crs_uuid, title = 'surface from ' + str(ps.title))
    assert surf is not None
    surf.set_from_point_set(ps, reorient = True, reorient_max_dip = None, extend_with_flange = False)
    surf.write_hdf5()
    surf.create_xml()
    assert surf.node_count() == 17


def test_surface_change_crs(example_model_and_crs):
    model, crs = example_model_and_crs

    # number of random points to use
    n = 20

    # create a set of random points
    x = np.random.random(n) * 1000.0 + 500.0
    y = np.random.random(n) * 1000.0 + 2500.0
    z = np.random.random(n) * 100.0 + 1200.0
    p = np.stack((x, y, z), axis = -1)

    # make a PointSet object
    ps = rqs.PointSet(model, crs_uuid = crs.uuid, points_array = p, title = 'random points in square')
    ps.write_hdf5()
    ps.create_xml()

    # make a new crs
    crs_ft_xy = rqc.Crs(model, xy_units = 'ft', z_units = 'm')
    crs_ft_xy.create_xml()

    # change crs
    surf = rqs.Surface(model, crs_uuid = ps.crs_uuid, title = 'surface from ' + str(ps.title))
    assert surf is not None
    surf.set_from_point_set(ps, reorient = False, extend_with_flange = False)
    _, sp = surf.triangles_and_points()
    assert_array_almost_equal(sp, p)
    old_uuid = surf.uuid
    surf.change_crs(crs_ft_xy)
    surf.write_hdf5()
    surf.create_xml()
    assert not bu.matching_uuids(surf.uuid, old_uuid)
    surf = rqs.Surface(model, uuid = surf.uuid)
    _, sp = surf.triangles_and_points()
    assert_array_almost_equal(sp[:, 2], p[:, 2])
    assert_array_almost_equal(sp[:, :2], p[:, :2] / 0.3048)
    assert surf.node_count() == 20


@pytest.mark.parametrize('mesh_file,mesh_format,firstval', [('Surface_roxartext.txt', 'rms', 0.4229),
                                                            ('Surface_roxartext.txt', 'roxar', 0.4229),
                                                            ('Surface_zmap.dat', 'zmap', 0.4648)])
def test_surface_from_mesh_file(example_model_and_crs, test_data_path, mesh_file, mesh_format, firstval):
    # Arrange
    model, crs = example_model_and_crs
    in_file = os.path.join(test_data_path, mesh_file)

    # Act
    surface = rqs.Surface(parent_model = model, mesh_file = in_file, mesh_format = mesh_format)

    # Assert
    assert surface is not None
    assert surface.patch_list[0].triangle_count == 12
    assert surface.triangle_count() == 12
    assert surface.patch_list[0].points[0][2] == firstval


def test_surface_from_tsurf_file(example_model_and_crs, test_data_path):
    # Arrange
    model, crs = example_model_and_crs
    in_file = os.path.join(test_data_path, 'Surface_tsurf.txt')

    # Act
    surface = rqs.Surface(parent_model = model, tsurf_file = in_file, title = 'horizon')
    surface.create_interpretation_and_feature(kind = 'horizon')

    # Assert
    assert surface is not None
    assert surface.represented_interpretation_root is not None
    assert surface.patch_list[0].triangle_count == 12
    assert surface.triangle_count() == 12
    assert surface.patch_list[0].points[0][2] == 0.4228516


def test_regular_mesh(example_model_and_crs):
    model, crs = example_model_and_crs

    #  number of points in mesh, origin spacing
    ni = 7
    nj = 5
    origin = (409000.0, 1605000.0, 0.0)
    di = dj = 50.0

    # create some random depths
    z = (np.random.random(ni * nj) * 20.0 + 1000.0).reshape((nj, ni))

    # make a regular mesh representation
    mesh = rqs.Mesh(model,
                    crs_uuid = crs.uuid,
                    mesh_flavour = 'reg&z',
                    ni = ni,
                    nj = nj,
                    origin = origin,
                    dxyz_dij = np.array([[di, 0.0, 0.0], [0.0, dj, 0.0]]),
                    z_values = z,
                    title = 'random mesh',
                    originator = 'Andy',
                    extra_metadata = {'testing mode': 'automated'})
    assert mesh is not None
    mesh.write_hdf5()
    mesh.create_xml()
    mesh_uuid = mesh.uuid

    # fully write model to disc
    model.store_epc()
    epc = model.epc_file

    # re-open model and check the mesh object is there
    model = rq.Model(epc)
    assert bu.matching_uuids(model.uuid(obj_type = 'Grid2dRepresentation', title = 'random mesh'), mesh_uuid)

    # establish a resqpy Mesh from the object in the RESQML dataset
    persistent_mesh = rqs.Mesh(model, uuid = mesh_uuid)

    # check some of the metadata
    assert persistent_mesh.ni == ni and persistent_mesh.nj == nj
    assert persistent_mesh.flavour == 'reg&z'
    assert_array_almost_equal(np.array(persistent_mesh.regular_origin), np.array(origin))
    assert_array_almost_equal(np.array(persistent_mesh.regular_dxyz_dij), np.array([[di, 0.0, 0.0], [0.0, dj, 0.0]]))

    # check a fully expanded version of the points
    assert_array_almost_equal(persistent_mesh.full_array_ref(), mesh.full_array_ref())

    # check that we can build a Surface from the Mesh
    surf = persistent_mesh.surface(quad_triangles = True)
    assert surf is not None
    assert surf.node_count() == 7 * 5 + 6 * 4  # quad triangles introduce extra nodes

    # do some basic checks that the surface looks consistent with the mesh
    t, p = surf.triangles_and_points()
    assert len(p) == (ni * nj) + ((ni - 1) * (nj - 1))  # quad triangles mode introduces the extra points
    assert len(t) == 4 * (ni - 1) * (nj - 1)
    assert_array_almost_equal(np.min(p, axis = 0), np.min(persistent_mesh.full_array_ref().reshape(-1, 3), axis = 0))
    assert_array_almost_equal(np.max(p, axis = 0), np.max(persistent_mesh.full_array_ref().reshape(-1, 3), axis = 0))
    assert len(surf.distinct_edges()) == 6 * (ni - 1) * (nj - 1) + (ni - 1) + (nj - 1)


# @pytest.mark.skip(reason = "Bug in Mesh for ref&z flavou needs fixing first")
def test_refandz_mesh(example_model_and_crs):
    model, crs = example_model_and_crs

    # number of points in mesh, origin spacing
    ni = 7
    nj = 5
    origin = (409000.0, 1605000.0, 0.0)
    di = dj = 50.0

    # create some random depths
    z = (np.random.random(ni * nj) * 20.0 + 1000.0).reshape((nj, ni))
    z_values = z * 2.5

    # make a regular mesh representation
    support = rqs.Mesh(model,
                       crs_uuid = crs.uuid,
                       mesh_flavour = 'regular',
                       ni = ni,
                       nj = nj,
                       origin = origin,
                       dxyz_dij = np.array([[di, 0.0, 0.0], [0.0, dj, 0.0]]),
                       z_values = z,
                       title = 'random regular mesh',
                       originator = 'Emma',
                       extra_metadata = {'testing mode': 'automated'})
    assert support is not None
    support.write_hdf5()
    support.create_xml()
    support_uuid = support.uuid

    # model.store_epc()
    # model = rq.Model(model.epc_file)

    refz = rqs.Mesh(model,
                    crs_uuid = crs.uuid,
                    z_values = z_values,
                    ni = ni,
                    nj = nj,
                    z_supporting_mesh_uuid = support_uuid,
                    title = 'random refz mesh',
                    originator = 'Emma',
                    extra_metadata = {'testing mode': 'automated'})

    assert refz is not None
    refz.write_hdf5()
    refz.create_xml()
    refz_uuid = refz.uuid

    # fully write model to disc
    model.store_epc()

    # re-open model and check the mesh object is there
    reload = rq.Model(model.epc_file)

    assert bu.matching_uuids(reload.uuid(obj_type = 'Grid2dRepresentation', title = 'random refz mesh'), refz_uuid)

    # establish a resqpy Mesh from the object in the RESQML dataset
    reload_refzmesh = rqs.Mesh(reload, uuid = refz_uuid)

    # check some of the metadata
    assert reload_refzmesh.ni == ni and reload_refzmesh.nj == nj
    assert reload_refzmesh.flavour == 'ref&z'
    assert bu.matching_uuids(reload_refzmesh.ref_uuid, support_uuid)

    # check a fully expanded version of the points
    assert_array_almost_equal(reload_refzmesh.full_array_ref(), refz.full_array_ref())


def test_explicit_mesh(example_model_and_crs):
    model, crs = example_model_and_crs

    # create some random x,y,z values
    x = (np.random.random(20) * 10000)
    y = (np.random.random(20) * 10000)
    z = (np.random.random(20) * 500 + 2000)
    # make a 20x20x3 array of values to test
    array = np.array([x, y, z]).T
    xyz_values = np.array(
        [np.multiply(array,
                     np.array([1 + val / 100, 1. + val / 100, 1 + val / 100]).T) for val in np.arange(0, 20)])

    # make an explicit mesh representation
    mesh = rqs.Mesh(model,
                    crs_uuid = crs.uuid,
                    mesh_flavour = 'explicit',
                    xyz_values = xyz_values,
                    title = 'random explicit mesh',
                    originator = 'Emma',
                    extra_metadata = {'testing mode': 'automated'})
    assert mesh is not None
    mesh.write_hdf5()
    mesh.create_xml()
    mesh_uuid = mesh.uuid

    # fully write model to disc
    model.store_epc()
    epc = model.epc_file

    # re-open model and check the mesh object is there
    model = rq.Model(epc)
    assert bu.matching_uuids(model.uuid(obj_type = 'Grid2dRepresentation', title = 'random explicit mesh'), mesh_uuid)

    # establish a resqpy Mesh from the object in the RESQML dataset
    persistent_mesh = rqs.Mesh(model, uuid = mesh_uuid)

    # check some of the metadata
    assert persistent_mesh.flavour == 'explicit'
    assert persistent_mesh.ni == len(x)
    assert persistent_mesh.nj == len(y)

    # check a fully expanded version of the points
    assert_array_almost_equal(persistent_mesh.full_array_ref(), mesh.full_array_ref())


@pytest.mark.parametrize('flavour,infile,filetype', [('explicit', 'Surface_roxartext.txt', 'roxar'),
                                                     ('explicit', 'Surface_roxartext.txt', 'rms'),
                                                     ('explicit', 'Surface_zmap.dat', 'zmap'),
                                                     ('regular', 'Surface_roxartext.txt', 'roxar'),
                                                     ('regular', 'Surface_roxartext.txt', 'rms'),
                                                     ('regular', 'Surface_zmap.dat', 'zmap'),
                                                     ('reg&z', 'Surface_roxartext.txt', 'roxar'),
                                                     ('reg&z', 'Surface_roxartext.txt', 'rms'),
                                                     ('reg&z', 'Surface_zmap.dat', 'zmap')])
def test_mesh_file(example_model_and_crs, test_data_path, flavour, infile, filetype):
    model, crs = example_model_and_crs

    mesh_file = os.path.join(test_data_path, infile)

    # make an explicit mesh representation
    mesh = rqs.Mesh(model,
                    crs_uuid = crs.uuid,
                    mesh_flavour = flavour,
                    mesh_file = mesh_file,
                    mesh_format = filetype,
                    title = 'mesh from file',
                    originator = 'Emma',
                    extra_metadata = {'testing mode': 'automated'})
    assert mesh is not None
    mesh.write_hdf5()
    mesh.create_xml()
    mesh_uuid = mesh.uuid

    # fully write model to disc
    model.store_epc()
    epc = model.epc_file

    # re-open model and check the mesh object is there
    model = rq.Model(epc)
    assert bu.matching_uuids(model.uuid(obj_type = 'Grid2dRepresentation', title = 'mesh from file'), mesh_uuid)

    # establish a resqpy Mesh from the object in the RESQML dataset
    persistent_mesh = rqs.Mesh(model, uuid = mesh_uuid)

    # check some of the metadata
    assert persistent_mesh.flavour == flavour

    # check a fully expanded version of the points
    if flavour != 'regular':
        assert_array_almost_equal(persistent_mesh.full_array_ref(), mesh.full_array_ref())


def test_pointset_from_array(example_model_and_crs):
    model, crs = example_model_and_crs

    # create some random x,y,z values
    x = (np.random.random(50) * 10000)
    y = (np.random.random(50) * 10000)
    z = (np.random.random(50) * 500 + 2000)

    # make a pointset representation
    points = rqs.PointSet(model,
                          crs_uuid = crs.uuid,
                          points_array = np.array([x, y, z]).T,
                          title = 'random points',
                          originator = 'Emma',
                          extra_metadata = {'testing mode': 'automated'})
    assert points is not None
    points.create_interpretation_and_feature(kind = 'horizon')
    points.write_hdf5()
    points.create_xml()
    points_uuid = points.uuid

    # fully write model to disc
    model.store_epc()
    epc = model.epc_file

    # re-open model and check the points object is there
    model = rq.Model(epc)
    assert bu.matching_uuids(model.uuid(obj_type = 'PointSetRepresentation', title = 'random points'), points_uuid)

    # establish a resqpy Pointset from the object in the RESQML dataset
    saved_points = rqs.PointSet(model, uuid = points_uuid)

    # check a fully expanded version of the points
    assert_array_almost_equal(saved_points.full_array_ref(), points.full_array_ref())
    assert saved_points.represented_interpretation_root is not None


def test_pointset_from_2d_array(example_model_and_crs):
    model, crs = example_model_and_crs

    # create some random x,y values
    x = (np.random.random(50) * 10000)
    y = (np.random.random(50) * 10000)

    # make a pointset representation
    points = rqs.PointSet(model,
                          crs_uuid = crs.uuid,
                          points_array = np.array([x, y]).T,
                          title = 'random 2d points',
                          originator = 'Emma',
                          extra_metadata = {'testing mode': 'automated'})
    assert points is not None
    points.write_hdf5()
    points.create_xml()
    points_uuid = points.uuid

    # fully write model to disc
    model.store_epc()
    epc = model.epc_file

    # re-open model and check the points object is there
    model = rq.Model(epc)
    assert bu.matching_uuids(model.uuid(obj_type = 'PointSetRepresentation', title = 'random 2d points'), points_uuid)

    # establish a resqpy Pointset from the object in the RESQML dataset
    saved_points = rqs.PointSet(model, uuid = points_uuid)

    # check a fully expanded version of the points
    assert_array_almost_equal(saved_points.full_array_ref(), points.full_array_ref())


def test_pointset_from_array_multipatch(example_model_and_crs):
    model, crs = example_model_and_crs

    # create some random x,y values
    x = (np.random.random(50) * 10000)
    y = (np.random.random(50) * 10000)

    x2 = (np.random.random(50) * 10000)
    y2 = (np.random.random(50) * 10000)

    # make a pointset representation
    points = rqs.PointSet(model,
                          crs_uuid = crs.uuid,
                          points_array = np.array([x, y]).T,
                          title = 'random 2d points',
                          originator = 'Emma',
                          extra_metadata = {'testing mode': 'automated'})
    assert points is not None
    points.add_patch(points_array = np.array([x2, y2]).T)
    points.write_hdf5()
    points.create_xml()
    points_uuid = points.uuid

    # fully write model to disc
    model.store_epc()
    epc = model.epc_file

    # re-open model and check the points object is there
    model = rq.Model(epc)
    assert bu.matching_uuids(model.uuid(obj_type = 'PointSetRepresentation', title = 'random 2d points'), points_uuid)

    # establish a resqpy Pointset from the object in the RESQML dataset
    saved_points = rqs.PointSet(model, uuid = points_uuid)

    # check a fully expanded version of the points
    assert_array_almost_equal(saved_points.full_array_ref(), points.full_array_ref())
    assert_array_almost_equal(saved_points.single_patch_array_ref(0), np.array([x, y, [0] * 50]).T)
    assert_array_almost_equal(saved_points.single_patch_array_ref(1), np.array([x2, y2, [0] * 50]).T)


def test_pointset_from_charisma(example_model_and_crs, test_data_path, tmp_path):
    # Set up a PointSet and save to resqml file
    model, crs = example_model_and_crs

    charisma_file = os.path.join(test_data_path, "Charisma_points.txt")
    points = rqs.PointSet(parent_model = model, charisma_file = str(charisma_file), crs_uuid = crs.uuid)
    points.write_hdf5()
    points.create_xml()
    model.store_epc()

    # Test reload from resqml
    model = rq.Model(epc_file = model.epc_file)
    reload = rqs.PointSet(parent_model = model, uuid = points.uuid)

    assert reload.title == str(charisma_file)

    coords = reload.full_array_ref()
    assert_array_almost_equal(coords[0], np.array([420691.19624, 6292314.22044, 2799.05591]))

    assert coords.shape == (15, 3), f'Expected shape (15,3), not {coords.shape}'

    # Test write back to file
    out_file = os.path.join(tmp_path, "Charisma_points_out.txt")
    reload.convert_to_charisma(out_file)

    assert os.path.exists(out_file)
    with open(out_file, 'r') as f:
        line = f.readline()

    assert line == 'INLINE :\t1 XLINE :\t1\t420691.19624\t6292314.22044\t2799.05591\n', 'Output Charisma file does not look as expected'


def test_pointset_from_irap(example_model_and_crs, test_data_path, tmp_path):
    # Set up a PointSet and save to resqml file
    model, crs = example_model_and_crs
    irap_file = os.path.join(test_data_path, "IRAP_points.txt")
    points = rqs.PointSet(parent_model = model, irap_file = str(irap_file), crs_uuid = crs.uuid)
    points.write_hdf5()
    points.create_xml()
    model.store_epc()

    # Test reload from resqml
    model = rq.Model(epc_file = model.epc_file)
    reload = rqs.PointSet(parent_model = model, uuid = points.uuid)

    assert reload.title == str(irap_file)

    coords = reload.full_array_ref()
    assert_array_almost_equal(coords[0], np.array([429450.658333, 6296954.224574, 2403.837646]))

    assert coords.shape == (9, 3), f'Expected shape (9,3), not {coords.shape}'

    # Test write back to file
    out_file = os.path.join(tmp_path, "IRAP_points_out.txt")
    reload.convert_to_irap(out_file)

    assert os.path.exists(out_file)
    with open(out_file, 'r') as f:
        line = f.readline()

    assert line == '429450.658333 6296954.224574 2403.837646\n', 'Output IRAP file does not look as expected'


def test_pointset_from_polyline(example_model_and_crs):
    # Set up a PolyLine and save to resqml file
    model, crs = example_model_and_crs

    coords = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [12, 13, 14, 15, 16]]).T
    lines = resqpy.lines.Polyline(parent_model = model,
                                  set_coord = coords,
                                  title = 'Polylines',
                                  set_crs = crs.uuid,
                                  is_closed = False)
    lines.write_hdf5()
    lines.create_xml()
    model.store_epc()

    # Reload the model, and generate pointset using the polyline
    model = rq.Model(epc_file = model.epc_file)
    reload_lines = resqpy.lines.Polyline(parent_model = model, uuid = lines.uuid)

    points = rqs.PointSet(parent_model = model, polyline = reload_lines, crs_uuid = crs.uuid)
    points.write_hdf5()
    points.create_xml()
    model.store_epc()

    # Reload the model, and ensure the coordinates are as expected
    model = rq.Model(epc_file = model.epc_file)
    reload = rqs.PointSet(parent_model = model, uuid = points.uuid)
    assert_array_almost_equal(reload.full_array_ref(), coords)


def test_pointset_from_polylineset(example_model_and_crs):
    # Set up a PolyLine and save to resqml file
    model, crs = example_model_and_crs

    coords1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [12, 13, 14, 15, 16]]).T
    coords2 = np.array([[11, 12, 13, 14, 15], [16, 17, 18, 19, 110], [112, 113, 114, 115, 116]]).T

    line1 = resqpy.lines.Polyline(parent_model = model,
                                  title = 'Polyline1',
                                  set_coord = coords1,
                                  set_crs = crs.uuid,
                                  is_closed = False)

    line2 = resqpy.lines.Polyline(parent_model = model,
                                  title = 'Polyline2',
                                  set_coord = coords2,
                                  set_crs = crs.uuid,
                                  is_closed = False)

    lines = resqpy.lines.PolylineSet(parent_model = model, title = 'Polylines', polylines = [line1, line2])
    lines.write_hdf5()
    lines.create_xml()
    model.store_epc()

    # Reload the model, and generate pointset using the polylineset
    model = rq.Model(epc_file = model.epc_file)
    reload_lines = resqpy.lines.PolylineSet(parent_model = model, uuid = lines.uuid)

    points = rqs.PointSet(parent_model = model, polyset = reload_lines, crs_uuid = crs.uuid)
    points.write_hdf5()
    points.create_xml()
    model.store_epc()

    # Reload the model, and ensure the coordinates are as expected
    model = rq.Model(epc_file = model.epc_file)
    reload = rqs.PointSet(parent_model = model, uuid = points.uuid)

    assert_array_almost_equal(reload.full_array_ref(), np.concatenate((coords1, coords2), axis = 0))


def test_pointset_minimum_xy_area_rectangle(example_model_and_crs):
    model, crs = example_model_and_crs
    r3_2 = maths.sqrt(3.0) / 2.0
    coords = np.array([(0.0, 0.0, 0.0), (140.0 * r3_2, 70.0, -12.7), (-17.0, 34.0 * r3_2, 5.6),
                       (140.0 * r3_2 - 17.0, 34.0 * r3_2 + 70.0, 923.0)],
                      dtype = float)
    ps = rqs.PointSet(model, points_array = coords, crs_uuid = crs.uuid, title = 'oblong')
    d1, d2, theta = ps.minimum_xy_area_rectangle(delta_theta = 3.0)
    assert 33.0 < d1 < 35.0
    assert 138.0 < d2 < 142.0
    assert 56.9 < theta < 63.1


def test_surface_normal_vectors(tmp_model):
    # Arrange
    points = np.array([[1.0, 1.0, 1.0], [2.0, 0.0, 1.0], [2.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
    triangles = tri.dt(points)
    surface = rqs.Surface(tmp_model)
    surface.set_from_triangles_and_points(triangles, points)
    normal_vectors_expected = np.array([[0.70710678, 0.0, -0.70710678], [0.0, 0.0, -1.0]])

    # Act
    normal_vectors = surface.normal_vectors()

    # Assert
    # NB: this assumes that the triangulation has returned the triangles in a particular order!
    assert_array_almost_equal(normal_vectors, normal_vectors_expected)


def test_sample_z_at_xy_points(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    s1 = rqs.Surface(model, crs_uuid = crs.uuid, title = 'test 1')
    s1.set_to_horizontal_plane(1066.0, box_xyz = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 0.0]], dtype = float))
    p3 = np.array([(30.0, 50.0, 12345.0), (20.0, 80.0, -1234.0), (50.0, 50.0, 0.0)], dtype = float)
    p2 = p3[:, :2]
    ez1 = np.full(len(p3), 1066.0, dtype = float)

    # Act
    z1_2 = s1.sample_z_at_xy_points(p2)
    z1_3 = s1.sample_z_at_xy_points(p3, multiple_handling = 'exception')

    # Assert
    assert_array_almost_equal(z1_2, ez1)
    assert_array_almost_equal(z1_2, z1_3)


def test_axial_edge_crossings(example_model_and_crs):
    model, crs = example_model_and_crs
    z_values = np.array([(-100.0, -100.0, -100.0, -100.0), (100.0, 100.0, 100.0, 100.0), (500.0, 500.0, 500.0, 500.0)],
                        dtype = float)
    trim = rqs.TriMesh(model,
                       t_side = 100.0,
                       nj = 3,
                       ni = 4,
                       z_values = z_values,
                       crs_uuid = crs.uuid,
                       title = 'test tri mesh')
    surf = rqs.Surface.from_tri_mesh(trim)

    # z axis
    z_cross = surf.axial_edge_crossings(2)
    assert z_cross.shape == (7, 3)
    xi = np.argsort(z_cross[:, 0])
    assert_array_almost_equal(z_cross[:, 2], 0.0)
    assert_array_almost_equal(z_cross[:, 1], 50.0 * maths.sqrt(3.0) / 2.0)
    assert_array_almost_equal(z_cross[xi[1:], 0] - z_cross[xi[:-1], 0], 50.0)
    assert maths.isclose(z_cross[xi[0], 0], 25.0)
    z_cross = surf.axial_edge_crossings(2, value = 300.0)
    assert z_cross.shape == (7, 3)
    xi = np.argsort(z_cross[:, 0])
    assert_array_almost_equal(z_cross[:, 2], 300.0)
    assert_array_almost_equal(z_cross[:, 1], 150.0 * maths.sqrt(3.0) / 2.0)
    assert_array_almost_equal(z_cross[xi[1:], 0] - z_cross[xi[:-1], 0], 50.0)
    assert maths.isclose(z_cross[xi[0], 0], 25.0)
    # y axis
    z_cross = surf.axial_edge_crossings(1, value = 150.0 * maths.sqrt(3.0) / 2.0)
    assert z_cross.shape == (7, 3)
    xi = np.argsort(z_cross[:, 0])
    assert_array_almost_equal(z_cross[:, 2], 300.0)
    assert_array_almost_equal(z_cross[:, 1], 150.0 * maths.sqrt(3.0) / 2.0)
    assert_array_almost_equal(z_cross[xi[1:], 0] - z_cross[xi[:-1], 0], 50.0)
    assert maths.isclose(z_cross[xi[0], 0], 25.0)
    # x axis
    z_cross = surf.axial_edge_crossings(0, value = 125.0)
    assert z_cross.shape == (5, 3)
    yi = np.argsort(z_cross[:, 1])
    assert_array_almost_equal(z_cross[yi, 2], (-100.0, 0.0, 100.0, 300.0, 500.0))
    assert_array_almost_equal(z_cross[yi[1:], 1] - z_cross[yi[:-1], 1], 50.0 * maths.sqrt(3.0) / 2.0)
    maths.isclose(z_cross[yi[0], 1], 0.0)
    assert_array_almost_equal(z_cross[:, 0], 125.0)


def test_adjust_flange_z(example_model_and_crs):
    model, crs = example_model_and_crs
    # create a bowl like point set, but invert z values on one side of bowl
    arc = np.array([(3, 0), (5, 1), (7, 3), (8, 6)], dtype = float) * 100.0
    points = np.expand_dims(np.array((0.0, 0.0, 0.0), dtype = float), axis = 0)
    for i in range(8):
        theta = maths.pi * float(i) / 4.0
        sin_theta = maths.sin(theta)
        cos_theta = maths.cos(theta)
        for j in range(len(arc)):
            z = 0.0
            if i < 3:
                z = arc[j, 1]
            elif 4 <= i < 7:
                z = -arc[j, 1]
            new_p = np.array((arc[j, 0] * cos_theta, arc[j, 0] * sin_theta, z), dtype = float)
            points = np.concatenate((points, np.expand_dims(new_p, axis = 0)), axis = 0)
    # add some flange extension points
    outer_start = len(points)
    outliers = np.array([(1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0)], dtype = float) * 3000.0
    ro = 1000.0 * maths.sqrt(18.0)
    points = np.concatenate((points, outliers), axis = 0)
    # triangulate in xy view
    t = tri.dt(points)
    # set up a flange property on the triangles, True for any triangle using an outlier point
    tout_set = set()
    for p_i in range(outer_start, len(points)):
        tout_set |= set(tri.triangles_using_point(t, p_i))
    flange = np.zeros(len(t), dtype = bool)
    flange[np.array(tuple(tout_set), dtype = int)] = True
    assert np.count_nonzero(flange) == 12
    # manually derived +ve z values corresponding to the saucer parameter values below
    ez = np.array((600.0 * ro / 800.0, 600.0 * (ro - 200.0) / 600.0, 50.0 + 550.0 *
                   (ro - 400.0) / 400.0, 200.0 + 400.0 * (ro - 600.0) / 200.0, 300.0 + 300.0 * (ro - 700.0) / 100.0),
                  dtype = float)
    # for a selection of saucer parameter values
    for i, sauciness in enumerate((0.0, 0.25, 0.5, 0.75, 0.9)):
        p = points.copy()
        # adjust z values of outliers using the function being tested
        rqs._adjust_flange_z(model, crs.uuid, p, outer_start, t, flange, sauciness)
        # check z values against expectation
        assert_array_almost_equal(p[:, :2], points[:, :2])  # no change in x or y
        assert_array_almost_equal(p[:outer_start, 2], points[:outer_start, 2])  # no change in z for non-outliers
        assert maths.isclose(p[outer_start, 2], ez[i])
        assert maths.isclose(p[outer_start + 1, 2], 0.0, abs_tol = 1.0e-6)
        assert maths.isclose(p[outer_start + 2, 2], -ez[i])
        assert maths.isclose(p[outer_start + 3, 2], 0.0, abs_tol = 1.0e-6)


def test_resampling(tmp_path):
    # Arrange
    model = rq.new_model(f"{tmp_path}\test.epc")
    crs = rqc.Crs(model, title = 'crs example')
    crs.create_xml()
    model.store_epc()

    x = np.linspace(1, 5, 5)
    y = np.linspace(1, 5, 5)
    xs, ys = np.meshgrid(x, y)
    zs = np.random.randint(low = 5, high = 10, size = xs.shape)
    points = np.array([xs, ys, zs], dtype = float).T

    pointset = rqs.PointSet(model, title = 'points', crs_uuid = crs.uuid, points_array = points)
    surf = rqs.Surface(model, title = 'surface', crs_uuid = crs.uuid, point_set = pointset)

    surf.write_hdf5()
    surf.create_xml()
    model.store_epc()
    ot, op = surf.triangles_and_points()

    # Act
    resampled = surf.resampled_surface(title = None)
    resampled_name = surf.resampled_surface(title = 'testing')
    resampled.write_hdf5()
    resampled.create_xml()
    resampled_name.write_hdf5()
    resampled_name.create_xml()
    model.store_epc()

    # Assert
    reload = rq.Model(model.epc_file)
    resampled = rqs.Surface(reload, resampled.uuid)
    resampled_name = rqs.Surface(reload, resampled_name.uuid)
    assert resampled is not None
    assert resampled_name is not None

    coords = np.array([[2.5, 2.5], [3.4, 3.4], [4.6, 4.6]])
    np.testing.assert_array_almost_equal(surf.sample_z_at_xy_points(coords), resampled.sample_z_at_xy_points(coords))
    rt, rp = resampled.triangles_and_points()

    assert len(rt) == 4 * len(ot)
    assert np.min(rp[:, 0]) == np.min(op[:, 0])
    assert np.max(rp[:, 0]) == np.max(op[:, 0])
    assert np.min(rp[:, 1]) == np.min(op[:, 1])
    assert np.max(rp[:, 1]) == np.max(op[:, 1])
    assert np.min(rp[:, 2]) == np.min(op[:, 2])
    assert np.max(rp[:, 2]) == np.max(op[:, 2])

    assert resampled.crs_uuid == surf.crs_uuid
    assert resampled.citation_title == surf.citation_title
    assert resampled_name.citation_title == 'testing'
    assert resampled.extra_metadata == {'resampled from surface': str(surf.uuid)}
    assert resampled_name.extra_metadata == {'resampled from surface': str(surf.uuid)}


def test_resample_surface_unique_edges(tmp_path):
    # Arrange
    model = rq.new_model(f"{tmp_path}\test.epc")
    crs = rqc.Crs(model, title = 'crs example')
    crs.create_xml()
    model.store_epc()

    x = np.linspace(1, 5, 5)
    y = np.linspace(1, 5, 5)
    xs, ys = np.meshgrid(x, y)
    zs = np.random.randint(low = 5, high = 10, size = xs.shape)
    points = np.array([xs, ys, zs], dtype = float).T

    pointset = rqs.PointSet(model, title = 'points', crs_uuid = crs.uuid, points_array = points)
    surf = rqs.Surface(model, title = 'surface', crs_uuid = crs.uuid, point_set = pointset)

    surf.write_hdf5()
    surf.create_xml()
    model.store_epc()
    ot, op = surf.triangles_and_points()

    # Act
    resampled = surf.resample_surface_unique_edges()
    resampled.write_hdf5()
    resampled.create_xml()
    model.store_epc()

    # Assert
    reload = rq.Model(model.epc_file)
    resampled = rqs.Surface(reload, resampled.uuid)
    assert resampled is not None

    coords = np.array([[2.5, 2.5], [3.4, 3.4], [4.6, 4.6]])
    np.testing.assert_array_almost_equal(surf.sample_z_at_xy_points(coords), resampled.sample_z_at_xy_points(coords))
    rt, rp = resampled.triangles_and_points()

    assert len(rt) == 112
    assert np.min(rp[:, 0]) == np.min(op[:, 0])
    assert np.max(rp[:, 0]) == np.max(op[:, 0])
    assert np.min(rp[:, 1]) == np.min(op[:, 1])
    assert np.max(rp[:, 1]) == np.max(op[:, 1])
    assert np.min(rp[:, 2]) == np.min(op[:, 2])
    assert np.max(rp[:, 2]) == np.max(op[:, 2])

    assert resampled.crs_uuid == surf.crs_uuid
    assert resampled.citation_title == surf.citation_title
    assert resampled.extra_metadata == {'resampled from surface': str(surf.uuid)}


def test_from_downsampling_surface(example_model_and_crs):
    model, crs = example_model_and_crs
    z_values = np.random.random((20, 25)) * 100.0 + 950.0
    trim = rqs.TriMesh(model,
                       t_side = 100.0,
                       nj = 20,
                       ni = 25,
                       z_values = z_values,
                       crs_uuid = crs.uuid,
                       title = 'tri mesh for downsampling')
    fine_surf = rqs.Surface.from_tri_mesh(trim)
    surf = rqs.Surface.from_downsampling_surface(fine_surf, point_count = 50, title = 'rough')
    assert surf is not None
    assert surf.title == 'rough'
    assert surf.node_count() == 50
    surf.write_hdf5()
    surf.create_xml()
    surf_reload = rqs.Surface(model, uuid = surf.uuid)
    assert surf_reload is not None
    t, p = surf_reload.triangles_and_points()
    assert p.shape == (50, 3)
    assert np.all(p[:, 2] >= 950.0) and np.all(p[:, 2] <= 1050.0)


def test_from_downsampling_surface_few_source_points(example_model_and_crs):
    model, crs = example_model_and_crs
    z_values = np.random.random((10, 5)) * 100.0 + 950.0
    trim = rqs.TriMesh(model,
                       t_side = 100.0,
                       nj = 10,
                       ni = 5,
                       z_values = z_values,
                       crs_uuid = crs.uuid,
                       title = 'small surface')
    fine_surf = rqs.Surface.from_tri_mesh(trim)
    surf = rqs.Surface.from_downsampling_surface(fine_surf, point_count = 60)
    assert surf is not None
    assert surf.title == 'small surface'
    assert surf.node_count() == 50
    surf.write_hdf5()
    surf.create_xml()
    surf_reload = rqs.Surface(model, uuid = surf.uuid)
    assert surf_reload is not None
    t, p = surf_reload.triangles_and_points()
    assert p.shape == (50, 3)
    ft, fp = fine_surf.triangles_and_points()
    assert np.all(t == ft)
    assert_array_almost_equal(p, fp)


def test_from_downsampling_surface_different_model(example_model_and_crs):
    model, crs = example_model_and_crs
    new_m = rq.new_model(model.epc_file[:-4] + '_new.epc')
    z_values = np.random.random((20, 25)) * 100.0 + 950.0
    trim = rqs.TriMesh(model,
                       t_side = 100.0,
                       nj = 20,
                       ni = 25,
                       z_values = z_values,
                       crs_uuid = crs.uuid,
                       title = 'tri mesh for downsampling')
    fine_surf = rqs.Surface.from_tri_mesh(trim)
    surf = rqs.Surface.from_downsampling_surface(fine_surf, point_count = 50, title = 'rough', target_model = new_m)
    assert surf is not None
    assert surf.title == 'rough'
    assert surf.node_count() == 50
    assert surf.model is new_m
    surf.write_hdf5()
    surf.create_xml()
    surf_reload = rqs.Surface(new_m, uuid = surf.uuid)
    assert surf_reload is not None
    t, p = surf_reload.triangles_and_points()
    assert p.shape == (50, 3)
    assert np.all(p[:, 2] >= 950.0) and np.all(p[:, 2] <= 1050.0)


def test_from_tri_mesh_excluding_nans(example_model_and_crs):
    model, crs = example_model_and_crs
    z_values = np.random.random((4, 5)) * 100.0 + 950.0
    z_values[0, 0] = np.nan
    z_values[2, 2] = np.nan
    trim = rqs.TriMesh(model,
                       t_side = 100.0,
                       nj = 4,
                       ni = 5,
                       z_values = z_values,
                       crs_uuid = crs.uuid,
                       title = 'surface with nans')
    surf_a = rqs.Surface.from_tri_mesh(trim, exclude_nans = False)
    surf_b = rqs.Surface.from_tri_mesh(trim, exclude_nans = True)
    t_a, p_a = surf_a.triangles_and_points()
    t_b, p_b = surf_b.triangles_and_points()
    assert p_a.shape == (20, 3)
    assert t_a.shape == (24, 3)
    assert p_b.shape == (18, 3)
    assert t_b.shape == (17, 3)
    assert not np.any(np.isnan(p_b))
    assert np.all(t_b >= 0)
    assert np.all(t_b < 18)
    sample_points = np.array([(50.0, 45.0), (50.0, 130.0), (140.0, 210.0)], dtype = float)
    z_sample = surf_b.sample_z_at_xy_points(sample_points, multiple_handling = 'any')
    assert np.isnan(z_sample[0])
    assert np.isnan(z_sample[2])
    assert 950.0 <= z_sample[1] <= 1050.0


def test_set_to_split_surface(example_model_and_crs):
    model, crs = example_model_and_crs
    z_values = np.random.random((30, 20)) * 100.0 + 950.0
    trim = rqs.TriMesh(model,
                       t_side = 100.0,
                       nj = 30,
                       ni = 20,
                       z_values = z_values,
                       crs_uuid = crs.uuid,
                       title = 'surface for split')
    surf = rqs.Surface.from_tri_mesh(trim)
    split_surf = rqs.Surface(model, crs_uuid = crs.uuid, title = 'split surface')
    line = np.array([(-100.0, 150.0, 0.0), (2100.0, 2500.0, 0.0)], dtype = float)
    delta_xyz = np.array([(50.0, -50.0, 50.0), (-30.0, 30.0, -30.0)], dtype = float)
    split_surf.set_to_split_surface(surf, line, delta_xyz)
    split_surf.write_hdf5()
    split_surf.create_xml()


def test_set_to_triangle(example_model_and_crs):

    def check_one(s):
        assert s is not None
        assert s.number_of_patches() == 1
        t, p = s.triangles_and_points()
        assert t.shape == (1, 3)
        assert p.shape == (3, 3)
        assert_array_almost_equal(s.edge_lengths(), np.array([(3.0, 5.0, 4.0)], dtype = float))

    model, crs = example_model_and_crs
    vertices = np.array([(1.0, 1.0, 1.0), (4.0, 1.0, 1.0), (1.0, 5.0, 1.0)], dtype = float)
    surf = rqs.Surface(model, crs_uuid = crs.uuid, title = 'pythagorean')
    surf.set_to_triangle(vertices)
    surf.write_hdf5()
    surf.create_xml()
    check_one(surf)
    surf = rqs.Surface(model, uuid = surf.uuid)
    check_one(surf)


def test_vertical_rescale_points(example_model_and_crs):
    model, crs = example_model_and_crs
    vertices = np.array([(1.0, 1.0, 1.0), (4.0, 1.0, 2.0), (1.0, 5.0, 3.0)], dtype = float)
    surf = rqs.Surface(model, crs_uuid = crs.uuid, title = 'pythagorean')
    surf.set_to_triangle(vertices)
    surf.vertical_rescale_points(ref_depth = 0.0, scaling_factor = 3.0)
    surf.write_hdf5()
    surf.create_xml()
    surf = rqs.Surface(model, uuid = surf.uuid)
    t, p = surf.triangles_and_points()
    assert np.all(t.flatten() == (0, 1, 2))
    assert_array_almost_equal(p[:, 2], (3.0, 6.0, 9.0))
    vertices = np.array([(1.0, 1.0, 1.0), (4.0, 1.0, 2.0), (1.0, 5.0, 3.0)], dtype = float)
    surf = rqs.Surface(model, crs_uuid = crs.uuid, title = 'pythagorean')
    surf.set_to_triangle(vertices)
    surf.vertical_rescale_points(ref_depth = -2.0, scaling_factor = 2.0)
    surf.write_hdf5()
    surf.create_xml()
    surf = rqs.Surface(model, uuid = surf.uuid)
    t, p = surf.triangles_and_points()
    assert_array_almost_equal(p[:, 2], (4.0, 6.0, 8.0))


def test_edge_lengths(example_model_and_crs):
    model, crs = example_model_and_crs
    surf = rqs.Surface(model, crs_uuid = crs.uuid, title = 'pair')
    corners = np.array([(1.0, 1.0, 0.0), (1.0, 4.0, 0.0), (5.0, 1.0, 0.0), (5.0, 4.0, 0.0)], dtype = float)
    surf.set_to_triangle_pair(corners)
    surf.write_hdf5()
    surf.create_xml()
    lengths = surf.edge_lengths()
    assert_array_almost_equal(lengths, np.array([(3.0, 4.0, 5.0), (5.0, 3.0, 4.0)], dtype = float))
    lengths = surf.edge_lengths(required_uom = 'cm')
    assert_array_almost_equal(lengths, np.array([(300.0, 400.0, 500.0), (500.0, 300.0, 400.0)], dtype = float))


def test_extended_surface_with_flange_extension(example_model_and_crs):
    model, crs = example_model_and_crs

    # create a set of points (must have a convex hull)
    x = np.array([-1, -1.5, -2, -1.5, -1, 0, 1, 1.5, 2, 1.5, 1, 0], dtype = float) + 5
    y = np.array([-1.1, -0.5, 0, 0.5, 1, 0.5, 1.1, 0.5, 0, -0.5, -1, 0], dtype = float) + 5
    z = np.zeros(shape = (12), dtype = float) + 5
    p = np.stack((x, y, z), axis = -1)

    # make a PointSet object
    ps = rqs.PointSet(model, crs_uuid = crs.uuid, points_array = p, title = 'points')
    ps.write_hdf5()
    ps.create_xml()

    # try out flange extension
    surf = rqs.Surface(model, crs_uuid = ps.crs_uuid, title = 'surface from ' + str(ps.title))
    assert surf is not None
    surf.set_from_point_set(ps, reorient = False, reorient_max_dip = None, extend_with_flange = False)
    surf.write_hdf5()
    surf.create_xml()
    orig_t, orig_p = surf.triangles_and_points()

    new_surf, flange_bool = surf.extend_surface_with_flange(reorient = False,
                                                            flange_radial_factor = 10.0,
                                                            flange_radial_distance = None,
                                                            saucer_parameter = None,
                                                            retriangulate = False)

    new_t, new_p = new_surf.triangles_and_points()

    assert np.all(flange_bool[len(orig_t):])
    assert not np.any(flange_bool[:len(orig_t)])
    assert np.all(np.isin(orig_t, new_t))

    np.testing.assert_array_almost_equal(
        np.array([[5, 28.07078471], [26.81071628, 12.43307607], [27.71578211, 1.25570298], [24.45543821, -7.28011087],
                  [5, -17.98745138], [-16.42279299, -3.40843501], [-17.86764418, 7.76400526],
                  [-15.03584362, 16.39531138]]), new_p[len(orig_p):, :2])


def test_extended_surface_with_flange_extension_saucer(example_model_and_crs):
    model, crs = example_model_and_crs

    # create a set of points (must have a convex hull)
    x = np.array([-1, -1.5, -2, -1.5, -1, 0, 1, 1.5, 2, 1.5, 1, 0], dtype = float) + 5
    y = np.array([-1.1, -0.5, 0, 0.5, 1, 0.5, 1.1, 0.5, 0, -0.5, -1, 0], dtype = float) + 5
    z = np.zeros(shape = (12), dtype = float) + 5
    p = np.stack((x, y, z), axis = -1)

    # make a PointSet object
    ps = rqs.PointSet(model, crs_uuid = crs.uuid, points_array = p, title = 'points')
    ps.write_hdf5()
    ps.create_xml()

    # try out flange extension
    surf = rqs.Surface(model, crs_uuid = ps.crs_uuid, title = 'surface from ' + str(ps.title))
    assert surf is not None
    surf.set_from_point_set(ps, reorient = False, reorient_max_dip = None, extend_with_flange = False)
    surf.write_hdf5()
    surf.create_xml()
    orig_t, orig_p = surf.triangles_and_points()

    new_surf, flange_bool = surf.extend_surface_with_flange(reorient = False,
                                                            flange_radial_factor = 10.0,
                                                            flange_radial_distance = None,
                                                            saucer_parameter = 45.0,
                                                            retriangulate = False)

    new_t, new_p = new_surf.triangles_and_points()

    assert np.all(flange_bool[len(orig_t):])
    assert not np.any(flange_bool[:len(orig_t)])
    assert np.all(np.isin(orig_t, new_t))

    np.testing.assert_array_almost_equal(
        np.array([[5, 28.07078471, -18.029118], [26.81071628, 12.43307607, -18.029118],
                  [27.71578211, 1.25570298, -18.029118], [24.45543821, -7.28011087, -18.029118],
                  [5, -17.98745138, -18.029118], [-16.42279299, -3.40843501, -18.029118],
                  [-17.86764418, 7.76400526, -18.029118], [-15.03584362, 16.39531138, -18.029118]]),
        new_p[len(orig_p):])


def test_extended_surface_with_flange_extension_retriangulate(example_model_and_crs):
    model, crs = example_model_and_crs

    # number of random points to use
    n = 20

    # create a set of random points
    x = np.random.random(n) * 1000.0 - 500.0
    y = np.random.random(n) * 1000.0 - 500.0
    z = np.random.random(n)  #  note: triangulation does not use z values
    p = np.stack((x, y, z), axis = -1)
    centre = np.mean(p, axis = 0)

    # make a PointSet object
    ps = rqs.PointSet(model, crs_uuid = crs.uuid, points_array = p, title = 'random points in square')
    ps.write_hdf5()
    ps.create_xml()

    # try out flange extension
    orig_surf = rqs.Surface(model, crs_uuid = ps.crs_uuid, title = 'surface from ' + str(ps.title))
    assert orig_surf is not None
    orig_surf.set_from_point_set(ps,
                            reorient = True,
                            reorient_max_dip = None,
                            extend_with_flange = False,
                            make_clockwise = False)
    orig_surf.write_hdf5()
    orig_surf.create_xml()

    # flange extension with simple saucer
    new_surf, flange_bool = orig_surf.extend_surface_with_flange(
        reorient = True,
        reorient_max_dip = None,
        flange_point_count = 12,
        flange_radial_factor = 2.0,
        flange_radial_distance = 2000.0,
        flange_inner_ring = True,
        saucer_parameter = -60.0,
        make_clockwise = False,
        retriangulate = True)
    new_surf.write_hdf5()
    new_surf.create_xml()

    assert new_surf.node_count() == 56
    _, p = new_surf.triangles_and_points()
    min_p = np.min(p, axis = 0)
    max_p = np.max(p, axis = 0)
    assert -2010.0 <= min_p[0] - centre[0] < -1900.0
    assert -2010.0 <= min_p[1] - centre[1] < -1900.0
    assert 1900.0 < max_p[0] - centre[0] <= 2010.0
    assert 1900.0 < max_p[1] - centre[1] <= 2010.0
    assert 0.0 <= min_p[2] <= 1.0
    assert 3464.0 < max_p[2] < 3465.5
