import numpy as np
from numpy.testing import assert_array_almost_equal
import os
import resqpy.grid
import resqpy.grid_surface as rqgs
import resqpy.lines as rql
import resqpy.model as rq
import resqpy.olio.uuid as bu
import resqpy.organize
import resqpy.surface

import pytest

# Integration tests for surface classes


def test_surface(tmp_model):
    # Set up a Surface
    title = 'Mountbatten'
    model = tmp_model
    surf = resqpy.surface.Surface(parent_model = model, title = title)
    surf.create_xml()

    # Add a interpretation
    assert surf.represented_interpretation_root is None
    surf.create_interpretation_and_feature(kind = 'fault')
    assert surf.represented_interpretation_root is not None

    # Check fault can be loaded in again
    model.store_epc()
    fault_interp = resqpy.organize.FaultInterpretation(model, uuid = surf.represented_interpretation_uuid)
    fault_feature = resqpy.organize.TectonicBoundaryFeature(model, uuid = fault_interp.tectonic_boundary_feature.uuid)

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
    surf = resqpy.surface.Surface(tmp_model, crs_uuid = crs.uuid)
    surf.set_from_triangles_and_points(triangles, points.reshape((-1, 3)))
    assert surf is not None
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
    ps = resqpy.surface.PointSet(model, crs_uuid = crs.uuid, points_array = p, title = 'random points in square')

    # make another PointSet as random points within a closed polyline
    vertices = np.array(
        ((50.0, 99.0, 13.0), (85.0, 60.0, 17.5), (62.7, 11.0, 10.0), (33.3, 15.3, 19.2), (12.8, 57.8, 15.0)))
    polygon = rql.Polyline(model, set_crs = crs.uuid, set_bool = True, set_coord = vertices, title = 'the pentagon')
    polygon.write_hdf5()
    polygon.create_xml()
    ps2 = resqpy.surface.PointSet(model,
                                  crs_uuid = crs.uuid,
                                  polyline = polygon,
                                  random_point_count = n,
                                  title = 'random points in polygon')

    # process the point sets into triangulated surfaces
    for point_set in (ps, ps2):
        point_set.write_hdf5()
        point_set.create_xml()
        surf = resqpy.surface.Surface(model, point_set = point_set, title = 'surface from ' + str(point_set.title))
        assert surf is not None
        surf.write_hdf5()
        surf.create_xml()
        # check that coordinate range of points looks okay
        triangles, points = surf.triangles_and_points()
        assert len(points) == n
        original_points = point_set.full_array_ref()
        assert_array_almost_equal(np.nanmin(original_points, axis = 0), np.nanmin(points, axis = 0))
        assert_array_almost_equal(np.nanmax(original_points, axis = 0), np.nanmax(points, axis = 0))


@pytest.mark.parametrize('mesh_file,mesh_format,firstval', [('Surface_roxartext.txt', 'rms', 0.4229),
                                                            ('Surface_roxartext.txt', 'roxar', 0.4229),
                                                            ('Surface_zmap.dat', 'zmap', 0.4648)])
def test_surface_from_mesh_file(example_model_and_crs, test_data_path, mesh_file, mesh_format, firstval):
    # Arrange
    model, crs = example_model_and_crs
    in_file = test_data_path / mesh_file

    # Act
    surface = resqpy.surface.Surface(parent_model = model, mesh_file = in_file, mesh_format = mesh_format)

    # Assert
    assert surface is not None
    assert surface.patch_list[0].triangle_count == 12
    assert surface.patch_list[0].points[0][2] == firstval


def test_surface_from_tsurf_file(example_model_and_crs, test_data_path):
    # Arrange
    model, crs = example_model_and_crs
    in_file = test_data_path / 'Surface_tsurf.txt'

    # Act
    surface = resqpy.surface.Surface(parent_model = model, tsurf_file = in_file, title = 'horizon')
    surface.create_interpretation_and_feature(kind = 'horizon')

    # Assert
    assert surface is not None
    assert surface.represented_interpretation_root is not None
    assert surface.patch_list[0].triangle_count == 12
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
    mesh = resqpy.surface.Mesh(model,
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
    persistent_mesh = resqpy.surface.Mesh(model, uuid = mesh_uuid)

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
    support = resqpy.surface.Mesh(model,
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

    refz = resqpy.surface.Mesh(model,
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
    reload_refzmesh = resqpy.surface.Mesh(reload, uuid = refz_uuid)

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
    mesh = resqpy.surface.Mesh(model,
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
    persistent_mesh = resqpy.surface.Mesh(model, uuid = mesh_uuid)

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

    mesh_file = test_data_path / infile

    # make an explicit mesh representation
    mesh = resqpy.surface.Mesh(model,
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
    persistent_mesh = resqpy.surface.Mesh(model, uuid = mesh_uuid)

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
    points = resqpy.surface.PointSet(model,
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
    saved_points = resqpy.surface.PointSet(model, uuid = points_uuid)

    # check a fully expanded version of the points
    assert_array_almost_equal(saved_points.full_array_ref(), points.full_array_ref())
    assert saved_points.represented_interpretation_root is not None


def test_pointset_from_2d_array(example_model_and_crs):
    model, crs = example_model_and_crs

    # create some random x,y values
    x = (np.random.random(50) * 10000)
    y = (np.random.random(50) * 10000)

    # make a pointset representation
    points = resqpy.surface.PointSet(model,
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
    saved_points = resqpy.surface.PointSet(model, uuid = points_uuid)

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
    points = resqpy.surface.PointSet(model,
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
    saved_points = resqpy.surface.PointSet(model, uuid = points_uuid)

    # check a fully expanded version of the points
    assert_array_almost_equal(saved_points.full_array_ref(), points.full_array_ref())
    assert_array_almost_equal(saved_points.single_patch_array_ref(0), np.array([x, y, [0] * 50]).T)
    assert_array_almost_equal(saved_points.single_patch_array_ref(1), np.array([x2, y2, [0] * 50]).T)


def test_pointset_from_charisma(example_model_and_crs, test_data_path, tmp_path):
    # Set up a PointSet and save to resqml file
    model, crs = example_model_and_crs

    charisma_file = test_data_path / "Charisma_points.txt"
    points = resqpy.surface.PointSet(parent_model = model, charisma_file = str(charisma_file), crs_uuid = crs.uuid)
    points.write_hdf5()
    points.create_xml()
    model.store_epc()

    # Test reload from resqml
    model = rq.Model(epc_file = model.epc_file)
    reload = resqpy.surface.PointSet(parent_model = model, uuid = points.uuid)

    assert reload.title == str(charisma_file)

    coords = reload.full_array_ref()
    assert_array_almost_equal(coords[0], np.array([420691.19624, 6292314.22044, 2799.05591]))

    assert coords.shape == (15, 3), f'Expected shape (15,3), not {coords.shape}'

    # Test write back to file
    out_file = str(tmp_path / "Charisma_points_out.txt")
    reload.convert_to_charisma(out_file)

    assert os.path.exists(out_file)
    with open(out_file, 'r') as f:
        line = f.readline()

    assert line == 'INLINE :\t1 XLINE :\t1\t420691.19624\t6292314.22044\t2799.05591\n', 'Output Charisma file does not look as expected'


def test_pointset_from_irap(example_model_and_crs, test_data_path, tmp_path):
    # Set up a PointSet and save to resqml file
    model, crs = example_model_and_crs
    irap_file = test_data_path / "IRAP_points.txt"
    points = resqpy.surface.PointSet(parent_model = model, irap_file = str(irap_file), crs_uuid = crs.uuid)
    points.write_hdf5()
    points.create_xml()
    model.store_epc()

    # Test reload from resqml
    model = rq.Model(epc_file = model.epc_file)
    reload = resqpy.surface.PointSet(parent_model = model, uuid = points.uuid)

    assert reload.title == str(irap_file)

    coords = reload.full_array_ref()
    assert_array_almost_equal(coords[0], np.array([429450.658333, 6296954.224574, 2403.837646]))

    assert coords.shape == (9, 3), f'Expected shape (9,3), not {coords.shape}'

    # Test write back to file
    out_file = str(tmp_path / "IRAP_points_out.txt")
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
                                  set_bool = False)
    lines.write_hdf5()
    lines.create_xml()
    model.store_epc()

    # Reload the model, and generate pointset using the polyline
    model = rq.Model(epc_file = model.epc_file)
    reload_lines = resqpy.lines.Polyline(parent_model = model, uuid = lines.uuid)

    points = resqpy.surface.PointSet(parent_model = model, polyline = reload_lines, crs_uuid = crs.uuid)
    points.write_hdf5()
    points.create_xml()
    model.store_epc()

    # Reload the model, and ensure the coordinates are as expected
    model = rq.Model(epc_file = model.epc_file)
    reload = resqpy.surface.PointSet(parent_model = model, uuid = points.uuid)
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
                                  set_bool = False)

    line2 = resqpy.lines.Polyline(parent_model = model,
                                  title = 'Polyline2',
                                  set_coord = coords2,
                                  set_crs = crs.uuid,
                                  set_bool = False)

    lines = resqpy.lines.PolylineSet(parent_model = model, title = 'Polylines', polylines = [line1, line2])
    lines.write_hdf5()
    lines.create_xml()
    model.store_epc()

    # Reload the model, and generate pointset using the polylineset
    model = rq.Model(epc_file = model.epc_file)
    reload_lines = resqpy.lines.PolylineSet(parent_model = model, uuid = lines.uuid)

    points = resqpy.surface.PointSet(parent_model = model, polyset = reload_lines, crs_uuid = crs.uuid)
    points.write_hdf5()
    points.create_xml()
    model.store_epc()

    # Reload the model, and ensure the coordinates are as expected
    model = rq.Model(epc_file = model.epc_file)
    reload = resqpy.surface.PointSet(parent_model = model, uuid = points.uuid)

    assert_array_almost_equal(reload.full_array_ref(), np.concatenate((coords1, coords2), axis = 0))


def test_tripatch_set_to_triangle(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    corners = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]])

    # Act
    tripatch = resqpy.surface.TriangulatedPatch(parent_model = model)
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
    tripatch = resqpy.surface.TriangulatedPatch(parent_model = model)
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
    tripatch = resqpy.surface.TriangulatedPatch(parent_model = model)
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
    tripatch = resqpy.surface.TriangulatedPatch(parent_model = model)
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
    tripatch = resqpy.surface.TriangulatedPatch(parent_model = model)
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
    tripatch = resqpy.surface.TriangulatedPatch(parent_model = model)
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
    tripatch = resqpy.surface.TriangulatedPatch(parent_model = model)
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
    tripatch = resqpy.surface.TriangulatedPatch(parent_model = model)
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
    tripatch = resqpy.surface.TriangulatedPatch(parent_model = model)
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
    tripatch = resqpy.surface.TriangulatedPatch(parent_model = model)
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
    tripatch = resqpy.surface.TriangulatedPatch(parent_model = model)
    tripatch.set_to_cell_faces_from_corner_points(cp = cp, quad_triangles = False)

    # Assert
    assert tripatch is not None
    assert tripatch.node_count == 8
    assert tripatch.points.shape == (8, 3)
    assert_array_almost_equal(tripatch.points[0], cp[0, 0, 0])
    assert_array_almost_equal(tripatch.points[2], cp[0, 1, 0])
    assert_array_almost_equal(tripatch.triangles[0], np.array([0, 3, 1]))
    assert_array_almost_equal(tripatch.triangles[-1], np.array([7, 1, 3]))
