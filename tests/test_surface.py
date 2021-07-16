import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.model as rq
import resqpy.surface
import resqpy.organize
import resqpy.grid
import resqpy.lines as rql
import resqpy.grid_surface as rqgs
import resqpy.olio.uuid as bu


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
   gcs = rqgs.find_faces_to_represent_surface(grid, surf, 'staffa', mode = 'staffa')
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
   assert bu.matching_uuids(tmp_model.uuid(obj_type = 'GridConnectionSetRepresentation'), gcs.uuid)


def test_delaunay_triangulation(example_model_and_crs):

   model, crs = example_model_and_crs

   # number of random points to use
   n = 20

   # create a set of random points
   x = np.random.random(n) * 1000.0
   y = np.random.random(n) * 1000.0
   z = np.random.random(n)  # note: triangulation does not use z values
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


def test_regular_mesh(example_model_and_crs):

   model, crs = example_model_and_crs

   # number of points in mesh, origin spacing
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
   peristent_mesh = resqpy.surface.Mesh(model, uuid = mesh_uuid)

   # check some of the metadata
   assert peristent_mesh.ni == ni and peristent_mesh.nj == nj
   assert peristent_mesh.flavour == 'reg&z'
   assert_array_almost_equal(np.array(peristent_mesh.regular_origin), np.array(origin))
   assert_array_almost_equal(np.array(peristent_mesh.regular_dxyz_dij), np.array([[di, 0.0, 0.0], [0.0, dj, 0.0]]))

   # check a fully expanded version of the points
   assert_array_almost_equal(peristent_mesh.full_array_ref(), mesh.full_array_ref())

   # check that we can build a Surface from the Mesh
   surf = peristent_mesh.surface(quad_triangles = True)
   assert surf is not None

   # do some basic checks that the surface looks consistent with the mesh
   t, p = surf.triangles_and_points()
   assert len(p) == (ni * nj) + ((ni - 1) * (nj - 1))  # quad triangles mode introduces the extra points
   assert len(t) == 4 * (ni - 1) * (nj - 1)
   assert_array_almost_equal(np.min(p, axis = 0), np.min(peristent_mesh.full_array_ref().reshape(-1, 3), axis = 0))
   assert_array_almost_equal(np.max(p, axis = 0), np.max(peristent_mesh.full_array_ref().reshape(-1, 3), axis = 0))
