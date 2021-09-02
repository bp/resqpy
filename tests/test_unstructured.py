import pytest
import os
import math as maths
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.unstructured as rug
import resqpy.olio.uuid as bu


def test_hexa_grid_from_grid(example_model_with_properties):

   model = example_model_with_properties

   ijk_grid_uuid = model.uuid(obj_type = 'IjkGridRepresentation')
   assert ijk_grid_uuid is not None

   # create an unstructured grid with hexahedral cells from an unsplit IJK grid (includes write_hdf5 and create_xml)
   hexa = rug.HexaGrid.from_unsplit_grid(model, ijk_grid_uuid, inherit_properties = True, title = 'HEXA')

   hexa_uuid = hexa.uuid

   epc = model.epc_file
   assert epc

   model.store_epc()

   # re-open model and check hexa grid

   model = rq.Model(epc)

   unstructured_uuids = model.uuids(obj_type = 'UnstructuredGridRepresentation')

   assert unstructured_uuids is not None and len(unstructured_uuids) == 1

   hexa_grid = grr.any_grid(model, uuid = unstructured_uuids[0])
   assert isinstance(hexa_grid, rug.HexaGrid)

   assert hexa_grid.cell_shape == 'hexahedral'

   hexa_grid.check_indices()
   hexa_grid.check_hexahedral()

   # instantiate ijk grid and compare hexa grid with it
   ijk_grid = grr.any_grid(model, uuid = ijk_grid_uuid)
   assert ijk_grid is not None
   assert not np.any(np.isnan(ijk_grid.points_ref()))

   assert hexa_grid.cell_count == ijk_grid.cell_count()
   assert hexa_grid.active_cell_count() == hexa_grid.cell_count
   assert hexa_grid.node_count == (ijk_grid.nk + 1) * (ijk_grid.nj + 1) * (ijk_grid.ni + 1)
   assert hexa_grid.face_count == ((ijk_grid.nk + 1) * ijk_grid.nj * ijk_grid.ni + ijk_grid.nk *
                                   (ijk_grid.nj + 1) * ijk_grid.ni + ijk_grid.nk * ijk_grid.nj * (ijk_grid.ni + 1))

   assert bu.matching_uuids(hexa_grid.extract_crs_uuid(), ijk_grid.crs_uuid)
   assert hexa_grid.crs_is_right_handed == rqc.Crs(model, uuid = ijk_grid.crs_uuid).is_right_handed_xyz()

   # points arrays should be identical for the two grids
   assert not np.any(np.isnan(hexa_grid.points_ref()))
   assert_array_almost_equal(hexa_grid.points_ref(), ijk_grid.points_ref(masked = False).reshape((-1, 3)))

   # compare centre points of cells (not sure if these would be coincident for irregular shaped cells)
   # note that the centre_point() method exercises several other methods
   hexa_centres = hexa_grid.centre_point()
   ijk_centres = ijk_grid.centre_point()
   assert_array_almost_equal(hexa_centres, ijk_centres.reshape((-1, 3)), decimal = 3)

   # check that face and node indices are in range
   hexa_grid.cache_all_geometry_arrays()
   assert np.all(0 <= hexa_grid.faces_per_cell)
   assert np.all(hexa_grid.faces_per_cell < hexa_grid.face_count)
   assert np.all(0 <= hexa_grid.nodes_per_face)
   assert np.all(hexa_grid.nodes_per_face < hexa_grid.node_count)
   assert len(hexa_grid.faces_per_cell_cl) == hexa_grid.cell_count
   assert len(hexa_grid.nodes_per_face_cl) == hexa_grid.face_count

   # check distinct nodes for first cell
   cell_nodes = hexa_grid.distinct_node_indices_for_cell(0)  # is sorted by node index
   assert len(cell_nodes) == 8
   ni1_nj1 = (ijk_grid.ni + 1) * (ijk_grid.nj + 1)
   expected_nodes = np.array((0, 1, ijk_grid.ni + 1, ijk_grid.ni + 2, ni1_nj1, ni1_nj1 + 1, ni1_nj1 + ijk_grid.ni + 1,
                              ni1_nj1 + ijk_grid.ni + 2),
                             dtype = int)
   assert np.all(cell_nodes == expected_nodes)

   # check that some simple convenience methods work okay
   assert hexa_grid.face_count_for_cell(0) == 6
   assert hexa_grid.max_face_count_for_any_cell() == 6
   assert hexa_grid.max_node_count_for_any_face() == 4

   # check that correct number of edges is found for a face
   edges = hexa_grid.edges_for_face(hexa_grid.face_count // 2)  # arbitrary face in middle
   assert edges.shape == (4, 2)
   edges = hexa_grid.edges_for_face_with_node_indices_ordered_within_pairs(hexa_grid.face_count // 2)
   assert edges.shape == (4, 2)
   for a, b in edges:  # check node within pair ordering
      assert a < b

   # compare corner points for first cell with those for ijk grid cell
   cp = hexa_grid.corner_points(0)
   assert cp.shape == (8, 3)
   assert_array_almost_equal(cp.reshape((2, 2, 2, 3)),
                             ijk_grid.corner_points(cell_kji0 = (0, 0, 0), cache_resqml_array = False))

   # have a look at handedness of cell faces
   assert len(hexa_grid.cell_face_is_right_handed) == 6 * hexa_grid.cell_count
   # following assertion only applies to HexaGrid built from_unsplit_grid()
   assert np.count_nonzero(hexa_grid.cell_face_is_right_handed) == 3 * hexa_grid.cell_count  # half are right handed

   # compare cell volumes for first cell
   hexa_vol = hexa_grid.volume(0)
   ijk_vol = ijk_grid.volume(cell_kji0 = 0, cache_resqml_array = False, cache_volume_array = False)
   assert maths.isclose(hexa_vol, ijk_vol)
   assert maths.isclose(hexa_vol, 1.0, rel_tol = 1.0e-3)

   # check face normal for first face (K- face of first cell)
   assert_array_almost_equal(hexa_grid.face_normal(0), (0.0, 0.0, -1.0))

   # check that planar approximation of last face is about the same as the original
   fi = hexa_grid.face_count - 1
   assert_array_almost_equal(hexa_grid.planar_face_points(fi),
                             hexa_grid.points_cached[hexa_grid.node_indices_for_face(fi)],
                             decimal = 3)

   # construct a Delauney triangulation of the last face
   triangulated = hexa_grid.face_triangulation(fi, local_nodes = True)
   assert triangulated.shape == (2, 3)
   assert np.all(triangulated.flatten() < 4)
   assert np.all(np.unique(triangulated) == (0, 1, 2, 3))
   # also test the triangulation using the global node indices, for all the faces of the last cell
   for face_index in hexa_grid.face_indices_for_cell(hexa_grid.cell_count - 1):
      triangulated = hexa_grid.face_triangulation(face_index, local_nodes = False)
      assert triangulated.shape == (2, 3)

   # check the area of a middle face (the example model has unit area faces)
   assert maths.isclose(hexa_grid.area_of_face(hexa_grid.face_count // 2), 1.0, rel_tol = 1.0e-3)

   # compare properties
   ijk_pc = ijk_grid.extract_property_collection()
   hexa_pc = hexa_grid.extract_property_collection()
   assert ijk_pc is not None and hexa_pc is not None
   assert ijk_pc.number_of_parts() <= hexa_pc.number_of_parts()  # hexa grid has extra active array in this test
   for ijk_part in ijk_pc.parts():
      p_title = ijk_pc.citation_title_for_part(ijk_part)
      hexa_part = hexa_pc.singleton(citation_title = p_title)
      assert hexa_part is not None
      assert hexa_part != ijk_part  # properties are different objects with distinct uuids
      assert hexa_pc.continuous_for_part(hexa_part) == ijk_pc.continuous_for_part(ijk_part)
      ijk_array = ijk_pc.cached_part_array_ref(ijk_part)
      hexa_array = hexa_pc.cached_part_array_ref(hexa_part)
      assert ijk_array is not None and hexa_array is not None
      if hexa_pc.continuous_for_part(hexa_part):
         assert_array_almost_equal(hexa_array.flatten(), ijk_array.flatten())
      else:
         assert np.all(hexa_array.flatten() == ijk_array.flatten())

   # test TetraGrid.from_unstructured_cell is as expected for a hexahedral cell
   tetra = rug.TetraGrid.from_unstructured_cell(hexa_grid, hexa_grid.cell_count // 2)
   tetra.check_indices()
   tetra.check_tetra()
   assert tetra.cell_count == 12  # 2 tetrahedra per hexa face
   assert tetra.node_count == 9  # 8 nodes of hexa cell plus centre point
   assert tetra.face_count == 6 * 2 + 12 + 6
   # check total volume of tetra grid version of cell
   assert maths.isclose(tetra.grid_volume(), 1.0, rel_tol = 1.0e-3)


def test_tetra_grid(tmp_path):

   epc = os.path.join(tmp_path, 'tetra_test.epc')
   model = rq.new_model(epc)
   crs = rqc.Crs(model)
   crs.create_xml()

   # create an empty TetraGrid
   tetra = rug.TetraGrid(model, title = 'star')
   assert tetra.cell_shape == 'tetrahedral'

   # hand craft all attribute data
   tetra.crs_uuid = model.uuid(obj_type = 'LocalDepth3dCrs')
   assert tetra.crs_uuid is not None
   assert bu.matching_uuids(tetra.crs_uuid, crs.uuid)
   tetra.set_cell_count(5)
   # faces
   tetra.face_count = 16
   tetra.faces_per_cell_cl = np.arange(4, 4 * 5 + 1, 4, dtype = int)
   tetra.faces_per_cell = np.empty(20, dtype = int)
   tetra.faces_per_cell[:4] = (0, 1, 2, 3)  # cell 0
   tetra.faces_per_cell[4:8] = (0, 4, 5, 6)  # cell 1
   tetra.faces_per_cell[8:12] = (1, 7, 8, 9)  # cell 2
   tetra.faces_per_cell[12:16] = (2, 10, 11, 12)  # cell 3
   tetra.faces_per_cell[16:] = (3, 13, 14, 15)  # cell 4
   # nodes
   tetra.node_count = 8
   tetra.nodes_per_face_cl = np.arange(3, 3 * 16 + 1, 3, dtype = int)
   tetra.nodes_per_face = np.empty(48, dtype = int)
   # internal faces (cell 0)
   tetra.nodes_per_face[:3] = (0, 1, 2)  # face 0
   tetra.nodes_per_face[3:6] = (0, 3, 1)  # face 1
   tetra.nodes_per_face[6:9] = (1, 3, 2)  # face 2
   tetra.nodes_per_face[9:12] = (2, 3, 0)  # face 3
   # external faces (cell 1)
   tetra.nodes_per_face[12:15] = (0, 1, 4)  # face 4
   tetra.nodes_per_face[15:18] = (1, 2, 4)  # face 5
   tetra.nodes_per_face[18:21] = (2, 0, 4)  # face 6
   # external faces (cell 2)
   tetra.nodes_per_face[21:24] = (0, 3, 5)  # face 7
   tetra.nodes_per_face[24:27] = (3, 1, 5)  # face 8
   tetra.nodes_per_face[27:30] = (1, 0, 5)  # face 9
   # external faces (cell 3)
   tetra.nodes_per_face[30:33] = (1, 3, 6)  # face 10
   tetra.nodes_per_face[33:36] = (3, 2, 6)  # face 11
   tetra.nodes_per_face[36:39] = (2, 1, 6)  # face 12
   # external faces (cell 4)
   tetra.nodes_per_face[39:42] = (2, 3, 7)  # face 10
   tetra.nodes_per_face[42:45] = (3, 0, 7)  # face 11
   tetra.nodes_per_face[45:] = (0, 2, 7)  # face 12
   # face handedness
   tetra.cell_face_is_right_handed = np.zeros(20, dtype = bool)  # False for all faces for external cells (1 to 4)
   tetra.cell_face_is_right_handed[:4] = True  # True for all faces of internal cell (0)
   # points
   tetra.points_cached = np.zeros((8, 3))
   # internal cell (0) points
   half_edge = 36.152
   one_over_root_two = 1.0 / maths.sqrt(2.0)
   tetra.points_cached[0] = (-half_edge, 0.0, -half_edge * one_over_root_two)
   tetra.points_cached[1] = (half_edge, 0.0, -half_edge * one_over_root_two)
   tetra.points_cached[2] = (0.0, half_edge, half_edge * one_over_root_two)
   tetra.points_cached[3] = (0.0, -half_edge, half_edge * one_over_root_two)
   # project remaining nodes outwards
   for fi, o_node in enumerate((3, 2, 0, 1)):
      fc = tetra.face_centre_point(fi)
      tetra.points_cached[4 + fi] = fc - (tetra.points_cached[o_node] - fc)

   # basic validity check
   tetra.check_tetra()

   # write arrays, create xml and store model
   tetra.write_hdf5()
   tetra.create_xml()
   model.store_epc()

   # re-open model and establish grid
   model = rq.Model(epc)
   assert model is not None
   tetra_uuid = model.uuid(obj_type = 'UnstructuredGridRepresentation', title = 'star')
   assert tetra_uuid is not None
   tetra = rug.TetraGrid(model, uuid = tetra_uuid)
   assert tetra is not None
   # perform basic checks
   assert tetra.cell_count == 5
   assert tetra.cell_shape == 'tetrahedral'
   tetra.check_tetra()

   # test volume calculation
   expected_cell_volume = ((2.0 * half_edge)**3) / (6.0 * maths.sqrt(2.0))
   for cell in range(tetra.cell_count):
      assert maths.isclose(tetra.volume(cell), expected_cell_volume, rel_tol = 1.0e-3)
   assert maths.isclose(tetra.grid_volume(), 5.0 * expected_cell_volume)

   # test face area
   expected_area = maths.sqrt(3.0 * half_edge * (half_edge**3))
   area = tetra.area_of_face(0)
   assert maths.isclose(area, expected_area, rel_tol = 1.0e-3)

   # test internal / external face lists
   assert np.all(tetra.external_face_indices() == np.arange(4, 16, dtype = int))
   inactive_mask = np.zeros(5, dtype = bool)
   assert np.all(tetra.external_face_indices_for_masked_cells(inactive_mask) == tetra.external_face_indices())
   assert np.all(tetra.internal_face_indices_for_masked_cells(inactive_mask) == np.arange(4, dtype = int))
   # mask out central cell
   inactive_mask[0] = True
   assert len(tetra.external_face_indices_for_masked_cells(inactive_mask)) == tetra.face_count
   assert len(tetra.internal_face_indices_for_masked_cells(inactive_mask)) == 0
