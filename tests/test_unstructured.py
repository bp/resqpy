import pytest
import math as maths
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.unstructured as rqu
import resqpy.olio.uuid as bu


def test_hexa_grid_from_grid(example_model_with_properties):

   model = example_model_with_properties
   # model = rq.Model('/users/andy/bifröst/bc/u_test.epc', copy_from = example_model_with_properties.epc_file)

   ijk_grid_uuid = model.uuid(obj_type = 'IjkGridRepresentation')
   assert ijk_grid_uuid is not None

   # create an unstructured grid with hexahedral cells from an unsplit IJK grid (includes write_hdf5 and create_xml)
   hexa = rqu.HexaGrid.from_unsplit_grid(model, ijk_grid_uuid, inherit_properties = True, title = 'HEXA')

   hexa_uuid = hexa.uuid

   epc = model.epc_file
   assert epc

   model.store_epc()

   # re-open model and check hexa grid

   model = rq.Model(epc)

   unstructured_uuids = model.uuids(obj_type = 'UnstructuredGridRepresentation')

   assert unstructured_uuids is not None and len(unstructured_uuids) == 1

   hexa_grid = grr.any_grid(model, uuid = unstructured_uuids[0])
   assert isinstance(hexa_grid, rqu.HexaGrid)

   assert hexa_grid.cell_shape == 'hexahedral'

   hexa_grid.check_hexahedral()

   # instantiate ijk grid and compare hexa grid with it
   ijk_grid = grr.any_grid(model, uuid = ijk_grid_uuid)
   assert ijk_grid is not None

   assert hexa_grid.cell_count == ijk_grid.cell_count()
   assert hexa_grid.active_cell_count() == hexa_grid.cell_count
   assert hexa_grid.node_count == (ijk_grid.nk + 1) * (ijk_grid.nj + 1) * (ijk_grid.ni + 1)
   assert hexa_grid.face_count == ((ijk_grid.nk + 1) * ijk_grid.nj * ijk_grid.ni + ijk_grid.nk *
                                   (ijk_grid.nj + 1) * ijk_grid.ni + ijk_grid.nk * ijk_grid.nj * (ijk_grid.ni + 1))

   assert bu.matching_uuids(hexa_grid.extract_crs_uuid(), ijk_grid.crs_uuid)
   assert hexa_grid.crs_is_right_handed == rqc.Crs(model, uuid = ijk_grid.crs_uuid).is_right_handed_xyz()

   # points arrays should be identical for the two grids
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
