import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.model as rq
import resqpy.grid as grr
import resqpy.unstructured as rqu


def test_hexa_grid_from_grid(example_model_with_properties):

   model = example_model_with_properties
   # model = rq.Model('/users/andy/bifröst/bc/u_test.epc', copy_from = example_model_with_properties.epc_file)

   ijk_grid_uuid = model.uuid(obj_type = 'IjkGridRepresentation')
   assert ijk_grid_uuid is not None

   # create an unstructured grid with hexahedral cells from an unsplit IJK grid
   hexa = rqu.HexaGrid.from_unsplit_grid(model, ijk_grid_uuid, inherit_properties = False, title = 'HEXA')
   hexa.write_hdf5()
   hexa.create_xml()

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

   # instantiate ijk grid and compare hexa grid with it
   ijk_grid = grr.any_grid(model, uuid = ijk_grid_uuid)
   assert ijk_grid is not None

   assert hexa_grid.cell_count == ijk_grid.cell_count()
   assert hexa_grid.node_count == (ijk_grid.nk + 1) * (ijk_grid.nj + 1) * (ijk_grid.ni + 1)
   assert hexa_grid.face_count == ((ijk_grid.nk + 1) * ijk_grid.nj * ijk_grid.ni + ijk_grid.nk *
                                   (ijk_grid.nj + 1) * ijk_grid.ni + ijk_grid.nk * ijk_grid.nj * (ijk_grid.ni + 1))

   # compare centre points of cells (not sure if these would be coincident for irregular shaped cells)
   hexa_centres = hexa_grid.centre_point()
   print(hexa_centres)
   ijk_centres = ijk_grid.centre_point()
   print(ijk_centres)
   assert_array_almost_equal(hexa_centres, ijk_centres.reshape((-1, 3)), decimal = 3)

   # check that face and node indices are in range
   hexa_grid.cache_all_geometry_arrays()
   assert np.all(0 <= hexa_grid.faces_per_cell)
   assert np.all(hexa_grid.faces_per_cell < hexa_grid.face_count)
   assert np.all(0 <= hexa_grid.nodes_per_face)
   assert np.all(hexa_grid.nodes_per_face < hexa_grid.node_count)
   assert len(hexa_grid.faces_per_cell_cl) == hexa_grid.cell_count
   assert len(hexa_grid.nodes_per_face_cl) == hexa_grid.face_count

   # have a look at handedness of cell faces
   assert len(hexa_grid.cell_face_is_right_handed) == 6 * hexa_grid.cell_count
   # following assertion only applies to HexaGrid built from_unsplit_grid()
   assert np.count_nonzero(hexa_grid.cell_face_is_right_handed) == 3 * hexa_grid.cell_count  # half are right handed
