import math as maths
import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import resqpy.model as rq
import resqpy.olio.uuid as bu
import resqpy.property as rqp
import resqpy.unstructured as rug
from resqpy.crs import Crs
# from resqpy.property import property_kind_and_facet_from_keyword, guess_uom


# ---- Test writing properties on Tetra grids, per-cell and per-node ---

def add_tetra_mesh(model):
    # add a tetra mesh to the model:
    #     the "star" mesh definition is copied from unit_tests/unstructured/test_unstructured.py
    # 
    tetra = rug.TetraGrid(model, title = 'star')
    assert tetra.cell_shape == 'tetrahedral'
    crs = Crs(model)
    crs.create_xml()

    # hand craft all attribute data
    tetra.crs_uuid = model.uuid(obj_type = 'LocalDepth3dCrs')
    assert tetra.crs_uuid is not None
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
    return tetra

def test_properties_on_tetra_grid(tmp_path):

    epc = os.path.join(tmp_path, 'tetra_test_prop.epc')
    model = rq.new_model(epc)
    tetra = add_tetra_mesh(model)

    # write arrays, create xml and store model
    tetra.write_hdf5()
    tetra.create_xml()

    # tetra.points_cached[0]
    poro_per_cell = np.random.rand(tetra.cell_count) * 0.5
    temp_per_vertex = np.random.rand(tetra.node_count) * 100 + 50

    _ = rqp.Property.from_array(model,
                                temp_per_vertex,
                                source_info = 'mock data',
                                keyword = 'Temperature',
                                support_uuid = tetra.uuid,
                                property_kind = 'thermodynamic temperature',
                                indexable_element = 'nodes',
                                uom = 'degC')

    _ = rqp.Property.from_array(model,
                                poro_per_cell,
                                source_info = 'mock data',
                                keyword = 'Porosity',
                                support_uuid = tetra.uuid,
                                property_kind = 'porosity',
                                indexable_element = 'cells',
                                uom = 'm3/m3')

    model.store_epc()



    model = rq.Model(epc)
    assert model is not None

    #
    # read mesh:  vertex positions and cell/tetrahedra definitions
    #
    tetra_uuid = model.uuid(obj_type = 'UnstructuredGridRepresentation', title = 'star')
    assert tetra_uuid is not None
    tetra = rug.TetraGrid(model, uuid = tetra_uuid)
    assert tetra is not None
    assert tetra.cell_shape == 'tetrahedral'
    
    # cells = np.array( [ tetra.distinct_node_indices_for_cell(i) for i in range(tetra.cell_count) ]  ) # cell indices are read using this function(?)
    tetra.check_tetra()

    #
    # read properties
    #

    temp_uuid = model.uuid(title = 'Temperature')
    assert temp_uuid is not None
    temp_prop = rqp.Property(model, uuid = temp_uuid)
    assert temp_prop.array_ref().shape[0] == tetra.node_count
    assert temp_prop.uom() == 'degC'
    assert temp_prop.indexable_element() == 'nodes'
    assert_array_almost_equal(temp_prop.array_ref(), temp_per_vertex )

    poro_uuid = model.uuid(title = 'Porosity')
    assert poro_uuid is not None
    poro_prop = rqp.Property(model, uuid = poro_uuid)
    assert poro_prop.array_ref().shape[0] == tetra.cell_count
    assert poro_prop.uom() == 'm3/m3'
    assert poro_prop.indexable_element() == 'cells'
    assert_array_almost_equal(poro_prop.array_ref(), poro_per_cell )


