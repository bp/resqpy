"""PyramidGrid class module."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.grid as grr
import resqpy.olio.volume as vol
import resqpy.unstructured
import resqpy.unstructured._unstructured_grid as rug


class PyramidGrid(rug.UnstructuredGrid):
    """Class for unstructured grids where every cell is a quadrilateral pyramid."""

    def __init__(self,
                 parent_model,
                 uuid = None,
                 find_properties = True,
                 cache_geometry = False,
                 title = None,
                 originator = None,
                 extra_metadata = {}):
        """Creates a new resqpy PyramidGrid object (RESQML UnstructuredGrid with cell shape pyramidal)

        arguments:
           parent_model (model.Model object): the model which this grid is part of
           uuid (uuid.UUID, optional): if present, the new grid object is populated from the RESQML object
           find_properties (boolean, default True): if True and uuid is present, a
              grid property collection is instantiated as an attribute, holding properties for which
              this grid is the supporting representation
           cache_geometry (boolean, default False): if True and uuid is present, all the geometry arrays
              are loaded into attributes of the new grid object
           title (str, optional): citation title for new grid; ignored if uuid is present
           originator (str, optional): name of person creating the grid; defaults to login id;
              ignored if uuid is present
           extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid;
              ignored if uuid is present

        returns:
           a newly created PyramidGrid object
        """

        super().__init__(parent_model = parent_model,
                         uuid = uuid,
                         find_properties = find_properties,
                         geometry_required = True,
                         cache_geometry = cache_geometry,
                         cell_shape = 'pyramidal',
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)

        if self.root is not None:
            assert grr.grid_flavour(self.root) == 'PyramidGrid'
            self.check_pyramidal()

        self.grid_representation = 'PyramidGrid'  #: flavour of grid; not much used

    def check_pyramidal(self):
        """Checks that each cell has 5 faces and each face has 3 or 4 nodes.

        note:
           currently only performs a cursory check, without checking nodes are shared or that there is exactly one
           quadrilateral face
        """

        assert self.cell_shape == 'pyramidal'
        self.cache_all_geometry_arrays()
        assert self.faces_per_cell_cl is not None and self.nodes_per_face_cl is not None
        assert self.faces_per_cell_cl[0] == 5 and np.all(self.faces_per_cell_cl[1:] - self.faces_per_cell_cl[:-1] == 5)
        nodes_per_face_count = np.empty(self.face_count)
        nodes_per_face_count[0] = self.nodes_per_face_cl[0]
        nodes_per_face_count[1:] = self.nodes_per_face_cl[1:] - self.nodes_per_face_cl[:-1]
        assert np.all(np.logical_or(nodes_per_face_count == 3, nodes_per_face_count == 4))

    def face_indices_for_cell(self, cell):
        """Returns numpy list of face indices for a single cell.

        arguments:
           cell (int): the index of the cell for which face indices are required

        returns:
           numpy int array of shape (5,) being the face indices of each of the 5 faces for the cell; the first
           index in the array is for the quadrilateral face

        note:
           the face indices are used when accessing the nodes per face data and can also be used to identify
           shared faces
        """

        faces = super().face_indices_for_cell(cell)
        assert len(faces) == 5
        result = -np.ones(5, dtype = int)
        i = 1
        for f in range(5):
            nc = self.node_count_for_face(faces[f])
            if nc == 3:
                assert i < 5, 'too many triangular faces for cell in pyramid grid'
                result[i] = faces[f]
                i += 1
            else:
                assert nc == 4, 'pyramid grid includes a face that is neither triangle nor quadrilateral'
                assert result[0] == -1, 'more than one quadrilateral face for cell in pyramid grid'
                result[0] = faces[f]
        return result

    def face_indices_and_handedness_for_cell(self, cell):
        """Returns numpy list of face indices for a single cell, and numpy boolean list of face right handedness.

        arguments:
           cell (int): the index of the cell for which face indices are required

        returns:
           numpy int array of shape (5,), numpy boolean array of shape (5, ):
           being the face indices of each of the 5 faces for the cell, and the right handedness (clockwise order)
           of the face nodes when viewed from within the cell; the first entry in each list is for the
           quadrilateral face

        note:
           the face indices are used when accessing the nodes per face data and can also be used to identify
           shared faces; the handedness (clockwise or anti-clockwise ordering of nodes) is significant for
           some processing of geometry such as volume calculations
        """

        faces, handednesses = super().face_indices_and_handedness_for_cell(cell)
        assert len(faces) == 5 and len(handednesses) == 5
        f_result = -np.ones(5, dtype = int)
        h_result = np.empty(5, dtype = bool)
        i = 1
        for f in range(5):
            nc = self.node_count_for_face(faces[f])
            if nc == 3:
                assert i < 5, 'too many triangular faces for cell in pyramid grid'
                f_result[i] = faces[f]
                h_result[i] = handednesses[f]
                i += 1
            else:
                assert nc == 4, 'pyramid grid includes a face that is neither triangle nor quadrilateral'
                assert f_result[0] == -1, 'more than one quadrilateral face for cell in pyramid grid'
                f_result[0] = faces[f]
                h_result[0] = handednesses[f]
        return f_result, h_result

    def volume(self, cell, required_uom = None):
        """Returns the volume of a single cell.

        arguments:
           cell (int): the index of the cell for which the volume is required

        returns:
           float being the volume of the pyramidal cell

        note:
            if required_uom is not specified, returned units will be cube of crs units if xy & z are the same
            and either 'm' or 'ft', otherwise 'm3' will be used
        """

        self._set_crs_handedness()
        self.cache_all_geometry_arrays()
        faces, hands = self.face_indices_and_handedness_for_cell(cell)
        nodes = self.distinct_node_indices_for_cell(cell)
        base_nodes = self.node_indices_for_face(faces[0])
        for node in nodes:
            if node not in base_nodes:
                apex_node = node
                break
        else:
            raise Exception('apex node not found for cell in pyramid grid')
        apex = self.points_cached[apex_node]
        abcd = self.points_cached[base_nodes]

        v = vol.pyramid_volume(apex,
                               abcd[0],
                               abcd[1],
                               abcd[2],
                               abcd[3],
                               crs_is_right_handed = (self.crs_is_right_handed == hands[0]))

        return self.adjusted_volume(v, required_uom = required_uom)

    # todo: add pyramidal specific method for centre_point()
