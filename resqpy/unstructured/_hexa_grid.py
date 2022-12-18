"""HexaGrid class module."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.grid as grr
import resqpy.olio.volume as vol
import resqpy.property as rqp
import resqpy.unstructured
import resqpy.unstructured._unstructured_grid as rug


class HexaGrid(rug.UnstructuredGrid):
    """Class for unstructured grids where every cell is hexahedral (faces may be degenerate)."""

    def __init__(self,
                 parent_model,
                 uuid = None,
                 find_properties = True,
                 cache_geometry = False,
                 title = None,
                 originator = None,
                 extra_metadata = {}):
        """Creates a new resqpy HexaGrid object (RESQML UnstructuredGrid with cell shape hexahedral)

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
           a newly created HexaGrid object
        """

        super().__init__(parent_model = parent_model,
                         uuid = uuid,
                         find_properties = find_properties,
                         geometry_required = True,
                         cache_geometry = cache_geometry,
                         cell_shape = 'hexahedral',
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)

        if self.root is not None:
            assert grr.grid_flavour(self.root) == 'HexaGrid'
            self.check_hexahedral()

        self.grid_representation = 'HexaGrid'  #: flavour of grid; not much used

    @classmethod
    def from_unsplit_grid(cls,
                          parent_model,
                          grid_uuid,
                          inherit_properties = True,
                          title = None,
                          extra_metadata = {},
                          write_active = None):
        """Creates a new (unstructured) HexaGrid from an existing resqpy unsplit (IJK) Grid without K gaps.

        arguments:
           parent_model (model.Model object): the model which this grid is part of
           grid_uuid (uuid.UUID): the uuid of an IjkGridRepresentation from which the hexa grid will be created
           inherit_properties (boolean, default True): if True, properties will be created for the new grid
           title (str, optional): citation title for the new grid
           extra_metadata (dict, optional): dictionary of extra metadata items to add to the grid
           write_active (boolean, optional): if True (or None and inactive property is established) then an
              active cell property is created (in addition to any inherited properties)

        returns:
           a newly created HexaGrid object

        note:
           this method includes the writing of hdf5 data, creation of xml for the new grid and adding it as a part
        """

        import resqpy.grid as grr

        # establish existing IJK grid
        ijk_grid = grr.Grid(parent_model, uuid = grid_uuid, find_properties = inherit_properties)
        assert ijk_grid is not None
        assert not ijk_grid.has_split_coordinate_lines, 'IJK grid has split coordinate lines (faults)'
        assert not ijk_grid.k_gaps, 'IJK grid has K gaps'
        ijk_grid.cache_all_geometry_arrays()
        ijk_points = ijk_grid.points_ref(masked = False)
        if title is None:
            title = ijk_grid.title

        # make empty unstructured hexa grid
        hexa_grid = cls(parent_model, title = title, extra_metadata = extra_metadata)

        # derive hexa grid attributes from ijk grid
        hexa_grid.crs_uuid = ijk_grid.crs_uuid
        hexa_grid.set_cell_count(ijk_grid.cell_count())
        if ijk_grid.inactive is not None:
            hexa_grid.inactive = ijk_grid.inactive.reshape((hexa_grid.cell_count,))
            hexa_grid.all_inactive = np.all(hexa_grid.inactive)
            if hexa_grid.all_inactive:
                log.warning(f'all cells marked as inactive for unstructured hexa grid {hexa_grid.title}')
        else:
            hexa_grid.all_inactive = False

        # inherit points (nodes) in IJK grid order, ie. K cycling fastest, then I, then J
        hexa_grid.points_cached = ijk_points.reshape((-1, 3))

        # setup faces per cell
        # ordering of faces (in nodes per face): all K faces, then all J faces, then all I faces
        # within J faces, ordering is all of J- faces for J = 0 first, then increasing planes in J
        # similarly for I faces
        nk_plus_1 = ijk_grid.nk + 1
        nj_plus_1 = ijk_grid.nj + 1
        ni_plus_1 = ijk_grid.ni + 1
        k_face_count = nk_plus_1 * ijk_grid.nj * ijk_grid.ni
        j_face_count = ijk_grid.nk * nj_plus_1 * ijk_grid.ni
        i_face_count = ijk_grid.nk * ijk_grid.nj * ni_plus_1
        kj_face_count = k_face_count + j_face_count
        hexa_grid.face_count = k_face_count + j_face_count + i_face_count
        hexa_grid.faces_per_cell_cl = 6 * (1 + np.arange(hexa_grid.cell_count, dtype = int))  # 6 faces per cell
        hexa_grid.faces_per_cell = np.empty(6 * hexa_grid.cell_count, dtype = int)
        arange = np.arange(hexa_grid.cell_count, dtype = int)
        hexa_grid.faces_per_cell[0::6] = arange  # K- faces
        hexa_grid.faces_per_cell[1::6] = ijk_grid.nj * ijk_grid.ni + arange  # K+ faces
        nki = ijk_grid.nk * ijk_grid.ni
        nkj = ijk_grid.nk * ijk_grid.nj
        # todo: vectorise following for loop
        for cell in range(hexa_grid.cell_count):
            k, j, i = ijk_grid.denaturalized_cell_index(cell)
            j_minus_face = k_face_count + nki * j + ijk_grid.ni * k + i
            hexa_grid.faces_per_cell[6 * cell + 2] = j_minus_face  # J- face
            hexa_grid.faces_per_cell[6 * cell + 3] = j_minus_face + nki  # J+ face
            i_minus_face = kj_face_count + nkj * i + ijk_grid.nj * k + j
            hexa_grid.faces_per_cell[6 * cell + 4] = i_minus_face  # I- face
            hexa_grid.faces_per_cell[6 * cell + 5] = i_minus_face + nkj  # I+ face

        # setup nodes per face, clockwise when viewed from negative side of face if ijk handedness matches xyz handedness
        # ordering of nodes in points array is as for the IJK grid
        hexa_grid.node_count = hexa_grid.points_cached.shape[0]
        assert hexa_grid.node_count == (ijk_grid.nk + 1) * (ijk_grid.nj + 1) * (ijk_grid.ni + 1)
        hexa_grid.nodes_per_face_cl = 4 * (1 + np.arange(hexa_grid.face_count, dtype = int))  # 4 nodes per face
        hexa_grid.nodes_per_face = np.empty(4 * hexa_grid.face_count, dtype = int)
        # todo: vectorise for loops
        # K faces
        face_base = 0
        for k in range(nk_plus_1):
            for j in range(ijk_grid.nj):
                for i in range(ijk_grid.ni):
                    hexa_grid.nodes_per_face[face_base] = (k * nj_plus_1 + j) * ni_plus_1 + i  # ip 0, jp 0
                    hexa_grid.nodes_per_face[face_base + 1] = (k * nj_plus_1 + j + 1) * ni_plus_1 + i  # ip 0, jp 1
                    hexa_grid.nodes_per_face[face_base + 2] = (k * nj_plus_1 + j + 1) * ni_plus_1 + i + 1  # ip 1, jp 1
                    hexa_grid.nodes_per_face[face_base + 3] = (k * nj_plus_1 + j) * ni_plus_1 + i + 1  # ip 1, jp 0
                    face_base += 4
        # J faces
        assert face_base == 4 * k_face_count
        for j in range(nj_plus_1):
            for k in range(ijk_grid.nk):
                for i in range(ijk_grid.ni):
                    hexa_grid.nodes_per_face[face_base] = (k * nj_plus_1 + j) * ni_plus_1 + i  # ip 0, kp 0
                    hexa_grid.nodes_per_face[face_base + 1] = (k * nj_plus_1 + j) * ni_plus_1 + i + 1  # ip 1, kp 0
                    hexa_grid.nodes_per_face[face_base +
                                             2] = ((k + 1) * nj_plus_1 + j) * ni_plus_1 + i + 1  # ip 1, kp 1
                    hexa_grid.nodes_per_face[face_base + 3] = ((k + 1) * nj_plus_1 + j) * ni_plus_1 + i  # ip 0, kp 1
                    face_base += 4
        # I faces
        assert face_base == 4 * kj_face_count
        for i in range(ni_plus_1):
            for k in range(ijk_grid.nk):
                for j in range(ijk_grid.nj):
                    hexa_grid.nodes_per_face[face_base] = (k * nj_plus_1 + j) * ni_plus_1 + i  # jp 0, kp 0
                    hexa_grid.nodes_per_face[face_base + 1] = ((k + 1) * nj_plus_1 + j) * ni_plus_1 + i  # jp 0, kp 1
                    hexa_grid.nodes_per_face[face_base +
                                             2] = ((k + 1) * nj_plus_1 + j + 1) * ni_plus_1 + i  # jp 1, kp 1
                    hexa_grid.nodes_per_face[face_base + 3] = (k * nj_plus_1 + j + 1) * ni_plus_1 + i  # jp 1, kp 0
                    face_base += 4
        assert face_base == 4 * hexa_grid.face_count

        # set cell face is right handed
        # todo: check Energistics documents for meaning of cell face is right handed
        # here the assumption is clockwise ordering of nodes viewed from within cell means 'right handed'
        hexa_grid.cell_face_is_right_handed = np.zeros(6 * hexa_grid.cell_count,
                                                       dtype = bool)  # initially set to left handed
        # if IJK grid's ijk handedness matches the xyz handedness, then set +ve faces to right handed; else -ve faces
        if ijk_grid.off_handed():
            hexa_grid.cell_face_is_right_handed[0::2] = True  # negative faces are right handed
        else:
            hexa_grid.cell_face_is_right_handed[1::2] = True  # positive faces are right handed

        hexa_grid.write_hdf5(write_active = write_active)
        hexa_grid.create_xml(write_active = write_active)

        if inherit_properties:
            ijk_pc = ijk_grid.extract_property_collection()
            hexa_pc = rqp.PropertyCollection(support = hexa_grid)
            for part in ijk_pc.parts():
                count = ijk_pc.count_for_part(part)
                hexa_part_shape = (hexa_grid.cell_count,) if count == 1 else (hexa_grid.cell_count, count)
                hexa_pc.add_cached_array_to_imported_list(
                    ijk_pc.cached_part_array_ref(part).reshape(hexa_part_shape),
                    'inherited from grid ' + str(ijk_grid.title),
                    ijk_pc.citation_title_for_part(part),
                    discrete = not ijk_pc.continuous_for_part(part),
                    uom = ijk_pc.uom_for_part(part),
                    time_index = ijk_pc.time_index_for_part(part),
                    null_value = ijk_pc.null_value_for_part(part),
                    property_kind = ijk_pc.property_kind_for_part(part),
                    local_property_kind_uuid = ijk_pc.local_property_kind_uuid(part),
                    facet_type = ijk_pc.facet_type_for_part(part),
                    facet = ijk_pc.facet_for_part(part),
                    realization = ijk_pc.realization_for_part(part),
                    indexable_element = ijk_pc.indexable_for_part(part),
                    count = count,
                    const_value = ijk_pc.constant_value_for_part(part))
                # todo: patch min & max values if present in ijk part
                hexa_pc.write_hdf5_for_imported_list()
                hexa_pc.create_xml_for_imported_list_and_add_parts_to_model(
                    support_uuid = hexa_grid.uuid,
                    time_series_uuid = ijk_pc.time_series_uuid_for_part(part),
                    string_lookup_uuid = ijk_pc.string_lookup_uuid_for_part(part),
                    extra_metadata = ijk_pc.extra_metadata_for_part(part))

        return hexa_grid

    def check_hexahedral(self):
        """Checks that each cell has 6 faces and each face has 4 nodes.

        notes:
           currently only performs a cursory check, without checking nodes are shared;
           assumes that degenerate faces still have four nodes identified
        """

        assert self.cell_shape == 'hexahedral'
        self.cache_all_geometry_arrays()
        assert self.faces_per_cell_cl is not None and self.nodes_per_face_cl is not None
        assert self.faces_per_cell_cl[0] == 6 and np.all(self.faces_per_cell_cl[1:] - self.faces_per_cell_cl[:-1] == 6)
        assert self.nodes_per_face_cl[0] == 4 and np.all(self.nodes_per_face_cl[1:] - self.nodes_per_face_cl[:-1] == 4)

    def corner_points(self, cell):
        """Returns corner points (nodes) of a single cell.

        arguments:
           cell (int): the index of the cell for which the corner points are required

        returns:
           numpy float array of shape (8, 3) being the xyz points of 8 nodes defining a single hexahedral cell

        note:
           if this hexa grid has been created using the from_unsplit_grid class method, then the result can be
           reshaped to (2, 2, 2, 3) for corner points compatible with those used by the Grid class
        """

        self.cache_all_geometry_arrays()
        return self.points_cached[self.distinct_node_indices_for_cell(cell)]

    def volume(self, cell, required_uom = None):
        """Returns the volume of a single cell.

        arguments:
           cell (int): the index of the cell for which the volume is required

        returns:
           float being the volume of the hexahedral cell

        note:
            if required_uom is not specified, returned units will be cube of crs units if xy & z are the same
            and either 'm' or 'ft', otherwise 'm3' will be used
        """

        self._set_crs_handedness()
        apex = self.cell_centre_point(cell)
        v = 0.0
        faces, handednesses = self.face_indices_and_handedness_for_cell(cell)

        for face_index, handedness in zip(faces, handednesses):
            nodes = self.node_indices_for_face(face_index)
            abcd = self.points_cached[nodes]
            assert abcd.shape == (4, 3)
            v += vol.pyramid_volume(apex,
                                    abcd[0],
                                    abcd[1],
                                    abcd[2],
                                    abcd[3],
                                    crs_is_right_handed = (self.crs_is_right_handed == handedness))

        return self.adjusted_volume(v, required_uom = required_uom)

    # todo: add hexahedral specific method for centre_point()?
    # todo: also add other methods equivalent to those in Grid class
