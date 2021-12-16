"""mesh.py: surface class based on resqml standard."""

version = '4th November 2021'

# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company
# GOCAD is also a trademark of Emerson

import logging
import warnings

log = logging.getLogger(__name__)

import math as maths

import numpy as np

import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
from resqpy.olio.xml_namespaces import curly_namespace as ns
from resqpy.olio.zmap_reader import read_mesh
from .base_surface import BaseSurface
from .surface import Surface


class Mesh(BaseSurface):
    """Class covering meshes (lattices: surfaces where points form a 2D grid; RESQML obj_Grid2dRepresentation)."""

    resqml_type = 'Grid2dRepresentation'

    def __init__(self,
                 parent_model,
                 root_node = None,
                 uuid = None,
                 mesh_file = None,
                 mesh_format = None,
                 mesh_flavour = 'explicit',
                 xyz_values = None,
                 nj = None,
                 ni = None,
                 origin = None,
                 dxyz_dij = None,
                 z_values = None,
                 z_supporting_mesh_uuid = None,
                 surface_role = 'map',
                 crs_uuid = None,
                 title = None,
                 originator = None,
                 extra_metadata = None):
        """Initialises a Mesh object from xml, or a regular mesh from arguments.

        arguments:
           parent_model (model.Model object): the model to which this Mesh object will be associated
           root_node (optional): DEPRECATED, use uuid instead; the root node for an obj_Grid2dRepresentation part;
              remaining arguments are ignored if uuid or root_node is not None
           uuid (uuid.UUID, optional): the uuid of an existing RESQML obj_Grid2dRepresentation object from which
              this resqpy Mesh object is populated
           mesh_file (string, optional): file name, required if initialising from an RMS text or zmap+ ascii file
           mesh_format (string, optional): 'rms' or 'zmap', required if initialising from an ascii file
           mesh_flavour (string, default 'explicit'): required flavour when reading from a mesh file; one of:
              'explicit', 'regular' (z values discarded), 'reg&z', 'ref&z'
           xyz_values (numpy int array of shape (nj, ni, 3), optional): can be used to create an explicit
              mesh directly from the full array of points
           nj (int, optional): when generating a regular or 'ref&z' mesh, the number of nodes (NB. not 'cells')
              in the j axis of the regular mesh
           ni (int, optional): the number of nodes in the i axis of the regular or ref&z mesh
           origin (triple float, optional): the xyz origin of the regular mesh; use z value of zero if irrelevant
           dxyz_dij (numpy float array of shape (2, 3), optional): the xyz increment for each step in i and j axes;
              use z increments of zero if not applicable; eg. [[50.0, 0.0, 0.0], [0.0, 50.0, 0.0]] for mesh with
              50 (m or ft) spacing where the I axis aligns with x axis and the J axis aligns with y axis; first of
              the two triplets relates to the I axis
           z_values (numpy int array of shape (nj, ni), optional): z values used when creating a ref&z flavour
              mesh; z_supporting_mesh_uuid must also be supplied
           z_supporting_mesh_uuid (uuid.UUID, optional): used to specify the supporting mesh when creating a
              ref&z or reg&z flavour mesh; z_values must also be supplied
           surface_role (string, default 'map'): 'map' or 'pick'; ignored if root_node is not None
           crs_uuid (uuid.Uuid or string, optional): required if generating a regular mesh, the uuid of the crs
           title (str, optional): the citation title to use for a new mesh;
              ignored if uuid or root_node is not None
           originator (str, optional): the name of the person creating the mesh, defaults to login id;
              ignored if uuid or root_node is not None
           extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the mesh;
              ignored if uuid or root_node is not None

        returns:
           the newly created Mesh object

        notes:
           a mesh is a set of x,y,z (or x,y) points organised into a 2D lattice indexed by j,i; the z values are
           sometimes not applicable and then can be set to zero; 3 flavours of mesh are supported (the RESQML
           standard might allow for others): regular, where a constant xyz delta is applied with each step in i
           and j, starting from an origin, to yield a planar surface; explicit, where the full xyz (or xy) data
           is held as an array; 'ref & z', where another mesh is referred to for xy data, and z values are held
           in an array;
           there are 5 ways to initialise a Mesh object, in order of precedence:
           1. pass root_node to initialise from xml
           2. pass mesh_file, mesh_format and crs_uuid to load an explicit mesh from an ascii file
           3. pass xyz_values and crs_uuid to create an explicit mesh from a numpy array
           4. pass nj, ni, origin, dxyz_dij and crs_uuid to initialise a regular mesh directly
           5. pass z_values, z_supporting_mesh_uuid and crs_uuid to initialise a 'ref & z' mesh
           6. leave all optional arguments as None for an empty Mesh object
        """

        assert surface_role in ['map', 'pick']

        self.surface_role = surface_role
        self.ni = None  # NB: these are the number of nodes (points) in the mesh, unlike 3D grids
        self.nj = None
        self.flavour = None  # 'explicit', 'regular', 'ref&z' or 'reg&z'
        self.full_array = None  # loaded on demand, shape (NJ, NI, 3 or 2) being xyz or xy points at each node
        self.explicit_h5_key_pair = None
        self.regular_origin = None  # xyz of origin of regular mesh
        self.regular_dxyz_dij = None  # numpy array of shape (2, 3) being xyz increment with each step in i, j
        self.ref_uuid = None
        self.ref_mesh = None
        self.ref_z_h5_key_pair = None
        # note: in this class, z values for ref&z meshes are held in the full_array (xy data will be duplicated in memory)
        self.crs_uuid = crs_uuid
        self.represented_interpretation_root = None

        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata,
                         root_node = root_node)

        self.crs_root = None if self.crs_uuid is None else self.model.root_for_uuid(self.crs_uuid)

        if self.root is not None:
            pass

        elif mesh_file and mesh_format and crs_uuid is not None:
            self.__load_from_mesh_file(mesh_file, mesh_flavour, mesh_format, crs_uuid, ni, nj)

        elif xyz_values is not None and crs_uuid is not None:
            self.__load_from_xyz_values(xyz_values)

        else:
            self.__load_from_arguments(ni, nj, origin, dxyz_dij, z_values, z_supporting_mesh_uuid, crs_uuid)

        assert self.crs_uuid is not None
        if not self.title:
            self.title = 'mesh'

    def __load_from_arguments(self, ni, nj, origin, dxyz_dij, z_values, z_supporting_mesh_uuid, crs_uuid):
        assert ni is not None and nj is not None, 'If loading from arguments ni and nj must be provided'

        if z_values is None:
            assert origin is not None and dxyz_dij is not None and crs_uuid is not None, 'origin, dxyz_dij and crs_uuid must be provided for regular mesh'
            self.__regular_mesh_from_arguments(ni, nj, origin, dxyz_dij, z_values)
        else:
            if z_supporting_mesh_uuid is not None:
                self.__refz_from_arguments(ni, nj, z_values, z_supporting_mesh_uuid)
            else:
                assert dxyz_dij is not None and crs_uuid is not None and origin is not None, 'dzy_dij, crs_uuid and origin must be provided for regular and z mesh'
                self.__regz_from_arguments(ni, nj, z_values, origin, dxyz_dij)

    def __regz_from_arguments(self, ni, nj, z_values, origin, dxyz_dij):
        # create a reg&z mesh from arguments
        assert nj > 0 and ni > 0
        assert len(origin) == 3
        assert dxyz_dij.shape == (2, 3)
        assert z_values.shape == (nj, ni) or z_values.shape == (nj * ni,)
        self.nj = nj
        self.ni = ni
        self.regular_origin = origin
        self.regular_dxyz_dij = np.array(dxyz_dij, dtype = float)
        assert self.crs_uuid is not None, 'crs uuid missing'
        self.flavour = 'regular'
        self.full_array = None
        _ = self.full_array_ref()
        self.full_array[..., 2] = z_values.reshape(tuple(self.full_array.shape[:-1]))
        self.flavour = 'reg&z'

    def __refz_from_arguments(self, ni, nj, z_values, z_supporting_mesh_uuid):
        # create a ref&z mesh from arguments
        assert nj > 0 and ni > 0
        assert z_values.shape == (nj, ni) or z_values.shape == (nj * ni,)
        self.flavour = 'ref&z'
        self.nj = nj
        self.ni = ni
        self.ref_uuid = z_supporting_mesh_uuid
        self.ref_mesh = Mesh(self.model, uuid = z_supporting_mesh_uuid)
        assert self.ref_mesh is not None
        assert self.ref_mesh.nj == nj and self.ref_mesh.ni == ni
        self.full_array = self.ref_mesh.full_array_ref().copy()
        self.full_array[..., 2] = z_values.reshape(tuple(self.full_array.shape[:-1]))
        assert self.crs_uuid is not None, 'crs uuid missing'

    def __regular_mesh_from_arguments(self, ni, nj, origin, dxyz_dij, z_values):
        # create a regular mesh from arguments
        assert nj > 0 and ni > 0
        assert len(origin) == 3
        assert dxyz_dij.shape == (2, 3)
        self.flavour = 'regular'
        self.nj = nj
        self.ni = ni
        self.regular_origin = origin
        self.regular_dxyz_dij = np.array(dxyz_dij, dtype = float)
        assert self.crs_uuid is not None, 'crs uuid missing'

    def __load_from_xyz_values(self, xyz_values):
        # create an explicit mesh directly from a numpy array of points
        assert xyz_values.ndim == 3 and xyz_values.shape[2] == 3 and xyz_values.shape[0] > 1 and xyz_values.shape[1] > 1
        self.flavour = 'explicit'
        self.nj = xyz_values.shape[0]
        self.ni = xyz_values.shape[1]
        self.full_array = xyz_values.copy()
        assert self.crs_uuid is not None, 'crs uuid missing'

    def __load_from_mesh_file(self, mesh_file, mesh_flavour, mesh_format, crs_uuid, ni, nj):
        # load an explicit mesh from an ascii file in RMS text or zmap format
        assert mesh_format in ['rms', 'roxar', 'zmap']  # 'roxar' is treated synonymously with 'rms'
        assert mesh_flavour in ['explicit', 'regular', 'reg&z', 'ref&z']
        x, y, z = read_mesh(mesh_file, format = mesh_format)
        self.flavour = mesh_flavour
        self.nj = z.shape[0]
        self.ni = z.shape[1]
        assert self.nj > 1 and self.ni > 1
        self.full_array = np.stack((x, y, z), axis = -1)
        if mesh_flavour != 'explicit':
            min_x = x.flatten()[0]
            max_x = x.flatten()[-1]
            min_y = y.flatten()[0]
            max_y = y.flatten()[-1]
            dxyz_dij = np.array([[(max_x - min_x) /
                                  (self.ni - 1), 0.0, 0.0], [0.0, (max_y - min_y) / (self.nj - 1), 0.0]],
                                dtype = float)
            if mesh_flavour in ['regular', 'reg&z']:
                self.regular_origin = (min_x, min_y, 0.0)
                self.regular_dxyz_dij = dxyz_dij
            elif mesh_flavour == 'ref&z':
                self.ref_mesh = Mesh(self.model,
                                     ni = self.ni,
                                     nj = self.nj,
                                     origin = self.regular_origin,
                                     dxyz_dij = dxyz_dij,
                                     crs_uuid = crs_uuid)
                assert self.ref_mesh is not None
                assert self.ref_mesh.nj == nj and self.ref_mesh.ni == ni
            else:
                log.critical('code failure')
        assert self.crs_uuid is not None, 'crs uuid missing'
        # todo: option to create a regular and ref&z pair instead of an explicit mesh

    def __load_from_xml_regular(self, point_node):
        self.flavour = 'regular'
        origin_node = rqet.find_tag(point_node, 'Origin')
        self.regular_origin = (rqet.find_tag_float(origin_node,
                                                   'Coordinate1'), rqet.find_tag_float(origin_node, 'Coordinate2'),
                               rqet.find_tag_float(origin_node, 'Coordinate3'))
        assert self.regular_origin is not None, 'origin missing in xml for regular mesh (lattice)'
        offset_nodes = rqet.list_of_tag(point_node, 'Offset')  # first occurrence for FastestAxis, ie. I; 2nd for J
        assert len(offset_nodes) == 2, 'missing (or too many) offset nodes in xml for regular mesh (lattice)'
        self.regular_dxyz_dij = np.empty((2, 3))
        for j_or_i in range(2):  # 0 = J, 1 = I
            axial_offset_node = rqet.find_tag(offset_nodes[j_or_i], 'Offset')
            assert axial_offset_node is not None, 'missing offset offset node in xml'
            self.regular_dxyz_dij[1 - j_or_i] = (rqet.find_tag_float(axial_offset_node, 'Coordinate1'),
                                                 rqet.find_tag_float(axial_offset_node, 'Coordinate2'),
                                                 rqet.find_tag_float(axial_offset_node, 'Coordinate3'))
            if not maths.isclose(vec.dot_product(self.regular_dxyz_dij[1 - j_or_i], self.regular_dxyz_dij[1 - j_or_i]),
                                 1.0):
                log.warning('non-orthogonal axes and/or scaling in xml for regular mesh (lattice)')
            spacing_node = rqet.find_tag(offset_nodes[j_or_i], 'Spacing')
            stride = rqet.find_tag_float(spacing_node, 'Value')
            count = rqet.find_tag_int(spacing_node, 'Count')
            assert stride is not None and count is not None, 'missing spacing info in xml'
            assert count == (self.nj, self.ni)[j_or_i] - 1, \
                'unexpected value for count in xml spacing info for regular mesh (lattice)'
            assert stride > 0.0, 'spacing distance is not positive in xml for regular mesh (lattice)'
            self.regular_dxyz_dij[1 - j_or_i] *= stride

    def __load_from_xml_explicit(self, point_node):
        self.flavour = 'explicit'
        self.explicit_h5_key_pair = self.model.h5_uuid_and_path_for_node(point_node, tag = 'Coordinates')
        # load full_array on demand later (see full_array_ref() method)

    def __load_from_xml_refz(self, support_geom_node):
        self.flavour = 'ref&z'
        # assert rqet.node_type(support_geom_node) == 'Point3dFromRepresentationLatticeArray'  # only this supported for now
        self.ref_uuid = rqet.find_nested_tags_text(support_geom_node, ['SupportingRepresentation', 'UUID'])
        assert self.ref_uuid, 'missing supporting representation info in xml for z-value mesh'
        self.ref_mesh = Mesh(self.model, uuid = self.ref_uuid)
        assert self.nj == self.ref_mesh.nj and self.ni == self.ref_mesh.ni  # only this supported for now
        niosr_node = rqet.find_tag(support_geom_node, 'NodeIndicesOnSupportingRepresentation')
        start_value = rqet.find_tag_int(niosr_node, 'StartValue')
        assert start_value == 0, 'only full use of supporting mesh catered for at present'
        offset_nodes = rqet.list_of_tag(niosr_node, 'Offset')  # first occurrence for FastestAxis, ie. I; 2nd for J
        assert len(offset_nodes) == 2, 'missing (or too many) offset nodes in xml for regular mesh (lattice)'
        for j_or_i in range(2):  # 0 = J, 1 = I
            assert rqet.node_type(offset_nodes[j_or_i]) == 'IntegerConstantArray', 'variable step not catered for'
            assert rqet.find_tag_int(offset_nodes[j_or_i], 'Value') == 1, 'step other than 1 not catered for'
            count = rqet.find_tag_int(offset_nodes[j_or_i], 'Count')
            assert count == (self.nj, self.ni)[j_or_i] - 1, \
                'unexpected value for count in xml spacing info for regular mesh (lattice)'

    def __load_from_xml_regz(self, support_geom_node):
        self.flavour = 'reg&z'
        orig_node = rqet.find_tag(support_geom_node, 'Origin')
        self.regular_origin = (rqet.find_tag_float(orig_node,
                                                   'Coordinate1'), rqet.find_tag_float(orig_node, 'Coordinate2'),
                               rqet.find_tag_float(orig_node, 'Coordinate3'))
        assert self.regular_origin is not None, 'origin missing in xml for reg&z mesh (lattice)'
        offset_nodes = rqet.list_of_tag(support_geom_node,
                                        'Offset')  # first occurrence for FastestAxis, ie. I; 2nd for J
        assert len(offset_nodes) == 2, 'missing (or too many) offset nodes in xml for reg&z mesh (lattice)'
        self.regular_dxyz_dij = np.empty((2, 3))
        for j_or_i in range(2):  # 0 = J, 1 = I
            axial_offset_node = rqet.find_tag(offset_nodes[j_or_i], 'Offset')
            assert axial_offset_node is not None, 'missing offset offset node in xml'
            self.regular_dxyz_dij[1 - j_or_i] = (rqet.find_tag_float(axial_offset_node, 'Coordinate1'),
                                                 rqet.find_tag_float(axial_offset_node, 'Coordinate2'),
                                                 rqet.find_tag_float(axial_offset_node, 'Coordinate3'))
            if not maths.isclose(vec.dot_product(self.regular_dxyz_dij[1 - j_or_i], self.regular_dxyz_dij[1 - j_or_i]),
                                 1.0):
                log.warning('non-orthogonal axes and/or scaling in xml for regular mesh (lattice)')
            spacing_node = rqet.find_tag(offset_nodes[j_or_i], 'Spacing')
            stride = rqet.find_tag_float(spacing_node, 'Value')
            count = rqet.find_tag_int(spacing_node, 'Count')
            assert stride is not None and count is not None, 'missing spacing info in xml'
            assert count == (self.nj, self.ni)[j_or_i] - 1, \
                'unexpected value for count in xml spacing info for regular mesh (lattice)'
            assert stride > 0.0, 'spacing distance is not positive in xml for regular mesh (lattice)'
            self.regular_dxyz_dij[1 - j_or_i] *= stride

    def __load_from_xml_basics(self):
        root_node = self.root
        assert root_node is not None
        self.surface_role = rqet.find_tag_text(root_node, 'SurfaceRole')
        ref_node = rqet.find_tag(root_node, 'RepresentedInterpretation')
        if ref_node is not None:
            self.represented_interpretation_root = self.model.referenced_node(ref_node)
        patch_node = rqet.find_tag(root_node, 'Grid2dPatch')
        assert rqet.find_tag_int(patch_node, 'PatchIndex') == 0
        self.ni = rqet.find_tag_int(patch_node, 'FastestAxisCount')
        self.nj = rqet.find_tag_int(patch_node, 'SlowestAxisCount')
        assert self.ni is not None and self.nj is not None, 'mesh extent info missing in xml'
        geom_node = rqet.find_tag(patch_node, 'Geometry')
        assert geom_node is not None, 'geometry missing in mesh xml'
        self.crs_uuid = rqet.find_nested_tags_text(geom_node, ['LocalCrs', 'UUID'])
        assert self.crs_uuid is not None, 'crs reference missing in mesh geometry xml'
        point_node = rqet.find_tag(geom_node, 'Points')
        assert point_node is not None, 'missing Points node in mesh geometry xml'
        flavour = rqet.node_type(point_node)

        return point_node, flavour

    def _load_from_xml(self):
        point_node, flavour = self.__load_from_xml_basics()

        if flavour == 'Point3dLatticeArray':
            self.__load_from_xml_regular(point_node)
        elif flavour == 'Point3dZValueArray':
            # note: only simple, full use of supporting mesh is handled at present
            z_ref_node = rqet.find_tag(point_node, 'ZValues')
            self.ref_z_h5_key_pair = self.model.h5_uuid_and_path_for_node(z_ref_node, tag = 'Values')
            support_geom_node = rqet.find_tag(point_node, 'SupportingGeometry')
            if rqet.node_type(support_geom_node) == 'Point3dFromRepresentationLatticeArray':
                self.__load_from_xml_refz(support_geom_node)
            elif rqet.node_type(support_geom_node) == 'Point3dLatticeArray':
                self.__load_from_xml_regz(support_geom_node)
        elif flavour in ['Point3dHdf5Array', 'Point2dHdf5Array']:
            self.__load_from_xml_explicit(point_node)
        else:
            raise Exception('unrecognised flavour for mesh points')

    def set_represented_interpretation_root(self, interp_root):
        """Makes a note of the xml root of the represented interpretation."""

        self.represented_interpretation_root = interp_root

    def __full_array_ref_explicit(self):
        # load array directly from hdf5 points reference; note: could be xyz or xy data
        assert self.explicit_h5_key_pair is not None, 'h5 key pair not established for mesh'
        try:
            self.model.h5_array_element(self.explicit_h5_key_pair,
                                        cache_array = True,
                                        object = self,
                                        array_attribute = 'full_array',
                                        dtype = 'float')
            # todo: could extend with z values of zero if only xy present?
        except Exception:
            log.exception('hdf5 points failure for mesh')

    def __full_array_ref_regular(self):
        self.full_array = np.empty((self.nj, self.ni, 3))
        x_i0 = np.linspace(0.0, (self.nj - 1) * self.regular_dxyz_dij[1, 0], num = self.nj)
        y_i0 = np.linspace(0.0, (self.nj - 1) * self.regular_dxyz_dij[1, 1], num = self.nj)
        z_i0 = np.linspace(0.0, (self.nj - 1) * self.regular_dxyz_dij[1, 2], num = self.nj)
        x_full = np.linspace(x_i0, x_i0 + (self.ni - 1) * self.regular_dxyz_dij[0, 0], num = self.ni, axis = -1)
        y_full = np.linspace(y_i0, y_i0 + (self.ni - 1) * self.regular_dxyz_dij[0, 1], num = self.ni, axis = -1)
        z_full = np.linspace(z_i0, z_i0 + (self.ni - 1) * self.regular_dxyz_dij[0, 2], num = self.ni, axis = -1)
        self.full_array = np.stack((x_full, y_full, z_full), axis = -1) + self.regular_origin

        if self.flavour == 'reg&z':  # overwrite regular z values with explicitly stored z values
            self.__full_array_ref_regz()

    def __full_array_ref_regz(self):
        assert self.ref_z_h5_key_pair is not None, 'h5 key pair missing for mesh z values'
        try:
            self.model.h5_array_element(self.ref_z_h5_key_pair,
                                        cache_array = True,
                                        object = self,
                                        array_attribute = 'temp_z',
                                        dtype = 'float')
        except Exception:
            log.exception('hdf5 failure for mesh z values')
        self.full_array[..., 2] = self.temp_z
        delattr(self, 'temp_z')

    def __full_array_ref_refz(self):
        # load array from referenced mesh and overwrite z values
        if self.ref_mesh is None:
            self.ref_mesh = Mesh(self.model, uuid = self.ref_uuid)
            assert self.ref_mesh is not None, 'failed to instantiate object for referenced mesh'
        self.full_array = self.ref_mesh.full_array_ref().copy()
        assert self.ref_z_h5_key_pair is not None, 'h5 key pair missing for mesh z values'
        try:
            self.model.h5_array_element(self.ref_z_h5_key_pair,
                                        cache_array = True,
                                        object = self,
                                        array_attribute = 'temp_z',
                                        dtype = 'float')
        except Exception:
            log.exception('hdf5 failure for mesh z values')
        self.full_array[..., 2] = self.temp_z
        #        if np.any(np.isnan(self.temp_z)): log.warning('z values include some NaNs')
        delattr(self, 'temp_z')

    def full_array_ref(self):
        """Populates a full 2D(+1) numpy array of shape (nj, ni, 3) with xyz values, caches and returns.

        note:
           z values may be zero or not applicable when using the mesh as support for properties.
        """

        if self.full_array is not None:
            return self.full_array

        if self.flavour == 'explicit':
            self.__full_array_ref_explicit()
        elif self.flavour in ['regular', 'reg&z']:
            self.__full_array_ref_regular()
        elif self.flavour == 'ref&z':
            self.__full_array_ref_refz()
        else:
            raise Exception(f'unrecognised mesh flavour when fetching full array: {self.flavour}')

        return self.full_array

    def surface(self, quad_triangles = False):
        """Returns a surface object generated from this mesh."""

        return Surface(self.model, mesh = self, quad_triangles = quad_triangles)

    def write_hdf5(self, file_name = None, mode = 'a', use_xy_only = False):
        """Create or append to an hdf5 file, writing datasets for the mesh depending on flavour."""

        if not file_name:
            file_name = self.model.h5_file_name()
        if self.uuid is None:
            self.uuid = bu.new_uuid()
        if self.flavour == 'regular':
            return
        # NB: arrays must have been set up prior to calling this function
        h5_reg = rwh5.H5Register(self.model)
        a = self.full_array_ref()
        if self.flavour == 'explicit':
            if use_xy_only:
                h5_reg.register_dataset(self.uuid, 'points', a[..., :2])  # todo: check what others use here
            else:
                h5_reg.register_dataset(self.uuid, 'points', a)
        elif self.flavour == 'ref&z' or self.flavour == 'reg&z':
            h5_reg.register_dataset(self.uuid, 'zvalues', a[..., 2])
        else:
            log.error('bad mesh flavour when writing hdf5 array')
        h5_reg.write(file_name, mode = mode)

    def __create_xml_regular(self, p_node):

        assert self.regular_origin is not None and self.regular_dxyz_dij is not None

        p_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dLatticeArray')

        self.model.create_solitary_point3d('Origin', p_node, self.regular_origin)  # todo: check xml namespace

        for j_or_i in range(2):  # 0 = J, 1 = I
            dxyz = self.regular_dxyz_dij[1 - j_or_i].copy()
            log.debug('dxyz: ' + str(dxyz))
            d_value = vec.dot_product(dxyz, dxyz)
            assert d_value > 0.0
            d_value = maths.sqrt(d_value)
            dxyz /= d_value
            o_node = rqet.SubElement(p_node, ns['resqml2'] + 'Offset')  # note: 1st Offset is for J axis , 2nd for I
            o_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dOffset')
            o_node.text = '\n'
            self.model.create_solitary_point3d('Offset', o_node, dxyz)
            space_node = rqet.SubElement(o_node, ns['resqml2'] + 'Spacing')
            space_node.set(ns['xsi'] + 'type',
                           ns['resqml2'] + 'DoubleConstantArray')  # nothing else catered for just now
            space_node.text = '\n'
            ov_node = rqet.SubElement(space_node, ns['resqml2'] + 'Value')
            ov_node.set(ns['xsi'] + 'type', ns['xsd'] + 'double')
            ov_node.text = str(d_value)
            oc_node = rqet.SubElement(space_node, ns['resqml2'] + 'Count')
            oc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            if j_or_i:
                oc_node.text = str(self.ni - 1)
            else:
                oc_node.text = str(self.nj - 1)

    def __create_xml_refandz(self, p_node, ext_uuid):
        assert ext_uuid is not None
        assert self.ref_uuid is not None

        p_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dZValueArray')

        sg_node = rqet.SubElement(p_node, ns['resqml2'] + 'SupportingGeometry')

        sg_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dFromRepresentationLatticeArray')
        sg_node.text = '\n'

        niosr_node = rqet.SubElement(sg_node, ns['resqml2'] + 'NodeIndicesOnSupportingRepresentation')
        niosr_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerLatticeArray')
        niosr_node.text = '\n'

        sv_node = rqet.SubElement(niosr_node, ns['resqml2'] + 'StartValue')
        sv_node.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
        sv_node.text = '0'  # no other possibility cater for at present

        for j_or_i in range(2):  # 0 = J, 1 = I
            o_node = rqet.SubElement(niosr_node, ns['resqml2'] + 'Offset')
            o_node.set(ns['xsi'] + 'type',
                       ns['resqml2'] + 'IntegerConstantArray')  # no other possibility cater for at present
            o_node.text = '\n'
            ov_node = rqet.SubElement(o_node, ns['resqml2'] + 'Value')
            ov_node.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
            ov_node.text = '1'  # no other possibility cater for at present
            oc_node = rqet.SubElement(o_node, ns['resqml2'] + 'Count')
            oc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            if j_or_i:
                oc_node.text = str(self.ni - 1)
            else:
                oc_node.text = str(self.nj - 1)

        ref_root = self.model.root_for_uuid(self.ref_uuid)
        self.model.create_ref_node('SupportingRepresentation',
                                   rqet.find_nested_tags_text(ref_root, ['Citation', 'Title']),
                                   self.ref_uuid,
                                   content_type = 'Grid2dRepresentation',
                                   root = sg_node)

        zv_node = rqet.SubElement(p_node, ns['resqml2'] + 'ZValues')
        zv_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
        zv_node.text = '\n'

        v_node = rqet.SubElement(zv_node, ns['resqml2'] + 'Values')
        v_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        v_node.text = '\n'

        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'zvalues', root = v_node)

    def __create_xml_regandz(self, p_node, ext_uuid):

        assert ext_uuid is not None

        p_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dZValueArray')

        sg_node = rqet.SubElement(p_node, ns['resqml2'] + 'SupportingGeometry')

        sg_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dLatticeArray')
        sg_node.text = '\n'

        assert self.regular_origin is not None and self.regular_dxyz_dij is not None

        self.model.create_solitary_point3d('Origin', sg_node, self.regular_origin)  # todo: check xml namespace

        for j_or_i in range(2):  # 0 = J, 1 = I; ie. J axis info first in xml, followed by I axis
            dxyz = self.regular_dxyz_dij[1 - j_or_i].copy()
            log.debug('dxyz: ' + str(dxyz))
            d_value = vec.dot_product(dxyz, dxyz)
            assert d_value > 0.0
            d_value = maths.sqrt(d_value)
            dxyz /= d_value
            o_node = rqet.SubElement(sg_node, ns['resqml2'] + 'Offset')
            o_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dOffset')
            o_node.text = '\n'
            self.model.create_solitary_point3d('Offset', o_node, dxyz)
            space_node = rqet.SubElement(o_node, ns['resqml2'] + 'Spacing')
            space_node.set(ns['xsi'] + 'type',
                           ns['resqml2'] + 'DoubleConstantArray')  # nothing else catered for just now
            space_node.text = '\n'
            ov_node = rqet.SubElement(space_node, ns['resqml2'] + 'Value')
            ov_node.set(ns['xsi'] + 'type', ns['xsd'] + 'double')
            ov_node.text = str(d_value)
            oc_node = rqet.SubElement(space_node, ns['resqml2'] + 'Count')
            oc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            if j_or_i:
                oc_node.text = str(self.ni - 1)
            else:
                oc_node.text = str(self.nj - 1)

        zv_node = rqet.SubElement(p_node, ns['resqml2'] + 'ZValues')
        zv_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'DoubleHdf5Array')
        zv_node.text = '\n'

        v_node = rqet.SubElement(zv_node, ns['resqml2'] + 'Values')
        v_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        v_node.text = '\n'

        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'zvalues', root = v_node)

    def __create_xml_explicit(self, p_node, ext_uuid, use_xy_only):
        assert ext_uuid is not None

        if use_xy_only:
            p_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point2dHdf5Array')
        else:
            p_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dHdf5Array')

        coords = rqet.SubElement(p_node, ns['resqml2'] + 'Coordinates')
        coords.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        coords.text = '\n'

        self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'points', root = coords)

    def __create_xml_basics(self, g2d_node):
        if self.represented_interpretation_root is not None:
            interp_root = self.represented_interpretation_root
            interp_uuid = bu.uuid_from_string(interp_root.attrib['uuid'])
            interp_part = self.model.part_for_uuid(interp_uuid)
            self.model.create_ref_node('RepresentedInterpretation',
                                       rqet.find_nested_tags_text(interp_root, ['Citation', 'Title']),
                                       interp_uuid,
                                       content_type = self.model.type_of_part(interp_part),
                                       root = g2d_node)

        role_node = rqet.SubElement(g2d_node, ns['resqml2'] + 'SurfaceRole')
        role_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'SurfaceRole')
        role_node.text = self.surface_role

        patch_node = rqet.SubElement(g2d_node, ns['resqml2'] + 'Grid2dPatch')
        patch_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Grid2dPatch')
        patch_node.text = '\n'

        pi_node = rqet.SubElement(patch_node, ns['resqml2'] + 'PatchIndex')
        pi_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
        pi_node.text = '0'

        fast_node = rqet.SubElement(patch_node, ns['resqml2'] + 'FastestAxisCount')
        fast_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
        fast_node.text = str(self.ni)

        slow_node = rqet.SubElement(patch_node, ns['resqml2'] + 'SlowestAxisCount')
        slow_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
        slow_node.text = str(self.nj)

        geom = rqet.SubElement(patch_node, ns['resqml2'] + 'Geometry')
        geom.set(ns['xsi'] + 'type', ns['resqml2'] + 'PointGeometry')
        geom.text = '\n'

        self.model.create_crs_reference(crs_uuid = self.crs_uuid, root = geom)

        p_node = rqet.SubElement(geom, ns['resqml2'] + 'Points')
        p_node.text = '\n'

        return p_node

    def __create_xml_add_parts(self, g2d_node, ref_root, add_relationships, ext_uuid):
        self.model.add_part('obj_Grid2dRepresentation', self.uuid, g2d_node)
        if add_relationships:
            self.model.create_reciprocal_relationship(g2d_node, 'destinationObject', self.crs_root, 'sourceObject')
            if self.represented_interpretation_root is not None:
                self.model.create_reciprocal_relationship(g2d_node, 'destinationObject',
                                                          self.represented_interpretation_root, 'sourceObject')
            if ref_root is not None:  # used for ref&z flavour
                self.model.create_reciprocal_relationship(g2d_node, 'destinationObject', ref_root, 'sourceObject')
            if self.flavour == 'ref&z' or self.flavour == 'explicit' or self.flavour == 'reg&z':
                ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
                ext_node = self.model.root_for_part(ext_part)
                self.model.create_reciprocal_relationship(g2d_node, 'mlToExternalPartProxy', ext_node,
                                                          'externalPartProxyToMl')

    def create_xml(self,
                   ext_uuid = None,
                   crs_root = None,
                   use_xy_only = False,
                   add_as_part = True,
                   add_relationships = True,
                   root = None,
                   title = None,
                   originator = None):
        """Creates a grid 2d representation xml node from this mesh object and optionally adds as part of model.

        arguments:
           ext_uuid (uuid.UUID, optional): the uuid of the hdf5 external part holding the mesh array
           crs_root (DEPRECATED): ignored, crs must now be established at time of initialisation
           use_xy_only (boolean, default False): if True and the flavour of this mesh is explicit, only
              the xy coordinates are stored in the hdf5 dataset, otherwise xyz are stored
           add_as_part (boolean, default True): if True, the newly created xml node is added as a part
              in the model
           add_relationships (boolean, default True): if True, a relationship xml part is created relating the
              new grid 2d part to the crs part (and optional interpretation part), and to the represented
              interpretation if present
           root (optional, usually None): if not None, the newly created grid 2d representation node is appended
              as a child to this node
           title (string, optional): used as the citation Title text; should be meaningful to a human
           originator (string, optional): the name of the human being who created the grid 2d representation part;
              default is to use the login name

        returns:
           the newly created grid 2d representation (mesh) xml node
        """

        if crs_root is not None:
            warnings.warn('crs_root argument is deprecated and ignored in Mesh.create_xml()')

        if ext_uuid is None and self.flavour != 'regular':
            ext_uuid = self.model.h5_uuid()

        g2d_node = super().create_xml(add_as_part = False, title = title, originator = originator)

        p_node = self.__create_xml_basics(g2d_node)

        ref_root = None

        if self.flavour == 'regular':
            self.__create_xml_regular(p_node)

        elif self.flavour == 'ref&z':
            self.__create_xml_refandz(p_node, ext_uuid)

        elif self.flavour == 'reg&z':
            self.__create_xml_regandz(p_node, ext_uuid)

        elif self.flavour == 'explicit':
            self.__create_xml_explicit(p_node, ext_uuid, use_xy_only)

        else:
            log.error('mesh has bad flavour when creating xml')
            return None

        if root is not None:
            root.append(g2d_node)
        if add_as_part:
            self.__create_xml_add_parts(g2d_node, ref_root, add_relationships, ext_uuid)

        return g2d_node
