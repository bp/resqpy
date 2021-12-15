"""surface.py: surface class based on resqml standard."""

version = '4th November 2021'

# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company
# GOCAD is also a trademark of Emerson

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.olio.triangulation as triangulate
import resqpy.olio.uuid as bu
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
from resqpy.olio.xml_namespaces import curly_namespace as ns
from resqpy.olio.zmap_reader import read_mesh
from .base_surface import BaseSurface
from .triangulated_patch import TriangulatedPatch


class Surface(BaseSurface):
    """Class for RESQML triangulated set surfaces."""

    resqml_type = 'TriangulatedSetRepresentation'

    def __init__(self,
                 parent_model,
                 uuid = None,
                 surface_root = None,
                 point_set = None,
                 mesh = None,
                 mesh_file = None,
                 mesh_format = None,
                 tsurf_file = None,
                 quad_triangles = False,
                 title = None,
                 surface_role = 'map',
                 crs_uuid = None,
                 originator = None,
                 extra_metadata = {}):
        """Create an empty Surface object (RESQML TriangulatedSetRepresentation).
        
        Optionally populates from xml, point set or mesh.

        arguments:
           parent_model (model.Model object): the model to which this surface belongs
           uuid (uuid.UUID, optional): if present, the surface is initialised from an existing RESQML object with this uuid
           surface_root (xml tree root node, optional): DEPRECATED: alternative to using uuid
           point_set (PointSet object, optional): if present, the surface is initialised as a Delaunay
              triangulation of the points in the point set; ignored if extracting from xml
           mesh (Mesh object, optional): if present, the surface is initialised as a triangulation of
              the mesh; ignored if extracting from xml or if point_set is present
           mesh_file (string, optional): the path of an ascii file holding a mesh in RMS text or zmap+ format;
              ignored if extracting from xml or point_set or mesh is present
           mesh_format (string, optional): 'rms' or 'zmap'; required if initialising from mesh_file
           tsurf_file (string, optional): the path of an ascii file holding details of triangles and points in GOCAD-Tsurf format;
              ignored if extraction from xml or point_set or mesh is present
           quad_triangles (boolean, default False): if initialising from mesh or mesh_file, each 'square'
              is represented by 2 triangles if quad_triangles is False, 4 triangles if True
           title (string, optional): used as the citation title for the new object, ignored if
              extracting from xml
           surface_role (string, default 'map'): 'map' or 'pick'; ignored if root_node is not None
           crs_uuid (uuid.UUID, optional): if present and not extracting from xml, is set as the crs uuid
              applicable to mesh etc. data
           originator (str, optional): the name of the person creating the object; defaults to login id; ignored
              when initialising from an existing RESQML object
           extra_metadata (dict): items in this dictionary are added as extra metadata; ignored
              when initialising from an existing RESQML object

        returns:
           a newly created surface object

        notes:
           there are 6 ways to initialise a surface object, in order of precendence:
           1. extracting from xml
           2. as a Delaunay triangulation of points in a PointSet
           3. as a simple triangulation of a Mesh object
           4. as a simple triangulation of a mesh in an ascii file
           5. from a GOCAD-TSurf format file
           5. as an empty surface
           if an empty surface is created, 'set_from_...' methods are available to then set for one of:
           - a horizontal plane
           - a single triangle
           - a 'sail' (a triangle wrapped onto a sphere)
           - etc.
           the quad_triangles option is only applied if initialising from a mesh or mesh_file that is fully
           defined (ie. no NaN's)

        :meta common:
        """

        assert surface_role in ['map', 'pick']

        self.surface_role = surface_role
        self.patch_list = []  # ordered list of patches
        self.crs_uuid = crs_uuid
        self.triangles = None  # composite triangles (all patches)
        self.points = None  # composite points (all patches)
        self.boundaries = None  # todo: read up on what this is for and look out for examples
        self.represented_interpretation_root = None
        self.title = title
        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata,
                         root_node = surface_root)
        if self.root is not None:
            pass
        elif point_set is not None:
            self.set_from_point_set(point_set)
        elif mesh is not None:
            self.set_from_mesh_object(mesh, quad_triangles = quad_triangles)
        elif mesh_file and mesh_format:
            self.set_from_mesh_file(mesh_file, mesh_format, quad_triangles = quad_triangles)
        elif tsurf_file is not None:
            self.set_from_tsurf_file(tsurf_file)

    def _load_from_xml(self):
        root_node = self.root
        assert root_node is not None
        self.extract_patches(root_node)
        ref_node = rqet.find_tag(root_node, 'RepresentedInterpretation')
        if ref_node is not None:
            interp_root = self.model.referenced_node(ref_node)
            self.set_represented_interpretation_root(interp_root)

    @property
    def represented_interpretation_uuid(self):
        """Returns the uuid of the represented surface interpretation, or None."""

        return rqet.uuid_for_part_root(self.represented_interpretation_root)

    def set_represented_interpretation_root(self, interp_root):
        """Makes a note of the xml root of the represented interpretation."""

        self.represented_interpretation_root = interp_root

    def extract_patches(self, surface_root):
        """Scan surface root for triangle patches, create TriangulatedPatch objects and build up patch_list."""

        if len(self.patch_list):
            return
        assert surface_root is not None
        paired_list = []
        self.patch_list = []
        for child in surface_root:
            if rqet.stripped_of_prefix(child.tag) != 'TrianglePatch':
                continue
            patch_index = rqet.find_tag_int(child, 'PatchIndex')
            assert patch_index is not None
            triangulated_patch = TriangulatedPatch(self.model, patch_index = patch_index, patch_node = child)
            assert triangulated_patch is not None
            if self.crs_uuid is None:
                self.crs_uuid = triangulated_patch.crs_uuid
            else:
                if not bu.matching_uuids(triangulated_patch.crs_uuid, self.crs_uuid):
                    log.warning('mixed coordinate reference systems in use within a surface')
            paired_list.append((patch_index, triangulated_patch))
        paired_list.sort()
        assert len(paired_list) and paired_list[0][0] == 0 and len(paired_list) == paired_list[-1][0] + 1
        for _, patch in paired_list:
            self.patch_list.append(patch)

    def set_model(self, parent_model):
        """Associate the surface with a resqml model (does not create xml or write hdf5 data)."""

        self.model = parent_model

    def triangles_and_points(self):
        """Returns arrays representing combination of all the patches in the surface.

        Returns:
           Tuple (triangles, points):

           * triangles (int array of shape[:, 3]): integer indices into points array,
             being the nodes of the corners of the triangles
           * points (float array of shape[:, 3]): flat array of xyz points, indexed by triangles

        :meta common:
        """

        if self.triangles is not None:
            return (self.triangles, self.points)
        self.extract_patches(self.root)
        points_offset = 0
        for triangulated_patch in self.patch_list:
            (t, p) = triangulated_patch.triangles_and_points()
            if points_offset == 0:
                self.triangles = t.copy()
                self.points = p.copy()
            else:
                self.triangles = np.concatenate((self.triangles, t.copy() + points_offset))
                self.points = np.concatenate((self.points, p.copy()))
            points_offset += p.shape[0]
        return (self.triangles, self.points)

    def distinct_edges(self):
        """Returns a numpy int array of shape (N, 2) being the ordered node pairs of distinct edges of triangles."""

        triangles, _ = self.triangles_and_points()
        assert triangles is not None
        tri_count = len(triangles)
        all_edges = np.empty((tri_count, 3, 2))
        for i in range(3):
            all_edges[:, i, 0] = triangles[:, i - 1]
            all_edges[:, i, 1] = triangles[:, i]
        return np.unique(np.sort(all_edges.reshape((-1, 2)), axis = 1), axis = 0)

    def set_from_triangles_and_points(self, triangles, points):
        """Populate this (empty) Surface object from an array of triangle corner indices and an array of points."""

        tri_patch = TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
        tri_patch.set_from_triangles_and_points(triangles, points)
        self.patch_list = [tri_patch]
        self.uuid = bu.new_uuid()

    def set_from_point_set(self, point_set, convexity_parameter = 5.0):
        """Populate this (empty) Surface object with a Delaunay triangulation of points in a PointSet object.

        arguments:
           point_set (PointSet): the set of points to be triangulated to form a surface
           convexity_parameter (float, default 5.0): controls how likely the resulting triangulation is to be
              convex; reduce to 1.0 to allow slightly more concavities; increase to 100.0 or more for very little
              chance of even a slight concavity
        """

        p = point_set.full_array_ref()
        log.debug('number of points going into dt: ' + str(len(p)))
        t = triangulate.dt(p[:, :2], container_size_factor = convexity_parameter)
        log.debug('number of triangles: ' + str(len(t)))
        self.crs_uuid = point_set.crs_uuid
        self.set_from_triangles_and_points(t, p)

    def set_from_irregular_mesh(self, mesh_xyz, quad_triangles = False):
        """Populate this (empty) Surface object from an untorn mesh array of shape (N, M, 3).

        arguments:
           mesh_xyz (numpy float array of shape (N, M, 3)): a 2D lattice of points in 3D space
           quad_triangles: (boolean, optional, default False): if True, each quadrangle is represented by
              4 triangles in the surface, with the mean of the 4 corner points used as a common centre node;
              if False (the default), only 2 triangles are used for each quadrangle; note that the 2 triangle
              mode gives a non-unique triangulated result
        """

        mesh_shape = mesh_xyz.shape
        assert len(mesh_shape) == 3 and mesh_shape[2] == 3
        tri_patch = TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
        tri_patch.set_from_irregular_mesh(mesh_xyz, quad_triangles = quad_triangles)
        self.patch_list = [tri_patch]
        self.uuid = bu.new_uuid()

    def set_from_sparse_mesh(self, mesh_xyz):
        """Populate this (empty) Surface object from a mesh array of shape (N, M, 3) with NaNs.

        arguments:
           mesh_xyz (numpy float array of shape (N, M, 3)): a 2D lattice of points in 3D space, with NaNs in z
        """

        mesh_shape = mesh_xyz.shape
        assert len(mesh_shape) == 3 and mesh_shape[2] == 3
        tri_patch = TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
        tri_patch.set_from_sparse_mesh(mesh_xyz)
        self.patch_list = [tri_patch]
        self.uuid = bu.new_uuid()

    def set_from_mesh_object(self, mesh, quad_triangles = False):
        """Populate the (empty) Surface object from a Mesh object."""

        xyz = mesh.full_array_ref()
        if np.any(np.isnan(xyz)):
            self.set_from_sparse_mesh(xyz)
        else:
            self.set_from_irregular_mesh(xyz, quad_triangles = quad_triangles)

    def set_from_torn_mesh(self, mesh_xyz, quad_triangles = False):
        """Populate this (empty) Surface object from a torn mesh array of shape (nj, ni, 2, 2, 3).

        arguments:
           mesh_xyz (numpy float array of shape (nj, ni, 2, 2, 3)): corner points of 2D faces in 3D space
           quad_triangles: (boolean, optional, default False): if True, each quadrangle (face) is represented
              by 4 triangles in the surface, with the mean of the 4 corner points used as a common centre node;
              if False (the default), only 2 triangles are used for each quadrangle; note that the 2 triangle
              mode gives a non-unique triangulated result
        """

        mesh_shape = mesh_xyz.shape
        assert len(mesh_shape) == 5 and mesh_shape[2:] == (2, 2, 3)
        tri_patch = TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
        tri_patch.set_from_torn_mesh(mesh_xyz, quad_triangles = quad_triangles)
        self.patch_list = [tri_patch]
        self.uuid = bu.new_uuid()

    def column_from_triangle_index(self, triangle_index):
        """For surface freshly built from fully defined mesh, returns (j, i) for given triangle index.

        argument:
           triangle_index (int or numpy int array): the triangle index (or array of indices) for which column(s) is/are
           being sought

        returns:
           pair of ints or pair of numpy int arrays: the (j0, i0) indices of the column(s) which the triangle(s) is/are
           part of

        notes:
           this function will only work if the surface has been freshly constructed with data from a mesh without NaNs,
           otherwise (None, None) will be returned;
           the information needed to map from triangle to column is not persistently stored as part of a resqml surface;
           if triangle_index is a numpy int array, a pair of similarly shaped numpy arrays is returned

        :meta common:
        """

        assert len(self.patch_list) == 1
        return self.patch_list[0].column_from_triangle_index(triangle_index)

    def set_to_single_cell_faces_from_corner_points(self, cp, quad_triangles = True):
        """Populates this (empty) surface to represent faces of a cell, from corner points of shape (2, 2, 2, 3)."""

        assert cp.size == 24
        tri_patch = TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
        tri_patch.set_to_cell_faces_from_corner_points(cp, quad_triangles = quad_triangles)
        self.patch_list = [tri_patch]
        self.uuid = bu.new_uuid()

    def set_to_multi_cell_faces_from_corner_points(self, cp, quad_triangles = True):
        """Populates this (empty) surface to represent faces of a set of cells.
        
        From corner points of shape (N, 2, 2, 2, 3).
        """
        assert cp.size % 24 == 0
        cp = cp.reshape((-1, 2, 2, 2, 3))
        self.patch_list = []
        p_index = 0
        for cell_cp in cp:
            tri_patch = TriangulatedPatch(self.model, patch_index = p_index, crs_uuid = self.crs_uuid)
            tri_patch.set_to_cell_faces_from_corner_points(cell_cp, quad_triangles = quad_triangles)
            self.patch_list.append(tri_patch)
            p_index += 1
        self.uuid = bu.new_uuid()

    def cell_axis_and_polarity_from_triangle_index(self, triangle_index):
        """For surface freshly built for cell faces, returns (cell_number, face_axis, polarity).

        arguments:
           triangle_index (int or numpy int array): the triangle index (or array of indices) for which cell face
              information is required

        returns:
           triple int: (cell_number, axis, polarity)

        note:
           if the surface was built for a single cell, the returned cell number will be zero
        """

        triangles_per_face = 4 if self.patch_list[0].quad_triangles else 2
        face_index = triangle_index // triangles_per_face
        cell_number, remainder = divmod(face_index, 6)
        axis, polarity = divmod(remainder, 2)
        return cell_number, axis, polarity

    def set_to_horizontal_plane(self, depth, box_xyz, border = 0.0):
        """Populate this (empty) surface with a patch of two triangles.
        
        Triangles define a flat, horizontal plane at a given depth.

        arguments:
            depth (float): z value to use in all points in the triangulated patch
            box_xyz (float[2, 3]): the min, max values of x, y (&z) giving the area to be covered (z ignored)
            border (float): an optional border width added around the x,y area defined by box_xyz

        :meta common:
        """

        tri_patch = TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
        tri_patch.set_to_horizontal_plane(depth, box_xyz, border = border)
        self.patch_list = [tri_patch]
        self.uuid = bu.new_uuid()

    def set_to_triangle(self, corners):
        """Populate this (empty) surface with a patch of one triangle."""

        tri_patch = TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
        tri_patch.set_to_triangle(corners)
        self.patch_list = [tri_patch]
        self.uuid = bu.new_uuid()

    def set_to_sail(self, n, centre, radius, azimuth, delta_theta):
        """Populate this (empty) surface with a patch representing a triangle wrapped on a sphere."""

        tri_patch = TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
        tri_patch.set_to_sail(n, centre, radius, azimuth, delta_theta)
        self.patch_list = [tri_patch]
        self.uuid = bu.new_uuid()

    def set_from_mesh_file(self, filename, format, quad_triangles = False):
        """Populate this (empty) surface from a zmap or RMS text mesh file."""

        assert format in ['zmap', 'rms', 'roxar']  # 'roxar' is synonymous with 'rms'
        x, y, z = read_mesh(filename, format = format)
        assert x is not None and y is not None or z is not None, 'failed to read surface from zmap file'
        assert x.size == y.size and x.size == z.size, 'non matching array sizes from zmap reader'
        assert x.shape == y.shape and x.shape == z.shape, 'non matching array shapes from zmap reader'

        xyz_mesh = np.stack((x, y, z), axis = -1)
        if np.any(np.isnan(z)):
            self.set_from_sparse_mesh(xyz_mesh)
        else:
            self.set_from_irregular_mesh(xyz_mesh, quad_triangles = quad_triangles)

    def set_from_tsurf_file(self, filename):
        """Populate this (empty) surface from a GOCAD tsurf file."""
        triangles, vertices = [], []
        with open(filename, 'r') as fl:
            lines = fl.readlines()
            for line in lines:
                if "VRTX" in line:
                    vertices.append(line.rstrip().split(" ")[2:])
                elif "TRGL" in line:
                    triangles.append(line.rstrip().split(" ")[1:])
        triangles = np.array(triangles, dtype = int)
        vertices = np.array(vertices, dtype = float)
        self.set_from_triangles_and_points(triangles = triangles, points = vertices)

    def set_from_zmap_file(self, filename, quad_triangles = False):
        """Populate this (empty) surface from a zmap mesh file."""

        self.set_from_mesh_file(filename, 'zmap', quad_triangles = quad_triangles)

    def set_from_roxar_file(self, filename, quad_triangles = False):
        """Populate this (empty) surface from an RMS text mesh file."""

        self.set_from_mesh_file(filename, 'rms', quad_triangles = quad_triangles)

    def set_from_rms_file(self, filename, quad_triangles = False):
        """Populate this (empty) surface from an RMS text mesh file."""

        self.set_from_mesh_file(filename, 'rms', quad_triangles = quad_triangles)

    def vertical_rescale_points(self, ref_depth = None, scaling_factor = 1.0):
        """Modify the z values of points by rescaling.
        
        Stretches the distance from reference depth by scaling factor.
        """
        if scaling_factor == 1.0:
            return
        if ref_depth is None:
            for patch in self.patch_list:
                patch_min = np.min(patch.points[:, 2])
                if ref_depth is None or patch_min < ref_depth:
                    ref_depth = patch_min
        assert ref_depth is not None, 'no z values found for vertical rescaling of surface'
        self.triangles = None  # invalidate any cached triangles & points in surface object
        self.points = None
        for patch in self.patch_list:
            patch.vertical_rescale_points(ref_depth, scaling_factor)

    def write_hdf5(self, file_name = None, mode = 'a'):
        """Create or append to an hdf5 file, writing datasets for the triangulated patches after caching arrays.

        :meta common:
        """

        if self.uuid is None:
            self.uuid = bu.new_uuid()
        # NB: patch arrays must all have been set up prior to calling this function
        h5_reg = rwh5.H5Register(self.model)
        # todo: sort patches by patch index and check sequence
        for triangulated_patch in self.patch_list:
            (t, p) = triangulated_patch.triangles_and_points()
            h5_reg.register_dataset(self.uuid, 'points_patch{}'.format(triangulated_patch.patch_index), p)
            h5_reg.register_dataset(self.uuid, 'triangles_patch{}'.format(triangulated_patch.patch_index), t)
        h5_reg.write(file_name, mode = mode)

    def create_xml(self,
                   ext_uuid = None,
                   add_as_part = True,
                   add_relationships = True,
                   crs_uuid = None,
                   title = None,
                   originator = None):
        """Creates a triangulated surface xml node from this surface object and optionally adds as part of model.

        arguments:
            ext_uuid (uuid.UUID): the uuid of the hdf5 external part holding the surface arrays
            add_as_part (boolean, default True): if True, the newly created xml node is added as a part
                in the model
            add_relationships (boolean, default True): if True, a relationship xml part is created relating the
                new triangulated representation part to the crs part (and optional interpretation part)
            crs_uuid (optional): the uuid of the coordinate reference system applicable to the surface points data;
                if None, the main crs for the model is assumed to apply
            title (string): used as the citation Title text; should be meaningful to a human
            originator (string, optional): the name of the human being who created the triangulated representation part;
                default is to use the login name

        returns:
            the newly created triangulated representation (surface) xml node

        :meta common:
        """

        if ext_uuid is None:
            ext_uuid = self.model.h5_uuid()
        if not self.title:
            self.title = 'surface'

        tri_rep = super().create_xml(add_as_part = False, title = title, originator = originator)

        # todo: if crs_root is None, attempt to derive from surface patch crs uuid (within patch loop, below)
        if crs_uuid is None:
            crs_root = self.model.crs_root  # maverick use of model's default crs
            crs_uuid = rqet.uuid_for_part_root(crs_root)
        else:
            crs_root = self.model.root_for_uuid(crs_uuid)

        if self.represented_interpretation_root is not None:
            interp_root = self.represented_interpretation_root
            interp_uuid = bu.uuid_from_string(interp_root.attrib['uuid'])
            interp_part = self.model.part_for_uuid(interp_uuid)
            interp_title = rqet.find_nested_tags_text(interp_root, ['Citation', 'Title'])
            self.model.create_ref_node('RepresentedInterpretation',
                                       interp_title,
                                       interp_uuid,
                                       content_type = self.model.type_of_part(interp_part),
                                       root = tri_rep)
            if interp_title and not title:
                title = interp_title

        # if not title: title = 'surface'
        # self.model.create_citation(root = tri_rep, title = title, originator = originator)

        role_node = rqet.SubElement(tri_rep, ns['resqml2'] + 'SurfaceRole')
        role_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'SurfaceRole')
        role_node.text = self.surface_role

        for patch in self.patch_list:

            p_node = rqet.SubElement(tri_rep, ns['resqml2'] + 'TrianglePatch')
            p_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'TrianglePatch')
            p_node.text = '\n'

            pi_node = rqet.SubElement(p_node, ns['resqml2'] + 'PatchIndex')
            pi_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
            pi_node.text = str(patch.patch_index)

            ct_node = rqet.SubElement(p_node, ns['resqml2'] + 'Count')
            ct_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
            ct_node.text = str(patch.triangle_count)

            cn_node = rqet.SubElement(p_node, ns['resqml2'] + 'NodeCount')
            cn_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
            cn_node.text = str(patch.node_count)

            triangles_node = rqet.SubElement(p_node, ns['resqml2'] + 'Triangles')
            triangles_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IntegerHdf5Array')
            triangles_node.text = '\n'

            # not sure if null value node is needed, not actually used in data
            triangles_null = rqet.SubElement(triangles_node, ns['resqml2'] + 'NullValue')
            triangles_null.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
            triangles_null.text = '-1'  # or set to number of points in surface coords?

            triangles_values = rqet.SubElement(triangles_node, ns['resqml2'] + 'Values')
            triangles_values.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            triangles_values.text = '\n'

            self.model.create_hdf5_dataset_ref(ext_uuid,
                                               self.uuid,
                                               'triangles_patch{}'.format(patch.patch_index),
                                               root = triangles_values)

            geom = rqet.SubElement(p_node, ns['resqml2'] + 'Geometry')
            geom.set(ns['xsi'] + 'type', ns['resqml2'] + 'PointGeometry')
            geom.text = '\n'

            self.model.create_crs_reference(crs_uuid = crs_uuid, root = geom)

            points_node = rqet.SubElement(geom, ns['resqml2'] + 'Points')
            points_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dHdf5Array')
            points_node.text = '\n'

            coords = rqet.SubElement(points_node, ns['resqml2'] + 'Coordinates')
            coords.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
            coords.text = '\n'

            self.model.create_hdf5_dataset_ref(ext_uuid,
                                               self.uuid,
                                               'points_patch{}'.format(patch.patch_index),
                                               root = coords)

            patch.node = p_node

        if add_as_part:
            self.model.add_part('obj_TriangulatedSetRepresentation', self.uuid, tri_rep)
            if add_relationships:
                # todo: add multiple crs'es (one per patch)?
                self.model.create_reciprocal_relationship(tri_rep, 'destinationObject', crs_root, 'sourceObject')
                if self.represented_interpretation_root is not None:
                    self.model.create_reciprocal_relationship(tri_rep, 'destinationObject',
                                                              self.represented_interpretation_root, 'sourceObject')

                ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
                ext_node = self.model.root_for_part(ext_part)
                self.model.create_reciprocal_relationship(tri_rep, 'mlToExternalPartProxy', ext_node,
                                                          'externalPartProxyToMl')

        return tri_rep
