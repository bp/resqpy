"""Surface class based on RESQML TriangulatedSetRepresentation class."""

# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company
# GOCAD is also a trademark of Emerson

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.crs as rqc
import resqpy.olio.intersection as meet
import resqpy.olio.triangulation as triangulate
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp
import resqpy.surface as rqs
import resqpy.surface._base_surface as rqsb
import resqpy.surface._triangulated_patch as rqstp
import resqpy.weights_and_measures as wam

from resqpy.olio.xml_namespaces import curly_namespace as ns
from resqpy.olio.zmap_reader import read_mesh


class Surface(rqsb.BaseSurface):
    """Class for RESQML triangulated set surfaces."""

    resqml_type = 'TriangulatedSetRepresentation'

    def __init__(self,
                 parent_model,
                 uuid = None,
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
           surface_role (string, default 'map'): 'map' or 'pick'; ignored if uuid is not None
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
           if an empty surface is created, set_from... methods are available to then set for one of:
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
        self.normal_vector = None  # a single derived vector that is roughly (true) normal to the surface
        self.title = title
        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)
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

    @classmethod
    def from_tri_mesh(cls, tri_mesh):
        """Create a Surface from a TriMesh."""
        assert isinstance(tri_mesh, rqs.TriMesh)
        surf = cls(tri_mesh.model,
                   crs_uuid = tri_mesh.crs_uuid,
                   title = tri_mesh.title,
                   surface_role = tri_mesh.surface_role,
                   extra_metadata = tri_mesh.extra_metadata if hasattr(tri_mesh, 'extra_metadata') else None)
        t, p = tri_mesh.triangles_and_points()
        surf.set_from_triangles_and_points(t, p)
        surf.represented_interpretation_root = tri_mesh.represented_interpretation_root
        return surf

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

        if len(self.patch_list) or surface_root is None:
            return
        paired_list = []
        self.patch_list = []
        for child in surface_root:
            if rqet.stripped_of_prefix(child.tag) != 'TrianglePatch':
                continue
            patch_index = rqet.find_tag_int(child, 'PatchIndex')
            assert patch_index is not None
            triangulated_patch = rqstp.TriangulatedPatch(self.model, patch_index = patch_index, patch_node = child)
            assert triangulated_patch is not None
            if self.crs_uuid is None:
                self.crs_uuid = triangulated_patch.crs_uuid
            else:
                if not bu.matching_uuids(triangulated_patch.crs_uuid, self.crs_uuid):
                    log.warning('mixed coordinate reference systems in use within a surface')
            paired_list.append((patch_index, triangulated_patch))
        assert len(paired_list), f'no triangulated patches found for surface: {self.title}'
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

    def triangle_count(self):
        """Return the numner of triangles in this surface."""

        self.extract_patches(self.root)
        if not self.patch_list:
            return 0
        return np.sum([tp.triangle_count for tp in self.patch_list])

    def node_count(self):
        """Return the number of nodes (points) used in this surface."""

        self.extract_patches(self.root)
        if not self.patch_list:
            return 0
        return np.sum([tp.node_count for tp in self.patch_list])

    def change_crs(self, required_crs):
        """Changes the crs of the surface, also sets a new uuid if crs changed.

        note:
           this method is usually used to change the coordinate system for a temporary resqpy object;
           to add as a new part, call write_hdf5() and create_xml() methods
        """

        old_crs = rqc.Crs(self.model, uuid = self.crs_uuid)
        self.crs_uuid = required_crs.uuid
        if required_crs == old_crs or not self.patch_list:
            log.debug(f'no crs change needed for {self.title}')
            return
        log.debug(f'crs change needed for {self.title} from {old_crs.title} to {required_crs.title}')
        for patch in self.patch_list:
            patch.triangles_and_points()
            required_crs.convert_array_from(old_crs, patch.points)
            patch.crs_uuid = self.crs_uuid
        self.triangles = None  # clear cached arrays for surface
        self.points = None
        self.uuid = bu.new_uuid()  # hope this doesn't cause problems
        assert self.root is None

    def set_to_trimmed_surface(self, large_surface, xyz_box = None, xy_polygon = None):
        """Populate this (empty) surface with triangles and points which overlap with a trimming volume.

        arguments:
            large_surface (Surface): the larger surface, a copy of which is to be trimmed
            xyz_box (numpy float array of shape (2, 3), optional): if present, a cuboid in xyz space
               against which to trim the surface
            xy_polygon (closed convex resqpy.lines.Polyline, optional): if present, an xy boundary
               against which to trim

        notes:
            at least one of xyz_box or xy_polygon must be present; if both are present, a triangle
            must have at least one point within both boundaries to survive the trimming;
            xyz_box and xy_polygon must be in the same crs as the large surface
        """

        assert xyz_box is not None or xy_polygon is not None
        box = None
        if xyz_box is not None:
            assert xyz_box.shape == (2, 3)
            # guard against reversed ranges in xyz_box
            box = np.empty((2, 3), dtype = float)
            box[0] = np.amin(xyz_box, axis = 0)
            box[1] = np.amax(xyz_box, axis = 0)
        log.debug(f'trimming surface {large_surface.title} from {large_surface.triangle_count()} triangles')
        if not self.title:
            self.title = str(large_surface.title) + ' trimmed'
        self.crs_uuid = large_surface.crs_uuid
        self.patch_list = []
        for triangulated_patch in large_surface.patch_list:
            trimmed_patch = rqstp.TriangulatedPatch(self.model,
                                                    patch_index = len(self.patch_list),
                                                    crs_uuid = self.crs_uuid)
            trimmed_patch.set_to_trimmed_patch(triangulated_patch, xyz_box = box, xy_polygon = xy_polygon)
            if trimmed_patch is not None and trimmed_patch.triangle_count > 0:
                self.patch_list.append(trimmed_patch)
        if len(self.patch_list):
            log.debug(f'trimmed surface {self.title} has {self.triangle_count()} triangles')
        else:
            log.warning('surface does not intersect trimming volume')
        self.uuid = bu.new_uuid()

    def set_to_split_surface(self, large_surface, line, delta_xyz):
        """Populate this (empty) surface with a version of a larger surface split by an xy line.

        arguments:
            large_surface (Surface): the larger surface, a copy of which is to be split
            line (numpy float array of shape (2, 3)): the end points of a line (z will be ignored)
            delta_xyz (numpy float array of shape (2, 3)): an xyz offset to add to the points on the
                right and left sides of the line repspectively

        notes:
            this method can be used to introduce a tear into a surface, typically to represent a horizon
            being split by a planar fault with a throw
        """

        assert line.shape == (2, 3)
        assert delta_xyz.shape == (2, 3)
        t, p = large_surface.triangles_and_points()
        assert p.ndim == 2 and p.shape[1] == 3
        pp = np.concatenate((p, line), axis = 0)
        tp = np.empty(p.shape, dtype = int)
        tp[:, 0] = len(p)
        tp[:, 1] = len(p) + 1
        tp[:, 2] = np.arange(len(p), dtype = int)
        cw = vec.clockwise_triangles(pp, tp)
        pai = np.where(cw >= 0.0, True, False)  # bool mask over p
        pbi = np.where(cw <= 0.0, True, False)  # bool mask over p
        tap = pai[t]
        tbp = pbi[t]
        ta = np.any(tap, axis = 1)  # bool array over t
        tb = np.any(tbp, axis = 1)  # bool array over t

        # here we stick the two halves together into a single patch
        # todo: keep as two patches as required by RESQML business rules
        p_combo = np.empty((0, 3))
        t_combo = np.empty((0, 3), dtype = int)
        for i, tab in enumerate((ta, tb)):
            p_keep = np.unique(t[tab])
            # note new point index for each old point that is being kept
            p_map = np.full(len(p), -1, dtype = int)
            p_map[p_keep] = np.arange(len(p_keep))
            # copy those unique points into a trimmed points array
            points_trimmed = p[p_keep].copy()
            # copy selected triangles, replacing p indices with compressed indices
            t_trim = t[tab]
            triangles_trimmed = p_map[t_trim].copy()
            assert np.all(triangles_trimmed >= 0)
            assert np.all(triangles_trimmed < len(points_trimmed))
            points_trimmed += np.expand_dims(delta_xyz[i], axis = 0)
            t_shift = len(p_combo)
            p_combo = np.concatenate((p_combo, points_trimmed), axis = 0)
            triangles_trimmed += t_shift
            t_combo = np.concatenate((t_combo, triangles_trimmed), axis = 0)

        self.set_from_triangles_and_points(t_combo, p_combo)

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

        tri_patch = rqstp.TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
        tri_patch.set_from_triangles_and_points(triangles, points)
        self.patch_list = [tri_patch]
        self.uuid = bu.new_uuid()

    def set_from_point_set(self,
                           point_set,
                           convexity_parameter = 5.0,
                           reorient = False,
                           reorient_max_dip = None,
                           extend_with_flange = False,
                           flange_point_count = 11,
                           flange_radial_factor = 10.0,
                           flange_radial_distance = None,
                           make_clockwise = False):
        """Populate this (empty) Surface object with a Delaunay triangulation of points in a PointSet object.

        arguments:
           point_set (PointSet): the set of points to be triangulated to form a surface
           convexity_parameter (float, default 5.0): controls how likely the resulting triangulation is to be
              convex; reduce to 1.0 to allow slightly more concavities; increase to 100.0 or more for very little
              chance of even a slight concavity
           reorient (bool, default False): if True, a copy of the points is made and reoriented to minimise the
              z range (ie. z axis is approximate normal to plane of points), to enhace the triangulation
           reorient_max_dip (float, optional): if present, the reorientation of perspective off vertical is
              limited to this angle in degrees
           extend_with_flange (bool, default False): if True, a ring of points is added around the outside of the
              points before the triangulation, effectively extending the surface with a flange
           flange_point_count (int, default 11): the number of points to generate in the flange ring; ignored if
              extend_with_flange is False
           flange_radial_factor (float, default 10.0): distance of flange points from centre of points, as a
              factor of the maximum radial distance of the points themselves; ignored if extend_with_flange is False
           flange_radial_distance (float, optional): if present, the minimum absolute distance of flange points from
              centre of points; units are those of the crs
           make_clockwise (bool, default False): if True, the returned triangles will all be clockwise when
              viewed in the direction -ve to +ve z axis; if reorient is also True, the clockwise aspect is
              enforced in the reoriented space

        returns:
           if extend_with_flange is True, numpy bool array with a value per triangle indicating flange triangles;
           if extent_with_flange is False, None

        notes:
           if extend_with_flange is True, then a boolean array is created for the surface, with a value per triangle,
           set to False (zero) for non-flange triangles and True (one) for flange triangles; this array is
           suitable for adding as a property for the surface, with indexable element 'faces';
           when flange extension occurs, the radius is the greater of the values determined from the radial factor
           and radial distance arguments
        """

        crs = rqc.Crs(self.model, uuid = point_set.crs_uuid)
        p = point_set.full_array_ref()
        if crs.xy_units == crs.z_units or not reorient:
            unit_adjusted_p = p
        else:
            unit_adjusted_p = p.copy()
            wam.convert_lengths(unit_adjusted_p[:, 2], crs.z_units, crs.xy_units)
        if reorient:
            p_xy, self.normal_vector, reorient_matrix = triangulate.reorient(unit_adjusted_p,
                                                                             max_dip = reorient_max_dip)
        else:
            p_xy = unit_adjusted_p
        if extend_with_flange:
            flange_points = triangulate.surrounding_xy_ring(p_xy,
                                                            count = flange_point_count,
                                                            radial_factor = flange_radial_factor,
                                                            radial_distance = flange_radial_distance)
            p_xy_e = np.concatenate((p_xy, flange_points), axis = 0)
            if reorient:
                # reorient back extenstion points into original p space
                flange_points_reverse_oriented = vec.rotate_array(reorient_matrix.T, flange_points)
                p_e = np.concatenate((unit_adjusted_p, flange_points_reverse_oriented), axis = 0)
            else:
                p_e = p_xy_e
        else:
            p_xy_e = p_xy
            p_e = unit_adjusted_p
        log.debug('number of points going into dt: ' + str(len(p_xy_e)))
        success = False
        try:
            t = triangulate.dt(p_xy_e[:, :2], container_size_factor = convexity_parameter, algorithm = "scipy")
            success = True
        except AssertionError:
            pass
        if not success:
            log.warning('triangulation failed, trying again with tiny perturbation of points')
            p_xy_e[:, :2] += (np.random.random((len(p_xy_e), 2)) - 0.5) * 0.001
            t = triangulate.dt(p_xy_e[:, :2], container_size_factor = convexity_parameter * 1.1)
        log.debug('number of triangles: ' + str(len(t)))
        if make_clockwise:
            triangulate.make_all_clockwise_xy(t, p_e)  # modifies t in situ
        if crs.xy_units != crs.z_units and reorient:
            wam.convert_lengths(p_e[:, 2], crs.xy_units, crs.z_units)
        self.crs_uuid = point_set.crs_uuid
        self.set_from_triangles_and_points(t, p_e)
        if extend_with_flange:
            flange_array = np.zeros(len(t), dtype = bool)
            flange_array[:] = np.where(np.any(t >= len(p), axis = 1), True, False)
            return flange_array
        return None

    def make_all_clockwise_xy(self, reorient = False):
        """Reorders cached triangles data such that all triangles are clockwise when viewed from -ve z axis.

        note:
           if reorient is set True, a copy of the points is first reoriented such that they lie roughly in
           the xy plane, and the clockwise-ness of the triangles is effected in that space
        """

        _, p = self.triangles_and_points()
        if reorient:
            unit_adjusted_p = self.unit_adjusted_points()
            p_xy, self.normal_vector, _ = triangulate.reorient(unit_adjusted_p)
        else:
            p_xy = p
        triangulate.make_all_clockwise_xy(self.triangles, p_xy)  # modifies t in situ
        # assert np.all(vec.clockwise_triangles(p_xy, self.triangles) >= 0.0)

    def unit_adjusted_points(self):
        """Returns cached points or copy thereof with z values adjusted to xy units."""

        _, p = self.triangles_and_points()
        crs = rqc.Crs(self.model, uuid = self.crs_uuid)
        if crs.xy_units == self.z_units:
            return p
        unit_adjusted_p = p.copy()
        wam.convert_lengths(unit_adjusted_p[:, 2], crs.z_units, crs.xy_units)
        return unit_adjusted_p

    def normal(self):
        """Returns a vector that is roughly normal to the surface.

        notes:
           the result becomes more meaningless the less planar the surface is;
           even for a parfectly planar surface, the result is approximate;
           true normal vector is found when xy & z units differ
        """

        if self.normal_vector is None:
            p = self.unit_adjusted_points()
            _, self.normal_vector, _ = triangulate.reorient(p)
        return self.normal_vector

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
        tri_patch = rqstp.TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
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
        tri_patch = rqstp.TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
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

        note:
           this method uses a single patch to represent the torn surface, whereas strictly the RESQML standard
           requires speparate patches where parts of a surface are completely disconnected
        """

        mesh_shape = mesh_xyz.shape
        assert len(mesh_shape) == 5 and mesh_shape[2:] == (2, 2, 3)
        tri_patch = rqstp.TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
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
        tri_patch = rqstp.TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
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
            tri_patch = rqstp.TriangulatedPatch(self.model, patch_index = p_index, crs_uuid = self.crs_uuid)
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

    def set_to_horizontal_plane(self, depth, box_xyz, border = 0.0, quad_triangles = False):
        """Populate this (empty) surface with a patch of two triangles.
        
        Triangles define a flat, horizontal plane at a given depth.

        arguments:
            depth (float): z value to use in all points in the triangulated patch
            box_xyz (float[2, 3]): the min, max values of x, y (&z) giving the area to be covered (z ignored)
            border (float): an optional border width added around the x,y area defined by box_xyz
            quad_triangles (bool, default False): if True, 4 triangles are used instead of 2

        :meta common:
        """

        tri_patch = rqstp.TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
        tri_patch.set_to_horizontal_plane(depth, box_xyz, border = border, quad_triangles = quad_triangles)
        self.patch_list = [tri_patch]
        self.uuid = bu.new_uuid()

    def set_to_triangle(self, corners):
        """Populate this (empty) surface with a patch of one triangle."""

        tri_patch = rqstp.TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
        tri_patch.set_to_triangle(corners)
        self.patch_list = [tri_patch]
        self.uuid = bu.new_uuid()

    def set_to_triangle_pair(self, corners):
        """Populate this (empty) surface with a patch of two triangles.

        arguments:
            corners (numpy float array of shape [2, 2, 3] or [4, 3]): 4 corners in logical ordering
        """

        tri_patch = rqstp.TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
        tri_patch.set_to_triangle_pair(corners.reshape((4, 3)))
        self.patch_list = [tri_patch]
        self.uuid = bu.new_uuid()

    def set_to_sail(self, n, centre, radius, azimuth, delta_theta):
        """Populate this (empty) surface with a patch representing a triangle wrapped on a sphere."""

        tri_patch = rqstp.TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
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
            v_index = None
            for line in lines:
                if "VRTX" in line:
                    words = line.rstrip().split(" ")
                    v_i = int(words[1])
                    if v_index is None:
                        v_index = v_i
                        index_offset = v_index
                    else:
                        assert v_i == v_index + 1, 'Tsurf vertex indices out of sequence'
                        v_index = v_i
                    vertices.append(words[2:5])
                elif "TRGL" in line:
                    triangles.append(line.rstrip().split(" ")[1:4])
        assert len(vertices) >= 3, 'vertices missing'
        assert len(triangles) > 0, 'triangles missing'
        triangles = np.array(triangles, dtype = int) - index_offset
        vertices = np.array(vertices, dtype = float)
        assert np.all(triangles >= 0) and np.all(triangles < len(vertices)), 'triangle vertex indices out of range'
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

    def line_intersection(self, line_p, line_v, line_segment = False):
        """Returns x,y,z of an intersection point of straight line with the surface, or None if no intersection found."""

        t, p = self.triangles_and_points()
        tp = p[t]
        intersects = meet.line_triangles_intersects(line_p, line_v, tp, line_segment = line_segment)
        indices = meet.intersects_indices(intersects)
        if not indices or len(indices) == 0:
            return None
        return intersects[indices[0]]

    def normal_vectors(self, add_as_property: bool = False) -> np.ndarray:
        """Returns the normal vectors for each triangle in the surface.
        
        arguments:
            add_as_property (bool): if True, face_surface_normal_vectors_array is added as a property to the model.

        returns:
            normal_vectors_array (np.ndarray): the normal vectors corresponding to each triangle in the surface.
        """
        crs = rqc.Crs(self.model, uuid = self.crs_uuid)
        triangles, points = self.triangles_and_points()
        if crs.xy_units != crs.z_units:
            points = points.copy()
            wam.convert_lengths(points[:, 2], crs.z_units, crs.xy_units)
        n_triangles = len(triangles)
        normal_vectors_array = np.empty((n_triangles, 3))
        for triangle_num in range(n_triangles):
            normal_vector = vec.triangle_normal_vector_numba(points[triangles[triangle_num]])
            if (normal_vector[2] > 0.0) == crs.z_inc_down:
                normal_vector *= -1.0
            normal_vectors_array[triangle_num] = normal_vector
        if add_as_property:
            pc = rqp.PropertyCollection()
            pc.set_support(support = self)
            crs = rqc.Crs(self.model, uuid = self.crs_uuid)
            pc.add_cached_array_to_imported_list(
                normal_vectors_array,
                "computed from surface",
                "normal vector",
                uom = None,  # uom not used for points properties
                property_kind = "normal vector",
                indexable_element = "faces",
                points = True)
            pc.write_hdf5_for_imported_list()
            pc.create_xml_for_imported_list_and_add_parts_to_model(find_local_property_kinds = True)
        return normal_vectors_array

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

        # todo: if crs_uuid is None, attempt to set to surface patch crs uuid (within patch loop, below)
        if crs_uuid is not None:
            self.crs_uuid = crs_uuid
        if self.crs_uuid is None:
            self.crs_uuid = self.model.crs_uuid  # maverick use of model's default crs
        if self.crs_uuid is None:
            crs = rqc.Crs(self.model)
            crs.create_xml()
            self.crs_uuid = crs.uuid
        assert self.crs_uuid is not None

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

            self.model.create_crs_reference(crs_uuid = self.crs_uuid, root = geom)

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
                crs_root = self.model.root_for_uuid(self.crs_uuid)
                self.model.create_reciprocal_relationship(tri_rep, 'destinationObject', crs_root, 'sourceObject')
                if self.represented_interpretation_root is not None:
                    self.model.create_reciprocal_relationship(tri_rep, 'destinationObject',
                                                              self.represented_interpretation_root, 'sourceObject')

                ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
                ext_node = self.model.root_for_part(ext_part)
                self.model.create_reciprocal_relationship(tri_rep, 'mlToExternalPartProxy', ext_node,
                                                          'externalPartProxyToMl')

        return tri_rep


def distill_triangle_points(t, p):
    """Returns a (triangles, points) pair with points distilled as only those used from p."""

    assert np.all(t < len(p))
    # find unique points used by triangles
    p_keep = np.unique(t)
    # note new point index for each old point that is being kept
    p_map = np.full(len(p), -1, dtype = int)
    p_map[p_keep] = np.arange(len(p_keep))
    # copy those unique points into a trimmed points array
    points_distilled = p[p_keep]
    # copy triangles, replacing p indices with compressed indices
    triangles_mapped = p_map[t]
    assert triangles_mapped.shape == t.shape
    assert np.all(triangles_mapped >= 0)
    assert np.all(triangles_mapped < len(points_distilled))

    return triangles_mapped, points_distilled
