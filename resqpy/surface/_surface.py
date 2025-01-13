"""Surface class based on RESQML TriangulatedSetRepresentation class."""

# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company
# GOCAD is also a trademark of Emerson

import logging

log = logging.getLogger(__name__)

import numpy as np
import math as maths

import resqpy.crs as rqc
import resqpy.lines as rql
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
        self._load_normal_vector_from_extra_metadata()

    @classmethod
    def from_list_of_patches(cls, model, patch_list, title, crs_uuid = None, extra_metadata = None):
        """Create a Surface from a prepared list of TriangulatedPatch objects.

        arguments:
            - model (Model): the model to which the surface will be associated
            - patch_list (list of TriangulatedPatch): the list of patches to be combined to form the surface
            - title (str): the citation title for the new surface
            - crs_uuid (uuid, optional): the uuid of a crs in model which the points are deemed to be in
            - extra_metadata (dict of (str: str), optional): extra metadata to add to the new surface

        returns:
            - new Surface comprised of a patch for each entry in the patch list

        notes:
            - the triangulated patch objects are used directly in the surface
            - the patches should not have had their hdf5 data written yet
            - the patch index values will be set, with any previous values ignored
            - the patches will be hijacked to the target model if their model is different
            - each patch will have its points converted in situ into the surface crs
            - if the crs_uuid argument is None, the crs_uuid is taken from the first patch
        """
        assert len(patch_list) > 0, 'attempting to create Surface from empty patch list'
        if crs_uuid is None:
            crs_uuid = patch_list[0].crs_uuid
            if model.uuid(uuid = crs_uuid) is None:
                model.copy_uuid_from_other_model(patch_list[0].model, crs_uuid)
        surf = cls(model, title = title, crs_uuid = crs_uuid, extra_metadata = extra_metadata)
        surf.patch_list = patch_list
        surf.crs_uuid = crs_uuid
        crs = rqc.Crs(model, uuid = crs_uuid)
        for i, patch in enumerate(surf.patch_list):
            assert patch.points is not None, f'points missing in patch {i} when making surface {title}'
            patch.index = i
            patch._set_t_type()
            if not bu.matching_uuids(patch.crs_uuid, crs_uuid):
                p_crs = rqc.Crs(patch.model, uuid = patch.crs_uuid)
                p_crs.convert_array_to(crs, patch.points)
            patch.model = model
        return surf

    @classmethod
    def from_list_of_patches_of_triangles_and_points(cls, model, t_p_list, title, crs_uuid, extra_metadata = None):
        """Create a Surface from a prepared list of pairs of (triangles, points).

        arguments:
            - model (Model): the model to which the surface will be associated
            - t_p_list (list of (numpy int array, numpy float array)): the list of patches of triangles and points;
              the int arrays have shape (N, 3) being the triangle vertex indices of points; the float array has
              shape (M, 3) being the xyx values for the points, in the crs identified by crs_uuid
            - title (str): the citation title for the new surface
            - crs_uuid (uuid): the uuid of a crs in model which the points are deemed to be in
            - extra_metadata (dict of (str: str), optional): extra metadata to add to the new surface

        returns:
            - new Surface comprised of a patch for each entry in the list of pairs of triangles and points data

        note:
            - each entry in the t_p_list will have its own patch in the resulting surface, indexed in order of list
        """
        assert t_p_list, f'no triangles and points pairs in list when generating surface: {title}'
        assert crs_uuid is not None
        patch_list = []
        for i, (t, p) in enumerate(t_p_list):
            patch = rqs.TriangulatedPatch(model, patch_index = i, crs_uuid = crs_uuid)
            patch.set_from_triangles_and_points(t, p)
            patch_list.append(patch)
        return cls.from_list_of_patches(model, patch_list, title, crs_uuid = crs_uuid, extra_metadata = extra_metadata)

    @classmethod
    def from_tri_mesh(cls, tri_mesh, exclude_nans = False):
        """Create a Surface from a TriMesh.

        arguments:
            tri_mesh (TriMesh): the tri mesh for which an equivalent Surface is required
            exclude_nans (bool, default False): if True, and tri mesh point involving a not-a-number is
                excluded from the surface points, along with any triangle that has such a point as a vertex

        returns:
            a new Surface using the points and triangles of the tri mesh

        note:
            this method does not write hdf5 data nor create xml for the new Surface
        """
        assert isinstance(tri_mesh, rqs.TriMesh)
        surf = cls(tri_mesh.model,
                   crs_uuid = tri_mesh.crs_uuid,
                   title = tri_mesh.title,
                   surface_role = tri_mesh.surface_role,
                   extra_metadata = tri_mesh.extra_metadata if hasattr(tri_mesh, 'extra_metadata') else None)
        t, p = tri_mesh.triangles_and_points()
        if exclude_nans:
            t, p = nan_removed_triangles_and_points(t, p)
        surf.set_from_triangles_and_points(t, p)
        surf.represented_interpretation_root = tri_mesh.represented_interpretation_root
        return surf

    @classmethod
    def from_downsampling_surface(cls,
                                  fine_surface,
                                  point_count = None,
                                  title = None,
                                  target_model = None,
                                  inherit_extra_metadata = True,
                                  inherit_represented_interpretation = True,
                                  convexity_parameter = 5.0):
        """Create a Surface by taking a random subset of points from an existing surface and re-triangulating.

        arguments:
           fine_surface (Surface): a finely triangulated existing surface
           point_count (int, optional): the number of randomly selected points to keep; defaults to 1% of
              the fine surface point count
           title (str, optional): the title for the new (coarse) surface; defaults to the title of the
              fine surface
           target_model (Model, optional): the model to receive the new surface; defaults to that of the
              fine surface
           inherit_extra_metadata (bool, default True): if True, the coarse surface will inherit any extra
              metadata that the fine surface has
           inherit_represented_interpretation (bool, default True): if True and the fine surface has a
              represented interpretation then it is inherited by the coarse surface
           convexity_parameter (float, default 5.0): controls how likely the resulting triangulation is to be
              convex; reduce to 1.0 to allow slightly more concavities; increase to 100.0 or more for very
              little chance of even a slight concavity

        returns:
            new Surface, without its hdf5 having been written, nor xml created

        note:
            if the fine surface does not have more points than point count, then a copy of that surface
            is returned without re-triangulating
        """

        assert fine_surface is not None
        if title is None:
            title = fine_surface.title
        inter_model = target_model is not None and target_model is not fine_surface.model
        if target_model is None:
            target_model = fine_surface.model
        fine_t, fine_p = fine_surface.triangles_and_points()
        if point_count is None:
            point_count = len(fine_p) // 100
        if point_count < 3:
            point_count = 3
        if inter_model:
            target_model.copy_uuid_from_other_model(fine_surface.model, fine_surface.crs_uuid)
            if inherit_represented_interpretation and fine_surface.represented_interpretation_root is not None:
                ri_uuid = fine_surface.model.uuid_for_root(fine_surface.represented_interpretation_root)
                target_model.copy_uuid_from_other_model(fine_surface.model, ri_uuid)
        em = (fine_surface.extra_metadata
              if inherit_extra_metadata and hasattr(fine_surface, 'extra_metadata') else None)
        surf = cls(target_model,
                   crs_uuid = fine_surface.crs_uuid,
                   title = title,
                   surface_role = fine_surface.surface_role,
                   extra_metadata = em)
        if point_count >= len(fine_p):
            t, p = fine_t, fine_p
        else:
            p = fine_p.copy()
            np.random.default_rng().shuffle(p, axis = 0)
            p = p[:point_count]
            t = triangulate.dt(p[:, :2], container_size_factor = convexity_parameter, algorithm = "scipy")
        surf.set_from_triangles_and_points(t, p)
        if inherit_represented_interpretation:
            surf.represented_interpretation_root = fine_surface.represented_interpretation_root
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

    def number_of_patches(self):
        """Returns the number of patches present in the surface."""

        self.extract_patches(self.root)
        return len(self.patch_list)

    def triangles_and_points(self, patch = None, copy = False):
        """Returns arrays representing one patch or a combination of all the patches in the surface.

        arguments:
           patch (int, optional): patch index; if None, combined arrays for all patches are returned
           copy (bool, default False): if True, a copy of the arrays is returned; if False, the cached
              arrays are returned

        returns:
           tuple (triangles, points):
              triangles (int array of shape[:, 3]): integer indices into points array,
                 being the nodes of the corners of the triangles;
              points (float array of shape[:, 3]): flat array of xyz points, indexed by triangles

        :meta common:
        """

        self.extract_patches(self.root)
        if patch is None:
            if self.triangles is None or self.points is None:
                if self.triangles is None:
                    points_offset = 0
                    for triangulated_patch in self.patch_list:
                        (t, p) = triangulated_patch.triangles_and_points()
                        if points_offset == 0:
                            self.triangles = t
                            self.points = p
                        else:
                            self.triangles = np.concatenate((self.triangles, t.copy() + points_offset))
                            self.points = np.concatenate((self.points, p))
                        points_offset += p.shape[0]
            if copy:
                return (self.triangles.copy(), self.points.copy())
            else:
                return (self.triangles, self.points)
        assert 0 <= patch < len(self.patch_list),  \
            ValueError(f'patch index {patch} out of range for surface with {len(self.patch_list)} patches')
        return self.patch_list[patch].triangles_and_points(copy = copy)

    def patch_index_for_triangle_index(self, triangle_index):
        """Returns the patch index for a triangle index (as applicable to triangles_and_points() triangles)."""
        if triangle_index is None or triangle_index < 0:
            return None
        self.extract_patches(self.root)
        if not self.patch_list:
            return None
        for i, patch in enumerate(self.patch_list):
            triangle_index -= patch.triangle_count
            if triangle_index < 0:
                return i
        return None

    def patch_indices_for_triangle_indices(self, triangle_indices, lazy = True):
        """Returns array of patch indices for array of triangle indices (as applicable to triangles_and_points() triangles)."""
        self.extract_patches(self.root)
        if not self.patch_list:
            return np.full(triangle_indices.shape, -1, dtype = np.int8)
        patch_count = len(self.patch_list)
        dtype = (np.int8 if patch_count < 127 else np.int32)
        if lazy and patch_count == 1:
            return np.zeros(triangle_indices.shape, dtype = np.int8)
        patch_limits = np.zeros(patch_count, dtype = np.int32)
        t_count = 0
        for p_i in range(patch_count):
            t_count += self.patch_list[p_i].triangle_count
            patch_limits[p_i] = t_count
        patches = np.empty(triangle_indices.shape, dtype = dtype)
        patches[:] = np.digitize(triangle_indices, patch_limits, right = False)
        if not lazy:
            patches[np.logical_or(triangle_indices < 0, patches == patch_count)] = -1
        return patches

    def decache_triangles_and_points(self):
        """Removes the cached composite triangles and points arrays."""
        self.points = None
        self.triangles = None

    def triangle_count(self, patch = None):
        """Return the numner of triangles in this surface, or in one patch.

        arguments:
           patch (int, optional): patch index; if None, a combined triangle count for all patches is returned

        returns:
           int being the number of trianges in the patch (if specified) or in all the patches
        """

        self.extract_patches(self.root)
        if patch is None:
            if not self.patch_list:
                return 0
            return np.sum([tp.triangle_count for tp in self.patch_list])
        assert 0 <= patch < len(self.patch_list),  \
            ValueError(f'patch index {patch} out of range for surface with {len(self.patch_list)} patches in triangle_count')
        return self.patch_list[patch].triangle_count

    def node_count(self, patch = None):
        """Return the number of nodes (points) used in this surface, or in one patch.

        arguments:
           patch (int, optional): patch index; if None, a combined node count for all patches is returned

        returns:
           int being the number of nodes in the patch (if specified) or in all the patches

        note:
           a multi patch surface might have more than one node colocated; this method will treat such coincident nodes
           as separate nodes
        """

        self.extract_patches(self.root)
        if patch is None:
            if not self.patch_list:
                return 0
            return np.sum([tp.node_count for tp in self.patch_list])
        assert 0 <= patch < len(self.patch_list),  \
            ValueError(f'patch index {patch} out of range for surface with {len(self.patch_list)} patches in node_count')
        return self.patch_list[patch].node_count

    def change_crs(self, required_crs):
        """Changes the crs of the surface, also sets a new uuid if crs changed.

        notes:
           this method is usually used to change the coordinate system for a temporary resqpy object;
           to add as a new part, call write_hdf5() and create_xml() methods;
           patches are maintained by this method;
           normal vector extra metadata item is updated if present; rotation matrix is removed
        """

        old_crs = rqc.Crs(self.model, uuid = self.crs_uuid)
        self.crs_uuid = required_crs.uuid
        if bu.matching_uuids(required_crs.uuid, old_crs.uuid) or not self.patch_list:
            log.debug(f'no crs change needed for {self.title}')
            return
        equivalent_crs = (required_crs == old_crs)
        log.debug(f'crs change needed for {self.title} from {old_crs.title} to {required_crs.title}')
        for patch in self.patch_list:
            assert bu.matching_uuids(patch.crs_uuid, old_crs.uuid)
            if not equivalent_crs:
                patch.triangles_and_points()
                required_crs.convert_array_from(old_crs, patch.points)
            patch.crs_uuid = self.crs_uuid
        self.triangles = None  # clear cached arrays for surface
        self.points = None
        if not equivalent_crs:
            if self.extra_metadata.pop('rotation matrix', None) is not None:
                log.debug(f'discarding rotation matrix extra metadata during crs change of: {self.title}')
            self._load_normal_vector_from_extra_metadata()
            if self.normal_vector is not None:
                if required_crs.z_inc_down != old_crs.z_inc_down:
                    self.normal_vector[2] = -self.normal_vector[2]
                theta = (wam.convert(required_crs.rotation, required_crs.rotation_units, 'dega') -
                         wam.convert(old_crs.rotation, old_crs.rotation_units, 'dega'))
                if not maths.isclose(theta, 0.0):
                    self.normal_vector = vec.rotate_vector(vec.rotation_matrix_3d_axial(2, theta), self.normal_vector)
                self.extra_metadata['normal vector'] = str(
                    f'{self.normal_vector[0]},{self.normal_vector[1]},{self.normal_vector[2]}')
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
        """Populate this (empty) surface with a version of a larger surface split by a straight xy line.

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
        t_type = np.int32 if len(pp) <= 2_147_483_647 else np.int64
        tp = np.empty(p.shape, dtype = t_type)
        tp[:, 0] = len(p)
        tp[:, 1] = len(p) + 1
        tp[:, 2] = np.arange(len(p), dtype = t_type)
        cw = vec.clockwise_triangles(pp, tp)
        pai = (cw >= 0.0)  # bool mask over p
        pbi = (cw <= 0.0)  # bool mask over p
        tap = pai[t]
        tbp = pbi[t]
        ta = np.any(tap, axis = 1)  # bool array over t
        tb = np.any(tbp, axis = 1)  # bool array over t

        # here we stick the two halves together into a single patch
        # todo: keep as two patches as required by RESQML business rules
        p_combo = np.empty((0, 3))
        t_combo = np.empty((0, 3), dtype = t_type)
        for i, tab in enumerate((ta, tb)):
            p_keep = np.unique(t[tab])
            # note new point index for each old point that is being kept
            p_map = np.full(len(p), -1, dtype = t_type)
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

    def distinct_edges(self, patch = None):
        """Returns a numpy int array of shape (N, 2) being the ordered node pairs of distinct edges of triangles.

        arguments:
           patch (int, optional): patch index; if None, a combination of edges for all patches is returned
        """

        triangles, _ = self.triangles_and_points(patch = patch)
        assert triangles is not None
        unique_edges, _ = triangulate.edges(triangles)
        return unique_edges

    def distinct_edges_and_counts(self, patch = None):
        """Returns unique edges as pairs of point indices, and a count of uses of each edge.

        arguments:
           patch (int, optional): patch index; if None, combined results for all patches are returned

        returns:
            numpy int array of shape (N, 2), numpy int array of shape (N,)
            where 2D array is list-like sorted points index pairs for unique edges
            and 1D array contains corresponding edge usage count (usually 1 or 2)

        notes:
            first entry in each pair is always the lower of the two point indices;
            for well formed surfaces, the count should everywhere be one or two;
            the function does not attempt to detect coincident points
        """

        triangles, _ = self.triangles_and_points(patch = patch)
        assert triangles is not None
        return triangulate.edges(triangles)

    def edge_lengths(self, required_uom = None, patch = None):
        """Returns float array of shape (N, 3) being triangle edge lengths.

        arguments:
            required_uom (str, optional): the required length uom for the resulting edge lengths; default is crs xy units
            patch (int, optional): patch index; if None, edge lengths for all patches are returned

        returns:
            numpy float array of shape (N, 3) where N is the number of triangles
        """

        t, p = self.triangles_and_points(patch = patch)
        crs = rqc.Crs(self.model, uuid = self.crs_uuid)
        if required_uom is None:
            required_uom = crs.xy_units
        if crs.xy_units != required_uom or crs.z_units != required_uom:
            p = p.copy()
            wam.convert_lengths(p[:, :2], crs.xy_units, required_uom)
            wam.convert_lengths(p[:, 2], crs.z_units, required_uom)
        t_end = np.empty_like(t)
        t_end[:, :2] = t[:, 1:]
        t_end[:, 2] = t[:, 0]
        edge_v = p[t_end, :] - p[t, :]
        return vec.naive_lengths(edge_v)

    def set_from_triangles_and_points(self, triangles, points):
        """Populate this (empty) Surface object from an array of triangle corner indices and an array of points."""

        tri_patch = rqstp.TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
        tri_patch.set_from_triangles_and_points(triangles, points)
        self.patch_list = [tri_patch]
        self.uuid = bu.new_uuid()
        self.triangles = triangles.copy()
        self.points = points.copy()

    def set_multi_patch_from_triangles_and_points(self, triangles_and_points_list):
        """Populate this (empty) Surface object from a list of paits: array of triangle corner indices, array of points."""

        self.patch_list = []
        self.trianges = None
        self.points = None
        for patch, entry in enumerate(triangles_and_points_list):
            assert len(entry) == 2, 'expecting pair of arrays (triangles, points) for each patch'
            triangles, points = entry
            tri_patch = rqstp.TriangulatedPatch(self.model, patch_index = patch, crs_uuid = self.crs_uuid)
            tri_patch.set_from_triangles_and_points(triangles, points)
            self.patch_list.append(tri_patch)
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
                           flange_inner_ring = False,
                           saucer_parameter = None,
                           make_clockwise = False,
                           normal_vector = None):
        """Populate this (empty) Surface object with a Delaunay triangulation of points in a PointSet object.

        arguments:
           point_set (PointSet): the set of points to be triangulated to form a surface
           convexity_parameter (float, default 5.0): controls how likely the resulting triangulation is to be
              convex; reduce to 1.0 to allow slightly more concavities; increase to 100.0 or more for very little
              chance of even a slight concavity
           reorient (bool, default False): if True, a copy of the points is made and reoriented to minimise the
              z range (ie. z axis is approximate normal to plane of points), to enhace the triangulation; if a
              normal_vector is supplied, the reorientation is based on that instead of minimising z
           reorient_max_dip (float, optional): if present, the reorientation of perspective off vertical is
              limited to this angle in degrees; ignored if normal_vector is specified
           extend_with_flange (bool, default False): if True, a ring of points is added around the outside of the
              points before the triangulation, effectively extending the surface with a flange
           flange_point_count (int, default 11): the number of points to generate in the flange ring; ignored if
              extend_with_flange is False
           flange_radial_factor (float, default 10.0): distance of flange points from centre of points, as a
              factor of the maximum radial distance of the points themselves; ignored if extend_with_flange is False
           flange_radial_distance (float, optional): if present, the minimum absolute distance of flange points from
              centre of points; units are those of the crs
           flange_inner_ring (bool, default False): if True, an inner ring of points, with double flange point counr,
              is created at a radius just outside that of the furthest flung original point; this improves
              triangulation of the extended point set when the original has a non-convex hull
           saucer_parameter (float, optional): if present, and extend_with_flange is True, then a parameter
              controlling the shift of flange points in a perpendicular direction away from the fault plane;
              see notes for how this parameter is interpreted
           make_clockwise (bool, default False): if True, the returned triangles will all be clockwise when
              viewed in the direction -ve to +ve z axis; if reorient is also True, the clockwise aspect is
              enforced in the reoriented space
           normal_vector (triple float, optional): if present and reorienting, the normal vector to use for reorientation;
              if None, the reorientation is made so as to minimise the z range

        returns:
           if extend_with_flange is True, numpy bool array with a value per triangle indicating flange triangles;
           if extend_with_flange is False, None

        notes:
           if extend_with_flange is True, then a boolean array is created for the surface, with a value per triangle,
           set to False (zero) for non-flange triangles and True (one) for flange triangles; this array is
           suitable for adding as a property for the surface, with indexable element 'faces';
           when flange extension occurs, the radius is the greater of the values determined from the radial factor
           and radial distance arguments;
           the saucer_parameter must be between -90.0 and 90.0, and is interpreted as an angle to apply out of
           the plane of the original points, to give a simple saucer shape; +ve angles result in the shift being in 
           the direction of the -ve z hemisphere; -ve angles result in the shift being in the +ve z hemisphere; in 
           either case the direction of the shift is perpendicular to the average plane of the original points;
           normal_vector, if supplied, should be in the crs of the point set
        """

        simple_saucer_angle = None
        if saucer_parameter is not None:
            assert -90.0 < saucer_parameter < 90.0, f'simple saucer angle parameter must be less than 90 degrees; too big: {saucer_parameter}'
            simple_saucer_angle = saucer_parameter
            saucer_parameter = None
        crs = rqc.Crs(self.model, uuid = point_set.crs_uuid)
        p = point_set.full_array_ref()
        assert p.ndim >= 2
        assert p.shape[-1] == 3
        p = p.reshape((-1, 3))
        nan_mask = np.isnan(p)
        if np.any(nan_mask):
            row_mask = np.logical_not(np.any(nan_mask, axis = -1))
            log.info(
                f'removing {len(p) - np.count_nonzero(row_mask)} NaN points from point set {point_set.title} prior to surface triangulation'
            )
            p = p[row_mask, :]
        if crs.xy_units == crs.z_units:
            unit_adjusted_p = p
        else:
            unit_adjusted_p = p.copy()
            wam.convert_lengths(unit_adjusted_p[:, 2], crs.z_units, crs.xy_units)
            # note: normal vector should already be for a crs with common xy  & z units
        # reorient the points to the fault normal vector
        if normal_vector is None:
            p_xy, self.normal_vector, reorient_matrix = triangulate.reorient(unit_adjusted_p,
                                                                             max_dip = reorient_max_dip)
        else:
            assert len(normal_vector) == 3
            self.normal_vector = np.array(normal_vector, dtype = np.float64)
            if self.normal_vector[2] < 0.0:
                self.normal_vector = -self.normal_vector
            incl = vec.inclination(normal_vector)
            if maths.isclose(incl, 0.0):
                reorient_matrix = vec.no_rotation_matrix()
                p_xy = unit_adjusted_p
            else:
                azi = vec.azimuth(normal_vector)
                reorient_matrix = vec.tilt_3d_matrix(azi, incl)
                p_xy = vec.rotate_array(reorient_matrix, unit_adjusted_p)
        if extend_with_flange:
            flange_points, radius = triangulate.surrounding_xy_ring(p_xy,
                                                                    count = flange_point_count,
                                                                    radial_factor = flange_radial_factor,
                                                                    radial_distance = flange_radial_distance,
                                                                    inner_ring = flange_inner_ring,
                                                                    saucer_angle = 0.0)
            flange_points_reverse_oriented = vec.rotate_array(reorient_matrix.T, flange_points)
            if reorient:
                p_xy_e = np.concatenate((p_xy, flange_points), axis = 0)
            else:
                p_xy_e = np.concatenate((unit_adjusted_p, flange_points_reverse_oriented), axis = 0)

        else:
            p_xy_e = unit_adjusted_p
            flange_array = None
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
            triangulate.make_all_clockwise_xy(t, p_xy_e)  # modifies t in situ
        if extend_with_flange:
            flange_array = np.zeros(len(t), dtype = bool)
            flange_array[:] = np.where(np.any(t >= len(p), axis = 1), True, False)
            if simple_saucer_angle is not None:
                assert abs(simple_saucer_angle) < 90.0
                z_shift = radius * maths.tan(vec.radians_from_degrees(simple_saucer_angle))
                flange_points[:, 2] -= z_shift
                flange_points_reverse_oriented = vec.rotate_array(reorient_matrix.T, flange_points)
            if crs.xy_units != crs.z_units:
                wam.convert_lengths(flange_points_reverse_oriented[:, 2], crs.xy_units, crs.z_units)
            p_e = np.concatenate((p, flange_points_reverse_oriented))
        else:
            p_e = p
        self.crs_uuid = point_set.crs_uuid
        self.set_from_triangles_and_points(t, p_e)
        return flange_array

    def extend_surface_with_flange(self,
                                   convexity_parameter = 5.0,
                                   reorient = False,
                                   reorient_max_dip = None,
                                   flange_point_count = 11,
                                   flange_radial_factor = 10.0,
                                   flange_radial_distance = None,
                                   flange_inner_ring = False,
                                   saucer_parameter = None,
                                   make_clockwise = False,
                                   retriangulate = False,
                                   normal_vector = None):
        """Returns a new Surface object where the original surface has been extended with a flange with a Delaunay triangulation of points in a PointSet object.

        arguments:
            convexity_parameter (float, default 5.0): controls how likely the resulting triangulation is to be
                convex; reduce to 1.0 to allow slightly more concavities; increase to 100.0 or more for very little
                chance of even a slight concavity
            reorient (bool, default False): if True, a copy of the points is made and reoriented to minimise the
                z range (ie. z axis is approximate normal to plane of points), to enhace the triangulation; if
                normal_vector is supplied that is used to determine the reorientation instead of minimising z
            reorient_max_dip (float, optional): if present, the reorientation of perspective off vertical is
                limited to this angle in degrees; ignored if normal_vector is specified
            flange_point_count (int, default 11): the number of points to generate in the flange ring; ignored if
                retriangulate is False
            flange_radial_factor (float, default 10.0): distance of flange points from centre of points, as a
                factor of the maximum radial distance of the points themselves; ignored if extend_with_flange is False
            flange_radial_distance (float, optional): if present, the minimum absolute distance of flange points from
                centre of points; units are those of the crs
            flange_inner_ring (bool, default False): if True, an inner ring of points, with double flange point counr,
                is created at a radius just outside that of the furthest flung original point; this improves
                triangulation of the extended point set when the original has a non-convex hull. Ignored if retriangulate
                is False
            saucer_parameter (float, optional): if present, and extend_with_flange is True, then a parameter
                controlling the shift of flange points in a perpendicular direction away from the fault plane;
                see notes for how this parameter is interpreted
            make_clockwise (bool, default False): if True, the returned triangles will all be clockwise when
                viewed in the direction -ve to +ve z axis; if reorient is also True, the clockwise aspect is
                enforced in the reoriented space
            retriangulate (bool, default False): if True, the surface will be generated with a retriangulation of
                the existing points. If False, the surface will be generated by adding flange points and triangles directly
                from the original surface edges, and will no retriangulate the input surface. If False the surface must not
                contain tears
            normal_vector (triple float, optional): if present and reorienting, the normal vector to use for reorientation;
                if None, the reorientation is made so as to minimise the z range

        returns:
            a new surface, and a boolean array of length N, where N is the number of triangles on the surface. This boolean
            array is False on original triangle points, and True for extended flange triangles

        notes:
            a boolean array is created for the surface, with a value per triangle, set to False (zero) for non-flange
            triangles and True (one) for flange triangles; this array is suitable for adding as a property for the
            surface, with indexable element 'faces';
            when flange extension occurs, the radius is the greater of the values determined from the radial factor
            and radial distance arguments;
            the saucer_parameter is interpreted in one of two ways: (1) +ve fractoinal values between zero and one
            are the fractional distance from the centre of the points to its rim at which to sample the surface for
            extrapolation and thereby modify the recumbent z of flange points; 0 will usually give shallower and
            smoother saucer; larger values (must be less than one) will lead to stronger and more erratic saucer
            shape in flange; (2) other values between -90.0 and 90.0 are interpreted as an angle to apply out of
            the plane of the original points, to give a simple (and less computationally demanding) saucer shape;
            +ve angles result in the shift being in the direction of the -ve z hemisphere; -ve angles result in
            the shift being in the +ve z hemisphere; in either case the direction of the shift is perpendicular
            to the average plane of the original points;
            normal_vector, if supplied, should be in the crs of this surface
        """
        prev_t, prev_p = self.triangles_and_points()
        point_set = rqs.PointSet(self.model, crs_uuid = self.crs_uuid, title = self.title, points_array = prev_p)
        if retriangulate:
            out_surf = Surface(self.model, crs_uuid = self.crs_uuid, title = self.title)
            return out_surf, out_surf.set_from_point_set(point_set, convexity_parameter, reorient, reorient_max_dip,
                                                         True, flange_point_count, flange_radial_factor,
                                                         flange_radial_distance, flange_inner_ring, saucer_parameter,
                                                         make_clockwise, normal_vector)
        else:
            simple_saucer_angle = None
            if saucer_parameter is not None and (saucer_parameter > 1.0 or saucer_parameter < 0.0):
                assert -90.0 < saucer_parameter < 90.0, f'simple saucer angle parameter must be less than 90 degrees; too big: {saucer_parameter}'
                simple_saucer_angle = saucer_parameter
                saucer_parameter = None
            assert saucer_parameter is None or 0.0 <= saucer_parameter < 1.0
            crs = rqc.Crs(self.model, uuid = point_set.crs_uuid)
            assert prev_p.ndim >= 2
            assert prev_p.shape[-1] == 3
            p = prev_p.reshape((-1, 3))
            if crs.xy_units == crs.z_units or not reorient:
                unit_adjusted_p = p
            else:
                unit_adjusted_p = p.copy()
                wam.convert_lengths(unit_adjusted_p[:, 2], crs.z_units, crs.xy_units)
            if reorient:
                p_xy, normal, reorient_matrix = triangulate.reorient(unit_adjusted_p, max_dip = reorient_max_dip)
            else:
                p_xy = unit_adjusted_p
                normal = self.normal()
                reorient_matrix = None

            centre_point = np.nanmean(p_xy.reshape((-1, 3)), axis = 0)  # work out the radius for the flange points
            p_radius_v = np.nanmax(np.abs(p.reshape((-1, 3)) - np.expand_dims(centre_point, axis = 0)), axis = 0)[:2]
            p_radius = maths.sqrt(np.sum(p_radius_v * p_radius_v))
            radius = p_radius * flange_radial_factor
            if flange_radial_distance is not None and flange_radial_distance > radius:
                radius = flange_radial_distance

            de, dc = self.distinct_edges_and_counts()  # find the distinct edges and counts
            unique_edge = de[dc == 1]  # find hull edges (edges on only a single triangle)
            hull_points = p_xy[unique_edge]  # find points defining the hull edges
            hull_centres = np.mean(hull_points, axis = 1)  # find the centre of each edge

            flange_points = np.empty(
                shape = (hull_centres.shape), dtype = float
            )  # loop over all the hull centres, generating a flange point and finding the azimuth from the centre to the hull centre point
            az = np.empty(shape = len(hull_centres), dtype = float)
            for i, c in enumerate(hull_centres):
                v = [centre_point[0] - c[0], centre_point[1] - c[1], centre_point[2] - c[2]]
                uv = -vec.unit_vector(v)
                az[i] = vec.azimuth(uv)
                flange_point = centre_point + radius * uv
                if simple_saucer_angle is not None:
                    z_shift = radius * maths.tan(vec.radians_from_degrees(simple_saucer_angle))
                    if reorient:
                        flange_point[2] -= z_shift
                    else:
                        flange_point -= (-vec.unit_vector(normal) * z_shift)
                flange_points[i] = flange_point

            sort_az_ind = np.argsort(np.array(az))  # sort by azimuth, to run through the hull points
            new_points = np.empty(shape = (len(flange_points), 3), dtype = float)
            new_triangles = np.empty(shape = (len(flange_points) * 2, 3), dtype = int)
            point_offset = len(p_xy)  # the indices of the new triangles will begin after this
            for i, ind in enumerate(sort_az_ind):  # loop over each point in azimuth order
                new_points[i] = flange_points[ind]
                this_hull_edge = unique_edge[ind]

                def az_for_point(c):
                    v = [centre_point[0] - c[0], centre_point[1] - c[1], centre_point[2] - c[2]]
                    uv = -vec.unit_vector(v)
                    return vec.azimuth(uv)

                this_edge_az_sort = np.array(
                    [az_for_point(p_xy[this_hull_edge[0]]),
                     az_for_point(p_xy[this_hull_edge[1]])])
                if np.min(this_edge_az_sort) < az[ind] < np.max(this_edge_az_sort):
                    first, second = np.argsort(this_edge_az_sort)
                else:
                    second, first = np.argsort(this_edge_az_sort)
                if i != len(sort_az_ind) - 1:
                    new_triangles[2 * i] = np.array(
                        [this_hull_edge[first], this_hull_edge[second],
                         i + point_offset])  # add a triangle between the two hull points and the flange point
                    new_triangles[(2 * i) + 1] = np.array(
                        [this_hull_edge[second], i + point_offset,
                         i + point_offset + 1])  # for all but the last point, hookup triangle to the next flange point
                else:
                    new_triangles[2 * i] = np.array(
                        [this_hull_edge[first], this_hull_edge[second],
                         i + point_offset])  # add a triangle between the two hull points and the first flange point
                    new_triangles[(2 * i) + 1] = np.array(
                        [this_hull_edge[second], point_offset,
                         i + point_offset])  # add in the final triangle between the first and last flange points

            all_points = np.concatenate((p_xy, new_points))  # concatenate triangle and points arrays
            all_triangles = np.concatenate((prev_t, new_triangles))

            flange_array = np.zeros(shape = all_triangles.shape[0], dtype = bool)
            flange_array[
                len(prev_t):] = True  # make a flange bool array, where all new triangles are flange and therefore True

            assert len(all_points) == (
                point_offset + len(flange_points)), "New point count should be old point count + flange point count"
            assert len(all_triangles) == (
                len(prev_t) +
                2 * len(flange_points)), "New triangle count should be old triangle count + 2 x #flange points"

            if saucer_parameter is not None:
                _adjust_flange_z(self.model, crs.uuid, all_points, len(all_points), all_triangles, flange_array,
                                 saucer_parameter)  # adjust the flange points if in saucer mode
            if reorient:
                all_points = vec.rotate_array(reorient_matrix.T, all_points)
            if crs.xy_units != crs.z_units and reorient:
                wam.convert_lengths(all_points[:, 2], crs.xy_units, crs.z_units)

            if make_clockwise:
                triangulate.make_all_clockwise_xy(all_triangles, all_points)  # modifies t in situ

            out_surf = Surface(self.model, crs_uuid = self.crs_uuid, title = self.title)
            out_surf.set_from_triangles_and_points(all_triangles, all_points)  # create the new surface
            return out_surf, flange_array

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
        if crs.xy_units == crs.z_units:
            return p
        unit_adjusted_p = p.copy()
        wam.convert_lengths(unit_adjusted_p[:, 2], crs.z_units, crs.xy_units)
        return unit_adjusted_p

    def normal(self):
        """Returns a vector that is roughly normal to the surface.

        notes:
           the result becomes more meaningless the less planar the surface is;
           even for a parfectly planar surface, the result is approximate;
           true normal vector is found when xy & z units differ, ie. for consistent units
        """

        self._load_normal_vector_from_extra_metadata()
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
                    words = line.rstrip().split()
                    v_i = int(words[1])
                    if v_index is None:
                        v_index = v_i
                        index_offset = v_index
                    else:
                        assert v_i == v_index + 1, 'Tsurf vertex indices out of sequence'
                        v_index = v_i
                    vertices.append(words[2:5])
                elif "TRGL" in line:
                    triangles.append(line.rstrip().split()[1:4])
        assert len(vertices) >= 3, 'vertices missing'
        assert len(triangles) > 0, 'triangles missing'
        t_type = np.int32 if len(vertices) <= 2_147_483_647 else np.int64
        triangles = np.array(triangles, dtype = t_type) - index_offset
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

    def line_intersection(self, line_p, line_v, line_segment = False, patch = None):
        """Returns x,y,z of an intersection point of straight line with the surface, or None if no intersection found."""

        t, p = self.triangles_and_points(patch = patch)
        tp = p[t]
        intersects = meet.line_triangles_intersects(line_p, line_v, tp, line_segment = line_segment)
        indices = meet.intersects_indices(intersects)
        if not indices or len(indices) == 0:
            return None
        return intersects[indices[0]]

    def sample_z_at_xy_points(self, points, multiple_handling = 'any', patch = None):
        """Returns interpolated z values for an array of xy values.

        arguments:
            points (numpy float array of shape (..., 2 or 3)): xy points to sample surface at (z values ignored)
            multiple_handling (str, default 'any'): one of 'any', 'minimum', 'maximum', 'exception'
            patch (int, optional): patch index; if None, results are for the full surface

        returns:
            numpy float array of shape points.shape[:-1] being z values interpolated from the surface z values

        notes:
            points must be in the same crs as the surface;
            NaN will be set for any points that do not intersect with the patch or surface in the xy projection;
            multiple_handling argument controls behaviour when one sample point intersects more than once:
            'any' a random one of the intersection z values is returned; 'minimum' or 'maximum': the
            numerical min or max of the z values is returned; 'exception': a ValueError is raised
        """

        assert points.ndim > 1 and 2 <= points.shape[-1] <= 3
        assert multiple_handling in ['any', 'minimum', 'maximum', 'exception'],  \
            f'invalid multiple handling mode: {multiple_handling}'
        if points.shape[-1] == 3:
            sample_xy = points.reshape((-1, 3))
        else:
            sample_xy = np.zeros((points.size // 2, 3), dtype = float)
            sample_xy[:, :2] = points.reshape((-1, 2))
        t, p = self.triangles_and_points(patch = patch)
        p_list = vec.points_in_triangles_njit(sample_xy, p[t], 1)
        vertical = np.array((0.0, 0.0, 1.0), dtype = float)
        z = np.full(sample_xy.shape[0], np.nan, dtype = float)
        for triangle_index, p_index, _ in p_list:
            # todo: replace following with cheaper trilinear interpolation, given vertical intersection line
            xyz = meet.line_triangle_intersect_numba(sample_xy[p_index], vertical, p[t[triangle_index]], t_tol = 0.05)
            if np.isnan(z[p_index]) or multiple_handling == 'any':
                z[p_index] = xyz[2]
            elif multiple_handling == 'minimum':
                if xyz[2] < z[p_index]:
                    z[p_index] = xyz[2]
            elif multiple_handling == 'maximum':
                if xyz[2] > z[p_index]:
                    z[p_index] = xyz[2]
            else:
                raise ValueError(f'multiple {self.title} surface intersections at xy: {sample_xy[p_index]}')
        return z.reshape(points.shape[:-1])

    def normal_vectors(self, add_as_property: bool = False, patch = None) -> np.ndarray:
        """Returns the normal vectors for each triangle in the patch or surface.

        arguments:
            add_as_property (bool): if True, face_surface_normal_vectors_array is added as a property to the model
            patch (int, optional): patch index; if None, normal vectors for triangles in all patches are returned

        returns:
            normal_vectors_array (np.ndarray): the normal vectors corresponding to each triangle in the surface
        """
        crs = rqc.Crs(self.model, uuid = self.crs_uuid)
        triangles, points = self.triangles_and_points(patch = patch)
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

    def axial_edge_crossings(self, axis, value = 0.0):
        """Returns list-like array of points interpolated from triangle edges that cross value in axis.

        arguments:
            axis (int): xyz axis; 0 for crossings in x axis, 1 for y, 2 for z
            value (float, default 0.0): the value in axis at which crossing points are sought

        returns:
            numpy float array of shape (N, 3) being interpolated triangle edge points with the crossing
                value in axis; or None if no crossings found
        """

        t, p = self.triangles_and_points()
        zero = p[:, axis] - value  # +ve one side of value, -ve the other side
        abs_zero = np.expand_dims(np.abs(zero), axis = -1)  # used to weight ends of crossing edges to interpolate
        e, _ = triangulate.edges(t)  # list-like pairs of p indices for ends of distinct edges

        # find crossing edges
        crossing = np.where(zero[e[:, 0]] * zero[e[:, 1]] < 0.0)  # zero crossing edge indices in e

        if not len(crossing[0]):
            return None

        ec = e[crossing]

        s = abs_zero[ec[:, 0]] + abs_zero[ec[:, 1]]
        # TODO: ignore divide by zero
        ep = (p[ec[:, 0]] * abs_zero[ec[:, 1]] + p[ec[:, 1]] * abs_zero[ec[:, 0]]) / s
        # TODO: unignore, and use midpoint of those edges?

        return ep

    def resampled_surface(self, title = None):
        """Creates a new surface which is a refined version of this surface; each triangle is divided equally into 4 new triangles.

        arguments:
            title (str): title for the output triangulated set, if None the title will be inherited from the input surface

        returns:
            resqpy.surface.Surface object, with extra_metadata ('resampled from surface': uuid), where uuid is for the original surface uuid
        """
        rt, rp = self.triangles_and_points()
        edge1 = np.mean(rp[rt[:]][:, ::2, :], axis = 1)
        edge2 = np.mean(rp[rt[:]][:, 1:, :], axis = 1)
        edge3 = np.mean(rp[rt[:]][:, :2, :], axis = 1)
        allpoints = np.concatenate((rp, edge1, edge2, edge3), axis = 0)
        count1 = len(rp)
        count2 = count1 + len(edge1)
        count3 = count2 + len(edge2)
        tris = []
        for i in range(len(rt)):
            tris.extend([[rt[i][0], count1 + i, count3 + i], [rt[i][1], count2 + i, count3 + i],
                         [rt[i][2], count1 + i, count2 + i], [count1 + i, count2 + i, count3 + i]])

        # TODO: implement alternate solution using edge functions in olio triangulation to optimise
        points_unique, inverse = np.unique(allpoints, axis = 0, return_inverse = True)
        t_type = np.int32 if len(allpoints) <= 2_147_483_647 else np.int64
        tris = np.array(tris, dtype = t_type)
        tris_unique = np.empty(shape = tris.shape, dtype = t_type)
        tris_unique[:, 0] = inverse[tris[:, 0]]
        tris_unique[:, 1] = inverse[tris[:, 1]]
        tris_unique[:, 2] = inverse[tris[:, 2]]

        if title is None:
            title = self.citation_title
        resampled = rqs.Surface(self.model,
                                title = title,
                                crs_uuid = self.crs_uuid,
                                extra_metadata = {'resampled from surface': str(self.uuid)})
        resampled.set_from_triangles_and_points(tris_unique, points_unique)

        return resampled

    def resample_surface_unique_edges(self):
        """Returns a new surface, with the same model, title and crs as the original, but with additional refined points along tears and edges.

        Each edge forming a tear or outer edge in the surface will have 3 additional points added, with 2 additional points
        on each edge of the original triangle. The output surface is re-triangulated using these new points (tears will be filled)

        returns: 
            new Surface object with extra_metadata ('unique edges resampled from surface': uuid), where uuid is for the original surface uuid
            
        note:
            this method involves a tr-triangulation
        """
        _, op = self.triangles_and_points()
        ref = self.resampled_surface()  # resample the original surface
        rt, rp = ref.triangles_and_points()
        de, dc = ref.distinct_edges_and_counts()  # find the distinct edges and counts for the resampled surface
        de_edge = de[dc == 1]  # find edges that only appear once - tears or surface edges
        edge_tri_index = np.sum(np.isin(rt, de_edge), axis = 1) == 2
        edge_tris = rp[rt[edge_tri_index]]
        mid = np.mean(rp[de_edge], axis = 1)  # get the midpoint of each surface edge
        edge_ref_points = np.unique(np.concatenate([op, edge_tris.reshape(-1, 3), mid]), axis = 0)  # combine all points

        points = rqs.PointSet(self.model, points_array = edge_ref_points, title = self.title,
                              crs_uuid = self.crs_uuid)  # generate a pointset from these points

        output = Surface(self.model, point_set = points,
                         extra_metadata = {'resampled from surface': str(self.uuid)
                                          })  # return a surface with generated from these points

        return output

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

        em = None
        if self.normal_vector is not None and (self.extra_metadata is None or
                                               'normal vector' not in self.extra_metadata):
            assert len(self.normal_vector) == 3
            em = {'normal vector': f'{self.normal_vector[0]},{self.normal_vector[1]},{self.normal_vector[2]}'}

        tri_rep = super().create_xml(add_as_part = False, title = title, originator = originator, extra_metadata = em)

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

    def _load_normal_vector_from_extra_metadata(self):
        if self.normal_vector is None and self.extra_metadata is not None:
            nv_str = self.extra_metadata.get('normal vector')
            if nv_str is not None:
                nv_words = nv_str.split(',')
                assert len(nv_words) == 3, f'failed to convert normal vector string into triplet: {nv_str}'
                self.normal_vector = np.empty(3, dtype = float)
                for i in range(3):
                    self.normal_vector[i] = float(nv_words[i])


def distill_triangle_points(t, p):
    """Returns a (triangles, points) pair with points distilled as only those used from p."""

    assert np.all(t < len(p))
    # find unique points used by triangles
    p_keep = np.unique(t)
    # note new point index for each old point that is being kept
    t_type = np.int32 if len(p) <= 2_147_483_647 else np.int64
    p_map = np.full(len(p), -1, dtype = t_type)
    p_map[p_keep] = np.arange(len(p_keep))
    # copy those unique points into a trimmed points array
    points_distilled = p[p_keep]
    # copy triangles, replacing p indices with compressed indices
    triangles_mapped = p_map[t]
    assert triangles_mapped.shape == t.shape
    assert np.all(triangles_mapped >= 0)
    assert np.all(triangles_mapped < len(points_distilled))

    return triangles_mapped, points_distilled


def nan_removed_triangles_and_points(t, p):
    """Returns a (triangles, points) pair which excludes any point involving a NaN and related triangles."""
    assert p.ndim == 2 and p.shape[1] == 3
    assert t.ndim == 2 and t.shape[1] == 3
    p_nan_mask = np.any(np.isnan(p), axis = 1)
    p_non_nan_mask = np.logical_not(p_nan_mask)
    expanded_mask = np.empty(p.shape, dtype = bool)
    expanded_mask[:] = np.expand_dims(p_non_nan_mask, axis = -1)
    p_filtered = p[expanded_mask].reshape((-1, 3))
    t_nan_mask = np.any(p_nan_mask[t], axis = 1)
    expanded_mask = np.empty(t.shape, dtype = bool)
    expanded_mask[:] = np.expand_dims(np.logical_not(t_nan_mask), axis = -1)
    t_filtered = t[expanded_mask].reshape((-1, 3))
    # modified the filtered t values to adjust for the compression of filtered p
    t_type = np.int32 if len(p) <= 2_147_483_647 else np.int64
    p_map = np.full(len(p), -1, dtype = t_type)
    p_map[p_non_nan_mask] = np.arange(len(p_filtered), dtype = t_type)
    t_filtered = p_map[t_filtered]
    assert t_filtered.ndim == 2 and t_filtered.shape[1] == 3
    assert not np.any(t_filtered < 0) and not np.any(t_filtered >= len(p_filtered))
    return (t_filtered, p_filtered)


def _adjust_flange_z(model, crs_uuid, p_xy_e, flange_start_index, t, flange_array, saucer_parameter):
    """Adjust the flange point z values (in recumbent space) by extrapolation of pair of points on original."""

    # reconstruct the hull (could be concave) of original points
    all_edges, edge_use_count = triangulate.edges(t)
    inner_edges = triangulate.internal_edges(all_edges, edge_use_count)
    t_for_inner_edges = triangulate.triangles_using_edges(t, inner_edges)
    assert np.all(t_for_inner_edges >= 0)
    flange_pairs = flange_array[t_for_inner_edges]
    rim_edges = inner_edges[np.where(flange_pairs[:, 0] != flange_pairs[:, 1])]
    assert rim_edges.ndim == 2 and rim_edges.shape[1] == 2 and len(rim_edges) > 0
    rim_edge_index_list, rim_point_index_list = triangulate.rims(rim_edges)
    assert len(rim_edge_index_list) == 1 and len(rim_point_index_list) == 1
    rim_edge_indices = rim_edge_index_list[0]
    rim_point_indices = rim_point_index_list[0]  # ordered list of points on original hull (could be concave)
    rim_pl = rql.Polyline(model,
                          set_coord = p_xy_e[rim_point_indices],
                          set_crs = crs_uuid,
                          is_closed = True,
                          title = 'rim')

    centre = np.mean(p_xy_e[:flange_start_index], axis = 0)
    # for each flange point, intersect a line from centre with the rim, and sample surface at saucer parameter
    for flange_pi in range(flange_start_index, len(p_xy_e)):
        f_xyz = p_xy_e[flange_pi]
        pl_seg, rim_x, rim_y = rim_pl.first_line_intersection(centre[0], centre[1], f_xyz[0], f_xyz[1])
        assert pl_seg is not None
        rim_xyz = rim_pl.segment_xyz_from_xy(pl_seg, rim_x, rim_y)
        sample_p = (1.0 - saucer_parameter) * centre + saucer_parameter * rim_xyz
        p_list = vec.points_in_triangles_njit(np.expand_dims(sample_p, axis = 0), p_xy_e[t], 1)
        vertical = np.array((0.0, 0.0, 1.0), dtype = float)
        assert len(p_list) > 0
        triangle_index, p_index, _ = p_list[0]
        start_xyz = meet.line_triangle_intersect_numba(sample_p, vertical, p_xy_e[t[triangle_index]], t_tol = 0.05)
        v_to_rim = rim_xyz - start_xyz
        v_to_flange_p = f_xyz - start_xyz
        if abs(v_to_rim[0]) > abs(v_to_rim[1]):
            f = (v_to_rim[0]) / (v_to_flange_p[0])
        else:
            f = (v_to_rim[1]) / (v_to_flange_p[1])
        assert 0.0 < f < 1.0
        z = (rim_xyz[2] - start_xyz[2]) / f + start_xyz[2]
        p_xy_e[flange_pi, 2] = z
