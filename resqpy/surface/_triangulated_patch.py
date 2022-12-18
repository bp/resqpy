"""TriangulatedPatch class used by Surface class."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.olio.xml_et as rqet


class TriangulatedPatch:
    """Class for RESQML TrianglePatch objects (used by Surface objects inter alia)."""

    def __init__(self, parent_model, patch_index = None, patch_node = None, crs_uuid = None):
        """Create an empty TriangulatedPatch (TrianglePatch) node and optionally load from xml.

        note:
           not usually instantiated directly by application code
        """

        self.model = parent_model
        self.node = patch_node
        self.patch_index = patch_index  # if not None and extracting from xml, patch_index must match xml
        self.triangle_count = 0
        self.node_count = 0
        self.triangles = None
        self.quad_triangles = None
        self.ni = None  # used to convert a triangle index back into a (j, i) pair when freshly built from mesh
        self.points = None
        self.crs_uuid = crs_uuid
        if patch_node is not None:
            xml_patch_index = rqet.find_tag_int(patch_node, 'PatchIndex')
            assert xml_patch_index is not None
            if self.patch_index is not None:
                assert self.patch_index == xml_patch_index, 'triangle patch index mismatch'
            else:
                self.patch_index = xml_patch_index
            self.triangle_count = rqet.find_tag_int(patch_node, 'Count')
            assert self.triangle_count is not None
            self.node_count = rqet.find_tag_int(patch_node, 'NodeCount')
            assert self.node_count is not None
            self.extract_crs_root_and_uuid()
            assert self.crs_uuid is not None

    def extract_crs_root_and_uuid(self):
        """Caches uuid for coordinate reference system, as stored in geometry xml sub-tree."""

        if self.crs_uuid is None:
            crs_root = rqet.find_nested_tags(self.node, ['Geometry', 'LocalCrs'])
            assert crs_root is not None, 'failed to find crs reference in triangulated patch xml'
            self.crs_uuid = bu.uuid_from_string(rqet.find_tag_text(crs_root, 'UUID'))
        else:
            crs_root = self.model.root_for_uuid(self.crs_uuid)
        return crs_root, self.crs_uuid

    def triangles_and_points(self):
        """Returns arrays representing the patch.

        Returns:
           Tuple (triangles, points):

           * triangles (int array of shape[:, 3]): integer indices into points array,
             being the nodes of the corners of the triangles
           * points (float array of shape[:, 3]): flat array of xyz points, indexed by triangles
        """
        if self.triangles is not None:
            return (self.triangles, self.points)
        assert self.triangle_count is not None and self.node_count is not None

        geometry_node = rqet.find_tag(self.node, 'Geometry')
        assert geometry_node is not None
        p_root = rqet.find_tag(geometry_node, 'Points')
        assert p_root is not None, 'Points xml node not found for triangle patch'
        assert rqet.node_type(p_root) == 'Point3dHdf5Array'
        h5_key_pair = self.model.h5_uuid_and_path_for_node(p_root, tag = 'Coordinates')
        if h5_key_pair is None:
            return (None, None)
        try:
            self.model.h5_array_element(h5_key_pair,
                                        cache_array = True,
                                        object = self,
                                        array_attribute = 'points',
                                        dtype = 'float')
        except Exception:
            log.error('hdf5 points failure for triangle patch ' + str(self.patch_index))
            raise
        triangles_node = rqet.find_tag(self.node, 'Triangles')
        h5_key_pair = self.model.h5_uuid_and_path_for_node(triangles_node)
        if h5_key_pair is None:
            log.warning('No Triangles found in xml for patch index: ' + str(self.patch_index))
            return (None, None)
        try:
            self.model.h5_array_element(h5_key_pair,
                                        cache_array = True,
                                        object = self,
                                        array_attribute = 'triangles',
                                        dtype = 'int')
        except Exception:
            log.error('hdf5 triangles failure for triangle patch ' + str(self.patch_index))
            raise
        return (self.triangles, self.points)

    def set_to_trimmed_patch(self, larger_patch, xyz_box = None, xy_polygon = None, internal = False):
        """Populate this (empty) patch with triangles and points that overlap with a trimming volume.

        arguments:
            larger_patch (TriangulatedPatch): the larger patch, a copy of which is to be trimmed
            xyz_box (numpy float array of shape (2, 3), optional): if present, a cuboid in xyz space
               against which to trim the patch
            xy_polygon (closed convex resqpy.lines.Polyline, optional): if present, an xy boundary
               against which to trim
            internal (bool, default False): if True, only those triangles where all three vertices
               are wtihin the trimming space are kept; if False, triangles with at least one vertex
               within the space are kept

        notes:
            at least one of xyz_box or xy_polygon must be present; if both are present, a triangle
            must be within both boundaries to survive the trimming;
            xyz_box and xy_polygon must be in the same crs as the larger patch
        """

        log.debug('trimming patch')
        assert xyz_box is not None or xy_polygon is not None
        if xyz_box is not None:
            log.debug(f'trim xyz_box: {xyz_box}')
        if xy_polygon is not None:
            log.debug(f'xy_polygon min xyz: {np.min(xy_polygon.coordinates, axis = 0)}')
            log.debug(f'xy_polygon max xyz: {np.max(xy_polygon.coordinates, axis = 0)}')
        large_t, large_p = larger_patch.triangles_and_points()
        log.debug(f'large surface min xyz: {np.min(large_p, axis = 0)}')
        log.debug(f'large surface max xyz: {np.max(large_p, axis = 0)}')
        # create bool per point indicating inclusion in box volume
        if xyz_box is None:
            points_in = np.ones(large_p.shape[:-1], dtype = bool)
        else:
            points_in = np.logical_and(np.all(large_p >= np.expand_dims(xyz_box[0], axis = 0), axis = -1),
                                       np.all(large_p <= np.expand_dims(xyz_box[1], axis = 0), axis = -1))
        # and check against xy polygon if present
        if xy_polygon is not None:
            points_in = np.logical_and(points_in, xy_polygon.points_are_inside_xy(large_p))
        # find where in large_t uses those points
        tp_in = points_in[large_t]
        t_in = np.all(tp_in, axis = -1) if internal else np.any(tp_in, axis = -1)
        # find unique points used by those triangles
        p_keep = np.unique(large_t[t_in])
        # note new point index for each old point that is being kept
        p_map = np.full(len(points_in), -1, dtype = int)
        p_map[p_keep] = np.arange(len(p_keep))
        # copy those unique points into a trimmed points array
        points_trimmed = large_p[p_keep]
        # copy selected triangles, replacing p indices with compressed indices
        t_trim = large_t[t_in]
        triangles_trimmed = p_map[t_trim]
        assert np.all(triangles_trimmed >= 0)
        assert np.all(triangles_trimmed < len(points_trimmed))
        self.crs_uuid = larger_patch.crs_uuid
        self.points = points_trimmed
        self.node_count = len(self.points)
        self.triangles = triangles_trimmed
        self.triangle_count = len(self.triangles)

    def set_to_horizontal_plane(self, depth, box_xyz, border = 0.0, quad_triangles = False):
        """Populate this (empty) patch with two triangles defining a flat, horizontal plane at a given depth.

        arguments:
           depth (float): z value to use in all points in the triangulated patch
           box_xyz (float[2, 3]): the min, max values of x, y (&z) giving the area to be covered (z ignored)
           border (float): an optional border width added around the x,y area defined by box_xyz
           quad_triangles (bool, default False): if True, 4 triangles are used instead of 2
        """

        # expand area by border
        box = box_xyz.copy()
        box[0, :2] -= border
        box[1, :2] += border
        self.node_count = 5 if quad_triangles else 4
        self.points = np.empty((self.node_count, 3))
        # set 4 points from corners of box
        self.points[0, :] = box[0, :]
        self.points[1, :] = box[1, :]
        self.points[2, 0], self.points[2, 1] = box[0, 0], box[1, 1]  # min x, max y
        self.points[3, 0], self.points[3, 1] = box[1, 0], box[0, 1]  # max x, min y
        if quad_triangles:
            self.points[4] = np.mean(box, axis = 0)
        # set depth for all points
        self.points[:, 2] = depth
        # create pair of triangles
        if quad_triangles:
            self.triangle_count = 4
            self.triangles = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4]], dtype = int)
        else:
            self.triangle_count = 2
            self.triangles = np.array([[0, 1, 2], [0, 3, 1]], dtype = int)

    def set_to_triangle(self, corners):
        """Populate this (empty) patch with a single triangle."""

        assert corners.shape == (3, 3)
        self.node_count = 3
        self.points = corners.copy()
        self.triangle_count = 1
        self.triangles = np.array([[0, 1, 2]], dtype = int)

    def set_to_triangle_pair(self, corners):
        """Populate this (empty) patch with a pair of triangles."""

        self.set_from_triangles_and_points(np.array([[0, 1, 3], [0, 3, 2]], dtype = int), corners)

    def set_from_triangles_and_points(self, triangles, points):
        """Populate this (empty) patch from triangle node indices and points from elsewhere."""

        assert triangles.ndim == 2 and triangles.shape[-1] == 3
        assert points.ndim == 2 and points.shape[1] in [2, 3]
        if points.shape[1] == 2:
            p = np.zeros((points.shape[0], 3))
            p[:, :2] = points
            points = p
        self.node_count = points.shape[0]
        self.points = points.copy()
        self.triangle_count = triangles.shape[0]
        self.triangles = triangles.copy()

    def set_to_sail(self, n, centre, radius, azimuth, delta_theta):
        """Populate this (empty) patch with triangles for a big triangle wrapped on a sphere."""

        def sail_point(centre, radius, a, d):
            # m = vec.rotation_3d_matrix((d, 0.0, 90.0 - a))
            # m = vec.tilt_3d_matrix(a, d)
            v = np.array((0.0, radius, 0.0))
            m = vec.rotation_matrix_3d_axial(0, d)
            v = vec.rotate_vector(m, v)
            m = vec.rotation_matrix_3d_axial(2, 90.0 + a)
            v = vec.rotate_vector(m, v)
            return centre + v

        assert n >= 1
        self.node_count = (n + 1) * (n + 2) // 2
        self.points = np.empty((self.node_count, 3))
        self.triangle_count = n * n
        self.triangles = np.empty((self.triangle_count, 3), dtype = int)
        self.points[0] = sail_point(centre, radius, azimuth, 0.0).copy()
        p = 0
        t = 0
        for row in range(n):
            azimuth -= 0.5 * delta_theta
            dip = (row + 1) * delta_theta
            az = azimuth
            for pp in range(row + 2):
                p += 1
                self.points[p] = sail_point(centre, radius, az, dip).copy()
                az += delta_theta

            #           log.debug('p: ' + str(p) + '; dip: {0:3.1f}; az: {1:3.1f}'.format(dip, az))
            p1 = row * (row + 1) // 2
            p2 = (row + 1) * (row + 2) // 2
            self.triangles[t] = (p1, p2, p2 + 1)
            t += 1
            for tri in range(row):
                self.triangles[t] = (p1, p1 + 1, p2 + 1)
                t += 1
                p1 += 1
                p2 += 1
                self.triangles[t] = (p1, p2, p2 + 1)
                t += 1
        log.debug('sail point zero: ' + str(self.points[0]))
        log.debug('sail xyz box: {0:4.2f}:{1:4.2f}  {2:4.2f}:{3:4.2f}  {4:4.2f}:{5:4.2f}'.format(
            np.min(self.points[:, 0]), np.max(self.points[:, 0]), np.min(self.points[:, 1]), np.max(self.points[:, 1]),
            np.min(self.points[:, 2]), np.max(self.points[:, 2])))

    def set_from_irregular_mesh(self, mesh_xyz, quad_triangles = False):
        """Populate this (empty) patch from an untorn mesh array of shape (N, M, 3)."""

        mesh_shape = mesh_xyz.shape
        assert len(mesh_shape) == 3 and mesh_shape[0] > 1 and mesh_shape[1] > 1 and mesh_shape[2] == 3
        ni = mesh_shape[1]
        if quad_triangles:
            # generate centre points:
            quad_centres = np.empty(((mesh_shape[0] - 1) * (mesh_shape[1] - 1), 3))
            quad_centres[:, :] = 0.25 * (mesh_xyz[:-1, :-1, :] + mesh_xyz[:-1, 1:, :] + mesh_xyz[1:, :-1, :] +
                                         mesh_xyz[1:, 1:, :]).reshape((-1, 3))
            self.points = np.concatenate((mesh_xyz.copy().reshape((-1, 3)), quad_centres))
            mesh_size = mesh_xyz.size // 3
            self.node_count = self.points.size // 3
            self.triangle_count = 4 * (mesh_shape[0] - 1) * (mesh_shape[1] - 1)
            self.quad_triangles = True
            triangles = np.empty((mesh_shape[0] - 1, mesh_shape[1] - 1, 4, 3), dtype = int)  # flatten later
            nic = ni - 1
            for j in range(mesh_shape[0] - 1):
                for i in range(nic):
                    triangles[j, i, :, 0] = mesh_size + j * nic + i  # quad centre
                    triangles[j, i, 0, 1] = j * ni + i
                    triangles[j, i, 0, 2] = triangles[j, i, 1, 1] = j * ni + i + 1
                    triangles[j, i, 1, 2] = triangles[j, i, 2, 1] = (j + 1) * ni + i + 1
                    triangles[j, i, 2, 2] = triangles[j, i, 3, 1] = (j + 1) * ni + i
                    triangles[j, i, 3, 2] = j * ni + i
        else:
            self.points = mesh_xyz.copy().reshape((-1, 3))
            self.node_count = mesh_shape[0] * mesh_shape[1]
            self.triangle_count = 2 * (mesh_shape[0] - 1) * (mesh_shape[1] - 1)
            self.quad_triangles = False
            triangles = np.empty((mesh_shape[0] - 1, mesh_shape[1] - 1, 2, 3), dtype = int)  # flatten later
            for j in range(mesh_shape[0] - 1):
                for i in range(mesh_shape[1] - 1):
                    triangles[j, i, 0, 0] = j * ni + i
                    triangles[j, i, :, 1] = j * ni + i + 1
                    triangles[j, i, :, 2] = (j + 1) * ni + i
                    triangles[j, i, 1, 0] = (j + 1) * ni + i + 1
        self.ni = ni - 1
        self.triangles = triangles.reshape((-1, 3))

    def set_from_sparse_mesh(self, mesh_xyz):
        """Populate this (empty) patch from a mesh array of shape (N, M, 3), with some NaNs in z."""

        # todo: add quad_triangles argument to apply to 'squares' with no NaNs
        mesh_shape = mesh_xyz.shape
        assert len(mesh_shape) == 3 and mesh_shape[0] > 1 and mesh_shape[1] > 1 and mesh_shape[2] == 3
        self.quad_triangles = False

        indices = self.get_indices_from_sparse_meshxyz(mesh_xyz)

        triangles = np.zeros((2 * (mesh_shape[0] - 1) * (mesh_shape[1] - 1), 3), dtype = int)  # truncate later
        nt = 0
        for j in range(mesh_shape[0] - 1):
            for i in range(mesh_shape[1] - 1):
                nan_nodes = np.count_nonzero(np.isnan(mesh_xyz[j:j + 2, i:i + 2, 2]))
                if nan_nodes > 1:
                    continue
                if nan_nodes == 0:
                    triangles[nt, 0] = indices[j, i]
                    triangles[nt:nt + 2, 1] = indices[j, i + 1]
                    triangles[nt:nt + 2, 2] = indices[j + 1, i]
                    triangles[nt + 1, 0] = indices[j + 1, i + 1]
                    nt += 2
                elif indices[j, i] < 0:
                    triangles[nt, 0] = indices[j, i + 1]
                    triangles[nt, 1] = indices[j + 1, i]
                    triangles[nt, 2] = indices[j + 1, i + 1]
                    nt += 1
                elif indices[j, i + 1] < 0:
                    triangles[nt, 0] = indices[j, i]
                    triangles[nt, 1] = indices[j + 1, i]
                    triangles[nt, 2] = indices[j + 1, i + 1]
                    nt += 1
                elif indices[j + 1, i] < 0:
                    triangles[nt, 0] = indices[j, i + 1]
                    triangles[nt, 1] = indices[j, i]
                    triangles[nt, 2] = indices[j + 1, i + 1]
                    nt += 1
                elif indices[j + 1, i + 1] < 0:
                    triangles[nt, 0] = indices[j, i + 1]
                    triangles[nt, 1] = indices[j + 1, i]
                    triangles[nt, 2] = indices[j, i]
                    nt += 1
                else:
                    raise Exception('code failure in sparse mesh processing')
        self.ni = None
        self.triangles = triangles[:nt, :]
        self.triangle_count = nt

    def get_indices_from_sparse_meshxyz(self, mesh_xyz):
        """Update self.points and self.node_count with non-nan points in a given mesh_xyz array.

        Returns the indices of these non_nan points.
        """
        points = np.zeros(mesh_xyz.shape).reshape((-1, 3))
        indices = np.zeros(mesh_xyz.shape[:2], dtype = int) - 1

        non_nans = np.where(~np.isnan(mesh_xyz[:, :, 2]))
        for i in range(len(non_nans[0])):
            points[i] = mesh_xyz[non_nans[0][i], non_nans[1][i]]
            indices[non_nans[0][i], non_nans[1][i]] = i
        self.points = points[:len(non_nans[0]), :]
        self.node_count = len(non_nans[0])

        return indices

    def set_from_torn_mesh(self, mesh_xyz, quad_triangles = False):
        """Populate this (empty) patch from a torn mesh array of shape (nj, ni, 2, 2, 3)."""

        mesh_shape = mesh_xyz.shape
        assert len(mesh_shape) == 5 and mesh_shape[0] > 0 and mesh_shape[1] > 0 and mesh_shape[2:] == (2, 2, 3)
        nj = mesh_shape[0]
        ni = mesh_shape[1]
        if quad_triangles:
            # generate centre points:
            quad_centres = np.empty((nj, ni, 3))
            quad_centres[:, :, :] = 0.25 * np.sum(mesh_xyz, axis = (2, 3))
            self.points = np.concatenate((mesh_xyz.copy().reshape((-1, 3)), quad_centres.reshape((-1, 3))))
            mesh_size = mesh_xyz.size // 3
            self.node_count = 5 * nj * ni
            self.triangle_count = 4 * nj * ni
            self.quad_triangles = True
            triangles = np.empty((nj, ni, 4, 3), dtype = int)  # flatten later
            for j in range(nj):
                for i in range(ni):
                    base_p = 4 * (j * ni + i)
                    triangles[j, i, :, 0] = mesh_size + j * ni + i  # quad centre
                    triangles[j, i, 0, 1] = base_p
                    triangles[j, i, 0, 2] = triangles[j, i, 1, 1] = base_p + 1
                    triangles[j, i, 1, 2] = triangles[j, i, 2, 1] = base_p + 3
                    triangles[j, i, 2, 2] = triangles[j, i, 3, 1] = base_p + 2
                    triangles[j, i, 3, 2] = base_p
        else:
            self.points = mesh_xyz.copy().reshape((-1, 3))
            self.node_count = 4 * nj * ni
            self.triangle_count = 2 * nj * ni
            self.quad_triangles = False
            triangles = np.empty((nj, ni, 2, 3), dtype = int)  # flatten later
            for j in range(nj):
                for i in range(ni):
                    base_p = 4 * (j * ni + i)
                    triangles[j, i, 0, 0] = base_p
                    triangles[j, i, :, 1] = base_p + 1
                    triangles[j, i, :, 2] = base_p + 2
                    triangles[j, i, 1, 0] = base_p + 3
        self.ni = ni
        self.triangles = triangles.reshape((-1, 3))

    def column_from_triangle_index(self, triangle_index):
        """For patch freshly built from fully defined mesh, returns (j, i) for given triangle index.

        argument:
           triangle_index (int or numpy int array): the triangle index (or array of indices) for which column(s) are being
           sought

        returns:
           pair of ints or pair of numpy int arrays: the (j0, i0) indices of the column(s) which the triangle(s) is/are
           part of

        notes:
           this function will only work if the surface has been freshly constructed with data from a mesh without NaNs,
           otherwise (None, None) will be returned;
           if triangle_index is a numpy int array, a pair of similarly shaped numpy arrays is returned
        """

        if self.quad_triangles is None or self.ni is None:
            return (None, None)
        if isinstance(triangle_index, int):
            if triangle_index >= self.triangle_count:
                return (None, None)
        else:
            if np.any(triangle_index >= self.triangle_count):
                return (None, None)
        if self.quad_triangles:
            face = triangle_index // 4
        else:
            face = triangle_index // 2
        if isinstance(face, int):
            return divmod(face, self.ni)
        return np.divmod(face, self.ni)

    def set_to_cell_faces_from_corner_points(self, cp, quad_triangles = True):
        """Populates this (empty) patch to represent faces of a cell, from corner points of shape (2, 2, 2, 3)."""

        assert cp.shape == (2, 2, 2, 3)
        self.quad_triangles = quad_triangles
        if quad_triangles:
            triangles = self.get_triangles_for_cell_faces_quad_true(cp)
        else:
            triangles = self.get_triangles_for_cell_faces_quad_false(cp)
        self.triangles = triangles.reshape((-1, 3))

    def get_triangles_for_cell_faces_quad_false(self, cp):
        """Returns the triangles for corner points representing cell faces, where quad_triangles is False."""

        self.triangle_count = 12
        self.node_count = 8
        self.points = cp.copy().reshape((-1, 3))
        triangles = np.empty((3, 2, 2, 3), dtype = int)  # flatten later
        for axis in range(3):
            if axis == 0:
                ip1, ip2 = 2, 1
            elif axis == 1:
                ip1, ip2 = 4, 1
            else:
                ip1, ip2 = 4, 2
            for ip in range(2):
                ips = -2 * ip + 1  # +1 for -ve face; -1 for +ve face!
                base_p = 7 * ip
                triangles[axis, ip, :, 0] = base_p
                triangles[axis, ip, :, 1] = base_p + ips * (ip1 + ip2)
                triangles[axis, ip, 0, 2] = base_p + ips * (ip2)
                triangles[axis, ip, 1, 2] = base_p + ips * (ip1)

        return triangles

    def get_triangles_for_cell_faces_quad_true(self, cp):
        """Returns the triangles for corner points representing cell faces, where quad_triangles is True."""

        self.triangle_count = 24
        quad_centres = np.empty((3, 2, 3))
        quad_centres[0, 0, :] = 0.25 * np.sum(cp[0, :, :, :], axis = (0, 1))  # K-
        quad_centres[0, 1, :] = 0.25 * np.sum(cp[1, :, :, :], axis = (0, 1))  # K+
        quad_centres[1, 0, :] = 0.25 * np.sum(cp[:, 0, :, :], axis = (0, 1))  # J-
        quad_centres[1, 1, :] = 0.25 * np.sum(cp[:, 1, :, :], axis = (0, 1))  # J+
        quad_centres[2, 0, :] = 0.25 * np.sum(cp[:, :, 0, :], axis = (0, 1))  # I-
        quad_centres[2, 1, :] = 0.25 * np.sum(cp[:, :, 1, :], axis = (0, 1))  # I+
        self.node_count = 14
        self.points = np.concatenate((cp.copy().reshape((-1, 3)), quad_centres.reshape((-1, 3))))
        triangles = np.empty((3, 2, 4, 3), dtype = int)  # flatten later
        for axis in range(3):
            if axis == 0:
                ip1, ip2 = 2, 1
            elif axis == 1:
                ip1, ip2 = 4, 1
            else:
                ip1, ip2 = 4, 2
            for ip in range(2):
                centre_p = 8 + 2 * axis + ip
                ips = -2 * ip + 1  # +1 for -ve face; -1 for +ve face!
                base_p = 7 * ip
                triangles[axis, ip, :, 0] = centre_p  # quad centre
                triangles[axis, ip, 0, 1] = base_p
                triangles[axis, ip, 0, 2] = triangles[axis, ip, 1, 1] = base_p + ips * (ip2)
                triangles[axis, ip, 1, 2] = triangles[axis, ip, 2, 1] = base_p + ips * (ip1 + ip2)
                triangles[axis, ip, 2, 2] = triangles[axis, ip, 3, 1] = base_p + ips * (ip1)
                triangles[axis, ip, 3, 2] = base_p

        return triangles

    def face_from_triangle_index(self, triangle_index):
        """For patch freshly built for cell faces, returns (axis, polarity) for given triangle index."""

        if self.quad_triangles:
            assert self.node_count == 30 and self.triangle_count == 24
            assert 0 <= triangle_index < 24
            face = triangle_index // 4
        else:
            assert self.node_count == 24 and self.triangle_count == 12
            assert 0 <= triangle_index < 12
            face = triangle_index // 2
        axis, polarity = divmod(face, 2)
        return axis, polarity

    def vertical_rescale_points(self, ref_depth, scaling_factor):
        """Rescale points along vertical direction.
        
        Modifies the z values of points for this patch by stretching the distance
        from reference depth by scaling factor.
        """
        _, _ = self.triangles_and_points()  # ensure points are loaded
        z_values = self.points[:, 2].copy()
        self.points[:, 2] = ref_depth + scaling_factor * (z_values - ref_depth)
