"""surface.py: surface class based on resqml standard."""

version = '2nd July 2021'

# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging
import warnings

log = logging.getLogger(__name__)
log.debug('surface.py version ' + version)

import math as maths
import numpy as np
# import xml.etree.ElementTree as et
# from lxml import etree as et

from resqpy.olio.base import BaseResqpy
import resqpy.olio.xml_et as rqet
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.crs as rcrs
import resqpy.olio.triangulation as triangulate
from resqpy.olio.xml_namespaces import curly_namespace as ns
from resqpy.olio.zmap_reader import read_mesh

import resqpy.organize as rqo


class _BaseSurface(BaseResqpy):
   """Base class to implement shared methods for other classes in this module"""

   def create_interpretation_and_feature(self,
                                         kind = 'horizon',
                                         name = None,
                                         interp_title_suffix = None,
                                         is_normal = True):
      """Creates xml and objects for a represented interpretaion and interpreted feature, if not already present."""

      assert kind in ['horizon', 'fault', 'fracture', 'geobody boundary']
      assert name or self.title, 'title missing'
      if not name:
         name = self.title

      if self.represented_interpretation_root is not None:
         log.debug(f'represented interpretation already exisrs for surface {self.title}')
         return
      if kind in ['horizon', 'geobody boundary']:
         feature = rqo.GeneticBoundaryFeature(self.model, kind = kind, feature_name = name)
         feature.create_xml()
         if kind == 'horizon':
            interp = rqo.HorizonInterpretation(self.model, genetic_boundary_feature = feature, domain = 'depth')
         else:
            interp = rqo.GeobodyBoundaryInterpretation(self.model, genetic_boundary_feature = feature, domain = 'depth')
      elif kind in ['fault', 'fracture']:
         feature = rqo.TectonicBoundaryFeature(self.model, kind = kind, feature_name = name)
         feature.create_xml()
         interp = rqo.FaultInterpretation(self.model,
                                          is_normal = is_normal,
                                          tectonic_boundary_feature = feature,
                                          domain = 'depth')  # might need more arguments
      else:
         log.critical('code failure')
      interp_root = interp.create_xml(title_suffix = interp_title_suffix)
      self.set_represented_interpretation_root(interp_root)


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
      self.crs_root = None
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

   def extract_crs_root_and_uuid(self):
      """Caches root and uuid for coordinate reference system, as stored in geometry xml sub-tree."""

      if self.crs_root is not None and self.crs_uuid is not None:
         return self.crs_root, self.crs_uuid
      self.crs_root = rqet.find_nested_tags(self.node, ['Geometry', 'LocalCrs'])
      if self.crs_root is None:
         self.crs_uuid = None
      else:
         self.crs_uuid = bu.uuid_from_string(rqet.find_tag_text(self.crs_root, 'UUID'))
      return self.crs_root, self.crs_uuid

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

   def set_to_horizontal_plane(self, depth, box_xyz, border = 0.0):
      """Populate this (empty) patch with two triangles defining a flat, horizontal plane at a given depth.

         arguments:
            depth (float): z value to use in all points in the triangulated patch
            box_xyz (float[2, 3]): the min, max values of x, y (&z) giving the area to be covered (z ignored)
            border (float): an optional border width added around the x,y area defined by box_xyz
      """

      # expand area by border
      box = box_xyz.copy()
      box[0, :2] -= border
      box[1, :2] += border
      self.node_count = 4
      self.points = np.empty((4, 3))
      # set 2 points common to both triangles
      self.points[0, :] = box[0, :]
      self.points[1, :] = box[1, :]
      # and 2 others to form a rectangle aligned with x,y axes
      self.points[2, 0], self.points[2, 1] = box[0, 0], box[1, 1]  # min x, max y
      self.points[3, 0], self.points[3, 1] = box[1, 0], box[0, 1]  # max x, min y
      # set depth for all points
      self.points[:, 2] = depth
      # create pair of triangles
      self.triangle_count = 2
      self.triangles = np.array([[0, 1, 2], [0, 3, 1]], dtype = int)

   def set_to_triangle(self, corners):
      """Populate this (empty) patch with a single triangle."""

      assert corners.shape == (3, 3)
      self.node_count = 3
      self.points = corners.copy()
      self.triangle_count = 1
      self.triangles = np.array([[0, 1, 2]], dtype = int)

   def set_from_triangles_and_points(self, triangles, points):
      """Populate this (empty) patch from triangle node indices and points from elsewhere."""

      assert triangles.ndim == 2 and triangles.shape[-1] == 3
      assert points.ndim == 2 and points.shape[1] == 3

      self.node_count = points.shape[0]
      self.points = points.copy()
      self.triangle_count = triangles.shape[0]
      self.triangles = triangles.copy()

   def set_to_sail(self, n, centre, radius, azimuth, delta_theta):
      """Populate this (empty) patch with triangles for a big triangle wrapped on a sphere."""

      def sail_point(centre, radius, a, d):
         #        m = vec.rotation_3d_matrix((d, 0.0, 90.0 - a))
         #        m = vec.tilt_3d_matrix(a, d)
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
      points = np.zeros(mesh_xyz.shape).reshape((-1, 3))
      indices = np.zeros(mesh_xyz.shape[:2], dtype = int) - 1
      n_p = 0
      # this could probably be speeded up with some numpy where operation
      for j in range(mesh_shape[0]):
         for i in range(mesh_shape[1]):
            if not np.isnan(mesh_xyz[j, i, 2]):
               points[n_p] = mesh_xyz[j, i]
               indices[j, i] = n_p
               n_p += 1
      self.points = points[:n_p, :]
      self.node_count = n_p
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
      else:
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
      self.triangles = triangles.reshape((-1, 3))

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
      """Modify the z values of points for this patch by stretching the distance from reference depth by scaling factor."""

      _, _ = self.triangles_and_points()  # ensure points are loaded
      z_values = self.points[:, 2].copy()
      self.points[:, 2] = ref_depth + scaling_factor * (z_values - ref_depth)


class Surface(_BaseSurface):
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
      """Create an empty Surface object (RESQML TriangulatedSetRepresentation) and optionally populates from xml, point set or mesh.

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

   def set_from_triangles_and_points(self, triangles, points):
      """Populate this (empty) Surface object from an array of triangle corner indices and an array of points."""

      tri_patch = TriangulatedPatch(self.model, patch_index = 0, crs_uuid = self.crs_uuid)
      tri_patch.set_from_triangles_and_points(triangles, points)
      self.patch_list = [tri_patch]
      self.uuid = bu.new_uuid()

   def set_from_point_set(self, point_set):
      """Populate this (empty) Surface object with a Delaunay triangulation of points in a PointSet object."""

      p = point_set.full_array_ref()
      log.debug('number of points going into dt: ' + str(len(p)))
      t = triangulate.dt(p[:, :2])
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
      """Populates this (empty) surface to represent faces of a set of cells, from corner points of shape (N, 2, 2, 2, 3)."""

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
      """For surface freshly built for cell faces, returns (cell_number, face_axis, polarity) for given triangle index.

      argument:
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
      """Populate this (empty) surface with a patch of two triangles defining a flat, horizontal plane at a given depth.

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
      """Modify the z values of points for this surface by stretching the distance from reference depth by scaling factor."""

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


class CombinedSurface:
   """Class allowing a collection of Surface objects to be treated as a single surface (not a RESQML class in its own right)."""

   def __init__(self, surface_list, crs_uuid = None):
      """Initialise a CombinedSurface object from a list of Surface (and/or CombinedSurface) objects.

      arguments:
         surface_list (list of Surface and/or CombinedSurface objects): the new object is the combination of these surfaces
         crs_uuid (uuid.UUID, optional): if present, all contributing surfaces must refer to this crs

      note:
         all contributing surfaces should be established before initialising this object;
         all contributing surfaces must refer to the same crs; this class of object is not part of the RESQML
         standard and cannot be saved in a RESQML dataset - it is a high level derived object class
      """

      assert len(surface_list) > 0
      self.surface_list = surface_list
      self.crs_uuid = crs_uuid
      if self.crs_uuid is None:
         self.crs_uuid = surface_list[0].crs_uuid
      self.patch_count_list = []
      self.triangle_count_list = []
      self.points_count_list = []
      self.is_combined_list = []
      self.triangles = None
      self.points = None
      for surface in surface_list:
         is_combined = isinstance(surface, CombinedSurface)
         self.is_combined_list.append(is_combined)
         if is_combined:
            self.patch_count_list.append(sum(surface.patch_count_list))
         else:
            self.patch_count_list.append(len(surface.patch_list))
         t, p = surface.triangles_and_points()
         self.triangle_count_list.append(len(t))
         self.points_count_list.append(len(p))

   def surface_index_for_triangle_index(self, tri_index):
      """For a triangle index in the combined surface, returns the index of the surface containing rhe triangle and local triangle index."""

      for s_i in range(len(self.surface_list)):
         if tri_index < self.triangle_count_list[s_i]:
            return s_i, tri_index
         tri_index -= self.triangle_count_list[s_i]
      return None

   def triangles_and_points(self):
      """Returns the composite triangles and points for the combined surface."""

      if self.triangles is None:
         points_offset = 0
         for surface in self.surface_list:
            (t, p) = surface.triangles_and_points()
            if points_offset == 0:
               self.triangles = t.copy()
               self.points = p.copy()
            else:
               self.triangles = np.concatenate((self.triangles, t.copy() + points_offset))
               self.points = np.concatenate((self.points, p.copy()))
            points_offset += p.shape[0]

      return self.triangles, self.points


class PointSet(_BaseSurface):
   """Class for RESQML Point Set Representation within resqpy model object."""  # TODO: Work in Progress

   resqml_type = 'PointSetRepresentation'

   def __init__(self,
                parent_model,
                point_set_root = None,
                uuid = None,
                load_hdf5 = False,
                points_array = None,
                crs_uuid = None,
                polyset = None,
                polyline = None,
                random_point_count = None,
                charisma_file = None,
                irap_file = None,
                title = None,
                originator = None,
                extra_metadata = None):
      """Creates an empty Point Set object and optionally populates from xml or other source.

      arguments:
         parent_model (model.Model object): the model to which the new point set belongs
         point_set_root (xml node, optional): DEPRECATED, use uuid instead;
            if present, the new point set is created based on the xml
         uuid (uuid.UUID, optional): if present, the object is populated from the RESQML PointSetRepresentation
            with this uuid
         load_hdf5 (boolean, default False): if True and point_set_root is present, the actual points are
            pre-loaded into a numpy array; otherwise the points will be loaded on demand
         points_array (numpy float array of shape (..., 3), optional): if present, the xyz data which
            will constitute the point set; ignored if point_set_root is not None
         crs_uuid (uuid.UUID, optional): if present, identifies the coordinate reference system for the points;
            ignored if point_set_root is not None; if None, 'imported' points will be associated with the
            default crs of the parent model
         polyset (optional): if present, creates a pointset from points in a polylineset
         polyline (optional): if present and random_point_count is None or zero, creates a pointset from
            points in a polyline; if present and random_point_count is set, creates random points within
            the (closed, convex) polyline
         random_point_count (int, optional): if present and polyline is present then the number of random
            points to generate within the (closed) polyline in the xy plane, with z set to 0.0
         charisma_file (optional): if present, creates a pointset from a charisma 3d interpretation file
         irap_file (optional): if present, creates a pointset from an IRAP classic points format file
         title (str, optional): the citation title to use for a new point set;
            ignored if uuid or point_set_root is not None
         originator (str, optional): the name of the person creating the point set, defaults to login id;
            ignored if uuid or point_set_root is not None
         extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the point set;
            ignored if uuid or point_set_root is not None

      returns:
         newly created PointSet object

      :meta common:
      """

      self.crs_uuid = crs_uuid
      self.patch_count = None
      self.patch_ref_list = []  # ordered list of (patch hdf5 ext uuid, path in hdf5, point count)
      self.patch_array_list = []  # ordered list of numpy float arrays (or None before loading), each of shape (N, 3)
      self.full_array = None
      self.points = None  # composite points (all patches)
      self.represented_interpretation_root = None
      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       originator = originator,
                       root_node = point_set_root)

      if self.root is not None:
         if load_hdf5:
            self.load_all_patches()

      elif points_array is not None:
         assert self.crs_uuid is not None, 'missing crs uuid when establishing point set from array'
         self.add_patch(points_array)

      elif polyline is not None:  # Points from or within polyline
         if random_point_count:
            assert polyline.is_convex()
            points = np.zeros((random_point_count, 3))
            points[:, :2] = np.random.random((random_point_count, 2))
            for p_i in range(random_point_count):
               points[p_i, :2] = polyline.denormalised_xy(points[p_i, 0], points[p_i, 1])
            self.add_patch(points)
         else:
            self.add_patch(polyline.coordinates)
            if polyline.rep_int_root is not None:
               self.set_represented_interpretation_root(polyline.rep_int_root)
         if self.crs_uuid is None:
            self.crs_uuid = polyline.crs_uuid
         else:
            assert bu.matching_uuids(self.crs_uuid, polyline.crs_uuid), 'mismatched crs uuids'
         if not self.title:
            self.title = polyline.title

      elif polyset is not None:  # Points from polylineSet
         for poly in polyset.polys:
            if poly == polyset.polys[0]:
               master_crs = rcrs.Crs(self.model, uuid = poly.crs_uuid)
               self.crs_root = poly.crs_root
               if poly.isclosed and vec.isclose(poly.coordinates[0], poly.coordinates[-1]):
                  poly_coords = poly.coordinates[:-1].copy()
               else:
                  poly_coords = poly.coordinates.copy()
            else:
               curr_crs = rcrs.Crs(self.model, uuid = poly.crs_uuid)
               if not curr_crs.is_equivalent(master_crs):
                  shifted = curr_crs.convert_array_to(master_crs, poly.coordinates)
                  if poly.isclosed and vec.isclose(shifted[0], shifted[-1]):
                     poly_coords = np.concatenate((poly_coords, shifted[:-1]))
                  else:
                     poly_coords = np.concatenate((poly_coords, shifted))
               else:
                  if poly.isclosed and vec.isclose(poly.coordinates[0], poly.coordinates[-1]):
                     poly_coords = np.concatenate((poly_coords, poly.coordinates[:-1]))
                  else:
                     poly_coords = np.concatenate((poly_coords, poly.coordinates))
         self.add_patch(poly_coords)
         if polyset.rep_int_root is not None:
            self.set_represented_interpretation_root(polyset.rep_int_root)
         if self.crs_uuid is None:
            self.crs_uuid = polyset.polys[0].crs_uuid
         else:
            assert bu.matching_uuids(self.crs_uuid, polyset.polys[0].crs_uuid), 'mismatched crs uuids'
         if not self.title:
            self.title = polyset.title

      elif charisma_file is not None:  # Points from Charisma 3D interpretation lines
         with open(charisma_file, 'r') as surf:
            for i, line in enumerate(surf.readlines()):
               if i == 0:
                  cpoints = np.array([[float(x) for x in line.split()[6:]]])
               else:
                  curr = np.array([[float(x) for x in line.split()[6:]]])
                  cpoints = np.concatenate((cpoints, curr))
         self.add_patch(cpoints)
         assert self.crs_uuid is not None, 'crs uuid missing when establishing point set from charisma file'
         if not self.title:
            self.title = charisma_file

      elif irap_file is not None:  # Points from IRAP simple points
         with open(irap_file, 'r') as points:
            for i, line in enumerate(points.readlines()):
               if i == 0:
                  cpoints = np.array([[float(x) for x in line.split(" ")]])
               else:
                  curr = np.array([[float(x) for x in line.split(" ")]])
                  cpoints = np.concatenate((cpoints, curr))
         self.add_patch(cpoints)
         assert self.crs_uuid is not None, 'crs uuid missing when establishing point set from irap file'
         if not self.title:
            self.title = irap_file

      if not self.title:
         self.title = 'point set'

   def _load_from_xml(self):
      root_node = self.root
      assert root_node is not None
      self.patch_count = rqet.count_tag(root_node, 'NodePatch')
      assert self.patch_count, 'no patches found in xml for point set'
      self.patch_array_list = [None for _ in range(self.patch_count)]
      patch_index = 0
      for child in rqet.list_of_tag(root_node, 'NodePatch'):
         point_count = rqet.find_tag_int(child, 'Count')
         geom_node = rqet.find_tag(child, 'Geometry')
         assert geom_node, 'geometry missing in xml for point set patch'
         crs_uuid = rqet.find_nested_tags_text(geom_node, ['LocalCrs', 'UUID'])
         assert crs_uuid, 'crs uuid missing in geometry xml for point set patch'
         if self.crs_uuid is None:
            self.crs_uuid = crs_uuid
         else:
            assert bu.matching_uuids(crs_uuid, self.crs_uuid), 'mixed coordinate reference systems in point set'
         ext_uuid = rqet.find_nested_tags_text(geom_node, ['Points', 'Coordinates', 'HdfProxy', 'UUID'])
         assert ext_uuid, 'missing hdf5 uuid in goemetry xml for point set patch'
         hdf5_path = rqet.find_nested_tags_text(geom_node, ['Points', 'Coordinates', 'PathInHdfFile'])
         assert hdf5_path, 'missing internal hdf5 path in goemetry xml for point set patch'
         self.patch_ref_list.append((ext_uuid, hdf5_path, point_count))
         patch_index += 1
      ref_node = rqet.find_tag(self.root, 'RepresentedInterpretation')
      if ref_node is not None:
         interp_root = self.model.referenced_node(ref_node)
         self.set_represented_interpretation_root(interp_root)
      # note: load of patches handled elsewhere

   def set_represented_interpretation_root(self, interp_root):
      """Makes a note of the xml root of the represented interpretation."""

      self.represented_interpretation_root = interp_root

   def single_patch_array_ref(self, patch_index):
      """Load numpy array for one patch of the point set from hdf5, cache and return it."""

      assert 0 <= patch_index < self.patch_count, 'point set patch index out of range'
      if self.patch_array_list[patch_index] is not None:
         return self.patch_array_list[patch_index]
      h5_key_pair = (self.patch_ref_list[patch_index][0], self.patch_ref_list[patch_index][1])  # ext uuid, path in hdf5
      try:
         self.model.h5_array_element(h5_key_pair,
                                     cache_array = True,
                                     object = self,
                                     array_attribute = 'temp_points',
                                     dtype = 'float')
      except Exception:
         log.exception('hdf5 points failure for point set patch ' + str(patch_index))
      assert self.temp_points.ndim == 2 and self.temp_points.shape[1] == 3, 'unsupported dimensionality to points array'
      self.patch_array_list[patch_index] = self.temp_points.copy()
      delattr(self, 'temp_points')
      return self.patch_array_list[patch_index]

   def load_all_patches(self):
      """Load hdf5 data for all patches and cache as separate numpy arrays; not usually called directly."""

      for patch_index in range(self.patch_count):
         self.single_patch_array_ref(patch_index)

   def full_array_ref(self):
      """Return a single numpy float array of shape (N, 3) containing all points from all patches.

      :meta common:
      """

      if self.full_array is not None:
         return self.full_array
      self.load_all_patches()
      if self.patch_count == 1:  # optimisation, as usually the case
         self.full_array = self.patch_array_list[0]
         return self.full_array
      point_count = 0
      for patch_index in range(self.patch_count):
         point_count += self.patch_ref_list[patch_index][2]
      self.full_array = np.empty((point_count, 3))
      full_index = 0
      for patch_index in range(self.patch_count):
         self.full_array[full_index:full_index +
                         self.patch_ref_list[patch_index][2]] = self.patch_array_list[patch_index]
         full_index += self.patch_ref_list[patch_index][2]
      assert full_index == point_count, 'point count mismatch when constructing full array for point set'
      return self.full_array

   def add_patch(self, points_array):
      """Extend the current point set with a new patch of points."""

      assert points_array.ndim >= 2 and points_array.shape[-1] == 3
      self.patch_array_list.append(points_array.reshape(-1, 3).copy())
      self.patch_ref_list.append((None, None, points_array.shape[0]))
      self.full_array = None
      if self.patch_count is None:
         self.patch_count = 0
      self.patch_count += 1

   def write_hdf5(self, file_name = None, mode = 'a'):
      """Create or append to an hdf5 file, writing datasets for the point set patches after caching arrays.

      :meta common:
      """

      if not file_name:
         file_name = self.model.h5_file_name()
      if self.uuid is None:
         self.uuid = bu.new_uuid()
      # NB: patch arrays must all have been set up prior to calling this function
      h5_reg = rwh5.H5Register(self.model)
      for patch_index in range(self.patch_count):
         h5_reg.register_dataset(self.uuid, 'points_{}'.format(patch_index), self.patch_array_list[patch_index])
      h5_reg.write(file_name, mode = mode)

   def create_xml(self,
                  ext_uuid = None,
                  crs_root = None,
                  add_as_part = True,
                  add_relationships = True,
                  root = None,
                  title = None,
                  originator = None):
      """Creates a point set representation xml node from this point set object and optionally adds as part of model.

         arguments:
            ext_uuid (uuid.UUID): the uuid of the hdf5 external part holding the points array(s)
            add_as_part (boolean, default True): if True, the newly created xml node is added as a part
               in the model
            add_relationships (boolean, default True): if True, a relationship xml part is created relating the
               new point set part to the crs part (and optional interpretation part)
            root (optional, usually None): if not None, the newly created point set representation node is appended
               as a child to this node
            title (string): used as the citation Title text; should be meaningful to a human
            originator (string, optional): the name of the human being who created the point set representation part;
               default is to use the login name

         returns:
            the newly created point set representation xml node

      :meta common:
      """

      if ext_uuid is None:
         ext_uuid = self.model.h5_uuid()

      ps_node = super().create_xml(add_as_part = False, title = title, originator = originator)

      if crs_root is None:
         if self.crs_uuid is None:
            crs_root = self.model.crs_root  # maverick use of model's default crs
            self.crs_uuid = rqet.uuid_for_part_root(crs_root)
         else:
            crs_root = self.model.root_for_part(self.model.part_for_uuid(self.crs_uuid))
      else:
         self.crs_uuid = rqet.uuid_for_part_root(crs_root)

      if self.represented_interpretation_root is not None:
         interp_root = self.represented_interpretation_root
         interp_uuid = bu.uuid_from_string(interp_root.attrib['uuid'])
         interp_part = self.model.part_for_uuid(interp_uuid)
         self.model.create_ref_node('RepresentedInterpretation',
                                    rqet.find_nested_tags_text(interp_root, ['Citation', 'Title']),
                                    interp_uuid,
                                    content_type = self.model.type_of_part(interp_part),
                                    root = ps_node)

      for patch_index in range(self.patch_count):

         p_node = rqet.SubElement(ps_node, ns['resqml2'] + 'NodePatch')
         p_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'NodePatch')
         p_node.text = '\n'

         pi_node = rqet.SubElement(p_node, ns['resqml2'] + 'PatchIndex')
         pi_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
         pi_node.text = str(patch_index)

         ct_node = rqet.SubElement(p_node, ns['resqml2'] + 'Count')
         ct_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
         ct_node.text = str(self.patch_ref_list[patch_index][2])

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

         self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'points_{}'.format(patch_index), root = coords)

      if root is not None:
         root.append(ps_node)
      if add_as_part:
         self.model.add_part('obj_PointSetRepresentation', self.uuid, ps_node)
         if add_relationships:
            # todo: add multiple crs'es (one per patch)?
            self.model.create_reciprocal_relationship(ps_node, 'destinationObject', crs_root, 'sourceObject')
            if self.represented_interpretation_root is not None:
               self.model.create_reciprocal_relationship(ps_node, 'destinationObject',
                                                         self.represented_interpretation_root, 'sourceObject')
            ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False)
            ext_node = self.model.root_for_part(ext_part)
            self.model.create_reciprocal_relationship(ps_node, 'mlToExternalPartProxy', ext_node,
                                                      'externalPartProxyToMl')

      return ps_node

   def convert_to_charisma(self, file_name):
      """Output to Charisma 3D interepretation format from a pointset

      If file_name exists, it will be overwritten.

      args:
          file_name: output file name to save to
      """
      #      hznname = self.title.replace(" ","_")
      lines = []
      self.load_all_patches()
      for patch in self.patch_array_list:
         for points in patch:
            lines.append(f"INLINE :\t1 XLINE :\t1\t{points[0]}\t{points[1]}\t{points[2]}\n")
      with open(file_name, 'w') as f:
         for item in lines:
            f.write(item)

   def convert_to_irap(self, file_name):
      """Output to IRAP simple points format from a pointset

      If file_name exists, it will be overwritten.

      args:
          file_name: output file name to save to
      """
      #      hznname = self.title.replace(" ","_")
      lines = []
      self.load_all_patches()
      for patch in self.patch_array_list:
         for points in patch:
            lines.append(f"{points[0]} {points[1]} {points[2]}\n")
      with open(file_name, 'w') as f:
         for item in lines:
            f.write(item)


class Mesh(_BaseSurface):
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

      elif xyz_values is not None and crs_uuid is not None:
         # create an explicit mesh directly from a numpy array of points
         assert xyz_values.ndim == 3 and xyz_values.shape[2] == 3 and xyz_values.shape[0] > 1 and xyz_values.shape[1] > 1
         self.flavour = 'explicit'
         self.nj = xyz_values.shape[0]
         self.ni = xyz_values.shape[1]
         self.full_array = xyz_values.copy()
         assert self.crs_uuid is not None, 'crs uuid missing'

      elif (nj is not None and ni is not None and origin is not None and dxyz_dij is not None and
            crs_uuid is not None and z_values is None):
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

      elif nj is not None and ni is not None and z_values is not None and z_supporting_mesh_uuid is not None:
         # create a ref&z mesh from arguments
         assert nj > 0 and ni > 0
         assert z_values.shape == (nj, ni) or z_values.shape == (nj * ni,)
         self.flavour = 'ref&z'
         self.nj = nj
         self.ni = ni
         self.ref_uuid = z_supporting_mesh_uuid
         self.ref_mesh = Mesh(self.model, root_node = self.model.root_for_uuid(z_supporting_mesh_uuid))
         assert self.ref_mesh is not None
         assert self.ref_mesh.nj == nj and self.ref_mesh.ni == ni
         self.full_array = self.ref_mesh.full_array_ref().copy()
         self.full_array[..., 2] = z_values.reshape(tuple(self.full_array.shape[:-1]))
         assert self.crs_uuid is not None, 'crs uuid missing'

      elif (nj is not None and ni is not None and z_values is not None and dxyz_dij is not None and
            crs_uuid is not None and origin is not None):
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

      assert self.crs_uuid is not None
      if not self.title:
         self.title = 'mesh'


#     log.debug(f'new mesh has flavour {self.flavour}')

   def _load_from_xml(self):
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

      if flavour == 'Point3dLatticeArray':
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
            assert count == (self.nj, self.ni)[j_or_i] - 1,  \
                   'unexpected value for count in xml spacing info for regular mesh (lattice)'
            assert stride > 0.0, 'spacing distance is not positive in xml for regular mesh (lattice)'
            self.regular_dxyz_dij[1 - j_or_i] *= stride

      elif flavour == 'Point3dZValueArray':
         # note: only simple, full use of supporting mesh is handled at present
         z_ref_node = rqet.find_tag(point_node, 'ZValues')
         self.ref_z_h5_key_pair = self.model.h5_uuid_and_path_for_node(z_ref_node, tag = 'Values')
         support_geom_node = rqet.find_tag(point_node, 'SupportingGeometry')
         if rqet.node_type(support_geom_node) == 'Point3dFromRepresentationLatticeArray':
            self.flavour = 'ref&z'
            # assert rqet.node_type(support_geom_node) == 'Point3dFromRepresentationLatticeArray'  # only this supported for now
            self.ref_uuid = rqet.find_nested_tags_text(support_geom_node, ['SupportingRepresentation', 'UUID'])
            assert self.ref_uuid, 'missing supporting representation info in xml for z-value mesh'
            self.ref_mesh = Mesh(self.model, root_node = self.model.root_for_uuid(self.ref_uuid))
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
               assert count == (self.nj, self.ni)[j_or_i] - 1,  \
                       'unexpected value for count in xml spacing info for regular mesh (lattice)'
         elif rqet.node_type(support_geom_node) == 'Point3dLatticeArray':
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
               if not maths.isclose(
                     vec.dot_product(self.regular_dxyz_dij[1 - j_or_i], self.regular_dxyz_dij[1 - j_or_i]), 1.0):
                  log.warning('non-orthogonal axes and/or scaling in xml for regular mesh (lattice)')
               spacing_node = rqet.find_tag(offset_nodes[j_or_i], 'Spacing')
               stride = rqet.find_tag_float(spacing_node, 'Value')
               count = rqet.find_tag_int(spacing_node, 'Count')
               assert stride is not None and count is not None, 'missing spacing info in xml'
               assert count == (self.nj, self.ni)[j_or_i] - 1,  \
                      'unexpected value for count in xml spacing info for regular mesh (lattice)'
               assert stride > 0.0, 'spacing distance is not positive in xml for regular mesh (lattice)'
               self.regular_dxyz_dij[1 - j_or_i] *= stride

      elif flavour in ['Point3dHdf5Array', 'Point2dHdf5Array']:
         self.flavour = 'explicit'
         self.explicit_h5_key_pair = self.model.h5_uuid_and_path_for_node(point_node, tag = 'Coordinates')
         # load full_array on demand later (see full_array_ref() method)

      else:
         raise Exception('unrecognised flavour for mesh points')

   def set_represented_interpretation_root(self, interp_root):
      """Makes a note of the xml root of the represented interpretation."""

      self.represented_interpretation_root = interp_root

   def full_array_ref(self):
      """Populates a full 2D(+1) numpy array of shape (nj, ni, 3) with xyz values, caches and returns.

      note:
         z values may be zero or not applicable when using the mesh as support for properties.
      """

      if self.full_array is not None:
         return self.full_array

      if self.flavour == 'explicit':
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

      elif self.flavour in ['regular', 'reg&z']:
         self.full_array = np.empty((self.nj, self.ni, 3))
         x_i0 = np.linspace(0.0, (self.nj - 1) * self.regular_dxyz_dij[1, 0], num = self.nj)
         y_i0 = np.linspace(0.0, (self.nj - 1) * self.regular_dxyz_dij[1, 1], num = self.nj)
         z_i0 = np.linspace(0.0, (self.nj - 1) * self.regular_dxyz_dij[1, 2], num = self.nj)
         x_full = np.linspace(x_i0, x_i0 + (self.ni - 1) * self.regular_dxyz_dij[0, 0], num = self.ni, axis = -1)
         y_full = np.linspace(y_i0, y_i0 + (self.ni - 1) * self.regular_dxyz_dij[0, 1], num = self.ni, axis = -1)
         z_full = np.linspace(z_i0, z_i0 + (self.ni - 1) * self.regular_dxyz_dij[0, 2], num = self.ni, axis = -1)
         self.full_array = np.stack((x_full, y_full, z_full), axis = -1) + self.regular_origin
         if self.flavour == 'reg&z':  # overwrite regular z values with explicitly stored z values
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

      elif self.flavour == 'ref&z':
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

      else:
         raise Exception('unrecognised mesh flavour when fetching full array')

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

      ref_root = None

      if self.flavour == 'regular':

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

      elif self.flavour == 'ref&z':

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
            oc_node.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
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

      elif self.flavour == 'reg&z':

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

      elif self.flavour == 'explicit':

         assert ext_uuid is not None

         if use_xy_only:
            p_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point2dHdf5Array')
         else:
            p_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3dHdf5Array')

         coords = rqet.SubElement(p_node, ns['resqml2'] + 'Coordinates')
         coords.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
         coords.text = '\n'

         self.model.create_hdf5_dataset_ref(ext_uuid, self.uuid, 'points', root = coords)

      else:
         log.error('mesh has bad flavour when creating xml')
         return None

      if root is not None:
         root.append(g2d_node)
      if add_as_part:
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

      return g2d_node
