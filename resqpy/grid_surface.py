"""grid_surface.py: Functions relating to intsection of resqml grid with surface or trajectory objects."""

version = '14th July 2021'

import logging

log = logging.getLogger(__name__)
log.debug('grid_surface.py version ' + version)

import numpy as np
import pandas as pd

import resqpy.olio.xml_et as rqet
import resqpy.olio.vector_utilities as vec
import resqpy.olio.intersection as meet
import resqpy.olio.box_utilities as bx
import resqpy.olio.uuid as bu

import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.fault as rqf
import resqpy.surface as rqs
import resqpy.well as rqw


class GridSkin:
   """Class of object consisting of outer skin of grid (not a RESQML class in its own right)."""

   def __init__(self, grid, quad_triangles = True, use_single_layer_tactics = True):
      """Returns a composite surface object consisting of outer skin of grid."""

      if grid.k_gaps:
         use_single_layer_tactics = False

      self.grid = grid
      self.k_gaps = 0 if grid.k_gaps is None else grid.k_gaps
      self.k_gap_after_layer_list = [
      ]  # list of layer numbers (zero based) where there is a gap after (ie. below, usually)
      self.has_split_coordinate_lines = grid.has_split_coordinate_lines
      self.quad_triangles = quad_triangles
      self.use_single_layer_tactics = use_single_layer_tactics
      self.skin = None  #: compostite surface constructed during initialisation
      self.fault_j_face_cols_ji0 = None  # split internal J faces
      self.fault_i_face_cols_ji0 = None  # split internal I faces
      self.polygon = None  # not yet in use

      k_gap_surf_list = self._make_k_gap_surfaces(quad_triangles = quad_triangles)

      if self.has_split_coordinate_lines:

         top_surf = generate_torn_surface_for_layer_interface(grid,
                                                              k0 = 0,
                                                              ref_k_faces = 'top',
                                                              quad_triangles = quad_triangles)
         base_surf = generate_torn_surface_for_layer_interface(grid,
                                                               k0 = grid.nk - 1,
                                                               ref_k_faces = 'base',
                                                               quad_triangles = quad_triangles)
         j_minus_surf = generate_torn_surface_for_x_section(grid,
                                                            'J',
                                                            ref_slice0 = 0,
                                                            plus_face = False,
                                                            quad_triangles = quad_triangles,
                                                            as_single_layer = use_single_layer_tactics)
         j_plus_surf = generate_torn_surface_for_x_section(grid,
                                                           'J',
                                                           ref_slice0 = self.grid.nj - 1,
                                                           plus_face = True,
                                                           quad_triangles = quad_triangles,
                                                           as_single_layer = use_single_layer_tactics)
         i_minus_surf = generate_torn_surface_for_x_section(grid,
                                                            'I',
                                                            ref_slice0 = 0,
                                                            plus_face = False,
                                                            quad_triangles = quad_triangles,
                                                            as_single_layer = use_single_layer_tactics)
         i_plus_surf = generate_torn_surface_for_x_section(grid,
                                                           'I',
                                                           ref_slice0 = self.grid.ni - 1,
                                                           plus_face = True,
                                                           quad_triangles = quad_triangles,
                                                           as_single_layer = use_single_layer_tactics)

         # fault face processing
         j_column_faces, i_column_faces = grid.split_column_faces()
         col_j0, col_i0 = np.where(j_column_faces)
         self.fault_j_face_cols_ji0 = np.stack((col_j0, col_i0), axis = -1)  # fault is on plus j face of these columns
         col_j0, col_i0 = np.where(i_column_faces)
         self.fault_i_face_cols_ji0 = np.stack((col_j0, col_i0), axis = -1)  # fault is on plus i face of these columns

         j_minus_fault_surf = j_plus_fault_surf = None
         j_minus_fault_surf_list = []
         j_plus_fault_surf_list = []
         for col_ji0 in self.fault_j_face_cols_ji0:
            #            log.debug(f'fault j face col_ji0: {col_ji0}')
            # note: use of single layer for internal fault surfaces results in some incorrect skin surface where k gaps are present
            # find_first_intersection_of_trajectory() method takes care of this but other uses of skin might need to be aware
            _, surf = create_column_face_mesh_and_surface(grid,
                                                          col_ji0,
                                                          1,
                                                          1,
                                                          quad_triangles = quad_triangles,
                                                          as_single_layer = True)
            j_plus_fault_surf_list.append(surf)
            _, surf = create_column_face_mesh_and_surface(grid, (col_ji0[0] + 1, col_ji0[1]),
                                                          1,
                                                          0,
                                                          quad_triangles = quad_triangles,
                                                          as_single_layer = True)
            j_minus_fault_surf_list.append(surf)
         if len(j_minus_fault_surf_list) > 0:
            j_minus_fault_surf = rqs.CombinedSurface(j_minus_fault_surf_list, grid.crs_uuid)
         if len(j_plus_fault_surf_list) > 0:
            j_plus_fault_surf = rqs.CombinedSurface(j_plus_fault_surf_list, grid.crs_uuid)

         i_minus_fault_surf = i_plus_fault_surf = None
         i_minus_fault_surf_list = []
         i_plus_fault_surf_list = []
         for col_ji0 in self.fault_i_face_cols_ji0:
            #            log.debug(f'fault i face col_ji0: {col_ji0}')
            _, surf = create_column_face_mesh_and_surface(grid,
                                                          col_ji0,
                                                          2,
                                                          1,
                                                          quad_triangles = quad_triangles,
                                                          as_single_layer = True)
            i_plus_fault_surf_list.append(surf)
            _, surf = create_column_face_mesh_and_surface(grid, (col_ji0[0], col_ji0[1] + 1),
                                                          2,
                                                          0,
                                                          quad_triangles = quad_triangles,
                                                          as_single_layer = True)
            i_minus_fault_surf_list.append(surf)
         if len(i_minus_fault_surf_list) > 0:
            i_minus_fault_surf = rqs.CombinedSurface(i_minus_fault_surf_list, grid.crs_uuid)
         if len(i_plus_fault_surf_list) > 0:
            i_plus_fault_surf = rqs.CombinedSurface(i_plus_fault_surf_list, grid.crs_uuid)

         surf_list = [top_surf, base_surf, j_minus_surf, j_plus_surf, i_minus_surf, i_plus_surf]
         for k_gap_surf in k_gap_surf_list:
            surf_list.append(k_gap_surf)
         for fault_surf in [j_minus_fault_surf, j_plus_fault_surf, i_minus_fault_surf, i_plus_fault_surf]:
            if fault_surf is not None:
               surf_list.append(fault_surf)

      else:

         top_surf = generate_untorn_surface_for_layer_interface(grid,
                                                                k0 = 0,
                                                                ref_k_faces = 'top',
                                                                quad_triangles = quad_triangles)
         base_surf = generate_untorn_surface_for_layer_interface(grid,
                                                                 k0 = grid.nk - 1,
                                                                 ref_k_faces = 'base',
                                                                 quad_triangles = quad_triangles)
         j_minus_surf = generate_untorn_surface_for_x_section(grid,
                                                              'J',
                                                              ref_slice0 = 0,
                                                              plus_face = False,
                                                              quad_triangles = quad_triangles,
                                                              as_single_layer = use_single_layer_tactics)
         j_plus_surf = generate_untorn_surface_for_x_section(grid,
                                                             'J',
                                                             ref_slice0 = self.grid.nj - 1,
                                                             plus_face = True,
                                                             quad_triangles = quad_triangles,
                                                             as_single_layer = use_single_layer_tactics)
         i_minus_surf = generate_untorn_surface_for_x_section(grid,
                                                              'I',
                                                              ref_slice0 = 0,
                                                              plus_face = False,
                                                              quad_triangles = quad_triangles,
                                                              as_single_layer = use_single_layer_tactics)
         i_plus_surf = generate_untorn_surface_for_x_section(grid,
                                                             'I',
                                                             ref_slice0 = self.grid.ni - 1,
                                                             plus_face = True,
                                                             quad_triangles = quad_triangles,
                                                             as_single_layer = use_single_layer_tactics)

         surf_list = [top_surf, base_surf, j_minus_surf, j_plus_surf, i_minus_surf, i_plus_surf]
         for k_gap_surf in k_gap_surf_list:
            surf_list.append(k_gap_surf)

      self.skin = rqs.CombinedSurface(surf_list)

   def find_first_intersection_of_trajectory(self,
                                             trajectory,
                                             start = 0,
                                             start_xyz = None,
                                             nudge = None,
                                             exclude_kji0 = None):
      """Returns the x,y,z and K,J,I and axis, polarity & segment of the first intersection of the trajectory with the torn skin.

      arguments:
         trajectory (well.Trajectory object): the trajectory to be intersected with the skin
         start (int, default 0): the trajectory segment number to start the search from
         start_xyz (triple float, optional): if present, this point should lie on the start segment and search continues from
            this point
         nudge (float, optional): if present and positive, the start point is nudged forward by this distance (grid uom);
            if present and negative (more typical for skin entry search), the start point is nudged back a little
         exclude_kji0 (triple int, optional): if present, the indices of a cell to exclude as a possible result

      returns:
         5-tuple (triple float, triple int, int, int, int): the first element is the xyz coordinates of the intersection point
         in the crs of the grid; second element is the kji0 of the cell that is intersected, which might be a pinched out
         or otherwise inactive cell; 3rd element is 0, 1 or 2 for K, J or I axis of cell face; 4th element is 0 for -ve
         face, 1 for +ve face; 5th element is the trajectory knot prior to the intersection (also the segment number)

      note:
         if the GridSkin object has been initialised using single layer tactics, then the k0 value will be zero for any
         initial entry through a sidewall of the grid or through a fault face
      """

      if exclude_kji0 is not None:
         xyz_1, segment_1, tri_index_1, xyz_2, segment_2, tri_index_2 =  \
            find_first_intersection_of_trajectory_with_surface(trajectory, self.skin, start = start, start_xyz = start_xyz,
                                                               nudge = nudge, return_second = True)
      else:
         xyz_1, segment_1, tri_index_1 = find_first_intersection_of_trajectory_with_surface(trajectory,
                                                                                            self.skin,
                                                                                            start = start,
                                                                                            start_xyz = start_xyz,
                                                                                            nudge = nudge)
         xyz_2 = segment_2 = tri_index_2 = None
      if xyz_1 is None:
         return None, None, None, None, None
      if xyz_2 is None:
         triplet_list = [(xyz_1, segment_1, tri_index_1)]
      else:
         triplet_list = [(xyz_1, segment_1, tri_index_1), (xyz_2, segment_2, tri_index_2)]

      for (xyz, segment, tri_index) in triplet_list:

         surf_index, surf_tri_index = self.skin.surface_index_for_triangle_index(tri_index)
         assert surf_index is not None

         if surf_index < 6:  # grid skin
            # following returns j,i pair for K faces; k,i for J faces; or k,j for I faces
            col = self.skin.surface_list[surf_index].column_from_triangle_index(surf_tri_index)
            if surf_index == 0:
               kji0 = (0, col[0], col[1])  # K- (top)
            elif surf_index == 1:
               kji0 = (self.grid.nk - 1, col[0], col[1])  # K+ (base)
            elif surf_index == 2:
               kji0 = (col[0], 0, col[1])  # J-
            elif surf_index == 3:
               kji0 = (col[0], self.grid.nj - 1, col[1])  # J+
            elif surf_index == 4:
               kji0 = (col[0], col[1], 0)  # I-
            else:
               kji0 = (col[0], col[1], self.grid.ni - 1)  # I+
            axis, polarity = divmod(surf_index, 2)
            if self.use_single_layer_tactics and surf_index > 1:  # now compare against layered representation of column face
               xyz, k0 = find_intersection_of_trajectory_interval_with_column_face(trajectory,
                                                                                   self.grid,
                                                                                   segment,
                                                                                   kji0[1:],
                                                                                   axis,
                                                                                   polarity,
                                                                                   start_xyz = xyz,
                                                                                   nudge = -1.0)
               if xyz is None:
                  log.error('unexpected failure to identify skin column face penetration point')
                  return None, None, None, None, None
               kji0 = (k0, kji0[1], kji0[2])

         elif surf_index < 6 + 2 * self.k_gaps:  # top or base face of a k gap
            col = self.skin.surface_list[surf_index].column_from_triangle_index(surf_tri_index)
            surf_index -= 6
            gap_index, top_base = divmod(surf_index, 2)
            axis = 0
            if top_base == 0:  # top of k gap (ie. base of layer before)
               kji0 = (self.k_gap_after_layer_list[gap_index], col[0], col[1])
               polarity = 1
            else:  # base of k gap (top of layer after)
               kji0 = (self.k_gap_after_layer_list[gap_index] + 1, col[0], col[1])
               polarity = 0

         else:  # fault face
            axis, polarity = divmod(surf_index - (6 + 2 * self.k_gaps), 2)
            axis += 1
            log.debug('penetration through fault face ' + 'KJI'[axis] + '-+'[polarity])
            assert self.skin.is_combined_list[surf_index]
            fault_surf_list = self.skin.surface_list[surf_index]
            col_surf_index, _ = fault_surf_list.surface_index_for_triangle_index(surf_tri_index)
            col_ji0 = np.empty((2,), dtype = int)
            surf_index -= 6 + 2 * self.k_gaps
            if surf_index in [0, 1]:  # J-/+ fault face
               col_ji0[:] = self.fault_j_face_cols_ji0[col_surf_index]
            elif surf_index in [2, 3]:  # I-/+ fault face
               col_ji0[:] = self.fault_i_face_cols_ji0[col_surf_index]
            else:  # should not be possible
               raise Exception('code failure')
            if polarity == 0:
               col_ji0[axis - 1] += 1
            xyz_f, k0 = find_intersection_of_trajectory_interval_with_column_face(trajectory,
                                                                                  self.grid,
                                                                                  segment,
                                                                                  col_ji0,
                                                                                  axis,
                                                                                  polarity,
                                                                                  start_xyz = xyz,
                                                                                  nudge = -1.0)
            if xyz_f is None:
               if self.k_gaps:
                  log.debug('failed to identify fault penetration point; assumed to be in k gap; nudging forward')
                  if nudge is None or nudge < 0.0:
                     nudge = 0.0
                  nudge += 0.1
                  return self.find_first_intersection_of_trajectory(trajectory,
                                                                    start = segment,
                                                                    start_xyz = xyz,
                                                                    nudge = nudge,
                                                                    exclude_kji0 = exclude_kji0)
               log.error('unexpected failure to identify fault penetration point')
               return None, None, None, None, None
            xyz = xyz_f
            kji0 = (k0, col_ji0[0], col_ji0[1])

         if exclude_kji0 is None or not np.all(kji0 == exclude_kji0):
            return xyz, kji0, axis, polarity, segment

      return None, None, None, None, None

   def _make_k_gap_surfaces(self, quad_triangles = True):
      """Returns a list of newly created surfaces representing top and base of each k gap."""

      k_gap_surf_list = [
      ]  # list of layer face surfaces, 2 per gap (top of gap, ie. base of layer above, then base of gap)
      self.k_gap_after_layer_list = []
      if self.k_gaps > 0:
         for k0 in range(self.grid.nk - 1):
            if self.grid.k_gap_after_array[k0]:
               if self.has_split_coordinate_lines:
                  gap_top_surf = generate_torn_surface_for_layer_interface(self.grid,
                                                                           k0 = k0,
                                                                           ref_k_faces = 'base',
                                                                           quad_triangles = quad_triangles)
                  gap_base_surf = generate_torn_surface_for_layer_interface(self.grid,
                                                                            k0 = k0 + 1,
                                                                            ref_k_faces = 'top',
                                                                            quad_triangles = quad_triangles)
               else:
                  gap_top_surf = generate_untorn_surface_for_layer_interface(self.grid,
                                                                             k0 = k0,
                                                                             ref_k_faces = 'base',
                                                                             quad_triangles = quad_triangles)
                  gap_base_surf = generate_untorn_surface_for_layer_interface(self.grid,
                                                                              k0 = k0 + 1,
                                                                              ref_k_faces = 'top',
                                                                              quad_triangles = quad_triangles)
               k_gap_surf_list.append(gap_top_surf)
               k_gap_surf_list.append(gap_base_surf)
               self.k_gap_after_layer_list.append(k0)
      assert len(k_gap_surf_list) == 2 * self.k_gaps
      assert len(self.k_gap_after_layer_list) == self.k_gaps
      return k_gap_surf_list


def generate_untorn_surface_for_layer_interface(grid,
                                                k0 = 0,
                                                ref_k_faces = 'top',
                                                quad_triangles = True,
                                                border = None):
   """Returns a Surface object generated from the grid layer interface points after any faults are 'healed'.

   arguments:
      grid (grid.Grid object): the grid object from which a layer interface is to be converted to a surface
      k0 (int): the layer number (zero based) to be used
      ref_k_faces (string): either 'top' (the default) or 'base', indicating whether the top or the base
         interface of the layer is to be used
      quad_triangles (boolean, optional, default True): if True, 4 triangles are used to represent each cell k face,
         which gives a unique solution with a shared node of the 4 triangles at the mean point of the 4 corners of
         the face; if False, only 2 triangles are used, which gives a non-unique solution
      border (float, optional): If given, an extra border row of quadrangles is added around the grid mesh

   returns:
      a resqml_surface.Surface object with a single triangulated patch

   notes:
      The resulting surface is assigned to the same model as grid, though xml is not generated and hdf5 is not
      written.
      If a border is specified and the outer grid cells have non-parallel edges, the resulting mesh might be
      messed up.
   """

   surf = rqs.Surface(grid.model)
   kp = 1 if ref_k_faces == 'base' else 0
   mesh = grid.horizon_points(ref_k0 = k0, heal_faults = True, kp = kp)
   if border is None or border <= 0.0:
      surf.set_from_irregular_mesh(mesh, quad_triangles = quad_triangles)
   else:
      #      origin = np.mean(mesh, axis = (0, 1))
      skirted_mesh = np.empty((mesh.shape[0] + 2, mesh.shape[1] + 2, 3))
      skirted_mesh[1:-1, 1:-1, :] = mesh
      # fill border values (other than corners)
      for j in range(1, mesh.shape[0] + 1):
         skirted_mesh[j, 0, :] = skirted_mesh[j, 1] + border * vec.unit_vector(skirted_mesh[j, 1] - skirted_mesh[j, 2])
         skirted_mesh[j,
                      -1, :] = skirted_mesh[j, -2] + border * vec.unit_vector(skirted_mesh[j, -2] - skirted_mesh[j, -3])
      for i in range(1, mesh.shape[1] + 1):
         skirted_mesh[0, i, :] = skirted_mesh[1, i] + border * vec.unit_vector(skirted_mesh[1, i] - skirted_mesh[2, i])
         skirted_mesh[-1,
                      i, :] = skirted_mesh[-2, i] + border * vec.unit_vector(skirted_mesh[-2, i] - skirted_mesh[-3, i])
      # fill in corner values
      skirted_mesh[0, 0, :] = skirted_mesh[0, 1] + skirted_mesh[1, 0] - skirted_mesh[1, 1]
      skirted_mesh[0, -1, :] = skirted_mesh[0, -2] + skirted_mesh[1, -1] - skirted_mesh[1, -2]
      skirted_mesh[-1, 0, :] = skirted_mesh[-1, 1] + skirted_mesh[-2, 0] - skirted_mesh[-2, 1]
      skirted_mesh[-1, -1, :] = skirted_mesh[-1, -2] + skirted_mesh[-2, -1] - skirted_mesh[-2, -2]
      surf.set_from_irregular_mesh(skirted_mesh, quad_triangles = quad_triangles)

   return surf


def generate_torn_surface_for_layer_interface(grid, k0 = 0, ref_k_faces = 'top', quad_triangles = True):
   """Returns a Surface object generated from the grid layer interface points.

   arguments:
      grid (grid.Grid object): the grid object from which a layer interface is to be converted to a surface
      k0 (int): the layer number (zero based) to be used
      ref_k_faces (string): either 'top' (the default) or 'base', indicating whether the top or the base
         interface of the layer is to be used
      quad_triangles (boolean, optional, default True): if True, 4 triangles are used to represent each cell k face,
         which gives a unique solution with a shared node of the 4 triangles at the mean point of the 4 corners of
         the face; if False, only 2 triangles are used, which gives a non-unique solution

   returns:
      a resqml_surface.Surface object with a single triangulated patch

   notes:
      The resulting surface is assigned to the same model as grid, though xml is not generated and hdf5 is not
      written.
      Strictly, the RESQML business rules for a triangulated surface require a separate patch for areas of the
      surface which are not joined; therefore, if fault tears cut off one area of the surface (eg. a fault running
      fully across the grid), then more than one patch should be generated; however, at present the code uses a
      single patch regardless.
   """

   surf = rqs.Surface(grid.model)
   kp = 1 if ref_k_faces == 'base' else 0
   mesh = grid.split_horizon_points(ref_k0 = k0, kp = kp)
   surf.set_from_torn_mesh(mesh, quad_triangles = quad_triangles)

   return surf


def generate_torn_surface_for_x_section(grid,
                                        axis,
                                        ref_slice0 = 0,
                                        plus_face = False,
                                        quad_triangles = True,
                                        as_single_layer = False):
   """Returns a Surface object generated from the grid cross section points.

   arguments:
      grid (grid.Grid object): the grid object from which a cross section is to be converted to a surface
      axis (string): 'I' or 'J' being the axis of the cross-sectional slice (ie. dimension being dropped)
      ref_slice0 (int, default 0): the reference value for indices in I or J (as defined in axis)
      plus_face (boolean, default False): if False, negative face is used; if True, positive
      quad_triangles (boolean, default True): if True, 4 triangles are used to represent each cell face, which
         gives a unique solution with a shared node of the 4 triangles at the mean point of the 4 corners of
         the face; if False, only 2 triangles are used, which gives a non-unique solution
      as_single_layer (boolean, default False): if True, the top points from the top layer are used together
         with the basal points from the base layer, to effect a single layer equivalent cross section surface

   returns:
      a resqml_surface.Surface object with a single triangulated patch

   notes:
      The resulting surface is assigned to the same model as grid, though xml is not generated and hdf5 is not
      written.
      Strictly, the RESQML business rules for a triangulated surface require a separate patch for areas of the
      surface which are not joined; therefore, a fault running down through the grid should result in separate
      patches; however, at present the code uses a single patch regardless.
   """

   assert axis.upper() in ['I', 'J']

   if grid.k_gaps is None or grid.k_gaps == 0:
      x_sect_points = grid.split_x_section_points(axis, ref_slice0 = ref_slice0, plus_face = plus_face)
      if as_single_layer:
         shape = np.array(x_sect_points.shape)
         shape[0] = 1
         x_sect_top = x_sect_points[0].reshape(tuple(shape))
         x_sect_base = x_sect_points[-1].reshape(tuple(shape))
      else:
         x_sect_top = x_sect_points[:-1]
         x_sect_base = x_sect_points[1:]
      x_sect_mesh = np.stack((x_sect_top, x_sect_base), axis = 2)
   else:
      x_sect_mesh = grid.split_gap_x_section_points(axis, ref_slice0 = ref_slice0, plus_face = plus_face)

   surf = rqs.Surface(grid.model)
   surf.set_from_torn_mesh(x_sect_mesh, quad_triangles = quad_triangles)

   return surf


def generate_untorn_surface_for_x_section(grid,
                                          axis,
                                          ref_slice0 = 0,
                                          plus_face = False,
                                          quad_triangles = True,
                                          as_single_layer = False):
   """Returns a Surface object generated from the grid cross section points for an unfaulted grid.

   arguments:
      grid (grid.Grid object): the grid object from which a cross section is to be converted to a surface
      axis (string): 'I' or 'J' being the axis of the cross-sectional slice (ie. dimension being dropped)
      ref_slice0 (int, default 0): the reference value for indices in I or J (as defined in axis)
      plus_face (boolean, default False): if False, negative face is used; if True, positive
      quad_triangles (boolean, default True): if True, 4 triangles are used to represent each cell face, which
         gives a unique solution with a shared node of the 4 triangles at the mean point of the 4 corners of
         the face; if False, only 2 triangles are used, which gives a non-unique solution
      as_single_layer (boolean, default False): if True, the top points from the top layer are used together
         with the basal points from the base layer, to effect a single layer equivalent cross section surface

   returns:
      a resqml_surface.Surface object with a single triangulated patch

   notes:
      The resulting surface is assigned to the same model as grid, though xml is not generated and hdf5 is not
      written.
      Strictly, the RESQML business rules for a triangulated surface require a separate patch for areas of the
      surface which are not joined; therefore, a fault running down through the grid should result in separate
      patches; however, at present the code uses a single patch regardless.
   """

   assert axis.upper() in ['I', 'J']

   x_sect_points = grid.x_section_points(axis, ref_slice0 = ref_slice0, plus_face = plus_face)
   if as_single_layer:
      shape = np.array(x_sect_points.shape)
      shape[0] = 1
      x_sect_top = x_sect_points[0]
      x_sect_base = x_sect_points[-1]
      x_sect_mesh = np.stack((x_sect_top, x_sect_base), axis = 0)
   else:
      x_sect_mesh = x_sect_points

   log.debug(f'x_sect_mesh.shape: {x_sect_mesh.shape}; grid.extent_kji: {grid.extent_kji}')

   surf = rqs.Surface(grid.model)
   surf.set_from_irregular_mesh(x_sect_mesh, quad_triangles = quad_triangles)

   return surf


def find_intersections_of_trajectory_with_surface(trajectory, surface):
   """Returns an array of triangle indices and an array of xyz of intersections of well trajectory with surface.

   arguments:
      trajectory (well.Trajectory object; or list thereof): the wellbore trajectory object(s) to find the intersections for;
         if a list of trajectories is provided then the return value is a corresponding list
      surface (surface.Surface object): the triangulated surface with which to intersect the trajectory

   returns:
      (numpy int array of shape (N,), numpy float array of shape (N, 3)): the first array is a list of surface
      triangle indices containing an intersection and the second array is the corresponding list of (x, y, z)
      intersection points; if the trajectory argument is a list of trajectories, then a correponding list of numpy
      array pairs is returned

   notes:
      interseections are found based on straight line segments between the trajectory control points, this will result
      in errors where there is significant curvature between neighbouring control points;
      a given triangle index might appear more than once in the first returned array
   """

   if isinstance(trajectory, list):
      trajectory_list = trajectory
   else:
      trajectory_list = [trajectory]

   t, p = surface.triangles_and_points()
   triangles = p[t]
   #   log.debug('finding intersections of wellbore trajectory(ies) with layer: ' + str(k0) + ' ' + ref_k_faces)
   results_list = []
   for traj in trajectory_list:
      all_intersects = meet.poly_line_triangles_intersects(traj.control_points, triangles)
      _, triangle_indices, intersect_points = meet.distilled_intersects(
         all_intersects)  # discard trajectory segment info
      if len(triangle_indices) == 0:
         results_list.append((None, None))
      else:
         results_list.append((triangle_indices, intersect_points))

   if isinstance(trajectory, list):
      return results_list
   return results_list[0]


def find_intersections_of_trajectory_with_layer_interface(trajectory,
                                                          grid,
                                                          k0 = 0,
                                                          ref_k_faces = 'top',
                                                          heal_faults = True,
                                                          quad_triangles = True):
   """Returns an array of column indices and an array of xyz of intersections of well trajectory with layer interface.

   arguments:
      trajectory (well.Trajectory object; or list thereof): the wellbore trajectory object(s) to find the intersections for;
         if a list of trajectories is provided then the return value is a corresponding list
      grid (grid.Grid object): the grid object from which a layer interface is to be converted to a surface
      k0 (int): the layer number (zero based) to be used
      ref_k_faces (string): either 'top' (the default) or 'base', indicating whether the top or the base
         interface of the layer is to be used
      heal_faults (boolean, default True): if True, faults will be 'healed' to give an untorn surface before looking
         for intersections; if False and the trajectory passes through a fault plane without intersecting the layer
         interface then no intersection will be identified; makes no difference if the grid is unfaulted
      quad_triangles (boolean, optional, default True): if True, 4 triangles are used to represent each cell k face,
         which gives a unique solution with a shared node of the 4 triangles at the mean point of the 4 corners of
         the face; if False, only 2 triangles are used, which gives a non-unique solution

   returns:
      (numpy int array of shape (N, 2), numpy float array of shape (N, 3)): the first array is a list of (j0, i0)
      indices of columns containing an intersection and the second array is the corresponding list of (x, y, z)
      intersection points; if the trajectory argument is a list of trajectories, then a correponding list of numpy
      array pairs is returned

   notes:
      interseections are found based on straight line segments between the trajectory control points, this will result
      in errors where there is significant curvature between neighbouring control points;
      a given (j0, i0) column might appear more than once in the first returned array
   """

   if isinstance(trajectory, list):
      trajectory_list = trajectory
   else:
      trajectory_list = [trajectory]

   log.debug('generating surface for layer: ' + str(k0) + ' ' + ref_k_faces)
   if grid.has_split_coordinate_lines and not heal_faults:
      surface = generate_torn_surface_for_layer_interface(grid,
                                                          k0 = k0,
                                                          ref_k_faces = ref_k_faces,
                                                          quad_triangles = quad_triangles)
   else:
      surface = generate_untorn_surface_for_layer_interface(grid,
                                                            k0 = k0,
                                                            ref_k_faces = ref_k_faces,
                                                            quad_triangles = quad_triangles)

   tri_intersect_list = find_intersections_of_trajectory_with_surface(trajectory_list, surface)

   results_list = []
   for triangle_indices, intersect_points in tri_intersect_list:
      if triangle_indices is None or intersect_points is None:
         results_list.append((None, None))
      else:
         j_list, i_list = surface.column_from_triangle_index(
            triangle_indices)  # todo: check this is valid for torn surfaces
         assert j_list is not None, 'failed to derive column indices from triangle indices'
         cols = np.stack((j_list, i_list), axis = -1)
         results_list.append((cols, intersect_points))

   if isinstance(trajectory, list):
      return results_list
   return results_list[0]


def find_first_intersection_of_trajectory_with_surface(trajectory,
                                                       surface,
                                                       start = 0,
                                                       start_xyz = None,
                                                       nudge = None,
                                                       return_second = False):
   """Returns xyz and other info of the first intersection of well trajectory with layer interface.

   arguments:
      trajectory (well.Trajectory object): the wellbore trajectory object(s) to find the intersection for
      surface (surface.Surface object): the triangulated surface with which to search for intersections
      start (int, default 0): an index into the trajectory knots to start the search from
      start_xyz (triple float, optional): if present, should lie on start segment and search starts from this point
      nudge (float, optional): if present and positive, starting xyz is nudged forward this distance along segment;
         if present and negative, starting xyz is nudged backward along segment
      return_second (boolean, default False): if True, a sextuplet is returned with the last 3 elements identifying
         the 'runner up' intersection in the same trajectory segment, or None, None, None if only one intersection found

   returns:
      a triplet if return_second is False; a sextuplet if return_second is True; the first triplet is:
      (numpy float array of shape (3,), int, int): being the (x, y, z) intersection point, and the trajectory segment number,
      and the triangle index of the first intersection point; or None, None, None if no intersection found;
      if return_second is True, the 4th, 5th & 6th return values are similar to the first three, conveying information
      about the second intersection of the same trajectory segment with the surface, or None, None, None if a no second
      intersection was found

   notes:
      interseections are found based on straight line segments between the trajectory control points, this will result
      in errors where there is significant curvature between neighbouring control points
   """

   if start >= trajectory.knot_count - 1:
      if return_second:
         return None, None, None, None, None, None
      return None, None, None

   t, p = surface.triangles_and_points()
   triangles = p[t]

   xyz = None
   tri = None
   knot = start
   xyz_2 = tri_2 = None
   if start_xyz is None:
      start_xyz = trajectory.control_points[knot]
   if knot < trajectory.knot_count - 2 and vec.isclose(start_xyz, trajectory.control_points[knot + 1]):
      knot += 1
   if nudge is not None and nudge != 0.0:
      remaining = trajectory.control_points[knot + 1] - start_xyz
      if vec.naive_length(remaining) <= nudge:
         start_xyz += 0.9 * remaining
      else:
         start_xyz += nudge * vec.unit_vector(remaining)
   while knot < trajectory.knot_count - 1:
      if start_xyz is None:
         line_p = trajectory.control_points[knot]
      else:
         line_p = start_xyz
      start_xyz = None
      line_v = trajectory.control_points[knot + 1] - line_p
      intersects = meet.line_triangles_intersects(line_p, line_v, triangles, line_segment = True)
      if not np.all(np.isnan(intersects)):
         intersects_indices = meet.intersects_indices(intersects)
         tri = intersects_indices[0]
         xyz = intersects[tri]
         if len(intersects_indices) > 1:
            best_manhattan = vec.manhattan_distance(line_p, xyz)
            runner_up = None
            for other_intersect_index_index in range(1, len(intersects_indices)):
               other_tri = intersects_indices[other_intersect_index_index]
               other_xyz = intersects[other_tri]
               other_manhattan = vec.manhattan_distance(line_p, other_xyz)
               if other_manhattan < best_manhattan:
                  if return_second:
                     tri_2 = tri
                     xyz_2 = xyz
                     runner_up = best_manhattan
                  tri = other_tri
                  xyz = other_xyz
                  best_manhattan = other_manhattan
               elif return_second and (runner_up is None or other_manhattan < runner_up):
                  tri_2 = other_tri
                  xyz_2 = other_xyz
                  runner_up = other_manhattan
         break
      knot += 1
   if knot == trajectory.knot_count - 1:
      knot = None

   if return_second:
      return xyz, knot, tri, xyz_2, knot, tri_2
   return xyz, knot, tri


def find_first_intersection_of_trajectory_with_layer_interface(trajectory,
                                                               grid,
                                                               k0 = 0,
                                                               ref_k_faces = 'top',
                                                               start = 0,
                                                               heal_faults = False,
                                                               quad_triangles = True):
   """Returns xyz and other info of the first intersection of well trajectory (or list of trajectories) with layer interface.

   arguments:
      trajectory (well.Trajectory object; or list thereof): the wellbore trajectory object(s) to find the intersection for;
         if a list of trajectories is provided then the return value is a corresponding list
      grid (grid.Grid object): the grid object from which a layer interface is to be converted to a surface
      k0 (int): the layer number (zero based) to be used
      ref_k_faces (string): either 'top' (the default) or 'base', indicating whether the top or the base
         interface of the layer is to be used
      start (int, default 0): an index into the trajectory knots to start the search from; is applied naively to all
         trajectories when a trajectory list is passed
      heal_faults (boolean, default False): if True, faults will be 'healed' to give an untorn surface before looking
         for intersections; if False and the trajectory passes through a fault plane without intersecting the layer
         interface then no intersection will be identified; makes no difference if the grid is unfaulted
      quad_triangles (boolean, optional, default True): if True, 4 triangles are used to represent each cell k face,
         which gives a unique solution with a shared node of the 4 triangles at the mean point of the 4 corners of
         the face; if False, only 2 triangles are used, which gives a non-unique solution

   returns:
      (numpy float array of shape (3,), int, (int, int)): being the (x, y, z) intersection point, and the trajectory segment number,
      and the (j0, i0) column number of the first intersection point;
      or None, None, (None, None) if no intersection found;
      if the trajectory argument is a list of trajectories, then correponding list of numpy array, list of int, list of int pair
      are returned

   notes:
      interseections are found based on straight line segments between the trajectory control points, this will result
      in errors where there is significant curvature between neighbouring control points
   """

   if isinstance(trajectory, list):
      trajectory_list = trajectory
   else:
      trajectory_list = [trajectory]

   #   log.debug('generating surface for layer: ' + str(k0) + ' ' + ref_k_faces)
   if grid.has_split_coordinate_lines and not heal_faults:
      surface = generate_torn_surface_for_layer_interface(grid,
                                                          k0 = k0,
                                                          ref_k_faces = ref_k_faces,
                                                          quad_triangles = quad_triangles)
   else:
      surface = generate_untorn_surface_for_layer_interface(grid,
                                                            k0 = k0,
                                                            ref_k_faces = ref_k_faces,
                                                            quad_triangles = quad_triangles)


#   log.debug('finding intersections of wellbore trajectory(ies) with layer: ' + str(k0) + ' ' + ref_k_faces)
   results_list = []
   segment_list = []
   col_list = []
   for traj in trajectory_list:
      xyz, knot, tri = find_first_intersection_of_trajectory_with_surface(traj, surface, start = start)
      col = surface.column_from_triangle_index(tri)  # j, i pair returned
      results_list.append(xyz)
      segment_list.append(knot)
      col_list.append(col)

   if isinstance(trajectory, list):
      return results_list, segment_list, col_list
   return results_list[0], segment_list[0], col_list[0]


def find_first_intersection_of_trajectory_with_cell_surface(trajectory,
                                                            grid,
                                                            kji0,
                                                            start_knot,
                                                            start_xyz = None,
                                                            nudge = 0.001,
                                                            quad_triangles = True):
   """Searches along the trajectory from a starting point until the first intersection with the cell's surface is encountered."""

   cp = grid.corner_points(kji0)
   cell_surface = rqs.Surface(grid.model)
   cell_surface.set_to_single_cell_faces_from_corner_points(cp, quad_triangles = quad_triangles)
   t, p = cell_surface.triangles_and_points()
   triangles = p[t]
   knot = start_knot
   xyz = None
   axis = polarity = None
   if start_xyz is not None:
      start_xyz = np.array(start_xyz)
   while knot < trajectory.knot_count - 1:
      if knot == start_knot and start_xyz is not None:
         if knot < trajectory.knot_count - 2 and vec.isclose(start_xyz, trajectory.control_points[knot + 1]):
            knot += 1
         if nudge is not None and nudge != 0.0:
            remaining = trajectory.control_points[knot + 1] - start_xyz
            if vec.naive_length(remaining) <= nudge:
               start_xyz += 0.9 * remaining
            else:
               start_xyz += nudge * vec.unit_vector(remaining)
         line_p = start_xyz
      else:
         line_p = trajectory.control_points[knot]
      line_v = trajectory.control_points[knot + 1] - line_p
      #      log.debug('kji0: ' + str(kji0) + '; knot: ' + str(knot) + '; line p: ' + str(line_p) + '; v: ' + str(line_v))
      intersects = meet.line_triangles_intersects(line_p, line_v, triangles, line_segment = True)
      if not np.all(np.isnan(intersects)):
         # if more than one intersect, could find one closest to line_p; should not be needed when starting inside cell
         tri = meet.intersects_indices(intersects)[0]
         xyz = intersects[tri]
         _, axis, polarity = cell_surface.cell_axis_and_polarity_from_triangle_index(tri)
         log.debug('intersection xyz: ' + str(xyz) + '; tri: ' + str(tri) + '; axis: ' + str(axis) + '; polarity: ' +
                   str(polarity))
         break
      knot += 1

   if knot == trajectory.knot_count - 1:
      knot = None
   return xyz, knot, axis, polarity


def point_is_within_cell(xyz, grid, kji0, cell_surface = None, false_on_pinchout = True):
   """Returns True if point xyz is within cell kji0, but not on its surface."""

   if false_on_pinchout and grid.pinched_out(kji0, cache_pinchout_array = False):
      return False
   if cell_surface is None:
      cp = grid.corner_points(kji0)
      cell_surface = rqs.Surface(grid.model)
      cell_surface.set_to_single_cell_faces_from_corner_points(cp, quad_triangles = True)
   t, p = cell_surface.triangles_and_points()
   triangles = p[t]
   centre = grid.centre_point(kji0)
   line_v = centre - xyz
   intersects = meet.line_triangles_intersects(xyz, line_v, triangles, line_segment = True)
   return np.all(np.isnan(intersects))


def create_column_face_mesh_and_surface(grid, col_ji0, axis, polarity, quad_triangles = True, as_single_layer = False):
   """Creates a Mesh and corresponding Surface representing a column face.

   arguments:
      grid (grid.Grid object)
      col_ji0 (int pair): the column indices, zero based
      axis (int): 1 for J face, 2 fo I face
      polarity (int): 0 for negative face, 1 for positive
      quad_triangles (boolean, default True): if True, 4 triangles are used per cell face; if False, 2 triangles
      as_single_layer (boolean, default False): if True, only the top and basal points are used, with the results being
         equivalent to the grid being treated as a single layer

   returns:
      surface.Mesh, surface.Surface (or None, surface.Surface if grid has k gaps)
   """

   assert axis in (1, 2)

   col_pm = grid.create_column_pillar_mapping()[col_ji0[0], col_ji0[1]]
   if axis == 1:  # J face
      pillar_index_pair = col_pm[polarity, :]
   else:  # I face
      pillar_index_pair = col_pm[:, polarity]
   if grid.k_gaps:
      points = grid.points_ref(masked = False).reshape(grid.nk_plus_k_gaps + 1, -1, 3)
   else:
      points = grid.points_ref(masked = False).reshape(grid.nk + 1, -1, 3)
   # note, here col_face_xyz is indexed by (j or i, k, xyz) whereas elsewhere (k, j or i, xyz) would be more typical
   # this protocol needs to align with re-use of Surface.column_for_triangle_index() method for layer identification

   if not as_single_layer and grid.k_gaps:
      col_face_mesh = None
      col_face_surface = rqs.Surface(grid.model)
      mesh = np.empty((1, grid.nk, 2, 2, 3))
      mesh[0, :, 0, 0, :] = points[grid.k_raw_index_array, pillar_index_pair[0], :]
      mesh[0, :, 1, 0, :] = points[grid.k_raw_index_array, pillar_index_pair[1], :]
      mesh[0, :, 0, 1, :] = points[grid.k_raw_index_array + 1, pillar_index_pair[0], :]
      mesh[0, :, 1, 1, :] = points[grid.k_raw_index_array + 1, pillar_index_pair[1], :]
      col_face_surface.set_from_torn_mesh(mesh, quad_triangles = quad_triangles)

   else:
      if as_single_layer:
         col_face_xyz = np.empty((2, 2, 3))
         col_face_xyz[0, 0] = points[0, pillar_index_pair[0]]
         col_face_xyz[0, 1] = points[-1, pillar_index_pair[0]]
         col_face_xyz[1, 0] = points[0, pillar_index_pair[1]]
         col_face_xyz[1, 1] = points[-1, pillar_index_pair[1]]
      else:
         col_face_xyz = np.empty((2, grid.nk + 1, 3))
         col_face_xyz[0] = points[:, pillar_index_pair[0]]
         col_face_xyz[1] = points[:, pillar_index_pair[1]]
      col_face_mesh = rqs.Mesh(grid.model, xyz_values = col_face_xyz, crs_uuid = grid.crs_uuid)
      title = 'column face for j0,i0: ' + str(col_ji0[0]) + ',' + str(
         col_ji0[1]) + ' face ' + 'KJI'[axis] + '-+'[polarity]
      col_face_surface = rqs.Surface(grid.model, mesh = col_face_mesh, quad_triangles = quad_triangles, title = title)

   return col_face_mesh, col_face_surface


def find_intersection_of_trajectory_interval_with_column_face(trajectory,
                                                              grid,
                                                              start_knot,
                                                              col_ji0,
                                                              axis,
                                                              polarity,
                                                              start_xyz = None,
                                                              nudge = None,
                                                              quad_triangles = True):
   """Searches for intersection of a single trajectory segment with an I or J column face.

   returns:
      xyz, k0

   note:
      does not support k gaps
   """

   # build a set of faces for the column face
   # extract column face points into a shape ready to become a mesh
   _, col_face_surface = create_column_face_mesh_and_surface(grid,
                                                             col_ji0,
                                                             axis,
                                                             polarity,
                                                             quad_triangles = quad_triangles)
   t, p = col_face_surface.triangles_and_points()
   triangles = p[t]
   log.debug(f"intersecting trajectory segment with column face ji0 {col_ji0} face {'KJI'[axis]}{'-+'[polarity]}")
   if start_xyz is not None:
      if nudge is not None and nudge != 0.0:  # here we typically nudge back up the wellbore
         start_xyz += nudge * vec.unit_vector(trajectory.control_points[start_knot + 1] - start_xyz)
      line_p = start_xyz
   else:
      line_p = trajectory.control_points[start_knot]
   line_v = trajectory.control_points[start_knot + 1] - line_p
   intersects = meet.line_triangles_intersects(line_p, line_v, triangles, line_segment = True)
   if np.all(np.isnan(intersects)):
      return None, None
   tri = meet.intersects_indices(intersects)[0]  # todo: if more than one, find one closest to line_p
   xyz = intersects[tri]
   _, k0 = col_face_surface.column_from_triangle_index(tri)  # 'column' in method name here maps to layers!
   return xyz, k0


def find_faces_to_represent_surface_staffa(grid, surface, name, progress_fn = None):
   """Returns a grid connection set containing those cell faces which are deemed to represent the surface."""

   #   log.debug('computing cell centres')
   centre_points = grid.centre_point()
   #   log.debug('computing inter cell centre vectors and boxes')
   if grid.nk > 1:
      v = centre_points[:-1, :, :]
      u = centre_points[1:, :, :]
      k_vectors = u - v
      combo = np.stack((v, u))
      k_vector_boxes = np.empty((grid.nk - 1, grid.nj, grid.ni, 2, 3))
      k_vector_boxes[:, :, :, 0, :] = np.amin(combo, axis = 0)
      k_vector_boxes[:, :, :, 1, :] = np.amax(combo, axis = 0)
      column_k_vector_boxes = np.empty((grid.nj, grid.ni, 2, 3))
      column_k_vector_boxes[:, :, 0, :] = np.amin(k_vector_boxes[:, :, :, 0, :], axis = 0)
      column_k_vector_boxes[:, :, 1, :] = np.amax(k_vector_boxes[:, :, :, 1, :], axis = 0)
      k_faces = np.zeros((grid.nk - 1, grid.nj, grid.ni), dtype = bool)
   else:
      k_vectors = None
      k_vector_boxes = None
      column_k_vector_boxes = None
      k_faces = None
   if grid.nj > 1:
      v = centre_points[:, :-1, :]
      u = centre_points[:, 1:, :]
      j_vectors = u - v
      combo = np.stack((v, u))
      j_vector_boxes = np.empty((grid.nk, grid.nj - 1, grid.ni, 2, 3))
      j_vector_boxes[:, :, :, 0, :] = np.amin(combo, axis = 0)
      j_vector_boxes[:, :, :, 1, :] = np.amax(combo, axis = 0)
      column_j_vector_boxes = np.empty((grid.nj - 1, grid.ni, 2, 3))
      column_j_vector_boxes[:, :, 0, :] = np.amin(j_vector_boxes[:, :, :, 0, :], axis = 0)
      column_j_vector_boxes[:, :, 1, :] = np.amax(j_vector_boxes[:, :, :, 1, :], axis = 0)
      j_faces = np.zeros((grid.nk, grid.nj - 1, grid.ni), dtype = bool)
   else:
      j_vectors = None
      j_vector_boxes = None
      column_j_vector_boxes = None
      j_faces = None
   if grid.ni > 1:
      i_vectors = centre_points[:, :, 1:] - centre_points[:, :, :-1]
      v = centre_points[:, :, :-1]
      u = centre_points[:, :, 1:]
      i_vectors = u - v
      combo = np.stack((v, u))
      i_vector_boxes = np.empty((grid.nk, grid.nj, grid.ni - 1, 2, 3))
      i_vector_boxes[:, :, :, 0, :] = np.amin(combo, axis = 0)
      i_vector_boxes[:, :, :, 1, :] = np.amax(combo, axis = 0)
      column_i_vector_boxes = np.empty((grid.nj, grid.ni - 1, 2, 3))
      column_i_vector_boxes[:, :, 0, :] = np.amin(i_vector_boxes[:, :, :, 0, :], axis = 0)
      column_i_vector_boxes[:, :, 1, :] = np.amax(i_vector_boxes[:, :, :, 1, :], axis = 0)
      i_faces = np.zeros((grid.nk, grid.nj, grid.ni - 1), dtype = bool)
   else:
      i_vectors = None
      i_vector_boxes = None
      column_i_vector_boxes = None
      i_faces = None

#   log.debug('finding surface triangle boxes')
   t, p = surface.triangles_and_points()
   triangles = p[t]
   assert triangles.size > 0, 'no triangles in surface'
   triangle_boxes = np.empty((triangles.shape[0], 2, 3))
   triangle_boxes[:, 0, :] = np.amin(triangles, axis = 1)
   triangle_boxes[:, 1, :] = np.amax(triangles, axis = 1)

   grid_box = grid.xyz_box(lazy = False)

   #   log.debug('looking for cell faces for each triangle')
   batch_size = 1000
   triangle_count = triangles.shape[0]
   progress_batch = min(1.0, float(batch_size) / float(triangle_count))
   progress_base = 0.0
   ti_base = 0
   if progress_fn is not None:
      progress_fn(0.0)
   while ti_base < triangle_count:
      ti_end = min(ti_base + batch_size, triangle_count)
      batch_box = np.empty((2, 3))
      batch_box[0, :] = np.amin(triangle_boxes[ti_base:ti_end, 0, :], axis = 0)
      batch_box[1, :] = np.amax(triangle_boxes[ti_base:ti_end, 1, :], axis = 0)
      if bx.boxes_overlap(grid_box, batch_box):
         for j in range(grid.nj):
            if progress_fn is not None:
               progress_fn(progress_base + progress_batch * (float(j) / float(grid.nj)))
            for i in range(grid.ni):
               if column_k_vector_boxes is not None and bx.boxes_overlap(batch_box, column_k_vector_boxes[j, i]):
                  full_intersects = meet.line_set_triangles_intersects(centre_points[:-1, j, i],
                                                                       k_vectors[:, j, i],
                                                                       triangles[ti_base:ti_end],
                                                                       line_segment = True)
                  distilled_intersects, _, _ = meet.distilled_intersects(full_intersects)
                  k_faces[distilled_intersects, j, i] = True
               if j < grid.nj - 1 and column_j_vector_boxes is not None and bx.boxes_overlap(
                     batch_box, column_j_vector_boxes[j, i]):
                  full_intersects = meet.line_set_triangles_intersects(centre_points[:, j, i],
                                                                       j_vectors[:, j, i],
                                                                       triangles[ti_base:ti_end],
                                                                       line_segment = True)
                  distilled_intersects, _, _ = meet.distilled_intersects(full_intersects)
                  j_faces[distilled_intersects, j, i] = True
               if i < grid.ni - 1 and column_i_vector_boxes is not None and bx.boxes_overlap(
                     batch_box, column_i_vector_boxes[j, i]):
                  full_intersects = meet.line_set_triangles_intersects(centre_points[:, j, i],
                                                                       i_vectors[:, j, i],
                                                                       triangles[ti_base:ti_end],
                                                                       line_segment = True)
                  distilled_intersects, _, _ = meet.distilled_intersects(full_intersects)
                  i_faces[distilled_intersects, j, i] = True
      ti_base = ti_end
      #      log.debug('triangles processed: ' + str(ti_base))
      #      log.debug('interim face counts: K: ' + str(np.count_nonzero(k_faces)) +
      #                                   '; J: ' + str(np.count_nonzero(j_faces)) +
      #                                   '; I: ' + str(np.count_nonzero(i_faces)))
      progress_base = min(1.0, progress_base + progress_batch)


#   log.debug('face counts: K: ' + str(np.count_nonzero(k_faces)) +
#                        '; J: ' + str(np.count_nonzero(j_faces)) +
#                        '; I: ' + str(np.count_nonzero(i_faces)))
   if progress_fn is not None:
      progress_fn(1.0)

   gcs = rqf.GridConnectionSet(grid.model,
                               grid = grid,
                               k_faces = k_faces,
                               j_faces = j_faces,
                               i_faces = i_faces,
                               feature_name = name,
                               create_organizing_objects_where_needed = True)
   return gcs


def find_faces_to_represent_surface(grid, surface, name, mode = 'staffa', progress_fn = None):
   """Returns a grid connection set containing those cell faces which are deemed to represent the surface."""

   log.debug('finding cell faces for surface')
   if mode == 'staffa':
      return find_faces_to_represent_surface_staffa(grid, surface, name, progress_fn = progress_fn)
   log.critical('unrecognised mode: ' + str(mode))
   return None


def generate_surface_for_blocked_well_cells(blocked_well,
                                            combined = True,
                                            active_only = False,
                                            min_k0 = 0,
                                            max_k0 = None,
                                            depth_limit = None,
                                            quad_triangles = True):
   """Returns a surface or list of surfaces representing the faces of the cells visited by the well."""

   assert blocked_well is not None and type(blocked_well) is rqw.BlockedWell
   cells_kji0, grid_list = blocked_well.cell_indices_and_grid_list()
   assert len(cells_kji0) == len(grid_list)
   cell_count = len(cells_kji0)
   if cell_count == 0:
      return None

   surface_list = []
   if combined:
      composite_cp = np.zeros((cell_count, 2, 2, 2, 3))

   cell_p = 0
   for cell in range(cell_count):
      grid = grid_list[cell]
      cell_kji0 = cells_kji0[cell]
      if active_only and grid.inactive is not None and grid.inactive[cell_kji0]:
         continue
      if cell_kji0[0] < min_k0 or (max_k0 is not None and cell_kji0[0] > max_k0):
         continue
      if depth_limit is not None and grid.centre_point(cell_kji0) > depth_limit:
         continue
      cp = grid.corner_points(cell_kji0)
      if combined:
         composite_cp[cell_p] = cp
         cell_p += 1
      else:
         cs = rqs.Surface(blocked_well.model)
         cs.set_to_single_cell_faces_from_corner_points(cp, quad_triangles = quad_triangles)
         surface_list.append(cs)

   if cell_p == 0 and len(surface_list) == 0:
      return None

   if combined:
      cs = rqs.Surface(blocked_well.model)
      cs.set_to_multi_cell_faces_from_corner_points(composite_cp[:cell_p])
      return cs

   return surface_list


def trajectory_grid_overlap(trajectory, grid, lazy = False):
   """Returns True if there is some overlap of the xyz boxes for the trajectory and grid, False otherwise.

   notes:
      overlap of the xyz boxes does not guarantee that the trajectory intersects the grid;
      a return value of False guarantees that the trajectory does not intersect the grid
   """

   traj_box = np.empty((2, 3))
   traj_box[0] = np.amin(trajectory.control_points, axis = 0)
   traj_box[1] = np.amax(trajectory.control_points, axis = 0)
   grid_box = grid.xyz_box(lazy = lazy, local = True)
   if not bu.matching_uuids(trajectory.crs_uuid, grid.crs_uuid):
      t_crs = rqc.Crs(trajectory.model, uuid = trajectory.crs_uuid)
      g_crs = rqc.Crs(grid.model, uuid = grid.crs_uuid)
      t_crs.convert_array_to(g_crs, traj_box)
   return bx.boxes_overlap(traj_box, grid_box)


def populate_blocked_well_from_trajectory(blocked_well,
                                          grid,
                                          active_only = False,
                                          quad_triangles = True,
                                          lazy = False,
                                          use_single_layer_tactics = True,
                                          check_for_reentry = True):
   """Populate an empty blocked well object based on the intersection of its trajectory with a grid.

   arguments:
      blocked_well (resqpy.well.BlockedWell object): a largely empty blocked well object to be populated by this
         function; note that the trajectory attribute must be set before calling this function
      grid (resqpy.grid.Grid object): the grid to intersect the well trajectory with
      active_only (boolean, default False): if True, intervals which cover inactive cells will be set as
         unblocked intervals; if False, intervals covering any cell will be set as blocked intervals
      quad_triangles (boolean, default True): if True, each cell face is represented by 4 triangles when
         calculating intersections; if False, only 2 triangles are used
      lazy (boolean, default False): if True, initial penetration must be through a top K face and blocking
         will cease as soon as the trajectory first leaves the gridded volume; if False, initial entry may be
         through any external face or a fault face and re-entry will be handled
      use_single_layer_tactics (boolean, default True): if True and not lazy and the grid does not have k gaps,
         fault planes and grid sidewall are initially treated as if the grid were a single layer, when looking
         for penetrations from outwith the grid
      check_for_reentry (boolean, default True): if True, the trajectory is tracked after leaving the grid through
         the outer skin (eg. base reservoir) in case of re-entry; if False, blocking stops upon the first exit
         of the trajectory through the skin; ignored (treated as False) if lazy is True

   returns:
      the blocked well object (same object as passed in) if successful; None if unsuccessful

   notes:
      the blocked_well trajectory attribute must be set before calling this function;
      grids with k gaps might result in very slow processing;
      the function represents a cell face as 2 or 4 triangles rather than a bilinear patch; setting quad_triangles
      False is not recommended as the 2 triangle formulation gives a non-unique representation of a face (though
      the code is designed to use the same representation for a shared face between neighbouring cells);
      where non-planar faults exist, the triangulation of faces may result in a small misalignment of abutted faces
      between the opposing sides of the fault; this could potentially result in an extra, small, unblocked interval
      being introduced as an artefact between the exit point of one cell and the entry point into the abutted cell
   """

   def find_next_cell(grid,
                      previous_kji0,
                      axis,
                      polarity,
                      trajectory,
                      segment,
                      seg_fraction,
                      xyz,
                      treat_skin_as_fault = False,
                      lazy = False,
                      use_single_layer_tactics = True):
      # returns for next cell entry: (shared transit point bool, kji0, axis, polarity, segment, seg_fraction, xyz)
      # or single None if no next cell identified (end of trajectory beyond edge of grid)
      # take care of: edges of model; pinchouts; faults, (k gaps), exact edge or corner crossings
      # note: identified cell may be active or inactive: calling code to handle that
      # note: for convenience, previous_kji0 may lie just outside the extent of the grid
      # note: polarity is relative to previous cell, so its complement applies to the next cell
      #      log.debug('finding next cell with previous kji0: ' + str(previous_kji0) + '; exit: ' + 'kji'[axis] + '-+'[polarity])
      kji0 = np.array(previous_kji0, dtype = int)
      polarity_sign = 2 * polarity - 1
      kji0[axis] += polarity_sign
      # if gone beyond external skin of model, return None
      if np.any(kji0 < 0) or np.any(kji0 >= grid.extent_kji) or (grid.k_gaps and axis == 0 and (
         (polarity == 1 and previous_kji0[0] >= 0 and grid.k_gap_after_array[previous_kji0[0]]) or
         (polarity == 0 and kji0[0] < grid.nk - 1 and grid.k_gap_after_array[kji0[0]]))):
         if check_for_reentry and not lazy:
            skin = grid.skin(use_single_layer_tactics = use_single_layer_tactics)
            # nudge in following will be problematic if gap has zero (or tiny) thickness at this location
            xyz_r, cell_kji0, axis, polarity, segment = skin.find_first_intersection_of_trajectory(trajectory,
                                                                                                   start = segment,
                                                                                                   start_xyz = xyz,
                                                                                                   nudge = +0.05)
            if xyz_r is None:
               log.debug('no skin re-entry found after exit through skin')
               return None
            log.debug(f"skin re-entry after skin exit: kji0: {cell_kji0}; face: {'KJI'[axis]}{'-+'[polarity]}")
            seg_fraction = segment_fraction(trajectory.control_points, segment, xyz_r)
            return (False, cell_kji0, axis, polarity, segment, seg_fraction, xyz_r)
         else:
            return None
      # pre-assess split pillar (fault) situation
      faulted = False
      if grid.has_split_coordinate_lines and axis != 0:
         faulted = grid.is_split_column_face(kji0[1], kji0[2], axis, 1 - polarity)
      if not faulted and treat_skin_as_fault and axis != 0:
         faulted = (kji0[axis] == 0 and polarity == 1) or (kji0[axis] == grid.extent_kji[axis] - 1 and polarity == 0)
      # handle the simplest case of a well behaved k neighbour or unsplit j or i neighbour
      if not grid.pinched_out(cell_kji0 = kji0, cache_pinchout_array = True) and not faulted:
         return (True, kji0, axis, 1 - polarity, segment, seg_fraction, xyz)
      if faulted:
         # look for intersections with column face
         xyz_f, k0 = find_intersection_of_trajectory_interval_with_column_face(trajectory,
                                                                               grid,
                                                                               segment,
                                                                               kji0[1:],
                                                                               axis,
                                                                               1 - polarity,
                                                                               start_xyz = xyz,
                                                                               nudge = -0.1,
                                                                               quad_triangles = True)
         if xyz_f is not None and k0 is not None:
            kji0[0] = k0
            seg_fraction = segment_fraction(trajectory.control_points, segment, xyz_f)
            return (vec.isclose(xyz, xyz_f, tolerance = 0.001), kji0, axis, 1 - polarity, segment, seg_fraction, xyz_f)
         log.debug('failed to find entry point in column face after crossing fault; checking entire cross section')
         x_sect_surf = generate_torn_surface_for_x_section(grid,
                                                           'KJI'[axis],
                                                           ref_slice0 = kji0[axis],
                                                           plus_face = (polarity == 0),
                                                           quad_triangles = True,
                                                           as_single_layer = False)
         xyz_f, segment_f, tri_index_f = find_first_intersection_of_trajectory_with_surface(trajectory,
                                                                                            x_sect_surf,
                                                                                            start = segment,
                                                                                            start_xyz = xyz,
                                                                                            nudge = -0.1)
         if xyz_f is not None:
            # back out cell info from triangle index; note 'column_from...' is actually x_section cell face
            k0, j_or_i0 = x_sect_surf.column_from_triangle_index(tri_index_f)
            kji0[0] = k0
            kji0[3 - axis] = j_or_i0
            seg_fraction = segment_fraction(trajectory.control_points, segment_f, xyz_f)
            return (vec.isclose(xyz, xyz_f,
                                tolerance = 0.001), kji0, axis, 1 - polarity, segment_f, seg_fraction, xyz_f)
         log.debug(
            f"failed to find entry point in cross section after crossing fault{'' if lazy else '; checking for skin re-entry'}"
         )
         if lazy:
            return None
         skin = grid.skin(use_single_layer_tactics = use_single_layer_tactics)
         # following is problematic due to skewed fault planes
         xyz_r, cell_kji0, axis, polarity, segment = skin.find_first_intersection_of_trajectory(
            trajectory, start = segment, start_xyz = xyz, nudge = -0.1, exclude_kji0 = previous_kji0)
         if xyz_r is None:
            log.warning('no skin re-entry found after exit through fault face')
            return None
         log.debug(f"skin re-entry after fault face exit: kji0: {cell_kji0}; face: {'KJI'[axis]}{'-+'[polarity]}")
         seg_fraction = segment_fraction(trajectory.control_points, segment, xyz_r)
         return (False, cell_kji0, axis, polarity, segment, seg_fraction, xyz_r)
      else:
         # skip pinched out cells
         pinchout_skip_sign = -1 if axis == 0 and polarity == 0 else 1
         while True:
            if not grid.pinched_out(cell_kji0 = kji0, cache_pinchout_array = True):
               break
            kji0[0] += pinchout_skip_sign
            if not (np.all(kji0 >= 0) and np.all(kji0 < grid.extent_kji)) or (grid.k_gaps and axis == 0 and (
               (polarity == 0 and grid.k_gap_after_array[kji0[0]]) or
               (polarity == 1 and grid.k_gap_after_array[kji0[0] - 1]))):
               log.debug(
                  f"trajectory reached edge of model {'or k gap ' if grid.k_gaps and axis == 0 else ''}at exit from cell kji0: {previous_kji0}"
               )
               if lazy:
                  return None
               skin = grid.skin(use_single_layer_tactics = use_single_layer_tactics)
               # nudge in following will be problematic if gap has zero (or tiny) thickness at this location
               xyz_r, cell_kji0, axis, polarity, segment = skin.find_first_intersection_of_trajectory(trajectory,
                                                                                                      start = segment,
                                                                                                      start_xyz = xyz,
                                                                                                      nudge = +0.01)
               if xyz_r is None:
                  return None  # no re-entry found after exit through skin
               seg_fraction = segment_fraction(trajectory.control_points, segment, xyz_r)
               return (False, cell_kji0, axis, polarity, segment, seg_fraction, xyz_r)
         return (True, kji0, axis, 1 - polarity, segment, seg_fraction, xyz)

   def segment_fraction(trajectory_xyz, segment, xyz):
      # returns fraction of way along segment that point xyz lies
      segment_vector = trajectory_xyz[segment + 1] - trajectory_xyz[segment]
      segment_length = vec.naive_length(segment_vector)  # note this is length in straight line, rather than diff in mds
      return vec.naive_length(xyz - trajectory_xyz[segment]) / segment_length

   def back_calculated_md(trajectory, segment, fraction):
      base_md = trajectory.measured_depths[segment]
      return base_md + fraction * (trajectory.measured_depths[segment + 1] - base_md)

   assert isinstance(blocked_well, rqw.BlockedWell)
   assert isinstance(blocked_well.trajectory, rqw.Trajectory)
   assert grid is not None

   log.debug(f'blocking well {rqw.well_name(blocked_well)} from trajectory and grid')

   if grid.k_gaps:
      use_single_layer_tactics = False
      log.debug('skin single layer tactics disabled')

   grid_crs = rqc.Crs(grid.model, uuid = grid.crs_uuid)
   if bu.matching_uuids(rqet.uuid_for_part_root(blocked_well.trajectory.crs_root), grid.crs_uuid):
      trajectory = blocked_well.trajectory
   else:
      # create a temporary orphanage model (in memory only) to host a copy of the trajectory for crs alignment
      # NB. temporary objects, relationships left in a mess
      model = rq.Model(create_basics = True)
      trajectory = rqw.Trajectory(model,
                                  blocked_well.trajectory.root_node,
                                  hdf5_source_model = blocked_well.trajectory.model)
      assert trajectory is not None
      traj_crs = rqc.Crs(blocked_well.model, uuid = rqet.uuid_for_part_root(blocked_well.trajectory.crs_root))
      traj_crs.convert_array_to(grid_crs, trajectory.control_points)  # trajectory xyz converted in situ to grid's crs
      trajectory.crs_root = model.duplicate_node(grid.crs_root)
      # note: any represented interpretation object will not be present in the temporary model
   traj_xyz = trajectory.control_points

   if not trajectory_grid_overlap(trajectory, grid):
      log.error('no overlap of trajectory with grid for trajectory uuid: ' + str(trajectory.uuid))
      return None
   grid_box = grid.xyz_box(lazy = False)
   if grid_crs.z_inc_down:
      z_sign = 1.0
      grid_top_z = grid_box[0, 2]
   else:
      z_sign = -1.0
      grid_top_z = -grid_box[1, 2]  # maximum is least negative, ie. shallowest

   assert z_sign * traj_xyz[0, 2] < grid_top_z, 'trajectory does not start above top of grid'  # min z
   if z_sign * traj_xyz[-1, 2] < grid_top_z:
      log.warning('end of trajectory (TD) is above top of grid')

   # scan down wellbore till top of grid reached
   knot = 0
   while knot < trajectory.knot_count - 1 and z_sign * traj_xyz[knot + 1, 2] < grid_top_z:
      knot += 1
   log.debug('skipped trajectory to knot: ' + str(knot))
   if knot == trajectory.knot_count - 1:
      return None  # entire well is above grid

   if lazy:

      skin = None
      # search for intersection with top of grid: will fail if well comes in from the side (or from below!) or penetrates through fault
      xyz, entry_knot, col_ji0 = find_first_intersection_of_trajectory_with_layer_interface(
         trajectory,
         grid,
         k0 = 0,
         ref_k_faces = 'top',
         start = knot,
         heal_faults = False,
         quad_triangles = quad_triangles)
      log.debug('top intersection x,y,z: ' + str(xyz) + '; knot: ' + str(entry_knot) + '; col j0,i0: ' +
                str(col_ji0[0]) + ', ' + str(col_ji0[1]))
      if xyz is None:
         log.error('failed to find intersection of trajectory with top surface of grid')
         return None
      cell_kji0 = np.array((0, col_ji0[0], col_ji0[1]), dtype = int)
      axis = 0
      polarity = 0

   else:  # not lazy

      # note: xyz and entry_fraction might be slightly off when penetrating a skewed fault plane – deemed immaterial for real cases
      skin = grid.skin(use_single_layer_tactics = use_single_layer_tactics)
      xyz, cell_kji0, axis, polarity, entry_knot = skin.find_first_intersection_of_trajectory(trajectory)
      if xyz is None:
         log.error('failed to find intersection of trajectory with outer skin of grid')
         return None
      else:
         log.debug('skin intersection x,y,z: ' + str(xyz) + '; knot: ' + str(entry_knot) + '; cell kji0: ' +
                   str(cell_kji0) + '; face: ' + 'KJI'[axis] + '-+'[polarity])
         cell_kji0 = np.array(cell_kji0, dtype = int)

   previous_kji0 = cell_kji0.copy()
   shift = polarity * 2 - 1
   previous_kji0[axis] += shift  # note: previous may legitimately be 'beyond' edge of grid
   entry_fraction = segment_fraction(traj_xyz, entry_knot, xyz)
   log.debug(f'initial previous kji0: {previous_kji0}')
   next_cell_info = find_next_cell(grid,
                                   previous_kji0,
                                   axis,
                                   1 - polarity,
                                   trajectory,
                                   entry_knot,
                                   entry_fraction,
                                   xyz,
                                   treat_skin_as_fault = use_single_layer_tactics,
                                   lazy = lazy,
                                   use_single_layer_tactics = use_single_layer_tactics)
   log.debug(f'initial next cell info: {next_cell_info}')
   node_mds_list = [back_calculated_md(trajectory, entry_knot, entry_fraction)]
   node_count = 1
   grid_indices_list = []
   cell_count = 0
   cell_indices_list = []
   face_pairs_list = []
   kissed = np.zeros(grid.extent_kji, dtype = bool)
   sample_test = np.zeros(grid.extent_kji, dtype = bool)

   while next_cell_info is not None:

      (entry_shared, kji0, entry_axis, entry_polarity, entry_knot, entry_fraction, entry_xyz) = next_cell_info

      log.debug('next cell entry x,y,z: ' + str(entry_xyz) + '; knot: ' + str(entry_knot) + '; cell kji0: ' +
                str(kji0) + '; face: ' + 'KJI'[entry_axis] + '-+'[entry_polarity])

      if not entry_shared:
         log.debug('adding unblocked interval')
         node_mds_list.append(back_calculated_md(trajectory, entry_knot, entry_fraction))
         node_count += 1
         grid_indices_list.append(-1)
         kissed[:] = False
         sample_test[:] = False

      exit_xyz, exit_knot, exit_axis, exit_polarity = find_first_intersection_of_trajectory_with_cell_surface(
         trajectory, grid, kji0, entry_knot, start_xyz = entry_xyz, nudge = 0.01, quad_triangles = True)

      if exit_xyz is None:
         log.debug('no exit')
      else:
         log.debug('cell exit x,y,z: ' + str(exit_xyz) + '; knot: ' + str(exit_knot) + '; face: ' + 'KJI'[exit_axis] +
                   '-+'[exit_polarity])

      if exit_xyz is None:

         if point_is_within_cell(traj_xyz[-1], grid, kji0):
            # well terminates within cell: add termination blocked interval (or unblocked if inactive cell)
            log.debug(f'adding termination interval for cell {kji0}')
            # todo: check for inactive cell and make an unblocked interval instead, if required
            node_mds_list.append(trajectory.measured_depths[-1])
            node_count += 1
            grid_indices_list.append(0)
            cell_indices_list.append(grid.natural_cell_index(kji0))
            cell_count += 1
            face_pairs_list.append(((entry_axis, entry_polarity), (-1, -1)))
            break

         else:
            next_cell_info = None
            # trajectory has kissed corner or edge of cell and nudge has gone outside cell
            # nudge forward and look for point inclusion in possible neighbours
            log.debug(f'kiss detected at cell kji0 {kji0} ' + 'KJI'[entry_axis] + '-+'[entry_polarity])
            kissed[tuple(kji0)] = True
            if np.all(np.array(previous_kji0, dtype = int) >= 0) and np.all(
                  np.array(previous_kji0, dtype = int) < grid.extent_kji):
               kissed[tuple(previous_kji0)] = True  # stops immediate revisit for kiss and tell tactics
            # setup kji offsets of neighbouring cells likely to contain sample point (could check for split column faces)
            axis_a = (entry_axis + 1) % 3
            axis_b = (entry_axis + 2) % 3
            offsets_kji = np.zeros((18, 3), dtype = int)
            offsets_kji[0, axis_a] = -1
            offsets_kji[1, axis_a] = 1
            offsets_kji[2, axis_b] = -1
            offsets_kji[3, axis_b] = 1
            offsets_kji[4, axis_a] = offsets_kji[4, axis_b] = -1
            offsets_kji[5, axis_a] = -1
            offsets_kji[5, axis_b] = 1
            offsets_kji[6, axis_a] = 1
            offsets_kji[6, axis_b] = -1
            offsets_kji[7, axis_a] = offsets_kji[7, axis_b] = 1
            offsets_kji[8:16] = offsets_kji[:8]
            offsets_kji[8:16, axis] = previous_kji0[axis] - kji0[axis]
            pinchout_skip_sign = -1 if entry_axis == 0 and entry_polarity == 1 else 1
            offsets_kji[16, axis] = pinchout_skip_sign
            log.debug(f'kiss pinchout skip sign: {pinchout_skip_sign}')
            for try_index in range(len(offsets_kji)):
               try_kji0 = kji0 + offsets_kji[try_index]
               while np.all(try_kji0 >= 0) and np.all(try_kji0 < grid.extent_kji) and grid.pinched_out(
                     cell_kji0 = try_kji0):
                  try_kji0[0] += pinchout_skip_sign
               if np.any(try_kji0 < 0) or np.any(try_kji0 >= grid.extent_kji) or kissed[tuple(try_kji0)]:
                  continue
               # use tiny negative nudge to look for entry into try cell with current segment; check that entry point is close
               try_entry_xyz, try_entry_knot, try_entry_axis, try_entry_polarity = find_first_intersection_of_trajectory_with_cell_surface(
                  trajectory, grid, try_kji0, entry_knot, start_xyz = entry_xyz, nudge = -0.01, quad_triangles = True)
               if try_entry_xyz is not None and vec.isclose(try_entry_xyz, entry_xyz, tolerance = 0.02):
                  log.debug(f'try accepted for cell: {try_kji0}')
                  try_entry_fraction = segment_fraction(traj_xyz, try_entry_knot, try_entry_xyz)
                  # replace the next cell entry info
                  next_cell_info = (True, try_kji0, try_entry_axis, try_entry_polarity, try_entry_knot,
                                    try_entry_fraction, try_entry_xyz)
                  break
            if next_cell_info is None:
               log.debug('kiss and tell failed to find next cell entry; switching to point in cell tactics')
               # take a sample point a little ahead on the trajectory
               remaining = traj_xyz[entry_knot + 1] - entry_xyz
               remaining_length = vec.naive_length(remaining)
               if remaining_length < 0.025:
                  log.debug(f'not much remaining of segnemt: {remaining_length}')
               nudge = 0.025
               if remaining_length <= nudge:
                  sample_point = entry_xyz + 0.9 * remaining
               else:
                  sample_point = entry_xyz + nudge * vec.unit_vector(remaining)
               sample_test[:] = False
               log.debug(f'sample point is {sample_point}')
               for try_index in range(len(offsets_kji)):
                  try_kji0 = np.array(kji0, dtype = int) + offsets_kji[try_index]
                  while np.all(try_kji0 >= 0) and np.all(try_kji0 < grid.extent_kji) and grid.pinched_out(
                        cell_kji0 = try_kji0):
                     try_kji0[0] += pinchout_skip_sign
                  if np.any(try_kji0 < 0) or np.any(try_kji0 >= grid.extent_kji):
                     continue
                  #                 log.debug(f'tentatively trying: {try_kji0}')
                  sample_test[tuple(try_kji0)] = True
                  if point_is_within_cell(sample_point, grid, try_kji0):
                     log.debug(f'sample point is in cell {try_kji0}')
                     try_entry_xyz, try_entry_knot, try_entry_axis, try_entry_polarity = find_first_intersection_of_trajectory_with_cell_surface(
                        trajectory,
                        grid,
                        try_kji0,
                        entry_knot,
                        start_xyz = entry_xyz,
                        nudge = -0.01,
                        quad_triangles = True)
                     if try_entry_xyz is None or try_entry_knot is None:
                        log.warning(f'failed to find entry to favoured cell {try_kji0}')
                     else:
                        if np.all(try_kji0 == kji0):
                           log.debug(f'staying in cell {kji0} after kiss')
                           # TODO: sort things out to discard kiss completely
                        try_entry_fraction = segment_fraction(traj_xyz, try_entry_knot, try_entry_xyz)
                        next_cell_info = (vec.isclose(try_entry_xyz, entry_xyz,
                                                      tolerance = 0.01), try_kji0, try_entry_axis, try_entry_polarity,
                                          try_entry_knot, try_entry_fraction, try_entry_xyz)
                     break
               if next_cell_info is None:
                  log.debug('sample point not found in immediate neighbours, switching to full column search')
                  k_offset = 0
                  found = False
                  while not found and k_offset <= max(kji0[0], grid.nk - kji0[0] - 1):
                     for k_plus_minus in [1, -1]:
                        for j_offset in [0, -1, 1]:
                           for i_offset in [0, -1, 1]:
                              try_kji0 = kji0 + np.array((k_offset * k_plus_minus, j_offset, i_offset))
                              if np.any(try_kji0 < 0) or np.any(
                                    try_kji0 >= grid.extent_kji) or sample_test[tuple(try_kji0)]:
                                 continue
                              sample_test[tuple(try_kji0)] = True
                              if point_is_within_cell(sample_point, grid, try_kji0):
                                 found = True
                                 log.debug(f'sample point is in cell {try_kji0}')
                                 try_entry_xyz, try_entry_knot, try_entry_axis, try_entry_polarity =  \
                                    find_first_intersection_of_trajectory_with_cell_surface(
                                       trajectory, grid, try_kji0, entry_knot, start_xyz = entry_xyz, nudge = -0.01, quad_triangles = True)
                                 if try_entry_xyz is None or try_entry_knot is None:
                                    log.warning(f'failed to find entry to column sampled cell {try_kji0}')
                                 else:
                                    try_entry_fraction = segment_fraction(traj_xyz, try_entry_knot, try_entry_xyz)
                                    next_cell_info = (vec.isclose(try_entry_xyz, entry_xyz, tolerance = 0.01), try_kji0,
                                                      try_entry_axis, try_entry_polarity, try_entry_knot,
                                                      try_entry_fraction, try_entry_xyz)
                                 break
                           if found:
                              break
                        if found:
                           break
                     k_offset += 1
               if next_cell_info is None:
                  log.debug('no success during full column search')
                  if np.any(previous_kji0 == 0) or np.any(previous_kji0 == grid.extent_kji - 1):
                     log.debug('looking for skin re-entry after possible kissing exit')
                     skin = grid.skin(use_single_layer_tactics = use_single_layer_tactics)
                     try_entry_xyz, try_kji0, try_entry_axis, try_entry_polarity, try_entry_knot =  \
                        skin.find_first_intersection_of_trajectory(trajectory, start = entry_knot, start_xyz = entry_xyz, nudge = +0.1)
                     if try_entry_xyz is not None:
                        log.debug('skin re-entry found')
                        try_entry_fraction = segment_fraction(traj_xyz, try_entry_knot, try_entry_xyz)
                        next_cell_info = (vec.isclose(try_entry_xyz, entry_xyz,
                                                      tolerance = 0.01), try_kji0, try_entry_axis, try_entry_polarity,
                                          try_entry_knot, try_entry_fraction, try_entry_xyz)

         if next_cell_info is None:
            log.warning('well blocking got stuck – cells probably omitted at tail of well')


#      if exit_xyz is not None:  # usual well-behaved case or non-standard due to kiss
      else:

         exit_fraction = segment_fraction(traj_xyz, exit_knot, exit_xyz)

         log.debug('adding blocked interval for cell kji0: ' + str(kji0))
         # todo: check for inactive cell and make an unblocked interval instead, if required
         node_mds_list.append(back_calculated_md(trajectory, exit_knot, exit_fraction))
         node_count += 1
         grid_indices_list.append(0)
         cell_indices_list.append(grid.natural_cell_index(kji0))
         cell_count += 1
         face_pairs_list.append(((entry_axis, entry_polarity), (exit_axis, exit_polarity)))

         previous_kji0 = kji0
         log.debug(f'previous kji0 set to {previous_kji0}')
         next_cell_info = find_next_cell(grid,
                                         kji0,
                                         exit_axis,
                                         exit_polarity,
                                         trajectory,
                                         exit_knot,
                                         exit_fraction,
                                         exit_xyz,
                                         treat_skin_as_fault = use_single_layer_tactics,
                                         lazy = lazy,
                                         use_single_layer_tactics = use_single_layer_tactics)
         kissed[:] = False
         sample_test[:] = False

   if node_count < 2:
      log.warning('no nodes found during attempt to block well')
      return None

   assert node_count > 1
   assert len(node_mds_list) == node_count
   assert len(grid_indices_list) == node_count - 1
   assert cell_count < node_count
   assert len(cell_indices_list) == cell_count
   assert len(face_pairs_list) == cell_count

   blocked_well.node_mds = np.array(node_mds_list)
   blocked_well.node_count = node_count
   blocked_well.grid_indices = np.array(grid_indices_list, dtype = int)
   blocked_well.cell_indices = np.array(cell_indices_list, dtype = int)
   blocked_well.face_pair_indices = np.array(face_pairs_list, dtype = int)
   blocked_well.cell_count = cell_count
   blocked_well.grid_list = [grid]

   assert cell_count == (node_count - np.count_nonzero(blocked_well.grid_indices == -1) - 1)

   log.info(str(cell_count) + ' cell' + _pl(cell_count) + ' blocked for well trajectory uuid: ' + str(trajectory.uuid))

   return blocked_well


def _pl(n, use_es = False):
   if n == 1:
      return ''
   return 'es' if use_es else 's'
