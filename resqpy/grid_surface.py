"""Functions relating to intsection of resqml grid with surface or trajectory objects."""

import logging

log = logging.getLogger(__name__)

import math as maths
import numpy as np
import numba  # type: ignore
from numba import njit
from typing import Tuple, Optional, Dict
import warnings

import resqpy.crs as rqc
import resqpy.fault as rqf
import resqpy.grid as grr
import resqpy.model as rq
import resqpy.olio.box_utilities as bx
import resqpy.olio.intersection as meet
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp
import resqpy.surface as rqs
import resqpy.well as rqw
import resqpy.weights_and_measures as wam


class GridSkin:
    """Class of object consisting of outer skin of grid (not a RESQML class in its own right)."""

    def __init__(self, grid, quad_triangles = True, use_single_layer_tactics = True, is_regular = False):
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
        self.is_regular = is_regular  #: indicates a simplified skin for a regular aligned grid

        k_gap_surf_list = self._make_k_gap_surfaces(quad_triangles = quad_triangles)

        if self.is_regular:

            # build a simplified two triangle surface for each of the six skin surfaces
            xyz_box = grid.xyz_box(local = True)
            for axis in range(3):
                if grid.block_dxyz_dkji[2 - axis, axis] < 0.0:
                    xyz_box[:, axis] = (xyz_box[1, axis], xyz_box[0, axis])
            min_x, min_y, min_z = xyz_box[0]
            max_x, max_y, max_z = xyz_box[1]
            top_surf = rqs.Surface(grid.model)
            top_surf.set_to_horizontal_plane(0.0, xyz_box)
            base_surf = rqs.Surface(grid.model)
            base_surf.set_to_horizontal_plane(grid.nk * grid.block_dxyz_dkji[0, 2], xyz_box)
            j_minus_surf = rqs.Surface(grid.model)
            corners = np.array([(min_x, min_y, min_z), (max_x, min_y, min_z), (min_x, min_y, max_z),
                                (max_x, min_y, max_z)])
            j_minus_surf.set_to_triangle_pair(corners)
            j_plus_surf = rqs.Surface(grid.model)
            corners = np.array([(min_x, max_y, min_z), (max_x, max_y, min_z), (min_x, max_y, max_z),
                                (max_x, max_y, max_z)])
            j_plus_surf.set_to_triangle_pair(corners)
            i_minus_surf = rqs.Surface(grid.model)
            corners = np.array([(min_x, min_y, min_z), (min_x, max_y, min_z), (min_x, min_y, max_z),
                                (min_x, max_y, max_z)])
            i_minus_surf.set_to_triangle_pair(corners)
            i_plus_surf = rqs.Surface(grid.model)
            corners = np.array([(max_x, min_y, min_z), (max_x, max_y, min_z), (max_x, min_y, max_z),
                                (max_x, max_y, max_z)])
            i_plus_surf.set_to_triangle_pair(corners)
            surf_list = [top_surf, base_surf, j_minus_surf, j_plus_surf, i_minus_surf, i_plus_surf]

        elif self.has_split_coordinate_lines:

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
            self.fault_j_face_cols_ji0 = np.stack((col_j0, col_i0),
                                                  axis = -1)  # fault is on plus j face of these columns
            col_j0, col_i0 = np.where(i_column_faces)
            self.fault_i_face_cols_ji0 = np.stack((col_j0, col_i0),
                                                  axis = -1)  # fault is on plus i face of these columns

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
        """Returns the first intersection of the trajectory with the torn skin.

        Returns the x,y,z and K,J,I and axis, polarity & segment.

        arguments:
           trajectory (well.Trajectory object): the trajectory to be intersected with the skin
           start (int, default 0): the trajectory segment number to start the search from
           start_xyz (triple float, optional): if present, this point should lie on the start segment and search continues from
              this point
           nudge (float, optional): if present and positive, the start point is nudged forward by this distance (grid uom);
              if present and negative (more typical for skin entry search), the start point is nudged back a little
           exclude_kji0 (triple int, optional): if present, the indices of a cell to exclude as a possible result

        returns:
           5-tuple of:

           - (triple float): xyz coordinates of the intersection point in the crs of the grid
           - (triple int): kji0 of the cell that is intersected, which might be a pinched out or otherwise inactive cell
           - (int): 0, 1 or 2 for K, J or I axis of cell face
           - (int): 0 for -ve face, 1 for +ve face
           - (int): trajectory knot prior to the intersection (also the segment number)

        note:
           if the GridSkin object has been initialised using single layer tactics, then the k0 value will be zero for any
           initial entry through a sidewall of the grid or through a fault face
        """

        if exclude_kji0 is not None:
            bundle = find_first_intersection_of_trajectory_with_surface(trajectory,
                                                                        self.skin,
                                                                        start = start,
                                                                        start_xyz = start_xyz,
                                                                        nudge = nudge,
                                                                        return_second = True)
            xyz_1, segment_1, tri_index_1, xyz_2, segment_2, tri_index_2 = bundle
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

        for (try_xyz, segment, tri_index) in triplet_list:

            xyz = try_xyz

            surf_index, surf_tri_index = self.skin.surface_index_for_triangle_index(tri_index)
            assert surf_index is not None

            if self.is_regular:
                assert 0 <= surf_index < 6
                axis, polarity = divmod(surf_index, 2)
                kji0 = np.zeros(3, dtype = int)
                if polarity:
                    kji0[axis] = self.grid.extent_kji[axis] - 1
                if axis == 0:  # K face (top or base)
                    kji0[1] = int(xyz[1] / self.grid.block_dxyz_dkji[1, 1])
                    kji0[2] = int(xyz[0] / self.grid.block_dxyz_dkji[2, 0])
                elif axis == 1:  # J face
                    kji0[0] = int(xyz[2] / self.grid.block_dxyz_dkji[0, 2])
                    kji0[2] = int(xyz[0] / self.grid.block_dxyz_dkji[2, 0])
                else:  # I face
                    kji0[0] = int(xyz[2] / self.grid.block_dxyz_dkji[0, 2])
                    kji0[1] = int(xyz[1] / self.grid.block_dxyz_dkji[1, 1])
                kji0 = tuple(kji0)

            elif surf_index < 6:  # grid skin
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

        # list of layer face surfaces, 2 per gap (top of gap, ie. base of layer above, then base of gap)
        k_gap_surf_list = []
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
        # yaml: disable
        for j in range(1, mesh.shape[0] + 1):
            skirted_mesh[j, 0, :] =  \
                skirted_mesh[j, 1] + border * vec.unit_vector(skirted_mesh[j, 1] - skirted_mesh[j, 2])
            skirted_mesh[j, -1, :] =  \
                skirted_mesh[j, -2] + border * vec.unit_vector(skirted_mesh[j, -2] - skirted_mesh[j, -3])
        for i in range(1, mesh.shape[1] + 1):
            skirted_mesh[0, i, :] =  \
                skirted_mesh[1, i] + border * vec.unit_vector(skirted_mesh[1, i] - skirted_mesh[2, i])
            skirted_mesh[-1, i, :] =  \
                skirted_mesh[-2, i] + border * vec.unit_vector(skirted_mesh[-2, i] - skirted_mesh[-3, i])
        # yaml: enable
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

    x_sect_points = grid.unsplit_x_section_points(axis, ref_slice0 = ref_slice0, plus_face = plus_face)
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
    """Returns xyz and other info of the first intersection of well trajectory with surface.

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
                                                               quad_triangles = True,
                                                               is_regular = False):
    """Returns info about the first intersection of well trajectory(s) with layer interface.

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
       is_regular (boolean, default False): set True if grid is a RegularGrid with IJK axes aligned with xyz axes

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

    # log.debug('generating surface for layer: ' + str(k0) + ' ' + ref_k_faces)
    if is_regular:
        interface_depth = grid.block_dxyz_dkji[0, 2] * (k0 + (1 if ref_k_faces == 'base' else 0))
        surface = rqs.Surface(grid.model)
        surface.set_to_horizontal_plane(interface_depth, grid.xyz_box())
    elif grid.has_split_coordinate_lines and not heal_faults:
        surface = generate_torn_surface_for_layer_interface(grid,
                                                            k0 = k0,
                                                            ref_k_faces = ref_k_faces,
                                                            quad_triangles = quad_triangles)
    else:
        surface = generate_untorn_surface_for_layer_interface(grid,
                                                              k0 = k0,
                                                              ref_k_faces = ref_k_faces,
                                                              quad_triangles = quad_triangles)

    # log.debug('finding intersections of wellbore trajectory(ies) with layer: ' + str(k0) + ' ' + ref_k_faces)
    results_list = []
    segment_list = []
    col_list = []
    for traj in trajectory_list:
        xyz, knot, tri = find_first_intersection_of_trajectory_with_surface(traj, surface, start = start)
        if is_regular:
            col_j = int(xyz[1] / grid.block_dxyz_dkji[1, 1])
            col_i = int(xyz[1] / grid.block_dxyz_dkji[2, 0])
            assert 0 <= col_i < grid.ni and 0 <= col_j < grid.nj
            col = (col_j, col_i)
        else:
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
    """Return first intersection with cell's surface found along a trajectory."""

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
            # log.debug('intersection xyz: ' + str(xyz) + '; tri: ' + str(tri) + '; axis: ' + str(axis) + '; polarity: ' +
            #           str(polarity))
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


def find_faces_to_represent_surface_staffa(grid, surface, name, feature_type = 'fault', progress_fn = None):
    """Returns a grid connection set containing those cell faces which are deemed to represent the surface."""

    if progress_fn is not None:
        progress_fn(0.0)
    # log.debug('computing cell centres')
    centre_points = grid.centre_point()
    # log.debug('computing inter cell centre vectors and boxes')
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

    # log.debug('finding surface triangle boxes')
    t, p = surface.triangles_and_points()
    if not bu.matching_uuids(grid.crs_uuid, surface.crs_uuid):
        log.debug('converting from surface crs to grid crs')
        s_crs = rqc.Crs(surface.model, uuid = surface.crs_uuid)
        s_crs.convert_array_to(grid.crs, p)
    triangles = p[t]
    assert triangles.size > 0, 'no triangles in surface'
    triangle_boxes = np.empty((triangles.shape[0], 2, 3))
    triangle_boxes[:, 0, :] = np.amin(triangles, axis = 1)
    triangle_boxes[:, 1, :] = np.amax(triangles, axis = 1)

    grid_box = grid.xyz_box(lazy = False, local = True)

    # log.debug('looking for cell faces for each triangle')
    batch_size = 1000
    triangle_count = triangles.shape[0]
    progress_batch = min(1.0, float(batch_size) / float(triangle_count))
    progress_base = 0.0
    ti_base = 0
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
        # log.debug('triangles processed: ' + str(ti_base))
        # log.debug('interim face counts: K: ' + str(np.count_nonzero(k_faces)) +
        #                              '; J: ' + str(np.count_nonzero(j_faces)) +
        #                              '; I: ' + str(np.count_nonzero(i_faces)))
        progress_base = min(1.0, progress_base + progress_batch)

    # log.debug('face counts: K: ' + str(np.count_nonzero(k_faces)) +
    #                      '; J: ' + str(np.count_nonzero(j_faces)) +
    #                      '; I: ' + str(np.count_nonzero(i_faces)))
    gcs = rqf.GridConnectionSet(grid.model,
                                grid = grid,
                                k_faces = k_faces,
                                j_faces = j_faces,
                                i_faces = i_faces,
                                feature_name = name,
                                feature_type = feature_type,
                                create_organizing_objects_where_needed = True)

    if progress_fn is not None:
        progress_fn(1.0)

    return gcs


def find_faces_to_represent_surface_regular(grid,
                                            surface,
                                            name,
                                            title = None,
                                            centres = None,
                                            agitate = False,
                                            feature_type = 'fault',
                                            progress_fn = None,
                                            consistent_side = False,
                                            return_properties = None):
    """Returns a grid connection set containing those cell faces which are deemed to represent the surface.

    arguments:
        grid (RegularGrid): the grid for which to create a grid connection set representation of the surface
        surface (Surface): the surface to be intersected with the grid
        name (str): the feature name to use in the grid connection set
        title (str, optional): the citation title to use for the grid connection set; defaults to name
        centres (numpy float array of shape (nk, nj, ni, 3), optional): precomputed cell centre points in
           local grid space, to avoid possible crs issues; required if grid's crs includes an origin (offset)?
        agitate (bool, default False): if True, the points of the surface are perturbed by a small random
           offset, which can help if the surface has been built from a regular mesh with a periodic resonance
           with the grid
        feature_type (str, default 'fault'): 'fault', 'horizon' or 'geobody boundary'
        progress_fn (f(x: float), optional): a callback function to be called at intervals by this function;
           the argument will progress from 0.0 to 1.0 in unspecified and uneven increments
        consistent_side (bool, default False): if True, the cell pairs will be ordered so that all the first
           cells in each pair are on one side of the surface, and all the second cells on the other
        return_properties (list of str, optional): if present, a list of property arrays to calculate and
           return as a dictionary; recognised values in the list are 'offset' and 'normal vector'; offset
           is a measure of the distance between the centre of the cell face and the intersection point of the
           inter-cell centre vector with a triangle in the surface; normal vector is a unit vector normal
           to the surface triangle; each array has an entry for each face in the gcs; the returned dictionary
           has the passed strings as keys and numpy arrays as values

    returns:
        gcs  or  (gcs, dict)
        where gcs is a new GridConnectionSet with a single feature, not yet written to hdf5 nor xml created;
        dict is a dictionary mapping from property name to numpy array; 'offset' will map to a numpy float
        array of shape (gcs.count, ); 'normal vector' will map to a numpy float array of shape (gcs.count, 3)
        holding a unit vector normal to the surface for each of the faces in the gcs; the dict is only
        returned if a non-empty list has been passed as return_properties

    notes:
        this function can handle the surface and grid being in different coordinate reference systems, as
        long as the implicit parent crs is shared; no trimming of the surface is carried out here: for
        computational efficiency, it is recommended to trim first;
        organisational objects for the feature are created if needed;
        if grid has differing xy & z units, this is accounted for here when generating normal vectors, ie.
        true normal unit vectors are returned
    """

    assert isinstance(grid, grr.RegularGrid)
    assert grid.is_aligned
    return_normal_vectors = False
    return_offsets = False
    if return_properties:
        assert all([p in ['offset', 'normal vector'] for p in return_properties])
        return_normal_vectors = ('normal vector' in return_properties)
        return_offsets = ('offset' in return_properties)

    if title is None:
        title = name

    if progress_fn is not None:
        progress_fn(0.0)

    log.debug(f'intersecting surface {surface.title} with regular grid {grid.title}')
    log.debug(f'grid extent kji: {grid.extent_kji}')

    grid_dxyz = (grid.block_dxyz_dkji[2, 0], grid.block_dxyz_dkji[1, 1], grid.block_dxyz_dkji[0, 2])
    if centres is None:
        centres = grid.centre_point(use_origin = True)
    if consistent_side:
        log.debug('making all triangles clockwise')
        # note: following will shuffle order of vertices within t cached in surface
        surface.make_all_clockwise_xy(reorient = True)
    t, p = surface.triangles_and_points()
    assert t is not None and p is not None, f'surface {surface.title} is empty'
    if agitate:
        p += 1.0e-5 * (np.random.random(p.shape) - 0.5)
    log.debug(f'surface: {surface.title}; p0: {p[0]}; crs uuid: {surface.crs_uuid}')
    log.debug(f'surface min xyz: {np.min(p, axis = 0)}')
    log.debug(f'surface max xyz: {np.max(p, axis = 0)}')
    if not bu.matching_uuids(grid.crs_uuid, surface.crs_uuid):
        log.debug('converting from surface crs to grid crs')
        s_crs = rqc.Crs(surface.model, uuid = surface.crs_uuid)
        s_crs.convert_array_to(grid.crs, p)
        surface.crs_uuid = grid.crs.uuid
        log.debug(f'surface: {surface.title}; p0: {p[0]}; crs uuid: {surface.crs_uuid}')
        log.debug(f'surface min xyz: {np.min(p, axis = 0)}')
        log.debug(f'surface max xyz: {np.max(p, axis = 0)}')

    log.debug(f'centres min xyz: {np.min(centres.reshape((-1, 3)), axis = 0)}')
    log.debug(f'centres max xyz: {np.max(centres.reshape((-1, 3)), axis = 0)}')

    t_count = len(t)

    # todo: batch up either centres or triangles to reduce memory requirement for large models

    # K direction (xy projection)
    if grid.nk > 1:
        log.debug('searching for k faces')
        k_faces = np.zeros((grid.nk - 1, grid.nj, grid.ni), dtype = bool)
        k_sides = np.zeros((grid.nk - 1, grid.nj, grid.ni), dtype = bool)
        k_offsets = np.full((grid.nk - 1, grid.nj, grid.ni), np.nan) if return_offsets else None
        k_normals = np.full((grid.nk - 1, grid.nj, grid.ni, 3), np.nan) if return_normal_vectors else None
        k_centres = centres[0, :, :].reshape((-1, 3))
        k_hits = vec.points_in_triangles(p, t, k_centres, projection = 'xy', edged = True).reshape(
            (t_count, grid.nj, grid.ni))
        del k_centres
        if consistent_side:
            cwt = (vec.clockwise_triangles(p, t, projection = 'xy') >= 0.0)
        for k_t, k_j, k_i in np.stack(np.where(k_hits), axis = -1):
            xyz = meet.line_triangle_intersect(centres[0, k_j, k_i],
                                               centres[-1, k_j, k_i] - centres[0, k_j, k_i],
                                               p[t[k_t]],
                                               line_segment = True,
                                               t_tol = 1.0e-6)
            if xyz is None:  # meeting point is outwith grid
                continue
            k_face = int((xyz[2] - centres[0, k_j, k_i, 2]) / grid_dxyz[2])
            if k_face == -1:  # handle rounding precision issues
                k_face = 0
            elif k_face == grid.nk - 1:
                k_face -= 1
            assert 0 <= k_face < grid.nk - 1
            k_faces[k_face, k_j, k_i] = True
            if consistent_side:
                k_sides[k_face, k_j, k_i] = cwt[k_t]
            if return_offsets:
                # compute offset as z diff between xyz and face
                k_offsets[k_face, k_j, k_i] =  \
                    xyz[2] - 0.5 * (centres[k_face, k_j, k_i, 2] + centres[k_face + 1, k_j, k_i, 2])
            if return_normal_vectors:
                k_normals[k_face, k_j, k_i] = vec.triangle_normal_vector(p[t[k_t]])
                # todo: if consistent side, could deliver information about horizon surface inversion
                if k_normals[k_face, k_j, k_i, 2] > 0.0:
                    k_normals[k_face, k_j, k_i] = -k_normals[k_face, k_j, k_i]  # -ve z hemisphere normal
        del k_hits
        log.debug(f'k face count: {np.count_nonzero(k_faces)}')
    else:
        k_faces = None
        k_sides = None
        k_normals = None

    if progress_fn is not None:
        progress_fn(0.3)

    # J direction (xz projection)
    if grid.nj > 1:
        log.debug('searching for j faces')
        j_faces = np.zeros((grid.nk, grid.nj - 1, grid.ni), dtype = bool)
        j_sides = np.zeros((grid.nk, grid.nj - 1, grid.ni), dtype = bool)
        j_offsets = np.full((grid.nk, grid.nj - 1, grid.ni), np.nan) if return_offsets else None
        j_normals = np.full((grid.nk, grid.nj - 1, grid.ni, 3), np.nan) if return_normal_vectors else None
        j_centres = centres[:, 0, :].reshape((-1, 3))
        j_hits = vec.points_in_triangles(p, t, j_centres, projection = 'xz', edged = True).reshape(
            (t_count, grid.nk, grid.ni))
        del j_centres
        if consistent_side:
            cwt = (vec.clockwise_triangles(p, t, projection = 'xz') >= 0.0)
        for j_t, j_k, j_i in np.stack(np.where(j_hits), axis = -1):
            xyz = meet.line_triangle_intersect(centres[j_k, 0, j_i],
                                               centres[j_k, -1, j_i] - centres[j_k, 0, j_i],
                                               p[t[j_t]],
                                               line_segment = True,
                                               t_tol = 1.0e-6)
            if xyz is None:  # meeting point is outwith grid
                continue
            j_face = int((xyz[1] - centres[j_k, 0, j_i, 1]) / grid_dxyz[1])
            if j_face == -1:  # handle rounding precision issues
                j_face = 0
            elif j_face == grid.nj - 1:
                j_face -= 1
            assert 0 <= j_face < grid.nj - 1
            j_faces[j_k, j_face, j_i] = True
            if consistent_side:
                j_sides[j_k, j_face, j_i] = cwt[j_t]
            if return_offsets:
                # compute offset as y diff between xyz and face
                j_offsets[j_k, j_face, j_i] =  \
                    xyz[1] - 0.5 * (centres[j_k, j_face, j_i, 1] + centres[j_k, j_face + 1, j_i, 1])
            if return_normal_vectors:
                j_normals[j_k, j_face, j_i] = vec.triangle_normal_vector(p[t[j_t]])
                if j_normals[j_k, j_face, j_i, 2] > 0.0:
                    j_normals[j_k, j_face, j_i] = -j_normals[j_k, j_face, j_i]  # -ve z hemisphere normal
        del j_hits
        log.debug(f'j face count: {np.count_nonzero(j_faces)}')
    else:
        j_faces = None
        j_sides = None
        j_normals = None

    if progress_fn is not None:
        progress_fn(0.6)

    # I direction (yz projection)
    if grid.ni > 1:
        log.debug('searching for i faces')
        i_faces = np.zeros((grid.nk, grid.nj, grid.ni - 1), dtype = bool)
        i_sides = np.zeros((grid.nk, grid.nj, grid.ni - 1), dtype = bool)
        i_offsets = np.full((grid.nk, grid.nj, grid.ni - 1), np.nan) if return_offsets else None
        i_normals = np.full((grid.nk, grid.nj, grid.ni - 1, 3), np.nan) if return_normal_vectors else None
        i_centres = centres[:, :, 0].reshape((-1, 3))
        i_hits = vec.points_in_triangles(p, t, i_centres, projection = 'yz', edged = True).reshape(
            (t_count, grid.nk, grid.nj))
        del i_centres
        if consistent_side:
            cwt = (vec.clockwise_triangles(p, t, projection = 'yz') >= 0.0)
        for i_t, i_k, i_j in np.stack(np.where(i_hits), axis = -1):
            xyz = meet.line_triangle_intersect(centres[i_k, i_j, 0],
                                               centres[i_k, i_j, -1] - centres[i_k, i_j, 0],
                                               p[t[i_t]],
                                               line_segment = True,
                                               t_tol = 1.0e-6)
            if xyz is None:  # meeting point is outwith grid
                continue
            i_face = int((xyz[0] - centres[i_k, i_j, 0, 0]) / grid_dxyz[0])
            if i_face == -1:  # handle rounding precision issues
                i_face = 0
            elif i_face == grid.ni - 1:
                i_face -= 1
            assert 0 <= i_face < grid.ni - 1
            i_faces[i_k, i_j, i_face] = True
            if consistent_side:
                i_sides[i_k, i_j, i_face] = cwt[i_t]
            if return_offsets:
                # compute offset as x diff between xyz and face
                i_offsets[i_k, i_j, i_face] =  \
                    xyz[0] - 0.5 * (centres[i_k, i_j, i_face, 0] + centres[i_k, i_j, i_face + 1, 0])
            if return_normal_vectors:
                i_normals[i_k, i_j, i_face] = vec.triangle_normal_vector(p[t[i_t]])
                if i_normals[i_k, i_j, i_face, 2] > 0.0:
                    i_normals[i_k, i_j, i_face] = -i_normals[i_k, i_j, i_face]  # -ve z hemisphere normal
        del i_hits
        log.debug(f'i face count: {np.count_nonzero(i_faces)}')
    else:
        i_faces = None
        i_sides = None
        i_normals = None

    if progress_fn is not None:
        progress_fn(0.9)

    if not consistent_side:
        k_sides = None
        j_sides = None
        i_sides = None

    log.debug('converting face sets into grid connection set')
    gcs = rqf.GridConnectionSet(grid.model,
                                grid = grid,
                                k_faces = k_faces,
                                j_faces = j_faces,
                                i_faces = i_faces,
                                k_sides = k_sides,
                                j_sides = j_sides,
                                i_sides = i_sides,
                                feature_name = name,
                                feature_type = feature_type,
                                title = title,
                                create_organizing_objects_where_needed = True)

    # NB. following assumes faces have been added to gcs in a particular order!
    if return_offsets:
        k_offsets_list = np.empty((0,)) if k_offsets is None else k_offsets[np.where(k_faces)]
        j_offsets_list = np.empty((0,)) if j_offsets is None else j_offsets[np.where(j_faces)]
        i_offsets_list = np.empty((0,)) if i_offsets is None else i_offsets[np.where(i_faces)]
        all_offsets = _all_offsets(grid.crs, k_offsets_list, j_offsets_list, i_offsets_list)
        log.debug(f'gcs count: {gcs.count}; all offsets shape: {all_offsets.shape}')
        assert all_offsets.shape == (gcs.count,)

    # NB. following assumes faces have been added to gcs in a particular order!
    if return_normal_vectors:
        k_normals_list = np.empty((0, 3)) if k_normals is None else k_normals[np.where(k_faces)]
        j_normals_list = np.empty((0, 3)) if j_normals is None else j_normals[np.where(j_faces)]
        i_normals_list = np.empty((0, 3)) if i_normals is None else i_normals[np.where(i_faces)]
        all_normals = np.concatenate((k_normals_list, j_normals_list, i_normals_list), axis = 0)
        log.debug(f'gcs count: {gcs.count}; all normals shape: {all_normals.shape}')
        assert all_normals.shape == (gcs.count, 3)
        if grid.crs.xy_units != grid.crs.z_units:
            wam.convert_lengths(all_normals[:, 2], grid.crs.z_units, grid.crs.xy_units)
            all_normals = vec.unit_vectors(all_normals)

    if progress_fn is not None:
        progress_fn(1.0)

    # if returning properties, construct dictionary
    if return_properties:
        props_dict = {}
        if return_offsets:
            props_dict['offset'] = all_offsets
        if return_normal_vectors:
            props_dict['normal vector'] = all_normals
        return (gcs, props_dict)

    return gcs


@njit
def where_true(data: np.ndarray):
    """Jitted NumPy 'where' function to improve performance on subsequent calls."""
    return np.where(data)


@njit
def intersect_numba(axis: int, index1: int, index2: int, hits: np.ndarray, n_axis: int, points: np.ndarray,
                    triangles: np.ndarray, grid_dxyz: Tuple[float], faces: np.ndarray, return_depths: bool,
                    depths: np.ndarray, return_offsets: bool, offsets: np.ndarray, return_triangles: bool,
                    triangle_per_face: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds the faces that intersect the surface in 3D.

    Args:
        axis (int): axis number. Axis i is 0, j is 1, and k is 2.
        index1 (int): the first index. Axis i is 0, j is 0, and k is 1.
        index2 (int): the second index. Axis i is 1, j is 2, and k is 2.
        hits (np.ndarray): boolean array of grid centres that intersected the surface for the given
            axis.
        n_axis (int): number of cells in the axis.
        points (np.ndarray): array of all the surface node points in 3D.
        triangles (np.ndarray): array of all the points indices creating each triangle.
        grid_dxyz (Tuple[float]): tuple of a cell's thickness in each axis.
        faces (np.ndarray): boolean array of each cell face that can represent the surface.
        return_depths (bool): if True, an array of the depths is populated.
        depths (np.ndarray): array of the z values of the
            intersection point of the inter-cell centre vector with a triangle in the surface.
        return_offsets (bool): if True, an array of the offsets is calculated.
        offsets (np.ndarray): array of the distance between the centre of the cell face and the
            intersection point of the inter-cell centre vector with a triangle in the surface.
        return_triangles (bool): if True, an array of triangle indices is returned.

    Returns:
        Tuple containing:

        - faces (np.ndarray): boolean array of each cell face that can represent the surface.
        - offsets (np.ndarray): array of the distance between the centre of the cell face and the
            intersection point of the inter-cell centre vector with a triangle in the surface.
        - triangle_per_face (np.ndarray): array of triangle numbers
    """
    n_faces = faces.shape[2 - axis]
    for i in numba.prange(len(hits)):
        tri, d1, d2 = hits[i]

        # Line start point in 3D which had a projection hit.
        centre_point_start = np.zeros(3, dtype = np.float64) + grid_dxyz[axis] / 2
        centre_point_start[2 - index1] = (d1 + 0.5) * grid_dxyz[2 - index1]
        centre_point_start[2 - index2] = (d2 + 0.5) * grid_dxyz[2 - index2]

        # Line end point in 3D.
        centre_point_end = np.copy(centre_point_start)
        centre_point_end[axis] = grid_dxyz[axis] * (n_axis - 0.5)

        xyz = meet.line_triangle_intersect_numba(
            centre_point_start,
            centre_point_end - centre_point_start,
            points[triangles[tri]],
            line_segment = True,
            t_tol = 1.0e-6,
        )
        if xyz is None:  # meeting point is outwith grid
            continue

        # The face corresponding to the grid and surface intersection at this point.
        face = int((xyz[axis] - centre_point_start[axis]) / grid_dxyz[axis])
        if face == -1:  # handle rounding precision issues
            face = 0
        elif face == n_faces:
            face -= 1
        assert 0 <= face < n_faces

        face_idx = np.zeros(3, dtype = np.int32)
        face_idx[index1] = d1
        face_idx[index2] = d2
        face_idx[2 - axis] = face

        faces[face_idx[0], face_idx[1], face_idx[2]] = True

        if return_depths:
            depths[face_idx[0], face_idx[1], face_idx[2]] = xyz[2]
        if return_offsets:
            offsets[face_idx[0], face_idx[1], face_idx[2]] = xyz[axis] - ((face + 1) * grid_dxyz[axis])
        if return_triangles:
            triangle_per_face[face_idx[0], face_idx[1], face_idx[2]] = tri

    return faces, offsets, triangle_per_face


def bisector_from_faces(grid_extent_kji: Tuple[int, int, int], k_faces: np.ndarray, j_faces: np.ndarray,
                        i_faces: np.ndarray, raw_bisector: bool) -> Tuple[np.ndarray, bool]:
    """Returns a numpy bool array denoting the bisection of the grid by the face sets.

    Args:
        grid_extent_kji (triple int): the shape of the grid
        k_faces, j_faces, i_faces (numpy bool arrays): True where an internal grid face forms part of the
            bisecting surface

    Returns:
        (numpy bool array of shape grid_extent_kji, bool) where the array is set True for cells on one side
        of the face sets deemed to be shallower (more strictly, lower K index on average); set False for cells
        on othe side; the bool value is True if the surface is a curtain (vertical), otherwise False

    Notes:
        the face sets must form a single 'sealed' cut of the grid (eg. not waving in and out of the grid);
        any 'boxed in' parts of the grid (completely enclosed by bisecting faces) will be consistently
        assigned to either the True or False part
    """
    assert len(grid_extent_kji) == 3
    # a = np.zeros(grid_extent_kji, dtype = numba.boolean)  # initialise to False
    # c = np.zeros(grid_extent_kji, dtype = numba.boolean)  # cells changing
    a = np.zeros(grid_extent_kji, dtype = np.bool_)  # initialise to False
    c = np.zeros(grid_extent_kji, dtype = np.bool_)  # cells changing
    open_k = np.logical_not(k_faces)
    open_j = np.logical_not(j_faces)
    open_i = np.logical_not(i_faces)
    # set one or more seeds; todo: more seeds to improve performance if needed
    a[0, 0, 0] = True
    # repeatedly spread True values to neighbouring cells that are not the other side of a face
    # todo: check that following works when a dimension has extent 1
    limit = grid_extent_kji[0] * grid_extent_kji[1] * grid_extent_kji[2]
    for _ in range(limit):
        c[:] = False
        # k faces
        c[1:, :, :] = np.logical_and(a[:-1, :, :], open_k)
        c[:-1, :, :] = np.logical_or(c[:-1, :, :], np.logical_and(a[1:, :, :], open_k))
        # j faces
        c[:, 1:, :] = np.logical_or(c[:, 1:, :], np.logical_and(a[:, :-1, :], open_j))
        c[:, :-1, :] = np.logical_or(c[:, :-1, :], np.logical_and(a[:, 1:, :], open_j))
        # i faces
        c[:, :, 1:] = np.logical_or(c[:, :, 1:], np.logical_and(a[:, :, :-1], open_i))
        c[:, :, :-1] = np.logical_or(c[:, :, :-1], np.logical_and(a[:, :, 1:], open_i))
        c[:] = np.logical_and(c, np.logical_not(a))
        if np.count_nonzero(c) == 0:
            break
        a[:] = np.logical_or(a, c)
    a_count = np.count_nonzero(a)
    cell_count = a.size
    assert 1 <= a_count < cell_count, 'face set for surface is leaky or empty (surface does not intersect grid)'
    # find mean K for a cells and not a cells; if not a cells mean K is lesser (ie shallower), negate a
    layer_cell_count = grid_extent_kji[1] * grid_extent_kji[2]
    a_k_sum = 0
    not_a_k_sum = 0
    for k in range(grid_extent_kji[0]):
        a_layer_count = np.count_nonzero(a[k])
        a_k_sum += (k + 1) * a_layer_count
        not_a_k_sum += (k + 1) * (layer_cell_count - a_layer_count)
    a_mean_k = float(a_k_sum) / float(a_count)
    not_a_mean_k = float(not_a_k_sum) / float(cell_count - a_count)
    is_curtain = False
    if abs(a_mean_k - not_a_mean_k) <= 0.001:
        # log.warning('unable to determine which side of surface is shallower')
        is_curtain = True
    elif a_mean_k > not_a_mean_k and not raw_bisector:
        a[:] = np.logical_not(a)
    return a, is_curtain


def shadow_from_faces(extent_kji, k_faces):
    """Returns a numpy int8 array indicating whether cells are above, below or between K faces.

    arguments:
        extent_kji (triple int): the shape of the grid
        k_faces (bool array): True where a K face is present; shaped (nk - 1, nj, ni)

    returns:
        numpy int8 array of shape extent_kji; values are: 0 neither above nor below a K face;
            1: above any K faces in the column; 2 below any K faces in the column;
            3: between K faces (one or more above and one or more below)
    """
    assert len(extent_kji) == 3
    limit = extent_kji[0] - 1  # maximum number of iterations needed to spead shadow
    shadow = np.zeros(extent_kji, dtype = np.int8)
    shadow[:-1] = np.where(k_faces, 1, 0)
    shadow[1:] += np.where(k_faces, 2, 0)
    for _ in range(limit):
        c = np.logical_and(shadow[:-1] == 0, shadow[1:] == 1)
        if np.count_nonzero(c) == 0:
            break
        shadow[:-1] = np.where(c, 1, shadow[:-1])
    for _ in range(limit):
        c = np.logical_and(shadow[1:] == 0, shadow[:-1] == 2)
        if np.count_nonzero(c) == 0:
            break
        shadow[1:] = np.where(c, 2, shadow[1:])
    for _ in range(limit):
        c = np.logical_and(shadow[:-1] >= 2, shadow[1:] == 1)
        if np.count_nonzero(c) == 0:
            break
        shadow[:-1] = np.where(c, 3, shadow[:-1])
        shadow[1:] = np.where(c, 3, shadow[1:])
    return shadow


@njit
def first_true(array: np.ndarray) -> Optional[int]:  # type: ignore
    """Returns the index + 1 of the first True value in the array."""
    for idx, val in np.ndenumerate(array):
        if val:
            return idx[0] + 1
    return array.size


def get_boundary(  # type: ignore
        k_faces: np.ndarray, j_faces: np.ndarray, i_faces: np.ndarray, grid_extent_kji: Tuple[int, int,
                                                                                              int]) -> Dict[str, int]:
    """Cretaes a dictionary of the indices that bound the surface (where the faces are True).
    
    Args:
        k_faces (np.ndarray): a boolean array of which faces represent the surface in the k dimension.
        j_faces (np.ndarray): a boolean array of which faces represent the surface in the j dimension.
        i_faces (np.ndarray): a boolean array of which faces represent the surface in the i dimension.
        grid_extent_kji (Tuple[int, int, int]): the shape of the grid.

    Returns:
        boundary (Dict[str, int]): a dictionary of the indices that bound the surface.
    """
    boundary = {
        "k_min": None,
        "k_max": None,
        "j_min": None,
        "j_max": None,
        "i_min": None,
        "i_max": None,
    }

    indices = ['k', 'j', 'i']
    values = [0, 1, 2]
    for i, faces in enumerate([k_faces, j_faces, i_faces]):
        true_values = list(where_true(faces))
        val = values.copy()
        del val[i]

        index_1 = indices[val[0]]
        index_2 = indices[val[1]]
        true_values_dim1 = true_values[val[0]]
        true_values_dim2 = true_values[val[1]]

        if true_values_dim1.size > 0 and true_values_dim2.size > 0:
            for index, dim in zip([index_1, index_2], [true_values_dim1, true_values_dim2]):  # type: ignore
                if boundary[f"{index}_min"] is None:
                    boundary[f"{index}_min"] = min(dim)
                elif min(dim) < boundary[f"{index}_min"]:
                    boundary[f"{index}_min"] = min(dim)
                if boundary[f"{index}_max"] is None:
                    boundary[f"{index}_max"] = max(dim)
                elif max(dim) > boundary[f"{index}_max"]:
                    boundary[f"{index}_max"] = max(dim)

    for i, index in enumerate(indices):
        if boundary[f"{index}_min"] != 0:
            boundary[f"{index}_min"] -= 1  # type: ignore
        if boundary[f"{index}_max"] != grid_extent_kji[i] - 1:
            boundary[f"{index}_max"] += 1  # type: ignore

    return boundary  # type: ignore


def bisector_from_faces_new(  # type: ignore
        grid_extent_kji: Tuple[int, int, int], k_faces: np.ndarray, j_faces: np.ndarray, i_faces: np.ndarray,
        raw_bisector: bool) -> Tuple[np.ndarray, bool]:
    """Creates a boolean array denoting the bisection of the grid by the face sets.

    Args:
        grid_extent_kji (Tuple[int, int, int]): the shape of the grid.
        k_faces (np.ndarray): a boolean array of which faces represent the surface in the k dimension.
        j_faces (np.ndarray): a boolean array of which faces represent the surface in the j dimension.
        i_faces (np.ndarray): a boolean array of which faces represent the surface in the i dimension.

    Returns:
        Tuple containing:

        - array (np.ndarray): boolean bisectors array where values are True for cells on the side
            of the surface that has a lower mean k index on average and False for cells on the other side.
        - is_curtain (bool): True if the surface is a curtain (vertical), otherwise False.

    Notes:
        The face sets must form a single 'sealed' cut of the grid (eg. not waving in and out of the grid).
        Any 'boxed in' parts of the grid (completely enclosed by bisecting faces) will be consistently
        assigned to either the True or False part.
    """
    assert len(grid_extent_kji) == 3

    # Finding the surface boundary.
    boundary = get_boundary(k_faces, j_faces, i_faces, grid_extent_kji)

    # Setting up the bisector array for the bounding box.
    bounding_array = np.zeros((boundary["k_max"] - boundary["k_min"] + 1, boundary["j_max"] - boundary["j_min"] + 1,
                               boundary["i_max"] - boundary["i_min"] + 1),
                              dtype = np.bool_)

    # Seeding the bisector array from (0, 0, 0) up to the first faces that represent the surface.
    boundary_values = tuple(boundary.values())
    bounding_array, first_k, first_j, first_i = seed_array((0, 0, 0), k_faces, j_faces, i_faces, boundary_values,
                                                           bounding_array)
    points = set()
    for dimension, first_true in enumerate([first_k, first_j, first_i]):
        for dimension_value in range(1, first_true):
            point = [0, 0, 0]
            point[dimension] = dimension_value
            point = tuple(point)  # type: ignore
            bounding_array, first_k_sub, first_j_sub, first_i_sub = seed_array(point, k_faces, j_faces, i_faces,
                                                                               boundary_values, bounding_array)
            for sub_dimension, first_true_sub in enumerate([first_k_sub, first_j_sub, first_i_sub]):
                if dimension != sub_dimension:
                    for sub_dimension_value in range(1, first_true_sub):
                        point = [0, 0, 0]
                        point[dimension] = dimension_value
                        point[sub_dimension] = sub_dimension_value
                        point = tuple(point)  # type: ignore
                        if point not in points:
                            points.add(point)
                            bounding_array, _, _, _ = seed_array(point, k_faces, j_faces, i_faces, boundary_values,
                                                                 bounding_array)

    # Setting up the array for the changing values.
    changing_array = np.zeros_like(bounding_array, dtype = np.bool_)

    # Repeatedly spreading True values to neighbouring cells that are not the other side of a face.
    open_k = np.logical_not(k_faces)[boundary["k_min"]:boundary["k_max"], boundary["j_min"]:boundary["j_max"] + 1,
                                     boundary["i_min"]:boundary["i_max"] + 1]
    open_j = np.logical_not(j_faces)[boundary["k_min"]:boundary["k_max"] + 1, boundary["j_min"]:boundary["j_max"],
                                     boundary["i_min"]:boundary["i_max"] + 1]
    open_i = np.logical_not(i_faces)[boundary["k_min"]:boundary["k_max"] + 1, boundary["j_min"]:boundary["j_max"] + 1,
                                     boundary["i_min"]:boundary["i_max"]]
    while True:
        changing_array[:] = False

        # k faces
        changing_array[1:, :, :] = np.logical_and(bounding_array[:-1, :, :], open_k)
        changing_array[:-1, :, :] = np.logical_or(changing_array[:-1, :, :],
                                                  np.logical_and(bounding_array[1:, :, :], open_k))

        # j faces
        changing_array[:, 1:, :] = np.logical_or(changing_array[:, 1:, :],
                                                 np.logical_and(bounding_array[:, :-1, :], open_j))
        changing_array[:, :-1, :] = np.logical_or(changing_array[:, :-1, :],
                                                  np.logical_and(bounding_array[:, 1:, :], open_j))

        # i faces
        changing_array[:, :, 1:] = np.logical_or(changing_array[:, :, 1:],
                                                 np.logical_and(bounding_array[:, :, :-1], open_i))
        changing_array[:, :, :-1] = np.logical_or(changing_array[:, :, :-1],
                                                  np.logical_and(bounding_array[:, :, 1:], open_i))

        changing_array[:] = np.logical_and(changing_array, np.logical_not(bounding_array))
        if np.count_nonzero(changing_array) == 0:
            break
        bounding_array = np.logical_or(bounding_array, changing_array)

    # Setting up the full bisectors array and assigning the bounding box values.
    array = np.zeros(grid_extent_kji, dtype = np.bool_)
    array[boundary["k_min"]:boundary["k_max"] + 1, boundary["j_min"]:boundary["j_max"] + 1,
          boundary["i_min"]:boundary["i_max"] + 1] = bounding_array

    # Setting values outside of the bounding box.
    if boundary["k_max"] != grid_extent_kji[0] - 1 and np.any(bounding_array[-1, :, :]):
        array[boundary["k_max"] + 1:, :, :] = True
    if boundary["k_min"] != 0:
        array[:boundary["k_min"], :, :] = True
    if boundary["j_max"] != grid_extent_kji[1] - 1 and np.any(bounding_array[:, -1, :]):
        array[:, boundary["j_max"] + 1:, :] = True
    if boundary["j_min"] != 0:
        array[:, :boundary["j_min"], :] = True
    if boundary["i_max"] != grid_extent_kji[2] - 1 and np.any(bounding_array[:, :, -1]):
        array[:, :, boundary["i_max"] + 1:] = True
    if boundary["i_min"] != 0:
        array[:, :, :boundary["i_min"]] = True

    # Check all array elements are not the same.
    true_count = np.count_nonzero(array)
    cell_count = array.size
    assert 0 < true_count < cell_count, 'Face set for surface is leaky or empty (surface does not intersect grid).'

    # Negate the array if it minimises the mean k and determine if the surface is a curtain.
    layer_cell_count = grid_extent_kji[1] * grid_extent_kji[2]
    array_k_sum = 0
    array_opposite_k_sum = 0
    is_curtain = False
    for k in range(grid_extent_kji[0]):
        array_layer_count = np.count_nonzero(array[k])
        array_k_sum += (k + 1) * array_layer_count
        array_opposite_k_sum += (k + 1) * (layer_cell_count - array_layer_count)
    array_mean_k = float(array_k_sum) / float(true_count)
    array_opposite_mean_k = float(array_opposite_k_sum) / float(cell_count - true_count)
    if array_mean_k > array_opposite_mean_k and not raw_bisector:
        array[:] = np.logical_not(array)
    if abs(array_mean_k - array_opposite_mean_k) <= 0.001:
        # log.warning('unable to determine which side of surface is shallower')
        is_curtain = True

    return array, is_curtain


@njit
def seed_array(point: Tuple[int, int, int], k_faces: np.ndarray, j_faces: np.ndarray, i_faces: np.ndarray,
               boundary: Tuple[int, int, int, int, int, int], array: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
    """Sets values of the array True up until a face is hit in each direction.
    
    Args:
        point (Tuple[int, int, int]): coordinates of the initial seed point.
        k_faces (np.ndarray): a boolean array of which faces represent the surface in the k dimension.
        j_faces (np.ndarray): a boolean array of which faces represent the surface in the j dimension.
        i_faces (np.ndarray): a boolean array of which faces represent the surface in the i dimension.
        boundary (Tuple[int, int, int, int, int, int]): the boundaries of the surface given in the order
            (minimum k, maximum k, minimum j, maximum j, minimum i, maximum i).
        array (np.ndarray): boolean array that will be seeded.
    
    Returns:
        Tuple containing:

        - array (np.ndarray): boolean array that has been seeded.
        - first_k (int): the index of the first k face in the k direction from the seed point or the
            array size in the k direction if there are no k faces.
        - first_j (int): the index of the first j face in the j direction from the seed point or the
            array size in the j direction if there are no j faces.
        - first_i (int): the index of the first i face in the i direction from the seed point or the
            array size in the i direction if there are no i faces.
    """
    k = point[0]
    j = point[1]
    i = point[2]

    first_k = 0
    if k == 0:
        first_k = first_true(k_faces[boundary[0]:boundary[1], boundary[2] + j, boundary[4] + i])
        array[:first_k, j, i] = True

    first_j = 0
    if j == 0:
        first_j = first_true(j_faces[boundary[0] + k, boundary[2]:boundary[3], boundary[4] + i])
        array[k, :first_j, i] = True

    first_i = 0
    if i == 0:
        first_i = first_true(i_faces[boundary[0] + k, boundary[2] + j, boundary[4]:boundary[5]])
        array[k, j, :first_i] = True

    return array, first_k, first_j, first_i


def column_bisector_from_faces(grid_extent_ji: Tuple[int, int], j_faces: np.ndarray, i_faces: np.ndarray) -> np.ndarray:
    """Returns a numpy bool array denoting the bisection of the top layer of the grid by the curtain face sets.

    Args:
        grid_extent_ji (pair of int): the shape of a layer of the grid
        j_faces, i_faces (numpy bool arrays): True where an internal grid face forms part of the
            bisecting surface, shaped for a single layer

    Returns:
        numpy bool array of shape grid_extent_ji, set True for cells on one side of the face sets;
        set False for cells on othe side

    Notes:
        the face sets must form a single 'sealed' cut of the grid (eg. not waving in and out of the grid);
        any 'boxed in' parts of the grid (completely enclosed by bisecting faces) will be consistently
        assigned to the False part;
        the resulting array is suitable for use as a grid property with indexable element of columns;
        the array is set True for the side of the curtain that contains cell [0, 0]
    """
    assert len(grid_extent_ji) == 2
    assert j_faces.ndim == 2 and i_faces.ndim == 2
    assert j_faces.shape == (grid_extent_ji[0] - 1, grid_extent_ji[1])
    assert i_faces.shape == (grid_extent_ji[0], grid_extent_ji[1] - 1)
    a = np.zeros(grid_extent_ji, dtype = np.bool_)  # initialise to False
    c = np.zeros(grid_extent_ji, dtype = np.bool_)  # cells changing
    open_j = np.logical_not(j_faces)
    open_i = np.logical_not(i_faces)
    # set one or more seeds; todo: more seeds to improve performance if needed
    a[0, 0] = True
    # repeatedly spread True values to neighbouring cells that are not the other side of a face
    # todo: check that following works when a dimension has extent 1
    limit = grid_extent_ji[0] * grid_extent_ji[1]
    for _ in range(limit):
        c[:] = False
        # j faces
        c[1:, :] = np.logical_or(c[1:, :], np.logical_and(a[:-1, :], open_j))
        c[:-1, :] = np.logical_or(c[:-1, :], np.logical_and(a[1:, :], open_j))
        # i faces
        c[:, 1:] = np.logical_or(c[:, 1:], np.logical_and(a[:, :-1], open_i))
        c[:, :-1] = np.logical_or(c[:, :-1], np.logical_and(a[:, 1:], open_i))
        c[:] = np.logical_and(c, np.logical_not(a))
        if np.count_nonzero(c) == 0:  # no more changes
            break
        a[:] = np.logical_or(a, c)
    if np.all(a):
        log.warning('curtain is leaky or misses grid when setting column bisector')
    # log.debug(f'returning bisector with count: {np.count_nonzero(a)} of {a.size}; shape: {a.shape}')
    return a


def find_faces_to_represent_surface_regular_optimised(
        grid,
        surface,
        name,
        title = None,
        centres = None,  # DEPRECATED; TODO: remove this argument
        agitate = False,
        feature_type = 'fault',
        is_curtain = False,
        progress_fn = None,
        consistent_side = False,  # DEPRECATED; functionality no longer supported
        return_properties = None,
        raw_bisector = False):
    """Returns a grid connection set containing those cell faces which are deemed to represent the surface.

    Args:
        grid (RegularGrid): the grid for which to create a grid connection set representation of the surface;
           must be aligned, ie. I with +x, J with +y, K with +z and local origin of (0.0, 0.0, 0.0)
        surface (Surface): the surface to be intersected with the grid
        name (str): the feature name to use in the grid connection set
        title (str, optional): the citation title to use for the grid connection set; defaults to name
        centres (numpy float array of shape (nk, nj, ni, 3), optional): DEPRECATED, no longer used
        agitate (bool, default False): if True, the points of the surface are perturbed by a small random
           offset, which can help if the surface has been built from a regular mesh with a periodic resonance
           with the grid
        feature_type (str, default 'fault'): 'fault', 'horizon' or 'geobody boundary'
        is_curtain (bool, default False): if True, only the top layer of the grid is processed and the bisector
           property, if requested, is generated with indexable element columns
        progress_fn (f(x: float), optional): a callback function to be called at intervals by this function;
           the argument will progress from 0.0 to 1.0 in unspecified and uneven increments
        consistent_side (bool, default False): DEPRECATED; True will result in an assertion exception
        return_properties (List[str]): if present, a list of property arrays to calculate and
           return as a dictionary; recognised values in the list are 'triangle', 'depth', 'offset',
           'flange bool', 'grid bisector', or 'grid shadow';
           triangle is an index into the surface triangles of the triangle detected for the gcs face; depth is
           the z value of the intersection point of the inter-cell centre vector with a triangle in the surface;
           offset is a measure of the distance between the centre of the cell face and the intersection point;
           grid bisector is a grid cell boolean property holding True for the set of cells on one
           side of the surface, deemed to be shallower;
           grid shadow is a grid cell int8 property holding 0: cell neither above nor below a K face of the
           gridded surface, 1 cell is above K face(s), 2 cell is below K face(s), 3 cell is between K faces;
           the returned dictionary has the passed strings as keys and numpy arrays as values
        raw_bisector (bool, default False): if True and grid bisector is requested then it is left in a raw
           form without assessing which side is shallower (True values indicate same side as origin cell)

    Returns:
        gcs  or  (gcs, gcs_props)
        where gcs is a new GridConnectionSet with a single feature, not yet written to hdf5 nor xml created;
        gcs_props is a dictionary mapping from requested return_properties string to numpy array

    Notes:
        this function is designed for aligned regular grids only;
        this function can handle the surface and grid being in different coordinate reference systems, as
        long as the implicit parent crs is shared;
        no trimming of the surface is carried out here: for computational efficiency, it is recommended
        to trim first;
        organisational objects for the feature are created if needed;
        if the offset return property is requested, the implicit units will be the z units of the grid's crs
    """

    if centres is not None:
        warnings.warn(
            'DEPRECATED: centres aegument to find_faces_to_represent_surface_regular_optimised() should be removed')
    assert not consistent_side, 'NO LONGER SUPPORTED: consistent side functionality has been removed from this function'

    assert isinstance(grid, grr.RegularGrid)
    assert grid.is_aligned
    return_triangles = False
    return_depths = False
    return_offsets = False
    return_bisector = False
    return_shadow = False
    return_flange_bool = False
    if return_properties:
        assert all([
            p in ['triangle', 'depth', 'offset', 'grid bisector', 'grid shadow', 'flange bool']
            for p in return_properties
        ])
        return_triangles = ('triangle' in return_properties)
        return_depths = ('depth' in return_properties)
        return_offsets = ('offset' in return_properties)
        return_bisector = ('grid bisector' in return_properties)
        return_shadow = ('grid shadow' in return_properties)
        return_flange_bool = ('flange bool' in return_properties)
        if return_flange_bool:
            return_triangles = True

    if title is None:
        title = name

    if progress_fn is not None:
        progress_fn(0.0)

    log.debug(f'intersecting surface {surface.title} with regular grid {grid.title}')
    # log.debug(f'grid extent kji: {grid.extent_kji}')

    grid_dxyz = (grid.block_dxyz_dkji[2, 0], grid.block_dxyz_dkji[1, 1], grid.block_dxyz_dkji[0, 2])
    triangles, points = surface.triangles_and_points()
    assert triangles is not None and points is not None, f'surface {surface.title} is empty'
    if agitate:
        points += 1.0e-5 * (np.random.random(points.shape) - 0.5)
    # log.debug(f'surface: {surface.title}; p0: {points[0]}; crs uuid: {surface.crs_uuid}')
    # log.debug(f'surface min xyz: {np.min(points, axis = 0)}')
    # log.debug(f'surface max xyz: {np.max(points, axis = 0)}')
    if not bu.matching_uuids(grid.crs_uuid, surface.crs_uuid):
        log.debug('converting from surface crs to grid crs')
        s_crs = rqc.Crs(surface.model, uuid = surface.crs_uuid)
        s_crs.convert_array_to(grid.crs, points)
        surface.crs_uuid = grid.crs.uuid
        # log.debug(f'surface: {surface.title}; p0: {points[0]}; crs uuid: {surface.crs_uuid}')
        # log.debug(f'surface min xyz: {np.min(points, axis = 0)}')
        # log.debug(f'surface max xyz: {np.max(points, axis = 0)}')

    nk = 1 if is_curtain else grid.nk
    # K direction (xy projection)
    if nk > 1:
        # log.debug("searching for k faces")
        k_faces = np.zeros((nk - 1, grid.nj, grid.ni), dtype = bool)
        k_triangles = np.full((nk - 1, grid.nj, grid.ni), -1, dtype = int)
        k_depths = np.full((nk - 1, grid.nj, grid.ni), np.nan)
        k_offsets = np.full((nk - 1, grid.nj, grid.ni), np.nan)
        p_xy = np.delete(points, 2, 1)

        # k_hits = vec.points_in_polygons(k_centres, p_xy[triangles], grid.ni)
        # k_hits = vec.points_in_triangles_njit(k_centres, p_xy[triangles], grid.ni)
        k_hits = vec.points_in_triangles_aligned_optimised(grid.ni, grid.nj, grid_dxyz[0], grid_dxyz[1],
                                                           p_xy[triangles])

        axis = 2
        index1 = 1
        index2 = 2
        k_faces, k_offsets, k_triangles =  \
            intersect_numba(axis, index1, index2, k_hits, nk, points,
                            triangles, grid_dxyz, k_faces,
                            return_depths, k_depths,
                            return_offsets, k_offsets, return_triangles, k_triangles)
        del k_hits
        del p_xy
        log.debug(f"k face count: {np.count_nonzero(k_faces)}")
    else:
        k_faces = None
        k_triangles = None
        k_depths = None
        k_offsets = None

    if progress_fn is not None:
        progress_fn(0.3)

    # J direction (xz projection)
    if grid.nj > 1:
        # log.debug("searching for j faces")
        j_faces = np.zeros((nk, grid.nj - 1, grid.ni), dtype = bool)
        j_triangles = np.full((nk, grid.nj - 1, grid.ni), -1, dtype = int)
        j_depths = np.full((nk, grid.nj - 1, grid.ni), np.nan)
        j_offsets = np.full((nk, grid.nj - 1, grid.ni), np.nan)
        p_xz = np.delete(points, 1, 1)

        # j_hits = vec.points_in_polygons(j_centres, p_xz[triangles], grid.ni)
        # j_hits = vec.points_in_triangles_njit(j_centres, p_xz[triangles], grid.ni)
        j_hits = vec.points_in_triangles_aligned_optimised(grid.ni, nk, grid_dxyz[0], grid_dxyz[2], p_xz[triangles])

        axis = 1
        index1 = 0
        index2 = 2
        j_faces, j_offsets, j_triangles =  \
            intersect_numba(axis, index1, index2, j_hits, grid.nj, points,
                            triangles, grid_dxyz, j_faces,
                            return_depths, j_depths,
                            return_offsets, j_offsets, return_triangles, j_triangles)
        del j_hits
        del p_xz
        if is_curtain and grid.nk > 1:  # expand arrays to all layers
            j_faces = np.repeat(j_faces, grid.nk, axis = 0)
            j_triangles = np.repeat(j_triangles, grid.nk, axis = 0)
            j_depths = np.repeat(j_depths, grid.nk, axis = 0)
            j_offsets = np.repeat(j_offsets, grid.nk, axis = 0)
        log.debug(f"j face count: {np.count_nonzero(j_faces)}")
    else:
        j_faces = None
        j_triangles = None
        j_depths = None
        j_offsets = None

    if progress_fn is not None:
        progress_fn(0.6)

    # I direction (yz projection)
    if grid.ni > 1:
        # log.debug("searching for i faces")
        i_faces = np.zeros((nk, grid.nj, grid.ni - 1), dtype = bool)
        i_triangles = np.full((nk, grid.nj, grid.ni - 1), -1, dtype = int)
        i_depths = np.full((nk, grid.nj, grid.ni - 1), np.nan)
        i_offsets = np.full((nk, grid.nj, grid.ni - 1), np.nan)
        p_yz = np.delete(points, 0, 1)

        # i_hits = vec.points_in_polygons(i_centres, p_yz[triangles], grid.nj)
        # i_hits = vec.points_in_triangles_njit(i_centres, p_yz[triangles], grid.nj)
        # log.debug('calling points_in_triangles_aligned_optimised()')
        i_hits = vec.points_in_triangles_aligned_optimised(grid.nj, nk, grid_dxyz[1], grid_dxyz[2], p_yz[triangles])

        axis = 0
        index1 = 0
        index2 = 1
        # log.debug('calling intersect_numba()')
        i_faces, i_offsets, i_triangles =  \
            intersect_numba(axis, index1, index2, i_hits, grid.ni, points,
                            triangles, grid_dxyz, i_faces,
                            return_depths, i_depths,
                            return_offsets, i_offsets, return_triangles, i_triangles)
        del i_hits
        del p_yz
        if is_curtain and grid.nk > 1:  # expand arrays to all layers
            # log.debug('expanding curtain faces')
            i_faces = np.repeat(i_faces, grid.nk, axis = 0)
            i_triangles = np.repeat(i_triangles, grid.nk, axis = 0)
            i_depths = np.repeat(i_depths, grid.nk, axis = 0)
            i_offsets = np.repeat(i_offsets, grid.nk, axis = 0)
        log.debug(f"i face count: {np.count_nonzero(i_faces)}")
    else:
        i_faces = None
        i_triangles = None
        i_depths = None
        i_offsets = None

    if progress_fn is not None:
        progress_fn(0.9)

    log.debug("converting face sets into grid connection set")
    gcs = rqf.GridConnectionSet(
        grid.model,
        grid = grid,
        k_faces = k_faces,
        j_faces = j_faces,
        i_faces = i_faces,
        k_sides = None,
        j_sides = None,
        i_sides = None,
        feature_name = name,
        feature_type = feature_type,
        title = title,
        create_organizing_objects_where_needed = True,
    )
    # log.debug('finished coversion to gcs')

    # NB. following assumes faces have been added to gcs in a particular order!
    if return_triangles:
        # log.debug('preparing triangles array')
        k_tri_list = np.empty((0,)) if k_triangles is None else k_triangles[where_true(k_faces)]
        j_tri_list = np.empty((0,)) if j_triangles is None else j_triangles[where_true(j_faces)]
        i_tri_list = np.empty((0,)) if i_triangles is None else i_triangles[where_true(i_faces)]
        all_tris = np.concatenate((k_tri_list, j_tri_list, i_tri_list), axis = 0)
        # log.debug(f'gcs count: {gcs.count}; all triangles shape: {all_tris.shape}')
        assert all_tris.shape == (gcs.count,)

    # NB. following assumes faces have been added to gcs in a particular order!
    if return_depths:
        # log.debug('preparing depths array')
        k_depths_list = np.empty((0,)) if k_depths is None else k_depths[where_true(k_faces)]
        j_depths_list = np.empty((0,)) if j_depths is None else j_depths[where_true(j_faces)]
        i_depths_list = np.empty((0,)) if i_depths is None else i_depths[where_true(i_faces)]
        all_depths = np.concatenate((k_depths_list, j_depths_list, i_depths_list), axis = 0)
        # log.debug(f'gcs count: {gcs.count}; all depths shape: {all_depths.shape}')
        assert all_depths.shape == (gcs.count,)

    # NB. following assumes faces have been added to gcs in a particular order!
    if return_offsets:
        # log.debug('preparing offsets array')
        k_offsets_list = np.empty((0,)) if k_offsets is None else k_offsets[where_true(k_faces)]
        j_offsets_list = np.empty((0,)) if j_offsets is None else j_offsets[where_true(j_faces)]
        i_offsets_list = np.empty((0,)) if i_offsets is None else i_offsets[where_true(i_faces)]
        all_offsets = _all_offsets(grid.crs, k_offsets_list, j_offsets_list, i_offsets_list)
        # log.debug(f'gcs count: {gcs.count}; all offsets shape: {all_offsets.shape}')
        assert all_offsets.shape == (gcs.count,)

    if return_flange_bool:
        # log.debug('preparing flange array')
        flange_bool_uuid = surface.model.uuid(title = 'flange bool',
                                              obj_type = 'DiscreteProperty',
                                              related_uuid = surface.uuid)
        assert flange_bool_uuid is not None, f"No flange bool property found for surface: {surface.title}"
        flange_bool = rqp.Property(surface.model, uuid = flange_bool_uuid)
        flange_array = flange_bool.array_ref()
        all_flange = np.take(flange_array, all_tris)
        assert all_flange.shape == (gcs.count,)

    # note: following is a grid cells property, not a gcs property
    if return_bisector:
        if is_curtain:
            log.debug('preparing columns bisector')
            bisector = column_bisector_from_faces((grid.nj, grid.ni), j_faces[0], i_faces[0])
            # log.debug('finished preparing columns bisector')
        else:
            log.debug('preparing cells bisector')
            bisector, is_curtain = bisector_from_faces(tuple(grid.extent_kji), k_faces, j_faces, i_faces, raw_bisector)
            if is_curtain:
                bisector = bisector[0]  # reduce to a columns property

    # note: following is a grid cells property, not a gcs property
    if return_shadow:
        log.debug('preparing cells shadow')
        shadow = shadow_from_faces(tuple(grid.extent_kji), k_faces)

    if progress_fn is not None:
        progress_fn(1.0)

    log.debug(f'finishing find_faces_to_represent_surface_regular_optimised for {name}')

    # if returning properties, construct dictionary
    if return_properties:
        props_dict = {}
        if return_triangles:
            props_dict['triangle'] = all_tris
        if return_depths:
            props_dict['depth'] = all_depths
        if return_offsets:
            props_dict['offset'] = all_offsets
        if return_bisector:
            props_dict['grid bisector'] = (bisector, is_curtain)
        if return_shadow:
            props_dict['grid shadow'] = shadow
        if return_flange_bool:
            props_dict['flange bool'] = all_flange
        return (gcs, props_dict)

    return gcs


def find_faces_to_represent_surface(grid, surface, name, mode = 'auto', feature_type = 'fault', progress_fn = None):
    """Returns a grid connection set containing those cell faces which are deemed to represent the surface."""

    log.debug('finding cell faces for surface')
    if mode == 'auto':
        if isinstance(grid, grr.RegularGrid) and grid.is_aligned:
            mode = 'regular_optimised'
        else:
            mode = 'staffa'
    if mode == 'staffa':
        return find_faces_to_represent_surface_staffa(grid,
                                                      surface,
                                                      name,
                                                      feature_type = feature_type,
                                                      progress_fn = progress_fn)
    elif mode == 'regular':
        return find_faces_to_represent_surface_regular(grid,
                                                       surface,
                                                       name,
                                                       feature_type = feature_type,
                                                       progress_fn = progress_fn)
    elif mode == 'regular_optimised':
        return find_faces_to_represent_surface_regular_optimised(grid,
                                                                 surface,
                                                                 name,
                                                                 feature_type = feature_type,
                                                                 progress_fn = progress_fn)
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
        t_box = traj_box.copy()
        traj_box[0] = np.min(t_box, axis = 0)
        traj_box[1] = np.max(t_box, axis = 0)
    log.debug(f'overlap check: traj: {traj_box}; grid{grid_box}; overlap: {bx.boxes_overlap(traj_box, grid_box)}')
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

    assert isinstance(blocked_well, rqw.BlockedWell)
    assert isinstance(blocked_well.trajectory, rqw.Trajectory)
    assert grid is not None

    flavour = grr.grid_flavour(grid.root)
    if not flavour.startswith('Ijk'):
        raise NotImplementedError('well blocking only implemented for IjkGridRepresentation')
    is_regular = (flavour == 'IjkBlockGrid') and hasattr(grid, 'is_aligned') and grid.is_aligned
    log.debug(f"well blocking: grid {'is' if is_regular else 'is not'} regular and aligned")

    if grid.k_gaps:
        use_single_layer_tactics = False
        log.debug('skin single layer tactics disabled')

    grid_crs = rqc.Crs(grid.model, uuid = grid.crs_uuid)
    trajectory = __trajectory_init(blocked_well, grid, grid_crs)
    traj_xyz = trajectory.control_points

    if not trajectory_grid_overlap(trajectory, grid):
        log.error(f'no overlap of trajectory xyz box with grid for trajectory uuid: {trajectory.uuid}')
        return None
    grid_box = grid.xyz_box(lazy = False, local = True).copy()
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
    knot_count = trajectory.knot_count - 1
    while knot < knot_count and z_sign * traj_xyz[knot + 1, 2] < grid_top_z:
        knot += 1
    log.debug(f'skipped trajectory to knot: {knot}')
    if knot == knot_count:
        log.warning('no well blocking due to entire trajectory being shallower than top of grid')
        return None  # entire well is above grid

    if lazy:
        # search for intersection with top of grid: will fail if well comes in from the side (or from below!) or
        # penetrates through fault
        xyz, entry_knot, col_ji0 = find_first_intersection_of_trajectory_with_layer_interface(
            trajectory,
            grid,
            k0 = 0,
            ref_k_faces = 'top',
            start = knot,
            heal_faults = False,
            quad_triangles = quad_triangles,
            is_regular = is_regular)
        log.debug(f'lazy top intersection x,y,z: {xyz}; knot: {entry_knot}; col j0,i0: {col_ji0[0]}, {col_ji0[1]}')
        if xyz is None:
            log.error('failed to lazily find intersection of trajectory with top surface of grid')
            return None
        cell_kji0 = np.array((0, col_ji0[0], col_ji0[1]), dtype = int)
        axis = 0
        polarity = 0

    else:  # not lazy
        # note: xyz and entry_fraction might be slightly off when penetrating a skewed fault plane – deemed immaterial
        # for real cases
        skin = grid.skin(use_single_layer_tactics = use_single_layer_tactics, is_regular = is_regular)
        xyz, cell_kji0, axis, polarity, entry_knot = skin.find_first_intersection_of_trajectory(trajectory)
        if xyz is None:
            log.error('failed to find intersection of trajectory with outer skin of grid')
            return None
        else:
            log.debug(f"skin intersection x,y,z: {xyz}; knot: {entry_knot}; cell kji0: {cell_kji0}; face: "
                      f"{'KJI'[axis]}{'-+'[polarity]}")
            cell_kji0 = np.array(cell_kji0, dtype = int)

    previous_kji0 = cell_kji0.copy()
    previous_kji0[axis] += polarity * 2 - 1  # note: previous may legitimately be 'beyond' edge of grid
    entry_fraction = __segment_fraction(traj_xyz, entry_knot, xyz)
    log.debug(f'initial previous kji0: {previous_kji0}')
    next_cell_info = __find_next_cell(grid,
                                      previous_kji0,
                                      axis,
                                      1 - polarity,
                                      trajectory,
                                      entry_knot,
                                      entry_fraction,
                                      xyz,
                                      check_for_reentry,
                                      treat_skin_as_fault = use_single_layer_tactics,
                                      lazy = lazy,
                                      use_single_layer_tactics = use_single_layer_tactics)
    log.debug(f'initial next cell info: {next_cell_info}')
    node_mds_list = [__back_calculated_md(trajectory, entry_knot, entry_fraction)]
    node_count = 1
    grid_indices_list = []
    cell_count = 0
    cell_indices_list = []
    face_pairs_list = []
    kissed = np.zeros(grid.extent_kji, dtype = bool)
    sample_test = np.zeros(grid.extent_kji, dtype = bool)

    while next_cell_info is not None:
        entry_shared, kji0, entry_axis, entry_polarity, entry_knot, entry_fraction, entry_xyz = next_cell_info

        # log.debug('next cell entry x,y,z: ' + str(entry_xyz) + '; knot: ' + str(entry_knot) + '; cell kji0: ' +
        #           str(kji0) + '; face: ' + 'KJI'[entry_axis] + '-+'[entry_polarity])

        if not entry_shared:
            log.debug('adding unblocked interval')
            node_mds_list.append(__back_calculated_md(trajectory, entry_knot, entry_fraction))
            node_count += 1
            grid_indices_list.append(-1)
            kissed[:] = False
            sample_test[:] = False

        exit_xyz, exit_knot, exit_axis, exit_polarity = find_first_intersection_of_trajectory_with_cell_surface(
            trajectory, grid, kji0, entry_knot, start_xyz = entry_xyz, nudge = 0.01, quad_triangles = True)

        #  if exit_xyz is None:
        #      log.debug('no exit')
        # else:
        #     log.debug('cell exit x,y,z: ' + str(exit_xyz) + '; knot: ' + str(exit_knot) + '; face: ' +
        #               'KJI'[exit_axis] + '-+'[exit_polarity])

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
                next_cell_info = __forward_nudge(axis, entry_axis, entry_knot, entry_polarity, entry_xyz, grid, kissed,
                                                 kji0, previous_kji0, sample_test, traj_xyz, trajectory,
                                                 use_single_layer_tactics)
            if next_cell_info is None:
                log.warning('well blocking got stuck – cells probably omitted at tail of well')
        #      if exit_xyz is not None:  # usual well-behaved case or non-standard due to kiss
        else:
            exit_fraction = __segment_fraction(traj_xyz, exit_knot, exit_xyz)
            log.debug(f'adding blocked interval for cell kji0: {kji0}')
            # todo: check for inactive cell and make an unblocked interval instead, if required
            node_mds_list.append(__back_calculated_md(trajectory, exit_knot, exit_fraction))
            node_count += 1
            grid_indices_list.append(0)
            cell_indices_list.append(grid.natural_cell_index(kji0))
            cell_count += 1
            face_pairs_list.append(((entry_axis, entry_polarity), (exit_axis, exit_polarity)))

            previous_kji0 = kji0
            # log.debug(f'previous kji0 set to {previous_kji0}')
            next_cell_info = __find_next_cell(grid,
                                              kji0,
                                              exit_axis,
                                              exit_polarity,
                                              trajectory,
                                              exit_knot,
                                              exit_fraction,
                                              exit_xyz,
                                              check_for_reentry,
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

    blocked_well.node_mds = np.array(node_mds_list, dtype = float)
    blocked_well.node_count = node_count
    blocked_well.grid_indices = np.array(grid_indices_list, dtype = int)
    blocked_well.cell_indices = np.array(cell_indices_list, dtype = int)
    blocked_well.face_pair_indices = np.array(face_pairs_list, dtype = int)
    blocked_well.cell_count = cell_count
    blocked_well.grid_list = [grid]

    assert cell_count == (node_count - np.count_nonzero(blocked_well.grid_indices == -1) - 1)

    log.info(f'{cell_count} cell{__pl(cell_count)} blocked for well trajectory uuid: {trajectory.uuid}')

    return blocked_well


def __find_next_cell(
    grid,
    previous_kji0,
    axis,
    polarity,
    trajectory,
    segment,
    seg_fraction,
    xyz,
    check_for_reentry,
    treat_skin_as_fault = False,
    lazy = False,
    use_single_layer_tactics = True,
):
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
            seg_fraction = __segment_fraction(trajectory.control_points, segment, xyz_r)
            return False, cell_kji0, axis, polarity, segment, seg_fraction, xyz_r
        else:
            return None
    # pre-assess split pillar (fault) situation
    faulted = False
    if axis != 0:
        if grid.has_split_coordinate_lines:
            faulted = grid.is_split_column_face(kji0[1], kji0[2], axis, 1 - polarity)
        if not faulted and treat_skin_as_fault:
            faulted = (kji0[axis] == 0 and polarity == 1) or (kji0[axis] == grid.extent_kji[axis] - 1 and polarity == 0)
    # handle the simplest case of a well behaved k neighbour or unsplit j or i neighbour
    if not grid.pinched_out(cell_kji0 = kji0, cache_pinchout_array = True) and not faulted:
        return True, kji0, axis, 1 - polarity, segment, seg_fraction, xyz
    if faulted:
        return __faulted(trajectory, grid, segment, kji0, axis, polarity, xyz, lazy, use_single_layer_tactics,
                         previous_kji0)
    else:
        # skip pinched out cells
        pinchout_skip_sign = -1 if axis == 0 and polarity == 0 else 1
        while grid.pinched_out(cell_kji0 = kji0, cache_pinchout_array = True):
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
                seg_fraction = __segment_fraction(trajectory.control_points, segment, xyz_r)
                return False, cell_kji0, axis, polarity, segment, seg_fraction, xyz_r
        return True, kji0, axis, 1 - polarity, segment, seg_fraction, xyz


def __faulted(trajectory, grid, segment, kji0, axis, polarity, xyz, lazy, use_single_layer_tactics, previous_kji0):
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
        seg_fraction = __segment_fraction(trajectory.control_points, segment, xyz_f)
        return vec.isclose(xyz, xyz_f, tolerance = 0.001), kji0, axis, 1 - polarity, segment, seg_fraction, xyz_f
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
        seg_fraction = __segment_fraction(trajectory.control_points, segment_f, xyz_f)
        return vec.isclose(xyz, xyz_f, tolerance = 0.001), kji0, axis, 1 - polarity, segment_f, seg_fraction, xyz_f
    log.debug("failed to find entry point in cross section after crossing fault"
              f"{'' if lazy else '; checking for skin re-entry'}")
    if lazy:
        return None
    skin = grid.skin(use_single_layer_tactics = use_single_layer_tactics)
    # following is problematic due to skewed fault planes
    xyz_r, cell_kji0, axis, polarity, segment = skin.find_first_intersection_of_trajectory(trajectory,
                                                                                           start = segment,
                                                                                           start_xyz = xyz,
                                                                                           nudge = -0.1,
                                                                                           exclude_kji0 = previous_kji0)
    if xyz_r is None:
        log.warning('no skin re-entry found after exit through fault face')
        return None
    log.debug(f"skin re-entry after fault face exit: kji0: {cell_kji0}; face: {'KJI'[axis]}{'-+'[polarity]}")
    seg_fraction = __segment_fraction(trajectory.control_points, segment, xyz_r)
    return False, cell_kji0, axis, polarity, segment, seg_fraction, xyz_r


def __segment_fraction(trajectory_xyz, segment, xyz):
    # returns fraction of way along segment that point xyz lies
    segment_vector = trajectory_xyz[segment + 1] - trajectory_xyz[segment]
    segment_length = vec.naive_length(segment_vector)  # note this is length in straight line, rather than diff in mds
    return vec.naive_length(xyz - trajectory_xyz[segment]) / segment_length


def __back_calculated_md(trajectory, segment, fraction):
    base_md = trajectory.measured_depths[segment]
    return base_md + fraction * (trajectory.measured_depths[segment + 1] - base_md)


def __forward_nudge(axis, entry_axis, entry_knot, entry_polarity, entry_xyz, grid, kissed, kji0, previous_kji0,
                    sample_test, traj_xyz, trajectory, use_single_layer_tactics):
    next_cell_info = None
    # trajectory has kissed corner or edge of cell and nudge has gone outside cell
    # nudge forward and look for point inclusion in possible neighbours
    log.debug(f"kiss detected at cell kji0 {kji0} {'KJI'[entry_axis]}{'-+'[entry_polarity]}")
    kissed[tuple(kji0)] = True
    if np.all(previous_kji0 >= 0) and np.all(previous_kji0 < grid.extent_kji):
        kissed[tuple(previous_kji0)] = True  # stops immediate revisit for kiss and tell tactics
    # setup kji offsets of neighbouring cells likely to contain sample point (could check for split
    # column faces)
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
    next_cell_info = __try_cell_entry(entry_knot, entry_xyz, grid, kissed, kji0, next_cell_info, offsets_kji,
                                      pinchout_skip_sign, traj_xyz, trajectory)
    if next_cell_info is not None:
        return next_cell_info
    else:
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
                try_entry_xyz, try_entry_knot, try_entry_axis, try_entry_polarity = \
                    find_first_intersection_of_trajectory_with_cell_surface(trajectory, grid, try_kji0, entry_knot,
                                                                            start_xyz=entry_xyz, nudge=-0.01,
                                                                            quad_triangles=True)
                if try_entry_xyz is None or try_entry_knot is None:
                    log.warning(f'failed to find entry to favoured cell {try_kji0}')
                else:
                    if np.all(try_kji0 == kji0):
                        log.debug(f'staying in cell {kji0} after kiss')
                        # TODO: sort things out to discard kiss completely
                    try_entry_fraction = __segment_fraction(traj_xyz, try_entry_knot, try_entry_xyz)
                    return vec.isclose(try_entry_xyz, entry_xyz, tolerance=0.01), try_kji0, try_entry_axis, \
                           try_entry_polarity, try_entry_knot, try_entry_fraction, try_entry_xyz
                break
        log.debug('sample point not found in immediate neighbours, switching to full column search')
        k_offset = 0
        while k_offset <= max(kji0[0], grid.nk - kji0[0] - 1):
            for k_plus_minus in [1, -1]:
                for j_offset in [0, -1, 1]:
                    for i_offset in [0, -1, 1]:
                        try_kji0 = kji0 + np.array((k_offset * k_plus_minus, j_offset, i_offset))
                        if np.any(try_kji0 < 0) or np.any(try_kji0 >= grid.extent_kji) or sample_test[tuple(try_kji0)]:
                            continue
                        sample_test[tuple(try_kji0)] = True
                        if point_is_within_cell(sample_point, grid, try_kji0):
                            log.debug(f'sample point is in cell {try_kji0}')
                            try_entry_xyz, try_entry_knot, try_entry_axis, try_entry_polarity = \
                                find_first_intersection_of_trajectory_with_cell_surface(
                                    trajectory, grid, try_kji0, entry_knot, start_xyz=entry_xyz,
                                    nudge=-0.01, quad_triangles=True)
                            if try_entry_xyz is None or try_entry_knot is None:
                                log.warning(f'failed to find entry to column sampled cell {try_kji0}')
                                return __no_success(entry_knot, entry_xyz, grid, next_cell_info, previous_kji0,
                                                    traj_xyz, trajectory, use_single_layer_tactics)
                            else:
                                try_entry_fraction = __segment_fraction(traj_xyz, try_entry_knot, try_entry_xyz)
                                return (vec.isclose(try_entry_xyz, entry_xyz,
                                                    tolerance = 0.01), try_kji0, try_entry_axis, try_entry_polarity,
                                        try_entry_knot, try_entry_fraction, try_entry_xyz)
            k_offset += 1
        return __no_success(entry_knot, entry_xyz, grid, next_cell_info, previous_kji0, traj_xyz, trajectory,
                            use_single_layer_tactics)


def __no_success(entry_knot, entry_xyz, grid, next_cell_info, previous_kji0, traj_xyz, trajectory,
                 use_single_layer_tactics):
    log.debug('no success during full column search')
    if np.any(previous_kji0 == 0) or np.any(previous_kji0 == grid.extent_kji - 1):
        log.debug('looking for skin re-entry after possible kissing exit')
        skin = grid.skin(use_single_layer_tactics = use_single_layer_tactics)
        try_entry_xyz, try_kji0, try_entry_axis, try_entry_polarity, try_entry_knot = \
            skin.find_first_intersection_of_trajectory(trajectory, start=entry_knot,
                                                       start_xyz=entry_xyz, nudge=+0.1)
        if try_entry_xyz is not None:
            log.debug('skin re-entry found')
            try_entry_fraction = __segment_fraction(traj_xyz, try_entry_knot, try_entry_xyz)
            next_cell_info = vec.isclose(try_entry_xyz, entry_xyz, tolerance=0.01), try_kji0, try_entry_axis, \
                             try_entry_polarity, try_entry_knot, try_entry_fraction, try_entry_xyz
    return next_cell_info


def __try_cell_entry(entry_knot, entry_xyz, grid, kissed, kji0, next_cell_info, offsets_kji, pinchout_skip_sign,
                     traj_xyz, trajectory):
    for try_index in range(len(offsets_kji)):
        try_kji0 = kji0 + offsets_kji[try_index]
        while np.all(try_kji0 >= 0) and np.all(try_kji0 < grid.extent_kji) and grid.pinched_out(cell_kji0 = try_kji0):
            try_kji0[0] += pinchout_skip_sign
        if np.any(try_kji0 < 0) or np.any(try_kji0 >= grid.extent_kji) or kissed[tuple(try_kji0)]:
            continue
        # use tiny negative nudge to look for entry into try cell with current segment; check that entry
        # point is close
        try_entry_xyz, try_entry_knot, try_entry_axis, try_entry_polarity = \
            find_first_intersection_of_trajectory_with_cell_surface(trajectory,
                                                                    grid,
                                                                    try_kji0,
                                                                    entry_knot,
                                                                    start_xyz=entry_xyz,
                                                                    nudge=-0.01,
                                                                    quad_triangles=True)
        if try_entry_xyz is not None and vec.isclose(try_entry_xyz, entry_xyz, tolerance = 0.02):
            log.debug(f'try accepted for cell: {try_kji0}')
            try_entry_fraction = __segment_fraction(traj_xyz, try_entry_knot, try_entry_xyz)
            # replace the next cell entry info
            next_cell_info = True, try_kji0, try_entry_axis, try_entry_polarity, try_entry_knot, try_entry_fraction, \
                             try_entry_xyz
            break
    return next_cell_info


def __trajectory_init(blocked_well, grid, grid_crs):
    if bu.matching_uuids(blocked_well.trajectory.crs_uuid, grid.crs_uuid):
        trajectory = blocked_well.trajectory
        assert grid_crs == rqc.Crs(blocked_well.model, uuid = trajectory.crs_uuid)
    else:
        # create a local trajectory object for crs alignment of control points with grid
        # NB. temporary objects, relationships left in a mess
        # model = rq.Model(create_basics = True)
        log.debug(f'creating temp trajectory in grid crs for well {rqw.well_name(blocked_well.trajectory)}')
        model = blocked_well.trajectory.model
        trajectory = rqw.Trajectory(model, uuid = blocked_well.trajectory.uuid)
        assert trajectory is not None and trajectory.control_points is not None
        traj_crs = rqc.Crs(model, uuid = trajectory.crs_uuid)
        traj_crs.convert_array_to(grid_crs, trajectory.control_points)  # trajectory xyz converted in situ to grid's crs
        trajectory.crs_uuid = grid_crs.uuid
        log.debug(
            f'temp traj xyz box: {np.min(trajectory.control_points, axis = 0)} to {np.max(trajectory.control_points, axis = 0)}'
        )
        # note: any represented interpretation object will not be present in the temporary model
    return trajectory


def _all_offsets(crs, k_offsets_list, j_offsets_list, i_offsets_list):
    if crs.xy_units == crs.z_units:
        return np.concatenate((k_offsets_list, j_offsets_list, i_offsets_list), axis = 0)
    ji_offsets = np.concatenate((j_offsets_list, i_offsets_list), axis = 0)
    wam.convert_lengths(ji_offsets, crs.xy_units, crs.z_units)
    return np.concatenate((k_offsets_list, ji_offsets), axis = 0)


def __pl(n, use_es = False):
    if n == 1:
        return ''
    return 'es' if use_es else 's'
