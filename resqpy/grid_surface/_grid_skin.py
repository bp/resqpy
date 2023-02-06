"""GridSkin class for representing outer skin of a Grid."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.surface as rqs
import resqpy.grid_surface as rqgs


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

            top_surf = rqgs.generate_torn_surface_for_layer_interface(grid,
                                                                      k0 = 0,
                                                                      ref_k_faces = 'top',
                                                                      quad_triangles = quad_triangles)
            base_surf = rqgs.generate_torn_surface_for_layer_interface(grid,
                                                                       k0 = grid.nk - 1,
                                                                       ref_k_faces = 'base',
                                                                       quad_triangles = quad_triangles)
            j_minus_surf = rqgs.generate_torn_surface_for_x_section(grid,
                                                                    'J',
                                                                    ref_slice0 = 0,
                                                                    plus_face = False,
                                                                    quad_triangles = quad_triangles,
                                                                    as_single_layer = use_single_layer_tactics)
            j_plus_surf = rqgs.generate_torn_surface_for_x_section(grid,
                                                                   'J',
                                                                   ref_slice0 = self.grid.nj - 1,
                                                                   plus_face = True,
                                                                   quad_triangles = quad_triangles,
                                                                   as_single_layer = use_single_layer_tactics)
            i_minus_surf = rqgs.generate_torn_surface_for_x_section(grid,
                                                                    'I',
                                                                    ref_slice0 = 0,
                                                                    plus_face = False,
                                                                    quad_triangles = quad_triangles,
                                                                    as_single_layer = use_single_layer_tactics)
            i_plus_surf = rqgs.generate_torn_surface_for_x_section(grid,
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
                _, surf = rqgs.create_column_face_mesh_and_surface(grid,
                                                                   col_ji0,
                                                                   1,
                                                                   1,
                                                                   quad_triangles = quad_triangles,
                                                                   as_single_layer = True)
                j_plus_fault_surf_list.append(surf)
                _, surf = rqgs.create_column_face_mesh_and_surface(grid, (col_ji0[0] + 1, col_ji0[1]),
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
                _, surf = rqgs.create_column_face_mesh_and_surface(grid,
                                                                   col_ji0,
                                                                   2,
                                                                   1,
                                                                   quad_triangles = quad_triangles,
                                                                   as_single_layer = True)
                i_plus_fault_surf_list.append(surf)
                _, surf = rqgs.create_column_face_mesh_and_surface(grid, (col_ji0[0], col_ji0[1] + 1),
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

            top_surf = rqgs.generate_untorn_surface_for_layer_interface(grid,
                                                                        k0 = 0,
                                                                        ref_k_faces = 'top',
                                                                        quad_triangles = quad_triangles)
            base_surf = rqgs.generate_untorn_surface_for_layer_interface(grid,
                                                                         k0 = grid.nk - 1,
                                                                         ref_k_faces = 'base',
                                                                         quad_triangles = quad_triangles)
            j_minus_surf = rqgs.generate_untorn_surface_for_x_section(grid,
                                                                      'J',
                                                                      ref_slice0 = 0,
                                                                      plus_face = False,
                                                                      quad_triangles = quad_triangles,
                                                                      as_single_layer = use_single_layer_tactics)
            j_plus_surf = rqgs.generate_untorn_surface_for_x_section(grid,
                                                                     'J',
                                                                     ref_slice0 = self.grid.nj - 1,
                                                                     plus_face = True,
                                                                     quad_triangles = quad_triangles,
                                                                     as_single_layer = use_single_layer_tactics)
            i_minus_surf = rqgs.generate_untorn_surface_for_x_section(grid,
                                                                      'I',
                                                                      ref_slice0 = 0,
                                                                      plus_face = False,
                                                                      quad_triangles = quad_triangles,
                                                                      as_single_layer = use_single_layer_tactics)
            i_plus_surf = rqgs.generate_untorn_surface_for_x_section(grid,
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
            bundle = rqgs.find_first_intersection_of_trajectory_with_surface(trajectory,
                                                                             self.skin,
                                                                             start = start,
                                                                             start_xyz = start_xyz,
                                                                             nudge = nudge,
                                                                             return_second = True)
            xyz_1, segment_1, tri_index_1, xyz_2, segment_2, tri_index_2 = bundle
        else:
            xyz_1, segment_1, tri_index_1 = rqgs.find_first_intersection_of_trajectory_with_surface(
                trajectory, self.skin, start = start, start_xyz = start_xyz, nudge = nudge)
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
                    xyz, k0 = rqgs.find_intersection_of_trajectory_interval_with_column_face(trajectory,
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
                xyz_f, k0 = rqgs.find_intersection_of_trajectory_interval_with_column_face(trajectory,
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
                        gap_top_surf = rqgs.generate_torn_surface_for_layer_interface(self.grid,
                                                                                      k0 = k0,
                                                                                      ref_k_faces = 'base',
                                                                                      quad_triangles = quad_triangles)
                        gap_base_surf = rqgs.generate_torn_surface_for_layer_interface(self.grid,
                                                                                       k0 = k0 + 1,
                                                                                       ref_k_faces = 'top',
                                                                                       quad_triangles = quad_triangles)
                    else:
                        gap_top_surf = rqgs.generate_untorn_surface_for_layer_interface(self.grid,
                                                                                        k0 = k0,
                                                                                        ref_k_faces = 'base',
                                                                                        quad_triangles = quad_triangles)
                        gap_base_surf = rqgs.generate_untorn_surface_for_layer_interface(
                            self.grid, k0 = k0 + 1, ref_k_faces = 'top', quad_triangles = quad_triangles)
                    k_gap_surf_list.append(gap_top_surf)
                    k_gap_surf_list.append(gap_base_surf)
                    self.k_gap_after_layer_list.append(k0)
        assert len(k_gap_surf_list) == 2 * self.k_gaps
        assert len(self.k_gap_after_layer_list) == self.k_gaps
        return k_gap_surf_list
