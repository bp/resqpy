"""Functions for finding intersections of wellbore trajectories with surfaces and grids."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.crs as rqc
import resqpy.surface as rqs
import resqpy.grid_surface as rqgs
import resqpy.olio.intersection as meet
import resqpy.olio.box_utilities as bx
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec


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
        surface = rqgs.generate_torn_surface_for_layer_interface(grid,
                                                                 k0 = k0,
                                                                 ref_k_faces = ref_k_faces,
                                                                 quad_triangles = quad_triangles)
    else:
        surface = rqgs.generate_untorn_surface_for_layer_interface(grid,
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
        surface = rqgs.generate_torn_surface_for_layer_interface(grid,
                                                                 k0 = k0,
                                                                 ref_k_faces = ref_k_faces,
                                                                 quad_triangles = quad_triangles)
    else:
        surface = rqgs.generate_untorn_surface_for_layer_interface(grid,
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
    _, col_face_surface = rqgs.create_column_face_mesh_and_surface(grid,
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
