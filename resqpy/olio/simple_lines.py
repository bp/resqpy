"""simple_lines.py: functions for handling simple lines in relation to a resqml grid."""

# line is represented as 2D numpy array of shape (NP, 3): [point index, xyz]

import logging

log = logging.getLogger(__name__)

import numpy as np


def read_lines(filename):
    """Returns a list of line arrays, read from ascii file.

    argument:
       filename (string): the path of the ascii file holding a set of poly-lines

    returns:
       list of numpy arrays, each array representing one poly-line

    notes:
       each line in the file must contain 3 floating point numbers: x, y, z;
       each poly-line must be terminated with a null marker line: 999.0 999.0 999.0
       there is no handling of units; elsewhere they will implicitly be assumed to be
       those of a crs for a grid object
    """

    lines_list = []
    end_of_line = np.array([999.0, 999.0, 999.0])
    try:
        with open(filename, 'r') as fp:
            point_count = 0
            point_list = None
            while True:
                line = fp.readline()
                if len(line) == 0:
                    assert point_count == 0, 'unterminated list of line points at end of file'
                    break
                words = line.split()
                if len(words) == 0:
                    continue  # blank line
                assert len(words) == 3, 'badly formed line (expecting 3 reals)'
                point = np.empty(3)
                for p in range(3):
                    point[p] = float(words[p])
                if np.all(np.isclose(point, end_of_line)):
                    if point_count:
                        lines_list.append(point_list)
                        point_count = 0
                elif point_count:
                    point_list = np.concatenate((point_list, point.reshape((1, 3))), axis = 0)
                    point_count += 1
                else:
                    point_list = np.empty((1, 3))
                    point_list[0, :] = point
                    point_count = 1
    except Exception:
        log.exception('failed to read simple lines from ascii file: ' + filename)
        lines_list = []
    log.info(str(len(lines_list)) + ' line(s) read from file: ' + filename)
    return lines_list


def polygon_line(line, tolerance = 0.001):
    """Returns a copy of the line with the last vertex stripped off if it is close to the first vertex.

    arguments:
       line (numpy array of floats): representation of a poly-line which might be closed (last vertex
          matches first vertex)
       tolerance (float, default = 0.001): the maximum Manhatten distance between two points for them
          to be treated as coincident

    returns:
       numpy array of floats which is either a copy of line, or a copy of line with the last point
       removed
    """

    last = len(line) - 1
    if last < 1:
        return line
    dims = len(line[0])
    manhattan = 0.0
    for d in range(dims):
        manhattan += abs(line[0][d] - line[last][d])
    if manhattan > tolerance:
        return line
    return line[:-1]


def duplicate_vertices_removed(line, tolerance = 0.001):
    """Returns a copy of the line with neighbouring duplicate vertices removed.

    arguments:
       line (2D numpy array of floats): representation of a poly-line
       tolerance (float, default = 0.001): the maximum Manhatten distance between two points for them
          to be treated as coincident

    returns:
       numpy array of floats which is either a copy of line, or a copy of line with some points
       removed

    notes:
       does not treat the line as a closed polyline, use polygon_line() function as well to remove
       duplicated first/last point;
       always preserves first and last point, even if they are identical and there are no other vertices
    """

    assert line.ndim == 2
    if len(line) < 3:
        return line
    dims = line.shape[1]
    whittled = np.zeros(line.shape)
    whittled[0] = line[0]
    c_i = 0
    for i in range(1, len(line)):
        manhattan = 0.0
        for d in range(dims):
            manhattan += abs(line[i, d] - whittled[c_i, d])
        if manhattan > tolerance:
            c_i += 1
            whittled[c_i] = line[i]
    if c_i == len(line) - 1:
        return line
    whittled[c_i] = line[-1]
    return whittled[:c_i + 1]


def nearest_pillars(line_list, grid, ref_k = 0, ref_kp = 0):
    """Finds pillars nearest to each point on each line; returns list of lists of (j0, i0).

    arguments:
       line_list (list of numpy arrays of floats): set of poly-lines, the points of which are used
          to find the nearest pillars in grid
       grid (grid.Grid object): the grid whose pillars are compared with the poly-line vertices
       ref_k (integer, default 0): the reference layer in the grid to compare against the vertices;
          zero based
       ref_kp (integer, default 0): 0 to indicate the top corners of the reference layer, 1 for the base

    returns:
       a list of lists of pairs of integers, each being the (j, i) pillar indices of the nearest pillar
       to the corresponding vertex of the poly-line; zero based indexing

    notes:
       this is a 2D search in the x, y plane; z values are ignored;
       poly-line x, y values must implicitly be in the same crs as the grid's points data
    """

    pillar_list_list = []
    for line in line_list:
        pillar_list = []
        p_count = line.shape[0]
        for p in range(p_count):
            xy = line[p, 0:2]
            ji = grid.nearest_pillar(xy, ref_k0 = ref_k + ref_kp)
            pillar_list.append(ji)
        pillar_list_list.append(pillar_list)
    return pillar_list_list


def nearest_rods(line_list, projection, grid, axis, ref_slice0 = 0, plus_face = False):
    """Finds rods nearest to each point on each line; returns list of lists of (k0, j0) or (k0, i0).

    arguments:
       line_list (list of numpy arrays of floats): set of poly-lines, the points of which are used
          to find the nearest rods in grid
       projection (string): 'xz' or 'yz'
       grid (grid.Grid object): the grid whose cross section points are compared with the poly-line vertices
       axis (string): 'I' or 'J' being the axis removed during slicing
       ref_slice0 (integer, default 0): the reference slice in the grid to compare against the vertices;
          zero based
       plus_face (boolean, default False): which face of the reference slice to use

    returns:
       a list of numpy arrays of pairs of integers, each being the (k, j) or (k, i) indices of the
       nearest rod to the corresponding vertex of the poly-line under projection; zero based indexing

    notes:
       this is a 2D search in the x, z or y, z plane;
       currently limited to unsplit grids without k gaps;
       poly-line x, y, z values must implicitly be in the same crs as the grid's points data
    """

    assert projection in ['xz', 'yz']
    assert axis.upper() in ['J', 'I']

    rod_list_list = []
    for line in line_list:
        line_a = np.array(line)
        rod_list_list.append(grid.nearest_rod(line_a, projection, axis, ref_slice0 = ref_slice0, plus_face = plus_face))
    return rod_list_list


def drape_lines(line_list, pillar_list_list, grid, ref_k = 0, ref_kp = 0, offset = -1.0, snap = False):
    """Roughly drapes lines over grid horizon; draped lines are suitable for 3D visualisation.

    arguments:
       line_list (list of numpy arrays of floats): the undraped poly-lines
       pillar_list_list: (list of lists of pairs of integers): as returned by nearest_pillars()
       grid: (grid.Grid object): the grid to which the poly-lines are to be draped
       ref_k (integer, default 0): the reference layer in the grid to which the lines will be
          draped; zero based
       ref_kp (integer, default 0): 0 to indicate the top corners of the reference layer, 1 for the base
       offset (float, default -1.0): the vertical offset to add to the z value of pillar points; positive
          drapes lines deeper, negative shallower (assumes grid's crs has z increasing downwards)
       snap (boolean, default False): if True, the x & y values of each vertex in the lines are moved to
          match the pillar point; if False, only the z values are adjusted

    returns:
       list of numpy arrays of floats, being the draped equivalents of the line_list

    notes:
       the units of the line list must implicitly be those of the crs for the grid;
       for grids with split coordinate lines (faults), only the primary pillars are currently used;
       the results of this function are intended for 3D visualisation of an indicative nature;
       resulting draped lines may penetrate the grid layer faces depending on the 3D geometry and
       the spacing of the vertices compared to cell sizes
    """

    assert len(line_list) == len(pillar_list_list)

    # note: currently uses primary pillars only: expect draping disasters by faults
    # pe_j = grid.extent_kji[1] + 1
    # pe_i = grid.extent_kji[2] + 1
    pillar_points = grid.horizon_points(ref_k0 = ref_k + ref_kp)

    draped_list = []
    for line, pillar_list in zip(line_list, pillar_list_list):
        drape = line.copy()
        p_count = line.shape[0]
        for p in range(p_count):
            ji = pillar_list[p]
            if snap:
                drape[p, :] = pillar_points[tuple(ji)]
            else:
                drape[p, 2] = pillar_points[tuple(ji)][2]
        drape[:, 2] += offset
        draped_list.append(drape)

    return draped_list


def drape_lines_to_rods(line_list,
                        rod_list_list,
                        projection,
                        grid,
                        axis,
                        ref_slice0 = 0,
                        plus_face = False,
                        offset = -1.0,
                        snap = False):
    """Roughly drapes lines near grid cross section; draped lines are suitable for 3D visualisation.

    arguments:
       line_list (list of numpy arrays of floats): the undraped poly-lines
       rod_list_list: (list of arrays of pairs of integers): as returned by nearest_rods()
       grid: (grid.Grid object): the grid to which the poly-lines are to be draped
       axis (string): 'I' or 'J' being the axis removed during slicing
       ref_slice0 (integer, default 0): the reference slice in the grid to drape to; zero based
       plus_face (boolean, default False): which face of the reference slice to use
       offset (float, default -1.0): the horzontal offset to add to the y or x value of rod points
       snap (boolean, default False): if True, the x & z or y & z values of each vertex in the lines
          are moved to match the rod point; if False, only the y or x values are adjusted

    returns:
       list of numpy arrays of floats, being the draped equivalents of the line_list

    notes:
       the units of the line list must implicitly be those of the crs for the grid;
       currently limited to unsplit grids with no k gaps;
       the results of this function are intended for 3D visualisation of an indicative nature;
       resulting draped lines may penetrate the grid layer faces depending on the 3D geometry and
       the spacing of the vertices compared to cell sizes
    """

    assert projection in ['xz', 'yz']
    assert axis.upper() in ['I', 'J']
    assert len(line_list) == len(rod_list_list)

    x_sect = grid.unsplit_x_section_points(axis, ref_slice0 = ref_slice0, plus_face = plus_face)

    draped_list = []
    for line, rod_list in zip(line_list, rod_list_list):
        drape = line.copy()
        p_count = line.shape[0]
        for p in range(p_count):
            kj_or_ki = rod_list[p]
            if snap:
                drape[p, :] = x_sect[tuple(kj_or_ki)]
            elif projection == 'xz':
                drape[p, 1] = x_sect[tuple(kj_or_ki)][1]
            else:
                drape[p, 0] = x_sect[tuple(kj_or_ki)][0]
        if projection == 'xz':
            drape[:, 1] += offset
        else:
            drape[:, 0] += offset
        draped_list.append(drape)

    return draped_list
