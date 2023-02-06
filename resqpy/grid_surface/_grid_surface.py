"""Functions relating to intersection of resqml grid with surface or trajectory objects."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.olio.intersection as meet
import resqpy.olio.vector_utilities as vec
import resqpy.surface as rqs


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
