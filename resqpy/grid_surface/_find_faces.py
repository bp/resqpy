"""Functions for finding grid cell faces to represent a surface."""

import logging

log = logging.getLogger(__name__)

import numpy as np
import warnings
import numba  # type: ignore
from numba import njit, prange  # type: ignore
from typing import Tuple, Union, Dict

import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.fault as rqf
import resqpy.property as rqp
import resqpy.weights_and_measures as wam
import resqpy.olio.box_utilities as bx
import resqpy.olio.intersection as meet
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec

# note: resqpy.grid_surface._grid_surface_cuda will be imported by the find_faces_to_represent_surface() function if needed


def find_faces_to_represent_surface_staffa(grid, surface, name, feature_type = "fault", progress_fn = None):
    """Returns a grid connection set containing those cell faces which are deemed to represent the surface.

    note:
        this version of the find faces function is designed for irregular grids
    """

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
        log.debug("converting from surface crs to grid crs")
        s_crs = rqc.Crs(surface.model, uuid = surface.crs_uuid)
        s_crs.convert_array_to(grid.crs, p)
    triangles = p[t]
    assert triangles.size > 0, "no triangles in surface"
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
                        full_intersects = meet.line_set_triangles_intersects(
                            centre_points[:-1, j, i],
                            k_vectors[:, j, i],
                            triangles[ti_base:ti_end],
                            line_segment = True,
                        )
                        distilled_intersects, _, _ = meet.distilled_intersects(full_intersects)
                        k_faces[distilled_intersects, j, i] = True
                    if (j < grid.nj - 1 and column_j_vector_boxes is not None and
                            bx.boxes_overlap(batch_box, column_j_vector_boxes[j, i])):
                        full_intersects = meet.line_set_triangles_intersects(
                            centre_points[:, j, i],
                            j_vectors[:, j, i],
                            triangles[ti_base:ti_end],
                            line_segment = True,
                        )
                        distilled_intersects, _, _ = meet.distilled_intersects(full_intersects)
                        j_faces[distilled_intersects, j, i] = True
                    if (i < grid.ni - 1 and column_i_vector_boxes is not None and
                            bx.boxes_overlap(batch_box, column_i_vector_boxes[j, i])):
                        full_intersects = meet.line_set_triangles_intersects(
                            centre_points[:, j, i],
                            i_vectors[:, j, i],
                            triangles[ti_base:ti_end],
                            line_segment = True,
                        )
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
    gcs = rqf.GridConnectionSet(
        grid.model,
        grid = grid,
        k_faces = k_faces,
        j_faces = j_faces,
        i_faces = i_faces,
        feature_name = name,
        feature_type = feature_type,
        create_organizing_objects_where_needed = True,
    )

    if progress_fn is not None:
        progress_fn(1.0)

    return gcs


def find_faces_to_represent_surface_regular(
    grid,
    surface,
    name,
    title = None,
    centres = None,
    agitate = False,
    random_agitation = False,
    feature_type = "fault",
    progress_fn = None,
    consistent_side = False,
    return_properties = None,
):
    """Returns a grid connection set containing those cell faces which are deemed to represent the surface.

    arguments:
        grid (RegularGrid): the grid for which to create a grid connection set representation of the surface
        surface (Surface): the surface to be intersected with the grid
        name (str): the feature name to use in the grid connection set
        title (str, optional): the citation title to use for the grid connection set; defaults to name
        centres (numpy float array of shape (nk, nj, ni, 3), optional): precomputed cell centre points in
           local grid space, to avoid possible crs issues; required if grid's crs includes an origin (offset)?
        agitate (bool, default False): if True, the points of the surface are perturbed by a small
           offset, which can help if the surface has been built from a regular mesh with a periodic resonance
           with the grid
        random_agitation (bool, default False): if True, the agitation is by a small random distance; if False,
           a constant positive shift of 5.0e-6 is applied to x, y & z values; ignored if agitate is False
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
        use find_faces_to_represent_surface_regular_optimised() instead, for faster computation;
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
        assert all([p in ["offset", "normal vector"] for p in return_properties])
        return_normal_vectors = "normal vector" in return_properties
        return_offsets = "offset" in return_properties

    if title is None:
        title = name

    if progress_fn is not None:
        progress_fn(0.0)

    log.debug(f"intersecting surface {surface.title} with regular grid {grid.title}")
    log.debug(f"grid extent kji: {grid.extent_kji}")

    grid_dxyz = (
        grid.block_dxyz_dkji[2, 0],
        grid.block_dxyz_dkji[1, 1],
        grid.block_dxyz_dkji[0, 2],
    )
    if centres is None:
        centres = grid.centre_point(use_origin = True)
    if consistent_side:
        log.debug("making all triangles clockwise")
        # note: following will shuffle order of vertices within t cached in surface
        surface.make_all_clockwise_xy(reorient = True)
    t, p = surface.triangles_and_points()
    assert t is not None and p is not None, f"surface {surface.title} is empty"
    if agitate:
        if random_agitation:
            p += 1.0e-5 * (np.random.random(p.shape) - 0.5)
        else:
            p += 5.0e-6
    log.debug(f"surface: {surface.title}; p0: {p[0]}; crs uuid: {surface.crs_uuid}")
    log.debug(f"surface min xyz: {np.min(p, axis = 0)}")
    log.debug(f"surface max xyz: {np.max(p, axis = 0)}")
    if not bu.matching_uuids(grid.crs_uuid, surface.crs_uuid):
        log.debug("converting from surface crs to grid crs")
        s_crs = rqc.Crs(surface.model, uuid = surface.crs_uuid)
        s_crs.convert_array_to(grid.crs, p)
        surface.crs_uuid = grid.crs.uuid
        log.debug(f"surface: {surface.title}; p0: {p[0]}; crs uuid: {surface.crs_uuid}")
        log.debug(f"surface min xyz: {np.min(p, axis = 0)}")
        log.debug(f"surface max xyz: {np.max(p, axis = 0)}")

    log.debug(f"centres min xyz: {np.min(centres.reshape((-1, 3)), axis = 0)}")
    log.debug(f"centres max xyz: {np.max(centres.reshape((-1, 3)), axis = 0)}")

    t_count = len(t)

    # todo: batch up either centres or triangles to reduce memory requirement for large models

    # K direction (xy projection)
    if grid.nk > 1:
        log.debug("searching for k faces")
        k_faces = np.zeros((grid.nk - 1, grid.nj, grid.ni), dtype = bool)
        k_sides = np.zeros((grid.nk - 1, grid.nj, grid.ni), dtype = bool)
        k_offsets = (np.full((grid.nk - 1, grid.nj, grid.ni), np.nan) if return_offsets else None)
        k_normals = (np.full((grid.nk - 1, grid.nj, grid.ni, 3), np.nan) if return_normal_vectors else None)
        k_centres = centres[0, :, :].reshape((-1, 3))
        k_hits = vec.points_in_triangles(p, t, k_centres, projection = "xy", edged = True).reshape(
            (t_count, grid.nj, grid.ni))
        del k_centres
        if consistent_side:
            cwt = vec.clockwise_triangles(p, t, projection = "xy") >= 0.0
        for k_t, k_j, k_i in np.stack(np.where(k_hits), axis = -1):
            xyz = meet.line_triangle_intersect(
                centres[0, k_j, k_i],
                centres[-1, k_j, k_i] - centres[0, k_j, k_i],
                p[t[k_t]],
                line_segment = True,
                t_tol = 1.0e-6,
            )
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
                k_offsets[k_face, k_j,
                          k_i] = xyz[2] - 0.5 * (centres[k_face, k_j, k_i, 2] + centres[k_face + 1, k_j, k_i, 2])
            if return_normal_vectors:
                k_normals[k_face, k_j, k_i] = vec.triangle_normal_vector(p[t[k_t]])
                # todo: if consistent side, could deliver information about horizon surface inversion
                if k_normals[k_face, k_j, k_i, 2] > 0.0:
                    k_normals[k_face, k_j, k_i] = -k_normals[k_face, k_j, k_i]  # -ve z hemisphere normal
        del k_hits
        log.debug(f"k face count: {np.count_nonzero(k_faces)}")
    else:
        k_faces = None
        k_sides = None
        k_normals = None

    if progress_fn is not None:
        progress_fn(0.3)

    # J direction (xz projection)
    if grid.nj > 1:
        log.debug("searching for j faces")
        j_faces = np.zeros((grid.nk, grid.nj - 1, grid.ni), dtype = bool)
        j_sides = np.zeros((grid.nk, grid.nj - 1, grid.ni), dtype = bool)
        j_offsets = (np.full((grid.nk, grid.nj - 1, grid.ni), np.nan) if return_offsets else None)
        j_normals = (np.full((grid.nk, grid.nj - 1, grid.ni, 3), np.nan) if return_normal_vectors else None)
        j_centres = centres[:, 0, :].reshape((-1, 3))
        j_hits = vec.points_in_triangles(p, t, j_centres, projection = "xz", edged = True).reshape(
            (t_count, grid.nk, grid.ni))
        del j_centres
        if consistent_side:
            cwt = vec.clockwise_triangles(p, t, projection = "xz") >= 0.0
        for j_t, j_k, j_i in np.stack(np.where(j_hits), axis = -1):
            xyz = meet.line_triangle_intersect(
                centres[j_k, 0, j_i],
                centres[j_k, -1, j_i] - centres[j_k, 0, j_i],
                p[t[j_t]],
                line_segment = True,
                t_tol = 1.0e-6,
            )
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
                j_offsets[j_k, j_face,
                          j_i] = xyz[1] - 0.5 * (centres[j_k, j_face, j_i, 1] + centres[j_k, j_face + 1, j_i, 1])
            if return_normal_vectors:
                j_normals[j_k, j_face, j_i] = vec.triangle_normal_vector(p[t[j_t]])
                if j_normals[j_k, j_face, j_i, 2] > 0.0:
                    j_normals[j_k, j_face, j_i] = -j_normals[j_k, j_face, j_i]  # -ve z hemisphere normal
        del j_hits
        log.debug(f"j face count: {np.count_nonzero(j_faces)}")
    else:
        j_faces = None
        j_sides = None
        j_normals = None

    if progress_fn is not None:
        progress_fn(0.6)

    # I direction (yz projection)
    if grid.ni > 1:
        log.debug("searching for i faces")
        i_faces = np.zeros((grid.nk, grid.nj, grid.ni - 1), dtype = bool)
        i_sides = np.zeros((grid.nk, grid.nj, grid.ni - 1), dtype = bool)
        i_offsets = (np.full((grid.nk, grid.nj, grid.ni - 1), np.nan) if return_offsets else None)
        i_normals = (np.full((grid.nk, grid.nj, grid.ni - 1, 3), np.nan) if return_normal_vectors else None)
        i_centres = centres[:, :, 0].reshape((-1, 3))
        i_hits = vec.points_in_triangles(p, t, i_centres, projection = "yz", edged = True).reshape(
            (t_count, grid.nk, grid.nj))
        del i_centres
        if consistent_side:
            cwt = vec.clockwise_triangles(p, t, projection = "yz") >= 0.0
        for i_t, i_k, i_j in np.stack(np.where(i_hits), axis = -1):
            xyz = meet.line_triangle_intersect(
                centres[i_k, i_j, 0],
                centres[i_k, i_j, -1] - centres[i_k, i_j, 0],
                p[t[i_t]],
                line_segment = True,
                t_tol = 1.0e-6,
            )
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
                i_offsets[i_k, i_j,
                          i_face] = xyz[0] - 0.5 * (centres[i_k, i_j, i_face, 0] + centres[i_k, i_j, i_face + 1, 0])
            if return_normal_vectors:
                i_normals[i_k, i_j, i_face] = vec.triangle_normal_vector(p[t[i_t]])
                if i_normals[i_k, i_j, i_face, 2] > 0.0:
                    i_normals[i_k, i_j, i_face] = -i_normals[i_k, i_j, i_face]  # -ve z hemisphere normal
        del i_hits
        log.debug(f"i face count: {np.count_nonzero(i_faces)}")
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

    log.debug("converting face sets into grid connection set")
    gcs = rqf.GridConnectionSet(
        grid.model,
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
        create_organizing_objects_where_needed = True,
    )

    # NB. following assumes faces have been added to gcs in a particular order!
    if return_offsets:
        k_offsets_list = (np.empty((0,)) if k_offsets is None else k_offsets[np.where(k_faces)])
        j_offsets_list = (np.empty((0,)) if j_offsets is None else j_offsets[np.where(j_faces)])
        i_offsets_list = (np.empty((0,)) if i_offsets is None else i_offsets[np.where(i_faces)])
        all_offsets = _all_offsets(grid.crs, k_offsets_list, j_offsets_list, i_offsets_list)
        log.debug(f"gcs count: {gcs.count}; all offsets shape: {all_offsets.shape}")
        assert all_offsets.shape == (gcs.count,)

    # NB. following assumes faces have been added to gcs in a particular order!
    if return_normal_vectors:
        k_normals_list = (np.empty((0, 3)) if k_normals is None else k_normals[np.where(k_faces)])
        j_normals_list = (np.empty((0, 3)) if j_normals is None else j_normals[np.where(j_faces)])
        i_normals_list = (np.empty((0, 3)) if i_normals is None else i_normals[np.where(i_faces)])
        all_normals = np.concatenate((k_normals_list, j_normals_list, i_normals_list), axis = 0)
        log.debug(f"gcs count: {gcs.count}; all normals shape: {all_normals.shape}")
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
            props_dict["offset"] = all_offsets
        if return_normal_vectors:
            props_dict["normal vector"] = all_normals
        return (gcs, props_dict)

    return gcs


def find_faces_to_represent_surface_regular_dense_optimised(grid,
                                                            surface,
                                                            name,
                                                            title = None,
                                                            agitate = False,
                                                            random_agitation = False,
                                                            feature_type = "fault",
                                                            is_curtain = False,
                                                            progress_fn = None,
                                                            return_properties = None,
                                                            raw_bisector = False,
                                                            n_batches = 20):
    """Returns a grid connection set containing those cell faces which are deemed to represent the surface.

    argumants:
        grid (RegularGrid): the grid for which to create a grid connection set representation of the surface;
           must be aligned, ie. I with +x, J with +y, K with +z and local origin of (0.0, 0.0, 0.0)
        surface (Surface): the surface to be intersected with the grid
        name (str): the feature name to use in the grid connection set
        title (str, optional): the citation title to use for the grid connection set; defaults to name
        agitate (bool, default False): if True, the points of the surface are perturbed by a small
           offset, which can help if the surface has been built from a regular mesh with a periodic resonance
           with the grid
        random_agitation (bool, default False): if True, the agitation is by a small random distance; if False,
           a constant positive shift of 5.0e-6 is applied to x, y & z values; ignored if agitate is False
        feature_type (str, default 'fault'): 'fault', 'horizon' or 'geobody boundary'
        is_curtain (bool, default False): if True, only the top layer of the grid is processed and the bisector
           property, if requested, is generated with indexable element columns
        progress_fn (f(x: float), optional): a callback function to be called at intervals by this function;
           the argument will progress from 0.0 to 1.0 in unspecified and uneven increments
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
        n_batches (int, default 20): the number of batches of triangles to use at the low level (numba multi
           threading allows some parallelism between the batches)

    returns:
        gcs  or  (gcs, gcs_props)
        where gcs is a new GridConnectionSet with a single feature, not yet written to hdf5 nor xml created;
        gcs_props is a dictionary mapping from requested return_properties string to numpy array

    notes:
        this function is designed for aligned regular grids only;
        this function can handle the surface and grid being in different coordinate reference systems, as
        long as the implicit parent crs is shared;
        no trimming of the surface is carried out here: for computational efficiency, it is recommended
        to trim first;
        organisational objects for the feature are created if needed;
        if the offset return property is requested, the implicit units will be the z units of the grid's crs;
        this version of the function uses fully explicit boolean arrays to capture the faces before conversion
        to a grid connection set; use the non-dense version of the function for a reduced memory footprint;
        this function is DEPRECATED pending proving of newer find_faces_to_represent_surface_regular_optimised()
    """
    warnings.warn('DEPRECATED: grid_surface.find_faces_to_represent_surface_regular_dense_optimised() function; ' +
                  'use find_faces_to_represent_surface_regular_optimised() instead')

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
            p in [
                "triangle",
                "depth",
                "offset",
                "grid bisector",
                "grid shadow",
                "flange bool",
            ] for p in return_properties
        ])
        return_triangles = "triangle" in return_properties
        return_depths = "depth" in return_properties
        return_offsets = "offset" in return_properties
        return_bisector = "grid bisector" in return_properties
        return_shadow = "grid shadow" in return_properties
        return_flange_bool = "flange bool" in return_properties
        if return_flange_bool:
            return_triangles = True

    if title is None:
        title = name

    if progress_fn is not None:
        progress_fn(0.0)

    log.debug(f"intersecting surface {surface.title} with regular grid {grid.title}")
    # log.debug(f'grid extent kji: {grid.extent_kji}')

    grid_dxyz = (
        grid.block_dxyz_dkji[2, 0],
        grid.block_dxyz_dkji[1, 1],
        grid.block_dxyz_dkji[0, 2],
    )
    triangles, points = surface.triangles_and_points()
    t_dtype = np.int32 if len(triangles) < 2_000_000_000 else np.int64
    assert (triangles is not None and points is not None), f"surface {surface.title} is empty"
    if agitate:
        points = points.copy()
        if random_agitation:
            points += 1.0e-5 * (np.random.random(points.shape) - 0.5)
        else:
            points += 5.0e-6
    # log.debug(f'surface: {surface.title}; p0: {points[0]}; crs uuid: {surface.crs_uuid}')
    # log.debug(f'surface min xyz: {np.min(points, axis = 0)}')
    # log.debug(f'surface max xyz: {np.max(points, axis = 0)}')
    if not bu.matching_uuids(grid.crs_uuid, surface.crs_uuid):
        log.debug("converting from surface crs to grid crs")
        s_crs = rqc.Crs(surface.model, uuid = surface.crs_uuid)
        s_crs.convert_array_to(grid.crs, points)
        surface.crs_uuid = grid.crs.uuid
        # log.debug(f'surface: {surface.title}; p0: {points[0]}; crs uuid: {surface.crs_uuid}')
        # log.debug(f'surface min xyz: {np.min(points, axis = 0)}')
        # log.debug(f'surface max xyz: {np.max(points, axis = 0)}')

    nk = 1 if is_curtain else grid.nk
    # K direction (xy projection)
    if nk > 1:
        # log.debug("searching for k faces")
        k_faces = np.zeros((nk - 1, grid.nj, grid.ni), dtype = bool)
        k_triangles = np.full((nk - 1, grid.nj, grid.ni), -1, dtype = t_dtype)
        k_depths = np.full((nk - 1, grid.nj, grid.ni), np.nan)
        k_offsets = np.full((nk - 1, grid.nj, grid.ni), np.nan)
        p_xy = np.delete(points, 2, 1)

        k_hits = vec.points_in_triangles_aligned_optimised(grid.ni, grid.nj, grid_dxyz[0], grid_dxyz[1],
                                                           p_xy[triangles], n_batches)

        del p_xy
        axis = 2
        index1 = 1
        index2 = 2
        k_faces, k_offsets, k_triangles = intersect_numba(
            axis,
            index1,
            index2,
            k_hits,
            nk,
            points,
            triangles,
            grid_dxyz,
            k_faces,
            return_depths,
            k_depths,
            return_offsets,
            k_offsets,
            return_triangles,
            k_triangles,
        )
        del k_hits
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
        j_triangles = np.full((nk, grid.nj - 1, grid.ni), -1, dtype = t_dtype)
        j_depths = np.full((nk, grid.nj - 1, grid.ni), np.nan)
        j_offsets = np.full((nk, grid.nj - 1, grid.ni), np.nan)
        p_xz = np.delete(points, 1, 1)

        j_hits = vec.points_in_triangles_aligned_optimised(grid.ni, nk, grid_dxyz[0], grid_dxyz[2], p_xz[triangles],
                                                           n_batches)

        del p_xz
        axis = 1
        index1 = 0
        index2 = 2
        j_faces, j_offsets, j_triangles = intersect_numba(
            axis,
            index1,
            index2,
            j_hits,
            grid.nj,
            points,
            triangles,
            grid_dxyz,
            j_faces,
            return_depths,
            j_depths,
            return_offsets,
            j_offsets,
            return_triangles,
            j_triangles,
        )
        del j_hits
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
        i_triangles = np.full((nk, grid.nj, grid.ni - 1), -1, dtype = t_dtype)
        i_depths = np.full((nk, grid.nj, grid.ni - 1), np.nan)
        i_offsets = np.full((nk, grid.nj, grid.ni - 1), np.nan)
        p_yz = np.delete(points, 0, 1)

        i_hits = vec.points_in_triangles_aligned_optimised(grid.nj, nk, grid_dxyz[1], grid_dxyz[2], p_yz[triangles],
                                                           n_batches)

        del p_yz
        axis = 0
        index1 = 0
        index2 = 1
        i_faces, i_offsets, i_triangles = intersect_numba(
            axis,
            index1,
            index2,
            i_hits,
            grid.ni,
            points,
            triangles,
            grid_dxyz,
            i_faces,
            return_depths,
            i_depths,
            return_offsets,
            i_offsets,
            return_triangles,
            i_triangles,
        )
        del i_hits
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
        k_tri_list = (np.empty((0,)) if k_triangles is None else k_triangles[_where_true(k_faces)])
        j_tri_list = (np.empty((0,)) if j_triangles is None else j_triangles[_where_true(j_faces)])
        i_tri_list = (np.empty((0,)) if i_triangles is None else i_triangles[_where_true(i_faces)])
        all_tris = np.concatenate((k_tri_list, j_tri_list, i_tri_list), axis = 0)
        # log.debug(f'gcs count: {gcs.count}; all triangles shape: {all_tris.shape}')
        assert all_tris.shape == (gcs.count,)

    # NB. following assumes faces have been added to gcs in a particular order!
    if return_depths:
        # log.debug('preparing depths array')
        k_depths_list = (np.empty((0,)) if k_depths is None else k_depths[_where_true(k_faces)])
        j_depths_list = (np.empty((0,)) if j_depths is None else j_depths[_where_true(j_faces)])
        i_depths_list = (np.empty((0,)) if i_depths is None else i_depths[_where_true(i_faces)])
        all_depths = np.concatenate((k_depths_list, j_depths_list, i_depths_list), axis = 0)
        # log.debug(f'gcs count: {gcs.count}; all depths shape: {all_depths.shape}')
        assert all_depths.shape == (gcs.count,)

    # NB. following assumes faces have been added to gcs in a particular order!
    if return_offsets:
        # log.debug('preparing offsets array')
        k_offsets_list = (np.empty((0,)) if k_offsets is None else k_offsets[_where_true(k_faces)])
        j_offsets_list = (np.empty((0,)) if j_offsets is None else j_offsets[_where_true(j_faces)])
        i_offsets_list = (np.empty((0,)) if i_offsets is None else i_offsets[_where_true(i_faces)])
        all_offsets = _all_offsets(grid.crs, k_offsets_list, j_offsets_list, i_offsets_list)
        # log.debug(f'gcs count: {gcs.count}; all offsets shape: {all_offsets.shape}')
        assert all_offsets.shape == (gcs.count,)

    if return_flange_bool:
        # log.debug('preparing flange array')
        flange_bool_uuid = surface.model.uuid(title = "flange bool",
                                              obj_type = "DiscreteProperty",
                                              related_uuid = surface.uuid)
        assert (flange_bool_uuid is not None), f"No flange bool property found for surface: {surface.title}"
        flange_bool = rqp.Property(surface.model, uuid = flange_bool_uuid)
        flange_array = flange_bool.array_ref(dtype = bool)
        all_flange = np.take(flange_array, all_tris)
        assert all_flange.shape == (gcs.count,)

    # note: following is a grid cells property, not a gcs property
    if return_bisector:
        if is_curtain:
            log.debug("preparing columns bisector")
            bisector = column_bisector_from_faces((grid.nj, grid.ni), j_faces[0], i_faces[0])
            # log.debug('finished preparing columns bisector')
        else:
            log.debug("preparing cells bisector")
            bisector, is_curtain = bisector_from_faces(tuple(grid.extent_kji), k_faces, j_faces, i_faces, raw_bisector)
            if is_curtain:
                bisector = bisector[0]  # reduce to a columns property

    # note: following is a grid cells property, not a gcs property
    if return_shadow:
        log.debug("preparing cells shadow")
        shadow = shadow_from_faces(tuple(grid.extent_kji), k_faces)

    if progress_fn is not None:
        progress_fn(1.0)

    log.debug(f"finishing find_faces_to_represent_surface_regular_dense_optimised for {name}")

    # if returning properties, construct dictionary
    if return_properties:
        props_dict = {}
        if return_triangles:
            props_dict["triangle"] = all_tris
        if return_depths:
            props_dict["depth"] = all_depths
        if return_offsets:
            props_dict["offset"] = all_offsets
        if return_bisector:
            props_dict["grid bisector"] = (bisector, is_curtain)
        if return_shadow:
            props_dict["grid shadow"] = shadow
        if return_flange_bool:
            props_dict["flange bool"] = all_flange
        return (gcs, props_dict)

    return gcs


def find_faces_to_represent_surface_regular_optimised(grid,
                                                      surface,
                                                      name,
                                                      title = None,
                                                      agitate = False,
                                                      random_agitation = False,
                                                      feature_type = "fault",
                                                      is_curtain = False,
                                                      progress_fn = None,
                                                      return_properties = None,
                                                      raw_bisector = False,
                                                      n_batches = 20,
                                                      packed_bisectors = False,
                                                      patch_indices = None):
    """Returns a grid connection set containing those cell faces which are deemed to represent the surface.

    argumants:
        grid (RegularGrid): the grid for which to create a grid connection set representation of the surface;
           must be aligned, ie. I with +x, J with +y, K with +z and local origin of (0.0, 0.0, 0.0)
        surface (Surface): the surface to be intersected with the grid
        name (str): the feature name to use in the grid connection set
        title (str, optional): the citation title to use for the grid connection set; defaults to name
        agitate (bool, default False): if True, the points of the surface are perturbed by a small
           offset, which can help if the surface has been built from a regular mesh with a periodic resonance
           with the grid
        random_agitation (bool, default False): if True, the agitation is by a small random distance; if False,
           a constant positive shift of 5.0e-6 is applied to x, y & z values; ignored if agitate is False
        feature_type (str, default 'fault'): 'fault', 'horizon' or 'geobody boundary'
        is_curtain (bool, default False): if True, only the top layer of the grid is processed and the bisector
           property, if requested, is generated with indexable element columns
        progress_fn (f(x: float), optional): a callback function to be called at intervals by this function;
           the argument will progress from 0.0 to 1.0 in unspecified and uneven increments
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
        n_batches (int, default 20): the number of batches of triangles to use at the low level (numba multi
           threading allows some parallelism between the batches)
        packed_bisectors (bool, default False): if True and return properties include 'grid bisector' then
           non curtain bisectors are returned in packed form
        patch_indices (numpy int array, optional): if present, an array over grid cells indicating which 
           patch of surface is applicable in terms of a bisector, for each cell

    returns:
        gcs  or  (gcs, gcs_props)
        where gcs is a new GridConnectionSet with a single feature, not yet written to hdf5 nor xml created;
        gcs_props is a dictionary mapping from requested return_properties string to numpy array (or tuple
            of numpy array and curtain bool in the case of grid bisector)

    notes:
        this function is designed for aligned regular grids only;
        this function can handle the surface and grid being in different coordinate reference systems, as
        long as the implicit parent crs is shared;
        no trimming of the surface is carried out here: for computational efficiency, it is recommended
        to trim first;
        organisational objects for the feature are created if needed;
        if the offset return property is requested, the implicit units will be the z units of the grid's crs;
        if patch_indices is present and grid bisectors are being returned, a composite bisector array is returned
        with elements set from individual bisectors for each patch of surface
    """

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
            p in [
                "triangle",
                "depth",
                "offset",
                "grid bisector",
                "grid shadow",
                "flange bool",
            ] for p in return_properties
        ])
        return_triangles = "triangle" in return_properties
        return_depths = "depth" in return_properties
        return_offsets = "offset" in return_properties
        return_bisector = "grid bisector" in return_properties
        return_shadow = "grid shadow" in return_properties
        return_flange_bool = "flange bool" in return_properties
        if return_flange_bool:
            return_triangles = True
    patchwork = return_bisector and patch_indices is not None
    if patchwork:
        return_triangles = True  # triangle numbers are used to infer patch index
        assert patch_indices.shape == tuple(grid.extent_kji)
    if title is None:
        title = name

    if progress_fn is not None:
        progress_fn(0.0)

    log.debug(f"intersecting surface {surface.title} with regular grid {grid.title}")
    # log.debug(f'grid extent kji: {grid.extent_kji}')

    triangles, points = surface.triangles_and_points(copy = True)
    surface.decache_triangles_and_points()

    t_dtype = np.int32 if len(triangles) < 2_147_483_648 else np.int64

    assert (triangles is not None and points is not None), f"surface {surface.title} is empty"
    if agitate:
        if random_agitation:
            points += 1.0e-5 * (np.random.random(points.shape) - 0.5)
        else:
            points += 5.0e-6
    # log.debug(f'surface: {surface.title}; p0: {points[0]}; crs uuid: {surface.crs_uuid}')
    # log.debug(f'surface min xyz: {np.min(points, axis = 0)}')
    # log.debug(f'surface max xyz: {np.max(points, axis = 0)}')
    if not bu.matching_uuids(grid.crs_uuid, surface.crs_uuid):
        log.debug("converting from surface crs to grid crs")
        s_crs = rqc.Crs(surface.model, uuid = surface.crs_uuid)
        s_crs.convert_array_to(grid.crs, points)
        surface.crs_uuid = grid.crs.uuid
        # log.debug(f'surface: {surface.title}; p0: {points[0]}; crs uuid: {surface.crs_uuid}')
        # log.debug(f'surface min xyz: {np.min(points, axis = 0)}')
        # log.debug(f'surface max xyz: {np.max(points, axis = 0)}')

    # convert surface points to work with unit cube grid cells
    dx = grid.block_dxyz_dkji[2, 0]
    dy = grid.block_dxyz_dkji[1, 1]
    dz = grid.block_dxyz_dkji[0, 2]
    points[:, 0] /= dx
    points[:, 1] /= dy
    points[:, 2] /= dz
    points[:] -= 0.5
    p = points[triangles]

    nk = 1 if is_curtain else grid.nk
    # K direction (xy projection)
    k_faces_kji0 = None
    k_triangles = None
    k_depths = None
    k_offsets = None
    k_props = None
    if nk > 1:
        # log.debug("searching for k faces")

        k_hits, k_depths = vec.points_in_triangles_aligned_unified(grid.ni, grid.nj, 0, 1, 2, p, n_batches)

        k_faces = np.floor(k_depths)
        mask = np.logical_and(k_faces >= 0, k_faces < nk - 1)

        if np.any(mask):
            k_hits = k_hits[mask, :]
            k_faces = k_faces[mask]
            k_depths = k_depths[mask]
            k_triangles = k_hits[:, 0]
            k_faces_kji0 = np.empty((len(k_faces), 3), dtype = np.int32)
            k_faces_kji0[:, 0] = k_faces
            k_faces_kji0[:, 1] = k_hits[:, 1]
            k_faces_kji0[:, 2] = k_hits[:, 2]
            if return_offsets:
                k_offsets = (k_depths - k_faces.astype(np.float64) - 0.5) * dz
            if return_depths:
                k_depths[:] += 0.5
                k_depths[:] *= dz
            k_props = []
            if return_triangles:
                k_props.append(k_triangles)
            if return_depths:
                k_props.append(k_depths)
            if return_offsets:
                k_props.append(k_offsets)
            log.debug(f"k face count: {len(k_faces_kji0)}")

        del k_hits
        del k_faces

    if progress_fn is not None:
        progress_fn(0.3)

    # J direction (xz projection)
    j_faces_kji0 = None
    j_triangles = None
    j_depths = None
    j_offsets = None
    j_props = None
    if grid.nj > 1:
        # log.debug("searching for J faces")

        j_hits, j_depths = vec.points_in_triangles_aligned_unified(grid.ni, nk, 0, 2, 1, p, n_batches)

        j_faces = np.floor(j_depths)
        mask = np.logical_and(j_faces >= 0, j_faces < grid.nj - 1)

        if np.any(mask):
            j_hits = j_hits[mask, :]
            j_faces = j_faces[mask]
            j_depths = j_depths[mask]
            j_triangles = j_hits[:, 0]
            j_faces_kji0 = np.empty((len(j_faces), 3), dtype = np.int32)
            j_faces_kji0[:, 0] = j_hits[:, 1]
            j_faces_kji0[:, 1] = j_faces
            j_faces_kji0[:, 2] = j_hits[:, 2]
            if return_offsets:
                j_offsets = (j_depths - j_faces.astype(np.float64) - 0.5) * dy
            if return_depths:
                j_depths[:] += 0.5
                j_depths[:] *= dy
            if is_curtain and grid.nk > 1:  # expand arrays to all layers
                j_faces = np.repeat(np.expand_dims(j_faces_kji0, axis = 0), grid.nk, axis = 0)
                j_faces[:, :, 0] = np.expand_dims(np.arange(grid.nk, dtype = np.int32), axis = -1)
                j_faces_kji0 = j_faces.reshape((-1, 3))
                j_triangles = np.repeat(j_triangles, grid.nk, axis = 0)
                if return_offsets:
                    j_offsets = np.repeat(j_offsets, grid.nk, axis = 0)
                if return_depths:
                    j_depths = np.repeat(j_depths, grid.nk, axis = 0)
            j_props = []
            if return_triangles:
                j_props.append(j_triangles)
            if return_depths:
                j_props.append(j_depths)
            if return_offsets:
                j_props.append(j_offsets)
            log.debug(f"j face count: {len(j_faces_kji0)}")

        del j_hits
        del j_faces

    if progress_fn is not None:
        progress_fn(0.6)

    # I direction (yz projection)
    i_faces_kji0 = None
    i_triangles = None
    i_depths = None
    i_offsets = None
    i_props = None
    if grid.ni > 1:
        # log.debug("searching for I faces")

        i_hits, i_depths = vec.points_in_triangles_aligned_unified(grid.nj, nk, 1, 2, 0, p, n_batches)

        i_faces = np.floor(i_depths)
        mask = np.logical_and(i_faces >= 0, i_faces < grid.ni - 1)

        if np.any(mask):
            i_hits = i_hits[mask, :]
            i_faces = i_faces[mask]
            i_depths = i_depths[mask]
            i_triangles = i_hits[:, 0]
            i_faces_kji0 = np.empty((len(i_faces), 3), dtype = np.int32)
            i_faces_kji0[:, 0] = i_hits[:, 1]
            i_faces_kji0[:, 1] = i_hits[:, 2]
            i_faces_kji0[:, 2] = i_faces
            if return_offsets:
                i_offsets = (i_depths - i_faces.astype(np.float64) - 0.5) * dx
            if return_depths:
                i_depths[:] += 0.5
                i_depths[:] *= dx
            if is_curtain and grid.nk > 1:  # expand arrays to all layers
                i_faces = np.repeat(np.expand_dims(i_faces_kji0, axis = 0), grid.nk, axis = 0)
                i_faces[:, :, 0] = np.expand_dims(np.arange(grid.nk, dtype = np.int32), axis = -1)
                i_faces_kji0 = i_faces.reshape((-1, 3))
                i_triangles = np.repeat(i_triangles, grid.nk, axis = 0)
                if return_offsets:
                    i_offsets = np.repeat(i_offsets, grid.nk, axis = 0)
                if return_depths:
                    i_depths = np.repeat(i_depths, grid.nk, axis = 0)
            i_props = []
            if return_triangles:
                i_props.append(i_triangles)
            if return_depths:
                i_props.append(i_depths)
            if return_offsets:
                i_props.append(i_offsets)
            log.debug(f"i face count: {len(i_faces_kji0)}")

        del i_hits
        del i_faces

    if progress_fn is not None:
        progress_fn(0.9)

    if ((k_faces_kji0 is None or len(k_faces_kji0) == 0) and (j_faces_kji0 is None or len(j_faces_kji0) == 0) and
        (i_faces_kji0 is None or len(i_faces_kji0) == 0)):
        log.error(f'did not find any faces to represent {name}: surface does not intersect grid?')
        if return_properties:
            return (None, {})
        else:
            return None

    log.debug("converting face sets into grid connection set")
    # NB: kji0 arrays in internal face protocol: used as cell_kji0 with polarity of 1
    # property lists have elements replaced with sorted and filtered equivalents
    gcs = rqf.GridConnectionSet.from_faces_indices(grid = grid,
                                                   k_faces_kji0 = k_faces_kji0,
                                                   j_faces_kji0 = j_faces_kji0,
                                                   i_faces_kji0 = i_faces_kji0,
                                                   remove_duplicates = not patchwork,
                                                   k_properties = k_props,
                                                   j_properties = j_props,
                                                   i_properties = i_props,
                                                   feature_name = name,
                                                   feature_type = feature_type,
                                                   create_organizing_objects_where_needed = True,
                                                   title = title)
    # log.debug('finished coversion to gcs')

    # NB. following assumes faces have been added to gcs in a particular order!
    all_tris = None
    if return_triangles:
        # log.debug('preparing triangles array')
        k_triangles = np.empty((0,), dtype = np.int32) if k_props is None else k_props.pop(0)
        j_triangles = np.empty((0,), dtype = np.int32) if j_props is None else j_props.pop(0)
        i_triangles = np.empty((0,), dtype = np.int32) if i_props is None else i_props.pop(0)
        all_tris = np.concatenate((k_triangles, j_triangles, i_triangles), axis = 0)
        # log.debug(f'gcs count: {gcs.count}; all triangles shape: {all_tris.shape}')
        assert all_tris.shape == (gcs.count,)

    # NB. following assumes faces have been added to gcs in a particular order!
    all_depths = None
    if return_depths:
        # log.debug('preparing depths array')
        k_depths = np.empty((0,), dtype = np.float64) if k_props is None else k_props.pop(0)
        j_depths = np.empty((0,), dtype = np.float64) if j_props is None else j_props.pop(0)
        i_depths = np.empty((0,), dtype = np.float64) if i_props is None else i_props.pop(0)
        all_depths = np.concatenate((k_depths, j_depths, i_depths), axis = 0)
        # log.debug(f'gcs count: {gcs.count}; all depths shape: {all_depths.shape}')
        assert all_depths.shape == (gcs.count,)

    # NB. following assumes faces have been added to gcs in a particular order!
    all_offsets = None
    if return_offsets:
        # log.debug('preparing offsets array')
        k_offsets = np.empty((0,), dtype = np.float64) if k_props is None else k_props[0]
        j_offsets = np.empty((0,), dtype = np.float64) if j_props is None else j_props[0]
        i_offsets = np.empty((0,), dtype = np.float64) if i_props is None else i_props[0]
        all_offsets = _all_offsets(grid.crs, k_offsets, j_offsets, i_offsets)
        # log.debug(f'gcs count: {gcs.count}; all offsets shape: {all_offsets.shape}')
        assert all_offsets.shape == (gcs.count,)

    all_flange = None
    if return_flange_bool:
        # log.debug('preparing flange array')
        flange_bool_uuid = surface.model.uuid(title = "flange bool",
                                              obj_type = "DiscreteProperty",
                                              related_uuid = surface.uuid)
        assert (flange_bool_uuid is not None), f"No flange bool property found for surface: {surface.title}"
        flange_bool = rqp.Property(surface.model, uuid = flange_bool_uuid)
        flange_array = flange_bool.array_ref(dtype = bool)
        all_flange = np.take(flange_array, all_tris)
        assert all_flange.shape == (gcs.count,)

    # note: following is a grid cells property, not a gcs property
    bisector = None
    if return_bisector:
        if is_curtain and not patchwork:
            log.debug(f'preparing columns bisector for: {surface.title}')
            if j_faces_kji0 is None:
                j_faces_ji0 = np.empty((0, 2), dtype = np.int32)
            else:
                j_faces_ji0 = j_faces_kji0[:, 1:]
            if i_faces_kji0 is None:
                i_faces_ji0 = np.empty((0, 2), dtype = np.int32)
            else:
                i_faces_ji0 = i_faces_kji0[:, 1:]
            bisector = column_bisector_from_face_indices((grid.nj, grid.ni), j_faces_ji0, i_faces_ji0)
            # log.debug('finished preparing columns bisector')
        elif patchwork:
            n_patches = surface.number_of_patches()
            log.info(f'surface: {surface.title}; number of patches: {n_patches}')
            nkf = 0 if k_faces_kji0 is None else len(k_faces_kji0)
            njf = 0 if j_faces_kji0 is None else len(j_faces_kji0)
            nif = 0 if i_faces_kji0 is None else len(i_faces_kji0)
            # fetch patch indices for triangle hits
            assert all_tris is not None and len(all_tris) == nkf + njf + nif
            patch_indices_k = surface.patch_indices_for_triangle_indices(all_tris[:nkf])
            patch_indices_j = surface.patch_indices_for_triangle_indices(all_tris[nkf:nkf + njf])
            patch_indices_i = surface.patch_indices_for_triangle_indices(all_tris[nkf + njf:])
            # add extra dimension to bisector array (at axis 0) for patches
            pb_shape = tuple([n_patches] + list(grid.extent_kji))
            if packed_bisectors:
                bisector = np.invert(np.zeros(_shape_packed(grid.extent_kji), dtype = np.uint8), dtype = np.uint8)
            else:
                bisector = np.ones(tuple(grid.extent_kji), dtype = np.bool_)
            # populate composite bisector
            for patch in range(n_patches):
                log.debug(f'processing patch {patch} of surface: {surface.title}')
                mask = (patch_indices == patch)
                mask_count = np.count_nonzero(mask)
                if mask_count == 0:
                    log.warning(f'patch {patch} of surface {surface.title} is not applicable to any cells in grid')
                    continue
                patch_box, box_count = get_box(mask)
                assert box_count == mask_count
                assert np.all(patch_box[1] > patch_box[0])
                patch_box = expanded_box(patch_box, tuple(grid.extent_kji))
                patch_box[0, 0] = 0
                patch_box[1, 0] = grid.extent_kji[0]
                patch_k_faces_kji0 = None
                if k_faces_kji0 is not None:
                    patch_k_faces_kji0 = k_faces_kji0[(patch_indices_k == patch).astype(bool)]
                patch_j_faces_kji0 = None
                if j_faces_kji0 is not None:
                    patch_j_faces_kji0 = j_faces_kji0[(patch_indices_j == patch).astype(bool)]
                patch_i_faces_kji0 = None
                if i_faces_kji0 is not None:
                    patch_i_faces_kji0 = i_faces_kji0[(patch_indices_i == patch).astype(bool)]
                if packed_bisectors:
                    mask = np.packbits(mask, axis = -1)
                    patch_bisector, is_curtain =  \
                        packed_bisector_from_face_indices(tuple(grid.extent_kji),
                                                          patch_k_faces_kji0,
                                                          patch_j_faces_kji0,
                                                          patch_i_faces_kji0,
                                                          raw_bisector,
                                                          patch_box)
                    bisector[:] = np.bitwise_or(np.bitwise_and(mask, patch_bisector),
                                                np.bitwise_and(np.invert(mask, dtype = np.uint8), bisector))
                else:
                    patch_bisector, is_curtain =  \
                        bisector_from_face_indices(tuple(grid.extent_kji),
                                                   patch_k_faces_kji0,
                                                   patch_j_faces_kji0,
                                                   patch_i_faces_kji0,
                                                   raw_bisector,
                                                   patch_box)
                    bisector[mask] = patch_bisector[mask]
                if is_curtain:
                    # TODO: downgrade following to debug once downstream functionality tested
                    log.warning(f'ignoring curtain nature of bisector for patch {patch} of surface: {surface.title}')
                    is_curtain = False
        else:
            log.info(f'preparing singlular cells bisector for surface: {surface.title}')  # could downgrade to debug
            if ((k_faces_kji0 is None or len(k_faces_kji0) == 0) and
                (j_faces_kji0 is None or len(j_faces_kji0) == 0) and (i_faces_kji0 is None or len(i_faces_kji0) == 0)):
                bisector = np.ones((grid.nj, grid.ni), dtype = bool)
                is_curtain = True
            elif packed_bisectors:
                bisector, is_curtain = packed_bisector_from_face_indices(tuple(grid.extent_kji), k_faces_kji0,
                                                                         j_faces_kji0, i_faces_kji0, raw_bisector, None)
                if is_curtain:
                    bisector = np.unpackbits(bisector[0], axis = -1,
                                             count = grid.ni).astype(bool)  # reduce to a columns property
            else:
                bisector, is_curtain = bisector_from_face_indices(tuple(grid.extent_kji), k_faces_kji0, j_faces_kji0,
                                                                  i_faces_kji0, raw_bisector, None)
                if is_curtain:
                    bisector = bisector[0]  # reduce to a columns property

    # note: following is a grid cells property, not a gcs property
    shadow = None
    if return_shadow:
        log.debug("preparing cells shadow")
        shadow = shadow_from_face_indices(tuple(grid.extent_kji), k_faces_kji0)

    if progress_fn is not None:
        progress_fn(1.0)

    log.debug(f"finishing find_faces_to_represent_surface_regular_optimised for {name}")

    # if returning properties, construct dictionary
    if return_properties:
        props_dict = {}
        if 'triangle' in return_properties:
            props_dict["triangle"] = all_tris
        if return_depths:
            props_dict["depth"] = all_depths
        if return_offsets:
            props_dict["offset"] = all_offsets
        if return_bisector:
            props_dict["grid bisector"] = (bisector, is_curtain)
        if return_shadow:
            props_dict["grid shadow"] = shadow
        if return_flange_bool:
            props_dict["flange bool"] = all_flange
        return (gcs, props_dict)

    return gcs


def find_faces_to_represent_surface(grid, surface, name, mode = "auto", feature_type = "fault", progress_fn = None):
    """Returns a grid connection set containing those cell faces which are deemed to represent the surface.

    arguments:
        grid (RegularGrid): the grid for which to create a grid connection set representation of the surface
        surface (Surface): the triangulated surface for which grid cell faces are required
        name (str): the feature name to use in the grid connection set
        mode (str, default 'auto'): one of 'auto', 'staffa', 'regular', 'regular_optimised', 'regular_cuda';
           auto will translate to regular_optimised for regulat grids, and staffa for irregular grids;
           regular_cude required GPU hardware and the correct installation of numba.cuda and cupy
        feature_type (str, default 'fault'): 'fault', 'horizon' or 'geobody boundary'
        progress_fn (f(x: float), optional): a callback function to be called at intervals by this function;
           the argument will progress from 0.0 to 1.0 in unspecified and uneven increments

    returns:
        a new GridConnectionSet with a single feature, not yet written to hdf5 nor xml created

    note:
        this is a wrapper function selecting between more specialised functions; use those directly for more options
    """

    log.debug("finding cell faces for surface")
    if mode == "auto":
        if isinstance(grid, grr.RegularGrid) and grid.is_aligned:
            mode = "regular_optimised"
        else:
            mode = "staffa"
    if mode == "regular_optimised":
        return find_faces_to_represent_surface_regular_optimised(grid,
                                                                 surface,
                                                                 name,
                                                                 feature_type = feature_type,
                                                                 progress_fn = progress_fn)
    elif mode == "regular_cuda":
        import resqpy.grid_surface.grid_surface_cuda as rgs_c

        return rgs_c.find_faces_to_represent_surface_regular_cuda_mgpu(grid,
                                                                       surface,
                                                                       name,
                                                                       feature_type = feature_type,
                                                                       progress_fn = progress_fn)
    elif mode == "staffa":
        return find_faces_to_represent_surface_staffa(grid,
                                                      surface,
                                                      name,
                                                      feature_type = feature_type,
                                                      progress_fn = progress_fn)
    elif mode == "regular_dense":
        return find_faces_to_represent_surface_regular_dense_optimised(grid,
                                                                       surface,
                                                                       name,
                                                                       feature_type = feature_type,
                                                                       progress_fn = progress_fn)
    elif mode == "regular":
        return find_faces_to_represent_surface_regular(grid,
                                                       surface,
                                                       name,
                                                       feature_type = feature_type,
                                                       progress_fn = progress_fn)
    log.critical("unrecognised mode: " + str(mode))
    return None


def bisector_from_faces(  # type: ignore
        grid_extent_kji: Tuple[int, int, int], k_faces: Union[np.ndarray, None], j_faces: Union[np.ndarray, None],
        i_faces: Union[np.ndarray, None], raw_bisector: bool) -> Tuple[np.ndarray, bool]:
    """Creates a boolean array denoting the bisection of the grid by the face sets.

    arguments:
        - grid_extent_kji (Tuple[int, int, int]): the shape of the grid
        - k_faces (np.ndarray): a boolean array of which faces represent the surface in the k dimension
        - j_faces (np.ndarray): a boolean array of which faces represent the surface in the j dimension
        - i_faces (np.ndarray): a boolean array of which faces represent the surface in the i dimension
        - raw_bisector (bool): if True, the bisector is returned without determining which side is shallower

    returns:
        Tuple containing:
        - array (np.ndarray): boolean bisectors array where values are True for cells on the side
          of the surface that has a lower mean k index on average and False for cells on the other side.
        - is_curtain (bool): True if the surface is a curtain (vertical), otherwise False.

    notes:
        - the face sets must form a single 'sealed' cut of the grid (eg. not waving in and out of the grid)
        - any 'boxed in' parts of the grid (completely enclosed by bisecting faces) will be consistently
          assigned to either the True or False part
        - this function is DEPRECATED, use newer indices based approach instead: bisector_from_face_indices()
    """
    warnings.warn('DEPRECATED: grid_surface.bisector_from_faces() function; use bisector_from_face_indices() instead')
    assert len(grid_extent_kji) == 3

    # find the surface boundary (includes a buffer slice where surface does not reach edge of grid)
    box = get_boundary(k_faces, j_faces, i_faces, grid_extent_kji)
    box_shape = box[1, :] - box[0, :]

    # set up the bisector array for the bounding box
    box_array = np.zeros(box_shape, dtype = np.bool_)

    # seed the bisector box array at (0, 0, 0)
    box_array[0, 0, 0] = True

    # prepare to spread True values to neighbouring cells that are not the other side of a face
    if k_faces is None:
        open_k = np.ones((box_shape[0] - 1, box_shape[1], box_shape[2]), dtype = bool)
    else:
        k_faces = k_faces[box[0, 0]:box[1, 0] - 1, box[0, 1]:box[1, 1], box[0, 2]:box[1, 2]]
        open_k = np.logical_not(k_faces)
    if j_faces is None:
        open_j = np.ones((box_shape[0], box_shape[1] - 1, box_shape[2]), dtype = bool)
    else:
        j_faces = j_faces[box[0, 0]:box[1, 0], box[0, 1]:box[1, 1] - 1, box[0, 2]:box[1, 2]]
        open_j = np.logical_not(j_faces)
    if i_faces is None:
        open_i = np.ones((box_shape[0], box_shape[1], box_shape[2] - 1), dtype = bool)
    else:
        i_faces = i_faces[box[0, 0]:box[1, 0], box[0, 1]:box[1, 1], box[0, 2]:box[1, 2] - 1]
        open_i = np.logical_not(i_faces)

    # populate bisector array for box
    _fill_bisector(box_array, open_k, open_j, open_i)

    # set up the full bisectors array and assigning the bounding box values
    array = np.zeros(grid_extent_kji, dtype = np.bool_)
    array[box[0, 0]:box[1, 0], box[0, 1]:box[1, 1], box[0, 2]:box[1, 2]] = box_array

    # set bisector values outside of the bounding box
    _set_bisector_outside_box(array, box, box_array)

    # check all array elements are not the same
    true_count = np.count_nonzero(array)
    cell_count = array.size
    if 0 < true_count < cell_count:
        # negate the array if it minimises the mean k and determine if the surface is a curtain
        is_curtain = _shallow_or_curtain(array, true_count, raw_bisector)
    else:
        assert raw_bisector, 'face set for surface is leaky or empty (surface does not intersect grid)'
        log.error('face set for surface is leaky or empty (surface does not intersect grid)')
        is_curtain = False

    return array, is_curtain


# yapf: disable
def bisector_from_face_indices(  # type: ignore
        grid_extent_kji: Tuple[int, int, int],
        k_faces_kji0: Union[np.ndarray, None],
        j_faces_kji0: Union[np.ndarray, None],
        i_faces_kji0: Union[np.ndarray, None],
        raw_bisector: bool,
        p_box: Union[np.ndarray, None]) -> Tuple[np.ndarray, bool]:
    # yapf: enable
    """Creates a boolean array denoting the bisection of the grid by the face sets.

    arguments:
        - grid_extent_kji (Tuple[int, int, int]): the shape of the grid
        - k_faces_kji0 (np.ndarray): an int array of indices of which faces represent the surface in the k dimension
        - j_faces_kji0 (np.ndarray): an int array of indices of which faces represent the surface in the j dimension
        - i_faces_kji0 (np.ndarray): an int array of indices of which faces represent the surface in the i dimension
        - raw_bisector (bool): if True, the bisector is returned without determining which side is shallower
        - p_box (np.ndarray): a python protocol box to limit the bisector evaluation over

    returns:
        Tuple containing:
        - array (np.ndarray): boolean bisectors array where values are True for cells on the side
          of the surface that has a lower mean k index on average and False for cells on the other side.
        - is_curtain (bool): True if the surface is a curtain (vertical), otherwise False.

    notes:
        - the face sets must form a single 'sealed' cut of the grid (eg. not waving in and out of the grid)
        - any 'boxed in' parts of the grid (completely enclosed by bisecting faces) will be consistently
          assigned to either the True or False part
    """
    assert len(grid_extent_kji) == 3

    # find the surface boundary (includes a buffer slice where surface does not reach edge of grid)
    face_box = get_boundary_from_indices(k_faces_kji0, j_faces_kji0, i_faces_kji0, grid_extent_kji)
    box = np.empty((2, 3), dtype = np.int32)
    if p_box is None:
        box[:] = face_box
    else:
        box[:] = box_intersection(p_box, face_box)
        if np.all(box == 0):
            box[:] = face_box
    # set k_faces as bool arrays covering box
    k_faces, j_faces, i_faces = _box_face_arrays_from_indices(k_faces_kji0, j_faces_kji0, i_faces_kji0, box)

    box_shape = box[1, :] - box[0, :]

    # set up the bisector array for the bounding box
    box_array = np.zeros(box_shape, dtype = np.bool_)

    # seed the bisector box array at (0, 0, 0)
    box_array[0, 0, 0] = True

    # prepare to spread True values to neighbouring cells that are not the other side of a face
    if k_faces is None:
        open_k = np.ones((box_shape[0] - 1, box_shape[1], box_shape[2]), dtype = bool)
    else:
        open_k = np.logical_not(k_faces)
    if j_faces is None:
        open_j = np.ones((box_shape[0], box_shape[1] - 1, box_shape[2]), dtype = bool)
    else:
        open_j = np.logical_not(j_faces)
    if i_faces is None:
        open_i = np.ones((box_shape[0], box_shape[1], box_shape[2] - 1), dtype = bool)
    else:
        open_i = np.logical_not(i_faces)

    # populate bisector array for box
    _fill_bisector(box_array, open_k, open_j, open_i)

    # set up the full bisectors array and assigning the bounding box values
    array = np.zeros(grid_extent_kji, dtype = np.bool_)
    array[box[0, 0]:box[1, 0], box[0, 1]:box[1, 1], box[0, 2]:box[1, 2]] = box_array

    # set bisector values outside of the bounding box
    _set_bisector_outside_box(array, box, box_array)

    # check all array elements are not the same
    true_count = np.count_nonzero(array)
    cell_count = array.size
    if 0 < true_count < cell_count:
        # negate the array if it minimises the mean k and determine if the surface is a curtain
        is_curtain = _shallow_or_curtain(array, true_count, raw_bisector)
    else:
        assert raw_bisector, 'face set for surface is leaky or empty (surface does not intersect grid)'
        log.error('face set for surface is leaky or empty (surface does not intersect grid)')
        is_curtain = False

    return array, is_curtain


# yapf: disable
def packed_bisector_from_face_indices(  # type: ignore
        grid_extent_kji: Tuple[int, int, int],
        k_faces_kji0: Union[np.ndarray, None],
        j_faces_kji0: Union[np.ndarray, None],
        i_faces_kji0: Union[np.ndarray, None],
        raw_bisector: bool,
        p_box: Union[np.ndarray, None]) -> Tuple[np.ndarray, bool]:
    # yapf: enable
    """Creates a uint8 (packed bool) array denoting the bisection of the grid by the face sets.

    arguments:
        - grid_extent_kji (Tuple[int, int, int]): the shape of the grid
        - k_faces_kji0 (np.ndarray): an int array of indices of which faces represent the surface in the k dimension
        - j_faces_kji0 (np.ndarray): an int array of indices of which faces represent the surface in the j dimension
        - i_faces_kji0 (np.ndarray): an int array of indices of which faces represent the surface in the i dimension
        - raw_bisector (bool): if True, the bisector is returned without determining which side is shallower
        - p_box (np.ndarray): a python protocol unshrunken box to limit the bisector evaluation over

    returns:
        Tuple containing:
        - array (np.uint8 array): packed boolean bisector array where values are 1 for cells on the side
          of the surface that has a lower mean k index on average and 0 for cells on the other side
        - is_curtain (bool): True if the surface is a curtain (vertical), otherwise False

    notes:
        - the face sets must form a single 'sealed' cut of the grid (eg. not waving in and out of the grid)
        - any 'boxed in' parts of the grid (completely enclosed by bisecting faces) will be consistently
          assigned to either the True or False part
        - the returned array is packed in the I axis; use np.unpackbits() to unpack
    """
    assert len(grid_extent_kji) == 3

    # find the surface boundary (includes a buffer slice where surface does not reach edge of grid), and shrink the I axis
    face_box = get_packed_boundary_from_indices(k_faces_kji0, j_faces_kji0, i_faces_kji0, grid_extent_kji)
    box = np.empty((2, 3), dtype = np.int32)
    if p_box is None:
        box[:] = face_box
    else:
        box[:] = box_intersection(shrunk_box_for_packing(p_box), face_box)
        if np.all(box == 0):
            box[:] = face_box

    # set k_faces, j_faces & i_faces as uint8 packed bool arrays covering box
    k_faces, j_faces, i_faces = _packed_box_face_arrays_from_indices(k_faces_kji0, j_faces_kji0, i_faces_kji0, box)

    box_shape = box[1, :] - box[0, :]

    # set up the bisector array for the bounding box
    box_array = np.zeros(box_shape, dtype = np.uint8)

    # seed the bisector box array at (0, 0, 0)
    box_array[0, 0, 0] = 0x80  # first bit only set

    # prepare to spread True values to neighbouring cells that are not the other side of a face
    if k_faces is None:
        open_k = np.invert(np.zeros((box_shape[0] - 1, box_shape[1], box_shape[2]), dtype = np.uint8), dtype = np.uint8)
    else:
        open_k = np.invert(k_faces, dtype = np.uint8)
    if j_faces is None:
        open_j = np.invert(np.zeros((box_shape[0], box_shape[1] - 1, box_shape[2]), dtype = np.uint8), dtype = np.uint8)
    else:
        open_j = np.invert(j_faces, dtype = np.uint8)
    if i_faces is None:
        open_i = np.invert(np.zeros(tuple(box_shape), dtype = np.uint8), dtype = np.uint8)
    else:
        open_i = np.invert(i_faces, dtype = np.uint8)

    # close off faces in padding bits, if within box
    if box[1, 2] * 8 > grid_extent_kji[2]:
        tail = grid_extent_kji[2] % 8  # number of valid bits in padded byte
        assert tail
        m = np.uint8((255 << (8 - tail)) & 255)
        open_k[:, :, -1] &= m
        open_j[:, :, -1] &= m
        m = np.uint8((m << 1) & 255)
        open_i[:, :, -1] &= m

    # populate bisector array for box
    _fill_packed_bisector(box_array, open_k, open_j, open_i)

    del open_i, open_j, open_k

    # set up the full bisectors array and assigning the bounding box values
    p_array = np.zeros(_shape_packed(grid_extent_kji), dtype = np.uint8)
    p_array[box[0, 0]:box[1, 0], box[0, 1]:box[1, 1], box[0, 2]:box[1, 2]] = box_array

    # set bisector values outside of the bounding box
    _set_packed_bisector_outside_box(p_array, box, box_array, grid_extent_kji[2] % 8)

    # check all array elements are not the same
    if hasattr(np, 'bitwise_count'):
        true_count = np.sum(np.bitwise_count(p_array))
    else:
        true_count = _bitwise_count_njit(p_array)  # note: will usually include some padding bits, so not so true!
    cell_count = np.prod(grid_extent_kji)
    if 0 < true_count < cell_count:
        # negate the array if it minimises the mean k and determine if the surface is a curtain
        # TODO: switch to _packed_shallow_or_curtain() when numba supports np.bitwise_count()
        is_curtain = _packed_shallow_or_curtain_temp_bitwise_count(p_array, true_count, raw_bisector)
    else:
        assert raw_bisector, 'face set for surface is leaky or empty (surface does not intersect grid)'
        log.error('face set for surface is leaky or empty (surface does not intersect grid)')
        is_curtain = False

    return p_array, is_curtain


def column_bisector_from_face_indices(grid_extent_ji: Tuple[int, int], j_faces_ji0: np.ndarray,
                                      i_faces_ji0: np.ndarray) -> np.ndarray:
    """Returns a numpy bool array denoting the bisection of the top layer of the grid by the curtain face sets.

    arguments:
        - grid_extent_ji (pair of int): the shape of a layer of the grid
        - j_faces_ji0, i_faces_ji0 (2D numpy int arrays of shape (N, 2)): indices of faces within a layer

    returns:
        numpy bool array of shape grid_extent_ji, set True for cells on one side of the face sets;
        set False for cells on othe side

    notes:
        - the face sets must form a single 'sealed' cut of the grid (eg. not waving in and out of the grid)
        - any 'boxed in' parts of the grid (completely enclosed by bisecting faces) will be consistently
          assigned to the False part
        - the resulting array is suitable for use as a grid property with indexable element of columns
        - the array is set True for the side of the curtain that contains cell [0, 0]
    """
    assert len(grid_extent_ji) == 2
    j_faces = np.zeros((grid_extent_ji[0] - 1, grid_extent_ji[1]), dtype = np.bool_)
    i_faces = np.zeros((grid_extent_ji[0], grid_extent_ji[1] - 1), dtype = np.bool_)
    j_faces[j_faces_ji0[:, 0], j_faces_ji0[:, 1]] = True
    i_faces[i_faces_ji0[:, 0], i_faces_ji0[:, 1]] = True
    return column_bisector_from_faces(grid_extent_ji, j_faces, i_faces)


def column_bisector_from_faces(grid_extent_ji: Tuple[int, int], j_faces: np.ndarray, i_faces: np.ndarray) -> np.ndarray:
    """Returns a numpy bool array denoting the bisection of the top layer of the grid by the curtain face sets.

    arguments:
        grid_extent_ji (pair of int): the shape of a layer of the grid
        j_faces, i_faces (numpy bool arrays): True where an internal grid face forms part of the
            bisecting surface, shaped for a single layer

    returns:
        numpy bool array of shape grid_extent_ji, set True for cells on one side of the face sets;
        set False for cells on othe side

    notes:
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
        # i faces
        c[:, 1:] = np.logical_or(c[:, 1:], np.logical_and(a[:, :-1], open_i))
        c[:, :-1] = np.logical_or(c[:, :-1], np.logical_and(a[:, 1:], open_i))
        c[:] = np.logical_and(c, np.logical_not(a))
        if np.count_nonzero(c) == 0:  # no more changes
            break
        a[:] = np.logical_or(a, c)
    if np.all(a):
        log.warning("curtain is leaky or misses grid when setting column bisector")
    # log.debug(f'returning bisector with count: {np.count_nonzero(a)} of {a.size}; shape: {a.shape}')
    return a


def shadow_from_face_indices(extent_kji, kji0):
    """Returns a numpy int8 array indicating whether cells are above, below or between K faces.

    arguments:
        extent_kji (triple int): the shape of the grid
        kji0 (numpy int array of shape (N, 3)): indices where a K face is present

    returns:
        numpy int8 array of shape extent_kji; values are: 0 neither above nor below a K face;
            1: above any K faces in the column; 2 below any K faces in the column;
            3: between K faces (one or more above and one or more below)
    """
    assert len(extent_kji) == 3
    limit = extent_kji[0] - 1  # maximum number of iterations needed to spead shadow
    shadow = np.zeros(extent_kji, dtype = np.int8)
    shadow[kji0[:, 0], kji0[:, 1], kji0[:, 2]] = 1
    shadow[kji0[:, 0] + 1, kji0[:, 1], kji0[:, 2]] += 2
    for _ in range(limit):
        c = np.logical_and(shadow[:-1] == 0, shadow[1:] == 1)
        if np.count_nonzero(c) == 0:
            break
        shadow[:-1][c] = 1
    for _ in range(limit):
        c = np.logical_and(shadow[1:] == 0, shadow[:-1] == 2)
        if np.count_nonzero(c) == 0:
            break
        shadow[1:][c] = 2
    for _ in range(limit):
        c = np.logical_and(shadow[:-1] >= 2, shadow[1:] == 1)
        if np.count_nonzero(c) == 0:
            break
        shadow[:-1][c] = 3
        shadow[1:][c] = 3
    return shadow


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
        shadow[:-1][c] = 1
    for _ in range(limit):
        c = np.logical_and(shadow[1:] == 0, shadow[:-1] == 2)
        if np.count_nonzero(c) == 0:
            break
        shadow[1:][c] = 2
    for _ in range(limit):
        c = np.logical_and(shadow[:-1] >= 2, shadow[1:] == 1)
        if np.count_nonzero(c) == 0:
            break
        shadow[:-1][c] = 3
        shadow[1:][c] = 3
    return shadow


def get_boundary(  # type: ignore
    k_faces: Union[np.ndarray, None],
    j_faces: Union[np.ndarray, None],
    i_faces: Union[np.ndarray, None],
    grid_extent_kji: Tuple[int, int, int],
) -> np.ndarray:
    """Cretaes a box of the indices that bound the surface (where the faces are True).

    arguments:
        k_faces (np.ndarray): a boolean array of which faces represent the surface in the k dimension
        j_faces (np.ndarray): a boolean array of which faces represent the surface in the j dimension
        i_faces (np.ndarray): a boolean array of which faces represent the surface in the i dimension
        grid_extent_kji (Tuple[int, int, int]): the shape of the grid

    returns:
        int array of shape (2, 3): bounding box in python protocol (max values have been incremented)

    note:
        input faces arrays are for internal grid faces (ie. extent reduced by 1 in axis of faces);
        a buffer slice is included where the surface does not reach the edge of the grid
    """

    boundary = np.zeros((2, 3), dtype = np.int32)

    starting = True

    for f_i, faces in enumerate([k_faces, j_faces, i_faces]):

        if faces is None:
            continue

        # NB. k, j & i for rest of loop refer to indices of faces, regardless of which face set is being processed

        where_k, where_j, where_i = _where_true(faces)

        if len(where_k) == 0:
            continue

        min_k = where_k[0]  # optimisation if np.where() guaranteed to return results in natural order
        max_k = where_k[-1]
        # min_k = np.amin(where_k)
        # max_k = np.amax(where_k)
        min_j = np.amin(where_j)
        max_j = np.amax(where_j)
        min_i = np.amin(where_i)
        max_i = np.amax(where_i)

        # include cells on both sides of internal faces
        # and add buffer slice where edge of grid not reached by surface faces
        if f_i == 0:
            max_k += 1
        else:
            if min_k > 0:
                min_k -= 1
            if max_k < grid_extent_kji[0] - 1:
                max_k += 1
        if f_i == 1:
            max_j += 1
        else:
            if min_j > 0:
                min_j -= 1
            if max_j < grid_extent_kji[1] - 1:
                max_j += 1
        if f_i == 2:
            max_i += 1
        else:
            if min_i > 0:
                min_i -= 1
            if max_i < grid_extent_kji[2] - 1:
                max_i += 1

        if starting:
            boundary[0, 0] = min_k
            boundary[1, 0] = max_k
            boundary[0, 1] = min_j
            boundary[1, 1] = max_j
            boundary[0, 2] = min_i
            boundary[1, 2] = max_i
            starting = False
        else:
            if min_k < boundary[0, 0]:
                boundary[0, 0] = min_k
            if max_k > boundary[1, 0]:
                boundary[1, 0] = max_k
            if min_j < boundary[0, 1]:
                boundary[0, 1] = min_j
            if max_j > boundary[1, 1]:
                boundary[1, 1] = max_j
            if min_i < boundary[0, 2]:
                boundary[0, 2] = min_i
            if max_i > boundary[1, 2]:
                boundary[1, 2] = max_i

    boundary[1, :] += 1  # increment max values to give python protocol box

    return boundary  # type: ignore


def get_boundary_dict(  # type: ignore
    k_faces: Union[np.ndarray, None],
    j_faces: Union[np.ndarray, None],
    i_faces: Union[np.ndarray, None],
    grid_extent_kji: Tuple[int, int, int],
) -> Dict[str, int]:
    """Cretaes a dictionary of the indices that bound the surface (where the faces are True).

    arguments:
        k_faces (np.ndarray): a boolean array of which faces represent the surface in the k dimension
        j_faces (np.ndarray): a boolean array of which faces represent the surface in the j dimension
        i_faces (np.ndarray): a boolean array of which faces represent the surface in the i dimension
        grid_extent_kji (Tuple[int, int, int]): the shape of the grid

    returns:
        boundary (Dict[str, int]): a dictionary of the indices that bound the surface

    note:
        input faces arrays are for internal grid faces (ie. extent reduced by 1 in axis of faces);
        a buffer slice is included where the surface does not reach the edge of the grid;
        max values are not increment, ie. need to be incremented to be used as an upper end of a python range
    """

    boundary = {
        "k_min": None,
        "k_max": None,
        "j_min": None,
        "j_max": None,
        "i_min": None,
        "i_max": None,
    }

    box = get_boundary(k_faces, j_faces, i_faces, grid_extent_kji)

    boundary["k_min"] = box[0, 0]
    boundary["k_max"] = box[1, 0] - 1
    boundary["j_min"] = box[0, 1]
    boundary["j_max"] = box[1, 1] - 1
    boundary["i_min"] = box[0, 2]
    boundary["i_max"] = box[1, 2] - 1

    return boundary  # type: ignore


@njit  # pragma: no cover
def _where_true(data: np.ndarray):
    """Jitted NumPy 'where' function to improve performance on subsequent calls."""
    return np.where(data)


@njit  # pragma: no cover
def _first_true(array: np.ndarray) -> int:  # type: ignore
    """Returns the index + 1 of the first True value in the array."""
    for idx, val in np.ndenumerate(array):
        if val:
            return idx[0] + 1
    return array.size


@njit  # pragma: no cover
def intersect_numba(
    axis: int,
    index1: int,
    index2: int,
    hits: np.ndarray,
    n_axis: int,
    points: np.ndarray,
    triangles: np.ndarray,
    grid_dxyz: Tuple[float],
    faces: np.ndarray,
    return_depths: bool,
    depths: np.ndarray,
    return_offsets: bool,
    offsets: np.ndarray,
    return_triangles: bool,
    triangle_per_face: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds the faces that intersect the surface in 3D.

    arguments:
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

    returns:
        Tuple containing:

        - faces (np.ndarray): boolean array of each cell face that can represent the surface.
        - offsets (np.ndarray): array of the distance between the centre of the cell face and the
            intersection point of the inter-cell centre vector with a triangle in the surface.
        - triangle_per_face (np.ndarray): array of triangle numbers
    """
    n_faces = faces.shape[2 - axis]
    for i in prange(len(hits)):
        tri, d1, d2 = hits[i]

        # Line start point in 3D which had a projection hit.
        centre_point_start = np.zeros(3, dtype = np.float64) + grid_dxyz[axis] / 2
        centre_point_start[2 - index1] = (d1 + 0.5) * grid_dxyz[2 - index1]
        centre_point_start[2 - index2] = (d2 + 0.5) * grid_dxyz[2 - index2]

        # Line end point in 3D.
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

        # dangerous: relies on indivisible read-modify-write of memory word containing multiple faces elements
        faces[face_idx[0], face_idx[1], face_idx[2]] = True

        if return_depths:
            depths[face_idx[0], face_idx[1], face_idx[2]] = xyz[2]
        if return_offsets:
            offsets[face_idx[0], face_idx[1], face_idx[2]] = xyz[axis] - ((face + 1) * grid_dxyz[axis])
        if return_triangles:
            triangle_per_face[face_idx[0], face_idx[1], face_idx[2]] = tri

    return faces, offsets, triangle_per_face


def _all_offsets(crs, k_offsets_list, j_offsets_list, i_offsets_list):
    if crs.xy_units == crs.z_units:
        return np.concatenate((k_offsets_list, j_offsets_list, i_offsets_list), axis = 0)
    ji_offsets = np.concatenate((j_offsets_list, i_offsets_list), axis = 0)
    wam.convert_lengths(ji_offsets, crs.xy_units, crs.z_units)
    return np.concatenate((k_offsets_list, ji_offsets), axis = 0)


@njit  # pragma: no cover
def _fill_bisector(bisect: np.ndarray, open_k: np.ndarray, open_j: np.ndarray, open_i: np.ndarray):
    nk: int = bisect.shape[0]
    nj: int = bisect.shape[1]
    ni: int = bisect.shape[2]
    going: bool = True
    while going:
        going = False
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    if bisect[k, j, i]:
                        continue
                    if ((k and bisect[k - 1, j, i] and open_k[k - 1, j, i]) or
                        (j and bisect[k, j - 1, i] and open_j[k, j - 1, i]) or
                        (i and bisect[k, j, i - 1] and open_i[k, j, i - 1]) or
                        (k < nk - 1 and bisect[k + 1, j, i] and open_k[k, j, i]) or
                        (j < nj - 1 and bisect[k, j + 1, i] and open_j[k, j, i]) or
                        (i < ni - 1 and bisect[k, j, i + 1] and open_i[k, j, i])):
                        bisect[k, j, i] = True
                        going = True


@njit  # pragma: no cover
def _fill_packed_bisector(bisect: np.ndarray, open_k: np.ndarray, open_j: np.ndarray, open_i: np.ndarray):
    nk: int = bisect.shape[0]
    nj: int = bisect.shape[1]
    ni: int = bisect.shape[2]
    going: bool = True
    m: np.uint8 = np.uint8(0)
    om: np.uint8 = np.uint8(0)
    oi: np.uint8 = np.uint8(0)
    while going:
        going = False
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    m = np.uint8(bisect[k, j, i])  # 8 bools packed into a uint8
                    if bisect[k, j, i] == np.uint8(0xFF):  # all 8 values already set
                        continue
                    om = m  # copy to check for changes later
                    if k:
                        m |= (bisect[k - 1, j, i] & open_k[k - 1, j, i])
                    if k < nk - 1:
                        m |= (bisect[k + 1, j, i] & open_k[k, j, i])
                    if j:
                        m |= (bisect[k, j - 1, i] & open_j[k, j - 1, i])
                    if j < nj - 1:
                        m |= (bisect[k, j + 1, i] & open_j[k, j, i])
                    oi = np.uint8(open_i[k, j, i])  # type: ignore
                    m |= (m >> 1) & (oi >> 1)  # type: ignore
                    m |= (m << 1) & oi  # type: ignore
                    # handle rollover bits for I
                    if i and (bisect[k, j, i - 1] & open_i[k, j, i - 1] & np.uint8(0x01)):
                        m |= np.uint8(0x80)
                    if (i < ni - 1) and (oi & 1) and (bisect[k, j, i + 1] & 0x80):
                        m |= np.uint8(0x01)
                    if m != om:
                        bisect[k, j, i] = m
                        going = True


@njit  # pragma: no cover
def _shallow_or_curtain(a: np.ndarray, true_count: int, raw: bool) -> bool:
    # negate the bool array if it minimises the mean k and determine if the bisector indicates a curtain
    assert a.ndim == 3
    layer_cell_count: int = a.shape[1] * a.shape[2]
    k_sum: int = 0
    opposite_k_sum: int = 0
    is_curtain: bool = False
    layer_count: int = 0
    for k in range(a.shape[0]):
        layer_count = np.count_nonzero(a[k])
        k_sum += (k + 1) * layer_count
        opposite_k_sum += (k + 1) * (layer_cell_count - layer_count)
    mean_k: float = float(k_sum) / float(true_count)
    opposite_mean_k: float = float(opposite_k_sum) / float(a.size - true_count)
    if mean_k > opposite_mean_k and not raw:
        a[:] = np.logical_not(a)
    if abs(mean_k - opposite_mean_k) <= 0.001:
        # log.warning('unable to determine which side of surface is shallower')
        is_curtain = True
    return is_curtain


@njit  # pragma: no cover
def _packed_shallow_or_curtain(a: np.ndarray, true_count: int, raw: bool) -> bool:
    # negate the packed bool array if it minimises the mean k and determine if the bisector indicates a curtain
    assert a.ndim == 3
    layer_cell_count: int = 8 * a.shape[1] * a.shape[2]  # note: includes padding bits
    k_sum: int = 0
    opposite_k_sum: int = 0
    is_curtain: bool = False
    layer_count: int = 0
    for k in range(a.shape[0]):
        # np.bitwise_count() not yet supported by numba
        layer_count = np.sum(np.bitwise_count(a[k]), dtype = np.int64)  # type: ignore
        k_sum += (k + 1) * layer_count
        opposite_k_sum += (k + 1) * (layer_cell_count - layer_count)
    mean_k: float = float(k_sum) / float(true_count)
    opposite_mean_k: float = float(opposite_k_sum) / float(8 * a.size - true_count)
    if mean_k > opposite_mean_k and not raw:
        a[:] = np.invert(a)
    if abs(mean_k - opposite_mean_k) <= 0.001:
        # log.warning('unable to determine which side of surface is shallower')
        is_curtain = True
    return is_curtain


@njit  # pragma: no cover
def _packed_shallow_or_curtain_temp_bitwise_count(a: np.ndarray, true_count: int, raw: bool) -> bool:
    # negate the packed bool array if it minimises the mean k and determine if the bisector indicates a curtain
    assert a.ndim == 3
    # note: following 'cell count' includes padding bits
    layer_cell_count: np.int64 = 8 * a.shape[1] * a.shape[2]  # type: ignore
    k_sum: np.int64 = 0  # type: ignore
    opposite_k_sum: np.int64 = 0  # type: ignore
    is_curtain: bool = False
    layer_count: np.int64 = 0  # type: ignore
    for k in range(a.shape[0]):
        layer_count = _bitwise_count_njit(a[k, :, :])
        k_sum += (k + 1) * layer_count
        opposite_k_sum += (k + 1) * (layer_cell_count - layer_count)
    mean_k: float = float(k_sum) / float(true_count)
    opposite_mean_k: float = float(opposite_k_sum) / float(8 * a.size - true_count)
    if mean_k > opposite_mean_k and not raw:
        a[:] = np.invert(a)
    if abs(mean_k - opposite_mean_k) <= 0.001:
        # log.warning('unable to determine which side of surface is shallower')
        is_curtain = True
    return is_curtain


def _set_bisector_outside_box(a: np.ndarray, box: np.ndarray, box_array: np.ndarray):  # type: ignore
    # set values outside of the bounding box
    if box[1, 0] < a.shape[0] and np.any(box_array[-1, :, :]):
        a[box[1, 0]:, :, :] = True
    if box[0, 0] != 0:
        a[:box[0, 0], :, :] = True
    if box[1, 1] < a.shape[1] and np.any(box_array[:, -1, :]):
        a[:, box[1, 1]:, :] = True
    if box[0, 1] != 0:
        a[:, :box[0, 1], :] = True
    if box[1, 2] < a.shape[2] and np.any(box_array[:, :, -1]):
        a[:, :, box[1, 2]:] = True
    if box[0, 2] != 0:
        a[:, :, :box[0, 2]] = True


def _set_packed_bisector_outside_box(a: np.ndarray, box: np.ndarray, box_array: np.ndarray, tail: int):
    # set values outside of the bounding box, working with packed arrays
    if box[1, 0] < a.shape[0] and np.any(box_array[-1, :, :]):
        a[box[1, 0]:, :, :] = 255
    if box[0, 0] != 0:
        a[:box[0, 0], :, :] = 255
    if box[1, 1] < a.shape[1] and np.any(box_array[:, -1, :]):
        a[:, box[1, 1]:, :] = 255
    if box[0, 1] != 0:
        a[:, :box[0, 1], :] = 255
    if box[1, 2] < a.shape[2] and np.any(np.bitwise_and(box_array[:, :, -1], 1)):
        a[:, :, box[1, 2]:] = 255
    if box[0, 2] != 0:
        a[:, :, :box[0, 2]] = 255
    if tail:
        m = np.uint8((255 << (8 - tail)) & 255)
        a[:, :, -1] &= m


def _box_face_arrays_from_indices(  # type: ignore
        k_faces_kji0: Union[np.ndarray, None], j_faces_kji0: Union[np.ndarray, None],
        i_faces_kji0: Union[np.ndarray, None], box: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    box_shape = box[1, :] - box[0, :]
    k_a = np.zeros((box_shape[0] - 1, box_shape[1], box_shape[2]), dtype = np.bool_)
    j_a = np.zeros((box_shape[0], box_shape[1] - 1, box_shape[2]), dtype = np.bool_)
    i_a = np.zeros((box_shape[0], box_shape[1], box_shape[2] - 1), dtype = np.bool_)
    ko = box[0, 0]
    jo = box[0, 1]
    io = box[0, 2]
    kr = box[1, 0] - ko
    jr = box[1, 1] - jo
    ir = box[1, 2] - io
    if k_faces_kji0 is not None:
        _set_face_array(k_a, k_faces_kji0, ko, jo, io, kr - 1, jr, ir)
    if j_faces_kji0 is not None:
        _set_face_array(j_a, j_faces_kji0, ko, jo, io, kr, jr - 1, ir)
    if i_faces_kji0 is not None:
        _set_face_array(i_a, i_faces_kji0, ko, jo, io, kr, jr, ir - 1)
    return k_a, j_a, i_a


def _packed_box_face_arrays_from_indices(  # type: ignore
        k_faces_kji0: Union[np.ndarray, None], j_faces_kji0: Union[np.ndarray, None],
        i_faces_kji0: Union[np.ndarray, None], box: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    box_shape = box[1, :] - box[0, :]  # note: I axis already shrunken
    k_a = np.zeros((box_shape[0] - 1, box_shape[1], box_shape[2]), dtype = np.uint8)
    j_a = np.zeros((box_shape[0], box_shape[1] - 1, box_shape[2]), dtype = np.uint8)
    i_a = np.zeros(tuple(box_shape), dtype = np.uint8)
    ko = box[0, 0]
    jo = box[0, 1]
    io = box[0, 2] * 8
    kr = box[1, 0] - ko
    jr = box[1, 1] - jo
    ir = box[1, 2] * 8 - io
    if k_faces_kji0 is not None:
        _set_packed_face_array(k_a, k_faces_kji0, ko, jo, io, kr - 1, jr, ir)
    if j_faces_kji0 is not None:
        _set_packed_face_array(j_a, j_faces_kji0, ko, jo, io, kr, jr - 1, ir)
    if i_faces_kji0 is not None:
        _set_packed_face_array(i_a, i_faces_kji0, ko, jo, io, kr, jr, ir)
    return k_a, j_a, i_a


@njit  # pragma: no cover
def _set_face_array(a: np.ndarray, indices: np.ndarray, ko: int, jo: int, io: int, kr: int, jr: int, ir: int) -> None:
    k: int = 0
    j: int = 0
    i: int = 0
    for ind in range(len(indices)):
        k = indices[ind, 0] - ko
        if k < 0 or k >= kr:
            continue
        j = indices[ind, 1] - jo
        if j < 0 or j >= jr:
            continue
        i = indices[ind, 2] - io
        if i < 0 or i >= ir:
            continue
        a[k, j, i] = True


@njit  # pragma: no cover
def _set_packed_face_array(a: np.ndarray, indices: np.ndarray, ko: int, jo: int, io: int, kr: int, jr: int,
                           ir: int) -> None:
    k: int = 0
    j: int = 0
    i: int = 0
    for ind in range(len(indices)):
        k = indices[ind, 0] - ko
        if k < 0 or k >= kr:
            continue
        j = indices[ind, 1] - jo
        if j < 0 or j >= jr:
            continue
        i = indices[ind, 2] - io
        if i < 0 or i >= ir:
            continue
        ii, ib = divmod(i, 8)
        a[k, j, ii] |= (1 << (7 - ib))


# yapf: disable
def get_boundary_from_indices(  # type: ignore
        k_faces_kji0: Union[np.ndarray, None],
        j_faces_kji0: Union[np.ndarray, None],
        i_faces_kji0: Union[np.ndarray, None],
        grid_extent_kji: Tuple[int, int, int]) -> np.ndarray:
    # yapf: enable
    """Return python protocol box containing indices"""
    k_min_kji0 = None if ((k_faces_kji0 is None) or (k_faces_kji0.size == 0)) else np.min(k_faces_kji0, axis = 0)
    k_max_kji0 = None if ((k_faces_kji0 is None) or (k_faces_kji0.size == 0)) else np.max(k_faces_kji0, axis = 0)
    j_min_kji0 = None if ((j_faces_kji0 is None) or (j_faces_kji0.size == 0)) else np.min(j_faces_kji0, axis = 0)
    j_max_kji0 = None if ((j_faces_kji0 is None) or (j_faces_kji0.size == 0)) else np.max(j_faces_kji0, axis = 0)
    i_min_kji0 = None if ((i_faces_kji0 is None) or (i_faces_kji0.size == 0)) else np.min(i_faces_kji0, axis = 0)
    i_max_kji0 = None if ((i_faces_kji0 is None) or (i_faces_kji0.size == 0)) else np.max(i_faces_kji0, axis = 0)
    box = np.empty((2, 3), dtype = np.int32)
    box[0, :] = grid_extent_kji
    box[1, :] = -1
    if k_min_kji0 is not None:
        box[0, 0] = k_min_kji0[0]
        box[0, 1] = k_min_kji0[1]
        box[0, 2] = k_min_kji0[2]
        box[1, 0] = k_max_kji0[0]  # type: ignore
        box[1, 1] = k_max_kji0[1]  # type: ignore
        box[1, 2] = k_max_kji0[2]  # type: ignore
    if j_min_kji0 is not None:
        box[0, 0] = min(box[0, 0], j_min_kji0[0])
        box[0, 1] = min(box[0, 1], j_min_kji0[1])
        box[0, 2] = min(box[0, 2], j_min_kji0[2])
        box[1, 0] = max(box[1, 0], j_max_kji0[0])  # type: ignore
        box[1, 1] = max(box[1, 1], j_max_kji0[1])  # type: ignore
        box[1, 2] = max(box[1, 2], j_max_kji0[2])  # type: ignore
    if i_min_kji0 is not None:
        box[0, 0] = min(box[0, 0], i_min_kji0[0])
        box[0, 1] = min(box[0, 1], i_min_kji0[1])
        box[0, 2] = min(box[0, 2], i_min_kji0[2])
        box[1, 0] = max(box[1, 0], i_max_kji0[0])  # type: ignore
        box[1, 1] = max(box[1, 1], i_max_kji0[1])  # type: ignore
        box[1, 2] = max(box[1, 2], i_max_kji0[2])  # type: ignore
    assert np.all(box[1] >= box[0]), 'attempt to find bounding box when all faces None'
    # include buffer layer where box does not reach edge of grid
    box[1, :] += 1  # switch to python protocol
    return expanded_box(box, grid_extent_kji)


def expanded_box(box: np.ndarray, extent_kji: Tuple[int, int, int]) -> np.ndarray:
    """Return a python protocol box expanded by a single slice on all six faces, where extent alloas."""
    # include buffer layer where box does not reach edge of grid
    np_extent_kji = np.array(extent_kji, dtype = np.int32)
    e_box = np.zeros((2, 3), dtype = np.int32)
    e_box[0, :] = np.maximum(box[0, :] - 1, 0)
    e_box[1, :] = np.minimum(box[1, :] + 1, extent_kji)
    assert np.all(e_box[0] >= 0)
    assert np.all(e_box[1] > e_box[0])
    assert np.all(e_box[1] <= np_extent_kji)
    return e_box


def get_packed_boundary_from_indices(  # type: ignore
        k_faces_kji0: Union[np.ndarray, None], j_faces_kji0: Union[np.ndarray, None],
        i_faces_kji0: Union[np.ndarray, None], grid_extent_kji: Tuple[int, int, int]) -> np.ndarray:
    """Return python protocol box containing indices, with I axis packed"""
    box = get_boundary_from_indices(k_faces_kji0, j_faces_kji0, i_faces_kji0, grid_extent_kji)
    return shrunk_box_for_packing(box)


def shrunk_box_for_packing(box: np.ndarray) -> np.ndarray:
    """Return box with I dimension shrunk for bit packing equivalent."""
    shrunk_box = box.copy()
    shrunk_box[0, 2] /= 8
    shrunk_box[1, 2] = ((box[1, 2] - 1) // 8) + 1
    return shrunk_box


def _shape_packed(unpacked_shape):
    """Return the equivalent packed shape for a given unpacked shape, as a tuple."""
    shrunken = ((unpacked_shape[-1] - 1) // 8) + 1
    if len(unpacked_shape) == 1:
        return (shrunken,)
    head = list(unpacked_shape[:-1])
    head.append(shrunken)
    return tuple(head)


@njit  # pragma: no cover
def _bitwise_count_njit(a: np.ndarray) -> np.int64:
    """Deprecated: only needed till numpy versions < 2.0.0 are dropped."""
    c: np.int64 = 0  # type: ignore
    c += np.count_nonzero(np.bitwise_and(a, 0x01))
    c += np.count_nonzero(np.bitwise_and(a, 0x02))
    c += np.count_nonzero(np.bitwise_and(a, 0x04))
    c += np.count_nonzero(np.bitwise_and(a, 0x08))
    c += np.count_nonzero(np.bitwise_and(a, 0x10))
    c += np.count_nonzero(np.bitwise_and(a, 0x20))
    c += np.count_nonzero(np.bitwise_and(a, 0x40))
    c += np.count_nonzero(np.bitwise_and(a, 0x80))
    return c


@njit
def box_intersection(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """Return a box which is the intersection of two boxes, python protocol; all zeros if no intersection."""
    box = np.zeros((2, 3), dtype = np.int32)
    box[0] = np.maximum(box_a[0], box_b[0])
    box[1] = np.minimum(box_a[1], box_b[1])
    if np.any(box[1] <= box[0]):
        box[:] = 0
    return box


@njit
def get_box(mask: np.ndarray) -> Tuple[np.ndarray, int]:  # pragma: no cover
    """Returns a python protocol box enclosing True elements of 3D boolean mask, and count which is zero if all False."""
    box = np.full((2, 3), -1, dtype = np.int32)
    count = 0
    for k in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for i in range(mask.shape[2]):
                if mask[k, j, i]:
                    if count == 0:
                        box[0, 0] = k
                        box[0, 1] = j
                        box[0, 2] = i
                        box[1, 0] = k + 1
                        box[1, 1] = j + 1
                        box[1, 2] = i + 1
                    else:
                        if k < box[0, 0]:
                            box[0, 0] = k
                        elif k >= box[1, 0]:
                            box[1, 0] = k + 1
                        if j < box[0, 1]:
                            box[0, 1] = j
                        elif j >= box[1, 1]:
                            box[1, 1] = j + 1
                        if i < box[0, 2]:
                            box[0, 2] = i
                        elif i >= box[1, 2]:
                            box[1, 2] = i + 1
                    count += 1
    return box, count
