"""Transmissibility functions for grids."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.fault as rqf
import resqpy.olio.vector_utilities as vec
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo


def half_cell_t(grid,
                perm_k = None,
                perm_j = None,
                perm_i = None,
                ntg = None,
                realization = None,
                darcy_constant = None,
                tolerance = 1.0e-6):
    """Creates a half cell transmissibilty property array for a regular or irregular IJK grid.

    arguments:
       grid (grid.Grid or grid.RegularGrid): the grid for which half cell transmissibilities are required
       perm_k, j, i (float arrays of shape (nk, nj, ni), optional): cell permeability values
          (for each direction), in mD; if None, the permeabilities are found in the grid's property collection
       ntg (float array of shape (nk, nj, ni), or float, optional): net to gross values to apply to I & J
          calculations; if a single float, is treated as a constant; if None, net to gross ratio data in the
          property collection is used
       realization (int, optional): if present and the property collection is scanned for perm or ntg
          arrays, only those properties for this realization will be used; ignored if arrays passed in
       darcy_constant (float, optional): if present, the value to use for the Darcy constant;
          if None, the grid's length units will determine the value as expected by Nexus
       tolerance (float, default 1.0e-6): minimum half axis length below which the transmissibility
          will be deemed uncomputable (for the axis in question); NaN values will be returned (not Inf);
          units are implicitly those of the grid's crs length units; ignored if grid is a RegularGrid

    returns:
       numpy float array of shape (nk, nj, ni, 3) if grid is a RegularGrid otherwise (nk, nj, ni, 3, 2)
          where the 3 covers K,J,I and (for irregular grids) the 2 covers the face polarity: - (0) and + (1);
          units will depend on the length units of the coordinate reference system for
          the grid (and implicitly on the units of the darcy_constant); if darcy_constant is None,
          the units will be m3.cP/(kPa.d) or bbl.cP/(psi.d) for grid length units of m and ft
          respectively

    notes:
       calls either for half_cell_t_irregular() or half_cell_t_regular() depending on class of grid;
       see also notes for half_cell_t_irregular() and half_cell_t_regular()
    """

    import resqpy.grid as grr

    if darcy_constant is None:
        length_units = grid.xy_units()
        assert length_units == 'm' or length_units.startswith('ft'), "Darcy constant must be specified"
        assert grid.z_units() == length_units
        darcy_constant = 0.008527 if length_units == 'm' else 0.001127  # these values from Nexus keyword ref.

    if perm_k is None or perm_j is None or perm_i is None or ntg is None:
        pc = grid.extract_property_collection()
        basic_5_parts = pc.basic_static_property_parts(realization = realization, share_perm_parts = True)
        if ntg is None and basic_5_parts[0] is not None:
            ntg = pc.cached_part_array_ref(basic_5_parts[0])
        if perm_k is None and basic_5_parts[4] is not None:
            perm_k = pc.cached_part_array_ref(basic_5_parts[4])
        if perm_j is None and basic_5_parts[3] is not None:
            perm_j = pc.cached_part_array_ref(basic_5_parts[3])
        if perm_i is None and basic_5_parts[2] is not None:
            perm_i = pc.cached_part_array_ref(basic_5_parts[2])

    assert perm_k is not None and perm_j is not None and perm_i is not None

    if isinstance(grid, grr.RegularGrid):
        return half_cell_t_regular(grid,
                                   perm_k = perm_k,
                                   perm_j = perm_j,
                                   perm_i = perm_i,
                                   ntg = ntg,
                                   darcy_constant = darcy_constant)
    elif isinstance(grid, grr.Grid):
        return half_cell_t_irregular(grid,
                                     perm_k = perm_k,
                                     perm_j = perm_j,
                                     perm_i = perm_i,
                                     ntg = ntg,
                                     darcy_constant = darcy_constant,
                                     tolerance = tolerance)
    else:
        raise ValueError(f'grid {type(grid)} is neither RegularGrid nor Grid object in call to half_cell_t()')


def half_cell_t_regular(grid, perm_k = None, perm_j = None, perm_i = None, ntg = None, darcy_constant = None):
    """Creates a half cell transmissibilty property array for a RegularGrid.

    arguments:
       grid (grid.RegularGrid): the grid for which half cell transmissibilities are required
       perm_k, j, i (float arrays of shape (nk, nj, ni), required): cell permeability values
          (for each direction), in mD;
       ntg (float array of shape (nk, nj, ni), or float, optional): net to gross values to apply to I & J
          calculations; if a single float, is treated as a constant; if None, a value of 1.0 is used
       darcy_constant (float, required): the value to use for the Darcy constant

    returns:
       numpy float array of shape (nk, nj, ni, 3) where the 3 covers K,J,I;
          units will depend on the length units of the coordinate reference system for
          the grid (and implicitly on the units of the darcy_constant); if darcy_constant is None,
          the units will be m3.cP/(kPa.d) or bbl.cP/(psi.d) for grid length units of m and ft
          respectively

    notes:
       the same half cell transmissibility value is applicable to both - and + polarity faces;
       the axes of the regular grid are assumed to be orthogonal;
       the net to gross factor is only applied to I & J transmissibilities, not K; the results
       include the Darcy Constant factor but not any transmissibility multiplier applied at the
       face; to compute the transmissibilty between neighbouring cells, take the harmonic mean of
       the two half cell transmissibilities and multiply by any transmissibility multiplier;
       returned array will need to be re-shaped before storing as a RESQML property
       with indexable elements of 'faces';
       the coordinate referemce system for the grid must have the same length units for xy and z;
       this function is vastly more computationally efficient than the general (irregular grid)
       function
    """

    assert perm_k is not None and perm_j is not None and perm_i is not None
    if ntg is None:
        ntg = 1.0

    axis_lengths = vec.naive_lengths(grid.block_dxyz_dkji)
    face_areas = np.array(
        (axis_lengths[1] * axis_lengths[2], axis_lengths[2] * axis_lengths[0], axis_lengths[0] * axis_lengths[1]))

    half_t = np.empty((grid.nk, grid.nj, grid.ni, 3))  # 3 is K,J,I
    half_t[..., 0] = perm_k * face_areas[0] / (0.5 * axis_lengths[0])
    half_t[..., 1] = ntg * perm_j * face_areas[1] / (0.5 * axis_lengths[1])
    half_t[..., 2] = ntg * perm_i * face_areas[2] / (0.5 * axis_lengths[2])

    return darcy_constant * half_t


def half_cell_t_irregular(grid,
                          perm_k = None,
                          perm_j = None,
                          perm_i = None,
                          ntg = None,
                          darcy_constant = None,
                          tolerance = 1.0e-6):
    """Creates a half cell transmissibilty property array for an IJK grid.

    arguments:
       grid (grid.Grid): the grid for which half cell transmissibilities are required
       perm_k, j, i (float arrays of shape (nk, nj, ni), reqwuired): cell permeability values
          (for each direction), in mD;
       ntg (float array of shape (nk, nj, ni), or float, required): net to gross values to apply to I & J
          calculations; if a single float, is treated as a constant;
       darcy_constant (float, required): the value to use for the Darcy constant
       tolerance (float, default 1.0e-6): minimum half axis length below which the transmissibility
          will be deemed uncomputable (for the axis in question); NaN values will be returned (not Inf);
          units are implicitly those of the grid's crs length units

    returns:
       numpy float array of shape (nk, nj, ni, 3, 2) where the 3 covers K,J,I and the 2 covers the
          face polarity: - (0) and + (1);
          units will depend on the length units of the coordinate reference system for
          the grid (and implicitly on the units of the darcy_constant); if darcy_constant is None,
          the units will be m3.cP/(kPa.d) or bbl.cP/(psi.d) for grid length units of m and ft
          respectively

    notes:
       the algorithm is equivalent to the half cell transmissibility element of the Nexus NEWTRAN
       calculation; each resulting value is effectively for the entire face, so area proportional
       fractions will be needed at faults with throw, or at grid boundaries; the net to gross
       factor is only applied to I & J transmissibilities, not K; the results
       include the Darcy Constant factor but not any transmissibility multiplier applied at the
       face; to compute the transmissibilty between neighbouring cells, take the harmonic mean of
       the two half cell transmissibilities and multiply by any transmissibility multiplier; if
       the two cells do not have simple sharing of a common face, first reduce each half cell
       transmissibility by the proportion of the face that is shared (which may be a different
       proportion for each of the two juxtaposed cells);
       returned array will need to be re-ordered and re-shaped before storing as a RESQML property
       with indexable elements of 'faces per cell';
       the coordinate referemce system for the grid must have the same length units for xy and z
    """

    # NB: axis argument is KJI index; this function also uses axes in xyz space

    assert perm_k is not None and perm_j is not None and perm_i is not None

    tolerance_sqr = tolerance * tolerance

    p = grid.points_ref(masked = False)
    edge_vectors = []
    pfc = None  # pillars for column mapping, for use when split pillars are present
    km = kp = None  # raw points k indices for K- and K+ faces, by layer, when K gaps present
    if grid.k_gaps:
        km = grid.k_raw_index_array
        kp = km + 1
    if grid.has_split_coordinate_lines:
        pfc = grid.create_column_pillar_mapping()
        if grid.k_gaps:
            edge_vectors.append(p[kp][:, pfc] - p[km][:, pfc])  # k edge vectors, shape (nk, nj, ni, 2, 2, 3)
            edge_vectors.append(
                p[:, pfc[:, :, 1, :]] -
                p[:, pfc[:, :, 0, :]])  # j edge vectors, shape (nk + k_gaps + 1, nj, ni, 2, 3) with jp removed
            edge_vectors.append(
                p[:, pfc[:, :, :, 1]] -
                p[:, pfc[:, :, :, 0]])  # i edge vectors, shape (nk + k_gaps + 1, nj, ni, 2, 3) with ip removed
        else:
            edge_vectors.append(p[1:, pfc] - p[:-1, pfc])  # k edge vectors, shape (nk, nj, ni, 2, 2, 3)
            edge_vectors.append(p[:, pfc[:, :, 1, :]] -
                                p[:, pfc[:, :, 0, :]])  # j edge vectors, shape (nk + 1, nj, ni, 2, 3) with jp removed
            edge_vectors.append(p[:, pfc[:, :, :, 1]] -
                                p[:, pfc[:, :, :, 0]])  # i edge vectors, shape (nk + 1, nj, ni, 2, 3) with ip removed
    else:
        if grid.k_gaps:
            edge_vectors.append(p[kp] - p[km])  # k edge vectors, shape (nk, nj + 1, ni + 1, 3)
            edge_vectors.append(p[:, 1:, :] - p[:, :-1, :])  # j edge vectors, shape (nk + k_gaps + 1, nj, ni + 1, 3)
            edge_vectors.append(p[:, :, 1:] - p[:, :, :-1])  # i edge vectors, shape (nk + k_gaps + 1, nj + 1, ni, 3)
        else:
            edge_vectors.append(p[1:] - p[:-1])  # k edge vectors, shape (nk, nj + 1, ni + 1, 3)
            edge_vectors.append(p[:, 1:] - p[:, :-1])  # j edge vectors, shape (nk + 1, nj, ni + 1, 3)
            edge_vectors.append(p[:, :, 1:] - p[:, :, :-1])  # i edge vectors, shape (nk + 1, nj + 1, ni, 3)

    half_t = np.empty((grid.nk, grid.nj, grid.ni, 3, 2))  # 3 is K,J,I; 2 is -/+ polarity (face)

    # note: for some reason half_axis_length_sqr values of 0.0 yield invalid value in numpy divide, rahter than divide by zero
    np.seterr(divide = 'ignore', invalid = 'ignore')

    for axis in range(3):

        if axis == 0:  # k
            if grid.has_split_coordinate_lines:
                half_axis_vectors = 0.125 * np.abs(np.sum(edge_vectors[0], axis = (
                    3, 4)))  # shape (nk, nj, ni, 3) with 3 being xyz length components of vector
                face_areas = (projected_tri_area(p[:, pfc[:, :, 0, 0]], p[:, pfc[:, :, 0, 1]], p[:, pfc[:, :, 1, 1]]) +
                              projected_tri_area(p[:, pfc[:, :, 0, 0]], p[:, pfc[:, :, 1, 1]], p[:, pfc[:, :, 1, 0]])
                             )  # shape (nk + k_gaps + 1, nj, ni, 3)
            else:
                half_axis_vectors = 0.125 * np.abs(edge_vectors[0][:, 1:, 1:] + edge_vectors[0][:, 1:, :-1] +
                                                   edge_vectors[0][:, :-1, 1:] + edge_vectors[0][:, :-1, :-1])
                face_areas = (projected_tri_area(p[:, :-1, :-1], p[:, :-1, 1:], p[:, 1:, 1:]) +
                              projected_tri_area(p[:, :-1, :-1], p[:, 1:, 1:], p[:, 1:, :-1])
                             )  # shape (nk + k_gaps + 1, nj, ni, 3) where 3 is xyz projection axis
            if grid.k_gaps:
                minus_face_areas = face_areas[
                    km]  # shape (nk, nj, ni, 3) where 3 is xyz projection axis (ie. yz plane, xz plane, xy plane)
                plus_face_areas = face_areas[kp]
            else:
                minus_face_areas = face_areas[:-1]
                plus_face_areas = face_areas[1:]
            half_axis_length_sqr = np.sum(half_axis_vectors * half_axis_vectors, axis = -1)  # shape (nk, nj, ni)
            zero_length_mask = np.logical_or(np.any(np.isnan(half_axis_vectors), axis = -1),
                                             half_axis_length_sqr < tolerance_sqr)
            minus_face_t = np.where(
                zero_length_mask, np.NaN,
                perm_k * np.sum(half_axis_vectors * minus_face_areas, axis = -1) / half_axis_length_sqr)
            plus_face_t = np.where(
                zero_length_mask, np.NaN,
                perm_k * np.sum(half_axis_vectors * plus_face_areas, axis = -1) / half_axis_length_sqr)

        elif axis == 1:  # j
            if grid.has_split_coordinate_lines:
                if grid.k_gaps:
                    half_axis_vectors = 0.125 * np.abs(edge_vectors[1][kp][:, :, :, 1] +
                                                       edge_vectors[1][kp][:, :, :, 0] +
                                                       edge_vectors[1][km][:, :, :, 1] +
                                                       edge_vectors[1][km][:, :, :, 0])
                    face_areas = (projected_tri_area(p[km][:, pfc[:, :, :, 0]], p[km][:, pfc[:, :, :, 1]],
                                                     p[kp][:, pfc[:, :, :, 1]]) +
                                  projected_tri_area(p[km][:, pfc[:, :, :, 0]], p[kp][:, pfc[:, :, :, 1]],
                                                     p[kp][:, pfc[:, :, :, 0]]))  # shape (nk, nj, ni, 2) where 2 is jp
                else:
                    half_axis_vectors = 0.125 * np.abs(edge_vectors[1][1:, :, :, 1] + edge_vectors[1][1:, :, :, 0] +
                                                       edge_vectors[1][:-1, :, :, 1] + edge_vectors[1][:-1, :, :, 0])
                    face_areas = (
                        projected_tri_area(p[:-1, pfc[:, :, :, 0]], p[:-1, pfc[:, :, :, 1]], p[1:, pfc[:, :, :, 1]]) +
                        projected_tri_area(p[:-1, pfc[:, :, :, 0]], p[1:, pfc[:, :, :, 1]], p[1:, pfc[:, :, :, 0]])
                    )  # shape (nk, nj, ni, 2) where 2 is jp
                minus_face_areas = face_areas[:, :, :, 0]
                plus_face_areas = face_areas[:, :, :, 1]
            else:
                if grid.k_gaps:
                    half_axis_vectors = 0.125 * np.abs(edge_vectors[1][kp][:, :, 1:] + edge_vectors[1][kp][:, :, :-1] +
                                                       edge_vectors[1][km][:, :, 1:] + edge_vectors[1][km][:, :, :-1])
                    face_areas = (projected_tri_area(p[km][:, :-1], p[km][:, :, 1:], p[kp][:, :, 1:]) +
                                  projected_tri_area(p[km][:, :-1], p[kp][:, :, 1:], p[kp][:, :, :-1]))
                else:
                    half_axis_vectors = 0.125 * np.abs(edge_vectors[1][1:, :, 1:] + edge_vectors[1][1:, :, :-1] +
                                                       edge_vectors[1][:-1, :, 1:] + edge_vectors[1][:-1, :, :-1])
                    face_areas = (projected_tri_area(p[:-1, :, :-1], p[:-1, :, 1:], p[1:, :, 1:]) +
                                  projected_tri_area(p[:-1, :, :-1], p[1:, :, 1:], p[1:, :, :-1])
                                 )  # shape (nk, nj + 1, ni)
                minus_face_areas = face_areas[:, :-1]
                plus_face_areas = face_areas[:, 1:]
            half_axis_length_sqr = np.sum(half_axis_vectors * half_axis_vectors, axis = -1)
            zero_length_mask = (half_axis_length_sqr < tolerance_sqr)
            minus_face_t = np.where(
                zero_length_mask, np.NaN,
                perm_j * np.sum(half_axis_vectors * minus_face_areas, axis = -1) / half_axis_length_sqr)
            plus_face_t = np.where(
                zero_length_mask, np.NaN,
                perm_j * np.sum(half_axis_vectors * plus_face_areas, axis = -1) / half_axis_length_sqr)

        else:  # i
            if grid.has_split_coordinate_lines:
                if grid.k_gaps:
                    half_axis_vectors = 0.125 * np.abs(edge_vectors[2][kp][:, :, :, 1] +
                                                       edge_vectors[2][kp][:, :, :, 0] +
                                                       edge_vectors[2][km][:, :, :, 1] +
                                                       edge_vectors[2][km][:, :, :, 0])
                    face_areas = (projected_tri_area(p[km][:, pfc[:, :, 0, :]], p[km][:, pfc[:, :, 1, :]],
                                                     p[kp][:, pfc[:, :, 1, :]]) +
                                  projected_tri_area(p[km][:, pfc[:, :, 0, :]], p[kp][:, pfc[:, :, 1, :]],
                                                     p[kp][:, pfc[:, :, 0, :]]))  # shape (nk, nj, ni, 2) where 2 is ip
                else:
                    half_axis_vectors = 0.125 * np.abs(edge_vectors[2][1:, :, :, 1] + edge_vectors[2][1:, :, :, 0] +
                                                       edge_vectors[2][:-1, :, :, 1] + edge_vectors[2][:-1, :, :, 0])
                    face_areas = (
                        projected_tri_area(p[:-1, pfc[:, :, 0, :]], p[:-1, pfc[:, :, 1, :]], p[1:, pfc[:, :, 1, :]]) +
                        projected_tri_area(p[:-1, pfc[:, :, 0, :]], p[1:, pfc[:, :, 1, :]], p[1:, pfc[:, :, 0, :]])
                    )  # shape (nk, nj, ni, 2) where 2 is ip
                minus_face_areas = face_areas[:, :, :, 0]
                plus_face_areas = face_areas[:, :, :, 1]
            else:
                if grid.k_gaps:
                    half_axis_vectors = 0.125 * np.abs(edge_vectors[2][kp][:, 1:, :] + edge_vectors[2][kp][:, :-1, :] +
                                                       edge_vectors[2][km][:, 1:, :] + edge_vectors[2][km][:, :-1, :])
                    face_areas = (projected_tri_area(p[km][:, :-1, :], p[km][:, 1:, :], p[kp][:, 1:, :]) +
                                  projected_tri_area(p[km][:, :-1, :], p[kp][:, 1:, :], p[kp][:, :-1, :]))
                else:
                    half_axis_vectors = 0.125 * np.abs(edge_vectors[2][1:, 1:, :] + edge_vectors[2][1:, :-1, :] +
                                                       edge_vectors[2][:-1, 1:, :] + edge_vectors[2][:-1, :-1, :])
                    face_areas = (projected_tri_area(p[:-1, :-1, :], p[:-1, 1:, :], p[1:, 1:, :]) +
                                  projected_tri_area(p[:-1, :-1, :], p[1:, 1:, :], p[1:, :-1, :]))
                minus_face_areas = face_areas[:, :, :-1]
                plus_face_areas = face_areas[:, :, 1:]
            half_axis_length_sqr = np.sum(half_axis_vectors * half_axis_vectors, axis = -1)
            zero_length_mask = (half_axis_length_sqr < tolerance_sqr)
            minus_face_t = np.where(
                zero_length_mask, np.NaN,
                perm_i * np.sum(half_axis_vectors * minus_face_areas, axis = -1) / half_axis_length_sqr)
            plus_face_t = np.where(
                zero_length_mask, np.NaN,
                perm_i * np.sum(half_axis_vectors * plus_face_areas, axis = -1) / half_axis_length_sqr)

        if axis != 0 and ntg is not None:
            minus_face_t *= ntg
            plus_face_t *= ntg

        half_t[:, :, :, axis, 0] = minus_face_t
        half_t[:, :, :, axis, 1] = plus_face_t

    np.seterr(divide = 'warn', invalid = 'warn')

    return np.abs(darcy_constant * half_t)


def half_cell_t_vertical_prism(vpg,
                               triple_perm_horizontal = None,
                               perm_k = None,
                               ntg = None,
                               darcy_constant = None,
                               tolerance = 1.0e-6):
    """Creates a half cell transmissibilty property array for a vertical prism grid.

    arguments:
       vpg (VerticalPrismGrid): the grid for which the half cell transmissibilities are required
       triple_perm_horizontal (numpy float array of shape (N, 3)): the directional permeabilities to apply
          to each of the three vertical faces per cell
       perm_k (numpy float array of shape (N,)): the permeability to use for the vertical transmissibilities
       ntg (numpy float array of shape (N,), optional): if present, acts as a multiplier in the
          computation of non-vertical transmissibilities
       darcy_constant (float, optional): the value to use for the Darcy constant; if None, a suitable
          value will be used depending on the length units of the vpg grid's crs
       tolerance (float, default 1.0e-6): minimum half axis length below which the transmissibility
          will be deemed uncomputable (for the axis in question); NaN values will be returned (not Inf);
          units are implicitly those of the grid's crs length units

    returns:
       numpy float array of shape (N, 5) being the per-face half cell transmissibilities for each cell

    note:
       order of 5 faces matches those of faces per cell, ie. top, base, then the 3 vertical faces
    """

    assert triple_perm_horizontal is not None
    assert triple_perm_horizontal.shape == (vpg.cell_count, 3)
    if perm_k is None:
        perm_k = np.mean(triple_perm_horizontal, axis = 1)
    if darcy_constant is None:
        length_units = vpg.xy_units()
        assert length_units == 'm' or length_units.startswith('ft')
        assert vpg.z_units() == length_units
        darcy_constant = 0.008527 if length_units == 'm' else 0.001127  # these values from Nexus keyword ref.

    # fetch triangulation and call precursor function
    p = vpg.points_ref()
    t = vpg.triangulation()
    a_t, d_t = half_cell_t_2d_triangular_precursor(p, t)
    # find heights of cells and faces
    # find horizontal area of triangles
    triangle_areas = vec.area_of_triangles(p, t, xy_projection = True).reshape((1, -1))
    cp = vpg.corner_points()
    half_thickness = 0.5 * vpg.thickness().reshape((vpg.nk, -1))

    # compute transmissibilities
    tr = np.zeros((vpg.cell_count, 5), dtype = float)
    # vertical
    tr[:, 0] = np.where(half_thickness < tolerance, np.NaN, (perm_k.reshape(
        (vpg.nk, -1)) * triangle_areas / half_thickness)).flatten()
    tr[:, 1] = tr[:, 0]
    # horizontal
    # TODO: compute dip adjustments for non-vertical transmissibilities
    dt_full = np.empty((vpg.nk, vpg.cell_count // vpg.nk, 3), dtype = float)
    dt_full[:] = d_t
    tr[:, 2:] = np.where(dt_full < tolerance, np.NaN,
                         triple_perm_horizontal.reshape((vpg.nk, -1, 3)) * a_t.reshape((1, -1, 3)) / dt_full).reshape(
                             (-1, 3))
    if ntg is not None:
        tr[:, 2:] *= ntg.reshape((-1, 1))

    tr *= darcy_constant
    return tr


def half_cell_t_2d_triangular_precursor(p, t):
    """Creates a precursor to horizontal transmissibility for prism grids (see notes).

    arguments:
       p (numpy float array of shape (N, 2 or 3)): the xy(&z) locations of cell vertices
       t (numpy int array of shape (M, 3)): the triangulation of p for which the transmissibility
          precursor is required

    returns:
       a pair of numpy float arrays, each of shape (M, 3) being the normal length and flow length
       relevant for flow across the face opposite each vertex as defined by t

    notes:
       this function acts as a precursor to the equivalent of the half cell transmissibility
       functions but for prism grids; for a resqpy VerticalPrismGrid, the triangulation can
       be shared by many layers with this function only needing to be called once; the first
       of the returned values (normal length) is the length of the triangle edge, in xy, when
       projected onto the normal of the flow direction; multiplying the normal length by a cell
       height will yield the area needed for transmissibility calculations; the second of the
       returned values (flow length) is the distance from the trangle centre to the midpoint of
       the edge and can be used as the distance term for a half cell transmissibilty; this
       function does not account for dip, it only handles the geometric aspects of half
       cell transmissibility in the xy plane
    """

    assert p.ndim == 2 and p.shape[1] in [2, 3]
    assert t.ndim == 2 and t.shape[1] == 3

    # centre points of triangles, in xy
    centres = np.mean(p[t], axis = 1)[:, :2]
    # midpoints of edges of triangles, in xy
    edge_midpoints = np.empty(tuple(list(t.shape) + [2]), dtype = float)
    edge_midpoints[:, 0, :] = 0.5 * (p[t[:, 1]] + p[t[:, 2]])[:, :2]
    edge_midpoints[:, 1, :] = 0.5 * (p[t[:, 2]] + p[t[:, 0]])[:, :2]
    edge_midpoints[:, 2, :] = 0.5 * (p[t[:, 0]] + p[t[:, 1]])[:, :2]
    # triangle edge vectors, projected in xy
    edge_vectors = np.empty(edge_midpoints.shape, dtype = float)
    edge_vectors[:, 0] = (p[t[:, 2]] - p[t[:, 1]])[:, :2]
    edge_vectors[:, 1] = (p[t[:, 0]] - p[t[:, 2]])[:, :2]
    edge_vectors[:, 2] = (p[t[:, 1]] - p[t[:, 0]])[:, :2]
    # vectors from triangle centres to mid points of edges (3 per triangle), in xy plane
    cem_vectors = edge_midpoints - centres.reshape((-1, 1, 2))
    cem_lengths = vec.naive_lengths(cem_vectors)
    # unit length vectors normal to cem_vectors, in the xy plane
    normal_vectors = np.zeros(edge_midpoints.shape)
    normal_vectors[:, :, 0] = cem_vectors[:, :, 1]
    normal_vectors[:, :, 1] = -cem_vectors[:, :, 0]
    normal_vectors = vec.unit_vectors(normal_vectors)
    # edge lengths projected onto normal vectors (length perpendicular to nominal flow direction)
    normal_lengths = np.abs(vec.dot_products(edge_vectors, normal_vectors))
    # return normal (cross-sectional) lengths and nominal flow direction lengths
    assert normal_lengths.shape == t.shape and cem_lengths.shape == t.shape

    return normal_lengths, cem_lengths


def fault_connection_set(grid, skip_inactive = False):
    """Builds a GridConnectionSet for juxtaposed faces where there is a split pillar, with fractional area data.

    arguments:
       grid (grid.Grid object): the grid for which a fault connection set is required
       skip_inactive (boolean, default False): if True, connections where either cell is inactive will be excluded

    returns:
       (GridConnectionSet, numpy float array of shape (count, 2)) where the connection set identifies all cell face
       pairs where there is juxtaposition and the array contains the fraction of the face areas that are juxtaposed;
       count is the number of cell face pairs in the connection set

    notes:
       the current algorithm is designed for faults where slip has occurred along pillars – sideways slip (strike-slip)
       will currently cause erroneous results; inaccuracies may also arise as pillars become less straight, less
       co-planar or less parallel; the combination of non-parallel pillars and layers of non-uniform thickness can
       produce inaccuracies in some situations; if the grid does not have split pillars (ie. is unfaulted), or if
       there are no qualifying connections across faults, then (None, None) will be returned;
       as fractional areas are returned, the results are applicable whether xy & z units are the same or differ
    """

    def all_nan(pillar):
        # return True if no valid points in pillar
        return np.all(np.any(np.isnan(pillar), axis = -1))

    def juxtapose(grid, p, pv, pam, pap, pbm, pbp, tol = 0.001):
        # return list of juxtaposed layer pairs and fractional areas: list of (ka, kb, fa, fb) and error info
        # TODO: orthogonal offset tolerance on pillars

        def pillar_flavour(pml_top, pml_bot, ppl_top, ppl_bot, tol = 0.001):
            # return flavour of points pattern on one pillar:
            # 1: both p above m top
            # 2: p top above m top, p bot within m
            # 3: p top above m top, p bot below m bot
            # 4: both p within m
            # 5: p top within m, p bot below m bot
            # 6: both p below m
            if ppl_top >= pml_top - tol and ppl_bot <= pml_bot + tol:
                return 4  # tolerance bias towards simplest flavour
            if ppl_bot <= pml_top + tol:
                return 1
            if ppl_top >= pml_bot - tol:
                return 6
            if ppl_top <= pml_top + tol:
                if ppl_bot <= pml_bot + tol:
                    return 2
                return 3
            return 5

        def basic_fr(g, h, ea, eb, tol):
            # returns fractional area for simple patterns (2 sides of overlap quadrilateral lie on pillars)
            if ea + eb < tol:
                return 0.0
            return (g + h) / (ea + eb)

        def fractional_area(paml_top,
                            paml_bot,
                            papl_top,
                            papl_bot,
                            pbml_top,
                            pbml_bot,
                            pbpl_top,
                            pbpl_bot,
                            tol = 0.001):
            # calculate fractional area of overlap, from m perspective (calling code swaps m & p for other perspective)
            fla = pillar_flavour(paml_top, paml_bot, papl_top, papl_bot, tol = tol)
            flb = pillar_flavour(pbml_top, pbml_bot, pbpl_top, pbpl_bot, tol = tol)
            if (fla, flb) == (4, 4):  # diagram 1
                return basic_fr(papl_bot - papl_top, pbpl_bot - pbpl_top, paml_bot - paml_top, pbml_bot - pbml_top, tol)
            elif (fla, flb) == (3, 3):  # diagram 1 (reverse perspective); diagram 3
                return 1.0
            elif (fla, flb) == (5, 5):  # diagram 2
                return basic_fr(paml_bot - papl_top, pbml_bot - pbpl_top, paml_bot - paml_top, pbml_bot - pbml_top, tol)
            elif (fla, flb) == (2, 2):  # diagram 2 (reverse perspective)
                return basic_fr(papl_bot - paml_top, pbpl_bot - pbml_top, paml_bot - paml_top, pbml_bot - pbml_top, tol)
            elif (fla, flb) == (5, 4):  # diagram 5 (diagram 4 is a special case of 5 and 2)
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = pbml_bot - pbpl_bot
                if eb < tol or g < tol:
                    sub = 0.0
                else:
                    v = papl_bot - paml_bot
                    sub = (g / (ea + eb)) * (1.0 - v / (v + g))
                return basic_fr(paml_bot - papl_top, pbml_bot - pbpl_top, paml_bot - paml_top, pbml_bot - pbml_top,
                                tol) - sub
            elif (fla, flb) == (4, 5):  # diagram 5 mirror (diagram 4 is a special case of 5)
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = paml_bot - papl_bot
                if ea < tol or g < tol:
                    sub = 0.0
                else:
                    v = pbpl_bot - pbml_bot
                    sub = (g / (ea + eb)) * (1.0 - v / (v + g))
                return basic_fr(pbml_bot - pbpl_top, paml_bot - papl_top, pbml_bot - pbml_top, paml_bot - paml_top,
                                tol) - sub
            elif (fla, flb) == (2, 3):  # diagram 5 (reverse perspective), similar to 1.0 - diagram 10
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = paml_bot - papl_bot
                if ea < tol or g < tol:
                    return 1.0
                v = pbpl_bot - pbml_bot
                return 1.0 - (g / (ea + eb)) * (1.0 - v / (v + g))
            elif (fla, flb) == (3, 2):  # diagram 5 mirror (reverse perspective)
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = pbml_bot - pbpl_bot
                if eb < tol or g < tol:
                    return 1.0
                v = papl_bot - paml_bot
                return 1.0 - (g / (ea + eb)) * (1.0 - v / (v + g))
            elif (fla, flb) == (5, 2):  # diagram 6
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = papl_top - paml_top
                if ea < tol or g < tol:
                    suba = 0.0
                else:
                    v = pbml_top - pbpl_top
                    suba = (g / (ea + eb)) * (1.0 - v / (v + g))
                g = pbml_bot - pbpl_bot
                if eb < tol or g < tol:
                    subb = 0.0
                else:
                    v = papl_bot - paml_bot
                    subb = (g / (ea + eb)) * (1.0 - v / (v + g))
                sub = suba + subb
                return 1.0 - sub
            elif (fla, flb) == (2, 5):  # diagram 6 mirror
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = pbpl_top - pbml_top
                if eb < tol or g < tol:
                    subb = 0.0
                else:
                    v = paml_top - papl_top
                    subb = (g / (ea + eb)) * (1.0 - v / (v + g))
                g = paml_bot - papl_bot
                if ea < tol or g < tol:
                    suba = 0.0
                else:
                    v = pbpl_bot - pbml_bot
                    suba = (g / (ea + eb)) * (1.0 - v / (v + g))
                sub = suba + subb
                return 1.0 - sub
            elif (fla, flb) == (2, 1):  # diagram 10
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = papl_bot - paml_top
                if ea < tol or g < tol:
                    return 0.0
                v = pbml_top - pbpl_bot
                return (g / (ea + eb)) * (1.0 - v / (v + g))
            elif (fla, flb) == (1, 2):  # diagram 10 mirror
                eb = pbml_bot - pbml_top
                ea = paml_bot - paml_top
                g = pbpl_bot - pbml_top
                if eb < tol or g < tol:
                    return 0.0
                v = paml_top - papl_bot
                return (g / (eb + ea)) * (1.0 - v / (v + g))
            elif (fla, flb) == (5, 6):  # diagram 10 (reverse perspective)
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = paml_bot - papl_top
                if ea < tol or g < tol:
                    return 0.0
                v = pbpl_top - pbml_bot
                return (g / (ea + eb)) * (1.0 - v / (v + g))
            elif (fla, flb) == (6, 5):  # diagram 10 mirror (reverse perspective)
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = pbml_bot - pbpl_top
                if eb < tol or g < tol:
                    return 0.0
                v = papl_top - paml_bot
                return (g / (ea + eb)) * (1.0 - v / (v + g))
            elif (fla, flb) == (2, 4):  # diagram 11
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = pbpl_top - pbml_top
                if eb < tol or g < tol:
                    sub = 0.0
                else:
                    v = paml_top - papl_top
                    sub = (g / (ea + eb)) * (1.0 - v / (v + g))
                return basic_fr(papl_bot - paml_top, pbpl_bot - pbml_top, paml_bot - paml_top, pbml_bot - pbml_top,
                                tol) - sub
            elif (fla, flb) == (4, 2):  # diagram 11 mirror
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = papl_top - paml_top
                if ea < tol or g < tol:
                    sub = 0.0
                else:
                    v = pbml_top - pbpl_top
                    sub = (g / (ea + eb)) * (1.0 - v / (v + g))
                return basic_fr(pbpl_bot - pbml_top, papl_bot - paml_top, pbml_bot - pbml_top, paml_bot - paml_top,
                                tol) - sub
            elif (fla, flb) == (5, 3):  # diagram 11 (reverse perspective) similar to 1.0 - diagram 10
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = papl_top - paml_top
                if ea < tol or g < tol:
                    return 1.0
                v = pbml_top - pbpl_top
                return 1.0 - (g / (ea + eb)) * (1.0 - v / (v + g))
            elif (fla, flb) == (3, 5):  # diagram 11 mirror (reverse perspective)
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = pbpl_top - pbml_top
                if eb < tol or g < tol:
                    return 1.0
                v = paml_top - papl_top
                return 1.0 - (g / (ea + eb)) * (1.0 - v / (v + g))
            elif (fla, flb) == (3, 1):  # diagram 7 (only accurate if pillars parallel and layer constant thickness?)
                s = papl_bot - paml_bot
                ea = paml_bot - paml_top
                if s + ea <= tol:
                    return 0.0
                t = pbml_bot - pbpl_bot
                v = pbml_top - pbpl_bot
                if t + v <= tol:
                    return 1.0
                return 0.5 * (s / (s + t) + 1.0 - v / (v + ea + s))
            elif (fla, flb) == (1, 3):  # diagram 7 mirror
                # (only accurate if pillars parallel and layer constant thickness?)
                s = pbpl_bot - pbml_bot
                eb = pbml_bot - pbml_top
                if s + eb <= tol:
                    return 0.0
                t = paml_bot - papl_bot
                v = paml_top - papl_bot
                if t + v <= tol:
                    return 1.0
                return 0.5 * (s / (s + t) + 1.0 - v / (v + eb + s))
            elif (fla, flb) == (4, 6):  # diagram 7 (reverse perspective)
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = paml_bot - papl_top
                if ea < tol or g < tol:
                    return 0.0
                v = pbpl_top - pbml_bot
                gs = paml_bot - papl_bot
                if gs < tol:
                    sub = 0.0
                else:
                    vs = pbpl_bot - pbml_bot
                    sub = (gs / (ea + eb)) * (1.0 - vs / (vs + gs))
                return (g / (ea + eb)) * (1.0 - v / (v + g)) - sub
            elif (fla, flb) == (6, 4):  # diagram 7 mirror (reverse perspective)
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = pbml_bot - pbpl_top
                if eb < tol or g < tol:
                    return 0.0
                v = papl_top - paml_bot
                gs = pbml_bot - pbpl_bot
                if gs < tol:
                    sub = 0.0
                else:
                    vs = papl_bot - paml_bot
                    sub = (gs / (ea + eb)) * (1.0 - vs / (vs + gs))
                return (g / (ea + eb)) * (1.0 - v / (v + g)) - sub
            elif (fla, flb) == (4, 1):  # diagram 12
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = papl_bot - paml_top
                if ea < tol or g < tol:
                    return 0.0
                gs = papl_top - paml_top
                if gs < tol:
                    sub = 0.0
                else:
                    v = pbml_top - pbpl_top
                    sub = (gs / (ea + eb)) * (1.0 - v / (v + gs))
                v = pbml_top - pbpl_bot
                return (g / (ea + eb)) * (1.0 - v / (v + g)) - sub
            elif (fla, flb) == (1, 4):  # diagram 12 mirror
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = pbpl_bot - pbml_top
                if eb < tol or g < tol:
                    return 0.0
                gs = pbpl_top - pbml_top
                if gs < tol:
                    sub = 0.0
                else:
                    v = paml_top - papl_top
                    sub = (gs / (ea + eb)) * (1.0 - v / (v + gs))
                v = paml_top - papl_bot
                return (g / (ea + eb)) * (1.0 - v / (v + g)) - sub
            elif (fla, flb) == (3, 6):  # diagram 12 (reverse perspective)
                s = pbpl_top - pbml_bot
                eb = pbml_bot - pbml_top
                if s + eb <= tol:
                    return 1.0
                t = paml_bot - papl_top
                v = paml_top - papl_top
                if t + v <= tol:
                    return 0.0
                return 1.0 - (0.5 * (s / (s + t) + 1.0 - v / (v + eb + s)))
            elif (fla, flb) == (6, 3):  # diagram 12 mirror (reverse perspective)
                s = papl_top - paml_bot
                ea = paml_bot - paml_top
                if s + ea <= tol:
                    return 1.0
                t = pbml_bot - pbpl_top
                v = pbml_top - pbpl_top
                if t + v <= tol:
                    return 0.0
                return 1.0 - (0.5 * (s / (s + t) + 1.0 - v / (v + ea + s)))
            elif (fla, flb) == (5, 1):  # diagram 9 (only accurate if pillars parallel and layer constant thickness?)
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = papl_top - paml_top
                if ea < tol or g < tol:
                    sub = 0.0
                else:
                    v = pbml_top - pbpl_top
                    sub = (g / (ea + eb)) * (1.0 - v / (v + g))
                s = papl_bot - paml_bot
                if s + ea <= tol:
                    return 0.0
                t = pbml_bot - pbpl_bot
                v = pbml_top - pbpl_bot
                if t + v <= tol:
                    return 1.0 - sub
                return 0.5 * (s / (s + t) + 1.0 - v / (v + ea + s)) - sub
            elif (fla, flb) == (1, 5):  # diagram 9 mirror
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = pbpl_top - pbml_top
                if eb < tol or g < tol:
                    sub = 0.0
                else:
                    v = paml_top - papl_top
                    sub = (g / (ea + eb)) * (1.0 - v / (v + g))
                s = pbpl_bot - pbml_bot
                if s + eb <= tol:
                    return 0.0
                t = paml_bot - papl_bot
                v = paml_top - papl_bot
                if t + v <= tol:
                    return 1.0 - sub
                return 0.5 * (s / (s + t) + 1.0 - v / (v + eb + s)) - sub
            elif (fla, flb) == (2, 6):  # diagram 9 (reverse perspective)
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = paml_bot - papl_bot
                if ea < tol or g < tol:
                    sub = 0.0
                else:
                    v = pbpl_bot - pbml_bot
                    sub = (g / (ea + eb)) * (1.0 - v / (v + g))
                s = paml_top - papl_top
                if s + ea <= tol:
                    return 0.0
                t = pbpl_top - pbml_top
                v = pbpl_top - pbml_bot
                if t + v <= tol:
                    return 1.0 - sub
                return 0.5 * (s / (s + t) + 1.0 - v / (v + ea + s)) - sub
            elif (fla, flb) == (6, 2):  # diagram 9 mirror (reverse perspective)
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                g = pbml_bot - pbpl_bot
                if eb < tol or g < tol:
                    sub = 0.0
                else:
                    v = papl_bot - paml_bot
                    sub = (g / (ea + eb)) * (1.0 - v / (v + g))
                s = pbml_top - pbpl_top
                if s + eb <= tol:
                    return 0.0
                t = papl_top - paml_top
                v = papl_top - paml_bot
                if t + v <= tol:
                    return 1.0 - sub
                return 0.5 * (s / (s + t) + 1.0 - v / (v + eb + s)) - sub
            elif (fla, flb) == (6, 1):  # diagram 8; solution only accurate if pillars are parallel
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                s = papl_top - paml_bot
                if s + ea <= tol:
                    suba = 0.0
                else:
                    t = pbml_bot - pbpl_top
                    v = pbml_top - pbpl_top
                    suba = 0.5 * (s / (s + t) + 1.0 - v / (v + ea + s)) if t + v > tol else 1.0
                s = pbml_top - pbpl_bot
                if s + eb <= tol:
                    subb = 0.0
                else:
                    t = papl_bot - paml_top
                    v = papl_bot - paml_bot
                    subb = 0.5 * (s / (s + t) + 1.0 - v / (v + eb + s)) if t + v > tol else 1.0
                sub = suba + subb
                return max(1.0 - sub, 0.0)
            elif (fla, flb) == (1, 6):  # diagram 8 mirror or reverse perspective
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                s = pbpl_top - pbml_bot
                if s + eb <= tol:
                    subb = 0.0
                else:
                    t = paml_bot - papl_top
                    v = paml_top - papl_top
                    subb = 0.5 * (s / (s + t) + 1.0 - v / (v + eb + s)) if t + v > tol else 1.0
                s = paml_top - papl_bot
                if s + ea <= tol:
                    suba = 0.0
                else:
                    t = pbpl_bot - pbml_top
                    v = pbpl_bot - pbml_bot
                    suba = 0.5 * (s / (s + t) + 1.0 - v / (v + ea + s)) if t + v > tol else 1.0
                sub = suba + subb
                return max(1.0 - sub, 0.0)
            elif (fla, flb) == (4, 3):  # diagram 13
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                if ea < tol:
                    return 1.0
                g = papl_top - paml_top
                if g < tol:
                    sub1 = 0.0
                else:
                    v = pbml_top - pbpl_top
                    sub1 = (g / (ea + eb)) * (1.0 - v / (v + g))
                g = paml_bot - papl_bot
                if g < tol:
                    sub2 = 0.0
                else:
                    v = pbpl_bot - pbml_bot
                    sub2 = (g / (ea + eb)) * (1.0 - v / (v + g))
                return 1.0 - (sub1 + sub2)
            elif (fla, flb) == (3, 4):  # diagram 13 mirror or reverse perspective
                ea = paml_bot - paml_top
                eb = pbml_bot - pbml_top
                if eb < tol:
                    return 1.0
                g = pbpl_top - pbml_top
                if g < tol:
                    sub1 = 0.0
                else:
                    v = paml_top - papl_top
                    sub1 = (g / (ea + eb)) * (1.0 - v / (v + g))
                g = pbml_bot - pbpl_bot
                if g < tol:
                    sub2 = 0.0
                else:
                    v = papl_bot - paml_bot
                    sub2 = (g / (ea + eb)) * (1.0 - v / (v + g))
                return 1.0 - (sub1 + sub2)
            else:
                raise Exception(f'unexpected juxtaposition pattern ({fla}, {flb})')

        pav = 0.5 * (pv[pam] + pv[pap])  # mean pillar unit vector for -I side of J face
        pbv = 0.5 * (pv[pbm] + pv[pbp])  # mean pillar unit vector for +I side of J face
        if all_nan(p[:, pam, :]) or all_nan(p[:, pap, :]) or all_nan(p[:, pbm, :]) or all_nan(p[:, pbp, :]):
            return []
        o = np.nanmin(p[:, pam, :], axis = 0)  # arbitrary local origin
        oe = np.expand_dims(o, 0)
        paml = p[:, pam, :] - oe  # pillar points translated to local space
        papl = p[:, pap, :] - oe
        pbml = p[:, pbm, :] - oe
        pbpl = p[:, pbp, :] - oe
        pamln = np.empty(
            paml.shape[0])  # pillar points normalised to scalar distances on mean pillar vector, arbitrary origin
        for i in range(pamln.size):
            pamln[i] = np.dot(paml[i], pav)
        papln = np.empty(papl.shape[0])
        for i in range(papln.size):
            papln[i] = np.dot(papl[i], pav)
        pbmln = np.empty(pbml.shape[0])
        for i in range(pbmln.size):
            pbmln[i] = np.dot(pbml[i], pbv)
        pbpln = np.empty(pbpl.shape[0])
        for i in range(pbpln.size):
            pbpln[i] = np.dot(pbpl[i], pbv)

        juxta_list = []
        fa_p_totals = np.zeros(grid.nk)
        fa_m_worst_scaling = 1.0
        fa_m_downscaling_count = 0
        for km in range(grid.nk):  # for each layer on -ve side of fault
            km_top = grid.k_raw_index_array[km] if grid.k_gaps else km  # index of top points on -ve side of fault
            km_bot = km_top + 1
            paml_top = pamln[
                km_top]  # point distances in local vector space; note local vector is different for pillars a and b
            paml_bot = pamln[km_bot]
            pbml_top = pbmln[km_top]
            pbml_bot = pbmln[km_bot]
            if np.any(np.isnan((paml_top, paml_bot, pbml_top, pbml_bot))):
                continue
            # scan layers on +ve side of fault looking for juxtaposition
            kp = -1
            km_start_index = len(juxta_list)
            fa_m_total = 0.0
            while True:
                kp += 1
                if kp >= grid.nk:
                    break
                kp_top = grid.k_raw_index_array[kp] if grid.k_gaps else kp  # index of top points on +ve side of fault
                kp_bot = kp_top + 1
                papl_top = papln[
                    kp_top]  # point distances in local vector space; note local vector is different for pillars a and b
                papl_bot = papln[kp_bot]
                pbpl_top = pbpln[kp_top]
                pbpl_bot = pbpln[kp_bot]
                if np.any(np.isnan((papl_top, papl_bot, pbpl_top, pbpl_bot))):
                    continue
                # in following comments, 'shallower' means less distance in local vector, which is a K direction vector of sorts
                if (paml_top >= papl_bot - tol) and (pbml_top >= pbpl_bot - tol):
                    continue  # p fully shallower than m
                if (paml_bot <= papl_top + tol) and (pbml_bot <= pbpl_top + tol):
                    continue  # p fully deeper than m
                # juxtaposition established, now determine fractional overlap area from perspectives of both sides of fault
                fa_m = fractional_area(paml_top,
                                       paml_bot,
                                       papl_top,
                                       papl_bot,
                                       pbml_top,
                                       pbml_bot,
                                       pbpl_top,
                                       pbpl_bot,
                                       tol = tol)
                fa_p = fractional_area(papl_top,
                                       papl_bot,
                                       paml_top,
                                       paml_bot,
                                       pbpl_top,
                                       pbpl_bot,
                                       pbml_top,
                                       pbml_bot,
                                       tol = tol)
                # log.debug(f'K {km} {fa_m}  <->  {fa_p} {kp}')
                fa_m = min(max(fa_m, 0.0), 1.0)
                fa_p = min(max(fa_p, 0.0), 1.0)
                juxta_list.append((km, kp, fa_m, fa_p))
                fa_m_total += fa_m
                fa_p_totals[kp] += fa_p
            # find sum of fractional areas by layer (separately for both perspectives); normalise to max of 1.0
            if fa_m_total > 1.0:
                fa_m_downscaling_count += 1
                if fa_m_total > fa_m_worst_scaling:
                    fa_m_worst_scaling = fa_m_total
                    # log.warning(f'downscaling fractional areas on minus side of fault by factor of {fa_m_total} in layer {km}')
                for i in range(km_start_index, len(juxta_list)):
                    (km, kp, fa_m, fa_p) = juxta_list[i]
                    juxta_list[i] = (km, kp, fa_m / fa_m_total, fa_p)
        any_p_scaling = False
        fa_p_worst_scaling = np.nanmax(fa_p_totals)
        fa_p_downscaling_count = 0
        for kp in range(grid.nk):
            if fa_p_totals[kp] > 1.0:
                fa_p_downscaling_count += 1
                # log.warning(f'downscaling fractional areas on plus side of fault by factor of {fa_p_totals[kp]} in layer {kp}')
                any_p_scaling = True
        if any_p_scaling:
            for i in range(len(juxta_list)):
                (km, kp, fa_m, fa_p) = juxta_list[i]
                if fa_p_totals[kp] > 1.0:
                    juxta_list[i] = (km, kp, fa_m, fa_p / fa_p_totals[kp])
        return juxta_list, (fa_m_downscaling_count, fa_p_downscaling_count, fa_m_worst_scaling, fa_p_worst_scaling)

    if not grid.has_split_coordinate_lines:
        return None, None
    skip_inactive = skip_inactive and hasattr(grid, 'inactive') and grid.inactive is not None

    p = grid.points_ref(masked = False)  # shape (nk + k_gaps + 1, np, 3)
    pv = vec.unit_vectors(p[-1] - p[0])  # pillar vectors; shape (np, 3); could postpone to individual pillar work
    col_j_split, col_i_split = grid.split_column_faces(
    )  # internal only column faces; shapes (nj - 1, ni), (nj, ni - 1)
    pfc = grid.create_column_pillar_mapping()  # pillars for column; shape (nj, ni, 2, 2)

    juxtaposed_j_list = []
    j_ji = np.stack(np.where(col_j_split)).T  # ji0 columns where +J face is split; shape (nc, 2)
    warning_count = 0
    for (j, i) in j_ji:
        # log.debug(f'J, I  {j}, {i}')
        pam = pfc[j, i, 1, 0]  # pillar index for -I edge of J face, for col on -J side of fault
        pap = pfc[j + 1, i, 0, 0]  # pillar index for -I edge of J face, for col on +J side of fault
        pbm = pfc[j, i, 1, 1]  # pillar index for +I edge of J face, for col on -J side of fault
        pbp = pfc[j + 1, i, 0, 1]  # pillar index for +I edge of J face, for col on +J side of fault
        ji_list, scaling_info = juxtapose(grid, p, pv, pam, pap, pbm, pbp)
        if scaling_info[2] > 1.2 or scaling_info[3] > 1.2:
            if warning_count < 20:
                log.warning(
                    f'severe downscaling for I+ face in column ({j}, {i}); worst m {scaling_info[2]}; worst p {scaling_info[3]}'
                )
            elif warning_count == 20:
                log.warning('other similar I face warnings suppressed')
            warning_count += 1
        for (km, kp, fa_m, fa_p) in ji_list:
            if skip_inactive and (grid.inactive[km, j, i] or grid.inactive[kp, j + 1, i]):
                continue
            juxtaposed_j_list.append(((km, j, i), (kp, j + 1, i), fa_m, fa_p))
    juxtaposed_i_list = []
    i_ji = np.stack(np.where(col_i_split)).T  # ji0 columns where +I face is split; shape (nc, 2)
    warning_count = 0
    for (j, i) in i_ji:
        pam = pfc[j, i, 0, 1]  # pillar index for -J edge of I face, for col on -I side of fault
        pap = pfc[j, i + 1, 0, 0]  # pillar index for -J edge of I face, for col on +I side of fault
        pbm = pfc[j, i, 1, 1]  # pillar index for +J edge of I face, for col on -I side of fault
        pbp = pfc[j, i + 1, 1, 0]  # pillar index for +J edge of I face, for col on +I side of fault
        ji_list, scaling_info = juxtapose(grid, p, pv, pam, pap, pbm, pbp)
        if scaling_info[2] > 1.2 or scaling_info[3] > 1.2:
            if warning_count < 20:
                log.warning(
                    f'severe downscaling for J+ face in column ({j}, {i}); worst m {scaling_info[2]}; worst p {scaling_info[3]}'
                )
            elif warning_count == 20:
                log.warning('other similar J face warnings suppressed')
            warning_count += 1
        for (km, kp, fa_m, fa_p) in ji_list:
            if skip_inactive and (grid.inactive[km, j, i] or grid.inactive[kp, j, i + 1]):
                continue
            juxtaposed_i_list.append(((km, j, i), (kp, j, i + 1), fa_m, fa_p))

    combo_list = juxtaposed_j_list + juxtaposed_i_list
    count = len(combo_list)
    if count == 0:
        return None, None

    # build connection set
    fcs = rqf.GridConnectionSet(grid.model, grid = grid)
    fcs.grid_list = [grid]
    fcs.count = count
    fcs.grid_index_pairs = np.zeros((count, 2), dtype = int)
    fcs.cell_index_pairs = np.zeros((count, 2), dtype = int)
    fcs.cell_index_pairs[:] = np.array([
        grid.natural_cell_indices(np.array([[k, j, i] for ((k, j, i), _, _, _) in combo_list])),
        grid.natural_cell_indices(np.array([[k, j, i] for (_, (k, j, i), _, _) in combo_list]))
    ]).T
    fcs.face_index_pairs = np.zeros((count, 2), dtype = int)
    fcs.face_index_pairs[:len(juxtaposed_j_list), 0] = fcs.face_index_map[1, 1]  # J+
    fcs.face_index_pairs[:len(juxtaposed_j_list), 1] = fcs.face_index_map[1, 0]  # J-
    fcs.face_index_pairs[len(juxtaposed_j_list):, 0] = fcs.face_index_map[2, 1]  # I+
    fcs.face_index_pairs[len(juxtaposed_j_list):, 1] = fcs.face_index_map[2, 0]  # I-

    feature_name = 'all faults with throw'
    fcs.feature_indices = np.zeros(count, dtype = int)  # could create seperate features by named fracture
    tbf = rqo.TectonicBoundaryFeature(grid.model, kind = 'fault', feature_name = feature_name)
    tbf_root = tbf.create_xml()
    fi = rqo.FaultInterpretation(grid.model, tectonic_boundary_feature = tbf, is_normal = True)
    fi_root = fi.create_xml(tbf_root, title_suffix = None)
    fi_uuid = rqet.uuid_for_part_root(fi_root)

    fcs.feature_list = [('obj_FaultInterpretation', fi_uuid, str(feature_name))]

    fa = np.array([[fa_m, fa_p] for (_, _, fa_m, fa_p) in combo_list])

    return fcs, fa


def projected_tri_area(pa, pb, pc):
    """Return array holding areas of triangles projected onto each of yz, xz, xy.

    arguments:
       pa, pb, pc (numpy float array of shape (..., 3): the corner points of a set of triangles (one corner
          in each of pa, pb & pc, for every triangle); last axis in arrays covers x,y,z

    returns:
       numpy float array of same shape as pa, pb & pc, with the last axis covering yz, xz, xy projections;
       the return values are the areas of the triangles projected in the three principal x,y,z axes

    note:
       assumes that units for x, y & z are the same; returned area units are those implicit units squared
    """

    assert pa.shape == pb.shape == pc.shape and pa.shape[-1] == 3

    area = np.empty(pa.shape)
    for xyz in range(3):
        xyz_0 = (xyz + 1) % 3
        xyz_1 = (xyz + 2) % 3
        pap = np.stack((pa[..., xyz_0], pa[..., xyz_1]), axis = -1)  # points projected onto 2D
        pbp = np.stack((pb[..., xyz_0], pb[..., xyz_1]), axis = -1)
        pcp = np.stack((pc[..., xyz_0], pc[..., xyz_1]), axis = -1)
        lap = vec.naive_lengths(pbp - pap)  # 2D triangle edge vector lengths
        lbp = vec.naive_lengths(pcp - pbp)
        lcp = vec.naive_lengths(pap - pcp)
        s = 0.5 * (lap + lbp + lcp)  # 2D triangle semi perimeter lengths
        area[..., xyz] = np.sqrt(s * (s - lap) * (s - lbp) * (s - lcp))

    return area
