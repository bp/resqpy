"""A submodule containing grid transmissibility functions"""

# note: only IJK Grid format supported at present
# see also rq_import.py

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.olio.transmission as rqtr

always_write_pillar_geometry_is_defined_array = False
always_write_cell_geometry_is_defined_array = False


def transmissibility(grid, tolerance = 1.0e-6, use_tr_properties = True, realization = None, modifier_mode = None):
    """Returns transmissibilities for standard (IJK neighbouring) connections within this grid.

    arguments:
       tolerance (float, default 1.0e-6): the minimum half cell transmissibility below which zero inter-cell
          transmissibility will be set; units are as for returned values (see notes)
       use_tr_properties (boolean, default True): if True, the grid's property collection is inspected for
          possible transmissibility arrays and if found, they are used instead of calculation; note that
          when this argument is False, the property collection is still used for the feed arrays to the
          calculation
       realization (int, optional) if present, only properties with this realization number will be used;
          applies to pre-computed transmissibility properties or permeability and net to gross ratio
          properties when computing
       modifier_mode (string, optional): if None, no transmissibility modifiers are applied; other
          options are: 'faces multiplier', for which directional transmissibility properties with indexable
          element of 'faces' will be used; 'faces per cell multiplier', in which case a transmissibility
          property with 'faces per cell' as the indexable element will be used to modify the half cell
          transmissibilities prior to combination; or 'absolute' in which case directional properties
          of local property kind 'fault transmissibility' (or 'mat transmissibility') and indexable
          element of 'faces' will be used as a third transmissibility term along with the two half
          cell transmissibilities at each face; see also the notes below

    returns:
       3 numpy float arrays of shape (nk + 1, nj, ni), (nk, nj + 1, ni), (nk, nj, ni + 1) being the
       neighbourly transmissibilities in K, J & I axes respectively

    notes:
       the 3 permeability arrays (and net to gross ratio if in use) must be identifiable in the property
       collection as they are used for the calculation;
       implicit units of measure of returned values will be m3.cP/(kPa.d) if grid crs length units are metres,
       bbl.cP/(psi.d) if length units are feet; the computation is compatible with the Nexus NEWTRAN formulation;
       values will be zero at pinchouts, and at column edges where there is a split pillar, even if there is
       juxtapostion of faces; the same is true of K gap faces (even where the gap is zero); NaNs in any of
       the feed properties also result in transmissibility values of zero;
       outer facing values will always be zero (included to be in accordance with RESQML faces properties);
       array caching in the grid object will only be used if realization is None; if a modifier mode of
       'faces multiplier' or 'faces per cell multiplier' is specified, properties will be searched for with
       local property kind 'transmissibility multiplier' and the appropriate indexable element (and direction
       facet in the case of 'faces multiplier'); the modifier mode of 'absolute' can be used to model the
       effect of faults and thin shales, tar mats etc. in a way which is independent of cell size;
       for 'aboslute' directional properties with indexable element of 'faces' and local property kind
       'fault transmissibility' (or 'mat transmissibility') will be used; such absolute faces transmissibilities
       should have a value of np.inf or np.nan where no modification is required; note that this method is only
       dealing with logically neighbouring cells and will not compute values for faces with a split pillar,
       which should be handled elsewhere
    """

    # todo: improve handling of units: check uom for half cell transmissibility property and for absolute modifiers

    k_tr = j_tr = i_tr = None

    if realization is None:
        if hasattr(grid, 'array_k_transmissibility') and grid.array_k_transmissibility is not None:
            k_tr = grid.array_k_transmissibility
        if hasattr(grid, 'array_j_transmissibility') and grid.array_j_transmissibility is not None:
            j_tr = grid.array_j_transmissibility
        if hasattr(grid, 'array_i_transmissibility') and grid.array_i_transmissibility is not None:
            i_tr = grid.array_i_transmissibility

    if use_tr_properties and (k_tr is None or j_tr is None or i_tr is None):

        pc = grid.extract_property_collection()

        if k_tr is None:
            k_tr = pc.single_array_ref(property_kind = 'transmissibility',
                                       realization = realization,
                                       continuous = True,
                                       count = 1,
                                       indexable = 'faces',
                                       facet_type = 'direction',
                                       facet = 'K')
            if k_tr is not None:
                assert k_tr.shape == (grid.nk + 1, grid.nj, grid.ni)
                if realization is None:
                    grid.array_k_transmissibility = k_tr

        if j_tr is None:
            j_tr = pc.single_array_ref(property_kind = 'transmissibility',
                                       realization = realization,
                                       continuous = True,
                                       count = 1,
                                       indexable = 'faces',
                                       facet_type = 'direction',
                                       facet = 'J')
            if j_tr is not None:
                assert j_tr.shape == (grid.nk, grid.nj + 1, grid.ni)
                if realization is None:
                    grid.array_j_transmissibility = j_tr

        if i_tr is None:
            i_tr = pc.single_array_ref(property_kind = 'transmissibility',
                                       realization = realization,
                                       continuous = True,
                                       count = 1,
                                       indexable = 'faces',
                                       facet_type = 'direction',
                                       facet = 'I')
            if i_tr is not None:
                assert i_tr.shape == (grid.nk, grid.nj, grid.ni + 1)
                if realization is None:
                    grid.array_i_transmissibility = i_tr

    if k_tr is None or j_tr is None or i_tr is None:

        half_t = grid.half_cell_transmissibility(use_property = use_tr_properties, realization = realization)

        if modifier_mode == 'faces per cell multiplier':
            half_t, pc = __faces_per_cell_multiplier(grid, half_t, pc, realization)

        if grid.has_split_coordinate_lines and (j_tr is None or i_tr is None):
            split_column_edges_j, split_column_edges_i = grid.split_column_faces()
        else:
            split_column_edges_j, split_column_edges_i = None, None

        np.seterr(divide = 'ignore')

        if k_tr is None:
            k_tr = __set_k_transmissibility(grid, half_t, k_tr, modifier_mode, pc, realization, tolerance)

        if j_tr is None:
            j_tr = __set_j_transmissibility(grid, half_t, j_tr, modifier_mode, pc, realization, split_column_edges_j,
                                            tolerance)

        if i_tr is None:
            i_tr = __set_i_transmissibility(grid, half_t, i_tr, modifier_mode, pc, realization, split_column_edges_i,
                                            tolerance)

        np.seterr(divide = 'warn')

    return k_tr, j_tr, i_tr


def __faces_per_cell_multiplier(grid, half_t, pc, realization):
    pc = grid.extract_property_collection()
    half_t_mult = pc.single_array_ref(property_kind = 'transmissibility multiplier',
                                      realization = realization,
                                      continuous = True,
                                      count = 1,
                                      indexable = 'faces per cell')
    if half_t_mult is None:
        log.warning('no faces per cell transmissibility multiplier found when calculating transmissibilities')
    else:
        log.debug('applying faces per cell transmissibility multipliers')
        half_t = np.where(np.isnan(half_t_mult), half_t, half_t * half_t_mult)
    return half_t, pc


def __set_i_transmissibility(grid, half_t, i_tr, modifier_mode, pc, realization, split_column_edges_i, tolerance):
    i_tr = np.zeros((grid.nk, grid.nj, grid.ni + 1))
    slice_a = half_t[:, :, :-1, 2, 1]  # note: internal faces only
    slice_b = half_t[:, :, 1:, 2, 0]
    internal_zero_mask = np.logical_or(np.logical_or(np.isnan(slice_a), slice_a < tolerance),
                                       np.logical_or(np.isnan(slice_b), slice_b < tolerance))
    if split_column_edges_i is not None:
        internal_zero_mask[:, split_column_edges_i] = True
    tr_mult = None
    if modifier_mode == 'faces multiplier':
        tr_mult = pc.single_array_ref(property_kind = 'transmissibility multiplier',
                                      realization = realization,
                                      facet_type = 'direction',
                                      facet = 'I',
                                      continuous = True,
                                      count = 1,
                                      indexable = 'faces')
        if tr_mult is None:
            log.warning('no I direction faces transmissibility multiplier found when calculating transmissibilities')
        else:
            assert tr_mult.shape == (grid.nk, grid.nj, grid.ni + 1)
            internal_zero_mask = np.logical_or(internal_zero_mask, np.isnan(tr_mult[:, :, 1:-1]))
    if tr_mult is None:
        tr_mult = 1.0
    tr_abs_r = 0.0
    if modifier_mode == 'absolute':
        tr_abs = pc.single_array_ref(property_kind = 'fault transmissibility',
                                     realization = realization,
                                     facet_type = 'direction',
                                     facet = 'I',
                                     continuous = True,
                                     count = 1,
                                     indexable = 'faces')
        if tr_abs is not None:
            log.debug('applying absolute I face transmissibility modification')
            assert tr_abs.shape == (grid.nk, grid.nj, grid.ni + 1)
            internal_zero_mask = np.logical_or(internal_zero_mask, tr_abs[:, :, 1:-1] <= 0.0)
            tr_abs_r = np.where(np.logical_or(np.isinf(tr_abs), np.isnan(tr_abs)), 0.0, 1.0 / tr_abs)
    i_tr[:, :, 1:-1] = np.where(internal_zero_mask, 0.0, tr_mult / ((1.0 / slice_a) + tr_abs_r + (1.0 / slice_b)))
    return i_tr


def __set_j_transmissibility(grid, half_t, j_tr, modifier_mode, pc, realization, split_column_edges_j, tolerance):
    j_tr = np.zeros((grid.nk, grid.nj + 1, grid.ni))
    slice_a = half_t[:, :-1, :, 1, 1]  # note: internal faces only
    slice_b = half_t[:, 1:, :, 1, 0]
    internal_zero_mask = np.logical_or(np.logical_or(np.isnan(slice_a), slice_a < tolerance),
                                       np.logical_or(np.isnan(slice_b), slice_b < tolerance))
    if split_column_edges_j is not None:
        internal_zero_mask[:, split_column_edges_j] = True
    tr_mult = None
    if modifier_mode == 'faces multiplier':
        tr_mult = pc.single_array_ref(property_kind = 'transmissibility multiplier',
                                      realization = realization,
                                      facet_type = 'direction',
                                      facet = 'J',
                                      continuous = True,
                                      count = 1,
                                      indexable = 'faces')
        if tr_mult is None:
            log.warning('no J direction faces transmissibility multiplier found when calculating transmissibilities')
        else:
            assert tr_mult.shape == (grid.nk, grid.nj + 1, grid.ni)
            internal_zero_mask = np.logical_or(internal_zero_mask, np.isnan(tr_mult[:, 1:-1, :]))
    if tr_mult is None:
        tr_mult = 1.0
    tr_abs_r = 0.0
    if modifier_mode == 'absolute':
        tr_abs = pc.single_array_ref(property_kind = 'fault transmissibility',
                                     realization = realization,
                                     facet_type = 'direction',
                                     facet = 'J',
                                     continuous = True,
                                     count = 1,
                                     indexable = 'faces')
        if tr_abs is not None:
            log.debug('applying absolute J face transmissibility modification')
            assert tr_abs.shape == (grid.nk, grid.nj + 1, grid.ni)
            internal_zero_mask = np.logical_or(internal_zero_mask, tr_abs[:, 1:-1, :] <= 0.0)
            tr_abs_r = np.where(np.logical_or(np.isinf(tr_abs), np.isnan(tr_abs)), 0.0, 1.0 / tr_abs)
    j_tr[:, 1:-1, :] = np.where(internal_zero_mask, 0.0, tr_mult / ((1.0 / slice_a) + tr_abs_r + (1.0 / slice_b)))
    if realization is None:
        grid.array_j_transmissibility = j_tr
    return j_tr


def __set_k_transmissibility(grid, half_t, k_tr, modifier_mode, pc, realization, tolerance):
    k_tr = np.zeros((grid.nk + 1, grid.nj, grid.ni))
    slice_a = half_t[:-1, :, :, 0, 1]  # note: internal faces only
    slice_b = half_t[1:, :, :, 0, 0]
    internal_zero_mask = np.logical_or(np.logical_or(np.isnan(slice_a), slice_a < tolerance),
                                       np.logical_or(np.isnan(slice_b), slice_b < tolerance))
    if grid.k_gaps:  # todo: scan K gaps for zero thickness gaps and allow transmission there
        internal_zero_mask[grid.k_gap_after_array, :, :] = True
    tr_mult = None
    if modifier_mode == 'faces multiplier':
        tr_mult = pc.single_array_ref(property_kind = 'transmissibility multiplier',
                                      realization = realization,
                                      facet_type = 'direction',
                                      facet = 'K',
                                      continuous = True,
                                      count = 1,
                                      indexable = 'faces')
        if tr_mult is not None:
            assert tr_mult.shape == (grid.nk + 1, grid.nj, grid.ni)
            internal_zero_mask = np.logical_or(internal_zero_mask, np.isnan(tr_mult[1:-1, :, :]))
    if tr_mult is None:
        tr_mult = 1.0
    tr_abs_r = 0.0
    if modifier_mode == 'absolute':
        tr_abs = pc.single_array_ref(property_kind = 'mat transmissibility',
                                     realization = realization,
                                     facet_type = 'direction',
                                     facet = 'K',
                                     continuous = True,
                                     count = 1,
                                     indexable = 'faces')
        if tr_abs is None:
            tr_abs = pc.single_array_ref(property_kind = 'mat transmissibility',
                                         realization = realization,
                                         continuous = True,
                                         count = 1,
                                         indexable = 'faces')
        if tr_abs is None:
            tr_abs = pc.single_array_ref(property_kind = 'fault transmissibility',
                                         realization = realization,
                                         facet_type = 'direction',
                                         facet = 'K',
                                         continuous = True,
                                         count = 1,
                                         indexable = 'faces')
        if tr_abs is not None:
            log.debug('applying absolute K face transmissibility modification')
            assert tr_abs.shape == (grid.nk + 1, grid.nj, grid.ni)
            internal_zero_mask = np.logical_or(internal_zero_mask, tr_abs[1:-1, :, :] <= 0.0)
            tr_abs_r = np.where(np.logical_or(np.isinf(tr_abs), np.isnan(tr_abs)), 0.0, 1.0 / tr_abs)
    k_tr[1:-1, :, :] = np.where(internal_zero_mask, 0.0, tr_mult / ((1.0 / slice_a) + tr_abs_r + (1.0 / slice_b)))
    if realization is None:
        grid.array_k_transmissibility = k_tr
    return k_tr


def half_cell_transmissibility(grid, use_property = True, realization = None, tolerance = 1.0e-6):
    """Returns (and caches if realization is None) half cell transmissibilities for this grid.

    arguments:
       use_property (boolean, default True): if True, the grid's property collection is inspected for
          a possible half cell transmissibility array and if found, it is used instead of calculation
       realization (int, optional) if present, only a property with this realization number will be used
       tolerance (float, default 1.0e-6): minimum half axis length below which the transmissibility
          will be deemed uncomputable (for the axis in question); NaN values will be returned (not Inf);
          units are implicitly those of the grid's crs length units

    returns:
       numpy float array of shape (nk, nj, ni, 3, 2) where the 3 covers K,J,I and the 2 covers the
          face polarity: - (0) and + (1); units will depend on the length units of the coordinate reference
          system for the grid; the units will be m3.cP/(kPa.d) or bbl.cP/(psi.d) for grid length units of m
          and ft respectively

    notes:
       the returned array is in the logical resqpy arrangement; it must be discombobulated before being
       added as a property; this method does not write to hdf5, nor create a new property or xml;
       if realization is None, a grid attribute cached array will be used; tolerance will only be
       used if the half cell transmissibilities are actually computed
    """

    # todo: allow passing of property uuids for ntg, k_k, j, i

    if realization is None and hasattr(grid, 'array_half_cell_t'):
        return grid.array_half_cell_t

    half_t = None

    if use_property:
        pc = grid.property_collection
        half_t_resqml = pc.single_array_ref(property_kind = 'transmissibility',
                                            realization = realization,
                                            continuous = True,
                                            count = 1,
                                            indexable = 'faces per cell')
        if half_t_resqml:
            assert half_t_resqml.shape == (grid.nk, grid.nj, grid.ni, 6)
            half_t = pc.combobulate(half_t_resqml)

    if half_t is None:
        # note: properties must be identifiable in property_collection
        half_t = rqtr.half_cell_t(grid, realization = realization)

    if realization is None:
        grid.array_half_cell_t = half_t

    return half_t
