"""High level fault throw scaling functions."""

import logging

log = logging.getLogger(__name__)

import os
import math as maths
import numpy as np

import resqpy.derived_model
import resqpy.fault as rqf
import resqpy.model as rq
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet

import resqpy.derived_model._common as rqdm_c
import resqpy.derived_model._copy_grid as rqdm_cg


def fault_throw_scaling(epc_file,
                        source_grid = None,
                        scaling_factor = None,
                        connection_set = None,
                        scaling_dict = None,
                        ref_k0 = 0,
                        ref_k_faces = 'top',
                        cell_range = 0,
                        store_displacement = False,
                        inherit_properties = False,
                        inherit_realization = None,
                        inherit_all_realizations = False,
                        inherit_gcs = True,
                        new_grid_title = None,
                        new_epc_file = None):
    """Extends epc with a new grid with fault throws multiplied by scaling factors.

    arguments:
       epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
       source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
          (or one 'ROOT' grid) which is used as the source grid
       scaling_factor (float, optional): if present, the default scaling factor to apply to split pillars which do not
          appear in any of the faults in the scaling dictionary; if None, such pillars are left unchanged
       connection_set (fault.GridConnectionSet object): the connection set with associated fault feature list, used to
          identify which faces (and hence pillars) belong to which named fault
       scaling_dict (dictionary mapping string to float): the scaling factor to apply to each named fault; any faults not
          included in the dictionary will be left unadjusted (unless a default scaling factor is given as scaling_factor arg)
       ref_k0 (integer, default 0): the reference layer (zero based) to use when determining the pre-existing throws
       ref_k_faces (string, default 'top'): 'top' or 'base' identifying which bounding interface to use as the reference
       cell_range (integer, default 0): the number of cells away from faults which will have depths adjusted to spatially
          smooth the effect of the throw scaling (ie. reduce sudden changes in gradient due to the scaling)
       store_displacement (boolean, default False): if True, 3 grid property parts are created, one each for x, y, & z
          displacement of cells' centres brought about by the fault throw scaling
       inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
          with the source grid
       inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
          inherit_properties is False
       inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
          realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
          inherit_properties is False or inherit_realization is not None
       inherit_gcs (boolean, default True): if True, any grid connection set objects related to the source grid will be
          inherited by the modified grid
       new_grid_title (string): used as the citation title text for the new grid object
       new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
          a new epc file (& associated h5 file) is created to contain the derived grid (& crs)

    returns:
       new grid (grid.Grid object), with fault throws scaled according to values in the scaling dictionary

    notes:
       grid points are moved along pillar lines;
       stretch is towards or away from mid-point of throw;
       same shift is applied to all layers along pillar;
       pillar lines assumed to be straight;
       if a large fault is represented by a series of parallel minor faults 'stepping' down, each minor fault will have the
       scaling factor applied independently, leading to some unrealistic results
    """

    assert epc_file or new_epc_file, 'epc file name not specified'
    if new_epc_file and epc_file and (
        (new_epc_file == epc_file) or
        (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
        new_epc_file = None
    model, source_grid = rqdm_c._establish_model_and_source_grid(epc_file, source_grid)
    assert source_grid.grid_representation == 'IjkGrid'
    assert model is not None

    assert source_grid.has_split_coordinate_lines, 'cannot scale fault throws in unfaulted grid'
    assert scaling_factor is not None or (connection_set is not None and scaling_dict is not None)

    if ref_k_faces == 'base':
        ref_k0 += 1
    assert ref_k0 >= 0 and ref_k0 <= source_grid.nk, 'reference layer out of range'

    # take a copy of the grid
    log.debug('copying grid')
    grid = rqdm_cg.copy_grid(source_grid, model)
    grid.cache_all_geometry_arrays()  # probably already cached anyway

    # todo: handle pillars with no geometry defined, and cells without geometry defined
    assert grid.geometry_defined_for_all_pillars(), 'not all pillars have defined geometry'

    primaries = (grid.nj + 1) * (grid.ni + 1)
    offsets = np.zeros(grid.points_cached.shape[1:])

    if scaling_factor is not None:  # apply global scaling to throws
        _set_offsets_based_on_scaling_factor(grid, scaling_factor, offsets, ref_k0, primaries)

    if connection_set is not None and scaling_dict is not None:  # overwrite any global offsets with named fault throw adjustments
        _set_offsets_based_on_scaling_dict(grid, connection_set, scaling_dict, offsets, ref_k0)

    # initialise flag array for adjustments
    adjusted = np.zeros((primaries,), dtype = bool)

    # insert adjusted throws to all layers of split pillars
    grid.points_cached[:, grid.split_pillar_indices_cached, :] += offsets[grid.split_pillar_indices_cached, :].reshape(
        1, -1, 3)
    adjusted[grid.split_pillar_indices_cached] = True
    grid.points_cached[:, primaries:, :] += offsets[primaries:, :].reshape(1, -1, 3)

    # iteratively look for pillars neighbouring adjusted pillars, adjusting by a decayed amount
    adjusted = adjusted.reshape((grid.nj + 1, grid.ni + 1))
    while cell_range > 0:
        newly_adjusted = _neighbourly_adjustment(grid, offsets, adjusted, cell_range)
        adjusted = np.logical_or(adjusted, newly_adjusted)
        cell_range -= 1

    # check cell edge relative directions (in x,y) to ensure geometry is still coherent
    log.debug('checking grid geometry coherence')
    grid.check_top_and_base_cell_edge_directions()

    # build cell displacement property array(s)
    if store_displacement:
        log.debug('generating cell displacement property arrays')
        displacement_collection = rqdm_c._displacement_properties(grid, source_grid)
    else:
        displacement_collection = None

    collection = rqdm_c._prepare_simple_inheritance(grid, source_grid, inherit_properties, inherit_realization,
                                                    inherit_all_realizations)
    if collection is None:
        collection = displacement_collection
    elif displacement_collection is not None:
        collection.inherit_imported_list_from_other_collection(displacement_collection, copy_cached_arrays = False)

    if new_grid_title is None or len(new_grid_title) == 0:
        new_grid_title = 'grid with fault throws scaled by ' + str(scaling_factor) + ' from ' +  \
                         str(rqet.citation_title_for_node(source_grid.root))

    gcs_list = []
    if inherit_gcs:
        gcs_uuids = model.uuids(obj_type = 'GridConnectionSetRepresentation', related_uuid = source_grid.uuid)
        for gcs_uuid in gcs_uuids:
            gcs = rqf.GridConnectionSet(model, uuid = gcs_uuid)
            gcs.cache_arrays()
            gcs_list.append((gcs, gcs.title))
        log.debug(f'{len(gcs_list)} grid connection sets to be inherited')

    # write model
    model.h5_release()
    if new_epc_file:
        rqdm_c._write_grid(new_epc_file,
                           grid,
                           property_collection = collection,
                           grid_title = new_grid_title,
                           mode = 'w')
        epc_file = new_epc_file
    else:
        ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(source_grid.root, ['Geometry', 'Points']),
                                                      'Coordinates')
        rqdm_c._write_grid(epc_file,
                           grid,
                           ext_uuid = ext_uuid,
                           property_collection = collection,
                           grid_title = new_grid_title,
                           mode = 'a')

    if len(gcs_list):
        log.debug(f'inheriting grid connection sets related to source grid: {source_grid.uuid}')
        _inherit_gcs_list(epc_file, gcs_list, source_grid, grid)

    return grid


def global_fault_throw_scaling(epc_file,
                               source_grid = None,
                               scaling_factor = None,
                               ref_k0 = 0,
                               ref_k_faces = 'top',
                               cell_range = 0,
                               store_displacement = False,
                               inherit_properties = False,
                               inherit_realization = None,
                               inherit_all_realizations = False,
                               inherit_gcs = True,
                               new_grid_title = None,
                               new_epc_file = None):
    """Rewrites epc with a new grid with all the fault throws multiplied by the same scaling factor.

    arguments:
       epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
       source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
          (or one 'ROOT' grid) which is used as the source grid
       scaling_factor (float): the scaling factor to apply to the throw across all split pillars
       ref_k0 (integer, default 0): the reference layer (zero based) to use when determining the pre-existing throws
       ref_k_faces (string, default 'top'): 'top' or 'base' identifying which bounding interface to use as the reference
       cell_range (integer, default 0): the number of cells away from faults which will have depths adjusted to spatially
          smooth the effect of the throw scaling (ie. reduce sudden changes in gradient due to the scaling)
       store_displacement (boolean, default False): if True, 3 grid property parts are created, one each for x, y, & z
          displacement of cells' centres brought about by the fault throw scaling
       inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
          with the source grid
       inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
          inherit_properties is False
       inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
          realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
          inherit_properties is False or inherit_realization is not None
       inherit_gcs (boolean, default True): if True, any grid connection set objects related to the source grid will be
          inherited by the modified grid
       new_grid_title (string): used as the citation title text for the new grid object
       new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
          a new epc file (& associated h5 file) is created to contain the derived grid (& crs)

    returns:
       new grid (grid.Grid object), with all fault throws scaled by the scaling factor

    notes:
       a scaling factor of 1 implies no change;
       calls fault_throw_scaling(), see also documentation for that function
    """

    return fault_throw_scaling(epc_file,
                               source_grid = source_grid,
                               scaling_factor = scaling_factor,
                               connection_set = None,
                               scaling_dict = None,
                               ref_k0 = ref_k0,
                               ref_k_faces = ref_k_faces,
                               cell_range = cell_range,
                               store_displacement = store_displacement,
                               inherit_properties = inherit_properties,
                               inherit_realization = inherit_realization,
                               inherit_all_realizations = inherit_all_realizations,
                               inherit_gcs = inherit_gcs,
                               new_grid_title = new_grid_title,
                               new_epc_file = new_epc_file)


def _set_offsets_based_on_scaling_factor(grid, scaling_factor, offsets, ref_k0, primaries):
    # fetch unsplit equivalent of grid points for reference layer interface
    log.debug('fetching unsplit equivalent grid points')
    unsplit_points = grid.unsplit_points_ref().reshape(grid.nk + 1, -1, 3)
    # determine existing throws on split pillars
    semi_throws = np.zeros(grid.points_cached.shape[1:])  # same throw applied to all layers
    unique_spi = np.unique(grid.split_pillar_indices_cached)
    semi_throws[unique_spi, :] = (grid.points_cached[ref_k0, unique_spi, :] - unsplit_points[ref_k0, unique_spi, :])
    semi_throws[primaries:, :] = (grid.points_cached[ref_k0, primaries:, :] -
                                  unsplit_points[ref_k0, grid.split_pillar_indices_cached, :]
                                 )  # unsplit points are mid points
    # ensure no adjustment in pillar where geometry is not defined in reference layer
    all_good = grid.geometry_defined_for_all_cells()
    if not all_good:
        log.warning('not all cells have defined geometry')
        semi_throws[:, :] = np.where(np.isnan(semi_throws), 0.0, semi_throws)
    # apply global scaling to throws
    offsets[:] = semi_throws * (scaling_factor - 1.0)


def _set_offsets_based_on_scaling_dict(grid, connection_set, scaling_dict, offsets, ref_k0):
    connection_set.cache_arrays()
    for fault_index in range(len(connection_set.feature_list)):
        fault_name = connection_set.fault_name_for_feature_index(fault_index)
        if fault_name not in scaling_dict:
            continue  # no scaling for this fault
        fault_scaling = scaling_dict[fault_name]
        if fault_scaling == 1.0:
            continue
        log.info('scaling throw on fault ' + fault_name + ' by factor of: {0:.4f}'.format(fault_scaling))
        kelp_j, kelp_i = connection_set.simplified_sets_of_kelp_for_feature_index(fault_index)
        p_list = []  # list of adjusted pillars
        for kelp in kelp_j:
            for ip in [0, 1]:
                p_a = grid.pillars_for_column[kelp[0], kelp[1], 1, ip]
                p_b = grid.pillars_for_column[kelp[0] + 1, kelp[1], 0, ip]  # other side of fault
                mid_point = 0.5 * (grid.points_cached[ref_k0, p_a] + grid.points_cached[ref_k0, p_b])
                if np.any(np.isnan(mid_point)):
                    continue
                if p_a not in p_list:
                    offsets[p_a] = (grid.points_cached[ref_k0, p_a] - mid_point) * (fault_scaling - 1.0)
                    p_list.append(p_a)
                if p_b not in p_list:
                    offsets[p_b] = (grid.points_cached[ref_k0, p_b] - mid_point) * (fault_scaling - 1.0)
                    p_list.append(p_b)
        for kelp in kelp_i:
            for jp in [0, 1]:
                p_a = grid.pillars_for_column[kelp[0], kelp[1], jp, 1]
                p_b = grid.pillars_for_column[kelp[0], kelp[1] + 1, jp, 0]  # other side of fault
                mid_point = 0.5 * (grid.points_cached[ref_k0, p_a] + grid.points_cached[ref_k0, p_b])
                if np.any(np.isnan(mid_point)):
                    continue
                if p_a not in p_list:
                    offsets[p_a] = (grid.points_cached[ref_k0, p_a] - mid_point) * (fault_scaling - 1.0)
                    p_list.append(p_a)
                if p_b not in p_list:
                    offsets[p_b] = (grid.points_cached[ref_k0, p_b] - mid_point) * (fault_scaling - 1.0)
                    p_list.append(p_b)


def _neighbourly_adjustment(grid, offsets, adjusted, cell_range):
    offset_decay = (maths.pow(2.0, cell_range) - 1.0) / (maths.pow(2.0, cell_range + 1) - 1.0)
    newly_adjusted = np.zeros((grid.nj + 1, grid.ni + 1), dtype = bool)
    for j in range(grid.nj + 1):
        for i in range(grid.ni + 1):
            if adjusted[j, i]:
                continue
            p = j * (grid.ni + 1) + i
            if p in grid.split_pillar_indices_cached:
                continue
            contributions = 0
            accum = 0.0
            if (i > 0) and adjusted[j, i - 1]:
                if j > 0:
                    accum += offsets[grid.pillars_for_column[j - 1, i - 1, 1, 0], 2]
                    contributions += 1
                if j < grid.nj:
                    accum += offsets[grid.pillars_for_column[j, i - 1, 0, 0], 2]
                    contributions += 1
            if (j > 0) and adjusted[j - 1, i]:
                if i > 0:
                    accum += offsets[grid.pillars_for_column[j - 1, i - 1, 0, 1], 2]
                    contributions += 1
                if i < grid.ni:
                    accum += offsets[grid.pillars_for_column[j - 1, i, 0, 0], 2]
                    contributions += 1
            if (i < grid.ni) and adjusted[j, i + 1]:
                if j > 0:
                    accum += offsets[grid.pillars_for_column[j - 1, i, 1, 1], 2]
                    contributions += 1
                if j < grid.nj:
                    accum += offsets[grid.pillars_for_column[j, i, 0, 1], 2]
                    contributions += 1
            if (j < grid.nj) and adjusted[j + 1, i]:
                if i > 0:
                    accum += offsets[grid.pillars_for_column[j, i - 1, 1, 1], 2]
                    contributions += 1
                if i < grid.ni:
                    accum += offsets[grid.pillars_for_column[j, i, 1, 0], 2]
                    contributions += 1
            if contributions == 0:
                continue
            dxy_dz = ((grid.points_cached[grid.nk, p, :2] - grid.points_cached[0, p, :2]) /
                      (grid.points_cached[grid.nk, p, 2] - grid.points_cached[0, p, 2]))
            offsets[p, 2] = offset_decay * accum / float(contributions)
            offsets[p, :2] = offsets[p, 2] * dxy_dz
            grid.points_cached[:, p, :] += offsets[p, :].reshape((1, 3))
            newly_adjusted[j, i] = True
    return newly_adjusted


def _inherit_gcs_list(epc_file, gcs_list, source_grid, grid):
    gcs_inheritance_model = rq.Model(epc_file)
    for gcs, gcs_title in gcs_list:
        # log.debug(f'inheriting gcs: {gcs_title}; old gcs uuid: {gcs.uuid}')
        gcs.uuid = bu.new_uuid()
        grid_list_modifications = []
        for gi, g in enumerate(gcs.grid_list):
            # log.debug(f'gcs uses grid: {g.title}; grid uuid: {g.uuid}')
            if bu.matching_uuids(g.uuid, source_grid.uuid):
                grid_list_modifications.append(gi)
        assert len(grid_list_modifications)
        for gi in grid_list_modifications:
            gcs.grid_list[gi] = grid
        gcs.model = gcs_inheritance_model
        gcs.write_hdf5()
        gcs.create_xml(title = gcs_title)
    gcs_inheritance_model.store_epc()
    gcs_inheritance_model.h5_release()
