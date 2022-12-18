"""High level refined grid function."""

import logging

log = logging.getLogger(__name__)

import os
import numpy as np

import resqpy.crs as rqc
import resqpy.derived_model
import resqpy.grid as grr
import resqpy.model as rq
import resqpy.olio.fine_coarse as fc
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp
import resqpy.rq_import as rqi

import resqpy.derived_model._common as rqdm_c


def refined_grid(epc_file,
                 source_grid,
                 fine_coarse,
                 inherit_properties = False,
                 inherit_realization = None,
                 inherit_all_realizations = False,
                 source_grid_uuid = None,
                 set_parent_window = None,
                 infill_missing_geometry = True,
                 new_grid_title = None,
                 new_epc_file = None):
    """Generates a refined version of the source grid, optionally inheriting properties.

    arguments:
       epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
       source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
          (or one 'ROOT' grid) which is used as the source grid unless source_grid_uuid is specified to identify the grid
       fine_coarse (resqpy.olio.fine_coarse.FineCoarse object): the mapping between cells in the fine (output) and
          coarse (source) grids
       inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
          with the source grid, with values resampled in the simplest way onto the finer grid
       inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
          inherit_properties is False
       inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
          realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
          inherit_properties is False or inherit_realization is not None
       source_grid_uuid (uuid.UUID, optional): the uuid of the source grid – an alternative to the source_grid argument
          as a way of identifying the grid
       set_parent_window (boolean or str, optional): if True or 'parent', the refined grid has its parent window attribute
          set; if False, the parent window is not set; if None, the default will be True if new_epc_file is None or False
          otherwise; if 'grandparent' then an intervening parent window with no refinement or coarsening will be skipped
          and its box used in the parent window for the new grid, relating directly to the original grid
       infill_missing_geometry (boolean, default True): if True, an attempt is made to generate grid geometry in the
          source grid wherever it is undefined; if False, any undefined geometry will result in an assertion failure
       new_grid_title (string): used as the citation title text for the new grid object
       new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
          a new epc file (& associated h5 file) is created to contain the refined grid (& crs)

    returns:
       new grid object being the refined grid; the epc and hdf5 files are written to as an intentional side effect

    notes:
       this function refines an entire grid; to refine a local area of a grid, first use the extract_box function
       and then use this function on the extracted grid; in such a case, using a value of 'grandparent' for the
       set_parent_window argument will relate the refined grid back to the original;
       if geometry infilling takes place, cached geometry and mask arrays within the source grid object will be
       modified as a side-effect of the function (but not written to hdf5 or changed in xml)
    """

    epc_file, model, model_in, source_grid =  \
        _establish_models_and_source_grid(epc_file, new_epc_file, source_grid, source_grid_uuid)

    assert fine_coarse is not None and isinstance(fine_coarse, fc.FineCoarse)
    if set_parent_window is None:
        set_parent_window = (new_epc_file is None)

    if infill_missing_geometry and (not source_grid.geometry_defined_for_all_cells() or
                                    not source_grid.geometry_defined_for_all_pillars()):
        log.debug('attempting infill of geometry missing in source grid')
        source_grid.set_geometry_is_defined(treat_as_nan = None,
                                            treat_dots_as_nan = True,
                                            complete_partial_pillars = True,
                                            nullify_partial_pillars = False,
                                            complete_all = True)
    assert source_grid.geometry_defined_for_all_pillars(), 'refinement requires geometry to be defined for all pillars'
    assert source_grid.geometry_defined_for_all_cells(), 'refinement requires geometry to be defined for all cells'

    assert tuple(fine_coarse.coarse_extent_kji) == tuple(source_grid.extent_kji),  \
           'fine_coarse mapping coarse extent does not match that of source grid'
    fine_coarse.assert_valid()

    source_grid.cache_all_geometry_arrays()
    if source_grid.has_split_coordinate_lines:
        source_grid.create_column_pillar_mapping()

    if model is not model_in:
        crs_part = model_in.part_for_uuid(source_grid.crs_uuid)
        assert crs_part is not None
        model.copy_part_from_other_model(model_in, crs_part)

    # todo: set nan-abled numpy operations?

    if source_grid.has_split_coordinate_lines:

        grid = _refined_faulted_grid(model, source_grid, fine_coarse)

    else:

        grid = _refined_unfaulted_grid(model, source_grid, fine_coarse)

    # todo: option of re-draping interpolated pillars to surface

    collection = None
    if inherit_properties:
        collection = _inherit_properties(source_grid, grid, fine_coarse, inherit_realization, inherit_all_realizations)

    if set_parent_window:
        _set_parent_window(set_parent_window, source_grid, grid, fine_coarse)

    # write grid
    if new_grid_title is None or len(new_grid_title) == 0:
        new_grid_title = 'grid refined from ' + str(rqet.citation_title_for_node(source_grid.root))

    model.h5_release()
    if model is not model_in:
        model_in.h5_release()

    if new_epc_file:
        rqdm_c._write_grid(new_epc_file,
                           grid,
                           property_collection = collection,
                           grid_title = new_grid_title,
                           mode = 'w')
    else:
        ext_uuid, _ = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(source_grid.root, ['Geometry', 'Points']),
                                                      'Coordinates')
        rqdm_c._write_grid(epc_file,
                           grid,
                           ext_uuid = ext_uuid,
                           property_collection = collection,
                           grid_title = new_grid_title,
                           mode = 'a')

    return grid


def _refined_faulted_grid(model, source_grid, fine_coarse):

    source_grid.corner_points(cache_cp_array = True)
    fnk, fnj, fni = fine_coarse.fine_extent_kji
    fine_cp = np.empty((fnk, fnj, fni, 2, 2, 2, 3))
    for ck0 in range(source_grid.nk):
        fine_k_base = fine_coarse.fine_base_for_coarse_axial(0, ck0)
        k_ratio = fine_coarse.ratio(0, ck0)
        k_interp = np.ones((k_ratio + 1,))
        k_interp[:-1] = fine_coarse.interpolation(0, ck0)
        for cj0 in range(source_grid.nj):
            fine_j_base = fine_coarse.fine_base_for_coarse_axial(1, cj0)
            j_ratio = fine_coarse.ratio(1, cj0)
            j_interp = np.ones((j_ratio + 1,))
            j_interp[:-1] = fine_coarse.interpolation(1, cj0)
            for ci0 in range(source_grid.ni):
                fine_i_base = fine_coarse.fine_base_for_coarse_axial(2, ci0)
                i_ratio = fine_coarse.ratio(2, ci0)
                i_interpolation = fine_coarse.interpolation(2, ci0)
                i_interp = np.ones((i_ratio + 1,))
                i_interp[:-1] = fine_coarse.interpolation(2, ci0)

                shared_fine_points = source_grid.interpolated_points((ck0, cj0, ci0), (k_interp, j_interp, i_interp))

                fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 0, 0, 0] =  \
                   shared_fine_points[:-1, :-1, :-1]
                fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 0, 0, 1] =  \
                   shared_fine_points[:-1, :-1, 1:]
                fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 0, 1, 0] =  \
                   shared_fine_points[:-1, 1:, :-1]
                fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 0, 1, 1] =  \
                   shared_fine_points[:-1, 1:, 1:]
                fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 1, 0, 0] =  \
                   shared_fine_points[1:, :-1, :-1]
                fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 1, 0, 1] =  \
                   shared_fine_points[1:, :-1, 1:]
                fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 1, 1, 0] =  \
                   shared_fine_points[1:, 1:, :-1]
                fine_cp[fine_k_base : fine_k_base + k_ratio, fine_j_base : fine_j_base + j_ratio, fine_i_base : fine_i_base + i_ratio, 1, 1, 1] =  \
                   shared_fine_points[1:, 1:, 1:]

    return rqi.grid_from_cp(model,
                            fine_cp,
                            source_grid.crs_uuid,
                            ijk_handedness = 'right' if source_grid.grid_is_right_handed else 'left')


def _refined_unfaulted_grid(model, source_grid, fine_coarse):

    source_points = source_grid.points_ref()
    assert source_points is not None, 'geometry not available for refinement of unfaulted grid'

    # create a new, empty grid object
    grid = grr.Grid(model)

    # inherit attributes from source grid
    grid.grid_representation = 'IjkGrid'
    grid.extent_kji = fine_coarse.fine_extent_kji
    grid.nk, grid.nj, grid.ni = grid.extent_kji[0], grid.extent_kji[1], grid.extent_kji[2]
    grid.k_direction_is_down = source_grid.k_direction_is_down
    grid.grid_is_right_handed = source_grid.grid_is_right_handed
    grid.pillar_shape = source_grid.pillar_shape
    grid.has_split_coordinate_lines = source_grid.has_split_coordinate_lines
    grid.split_pillars_count = source_grid.split_pillars_count
    grid.k_gaps = source_grid.k_gaps
    if grid.k_gaps:
        grid.k_gap_after_array = np.zeros((grid.nk - 1,), dtype = bool)
        grid.k_raw_index_array = np.zeros((grid.nk,), dtype = int)
        # k gap arrays populated below
    # inherit the coordinate reference system used by the grid geometry
    grid.crs_uuid = source_grid.crs_uuid
    if source_grid.model is not model:
        model.duplicate_node(source_grid.model.root_for_uuid(grid.crs_uuid), add_as_part = True)
    grid.crs = rqc.Crs(model, uuid = grid.crs_uuid)

    refined_points = np.empty((grid.nk_plus_k_gaps + 1, grid.nj + 1, grid.ni + 1, 3))

    # log.debug(f'source grid: {source_grid.extent_kji}; k gaps: {source_grid.k_gaps}')
    # log.debug(f'refined grid: {grid.extent_kji}; k gaps: {grid.k_gaps}')
    fk0 = 0
    gaps_so_far = 0
    for ck0 in range(fine_coarse.coarse_extent_kji[0] + 1):
        end_k = (ck0 == fine_coarse.coarse_extent_kji[0])
        if end_k:
            k_ratio = 1
            k_interpolation = [0.0]
        else:
            k_ratio = fine_coarse.ratio(0, ck0)
            k_interpolation = fine_coarse.interpolation(0, ck0)
        one_if_gap = 1 if source_grid.k_gaps and ck0 < fine_coarse.coarse_extent_kji[
            0] - 1 and source_grid.k_gap_after_array[ck0] else 0
        for flk0 in range(k_ratio + one_if_gap):
            # log.debug(f'ck0: {ck0}; fk0: {fk0}; flk0: {flk0}; k_ratio: {k_ratio}; one_if_gap: {one_if_gap}; gaps so far: {gaps_so_far}')
            if flk0 < k_ratio:
                k_fraction = k_interpolation[flk0]
            else:
                k_fraction = 1.0
            if grid.k_gaps:
                if end_k:
                    k_plane = source_points[source_grid.k_raw_index_array[ck0 - 1] + 1, :, :, :]
                else:
                    k_plane = (k_fraction * source_points[source_grid.k_raw_index_array[ck0] + 1, :, :, :] +
                               (1.0 - k_fraction) * source_points[source_grid.k_raw_index_array[ck0], :, :, :])
                if flk0 == k_ratio:
                    grid.k_gap_after_array[fk0 - 1] = True
                elif fk0 < grid.nk:
                    grid.k_raw_index_array[fk0] = fk0 + gaps_so_far
            else:
                if end_k:
                    k_plane = source_points[ck0, :, :, :]
                else:
                    k_plane = k_fraction * source_points[ck0 + 1, :, :, :] + (1.0 -
                                                                              k_fraction) * source_points[ck0, :, :, :]
            fj0 = 0
            for cj0 in range(fine_coarse.coarse_extent_kji[1] + 1):
                end_j = (cj0 == fine_coarse.coarse_extent_kji[1])
                if end_j:
                    j_ratio = 1
                    j_interpolation = [0.0]
                else:
                    j_ratio = fine_coarse.ratio(1, cj0)
                    j_interpolation = fine_coarse.interpolation(1, cj0)
                for flj0 in range(j_ratio):
                    j_fraction = j_interpolation[flj0]
                    # note: shape of j_line will be different if there are split pillars in play
                    if end_j:
                        j_line = k_plane[cj0, :, :]
                    else:
                        j_line = j_fraction * k_plane[cj0 + 1, :, :] + (1.0 - j_fraction) * k_plane[cj0, :, :]

                    fi0 = 0
                    for ci0 in range(fine_coarse.coarse_extent_kji[2] + 1):
                        end_i = (ci0 == fine_coarse.coarse_extent_kji[2])
                        if end_i:
                            i_ratio = 1
                            i_interpolation = [0.0]
                        else:
                            i_ratio = fine_coarse.ratio(2, ci0)
                            i_interpolation = fine_coarse.interpolation(2, ci0)
                        for fli0 in range(i_ratio):
                            i_fraction = i_interpolation[fli0]
                            if end_i:
                                p = j_line[ci0, :]
                            else:
                                p = i_fraction * j_line[ci0 + 1, :] + (1.0 - i_fraction) * j_line[ci0, :]

                            refined_points[fk0 + gaps_so_far, fj0, fi0] = p

                            fi0 += 1

                    assert fi0 == fine_coarse.fine_extent_kji[2] + 1

                    fj0 += 1

            assert fj0 == fine_coarse.fine_extent_kji[1] + 1

            if flk0 == k_ratio:
                gaps_so_far += 1
            else:
                fk0 += 1

    assert fk0 == fine_coarse.fine_extent_kji[0] + 1
    assert grid.nk + gaps_so_far == grid.nk_plus_k_gaps

    grid.points_cached = refined_points

    grid.geometry_defined_for_all_pillars_cached = True
    grid.geometry_defined_for_all_cells_cached = True
    grid.array_cell_geometry_is_defined = np.full(tuple(grid.extent_kji), True, dtype = bool)

    return grid


def _inherit_properties(source_grid, grid, fine_coarse, inherit_realization, inherit_all_realizations):
    source_collection = source_grid.extract_property_collection()
    source_collection = rqp.selective_version_of_collection(source_collection,
                                                            indexable = 'cells',
                                                            count = 1,
                                                            points = False)
    # todo: support other indexable elements, especially columns
    collection = None
    if source_collection is not None:
        #  do not inherit the inactive property array by this mechanism
        collection = rqp.GridPropertyCollection()
        collection.set_grid(grid)
        collection.extend_imported_list_copying_properties_from_other_grid_collection(
            source_collection,
            refinement = fine_coarse,
            realization = inherit_realization,
            copy_all_realizations = inherit_all_realizations)
    return collection


def _set_parent_window(set_parent_window, source_grid, grid, fine_coarse):
    pw_grid_uuid = source_grid.uuid
    if isinstance(set_parent_window, str):
        if set_parent_window == 'grandparent':
            assert fine_coarse.within_coarse_box is None or (np.all(fine_coarse.within_coarse_box[0] == 0) and
                                                             np.all(fine_coarse.within_coarse_box[1]) == source_grid.extent_kji - 1),  \
               'attempt to set grandparent window for grid when parent window is present'
            source_fine_coarse = source_grid.parent_window
            if source_fine_coarse is not None and (source_fine_coarse.within_fine_box is not None or
                                                   source_fine_coarse.within_coarse_box is not None):
                assert source_fine_coarse.fine_extent_kji == source_fine_coarse.coarse_extent_kji, 'parentage involves refinement or coarsening'
                if source_fine_coarse.within_coarse_box is not None:
                    fine_coarse.within_coarse_box = source_fine_coarse.within_coarse_box
                else:
                    fine_coarse.within_coarse_box = source_fine_coarse.within_fine_box
                pw_grid_uuid = bu.uuid_from_string(
                    rqet.find_nested_tags_text(source_grid.root, ['ParentWindow', 'ParentGrid', 'UUID']))
        else:
            assert set_parent_window == 'parent', 'set_parent_window value not recognized: ' + set_parent_window
    grid.set_parent(pw_grid_uuid, True, fine_coarse)


def _establish_models_and_source_grid(epc_file, new_epc_file, source_grid, source_grid_uuid):
    assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
    if not epc_file:
        epc_file = source_grid.model.epc_file
        assert epc_file, 'unable to ascertain epc filename from grid object'
    if new_epc_file and epc_file and (
        (new_epc_file == epc_file) or
        (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
        new_epc_file = None
    model = None
    if new_epc_file:
        log.debug('creating fresh model for refined grid')
        model = rq.Model(epc_file = new_epc_file, new_epc = True, create_basics = True, create_hdf5_ext = True)
    if epc_file:
        model_in = rq.Model(epc_file)
        if source_grid is None:
            if source_grid_uuid is None:
                log.debug('using default source grid from existing epc')
                source_grid = model_in.grid()
            else:
                log.debug('selecting source grid from existing epc based on uuid')
                source_grid = grr.Grid(model_in, uuid = source_grid_uuid)
        else:
            if source_grid_uuid is not None:
                assert bu.matching_uuids(source_grid_uuid, source_grid.uuid)
            grid_uuid = source_grid.uuid
            log.debug('reloading source grid from existing epc file')
            source_grid = grr.Grid(model_in, uuid = grid_uuid)
        if model is None:
            model = model_in
    else:
        model_in = source_grid.model
    assert model_in is not None
    assert model is not None
    assert source_grid is not None
    assert source_grid.grid_representation in ['IjkGrid', 'IjkBlockGrid']
    return epc_file, model, model_in, source_grid
