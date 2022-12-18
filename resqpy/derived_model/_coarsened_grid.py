"""High level coarsened grid function."""

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

import resqpy.derived_model._common as rqdm_c


def coarsened_grid(epc_file,
                   source_grid,
                   fine_coarse,
                   inherit_properties = False,
                   inherit_realization = None,
                   inherit_all_realizations = False,
                   set_parent_window = None,
                   infill_missing_geometry = True,
                   new_grid_title = None,
                   new_epc_file = None):
    """Generates a coarsened version of an unsplit source grid, optionally inheriting properties.

    arguments:
       epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
       source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
          (or one 'ROOT' grid) which is used as the source grid
       fine_coarse (resqpy.olio.fine_coarse.FineCoarse object): the mapping between cells in the fine (source) and
          coarse (output) grids
       inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
          with the source grid, with values upscaled or sampled
       inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
          inherit_properties is False
       inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
          realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
          inherit_properties is False or inherit_realization is not None
       set_parent_window (boolean or str, optional): if True or 'parent', the coarsened grid has its parent window attribute
          set; if False, the parent window is not set; if None, the default will be True if new_epc_file is None or False
          otherwise; if 'grandparent' then an intervening parent window with no refinement or coarsening will be skipped
          and its box used in the parent window for the new grid, relating directly to the original grid
       infill_missing_geometry (boolean, default True): if True, an attempt is made to generate grid geometry in the
          source grid wherever it is undefined; if False, any undefined geometry will result in an assertion failure
       new_grid_title (string): used as the citation title text for the new grid object
       new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
          a new epc file (& associated h5 file) is created to contain the refined grid (& crs)

    returns:
       new grid object being the coarsened grid; the epc and hdf5 files are written to as an intentional side effect

    note:
       this function coarsens an entire grid; to coarsen a local area of a grid, first use the extract_box function
       and then use this function on the extracted grid; in such a case, using a value of 'grandparent' for the
       set_parent_window argument will relate the coarsened grid back to the original
    """

    new_epc_file, model, source_grid = _establish_files_and_model(epc_file, new_epc_file, source_grid)

    if set_parent_window is None:
        set_parent_window = (new_epc_file is None)
    assert fine_coarse is not None and isinstance(fine_coarse, fc.FineCoarse)

    assert not source_grid.has_split_coordinate_lines, 'coarsening only available for unsplit grids: use other functions to heal faults first'

    if infill_missing_geometry and (not source_grid.geometry_defined_for_all_cells() or
                                    not source_grid.geometry_defined_for_all_pillars()):
        log.debug('attempting infill of geometry missing in source grid')
        source_grid.set_geometry_is_defined(treat_as_nan = None,
                                            treat_dots_as_nan = True,
                                            complete_partial_pillars = True,
                                            nullify_partial_pillars = False,
                                            complete_all = True)

    assert source_grid.geometry_defined_for_all_pillars(), 'coarsening requires geometry to be defined for all pillars'
    assert source_grid.geometry_defined_for_all_cells(), 'coarsening requires geometry to be defined for all cells'
    assert not source_grid.k_gaps, 'coarsening of grids with k gaps not currently supported'

    assert tuple(fine_coarse.fine_extent_kji) == tuple(source_grid.extent_kji),  \
           'fine_coarse mapping fine extent does not match that of source grid'
    fine_coarse.assert_valid()

    source_grid.cache_all_geometry_arrays()
    source_points = source_grid.points_ref().reshape((source_grid.nk + 1), (source_grid.nj + 1) * (source_grid.ni + 1),
                                                     3)

    # create a new, empty grid object
    grid = grr.Grid(model)

    # inherit attributes from source grid
    grid.grid_representation = 'IjkGrid'
    grid.extent_kji = fine_coarse.coarse_extent_kji
    grid.nk, grid.nj, grid.ni = grid.extent_kji[0], grid.extent_kji[1], grid.extent_kji[2]
    grid.k_direction_is_down = source_grid.k_direction_is_down
    grid.grid_is_right_handed = source_grid.grid_is_right_handed
    grid.pillar_shape = source_grid.pillar_shape
    grid.has_split_coordinate_lines = False
    grid.split_pillars_count = None
    # inherit the coordinate reference system used by the grid geometry
    grid.crs_uuid = source_grid.crs_uuid
    if source_grid.model is not model:
        model.duplicate_node(source_grid.model.root_for_uuid(grid.crs_uuid), add_as_part = True)
    grid.crs = rqc.Crs(model, grid.crs_uuid)

    coarsened_points = np.empty(
        (grid.nk + 1, (grid.nj + 1) * (grid.ni + 1), 3))  # note: gets reshaped after being populated

    k_ratio_constant = fine_coarse.constant_ratios[0]
    if k_ratio_constant:
        k_indices = None
    else:
        k_indices = np.empty(grid.nk + 1, dtype = int)
        k_indices[0] = 0
        for k in range(grid.nk):
            k_indices[k + 1] = k_indices[k] + fine_coarse.vector_ratios[0][k]
        assert k_indices[-1] == source_grid.nk

    for cjp in range(grid.nj + 1):
        for cji in range(grid.ni + 1):
            natural_coarse_pillar = cjp * (grid.ni + 1) + cji
            natural_fine_pillar = fine_coarse.fine_for_coarse_natural_pillar_index(natural_coarse_pillar)
            if k_ratio_constant:
                coarsened_points[:, natural_coarse_pillar, :] = source_points[0:source_grid.nk + 1:k_ratio_constant,
                                                                              natural_fine_pillar, :]
            else:
                coarsened_points[:, natural_coarse_pillar, :] = source_points[k_indices, natural_fine_pillar, :]

    grid.points_cached = coarsened_points.reshape(((grid.nk + 1), (grid.nj + 1), (grid.ni + 1), 3))

    grid.geometry_defined_for_all_pillars_cached = True
    grid.geometry_defined_for_all_cells_cached = True
    grid.array_cell_geometry_is_defined = np.full(tuple(grid.extent_kji), True, dtype = bool)

    collection = None
    if inherit_properties:
        source_collection = source_grid.extract_property_collection()
        if source_collection is not None:
            collection = rqp.GridPropertyCollection()
            collection.set_grid(grid)
            collection.extend_imported_list_copying_properties_from_other_grid_collection(
                source_collection,
                coarsening = fine_coarse,
                realization = inherit_realization,
                copy_all_realizations = inherit_all_realizations)

    _set_parent_window_in_grid(set_parent_window, source_grid, grid, fine_coarse)

    # write grid
    if new_grid_title is None or len(new_grid_title) == 0:
        new_grid_title = 'grid coarsened from ' + str(rqet.citation_title_for_node(source_grid.root))

    model.h5_release()
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


def _set_parent_window_in_grid(set_parent_window, source_grid, grid, fine_coarse):
    if set_parent_window:
        pw_grid_uuid = source_grid.uuid
        if isinstance(set_parent_window, str):
            if set_parent_window == 'grandparent':
                assert fine_coarse.within_fine_box is None or (np.all(fine_coarse.within_fine_box[0] == 0) and
                                                               np.all(fine_coarse.within_fine_box[1]) == source_grid.extent_kji - 1),  \
                   'attempt to set grandparent window for grid when parent window is present'
                source_fine_coarse = source_grid.parent_window
                if source_fine_coarse is not None and (source_fine_coarse.within_fine_box is not None or
                                                       source_fine_coarse.within_coarse_box is not None):
                    assert source_fine_coarse.fine_extent_kji == source_fine_coarse.coarse_extent_kji, 'parentage involves refinement or coarsening'
                    if source_fine_coarse.within_fine_box is not None:
                        fine_coarse.within_fine_box = source_fine_coarse.within_fine_box
                    else:
                        fine_coarse.within_fine_box = source_fine_coarse.within_coarse_box
                    pw_grid_uuid = bu.uuid_from_string(
                        rqet.find_nested_tags_text(source_grid.root, ['ParentWindow', 'ParentGrid', 'UUID']))
            else:
                assert set_parent_window == 'parent', 'set_parent_window value not recognized: ' + set_parent_window
        grid.set_parent(pw_grid_uuid, False, fine_coarse)


def _establish_files_and_model(epc_file, new_epc_file, source_grid):
    assert epc_file or new_epc_file, 'epc file name not specified'
    if new_epc_file and epc_file and (
        (new_epc_file == epc_file) or
        (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
        new_epc_file = None
    assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
    if epc_file:
        model = rq.Model(epc_file)
        if source_grid is None:
            source_grid = model.grid()  # requires there to be exactly one grid in model (or one named ROOT)
    else:
        model = source_grid.model
    assert source_grid.grid_representation == 'IjkGrid'
    assert model is not None
    return new_epc_file, model, source_grid
