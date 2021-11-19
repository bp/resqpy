"""High level add faults function."""

import logging

log = logging.getLogger(__name__)

import os
import numpy as np

import resqpy.crs as rqcrs
import resqpy.grid as grr
import resqpy.lines as rql
import resqpy.model as rq
import resqpy.olio.grid_functions as gf
import resqpy.olio.simple_lines as sl
import resqpy.olio.vector_utilities as vec
import resqpy.olio.xml_et as rqet

from resqpy.derived_model.dm_common import __prepare_simple_inheritance, __write_grid
from resqpy.derived_model.dm_copy_grid import copy_grid


def add_faults(epc_file,
               source_grid,
               polylines = None,
               lines_file_list = None,
               lines_crs_uuid = None,
               full_pillar_list_dict = None,
               left_right_throw_dict = None,
               create_gcs = True,
               inherit_properties = False,
               inherit_realization = None,
               inherit_all_realizations = False,
               new_grid_title = None,
               new_epc_file = None):
    """Extends epc file with a new grid which is a version of the source grid with new curtain fault(s) added.

    arguments:
       epc_file (string): file name to rewrite the model's xml to; if source grid is None, model is loaded from this file
       source_grid (grid.Grid object, optional): if None, the epc_file is loaded and it should contain one ijk grid object
          (or one 'ROOT' grid) which is used as the source grid
       polylines (lines.PolylineSet or list of lines.Polyline, optional): list of poly lines for which curtain faults
          are to be added; either this or lines_file_list or full_pillar_list_dict must be present
       lines_file_list (list of str, optional): a list of file paths, each containing one or more poly lines in simple
          ascii format§; see notes; either this or polylines or full_pillar_list_dicr must be present
       lines_crs_uuid (uuid, optional): if present, the uuid of a coordinate reference system with which to interpret
          the contents of the lines files; if None, the crs used by the grid will be assumed
       full_pillar_list_dict (dict mapping str to list of pairs of ints, optional): dictionary mapping from a fault name
          to a list of pairs of ints being the ordered neigbouring primary pillar (j0, i0) defining the curtain fault;
          either this or polylines or lines_file_list must be present
       left_right_throw_dict (dict mapping str to pair of floats, optional): dictionary mapping from a fault name to a
          pair of floats being the semi-throw adjustment on the left and the right of the fault (see notes); semi-throw
          values default to (+0.5, -0.5)
       create_gcs (boolean, default True): if True, and faults are being defined by lines, a grid connection set is
          created with one feature per new fault and associated organisational objects are also created; ignored if
          lines_file_list is None
       inherit_properties (boolean, default False): if True, the new grid will have a copy of any properties associated
          with the source grid
       inherit_realization (int, optional): realization number for which properties will be inherited; ignored if
          inherit_properties is False
       inherit_all_realizations (boolean, default False): if True (and inherit_realization is None), properties for all
          realizations will be inherited; if False, only properties with a realization of None are inherited; ignored if
          inherit_properties is False or inherit_realization is not None
       new_grid_title (string): used as the citation title text for the new grid object
       new_epc_file (string, optional): if None, the source epc_file is extended with the new grid object; if present,
          a new epc file (& associated h5 file) is created to contain the unsplit grid (& crs)

    returns:
       a new grid (grid.Grid object) which is a copy of the source grid with the structure modified to incorporate
       the new faults

    notes:
       full_pillar_list_dict is typically generated by Grid.make_face_sets_from_pillar_lists();
       pillars will be split as needed to model the new faults, though existing splits will be used as appropriate, so
       this function may also be used to add a constant to the throw of existing faults;
       the left_right_throw_dict contains a pair of floats for each fault name (as found in keys of full_pillar_list_dict);
       these throw values are lengths in the uom of the crs used by the grid (which must have the same xy units as z units);

       this function does not add a GridConnectionSet to the model – calling code may wish to do that
    """

    log.info('adding faults')

    assert epc_file or new_epc_file, 'epc file name not specified'
    assert epc_file or source_grid is not None, 'neither epc file name nor source grid supplied'
    if new_epc_file and epc_file and (
        (new_epc_file == epc_file) or
        (os.path.exists(new_epc_file) and os.path.exists(epc_file) and os.path.samefile(new_epc_file, epc_file))):
        new_epc_file = None
    if source_grid is None:
        model = rq.Model(epc_file)
        source_grid = model.grid()  # requires there to be exactly one grid in model
    else:
        model = source_grid.model
    assert source_grid.grid_representation in ['IjkGrid', 'IjkBlockGrid']  # unstructured grids not catered for
    assert model is not None
    assert len([arg for arg in (polylines, lines_file_list, full_pillar_list_dict) if arg is not None]) == 1

    # take a copy of the resqpy grid object, without writing to hdf5 or creating xml
    # the copy will be a Grid, even if the source is a RegularGrid
    grid = copy_grid(source_grid, model)
    grid_crs = rqcrs.Crs(model, uuid = grid.crs_uuid)
    assert grid_crs is not None

    if isinstance(polylines, rql.PolylineSet):
        polylines = polylines.convert_to_polylines()

    composite_face_set_dict = {}

    # build pillar list dict for polylines if necessary
    if full_pillar_list_dict is None:
        full_pillar_list_dict = {}
        __populate_composite_face_sets_for_polylines(model, grid, polylines, lines_crs_uuid, grid_crs, lines_file_list,
                                                     full_pillar_list_dict, composite_face_set_dict)

    else:  # populate composite face set dictionary from full pillar list
        __populate_composite_face_sets_for_pillar_lists(source_grid, full_pillar_list_dict, composite_face_set_dict)

    # log.debug(f'full_pillar_list_dict:\n{full_pillar_list_dict}')

    __process_full_pillar_list_dict(grid, full_pillar_list_dict, left_right_throw_dict)

    collection = __prepare_simple_inheritance(grid, source_grid, inherit_properties, inherit_realization,
                                              inherit_all_realizations)
    # todo: recompute depth properties (and volumes, cell lengths etc. if being strict)

    if new_grid_title is None or len(new_grid_title) == 0:
        new_grid_title = 'copy of ' + str(rqet.citation_title_for_node(source_grid.root)) + ' with added faults'

    # write model
    if new_epc_file:
        __write_grid(new_epc_file, grid, property_collection = collection, grid_title = new_grid_title, mode = 'w')
    else:
        ext_uuid = model.h5_uuid()

        __write_grid(epc_file,
                     grid,
                     ext_uuid = ext_uuid,
                     property_collection = collection,
                     grid_title = new_grid_title,
                     mode = 'a')

    # create grid connection set if requested
    __create_gcs_if_requested(create_gcs, composite_face_set_dict, new_epc_file, grid)

    return grid


def __make_face_sets_for_new_lines(new_lines, face_set_id, grid, full_pillar_list_dict, composite_face_set_dict):
    """Adds entries to full_pillar_list_dict & composite_face_set_dict for new lines."""
    pillar_list_list = sl.nearest_pillars(new_lines, grid)
    face_set_dict, full_pll_dict = grid.make_face_sets_from_pillar_lists(pillar_list_list, face_set_id)
    for key, pll in full_pll_dict.items():
        full_pillar_list_dict[key] = pll
    for key, fs_info in face_set_dict.items():
        composite_face_set_dict[key] = fs_info


def __populate_composite_face_sets_for_pillar_lists(grid, full_pillar_list_dict, composite_face_set_dict):
    for key, pillar_list in full_pillar_list_dict.items():
        face_set_dict, _ = grid.make_face_sets_from_pillar_lists([pillar_list], key)
        for k, fs_info in face_set_dict.items():
            composite_face_set_dict[k] = fs_info


def __fault_from_pillar_list(grid, full_pillar_list, delta_throw_left, delta_throw_right):
    """Creates and/or adjusts throw on a single fault defined by a full pillar list, in memory.

    arguments:
       grid (grid.Grid): the grid object to be adjusted in memory (should have originally been copied
          without the hdf5 arrays having been written yet, nor xml created)
       full_pillar_list (list of pairs of ints (j0, i0)): the full list of primary pillars defining
          the fault; neighbouring pairs must differ by exactly one in either j0 or i0 but not both
       delta_throw_left (float): the amount to add to the 'depth' of points to the left of the line
          when viewed from above, looking along the line in the direction of the pillar list entries;
          units are implicitly the length units of the crs used by the grid; see notes about 'depth'
       delta_throw_right (float): as for delta_throw_left but applied to points to the right of the
          line
    """

    # this function introduces new data into the RESQML arrays representing split pillars
    # familiarity with those array representations is needed if working on this function

    if full_pillar_list is None or len(full_pillar_list) < 3:
        return
    assert grid.z_units() == grid.xy_units()
    grid.cache_all_geometry_arrays()
    assert hasattr(grid, 'points_cached')
    # make grid into a faulted grid if hitherto unfaulted
    if not grid.has_split_coordinate_lines:
        grid.points_cached = grid.points_cached.reshape((grid.nk_plus_k_gaps + 1, (grid.nj + 1) * (grid.ni + 1), 3))
        grid.split_pillar_indices_cached = np.array([], dtype = int)
        grid.cols_for_split_pillars = np.array([], dtype = int)
        grid.cols_for_split_pillars_cl = np.array([], dtype = int)
        grid.has_split_coordinate_lines = True
    assert grid.points_cached.ndim == 3
    if len(grid.cols_for_split_pillars_cl) == 0:
        cl = 0
    else:
        cl = grid.cols_for_split_pillars_cl[-1]
    original_p = np.zeros((grid.nk_plus_k_gaps + 1, 3), dtype = float)
    n_primaries = (grid.nj + 1) * (grid.ni + 1)
    for p_index in range(1, len(full_pillar_list) - 1):
        primary_ji0 = full_pillar_list[p_index]
        primary = primary_ji0[0] * (grid.ni + 1) + primary_ji0[1]
        p_vector = np.array(__pillar_vector(grid, primary), dtype = float)
        if p_vector is None:
            continue
        throw_left_vector = np.expand_dims(delta_throw_left * p_vector, axis = 0)
        throw_right_vector = np.expand_dims(delta_throw_right * p_vector, axis = 0)
        # log.debug(f'T: p ji0: {primary_ji0}; p vec: {p_vector}; left v: {throw_left_vector}; right v: {throw_right_vector}')
        existing_foursome = grid.pillar_foursome(primary_ji0, none_if_unsplit = False)
        lr_foursome = gf.left_right_foursome(full_pillar_list, p_index)
        cl = __processs_foursome(grid, n_primaries, primary, original_p, existing_foursome, lr_foursome, primary_ji0,
                                 throw_right_vector, throw_left_vector, cl)


def __pillar_vector(grid, p_index):
    # return a unit vector for direction of pillar, in direction of increasing k
    if np.all(np.isnan(grid.points_cached[:, p_index])):
        return None
    k_top = 0
    while np.any(np.isnan(grid.points_cached[k_top, p_index])):
        k_top += 1
    k_bot = grid.nk_plus_k_gaps - 1
    while np.any(np.isnan(grid.points_cached[k_bot, p_index])):
        k_bot -= 1
    if k_bot == k_top:  # following coded to treat None directions as downwards
        if grid.k_direction_is_down is False:
            if grid.z_inc_down() is False:
                return (0.0, 0.0, 1.0)
            else:
                return (0.0, 0.0, -1.0)
        else:
            if grid.z_inc_down() is False:
                return (0.0, 0.0, -1.0)
            else:
                return (0.0, 0.0, 1.0)
    else:
        return vec.unit_vector(grid.points_cached[k_bot, p_index] - grid.points_cached[k_top, p_index])


def __extend_points_cached(grid, exist_p):
    s = grid.points_cached.shape
    e = np.empty((s[0], s[1] + 1, s[2]), dtype = float)
    e[:, :-1, :] = grid.points_cached
    e[:, -1, :] = grid.points_cached[:, exist_p, :]
    grid.points_cached = e


def __np_int_extended(a, i):
    e = np.empty(a.size + 1, dtype = int)
    e[:-1] = a
    e[-1] = i
    return e


def __create_gcs_if_requested(create_gcs, composite_face_set_dict, new_epc_file, grid):
    if create_gcs and len(composite_face_set_dict) > 0:
        if new_epc_file is not None:
            grid_uuid = grid.uuid
            model = rq.Model(new_epc_file)
            grid = grr.Grid(model, root = model.root(uuid = grid_uuid), find_properties = False)
        grid.set_face_set_gcs_list_from_dict(composite_face_set_dict, create_organizing_objects_where_needed = True)
        combined_gcs = grid.face_set_gcs_list[0]
        for gcs in grid.face_set_gcs_list[1:]:
            combined_gcs.append(gcs)
        combined_gcs.write_hdf5()
        combined_gcs.create_xml(title = 'faults added from lines')
        grid.clear_face_sets()
        grid.model.store_epc()


def __processs_foursome(grid, n_primaries, primary, original_p, existing_foursome, lr_foursome, primary_ji0,
                        throw_right_vector, throw_left_vector, cl):
    p_j, p_i = primary_ji0
    # log.debug(f'P: p ji0: {primary_ji0}; e foursome:\n{existing_foursome}; lr foursome:\n{lr_foursome}')
    for exist_p in np.unique(existing_foursome):
        exist_lr = None
        new_p_made = False
        for jp in range(2):
            if (p_j == 0 and jp == 0) or (p_j == grid.nj and jp == 1):
                continue
            for ip in range(2):
                if (p_i == 0 and ip == 0) or (p_i == grid.ni and ip == 1):
                    continue
                if existing_foursome[jp, ip] != exist_p:
                    continue
                if exist_lr is None:
                    original_p[:] = grid.points_cached[:, exist_p, :]
                    exist_lr = lr_foursome[jp, ip]
                    # log.debug(f'A: p ji0: {primary_ji0}; exist_p: {exist_p}; jp,ip: {(jp,ip)}; exist_lr: {exist_lr}')
                    grid.points_cached[:, exist_p, :] += throw_right_vector if exist_lr else throw_left_vector
                    continue
                if lr_foursome[jp, ip] == exist_lr:
                    continue
                natural_col = (p_j + jp - 1) * grid.ni + p_i + ip - 1
                if exist_p != primary:  # remove one of the columns currently assigned to exist_p
                    extra_p = exist_p - n_primaries
                    # log.debug(f're-split: primary: {primary}; exist: {exist_p}; col: {natural_col}; extra: {extra_p}')
                    # log.debug(f'pre re-split: cols: {grid.cols_for_split_pillars}')
                    # log.debug(f'pre re-split: ccl:  {grid.cols_for_split_pillars_cl}')
                    assert grid.split_pillar_indices_cached[extra_p] == primary
                    if extra_p == 0:
                        start = 0
                    else:
                        start = grid.cols_for_split_pillars_cl[extra_p - 1]
                    found = False
                    for cols_index in range(start, start + grid.cols_for_split_pillars_cl[extra_p]):
                        if grid.cols_for_split_pillars[cols_index] == natural_col:
                            grid.cols_for_split_pillars = np.concatenate((grid.cols_for_split_pillars[:cols_index],
                                                                          grid.cols_for_split_pillars[cols_index + 1:]))
                            found = True
                            break
                    assert found
                    grid.cols_for_split_pillars_cl[extra_p:] -= 1
                    cl -= 1
                    assert grid.cols_for_split_pillars_cl[extra_p] > 0
                # log.debug(f'post re-split: cols: {grid.cols_for_split_pillars}')
                # log.debug(f'post re-split: ccl:  {grid.cols_for_split_pillars_cl}')
                if not new_p_made:  # create a new split of pillar
                    __extend_points_cached(grid, exist_p)
                    # log.debug(f'B: p ji0: {primary_ji0}; exist_p: {exist_p}; jp,ip: {(jp,ip)}; lr: {lr_foursome[jp, ip]}; c ji0: {natural_col}')
                    grid.points_cached[:, -1, :] = original_p + (throw_right_vector
                                                                 if lr_foursome[jp, ip] else throw_left_vector)
                    grid.split_pillar_indices_cached = __np_int_extended(grid.split_pillar_indices_cached, primary)
                    if grid.split_pillars_count is None:
                        grid.split_pillars_count = 0
                    grid.split_pillars_count += 1
                    grid.cols_for_split_pillars = __np_int_extended(grid.cols_for_split_pillars, natural_col)
                    cl += 1
                    grid.cols_for_split_pillars_cl = __np_int_extended(grid.cols_for_split_pillars_cl, cl)
                    new_p_made = True
                else:  # include this column in newly split version of pillar
                    # log.debug(f'C: p ji0: {primary_ji0}; exist_p: {exist_p}; jp,ip: {(jp,ip)}; lr: {lr_foursome[jp, ip]}; c ji0: {natural_col}')
                    grid.cols_for_split_pillars = __np_int_extended(grid.cols_for_split_pillars, natural_col)
                    cl += 1
                    grid.cols_for_split_pillars_cl[-1] = cl
    return cl


def __process_full_pillar_list_dict(grid, full_pillar_list_dict, left_right_throw_dict):
    for fault_key in full_pillar_list_dict:
        full_pillar_list = full_pillar_list_dict[fault_key]
        left_right_throw = None
        if left_right_throw_dict is not None:
            left_right_throw = left_right_throw_dict.get(fault_key)
        if left_right_throw is None:
            left_right_throw = (+0.5, -0.5)
        log.debug(
            f'generating fault {fault_key} pillar count {len(full_pillar_list)}; left, right throw {left_right_throw}')
        __fault_from_pillar_list(grid, full_pillar_list, left_right_throw[0], left_right_throw[1])


def __populate_composite_face_sets_for_polylines(model, grid, polylines, lines_crs_uuid, grid_crs, lines_file_list,
                                                 full_pillar_list_dict, composite_face_set_dict):
    lines_crs = None if lines_crs_uuid is None else rqcrs.Crs(model, uuid = lines_crs_uuid)
    if polylines:
        for i, polyline in enumerate(polylines):
            new_line = polyline.coordinates.copy()
            if polyline.crs_uuid is not None and polyline.crs_uuid != lines_crs_uuid:
                lines_crs_uuid = polyline.crs_uuid
                lines_crs = rqcrs.Crs(model, uuid = lines_crs_uuid)
            if lines_crs:
                lines_crs.convert_array_to(grid_crs, new_line)
            title = polyline.title if polyline.title else 'fault_' + str(i)
            __make_face_sets_for_new_lines([new_line], title, grid, full_pillar_list_dict, composite_face_set_dict)
    else:
        for filename in lines_file_list:
            new_lines = sl.read_lines(filename)
            if lines_crs is not None:
                for a in new_lines:
                    lines_crs.convert_array_to(grid_crs, a)
            _, f_name = os.path.split(filename)
            if f_name.lower().endswith('.dat'):
                face_set_id = f_name[:-4]
            else:
                face_set_id = f_name
            __make_face_sets_for_new_lines(new_lines, face_set_id, grid, full_pillar_list_dict, composite_face_set_dict)
