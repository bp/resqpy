"""High level gather ensemble function."""

import logging

log = logging.getLogger(__name__)

from time import time

import resqpy.grid as grr
import resqpy.model as rq
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp


def gather_ensemble(case_epc_list,
                    new_epc_file,
                    consolidate = True,
                    shared_grids = True,
                    shared_time_series = True,
                    create_epc_lookup = True):
    """Creates a composite resqml dataset by merging all parts from all models in list, assigning realization numbers.

    arguments:
       case_epc_list (list of strings): paths of individual realization epc files
       new_epc_file (string): path of new composite epc to be created (with paired hdf5 file)
       consolidate (boolean, default True): if True, simple parts are tested for equivalence and where similar enough
          a single shared object is established in the composite dataset
       shared_grids (boolean, default True): if True and consolidate is True, then grids are also consolidated
          with equivalence based on extent of grids (and citation titles if grid extents within the first case
          are not distinct); ignored if consolidate is False
       shared_time_series (boolean, default False): if True and consolidate is True, then time series are consolidated
          with equivalence based on title, without checking that timestamp lists are the same
       create_epc_lookup (boolean, default True): if True, a StringLookupTable is created to map from realization
          number to case epc path

    notes:
       property objects will have an integer realization number assigned, which matches the corresponding index into
       the case_epc_list;
       if consolidating with shared grids, then only properties will be gathered from realisations after the first and
       an exception will be raised if the grids are not matched between realisations
    """

    if not consolidate:
        shared_grids = False

    composite_model = rq.Model(new_epc_file, new_epc = True, create_basics = True, create_hdf5_ext = True)

    epc_lookup_dict = {}

    for r, case_epc in enumerate(case_epc_list):
        t_r_start = time()  # debug
        log.debug(f'gathering realisation {r}: {case_epc}')
        epc_lookup_dict[r] = case_epc
        case_model = rq.Model(case_epc)
        if r == 0:  # first case
            composite_model.copy_all_parts_from_other_model(case_model, realization = 0, consolidate = consolidate)
            if shared_time_series:
                host_ts_uuids = case_model.uuids(obj_type = 'TimeSeries')
                host_ts_titles = []
                for ts_uuid in host_ts_uuids:
                    host_ts_titles.append(case_model.title(uuid = ts_uuid))
            if shared_grids:
                host_grid_uuids = case_model.uuids(obj_type = 'IjkGridRepresentation')
                host_grid_shapes = []
                host_grid_titles = []
                title_match_required = False
                for grid_uuid in host_grid_uuids:
                    grid_root = case_model.root(uuid = grid_uuid)
                    host_grid_shapes.append(grr.extent_kji_from_root(grid_root))
                    host_grid_titles.append(rqet.citation_title_for_node(grid_root))
                if len(set(host_grid_shapes)) < len(host_grid_shapes):
                    log.warning(
                        'shapes of representative grids are not distinct, grid titles must match during ensemble gathering'
                    )
                    title_match_required = True
        else:  # subsequent cases
            composite_model.consolidation = None  # discard any previous mappings to limit dictionary growth
            if shared_time_series:
                for ts_uuid in case_model.uuids(obj_type = 'TimeSeries'):
                    ts_title = case_model.title(uuid = ts_uuid)
                    ts_index = host_ts_titles.index(ts_title)
                    host_ts_uuid = host_ts_uuids[ts_index]
                    composite_model.force_consolidation_uuid_equivalence(ts_uuid, host_ts_uuid)
            if shared_grids:
                for grid_uuid in case_model.uuids(obj_type = 'IjkGridRepresentation'):
                    grid_root = case_model.root(uuid = grid_uuid)
                    grid_extent = grr.extent_kji_from_root(grid_root)
                    host_index = None
                    if grid_extent in host_grid_shapes:
                        if title_match_required:
                            case_grid_title = rqet.citation_title_for_node(grid_root)
                            for host_grid_index in len(host_grid_uuids):
                                if grid_extent == host_grid_shapes[
                                        host_grid_index] and case_grid_title == host_grid_titles[host_grid_index]:
                                    host_index = host_grid_index
                                    break
                        else:
                            host_index = host_grid_shapes.index(grid_extent)
                    assert host_index is not None, 'failed to match grids when gathering ensemble'
                    composite_model.force_consolidation_uuid_equivalence(grid_uuid, host_grid_uuids[host_index])
                    grid_relatives = case_model.parts(related_uuid = grid_uuid)
                    t_props = 0.0
                    composite_h5_file_name = composite_model.h5_file_name()
                    composite_h5_uuid = composite_model.h5_uuid()
                    case_h5_file_name = case_model.h5_file_name()
                    for part in grid_relatives:
                        if 'Property' in part:
                            t_p_start = time()
                            composite_model.copy_part_from_other_model(case_model,
                                                                       part,
                                                                       realization = r,
                                                                       consolidate = True,
                                                                       force = shared_time_series,
                                                                       self_h5_file_name = composite_h5_file_name,
                                                                       h5_uuid = composite_h5_uuid,
                                                                       other_h5_file_name = case_h5_file_name)
                            t_props += time() - t_p_start
            else:
                composite_model.copy_all_parts_from_other_model(case_model, realization = r, consolidate = consolidate)
        log.debug(f'case time: {time() - t_r_start:.2f} secs')  # debug

    if create_epc_lookup and len(epc_lookup_dict):
        epc_lookup = rqp.StringLookup(composite_model, int_to_str_dict = epc_lookup_dict, title = 'ensemble epc table')
        epc_lookup.create_xml()

    composite_model.store_epc()

    log.info(f'{len(epc_lookup_dict)} realizations merged into ensemble {new_epc_file}')
