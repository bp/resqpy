import numpy as np

import resqpy.olio.fine_coarse as fc
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
from .intervals_info import  IntervalsInfo

import logging

version = '24/11/2021'
log = logging.getLogger(__name__)
log.debug('grid.py version ' + version)


def extract_grid_parent(grid):
    if grid.extent_kji is None:
        grid.extract_extent_kji()
    if grid.parent_grid_uuid is not None:
        return grid.parent_grid_uuid
    grid.parent_window = None  # FineCoarse cell index mapping info with respect to parent
    grid.is_refinement = None
    pw_node = rqet.find_tag(grid.root, 'ParentWindow')
    if pw_node is None:
        return None
    # load a FineCoarse object as parent_window attribute and set parent_grid_uuid attribute
    grid.parent_grid_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(pw_node, ['ParentGrid', 'UUID']))
    assert grid.parent_grid_uuid is not None
    parent_grid_root = grid.model.root(uuid=grid.parent_grid_uuid)
    if parent_grid_root is None:
        log.warning('parent grid not present in model, unable to treat as local grid')
        return None
    # etxract parent grid extent directly from xml to avoid risk of circular references
    parent_grid_extent_kji = np.array(
        (rqet.find_tag_int(parent_grid_root, 'Nk'), rqet.find_tag_int(
            parent_grid_root, 'Nj'), rqet.find_tag_int(parent_grid_root, 'Ni')),
        dtype=int)
    parent_initials = []
    intervals_count_list = []
    parent_count_list_list = []
    child_count_list_list = []
    child_weight_list_list = []
    refining_flag = None  # gets set True if local grid is a refinement, False if a coarsening
    parent_box = np.zeros((2, 3), dtype=int)
    for axis in range(3):
        regrid_node = rqet.find_tag(pw_node, 'KJI'[axis] + 'Regrid')
        assert regrid_node is not None
        pii = rqet.find_tag_int(regrid_node, 'InitialIndexOnParentGrid')
        assert pii is not None and 0 <= pii < parent_grid_extent_kji[axis]
        parent_initials.append(pii)
        parent_box[0, axis] = pii
        intervals_node = rqet.find_tag(regrid_node, 'Intervals')
        if intervals_node is None:  # implicit one-to-one mapping
            intervals_count_list.append(1)
            parent_count_list_list.append(np.array(grid.extent_kji[axis], dtype=int))
            parent_box[1, axis] = parent_box[0, axis] + grid.extent_kji[axis] - 1
            assert parent_box[1, axis] < parent_grid_extent_kji[axis]
            child_count_list_list.append(np.array(grid.extent_kji[axis], dtype=int))
            child_weight_list_list.append(None)
        else:
            intervals_info = IntervalsInfo()
            intervals_count = rqet.find_tag_int(intervals_node, 'IntervalCount')
            assert intervals_count is not None and intervals_count > 0
            pcpi_node = rqet.find_tag(intervals_node, 'ParentCountPerInterval')
            assert pcpi_node is not None
            h5_key_pair = grid.model.h5_uuid_and_path_for_node(pcpi_node)
            assert h5_key_pair is not None
            grid.model.h5_array_element(h5_key_pair,
                                        index=None,
                                        cache_array=True,
                                        object=intervals_info,
                                        array_attribute='parent_count_per_interval',
                                        dtype='int')
            assert hasattr(intervals_info, 'parent_count_per_interval')
            assert intervals_info.parent_count_per_interval.ndim == 1 and intervals_info.parent_count_per_interval.size == intervals_count
            parent_box[1, axis] = parent_box[0, axis] + np.sum(intervals_info.parent_count_per_interval) - 1
            assert parent_box[1, axis] < parent_grid_extent_kji[axis]
            ccpi_node = rqet.find_tag(intervals_node, 'ChildCountPerInterval')
            assert ccpi_node is not None
            h5_key_pair = grid.model.h5_uuid_and_path_for_node(ccpi_node)
            assert h5_key_pair is not None
            grid.model.h5_array_element(h5_key_pair,
                                        index=None,
                                        cache_array=True,
                                        object=intervals_info,
                                        array_attribute='child_count_per_interval',
                                        dtype='int')
            assert hasattr(intervals_info, 'child_count_per_interval')
            assert intervals_info.child_count_per_interval.ndim == 1 and intervals_info.child_count_per_interval.size == intervals_count
            assert np.sum(intervals_info.child_count_per_interval) == grid.extent_kji[
                axis]  # assumes both local and parent grids are IjkGrids
            for interval in range(intervals_count):
                if intervals_info.child_count_per_interval[interval] == intervals_info.parent_count_per_interval[
                    interval]:
                    continue  # one-to-one
                if refining_flag is None:
                    refining_flag = (intervals_info.child_count_per_interval[interval] >
                                     intervals_info.parent_count_per_interval[interval])
                assert refining_flag == (intervals_info.child_count_per_interval[interval] >
                                         intervals_info.parent_count_per_interval[interval]), \
                    'mixture of refining and coarsening in one local grid – allowed by RESQML but not handled by this code'
                if refining_flag:
                    assert intervals_info.child_count_per_interval[interval] % intervals_info.parent_count_per_interval[
                        interval] == 0, \
                        'within a single refinement interval, fine and coarse cell boundaries are not obviously aligned'
                else:
                    assert intervals_info.parent_count_per_interval[interval] % intervals_info.child_count_per_interval[
                        interval] == 0, \
                        'within a single coarsening interval, fine and coarse cell boundaries are not obviously aligned'
            ccw_node = rqet.find_tag(intervals_node, 'ChildCellWeights')
            if ccw_node is None:
                intervals_info.child_cell_weights = None
            else:
                h5_key_pair = grid.model.h5_uuid_and_path_for_node(ccw_node)
                assert h5_key_pair is not None
                grid.model.h5_array_element(h5_key_pair,
                                            index=None,
                                            cache_array=True,
                                            object=intervals_info,
                                            array_attribute='child_cell_weights',
                                            dtype='float')
                assert hasattr(intervals_info, 'child_cell_weights')
                assert intervals_info.child_cell_weights.ndim == 1 and intervals_info.child_cell_weights.size == \
                       grid.extent_kji[
                           axis]
            intervals_count_list.append(intervals_count)
            parent_count_list_list.append(intervals_info.parent_count_per_interval)
            child_count_list_list.append(intervals_info.child_count_per_interval)
            child_weight_list_list.append(intervals_info.child_cell_weights)
    cell_overlap_node = rqet.find_tag(pw_node, 'CellOverlap')
    if cell_overlap_node is not None:
        log.warning('ignoring cell overlap information in grid relationship')
    omit_node = rqet.find_tag(pw_node, 'OmitParentCells')
    if omit_node is not None:
        log.warning('unable to handle parent cell omissions in local grid definition – ignoring')
    # todo: handle omissions

    if refining_flag is None:
        log.warning('local grid has no refinement nor coarsening – treating as a refined grid')
        refining_flag = True
    grid.is_refinement = refining_flag

    if refining_flag:  # local grid is a refinement
        grid.parent_window = fc.FineCoarse(grid.extent_kji,
                                           parent_box[1] - parent_box[0] + 1,
                                           within_coarse_box=parent_box)
        for axis in range(3):
            if intervals_count_list[axis] == 1:
                grid.parent_window.set_constant_ratio(axis)
                constant_ratio = grid.extent_kji[axis] // (parent_box[1, axis] - parent_box[0, axis] + 1)
                ratio_vector = None
            else:
                constant_ratio = None
                ratio_vector = child_count_list_list[axis] // parent_count_list_list[axis]
                grid.parent_window.set_ratio_vector(axis, ratio_vector)
            if child_weight_list_list[axis] is None:
                grid.parent_window.set_equal_proportions(axis)
            else:
                proportions_list = []
                place = 0
                for coarse_slice in range(parent_box[1, axis] - parent_box[0, axis] + 1):
                    if ratio_vector is None:
                        proportions_list.append(np.array(child_weight_list_list[axis][place:place +
                                                                                            constant_ratio]))
                        place += constant_ratio
                    else:
                        proportions_list.append(
                            np.array(child_weight_list_list[axis][place:place + ratio_vector[coarse_slice]]))
                        place += ratio_vector[coarse_slice]
                grid.parent_window.set_proportions_list_of_vectors(axis, proportions_list)

    else:  # local grid is a coarsening
        grid.parent_window = fc.FineCoarse(parent_box[1] - parent_box[0] + 1,
                                           grid.extent_kji,
                                           within_fine_box=parent_box)
        for axis in range(3):
            if intervals_count_list[axis] == 1:
                grid.parent_window.set_constant_ratio(axis)
                constant_ratio = (parent_box[1, axis] - parent_box[0, axis] + 1) // grid.extent_kji[axis]
                ratio_vector = None
            else:
                constant_ratio = None
                ratio_vector = parent_count_list_list[axis] // child_count_list_list[axis]
                grid.parent_window.set_ratio_vector(axis, ratio_vector)
            if child_weight_list_list[axis] is None:
                grid.parent_window.set_equal_proportions(axis)
            else:
                proportions_list = []
                place = 0
                for coarse_slice in range(grid.extent_kji[axis]):
                    if ratio_vector is None:
                        proportions_list.append(np.array(child_weight_list_list[axis][place:place +
                                                                                            constant_ratio]))
                        place += constant_ratio
                    else:
                        proportions_list.append(
                            np.array(child_weight_list_list[axis][place:place + ratio_vector[coarse_slice]]))
                        place += ratio_vector[coarse_slice]
                grid.parent_window.set_proportions_list_of_vectors(axis, proportions_list)

    grid.parent_window.assert_valid()

    return grid.parent_grid_uuid
