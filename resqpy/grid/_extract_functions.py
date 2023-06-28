"""A submodule containing functions for extracting grid information"""

import logging

import numpy as np

import resqpy.olio.fine_coarse as fc
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.property as rprop
import resqpy.grid
import resqpy.grid._intervals_info as grr_ii
import resqpy.grid._defined_geometry as grr_dg

log = logging.getLogger(__name__)


def extract_grid_parent(grid):
    """Returns the uuid of the parent grid for the supplied grid"""
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
    parent_grid_root = grid.model.root(uuid = grid.parent_grid_uuid)
    if parent_grid_root is None:
        log.warning('parent grid not present in model, unable to treat as local grid')
        return None
    # etxract parent grid extent directly from xml to avoid risk of circular references
    parent_grid_extent_kji = np.array((rqet.find_tag_int(
        parent_grid_root, 'Nk'), rqet.find_tag_int(parent_grid_root, 'Nj'), rqet.find_tag_int(parent_grid_root, 'Ni')),
                                      dtype = int)
    parent_initials = []
    intervals_count_list = []
    parent_count_list_list = []
    child_count_list_list = []
    child_weight_list_list = []
    refining_flag = None  # gets set True if local grid is a refinement, False if a coarsening
    parent_box = np.zeros((2, 3), dtype = int)
    for axis in range(3):
        refining_flag = __process_axis(axis, child_count_list_list, child_weight_list_list, grid, intervals_count_list,
                                       parent_box, parent_count_list_list, parent_grid_extent_kji, parent_initials,
                                       pw_node, refining_flag)
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
        __extract_refined_parent(child_count_list_list, child_weight_list_list, grid, intervals_count_list, parent_box,
                                 parent_count_list_list)

    else:  # local grid is a coarsening
        __extract_coarsening_parent(child_count_list_list, child_weight_list_list, grid, intervals_count_list,
                                    parent_box, parent_count_list_list)

    grid.parent_window.assert_valid()

    return grid.parent_grid_uuid


def __extract_coarsening_parent(child_count_list_list, child_weight_list_list, grid, intervals_count_list, parent_box,
                                parent_count_list_list):
    grid.parent_window = fc.FineCoarse(parent_box[1] - parent_box[0] + 1, grid.extent_kji, within_fine_box = parent_box)
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
                    proportions_list.append(np.array(child_weight_list_list[axis][place:place + constant_ratio]))
                    place += constant_ratio
                else:
                    proportions_list.append(
                        np.array(child_weight_list_list[axis][place:place + ratio_vector[coarse_slice]]))
                    place += ratio_vector[coarse_slice]
            grid.parent_window.set_proportions_list_of_vectors(axis, proportions_list)


def __extract_refined_parent(child_count_list_list, child_weight_list_list, grid, intervals_count_list, parent_box,
                             parent_count_list_list):
    grid.parent_window = fc.FineCoarse(grid.extent_kji,
                                       parent_box[1] - parent_box[0] + 1,
                                       within_coarse_box = parent_box)
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
                    proportions_list.append(np.array(child_weight_list_list[axis][place:place + constant_ratio]))
                    place += constant_ratio
                else:
                    proportions_list.append(
                        np.array(child_weight_list_list[axis][place:place + ratio_vector[coarse_slice]]))
                    place += ratio_vector[coarse_slice]
            grid.parent_window.set_proportions_list_of_vectors(axis, proportions_list)


def __process_axis(axis, child_count_list_list, child_weight_list_list, grid, intervals_count_list, parent_box,
                   parent_count_list_list, parent_grid_extent_kji, parent_initials, pw_node, refining_flag):
    regrid_node = rqet.find_tag(pw_node, 'KJI'[axis] + 'Regrid')
    assert regrid_node is not None
    pii = rqet.find_tag_int(regrid_node, 'InitialIndexOnParentGrid')
    assert pii is not None and 0 <= pii < parent_grid_extent_kji[axis]
    parent_initials.append(pii)
    parent_box[0, axis] = pii
    intervals_node = rqet.find_tag(regrid_node, 'Intervals')
    if intervals_node is None:  # implicit one-to-one mapping
        intervals_count_list.append(1)
        parent_count_list_list.append(np.array(grid.extent_kji[axis], dtype = int))
        parent_box[1, axis] = parent_box[0, axis] + grid.extent_kji[axis] - 1
        assert parent_box[1, axis] < parent_grid_extent_kji[axis]
        child_count_list_list.append(np.array(grid.extent_kji[axis], dtype = int))
        child_weight_list_list.append(None)
    else:
        intervals_info = grr_ii.IntervalsInfo()
        intervals_count = rqet.find_tag_int(intervals_node, 'IntervalCount')
        assert intervals_count is not None and intervals_count > 0
        pcpi_node = rqet.find_tag(intervals_node, 'ParentCountPerInterval')
        assert pcpi_node is not None
        h5_key_pair = grid.model.h5_uuid_and_path_for_node(pcpi_node)
        assert h5_key_pair is not None
        grid.model.h5_array_element(h5_key_pair,
                                    index = None,
                                    cache_array = True,
                                    object = intervals_info,
                                    array_attribute = 'parent_count_per_interval',
                                    dtype = 'int')
        assert hasattr(intervals_info, 'parent_count_per_interval')
        assert intervals_info.parent_count_per_interval.ndim == 1 and intervals_info.parent_count_per_interval.size == intervals_count
        parent_box[1, axis] = parent_box[0, axis] + np.sum(intervals_info.parent_count_per_interval) - 1
        assert parent_box[1, axis] < parent_grid_extent_kji[axis]
        ccpi_node = rqet.find_tag(intervals_node, 'ChildCountPerInterval')
        assert ccpi_node is not None
        h5_key_pair = grid.model.h5_uuid_and_path_for_node(ccpi_node)
        assert h5_key_pair is not None
        grid.model.h5_array_element(h5_key_pair,
                                    index = None,
                                    cache_array = True,
                                    object = intervals_info,
                                    array_attribute = 'child_count_per_interval',
                                    dtype = 'int')
        assert hasattr(intervals_info, 'child_count_per_interval')
        assert intervals_info.child_count_per_interval.ndim == 1 and intervals_info.child_count_per_interval.size == intervals_count
        assert np.sum(intervals_info.child_count_per_interval) == grid.extent_kji[
            axis]  # assumes both local and parent grids are IjkGrids
        for interval in range(intervals_count):
            if intervals_info.child_count_per_interval[interval] == intervals_info.parent_count_per_interval[interval]:
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
                                        index = None,
                                        cache_array = True,
                                        object = intervals_info,
                                        array_attribute = 'child_cell_weights',
                                        dtype = 'float')
            assert hasattr(intervals_info, 'child_cell_weights')
            assert intervals_info.child_cell_weights.ndim == 1 and intervals_info.child_cell_weights.size == \
                   grid.extent_kji[
                       axis]
        intervals_count_list.append(intervals_count)
        parent_count_list_list.append(intervals_info.parent_count_per_interval)
        child_count_list_list.append(intervals_info.child_count_per_interval)
        child_weight_list_list.append(intervals_info.child_cell_weights)
    return refining_flag


def extract_extent_kji(grid):
    """Returns the grid extent; for IJK grids this is a 3 integer numpy array, order is Nk, Nj, Ni.

    returns:
       numpy int array of shape (3,) being number of cells in k, j & i axes respectively;
       the return value is cached in attribute extent_kji, which can alternatively be referenced
       directly by calling code as the value is set from xml on initialisation
    """

    if grid.extent_kji is not None:
        return grid.extent_kji
    grid.extent_kji = np.ones(3, dtype = 'int')  # todo: handle other varieties of grid
    grid.extent_kji[0] = int(rqet.find_tag(grid.root, 'Nk').text)
    grid.extent_kji[1] = int(rqet.find_tag(grid.root, 'Nj').text)
    grid.extent_kji[2] = int(rqet.find_tag(grid.root, 'Ni').text)
    return grid.extent_kji


def extract_grid_is_right_handed(grid):
    """Returns boolean indicating whether grid IJK axes are right handed, as stored in xml.

    returns:
       boolean: True if grid is right handed; False if left handed

    notes:
       this is the actual handedness of the IJK indexing of grid cells;
       the coordinate reference system has its own implicit handedness for xyz axes;
       Nexus requires the IJK space to be righthanded so if it is not, the handedness of the xyz space is
       falsified when exporting for Nexus (as Nexus allows xyz to be right or lefthanded and it is the
       handedness of the IJK space with respect to the xyz space that matters)
    """

    if grid.grid_is_right_handed is not None:
        return grid.grid_is_right_handed
    rh_node = grid.resolve_geometry_child('GridIsRighthanded')
    if rh_node is None:
        return None
    grid.grid_is_right_handed = (rh_node.text.lower() == 'true')
    return grid.grid_is_right_handed


def extract_k_direction_is_down(grid):
    """Returns boolean indicating whether increasing K indices are generally for deeper cells, as stored in xml.

    returns:
       boolean: True if increasing K generally indicates increasing depth

    notes:
       resqml allows layers to fold back over themselves, so the relationship between k and depth might not
       be monotonic;
       higher level code sometimes requires k to increase with depth;
       independently of this, z values may increase upwards or downwards in a coordinate reference system;
       this method does not modify the grid_is_righthanded indicator
    """

    if grid.k_direction_is_down is not None:
        return grid.k_direction_is_down
    k_dir_node = grid.resolve_geometry_child('KDirection')
    if k_dir_node is None:
        return None
    grid.k_direction_is_down = (k_dir_node.text.lower() == 'down')
    return grid.k_direction_is_down


def extract_geometry_time_index(grid):
    """Returns integer time index, or None, for the grid geometry, as stored in xml for dynamic geometries.

    notes:
       if the value is not None, it represents the time index as stored in the xml, or the time index as
       updated when setting the node points from a points property
    """
    if grid.time_index is not None and grid.time_series_uuid is not None:
        return grid.time_index
    grid.time_index = None
    grid.time_series_uuid = None
    ti_node = grid.resolve_geometry_child('TimeIndex')
    if ti_node is None:
        return None
    grid.time_index = rqet.find_tag_int(ti_node, 'Index')
    grid.time_series_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(ti_node, ['TimeSeries', 'UUID']))
    return grid.time_index


def extract_crs_uuid(grid):
    """Returns uuid for coordinate reference system, as stored in geometry xml tree.

    returns:
       uuid.UUID object
    """

    if grid.crs_uuid is not None:
        return grid.crs_uuid
    crs_root = grid.resolve_geometry_child('LocalCrs')
    uuid_str = rqet.find_tag_text(crs_root, 'UUID')
    if not uuid_str and hasattr(grid, 'extra_metadata'):
        uuid_str = grid.extra_metadata.get('crs uuid')
    if uuid_str:
        grid.crs_uuid = bu.uuid_from_string(uuid_str)
    return grid.crs_uuid


def extract_crs_root(grid):
    """Returns root in parent model xml parts forest of coordinate reference system used by this grid geomwtry.

    returns:
       root node in xml tree for coordinate reference system

    note:
       resqml allows a part to refer to another part that is not actually present in the same epc package;
       in practice, the crs is a tiny part and has always been included in datasets encountered so far;
       if the crs is not present, this method will return None (I think)
    """

    if grid.crs_root is not None:
        return grid.crs_root
    crs_uuid = grid.extract_crs_uuid()
    if crs_uuid is None:
        return None
    grid.crs_root = grid.model.root(uuid = crs_uuid)
    return grid.crs_root


def extract_pillar_shape(grid):
    """Returns string indicating whether whether pillars are curved, straight, or vertical as stored in xml.

    returns:
       string: either 'curved', 'straight' or 'vertical'

    note:
       resqml datasets often have 'curved', even when the pillars are actually 'vertical' or 'straight';
       use actual_pillar_shape() method to determine the shape from the actual xyz points data
    """

    if grid.pillar_shape is not None:
        return grid.pillar_shape
    ps_node = grid.resolve_geometry_child('PillarShape')
    if ps_node is None:
        return None
    grid.pillar_shape = ps_node.text
    return grid.pillar_shape


def extract_has_split_coordinate_lines(grid):
    """Returns boolean indicating whether grid geometry has any split coordinate lines (split pillars, ie. faults).

    returns:
       boolean: True if the grid has one or more split pillars; False if all pillars are unsplit

    notes:
       the return value is based on the array elements present in the xml tree, unless it has already been
       determined;
       resqml ijk grids with split coordinate lines have extra arrays compared to unfaulted grids, and the main
       Points array is indexed differently: [k', pillar_index, xyz] instead of [k', j', i', xyz] (where k', j', i'
       range of nk+k_gaps+1, nj+1, ni+1 respectively)
    """

    if grid.has_split_coordinate_lines is not None:
        return grid.has_split_coordinate_lines
    split_node = grid.resolve_geometry_child('SplitCoordinateLines')
    grid.has_split_coordinate_lines = (split_node is not None)
    if split_node is not None:
        grid.split_pillars_count = int(rqet.find_tag(split_node, 'Count').text.strip())
    return grid.has_split_coordinate_lines


def extract_k_gaps(grid):
    """Returns information about gaps (voids) between layers in the grid.

    returns:
       (int, numpy bool array, numpy int array) being the number of gaps between layers;
       a 1D bool array of extent nk-1 set True where there is a gap below the layer; and
       a 1D int array being the k index to actually use in the points data for each layer k0

    notes:
       all returned elements are stored as attributes in the grid object; int and bool array elements
       will be None if there are no k gaps; each k gap implies an extra element in the points data for
       each pillar; when wanting to index k interfaces (horizons) rather than layers, the last of the
       returned values can be used to index the k axis of the points data to yield the top face of the
       layer and the successor in k will always index the basal face of the same layer
    """

    if grid.k_gaps is not None:
        return grid.k_gaps, grid.k_gap_after_array, grid.k_raw_index_array
    grid.k_gaps = rqet.find_nested_tags_int(grid.root, ['KGaps', 'Count'])
    if grid.k_gaps:
        k_gap_after_root = rqet.find_nested_tags(grid.root, ['KGaps', 'GapAfterLayer'])
        assert k_gap_after_root is not None
        bool_array_type = rqet.node_type(k_gap_after_root)
        assert bool_array_type == 'BooleanHdf5Array'  # could be a constant array but not handled by this code
        h5_key_pair = grid.model.h5_uuid_and_path_for_node(k_gap_after_root)
        assert h5_key_pair is not None
        grid.model.h5_array_element(h5_key_pair,
                                    index = None,
                                    cache_array = True,
                                    object = grid,
                                    array_attribute = 'k_gap_after_array',
                                    dtype = 'bool')
        assert hasattr(grid, 'k_gap_after_array')
        assert grid.k_gap_after_array.ndim == 1 and grid.k_gap_after_array.size == grid.nk - 1
        grid._set_k_raw_index_array()
    else:
        grid.k_gap_after_array = None
        grid.k_raw_index_array = np.arange(grid.nk, dtype = int)
    return grid.k_gaps, grid.k_gap_after_array, grid.k_raw_index_array


def extract_stratigraphy(grid):
    """Loads stratigraphic information from xml."""

    grid.stratigraphic_column_rank_uuid = None
    grid.stratigraphic_units = None
    strata_node = rqet.find_tag(grid.root, 'IntervalStratigraphicUnits')
    if strata_node is None:
        return
    grid.stratigraphic_column_rank_uuid = \
        bu.uuid_from_string(rqet.find_nested_tags_text(strata_node, ['StratigraphicOrganization', 'UUID']))
    assert grid.stratigraphic_column_rank_uuid is not None
    unit_indices_node = rqet.find_tag(strata_node, 'UnitIndices')
    h5_key_pair = grid.model.h5_uuid_and_path_for_node(unit_indices_node)
    grid.model.h5_array_element(h5_key_pair,
                                index = None,
                                cache_array = True,
                                object = grid,
                                array_attribute = 'stratigraphic_units',
                                dtype = 'int')
    assert len(grid.stratigraphic_units) == grid.nk_plus_k_gaps


def extract_children(grid):
    """Looks for LGRs related to this grid and sets the local_grid_uuid_list attribute."""
    assert grid.uuid is not None
    if grid.local_grid_uuid_list is not None:
        return grid.local_grid_uuid_list
    grid.local_grid_uuid_list = []
    related_grid_roots = grid.model.roots(obj_type = 'IjkGridRepresentation', related_uuid = grid.uuid)
    if related_grid_roots is not None:
        for related_root in related_grid_roots:
            parent_uuid = rqet.find_nested_tags_text(related_root, ['ParentWindow', 'ParentGrid', 'UUID'])
            if parent_uuid is None:
                continue
            parent_uuid = bu.uuid_from_string(parent_uuid)
            if bu.matching_uuids(grid.uuid, parent_uuid):
                grid.local_grid_uuid_list.append(parent_uuid)
    return grid.local_grid_uuid_list


def extract_property_collection(grid):
    """Load grid property collection object holding lists of all properties in model that relate to this grid.

    returns:
       resqml_property.GridPropertyCollection object

    note:
       a reference to the grid property collection is cached in this grid object; if the properties change,
       for example by generating some new properties, the property_collection attribute of the grid object
       would need to be reset to None elsewhere before calling this method again
    """

    if grid.property_collection is not None:
        return grid.property_collection
    grid.property_collection = rprop.GridPropertyCollection(grid = grid)
    return grid.property_collection


def extract_inactive_mask(grid, check_pinchout = False):
    """Returns boolean numpy array indicating which cells are inactive, if (in)active property found in this grid.

    returns:
       numpy array of booleans, of shape (nk, nj, ni) being True for cells which are inactive; False for active

    note:
       RESQML does not have a built-in concept of inactive (dead) cells, though the usage guide advises to use a
       discrete property with a local property kind of 'active'; this resqpy code can maintain an 'inactive'
       attribute for the grid object, which is a boolean numpy array indicating which cells are inactive
    """

    if grid.inactive is not None and not check_pinchout:
        return grid.inactive
    geom_defined = grr_dg.cell_geometry_is_defined_ref(grid)
    if grid.inactive is None:
        if geom_defined is None or geom_defined is True:
            grid.inactive = np.zeros(tuple(grid.extent_kji))  # ie. all active
        else:
            grid.inactive = np.logical_not(grr_dg.cell_geometry_is_defined_ref(grid))
    if check_pinchout:
        grid.inactive = np.logical_or(grid.inactive, grid.pinched_out())
    gpc = grid.extract_property_collection()
    if gpc is None:
        grid.all_inactive = np.all(grid.inactive)
        return grid.inactive
    active_gpc = rprop.GridPropertyCollection()
    # note: use of bespoke (local) property kind 'active' as suggested in resqml usage guide
    active_gpc.inherit_parts_selectively_from_other_collection(other = gpc,
                                                               property_kind = 'active',
                                                               indexable = 'cells',
                                                               continuous = False)
    active_parts = active_gpc.parts()
    if len(active_parts) > 1:
        # try further filtering based on grid's time index data (or filtering out time based arrays)
        grid.extract_geometry_time_index()
        if grid.time_index is not None and grid.time_series_uuid is not None:
            active_gpc = rprop.selective_version_of_collection(active_gpc,
                                                               time_index = grid.time_index,
                                                               time_series_uuid = grid.time_series_uuid)
        else:
            active_parts = []
            for part in active_gpc.parts():
                if active_gpc.time_series_uuid_for_part(part) is None and active_gpc.time_index_for_part(part) is None:
                    active_parts.append(part)
    if len(active_parts) > 0:
        if len(active_parts) > 1:
            log.warning('more than one property found with bespoke kind "active", using last encountered')
        active_part = active_parts[-1]
        active_array = active_gpc.cached_part_array_ref(active_part, dtype = 'bool')
        grid.inactive = np.logical_or(grid.inactive, np.logical_not(active_array))
        grid.active_property_uuid = active_gpc.uuid_for_part(active_part)
        active_gpc.uncache_part_array(active_part)
    else:  # for backward compatibility with earlier versions of resqpy
        inactive_gpc = rprop.GridPropertyCollection()
        inactive_gpc.inherit_parts_selectively_from_other_collection(other = gpc,
                                                                     property_kind = 'code',
                                                                     facet_type = 'what',
                                                                     facet = 'inactive')
        if inactive_gpc.number_of_parts() == 1:
            inactive_part = inactive_gpc.parts()[0]
            inactive_array = inactive_gpc.cached_part_array_ref(inactive_part, dtype = 'bool')
            grid.inactive = np.logical_or(grid.inactive, inactive_array)
            inactive_gpc.uncache_part_array(inactive_part)

    grid.all_inactive = np.all(grid.inactive)
    return grid.inactive


def extent_kji_from_root(root_node):
    """Returns kji extent as stored in xml."""

    return (rqet.find_tag_int(root_node, 'Nk'), rqet.find_tag_int(root_node, 'Nj'), rqet.find_tag_int(root_node, 'Ni'))


def set_k_direction_from_points(grid):
    """Sets the K direction indicator based on z direction and mean z values for top and base.

    note:
       this method does not modify the grid_is_righthanded indicator
    """

    p = grid.points_ref(masked = False)
    grid.k_direction_is_down = True  # arbitrary default
    if p is not None:
        diff = np.nanmean(p[-1] - p[0])
        if not np.isnan(diff):
            grid.k_direction_is_down = ((diff >= 0.0) == grid.z_inc_down())
    return grid.k_direction_is_down
