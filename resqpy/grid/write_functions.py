"""Submodule containing the functions relating to writing grid information."""

import logging

import numpy as np

log = logging.getLogger(__name__)

import resqpy.olio.grid_functions as gf
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.property as rprop
import resqpy.olio.trademark as tm
import resqpy.olio.write_data as wd

always_write_pillar_geometry_is_defined_array = False
always_write_cell_geometry_is_defined_array = False


def write_hdf5_from_caches(grid,
                           file = None,
                           mode = 'a',
                           geometry = True,
                           imported_properties = None,
                           write_active = None,
                           stratigraphy = True,
                           expand_const_arrays = False):
    """Create or append to an hdf5 file.

    Writes datasets for the grid geometry (and parent grid mapping) and properties from cached arrays.
    """
    # NB: when writing a new geometry, all arrays must be set up and exist as the appropriate attributes prior to calling this function
    # if saving properties, active cell array should be added to imported_properties based on logical negation of inactive attribute
    # xml is not created here for property objects

    if write_active is None:
        write_active = geometry

    grid.cache_all_geometry_arrays()

    if not file:
        file = grid.model.h5_file_name()
    h5_reg = rwh5.H5Register(grid.model)

    if stratigraphy and grid.stratigraphic_units is not None:
        h5_reg.register_dataset(grid.uuid, 'unitIndices', grid.stratigraphic_units, dtype = 'uint32')

    if geometry:
        __write_geometry(grid, h5_reg)

    if write_active and grid.inactive is not None:
        if imported_properties is None:
            imported_properties = rprop.PropertyCollection()
            imported_properties.set_support(support = grid)
        else:
            filtered_list = []
            for entry in imported_properties.imported_list:
                if entry[2].upper() == 'ACTIVE' or entry[10] == 'active':
                    continue  # keyword or property kind
                filtered_list.append(entry)
            imported_properties.imported_list = filtered_list  # might have unintended side effects elsewhere
        active_mask = np.logical_not(grid.inactive)
        imported_properties.add_cached_array_to_imported_list(active_mask,
                                                              'active cell mask',
                                                              'ACTIVE',
                                                              discrete = True,
                                                              property_kind = 'active')

    if imported_properties is not None and imported_properties.imported_list is not None:
        for entry in imported_properties.imported_list:
            tail = 'points_patch0' if entry[18] else 'values_patch0'
            # expand constant arrays if required
            if not hasattr(imported_properties, entry[3]) and expand_const_arrays and entry[17] is not None:
                value = float(entry[17]) if isinstance(entry[17], str) else entry[17]
                assert entry[14] == 'cells' and entry[15] == 1
                imported_properties.__dict__[entry[3]] = np.full(grid.extent_kji, value)
            if hasattr(imported_properties, entry[3]):  # otherwise constant array not being expanded
                h5_reg.register_dataset(entry[0], tail, imported_properties.__dict__[entry[3]])
            if entry[10] == 'active':
                grid.active_property_uuid = entry[0]
    h5_reg.write(file, mode = mode)


def __write_geometry(grid, h5_reg):
    if always_write_pillar_geometry_is_defined_array or not grid.geometry_defined_for_all_pillars(cache_array = True):
        if not hasattr(grid, 'array_pillar_geometry_is_defined') or grid.array_pillar_geometry_is_defined is None:
            grid.array_pillar_geometry_is_defined = np.full((grid.nj + 1, grid.ni + 1), True, dtype = bool)
        h5_reg.register_dataset(grid.uuid,
                                'PillarGeometryIsDefined',
                                grid.array_pillar_geometry_is_defined,
                                dtype = 'uint8')
    if always_write_cell_geometry_is_defined_array or not grid.geometry_defined_for_all_cells(cache_array = True):
        if not hasattr(grid, 'array_cell_geometry_is_defined') or grid.array_cell_geometry_is_defined is None:
            grid.array_cell_geometry_is_defined = np.full((grid.nk, grid.nj, grid.ni), True, dtype = bool)
        h5_reg.register_dataset(grid.uuid,
                                'CellGeometryIsDefined',
                                grid.array_cell_geometry_is_defined,
                                dtype = 'uint8')
    # todo: PillarGeometryIsDefined ?
    h5_reg.register_dataset(grid.uuid, 'Points', grid.points_cached)
    if grid.has_split_coordinate_lines:
        h5_reg.register_dataset(grid.uuid, 'PillarIndices', grid.split_pillar_indices_cached, dtype = 'uint32')
        h5_reg.register_dataset(grid.uuid,
                                'ColumnsPerSplitCoordinateLine/elements',
                                grid.cols_for_split_pillars,
                                dtype = 'uint32')
        h5_reg.register_dataset(grid.uuid,
                                'ColumnsPerSplitCoordinateLine/cumulativeLength',
                                grid.cols_for_split_pillars_cl,
                                dtype = 'uint32')
    if grid.k_gaps:
        assert grid.k_gap_after_array is not None
        h5_reg.register_dataset(grid.uuid, 'GapAfterLayer', grid.k_gap_after_array, dtype = 'uint8')
    if grid.parent_window is not None:
        for axis in range(3):
            if grid.parent_window.fine_extent_kji[axis] == grid.parent_window.coarse_extent_kji[axis]:
                continue  # one-to-noe mapping
            # reconstruct hdf5 arrays from FineCoarse object and register for write
            if grid.parent_window.constant_ratios[axis] is not None:
                if grid.is_refinement:
                    pcpi = np.array([grid.parent_window.coarse_extent_kji[axis]], dtype = int)  # ParentCountPerInterval
                    ccpi = np.array([grid.parent_window.fine_extent_kji[axis]], dtype = int)  # ChildCountPerInterval
                else:
                    pcpi = np.array([grid.parent_window.fine_extent_kji[axis]], dtype = int)
                    ccpi = np.array([grid.parent_window.coarse_extent_kji[axis]], dtype = int)
            else:
                if grid.is_refinement:
                    interval_count = grid.parent_window.coarse_extent_kji[axis]
                    pcpi = np.ones(interval_count, dtype = int)
                    ccpi = np.array(grid.parent_window.vector_ratios[axis], dtype = int)
                else:
                    interval_count = grid.parent_window.fine_extent_kji[axis]
                    pcpi = np.array(grid.parent_window.vector_ratios[axis], dtype = int)
                    ccpi = np.ones(interval_count, dtype = int)
            h5_reg.register_dataset(grid.uuid, 'KJI'[axis] + 'Regrid/ParentCountPerInterval', pcpi)
            h5_reg.register_dataset(grid.uuid, 'KJI'[axis] + 'Regrid/ChildCountPerInterval', ccpi)
            if grid.is_refinement and not grid.parent_window.equal_proportions[axis]:
                child_cell_weights = np.concatenate(grid.parent_window.vector_proportions[axis])
                h5_reg.register_dataset(grid.uuid, 'KJI'[axis] + 'Regrid/ChildCellWeights', child_cell_weights)


def write_nexus_corp(self,
                     file_name,
                     local_coords = False,
                     global_xy_units = None,
                     global_z_units = None,
                     global_z_increasing_downward = True,
                     write_nx_ny_nz = False,
                     write_units_keyword = False,
                     write_rh_keyword_if_needed = False,
                     write_corp_keyword = False,
                     use_binary = False,
                     binary_only = False,
                     nan_substitute_value = None):
    """Write grid geometry to file in Nexus CORP ordering."""

    log.info('caching Nexus corner points')
    tm.log_nexus_tm('info')
    self.corner_points(cache_cp_array = True)
    log.debug('duplicating Nexus corner points')
    cp = self.array_corner_points.copy()
    log.debug('resequencing duplicated Nexus corner points')
    gf.resequence_nexus_corp(cp, eight_mode = False, undo = True)
    corp_extent = np.zeros(3, dtype = 'int')
    corp_extent[0] = self.cell_count()  # total number of cells in grid
    corp_extent[1] = 8  # 8 corners of cell: k -/+; j -/+; i -/+
    corp_extent[2] = 3  # x, y, z
    ijk_right_handed = self.extract_grid_is_right_handed()
    if ijk_right_handed is None:
        log.warning('ijk handedness not known')
    elif not ijk_right_handed:
        log.warning('ijk axes are left handed; inverted (fake) xyz handedness required')
    crs_root = self.extract_crs_root()
    if not local_coords:
        if not global_z_increasing_downward:
            log.warning('global z is not increasing with depth as expected by Nexus')
            tm.log_nexus_tm('warning')
        if crs_root is not None:  # todo: otherwise raise exception?
            log.info('converting corner points from local to global reference system')
            self.local_to_global_crs(cp,
                                     crs_root,
                                     global_xy_units = global_xy_units,
                                     global_z_units = global_z_units,
                                     global_z_increasing_downward = global_z_increasing_downward)
    log.info('writing simulator corner point file ' + file_name)
    with open(file_name, 'w') as header:
        header.write('! Nexus corner point data written by resqml_grid module\n')
        header.write('! Nexus is a registered trademark of the Halliburton Company\n\n')
        if write_units_keyword:
            if local_coords:
                if crs_root is not None:
                    crs_xy_units_text = rqet.find_tag(crs_root, 'ProjectedUom').text
                    crs_z_units_text = rqet.find_tag(crs_root, 'VerticalUom').text
                    if crs_xy_units_text == 'm' and crs_z_units_text == 'm':
                        header.write('METRIC\n\n')
                    elif crs_xy_units_text == 'ft' and crs_z_units_text == 'ft':
                        header.write('ENGLISH\n\n')
                    else:
                        header.write('! local coordinates mixed (or not recognized)\n\n')
                else:
                    header.write('! local coordinates unknown\n\n')
            elif global_xy_units is not None and global_z_units is not None and global_xy_units == global_z_units:
                if global_xy_units in ['m', 'metre', 'metres']:
                    header.write('METRIC\n\n')
                elif global_xy_units in ['ft', 'feet', 'foot']:
                    header.write('ENGLISH\n\n')
                else:
                    header.write('! globsl coordinates not recognized\n\n')
            else:
                header.write('! global units unknown or mixed\n\n')
        if write_nx_ny_nz:
            header.write('NX      NY      NZ\n')
            header.write('{0:<7d} {1:<7d} {2:<7d}\n\n'.format(self.extent_kji[2], self.extent_kji[1],
                                                              self.extent_kji[0]))
        if write_rh_keyword_if_needed:
            if ijk_right_handed is None or crs_root is None:
                log.warning('unable to determine whether RIGHTHANDED keyword is needed')
            else:
                xy_axes = rqet.find_tag(crs_root, 'ProjectedAxisOrder').text
                if local_coords:
                    z_inc_down = self.z_inc_down()
                    if not z_inc_down:
                        log.warning('local z is not increasing with depth as expected by Nexus')
                        tm.log_nexus_tm('warning')
                else:
                    z_inc_down = global_z_increasing_downward
                xyz_handedness = rqet.xyz_handedness(xy_axes, z_inc_down)
                if xyz_handedness == 'unknown':
                    log.warning(
                        'xyz handedness is not known; unable to determine whether RIGHTHANDED keyword is needed')
                else:
                    if ijk_right_handed == (xyz_handedness == 'right'):  # if either both True or both False
                        header.write('RIGHTHANDED\n\n')
    if write_corp_keyword:
        keyword = 'CORP VALUE'
    else:
        keyword = None
    wd.write_array_to_ascii_file(file_name,
                                 corp_extent,
                                 cp.reshape(tuple(corp_extent)),
                                 target_simulator = 'nexus',
                                 keyword = keyword,
                                 columns = 3,
                                 blank_line_after_i_block = False,
                                 blank_line_after_j_block = True,
                                 append = True,
                                 use_binary = use_binary,
                                 binary_only = binary_only,
                                 nan_substitute_value = nan_substitute_value)
