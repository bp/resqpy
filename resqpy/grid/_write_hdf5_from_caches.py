"""Submodule containing the functions relating to writing grid information."""

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.olio.write_hdf5 as rwh5
import resqpy.property as rqp

always_write_pillar_geometry_is_defined_array = False
always_write_cell_geometry_is_defined_array = False


def _write_hdf5_from_caches(grid,
                            file = None,
                            mode = 'a',
                            geometry = True,
                            imported_properties = None,
                            write_active = None,
                            stratigraphy = True,
                            expand_const_arrays = False,
                            use_int32 = None):
    """Create or append to an hdf5 file.

    Writes datasets for the grid geometry (and parent grid mapping) and properties from cached arrays.
    """
    # NB: when writing a new geometry, all arrays must be set up and exist as the appropriate attributes prior to calling this function
    # if saving properties, active cell array should be added to imported_properties based on logical negation of inactive attribute
    # xml is not created here for property objects

    if write_active is None:
        existing_active = rqp.property_parts(grid.model,
                                             obj_type = 'DiscreteProperty',
                                             property_kind = 'active',
                                             related_uuid = grid.uuid)
        write_active = geometry and not existing_active

    if geometry:
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
            imported_properties = rqp.PropertyCollection()
            imported_properties.set_support(support = grid)
        else:
            filtered_list = []
            for entry in imported_properties.imported_list:
                if (entry[2].upper() == 'ACTIVE' or entry[10] == 'active') and entry[14] == 'cells':
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
    h5_reg.write(file, mode = mode, use_int32 = use_int32)


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
