"""Copy grid function."""

import numpy as np

import resqpy.grid as grr
import resqpy.olio.xml_et as rqet


def copy_grid(source_grid, target_model = None, copy_crs = True):
    """Creates a copy of the IJK grid object in the target model (usually prior to modifying points in situ).

    note:
       this function is not usually called directly by application code; it does not write to the hdf5
       file nor create xml for the copied grid;
       the copy will be a resqpy Grid even if the source grid is a RegularGrid
    """

    assert source_grid.grid_representation in ['IjkGrid', 'IjkBlockGrid']

    model = source_grid.model
    if target_model is None:
        target_model = model
    if target_model is model:
        copy_crs = False

    # if the source grid is a RegularGrid, ensure that it has explicit points
    if source_grid.grid_representation == 'IjkBlockGrid':
        source_grid.make_regular_points_cached()

    # create empty grid object (with new uuid)
    grid = grr.Grid(target_model)

    # inherit attributes from source grid
    grid.grid_representation = 'IjkGrid'
    grid.extent_kji = np.array(source_grid.extent_kji, dtype = 'int')
    grid.nk, grid.nj, grid.ni = source_grid.nk, source_grid.nj, source_grid.ni
    grid.k_direction_is_down = source_grid.k_direction_is_down
    grid.grid_is_right_handed = source_grid.grid_is_right_handed
    grid.pillar_shape = source_grid.pillar_shape
    grid.has_split_coordinate_lines = source_grid.has_split_coordinate_lines
    grid.k_gaps = source_grid.k_gaps
    if grid.k_gaps:
        grid.k_gap_after_array = source_grid.k_gap_after_array.copy()
        grid.k_raw_index_array = source_grid.k_raw_index_array.copy()

    # inherit a copy of the coordinate reference system used by the grid geometry
    if copy_crs:
        grid.crs_root = model.duplicate_node(source_grid.crs_root)
    else:
        grid.crs_root = source_grid.crs_root  # pointer to a source model xml tree
    grid.crs_uuid = rqet.uuid_for_part_root(grid.crs_root)

    # inherit a copy of the inactive cell mask
    if source_grid.inactive is None:
        grid.inactive = None
    else:
        grid.inactive = source_grid.inactive.copy()
    grid.active_property_uuid = source_grid.active_property_uuid

    # take a copy of the grid geometry
    source_grid.cache_all_geometry_arrays()
    grid.geometry_defined_for_all_pillars_cached = source_grid.geometry_defined_for_all_pillars_cached
    if hasattr(source_grid, 'array_pillar_geometry_is_defined'):
        grid.array_pillar_geometry_is_defined = source_grid.array_pillar_geometry_is_defined.copy()
    if hasattr(source_grid, 'array_cell_geometry_is_defined'):
        grid.array_cell_geometry_is_defined = source_grid.array_cell_geometry_is_defined.copy()
    grid.geometry_defined_for_all_cells_cached = source_grid.geometry_defined_for_all_cells_cached
    grid.points_cached = source_grid.points_cached.copy()
    if grid.has_split_coordinate_lines:
        source_grid.create_column_pillar_mapping()
        grid.split_pillar_indices_cached = source_grid.split_pillar_indices_cached.copy()
        grid.cols_for_split_pillars = source_grid.cols_for_split_pillars.copy()
        grid.cols_for_split_pillars_cl = source_grid.cols_for_split_pillars_cl.copy()
        grid.split_pillars_count = source_grid.split_pillars_count
        grid.pillars_for_column = source_grid.pillars_for_column.copy()

    return grid
