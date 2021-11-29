"""Creating a derived resqml model from an existing one; mostly grid manipulations."""

__all__ = [
    'add_edges_per_column_property_array', 'add_faults', 'add_one_blocked_well_property', 'add_one_grid_property_array',
    'add_single_cell_grid', 'add_wells_from_ascii_file', 'add_zone_by_layer_property', 'coarsened_grid', 'copy_grid',
    'drape_to_surface', 'extract_box_for_well', 'extract_box', 'fault_throw_scaling', 'global_fault_throw_scaling',
    'gather_ensemble', 'interpolated_grid', 'local_depth_adjustment', 'refined_grid', 'tilted_grid', 'unsplit_grid',
    'zonal_grid', 'single_layer_grid', 'zone_layer_ranges_from_array'
]

from ._add_edges_per_column_property_array import add_edges_per_column_property_array
from ._add_faults import add_faults
from ._add_one_blocked_well_property import add_one_blocked_well_property
from ._add_one_grid_property_array import add_one_grid_property_array
from ._add_single_cell_grid import add_single_cell_grid
from ._add_wells_from_ascii_file import add_wells_from_ascii_file
from ._add_zone_by_layer_property import add_zone_by_layer_property
from ._coarsened_grid import coarsened_grid
from ._copy_grid import copy_grid
from ._drape_to_surface import drape_to_surface
from ._extract_box_for_well import extract_box_for_well
from ._extract_box import extract_box
from ._fault_throw_scaling import fault_throw_scaling, global_fault_throw_scaling
from ._gather_ensemble import gather_ensemble
from ._interpolated_grid import interpolated_grid
from ._local_depth_adjustment import local_depth_adjustment
from ._refined_grid import refined_grid
from ._tilted_grid import tilted_grid
from ._unsplit_grid import unsplit_grid
from ._zonal_grid import zonal_grid, single_layer_grid
from ._zone_layer_ranges_from_array import zone_layer_ranges_from_array

# Set "module" attribute of all public objects to this path.
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
