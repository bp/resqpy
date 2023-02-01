"""Multiprocessing module for running specific functions concurrently."""

__all__ = [
    'function_multiprocessing', 'find_faces_to_represent_surface_regular_wrapper',
    'mesh_from_regular_grid_column_property_wrapper', 'mesh_from_regular_grid_column_property_batch',
    'blocked_well_from_trajectory_wrapper', 'blocked_well_from_trajectory_batch'
]

from ._multiprocessing import function_multiprocessing
from .wrappers.grid_surface_mp import find_faces_to_represent_surface_regular_wrapper
from .wrappers.mesh_mp import mesh_from_regular_grid_column_property_wrapper, mesh_from_regular_grid_column_property_batch
from .wrappers.blocked_well_mp import blocked_well_from_trajectory_wrapper, blocked_well_from_trajectory_batch

# Set "module" attribute of all public objects to this path.
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
