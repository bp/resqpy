"""Multiprocessing module for running specific functions concurrently."""

__all__ = ['function_multiprocessing', 'find_faces_to_represent_surface_regular_wrapper']

from .multiprocessing import function_multiprocessing
from .wrappers.grid_surface import find_faces_to_represent_surface_regular_wrapper

# Set "module" attribute of all public objects to this path.
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
