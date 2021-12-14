"""Polyline and PolylineSet classes and associated functions."""

__all__ = ['Polyline', 'PolylineSet', 'load_hdf5_array', 'shift_polyline', 'flatten_polyline', 'tangents', 'spline']

from ._common import load_hdf5_array, shift_polyline, flatten_polyline, tangents, spline
from ._polyline import Polyline
from ._polyline_set import PolylineSet

# Set "module" attribute of all public objects to this path. Fixes issue #310
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
