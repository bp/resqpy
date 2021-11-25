"""Package containing Polyline and PolylineSet classes."""

__all__ = ['Polyline', 'PolylineSet', 'load_hdf5_array', 'shift_polyline', 'flatten_polyline', 'tangents', 'spline']

from ._common import load_hdf5_array, shift_polyline, flatten_polyline, tangents, spline
from ._polyline import Polyline
from ._polyline_set import PolylineSet
