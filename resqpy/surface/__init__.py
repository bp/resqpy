"""Classes for RESQML objects related to surfaces."""

__all__ = ['BaseSurface', 'CombinedSurface', 'Mesh', 'TriangulatedPatch', 'PointSet', 'Surface']

from .base_surface import BaseSurface
from .combined_surface import CombinedSurface
from .mesh import Mesh
from .triangulated_patch import TriangulatedPatch
from .pointset import PointSet
from .surface import Surface

# Set "module" attribute of all public objects to this path.
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
