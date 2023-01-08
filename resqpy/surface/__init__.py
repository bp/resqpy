"""Classes for RESQML objects related to surfaces."""

__all__ = [
    'BaseSurface', 'CombinedSurface', 'Mesh', 'TriangulatedPatch', 'PointSet', 'Surface', 'TriMesh',
    'distill_triangle_points'
]

from ._base_surface import BaseSurface
from ._combined_surface import CombinedSurface
from ._mesh import Mesh
from ._triangulated_patch import TriangulatedPatch
from ._pointset import PointSet
from ._surface import Surface, distill_triangle_points
from ._tri_mesh import TriMesh

# Set "module" attribute of all public objects to this path.
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
