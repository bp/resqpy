"""Unstructured grid and derived classes."""

__all__ = [
    'UnstructuredGrid', 'HexaGrid', 'PrismGrid', 'VerticalPrismGrid', 'PyramidGrid', 'TetraGrid', 'valid_cell_shapes'
]

from ._unstructured_grid import UnstructuredGrid, valid_cell_shapes
from ._hexa_grid import HexaGrid
from ._prism_grid import PrismGrid, VerticalPrismGrid
from ._pyramid_grid import PyramidGrid
from ._tetra_grid import TetraGrid

# Set "module" attribute of all public objects to this path. Fixes issue #310
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
