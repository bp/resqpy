"""Miscellaneous functions for importing from other formats."""

# Nexus is a trademark of Halliburton

__all__ = [
    'import_nexus', 'import_vdb_all_grids', 'grid_from_cp', 'import_vdb_ensemble', 'add_ab_properties', 'add_surfaces'
]

from ._grid_from_cp import grid_from_cp
from ._import_nexus import import_nexus
from ._import_vdb_all_grids import import_vdb_all_grids
from ._import_vdb_ensemble import import_vdb_ensemble
from ._add_ab_properties import add_ab_properties
from ._add_surfaces import add_surfaces

# Set "module" attribute of all public objects to this path. Fixes issue #398
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
