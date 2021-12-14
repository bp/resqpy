"""Model class, roughly equivalent to a RESQML epc file."""

__all__ = ['Model', 'ModelContext', 'new_model']

from ._model import Model, new_model
from ._context import ModelContext

# Set "module" attribute of all public objects to this path. Fixes issue #310
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
