"""Thin wrapper around numba.njit that optionally enables numba caching.

Cache behaviour can be controlled via enviornment varibales:

    RESQPY_NUMBA_CACHE :
        Set to "true" or "TRUE" to enable or on-disk caching for resqpy functions
        which are decorated by njit. If unset, caching is disabled.

For more info, see <https://numba.readthedocs.io/en/stable/developer/caching.html>.
"""
import os

from numba import njit as _numba_njit

# Accept "TRUE" or "true"
ENABLE_NUMBA_CACHE = os.environ.get("RESQPY_NUMBA_CACHE", "false").lower().startswith("t")


def njit(*args, **kwargs):
    """Drop-in replacement for numba.njit that sets "cache" kwarg according to a global config.

    If "cache" is defined in the decorator, that takes priority.
    
    Supports both decorator forms, with or without brackets:
    - `@njit`
    - `@njit(parallel = True)`
    """
    kwargs.setdefault('cache', ENABLE_NUMBA_CACHE)
    if args and callable(args[0]):
        # bare decorator: @njit applied directly to a function
        func, *rest = args
        return _numba_njit(func, *rest, **kwargs)
    # parametrised decorator: @njit(...) returning a decorator
    return _numba_njit(*args, **kwargs)
