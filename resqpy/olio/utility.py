"""Generic small utility functions."""

version = '29th April 2021'

import numpy as np
import numpy.typing as npt
from typing import Union, List, Tuple


def kji0_from_ijk1(index_ijk1: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:  # reverse order of indices and subtract 1
    """Returns index converted from simulator protocol to python protocol."""
    dims = index_ijk1.size
    result = np.zeros(dims, dtype = 'int')
    for d in range(dims):
        result[d] = index_ijk1[dims - d - 1] - 1
    return result


def ijk1_from_kji0(index_kji0: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:  # reverse order of indices and add 1
    """Returns index converted from python protocol to simulator protocol."""
    dims = index_kji0.size
    result = np.zeros(dims, dtype = 'int')
    for d in range(dims):
        result[d] = index_kji0[dims - d - 1] + 1
    return result


def extent_switch_ijk_kji(
        extent_in: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:  # reverse order of elements in extent
    """Returns equivalent grid extent switched either way between simulator and python protocols."""
    dims = extent_in.size
    result = np.zeros(dims, dtype = 'int')
    for d in range(dims):
        result[d] = extent_in[dims - d - 1]
    return result


def cell_count_from_extent(extent: Union[List[int], Tuple[int], npt.NDArray[np.int_]]) -> int:
    """Returns the number of cells in a grid with the given extent."""
    result = 1
    for val in extent:  # list, tuple or 1D numpy array
        result *= val
    return result


def string_ijk1_for_cell_kji0(cell_kji0: npt.NDArray[np.int_]) -> str:
    """Returns a string showing indices for a cell in simulator protocol, from data in python protocol."""
    return '[{:}, {:}, {:}]'.format(cell_kji0[2] + 1, cell_kji0[1] + 1, cell_kji0[0] + 1)


def string_ijk1_for_cell_ijk1(cell_ijk1: npt.NDArray[np.int_]) -> str:
    """Returns a string showing indices for a cell in simulator protocol, from data in simulator protocol."""
    return '[{:}, {:}, {:}]'.format(cell_ijk1[0], cell_ijk1[1], cell_ijk1[2])


def string_ijk_for_extent_kji(extent_kji: npt.NDArray[np.int_]) -> str:
    """Returns a string showing grid extent in simulator protocol, from data in python protocol."""
    return '[{:}, {:}, {:}]'.format(extent_kji[2], extent_kji[1], extent_kji[0])


def string_xyz(xyz: npt.NDArray[np.int_]) -> str:
    """Returns an xyz point as a string like (121234.56, 567890.12, 3456.789)"""

    return '({0:4.2f}, {1:4.2f}, {2:5.3f})'.format(xyz[0], xyz[1], xyz[2])


def horizon_float(k0: int, plus_or_minus: str) -> float:
    """Returns a floating point representation of k direction face index; eg.

    3.5 for face between layers 3 and 4.
    """
    result = float(k0)
    if plus_or_minus == '+':
        result += 0.5
    elif plus_or_minus == '-':
        result -= 0.5
    else:
        assert False
    return result
