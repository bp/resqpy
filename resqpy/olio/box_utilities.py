"""Simple functions relating to cartesian grid boxes.

A box is a logical cuboid subset of the cells of a cartesian grid.
A box is defined by a small numpy array: [[min_k, min_j, min_i], [max_k, max_j, max_i]].
The cells identified by the max indices are included in the box (not following the python convention)
The ordering of the i,j & k indices might be reversed - identifier names then have a suffix of _ijk instead of _kji.
The indices can be in simulator convention, starting at 1, or python convention, starting at 0, indicated by suffix of 0 or 1
"""

import logging

log = logging.getLogger(__name__)

import numpy as np


def extent_of_box(box):  # returns 3 element extent of box (box can be kji or ijk, 0 or 1 based)
    """Returns a 3 integer numpy array holding the size of the box, with the same ordering as the box.

    input argument (unmodified):
       box: numpy int array of shape (2, 3)
          lower & upper indices in 3 dimensions defining a logical cuboid subset of a 3D cartesian grid

    returns: numpy int array of shape (3)
          the extent (shape) of the cuboid defined by box
    """

    assert box.ndim == 2 and box.shape == (2, 3)
    return box[1] - box[0] + 1  # numpy array operation


def volume_of_box(box):
    """Returns the number of cells in the logical 3D cell space defined by box.

    input argument (unmodified):
       box: numpy int array of shape (2, 3)
          lower & upper indices in 3 dimensions defining a logical cuboid subset of a 3D cartesian grid

    returns: int
          the total number of cells in box
    """

    return (box[1, 0] - box[0, 0] + 1) * (box[1, 1] - box[0, 1] + 1) * (box[1, 2] - box[0, 2] + 1)


def central_cell(box):
    """Returns the indices of the cell at the centre of the box."""
    return box[0] + ((box[1] - box[0]) // 2)


def string_iijjkk1_for_box_kji0(box_kji0):
    """Returns a string representing the box space in simulator protocol, eg. '[1:5, 3:20, 100:103]'.

    input argument (unmodified):
       box_kji0: numpy int array of shape (2, 3)
          lower & upper indices in 3 dimensions defining a logical cuboid subset of a 3D cartesian grid
          with python kji ordering and zero start for indices

    returns: string
       human readable representation of box in Fortran/simulator ijk protocol starting 1
    """

    return '[' + str(box_kji0[0, 2] + 1) + ':' + str(box_kji0[1, 2] + 1) + ', ' +  \
                 str(box_kji0[0, 1] + 1) + ':' + str(box_kji0[1, 1] + 1) + ', ' +  \
                 str(box_kji0[0, 0] + 1) + ':' + str(box_kji0[1, 0] + 1) + ']'


def spaced_string_iijjkk1_for_box_kji0(box_kji0, colon_separator = ' '):
    """Returns a string representing the box space in simulator input format, eg. '1 5  3 20  100 103'.

    input arguments (unmodified):
       box_kji0: numpy int array of shape (2, 3)
          lower & upper indices in 3 dimensions defining a logical cuboid subset of a 3D cartesian grid
          with python kji ordering and zero start for indices
       colon_separator: string (typically ':' or ' ')
          the character(s) included in the return string between lower and upper bounds in each direction

    returns: string
       ascii representation of box in Fortran/simulator ijk protocol starting 1, suitable for use in include files
    """

    return str(box_kji0[0, 2] + 1) + colon_separator + str(box_kji0[1, 2] + 1) + '  ' +  \
           str(box_kji0[0, 1] + 1) + colon_separator + str(box_kji0[1, 1] + 1) + '  ' +  \
           str(box_kji0[0, 0] + 1) + colon_separator + str(box_kji0[1, 0] + 1)


def box_kji0_from_words_iijjkk1(words):
    """Returns an integer array of extent [2, 3] converted from a list of words representing logical box.

    input argument (unmodified):
       words: a list of strings with at least 6 elements castable to int
          [min_i, max_i, min_j, max_j, min_k, max_k] in Fortran/simulator protocol (indices start at 1)

    returns: 2D numpy int array of shape (2, 3)
       [min, max][k, j, i] with cell indices in python protocol (zero base)

    notes:
       designed to take string format numbers: minI maxI minJ maxJ minK maxK
       and convert to a pair of integer cell id triplets: min(k, j, i), max(k, j, i)
       NB: output indices have been decremented by 1 (for python indexing starting at zero)
    """

    assert len(words) >= 6  # expecting minI maxI minJ maxJ minK maxK value
    box = np.zeros([2, 3], dtype = 'int')
    box[0, 0] = int(words[4]) - 1  # k min
    box[1, 0] = int(words[5]) - 1  # k max
    box[0, 1] = int(words[2]) - 1  # j min
    box[1, 1] = int(words[3]) - 1  # j max
    box[0, 2] = int(words[0]) - 1  # i min
    box[1, 2] = int(words[1]) - 1  # i max
    assert box[0, 0] <= box[1, 0] and box[0, 1] <= box[1, 1] and box[0, 2] <= box[1, 2]

    return box


def cell_in_box(cell, box):
    """Returns True if cell is within box, otherwise False.

    input arguments (unmodified):
       cell: numpy int array of shape (3)
          index of a cell in a 3D cartesian grid, in the same protocol as box (usually python protocol kji, zero base)
       box: numpy int array of shape (2, 3)
          lower & upper indices in 3 dimensions defining a logical cuboid subset of a 3D cartesian grid
          in the same protocol as cell

    returns: boolean
       True if cell is within box, False otherwise
    """

    return (box[0, 0] <= cell[0] <= box[1, 0]) and (box[0, 1] <= cell[1] <= box[1, 1]) and (box[0, 2] <= cell[2] <=
                                                                                            box[1, 2])


def valid_box(box, host_extent):
    """Returns True if the entire box is within a grid of size host_extent.

    input arguments (unmodified):
       box: numpy int array of shape (2, 3)
          lower & upper indices in 3 dimensions defining a logical cuboid subset of a 3D cartesian grid
          in python protocol of zero base, kji (normally) or ijk ordering same as for host_extent
       host_extent: triple int
          the extent (shape) of a 3D cartesian grid

    returns: boolean
       True if box is a valid box within a grid of shape host_extent, False otherwise
    """

    if box.ndim != 2 or box.shape != (2, 3) or box.dtype != 'int':
        return False
    if len(host_extent) != 3:
        return False
    for d in range(3):
        if box[0, d] < 0 or box[0, d] > box[1, d] or box[1, d] >= host_extent[d]:
            return False
    return True


def single_cell_box(cell):
    """Returns a box containing the single given cell; protocol for box matches that of cell.

    input argument (unmodified):
       cell: numpy int array of shape (3)
          indices of a cell within a 3D cartesian grid, usually in python protocol (kji ordering, zero base)

    returns: numpy int array of shape (2, 3)
       indices defining a minimal box containing a single cell; protocol is same as that of cell
    """

    assert cell.ndim == 1 and cell.size == 3
    box = np.zeros((2, 3), dtype = 'int')
    box[0] = cell  # numpy 1D array op
    box[1] = cell  # numpy 1D array op
    return box


def full_extent_box0(extent):
    """Returns a box containing all the cells in a grid of the given extent.

    input argument (unmodified):
       extent: numpy int array of shape (3)
          extent (shape) of a 3D cartesian grid, usually in kji python protocol

    returns: numpy int array of shape (2, 3)
       indices defining a maximal box containing the entire grid; kji ordering is same as that of extent; zero base
    """

    assert extent.ndim == 1 and extent.size == 3
    box = np.zeros((2, 3), dtype = 'int')
    box[1, :] = extent - 1  # numpy 1D array op
    return box


def union(box_1, box_2):
    """Returns the box which contains both box_1 and box_2."""

    if box_1 is None and box_2 is None:
        return None
    if box_1 is None:
        return box_2.copy()
    if box_2 is None:
        return box_1.copy()
    result = np.zeros((2, 3), dtype = 'int')
    for dir in range(3):
        result[0][dir] = min(box_1[0][dir], box_2[0][dir])
        result[1][dir] = max(box_1[1][dir], box_2[1][dir])
    return result


def parent_cell_from_local_box_cell(box, box_cell, based_0_or_1 = 0):
    """Given a box and a local cell index triplet, converts to the equivalent cell index triplet in the host grid.

    input arguments (unmodified):
       box: numpy int array of shape (2, 3)
          lower & upper indices in 3 dimensions defining a logical cuboid subset of a 3D cartesian grid
          indices in the same ordering as box_cell; start value (python or Fortran/simulator) given by based_0_or_1
       box_cell: numpy int array of shape (3)
          indices of a cell within box, in coords local to box
          indices in the same ordering as box; start value (python or Fortran/simulator) given by based_0_or_1
       based_0_or_1: int, value 0 or 1
          start value (base) for indices of box and box_cell arguments, and of return value

    returns: numpy int array of shape (3)
       indices defining the cell in the host grid space equivalent to box_cell
       ordering of indices is same as that of box and box_cell; base is given by based_0_or_1 argument
    """

    assert box.ndim == 2 and box.shape == (2, 3)
    assert box_cell.ndim == 1 and box_cell.size == 3
    return box[0] + box_cell - based_0_or_1  # numpy 1D array op


def local_box_cell_from_parent_cell(box, parent_cell, based_0_or_1 = 0):
    """Given a cell index triplet in the host grid, and a box, returns the equivalent local cell index triplet.

    input arguments (unmodified):
       box: numpy int array of shape (2, 3)
          lower & upper indices in 3 dimensions defining a logical cuboid subset of a 3D cartesian grid
          indices in the same ordering as parent_cell; start value (python or Fortran/simulator) given by based_0_or_1
       parent_cell: numpy int array of shape (3)
          indices of a cell within host grid
          indices in the same ordering as box; start value (python or Fortran/simulator) given by based_0_or_1
       based_0_or_1: int, value 0 or 1
          start value (base) for indices of box and parent_cell arguments, and of return value

    returns: numpy int array of shape (3); or None
       indices defining the parent_cell in coords local to box, if the cell is within the box
       if parent_cell is not within box, None is returned
    """

    assert box.ndim == 2 and box.shape == (2, 3)
    assert parent_cell.ndim == 1 and parent_cell.size == 3
    if np.all(box[0] <= parent_cell) and np.all(parent_cell <= box[1]):  # numpy 1D array ops
        return parent_cell - box[0] + based_0_or_1  # numpy 1D array op
    else:
        return None


def boxes_overlap(box_a, box_b):
    """Returns True if the two boxes have any overlap in 3D, otherwise False.

    Arguments:
       box_a: numpy int or float array of shape (2, 3)
       box_b: numpy int or float array of shape (2, 3)

    if int arrays, each is lower & upper indices in 3 dimensions defining a logical cuboid
    subset of a 3D cartesian grid protocol of indices for the two boxes must be the same
    if float arrays, each is min & max x,y,z triplets

    returns: boolean
       True if box_a and box_b overlap, False otherwise
    """

    return not ((box_a[1, 0] < box_b[0, 0]) or (box_a[0, 0] > box_b[1, 0]) or (box_a[1, 1] < box_b[0, 1]) or
                (box_a[0, 1] > box_b[1, 1]) or (box_a[1, 2] < box_b[0, 2]) or (box_a[0, 2] > box_b[1, 2]))


def overlapping_boxes(established_box, new_box, trim_box):
    """Checks for 3D overlap of two boxes; returns True and sets trim_box if there is overlap, otherwise False.
    
    trim_box is modified in place.

    Arguments:
       established_box: numpy int array of shape (2, 3)
       new_box: numpy int array of shape (2, 3)
          each is lower & upper indices in 3 dimensions defining a logical cuboid subset of a 3D cartesian grid
          protocol of indices for the two boxes must be the same
       trim_box: numpy int array of shape (2, 3)
          set to lower & upper indices in 3 dimensions defining a logical cuboid subset of a 3D cartesian grid
          a subset of new_box such that if removed from new_box, a valid box would remain with no overlap with established_box
          indices protocol is the same as that used for established_box and new_box (if return value is True)
          if there is no overlap (return value False), all elements of trim_box are set to 0

    note:
       when there is overlap between the boxes, there can be more than one way to trim the new_box,
       with trim_box fully covering either ij, jk or ik planes of new_box
       the function selects the trim_box containing the minimum number of cells (minimum 'loss' to trimming)
       this function does not actually apply the trimming, ie. new_box is not modified here

    returns: boolean
       True if established_box and new_box overlap (implies trim_box valid), False otherwise (trim_box elements all 0)
    """

    assert established_box.ndim == 2 and established_box.shape == (2, 3) and established_box.dtype == 'int'
    assert new_box.ndim == 2 and new_box.shape == (2, 3) and new_box.dtype == 'int'
    assert trim_box.ndim == 2 and trim_box.shape == (2, 3) and trim_box.dtype == 'int'
    trim_box[:, :] = 0
    if not boxes_overlap(established_box, new_box):
        return False
    # determine trim direction based on minimizing number of cells to be trimmed
    new_box_area = np.zeros(
        3, dtype = 'int')  # compute number of cells in a 2D slice of new_box, taken in k, j or i directions
    # note: '_area' is a cell count, not an area in xyz space
    new_box_area[0] = (new_box[1, 1] - new_box[0, 1] + 1) * (new_box[1, 2] - new_box[0, 2] + 1)  # k slice
    new_box_area[1] = (new_box[1, 0] - new_box[0, 0] + 1) * (new_box[1, 2] - new_box[0, 2] + 1)  # j slice
    new_box_area[2] = (new_box[1, 0] - new_box[0, 0] + 1) * (new_box[1, 1] - new_box[0, 1] + 1)  # i slice
    # find the actual box of overlap and 'cost' of trimming in each dimension and minimize over k,j,i
    trim_cost = np.zeros(3, dtype = 'int')
    trim_direction = -1
    min_trim_cost = volume_of_box(new_box) + 1  # anything will be better than this
    overlap_box = np.zeros((2, 3), dtype = 'int')
    for dir in range(3):
        overlap_box[0, dir] = max(established_box[0, dir], new_box[0, dir])
        overlap_box[1, dir] = min(established_box[1, dir], new_box[1, dir])
        trim_cost[dir] = (overlap_box[1, dir] - overlap_box[0, dir] + 1) * new_box_area[dir]
        if trim_cost[dir] < min_trim_cost:
            min_trim_cost = trim_cost[dir]
            trim_direction = dir
    # set trim_box for chosen direction
    assert 0 <= trim_direction <= 2
    other_dir_a = (trim_direction + 1) % 3
    other_dir_b = (trim_direction + 2) % 3
    trim_box[0, trim_direction] = overlap_box[0, trim_direction]
    trim_box[1, trim_direction] = overlap_box[1, trim_direction]
    trim_box[0, other_dir_a] = new_box[0, other_dir_a]
    trim_box[1, other_dir_a] = new_box[1, other_dir_a]
    trim_box[0, other_dir_b] = new_box[0, other_dir_b]
    trim_box[1, other_dir_b] = new_box[1, other_dir_b]
    return True


def trim_box_by_box_returning_new_mask(box_to_be_trimmed, trim_box, mask_kji0):
    """Reduces box_to_be_trimmed by trim_box; trim_box must be a neat subset box at one face of box_to_be_trimmed.

    input/output argument (modified):
       box_to_be_trimmed: numpy int array of shape (2, 3)
          lower & upper indices in 3 dimensions defining a logical cuboid subset of a 3D cartesian grid
          indices protocol is python kji ordering with zero base
          modified to exclude space occupied by trim_box

    input arguments (unmodified):
       trim_box: numpy int array of shape (2, 3)
          lower & upper indices in 3 dimensions defining a logical cuboid subset of a 3D cartesian grid
          indices protocol is python kji ordering with zero base
          the volume to be removed from box_to_be_trimmed,
          must be a subset of box_to_be_trimmed completely covering one face of box_to_be_trimmed
          (thereby ensuring that after trimming, a valid cuboid box results)
       mask_kji0: numpy 3D boolean array of shape matching extent of input box_to_be_trimmed
          indices protocol is python kji ordering with zero base

    returns: numpy 3D boolean array of shape matching extent of output box_to_be_trimmed
       the return array is a version of mask_kji0 that has been trimmed in accordance with the box trimming
    """

    local_box = np.zeros((2, 3), dtype = 'int')
    local_box[1] = extent_of_box(box_to_be_trimmed) - 1
    if np.all(trim_box[0] == box_to_be_trimmed[0]):  # shift box_to_be_trimmed [0] (minumum) up (in logical space)
        if trim_box[1, 0] != box_to_be_trimmed[1, 0]:  # trim in k direction
            assert (trim_box[1, 1] == box_to_be_trimmed[1, 1]) and (trim_box[1, 2] == box_to_be_trimmed[1, 2])
            local_box[0, 0] = trim_box[1, 0] - box_to_be_trimmed[0, 0] + 1
            box_to_be_trimmed[0, 0] = trim_box[1, 0] + 1
        elif trim_box[1, 1] != box_to_be_trimmed[1, 1]:  # trim in j direction
            assert (trim_box[1, 2] == box_to_be_trimmed[1, 2])
            local_box[0, 1] = trim_box[1, 1] - box_to_be_trimmed[0, 1] + 1
            box_to_be_trimmed[0, 1] = trim_box[1, 1] + 1
        else:  # trim in i direction
            assert trim_box[1, 2] != box_to_be_trimmed[1, 2]
            local_box[0, 2] = trim_box[1, 2] - box_to_be_trimmed[0, 2] + 1
            box_to_be_trimmed[0, 2] = trim_box[1, 2] + 1
    elif np.all(trim_box[1] == box_to_be_trimmed[1]):  # shift box_to_be_trimmed [1] (maxumum) down (in logical space)
        if trim_box[0, 0] != box_to_be_trimmed[0, 0]:  # trim in k direction
            assert (trim_box[0, 1] == box_to_be_trimmed[0, 1]) and (trim_box[0, 2] == box_to_be_trimmed[0, 2])
            local_box[1, 0] -= box_to_be_trimmed[1, 0] - trim_box[0, 0] + 1
            box_to_be_trimmed[1, 0] = trim_box[0, 0] - 1
        elif trim_box[0, 1] != box_to_be_trimmed[0, 1]:  # trim in j direction
            assert (trim_box[0, 2] == box_to_be_trimmed[0, 2])
            local_box[1, 1] -= box_to_be_trimmed[1, 1] - trim_box[0, 1] + 1
            box_to_be_trimmed[1, 1] = trim_box[0, 1] - 1
        else:  # trim in i direction
            assert trim_box[0, 2] != box_to_be_trimmed[0, 2]
            local_box[1, 2] -= box_to_be_trimmed[1, 2] - trim_box[0, 2] + 1
            box_to_be_trimmed[1, 2] = trim_box[0, 2] - 1
    else:  # shouldn't happen
        log.critical('Trim box code failure: box to be trimmed is %s', string_iijjkk1_for_box_kji0(box_to_be_trimmed))
        log.critical('Trim box code failure:          trim box is %s', string_iijjkk1_for_box_kji0(trim_box))
        assert False
    assert np.all(box_to_be_trimmed[0] <= box_to_be_trimmed[1])
    return (mask_kji0[local_box[0, 0]:local_box[1, 0] + 1, local_box[0, 1]:local_box[1, 1] + 1,
                      local_box[0, 2]:local_box[1, 2] + 1]).copy()


def trim_box_to_mask_returning_new_mask(bounding_box_kji0, mask_kji0):
    """Reduce the coverage of bounding box to the minimum needed to contain True elements of mask.
    
    Returns trimmed mask.
    """
    # NB: bounding box is modified by this function
    assert bounding_box_kji0.ndim == 2 and bounding_box_kji0.shape == (2, 3) and bounding_box_kji0.dtype == 'int'
    assert (mask_kji0.ndim == 3 and
            np.all(np.array([mask_kji0.shape], dtype = 'int') == extent_of_box(bounding_box_kji0)) and
            mask_kji0.dtype == 'bool')
    assert np.any(mask_kji0)
    box_extent = extent_of_box(bounding_box_kji0)
    local_box = np.zeros((2, 3), dtype = 'int')
    local_box[0] = box_extent.copy()
    local_box[1] = -1
    for k in range(box_extent[0]):
        if np.any(mask_kji0[k, :, :]):
            local_box[1, 0] = k
            if local_box[0, 0] == box_extent[0]:
                local_box[0, 0] = k
    for j in range(box_extent[1]):
        if np.any(mask_kji0[:, j, :]):
            local_box[1, 1] = j
            if local_box[0, 1] == box_extent[1]:
                local_box[0, 1] = j
    for i in range(box_extent[2]):
        if np.any(mask_kji0[:, :, i]):
            local_box[1, 2] = i
            if local_box[0, 2] == box_extent[2]:
                local_box[0, 2] = i
    assert np.all(local_box >= 0)
    assert np.all(local_box[1] >= local_box[0])
    assert np.all(local_box[1] < box_extent)
    bounding_box_kji0[0] += local_box[0]
    bounding_box_kji0[1] -= (box_extent - local_box[1] - 1)
    assert np.all(bounding_box_kji0[0] <= bounding_box_kji0[1])
    return (mask_kji0[local_box[0, 0]:local_box[1, 0] + 1, local_box[0, 1]:local_box[1, 1] + 1,
                      local_box[0, 2]:local_box[1, 2] + 1]).copy()


# end of box_utilities module
