"""volume.py: Functions to calculate volumes of hexahedral cells; assumes consistent length units."""

import numpy as np


def _pyr(ra, ab, ac, ad, db):  # returns 6 * volume of one quad pyramid, defined by specific vectors
    return np.dot(ra, np.cross(db, ac)) + 0.5 * np.dot(ac, np.cross(ad, ab))


def pyramid_volume(apex, a, b, c, d, crs_is_right_handed = False):
    """Returns volume of a quadrilateral pyramid.

    arguments:
       apex (triple float): location of the apex of the pyramid
       a, b, c, d (each triple float): locations of corners of base of pyramid; clockwise viewed from apex
          for a left handed crs
       crs_is_right_handed (boolean, default False): set True if xyz axes of crs are right handed

    returns:
       float, being the volume of the pyramid; units are implied by crs units in use by the vertices
    """

    apex = np.array(apex)
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    v = _pyr(a - apex, b - a, c - a, d - a, b - d) / 6.0
    if not crs_is_right_handed:
        v = -v
    return v


def tetrahedron_volume(a, b, c, d):
    """Returns volume of a tetrahedron.

    arguments:
       a, b, c, d (each triple float): locations of corners of tetrahedron
       crs_is_right_handed (boolean, default False): set True if xyz axes of crs are right handed

    returns:
       float, being the volume of the tetrahedron; units are implied by crs units in use by the vertices
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)

    return np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0


def tetra_cell_volume(cp, centre = None, off_hand = False):
    """Returns volume of single hexahedral cell with corner points cp of shape (2, 2, 2, 3); assumes bilinear faces."""

    if centre is None:
        centre = np.mean(cp.reshape((8, 3)), axis = 0)
    r0 = cp[0, 0, 0] - centre
    r1 = cp[1, 1, 1] - centre
    v = 0.0
    v += _pyr(r0, cp[0, 0, 1] - cp[0, 0, 0], cp[0, 1, 1] - cp[0, 0, 0], cp[0, 1, 0] - cp[0, 0, 0],
              cp[0, 1, 0] - cp[0, 0, 1])
    v += _pyr(r0, cp[1, 0, 0] - cp[0, 0, 0], cp[1, 0, 1] - cp[0, 0, 0], cp[0, 0, 1] - cp[0, 0, 0],
              cp[0, 0, 1] - cp[1, 0, 0])
    v += _pyr(r0, cp[0, 1, 0] - cp[0, 0, 0], cp[1, 1, 0] - cp[0, 0, 0], cp[1, 0, 0] - cp[0, 0, 0],
              cp[1, 0, 0] - cp[0, 1, 0])
    v += _pyr(r1, cp[1, 0, 1] - cp[1, 1, 1], cp[1, 0, 0] - cp[1, 1, 1], cp[1, 1, 0] - cp[1, 1, 1],
              cp[1, 1, 0] - cp[1, 0, 1])
    v += _pyr(r1, cp[1, 1, 0] - cp[1, 1, 1], cp[0, 1, 0] - cp[1, 1, 1], cp[0, 1, 1] - cp[1, 1, 1],
              cp[0, 1, 1] - cp[1, 1, 0])
    v += _pyr(r1, cp[0, 1, 1] - cp[1, 1, 1], cp[0, 0, 1] - cp[1, 1, 1], cp[1, 0, 1] - cp[1, 1, 1],
              cp[1, 0, 1] - cp[0, 1, 1])

    if off_hand:
        v = -v

    return v / 6.0


def tetra_volumes_slow(cp, centres = None, off_hand = False):
    """Returns volume array for all hexahedral cells assuming bilinear faces, using loop over cells."""

    # NB: deprecated, superceded by much faster function below
    # todo: handle NaNs
    # Pagoda style corner point data
    assert cp.ndim == 7

    flat = cp.reshape(-1, 2, 2, 2, 3)
    cells = flat.shape[0]
    if centres is None:
        centres = np.mean(flat, axis = (1, 2, 3))
    else:
        centres = centres.reshape((-1, 3))
    volumes = np.zeros(cells)
    for cell in range(cells):
        volumes[cell] = tetra_cell_volume(flat[cell], centre = centres[cell], off_hand = off_hand)
    return volumes.reshape(cp.shape[0:3])


def tetra_volumes(cp, centres = None, off_hand = False):
    """Returns volume array for all hexahedral cells assuming bilinear faces, using numpy operations.

    arguments:
       cp (7D numpy array of floats): cell corner point data in Pagoda 7D format [nk, nj, ni, kp, jp, ip, xyz]
       centres (optional, 4D numpy array of floats): cell centre points [nk, nj, ni, xyz]; calculated if None
       off_hand (boolean, default False): if True, the handedness of IJK space is the opposite of that for xyz
          space; if this argument is not set correctly, negative volumes will be returned

    returns:
       numpy 3D array of floats being the cell volumes [nk, nj, ni]

    note:
       length units are assumed to be consistent in x, y & z; and untis of returned volumes are implicitly
       those length units cubed
    """

    def pyrs(ra, ab, ac, ad, db):  # returns 6 * volume of one quad pyramid (for each cell)
        return np.sum(ra * np.cross(db, ac), axis = -1) + 0.5 * np.sum(ac * np.cross(ad, ab), axis = -1)

    # Pagoda style corner point data
    assert cp.ndim == 7

    flat = cp.reshape(-1, 2, 2, 2, 3)
    cells = flat.shape[0]
    if centres is None:
        centres = np.mean(flat, axis = (1, 2, 3))
    else:
        centres = centres.reshape((-1, 3))
    v = np.zeros(cells)
    # todo: numpy array operation covering all cells in grid

    r0 = flat[:, 0, 0, 0] - centres
    r1 = flat[:, 1, 1, 1] - centres

    v += pyrs(r0, flat[:, 0, 0, 1] - flat[:, 0, 0, 0], flat[:, 0, 1, 1] - flat[:, 0, 0, 0],
              flat[:, 0, 1, 0] - flat[:, 0, 0, 0], flat[:, 0, 1, 0] - flat[:, 0, 0, 1])
    v += pyrs(r0, flat[:, 1, 0, 0] - flat[:, 0, 0, 0], flat[:, 1, 0, 1] - flat[:, 0, 0, 0],
              flat[:, 0, 0, 1] - flat[:, 0, 0, 0], flat[:, 0, 0, 1] - flat[:, 1, 0, 0])
    v += pyrs(r0, flat[:, 0, 1, 0] - flat[:, 0, 0, 0], flat[:, 1, 1, 0] - flat[:, 0, 0, 0],
              flat[:, 1, 0, 0] - flat[:, 0, 0, 0], flat[:, 1, 0, 0] - flat[:, 0, 1, 0])
    v += pyrs(r1, flat[:, 1, 0, 1] - flat[:, 1, 1, 1], flat[:, 1, 0, 0] - flat[:, 1, 1, 1],
              flat[:, 1, 1, 0] - flat[:, 1, 1, 1], flat[:, 1, 1, 0] - flat[:, 1, 0, 1])
    v += pyrs(r1, flat[:, 1, 1, 0] - flat[:, 1, 1, 1], flat[:, 0, 1, 0] - flat[:, 1, 1, 1],
              flat[:, 0, 1, 1] - flat[:, 1, 1, 1], flat[:, 0, 1, 1] - flat[:, 1, 1, 0])
    v += pyrs(r1, flat[:, 0, 1, 1] - flat[:, 1, 1, 1], flat[:, 0, 0, 1] - flat[:, 1, 1, 1],
              flat[:, 1, 0, 1] - flat[:, 1, 1, 1], flat[:, 1, 0, 1] - flat[:, 0, 1, 1])

    v /= 6.0

    if off_hand:
        v = np.negative(v)
    return v.reshape(cp.shape[0:3])
