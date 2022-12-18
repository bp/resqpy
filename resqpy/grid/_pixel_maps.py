"""RESQML grid module handling grid pixel maps."""

import numpy as np

import resqpy.olio.point_inclusion as pip


def pixel_maps(grid, origin, width, height, dx, dy = None, k0 = None, vertical_ref = 'top'):
    """Makes a mapping from pixels to cell j, i indices, based on split horizon points for a single horizon.

    args:
       origin (float pair): x, y of south west corner of area covered by pixel rectangle, in local crs
       width (int): the width of the pixel rectangle (number of pixels)
       height (int): the height of the pixel rectangle (number of pixels)
       dx (float): the size (west to east) of a pixel, in locel crs
       dy (float, optional): the size (south to north) of a pixel, in locel crs; defaults to dx
       k0 (int, default None): if present, the single layer to create a 2D pixel map for; if None, a 3D map
          is created with one layer per layer of the grid
       vertical_ref (string, default 'top'): 'top' or 'base'

    returns:
       numpy int array of shape (height, width, 2), or (nk, height, width, 2), being the j, i indices of cells
       that the pixel centres lie within; values of -1 are used as null (ie. pixel not within any cell)
    """

    if len(origin) == 3:
        origin = tuple(origin[0:2])
    assert len(origin) == 2
    assert width > 0 and height > 0
    if dy is None:
        dy = dx
    assert dx > 0.0 and dy > 0.0
    if k0 is not None:
        assert 0 <= k0 < grid.nk
    assert vertical_ref in ['top', 'base']

    kp = 0 if vertical_ref == 'top' else 1
    if k0 is not None:
        hp = grid.split_horizon_points(ref_k0 = k0, masked = False, kp = kp)
        p_map = grid.pixel_map_for_split_horizon_points(hp, origin, width, height, dx, dy = dy)
    else:
        _, _, raw_k = grid.extract_k_gaps()
        hp = grid.split_horizons_points(masked = False)
        p_map = np.empty((grid.nk, height, width, 2), dtype = int)
        for k0 in range(grid.nk):
            rk0 = raw_k[k0] + kp
            p_map[k0] = grid.pixel_map_for_split_horizon_points(hp[rk0], origin, width, height, dx, dy = dy)
    return p_map


def pixel_map_for_split_horizon_points(grid, horizon_points, origin, width, height, dx, dy = None):
    """Makes a mapping from pixels to cell j, i indices, based on split horizon points for a single horizon.

    args:
       horizon_points (numpy array of shape (nj, ni, 2, 2, 2+)): corner point x,y,z values for cell
          corners (j, i, jp, ip); as returned by split_horizon_points()
       origin (float pair): x, y of south west corner of area covered by pixel rectangle, in local crs
       width (int): the width of the pixel rectangle (number of pixels)
       height (int): the height of the pixel rectangle (number of pixels)
       dx (float): the size (west to east) of a pixel, in locel crs
       dx (float, optional): the size (south to north) of a pixel, in locel crs; defaults to dx

    returns:
       numpy int array of shape (height, width, 2), being the j, i indices of cells that the pixel centres lie within;
       values of -1 are used as null (ie. pixel not within any cell)
    """

    if dy is None:
        dy = dx
    half_dx = 0.5 * dx
    half_dy = 0.5 * dy
    d = np.array((dx, dy))
    half_d = np.array((half_dx, half_dy))

    #     north_east = np.array(origin) + np.array((width * dx, height * dy))

    p_map = np.full((height, width, 2), -1, dtype = int)

    # switch from logical corner ordering to polygon ordering
    poly_points = horizon_points[..., :2].copy()
    poly_points[:, :, 1, 1] = horizon_points[:, :, 1, 0, :2]
    poly_points[:, :, 1, 0] = horizon_points[:, :, 1, 1, :2]
    poly_points = poly_points.reshape(horizon_points.shape[0], horizon_points.shape[1], 4, 2)

    poly_box = np.empty((2, 2))
    patch_p_origin = np.empty((2,), dtype = int)  # NB. ordering is (ncol, nrow)
    patch_origin = np.empty((2,))
    patch_extent = np.empty((2,), dtype = int)  # NB. ordering is (ncol, nrow)

    for j in range(poly_points.shape[0]):
        for i in range(poly_points.shape[1]):
            if np.any(np.isnan(poly_points[j, i])):
                continue
            poly_box[0] = np.min(poly_points[j, i], axis = 0) - half_d
            poly_box[1] = np.max(poly_points[j, i], axis = 0) + half_d
            patch_p_origin[:] = (poly_box[0] - origin) / (dx, dy)
            if patch_p_origin[0] < 0 or patch_p_origin[1] < 0:
                continue
            patch_extent[:] = np.ceil((poly_box[1] - poly_box[0]) / (dx, dy))
            if patch_p_origin[0] + patch_extent[0] > width or patch_p_origin[1] + patch_extent[1] > height:
                continue
            patch_origin = origin + d * patch_p_origin + half_d
            scan_mask = pip.scan(patch_origin, patch_extent[0], patch_extent[1], dx, dy, poly_points[j, i])
            patch_mask = np.stack((scan_mask, scan_mask), axis = -1)
            old_patch = p_map[patch_p_origin[1]:patch_p_origin[1] + patch_extent[1],
                              patch_p_origin[0]:patch_p_origin[0] + patch_extent[0], :].copy()
            new_patch = np.empty(old_patch.shape, dtype = int)
            new_patch[:, :] = (j, i)
            p_map[patch_p_origin[1]:patch_p_origin[1] + patch_extent[1],
            patch_p_origin[0]:patch_p_origin[0] + patch_extent[0], :] = \
                np.where(patch_mask, new_patch, old_patch)
    return p_map
