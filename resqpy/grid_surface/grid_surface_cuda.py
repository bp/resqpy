"""Cuda based grid surface intersection functionality for GPU processing.

note:
   use of this module requires accessible GPUs and the corresponding numba.cuda and cupy packages to be installed
"""

import logging

log = logging.getLogger(__name__)

import math as maths
import numpy as np
from typing import Tuple, Optional, Dict
import threading

import numba  # type: ignore
from numba import njit, cuda  # type: ignore
from numba.cuda.cudadrv.devicearray import DeviceNDArray  # type: ignore
import cupy  # type: ignore

import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.fault as rqf
import resqpy.property as rqp
import resqpy.grid_surface._find_faces as rgs_ff
import resqpy.olio.uuid as bu

compiler_lock = threading.Lock()  # Numba compiler is not threadsafe


# cuda device wrappers for numpy functions
@cuda.jit(device = True)
def _cross_d(A: DeviceNDArray, B: DeviceNDArray, c: DeviceNDArray):
    c[0] = A[1] * B[2] - A[2] * B[1]
    c[1] = A[2] * B[0] - A[0] * B[2]
    c[2] = A[0] * B[1] - A[1] * B[0]


@cuda.jit(device = True)
def _negative_d(v: DeviceNDArray, nv: DeviceNDArray):
    for d in range(v.shape[0]):
        nv[d] = numba.float32(-1.) * v[d]


@cuda.jit(device = True)
def _dot_d(v1: DeviceNDArray, v2: DeviceNDArray, prod: DeviceNDArray):
    prod[0] = 0.0
    for d in range(v1.shape[0]):
        prod[0] += v1[d] * v2[d]


@cuda.jit(device = True)
def _norm_d(v: DeviceNDArray, n: DeviceNDArray):
    n[0] = 0.
    for dim in range(3):
        n[0] += v[dim]**2.
    n[0] = maths.sqrt(n[0])


@cuda.jit
def project_polygons_to_surfaces(faces: DeviceNDArray, triangles: DeviceNDArray, axis: int, index1: int, index2: int,
                                 colx: int, coly: int, nx: int, ny: int, nz: int, dx: float, dy: float, dz: float,
                                 l_tol: float, t_tol: float, return_normal_vectors: bool, normals: DeviceNDArray,
                                 return_depths: bool, depths: DeviceNDArray, return_offsets: bool,
                                 offsets: DeviceNDArray, return_triangles: bool, triangle_per_face: DeviceNDArray):
    """Maps the projection of a 3D polygon to 2D grid surfaces along a given axis, using GPUs.

    arguments:
        faces (DeviceNDArray.bool): boolean array of each cell face that can represent the surface.
            nb. ordered k,j,i and sized (k,j,i)[axis] -= 1
        triangles (DeviceNDArray.float): ntriangles x naxis array containing (x,y,z) coordinates of each traingle.
        n_axis (int): number of cells in the axis.
        axis (int): axis number. Axis i is 0, j is 1, and k is 2.
        index1 (int): the first index. Axis i is 0, j is 0, and k is 1.
        index2 (int): the second index. Axis i is 1, j is 2, and k is 2.
        nx (int): number of points in x axis.
        ny (int): number of points in y axis.
        dx (float): cell's thickness along x-axis.
        dy (float): cell's thickness along y-axis.
        dz (float): cell's thickness along z-axis.
        l_tol (float, default 0.0): a fraction of the line length to allow for an intersection to be found
            just outside the segment.
        t_tol (float, default 0.0): a fraction of the triangle size to allow for an intersection to be found
            just outside the triangle.

    returns:
        void: modified faces array (INTENT OUT).
    """

    # define thread-local arrays used generally
    grid_nxyz = numba.cuda.local.array(3, numba.int32)
    grid_dxyz = numba.cuda.local.array(3, numba.float64)
    grid_nxyz[0] = nx
    grid_nxyz[1] = ny
    grid_nxyz[2] = nz
    grid_dxyz[0] = numba.float64(dx)
    grid_dxyz[1] = numba.float64(dy)
    grid_dxyz[2] = numba.float64(dz)
    # define thread-local arrays for section 3
    tp = numba.cuda.local.array((3, 3), numba.float64)
    line_p = numba.cuda.local.array(3, numba.float64)
    line_v = numba.cuda.local.array(3, numba.float64)
    p01 = numba.cuda.local.array(3, numba.float64)
    p02 = numba.cuda.local.array(3, numba.float64)
    lp_t0 = numba.cuda.local.array(3, numba.float64)
    norm = numba.cuda.local.array(3, numba.float64)
    line_rv = numba.cuda.local.array(3, numba.float64)
    tmp = numba.cuda.local.array(3, numba.float64)
    face_idx = numba.cuda.local.array(3, numba.int32)
    norm_idx = numba.cuda.local.array(3, numba.int32)
    xyz = numba.cuda.local.array(3, numba.float64)
    # scalars that must be returned from device functions must be mutable
    # => make them arrays
    u = numba.cuda.local.array(1, numba.float64)
    v = numba.cuda.local.array(1, numba.float64)
    denom = numba.cuda.local.array(1, numba.float64)
    t = numba.cuda.local.array(1, numba.float64)

    # extract useful dimension info
    n_axis = grid_nxyz[axis]  # get length of projection axis
    n_faces = faces.shape[2 - axis]  # n_faces == n_axis -1
    ntriangles = triangles.shape[0]

    #  cuda.grid(1) gives the thread index (blockIdx.x*blockDim.x + threadIdx.x)
    if cuda.grid(1) >= ntriangles:  # cuda.grid(1) evaluates to 1 int
        return  # this is actually unnecessary as the for-loop takes care of bounds

    # we have a set number of threads in a grid, so process each thread's
    # data and move it along to its next point in the array
    # Array:                          * * * * * * * * * * * * * * * * * * * * * * * * * * * * *|
    # > iteration 1- thread position: ^ ^ ^ ^ ^ ^ ^ ^                                          |        # cuda.grid(1)
    # > iteration 2- thread position:                 ^ ^ ^ ^ ^ ^ ^ ^                          |        # cuda.grid(1) + 1*cuda.gridsize(1)
    # > iteration 3- thread position:                                 ^ ^ ^ ^ ^ ^ ^ ^          |        # cuda.grid(1) + 2*cuda.gridsize(1)
    # > iteration 4- thread position:                                                 ^ ^ ^ ^ ^|x x x   # cuda.grid(1) + 3*cuda.gridsize(1)
    for triangle_num in range(cuda.grid(1), ntriangles, cuda.gridsize(1)):
        # the number of threads spawned should be enough to cover all triangles in one iteration
        # ...just imagine that this is the parallel section (like #pragma omp parallel)
        # 1. find triangle bounding box in this projection
        # 1a. get triangle under consideration
        # 1b. convert triangle-points coordinate to index

        for ver in range(3):  # for v in vertices
            for dim in range(3):  # for d in dimnensions
                tp[ver,
                   dim] = (numba.float64(triangles[triangle_num, ver, dim]) /
                           numba.float64(grid_dxyz[dim])) - numba.float64(0.5)  # get index of each aligned triangle

        # 1c. find triangle bounding box
        min_tpx = max(maths.ceil(min(tp[0, colx], tp[1, colx], tp[2, colx])), 0)
        max_tpx = min(maths.floor(max(tp[0, colx], tp[1, colx], tp[2, colx])), grid_nxyz[colx] - 1)
        if max_tpx < min_tpx:
            continue  # skip: triangle outside of grid
        min_tpy = max(maths.ceil(min(tp[0, coly], tp[1, coly], tp[2, coly])), 0)
        max_tpy = min(maths.floor(max(tp[0, coly], tp[1, coly], tp[2, coly])), grid_nxyz[coly] - 1)
        if max_tpy < min_tpy:
            continue  # skip: triangle outside of grid

        # 2. iterate over all points that fall within bounding box and
        # check whether points falls in triangle
        for py in range(min_tpy, max_tpy + 1, 1):
            for px in range(min_tpx, max_tpx + 1, 1):

                inside = False
                # 2a. use cross-product to work out Barycentric weights
                # this could be made prettier by refactoring a device function
                w1_denom = ((tp[1, coly] - tp[0, coly]) * (tp[2, colx] - tp[0, colx]) - (tp[1, colx] - tp[0, colx]) *
                            (tp[2, coly] - tp[0, coly]))
                w2_denom = (tp[2, coly] - tp[0, coly])
                if w1_denom == 0. or w2_denom == 0.:
                    inside = True  # point lies on a triangle which is actually a line (normally at boundaries)
                else:
                    w1 = (tp[0, colx] - numba.float64(px)) * (tp[2, coly] - tp[0, coly]) + (
                        numba.float64(py) - tp[0, coly]) * (tp[2, colx] - tp[0, colx])
                    w1 /= w1_denom
                    w2 = (numba.float64(py) - tp[0, coly] - w1 * (tp[1, coly] - tp[0, coly]))
                    w2 /= w2_denom
                    if (w1 >= 0. and w2 >= 0. and (w1 + w2) <= 1.):  # inside
                        inside = True  # point lies in triangle

                # 2b. the point is inside if Barycentric weights meet this condition
                if inside:
                    # 3. find intersection point with column centre
                    # 3a. Line start point in 3D which had a projection hit
                    line_p[axis] = numba.float64(grid_dxyz[axis]) / 2.
                    line_p[2 - index1] = (py + 0.5) * grid_dxyz[2 - index1]  # kji / xyz & py=d1
                    line_p[2 - index2] = (px + 0.5) * grid_dxyz[2 - index2]  # kji / xyz & px=d2

                    # 3b. Line end point in 3D
                    for dim in range(3):
                        line_v[dim] = line_p[dim]
                    line_v[axis] = numba.float64(grid_dxyz[axis]) * (n_axis - numba.float64(0.5))  #!
                    for dim in range(3):  # for d in dimensions
                        line_v[dim] -= line_p[dim]

                    # 3c.find depth intersection
                    for dim in range(3):  # for d in dimensions
                        p01[dim] = (tp[1, dim] - tp[0, dim]) * grid_dxyz[dim]
                        p02[dim] = (tp[2, dim] - tp[0, dim]) * grid_dxyz[dim]

                    _cross_d(p01, p02, norm)  # normal to plane
                    _negative_d(line_v, line_rv)
                    _dot_d(line_rv, norm, denom)

                    if denom[0] == 0.0:
                        continue  # line is parallel to plane

                    for dim in range(3):
                        lp_t0[dim] = line_p[dim] - (tp[0, dim] + 0.5) * grid_dxyz[dim]

                    _dot_d(norm, lp_t0, t)
                    t[0] /= denom[0]
                    if (t[0] < 0.0 - l_tol or t[0] > 1.0 + l_tol):
                        continue

                    _cross_d(p02, line_rv, tmp)
                    _dot_d(tmp, lp_t0, u)
                    u[0] /= denom[0]
                    if u[0] < 0.0 - t_tol or u[0] > 1.0 + t_tol:
                        continue

                    _cross_d(line_rv, p01, tmp)
                    _dot_d(tmp, lp_t0, v)
                    v[0] /= denom[0]
                    if v[0] < 0.0 - t_tol or u[0] + v[0] > 1.0 + t_tol:
                        continue

                    for dim in range(3):  # for d in dimensions
                        xyz[dim] = line_p[dim] + t[0] * line_v[dim]

                    # 4. mark the face corresponding to the grid and surface intersection at this point.
                    face = numba.int32((xyz[axis] - line_p[axis]) / grid_dxyz[axis])

                    if face == -1:  # handle rounding precision issues
                        face = 0
                    elif face == n_faces:
                        face -= 1
                    assert 0 <= face < n_faces

                    face_idx[index1] = int(py)
                    face_idx[index2] = int(px)
                    face_idx[2 - axis] = int(face)

                    faces[face_idx[0], face_idx[1], face_idx[2]] = True

                    if return_depths:
                        depths[face_idx[0], face_idx[1], face_idx[2]] = xyz[2]

                    if return_offsets:
                        offsets[face_idx[0], face_idx[1], face_idx[2]] = xyz[axis] - ((face + 1) * grid_dxyz[axis])

                    if return_normal_vectors:
                        for dim in range(3):
                            line_p[dim] = (tp[0, dim] - tp[1, dim]) * grid_dxyz[dim]
                            line_v[dim] = (tp[0, dim] - tp[2, dim]) * grid_dxyz[dim]
                        _cross_d(line_p, line_v, tmp)
                        _norm_d(tmp, v)
                        for dim in range(3):
                            normals[face_idx[0], face_idx[1], face_idx[2], dim] = -1. * tmp[dim] / v[0]
                        norm_idx[index2] = int(px)
                        if normals[norm_idx[0], norm_idx[1], norm_idx[2], 2] > 0.0:
                            for dim in range(3):
                                normals[face_idx[0], face_idx[1], face_idx[2], dim] *= -1.

                    if return_triangles:
                        triangle_per_face[face_idx[0], face_idx[1], face_idx[2]] = triangle_num


@cuda.jit
def _diffuse_closed_faces(a, k_faces, j_faces, i_faces, index1, index2, axis, start, stop, inc):

    tidx, tidy = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    maxidx, maxidy = a.shape[index1] - 2, a.shape[index2] - 2
    indices = numba.cuda.local.array(3, numba.int32)
    for D1 in range(tidx, maxidx, stridex):  # k vectorized
        for D2 in range(tidy, maxidy, stridey):  # j vectorized
            indices[index1] = D1 + 1
            indices[index2] = D2 + 1
            for D3 in range(start, stop, inc):  # iterate along i in kj-planes
                indices[axis] = D3 + 1
                i, j, k = indices[:]
                iF, jF, kF = i - 1, j - 1, k - 1  # faces arrays aren't padded
                fault_above = k_faces[iF - 1, jF, kF]
                fault_below = k_faces[iF, jF, kF]
                fault_left = j_faces[iF, jF - 1, kF]
                fault_right = j_faces[iF, jF, kF]
                fault_behind = i_faces[iF, jF, kF - 1]
                fault_back = i_faces[iF, jF, kF]
                cuda.syncthreads()
                a[i,j,k] = (a[i-1,j,k] and (not fault_above))  or (a[i+1,j,k] and (not fault_below)) \
                        or (a[i,j-1,k] and (not fault_left))   or (a[i,j+1,k] and (not fault_right)) \
                        or (a[i,j,k-1] and (not fault_behind)) or (a[i,j,k+1] and (not fault_back)) \
                        or a[i,j,k] # already closed
                cuda.syncthreads()


def bisector_from_faces_cuda(grid_extent_kji: Tuple[int, int, int], k_faces: np.ndarray, j_faces: np.ndarray,
                             i_faces: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Returns a numpy bool array denoting the bisection of the grid by the face sets, using GPUs.

    arguments:
        grid_extent_kji (triple int): the shape of the grid
        k_faces, j_faces, i_faces (numpy bool arrays): True where an internal grid face forms part of the
            bisecting surface

    returns:
        (numpy bool array of shape grid_extent_kji, bool) where the array is set True for cells on one side
        of the face sets deemed to be shallower (more strictly, lower K index on average); set False for cells
        on othe side; the bool value is True if the surface is a curtain (vertical), otherwise False

    notes:
        the face sets must form a single 'sealed' cut of the grid (eg. not waving in and out of the grid);
        any 'boxed in' parts of the grid (completely enclosed by bisecting faces) will be consistently
        assigned to either the True or False part
    """
    assert len(grid_extent_kji) == 3
    padded_extent_kji = (grid_extent_kji[0] + 2, grid_extent_kji[1] + 2, grid_extent_kji[2] + 2)
    a = cupy.zeros(padded_extent_kji, dtype = bool)
    a[1, 1, 1] = True

    a_count = a_count_before = 0
    blockSize = (16, 16)
    gridSize_k = ((grid_extent_kji[1] + blockSize[0] - 1) // blockSize[0],
                  (grid_extent_kji[2] + blockSize[1] - 1) // blockSize[1])
    gridSize_j = ((grid_extent_kji[0] + blockSize[0] - 1) // blockSize[0],
                  (grid_extent_kji[2] + blockSize[1] - 1) // blockSize[1])
    gridSize_i = ((grid_extent_kji[0] + blockSize[0] - 1) // blockSize[1],
                  (grid_extent_kji[1] + blockSize[1] - 1) // blockSize[1])

    while True:
        # forward sweeps
        _diffuse_closed_faces[gridSize_k, blockSize](a, k_faces, j_faces, i_faces, 1, 2, 0, 0, grid_extent_kji[0],
                                                     1)  # k-direction
        _diffuse_closed_faces[gridSize_j, blockSize](a, k_faces, j_faces, i_faces, 0, 2, 1, 0, grid_extent_kji[1],
                                                     1)  # j-direction
        _diffuse_closed_faces[gridSize_i, blockSize](a, k_faces, j_faces, i_faces, 0, 1, 2, 0, grid_extent_kji[2],
                                                     1)  # i-direction
        # reverse sweeps
        _diffuse_closed_faces[gridSize_k, blockSize](a, k_faces, j_faces, i_faces, 1, 2, 0, grid_extent_kji[0] - 1, -1,
                                                     -1)  # k-direction
        _diffuse_closed_faces[gridSize_j, blockSize](a, k_faces, j_faces, i_faces, 0, 2, 1, grid_extent_kji[1] - 1, -1,
                                                     -1)  # j-direction
        _diffuse_closed_faces[gridSize_i, blockSize](a, k_faces, j_faces, i_faces, 0, 1, 2, grid_extent_kji[2] - 1, -1,
                                                     -1)  # i-direction

        a_count = cupy.count_nonzero(a)

        if a_count == a_count_before:
            break
        a_count_before = a_count

    a = cupy.asnumpy(a[1:-1, 1:-1, 1:-1])
    cell_count = a.size
    assert 1 <= a_count < cell_count, 'face set for surface is leaky or empty (surface does not intersect grid)'

    # find mean K for a cells and not a cells; if not a cells mean K is lesser (ie shallower), negate a
    layer_cell_count = grid_extent_kji[1] * grid_extent_kji[2]
    a_k_sum = 0
    not_a_k_sum = 0
    for k in range(grid_extent_kji[0]):
        a_layer_count = np.count_nonzero(a[k])
        a_k_sum += (k + 1) * a_layer_count
        not_a_k_sum += (k + 1) * (layer_cell_count - a_layer_count)
    a_mean_k = float(a_k_sum) / float(a_count)
    not_a_mean_k = float(not_a_k_sum) / float(cell_count - a_count)
    is_curtain = False
    if a_mean_k > not_a_mean_k:
        a[:] = np.logical_not(a)
    elif abs(a_mean_k - not_a_mean_k) <= 0.01:
        # log.warning('unable to determine which side of surface is shallower')
        is_curtain = True

    return a, is_curtain


def find_faces_to_represent_surface_regular_cuda_sgpu(
    grid,
    surfaces,
    name,
    title = None,
    agitate = False,
    feature_type = 'fault',
    progress_fn = None,
    return_properties = None,
    i_surface = 0,
    i_gpu = 0,
    gcs_list = None,
    props_dict_list = None,
):
    """Returns a grid connection set containing those cell faces which are deemed to represent the surface, using GPUs.

    arguments:
        grid (RegularGrid): the grid for which to create a grid connection set representation of the surface;
           must be aligned, ie. I with +x, J with +y, K with +z and local origin of (0.0, 0.0, 0.0)
        surface (list(Surface)): the surface to be intersected with the grid
        name (str): the feature name to use in the grid connection set
        title (str, optional): the citation title to use for the grid connection set; defaults to name
        agitate (bool, default False): if True, the points of the surface are perturbed by a small random
           offset, which can help if the surface has been built from a regular mesh with a periodic resonance
           with the grid
        feature_type (str, default 'fault'): 'fault', 'horizon' or 'geobody boundary'
        progress_fn (f(x: float), optional): a callback function to be called at intervals by this function;
           the argument will progress from 0.0 to 1.0 in unspecified and uneven increments

    returns:
        gcs  or  (gcs, gcs_props)
        where gcs is a new GridConnectionSet with a single feature, not yet written to hdf5 nor xml created;
        gcs_props is a dictionary mapping from requested return_properties string to numpy array

    notes:
        this function is designed for aligned regular grids only;
        this function can handle the surface and grid being in different coordinate reference systems, as
        long as the implicit parent crs is shared;
        no trimming of the surface is carried out here: for computational efficiency, it is recommended
        to trim first;
        organisational objects for the feature are created if needed
    """
    # todo: update with extra arguments to keep functionality aligned with find_faces...regular_optimised

    cuda.select_device(i_gpu)  # bind device to thread
    device = cuda.get_current_device()  # if no GPU present - this will throw an exception and fall back to CPU

    assert isinstance(grid, grr.RegularGrid)
    assert grid.is_aligned
    return_triangles = False
    return_normal_vectors = False
    return_depths = False
    return_offsets = False
    return_bisector = False
    return_flange_bool = False
    if return_properties:
        assert all([
            p in ['triangle', 'depth', 'offset', 'normal vector', 'grid bisector', 'flange bool']
            for p in return_properties
        ])
        return_triangles = ('triangle' in return_properties)
        return_normal_vectors = ('normal vector' in return_properties)
        return_depths = ('depth' in return_properties)
        return_offsets = ('offset' in return_properties)
        return_bisector = ('grid bisector' in return_properties)
        return_flange_bool = ('flange bool' in return_properties)
        if return_flange_bool:
            return_triangles = True

    if title is None:
        title = name

    if progress_fn is not None:
        progress_fn(0.0)

    # prepare surfaces
    surface = surfaces[i_surface]  # get surface under consideration
    log.debug(f'intersecting surface {surface.title} with regular grid {grid.title} on a GPU')
    # log.debug(f'grid extent kji: {grid.extent_kji}')

    # print some information about the CUDA card
    log.debug(f'{device.name} | Device Controller {i_gpu} | ' +
              f'CC {device.COMPUTE_CAPABILITY_MAJOR}.{device.COMPUTE_CAPABILITY_MINOR} | ' +
              f'Processing surface {i_surface}')
    # get device attributes to calculate thread dimensions
    nSMs = device.MULTIPROCESSOR_COUNT  # number of SMs
    maxBlockSize = device.MAX_BLOCK_DIM_X / 2  # max number of threads per block in x-dim
    gridSize = 2 * nSMs  # prefer 2*nSMs blocks for full occupancy
    # take the reverse diagonal for relationship between xyz & ijk
    grid_dxyz = (grid.block_dxyz_dkji[2, 0], grid.block_dxyz_dkji[1, 1], grid.block_dxyz_dkji[0, 2])
    # extract polygons from surface
    with compiler_lock:  # HDF5 handles seem not to be threadsafe
        triangles, points = surface.triangles_and_points()
    assert triangles is not None and points is not None, f'surface {surface.title} is empty'

    if agitate:
        points += 1.0e-5 * (np.random.random(points.shape) - 0.5)  # +/- uniform err.
    # log.debug(f'surface: {surface.title}; p0: {points[0]}; crs uuid: {surface.crs_uuid}')
    # log.debug(f'surface min xyz: {np.min(points, axis = 0)}')
    # log.debug(f'surface max xyz: {np.max(points, axis = 0)}')
    if not bu.matching_uuids(grid.crs_uuid, surface.crs_uuid):
        log.debug('converting from surface crs to grid crs')
        s_crs = rqc.Crs(surface.model, uuid = surface.crs_uuid)
        s_crs.convert_array_to(grid.crs, points)
        surface.crs_uuid = grid.crs.uuid
        # log.debug(f'surface: {surface.title}; p0: {points[0]}; crs uuid: {surface.crs_uuid}')
        # log.debug(f'surface min xyz: {np.min(points, axis = 0)}')
        # log.debug(f'surface max xyz: {np.max(points, axis = 0)}')

    p_tri_xyz = points[triangles]
    p_tri_xyz_d = cupy.asarray(p_tri_xyz)

    # K direction (xy projection)
    if grid.nk > 1:
        log.debug("searching for k faces")
        k_faces = np.zeros((grid.nk - 1, grid.nj, grid.ni), dtype = bool)
        k_triangles = np.full((grid.nk - 1, grid.nj, grid.ni), -1, dtype = int) if return_triangles else np.full(
            (1, 1, 1), -1, dtype = int)
        k_depths = np.full((grid.nk - 1, grid.nj, grid.ni), np.nan) if return_depths else np.full((1, 1, 1), np.nan)
        k_offsets = np.full((grid.nk - 1, grid.nj, grid.ni), np.nan) if return_offsets else np.full((1, 1, 1), np.nan)
        k_normals = np.full((grid.nk - 1, grid.nj, grid.ni, 3), np.nan) if return_normal_vectors else np.full(
            (1, 1, 1, 1), np.nan)
        k_faces_d = cupy.asarray(k_faces)
        k_triangles_d = cupy.asarray(k_triangles)
        k_depths_d = cupy.asarray(k_depths)
        k_offsets_d = cupy.asarray(k_offsets)
        k_normals_d = cupy.asarray(k_normals)
        colx = 0
        coly = 1
        axis = 2
        index1 = 1
        index2 = 2
        blockSize = (p_tri_xyz.shape[0] - 1) // (gridSize - 1) if (
            p_tri_xyz.shape[0] < gridSize * maxBlockSize) else 64  # prefer factors of 32 (threads per warp)
        log.debug(
            f'Executing polygon-intersection GPU-kernel along k-axis using gridSize={gridSize}, blockSize={blockSize}')
        project_polygons_to_surfaces[gridSize, blockSize](
            k_faces_d,
            p_tri_xyz_d,
            axis,
            index1,
            index2,
            colx,
            coly,
            grid.ni,
            grid.nj,
            grid.nk,
            grid_dxyz[0],
            grid_dxyz[1],
            grid_dxyz[2],
            0.,
            0.,
            return_normal_vectors,
            k_normals_d,
            return_depths,
            k_depths_d,
            return_offsets,
            k_offsets_d,
            return_triangles,
            k_triangles_d,
        )
        k_faces = cupy.asnumpy(k_faces_d)
        if not return_bisector:
            del k_faces_d
        k_triangles = cupy.asnumpy(k_triangles_d)
        del k_triangles_d
        k_depths = cupy.asnumpy(k_depths_d)
        del k_depths_d
        k_offsets = cupy.asnumpy(k_offsets_d)
        del k_offsets_d
        k_normals = cupy.asnumpy(k_normals_d)
        del k_normals_d
        log.debug(f"k face count: {np.count_nonzero(k_faces)}")
    else:
        k_faces = None

    if progress_fn is not None:
        progress_fn(0.3)

    # J direction (xz projection)
    if grid.nj > 1:
        log.debug("searching for j faces")
        j_faces = np.zeros((grid.nk, grid.nj - 1, grid.ni), dtype = bool)
        j_triangles = np.full((grid.nk, grid.nj - 1, grid.ni), -1, dtype = int) if return_triangles else np.full(
            (1, 1, 1), -1, dtype = int)
        j_depths = np.full((grid.nk, grid.nj - 1, grid.ni), np.nan) if return_depths else np.full((1, 1, 1), np.nan)
        j_offsets = np.full((grid.nk, grid.nj - 1, grid.ni), np.nan) if return_offsets else np.full((1, 1, 1), np.nan)
        j_normals = np.full((grid.nk, grid.nj - 1, grid.ni, 3), np.nan) if return_normal_vectors else np.full(
            (1, 1, 1, 1), np.nan)
        j_faces_d = cupy.asarray(j_faces)
        j_triangles_d = cupy.asarray(j_triangles)
        j_depths_d = cupy.asarray(j_depths)
        j_offsets_d = cupy.asarray(j_offsets)
        j_normals_d = cupy.asarray(j_normals)
        colx = 0
        coly = 2
        axis = 1
        index1 = 0
        index2 = 2
        blockSize = (p_tri_xyz.shape[0] - 1) // (gridSize - 1) if (
            p_tri_xyz.shape[0] < gridSize * maxBlockSize) else 64  # prefer factors of 32 (threads per warp)
        log.debug(
            f'Executing polygon-intersection GPU-kernel along j-axis using gridSize={gridSize}, blockSize={blockSize}')
        project_polygons_to_surfaces[gridSize, blockSize](
            j_faces_d,
            p_tri_xyz_d,
            axis,
            index1,
            index2,
            colx,
            coly,
            grid.ni,
            grid.nj,
            grid.nk,
            grid_dxyz[0],
            grid_dxyz[1],
            grid_dxyz[2],
            0.,
            0.,
            return_normal_vectors,
            j_normals_d,
            return_depths,
            j_depths_d,
            return_offsets,
            j_offsets_d,
            return_triangles,
            j_triangles_d,
        )
        j_faces = cupy.asnumpy(j_faces_d)
        if not return_bisector:
            del j_faces_d
        j_triangles = cupy.asnumpy(j_triangles_d)
        del j_triangles_d
        j_depths = cupy.asnumpy(j_depths_d)
        del j_depths_d
        j_offsets = cupy.asnumpy(j_offsets_d)
        del j_offsets_d
        j_normals = cupy.asnumpy(j_normals_d)
        del j_normals_d
        log.debug(f"j face count: {np.count_nonzero(j_faces)}")
    else:
        j_faces = None

    if progress_fn is not None:
        progress_fn(0.6)

    # I direction (yz projection)
    if grid.ni > 1:
        log.debug("searching for i faces")
        i_faces = np.zeros((grid.nk, grid.nj, grid.ni - 1), dtype = bool)
        i_triangles = np.full((grid.nk, grid.nj, grid.ni - 1), -1, dtype = int) if return_triangles else np.full(
            (1, 1, 1), -1, dtype = int)
        i_depths = np.full((grid.nk, grid.nj, grid.ni - 1), np.nan) if return_depths else np.full((1, 1, 1), np.nan)
        i_offsets = np.full((grid.nk, grid.nj, grid.ni - 1), np.nan) if return_offsets else np.full((1, 1, 1), np.nan)
        i_normals = np.full((grid.nk, grid.nj, grid.ni - 1, 3), np.nan) if return_normal_vectors else np.full(
            (1, 1, 1, 1), np.nan)
        i_faces_d = cupy.asarray(i_faces)
        i_triangles_d = cupy.asarray(i_triangles)
        i_depths_d = cupy.asarray(i_depths)
        i_offsets_d = cupy.asarray(i_offsets)
        i_normals_d = cupy.asarray(i_normals)
        colx = 1
        coly = 2
        axis = 0
        index1 = 0
        index2 = 1
        blockSize = (p_tri_xyz.shape[0] - 1) // (gridSize - 1) if (
            p_tri_xyz.shape[0] < gridSize * maxBlockSize) else 64  # prefer factors of 32 (threads per warp)
        log.debug(
            f'Executing polygon-intersection GPU-kernel along i-axis using gridSize={gridSize}, blockSize={blockSize}')
        project_polygons_to_surfaces[gridSize, blockSize](
            i_faces_d,
            p_tri_xyz_d,
            axis,
            index1,
            index2,
            colx,
            coly,
            grid.ni,
            grid.nj,
            grid.nk,
            grid_dxyz[0],
            grid_dxyz[1],
            grid_dxyz[2],
            0.,
            0.,
            return_normal_vectors,
            i_normals_d,
            return_depths,
            i_depths_d,
            return_offsets,
            i_offsets_d,
            return_triangles,
            i_triangles_d,
        )
        i_faces = cupy.asnumpy(i_faces_d)
        if not return_bisector:
            del i_faces_d
        i_triangles = cupy.asnumpy(i_triangles_d)
        del i_triangles_d
        i_depths = cupy.asnumpy(i_depths_d)
        del i_depths_d
        i_offsets = cupy.asnumpy(i_offsets_d)
        del i_offsets_d
        i_normals = cupy.asnumpy(i_normals_d)
        del i_normals_d
        log.debug(f"i face count: {np.count_nonzero(i_faces)}")
    else:
        i_faces = None

    del p_tri_xyz_d

    if progress_fn is not None:
        progress_fn(0.9)

    log.debug("converting face sets into grid connection set")
    gcs_list[i_surface] = rqf.GridConnectionSet(
        grid.model,
        grid = grid,
        k_faces = k_faces,
        j_faces = j_faces,
        i_faces = i_faces,
        k_sides = None,
        j_sides = None,
        i_sides = None,
        feature_name = name,
        feature_type = feature_type,
        title = title,
        create_organizing_objects_where_needed = True,
    )

    # NB. following assumes faces have been added to gcs in a particular order!
    if return_triangles:
        k_tri_list = np.empty((0,)) if k_triangles is None else k_triangles[rgs_ff.where_true(k_faces)]
        j_tri_list = np.empty((0,)) if j_triangles is None else j_triangles[rgs_ff.where_true(j_faces)]
        i_tri_list = np.empty((0,)) if i_triangles is None else i_triangles[rgs_ff.where_true(i_faces)]
        all_tris = np.concatenate((k_tri_list, j_tri_list, i_tri_list), axis = 0)
        # log.debug(f'gcs count: {gcs.count}; all triangles shape: {all_tris.shape}')
        assert all_tris.shape == (gcs_list[i_surface].count,)

    # NB. following assumes faces have been added to gcs in a particular order!
    if return_depths:
        k_depths_list = np.empty((0,)) if k_depths is None else k_depths[rgs_ff.where_true(k_faces)]
        j_depths_list = np.empty((0,)) if j_depths is None else j_depths[rgs_ff.where_true(j_faces)]
        i_depths_list = np.empty((0,)) if i_depths is None else i_depths[rgs_ff.where_true(i_faces)]
        all_depths = np.concatenate((k_depths_list, j_depths_list, i_depths_list), axis = 0)
        # log.debug(f'gcs count: {gcs.count}; all depths shape: {all_depths.shape}')
        assert all_depths.shape == (gcs_list[i_surface].count,)

    # NB. following assumes faces have been added to gcs in a particular order!
    if return_offsets:
        k_offsets_list = np.empty((0,)) if k_offsets is None else k_offsets[rgs_ff.where_true(k_faces)]
        j_offsets_list = np.empty((0,)) if j_offsets is None else j_offsets[rgs_ff.where_true(j_faces)]
        i_offsets_list = np.empty((0,)) if i_offsets is None else i_offsets[rgs_ff.where_true(i_faces)]
        all_offsets = np.concatenate((k_offsets_list, j_offsets_list, i_offsets_list), axis = 0)
        # log.debug(f'gcs count: {gcs.count}; all offsets shape: {all_offsets.shape}')
        assert all_offsets.shape == (gcs_list[i_surface].count,)

    if return_flange_bool:
        flange_bool_uuid = surface.model.uuid(title = 'flange bool',
                                              obj_type = 'DiscreteProperty',
                                              related_uuid = surface.uuid)
        assert flange_bool_uuid is not None, f"No flange bool property found for surface: {surface.title}"
        flange_bool = rqp.Property(surface.model, uuid = flange_bool_uuid)
        flange_array = flange_bool.array_ref()
        all_flange = np.take(flange_array, all_tris)
        assert all_flange.shape == (gcs_list[i_surface].count,)

    # NB. following assumes faces have been added to gcs in a particular order!
    if return_normal_vectors:
        k_normals_list = np.empty((0, 3)) if k_normals is None else k_normals[rgs_ff.where_true(k_faces)]
        j_normals_list = np.empty((0, 3)) if j_normals is None else j_normals[rgs_ff.where_true(j_faces)]
        i_normals_list = np.empty((0, 3)) if i_normals is None else i_normals[rgs_ff.where_true(i_faces)]
        all_normals = np.concatenate((k_normals_list, j_normals_list, i_normals_list), axis = 0)
        # log.debug(f'gcs count: {gcs.count}; all normals shape: {all_normals.shape}')
        assert all_normals.shape == (gcs_list[i_surface].count, 3)

    # note: following is a grid cells property, not a gcs property
    if return_bisector:
        bisector, is_curtain = bisector_from_faces_cuda(tuple(grid.extent_kji), k_faces_d, j_faces_d, i_faces_d)
        del k_faces_d, j_faces_d, i_faces_d

    if progress_fn is not None:
        progress_fn(1.0)

    # if returning properties, construct dictionary
    if return_properties:
        props_dict_list[i_surface] = {}
        if return_triangles:
            props_dict_list[i_surface]['triangle'] = all_tris
        if return_depths:
            props_dict_list[i_surface]['depth'] = all_depths
        if return_offsets:
            props_dict_list[i_surface]['offset'] = all_offsets
        if return_normal_vectors:
            props_dict_list[i_surface]['normal vector'] = all_normals
        if return_bisector:
            props_dict_list[i_surface]['grid bisector'] = (bisector, is_curtain)
        if return_flange_bool:
            props_dict_list[i_surface]['flange bool'] = all_flange


def find_faces_to_represent_surface_regular_cuda_mgpu(
    grid,
    surface,
    name,
    title = None,
    agitate = False,
    feature_type = 'fault',
    progress_fn = None,
    return_properties = None,
):
    """Returns a grid connection set containing those cell faces which are deemed to represent the surface, using GPUs.

    arguments:
        grid (RegularGrid): the grid for which to create a grid connection set representation of the surface;
           must be aligned, ie. I with +x, J with +y, K with +z and local origin of (0.0, 0.0, 0.0)
        surface (Surface or list(Surface)): the surface(s) to be intersected with the grid
        name (str): the feature name to use in the grid connection set
        title (str, optional): the citation title to use for the grid connection set; defaults to name
        agitate (bool, default False): if True, the points of the surface are perturbed by a small random
           offset, which can help if the surface has been built from a regular mesh with a periodic resonance
           with the grid
        feature_type (str, default 'fault'): 'fault', 'horizon' or 'geobody boundary'
        progress_fn (f(x: float), optional): a callback function to be called at intervals by this function;
           the argument will progress from 0.0 to 1.0 in unspecified and uneven increments

    returns:
        gcs  or  (gcs, gcs_props)
        where gcs is a new GridConnectionSet with a single feature, not yet written to hdf5 nor xml created;
        gcs_props is a dictionary mapping from requested return_properties string to numpy array

    notes:
        this function is designed for aligned regular grids only;
        this function can handle the surface and grid being in different coordinate reference systems, as
        long as the implicit parent crs is shared;
        no trimming of the surface is carried out here: for computational efficiency, it is recommended
        to trim first;
        organisational objects for the feature are created if needed
    """

    surfaces = surface if isinstance(surface, list) else [surface]

    n_surfs = len(surfaces)
    n_gpus = len(cuda.list_devices())
    log.debug("distributing %d surface between %d GPUs" % (n_surfs, n_gpus))
    gcs_list = [None] * n_surfs
    props_dict_list = [None] * n_surfs
    threads = [None] * n_gpus
    for i_surface in range(n_surfs):
        threads[i_surface % n_gpus] = threading.Thread(target = find_faces_to_represent_surface_regular_cuda_sgpu,
                                                       args = (
                                                           grid,
                                                           surfaces,
                                                           name,
                                                           title,
                                                           agitate,
                                                           feature_type,
                                                           progress_fn,
                                                           return_properties,
                                                           i_surface,
                                                           i_surface % n_gpus,
                                                           gcs_list,
                                                           props_dict_list,
                                                       ))
        threads[i_surface % n_gpus].start()  # start parallel run
        # if this is the last GPU available or we're at the last array ...
        if (i_surface + 1) % n_gpus == 0 or (i_surface + 1) == n_surfs:
            # ... sync all the GPUs being used
            for i_gpu in range(i_surface % n_gpus + 1):  # up to the number of GPUs being used
                threads[i_gpu].join()  # rejoin the main thread (syncthreads)

    if n_surfs > 1:
        return (gcs_list, props_dict_list) if return_properties else gcs_list
    else:
        return (gcs_list[0], props_dict_list[0]) if return_properties else gcs_list[0]
