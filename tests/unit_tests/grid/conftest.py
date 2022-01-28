import pytest

import resqpy.property as rqp
from resqpy.model import Model
from resqpy.grid import Grid

from inspect import getsourcefile

import math as maths
import os

import numpy as np
import pandas as pd

import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.model as rq
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.olio.xml_et as rqet
import resqpy.rq_import as rqi
import resqpy.well as rqw

from inspect import getsourcefile


@pytest.fixture
def model_test(tmp_path) -> Model:
    epc = os.path.join(tmp_path, 'test.epc')
    return rq.new_model(epc)


@pytest.fixture
def basic_regular_grid(model_test: Model) -> Grid:
    return grr.RegularGrid(model_test, extent_kji = (2, 2, 2), dxyz = (100.0, 50.0, 20.0), as_irregular_grid = True)


@pytest.fixture
def example_model_with_properties(tmp_path) -> Model:
    """Model with a grid (5x5x3) and properties.
   Properties:
   - Zone (discrete)
   - VPC (discrete)
   - Fault block (discrete)
   - Facies (discrete)
   - NTG (continuous)
   - POR (continuous)
   - SW (continuous)
   """
    model_path = str(tmp_path / 'test_no_rels.epc')
    model = Model(create_basics = True, create_hdf5_ext = True, epc_file = model_path, new_epc = True)
    model.store_epc(model.epc_file)

    grid = grr.RegularGrid(parent_model = model,
                           origin = (0, 0, 0),
                           extent_kji = (3, 5, 5),
                           crs_uuid = rqet.uuid_for_part_root(model.crs_root),
                           set_points_cached = True,
                           as_irregular_grid = True,
                           dxyz = (100.0, 50.0, 20.0))
    grid.cache_all_geometry_arrays()
    grid.write_hdf5_from_caches(file = model.h5_file_name(file_must_exist = False), mode = 'w')

    grid.create_xml(ext_uuid = model.h5_uuid(),
                    title = 'grid',
                    write_geometry = True,
                    add_cell_length_properties = False)
    model.store_epc()

    model = Model(model_path)

    zone = np.ones(shape = (5, 5))
    zone_array = np.array([zone, zone + 1, zone + 2], dtype = 'int')

    vpc = np.array([[1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [1, 1, 1, 2, 2]])
    vpc_array = np.array([vpc, vpc, vpc])

    facies = np.array([[1, 1, 1, 2, 2], [1, 1, 2, 2, 2], [1, 2, 2, 2, 3], [2, 2, 2, 3, 3], [2, 2, 3, 3, 3]])
    facies_array = np.array([facies, facies, facies])

    fb = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]])
    fb_array = np.array([fb, fb, fb])

    ntg = np.array([[0, 0.5, 0, 0.5, 0], [0.5, 0, 0.5, 0, 0.5], [0, 0.5, 0, 0.5, 0], [0.5, 0, 0.5, 0, 0.5],
                    [0, 0.5, 0, 0.5, 0]])
    ntg_array = np.array([ntg, ntg, ntg])

    por = np.array([[1, 1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5, 0.5],
                    [1, 1, 1, 1, 1]])
    por_array = np.array([por, por, por])

    sat = np.array([[1, 0.5, 1, 0.5, 1], [1, 0.5, 1, 0.5, 1], [1, 0.5, 1, 0.5, 1], [1, 0.5, 1, 0.5, 1],
                    [1, 0.5, 1, 0.5, 1]])
    sat_array = np.array([sat, sat, sat])

    perm = np.array([[1, 10, 10, 100, 100], [1, 10, 10, 100, 100], [1, 10, 10, 100, 100], [1, 10, 10, 100, 100],
                     [1, 10, 10, 100, 100]])
    perm_array = np.array([perm, perm, perm], dtype = 'float')

    collection = rqp.GridPropertyCollection()
    collection.set_grid(grid)
    for array, name, kind, discrete, facet_type, facet in zip(
        [zone_array, vpc_array, fb_array, facies_array, ntg_array, por_array, sat_array, perm_array],
        ['Zone', 'VPC', 'Fault block', 'Facies', 'NTG', 'POR', 'SW', 'Perm'], [
            'discrete', 'discrete', 'discrete', 'discrete', 'net to gross ratio', 'porosity', 'saturation',
            'permeability rock'
        ], [True, True, True, True, False, False, False, False],
        [None, None, None, None, None, None, None, 'direction'], [None, None, None, None, None, None, None, 'I']):
        collection.add_cached_array_to_imported_list(cached_array = array,
                                                     source_info = '',
                                                     keyword = name,
                                                     discrete = discrete,
                                                     uom = None,
                                                     time_index = None,
                                                     null_value = None,
                                                     property_kind = kind,
                                                     facet_type = facet_type,
                                                     facet = facet,
                                                     realization = None)
        collection.write_hdf5_for_imported_list()
        collection.create_xml_for_imported_list_and_add_parts_to_model()
    model.store_epc()

    return model


@pytest.fixture()
def faulted_grid(test_data_path) -> Grid:
    current_filename = os.path.split(getsourcefile(lambda: 0))[0]
    base_folder = os.path.dirname(os.path.dirname(current_filename))
    epc_file = base_folder + '/test_data/wren/wren.epc'
    model = Model(epc_file = epc_file)
    return model.grid(title = 'faulted grid')


@pytest.fixture()
def s_bend_model(tmp_path):
    epc = str(os.path.join(tmp_path, f"{bu.new_uuid()}.epc"))
    return rq.Model(epc_file = epc, new_epc = True, create_basics = True, create_hdf5_ext = True)


@pytest.fixture()
def s_bend_grid(s_bend_model):
    nk = 5
    nj = 12
    ni_tail = 5
    ni_bend = 18
    ni_half_mid = 2
    ni = 2 * (ni_tail + ni_bend + ni_half_mid)

    total_thickness = 12.0
    layer_thickness = total_thickness / float(nk)
    flat_dx_di = 10.0
    horst_dx_dk = 0.25 * flat_dx_di / float(nk)
    dy_dj = 8.0
    top_depth = 100.0

    bend_theta_di = maths.pi / float(ni_bend)
    outer_radius = 2.0 * total_thickness

    bend_a_centre_xz = (flat_dx_di * float(ni_tail), top_depth + outer_radius)
    bend_b_centre_xz = (flat_dx_di * float(ni_tail - 2.0 * ni_half_mid),
                        top_depth + 3.0 * outer_radius - total_thickness)

    points = np.empty((nk + 1, nj + 1, ni + 1, 3))

    for k in range(nk + 1):
        if k == nk // 2 + 1:
            points[k] = points[k - 1]  # pinched out layer
        else:
            for i in range(ni + 1):
                if i < ni_tail + 1:
                    x = flat_dx_di * float(i)
                    z = top_depth + float(k) * layer_thickness  # will introduce a thick layer after pinchout
                elif i < ni_tail + ni_bend:
                    theta = (i - ni_tail) * bend_theta_di
                    radius = outer_radius - float(k) * layer_thickness
                    x = bend_a_centre_xz[0] + radius * maths.sin(theta)
                    z = bend_a_centre_xz[1] - radius * maths.cos(theta)
                elif i < ni_tail + ni_bend + 2 * ni_half_mid + 1:
                    x = flat_dx_di * float(ni_tail - (i - (ni_tail + ni_bend)))
                    z = top_depth + 2.0 * outer_radius - float(k) * layer_thickness
                elif i < ni_tail + 2 * ni_bend + 2 * ni_half_mid:
                    theta = (i - (ni_tail + ni_bend + 2 * ni_half_mid)) * bend_theta_di
                    radius = outer_radius - float(nk - k) * layer_thickness
                    x = bend_b_centre_xz[0] - radius * maths.sin(theta)
                    z = bend_b_centre_xz[1] - radius * maths.cos(theta)
                else:
                    x = flat_dx_di * float((i - (ni - ni_tail)) + ni_tail - 2 * ni_half_mid)
                    if i == ni - 1 or i == ni - 4:
                        x += horst_dx_dk * float(k)
                    elif i == ni - 2 or i == ni - 3:
                        x -= horst_dx_dk * float(k)
                    z = top_depth + 4.0 * outer_radius + float(k) * layer_thickness - 2.0 * total_thickness
                points[k, :, i] = (x, 0.0, z)

    for j in range(nj + 1):
        points[:, j, :, 1] = dy_dj * float(j)

    grid = grr.Grid(s_bend_model)

    crs = rqc.Crs(s_bend_model)
    crs_node = crs.create_xml()

    grid.grid_representation = 'IjkGrid'
    grid.extent_kji = np.array((nk, nj, ni), dtype = 'int')
    grid.nk, grid.nj, grid.ni = nk, nj, ni
    grid.k_direction_is_down = True  # dominant layer direction, or paleo-direction
    grid.pillar_shape = 'straight'
    grid.has_split_coordinate_lines = False
    grid.k_gaps = None
    grid.crs_uuid = crs.uuid
    grid.crs_root = crs_node

    grid.points_cached = points

    grid.geometry_defined_for_all_pillars_cached = True
    grid.geometry_defined_for_all_cells_cached = True
    grid.grid_is_right_handed = crs.is_right_handed_xyz()

    # grid.write_hdf5_from_caches()
    # grid.create_xml()
    return grid


@pytest.fixture()
def s_bend_faulted_grid(s_bend_model):
    nk = 5
    nj = 12
    ni_tail = 5
    ni_bend = 18
    ni_half_mid = 2
    ni = 2 * (ni_tail + ni_bend + ni_half_mid)

    total_thickness = 12.0
    layer_thickness = total_thickness / float(nk)
    flat_dx_di = 10.0
    horst_dx_dk = 0.25 * flat_dx_di / float(nk)
    horst_dz = 1.73 * layer_thickness
    horst_half_dx = horst_dx_dk * horst_dz / layer_thickness
    dy_dj = 8.0
    top_depth = 100.0

    bend_theta_di = maths.pi / float(ni_bend)
    outer_radius = 2.0 * total_thickness

    bend_a_centre_xz = (flat_dx_di * float(ni_tail), top_depth + outer_radius)
    bend_b_centre_xz = (flat_dx_di * float(ni_tail - 2.0 * ni_half_mid),
                        top_depth + 3.0 * outer_radius - total_thickness)

    points = np.empty((nk + 1, nj + 1, ni + 1, 3))

    for k in range(nk + 1):
        if k == nk // 2 + 1:
            points[k] = points[k - 1]  # pinched out layer
        else:
            for i in range(ni + 1):
                if i < ni_tail + 1:
                    x = flat_dx_di * float(i)
                    z = top_depth + float(k) * layer_thickness  # will introduce a thick layer after pinchout
                elif i < ni_tail + ni_bend:
                    theta = (i - ni_tail) * bend_theta_di
                    radius = outer_radius - float(k) * layer_thickness
                    x = bend_a_centre_xz[0] + radius * maths.sin(theta)
                    z = bend_a_centre_xz[1] - radius * maths.cos(theta)
                elif i < ni_tail + ni_bend + 2 * ni_half_mid + 1:
                    x = flat_dx_di * float(ni_tail - (i - (ni_tail + ni_bend)))
                    z = top_depth + 2.0 * outer_radius - float(k) * layer_thickness
                elif i < ni_tail + 2 * ni_bend + 2 * ni_half_mid:
                    theta = (i - (ni_tail + ni_bend + 2 * ni_half_mid)) * bend_theta_di
                    radius = outer_radius - float(nk - k) * layer_thickness
                    x = bend_b_centre_xz[0] - radius * maths.sin(theta)
                    z = bend_b_centre_xz[1] - radius * maths.cos(theta)
                else:
                    x = flat_dx_di * float((i - (ni - ni_tail)) + ni_tail - 2 * ni_half_mid)
                    if i == ni - 1 or i == ni - 4:
                        x += horst_dx_dk * float(k)
                    elif i == ni - 2 or i == ni - 3:
                        x -= horst_dx_dk * float(k)
                    z = top_depth + 4.0 * outer_radius + float(k) * layer_thickness - 2.0 * total_thickness
                points[k, :, i] = (x, 0.0, z)

    for j in range(nj + 1):
        points[:, j, :, 1] = dy_dj * float(j)

    grid = grr.Grid(s_bend_model)

    crs = rqc.Crs(s_bend_model)
    crs_node = crs.create_xml()

    grid.grid_representation = 'IjkGrid'
    grid.extent_kji = np.array((nk, nj, ni), dtype = 'int')
    grid.nk, grid.nj, grid.ni = nk, nj, ni
    grid.k_direction_is_down = True  # dominant layer direction, or paleo-direction
    grid.pillar_shape = 'straight'
    grid.has_split_coordinate_lines = False
    grid.k_gaps = None
    grid.crs_uuid = crs.uuid
    grid.crs_root = crs_node

    grid.points_cached = points

    grid.geometry_defined_for_all_pillars_cached = True
    grid.geometry_defined_for_all_cells_cached = True
    grid.grid_is_right_handed = crs.is_right_handed_xyz()
    cp = grid.corner_points(cache_cp_array = True).copy()

    bend_theta_di = maths.pi / float(ni_bend)

    # IK plane faults
    cp[:, 3:, :, :, :, :, :] += (flat_dx_di * 0.7, 0.0, layer_thickness * 1.3)
    cp[:, 5:, :, :, :, :, :] += (flat_dx_di * 0.4, 0.0, layer_thickness * 0.9)
    cp[:, 8:, :, :, :, :, :] += (flat_dx_di * 0.3, 0.0, layer_thickness * 0.6)

    # JK plane faults
    cp[:, :, ni_tail + ni_bend // 2:, :, :, :, 0] -= flat_dx_di * 0.57  # horizontal break mid top bend
    cp[:, :, ni_tail + ni_bend + ni_half_mid:, :, :, :, 2] += layer_thickness * 1.27  # vertical break in mid section

    # zig-zag fault
    j_step = nj // (ni_tail - 2)
    for i in range(ni_tail - 1):
        j_start = i * j_step
        if j_start >= nj:
            break
        cp[:, j_start:, i, :, :, :, 2] += 1.1 * total_thickness

    # JK horst blocks
    cp[:, :, ni - 4, :, :, :, :] -= (horst_half_dx, 0.0, horst_dz)
    cp[:, :, ni - 3:, :, :, :, 0] -= 2.0 * horst_half_dx
    cp[:, :, ni - 2, :, :, :, :] += (-horst_half_dx, 0.0, horst_dz)
    cp[:, :, ni - 1:, :, :, :, 0] -= 2.0 * horst_half_dx

    # JK horst block mid lower bend
    bend_horst_dz = horst_dz * maths.tan(bend_theta_di)
    cp[:, :, ni - (ni_tail + ni_bend // 2 + 1):ni - (ni_tail + ni_bend // 2 - 1), :, :, :, :] -= (horst_dz, 0.0,
                                                                                                  bend_horst_dz)
    cp[:, :, ni - (ni_tail + ni_bend // 2 - 1):, :, :, :, 2] -= 2.0 * bend_horst_dz

    faulted_grid = rqi.grid_from_cp(s_bend_model,
                                    cp,
                                    crs.uuid,
                                    max_z_void = 0.01,
                                    split_pillars = True,
                                    split_tolerance = 0.01,
                                    ijk_handedness = 'right' if grid.grid_is_right_handed else 'left',
                                    known_to_be_straight = True)

    # faulted_grid.write_hdf5_from_caches()
    # faulted_grid.create_xml()
    return faulted_grid


@pytest.fixture()
def s_bend_k_gap_grid(s_bend_model):
    nk = 5
    nj = 12
    ni_tail = 5
    ni_bend = 18
    ni_half_mid = 2
    ni = 2 * (ni_tail + ni_bend + ni_half_mid)

    total_thickness = 12.0
    layer_thickness = total_thickness / float(nk)
    flat_dx_di = 10.0
    horst_dx_dk = 0.25 * flat_dx_di / float(nk)
    horst_dz = 1.73 * layer_thickness
    horst_half_dx = horst_dx_dk * horst_dz / layer_thickness
    dy_dj = 8.0
    top_depth = 100.0

    bend_theta_di = maths.pi / float(ni_bend)
    outer_radius = 2.0 * total_thickness

    bend_a_centre_xz = (flat_dx_di * float(ni_tail), top_depth + outer_radius)
    bend_b_centre_xz = (flat_dx_di * float(ni_tail - 2.0 * ni_half_mid),
                        top_depth + 3.0 * outer_radius - total_thickness)

    points = np.empty((nk + 1, nj + 1, ni + 1, 3))

    for k in range(nk + 1):
        if k == nk // 2 + 1:
            points[k] = points[k - 1]  # pinched out layer
        else:
            for i in range(ni + 1):
                if i < ni_tail + 1:
                    x = flat_dx_di * float(i)
                    z = top_depth + float(k) * layer_thickness  # will introduce a thick layer after pinchout
                elif i < ni_tail + ni_bend:
                    theta = (i - ni_tail) * bend_theta_di
                    radius = outer_radius - float(k) * layer_thickness
                    x = bend_a_centre_xz[0] + radius * maths.sin(theta)
                    z = bend_a_centre_xz[1] - radius * maths.cos(theta)
                elif i < ni_tail + ni_bend + 2 * ni_half_mid + 1:
                    x = flat_dx_di * float(ni_tail - (i - (ni_tail + ni_bend)))
                    z = top_depth + 2.0 * outer_radius - float(k) * layer_thickness
                elif i < ni_tail + 2 * ni_bend + 2 * ni_half_mid:
                    theta = (i - (ni_tail + ni_bend + 2 * ni_half_mid)) * bend_theta_di
                    radius = outer_radius - float(nk - k) * layer_thickness
                    x = bend_b_centre_xz[0] - radius * maths.sin(theta)
                    z = bend_b_centre_xz[1] - radius * maths.cos(theta)
                else:
                    x = flat_dx_di * float((i - (ni - ni_tail)) + ni_tail - 2 * ni_half_mid)
                    if i == ni - 1 or i == ni - 4:
                        x += horst_dx_dk * float(k)
                    elif i == ni - 2 or i == ni - 3:
                        x -= horst_dx_dk * float(k)
                    z = top_depth + 4.0 * outer_radius + float(k) * layer_thickness - 2.0 * total_thickness
                points[k, :, i] = (x, 0.0, z)

    for j in range(nj + 1):
        points[:, j, :, 1] = dy_dj * float(j)

    grid = grr.Grid(s_bend_model)

    crs = rqc.Crs(s_bend_model)
    crs_node = crs.create_xml()

    grid.grid_representation = 'IjkGrid'
    grid.extent_kji = np.array((nk, nj, ni), dtype = 'int')
    grid.nk, grid.nj, grid.ni = nk, nj, ni
    grid.k_direction_is_down = True  # dominant layer direction, or paleo-direction
    grid.pillar_shape = 'straight'
    grid.has_split_coordinate_lines = False
    grid.k_gaps = None
    grid.crs_uuid = crs.uuid
    grid.crs_root = crs_node

    grid.points_cached = points

    grid.geometry_defined_for_all_pillars_cached = True
    grid.geometry_defined_for_all_cells_cached = True
    grid.grid_is_right_handed = crs.is_right_handed_xyz()
    cp = grid.corner_points(cache_cp_array = True).copy()

    bend_theta_di = maths.pi / float(ni_bend)

    # IK plane faults
    cp[:, 3:, :, :, :, :, :] += (flat_dx_di * 0.7, 0.0, layer_thickness * 1.3)
    cp[:, 5:, :, :, :, :, :] += (flat_dx_di * 0.4, 0.0, layer_thickness * 0.9)
    cp[:, 8:, :, :, :, :, :] += (flat_dx_di * 0.3, 0.0, layer_thickness * 0.6)

    # JK plane faults
    cp[:, :, ni_tail + ni_bend // 2:, :, :, :, 0] -= flat_dx_di * 0.57  # horizontal break mid top bend
    cp[:, :, ni_tail + ni_bend + ni_half_mid:, :, :, :, 2] += layer_thickness * 1.27  # vertical break in mid section

    # zig-zag fault
    j_step = nj // (ni_tail - 2)
    for i in range(ni_tail - 1):
        j_start = i * j_step
        if j_start >= nj:
            break
        cp[:, j_start:, i, :, :, :, 2] += 1.1 * total_thickness

    # JK horst blocks
    cp[:, :, ni - 4, :, :, :, :] -= (horst_half_dx, 0.0, horst_dz)
    cp[:, :, ni - 3:, :, :, :, 0] -= 2.0 * horst_half_dx
    cp[:, :, ni - 2, :, :, :, :] += (-horst_half_dx, 0.0, horst_dz)
    cp[:, :, ni - 1:, :, :, :, 0] -= 2.0 * horst_half_dx

    # JK horst block mid lower bend
    bend_horst_dz = horst_dz * maths.tan(bend_theta_di)
    cp[:, :, ni - (ni_tail + ni_bend // 2 + 1):ni - (ni_tail + ni_bend // 2 - 1), :, :, :, :] -= (horst_dz, 0.0,
                                                                                                  bend_horst_dz)
    cp[:, :, ni - (ni_tail + ni_bend // 2 - 1):, :, :, :, 2] -= 2.0 * bend_horst_dz

    k_gap_grid = rqi.grid_from_cp(s_bend_model,
                                  cp,
                                  crs.uuid,
                                  max_z_void = 0.01,
                                  split_pillars = True,
                                  split_tolerance = 0.01,
                                  ijk_handedness = 'right' if grid.grid_is_right_handed else 'left',
                                  known_to_be_straight = True)

    # convert second layer to a K gap
    k_gap_grid.nk -= 1
    k_gap_grid.extent_kji[0] = k_gap_grid.nk
    k_gap_grid.k_gaps = 1
    k_gap_grid.k_gap_after_array = np.zeros(k_gap_grid.nk - 1, dtype = bool)
    k_gap_grid.k_gap_after_array[0] = True
    k_gap_grid.k_raw_index_array = np.zeros(k_gap_grid.nk, dtype = int)
    for k in range(1, k_gap_grid.nk):
        k_gap_grid.k_raw_index_array[k] = k + 1

    # clear some attributes which may no longer be valid
    k_gap_grid.pinchout = None
    k_gap_grid.inactive = None
    k_gap_grid.grid_skin = None
    if hasattr(k_gap_grid, 'array_thickness'):
        delattr(k_gap_grid, 'array_thickness')

    # k_gap_grid.write_hdf5_from_caches()
    # k_gap_grid.create_xml()
    return k_gap_grid
