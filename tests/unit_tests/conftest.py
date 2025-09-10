""" Shared fixtures for tests """

import logging
from pathlib import Path
from shutil import copytree
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import pytest

import resqpy.grid as grr
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp
from resqpy.crs import Crs
from resqpy.model import Model, new_model
from resqpy.organize import WellboreFeature, WellboreInterpretation
from resqpy.well import MdDatum, Trajectory, WellboreFrame
import resqpy.time_series as rqts
import resqpy.olio.fine_coarse as rqfc
import resqpy.olio.triangulation as tri
import resqpy.surface as rqs


@pytest.fixture(autouse = True)
def capture_logs(caplog):
    """Always capture log messages from respy"""

    caplog.set_level(logging.DEBUG, logger = "resqpy")


@pytest.fixture
def tmp_model(tmp_path):
    """Example resqpy model in a temporary directory unique to each test"""

    return new_model(str(tmp_path / 'tmp_model.epc'))


@pytest.fixture
def example_model_and_crs(tmp_model):
    """ Returns a fresh RESQML Model and Crs, in a temporary directory """

    # TODO: parameterise with m or feet
    xyz_uom = 'm'

    # Create a model with a coordinate reference system
    crs = Crs(parent_model = tmp_model, z_inc_down = True, xy_units = xyz_uom, z_units = xyz_uom)
    crs.create_xml()

    return tmp_model, crs


@pytest.fixture
def example_model_and_mixed_units_crs(tmp_model):
    """Returns a fresh RESQML Model and mixed units Crs, in a temporary directory."""

    xy_uom = 'm'
    z_uom = 'ft'

    # Create a model with a coordinate reference system
    crs = Crs(parent_model = tmp_model, z_inc_down = True, xy_units = xy_uom, z_units = z_uom)
    crs.create_xml()

    return tmp_model, crs


@pytest.fixture
def example_model_with_well(example_model_and_crs):
    """ Model with a single well with a vertical trajectory """

    wellname = 'well A'
    elevation = 100
    md_uom = 'm'

    model, crs = example_model_and_crs

    # Create a single well feature and interpretation
    well_feature = WellboreFeature(parent_model = model, feature_name = wellname)
    well_feature.create_xml()
    well_interp = WellboreInterpretation(parent_model = model, wellbore_feature = well_feature, is_drilled = True)
    well_interp.create_xml()

    # Create a measured depth datum
    datum = MdDatum(parent_model = model,
                    crs_uuid = crs.uuid,
                    location = (0, 0, -elevation),
                    md_reference = 'kelly bushing')
    datum.create_xml()

    # Create trajectory of a vertical well
    mds = np.array([0, 1000, 2000])
    zs = mds - elevation
    traj = Trajectory(parent_model = model,
                      md_datum = datum,
                      data_frame = pd.DataFrame(dict(MD = mds, X = 0, Y = 0, Z = zs)),
                      length_uom = md_uom,
                      represented_interp = well_interp)
    traj.write_hdf5(mode = 'w')
    traj.create_xml()

    return model, well_interp, datum, traj


@pytest.fixture
def example_model_with_logs(example_model_with_well):
    model, well_interp, datum, traj = example_model_with_well

    frame = WellboreFrame(
        model,
        trajectory = traj,
        represented_interp = well_interp,
        mds = [1, 2, 3, 4],
    )
    frame.write_hdf5()
    frame.create_xml(title = 'Log run A')

    log_collection = frame.extract_log_collection()
    log_collection.add_log("GR", [1, 2, 1, 2], 'gAPI')
    log_collection.add_log("NPHI", [0.1, 0.1, np.nan, np.nan], 'v/v')

    return model, well_interp, datum, traj, frame, log_collection


@pytest.fixture
def test_data_path(tmp_path):
    """ Return pathlib.Path pointing to temporary copy of tests/example_data

   Use a fresh temporary directory for each test.
   """
    master_path = (Path(__file__) / '../../test_data').resolve()
    data_path = os.path.join(tmp_path, 'test_data')

    assert master_path.exists()
    assert not os.path.exists(data_path)
    copytree(str(master_path), data_path)
    return data_path


@pytest.fixture
def example_model_with_properties(tmp_path):
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
                           set_points_cached = True)
    grid.cache_all_geometry_arrays()
    grid.write_hdf5_from_caches(file = model.h5_file_name(file_must_exist = False), mode = 'w')

    grid.create_xml(ext_uuid = model.h5_uuid(),
                    title = 'grid',
                    write_geometry = True,
                    add_cell_length_properties = False,
                    use_lattice = False)
    model.store_epc()

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
    perm_v_array = perm_array * 0.1

    collection = rqp.GridPropertyCollection()
    collection.set_grid(grid)
    for array, name, kind, discrete, facet_type, facet in zip(
        [zone_array, vpc_array, fb_array, facies_array, ntg_array, por_array, sat_array, perm_array, perm_v_array],
        ['Zone', 'VPC', 'Fault block', 'Facies', 'NTG', 'POR', 'SW', 'Perm', 'PERMZ'], [
            'zone', 'vpc', 'fault block', 'facies', 'net to gross ratio', 'porosity', 'saturation', 'rock permeability',
            'permeability rock'
        ], [True, True, True, True, False, False, False, False, False],
        [None, None, None, None, None, None, 'what', 'direction', 'direction'],
        [None, None, None, None, None, None, 'water', 'I', 'K']):
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


def model_with_prop_ts_rels(model_path):
    """Model with a grid (5x5x3) and properties, including a time series and string lookup.

    Properties:
    - Zone (discrete)
    - VPC (discrete)
    - Fault block (discrete)
    - Facies (discrete)
    - NTG (continuous)
    - POR (continuous)
    - SW (continuous) (recurrent)
    """
    model = Model(create_basics = True, create_hdf5_ext = True, epc_file = model_path, new_epc = True)
    model.store_epc(model.epc_file)

    grid = grr.RegularGrid(parent_model = model,
                           origin = (0, 0, 0),
                           extent_kji = (3, 5, 5),
                           crs_uuid = rqet.uuid_for_part_root(model.crs_root),
                           set_points_cached = True)
    grid.cache_all_geometry_arrays()
    grid.write_hdf5_from_caches(file = model.h5_file_name(file_must_exist = False), mode = 'w')

    grid.create_xml(ext_uuid = model.h5_uuid(),
                    title = 'grid',
                    write_geometry = True,
                    add_cell_length_properties = False,
                    use_lattice = False)
    model.store_epc()

    zone = np.ones(shape = (5, 5), dtype = 'int')
    zone_array = np.array([zone, zone + 1, zone + 2], dtype = 'int')

    vpc = np.array([[1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [1, 1, 1, 2, 2]], dtype = 'int')
    vpc_array = np.array([vpc, vpc, vpc], dtype = 'int')

    facies = np.array([[1, 1, 1, 2, 2], [1, 1, 2, 2, 2], [1, 2, 2, 2, 3], [2, 2, 2, 3, 3], [2, 2, 3, 3, 3]],
                      dtype = 'int')
    facies_array = np.array([facies, facies, facies], dtype = 'int')

    perm = np.array([[1, 1, 1, 10, 10], [1, 1, 1, 10, 10], [1, 1, 1, 10, 10], [1, 1, 1, 10, 10], [1, 1, 1, 10, 10]])
    perm_array = np.array([perm, perm, perm], dtype = 'float')

    fb = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]], dtype = 'int')
    fb_array = np.array([fb, fb, fb], dtype = 'int')

    ntg = np.array([[0, 0.5, 0, 0.5, 0], [0.5, 0, 0.5, 0, 0.5], [0, 0.5, 0, 0.5, 0], [0.5, 0, 0.5, 0, 0.5],
                    [0, 0.5, 0, 0.5, 0]])
    ntg1_array = np.array([ntg, ntg, ntg])
    ntg2_array = np.array([ntg + 0.1, ntg + 0.1, ntg + 0.1])

    por = np.array([[1, 1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5, 0.5],
                    [1, 1, 1, 1, 1]])
    por1_array = np.array([por, por, por])
    por2_array = np.array([por - 0.1, por - 0.1, por - 0.1])

    sat = np.array([[1, 0.5, 1, 0.5, 1], [1, 0.5, 1, 0.5, 1], [1, 0.5, 1, 0.5, 1], [1, 0.5, 1, 0.5, 1],
                    [1, 0.5, 1, 0.5, 1]])
    sat1_array = np.array([sat, sat, sat])
    sat2_array = np.array([sat, sat, np.where(sat == 0.5, 0.75, sat)])
    sat3_array = np.array(
        [np.where(sat == 0.5, 0.75, sat),
         np.where(sat == 0.5, 0.75, sat),
         np.where(sat == 0.5, 0.75, sat)])

    collection = rqp.GridPropertyCollection()
    collection.set_grid(grid)

    ts = rqts.TimeSeries(parent_model = model, first_timestamp = '2000-01-01Z')
    ts.extend_by_days(365)
    ts.extend_by_days(365)

    ts.create_xml()

    lookup = rqp.StringLookup(parent_model = model, int_to_str_dict = {1: 'channel', 2: 'interbedded', 3: 'shale'})
    lookup.create_xml()

    model.store_epc()

    # Add non-varying properties
    for array, name, kind, discrete, facet_type, facet in zip([zone_array, vpc_array, fb_array, perm_array],
                                                              ['Zone', 'VPC', 'Fault block', 'Perm'],
                                                              ['zone', 'vpc', 'fault block', 'permeability rock'],
                                                              [True, True, True, False],
                                                              [None, None, None, 'direction'], [None, None, None, 'J']):
        collection.add_cached_array_to_imported_list(cached_array = array,
                                                     source_info = 'test model with time series',
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

    # Add realisation varying properties
    for array, name, kind, rel in zip([ntg1_array, por1_array, ntg2_array, por2_array], ['NTG', 'POR', 'NTG', 'POR'],
                                      ['net to gross ratio', 'porosity', 'net to gross ratio', 'porosity'],
                                      [0, 0, 1, 1]):
        collection.add_cached_array_to_imported_list(cached_array = array,
                                                     source_info = '',
                                                     keyword = name,
                                                     discrete = False,
                                                     uom = None,
                                                     time_index = None,
                                                     null_value = None,
                                                     property_kind = kind,
                                                     facet_type = None,
                                                     facet = None,
                                                     realization = rel)
        collection.write_hdf5_for_imported_list()
        collection.create_xml_for_imported_list_and_add_parts_to_model()

    # Add categorial property
    collection.add_cached_array_to_imported_list(cached_array = facies_array,
                                                 source_info = '',
                                                 keyword = 'Facies',
                                                 discrete = True,
                                                 uom = None,
                                                 time_index = None,
                                                 null_value = None,
                                                 property_kind = 'facies',
                                                 facet_type = None,
                                                 facet = None,
                                                 realization = None)
    collection.write_hdf5_for_imported_list()
    collection.create_xml_for_imported_list_and_add_parts_to_model(string_lookup_uuid = lookup.uuid)

    # Add time varying properties
    for array, ts_index in zip([sat1_array, sat2_array, sat3_array], [0, 1, 2]):
        collection.add_cached_array_to_imported_list(cached_array = array,
                                                     source_info = '',
                                                     keyword = 'SW',
                                                     discrete = False,
                                                     uom = None,
                                                     time_index = ts_index,
                                                     null_value = None,
                                                     property_kind = 'saturation',
                                                     facet_type = 'what',
                                                     facet = 'water',
                                                     realization = None)
        collection.write_hdf5_for_imported_list()
        collection.create_xml_for_imported_list_and_add_parts_to_model(time_series_uuid = ts.uuid)
    model.store_epc()

    return model


@pytest.fixture
def example_model_with_prop_ts_rels(tmp_path):
    """Model with a grid (5x5x3) and properties, including a time series and string lookup.
    Properties:
    - Zone (discrete)
    - VPC (discrete)
    - Fault block (discrete)
    - Facies (discrete)
    - NTG (continuous)
    - POR (continuous)
    - SW (continuous) (recurrent)
    """
    epc = os.path.join(tmp_path, 'test_model.epc')
    return model_with_prop_ts_rels(epc)


@pytest.fixture
def pair_of_models_with_prop_ts_rels(tmp_path):
    """Model with a grid (5x5x3) and properties, including a time series and string lookup.
    Properties:
    - Zone (discrete)
    - VPC (discrete)
    - Fault block (discrete)
    - Facies (discrete)
    - NTG (continuous)
    - POR (continuous)
    - SW (continuous) (recurrent)
    """
    epc1 = os.path.join(tmp_path, 'test_model_1.epc')
    epc2 = os.path.join(tmp_path, 'test_model_2.epc')
    m1 = model_with_prop_ts_rels(epc1)
    m2 = model_with_prop_ts_rels(epc2)
    return (m1, m2)


@pytest.fixture
def example_model_and_cellio(example_model_and_crs, tmp_path):
    model, crs = example_model_and_crs
    grid = grr.RegularGrid(model,
                           extent_kji = (3, 4, 3),
                           dxyz = (50.0, -50.0, 50.0),
                           origin = (0.0, 0.0, 100.0),
                           crs_uuid = crs.uuid,
                           set_points_cached = True)

    grid.write_hdf5()
    grid.create_xml(write_geometry = True, use_lattice = False)
    grid_uuid = grid.uuid
    cellio_file = os.path.join(model.epc_directory, 'cellio.dat')
    well_name = 'Banoffee'
    source_df = pd.DataFrame(
        [[2, 2, 1, 25, -25, 125, 26, -26, 126], [2, 2, 2, 26, -26, 126, 27, -27, 127],
         [2, 2, 3, 27, -27, 127, 28, -28, 128]],
        columns = ['i_index', 'j_index', 'k_index', 'x_in', 'y_in', 'z_in', 'x_out', 'y_out', 'z_out'])

    with open(cellio_file, 'w') as fp:
        fp.write('1.0\n')
        fp.write('Undefined\n')
        fp.write(f'{well_name}\n')
        fp.write('9\n')
        for col in source_df.columns:
            fp.write(f' {col}\n')
        for row_index in range(len(source_df)):
            row = source_df.iloc[row_index]
            for col in source_df.columns:
                fp.write(f' {int(row[col])}')
            fp.write('\n')

    return model, cellio_file, well_name


@pytest.fixture
def example_fine_coarse_model(example_model_and_crs):
    model, crs = example_model_and_crs

    coarse_grid = grr.RegularGrid(parent_model = model,
                                  origin = (0, 0, 0),
                                  extent_kji = (3, 5, 5),
                                  crs_uuid = crs.uuid,
                                  dxyz = (10, 10, 10))
    coarse_grid.cache_all_geometry_arrays()
    coarse_grid.write_hdf5_from_caches(file = model.h5_file_name(file_must_exist = False), mode = 'w')
    coarse_grid.create_xml(ext_uuid = model.h5_uuid(),
                           title = 'Coarse',
                           write_geometry = True,
                           add_cell_length_properties = True,
                           use_lattice = False)

    fine_grid = grr.RegularGrid(parent_model = model,
                                origin = (0, 0, 0),
                                extent_kji = (6, 10, 10),
                                crs_uuid = crs.uuid,
                                dxyz = (5, 5, 5))
    fine_grid.cache_all_geometry_arrays()
    fine_grid.write_hdf5_from_caches(file = model.h5_file_name(file_must_exist = True), mode = 'a')
    fine_grid.create_xml(ext_uuid = model.h5_uuid(),
                         title = 'Fine',
                         write_geometry = True,
                         add_cell_length_properties = True,
                         use_lattice = False)

    model.store_epc()
    model = Model(model.epc_file)

    coarse = grr.Grid(parent_model = model, uuid = coarse_grid.uuid)
    fine = grr.Grid(parent_model = model, uuid = fine_grid.uuid)

    fc = rqfc.FineCoarse(fine_extent_kji = (6, 10, 10), coarse_extent_kji = (3, 5, 5))
    fc.set_all_ratios_constant()
    fc.set_all_proportions_equal()

    return model, coarse, fine, fc


@pytest.fixture
def small_grid_and_surface(tmp_model: Model) -> Tuple[grr.RegularGrid, rqs.Surface]:
    """Creates a small RegularGrid and a random triangular surface."""
    crs = Crs(tmp_model)
    crs.create_xml()

    extent = 10
    extent_kji = (extent, extent, extent)
    dxyz = (1.0, 1.0, 1.0)
    crs_uuid = crs.uuid
    title = "small_grid"
    grid = grr.RegularGrid(tmp_model, extent_kji = extent_kji, dxyz = dxyz, crs_uuid = crs_uuid, title = title)
    grid.create_xml()

    n_points = 100
    points = np.random.rand(n_points, 3) * extent
    triangles = tri.dt(points)
    surface = rqs.Surface(tmp_model, crs_uuid = crs_uuid, title = "small_surface")
    surface.set_from_triangles_and_points(triangles, points)
    surface.triangles_and_points()
    surface.write_hdf5()
    surface.create_xml()

    tmp_model.store_epc()

    return grid, surface


@pytest.fixture
def small_grid_and_surface_no_k(tmp_model: Model) -> Tuple[grr.RegularGrid, rqs.Surface]:
    """Creates a small RegularGrid and a curtain triangular surface."""
    crs = Crs(tmp_model)
    crs.create_xml()

    extent = 10
    extent_kji = (extent, extent, extent)
    dxyz = (1.0, 1.0, 1.0)
    crs_uuid = crs.uuid
    title = "small_grid"
    grid = grr.RegularGrid(tmp_model, extent_kji = extent_kji, dxyz = dxyz, crs_uuid = crs_uuid, title = title)
    grid.create_xml()

    points = np.array(([0.37, -0.043, -0.017], [0.37, -0.043, 1.021], [0.61, 1.003, -0.027], [0.61, 1.003, 1.081]),
                      dtype = float) * extent
    triangles = np.array(([0, 1, 2], [1, 2, 3]), dtype = np.int32)
    surface = rqs.Surface(tmp_model, crs_uuid = crs_uuid, title = "small_curtain")
    surface.set_from_triangles_and_points(triangles, points)
    surface.triangles_and_points()
    surface.write_hdf5()
    surface.create_xml()

    tmp_model.store_epc()

    return grid, surface


@pytest.fixture
def small_grid_and_i_curtain_surface(tmp_model: Model) -> Tuple[grr.RegularGrid, rqs.Surface]:
    """Creates a small RegularGrid and a curtain triangular surface."""
    crs = Crs(tmp_model)
    crs.create_xml()

    extent = 10
    extent_kji = (extent, extent, extent)
    dxyz = (1.0, 1.0, 1.0)
    crs_uuid = crs.uuid
    title = "small_grid"
    grid = grr.RegularGrid(tmp_model, extent_kji = extent_kji, dxyz = dxyz, crs_uuid = crs_uuid, title = title)
    grid.create_xml()

    points = np.array(([0.47, -0.043, -0.017], [0.47, -0.043, 1.021], [0.47, 1.003, -0.027], [0.47, 1.003, 1.081]),
                      dtype = float) * extent
    triangles = np.array(([0, 1, 2], [1, 2, 3]), dtype = np.int32)
    surface = rqs.Surface(tmp_model, crs_uuid = crs_uuid, title = "small_curtain")
    surface.set_from_triangles_and_points(triangles, points)
    surface.triangles_and_points()
    surface.write_hdf5()
    surface.create_xml()

    tmp_model.store_epc()

    return grid, surface


@pytest.fixture
def small_grid_and_j_curtain_surface(tmp_model: Model) -> Tuple[grr.RegularGrid, rqs.Surface]:
    """Creates a small RegularGrid and a curtain triangular surface."""
    crs = Crs(tmp_model)
    crs.create_xml()

    extent = 10
    extent_kji = (extent, extent, extent)
    dxyz = (1.0, 1.0, 1.0)
    crs_uuid = crs.uuid
    title = "small_grid"
    grid = grr.RegularGrid(tmp_model, extent_kji = extent_kji, dxyz = dxyz, crs_uuid = crs_uuid, title = title)
    grid.create_xml()

    points = np.array(([-0.043, 0.59, -0.017], [-0.043, 0.59, 1.021], [1.003, 0.59, -0.027], [1.003, 0.59, 1.081]),
                      dtype = float) * extent
    triangles = np.array(([0, 1, 2], [1, 2, 3]), dtype = np.int32)
    surface = rqs.Surface(tmp_model, crs_uuid = crs_uuid, title = "small_curtain")
    surface.set_from_triangles_and_points(triangles, points)
    surface.triangles_and_points()
    surface.write_hdf5()
    surface.create_xml()

    tmp_model.store_epc()

    return grid, surface


@pytest.fixture
def small_grid_and_missing_surface(tmp_model: Model) -> Tuple[grr.RegularGrid, rqs.Surface]:
    """Creates a small RegularGrid and a triangular surface that does not intersect the grid."""
    crs = Crs(tmp_model)
    crs.create_xml()

    extent = 10
    extent_kji = (extent, extent, extent)
    dxyz = (1.0, 1.0, 1.0)
    crs_uuid = crs.uuid
    title = "small_grid"
    grid = grr.RegularGrid(tmp_model, extent_kji = extent_kji, dxyz = dxyz, crs_uuid = crs_uuid, title = title)
    grid.create_xml()

    points = np.array(([0.5, 0.0, 3.0], [0.0, 0.5, 3.0], [-3.0, 0.0, 0.0], [0.0, -3.0, 0.0]), dtype = float) * extent
    triangles = np.array(([0, 1, 2], [1, 2, 3]), dtype = np.int32)
    surface = rqs.Surface(tmp_model, crs_uuid = crs_uuid, title = "small_missing")
    surface.set_from_triangles_and_points(triangles, points)
    surface.triangles_and_points()
    surface.write_hdf5()
    surface.create_xml()

    tmp_model.store_epc()

    return grid, surface


@pytest.fixture
def small_grid_and_surface_no_j(tmp_model: Model) -> Tuple[grr.RegularGrid, rqs.Surface]:
    """Creates a small RegularGrid and tiny triangular surface which will map to no J faces."""
    crs = Crs(tmp_model)
    crs.create_xml()

    extent = 10
    extent_kji = (extent, extent, extent)
    dxyz = (1.0, 1.0, 1.0)
    crs_uuid = crs.uuid
    title = "small_grid"
    grid = grr.RegularGrid(tmp_model, extent_kji = extent_kji, dxyz = dxyz, crs_uuid = crs_uuid, title = title)
    grid.create_xml()

    n_points = 100
    points = np.array(([0.37, 0.017, 0.043], [0.37, 0.921, 0.043], [0.61, 0.027, 0.903], [0.61, 0.981, 0.903]),
                      dtype = float) * extent
    triangles = np.array(([0, 1, 2], [1, 2, 3]), dtype = np.int32)
    surface = rqs.Surface(tmp_model, crs_uuid = crs_uuid, title = "small_surface_no_j")
    surface.set_from_triangles_and_points(triangles, points)
    surface.triangles_and_points()
    surface.write_hdf5()
    surface.create_xml()

    tmp_model.store_epc()

    return grid, surface


@pytest.fixture
def small_grid_and_surface_no_i(tmp_model: Model) -> Tuple[grr.RegularGrid, rqs.Surface]:
    """Creates a small RegularGrid and tiny triangular surface which will map to no I faces."""
    crs = Crs(tmp_model)
    crs.create_xml()

    extent = 10
    extent_kji = (extent, extent, extent)
    dxyz = (1.0, 1.0, 1.0)
    crs_uuid = crs.uuid
    title = "small_grid"
    grid = grr.RegularGrid(tmp_model, extent_kji = extent_kji, dxyz = dxyz, crs_uuid = crs_uuid, title = title)
    grid.create_xml()

    n_points = 100
    points = np.array(([0.017, 0.37, 0.043], [0.921, 0.37, 0.043], [0.027, 0.61, 0.903], [0.981, 0.61, 0.903]),
                      dtype = float) * extent
    triangles = np.array(([0, 1, 2], [1, 2, 3]), dtype = np.int32)
    surface = rqs.Surface(tmp_model, crs_uuid = crs_uuid, title = "small_surface_no_j")
    surface.set_from_triangles_and_points(triangles, points)
    surface.triangles_and_points()
    surface.write_hdf5()
    surface.create_xml()

    tmp_model.store_epc()

    return grid, surface


@pytest.fixture
def small_grid_and_extended_surface(tmp_model: Model) -> Tuple[grr.RegularGrid, rqs.Surface]:
    """Creates a small RegularGrid and a random triangular surface extended with a flange."""
    crs = Crs(tmp_model)
    crs.create_xml()

    extent = 10
    extent_kji = (extent, extent + 1, extent + 2)
    dxyz = (1.0, 1.0, 1.0)
    crs_uuid = crs.uuid
    title = "small_grid"
    grid = grr.RegularGrid(tmp_model, extent_kji = extent_kji, dxyz = dxyz, crs_uuid = crs_uuid, title = title)
    grid.create_xml()

    n_points = 100
    points = np.random.rand(n_points, 3) * extent
    surface = rqs.Surface(tmp_model, crs_uuid = crs_uuid, title = "small_surface")
    ps = rqs.PointSet(tmp_model, points_array = points, crs_uuid = crs_uuid, title = 'temp point set')
    surface.set_from_point_set(ps,
                               convexity_parameter = 0.05,
                               reorient = True,
                               extend_with_flange = True,
                               flange_point_count = 11,
                               flange_radial_factor = 10.0,
                               flange_radial_distance = None,
                               make_clockwise = False)
    surface.triangles_and_points()
    surface.write_hdf5()
    surface.create_xml()

    tmp_model.store_epc()

    return grid, surface


@pytest.fixture
def small_grid_with_properties(tmp_model: Model) -> Tuple[grr.RegularGrid, List[str]]:
    crs = Crs(tmp_model)
    crs.create_xml()

    extent = 10
    extent_kji = (extent, extent, extent)
    dxyz = (1.0, 1.0, 1.0)
    crs_uuid = crs.uuid
    title = "small_grid"
    grid = grr.RegularGrid(tmp_model, extent_kji = extent_kji, dxyz = dxyz, crs_uuid = crs_uuid, title = title)
    grid.create_xml()

    pc = grid.extract_property_collection()
    col_prop_shape = (grid.nj, grid.ni)

    for i in range(20):
        a = (np.random.random(col_prop_shape) + 1.0) * 3000.0
        pc.add_cached_array_to_imported_list(
            a,
            source_info = "test data",
            keyword = "DEPTH",
            discrete = False,
            uom = "m",
            property_kind = "depth",
            realization = i,
            indexable_element = "columns",
        )
    pc.write_hdf5_for_imported_list()
    prop_uuids = pc.create_xml_for_imported_list_and_add_parts_to_model()
    tmp_model.store_epc()

    return grid, prop_uuids

@pytest.fixture
def small_grid_and_surface_nonrandom(tmp_model: Model) -> Tuple[grr.RegularGrid, rqs.Surface]:
    """Creates a small RegularGrid and a random triangular surface."""
    crs = Crs(tmp_model)
    crs.create_xml()

    extent = 10
    extent_kji = (extent, extent, extent)
    dxyz = (1.0, 1.0, 1.0)
    crs_uuid = crs.uuid
    title = "small_grid"
    grid = grr.RegularGrid(tmp_model, extent_kji = extent_kji, dxyz = dxyz, crs_uuid = crs_uuid, title = title)
    grid.create_xml()

    points = np.array([
        [0.08106761, 1.3295167 , 0.11162371], [1.12741581, 2.38222333, 0.68750453],
        [2.30178467, 0.46994407, 0.32888339], [0.1420671 , 2.88356852, 0.63064418],
        [1.86901752, 0.89184046, 2.20140932], [2.05943122, 2.72558781, 0.48310686],
        [2.34584349, 0.46591499, 0.28747912], [2.89670254, 2.76028382, 0.49175011],
        [0.90551676, 1.15075557, 1.94205026], [0.84642013, 0.22261785, 1.59408085],
        [1.72919692, 2.34885194, 1.01135179]
    ])
    triangles = tri.dt(points)
    surface = rqs.Surface(tmp_model, crs_uuid = crs_uuid, title = "small_surface")
    surface.set_from_triangles_and_points(triangles, points)
    surface.triangles_and_points()
    surface.write_hdf5()
    surface.create_xml()

    tmp_model.store_epc()

    return grid, surface
