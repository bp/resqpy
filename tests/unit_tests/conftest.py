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

    n_points = 100
    points = np.array(
        [[5.44992339, 4.57510223, 7.63222857],
         [2.84841428, 7.25510613, 7.90621386],
         [4.59058197, 3.39498391, 2.99842991],
         [5.60815621, 5.76022692, 0.96399591],
         [7.69817082, 6.7353259 , 4.25015747],
         [1.88977698, 0.1365234 , 9.27407331],
         [5.31233185, 2.54577529, 9.83747969],
         [1.7497728 , 3.05790326, 6.23047258],
         [8.54294837, 8.82085419, 7.46614124],
         [3.3354642 , 4.48070594, 0.20676581],
         [3.94850556, 6.40704933, 2.8844538 ],
         [3.49491911, 5.77442821, 4.32491584],
         [5.52392142, 5.97600181, 7.9583256 ],
         [5.80288195, 2.26139807, 7.22527173],
         [1.9586503 , 4.18214254, 1.10391511],
         [5.3671814 , 0.6623936 , 5.61769963],
         [9.4167267 , 9.9012364 , 5.02808068],
         [8.29894233, 6.51445587, 8.83799604],
         [7.51900558, 4.35580993, 4.36026085],
         [2.31980242, 9.46793806, 8.6065574 ],
         [3.56706032, 8.55525588, 3.17346409],
         [5.3076571 , 7.61781483, 0.74181143],
         [5.80636451, 3.46216862, 6.42855982],
         [4.72596234, 9.1798443 , 9.43323067],
         [4.81651242, 0.82140639, 1.87322352],
         [3.40180235, 7.63392946, 0.73726016],
         [3.97004708, 2.26344254, 8.76564552],
         [1.0227386 , 2.05470715, 9.12068908],
         [5.06420936, 8.7280928 , 3.35934004],
         [9.19432578, 0.22380681, 9.05637998],
         [1.00280307, 0.6387967 , 7.80660596],
         [7.91293707, 1.11843899, 3.67861994],
         [9.06223405, 4.78302107, 6.6657562 ],
         [2.46721061, 0.98896496, 9.52104773],
         [6.51829782, 6.37568074, 8.75105698],
         [1.23003336, 8.46412852, 6.86396878],
         [3.37472448, 0.90629028, 8.36667716],
         [3.22547853, 1.27684556, 0.2205579 ],
         [4.60694631, 8.59846521, 4.42545959],
         [0.33810172, 1.28959538, 4.66684351],
         [8.86707237, 2.31556092, 5.61761342],
         [8.29342091, 7.57828637, 9.90454607],
         [7.57304137, 9.5597469 , 2.72396693],
         [5.41920482, 3.64372197, 2.8986212 ],
         [1.61710389, 2.48145114, 5.92443993],
         [1.06070718, 9.87278419, 1.5857908 ],
         [5.66384914, 3.21059754, 8.87297633],
         [6.74312617, 8.37951487, 5.1598423 ],
         [5.31595661, 6.01022342, 6.74442087],
         [6.38409299, 3.8168544 , 6.35371444],
         [4.11752689, 2.20341714, 1.97478715],
         [8.45018345, 3.20132494, 3.51045747],
         [5.94449365, 5.57237464, 8.42617332],
         [9.09768297, 2.45592411, 2.66225233],
         [1.61741103, 7.77540497, 3.17967567],
         [7.9319912 , 0.62844482, 2.03823047],
         [8.03269802, 2.97722898, 5.82367477],
         [7.30085122, 3.28678997, 3.82172844],
         [8.4834966 , 3.20341164, 5.67370153],
         [9.10338544, 7.4515019 , 5.59932125],
         [9.84928031, 3.12325758, 9.88978624],
         [7.20445929, 7.29108251, 9.67955954],
         [2.16574505, 4.25835637, 2.27465065],
         [2.31747158, 1.38830427, 8.2312226 ],
         [1.32588333, 6.52128076, 7.05212993],
         [5.93174865, 7.93637211, 2.0910971 ],
         [3.59508474, 4.37925397, 2.3556156 ],
         [2.53134179, 6.44027183, 3.81828256],
         [8.47859345, 6.30646538, 2.53014201],
         [1.20025006, 5.61519942, 6.44476921],
         [7.32588208, 5.03579262, 5.18256064],
         [7.65214612, 4.45107698, 0.38262677],
         [9.30354999, 4.62954911, 3.93158791],
         [3.37731101, 5.15089561, 3.37875287],
         [3.95256491, 2.11237923, 2.90032595],
         [9.7658164 , 3.99220241, 0.83294815],
         [0.53508462, 7.5640378 , 5.12815476],
         [1.96167945, 2.25391886, 6.37903208],
         [3.25074919, 9.1114962 , 3.29059579],
         [9.09776506, 8.58704112, 5.24490921],
         [4.08783038, 1.42431072, 4.49632373],
         [3.66879966, 5.35736116, 0.10943462],
         [1.04039694, 2.41989899, 8.50810905],
         [8.57543494, 6.09971209, 1.8889553 ],
         [1.11034537, 9.7772489 , 1.50799651],
         [3.14901058, 1.9150315 , 4.82246888],
         [7.96809542, 2.82821335, 9.38425406],
         [8.37522193, 0.23776627, 3.05256498],
         [3.72200456, 2.60654862, 7.0266145 ],
         [9.11067587, 7.76915762, 7.16591543],
         [5.8554316 , 8.66403807, 1.88641128],
         [5.38474232, 8.75881894, 0.37795057],
         [6.07120636, 4.73078781, 6.97892226],
         [8.83465985, 2.71555428, 1.9137528 ],
         [2.260986  , 9.01290557, 7.24451656],
         [7.94538634, 9.32381705, 4.89336086],
         [1.03173338, 3.10918616, 4.58704131],
         [4.21456479, 1.37300124, 2.57583847],
         [1.09410637, 4.60533303, 6.97173082],
         [3.37516656, 6.10493044, 9.60024728]
        ])
    triangles = tri.dt(points)
    surface = rqs.Surface(tmp_model, crs_uuid = crs_uuid, title = "small_surface")
    surface.set_from_triangles_and_points(triangles, points)
    surface.triangles_and_points()
    surface.write_hdf5()
    surface.create_xml()

    tmp_model.store_epc()

    return grid, surface
