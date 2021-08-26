""" Shared fixtures for tests """

import logging

import pytest
from pathlib import Path
from shutil import copytree
import numpy as np
import pandas as pd
import os

from resqpy.model import Model, new_model
from resqpy.organize import WellboreFeature, WellboreInterpretation
from resqpy.well import Trajectory, MdDatum, WellboreFrame
from resqpy.crs import Crs
import resqpy.grid as grr
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp


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

   log_collection = frame.logs
   log_collection.add_log("GR", [1, 2, 1, 2], 'gAPI')
   log_collection.add_log("NPHI", [0.1, 0.1, np.NaN, np.NaN], 'v/v')

   return model, well_interp, datum, traj, frame, log_collection


@pytest.fixture
def test_data_path(tmp_path):
   """ Return pathlib.Path pointing to temporary copy of tests/example_data

   Use a fresh temporary directory for each test.
   """
   master_path = (Path(__file__) / '../test_data').resolve()
   data_path = Path(tmp_path) / 'test_data'

   assert master_path.exists()
   assert not data_path.exists()
   copytree(str(master_path), str(data_path))
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
                   add_cell_length_properties = False)
   model.store_epc()

   zone = np.ones(shape = (5, 5))
   zone_array = np.array([zone, zone + 1, zone + 2])

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

   collection = rqp.GridPropertyCollection()
   collection.set_grid(grid)
   for array, name, kind, discrete in zip(
      [zone_array, vpc_array, fb_array, facies_array, ntg_array, por_array, sat_array],
      ['Zone', 'VPC', 'Fault block', 'Facies', 'NTG', 'POR', 'SW'],
      ['discrete', 'discrete', 'discrete', 'discrete', 'net to gross ratio', 'porosity', 'saturation'],
      [True, True, True, True, False, False, False]):
      collection.add_cached_array_to_imported_list(cached_array = array,
                                                   source_info = '',
                                                   keyword = name,
                                                   discrete = discrete,
                                                   uom = None,
                                                   time_index = None,
                                                   null_value = None,
                                                   property_kind = kind,
                                                   facet_type = None,
                                                   facet = None,
                                                   realization = None)
      collection.write_hdf5_for_imported_list()
      collection.create_xml_for_imported_list_and_add_parts_to_model()
   model.store_epc()

   return model
