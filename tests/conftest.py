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
