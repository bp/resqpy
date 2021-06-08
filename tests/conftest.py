""" Shared fixtures for tests """

import logging

import pytest
import numpy as np
import pandas as pd

from resqpy.model import Model
from resqpy.organize import WellboreFeature, WellboreInterpretation
from resqpy.well import Trajectory, MdDatum
from resqpy.crs import Crs


@pytest.fixture(autouse=True)
def capture_logs(caplog):
   """Always capture log messages from respy"""

   caplog.set_level(logging.DEBUG, logger="resqpy")


@pytest.fixture
def example_model(tmp_path):
   """ Returns a fresh RESQML Model and Crs, in a temporary directory """

   # TODO: parameterise with m or feet
   xyz_uom = 'm'

   # Create a model with a coordinate reference system
   model = Model(create_basics=True)
   crs = Crs(parent_model=model, z_inc_down=True, xy_units=xyz_uom, z_units=xyz_uom)
   crs.create_xml()

   # Using tmp_path to get a temporary directory unique to each test
   model_path = str(tmp_path / f'test_well_{crs.uuid}.epc')
   model.set_epc_file_and_directory(model_path)

   # Add an hdf5 reference part in anticipation of some arrays being used later on
   model.create_hdf5_ext(add_as_part=True, file_name=model_path[:-4] + '.h5')

   return model, crs


@pytest.fixture
def example_model_with_well(example_model):
   """ Model with a single well with a vertical trajectory """

   wellname = 'well A'
   elevation = 100
   md_uom = 'm'

   model, crs = example_model

   # Create a single well feature and interpretation
   well_feature = WellboreFeature(parent_model=model, feature_name=wellname)
   well_feature.create_xml()
   well_interp = WellboreInterpretation(parent_model=model, wellbore_feature=well_feature, is_drilled=True)
   well_interp.create_xml()

   # Create a measured depth datum
   datum = MdDatum(
      parent_model=model, crs_root=crs.crs_root, location=(0, 0, -elevation), md_reference='kelly bushing'
   )
   datum.create_xml()

   # Create trajectory of a vertical well
   mds = np.array([0, 1000, 2000])
   zs = mds - elevation
   traj = Trajectory(
      parent_model=model,
      md_datum=datum,
      data_frame=pd.DataFrame(dict(MD=mds, X=0, Y=0, Z=zs)),
      length_uom=md_uom,
      represented_interp=well_interp
   )
   traj.write_hdf5(mode='w')
   traj.create_xml()

   return model, well_interp, datum, traj


