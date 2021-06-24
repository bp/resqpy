from pathlib import Path
from resqpy.organize import WellboreFeature

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal

from resqpy.model import Model
from resqpy.crs import Crs
import resqpy.well


def test_MdDatum(example_model_and_crs):

   # Set up a new datum
   model, crs = example_model_and_crs
   epc = model.epc_file
   data = dict(
      location=(0, -99999, 3.14),
      md_reference='mean low water',
   )
   datum = resqpy.well.MdDatum(
      parent_model=model, crs_root=crs.root, **data
   )
   uuid = datum.uuid

   # Save to disk and reload
   datum.create_xml()
   model.store_epc()

   del model, crs, datum
   model2 = Model(epc_file=epc)
   datum2 = resqpy.well.MdDatum(parent_model=model2, uuid=uuid)
   
   for key, expected_value in data.items():
      assert getattr(datum2, key) == expected_value, f"Issue with {key}"


def test_trajectory_iterators(example_model_with_well):
   # Test that trajectories can be found via iterators

   model, well_interp, datum, traj = example_model_with_well

   # Iterate via wells
   uuids_1 = []
   for well in model.iter_wellbore_interpretations():
      for traj in well.iter_trajectories():
         uuids_1.append(traj.uuid)

   # Iterate directly
   uuids_2 = []
   for traj in model.iter_trajectories():
      uuids_2.append(traj.uuid)

   # Two sets of trajectories should match, assuming all trajectories have a parent well
   assert len(uuids_1) > 0
   assert sorted(uuids_1) == sorted(uuids_2)


def test_logs(example_model_with_logs):

   model, well_interp, datum, traj, frame, log_collection = example_model_with_logs

   # Check logs can be extracted from resqml dataset

   # Check exactly one wellbore frame exists in the model
   discovered_frames = 0
   for trajectory in model.iter_trajectories():
      for frame in trajectory.iter_wellbore_frames():
         discovered_frames += 1
   assert discovered_frames == 1

   # Check MDs
   mds = frame.node_mds
   assert isinstance(mds, np.ndarray)
   assert_array_almost_equal(mds, [1,2,3,4])

   # Check logs
   log_list = list(log_collection.iter_logs())
   assert len(log_list) == 2

   # TODO: would be nice to write: log_collection.get_curve("GR")
   gr = log_list[0]
   assert gr.title == "GR"
   assert gr.uom == "gAPI"
   assert_array_almost_equal(gr.values(), [1,2,1,2])

   nphi = log_list[1]
   assert nphi.title == 'NPHI'

   # TODO: get more units working
   # assert nphi.uom == "v/v"
   assert_array_almost_equal(nphi.values(), [0.1, 0.1, np.NaN, np.NaN])
   

def test_logs_conversion(example_model_with_logs):

   model, well_interp, datum, traj, frame, log_collection = example_model_with_logs

   # Pandas
   df = log_collection.to_df()
   df_expected = pd.DataFrame(
      data={"GR": [1,2,1,2], "NPHI": [0.1, 0.1, np.NaN, np.NaN]},
      index=[1,2,3,4]
   )
   assert_frame_equal(df_expected, df, check_dtype=False)
   
   # LAS
   las = log_collection.to_las()
   assert las.well.WELL.value == 'well A'

   gr = las.get_curve("GR")
   assert gr.unit.casefold() == "GAPI".casefold()
   assert_array_almost_equal(gr.data, [1,2,1,2])

   nphi = las.get_curve("NPHI")
   # assert nphi.unit == "GAPI"
   assert_array_almost_equal(nphi.data, [0.1, 0.1, np.NaN, np.NaN])


# Trajectory

def test_Trajectory_add_well_feature_and_interp(example_model_and_crs):

   # Prepare an example Trajectory without a well feature
   wellname = "Hullabaloo"
   model, crs = example_model_and_crs
   datum = resqpy.well.MdDatum(
      parent_model=model, crs_root=crs.root, location=(0, 0, -100), md_reference='kelly bushing'
   )
   datum.create_xml()
   traj = resqpy.well.Trajectory(parent_model=model, md_datum=datum, well_name=wellname)


   # Add the well interp
   assert traj.wellbore_feature is None
   assert traj.wellbore_interpretation is None
   traj.create_feature_and_interpretation()

   # Check well is present
   assert traj.wellbore_feature is not None
   assert traj.wellbore_feature.feature_name == wellname


# Deviation Survey tests

def test_DeviationSurvey(example_model_with_well):
   # Test that all attrbutes are correctly saved and loaded from disk

   # --------- Arrange ----------
   # Create a Deviation Survey object in memory

   # Load example model from a fixture
   model, well_interp, datum, traj = example_model_with_well
   epc_path = model.epc_file

   # Create the survey
   data = dict(
      title='Majestic Umlaut ö',
      originator='Thor, god of sparkles',
      md_uom='ft',
      angle_uom='rad',
      is_final=True,
   )
   array_data = dict(
      measured_depths=np.array([1, 2, 3], dtype=float),
      azimuths=np.array([4, 5, 6], dtype=float),
      inclinations=np.array([7, 8, 9], dtype=float),
      first_station=np.array([0, -1, 999], dtype=float),
   )
   survey = resqpy.well.DeviationSurvey(
      parent_model=model,
      represented_interp=well_interp,
      md_datum=datum,
      **data,
      **array_data,
   )
   survey_uuid = survey.uuid

   # ----------- Act ---------

   # Save to disk
   survey.write_hdf5()
   survey.create_xml()
   model.store_epc()
   model.h5_release()

   # import shutil
   # shutil.copy(epc_path, r'C:\Temp')
   # shutil.copy(epc_path.replace('.epc', '.h5'), r'C:\Temp')

   # Clear memory
   del model, well_interp, datum, traj, survey

   # Reload from disk
   model2 = Model(epc_file=epc_path)
   survey2 = resqpy.well.DeviationSurvey(model2, uuid=survey_uuid)

   # --------- Assert --------------
   # Check all attributes were loaded from disk correctly

   for key, expected_value in data.items():
      assert getattr(survey2, key) == expected_value, f"Error for {key}"
   
   for key, expected_value in array_data.items():
      assert_array_almost_equal(
         getattr(survey2, key), expected_value, err_msg=f"Error for {key}"
      )
