from pathlib import Path
from resqpy.organize import WellboreFeature

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from resqpy.model import Model
import resqpy.well

@pytest.mark.skip(reason="Example data not yet available")
def test_trajectory_iterators():
   # Test that trajectories can be found via iterators
   model = Model()

   # Iterate via wells
   uuids_1 = []
   for well in model.wells():
      for traj in well.trajectories():
         uuids_1.append(traj.uuid)

   # Iterate directly
   uuids_2 = []
   for traj in model.trajectories():
      uuids_2.append(traj.uuid)

   # Two sets of trajectories should match, assuming all trajectories have a parent well
   assert len(uuids_1) > 0
   assert sorted(uuids_1) == sorted(uuids_2)


@pytest.mark.skip(reason="Example data not yet available")
def test_logs():
   # Check logs can be extracted from resqml dataset

   epc_path = Path(__file__).parent / 'example_data/my_model.epc'
   model = Model(epc_file=str(epc_path))

   discovered_logs = 0

   for trajectory in model.trajectories():
      for frame in trajectory.wellbore_frames():

         # Measured depths
         mds = frame.node_mds
         assert isinstance(mds, np.ndarray)
         assert len(mds) > 0

         # Logs
         # TODO: some way of testing whether log collection is empty or not
         log_collection = frame.logs

         # Test conversion
         df = log_collection.to_pandas()
         las = log_collection.to_las()
         assert len(df.columns) > 0
         assert len(df) > 0
         assert len(las.well.WELL) > 0

         for log in log_collection:
            values = log.values()

            assert len(log.name) > 0
            assert values.shape[-1] == len(mds)
            discovered_logs += 1

   assert discovered_logs > 0


# Trajectory

def test_Trajectory_add_well_feature_and_interp(example_model):

   # Prepare an example Trajectory without a well feature
   wellname = "Hullabaloo"
   model, crs = example_model
   datum = resqpy.well.MdDatum(
      parent_model=model, crs_root=crs.crs_root, location=(0, 0, -100), md_reference='kelly bushing'
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
