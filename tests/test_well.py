from pathlib import Path

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


# Deviation Survey tests

def test_DeviationSurvey(example_model_with_well):

   # Load existing objects, using type hints
   model: Model
   model, well_interp, datum, traj = example_model_with_well
   epc_path = model.epc_file

   # Create new survey
   survey = resqpy.well.DeviationSurvey(
      parent_model=model,
      represented_interp=well_interp,
      md_datum=datum,
      measured_depths=[1,2,3],
      azimuths=[1,2,3],
      inclinations=[1,2,3],
      first_station=[4,5,6],
   )

   # Save to disk
   survey.write_hdf5()
   survey.create_xml()
   model.store_epc()
   model.h5_release()

   # Clear memory
   del model, well_interp, datum, traj, survey

   # Reload from disk, check survey can be found
   model2 = Model(epc_file=epc_path)
   parts = model2.parts_list_of_type("DeviationSurveyRepresentation")
   assert len(parts) == 1

   # Load array data into memory
   node = model2.root_for_part(parts[0])
   survey2 = resqpy.well.DeviationSurvey.from_xml(model2, node=node)
   assert_array_almost_equal(survey2.measured_depths, np.array([1,2,3]))
