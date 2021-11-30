import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal


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
    assert_array_almost_equal(mds, [1, 2, 3, 4])

    # Check logs
    log_list = list(log_collection.iter_logs())
    assert len(log_list) == 2

    # TODO: would be nice to write: log_collection.get_curve("GR")
    gr = log_list[0]
    assert gr.title == "GR"
    assert gr.uom == "gAPI"
    assert_array_almost_equal(gr.values(), [1, 2, 1, 2])

    nphi = log_list[1]
    assert nphi.title == 'NPHI'

    # TODO: get more units working
    # assert nphi.uom == "v/v"
    assert_array_almost_equal(nphi.values(), [0.1, 0.1, np.NaN, np.NaN])


def test_logs_conversion(example_model_with_logs):

    model, well_interp, datum, traj, frame, log_collection = example_model_with_logs

    # Pandas
    df = log_collection.to_df()
    df_expected = pd.DataFrame(data = {"GR": [1, 2, 1, 2], "NPHI": [0.1, 0.1, np.NaN, np.NaN]}, index = [1, 2, 3, 4])
    assert_frame_equal(df_expected, df, check_dtype = False)

    # LAS
    las = log_collection.to_las()
    assert las.well.WELL.value == 'well A'

    gr = las.get_curve("GR")
    assert gr.unit.casefold() == "GAPI".casefold()
    assert_array_almost_equal(gr.data, [1, 2, 1, 2])

    nphi = las.get_curve("NPHI")
    # assert nphi.unit == "GAPI"
    assert_array_almost_equal(nphi.data, [0.1, 0.1, np.NaN, np.NaN])
