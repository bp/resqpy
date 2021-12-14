import os
import numpy as np
import pandas as pd
import pytest
import math

from resqpy.grid import RegularGrid
from resqpy.model import Model
import resqpy.olio.vector_utilities as vec
import resqpy.well


def test_Trajectory_load_from_xml(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    epc_path = model.epc_file
    elevation = 100
    # Create a measured depth datum
    datum = resqpy.well.MdDatum(parent_model = model,
                                crs_uuid = crs.uuid,
                                location = (0, 0, -elevation),
                                md_reference = 'kelly bushing')
    datum.create_xml()
    mds = np.array([300, 310, 330, 340])
    zs = mds - elevation
    well_name = 'JubJub'
    source_dataframe = pd.DataFrame({
        'MD': mds,
        'X': [100, 120, 140, 160],
        'Y': [345, 365, 386, 400],
        'Z': zs,
        'WELL': ['JubJub', 'JubJub', 'JubJub', 'JubJub']
    })

    # Create a trajectory from dataframe
    trajectory = resqpy.well.Trajectory(parent_model = model,
                                        data_frame = source_dataframe,
                                        well_name = well_name,
                                        md_datum = datum,
                                        length_uom = 'm')

    # --------- Act ----------
    # Save to disk
    trajectory.write_hdf5()
    trajectory.create_xml()
    trajectory_uuid = trajectory.uuid
    model.store_epc()
    model.h5_release()

    # Reload from disk
    model2 = Model(epc_file = epc_path)
    trajectory2 = resqpy.well.Trajectory(model2, uuid = trajectory_uuid, set_tangent_vectors = True)

    # --------- Assert --------------
    # Check all attributes were loaded from disk correctly
    assert trajectory2 is not None
    assert trajectory2.well_name == well_name
    np.testing.assert_almost_equal(trajectory2.measured_depths, mds)
    assert trajectory2.md_datum == datum
    assert trajectory2.md_uom == 'm'


def test_Trajectory_add_well_feature_and_interp(example_model_and_crs):

    # Prepare an example Trajectory without a well feature
    wellname = "Hullabaloo"
    model, crs = example_model_and_crs
    datum = resqpy.well.MdDatum(parent_model = model,
                                crs_uuid = crs.uuid,
                                location = (0, 0, -100),
                                md_reference = 'kelly bushing')
    datum.create_xml()
    traj = resqpy.well.Trajectory(parent_model = model, md_datum = datum, well_name = wellname)

    # Add the well interp
    assert traj.wellbore_feature is None
    assert traj.wellbore_interpretation is None
    traj.create_feature_and_interpretation()

    # Check well is present
    assert traj.wellbore_feature is not None
    assert traj.wellbore_feature.feature_name == wellname


def test_compute_from_deviation_survey(example_model_with_well):

    # --------- Arrange ----------
    # Create a Deviation Survey object in memory
    # Load example model from a fixture
    model, well_interp, datum, _ = example_model_with_well

    # Create a deviation survey
    data = dict(
        title = 'Majestic Umlaut ö',
        originator = 'Thor, god of sparkles',
        md_uom = 'ft',
        angle_uom = 'rad',
        is_final = True,
    )
    array_data = dict(
        measured_depths = np.array([1, 2, 3], dtype = float) + 1000.0,
        azimuths = np.array([4, 5, 6], dtype = float),
        inclinations = np.array([1, 2, 3], dtype = float),
        first_station = np.array([0, -1, 999], dtype = float),
    )

    survey = resqpy.well.DeviationSurvey(
        parent_model = model,
        represented_interp = well_interp,
        md_datum = datum,
        **data,
        **array_data,
    )

    # --------- Act ----------
    # Create a trajectory from the deviation survey
    trajectory_from_deviation_survey = resqpy.well.Trajectory(parent_model = model, deviation_survey = survey)

    # --------- Assert ----------
    assert trajectory_from_deviation_survey is not None
    assert trajectory_from_deviation_survey.knot_count == survey.station_count == \
           trajectory_from_deviation_survey.control_points.shape[0] == 3
    assert trajectory_from_deviation_survey.md_datum == datum
    assert trajectory_from_deviation_survey.measured_depths[0] == 1001.0


def test_load_from_wellspec(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (4, 4, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    wellspec_file = os.path.join(model.epc_directory, 'wellspec.dat')
    well_name = 'DOGLEG'
    source_df = pd.DataFrame([[2, 2, 1, 0.0, 0.0, 0.0, 0.25], [2, 2, 2, 0.45, -90.0, 2.5, 0.25],
                              [2, 3, 2, 0.45, -90.0, 1.0, 0.20], [2, 3, 3, 0.0, 0.0, -0.5, 0.20]],
                             columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW'])
    with open(wellspec_file, 'w') as fp:
        fp.write(F'WELLSPEC {well_name}\n')
        for col in source_df.columns:
            fp.write(f' {col:>6s}')
        fp.write('\n')
        for row_index in range(len(source_df)):
            row = source_df.iloc[row_index]
            for col in source_df.columns:
                if col in ['IW', 'JW', 'L']:
                    fp.write(f' {int(row[col]):6d}')
                else:
                    fp.write(f' {row[col]:6.2f}')
            fp.write('\n')

    # --------- Act ----------
    # Create a trajectory from the WELLSPEC file
    trajectory_from_wellspec = resqpy.well.Trajectory(parent_model = model,
                                                      well_name = well_name,
                                                      length_uom = 'm',
                                                      grid = grid,
                                                      wellspec_file = wellspec_file)

    # --------- Assert ----------
    assert trajectory_from_wellspec is not None
    assert trajectory_from_wellspec.knot_count == len(trajectory_from_wellspec.control_points)


def test_load_from_ascii_file(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    well_names = [None, 'SnowCone']
    elevation = 100
    # Create a measured depth datum
    datum = resqpy.well.MdDatum(parent_model = model,
                                crs_uuid = crs.uuid,
                                location = (0, 0, -elevation),
                                md_reference = 'kelly bushing')
    mds = np.array([300, 310, 330, 340])
    zs = mds - elevation
    source_dataframe = pd.DataFrame({
        'MD': mds,
        'X': [1, 2, 3, 4],
        'Y': [1, 2, 3, 4],
        'Z': zs,
        'WELL': ['SnowCone', 'SnowCone', 'Paime', 'Paime']
    })
    trajectory_data_file_path = os.path.join(model.epc_directory, 'trajectory_data.csv')
    source_dataframe.to_csv(trajectory_data_file_path)

    for well_name in well_names:
        if well_name is None:
            # --------- Act ----------
            # Create a trajectory from the csv file
            with pytest.raises(Exception) as excinfo:
                trajectory_from_ascii = resqpy.well.Trajectory(parent_model = model,
                                                               ascii_trajectory_file = trajectory_data_file_path,
                                                               length_uom = 'm',
                                                               md_datum = datum)

            # -------- Assert ---------
            assert "attempt to set trajectory for unidentified well from ascii file holding data for multiple wells" in str(
                excinfo.value)
        else:
            # --------- Act ----------
            # Create a trajectory from the csv file
            trajectory_from_ascii = resqpy.well.Trajectory(parent_model = model,
                                                           ascii_trajectory_file = trajectory_data_file_path,
                                                           well_name = well_name,
                                                           length_uom = 'm',
                                                           md_datum = datum,
                                                           set_tangent_vectors = True)
            # -------- Assert ---------
            assert trajectory_from_ascii is not None
            np.testing.assert_almost_equal(trajectory_from_ascii.tangent_vectors[0],
                                           trajectory_from_ascii.tangent_vectors[1])
            vec.isclose(trajectory_from_ascii.tangent_vectors[:, 2], np.array([1, 1]), 0.01)


def test_set_tangents(example_model_and_crs):
    # --------- Arrange ----------
    model, crs = example_model_and_crs
    elevation = 100
    # Create a measured depth datum
    datum = resqpy.well.MdDatum(parent_model = model,
                                crs_uuid = crs.uuid,
                                location = (0, 0, -elevation),
                                md_reference = 'kelly bushing')
    mds = np.array([300, 310, 330, 340])
    zs = mds - elevation
    well_name = 'JubJub'
    source_dataframe = pd.DataFrame({
        'MD': mds,
        'X': [1, 2, 3, 4],
        'Y': [1, 2, 3, 4],
        'Z': zs,
        'WELL': ['JubJub', 'JubJub', 'JubJub', 'JubJub']
    })

    # Create a trajectory from dataframe
    trajectory = resqpy.well.Trajectory(parent_model = model,
                                        data_frame = source_dataframe,
                                        well_name = well_name,
                                        md_datum = datum,
                                        length_uom = 'm')

    # --------- Act ----------
    # Calculate tangent vectors based on control points
    trajectory.set_tangents()
    # --------- Assert ----------
    assert trajectory.tangent_vectors is not None
    assert trajectory.tangent_vectors.shape == (trajectory.knot_count, 3)
    assert vec.isclose(trajectory.tangent_vectors[0], trajectory.tangent_vectors[1], tolerance = 0.01)
    np.testing.assert_equal(vec.azimuths(trajectory.tangent_vectors),
                            np.array([45, 45, 45, 45]))  # X and Y coordinates follow straight line x = y
    for v in trajectory.tangent_vectors:
        assert vec.inclination(v) < 10


def test_xyz_for_md(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    elevation = 100
    # Create a measured depth datum
    datum = resqpy.well.MdDatum(parent_model = model,
                                crs_uuid = crs.uuid,
                                location = (0, 0, -elevation),
                                md_reference = 'kelly bushing')
    mds = np.array([300, 310, 330, 340])
    zs = mds - elevation
    well_name = 'CoconutDrop'
    source_dataframe = pd.DataFrame({
        'MD': mds,
        'X': [1, 2, 3, 4],
        'Y': [1, 2, 3, 4],
        'Z': zs,
        'WELL': ['CoconutDrop', 'CoconutDrop', 'CoconutDrop', 'CoconutDrop']
    })

    # Create a trajectory from dataframe
    trajectory = resqpy.well.Trajectory(parent_model = model,
                                        data_frame = source_dataframe,
                                        well_name = well_name,
                                        md_datum = datum,
                                        length_uom = 'm')
    # --------- Act ----------
    # # Get the xyz triplet for a given measured depth
    x, y, z = trajectory.xyz_for_md(md = 305)

    # # -------- Assert ---------
    assert (math.isclose(x, 1.5)) & (math.isclose(y, 1.5)) & (math.isclose(z, 205))


def test_splined_trajectory(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    elevation = 100
    # Create a measured depth datum
    datum = resqpy.well.MdDatum(parent_model = model,
                                crs_uuid = crs.uuid,
                                location = (0, 0, -elevation),
                                md_reference = 'kelly bushing')
    mds = np.array([300, 310, 320, 340])
    zs = mds - elevation
    well_name = 'CoconutDrop'
    source_dataframe = pd.DataFrame({
        'MD': mds,
        'X': [1, 2, 3, 4],
        'Y': [1, 2, 3, 4],
        'Z': zs,
        'WELL': ['CoconutDrop', 'CoconutDrop', 'CoconutDrop', 'CoconutDrop']
    })

    # Create a trajectory from dataframe
    trajectory = resqpy.well.Trajectory(parent_model = model,
                                        data_frame = source_dataframe,
                                        well_name = well_name,
                                        md_datum = datum,
                                        length_uom = 'm')
    # --------- Act ----------
    # Get splined trajectory
    splined_trajectory = trajectory.splined_trajectory(well_name = 'CoconutDrop')

    # -------- Assert ---------
    assert splined_trajectory is not None
    np.testing.assert_almost_equal(trajectory.measured_depths[0], splined_trajectory.measured_depths[0], 0)
    np.testing.assert_almost_equal(trajectory.measured_depths[-1], splined_trajectory.measured_depths[-1], 0)


def test_dataframe(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    elevation = 100
    # Create a measured depth datum
    datum = resqpy.well.MdDatum(parent_model = model,
                                crs_uuid = crs.uuid,
                                location = (0, 0, -elevation),
                                md_reference = 'kelly bushing')
    mds = np.array([300, 310, 330, 340])
    zs = mds - elevation
    source_dataframe = pd.DataFrame({
        'MD': mds,
        'X': [1, 2, 3, 4],
        'Y': [1, 2, 3, 4],
        'Z': zs,
    })

    # Create a trajectory from dataframe
    trajectory = resqpy.well.Trajectory(parent_model = model,
                                        data_frame = source_dataframe,
                                        md_datum = datum,
                                        length_uom = 'm')

    # --------- Act ----------
    # Get the dataframe containing the trajectory data
    returned_dataframe = trajectory.dataframe()

    # -------- Assert ---------
    pd.testing.assert_frame_equal(source_dataframe.astype(float), returned_dataframe)
