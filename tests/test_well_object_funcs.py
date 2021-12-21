import os
import pandas as pd
import numpy as np
import pytest

from resqpy.well.well_object_funcs import add_wells_from_ascii_file, well_name, add_blocked_wells_from_wellspec, add_logs_from_cellio, lookup_from_cellio
import resqpy.well
from resqpy.grid import RegularGrid


def test_add_wells_from_ascii_file(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    crs_uuid = crs.uuid
    ascii_file = os.path.join(model.epc_directory, 'ascii.dat')
    well_name = 'Well_1'

    source_df = pd.DataFrame([[well_name, 200, 1, 1, 150], [well_name, 250, 2, 2, 200], [well_name, 300, 2, 3, 250]],
                             columns = ['NotaWell', 'MD', 'X', 'Y', 'Z'])
    with open(ascii_file, 'w') as fp:
        for col in source_df.columns:
            fp.write(f' {col}')
        fp.write('\n')
        for row_index in range(len(source_df)):
            row = source_df.iloc[row_index]
            for col in source_df.columns:
                if col == 'NotaWell':
                    fp.write(f' {(row[col])}')
                else:
                    fp.write(f' {row[col]}')
            fp.write('\n')

    # Use incorrect name for the well column
    # --------- Act ----------
    with pytest.raises(AssertionError):
        (feature_list, interpretation_list, trajectory_list,
         md_datum_list) = add_wells_from_ascii_file(model = model,
                                                    crs_uuid = crs_uuid,
                                                    trajectory_file = ascii_file,
                                                    well_col = None,
                                                    space_separated_instead_of_csv = True)

    # Use correct name for well column
    # --------- Act ----------
    (feature_list2, interpretation_list2, trajectory_list2,
     md_datum_list2) = add_wells_from_ascii_file(model = model,
                                                 crs_uuid = crs_uuid,
                                                 trajectory_file = ascii_file,
                                                 well_col = 'NotaWell',
                                                 space_separated_instead_of_csv = True)

    test_trajectory = trajectory_list2[0]
    # --------- Assert ----------
    assert len(feature_list2) == len(interpretation_list2) == len(trajectory_list2) == len(md_datum_list2) == 1
    np.testing.assert_equal(test_trajectory.measured_depths, np.array([200., 250., 300.]))


def test_best_root_for_object(example_model_with_well):

    # --------- Arrange ----------
    model, well_interp, datum, traj = example_model_with_well
    traj_title = traj.title
    wellbore_feature_uuid = model.uuid_for_part(part_name = model.part(obj_type = 'obj_WellboreFeature'))
    wellbore_marker_frame = resqpy.well.WellboreMarkerFrame(parent_model = model, trajectory_uuid = traj.uuid)
    wellbore_marker_frame.create_xml()
    wellbore_frame = resqpy.well.WellboreFrame(parent_model = model, trajectory = traj)
    wellbore_frame.create_xml()
    deviation_survey = resqpy.well.DeviationSurvey(parent_model = model,
                                                   md_datum = datum,
                                                   measured_depths = [200, 210, 250],
                                                   azimuths = [10, 10, 10],
                                                   inclinations = [45, 45, 45],
                                                   first_station = (0, 0, 0))
    deviation_survey.create_xml()

    # --------- Act ----------
    citation_title1 = well_name(well_object = well_interp, model = model)
    citation_title2 = well_name(well_object = wellbore_marker_frame, model = model)
    citation_title3 = well_name(well_object = wellbore_frame, model = model)
    citation_title4 = well_name(well_object = datum, model = model)
    citation_title5 = well_name(well_object = deviation_survey, model = model)
    citation_title6 = well_name(well_object = wellbore_feature_uuid, model = model)

    # --------- Assert ----------
    assert citation_title1 == citation_title2 == citation_title3 == citation_title6 == traj_title
    assert citation_title4 == citation_title5 == datum.title


def test_add_blocked_wells_from_wellspec(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (3, 4, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    wellspec_file = os.path.join(model.epc_directory, 'wellspec.dat')
    well_name = 'DOGLEG'
    source_df = pd.DataFrame([[2, 2, 1, 0.0, 0.0, 0.0, 0.25, 0.9], [2, 2, 2, 0.45, -90.0, 2.5, 0.25, 0.9],
                              [2, 3, 2, 0.45, -90.0, 1.0, 0.20, 0.9], [2, 3, 3, 0.0, 0.0, -0.5, 0.20, 0.9]],
                             columns = ['IW', 'JW', 'L', 'ANGLV', 'ANGLA', 'SKIN', 'RADW', 'PPERF'])
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
    count = add_blocked_wells_from_wellspec(model = model, grid = grid, wellspec_file = wellspec_file)

    # --------- Assert ----------
    assert count == 1


def test_add_logs_from_cellio_file(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (5, 4, 4),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)

    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    elevation = 100
    # Create a measured depth datum
    datum = resqpy.well.MdDatum(parent_model = model,
                                crs_uuid = crs.uuid,
                                location = (0, 0, -elevation),
                                md_reference = 'kelly bushing')
    mds = np.array([100, 210, 230, 240, 250])
    zs = mds - elevation
    well_name = 'Coconut'
    source_dataframe = pd.DataFrame({
        'MD': mds,
        'X': [25, 50, 75, 100, 100],
        'Y': [25, -50, -75, -100, -100],
        'Z': zs,
        'WELL': ['Coconut', 'Coconut', 'Coconut', 'Coconut', 'Coconut']
    })

    # Create a trajectory from dataframe
    trajectory = resqpy.well.Trajectory(parent_model = model,
                                        data_frame = source_dataframe,
                                        well_name = well_name,
                                        md_datum = datum,
                                        length_uom = 'm')
    trajectory.write_hdf5()
    trajectory.create_xml()

    # Create a blocked well using the trajectory
    bw = resqpy.well.BlockedWell(model,
                                 well_name = well_name,
                                 grid = grid,
                                 trajectory = trajectory,
                                 use_face_centres = True)
    bw.cell_grid_link = bw.map_cell_and_grid_indices()  # TODO: confirm this is valid
    bw.write_hdf5()
    bw.create_xml()

    cellio_file = os.path.join(model.epc_directory, 'cellio.dat')
    source_df = pd.DataFrame(
        [[1, 1, 1, 25, 25, 0, 26, 26, 1, 120, 0.12], [2, 2, 1, 26, -26, 126, 27, -27, 127, 117, 0.20],
         [2, 3, 1, 27, -27, 127, 28, -28, 128, 135, 0.15]],
        columns = [
            'i_index unit1 scale1', 'j_index unit1 scale1', 'k_index unit1 scale1', 'x_in unit1 scale1',
            'y_in unit1 scale1', 'z_in unit1 scale1', 'x_out unit1 scale1', 'y_out unit1 scale1', 'z_out unit1 scale1',
            'Perm unit1 scale1', 'Poro unit1 scale1'
        ])

    with open(cellio_file, 'w') as fp:
        fp.write('1.0\n')
        fp.write('Undefined\n')
        fp.write(f'{well_name} terrible day\n')
        fp.write('11\n')
        for col in source_df.columns:
            fp.write(f' {col}\n')
        for row_index in range(len(source_df)):
            row = source_df.iloc[row_index]
            for col in source_df.columns:
                fp.write(f' {int(row[col])}')
            fp.write('\n')

    # --------- Act ----------
    add_logs_from_cellio(blockedwell = bw, cellio = cellio_file)

    # --------- Assert ----------
    assert 'cellio.Coconut.Perm' in model.titles()
    assert 'cellio.Coconut.Poro' in model.titles()


def test_lookup_from_cellio(example_model_and_crs):
    # --------- Arrange ----------
    model, crs = example_model_and_crs
    line = 'Perm continuous 23 34 45 67 87 99'

    # --------- Act ----------
    uuid = lookup_from_cellio(line = line, model = model)
    uuid2 = lookup_from_cellio(line = line, model = model)

    # --------- Assert ----------
    assert uuid is not None
    assert 'Perm' in model.titles()
    assert uuid2 is uuid
