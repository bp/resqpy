import os
import pandas as pd
import pytest

from resqpy.well.well_object_funcs import add_wells_from_ascii_file, well_name
import resqpy.well


def test_add_wells_from_ascii_file(example_model_and_crs, caplog):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    crs_uuid = crs.uuid
    ascii_file = os.path.join(model.epc_directory, 'ascii.dat')
    well_name = 'Well_1'

    source_df = pd.DataFrame([[well_name, 200, 1, 1, 150], [well_name, 250, 2, 2, 200], [well_name, 300, 2, 3, 250]],
                             columns = ['WELL', 'MD', 'X', 'Y', 'Z'])
    with open(ascii_file, 'w') as fp:
        for col in source_df.columns:
            fp.write(f' {col:>6s}')
        fp.write('\n')
        for row_index in range(len(source_df)):
            row = source_df.iloc[row_index]
            for col in source_df.columns:
                if col == 'WELL':
                    fp.write(f' {(row[col]):6s}')
                else:
                    fp.write(f' {row[col]:6.2f}')
            fp.write('\n')

    well_col_name = 'Wellname'
    with pytest.raises(AssertionError) as exc:
        # --------- Act ----------
        (feature_list, interpretation_list, trajectory_list,
         md_datum_list) = add_wells_from_ascii_file(model = model,
                                                    crs_uuid = crs_uuid,
                                                    trajectory_file = ascii_file,
                                                    well_col = well_col_name)
    # --------- Assert ----------
    assert 'well column ' + str(well_col_name) + ' not found in ascii trajectory file: ' + str(
        ascii_file) in caplog.text


def test_best_root_for_object(example_model_with_well):

    # --------- Arrange ----------
    model, well_interp, datum, traj = example_model_with_well
    traj_title = traj.title
    wellbore_marker_frame = resqpy.well.WellboreMarkerFrame(parent_model = model, trajectory = traj)
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

    # --------- Assert ----------
    assert citation_title1 == citation_title2 == citation_title3 == traj_title
    assert citation_title4 == citation_title5 == datum.title
