import os
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd
import pytest

from resqpy.model import Model
import resqpy.well


def test_DeviationSurvey(example_model_with_well, tmp_path):
    # Test that all attributes are correctly saved and loaded from disk

    # --------- Arrange ----------
    # Create a Deviation Survey object in memory

    # Load example model from a fixture
    model, well_interp, datum, traj = example_model_with_well
    epc_path = model.epc_file

    # Create 3 copies of the survey, using different initialisers
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
    survey_uuid = survey.uuid

    df = pd.DataFrame(columns = ['MD', 'AZIM_GN', 'INCL', 'X', 'Y', 'Z'])
    for col, key in zip(('MD', 'AZIM_GN', 'INCL'), ('measured_depths', 'azimuths', 'inclinations')):
        df[col] = array_data[key]
    for axis, col in enumerate(('X', 'Y', 'Z')):
        df[col] = np.NaN
        df.loc[0, col] = array_data['first_station'][axis]

    survey_b = resqpy.well.DeviationSurvey.from_data_frame(parent_model = model,
                                                           data_frame = df,
                                                           md_datum = datum,
                                                           md_uom = data['md_uom'],
                                                           angle_uom = data['angle_uom'])
    survey_b_uuid = survey_b.uuid

    csv_file = os.path.join(tmp_path, 'survey_c.csv')
    df.to_csv(csv_file)

    survey_c = resqpy.well.DeviationSurvey.from_ascii_file(parent_model = model,
                                                           deviation_survey_file = csv_file,
                                                           md_datum = datum,
                                                           md_uom = data['md_uom'],
                                                           angle_uom = data['angle_uom'])
    survey_c_uuid = survey_c.uuid

    # ----------- Act ---------

    # Save to disk
    for s in (survey, survey_b, survey_c):
        s.write_hdf5()
        s.create_xml()
    model.store_epc()
    model.h5_release()

    # Clear memory
    del model, well_interp, datum, traj, survey, survey_b, survey_c

    # Reload from disk
    model2 = Model(epc_file = epc_path)
    survey2 = resqpy.well.DeviationSurvey(model2, uuid = survey_uuid)
    survey_b2 = resqpy.well.DeviationSurvey(model2, uuid = survey_b_uuid)
    survey_c2 = resqpy.well.DeviationSurvey(model2, uuid = survey_c_uuid)

    # --------- Assert --------------
    # Check all attributes were loaded from disk correctly

    for key, expected_value in data.items():
        assert getattr(survey2, key) == expected_value, f"Error for {key}"
        if 'uom' in key:
            for s in (survey_b2, survey_c2):
                assert getattr(s, key) == expected_value, f"Error for {key}"

    for s in (survey2, survey_b2, survey_c2):
        for key, expected_value in array_data.items():
            assert_array_almost_equal(getattr(s, key), expected_value, err_msg = f"Error for {key}")
        assert s.station_count == len(array_data['azimuths'])


def test_get_md_datum(example_model_with_well):
    # Test that a ValueError is raised if a valid Md Datum root node has not been provided and cannot be created.

    # --------- Arrange ----------
    # Create a Deviation Survey object in memory

    # Load example model from a fixture
    model, well_interp, datum, traj = example_model_with_well
    epc_path = model.epc_file

    # Create a survey
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
        md_datum = None,
        **data,
        **array_data,
    )

    # ----------- Act ---------
    survey.write_hdf5()
    with pytest.raises(ValueError) as excinfo:
        survey._DeviationSurvey__get_md_datum_root(md_datum_root = None, md_datum_xyz = None)

    # -------- Assert ---------
    assert "Must provide a MD Datum for the DeviationSurvey" in str(excinfo.value)
