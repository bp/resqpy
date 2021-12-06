import os
import numpy as np
import pandas as pd

import resqpy.olio.xml_et as rqet
import resqpy.well
from resqpy.well.well_utils import load_hdf5_array, extract_xyz, find_entry_and_exit, _as_optional_array, _pl, well_names_in_cellio_file
from resqpy.grid import RegularGrid


def test_load_hdf5_array(example_model_with_well):

    # --------- Arrange ----------
    # Create a Deviation Survey object in memory

    # Load example model from a fixture
    model, well_interp, datum, traj = example_model_with_well

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
        inclinations = np.array([7, 8, 9], dtype = float),
        first_station = np.array([0, -1, 999], dtype = float),
    )

    survey = resqpy.well.DeviationSurvey(
        parent_model = model,
        represented_interp = well_interp,
        md_datum = datum,
        **data,
        **array_data,
    )

    # ----------- Act ---------
    survey.write_hdf5()
    survey.create_xml()
    mds_node = rqet.find_tag(survey.root, 'Mds', must_exist = True)
    # the only purpose of the call is to ensure that the array is cached in memory so should return None
    # a copy of the whole array is cached in memory as an attribute of the object
    expected_result = load_hdf5_array(survey, mds_node, 'measured_depths')

    # -------- Assert ---------
    np.testing.assert_equal(survey.__dict__['measured_depths'], array_data['measured_depths'])
    assert expected_result is None


def test_extract_xyz(example_model_with_well):

    # --------- Arrange ----------
    # Create a Deviation Survey object in memory

    # Load example model from a fixture
    model, well_interp, datum, traj = example_model_with_well

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
        inclinations = np.array([7, 8, 9], dtype = float),
        first_station = np.array([0, -1, 999], dtype = float),
    )

    survey = resqpy.well.DeviationSurvey(
        parent_model = model,
        represented_interp = well_interp,
        md_datum = datum,
        **data,
        **array_data,
    )

    # ----------- Act ---------
    survey.write_hdf5()
    survey.create_xml()
    first_station_node = rqet.find_tag(survey.root, 'FirstStationLocation', must_exist = True)
    first_station_xyz = extract_xyz(first_station_node)

    # ----------- Assert ---------
    np.testing.assert_equal(first_station_xyz, array_data['first_station'])


def test_find_entry_and_exit(example_model_and_crs):

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    grid = RegularGrid(model,
                       extent_kji = (5, 3, 3),
                       dxyz = (50.0, -50.0, 50.0),
                       origin = (0.0, 0.0, 100.0),
                       crs_uuid = crs.uuid,
                       set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
    well_name = 'DOGLEG'
    bw = resqpy.well.BlockedWell(model, well_name = well_name, use_face_centres = True, add_wellspec_properties = True)
    # populate empty blocked well object for a 'vertical' well in the given column
    bw.set_for_column(well_name = well_name, grid = grid, col_ji0 = (1, 1))
    cp = grid.corner_points(cell_kji0 = (1, 1, 1))
    cell_centre = np.mean(cp, axis = (0, 1, 2))
    entry_xyz = np.array([25, -25, 125])
    exit_xyz = np.array([75, -50, 175])
    entry_vector = 100.0 * (entry_xyz - cell_centre)
    exit_vector = 100.0 * (exit_xyz - cell_centre)

    # --------- Act ----------
    (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz) =\
        find_entry_and_exit(cp=cp, entry_vector=entry_vector, exit_vector=exit_vector, well_name=well_name)

    # --------- Assert ----------
    # TODO: less trivial assertion statement, understand the math behind the calcs
    for result in (entry_axis, entry_polarity, entry_xyz, exit_axis, exit_polarity, exit_xyz):
        assert result is not None


def test_as_optional_array():

    # --------- Arrange ----------
    to_test = [[1, 2, 3], None]

    # --------- Act ----------
    result1, result2 = _as_optional_array(to_test[0]), _as_optional_array(to_test[1])

    # --------- Assert ----------
    np.testing.assert_equal(result1, np.array([1, 2, 3]))
    assert result2 is None


def test_pl():

    # --------- Arrange ----------
    to_test = [1, 'Banoffee']

    # --------- Act ----------
    result1, result2, result3 = _pl(i = to_test[0]), _pl(i = to_test[1]), _pl(i = to_test[1], e = True)

    # --------- Assert ----------
    assert result1 == ''
    assert result2 == 's'
    assert result3 == 'es'


def test_well_names_in_cellio_file(tmp_path):

    # --------- Arrange ----------
    well_name = 'Banoffee'
    cellio_file = os.path.join(tmp_path, 'cellio.dat')
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

    # --------- Arrange ----------
    well_list = well_names_in_cellio_file(cellio_file = cellio_file)

    # --------- Assert ----------
    assert len(well_list) == 1
    assert set(well_list) == {well_name}
