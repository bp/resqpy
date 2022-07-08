import pytest
import pandas as pd
import resqpy.olio.wellspec_keywords as wk


def test_check_value_unknown_keyword():
    # Arrange
    keyword = "UNKNOWN"
    value = 0.0

    boolean_return_expected = False

    # Act
    boolean_return = wk.check_value(keyword, value)

    # Assert
    assert boolean_return is boolean_return_expected


@pytest.mark.parametrize(
    "keyword, value, boolean_return_expected",
    [
        ("IW", 0.0, False),
        ("JW", 1.0, True),
        ("GRID", "Test", True),
        ("STAT", "On", True),
        ("STAT", 0.0, False),
        ("ANGLA", 0.0, True),
        ("ANGLA", 400.0, False),
        ("ANGLV", 90.0, True),
        ("ANGLV", -90.0, False),
        ("RADW", 0.0, False),
        ("WI", 0.0, True),
        ("PPERF", 0.0, True),
        ("ANGLE", 0.0, True),
        ("ANGLE", -180.0, False),
        ("SKIN", 0.0, True),
        ("SKIN", "Test", False),
    ],
)
def test_check_value_known_keyword(keyword, value, boolean_return_expected):
    # Act
    boolean_return = wk.check_value(keyword, value)

    # Assert
    assert boolean_return is boolean_return_expected


def test_load_wellspecs_single_well(wellspec_file_one_well, test_well_dataframe):
    # Arrange
    well = None

    # Act
    well_dict = wk.load_wellspecs(wellspec_file_one_well, well)

    # Assert
    assert len(well_dict) == 1
    pd.testing.assert_frame_equal(well_dict["TEST_WELL"], test_well_dataframe)


def test_load_wellspecs_specific_well(wellspec_file_two_wells, test_well2_dataframe):
    # Arrange
    well = "TEST_WELL2"

    # Act
    well_dict = wk.load_wellspecs(wellspec_file_two_wells, well)

    # Assert
    assert len(well_dict) == 1
    pd.testing.assert_frame_equal(well_dict[well], test_well2_dataframe)


def test_load_wellspecs_column_list(wellspec_file_one_well, test_well_dataframe):
    # Arrange
    column_list = ["IW", "JW", "L", "LENGTH", "DEPTH"]

    well_data = test_well_dataframe[column_list]

    # Act
    well_dict = wk.load_wellspecs(wellspec_file_one_well, column_list = column_list)

    # Assert
    assert len(well_dict) == 1
    pd.testing.assert_frame_equal(well_dict["TEST_WELL"], well_data)


def test_load_wellspecs_column_list_none(wellspec_file_two_wells):
    # Arrange
    column_list = None

    # Act
    well_dict = wk.load_wellspecs(wellspec_file_two_wells, column_list = column_list)

    # Assert
    assert len(well_dict) == 2
    assert all([value is None for value in well_dict.values()])


def test_load_wellspecs_all_null(wellspec_file_null_well):
    # Act
    well_dict = wk.load_wellspecs(wellspec_file_null_well)

    # Assert
    assert well_dict == {}


def test_get_well_pointers(wellspec_file_two_wells):
    # Act
    well_pointers = wk.get_well_pointers(wellspec_file_two_wells)

    # Assert
    assert len(well_pointers) == 2
    assert well_pointers["TEST_WELL1"] == 21
    assert well_pointers["TEST_WELL2"] == 529


def test_get_well_data(wellspec_file_one_well, test_well_dataframe):
    # Arrange
    well_name = "TEST_WELL"
    pointer = 20

    # Act
    with open(wellspec_file_one_well, "r") as file:
        well_data = wk.get_well_data(
            file,
            well_name,
            pointer,
        )

    # Assert
    pd.testing.assert_frame_equal(well_data, test_well_dataframe)


def test_get_well_data_duplicates(wellspec_file_duplicates, test_well_dataframe_duplicates_removed):
    # Arrange
    well_name = "TEST_WELL"
    pointer = 20
    keep_duplicate_cells = False

    # Act
    with open(wellspec_file_duplicates, "r") as file:
        well_data = wk.get_well_data(file, well_name, pointer, keep_duplicate_cells = keep_duplicate_cells)

    # Assert
    pd.testing.assert_frame_equal(well_data, test_well_dataframe_duplicates_removed)


def test_get_well_data_keep_duplicates(wellspec_file_duplicates, test_well_dataframe_duplicates_kept):
    # Arrange
    well_name = "TEST_WELL"
    pointer = 20

    # Act
    with open(wellspec_file_duplicates, "r") as file:
        well_data = wk.get_well_data(file, well_name, pointer)

    # Assert
    pd.testing.assert_frame_equal(well_data, test_well_dataframe_duplicates_kept)


def test_get_well_data_keep_null_columns_false(wellspec_file_one_well, test_well_dataframe_null_columns_dropped):
    # Arrange
    well_name = "TEST_WELL"
    pointer = 20
    keep_null_columns = False

    # Act
    with open(wellspec_file_one_well, "r") as file:
        well_data = wk.get_well_data(file, well_name, pointer, keep_null_columns = False)

    # Assert
    pd.testing.assert_frame_equal(well_data, test_well_dataframe_null_columns_dropped)