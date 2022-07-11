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


def test_load_wellspecs_remove_duplicate_cells(wellspec_file_duplicates, test_well_dataframe_duplicates_removed):
    # Act
    well_dict = wk.load_wellspecs(wellspec_file_duplicates, keep_duplicate_cells = False)

    # Assert
    assert len(well_dict) == 1
    pd.testing.assert_frame_equal(well_dict["TEST_WELL"], test_well_dataframe_duplicates_removed)


def test_load_wellspecs_keep_duplicate_cells(wellspec_file_duplicates, test_well_dataframe_duplicates_kept):
    # Act
    well_dict = wk.load_wellspecs(wellspec_file_duplicates, keep_duplicate_cells = True)

    # Assert
    assert len(well_dict) == 1
    pd.testing.assert_frame_equal(well_dict["TEST_WELL"], test_well_dataframe_duplicates_kept)


def test_load_wellspecs_all_data(wellspec_file_multiple_wells, test_well_dataframe_all_data):
    # Arrange
    well = "TEST_WELL2"

    # Act
    well_dict = wk.load_wellspecs(wellspec_file_multiple_wells, well = well, last_data_only = False)

    # Assert
    assert len(well_dict) == 1
    pd.testing.assert_frame_equal(well_dict[well], test_well_dataframe_all_data)


def test_load_wellspecs_last_data_only(wellspec_file_multiple_wells, test_well_dataframe_last_data_only):
    # Arrange
    well = "TEST_WELL2"

    # Act
    well_dict = wk.load_wellspecs(wellspec_file_multiple_wells, well = well, last_data_only = True)

    # Assert
    assert len(well_dict) == 1
    pd.testing.assert_frame_equal(well_dict[well], test_well_dataframe_last_data_only)


def test_load_wellspecs_all_null(wellspec_file_null_well):
    # Act
    well_dict = wk.load_wellspecs(wellspec_file_null_well)

    # Assert
    assert well_dict == {}


def test_get_well_pointers(wellspec_file_two_wells):
    # Act
    well_pointers = wk.get_well_pointers(wellspec_file_two_wells)

    # Assert
    assert well_pointers == {"TEST_WELL1": [(21, None)], "TEST_WELL2": [(529, None)]}


@pytest.mark.parametrize(
    "usa_date_format, well_pointers_expected",
    [
        (
            False,
            {
                "TEST_WELL1": [(21, None)],
                "TEST_WELL2": [(333, None), (1615, "1994-03-12T00:00:00")],
                "TEST_WELL3": [(662, "1993-03-12T00:00:00")],
                "TEST_WELL4": [(991, "1994-03-12T00:00:00")],
                "TEST_WELL5": [(1303, "1994-03-12T00:00:00")],
            },
        ),
        (
            True,
            {
                "TEST_WELL1": [(21, None)],
                "TEST_WELL2": [(333, None), (1615, "1994-12-03T00:00:00")],
                "TEST_WELL3": [(662, "1993-12-03T00:00:00")],
                "TEST_WELL4": [(991, "1994-12-03T00:00:00")],
                "TEST_WELL5": [(1303, "1994-12-03T00:00:00")],
            },
        ),
    ],
)
def test_get_well_pointers_multiple_with_times(wellspec_file_multiple_wells, usa_date_format, well_pointers_expected):
    # Act
    well_pointers = wk.get_well_pointers(wellspec_file_multiple_wells, usa_date_format = usa_date_format)

    # Assert
    print(well_pointers)
    assert well_pointers == well_pointers_expected


def test_get_well_pointers_invalid_date(wellspec_file_invalid_date):
    # Act & Assert
    with pytest.raises(ValueError):
        well_pointers = wk.get_well_pointers(wellspec_file_invalid_date)


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

    # Act
    with open(wellspec_file_duplicates, "r") as file:
        well_data = wk.get_well_data(file, well_name, pointer, keep_duplicate_cells = False)

    # Assert
    pd.testing.assert_frame_equal(well_data, test_well_dataframe_duplicates_removed)


def test_get_well_data_keep_duplicate_cells(wellspec_file_duplicates, test_well_dataframe_duplicates_kept):
    # Arrange
    well_name = "TEST_WELL"
    pointer = 20

    # Act
    with open(wellspec_file_duplicates, "r") as file:
        well_data = wk.get_well_data(file, well_name, pointer)

    # Assert
    pd.testing.assert_frame_equal(well_data, test_well_dataframe_duplicates_kept)


def test_get_well_data_remove_duplicate_cells(wellspec_file_duplicates, test_well_dataframe_duplicates_removed):
    # Arrange
    well_name = "TEST_WELL"
    pointer = 20

    # Act
    with open(wellspec_file_duplicates, "r") as file:
        well_data = wk.get_well_data(file, well_name, pointer, keep_duplicate_cells = False)

    # Assert
    pd.testing.assert_frame_equal(well_data, test_well_dataframe_duplicates_removed)


def test_get_well_data_keep_null_columns_false(wellspec_file_one_well, test_well_dataframe_null_columns_dropped):
    # Arrange
    well_name = "TEST_WELL"
    pointer = 20
    keep_null_columns = False

    # Act
    with open(wellspec_file_one_well, "r") as file:
        well_data = wk.get_well_data(file, well_name, pointer, keep_null_columns = keep_null_columns)

    # Assert
    pd.testing.assert_frame_equal(well_data, test_well_dataframe_null_columns_dropped)


def test_get_all_well_data_single_well(wellspec_file_one_well, test_well_dataframe):
    # Arrange
    well = None
    pointers = [(20, None)]

    # Act
    with open(wellspec_file_one_well, "r") as file:
        df = wk.get_all_well_data(file, well, pointers)

    # Assert
    pd.testing.assert_frame_equal(df, test_well_dataframe)


def test_get_all_well_data_specific_well(wellspec_file_two_wells, test_well2_dataframe):
    # Arrange
    well = "TEST_WELL2"
    pointers = [(20, None), (529, None)]

    # Act
    with open(wellspec_file_two_wells, "r") as file:
        df = wk.get_all_well_data(file, well, pointers)

    # Assert
    pd.testing.assert_frame_equal(df, test_well2_dataframe)


def test_get_all_well_data_column_list(wellspec_file_one_well, test_well_dataframe):
    # Arrange
    well = None
    pointers = [(20, None)]
    column_list = ["IW", "JW", "L", "LENGTH", "DEPTH"]

    well_data = test_well_dataframe[column_list]

    # Act
    with open(wellspec_file_one_well, "r") as file:
        df = wk.get_all_well_data(file, well, pointers, column_list = column_list, selecting = True)

    # Assert
    pd.testing.assert_frame_equal(df, well_data)


def test_get_all_well_data_column_list_none(wellspec_file_two_wells):
    # Arrange
    well = "TEST_WELL1"
    pointers = [(20, None)]
    column_list = None

    # Act
    with open(wellspec_file_two_wells, "r") as file:
        df = wk.get_all_well_data(file, well, pointers, column_list = column_list)

    # Assert
    assert df is None


def test_get_all_well_data_remove_duplicate_cells(wellspec_file_duplicates, test_well_dataframe_duplicates_removed):
    # Arrange
    well = "TEST_WELL"
    pointers = [(20, None)]

    # Act
    with open(wellspec_file_duplicates, "r") as file:
        df = wk.get_all_well_data(file, well, pointers, keep_duplicate_cells = False)

    # Assert
    pd.testing.assert_frame_equal(df, test_well_dataframe_duplicates_removed)


def test_get_all_well_data_keep_duplicate_cells(wellspec_file_duplicates, test_well_dataframe_duplicates_kept):
    # Arrange
    well = "TEST_WELL"
    pointers = [(20, None)]

    # Act
    with open(wellspec_file_duplicates, "r") as file:
        df = wk.get_all_well_data(file, well, pointers, keep_duplicate_cells = True)

    # Assert
    pd.testing.assert_frame_equal(df, test_well_dataframe_duplicates_kept)


def test_get_all_well_data_all_data(wellspec_file_multiple_wells, test_well_dataframe_all_data):
    # Arrange
    well = "TEST_WELL2"
    pointers = [(333, None), (1615, "1994-03-12T00:00:00")]

    # Act
    with open(wellspec_file_multiple_wells, "r") as file:
        df = wk.get_all_well_data(file, well, pointers, last_data_only = False)

    # Assert
    pd.testing.assert_frame_equal(df, test_well_dataframe_all_data)


def test_get_all_well_data_last_data_only(wellspec_file_multiple_wells, test_well_dataframe_last_data_only):
    # Arrange
    well = "TEST_WELL2"
    pointers = [(333, None), (1615, "1994-03-12T00:00:00")]

    # Act
    with open(wellspec_file_multiple_wells, "r") as file:
        df = wk.get_all_well_data(file, well, pointers, last_data_only = True)

    # Assert
    pd.testing.assert_frame_equal(df, test_well_dataframe_last_data_only)


def test_get_all_well_data_all_null(wellspec_file_null_well):
    # Arrange
    well = "TEST_WELL"
    pointers = [(20, None)]

    # Act
    with open(wellspec_file_null_well, "r") as file:
        df = wk.get_all_well_data(file, well, pointers)

    # Assert
    assert df is None
