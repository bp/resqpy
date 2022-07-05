import pytest
import numpy as np
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


def test_load_wellspecs_single_well(tmp_path):
    # Arrange
    wellspec_file = f"{tmp_path}/test.dat"
    well = None
    column_list = []

    with open(wellspec_file, "w") as file:
        file.write("""
WELLSPEC TEST_WELL
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    4    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    5    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
            """)

    well_data = pd.DataFrame({
        "IW": [18, 18, 18, 18],
        "JW": [28, 28, 28, 28],
        "L": [2, 3, 4, 5],
        "KH": [np.nan, np.nan, np.nan, np.nan],
        "RADW": [0.32, 0.32, 0.32, 0.32],
        "SKIN": [0.0, 0.0, 0.0, 0.0],
        "RADB": [np.nan, np.nan, np.nan, np.nan],
        "WI": [np.nan, np.nan, np.nan, np.nan],
        "STAT": ["ON", "ON", "ON", "ON"],
        "LENGTH": [5.0, 5.0, 5.0, 5.0],
        "ANGLV": [88.08, 88.08, 88.08, 88.08],
        "ANGLA": [86.8, 86.8, 86.8, 86.8],
        "DEPTH": [9165.28, 9165.28, 9165.28, 9165.28],
    })

    # Act
    well_dict = wk.load_wellspecs(wellspec_file, well, column_list)

    # Assert
    assert len(well_dict) == 1
    pd.testing.assert_frame_equal(well_dict["TEST_WELL"], well_data)


def test_load_wellspecs_specific_well(tmp_path):
    # Arrange
    wellspec_file = f"{tmp_path}/test.dat"
    well = "TEST_WELL2"
    column_list = []

    with open(wellspec_file, "w") as file:
        file.write("""
WELLSPEC TEST_WELL1
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    4    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    5    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280

WELLSPEC TEST_WELL2
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
23    52    9    NA    0.260   0.000   NA      NA    ON      4.000     89.240   88.230   5492.600
23    52    10   NA    0.260   0.000   NA      NA    ON      4.000     89.240   88.230   5492.600
23    52    11   NA    0.260   0.000   NA      NA    ON      4.000     89.240   88.230   5492.600
23    52    12   NA    0.260   0.000   NA      NA    ON      4.000     89.240   88.230   5492.600
            """)

    well_data = pd.DataFrame({
        "IW": [23, 23, 23, 23],
        "JW": [52, 52, 52, 52],
        "L": [9, 10, 11, 12],
        "KH": [np.nan, np.nan, np.nan, np.nan],
        "RADW": [0.26, 0.26, 0.26, 0.26],
        "SKIN": [0.0, 0.0, 0.0, 0.0],
        "RADB": [np.nan, np.nan, np.nan, np.nan],
        "WI": [np.nan, np.nan, np.nan, np.nan],
        "STAT": ["ON", "ON", "ON", "ON"],
        "LENGTH": [4.0, 4.0, 4.0, 4.0],
        "ANGLV": [89.24, 89.24, 89.24, 89.24],
        "ANGLA": [88.23, 88.23, 88.23, 88.23],
        "DEPTH": [5492.6, 5492.6, 5492.6, 5492.6],
    })

    # Act
    well_dict = wk.load_wellspecs(wellspec_file, well, column_list)

    # Assert
    assert len(well_dict) == 1
    pd.testing.assert_frame_equal(well_dict[well], well_data)


def test_load_wellspecs_column_list(tmp_path):
    # Arrange
    wellspec_file = f"{tmp_path}/test.dat"
    well = None
    column_list = ["IW", "JW", "L", "LENGTH", "DEPTH"]

    with open(wellspec_file, "w") as file:
        file.write("""
WELLSPEC TEST_WELL
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    4    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    5    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
            """)

    well_data = pd.DataFrame({
        "IW": [18, 18, 18, 18],
        "JW": [28, 28, 28, 28],
        "L": [2, 3, 4, 5],
        "LENGTH": [5.0, 5.0, 5.0, 5.0],
        "DEPTH": [9165.28, 9165.28, 9165.28, 9165.28],
    })

    # Act
    well_dict = wk.load_wellspecs(wellspec_file, well, column_list)

    # Assert
    assert len(well_dict) == 1
    pd.testing.assert_frame_equal(well_dict["TEST_WELL"], well_data)


def test_load_wellspecs_column_list_none(tmp_path):
    # Arrange
    wellspec_file = f"{tmp_path}/test.dat"
    well = None
    column_list = None

    with open(wellspec_file, "w") as file:
        file.write("""
WELLSPEC TEST_WELL1
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    4    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    5    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280

WELLSPEC TEST_WELL2
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
23    52    9    NA    0.260   0.000   NA      NA    ON      4.000     89.240   88.230   5492.600
23    52    10   NA    0.260   0.000   NA      NA    ON      4.000     89.240   88.230   5492.600
23    52    11   NA    0.260   0.000   NA      NA    ON      4.000     89.240   88.230   5492.600
23    52    12   NA    0.260   0.000   NA      NA    ON      4.000     89.240   88.230   5492.600
            """)

    # Act
    well_dict = wk.load_wellspecs(wellspec_file, well, column_list)

    # Assert
    assert len(well_dict) == 2
    assert all([value is None for value in well_dict.values()])


def test_load_wellspecs_all_null(tmp_path):
    # Arrange
    wellspec_file = f"{tmp_path}/test.dat"
    well = None
    column_list = []

    with open(wellspec_file, "w") as file:
        file.write("""
WELLSPEC TEST_WELL
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
NA    NA    NA   NA    NA      NA      NA      NA    NA      NA        NA       NA       NA
NA    NA    NA   NA    NA      NA      NA      NA    NA      NA        NA       NA       NA
            """)

    # Act
    well_dict = wk.load_wellspecs(wellspec_file, well, column_list)

    # Assert
    assert well_dict == {}
