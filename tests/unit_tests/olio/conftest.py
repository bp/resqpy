import pytest
import numpy as np
import pandas as pd


@pytest.fixture()
def wellspec_file_one_well(tmp_path) -> str:
    wellspec_file = f"{tmp_path}/test.dat"

    with open(wellspec_file, "w") as file:
        file.write("""
WELLSPEC TEST_WELL
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    4    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    5    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
            """)

    return wellspec_file


@pytest.fixture()
def wellspec_file_invalid_date(tmp_path) -> str:
    wellspec_file = f"{tmp_path}/test.dat"

    with open(wellspec_file, "w") as file:
        file.write("""
TIME 1974/11/17

WELLSPEC TEST_WELL
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    4    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    5    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
            """)

    return wellspec_file


@pytest.fixture()
def wellspec_file_two_wells(tmp_path) -> str:
    wellspec_file = f"{tmp_path}/test.dat"

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

    return wellspec_file


@pytest.fixture()
def wellspec_file_multiple_wells(tmp_path) -> str:
    wellspec_file = f"{tmp_path}/test.dat"

    with open(wellspec_file, "w") as file:
        file.write("""
WELLSPEC TEST_WELL1
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280

WELLSPEC TEST_WELL2
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280

TIME 12/03/1993

WELLSPEC TEST_WELL3
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280

TIME 12/03/1994

WELLSPEC TEST_WELL4
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280

WELLSPEC TEST_WELL5
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280

WELLSPEC TEST_WELL2
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280

TIME 12/03/1995
            """)

    return wellspec_file


@pytest.fixture()
def wellspec_file_null_well(tmp_path) -> str:
    wellspec_file = f"{tmp_path}/test.dat"

    with open(wellspec_file, "w") as file:
        file.write("""
WELLSPEC TEST_WELL
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
NA    NA    NA   NA    NA      NA      NA      NA    NA      NA        NA       NA       NA
NA    NA    NA   NA    NA      NA      NA      NA    NA      NA        NA       NA       NA
            """)

    return wellspec_file


@pytest.fixture()
def wellspec_file_duplicates(tmp_path) -> str:
    wellspec_file = f"{tmp_path}/test.dat"

    with open(wellspec_file, "w") as file:
        file.write("""
WELLSPEC TEST_WELL
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     89.020   86.800   9128.940
18    28    3    NA    0.320   0.000   NA      NA    OFF     5.000     88.080   86.800   9165.280
18    28    4    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    5    NA    0.320   0.000   NA      NA    OFF     5.000     88.080   86.800   9165.280
19    28    5    NA    0.320   0.000   NA      NA    ON      5.000     88.720   86.800   9193.400
            """)

    return wellspec_file


@pytest.fixture()
def test_well_dataframe() -> pd.DataFrame:

    return pd.DataFrame({
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


@pytest.fixture()
def test_well2_dataframe() -> pd.DataFrame:

    return pd.DataFrame({
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


@pytest.fixture()
def test_well_dataframe_duplicates_removed() -> pd.DataFrame:

    return pd.DataFrame({
        "IW": {
            0: 18,
            2: 18,
            3: 18,
            4: 18,
            5: 19,
        },
        "JW": {
            0: 28,
            2: 28,
            3: 28,
            4: 28,
            5: 28
        },
        "L": {
            0: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 5
        },
        "KH": {
            0: np.nan,
            2: np.nan,
            3: np.nan,
            4: np.nan,
            5: np.nan
        },
        "RADW": {
            0: 0.32,
            2: 0.32,
            3: 0.32,
            4: 0.32,
            5: 0.32
        },
        "SKIN": {
            0: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
            5: 0.0
        },
        "RADB": {
            0: np.nan,
            2: np.nan,
            3: np.nan,
            4: np.nan,
            5: np.nan
        },
        "WI": {
            0: np.nan,
            2: np.nan,
            3: np.nan,
            4: np.nan,
            5: np.nan
        },
        "STAT": {
            0: "ON",
            2: "OFF",
            3: "ON",
            4: "OFF",
            5: "ON"
        },
        "LENGTH": {
            0: 5.0,
            2: 5.0,
            3: 5.0,
            4: 5.0,
            5: 5.0
        },
        "ANGLV": {
            0: 88.08,
            2: 88.08,
            3: 88.08,
            4: 88.08,
            5: 88.72
        },
        "ANGLA": {
            0: 86.8,
            2: 86.8,
            3: 86.8,
            4: 86.8,
            5: 86.8
        },
        "DEPTH": {
            0: 9165.28,
            2: 9165.28,
            3: 9165.28,
            4: 9165.28,
            5: 9193.4
        },
    })


@pytest.fixture()
def test_well_dataframe_duplicates_kept() -> pd.DataFrame:

    return pd.DataFrame({
        "IW": {
            0: 18,
            1: 18,
            2: 18,
            3: 18,
            4: 18,
            5: 19
        },
        "JW": {
            0: 28,
            1: 28,
            2: 28,
            3: 28,
            4: 28,
            5: 28
        },
        "L": {
            0: 2,
            1: 3,
            2: 3,
            3: 4,
            4: 5,
            5: 5
        },
        "KH": {
            0: np.nan,
            1: np.nan,
            2: np.nan,
            3: np.nan,
            4: np.nan,
            5: np.nan
        },
        "RADW": {
            0: 0.32,
            1: 0.32,
            2: 0.32,
            3: 0.32,
            4: 0.32,
            5: 0.32
        },
        "SKIN": {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
            5: 0.0
        },
        "RADB": {
            0: np.nan,
            1: np.nan,
            2: np.nan,
            3: np.nan,
            4: np.nan,
            5: np.nan
        },
        "WI": {
            0: np.nan,
            1: np.nan,
            2: np.nan,
            3: np.nan,
            4: np.nan,
            5: np.nan
        },
        "STAT": {
            0: "ON",
            1: "ON",
            2: "OFF",
            3: "ON",
            4: "OFF",
            5: "ON"
        },
        "LENGTH": {
            0: 5.0,
            1: 5.0,
            2: 5.0,
            3: 5.0,
            4: 5.0,
            5: 5.0
        },
        "ANGLV": {
            0: 88.08,
            1: 89.02,
            2: 88.08,
            3: 88.08,
            4: 88.08,
            5: 88.72
        },
        "ANGLA": {
            0: 86.8,
            1: 86.8,
            2: 86.8,
            3: 86.8,
            4: 86.8,
            5: 86.8
        },
        "DEPTH": {
            0: 9165.28,
            1: 9128.94,
            2: 9165.28,
            3: 9165.28,
            4: 9165.28,
            5: 9193.4,
        },
    })


@pytest.fixture()
def test_well_dataframe_null_columns_dropped() -> pd.DataFrame:

    return pd.DataFrame({
        "IW": [18, 18, 18, 18],
        "JW": [28, 28, 28, 28],
        "L": [2, 3, 4, 5],
        "RADW": [0.32, 0.32, 0.32, 0.32],
        "SKIN": [0.0, 0.0, 0.0, 0.0],
        "STAT": ["ON", "ON", "ON", "ON"],
        "LENGTH": [5.0, 5.0, 5.0, 5.0],
        "ANGLV": [88.08, 88.08, 88.08, 88.08],
        "ANGLA": [86.8, 86.8, 86.8, 86.8],
        "DEPTH": [9165.28, 9165.28, 9165.28, 9165.28],
    })


@pytest.fixture()
def test_well_dataframe_last_data_only() -> pd.DataFrame:

    return pd.DataFrame({
        "IW": {
            0: 18,
            1: 18
        },
        "JW": {
            0: 28,
            1: 28
        },
        "L": {
            0: 2,
            1: 3
        },
        "KH": {
            0: np.nan,
            1: np.nan
        },
        "RADW": {
            0: 0.32,
            1: 0.32
        },
        "SKIN": {
            0: 0.0,
            1: 0.0
        },
        "RADB": {
            0: np.nan,
            1: np.nan
        },
        "WI": {
            0: np.nan,
            1: np.nan
        },
        "STAT": {
            0: "ON",
            1: "ON"
        },
        "LENGTH": {
            0: 5.0,
            1: 5.0
        },
        "ANGLV": {
            0: 88.08,
            1: 88.08
        },
        "ANGLA": {
            0: 86.8,
            1: 86.8
        },
        "DEPTH": {
            0: 9165.28,
            1: 9165.28
        },
    })


@pytest.fixture()
def test_well_dataframe_all_data() -> pd.DataFrame:

    return pd.DataFrame({
        "IW": {
            0: 18,
            1: 18,
            2: 18,
            3: 18
        },
        "JW": {
            0: 28,
            1: 28,
            2: 28,
            3: 28
        },
        "L": {
            0: 2,
            1: 3,
            2: 2,
            3: 3
        },
        "KH": {
            0: np.nan,
            1: np.nan,
            2: np.nan,
            3: np.nan
        },
        "RADW": {
            0: 0.32,
            1: 0.32,
            2: 0.32,
            3: 0.32
        },
        "SKIN": {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.0
        },
        "RADB": {
            0: np.nan,
            1: np.nan,
            2: np.nan,
            3: np.nan
        },
        "WI": {
            0: np.nan,
            1: np.nan,
            2: np.nan,
            3: np.nan
        },
        "STAT": {
            0: "ON",
            1: "ON",
            2: "ON",
            3: "ON"
        },
        "LENGTH": {
            0: 5.0,
            1: 5.0,
            2: 5.0,
            3: 5.0
        },
        "ANGLV": {
            0: 88.08,
            1: 88.08,
            2: 88.08,
            3: 88.08
        },
        "ANGLA": {
            0: 86.8,
            1: 86.8,
            2: 86.8,
            3: 86.8
        },
        "DEPTH": {
            0: 9165.28,
            1: 9165.28,
            2: 9165.28,
            3: 9165.28
        },
        "Date": {
            0: None,
            1: None,
            2: "1994-03-12T00:00:00",
            3: "1994-03-12T00:00:00",
        },
    })
