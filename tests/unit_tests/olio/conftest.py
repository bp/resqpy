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
        "IW": np.array([18, 18, 18, 18], dtype = np.int32),
        "JW": np.array([28, 28, 28, 28], dtype = np.int32),
        "L": np.array([2, 3, 4, 5], dtype = np.int32),
        "KH": [np.nan, np.nan, np.nan, np.nan],
        "RADW": [0.32, 0.32, 0.32, 0.32],
        "SKIN": [0.0, 0.0, 0.0, 0.0],
        "RADB": [np.nan, np.nan, np.nan, np.nan],
        "WI": [np.nan, np.nan, np.nan, np.nan],
        "STAT": np.array([1, 1, 1, 1], dtype = np.int8),
        "LENGTH": [5.0, 5.0, 5.0, 5.0],
        "ANGLV": [88.08, 88.08, 88.08, 88.08],
        "ANGLA": [86.8, 86.8, 86.8, 86.8],
        "DEPTH": [9165.28, 9165.28, 9165.28, 9165.28],
    })


@pytest.fixture()
def test_well2_dataframe() -> pd.DataFrame:

    return pd.DataFrame({
        "IW": np.array([23, 23, 23, 23], dtype = np.int32),
        "JW": np.array([52, 52, 52, 52], dtype = np.int32),
        "L": np.array([9, 10, 11, 12], dtype = np.int32),
        "KH": [np.nan, np.nan, np.nan, np.nan],
        "RADW": [0.26, 0.26, 0.26, 0.26],
        "SKIN": [0.0, 0.0, 0.0, 0.0],
        "RADB": [np.nan, np.nan, np.nan, np.nan],
        "WI": [np.nan, np.nan, np.nan, np.nan],
        "STAT": np.array([1, 1, 1, 1], dtype = np.int8),
        "LENGTH": [4.0, 4.0, 4.0, 4.0],
        "ANGLV": [89.24, 89.24, 89.24, 89.24],
        "ANGLA": [88.23, 88.23, 88.23, 88.23],
        "DEPTH": [5492.6, 5492.6, 5492.6, 5492.6],
    })


@pytest.fixture()
def test_well_dataframe_duplicates_removed() -> pd.DataFrame:

    return pd.DataFrame({
        "IW": {
            0: np.int32(18),
            2: np.int32(18),
            3: np.int32(18),
            4: np.int32(18),
            5: np.int32(19)
        },
        "JW": {
            0: np.int32(28),
            2: np.int32(28),
            3: np.int32(28),
            4: np.int32(28),
            5: np.int32(28)
        },
        "L": {
            0: np.int32(2),
            2: np.int32(3),
            3: np.int32(4),
            4: np.int32(5),
            5: np.int32(5)
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
            0: np.int8(1),
            2: np.int8(0),
            3: np.int8(1),
            4: np.int8(0),
            5: np.int8(1)
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
            0: np.int32(18),
            1: np.int32(18),
            2: np.int32(18),
            3: np.int32(18),
            4: np.int32(18),
            5: np.int32(19)
        },
        "JW": {
            0: np.int32(28),
            1: np.int32(28),
            2: np.int32(28),
            3: np.int32(28),
            4: np.int32(28),
            5: np.int32(28)
        },
        "L": {
            0: np.int32(2),
            1: np.int32(3),
            2: np.int32(3),
            3: np.int32(4),
            4: np.int32(5),
            5: np.int32(5)
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
            0: np.int8(1),
            1: np.int8(1),
            2: np.int8(0),
            3: np.int8(1),
            4: np.int8(0),
            5: np.int8(1)
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
        "IW": np.array([18, 18, 18, 18], dtype = np.int32),
        "JW": np.array([28, 28, 28, 28], dtype = np.int32),
        "L": np.array([2, 3, 4, 5], dtype = np.int32),
        "RADW": [0.32, 0.32, 0.32, 0.32],
        "SKIN": [0.0, 0.0, 0.0, 0.0],
        "STAT": np.array([1, 1, 1, 1], dtype = np.int8),
        "LENGTH": [5.0, 5.0, 5.0, 5.0],
        "ANGLV": [88.08, 88.08, 88.08, 88.08],
        "ANGLA": [86.8, 86.8, 86.8, 86.8],
        "DEPTH": [9165.28, 9165.28, 9165.28, 9165.28],
    })


@pytest.fixture()
def test_well_dataframe_last_data_only() -> pd.DataFrame:

    return pd.DataFrame({
        "IW": {
            0: np.int32(18),
            1: np.int32(18)
        },
        "JW": {
            0: np.int32(28),
            1: np.int32(28)
        },
        "L": {
            0: np.int32(2),
            1: np.int32(3)
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
            0: np.int8(1),
            1: np.int8(1)
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
            0: np.int32(18),
            1: np.int32(18),
            2: np.int32(18),
            3: np.int32(18)
        },
        "JW": {
            0: np.int32(28),
            1: np.int32(28),
            2: np.int32(28),
            3: np.int32(28)
        },
        "L": {
            0: np.int32(2),
            1: np.int32(3),
            2: np.int32(2),
            3: np.int32(3)
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
            0: np.int8(1),
            1: np.int8(1),
            2: np.int8(1),
            3: np.int8(1)
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
        "DATE": {
            0: None,
            1: None,
            2: "1994-03-12",
            3: "1994-03-12",
        },
    })
