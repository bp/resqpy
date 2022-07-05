from aiohttp import streamer
import pytest
import numpy as np
import pandas as pd


@pytest.fixture()
def wellspec_file_one_well(tmp_path) -> str:
    wellspec_file = f"{tmp_path}/test.dat"

    with open(wellspec_file, "w") as file:
        file.write(
            """
WELLSPEC TEST_WELL
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
18    28    2    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    3    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    4    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
18    28    5    NA    0.320   0.000   NA      NA    ON      5.000     88.080   86.800   9165.280
            """
        )

    return wellspec_file


@pytest.fixture()
def wellspec_file_two_wells(tmp_path) -> str:
    wellspec_file = f"{tmp_path}/test.dat"

    with open(wellspec_file, "w") as file:
        file.write(
            """
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
            """
        )

    return wellspec_file


@pytest.fixture()
def wellspec_file_null_well(tmp_path) -> str:
    wellspec_file = f"{tmp_path}/test.dat"

    with open(wellspec_file, "w") as file:
        file.write(
            """
WELLSPEC TEST_WELL
IW    JW    L    KH    RADW    SKIN    RADB    WI    STAT    LENGTH    ANGLV    ANGLA    DEPTH
NA    NA    NA   NA    NA      NA      NA      NA    NA      NA        NA       NA       NA
NA    NA    NA   NA    NA      NA      NA      NA    NA      NA        NA       NA       NA
            """
        )

    return wellspec_file


@pytest.fixture()
def test_well_dataframe() -> pd.DataFrame:

    return pd.DataFrame(
        {
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
        }
    )


@pytest.fixture()
def test_well2_dataframe() -> pd.DataFrame:

    return pd.DataFrame(
        {
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
        }
    )
