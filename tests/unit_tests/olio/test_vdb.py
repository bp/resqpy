import pytest

import resqpy.olio.vdb as vdb
import numpy as np


@pytest.mark.parametrize("byte_array, expected_array", [(b"\x20\x01\x00\x00\xae\x01\x00\x00", np.array([288, 430])),
                                                        (b"\xca\x02\x00\x00\xe8\x02\x00\x00", np.array([714, 744]))])
def test_raw_data_class_item_type_P(mocker, tmp_path, byte_array, expected_array):
    # Arrange
    place = 0
    item_type = 'P'
    count = 2
    max_count = 20

    open_mock = mocker.mock_open(read_data = byte_array)
    mocker.patch("builtins.open", open_mock)

    # Act
    with open(tmp_path) as fp:
        raw_data = vdb.RawData(fp, place, item_type, count, max_count)

    # Assert
    np.testing.assert_array_almost_equal(raw_data.a, expected_array)
