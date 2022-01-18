import numpy as np
import pytest
import resqpy.olio.write_data as wd
from pytest_mock import MockerFixture


def test_write_pure_binary_data(mocker: MockerFixture, caplog):
    # Arrange
    test_array = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    open_mock = mocker.mock_open()
    fileno_mock = mocker.Mock(return_value = 1, name = 'fileno_mock')
    open_mock.return_value.fileno = fileno_mock

    mocker.patch("builtins.open", open_mock)
    expected_calls = [mocker.call('test', 'wb'), mocker.call().__enter__()]

    # Act
    wd.write_pure_binary_data('test', test_array)

    # Assert
    open_mock.assert_has_calls(expected_calls)
    assert 'Binary data file test created' in caplog.text


def test_write_array_to_ascii_file_decimals_equal_0(mocker: MockerFixture):
    # Arrange
    test_extent = np.array([2, 2, 2])
    test_array = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    open_mock = mocker.mock_open()
    mocker.patch("builtins.open", open_mock)
    expected_calls = [
        mocker.call('test', 'w'),
        mocker.call().__enter__(),
        mocker.call().write('! Data written by write_array_to_ascii_file() python function\n'),
        mocker.call().write('! Extent of array is: [2, 2, 2]\n'),
        mocker.call().write('! Maximum 20 data items per line\n'),
        mocker.call().write('\n'),
        mocker.call().write('0'),
        mocker.call().write('\t'),
        mocker.call().write('0'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('0'),
        mocker.call().write('\t'),
        mocker.call().write('0'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('0'),
        mocker.call().write('\t'),
        mocker.call().write('0'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('0'),
        mocker.call().write('\t'),
        mocker.call().write('0'),
        mocker.call().write('\n'),
        mocker.call().__exit__(None, None, None)
    ]

    # Act
    wd.write_array_to_ascii_file('test', test_extent, test_array, decimals = 0)

    # Assert
    open_mock.assert_has_calls(expected_calls)


def test_write_array_to_ascii_file_binary(mocker: MockerFixture, caplog):
    # Arrange
    test_extent = np.array([2, 2, 2])
    test_array = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    open_mock = mocker.mock_open()
    fileno_mock = mocker.Mock(return_value = 1, name = 'fileno_mock')
    open_mock.return_value.fileno = fileno_mock

    mocker.patch("builtins.open", open_mock)
    expected_calls = [mocker.call('test.db', 'wb'), mocker.call().__enter__()]

    # Act
    wd.write_array_to_ascii_file('test', test_extent, test_array, use_binary = True, binary_only = True)

    # Assert
    open_mock.assert_has_calls(expected_calls)
    assert 'Failed to write data to binary file test' not in caplog.text


def test_write_array_to_ascii_file_no_headers(mocker: MockerFixture):
    # Arrange
    test_extent = np.array([2, 2, 2])
    test_array = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    open_mock = mocker.mock_open()
    mocker.patch("builtins.open", open_mock)
    unexpected_calls = [
        mocker.call().write('! Data written by write_array_to_ascii_file() python function\n'),
        mocker.call().write('! Extent of array is: [2, 2, 2]\n'),
        mocker.call().write('! Maximum 20 data items per line\n')
    ]

    # Act
    wd.write_array_to_ascii_file('test', test_extent, test_array, headers = False)

    # Assert
    assert unexpected_calls not in open_mock.mock_calls


def test_write_array_to_ascii_file_bool(mocker: MockerFixture):
    # Arrange
    test_extent = np.array([2, 2, 2])
    test_array = np.array([[[0, 1], [0, 0]], [[1, 0], [1, 0]]])
    open_mock = mocker.mock_open()
    mocker.patch("builtins.open", open_mock)
    expected_calls = [
        mocker.call('test', 'w'),
        mocker.call().__enter__(),
        mocker.call().write('! Data written by write_array_to_ascii_file() python function\n'),
        mocker.call().write('! Extent of array is: [2, 2, 2]\n'),
        mocker.call().write('! Maximum 20 data items per line\n'),
        mocker.call().write('\n'),
        mocker.call().write('0'),
        mocker.call().write('\t'),
        mocker.call().write('1'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('0'),
        mocker.call().write('\t'),
        mocker.call().write('0'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('1'),
        mocker.call().write('\t'),
        mocker.call().write('0'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('1'),
        mocker.call().write('\t'),
        mocker.call().write('0'),
        mocker.call().write('\n'),
        mocker.call().__exit__(None, None, None)
    ]

    # Act
    wd.write_array_to_ascii_file('test', test_extent, test_array, data_type = 'bool')

    # Assert
    open_mock.assert_has_calls(expected_calls)


def test_write_array_to_ascii_file_append(mocker: MockerFixture):
    # Arrange
    test_extent = np.array([2, 2, 2])
    test_array = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    open_mock = mocker.mock_open()
    mocker.patch("builtins.open", open_mock)
    expected_calls = [mocker.call('test', 'a')]

    # Act
    wd.write_array_to_ascii_file('test', test_extent, test_array, append = True)

    # Assert
    open_mock.assert_has_calls(expected_calls)


def test_write_array_to_ascii_file_space_separated(mocker: MockerFixture):
    # Arrange
    test_extent = np.array([2, 2, 2])
    test_array = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    open_mock = mocker.mock_open()
    mocker.patch("builtins.open", open_mock)
    expected_calls = [
        mocker.call('test', 'w'),
        mocker.call().__enter__(),
        mocker.call().write('! Data written by write_array_to_ascii_file() python function\n'),
        mocker.call().write('! Extent of array is: [2, 2, 2]\n'),
        mocker.call().write('! Maximum 20 data items per line\n'),
        mocker.call().write('\n'),
        mocker.call().write('0.000'),
        mocker.call().write(' '),
        mocker.call().write('0.000'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('0.000'),
        mocker.call().write(' '),
        mocker.call().write('0.000'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('0.000'),
        mocker.call().write(' '),
        mocker.call().write('0.000'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('0.000'),
        mocker.call().write(' '),
        mocker.call().write('0.000'),
        mocker.call().write('\n'),
        mocker.call().__exit__(None, None, None)
    ]

    # Act
    wd.write_array_to_ascii_file('test', test_extent, test_array, space_separated = True)

    # Assert
    open_mock.assert_has_calls(expected_calls)


def test_write_array_to_ascii_file_nan_substitute(mocker: MockerFixture):
    # Arrange
    test_extent = np.array([2, 2, 2])
    test_array = np.array([[[1, np.nan], [3, 2]], [[1, 5], [np.nan, np.nan]]])
    open_mock = mocker.mock_open()
    mocker.patch("builtins.open", open_mock)
    expected_calls = [
        mocker.call('test', 'w'),
        mocker.call().__enter__(),
        mocker.call().write('! Data written by write_array_to_ascii_file() python function\n'),
        mocker.call().write('! Extent of array is: [2, 2, 2]\n'),
        mocker.call().write('! Maximum 20 data items per line\n'),
        mocker.call().write('\n'),
        mocker.call().write('1.000'),
        mocker.call().write('\t'),
        mocker.call().write('0.000'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('3.000'),
        mocker.call().write('\t'),
        mocker.call().write('2.000'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('1.000'),
        mocker.call().write('\t'),
        mocker.call().write('5.000'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('0.000'),
        mocker.call().write('\t'),
        mocker.call().write('0.000'),
        mocker.call().write('\n'),
        mocker.call().__exit__(None, None, None)
    ]

    # Act
    wd.write_array_to_ascii_file('test', test_extent, test_array, nan_substitute_value = 0)

    # Assert
    open_mock.assert_has_calls(expected_calls)


def test_write_array_to_ascii_file_blank_line_after_j_block(mocker: MockerFixture):
    # Arrange
    test_extent = np.array([2, 2, 2])
    test_array = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    open_mock = mocker.mock_open()
    mocker.patch("builtins.open", open_mock)
    expected_calls = [
        mocker.call('test', 'w'),
        mocker.call().__enter__(),
        mocker.call().write('! Data written by write_array_to_ascii_file() python function\n'),
        mocker.call().write('! Extent of array is: [2, 2, 2]\n'),
        mocker.call().write('! Maximum 20 data items per line\n'),
        mocker.call().write('\n'),
        mocker.call().write('0.000'),
        mocker.call().write('\t'),
        mocker.call().write('0.000'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('0.000'),
        mocker.call().write('\t'),
        mocker.call().write('0.000'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('0.000'),
        mocker.call().write('\t'),
        mocker.call().write('0.000'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().write('0.000'),
        mocker.call().write('\t'),
        mocker.call().write('0.000'),
        mocker.call().write('\n'),
        mocker.call().write('\n'),
        mocker.call().__exit__(None, None, None)
    ]

    # Act
    wd.write_array_to_ascii_file('test', test_extent, test_array, blank_line_after_j_block = True)

    # Assert
    open_mock.assert_has_calls(expected_calls)
