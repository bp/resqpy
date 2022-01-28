from resqpy.grid.write_nexus_corp import write_nexus_corp
from pytest_mock import MockerFixture
from typing import List


def get_expected_calls(mocker: MockerFixture, file_name: str) -> List:
    """List of the first 20 expected open calls for the write_nexus_corp method with default args."""
    expected_calls = [
        mocker.call(file_name, 'w'),
        mocker.call().__enter__(),
        mocker.call().write('! Nexus corner point data written by resqml_grid module\n'),
        mocker.call().write('! Nexus is a registered trademark of the Halliburton Company\n\n'),
        mocker.call().__exit__(None, None, None),
        mocker.call(file_name, 'a'),
        mocker.call().__enter__(),
        mocker.call().write('! Data written by write_array_to_ascii_file() python function\n'),
        mocker.call().write('! Extent of array is: [3, 8, 8]\n'),
        mocker.call().write('! Maximum 3 data items per line\n'),
        mocker.call().write('\n'),
        mocker.call().write('0.000'),  # Only checking the first few lines are correct
        mocker.call().write('\t'),
        mocker.call().write('0.000'),
        mocker.call().write('\t'),
        mocker.call().write('0.000'),
        mocker.call().write('\n'),
        mocker.call().write('100.000'),
        mocker.call().write('\t'),
        mocker.call().write('0.000'),
    ]
    return expected_calls


def test_defaults(mocker: MockerFixture, tmp_path, basic_regular_grid):
    # Arrange
    file_name = f'{tmp_path}/test'
    open_mock = mocker.mock_open()
    mocker.patch("builtins.open", open_mock)
    expected_calls = get_expected_calls(mocker, file_name)

    # Act
    write_nexus_corp(basic_regular_grid, file_name)

    # Assert
    open_mock.assert_has_calls(expected_calls)


def test_write_nx_ny_nz_true(mocker: MockerFixture, tmp_path, basic_regular_grid):
    # Arrange
    file_name = f'{tmp_path}/test'
    open_mock = mocker.mock_open()
    mocker.patch("builtins.open", open_mock)
    expected_calls = get_expected_calls(mocker, file_name)
    expected_calls[4:4] = [
        mocker.call().write('NX      NY      NZ\n'),
        mocker.call().write('2       2       2      \n\n')
    ]

    # Act
    write_nexus_corp(basic_regular_grid, file_name, write_nx_ny_nz = True)

    # Assert
    open_mock.assert_has_calls(expected_calls)


def test_write_units_keyword_true(mocker: MockerFixture, tmp_path, basic_regular_grid):
    # Arrange
    file_name = f'{tmp_path}/test'
    open_mock = mocker.mock_open()
    mocker.patch("builtins.open", open_mock)
    expected_calls = get_expected_calls(mocker, file_name)
    expected_calls.insert(4, mocker.call().write('! global units unknown or mixed\n\n'))

    # Act
    write_nexus_corp(basic_regular_grid, file_name, write_units_keyword = True)

    # Assert
    open_mock.assert_has_calls(expected_calls)


def test_local_coords_true(mocker: MockerFixture, tmp_path, basic_regular_grid):
    # Arrange
    file_name = f'{tmp_path}/test'
    open_mock = mocker.mock_open()
    mocker.patch("builtins.open", open_mock)
    expected_calls = get_expected_calls(mocker, file_name)
    expected_calls.insert(4, mocker.call().write('METRIC\n\n'))

    # Act
    write_nexus_corp(basic_regular_grid, file_name, write_units_keyword = True, local_coords = True)

    # Assert
    open_mock.assert_has_calls(expected_calls)


def test_write_rh_keyword_if_needed_true(mocker: MockerFixture, tmp_path, basic_regular_grid):
    # Arrange
    file_name = f'{tmp_path}/test'
    open_mock = mocker.mock_open()
    mocker.patch("builtins.open", open_mock)
    expected_calls = get_expected_calls(mocker, file_name)
    expected_calls.insert(4, mocker.call().write('RIGHTHANDED\n\n'))

    # Act
    write_nexus_corp(basic_regular_grid, file_name, write_rh_keyword_if_needed = True)

    # Assert
    open_mock.assert_has_calls(expected_calls)
