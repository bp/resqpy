import pandas as pd
import pytest
import resqpy.olio.write_faults as wf
from pytest_mock import MockerFixture


@pytest.mark.parametrize("test_df", [pd.DataFrame(), None])
def test_no_data_in_faults(tmp_path, test_df):
    # Act & Assert
    with pytest.raises(AssertionError):
        wf.write_faults_nexus(f'{tmp_path}/test.txt', test_df)


def test_writing_faults_to_file(tmp_path, mocker: MockerFixture):
    # Arrange
    test_df = pd.DataFrame({
        'name': ['fault1', 'fault2'],
        'i1': [720, 305],
        'i2': [875, 342],
        'j1': [311, 32],
        'j2': [103, 800],
        'k1': [791, 847],
        'k2': [994, 494],
        'face': ['I+', 'I+']
    })
    open_mock = mocker.mock_open()
    mocker.patch("builtins.open", open_mock)
    expected_calls = [
        mocker.call(f'{tmp_path}/test.txt', 'w'),
        mocker.call().__enter__(),
        mocker.call().write('\nMULT\tTX\tALL\tPLUS\tMULT\n'),
        mocker.call().write('\tGRID\tROOT\n'),
        mocker.call().write('\tFNAME\tfault1\n'),
        mocker.call().write('\t720\t875\t311\t103\t791\t994\t1.0\n'),
        mocker.call().write('\nMULT\tTX\tALL\tPLUS\tMULT\n'),
        mocker.call().write('\tGRID\tROOT\n'),
        mocker.call().write('\tFNAME\tfault2\n'),
        mocker.call().write('\t305\t342\t32\t800\t847\t494\t1.0\n'),
        mocker.call().__exit__(None, None, None)
    ]

    # Act
    wf.write_faults_nexus(f'{tmp_path}/test.txt', test_df)

    # Assert
    open_mock.assert_has_calls(expected_calls)


def test_log_info(tmp_path, caplog):
    # Arrange
    test_df = pd.DataFrame({
        'name': ['fault1', 'fault2'],
        'i1': [720, 305],
        'i2': [875, 342],
        'j1': [311, 32],
        'j2': [103, 800],
        'k1': [791, 847],
        'k2': [994, 494],
        'face': ['I+', 'I+']
    })

    # Act
    wf.write_faults_nexus(f'{tmp_path}/test.txt', test_df)

    # Assert
    for record in caplog.records:
        assert record.levelname == "INFO"
    assert f'writing FNAME data in Nexus format to file: {tmp_path}/test.txt in caplog.text'


def test_fault_name_greater_than_256_characters(tmp_path, caplog):
    # Arrange
    long_name = 'f' * 257
    test_df = pd.DataFrame({
        'name': ['fault1', long_name],
        'i1': [720, 305],
        'i2': [875, 342],
        'j1': [311, 32],
        'j2': [103, 800],
        'k1': [791, 847],
        'k2': [994, 494],
        'face': ['I+', 'I+']
    })

    # Act
    wf.write_faults_nexus(f'{tmp_path}/test.txt', test_df)

    # Assert
    assert f'exported fault name longer than Nexus limit of 256 characters: {long_name}' \
           in caplog.text
