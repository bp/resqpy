import datetime

from resqpy.time_series._from_nexus_summary import _process_summary_entries, time_series_from_nexus_summary


def test_process_summary_entries_none():
    # arrange
    summary_entries = []
    # act
    result = _process_summary_entries(summary_entries)
    # assert
    assert result is None


# (time step no., cumulative time (days), date (dd/mm/yyyy))


def test_process_summary_entries_first():
    # arrange
    summary_entries = [(0, 1, datetime.datetime.strptime('24052010', '%d%m%Y').date())]
    # act
    result = _process_summary_entries(summary_entries)
    # assert
    assert len(result.timestamps) == 1
    assert result.timestamps[0] == '2010-05-24T00:00:00Z'


def test_process_summary_entries_delta():
    # arrange
    summary_entries = [(1, 1, datetime.datetime.strptime('24052010', '%d%m%Y').date())]
    # act
    result = _process_summary_entries(summary_entries)
    # assert
    assert len(result.timestamps) == 2
    assert result.timestamps[1] == '2010-05-24T00:00:00Z'
    assert result.timestamps[0] == '2010-05-23T00:00:00Z'


def test_process_summary_entries_out_of_sequence():
    # arrange
    summary_entries = [(0, 1, datetime.datetime.strptime('24052010', '%d%m%Y').date()),
                       (0, 1, datetime.datetime.strptime('24052010', '%d%m%Y').date())]
    # act
    result = _process_summary_entries(summary_entries)
    # assert
    assert len(result.timestamps) == 1
    assert result.timestamps[0] == '2010-05-24T00:00:00Z'


def test_process_summary_entries_out_of_sequence_v2():
    # arrange
    summary_entries = [(1, 1, datetime.datetime.strptime('24052010', '%d%m%Y').date()),
                       (3, 4, datetime.datetime.strptime('28052010', '%d%m%Y').date()),
                       (2, 2, datetime.datetime.strptime('26052010', '%d%m%Y').date())]
    # act
    result = _process_summary_entries(summary_entries)
    # assert
    assert len(result.timestamps) == 4
    assert result.timestamps[0] == '2010-05-23T00:00:00Z'
    assert result.timestamps[1] == '2010-05-24T00:00:00Z'
    assert result.timestamps[2] == '2010-05-25T00:00:00Z'
    assert result.timestamps[3] == '2010-05-27T00:00:00Z'


def test_process_summary_entries_in_sequence():
    # arrange
    summary_entries = [(1, 1, datetime.datetime.strptime('24052010', '%d%m%Y').date()),
                       (3, 4, datetime.datetime.strptime('28052010', '%d%m%Y').date())]
    # act
    result = _process_summary_entries(summary_entries)
    # assert
    assert len(result.timestamps) == 4
    assert result.timestamps[0] == '2010-05-23T00:00:00Z'
    assert result.timestamps[1] == '2010-05-24T00:00:00Z'
    assert result.timestamps[2] == '2010-05-24T00:00:00Z'
    assert result.timestamps[3] == '2010-05-27T00:00:00Z'


def test_process_summary_entries_incorrect_days():
    # arrange
    summary_entries = [(1, 1, datetime.datetime.strptime('24052010', '%d%m%Y').date()),
                       (2, 0, datetime.datetime.strptime('28052010', '%d%m%Y').date())]
    # act
    result = _process_summary_entries(summary_entries)
    # assert
    assert len(result.timestamps) == 3
    assert result.timestamps[0] == '2010-05-23T00:00:00Z'
    assert result.timestamps[1] == '2010-05-24T00:00:00Z'
    assert result.timestamps[2] == '2010-05-24T00:00:00Z'


def test_time_series_from_nexus_summary_is_not_file(mocker):
    # arrange
    is_file_mock = mocker.MagicMock(return_value = False)
    mocker.patch('os.path.isfile', is_file_mock)
    # log_warning_mock = mocker.MagicMock()
    # logging_mock = mocker.MagicMock()
    # logging_mock.getlogger.warning = log_warning_mock
    # mocker.patch(logging, logging_mock)
    # act
    result = time_series_from_nexus_summary('test')
    # assert
    assert result is None
    # log_warning_mock.assert_called_once_with('Summary file not found: ' + 'test')


def test_time_series_from_nexus_summary_if_no_file():
    # arrange
    # act
    result = time_series_from_nexus_summary('')
    # assert
    assert result is None


# def test_time_series_from_nexus_summary_open_file(mocker):
#     # arrange
#     is_file_mock = mocker.MagicMock(return_value=True)
#     mocker.patch('os.path.isfile', is_file_mock)
#     summary_entries = [(1, 1, datetime.datetime.strptime('24052010', '%d%m%Y').date()),
#                        (2, 0, datetime.datetime.strptime('28052010', '%d%m%Y').date())]
#     open_file_mock = mocker.MagicMock(return_value=summary_entries)
#     mocker.patch(resqpy.time_series.time_series_from_nexus_summary.open_file, open_file_mock)
#     # act
#     result = time_series_from_nexus_summary('test')
#     # assert
#     assert result is None
#     open_file_mock.assert_called_once_with('test')
