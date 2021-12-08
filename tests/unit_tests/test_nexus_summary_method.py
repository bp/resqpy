import logging
import pytest

import resqpy
from resqpy.time_series.time_series_from_nexus_summary import open_file, time_series_from_nexus_summary
import datetime


def test_time_series_from_nexus_summary_open_file(mocker):
    # arrange
    is_file_mock = mocker.MagicMock(return_value=True)
    mocker.patch('os.path.isfile', is_file_mock)
    summary_entries = [(1, 1, datetime.datetime.strptime('24052010', '%d%m%Y').date()),
                       (2, 0, datetime.datetime.strptime('28052010', '%d%m%Y').date())]
    open_file_mock = mocker.MagicMock(return_value=summary_entries)
    mocker.patch(resqpy.time_series.time_series_from_nexus_summary.open_file, open_file_mock)
    # act
    result = time_series_from_nexus_summary('test')
    # assert
    assert result is None
    open_file_mock.assert_called_once_with('test')