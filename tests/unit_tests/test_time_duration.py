import logging
import pytest
import datetime as dt

log = logging.getLogger(__name__)
log.debug('resqml_time_series.py version ')


@pytest.mark.parametrize('later_timestamp, later_timestamp_end',
                         [('2014-01-01T00:00:00.588Z', '2014-01-01T00:00:00.588')])
def test_timestamp_before_duration(later_timestamp, later_timestamp_end):
    # arrange
    test_later_timestamp = later_timestamp
    test_later_timestamp_end = later_timestamp_end
    # act
    test_later_timestamp = test_later_timestamp[:-1]
    # assert
    assert test_later_timestamp == test_later_timestamp_end


@pytest.mark.parametrize('later_timestamp_end', ['2014-01-01T00:00:00.588'])
def test_timestamp_iso(later_timestamp_end):
    # arrange
    test_later_timestamp = later_timestamp_end
    test_dt_later = dt.datetime.fromisoformat(test_later_timestamp)
    # act
    test_dt_result = test_dt_later
    # assert
    return test_dt_result.isoformat() + 'Z'
