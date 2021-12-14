import pytest
import datetime as dt

from resqpy.time_series import TimeDuration


@pytest.mark.parametrize('days, hours, earlier_timestamp, later_timestamp',
                         [(1, 12, '2021-12-13T00:00:00Z', '2021-12-14T12:00:00Z'),
                          (30, 0, '2021-12-13T00:00:00Z', '2022-01-12T00:00:00Z'),
                          (30, 23, '2022-04-13T06:30:00Z', '2022-05-14T05:30:00Z'),
                          (365, 0, '2024-01-01T00:00:00Z', '2024-12-31T00:00:00Z'),
                          (0, 0, '2014-01-01T00:00:00Z', '2014-01-01T00:00:00Z')])
def test_timestamp_before_and_after_duration(days, hours, earlier_timestamp, later_timestamp):
    duration = TimeDuration(days = days, hours = hours)
    assert duration.timestamp_before_duration(later_timestamp) == earlier_timestamp
    assert duration.timestamp_after_duration(earlier_timestamp) == later_timestamp
