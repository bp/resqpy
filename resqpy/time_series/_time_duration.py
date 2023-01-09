"""Time duration"""

import logging

log = logging.getLogger(__name__)

import datetime as dt
import resqpy.time_series as rqts


class TimeDuration:
    """A thin wrapper around python's datetime timedelta objects (not a RESQML class)."""

    def __init__(self,
                 days = None,
                 hours = None,
                 minutes = None,
                 seconds = None,
                 earlier_timestamp = None,
                 later_timestamp = None):
        """Create a TimeDuration object either from days and seconds or from a pair of timestamps."""

        # for negative durations, earlier_timestamp should be later than later_timestamp
        # or days, hours etc. should typically all be non-positive
        # ie. days = -1, hours = -12 will be a negative one and a half day duration
        # whilst days = -1, hours = 12 will be a negative half day duration
        self.duration = None
        if earlier_timestamp is not None and later_timestamp is not None:
            rqts.check_timestamp(earlier_timestamp)
            rqts.check_timestamp(later_timestamp)
            if earlier_timestamp.endswith('Z'):
                earlier_timestamp = earlier_timestamp[:-1]  # Trailing Z is not part of iso format
            if later_timestamp.endswith('Z'):
                later_timestamp = later_timestamp[:-1]
            dt_earlier = dt.datetime.fromisoformat(earlier_timestamp)
            dt_later = dt.datetime.fromisoformat(later_timestamp)
            self.duration = dt_later - dt_earlier
        else:
            if days is None:
                days = 0
            if hours is None:
                hours = 0
            if minutes is None:
                minutes = 0
            if seconds is None:
                seconds = 0
            self.duration = dt.timedelta(days = days, hours = hours, minutes = minutes, seconds = seconds)

    def timestamp_after_duration(self, earlier_timestamp):
        """Create a new timestamp from this duration and an earlier timestamp."""

        if earlier_timestamp.endswith('Z'):
            earlier_timestamp = earlier_timestamp[:-1]
        rqts.check_timestamp(earlier_timestamp)
        dt_earlier = dt.datetime.fromisoformat(earlier_timestamp)
        dt_result = dt_earlier + self.duration
        return dt_result.isoformat() + 'Z'

    def timestamp_before_duration(self, later_timestamp):
        """Create a new timestamp from this duration and a later timestamp."""

        if later_timestamp.endswith('Z'):
            later_timestamp = later_timestamp[:-1]
        rqts.check_timestamp(later_timestamp)
        dt_later = dt.datetime.fromisoformat(later_timestamp)
        dt_result = dt_later - self.duration
        return dt_result.isoformat() + 'Z'
