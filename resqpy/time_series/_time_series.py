"""TimeSeries class handling normal (non-geological) time series."""

import logging

log = logging.getLogger(__name__)

import datetime as dt
import warnings

import resqpy.time_series
import resqpy.time_series._any_time_series as ats
import resqpy.time_series._time_duration as td


class TimeSeries(ats.AnyTimeSeries):
    """Class for RESQML Time Series without year offsets.

    notes:
       individual RESQML timestamps are strings formatted in accordance with ISO 8601;
       use this class for time series on a human timeframe;
       use the resqpy GeologicTimeSeries class instead if the time series is on a geological timeframe
    """

    def __init__(self,
                 parent_model,
                 uuid = None,
                 first_timestamp = None,
                 daily = None,
                 monthly = None,
                 quarterly = None,
                 yearly = None,
                 title = None,
                 originator = None,
                 extra_metadata = None):
        """Create a TimeSeries object, either from a time series node in parent model, or from given data.

        arguments:
           parent_model (model.Model): the resqpy model to which the time series will belong
           uuid (uuid.UUID, optional): the uuid of a TimeSeries object to be loaded from xml
           first_time_stamp (str, optional): the first timestamp (in RESQML format) if not loading from xml;
              this and the remaining arguments are ignored if loading from xml; if present, timestamp must
              be in ISO 8601 format, eg '2023-01-31' or '2023-01-31T13:30:00Z' or '2023-01-31T13:30:00.912'
           daily (non-negative int, optional): the number of one day interval timesteps to start the series
           monthly (non-negative int, optional): the number of 30 day interval timesteps to follow the daily
              timesteps
           quarterly (non-negative int, optional): the number of 90 day interval timesteps to follow the
              monthly timesteps
           yearly (non-negative int, optional): the number of 365 day interval timesteps to follow the
              quarterly timesteps
           title (str, optional): the citation title to use for a new time series;
              ignored if uuid is not None
           originator (str, optional): the name of the person creating the time series, defaults to login id;
              ignored if uuid is not None
           extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the time series;
              ignored if uuid is not None

        returns:
           newly instantiated TimeSeries object

        note:
           a new bespoke time series can be populated by passing the first timestamp here and using the
           add_timestamp() and/or extend_by...() methods

        :meta common:
        """

        self.timeframe = 'human'
        self.timestamps = []  # ordered list of timestamp strings in resqml/iso format
        if first_timestamp is not None:
            check_timestamp(first_timestamp)
            self.timestamps.append(first_timestamp)  # todo: check format of first_timestamp
            if daily is not None:
                for _ in range(daily):
                    self.extend_by_days(1)
            if monthly is not None:
                for _ in range(monthly):
                    self.extend_by_days(30)
            if quarterly is not None:
                for _ in range(quarterly):
                    self.extend_by_days(90)  # could use 91
            if yearly is not None:
                for _ in range(yearly):
                    self.extend_by_days(365)  # could use 360
        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)
        if self.extra_metadata is not None and self.extra_metadata.get('timeframe') == 'geologic':
            raise ValueError('attempt to instantiate a human timeframe time series for a geologic time series')

    def is_equivalent(self, other_ts, tol_seconds = 1):
        """Returns True if the this timestep series is essentially identical to the other; otherwise False."""

        super_equivalence = super().is_equivalent(other_ts)
        if super_equivalence is not None:
            return super_equivalence
        tolerance = td.TimeDuration(seconds = tol_seconds)
        for t_index in range(self.number_of_timestamps()):
            diff = td.TimeDuration(earlier_timestamp = self.timestamps[t_index],
                                   later_timestamp = other_ts.timestamps[t_index])
            if abs(diff.duration) > tolerance.duration:
                return False
        return True

    def index_for_timestamp_not_later_than(self, timestamp):
        """Returns the index of the latest timestamp that is not later than the specified date.

        :meta common:
        """
        check_timestamp(timestamp)
        index = len(self.timestamps) - 1
        while (index >= 0) and (self.timestamps[index] > timestamp):
            index -= 1
        if index < 0:
            return None
        return index

    def index_for_timestamp_not_earlier_than(self, timestamp):
        """Returns the index of the earliest timestamp that is not earlier than the specified date.

        :meta common:
        """
        check_timestamp(timestamp)
        index = 0
        while (index < len(self.timestamps)) and (self.timestamps[index] < timestamp):
            index += 1
        if index >= len(self.timestamps):
            return None
        return index

    def index_for_timestamp_closest_to(self, timestamp):
        """Returns the index of the timestamp that is closest to the specified date.

        :meta common:
        """
        check_timestamp(timestamp)
        if not self.timestamps:
            return None
        before = self.index_for_timestamp_not_later_than(timestamp)
        if not before:
            return 0
        if before == len(self.timestamps) - 1 or self.timestamps[before] == timestamp:
            return before
        after = before + 1
        early_delta = td.TimeDuration(earlier_timestamp = self.timestamps[before], later_timestamp = timestamp)
        later_delta = td.TimeDuration(earlier_timestamp = timestamp, later_timestamp = self.timestamps[after])
        return before if early_delta.duration <= later_delta.duration else after

    def duration_between_timestamps(self, earlier_index, later_index):
        """Returns the duration between a pair of timestamps.

        :meta common:
        """
        if earlier_index < 0 or later_index >= len(self.timestamps) or later_index < earlier_index:
            return None
        return td.TimeDuration(earlier_timestamp = self.timestamps[earlier_index],
                               later_timestamp = self.timestamps[later_index])

    def days_between_timestamps(self, earlier_index, later_index):
        """Returns the number of whole days between a pair of timestamps, as an integer."""
        delta = self.duration_between_timestamps(earlier_index, later_index)
        if delta is None:
            return None
        return delta.duration.days

    def duration_since_start(self, index):
        """Returns the duration between the start of the time series and the indexed timestamp.

        :meta common:
        """
        if index < 0 or index >= len(self.timestamps):
            return None
        return self.duration_between_timestamps(0, index)

    def days_since_start(self, index):
        """Returns the number of days between the start of the time series and the indexed timestamp."""
        return self.duration_since_start(index).duration.days

    def step_duration(self, index):
        """Returns the duration of the time step between the indexed timestamp and preceding one.

        :meta common:
        """
        if index < 1 or index >= len(self.timestamps):
            return None
        return self.duration_between_timestamps(index - 1, index)

    def step_days(self, index):
        """Returns the number of days between the indexed timestamp and preceding one."""
        delta = self.step_duration(index)
        if delta is None:
            return None
        return delta.duration.days

    # NB: Following functions modify the time series, which is dangerous if the series is in use by a model
    # Could check for relationships involving the time series and disallow changes if any found?
    def add_timestamp(self, new_timestamp, allow_insertion = False):
        """Inserts a new timestamp into the time series."""
        check_timestamp(new_timestamp)
        if allow_insertion:
            # NB: This can insert a timestamp anywhere in the series, will invalidate indices, possibly corrupting model
            index = self.index_for_timestamp_not_later_than(new_timestamp)
            if index is None:
                index = 0
            else:
                index += 1
            self.timestamps.insert(index, new_timestamp)
        else:
            last = self.last_timestamp()
            if last is not None:
                assert (new_timestamp > self.last_timestamp())
            self.timestamps.append(new_timestamp)

    def extend_by_duration(self, duration):
        """Adds a timestamp to the end of the series, at duration beyond the last timestamp."""
        assert (duration.duration.days >= 0)  # duration may not be negative
        assert (len(self.timestamps) > 0)  # there must be something to extend from
        self.timestamps.append(duration.timestamp_after_duration(self.last_timestamp()))

    def extend_by_days(self, days):
        """Adds a timestamp to the end of the series, at a duration of days beyond the last timestamp."""
        duration = td.TimeDuration(days = days)
        self.extend_by_duration(duration)

    def datetimes(self):
        """Returns the timestamps as a list of python-datetime objects."""
        return [dt.datetime.fromisoformat(t.rstrip('Z')) for t in self.timestamps]


def check_timestamp(timestamp):
    """Check format of timestamp and raise ValueError if badly formed."""
    if timestamp.endswith('Z'):
        timestamp = timestamp[:-1]
    _ = dt.datetime.fromisoformat(timestamp)
