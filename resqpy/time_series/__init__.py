"""Time series classes and functions."""

__all__ = [
    'TimeSeries', 'GeologicTimeSeries', 'AnyTimeSeries', 'TimeDuration', 'selected_time_series', 'simplified_timestamp',
    'cleaned_timestamp', 'time_series_from_list', 'merge_timeseries_from_uuid', 'geologic_time_str',
    'timeframe_for_time_series_uuid', 'any_time_series', 'time_series_from_nexus_summary', 'check_timestamp'
]

from ._any_time_series import AnyTimeSeries
from ._geologic_time_series import GeologicTimeSeries
from ._time_duration import TimeDuration
from ._time_series import TimeSeries, check_timestamp
from ._functions import selected_time_series, simplified_timestamp, cleaned_timestamp, time_series_from_list, \
    merge_timeseries_from_uuid, geologic_time_str, timeframe_for_time_series_uuid, any_time_series
from ._from_nexus_summary import time_series_from_nexus_summary

# Set "module" attribute of all public objects to this path.
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
