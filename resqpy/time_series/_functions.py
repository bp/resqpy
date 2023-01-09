"""Auxilliary functions for working with time series."""

# At present, no time zone information is handled

# This module differentiates between 'human timeframe' and 'geologic timeframe' time series, though RESQML does
#  not make such a distinction explicitly; the code here uses presence of YearOffset xml data to imply a time
# series on a geological timeframe

import logging

log = logging.getLogger(__name__)

import resqpy.time_series as rqts
import resqpy.olio.xml_et as rqet


def selected_time_series(full_series, indices_list, title = None):
    """Returns a new TimeSeries or GeologicTimeSeries object with timestamps selected from the full series by a list of indices."""

    from resqpy.time_series._geologic_time_series import GeologicTimeSeries
    from resqpy.time_series._time_series import TimeSeries

    if isinstance(full_series, TimeSeries):
        selected_ts = TimeSeries(full_series.model, title = title)
    else:
        assert isinstance(full_series, GeologicTimeSeries)
        selected_ts = GeologicTimeSeries(full_series.model, title = title)
    selected_ts.timestamps = [full_series.timestamps[i] for i in sorted(indices_list)]
    return selected_ts


def simplified_timestamp(timestamp):
    """Return a more readable version of the timestamp."""

    if timestamp is None:
        return None
    if isinstance(timestamp, int):
        return geologic_time_str(int)
    timestamp = cleaned_timestamp(timestamp)
    if len(timestamp) > 10:
        timestamp = timestamp[:10] + ' ' + timestamp[11:]
    return timestamp[:19]


def cleaned_timestamp(timestamp):
    """Return a cleaned version of the timestamp."""

    if timestamp is None:
        return None
    if isinstance(timestamp, int):
        return geologic_time_str(int)
    timestamp = str(timestamp)
    rqts.check_timestamp(timestamp)
    if len(timestamp) < 19 or timestamp[11:19] == '00:00:00':
        return timestamp[:10]
    return timestamp[:10] + 'T' + timestamp[11:19] + 'Z'


def time_series_from_list(timestamp_list, parent_model = None, title = None):
    """Create a TimeSeries object from a list of timestamps (model and node set to None).

    note:
       timestamps in the list should be in the correct string format for human timeframe series,
       or large negative integers for geologic timeframe series
    """

    from resqpy.time_series._geologic_time_series import GeologicTimeSeries
    from resqpy.time_series._time_series import TimeSeries

    assert (len(timestamp_list) > 0)
    sorted_timestamps = sorted(timestamp_list)
    if isinstance(sorted_timestamps[0], int):
        sorted_timestamps = sorted([-t if t > 0 else t for t in timestamp_list])
        time_series = GeologicTimeSeries(parent_model = parent_model, title = title)
        time_series.timestamps = sorted_timestamps
    else:
        sorted_timestamps = sorted(timestamp_list)
        time_series = TimeSeries(parent_model = parent_model,
                                 first_timestamp = cleaned_timestamp(sorted_timestamps[0]),
                                 title = title)
        for raw_timestamp in sorted_timestamps[1:]:
            timestamp = cleaned_timestamp(raw_timestamp)
            time_series.add_timestamp(timestamp)
    return time_series


def merge_timeseries_from_uuid(model, timeseries_uuid_iter):
    """Create a TimeSeries object from an iterable object of existing timeseries UUIDs of timeseries.

    iterable can be a list, array, or iterable generator (model must be provided). The new timeseries is sorted in
    ascending order. Returns the new time series, the new time series uuid, and the list of timeseries objects used to
    generate the list
    """

    reverse = False

    alltimestamps = set({})
    timeserieslist = []
    timeframe = None
    for timeseries_uuid in timeseries_uuid_iter:
        timeseriesroot = model.root(uuid = timeseries_uuid)
        assert (rqet.node_type(timeseriesroot) == 'obj_TimeSeries')

        singlets = any_time_series(model, uuid = timeseries_uuid)
        if timeframe is None:
            timeframe = singlets.timeframe
        else:
            assert timeframe == singlets.timeframe, 'attempt to merge human and geologic timeframe time series'

        timeserieslist.append(singlets)
        # alltimestamps.update( set(singlets.timestamps) )
        alltimestamps.update(set(singlets.datetimes()))

    sortedtimestamps = sorted(list(alltimestamps), reverse = reverse)

    new_time_series = time_series_from_list(sortedtimestamps, parent_model = model)
    new_time_series_uuid = new_time_series.uuid
    return new_time_series, new_time_series_uuid, timeserieslist


def geologic_time_str(years):
    """Returns a string representing a geological time for a large int representing number of years before present."""

    assert isinstance(years, int)
    years = abs(years)  # positive and negative values are both interpreted as the same: years before present
    if years < 10000000 and years % 1000000:
        stamp = f'{float(-years) / 1.0e6:.3f} Ma'
    else:
        stamp = f'{-years // 1000000} Ma'
    return stamp


def timeframe_for_time_series_uuid(model, uuid):
    """Returns string 'human' or 'geologic' indicating timeframe of the RESQML time series with a given uuid."""

    assert model is not None
    t = model.type_of_uuid(uuid = uuid, strip_obj = True)
    assert t is not None, 'time series uuid {uuid} not present in model'
    assert t == 'TimeSeries'

    root = model.root(uuid = uuid)

    em = rqet.load_metadata_from_xml(root)
    if em is not None:
        timeframe = em.get('timeframe')
        if timeframe:
            return timeframe

    t_node = rqet.find_tag(root, 'Time')
    assert t_node is not None

    return 'human' if rqet.find_tag(t_node, 'YearOffset') is None else 'geologic'


def any_time_series(parent_model, uuid):
    """Returns a resqpy TimeSeries or GeologicTimeSeries object for an existing RESQML time series with a given uuid."""

    from resqpy.time_series._geologic_time_series import GeologicTimeSeries
    from resqpy.time_series._time_series import TimeSeries

    timeframe = timeframe_for_time_series_uuid(parent_model, uuid)
    if timeframe == 'human':
        return TimeSeries(parent_model, uuid = uuid)
    assert timeframe == 'geologic'
    return GeologicTimeSeries(parent_model, uuid = uuid)
