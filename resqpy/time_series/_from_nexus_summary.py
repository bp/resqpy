"""Function for creating a time series from a Nexus summary file."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import datetime as dt
import os

import resqpy.time_series
import resqpy.time_series._time_series as rqts
import resqpy.weights_and_measures as wam


def time_series_from_nexus_summary(summary_file, parent_model = None, start_date = None):
    """Create a TimeSeries object based on time steps reported in a Nexus summary file (.sum).

    arguments:
        summary_file (str): path of Nexus summary file
        parent_model (Model): the model to which the new time series will be attached
        start_date (str, optional): if the summary file does not contain dates, this will be
            used as the start date; required format is 'YYYY-MM-DD'

    returns:
        newly created TimeSeries

    note:
        this function does not create the xml for the new TimeSeries, nor add it as a part to the parent model
    """

    if not summary_file:
        log.warning('no summary file specified')
        return None

    if not os.path.isfile(summary_file):
        log.error(f'summary file not found: {summary_file}')
        return None

    try:

        summary_entries = _open_file(summary_file)

        time_series = _process_summary_entries(summary_entries, parent_model, start_date = start_date)

    except Exception:
        log.exception(f'failed to create TimeSeries object from summary file: {summary_file}')
        return None

    return time_series


def _process_summary_entries(summary_entries, parent_model = None, start_date = None):
    """Create a TimeSeries object based on time steps reported in a Nexus summary file (.sum)."""
    if start_date is not None:
        assert len(start_date) == 10 and start_date[4] == '-' and start_date[7] == '-',  \
            f'start date {start_date} specified for summary data is not in format YYYY-MM-DD'
        start_date = dt.date(int(start_date[:4]), int(start_date[5:7]), int(start_date[8:]))
    if len(summary_entries) == 0:
        log.warning('no entries read from Nexus summary file')
        return None  # no entries extracted from summary file, could raise error?
    summary_entries.sort()
    # sort out the start date
    tz_date = summary_entries[0][2]
    if tz_date is None:
        if start_date is None:
            log.warning('summary file does not include dates and no start date specified; using arbitrary date')
            tz_date = dt.date(2020, 1, 1)
            start_date = tz_date
        else:
            tz_date = start_date
    else:
        if summary_entries[0][0] > 0:
            delta = dt.timedelta(days = summary_entries[0][1])
            tz_date = tz_date - delta
            # todo: round to the nearest previous midnight?
        if start_date is not None and tz_date != start_date:
            log.warning(f'ignoring specified start date {start_date} in favour of summary file date {tz_date}')
            start_date = tz_date
    if summary_entries[0][0] == 0:  # first entry is for time step zero, handled as first timestamp
        summary_entries.pop(0)
    # intiialise the time series with just the start date
    time_series = rqts.TimeSeries(parent_model = parent_model, first_timestamp = tz_date.isoformat() + 'T00:00:00Z')
    last_timestep = 0
    last_cumulative_time = 0.0
    cumulative_time_error = False
    for entry in summary_entries:
        if entry[0] <= last_timestep:
            log.warning(f'ignoring out of sequence time step in summary file: {entry[0]}')
            continue
        while entry[0] > last_timestep + 1:  # add filler steps
            time_series.extend_by_days(0)
            last_timestep += 1
        if entry[1] < last_cumulative_time:  # should never happen
            time_series.extend_by_days(0)
            cumulative_time_error = True
        else:
            time_series.extend_by_days(entry[1] - last_cumulative_time)
            last_cumulative_time = entry[1]
        last_timestep += 1
    if cumulative_time_error:
        log.error('cumulative times in Nexus summary file are not monotonically increasing')
    return time_series


def _open_file(summary_file):
    """Opens and reads the Nexus summary file into a list of entries.

    arguments:
        summary_file (str): path of Nexus output summary file

    returns:
        list of (timestep_number, time_in_days, date) where timestep_number is a non-negative int,
        time_in_days is a float, and date is a 10 charater date in format YYYY-MM-DD (or None if
        dates are not present in summary file)
    """

    us_date_format = True
    dates_present = True
    time_uom = 'd'
    summary_entries = []  # list of (time step no., cumulative time (days), date (dd/mm/yyyy))

    with open(summary_file, "r") as fp:
        while True:
            line = fp.readline()
            if len(line) == 0:
                break  # end of file
            words = line.split()
            if len(words) < 5:
                continue
            if not words[0].isdigit():
                if words[0].upper() == 'NO.':
                    assert words[3] in ['DAYS', 'HRS']
                    time_uom = 'd' if words[3] == 'DAYS' else 'h'
                    if len(words[4]) == 10 and words[4][2] == '/':
                        dates_present = True
                        date_format = words[4].upper()
                        us_date_format = (date_format == 'MM/DD/YYYY')
                        if not us_date_format:
                            assert date_format == 'DD/MM/YYYY', f'unrecognised date format in Nexus summary file: {date_format}'
                    else:
                        dates_present = False
                continue
            ts_number = int(words[0])
            time_in_days_or_hours = float(words[3])
            time_in_days = wam.convert(time_in_days_or_hours, time_uom, 'd', quantity = 'time')
            if dates_present:
                date_str = words[4]
                assert 9 <= len(date_str) <= 10, 'invalid date string length in summary file: ' + date_str
                if us_date_format:
                    date = dt.date(int(date_str[-4:]), int(date_str[:-8]), int(date_str[-7:-5]))
                else:
                    date = dt.date(int(date_str[-4:]), int(date_str[-7:-5]), int(date_str[:-8]))
            else:
                date = None
            summary_entries.append(
                (ts_number, time_in_days, date))  # TODO: check handling of dates and times, including sub-second times
    return summary_entries
