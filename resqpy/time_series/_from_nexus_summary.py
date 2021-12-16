"""Function for creating a time series from a Nexus summary file."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import datetime as dt
import os
from ._time_series import TimeSeries


def time_series_from_nexus_summary(summary_file, parent_model = None):
    """Create a TimeSeries object based on time steps reported in a Nexus summary file (.sum).

    note:
        this function does not create the xml for the new TimeSeries, nor add it as a part to the parent model
    """

    if not summary_file:
        return None

    if not os.path.isfile(summary_file):
        log.warning('Summary file not found: ' + summary_file)
        return None

    try:

        summary_entries = _open_file(summary_file)

        time_series = _process_summary_entries(summary_entries, parent_model)

    except Exception:
        log.exception('failed to create TimeSeries object from summary file: ' + summary_file)
        return None

    return time_series


def _process_summary_entries(summary_entries, parent_model = None):
    """Create a TimeSeries object based on time steps reported in a Nexus summary file (.sum)."""
    if len(summary_entries) == 0:
        return None  # no entries extracted from summary file, could raise error?
    summary_entries.sort()
    if summary_entries[0][0] == 0:  # first entry is for time step zero
        tz_date = summary_entries[0][2]
        summary_entries.pop(0)
    else:  # back calculate time zero from first entry
        delta = dt.timedelta(days = summary_entries[0][1])
        tz_date = summary_entries[0][2] - delta
    time_series = TimeSeries(parent_model = parent_model, first_timestamp = tz_date.isoformat() + 'T00:00:00Z')
    last_timestep = 0
    last_cumulative_time = 0.0
    for entry in summary_entries:
        if entry[0] <= last_timestep:
            log.warning('ignoring out of sequence time step in summary file: ' + str(entry[0]))
            continue
        while entry[0] > last_timestep + 1:  # add filler steps
            time_series.extend_by_days(0)
            last_timestep += 1
        if entry[1] < last_cumulative_time:  # should never happen
            time_series.extend_by_days(0)
        else:
            time_series.extend_by_days(entry[1] - last_cumulative_time)
            last_cumulative_time = entry[1]
        last_timestep += 1
    return time_series


def _open_file(summary_file):
    """Opens the Nexus summary file."""

    us_date_format = True
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
                if words[0].upper() == 'NO.' and len(words[4]) == 10:
                    date_format = words[4].upper()
                    us_date_format = (date_format == 'MM/DD/YYYY')
                    if not us_date_format:
                        assert date_format == 'DD/MM/YYYY'
                continue
            ts_number = int(words[0])
            time_in_days = float(words[3])
            date_str = words[4]
            assert 9 <= len(date_str) <= 10, 'invalid date string length in summary file: ' + date_str
            if us_date_format:
                date = dt.date(int(date_str[-4:]), int(date_str[:-8]), int(date_str[-7:-5]))
            else:
                date = dt.date(int(date_str[-4:]), int(date_str[-7:-5]), int(date_str[:-8]))
            summary_entries.append((ts_number, time_in_days, date))
    return summary_entries
