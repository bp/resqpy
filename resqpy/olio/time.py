"""time.py: A very thin wrapper around python datetime functionality, to meet resqml standard."""

import datetime as dt


def now(use_utc = False):
    """Returns an iso format string representation of the current time, to the nearest second.

    argument:
       use_utc (boolean, default False): if True, the current UTC time is used, otherwise local time

    returns:
       string of form YYYY-MM-DDThh:mm:ssZ representing the current time in iso format

    note:
       this is the format used by the resqml standard for representing date-times
    """

    if use_utc:
        stamp = dt.datetime.utcnow()  # NB: naive use of UTC time
    else:
        stamp = dt.datetime.now()
    return str(stamp.isoformat(sep = 'T', timespec = 'seconds')) + 'Z'
