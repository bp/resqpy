"""Geologic time series."""

import logging

log = logging.getLogger(__name__)

import resqpy.time_series
import resqpy.time_series._any_time_series as ats


class GeologicTimeSeries(ats.AnyTimeSeries):
    """Class for RESQML Time Series using only year offsets (for geological time frames)."""

    def __init__(self, parent_model, uuid = None, title = None, originator = None, extra_metadata = None):
        """Create a GeologicTimeSeries object, either from a time series node in parent model, or empty.

        arguments:
           parent_model (model.Model): the resqpy model to which the time series will belong
           uuid (uuid.UUID, optional): the uuid of a TimeSeries object to be loaded from xml
           title (str, optional): the citation title to use for a new time series;
              ignored if uuid is not None
           originator (str, optional): the name of the person creating the time series, defaults to login id;
              ignored if uuid is not None
           extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the time series;
              ignored if uuid is not None

        returns:
           newly instantiated GeologicTimeSeries object

        note:
           if instantiating from an existing RESQML time series, its Time entries must all have YearOffset data
           which should be large negative integers

        :meta common:
        """
        self.timeframe = 'geologic'
        self.timestamps = []  # ordered list of (large negative) ints being year offsets from present
        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)
        if self.extra_metadata is not None and self.extra_metadata.get('timeframe') == 'human':
            raise ValueError('attempt to instantiate a geologic time series for a human timeframe time series')

    @classmethod
    def from_year_list(cls, parent_model, year_list, title = None, originator = None, extra_metadata = {}):
        """Creates a new GeologicTimeSeries from a list of large integers representing years before present.

        note:
           the years will be converted to negative numbers if positive, and sorted from oldest (most negative)
           to youngest (least negative)

        :meta common:
        """

        assert isinstance(year_list, list) and len(year_list) > 0
        negative_list = []
        for year in year_list:
            assert isinstance(year, int)
            if year > 0:
                negative_list.append(-year)
            else:
                negative_list.append(year)

        gts = cls(parent_model, title = title, originator = originator, extra_metadata = extra_metadata)

        gts.timestamps = sorted(negative_list)

        return gts

    def is_equivalent(self, other_ts):
        """Returns True if the this geologic time series is essentially identical to the other; otherwise False."""

        super_equivalence = super().is_equivalent(other_ts)
        if super_equivalence is not None:
            return super_equivalence
        return self.timestamps == other_ts.timestamps  # has no tolerance of small differences
