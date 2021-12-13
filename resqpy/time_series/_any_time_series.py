"""Code common to resqpy TimeSeries and GeologicTimeSeries classes."""

import logging

log = logging.getLogger(__name__)

import resqpy.time_series._functions as tsf
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class AnyTimeSeries(BaseResqpy):
    """Abstract class for a RESQML Time Series; use resqpy TimeSeries or GeologicTimeSeries.

    notes:
       this class has no direct initialisation method;
       call the any_time_series() function to generate the appropriate derived class object for a given uuid;
       the resqpy code differentiates between time series on a human timeframe, where the timestamps are
       used without a year offset, and those on a geologic timeframe, where the timestamps are ignored and
       only the year offset is significant
    """

    resqml_type = 'TimeSeries'

    def _load_from_xml(self):
        time_series_root = self.root
        assert time_series_root is not None
        # for human timeframe series, timestamps is an ordered list of timestamp strings in resqml/iso format
        # for geological timeframe series, timestamps is an ordered list of ints being the year offsets from present
        self.timestamps = []
        for child in rqet.list_of_tag(time_series_root, 'Time'):
            dt_text = rqet.find_tag_text(child, 'DateTime')
            assert dt_text, 'missing DateTime field in xml for time series'
            year_offset = rqet.find_tag_int(child, 'YearOffset')
            if year_offset:
                assert self.timeframe == 'geologic'
                self.timestamps.append(year_offset)  # todo: trim and check timestamp
            else:
                assert self.timeframe == 'human'
                self.timestamps.append(dt_text)  # todo: trim and check timestamp
            self.timestamps.sort()

    def is_equivalent(self, other_ts):
        """Performs partial equivalence check on two time series (derived class methods finish the work)."""

        if other_ts is None:
            return False
        if self is other_ts:
            return True
        if bu.matching_uuids(self.uuid, other_ts.uuid):
            return True
        if self.number_of_timestamps() != other_ts.number_of_timestamps():
            return False
        if self.timeframe != other_ts.timeframe:
            return False
        return None

    def set_model(self, parent_model):
        """Associate the time series with a resqml model (does not create xml or write hdf5 data)."""
        self.model = parent_model

    def number_of_timestamps(self):
        """Returns the number of timestamps in the series.

        :meta common:
        """
        return len(self.timestamps)

    def timestamp(self, index, as_string = True):
        """Returns an individual timestamp, indexed by its position in the series.

        arguments:
           index (int): the time index for which the timestamp is required
           as_string (boolean, default True): if True and this is series is on a geologic timeframe, the return
              value is a string, otherwise an int; for human timeframe series, this argument has no effect

        returns:
           string or int being the selected timestamp

        notes:
           index may be negative in which case it is taken to be relative to the end of the series
           with the last timestamp being referenced by an index of -1;
           the string form of a geologic timestamp is a positive number in millions of years,
           with the suffix Ma

        :meta common:
        """

        # individual timestamp is in iso format with a Z appended, eg: 2019-08-23T14:30:00Z
        if not -len(self.timestamps) <= index < len(self.timestamps):
            return None
        stamp = self.timestamps[index]
        if as_string and isinstance(stamp, int):
            stamp = tsf.geologic_time_str(stamp)
        return stamp

    def iter_timestamps(self, as_string = True):
        """Iterator over timestamps.

        :meta common:
        """
        for index in range(len(self.timestamps)):
            yield self.timestamp(index, as_string = as_string)

    def last_timestamp(self, as_string = True):
        """Returns the last timestamp in the series.

        :meta common:
        """
        return self.timestamp(-1, as_string = as_string) if self.timestamps else None

    def index_for_timestamp(self, timestamp):
        """Returns the index for a given timestamp.

        note:
           this method uses a simplistic string or int comparison;
           if the timestamp is not found, None is returned

        :meta common:
        """
        as_string = not isinstance(timestamp, int)
        for index in range(len(self.timestamps)):
            if self.timestamp(index, as_string = as_string) == timestamp:
                return index
        return None

    def create_xml(self, add_as_part = True, title = None, originator = None, reuse = True):
        """Create a RESQML time series xml node from a TimeSeries or GeologicTimeSeries object, optionally add as part.

        arguments:
           add_as_part (boolean, default True): if True, the newly created xml node is added as a part
              in the model
           title (string): used as the citation Title text for the new time series node
           originator (string, optional): the name of the human being who created the time series object;
              default is to use the login name

        returns:
           the newly created time series xml node

        :meta common:
        """

        if reuse and self.try_reuse():
            return self.root  # check for reusable (equivalent) object

        if self.extra_metadata is None:
            self.extra_metadata = {}
        self.extra_metadata['timeframe'] = self.timeframe

        ts_node = super().create_xml(add_as_part = False, title = title, originator = originator)

        for index in range(self.number_of_timestamps()):
            time_node = rqet.SubElement(ts_node, ns['resqml2'] + 'Time')
            time_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Timestamp')
            time_node.text = rqet.null_xml_text
            dt_node = rqet.SubElement(time_node, ns['resqml2'] + 'DateTime')
            dt_node.set(ns['xsi'] + 'type', ns['xsd'] + 'dateTime')
            if self.timeframe == 'geologic':
                assert isinstance(self.timestamps[index], int)
                dt_node.text = '0000-01-01T00:00:00Z'
                yo_node = rqet.SubElement(time_node, ns['resqml2'] + 'YearOffset')
                yo_node.set(ns['xsi'] + 'type', ns['xsd'] + 'long')
                yo_node.text = str(self.timestamps[index])
            else:
                dt_node.text = self.timestamp(index)

        if add_as_part:
            self.model.add_part('obj_TimeSeries', self.uuid, ts_node)

        return ts_node

    def create_time_index(self, time_index, root = None, check_valid_index = True):
        """Create a time index node, including time series reference, optionally add to root.

        arguments:
           time_index (int): non-negative integer index into the time series, identifying the datetime
              being referenced by the new node
           root (optional): if present, the newly created time index xml node is added
              as a child to this node
           check_valid_index (boolean, default True): if True, an assertion error is raised
              if the time index is out of range for this time series

        returns:
           the newly created time index xml node

        note:
           a time index node is typically part of a recurrent property object; it identifies
           the datetime relevant to the property array (or other data) by indexing into a time series
           object
        """

        assert self.uuid is not None
        if check_valid_index:
            assert 0 <= time_index < len(self.timestamps), 'time index out of range for time series'

        t_node = rqet.Element(ns['resqml2'] + 'TimeIndex')
        t_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'TimeIndex')
        t_node.text = rqet.null_xml_text

        index_node = rqet.SubElement(t_node, ns['resqml2'] + 'Index')
        index_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
        index_node.text = str(time_index)

        self.model.create_time_series_ref(self.uuid, root = t_node)

        if root is not None:
            root.append(t_node)
        return t_node
