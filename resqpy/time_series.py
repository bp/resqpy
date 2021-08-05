"""time_series.py: RESQML time series class."""

version = '2nd July 2021'

# Nexus is a registered trademark of the Halliburton Company

# At present, no time zone information is handled

import logging

log = logging.getLogger(__name__)
log.debug('resqml_time_series.py version ' + version)

import os
import datetime as dt
import warnings

from resqpy.olio.base import BaseResqpy
import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu
from resqpy.olio.xml_namespaces import curly_namespace as ns


class TimeDuration:
   """A thin wrapper around python's datatime timedelta objects (not a RESQML class)."""

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
      dt_earlier = dt.datetime.fromisoformat(earlier_timestamp)
      dt_result = dt_earlier + self.duration
      return dt_result.isoformat() + 'Z'

   def timestamp_before_duration(self, later_timestamp):
      """Create a new timestamp from this duration and a later timestamp."""

      if later_timestamp.endswith('Z'):
         later_timestamp = later_timestamp[:-1]
      dt_later = dt.datetime.fromisoformat(later_timestamp)
      dt_result = dt_later - self.duration
      return dt_result.isoformat() + 'Z'


class TimeSeries(BaseResqpy):
   """Class for RESQML Time Series within RESQML model object."""

   resqml_type = 'TimeSeries'

   def __init__(self,
                parent_model,
                uuid = None,
                time_series_root = None,
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
         time_series_root (xml node, DEPRECATED): the xml root node; use uuid instead
         first_time_stamp (str, optional): the first timestamp (in RESQML format) if not loading from xml;
            this and the remaining arguments are ignored if loading from xml
         daily (non-negative int, optional): the number of one day interval timesteps to start the series
         monthly (non-negative int, optional): the number of 30 day interval timesteps to follow the daily
            timesteps
         quarterly (non-negative int, optional): the number of 90 day interval timesteps to follow the
            monthly timesteps
         yearly (non-negative int, optional): the number of 365 day interval timesteps to follow the
            quarterly timesteps
         title (str, optional): the citation title to use for a new time series;
            ignored if uuid or time_series_root is not None
         originator (str, optional): the name of the person creating the time series, defaults to login id;
            ignored if uuid or time_series_root is not None
         extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the time series;
            ignored if uuid or time_series_root is not None

      returns:
         newly instantiated TimeSeries object

      note:
         a new bespoke time series can be populated by passing the first timestamp here and using the
         add_timestamp() and/or extend_by...() methods

      :meta common:
      """

      self.timestamps = []  # ordered list of timestamp strings in resqml/iso format
      if first_timestamp is not None:
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
                       extra_metadata = extra_metadata,
                       root_node = time_series_root)

   def _load_from_xml(self):
      time_series_root = self.root
      assert time_series_root is not None
      for child in time_series_root:
         if rqet.stripped_of_prefix(child.tag) != 'Time':
            continue
         dt_node = rqet.find_tag(child, 'DateTime')
         if dt_node is None:
            continue  # could raise an exception here
         self.timestamps.append(dt_node.text)  # todo: trim and check timestamp
         self.timestamps.sort()

   def is_equivalent(self, other_ts, tol_seconds = 1):
      """Returns True if the this timestep series is essentially identical to the other; otherwise False."""

      if other_ts is None:
         return False
      if self is other_ts:
         return True
      if bu.matching_uuids(self.uuid, other_ts.uuid):
         return True
      if self.number_of_timestamps() != other_ts.number_of_timestamps():
         return False
      tolerance = TimeDuration(seconds = tol_seconds)
      for t_index in range(self.number_of_timestamps()):
         diff = TimeDuration(earlier_timestamp = self.timestamps[t_index],
                             later_timestamp = other_ts.timestamps[t_index])
         if abs(diff.duration) > tolerance.duration:
            return False
      return True

   def set_model(self, parent_model):
      """Associate the time series with a resqml model (does not create xml or write hdf5 data)."""
      self.model = parent_model

   def number_of_timestamps(self):
      """Returns the number of timestamps in the series.

      :meta common:
      """
      return len(self.timestamps)

   def timestamp(self, index):
      """Returns an individual timestamp, indexed by its position in the series.

      :meta common:
      """

      # individual timestamp is in iso format with a Z appended, eg: 2019-08-23T14:30:00Z
      if index < 0 or index >= len(self.timestamps):
         return None
      return self.timestamps[index]

   def iter_timestamps(self):
      """Iterator over timestamps.

      :meta common:
      """
      for t_stamp in self.timestamps:
         yield t_stamp

   def last_timestamp(self):
      """Returns the last timestamp in the series.

      :meta common:
      """
      return self.timestamps[-1] if self.timestamps else None

   def index_for_timestamp(self, timestamp):
      """Returns the index for a given timestamp.

      note:
         this method uses a simplistic string comparison

      :meta common:
      """
      for index in range(len(self.timestamps)):
         if self.timestamps[index] == timestamp:
            return index
      return None

   def index_for_timestamp_not_later_than(self, timestamp):
      """Returns the index of the latest timestamp that is not later than the specified date.

      :meta common:
      """
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
      if not self.timestamps:
         return None
      before = self.index_for_timestamp_not_later_than(self, timestamp)
      if not before:
         return 0
      if before == len(self.timestamps) - 1 or self.timestamps[before] == timestamp:
         return before
      after = before + 1
      early_delta = TimeDuration(earlier_timestamp = self.timestamps[before], later_timestamp = timestamp)
      later_delta = TimeDuration(earlier_timestamp = timestamp, later_timestamp = self.timestamps[after])
      return before if early_delta <= later_delta else after

   def duration_between_timestamps(self, earlier_index, later_index):
      """Returns the duration between a pair of timestamps.

      :meta common:
      """
      if earlier_index < 0 or later_index >= len(self.timestamps) or later_index < earlier_index:
         return None
      return TimeDuration(earlier_timestamp = self.timestamps[earlier_index],
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
      """Returns the duration of the time step between the indexed timestamp and preceeding one.

      :meta common:
      """
      if index < 1 or index >= len(self.timestamps):
         return None
      return self.duration_between_timestamps(index - 1, index)

   def step_days(self, index):
      """Returns the number of days between the indexed timestamp and preceeding one."""
      delta = self.step_duration(index)
      if delta is None:
         return None
      return delta.duration.days

   # NB: Following functions modify the time series, which is dangerous if the series is in use by a model
   # Could check for relationships involving the time series and disallow changes if any found?
   def add_timestamp(self, new_timestamp, allow_insertion = False):
      """Inserts a new timestamp into the time series."""
      # todo: check that new_timestamp is in valid format (iso format + 'Z')
      if allow_insertion:
         # NB: This can insert a timestamp anywhere in the series, which will invalidate indices, possibly corrupting model
         index = self.index_for_timestamp_not_later_than(new_timestamp)
         if index is None:
            index = 0
         else:
            index += 1
         self.timestamps.insert(new_timestamp, index)
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
      duration = TimeDuration(days = days)
      self.extend_by_duration(duration)

   def datetimes(self):
      """Returns a list of python-datetime objects """
      return [dt.datetime.fromisoformat(t.rstrip('Z')) for t in self.timestamps]

   def create_xml(self, add_as_part = True, title = None, originator = None, reuse = True):
      """Create a time series node from a TimeSeries object, optionally add as part.

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

      ts_node = super().create_xml(add_as_part = False, title = title, originator = originator)

      for index in range(self.number_of_timestamps()):
         time_node = rqet.SubElement(ts_node, ns['resqml2'] + 'Time')
         time_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Timestamp')
         time_node.text = rqet.null_xml_text
         dt_node = rqet.SubElement(time_node, ns['resqml2'] + 'DateTime')
         dt_node.set(ns['xsi'] + 'type', ns['xsd'] + 'dateTime')
         dt_node.text = self.timestamp(index)

      if add_as_part:
         self.model.add_part('obj_TimeSeries', self.uuid, ts_node)

      return ts_node

   def create_time_index(self, time_index, root = None):
      """Create a time index node, including time series reference, optionally add to root.

      arguments:
         time_index (int): non-negative integer index into the time series, identifying the datetime
            being referenced by the new node
         root (optional): if present, the newly created time index xml node is added
            as a child to this node

      returns:
         the newly created time index xml node

      note:
         a time index node is typically part of a recurrent grid property object; it identifies
         the datetime relevant to the property array (or other data) by indexing into a time series
         object
      """

      assert self.uuid is not None

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

   @property
   def time_series_root(self):
      """DEPRECATED. Alias for root"""
      warnings.warn("Attribute 'time_series_root' is deprecated. Use 'root'", DeprecationWarning)
      return self.root


def selected_time_series(full_series, indices_list, title = None):
   """Returns a new TimeSeries object with timestamps selected from the full series by a list of indices."""

   selected_ts = TimeSeries(full_series.model, title = title)
   selected_ts.timestamps = [full_series.timestamps[i] for i in sorted(indices_list)]
   return selected_ts


def simplified_timestamp(timestamp):
   """Return a more readable version of the timestamp."""

   if timestamp is None:
      return None
   timestamp = cleaned_timestamp(timestamp)
   if len(timestamp) > 10:
      timestamp = timestamp[:10] + ' ' + timestamp[11:]
   return timestamp[:19]


def cleaned_timestamp(timestamp):
   """Return a cleaned version of the timestamp."""

   if timestamp is None:
      return None
   timestamp = str(timestamp)
   if len(timestamp) < 19 or timestamp[11:19] == '00:00:00':
      return timestamp[:10]
   return timestamp[:10] + 'T' + timestamp[11:19] + 'Z'


def time_series_from_list(timestamp_list, parent_model = None):
   """Create a TimeSeries object from a list of timestamps (model and node set to None)."""

   assert (len(timestamp_list) > 0)
   sorted_timestamps = sorted(timestamp_list)
   time_series = TimeSeries(parent_model = parent_model, first_timestamp = cleaned_timestamp(sorted_timestamps[0]))
   for raw_timestamp in sorted_timestamps[1:]:
      timestamp = cleaned_timestamp(raw_timestamp)
      time_series.add_timestamp(timestamp)
   return time_series


def merge_timeseries_from_uuid(model, timeseries_uuid_iter):
   """Create a TimeSeries object from an iteratable object of existing timeseries UUIDs of timeseries. iterable can be a list, array, or iteratable generator (model must be provided). The new timeseries is sorted in ascending order. Returns the new time series, the new time series uuid, and the list of timeseries objects used to generate the list"""

   reverse = False

   alltimestamps = set({})
   timeserieslist = []
   for timeseries_uuid in timeseries_uuid_iter:
      timeseriesroot = model.root(uuid = timeseries_uuid)
      assert (rqet.node_type(timeseriesroot) == 'obj_TimeSeries')

      singlets = TimeSeries(model, uuid = timeseries_uuid)
      timeserieslist.append(singlets)
      #alltimestamps.update( set(singlets.timestamps) )
      alltimestamps.update(set(singlets.datetimes()))

   sortedtimestamps = sorted(list(alltimestamps), reverse = reverse)

   new_time_series = time_series_from_list(sortedtimestamps, parent_model = model)
   new_time_series_uuid = new_time_series.uuid
   return (new_time_series, new_time_series_uuid, timeserieslist)


def time_series_from_nexus_summary(summary_file, parent_model = None):
   """Create a TimeSeries object based on time steps reported in a nexus summary file (.sum)."""

   if not summary_file:
      return None
   if not os.path.isfile(summary_file):
      log.warning('Summary file not found: ' + summary_file)
      return None
   try:
      us_date_format = True
      date_format = 'MM/DD/YYYY'
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
      if len(summary_entries) == 0:
         return None  # no entries extracted from summary file, could raise error?
      summary_entries.sort()
      if summary_entries[0][0] == 0:  # first entry is for time zero
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
   except Exception:
      log.exception('failed to create TimeSeries object from summary file: ' + summary_file)
   return None
