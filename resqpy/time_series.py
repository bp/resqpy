"""time_series.py: RESQML time series class."""

version = '19th May 2021'

# Nexus is a registered trademark of the Halliburton Company

# At present, no time zone information is handled

import logging
log = logging.getLogger(__name__)
log.debug('resqml_time_series.py version ' + version)

import os
import datetime as dt
# import xml.etree.ElementTree as et
# from lxml import etree as et

import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu
from resqpy.olio.xml_namespaces import curly_namespace as ns



class TimeDuration():
   """A thin wrapper around python's datatime timedelta objects (not a RESQML class)."""

   def __init__(self, days = None, hours = None, minutes = None, seconds = None,
                earlier_timestamp = None, later_timestamp = None):
      """Create a TimeDuration object either from days and seconds or from a pair of timestamps."""

      # for negative durations, earlier_timestamp should be later than later_timestamp
      # or days, hours etc. should typically all be non-positive
      # ie. days = -1, hours = -12 will be a negative one and a half day duration
      # whilst days = -1, hours = 12 will be a negative half day duration
      self.duration = None
      if earlier_timestamp is not None and later_timestamp is not None:
         if earlier_timestamp.endswith('Z'): earlier_timestamp = earlier_timestamp[:-1]   # Trailing Z is not part of iso format
         if later_timestamp.endswith('Z'): later_timestamp = later_timestamp[:-1]
         dt_earlier = dt.datetime.fromisoformat(earlier_timestamp)
         dt_later   = dt.datetime.fromisoformat(later_timestamp)
         self.duration = dt_later - dt_earlier
      else:
         if days is None: days = 0
         if hours is None: hours = 0
         if minutes is None: minutes = 0
         if seconds is None: seconds = 0
         self.duration = dt.timedelta(days = days, hours = hours, minutes = minutes, seconds = seconds)


   def timestamp_after_duration(self, earlier_timestamp):
      """Create a new timestamp from this duration and an earlier timestamp."""

      if earlier_timestamp.endswith('Z'): earlier_timestamp = earlier_timestamp[:-1]
      dt_earlier = dt.datetime.fromisoformat(earlier_timestamp)
      dt_result = dt_earlier + self.duration
      return dt_result.isoformat() + 'Z'


   def timestamp_before_duration(self, later_timestamp):
      """Create a new timestamp from this duration and a later timestamp."""

      if later_timestamp.endswith('Z'): later_timestamp = later_timestamp[:-1]
      dt_later = dt.datetime.fromisoformat(later_timestamp)
      dt_result = dt_later - self.duration
      return dt_result.isoformat() + 'Z'



class TimeSeries():
   """Class for RESQML Time Series within RESQML model object."""

   def __init__(self, parent_model, extract_from_xml = True, time_series_root = None,
                first_timestamp = None, daily = None, monthly = None, quarterly = None, yearly = None):
      """Create a TimeSeries object, either from a time series node in parent model, or from given data.

      arguments:
         parent_model (model.Model): the resqpy model to which the time series will belong
         extract_from_xml (boolean, default True): if True, the time series is populated from the xml
            for an existing part in the model
         time_series_root (xml node, optional): if extract_from_xml is True, then this argument is
            usually passed to identify the xml root node; if absent the root for the 'main' time series
            in the model is used
         first_time_stamp (str, optional): the first timestamp (in RESQML format) if not loading from xml;
            this and the remaining arguments are ignored if loading from xml
         daily (non-negative int, optional): the number of one day interval timesteps to start the series
         monthly (non-negative int, optional): the number of 30 day interval timesteps to follow the daily
            timesteps
         quarterly (non-negative int, optional): the number of 90 day interval timesteps to follow the
            monthly timesteps
         yearly (non-negative int, optional): the number of 365 day interval timesteps to follow the
            quarterly timesteps

      returns:
         newly instantiated TimeSeries object

      note:
         a new bespoke time series can be populated by passing the first timestamp here and using the
         add_timestamp() and/or extend_by...() methods

      :meta common:
      """

      self.model = parent_model
      self.time_series_root = time_series_root
      self.timestamps = []    # ordered list of timestamp strings in resqml/iso format
      self.uuid = None
      if extract_from_xml:
         self.time_series_root = self.model.resolve_time_series_root(time_series_root)
         if self.time_series_root is None: return  # no time series in model
         self.uuid = self.time_series_root.attrib['uuid']
         for child in self.time_series_root:
            if rqet.stripped_of_prefix(child.tag) != 'Time': continue
            dt_node = rqet.find_tag(child, 'DateTime')
            if dt_node is None: continue          # could raise an exception here
            self.timestamps.append(dt_node.text)  # todo: trim and check timestamp
            self.timestamps.sort()
      elif first_timestamp is not None:
         self.timestamps.append(first_timestamp)  # todo: check format of first_timestamp
         if daily is not None:
            for _ in range(daily):
               self.extend_by_days(1)
         if monthly is not None:
            for _ in range(monthly):
               self.extend_by_days(30)
         if quarterly is not None:
            for _ in range(quarterly):
               self.extend_by_days(90)   # could use 91
         if yearly is not None:
            for _ in range(yearly):
               self.extend_by_days(365)  # could use 360
      if self.uuid is None: self.uuid = bu.new_uuid()


   def is_equivalent(self, other_ts, tol_seconds = 1):
      """Returns True if the this timestep series is essentially identical to the other; otherwise False."""

      if other_ts is None: return False
      if self is other_ts: return True
      if bu.matching_uuids(self.uuid, other_ts.uuid): return True
      if self.number_of_timestamps() != other_ts.number_of_timestamps(): return False
      tolerance = TimeDuration(seconds = tol_seconds)
      for t_index in range(self.number_of_timestamps()):
         diff = TimeDuration(earlier_timestamp = self.timestamps[t_index],
                             later_timestamp = other_ts.timestamps[t_index])
         if abs(diff.duration) > tolerance.duration: return False
      return True


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


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
      if index < 0 or index >= len(self.timestamps): return None
      return self.timestamps[index]


   def last_timestamp(self):
      """Returns the last timestamp in the series.

      :meta common:
      """

      index = len(self.timestamps) - 1
      if index < 0: return None
      return self.timestamps[index]


   def index_for_timestamp(self, timestamp):
      """Returns the index for a given timestamp.

      note:
         this method uses a simplistic string comparison

      :meta common:
      """

      for index in range(len(self.timestamps)):
         if self.timestamps[index] == timestamp: return index
      return None


   def index_for_timestamp_not_later_than(self, timestamp):
      """Returns the index of the latest timestamp that is not later than the specified timestamp.

      :meta common:
      """

      index = len(self.timestamps) - 1
      while (index >= 0) and (self.timestamps[index] > timestamp): index -= 1
      if index < 0: return None
      return index


   def duration_between_timestamps(self, earlier_index, later_index):
      """Returns the duration between a pair of timestamps.

      :meta common:
      """

      if earlier_index < 0 or later_index >= len(self.timestamps) or later_index < earlier_index: return None
      return TimeDuration(earlier_timestamp = self.timestamps[earlier_index],
                          later_timestamp = self.timestamps[later_index])


   def days_between_timestamps(self, earlier_index, later_index):
      """Returns the number of whole days between a pair of timestamps, as an integer."""

      delta = self.duration_between_timestamps(earlier_index, later_index)
      if delta is None: return None
      return delta.duration.days


   def duration_since_start(self, index):
      """Returns the duration between the start of the time series and the indexed timestamp.

      :meta common:
      """

      if index < 0 or index >= len(self.timestamps): return None
      return self.duration_between_timestamps(0, index)


   def days_since_start(self, index):
      """Returns the number of days between the start of the time series and the indexed timestamp."""

      return self.duration_since_start(index).duration.days


   def step_duration(self, index):
      """Returns the duration of the time step between the indexed timestamp and preceeding one.

      :meta common:
      """

      if index < 1 or index >= len(self.timestamps): return None
      return self.duration_between_timestamps(index - 1, index)


   def step_days(self, index):
      """Returns the number of days between the indexed timestamp and preceeding one."""

      delta = self.step_duration(index)
      if delta is None: return None
      return delta.duration.days


   # NB: Following functions modify the time series, which is dangerous if the series is in use by a model
   # Could check for relationships involving the time series and disallow changes if any found?
   def add_timestamp(self, new_timestamp, allow_insertion = False):
      """Inserts a new timestamp into the time series."""

      # todo: check that new_timestamp is in valid format (iso format + 'Z')
      if allow_insertion:
         # NB: This can insert a timestamp anywhere in the series, which will invalidate indices, possibly corrupting model
         index = self.index_for_timestamp_not_later_than(new_timestamp)
         if index is None: index = 0
         else: index += 1
         self.timestamps.insert(new_timestamp, index)
      else:
         last = self.last_timestamp()
         if last is not None: assert(new_timestamp > self.last_timestamp())
         self.timestamps.append(new_timestamp)


   def extend_by_duration(self, duration):
      """Adds a timestamp to the end of the series, at duration beyond the last timestamp."""

      assert(duration.duration.days >= 0)    # duration may not be negative
      assert(len(self.timestamps) > 0)       # there must be something to extend from
      self.timestamps.append(duration.timestamp_after_duration(self.last_timestamp()))


   def extend_by_days(self, days):
      """Adds a timestamp to the end of the series, at a duration of days beyond the last timestamp."""

      duration = TimeDuration(days = days)
      self.extend_by_duration(duration)


   def create_xml(self, add_as_part = True, root = None,
                  title = 'time series', originator = None):
      """Create a time series node from a TimeSeries object, optionally add as part.

      arguments:
         add_as_part (boolean, default True): if True, the newly created xml node is added as a part
            in the model
         root (optional, usually None): if present, the newly created xml node is appended as a child
            in this node
         title (string): used as the citation Title text for the new time series node
         originator (string, optional): the name of the human being who created the time series object;
            default is to use the login name

      returns:
         the newly created time series xml node

      :meta common:
      """

      ts_node = self.model.new_obj_node('TimeSeries')

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(ts_node.attrib['uuid'])
      else:
         ts_node.attrib['uuid'] = str(self.uuid)

      self.model.create_citation(root = ts_node, title = title, originator = originator)

      for index in range(self.number_of_timestamps()):
         time_node = rqet.SubElement(ts_node, ns['resqml2'] + 'Time')
         time_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Timestamp')
         time_node.text = rqet.null_xml_text
         dt_node = rqet.SubElement(time_node, ns['resqml2'] + 'DateTime')
         dt_node.set(ns['xsi'] + 'type', ns['xsd'] + 'dateTime')
         dt_node.text = self.timestamp(index)

      if root is not None: root.append(ts_node)
      if add_as_part: self.model.add_part('obj_TimeSeries', self.uuid, ts_node)

      self.time_series_root = ts_node

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

      if root is not None: root.append(t_node)
      return t_node



def selected_time_series(full_series, indices_list):
   """Returns a new TimeSeries object with timestamps selected from the full series by a list of indices."""

   selected_ts = TimeSeries(full_series.model, extract_from_xml = False)
   for full_index in indices_list:
      selected_ts.add_timestamp(full_series.timestamp(full_index))
   return selected_time_series


def simplified_timestamp(timestamp):
   """Return a more readable version of the timestamp."""

   if timestamp is None: return None
   timestamp = cleaned_timestamp(timestamp)
   if len(timestamp) > 10: timestamp = timestamp[:10] + ' ' + timestamp[11:]
   return timestamp[:19]


def cleaned_timestamp(timestamp):
   """Return a cleaned version of the timestamp."""

   if timestamp is None: return None
   timestamp = str(timestamp)
   if len(timestamp) < 19 or timestamp[11:19] == '00:00:00': return timestamp[:10]
   return timestamp[:10] + 'T' + timestamp[11:19] + 'Z'


def time_series_from_list(timestamp_list, parent_model = None):
   """Create a TimeSeries object from a list of timestamps (model and node set to None)."""

   assert(len(timestamp_list) > 0)
   sorted_timestamps = sorted(timestamp_list)
   time_series = TimeSeries(parent_model = parent_model, extract_from_xml = False,
                            first_timestamp = cleaned_timestamp(sorted_timestamps[0]))
   for raw_timestamp in sorted_timestamps[1:]:
      timestamp = cleaned_timestamp(raw_timestamp)
      time_series.add_timestamp(timestamp)
   return time_series


def time_series_from_nexus_summary(summary_file, parent_model = None):
   """Create a TimeSeries object based on time steps reported in a nexus summary file (.sum)."""

   if not summary_file: return None
   if not os.path.isfile(summary_file):
      log.warning('Summary file not found: ' + summary_file)
      return None
   try:
      us_date_format = True
      date_format = 'MM/DD/YYYY'
      summary_entries = []      # list of (time step no., cumulative time (days), date (dd/mm/yyyy))
      with open(summary_file, "r") as fp:
         while True:
            line = fp.readline()
            if len(line) == 0: break   # end of file
            words = line.split()
            if len(words) < 5: continue
            if not words[0].isdigit():
               if words[0].upper() == 'NO.' and len(words[4]) == 10:
                  date_format = words[4].upper()
                  us_date_format = (date_format == 'MM/DD/YYYY')
                  if not us_date_format: assert date_format == 'DD/MM/YYYY'
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
      if len(summary_entries) == 0: return None      # no entries extracted from summary file, could raise error?
      summary_entries.sort()
      if summary_entries[0][0] == 0:   # first entry is for time zero
         tz_date = summary_entries[0][2]
         summary_entries.pop(0)
      else:   # back calculate time zero from first entry
         delta = dt.timedelta(days = summary_entries[0][1])
         tz_date = summary_entries[0][2] - delta
      time_series = TimeSeries(parent_model = parent_model, extract_from_xml = False,
                               first_timestamp = tz_date.isoformat() + 'T00:00:00Z')
      last_timestep = 0
      last_cumulative_time = 0.0
      for entry in summary_entries:
         if entry[0] <= last_timestep:
            log.warning('ignoring out of sequence time step in summary file: ' + str(entry[0]))
            continue
         while entry[0] > last_timestep + 1:   # add filler steps
            time_series.extend_by_days(0)
            last_timestep += 1
         if entry[1] < last_cumulative_time:   # should never happen
            time_series.extend_by_days(0)
         else:
            time_series.extend_by_days(entry[1] - last_cumulative_time)
            last_cumulative_time = entry[1]
         last_timestep += 1
      return time_series
   except:
      log.exception('failed to create TimeSeries object from summary file: ' + summary_file)
   return None
