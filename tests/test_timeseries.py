import pytest

import os

import resqpy.model as rq
import resqpy.time_series as rqts

#reversemode=True


#@pytest.mark.parametrize("reversemode", [True, False])
def test_merge_timeseries():
   model = rq.Model(create_basics = True)

   timestamps1 = ['2020-01-01T00:00:00Z', '2015-01-01T00:00:00Z']
   timestamps2 = ['2021-01-01T00:00:00Z', '2014-01-01T00:00:00Z']

   timeseries1 = rqts.time_series_from_list(timestamps1, parent_model = model)
   timeseries1.set_model(model)
   timeseries1.create_xml()
   timeseries2 = rqts.time_series_from_list(timestamps2, parent_model = model)
   timeseries2.set_model(model)
   timeseries2.create_xml()

   sortedtimestamps = sorted(timeseries1.datetimes() + timeseries2.datetimes())

   timeseries_uuids = (timeseries1.uuid, timeseries2.uuid)

   newts, newtsuuid, timeserieslist = rqts.merge_timeseries_from_uuid(model, timeseries_uuids)

   assert len(newts.timestamps) == len(timeseries1.timestamps) + len(timeseries2.timestamps)

   for idx, timestamp in enumerate(newts.datetimes()):
      assert timestamp == sortedtimestamps[idx]

   #Now test duplication doesn't create duplicate timestamps during merge, I want a unique set of merged timestamps

   timestamps3 = [timestamp for timestamp in timestamps1]
   timeseries3 = rqts.time_series_from_list(timestamps3, parent_model = model)
   timeseries3.set_model(model)
   timeseries3.create_xml()

   timeseries_uuids = (timeseries1.uuid, timeseries2.uuid, timeseries3.uuid)

   newts2, _, _ = rqts.merge_timeseries_from_uuid(model, timeseries_uuids)

   assert len(newts.timestamps) == len(newts2.timestamps)

   for idx, timestamp in enumerate(newts2.datetimes()):
      assert timestamp == sortedtimestamps[idx]

   return True


def test_time_series_from_list(tmp_path):
   epc = os.path.join(tmp_path, 'ts_list.epc')
   model = rq.new_model(epc)
   ts_list = ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01']
   ts = rqts.time_series_from_list(ts_list, parent_model = model)
   assert ts.number_of_timestamps() == 4
   assert ts.days_between_timestamps(0, 3) == 31 + 28 + 31
   ts.create_xml()
   model.store_epc()
   model = rq.Model(epc)
   ts_uuid = model.uuid(obj_type = 'TimeSeries')
   assert ts_uuid is not None
   ts = rqts.TimeSeries(model, uuid = ts_uuid)
   assert ts.number_of_timestamps() == 4
   assert ts.days_between_timestamps(0, 3) == 31 + 28 + 31


def test_time_series_from_args(tmp_path):
   epc = os.path.join(tmp_path, 'ts_args.epc')
   model = rq.new_model(epc)
   ts = rqts.TimeSeries(model,
                        first_timestamp = '1963-08-23',
                        daily = 8,
                        monthly = 4,
                        quarterly = 8,
                        yearly = 5,
                        title = 'late 60s')
   assert ts.number_of_timestamps() == 26
   assert ts.days_between_timestamps(2, 3) == 1
   assert ts.days_between_timestamps(0, 25) == 8 + 4 * 30 + 8 * 90 + 5 * 365
   ts.create_xml()
   model.store_epc()
   model = rq.Model(epc)
   ts_uuid = model.uuid(obj_type = 'TimeSeries')
   assert ts_uuid is not None
   ts = rqts.TimeSeries(model, uuid = ts_uuid)
   assert ts.number_of_timestamps() == 26
   assert ts.days_between_timestamps(2, 3) == 1
   assert ts.days_between_timestamps(0, 25) == 8 + 4 * 30 + 8 * 90 + 5 * 365
