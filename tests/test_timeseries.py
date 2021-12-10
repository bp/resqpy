import os

import pytest

import resqpy.model as rq
import resqpy.time_series as rqts


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

    # Now test duplication doesn't create duplicate timestamps during merge, I want a unique set of merged timestamps

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
    ts = rqts.any_time_series(model, uuid = ts_uuid)
    assert isinstance(ts, rqts.TimeSeries)
    assert ts.number_of_timestamps() == 4
    assert ts.days_between_timestamps(0, 3) == 31 + 28 + 31
    assert ts.timeframe == 'human'


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


def test_geologic_time_series(tmp_path):
    epc = os.path.join(tmp_path, 'ts_geologic.epc')
    model = rq.new_model(epc)
    # Cretaceous Age start times, in Ma, with random use of sign to check it is ignored
    ma_list = (145, 72.1, -83.6, 86.3, 89.8, 93.9, 100.5, -113, -125, 129.4, 132.9, 139.8)
    ts_list = [int(round(ma * 1000000)) for ma in ma_list]
    ts_list_2 = [int(round(ma * 2000000)) for ma in ma_list]
    ts = rqts.time_series_from_list(ts_list, parent_model = model)
    assert ts.number_of_timestamps() == 12
    ts.create_xml()
    ts_2 = rqts.GeologicTimeSeries.from_year_list(model, year_list = ts_list_2, title = 'using class method')
    ts_2.create_xml()
    model.store_epc()
    model = rq.Model(epc)
    ts_uuids = model.uuids(obj_type = 'TimeSeries')
    assert ts_uuids is not None and len(ts_uuids) == 2
    for ts_uuid in ts_uuids:
        ts = rqts.any_time_series(model, uuid = ts_uuid)
        assert isinstance(ts, rqts.GeologicTimeSeries)
        assert ts.timeframe == 'geologic'
        assert ts.number_of_timestamps() == 12
        assert ((ts.timestamps[0] == -145000000 and ts.timestamps[-1] == -72100000) or
                (ts.timestamps[0] == -145000000 * 2 and ts.timestamps[-1] == -72100000 * 2))


def test_geologic_time_str_fails_when_not_int():
    with pytest.raises(AssertionError):
        rqts.geologic_time_str('hello world')


@pytest.mark.parametrize('years_value, expected_result', [(10_000_000, '-10 Ma'), (1_000_000, '-1 Ma'),
                                                          (11_000_000, '-11 Ma'), (1, '-0.000 Ma'),
                                                          (1_500_000, '-1.500 Ma'), (15_060_000, '-16 Ma'),
                                                          (15_040_000, '-16 Ma')])
def test_geologic_time_str(years_value, expected_result):
    # arrange
    test_years_value = years_value
    # act
    result = rqts.geologic_time_str(test_years_value)
    # assert
    assert result == expected_result
