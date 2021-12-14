from resqpy.time_series._any_time_series import AnyTimeSeries
import uuid
import pytest


def test_is_equivalent_other_ts_is_none(example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = AnyTimeSeries(model)
    # act
    result = any_time_series.is_equivalent(None)
    # assert
    assert result is False


def test_is_equivalent_self_is_other_ts(example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = AnyTimeSeries(model)
    # act
    result = any_time_series.is_equivalent(any_time_series)
    # assert
    assert result is True


def test_is_equivalent_matching_uuids(example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = AnyTimeSeries(model)
    test_time_series = AnyTimeSeries(model)
    test_uuid = uuid.uuid4()
    any_time_series.uuid = test_uuid
    test_time_series.uuid = test_uuid
    # act
    result = any_time_series.is_equivalent(test_time_series)
    # assert
    assert result is True


def test_is_equivalent_none_matching_uuids_none_matching_timestamps(example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = AnyTimeSeries(model)
    test_time_series = AnyTimeSeries(model)
    test_uuid = uuid.uuid4()
    any_time_series.uuid = test_uuid
    test_time_series.uuid = uuid.uuid4()
    any_time_series.timestamps = [1, 2]
    test_time_series.timestamps = [1, 2, 3]
    # act
    result = any_time_series.is_equivalent(test_time_series)
    # assert
    assert result is False


def test_is_equivalent_matching_uuids_matching_timestamps_none_matching_timeframe(example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = AnyTimeSeries(model)
    test_time_series = AnyTimeSeries(model)
    test_uuid = uuid.uuid4()
    any_time_series.uuid = test_uuid
    test_time_series.uuid = uuid.uuid4()
    any_time_series.timestamps = [1, 2]
    test_time_series.timestamps = [1, 2]
    timeframe = 'test'
    test_timeframe = 'test_1'
    any_time_series.timeframe = timeframe
    test_time_series.timeframe = test_timeframe
    # act
    result = any_time_series.is_equivalent(test_time_series)
    # assert
    assert result is False


def test_is_equivalent_matching_uuids_matching_timestamps_matching_timeframe(example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = AnyTimeSeries(model)
    test_time_series = AnyTimeSeries(model)
    test_uuid = uuid.uuid4()
    any_time_series.uuid = test_uuid
    test_time_series.uuid = uuid.uuid4()
    any_time_series.timestamps = [1, 2]
    test_time_series.timestamps = [1, 2]
    timeframe = 'test'
    test_timeframe = timeframe
    any_time_series.timeframe = timeframe
    test_time_series.timeframe = test_timeframe
    # act
    result = any_time_series.is_equivalent(test_time_series)
    # assert
    assert result is None


@pytest.mark.parametrize('timestamp', [['2014-01-01T00:00:00.588Z']])
def test_timestamp(timestamp, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = AnyTimeSeries(model)
    any_time_series.timestamps = timestamp
    # act
    result = any_time_series.timestamp(1)
    # assert
    assert result is None


@pytest.mark.parametrize('timestamp', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_valid_timestamp(timestamp, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = AnyTimeSeries(model)
    any_time_series.timestamps = timestamp
    # act
    result = any_time_series.timestamp(0)
    # assert
    assert result == '2014-01-01T00:00:00.588Z'


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_itr_timestamp(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = AnyTimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.iter_timestamps()
    # assert
    assert list(result) == ['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_index_timestamp(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = AnyTimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.index_for_timestamp('2014-01-01T00:00:00.588Z')
    # assert
    assert result == 0


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_index_timestamp_not_present(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = AnyTimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.index_for_timestamp('2016-01-01T00:00:00.588Z')
    # assert
    assert result is None
