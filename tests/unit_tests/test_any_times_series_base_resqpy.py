from resqpy.time_series.any_time_series_base_resqpy import AnyTimeSeries
import uuid


def test_is_equivalent_other_ts_is_none(example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = AnyTimeSeries(model)
    # act
    result = any_time_series.is_equivalent(None)
    # assert
    assert result == False


def test_is_equivalent_self_is_other_ts(example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = AnyTimeSeries(model)
    # act
    result = any_time_series.is_equivalent(any_time_series)
    # assert
    assert result == True


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
    assert result == True


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
    assert result == False


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
    assert result == False


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
    assert result == True
