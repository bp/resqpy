import logging

import pytest

log = logging.getLogger(__name__)
log.debug('resqml_time_series.py version ')

from resqpy.time_series import TimeSeries


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_index_for_timestamp_not_later_than_later_timestamp(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.index_for_timestamp_not_later_than('2016-01-01T00:00:00.588Z')
    # assert
    assert result == 1


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_index_for_timestamp_not_later_than_same_timestamp(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.index_for_timestamp_not_later_than('2015-02-02T00:00:00.588Z')
    # assert
    assert result == 1


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_index_for_timestamp_not_later_is_none(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.index_for_timestamp_not_later_than('2013-02-02T00:00:00.588Z')
    # assert
    assert result is None


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_index_for_timestamp_not_earlier_than_earlier_timestamp(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.index_for_timestamp_not_earlier_than('2013-02-02T00:00:00.588Z')
    # assert
    assert result == 0


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_index_for_timestamp_not_earlier_than_later_timestamp(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.index_for_timestamp_not_earlier_than('2016-02-02T00:00:00.588Z')
    # assert
    assert result is None


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_index_for_timestamp_not_earlier_than_same_timestamp(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.index_for_timestamp_not_earlier_than('2015-02-02T00:00:00.588Z')
    # assert
    assert result == 1


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_index_for_timestamp_closest_to_empty_timestamps(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = None
    # act
    result = any_time_series.index_for_timestamp_closest_to('2000-01-01T00:00:00.588Z')
    # assert
    assert result is None


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_index_for_timestamp_closest_to_timestamp_too_early(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.index_for_timestamp_closest_to('2013-02-02T00:00:00.588Z')
    # assert
    assert result == 0


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_index_for_timestamp_closest_to_timestamp_len(mocker, timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    index_for_timestamp_not_later_mock = mocker.Mock(return_value = 1)
    any_time_series.index_for_timestamp_not_later_than = index_for_timestamp_not_later_mock
    # act
    result = any_time_series.index_for_timestamp_closest_to('2013-02-02T00:00:00.588Z')
    # assert
    assert result == 1


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_index_for_timestamp_closest_to_timestamp_equals_before(mocker, timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    index_for_timestamp_not_later_mock = mocker.Mock(return_value = 0)
    any_time_series.index_for_timestamp_not_later_than = index_for_timestamp_not_later_mock
    # act
    result = any_time_series.index_for_timestamp_closest_to('2014-01-01T00:00:00.588Z')
    # assert
    assert result == 0
    index_for_timestamp_not_later_mock.assert_called_once_with('2014-01-01T00:00:00.588Z')


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_index_for_timestamp_closest_to_timestamp_early_delta(mocker, timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    index_for_timestamp_not_later_mock = mocker.Mock(return_value = 0)
    any_time_series.index_for_timestamp_not_later_than = index_for_timestamp_not_later_mock
    # act
    result = any_time_series.index_for_timestamp_closest_to('2014-02-02T00:00:00.588Z')
    # assert
    assert result == 0
    index_for_timestamp_not_later_mock.assert_called_once_with('2014-02-02T00:00:00.588Z')


@pytest.mark.parametrize('timestamps',
                         [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z', '2016-03-02T00:00:00.588Z']])
def test_index_for_timestamp_closest_to_timestamp_later_delta(mocker, timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    index_for_timestamp_not_later_mock = mocker.Mock(return_value = 1)
    any_time_series.index_for_timestamp_not_later_than = index_for_timestamp_not_later_mock
    # act
    result = any_time_series.index_for_timestamp_closest_to('2016-01-01T00:00:00.588Z')
    # assert
    assert result == 2
    index_for_timestamp_not_later_mock.assert_called_once_with('2016-01-01T00:00:00.588Z')


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_duration_since_start_invalid_index(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.duration_since_start(-1)
    # assert
    assert result is None


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_duration_since_start_later_index(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.duration_since_start(2)
    # assert
    assert result is None


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_step_duration_invalid_index(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.step_duration(0)
    # assert
    assert result is None


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2015-02-02T00:00:00.588Z']])
def test_step_duration_later_index(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.step_duration(2)
    # assert
    assert result is None


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2014-01-03T00:00:00.588Z']])
def test_step_duration_between_index(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.step_duration(1)
    # assert
    assert result.duration.days == 2


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2014-01-03T00:00:00.588Z']])
def test_step_days_none(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.step_days(0)
    # assert
    assert result is None


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2014-01-03T00:00:00.588Z']])
def test_step_days_number(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    # act
    result = any_time_series.step_days(1)
    # assert
    assert result == 2


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2014-01-03T00:00:00.588Z']])
def test_add_timestamp_before(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    new_timestamp = '2013-01-01T00:00:00.588Z'
    # act
    any_time_series.add_timestamp(new_timestamp, allow_insertion = True)
    # assert
    assert any_time_series.timestamps[0] == new_timestamp
    assert any_time_series.timestamps[1] == '2014-01-01T00:00:00.588Z'
    assert any_time_series.timestamps[2] == '2014-01-03T00:00:00.588Z'


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2014-01-03T00:00:00.588Z']])
def test_add_timestamp_after(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    new_timestamp = '2015-01-01T00:00:00.588Z'
    # act
    any_time_series.add_timestamp(new_timestamp, allow_insertion = True)
    # assert
    assert any_time_series.timestamps[0] == '2014-01-01T00:00:00.588Z'
    assert any_time_series.timestamps[1] == '2014-01-03T00:00:00.588Z'
    assert any_time_series.timestamps[2] == new_timestamp


@pytest.mark.parametrize('timestamps', [['2014-01-01T00:00:00.588Z', '2014-01-03T00:00:00.588Z']])
def test_add_timestamp_between(timestamps, example_model_and_crs):
    # arrange
    model, crs = example_model_and_crs
    any_time_series = TimeSeries(model)
    any_time_series.timestamps = timestamps
    new_timestamp = '2014-01-02T00:00:00.588Z'
    # act
    any_time_series.add_timestamp(new_timestamp, allow_insertion = True)
    # assert
    assert any_time_series.timestamps[0] == '2014-01-01T00:00:00.588Z'
    assert any_time_series.timestamps[1] == new_timestamp
    assert any_time_series.timestamps[2] == '2014-01-03T00:00:00.588Z'
