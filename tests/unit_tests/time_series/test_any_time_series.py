import uuid
import pytest
import logging

import resqpy.model as rq
from resqpy.time_series._any_time_series import AnyTimeSeries
from resqpy.time_series._geologic_time_series import GeologicTimeSeries

from lxml import etree

log = logging.getLogger('resqpy')


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


def test_warns_positive_year_offset(example_model_and_crs, caplog):
    # arrange
    log.setLevel('WARNING')
    model, crs = example_model_and_crs
    year_list = [-130_000_000, -57_000_000, 0, 55_000]
    gts = GeologicTimeSeries.from_year_list(model, year_list, title = 'geo time series')
    assert gts.timeframe == 'geologic'
    gts.timestamps = year_list  # force original year offsets to keep a positive value
    gts.create_xml()
    gts_uuid = gts.uuid
    model.store_epc()
    model = rq.Model(model.epc_file)
    gts = GeologicTimeSeries(model, uuid = gts_uuid)
    assert gts is not None
    assert gts.timeframe == 'geologic'
    assert len(gts.timestamps) == 4
    assert all([isinstance(t, int) for t in gts.timestamps])
    assert gts.timestamps == year_list


def test_load_from_xml_human_zero_year_offset(example_model_and_crs):
    # arrange
    xml_string = """
        <resqml2:TimeSeries
            xmlns:resqml2="http://www.energistics.org/energyml/data/resqmlv2"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            schemaVersion="2.0"
            uuid="fc2611f3-d071-9a55-3e45df8d4e24"
            xsi:type="resqml2:obj_TimeSeries">
            <resqml2:Time xsi:type="resqml2:Timestamp">
                <resqml2:DateTime xsi:type="xsd:dateTime">2000-01-01T00:00:00Z</resqml2:DateTime>
                <resqml2:YearOffset xsi:type="xsd:long">0</resqml2:YearOffset>
            </resqml2:Time>
            <resqml2:Time xsi:type="resqml2:Timestamp">
                <resqml2:DateTime xsi:type="xsd:dateTime">2000-04-01T00:00:00Z</resqml2:DateTime>
                <resqml2:YearOffset xsi:type="xsd:long">0</resqml2:YearOffset>
            </resqml2:Time>
        </resqml2:TimeSeries>
        """

    root = etree.fromstring(xml_string)
    model, crs = example_model_and_crs
    ts = Wrapper(model, root, "human")
    # act
    ts._load_from_xml()
    # assert
    assert ts.number_of_timestamps() == 2


def test_load_from_xml_human_empty_year_offset(example_model_and_crs):
    # arrange
    xml_string = """
        <resqml2:TimeSeries
            xmlns:resqml2="http://www.energistics.org/energyml/data/resqmlv2"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            schemaVersion="2.0"
            uuid="fc2611f3-d071-9a55-3e45df8d4e24"
            xsi:type="resqml2:obj_TimeSeries">
            <resqml2:Time xsi:type="resqml2:Timestamp">
                <resqml2:DateTime xsi:type="xsd:dateTime">2000-01-01T00:00:00Z</resqml2:DateTime>
                <resqml2:YearOffset xsi:type="xsd:long"></resqml2:YearOffset>
            </resqml2:Time>
            <resqml2:Time xsi:type="resqml2:Timestamp">
                <resqml2:DateTime xsi:type="xsd:dateTime">2000-04-01T00:00:00Z</resqml2:DateTime>
                <resqml2:YearOffset xsi:type="xsd:long"></resqml2:YearOffset>
            </resqml2:Time>
        </resqml2:TimeSeries>
        """

    root = etree.fromstring(xml_string)
    model, crs = example_model_and_crs
    ts = Wrapper(model, root, "human")
    # act
    ts._load_from_xml()
    # assert
    assert ts.number_of_timestamps() == 2


def test_load_from_xml_human_year_offset_set(example_model_and_crs):
    # arrange
    xml_string = """
        <resqml2:TimeSeries
            xmlns:resqml2="http://www.energistics.org/energyml/data/resqmlv2"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            schemaVersion="2.0"
            uuid="fc2611f3-d071-9a55-3e45df8d4e24"
            xsi:type="resqml2:obj_TimeSeries">
            <resqml2:Time xsi:type="resqml2:Timestamp">
                <resqml2:DateTime xsi:type="xsd:dateTime">2000-01-01T00:00:00Z</resqml2:DateTime>
                <resqml2:YearOffset xsi:type="xsd:long">-100000000</resqml2:YearOffset>
            </resqml2:Time>
            <resqml2:Time xsi:type="resqml2:Timestamp">
                <resqml2:DateTime xsi:type="xsd:dateTime">2000-04-01T00:00:00Z</resqml2:DateTime>
                <resqml2:YearOffset xsi:type="xsd:long">-200000000</resqml2:YearOffset>
            </resqml2:Time>
        </resqml2:TimeSeries>
        """

    root = etree.fromstring(xml_string)
    model, crs = example_model_and_crs
    ts = Wrapper(model, root, "human")
    # act
    with pytest.raises(AssertionError, match = "Invalid combination"):
        ts._load_from_xml()


def test_load_from_xml_geologic_empty_year_offset(example_model_and_crs):
    # arrange
    xml_string = """
        <resqml2:TimeSeries
            xmlns:resqml2="http://www.energistics.org/energyml/data/resqmlv2"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            schemaVersion="2.0"
            uuid="fc2611f3-d071-9a55-3e45df8d4e24"
            xsi:type="resqml2:obj_TimeSeries">
            <resqml2:Time xsi:type="resqml2:Timestamp">
                <resqml2:DateTime xsi:type="xsd:dateTime">2000-01-01T00:00:00Z</resqml2:DateTime>
                <resqml2:YearOffset xsi:type="xsd:long"></resqml2:YearOffset>
            </resqml2:Time>
            <resqml2:Time xsi:type="resqml2:Timestamp">
                <resqml2:DateTime xsi:type="xsd:dateTime">2000-04-01T00:00:00Z</resqml2:DateTime>
                <resqml2:YearOffset xsi:type="xsd:long"></resqml2:YearOffset>
            </resqml2:Time>
        </resqml2:TimeSeries>
        """

    root = etree.fromstring(xml_string)
    model, crs = example_model_and_crs
    ts = Wrapper(model, root, "geologic")
    # act
    with pytest.raises(AssertionError, match = "Invalid combination"):
        ts._load_from_xml()


def test_load_from_xml_geologic_zero_year_offset(example_model_and_crs):
    # arrange
    xml_string = """
        <resqml2:TimeSeries
            xmlns:resqml2="http://www.energistics.org/energyml/data/resqmlv2"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            schemaVersion="2.0"
            uuid="fc2611f3-d071-9a55-3e45df8d4e24"
            xsi:type="resqml2:obj_TimeSeries">
            <resqml2:Time xsi:type="resqml2:Timestamp">
                <resqml2:DateTime xsi:type="xsd:dateTime">2000-01-01T00:00:00Z</resqml2:DateTime>
                <resqml2:YearOffset xsi:type="xsd:long">0</resqml2:YearOffset>
            </resqml2:Time>
            <resqml2:Time xsi:type="resqml2:Timestamp">
                <resqml2:DateTime xsi:type="xsd:dateTime">2000-04-01T00:00:00Z</resqml2:DateTime>
                <resqml2:YearOffset xsi:type="xsd:long">0</resqml2:YearOffset>
            </resqml2:Time>
        </resqml2:TimeSeries>
        """

    root = etree.fromstring(xml_string)
    model, crs = example_model_and_crs
    ts = Wrapper(model, root, "geologic")
    # act
    ts._load_from_xml()
    # assert
    assert ts.number_of_timestamps() == 2


def test_load_from_xml_geologic_year_offset_set(example_model_and_crs):
    # arrange
    xml_string = """
        <resqml2:TimeSeries
            xmlns:resqml2="http://www.energistics.org/energyml/data/resqmlv2"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            schemaVersion="2.0"
            uuid="fc2611f3-d071-9a55-3e45df8d4e24"
            xsi:type="resqml2:obj_TimeSeries">
            <resqml2:Time xsi:type="resqml2:Timestamp">
                <resqml2:DateTime xsi:type="xsd:dateTime">2000-01-01T00:00:00Z</resqml2:DateTime>
                <resqml2:YearOffset xsi:type="xsd:long">1</resqml2:YearOffset>
            </resqml2:Time>
            <resqml2:Time xsi:type="resqml2:Timestamp">
                <resqml2:DateTime xsi:type="xsd:dateTime">2000-04-01T00:00:00Z</resqml2:DateTime>
                <resqml2:YearOffset xsi:type="xsd:long">2</resqml2:YearOffset>
            </resqml2:Time>
        </resqml2:TimeSeries>
        """

    root = etree.fromstring(xml_string)
    model, crs = example_model_and_crs
    ts = Wrapper(model, root, "geologic")
    # act
    ts._load_from_xml()
    # assert
    assert ts.number_of_timestamps() == 2


class Wrapper(AnyTimeSeries):
    """
    Wrapper around AnyTimeSeries to allow the manipulation of the
    xml root to allow testing of _load_from_xml
    """

    def __init__(
        self,
        model,
        root,
        timeframe,
        uuid = None,
        title = None,
        originator = None,
        extra_metadata = None,
    ):
        self.test_root = root
        self.timeframe = timeframe
        super().__init__(model, uuid, title, originator, extra_metadata)

    @property
    def root(self):
        return self.test_root
