import pytest

import resqpy.time_series as rqts

from resqpy import model as makemodel

#reversemode=True

#@pytest.mark.parametrize("reversemode", [True, False])
def test_merge_timeseries():
    model=makemodel.Model(create_basics=True)

    timestamps1=['2020-01-01T00:00:00Z', '2015-01-01T00:00:00Z']
    timestamps2=['2021-01-01T00:00:00Z', '2014-01-01T00:00:00Z']


    timeseries1=rqts.time_series_from_list(timestamps1, parent_model=model)
    timeseries1.set_model(model)
    timeseries1.create_xml()
    timeseries2=rqts.time_series_from_list(timestamps2, parent_model=model)
    timeseries2.set_model(model)
    timeseries2.create_xml()
    

    sortedtimestamps=sorted(timeseries1.datetimes() + timeseries2.datetimes())

    timeseries_uuids=(timeseries1.uuid, timeseries2.uuid)

    newts, newtsuuid, timeserieslist = rqts.merge_timeseries_from_uuid(model, timeseries_uuids)

    assert len(newts.timestamps) == len(timeseries1.timestamps) + len(timeseries2.timestamps)

    for idx, timestamp in enumerate(newts.datetimes()):
        assert timestamp==sortedtimestamps[idx]
    

    #Now test duplication doesn't create duplicate timestamps during merge, I want a unique set of merged timestamps
    
    timestamps3=[timestamp for timestamp in timestamps1]
    timeseries3=rqts.time_series_from_list(timestamps3, parent_model=model)
    timeseries3.set_model(model)
    timeseries3.create_xml()
    
    timeseries_uuids=(timeseries1.uuid, timeseries2.uuid, timeseries3.uuid)
    

    newts2, _, _ = rqts.merge_timeseries_from_uuid(model, timeseries_uuids)

    assert len(newts.timestamps) == len(newts2.timestamps)
    
    for idx, timestamp in enumerate(newts2.datetimes()):
        assert timestamp==sortedtimestamps[idx]
    
    
    return True

