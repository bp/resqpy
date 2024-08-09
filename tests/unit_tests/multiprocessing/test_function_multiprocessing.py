import pytest
import os
import numpy as np

import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.organize as rqo
import resqpy.time_series as rqts
import resqpy.property as rqp
import resqpy.well as rqw
import resqpy.multi_processing._multiprocessing as rqmp
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet


def make_feature(name, index, parent_tmp_dir):
    epc = os.path.join(parent_tmp_dir, f'mp_test_{index}.epc')
    model = rq.new_model(epc)
    feat = rqo.OrganizationFeature(model, feature_name = f'{name} {index}', organization_kind = 'structural')
    feat.create_xml()
    model.store_epc()
    return (index, True, epc, [feat.uuid])


def make_wellbore_frame_with_time_series_prop(x, y, t0_value, t2_value, index, parent_tmp_dir):
    epc = os.path.join(parent_tmp_dir, f'mp_ts_test_{index}.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    mdd = rqw.MdDatum(model, crs_uuid = crs.uuid, location = (x, y, 0.0), title = f'datum {index}')
    mdd.create_xml()
    traj = rqw.Trajectory(model,
                          crs_uuid = crs.uuid,
                          md_datum = mdd,
                          mds = (0.0, 1000.0),
                          control_points = [(x, y, 0.0), (x, y, 1000.0)],
                          length_uom = 'm',
                          well_name = f'well {index}')
    traj.write_hdf5()
    traj.create_xml()
    wbf = rqw.WellboreFrame(model, trajectory = traj, mds = (500.0, 600.0), title = f'frame {index}')
    wbf.write_hdf5()
    wbf.create_xml()
    ts = rqts.TimeSeries(model, first_timestamp = '2024-08-23', quarterly = 2, title = 'frame time series')
    ts.create_xml()
    prop_uuids = []
    for ti in range(3):
        v = (t0_value, 0.5 * (t0_value + t2_value), t2_value)[ti]
        prop = rqp.Property.from_array(model,
                                       cached_array = np.array((v,), dtype = float),
                                       source_info = 'reaspy test suite',
                                       keyword = 'framed property',
                                       support_uuid = wbf.uuid,
                                       property_kind = 'pressure',
                                       discrete = False,
                                       uom = 'kPa',
                                       indexable_element = 'intervals',
                                       time_series_uuid = ts.uuid,
                                       time_index = ti)
        assert prop.uuid is not None
        prop_uuids.append(prop.uuid)
    model.store_epc()
    uuids = [crs.uuid, mdd.uuid, traj.uuid, wbf.uuid, ts.uuid] + prop_uuids
    return (index, True, epc, uuids)


def test_fn_multiprocessing(tmp_path):
    combo_epc = str(tmp_path / 'combo.epc')
    args_list = []
    n = 7
    for _ in range(n):
        args_list.append({'name': 'structure'})
    good = rqmp.function_multiprocessing(make_feature,
                                         kwargs_list = args_list,
                                         recombined_epc = combo_epc,
                                         cluster = None,
                                         consolidate = True,
                                         require_success = True,
                                         tmp_dir_path = tmp_path,
                                         backend = 'dask')
    assert len(good) == n
    assert all(good)
    m = rq.Model(combo_epc)
    names = m.titles(obj_type = 'OrganizationFeature')
    assert len(names) == n
    assert all([t.startswith('structure') for t in names])


def test_ts_recombination(tmp_path):
    combo_epc = str(tmp_path / 'ts_combo.epc')
    args_list = [{
        'x': 45.67,
        'y': 56.78,
        't0_value': 1230.0,
        't2_value': 1250.0
    }, {
        'x': 12.34,
        'y': 23.45,
        't0_value': 1550.0,
        't2_value': 1600.0
    }, {
        'x': 78.90,
        'y': 89.00,
        't0_value': 760.0,
        't2_value': 790.0
    }]
    good = rqmp.function_multiprocessing(make_wellbore_frame_with_time_series_prop,
                                         kwargs_list = args_list,
                                         recombined_epc = combo_epc,
                                         cluster = None,
                                         consolidate = True,
                                         require_success = True,
                                         tmp_dir_path = tmp_path,
                                         backend = 'dask')
    assert len(good) == 3
    assert all(good)
    m = rq.Model(combo_epc)
    ts_uuids = m.uuids(obj_type = 'TimeSeries')
    assert len(ts_uuids) == 1, 'time series failed to consolidate'
    ts = rqts.TimeSeries(m, uuid = ts_uuids[0])
    assert len(ts.timestamps) == 3
    prop_uuids = m.uuids(obj_type = 'ContinuousProperty')
    assert len(prop_uuids) == 9
    ts_prop_parts = m.parts_list_related_to_uuid_of_type(ts.uuid, 'ContinuousProperty')
    assert len(ts_prop_parts) == 9
    for pp in ts_prop_parts:
        root = m.root_for_part(pp)
        assert root is not None
        ref_nodes = rqet.list_obj_references(root)
        assert len(ref_nodes) == 2
        for ref_node in ref_nodes:
            referred_root = m.referenced_node(ref_node)
            assert referred_root is not None
            referred_uuid = m.uuid_for_root(referred_root)
            assert referred_uuid is not None
            ref_type = m.type_of_uuid(referred_uuid, strip_obj = True)
            assert ref_type in ['TimeSeries', 'WellboreFrameRepresentation']
            if ref_type != 'TimeSeries':
                continue
            assert bu.matching_uuids(referred_uuid, ts.uuid)
    # TODO: test soft relationships mirror hard relationships for ts properties
