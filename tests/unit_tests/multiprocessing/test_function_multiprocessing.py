import pytest
import os

import resqpy.model as rq
import resqpy.organize as rqo
import resqpy.multi_processing._multiprocessing as rqmp


def make_feature(name, index, parent_tmp_dir):
    epc = os.path.join(parent_tmp_dir, f'mp_test_{index}.epc')
    model = rq.new_model(epc)
    feat = rqo.OrganizationFeature(model, feature_name = f'{name} {index}', organization_kind = 'structural')
    feat.create_xml()
    model.store_epc()
    return (index, True, epc, [feat.uuid])


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
