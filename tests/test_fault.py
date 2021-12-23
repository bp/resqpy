import os

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

import resqpy.derived_model as rqdm
import resqpy.fault as rqf
import resqpy.grid as grr
import resqpy.lines as rql
import resqpy.model as rq
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp


# yapf: disable
@pytest.mark.parametrize('inc_list,tmult_dict,expected_mult',
   [(['fault_1.inc'], {}, {'fault_1': 1}),
    (['fault_1.inc'], {'fault_1': 2}, {'fault_1': 2}),
    (['fault_1.inc', 'fault_2.inc'], {'fault_1': 2}, {'fault_1': 2, 'fault_2': 2})])
# yapf: enable
def test_add_connection_set_and_tmults(example_model_with_properties, test_data_path, inc_list, tmult_dict,
                                       expected_mult):
    model = example_model_with_properties

    inc_list = [os.path.join(test_data_path, inc) for inc in inc_list]

    gcs_uuid = rqf.add_connection_set_and_tmults(model, inc_list, tmult_dict)

    assert gcs_uuid is not None, 'Grid connection set not generated'

    reload_model = rq.Model(epc_file = model.epc_file)

    faults = reload_model.parts_list_of_type('obj_FaultInterpretation')
    assert len(faults) == len(expected_mult.keys()),  \
        f'Expected a {len(expected_mult.keys())} faults, found {len(faults)}'
    for fault in faults:
        metadata = rqet.load_metadata_from_xml(reload_model.root_for_part(fault))
        title = reload_model.citation_title_for_part(fault)
        expected_str = str(float(expected_mult[title]))
        assert metadata["Transmissibility multiplier"] == expected_str,  \
            f'Expected mult for fault {title} to be {expected_str}, found {metadata["Transmissibility multiplier"]}'

    # check that a transmissibility multiplier property has been created
    gcs = rqf.GridConnectionSet(reload_model, uuid = gcs_uuid, find_properties = True)
    assert gcs is not None
    pc = gcs.property_collection
    assert pc is not None and pc.number_of_parts() > 0
    part = pc.singleton(property_kind = 'transmissibility multiplier')
    assert part is not None
    # check property values are in expected set
    a = pc.cached_part_array_ref(part)
    assert a is not None and a.ndim == 1
    expect = [x for x in expected_mult.values()]
    assert all([v in expect for v in a])
    # see if a local property kind has been set up correctly
    pku = pc.local_property_kind_uuid(part)
    assert pku is not None
    pk = rqp.PropertyKind(reload_model, uuid = pku)
    assert pk is not None
    assert pk.title == 'transmissibility multiplier'


def test_gcs_property_inheritance(tmp_path):

    epc = os.path.join(tmp_path, 'gcs_prop_inherit.epc')

    model = rq.Model(epc, new_epc = True, create_basics = True, create_hdf5_ext = True)

    # create a grid
    g = grr.RegularGrid(model, (5, 3, 3), dxyz = (10.0, 10.0, 1.0))
    g.write_hdf5()
    g.create_xml(title = 'unsplit grid')

    # define an L shaped (in plan view) fault
    j_faces = np.zeros((g.nk, g.nj - 1, g.ni), dtype = bool)
    j_faces[:, 0, 1:] = True
    i_faces = np.zeros((g.nk, g.nj, g.ni - 1), dtype = bool)
    i_faces[:, 1:, 0] = True
    gcs = rqf.GridConnectionSet(model,
                                grid = g,
                                j_faces = j_faces,
                                i_faces = i_faces,
                                feature_name = 'L fault',
                                create_organizing_objects_where_needed = True,
                                create_transmissibility_multiplier_property = False)
    # check that connection set has the right number of cell face pairs
    assert gcs.count == g.nk * ((g.nj - 1) + (g.ni - 1))

    # create a transmissibility multiplier property
    tm = np.arange(gcs.count).astype(float)
    if gcs.property_collection is None:
        gcs.property_collection = rqp.PropertyCollection()
        gcs.property_collection.set_support(support = gcs)
    pc = gcs.property_collection
    pc.add_cached_array_to_imported_list(
        tm,
        'unit test',
        'TMULT',
        uom = 'Euc',  # actually a ratio of transmissibilities
        property_kind = 'transmissibility multiplier',
        local_property_kind_uuid = None,
        realization = None,
        indexable_element = 'faces')
    # write gcs which should also write property collection and create a local property kind
    gcs.write_hdf5()
    gcs.create_xml(write_new_properties = True)

    # check that a local property kind has materialised
    pk_uuid = model.uuid(obj_type = 'PropertyKind', title = 'transmissibility multiplier')
    assert pk_uuid is not None

    # create a derived grid connection set using a layer range
    thin_gcs, thin_indices = gcs.filtered_by_layer_range(min_k0 = 1, max_k0 = 3, return_indices = True)
    assert thin_gcs is not None and thin_indices is not None
    assert thin_gcs.count == 3 * ((g.nj - 1) + (g.ni - 1))
    # inherit the transmissibility multiplier property
    thin_gcs.inherit_properties_for_selected_indices(gcs, thin_indices)
    thin_gcs.write_hdf5()
    thin_gcs.create_xml()  # by default will include write of new properties

    # check that the inheritance has worked
    assert thin_gcs.property_collection is not None and thin_gcs.property_collection.number_of_parts() > 0
    thin_pc = thin_gcs.property_collection
    tm_part = thin_pc.singleton(property_kind = 'transmissibility multiplier')
    assert tm_part is not None
    thin_tm = thin_pc.cached_part_array_ref(tm_part)
    assert thin_tm is not None and thin_tm.ndim == 1
    assert thin_tm.size == thin_gcs.count
    assert_array_almost_equal(thin_tm, tm[thin_indices])

    # check that get_combined...() method can execute using property collection
    b_a, i_a, f_a = gcs.get_combined_fault_mask_index_value_arrays(min_k = 1,
                                                                   max_k = 3,
                                                                   property_name = 'Transmissibility multiplier',
                                                                   ref_k = 2)
    assert b_a is not None and i_a is not None and f_a is not None
    # check that transmissibility multiplier values have been sampled correctly from property array
    assert f_a.shape == (g.nj, g.ni, 2, 2)
    assert np.count_nonzero(np.isnan(f_a)) == 4 * g.nj * g.ni - 2 * ((g.nj - 1) + (g.ni - 1))
    assert np.nanmax(f_a) > np.nanmin(f_a)
    restore = np.seterr(all = 'ignore')
    assert np.all(np.logical_or(np.isnan(f_a), f_a >= np.nanmin(thin_tm)))
    assert np.all(np.logical_or(np.isnan(f_a), f_a <= np.nanmax(thin_tm)))
    np.seterr(**restore)


def test_pinchout_and_k_gap_gcs(tmp_path):

    epc = os.path.join(tmp_path, 'gcs_pinchout_k_gap.epc')
    model = rq.new_model(epc)

    # create a grid
    g = grr.RegularGrid(model, (5, 5, 5), dxyz = (100.0, 100.0, 10.0), as_irregular_grid = True)
    # patch points to generate a pinchout
    p = g.points_cached
    assert p.shape == (6, 6, 6, 3)
    p[2, :3, :3] = p[1, :3, :3]
    # convert one layer to a K gap with pinchout
    p[4, 3:, 3:] = p[3, 3:, 3:]
    g.nk -= 1
    g.extent_kji = np.array((g.nk, g.nj, g.ni), dtype = int)
    g.k_gaps = 1
    g.k_gap_after_array = np.zeros(g.nk - 1, dtype = bool)
    g.k_gap_after_array[2] = True
    g._set_k_raw_index_array()
    g.write_hdf5()
    g.create_xml(title = 'pinchout k gap grid')
    model.store_epc()

    # reload the grid
    model = rq.Model(epc)
    grid = model.grid()
    assert grid is not None
    assert grid.k_gaps == 1
    assert tuple(grid.extent_kji) == (4, 5, 5)

    # create a pinchout connection set
    po_gcs = rqf.pinchout_connection_set(grid)
    assert po_gcs is not None
    po_gcs.write_hdf5()
    po_gcs.create_xml()
    po_uuid = po_gcs.uuid

    # create a K gap connection set
    kg_gcs = rqf.k_gap_connection_set(grid)
    assert kg_gcs is not None
    kg_gcs.write_hdf5()
    kg_gcs.create_xml()
    kg_uuid = kg_gcs.uuid

    model.store_epc()

    # re-open the model and load the connection sets
    model = rq.Model(epc)
    po_gcs = rqf.GridConnectionSet(model, uuid = po_uuid)
    assert po_gcs is not None
    po_gcs.cache_arrays()
    kg_gcs = rqf.GridConnectionSet(model, uuid = kg_uuid)
    assert kg_gcs is not None
    kg_gcs.cache_arrays()

    # check face pairs in the pinchout connection set
    assert po_gcs.count == 4
    assert po_gcs.cell_index_pairs.shape == (4, 2)
    assert po_gcs.face_index_pairs.shape == (4, 2)
    assert np.all(po_gcs.cell_index_pairs[:, 0] != po_gcs.cell_index_pairs[:, 1])
    assert np.all(po_gcs.face_index_pairs[:, 0] != po_gcs.cell_index_pairs[:, 1])
    assert np.all(np.logical_or(po_gcs.face_index_pairs == 0, po_gcs.face_index_pairs == 1))
    for cell in po_gcs.cell_index_pairs.flatten():
        assert cell in [0, 1, 5, 6, 50, 51, 55, 56]
    assert np.all(np.abs(po_gcs.cell_index_pairs[:, 1] - po_gcs.cell_index_pairs[:, 0]) == 50)

    # check face pairs in K gap connection set
    assert kg_gcs.count == 4
    assert kg_gcs.cell_index_pairs.shape == (4, 2)
    assert kg_gcs.face_index_pairs.shape == (4, 2)
    assert np.all(kg_gcs.cell_index_pairs[:, 0] != kg_gcs.cell_index_pairs[:, 1])
    assert np.all(kg_gcs.face_index_pairs[:, 0] != kg_gcs.cell_index_pairs[:, 1])
    assert np.all(np.logical_or(kg_gcs.face_index_pairs == 0, kg_gcs.face_index_pairs == 1))
    for cell in kg_gcs.cell_index_pairs.flatten():
        assert cell in [74, 73, 69, 68, 99, 98, 94, 93]
    assert np.all(np.abs(kg_gcs.cell_index_pairs[:, 1] - kg_gcs.cell_index_pairs[:, 0]) == 25)

    # test compact indices method
    ci = po_gcs.compact_indices()
    assert ci.shape == (4, 2)
    for cf in ci.flatten():
        assert cf in [1, 7, 31, 37, 300, 306, 330, 336]
    assert np.all(np.abs(ci[:, 1] - ci[:, 0]) == 299)

    # test write simulator method
    files = ('fault_ff.dat', 'fault_tf.dat', 'fault_ft.dat')
    both_sides = (False, True, False)
    minus = (False, False, True)
    for filename, inc_both_sides, use_minus in zip(files, both_sides, minus):
        dat_path = os.path.join(tmp_path, filename)
        po_gcs.write_simulator(dat_path, include_both_sides = inc_both_sides, use_minus = use_minus)
        assert os.path.exists(dat_path)


def test_two_fault_gcs(tmp_path):

    epc = make_epc_with_gcs(tmp_path)

    # re-open the model and check the gcs
    model = rq.Model(epc)
    gcs_uuid = model.uuid(obj_type = 'GridConnectionSetRepresentation')
    assert gcs_uuid is not None
    gcs = rqf.GridConnectionSet(model, uuid = gcs_uuid)
    assert gcs is not None
    assert gcs.number_of_features() == 2
    feature_names = gcs.list_of_feature_names()
    assert len(feature_names) == 2
    assert 'F1' in feature_names and 'F2' in feature_names
    fault_names = gcs.list_of_fault_names()
    assert fault_names == feature_names
    for fi in (0, 1):
        assert gcs.feature_name_for_feature_index(fi) in ('F1', 'F2')
    assert gcs.feature_name_for_feature_index(0) != gcs.feature_name_for_feature_index(1)
    fi, f_uuid = gcs.feature_index_and_uuid_for_fault_name('F1')
    assert fi is not None and fi in (0, 1)
    assert f_uuid is not None
    assert gcs.fault_name_for_feature_index(fi) == 'F1'
    assert gcs.feature_index_for_cell_face((1, 1, 1), 0, 1) is None
    fi_a = gcs.feature_index_for_cell_face((1, 1, 1), 2, 1)
    assert fi_a in (0, 1)
    fi_b = gcs.feature_index_for_cell_face((1, 1, 1), 1, 1)
    assert fi_b in (0, 1)
    assert fi_a != fi_b
    gcs.rework_face_pairs()


def test_feature_inheritance(tmp_path):

    epc = make_epc_with_gcs(tmp_path)

    # introduce a split version of the grid
    model = rq.Model(epc)
    simple_gcs_uuid = model.uuid(obj_type = 'GridConnectionSetRepresentation')
    assert simple_gcs_uuid is not None
    crs_uuid = model.uuid(obj_type = 'LocalDepth3dCrs')
    assert crs_uuid is not None
    line_a = rql.Polyline(model,
                          set_bool = False,
                          set_crs = crs_uuid,
                          title = 'line a',
                          set_coord = np.array([[200.0, -50.0, 0.0], [200.0, 450.0, 0.0]]))
    line_b = rql.Polyline(model,
                          set_bool = False,
                          set_crs = crs_uuid,
                          title = 'line b',
                          set_coord = np.array([[-50.0, 200.0, 0.0], [350.0, 200.0, 0.0]]))
    fault_lines = rql.PolylineSet(model, polylines = [line_a, line_b], title = 'fault lines')
    assert fault_lines is not None
    fault_lines.write_hdf5()
    fault_lines.create_xml()
    model.store_epc()
    model.h5_release()

    i_grid = rqdm.add_faults(epc,
                             polylines = fault_lines,
                             source_grid = None,
                             inherit_properties = True,
                             new_grid_title = 'interim',
                             create_gcs = False)
    assert i_grid is not None

    # increase the throw on the faults
    rqdm.global_fault_throw_scaling(epc,
                                    source_grid = i_grid,
                                    scaling_factor = 10.0,
                                    cell_range = 1,
                                    inherit_properties = True,
                                    new_grid_title = 'faulted')

    # re-open the model and load the faulted grid
    model = rq.Model(epc)
    grid = model.grid(title = 'faulted')
    assert grid is not None

    # establish the original simple gcs (even though it relates to a different grid?)
    simple_gcs = rqf.GridConnectionSet(model, uuid = simple_gcs_uuid)

    # derive a grid connection set from fault juxtaposition, inheriting features from simple gcs
    juxta_gcs, tr = grid.fault_connection_set(compute_transmissibility = True,
                                              add_to_model = True,
                                              inherit_features_from = simple_gcs,
                                              title = 'juxtaposed')
    assert juxta_gcs is not None
    assert tr is not None
    assert tr.ndim == 1 and len(tr) == juxta_gcs.count

    # check that features have been inherited correctly
    assert juxta_gcs.number_of_features() == 2
    feature_names = juxta_gcs.list_of_feature_names()
    for fi in range(2):
        if 'F1' in feature_names[fi]:
            axis = 2
        else:
            assert 'F2' in feature_names[fi]
            axis = 1
        cfp_lists = juxta_gcs.list_of_cell_face_pairs_for_feature_index(fi)
        assert len(cfp_lists) == 2 and len(cfp_lists[0]) == len(cfp_lists[1]) and len(cfp_lists[1]) > 0
        for fip in cfp_lists[1]:
            assert fip.shape == (2, 2)
            assert np.all(fip[:, 0] == axis)
            assert np.all(fip[0, 1] == 1 - fip[1, 1])

    # check that transmissibility property has been created okay
    tr_uuid = model.uuid(obj_type = 'ContinuousProperty', related_uuid = juxta_gcs.uuid)
    assert tr_uuid is not None
    trp = rqp.Property(model, uuid = tr_uuid)
    assert trp is not None
    assert trp.is_continuous()
    assert trp.property_kind() == 'transmissibility'
    assert_array_almost_equal(tr, trp.array_ref())


def test_two_grid_gcs(tmp_path):

    epc = make_epc_with_abutting_grids(tmp_path)

    # re-open the model and establish the abutting grid connection set
    model = rq.Model(epc)
    gcs_uuid = model.uuid(obj_type = 'GridConnectionSetRepresentation')
    assert gcs_uuid is not None
    gcs = rqf.GridConnectionSet(model, uuid = gcs_uuid)
    assert gcs is not None
    gcs.cache_arrays()

    # check that attributes have been preserved
    assert gcs.number_of_grids() == 2
    assert len(gcs.grid_list) == 2
    assert not bu.matching_uuids(gcs.grid_list[0].uuid, gcs.grid_list[1].uuid)
    assert gcs.count == 6
    assert gcs.grid_index_pairs.shape == (6, 2)
    assert np.all(gcs.grid_index_pairs[:, 0] == 0)
    assert np.all(gcs.grid_index_pairs[:, 1] == 1)
    assert gcs.face_index_pairs.shape == (6, 2)
    assert np.all(gcs.face_index_pairs[:, 0] == gcs.face_index_map[1, 1])  # J+
    assert np.all(gcs.face_index_pairs[:, 1] == gcs.face_index_map[1, 0])  # J-
    assert tuple(gcs.face_index_inverse_map[gcs.face_index_pairs[0, 0]]) == (1, 1)
    assert tuple(gcs.face_index_inverse_map[gcs.face_index_pairs[0, 1]]) == (1, 0)
    assert gcs.cell_index_pairs.shape == (6, 2)
    assert np.all(gcs.cell_index_pairs >= 0)
    assert np.all(gcs.cell_index_pairs < 60)


def make_epc_with_gcs(tmp_path):

    epc = os.path.join(tmp_path, 'two_fault.epc')
    model = rq.new_model(epc)

    # create a grid
    g = grr.RegularGrid(model, (5, 4, 3), dxyz = (100.0, 100.0, 10.0))
    g.create_xml()

    # create an empty grid connection set
    gcs = rqf.GridConnectionSet(model, grid = g)

    # prepare two named faults as a dataframe
    df = pd.DataFrame(columns = ['name', 'face', 'i1', 'i2', 'j1', 'j2', 'k1', 'k2', 'mult'])
    df = df.append({
        'name': 'F1',
        'face': 'I+',
        'i1': 1,
        'i2': 1,
        'j1': 0,
        'j2': 3,
        'k1': 0,
        'k2': 4,
        'mult': 0.1
    },
                   ignore_index = True)
    df = df.append({
        'name': 'F2',
        'face': 'J-',
        'i1': 0,
        'i2': 2,
        'j1': 2,
        'j2': 2,
        'k1': 0,
        'k2': 4,
        'mult': 0.05
    },
                   ignore_index = True)

    # set grid connection set from dataframe
    gcs.set_pairs_from_faces_df(df,
                                create_organizing_objects_where_needed = True,
                                create_mult_prop = True,
                                fault_tmult_dict = None,
                                one_based_indexing = False)

    # save the grid connection set
    gcs.write_hdf5()
    gcs.create_xml(title = 'two fault gcs')
    model.store_epc()

    # add some basic grid properties
    porosity_uuid = rqdm.add_one_grid_property_array(epc,
                                                     np.full(g.extent_kji, 0.27),
                                                     property_kind = 'porosity',
                                                     title = 'porosity',
                                                     uom = 'm3/m3')
    assert porosity_uuid is not None
    perm_uuid = rqdm.add_one_grid_property_array(epc,
                                                 np.full(g.extent_kji, 152.0),
                                                 property_kind = 'rock permeability',
                                                 uom = 'mD',
                                                 facet_type = 'direction',
                                                 facet = 'IJK',
                                                 title = 'permeability')
    assert perm_uuid is not None

    return epc


def make_epc_with_abutting_grids(tmp_path):

    epc = os.path.join(tmp_path, 'abutting_grids.epc')
    model = rq.new_model(epc)

    # create a grid
    g0 = grr.RegularGrid(model, (5, 4, 3), dxyz = (100.0, 100.0, 10.0))
    g0.create_xml()

    g1 = grr.RegularGrid(model, (5, 4, 3), dxyz = (100.0, 100.0, 10.0), origin = (100.0, 400.0, 20.0))
    g1.create_xml()

    # create an empty grid connection set
    gcs = rqf.GridConnectionSet(model, title = 'abut')

    # populate the grid connection set at low level due to lack of multi-grid methods
    gcs.grid_list = [g0, g1]
    gcs.count = 6
    gcs.grid_index_pairs = np.zeros((6, 2), dtype = int)
    gcs.grid_index_pairs[:, 1] = 1
    gcs.face_index_pairs = np.empty((6, 2), dtype = int)
    gcs.face_index_pairs[:, 0] = gcs.face_index_map[1, 1]  # J+
    gcs.face_index_pairs[:, 1] = gcs.face_index_map[1, 0]  # J-
    gcs.cell_index_pairs = np.empty((6, 2), dtype = int)
    cell = 0
    for k in range(3):
        for i in range(2):
            gcs.cell_index_pairs[cell, 0] = g0.natural_cell_index((k + 2, 3, i + 1))
            gcs.cell_index_pairs[cell, 1] = g1.natural_cell_index((k, 0, i))
            cell += 1
    # leave optional feature list & indices as None

    # save the grid connection set
    gcs.write_hdf5()
    gcs.create_xml()
    model.store_epc()

    return epc
