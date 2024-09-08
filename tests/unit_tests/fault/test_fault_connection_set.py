import os

import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.crs as rqc
import resqpy.derived_model as rqdm
import resqpy.fault as rqf
import resqpy.grid as grr
import resqpy.lines as rql
import resqpy.model as rq
import resqpy.property as rqp
import resqpy.organize as rqo
import resqpy.olio.transmission as rqtr
import resqpy.grid_surface as rqgs
import resqpy.olio.uuid as bu


def test_fault_connection_set(tmp_path):

    gm = os.path.join(tmp_path, 'resqpy_test_fgcs.epc')

    model = rq.new_model(gm)
    crs = rqc.Crs(model)
    crs.create_xml()

    # unsplit grid

    g0 = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))
    g0.write_hdf5()
    g0.create_xml(title = 'G0 unsplit')
    g0_fcs, g0_fa = rqtr.fault_connection_set(g0)

    assert g0_fcs is None
    assert g0_fa is None

    # J face split with no juxtaposition
    throw = 2.0

    g1 = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    g1.grid_representation = 'IjkGrid'

    pu_pillar_count = (g1.nj + 1) * (g1.ni + 1)
    pu = g1.points_ref(masked = False).reshape(g1.nk + 1, pu_pillar_count, 3)
    p = np.zeros((g1.nk + 1, pu_pillar_count + g1.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, g1.ni + 1:2 * (g1.ni + 1), :]
    p[:, 2 * (g1.ni + 1):, 2] += throw
    g1.points_cached = p
    g1.has_split_coordinate_lines = True
    g1.split_pillars_count = g1.ni + 1
    g1.split_pillar_indices_cached = np.array([i for i in range(g1.ni + 1, 2 * (g1.ni + 1))], dtype = int)
    g1.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    g1.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    g1.write_hdf5()
    g1.create_xml(title = 'G1 J no juxtaposition')

    # model.store_epc()
    # model.h5_release()

    g1_fcs, g1_fa = rqtr.fault_connection_set(g1)

    assert g1_fcs is None
    assert g1_fa is None

    # I face split with no juxtaposition
    throw = 2.0

    g1 = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    g1.grid_representation = 'IjkGrid'

    pu_pillar_count = (g1.nj + 1) * (g1.ni + 1)
    pr = g1.points_ref(masked = False)
    pr[:, :, -1, 2] += throw
    pu = pr.reshape(g1.nk + 1, pu_pillar_count, 3)
    p = np.zeros((g1.nk + 1, pu_pillar_count + g1.nj + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pr[:, :, 1, :].reshape(g1.nk + 1, g1.nj + 1, 3)
    p[:, pu_pillar_count:, 2] += throw
    g1.points_cached = p
    g1.has_split_coordinate_lines = True
    g1.split_pillars_count = g1.ni + 1
    g1.split_pillar_indices_cached = np.array([i for i in range(1, (g1.nj + 1) * (g1.ni + 1), g1.nj + 1)], dtype = int)
    g1.cols_for_split_pillars = np.array((1, 1, 3, 3), dtype = int)
    g1.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    g1.write_hdf5()
    g1.create_xml(title = 'G1 I no juxtaposition')

    # model.store_epc()

    g1_fcs, g1_fa = rqtr.fault_connection_set(g1)

    assert g1_fcs is None
    assert g1_fa is None

    # J face split with full juxtaposition of kji0 (1, 0, *) with (0, 1, *)
    # pattern 4, 4 (or 3, 3)

    throw = 1.0

    g2 = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    g2.grid_representation = 'IjkGrid'

    pu_pillar_count = (g2.nj + 1) * (g2.ni + 1)
    pu = g2.points_ref(masked = False).reshape(g2.nk + 1, pu_pillar_count, 3)
    p = np.zeros((g2.nk + 1, pu_pillar_count + g2.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, g2.ni + 1:2 * (g2.ni + 1), :]
    p[:, 2 * (g2.ni + 1):, 2] += throw
    g2.points_cached = p
    g2.has_split_coordinate_lines = True
    g2.split_pillars_count = g2.ni + 1
    g2.split_pillar_indices_cached = np.array([i for i in range(g2.ni + 1, 2 * (g2.ni + 1))], dtype = int)
    g2.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    g2.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    g2.write_hdf5()
    g2.create_xml(title = 'G2 J full juxtaposition of kji0 (1, 0, *) with (0, 1, *)')

    # model.store_epc()

    g2_fcs, g2_fa = rqtr.fault_connection_set(g2)
    g2_fcs_again, _ = rqtr.fault_connection_set(g2)

    assert g2_fcs is not None
    assert g2_fa is not None
    assert g2_fcs_again is not None

    # show_fa(g2, g2_fcs, g2_fa)

    assert g2_fcs.count == 2
    assert np.all(np.isclose(g2_fa, 1.0, atol = 0.01))

    # check grid faces property array generation
    g2_fcs.write_hdf5()
    g2_fcs.create_xml()
    g2_fcs_again.write_hdf5()
    g2_fcs_again.create_xml()
    g2_fcs_pc = g2_fcs.extract_property_collection()
    g2_fcs_pc.add_cached_array_to_imported_list(np.array([True, True], dtype = bool),
                                                'unit test active',
                                                'ACTIVE',
                                                property_kind = 'active',
                                                discrete = True,
                                                indexable_element = 'faces')
    g2_fcs_pc.write_hdf5_for_imported_list()
    g2_fcs_pc.create_xml_for_imported_list_and_add_parts_to_model()
    gcs_p_a = np.array((37, 51), dtype = int)
    p = rqp.Property.from_array(model,
                                gcs_p_a,
                                'test',
                                'juxtapose',
                                g2_fcs.uuid,
                                property_kind = 'discrete test pk',
                                indexable_element = 'faces',
                                discrete = True,
                                null_value = -1)
    gcs_trm_a = np.array((0.1, 0.5), dtype = float)
    gcs_trm_b = np.array((0.2, 0.4), dtype = float)
    trm_p = rqp.Property.from_array(model,
                                    gcs_trm_a,
                                    'test',
                                    'trmult',
                                    g2_fcs.uuid,
                                    property_kind = 'transmissibility multiplier',
                                    indexable_element = 'faces',
                                    discrete = False,
                                    uom = 'Euc')
    trm_again_p = rqp.Property.from_array(model,
                                          gcs_trm_b,
                                          'test',
                                          'trmult',
                                          g2_fcs_again.uuid,
                                          property_kind = 'transmissibility multiplier',
                                          indexable_element = 'faces',
                                          discrete = False,
                                          uom = 'Euc')
    for lazy in [True, False]:
        pk, pj, pi = g2_fcs.grid_face_arrays(property_uuid = p.uuid,
                                             default_value = p.null_value(),
                                             feature_index = None,
                                             lazy = lazy)
        assert pk is not None and pj is not None and pi is not None
        assert pk.shape == (3, 2, 2) and pj.shape == (2, 3, 2) and pi.shape == (2, 2, 3)
        assert np.all(pk == -1)
        assert np.all(pi == -1)
        assert np.count_nonzero(pj > 0) == 2 if lazy else 4
        assert tuple(np.unique(pj)) == (-1, 37, 51)
    for pair, gcs_uuid_list, tr_uuid_list in [(False, [g2_fcs.uuid], [trm_p.uuid]),
                                              (True, [g2_fcs.uuid, g2_fcs_again.uuid], [trm_p.uuid, trm_again_p.uuid])]:
        for merge_mode in ['minimum', 'maximum', 'multiply']:
            for sided in ([False] if merge_mode == 'multiply' else [False, True]):
                c_trm_k,  c_trm_j,  c_trm_i =  \
                    rqf.combined_tr_mult_from_gcs_mults(model,
                                                        tr_uuid_list,
                                                        merge_mode = merge_mode,
                                                        sided = sided,
                                                        fill_value = 1.0)
                assert all([a is not None for a in (c_trm_k, c_trm_j, c_trm_i)])
                assert c_trm_k.shape == (3, 2, 2) and c_trm_j.shape == (2, 3, 2) and c_trm_i.shape == (2, 2, 3)
                assert_array_almost_equal(c_trm_k, 1.0)
                assert_array_almost_equal(c_trm_i, 1.0)
                assert np.count_nonzero(c_trm_j < 0.9) == 2 if lazy else 4
                assert not np.any(np.isnan(c_trm_j))
                unique_c_trm_j = np.unique(c_trm_j)
                assert len(unique_c_trm_j) == 3
                if merge_mode == 'minimum':
                    assert np.isclose(np.min(c_trm_j), 0.1)
                    assert np.isclose(unique_c_trm_j[1], 0.4 if pair else 0.5)
                elif merge_mode == 'maximum':
                    assert np.isclose(np.min(c_trm_j), 0.2 if pair else 0.1)
                    assert np.isclose(unique_c_trm_j[1], 0.5)
                else:  # multiply
                    assert np.isclose(np.min(c_trm_j), 0.02 if pair else 0.1)
                    assert np.isclose(unique_c_trm_j[1], 0.2 if pair else 0.5)
                assert np.isclose(unique_c_trm_j[2], 1.0)
                g_trm_k_uuid,  g_trm_j_uuid,  g_trm_i_uuid =  \
                    g2.combined_tr_mult_properties_from_gcs_mults(gcs_uuid_list = gcs_uuid_list,
                                                                  merge_mode = merge_mode,
                                                                  sided = sided,
                                                                  fill_value = 1.0,
                                                                  composite_property = False)
                assert g_trm_k_uuid is not None and g_trm_j_uuid is not None and g_trm_i_uuid is not None
                g_trm_k = rqp.Property(model, uuid = g_trm_k_uuid).array_ref()
                g_trm_j = rqp.Property(model, uuid = g_trm_j_uuid).array_ref()
                g_trm_i = rqp.Property(model, uuid = g_trm_i_uuid).array_ref()
                assert g_trm_k is not None and g_trm_j is not None and g_trm_i is not None
                assert np.all(g_trm_k == c_trm_k)
                assert np.all(g_trm_j == c_trm_j)
                assert np.all(g_trm_i == c_trm_i)
                g_trm_list = g2.combined_tr_mult_properties_from_gcs_mults(gcs_uuid_list = gcs_uuid_list,
                                                                           merge_mode = merge_mode,
                                                                           sided = sided,
                                                                           fill_value = 1.0,
                                                                           composite_property = True)
                assert len(g_trm_list) == 1
                g_trm = rqp.Property(model, uuid = g_trm_list[0]).array_ref()
                assert g_trm.ndim == 1
                assert g_trm.size == g_trm_k.size + g_trm_j.size + g_trm_i.size
                assert np.all(g_trm[:g_trm_k.size] == g_trm_k.flat)
                assert np.all(g_trm[g_trm_k.size:g_trm_k.size + g_trm_j.size] == g_trm_j.flat)
                assert np.all(g_trm[g_trm_k.size + g_trm_j.size:] == g_trm_i.flat)

    # I face split with full juxtaposition of kji0 (1, *, 0) with (0, *, 1)
    # pattern 4, 4 (or 3, 3) diagram 1

    throw = 1.0

    g1 = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    g1.grid_representation = 'IjkGrid'

    pu_pillar_count = (g1.nj + 1) * (g1.ni + 1)
    pr = g1.points_ref(masked = False)
    pr[:, :, -1, 2] += throw
    pu = pr.reshape(g1.nk + 1, pu_pillar_count, 3)
    p = np.zeros((g1.nk + 1, pu_pillar_count + g1.nj + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pr[:, :, 1, :].reshape(g1.nk + 1, g1.nj + 1, 3)
    p[:, pu_pillar_count:, 2] += throw
    g1.points_cached = p
    g1.has_split_coordinate_lines = True
    g1.split_pillars_count = g1.ni + 1
    g1.split_pillar_indices_cached = np.array([i for i in range(1, (g1.nj + 1) * (g1.ni + 1), g1.nj + 1)], dtype = int)
    g1.cols_for_split_pillars = np.array((1, 1, 3, 3), dtype = int)
    g1.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    g1.write_hdf5()
    g1.create_xml(title = 'G2 I full juxtaposition of kji0 (1, *, 0) with (0, *, 1)')

    # model.store_epc()

    g1_fcs, g1_fa = rqtr.fault_connection_set(g1)

    assert g1_fcs is not None
    assert g1_fa is not None

    # show_fa(g1, g1_fcs, g1_fa)

    assert g1_fcs.count == 2
    assert np.all(np.isclose(g1_fa, 1.0, atol = 0.01))

    # J face split with half juxtaposition of kji0 (*, 0, *) with (*, 1, *); and (1, 0, *) with (0, 1, *)
    # pattern 5, 5 (or 2, 2) diagram 2

    throw = 0.5

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]
    p[:, 2 * (grid.ni + 1):, 2] += throw
    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G3 J half juxtaposition of kji0 (*, 0, *) with (*, 1, *); and (1, 0, *) with (0, 1, *)')

    # model.store_epc()

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 6
    assert np.all(np.isclose(grid_fa, 0.5, atol = 0.01))

    # I face split with half juxtaposition of kji0 (*, *, 0) with (*, *, 1); and (1, *, 0) with (0, *, 1)
    # pattern 5, 5 (or 2, 2) diagram 2

    throw = 0.5

    g1 = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    g1.grid_representation = 'IjkGrid'

    pu_pillar_count = (g1.nj + 1) * (g1.ni + 1)
    pr = g1.points_ref(masked = False)
    pr[:, :, -1, 2] += throw
    pu = pr.reshape(g1.nk + 1, pu_pillar_count, 3)
    p = np.zeros((g1.nk + 1, pu_pillar_count + g1.nj + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pr[:, :, 1, :].reshape(g1.nk + 1, g1.nj + 1, 3)
    p[:, pu_pillar_count:, 2] += throw
    g1.points_cached = p
    g1.has_split_coordinate_lines = True
    g1.split_pillars_count = g1.ni + 1
    g1.split_pillar_indices_cached = np.array([i for i in range(1, (g1.nj + 1) * (g1.ni + 1), g1.nj + 1)], dtype = int)
    g1.cols_for_split_pillars = np.array((1, 1, 3, 3), dtype = int)
    g1.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    g1.write_hdf5()
    g1.create_xml(title = 'G3 I half juxtaposition of kji0 (*, *, 0) with (*, *, 1); and (1, *, 0) with (0, *, 1)')

    # model.store_epc()

    g1_fcs, g1_fa = rqtr.fault_connection_set(g1)

    assert g1_fcs is not None
    assert g1_fa is not None

    # show_fa(g1, g1_fcs, g1_fa)

    assert g1_fcs.count == 6
    assert np.all(np.isclose(g1_fa, 0.5, atol = 0.01))

    # J face split with 0.25 juxtaposition of kji0 (*, 0, *) with (*, 1, *); and 0.75 of (1, 0, *) with (0, 1, *)
    # pattern 5, 5 (or 2, 2) diagram 2

    throw = 0.75

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]
    p[:, 2 * (grid.ni + 1):, 2] += throw
    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(
        title = 'G4 J 0.25 juxtaposition of kji0 (*, 0, *) with (*, 1, *); and 0.75 of (1, 0, *) with (0, 1, *)')

    # model.store_epc()

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 6
    # following assertion assumes lists are in certain order, which is not a functional requirement
    assert np.all(
        np.isclose(grid_fa,
                   np.array([[0.25, 0.25], [0.75, 0.75], [0.25, 0.25], [0.25, 0.25], [0.75, 0.75], [0.25, 0.25]]),
                   atol = 0.01))

    # I face split with 0.25 juxtaposition of kji0 (*, *, 0) with (*, *, 1); and 0.75 of (1, *, 0) with (0, *, 1)
    # pattern 5, 5 (or 2, 2) diagram 2

    throw = 0.75

    g1 = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    g1.grid_representation = 'IjkGrid'

    pu_pillar_count = (g1.nj + 1) * (g1.ni + 1)
    pr = g1.points_ref(masked = False)
    pr[:, :, -1, 2] += throw
    pu = pr.reshape(g1.nk + 1, pu_pillar_count, 3)
    p = np.zeros((g1.nk + 1, pu_pillar_count + g1.nj + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pr[:, :, 1, :].reshape(g1.nk + 1, g1.nj + 1, 3)
    p[:, pu_pillar_count:, 2] += throw
    g1.points_cached = p
    g1.has_split_coordinate_lines = True
    g1.split_pillars_count = g1.ni + 1
    g1.split_pillar_indices_cached = np.array([i for i in range(1, (g1.nj + 1) * (g1.ni + 1), g1.nj + 1)], dtype = int)
    g1.cols_for_split_pillars = np.array((1, 1, 3, 3), dtype = int)
    g1.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    g1.write_hdf5()
    g1.create_xml(
        title = 'G4 I 0.25 juxtaposition of kji0 (*, *, 0) with (*, *, 1); and 0.75 of (1, *, 0) with (0, *, 1)')

    g1_fcs, g1_fa = rqtr.fault_connection_set(g1)

    assert g1_fcs is not None
    assert g1_fa is not None

    # show_fa(g1, g1_fcs, g1_fa)

    assert g1_fcs.count == 6
    # following assertion assumes lists are in certain order, which is not a functional requirement
    assert np.all(
        np.isclose(g1_fa,
                   np.array([[0.25, 0.25], [0.75, 0.75], [0.25, 0.25], [0.25, 0.25], [0.75, 0.75], [0.25, 0.25]]),
                   atol = 0.01))

    # J face split with full full (1, 0, 0) with (0, 1, 0); and 0.5 of (*, 0, 1) with (*, 1, 1) and layer crossover
    # diagrams 4 & 2

    throw = 1.0

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]
    p[:, 2 * (grid.ni + 1):-1, 2] += throw
    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(
        title = 'G5 J full (1, 0, 0) with (0, 1, 0); and 0.5 of (*, 0, 1) with (*, 1, 1) and layer crossover')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 4
    # following assertion assumes lists are in certain order, which is not a functional requirement
    assert np.all(np.isclose(grid_fa, np.array([[1., 1.], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]), atol = 0.01))

    # I face split with full (1, 0, 0) with (0, 0, 1); and 0.5 of (*, 1, 0) with (*, 1, 1) and layer crossover

    throw = 1.0

    g1 = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    g1.grid_representation = 'IjkGrid'

    pu_pillar_count = (g1.nj + 1) * (g1.ni + 1)
    pr = g1.points_ref(masked = False)
    pr[:, :, -1, 2] += throw
    pu = pr.reshape(g1.nk + 1, pu_pillar_count, 3)
    p = np.zeros((g1.nk + 1, pu_pillar_count + g1.nj + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pr[:, :, 1, :].reshape(g1.nk + 1, g1.nj + 1, 3)
    p[:, pu_pillar_count:-1, 2] += throw
    g1.points_cached = p
    g1.has_split_coordinate_lines = True
    g1.split_pillars_count = g1.ni + 1
    g1.split_pillar_indices_cached = np.array([i for i in range(1, (g1.nj + 1) * (g1.ni + 1), g1.nj + 1)], dtype = int)
    g1.cols_for_split_pillars = np.array((1, 1, 3, 3), dtype = int)
    g1.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    g1.write_hdf5()
    g1.create_xml(title = 'G5 I full (1, 0, 0) with (0, 0, 1); and 0.5 of (*, 1, 0) with (*, 1, 1) and layer crossover')

    g1_fcs, g1_fa = rqtr.fault_connection_set(g1)

    assert g1_fcs is not None
    assert g1_fa is not None

    # show_fa(g1, g1_fcs, g1_fa)

    assert g1_fcs.count == 4
    # following assertion assumes lists are in certain order, which is not a functional requirement
    assert np.all(np.isclose(g1_fa, np.array([[1., 1.], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]), atol = 0.01))

    grid_fa

    # J face split diagram 4

    throw = 0.5

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):-1, 2] += throw

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G6 J diagram 4')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 6
    # following assertion assumes lists are in certain order, which is not a functional requirement
    assert np.all(
        np.isclose(grid_fa,
                   np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.75, 0.75], [0.25, 0.25], [0.75, 0.75]]),
                   atol = 0.01))

    # J face split
    # diagram 5

    throw = 0.5

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[1:, 2 * (grid.ni + 1), 2] -= 0.9
    p[1:, 3 * (grid.ni + 1), 2] -= 1.0

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G7 diagram 5')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 7
    assert np.all(
        np.isclose(grid_fa,
                   np.array([[0.375, 0.75], [0.125, 0.125], [0.125, 0.25], [0.75, 0.75], [0.5, 0.5], [0.5, 0.5],
                             [0.5, 0.5]]),
                   atol = 0.01))

    # bl.set_log_level('info')

    # J face split
    # diagram 5m

    throw = 0.5

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[1:, 3 * (grid.ni + 1) - 1, 2] -= 0.9
    p[1:, 4 * (grid.ni + 1) - 1, 2] -= 1.0

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G7m diagram 5m')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 7
    assert np.all(
        np.isclose(grid_fa,
                   np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.375, 0.75], [0.125, 0.125], [0.125, 0.25],
                             [0.75, 0.75]]),
                   atol = 0.01))

    # bl.set_log_level('info')

    # J face split
    # deeper layer half thickness of top layer

    throw = 0.5

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G8 lower layer half thickness; throw 0.5')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 4
    assert np.all(np.isclose(grid_fa, np.array([[0.5, 0.5], [1.0, 0.5], [0.5, 0.5], [1.0, 0.5]]), atol = 0.01))

    # J face split
    # deeper layer half thickness of top layer

    throw = 0.75

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G9 lower layer half thickness; throw 0.75')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 4
    assert np.all(np.isclose(grid_fa, np.array([[0.25, 0.25], [1.0, 0.5], [0.25, 0.25], [1.0, 0.5]]), atol = 0.01))

    # J face split
    # deeper layer half thickness of top layer

    throw = 0.25

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G10 lower layer half thickness; throw 0.25')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 6
    assert np.all(
        np.isclose(grid_fa,
                   np.array([[0.75, 0.75], [0.5, 0.25], [0.5, 0.5], [0.75, 0.75], [0.5, 0.25], [0.5, 0.5]]),
                   atol = 0.01))

    # J face split
    # deeper layer half thickness of top layer

    throw = 0.75

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[:, 2 * (grid.ni + 1), 2] -= 1.0
    p[:, 3 * (grid.ni + 1), 2] -= 1.0
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G11 lower layer half thickness; throw 0.75 except for -0.25 on one wing')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 6
    assert np.all(
        np.isclose(
            grid_fa,
            np.array([
                [11.0 / 16, 11.0 / 16],  # 0.6875, 0.6875
                [1.0 / 32, 1.0 / 16],  # 0.03125, 0.0625
                [0.5, 0.25],
                [0.4375, 0.4375],
                [0.25, 0.25],
                [1.0, 0.5]
            ]),
            atol = 0.01))

    # J face split
    # deeper layer half thickness of top layer

    throw = 0.75

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[:, 3 * (grid.ni + 1) - 1, 2] -= 1.0
    p[:, 4 * (grid.ni + 1) - 1, 2] -= 1.0
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G12 lower layer half thickness; throw 0.75 except for -0.25 on one wing')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 6
    assert np.all(
        np.isclose(
            grid_fa,
            np.array([
                [0.25, 0.25],
                [1.0, 0.5],
                [11.0 / 16, 11.0 / 16],  # 0.6875, 0.6875
                [1.0 / 32, 1.0 / 16],  # 0.03125, 0.0625
                [0.5, 0.25],
                [0.4375, 0.4375]
            ]),
            atol = 0.01))

    # J face split
    # deeper layer half thickness of top layer

    throw = 0.75

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[:, 3 * (grid.ni + 1) - 1, 2] -= 1.5
    p[:, 4 * (grid.ni + 1) - 1, 2] -= 1.5
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G13 lower layer half thickness; throw 0.75 except for -0.25 on one wing')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 6
    assert np.all(
        np.isclose(
            grid_fa,
            np.array([
                [0.25, 0.25],
                [1.0, 0.5],
                [5.0 / 8, 5.0 / 8],  # 0.625, 0.625
                [1.0 / 6, 1.0 / 3],
                [1.0 / 3, 1.0 / 6],
                [1.0 / 3, 1.0 / 3]
            ]),
            atol = 0.01))

    # J face split
    # deeper layer half thickness of top layer

    throw = 0.75

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[0, 2 * (grid.ni + 1), 2] += 0.5
    p[0, 3 * (grid.ni + 1), 2] += 0.5
    p[1:, 2 * (grid.ni + 1), 2] -= 0.5
    p[1:, 3 * (grid.ni + 1), 2] -= 0.5
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G14 lower layer half thickness; throw 0.75 except for top layer pinching on one wing')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 5
    assert np.all(
        np.isclose(
            grid_fa,
            np.array([
                [1.0 / 16, 1.0 / 8],  # 0.0625, 0.125
                [0.75, 0.75],
                [0.125, 0.125],
                [0.25, 0.25],
                [1.0, 0.5]
            ]),
            atol = 0.01))

    # J face split
    # deeper layer half thickness of top layer

    throw = -0.25

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[0, 2 * (grid.ni + 1), 2] += 0.5
    p[0, 3 * (grid.ni + 1), 2] += 0.5
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G15 lower layer half thickness; throw -0.25 except for top layer pinching on one wing')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 6
    assert np.all(
        np.isclose(grid_fa,
                   np.array([[11.0 / 16, 11.0 / 12], [0.25, 0.5], [0.5, 0.5], [0.75, 0.75], [0.25, 0.5], [0.5, 0.5]]),
                   atol = 0.01))

    # J face split
    # deeper layer half thickness of top layer

    throw = -0.75

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[:, 2 * (grid.ni + 1), 2] -= 1.0
    p[:, 3 * (grid.ni + 1), 2] -= 1.0
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G16 lower layer half thickness; throw -0.75 except for -1.75 on one wing')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 4
    assert np.all(
        np.isclose(grid_fa, np.array([[1.0 / 32, 1.0 / 32], [0.25, 0.5], [0.25, 0.25], [0.5, 1.0]]), atol = 0.01))

    # J face split
    # deeper layer half thickness of top layer

    throw = +0.75

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[:, 2 * (grid.ni + 1), 2] += 1.0
    p[:, 3 * (grid.ni + 1), 2] += 1.0
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G17 lower layer half thickness; throw +0.75 except for +1.75 on one wing')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 4
    assert np.all(
        np.isclose(grid_fa, np.array([[1.0 / 32, 1.0 / 32], [0.5, 0.25], [0.25, 0.25], [1.0, 0.5]]), atol = 0.01))

    # J face split
    # deeper layer half thickness of top layer

    throw = +0.75

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[:, 3 * (grid.ni + 1) - 1, 2] += 1.0
    p[:, 4 * (grid.ni + 1) - 1, 2] += 1.0
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G18 lower layer half thickness; throw +0.75 except for +1.75 on one wing')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 4
    assert np.all(
        np.isclose(grid_fa, np.array([[0.25, 0.25], [1.0, 0.5], [1.0 / 32, 1.0 / 32], [0.5, 0.25]]), atol = 0.01))

    # J face split
    # deeper layer half thickness of top layer

    throw = +0.25

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[0, 2 * (grid.ni + 1), 2] -= 0.5
    p[0, 3 * (grid.ni + 1), 2] -= 0.5
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G19 lower layer half thickness; throw +0.25 except for top edge -0.25 on one wing')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 6
    assert np.all(
        np.isclose(grid_fa,
                   np.array([[15.0 / 16, 0.75], [0.5, 0.2], [0.5, 0.5], [0.75, 0.75], [0.5, 0.25], [0.5, 0.5]]),
                   atol = 0.01))

    # J face split
    # deeper layer half thickness of top layer

    throw = +0.25

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[0, 3 * (grid.ni + 1) - 1, 2] -= 0.5
    p[0, 4 * (grid.ni + 1) - 1, 2] -= 0.5
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G20 lower layer half thickness; throw +0.25 except for top edge -0.25 on one wing')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 6
    assert np.all(
        np.isclose(grid_fa,
                   np.array([[0.75, 0.75], [0.5, 0.25], [0.5, 0.5], [15.0 / 16, 0.75], [0.5, 0.2], [0.5, 0.5]]),
                   atol = 0.01))

    # bl.set_log_level('info')

    # J face split
    # deeper layer half thickness of top layer

    throw = +0.25

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[1, 2 * (grid.ni + 1), 2] -= 0.5
    p[1, 3 * (grid.ni + 1), 2] -= 0.5
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G21 lower layer half thickness; throw +0.25 except for mid edge -0.25 on one wing')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 7
    assert np.all(
        np.isclose(grid_fa,
                   np.array([[11.0 / 16, 11.0 / 12], [1.0 / 16, 1.0 / 12], [1.0 / 8, 1.0 / 12], [7.0 / 8, 7.0 / 12],
                             [0.75, 0.75], [0.5, 0.25], [0.5, 0.5]]),
                   atol = 0.01))

    # bl.set_log_level('info')

    # bl.set_log_level('debug')
    # local_log_out = bl.log_fresh()
    # display(local_log_out)

    # J face split
    # deeper layer half thickness of top layer

    throw = +0.5

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[:2, 2 * (grid.ni + 1), 2] -= 0.5
    p[:2, 3 * (grid.ni + 1), 2] -= 0.5
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G22 lower layer half thickness; throw +0.5 except for top, mid edge on one wing')

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 5
    assert np.all(
        np.isclose(grid_fa, np.array([[0.75, 0.75], [0.5, 0.25], [0.5, 1.0 / 3], [0.5, 0.5], [1.0, 0.5]]), atol = 0.01))

    # bl.set_log_level('debug')
    # local_log_out = bl.log_fresh()
    # display(local_log_out)

    # J face split
    # deeper layer half thickness of top layer

    throw = +0.5

    grid = grr.RegularGrid(model, extent_kji = (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

    grid.grid_representation = 'IjkGrid'

    pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
    pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
    p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
    p[:, :pu_pillar_count, :] = pu
    p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

    p[:, 2 * (grid.ni + 1):, 2] += throw
    p[:2, 3 * (grid.ni + 1) - 1, 2] -= 0.5
    p[:2, 4 * (grid.ni + 1) - 1, 2] -= 0.5
    p[-1, :, 2] -= 0.5

    grid.points_cached = p
    grid.has_split_coordinate_lines = True
    grid.split_pillars_count = grid.ni + 1
    grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
    grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
    grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
    grid.write_hdf5()
    grid.create_xml(title = 'G23 lower layer half thickness; throw +0.5 except for top, mid edge on one wing')

    model.store_epc()
    model.h5_release()

    grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

    assert grid_fcs is not None
    assert grid_fa is not None

    # show_fa(grid, grid_fcs, grid_fa)

    assert grid_fcs.count == 5
    assert np.all(
        np.isclose(grid_fa, np.array([[0.5, 0.5], [1.0, 0.5], [0.75, 0.75], [0.5, 0.25], [0.5, 1.0 / 3]]), atol = 0.01))


def test_add_faults(tmp_path):

    def write_poly(filename, a, mode = 'w'):
        nines = 999.0
        with open(filename, mode = mode) as fp:
            for row in range(len(a)):
                fp.write(f'{a[row, 0]:8.3f} {a[row, 1]:8.3f} {a[row, 2]:8.3f}\n')
            fp.write(f'{nines:8.3f} {nines:8.3f} {nines:8.3f}\n')

    def make_poly(model, a, title, crs):
        return [rql.Polyline(model, is_closed = False, set_coord = a, set_crs = crs.uuid, title = title)]

    epc = os.path.join(tmp_path, 'tic_tac_toe.epc')

    for test_mode in ['file', 'polyline']:

        model = rq.new_model(epc)
        grid = grr.RegularGrid(model, extent_kji = (1, 3, 3), set_points_cached = True)
        grid.write_hdf5()
        grid.create_xml(write_geometry = True)
        crs = rqc.Crs(model, uuid = grid.crs_uuid)
        model.store_epc()
        # add a permeability array for the grid
        grid_pc = grid.extract_property_collection()
        perm = np.linspace(start = 20.0, stop = 200.0, num = grid.cell_count()).reshape(tuple(grid.extent_kji))
        rqdm.add_one_grid_property_array(epc,
                                         perm,
                                         property_kind = 'rock permeability',
                                         grid_uuid = grid.uuid,
                                         title = 'PERM',
                                         uom = 'mD')

        # single straight fault
        a = np.array([[-0.2, 2.0, -0.1], [3.2, 2.0, -0.1]])
        f = os.path.join(tmp_path, 'ttt_f1.dat')
        if test_mode == 'file':
            write_poly(f, a)
            lines_file_list = [f]
            polylines = None
        else:
            lines_file_list = None
            polylines = make_poly(model, a, 'ttt_f1', crs)
        g = rqdm.add_faults(epc,
                            source_grid = None,
                            polylines = polylines,
                            lines_file_list = lines_file_list,
                            inherit_properties = True,
                            new_grid_title = 'ttt_f1 straight')

        # single zig-zag fault
        a = np.array([[-0.2, 1.0, -0.1], [1.0, 1.0, -0.1], [1.0, 2.0, -0.1], [3.2, 2.0, -0.1]])
        f = os.path.join(tmp_path, 'ttt_f2.dat')
        if test_mode == 'file':
            write_poly(f, a)
            lines_file_list = [f]
            polylines = None
        else:
            lines_file_list = None
            polylines = make_poly(model, a, 'ttt_f2', crs)
        g = rqdm.add_faults(epc,
                            source_grid = None,
                            polylines = polylines,
                            lines_file_list = lines_file_list,
                            inherit_properties = True,
                            new_grid_title = 'ttt_f2 zig_zag')

        # single zig-zag-zig fault
        a = np.array([[-0.2, 1.0, -0.1], [1.0, 1.0, -0.1], [1.0, 2.0, -0.1], [2.0, 2.0, -0.1], [2.0, 1.0, -0.1],
                      [3.2, 1.0, -0.1]])
        f = os.path.join(tmp_path, 'ttt_f3.dat')
        if test_mode == 'file':
            write_poly(f, a)
            lines_file_list = [f]
            polylines = None
        else:
            lines_file_list = None
            polylines = make_poly(model, a, 'ttt_f3', crs)
        g = rqdm.add_faults(epc,
                            source_grid = None,
                            polylines = polylines,
                            lines_file_list = lines_file_list,
                            inherit_properties = True,
                            new_grid_title = 'ttt_f3 zig_zag_zig')

        # horst block
        a = np.array([[-0.2, 1.0, -0.1], [3.2, 1.0, -0.1]])
        b = np.array([[3.2, 2.0, -0.1], [-0.2, 2.0, -0.1]])
        fa = os.path.join(tmp_path, 'ttt_f4a.dat')
        fb = os.path.join(tmp_path, 'ttt_f4b.dat')
        if test_mode == 'file':
            write_poly(fa, a)
            write_poly(fb, b)
            lines_file_list = [fa, fb]
            polylines = None
        else:
            lines_file_list = None
            polylines = make_poly(model, a, 'ttt_f4a', crs) + make_poly(model, b, 'ttt_f4b', crs)
        g = rqdm.add_faults(epc,
                            source_grid = None,
                            polylines = polylines,
                            lines_file_list = lines_file_list,
                            inherit_properties = True,
                            new_grid_title = 'ttt_f4 horst')

        # asymmetrical horst block
        lr_throw_dict = {'ttt_f4a': (0.0, -0.3), 'ttt_f4b': (0.0, -0.6)}
        g = rqdm.add_faults(epc,
                            source_grid = None,
                            polylines = polylines,
                            lines_file_list = lines_file_list,
                            left_right_throw_dict = lr_throw_dict,
                            inherit_properties = True,
                            new_grid_title = 'ttt_f5 horst')
        assert g is not None

        # scaled version of asymmetrical horst block; with some testing of grid connection set properties
        model = rq.Model(epc)
        grid = model.grid(title = 'ttt_f5 horst')
        assert grid is not None
        gcs_uuids = model.uuids(obj_type = 'GridConnectionSetRepresentation', related_uuid = grid.uuid)
        assert gcs_uuids
        scaling_dict = {'ttt_f4a': 3.0, 'ttt_f4b': 1.7}
        for i, gcs_uuid in enumerate(gcs_uuids):
            gcs = rqf.GridConnectionSet(model, uuid = gcs_uuid)
            # scale the fault throw
            rqdm.fault_throw_scaling(epc,
                                     source_grid = grid,
                                     scaling_factor = None,
                                     connection_set = gcs,
                                     scaling_dict = scaling_dict,
                                     ref_k0 = 0,
                                     ref_k_faces = 'top',
                                     cell_range = 0,
                                     store_displacement = False,
                                     inherit_properties = True,
                                     inherit_realization = None,
                                     inherit_all_realizations = False,
                                     new_grid_title = f'ttt_f6 scaled {i+1}',
                                     new_epc_file = None)
            model = rq.Model(epc)
            grid = model.grid(title = f'ttt_f6 scaled {i+1}')
            assert grid is not None
        # generate a new grid connection set based on juxtaposition
        juxta_gcs, fa = rqtr.fault_connection_set(grid)
        assert len(fa) == juxta_gcs.count
        # add some (arbitrary) transmissibility multiplier data for each gcs
        trm = np.linspace(start = 0.0, stop = float(i + 1), num = juxta_gcs.count)
        gcs_pc = juxta_gcs.extract_property_collection()
        gcs_pc.add_cached_array_to_imported_list(trm,
                                                 'unit test',
                                                 'TMULT',
                                                 uom = 'Euc',
                                                 property_kind = 'transmissibility multiplier',
                                                 indexable_element = 'faces')
        juxta_gcs.write_hdf5_and_create_xml_for_new_properties()
        # calculate transmissibility across the connection set cell face pairs
        tr = juxta_gcs.tr_property_array(fa = fa, apply_multipliers = True)
        assert tr is not None
        assert tr.shape == (juxta_gcs.count,)
        model.store_epc()

        # two intersecting straight faults
        a = np.array([[-0.2, 2.0, -0.1], [3.2, 2.0, -0.1]])
        b = np.array([[1.0, -0.2, -0.1], [1.0, 3.2, -0.1]])
        f = os.path.join(tmp_path, 'ttt_f7.dat')
        write_poly(f, a)
        write_poly(f, b, mode = 'a')
        if test_mode == 'file':
            write_poly(f, a)
            write_poly(f, b, mode = 'a')
            lines_file_list = [f]
            polylines = None
        else:
            lines_file_list = None
            polylines = make_poly(model, a, 'ttt_f7_1', crs) + make_poly(model, b, 'ttt_f7_2', crs)
        g = rqdm.add_faults(epc,
                            source_grid = None,
                            polylines = polylines,
                            lines_file_list = lines_file_list,
                            inherit_properties = True,
                            new_grid_title = 'ttt_f7')

        # re-open and check a few things
        model = rq.Model(epc)
        assert len(model.titles(obj_type = 'IjkGridRepresentation')) == 8
        g1 = model.grid(title = 'ttt_f7')
        assert g1.split_pillars_count == 9
        cpm = g1.create_column_pillar_mapping()
        assert cpm.shape == (3, 3, 2, 2)
        extras = (cpm >= 16)
        assert np.count_nonzero(extras) == 11
        assert np.all(np.sort(np.unique(cpm)) == np.arange(25))


def test_gcs_face_surface_normal_vectors(example_model_with_properties):
    # Arrange
    surface_normal_vectors = np.array([[-0.70710678, 0.0, 0.70710678], [0.0, 0.0, 1.0]])
    face_surface_normal_vectors_expected = np.array([
        [0., 0., 1.],
        [0., 0., 1.],
        [-0.70710678, 0., 0.70710678],
        [-0.70710678, 0., 0.70710678],
        [-0.70710678, 0., 0.70710678],
        [-0.70710678, 0., 0.70710678],
    ])

    # Act
    gcs = rqf.GridConnectionSet(example_model_with_properties)
    face_surface_normal_vectors = gcs.face_surface_normal_vectors(np.array([1, 1, 0, 0, 0, 0]), surface_normal_vectors)

    # Assert
    assert_array_almost_equal(face_surface_normal_vectors, face_surface_normal_vectors_expected)


def test_feature_index(tmp_path):
    epc = os.path.join(tmp_path, 'gcs_with_features.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    grid = grr.RegularGrid(model, extent_kji = (4, 5, 6), crs_uuid = crs.uuid)
    grid.create_xml()
    gcs_0 = rqf.GridConnectionSet(model, find_properties = False, grid = grid, title = 'fault zero gcs')
    gcs_0.set_pairs_from_kelp(
        kelp_0 = [(2, 0), (2, 1), (2, 2)],  # J face columns
        kelp_1 = [],  # I face columns
        feature_name = 'fault zero',
        create_organizing_objects_where_needed = True)
    gcs_0.write_hdf5()
    gcs_0.create_xml()
    pc = gcs_0.extract_property_collection()
    a = np.arange(12, dtype = int) + 100
    pc.add_cached_array_to_imported_list(a,
                                         'test',
                                         'this is the property of a gcs',
                                         discrete = True,
                                         null_value = -1,
                                         property_kind = 'kiwi',
                                         indexable_element = 'faces')
    pc.write_hdf5_for_imported_list()
    uuids = pc.create_xml_for_imported_list_and_add_parts_to_model()
    assert len(uuids) == 1
    kiwi_uuids = uuids
    gcs_1 = rqf.GridConnectionSet(model, find_properties = False, grid = grid, title = 'fault one gcs')
    gcs_1.set_pairs_from_kelp(
        kelp_0 = [],  # J face columns
        kelp_1 = [(2, 2)],  # I face columns
        feature_name = 'fault one',
        create_organizing_objects_where_needed = True)
    gcs_1.write_hdf5()
    gcs_1.create_xml()
    pc = gcs_1.extract_property_collection()
    a = np.arange(4, dtype = int) + 200
    pc.add_cached_array_to_imported_list(a,
                                         'test',
                                         'this is the property of a gcs',
                                         discrete = True,
                                         null_value = -1,
                                         property_kind = 'kiwi',
                                         indexable_element = 'faces')
    pc.write_hdf5_for_imported_list()
    uuids = pc.create_xml_for_imported_list_and_add_parts_to_model()
    assert len(uuids) == 1
    kiwi_uuids += uuids
    gcs_2 = rqf.GridConnectionSet(model, find_properties = False, grid = grid, title = 'fault two gcs')
    gcs_2.set_pairs_from_kelp(
        kelp_0 = [(1, 3), (1, 4), (1, 5)],  # J face columns
        kelp_1 = [],  # I face columns
        feature_name = 'fault two',
        create_organizing_objects_where_needed = True)
    gcs_2.write_hdf5()
    gcs_2.create_xml()
    pc = gcs_2.extract_property_collection()
    a = np.arange(12, dtype = int) + 300
    pc.add_cached_array_to_imported_list(a,
                                         'test',
                                         'this is the property of a gcs',
                                         discrete = True,
                                         null_value = -1,
                                         property_kind = 'kiwi',
                                         indexable_element = 'faces')
    pc.write_hdf5_for_imported_list()
    uuids = pc.create_xml_for_imported_list_and_add_parts_to_model()
    assert len(uuids) == 1
    kiwi_uuids += uuids
    model.store_epc()
    # check that fault interpretations have been generated
    fi_titles = model.titles(obj_type = 'FaultInterpretation')
    assert len(fi_titles) == 3
    assert set(fi_titles) == set(['fault zero', 'fault one', 'fault two'])
    # check one of the single feature grid connexion sets
    fi_one_uuid = model.uuid(obj_type = 'FaultInterpretation', title = 'fault one')
    assert fi_one_uuid is not None
    gcs_one_uuid = model.uuid(obj_type = 'GridConnectionSetRepresentation', related_uuid = fi_one_uuid)
    assert gcs_one_uuid is not None
    assert bu.matching_uuids(gcs_one_uuid, gcs_1.uuid)
    gcs_reload = rqf.GridConnectionSet(model, uuid = gcs_one_uuid)
    assert gcs_reload is not None
    assert gcs_reload.number_of_grids() == 1
    assert gcs_reload.number_of_features() == 1
    assert gcs_reload.title == 'fault one gcs'
    assert gcs_reload.list_of_feature_names(strip = False) == ['fault one']
    # make a composite grid connexion set with 3 features
    gcs = rqf.GridConnectionSet.from_gcs_uuid_list(model,
                                                   source_model = model,
                                                   gcs_uuid_list = [gcs_0.uuid, gcs_1.uuid, gcs_2.uuid],
                                                   title = 'multi feature fault system',
                                                   gcs_property_uuid_list_of_lists = [kiwi_uuids])
    model.store_epc()
    # above class method includes writing hdf5 and creating xml
    # reopen the composite grid connexion set
    gcs = rqf.GridConnectionSet(model, uuid = gcs.uuid)
    assert gcs is not None
    gcs.cache_arrays()
    assert gcs.number_of_grids() == 1
    assert gcs.number_of_features() == 3
    assert gcs.count == 28
    # check that a property exists on the composite gcs
    pc = gcs.extract_property_collection()
    # a = np.concatenate((np.arange(12, dtype = int) + 100, np.arange(4, dtype = int) + 200, np.arange(12, dtype = int) + 300))
    # pc.add_cached_array_to_imported_list(a, 'test', 'this is the property of a gcs', discrete = True, null_value = -1,
    #                                      property_kind = 'kiwi', indexable_element = 'faces')
    # pc.write_hdf5_for_imported_list()
    # uuids = pc.create_xml_for_imported_list_and_add_parts_to_model()
    # assert len(uuids) == 1
    assert pc.number_of_parts() == 1
    kiwi_uuid = model.uuid(obj_type = 'DiscreteProperty', related_uuid = gcs.uuid)
    assert kiwi_uuid is not None
    # check features
    assert gcs.list_of_feature_names(strip = False) == ['fault zero', 'fault one', 'fault two']
    assert gcs.list_of_fault_names(strip = False) == ['fault zero', 'fault one', 'fault two']
    index, fi_uuid = gcs.feature_index_and_uuid_for_fault_name('fault two')
    assert index == 2
    assert fi_uuid is not None
    fi = rqo.FaultInterpretation(model, uuid = fi_uuid)
    assert fi is not None
    assert fi.title == 'fault two'
    assert gcs.feature_name_for_feature_index(0, strip = False) == 'fault zero'
    assert gcs.fault_name_for_feature_index(1, strip = False) == 'fault one'
    # check a few faces are associated with expected features
    assert gcs.feature_index_for_cell_face((1, 3, 4), 1, 0) is None
    assert gcs.feature_index_for_cell_face((2, 2, 2), 2, 1) == 1
    assert gcs.feature_index_for_cell_face((2, 2, 2), 1, 1) == 0
    assert gcs.feature_index_for_cell_face((0, 2, 4), 1, 0) == 2
    # check faces associated with each feature
    k_faces, j_faces, i_faces = gcs.grid_face_arrays(kiwi_uuid,
                                                     default_value = -1,
                                                     feature_index = 2,
                                                     active_only = False,
                                                     lazy = False,
                                                     baffle_uuid = None)
    assert k_faces is not None and j_faces is not None and i_faces is not None
    assert k_faces.shape == (5, 5, 6)
    assert j_faces.shape == (4, 6, 6)
    assert i_faces.shape == (4, 5, 7)
    assert np.all(k_faces == -1)
    assert np.count_nonzero(j_faces >= 0) == 12
    assert np.all(i_faces == -1)
    assert np.all(np.logical_or(j_faces < 0, j_faces >= 300))
    k_faces, j_faces, i_faces = gcs.grid_face_arrays(kiwi_uuid,
                                                     default_value = -1,
                                                     feature_index = 1,
                                                     active_only = False,
                                                     lazy = False,
                                                     baffle_uuid = None)
    assert k_faces is not None and j_faces is not None and i_faces is not None
    assert k_faces.shape == (5, 5, 6)
    assert j_faces.shape == (4, 6, 6)
    assert i_faces.shape == (4, 5, 7)
    assert np.all(k_faces == -1)
    assert np.all(j_faces == -1)
    assert np.count_nonzero(i_faces >= 0) == 4
    assert np.all(np.logical_or(i_faces < 0, np.logical_and(i_faces >= 200, i_faces < 204)))
    k_faces, j_faces, i_faces = gcs.grid_face_arrays(kiwi_uuid,
                                                     default_value = -1,
                                                     feature_index = 0,
                                                     active_only = False,
                                                     lazy = False,
                                                     baffle_uuid = None)
    assert k_faces is not None and j_faces is not None and i_faces is not None
    assert k_faces.shape == (5, 5, 6)
    assert j_faces.shape == (4, 6, 6)
    assert i_faces.shape == (4, 5, 7)
    assert np.all(k_faces == -1)
    assert np.count_nonzero(j_faces >= 0) == 12
    assert np.all(i_faces == -1)
    assert np.all(j_faces < 112)


def test_feature_index_bool_pack(tmp_path):
    epc = os.path.join(tmp_path, 'gcs_with_packed_bool.epc')
    model = rq.new_model(epc)
    crs = rqc.Crs(model)
    crs.create_xml()
    grid = grr.RegularGrid(model, extent_kji = (4, 5, 6), crs_uuid = crs.uuid)
    grid.create_xml()
    gcs_0 = rqf.GridConnectionSet(model, find_properties = False, grid = grid, title = 'fault zero gcs')
    gcs_0.set_pairs_from_kelp(
        kelp_0 = [(2, 0), (2, 1), (2, 2)],  # J face columns
        kelp_1 = [],  # I face columns
        feature_name = 'fault zero',
        create_organizing_objects_where_needed = True)
    gcs_0.write_hdf5()
    gcs_0.create_xml()
    pc = gcs_0.extract_property_collection()
    a = np.array([False, True] * 6, dtype = bool)
    pc.add_cached_array_to_imported_list(a,
                                         'test',
                                         'this is a pack boolean',
                                         discrete = True,
                                         property_kind = 'flag',
                                         indexable_element = 'faces')
    pc.write_hdf5_for_imported_list(dtype = np.uint8, use_pack = True)
    uuids = pc.create_xml_for_imported_list_and_add_parts_to_model()
    assert len(uuids) == 1
    kiwi_uuids = uuids
    gcs_1 = rqf.GridConnectionSet(model, find_properties = False, grid = grid, title = 'fault one gcs')
    gcs_1.set_pairs_from_kelp(
        kelp_0 = [],  # J face columns
        kelp_1 = [(2, 2)],  # I face columns
        feature_name = 'fault one',
        create_organizing_objects_where_needed = True)
    gcs_1.write_hdf5()
    gcs_1.create_xml()
    pc = gcs_1.extract_property_collection()
    a = np.array((True, False, True, True), dtype = bool)
    pc.add_cached_array_to_imported_list(a,
                                         'test',
                                         'this is a single element packed bool!',
                                         discrete = True,
                                         property_kind = 'flag',
                                         indexable_element = 'faces')
    pc.write_hdf5_for_imported_list(dtype = np.uint8, use_pack = True)
    uuids = pc.create_xml_for_imported_list_and_add_parts_to_model()
    assert len(uuids) == 1
    kiwi_uuids += uuids
    gcs_2 = rqf.GridConnectionSet(model, find_properties = False, grid = grid, title = 'fault two gcs')
    gcs_2.set_pairs_from_kelp(
        kelp_0 = [(1, 3), (1, 4), (1, 5)],  # J face columns
        kelp_1 = [],  # I face columns
        feature_name = 'fault two',
        create_organizing_objects_where_needed = True)
    gcs_2.write_hdf5()
    gcs_2.create_xml()
    pc = gcs_2.extract_property_collection()
    a = np.array([True] * 8 + [False] * 4, dtype = bool)
    pc.add_cached_array_to_imported_list(a,
                                         'test',
                                         'this is another packed bool',
                                         discrete = True,
                                         property_kind = 'flag',
                                         indexable_element = 'faces')
    pc.write_hdf5_for_imported_list(dtype = np.uint8, use_pack = True)
    uuids = pc.create_xml_for_imported_list_and_add_parts_to_model()
    assert len(uuids) == 1
    kiwi_uuids += uuids
    model.store_epc()
    # check that fault interpretations have been generated
    fi_titles = model.titles(obj_type = 'FaultInterpretation')
    assert len(fi_titles) == 3
    assert set(fi_titles) == set(['fault zero', 'fault one', 'fault two'])
    # check one of the single feature grid connexion sets
    fi_one_uuid = model.uuid(obj_type = 'FaultInterpretation', title = 'fault one')
    assert fi_one_uuid is not None
    gcs_one_uuid = model.uuid(obj_type = 'GridConnectionSetRepresentation', related_uuid = fi_one_uuid)
    assert gcs_one_uuid is not None
    assert bu.matching_uuids(gcs_one_uuid, gcs_1.uuid)
    gcs_reload = rqf.GridConnectionSet(model, uuid = gcs_one_uuid)
    assert gcs_reload is not None
    assert gcs_reload.number_of_grids() == 1
    assert gcs_reload.number_of_features() == 1
    assert gcs_reload.title == 'fault one gcs'
    assert gcs_reload.list_of_feature_names(strip = False) == ['fault one']
    # make a composite grid connexion set with 3 features
    gcs = rqf.GridConnectionSet.from_gcs_uuid_list(model,
                                                   source_model = model,
                                                   gcs_uuid_list = [gcs_0.uuid, gcs_1.uuid, gcs_2.uuid],
                                                   title = 'multi feature fault system',
                                                   gcs_property_uuid_list_of_lists = [kiwi_uuids])
    model.store_epc()
    # above class method includes writing hdf5 and creating xml
    # reopen the composite grid connexion set
    gcs = rqf.GridConnectionSet(model, uuid = gcs.uuid)
    assert gcs is not None
    gcs.cache_arrays()
    assert gcs.number_of_grids() == 1
    assert gcs.number_of_features() == 3
    assert gcs.count == 28
    # check that a property exists on the composite gcs
    pc = gcs.extract_property_collection()
    # a = np.concatenate((np.arange(12, dtype = int) + 100, np.arange(4, dtype = int) + 200, np.arange(12, dtype = int) + 300))
    # pc.add_cached_array_to_imported_list(a, 'test', 'this is the property of a gcs', discrete = True, null_value = -1,
    #                                      property_kind = 'kiwi', indexable_element = 'faces')
    # pc.write_hdf5_for_imported_list()
    # uuids = pc.create_xml_for_imported_list_and_add_parts_to_model()
    # assert len(uuids) == 1
    assert pc.number_of_parts() == 1
    kiwi_uuid = model.uuid(obj_type = 'DiscreteProperty', related_uuid = gcs.uuid)
    assert kiwi_uuid is not None
    # check features
    assert gcs.list_of_feature_names(strip = False) == ['fault zero', 'fault one', 'fault two']
    assert gcs.list_of_fault_names(strip = False) == ['fault zero', 'fault one', 'fault two']
    index, fi_uuid = gcs.feature_index_and_uuid_for_fault_name('fault two')
    assert index == 2
    assert fi_uuid is not None
    fi = rqo.FaultInterpretation(model, uuid = fi_uuid)
    assert fi is not None
    assert fi.title == 'fault two'
    assert gcs.feature_name_for_feature_index(0, strip = False) == 'fault zero'
    assert gcs.fault_name_for_feature_index(1, strip = False) == 'fault one'
    # check a few faces are associated with expected features
    assert gcs.feature_index_for_cell_face((1, 3, 4), 1, 0) is None
    assert gcs.feature_index_for_cell_face((2, 2, 2), 2, 1) == 1
    assert gcs.feature_index_for_cell_face((2, 2, 2), 1, 1) == 0
    assert gcs.feature_index_for_cell_face((0, 2, 4), 1, 0) == 2
    # check faces associated with each feature
    k_faces, j_faces, i_faces = gcs.grid_face_arrays(kiwi_uuid,
                                                     default_value = False,
                                                     feature_index = 2,
                                                     active_only = False,
                                                     lazy = False,
                                                     baffle_uuid = None,
                                                     dtype = bool)
    assert k_faces is not None and j_faces is not None and i_faces is not None
    assert k_faces.shape == (5, 5, 6)
    assert j_faces.shape == (4, 6, 6)
    assert i_faces.shape == (4, 5, 7)
    assert not np.any(k_faces)
    assert np.count_nonzero(j_faces) == 8
    assert not np.any(i_faces)
    k_faces, j_faces, i_faces = gcs.grid_face_arrays(kiwi_uuid,
                                                     default_value = False,
                                                     feature_index = 1,
                                                     active_only = False,
                                                     lazy = False,
                                                     baffle_uuid = None,
                                                     dtype = bool)
    assert k_faces is not None and j_faces is not None and i_faces is not None
    assert k_faces.shape == (5, 5, 6)
    assert j_faces.shape == (4, 6, 6)
    assert i_faces.shape == (4, 5, 7)
    assert not np.any(k_faces)
    assert not np.any(j_faces)
    assert np.count_nonzero(i_faces) == 3
    k_faces, j_faces, i_faces = gcs.grid_face_arrays(kiwi_uuid,
                                                     default_value = False,
                                                     feature_index = 0,
                                                     active_only = False,
                                                     lazy = False,
                                                     baffle_uuid = None,
                                                     dtype = bool)
    assert k_faces is not None and j_faces is not None and i_faces is not None
    assert k_faces.shape == (5, 5, 6)
    assert j_faces.shape == (4, 6, 6)
    assert i_faces.shape == (4, 5, 7)
    assert not np.any(k_faces)
    assert np.count_nonzero(j_faces) == 6
    assert not np.any(i_faces)
