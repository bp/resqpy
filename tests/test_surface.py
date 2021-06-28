import numpy as np

import resqpy.surface
import resqpy.organize
import resqpy.grid
import resqpy.grid_surface as rqgs


def test_surface(tmp_model):

    # Set up a Surface
    title = 'Mountbatten'
    model = tmp_model
    surf = resqpy.surface.Surface(
        parent_model=model, title=title
    )
    surf.create_xml()

    # Add a interpretation
    assert surf.represented_interpretation_root is None
    surf.create_interpretation_and_feature(kind='fault')
    assert surf.represented_interpretation_root is not None

    # Check fault can be loaded in again
    model.store_epc()
    fault_interp = resqpy.organize.FaultInterpretation(
        model, uuid=surf.represented_interpretation_uuid
    )
    fault_feature = resqpy.organize.TectonicBoundaryFeature(
        model, uuid=fault_interp.tectonic_boundary_feature.uuid
    )

    # Check title matches expected title
    assert fault_feature.feature_name == title


def test_faces_for_surface(tmp_model):
    grid = resqpy.grid.RegularGrid(tmp_model, extent_kji = (3, 3, 3), set_points_cached = True)
    # todo: create sloping planar surface
    # call find faces for each of 3 different methods
    points = np.zeros((2, 2, 3))
    points[1, :, 1] = 3.0
    points[:, 1, 0] = 3.0
    points[:, 1, 2] = 3.0
    points[:, :, 2] += 0.25
    triangles = np.zeros((2, 3), dtype = int)
    triangles[0] = (0, 1, 2)
    triangles[1] = (3, 1, 2)
    surf = resqpy.surface.Surface(tmp_model).set_from_triangles_and_points(triangles, points)
    assert surf is not None
    gcs1 = rqgs.find_faces_to_represent_surface(grid, surf, 'elephantine', mode = 'elephantine')
    assert gcs1 is not None
    gcs2 = rqgs.find_faces_to_represent_surface(grid, surf, 'loopy', mode = 'loopy')
    assert gcs2 is not None
    gcs3 = rqgs.find_faces_to_represent_surface(grid, surf, 'staffa', mode = 'staffa')
    assert gcs3 is not None
    assert gcs1.count == gcs2.count == gcs3.count == 12
    cip1 = set([tuple(pair) for pair in gcs1.cell_index_pairs])
    cip2 = set([tuple(pair) for pair in gcs2.cell_index_pairs])
    cip3 = set([tuple(pair) for pair in gcs3.cell_index_pairs])
    expected_cip = grid.normalized_cell_indices(
        np.array([[[0,0,0], [1,0,0]], [[0,0,1], [1,0,1]], [[0,0,2], [1,0,2]],
                  [[1,0,0], [1,1,0]], [[1,0,1], [1,1,1]], [[1,0,2], [1,1,2]],
                  [[1,1,0], [2,1,0]], [[1,1,1], [2,1,1]], [[1,1,2], [2,1,2]],
                  [[2,1,0], [2,2,0]], [[2,1,1], [2,2,1]], [[2,1,2], [2,2,2]]], dtype = int))
    e_cip = set([tuple(pair) for pair in expected_cip])
    assert cip1 == cip2 == cip3 == e_cip
    # todo: check face indices
    gcs_uuids = set()
    for gcs in (gcs1, gcs2, gcs3):
        gcs.write_hdf5()
        gcs.create_xml()
        gcs_uuids += gcs.uuid
    for gcs in tmp_model.iter_grid_connection_sets():
        assert gcs.uuid in gcs_uuids
        gcs_uuids -= gcs.uuid
    assert len(gcs_uuids) == 0
