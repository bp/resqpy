import numpy as np

import resqpy.surface
import resqpy.organize
import resqpy.grid
import resqpy.grid_surface as rqgs
import resqpy.olio.uuid as bu


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
    crs = resqpy.crs.Crs(tmp_model)
    crs.create_xml()
    grid = resqpy.grid.RegularGrid(tmp_model, extent_kji = (3, 3, 3),
                                   crs_uuid = crs.uuid, set_points_cached = True)
    grid.write_hdf5()
    grid.create_xml(write_geometry = True)
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
    surf = resqpy.surface.Surface(tmp_model, crs_uuid = crs.uuid)
    surf.set_from_triangles_and_points(triangles, points.reshape((-1, 3)))
    assert surf is not None
    gcs = rqgs.find_faces_to_represent_surface(grid, surf, 'staffa', mode = 'staffa')
    assert gcs is not None
    assert gcs.count == 12
    cip = set([tuple(pair) for pair in gcs.cell_index_pairs])
    expected_cip = grid.natural_cell_indices(
        np.array([[[0,0,0], [1,0,0]], [[0,1,0], [1,1,0]], [[0,2,0], [1,2,0]],
                  [[1,0,0], [1,0,1]], [[1,1,0], [1,1,1]], [[1,2,0], [1,2,1]],
                  [[1,0,1], [2,0,1]], [[1,1,1], [2,1,1]], [[1,2,1], [2,2,1]],
                  [[2,0,1], [2,0,2]], [[2,1,1], [2,1,2]], [[2,2,1], [2,2,2]]], dtype = int))
    e_cip = set([tuple(pair) for pair in expected_cip])
    assert cip == e_cip  # note: this assumes lower cell index is first, which happens to be true
    # todo: check face indices
    gcs.write_hdf5()
    gcs.create_xml()
    assert bu.matching_uuids(tmp_model.uuid(obj_type = 'GridConnectionSetRepresentation'), gcs.uuid)
