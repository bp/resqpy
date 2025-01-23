import numpy as np
import pytest

import resqpy.crs as rqc
import resqpy.surface as rqs
import resqpy.grid_surface as rqgs
import resqpy.property as rqp


def test_find_faces_to_represent_surface_regular_optimised(small_grid_and_surface):
    # Arrange
    grid = small_grid_and_surface[0]
    surface = small_grid_and_surface[1]
    old_fuddy_duddy_crs = rqc.Crs(surface.model, xy_units = 'ft', z_units = 'chain')
    old_fuddy_duddy_crs.create_xml()
    surface.model.store_epc()
    s2 = rqs.Surface(surface.model, uuid = surface.uuid)
    s2.change_crs(old_fuddy_duddy_crs)
    name = "test"
    assert grid.is_aligned

    # Act
    gcs_normal = rqgs.find_faces_to_represent_surface_regular(grid, surface, name)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs

    gcs_dense = rqgs.find_faces_to_represent_surface_regular_dense_optimised(grid, surface, name)
    cip_dense = gcs_dense.cell_index_pairs
    fip_dense = gcs_dense.face_index_pairs

    gcs_optimised = rqgs.find_faces_to_represent_surface_regular_optimised(grid, surface, name)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs

    gcs_old_fuddy_duddy = rqgs.find_faces_to_represent_surface_regular_optimised(grid, s2, name)
    cip_old_fuddy_duddy = gcs_old_fuddy_duddy.cell_index_pairs
    fip_old_fuddy_duddy = gcs_old_fuddy_duddy.face_index_pairs

    # Assert – quite harsh as gcs face ordering could legitimately vary
    np.testing.assert_array_equal(cip_normal, cip_dense)
    np.testing.assert_array_equal(fip_normal, fip_dense)
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)
    np.testing.assert_array_equal(cip_normal, cip_old_fuddy_duddy)
    np.testing.assert_array_equal(fip_normal, fip_old_fuddy_duddy)


def test_find_k_faces_only_to_represent_surface_regular_optimised(small_grid_and_surface):
    grid = small_grid_and_surface[0]
    surface = small_grid_and_surface[1]
    old_fuddy_duddy_crs = rqc.Crs(surface.model, xy_units = 'ft', z_units = 'chain')
    old_fuddy_duddy_crs.create_xml()
    surface.model.store_epc()
    s2 = rqs.Surface(surface.model, uuid = surface.uuid)
    s2.change_crs(old_fuddy_duddy_crs)
    name = "test"
    assert grid.is_aligned

    gcs_normal = rqgs.find_faces_to_represent_surface_regular(grid, surface, name)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs

    gcs_optimised = rqgs.find_faces_to_represent_surface_regular_optimised(grid, surface, name, direction = 'K')
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs

    gcs_old_fuddy_duddy = rqgs.find_faces_to_represent_surface_regular_optimised(grid, s2, name, direction = 'K')
    cip_old_fuddy_duddy = gcs_old_fuddy_duddy.cell_index_pairs
    fip_old_fuddy_duddy = gcs_old_fuddy_duddy.face_index_pairs

    assert len(cip_optimised) < len(cip_normal)
    assert len(fip_optimised) == len(cip_optimised)
    assert np.all(fip_optimised) < 2  # all +/- K faces
    np.testing.assert_array_equal(cip_optimised, cip_old_fuddy_duddy)
    np.testing.assert_array_equal(fip_optimised, fip_old_fuddy_duddy)


def test_find_faces_to_represent_surface_no_k_faces(small_grid_and_surface_no_k):
    # Arrange
    grid = small_grid_and_surface_no_k[0]
    surface = small_grid_and_surface_no_k[1]
    old_fuddy_duddy_crs = rqc.Crs(surface.model, xy_units = 'ft', z_units = 'chain')
    old_fuddy_duddy_crs.create_xml()
    surface.model.store_epc()
    s2 = rqs.Surface(surface.model, uuid = surface.uuid)
    s2.change_crs(old_fuddy_duddy_crs)
    name = "test"
    assert grid.is_aligned

    # Act
    gcs_normal = rqgs.find_faces_to_represent_surface_regular(grid, surface, name)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs

    gcs_dense = rqgs.find_faces_to_represent_surface_regular_dense_optimised(grid, surface, name)
    cip_dense = gcs_dense.cell_index_pairs
    fip_dense = gcs_dense.face_index_pairs

    gcs_optimised = rqgs.find_faces_to_represent_surface_regular_optimised(grid, surface, name)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs

    gcs_old_fuddy_duddy = rqgs.find_faces_to_represent_surface_regular_optimised(grid, s2, name)
    cip_old_fuddy_duddy = gcs_old_fuddy_duddy.cell_index_pairs
    fip_old_fuddy_duddy = gcs_old_fuddy_duddy.face_index_pairs

    # Assert – quite harsh as gcs face ordering could legitimately vary
    np.testing.assert_array_equal(cip_normal, cip_dense)
    np.testing.assert_array_equal(fip_normal, fip_dense)
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)
    np.testing.assert_array_equal(cip_normal, cip_old_fuddy_duddy)
    np.testing.assert_array_equal(fip_normal, fip_old_fuddy_duddy)


def test_find_faces_to_represent_surface_no_j_faces(small_grid_and_surface_no_j):
    # Arrange
    grid = small_grid_and_surface_no_j[0]
    surface = small_grid_and_surface_no_j[1]
    old_fuddy_duddy_crs = rqc.Crs(surface.model, xy_units = 'ft', z_units = 'chain')
    old_fuddy_duddy_crs.create_xml()
    surface.model.store_epc()
    s2 = rqs.Surface(surface.model, uuid = surface.uuid)
    s2.change_crs(old_fuddy_duddy_crs)
    name = "test"
    assert grid.is_aligned

    # Act
    gcs_normal = rqgs.find_faces_to_represent_surface_regular(grid, surface, name)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs

    gcs_dense = rqgs.find_faces_to_represent_surface_regular_dense_optimised(grid, surface, name)
    cip_dense = gcs_dense.cell_index_pairs
    fip_dense = gcs_dense.face_index_pairs

    gcs_optimised = rqgs.find_faces_to_represent_surface_regular_optimised(grid, surface, name)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs

    gcs_old_fuddy_duddy = rqgs.find_faces_to_represent_surface_regular_optimised(grid, s2, name)
    cip_old_fuddy_duddy = gcs_old_fuddy_duddy.cell_index_pairs
    fip_old_fuddy_duddy = gcs_old_fuddy_duddy.face_index_pairs

    # Assert – quite harsh as gcs face ordering could legitimately vary
    np.testing.assert_array_equal(cip_normal, cip_dense)
    np.testing.assert_array_equal(fip_normal, fip_dense)
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)
    np.testing.assert_array_equal(cip_normal, cip_old_fuddy_duddy)
    np.testing.assert_array_equal(fip_normal, fip_old_fuddy_duddy)


def test_find_faces_to_represent_surface_no_i_faces(small_grid_and_surface_no_i):
    # Arrange
    grid = small_grid_and_surface_no_i[0]
    surface = small_grid_and_surface_no_i[1]
    old_fuddy_duddy_crs = rqc.Crs(surface.model, xy_units = 'ft', z_units = 'chain')
    old_fuddy_duddy_crs.create_xml()
    surface.model.store_epc()
    s2 = rqs.Surface(surface.model, uuid = surface.uuid)
    s2.change_crs(old_fuddy_duddy_crs)
    name = "test"
    assert grid.is_aligned

    # Act
    gcs_normal = rqgs.find_faces_to_represent_surface_regular(grid, surface, name)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs

    gcs_dense = rqgs.find_faces_to_represent_surface_regular_dense_optimised(grid, surface, name)
    cip_dense = gcs_dense.cell_index_pairs
    fip_dense = gcs_dense.face_index_pairs

    gcs_optimised = rqgs.find_faces_to_represent_surface_regular_optimised(grid, surface, name)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs

    gcs_old_fuddy_duddy = rqgs.find_faces_to_represent_surface_regular_optimised(grid, s2, name)
    cip_old_fuddy_duddy = gcs_old_fuddy_duddy.cell_index_pairs
    fip_old_fuddy_duddy = gcs_old_fuddy_duddy.face_index_pairs

    # Assert – quite harsh as gcs face ordering could legitimately vary
    np.testing.assert_array_equal(cip_normal, cip_dense)
    np.testing.assert_array_equal(fip_normal, fip_dense)
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)
    np.testing.assert_array_equal(cip_normal, cip_old_fuddy_duddy)
    np.testing.assert_array_equal(fip_normal, fip_old_fuddy_duddy)


def test_find_faces_to_represent_surface_no_i_or_k_faces(small_grid_and_j_curtain_surface):
    # Arrange
    grid = small_grid_and_j_curtain_surface[0]
    surface = small_grid_and_j_curtain_surface[1]
    old_fuddy_duddy_crs = rqc.Crs(surface.model, xy_units = 'ft', z_units = 'chain')
    old_fuddy_duddy_crs.create_xml()
    surface.model.store_epc()
    s2 = rqs.Surface(surface.model, uuid = surface.uuid)
    s2.change_crs(old_fuddy_duddy_crs)
    name = "test"
    assert grid.is_aligned

    # Act
    gcs_normal = rqgs.find_faces_to_represent_surface_regular(grid, surface, name)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs

    gcs_dense = rqgs.find_faces_to_represent_surface_regular_dense_optimised(grid, surface, name)
    cip_dense = gcs_dense.cell_index_pairs
    fip_dense = gcs_dense.face_index_pairs

    gcs_optimised = rqgs.find_faces_to_represent_surface_regular_optimised(grid, surface, name)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs

    gcs_old_fuddy_duddy = rqgs.find_faces_to_represent_surface_regular_optimised(grid, s2, name)
    cip_old_fuddy_duddy = gcs_old_fuddy_duddy.cell_index_pairs
    fip_old_fuddy_duddy = gcs_old_fuddy_duddy.face_index_pairs

    # Assert – quite harsh as gcs face ordering could legitimately vary
    np.testing.assert_array_equal(cip_normal, cip_dense)
    np.testing.assert_array_equal(fip_normal, fip_dense)
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)
    np.testing.assert_array_equal(cip_normal, cip_old_fuddy_duddy)
    np.testing.assert_array_equal(fip_normal, fip_old_fuddy_duddy)
    _, fip = gcs_optimised.list_of_cell_face_pairs_for_feature_index()
    assert np.all(fip[:, :, 0] == 1)  # all J faces


def test_find_faces_to_represent_surface_no_j_or_k_faces(small_grid_and_i_curtain_surface):
    # Arrange
    grid = small_grid_and_i_curtain_surface[0]
    surface = small_grid_and_i_curtain_surface[1]
    old_fuddy_duddy_crs = rqc.Crs(surface.model, xy_units = 'ft', z_units = 'chain')
    old_fuddy_duddy_crs.create_xml()
    surface.model.store_epc()
    s2 = rqs.Surface(surface.model, uuid = surface.uuid)
    s2.change_crs(old_fuddy_duddy_crs)
    name = "test"
    assert grid.is_aligned

    # Act
    gcs_normal = rqgs.find_faces_to_represent_surface_regular(grid, surface, name)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs

    gcs_dense = rqgs.find_faces_to_represent_surface_regular_dense_optimised(grid, surface, name)
    cip_dense = gcs_dense.cell_index_pairs
    fip_dense = gcs_dense.face_index_pairs

    gcs_optimised = rqgs.find_faces_to_represent_surface_regular_optimised(grid, surface, name)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs

    gcs_old_fuddy_duddy = rqgs.find_faces_to_represent_surface_regular_optimised(grid, s2, name)
    cip_old_fuddy_duddy = gcs_old_fuddy_duddy.cell_index_pairs
    fip_old_fuddy_duddy = gcs_old_fuddy_duddy.face_index_pairs

    # Assert – quite harsh as gcs face ordering could legitimately vary
    np.testing.assert_array_equal(cip_normal, cip_dense)
    np.testing.assert_array_equal(fip_normal, fip_dense)
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)
    np.testing.assert_array_equal(cip_normal, cip_old_fuddy_duddy)
    np.testing.assert_array_equal(fip_normal, fip_old_fuddy_duddy)
    _, fip = gcs_optimised.list_of_cell_face_pairs_for_feature_index()
    assert np.all(fip[:, :, 0] == 2)  # all I faces


def test_find_faces_to_represent_surface_regular_optimised_constant_agitation(small_grid_and_surface):
    # Arrange
    grid = small_grid_and_surface[0]
    surface = small_grid_and_surface[1]
    name = "test"
    assert grid.is_aligned

    # Act
    gcs_normal = rqgs.find_faces_to_represent_surface_regular(grid,
                                                              surface,
                                                              name,
                                                              agitate = True,
                                                              random_agitation = False)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs

    gcs_optimised = rqgs.find_faces_to_represent_surface_regular_optimised(grid,
                                                                           surface,
                                                                           name,
                                                                           agitate = True,
                                                                           random_agitation = False)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs

    # Assert – quite harsh as gcs face ordering could differ
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)


def test_find_faces_to_represent_surface_regular_optimised_random_agitation(small_grid_and_surface):
    # Arrange
    grid = small_grid_and_surface[0]
    surface = small_grid_and_surface[1]
    name = "test"
    assert grid.is_aligned

    # Act
    gcs_normal = rqgs.find_faces_to_represent_surface_regular(grid,
                                                              surface,
                                                              name,
                                                              agitate = True,
                                                              random_agitation = True)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs

    gcs_optimised = rqgs.find_faces_to_represent_surface_regular_optimised(grid,
                                                                           surface,
                                                                           name,
                                                                           agitate = True,
                                                                           random_agitation = True)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs

    # Assert
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)


def test_find_faces_to_represent_surface_regular_optimised_with_return_properties(small_grid_and_surface):
    # Arrange
    grid = small_grid_and_surface[0]
    surface = small_grid_and_surface[1]
    surf_flange = rqp.Property.from_array(
        surface.model,
        cached_array = None,
        source_info = "constant False",
        keyword = "flange bool",
        support_uuid = surface.uuid,
        property_kind = "flange bool",
        indexable_element = "faces",
        discrete = True,
        const_value = 0,
        expand_const_arrays = False,
        dtype = bool,
    )
    name = "test"
    return_properties = ["offset"]

    # Act
    gcs_normal, properties_dict = rqgs.find_faces_to_represent_surface_regular(grid,
                                                                               surface,
                                                                               name,
                                                                               return_properties = return_properties)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs
    offsets_normal = properties_dict["offset"]

    return_properties.append("depth")
    return_properties.append("triangle")
    return_properties.append("flange bool")
    (
        gcs_optimised,
        properties_optimised,
    ) = rqgs.find_faces_to_represent_surface_regular_optimised(grid,
                                                               surface,
                                                               name,
                                                               return_properties = return_properties)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs
    triangles_optimised = properties_optimised["triangle"]
    depths_optimised = properties_optimised["depth"]
    offsets_optimised = properties_optimised["offset"]
    flange_optimised = properties_optimised["flange bool"]

    # Assert – quite harsh as faces could legitimately be in different order
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)
    # offsets are no longer all matching due to different handling of duplicate hits
    # np.testing.assert_array_almost_equal(offsets_normal, offsets_optimised)
    assert np.count_nonzero(np.isclose(offsets_normal, offsets_optimised)) > 2 * offsets_normal.size // 3
    assert depths_optimised.shape == offsets_optimised.shape
    assert np.all(depths_optimised > 0.0)
    assert triangles_optimised.shape == offsets_optimised.shape
    assert np.all(triangles_optimised >= 0)
    assert flange_optimised.shape == offsets_optimised.shape
    assert not np.any(flange_optimised)


def test_find_faces_to_represent_curtain_regular_optimised_with_return_properties(small_grid_and_surface_no_k):
    # Arrange
    grid = small_grid_and_surface_no_k[0]
    surface = small_grid_and_surface_no_k[1]
    surf_flange = rqp.Property.from_array(
        surface.model,
        cached_array = None,
        source_info = "constant False",
        keyword = "flange bool",
        support_uuid = surface.uuid,
        property_kind = "flange bool",
        indexable_element = "faces",
        discrete = True,
        const_value = 0,
        expand_const_arrays = False,
        dtype = bool,
    )
    name = "test"
    return_properties = ["offset"]

    # Act
    gcs_normal, properties_dict = rqgs.find_faces_to_represent_surface_regular(grid,
                                                                               surface,
                                                                               name,
                                                                               return_properties = return_properties)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs
    offsets_normal = properties_dict["offset"]

    return_properties.append("depth")
    return_properties.append("triangle")
    return_properties.append("flange bool")
    return_properties.append("grid bisector")
    (
        gcs_optimised,
        properties_optimised,
    ) = rqgs.find_faces_to_represent_surface_regular_optimised(grid,
                                                               surface,
                                                               name,
                                                               return_properties = return_properties)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs
    triangles_optimised = properties_optimised["triangle"]
    depths_optimised = properties_optimised["depth"]
    offsets_optimised = properties_optimised["offset"]
    flange_optimised = properties_optimised["flange bool"]
    bisector_optimised, is_curtain_optimised = properties_optimised["grid bisector"]
    (
        gcs_optimised_packed,
        properties_optimised_packed,
    ) = rqgs.find_faces_to_represent_surface_regular_optimised(grid,
                                                               surface,
                                                               name,
                                                               return_properties = return_properties,
                                                               packed_bisectors = True)
    bisector_packed, is_curtain_packed = properties_optimised["grid bisector"]

    # Assert – quite harsh as faces could legitimately be in different order
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)
    np.testing.assert_array_equal(gcs_optimised_packed.cell_index_pairs, cip_optimised)
    np.testing.assert_array_equal(gcs_optimised_packed.face_index_pairs, fip_optimised)
    # offsets are no longer all matching due to different handling of duplicate hits
    assert offsets_optimised.shape == offsets_normal.shape
    assert offsets_optimised.size == gcs_optimised.count
    assert depths_optimised.shape == offsets_optimised.shape
    assert np.all(depths_optimised > 0.0)
    assert triangles_optimised.shape == offsets_optimised.shape
    assert np.all(triangles_optimised >= 0)
    assert flange_optimised.shape == offsets_optimised.shape
    assert not np.any(flange_optimised)
    assert bisector_optimised.shape == (grid.nj, grid.ni)
    assert is_curtain_optimised
    assert bisector_packed.shape == (grid.nj, grid.ni)  # curtain bisectors are returned unpacked anyway!
    assert is_curtain_packed
    assert np.all(bisector_packed == bisector_optimised)


def test_find_faces_to_represent_surface_regular_dense_optimised_with_return_properties(small_grid_and_surface,):
    # Arrange
    grid = small_grid_and_surface[0]
    surface = small_grid_and_surface[1]
    surf_flange = rqp.Property.from_array(
        surface.model,
        cached_array = None,
        source_info = "constant False",
        keyword = "flange bool",
        support_uuid = surface.uuid,
        property_kind = "flange bool",
        indexable_element = "faces",
        discrete = True,
        const_value = 0,
        expand_const_arrays = False,
        dtype = bool,
    )
    name = "test"
    return_properties = ["offset"]

    # Act
    gcs_normal, properties_dict = rqgs.find_faces_to_represent_surface_regular(grid,
                                                                               surface,
                                                                               name,
                                                                               return_properties = return_properties)
    cip_normal = gcs_normal.cell_index_pairs
    fip_normal = gcs_normal.face_index_pairs
    offsets_normal = properties_dict["offset"]

    return_properties.append("depth")
    return_properties.append("triangle")
    return_properties.append("flange bool")
    (
        gcs_optimised,
        properties_optimised,
    ) = rqgs.find_faces_to_represent_surface_regular_dense_optimised(grid,
                                                                     surface,
                                                                     name,
                                                                     return_properties = return_properties)
    cip_optimised = gcs_optimised.cell_index_pairs
    fip_optimised = gcs_optimised.face_index_pairs
    triangles_optimised = properties_optimised["triangle"]
    depths_optimised = properties_optimised["depth"]
    offsets_optimised = properties_optimised["offset"]
    flange_optimised = properties_optimised["flange bool"]

    # Assert – quite harsh as faces could legitimately be in different order
    np.testing.assert_array_equal(cip_normal, cip_optimised)
    np.testing.assert_array_equal(fip_normal, fip_optimised)
    # offsets are no longer all matching due to different handling of duplicate hits
    # np.testing.assert_array_almost_equal(offsets_normal, offsets_optimised)
    assert np.count_nonzero(np.isclose(offsets_normal, offsets_optimised)) > 2 * offsets_normal.size // 3
    assert depths_optimised.shape == offsets_optimised.shape
    assert np.all(depths_optimised > 0.0)
    assert triangles_optimised.shape == offsets_optimised.shape
    assert np.all(triangles_optimised >= 0)
    assert flange_optimised.shape == offsets_optimised.shape
    assert not np.any(flange_optimised)


def test_bisector_from_faces_flat_surface_k():
    # Arrange
    grid_extent_kji = (3, 3, 3)
    k_faces = np.array(
        [
            [[True, True, True], [True, True, True], [True, True, True]],
            [[False, False, False], [False, False, False], [False, False, False]],
        ],
        dtype = bool,
    )
    k_face_indices = np.array([(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 1),
                               (0, 2, 2)],
                              dtype = np.int32)
    j_faces = np.array(
        [
            [[False, False, False], [False, False, False]],
            [[False, False, False], [False, False, False]],
            [[False, False, False], [False, False, False]],
        ],
        dtype = bool,
    )
    i_faces = np.array(
        [
            [[False, False], [False, False], [False, False]],
            [[False, False], [False, False], [False, False]],
            [[False, False], [False, False], [False, False]],
        ],
        dtype = bool,
    )

    # Act
    a, is_curtain = rqgs.bisector_from_faces(grid_extent_kji, k_faces, j_faces, i_faces, False)
    pa, p_is_curtain = rqgs.packed_bisector_from_face_indices(grid_extent_kji, k_face_indices, None, None, False, None)
    bounds = rqgs.get_boundary_dict(k_faces, j_faces, i_faces, grid_extent_kji)

    # Assert
    np.all(a == np.array(
        [
            [[True, True, True], [True, True, True], [True, True, True]],
            [[False, False, False], [False, False, False], [False, False, False]],
            [[False, False, False], [False, False, False], [False, False, False]],
        ],
        dtype = bool,
    ))
    assert is_curtain is False
    assert all([(bounds[f"{axis}_min"] == 0) for axis in "kji"])
    assert bounds["k_max"] == 1
    assert bounds["j_max"] == 2
    assert bounds["i_max"] == 2
    assert pa.ndim == 3
    assert pa.shape[:2] == a.shape[:2]
    assert pa.shape[2] == (a.shape[2] - 1) // 8 + 1
    assert np.all(np.unpackbits(pa, axis = 2, count = a.shape[2]).astype(bool) == a)
    assert p_is_curtain is False
    assert np.all(pa[0, :, -1] == 0xE0)  # 3 bits set in padded bytes (as ni < 8, all bytes are padded bytes here)


def test_bisector_from_faces_flat_surface_k_patch_box():
    # Arrange
    grid_extent_kji = (3, 3, 3)
    k_faces = np.array(
        [
            [[True, True, True], [True, True, True], [True, True, True]],
            [[False, False, False], [False, False, False], [False, False, False]],
        ],
        dtype = bool,
    )
    k_face_indices = np.array([(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 1),
                               (0, 2, 2)],
                              dtype = np.int32)
    j_faces = np.array(
        [
            [[False, False, False], [False, False, False]],
            [[False, False, False], [False, False, False]],
            [[False, False, False], [False, False, False]],
        ],
        dtype = bool,
    )
    i_faces = np.array(
        [
            [[False, False], [False, False], [False, False]],
            [[False, False], [False, False], [False, False]],
            [[False, False], [False, False], [False, False]],
        ],
        dtype = bool,
    )

    patch_box = np.array([(0, 0, 0), (2, 3, 3)], dtype = int)

    # Act
    a, is_curtain = rqgs.bisector_from_faces(grid_extent_kji, k_faces, j_faces, i_faces, False)
    pa, p_is_curtain = rqgs.packed_bisector_from_face_indices(grid_extent_kji, k_face_indices, None, None, False,
                                                              patch_box)
    bounds = rqgs.get_boundary_dict(k_faces, j_faces, i_faces, grid_extent_kji)

    # Assert
    np.all(a == np.array(
        [
            [[True, True, True], [True, True, True], [True, True, True]],
            [[False, False, False], [False, False, False], [False, False, False]],
            [[False, False, False], [False, False, False], [False, False, False]],
        ],
        dtype = bool,
    ))
    assert is_curtain is False
    assert all([(bounds[f"{axis}_min"] == 0) for axis in "kji"])
    assert bounds["k_max"] == 1
    assert bounds["j_max"] == 2
    assert bounds["i_max"] == 2
    assert pa.ndim == 3
    assert pa.shape[:2] == a.shape[:2]
    assert pa.shape[2] == (a.shape[2] - 1) // 8 + 1
    assert np.all(np.unpackbits(pa, axis = 2, count = a.shape[2]).astype(bool) == a)
    assert p_is_curtain is False
    assert np.all(pa[0, :, -1] == 0xE0)  # 3 bits set in padded bytes (as ni < 8, all bytes are padded bytes here)


def test_where_true_and_get_boundary():
    grid_extent_kji = (7, 8, 9)
    nk, nj, ni = grid_extent_kji
    k_faces = np.zeros((nk - 1, nj, ni), dtype = bool)
    j_faces = np.zeros((nk, nj - 1, ni), dtype = bool)
    i_faces = np.zeros((nk, nj, ni - 1), dtype = bool)
    k_faces[3, 3:7, 4:6] = True
    j_faces[4:6, 2, 4] = True
    i_faces[2:5, 3:6, 5] = True

    w_k, w_j, w_i = rqgs._where_true(k_faces)
    assert len(w_k) == 8
    assert np.all(np.unique(w_k) == (3,))
    assert np.all(np.unique(w_j) == (3, 4, 5, 6))
    assert np.all(np.unique(w_i) == (4, 5))
    w_k, w_j, w_i = rqgs._where_true(j_faces)
    assert len(w_k) == 2
    assert np.all(np.unique(w_k) == (4, 5))
    assert np.all(np.unique(w_j) == (2,))
    assert np.all(np.unique(w_i) == (4,))
    w_k, w_j, w_i = rqgs._where_true(i_faces)
    assert len(w_k) == 9
    assert np.all(np.unique(w_k) == (2, 3, 4))
    assert np.all(np.unique(w_j) == (3, 4, 5))
    assert np.all(np.unique(w_i) == (5,))

    bounds = rqgs.get_boundary_dict(k_faces, j_faces, i_faces, grid_extent_kji)

    # note: get_boundary_dict() includes a buffer slice where faces do not reach edge of grid
    assert bounds["k_min"] == 1
    assert bounds["k_max"] == 6
    assert bounds["j_min"] == 2
    assert bounds["j_max"] == 7
    assert bounds["i_min"] == 3
    assert bounds["i_max"] == 6


def test_bisector_from_faces_flat_surface_j():
    # Arrange
    grid_extent_kji = (3, 3, 3)
    k_faces = np.array(
        [
            [[False, False, False], [False, False, False], [False, False, False]],
            [[False, False, False], [False, False, False], [False, False, False]],
        ],
        dtype = bool,
    )
    j_faces = np.array(
        [
            [[True, True, True], [False, False, False]],
            [[True, True, True], [False, False, False]],
            [[True, True, True], [False, False, False]],
        ],
        dtype = bool,
    )
    j_face_indices = np.array([(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2), (2, 0, 0), (2, 0, 1),
                               (2, 0, 2)],
                              dtype = np.int32)
    i_faces = np.array(
        [
            [[False, False], [False, False], [False, False]],
            [[False, False], [False, False], [False, False]],
            [[False, False], [False, False], [False, False]],
        ],
        dtype = bool,
    )

    # Act
    a, is_curtain = rqgs.bisector_from_faces(grid_extent_kji, k_faces, j_faces, i_faces, False)
    pa, p_is_curtain = rqgs.packed_bisector_from_face_indices(grid_extent_kji, None, j_face_indices, None, False, None)
    ca = rqgs.column_bisector_from_faces(grid_extent_kji[1:], j_faces[0], i_faces[0])

    # Assert
    np.all(a == np.array(
        [
            [[True, True, True], [False, False, False], [False, False, False]],
            [[True, True, True], [False, False, False], [False, False, False]],
            [[True, True, True], [False, False, False], [False, False, False]],
        ],
        dtype = bool,
    ))
    assert is_curtain is True
    assert ca.shape == tuple(grid_extent_kji[1:])
    assert np.all(ca == a[0]) or np.all(ca == np.logical_not(a[0]))
    assert pa.ndim == 3
    assert pa.shape[:2] == a.shape[:2]
    assert pa.shape[2] == (a.shape[2] - 1) // 8 + 1
    assert np.all(np.unpackbits(pa, axis = 2, count = a.shape[2]).astype(bool) == a)
    assert p_is_curtain is True
    assert np.all(pa[:, 0, -1] == 0xE0)  # 3 bits set in padded bytes
    assert np.all(pa[:, 1:, -1] == 0)


def test_shadow_from_faces_curtain():
    # Arrange
    grid_extent_kji = (3, 3, 3)
    k_faces = np.array(
        [
            [[False, False, False], [False, False, False], [False, False, False]],
            [[False, False, False], [False, False, False], [False, False, False]],
        ],
        dtype = bool,
    )

    # Act
    a = rqgs.shadow_from_faces(grid_extent_kji, k_faces)

    # Assert
    assert a.shape == grid_extent_kji
    assert np.all(a == 0)


def test_bisector_from_faces_flat_surface_i():
    # Arrange
    grid_extent_kji = (3, 3, 3)
    k_faces = np.array(
        [
            [[False, False, False], [False, False, False], [False, False, False]],
            [[False, False, False], [False, False, False], [False, False, False]],
        ],
        dtype = bool,
    )
    j_faces = np.array(
        [
            [[False, False, False], [False, False, False]],
            [[False, False, False], [False, False, False]],
            [[False, False, False], [False, False, False]],
        ],
        dtype = bool,
    )
    i_faces = np.array(
        [
            [[True, False], [True, False], [True, False]],
            [[True, False], [True, False], [True, False]],
            [[True, False], [True, False], [True, False]],
        ],
        dtype = bool,
    )
    i_face_indices = np.array([(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0), (1, 1, 0), (1, 2, 0), (2, 0, 0), (2, 1, 0),
                               (2, 2, 0)],
                              dtype = np.int32)

    # Act
    a, is_curtain = rqgs.bisector_from_faces(grid_extent_kji, k_faces, j_faces, i_faces, False)
    pa, p_is_curtain = rqgs.packed_bisector_from_face_indices(grid_extent_kji, None, None, i_face_indices, False, None)

    # Assert
    np.all(a == np.array(
        [
            [[True, False, False], [True, False, False], [True, False, False]],
            [[True, False, False], [True, False, False], [True, False, False]],
            [[True, False, False], [True, False, False], [True, False, False]],
        ],
        dtype = bool,
    ))
    assert is_curtain is True
    assert pa.shape[:2] == a.shape[:2]
    assert pa.shape[2] == (a.shape[2] - 1) // 8 + 1
    assert np.all(np.unpackbits(pa, axis = 2, count = a.shape[2]).astype(bool) == a)
    assert p_is_curtain is True
    assert np.all(pa[:, :, -1] == 0x80)  # one bit set


def test_bisector_from_faces_flat_surface_k_hole():
    # Arrange
    grid_extent_kji = (3, 3, 3)
    k_faces = np.array(
        [
            [[True, True, True], [True, False, True], [True, True, True]],
            [[False, False, False], [False, False, False], [False, False, False]],
        ],
        dtype = bool,
    )
    k_face_indices = np.array([(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 2), (0, 2, 0), (0, 2, 1), (0, 2, 2)],
                              dtype = np.int32)
    j_faces = np.array(
        [
            [[False, False, False], [False, False, False]],
            [[False, False, False], [False, False, False]],
            [[False, False, False], [False, False, False]],
        ],
        dtype = bool,
    )
    i_faces = np.array(
        [
            [[False, False], [False, False], [False, False]],
            [[False, False], [False, False], [False, False]],
            [[False, False], [False, False], [False, False]],
        ],
        dtype = bool,
    )

    # Act & Assert
    with pytest.raises(AssertionError):
        rqgs.bisector_from_faces(grid_extent_kji, k_faces, j_faces, i_faces, False)
    with pytest.raises(AssertionError):
        rqgs.packed_bisector_from_face_indices(grid_extent_kji, k_face_indices, None, None, False, None)


def test_shadow_from_faces_flat_surface_k_hole():
    # Arrange
    grid_extent_kji = (3, 3, 3)
    k_faces = np.array(
        [
            [[True, True, True], [True, False, True], [True, True, True]],
            [[False, False, False], [False, False, False], [True, False, False]],
        ],
        dtype = bool,
    )

    # Act
    a = rqgs.shadow_from_faces(grid_extent_kji, k_faces)

    # Assert
    assert a.shape == grid_extent_kji
    assert np.all(a == np.array([
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        [[2, 2, 2], [2, 0, 2], [3, 2, 2]],
        [[2, 2, 2], [2, 0, 2], [2, 2, 2]],
    ]))


def test_find_faces_to_represent_surface_missing_grid(small_grid_and_missing_surface):
    # Arrange
    grid = small_grid_and_missing_surface[0]
    surface = small_grid_and_missing_surface[1]
    old_fuddy_duddy_crs = rqc.Crs(surface.model, xy_units = 'ft', z_units = 'chain')
    old_fuddy_duddy_crs.create_xml()
    surface.model.store_epc()
    s2 = rqs.Surface(surface.model, uuid = surface.uuid)
    s2.change_crs(old_fuddy_duddy_crs)
    name = "test"
    assert grid.is_aligned

    # Act
    gcs_optimised = rqgs.find_faces_to_represent_surface_regular_optimised(grid, surface, name)

    # Assert – quite harsh as gcs face ordering could legitimately vary
    assert gcs_optimised is None
