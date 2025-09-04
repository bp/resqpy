from typing import Tuple
import numpy as np

import resqpy.model as rq
import resqpy.property as rqp
import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu
from resqpy.model import Model
from resqpy.grid import RegularGrid
from resqpy.surface import Surface, PointSet
from resqpy.multi_processing.wrappers.grid_surface_mp import find_faces_to_represent_surface_regular_wrapper
from resqpy.multi_processing._multiprocessing import rm_tree
from resqpy.olio.random_seed import seed

seed(83469613)


def test_find_faces_to_represent_surface_regular_wrapper(small_grid_and_surface: Tuple[RegularGrid, Surface]):
    # Arrange
    grid, surface = small_grid_and_surface
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(input_index,
                                                                                          "tmp_dir",
                                                                                          use_index_as_realisation,
                                                                                          grid_epc,
                                                                                          grid_uuid,
                                                                                          surface_epc,
                                                                                          surface_uuid,
                                                                                          name,
                                                                                          random_agitation = False,
                                                                                          trimmed = True)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids()) == 9
    assert len(uuid_list) == 7


def test_find_faces_to_represent_surface_regular_wrapper_point_set(small_grid_and_surface: Tuple[RegularGrid, Surface]):
    # Arrange
    grid, surface = small_grid_and_surface
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    _, p = surface.triangles_and_points()
    ps = PointSet(surface.model, points_array = p, crs_uuid = surface.crs_uuid, title = surface.title)
    ps.write_hdf5()
    ps.create_xml()
    ps_uuid = ps.uuid
    ps.model.store_epc()

    name = "test pointset"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(input_index,
                                                                                          "tmp_dir",
                                                                                          use_index_as_realisation,
                                                                                          grid_epc,
                                                                                          grid_uuid,
                                                                                          surface_epc,
                                                                                          ps_uuid,
                                                                                          name,
                                                                                          random_agitation = False,
                                                                                          trimmed = True)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'PointSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids()) == 10
    assert len(uuid_list) == 7


def test_find_faces_to_represent_surface_regular_wrapper_random_agitation(small_grid_and_surface: Tuple[RegularGrid,
                                                                                                        Surface]):
    # Arrange
    grid, surface = small_grid_and_surface
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(input_index,
                                                                                          "tmp_dir",
                                                                                          use_index_as_realisation,
                                                                                          grid_epc,
                                                                                          grid_uuid,
                                                                                          surface_epc,
                                                                                          surface_uuid,
                                                                                          name,
                                                                                          random_agitation = True,
                                                                                          trimmed = True)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids()) == 9
    assert len(uuid_list) == 7


def test_find_faces_to_represent_surface_regular_wrapper_properties(small_grid_and_surface: Tuple[RegularGrid,
                                                                                                  Surface]):
    # Arrange
    grid, surface = small_grid_and_surface
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid
    return_properties = ["triangle", "offset"]

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index,
        "tmp_dir",
        use_index_as_realisation,
        grid_epc,
        grid_uuid,
        surface_epc,
        surface_uuid,
        name,
        return_properties = return_properties)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 2
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids(obj_type = 'DiscreteProperty')) == 1
    assert len(model.uuids(obj_type = 'ContinuousProperty')) == 4
    assert len(model.uuids(obj_type = 'PropertyKind', title = "offset")) == 1
    assert len(model.uuids()) == 14
    assert len(uuid_list) == 9


def test_find_faces_to_represent_surface_extended_bisector(small_grid_and_extended_surface: Tuple[RegularGrid,
                                                                                                  Surface]):
    # Arrange
    grid, surface = small_grid_and_extended_surface
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid
    return_properties = ["triangle", "offset", "grid bisector", "grid shadow"]

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index,
        "tmp_dir",
        use_index_as_realisation,
        grid_epc,
        grid_uuid,
        surface_epc,
        surface_uuid,
        name,
        return_properties = return_properties)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 2
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids(obj_type = 'DiscreteProperty')) == 3
    assert len(model.uuids(obj_type = 'ContinuousProperty')) == 4
    assert len(model.uuids()) == 18
    assert len(uuid_list) == 11


def test_find_faces_to_represent_surface_regular_wrapper_properties_flange(small_grid_and_surface: Tuple[RegularGrid,
                                                                                                         Surface]):
    # Arrange
    grid, surface = small_grid_and_surface
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid
    return_properties = ["triangle", "offset"]

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index,
        "tmp_dir",
        use_index_as_realisation,
        grid_epc,
        grid_uuid,
        surface_epc,
        surface_uuid,
        name,
        return_properties = return_properties,
        extend_fault_representation = True)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 2
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids(obj_type = 'DiscreteProperty')) == 2
    assert len(model.uuids(obj_type = 'ContinuousProperty')) == 4
    assert len(model.uuids()) == 16
    assert len(uuid_list) == 10


def test_find_faces_to_represent_surface_regular_wrapper_flange_radius(small_grid_and_surface: Tuple[RegularGrid,
                                                                                                     Surface]):
    # Arrange
    grid, surface = small_grid_and_surface
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid
    return_properties = ["triangle", "offset", "flange bool"]

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index,
        "tmp_dir",
        use_index_as_realisation,
        grid_epc,
        grid_uuid,
        surface_epc,
        surface_uuid,
        name,
        return_properties = return_properties,
        extend_fault_representation = True,
        flange_radius = 3000.0)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 2
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids(obj_type = 'DiscreteProperty')) == 3
    assert len(model.uuids(obj_type = 'ContinuousProperty')) == 4
    assert len(model.uuids()) == 17
    assert len(uuid_list) == 17


def test_find_faces_to_represent_surface_extended_bisector_use_pack(small_grid_and_extended_surface: Tuple[RegularGrid,
                                                                                                           Surface]):
    # Arrange
    grid, surface = small_grid_and_extended_surface
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid
    return_properties = ["triangle", "offset", "grid bisector", "grid shadow"]

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index,
        "tmp_dir",
        use_index_as_realisation,
        grid_epc,
        grid_uuid,
        surface_epc,
        surface_uuid,
        name,
        return_properties = return_properties,
        use_pack = True)
    model = Model(epc_file = epc_file)

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 2
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids(obj_type = 'DiscreteProperty')) == 3
    assert len(model.uuids(obj_type = 'ContinuousProperty')) == 4
    assert len(model.uuids()) == 18
    assert len(uuid_list) == 11

    for uuid in model.uuids(obj_type = 'DiscreteProperty'):
        a = rqp.Property(model, uuid = uuid).array_ref()
        assert a is not None

    rm_tree("tmp_dir")


def test_find_faces_to_represent_surface_regular_wrapper_flange_radius_saucer_noreorient(
        small_grid_and_surface: Tuple[RegularGrid, Surface]):
    # Arrange
    grid, surface = small_grid_and_surface
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid
    return_properties = ["triangle", "offset"]

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index,
        "tmp_dir",
        use_index_as_realisation,
        grid_epc,
        grid_uuid,
        surface_epc,
        surface_uuid,
        name,
        return_properties = return_properties,
        extend_fault_representation = True,
        flange_radius = 3000.0,
        reorient = False,
        saucer_parameter = -60)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 2
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids(obj_type = 'DiscreteProperty')) == 2
    assert len(model.uuids(obj_type = 'ContinuousProperty')) == 4
    assert len(model.uuids()) == 16
    assert len(uuid_list) == 10


def test_find_faces_to_represent_surface_regular_wrapper_flange_radius_saucer_reorient(
        small_grid_and_surface: Tuple[RegularGrid, Surface]):
    # Arrange
    grid, surface = small_grid_and_surface
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid
    return_properties = ["triangle", "offset"]

    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index,
        "tmp_dir",
        use_index_as_realisation,
        grid_epc,
        grid_uuid,
        surface_epc,
        surface_uuid,
        name,
        return_properties = return_properties,
        extend_fault_representation = True,
        flange_radius = 3000.0,
        reorient = True,
        saucer_parameter = -60)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 2
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids(obj_type = 'DiscreteProperty')) == 2
    assert len(model.uuids(obj_type = 'ContinuousProperty')) == 4
    assert len(model.uuids()) == 16
    assert len(uuid_list) == 10


def test_find_faces_to_represent_surface_regular_wrapper_patchwork(small_grid_and_surface: Tuple[RegularGrid, Surface]):
    # Arrange
    grid, surface = small_grid_and_surface
    grid_epc = surface_epc = grid.model.epc_file
    assert surface.model is grid.model
    model = grid.model
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid
    region_lookup = rqp.StringLookup(model,
                                     int_to_str_dict = {
                                         0: 'a',
                                         1: 'b',
                                         2: 'c',
                                         3: 'd',
                                         4: 'e'
                                     },
                                     title = 'regions')
    region_lookup.create_xml()
    assert surface.number_of_patches() == 1
    grid_reg_prop = rqp.Property.from_array(model,
                                            cached_array = None,
                                            const_value = 3,
                                            source_info = 'testing',
                                            keyword = 'region',
                                            support_uuid = grid.uuid,
                                            property_kind = 'region initialization',
                                            indexable_element = 'cells',
                                            facet_type = 'what',
                                            facet = 'equilibration',
                                            discrete = True,
                                            null_value = -1,
                                            string_lookup_uuid = region_lookup.uuid,
                                            expand_const_arrays = True,
                                            dtype = np.int8)
    surf_reg_prop = rqp.Property.from_array(model,
                                            cached_array = np.array([3], dtype = np.int8),
                                            source_info = 'testing',
                                            keyword = 'region',
                                            support_uuid = surface.uuid,
                                            property_kind = 'region initialization',
                                            indexable_element = 'patches',
                                            facet_type = 'what',
                                            facet = 'equilibration',
                                            discrete = True,
                                            null_value = -1,
                                            string_lookup_uuid = region_lookup.uuid,
                                            dtype = np.int8)
    model.store_epc()

    return_properties = ["triangle", "grid bisector"]
    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index,
        "tmp_dir",
        use_index_as_realisation,
        grid_epc,
        grid_uuid,
        surface_epc,
        surface_uuid,
        name,
        return_properties = return_properties,
        raw_bisector = True,
        patchwork = True,
        grid_patching_property_uuid = grid_reg_prop.uuid,
        surface_patching_property_uuid = surf_reg_prop.uuid)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids(obj_type = 'DiscreteProperty')) == 2
    assert len(model.uuids(obj_type = 'ContinuousProperty')) == 3
    assert len(model.uuids(obj_type = 'PropertyKind', title = "triangle index")) == 1
    assert len(model.uuids(obj_type = 'PropertyKind', title = "grid bisector")) == 1
    assert len(model.uuids()) == 16
    assert len(uuid_list) == 9


def test_find_faces_to_represent_surface_extended_patchwork(small_grid_and_extended_surface: Tuple[RegularGrid,
                                                                                                   Surface]):
    # Arrange
    grid, surface = small_grid_and_extended_surface
    assert surface.model is grid.model
    model = grid.model
    grid_epc = surface_epc = grid.model.epc_file
    grid_uuid = grid.uuid
    surface_uuid = surface.uuid
    region_lookup = rqp.StringLookup(model,
                                     int_to_str_dict = {
                                         0: 'a',
                                         1: 'b',
                                         2: 'c',
                                         3: 'd',
                                         4: 'e'
                                     },
                                     title = 'regions')
    region_lookup.create_xml()
    assert surface.number_of_patches() == 1
    grid_reg_prop = rqp.Property.from_array(model,
                                            cached_array = None,
                                            const_value = 3,
                                            source_info = 'testing',
                                            keyword = 'region',
                                            support_uuid = grid.uuid,
                                            property_kind = 'region initialization',
                                            indexable_element = 'cells',
                                            facet_type = 'what',
                                            facet = 'equilibration',
                                            discrete = True,
                                            null_value = -1,
                                            string_lookup_uuid = region_lookup.uuid,
                                            expand_const_arrays = True,
                                            dtype = np.int8)
    surf_reg_prop = rqp.Property.from_array(model,
                                            cached_array = np.array([3], dtype = np.int8),
                                            source_info = 'testing',
                                            keyword = 'region',
                                            support_uuid = surface.uuid,
                                            property_kind = 'region initialization',
                                            indexable_element = 'patches',
                                            facet_type = 'what',
                                            facet = 'equilibration',
                                            discrete = True,
                                            null_value = -1,
                                            string_lookup_uuid = region_lookup.uuid,
                                            dtype = np.int8)
    model.store_epc()

    return_properties = ["triangle", "offset", "grid bisector", "grid shadow"]
    name = "test"
    input_index = 0
    use_index_as_realisation = False

    # Act
    index, success, epc_file, uuid_list = find_faces_to_represent_surface_regular_wrapper(
        input_index,
        "tmp_dir",
        use_index_as_realisation,
        grid_epc,
        grid_uuid,
        surface_epc,
        surface_uuid,
        name,
        return_properties = return_properties,
        patchwork = True,
        grid_patching_property_uuid = grid_reg_prop.uuid,
        surface_patching_property_uuid = surf_reg_prop.uuid)
    model = Model(epc_file = epc_file)
    rm_tree("tmp_dir")

    # Assert
    assert success is True
    assert index == input_index
    assert len(model.uuids(obj_type = 'LocalDepth3dCrs')) == 1
    assert len(model.uuids(obj_type = 'IjkGridRepresentation')) == 1
    assert len(model.uuids(obj_type = 'TriangulatedSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'GridConnectionSetRepresentation')) == 1
    assert len(model.uuids(obj_type = 'FaultInterpretation')) == 1
    assert len(model.uuids(obj_type = 'TectonicBoundaryFeature')) == 1
    assert len(model.uuids(obj_type = 'DiscreteProperty')) == 3
    assert len(model.uuids(obj_type = 'ContinuousProperty')) == 4
    assert len(model.uuids()) == 20
    assert len(uuid_list) == 11
