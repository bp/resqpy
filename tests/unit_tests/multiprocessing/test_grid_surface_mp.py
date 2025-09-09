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
    triangle_array = rqp.Property(model, uuid=model.uuid(title='small_surface trimmed triangle')).array_ref()
    expected_triangles = np.array([ 91,  91,  26, 162,  27,  41, 152, 159, 163,  23, 164, 179, 134,
                                    89, 143,  70, 158, 185, 123,  38,  88,  83,  66, 178, 138,  65,
                                   144, 133,  48, 133,  92,  70,   0, 117,  99, 154,   7,   2,   4,
                                     3,  50,  50,  57, 116, 168,  66,  14,  50,  17, 176,  50,  34,
                                    33,  60,  99,  99,  78,   7, 173, 156,   6, 170,  85,  18, 148,
                                    35,  72,  77,  99,   7, 108,  58, 106,  47,   5,  10, 110,  81,
                                    19, 146,  52, 147, 100,   4,  72, 131, 101, 103,  56, 114,  75,
                                    32, 149,  91, 139,  92,  27,  96, 150, 158, 163, 161,  25, 162,
                                    65, 149,  89,  35, 143,  92,  96,  70, 150, 158,  40, 152, 185,
                                    44, 125,  57,  38, 159, 110,  14,  83, 164, 179, 178, 138,  65,
                                   133, 149,  35,  53,  92, 141,  96,  70,  28, 150, 154,  40,   4,
                                   174, 123,  38,  46,   8, 124,  57, 159, 110,  66,  14, 176, 138,
                                    64,  65,  21, 136,  60,  48, 144, 149,  35,  52,  71, 117,  98,
                                   140,  97,  99,  28,   0,  54, 150, 154,  40,   6,   2, 172, 174,
                                     4,   5, 123,  57,  38, 116, 169,  47,   9,  14, 110, 168,  18,
                                   138,  62,  64,  21, 136,  16, 144, 148,  35,  73,  60,  52,  72,
                                   117,  98, 140,  97,  76,  78,   0, 150, 154, 171, 107, 106, 157,
                                     4,   4,   5,  57,  57, 116,  47,  47,   9,  14, 110, 137,  80,
                                    19, 148,  35,  16, 144,  72,  74,  60,  52,  77, 117,  93,  98,
                                    99,  99,  78, 104, 155, 126, 157,  47,   4,  10, 116,  47,  15,
                                   110,  21,  90,  81,  72,  74,  60, 146,  75, 100, 100,  99,  99,
                                    77,  56, 104,  55, 157,  10, 115,   4,  15,  81,  74,  32,  75,
                                   131, 100, 100,  99,  74, 163, 181,  26, 162,  66, 181, 149,  35,
                                    36,  68,  41, 108, 182, 185,  18, 163, 176,  83, 179, 137, 149,
                                    27,  36,  69, 121,  41,   2, 108, 158, 186,  10, 125, 160, 116,
                                    87,  17, 176,  66,  25,  18, 165, 178, 137,  63, 148,  89,  60,
                                   144, 147,  27, 143, 139,  37,  69, 117,  97,  29,  41,   2, 108,
                                   175, 156,   5, 186, 185,   9, 125, 122,  57, 161, 116,  84,  66,
                                    25,  17, 176,  83,  66,  18, 178, 137,  62,  16, 145, 148,  89,
                                    33,  23,  48, 144, 147,  72, 133, 121, 143,  70, 133, 118,  99,
                                    29, 155, 154,   7,   2, 108, 156,   6,  10,  20, 186, 185,   9,
                                    50, 167,  67,  50, 176,  85,  66,  50,  18, 178, 137,  33,  62,
                                    50,  16, 145, 148,  90,  33,  34, 144, 147,  72, 133, 121, 147,
                                    76, 130, 118,  99,  78, 155, 154,   7,   2,  54, 108,  10, 172,
                                    58, 174, 185, 113,  10, 125, 159,  85,  17, 138, 137,  81,  49,
                                    19, 148,  35,  33,  60, 146, 147,  72, 134, 121,  52, 147,  77,
                                   132, 118, 100, 155,   3,  54, 108,  10, 170, 188, 185, 113,  47,
                                     5, 109,  81, 145,  35,  73,  60, 147,  71,  32,  77, 131, 100,
                                   101, 155,   3,  54,  56,  10, 185, 114, 145,  72,  34,  71,  32,
                                   184,  75])
    np.testing.assert_array_equal(triangle_array, expected_triangles)
    rm_tree("tmp_dir")

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
    assert len(uuid_list) == 11


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
