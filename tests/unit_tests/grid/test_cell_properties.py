import numpy as np
import pytest

from resqpy.grid import Grid
import resqpy.grid as grr
from resqpy.model import Model
import resqpy.grid.cell_properties as cp
import resqpy.property.grid_property_collection as gpc


def test_thickness_array_thickness_already_set(basic_regular_grid: Grid):
    # Arrange
    extent = basic_regular_grid.extent_kji
    array_thickness = np.random.random(extent)
    basic_regular_grid.array_thickness = array_thickness  # type: ignore

    # Act
    thickness = cp.thickness(basic_regular_grid)

    # Assert
    np.testing.assert_array_almost_equal(thickness, array_thickness)


def test_thickness_array_thickness_already_set_cell_kji0(basic_regular_grid: Grid):
    # Arrange
    extent = basic_regular_grid.extent_kji
    array_thickness = np.random.random(extent)
    basic_regular_grid.array_thickness = array_thickness  # type: ignore
    cell_kji0 = (1, 1, 1)

    # Act
    thickness = cp.thickness(basic_regular_grid, cell_kji0 = cell_kji0)

    # Assert
    assert thickness == array_thickness[cell_kji0]


def test_thickness_faulted_grid(faulted_grid: Grid):
    # Arrange
    expected_thickness = np.array([[[20., 20., 20., 20., 20., 20., 20., 20.], [20., 20., 20., 20., 20., 20., 20., 20.],
                                    [20., 20., 20., 20., 20., 20., 20., 20.], [20., 20., 20., 20., 20., 20., 20., 20.],
                                    [20., 20., 20., 20., 20., 20., 20., 20.]],
                                   [[20., 20., 20., 20., 20., 20., 20., 20.], [20., 20., 20., 20., 20., 20., 20., 20.],
                                    [20., 20., 20., 20., 20., 20., 20., 20.], [20., 20., 20., 20., 20., 20., 20., 20.],
                                    [20., 20., 20., 20., 20., 20., 20., 20.]],
                                   [[10., 10., 5., 0., 0., 5., 10., 10.], [10., 10., 5., 0., 0., 5., 10., 10.],
                                    [10., 10., 5., 0., 0., 5., 10., 10.], [10., 10., 5., 0., 0., 5., 10., 10.],
                                    [10., 10., 5., 0., 0., 5., 10., 10.]]])

    # Act
    thickness = cp.thickness(faulted_grid)

    # Assert
    np.testing.assert_array_almost_equal(thickness, expected_thickness)


def test_thickness_blank_property_collection(basic_regular_grid: Grid):
    # Arrange
    property_collection = gpc.GridPropertyCollection()

    # Act
    thickness = cp.thickness(basic_regular_grid, property_collection = property_collection)

    # Assert
    assert thickness is None


def test_thickness_property_collection(example_model_with_properties: Model):
    # Arrange
    grid = example_model_with_properties.grid()
    extent = grid.extent_kji
    property_collection = grid.property_collection
    thickness_array = np.random.random(extent)
    property_collection.add_cached_array_to_imported_list(thickness_array,
                                                          'test data',
                                                          'DZ',
                                                          False,
                                                          uom = grid.z_units(),
                                                          property_kind = 'cell length',
                                                          facet_type = 'direction',
                                                          indexable_element = 'cells',
                                                          facet = 'K')
    property_collection.write_hdf5_for_imported_list()
    property_collection.create_xml_for_imported_list_and_add_parts_to_model()
    if hasattr(grid, 'array_thickness'):
        delattr(grid, 'array_thickness')

    # Act
    thickness = cp.thickness(grid, property_collection = property_collection)

    # Assert
    np.testing.assert_array_almost_equal(thickness, thickness_array)


def test_thickness_multiple_property_collection(example_model_with_properties: Model):
    # Arrange
    grid = example_model_with_properties.grid()
    extent = grid.extent_kji
    property_collection = grid.property_collection
    thickness_array_gross = np.random.random(extent)
    property_collection.add_cached_array_to_imported_list(thickness_array_gross,
                                                          'test data',
                                                          'DZ',
                                                          False,
                                                          uom = grid.z_units(),
                                                          property_kind = 'thickness',
                                                          facet_type = 'netgross',
                                                          indexable_element = 'cells',
                                                          facet = 'gross')
    thickness_array_net = np.random.random(extent) / 2
    property_collection.add_cached_array_to_imported_list(thickness_array_net,
                                                          'test data',
                                                          'DZ',
                                                          False,
                                                          uom = grid.z_units(),
                                                          property_kind = 'thickness',
                                                          facet_type = 'netgross',
                                                          indexable_element = 'cells',
                                                          facet = 'net')
    property_collection.write_hdf5_for_imported_list()
    property_collection.create_xml_for_imported_list_and_add_parts_to_model()
    if hasattr(grid, 'array_thickness'):
        delattr(grid, 'array_thickness')

    # Act
    thickness = cp.thickness(grid, property_collection = property_collection)

    # Assert
    np.testing.assert_array_almost_equal(thickness, thickness_array_gross)


def test_thickness_from_points(example_model_with_properties: Model):
    # Arrange
    grid = example_model_with_properties.grid()
    if hasattr(grid, 'array_thickness'):
        delattr(grid, 'array_thickness')
    if hasattr(grid, 'property_collection'):
        delattr(grid, 'property_collection')

    # Act
    thickness = cp.thickness(grid)

    # Assert
    np.testing.assert_array_almost_equal(thickness, 20.0)


def test_volume_array_volume_already_set(basic_regular_grid: Grid):
    # Arrange
    extent = basic_regular_grid.extent_kji
    array_volume = np.random.random(extent)
    basic_regular_grid.array_volume = array_volume  # type: ignore

    # Act
    volume = cp.volume(basic_regular_grid)

    # Assert
    np.testing.assert_array_almost_equal(volume, array_volume)


def test_volume_array_volume_already_set_cell_kji0(basic_regular_grid: Grid):
    # Arrange
    extent = basic_regular_grid.extent_kji
    array_volume = np.random.random(extent)
    basic_regular_grid.array_volume = array_volume  # type: ignore
    cell_kji0 = (1, 1, 1)

    # Act
    volume = cp.volume(basic_regular_grid, cell_kji0 = cell_kji0)

    # Assert
    assert volume == array_volume[cell_kji0]


def test_volume_faulted_grid(faulted_grid: Grid):
    # Arrange
    expected_volume = np.array([[[200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.]],
                                [[200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.],
                                 [200000., 200000., 200000., 200000., 200000., 200000., 200000., 200000.]],
                                [[100000., 100000., 50000., 0., 0., 50000., 100000., 100000.],
                                 [100000., 100000., 50000., 0., 0., 50000., 100000., 100000.],
                                 [100000., 100000., 50000., 0., 0., 50000., 100000., 100000.],
                                 [100000., 100000., 50000., 0., 0., 50000., 100000., 100000.],
                                 [100000., 100000., 50000., 0., 0., 50000., 100000., 100000.]]])

    # Act
    volume = cp.volume(faulted_grid)

    # Assert
    np.testing.assert_array_almost_equal(volume, expected_volume)


def test_volume_blank_property_collection(basic_regular_grid: Grid):
    # Arrange
    property_collection = gpc.GridPropertyCollection()

    # Act
    volume = cp.volume(basic_regular_grid, property_collection = property_collection)

    # Assert
    assert volume is None


def test_volume_property_collection(example_model_with_properties: Model):
    # Arrange
    grid = example_model_with_properties.grid()
    extent = grid.extent_kji
    property_collection = grid.property_collection
    volume_array = np.random.random(extent)
    property_collection.add_cached_array_to_imported_list(volume_array,
                                                          'test data',
                                                          'DZ',
                                                          property_kind = 'rock volume')
    property_collection.write_hdf5_for_imported_list()
    property_collection.create_xml_for_imported_list_and_add_parts_to_model()
    if hasattr(grid, 'array_volume'):
        delattr(grid, 'array_volume')

    # Act
    volume = cp.volume(grid, property_collection = property_collection)

    # Assert
    np.testing.assert_array_almost_equal(volume, volume_array)


def test_volume_multiple_property_collection(example_model_with_properties: Model):
    # Arrange
    grid = example_model_with_properties.grid()
    extent = grid.extent_kji
    property_collection = grid.property_collection
    volume_array_gross = np.random.random(extent)
    property_collection.add_cached_array_to_imported_list(volume_array_gross,
                                                          'test data',
                                                          'DZ',
                                                          property_kind = 'rock volume',
                                                          facet_type = 'netgross',
                                                          facet = 'gross')
    volume_array_net = np.random.random(extent) / 2
    property_collection.add_cached_array_to_imported_list(volume_array_net,
                                                          'test data',
                                                          'DZ',
                                                          property_kind = 'rock volume',
                                                          facet_type = 'netgross',
                                                          facet = 'net')
    property_collection.write_hdf5_for_imported_list()
    property_collection.create_xml_for_imported_list_and_add_parts_to_model()
    if hasattr(grid, 'array_volume'):
        delattr(grid, 'array_volume')

    # Act
    volume = cp.volume(grid, property_collection = property_collection)

    # Assert
    np.testing.assert_array_almost_equal(volume, volume_array_gross)


def test_volume_from_points(example_model_with_properties: Model):
    # Arrange
    grid = example_model_with_properties.grid()
    if hasattr(grid, 'array_volume'):
        delattr(grid, 'array_thickness')
    if hasattr(grid, 'property_volume'):
        delattr(grid, 'property_collection')

    # Act
    volume = cp.volume(grid)

    # Assert
    np.testing.assert_array_almost_equal(volume, 100000.0)


def test_cell_inactive_already_set(basic_regular_grid: Grid):
    # Arrange
    extent = basic_regular_grid.extent_kji
    inactive = np.random.choice([True, False], extent)
    basic_regular_grid.inactive = inactive  # type: ignore
    cell_kji0 = (1, 1, 1)

    # Act
    cell_inactive = cp.cell_inactive(basic_regular_grid, cell_kji0 = cell_kji0)

    # Assert
    assert cell_inactive == inactive[cell_kji0]


def test_cell_inactive_extract_inactive_mask(basic_regular_grid: Grid):
    # Arrange
    extent = tuple(basic_regular_grid.extent_kji)

    # Act & Assert
    for x, y, z in np.ndindex(extent):
        cell = (x, y, z)
        cell_inactive = cp.cell_inactive(basic_regular_grid, cell_kji0 = cell)
        assert cell_inactive is not True


@pytest.mark.parametrize("dxyz", [(100.0, 50.0, 20.0), (40.0, 60.0, 30.0), (72.1, 28.7, 84.6)])
def test_interface_length(model_test: Model, dxyz):
    # Arrange
    grid = grr.RegularGrid(model_test, extent_kji = (2, 2, 2), dxyz = dxyz, as_irregular_grid = True)
    cell_kji0 = (1, 1, 1)

    # Act & Assert
    for axis in range(3):
        interface_length = cp.interface_length(grid, cell_kji0 = cell_kji0, axis = axis)
        assert interface_length == dxyz[2 - axis]


@pytest.mark.parametrize("dxyz", [(100.0, 50.0, 20.0), (40.0, 60.0, 30.0), (72.1, 28.7, 84.6)])
def test_interface_lengths_kji(model_test: Model, dxyz):
    # Arrange
    grid = grr.RegularGrid(model_test, extent_kji = (2, 2, 2), dxyz = dxyz, as_irregular_grid = True)
    cell_kji0 = (1, 1, 1)

    # Act
    interface_length = cp.interface_lengths_kji(grid, cell_kji0 = cell_kji0)

    # Assert
    np.testing.assert_array_almost_equal(interface_length, dxyz[::-1])
