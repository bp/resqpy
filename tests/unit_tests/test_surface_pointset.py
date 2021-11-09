import numpy as np
from numpy.testing import assert_array_almost_equal
import resqpy.grid
import resqpy.organize
import resqpy.surface

import pytest

# Unit tests for surface/Pointset methods


def test_from_charisma_method(mocker, example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    charisma_file = "INLINE :  25701 XLINE :  23693    420691.19624   6292314.22044      2799.05591\nINLINE :  25701 XLINE :  23694    420680.15765   6292308.35532      2798.08496"
    open_mock = mocker.mock_open(read_data = charisma_file)
    mocker.patch("builtins.open", open_mock)
    test_path = "path/to/file"

    array = np.array([[420691.19624, 6292314.22044, 2799.05591], [420680.15765, 6292308.35532, 2798.08496]])

    patch_mock = mocker.MagicMock(name = 'add_patch')
    # mocker.patch(resqpy.surface.PointSet.add_patch, patch_mock)

    # Act
    pointset = resqpy.surface.PointSet(model, crs_uuid = crs.uuid)
    pointset.add_patch = patch_mock
    pointset.from_charisma("path/to/file")

    # Assert
    open_mock.assert_called_once_with(test_path, 'r')
    patch_mock.assert_called_once()
    assert_array_almost_equal(patch_mock.call_args[0][0], array)


def test_from_charisma_method_failure(mocker, example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    charisma_file = "INLINE :  25701 XLINE :  23693    420691.19624   6292314.22044      2799.05591\nINLINE :  25701 XLINE :  23694    420680.15765   6292308.35532      2798.08496"
    open_mock = mocker.mock_open(read_data = charisma_file)
    mocker.patch("builtins.open", open_mock)

    # Act
    with pytest.raises(AssertionError) as e_info:
        pointset = resqpy.surface.PointSet(model, charisma_file = "path/to/file")

    # Assert
    assert str(e_info.value) == 'crs uuid missing when establishing point set from charisma file'


def test_from_irap_method(mocker, example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    irap_file = "429450.658333 6296954.224574 2403.837646\n429444.793211 6296965.263155 2403.449707"
    open_mock = mocker.mock_open(read_data = irap_file)
    mocker.patch("builtins.open", open_mock)
    test_path = "path/to/file"

    array = np.array([[429450.658333, 6296954.224574, 2403.837646], [429444.793211, 6296965.263155, 2403.449707]])

    patch_mock = mocker.MagicMock(name = 'add_patch')

    # Act
    pointset = resqpy.surface.PointSet(model, crs_uuid = crs.uuid)
    pointset.add_patch = patch_mock
    pointset.from_irap("path/to/file")

    # Assert
    open_mock.assert_called_once_with(test_path, 'r')
    patch_mock.assert_called_once()
    assert_array_almost_equal(patch_mock.call_args[0][0], array)


def test_from_irap_method_failure(mocker, example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    irap_file = "429450.658333 6296954.224574 2403.837646\n429444.793211 6296965.263155 2403.449707"
    open_mock = mocker.mock_open(read_data = irap_file)
    mocker.patch("builtins.open", open_mock)

    # Act
    with pytest.raises(AssertionError) as e_info:
        pointset = resqpy.surface.PointSet(model, irap_file = "path/to/file")

    # Assert
    assert str(e_info.value) == 'crs uuid missing when establishing point set from irap file'


@pytest.mark.parametrize('closed,coords1,coords2,expected',
                         [(True, np.array([[2, 2, 2], [3, 3, 3], [2, 2, 2]]), np.array(
                             [[1, 1, 1]]), np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])),
                          (False, np.array([[2, 2, 2], [3, 3, 3], [2, 2, 2]]), np.array(
                              [[1, 1, 1]]), np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [2, 2, 2]]))])
def test_concat_polyset_points(closed, coords1, coords2, expected):
    # Act
    result = resqpy.surface.pointset.concat_polyset_points(closed, coords1, coords2)
    # Assert
    assert_array_almost_equal(result, expected)
