import pytest

import resqpy.olio.uuid as uuid
import resqpy.organize as rqo
from resqpy.model import Model


def test_is_equivalent_no_other(tmp_model):
    #Arrange
    fault_interp = rqo.FaultInterpretation(
        tmp_model,
        title = 'test_fault_interpretation',
        domain = "depth",
        is_normal = True,
        maximum_throw = 3,
    )
    #Act
    result = fault_interp.is_equivalent(other = None)

    #Assert
    assert result is False


def test_is_equivalent_other_not_FaultInterpretation(tmp_model):
    # Arrange
    fault_interp = rqo.FaultInterpretation(
        tmp_model,
        title = 'test_fault_interpretation',
        domain = "depth",
        is_normal = True,
        maximum_throw = 3,
    )
    # Act
    result = fault_interp.is_equivalent(other = 55)

    # Assert
    assert result is False


def test_is_equivalent_is_other(tmp_model):
    # Arrange
    fault_interp = rqo.FaultInterpretation(
        tmp_model,
        title = 'test_fault_interpretation',
    )
    # Act
    other = fault_interp
    result = fault_interp.is_equivalent(other = other)

    # Assert
    assert result is True


def test_is_equivalent_is_sameUUIDs(tmp_model):

    # Arrange
    object_uuid = uuid.new_uuid()
    fault_interp = rqo.FaultInterpretation(
        tmp_model,
        title = 'test_fault_interpretation',
    )
    fault_interp.uuid = object_uuid

    # Act
    other = rqo.FaultInterpretation(
        tmp_model,
        title = 'test_fault_interpretation_other',
    )
    other.uuid = object_uuid
    result = fault_interp.is_equivalent(other = other)

    # Assert
    assert result is True


def test_is_equivalent_is_tectonic_boundary_one_none(tmp_model):

    # Arrange
    fault_interp = rqo.FaultInterpretation(
        tmp_model,
        title = 'tectonic_boundary_test',
    )
    fault_interp.tectonic_boundary_feature = 'something'

    # Act
    other = rqo.FaultInterpretation(
        tmp_model,
        title = 'tectonic_boundary_test',
    )
    other.tectonic_boundary_feature = None
    result = fault_interp.is_equivalent(other = other)

    # Assert
    assert result is False


def test_is_equivalent_non_equivalent_tectonic_boundary(tmp_model):

    # Arrange
    tect_boundary = rqo.TectonicBoundaryFeature(tmp_model, kind = 'fault')

    fault_interp = rqo.FaultInterpretation(
        tmp_model,
        title = 'test_fault_interpretation',
    )
    fault_interp.tectonic_boundary_feature = tect_boundary

    # Act
    tect_boundary_other = rqo.TectonicBoundaryFeature(tmp_model, kind = 'fracture')

    other = rqo.FaultInterpretation(
        tmp_model,
        title = 'test_fault_interpretation',
    )
    other.tectonic_boundary_feature = tect_boundary_other
    result = fault_interp.is_equivalent(other = other)

    # Assert
    assert result is False


def test_is_equivalent_throw_math_is_close(tmp_model):

    # Arrange
    fault_interp = rqo.FaultInterpretation(
        tmp_model,
        title = 'test_fault_interpretation',
    )
    fault_interp.maximum_throw = 1

    # Act

    other = rqo.FaultInterpretation(
        tmp_model,
        title = 'test_fault_interpretation',
    )
    other.maximum_throw = 1.002

    result = fault_interp.is_equivalent(other = other)

    # Assert
    assert result is False
