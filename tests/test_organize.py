
import pytest

import resqpy.organize as rqo
from resqpy.model import Model


# Test saving and loading from disk
@pytest.mark.parametrize("cls, data", [
    (
        rqo.OrganizationFeature,
        dict(feature_name='hello', organization_kind='stratigraphic'),
    ),
    (
        rqo.GeobodyFeature,
        dict(feature_name='hi'),
    ),
    (
        rqo.BoundaryFeature,
        dict(feature_name='foobar'),
    ),
    (
        rqo.FrontierFeature,
        dict(feature_name='foobar'),
    ),
    (
        rqo.GeologicUnitFeature,
        dict(feature_name='foobar'),
    ),
    (
        rqo.FluidBoundaryFeature,
        dict(feature_name='foobar', kind='gas oil contact'),
    ),
])
def test_organize_classes(example_model, cls, data):

    # Load example model from a fixture
    model: Model
    model, _ = example_model
    epc = model.epc_file

    # Create the feature
    obj = cls(parent_model=model, **data)
    uuid = obj.uuid

    # Save to disk
    obj.create_xml()
    model.store_epc()
    model.h5_release()

    # Reload from disk
    del model, obj
    model2 = Model(epc_file=epc)
    obj2 = cls(parent_model=model2, uuid=uuid)

    # Check all attributes were loaded correctly
    for key, expected_value in data.items():
        assert getattr(obj2, key) == expected_value, f"Error for {key}"


def test_RockFluidUnitFeature(example_model):
    model, _ = example_model

    # Create the features
    top = rqo.BoundaryFeature(model, feature_name='the top')
    base = rqo.BoundaryFeature(model, feature_name='the base')
    rfuf_1 = rqo.RockFluidUnitFeature(
        parent_model=model, feature_name='foobar', phase='seal',
        top_boundary_feature=top, base_boundary_feature=base
    )
    uuid = rfuf_1.uuid

    # Save to disk
    top.create_xml()
    base.create_xml()
    rfuf_1.create_xml()
    model.store_epc()

    # Reload from disk
    rfuf_2 = rqo.RockFluidUnitFeature(parent_model=model, uuid=uuid)

    # Check properties the same
    assert rfuf_2.feature_name == 'foobar'
    assert rfuf_2.phase == 'seal'
    assert rfuf_2.top_boundary_feature.feature_name == 'the top'
    assert rfuf_2.base_boundary_feature.feature_name == 'the base'
