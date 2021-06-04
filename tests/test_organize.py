
import pytest

from resqpy import organize
from resqpy.model import Model


# Test saving and loading from disk
@pytest.mark.parametrize("cls, data", [
    (
        organize.OrganizationFeature,
        dict(feature_name='hello', organization_kind='stratigraphic'),
    ),
    (
        organize.GeobodyFeature,
        dict(feature_name='hi'),
    ),
    (
        organize.BoundaryFeature,
        dict(feature_name='foobar'),
    )
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
