import pytest

import resqpy.well
from resqpy.model import Model


def test_MdDatum(example_model_and_crs):

    # Set up a new datum
    model, crs = example_model_and_crs
    epc = model.epc_file
    data = dict(
        location = (0, -99999, 3.14),
        md_reference = 'mean low water',
    )
    datum = resqpy.well.MdDatum(parent_model = model, crs_uuid = crs.uuid, **data)
    uuid = datum.uuid

    # Save to disk and reload
    datum.create_part()
    model.store_epc()

    del model, crs, datum
    model2 = Model(epc_file = epc)
    datum2 = resqpy.well.MdDatum(parent_model = model2, uuid = uuid)

    for key, expected_value in data.items():
        assert getattr(datum2, key) == expected_value, f"Issue with {key}"

    identical = resqpy.well.MdDatum(parent_model = model2, crs_uuid = datum2.crs_uuid, **data)
    data['md_reference'] = 'kelly bushing'
    different = resqpy.well.MdDatum(parent_model = model2, crs_uuid = datum2.crs_uuid, **data)
    assert identical == datum2
    assert different != datum2


@pytest.mark.parametrize('other_data,expected',
                         [(dict(location = (0, -99999, 3.14), md_reference = 'mean low water'), True),
                          (dict(location = (5, -30000, 5), md_reference = 'kelly bushing'), False)])
def test_is_equivalent(example_model_and_crs, other_data, expected):
    # Test that comparison of metadata items returns the expected result

    # --------- Arrange ----------
    model, crs = example_model_and_crs
    data = dict(location = (0, -99999, 3.14), md_reference = 'mean low water')

    # --------- Act ----------
    datum1 = resqpy.well.MdDatum(parent_model = model, crs_uuid = crs.uuid, **data)
    datum2 = resqpy.well.MdDatum(parent_model = model, crs_uuid = crs.uuid, **other_data)
    result = datum1.is_equivalent(other = datum2)

    # --------- Assert ----------
    assert result is expected
