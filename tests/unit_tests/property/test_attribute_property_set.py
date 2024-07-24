import math as maths
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import resqpy.property as rqp
import resqpy.olio.uuid as bu

#import resqpy.derived_model as rqdm
#import resqpy.grid as grr
#import resqpy.model as rq
#import resqpy.olio.vector_utilities as vec
#import resqpy.property as rqp
#import resqpy.property._collection_get_attributes as pcga
#import resqpy.time_series as rqts
#import resqpy.weights_and_measures as bwam
#import resqpy.surface as rqs
#import resqpy.olio.xml_et as rqet
#import resqpy.well as rqw
#import resqpy.lines as rql
#from resqpy.crs import Crs

#from resqpy.property import property_kind_and_facet_from_keyword, guess_uom


def test_load_attribute_property_collection_pk(example_model_with_prop_ts_rels):
    # Arrange
    model = example_model_with_prop_ts_rels
    grid = model.grid()

    # Act
    aps = rqp.AttributePropertySet(support = grid)

    # Assert
    assert aps is not None
    assert len(aps) == 12
    for k in aps.keys():
        assert k is not None
        assert isinstance(k, str)
    for p in aps.properties():
        assert p is not None
        assert isinstance(p, rqp.ApsProperty)
    # check zone property
    assert aps.zone.property_kind == 'zone'
    assert aps.zone.facet_type is None
    assert aps.zone.facet is None
    assert aps.zone.indexable == 'cells'
    assert aps.zone.indexable_element == 'cells'
    assert aps.zone.is_continuous is False
    assert aps.zone.is_discrete is True
    assert aps.zone.is_categorical is False
    assert aps.zone.is_points is False
    assert aps.zone.count == 1
    assert aps.zone.uom is None
    assert aps.zone.null_value == -1
    assert aps.zone.realization is None
    assert aps.zone.time_index is None
    assert aps.zone.title == 'Zone'
    assert aps.zone.citation_title == 'Zone'
    assert aps.zone.min_value == 1
    assert aps.zone.max_value == 3
    assert aps.zone.constant_value is None
    assert aps.zone.extra == {}
    assert aps.zone.extra_metadata == {}
    assert bu.matching_uuids(aps.zone.support_uuid, grid.uuid)
    assert aps.zone.string_lookup_uuid is None
    assert aps.zone.time_series_uuid is None
    assert aps.zone.local_property_kind_uuid is not None
    v = aps.zone.values
    assert isinstance(v, np.ndarray)
    assert v.shape == (3, 5, 5)
    assert tuple(np.unique(v)) == (1, 2, 3)
    assert np.all(v == aps.zone.array_ref)
    assert aps.zone.uuid is not None
    assert aps.zone.node is not None
    # check net to gross ratio properties
    assert aps.net_to_gross_ratio_r0.property_kind == 'net to gross ratio'
    assert aps.net_to_gross_ratio_r0.facet_type is None
    assert aps.net_to_gross_ratio_r0.facet is None
    assert aps.net_to_gross_ratio_r0.indexable == 'cells'
    assert aps.net_to_gross_ratio_r0.indexable_element == 'cells'
    assert aps.net_to_gross_ratio_r0.is_continuous is True
    assert aps.net_to_gross_ratio_r0.is_discrete is False
    assert aps.net_to_gross_ratio_r0.is_categorical is False
    assert aps.net_to_gross_ratio_r0.is_points is False
    assert aps.net_to_gross_ratio_r0.count == 1
    assert aps.net_to_gross_ratio_r0.uom == 'm3/m3'
    assert np.isnan(aps.net_to_gross_ratio_r0.null_value)
    assert aps.net_to_gross_ratio_r0.realization == 0
    assert aps.net_to_gross_ratio_r1.realization == 1
    assert aps.net_to_gross_ratio_r0.time_index is None
    assert aps.net_to_gross_ratio_r0.title == 'NTG'
    assert aps.net_to_gross_ratio_r0.citation_title == 'NTG'
    assert maths.isclose(aps.net_to_gross_ratio_r0.min_value, 0.0)
    assert maths.isclose(aps.net_to_gross_ratio_r0.max_value, 0.5)
    assert aps.net_to_gross_ratio_r0.constant_value is None
    assert aps.net_to_gross_ratio_r0.extra == {}
    assert aps.net_to_gross_ratio_r0.extra_metadata == {}
    assert bu.matching_uuids(aps.net_to_gross_ratio_r0.support_uuid, grid.uuid)
    assert aps.net_to_gross_ratio_r0.string_lookup_uuid is None
    assert aps.net_to_gross_ratio_r0.time_series_uuid is None
    assert aps.net_to_gross_ratio_r0.local_property_kind_uuid is None
    v = aps.net_to_gross_ratio_r0.values
    assert isinstance(v, np.ndarray)
    assert v.shape == (3, 5, 5)
    assert np.isclose(np.min(v), 0.0)
    assert np.isclose(np.max(v), 0.5)
    assert np.all(v == aps.net_to_gross_ratio_r0.array_ref)
    # check a saturation property
    assert aps.saturation_water_t2.property_kind == 'saturation'
    assert aps.saturation_water_t2.facet_type == 'what'
    assert aps.saturation_water_t2.facet == 'water'
    assert aps.saturation_water_t2.indexable == 'cells'
    assert aps.saturation_water_t2.indexable_element == 'cells'
    assert aps.saturation_water_t2.is_continuous is True
    assert aps.saturation_water_t2.is_discrete is False
    assert aps.saturation_water_t2.is_categorical is False
    assert aps.saturation_water_t2.is_points is False
    assert aps.saturation_water_t2.count == 1
    assert aps.saturation_water_t2.uom == 'm3/m3'
    assert np.isnan(aps.saturation_water_t2.null_value)
    assert aps.saturation_water_t2.realization is None
    assert aps.saturation_water_t2.time_index == 2
    assert aps.saturation_water_t2.title == 'SW'
    assert aps.saturation_water_t2.citation_title == 'SW'
    assert maths.isclose(aps.saturation_water_t2.min_value, 0.75)
    assert maths.isclose(aps.saturation_water_t2.max_value, 1.0)
    assert aps.saturation_water_t2.constant_value is None
    assert aps.saturation_water_t2.extra == {}
    assert aps.saturation_water_t2.extra_metadata == {}
    assert bu.matching_uuids(aps.saturation_water_t2.support_uuid, grid.uuid)
    assert aps.saturation_water_t2.string_lookup_uuid is None
    assert aps.saturation_water_t2.time_series_uuid is not None
    assert aps.saturation_water_t2.local_property_kind_uuid is None
    v = aps.saturation_water_t2.values
    assert isinstance(v, np.ndarray)
    assert v.shape == (3, 5, 5)
    assert np.isclose(np.min(v), 0.75)
    assert np.isclose(np.max(v), 1.0)
    assert np.all(v == aps.saturation_water_t2.array_ref)


def test_load_attribute_property_collection_pk_duplicates(example_model_with_prop_ts_rels, capfd, caplog):
    # Arrange
    model = example_model_with_prop_ts_rels
    grid = model.grid()
    pc = grid.extract_property_collection()

    # Act
    aps = rqp.AttributePropertySet(support = grid)
    facies_col = np.max(aps.facies.values, axis = 0)
    pc.add_similar_to_imported_list(similar_uuid = aps.facies.uuid,
                                    cached_array = facies_col,
                                    indexable_element = 'columns')
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()

    # check indexable used when needed in key
    aps = rqp.AttributePropertySet(support = grid, indexable = 'cells')
    assert aps is not None
    assert len(aps) == 13
    assert 'facies' in aps.keys()
    assert 'facies_columns' in aps.keys()

    # check duplicate ignored
    aps = rqp.AttributePropertySet(support = grid, indexable = None, multiple_handling = 'ignore')
    assert aps is not None
    assert len(aps) == 13
    key_list = list(aps.keys())
    assert len(key_list) == 13  # all parts still exist in the PropertyCollection
    assert len(
        set(key_list)) == 12  # but two will have the same aps key (only the first is visible as AttributeProperty)
    assert 'facies' in aps.keys()
    assert 'facies_columns' not in aps.keys()

    # check that multiple handling 'exception' raises key error
    with pytest.raises(KeyError) as e_info:
        aps = rqp.AttributePropertySet(support = grid, multiple_handling = 'exception')

    # check that multiple handling 'warn' generates log warning
    aps = rqp.AttributePropertySet(support = grid, multiple_handling = 'warn')
    assert len(caplog.records) > 0
    assert caplog.records[-1].getMessage().endswith(
        "duplicate key in AttributePropertySet; only first instance included: facies")
    assert aps is not None
    assert len(aps) == 13
    key_list = list(aps.keys())
    assert len(key_list) == 13
    assert len(set(key_list)) == 12
    assert 'facies' in aps.keys()
    assert 'facies_columns' not in aps.keys()


def test_load_attribute_property_collection_title(example_model_with_prop_ts_rels):
    # Arrange
    model = example_model_with_prop_ts_rels
    grid = model.grid()

    # Act
    aps = rqp.AttributePropertySet(support = grid, key_mode = 'title')

    # Assert
    assert aps is not None
    for k, e in aps.items():
        assert k is not None
        assert isinstance(k, str)
        assert e is not None
        assert isinstance(e, rqp.ApsProperty)
    assert len(aps) == 12
    # check zone property
    assert aps.Zone.property_kind == 'zone'
    assert aps.Zone.facet_type is None
    assert aps.Zone.facet is None
    assert aps.Zone.indexable == 'cells'
    assert aps.Zone.indexable_element == 'cells'
    assert aps.Zone.is_continuous is False
    assert aps.Zone.is_discrete is True
    assert aps.Zone.is_categorical is False
    assert aps.Zone.is_points is False
    assert aps.Zone.count == 1
    assert aps.Zone.uom is None
    assert aps.Zone.null_value == -1
    assert aps.Zone.realization is None
    assert aps.Zone.time_index is None
    assert aps.Zone.title == 'Zone'
    assert aps.Zone.citation_title == 'Zone'
    assert aps.Zone.min_value == 1
    assert aps.Zone.max_value == 3
    assert aps.Zone.constant_value is None
    assert aps.Zone.extra == {}
    assert aps.Zone.extra_metadata == {}
    assert bu.matching_uuids(aps.Zone.support_uuid, grid.uuid)
    assert aps.Zone.string_lookup_uuid is None
    assert aps.Zone.time_series_uuid is None
    assert aps.Zone.local_property_kind_uuid is not None
    v = aps.Zone.values
    assert isinstance(v, np.ndarray)
    assert v.shape == (3, 5, 5)
    assert tuple(np.unique(v)) == (1, 2, 3)
    assert np.all(v == aps.Zone.array_ref)
    # check net to gross ratio properties
    assert aps.NTG_r0.property_kind == 'net to gross ratio'
    assert aps.NTG_r0.facet_type is None
    assert aps.NTG_r0.facet is None
    assert aps.NTG_r0.indexable == 'cells'
    assert aps.NTG_r0.indexable_element == 'cells'
    assert aps.NTG_r0.is_continuous is True
    assert aps.NTG_r0.is_discrete is False
    assert aps.NTG_r0.is_categorical is False
    assert aps.NTG_r0.is_points is False
    assert aps.NTG_r0.count == 1
    assert aps.NTG_r0.uom == 'm3/m3'
    assert np.isnan(aps.NTG_r0.null_value)
    assert aps.NTG_r0.realization == 0
    assert aps.NTG_r1.realization == 1
    assert aps.NTG_r0.time_index is None
    assert aps.NTG_r0.title == 'NTG'
    assert aps.NTG_r0.citation_title == 'NTG'
    assert maths.isclose(aps.NTG_r0.min_value, 0.0)
    assert maths.isclose(aps.NTG_r0.max_value, 0.5)
    assert aps.NTG_r0.constant_value is None
    assert aps.NTG_r0.extra == {}
    assert aps.NTG_r0.extra_metadata == {}
    assert bu.matching_uuids(aps.NTG_r0.support_uuid, grid.uuid)
    assert aps.NTG_r0.string_lookup_uuid is None
    assert aps.NTG_r0.time_series_uuid is None
    assert aps.NTG_r0.local_property_kind_uuid is None
    v = aps.NTG_r0.values
    assert isinstance(v, np.ndarray)
    assert v.shape == (3, 5, 5)
    assert np.isclose(np.min(v), 0.0)
    assert np.isclose(np.max(v), 0.5)
    assert np.all(v == aps.NTG_r0.array_ref)
    # check a saturation property
    assert aps.SW_t2.property_kind == 'saturation'
    assert aps.SW_t2.facet_type == 'what'
    assert aps.SW_t2.facet == 'water'
    assert aps.SW_t2.indexable == 'cells'
    assert aps.SW_t2.indexable_element == 'cells'
    assert aps.SW_t2.is_continuous is True
    assert aps.SW_t2.is_discrete is False
    assert aps.SW_t2.is_categorical is False
    assert aps.SW_t2.is_points is False
    assert aps.SW_t2.count == 1
    assert aps.SW_t2.uom == 'm3/m3'
    assert np.isnan(aps.SW_t2.null_value)
    assert aps.SW_t2.realization is None
    assert aps.SW_t2.time_index == 2
    assert aps.SW_t2.title == 'SW'
    assert aps.SW_t2.citation_title == 'SW'
    assert maths.isclose(aps.SW_t2.min_value, 0.75)
    assert maths.isclose(aps.SW_t2.max_value, 1.0)
    assert aps.SW_t2.constant_value is None
    assert aps.SW_t2.extra == {}
    assert aps.SW_t2.extra_metadata == {}
    assert bu.matching_uuids(aps.SW_t2.support_uuid, grid.uuid)
    assert aps.SW_t2.string_lookup_uuid is None
    assert aps.SW_t2.time_series_uuid is not None
    assert aps.SW_t2.local_property_kind_uuid is None
    v = aps.SW_t2.values
    assert isinstance(v, np.ndarray)
    assert v.shape == (3, 5, 5)
    assert np.isclose(np.min(v), 0.75)
    assert np.isclose(np.max(v), 1.0)
    assert np.all(v == aps.SW_t2.array_ref)
