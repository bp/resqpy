"""Submodule containing functions for adding properties to a property collection."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import numpy as np

import resqpy.property
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.property.property_common as rqp_c
import resqpy.property._collection_get_attributes as pcga


def _add_selected_part_from_other_dict(collection, part, other, realization, support_uuid, uuid, continuous,
                                       categorical, count, points, indexable, property_kind, facet_type, facet,
                                       citation_title, citation_title_match_mode, time_series_uuid, time_index,
                                       string_lookup_uuid, related_uuid, const_value, ignore_clashes):
    if _check_not_none_and_not_equals(realization, other.realization_for_part, part):
        return
    if _check_not_none_and_not_uuid_match(support_uuid, other.support_uuid_for_part, part):
        return
    if _check_not_none_and_not_uuid_match(uuid, other.uuid_for_part, part):
        return
    if _check_not_none_and_not_equals(continuous, other.continuous_for_part, part):
        return
    if _check_categorical_and_lookup(categorical, other, part):
        return
    if _check_not_none_and_not_equals(count, other.count_for_part, part):
        return
    if _check_not_none_and_not_equals(points, other.points_for_part, part):
        return
    if _check_not_none_and_not_equals(indexable, other.indexable_for_part, part):
        return
    if property_kind is not None and not rqp_c.same_property_kind(other.property_kind_for_part(part), property_kind):
        return
    if _check_not_none_and_not_equals(facet_type, other.facet_type_for_part, part):
        return
    if _check_not_none_and_not_equals(facet, other.facet_for_part, part):
        return
    if _check_citation_title(citation_title, citation_title_match_mode, other, part):
        return
    if _check_not_none_and_not_uuid_match(time_series_uuid, other.time_series_uuid_for_part, part):
        return
    if _check_not_none_and_not_equals(time_index, other.time_index_for_part, part):
        return
    if _check_not_none_and_not_uuid_match(string_lookup_uuid, other.string_lookup_uuid_for_part, part):
        return
    if _check_not_none_and_not_equals(const_value, other.constant_value_for_part, part):
        return
    if related_uuid is not None:
        assert other.model is not None
        if other.model.part(parts_list = [part], related_uuid = related_uuid) is None:
            return
    if part in collection.dict.keys():
        if ignore_clashes:
            return
        assert (False)
    collection.dict[part] = other.dict[part]


def _add_part_to_dict_get_realization(collection, realization, xml_node):
    if realization is not None and collection.realization is not None:
        assert (realization == collection.realization)
    if realization is None:
        realization = collection.realization
    realization_node = rqet.find_tag(xml_node, 'RealizationIndex')  # optional; if present use to populate realization
    if realization_node is not None:
        realization = int(realization_node.text)
    return realization


def _add_part_to_dict_get_type_details(collection, part, continuous, xml_node):
    sl_ref_node = None
    type = collection.model.type_of_part(part)
    #      log.debug('adding part ' + part + ' of type ' + type)
    assert type in ['obj_ContinuousProperty', 'obj_DiscreteProperty', 'obj_CategoricalProperty', 'obj_PointsProperty']
    if continuous is None:
        continuous = (type in ['obj_ContinuousProperty', 'obj_PointsProperty'])
    else:
        assert continuous == (type in ['obj_ContinuousProperty', 'obj_PointsProperty'])
    points = (type == 'obj_PointsProperty')
    string_lookup_uuid = None
    if type == 'obj_CategoricalProperty':
        sl_ref_node = rqet.find_tag(xml_node, 'Lookup')
        string_lookup_uuid = bu.uuid_from_string(rqet.find_tag_text(sl_ref_node, 'UUID'))

    return type, continuous, points, string_lookup_uuid, sl_ref_node


def _add_part_to_dict_get_support_uuid(collection, part):
    support_uuid = collection.model.supporting_representation_for_part(part)
    if support_uuid is None:
        support_uuid = collection.support_uuid
    elif collection.support_uuid is None:
        collection.set_support(support_uuid)
    elif not bu.matching_uuids(support_uuid, collection.support.uuid):  # multi-support collection
        collection.set_support(None)
    if isinstance(support_uuid, str):
        support_uuid = bu.uuid_from_string(support_uuid)
    return support_uuid


def _add_part_to_dict_get_uom(collection, part, continuous, xml_node, trust_uom, property_kind, minimum, maximum, facet,
                              facet_type, points):
    uom = None
    if continuous and not points:
        uom_node = rqet.find_tag(xml_node, 'UOM')
        if uom_node is not None and (trust_uom or uom_node.text not in ['', 'Euc']):
            uom = uom_node.text
        else:
            if not collection.guess_warning:
                collection.guess_warning = True
                log.warning("Guessing unit of measure for one or more properties")
            uom = rqp_c.guess_uom(property_kind,
                                  minimum,
                                  maximum,
                                  collection.support,
                                  facet_type = facet_type,
                                  facet = facet)
    return uom


def _check_not_none_and_not_equals(attrib, method, part):
    if attrib is None:
        return False
    if attrib == '*':
        return method(part) is None
    elif attrib == 'none':
        return method(part) is not None
    return method(part) != attrib


def _check_not_none_and_not_uuid_match(uuid, method, part):
    return uuid is not None and not bu.matching_uuids(uuid, method(part))


def _check_citation_title(citation_title, citation_title_match_mode, other, part):
    if citation_title is not None:
        if isinstance(citation_title_match_mode, bool):
            citation_title_match_mode = 'starts' if citation_title_match_mode else None
        if citation_title_match_mode is None or citation_title_match_mode == 'is':
            return other.citation_title_for_part(part) != citation_title
        elif citation_title_match_mode == 'starts':
            return not other.citation_title_for_part(part).startswith(citation_title)
        elif citation_title_match_mode == 'ends':
            return not other.citation_title_for_part(part).endswith(citation_title)
        elif citation_title_match_mode == 'contains':
            return citation_title not in other.citation_title_for_part(part)
        elif citation_title_match_mode == 'is not':
            return other.citation_title_for_part(part) == citation_title
        elif citation_title_match_mode == 'does not start':
            return other.citation_title_for_part(part).startswith(citation_title)
        elif citation_title_match_mode == 'does not end':
            return other.citation_title_for_part(part).endswith(citation_title)
        elif citation_title_match_mode == 'does not contain':
            return citation_title in other.citation_title_for_part(part)
        else:
            raise ValueError(f'invalid title mode {citation_title_match_mode} in property filtering')
    return False


def _check_categorical_and_lookup(categorical, other, part):
    if categorical is not None:
        if categorical:
            if other.continuous_for_part(part) or (other.string_lookup_uuid_for_part(part) is None):
                return True
        else:
            if not (other.continuous_for_part(part) or (other.string_lookup_uuid_for_part(part) is None)):
                return True
    return False


def _process_imported_property(collection, attributes, property_kind_uuid, string_lookup_uuid, time_series_uuid,
                               ext_uuid, support_uuid, selected_time_indices_list, find_local_property_kinds,
                               extra_metadata, expand_const_arrays):
    (p_uuid, p_file_name, p_keyword, p_cached_name, p_discrete, p_uom, p_time_index, p_null_value, p_min_value,
     p_max_value, property_kind, facet_type, facet, realization, indexable_element, count, local_property_kind_uuid,
     const_value, points, p_time_series_uuid) = attributes

    log.debug('processing imported property ' + str(p_keyword))
    assert not points or not p_discrete
    if local_property_kind_uuid is None:
        local_property_kind_uuid = property_kind_uuid
    if not p_discrete:
        string_lookup_uuid = None

    property_kind = _process_imported_property_get_property_kind(collection, property_kind, local_property_kind_uuid,
                                                                 p_keyword, p_discrete, string_lookup_uuid, points)

    p_array = _process_imported_property_get_p_array(collection, p_cached_name)
    p_array_bool = None if p_array is None else p_array.dtype in [bool, np.int8]

    add_min_max = pcga._process_imported_property_get_add_min_max(points, property_kind, string_lookup_uuid,
                                                                  local_property_kind_uuid, p_array_bool)

    if selected_time_indices_list is not None and p_time_index is not None:
        p_time_index = selected_time_indices_list.index(p_time_index)
    p_node = collection.create_xml(
        ext_uuid = ext_uuid,
        property_array = p_array,
        title = p_keyword,
        property_kind = property_kind,
        support_uuid = support_uuid,
        p_uuid = p_uuid,
        facet_type = facet_type,
        facet = facet,
        discrete = p_discrete,  # todo: time series bits
        time_series_uuid = time_series_uuid if p_time_series_uuid is None else p_time_series_uuid,
        time_index = p_time_index,
        uom = p_uom,
        null_value = p_null_value,
        originator = None,
        source = p_file_name,
        add_as_part = True,
        add_relationships = True,
        add_min_max = add_min_max,
        min_value = p_min_value,
        max_value = p_max_value,
        realization = realization,
        string_lookup_uuid = string_lookup_uuid,
        property_kind_uuid = local_property_kind_uuid,
        indexable_element = indexable_element,
        count = count,
        points = points,
        find_local_property_kinds = find_local_property_kinds,
        extra_metadata = extra_metadata,
        const_value = const_value,
        expand_const_arrays = expand_const_arrays)
    if p_node is not None:
        return p_node
    else:
        return None


def _process_imported_property_get_property_kind(collection, property_kind, local_property_kind_uuid, p_keyword,
                                                 p_discrete, string_lookup_uuid, points):
    if property_kind is None:
        if local_property_kind_uuid is not None:
            # note: requires local property kind to be present
            property_kind = collection.model.title(uuid = local_property_kind_uuid)
        else:
            # todo: only if None in ab_property_list
            (property_kind, facet_type, facet) = rqp_c.property_kind_and_facet_from_keyword(p_keyword)
    if property_kind is None:
        # todo: the following are abstract standard property kinds, which shouldn't really have data directly associated with them
        if p_discrete:
            if string_lookup_uuid is not None:
                property_kind = 'categorical'
            else:
                property_kind = 'discrete'
        elif points:
            property_kind = 'length'
        else:
            property_kind = 'continuous'
    return property_kind


def _process_imported_property_get_p_array(collection, p_cached_name):
    if hasattr(collection, p_cached_name):
        return collection.__dict__[p_cached_name]
    else:
        return None


def _add_part_to_dict_get_count_and_indexable(xml_node):
    count_node = rqet.find_tag(xml_node, 'Count')
    assert count_node is not None
    count = int(count_node.text)

    indexable_node = rqet.find_tag(xml_node, 'IndexableElement')
    assert indexable_node is not None
    indexable = indexable_node.text

    return count, indexable


def _add_part_to_dict_get_property_kind(xml_node, citation_title):
    perm_synonyms = ['permeability rock', 'rock permeability']
    (p_kind_from_keyword, facet_type, facet) = rqp_c.property_kind_and_facet_from_keyword(citation_title)
    prop_kind_node = rqet.find_tag(xml_node, 'PropertyKind')
    assert (prop_kind_node is not None)
    kind_node = rqet.find_tag(prop_kind_node, 'Kind')
    property_kind_uuid = None  # only used for bespoke (local) property kinds
    if kind_node is not None:
        property_kind = kind_node.text  # could check for consistency with that derived from citation title
        lpk_node = None
    else:
        lpk_node = rqet.find_tag(prop_kind_node, 'LocalPropertyKind')
        if lpk_node is not None:
            property_kind = rqet.find_tag_text(lpk_node, 'Title')
            property_kind_uuid = rqet.find_tag_text(lpk_node, 'UUID')
    assert property_kind is not None and len(property_kind) > 0
    if (p_kind_from_keyword and p_kind_from_keyword != property_kind and
        (p_kind_from_keyword not in ['cell length', 'length', 'thickness'] or
         property_kind not in ['cell length', 'length', 'thickness'])):
        if property_kind not in perm_synonyms or p_kind_from_keyword not in perm_synonyms:
            log.warning(
                f'property kind {property_kind} not the expected {p_kind_from_keyword} for keyword {citation_title}')
    return property_kind, property_kind_uuid, lpk_node


def _add_part_to_dict_get_facet(xml_node):
    facet_type = None
    facet = None
    facet_node = rqet.find_tag(xml_node, 'Facet')  # todo: handle more than one facet for a property
    if facet_node is not None:
        facet_type = rqet.find_tag(facet_node, 'Facet').text
        facet = rqet.find_tag(facet_node, 'Value').text
        if facet_type is not None and facet_type == '':
            facet_type = None
        if facet is not None and facet == '':
            facet = None
    return facet_type, facet


def _add_part_to_dict_get_timeseries(xml_node):
    time_series_uuid = None
    time_index = None
    time_node = rqet.find_tag(xml_node, 'TimeIndex')
    if time_node is not None:
        time_index = int(rqet.find_tag(time_node, 'Index').text)
        time_series_uuid = bu.uuid_from_string(rqet.find_tag(rqet.find_tag(time_node, 'TimeSeries'), 'UUID').text)

    return time_series_uuid, time_index


def _add_part_to_dict_get_minmax(xml_node):
    minimum = None
    min_node = rqet.find_tag(xml_node, 'MinimumValue')
    if min_node is not None:
        minimum = min_node.text  # NB: left as text
    maximum = None
    max_node = rqet.find_tag(xml_node, 'MaximumValue')
    if max_node is not None:
        maximum = max_node.text  # NB: left as text

    return minimum, maximum


def _add_part_to_dict_get_null_constvalue_points(xml_node, continuous, points):
    null_value = None
    if not continuous:
        null_value = rqet.find_nested_tags_int(xml_node, ['PatchOfValues', 'Values', 'NullValue'])
    const_value = None
    if points:
        values_node = rqet.find_nested_tags(xml_node, ['PatchOfPoints', 'Points'])
    else:
        values_node = rqet.find_nested_tags(xml_node, ['PatchOfValues', 'Values'])
    values_type = rqet.node_type(values_node)
    assert values_type is not None
    if values_type.endswith('ConstantArray'):
        if continuous:
            const_value = rqet.find_tag_float(values_node, 'Value')
        elif values_type.startswith('Bool'):
            const_value = rqet.find_tag_bool(values_node, 'Value')
        else:
            const_value = rqet.find_tag_int(values_node, 'Value')

    return null_value, const_value
