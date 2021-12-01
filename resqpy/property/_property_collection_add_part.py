"""_property_collection_add_part.py: submodule containing functions for adding properties to a property collection."""

version = '1st December 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)
log.debug('_property_collection_add_part.py version ' + version)

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet

from .property_common import same_property_kind, guess_uom, property_kind_and_facet_from_keyword
import resqpy.property._property_collection_get_attributes as pcga


def _add_selected_part_from_other_dict(collection, part, other, realization, support_uuid, uuid, continuous,
                                       categorical, count, points, indexable, property_kind, facet_type, facet,
                                       citation_title, citation_title_match_starts_with, time_series_uuid, time_index,
                                       string_lookup_uuid, ignore_clashes):
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
    if property_kind is not None and not same_property_kind(other.property_kind_for_part(part), property_kind):
        return
    if _check_not_none_and_not_equals(facet_type, other.facet_type_for_part, part):
        return
    if _check_not_none_and_not_equals(facet, other.facet_for_part, part):
        return
    if _check_citation_title(citation_title, citation_title_match_starts_with, other, part):
        return
    if _check_not_none_and_not_uuid_match(time_series_uuid, other.time_series_uuid_for_part, part):
        return
    if _check_not_none_and_not_equals(time_index, other.time_index_for_part, part):
        return
    if _check_not_none_and_not_uuid_match(string_lookup_uuid, other.string_lookup_uuid_for_part, part):
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
                              facet_type):
    uom = None
    if continuous:
        uom_node = rqet.find_tag(xml_node, 'UOM')
        if uom_node is not None and (trust_uom or uom_node.text not in ['', 'Euc']):
            uom = uom_node.text
        else:
            uom = guess_uom(property_kind, minimum, maximum, collection.support, facet_type = facet_type, facet = facet)
    return uom


def _check_not_none_and_not_equals(attrib, method, part):
    return attrib is not None and method(part) != attrib


def _check_not_none_and_not_uuid_match(uuid, method, part):
    return uuid is not None and not bu.matching_uuids(uuid, method(part))


def _check_citation_title(citation_title, citation_title_match_starts_with, other, part):
    if citation_title is not None:
        if citation_title_match_starts_with:
            if not other.citation_title_for_part(part).startswith():
                return True
        else:
            if other.citation_title_for_part(part) != citation_title:
                return True
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
     const_value, points) = attributes

    log.debug('processing imported property ' + str(p_keyword))
    assert not points or not p_discrete
    if local_property_kind_uuid is None:
        local_property_kind_uuid = property_kind_uuid

    property_kind = _process_imported_property_get_property_kind(collection, property_kind, local_property_kind_uuid,
                                                                 p_keyword, p_discrete, string_lookup_uuid, points)

    p_array = _process_imported_property_get_p_array(collection, p_cached_name)

    add_min_max = pcga._process_imported_property_get_add_min_max(points, property_kind, string_lookup_uuid,
                                                                  local_property_kind_uuid)

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
        time_series_uuid = time_series_uuid,
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
            (property_kind, facet_type, facet) = property_kind_and_facet_from_keyword(p_keyword)
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
