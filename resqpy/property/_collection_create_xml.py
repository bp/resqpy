"""Submodule containing functions for creating xml for a property collection."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import math as maths
import numpy as np

import resqpy.property
import resqpy.time_series as rts
import resqpy.weights_and_measures as wam
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.property._collection_get_attributes as pcga
import resqpy.property.property_common as rqp_c
from resqpy.olio.xml_namespaces import curly_namespace as ns


def _create_xml_add_as_part(collection, add_as_part, p_uuid, p_node, add_relationships, support_root,
                            property_kind_uuid, related_time_series_node, sl_root, discrete, string_lookup_uuid,
                            const_value, ext_uuid):
    if add_as_part:
        collection.model.add_part('obj_' + collection.d_or_c_text + 'Property', p_uuid, p_node)
        if add_relationships:
            _create_xml_add_relationships(collection, p_node, support_root, property_kind_uuid,
                                          related_time_series_node, sl_root, discrete, string_lookup_uuid, const_value,
                                          ext_uuid)


def _create_property_set_xml_add_as_part(collection, ps_node, ps_uuid, add_relationships, parent_set_ref_node,
                                         prop_node_list):
    collection.model.add_part('obj_PropertySet', ps_uuid, ps_node)
    if add_relationships:
        # todo: add relationship with time series if time set kind is not 'not a time set'?
        if collection.parent_set_root is not None:
            collection.model.create_reciprocal_relationship(ps_node, 'destinationObject', parent_set_ref_node,
                                                            'sourceObject')
        for prop_node in prop_node_list:
            collection.model.create_reciprocal_relationship(ps_node, 'destinationObject', prop_node, 'sourceObject')


def _create_xml_get_basics(collection, discrete, points, const_value, facet_type, null_value, support_uuid, ext_uuid):
    assert not discrete or not points
    assert not points or const_value is None
    assert not points or facet_type is None
    assert collection.model is not None

    if null_value is not None:
        collection.null_value = null_value

    if support_uuid is None:
        support_uuid = collection.support_uuid
    assert support_uuid is not None
    support_root = collection.model.root_for_uuid(support_uuid)
    # assert support_root is not None

    if ext_uuid is None:
        ext_uuid = collection.model.h5_uuid()

    return support_root, support_uuid, ext_uuid


def _create_xml_property_kind(collection, p_node, find_local_property_kinds, property_kind, uom, discrete,
                              property_kind_uuid):
    if property_kind == 'permeability rock':
        property_kind = 'rock permeability'
    p_kind_node = rqet.SubElement(p_node, ns['resqml2'] + 'PropertyKind')
    p_kind_node.text = rqet.null_xml_text
    if find_local_property_kinds and property_kind not in rqp_c.supported_property_kind_list:
        property_kind_uuid = pcga._get_property_kind_uuid(collection, property_kind_uuid, property_kind, uom, discrete)

    if property_kind_uuid is None:
        p_kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'StandardPropertyKind')  # todo: local prop kind ref
        kind_node = rqet.SubElement(p_kind_node, ns['resqml2'] + 'Kind')
        kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ResqmlPropertyKind')
        kind_node.text = property_kind
    else:
        p_kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'LocalPropertyKind')  # todo: local prop kind ref
        collection.model.create_ref_node('LocalPropertyKind',
                                         property_kind,
                                         property_kind_uuid,
                                         content_type = 'obj_PropertyKind',
                                         root = p_kind_node)
    return property_kind_uuid


def _create_xml_patch_node(collection, p_node, points, const_value, indexable_element, direction, p_uuid, ext_uuid,
                           expand_const_arrays):
    # create patch node
    const_count = None
    if const_value is not None and not expand_const_arrays:
        s_shape = collection.supporting_shape(indexable_element = indexable_element, direction = direction)
        assert s_shape is not None
        const_count = np.product(np.array(s_shape, dtype = int))
    else:
        const_value = None
    _ = collection.model.create_patch(p_uuid,
                                      ext_uuid,
                                      root = p_node,
                                      hdf5_type = collection.hdf5_type,
                                      xsd_type = collection.xsd_type,
                                      null_value = collection.null_value,
                                      const_value = const_value,
                                      const_count = const_count,
                                      points = points)


def _create_xml_property_min_max(collection, property_array, const_value, discrete, add_min_max, p_node, min_value,
                                 max_value, categorical, null_value, points):
    if add_min_max and not categorical and not points:
        # todo: use active cell mask on numpy min and max operations; exclude null values on discrete min max
        min_value, max_value = pcga._get_property_array_min_max_value(collection, property_array, const_value, discrete,
                                                                      min_value, max_value, categorical, null_value)
        if min_value is not None:
            min_node = rqet.SubElement(p_node, ns['resqml2'] + 'MinimumValue')
            min_node.set(ns['xsi'] + 'type', ns['xsd'] + collection.xsd_type)
            if discrete:
                min_node.text = str(maths.floor(min_value))
            else:
                min_node.text = str(min_value)
        if max_value is not None:
            max_node = rqet.SubElement(p_node, ns['resqml2'] + 'MaximumValue')
            max_node.set(ns['xsi'] + 'type', ns['xsd'] + collection.xsd_type)
            if discrete:
                max_node.text = str(maths.ceil(max_value))
            else:
                max_node.text = str(max_value)


def _create_xml_lookup_node(collection, p_node, string_lookup_uuid):
    sl_root = None
    if string_lookup_uuid is not None:
        sl_root = collection.model.root_for_uuid(string_lookup_uuid)
        assert sl_root is not None, 'string table lookup is missing whilst importing categorical property'
        assert rqet.node_type(sl_root) == 'obj_StringTableLookup', 'referenced uuid is not for string table lookup'
        collection.model.create_ref_node('Lookup',
                                         collection.model.title_for_root(sl_root),
                                         string_lookup_uuid,
                                         content_type = 'obj_StringTableLookup',
                                         root = p_node)
    return sl_root


def _create_xml_uom_node(collection, p_node, uom, property_kind, min_value, max_value, facet_type, facet, title,
                         points):
    if points:
        return
    if not uom:
        uom = rqp_c.guess_uom(property_kind,
                              min_value,
                              max_value,
                              collection.support,
                              facet_type = facet_type,
                              facet = facet)
        if not uom:
            uom = 'Euc'  # todo: put RESQML base uom for quantity class here, instead of Euc
            log.warning(f'uom set to Euc for property {title} of kind {property_kind}')
    if uom in wam.valid_uoms():
        collection.model.uom_node(p_node, uom)
    else:
        collection.model.uom_node(p_node, 'Euc')
        rqet.create_metadata_xml(p_node, {'uom': uom})


def _create_xml_add_relationships(collection, p_node, support_root, property_kind_uuid, related_time_series_node,
                                  sl_root, discrete, string_lookup_uuid, const_value, ext_uuid):
    if support_root is not None:
        collection.model.create_reciprocal_relationship(p_node, 'destinationObject', support_root, 'sourceObject')
    if property_kind_uuid is not None:
        pk_node = collection.model.root_for_uuid(property_kind_uuid)
        if pk_node is not None:
            collection.model.create_reciprocal_relationship(p_node, 'destinationObject', pk_node, 'sourceObject')
    if related_time_series_node is not None:
        collection.model.create_reciprocal_relationship(p_node, 'destinationObject', related_time_series_node,
                                                        'sourceObject')
    if discrete and string_lookup_uuid is not None:
        collection.model.create_reciprocal_relationship(p_node, 'destinationObject', sl_root, 'sourceObject')

    if const_value is None:
        ext_node = collection.model.root_for_part(
            rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False))
        collection.model.create_reciprocal_relationship(p_node, 'mlToExternalPartProxy', ext_node,
                                                        'externalPartProxyToMl')


def _create_xml_add_basics_to_p_node(collection, p_node, title, originator, extra_metadata, source, count,
                                     indexable_element):
    collection.model.create_citation(root = p_node, title = title, originator = originator)
    rqet.create_metadata_xml(node = p_node, extra_metadata = extra_metadata)

    if source is not None and len(source) > 0:
        collection.model.create_source(source = source, root = p_node)

    count_node = rqet.SubElement(p_node, ns['resqml2'] + 'Count')
    count_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
    count_node.text = str(count)

    ie_node = rqet.SubElement(p_node, ns['resqml2'] + 'IndexableElement')
    ie_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IndexableElements')
    ie_node.text = indexable_element


def _create_xml_realization_node(realization, p_node):
    if realization is not None and realization >= 0:
        ri_node = rqet.SubElement(p_node, ns['resqml2'] + 'RealizationIndex')
        ri_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
        ri_node.text = str(realization)


def _create_xml_time_series_node(collection, time_series_uuid, time_index, p_node, support_uuid, support_type,
                                 support_root):
    if time_series_uuid is None or time_index is None:
        related_time_series_node = None
    else:
        related_time_series_node = collection.model.root(uuid = time_series_uuid)
        time_series = rts.any_time_series(collection.model, uuid = time_series_uuid)
        time_series.create_time_index(time_index, root = p_node)

    support_title = '' if support_root is None else rqet.citation_title_for_node(support_root)
    collection.model.create_supporting_representation(support_uuid = support_uuid,
                                                      root = p_node,
                                                      title = support_title,
                                                      content_type = support_type)
    return related_time_series_node


def _create_xml_get_p_node(collection, p_uuid):
    p_node = collection.model.new_obj_node(collection.d_or_c_text + 'Property')
    if p_uuid is None:
        p_uuid = bu.uuid_from_string(p_node.attrib['uuid'])
    else:
        p_node.attrib['uuid'] = str(p_uuid)
    return p_node, p_uuid


def _create_xml_facet_node(facet_type, facet, p_node):
    if facet_type is not None and facet is not None:
        facet_node = rqet.SubElement(p_node, ns['resqml2'] + 'Facet')
        facet_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'PropertyKindFacet')
        facet_node.text = rqet.null_xml_text
        facet_type_node = rqet.SubElement(facet_node, ns['resqml2'] + 'Facet')
        facet_type_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Facet')
        facet_type_node.text = facet_type
        facet_value_node = rqet.SubElement(facet_node, ns['resqml2'] + 'Value')
        facet_value_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
        facet_value_node.text = facet


def _check_shape_list(collection, indexable_element, direction, property_array, points, count):
    shape_list = collection.supporting_shape(indexable_element = indexable_element, direction = direction)
    if shape_list is not None:
        if count > 1:
            shape_list.append(count)
        if points:
            shape_list.append(3)
        if property_array is not None:
            assert tuple(shape_list) == property_array.shape, \
                f'property array shape {property_array.shape} is not the expected {tuple(shape_list)}'
