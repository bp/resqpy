"""_property_collection_get_attributes.py: submodule containing functions for attribute extraction for a property collection."""

version = '30th November 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)
log.debug('_property_collection_get_attributes.py version ' + version)

import numpy as np
import numpy.ma as ma

from .property_kind import PropertyKind


def _min_max_of_cached_array(collection, cached_name, cached_array, null_value, discrete):
    collection.__dict__[cached_name] = cached_array
    zorro = collection.masked_array(cached_array, exclude_value = null_value)
    if not discrete and np.all(np.isnan(zorro)):
        min_value = max_value = None
    elif discrete:
        min_value = int(np.nanmin(zorro))
        max_value = int(np.nanmax(zorro))
    else:
        min_value = np.nanmin(zorro)
        max_value = np.nanmax(zorro)
    if min_value is ma.masked or min_value == np.NaN:
        min_value = None
    if max_value is ma.masked or max_value == np.NaN:
        max_value = None
    return min_value, max_value


def _get_property_kind_uuid(collection, property_kind_uuid, property_kind, uom, discrete):
    if property_kind_uuid is None:
        pk_parts_list = collection.model.parts_list_of_type('PropertyKind')
        for part in pk_parts_list:
            if collection.model.citation_title_for_part(part) == property_kind:
                property_kind_uuid = collection.model.uuid_for_part(part)
                break
        if property_kind_uuid is None:
            # create local property kind object and fetch uuid
            lpk = PropertyKind(collection.model,
                               title = property_kind,
                               example_uom = uom,
                               parent_property_kind = 'discrete' if discrete else 'continuous')
            lpk.create_xml()
            property_kind_uuid = lpk.uuid
    return property_kind_uuid


def _get_property_type_details(collection, discrete, string_lookup_uuid, points):
    if discrete:
        if string_lookup_uuid is None:
            collection.d_or_c_text = 'Discrete'

        else:
            collection.d_or_c_text = 'Categorical'
        collection.xsd_type = 'integer'
        collection.hdf5_type = 'IntegerHdf5Array'
    elif points:
        collection.d_or_c_text = 'Points'
        collection.xsd_type = 'double'
        collection.hdf5_type = 'Point3dHdf5Array'
        collection.null_value = None
    else:
        collection.d_or_c_text = 'Continuous'
        collection.xsd_type = 'double'
        collection.hdf5_type = 'DoubleHdf5Array'
        collection.null_value = None


def _get_property_array_min_max_value(collection, property_array, const_value, discrete, min_value, max_value):
    if const_value is not None:
        return _get_property_array_min_max_const(const_value, collection.null_value, min_value, max_value, discrete)
    elif property_array is not None:
        return _get_property_array_min_max_array(property_array, min_value, max_value, discrete)


def _get_property_array_min_max_const(const_value, null_value, min_value, max_value, discrete):
    if (discrete and const_value != null_value) or (not discrete and not np.isnan(const_value)):
        if min_value is None:
            min_value = const_value
        if max_value is None:
            max_value = const_value
    return min_value, max_value


def _get_property_array_min_max_array(property_array, min_value, max_value, discrete):
    if discrete:
        min_value, max_value = _get_property_array_min_max_array_discrete(property_array, min_value, max_value)
    else:
        min_value, max_value = _get_property_array_min_max_array_continuous(property_array, min_value, max_value)

    return min_value, max_value


def _get_property_array_min_max_array_discrete(property_array, min_value, max_value):
    if min_value is None:
        try:
            min_value = int(property_array.min())
        except Exception:
            min_value = None
            log.warning('no xml minimum value set for discrete property')
    if max_value is None:
        try:
            max_value = int(property_array.max())
        except Exception:
            max_value = None
            log.warning('no xml maximum value set for discrete property')
    return min_value, max_value


def _get_property_array_min_max_array_continuous(property_array, min_value, max_value):
    if min_value is None or max_value is None:
        all_nan = np.all(np.isnan(property_array))
    if min_value is None and not all_nan:
        min_value = np.nanmin(property_array)
        if np.isnan(min_value) or min_value is ma.masked:
            min_value = None
    if max_value is None and not all_nan:
        max_value = np.nanmax(property_array)
        if np.isnan(max_value) or max_value is ma.masked:
            max_value = None
    return min_value, max_value


def _find_single_part(collection, kind, realization):
    try:
        part = collection.singleton(realization = realization, property_kind = kind)
    except Exception:
        log.error(f'problem with {kind} (more than one array present?)')
        part = None
    return part


def _get_single_perm_ijk_parts(collection, perms, share_perm_parts, perm_k_mode, perm_k_ratio, ntg_part):
    if perms.number_of_parts() == 1:
        return _get_single_perm_ijk_parts_one(perms, share_perm_parts)
    else:
        perm_i_part = _get_single_perm_ijk_for_direction(perms, 'I')
        perm_j_part = _get_single_perm_ijk_for_direction(perms, 'J')
        if perm_j_part is None and share_perm_parts:
            perm_j_part = perm_i_part
        elif perm_i_part is None and share_perm_parts:
            perm_i_part = perm_j_part
        perm_k_part = _get_single_perm_ijk_for_direction(perms, 'K')
        if perm_k_part is None:
            assert perm_k_mode in [None, 'none', 'shared', 'ratio', 'ntg', 'ntg squared']
            # note: could switch ratio mode to shared if perm_k_ratio is 1.0
            if perm_k_mode is None or perm_k_mode == 'none':
                pass
            else:
                perm_k_part = collection._get_perm_k_part(perms, perm_k_mode, share_perm_parts, perm_i_part,
                                                          perm_j_part, perm_k_ratio, ntg_part)
    return perm_i_part, perm_j_part, perm_k_part


def _get_perm_k_part(collection, perms, perm_k_mode, share_perm_parts, perm_i_part, perm_j_part, perm_k_ratio,
                     ntg_part):
    if perm_k_mode == 'shared':
        if share_perm_parts:
            perm_k_part = perm_i_part
    elif perm_i_part is not None:
        log.info('generating K permeability data using mode ' + str(perm_k_mode))
        if perm_j_part is not None and perm_j_part != perm_i_part:
            # generate root mean square of I & J permeabilities to use as horizontal perm
            kh = np.sqrt(perms.cached_part_array_ref(perm_i_part) * perms.cached_part_array_ref(perm_j_part))
        else:  # use I permeability as horizontal perm
            kh = perms.cached_part_array_ref(perm_i_part)
        kv = kh * perm_k_ratio
        if ntg_part is not None:
            if perm_k_mode == 'ntg':
                kv *= collection.cached_part_array_ref(ntg_part)
            elif perm_k_mode == 'ntg squared':
                ntg = collection.cached_part_array_ref(ntg_part)
                kv *= ntg * ntg
        kv_collection = resqpy.property.PropertyCollection()
        kv_collection.set_support(support_uuid = collection.support_uuid, model = collection.model)
        kv_collection.add_cached_array_to_imported_list(kv,
                                                        'derived from horizontal perm with mode ' + str(perm_k_mode),
                                                        'KK',
                                                        discrete = False,
                                                        uom = 'mD',
                                                        time_index = None,
                                                        null_value = None,
                                                        property_kind = 'permeability rock',
                                                        facet_type = 'direction',
                                                        facet = 'K',
                                                        realization = perms.realization_for_part(perm_i_part),
                                                        indexable_element = perms.indexable_for_part(perm_i_part),
                                                        count = 1,
                                                        points = False)
        collection.model.h5_release()
        kv_collection.write_hdf5_for_imported_list()
        kv_collection.create_xml_for_imported_list_and_add_parts_to_model()
        collection.inherit_parts_from_other_collection(kv_collection)
        perm_k_part = kv_collection.singleton()
    return perm_k_part


def _get_single_perm_ijk_parts_one(perms, share_perm_parts):
    perm_i_part = perms.singleton()
    if share_perm_parts:
        perm_j_part = perm_k_part = perm_i_part
    elif perms.facet_type_for_part(perm_i_part) == 'direction':
        direction = perms.facet_for_part(perm_i_part)
        if direction == 'J':
            perm_j_part = perm_i_part
            perm_i_part = None
        elif direction == 'IJ':
            perm_j_part = perm_i_part
        elif direction == 'K':
            perm_k_part = perm_i_part
            perm_i_part = None
        elif direction == 'IJK':
            perm_j_part = perm_k_part = perm_i_part
    return perm_i_part, perm_j_part, perm_k_part


def _get_single_perm_ijk_for_direction(perms, direction):
    facet_options, title_options = _get_facet_title_options_for_direction(direction)

    try:
        part = None
        for facet_op in facet_options:
            if not part:
                part = perms.singleton(facet_type = 'direction', facet = facet_op)
        if not part:
            for title in title_options:
                if not part:
                    part = perms.singleton(citation_title = title)
        if not part:
            log.error(f'unable to discern which rock permeability to use for {direction} direction')
    except Exception:
        log.error(f'problem with permeability data (more than one {direction} direction array present?)')
        part = None
    return part


def _get_facet_title_options_for_direction(direction):
    if direction == 'I':
        facet_options = ['I', 'IJ', 'IJK']
        title_options = ['KI', 'PERMI', 'KX', 'PERMX']
    elif direction == 'J':
        facet_options = ['J', 'IJ', 'IJK']
        title_options = ['KJ', 'PERMJ', 'KY', 'PERMY']
    else:
        facet_options = ['K', 'IJK']
        title_options = ['KK', 'PERMK', 'KZ', 'PERMZ']
    return facet_options, title_options


def _part_direction(collection, part):
    facet_t = collection.facet_type_for_part(part)
    if facet_t is None or facet_t != 'direction':
        return None
    return collection.facet_for_part(part)
