"""Submodule containing functions for attribute extraction for a property collection."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import numpy as np
import numpy.ma as ma

import resqpy
import resqpy.property as rqp
import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu
import resqpy.property.property_kind as rqpk


def _min_max_of_cached_array(collection, cached_name, cached_array, null_value, discrete):
    collection.__dict__[cached_name] = cached_array
    if discrete:
        if null_value is None:
            min_value = np.min(cached_array)
            max_value = np.max(cached_array)
        else:
            zorro = collection.masked_array(cached_array, exclude_value = null_value)
            min_value = np.min(zorro)
            max_value = np.max(zorro)
            if min_value is ma.masked:
                min_value = None
            if max_value is ma.masked:
                max_value = None
    else:
        min_value = np.nanmin(cached_array)
        max_value = np.nanmax(cached_array)
        if np.isnan(min_value):
            min_value = None
        if np.isnan(max_value):
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
            lpk = rqpk.PropertyKind(collection.model,
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


def _get_property_array_min_max_value(collection, property_array, const_value, discrete, min_value, max_value,
                                      categorical, null_value):
    if const_value is not None:
        return _get_property_array_min_max_const(const_value, collection.null_value, min_value, max_value, discrete)
    elif property_array is not None:
        return _get_property_array_min_max_array(property_array, min_value, max_value, discrete, categorical,
                                                 null_value)
    return None, None


def _get_property_array_min_max_const(const_value, null_value, min_value, max_value, discrete):
    if (discrete and const_value != null_value) or (not discrete and not np.isnan(const_value)):
        if min_value is None:
            min_value = const_value
        if max_value is None:
            max_value = const_value
    return min_value, max_value


def _get_property_array_min_max_array(property_array, min_value, max_value, discrete, categorical, null_value):
    if discrete:
        if categorical:
            min_value, max_value = _get_property_array_min_max_array_categorical(property_array, min_value, max_value,
                                                                                 null_value)
        else:
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


def _get_property_array_min_max_array_categorical(property_array, min_value, max_value, null_value):
    if min_value is None or max_value is None:
        unique_values = np.unique(property_array)
        if null_value is not None and len(unique_values):
            if unique_values[0] == null_value:
                unique_values = unique_values[1:]
            elif unique_values[-1] == null_value:
                unique_values = unique_values[:-1]
    if min_value is None:
        if len(unique_values):
            min_value = unique_values[0]
        else:
            log.warning('no xml minimum value set for categorical property')
    if max_value is None:
        if len(unique_values):
            max_value = unique_values[-1]
        else:
            log.warning('no xml maximum value set for categorical property')
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
                perm_k_part = _get_perm_k_part(collection, perms, perm_k_mode, share_perm_parts, perm_i_part,
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
        kv_collection = rqp.PropertyCollection()
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
    perm_j_part = perm_k_part = None
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


def _cached_part_array_ref_get_node_points(part_node, dtype):
    patch_list = rqet.list_of_tag(part_node, 'PatchOfPoints')
    assert len(patch_list) == 1  # todo: handle more than one patch of points
    first_values_node = rqet.find_tag(patch_list[0], 'Points')
    if first_values_node is None:
        return None  # could treat as fatal error
    if dtype is None:
        dtype = 'float'
    else:
        assert dtype in ['float', float, np.float32, np.float64]
    tag = 'Coordinates'
    return first_values_node, tag, dtype


def _cached_part_array_ref_get_node_values(part_node, dtype):
    patch_list = rqet.list_of_tag(part_node, 'PatchOfValues')
    assert len(patch_list) == 1  # todo: handle more than one patch of values
    first_values_node = rqet.find_tag(patch_list[0], 'Values')
    if first_values_node is None:
        return None  # could treat as fatal error
    if dtype is None:
        array_type = rqet.node_type(first_values_node)
        assert array_type is not None
        if array_type == 'DoubleHdf5Array':
            dtype = 'float'
        elif array_type == 'IntegerHdf5Array':
            dtype = 'int'
        elif array_type == 'BooleanHdf5Array':
            dtype = 'bool'
        else:
            raise ValueError('array type not catered for: ' + str(array_type))
    tag = 'Values'
    return first_values_node, tag, dtype


def _process_imported_property_get_add_min_max(points, property_kind, string_lookup_uuid, local_property_kind_uuid,
                                               is_bool):
    if points or property_kind == 'categorical':
        add_min_max = True
    elif property_kind == 'discrete' and is_bool:
        add_min_max = False
    elif local_property_kind_uuid is not None and string_lookup_uuid is not None:
        add_min_max = False
    else:
        add_min_max = True
    return add_min_max


def _normalized_part_array_apply_discrete_cycle(discrete_cycle, p_array, min_value, max_value):
    if 'int' in str(
            p_array.dtype) and discrete_cycle is not None:  # could use continuous flag in metadata instead of dtype
        p_array = p_array % discrete_cycle
        min_value = 0
        max_value = discrete_cycle - 1
    elif str(p_array.dtype).startswith('bool'):
        min_value = int(min_value)
        max_value = int(max_value)
    return min_value, max_value, p_array


def _normalized_part_array_nan_if_masked(min_value, max_value, masked):
    min_value = float(min_value)  # will return np.ma.masked if all values are masked out
    if masked and min_value is ma.masked:
        min_value = np.nan
    max_value = float(max_value)
    if masked and max_value is ma.masked:
        max_value = np.nan
    return min_value, max_value


def _normalized_part_array_use_logarithm(min_value, n_prop, masked):
    if min_value <= 0.0:
        n_prop[:] = np.where(n_prop < 0.0001, 0.0001, n_prop)
    n_prop = np.log10(n_prop)
    min_value = np.nanmin(n_prop)
    max_value = np.nanmax(n_prop)
    if masked:
        if min_value is ma.masked:
            min_value = np.nan
        if max_value is ma.masked:
            max_value = np.nan
    return n_prop, min_value, max_value


def _normalized_part_array_fix_zero_at(min_value, max_value, n_prop, fix_zero_at):
    if fix_zero_at <= 0.0:
        if min_value < 0.0:
            n_prop[:] = np.where(n_prop < 0.0, 0.0, n_prop)
        min_value = 0.0
    elif fix_zero_at >= 1.0:
        if max_value > 0.0:
            n_prop[:] = np.where(n_prop > 0.0, 0.0, n_prop)
        max_value = 0.0
    else:
        upper_scaling = max_value / (1.0 - fix_zero_at)
        lower_scaling = -min_value / fix_zero_at
        if upper_scaling >= lower_scaling:
            min_value = -upper_scaling * fix_zero_at
            n_prop[:] = np.where(n_prop < min_value, min_value, n_prop)
        else:
            max_value = lower_scaling * (1.0 - fix_zero_at)
            n_prop[:] = np.where(n_prop > max_value, max_value, n_prop)
    return min_value, max_value, n_prop


def _supporting_shape_grid(support, indexable_element, direction):
    if indexable_element is None or indexable_element == 'cells':
        shape_list = [support.nk, support.nj, support.ni]
    elif indexable_element == 'columns':
        shape_list = [support.nj, support.ni]
    elif indexable_element == 'layers':
        shape_list = [support.nk]
    elif indexable_element == 'faces':
        shape_list = _supporting_shape_grid_faces(direction, support)
    elif indexable_element == 'column edges':
        # I edges first; include outer edges
        shape_list = [(support.nj * (support.ni + 1)) + ((support.nj + 1) * support.ni)]
    elif indexable_element == 'edges per column':
        shape_list = [support.nj, support.ni, 4]  # assume I-, J+, I+, J- ordering
    elif indexable_element == 'faces per cell':
        shape_list = [support.nk, support.nj, support.ni, 6]  # assume K-, K+, J-, I+, J+, I- ordering
        # TODO: resolve ordering of edges and make consistent with maps code (edges per column) and fault module (gcs faces)
    elif indexable_element == 'nodes per cell':
        # kp, jp, ip within each cell; todo: check RESQML shaping
        shape_list = [support.nk, support.nj, support.ni, 2, 2, 2]
    elif indexable_element == 'nodes':
        shape_list = _supporting_shape_grid_nodes(support)
    return shape_list


def _supporting_shape_grid_faces(direction, support):
    assert direction is not None and direction.upper() in 'IJK'
    axis = 'KJI'.index(direction.upper())
    shape_list = [support.nk, support.nj, support.ni]
    shape_list[axis] += 1  # note: properties for grid faces include outer faces
    return shape_list


def _supporting_shape_grid_nodes(support):
    assert not support.k_gaps, 'indexable element of nodes not currently supported for grids with K gaps'
    if support.has_split_coordinate_lines:
        pillar_count = (support.nj + 1) * (support.ni + 1) + support.split_pillars_count
        shape_list = [support.nk + 1, pillar_count]
    else:
        shape_list = [support.nk + 1, support.nj + 1, support.ni + 1]
    return shape_list


def _supporting_shape_wellboreframe(support, indexable_element):
    if indexable_element is None or indexable_element == 'nodes':
        shape_list = [support.node_count]
    elif indexable_element == 'intervals':
        shape_list = [support.node_count - 1]
    return shape_list


def _supporting_shape_wellboremarkerframe(support, indexable_element):
    if indexable_element is None or indexable_element == 'nodes':
        shape_list = [support.node_count]
    elif indexable_element == 'intervals':
        shape_list = [support.node_count - 1]
    return shape_list


def _supporting_shape_blockedwell(support, indexable_element):
    if indexable_element is None or indexable_element == 'intervals':
        shape_list = [support.node_count - 1]  # all intervals, including unblocked
    elif indexable_element == 'nodes':
        shape_list = [support.node_count]
    elif indexable_element == 'cells':
        shape_list = [support.cell_count]  # ie. blocked intervals only
    return shape_list


def _supporting_shape_mesh(support, indexable_element):
    if indexable_element is None or indexable_element == 'cells' or indexable_element == 'columns':
        shape_list = [support.nj - 1, support.ni - 1]
    elif indexable_element == 'nodes':
        shape_list = [support.nj, support.ni]
    return shape_list


def _supporting_shape_surface(support, indexable_element):
    if indexable_element is None or indexable_element == 'faces':
        shape_list = [support.triangle_count()]
    elif indexable_element == 'nodes':
        shape_list = [support.node_count()]
    return shape_list


def _supporting_shape_gridconnectionset(support, indexable_element):
    if indexable_element is None or indexable_element == 'faces':
        shape_list = [support.count]
    return shape_list


def _supporting_shape_other(support, indexable_element):
    if indexable_element is None or indexable_element == 'cells':
        shape_list = [support.cell_count]
    elif indexable_element == 'faces per cell':
        support.cache_all_geometry_arrays()
        shape_list = [len(support.faces_per_cell)]
    return shape_list, support


def _realizations_array_ref_get_r_extent(fill_missing, r_list):
    if fill_missing:
        r_extent = r_list[-1] + 1
    else:
        r_extent = len(r_list)
    return r_extent


def _get_indexable_element(indexable_element, support_type):
    if indexable_element is None:
        if support_type in [
                'obj_IjkGridRepresentation', 'obj_BlockedWellboreRepresentation', 'obj_Grid2dRepresentation',
                'obj_UnstructuredGridRepresentation'
        ]:
            indexable_element = 'cells'
        elif support_type in ['obj_WellboreFrameRepresentation', 'obj_WellboreMarkerFrameRepresentation']:
            indexable_element = 'nodes'  # note: could be 'intervals'
        elif support_type in ['obj_GridConnectionSetRepresentation', 'obj_TriangulatedSetRepresentation']:
            indexable_element = 'faces'
        else:
            raise Exception('indexable element unknown for unsupported supporting representation object')
    return indexable_element


def _cached_part_array_ref_get_array(collection, part, dtype, model, cached_array_name):
    const_value = collection.constant_value_for_part(part)
    if const_value is None:
        _cached_part_array_ref_const_none(collection, part, dtype, model, cached_array_name)
    else:
        _cached_part_array_ref_const_notnone(collection, part, const_value, cached_array_name)
    if not hasattr(collection, cached_array_name):
        return None


def _cached_part_array_ref_const_none(collection, part, dtype, model, cached_array_name):
    part_node = collection.node_for_part(part)
    if part_node is None:
        return None
    if collection.points_for_part(part):
        first_values_node, tag, dtype = _cached_part_array_ref_get_node_points(part_node, dtype)
    else:
        first_values_node, tag, dtype = _cached_part_array_ref_get_node_values(part_node, dtype)

    h5_key_pair = model.h5_uuid_and_path_for_node(first_values_node, tag = tag)
    if h5_key_pair is None:
        return None
    model.h5_array_element(h5_key_pair,
                           index = None,
                           cache_array = True,
                           object = collection,
                           array_attribute = cached_array_name,
                           dtype = dtype)


def _cached_part_array_ref_const_notnone(collection, part, const_value, cached_array_name):
    assert not collection.points_for_part(part), 'constant arrays not supported for points properties'
    assert collection.support is not None
    shape = collection.supporting_shape(indexable_element = collection.indexable_for_part(part),
                                        direction = _part_direction(collection, part))
    assert shape is not None
    a = np.full(shape, const_value, dtype = float if collection.continuous_for_part(part) else int)
    setattr(collection, cached_array_name, a)


def _normalized_part_array_get_minmax(collection, trust_min_max, part, p_array, masked):
    min_value = max_value = None
    if trust_min_max:
        min_value = collection.minimum_value_for_part(part)
        max_value = collection.maximum_value_for_part(part)
    if min_value is None or max_value is None:
        min_value = np.nanmin(p_array)
        if masked and min_value is ma.masked:
            min_value = None
        max_value = np.nanmax(p_array)
        if masked and max_value is ma.masked:
            max_value = None
        collection.override_min_max(part, min_value, max_value)  # NB: this does not modify xml
    return min_value, max_value


def _realizations_array_ref_fill_missing(collection, r_extent, dtype, a):
    for part in collection.parts():
        realization = collection.realization_for_part(part)
        assert realization is not None and 0 <= realization < r_extent, 'realization missing (or out of range?)'
        pa = collection.cached_part_array_ref(part, dtype = dtype)
        a[realization] = pa
        collection.uncache_part_array(part)
    return a


def _realizations_array_ref_not_fill_missing(collection, r_list, dtype, a):
    for index in range(len(r_list)):
        realization = r_list[index]
        part = collection.singleton(realization = realization)
        pa = collection.cached_part_array_ref(part, dtype = dtype)
        a[index] = pa
        collection.uncache_part_array(part)
    return a


def _realizations_array_ref_get_shape_list(collection, indexable_element, r_extent):
    shape_list = collection.supporting_shape(indexable_element = indexable_element)
    shape_list.insert(0, r_extent)
    if collection.points_for_part(collection.parts()[0]):
        shape_list.append(3)
    return shape_list


def _realizations_array_ref_initial_checks(collection):
    assert collection.support is not None,  \
        'attempt to build realizations array for property collection without supporting representation'
    assert collection.number_of_parts() > 0,  \
        'attempt to build realizations array for empty property collection'
    assert collection.has_single_property_kind(),  \
        'attempt to build realizations array for collection with multiple property kinds'
    assert collection.has_single_indexable_element(),  \
        'attempt to build realizations array for collection containing a variety of indexable elements'
    assert collection.has_single_uom(),  \
        'attempt to build realizations array for collection containing multiple units of measure'
    r_list = collection.realization_list(sort_list = True)
    assert collection.number_of_parts() == len(r_list),  \
        'collection covers more than realizations of a single property'
    continuous = collection.all_continuous()
    if not continuous:
        assert collection.all_discrete(), 'mixture of continuous and discrete properties in collection'
    return r_list, continuous


def _time_array_ref_initial_checks(collection):
    assert collection.support is not None,  \
        'attempt to build time series array for property collection without supporting representation'
    assert collection.number_of_parts() > 0,  \
        'attempt to build time series array for empty property collection'
    assert collection.has_single_property_kind(),  \
        'attempt to build time series array for collection with multiple property kinds'
    assert collection.has_single_indexable_element(),  \
        'attempt to build time series array for collection containing a variety of indexable elements'
    assert collection.has_single_uom(),  \
        'attempt to build time series array for collection containing multiple units of measure'

    ti_list = collection.time_index_list(sort_list = True)
    assert collection.number_of_parts() == len(ti_list),  \
        'collection covers more than time indices of a single property'

    continuous = collection.all_continuous()
    if not continuous:
        assert collection.all_discrete(), 'mixture of continuous and discrete properties in collection'
    return ti_list, continuous


def _time_array_ref_fill_missing(collection, ti_extent, dtype, a):
    for part in collection.parts():
        time_index = collection.time_index_for_part(part)
        assert time_index is not None and 0 <= time_index < ti_extent, 'time index missing (or out of range?)'
        pa = collection.cached_part_array_ref(part, dtype = dtype)
        a[time_index] = pa
        collection.uncache_part_array(part)
    return a


def _time_array_ref_not_fill_missing(collection, ti_list, dtype, a):
    for index in range(len(ti_list)):
        time_index = ti_list[index]
        part = collection.singleton(time_index = time_index)
        pa = collection.cached_part_array_ref(part, dtype = dtype)
        a[index] = pa
        collection.uncache_part_array(part)
    return a


def _facet_array_ref_checks(collection):
    assert collection.support is not None,  \
        'attempt to build facets array for property collection without supporting representation'
    assert collection.number_of_parts() > 0,  \
        'attempt to build facets array for empty property collection'
    assert collection.has_single_property_kind(),  \
        'attempt to build facets array for collection containing multiple property kinds'
    assert collection.has_single_indexable_element(),  \
        'attempt to build facets array for collection containing a variety of indexable elements'
    assert collection.has_single_uom(),  \
        'attempt to build facets array for collection containing multiple units of measure'
