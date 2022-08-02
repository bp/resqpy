"""_hdf5.py: functions supporting Model methods relating to hdf5 access."""

import logging

log = logging.getLogger(__name__)

import os
import h5py
import numpy as np

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
from resqpy.olio.xml_namespaces import curly_namespace as ns

# see also _external_parts_list() function in _catalogue module
# and _create_hdf5_ext() and _create_hdf5_dataset_ref() functions in _xml module


def _h5_set_default_override(model, override):
    """Sets the default hdf5 filename override mode for the model."""

    assert override in ('none', 'dir', 'full')
    model.default_h5_override = override


def _h5_uuid_and_path_for_node(model, node, tag = 'Values'):
    """Returns a (hdf5_uuid, hdf5_internal_path) pair for an xml array node."""

    child = rqet.find_tag(node, tag)
    if child is None:
        return None
    assert rqet.node_type(child) == 'Hdf5Dataset'
    h5_path = rqet.find_tag(child, 'PathInHdfFile').text
    h5_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(child, ['HdfProxy', 'UUID']))
    return (h5_uuid, h5_path)


def _h5_uuid_list(model, node):
    """Returns a list of all uuids for hdf5 external part(s) referred to in recursive tree."""

    # todo: check that uuid.UUID has __EQ__ defined and that set union function applies it correctly

    def recursive_uuid_set(node):
        uuid_set = set()
        for child in node:
            uuid_set = uuid_set.union(recursive_uuid_set(child))
        if rqet.node_type(node) == 'Hdf5Dataset':
            h5_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(node, ['HdfProxy', 'UUID']))
            uuid_set.add(h5_uuid)
        return uuid_set

    return list(recursive_uuid_set(node))


def _h5_uuid(model):
    """Returns the uuid of the 'main' hdf5 file."""

    if model.main_h5_uuid is None:
        uuid_list = None
        ext_parts = model.parts_list_of_type('EpcExternalPartReference')
        if len(ext_parts) == 1:
            model.main_h5_uuid = model.uuid_for_part(ext_parts[0])
        else:
            try:
                grid_root = model.resolve_grid_root()
                if grid_root is not None:
                    uuid_list = _h5_uuid_list(model, grid_root)
            except Exception:
                uuid_list = None
            if uuid_list is None or len(uuid_list) == 0:
                for part, (_, _, tree) in model.parts_forest.items():
                    uuid_list = _h5_uuid_list(model, tree.getroot())
                    if uuid_list is not None and len(uuid_list) > 0:
                        break
            if uuid_list is not None and len(uuid_list) > 0:
                model.main_h5_uuid = uuid_list[0]  # arbitrary use of first hdf5 uuid encountered
    return model.main_h5_uuid


def _h5_file_name(model, uuid = None, override = 'default', file_must_exist = True):
    """Returns full path for hdf5 file with given ext uuid."""

    if uuid is None:
        uuid = _h5_uuid(model)
    if uuid is not None:
        if isinstance(uuid, str):
            uuid = bu.uuid_from_string(uuid)
        if uuid.bytes in model.h5_dict:
            return model.h5_dict[uuid.bytes]
    h5_full_path = _h5_apply_override(model, override, None, uuid)
    if file_must_exist:
        if not h5_full_path:
            raise FileNotFoundError('unable to determine hdf5 file name')
        if not os.path.exists(h5_full_path):
            raise FileNotFoundError(f'hdf5 file missing: {h5_full_path}')
    if h5_full_path and uuid is not None:
        model.h5_dict[uuid.bytes] = h5_full_path
    return h5_full_path


def _h5_apply_override(model, override, supplied, uuid):
    if isinstance(override, bool):
        # could raise a deprecation warning here
        override = 'full' if override else 'dir'
    elif override == 'default':
        override = model.default_h5_override
    assert override in ('none', 'dir', 'full')
    if override == 'full':
        assert model.epc_file and model.epc_file.endswith('.epc')
        return model.epc_file[:-4] + '.h5'
    if not supplied:
        return _h5_target_path_from_rels(model, uuid, override == 'dir')
    if (override == 'dir' or os.sep not in supplied) and model.epc_directory:
        return os.path.join(model.epc_directory, os.path.basename(supplied))
    return supplied


def _h5_target_path_from_rels(model, uuid, override_dir):
    """Extracts an hdf5 file name from the Target attribute of relationships xml."""

    log.debug(f'looking for ext uuid: {uuid}')
    for rel_name, entry in model.rels_forest.items():
        log.debug(f'considering rels: {rel_name}')
        if uuid is None or bu.matching_uuids(uuid, entry[0]):
            log.debug(f'found hdf5 rels part: {rel_name}')
            rel_root = entry[1].getroot()
            for child in rel_root:
                if child.attrib['Id'] == 'Hdf5File' and child.attrib['TargetMode'] == 'External':
                    target_path = child.attrib['Target']
                    if not target_path:
                        return None
                    return _h5_apply_override(model, 'dir' if override_dir else 'none', target_path, uuid)
    log.warning('h5 target path not found in rels')
    return None


def _h5_access(model, uuid = None, mode = 'r', override = 'default', file_path = None):
    """Returns an open h5 file handle for the hdf5 file with the given ext uuid."""

    if model.h5_currently_open_mode is not None and model.h5_currently_open_mode != mode:
        _h5_release(model)
    if file_path:
        file_name = _h5_apply_override(model, override, file_path, uuid)
    else:
        file_name = _h5_file_name(model, uuid = uuid, override = override, file_must_exist = (mode == 'r'))
    if mode == 'a' and not os.path.exists(file_name):
        mode = 'w'
    if model.h5_currently_open_root is not None and os.path.samefile(model.h5_currently_open_path, file_name):
        return model.h5_currently_open_root
    if model.h5_currently_open_root is not None:
        _h5_release(model)
    model.h5_currently_open_path = file_name
    model.h5_currently_open_mode = mode
    model.h5_currently_open_root = h5py.File(file_name, mode)  # could use try to trap file in use errors?
    return model.h5_currently_open_root


def _h5_release(model):
    """Releases (closes) the currently open hdf5 file."""

    if model.h5_currently_open_root is not None:
        model.h5_currently_open_root.close()
        model.h5_currently_open_root = None
    model.h5_currently_open_path = None
    model.h5_currently_open_mode = None


def _h5_array_shape_and_type(model, h5_key_pair):
    """Returns the shape and dtype of the array, as stored in the hdf5 file."""

    h5_root = _h5_access(model, h5_key_pair[0])
    if h5_root is None:
        return (None, None)
    shape_tuple = tuple(h5_root[h5_key_pair[1]].shape)
    dtype = h5_root[h5_key_pair[1]].dtype
    return (shape_tuple, dtype)


def _h5_array_element(model,
                      h5_key_pair,
                      index = None,
                      cache_array = False,
                      object = None,
                      array_attribute = None,
                      dtype = 'float',
                      required_shape = None):
    """Returns one element from an hdf5 array and/or caches the array."""

    def reshaped_index(index, shape_tuple, required_shape):
        tail = len(shape_tuple) - len(index)
        if tail > 0:
            assert shape_tuple[-tail:] == required_shape[-tail:], 'not enough indices to allow reshaped indexing'
        natural = 0
        extent = 1
        for axis in range(len(shape_tuple) - tail - 1, -1, -1):
            natural += index[axis] * extent
            extent *= shape_tuple[axis]
        r_extent = np.empty(len(required_shape) - tail, dtype = int)
        r_extent[-1] = required_shape[-(tail + 1)]
        for axis in range(len(required_shape) - tail - 2, -1, -1):
            r_extent[axis] = required_shape[axis] * r_extent[axis + 1]
        r_index = np.empty(len(required_shape) - tail, dtype = int)
        for axis in range(len(r_index) - 1):
            r_index[axis], natural = divmod(natural, r_extent[axis + 1])
        r_index[-1] = natural
        return r_index

    if object is None:
        object = model

    # Check if attribute has already be cached
    if array_attribute is not None:
        existing_value = getattr(object, array_attribute, None)

        # Watch out for np.array(None): check existing_value has a valid "shape"
        if existing_value is not None and getattr(existing_value, "shape", False):
            if index is None:
                return None  # this option allows caching of array without actually referring to any element
            return existing_value[tuple(index)]

    h5_root = _h5_access(model, h5_key_pair[0])
    if h5_root is None:
        return None
    if cache_array:
        shape_tuple = tuple(h5_root[h5_key_pair[1]].shape)
        if required_shape is None or shape_tuple == required_shape:
            object.__dict__[array_attribute] = np.zeros(shape_tuple, dtype = dtype)
            object.__dict__[array_attribute][:] = h5_root[h5_key_pair[1]]
        else:
            object.__dict__[array_attribute] = np.zeros(required_shape, dtype = dtype)
            object.__dict__[array_attribute][:] = np.array(h5_root[h5_key_pair[1]],
                                                           dtype = dtype).reshape(required_shape)
        _h5_release(model)
        if index is None:
            return None
        return object.__dict__[array_attribute][tuple(index)]
    else:
        if index is None:
            return None
        if required_shape is None:
            result = h5_root[h5_key_pair[1]][tuple(index)]
        else:
            shape_tuple = tuple(h5_root[h5_key_pair[1]].shape)
            if shape_tuple == required_shape:
                result = h5_root[h5_key_pair[1]][tuple(index)]
            else:
                index = reshaped_index(index, required_shape, shape_tuple)
                result = h5_root[h5_key_pair[1]][tuple(index)]
        _h5_release(model)
        if dtype is None:
            return result
        if result.size == 1:
            if dtype is float or (isinstance(dtype, str) and dtype.startswith('float')):
                return float(result)
            elif dtype is int or (isinstance(dtype, str) and dtype.startswith('int')):
                return int(result)
            elif dtype is bool or (isinstance(dtype, str) and dtype.startswith('bool')):
                return bool(result)
        return np.array(result, dtype = dtype)


def _h5_array_slice(model, h5_key_pair, slice_tuple):
    """Loads a slice of an hdf5 array."""

    h5_root = _h5_access(model, h5_key_pair[0])
    return h5_root[h5_key_pair[1]][slice_tuple]


def _h5_overwrite_array_slice(model, h5_key_pair, slice_tuple, array_slice):
    """Overwrites (updates) a slice of an hdf5 array."""

    h5_root = _h5_access(model, h5_key_pair[0], mode = 'a')
    dset = h5_root[h5_key_pair[1]]
    dset[slice_tuple] = array_slice


def h5_clear_filename_cache(model):
    """Clears the cached filenames associated with all ext uuids."""

    _h5_release(model)
    model.h5_dict = {}


def _change_hdf5_uuid_in_hdf5_references(model, node, old_uuid, new_uuid):
    """Scan node for hdf5 references and set the uuid of the hdf5 file itself to new_uuid."""

    count = 0
    old_uuid_str = str(old_uuid)
    new_uuid_str = str(new_uuid)
    for ref_node in node.iter(ns['eml'] + 'HdfProxy'):
        try:
            uuid_node = rqet.find_tag(ref_node, 'UUID')
            if old_uuid is None or uuid_node.text == old_uuid_str:
                uuid_node.text = new_uuid_str
                count += 1
        except Exception:
            pass
    # if count == 1:
    #     log.debug('one hdf5 reference modified')
    # else:
    #     log.debug(str(count) + ' hdf5 references modified')
    if count > 0:
        model.set_modified()


def _change_uuid_in_hdf5_references(model, node, old_uuid, new_uuid):
    """Scan node for hdf5 references using the old_uuid and replace with the new_uuid."""

    count = 0
    old_uuid_str = str(old_uuid)
    new_uuid_str = str(new_uuid)
    for ref_node in node.iter(ns['eml'] + 'PathInHdfFile'):
        try:
            uuid_place = ref_node.text.index(old_uuid_str)
            new_path_in_hdf = ref_node.text[:uuid_place] + new_uuid_str + ref_node.text[uuid_place + len(old_uuid_str):]
            log.debug('path in hdf update from: ' + ref_node.text + ' to: ' + new_path_in_hdf)
            ref_node.text = new_path_in_hdf
            count += 1
        except Exception:
            pass
    if count == 1:
        log.debug('one hdf5 reference modified')
    else:
        log.debug(str(count) + ' hdf5 references modified')
    if count > 0:
        model.set_modified()


def _change_filename_in_hdf5_rels(model, new_hdf5_filename = None):
    """Scan relationships forest for hdf5 external parts and patch in a new filename."""

    if not new_hdf5_filename and model.epc_file and model.epc_file.endswith('.epc'):
        new_hdf5_filename = os.path.split(model.epc_file)[1][:-4] + '.h5'
    count = 0
    for rel_name, entry in model.rels_forest.items():
        rel_root = entry[1].getroot()
        for child in rel_root:
            if child.attrib['Id'] == 'Hdf5File' and child.attrib['TargetMode'] == 'External':
                child.attrib['Target'] = new_hdf5_filename
                count += 1
    log.info(str(count) + ' hdf5 filename' + _pl(count) + ' set to: ' + new_hdf5_filename)
    if count > 0:
        model.set_modified()


def _pl(i, e = False):
    return '' if i == 1 else 'es' if e else 's'
