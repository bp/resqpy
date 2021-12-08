"""_forestry.py: functions to support management of parts in Model."""

import logging

log = logging.getLogger(__name__)

import copy
import os
import shutil
import zipfile as zf

import resqpy.olio.consolidation as cons
import resqpy.olio.uuid as bu
import resqpy.olio.write_hdf5 as whdf5
import resqpy.olio.xml_et as rqet
from resqpy.olio.xml_namespaces import curly_namespace as ns
from resqpy.olio.xml_namespaces import namespace as ns_url


def _load_part(model, epc, part_name, is_rels = None):
    """Load and parse xml tree for given part name, storing info in parts forest (or rels forest)."""

    # note: epc is 'open' ZipFile handle
    if part_name.startswith('/'):
        part_name = part_name[1:]

    try:

        # log.debug('loading part ' + part_name)
        if is_rels is None:
            is_rels = part_name.endswith('.rels')
        is_other = not is_rels and part_name.startswith('docProps')

        part_type = None
        part_uuid = None

        if is_rels:
            if part_name in model.rels_forest:
                (part_uuid, _) = model.rels_forest[part_name]
            if part_uuid is None:
                part_uuid = rqet.uuid_in_part_name(part_name)
        elif is_other:
            (part_type, _) = model.other_forest[part_name]
        else:
            (part_type, part_uuid, _) = model.parts_forest[part_name]  # part_type must already have been established

        with epc.open(part_name) as part_xml:
            part_tree = rqet.parse(part_xml)
            if is_rels:
                model.rels_forest[part_name] = (part_uuid, part_tree)
            elif is_other:
                model.other_forest[part_name] = (part_type, part_tree)
            else:
                uuid_from_tree = rqet.uuid_for_part_root(part_tree.getroot())
                if part_uuid is None:
                    part_uuid = uuid_from_tree
                elif uuid_from_tree is not None:
                    assert bu.matching_uuids(part_uuid, uuid_from_tree)
                model.parts_forest[part_name] = (part_type, part_uuid, part_tree)
                _set_uuid_to_part(model, part_name)
                if model.crs_uuid is None and part_type == 'obj_LocalDepth3dCrs':  # randomly assign first crs as primary crs for model
                    model.crs_uuid = part_uuid

        return True

    except Exception:

        log.exception('(okay to continue?) failed to load part: ' + part_name)
        return False


def _fell_part(model, part_name):
    """Removes the named part from the in-memory parts forest."""

    try:
        _del_uuid_to_part(model, part_name)
    except Exception:
        pass
    try:
        del model.parts_forest[part_name]
    except Exception:
        pass
    try:
        del model.rels_forest[part_name]
    except Exception:
        pass
    try:
        del model.other_forest[part_name]
    except Exception:
        pass


def _remove_part_from_main_tree(model, part):
    """Removes the named part from the main (Content_Types) tree."""

    for child in model.main_root:
        if rqet.stripped_of_prefix(child.tag) == 'Override':
            part_name = child.attrib['PartName']
            if part_name[0] == '/':
                part_name = part_name[1:]
            if part_name == part:
                log.debug('removing part from main xml tree: ' + part)
                model.main_root.remove(child)
                break


def _tidy_up_forests(model, tidy_main_tree = True, tidy_others = False, remove_extended_core = True):
    """Removes any parts that do not have any related data in dictionaries."""

    deletion_list = []
    for part, info in model.parts_forest.items():
        if info == (None, None, None):
            deletion_list.append(part)
    for part in deletion_list:
        log.debug('removing part due to lack of xml tree etc.: ' + str(part))
        if tidy_main_tree:
            _remove_part_from_main_tree(model, part)
        _del_uuid_to_part(model, part)
        del model.parts_forest[part]
    deletion_list = []
    for part, info in model.rels_forest.items():
        if info == (None, None):
            deletion_list.append(part)
    for part in deletion_list:
        log.debug('removing rels part due to lack of xml tree etc.: ' + str(part))
        if tidy_main_tree:
            _remove_part_from_main_tree(model, part)
        del model.rels_forest[part]
    if tidy_others:
        for part, info in model.other_forest.items():
            if info == (None, None):
                deletion_list.append(part)
        for part in deletion_list:
            log.debug('removing docProps part due to lack of xml tree etc.: ' + str(part))
            if tidy_main_tree:
                _remove_part_from_main_tree(model, part)
            del model.other_forest[part]
    if remove_extended_core and 'docProps/extendedCore.xml' in model.other_forest:  # more trouble than it's worth
        part = 'docProps/extendedCore.xml'
        if tidy_main_tree:
            _remove_part_from_main_tree(model, part)
        del model.other_forest[part]


def _load_epc(model, epc_file, full_load = True, epc_subdir = None, copy_from = None):
    """Load xml parts of model from epc file (HDF5 arrays are not loaded)."""

    if not epc_file.endswith('.epc'):
        epc_file += '.epc'

    _copy_dataset_if_requested(copy_from, epc_file)

    log.info('loading resqml model from epc file ' + epc_file)

    if model.modified:
        log.warning('loading model from epc, discarding previous in-memory modifications')
        model.initialize()

    model.set_epc_file_and_directory(epc_file)

    with zf.ZipFile(epc_file) as epc:
        names = epc.namelist()
        _set_part_names_in_forests(model, epc_subdir, names)
        with epc.open('[Content_Types].xml') as main_xml:
            model.main_tree = rqet.parse(main_xml)
            model.main_root = model.main_tree.getroot()
            for child in model.main_root:
                _complete_forest_entry_for_part(epc, model, epc_subdir, full_load, child)
        if model.rels_present and full_load:
            _load_relationships(epc, model, epc_subdir, names)
            if copy_from:
                model.change_filename_in_hdf5_rels(os.path.split(epc_file)[1][:-4] + '.h5')
        elif not model.rels_present:
            assert len(model.rels_forest) == 0
        if full_load:
            _tidy_up_forests(model)


def _copy_dataset_if_requested(copy_from, epc_file):
    if copy_from:
        if not copy_from.endswith('.epc'):
            copy_from += '.epc'
        log.info('copying ' + copy_from + ' to ' + epc_file + ' along with paired .h5 files')
        shutil.copy(copy_from, epc_file)
        shutil.copy(copy_from[:-4] + '.h5', epc_file[:-4] + '.h5')


def _exclude(name, epc_subdir):
    if epc_subdir is None:
        return False
    if '/' not in name:
        return False
    if name.startswith('docProps') or name.startswith('_rels'):
        return False
    if isinstance(epc_subdir, str):
        epc_subdir = [epc_subdir]
    for subdir in epc_subdir:
        if subdir.endswith('/'):
            head = subdir
        else:
            head = subdir + '/'
        if name.startswith(head):
            return False
    return True


def _set_part_names_in_forests(model, epc_subdir, names):
    for name in names:
        if _exclude(name, epc_subdir):
            continue
        if name != '[Content_Types].xml':
            if name.startswith('docProps'):
                model.other_forest[name] = (None, None)  # used for non-uuid parts, ie. docProps
            else:
                part_uuid = rqet.uuid_in_part_name(name)
                if '_rels' in name:
                    model.rels_forest[name] = (part_uuid, None)
                else:
                    model.parts_forest[name] = (None, part_uuid, None)
                    _set_uuid_to_part(model, name)


def _complete_forest_entry_for_part(epc, model, epc_subdir, full_load, child):
    if rqet.stripped_of_prefix(child.tag) == 'Override':
        attrib_dict = child.attrib
        part_name = attrib_dict['PartName']
        if part_name[0] == '/':
            part_name = part_name[1:]
        part_type = rqet.content_type(attrib_dict['ContentType'])
        if part_name.startswith('docProps'):
            if part_name not in model.other_forest:
                log.warning('docProps entry in Content_Types does not exist as part in epc: ' + part_name)
                return
            model.other_forest[part_name] = (part_type, None)
        else:
            if part_name not in model.parts_forest:
                if epc_subdir is None:
                    log.warning('entry in Content_Types does not exist as part in epc: ' + part_name)
                return
            part_uuid = model.parts_forest[part_name][1]
            model.parts_forest[part_name] = (part_type, part_uuid, None)
        if full_load:
            load_success = _load_part(model, epc, part_name)
            if not load_success:
                _fell_part(model, part_name)
    elif rqet.stripped_of_prefix(child.tag) == 'Default':
        if 'Extension' in child.attrib.keys() and child.attrib['Extension'] == 'rels':
            assert not model.rels_present
            model.rels_present = True
    else:
        # todo: check standard for other valid tags
        pass


def _load_relationships(epc, model, epc_subdir, names):
    for name in names:
        if _exclude(name, epc_subdir):
            continue
        if name.startswith('_rels/'):
            load_success = _load_part(model, epc, name, is_rels = True)
            if not load_success:
                _fell_part(model, name)


def _store_epc(model, epc_file = None, main_xml_name = '[Content_Types].xml', only_if_modified = False):
    """Write xml parts of model to epc file (HDF5 arrays are not written here)."""

    # for prefix, uri in ns.items():
    #     et.register_namespace(prefix, uri)

    if not epc_file:
        epc_file = model.epc_file
    assert epc_file, 'no file name given or known when attempting to store epc'

    if only_if_modified and not model.modified:
        return

    log.info('storing resqml model to epc file ' + epc_file)

    assert model.main_tree is not None
    if model.main_root is None:
        model.main_root = model.main_tree.getroot()

    with zf.ZipFile(epc_file, mode = 'w') as epc:
        with epc.open(main_xml_name, mode = 'w') as main_xml:
            log.debug('Writing main xml: ' + main_xml_name)
            rqet.write_xml(main_xml, model.main_tree, standalone = 'yes')
        for part_name, (_, _, part_tree) in model.parts_forest.items():
            if part_tree is None:
                log.warning('No xml tree present to write for part: ' + part_name)
                continue
            if part_name[0] == '/':
                part_name = part_name[1:]
            with epc.open(part_name, mode = 'w') as part_xml:
                rqet.write_xml(part_xml, part_tree, standalone = None)
        for part_name, (_, part_tree) in model.other_forest.items():
            if part_tree is None:
                log.warning('No xml tree present to write for other part: ' + part_name)
                continue
            if part_name[0] == '/':
                part_name = part_name[1:]
            with epc.open(part_name, mode = 'w') as part_xml:
                rqet.write_xml(part_xml, part_tree, standalone = 'yes')
        if model.rels_present:
            for part_name, (_, part_tree) in model.rels_forest.items():
                if part_tree is None:
                    log.warning('No xml tree present to write for rels part: ' + part_name)
                    continue
                with epc.open(part_name, mode = 'w') as part_xml:
                    rqet.write_xml(part_xml, part_tree, standalone = 'yes')
        # todo: other parts (documentation etc.)
    model.set_epc_file_and_directory(epc_file)
    model.modified = False


def _copy_part(model, existing_uuid, new_uuid, change_hdf5_refs = False):
    """Makes a new part as a copy of an existing part with only a new uuid set; the new part can then be modified."""

    old_uuid_str = str(existing_uuid)
    new_uuid_str = str(new_uuid)
    log.debug('copying xml part from uuid: ' + old_uuid_str + ' to uuid: ' + new_uuid_str)
    existing_parts_list = model.parts_list_of_type(uuid = existing_uuid)
    if len(existing_parts_list) == 0:
        log.warning('failed to find existing part for copying with uuid: ' + old_uuid_str)
        return None
    assert len(existing_parts_list) == 1, 'more than one existing part found with uuid: ' + old_uuid_str
    (part_type, old_uuid, old_tree) = model.parts_forest[existing_parts_list[0]]
    assert bu.matching_uuids(old_uuid, existing_uuid)
    new_tree = copy.deepcopy(old_tree)
    new_root = new_tree.getroot()
    part_name = rqet.patch_uuid_in_part_root(new_root, new_uuid)
    if change_hdf5_refs:
        model.change_uuid_in_hdf5_references(new_root, old_uuid_str, new_uuid_str)
    _add_part(model, part_type, new_uuid, new_root, add_relationship_part = False)
    return part_name


def _add_part(model, content_type, uuid, root, add_relationship_part = True, epc_subdir = None):
    """Adds a (recently created) node as a new part in the model's parts forest."""

    if content_type[0].isupper():
        content_type = 'obj_' + content_type
    use_other = (content_type == 'docProps')
    if use_other:
        if rqet.pretend_to_be_fesapi or rqet.use_fesapi_quirks:
            prefix = '/'
        else:
            prefix = ''
        part_name = prefix + 'docProps/core.xml'
        ct = 'application/vnd.openxmlformats-package.core-properties+xml'
    else:
        part_name = rqet.part_name_for_object(content_type, uuid, prefixed = False, epc_subdir = epc_subdir)
        if 'EpcExternalPartReference' in content_type:
            ct = 'application/x-eml+xml;version=2.0;type=' + content_type
        else:
            ct = 'application/x-resqml+xml;version=2.0;type=' + content_type

    # log.debug('adding part: ' + part_name)
    if isinstance(uuid, str):
        uuid = bu.uuid_from_string(uuid)
    part_tree = rqet.ElementTree(element = root)
    if use_other:
        model.other_forest[part_name] = (content_type, part_tree)
    else:
        model.parts_forest[part_name] = (content_type, uuid, part_tree)
        _set_uuid_to_part(model, part_name)
    main_ref = rqet.SubElement(model.main_root, ns['content_types'] + 'Override')
    main_ref.set('PartName', part_name)
    main_ref.set('ContentType', ct)
    if add_relationship_part and model.rels_present:
        rels_node = rqet.Element(ns['rels'] + 'Relationships')
        rels_node.text = '\n'
        rels_tree = rqet.ElementTree(element = rels_node)
        if use_other:
            rels_part_name = '_rels/.rels'
        else:
            rels_part_name = rqet.rels_part_name_for_part(part_name)
        model.rels_forest[rels_part_name] = (uuid, rels_tree)
    model.set_modified()


def _patch_root_for_part(model, part, root):
    """Updates the xml tree for the part without changing the uuid."""

    content_type, uuid, part_tree = model.parts_forest[part]
    assert bu.matching_uuids(uuid, rqet.uuid_for_part_root(root))
    part_tree = rqet.ElementTree(element = root)
    model.parts_forest[part] = (content_type, uuid, part_tree)


def _remove_part(model, part_name, remove_relationship_part):
    """Removes a part from the parts forest; optionally remove corresponding rels part and other relationships."""

    _del_uuid_to_part(model, part_name)
    model.parts_forest.pop(part_name)
    if remove_relationship_part:
        if 'docProps' in part_name:
            rels_part_name = '_rels/.rels'
        else:
            related_parts = model.parts_list_filtered_by_related_uuid(model.list_of_parts(),
                                                                      rqet.uuid_in_part_name(part_name))
            for relative in related_parts:
                (rel_uuid, rel_tree) = model.rels_forest[rqet.rels_part_name_for_part(relative)]
                rel_root = rel_tree.getroot()
                for child in rel_root:
                    if rqet.stripped_of_prefix(child.tag) != 'Relationship':
                        continue
                    if child.attrib['Target'] == part_name:
                        rel_root.remove(child)
            rels_part_name = rqet.rels_part_name_for_part(part_name)
        model.rels_forest.pop(rels_part_name)
    _remove_part_from_main_tree(model, part_name)
    model.set_modified()


def _duplicate_node(model, existing_node, add_as_part = True):
    """Creates a deep copy of the xml node (typically from another model) and optionally adds as part."""

    new_node = copy.deepcopy(existing_node)
    if add_as_part:
        uuid = rqet.uuid_for_part_root(new_node)
        if model.part_for_uuid(uuid) is None:
            _add_part(model, rqet.node_type(new_node), uuid, new_node)
        else:
            log.warning('rejected attempt to add a duplicated part with an existing uuid')
    return new_node


def _force_consolidation_uuid_equivalence(model, immigrant_uuid, resident_uuid):
    """Force immigrant object to be teated as equivalent to resident during consolidation."""

    if model.consolidation is None:
        model.consolidation = cons.Consolidation(model)
    model.consolidation.force_uuid_equivalence(immigrant_uuid, resident_uuid)


def _copy_part_from_other_model(model,
                                other_model,
                                part,
                                realization = None,
                                consolidate = True,
                                force = False,
                                cut_refs_to_uuids = None,
                                cut_node_types = None,
                                self_h5_file_name = None,
                                h5_uuid = None,
                                other_h5_file_name = None):
    """Fully copies part in from another model, with referenced parts, hdf5 data and relationships."""

    # todo: double check behaviour around equivalent CRSes, especially any default crs in model

    assert other_model is not None
    if other_model is model:
        return
    assert part is not None
    if realization is not None:
        assert isinstance(realization, int) and realization >= 0
    if force:
        assert consolidate
    if not other_h5_file_name:
        other_h5_file_name = other_model.h5_file_name()
    if not self_h5_file_name:
        self_h5_file_name = model.h5_file_name(file_must_exist = False)
    hdf5_copy_needed = not os.path.samefile(self_h5_file_name, other_h5_file_name)

    # check whether already existing in this model
    if part in model.parts_forest.keys():
        return

    if other_model.type_of_part(part) == 'obj_EpcExternalPartReference':
        log.debug('refusing to copy hdf5 ext part from other model')
        return

    log.debug('copying part: ' + str(part))

    uuid = rqet.uuid_in_part_name(part)
    if not force:
        assert model.part_for_uuid(uuid) is None, 'part copying failure: uuid exists for different part!'

    # duplicate xml tree and add as a part
    other_root = other_model.root_for_part(part, is_rels = False)
    if other_root is None:
        log.error('failed to copy part (missing in source model?): ' + str(part))
        return

    resident_uuid = _unforced_consolidation(model, other_model, consolidate, force, part)

    if resident_uuid is None:

        root_node = _duplicate_node(model, other_root)  # adds duplicated node as part
        assert root_node is not None

        _set_realization_index_node_if_required(realization, root_node)

        if hdf5_copy_needed:
            # copy hdf5 data
            hdf5_internal_paths = [node.text for node in rqet.list_of_descendant_tag(other_root, 'PathInHdfFile')]
            hdf5_count = whdf5.copy_h5_path_list(other_h5_file_name, self_h5_file_name, hdf5_internal_paths, mode = 'a')
            # create relationship with hdf5 if needed and modify h5 file uuid in xml references
            _copy_part_hdf5_setup(model, hdf5_count, h5_uuid, root_node)
        # NB. assumes ext part is already established when sharing a common hdf5 file

        # cut references to objects to be excluded
        if cut_refs_to_uuids:
            rqet.cut_obj_references(root_node, cut_refs_to_uuids)

        if cut_node_types:
            rqet.cut_nodes_of_types(root_node, cut_node_types)

        # recursively copy in referenced parts where they don't already exist in this model
        _copy_referenced_parts(model, other_model, realization, consolidate, force, cut_refs_to_uuids, cut_node_types,
                               self_h5_file_name, h5_uuid, other_h5_file_name, root_node)

        resident_uuid = uuid

    else:

        root_node = model.root_for_uuid(resident_uuid)

    # copy relationships where target part is present in this model – this part is source, then destination
    _copy_relationships_for_present_targets(model, other_model, consolidate, force, resident_uuid, root_node)


def _unforced_consolidation(model, other_model, consolidate, force, part):
    if consolidate and not force:
        if model.consolidation is None:
            model.consolidation = cons.Consolidation(model)
        resident_uuid = model.consolidation.equivalent_uuid_for_part(part, immigrant_model = other_model)
    else:
        resident_uuid = None
    return resident_uuid


def _set_realization_index_node_if_required(realization, root_node):
    if realization is not None and rqet.node_type(root_node).endswith('Property'):
        ri_node = rqet.find_tag(root_node, 'RealizationIndex')
        if ri_node is None:
            ri_node = rqet.SubElement(root_node, ns['resqml2'] + 'RealizationIndex')
            ri_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
        # NB. this intentionally overwrites any pre-existing realization number
        ri_node.text = str(realization)


def _copy_part_hdf5_setup(model, hdf5_count, h5_uuid, root_node):
    if hdf5_count:
        if h5_uuid is None:
            h5_uuid = model.h5_uuid()
        if h5_uuid is None:
            model.create_hdf5_ext()
            h5_uuid = model.h5_uuid()
        model.change_hdf5_uuid_in_hdf5_references(root_node, None, h5_uuid)
        ext_part = rqet.part_name_for_object('obj_EpcExternalPartReference', h5_uuid, prefixed = False)
        ext_node = model.root_for_part(ext_part)
        model.create_reciprocal_relationship(root_node,
                                             'mlToExternalPartProxy',
                                             ext_node,
                                             'externalPartProxyToMl',
                                             avoid_duplicates = False)


def _copy_referenced_parts(model, other_model, realization, consolidate, force, cut_refs_to_uuids, cut_node_types,
                           self_h5_file_name, h5_uuid, other_h5_file_name, root_node):
    for ref_node in rqet.list_obj_references(root_node):
        resident_referred_node = None
        if consolidate:
            resident_referred_node = model.referenced_node(ref_node, consolidate = True)
        if force:
            continue
        if resident_referred_node is None:
            referred_node = other_model.referenced_node(ref_node)
            if referred_node is None:
                log.warning('referred node not found in other model for ' +
                            f'{rqet.find_tag_text(ref_node, "Title")}; ' +
                            f'uuid: {rqet.find_tag_text(ref_node, "UUID")}')
            else:
                referred_part = rqet.part_name_for_part_root(referred_node)
                if other_model.type_of_part(referred_part) == 'obj_EpcExternalPartReference':
                    continue
                if referred_part in model.list_of_parts():
                    continue
                _copy_part_from_other_model(model,
                                            other_model,
                                            referred_part,
                                            realization = realization,
                                            consolidate = consolidate,
                                            force = force,
                                            cut_refs_to_uuids = cut_refs_to_uuids,
                                            cut_node_types = cut_node_types,
                                            self_h5_file_name = self_h5_file_name,
                                            h5_uuid = h5_uuid,
                                            other_h5_file_name = other_h5_file_name)


def _copy_relationships_for_present_targets(model, other_model, consolidate, force, resident_uuid, root_node):
    for source_flag in [True, False]:
        other_related_parts = other_model.parts_list_filtered_by_related_uuid(other_model.list_of_parts(),
                                                                              resident_uuid,
                                                                              uuid_is_source = source_flag)
        for related_part in other_related_parts:
            # log.debug('considering relationship with: ' + str(related_part))
            if not force and (related_part in model.parts_forest):
                resident_related_part = related_part
            else:
                # log.warning('skipping relationship between ' + str(part) + ' and ' + str(related_part))
                if consolidate:
                    resident_related_uuid = model.consolidation.equivalent_uuid_for_part(related_part,
                                                                                         immigrant_model = other_model)
                    if resident_related_uuid is None:
                        continue
                    resident_related_part = rqet.part_name_for_object(other_model.type_of_part(related_part),
                                                                      resident_related_uuid)
                    if resident_related_part is None:
                        continue
                else:
                    continue
            if not force and resident_related_part in model.parts_list_filtered_by_related_uuid(
                    model.list_of_parts(), resident_uuid):
                continue
            related_node = model.root_for_part(resident_related_part)
            assert related_node is not None

            if source_flag:
                sd_a, sd_b = 'sourceObject', 'destinationObject'
            else:
                sd_b, sd_a = 'sourceObject', 'destinationObject'
            model.create_reciprocal_relationship(root_node, sd_a, related_node, sd_b)


def _copy_all_parts_from_other_model(model, other_model, realization = None, consolidate = True):
    """Fully copies parts in from another model, with referenced parts, hdf5 data and relationships."""

    assert other_model is not None and other_model is not model

    other_parts_list = other_model.parts()
    if not other_parts_list:
        log.warning('no parts found in other model for merging')
        return

    if consolidate:
        other_parts_list = cons.sort_parts_list(other_model, other_parts_list)

    self_h5_file_name = model.h5_file_name(file_must_exist = False)
    self_h5_uuid = model.h5_uuid()
    other_h5_file_name = other_model.h5_file_name()
    for part in other_parts_list:
        _copy_part_from_other_model(model,
                                    other_model,
                                    part,
                                    realization = realization,
                                    consolidate = consolidate,
                                    self_h5_file_name = self_h5_file_name,
                                    h5_uuid = self_h5_uuid,
                                    other_h5_file_name = other_h5_file_name)

    if consolidate and model.consolidation is not None:
        model.consolidation.check_map_integrity()


def _set_uuid_to_part(model, part_name):
    """Adds an entry to the dictionary mapping from uuid to part name."""

    uuid = rqet.uuid_in_part_name(part_name)
    model.uuid_part_dict[bu.uuid_as_int(uuid)] = part_name


def _del_uuid_to_part(model, part_name):
    """Deletes an entry from the dictionary mapping from uuid to part name."""

    uuid = rqet.uuid_in_part_name(part_name)
    try:
        del model.uuid_part_dict[bu.uuid_as_int(uuid)]
    except Exception:
        pass
