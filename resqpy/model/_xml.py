"""_xml.py: functions to support xml creation methods in the Model class."""

import logging

log = logging.getLogger(__name__)

import getpass
import os
import warnings

import resqpy.crs as rqc
import resqpy.olio.time as time
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
from resqpy.olio.xml_namespaces import curly_namespace as ns
from resqpy.olio.xml_namespaces import namespace as ns_url
from resqpy import __version__

# following should be kept in line with major.minor tag values in repository
citation_format = f'bp:resqpy:v{__version__}'
use_version_string = False


def _create_tree_if_none(model):
    """Checks that model has an xml tree; if not, an empty tree is created; not usually called directly."""

    if model.main_tree is None:
        model.main_tree = rqet.ElementTree()
        model.modified = True


def _change_uuid_in_supporting_representation_reference(model, node, old_uuid, new_uuid, new_title = None):
    """Look for supporting representation reference using the old_uuid and replace with the new_uuid."""

    if isinstance(old_uuid, str):
        old_uuid = bu.uuid_from_string(old_uuid)
    if isinstance(new_uuid, str):
        new_uuid = bu.uuid_from_string(new_uuid)

    ref_node = rqet.find_tag(node, 'SupportingRepresentation')
    if ref_node is None:
        return False
    uuid_node = rqet.find_tag(ref_node, 'UUID')
    if uuid_node is None:
        return False
    if not bu.matching_uuids(uuid_node.text, old_uuid):
        return False

    uuid_node_int = rqet.uuid_for_part_root(node).int
    relations = model.uuid_rels_dict[uuid_node_int]
    if old_uuid.int in relations[0]:
        relations[0].remove(old_uuid.int)
    relations[0].add(new_uuid.int)

    relations = model.uuid_rels_dict[old_uuid.int]
    if uuid_node_int in relations[1]:
        relations[1].remove(uuid_node_int)

    relations = model.uuid_rels_dict[new_uuid.int]
    relations[1].add(uuid_node_int)

    uuid_node.text = str(new_uuid)
    if new_title:
        title_node = rqet.find_tag(ref_node, 'Title')
        if title_node is not None:
            title_node.text = str(new_title)
    model.set_modified()
    return True


def _create_root(model):
    """Initialises an empty main xml tree for model."""

    assert (model.main_tree is None)
    assert (model.main_root is None)
    model.main_root = rqet.Element(ns['content_types'] + 'Types')
    model.main_tree = rqet.ElementTree(element = model.main_root)


def _new_obj_node(flavour, name_space = 'resqml2', is_top_lvl_obj = True):
    """Creates a new main object element and sets attributes (does not add children)."""

    if flavour.startswith('obj_'):
        flavour = flavour[4:]

    node = rqet.Element(ns[name_space] + flavour)
    node.set('schemaVersion', '2.0')
    node.set('uuid', str(bu.new_uuid()))
    if is_top_lvl_obj:
        node.set(ns['xsi'] + 'type', ns[name_space] + 'obj_' + flavour)
    node.text = rqet.null_xml_text

    return node


def _referenced_node(model, ref_node, consolidate = False):
    """For a given xml reference node, returns the node for the object referred to, if present."""

    # log.debug(f'ref node called for: {ref_node}')
    if ref_node is None:
        return None
    # content_type = rqet.find_tag_text(ref_node, 'ContentType')
    # log.debug(f'ref node title: {rqet.citation_title_for_node(rqet.find_tag(ref_node, "Title"))}')
    uuid = bu.uuid_from_string(rqet.find_tag_text(ref_node, 'UUID'))
    # log.debug(f'ref node uuid: {uuid}')
    if uuid is None:
        return None
    # return model.root_for_part(model.parts_list_of_type(type_of_interest = content_type, uuid = uuid))
    if consolidate and model.consolidation is not None and uuid.int in model.consolidation.map:
        resident_uuid_int = model.consolidation.map[uuid.int]
        if resident_uuid_int is None:
            return None
        resident_uuid = bu.uuid_for_int(resident_uuid_int)
        node = model.root_for_part(model.part_for_uuid(resident_uuid))
        if node is not None:
            # patch resident uuid and title into ref node!
            uuid_node = rqet.find_tag(ref_node, 'UUID')
            uuid_node.text = str(resident_uuid)
            title_node = rqet.find_tag(ref_node, 'Title')
            if title_node is not None:
                title = rqet.citation_title_for_node(node)
                if title:
                    title_node.text = str(title)
    else:
        node = model.root_for_part(model.part_for_uuid(uuid))
    # log.debug(f'ref_node return node: {node}')
    return node


def _create_ref_node(model, flavour, title, uuid, content_type = None, root = None):
    """Create a reference node, optionally add to root."""

    assert uuid is not None

    if flavour.startswith('obj_'):
        flavour = flavour[4:]

    if not content_type:
        content_type = 'obj_' + flavour
    else:
        if content_type[0].isupper():
            content_type = 'obj_' + content_type

    prefix = ns['eml'] if flavour == 'HdfProxy' else ns['resqml2']
    ref_node = rqet.Element(prefix + flavour)
    ref_node.set(ns['xsi'] + 'type', ns['eml'] + 'DataObjectReference')
    ref_node.text = rqet.null_xml_text

    ct_node = rqet.SubElement(ref_node, ns['eml'] + 'ContentType')
    ct_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
    if 'EpcExternalPartReference' in content_type:
        ct_node.text = 'application/x-eml+xml;version=2.0;type=' + content_type
    else:
        ct_node.text = 'application/x-resqml+xml;version=2.0;type=' + content_type

    if not title:
        title = model.title(uuid = uuid)
        if title is None:
            title = ''
    title_node = rqet.SubElement(ref_node, ns['eml'] + 'Title')
    title_node.set(ns['xsi'] + 'type', ns['eml'] + 'DescriptionString')
    title_node.text = title

    uuid_node = rqet.SubElement(ref_node, ns['eml'] + 'UUID')
    uuid_node.set(ns['xsi'] + 'type', ns['eml'] + 'UuidString')
    uuid_node.text = str(uuid)

    if use_version_string:
        version_str = rqet.SubElement(ref_node, ns['eml'] + 'VersionString')  # I'm guessing what this is
        version_str.set(ns['xsi'] + 'type', ns['eml'] + 'NameString')
        version_str.text = bu.version_string(uuid)

    if root is not None:
        root.append(ref_node)

    return ref_node


def _uom_node(root, uom):
    """Add a generic unit of measure sub element to root."""

    assert root is not None and uom is not None and len(uom)
    # todo: could assert that uom is a valid unit of measure

    node = rqet.SubElement(root, ns['resqml2'] + 'UOM')
    node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ResqmlUom')
    node.text = uom

    return node


def _create_rels_part(model):
    """Adds a relationships reference node as a new part in the model's parts forest."""

    rels = rqet.SubElement(model.main_root, ns['content_types'] + 'Default')
    rels.set('Extension', 'rels')
    rels.set('ContentType', 'application/vnd.openxmlformats-package.relationships+xml')
    model.rels_present = True
    model.set_modified()

    return rels


def _create_citation(root = None, title = '', originator = None):
    """Creates a citation xml node and optionally appends as a child of root."""

    if title is None:
        title = ''

    citation = rqet.Element(ns['eml'] + 'Citation')
    citation.set(ns['xsi'] + 'type', ns['eml'] + 'Citation')
    citation.text = rqet.null_xml_text

    title_node = rqet.SubElement(citation, ns['eml'] + 'Title')
    title_node.set(ns['xsi'] + 'type', ns['eml'] + 'DescriptionString')
    title_node.text = str(title)

    originator_node = rqet.SubElement(citation, ns['eml'] + 'Originator')
    if originator is None:
        try:
            originator = str(getpass.getuser())
        except Exception:
            originator = 'unknown'
    originator_node.set(ns['xsi'] + 'type', ns['eml'] + 'NameString')
    originator_node.text = originator

    creation_node = rqet.SubElement(citation, ns['eml'] + 'Creation')
    creation_node.set(ns['xsi'] + 'type', ns['xsd'] + 'dateTime')
    creation_node.text = time.now()

    format_node = rqet.SubElement(citation, ns['eml'] + 'Format')
    format_node.set(ns['xsi'] + 'type', ns['eml'] + 'DescriptionString')
    if rqet.pretend_to_be_fesapi:
        format_node.text = '[F2I-CONSULTING:fesapi]'
    else:
        format_node.text = citation_format

    # todo: add optional description field

    if root is not None:
        root.append(citation)

    return citation


def _create_unknown(root = None):
    """Creates an Unknown node and optionally adds as child of root."""

    unknown = rqet.Element(ns['eml'] + 'Unknown')
    unknown.set(ns['xsi'] + 'type', ns['eml'] + 'DescriptionString')
    unknown.text = 'Unknown'
    if root is not None:
        root.append(unknown)
    return unknown


def _create_doc_props(model, add_as_part = True, root = None, originator = None):
    """Creates a document properties stub node and optionally adds as child of root and/or to parts forest."""

    dp = rqet.Element(ns['cp'] + 'coreProperties')
    dp.text = rqet.null_xml_text

    created = rqet.SubElement(dp, ns['dcterms'] + 'created')
    created.set(ns['xsi'] + 'type', ns['dcterms'] + 'W3CDTF')  # not sure of namespace here
    created.text = time.now()

    if originator is None:
        try:
            originator = str(os.getlogin())
        except Exception:
            originator = 'unknown'
    creator = rqet.SubElement(dp, ns['dc'] + 'creator')
    creator.text = originator

    ver = rqet.SubElement(dp, ns['cp'] + 'version')
    ver.text = '1.0'

    if root is not None:
        root.append(dp)
    if add_as_part:
        model.add_part('docProps', None, dp)
        if model.rels_present:
            (_, rel_tree) = model.rels_forest['_rels/.rels']
            core_rel = rqet.SubElement(rel_tree.getroot(), ns['rels'] + 'Relationship')
            core_rel.set('Id', 'CoreProperties')
            core_rel.set('Type', ns_url['rels_md'] + 'core-properties')
            core_rel.set('Target', 'docProps/core.xml')
    return dp


def _create_crs_reference(model, root = None, crs_uuid = None):
    """Creates a node refering to an existing crs node and optionally adds as child of root."""

    assert crs_uuid is not None
    crs_root = model.root_for_uuid(crs_uuid)
    assert crs_root is not None
    crs_type = model.type_of_uuid(crs_uuid)
    assert crs_type in ['obj_LocalDepth3dCrs', 'obj_LocalTime3dCrs']

    return _create_ref_node(model,
                            'LocalCrs',
                            rqet.find_nested_tags_text(crs_root, ['Citation', 'Title']),
                            crs_uuid,
                            content_type = crs_type,
                            root = root)


def _create_md_datum_reference(model, md_datum_root, root = None):
    """Creates a node refering to an existing measured depth datum and optionally adds as child of root."""

    return _create_ref_node(model,
                            'MdDatum',
                            rqet.find_nested_tags_text(md_datum_root, ['Citation', 'Title']),
                            bu.uuid_from_string(md_datum_root.attrib['uuid']),
                            content_type = 'obj_MdDatum',
                            root = root)


def _create_hdf5_ext(model,
                     add_as_part = True,
                     root = None,
                     title = 'Hdf Proxy',
                     originator = None,
                     file_name = None,
                     uuid = None,
                     discard_path = True):
    """Creates an hdf5 external node and optionally adds as child of root and/or to parts forest."""

    ext = _new_obj_node('EpcExternalPartReference', name_space = 'eml')
    assert ext is not None
    if uuid is not None:  # preserve ext uuid if supplied
        ext.set('uuid', str(uuid))

    _create_citation(root = ext, title = title, originator = originator)

    mime_type = rqet.SubElement(ext, ns['eml'] + 'MimeType')
    mime_type.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
    mime_type.text = 'application/x-hdf5'

    if root is not None:
        root.append(ext)
    if add_as_part:
        ext_uuid = bu.uuid_from_string(ext.attrib['uuid'])
        model.add_part('obj_EpcExternalPartReference', ext_uuid, ext)
        if not file_name:
            file_name = model.h5_file_name(override = 'full', file_must_exist = False)
        elif os.sep not in file_name:
            file_name = os.path.join(model.epc_directory, file_name)
        assert file_name
        log.debug(f'creating ext part for hdf5 file: {file_name}')
        model.h5_dict[ext_uuid.bytes] = file_name
        if model.main_h5_uuid is None:
            model.main_h5_uuid = ext_uuid
        if model.rels_present and file_name:
            if discard_path:  # NB. discard directory part of path; hdf5 must be in same directory as epc!
                _, file_name = os.path.split(file_name)
            (uuid, rel_tree) = model.rels_forest[rqet.rels_part_name_for_part(
                rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid))]
            assert (bu.matching_uuids(uuid, ext_uuid))
            rel_node = rqet.SubElement(rel_tree.getroot(), ns['rels'] + 'Relationship')
            rel_node.set('Id', 'Hdf5File')
            rel_node.set('Type', ns_url['rels_ext'] + 'externalResource')
            rel_node.set('Target', file_name)
            rel_node.set('TargetMode', 'External')
    return ext


def _create_hdf5_dataset_ref(model, hdf5_uuid, object_uuid, group_tail, root, title = 'Hdf Proxy'):
    """Creates a pair of nodes referencing an hdf5 dataset (array) and adds to root."""

    assert root is not None
    assert group_tail

    if group_tail[0] == '/':
        group_tail = group_tail[1:]
    if group_tail[-1] == '/':
        group_tail = group_tail[:-1]
    hdf5_path = '/RESQML/' + str(object_uuid) + '/' + group_tail

    path_node = rqet.Element(ns['eml'] + 'PathInHdfFile')
    path_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
    path_node.text = hdf5_path
    root.append(path_node)

    _create_ref_node(model, 'HdfProxy', title, hdf5_uuid, content_type = 'obj_EpcExternalPartReference', root = root)

    return path_node


def _create_supporting_representation(model,
                                      support_root = None,
                                      support_uuid = None,
                                      root = None,
                                      title = None,
                                      content_type = 'obj_IjkGridRepresentation'):
    """Craate a supporting representation reference node refering to an IjkGrid and optionally add to root."""

    assert support_root is not None or support_uuid is not None

    # todo: check that support_root is for a RESQML class that can support properties, matching content_type

    if support_root is not None:
        uuid = rqet.uuid_for_part_root(support_root)
        if uuid is not None:
            support_uuid = uuid
        if title is None:
            title = rqet.citation_title_for_node(support_root)
    assert support_uuid is not None
    if not title:
        title = model.title(uuid = support_uuid)
        if not title:
            title = 'supporting representation'

    return _create_ref_node(model,
                            'SupportingRepresentation',
                            title,
                            support_uuid,
                            content_type = content_type,
                            root = root)


def _create_source(source, root = None):
    """Create an extra meta data node holding information on the source of the data, optionally add to root."""

    emd_node = rqet.Element(ns['resqml2'] + 'ExtraMetadata')
    emd_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'NameValuePair')
    emd_node.text = rqet.null_xml_text

    name_node = rqet.SubElement(emd_node, ns['resqml2'] + 'Name')
    name_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
    name_node.text = 'source'

    value_node = rqet.SubElement(emd_node, ns['resqml2'] + 'Value')
    value_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
    value_node.text = source

    if root is not None:
        root.append(emd_node)

    return emd_node


def _create_patch(model,
                  p_uuid,
                  ext_uuid = None,
                  root = None,
                  patch_index = 0,
                  hdf5_type = 'DoubleHdf5Array',
                  xsd_type = 'double',
                  null_value = None,
                  const_value = None,
                  const_count = None,
                  points = False):
    """Create a node for a patch of values, including ref to hdf5 data set, optionally add to root."""

    if const_value is None:
        assert ext_uuid is not None
    else:
        assert const_count is not None and const_count > 0
        if hdf5_type.endswith('Hdf5Array'):
            hdf5_type = hdf5_type[:-9] + 'ConstantArray'

    lxt = str(xsd_type).lower()
    discrete = ('int' in lxt) or ('bool' in lxt)

    if points:
        assert not discrete
        patch_node = rqet.Element(ns['resqml2'] + 'PatchOfPoints')
        patch_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'PatchOfPoints')
        patch_node.text = rqet.null_xml_text
        outer_values_tag = 'Points'
        inner_values_tag = 'Coordinates'
        hdf_path_tail = 'points_patch'
    else:
        patch_node = rqet.Element(ns['resqml2'] + 'PatchOfValues')
        patch_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'PatchOfValues')
        patch_node.text = rqet.null_xml_text
        outer_values_tag = 'Values'
        inner_values_tag = 'Values'
        hdf_path_tail = 'values_patch'

    rep_patch_index = rqet.SubElement(patch_node, ns['resqml2'] + 'RepresentationPatchIndex')
    rep_patch_index.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
    rep_patch_index.text = str(patch_index)

    outer_values_node = rqet.SubElement(patch_node, ns['resqml2'] + outer_values_tag)
    outer_values_node.set(ns['xsi'] + 'type', ns['resqml2'] + hdf5_type)  # may also be constant array type
    outer_values_node.text = rqet.null_xml_text

    if discrete and const_value is None:
        if null_value is None:
            if str(xsd_type).startswith('u'):
                null_value = 4294967295  # 2^32 - 1, used as default even for 64 bit data!
            else:
                null_value = -1
        null_value_node = rqet.SubElement(outer_values_node, ns['resqml2'] + 'NullValue')
        null_value_node.set(ns['xsi'] + 'type', ns['xsd'] + xsd_type)
        null_value_node.text = str(null_value)

    if const_value is None:

        inner_values_node = rqet.SubElement(outer_values_node, ns['resqml2'] + inner_values_tag)
        inner_values_node.set(ns['xsi'] + 'type', ns['eml'] + 'Hdf5Dataset')
        inner_values_node.text = rqet.null_xml_text

        _create_hdf5_dataset_ref(model, ext_uuid, p_uuid, f'{hdf_path_tail}{patch_index}', root = inner_values_node)

    else:

        const_value_node = rqet.SubElement(outer_values_node, ns['resqml2'] + 'Value')
        const_value_node.set(ns['xsi'] + 'type', ns['xsd'] + xsd_type)
        const_value_node.text = str(const_value)

        const_count_node = rqet.SubElement(outer_values_node, ns['resqml2'] + 'Count')
        const_count_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
        const_count_node.text = str(const_count)

    if root is not None:
        root.append(patch_node)

    return patch_node


def _create_solitary_point3d(flavour, root, xyz):
    """Creates a subelement to root for a solitary point in 3D space."""

    # todo: check namespaces
    p3d = rqet.SubElement(root, ns['resqml2'] + flavour)
    p3d.set(ns['xsi'] + 'type', ns['resqml2'] + 'Point3d')
    p3d.text = rqet.null_xml_text

    for axis in range(3):
        coord_node = rqet.SubElement(p3d, ns['resqml2'] + 'Coordinate' + str(axis + 1))
        coord_node.set(ns['xsi'] + 'type', ns['xsd'] + 'double')
        coord_node.text = str(xyz[axis])

    return p3d


def _create_reciprocal_relationship(model, node_a, rel_type_a, node_b, rel_type_b, avoid_duplicates = True):
    """Adds a node to each of a pair of trees in the rels forest, to represent a two-way relationship."""

    def id_str(uuid):
        stringy = str(uuid)
        if not (rqet.pretend_to_be_fesapi or rqet.use_fesapi_quirks) or not stringy[0].isdigit():
            return stringy
        return '_' + stringy

    assert (model.rels_present)

    if node_a is None or node_b is None:
        log.error('attempt to create relationship with missing object')
        return

    uuid_a = node_a.attrib['uuid']
    obj_type_a = rqet.stripped_of_prefix(rqet.content_type(node_a.attrib[ns['xsi'] + 'type']))
    part_name_a = rqet.part_name_for_object(obj_type_a, uuid_a)
    rel_part_name_a = rqet.rels_part_name_for_part(part_name_a)
    (rel_uuid_a, rel_tree_a) = model.rels_forest[rel_part_name_a]
    rel_root_a = rel_tree_a.getroot()

    uuid_b = node_b.attrib['uuid']
    obj_type_b = rqet.stripped_of_prefix(rqet.content_type(node_b.attrib[ns['xsi'] + 'type']))
    part_name_b = rqet.part_name_for_object(obj_type_b, uuid_b)
    rel_part_name_b = rqet.rels_part_name_for_part(part_name_b)
    (rel_uuid_b, rel_tree_b) = model.rels_forest[rel_part_name_b]
    rel_root_b = rel_tree_b.getroot()

    create_a = True
    if avoid_duplicates:
        existing_rel_nodes = rqet.list_of_tag(rel_root_a, 'Relationship')
        for existing in existing_rel_nodes:
            # if (rqet.stripped_of_prefix(existing.attrib['Type']) == rel_type_a and
            #         existing.attrib['Target'] == part_name_b):
            if existing.attrib['Target'] == part_name_b:
                create_a = False
                break
    if create_a:
        rel_a = rqet.SubElement(rel_root_a, ns['rels'] + 'Relationship')
        rel_a.set('Id',
                  id_str(uuid_b))  # NB: fesapi prefixes uuid with _ for some rels only (where uuid starts with a digit)
        rel_a.set('Type', ns_url['rels_ext'] + rel_type_a)
        rel_a.set('Target', part_name_b)

    create_b = True
    if avoid_duplicates:
        existing_rel_nodes = rqet.list_of_tag(rel_root_b, 'Relationship')
        for existing in existing_rel_nodes:
            # if (rqet.stripped_of_prefix(existing.attrib['Type']) == rel_type_b and
            #         existing.attrib['Target'] == part_name_a):
            if existing.attrib['Target'] == part_name_a:
                create_b = False
                break
    if create_b:
        rel_b = rqet.SubElement(rel_root_b, ns['rels'] + 'Relationship')
        rel_b.set('Id',
                  id_str(uuid_a))  # NB: fesapi prefixes uuid with _ for some rels only (where uuid starts with a digit)
        rel_b.set('Type', ns_url['rels_ext'] + rel_type_b)
        rel_b.set('Target', part_name_a)

    if "EpcExternalPart" in rel_part_name_a or "EpcExternalPart" in rel_part_name_b:
        return

    rel_uuid_a_int = rel_uuid_a.int
    rel_uuid_b_int = rel_uuid_b.int

    value = model.uuid_rels_dict.get(rel_uuid_a_int)
    if value is not None and rel_uuid_b_int not in value[0] and rel_uuid_b_int not in value[1]:
        value[2].add(rel_uuid_b_int)

    value = model.uuid_rels_dict.get(rel_uuid_b_int)
    if value is not None and rel_uuid_a_int not in value[0] and rel_uuid_a_int not in value[1]:
        value[2].add(rel_uuid_a_int)
