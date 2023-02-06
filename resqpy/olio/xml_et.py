"""xml_et.py: Resqml xml element tree utilities module."""

import logging

log = logging.getLogger(__name__)

import os

# import xml element tree parse method and classes here to allow single point for switching between lxml and etree
# alternative to lxml.etree: xml.etree.ElementTree
from lxml.etree import (  # type: ignore
    Element, ElementTree, SubElement, _Element,  # noqa
    parse)

import resqpy.olio.uuid as bu
from resqpy.olio.xml_namespaces import curly_namespace as cns
from resqpy.olio.xml_namespaces import inverse_namespace as inv_ns
from resqpy.olio.xml_namespaces import namespace as ns

pretend_to_be_fesapi = False
use_fesapi_quirks = True
use_tabs = False

if use_fesapi_quirks:
    null_xml_text = '\n'
else:
    null_xml_text = ''


def strip_path(full_path):
    """Returns the filename part of full_path with any directory path removed.

    :meta private:
    """

    return os.path.basename(full_path)


def stripped_of_prefix(s):
    """Returns a simplified version of an xml tag or other str with any {xsd defining prefix} stripped off."""

    if s is None:
        return None
    p = s.rfind('}')
    if p >= 0:
        return s[p + 1:]
    return s[s.rfind(':') + 1:]


def colon_prefixed(curly_prefixed):
    """Returns a version of an xml tag with {url} prefix replaced with nsi: equivalent; also returns the nsi prefix."""

    if not curly_prefixed:
        return None, None
    if curly_prefixed[0] != '{':
        colon = curly_prefixed.find(':')
        if colon < 0:
            return curly_prefixed, None
        return curly_prefixed, curly_prefixed[:colon]
    pre_end = curly_prefixed.rfind('}')
    try:
        pre_colon = inv_ns[curly_prefixed[1:pre_end]]
    except Exception:
        return curly_prefixed, None
    return pre_colon + ':' + curly_prefixed[pre_end + 1:], pre_colon


def find_tag(root, tag_name, must_exist = False):
    """Finds the first child in xml node with a (prefix-stripped) tag matching given tag name."""

    if root is None:
        return None
    for child in root:
        if stripped_of_prefix(child.tag) == tag_name:
            return child
    if must_exist:
        raise ValueError(f"Expected tag {tag_name} not found in root {root}")
    return None


def find_tag_text(root, tag_name, must_exist = False):
    """Finds the first child in xml node with a tag matching given tag name; returns stripped text field."""

    if root is None:
        return None
    for child in root:
        if stripped_of_prefix(child.tag) == tag_name:
            return node_text(child)
    if must_exist:
        raise ValueError(f"Expected tag {tag_name} not found in root {root}")
    return None


def find_tag_bool(root, tag_name, must_exist = False):
    """Finds the first child in xml node with a tag matching given tag name; returns stripped text field as bool."""

    if root is None:
        return None
    for child in root:
        if stripped_of_prefix(child.tag) == tag_name:
            return node_bool(child)
    if must_exist:
        raise ValueError(f"Expected tag {tag_name} not found in root {root}")
    return None


def find_tag_int(root, tag_name, must_exist = False):
    """Finds the first child in xml node with a tag matching given tag name; returns stripped text field as int."""

    if root is None:
        return None
    for child in root:
        if stripped_of_prefix(child.tag) == tag_name:
            return node_int(child)
    if must_exist:
        raise ValueError(f"Expected tag {tag_name} not found in root {root}")
    return None


def find_tag_float(root, tag_name, must_exist = False):
    """Finds the first child in xml node with a tag matching given tag name; returns stripped text field as float."""

    if root is None:
        return None
    for child in root:
        if stripped_of_prefix(child.tag) == tag_name:
            return node_float(child)
    if must_exist:
        raise ValueError(f"Expected tag {tag_name} not found in root {root}")
    return None


def find_nested_tags(root, tag_list):
    """Follows a list of tags in a nested xml hierarchy, returning the node at the deepest level."""

    if not tag_list:
        return None
    head = find_tag(root, tag_list[0])
    if head is None:
        return None
    if len(tag_list) == 1:
        return head
    return find_nested_tags(head, tag_list[1:])


def find_nested_tags_cast(root, tag_list, dtype = None):
    """Return value of nested tags as desired dtype.

    Follows a list of tags in a nested xml hierarchy, returning the stripped text of the node at the deepest level.
    """

    cast_func = {
        int: node_int,
        float: node_float,
        bool: node_bool,
        str: node_text,
        None: lambda x: x,
    }[dtype]

    node = find_nested_tags(root, tag_list)
    return cast_func(node)


def find_nested_tags_text(root, tag_list):
    """Return stripped text of node at deepest level of xml hierarchy.
    
    arguments:
        tag_list (list of str): list of tags in a nested xml hierarchy
    """

    node = find_nested_tags(root, tag_list)
    return node_text(node)


def find_nested_tags_bool(root, tag_list):
    """Return stripped text of node at deepest level of xml hierarchy as a bool.
    
    arguments:
        tag_list (list of str): list of tags in a nested xml hierarchy
    """
    node = find_nested_tags(root, tag_list)
    return node_bool(node)


def find_nested_tags_int(root, tag_list):
    """Return stripped text of node at deepest level of xml hierarchy as an int.
    
    arguments:
        tag_list (list of str): list of tags in a nested xml hierarchy
    """
    node = find_nested_tags(root, tag_list)
    return node_int(node)


def find_nested_tags_float(root, tag_list):
    """Return stripped text of node at deepest level of xml hierarchy as a float.
    
    arguments:
        tag_list (list of str): list of tags in a nested xml hierarchy
    """
    node = find_nested_tags(root, tag_list)
    return node_float(node)


def count_tag(root, tag_name):
    """Returns the number of children in xml node with a (prefix-stripped) tag matching given tag name."""

    if root is None:
        return None
    count = 0
    for child in root:
        if stripped_of_prefix(child.tag) == tag_name:
            count += 1
    return count


def list_of_tag(root, tag_name):
    """Returns a list of children in xml node with a (prefix-stripped) tag matching given tag name."""

    if root is None:
        return None
    results = []
    for child in root:
        if stripped_of_prefix(child.tag) == tag_name:
            results.append(child)
    return results


def list_of_descendant_tag(root, tag_name):
    """Returns a list of descendants in xml node tree with a (prefix-stripped) tag matching given tag name."""

    if root is None:
        return None
    results = []
    for child in root.iterdescendants():
        if stripped_of_prefix(child.tag) == tag_name:
            results.append(child)
    return results


def list_obj_references(root, skip_hdf5 = True):
    """Returns list of nodes of type DataObjectReference."""

    if root is None:
        return None
    results = []
    if node_type(root) == 'DataObjectReference' and not (skip_hdf5 and stripped_of_prefix(root.tag) == 'HdfProxy'):
        results.append(root)
    for child in root:
        results += list_obj_references(child, skip_hdf5 = skip_hdf5)
    return results


def cut_obj_references(root, uuids_to_be_cut):
    """Deletes any object reference nodes to uuids in given list."""

    if root is None or not uuids_to_be_cut:
        return
    for child in root:
        if node_type(child) == 'DataObjectReference':
            referred_uuid = bu.uuid_from_string(find_tag_text(child, 'UUID', must_exist = True))
            for cut_uuid in uuids_to_be_cut:
                if bu.matching_uuids(referred_uuid, cut_uuid):
                    root.remove(child)
                    break
        else:
            cut_obj_references(child, uuids_to_be_cut)


def cut_nodes_of_types(root, types_to_be_cut):
    """Deletes any nodes of a type matching one in the given list."""

    if root is None or not types_to_be_cut:
        return
    for child in root:
        if node_type(child) in types_to_be_cut:
            root.remove(child)  # hope this doesn't mess up the iteration
        else:
            cut_nodes_of_types(child, types_to_be_cut)


def cut_extra_metadata(root):
    """Removes all the extra metadata children under root node."""

    for child in root:
        if child.tag == 'ExtraMetadata':
            root.remove(child)


def content_type(content_type_str):
    """Returns the actual type, as embedded in an xml ContentType attribute; application and version are disregarded."""

    if content_type_str is None:
        return None
    if 'type=' in content_type_str:
        return content_type_str[content_type_str.rfind('type=') + 5:]

    #   if ':' in content_type_str:
    #      return content_type_str[content_type_str.rfind(':') + 1:]
    return content_type_str


def node_type(node, is_rels = False, strip_obj = False):
    """Returns the type as held in attributes of xml node; defining authority is stripped out."""

    result = None
    if node is None:
        return None
    if is_rels:
        if 'Type' not in node.attrib.keys():
            return None
        type_str = node.attrib['Type']
        result = type_str[type_str.rfind('/') + 1:]
    else:
        for key in node.attrib.keys():
            if stripped_of_prefix(key) == 'type':
                #           type_str = node.attrib[key]
                #           return type_str[type_str.rfind(':') + 1:]
                type_str = stripped_of_prefix(node.attrib[key])
                result = type_str
    if result and strip_obj and result.startswith('obj_'):
        result = result[4:]
    return result


def print_xml_tree(root,
                   level = 0,
                   max_level = None,
                   strip_tag_refs = True,
                   to_log = False,
                   log_level = None,
                   max_lines = 0,
                   line_count = 0):
    """Print an xml tree in an indented semi-readable format; return accumulated number of lines."""

    if root is None or (max_level is not None and level > max_level):
        return line_count
    if log_level is not None:
        to_log = True
    if log_level is None or log_level == 'info':
        print_fn = log.info
    elif log_level == 'debug':
        print_fn = log.debug
    elif log_level in ['warn', 'warning']:
        print_fn = log.warning
    elif log_level == 'error':
        print_fn = log.error
    else:
        print_fn = log.critical
    if max_lines and line_count >= max_lines:
        if line_count == max_lines:
            message = '(...xml tree output truncated after ' + str(max_lines) + ' lines)'
            if to_log:
                print_fn(message)
            else:
                print(message)
            line_count += 1
        return line_count

    value = root.text
    type_attr = None
    if strip_tag_refs:
        tag = stripped_of_prefix(root.tag)
        attrib_dict = {}
        for key in root.attrib.keys():
            stripped_key = stripped_of_prefix(key)
            attrib_dict[stripped_key] = root.attrib[key]
            if stripped_key == 'type':
                type_attr = stripped_of_prefix(root.attrib[key])
    else:
        tag = root.tag
        attrib_dict = root.attrib
        for key in root.attrib.keys():
            if stripped_of_prefix(key) == 'type':
                type_attr = root.attrib[key]
    if to_log:
        message = (3 * level * ' ') + tag + ' # ' + str(attrib_dict)
        if type_attr is not None:
            message += ' ## ' + type_attr
        if value is not None:
            message += ' ### ' + root.text.replace('\n', '\\n')
        print_fn(message)
    else:
        print((3 * level * ' ') + tag, '#', attrib_dict, end = ' ')
        if type_attr is not None:
            print('##', type_attr, end = ' ')
        if value is not None:
            print('###', root.text.replace('\n', '\\n'), end = '')
        print('')
    line_count += 1
    for child in root:
        line_count = print_xml_tree(child,
                                    level = level + 1,
                                    max_level = max_level,
                                    strip_tag_refs = strip_tag_refs,
                                    to_log = to_log,
                                    log_level = log_level,
                                    max_lines = max_lines,
                                    line_count = line_count)
        if line_count > max_lines:
            break
    return line_count


def uuid_in_part_name(part_name):
    """Returns uuid as embedded in part name."""

    # This might not always work
    if part_name is None:
        return None
    if part_name.endswith('.xml') and len(part_name) >= 40:
        return bu.uuid_from_string(part_name[-40:-4])
    elif part_name.endswith('.xml.rels') and len(part_name) >= 45:
        return bu.uuid_from_string(part_name[-45:-9])
    return None


def part_name_for_object(obj_type, uuid, prefixed = False, epc_subdir = None):
    """Returns the standard part name comprised of the object type, uuid and .xml extension."""

    if prefixed and (pretend_to_be_fesapi or use_fesapi_quirks) and obj_type[0] != '/':
        prefix = '/'
    else:
        prefix = ''
    if not obj_type.startswith('obj_'):
        prefix += 'obj_'
    if epc_subdir:
        if not epc_subdir.endswith('/'):
            epc_subdir += '/'
        prefix = epc_subdir + prefix
    return prefix + obj_type + '_' + str(uuid) + '.xml'


def rels_part_name_for_part(part_name):
    """Returns the paired relationships part name for the given part name."""

    pn = stripped_of_prefix(part_name)
    if pn is None or len(pn) == 0:
        return None
    dir_place = pn.rfind('/')
    if dir_place == -1:
        return '_rels/' + pn + '.rels'
    if dir_place == 0:
        return '_rels' + pn + '.rels'
    if dir_place == len(pn) - 1:
        return None
    return pn[:dir_place + 1] + '_rels' + pn[dir_place:] + '.rels'


def uuid_for_part_root(root):
    """Returns uuid as stored in xml attribs for root."""

    if root is None:
        return None
    uuid_str = root.attrib.get('uuid')
    if not uuid_str:
        return None
    return bu.uuid_from_string(uuid_str)


def patch_uuid_in_part_root(root, uuid):
    """Returns modified part name with uuid swapped to uuid argument; root attrib is also changed."""

    if root is None or uuid is None:
        return None
    # This might not always work
    root.attrib['uuid'] = str(uuid)
    return part_name_for_part_root(root)


def part_name_for_part_root(root, is_rels = False, epc_subdir = None):
    """Returns the part name given the root node for the part's xml."""

    if root is None:
        return None
    obj_type = node_type(root, is_rels = is_rels)
    uuid = uuid_for_part_root(root)
    if obj_type is None or uuid is None:
        return None
    return part_name_for_object(obj_type, uuid, epc_subdir = epc_subdir)


# the next two functions aren't really much to do with the xml element tree


def find_in_ordered_data(value, array_1d):
    """Returns the index in the ordered list-like array of value; or None if not present."""

    def find_in_subset(value, array_1d, start, end):
        # recursive binary split
        if start >= end:
            return None
        mid = start + (end - start) // 2
        sample = array_1d[mid]
        if sample == value:
            while mid > 0 and array_1d[mid - 1] == value:
                mid -= 1
            return mid
        if sample > value:
            return find_in_subset(value, array_1d, start, mid)
        if mid == start:
            mid += 1
        return find_in_subset(value, array_1d, mid, end)

    return find_in_subset(value, array_1d, 0, len(array_1d))


def simplified_data_type(array_dtype):
    """Returns a simplified string version of the elemental data type (typically for a numpy or hdf5 array)."""

    str_dtype = str(array_dtype)
    if str_dtype.startswith('int'):
        return 'int'
    if str_dtype.startswith('float') or str_dtype.startswith('real'):
        return 'float'
    if str_dtype.startswith('bool'):
        return 'bool'
    return str_dtype


# following functions mostly previously in resqml_print.py


def bool_from_text(text):
    """Returns boolean value for string 'true' or 'false'; anything else results in None."""

    if text is None:
        return None
    if text.strip().lower() == 'true':
        return True
    if text.strip().lower() == 'false':
        return False
    return None


def node_text(node, unknown_if_none = False):
    """Returns stripped node text or 'unknown' if node is None or text is blank or newline."""

    if node is None or node.text is None:
        return 'unknown' if unknown_if_none else None
    text = node.text.strip()
    if len(text):
        return text
    return 'unknown' if unknown_if_none else None


def node_bool(node):
    """Returns stripped node text as bool, or None."""

    if node is None:
        return None
    return bool_from_text(node.text)


def node_int(node):
    """Returns stripped node text as int, or None."""

    if node is None:
        return None
    text = node.text.strip()
    if text.lower() == 'none':
        return None
    if len(text):
        return int(text)
    return None


def node_float(node):
    """Returns stripped node text as float, or None."""

    if node is None:
        return None
    text = node.text.strip()
    if text.lower() == 'none':
        return None
    if len(text):
        return float(text)
    return None


def length_units_from_node(node):
    """Returns standard length units string based on node text, or 'unknown'."""

    if node is None or node.text == '' or node.text == '\n':
        return 'unknown'
    else:
        return node.text.strip()


def time_units_from_node(node):
    """Returns standard time units string based on node text, or 'unknown'."""

    if node is None or node.text == '' or node.text == '\n':
        return 'unknown'
    else:
        return node.text.strip()


def xyz_handedness(xy_axes: str, z_inc_down: bool):
    """Return xyz true handedness as 'left', 'right' or 'unknown'."""

    if xy_axes is None or z_inc_down is None:
        return 'unknown'
    xy_axes_split = xy_axes.lower().split()
    if len(xy_axes_split) != 2:
        return 'unknown'
    if xy_axes_split not in [
        ['easting', 'northing'],
        ['northing', 'easting'],  # only these 6 options allowed in resqml
        ['westing', 'southing'],
        ['southing', 'westing'],
        ['northing', 'westing'],
        ['westing', 'northing']
    ]:
        return 'unknown'
    right_handed = z_inc_down
    if xy_axes_split in [['easting', 'northing'], ['westing', 'southing'], ['northing', 'westing']]:
        right_handed = not right_handed
    if right_handed:
        return 'right'
    else:
        return 'left'


def ijk_handedness(geom_node):
    """Returns ijk true handedness as 'left', 'right' or 'unknown'.
    
    arguments:
        geom_node: GridIsRightHanded node in grid geometry node.
    """

    if geom_node is None:
        return 'unknown'
    right_handed = bool_from_text(node_text(find_tag(geom_node, 'GridIsRighthanded')))
    if right_handed is None:
        return 'unknown'
    if right_handed:
        return 'right'
    return 'left'


def citation_title_for_node(node):
    """Looks for a citation node as a child of node and returns the title text."""

    return find_nested_tags_text(node, ['Citation', 'Title'])


def creation_date_for_node(node):
    """Looks for a citation node as a child of node and returns the creation (date-time) text."""

    return find_nested_tags_text(node, ['Citation', 'Creation'])


def write_xml_node(xml_fp, root, level = 0, namespace_keys = []):
    """Recursively write an xml node to an open file; return number of nodes written."""

    def _escaped_text(text):
        # todo: include quotes if needed
        e = ''
        for ch in str(text):
            if ch == '<':
                e += '&lt;'
            elif ch == '>':
                e += '&gt;'
            elif ch == '&':
                e += '&amp;'
            else:
                e += ch
        return e

    if root is None:
        return 0

    ns_keys = namespace_keys.copy()

    type_attr = None
    tag, pre_colon = colon_prefixed(root.tag)

    if pre_colon in ['content_types', 'rels']:
        tag = tag[len(pre_colon) + 1:]
        ct_special = True
    else:
        ct_special = False

    if use_tabs:
        line = (level * '\t')
    else:
        line = (3 * level * ' ')
    line += '<' + tag  # todo: if any tags involve special characters, use _escaped_text(tag)
    if pre_colon and pre_colon not in ns_keys:
        line += ' xmlns'
        if not ct_special:
            line += ':' + pre_colon
        line += '="' + ns[pre_colon] + '"'
        ns_keys.append(pre_colon)

    attrib_ns_list = []
    attrib_list = []
    type_pre_colon = None
    for key in root.attrib.keys():
        colon_attrib_key, pre_colon_attrib = colon_prefixed(key)
        if pre_colon_attrib and pre_colon_attrib not in ns_keys:
            attrib_ns_list.append(pre_colon_attrib)
        if stripped_of_prefix(key) == 'type':
            type_attr, type_pre_colon = colon_prefixed(root.attrib[key])
            attrib_list.append(colon_attrib_key + '="' + type_attr + '"')
        elif ct_special and colon_attrib_key == 'PartName' and root.attrib[key].startswith('obj_'):
            attrib_list.append(colon_attrib_key + '="/' + root.attrib[key] + '"')
        else:
            attrib_list.append(colon_attrib_key + '="' + root.attrib[key] + '"')

    for attrib_ns in attrib_ns_list:
        line += ' xmlns:' + attrib_ns + '="' + ns[attrib_ns] + '"'
        ns_keys.append(attrib_ns)
    if type_pre_colon and type_pre_colon not in ns_keys:  # must be included in the local xml line?
        line += ' xmlns:' + type_pre_colon + '="' + ns[type_pre_colon] + '"'
        ns_keys.append(type_pre_colon)

    for attrib in attrib_list:
        line += ' ' + attrib

    node_count = 1

    if ct_special and len(root) == 0:

        line += '/>\n'
        xml_fp.write(line.encode())
    #      print(line, end = '') # debug

    else:

        line += '>'

        if root.text and not root.text.isspace():
            line += _escaped_text(root.text)

        xml_fp.write(line.encode())
        #      print(line, end = '') # debug
        indentation = ''

        if len(root):
            xml_fp.write(b'\n')
            #         print()  # debug
            if use_tabs:
                indentation = (level * '\t')
            else:
                indentation = (3 * level * ' ')
            for child in root:
                node_count += write_xml_node(xml_fp, child, level = level + 1, namespace_keys = ns_keys)

        line = indentation + '</' + tag + '>\n'
        xml_fp.write(line.encode())

    #      print(line, end = '') # debug

    return node_count


def write_xml(xml_fp, tree, standalone = None):
    """Write an xml tree to file in an indented format; gSOAP/FESAPI compatible; return number of nodes written."""

    #   print('-------------------------------------------------------------------------------------') # debug
    line = '<?xml version="1.0" encoding="UTF-8"'
    if standalone is not None:
        line += ' standalone="' + standalone + '"'
    line += '?>\n'
    xml_fp.write(line.encode())
    #   print(line, end = '')  # debug
    nodes = write_xml_node(xml_fp, tree.getroot())

    return nodes


def load_metadata_from_xml(node):
    """Loads the ExtraMetaData stored in a RESQML part as a dictionary."""

    if node is None:
        return None
    extra_metadata = {}
    meta_nodes = list_of_tag(node, 'ExtraMetadata')
    for meta in meta_nodes:
        name = find_tag_text(meta, 'Name')
        value = find_tag_text(meta, 'Value')
        extra_metadata[name] = value
    return extra_metadata


def create_metadata_xml(node, extra_metadata):
    """Writes the xml for the given metadata dictionary."""

    if extra_metadata:
        for data in extra_metadata.keys():
            metadata = SubElement(node, cns['resqml2'] + 'ExtraMetadata')
            metadata.set(cns['xsi'] + 'type', cns['resqml2'] + 'NameValuePair')
            metadata.text = null_xml_text

            name = SubElement(metadata, cns['resqml2'] + 'Name')
            name.set(cns['xsi'] + 'type', cns['xsd'] + 'string')
            name.text = str(data)

            value = SubElement(metadata, cns['resqml2'] + 'Value')
            value.set(cns['xsi'] + 'type', cns['xsd'] + 'string')
            value.text = str(extra_metadata[data])

    return node


def is_node(obj):
    """Returns True if type of object is element tree node; False otherwise."""

    # note: only tested for lxml

    return type(obj) is _Element or type(obj) is Element
