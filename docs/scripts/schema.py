"""schema.py: Functions for working with Energistics schema definitions.

To run as a script, set the following 3 environment variables and a file will be created with rst-format tables
listing the elements for each obj_* class:

    EML_SCHEMA_SPECIFIC_DIR: directory containing main xsd files for the schema
    EML_SCHEMA_COMMON_DIR: directory containing common xsd files
    EML_RST_TABLE_FILE: output filename (will be overwritten) for the tables
"""

# note: some parts of this module require the main class names to begin 'obj_'
# this will need furher work for some standards other than RESQML 2.0.1

is_main = (__name__ == '__main__')

import os
# from lxml.etree import parse
from xml.etree.ElementTree import parse
import glob

# the xsd files are xml files, so we reuse resqpy xml processing functions here to look at the schema
import resqpy.olio.xml_et as rqet


def xsd_to_dicts(schema_dirs):
    """Reads schema definition xsd files and returns dictionaries.

    arguments:
        schema_dirs (list of str): directories containing the schema definition files

    returns:
        (simple_types, complex_types) each being a dictionary;
        simple types: dict of name -> (restriction_base, list of valid (value, value_doc), type_doc);
        complex types: dict of name -> (extension_base, list of elements, type_doc);
        where element is (tag_name, type, min_occurs, max_occurs)

    notes:
        restriction_base, extension_base, tag_name and type items are strings;
        type_doc and value_doc items are the human readable strings from the xsd annotation documentation fields
    """

    def _process_simple_types(schema, simple_types):
        for simpleton in rqet.list_of_tag(schema, 'simpleType'):
            name = simpleton.attrib['name']
            doc = rqet.find_nested_tags_text(simpleton, ['annotation', 'documentation'])
            restriction = rqet.find_tag(simpleton, 'restriction')
            restrict = None if restriction is None else rqet.stripped_of_prefix(restriction.attrib['base'])
            value_list = []
            enum_list = rqet.list_of_tag(restriction, 'enumeration')
            if enum_list is not None:
                for enumeration in enum_list:
                    value = enumeration.attrib['value']
                    v_doc = rqet.find_nested_tags_text(enumeration, ['annotation', 'documentation'])
                    if v_doc is not None:
                        v_doc = str(v_doc).replace('\n', ' ')
                    value_list.append((value, v_doc))
            simple_types[name] = (restrict, value_list, doc)

    def _process_complex_types(schema, complex_types):
        for completon in rqet.list_of_tag(schema, 'complexType'):
            name = completon.attrib['name']
            doc = rqet.find_nested_tags_text(completon, ['annotation', 'documentation'])
            content = rqet.find_tag(completon, 'complexContent')
            extension = rqet.find_tag(content, 'extension')
            if extension is None:
                e_base = None
            else:
                e_base = extension.attrib['base']
                content = extension
            if content is None:
                content = completon
            sequence = rqet.find_tag(content, 'sequence')
            if sequence is None:
                sequence = content
            element_list = []
            e_nodes = rqet.list_of_tag(sequence, 'element')
            if e_nodes is not None:
                for element in e_nodes:
                    e_name = element.attrib['name']
                    flavour = element.attrib['type']  # could strip prefix here?
                    min_occurs = element.get('minOccurs')
                    max_occurs = element.get('maxOccurs')
                    element_list.append((e_name, flavour, min_occurs, max_occurs))
            complex_types[name] = (e_base, element_list, doc)

    simple_types = {}
    complex_types = {}

    for directory in schema_dirs:
        xsd_list = glob.glob(os.path.join(directory, '*.xsd'))
        for xsd in xsd_list:
            xml_tree = parse(xsd)
            schema = xml_tree.getroot()
            _process_simple_types(schema, simple_types)
            _process_complex_types(schema, complex_types)

    return (simple_types, complex_types)


def expand_complex_types(complex_types):
    """Expands the elements within the complex types and returns a flattened dictionary.

    arguments:
        complex_types (dict): as returned by xsd_to_dicts()[1]

    returns:
        flattened_types: dict of name -> list of elements;
        where element is (tag_name, type, min_occurs, max_occurs)

    notes:
        the elements returned will be simple types or abstract types; an abstract type indicates
        a choice of possible data sub-structures
    """

    def _expand(key, complex_types, expansion_state, flattened_types):
        # recursively expand extension base elements
        if expansion_state[key] > 0:
            return
        expansion_state[key] = 1
        e_base, element_list, _ = complex_types[key]
        if e_base is not None:
            e_base = e_base.split(sep = ':')[-1]
            _expand(e_base, complex_types, expansion_state, flattened_types)
            assert expansion_state[e_base] != 1, f'circular reference in {e_base}'
            element_list = flattened_types[e_base] + element_list
        flattened_types[key] = element_list
        expansion_state[key] = 2

    flattened_types = {}
    expansion_state = {}  # tag -> int; 0: not expanded; 1: in progress; 2: expanded
    for key in complex_types.keys():
        expansion_state[key] = 0

    for name in complex_types.keys():
        _expand(name, complex_types, expansion_state, flattened_types)

    return flattened_types


def obj_names(complex_types):
    """Returns list of 'obj_*' high level object type names, sorted alphabetically.

    arguments:
        complex_types (dict): as returned by xsd_to_dicts()[1]

    returns:
        list of str
    """

    return sorted([k for k in complex_types.keys() if k.startswith('obj_')])


def print_flattened(complex_types, flattened_types, type_key, max_levels = 99, indent = 0):
    """Recursively prints a list of the (flattened) elements for a complex type.

    arguments:
        flattened_types (dict): as returned by expand_complex_types()
        type_key (str): the name of the complex type for which an element list print is required
        max_levels (int): the maximum number of nested levels to show in printout
        indent (int, default 0): initial indentation level
    """

    if indent > max_levels:
        return
    for e in flattened_types[type_key]:
        print(' ' * 3 * indent, f'{e[0]}: ({e[1]}) ', end = '')
        if e[2] == '0' and e[3] == '1':
            print('optional')
        elif e[3] == 'unbounded':
            print(f'{e[2]} or more')
        elif e[2] == '1' and e[3] == '1':
            print()
        else:
            print(f'{e[2]} to {e[3]}')
        if e[1].startswith('resqml:'):
            ref_key = e[1].split(':')[-1]
            if ref_key in complex_types.keys():
                print_flattened(complex_types, flattened_types, ref_key, max_levels = max_levels, indent = indent + 1)


def manifestations(complex_types, abstract_type, is_obj = False):
    """Returns a list of type names which are manifestations of a given abstract type.

    arguments:
        complex_types (dict): as returned by xsd_to_dicts()[1]
        abstract_type (str): the name of the abstract type of interest
        is_obj (bool, default False): if True, returns a list of obj_* types that are direct
            extensions of the abstract type; if False, returns a list of the (non obj_*) types
            which are the alternative representations of the abstract type

    returns:
        list of str
    """

    m_list = []

    for t in complex_types.keys():
        e_base = complex_types[t][0]
        if e_base is not None and e_base.split(sep = ':')[-1] == abstract_type and t.startswith('obj_') == is_obj:
            m_list.append(t)

    return m_list


def is_abstract_type(type_name):
    """Returns True if type_name appears to be the name of an abstract type; False otherwise."""

    return type_name.startswith('Abstract')


def abstract_types(complex_types):
    """Returns list of abstract types found amongst complex types."""

    return [k for k in complex_types.keys() if is_abstract_type(k)]


def max_sizes_for_rst(flattened_types, complex_types, key = None):
    """Returns a list of 3 integers being the required column widths for rst format tables."""

    def _max_sizes(flattened_types, complex_types, key, indent = 0, max_so_far = [0, 0, 0]):
        for e in flattened_types[key]:
            s0 = 3 * indent + len(e[0])
            if s0 > max_so_far[0]:
                max_so_far[0] = s0
            s1 = len(e[1])
            if s1 > max_so_far[1]:
                max_so_far[1] = s1
            if e[1].startswith('resqml:'):
                ref_key = e[1].split(':')[-1]
                if ref_key in complex_types:
                    _max_sizes(flattened_types, complex_types, ref_key, indent = indent + 1, max_so_far = max_so_far)
        return max_so_far

    ms = [0, 0, 10]

    key_list = obj_names(complex_types) if key is None else [key]
    for key in key_list:
        ms = _max_sizes(flattened_types, complex_types, key, max_so_far = ms)

    return ms


def write_rst_table(fp, flattened_types, complex_types, key, max_sizes, header = True, max_levels = 99, sort = False):
    """Writes an rst format table of elements for a complex type, to an already opened file."""

    def _write_bar(fp, max_sizes):
        line = '+' + '-' * max_sizes[0] + '+' + '-' * max_sizes[1] + '+' + '-' * max_sizes[2] + '+\n'
        fp.write(line)

    def _pf_table(fp, flattened_types, complex_types, key, ms, max_levels, indent = 0, sort = False):
        if indent > max_levels:
            return
        table_width = ms[0] + ms[1] + ms[2] + 4
        elements = flattened_types[key]
        if sort:
            elements = sorted(elements)
        for e in elements:
            line = '|' + '·' * 3 * indent + e[0]
            if len(line) < ms[0] + 1:
                line += ' ' * (ms[0] + 1 - len(line))
            line += '|' + e[1]
            if len(line) < ms[0] + ms[1] + 2:
                line += ' ' * (ms[0] + ms[1] + 2 - len(line))
            line += '|'
            if e[2] == '0' and e[3] == '1':
                line += 'optional'
            elif e[3] == 'unbounded':
                line += f'{e[2]} or more'
            elif e[2] == '1' and e[3] == '1':
                line += 'required'
            else:
                line += f'{e[2]} to {e[3]}'
            if len(line) < table_width - 1:
                line += ' ' * (table_width - 1 - len(line))
            line += '|\n'
            fp.write(line)
            _write_bar(fp, ms)
            if e[1].startswith('resqml:'):
                ref_key = e[1].split(':')[-1]
                if ref_key in complex_types.keys():
                    _pf_table(fp,
                              flattened_types,
                              complex_types,
                              ref_key,
                              ms,
                              max_levels = max_levels,
                              indent = indent + 1,
                              sort = sort)

    if header:
        fp.write(f'\n{key}\n')
        fp.write('-' * len(key) + '\n\n')

    _write_bar(fp, max_sizes)
    _pf_table(fp, flattened_types, complex_types, key, max_sizes, max_levels, sort = sort)


def write_all_obj_rst_tables(flattened_types, complex_types, file_name, mode = 'a', sort_elements = False):
    """Writes a sequence of tables in rst format, one for each obj_* class, to a named file."""

    max_sizes = max_sizes_for_rst(flattened_types, complex_types)

    with open(file_name, mode) as fp:
        for key in obj_names(complex_types):
            write_rst_table(fp, flattened_types, complex_types, key, max_sizes, sort = sort_elements)


# if running as a script, check and use 3 environment variable settings
if is_main:
    schema_main_dir = os.environ.get('EML_SCHEMA_SPECIFIC_DIR')
    schema_common_dir = os.environ.get('EML_SCHEMA_COMMON_DIR')
    table_file = os.environ.get('EML_RST_TABLE_FILE')
    if schema_main_dir is None or schema_common_dir is None or table_file is None:
        print('set the following 3 environment variables before running this script:')
        for env_var in ['EML_SCHEMA_SPECIFIC_DIR', 'EML_SCHEMA_COMMON_DIR', 'EML_RST_TABLE_FILE']:
            print(f'   {env_var}')
        raise ValueError('environment variable not set')
    simple_types, complex_types = xsd_to_dicts([schema_main_dir, schema_common_dir])
    flattened_types = expand_complex_types(complex_types)
    write_all_obj_rst_tables(flattened_types, complex_types, table_file, mode = 'w', sort_elements = False)
