"""Script to construct JSON database of UOMs and property kinds.

Expects two files to exist in the same folder:
  - Energistics_Unit_of_Measure_Dictionary_V1.0.xml    (from units of measure standard)
  - Properties.xsd                                     (from resqml standard)

Creates properties.json containing parsed uoms and property kinds.
"""

import json
import math as maths
from pathlib import Path

from lxml import etree


def main():
    """Main function for parsing schema definition files and generating json file."""

    uom_dict_xml_path = Path(__file__).parent / 'Energistics_Unit_of_Measure_Dictionary_V1.0.xml'
    properties_xsd_path = Path(__file__).parent / 'Properties.xsd'
    json_path = Path(__file__).parent / 'properties.json'

    # Load xml
    assert uom_dict_xml_path.exists()
    assert properties_xsd_path.exists()
    uom_dict_root = etree.parse(str(uom_dict_xml_path)).getroot()
    properties_xsd_root = etree.parse(str(properties_xsd_path)).getroot()

    # Parse into memory
    data = dict(dimensions = parse_dimensions(uom_dict_root),
                quantities = parse_quantities(uom_dict_root),
                units = parse_units(uom_dict_root),
                prefixes = parse_prefixes(uom_dict_root),
                property_kinds = parse_property_kinds(properties_xsd_root))

    # Save to JSON
    with open(json_path, 'w', encoding = 'utf-8') as f:
        json.dump(data, f, ensure_ascii = False, indent = 2)


def parse_dimensions(root):
    """Returns a dictionary of dimensionalities."""

    dims = {}
    for node in root.find("{*}unitDimensionSet"):

        dimension = node.find('{*}dimension').text
        dims[dimension] = dict(
            name = node.find('{*}name').text,
            baseForConversion = node.find('{*}baseForConversion').text,
            canonicalUnit = node.find('{*}canonicalUnit').text,
        )
    return dims


def parse_quantities(root):
    """Returns a dictionary of quanitity classes."""

    quantities = {}
    for node in root.find("{*}quantityClassSet"):
        name = node.find('{*}name').text
        quantities[name] = dict(
            dimension = node.find('{*}dimension').text,
            baseForConversion = node.find('{*}baseForConversion').text,
        )
        alt = node.find('{*}alternativeBase')
        if alt is not None:
            quantities[name]['alternativeBase'] = alt.text

        members = []
        for member in node.findall('{*}memberUnit'):
            members.append(member.text)
        if members:
            quantities[name]["members"] = members

    return quantities


def parse_units(root):
    """Returns a dictionary of units of measure."""

    units = {}
    for node in root.find("{*}unitSet"):
        symbol = node.find('{*}symbol').text
        unit_dict = {}
        for key, typ in [
            ('name', str),
            ('dimension', str),
            ('isSI', as_bool),
            ('category', str),
            ('baseUnit', str),
            ('conversionRef', str),
            ('isExact', as_bool),
            ('A', as_numeric),
            ('B', as_numeric),
            ('C', as_numeric),
            ('D', as_numeric),
            ('conversionRef', str),
            ('description', str),
        ]:
            prop = node.find("{*}" + key)
            if prop is not None:
                unit_dict[key] = typ(prop.text)

        units[symbol] = unit_dict

    return units


def parse_prefixes(root):
    """Returns a dictionary of unit of measure prefixes."""

    prefixes = {}
    for node in root.find("{*}prefixSet"):
        name = node.find('{*}name').text
        prefixes[name] = dict(
            symbol = node.find('{*}symbol').text,
            multiplier = node.find('{*}multiplier').text,
        )
        common_name = node.find('{*}commonName')
        if common_name is not None:
            prefixes[name]['common_name'] = common_name.text

    return prefixes


def parse_property_kinds(root):
    """Return dict of valid RESQML property kinds.

    Dict keys are the valid property kinds, e.g. 'angle per time' Dict values are the description, which may be None
    """

    kind_list = get_nodes_with_name(root, 'ResqmlPropertyKind')[1]
    kind_dict = {}
    for child in kind_list:
        kind = child.get('value')
        try:
            doc = child[0][0].text
        except IndexError:
            doc = None
        kind_dict[kind] = doc
    return kind_dict


def get_nodes_with_name(root, name):
    """Find xml child node with given name."""
    for child in root:
        if child.get('name') == name:
            return child
    raise ValueError(f'{name} not found')


def as_bool(a_string):
    """Returns python bool value from string."""

    return a_string.lower() == "true"


def as_numeric(a_string):
    """Returns numeric (int or float) value from a string."""

    if a_string == 'PI':
        return maths.pi
    elif a_string == "2*PI":
        return 2 * maths.pi
    elif a_string == "4*PI":
        return 4 * maths.pi
    elif a_string.isdigit():
        return int(a_string)
    else:
        return float(a_string)


if __name__ == "__main__":
    main()
