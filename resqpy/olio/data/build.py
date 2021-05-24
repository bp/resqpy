""" Script to construct JSON database of UOMs and property kinds.

Expects Properties.xsd to exist in the same folder.
Creates properties.json containing parsed uoms and property kinds.
"""

from pathlib import Path
import lxml.etree
import json


def _get_node(root, name):
   """ Find xml child node with given name """
   for child in root:
      if child.get('name') == name:
         return child
   raise ValueError(f'{name} not found')


def parse_property_kinds(root):
   """ Return dict of valid RESQML property kinds

   Dict keys are the valid property kinds, e.g. 'angle per time'
   Dict values are the description, which may be None
   """
   kind_list = _get_node(root, 'ResqmlPropertyKind')[1]

   kind_dict = {}
   for child in kind_list:
      kind = child.get('value')
      try:
         doc = child[0][0].text
      except IndexError:
         doc = None
      kind_dict[kind] = doc
   return kind_dict


def main():
   xsd_path = Path(__file__).parent / 'Properties.xsd'
   json_path = Path(__file__).parent / 'properties.json'

   # Parse xsd
   assert xsd_path.exists()
   tree = lxml.etree.parse(str(xsd_path))
   root = tree.getroot()

   # Extract uoms
   uom_list = _get_node(root, 'ResqmlUom')[0]
   uoms = sorted([item.get('value') for item in uom_list])

   # Extract property kinds as a dict
   property_kinds = parse_property_kinds(root)

   # Save to json
   data = dict(uoms=uoms, property_kinds=property_kinds)
   with open(json_path, 'w', encoding='utf-8') as f:
      json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
   main()
