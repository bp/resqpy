"""Class containing resqml stringlookup class."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

# import xml.etree.ElementTree as et

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class StringLookup(BaseResqpy):
    """Class catering for RESQML obj_StringLookupTable objects."""

    resqml_type = "StringTableLookup"

    def __init__(self,
                 parent_model,
                 uuid = None,
                 int_to_str_dict = None,
                 title = None,
                 extra_metadata = None,
                 originator = None):
        """Creates a new string lookup (RESQML obj_StringTableLookup) object.

        arguments:
           parent_model: the model to which this string lookup belongs
           uuid (optional): if present, the uuid for an exising StringTableLookup from which this object is populated
           int_to_str_dict (optional): if present, a dictionary mapping from integers to strings, used to populate the lookup;
              ignored if uuid is present
           title (string, optional): if present, is used as the citation title for the object; ignored if uuid is not None

        returns:
           the new StringLookup object

        :meta common:
        """

        self.min_index = None
        self.max_index = None
        self.str_list = []
        self.str_dict = {}
        self.stored_as_list = False
        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)
        if uuid is None:
            self.load_from_dict(int_to_str_dict)

    def _load_from_xml(self):
        root_node = self.root
        for v_node in rqet.list_of_tag(root_node, 'Value'):
            key = rqet.find_tag_int(v_node, 'Key')
            value = rqet.find_tag_text(v_node, 'Value')
            assert key not in self.str_dict, 'key value ' + str(
                key) + ' occurs more than once in string lookup table xml'
            self.str_dict[key] = value
            if self.min_index is None or key < self.min_index:
                self.min_index = key
            if self.max_index is None or key > self.max_index:
                self.max_index = key

    def load_from_dict(self, int_to_str_dict):
        """Sets the contents of this string lookup based on a dict mapping int to str."""

        if int_to_str_dict is None:
            return
        assert len(int_to_str_dict), 'empty dictionary passed to string lookup initialisation'
        self.str_dict = int_to_str_dict.copy()
        self.min_index = min(self.str_dict.keys())
        self.max_index = max(self.str_dict.keys())
        self.set_list_from_dict_conditionally()

    def as_dict(self):
        """Returns the string lookup as a python dictionary."""

        return self.str_dict

    def is_equivalent(self, other):
        """Returns True if this lookup is the same as other (apart from uuid); False otherwise."""

        if other is None:
            return False
        if self is other:
            return True
        if bu.matching_uuids(self.uuid, other.uuid):
            return True
        if self.title != other.title or self.min_index != other.min_index or self.max_index != other.max_index:
            return False
        return self.str_dict == other.str_dict

    def set_list_from_dict_conditionally(self):
        """Sets a list copy of the lookup table, which can be indexed directly, if it makes sense to do so."""

        self.str_list = []
        self.stored_as_list = False
        if self.min_index >= 0 and (self.max_index < 50 or 10 * len(self.str_dict) // self.max_index > 8):
            for key in range(self.max_index + 1):
                if key in self.str_dict:
                    self.str_list.append(self.str_dict[key])
                else:
                    self.str_list.append(None)
            self.stored_as_list = True

    def set_string(self, key, value):
        """Sets the string associated with a given integer key."""

        self.str_dict[key] = value
        limits_changed = False
        if self.min_index is None or value < self.min_index:
            self.min_index = value
            limits_changed = True
        if self.max_index is None or value > self.max_index:
            self.max_index = value
            limits_changed = True
        if self.stored_as_list:
            if limits_changed:
                self.set_list_from_dict_conditionally()
            else:
                self.str_list[key] = value

    def get_string(self, key):
        """Returns the string associated with the integer key, or None if not found.

        :meta common:
        """

        if key < self.min_index or key > self.max_index:
            return None
        if self.stored_as_list:
            return self.str_list[key]
        if key not in self.str_dict:
            return None
        return self.str_dict[key]

    def get_list(self):
        """Returns a list of values, sorted by key.

        :meta common:
        """

        if self.stored_as_list:
            return self.str_list
        return list(dict(sorted(list(self.str_dict.items()))).values())

    def length(self):
        """Returns the nominal length of the lookup table.

        :meta common:
        """

        if self.stored_as_list:
            return len(self.str_list)
        return len(self.str_dict)

    def get_index_for_string(self, string):
        """Returns the integer key for the given string (exact match required), or None if not found.

        :meta common:
        """

        if self.stored_as_list:
            try:
                index = self.str_list.index(string)
                return index
            except Exception:
                return None
        if string not in self.str_dict.values():
            return None
        for k, v in self.str_dict.items():
            if v == string:
                return k
        return None

    def create_xml(self, title = None, originator = None, add_as_part = True, reuse = True):
        """Creates an xml node for the string table lookup.

        arguments:
           title (string, optional): if present, overrides the object's title attribute to be used as citation title
           originator (string, optional): if present, used as the citation creator (otherwise login name is used)
           add_as_part (boolean, default True): if True, the property set is added to the model as a part

        :meta common:
        """

        if title:
            self.title = title

        if reuse and self.try_reuse():
            return self.root  # check for reusable (equivalent) object

        sl_node = super().create_xml(add_as_part = False, originator = originator)

        for k, v in self.str_dict.items():

            pair_node = rqet.SubElement(sl_node, ns['resqml2'] + 'Value')
            pair_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'StringLookup')
            pair_node.text = rqet.null_xml_text

            key_node = rqet.SubElement(pair_node, ns['resqml2'] + 'Key')
            key_node.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
            key_node.text = str(k)

            value_node = rqet.SubElement(pair_node, ns['resqml2'] + 'Value')
            value_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
            value_node.text = str(v)

        if add_as_part:
            self.model.add_part('obj_StringTableLookup', self.uuid, sl_node)

        return sl_node
