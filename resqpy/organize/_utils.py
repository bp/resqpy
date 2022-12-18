"""Helper functions for RESQML Feature and Interpretation classes."""

import resqpy.olio.xml_et as rqet
from resqpy.olio.xml_namespaces import curly_namespace as ns


def alias_for_attribute(attribute_name):
    """Return an attribute that is a direct alias for an existing attribute."""

    def fget(self):
        return getattr(self, attribute_name)

    def fset(self, value):
        return setattr(self, attribute_name, value)

    return property(fget, fset, doc = f"Alias for {attribute_name}")


def equivalent_extra_metadata(a, b):
    """Returns True if the two objects have identical extra metadata"""
    a_has = hasattr(a, 'extra_metadata')
    b_has = hasattr(b, 'extra_metadata')
    if a_has:
        a_em = a.extra_metadata
        a_has = len(a_em) > 0
    else:
        a_em = rqet.load_metadata_from_xml(a.root)
        a_has = a_em is not None and len(a_em) > 0
    if b_has:
        b_em = b.extra_metadata
        b_has = len(b_em) > 0
    else:
        b_em = rqet.load_metadata_from_xml(b.root)
        b_has = b_em is not None and len(b_em) > 0
    if a_has != b_has:
        return False
    if not a_has:
        return True
    return a_em == b_em


def extract_has_occurred_during(parent_node, tag = 'HasOccuredDuring'):  # RESQML Occured (stet)
    """Extracts UUIDs of chrono bottom and top from xml for has occurred during sub-node, or (None, None)."""
    hod_node = rqet.find_tag(parent_node, tag)
    if hod_node is None:
        return (None, None)
    else:
        return (rqet.find_nested_tags_text(hod_node, ['ChronoBottom', 'UUID']),
                rqet.find_nested_tags_text(hod_node, ['ChronoTop', 'UUID']))


def equivalent_chrono_pairs(pair_a, pair_b, model = None):
    """Returns True if the two chronostratigraphic pairs are equivalent"""
    if pair_a == pair_b:
        return True
    if pair_a is None or pair_b is None:
        return False
    if pair_a == (None, None) or pair_b == (None, None):
        return False
    if model is not None:
        # todo: compare chrono info by looking up xml based on the uuids
        pass
    return False  # cautious


def create_xml_has_occurred_during(model, parent_node, hod_pair, tag = 'HasOccuredDuring'):
    """Creates XML sub-tree 'HasOccuredDuring' node"""
    if hod_pair is None:
        return
    base_chrono_uuid, top_chrono_uuid = hod_pair
    if base_chrono_uuid is None or top_chrono_uuid is None:
        return
    hod_node = rqet.SubElement(parent_node, tag)
    hod_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'TimeInterval')
    hod_node.text = rqet.null_xml_text
    chrono_base_root = model.root_for_uuid(base_chrono_uuid)
    chrono_top_root = model.root_for_uuid(top_chrono_uuid)
    model.create_ref_node('ChronoBottom', model.title_for_root(chrono_base_root), base_chrono_uuid, root = hod_node)
    model.create_ref_node('ChronoTop', model.title_for_root(chrono_top_root), top_chrono_uuid, root = hod_node)
