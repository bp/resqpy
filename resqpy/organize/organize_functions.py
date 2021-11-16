"""organize_old.py: RESQML Feature and Interpretation classes."""

version = '9th August 2021'

import resqpy.olio.xml_et as rqet


def alias_for_attribute(attribute_name):
    """Return an attribute that is a direct alias for an existing attribute."""

    def fget(self):
        return getattr(self, attribute_name)

    def fset(self, value):
        return setattr(self, attribute_name, value)

    return property(fget, fset, doc = f"Alias for {attribute_name}")


def equivalent_extra_metadata(a, b):
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
