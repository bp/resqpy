"""Containing resqml propertykind class"""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class PropertyKind(BaseResqpy):
    """Class catering for RESQML bespoke PropertyKind objects."""

    resqml_type = "PropertyKind"

    def __init__(self,
                 parent_model,
                 uuid = None,
                 title = None,
                 is_abstract = False,
                 example_uom = None,
                 naming_system = 'urn:resqml:bp.com:resqpy',
                 parent_property_kind = 'continuous',
                 extra_metadata = None,
                 originator = None):
        """Initialise a new bespoke property kind."""

        self.is_abstract = is_abstract
        self.naming_system = naming_system
        self.example_uom = example_uom
        self.parent_kind = parent_property_kind
        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)

    def _load_from_xml(self):
        root_node = self.root
        self.is_abstract = rqet.find_tag_bool(root_node, 'IsAbstract')
        self.naming_system = rqet.find_tag_text(root_node, 'NamingSystem')
        self.example_uom = rqet.find_tag_text(root_node, 'RepresentativeUom')
        ppk_node = rqet.find_tag(root_node, 'ParentPropertyKind')
        assert ppk_node is not None
        ppk_kind_node = rqet.find_tag(ppk_node, 'Kind')
        assert ppk_kind_node is not None, 'only standard property kinds supported as parent kind'
        self.parent_kind = ppk_kind_node.text

    def is_equivalent(self, other_pk, check_extra_metadata = True):
        """Returns True if this property kind is essentially the same as the other; False otherwise."""

        if other_pk is None:
            return False
        if self is other_pk:
            return True
        if bu.matching_uuids(self.uuid, other_pk.uuid):
            return True
        if (self.parent_kind != other_pk.parent_kind or self.title != other_pk.title or
                self.is_abstract != other_pk.is_abstract or self.naming_system != other_pk.naming_system):
            return False
        if (self.example_uom and other_pk.example_uom) and self.example_uom != other_pk.example_uom:
            return False
        if check_extra_metadata:
            if (self.extra_metadata or other_pk.extra_metadata) and self.extra_metadata != other_pk.extra_metadata:
                return False
        return True

    def create_xml(self, add_as_part = True, originator = None, reuse = True):
        """Create xml for this bespoke property kind."""

        if reuse and self.try_reuse():
            return self.root  # check for reusable (equivalent) object

        pk = super().create_xml(add_as_part = False, originator = originator)

        ns_node = rqet.SubElement(pk, ns['resqml2'] + 'NamingSystem')
        ns_node.set(ns['xsi'] + 'type', ns['xsd'] + 'anyURI')
        ns_node.text = str(self.naming_system)

        ia_node = rqet.SubElement(pk, ns['resqml2'] + 'IsAbstract')
        ia_node.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
        ia_node.text = str(self.is_abstract).lower()

        # note: schema definition requires this field, even for discrete property kinds
        uom = self.example_uom
        if uom is None:
            uom = 'Euc'
        ru_node = rqet.SubElement(pk, ns['resqml2'] + 'RepresentativeUom')
        ru_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ResqmlUom')
        ru_node.text = str(uom)

        ppk_node = rqet.SubElement(pk, ns['resqml2'] + 'ParentPropertyKind')
        ppk_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'StandardPropertyKind')
        ppk_node.text = rqet.null_xml_text

        ppk_kind_node = rqet.SubElement(ppk_node, ns['resqml2'] + 'Kind')
        ppk_kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ResqmlPropertyKind')
        ppk_kind_node.text = str(self.parent_kind)

        if add_as_part:
            self.model.add_part('obj_PropertyKind', self.uuid, pk)
            # no relationships at present, if local parent property kinds were to be supported then a rel. is needed there

        return pk


def create_transmisibility_multiplier_property_kind(model):
    """Create a local property kind 'transmisibility multiplier' for a given model.

    argument:
       model: resqml model object

    returns:
       property kind uuid
    """
    log.debug("Making a new property kind 'Transmissibility multiplier'")
    tmult_kind = PropertyKind(parent_model = model,
                              title = 'transmissibility multiplier',
                              parent_property_kind = 'continuous')
    tmult_kind.create_xml()
    tmult_kind_uuid = tmult_kind.uuid
    model.store_epc()
    return tmult_kind_uuid


def establish_zone_property_kind(model):
    """Returns zone local property kind object, creating the xml and adding as part if not found in model."""

    zone_pk_uuid = model.uuid(obj_type = 'LocalPropertyKind', title = 'zone')
    if zone_pk_uuid is None:
        zone_pk = PropertyKind(model, title = 'zone', parent_property_kind = 'discrete')
        zone_pk.create_xml()
    else:
        zone_pk = PropertyKind(model, uuid = zone_pk_uuid)
    return zone_pk
