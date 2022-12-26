"""Class for generic RESQML Organization Feature objects."""

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize
import resqpy.organize._utils as ou
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class OrganizationFeature(BaseResqpy):
    """Class for generic RESQML Organization Feature objects."""

    resqml_type = "OrganizationFeature"
    valid_kinds = ['earth model', 'fluid', 'stratigraphic', 'structural']
    feature_name = ou.alias_for_attribute("title")

    def __init__(self,
                 parent_model,
                 uuid = None,
                 feature_name = None,
                 organization_kind = None,
                 originator = None,
                 extra_metadata = None):
        """Initialises an organization feature object."""

        self.organization_kind = organization_kind
        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = feature_name,
                         originator = originator,
                         extra_metadata = extra_metadata)

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this feature is essentially the same as the other; otherwise False."""

        if not isinstance(other, OrganizationFeature):
            return False
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
            return True
        return (self.feature_name == other.feature_name and self.organization_kind == other.organization_kind and
                ((not check_extra_metadata) or ou.equivalent_extra_metadata(self, other)))

    def _load_from_xml(self):
        self.organization_kind = rqet.find_tag_text(self.root, 'OrganizationKind')

    def create_xml(self, add_as_part = True, originator = None, reuse = True):
        """Creates an organization feature xml node from this organization feature object."""

        if reuse and self.try_reuse():
            return self.root  # check for reusable (equivalent) object
        # create node with citation block
        ofn = super().create_xml(add_as_part = False, originator = originator)

        # Extra element for organization_kind
        if self.organization_kind not in self.valid_kinds:
            raise ValueError(self.organization_kind)
        kind_node = rqet.SubElement(ofn, ns['resqml2'] + 'OrganizationKind')
        kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'OrganizationKind')
        kind_node.text = self.organization_kind

        if add_as_part:
            self.model.add_part('obj_OrganizationFeature', self.uuid, ofn)

        return ofn
