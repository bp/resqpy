"""Class for RESQML Earth Model Interpretation organizational objects."""

from ._utils import (equivalent_extra_metadata, extract_has_occurred_during, equivalent_chrono_pairs,
                     create_xml_has_occurred_during)

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize
import resqpy.organize.organization_feature as oof
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class EarthModelInterpretation(BaseResqpy):
    """Class for RESQML Earth Model Interpretation organizational objects."""

    # TODO: add support for StratigraphicColumn reference and other optional references

    resqml_type = 'EarthModelInterpretation'
    valid_domains = ('depth', 'time', 'mixed')

    def __init__(self,
                 parent_model,
                 uuid = None,
                 title = None,
                 organization_feature = None,
                 domain = 'depth',
                 extra_metadata = None):
        """Initialises an earth model interpretation organisational object."""
        self.domain = domain
        self.organization_feature = organization_feature  # InterpretedFeature RESQML field
        self.feature_root = None if self.organization_feature is None else self.organization_feature.root
        self.has_occurred_during = (None, None)
        if (not title) and organization_feature is not None:
            title = organization_feature.feature_name
        super().__init__(model = parent_model, uuid = uuid, title = title, extra_metadata = extra_metadata)

    def _load_from_xml(self):
        root = self.root
        self.domain = rqet.find_tag_text(root, 'Domain')
        interp_feature_ref_node = rqet.find_tag(root, 'InterpretedFeature')
        assert interp_feature_ref_node is not None
        self.feature_root = self.model.referenced_node(interp_feature_ref_node)
        if self.feature_root is not None:
            self.organization_feature = oof.OrganizationFeature(self.model,
                                                                uuid = self.feature_root.attrib['uuid'],
                                                                feature_name = self.model.title_for_root(
                                                                    self.feature_root))
        self.has_occurred_during = extract_has_occurred_during(root)

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this interpretation is essentially the same as the other; otherwise False."""
        if other is None or not isinstance(other, EarthModelInterpretation):
            return False
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
            return True
        if self.organization_feature is not None:
            if not self.organization_feature.is_equivalent(other.organization_feature,
                                                           check_extra_metadata = check_extra_metadata):
                return False
        elif other.organization_feature is not None:
            return False
        if self.root is not None and other.root is not None:
            if rqet.citation_title_for_node(self.root) != rqet.citation_title_for_node(other.root):
                return False
        elif self.root is not None or other.root is not None:
            return False
        if check_extra_metadata and not equivalent_extra_metadata(self, other):
            return False
        return self.domain == other.domain and equivalent_chrono_pairs(self.has_occurred_during,
                                                                       other.has_occurred_during)

    def create_xml(self,
                   organization_feature_root = None,
                   add_as_part = True,
                   add_relationships = True,
                   originator = None,
                   title_suffix = None,
                   reuse = True):
        """Creates an earth model interpretation organisational xml node from an earth model interpretation object."""

        # note: related organization feature node should be created first and referenced here

        if not self.title:
            self.title = self.organization_feature.feature_name
        if title_suffix:
            self.title += ' ' + title_suffix

        if reuse and self.try_reuse():
            return self.root
        emi = super().create_xml(add_as_part = False, originator = originator)

        if self.organization_feature is not None:
            of_root = self.organization_feature.root
            if of_root is not None:
                if organization_feature_root is None:
                    organization_feature_root = of_root
                else:
                    assert of_root is organization_feature_root, 'organization feature mismatch'

        assert organization_feature_root is not None, 'interpreted feature not established for model interpretation'

        assert self.domain in self.valid_domains, 'illegal domain value for earth model interpretation'
        dom_node = rqet.SubElement(emi, ns['resqml2'] + 'Domain')
        dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
        dom_node.text = self.domain

        self.model.create_ref_node('InterpretedFeature',
                                   self.model.title_for_root(organization_feature_root),
                                   organization_feature_root.attrib['uuid'],
                                   content_type = 'obj_OrganizationFeature',
                                   root = emi)

        create_xml_has_occurred_during(self.model, emi, self.has_occurred_during)

        if add_as_part:
            self.model.add_part('obj_EarthModelInterpretation', self.uuid, emi)
            if add_relationships:
                self.model.create_reciprocal_relationship(emi, 'destinationObject', organization_feature_root,
                                                          'sourceObject')

        return emi
