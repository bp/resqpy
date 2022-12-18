"""Class for RESQML Boundary Feature Interpretation organizational objects."""

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize
import resqpy.organize.boundary_feature as obf
import resqpy.organize._utils as ou
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class BoundaryFeatureInterpretation(BaseResqpy):
    """Class for RESQML Horizon Interpretation organizational objects."""

    resqml_type = 'BoundaryFeatureInterpretation'
    valid_domains = ('depth', 'time', 'mixed')

    def __init__(self,
                 parent_model,
                 uuid = None,
                 title = None,
                 boundary_feature = None,
                 domain = 'depth',
                 extra_metadata = None):
        """Initialises a boundary feature interpretation organisational object."""

        # note: will create a paired BoundaryFeature object when loading from xml

        self.domain = domain
        self.boundary_feature = boundary_feature  # InterpretedFeature RESQML field, when not loading from xml
        self.feature_uuid = None if self.boundary_feature is None else self.boundary_feature.uuid
        if (not title) and self.boundary_feature is not None:
            title = self.boundary_feature.feature_name
        self.has_occurred_during = (None, None)
        super().__init__(model = parent_model, uuid = uuid, title = title, extra_metadata = extra_metadata)

    def _load_from_xml(self):
        root_node = self.root
        self.domain = rqet.find_tag_text(root_node, 'Domain')
        interp_feature_ref_node = rqet.find_tag(root_node, 'InterpretedFeature')
        assert interp_feature_ref_node is not None
        feature_root = self.model.referenced_node(interp_feature_ref_node)
        if feature_root is not None:
            self.feature_uuid = feature_root.attrib['uuid']
            self.boundary_feature = obf.BoundaryFeature(self.model,
                                                        uuid = self.feature_uuid,
                                                        feature_name = self.model.title_for_root(feature_root))
        self.has_occurred_during = ou.extract_has_occurred_during(root_node)

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this interpretation is essentially the same as the other; otherwise False."""

        if other is None or not isinstance(other, BoundaryFeatureInterpretation):
            return False
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
            return True
        if self.boundary_feature is not None:
            if not self.boundary_feature.is_equivalent(other.boundary_feature,
                                                       check_extra_metadata = check_extra_metadata):
                return False
        elif other.boundary_feature is not None:
            return False
        if self.title != other.title:
            return False
        if ((self.domain != other.domain) or
                not ou.equivalent_chrono_pairs(self.has_occurred_during, other.has_occurred_during)):
            return False
        if check_extra_metadata and not ou.equivalent_extra_metadata(self, other):
            return False
        return True

    def create_xml(self, add_as_part = True, add_relationships = True, originator = None, reuse = True):
        """Creates a boundary feature interpretation organisational xml tree."""

        # note: related boundary feature node should be created first and referenced here

        if not self.title:
            self.title = self.boundary_feature.feature_name

        if reuse and self.try_reuse():
            return self.root
        bfi = super().create_xml(add_as_part = False, originator = originator)

        if self.boundary_feature.root is None:
            bf_root = self.boundary_feature.create_xml(reuse = True)
            self.feature_uuid = bf_root.attrib['uuid']

        assert self.domain in self.valid_domains, 'illegal domain value for boundary feature interpretation'
        dom_node = rqet.SubElement(bfi, ns['resqml2'] + 'Domain')
        dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
        dom_node.text = self.domain

        ou.create_xml_has_occurred_during(self.model, bfi, self.has_occurred_during)

        self.model.create_ref_node('InterpretedFeature',
                                   self.model.title(uuid = self.feature_uuid),
                                   self.feature_uuid,
                                   content_type = 'obj_BoundaryFeature',
                                   root = bfi)

        if add_as_part:
            self.model.add_part('obj_BoundaryFeatureInterpretation', self.uuid, bfi)
            if add_relationships:
                self.model.create_reciprocal_relationship(bfi, 'destinationObject', self.boundary_feature.root,
                                                          'sourceObject')

        return bfi
