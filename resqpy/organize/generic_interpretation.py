"""Class for RESQML Generic Feature Interpretation objects."""

import resqpy.organize._utils as ou
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class GenericInterpretation(BaseResqpy):
    """Class for RESQML Generic Feature Interpretation objects."""

    resqml_type = 'GenericFeatureInterpretation'
    valid_domains = ('depth', 'time', 'mixed')

    def __init__(self,
                 parent_model,
                 uuid = None,
                 title = None,
                 feature_uuid = None,
                 domain = 'depth',
                 composition = None,
                 material_implacement = None,
                 geobody_shape = None,
                 extra_metadata = None):
        """Initialise a new geobody interpretation object, either from xml or explicitly."""

        assert domain in self.valid_domains

        self.uuid = None
        self.title = None
        self.feature_uuid = feature_uuid
        self.domain = domain
        self.has_occurred_during = (None, None)

        super().__init__(model = parent_model, uuid = uuid, title = title, extra_metadata = extra_metadata)

        if not self.title and feature_uuid is not None:
            title = parent_model.title_for_root(parent_model.root_for_uuid(feature_uuid))

    def _load_from_xml(self):
        root_node = self.root
        self.domain = rqet.find_tag_text(root_node, 'Domain')
        interp_feature_uuid = rqet.find_nested_tags_text(root_node, ['InterpretedFeature', 'UUID'])
        assert interp_feature_uuid is not None
        self.feature_uuid = bu.uuid_from_string(interp_feature_uuid)
        self.has_occurred_during = ou.extract_has_occurred_during(root_node)

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this interpretation is essentially the same as the other; otherwise False."""

        if other is None or not isinstance(other, GenericInterpretation):
            return False
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
            return True
        if self.title != other.title:
            return False
        if not bu.matching_uuids(self.feature_uuid, other.feature_uuid):
            return False
        if self.domain != other.domain:
            return False
        if not ou.equivalent_chrono_pairs(self.main_has_occurred_during, other.main_has_occurred_during):
            return False
        if check_extra_metadata and not ou.equivalent_extra_metadata(self, other):
            return False
        return True

    def create_xml(self, add_as_part = True, add_relationships = True, originator = None, reuse = True):
        """Creates a generic feature interpretation organisational xml node from a generic interpretation object."""

        # note: related feature node should be created first and referenced here
        assert self.feature_uuid is not None
        feature_root = self.model.root_for_uuid(self.feature_uuid)
        assert feature_root is not None

        if not self.title:
            self.title = rqet.find_nested_tags_text(feature_root, ['Citation', 'Title'])

        if reuse and self.try_reuse():
            return self.root

        gfi = super().create_xml(add_as_part = False, originator = originator)

        assert self.domain in self.valid_domains, 'illegal domain value for fault interpretation'
        dom_node = rqet.SubElement(gfi, ns['resqml2'] + 'Domain')
        dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
        dom_node.text = self.domain

        self.model.create_ref_node('InterpretedFeature',
                                   self.model.title_for_root(feature_root),
                                   self.feature_uuid,
                                   content_type = self.model.type_of_uuid(self.feature_uuid),
                                   root = gfi)

        ou.create_xml_has_occurred_during(self.model, gfi, self.has_occurred_during)

        if add_as_part:
            self.model.add_part('obj_GenericFeatureInterpretation', self.uuid, gfi)
            if add_relationships:
                self.model.create_reciprocal_relationship(gfi, 'destinationObject', feature_root, 'sourceObject')

        return gfi
