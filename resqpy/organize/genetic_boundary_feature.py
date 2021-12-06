"""Class for RESQML Genetic Boundary Feature (horizon) organizational objects."""

from ._utils import (equivalent_extra_metadata, alias_for_attribute, extract_has_occurred_during,
                     equivalent_chrono_pairs, create_xml_has_occurred_during)

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class GeneticBoundaryFeature(BaseResqpy):
    """Class for RESQML Genetic Boundary Feature (horizon) organizational objects."""

    resqml_type = "GeneticBoundaryFeature"
    feature_name = alias_for_attribute("title")
    valid_kinds = ('horizon', 'geobody boundary')

    def __init__(self,
                 parent_model,
                 root_node = None,
                 uuid = None,
                 kind = None,
                 feature_name = None,
                 extra_metadata = None):
        """Initialises a genetic boundary feature (horizon or geobody boundary) organisational object."""
        self.kind = kind
        self.absolute_age = None  # (timestamp, year offset) pair, or None; todo: support setting from args
        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = feature_name,
                         extra_metadata = extra_metadata,
                         root_node = root_node)

    def _load_from_xml(self):
        self.kind = rqet.find_tag_text(self.root, 'GeneticBoundaryKind')
        age_node = rqet.find_tag(self.root, 'AbsoluteAge')
        if age_node:
            self.absolute_age = (rqet.find_tag_text(age_node, 'DateTime'), rqet.find_tag_int(age_node, 'YearOffset')
                                )  # year offset may be None

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this feature is essentially the same as the other; otherwise False."""
        if other is None or not isinstance(other, GeneticBoundaryFeature):
            return False
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
            return True
        if check_extra_metadata and not equivalent_extra_metadata(self, other):
            return False
        return self.feature_name == other.feature_name and self.kind == other.kind and self.absolute_age == other.absolute_age

    def create_xml(self, add_as_part = True, originator = None, reuse = True):
        """Creates a genetic boundary feature organisational xml node from this genetic boundary feature object."""

        if reuse and self.try_reuse():
            return self.root  # check for reusable (equivalent) object
        # create node with citation block
        gbf = super().create_xml(add_as_part = False, originator = originator)

        assert self.kind in self.valid_kinds
        kind_node = rqet.SubElement(gbf, ns['resqml2'] + 'GeneticBoundaryKind')
        kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'GeneticBoundaryKind')
        kind_node.text = self.kind

        if self.absolute_age is not None:
            date_time, year_offset = self.absolute_age
            age_node = rqet.SubElement(gbf, ns['resqml2'] + 'AbsoluteAge')
            age_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Timestamp')
            age_node.text = rqet.null_xml_text
            dt_node = rqet.SubElement(age_node, ns['resqml2'] + 'DateTime')
            dt_node.set(ns['xsi'] + 'type', ns['xsd'] + 'dateTime')
            dt_node.text = str(date_time)
            if year_offset is not None:
                yo_node = rqet.SubElement(age_node, ns['resqml2'] + 'YearOffset')
                yo_node.set(ns['xsi'] + 'type', ns['xsd'] + 'long')
                yo_node.text = str(year_offset)

        if add_as_part:
            self.model.add_part('obj_GeneticBoundaryFeature', self.uuid, gbf)

        return gbf
