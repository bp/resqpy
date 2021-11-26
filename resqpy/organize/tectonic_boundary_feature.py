"""Class for RESQML Tectonic Boundary Feature (fault) organizational objects."""

from ._utils import (equivalent_extra_metadata, alias_for_attribute, extract_has_occurred_during,
                     equivalent_chrono_pairs, create_xml_has_occurred_during)

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class TectonicBoundaryFeature(BaseResqpy):
    """Class for RESQML Tectonic Boundary Feature (fault) organizational objects."""

    resqml_type = "TectonicBoundaryFeature"
    feature_name = alias_for_attribute("title")
    valid_kinds = ('fault', 'fracture')

    def __init__(self,
                 parent_model,
                 root_node = None,
                 uuid = None,
                 kind = None,
                 feature_name = None,
                 extra_metadata = None):
        """Initialises a tectonic boundary feature (fault or fracture) organisational object."""
        self.kind = kind
        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = feature_name,
                         extra_metadata = extra_metadata,
                         root_node = root_node)

    def _load_from_xml(self):
        self.kind = rqet.find_tag_text(self.root, 'TectonicBoundaryKind')
        if not self.kind:
            self.kind = 'fault'

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this feature is essentially the same as the other; otherwise False."""
        if other is None or not isinstance(other, TectonicBoundaryFeature):
            return False
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
            return True
        if check_extra_metadata and not equivalent_extra_metadata(self, other):
            return False
        return self.feature_name == other.feature_name and self.kind == other.kind

    def create_xml(self, add_as_part = True, originator = None, reuse = True):
        """Creates a tectonic boundary feature organisational xml node from this tectonic boundary feature object."""

        if reuse and self.try_reuse():
            return self.root  # check for reusable (equivalent) object
        # create node with citation block
        tbf = super().create_xml(add_as_part = False, originator = originator)

        assert self.kind in self.valid_kinds
        kind_node = rqet.SubElement(tbf, ns['resqml2'] + 'TectonicBoundaryKind')
        kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'TectonicBoundaryKind')
        kind_node.text = self.kind

        if add_as_part:
            self.model.add_part('obj_TectonicBoundaryFeature', self.uuid, tbf)

        return tbf
