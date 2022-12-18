"""Class for RESQML Fluid Boundary Feature (contact) organizational objects."""

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize
import resqpy.organize._utils as ou
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class FluidBoundaryFeature(BaseResqpy):
    """Class for RESQML Fluid Boundary Feature (contact) organizational objects."""

    resqml_type = "FluidBoundaryFeature"
    feature_name = ou.alias_for_attribute("title")
    valid_kinds = ('free water contact', 'gas oil contact', 'gas water contact', 'seal', 'water oil contact')

    def __init__(self, parent_model, uuid = None, kind = None, feature_name = None, extra_metadata = None):
        """Initialises a fluid boundary feature (contact) organisational object."""

        self.kind = kind
        super().__init__(model = parent_model, uuid = uuid, title = feature_name, extra_metadata = extra_metadata)

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this feature is essentially the same as the other; otherwise False."""

        if other is None or not isinstance(other, FluidBoundaryFeature):
            return False
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
            return True
        if check_extra_metadata and not ou.equivalent_extra_metadata(self, other):
            return False
        return self.feature_name == other.feature_name and self.kind == other.kind

    def _load_from_xml(self):
        self.kind = rqet.find_tag_text(self.root, 'FluidContact')

    def create_xml(self, add_as_part = True, originator = None, reuse = True):
        """Creates a fluid boundary feature organisational xml node from this fluid boundary feature object."""

        if reuse and self.try_reuse():
            return self.root  # check for reusable (equivalent) object
        # create node with citation block
        fbf = super().create_xml(add_as_part = False, originator = originator)

        # Extra element for kind
        if self.kind not in self.valid_kinds:
            raise ValueError(f"fluid boundary feature kind '{self.kind}' not recognized")

        kind_node = rqet.SubElement(fbf, ns['resqml2'] + 'FluidContact')
        kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'FluidContact')
        kind_node.text = self.kind

        if add_as_part:
            self.model.add_part('obj_FluidBoundaryFeature', self.uuid, fbf)

        return fbf
