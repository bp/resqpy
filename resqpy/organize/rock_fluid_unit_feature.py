"""Class for RESQML Rock Fluid Unit Feature organizational objects."""

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize
import resqpy.organize.boundary_feature as obf
import resqpy.organize._utils as ou
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class RockFluidUnitFeature(BaseResqpy):
    """Class for RESQML Rock Fluid Unit Feature organizational objects."""

    resqml_type = "RockFluidUnitFeature"
    feature_name = ou.alias_for_attribute("title")
    valid_phases = ('aquifer', 'gas cap', 'oil column', 'seal')

    def __init__(self,
                 parent_model,
                 uuid = None,
                 phase = None,
                 feature_name = None,
                 top_boundary_feature = None,
                 base_boundary_feature = None,
                 extra_metadata = None):
        """Initialises a rock fluid unit feature organisational object."""

        self.phase = phase
        self.top_boundary_feature = top_boundary_feature
        self.base_boundary_feature = base_boundary_feature

        super().__init__(model = parent_model, uuid = uuid, title = feature_name, extra_metadata = extra_metadata)

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this feature is essentially the same as the other; otherwise False."""

        if other is None or not isinstance(other, RockFluidUnitFeature):
            return False
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
            return True
        if self.feature_name != other.feature_name or self.phase != other.phase:
            return False
        if self.top_boundary_feature is not None:
            if not self.top_boundary_feature.is_equivalent(other.top_boundary_feature):
                return False
        elif other.top_boundary_feature is not None:
            return False
        if self.base_boundary_feature is not None:
            if not self.base_boundary_feature.is_equivalent(other.base_boundary_feature):
                return False
        elif other.base_boundary_feature is not None:
            return False
        if check_extra_metadata and not ou.equivalent_extra_metadata(self, other):
            return False
        return True

    def _load_from_xml(self):

        self.phase = rqet.find_tag_text(self.root, 'Phase')

        feature_ref_node = rqet.find_tag(self.root, 'FluidBoundaryTop')
        assert feature_ref_node is not None
        feature_root = self.model.referenced_node(feature_ref_node)
        feature_uuid = rqet.uuid_for_part_root(feature_root)
        assert feature_uuid is not None, 'rock fluid top boundary feature missing from model'
        self.top_boundary_feature = obf.BoundaryFeature(self.model, uuid = feature_uuid)

        feature_ref_node = rqet.find_tag(self.root, 'FluidBoundaryBottom')
        assert feature_ref_node is not None
        feature_root = self.model.referenced_node(feature_ref_node)
        feature_uuid = rqet.uuid_for_part_root(feature_root)
        assert feature_uuid is not None, 'rock fluid bottom boundary feature missing from model'
        self.base_boundary_feature = obf.BoundaryFeature(self.model, uuid = feature_uuid)

    def create_xml(self, add_as_part = True, add_relationships = True, originator = None, reuse = True):
        """Creates a rock fluid unit feature organisational xml node from this rock fluid unit feature object."""

        assert self.feature_name and self.phase and self.top_boundary_feature and self.base_boundary_feature
        if self.phase not in self.valid_phases:
            raise ValueError(f"Phase '{self.phase}' not recognized")

        if reuse and self.try_reuse():
            return self.root  # check for reusable (equivalent) object
        # create node with citation block
        rfuf = super().create_xml(add_as_part = False, originator = originator)

        phase_node = rqet.SubElement(rfuf, ns['resqml2'] + 'Phase')
        phase_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Phase')
        phase_node.text = self.phase

        top_boundary_root = self.top_boundary_feature.root
        assert top_boundary_root is not None
        self.model.create_ref_node('FluidBoundaryTop',
                                   self.model.title_for_root(top_boundary_root),
                                   top_boundary_root.attrib['uuid'],
                                   content_type = 'obj_BoundaryFeature',
                                   root = rfuf)

        base_boundary_root = self.base_boundary_feature.root
        assert base_boundary_root is not None
        self.model.create_ref_node('FluidBoundaryBottom',
                                   self.model.title_for_root(base_boundary_root),
                                   base_boundary_root.attrib['uuid'],
                                   content_type = 'obj_BoundaryFeature',
                                   root = rfuf)

        if add_as_part:
            self.model.add_part('obj_RockFluidUnitFeature', self.uuid, rfuf)
            if add_relationships:
                self.model.create_reciprocal_relationship(rfuf, 'destinationObject', top_boundary_root, 'sourceObject')
                self.model.create_reciprocal_relationship(rfuf, 'destinationObject', base_boundary_root, 'sourceObject')

        return rfuf
