"""Class for RESQML Wellbore Interpretation organizational objects."""

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize
import resqpy.organize.wellbore_feature as wbf
import resqpy.organize._utils as ou
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class WellboreInterpretation(BaseResqpy):
    """Class for RESQML Wellbore Interpretation organizational objects.

    RESQML documentation:

       May refer to one of these:

       * **Wellbore**. A unique, oriented path from the bottom of a drilled borehole to the surface of the earth.
         The path must not overlap or cross itself.
       * **Borehole**. A hole excavated in the earth as a result of drilling or boring operations. The borehole
         may represent the hole of an entire wellbore (when no sidetracks are present), or a sidetrack extension.
         A borehole extends from an originating point (the surface location for the initial borehole or kickoff point
         for sidetracks) to a terminating (bottomhole) point.
       * **Sidetrack**. A borehole that originates in another borehole as opposed to originating at the surface.
    """

    resqml_type = 'WellboreInterpretation'
    valid_domains = ('depth', 'time', 'mixed')

    def __init__(self,
                 parent_model,
                 uuid = None,
                 title = None,
                 is_drilled = None,
                 wellbore_feature = None,
                 domain = 'depth',
                 extra_metadata = None):
        """Initialises a wellbore interpretation organisational object."""

        # note: will create a paired WellboreFeature object when loading from xml

        self.is_drilled = is_drilled
        self.wellbore_feature = wellbore_feature
        self.feature_root = None if self.wellbore_feature is None else self.wellbore_feature.root
        if (not title) and self.wellbore_feature is not None:
            title = self.wellbore_feature.feature_name
        self.domain = domain
        super().__init__(model = parent_model, uuid = uuid, title = title, extra_metadata = extra_metadata)

    def _load_from_xml(self):
        root_node = self.root
        self.is_drilled = rqet.find_tag_bool(root_node, 'IsDrilled')
        self.domain = rqet.find_tag_text(root_node, 'Domain')
        interp_feature_ref_node = rqet.find_tag(root_node, 'InterpretedFeature')
        if interp_feature_ref_node is not None:
            self.feature_root = self.model.referenced_node(interp_feature_ref_node)
            if self.feature_root is not None:
                self.wellbore_feature = wbf.WellboreFeature(self.model,
                                                            uuid = self.feature_root.attrib['uuid'],
                                                            feature_name = self.model.title_for_root(self.feature_root))

    def iter_trajectories(self):
        """Iterable of associated trajectories."""

        import resqpy.well

        uuids = self.model.uuids(obj_type = "WellboreTrajectoryRepresentation", related_uuid = self.uuid)
        for uuid in uuids:
            yield resqpy.well.Trajectory(self.model, uuid = uuid)

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this interpretation is essentially the same as the other; otherwise False."""

        if other is None or not isinstance(other, WellboreInterpretation):
            return False
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
            return True
        if self.wellbore_feature is not None:
            if not self.wellbore_feature.is_equivalent(other.wellbore_feature,
                                                       check_extra_metadata = check_extra_metadata):
                return False
        elif other.wellbore_feature is not None:
            return False
        if self.root is not None and other.root is not None:
            if rqet.citation_title_for_node(self.root) != rqet.citation_title_for_node(other.root):
                return False
            if self.domain != other.domain:
                return False
        elif self.root is not None or other.root is not None:
            return False
        if check_extra_metadata and not ou.equivalent_extra_metadata(self, other):
            return False
        return (self.title == other.title and self.is_drilled == other.is_drilled)

    def create_xml(self,
                   wellbore_feature_root = None,
                   add_as_part = True,
                   add_relationships = True,
                   originator = None,
                   title_suffix = None,
                   reuse = True):
        """Creates a wellbore interpretation organisational xml node from a wellbore interpretation object."""

        # note: related wellbore feature node should be created first and referenced here

        if not self.title:
            self.title = self.wellbore_feature.feature_name
        if title_suffix:
            self.title += ' ' + title_suffix

        if reuse and self.try_reuse():
            return self.root
        wi = super().create_xml(add_as_part = False, originator = originator)

        if self.wellbore_feature is not None:
            wbf_root = self.wellbore_feature.root
            if wbf_root is not None:
                if wellbore_feature_root is None:
                    wellbore_feature_root = wbf_root
                else:
                    assert wbf_root is wellbore_feature_root, 'wellbore feature mismatch'

        if self.is_drilled is None:
            self.is_drilled = False

        id_node = rqet.SubElement(wi, ns['resqml2'] + 'IsDrilled')
        id_node.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
        id_node.text = str(self.is_drilled).lower()

        assert self.domain in self.valid_domains, 'illegal domain value for wellbore interpretation'
        domain_node = rqet.SubElement(wi, ns['resqml2'] + 'Domain')
        domain_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
        domain_node.text = str(self.domain).lower()

        self.model.create_ref_node('InterpretedFeature',
                                   self.model.title_for_root(wellbore_feature_root),
                                   wellbore_feature_root.attrib['uuid'],
                                   content_type = 'obj_WellboreFeature',
                                   root = wi)

        if add_as_part:
            self.model.add_part('obj_WellboreInterpretation', self.uuid, wi)
            if add_relationships:
                self.model.create_reciprocal_relationship(wi, 'destinationObject', wellbore_feature_root,
                                                          'sourceObject')

        return wi
