"""Class for RESQML Horizon Interpretation organizational objects."""

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize
import resqpy.organize.genetic_boundary_feature as gbf
import resqpy.organize._utils as ou
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class HorizonInterpretation(BaseResqpy):
    """Class for RESQML Horizon Interpretation organizational objects."""

    resqml_type = 'HorizonInterpretation'
    valid_domains = ('depth', 'time', 'mixed')
    valid_sequence_stratigraphy_surfaces = ('flooding', 'ravinement', 'maximum flooding', 'transgressive')
    valid_boundary_relations = ('conformable', 'unconformable below and above', 'unconformable above',
                                'unconformable below')

    def __init__(self,
                 parent_model,
                 uuid = None,
                 title = None,
                 genetic_boundary_feature = None,
                 domain = 'depth',
                 boundary_relation_list = None,
                 sequence_stratigraphy_surface = None,
                 extra_metadata = None):
        """Initialises a horizon interpretation organisational object."""

        # note: will create a paired GeneticBoundaryFeature object when loading from xml (and possibly a Surface object)

        self.domain = domain
        self.genetic_boundary_feature = genetic_boundary_feature  # InterpretedFeature RESQML field, when not loading from xml
        self.feature_root = None if self.genetic_boundary_feature is None else self.genetic_boundary_feature.root
        if (not title) and self.genetic_boundary_feature is not None:
            title = self.genetic_boundary_feature.feature_name
        self.has_occurred_during = (None, None)
        self.boundary_relation_list = None if not boundary_relation_list else boundary_relation_list.copy()
        self.sequence_stratigraphy_surface = sequence_stratigraphy_surface
        super().__init__(model = parent_model, uuid = uuid, title = title, extra_metadata = extra_metadata)

    def _load_from_xml(self):
        root_node = self.root
        self.domain = rqet.find_tag_text(root_node, 'Domain')
        interp_feature_ref_node = rqet.find_tag(root_node, 'InterpretedFeature')
        assert interp_feature_ref_node is not None
        self.feature_root = self.model.referenced_node(interp_feature_ref_node)
        if self.feature_root is not None:
            self.genetic_boundary_feature = gbf.GeneticBoundaryFeature(self.model,
                                                                       kind = 'horizon',
                                                                       uuid = self.feature_root.attrib['uuid'],
                                                                       feature_name = self.model.title_for_root(
                                                                           self.feature_root))
        self.has_occurred_during = ou.extract_has_occurred_during(root_node)
        br_node_list = rqet.list_of_tag(root_node, 'BoundaryRelation')
        if br_node_list is not None and len(br_node_list) > 0:
            self.boundary_relation_list = []
            for br_node in br_node_list:
                self.boundary_relation_list.append(br_node.text)
        self.sequence_stratigraphy_surface = rqet.find_tag_text(root_node, 'SequenceStratigraphySurface')

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this interpretation is essentially the same as the other; otherwise False."""

        if other is None or not isinstance(other, HorizonInterpretation):
            return False
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
            return True
        if self.genetic_boundary_feature is not None:
            if not self.genetic_boundary_feature.is_equivalent(other.genetic_boundary_feature,
                                                               check_extra_metadata = check_extra_metadata):
                return False
        elif other.genetic_boundary_feature is not None:
            return False
        if self.root is not None and other.root is not None:
            if rqet.citation_title_for_node(self.root) != rqet.citation_title_for_node(other.root):
                return False
        elif self.root is not None or other.root is not None:
            return False
        if ((self.domain != other.domain) or
                not ou.equivalent_chrono_pairs(self.has_occurred_during, other.has_occurred_during) or
                self.sequence_stratigraphy_surface != other.sequence_stratigraphy_surface):
            return False
        if check_extra_metadata and not ou.equivalent_extra_metadata(self, other):
            return False
        if not self.boundary_relation_list and not other.boundary_relation_list:
            return True
        if not self.boundary_relation_list or not other.boundary_relation_list:
            return False
        return set(self.boundary_relation_list) == set(other.boundary_relation_list)

    def create_xml(self,
                   genetic_boundary_feature_root = None,
                   add_as_part = True,
                   add_relationships = True,
                   originator = None,
                   title_suffix = None,
                   reuse = True):
        """Creates a horizon interpretation organisational xml node from a horizon interpretation object."""

        # note: related genetic boundary feature node should be created first and referenced here

        if not self.title:
            self.title = self.genetic_boundary_feature.feature_name
        if title_suffix:
            self.title += ' ' + title_suffix

        if reuse and self.try_reuse():
            return self.root
        hi = super().create_xml(add_as_part = False, originator = originator)

        if self.genetic_boundary_feature is not None:
            gbf_root = self.genetic_boundary_feature.root
            if gbf_root is not None:
                if genetic_boundary_feature_root is None:
                    genetic_boundary_feature_root = gbf_root
                else:
                    assert gbf_root is genetic_boundary_feature_root, 'genetic boundary feature mismatch'

        assert self.domain in self.valid_domains, 'illegal domain value for horizon interpretation'
        dom_node = rqet.SubElement(hi, ns['resqml2'] + 'Domain')
        dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
        dom_node.text = self.domain

        ou.create_xml_has_occurred_during(self.model, hi, self.has_occurred_during)

        if self.boundary_relation_list is not None:
            for boundary_relation in self.boundary_relation_list:
                assert boundary_relation in self.valid_boundary_relations
                br_node = rqet.SubElement(hi, ns['resqml2'] + 'BoundaryRelation')
                br_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'BoundaryRelation')
                br_node.text = boundary_relation

        if self.sequence_stratigraphy_surface is not None:
            assert self.sequence_stratigraphy_surface in self.valid_sequence_stratigraphy_surfaces
            sss_node = rqet.SubElement(hi, ns['resqml2'] + 'SequenceStratigraphySurface')
            sss_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'SequenceStratigraphySurface')
            sss_node.text = self.sequence_stratigraphy_surface

        self.model.create_ref_node('InterpretedFeature',
                                   self.model.title_for_root(genetic_boundary_feature_root),
                                   genetic_boundary_feature_root.attrib['uuid'],
                                   content_type = 'obj_GeneticBoundaryFeature',
                                   root = hi)

        if add_as_part:
            self.model.add_part('obj_HorizonInterpretation', self.uuid, hi)
            if add_relationships:
                self.model.create_reciprocal_relationship(hi, 'destinationObject', genetic_boundary_feature_root,
                                                          'sourceObject')

        return hi
