"""Class for RESQML Geobody Boudary Interpretation organizational objects."""

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize
import resqpy.organize.genetic_boundary_feature as gbf
import resqpy.organize._utils as ou
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class GeobodyBoundaryInterpretation(BaseResqpy):
    """Class for RESQML Geobody Boundary Interpretation organizational objects."""

    resqml_type = 'GeobodyBoundaryInterpretation'
    valid_domains = ('depth', 'time', 'mixed')
    valid_boundary_relations = ('conformable', 'unconformable below and above', 'unconformable above',
                                'unconformable below')

    def __init__(self,
                 parent_model,
                 uuid = None,
                 title = None,
                 genetic_boundary_feature = None,
                 domain = 'depth',
                 boundary_relation_list = None,
                 extra_metadata = None):
        """ Instantiate a GeobodyBoundaryInterpretation object.

        arguments:
            parent_model(model.Model): Model to which the feature belongs
            uuid(UUID, Optional): The UUID of an existing Geobody Boundary Interpretation object; if present,
                all the other optional arguments are ignored
            title(str, Optional): Citation title when creating a new object
            genetic_boundary_feature(GeneticBoundaryFeature,Optional): Interpreted feature when creating a new object
            domain(str,Optional): One of ('depth', 'time', 'mixed') when creating a new object
            boundary_relation_list(list,Optional): Set of ('conformable', 'unconformable below and above', 'unconformable above',
                'unconformable below') when creating a new object
            extra_metadata(dict,Optional): Extra metadata items added when creating a new object
        """

        self.domain = domain
        self.boundary_relation_list = None if not boundary_relation_list else boundary_relation_list.copy()
        self.genetic_boundary_feature = genetic_boundary_feature  # InterpretedFeature RESQML field, when not loading from xml
        self.feature_root = None if self.genetic_boundary_feature is None else self.genetic_boundary_feature.root
        if (not title) and self.genetic_boundary_feature is not None:
            title = self.genetic_boundary_feature.feature_name
        self.has_occurred_during = (None, None)
        super().__init__(model = parent_model, uuid = uuid, title = title, extra_metadata = extra_metadata)

    def _load_from_xml(self):
        root_node = self.root
        self.domain = rqet.find_tag_text(root_node, 'Domain')
        interp_feature_ref_node = rqet.find_tag(root_node, 'InterpretedFeature')
        assert interp_feature_ref_node is not None
        self.feature_root = self.model.referenced_node(interp_feature_ref_node)
        if self.feature_root is not None:
            self.genetic_boundary_feature = gbf.GeneticBoundaryFeature(self.model,
                                                                       kind = 'geobody boundary',
                                                                       uuid = self.feature_root.attrib['uuid'],
                                                                       feature_name = self.model.title_for_root(
                                                                           self.feature_root))
        self.has_occurred_during = ou.extract_has_occurred_during(root_node)
        br_node_list = rqet.list_of_tag(root_node, 'BoundaryRelation')
        if br_node_list is not None and len(br_node_list) > 0:
            self.boundary_relation_list = []
            for br_node in br_node_list:
                self.boundary_relation_list.append(br_node.text)

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this interpretation is essentially the same as the other; otherwise False."""

        if other is None or not isinstance(other, GeobodyBoundaryInterpretation):
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
                not ou.equivalent_chrono_pairs(self.has_occurred_during, other.has_occurred_during)):
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
        """Create a organisational xml node from a geobody boundary interpretation object."""

        if not self.title:
            self.title = self.genetic_boundary_feature.feature_name
        if title_suffix:
            self.title += ' ' + title_suffix

        if reuse and self.try_reuse():
            return self.root
        gbi = super().create_xml(add_as_part = False, originator = originator)

        if self.genetic_boundary_feature is not None:
            gbf_root = self.genetic_boundary_feature.root
            if gbf_root is not None:
                if genetic_boundary_feature_root is None:
                    genetic_boundary_feature_root = gbf_root
                else:
                    assert gbf_root is genetic_boundary_feature_root, 'genetic boundary feature mismatch'
        else:
            if genetic_boundary_feature_root is None:
                genetic_boundary_feature_root = self.feature_root
            assert genetic_boundary_feature_root is not None
            self.genetic_boundary_feature = gbf.GeneticBoundaryFeature(
                self.model, uuid = genetic_boundary_feature_root.attrib['uuid'])
        self.feature_root = genetic_boundary_feature_root

        assert self.domain in self.valid_domains, 'illegal domain value for geobody boundary interpretation'
        dom_node = rqet.SubElement(gbi, ns['resqml2'] + 'Domain')
        dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
        dom_node.text = self.domain

        ou.create_xml_has_occurred_during(self.model, gbi, self.has_occurred_during)

        if self.boundary_relation_list is not None:
            for boundary_relation in self.boundary_relation_list:
                assert boundary_relation in self.valid_boundary_relations
                br_node = rqet.SubElement(gbi, ns['resqml2'] + 'BoundaryRelation')
                br_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'BoundaryRelation')
                br_node.text = str(boundary_relation)

        self.model.create_ref_node('InterpretedFeature',
                                   self.model.title_for_root(genetic_boundary_feature_root),
                                   genetic_boundary_feature_root.attrib['uuid'],
                                   content_type = 'obj_GeneticBoundaryFeature',
                                   root = gbi)

        if add_as_part:
            self.model.add_part('obj_GeobodyBoundaryInterpretation', self.uuid, gbi)
            if add_relationships:
                self.model.create_reciprocal_relationship(gbi, 'destinationObject', genetic_boundary_feature_root,
                                                          'sourceObject')

        return gbi
