"""Class for RESQML Geobody Interpretation objects."""

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize
import resqpy.organize.geobody_feature as ogf
import resqpy.organize._utils as ou
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class GeobodyInterpretation(BaseResqpy):
    """Class for RESQML Geobody Interpretation objects."""

    resqml_type = 'GeobodyInterpretation'
    valid_domains = ('depth', 'time', 'mixed')
    valid_compositions = (
        'intrusive clay',
        'organic',
        'intrusive mud',
        'evaporite salt',
        'evaporite non salt',
        'sedimentary siliclastic',
        'carbonate',
        'magmatic intrusive granitoid',
        'magmatic intrusive pyroclastic',
        'magmatic extrusive lava flow',
        'other chemichal rock',  # chemichal (stet: from xsd)
        'other chemical rock',
        'sedimentary turbidite')
    valid_implacements = ('autochtonous', 'allochtonous')
    valid_geobody_shapes = ('dyke', 'silt', 'sill', 'dome', 'sheeth', 'sheet', 'diapir', 'batholith', 'channel',
                            'delta', 'dune', 'fan', 'reef', 'wedge')

    def __init__(self,
                 parent_model,
                 uuid = None,
                 title = None,
                 geobody_feature = None,
                 domain = 'depth',
                 composition = None,
                 material_implacement = None,
                 geobody_shape = None,
                 extra_metadata = None):
        """Initialise a new geobody interpretation object, either from xml or explicitly."""

        self.domain = domain
        self.geobody_feature = geobody_feature  # InterpretedFeature RESQML field, when not loading from xml
        self.feature_root = None if self.geobody_feature is None else self.geobody_feature.root
        self.has_occurred_during = (None, None)
        self.composition = composition
        self.implacement = material_implacement
        self.geobody_shape = geobody_shape
        super().__init__(model = parent_model, uuid = uuid, title = title, extra_metadata = extra_metadata)

    def _load_from_xml(self):
        root_node = self.root
        interp_feature_ref_node = rqet.find_tag(root_node, 'InterpretedFeature')
        assert interp_feature_ref_node is not None
        self.feature_root = self.model.referenced_node(interp_feature_ref_node)
        if self.feature_root is not None:
            self.geobody_feature = ogf.GeobodyFeature(self.model,
                                                      uuid = self.feature_root.attrib['uuid'],
                                                      feature_name = self.model.title_for_root(self.feature_root))
        self.has_occurred_during = ou.extract_has_occurred_during(root_node)
        self.composition = rqet.find_tag_text(root_node, 'GeologicUnitComposition')
        self.implacement = rqet.find_tag_text(root_node, 'GeologicUnitMaterialImplacement')
        self.geobody_shape = rqet.find_tag_text(root_node, 'Geobody3dShape')

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this interpretation is essentially the same as the other; otherwise False."""

        if other is None or not isinstance(other, GeobodyInterpretation):
            return False
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
            return True
        if self.geobody_feature is not None:
            if not self.geobody_feature.is_equivalent(other.geobody_feature,
                                                      check_extra_metadata = check_extra_metadata):
                return False
        elif other.geobody_feature is not None:
            return False
        if self.root is not None and other.root is not None:
            if rqet.citation_title_for_node(self.root) != rqet.citation_title_for_node(other.root):
                return False
        elif self.root is not None or other.root is not None:
            return False
        if check_extra_metadata and not ou.equivalent_extra_metadata(self, other):
            return False
        return (self.domain == other.domain and
                ou.equivalent_chrono_pairs(self.has_occurred_during, other.has_occurred_during) and
                self.composition == other.composition and self.implacement == other.implacement and
                self.geobody_shape == other.geobody_shape)

    def create_xml(self,
                   geobody_feature_root = None,
                   add_as_part = True,
                   add_relationships = True,
                   originator = None,
                   title_suffix = None,
                   reuse = True):
        """Creates an XML tree in memory and optionally adds it as a 'part' in the Model"""

        if not self.title:
            self.title = self.geobody_feature.feature_name
        if title_suffix:
            self.title += ' ' + title_suffix

        if reuse and self.try_reuse():
            return self.root
        gi = super().create_xml(add_as_part = False, originator = originator)

        if self.geobody_feature is not None:
            gbf_root = self.geobody_feature.root
            if gbf_root is not None:
                if geobody_feature_root is None:
                    geobody_feature_root = gbf_root
                else:
                    assert gbf_root is geobody_feature_root, 'geobody feature mismatch'
        else:
            if geobody_feature_root is None:
                geobody_feature_root = self.feature_root
            assert geobody_feature_root is not None
            self.geobody_feature = ogf.GeobodyFeature(self.model, uuid = geobody_feature_root.attrib['uuid'])
        self.feature_root = geobody_feature_root

        assert self.domain in self.valid_domains, 'illegal domain value for geobody interpretation'
        dom_node = rqet.SubElement(gi, ns['resqml2'] + 'Domain')
        dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
        dom_node.text = self.domain

        ou.create_xml_has_occurred_during(self.model, gi, self.has_occurred_during)

        if self.composition:
            assert self.composition in self.valid_compositions
            guc_node = rqet.SubElement(gi, ns['resqml2'] + 'GeologicUnitComposition')
            guc_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'GeologicUnitComposition')
            guc_node.text = self.composition
            # if self.composition.startswith('intrusive'): guc_node.text += ' '

        if self.implacement:
            assert self.implacement in self.valid_implacements
            gumi_node = rqet.SubElement(gi, ns['resqml2'] + 'GeologicUnitMaterialImplacement')
            gumi_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'GeologicUnitMaterialImplacement')
            gumi_node.text = self.implacement

        if self.geobody_shape:
            # note: 'silt' & 'sheeth' believed erroneous, so 'sill' and 'sheet' added
            assert self.geobody_shape in self.valid_geobody_shapes
            gs_node = rqet.SubElement(gi, ns['resqml2'] + 'Geobody3dShape')
            gs_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Geobody3dShape')
            gs_node.text = self.geobody_shape

        self.model.create_ref_node('InterpretedFeature',
                                   self.model.title_for_root(geobody_feature_root),
                                   geobody_feature_root.attrib['uuid'],
                                   content_type = 'obj_GeobodyFeature',
                                   root = gi)

        if add_as_part:
            self.model.add_part('obj_GeobodyInterpretation', self.uuid, gi)
            if add_relationships:
                self.model.create_reciprocal_relationship(gi, 'destinationObject', geobody_feature_root, 'sourceObject')

        return gi
