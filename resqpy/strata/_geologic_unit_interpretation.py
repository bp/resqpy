"""RESQML GeologicUnitInterpretation class."""

# NB: in this module, the term 'unit' refers to a geological stratigraphic unit, i.e. a layer of rock, not a unit of measure

import logging

log = logging.getLogger(__name__)

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
import resqpy.strata
import resqpy.strata._strata_common as rqstc
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


class GeologicUnitInterpretation(BaseResqpy):
    """Class for RESQML Geologic Unit Interpretation objects.

    These objects can be parts in their own right. NB: Various more specialised classes also derive from this.

    RESQML documentation:

       The main class for data describing an opinion of a volume-based geologic feature or unit.
    """

    resqml_type = 'GeologicUnitInterpretation'

    def __init__(
            self,
            parent_model,
            uuid = None,
            title = None,
            domain = 'time',  # or should this be depth?
            geologic_unit_feature = None,
            composition = None,
            material_implacement = None,
            extra_metadata = None):
        """Initialises an geologic unit interpretation object.

        arguments:
           parent_model (model.Model): the model with which the new interpretation will be associated
           uuid (uuid.UUID, optional): the uuid of an existing RESQML geologic unit interpretation from which
              this object will be initialised
           title (str, optional): the citation title (feature name) of the new interpretation;
              ignored if uuid is not None
           domain (str, default 'time'): 'time', 'depth' or 'mixed', being the domain of the interpretation;
              ignored if uuid is not None
           geologic_unit_feature (organize.GeologicUnitFeature or StratigraphicUnitFeature, optional): the feature
              which this object is an interpretation of; ignored if uuid is not None
           composition (str, optional): the interpreted composition of the geologic unit; if present, must be
              in valid_compositions; ignored if uuid is not None
           material_implacement (str, optional): the interpeted material implacement of the geologic unit;
              if present, must be in valid_implacements, ie. 'autochtonous' or 'allochtonous';
              ignored if uuid is not None
           extra_metadata (dict, optional): extra metadata items for the new interpretation

        returns:
           a new geologic unit interpretation resqpy object which may be the basis of a derived class object

        note:
           the RESQML 2.0.1 schema definition includes a spurious trailing space in the names of two compositions;
           resqpy removes such spaces in the composition attribute as presented to calling code (but includes them
           in xml)
        """

        self.domain = domain
        self.geologic_unit_feature = geologic_unit_feature  # InterpretedFeature RESQML field
        self.has_occurred_during = (None, None)  # optional RESQML item
        if (not title) and geologic_unit_feature is not None:
            title = geologic_unit_feature.feature_name
        self.composition = composition  # optional RESQML item
        self.material_implacement = material_implacement  # optional RESQML item
        super().__init__(model = parent_model, uuid = uuid, title = title, extra_metadata = extra_metadata)
        if self.composition:
            assert self.composition in rqstc.valid_compositions,  \
               f'invalid composition {self.composition} for geological unit interpretation'
            self.composition = self.composition.strip()
        if self.material_implacement:
            assert self.material_implacement in rqstc.valid_implacements,  \
               f'invalid material implacement {self.material_implacement} for geological unit interpretation'

    def _load_from_xml(self):
        """Loads class specific attributes from xml for an existing RESQML object; called from BaseResqpy."""
        root_node = self.root
        assert root_node is not None
        self.domain = rqet.find_tag_text(root_node, 'Domain')
        # following allows derived StratigraphicUnitInterpretation to instantiate its own interpreted feature
        if self.resqml_type == 'GeologicUnitInterpretation':
            feature_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(root_node, ['InterpretedFeature', 'UUID']))
            if feature_uuid is not None:
                self.geologic_unit_feature = rqo.GeologicUnitFeature(
                    self.model, uuid = feature_uuid, feature_name = self.model.title(uuid = feature_uuid))
        self.has_occurred_during = rqo.extract_has_occurred_during(root_node)
        self.composition = rqet.find_tag_text(root_node, 'GeologicUnitComposition')
        self.material_implacement = rqet.find_tag_text(root_node, 'GeologicUnitMaterialImplacement')

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this interpretation is essentially the same as the other; otherwise False.

        arguments:
           other (GeologicUnitInterpretation or StratigraphicUnitInterpretation): the other interpretation to
              compare this one against
           check_extra_metadata (bool, default True): if True, then extra metadata items must match for the two
              interpretations to be deemed equivalent; if False, extra metadata is ignored in the comparison

        returns:
           bool: True if this interpretation is essentially the same as the other; False otherwise
        """

        # this method is coded to allow use by the derived StratigraphicUnitInterpretation class
        if other is None or not isinstance(other, type(self)):
            return False
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
            return True
        if self.geologic_unit_feature is not None:
            if not self.geologic_unit_feature.is_equivalent(other.geologic_unit_feature):
                return False
        elif other.geologic_unit_feature is not None:
            return False
        if self.root is not None and other.root is not None:
            if rqet.citation_title_for_node(self.root) != rqet.citation_title_for_node(other.root):
                return False
        elif self.root is not None or other.root is not None:
            return False
        if check_extra_metadata and not rqo.equivalent_extra_metadata(self, other):
            return False
        return (self.composition == other.composition and self.material_implacement == other.material_implacement and
                self.domain == other.domain and
                rqo.equivalent_chrono_pairs(self.has_occurred_during, other.has_occurred_during))

    def create_xml(self, add_as_part = True, add_relationships = True, originator = None, reuse = True):
        """Creates a geologic unit interpretation xml tree.

        arguments:
           add_as_part (bool, default True): if True, the interpretation is added to the parent model as a high level part
           add_relationships (bool, default True): if True and add_as_part is True, a relationship is created with
              the referenced geologic unit feature
           originator (str, optional): if present, is used as the originator field of the citation block
           reuse (bool, default True): if True, the parent model is inspected for any equivalent interpretation and, if found,
              the uuid of this interpretation is set to that of the equivalent part

        returns:
           lxml.etree._Element: the root node of the newly created xml tree for the interpretation
        """

        # note: related feature xml must be created first and is referenced here
        # this method is coded to allow use by the derived StratigraphicUnitInterpretation class

        if reuse and self.try_reuse():
            return self.root
        gu = super().create_xml(add_as_part = False, originator = originator)

        assert self.geologic_unit_feature is not None
        guf_root = self.geologic_unit_feature.root
        assert guf_root is not None, 'interpreted feature not established for geologic unit interpretation'

        assert self.domain in rqstc.valid_domains, 'illegal domain value for geologic unit interpretation'
        dom_node = rqet.SubElement(gu, ns['resqml2'] + 'Domain')
        dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
        dom_node.text = self.domain

        self.model.create_ref_node('InterpretedFeature',
                                   self.geologic_unit_feature.title,
                                   self.geologic_unit_feature.uuid,
                                   content_type = self.geologic_unit_feature.resqml_type,
                                   root = gu)

        rqo.create_xml_has_occurred_during(self.model, gu, self.has_occurred_during)

        if self.composition is not None:
            assert self.composition in rqstc.valid_compositions,  \
                f'invalid composition {self.composition} for geologic unit interpretation'
            comp_node = rqet.SubElement(gu, ns['resqml2'] + 'GeologicUnitComposition')
            comp_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'GeologicUnitComposition')
            comp_node.text = self.composition
            if self.composition + ' ' in rqstc.valid_compositions:  # RESQML xsd has spurious trailing space for two compositions
                comp_node.text += ' '

        if self.material_implacement is not None:
            assert self.material_implacement in rqstc.valid_implacements,  \
                f'invalid material implacement {self.material_implacement} for geologic unit interpretation'
            mi_node = rqet.SubElement(gu, ns['resqml2'] + 'GeologicUnitMaterialImplacement')
            mi_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'GeologicUnitMaterialImplacement')
            mi_node.text = self.material_implacement

        if add_as_part:
            self.model.add_part('obj_' + self.resqml_type, self.uuid, gu)
            if add_relationships:
                self.model.create_reciprocal_relationship(gu, 'destinationObject', guf_root, 'sourceObject')

        return gu
