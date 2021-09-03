"""strata.py: RESQML stratigraphy classes."""

version = '24th August 2021'

# NB: in this module, the term 'unit' refers to a geological stratigraphic unit, i.e. a layer of rock, not a unit of measure
# RMS is a registered trademark of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)
log.debug('strata.py version ' + version)

import warnings

import resqpy.organize as rqo
import resqpy.weights_and_measures as wam
import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu
from resqpy.olio.xml_namespaces import curly_namespace as ns
from resqpy.olio.base import BaseResqpy

# note: two compositions have a spurious trailing space in the RESQML xsd; resqpy hides this from calling code
valid_compositions = [
   'intrusive clay ', 'intrusive clay', 'organic', 'intrusive mud ', 'intrusive mud', 'evaporite salt',
   'evaporite non salt', 'sedimentary siliclastic', 'carbonate', 'magmatic intrusive granitoid',
   'magmatic intrusive pyroclastic', 'magmatic extrusive lava flow', 'other chemichal rock', 'sedimentary turbidite'
]

valid_implacements = ['autochtonous', 'allochtonous']

valid_domains = ('depth', 'time', 'mixed')

valid_deposition_modes = [
   'proportional between top and bottom', 'parallel to bottom', 'parallel to top', 'parallel to another boundary'
]

valid_ordering_criteria = ['age', 'apparent depth', 'measured depth']  # stratigraphic column must be ordered by age

valid_contact_relationships = [
   'frontier feature to frontier feature', 'genetic boundary to frontier feature',
   'genetic boundary to genetic boundary', 'genetic boundary to tectonic boundary',
   'stratigraphic unit to frontier feature', 'stratigraphic unit to stratigraphic unit',
   'tectonic boundary to frontier feature', 'tectonic boundary to genetic boundary',
   'tectonic boundary to tectonic boundary'
]

valid_contact_verbs = ['splits', 'interrupts', 'contains', 'erodes', 'stops at', 'crosses', 'includes']

valid_contact_sides = ['footwall', 'hanging wall', 'north', 'south', 'east', 'west', 'younger', 'older', 'both']

valid_contact_modes = ['baselap', 'erosion', 'extended', 'proportional']


class StratigraphicUnitFeature(BaseResqpy):
   """Class for RESQML Stratigraphic Unit Feature objects.

   RESQML documentation:

      A stratigraphic unit that can have a well-known (e.g., "Jurassic") chronostratigraphic top and
      chronostratigraphic bottom. These chronostratigraphic units have no associated interpretations or representations.

      BUSINESS RULE: The name must reference a well-known chronostratigraphic unit (such as "Jurassic"),
      for example, from the International Commission on Stratigraphy (http://www.stratigraphy.org).
   """

   resqml_type = 'StratigraphicUnitFeature'

   def __init__(self,
                parent_model,
                uuid = None,
                top_unit_uuid = None,
                bottom_unit_uuid = None,
                title = None,
                originator = None,
                extra_metadata = None):
      """Initialises a stratigraphic unit feature object.

      arguments:
         parent_model (model.Model): the model with which the new feature will be associated
         uuid (uuid.UUID, optional): the uuid of an existing RESQML stratigraphic unit feature from which
            this object will be initialised
         top_unit_uuid (uuid.UUID, optional): the uuid of a geologic or stratigraphic unit feature which is
            at the top of the new unit; ignored if uuid is not None
         bottom_unit_uuid (uuid.UUID, optional): the uuid of a geologic or stratigraphic unit feature which is
            at the bottom of the new unit; ignored if uuid is not None
         title (str, optional): the citation title (feature name) of the new feature; ignored if uuid is not None
         originator (str, optional): the name of the person creating the new feature; ignored if uuid is not None
         extra_metadata (dict, optional): extra metadata items for the new feature

      returns:
         a new stratigraphic unit feature resqpy object
      """

      # todo: clarify with Energistics whether the 2 references are to other StratigraphicUnitFeatures or what?
      self.top_unit_uuid = top_unit_uuid
      self.bottom_unit_uuid = bottom_unit_uuid

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata)

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False.

      arguments:
         other (StratigraphicUnitFeature): the other feature to compare this one against
         check_extra_metadata (bool, default True): if True, then extra metadata items must match for the two
            features to be deemed equivalent; if False, extra metadata is ignored in the comparison

      returns:
         bool: True if this feature is essentially the same as the other feature; False otherwise
      """

      if not isinstance(other, StratigraphicUnitFeature):
         return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid):
         return True
      return (self.title == other.title and ((not check_extra_metadata) or rqo.equivalent_extra_metadata(self, other)))

   def _load_from_xml(self):
      """Loads class specific attributes from xml for an existing RESQML object; called from BaseResqpy."""
      root_node = self.root
      assert root_node is not None
      bottom_ref_uuid = rqet.find_nested_tags_text(root_node, ['ChronostratigraphicBottom', 'UUID'])
      top_ref_uuid = rqet.find_nested_tags_text(root_node, ['ChronostratigraphicTop', 'UUID'])
      # todo: find out if these are meant to be references to other stratigraphic unit features or geologic unit features
      # and if so, instantiate those objects?
      # for now, simply note the uuids
      if bottom_ref_uuid is not None:
         self.bottom_unit_uuid = bu.uuid_from_string(bottom_ref_uuid)
      if top_ref_uuid is not None:
         self.top_unit_uuid = bu.uuid_from_string(top_ref_uuid)

   def create_xml(self, add_as_part = True, originator = None, reuse = True, add_relationships = True):
      """Creates xml for this stratigraphic unit feature.

      arguments:
         add_as_part (bool, default True): if True, the feature is added to the parent model as a high level part
         originator (str, optional): if present, is used as the originator field of the citation block
         reuse (bool, default True): if True, the parent model is inspected for any equivalent feature and, if found,
            the uuid of this feature is set to that of the equivalent part
         add_relationships (bool, default True): if True and add_as_part is True, relationships are created with
            the referenced top and bottom units, if present

      returns:
         lxml.etree._Element: the root node of the newly created xml tree for the feature
      """

      if reuse and self.try_reuse():
         return self.root  # check for reusable (equivalent) object
      # create node with citation block
      suf = super().create_xml(add_as_part = False, originator = originator)

      if self.bottom_unit_uuid is not None:
         self.model.create_ref_node('ChronostratigraphicBottom',
                                    self.model.title(uuid = self.bottom_unit_uuid),
                                    self.bottom_unit_uuid,
                                    content_type = self.model.type_of_uuid(self.bottom_unit_uuid),
                                    root = suf)

      if self.top_unit_uuid is not None:
         self.model.create_ref_node('ChronostratigraphicTop',
                                    self.model.title(uuid = self.top_unit_uuid),
                                    self.top_unit_uuid,
                                    content_type = self.model.type_of_uuid(self.top_unit_uuid),
                                    root = suf)

      if add_as_part:
         self.model.add_part('obj_StratigraphicUnitFeature', self.uuid, suf)
         if add_relationships:
            if self.bottom_unit_uuid is not None:
               self.model.create_reciprocal_relationship(suf, 'destinationObject',
                                                         self.model.root(uuid = self.bottom_unit_uuid), 'sourceObject')
            if self.top_unit_uuid is not None and not bu.matching_uuids(self.bottom_unit_uuid, self.top_unit_uuid):
               self.model.create_reciprocal_relationship(suf, 'destinationObject',
                                                         self.model.root(uuid = self.top_unit_uuid), 'sourceObject')

      return suf


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
         assert self.composition in valid_compositions,  \
            f'invalid composition {self.composition} for geological unit interpretation'
         self.composition = self.composition.strip()
      if self.material_implacement:
         assert self.material_implacement in valid_implacements,  \
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
            self.geologic_unit_feature = rqo.GeologicUnitFeature(self.model,
                                                                 uuid = feature_uuid,
                                                                 feature_name = self.model.title(uuid = feature_uuid))
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

      assert self.domain in valid_domains, 'illegal domain value for geologic unit interpretation'
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
         assert self.composition in valid_compositions, f'invalid composition {self.composition} for geologic unit interpretation'
         comp_node = rqet.SubElement(gu, ns['resqml2'] + 'GeologicUnitComposition')
         comp_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'GeologicUnitComposition')
         comp_node.text = self.composition
         if self.composition + ' ' in valid_compositions:  # RESQML xsd has spurious trailing space for two compositions
            comp_node.text += ' '

      if self.material_implacement is not None:
         assert self.material_implacement in valid_implacements,  \
            f'invalid material implacement {self.material_implacement} for geologic unit interpretation'
         mi_node = rqet.SubElement(gu, ns['resqml2'] + 'GeologicUnitMaterialImplacement')
         mi_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'GeologicUnitMaterialImplacement')
         mi_node.text = self.material_implacement

      if add_as_part:
         self.model.add_part('obj_' + self.resqml_type, self.uuid, gu)
         if add_relationships:
            self.model.create_reciprocal_relationship(gu, 'destinationObject', guf_root, 'sourceObject')

      return gu


class StratigraphicUnitInterpretation(GeologicUnitInterpretation):
   """Class for RESQML Stratigraphic Unit Interpretation objects.

   RESQML documentation:

      Interpretation of a stratigraphic unit which includes the knowledge of the top, the bottom,
      the deposition mode.
   """

   resqml_type = 'StratigraphicUnitInterpretation'

   def __init__(
         self,
         parent_model,
         uuid = None,
         title = None,
         domain = 'time',  # or should this be depth?
         stratigraphic_unit_feature = None,
         composition = None,
         material_implacement = None,
         deposition_mode = None,
         min_thickness = None,
         max_thickness = None,
         thickness_uom = None,
         extra_metadata = None):
      """Initialises a stratigraphic unit interpretation object.

      arguments:
         parent_model (model.Model): the model with which the new interpretation will be associated
         uuid (uuid.UUID, optional): the uuid of an existing RESQML stratigraphic unit interpretation from which
            this object will be initialised
         title (str, optional): the citation title (feature name) of the new interpretation;
            ignored if uuid is not None
         domain (str, default 'time'): 'time', 'depth' or 'mixed', being the domain of the interpretation;
            ignored if uuid is not None
         stratigraphic_unit_feature (StratigraphicUnitFeature, optional): the feature which this object is
            an interpretation of; ignored if uuid is not None
         composition (str, optional): the interpreted composition of the stratigraphic unit; if present, must be
            in valid_compositions; ignored if uuid is not None
         material_implacement (str, optional): the interpeted material implacement of the stratigraphic unit;
            if present, must be in valid_implacements, ie. 'autochtonous' or 'allochtonous';
            ignored if uuid is not None
         deposition_mode (str, optional): indicates whether deposition within the unit is interpreted as parallel
            to top, base or another boundary, or is proportional to thickness; if present, must be in
            valid_deposition_modes; ignored if uuid is not None
         min_thickness (float, optional): the minimum thickness of the unit; ignored if uuid is not None
         max_thickness (float, optional): the maximum thickness of the unit; ignored if uuid is not None
         thickness_uom (str, optional): the length unit of measure of the minimum and maximum thickness; required
            if either thickness argument is provided and uuid is None; if present, must be a valid length uom
         extra_metadata (dict, optional): extra metadata items for the new interpretation

      returns:
         a new stratigraphic unit interpretation resqpy object

      notes:
         if given, the thickness_uom must be a valid RESQML length unit of measure; the set of valid uoms is
         returned by: weights_and_measures.valid_uoms(quantity = 'length');
         the RESQML 2.0.1 schema definition includes a spurious trailing space in the names of two compositions;
         resqpy removes such spaces in the composition attribute as presented to calling code (but includes them
         in xml)
      """

      self.deposition_mode = deposition_mode
      self.min_thickness = min_thickness
      self.max_thickness = max_thickness
      self.thickness_uom = thickness_uom
      super().__init__(parent_model,
                       uuid = uuid,
                       title = title,
                       domain = domain,
                       geologic_unit_feature = stratigraphic_unit_feature,
                       composition = composition,
                       material_implacement = material_implacement,
                       extra_metadata = extra_metadata)
      if self.deposition_mode is not None:
         assert self.deposition_mode in valid_deposition_modes
      if self.min_thickness is not None or self.max_thickness is not None:
         assert self.thickness_uom in wam.valid_uoms(quantity = 'length')

   @property
   def stratigraphic_unit_feature(self):
      return self.geologic_unit_feature

   def _load_from_xml(self):
      """Loads class specific attributes from xml for an existing RESQML object; called from BaseResqpy."""
      super()._load_from_xml()
      root_node = self.root
      assert root_node is not None
      feature_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(root_node, ['InterpretedFeature', 'UUID']))
      if feature_uuid is not None:
         self.geologic_unit_feature = StratigraphicUnitFeature(self.model,
                                                               uuid = feature_uuid,
                                                               title = self.model.title(uuid = feature_uuid))
      # load deposition mode and min & max thicknesses (& uom), if present
      self.deposition_mode = rqet.find_tag_text(root_node, 'DepositionMode')
      for min_max in ['Min', 'Max']:
         thick_node = rqet.find_tag(root_node, min_max + 'Thickness')
         if thick_node is not None:
            thick = float(thick_node.text)
            if min_max == 'Min':
               self.min_thickness = thick
            else:
               self.max_thickness = thick
            thick_uom = thick_node.attrib['uom']  # todo: check this is correct uom representation
            if self.thickness_uom is None:
               self.thickness_uom = thick_uom
            else:
               assert thick_uom == self.thickness_uom, 'inconsistent length units of measure for stratigraphic thicknesses'

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this interpretation is essentially the same as the other; otherwise False.

      arguments:
         other (StratigraphicUnitInterpretation): the other interpretation to compare this one against
         check_extra_metadata (bool, default True): if True, then extra metadata items must match for the two
            interpretations to be deemed equivalent; if False, extra metadata is ignored in the comparison

      returns:
         bool: True if this interpretation is essentially the same as the other; False otherwise
      """
      if not super().is_equivalent(other):
         return False
      if self.deposition_mode is not None and other.deposition_mode is not None:
         return self.deposition_mode == other.deposition_mode
      # note: thickness range information might be lost as not deemed of significance in comparison
      return True

   def create_xml(self, add_as_part = True, add_relationships = True, originator = None, reuse = True):
      """Creates a stratigraphic unit interpretation xml tree.

      arguments:
         add_as_part (bool, default True): if True, the interpretation is added to the parent model as a high level part
         add_relationships (bool, default True): if True and add_as_part is True, a relationship is created with
            the referenced stratigraphic unit feature
         originator (str, optional): if present, is used as the originator field of the citation block
         reuse (bool, default True): if True, the parent model is inspected for any equivalent interpretation and, if found,
            the uuid of this interpretation is set to that of the equivalent part

      returns:
         lxml.etree._Element: the root node of the newly created xml tree for the interpretation
      """

      if reuse and self.try_reuse():
         return self.root

      sui = super().create_xml(add_as_part = add_as_part,
                               add_relationships = add_relationships,
                               originator = originator,
                               reuse = False)
      assert sui is not None

      if self.deposition_mode is not None:
         assert self.deposition_mode in valid_deposition_modes,  \
            f'invalid deposition mode {self.deposition_mode} for stratigraphic unit interpretation'
         dm_node = rqet.SubElement(sui, ns['resqml2'] + 'DepositionMode')
         dm_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'DepositionMode')
         dm_node.text = self.deposition_mode

      if self.min_thickness is not None or self.max_thickness is not None:
         assert self.thickness_uom in wam.valid_uoms(quantity = 'length')

      if self.min_thickness is not None:
         min_thick_node = rqet.SubElement(sui, ns['resqml2'] + 'MinThickness')
         min_thick_node.set(ns['xsi'] + 'type', ns['eml'] + 'LengthMeasure')
         min_thick_node.set('uom', self.thickness_uom)  # todo: check this
         min_thick_node.text = str(self.min_thickness)

      if self.max_thickness is not None:
         max_thick_node = rqet.SubElement(sui, ns['resqml2'] + 'MaxThickness')
         max_thick_node.set(ns['xsi'] + 'type', ns['eml'] + 'LengthMeasure')
         max_thick_node.set('uom', self.thickness_uom)
         max_thick_node.text = str(self.max_thickness)

      return sui


class StratigraphicColumn(BaseResqpy):
   """Class for RESQML stratigraphic column objects.

   RESQML documentation:

      A global interpretation of the stratigraphy, which can be made up of
      several ranks of stratigraphic unit interpretations.

      All stratigraphic column rank interpretations that make up a stratigraphic column
      must be ordered by age.
   """

   resqml_type = 'StratigraphicColumn'

   # todo: integrate with EarthModelInterpretation class which can optionally refer to a stratigraphic column

   def __init__(self, parent_model, uuid = None, rank_uuid_list = None, title = None, extra_metadata = None):
      """Initialises a Stratigraphic Column object.

      arguments:
         parent_model (model.Model): the model with which the new stratigraphic column will be associated
         uuid (uuid.UUID, optional): the uuid of an existing RESQML stratigraphic column from which
            this object will be initialised
         rank_uuid_list (list of uuid, optional): if not initialising for an existing stratigraphic column,
            the ranks can be established from this list of uuids for existing stratigraphic column ranks;
            ranks should be ordered from geologically oldest to youngest with increasing index values
         title (str, optional): the citation title of the new stratigraphic column; ignored if uuid is not None
         extra_metadata (dict, optional): extra metadata items for the new feature
      """

      self.ranks = []  # list of Stratigraphic Column Rank Interpretation objects, maintained in rank index order

      super().__init__(model = parent_model, uuid = uuid, title = title, extra_metadata = extra_metadata)

      if self.root is None and rank_uuid_list:
         for rank_uuid in rank_uuid_list:
            rank = StratigraphicColumnRank(self.model, uuid = rank_uuid)
            self.add_rank(rank)

   def _load_from_xml(self):
      """Loads class specific attributes from xml for an existing RESQML object; called from BaseResqpy."""
      rank_node_list = rqet.list_of_tag(self.root, 'Ranks')
      assert rank_node_list is not None, 'no stratigraphic column ranks in xml for stratigraphic column'
      for rank_node in rank_node_list:
         rank = StratigraphicColumnRank(self.model, uuid = rqet.find_tag_text(rank_node, 'UUID'))
         self.add_rank(rank)

   def iter_ranks(self):
      """Yields the stratigraphic column ranks which constitute this stratigraphic colunn."""

      for rank in self.ranks:
         yield rank

   def add_rank(self, rank):
      """Adds another stratigraphic column rank to this stratigraphic column.

      arguments:
         rank (StratigraphicColumnRank): an established rank to be added to this stratigraphic column

      note:
         ranks should be ordered from geologically oldest to youngest, with increasing index values
      """

      assert rank is not None and rank.index is not None
      self.ranks.append(rank)
      self._sort_ranks()

   def _sort_ranks(self):
      """Sort the list of ranks, in situ, by their index values."""
      self.ranks.sort(key = _index_attr)

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this interpretation is essentially the same as the other; otherwise False.

      arguments:
         other (StratigraphicColumn): the other stratigraphic column to compare this one against
         check_extra_metadata (bool, default True): if True, then extra metadata items must match for the two
            columns to be deemed equivalent; if False, extra metadata is ignored in the comparison

      returns:
         bool: True if this stratigraphic column is essentially the same as the other; False otherwise
      """

      if not isinstance(other, StratigraphicColumn):
         return False
      if self is other or bu.matching_uuid(self.uuid, other.uuid):
         return True
      if len(self.ranks) != len(other.ranks):
         return False
      for rank_a, rank_b in zip(self.ranks, other.ranks):
         if rank_a != rank_b:
            return False
      if check_extra_metadata and not rqo.equivalent_extra_metadata(self, other):
         return False
      return True

   def create_xml(self, add_as_part = True, add_relationships = True, originator = None, reuse = True):
      """Creates xml tree for a stratigraphic column object.

      arguments:
         add_as_part (bool, default True): if True, the stratigraphic column is added to the parent model as a high level part
         add_relationships (bool, default True): if True and add_as_part is True, a relationship is created with each of
            the referenced stratigraphic column rank interpretations
         originator (str, optional): if present, is used as the originator field of the citation block
         reuse (bool, default True): if True, the parent model is inspected for any equivalent stratigraphic column and,
            if found, the uuid of this stratigraphic column is set to that of the equivalent part

      returns:
         lxml.etree._Element: the root node of the newly created xml tree for the stratigraphic column
      """

      assert self.ranks, 'attempt to create xml for stratigraphic column without any contributing ranks'

      if reuse and self.try_reuse():
         return self.root

      sci = super().create_xml(add_as_part = False, originator = originator)

      assert sci is not None

      for rank in self.ranks:
         self.model.create_ref_node('Ranks',
                                    rank.title,
                                    rank.uuid,
                                    content_type = 'obj_StratigraphicColumnRankInterpretation',
                                    root = sci)

      if add_as_part:
         self.model.add_part('obj_StratigraphicColumn', self.uuid, sci)
         if add_relationships:
            for rank in self.ranks:
               self.model.create_reciprocal_relationship(sci, 'destinationObject', rank.root, 'sourceObject')

      return sci


class StratigraphicColumnRank(BaseResqpy):
   """Class for RESQML StratigraphicColumnRankInterpretation objects.

   A list of stratigraphic unit interpretations, ordered from geologically oldest to youngest.
   """

   resqml_type = 'StratigraphicColumnRankInterpretation'

   # note: ordering of stratigraphic units and contacts is from geologically oldest to youngest

   def __init__(
         self,
         parent_model,
         uuid = None,
         domain = 'time',  # or should this be depth?
         rank_index = None,
         earth_model_feature_uuid = None,
         strata_uuid_list = None,  # ordered list of stratigraphic unit interpretations
         title = None,
         extra_metadata = None):
      """Initialises a Stratigraphic Column Rank resqpy object (RESQML StratigraphicColumnRankInterpretation).

      arguments:
         parent_model (model.Model): the model with which the new interpretation will be associated
         uuid (uuid.UUID, optional): the uuid of an existing RESQML stratigraphic column rank interpretation
            from which this object will be initialised
         domain (str, default 'time'): 'time', 'depth' or 'mixed', being the domain of the interpretation;
            ignored if uuid is not None
         rank_index (int, optional): the rank index (RESQML index) for this rank when multiple ranks are used;
            will default to zero if not set; ignored if uuid is not None
         earth_model_feature_uuid (uuid.UUID, optional): the uuid of an existing organization feature of kind
            'earth model' which this stratigraphic column is for; ignored if uuid is not None
         strata_uuid_list (list of uuid.UUID, optional): a list of uuids of existing stratigraphic unit
            interpretations, ordered from geologically oldest to youngest; ignored if uuid is not None
         title (str, optional): the citation title (feature name) of the new interpretation;
            ignored if uuid is not None
         extra_metadata (dict, optional): extra metadata items for the new stratigraphic column rank interpretation
      """

      self.feature_uuid = earth_model_feature_uuid  # interpreted earth model feature uuid
      self.domain = domain
      self.index = rank_index
      self.has_occurred_during = (None, None)  # optional RESQML item
      self.units = []  # ordered list of (index, stratagraphic unit interpretation uuid)
      self.contacts = []  # optional ordered list of binary contact interpretations

      super().__init__(model = parent_model, uuid = uuid, title = title, extra_metadata = extra_metadata)

      if self.root is None and strata_uuid_list is not None:
         self.set_units(strata_uuid_list)

   def _load_from_xml(self):
      """Loads class specific attributes from xml for an existing RESQML object; called from BaseResqpy."""
      root_node = self.root
      assert root_node is not None
      assert rqet.find_tag_text(root_node, 'OrderingCriteria') == 'age',  \
         'stratigraphic column rank interpretation ordering criterion must be age'
      self.domain = rqet.find_tag_text(root_node, 'Domain')
      self.feature_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(root_node, ['InterpretedFeature', 'UUID']))
      self.has_occurred_during = rqo.extract_has_occurred_during(root_node)
      self.index = rqet.find_tag_int(root_node, 'Index')
      self.units = []
      for su_node in rqet.list_of_tag(root_node, 'StratigraphicUnits'):
         index = rqet.find_tag_int(su_node, 'Index')
         unit_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(su_node, ['Unit', 'UUID']))
         assert index is not None and unit_uuid is not None
         assert self.model.type_of_uuid(unit_uuid, strip_obj = True) == 'StratigraphicUnitInterpretation'
         self.units.append((index, StratigraphicUnitInterpretation(self.model, uuid = unit_uuid)))
      self._sort_units()
      self.contacts = []
      for contact_node in rqet.list_of_tag(root_node, 'ContactInterpretation'):
         self.contacts.append(BinaryContactInterpretation(self.model, existing_xml_node = contact_node))
      self._sort_contacts()

   def set_units(self, strata_uuid_list):
      """Discard any previous units and set list of units based on ordered list of uuids.

      arguments:
         strata_uuid_list (list of uuid.UUID): the uuids of the stratigraphic unit interpretations which constitute
            this stratigraphic column, ordered from geologically oldest to youngest
      """

      self.units = []
      for i, uuid in enumerate(strata_uuid_list):
         assert self.model.type_of_uuid(uuid, strip_obj = True) == 'StratigraphicUnitInterpretation'
         self.units.append((i, StratigraphicUnitInterpretation(self.model, uuid = uuid)))

   def _sort_units(self):
      """Sorts units list, in situ, into increasing order of index values."""
      self.units.sort()

   def _sort_contacts(self):
      """Sorts contacts list, in situ, into increasing order of index values."""
      self.contacts.sort(key = _index_attr)

   def set_contacts_from_horizons(self, horizon_uuids, older_contact_mode = None, younger_contact_mode = None):
      """Sets the list of contacts from an ordered list of horizons, of length one less than the number of units.

      arguments:
         horizon_uuids (list of uuid.UUID): list of horizon interpretation uuids, ordered from geologically oldest
            to youngest, with one horizon for each neighbouring pair of stratigraphic units in this column
         older_contact_mode (str, optional): if present, the contact mode to set for the older unit for all of the
            contacts; must be in valid_contact_modes
         younger_contact_mode (str, optional): if present, the contact mode to set for the younger unit for all of
            the contacts; must be in valid_contact_modes

      notes:
         units must be established and sorted before calling this method; any previous contacts are discarded;
         if differing modes are required for the various contacts, leave the contact mode arguments as None,
         then interate over the contacts setting each mode individually (after this method returns)
      """

      # this method uses older unit as subject, younger as direct object
      # todo: check this is consistent with any RESQML usage guidance and/or any RMS usage

      self.contacts = []
      assert len(horizon_uuids) == len(self.units) - 1
      for i, horizon_uuid in enumerate(horizon_uuids):
         assert self.model.type_of_uuid(horizon_uuid, strip_obj = True) == 'HorizonInterpretation'
         contact = BinaryContactInterpretation(self.model,
                                               index = i,
                                               contact_relationship = 'stratigraphic unit to stratigraphic unit',
                                               verb = 'stops at',
                                               subject_uuid = self.units[i][1].uuid,
                                               direct_object_uuid = self.units[i + 1][1].uuid,
                                               subject_contact_side = 'older',
                                               subject_contact_mode = older_contact_mode,
                                               direct_object_contact_side = 'younger',
                                               direct_object_contact_mode = younger_contact_mode,
                                               part_of_uuid = horizon_uuid)
         self.contacts.append(contact)

   def iter_units(self):
      """Yields the ordered stratigraphic unit interpretations which constitute this stratigraphic colunn."""

      for _, unit in self.units:
         yield unit

   def unit_for_unit_index(self, index):
      """Returns the stratigraphic unit interpretation with the given index in this stratigraphic colunn."""

      for i, unit in self.units:
         if i == index:
            return unit
      return None

   def iter_contacts(self):
      """Yields the internal binary contact interpretations of this stratigraphic colunn."""

      for contact in self.contacts:
         yield contact

   # todo: implement a strict validation method checking contact exists for each neighbouring unit pair

   def create_xml(self, add_as_part = True, add_relationships = True, originator = None, reuse = True):
      """Creates a stratigraphic column rank interpretation xml tree.

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

      # note: xml for referenced objects must be created before calling this method

      assert len(self.units), 'attempting to create xml for stratigraphic column rank without any units'

      if reuse and self.try_reuse():
         return self.root

      if self.index is None:
         self.index = 0

      scri = super().create_xml(add_as_part = False, originator = originator)

      assert self.feature_uuid is not None
      assert self.model.type_of_uuid(self.feature_uuid, strip_obj = True) == 'OrganizationFeature'
      # todo: could also check that the kind is 'earth model'
      self.model.create_ref_node('InterpretedFeature',
                                 self.model.title(uuid = self.feature_uuid),
                                 self.feature_uuid,
                                 content_type = 'OrganizationFeature',
                                 root = scri)

      assert self.domain in valid_domains, 'illegal domain value for stratigraphic column rank interpretation'
      dom_node = rqet.SubElement(scri, ns['resqml2'] + 'Domain')
      dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
      dom_node.text = self.domain

      rqo.create_xml_has_occurred_during(self.model, scri, self.has_occurred_during)

      oc_node = rqet.SubElement(scri, ns['resqml2'] + 'OrderingCriteria')
      oc_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'OrderingCriteria')
      oc_node.text = 'age'

      i_node = rqet.SubElement(scri, ns['resqml2'] + 'Index')
      i_node.set(ns['xsi'] + 'type', ns['xsi'] + 'nonNegativeInteger')
      i_node.text = str(self.index)

      for i, unit in self.units:

         su_node = rqet.SubElement(scri, ns['resqml2'] + 'StratigraphicUnits')
         su_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'StratigraphicUnitInterpretationIndex')
         su_node.text = ''

         si_node = rqet.SubElement(su_node, ns['resqml2'] + 'Index')
         si_node.set(ns['xsi'] + 'type', ns['xsi'] + 'nonNegativeInteger')
         si_node.text = str(i)

         self.model.create_ref_node('Unit',
                                    unit.title,
                                    unit.uuid,
                                    content_type = 'StratigraphicUnitInterpretation',
                                    root = su_node)

      for contact in self.contacts:
         contact.create_xml(scri)

      if add_as_part:
         self.model.add_part('obj_StratigraphicColumnRankInterpretation', self.uuid, scri)
         if add_relationships:
            emi_root = self.model.root(uuid = self.feature_uuid)
            self.model.create_reciprocal_relationship(scri, 'destinationObject', emi_root, 'sourceObject')
            for _, unit in self.units:
               self.model.create_reciprocal_relationship(scri, 'destinationObject', unit.root, 'sourceObject')
            for contact in self.contacts:
               if contact.part_of_uuid is not None:
                  horizon_root = self.model.root(uuid = contact.part_of_uuid)
                  if horizon_root is not None:
                     self.model.create_reciprocal_relationship(scri, 'destinationObject', horizon_root, 'sourceObject')

      return scri


class BinaryContactInterpretation:
   """Internal class for contact between 2 geological entities; not a high level class but used by others."""

   def __init__(
         self,
         model,
         existing_xml_node = None,
         index = None,
         contact_relationship: str = None,
         verb: str = None,
         subject_uuid = None,
         direct_object_uuid = None,
         subject_contact_side = None,  # optional
         subject_contact_mode = None,  # optional
         direct_object_contact_side = None,  # optional
         direct_object_contact_mode = None,  # optional
         part_of_uuid = None):  # optional
      """Creates a new binary contact interpretation internal object.

      note:
         if an existing xml node is present, then all the later arguments are ignored
      """
      # index (non-negative integer, should increase with decreasing age for horizon contacts)
      # contact relationship (one of valid_contact_relationships)
      # subject (reference to e.g. stratigraphic unit interpretation)
      # verb (one of valid_contact_verbs)
      # direct object (reference to e.g. stratigraphic unit interpretation)
      # contact sides and modes (optional)
      # part of (reference to e.g. horizon interpretation, optional)

      self.model = model

      if existing_xml_node is not None:
         self._load_from_xml(existing_xml_node)

      else:
         assert index >= 0
         assert contact_relationship in valid_contact_relationships
         assert verb in valid_contact_verbs
         assert subject_uuid is not None and direct_object_uuid is not None
         if subject_contact_side is not None:
            assert subject_contact_side in valid_contact_sides
         if subject_contact_mode is not None:
            assert subject_contact_mode in valid_contact_modes
         if direct_object_contact_side is not None:
            assert direct_object_contact_side in valid_contact_sides
         if direct_object_contact_mode is not None:
            assert direct_object_contact_mode in valid_contact_modes
         self.index = index
         self.contact_relationship = contact_relationship
         self.verb = verb
         self.subject_uuid = subject_uuid
         self.direct_object_uuid = direct_object_uuid
         self.subject_contact_side = subject_contact_side
         self.subject_contact_mode = subject_contact_mode
         self.direct_object_contact_side = direct_object_contact_side
         self.direct_object_contact_mode = direct_object_contact_mode
         self.part_of_uuid = part_of_uuid

   def _load_from_xml(self, bci_node):
      """Populates this binary contact interpretation based on existing xml.

      arguments:
         bci_node (lxml.etree._Element): the root xml node for the binary contact interpretation sub-tree
      """

      assert bci_node is not None

      self.contact_relationship = rqet.find_tag_text(bci_node, 'ContactRelationship')
      assert self.contact_relationship in valid_contact_relationships,  \
         f'missing or invalid contact relationship {self.contact_relationship} in xml for binary contact interpretation'

      self.index = rqet.find_tag_int(bci_node, 'Index')
      assert self.index is not None, 'missing index in xml for binary contact interpretation'

      self.part_of_uuid = bu.uuid_from_string(rqet.find_nested_tags_text(bci_node, ['PartOf', 'UUID']))

      sr_node = rqet.find_tag(bci_node, 'Subject')
      assert sr_node is not None, 'missing subject in xml for binary contact interpretation'
      self.subject_uuid = bu.uuid_from_string(rqet.find_tag_text(sr_node, 'UUID'))
      assert self.subject_uuid is not None
      self.subject_contact_side = rqet.find_tag_text(sr_node, 'Qualifier')
      self.subject_contact_mode = rqet.find_tag_text(sr_node, 'SecondaryQualifier')

      dor_node = rqet.find_tag(bci_node, 'DirectObject')
      assert dor_node is not None, 'missing direct object in xml for binary contact interpretation'
      self.direct_object_uuid = bu.uuid_from_string(rqet.find_tag_text(dor_node, 'UUID'))
      assert self.direct_object_uuid is not None
      self.direct_object_contact_side = rqet.find_tag_text(dor_node, 'Qualifier')
      self.direct_object_contact_mode = rqet.find_tag_text(dor_node, 'SecondaryQualifier')

      self.verb = rqet.find_tag_text(bci_node, 'Verb')
      assert self.verb in valid_contact_verbs,  \
         f'missing or invalid contact verb {self.verb} in xml for binary contact interpretation'

   def create_xml(self, parent_node = None):
      """Generates xml sub-tree for this contact interpretation, for inclusion as element of high level interpretation.

      arguments:
         parent_node (lxml.etree._Element, optional): if present, the created sub-tree is added as a child to this node

      returns:
         lxml.etree._Element: the root node of the newly created xml sub-tree for the contact interpretation
      """

      bci = rqet.Element(ns['resqml2'] + 'ContactInterpretation')
      bci.set(ns['xsi'] + 'type', ns['resqml2'] + 'BinaryContactInterpretationPart')
      bci.text = ''

      cr_node = rqet.SubElement(bci, ns['resqml2'] + 'ContactRelationship')
      cr_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ContactRelationship')
      cr_node.text = self.contact_relationship

      i_node = rqet.SubElement(bci, ns['resqml2'] + 'Index')
      i_node.set(ns['xsi'] + 'type', ns['xsi'] + 'nonNegativeInteger')
      i_node.text = str(self.index)

      if self.part_of_uuid is not None:
         self.model.create_ref_node('PartOf',
                                    self.model.title(uuid = self.part_of_uuid),
                                    self.part_of_uuid,
                                    content_type = self.model.type_of_uuid(self.part_of_uuid),
                                    root = bci)

      dor_node = self.model.create_ref_node('DirectObject',
                                            self.model.title(uuid = self.direct_object_uuid),
                                            self.direct_object_uuid,
                                            content_type = self.model.type_of_uuid(self.direct_object_uuid),
                                            root = bci)
      dor_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ContactElementReference')

      if self.direct_object_contact_side:
         doq_node = rqet.SubElement(dor_node, ns['resqml2'] + 'Qualifier')
         doq_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ContactSide')
         doq_node.text = self.direct_object_contact_side

      if self.direct_object_contact_mode:
         dosq_node = rqet.SubElement(dor_node, ns['resqml2'] + 'SecondaryQualifier')
         dosq_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ContactMode')
         dosq_node.text = self.direct_object_contact_mode

      v_node = rqet.SubElement(bci, ns['resqml2'] + 'Verb')
      v_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ContactVerb')
      v_node.text = self.verb

      sr_node = self.model.create_ref_node('Subject',
                                           self.model.title(uuid = self.subject_uuid),
                                           self.subject_uuid,
                                           content_type = self.model.type_of_uuid(self.subject_uuid),
                                           root = bci)
      sr_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ContactElementReference')

      if self.subject_contact_side:
         sq_node = rqet.SubElement(sr_node, ns['resqml2'] + 'Qualifier')
         sq_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ContactSide')
         sq_node.text = self.subject_contact_side

      if self.subject_contact_mode:
         ssq_node = rqet.SubElement(sr_node, ns['resqml2'] + 'SecondaryQualifier')
         ssq_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ContactMode')
         ssq_node.text = self.subject_contact_mode

      if parent_node is not None:
         parent_node.append(bci)

      return bci


def _index_attr(obj):
   """Returns the index attribute of any object – typically used as a sort key function."""
   return obj.index
