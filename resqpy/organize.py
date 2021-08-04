"""organize.py: RESQML Feature and Interpretation classes."""

version = '2nd July 2021'

# For now, fault features and interpretations, plus stubs for horizons

import logging

log = logging.getLogger(__name__)
log.debug('organize.py version ' + version)

# import xml.etree.ElementTree as et
# from lxml import etree as et

import math as maths
import warnings

import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu
from resqpy.olio.xml_namespaces import curly_namespace as ns
from resqpy.olio.base import BaseResqpy


def extract_has_occurred_during(parent_node, tag = 'HasOccuredDuring'):  # RESQML Occured (stet)
   """Extracts UUIDs of chrono bottom and top from xml for has occurred during sub-node, or (None, None)."""
   hod_node = rqet.find_tag(parent_node, tag)
   if hod_node is None:
      return (None, None)
   else:
      return (rqet.find_nested_tags_text(hod_node, ['ChronoBottom', 'UUID']),
              rqet.find_nested_tags_text(hod_node, ['ChronoTop', 'UUID']))


def equivalent_chrono_pairs(pair_a, pair_b, model = None):
   if pair_a == pair_b:
      return True
   if pair_a is None or pair_b is None:
      return False
   if pair_a == (None, None) or pair_b == (None, None):
      return False
   if model is not None:
      # todo: compare chrono info by looking up xml based on the uuids
      pass
   return False  # cautious


def equivalent_extra_metadata(a, b):
   a_has = hasattr(a, 'extra_metadata')
   b_has = hasattr(b, 'extra_metadata')
   if a_has:
      a_em = a.extra_metadata
      a_has = len(a_em) > 0
   else:
      a_em = rqet.load_metadata_from_xml(a.root)
      a_has = a_em is not None and len(a_em) > 0
   if b_has:
      b_em = b.extra_metadata
      b_has = len(b_em) > 0
   else:
      b_em = rqet.load_metadata_from_xml(b.root)
      b_has = b_em is not None and len(b_em) > 0
   if a_has != b_has:
      return False
   if not a_has:
      return True
   return a_em == b_em


def create_xml_has_occurred_during(model, parent_node, hod_pair, tag = 'HasOccuredDuring'):
   if hod_pair is None:
      return
   base_chrono_uuid, top_chrono_uuid = hod_pair
   if base_chrono_uuid is None or top_chrono_uuid is None:
      return
   hod_node = rqet.SubElement(parent_node, tag)
   hod_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'TimeInterval')
   hod_node.text = rqet.null_xml_text
   chrono_base_root = model.root_for_uuid(base_chrono_uuid)
   chrono_top_root = model.root_for_uuid(top_chrono_uuid)
   model.create_ref_node('ChronoBottom', model.title_for_root(chrono_base_root), base_chrono_uuid, root = hod_node)
   model.create_ref_node('ChronoTop', model.title_for_root(chrono_top_root), top_chrono_uuid, root = hod_node)


def _alias_for_attribute(attribute_name):
   """Return an attribute that is a direct alias for an existing attribute"""

   def fget(self):
      return getattr(self, attribute_name)

   def fset(self, value):
      return setattr(self, attribute_name, value)

   return property(fget, fset, doc = f"Alias for {attribute_name}")


class OrganizationFeature(BaseResqpy):
   """Class for generic RESQML Organization Feature objects."""

   resqml_type = "OrganizationFeature"
   feature_name = _alias_for_attribute("title")

   def __init__(self,
                parent_model,
                root_node = None,
                uuid = None,
                feature_name = None,
                organization_kind = None,
                originator = None,
                extra_metadata = None):
      """Initialises an organization feature object."""

      self.organization_kind = organization_kind
      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = feature_name,
                       originator = originator,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if not isinstance(other, OrganizationFeature):
         return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid):
         return True
      return (self.feature_name == other.feature_name and self.organization_kind == other.organization_kind and
              ((not check_extra_metadata) or equivalent_extra_metadata(self, other)))

   def _load_from_xml(self):
      self.organization_kind = rqet.find_tag_text(self.root, 'OrganizationKind')

   def create_xml(self, add_as_part = True, originator = None, reuse = True):
      """Creates an organization feature xml node from this organization feature object."""

      if reuse and self.try_reuse():
         return self.root  # check for reusable (equivalent) object
      # create node with citation block
      ofn = super().create_xml(add_as_part = False, originator = originator)

      # Extra element for organization_kind
      if self.organization_kind not in ['earth model', 'fluid', 'stratigraphic', 'structural']:
         raise ValueError(self.organization_kind)
      kind_node = rqet.SubElement(ofn, ns['resqml2'] + 'OrganizationKind')
      kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'OrganizationKind')
      kind_node.text = self.organization_kind

      if add_as_part:
         self.model.add_part('obj_OrganizationFeature', self.uuid, ofn)

      return ofn


class GeobodyFeature(BaseResqpy):
   """Class for RESQML Geobody Feature objects (note: definition may be incomplete in RESQML 2.0.1)."""

   resqml_type = "GeobodyFeature"
   feature_name = _alias_for_attribute("title")

   def __init__(self, parent_model, root_node = None, uuid = None, feature_name = None, extra_metadata = None):
      """Initialises a geobody feature object."""

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = feature_name,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, self.__class__):
         return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid):
         return True
      if check_extra_metadata and not equivalent_extra_metadata(self, other):
         return False
      return self.feature_name == other.feature_name

   def create_xml(self, add_as_part = True, originator = None, reuse = True):
      """Creates a geobody feature xml node from this geobody feature object."""
      if reuse and self.try_reuse():
         return self.root  # check for reusable (equivalent) object
      return super().create_xml(add_as_part = add_as_part, originator = originator)


class BoundaryFeature(BaseResqpy):
   """Class for RESQML Boudary Feature organizational objects."""

   resqml_type = "BoundaryFeature"
   feature_name = _alias_for_attribute("title")

   def __init__(self, parent_model, root_node = None, uuid = None, feature_name = None, extra_metadata = None):
      """Initialises a boundary feature organisational object."""

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = feature_name,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, BoundaryFeature):
         return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid):
         return True
      if check_extra_metadata and not equivalent_extra_metadata(self, other):
         return False
      return self.feature_name == other.feature_name

   def create_xml(self, add_as_part = True, originator = None, reuse = True):
      """Creates a geobody feature xml node from this geobody feature object."""
      if reuse and self.try_reuse():
         return self.root  # check for reusable (equivalent) object
      return super().create_xml(add_as_part = add_as_part, originator = originator)


class FrontierFeature(BaseResqpy):
   """Class for RESQML Frontier Feature organizational objects."""

   resqml_type = "FrontierFeature"
   feature_name = _alias_for_attribute("title")

   def __init__(self, parent_model, root_node = None, uuid = None, feature_name = None, extra_metadata = None):
      """Initialises a frontier feature organisational object."""

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = feature_name,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, FrontierFeature):
         return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid):
         return True
      if check_extra_metadata and not equivalent_extra_metadata(self, other):
         return False
      return self.feature_name == other.feature_name

   def create_xml(self, add_as_part = True, originator = None, reuse = True):
      """Creates a frontier feature organisational xml node from this frontier feature object."""
      if reuse and self.try_reuse():
         return self.root  # check for reusable (equivalent) object
      return super().create_xml(add_as_part = add_as_part, originator = originator)


class GeologicUnitFeature(BaseResqpy):
   """Class for RESQML Geologic Unit Feature organizational objects."""

   resqml_type = "GeologicUnitFeature"
   feature_name = _alias_for_attribute("title")

   def __init__(self, parent_model, root_node = None, uuid = None, feature_name = None, extra_metadata = None):
      """Initialises a geologic unit feature organisational object."""

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = feature_name,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, GeologicUnitFeature):
         return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid):
         return True
      if check_extra_metadata and not equivalent_extra_metadata(self, other):
         return False
      return self.feature_name == other.feature_name

   def create_xml(self, add_as_part = True, originator = None, reuse = True):
      """Creates a geologic unit feature organisational xml node from this geologic unit feature object."""
      if reuse and self.try_reuse():
         return self.root  # check for reusable (equivalent) object
      return super().create_xml(add_as_part = add_as_part, originator = originator)


class FluidBoundaryFeature(BaseResqpy):
   """Class for RESQML Fluid Boundary Feature (contact) organizational objects."""

   resqml_type = "FluidBoundaryFeature"
   feature_name = _alias_for_attribute("title")
   valid_kinds = ('free water contact', 'gas oil contact', 'gas water contact', 'seal', 'water oil contact')

   def __init__(self,
                parent_model,
                root_node = None,
                uuid = None,
                kind = None,
                feature_name = None,
                extra_metadata = None):
      """Initialises a fluid boundary feature (contact) organisational object."""

      self.kind = kind
      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = feature_name,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, FluidBoundaryFeature):
         return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid):
         return True
      if check_extra_metadata and not equivalent_extra_metadata(self, other):
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


class RockFluidUnitFeature(BaseResqpy):
   """Class for RESQML Rock Fluid Unit Feature organizational objects."""

   resqml_type = "RockFluidUnitFeature"
   feature_name = _alias_for_attribute("title")
   valid_phases = ('aquifer', 'gas cap', 'oil column', 'seal')

   def __init__(self,
                parent_model,
                root_node = None,
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

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = feature_name,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

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
      if check_extra_metadata and not equivalent_extra_metadata(self, other):
         return False
      return True

   def _load_from_xml(self):

      self.phase = rqet.find_tag_text(self.root, 'Phase')

      feature_ref_node = rqet.find_tag(self.root, 'FluidBoundaryTop')
      assert feature_ref_node is not None
      feature_root = self.model.referenced_node(feature_ref_node)
      feature_uuid = rqet.uuid_for_part_root(feature_root)
      assert feature_uuid is not None, 'rock fluid top boundary feature missing from model'
      self.top_boundary_feature = BoundaryFeature(self.model, uuid = feature_uuid)

      feature_ref_node = rqet.find_tag(self.root, 'FluidBoundaryBottom')
      assert feature_ref_node is not None
      feature_root = self.model.referenced_node(feature_ref_node)
      feature_uuid = rqet.uuid_for_part_root(feature_root)
      assert feature_uuid is not None, 'rock fluid bottom boundary feature missing from model'
      self.base_boundary_feature = BoundaryFeature(self.model, uuid = feature_uuid)

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


class TectonicBoundaryFeature(BaseResqpy):
   """Class for RESQML Tectonic Boundary Feature (fault) organizational objects."""

   resqml_type = "TectonicBoundaryFeature"
   feature_name = _alias_for_attribute("title")
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


class GeneticBoundaryFeature(BaseResqpy):
   """Class for RESQML Genetic Boundary Feature (horizon) organizational objects."""

   resqml_type = "GeneticBoundaryFeature"
   feature_name = _alias_for_attribute("title")
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


class WellboreFeature(BaseResqpy):
   """Class for RESQML Wellbore Feature organizational objects."""

   # note: optional WITSML link not supported

   resqml_type = "WellboreFeature"
   feature_name = _alias_for_attribute("title")

   def __init__(self, parent_model, root_node = None, uuid = None, feature_name = None, extra_metadata = None):
      """Initialises a wellbore feature organisational object."""
      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = feature_name,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""
      if other is None or not isinstance(other, WellboreFeature):
         return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid):
         return True
      if check_extra_metadata and not equivalent_extra_metadata(self, other):
         return False
      return self.feature_name == other.feature_name

   def create_xml(self, add_as_part = True, originator = None, reuse = True):
      """Creates a wellbore feature organisational xml node from this wellbore feature object."""
      if reuse and self.try_reuse():
         return self.root  # check for reusable (equivalent) object
      return super().create_xml(add_as_part = add_as_part, originator = originator)


class FaultInterpretation(BaseResqpy):
   """Class for RESQML Fault Interpretation organizational objects.

   RESQML documentation:

      A type of boundary feature, this class contains the data describing an opinion
      about the characterization of the fault, which includes the attributes listed below.

   """

   resqml_type = "FaultInterpretation"
   valid_domains = ('depth', 'time', 'mixed')

   # note: many of the attributes could be deduced from geometry

   def __init__(self,
                parent_model,
                root_node = None,
                uuid = None,
                title = None,
                tectonic_boundary_feature = None,
                domain = 'depth',
                is_normal = None,
                is_listric = None,
                maximum_throw = None,
                mean_azimuth = None,
                mean_dip = None,
                extra_metadata = None):
      """Initialises a Fault interpretation organisational object."""

      # note: will create a paired TectonicBoundaryFeature object when loading from xml
      # if not extracting from xml,:
      # tectonic_boundary_feature is required and must be a TectonicBoundaryFeature object
      # domain is required and must be one of 'depth', 'time' or 'mixed'
      # is_listric is required if the fault is not normal (and must be None if normal)
      # max throw, azimuth & dip are all optional
      # the throw interpretation list is not supported for direct initialisation

      self.tectonic_boundary_feature = tectonic_boundary_feature  # InterpretedFeature RESQML field, when not loading from xml
      self.feature_root = None if self.tectonic_boundary_feature is None else self.tectonic_boundary_feature.root
      if (not title) and self.tectonic_boundary_feature is not None:
         title = self.tectonic_boundary_feature.feature_name
      self.main_has_occurred_during = (None, None)
      self.is_normal = is_normal  # extra field, not explicitly in RESQML
      self.domain = domain
      # RESQML xml business rule: IsListric must be present if the fault is normal; must not be present if the fault is not normal
      self.is_listric = is_listric
      self.maximum_throw = maximum_throw
      self.mean_azimuth = mean_azimuth
      self.mean_dip = mean_dip
      self.throw_interpretation_list = None  # list of (list of throw kind, (base chrono uuid, top chrono uuid)))

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

   @property
   def feature_uuid(self):
      # TODO: rewrite using uuid as primary key
      return rqet.uuid_for_part_root(self.feature_root)

   def _load_from_xml(self):
      root_node = self.root
      self.domain = rqet.find_tag_text(root_node, 'Domain')
      interp_feature_ref_node = rqet.find_tag(root_node, 'InterpretedFeature')
      assert interp_feature_ref_node is not None
      self.feature_root = self.model.referenced_node(interp_feature_ref_node)
      if self.feature_root is not None:
         self.tectonic_boundary_feature = TectonicBoundaryFeature(self.model,
                                                                  uuid = self.feature_root.attrib['uuid'],
                                                                  feature_name = self.model.title_for_root(
                                                                     self.feature_root))
         self.main_has_occurred_during = extract_has_occurred_during(root_node)
         self.is_listric = rqet.find_tag_bool(root_node, 'IsListric')
         self.is_normal = (self.is_listric is None)
         self.maximum_throw = rqet.find_tag_float(root_node, 'MaximumThrow')
         # todo: check that type="eml:LengthMeasure" is simple float
         self.mean_azimuth = rqet.find_tag_float(root_node, 'MeanAzimuth')
         self.mean_dip = rqet.find_tag_float(root_node, 'MeanDip')
         throw_interpretation_nodes = rqet.list_of_tag(root_node, 'ThrowInterpretation')
         if throw_interpretation_nodes is not None and len(throw_interpretation_nodes):
            self.throw_interpretation_list = []
            for ti_node in throw_interpretation_nodes:
               hod_pair = extract_has_occurred_during(ti_node)
               throw_kind_list = rqet.list_of_tag(ti_node, 'Throw')
               for tk_node in throw_kind_list:
                  self.throw_interpretation_list.append((tk_node.text, hod_pair))

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this interpretation is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, FaultInterpretation):
         return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid):
         return True
      if self.tectonic_boundary_feature is not None:
         if not self.tectonic_boundary_feature.is_equivalent(other.tectonic_boundary_feature):
            return False
      elif other.tectonic_boundary_feature is not None:
         return False
      if self.root is not None and other.root is not None:
         if rqet.citation_title_for_node(self.root) != rqet.citation_title_for_node(other.root):
            return False
      elif self.root is not None or other.root is not None:
         return False
      if (not equivalent_chrono_pairs(self.main_has_occurred_during, other.main_has_occurred_during) or
          self.is_normal != other.is_normal or self.domain != other.domain or self.is_listric != other.is_listric):
         return False
      if ((self.maximum_throw is None) != (other.maximum_throw is None) or (self.mean_azimuth is None) !=
          (other.mean_azimuth is None) or (self.mean_dip is None) != (other.mean_dip is None)):
         return False
      if self.maximum_throw is not None and not maths.isclose(self.maximum_throw, other.maximum_throw, rel_tol = 1e-3):
         return False
      if self.mean_azimuth is not None and not maths.isclose(self.mean_azimuth, other.mean_azimuth, abs_tol = 0.5):
         return False
      if self.mean_dip is not None and not maths.isclose(self.mean_dip, other.mean_dip, abs_tol = 0.5):
         return False
      if check_extra_metadata and not equivalent_extra_metadata(self, other):
         return False
      if not self.throw_interpretation_list and not other.throw_interpretation_list:
         return True
      if not self.throw_interpretation_list or not other.throw_interpretation_list:
         return False
      if len(self.throw_interpretation_list) != len(other.throw_interpretation_list):
         return False
      for this_ti, other_ti in zip(self.throw_interpretation_list, other.throw_interpretation_list):
         if this_ti[0] != other_ti[0]:
            return False  # throw kind
         if not equivalent_chrono_pairs(this_ti[1], other_ti[1]):
            return False
      return True

   def create_xml(self,
                  tectonic_boundary_feature_root = None,
                  add_as_part = True,
                  add_relationships = True,
                  originator = None,
                  title_suffix = None,
                  reuse = True):
      """Creates a fault interpretation organisational xml node from a fault interpretation object."""

      # note: related tectonic boundary feature node should be created first and referenced here

      assert self.is_normal == (self.is_listric is None)
      if not self.title:
         if tectonic_boundary_feature_root is not None:
            title = rqet.find_nested_tags_text(tectonic_boundary_feature_root, ['Citation', 'Title'])
         else:
            title = 'fault interpretation'
         if title_suffix:
            title += ' ' + title_suffix

      if reuse and self.try_reuse():
         return self.root

      fi = super().create_xml(add_as_part = False, originator = originator)

      if self.tectonic_boundary_feature is not None:
         tbf_root = self.tectonic_boundary_feature.root
         if tbf_root is not None:
            if tectonic_boundary_feature_root is None:
               tectonic_boundary_feature_root = tbf_root
            else:
               assert tbf_root is tectonic_boundary_feature_root, 'tectonic boundary feature mismatch'
      assert tectonic_boundary_feature_root is not None

      assert self.domain in self.valid_domains, 'illegal domain value for fault interpretation'
      dom_node = rqet.SubElement(fi, ns['resqml2'] + 'Domain')
      dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
      dom_node.text = self.domain

      self.model.create_ref_node('InterpretedFeature',
                                 self.model.title_for_root(tectonic_boundary_feature_root),
                                 tectonic_boundary_feature_root.attrib['uuid'],
                                 content_type = 'obj_TectonicBoundaryFeature',
                                 root = fi)

      create_xml_has_occurred_during(self.model, fi, self.main_has_occurred_during)

      if self.is_listric is not None:
         listric = rqet.SubElement(fi, ns['resqml2'] + 'IsListric')
         listric.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
         listric.text = str(self.is_listric).lower()

      # todo: check eml:LengthMeasure and PlaneAngleMeasure structures: uom?
      if self.maximum_throw is not None:
         max_throw = rqet.SubElement(fi, ns['resqml2'] + 'MaximumThrow')
         max_throw.set(ns['xsi'] + 'type', ns['eml'] + 'LengthMeasure')
         max_throw.text = str(self.maximum_throw)

      if self.mean_azimuth is not None:
         azimuth = rqet.SubElement(fi, ns['resqml2'] + 'MeanAzimuth')
         azimuth.set(ns['xsi'] + 'type', ns['eml'] + 'PlaneAngleMeasure')
         azimuth.text = str(self.mean_azimuth)

      if self.mean_dip is not None:
         dip = rqet.SubElement(fi, ns['resqml2'] + 'MeanDip')
         dip.set(ns['xsi'] + 'type', ns['eml'] + 'PlaneAngleMeasure')
         dip.text = str(self.mean_azimuth)

      if self.throw_interpretation_list is not None and len(self.throw_interpretation_list):
         previous_has_occurred_during = ('never', 'never')
         ti_node = None
         for (throw_kind, (base_chrono_uuid, top_chrono_uuid)) in self.throw_interpretation_list:
            if (str(base_chrono_uuid), str(top_chrono_uuid)) != previous_has_occurred_during:
               previous_has_occurred_during = (str(base_chrono_uuid), str(top_chrono_uuid))
               ti_node = rqet.SubElement(fi, 'ThrowInterpretation')
               ti_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'FaultThrow')
               ti_node.text = rqet.null_xml_text
               create_xml_has_occurred_during(self.model, ti_node, (base_chrono_uuid, top_chrono_uuid))
            tk_node = rqet.SubElement(ti_node, 'Throw')
            tk_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ThrowKind')
            tk_node.text = throw_kind

      if add_as_part:
         self.model.add_part('obj_FaultInterpretation', self.uuid, fi)
         if add_relationships:
            self.model.create_reciprocal_relationship(fi, 'destinationObject', tectonic_boundary_feature_root,
                                                      'sourceObject')

      return fi


class EarthModelInterpretation(BaseResqpy):
   """Class for RESQML Earth Model Interpretation organizational objects."""

   resqml_type = 'EarthModelInterpretation'
   valid_domains = ('depth', 'time', 'mixed')

   def __init__(self,
                parent_model,
                root_node = None,
                uuid = None,
                title = None,
                organization_feature = None,
                domain = 'depth',
                extra_metadata = None):
      """Initialises an earth model interpretation organisational object."""
      self.domain = domain
      self.organization_feature = organization_feature  # InterpretedFeature RESQML field
      self.feature_root = None if self.organization_feature is None else self.organization_feature.root
      self.has_occurred_during = (None, None)
      if (not title) and organization_feature is not None:
         title = organization_feature.feature_name
      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

   def _load_from_xml(self):
      self.domain = rqet.find_tag_text(self.root, 'Domain')
      interp_feature_ref_node = rqet.find_tag(self.root, 'InterpretedFeature')
      assert interp_feature_ref_node is not None
      self.feature_root = self.model.referenced_node(interp_feature_ref_node)
      if self.feature_root is not None:
         self.organization_feature = OrganizationFeature(self.model,
                                                         uuid = self.feature_root.attrib['uuid'],
                                                         feature_name = self.model.title_for_root(self.feature_root))
      self.has_occurred_during = extract_has_occurred_during(self.root)

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this interpretation is essentially the same as the other; otherwise False."""
      if other is None or not isinstance(other, EarthModelInterpretation):
         return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid):
         return True
      if self.organization_feature is not None:
         if not self.organization_feature.is_equivalent(other.organization_feature):
            return False
      elif other.organization_feature is not None:
         return False
      if self.root is not None and other.root is not None:
         if rqet.citation_title_for_node(self.root) != rqet.citation_title_for_node(other.root):
            return False
      elif self.root is not None or other.root is not None:
         return False
      if check_extra_metadata and not equivalent_extra_metadata(self, other):
         return False
      return self.domain == other.domain and equivalent_chrono_pairs(self.has_occurred_during,
                                                                     other.has_occurred_during)

   def create_xml(self,
                  organization_feature_root = None,
                  add_as_part = True,
                  add_relationships = True,
                  originator = None,
                  title_suffix = None,
                  reuse = True):
      """Creates an earth model interpretation organisational xml node from an earth model interpretation object."""

      # note: related organization feature node should be created first and referenced here

      if not self.title:
         self.title = self.organization_feature.feature_name
      if title_suffix:
         self.title += ' ' + title_suffix

      if reuse and self.try_reuse():
         return self.root
      emi = super().create_xml(add_as_part = False, originator = originator)

      if self.organization_feature is not None:
         of_root = self.organization_feature.root
         if of_root is not None:
            if organization_feature_root is None:
               organization_feature_root = of_root
            else:
               assert of_root is organization_feature_root, 'organization feature mismatch'

      assert organization_feature_root is not None, 'interpreted feature not established for model interpretation'

      assert self.domain in self.valid_domains, 'illegal domain value for earth model interpretation'
      dom_node = rqet.SubElement(emi, ns['resqml2'] + 'Domain')
      dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
      dom_node.text = self.domain

      self.model.create_ref_node('InterpretedFeature',
                                 self.model.title_for_root(organization_feature_root),
                                 organization_feature_root.attrib['uuid'],
                                 content_type = 'obj_OrganizationFeature',
                                 root = emi)

      create_xml_has_occurred_during(self.model, emi, self.has_occurred_during)

      if add_as_part:
         self.model.add_part('obj_EarthModelInterpretation', self.uuid, emi)
         if add_relationships:
            self.model.create_reciprocal_relationship(emi, 'destinationObject', organization_feature_root,
                                                      'sourceObject')

      return emi


class HorizonInterpretation(BaseResqpy):
   """Class for RESQML Horizon Interpretation organizational objects."""

   resqml_type = 'HorizonInterpretation'
   valid_domains = ('depth', 'time', 'mixed')
   valid_sequence_stratigraphy_surfaces = ('flooding', 'ravinement', 'maximum flooding', 'transgressive')
   valid_boundary_relations = ('conformable', 'unconformable below and above', 'unconformable above',
                               'unconformable below')

   def __init__(self,
                parent_model,
                root_node = None,
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
      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

   def _load_from_xml(self):
      self.domain = rqet.find_tag_text(self.root, 'Domain')
      interp_feature_ref_node = rqet.find_tag(self.root, 'InterpretedFeature')
      assert interp_feature_ref_node is not None
      self.feature_root = self.model.referenced_node(interp_feature_ref_node)
      if self.feature_root is not None:
         self.genetic_boundary_feature = GeneticBoundaryFeature(self.model,
                                                                kind = 'horizon',
                                                                uuid = self.feature_root.attrib['uuid'],
                                                                feature_name = self.model.title_for_root(
                                                                   self.feature_root))
      self.has_occurred_during = extract_has_occurred_during(self.root)
      br_node_list = rqet.list_of_tag(self.root, 'BoundaryRelation')
      if br_node_list is not None and len(br_node_list) > 0:
         self.boundary_relation_list = []
         for br_node in br_node_list:
            self.boundary_relation_list.append(br_node.text)
      self.sequence_stratigraphy_surface = rqet.find_tag_text(self.root, 'SequenceStratigraphySurface')

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this interpretation is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, HorizonInterpretation):
         return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid):
         return True
      if self.genetic_boundary_feature is not None:
         if not self.genetic_boundary_feature.is_equivalent(other.genetic_boundary_feature):
            return False
      elif other.genetic_boundary_feature is not None:
         return False
      if self.root is not None and other.root is not None:
         if rqet.citation_title_for_node(self.root) != rqet.citation_title_for_node(other.root):
            return False
      elif self.root is not None or other.root is not None:
         return False
      if (self.domain != other.domain or
          not equivalent_chrono_pairs(self.has_occurred_during, other.has_occurred_during) or
          self.sequence_stratigraphy_surface != other.sequence_stratigraphy_surface):
         return False
      if check_extra_metadata and not equivalent_extra_metadata(self, other):
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

      create_xml_has_occurred_during(self.model, hi, self.has_occurred_during)

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


class GeobodyBoundaryInterpretation(BaseResqpy):
   """Class for RESQML Geobody Boudary Interpretation organizational objects."""

   resqml_type = 'GeobodyBoundaryInterpretation'
   valid_domains = ('depth', 'time', 'mixed')
   valid_boundary_relations = ('conformable', 'unconformable below and above', 'unconformable above',
                               'unconformable below')

   def __init__(self,
                parent_model,
                root_node = None,
                uuid = None,
                title = None,
                genetic_boundary_feature = None,
                domain = 'depth',
                boundary_relation_list = None,
                extra_metadata = None):

      self.domain = domain
      self.boundary_relation_list = None if not boundary_relation_list else boundary_relation_list.copy()
      self.genetic_boundary_feature = genetic_boundary_feature  # InterpretedFeature RESQML field, when not loading from xml
      self.feature_root = None if self.genetic_boundary_feature is None else self.genetic_boundary_feature.root
      if (not title) and self.genetic_boundary_feature is not None:
         title = self.genetic_boundary_feature.feature_name
      self.has_occurred_during = (None, None)
      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

   def _load_from_xml(self):
      self.domain = rqet.find_tag_text(self.root, 'Domain')
      interp_feature_ref_node = rqet.find_tag(self.root, 'InterpretedFeature')
      assert interp_feature_ref_node is not None
      self.feature_root = self.model.referenced_node(interp_feature_ref_node)
      if self.feature_root is not None:
         self.genetic_boundary_feature = GeneticBoundaryFeature(self.model,
                                                                kind = 'geobody boundary',
                                                                uuid = self.feature_root.attrib['uuid'],
                                                                feature_name = self.model.title_for_root(
                                                                   self.feature_root))
      self.has_occurred_during = extract_has_occurred_during(self.root)
      br_node_list = rqet.list_of_tag(self.root, 'BoundaryRelation')
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
         if not self.genetic_boundary_feature.is_equivalent(other.genetic_boundary_feature):
            return False
      elif other.genetic_boundary_feature is not None:
         return False
      if self.root is not None and other.root is not None:
         if rqet.citation_title_for_node(self.root) != rqet.citation_title_for_node(other.root):
            return False
      elif self.root is not None or other.root is not None:
         return False
      if (self.domain != other.domain or
          not equivalent_chrono_pairs(self.has_occurred_during, other.has_occurred_during)):
         return False
      if check_extra_metadata and not equivalent_extra_metadata(self, other):
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
      """Creates a geobody boundary interpretation organisational xml node from a geobody boundary interpretation object."""

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
         self.genetic_boundary_feature = GeneticBoundaryFeature(self.model,
                                                                uuid = genetic_boundary_feature_root.attrib['uuid'])
      self.feature_root = genetic_boundary_feature_root

      assert self.domain in self.valid_domains, 'illegal domain value for geobody boundary interpretation'
      dom_node = rqet.SubElement(gbi, ns['resqml2'] + 'Domain')
      dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
      dom_node.text = self.domain

      create_xml_has_occurred_during(self.model, gbi, self.has_occurred_during)

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
   valid_geobody_shapes = ('dyke', 'silt', 'sill', 'dome', 'sheeth', 'sheet', 'diapir', 'batholith', 'channel', 'delta',
                           'dune', 'fan', 'reef', 'wedge')

   def __init__(self,
                parent_model,
                root_node = None,
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
      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

   def _load_from_xml(self):
      interp_feature_ref_node = rqet.find_tag(self.root, 'InterpretedFeature')
      assert interp_feature_ref_node is not None
      self.feature_root = self.model.referenced_node(interp_feature_ref_node)
      if self.feature_root is not None:
         self.geobody_feature = GeobodyFeature(self.model,
                                               uuid = self.feature_root.attrib['uuid'],
                                               feature_name = self.model.title_for_root(self.feature_root))
      self.has_occurred_during = extract_has_occurred_during(self.root)
      self.composition = rqet.find_tag_text(self.root, 'GeologicUnitComposition')
      self.implacement = rqet.find_tag_text(self.root, 'GeologicUnitMaterialImplacement')
      self.geobody_shape = rqet.find_tag_text(self.root, 'Geobody3dShape')

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this interpretation is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, GeobodyInterpretation):
         return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid):
         return True
      if self.geobody_feature is not None:
         if not self.geobody_feature.is_equivalent(other.geobody_feature):
            return False
      elif other.geobody_feature is not None:
         return False
      if self.root is not None and other.root is not None:
         if rqet.citation_title_for_node(self.root) != rqet.citation_title_for_node(other.root):
            return False
      elif self.root is not None or other.root is not None:
         return False
      if check_extra_metadata and not equivalent_extra_metadata(self, other):
         return False
      return (self.domain == other.domain and
              equivalent_chrono_pairs(self.has_occurred_during, other.has_occurred_during) and
              self.composition == other.composition and self.implacement == other.implacement and
              self.geobody_shape == other.geobody_shape)

   def create_xml(self,
                  geobody_feature_root = None,
                  add_as_part = True,
                  add_relationships = True,
                  originator = None,
                  title_suffix = None,
                  reuse = True):

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
         self.geobody_feature = GeobodyFeature(self.model, uuid = geobody_feature_root.attrib['uuid'])
      self.feature_root = geobody_feature_root

      assert self.domain in self.valid_domains, 'illegal domain value for geobody interpretation'
      dom_node = rqet.SubElement(gi, ns['resqml2'] + 'Domain')
      dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
      dom_node.text = self.domain

      create_xml_has_occurred_during(self.model, gi, self.has_occurred_during)

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
                root_node = None,
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
      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

   def _load_from_xml(self):
      root_node = self.root
      self.is_drilled = rqet.find_tag_bool(root_node, 'IsDrilled')
      self.domain = rqet.find_tag_text(root_node, 'Domain')
      interp_feature_ref_node = rqet.find_tag(root_node, 'InterpretedFeature')
      if interp_feature_ref_node is not None:
         self.feature_root = self.model.referenced_node(interp_feature_ref_node)
         if self.feature_root is not None:
            self.wellbore_feature = WellboreFeature(self.model,
                                                    uuid = self.feature_root.attrib['uuid'],
                                                    feature_name = self.model.title_for_root(self.feature_root))

   def iter_trajectories(self):
      """ Iterable of associated trajectories """

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
         if not self.wellbore_feature.is_equivalent(other.wellbore_feature):
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
      if check_extra_metadata and not equivalent_extra_metadata(self, other):
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
            self.model.create_reciprocal_relationship(wi, 'destinationObject', wellbore_feature_root, 'sourceObject')

      return wi
