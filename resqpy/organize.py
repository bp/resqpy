"""organize.py: RESQML Feature and Interpretation classes."""

version = '8th May 2021'

# For now, fault features and interpretations, plus stubs for horizons

import logging
log = logging.getLogger(__name__)
log.debug('organize.py version ' + version)

# import xml.etree.ElementTree as et
# from lxml import etree as et

import math as maths

import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu
from resqpy.olio.xml_namespaces import curly_namespace as ns


def extract_has_occurred_during(parent_node, tag = 'HasOccuredDuring'):  # RESQML Occured (stet)
   """Extracts UUIDs of chrono bottom and top from xml for has occurred during sub-node, or (None, None)."""
   hod_node = rqet.find_tag(parent_node, tag)
   if hod_node is None:
      return (None, None)
   else:
      return (rqet.find_nested_tags_text(hod_node, ['ChronoBottom', 'UUID']),
              rqet.find_nested_tags_text(hod_node, ['ChronoTop', 'UUID']))


def equivalent_chrono_pairs(pair_a, pair_b, model = None):
   if pair_a == pair_b: return True
   if pair_a is None or pair_b is None: return False
   if pair_a == (None, None) or pair_b == (None, None): return False
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
      a_em = rqet.load_metadata_from_xml(a.root_node)
      a_has = a_em is not None and len(a_em) > 0
   if b_has:
      b_em = b.extra_metadata
      b_has = len(b_em) > 0
   else:
      b_em = rqet.load_metadata_from_xml(b.root_node)
      b_has = b_em is not None and len(b_em) > 0
   if a_has != b_has: return False
   return a_em == b_em


def create_xml_has_occurred_during(model, parent_node, hod_pair, tag = 'HasOccuredDuring'):
   if hod_pair is None: return
   base_chrono_uuid, top_chrono_uuid = hod_pair
   if base_chrono_uuid is None or top_chrono_uuid is None: return
   hod_node = rqet.SubElement(parent_node, tag)
   hod_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'TimeInterval')
   hod_node.text = rqet.null_xml_text
   chrono_base_root = model.root_for_uuid(base_chrono_uuid)
   chrono_top_root = model.root_for_uuid(top_chrono_uuid)
   model.create_ref_node('ChronoBottom', model.title_for_root(chrono_base_root), base_chrono_uuid, root = hod_node)
   model.create_ref_node('ChronoTop', model.title_for_root(chrono_top_root), top_chrono_uuid, root = hod_node)



class OrganizationFeature():
   """Class for generic RESQML Organization Feature objects."""

   def __init__(self, parent_model, root_node = None, extract_from_xml = True, feature_name = None, organization_kind = None):
      """Initialises an organization feature object."""

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.feature_name = None
      self.organization_kind = None

      if extract_from_xml and self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         self.feature_name = rqet.find_nested_tags_text(self.root_node, ['Citation', 'Title'])
         self.organization_kind = rqet.find_tag_text(self.root_node, 'OrganizationKind')
      else:
         self.uuid = bu.new_uuid()
         self.feature_name = feature_name
         assert organization_kind in ['earth model', 'fluid', 'stratigraphic', 'structural']
         self.organization_kind = organization_kind


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, OrganizationFeature): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if self.feature_name != other.feature_name or self.organization_kind != other.organization_kind: return False
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      return True


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, add_as_part = True, originator = None):
      """Creates an organization feature xml node from this organization feature object."""

      ofn = self.model.new_obj_node('OrganizationFeature')

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(ofn.attrib['uuid'])
      else:
         ofn.attrib['uuid'] = str(self.uuid)

      self.model.create_citation(root = ofn, title = self.feature_name, originator = originator)

      kind_node = rqet.SubElement(ofn, ns['resqml2'] + 'OrganizationKind')
      kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'OrganizationKind')
      kind_node.text = self.organization_kind

      if add_as_part:
         self.model.add_part('obj_OrganizationFeature', self.uuid, ofn)

      self.root_node = ofn

      return ofn



class GeobodyFeature():
   """Class for RESQML Geobody Feature objects (note: definition may be incomplete in RESQML 2.0.1)."""

   def __init__(self, parent_model, root_node = None, extract_from_xml = True, feature_name = None):
      """Initialises a geobody feature object."""

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.feature_name = None

      if extract_from_xml and self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         self.feature_name = rqet.find_nested_tags_text(self.root_node, ['Citation', 'Title'])
      else:
         self.uuid = bu.new_uuid()
         self.feature_name = feature_name


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, GeobodyFeature): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      return self.feature_name == other.feature_name


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, add_as_part = True, originator = None):
      """Creates a geobody feature xml node from this geobody feature object."""

      gfn = self.model.new_obj_node('GeobodyFeature')

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(gfn.attrib['uuid'])
      else:
         gfn.attrib['uuid'] = str(self.uuid)

      self.model.create_citation(root = gfn, title = self.feature_name, originator = originator)

      if add_as_part:
         self.model.add_part('obj_GeobodyFeature', self.uuid, gfn)

      self.root_node = gfn

      return gfn



class BoundaryFeature():
   """Class for RESQML Boudary Feature organizational objects."""

   def __init__(self, parent_model, root_node = None, feature_name = None):
      """Initialises a boundary feature organisational object."""

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.feature_name = None

      if self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         self.feature_name = rqet.find_nested_tags_text(self.root_node, ['Citation', 'Title'])
      else:
         self.uuid = bu.new_uuid()
         self.feature_name = feature_name


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, BoundaryFeature): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      return self.feature_name == other.feature_name


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, add_as_part = True, originator = None):
      """Creates a boundary feature organisational xml node from this boundary feature object."""

      bf = self.model.new_obj_node('BoundaryFeature')

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(bf.attrib['uuid'])
      else:
         bf.attrib['uuid'] = str(self.uuid)

      self.model.create_citation(root = bf, title = self.feature_name, originator = originator)

      if add_as_part:
         self.model.add_part('obj_BoundaryFeature', self.uuid, bf)

      self.root_node = bf

      return bf



class FrontierFeature():
   """Class for RESQML Frontier Feature organizational objects."""

   def __init__(self, parent_model, root_node = None, feature_name = None):
      """Initialises a frontier feature organisational object."""

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.feature_name = None

      if self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         self.feature_name = rqet.find_nested_tags_text(self.root_node, ['Citation', 'Title'])
      else:
         self.uuid = bu.new_uuid()
         self.feature_name = feature_name


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, FrontierFeature): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      return self.feature_name == other.feature_name


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, add_as_part = True, originator = None):
      """Creates a frontier feature organisational xml node from this frontier feature object."""

      ff = self.model.new_obj_node('FrontierFeature')

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(ff.attrib['uuid'])
      else:
         ff.attrib['uuid'] = str(self.uuid)

      self.model.create_citation(root = ff, title = self.feature_name, originator = originator)

      if add_as_part:
         self.model.add_part('obj_FrontierFeature', self.uuid, ff)

      self.root_node = ff

      return ff



class GeologicUnitFeature():
   """Class for RESQML Geologic Unit Feature organizational objects."""

   def __init__(self, parent_model, root_node = None, feature_name = None):
      """Initialises a geologic unit feature organisational object."""

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.feature_name = None

      if self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         self.feature_name = rqet.find_nested_tags_text(self.root_node, ['Citation', 'Title'])
      else:
         self.uuid = bu.new_uuid()
         self.feature_name = feature_name


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, GeologicUnitFeature): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      return self.feature_name == other.feature_name


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, add_as_part = True, originator = None):
      """Creates a geologic unit feature organisational xml node from this geologic unit feature object."""

      guf = self.model.new_obj_node('GeologicUnitFeature')

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(guf.attrib['uuid'])
      else:
         guf.attrib['uuid'] = str(self.uuid)

      self.model.create_citation(root = guf, title = self.feature_name, originator = originator)

      if add_as_part:
         self.model.add_part('obj_GeologicUnitFeature', self.uuid, guf)

      self.root_node = guf

      return guf



class FluidBoundaryFeature():
   """Class for RESQML Fluid Boundary Feature (contact) organizational objects."""

   def __init__(self, parent_model, root_node = None, kind = None, feature_name = None):
      """Initialises a fluid boundary feature (contact) organisational object."""

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.kind = None
      self.feature_name = None

      if self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         xml_kind = rqet.find_tag_text(self.root_node, 'TectonicBoundaryKind')
         if xml_kind and self.kind:
            assert xml_kind.lower() == self.kind.lower(), 'Tectonic boundary kind mismatch'
         else:
            self.kind = xml_kind
         self.feature_name = rqet.find_nested_tags_text(self.root_node, ['Citation', 'Title'])
      elif kind and feature_name:
         self.uuid = bu.new_uuid()
         self.kind = kind
         self.feature_name = feature_name

      if self.uuid is None: self.uuid = bu.new_uuid()


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, FluidBoundaryFeature): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      return self.feature_name == other.feature_name and self.kind == other.kind


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, add_as_part = True, originator = None):
      """Creates a fluid boundary feature organisational xml node from this fluid boundary feature object."""

      fbf = self.model.new_obj_node('FluidBoundaryFeature')

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(fbf.attrib['uuid'])
      else:
         fbf.attrib['uuid'] = str(self.uuid)

      self.model.create_citation(root = fbf, title = self.feature_name, originator = originator)

      assert self.kind in ['free water contact', 'gas oil contact', 'gas water contact',
                           'seal', 'water oil contact'], 'fluid boundary feature kind not recognized'
      kind_node = rqet.SubElement(fbf, ns['resqml2'] + 'FluidContact')
      kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'FluidContact')
      kind_node.text = self.kind

      if add_as_part:
         self.model.add_part('obj_FluidBoundaryFeature', self.uuid, fbf)

      self.root_node = fbf

      return fbf



class RockFluidUnitFeature():
   """Class for RESQML Rock Fluid Unit Feature organizational objects."""

   def __init__(self, parent_model, root_node = None, phase = None, feature_name = None,
                top_boundary_feature = None, base_boundary_feature = None):
      """Initialises a rock fluid unit feature organisational object."""

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.phase = None
      self.feature_name = None
      self.top_boundary_feature = None
      self.base_boundary_feature = None

      if self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         self.phase = rqet.find_tag_text(self.root_node, 'Phase')
         self.feature_name = rqet.find_nested_tags_text(self.root_node, ['Citation', 'Title'])
         feature_ref_node = rqet.find_tag(self.root_node, 'FluidBoundaryTop')
         assert feature_ref_node is not None
         feature_root = self.model.referenced_node(feature_ref_node)
         assert feature_root is not None, 'rock fluid top boundary feature missing from model'
         self.top_boundary_feature = BoundaryFeature(self.model, root_node = feature_root)
         feature_ref_node = rqet.find_tag(self.root_node, 'FluidBoundaryBottom')
         assert feature_ref_node is not None
         feature_root = self.model.referenced_node(feature_ref_node)
         assert feature_root is not None, 'rock fluid bottom boundary feature missing from model'
         self.base_boundary_feature = BoundaryFeature(self.model, root_node = feature_root)
      elif phase and feature_name and top_boundary_feature and base_boundary_feature:
         self.uuid = bu.new_uuid()
         self.phase = phase
         self.feature_name = feature_name
         self.top_boundary_feature = top_boundary_feature
         self.base_boundary_feature = base_boundary_feature

      if self.uuid is None: self.uuid = bu.new_uuid()


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, RockFluidUnitFeature): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if self.feature_name != other.feature_name or self.phase != other.phase: return False
      if self.top_boundary_feature is not None:
          if not self.top_boundary_feature.is_equivalent(other.top_boundary_feature): return False
      elif other.top_boundary_feature is not None: return False
      if self.base_boundary_feature is not None:
          if not self.base_boundary_feature.is_equivalent(other.base_boundary_feature): return False
      elif other.base_boundary_feature is not None: return False
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      return True


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, add_as_part = True, add_relationships = True, originator = None):
      """Creates a rock fluid unit feature organisational xml node from this rock fluid unit feature object."""

      assert self.feature_name and self.phase and self.top_boundary_feature and self.base_boundary_feature

      rfuf = self.model.new_obj_node('RockFluidUnitFeature')

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(rfuf.attrib['uuid'])
      else:
         rfuf.attrib['uuid'] = str(self.uuid)

      self.model.create_citation(root = rfuf, title = self.feature_name, originator = originator)

      assert self.phase in ['aquifer', 'gas cap', 'oil column', 'seal'], 'phase not recognized'
      phase_node = rqet.SubElement(rfuf, ns['resqml2'] + 'Phase')
      phase_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Phase')
      phase_node.text = self.phase

      top_boundary_root = self.top_boundary_feature.root_node
      assert top_boundary_root is not None
      self.model.create_ref_node('FluidBoundaryTop', self.model.title_for_root(top_boundary_root),
                                 top_boundary_root.attrib['uuid'],
                                 content_type = 'obj_BoundaryFeature', root = rfuf)

      base_boundary_root = self.base_boundary_feature.root_node
      assert base_boundary_root is not None
      self.model.create_ref_node('FluidBoundaryBottom', self.model.title_for_root(base_boundary_root),
                                 base_boundary_root.attrib['uuid'],
                                 content_type = 'obj_BoundaryFeature', root = rfuf)

      if add_as_part:
         self.model.add_part('obj_RockFluidUnitFeature', self.uuid, rfuf)
         if add_relationships:
            self.model.create_reciprocal_relationship(rfuf, 'destinationObject', top_boundary_root, 'sourceObject')
            self.model.create_reciprocal_relationship(rfuf, 'destinationObject', base_boundary_root, 'sourceObject')

      self.root_node = rfuf

      return rfuf



class TectonicBoundaryFeature():
   """Class for RESQML Tectonic Boundary Feature (fault) organizational objects."""

   def __init__(self, parent_model, root_node = None, extract_from_xml = True, kind = None, feature_name = None):
      """Initialises a tectonic boundary feature (fault or fracture) organisational object."""

      if kind is None:
         assert extract_from_xml, 'kind must be specified unless extracting tectonic boundary feature from xml'
      else:
         assert kind in ['fault', 'fracture'], 'tectonic boundary feature kind not recognized'

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.kind = kind
      self.feature_name = None

      if extract_from_xml and self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         xml_kind = rqet.find_tag_text(self.root_node, 'TectonicBoundaryKind')
         if xml_kind and self.kind:
            assert xml_kind.lower() == self.kind.lower(), 'Tectonic boundary kind mismatch'
         else:
            self.kind = xml_kind
         self.feature_name = rqet.find_nested_tags_text(self.root_node, ['Citation', 'Title'])
      else:
         self.uuid = bu.new_uuid()
         if not self.kind: self.kind = 'fault'
         self.feature_name = feature_name

      if self.uuid is None: self.uuid = bu.new_uuid()


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, TectonicBoundaryFeature): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      return self.feature_name == other.feature_name and self.kind == other.kind


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, add_as_part = True, originator = None):
      """Creates a tectonic boundary feature organisational xml node from this tectonic boundary feature object."""

      tbf = self.model.new_obj_node('TectonicBoundaryFeature')

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(tbf.attrib['uuid'])
      else:
         tbf.attrib['uuid'] = str(self.uuid)

      self.model.create_citation(root = tbf, title = self.feature_name, originator = originator)

      kind_node = rqet.SubElement(tbf, ns['resqml2'] + 'TectonicBoundaryKind')
      kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'TectonicBoundaryKind')
      kind_node.text = self.kind

      if add_as_part:
         self.model.add_part('obj_TectonicBoundaryFeature', self.uuid, tbf)

      self.root_node = tbf

      return tbf



class GeneticBoundaryFeature():
   """Class for RESQML Genetic Boundary Feature (horizon) organizational objects."""

   def __init__(self, parent_model, root_node = None, extract_from_xml = True, kind = None, feature_name = None):
      """Initialises a genetic boundary feature (horizon or geobody boundary) organisational object."""

      if kind is None:
         if not extract_from_xml: kind = 'horizon'
      else:
         assert kind in ['horizon', 'geobody boundary'], 'genetic boundary feature kind not recognized'

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.kind = kind
      self.absolute_age = None   # (timestamp, year offset) pair, or None
      self.feature_name = None

      if extract_from_xml and self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         xml_kind = rqet.find_tag_text(self.root_node, 'GeneticBoundaryKind')
         if xml_kind and self.kind:
            assert xml_kind.lower() == self.kind.lower(), 'Genetic boundary kind mismatch'
         else:
            self.kind = xml_kind
         self.feature_name = rqet.find_nested_tags_text(self.root_node, ['Citation', 'Title'])
         age_node = rqet.find_tag(self.root_node, 'AbsoluteAge')
         if age_node:
            self.absolute_age = (rqet.find_tag_text(age_node, 'DateTime'),
                                 rqet.find_tag_int(age_node, 'YearOffset'))  # year offset may be None
      else:
         self.uuid = bu.new_uuid()
         self.feature_name = feature_name

      if self.uuid is None: self.uuid = bu.new_uuid()


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, GeneticBoundaryFeature): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      return self.feature_name == other.feature_name and self.kind == other.kind and self.absolute_age == other.absolute_age


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, add_as_part = True, originator = None):
      """Creates a genetic boundary feature organisational xml node from this genetic boundary feature object."""

      gbf = self.model.new_obj_node('GeneticBoundaryFeature')

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(gbf.attrib['uuid'])
      else:
         gbf.attrib['uuid'] = str(self.uuid)

      self.model.create_citation(root = gbf, title = self.feature_name, originator = originator)

      assert self.kind in ['horizon', 'geobody boundary']
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

      self.root_node = gbf

      return gbf



class WellboreFeature():
   """Class for RESQML Wellbore Feature organizational objects."""

   # note: optional WITSML link not supported

   def __init__(self, parent_model, root_node = None, extract_from_xml = True, feature_name = None):
      """Initialises a wellbore feature organisational object."""

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.feature_name = None

      if extract_from_xml and self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         self.feature_name = rqet.find_nested_tags_text(self.root_node, ['Citation', 'Title'])
      else:
         self.uuid = bu.new_uuid()
         self.feature_name = feature_name


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, WellboreFeature): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      return self.feature_name == other.feature_name


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, add_as_part = True, originator = None):
      """Creates a wellbore feature organisational xml node from this wellbore feature object."""

      wbf = self.model.new_obj_node('WellboreFeature')

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(wbf.attrib['uuid'])
      else:
         wbf.attrib['uuid'] = str(self.uuid)

      self.model.create_citation(root = wbf, title = self.feature_name, originator = originator)

      if add_as_part:
         self.model.add_part('obj_WellboreFeature', self.uuid, wbf)

      self.root_node = wbf

      return wbf



class FaultInterpretation():
   """Class for RESQML Fault Interpretation organizational objects.

   RESQML documentation:

      A type of boundary feature, this class contains the data describing an opinion
      about the characterization of the fault, which includes the attributes listed below.

   """

   # note: many of the attributes could be deduced from geometry

   def __init__(self, parent_model, root_node = None, extract_from_xml = True,
                tectonic_boundary_feature = None, domain = 'depth',
                is_normal = None, is_listric = None,
                maximum_throw = None, mean_azimuth = None,
                mean_dip = None):
      """Initialises a Fault interpretation organisational object."""

      # note: will create a paired TectonicBoundaryFeature object when loading from xml

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.tectonic_boundary_feature = None  # InterpretedFeature RESQML field, when not loading from xml
      self.feature_root = None
      self.main_has_occurred_during = (None, None)
      self.is_normal = None                  # extra field, not explicitly in RESQML
      self.domain = None
      # RESQML xml business rule: IsListric must be present if the fault is normal; must not be present if the fault is not normal
      self.is_listric = None
      self.maximum_throw = None
      self.mean_azimuth = None
      self.mean_dip = None
      self.throw_interpretation_list = None  # list of (list of throw kind, (base chrono uuid, top chrono uuid)))
      self.extra_metadata = {}

      if extract_from_xml and self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         self.domain = rqet.find_tag_text(self.root_node, 'Domain')
         interp_feature_ref_node = rqet.find_tag(self.root_node, 'InterpretedFeature')
         assert interp_feature_ref_node is not None
         self.feature_root = self.model.referenced_node(interp_feature_ref_node)
         if self.feature_root is not None:
            self.tectonic_boundary_feature = TectonicBoundaryFeature(self.model,
                                                root_node = self.feature_root,
                                                feature_name = self.model.title_for_root(self.feature_root))
         self.main_has_occurred_during = extract_has_occurred_during(self.root_node)
         self.is_listric = rqet.find_tag_bool(self.root_node, 'IsListric')
         self.is_normal = (self.is_listric is None)
         self.maximum_throw = rqet.find_tag_float(self.root_node, 'MaximumThrow')  # todo: check that type="eml:LengthMeasure" is simple float
         self.mean_azimuth = rqet.find_tag_float(self.root_node, 'MeanAzimuth')
         self.mean_dip = rqet.find_tag_float(self.root_node, 'MeanDip')
         throw_interpretation_nodes = rqet.list_of_tag(self.root_node, 'ThrowInterpretation')
         if throw_interpretation_nodes is not None and len(throw_interpretation_nodes):
            self.throw_interpretation_list = []
            for ti_node in throw_interpretation_nodes:
               hod_pair = extract_has_occurred_during(ti_node)
               throw_kind_list = rqet.list_of_tag(ti_node, 'Throw')
               for tk_node in throw_kind_list:
                  self.throw_interpretation_list.append((tk_node.text, hod_pair))
         self.extra_metadata = rqet.load_metadata_from_xml(root_node)
      else:
         assert tectonic_boundary_feature is not None
         assert domain in ['depth', 'time', 'mixed'], 'unrecognised domain value for fault interpretation'
         assert is_normal is not None and isinstance(is_normal, bool) and is_normal == (is_listric is None)
         self.uuid = bu.new_uuid()
         self.is_normal = is_normal
         self.tectonic_boundary_feature = tectonic_boundary_feature
         self.domain = domain
         self.is_listric = is_listric
         self.maximum_throw = maximum_throw
         self.mean_azimuth = mean_azimuth
         self.mean_dip = mean_dip
         self.throw_interpretation_list = None  # not curerntly supported for direct initialisation

      # if not extracting from xml,:
      # tectonic_boundary_feature is required and must be a TectonicBoundaryFeature object
      # domain is required and must be one of 'depth', 'time' or 'mixed'
      # is_listric is required if the fault is not normal (and must be None if normal)
      # max throw, azimuth & dip are all optional
      # the throw interpretation list is not supported for direct initialisation

      if self.uuid is None: self.uuid = bu.new_uuid()


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this interpretation is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, FaultInterpretation): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if self.tectonic_boundary_feature is not None:
         if not self.tectonic_boundary_feature.is_equivalent(other.tectonic_boundary_feature): return False
      elif other.tectonic_boundary_feature is not None: return False
      if self.root_node is not None and other.root_node is not None:
         if rqet.citation_title_for_node(self.root_node) !=  rqet.citation_title_for_node(other.root_node): return False
      elif self.root_node is not None or other.root_node is not None: return False
      if (not equivalent_chrono_pairs(self.main_has_occurred_during, other.main_has_occurred_during) or
          self.is_normal != other.is_normal or
          self.domain != other.domain or
          self.is_listric != other.is_listric): return False
      if ((self.maximum_throw is None) != (other.maximum_throw is None) or
          (self.mean_azimuth is None) != (other.mean_azimuth is None) or
          (self.mean_dip is None) != (other.mean_dip is None)): return False
      if self.maximum_throw is not None and not maths.isclose(self.maximum_throw, other.maximum_throw, rel_tol = 1e-3): return False
      if self.mean_azimuth is not None and not maths.isclose(self.mean_azimuth, other.mean_azimuth, abs_tol = 0.5): return False
      if self.mean_dip is not None and not maths.isclose(self.mean_dip, other.mean_dip, abs_tol = 0.5): return False
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      if not self.throw_interpretation_list and not other.throw_interpretation_list: return True
      if not self.throw_interpretation_list or not other.throw_interpretation_list: return False
      if len(self.throw_interpretation_list) != len(other.throw_interpretation_list): return False
      for this_ti, other_ti in zip(self.throw_interpretation_list, other.throw_interpretation_list):
         if this_ti[0] != other_ti[0]: return False   # throw kind
         if not equivalent_chrono_pairs(this_ti[1], other_ti[1]): return False
      return True


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, tectonic_boundary_feature_root = None,
                  add_as_part = True, add_relationships = True, originator = None,
                  title_suffix = 'fault interpretation'):
      """Creates a fault interpretation organisational xml node from a fault interpretation object."""

      # note: related tectonic boundary feature node should be created first and referenced here

      assert self.is_normal == (self.is_listric is None)

      fi = self.model.new_obj_node('FaultInterpretation')

      if self.tectonic_boundary_feature is not None:
         tbf_root = self.tectonic_boundary_feature.root_node
         if tbf_root is not None:
            if tectonic_boundary_feature_root is None:
               tectonic_boundary_feature_root = tbf_root
            else:
               assert tbf_root is tectonic_boundary_feature_root, 'tectonic boundary feature mismatch'
      assert tectonic_boundary_feature_root is not None
      title = rqet.find_nested_tags_text(tectonic_boundary_feature_root, ['Citation', 'Title'])

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(fi.attrib['uuid'])
      else:
         fi.attrib['uuid'] = str(self.uuid)

      if title_suffix: title += ' ' + title_suffix
      self.model.create_citation(root = fi, title = title, originator = originator)

      if not self.extra_metadata == {}:
         rqet.create_metadata_xml(node = fi, extra_metadata=self.extra_metadata)

      assert self.domain in ['depth', 'time', 'mixed'], 'illegal domain value for fault interpretation'
      dom_node = rqet.SubElement(fi, ns['resqml2'] + 'Domain')
      dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
      dom_node.text = self.domain

      self.model.create_ref_node('InterpretedFeature', self.model.title_for_root(tectonic_boundary_feature_root),
                                 tectonic_boundary_feature_root.attrib['uuid'],
                                 content_type = 'obj_TectonicBoundaryFeature', root = fi)

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
            self.model.create_reciprocal_relationship(fi, 'destinationObject', tectonic_boundary_feature_root, 'sourceObject')

      self.root_node = fi

      return fi



class EarthModelInterpretation():
   """Class for RESQML Earth Model Interpretation organizational objects."""

   def __init__(self, parent_model, root_node = None, extract_from_xml = True,
                organization_feature = None, domain = 'depth'):
      """Initialises an earth model interpretation organisational object."""

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.domain = None
      self.organization_feature = None  # InterpretedFeature RESQML field
      self.feature_root = None
      self.has_occurred_during = (None, None)
      self.extra_metadata = {}

      if extract_from_xml and self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         self.domain = rqet.find_tag_text(self.root_node, 'Domain')
         interp_feature_ref_node = rqet.find_tag(self.root_node, 'InterpretedFeature')
         assert interp_feature_ref_node is not None
         self.feature_root = self.model.referenced_node(interp_feature_ref_node)
         if self.feature_root is not None:
            self.organization_feature = OrganizationFeature(self.model,
                                                            root_node = self.feature_root,
                                                            feature_name = self.model.title_for_root(self.feature_root))
         self.has_occurred_during = extract_has_occurred_during(self.root_node)
         self.extra_metadata = rqet.load_metadata_from_xml(self.root_node)
      else:
         assert organization_feature is not None
         assert domain in ['depth', 'time', 'mixed'], 'unrecognised domain value for earth model interpretation'
         self.uuid = bu.new_uuid()
         self.organization_feature = organization_feature
         self.domain = domain

      if self.uuid is None: self.uuid = bu.new_uuid()


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this interpretation is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, EarthModelInterpretation): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if self.organization_feature is not None:
         if not self.organization_feature.is_equivalent(other.organization_feature): return False
      elif other.organization_feature is not None: return False
      if self.root_node is not None and other.root_node is not None:
         if rqet.citation_title_for_node(self.root_node) !=  rqet.citation_title_for_node(other.root_node): return False
      elif self.root_node is not None or other.root_node is not None: return False
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      return self.domain == other.domain and equivalent_chrono_pairs(self.has_occurred_during, other.has_occurred_during)


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, organization_feature_root = None,
                  add_as_part = True, add_relationships = True, originator = None,
                  title_suffix = 'earth model interpretation'):
      """Creates an earth model interpretation organisational xml node from an earth model interpretation object."""

      # note: related organization feature node should be created first and referenced here

      emi = self.model.new_obj_node('EarthModelInterpretation')

      if self.organization_feature is not None:
         of_root = self.organization_feature.root_node
         if of_root is not None:
            if organization_feature_root is None:
               organization_feature_root = of_root
            else:
               assert of_root is organization_feature_root, 'organization feature mismatch'

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(emi.attrib['uuid'])
      else:
         emi.attrib['uuid'] = str(self.uuid)

      title = self.organization_feature.feature_name
      if title_suffix: title += ' ' + title_suffix
      self.model.create_citation(root = emi, title = title, originator = originator)

      if not self.extra_metadata == {}:
         rqet.create_metadata_xml(node = emi, extra_metadata=self.extra_metadata)

      assert self.domain in ['depth', 'time', 'mixed'], 'illegal domain value for horizon interpretation'
      dom_node = rqet.SubElement(emi, ns['resqml2'] + 'Domain')
      dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
      dom_node.text = self.domain

      self.model.create_ref_node('InterpretedFeature', self.model.title_for_root(organization_feature_root),
                                 organization_feature_root.attrib['uuid'],
                                 content_type = 'obj_OrganizationFeature', root = emi)

      create_xml_has_occurred_during(self.model, emi, self.has_occurred_during)

      if add_as_part:
         self.model.add_part('obj_EarthModelInterpretation', self.uuid, emi)
         if add_relationships:
            self.model.create_reciprocal_relationship(emi, 'destinationObject', organization_feature_root, 'sourceObject')

      self.root_node = emi

      return emi



class HorizonInterpretation():
   """Class for RESQML Horizon Interpretation organizational objects."""

   def __init__(self, parent_model, root_node = None, extract_from_xml = True,
                genetic_boundary_feature = None, domain = 'depth',
                boundary_relation_list = None,
                sequence_stratigraphy_surface = None):
      """Initialises a horizon interpretation organisational object."""

      # note: will create a paired GeneticBoundaryFeature object when loading from xml (and possibly a Surface object)

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.domain = None
      self.genetic_boundary_feature = None  # InterpretedFeature RESQML field, when not loading from xml
      self.feature_root = None
      self.has_occurred_during = (None, None)
      self.boundary_relation_list = None
      self.sequence_stratigraphy_surface = None
      self.extra_metadata = {}

      if extract_from_xml and self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         self.domain = rqet.find_tag_text(self.root_node, 'Domain')
         interp_feature_ref_node = rqet.find_tag(self.root_node, 'InterpretedFeature')
         assert interp_feature_ref_node is not None
         self.feature_root = self.model.referenced_node(interp_feature_ref_node)
         if self.feature_root is not None:
            self.genetic_boundary_feature = GeneticBoundaryFeature(self.model, kind = 'horizon',
                                                root_node = self.feature_root,
                                                feature_name = self.model.title_for_root(self.feature_root))
         self.has_occurred_during = extract_has_occurred_during(self.root_node)
         br_node_list = rqet.list_of_tag(self.root_node, 'BoundaryRelation')
         if br_node_list is not None and len(br_node_list) > 0:
            self.boundary_relation_list = []
            for br_node in br_node_list:
               self.boundary_relation_list.append(br_node.text)
         self.sequence_stratigraphy_surface = rqet.find_tag_text(self.root_node, 'SequenceStratigraphySurface')
         self.extra_metadata = rqet.load_metadata_from_xml(self.root_node)
      else:
         assert genetic_boundary_feature is not None
         assert domain in ['depth', 'time', 'mixed'], 'unrecognised domain value for horizon interpretation'
         if sequence_stratigraphy_surface is not None:
            assert sequence_stratigraphy_surface in ['flooding', 'ravinement', 'maximum flooding', 'transgressive']
            self.sequence_stratigraphy_surface = sequence_stratigraphy_surface
         self.uuid = bu.new_uuid()
         self.genetic_boundary_feature = genetic_boundary_feature
         self.domain = domain
         self.boundary_relation_list = boundary_relation_list
         if self.boundary_relation_list is not None:
            for boundary_relation in self.boundary_relation_list:
               assert boundary_relation in ['conformable', 'unconformable below and above',
                                            'unconformable above', 'unconformable below']

      if self.uuid is None: self.uuid = bu.new_uuid()


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this interpretation is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, HorizonInterpretation): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if self.genetic_boundary_feature is not None:
         if not self.genetic_boundary_feature.is_equivalent(other.genetic_boundary_feature): return False
      elif other.genetic_boundary_feature is not None: return False
      if self.root_node is not None and other.root_node is not None:
         if rqet.citation_title_for_node(self.root_node) !=  rqet.citation_title_for_node(other.root_node): return False
      elif self.root_node is not None or other.root_node is not None: return False
      if (self.domain != other.domain or
          not equivalent_chrono_pairs(self.has_occurred_during, other.has_occurred_during) or
          self.sequence_stratigraphy_surface != other.sequence_stratigraphy_surface): return False
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      if not self.boundary_relation_list and not other.boundary_relation_list: return True
      if not self.boundary_relation_list or not other.boundary_relation_list: return False
      return set(self.boundary_relation_list) == set(other.boundary_relation_list)


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, genetic_boundary_feature_root = None,
                  add_as_part = True, add_relationships = True, originator = None,
                  title_suffix = 'horizon interpretation'):
      """Creates a horizon interpretation organisational xml node from a horizon interpretation object."""

      # note: related genetic boundary feature node should be created first and referenced here

      hi = self.model.new_obj_node('HorizonInterpretation')

      if self.genetic_boundary_feature is not None:
         gbf_root = self.genetic_boundary_feature.root_node
         if gbf_root is not None:
            if genetic_boundary_feature_root is None:
               genetic_boundary_feature_root = gbf_root
            else:
               assert gbf_root is genetic_boundary_feature_root, 'genetic boundary feature mismatch'

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(hi.attrib['uuid'])
      else:
         hi.attrib['uuid'] = str(self.uuid)

      title = self.genetic_boundary_feature.feature_name
      if title_suffix: title += ' ' + title_suffix
      self.model.create_citation(root = hi, title = title, originator = originator)

      if not self.extra_metadata == {}:
         rqet.create_metadata_xml(node = hi, extra_metadata=self.extra_metadata)

      assert self.domain in ['depth', 'time', 'mixed'], 'illegal domain value for horizon interpretation'
      dom_node = rqet.SubElement(hi, ns['resqml2'] + 'Domain')
      dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
      dom_node.text = self.domain

      create_xml_has_occurred_during(self.model, hi, self.has_occurred_during)

      if self.boundary_relation_list is not None:
         for boundary_relation in self.boundary_relation_list:
            assert boundary_relation in ['conformable', 'unconformable below and above',
                                         'unconformable above', 'unconformable below']
            br_node = rqet.SubElement(hi, ns['resqml2'] + 'BoundaryRelation')
            br_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'BoundaryRelation')
            br_node.text = boundary_relation

      if self.sequence_stratigraphy_surface is not None:
         assert self.sequence_stratigraphy_surface in ['flooding', 'ravinement', 'maximum flooding', 'transgressive']
         sss_node = rqet.SubElement(hi, ns['resqml2'] + 'SequenceStratigraphySurface')
         sss_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'SequenceStratigraphySurface')
         sss_node.text = self.sequence_stratigraphy_surface

      self.model.create_ref_node('InterpretedFeature', self.model.title_for_root(genetic_boundary_feature_root),
                                 genetic_boundary_feature_root.attrib['uuid'],
                                 content_type = 'obj_GeneticBoundaryFeature', root = hi)

      if add_as_part:
         self.model.add_part('obj_HorizonInterpretation', self.uuid, hi)
         if add_relationships:
            self.model.create_reciprocal_relationship(hi, 'destinationObject', genetic_boundary_feature_root, 'sourceObject')

      self.root_node = hi

      return hi



class GeobodyBoundaryInterpretation():
   """Class for RESQML Geobody Boudary Interpretation organizational objects."""

   def __init__(self, parent_model, root_node = None, extract_from_xml = True,
                genetic_boundary_feature = None, domain = 'depth',
                boundary_relation_list = None):

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.domain = None
      self.boundary_relation_list = None
      self.genetic_boundary_feature = None  # InterpretedFeature RESQML field, when not loading from xml
      self.feature_root = None
      self.has_occurred_during = (None, None)
      self.extra_metadata = {}

      if extract_from_xml and self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         self.domain = rqet.find_tag_text(self.root_node, 'Domain')
         interp_feature_ref_node = rqet.find_tag(self.root_node, 'InterpretedFeature')
         assert interp_feature_ref_node is not None
         self.feature_root = self.model.referenced_node(interp_feature_ref_node)
         if self.feature_root is not None:
            self.genetic_boundary_feature = GeneticBoundaryFeature(self.model, kind = 'geobody boundary',
                                                root_node = self.feature_root,
                                                feature_name = self.model.title_for_root(self.feature_root))
         self.has_occurred_during = extract_has_occurred_during(self.root_node)
         br_node_list = rqet.list_of_tag(self.root_node, 'BoundaryRelation')
         if br_node_list is not None and len(br_node_list) > 0:
            self.boundary_relation_list = []
            for br_node in br_node_list:
               self.boundary_relation_list.append(br_node.text)
         self.extra_metadata = rqet.load_metadata_from_xml(self.root_node)
      else:
         assert genetic_boundary_feature is not None
         assert domain in ['depth', 'time', 'mixed']
         self.uuid = bu.new_uuid()
         self.domain = domain
         self.genetic_boundary_feature = genetic_boundary_feature
         self.boundary_relation_list = None
         if boundary_relation_list is not None:
            self.boundary_relation_list = boundary_relation_list.copy()
            for boundary_relation in self.boundary_relation_list:
               assert boundary_relation in ['conformable', 'unconformable below and above',
                                            'unconformable above', 'unconformable below']

      if self.uuid is None: self.uuid = bu.new_uuid()


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this interpretation is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, GeobodyBoundaryInterpretation): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if self.genetic_boundary_feature is not None:
         if not self.genetic_boundary_feature.is_equivalent(other.genetic_boundary_feature): return False
      elif other.genetic_boundary_feature is not None: return False
      if self.root_node is not None and other.root_node is not None:
         if rqet.citation_title_for_node(self.root_node) !=  rqet.citation_title_for_node(other.root_node): return False
      elif self.root_node is not None or other.root_node is not None: return False
      if (self.domain != other.domain or
          not equivalent_chrono_pairs(self.has_occurred_during, other.has_occurred_during)): return False
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      if not self.boundary_relation_list and not other.boundary_relation_list: return True
      if not self.boundary_relation_list or not other.boundary_relation_list: return False
      return set(self.boundary_relation_list) == set(other.boundary_relation_list)


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, genetic_boundary_feature_root = None,
                  add_as_part = True, add_relationships = True, originator = None,
                  title_suffix = 'geobody boundary interpretation'):
      """Creates a geobody boundary interpretation organisational xml node from a geobody boundary interpretation object."""

      gbi = self.model.new_obj_node('GeobodyBoundaryInterpretation')

      if self.genetic_boundary_feature is not None:
         gbf_root = self.genetic_boundary_feature.root_node
         if gbf_root is not None:
            if genetic_boundary_feature_root is None:
               genetic_boundary_feature_root = gbf_root
            else:
               assert gbf_root is genetic_boundary_feature_root, 'genetic boundary feature mismatch'
      else:
         if genetic_boundary_feature_root is None: genetic_boundary_feature_root = self.feature_root
         assert genetic_boundary_feature_root is not None
         self.genetic_boundary_feature = GeneticBoundaryFeature(self.model, root_node = genetic_boundary_feature_root)
      self.feature_root = genetic_boundary_feature_root

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(gbi.attrib['uuid'])
      else:
         gbi.attrib['uuid'] = str(self.uuid)

      title = self.genetic_boundary_feature.feature_name
      if title_suffix: title += ' ' + title_suffix
      self.model.create_citation(root = gbi, title = title, originator = originator)

      if not self.extra_metadata == {}:
         rqet.create_metadata_xml(node = gbi, extra_metadata=self.extra_metadata)

      assert self.domain in ['depth', 'time', 'mixed'], 'illegal domain value for geobody boundary interpretation'
      dom_node = rqet.SubElement(gbi, ns['resqml2'] + 'Domain')
      dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
      dom_node.text = self.domain

      create_xml_has_occurred_during(self.model, gbi, self.has_occurred_during)

      if self.boundary_relation_list is not None:
         for boundary_relation in self.boundary_relation_list:
            assert boundary_relation in ['conformable', 'unconformable below and above',
                                         'unconformable above', 'unconformable below']
            br_node = rqet.SubElement(gbi, ns['resqml2'] + 'BoundaryRelation')
            br_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'BoundaryRelation')
            br_node.text = str(boundary_relation)

      self.model.create_ref_node('InterpretedFeature', self.model.title_for_root(genetic_boundary_feature_root),
                                 genetic_boundary_feature_root.attrib['uuid'],
                                 content_type = 'obj_GeneticBoundaryFeature', root = gbi)

      if add_as_part:
         self.model.add_part('obj_GeobodyBoundaryInterpretation', self.uuid, gbi)
         if add_relationships:
            self.model.create_reciprocal_relationship(gbi, 'destinationObject', genetic_boundary_feature_root, 'sourceObject')

      self.root_node = gbi

      return gbi



class GeobodyInterpretation():
   """Class for RESQML Geobody Interpretation objects."""

   def __init__(self, parent_model, root_node = None, geobody_feature = None, domain = 'depth',
                composition = None, material_implacement = None, geobody_shape = None):
      """Initialise a new geobody interpretation object, either from xml or explicitly."""

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.domain = None
      self.geobody_feature = None  # InterpretedFeature RESQML field, when not loading from xml
      self.feature_root = None
      self.has_occurred_during = (None, None)
      self.composition = None
      self.implacement = None
      self.geobody_shape = None
      self.extra_metadata = {}

      if root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         self.domain = rqet.find_tag_text(self.root_node, 'Domain')
         interp_feature_ref_node = rqet.find_tag(self.root_node, 'InterpretedFeature')
         assert interp_feature_ref_node is not None
         self.feature_root = self.model.referenced_node(interp_feature_ref_node)
         if self.feature_root is not None:
            self.geobody_feature = GeobodyFeature(self.model,
                                                  root_node = self.feature_root,
                                                  feature_name = self.model.title_for_root(self.feature_root))
         self.has_occurred_during = extract_has_occurred_during(self.root_node)
         self.composition = rqet.find_tag_text(self.root_node, 'GeologicUnitComposition')
         self.implacement = rqet.find_tag_text(self.root_node, 'GeologicUnitMaterialImplacement')
         self.geobody_shape = rqet.find_tag_text(self.root_node, 'Geobody3dShape')
         self.extra_metadata = rqet.load_metadata_from_xml(self.root_node)
      elif geobody_feature is not None:
         self.uuid = bu.new_uuid()
         self.domain = domain
         self.geobody_feature = geobody_feature
         self.feature_root = geobody_feature.root_node
         self.composition = composition
         self.implacement = material_implacement
         self.geobody_shape = geobody_shape

      if self.uuid is None: self.uuid = bu.new_uuid()


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this interpretation is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, GeobodyInterpretation): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if self.geobody_feature is not None:
         if not self.geobody_feature.is_equivalent(other.geobody_feature): return False
      elif other.geobody_feature is not None: return False
      if self.root_node is not None and other.root_node is not None:
         if rqet.citation_title_for_node(self.root_node) !=  rqet.citation_title_for_node(other.root_node): return False
      elif self.root_node is not None or other.root_node is not None: return False
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      return (self.domain == other.domain and
              equivalent_chrono_pairs(self.has_occurred_during, other.has_occurred_during) and
              self.composition == other.composition and
              self.implacement == other.implacement and
              self.geobody_shape == other.geobody_shape)


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, geobody_feature_root = None,
                  add_as_part = True, add_relationships = True, originator = None,
                  title_suffix = 'geobody interpretation'):

      gi = self.model.new_obj_node('GeobodyInterpretation')

      if self.geobody_feature is not None:
         gbf_root = self.geobody_feature.root_node
         if gbf_root is not None:
            if geobody_feature_root is None:
               geobody_feature_root = gbf_root
            else:
               assert gbf_root is geobody_feature_root, 'geobody feature mismatch'
      else:
         if geobody_feature_root is None: geobody_feature_root = self.feature_root
         assert geobody_feature_root is not None
         self.geobody_feature = GeobodyFeature(self.model, root_node = geobody_feature_root)
      self.feature_root = geobody_feature_root

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(gi.attrib['uuid'])
      else:
         gi.attrib['uuid'] = str(self.uuid)

      title = self.geobody_feature.feature_name
      if title_suffix: title += ' ' + title_suffix
      self.model.create_citation(root = gi, title = title, originator = originator)

      if not self.extra_metadata == {}:
         rqet.create_metadata_xml(node = gi, extra_metadata=self.extra_metadata)

      assert self.domain in ['depth', 'time', 'mixed'], 'illegal domain value for geobody interpretation'
      dom_node = rqet.SubElement(gi, ns['resqml2'] + 'Domain')
      dom_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
      dom_node.text = self.domain

      create_xml_has_occurred_during(self.model, gi, self.has_occurred_during)

      if self.composition:
         assert self.composition in ['intrusive clay', 'organic', 'intrusive mud', 'evaporite salt',
                                     'evaporite non salt', 'sedimentary siliclastic', 'carbonate',
                                     'magmatic intrusive granitoid', 'magmatic intrusive pyroclastic',
                                     'magmatic extrusive lava flow', 'other chemichal rock',  # chemichal (stet)
                                     'other chemical rock', 'sedimentary turbidite']
         guc_node = rqet.SubElement(gi, ns['resqml2'] + 'GeologicUnitComposition')
         guc_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'GeologicUnitComposition')
         guc_node.text = self.composition
         # if self.composition.startswith('intrusive'): guc_node.text += ' '

      if self.implacement:
         assert self.implacement in ['autochtonous', 'allochtonous']
         gumi_node = rqet.SubElement(gi, ns['resqml2'] + 'GeologicUnitMaterialImplacement')
         gumi_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'GeologicUnitMaterialImplacement')
         gumi_node.text = self.implacement

      if self.geobody_shape:
         # note: 'silt' & 'sheeth' believed erroneous, so 'sill' and 'sheet' added
         assert self.geobody_shape in ['dyke', 'silt', 'sill', 'dome', 'sheeth', 'sheet', 'diapir',
                                       'batholith', 'channel', 'delta', 'dune', 'fan', 'reef', 'wedge']
         gs_node = rqet.SubElement(gi, ns['resqml2'] + 'Geobody3dShape')
         gs_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Geobody3dShape')
         gs_node.text = self.geobody_shape

      self.model.create_ref_node('InterpretedFeature', self.model.title_for_root(geobody_feature_root),
                                 geobody_feature_root.attrib['uuid'],
                                 content_type = 'obj_GeobodyFeature', root = gi)

      if add_as_part:
         self.model.add_part('obj_GeobodyInterpretation', self.uuid, gi)
         if add_relationships:
            self.model.create_reciprocal_relationship(gi, 'destinationObject', geobody_feature_root, 'sourceObject')

      self.root_node = gi

      return gi



class WellboreInterpretation():
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

   def __init__(self, parent_model, root_node = None, extract_from_xml = True, is_drilled = None,
                wellbore_feature = None, domain = 'depth'):
      """Initialises a wellbore interpretation organisational object."""

      # note: will create a paired WellboreFeature object when loading from xml

      self.model = parent_model
      self.root_node = root_node
      self.uuid = None
      self.is_drilled = None
      self.feature_root = None
      self.wellbore_feature = None
      self.title = None
      self.domain = None
      self.extra_metadata = {}

      if extract_from_xml and self.root_node is not None:
         self.uuid = self.root_node.attrib['uuid']
         self.is_drilled = rqet.find_tag_bool(self.root_node, 'IsDrilled')
         self.title = self.model.title_for_root(self.root_node)
         self.domain = rqet.find_tag_text(self.root_node, 'Domain')
         interp_feature_ref_node = rqet.find_tag(self.root_node, 'InterpretedFeature')
         if interp_feature_ref_node is not None:
            self.feature_root = self.model.referenced_node(interp_feature_ref_node)
            if self.feature_root is not None:
               self.wellbore_feature = WellboreFeature(self.model,
                                                       root_node = self.feature_root,
                                                       feature_name = self.model.title_for_root(self.feature_root))
         self.extra_metadata = rqet.load_metadata_from_xml(self.root_node)
      else:
         self.is_drilled = is_drilled
         self.wellbore_feature = wellbore_feature
         assert domain in ['depth', 'time', 'mixed'], 'unrecognised domain value for wellbore interpretation'
         self.domain = domain
         if wellbore_feature is not None: self.title = wellbore_feature.feature_name

      if self.uuid is None: self.uuid = bu.new_uuid()


   def trajectories(self):
      """ Iterable of associated trajectories """

      import resqpy.well

      parts = self.model.parts_list_related_to_uuid_of_type(
         self.uuid, type_of_interest='WellboreTrajectoryRepresentation'
      )
      for part in parts:
         traj_root = self.model.root_for_part(part)
         traj = resqpy.well.Trajectory(self.model, trajectory_root=traj_root)
         yield traj


   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this interpretation is essentially the same as the other; otherwise False."""

      if other is None or not isinstance(other, WellboreInterpretation): return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid): return True
      if self.wellbore_feature is not None:
         if not self.wellbore_feature.is_equivalent(other.wellbore_feature): return False
      elif other.wellbore_feature is not None: return False
      if self.root_node is not None and other.root_node is not None:
         if rqet.citation_title_for_node(self.root_node) !=  rqet.citation_title_for_node(other.root_node): return False
         if self.domain != other.domain: return False
      elif self.root_node is not None or other.root_node is not None: return False
      if check_extra_metadata and not equivalent_extra_metadata(self, other): return False
      return (self.title == other.title and self.is_drilled == other.is_drilled)


   def __eq__(self, other):
      return self.is_equivalent(other)


   def __ne__(self, other):
      return not self.is_equivalent(other)


   def create_xml(self, wellbore_feature_root = None,
                  add_as_part = True, add_relationships = True, originator = None,
                  title_suffix = None):
      """Creates a wellbore interpretation organisational xml node from a wellbore interpretation object."""

      # note: related wellbore feature node should be created first and referenced here

      wi = self.model.new_obj_node('WellboreInterpretation')

      if self.wellbore_feature is not None:
         wbf_root = self.wellbore_feature.root_node
         if wbf_root is not None:
            if wellbore_feature_root is None:
               wellbore_feature_root = wbf_root
            else:
               assert wbf_root is wellbore_feature_root, 'wellbore feature mismatch'

      if self.uuid is None:
         self.uuid = bu.uuid_from_string(wi.attrib['uuid'])
      else:
         wi.attrib['uuid'] = str(self.uuid)

      title = self.wellbore_feature.feature_name
      if title_suffix: title += ' ' + title_suffix
      self.model.create_citation(root = wi, title = title, originator = originator)

      if not self.extra_metadata == {}:
         rqet.create_metadata_xml(node = wi, extra_metadata=self.extra_metadata)

      if self.is_drilled is None: self.is_drilled = False

      id_node = rqet.SubElement(wi, ns['resqml2'] + 'IsDrilled')
      id_node.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
      id_node.text = str(self.is_drilled).lower()

      assert self.domain in ['depth', 'time', 'mixed'], 'illegal domain value for wellbore interpretation'
      domain_node = rqet.SubElement(wi, ns['resqml2'] + 'Domain')
      domain_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Domain')
      domain_node.text = str(self.domain).lower()

      self.model.create_ref_node('InterpretedFeature', self.model.title_for_root(wellbore_feature_root),
                                 wellbore_feature_root.attrib['uuid'],
                                 content_type = 'obj_WellboreFeature', root = wi)

      if add_as_part:
         self.model.add_part('obj_WellboreInterpretation', self.uuid, wi)
         if add_relationships:
            self.model.create_reciprocal_relationship(wi, 'destinationObject', wellbore_feature_root, 'sourceObject')

      self.root_node = wi

      return wi
