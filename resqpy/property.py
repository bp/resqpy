"""property.py: module handling collections of RESQML properties for grids, wellbore frames, grid connection sets etc."""

version = '7th September 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)
log.debug('property.py version ' + version)

import os
import numpy as np
import numpy.ma as ma
import pandas as pd
# import xml.etree.ElementTree as et
from datetime import datetime
from functools import lru_cache
#  from lxml import etree as et

import lasio

from resqpy.olio.base import BaseResqpy
import resqpy.olio.ab_toolbox as abt
import resqpy.olio.write_data as wd
import resqpy.olio.load_data as ld
import resqpy.olio.uuid as bu
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.olio.box_utilities as bxu
from resqpy.olio.xml_namespaces import curly_namespace as ns
import resqpy.weights_and_measures as bwam

import resqpy.time_series as rts

# following are loaded dynamically, to avoid circular reference during import (issue for python version < 3.7)
# import resqpy.grid as grr
# import resqpy.well as rqw
# import resqpy.grid as rqs
# import resqpy.fault as rqf

# the following resqml property kinds and facet types are 'known about' by this module in relation to nexus
# other property kinds should be handled okay but without any special treatment
# see property_kind_and_facet_from_keyword() for simulator keyword to property kind and facet mapping

supported_property_kind_list = [
   'code', 'index', 'depth', 'rock volume', 'pore volume', 'volume', 'thickness', 'length', 'cell length',
   'net to gross ratio', 'porosity', 'permeability thickness', 'permeability length', 'permeability rock',
   'rock permeability', 'fluid volume', 'transmissibility', 'pressure', 'saturation', 'solution gas-oil ratio',
   'vapor oil-gas ratio', 'property multiplier', 'thermodynamic temperature', 'continuous', 'discrete', 'categorical'
]

supported_local_property_kind_list = [
   'active', 'transmissibility multiplier', 'fault transmissibility', 'mat transmissibility'
]

supported_facet_type_list = ['direction', 'netgross', 'what']

# current implementation limits a property to having at most one facet; resqml standard allows for many
# if a property kind does not appear in the following dictionary, facet_type and facet are expected to be None
# mapping from property kind to (facet_type, list of possible facet values), only applicable when indexable element is 'cells'
# use of the following in code is DEPRECATED; it remains here to give a hint of facet types and facets in use

expected_facet_type_dict = {
   'depth': ('what', ['cell centre', 'cell top']),  # made up
   'rock volume': ('netgross', ['net', 'gross']),  # resqml standard
   'thickness': ('netgross', ['net', 'gross']),
   'length': ('direction', ['X', 'Y', 'Z']),  # facet values made up
   'cell length': ('direction', ['I', 'J', 'K']),  # facet values made up
   'permeability rock': ('direction', ['I', 'J', 'K']),  # todo: allow IJ and IJK
   'rock permeability': ('direction', ['I', 'J', 'K']),  # todo: allow IJ and IJK
   'transmissibility': ('direction', ['I', 'J', 'K']),
   'transmissibility multiplier': (
      'direction',  # local property kind
      ['I', 'J', 'K']),
   'fault transmissibility': ('direction', ['I', 'J']),  # local property kind; K faces also permitted
   'mat transmissibility': ('direction', ['K']),  # local property kind; I & J faces also permitted
   'saturation': (
      'what',
      [
         'water',
         'oil',
         'gas',  # made up but probably good
         'water minimum',
         'gas minimum',
         'oil minimum',  # made up for end-points
         'water residual',
         'gas residual',
         'oil residual',
         'water residual to oil',
         'gas residual to oil'
      ]),
   'fluid volume': (
      'what',
      [
         'water',
         'oil',
         'gas',  # made up but probably good
         'water (mobile)',
         'oil (mobile)',
         'gas (mobile)'
      ]),  # made up
   'property multiplier': (
      'what',
      [
         'rock volume',  # made up; todo: add rock permeability?
         'pore volume',
         'transmissibility'
      ]),
   'code': ('what', ['inactive', 'active']),  # NB: user defined keywords also accepted
   'index': ('what', ['uid', 'pixel map'])
}


class PropertyCollection():
   """Class for RESQML Property collection for any supporting representation (or mix of supporting representations).

   notes:
      this is a base class inherited by GridPropertyCollection and WellLogCollection (and others to follow), application
      code usually works with the derived classes;
      resqml caters for three types of property: Continuous (ie. real data, aka floating point);
      Discrete (ie. integer data, or boolean); Categorical (integer data, usually non-negative, with an associated
      look-up table to convert to a string)
   """

   def __init__(self, support = None, property_set_root = None, realization = None):
      """Initialise an empty Property Collection; if support is not None, populate with properties for that representation.

      arguments:
         support (optional): a grid.Grid object or a well.WellboreFrame object which belongs to a resqpy.Model which includes
            associated properties; if this argument is given, and property_set_root is None, the properties in the support's
            parent model which are for this representation (ie. have this grid or wellbore frame as the supporting representation)
            are added to this collection as part of the initialisation
         property_set_root (optional): if present, the collection is populated with the properties defined in the xml tree
            of the property set
         realization (integer, optional): if present, the single realisation (within an ensemble) that this collection is for;
            if None, then the collection is either covering a whole ensemble (individual properties can each be flagged with a
            realisation number), or is for properties that do not have multiple realizations

      note:
         at present, if the collection is being initialised from a property set, the support argument must also be specified;
         also for now, if not initialising from a property set, all properties related to the support are included, whether
         the relationship is supporting representation or some other relationship;
         the full handling of resqml property sets and property series is still under development

      :meta common:
      """

      assert property_set_root is None or support is not None,  \
         'support (grid, wellbore frame, blocked well, mesh, or grid connection set) must be specified when populating property collection from property set'

      self.dict = {}  # main dictionary of model property parts which are members of the collection
      # above is mapping from part_name to:
      # (realization, support, uuid, xml_node, continuous, count, indexable, prop_kind, facet_type, facet, citation_title,
      #   time_series_uuid, time_index, min, max, uom, string_lookup_uuid, property_kind_uuid, extra_metadata, null_value, const_value)
      #  0            1        2     3         4           5      6          7          8           9      10
      #   11                12          13   14   15   16                  17                  18              19          20
      # note: grid is included to allow for super-collections covering more than one grid
      # todo: replace items 8 & 9 with a facet dictionary (to allow for multiple facets)
      self.model = None
      self.support = None
      self.support_root = None
      self.support_uuid = None
      self.property_set_root = None
      self.time_set_kind_attr = None
      self.has_single_property_kind_flag = None
      self.has_single_uom_flag = None
      self.has_single_indexable_element_flag = None
      self.has_multiple_realizations_flag = None
      self.parent_set_root = None
      self.realization = realization  # model realization number within an ensemble
      self.imported_list = []  # list of (uuid, file_name, keyword, cached_name, discrete, uom, time_index, null_value,
      # min_value, max_value, property_kind, facet_type, facet, realization,
      # indexable_element, count, local_property_kind_uuid, const_value)
      if support is not None:
         self.model = support.model
         self.set_support(support = support)
         assert self.model is not None
         # assert self.support_root is not None
         assert self.support_uuid is not None
         if property_set_root is None:
            # todo: make more rigorous by looking up supporting representation node uuids
            props_list = self.model.parts_list_of_type(type_of_interest = 'obj_DiscreteProperty')
            discrete_props_list = self.model.parts_list_filtered_by_supporting_uuid(props_list, self.support_uuid)
            self.add_parts_list_to_dict(discrete_props_list)
            props_list = self.model.parts_list_of_type(type_of_interest = 'obj_CategoricalProperty')
            categorical_props_list = self.model.parts_list_filtered_by_supporting_uuid(props_list, self.support_uuid)
            self.add_parts_list_to_dict(categorical_props_list)
            props_list = self.model.parts_list_of_type(type_of_interest = 'obj_ContinuousProperty')
            continuous_props_list = self.model.parts_list_filtered_by_supporting_uuid(props_list, self.support_uuid)
            self.add_parts_list_to_dict(continuous_props_list)
         else:
            self.populate_from_property_set(property_set_root)

   def set_support(self, support_uuid = None, support = None, model = None, modify_parts = True):
      """Sets the supporting object associated with this collection, without loading props, if not done so at initialisation.

      Arguments:
         support_uuid: the uuid of the supporting representation which the properties in this collection are for
         support: a grid.Grid, unstructured.UnstructuredGrid (or derived class), well.WellboreFrame, well.BlockedWell,
            surface.Mesh, or fault.GridConnectionSet object which the properties in this collection are for
         model (model.Model object, optional): if present, the model associated with this collection is set to this;
            otherwise the model is assigned from the supporting object
         modify_parts (boolean, default True): if True, any parts already in this collection have their individual
            support_uuid set
      """

      # when at global level was causing circular reference loading issues as grid imports this module
      import resqpy.grid as grr
      import resqpy.unstructured as rug
      import resqpy.well as rqw
      import resqpy.surface as rqs
      import resqpy.fault as rqf

      # todo: check uuid's of individual parts' supports match that of support being set for whole collection

      if model is None and support is not None:
         model = support.model
      if model is None:
         model = self.model
      else:
         self.model = model

      if support_uuid is None and support is not None:
         support_uuid = support.uuid

      if support_uuid is None:
         if self.support_uuid is not None:
            log.warning('clearing supporting representation for property collection')
#         self.model = None
         self.support = None
         self.support_root = None
         self.support_uuid = None

      else:
         assert model is not None, 'model not established when setting support for property collection'
         if self.support_uuid is not None and not bu.matching_uuids(support_uuid, self.support_uuid):
            log.warning('changing supporting representation for property collection')
         self.support_uuid = support_uuid
         self.support = support
         if self.support is None:
            support_part = model.part_for_uuid(support_uuid)
            assert support_part is not None, 'supporting representation part missing in model'
            self.support_root = model.root_for_part(support_part)
            support_type = model.type_of_part(support_part)
            assert support_type is not None
            if support_type == 'obj_IjkGridRepresentation':
               self.support = grr.any_grid(model, uuid = self.support_uuid, find_properties = False)
            elif support_type == 'obj_WellboreFrameRepresentation':
               self.support = rqw.WellboreFrame(model, frame_root = self.support_root)
            elif support_type == 'obj_BlockedWellboreRepresentation':
               self.support = rqw.BlockedWell(model, blocked_well_root = self.support_root)
            elif support_type == 'obj_Grid2dRepresentation':
               self.support = rqs.Mesh(model, uuid = self.support_uuid)
            elif support_type == 'obj_GridConnectionSetRepresentation':
               self.support = rqf.GridConnectionSet(model, uuid = self.support_uuid)
            elif support_type == 'obj_UnstructuredGridRepresentation':
               self.support = rug.UnstructuredGrid(model,
                                                   uuid = self.support_uuid,
                                                   geometry_required = False,
                                                   find_properties = False)
            else:
               raise TypeError('unsupported property supporting representation class: ' + str(support_type))
         else:
            if type(self.support) in [
                  grr.Grid, grr.RegularGrid, rqw.WellboreFrame, rqw.BlockedWell, rqs.Mesh, rqf.GridConnectionSet,
                  rug.UnstructuredGrid, rug.HexaGrid, rug.TetraGrid, rug.PrismGrid, rug.VerticalPrismGrid,
                  rug.PyramidGrid
            ]:
               self.support_root = self.support.root
            else:
               raise TypeError('unsupported property supporting representation class: ' + str(type(self.support)))
         if modify_parts:
            for (part, info) in self.dict.items():
               if info[1] is not None:
                  modified = list(info)
                  modified[1] = support_uuid
                  self.dict[part] = tuple(modified)

   def supporting_shape(self, indexable_element = None, direction = None):
      """Returns the shape of the supporting representation with respect to the given indexable element, as a list of ints.

      arguments:
         indexable_element (string, optional): if None, a hard-coded default depending on the supporting representation class
            will be used
         direction (string, optional): must be passed if required for the combination of support class and indexable element;
            currently only used for Grid faces.

      returns:
         list of int, being required shape of numpy array, or None if not coded for

      note:
         individual property arrays will only match this shape if they have the same indexable element and a count of one
      """

      # when at global level was causing circular reference loading issues as grid imports this module
      import resqpy.grid as grr
      import resqpy.unstructured as rug
      import resqpy.well as rqw
      import resqpy.surface as rqs
      import resqpy.fault as rqf

      shape_list = None
      support = self.support

      if isinstance(support, grr.Grid):
         if indexable_element is None or indexable_element == 'cells':
            shape_list = [support.nk, support.nj, support.ni]
         elif indexable_element == 'columns':
            shape_list = [support.nj, support.ni]
         elif indexable_element == 'layers':
            shape_list = [support.nk]
         elif indexable_element == 'faces':
            assert direction is not None and direction.upper() in 'IJK'
            axis = 'KJI'.index(direction.upper())
            shape_list = [support.nk, support.nj, support.ni]
            shape_list[axis] += 1  # note: properties for grid faces include outer faces
         elif indexable_element == 'column edges':
            shape_list = [(support.nj * (support.ni + 1)) + ((support.nj + 1) * support.ni)
                         ]  # I edges first; include outer edges
         elif indexable_element == 'edges per column':
            shape_list = [support.nj, support.ni, 4]  # assume I-, J+, I+, J- ordering
         elif indexable_element == 'faces per cell':
            shape_list = [support.nk, support.nj, support.ni, 6]  # assume K-, K+, J-, I+, J+, I- ordering
            # TODO: resolve ordering of edges and make consistent with maps code (edges per column) and fault module (gcs faces)

      elif isinstance(support, rqw.WellboreFrame):
         if indexable_element is None or indexable_element == 'nodes':
            shape_list = [support.node_count]
         elif indexable_element == 'intervals':
            shape_list = [support.node_count - 1]

      elif isinstance(support, rqw.BlockedWell):
         if indexable_element is None or indexable_element == 'intervals':
            shape_list = [support.node_count - 1]  # all intervals, including unblocked
#            shape_list = [support.cell_count]  # for blocked intervals only – use 'cells' as indexable element
         elif indexable_element == 'nodes':
            shape_list = [support.node_count]
         elif indexable_element == 'cells':
            shape_list = [support.cell_count]  # ie. blocked intervals only

      elif isinstance(support, rqs.Mesh):
         if indexable_element is None or indexable_element == 'cells' or indexable_element == 'columns':
            shape_list = [support.nj - 1, support.ni - 1]
         elif indexable_element == 'nodes':
            shape_list = [support.nj, support.ni]

      elif isinstance(support, rqf.GridConnectionSet):
         if indexable_element is None or indexable_element == 'faces':
            shape_list = [support.count]

      elif type(support) in [
            rug.UnstructuredGrid, rug.HexaGrid, rug.TetraGrid, rug.PrismGrid, rug.VerticalPrismGrid, rug.PyramidGrid
      ]:
         if indexable_element is None or indexable_element == 'cells':
            shape_list = [support.cell_count]
         elif indexable_element == 'faces per cell':
            support.cache_all_geometry_arrays()
            shape_list = [len(support.faces_per_cell)]

      else:
         raise Exception(f'unsupported support class {type(support)} for property')

      return shape_list

   def populate_from_property_set(self, property_set_root):
      """Populates this (newly created) collection based on xml members of property set."""

      assert property_set_root is not None, 'missing property set xml root'
      assert self.model is not None and self.support is not None, 'set support for collection before populating from property set'

      self.property_set_root = property_set_root
      self.time_set_kind_attr = rqet.find_tag_text(property_set_root, 'TimeSetKind')
      self.has_single_property_kind_flag = rqet.find_tag_bool(property_set_root, 'HasSinglePropertyKind')
      self.has_multiple_realizations_flag = rqet.find_tag_bool(property_set_root, 'HasMultipleRealizations')
      parent_set_ref_root = rqet.find_tag(property_set_root, 'ParentSet')  # at most one parent set handled here
      if parent_set_ref_root is not None:
         self.parent_set_root = self.model.referenced_node(parent_set_ref_root)
      # loop over properties in property set xml, adding parts to main dictionary
      for child in property_set_root:
         if rqet.stripped_of_prefix(child.tag) != 'Properties':
            continue
         property_root = self.model.referenced_node(child)
         if property_root is None:
            log.warning('property set member missing from resqml dataset')
            continue
         self.add_part_to_dict(rqet.part_name_for_part_root(property_root))

   def set_realization(self, realization):
      """Sets the model realization number (within an ensemble) for this collection.

         argument:
            realization (non-negative integer): the realization number of the whole collection within an ensemble of
                    collections

         note:
            the resqml Property classes allow for a realization index to be associated with an individual property
            array; this module supports this by associating a realization number (equivalent to realization index) for
            each part (ie. for each property array); however, the collection can be given a realization number which is
            then applied to each member of the collection as it is added, if no part-specific realization number is
            provided
      """

      # the following assertion might need to be downgraded to a warning, to allow for reassignment of realization numbers
      assert self.realization is None
      self.realization = realization

   def add_part_to_dict(self, part, continuous = None, realization = None, trust_uom = True):
      """Add the named part to the dictionary for this collection.

         arguments:
            part (string): the name of a part (which exists in the support's parent model) to be added to this collection
            continuous (boolean, optional): whether the property is of a continuous (real) kind; if not None,
                     is checked against the property's type and an assertion error is raised if there is a mismatch
            realization (integer, optional): if present, must match this collection's realization number if that is
                     not None; if this argument is None then the part is assigned the realization number associated
                     with this collection as a whole; if the xml for the part includes a realization index then that
                     overrides these other sources to become the realization number
            trust_uom (boolean, default True): if True, the uom stored in the part's xml is used as the part's uom
                     in this collection; if False and the uom in the xml is an empty string or 'Euc', then the
                     part's uom in this collection is set to a guess based on the property kind and min & max values;
                     note that this guessed value is not used to overwrite the value in the xml
      """

      if part is None:
         return
      assert part not in self.dict
      if realization is not None and self.realization is not None:
         assert (realization == self.realization)
      if realization is None:
         realization = self.realization
      uuid = self.model.uuid_for_part(part, is_rels = False)
      assert uuid is not None
      xml_node = self.model.root_for_part(part, is_rels = False)
      assert xml_node is not None
      type = self.model.type_of_part(part)
      #      log.debug('adding part ' + part + ' of type ' + type)
      assert type in ['obj_ContinuousProperty', 'obj_DiscreteProperty', 'obj_CategoricalProperty']
      if continuous is None:
         continuous = (type == 'obj_ContinuousProperty')
      else:
         assert continuous == (type == 'obj_ContinuousProperty')
      string_lookup_uuid = None
      if type == 'obj_CategoricalProperty':
         sl_ref_node = rqet.find_tag(xml_node, 'Lookup')
         string_lookup_uuid = bu.uuid_from_string(rqet.find_tag_text(sl_ref_node, 'UUID'))
      extra_metadata = rqet.load_metadata_from_xml(xml_node)
      count_node = rqet.find_tag(xml_node, 'Count')
      assert count_node is not None
      count = int(count_node.text)
      realization_node = rqet.find_tag(xml_node, 'RealizationIndex')  # optional; if present use to populate realization
      if realization_node is not None:
         realization = int(realization_node.text)
      indexable_node = rqet.find_tag(xml_node, 'IndexableElement')
      assert indexable_node is not None
      indexable = indexable_node.text
      citation_title = rqet.find_tag(rqet.find_tag(xml_node, 'Citation'), 'Title').text
      (property_kind, facet_type, facet) = property_kind_and_facet_from_keyword(citation_title)
      prop_kind_node = rqet.find_tag(xml_node, 'PropertyKind')
      assert (prop_kind_node is not None)
      kind_node = rqet.find_tag(prop_kind_node, 'Kind')
      property_kind_uuid = None  # only used for bespoke (local) property kinds
      if kind_node is not None:
         property_kind = kind_node.text  # could check for consistency with that derived from citation title
      else:
         lpk_node = rqet.find_tag(prop_kind_node, 'LocalPropertyKind')
         if lpk_node is not None:
            property_kind = rqet.find_tag_text(lpk_node, 'Title')
            property_kind_uuid = rqet.find_tag_text(lpk_node, 'UUID')
      assert property_kind is not None and len(property_kind) > 0
      facet_type = None
      facet = None
      facet_node = rqet.find_tag(xml_node, 'Facet')  # might have to handle more than one facet for a property?
      if facet_node is not None:
         facet_type = rqet.find_tag(facet_node, 'Facet').text
         facet = rqet.find_tag(facet_node, 'Value').text
         if facet_type is not None and facet_type == '':
            facet_type = None
         if facet is not None and facet == '':
            facet = None
      time_series_uuid = None
      time_index = None
      time_node = rqet.find_tag(xml_node, 'TimeIndex')
      if time_node is not None:
         time_index = int(rqet.find_tag(time_node, 'Index').text)
         time_series_uuid = bu.uuid_from_string(rqet.find_tag(rqet.find_tag(time_node, 'TimeSeries'), 'UUID').text)
      minimum = None
      min_node = rqet.find_tag(xml_node, 'MinimumValue')
      if min_node is not None:
         minimum = min_node.text  # NB: left as text
      maximum = None
      max_node = rqet.find_tag(xml_node, 'MaximumValue')
      if max_node is not None:
         maximum = max_node.text  # NB: left as text
      uom = None
      support_uuid = self.model.supporting_representation_for_part(part)
      if support_uuid is None:
         support_uuid = self.support_uuid
      elif self.support_uuid is not None and not bu.matching_uuids(support_uuid, self.support.uuid):
         self.set_support(None)
      if continuous:
         uom_node = rqet.find_tag(xml_node, 'UOM')
         if uom_node is not None and (trust_uom or uom_node.text not in ['', 'Euc']):
            uom = uom_node.text
         else:
            uom = guess_uom(property_kind, minimum, maximum, self.support, facet_type = facet_type, facet = facet)
      null_value = None
      if not continuous:
         null_value = rqet.find_nested_tags_int(xml_node, ['PatchOfValues', 'Values', 'NullValue'])
      const_value = None
      values_node = rqet.find_nested_tags(xml_node, ['PatchOfValues', 'Values'])
      values_type = rqet.node_type(values_node)
      assert values_type is not None
      if values_type.endswith('ConstantArray'):
         if continuous:
            const_value = rqet.find_tag_float(values_node, 'Value')
         elif values_type.startswith('Bool'):
            const_value = rqet.find_tag_bool(values_node, 'Value')
         else:
            const_value = rqet.find_tag_int(values_node, 'Value')
      self.dict[part] = (realization, support_uuid, uuid, xml_node, continuous, count, indexable, property_kind,
                         facet_type, facet, citation_title, time_series_uuid, time_index, minimum, maximum, uom,
                         string_lookup_uuid, property_kind_uuid, extra_metadata, null_value, const_value)

   def add_parts_list_to_dict(self, parts_list):
      """Add all the parts named in the parts list to the dictionary for this collection.

      argument:
         parts_list: a list of strings, each being the name of a part in the support's parent model

      note:
         the add_part_to_dict() function is called for each part in the list
      """

      for part in parts_list:
         self.add_part_to_dict(part)

   def remove_part_from_dict(self, part):
      """Remove the named part from the dictionary for this collection.

      argument:
         part (string): the name of a part which might be in this collection, to be removed

      note:
         if the part is not in the collection, no action is taken and no exception is raised
      """

      if part is None:
         return
      if part not in self.dict:
         return
      del self.dict[part]

   def remove_parts_list_from_dict(self, parts_list):
      """Remove all the parts named in the parts list from the dictionary for this collection.

      argument:
         parts_list: a list of strings, each being the name of a part which might be in the collection

      note:
         the remove_part_from_dict() function is called for each part in the list
      """

      for part in parts_list:
         self.remove_part_from_dict(part)

   def inherit_imported_list_from_other_collection(self, other, copy_cached_arrays = True, exclude_inactive = False):
      """Extends this collection's imported list with items from other's imported list.

      arguments:
         other: another PropertyCollection object with some imported arrays
         copy_cached_array (boolean, default True): if True, arrays cached with the other
            collection are copied and cached with this collection
         exclude_inactive (boolean, default False): if True, any item in the other imported list
            which has INACTIVE or ACTIVE as the keyword is excluded from the inheritance

      note:
         the imported list is a list of cached imported arrays with basic info for each array;
         it is used as a staging post before fully incorporating the imported arrays as parts
         of the support's parent model and writing the arrays to the hdf5 file
      """

      # note: does not inherit parts
      if exclude_inactive:
         other_list = []
         for imp in other.imported_list:
            if imp[2].upper() not in ['INACTIVE', 'ACTIVE']:
               other_list.append(imp)
      else:
         other_list = other.imported_list
      self.imported_list += other_list
      if copy_cached_arrays:
         for imp in other_list:
            if imp[17] is not None:
               continue  # constant array
            cached_name = imp[3]
            self.__dict__[cached_name] = other.__dict__[cached_name].copy()

   def inherit_parts_from_other_collection(self, other, ignore_clashes = False):
      """Adds all the parts in the other PropertyCollection to this one.

      Arguments:
         other: another PropertyCollection object related to the same support as this collection
         ignore_clashes (boolean, default False): if False, any part in other which is already in
            this collection will result in an assertion error; if True, such duplicates are
            simply skipped without modifying the existing part in this collection
      """

      assert self.support_uuid is None or other.support_uuid is None or bu.matching_uuids(
         self.support_uuid, other.support_uuid)
      if self.support_uuid is None and self.number_of_parts() == 0 and other.support_uuid is not None:
         self.set_support(support_uuid = other.support_uuid, support = other.support)
      if self.realization is not None and other.realization is not None:
         assert self.realization == other.realization
      for (part, info) in other.dict.items():
         if part in self.dict.keys():
            if ignore_clashes:
               continue
            assert False, 'attempt to inherit a part which already exists in property collection: ' + part
         self.dict[part] = info

   def inherit_parts_selectively_from_other_collection(
         self,
         other,
         realization = None,
         support_uuid = None,
         grid = None,  # for backward compatibility
         uuid = None,
         continuous = None,
         count = None,
         indexable = None,
         property_kind = None,
         facet_type = None,
         facet = None,
         citation_title = None,
         citation_title_match_starts_with = False,
         time_series_uuid = None,
         time_index = None,
         uom = None,
         string_lookup_uuid = None,
         categorical = None,
         ignore_clashes = False):
      """Adds those parts from the other PropertyCollection which match all arguments that are not None.

      arguments:
         other: another PropertyCollection object related to the same support as this collection
         citation_title_match_starts_with (boolean, default False): if True, any citation title that
            starts with the given citation_title argument will be deemed to have passed that filter
         ignore_clashes (boolean, default False): if False, any part in other which passes the filters
            yet is already in this collection will result in an assertion error; if True, such duplicates
            are simply skipped without modifying the existing part in this collection

      Other optional arguments (realization, grid, uuid, continuous, count, indexable, property_kind, facet_type,
      facet, citation_title, time_series_uuid, time_index, uom, string_lookup_uuid, categorical):

      For each of these arguments: if None, then all members of collection pass this filter;
      if not None then only those members with the given value pass this filter;
      finally, the filters for all the attributes must be passed for a given member (part)
      to be inherited.

      note:

         the grid argument is maintained for backward compatibility; it is treated synonymously with support
         which takes precendence; the categorical boolean argument can be used to filter only Categorical
         (or non-Categorical) properties

      """

      #      log.debug('inheriting parts selectively')
      if support_uuid is None and grid is not None:
         support_uuid = grid.uuid
      if support_uuid is not None and self.support_uuid is not None:
         assert bu.matching_uuids(support_uuid, self.support_uuid)
      assert other is not None
      if self.model is None:
         self.model = other.model
      else:
         assert self.model is other.model
      assert self.support_uuid is None or other.support_uuid is None or bu.matching_uuids(
         self.support_uuid, other.support_uuid)
      if self.support_uuid is None and self.number_of_parts() == 0:
         self.set_support(support_uuid = other.support_uuid, support = other.support)
      if self.realization is not None and other.realization is not None:
         assert self.realization == other.realization
      if time_index is not None:
         assert time_index >= 0
      for (part, info) in other.dict.items():
         if realization is not None and other.realization_for_part(part) != realization:
            continue
         if support_uuid is not None and not bu.matching_uuids(support_uuid, other.support_uuid_for_part(part)):
            continue
         if uuid is not None and not bu.matching_uuids(uuid, other.uuid_for_part(part)):
            continue
         if continuous is not None and other.continuous_for_part(part) != continuous:
            continue
         if categorical is not None:
            if categorical:
               if other.continuous_for_part(part) or (other.string_lookup_uuid_for_part(part) is None):
                  continue
            else:
               if not (other.continuous_for_part(part) or (other.string_lookup_uuid_for_part(part) is None)):
                  continue
         if count is not None and other.count_for_part(part) != count:
            continue
         if indexable is not None and other.indexable_for_part(part) != indexable:
            continue
         if property_kind is not None and not same_property_kind(other.property_kind_for_part(part), property_kind):
            continue
         if facet_type is not None and other.facet_type_for_part(part) != facet_type:
            continue
         if facet is not None and other.facet_for_part(part) != facet:
            continue
         if citation_title is not None:
            if citation_title_match_starts_with:
               if not other.citation_title_for_part(part).startswith():
                  continue
            else:
               if other.citation_title_for_part(part) != citation_title:
                  continue
         if time_series_uuid is not None and not bu.matching_uuids(time_series_uuid,
                                                                   other.time_series_uuid_for_part(part)):
            continue
         if time_index is not None and other.time_index_for_part(part) != time_index:
            continue
         if string_lookup_uuid is not None and not bu.matching_uuids(string_lookup_uuid,
                                                                     other.string_lookup_uuid_for_part(part)):
            continue
         if part in self.dict.keys():
            if ignore_clashes:
               continue
            assert (False)
         self.dict[part] = other.dict[part]

   def inherit_similar_parts_for_time_series_from_other_collection(self,
                                                                   other,
                                                                   example_part,
                                                                   citation_title_match_starts_with = False,
                                                                   ignore_clashes = False):
      """Adds the example part from other collection and any other parts for the same property at different times.

      arguments:
         other: another PropertyCollection object related to the same grid as this collection, from which to inherit
         example_part (string): the part name of an example member of other (which has an associated time_series)
         citation_title_match_starts_with (booleam, default False): if True, any citation title that starts with
            the example's citation_title after removing trailing digits, will be deemed to have passed that filter
         ignore_clashes (boolean, default False): if False, any part in other which passes the filters
            yet is already in this collection will result in an assertion error; if True, such duplicates
            are simply skipped without modifying the existing part in this collection

      note:
         at present, the citation title must match (as well as the other identifying elements) for a part to be
         inherited
      """

      assert other is not None
      assert other.part_in_collection(example_part)
      time_series_uuid = other.time_series_uuid_for_part(example_part)
      assert time_series_uuid is not None
      title = other.citation_title_for_part(example_part)
      if citation_title_match_starts_with:
         while title and title[-1].isdigit():
            title = title[:-1]
      self.inherit_parts_selectively_from_other_collection(
         other,
         realization = other.realization_for_part(example_part),
         support_uuid = other.support_uuid_for_part(example_part),
         continuous = other.continuous_for_part(example_part),
         indexable = other.indexable_for_part(example_part),
         property_kind = other.property_kind_for_part(example_part),
         facet_type = other.facet_type_for_part(example_part),
         facet = other.facet_for_part(example_part),
         citation_title = title,
         citation_title_match_starts_with = citation_title_match_starts_with,
         time_series_uuid = time_series_uuid,
         ignore_clashes = ignore_clashes)

   def inherit_similar_parts_for_facets_from_other_collection(self,
                                                              other,
                                                              example_part,
                                                              citation_title_match_starts_with = False,
                                                              ignore_clashes = False):
      """Adds the example part from other collection and any other parts for same property with different facets.

      arguments:
         other: another PropertyCollection object related to the same grid as this collection, from which to inherit
         example_part (string): the part name of an example member of other
         citation_title_match_starts_with (booleam, default False): if True, any citation title that starts with
            the example's citation_title after removing trailing digits, will be deemed to have passed that filter
         ignore_clashes (boolean, default False): if False, any part in other which passes the filters
            yet is already in this collection will result in an assertion error; if True, such duplicates
            are simply skipped without modifying the existing part in this collection

      note:
         at present, the citation title must match (as well as the other identifying elements) for a part to be
         inherited
      """

      assert other is not None
      assert other.part_in_collection(example_part)
      title = other.citation_title_for_part(example_part)
      if citation_title_match_starts_with:
         while title and title[-1].isdigit():
            title = title[:-1]
      self.inherit_parts_selectively_from_other_collection(
         other,
         realization = other.realization_for_part(example_part),
         support_uuid = other.support_uuid_for_part(example_part),
         continuous = other.continuous_for_part(example_part),
         indexable = other.indexable_for_part(example_part),
         property_kind = other.property_kind_for_part(example_part),
         citation_title = title,
         time_series_uuid = other.time_series_uuid_for_part(example_part),
         time_index = other.time_index_for_part(example_part),
         ignore_clashes = ignore_clashes)

   def inherit_similar_parts_for_realizations_from_other_collection(self,
                                                                    other,
                                                                    example_part,
                                                                    citation_title_match_starts_with = False,
                                                                    ignore_clashes = False):
      """Adds the example part from other collection and any other parts for same property with different realizations.

      arguments:
         other: another PropertyCollection object related to the same support as this collection, from which to inherit
         example_part (string): the part name of an example member of other
         citation_title_match_starts_with (booleam, default False): if True, any citation title that starts with
            the example's citation_title after removing trailing digits, will be deemed to have passed that filter
         ignore_clashes (boolean, default False): if False, any part in other which passes the filters
            yet is already in this collection will result in an assertion error; if True, such duplicates
            are simply skipped without modifying the existing part in this collection

      note:
         at present, the citation title must match (as well as the other identifying elements) for a part to be
         inherited
      """

      assert other is not None
      assert other.part_in_collection(example_part)
      title = other.citation_title_for_part(example_part)
      if citation_title_match_starts_with:
         while title and title[-1].isdigit():
            title = title[:-1]
      self.inherit_parts_selectively_from_other_collection(
         other,
         realization = None,
         support_uuid = other.support_uuid_for_part(example_part),
         continuous = other.continuous_for_part(example_part),
         indexable = other.indexable_for_part(example_part),
         property_kind = other.property_kind_for_part(example_part),
         facet_type = other.facet_type_for_part(example_part),
         facet = other.facet_for_part(example_part),
         citation_title = title,
         citation_title_match_starts_with = citation_title_match_starts_with,
         time_series_uuid = other.time_series_uuid_for_part(example_part),
         time_index = other.time_index_for_part(example_part),
         ignore_clashes = ignore_clashes)

   def number_of_imports(self):
      """Returns the number of property arrays in the imported list for this collection.

      returns:
         count of number of cached property arrays in the imported list for this collection (non-negative integer)

      note:
         the importation list is cleared after creation of xml trees for the imported properties, so this
         function will return zero at that point, until a new list of imports is built up
      """

      return len(self.imported_list)

   def parts(self):
      """Return list of parts in this collection.

      returns:
         list of part names (strings) being the members of this collection; there is one part per property array

      :meta common:
      """

      return list(self.dict.keys())

   def uuids(self):
      """Return list of uuids in this collection.

      returns:
         list of uuids being the members of this collection; there is one uuid per property array

      :meta common:
      """
      return [self.model.uuid_for_part(p) for p in self.dict.keys()]

   def selective_parts_list(
         self,
         realization = None,
         support = None,  # maintained for backward compatibility
         support_uuid = None,
         grid = None,  # maintained for backward compatibility
         continuous = None,
         count = None,
         indexable = None,
         property_kind = None,
         facet_type = None,
         facet = None,
         citation_title = None,
         time_series_uuid = None,
         time_index = None,
         uom = None,
         string_lookup_uuid = None,
         categorical = None):
      """Returns a list of parts filtered by those arguments which are not None.

      All arguments are optional.

      For each of these arguments: if None, then all members of collection pass this filter;
      if not None then only those members with the given value pass this filter;
      finally, the filters for all the attributes must be passed for a given member (part)
      to be included in the returned list of parts

      returns:
         list of part names (strings) of those parts which match any selection arguments which are not None

      note:
         the support and grid keyword arguments are maintained for backward compatibility;
         support_uuid takes precedence over support and both take precedence over grid

      :meta common:
      """

      if support is None:
         support = grid
      if support_uuid is None and support is not None:
         support_uuid = support.uuid

      temp_collection = selective_version_of_collection(self,
                                                        realization = realization,
                                                        support_uuid = support_uuid,
                                                        continuous = continuous,
                                                        count = count,
                                                        indexable = indexable,
                                                        property_kind = property_kind,
                                                        facet_type = facet_type,
                                                        facet = facet,
                                                        citation_title = citation_title,
                                                        time_series_uuid = time_series_uuid,
                                                        time_index = time_index,
                                                        uom = uom,
                                                        categorical = categorical,
                                                        string_lookup_uuid = string_lookup_uuid)
      parts_list = temp_collection.parts()
      return parts_list

   def singleton(
         self,
         realization = None,
         support = None,  # for backward compatibility
         support_uuid = None,
         grid = None,  # for backward compatibility
         uuid = None,
         continuous = None,
         count = None,
         indexable = None,
         property_kind = None,
         facet_type = None,
         facet = None,
         citation_title = None,
         time_series_uuid = None,
         time_index = None,
         uom = None,
         string_lookup_uuid = None,
         categorical = None):
      """Returns a single part selected by those arguments which are not None.

      For each argument: if None, then all members of collection pass this filter;
      if not None then only those members with the given value pass this filter;
      finally, the filters for all the attributes must be passed for a given member (part)
      to be selected

      returns:
         part name (string) of the part which matches all selection arguments which are not None;
         returns None if no parts match; raises an assertion error if more than one part matches

      :meta common:
      """

      if support is None:
         support = grid
      if support_uuid is None and support is not None:
         support_uuid = support.uuid

      temp_collection = selective_version_of_collection(self,
                                                        realization = realization,
                                                        support_uuid = support_uuid,
                                                        uuid = uuid,
                                                        continuous = continuous,
                                                        count = count,
                                                        indexable = indexable,
                                                        property_kind = property_kind,
                                                        facet_type = facet_type,
                                                        facet = facet,
                                                        citation_title = citation_title,
                                                        time_series_uuid = time_series_uuid,
                                                        time_index = time_index,
                                                        uom = uom,
                                                        string_lookup_uuid = string_lookup_uuid,
                                                        categorical = categorical)
      parts_list = temp_collection.parts()
      if len(parts_list) == 0:
         return None
      assert len(parts_list) == 1, 'More than one property part matches selection criteria'
      return parts_list[0]

   def single_array_ref(
         self,
         realization = None,
         support = None,  # for backward compatibility
         support_uuid = None,
         grid = None,  # for backward compatibility
         uuid = None,
         continuous = None,
         count = None,
         indexable = None,
         property_kind = None,
         facet_type = None,
         facet = None,
         citation_title = None,
         time_series_uuid = None,
         time_index = None,
         uom = None,
         string_lookup_uuid = None,
         categorical = None,
         dtype = None,
         masked = False,
         exclude_null = False):
      """Returns the array of data for a single part selected by those arguments which are not None.

      arguments:
         dtype (optional, default None): the element data type of the array to be accessed, eg 'float' or 'int';
            if None (recommended), the dtype of the returned numpy array matches that in the hdf5 dataset
         masked (boolean, optional, default False): if True, a masked array is returned instead of a simple
            numpy array; the mask is set to the inactive array attribute of the grid object
         exclude_null (boolean, default False): it True and masked is True, elements holding the null value
            will also be masked out

      Other optional arguments:
      realization, support, support_uuid, grid, continuous, count, indexable, property_kind, facet_type, facet,
      citation_title, time_series_uuid, time_index, uom, string_lookup_id, categorical:

      For each of these arguments: if None, then all members of collection pass this filter;
      if not None then only those members with the given value pass this filter;
      finally, the filters for all the attributes must be passed for a given member (part)
      to be selected

      returns:
         reference to a cached numpy array containing the actual property data for the part which matches all
         selection arguments which are not None

      notes:
         returns None if no parts match; raises an assertion error if more than one part matches;
         multiple calls will return the same cached array so calling code should copy if duplication is needed;
         support and grid arguments are for backward compatibilty: support_uuid takes precedence over support and
         both take precendence over grid

      :meta common:
      """

      if support is None:
         support = grid
      if support_uuid is None and support is not None:
         support_uuid = support.uuid

      part = self.singleton(realization = realization,
                            support_uuid = support_uuid,
                            uuid = uuid,
                            continuous = continuous,
                            count = count,
                            indexable = indexable,
                            property_kind = property_kind,
                            facet_type = facet_type,
                            facet = facet,
                            citation_title = citation_title,
                            time_series_uuid = time_series_uuid,
                            time_index = time_index,
                            uom = uom,
                            string_lookup_uuid = string_lookup_uuid,
                            categorical = categorical)
      if part is None:
         return None
      return self.cached_part_array_ref(part, dtype = dtype, masked = masked, exclude_null = exclude_null)

   def number_of_parts(self):
      """Returns the number of parts (properties) in this collection.

      returns:
         count of the number of parts (members) in this collection; there is one part per property array (non-negative integer)

      :meta common:
      """

      return len(self.dict)

   def part_in_collection(self, part):
      """Returns True if named part is member of this collection; otherwise False.

      arguments:
         part (string): part name to be tested for membership of this collection

      returns:
         boolean
      """

      return part in self.dict

   # 'private' function for accessing an element from the tuple for the part
   # the main dictionary maps from part name to a tuple of information
   # this function simply extracts one element of the tuple in a way that returns None if the part is awol
   def element_for_part(self, part, index):
      if part not in self.dict:
         return None
      return self.dict[part][index]

   # 'private' function for returning a list of unique values for an element from the tuples within the collection
   # excludes None from list
   def unique_element_list(self, index, sort_list = True):
      s = set()
      for _, t in self.dict.items():
         e = t[index]
         if e is not None:
            s = s.union({e})
      result = list(s)
      if sort_list:
         result.sort()
      return result

   def part_str(self, part, include_citation_title = True):
      """Returns a human-readable string identifying the part.

      arguments:
         part (string): the part name for which a displayable string is required
         include_citation_title (boolean, default True): if True, the citation title for the part is
            included in parenthesis at the end of the returned string; otherwise it does not appear

      returns:
         a human readable string consisting of the property kind, the facet (if present), the
         time index (if applicable), and the citation title (if requested)

      note:
         the time index is labelled 'timestep' in the returned string; however, resqml differentiates
         between the simulator timestep number and a time index into a time series; at present this
         module conflates the two

      :meta common:
      """

      text = self.property_kind_for_part(part)
      facet = self.facet_for_part(part)
      if facet:
         text += ': ' + facet
      time_index = self.time_index_for_part(part)
      if time_index is not None:
         text += '; timestep: ' + str(time_index)
      if include_citation_title:
         title = self.citation_title_for_part(part)
         if title:
            text += ' (' + title + ')'
      return text

   def part_filename(self, part):
      """Returns a string which can be used as the starting point of a filename relating to part.

      arguments:
         part (string): the part name for which a partial filename is required

      returns:
         a string suitable as the basis of a filename for the part (typically used when exporting)
      """

      text = self.property_kind_for_part(part).replace(' ', '_')
      facet = self.facet_for_part(part)
      if facet:
         text += '_' + facet  # could insert facet_type prior to this
      time_index = self.time_index_for_part(part)
      if time_index is not None:
         text += '_ts_' + str(time_index)
      return text

   def realization_for_part(self, part):
      """Returns realization number (within ensemble) that the property relates to.

      arguments:
         part (string): the part name for which the realization number (realization index) is required

      returns:
         integer or None

      :meta common:
      """

      return self.element_for_part(part, 0)

   def realization_list(self, sort_list = True):
      """Returns a list of unique realization numbers present in the collection."""

      return self.unique_element_list(0)

   def support_uuid_for_part(self, part):
      """Returns supporting representation object's uuid that the property relates to.

      arguments:
         part (string): the part name for which the related support object uuid is required

      returns:
         uuid.UUID object (or string representation thereof)
      """

      return self.element_for_part(part, 1)

   def grid_for_part(self, part):
      """Returns grid object that the property relates to.

      arguments:
         part (string): the part name for which the related grid object is required

      returns:
         grid.Grid object reference

      note:
         this method maintained for backward compatibility and kept in base PropertyClass
         for pragmatic reasons (rather than being method in GridPropertyCollection)
      """

      import resqpy.grid as grr
      import resqpy.unstructured as rug

      support_uuid = self.support_uuid_for_part(part)
      if support_uuid is None:
         return None
      if bu.matching_uuids(self.support_uuid, support_uuid):
         return self.support
      assert self.model is not None
      part = self.model.part_for_uuid(support_uuid)
      assert part is not None and self.model.type_of_part(part) in [
         'obj_IjkGridRepresentation', 'obj_UnstructuredGridRepresentation'
      ]
      return grr.any_grid(self.model, uuid = support_uuid, find_properties = False)

   def uuid_for_part(self, part):
      """Returns UUID object for the property part.

      arguments:
         part (string): the part name for which the UUID is required

      returns:
         uuid.UUID object reference; use str(uuid_for_part()) to convert to string

      :meta common:
      """

      return self.element_for_part(part, 2)

   def node_for_part(self, part):
      """Returns the xml node for the property part.

      arguments:
         part (string): the part name for which the xml node is required

      returns:
         xml Element object reference for the main xml node for the part
      """

      return self.element_for_part(part, 3)

   def extra_metadata_for_part(self, part):
      """Returns the extra_metadata dictionary for the part.

      arguments:
         part (string): the part name for which the xml node is required

      returns:
         dictionary containing extra_metadata for part
      """
      try:
         meta = self.element_for_part(part, 18)
      except Exception:
         pass
         meta = {}
      return meta

   def null_value_for_part(self, part):
      """Returns the null value for the (discrete) property part; np.NaN for continuous parts.

      arguments:
         part (string): the part name for which the null value is required

      returns:
         int or np.NaN
      """

      if self.continuous_for_part(part):
         return np.NaN
      return self.element_for_part(part, 19)

   def continuous_for_part(self, part):
      """Returns True if the property is continuous; False if it is discrete (or categorical).

      arguments:
         part (string): the part name for which the continuous versus discrete flag is required

      returns:
         True if the part is representing a continuous property, ie. the array elements are real numbers
         (float); False if the part is representing a discrete property or a categorical property, ie the
         array elements are integers (or boolean)

      note:
         resqml differentiates between discrete and categorical properties; discrete properties are
         unbounded integers where the values have numerical significance (eg. could be added together),
         whilst categorical properties have an associated dictionary mapping from a finite set of integer
         key values onto strings (eg. {1: 'background', 2: 'channel sand', 3: 'mud drape'}); however, this
         module treats categorical properties as a special case of discrete properties

      :meta common:
      """

      return self.element_for_part(part, 4)

   def all_continuous(self):
      """Returns True if all the parts are for continuous (real) properties."""

      unique_elements = self.unique_element_list(4, sort_list = False)
      if len(unique_elements) != 1:
         return False
      return unique_elements[0]

   def all_discrete(self):
      """Returns True if all the parts are for discrete or categorical (integer) properties."""

      unique_elements = self.unique_element_list(4, sort_list = False)
      if len(unique_elements) != 1:
         return False
      return not unique_elements[0]

   def count_for_part(self, part):
      """Returns the Count value for the property part; usually 1.

      arguments:
         part (string): the part name for which the count is required

      returns:
         integer reflecting the count attribute for the part (usually one); if greater than one,
         the array has an extra axis, cycling fastest, having this extent

      note:
         this mechanism allows a vector of values to be associated with a single indexable element
         in the supporting representation
      """

      return self.element_for_part(part, 5)

   def all_count_one(self):
      """Returns True if the low level Count value is 1 for all the parts in the collection."""

      unique_elements = self.unique_element_list(5, sort_list = False)
      if len(unique_elements) != 1:
         return False
      return unique_elements[0] == 1

   def indexable_for_part(self, part):
      """Returns the text of the IndexableElement for the property part; usually 'cells' for grid properties.

      arguments:
         part (string): the part name for which the indexable element is required

      returns:
         string, usually 'cells' when the supporting representation is a grid or 'nodes' when a wellbore frame

      note:
         see tail of Representations.xsd for overview of indexable elements usable for other object classes

      :meta common:
      """

      return self.element_for_part(part, 6)

   def unique_indexable_element_list(self, sort_list = False):
      """Returns a list of unique values for the IndexableElement of the property parts in the collection."""

      return self.unique_element_list(6, sort_list = sort_list)

   def property_kind_for_part(self, part):
      """Returns the resqml property kind for the property part.

      arguments:
         part (string): the part name for which the property kind is required

      returns:
         standard resqml property kind or local property kind for this part, as a string, eg. 'porosity'

      notes:
         see attributes of this module named supported_property_kind_list and supported_local_property_kind_list
         for the property kinds which this module can relate to simulator keywords (Nexus); however, other property
         kinds should be handled okay in a generic way;
         for bespoke (local) property kinds, this is the property kind title as stored in the xml reference node

      :meta common:
      """

      return self.element_for_part(part, 7)

   def property_kind_list(self, sort_list = True):
      """Returns a list of unique property kinds found amongst the parts of the collection."""

      return self.unique_element_list(7, sort_list = sort_list)

   def local_property_kind_uuid(self, part):
      """Returns the uuid of the bespoke (local) property kind for this part, or None for a standard property kind."""

      return self.element_for_part(part, 17)

   def facet_type_for_part(self, part):
      """If relevant, returns the resqml Facet Facet for the property part, eg. 'direction'; otherwise None.

      arguments:
         part (string): the part name for which the facet type is required

      returns:
         standard resqml facet type for this part (string), or None

      notes:
         resqml refers to Facet Facet and Facet Value; the equivalents in this module are facet_type and facet;
         the resqml standard allows a property to have any number of facets; this module currently limits a
         property to having at most one facet; the facet_type and facet should be either both None or both not None

      :meta common:
      """

      return self.element_for_part(part, 8)

   def facet_type_list(self, sort_list = True):
      """Returns a list of unique facet types found amongst the parts of the collection."""

      return self.unique_element_list(8, sort_list = sort_list)

   def facet_for_part(self, part):
      """If relevant, returns the resqml Facet Value for the property part, eg. 'I'; otherwise None.

      arguments:
         part (string): the part name for which the facet value is required

      returns:
         facet value for this part (string), for the facet type returned by the facet_type_for_part() function,
         or None

      see notes for facet_type_for_part()

      :meta common:
      """

      return self.element_for_part(part, 9)

   def facet_list(self, sort_list = True):
      """Returns a list of unique facet values found amongst the parts of the collection."""

      return self.unique_element_list(9, sort_list = sort_list)

   def citation_title_for_part(self, part):
      """Returns the citation title for the property part.

      arguments:
         part (string): the part name for which the citation title is required

      returns:
         citation title (string) for this part

      note:
         for simulation grid properties, the citation title is often a property keyword specific to a simulator

      :meta common:
      """

      return self.element_for_part(part, 10)

   def title_for_part(self, part):
      """Synonymous with citation_title_for_part()."""

      return self.citation_title_for_part(part)

   def titles(self):
      """Returns a list of citation titles for the parts in the collection."""

      return [self.citation_title_for_part(p) for p in self.parts()]

   def time_series_uuid_for_part(self, part):
      """If the property has an associated time series (is not static), returns the uuid for the time series.

      arguments:
         part (string): the part name for which the time series uuid is required

      returns:
         time series uuid (uuid.UUID) for this part
      """

      return self.element_for_part(part, 11)

   def time_series_uuid_list(self, sort_list = True):
      """Returns a list of unique time series uuids found amongst the parts of the collection."""

      return self.unique_element_list(11, sort_list = sort_list)

   def time_index_for_part(self, part):
      """If the property has an associated time series (is not static), returns the time index within the time series.

      arguments:
         part (string): the part name for which the time index is required

      returns:
         time index (integer) for this part

      :meta common:
      """

      return self.element_for_part(part, 12)

   def time_index_list(self, sort_list = True):
      """Returns a list of unique time indices found amongst the parts of the collection."""

      return self.unique_element_list(12, sort_list = sort_list)

   def minimum_value_for_part(self, part):
      """Returns the minimum value for the property part, as stored in the xml.

      arguments:
         part (string): the part name for which the minimum value is required

      returns:
         minimum value (as string or float or int!) for this part

      note:
         this method merely returns the minimum value recorded in the xml for the property, it does not check
         the array data

      :meta common:
      """

      return self.element_for_part(part, 13)

   def maximum_value_for_part(self, part):
      """Returns the maximum value for the property part, as stored in the xml.

      arguments:
         part (string): the part name for which the maximum value is required

      returns:
         maximum value (as string or float ir int!) for this part

      note:
         this method merely returns the maximum value recorded in the xml for the property, it does not check
         the array data

      :meta common:
      """

      return self.element_for_part(part, 14)

   def patch_min_max_for_part(self, part, minimum = None, maximum = None, model = None):
      """Updates the minimum and/ox maximum values stored in the metadata, optionally updating xml tree too.

      arguments:
         part (str): the part name of the property
         minimum (float or int, optional): the new minimum value to be set in the metadata (unchanged if None)
         maximum (float or int, optional): the new maximum value to be set in the metadata (unchanged if None)
         model (model.Model, optional): if present and containing xml for the part, that xml is also patched

      notes:
         this method is rarely needed: only if a property array is being re-populated after being initialised
         with temporary values; the xml tree for the part in the model will only be updated where the minimum
         and/or maximum nodes already exist in the tree
      """

      info = list(self.dict[part])
      if minimum is not None:
         info[13] = minimum
      if maximum is not None:
         info[14] = maximum
      self.dict[part] = tuple(info)
      if model is not None:
         p_root = model.root_for_part(part)
         if p_root is not None:
            if minimum is not None:
               min_node = rqet.find_tag(p_root, 'MinimumValue')
               if min_node is not None:
                  min_node.text = str(minimum)
                  model.set_modified()
            if maximum is not None:
               max_node = rqet.find_tag(p_root, 'MaximumValue')
               if max_node is not None:
                  max_node.text = str(maximum)
                  model.set_modified()

   def uom_for_part(self, part):
      """Returns the resqml units of measure for the property part.

      arguments:
         part (string): the part name for which the units of measure is required

      returns:
         resqml units of measure (string) for this part

      :meta common:
      """

      # NB: this field is not set correctly in data sets generated by DGI
      return self.element_for_part(part, 15)

   def uom_list(self, sort_list = True):
      """Returns a list of unique units of measure found amongst the parts of the collection."""

      return self.unique_element_list(15, sort_list = sort_list)

   def string_lookup_uuid_for_part(self, part):
      """If the property has an associated string lookup (is categorical), returns the uuid for the string table lookup.

      arguments:
         part (string): the part name for which the string lookup uuid is required

      returns:
         string lookup uuid (uuid.UUID) for this part
      """

      return self.element_for_part(part, 16)

   def string_lookup_for_part(self, part):
      """Returns a StringLookup object for the part, if it has a string lookup uuid, otherwise None."""

      sl_uuid = self.string_lookup_uuid_for_part(part)
      if sl_uuid is None:
         return None
      sl_root = self.model.root_for_uuid(sl_uuid)
      assert sl_root is not None, 'string table lookup referenced by property is not present in model'
      return StringLookup(self.model, sl_root)

   def string_lookup_uuid_list(self, sort_list = True):
      """Returns a list of unique string lookup uuids found amongst the parts of the collection."""

      return self.unique_element_list(16, sort_list = sort_list)

   def part_is_categorical(self, part):
      """Returns True if the property is categorical (not conintuous and has an associated string lookup).

      :meta common:
      """

      return not self.continuous_for_part(part) and self.string_lookup_uuid_for_part(part) is not None

   def constant_value_for_part(self, part):
      """Returns the value (float or int) of a constant array part, or None for an hdf5 array.

      :meta common:
      """

      return self.element_for_part(part, 20)

   def override_min_max(self, part, min_value, max_value):
      """Sets the minimum and maximum values in the metadata for the part.

      arguments:
         part (string): the part name for which the minimum and maximum values are to be set
         min_value (float or int or string): the minimum value to be stored in the metadata
         max_value (float or int or string): the maximum value to be stored in the metadata

      note:
         this function is typically called if the existing min & max metadata is missing or
         distrusted; the min and max values passed in are typically the result of numpy
         min and max function calls (possibly skipping NaNs) on the property array or
         a version of it masked for inactive cells
      """

      if part not in self.dict:
         return
      property_part = list(self.dict[part])
      property_part[13] = min_value
      property_part[14] = max_value
      self.dict[part] = tuple(property_part)

   def establish_time_set_kind(self):
      """Re-evaulates the time set kind attribute based on all properties having same time index in the same time series."""

      self.time_set_kind_attr = 'single time'
      #  note: other option of 'equivalent times' not catered for in this code
      common_time_index = None
      common_time_series_uuid = None
      for part in self.parts():
         part_time_index = self.time_index_for_part(part)
         if part_time_index is None:
            self.time_set_kind_attr = 'not a time set'
            break
         if common_time_index is None:
            common_time_index = part_time_index
         elif common_time_index != part_time_index:
            self.time_set_kind_attr = 'not a time set'
            break
         part_ts_uuid = self.time_series_uuid_for_part(part)
         if part_ts_uuid is None:
            self.time_set_kind_attr = 'not a time set'
            break
         if common_time_series_uuid is None:
            common_time_series_uuid = part_ts_uuid
         elif not bu.matching_uuids(common_time_series_uuid, part_ts_uuid):
            self.time_set_kind_attr = 'not a time set'
            break
      return self.time_set_kind_attr

   def time_set_kind(self):
      """Returns the time set kind attribute based on all properties having same time index in the same time series."""

      if self.time_set_kind_attr is None:
         self.establish_time_set_kind()
      return self.time_set_kind_attr

   def establish_has_single_property_kind(self):
      """Re-evaluates the has single property kind attribute depending on whether all properties are of the same kind."""

      self.has_single_property_kind_flag = True
      common_property_kind = None
      for part in self.parts():
         part_kind = self.property_kind_for_part(part)
         if common_property_kind is None:
            common_property_kind = part_kind
         elif part_kind != common_property_kind:
            self.has_single_property_kind_flag = False
            break
      return self.has_single_property_kind_flag

   def has_single_property_kind(self):
      """Returns the has single property kind flag depending on whether all properties are of the same kind."""

      if self.has_single_property_kind_flag is None:
         self.establish_has_single_property_kind()
      return self.has_single_property_kind_flag

   def establish_has_single_indexable_element(self):
      """Re-evaluates the has single indexable element attribute depending on whether all properties have the same."""

      self.has_single_indexable_element_flag = True
      common_ie = None
      for part in self.parts():
         ie = self.indexable_for_part(part)
         if common_ie is None:
            common_ie = ie
         elif ie != common_ie:
            self.has_single_indexable_element_flag = False
            break
      return self.has_single_indexable_element_flag

   def has_single_indexable_element(self):
      """Returns the has single indexable element flag depending on whether all properties have the same."""

      if self.has_single_indexable_element_flag is None:
         self.establish_has_single_indexable_element()
      return self.has_single_indexable_element_flag

   def establish_has_multiple_realizations(self):
      """Re-evaluates the has multiple realizations attribute based on whether properties belong to more than one realization."""

      self.has_multiple_realizations_flag = False
      common_realization = None
      for part in self.parts():
         part_realization = self.realization_for_part(part)
         if part_realization is None:
            continue
         if common_realization is None:
            common_realization = part_realization
            continue
         if part_realization != common_realization:
            self.has_multiple_realizations_flag = True
            self.realization = None  # override single realization number applicable to whole collection
            break
      if not self.has_multiple_realizations_flag and common_realization is not None:
         self.realization = common_realization
      return self.has_multiple_realizations_flag

   def has_multiple_realizations(self):
      """Returns the has multiple realizations flag based on whether properties belong to more than one realization.

      :meta common:
      """

      if self.has_multiple_realizations_flag is None:
         self.establish_has_multiple_realizations()
      return self.has_multiple_realizations_flag

   def establish_has_single_uom(self):
      """Re-evaluates the has single uom attribute depending on whether all properties have the same units of measure."""

      self.has_single_uom_flag = True
      common_uom = None
      for part in self.parts():
         part_uom = self.uom_for_part(part)
         if common_uom is None:
            common_uom = part_uom
         elif part_uom != common_uom:
            self.has_single_uom_flag = False
            break
      if common_uom is None:
         self.has_single_uom_flag = True  # all uoms are None (probably discrete properties)
      return self.has_single_uom_flag

   def has_single_uom(self):
      """Returns the has single uom flag depending on whether all properties have the same units of measure."""

      if self.has_single_uom_flag is None:
         self.establish_has_single_uom()
      return self.has_single_uom_flag

   def assign_realization_numbers(self):
      """Assigns a distinct realization number to each property, after checking for compatibility.

      note:
         this method does not modify realization information in any established xml; it is intended primarily as
         a convenience to allow realization based processing of any collection of compatible properties
      """

      assert self.has_single_property_kind(), 'attempt to assign realization numbers to properties of differing kinds'
      assert self.has_single_indexable_element(
      ), 'attempt to assign realizations to properties of differing indexable elements'
      assert self.has_single_uom(
      ), 'attempt to assign realization numbers to properties with differing units of measure'

      new_dict = {}
      realization = 0
      for key, entry in self.dict.items():
         entry_list = list(entry)
         entry_list[0] = realization
         new_dict[key] = tuple(entry_list)
         realization += 1
      self.dict = new_dict
      self.has_multiple_realizations_flag = (realization > 1)

   def masked_array(self, simple_array, exclude_inactive = True, exclude_value = None):
      """Returns a masked version of simple_array, using inactive mask associated with support for this property collection.

      arguments:
         simple_array (numpy array): an unmasked numpy array with the same shape as property arrays for the support
            (and indexable element) associated with this collection
         exclude_inactive (boolean, default True): elements which are flagged as inactive in the supporting representation
            are masked out if this argument is True
         exclude_value (float or int, optional): if present, elements which match this value are masked out; if not None
            then usually set to np.NaN for continuous data or null_value_for_part() for discrete data

      returns:
         a masked version of the array, with the mask set to exclude cells which are inactive in the support

      notes:
         when requesting a reference to a cached copy of a property array (using other functions), a masked argument
         can be used to apply the inactive mask; this function is therefore rarely needed by calling code (it is used
         internally by this module); the simple_array need not be part of this collection
      """

      mask = None
      if (exclude_inactive and self.support is not None and hasattr(self.support, 'inactive') and
          self.support.inactive is not None and self.support.inactive.shape == simple_array.shape):
         mask = self.support.inactive
      if exclude_value:
         null_mask = (simple_array == exclude_value)
         if mask is None:
            mask = null_mask
         else:
            mask = np.logical_or(mask, null_mask)
      if mask is None:
         mask = ma.nomask
      return ma.masked_array(simple_array, mask = mask)

   def h5_key_pair_for_part(self, part):
      """Return hdf5 key pair (ext uuid, internal path) for the part."""

      model = self.model
      part_node = self.node_for_part(part)
      if part_node is None:
         return None
      patch_list = rqet.list_of_tag(part_node, 'PatchOfValues')
      assert len(patch_list) == 1  # todo: handle more than one patch of values
      first_values_node = rqet.find_tag(patch_list[0], 'Values')
      if first_values_node is None:
         return None  # could treat as fatal error
      return model.h5_uuid_and_path_for_node(first_values_node)  # note: Values node within Values node for properties

   def cached_part_array_ref(self, part, dtype = None, masked = False, exclude_null = False):
      """Returns a numpy array containing the data for the property part; the array is cached in this collection.

      arguments:
         part (string): the part name for which the array reference is required
         dtype (optional, default None): the element data type of the array to be accessed, eg 'float' or 'int';
            if None (recommended), the dtype of the returned numpy array matches that in the hdf5 dataset
         masked (boolean, default False): if True, a masked array is returned instead of a simple numpy array;
            the mask is set to the inactive array attribute of the support object if present
         exclude_null (boolean, default False): if True, and masked is also True, then elements of the array
            holding the null value will also be masked out

      returns:
         reference to a cached numpy array containing the actual property data; multiple calls will return
         the same cached array so calling code should copy if duplication is needed

      notes:
         this function is the usual way to get at the actual property array; at present, the funtion only works
         if the entire array is stored as a single patch in the hdf5 file (resqml allows multiple patches per
         array); the masked functionality can be used to apply a common mask, stored in the supporting
         representation object with the attribute name 'inactive', to multiple properties (this will only work
         if the indexable element is set to the typical value for the class of supporting representation, eg.
         'cells' for grid objects); if exclude_null is set True then null value elements will also be masked out
         (as long as masked is True); however, it is recommended simply to use np.NaN values in floating point
         property arrays if the commonality is not needed

      :meta common:
      """

      model = self.model
      cached_array_name = _cache_name(part)
      if cached_array_name is None:
         return None

      if not hasattr(self, cached_array_name):

         const_value = self.constant_value_for_part(part)
         if const_value is None:
            part_node = self.node_for_part(part)
            if part_node is None:
               return None
            patch_list = rqet.list_of_tag(part_node, 'PatchOfValues')
            assert len(patch_list) == 1  # todo: handle more than one patch of values
            first_values_node = rqet.find_tag(patch_list[0], 'Values')
            if first_values_node is None:
               return None  # could treat as fatal error
            if dtype is None:
               array_type = rqet.node_type(first_values_node)
               assert array_type is not None
               if array_type == 'DoubleHdf5Array':
                  dtype = 'float'
               elif array_type == 'IntegerHdf5Array':
                  dtype = 'int'
               elif array_type == 'BooleanHdf5Array':
                  dtype = 'bool'
               else:
                  raise ValueError('array type not catered for: ' + str(array_type))
            h5_key_pair = model.h5_uuid_and_path_for_node(
               first_values_node)  # note: Values node within Values node for properties
            if h5_key_pair is None:
               return None
            model.h5_array_element(h5_key_pair,
                                   index = None,
                                   cache_array = True,
                                   object = self,
                                   array_attribute = cached_array_name,
                                   dtype = dtype)
         else:
            assert self.support is not None
            shape = self.supporting_shape()
            assert shape is not None
            a = np.full(shape, const_value, dtype = float if self.continuous_for_part(part) else int)
            setattr(self, cached_array_name, a)
         if not hasattr(self, cached_array_name):
            return None

      if masked:
         exclude_value = self.null_value_for_part(part) if exclude_null else None
         return self.masked_array(self.__dict__[cached_array_name], exclude_value = exclude_value)
      else:
         return self.__dict__[cached_array_name]

   def h5_slice(self, part, slice_tuple):
      """Returns a subset of the array for part, without loading the whole array.

      arguments:
         part (string): the part name for which the array slice is required
         slice_tuple (tuple of slice objects): each element should be constructed using the python built-in
            function slice()

      returns:
         numpy array that is a hyper-slice of the hdf5 array, with the same ndim as the source hdf5 array

      note:
         this method always fetches from the hdf5 file and does not attempt local caching; the whole array
         is not loaded; all axes continue to exist in the returned array, even where the sliced extent of
         an axis is 1
      """

      h5_key_pair = self.h5_key_pair_for_part(part)
      if h5_key_pair is None:
         return None
      return self.model.h5_array_slice(h5_key_pair, slice_tuple)

   def h5_overwrite_slice(self, part, slice_tuple, array_slice, update_cache = True):
      """Overwrites a subset of the array for part, in the hdf5 file.

      arguments:
         part (string): the part name for which the array slice is to be overwritten
         slice_tuple (tuple of slice objects): each element should be constructed using the python built-in
            function slice()
         array_slice (numpy array of shape to match slice_tuple): the data to be written
         update_cache (boolean, default True): if True and the part is currently cached within this
            PropertyCollection, then the cached array is also updated; if False, the part is uncached

      notes:
         this method naively writes the slice to hdf5 without using mpi to look after parallel writes;
         if a cached copy of the array is updated, this is in an unmasked form; if calling code has a
         reterence to a masked version of the array then the mask will not be updated by this method;
         if the part is not currently cached, this method will not cause it to become cached,
         regardless of the update_cache argument
      """

      h5_key_pair = self.h5_key_pair_for_part(part)
      assert h5_key_pair is not None
      self.model.h5_overwrite_array_slice(h5_key_pair, slice_tuple, array_slice)
      cached_array_name = _cache_name(part)
      if cached_array_name is None:
         return
      if hasattr(self, cached_array_name):
         if update_cache:
            self.__dict__[cached_array_name][slice_tuple] = array_slice
         else:
            delattr(self, cached_array_name)

   def shape_and_type_of_part(self, part):
      """Returns shape tuple and element type of cached or hdf5 array for part."""

      model = self.model
      cached_array_name = _cache_name(part)
      if cached_array_name is None:
         return None, None

      if hasattr(self, cached_array_name):
         return tuple(self.__dict__[cached_array_name].shape), self.__dict__[cached_array_name].dtype

      part_node = self.node_for_part(part)
      if part_node is None:
         return None, None

      if self.constant_value_for_part(part) is not None:
         assert self.support is not None
         shape = self.supporting_shape()
         assert shape is not None
         return shape, (float if self.continuous_for_part(part) else int)

      patch_list = rqet.list_of_tag(part_node, 'PatchOfValues')
      assert len(patch_list) == 1  # todo: handle more than one patch of values
      h5_key_pair = model.h5_uuid_and_path_for_node(rqet.find_tag(patch_list[0], 'Values'))
      if h5_key_pair is None:
         return None, None
      return model.h5_array_shape_and_type(h5_key_pair)

   def facets_array_ref(self, use_32_bit = False, indexable_element = None):  # todo: add masked argument
      """Returns a +1D array of all parts with first axis being over facet values; Use facet_list() for lookup.

      arguments:
         use_32_bit (boolean, default False): if True, the resulting numpy array will use a 32 bit dtype; if False, 64 bit
         indexable_element (string, optional): the indexable element for the properties in the collection; if None, will
            be determined from the data

      returns:
         numpy array containing all the data in the collection, the first axis being over facet values and the rest of
         the axes matching the shape of the individual property arrays

      notes:
         the property collection should be constructed so as to hold a suitably coherent set of properties before
         calling this method;
         the facet_list() method will return the facet values that correspond to slices in the first axis of the
         resulting array
      """

      assert self.support is not None, 'attempt to build facets array for property collection without supporting representation'
      assert self.number_of_parts() > 0, 'attempt to build facets array for empty property collection'
      assert self.has_single_property_kind(
      ), 'attempt to build facets array for collection containing multiple property kinds'
      assert self.has_single_indexable_element(
      ), 'attempt to build facets array for collection containing a variety of indexable elements'
      assert self.has_single_uom(), 'attempt to build facets array for collection containing multiple units of measure'

      #  could check that facet_type_list() has exactly one value
      facet_list = self.facet_list(sort_list = True)
      facet_count = len(facet_list)
      assert facet_count > 0, 'no facets found in property collection'
      assert self.number_of_parts() == facet_count, 'collection covers more than facet variability'

      continuous = self.all_continuous()
      if not continuous:
         assert self.all_discrete(), 'mixture of continuous and discrete properties in collection'

      if indexable_element is None:
         indexable_element = self.indexable_for_part(self.parts()[0])

      dtype = dtype_flavour(continuous, use_32_bit)
      shape_list = self.supporting_shape(indexable_element = indexable_element)
      shape_list.insert(0, facet_count)

      a = np.zeros(shape_list, dtype = dtype)

      for part in self.parts():
         facet_index = facet_list.index(self.facet_for_part(part))
         pa = self.cached_part_array_ref(part, dtype = dtype)
         a[facet_index] = pa
         self.uncache_part_array(part)

      return a

   def realizations_array_ref(self,
                              use_32_bit = False,
                              fill_missing = True,
                              fill_value = None,
                              indexable_element = None):
      """Returns a +1D array of all parts with first axis being over realizations.

      arguments:
         use_32_bit (boolean, default False): if True, the resulting numpy array will use a 32 bit dtype; if False, 64 bit
         fill_missing (boolean, default True): if True, the first axis of the resulting numpy array will range from 0 to
            the maximum realization number present and slices for any missing realizations will be filled with fill_value;
            if False, the extent of the first axis will only cpver the number pf realizations actually present (see also notes)
         fill_value (int or float, optional): the value to use for missing realization slices; if None, will default to
            np.NaN if data is continuous, -1 otherwise; irrelevant if fill_missing is False
         indexable_element (string, optional): the indexable element for the properties in the collection; if None, will
            be determined from the data

      returns:
         numpy array containing all the data in the collection, the first axis being over realizations and the rest of
         the axes matching the shape of the individual property arrays

      notes:
         the property collection should be constructed so as to hold a suitably coherent set of properties before
         calling this method;
         if fill_missing is False, the realization axis indices range from zero to the number of realizations present;
         if True, the realization axis indices range from zero to the maximum realization number and slices for missing
         realizations will be filled with the fill_value

      :meta common:
      """

      assert self.support is not None, 'attempt to build realizations array for property collection without supporting representation'
      assert self.number_of_parts() > 0, 'attempt to build realizations array for empty property collection'
      assert self.has_single_property_kind(
      ), 'attempt to build realizations array for collection with multiple property kinds'
      assert self.has_single_indexable_element(
      ), 'attempt to build realizations array for collection containing a variety of indexable elements'
      assert self.has_single_uom(
      ), 'attempt to build realizations array for collection containing multiple units of measure'

      r_list = self.realization_list(sort_list = True)
      assert self.number_of_parts() == len(r_list), 'collection covers more than realizations of a single property'

      continuous = self.all_continuous()
      if not continuous:
         assert self.all_discrete(), 'mixture of continuous and discrete properties in collection'

      if fill_value is None:
         fill_value = np.NaN if continuous else -1

      if indexable_element is None:
         indexable_element = self.indexable_for_part(self.parts()[0])

      if fill_missing:
         r_extent = r_list[-1] + 1
      else:
         r_extent = len(r_list)

      dtype = dtype_flavour(continuous, use_32_bit)
      shape_list = self.supporting_shape(indexable_element = indexable_element)
      shape_list.insert(0, r_extent)

      a = np.full(shape_list, fill_value, dtype = dtype)

      if fill_missing:
         for part in self.parts():
            realization = self.realization_for_part(part)
            assert realization is not None and 0 <= realization < r_extent, 'realization missing (or out of range?)'
            pa = self.cached_part_array_ref(part, dtype = dtype)
            a[realization] = pa
            self.uncache_part_array(part)
      else:
         for index in range(len(r_list)):
            realization = r_list[index]
            part = self.singleton(realization = realization)
            pa = self.cached_part_array_ref(part, dtype = dtype)
            a[index] = pa
            self.uncache_part_array(part)

      return a

   def time_series_array_ref(self,
                             use_32_bit = False,
                             fill_missing = True,
                             fill_value = None,
                             indexable_element = None):
      """Returns a +1D array of all parts with first axis being over time indices.

      arguments:
         use_32_bit (boolean, default False): if True, the resulting numpy array will use a 32 bit dtype; if False, 64 bit
         fill_missing (boolean, default True): if True, the first axis of the resulting numpy array will range from 0 to
            the maximum time index present and slices for any missing indices will be filled with fill_value; if False,
            the extent of the first axis will only cpver the number pf time indices actually present (see also notes)
         fill_value (int or float, optional): the value to use for missing time index slices; if None, will default to
            np.NaN if data is continuous, -1 otherwise; irrelevant if fill_missing is False
         indexable_element (string, optional): the indexable element for the properties in the collection; if None, will
            be determined from the data

      returns:
         numpy array containing all the data in the collection, the first axis being over time indices and the rest of
         the axes matching the shape of the individual property arrays

      notes:
         the property collection should be constructed so as to hold a suitably coherent set of properties before
         calling this method;
         if fill_missing is False, the time axis indices range from zero to the number of time indices present,
         with the list of tine index values available by calling the method time_index_list(sort_list = True);
         if fill_missing is True, the time axis indices range from zero to the maximum time index and slices for
         missing time indices will be filled with the fill_value

      :meta common:
      """

      assert self.support is not None, 'attempt to build time series array for property collection without supporting representation'
      assert self.number_of_parts() > 0, 'attempt to build time series array for empty property collection'
      assert self.has_single_property_kind(
      ), 'attempt to build time series array for collection with multiple property kinds'
      assert self.has_single_indexable_element(
      ), 'attempt to build time series array for collection containing a variety of indexable elements'
      assert self.has_single_uom(
      ), 'attempt to build time series array for collection containing multiple units of measure'

      ti_list = self.time_index_list(sort_list = True)
      assert self.number_of_parts() == len(ti_list), 'collection covers more than time indices of a single property'

      continuous = self.all_continuous()
      if not continuous:
         assert self.all_discrete(), 'mixture of continuous and discrete properties in collection'

      if fill_value is None:
         fill_value = np.NaN if continuous else -1

      if indexable_element is None:
         indexable_element = self.indexable_for_part(self.parts()[0])

      if fill_missing:
         ti_extent = ti_list[-1] + 1
      else:
         ti_extent = len(ti_list)

      dtype = dtype_flavour(continuous, use_32_bit)
      shape_list = self.supporting_shape(indexable_element = indexable_element)
      shape_list.insert(0, ti_extent)

      a = np.full(shape_list, fill_value, dtype = dtype)

      if fill_missing:
         for part in self.parts():
            time_index = self.time_index_for_part(part)
            assert time_index is not None and 0 <= time_index < ti_extent, 'time index missing (or out of range?)'
            pa = self.cached_part_array_ref(part, dtype = dtype)
            a[time_index] = pa
            self.uncache_part_array(part)
      else:
         for index in range(len(ti_list)):
            time_index = ti_list[index]
            part = self.singleton(time_index = time_index)
            pa = self.cached_part_array_ref(part, dtype = dtype)
            a[index] = pa
            self.uncache_part_array(part)

      return a

   def combobulated_face_array(self, resqml_a):
      """Returns a logically ordered copy of RESQML faces-per-cell property array resqml_a.

      argument:
         resqml_a (numpy array of shape (..., 6): a RESQML property array with indexable element faces per cell

      returns:
         numpy array of shape (..., 3, 2) where the 3 covers K,J,I and the 2 the -/+ face polarities being a resqpy logically
            arranged copy of resqml_a

      notes:
         this method is for properties of IJK grids only;
         RESQML documentation is not entirely clear about the required ordering of -I, +I, -J, +J faces;
         current implementation assumes count = 1 for the property
      """

      assert resqml_a.shape[-1] == 6

      resqpy_a_shape = tuple(list(resqml_a.shape[:-1]) + [3, 2])
      resqpy_a = np.empty(resqpy_a_shape, dtype = resqml_a.dtype)

      for axis in range(3):
         for polarity in range(2):
            resqpy_a[..., axis, polarity] = resqml_a[..., self.face_index_map[axis, polarity]]

      return resqpy_a

   def discombobulated_face_array(self, resqpy_a):
      """Returns a RESQML format copy of logical face property array a, re-ordered and reshaped regarding the six facial directions.

      argument:
         resqpy_a (numpy array of shape (..., 3, 2)): the penultimate array axis represents K,J,I and the final axis is -/+ face
            polarity; the resqpy logically arranged property array to be converted to illogical RESQML ordering and shape

      returns:
         numpy array of shape (..., 6) being a copy of resqpy_a with slices reordered before collapsing the last 2 axes into 1;
            ready to be stored as a RESQML property array with indexable element faces per cell

      notes:
         this method is for properties of IJK grids only;
         RESQML documentation is not entirely clear about the required ordering of -I, +I, -J, +J faces;
         current implementation assumes count = 1 for the property
      """

      assert resqpy_a.ndim >= 2 and resqpy_a.shape[-2] == 3 and resqpy_a.shape[-1] == 2

      resqml_a_shape = tuple(list(resqpy_a.shape[:-2]).append(6))
      resqml_a = np.empty(resqml_a_shape, dtype = resqpy_a.dtype)

      for face in range(6):
         resqml_a[..., face] = resqpy_a[..., self.face_index_inverse_map[face]]

      return resqml_a

   def cached_normalized_part_array_ref(self,
                                        part,
                                        masked = False,
                                        use_logarithm = False,
                                        discrete_cycle = None,
                                        trust_min_max = False):
      """DEPRECATED: replaced with normalized_part_array() method."""

      return self.normalized_part_array(part,
                                        masked = masked,
                                        use_logarithm = use_logarithm,
                                        discrete_cycle = discrete_cycle,
                                        trust_min_max = trust_min_max)

   def normalized_part_array(self,
                             part,
                             masked = False,
                             use_logarithm = False,
                             discrete_cycle = None,
                             trust_min_max = False,
                             fix_zero_at = None):
      """Returns a triplet of: a numpy float array containing the data normalized between 0.0 and 1.0, the min value, the max value.

      arguments:
         part (string): the part name for which the normalized array reference is required
         masked (boolean, optional, default False): if True, the masked version of the property array is used to
            determine the range of values to map onto the normalized range of 0 to 1 (the mask removes inactive cells
            from having any impact); if False, the values of inactive cells are included in the operation; the returned
            normalized array is masked or not depending on this argument
         use_logarithm (boolean, optional, default False): if False, the property values are linearly mapped to the
            normalized range; if True, the logarithm (base 10) of the property values are mapped to the normalized range
         discrete_cycle (positive integer, optional, default None): if a value is supplied and the property array
            contains integer data (discrete or categorical), the modulus of the property values are calculated
            against this value before conversion to floating point and mapping to the normalized range
         trust_min_max (boolean, optional, default False): if True, the minimum and maximum values from the property's
            metadata is used as the range of the property values; if False, the values are determined using numpy
            min and max operations
         fix_zero_at (float, optional): if present, a value between 0.0 and 1.0 (typically 0.0 or 0.5) to pin zero at

      returns:
         (normalized_array, min_value, max_value) where:
         normalized_array is a numpy array of floats, masked or unmasked depending on the masked argument, with values
         ranging between 0 and 1; in the case of a masked array the values for excluded cells are meaningless and may
         lie outside the range 0 to 1
         min_value and max_value: the property values that have been mapped to 0 and 1 respectively

      notes:
         this function is typically used to map property values onto the range required for colouring in;
         in case of failure, (None, None, None) is returned;
         if use_logarithm is True, the min_value and max_value returned are the log10 values, not the original
         property values;
         also, if use logarithm is True and the minimum property value is not greater than zero, then values less than
         0.0001 are set to 0.0001, prior to taking the logarithm;
         fix_zero_at is mutually incompatible with use_logarithm; to force the normalised data to have a true zero,
         set fix_zero_at to 0.0; for divergent data fixing zero at 0.5 will often be appropriate;
         fixing zero at 0.0 or 1.0 may result in normalised values being clipped;
         for floating point data, NaN values will be handled okay; if all data are NaN, (None, NaN, NaN) is returned;
         for integer data, null values are not currently supported (though the RESQML metadata can hold a null value);
         the masked argument is most applicable to properties for grid objects; note that NaN values are excluded when
         determining the min and max regardless of the value of the masked argument
      """

      assert fix_zero_at is None or not use_logarithm

      p_array = self.cached_part_array_ref(part, masked = masked)

      if p_array is None:
         return None, None, None
      min_value = max_value = None
      if trust_min_max:
         min_value = self.minimum_value_for_part(part)
         max_value = self.maximum_value_for_part(part)
      if min_value is None or max_value is None:
         min_value = np.nanmin(p_array)
         if masked and min_value is ma.masked:
            min_value = None
         max_value = np.nanmax(p_array)
         if masked and max_value is ma.masked:
            max_value = None
         self.override_min_max(part, min_value, max_value)  # NB: this does not modify xml
      if min_value is None or max_value is None:
         return None, min_value, max_value
      if 'int' in str(
            p_array.dtype) and discrete_cycle is not None:  # could use continuous flag in metadata instead of dtype
         p_array = p_array % discrete_cycle
         min_value = 0
         max_value = discrete_cycle - 1
      elif str(p_array.dtype).startswith('bool'):
         min_value = int(min_value)
         max_value = int(max_value)
      min_value = float(min_value)  # will return np.ma.masked if all values are masked out
      if masked and min_value is ma.masked:
         min_value = np.nan
      max_value = float(max_value)
      if masked and max_value is ma.masked:
         max_value = np.nan
      if min_value == np.nan or max_value == np.nan:
         return None, min_value, max_value
      if max_value < min_value:
         return None, min_value, max_value
      n_prop = p_array.astype(float)
      if use_logarithm:
         if min_value <= 0.0:
            n_prop[:] = np.where(n_prop < 0.0001, 0.0001, n_prop)
         n_prop = np.log10(n_prop)
         min_value = np.nanmin(n_prop)
         max_value = np.nanmax(n_prop)
         if masked:
            if min_value is ma.masked:
               min_value = np.nan
            if max_value is ma.masked:
               max_value = np.nan
         if min_value == np.nan or max_value == np.nan:
            return None, min_value, max_value
      if fix_zero_at is not None:
         if fix_zero_at <= 0.0:
            if min_value < 0.0:
               n_prop[:] = np.where(n_prop < 0.0, 0.0, n_prop)
            min_value = 0.0
         elif fix_zero_at >= 1.0:
            if max_value > 0.0:
               n_prop[:] = np.where(n_prop > 0.0, 0.0, n_prop)
            max_value = 0.0
         else:
            upper_scaling = max_value / (1.0 - fix_zero_at)
            lower_scaling = -min_value / fix_zero_at
            if upper_scaling >= lower_scaling:
               min_value = -upper_scaling * fix_zero_at
               n_prop[:] = np.where(n_prop < min_value, min_value, n_prop)
            else:
               max_value = lower_scaling * (1.0 - fix_zero_at)
               n_prop[:] = np.where(n_prop > max_value, max_value, n_prop)
      if max_value == min_value:
         n_prop[:] = 0.5
         return n_prop, min_value, max_value
      return (n_prop - min_value) / (max_value - min_value), min_value, max_value

   def uncache_part_array(self, part):
      """Removes the cached copy of the array of data for the named property part.

      argument:
         part (string): the part name for which the cached array is to be removed

      note:
         this function applies a python delattr() which will mark the array as no longer being in use
         here; however, actual freeing of the memory only happens when all other references to the
         array are released
      """

      cached_array_name = _cache_name(part)
      if cached_array_name is not None and hasattr(self, cached_array_name):
         delattr(self, cached_array_name)

   def add_cached_array_to_imported_list(self,
                                         cached_array,
                                         source_info,
                                         keyword,
                                         discrete = False,
                                         uom = None,
                                         time_index = None,
                                         null_value = None,
                                         property_kind = None,
                                         local_property_kind_uuid = None,
                                         facet_type = None,
                                         facet = None,
                                         realization = None,
                                         indexable_element = None,
                                         count = 1,
                                         const_value = None):
      """Caches array and adds to the list of imported properties (but not to the collection dict).

      arguments:
         cached_array: a numpy array to be added to the imported list for this collection (prior to being added
            as a part); for a constant array set cached_array to None (and use const_value)
         source_info (string): typically the name of a file from which the array has been read but can be any
            information regarding the source of the data
         keyword (string): this will be used as the citation title when a part is generated for the array
         discrete (boolean, optional, default False): if True, the array should contain integer (or boolean)
            data; if False, float
         uom (string, optional, default None): the resqml units of measure for the data
         time_index (integer, optional, default None): if not None, the time index to be used when creating
            a part for the array
         null_value (int or float, optional, default None): if present, this is used in the metadata to
            indicate that this value is to be interpreted as a null value wherever it appears in the data
         property_kind (string): resqml property kind, or None
         local_property_kind_uuid (uuid.UUID or string): uuid of local property kind, or None
         facet_type (string): resqml facet type, or None
         facet (string): resqml facet, or None
         realization (int): realization number, or None
         indexable_element (string, optional): the indexable element in the supporting representation
         count (int, default 1): the number of values per indexable element; if greater than one then this
            must be the fastest cycling axis in the cached array, ie last index
         const_value (int, float or bool, optional): the value with which a constant array is filled;
            required if cached_array is None, must be None otherwise

      returns:
         uuid of nascent property object

      note:
         the process of importing property arrays follows these steps:
         1. read (or generate) array of data into a numpy array in memory (cache)
         2. add to the imported list using this function (which also takes takes note of some metadata)
         3. write imported list arrays to hdf5 file
         4. create resqml xml nodes for imported list arrays and add as parts to model
         5. include newly added parts in collection

      :meta common:
      """

      assert (cached_array is not None and const_value is None) or (cached_array is None and const_value is not None)
      assert count > 0
      if self.imported_list is None:
         self.imported_list = []

      uuid = bu.new_uuid()
      cached_name = _cache_name_for_uuid(uuid)
      if cached_array is not None:
         self.__dict__[cached_name] = cached_array
         zorro = self.masked_array(cached_array, exclude_value = null_value)
         if not discrete and np.all(np.isnan(zorro)):
            min_value = max_value = None
         else:
            min_value = np.nanmin(zorro)
            max_value = np.nanmax(zorro)
         if min_value is ma.masked or min_value == np.NaN:
            min_value = None
         if max_value is ma.masked or max_value == np.NaN:
            max_value = None
      else:
         if const_value == null_value or (not discrete and np.isnan(const_value)):
            min_value = max_value = None
         else:
            min_value = max_value = const_value
      self.imported_list.append((uuid, source_info, keyword, cached_name, discrete, uom, time_index, null_value,
                                 min_value, max_value, property_kind, facet_type, facet, realization, indexable_element,
                                 count, local_property_kind_uuid, const_value))
      return uuid

   def remove_cached_imported_arrays(self):
      """Removes any cached arrays that are mentioned in imported list."""

      for imported in self.imported_list:
         cached_name = imported[3]
         if hasattr(self, cached_name):
            delattr(self, cached_name)

   def remove_cached_part_arrays(self):
      """Removes any cached arrays for parts of the collection."""

      for part in self.dict:
         self.uncache_part_array(part)

   def remove_all_cached_arrays(self):
      """Removes any cached arrays for parts or mentioned in imported list."""

      self.remove_cached_imported_arrays()
      self.remove_cached_part_arrays()

   def write_hdf5_for_imported_list(self, file_name = None, mode = 'a'):
      """Create or append to an hdf5 file, writing datasets for the imported arrays.

      :meta common:
      """

      # NB: imported array data must all have been cached prior to calling this function
      assert self.imported_list is not None
      h5_reg = rwh5.H5Register(self.model)
      for entry in self.imported_list:
         if entry[17] is not None:
            continue  # constant array – handled entirely in xml
         h5_reg.register_dataset(entry[0], 'values_patch0', self.__dict__[entry[3]])
      h5_reg.write(file = file_name, mode = mode)

   def write_hdf5_for_part(self, part, file_name = None, mode = 'a'):
      """Create or append to an hdf5 file, writing dataset for the specified part."""

      if self.constant_value_for_part(part) is not None:
         return
      h5_reg = rwh5.H5Register(self.model)
      a = self.cached_part_array_ref(part)
      h5_reg.register_dataset(self.uuid_for_part(part), 'values_patch0', a)
      h5_reg.write(file = file_name, mode = mode)

   def create_xml_for_imported_list_and_add_parts_to_model(self,
                                                           ext_uuid = None,
                                                           support_uuid = None,
                                                           time_series_uuid = None,
                                                           selected_time_indices_list = None,
                                                           string_lookup_uuid = None,
                                                           property_kind_uuid = None,
                                                           find_local_property_kinds = True,
                                                           extra_metadata = {}):
      """Add imported or generated grid property arrays as parts in parent model, creating xml; hdf5 should already have been written.

      arguments:
         ext_uuid: uuid for the hdf5 external part, which must be known to the model's hdf5 dictionary
         support_uuid (optional): the uuid of the supporting representation that the imported properties relate to
         time_series_uuid (optional): the uuid of the full or reduced time series for which any recurrent properties'
            timestep numbers can be used as a time index; in the case of a reduced series, the selected_time_indices_list
            argument must be passed and the properties timestep numbers are found in the list with the position yielding
            the time index for the reduced list; time_series_uuid should be present if there are any recurrent properties
            in the imported list
         selected_time_indices_list (list of int, optional): if time_series_uuid is for a reduced time series then this
            argument must be present and its length must match the number of timestamps in the reduced series; the values
            in the list are indices in the full time series
         string_lookup_uuid (optional): if present, the uuid of the string table lookup which any non-continuous
            properties relate to (ie. they are all taken to be categorical)
         property_kind_uuid (optional): if present, the uuid of the bespoke (local) property kind for all the
            property arrays in the imported list (except those with an individual local property kind uuid)
         find_local_property_kinds (boolean, default True): if True, local property kind uuids need not be provided as
            long as the property kinds are set to match the titles of the appropriate local property kind objects
         extra_metadata (optional): if present, a dictionary of extra metadata to be added for the part

      returns:
         list of uuid.UUID, being the uuids of the newly added property parts

      notes:
         the imported list should have been built up, and associated hdf5 arrays written, before calling this method;
         the imported list is cleared as a deliberate side-effect of this method (so a new set of imports can be
         started hereafter);
         discrete and categorical properties cannot be mixed in the same import list - process as separate lists;
         all categorical properties in the import list must refer to the same string table lookup;
         when importing categorical properties, establish the xml for the string table lookup before calling this method;
         if importing properties of a bespoke (local) property kind, ensure the property kind objects exist as parts in
         the model before calling this method

      :meta common:
      """

      if self.imported_list is None:
         return []
      if ext_uuid is None:
         ext_uuid = self.model.h5_uuid()
      prop_parts_list = []
      uuid_list = []
      for (p_uuid, p_file_name, p_keyword, p_cached_name, p_discrete, p_uom, p_time_index, p_null_value, p_min_value,
           p_max_value, property_kind, facet_type, facet, realization, indexable_element, count,
           local_property_kind_uuid, const_value) in self.imported_list:
         log.debug('processing imported property ' + str(p_keyword))
         if local_property_kind_uuid is None:
            local_property_kind_uuid = property_kind_uuid
         if property_kind is None:
            if local_property_kind_uuid is not None:
               property_kind = self.model.title(
                  uuid = local_property_kind_uuid)  # note: requires local property kind to be present
            else:
               (property_kind, facet_type,
                facet) = property_kind_and_facet_from_keyword(p_keyword)  # todo: only if None in ab_property_list
         if property_kind is None:
            # todo: the following are abstract standard property kinds, which shouldn't really have data directly associated with them
            if p_discrete:
               if string_lookup_uuid is not None:
                  property_kind = 'categorical'
               else:
                  property_kind = 'discrete'
            else:
               property_kind = 'continuous'
         if hasattr(self, p_cached_name):
            p_array = self.__dict__[p_cached_name]
         else:
            p_array = None
         if property_kind == 'categorical':
            add_min_max = False
         elif local_property_kind_uuid is not None and string_lookup_uuid is not None:
            add_min_max = False
         else:
            add_min_max = True
         if selected_time_indices_list is not None and p_time_index is not None:
            p_time_index = selected_time_indices_list.index(p_time_index)
         p_node = self.create_xml(
            ext_uuid = ext_uuid,
            property_array = p_array,
            title = p_keyword,
            property_kind = property_kind,
            support_uuid = support_uuid,
            p_uuid = p_uuid,
            facet_type = facet_type,
            facet = facet,
            discrete = p_discrete,  # todo: time series bits
            time_series_uuid = time_series_uuid,
            time_index = p_time_index,
            uom = p_uom,
            null_value = p_null_value,
            originator = None,
            source = p_file_name,
            add_as_part = True,
            add_relationships = True,
            add_min_max = add_min_max,
            min_value = p_min_value,
            max_value = p_max_value,
            realization = realization,
            string_lookup_uuid = string_lookup_uuid,
            property_kind_uuid = local_property_kind_uuid,
            indexable_element = indexable_element,
            count = count,
            find_local_property_kinds = find_local_property_kinds,
            extra_metadata = extra_metadata,
            const_value = const_value)
         if p_node is not None:
            prop_parts_list.append(rqet.part_name_for_part_root(p_node))
            uuid_list.append(rqet.uuid_for_part_root(p_node))
      self.add_parts_list_to_dict(prop_parts_list)
      self.imported_list = []
      return uuid_list

   def create_xml(self,
                  ext_uuid,
                  property_array,
                  title,
                  property_kind,
                  support_uuid = None,
                  p_uuid = None,
                  facet_type = None,
                  facet = None,
                  discrete = False,
                  time_series_uuid = None,
                  time_index = None,
                  uom = None,
                  null_value = None,
                  originator = None,
                  source = None,
                  add_as_part = True,
                  add_relationships = True,
                  add_min_max = True,
                  min_value = None,
                  max_value = None,
                  realization = None,
                  string_lookup_uuid = None,
                  property_kind_uuid = None,
                  find_local_property_kinds = True,
                  indexable_element = None,
                  count = 1,
                  extra_metadata = {},
                  const_value = None):
      """Create a property xml node for a single property related to a given supporting representation node.

      arguments:
         ext_uuid (uuid.UUID): the uuid of the hdf5 external part
         property_array (numpy array): the actual property array (used to populate xml min & max values);
            may be None if min_value and max_value are passed or add_min_max is False
         title (string): used for the citation Title text for the property; often set to a simulator keyword for
            grid properties
         property_kind (string): the resqml property kind of the property; in the case of a bespoke (local)
            property kind, this is used as the title in the local property kind reference and the
            property_kind_uuid argument must also be passed or find_local_property_kinds set True
         support_uuid (uuid.UUID, optional): if None, the support for the collection is used
         p_uuid (uuid.UUID, optional): if None, a new uuid is generated for the property; otherwise this
            uuid is used
         facet_type (string, optional): if present, a resqml facet type whose value is supplied in the facet argument
         facet (string, optional): required if facet_type is supplied; the value of the facet
         discrete (boolean, default False): if True, a discrete or categorical property node is created (depending
            on whether string_lookup_uuid is None or present); if False (default), a continuous property node is created
         time_series_uuid (uuid.UUID, optional): if present, the uuid of the time series that this (recurrent)
            property relates to
         time_index (int, optional): if time_series_uuid is not None, this argument is required and provides
            the time index into the time series for this property array
         uom (string): the resqml unit of measure for the property (only used for continuous properties)
         null_value (optional): the value that is to be interpreted as null if it appears in the property array
         originator (string, optional): the name of the human being who created the property object;
            default is to use the login name
         source (string, optional): if present, an extra metadata node is added as a child to the property
            node, with this string indicating the source of the property data
         add_as_part (boolean, default True): if True, the newly created xml node is added as a part
            in the model
         add_relationships (boolean, default True): if True, relationship xml parts are created relating the
            new property part to: the support, the hdf5 external part; and the time series part (if applicable)
         add_min_max (boolean, default True): if True, min and max values are included as children in the
            property node
         min_value (optional): if present and add_min_max is True, this is used as the minimum value (otherwise it
            is calculated from the property array)
         max_value (optional): if present and add_min_max is True, this is used as the maximum value (otherwise it
            is calculated from the property array)
         realization (int, optional): if present, is used as the realization number in the property node; if None,
            no realization child is created
         string_lookup_uuid (optional): if present, and discrete is True, a categorical property node is created
            which refers to this string table lookup
         property_kind_uuid (optional): if present, the property kind is a local property kind; must be None for a
            standard property kind
         find_local_property_kinds (boolean, default True): if True and property_kind is not in standard supported
            property kind list and property_kind_uuid is None, the citation titles of PropertyKind objects in the
            model are compared with property_kind and if a match is found, that local property kind is used
         indexable_element (string, optional): if present, is used as the indexable element in the property node;
            if None, 'cells' are used for grid properties and 'nodes' for wellbore frame properties
         count (int, default 1): the number of values per indexable element; if greater than one then this axis
            must cycle fastest in the array, ie. be the last index
         extra_metadata (dictionary, optional): if present, adds extra metadata in the xml
         const_value (float or int, optional): if present, create xml for a constant array filled with this value

      returns:
         the newly created property xml node

      notes:
         this function doesn't write the actual array data to the hdf5 file: that should be done
         before calling this function
         this code (and elsewhere) only supports at most one facet per property, though the resqml standard
         allows for multiple facets
      """

      #      log.debug('creating property node for ' + title)
      # currently assumes discrete properties to be 32 bit integers and continuous to be 64 bit reals
      # also assumes property_kind is one of the standard resqml property kinds; todo: allow local p kind node as optional arg
      assert self.model is not None
      if support_uuid is None:
         support_uuid = self.support_uuid
      assert support_uuid is not None
      support_root = self.model.root_for_uuid(support_uuid)
      assert support_root is not None

      if ext_uuid is None:
         ext_uuid = self.model.h5_uuid()

      support_type = self.model.type_of_part(self.model.part_for_uuid(support_uuid))
      if indexable_element is None:
         if support_type in [
               'obj_IjkGridRepresentation', 'obj_BlockedWellboreRepresentation', 'obj_Grid2dRepresentation',
               'obj_UnstructuredGridRepresentation'
         ]:
            indexable_element = 'cells'
         elif support_type == 'obj_WellboreFrameRepresentation':
            indexable_element = 'nodes'  # note: could be 'intervals'
         elif support_type == 'obj_GridConnectionSetRepresentation':
            indexable_element = 'faces'
         else:
            raise Exception('indexable element unknown for unsupported supporting representation object')

      if self.support is not None:
         shape_list = self.supporting_shape(indexable_element = indexable_element)
         if shape_list is not None:
            if count > 1:
               shape_list.append(count)
            if property_array is not None:
               assert tuple(shape_list) == property_array.shape, 'property array does not have the correct shape'
      # todo: assertions:
      #    numpy data type matches discrete flag (and assumptions about precision)
      #    uom are valid units for property_kind
      assert property_kind, 'missing property kind when creating xml for property'

      if discrete:
         if string_lookup_uuid is None:
            d_or_c_text = 'Discrete'
         else:
            d_or_c_text = 'Categorical'
         xsd_type = 'integer'
         hdf5_type = 'IntegerHdf5Array'
      else:
         d_or_c_text = 'Continuous'
         xsd_type = 'double'
         hdf5_type = 'DoubleHdf5Array'
         null_value = None

      p_node = self.model.new_obj_node(d_or_c_text + 'Property')
      if p_uuid is None:
         p_uuid = bu.uuid_from_string(p_node.attrib['uuid'])
      else:
         p_node.attrib['uuid'] = str(p_uuid)

      self.model.create_citation(root = p_node, title = title, originator = originator)

      rqet.create_metadata_xml(node = p_node, extra_metadata = extra_metadata)

      if source is not None and len(source) > 0:
         self.model.create_source(source = source, root = p_node)

      count_node = rqet.SubElement(p_node, ns['resqml2'] + 'Count')
      count_node.set(ns['xsi'] + 'type', ns['xsd'] + 'positiveInteger')
      count_node.text = str(count)

      ie_node = rqet.SubElement(p_node, ns['resqml2'] + 'IndexableElement')
      ie_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'IndexableElements')
      ie_node.text = indexable_element

      if realization is not None and realization >= 0:
         ri_node = rqet.SubElement(p_node, ns['resqml2'] + 'RealizationIndex')
         ri_node.set(ns['xsi'] + 'type', ns['xsd'] + 'nonNegativeInteger')
         ri_node.text = str(realization)

      if time_series_uuid is None or time_index is None:
         related_time_series_node = None
      else:
         related_time_series_node = self.model.root(uuid = time_series_uuid)
         time_series = rts.TimeSeries(self.model, uuid = time_series_uuid)
         time_series.create_time_index(time_index, root = p_node)

      self.model.create_supporting_representation(support_uuid = support_uuid,
                                                  root = p_node,
                                                  title = rqet.citation_title_for_node(support_root),
                                                  content_type = support_type)

      p_kind_node = rqet.SubElement(p_node, ns['resqml2'] + 'PropertyKind')
      p_kind_node.text = rqet.null_xml_text
      if find_local_property_kinds and property_kind not in supported_property_kind_list:
         if property_kind_uuid is None:
            pk_parts_list = self.model.parts_list_of_type('PropertyKind')
            for part in pk_parts_list:
               if self.model.citation_title_for_part(part) == property_kind:
                  property_kind_uuid = self.model.uuid_for_part(part)
                  break
            if property_kind_uuid is None:
               # create local property kind object and fetch uuid
               lpk = PropertyKind(self.model,
                                  title = property_kind,
                                  example_uom = uom,
                                  parent_property_kind = 'discrete' if discrete else 'continuous')
               lpk.create_xml()
               property_kind_uuid = lpk.uuid
      if property_kind_uuid is None:
         p_kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'StandardPropertyKind')  # todo: local prop kind ref
         kind_node = rqet.SubElement(p_kind_node, ns['resqml2'] + 'Kind')
         kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ResqmlPropertyKind')
         kind_node.text = property_kind
      else:
         p_kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'LocalPropertyKind')  # todo: local prop kind ref
         self.model.create_ref_node('LocalPropertyKind',
                                    property_kind,
                                    property_kind_uuid,
                                    content_type = 'obj_PropertyKind',
                                    root = p_kind_node)

      # create patch node
      const_count = None
      if const_value is not None:
         s_shape = self.supporting_shape(indexable_element = indexable_element,
                                         direction = facet if facet_type == 'direction' else None)
         assert s_shape is not None
         const_count = np.product(np.array(s_shape, dtype = int))
      _ = self.model.create_patch(p_uuid,
                                  ext_uuid,
                                  root = p_node,
                                  hdf5_type = hdf5_type,
                                  xsd_type = xsd_type,
                                  null_value = null_value,
                                  const_value = const_value,
                                  const_count = const_count)

      if facet_type is not None and facet is not None:
         facet_node = rqet.SubElement(p_node, ns['resqml2'] + 'Facet')
         facet_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'PropertyKindFacet')
         facet_node.text = rqet.null_xml_text
         facet_type_node = rqet.SubElement(facet_node, ns['resqml2'] + 'Facet')
         facet_type_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Facet')
         facet_type_node.text = facet_type
         facet_value_node = rqet.SubElement(facet_node, ns['resqml2'] + 'Value')
         facet_value_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
         facet_value_node.text = facet

      if add_min_max:
         # todo: use active cell mask on numpy min and max operations; exclude null values on discrete min max
         if const_value is not None:
            if (discrete and const_value != null_value) or (not discrete and not np.isnan(const_value)):
               if min_value is None:
                  min_value = const_value
               if max_value is None:
                  max_value = const_value
         elif property_array is not None:
            if discrete:
               if min_value is None:
                  try:
                     min_value = int(property_array.min())
                  except Exception:
                     min_value = None
                     log.warning('no xml minimum value set for discrete property')
               if max_value is None:
                  try:
                     max_value = int(property_array.max())
                  except Exception:
                     max_value = None
                     log.warning('no xml maximum value set for discrete property')
            else:
               if min_value is None or max_value is None:
                  all_nan = np.all(np.isnan(property_array))
               if min_value is None and not all_nan:
                  min_value = np.nanmin(property_array)
                  if np.isnan(min_value) or min_value is ma.masked:
                     min_value = None
               if max_value is None and not all_nan:
                  max_value = np.nanmax(property_array)
                  if np.isnan(max_value) or max_value is ma.masked:
                     max_value = None
         if min_value is not None:
            min_node = rqet.SubElement(p_node, ns['resqml2'] + 'MinimumValue')
            min_node.set(ns['xsi'] + 'type', ns['xsd'] + xsd_type)
            min_node.text = str(min_value)
         if max_value is not None:
            max_node = rqet.SubElement(p_node, ns['resqml2'] + 'MaximumValue')
            max_node.set(ns['xsi'] + 'type', ns['xsd'] + xsd_type)
            max_node.text = str(max_value)

      if discrete:
         if string_lookup_uuid is not None:
            sl_root = self.model.root_for_uuid(string_lookup_uuid)
            assert sl_root is not None, 'string table lookup is missing whilst importing categorical property'
            assert rqet.node_type(sl_root) == 'obj_StringTableLookup', 'referenced uuid is not for string table lookup'
            self.model.create_ref_node('Lookup',
                                       self.model.title_for_root(sl_root),
                                       string_lookup_uuid,
                                       content_type = 'obj_StringTableLookup',
                                       root = p_node)
      else:  # continuous
         if not uom:
            uom = guess_uom(property_kind, min_value, max_value, self.support, facet_type = facet_type, facet = facet)
            if not uom:
               uom = 'Euc'  # todo: put RESQML base uom for quantity class here, instead of Euc
               log.warning(f'uom set to Euc for property {title} of kind {property_kind}')
         self.model.uom_node(p_node, uom)

      if add_as_part:
         self.model.add_part('obj_' + d_or_c_text + 'Property', p_uuid, p_node)
         if add_relationships:
            self.model.create_reciprocal_relationship(p_node, 'destinationObject', support_root, 'sourceObject')
            if property_kind_uuid is not None:
               pk_node = self.model.root_for_uuid(property_kind_uuid)
               if pk_node is not None:
                  self.model.create_reciprocal_relationship(p_node, 'destinationObject', pk_node, 'sourceObject')
            if related_time_series_node is not None:
               self.model.create_reciprocal_relationship(p_node, 'destinationObject', related_time_series_node,
                                                         'sourceObject')
            if discrete and string_lookup_uuid is not None:
               self.model.create_reciprocal_relationship(p_node, 'destinationObject', sl_root, 'sourceObject')


#           ext_node = self.model.root_for_part(rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = True))
            if const_value is None:
               ext_node = self.model.root_for_part(
                  rqet.part_name_for_object('obj_EpcExternalPartReference', ext_uuid, prefixed = False))
               self.model.create_reciprocal_relationship(p_node, 'mlToExternalPartProxy', ext_node,
                                                         'externalPartProxyToMl')

      return p_node

   def create_property_set_xml(self,
                               title,
                               ps_uuid = None,
                               originator = None,
                               add_as_part = True,
                               add_relationships = True):
      """Creates an xml node for a property set to represent this collection of properties.

      arguments:
         title (string): to be used as citation title
         ps_uuid (string, optional): if present, used as the uuid for the property set, otherwise a new uuid is generated
         originator (string, optional): if present, used as the citation creator (otherwise login name is used)
         add_as_part (boolean, default True): if True, the property set is added to the model as a part
         add_relationships (boolean, default True): if True, the relationships to the member properties are added

      note:
         xml for individual properties should exist before calling this method
      """

      assert self.model is not None, 'cannot create xml for property set as model is not set'
      assert self.number_of_parts() > 0, 'cannot create xml for property set as no parts in collection'

      ps_node = self.model.new_obj_node('PropertySet')
      if ps_uuid is None:
         ps_uuid = bu.uuid_from_string(ps_node.attrib['uuid'])
      else:
         ps_node.attrib['uuid'] = str(ps_uuid)

      self.model.create_citation(root = ps_node, title = title, originator = originator)

      tsk_node = rqet.SubElement(ps_node, ns['resqml2'] + 'TimeSetKind')
      tsk_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'TimeSetKind')
      tsk_node.text = self.time_set_kind()

      hspk_node = rqet.SubElement(ps_node, ns['resqml2'] + 'HasSinglePropertyKind')
      hspk_node.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
      hspk_node.text = str(self.has_single_property_kind()).lower()

      hmr_node = rqet.SubElement(ps_node, ns['resqml2'] + 'HasMultipleRealizations')
      hmr_node.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
      hmr_node.text = str(self.has_multiple_realizations()).lower()

      if self.parent_set_root is not None:
         parent_set_ref_node = self.model.create_ref_node('ParentSet',
                                                          self.model.title_for_root(self.parent_set_root),
                                                          self.parent_set_root.attrib['uuid'],
                                                          content_type = 'obj_PropertySet',
                                                          root = ps_node)

      prop_node_list = []
      for part in self.parts():
         part_root = self.model.root_for_part(part)
         self.model.create_ref_node('Properties',
                                    self.model.title_for_root(part_root),
                                    part_root.attrib['uuid'],
                                    content_type = self.model.type_of_part(part),
                                    root = ps_node)
         if add_as_part and add_relationships:
            prop_node_list.append(part_root)

      if add_as_part:
         self.model.add_part('obj_PropertySet', ps_uuid, ps_node)
         if add_relationships:
            # todo: add relationship with time series if time set kind is not 'not a time set'?
            if self.parent_set_root is not None:
               self.model.create_reciprocal_relationship(ps_node, 'destinationObject', parent_set_ref_node,
                                                         'sourceObject')
            for prop_node in prop_node_list:
               self.model.create_reciprocal_relationship(ps_node, 'destinationObject', prop_node, 'sourceObject')

      return ps_node

   def basic_static_property_parts(self,
                                   realization = None,
                                   share_perm_parts = False,
                                   perm_k_mode = None,
                                   perm_k_ratio = 1.0):
      """Returns five parts: net to gross ratio, porosity, permeability rock I, J & K; each returned part may be None.

      arguments:
         realization: (int, optional): if present, only properties with the given realization are considered; if None,
            all properties in the collection are considered
         share_perm_parts (boolean, default False): if True, the permeability I part will also be returned for J and/or K
            if no other properties are found for those directions; if False, None will be returned for such parts
         perm_k_mode (string, optional): if present, indicates what action to take when no K direction permeability is
            found; valid values are:
            'none': same as None, perm K part return value will be None
            'shared': if share_perm_parts is True, then perm I value will be used for perm K, else same as None
            'ratio': multiply IJ permeability by the perm_k_ratio argument
            'ntg': multiply IJ permeability by ntg and by perm_k_ratio
            'ntg squared': multiply IJ permeability by square of ntg and by perm_k_ratio
         perm_k_ratio (float, default 1.0): a Kv:Kh ratio, typically in the range zero to one, applied if generating
            a K permeability array (perm_k_mode is 'ratio', 'ntg' or 'ntg squared' and no existing K permeability found);
            ignored otherwise

      returns:
         tuple of 5 strings being part names for: net to gross ratio, porosity, perm i, perm j, perm k (respectively);
         any of the returned elements may be None if no appropriate property was identified

      note:
         if generating a K permeability array, the data is appended to the hdf5 file and the xml is created, however
         the epc re-write must be carried out by the calling code

      :meta common:
      """

      ntg_part = poro_part = perm_i_part = perm_j_part = perm_k_part = None
      try:
         ntg_part = self.singleton(realization = realization, property_kind = 'net to gross ratio')
      except Exception:
         log.error('problem with net to gross ratio data (more than one array present?)')
         ntg_part = None
      try:
         poro_part = self.singleton(realization = realization, property_kind = 'porosity')
      except Exception:
         log.error('problem with porosity data (more than one array present?)')
         poro_part = None
      perms = selective_version_of_collection(self, realization = realization, property_kind = 'permeability rock')
      if perms is None or perms.number_of_parts() == 0:
         log.error('no rock permeabilities present')
      else:
         if perms.number_of_parts() == 1:
            perm_i_part = perms.singleton()
            if share_perm_parts:
               perm_j_part = perm_k_part = perm_i_part
            elif perms.facet_type_for_part(perm_i_part) == 'direction':
               direction = perms.facet_for_part(perm_i_part)
               if direction == 'J':
                  perm_j_part = perm_i_part
                  perm_i_part = None
               elif direction == 'IJ':
                  perm_j_part = perm_i_part
               elif direction == 'K':
                  perm_k_part = perm_i_part
                  perm_i_part = None
               elif direction == 'IJK':
                  perm_j_part = perm_k_part = perm_i_part
         else:
            try:
               perm_i_part = perms.singleton(facet_type = 'direction', facet = 'I')
               if not perm_i_part:
                  perm_i_part = perms.singleton(facet_type = 'direction', facet = 'IJ')
               if not perm_i_part:
                  perm_i_part = perms.singleton(facet_type = 'direction', facet = 'IJK')
               if not perm_i_part:
                  perm_i_part = perms.singleton(citation_title = 'KI')
               if not perm_i_part:
                  perm_i_part = perms.singleton(citation_title = 'PERMI')
               if not perm_i_part:
                  perm_i_part = perms.singleton(citation_title = 'KX')
               if not perm_i_part:
                  perm_i_part = perms.singleton(citation_title = 'PERMX')
               if not perm_i_part:
                  log.error('unable to discern which rock permeability to use for I direction')
            except Exception:
               log.error('problem with permeability data (more than one I direction array present?)')
               perm_i_part = None
            try:
               perm_j_part = perms.singleton(facet_type = 'direction', facet = 'J')
               if not perm_j_part:
                  perm_j_part = perms.singleton(facet_type = 'direction', facet = 'IJ')
               if not perm_j_part:
                  perm_j_part = perms.singleton(facet_type = 'direction', facet = 'IJK')
               if not perm_j_part:
                  perm_j_part = perms.singleton(citation_title = 'KJ')
               if not perm_j_part:
                  perm_j_part = perms.singleton(citation_title = 'PERMJ')
               if not perm_j_part:
                  perm_j_part = perms.singleton(citation_title = 'KY')
               if not perm_j_part:
                  perm_j_part = perms.singleton(citation_title = 'PERMY')
            except Exception:
               log.error('problem with permeability data (more than one J direction array present?)')
               perm_j_part = None
            if perm_j_part is None and share_perm_parts:
               perm_j_part = perm_i_part
            elif perm_i_part is None and share_perm_parts:
               perm_i_part = perm_j_part
            try:
               perm_k_part = perms.singleton(facet_type = 'direction', facet = 'K')
               if perm_k_part is None:
                  perm_k_part = perms.singleton(facet_type = 'direction', facet = 'IJK')
               if not perm_k_part:
                  perm_k_part = perms.singleton(citation_title = 'KK')
               if not perm_k_part:
                  perm_k_part = perms.singleton(citation_title = 'PERMK')
               if not perm_k_part:
                  perm_k_part = perms.singleton(citation_title = 'KZ')
               if not perm_k_part:
                  perm_k_part = perms.singleton(citation_title = 'PERMZ')
            except Exception:
               log.error('problem with permeability data (more than one K direction array present?)')
               perm_k_part = None
            if perm_k_part is None:
               assert perm_k_mode in [None, 'none', 'shared', 'ratio', 'ntg', 'ntg squared']
               # note: could switch ratio mode to shared if perm_k_ratio is 1.0
               if perm_k_mode is None or perm_k_mode == 'none':
                  pass
               elif perm_k_mode == 'shared':
                  if share_perm_parts:
                     perm_k_part = perm_i_part
               elif perm_i_part is not None:
                  log.info('generating K permeability data using mode ' + str(perm_k_mode))
                  if perm_j_part is not None and perm_j_part != perm_i_part:
                     # generate root mean square of I & J permeabilities to use as horizontal perm
                     kh = np.sqrt(perms.cached_part_array_ref(perm_i_part) * perms.cached_part_array_ref(perm_j_part))
                  else:  # use I permeability as horizontal perm
                     kh = perms.cached_part_array_ref(perm_i_part)
                  kv = kh * perm_k_ratio
                  if ntg_part is not None:
                     if perm_k_mode == 'ntg':
                        kv *= self.cached_part_array_ref(ntg_part)
                     elif perm_k_mode == 'ntg squared':
                        ntg = self.cached_part_array_ref(ntg_part)
                        kv *= ntg * ntg
                  kv_collection = PropertyCollection()
                  kv_collection.set_support(support_uuid = self.support_uuid, model = self.model)
                  kv_collection.add_cached_array_to_imported_list(
                     kv,
                     'derived from horizontal perm with mode ' + str(perm_k_mode),
                     'KK',
                     discrete = False,
                     uom = 'mD',
                     time_index = None,
                     null_value = None,
                     property_kind = 'permeability rock',
                     facet_type = 'direction',
                     facet = 'K',
                     realization = perms.realization_for_part(perm_i_part),
                     indexable_element = perms.indexable_for_part(perm_i_part),
                     count = 1)
                  self.model.h5_release()
                  kv_collection.write_hdf5_for_imported_list()
                  kv_collection.create_xml_for_imported_list_and_add_parts_to_model()
                  self.inherit_parts_from_other_collection(kv_collection)
                  perm_k_part = kv_collection.singleton()

      return ntg_part, poro_part, perm_i_part, perm_j_part, perm_k_part

   def basic_static_property_parts_dict(self,
                                        realization = None,
                                        share_perm_parts = False,
                                        perm_k_mode = None,
                                        perm_k_ratio = 1.0):
      """Same as basic_static_property_parts() method but returning a dictionary with 5 items.

      note:
         returned dictionary contains following keys: 'NTG', 'PORO', 'PERMI', 'PERMJ', 'PERMK'
      """

      five_parts = self.basic_static_property_parts(realization = realization,
                                                    share_perm_parts = share_perm_parts,
                                                    perm_k_mode = perm_k_mode,
                                                    perm_k_ratio = perm_k_ratio)
      return {
         'NTG': five_parts[0],
         'PORO': five_parts[1],
         'PERMI': five_parts[2],
         'PERMJ': five_parts[3],
         'PERMK': five_parts[4]
      }

   def basic_static_property_uuids(self,
                                   realization = None,
                                   share_perm_parts = False,
                                   perm_k_mode = None,
                                   perm_k_ratio = 1.0):
      """Returns five uuids: net to gross ratio, porosity, permeability rock I, J & K; each returned uuid may be None.

      note:
         see basic_static_property_parts() method for argument documentation

      :meta common:
      """

      five_parts = self.basic_static_property_parts(realization = realization,
                                                    share_perm_parts = share_perm_parts,
                                                    perm_k_mode = perm_k_mode,
                                                    perm_k_ratio = perm_k_ratio)
      uuid_list = []
      for part in five_parts:
         if part is None:
            uuid_list.append(None)
         else:
            uuid_list.append(rqet.uuid_in_part_name(part))
      return tuple(uuid_list)

   def basic_static_property_uuids_dict(self,
                                        realization = None,
                                        share_perm_parts = False,
                                        perm_k_mode = None,
                                        perm_k_ratio = 1.0):
      """Same as basic_static_property_uuids() method but returning a dictionary with 5 items.

      note:
         returned dictionary contains following keys: 'NTG', 'PORO', 'PERMI', 'PERMJ', 'PERMK'
      """

      five_uuids = self.basic_static_property_uuids(realization = realization,
                                                    share_perm_parts = share_perm_parts,
                                                    perm_k_mode = perm_k_mode,
                                                    perm_k_ratio = perm_k_ratio)
      return {
         'NTG': five_uuids[0],
         'PORO': five_uuids[1],
         'PERMI': five_uuids[2],
         'PERMJ': five_uuids[3],
         'PERMK': five_uuids[4]
      }


class Property(BaseResqpy):
   """Class for an individual property object; uses a single element PropertyCollection behind the scenes."""

   @property
   def resqml_type(self):
      root_node = self.root
      if root_node is not None:
         return rqet.node_type(root_node, strip_obj = True)
      if (not hasattr(self, 'collection') or self.collection.number_of_parts() != 1 or self.is_continuous()):
         return 'ContinuousProperty'
      return 'CategoricalProperty' if self.is_categorical() else 'DiscreteProperty'

   def __init__(self, parent_model, uuid = None, title = None, support_uuid = None, extra_metadata = None):
      """Initialises a resqpy Property object, either for an existing RESQML property, or empty for support.

      arguments:
         parent_model (model.Model): the model to which the property belongs
         uuid (uuid.UUID, optional): required if initialising from an existing RESQML property object
         title (str, optional): the citation title to use for the property; ignored if uuid is present
         support_uuid (uuid.UUID, optional): identifies the supporting representation for the property;
            ignored if uuid is present
         extra_metadata (dict, optional): if present, the dictionary items are added as extra metadata;
            ignored if uuid is present

      returns:
         new resqpy Property object
      """

      self.collection = PropertyCollection()
      super().__init__(model = parent_model, uuid = uuid, title = title, extra_metadata = extra_metadata)
      if support_uuid is not None:
         if self.collection.support_uuid is None:
            self.collection.set_support(model = parent_model, support_uuid = support_uuid)
         else:
            assert bu.matching_uuids(support_uuid, self.collection.support_uuid)

   def _load_from_xml(self):
      """Populates this property object from xml; not usually called directly."""

      part = self.part
      assert part is not None
      if self.collection is None:
         self.collection = PropertyCollection()
      if self.collection.model is None:
         self.collection.model = self.model
      self.collection.add_part_to_dict(part)
      self.extra_metadata = self.collection.extra_metadata_for_part(
         part)  # duplicate, as standard attribute in BaseResqpy
      self.collection.has_single_property_kind_flag = True
      self.collection.has_single_indexable_element_flag = True
      self.collection.has_multiple_realizations_flag = False
      assert self.collection.number_of_parts() == 1

   @classmethod
   def from_singleton_collection(cls, property_collection: PropertyCollection):
      """Populates a new Property from a PropertyCollection containing just one part.

      arguments:
         property_collection (PropertyCollection): the singleton collection from which to populate this Property
      """

      # Validate
      assert property_collection.model is not None
      assert property_collection is not None
      assert property_collection.number_of_parts() == 1

      # Instantiate the object i.e. call the class __init__ method
      part = property_collection.parts()[0]
      prop = cls(parent_model = property_collection.model,
                 uuid = property_collection.uuid_for_part(part),
                 support_uuid = property_collection.support_uuid,
                 title = property_collection.citation_title_for_part(part),
                 extra_metadata = property_collection.extra_metadata_for_part(part))

      # Edit properties of collection attribute
      prop.collection.has_single_property_kind_flag = True
      prop.collection.has_single_indexable_element_flag = True
      prop.collection.has_multiple_realizations_flag = False

      return prop

   @classmethod
   def from_array(cls,
                  parent_model,
                  cached_array,
                  source_info,
                  keyword,
                  support_uuid,
                  property_kind = None,
                  local_property_kind_uuid = None,
                  indexable_element = None,
                  facet_type = None,
                  facet = None,
                  discrete = False,
                  uom = None,
                  null_value = None,
                  time_series_uuid = None,
                  time_index = None,
                  realization = None,
                  count = 1,
                  const_value = None,
                  string_lookup_uuid = None,
                  find_local_property_kind = True,
                  extra_metadata = {}):
      """Populates a new Property from a numpy array and metadata; NB. Writes data to hdf5 and adds part to model.

      arguments:
         parent_model (model.Model): the model to which the new property belongs
         cached_array: a numpy array to be made into a property; for a constant array set cached_array to None
            (and use const_value)
         source_info (string): typically the name of a file from which the array has been read but can be any
            information regarding the source of the data
         keyword (string): this will be used as the citation title for the property
         support_uuid (uuid): the uuid of the supporting representation
         property_kind (string): resqml property kind (required unless a local propery kind is identified with
            local_property_kind_uuid)
         local_property_kind_uuid (uuid.UUID or string, optional): uuid of local property kind, or None if a
            standard property kind applies or if the local property kind is identified by its name (see
            find_local_property_kind)
         indexable_element (string): the indexable element in the supporting representation; if None
            then the 'typical' indexable element for the class of supporting representation will be assumed
         facet_type (string, optional): resqml facet type, or None; resqpy only supports at most one facet per
            property though RESQML allows for multiple)
         facet (string, optional): resqml facet, or None; resqpy only supports at most one facet per property
            though RESQML allows for multiple)
         discrete (boolean, optional, default False): if True, the array should contain integer (or boolean)
            data, if False, float; set True for any discrete data including categorical
         uom (string, optional, default None): the resqml units of measure for the data; required for
            continuous (float) array
         null_value (int, optional, default None): if present, this is used in the metadata to indicate that
            the value is to be interpreted as a null value wherever it appears in the discrete data; not used
            for continuous data where NaN is always the null value
         time_series_uuid (optional): the uuid of the full or reduced time series that the time_index is for
         time_index (integer, optional, default None): if not None, the time index for the property (see also
            time_series_uuid)
         realization (int, optional): realization number, or None
         count (int, default 1): the number of values per indexable element; if greater than one then this
            must be the fastest cycling axis in the cached array, ie last index
         const_value (int, float or bool, optional): the value with which a constant array is filled;
            required if cached_array is None, must be None otherwise
         string_lookup_uuid (optional): if present, the uuid of the string table lookup which the categorical data
            relates to; if None, the property will not be configured as categorical
         find_local_property_kind (boolean, default True): if True, local property kind uuid need not be provided as
            long as the property_kind is set to match the title of the appropriate local property kind object
         extra_metadata (optional): if present, a dictionary of extra metadata to be added for the part

      returns:
         new Property object built from numpy array; the hdf5 data has been written, xml created and the part
         added to the model

      notes:
         this method writes to the hdf5 file and creates the xml node, which is added as a part to the model;
         calling code must still call the model's store_epc() method;
         this from_array() method is a convenience method calling self.collection.set_support(),
         self.prepare_import(), self.write_hdf5() and self.create_xml()

      :meta common:
      """
      # Validate
      assert parent_model is not None
      assert cached_array is not None or const_value is not None

      # Instantiate the object i.e. call the class __init__ method
      prop = cls(parent_model = parent_model, title = keyword, extra_metadata = extra_metadata)

      # Prepare array data in collection attribute and add to model
      prop.collection.set_support(model = prop.model,
                                  support_uuid = support_uuid)  # this can be expensive; todo: optimise
      if prop.collection.model is None:
         prop.collection.model = prop.model
      prop.prepare_import(cached_array,
                          source_info,
                          keyword,
                          discrete = discrete,
                          uom = uom,
                          time_index = time_index,
                          null_value = null_value,
                          property_kind = property_kind,
                          local_property_kind_uuid = local_property_kind_uuid,
                          facet_type = facet_type,
                          facet = facet,
                          realization = realization,
                          indexable_element = indexable_element,
                          count = count,
                          const_value = const_value)
      prop.write_hdf5()
      prop.create_xml(support_uuid = support_uuid,
                      time_series_uuid = time_series_uuid,
                      string_lookup_uuid = string_lookup_uuid,
                      property_kind_uuid = local_property_kind_uuid,
                      find_local_property_kind = find_local_property_kind,
                      extra_metadata = extra_metadata)
      return prop

   def array_ref(self, dtype = None, masked = False, exclude_null = False):
      """Returns a (cached) numpy array containing the property values.

      arguments:
         dtype (str or type, optional): the required dtype of the array (usually float, int or bool)
         masked (boolean, default False): if True, a numpy masked array is returned, with the mask determined by
            the inactive cell mask in the case of a Grid property
         exclude_null (boolean, default False): if True and masked is True, elements whose value is the null value
            (NaN for floats) will be masked out

      returns:
         numpy array

      :meta common:
      """

      return self.collection.cached_part_array_ref(self.part,
                                                   dtype = dtype,
                                                   masked = masked,
                                                   exclude_null = exclude_null)


#   def citation_title(self):
#      return self.collection.citation_title_for_part(self.part)

   def is_continuous(self):
      """Returns boolean indicating that the property contains continuous (ie. float) data.

      :meta common:
      """
      return self.collection.continuous_for_part(self.part)

   def is_categorical(self):
      """Returns boolean indicating that the property contains categorical data.

      :meta common:
      """
      return self.collection.part_is_categorical(self.part)

   def null_value(self):
      """Returns int being the null value for the (discrete) property."""
      return self.collection.null_value_for_part(self.part)

   def count(self):
      """Returns int being the number of values for each element (usually 1)."""
      return self.collection.count_for_part(self.part)

   def indexable_element(self):
      """Returns string being the indexable element for the property.

      :meta common:
      """
      return self.collection.indexable_for_part(self.part)

   def property_kind(self):
      """Returns string being the property kind for the property.

      :meta common:
      """
      return self.collection.property_kind_for_part(self.part)

   def local_property_kind_uuid(self):
      """Returns uuid of the local property kind for the property (if applicable, otherwise None)."""
      return self.collection.local_property_kind_uuid(self.part)

   def facet_type(self):
      """Returns string being the facet type for the property, or None if no facet present.

      note: resqpy currently supports at most one facet per property, though RESQML allows for multiple.

      :meta common:
      """
      return self.collection.facet_type_for_part(self.part)

   def facet(self):
      """Returns string being the facet value for the property, or None if no facet present.

      note: resqpy currently supports at most one facet per property, though RESQML allows for multiple.

      :meta common:
      """
      return self.collection.facet_for_part(self.part)

   def time_series_uuid(self):
      """Returns the uuid of the related time series (if applicable, otherwise None)."""
      return self.collection.time_series_uuid_for_part(self.part)

   def time_index(self):
      """Returns int being the index into the related time series (if applicable, otherwise None)."""
      return self.collection.time_index_for_part(self.part)

   def minimum_value(self):
      """Returns the minimum value for the property as stored in xml (float or int or None)."""
      return self.collection.minimum_value_for_part(self.part)

   def maximum_value(self):
      """Returns the maximum value for the property as stored in xml (float or int or None)."""
      return self.collection.maximum_value_for_part(self.part)

   def uom(self):
      """Returns string being the units of measure (for a continuous property, otherwise None).

      :meta common:
      """
      return self.collection.uom_for_part(self.part)

   def string_lookup_uuid(self):
      """Returns the uuid of the related string lookup table (for a categorical property, otherwise None)."""
      return self.collection.string_lookup_uuid_for_part(self.part)

   def string_lookup(self):
      """Returns a resqpy StringLookup table (for a categorical property, otherwise None)."""
      return self.collection.string_lookup_for_part(self.part)

   def constant_value(self):
      """For a constant property, returns the constant value (float or int or bool, or None if not constant)."""
      return self.collection.constant_value_for_part(self.part)

   def prepare_import(self,
                      cached_array,
                      source_info,
                      keyword,
                      discrete = False,
                      uom = None,
                      time_index = None,
                      null_value = None,
                      property_kind = None,
                      local_property_kind_uuid = None,
                      facet_type = None,
                      facet = None,
                      realization = None,
                      indexable_element = None,
                      count = 1,
                      const_value = None):
      """Takes a numpy array and metadata and sets up a single array import list; not usually called directly.

      note:
         see the documentation for the convenience method from_array()
      """
      assert self.collection.number_of_parts() == 0
      assert not self.collection.imported_list
      self.collection.add_cached_array_to_imported_list(cached_array,
                                                        source_info,
                                                        keyword,
                                                        discrete = discrete,
                                                        uom = uom,
                                                        time_index = time_index,
                                                        null_value = null_value,
                                                        property_kind = property_kind,
                                                        local_property_kind_uuid = local_property_kind_uuid,
                                                        facet_type = facet_type,
                                                        facet = facet,
                                                        realization = realization,
                                                        indexable_element = indexable_element,
                                                        count = count,
                                                        const_value = const_value)

   def write_hdf5(self, file_name = None, mode = 'a'):
      """Writes the array data to the hdf5 file; not usually called directly.

      notes:
         see the documentation for the convenience method from_array();
      """
      if not self.collection.imported_list:
         log.warning('no imported Property array to write to hdf5')
         return
      self.collection.write_hdf5_for_imported_list(file_name = file_name, mode = mode)

   def create_xml(self,
                  ext_uuid = None,
                  support_uuid = None,
                  time_series_uuid = None,
                  string_lookup_uuid = None,
                  property_kind_uuid = None,
                  find_local_property_kind = True,
                  extra_metadata = {}):
      """Creates an xml tree for the property and adds it as a part to the model; not usually called directly.

      note:
         see the documentation for the convenience method from_array()
         NB. this method has the deliberate side effect of modifying the uuid and title of self to match those
         of the property collection part!
      """
      if not self.collection.imported_list:
         log.warning('no imported Property array to create xml for')
         return
      self.collection.create_xml_for_imported_list_and_add_parts_to_model(
         ext_uuid = ext_uuid,
         support_uuid = support_uuid,
         time_series_uuid = time_series_uuid,
         selected_time_indices_list = None,
         string_lookup_uuid = string_lookup_uuid,
         property_kind_uuid = property_kind_uuid,
         find_local_property_kinds = find_local_property_kind,
         extra_metadata = extra_metadata)
      self.collection.has_single_property_kind_flag = True
      self.collection.has_single_uom_flag = True
      self.collection.has_single_indexable_element_flag = True
      self.collection.has_multiple_realizations_flag = False
      part = self.collection.parts()[0]
      assert part is not None
      self.uuid = self.collection.uuid_for_part(part)
      self.title = self.collection.citation_title_for_part(part)
      return self.root


class GridPropertyCollection(PropertyCollection):
   """Class for RESQML Property collection for an IJK Grid, inheriting from PropertyCollection."""

   def __init__(self, grid = None, property_set_root = None, realization = None):
      """Creates a new property collection related to an IJK grid.

      arguments:
         grid (grid.Grid object, optional): must be present unless creating a completely blank, empty collection
         property_set_root (optional): if present, the collection is populated with the properties defined in the xml tree
            of the property set; grid must not be None when using this argument
         realization (integer, optional): if present, the single realisation (within an ensemble) that this collection is for;
            if None, then the collection is either covering a whole ensemble (individual properties can each be flagged with a
            realisation number), or is for properties that do not have multiple realizations

      returns:
         the new GridPropertyCollection object

      note:
         usually a grid should be passed, however a completely blank collection may be created prior to using
         collection inheritance methods to populate from another collection, in which case the grid can be lazily left
         as None here

      :meta common:
      """

      if grid is not None:
         log.debug('initialising grid property collection for grid: ' + str(rqet.citation_title_for_node(grid.root)))
         log.debug('grid uuid: ' + str(grid.uuid))
      super().__init__(support = grid, property_set_root = property_set_root, realization = realization)
      self._copy_support_to_grid_attributes()

      # NB: RESQML documentation is not clear which order is correct; should be kept consistent with same data in fault.py
      # face_index_map maps from (axis, p01) to face index value in range 0..5
      #     self.face_index_map = np.array([[0, 1], [4, 2], [5, 3]], dtype = int)
      self.face_index_map = np.array([[0, 1], [2, 4], [5, 3]], dtype = int)  # order: top, base, J-, I+, J+, I-
      # and the inverse, maps from 0..5 to (axis, p01)
      #     self.face_index_inverse_map = np.array([[0, 0], [0, 1], [1, 1], [2, 1], [1, 0], [2, 0]], dtype = int)
      self.face_index_inverse_map = np.array([[0, 0], [0, 1], [1, 0], [2, 1], [1, 1], [2, 0]], dtype = int)

   def _copy_support_to_grid_attributes(self):
      # following three pseudonyms are for backward compatibility
      self.grid = self.support
      self.grid_root = self.support_root
      self.grid_uuid = self.support_uuid

   def set_grid(self, grid, grid_root = None, modify_parts = True):
      """Sets the supporting representation object for the collection to be the given grid.

      note:
         this method does not need to be called if the grid was passed during object initialisation.
      """

      self.set_support(support = grid, modify_parts = modify_parts)
      self._copy_support_to_grid_attributes()

   def h5_slice_for_box(self, part, box):
      """Returns a subset of the array for part, without loading the whole array (unless already cached).

      arguments:
         part (string): the part name for which the array slice is required
         box (numpy int array of shape (2, 3)): the min, max indices for K, J, I axes for which an array
            extract is required

      returns:
         numpy array that is a hyper-slice of the hdf5 array

      note:
         this method always fetches from the hdf5 file and does not attempt local caching; the whole array
         is not loaded; all axes continue to exist in the returned array, even where the sliced extent of
         an axis is 1; the upper indices indicated in the box are included in the data (unlike the python
         protocol)
      """

      slice_tuple = (slice(box[0, 0], box[1, 0] + 1), slice(box[0, 1], box[1, 1] + 1), slice(box[0, 2], box[1, 2] + 1))
      return self.h5_slice(part, slice_tuple)

   def extend_imported_list_copying_properties_from_other_grid_collection(self,
                                                                          other,
                                                                          box = None,
                                                                          refinement = None,
                                                                          coarsening = None,
                                                                          realization = None,
                                                                          copy_all_realizations = False,
                                                                          uncache_other_arrays = True):
      """Extends this collection's imported list with properties from other collection, optionally extracting for a box.

      arguments:
         other: another GridPropertyCollection object which might relate to a different grid object
         box: (numpy int array of shape (2, 3), optional): if present, a logical ijk cuboid subset of the source arrays
            is extracted, box axes are (min:max, kji0); if None, the full arrays are copied
         refinement (resqpy.olio.fine_coarse.FineCoarse object, optional): if present, other is taken to be a collection
            for a coarser grid and the property values are sampled for a finer grid based on the refinement mapping
         coarsening (resqpy.olio.fine_coarse.FineCoarse object, optional): if present, other is taken to be a collection
            for a finer grid and the property values are upscaled for a coarser grid based on the coarsening mapping
         realization (int, optional): if present, only properties for this realization are copied; if None, only
            properties without a realization number are copied unless copy_all_realizations is True
         copy_all_realizations (boolean, default False): if True (and realization is None), all properties are copied;
            if False, only properties with a realization of None are copied; ignored if realization is not None
         uncache_other_arrays (boolean, default True): if True, after each array is copied, the original is uncached from
            the source collection (to free up memory)

      notes:
         this function can be used to copy properties between one grid object and another compatible one, for example
         after a grid manipulation has generated a new version of the geometry; it can also be used to select an ijk box
         subset of the property data and/or refine or coarsen the property data;
         if a box is supplied, the first index indicates min (0) or max (1) and the second index indicates k (0), j (1) or i (2);
         the values in box are zero based and cells matching the maximum indices are included (unlike with python ranges);
         when coarsening or refining, this function ignores the 'within...box' attributes of the FineCoarse object so
         the box argument must be used to effect a local grid coarsening or refinement
      """

      def array_box(collection, part, box = None, uncache_other_arrays = True):
         full_array = collection.cached_part_array_ref(part)
         if box is None:
            a = full_array.copy()
         else:
            a = full_array[box[0, 0]:box[1, 0] + 1, box[0, 1]:box[1, 1] + 1, box[0, 2]:box[1, 2] + 1].copy()
         full_array = None
         if uncache_other_arrays:
            other.uncache_part_array(part)
         return a

      def coarsening_sum(coarsening, a):
         a_coarsened = np.empty(tuple(coarsening.coarse_extent_kji))
         assert a.shape == tuple(coarsening.fine_extent_kji)
         # todo: try to figure out some numpy slice operations to avoid use of for loops
         for k in range(coarsening.coarse_extent_kji[0]):
            for j in range(coarsening.coarse_extent_kji[1]):
               for i in range(coarsening.coarse_extent_kji[2]):
                  cell_box = coarsening.fine_box_for_coarse(
                     (k, j, i))  # local box within lgc space of fine cells, for 1 coarse cell
                  a_coarsened[k, j, i] = np.nansum(a[cell_box[0, 0]:cell_box[1, 0] + 1,
                                                     cell_box[0, 1]:cell_box[1, 1] + 1, cell_box[0,
                                                                                                 2]:cell_box[1, 2] + 1])
         return a_coarsened

      def coarsening_weighted_mean(coarsening, a, fine_weight, coarse_weight = None, zero_weight_result = np.NaN):
         a_coarsened = np.empty(tuple(coarsening.coarse_extent_kji))
         assert a.shape == tuple(coarsening.fine_extent_kji)
         assert fine_weight.shape == a.shape
         if coarse_weight is not None:
            assert coarse_weight.shape == a_coarsened.shape
         for k in range(coarsening.coarse_extent_kji[0]):
            for j in range(coarsening.coarse_extent_kji[1]):
               for i in range(coarsening.coarse_extent_kji[2]):
                  cell_box = coarsening.fine_box_for_coarse(
                     (k, j, i))  # local box within lgc space of fine cells, for 1 coarse cell
                  a_coarsened[k, j, i] = np.nansum(
                     a[cell_box[0, 0]:cell_box[1, 0] + 1, cell_box[0, 1]:cell_box[1, 1] + 1,
                       cell_box[0, 2]:cell_box[1, 2] + 1] *
                     fine_weight[cell_box[0, 0]:cell_box[1, 0] + 1, cell_box[0, 1]:cell_box[1, 1] + 1,
                                 cell_box[0, 2]:cell_box[1, 2] + 1])
                  if coarse_weight is None:
                     weight = np.nansum(fine_weight[cell_box[0, 0]:cell_box[1, 0] + 1,
                                                    cell_box[0, 1]:cell_box[1, 1] + 1, cell_box[0,
                                                                                                2]:cell_box[1, 2] + 1])
                     if np.isnan(weight) or weight == 0.0:
                        a_coarsened[k, j, i] = zero_weight_result
                     else:
                        a_coarsened[k, j, i] /= weight
         if coarse_weight is not None:
            mask = np.logical_or(np.isnan(coarse_weight), coarse_weight == 0.0)
            a_coarsened = np.where(mask, zero_weight_result, a_coarsened / coarse_weight)
         return a_coarsened

      def add_to_imported(collection, a, title, info, null_value = None, const_value = None):
         collection.add_cached_array_to_imported_list(
            a,
            title,
            info[10],  # citation_title
            discrete = not info[4],
            indexable_element = 'cells',
            uom = info[15],
            time_index = info[12],
            null_value = null_value,
            property_kind = info[7],
            local_property_kind_uuid = info[17],
            facet_type = info[8],
            facet = info[9],
            realization = info[0],
            const_value = const_value)

      import resqpy.grid as grr  # at global level was causing issues due to circular references, ie. grid importing this module

      assert other.support is not None and isinstance(other.support,
                                                      grr.Grid), 'other property collection has no grid support'
      assert refinement is None or coarsening is None, 'refinement and coarsening both specified simultaneously'

      if box is not None:
         assert bxu.valid_box(box, other.grid.extent_kji)
         if refinement is not None:
            assert tuple(bxu.extent_of_box(box)) == tuple(refinement.coarse_extent_kji)
         elif coarsening is not None:
            assert tuple(bxu.extent_of_box(box)) == tuple(coarsening.fine_extent_kji)
      # todo: any contraints on realization numbers ?

      if coarsening is not None:  # static upscaling of key property kinds, simple sampling of others

         assert self.support is not None and tuple(self.support.extent_kji) == tuple(coarsening.coarse_extent_kji)

         # look for properties by kind, process in order: rock volume, net to gross ratio, porosity, permeability, saturation
         source_rv = selective_version_of_collection(other, realization = realization, property_kind = 'rock volume')
         source_ntg = selective_version_of_collection(other,
                                                      realization = realization,
                                                      property_kind = 'net to gross ratio')
         source_poro = selective_version_of_collection(other, realization = realization, property_kind = 'porosity')
         source_sat = selective_version_of_collection(other, realization = realization, property_kind = 'saturation')
         source_perm = selective_version_of_collection(other,
                                                       realization = realization,
                                                       property_kind = 'permeability rock')
         # todo: add kh and some other property kinds

         # bulk rock volume
         fine_rv_array = coarse_rv_array = None
         if source_rv.number_of_parts() == 0:
            log.debug('computing bulk rock volume from fine and coarse grid geometries')
            source_rv_array = other.support.volume()
            if box is None:
               fine_rv_array = source_rv_array
            else:
               fine_rv_array = source_rv_array[box[0, 0]:box[1, 0] + 1, box[0, 1]:box[1, 1] + 1, box[0,
                                                                                                     2]:box[1, 2] + 1]
            coarse_rv_array = self.support.volume()
         else:
            for (part, info) in source_rv.dict.items():
               if not copy_all_realizations and info[0] != realization:
                  continue
               fine_rv_array = array_box(other, part, box = box, uncache_other_arrays = uncache_other_arrays)
               coarse_rv_array = coarsening_sum(coarsening, fine_rv_array)
               add_to_imported(self, coarse_rv_array, 'coarsened from grid ' + str(other.support.uuid), info)

         # net to gross ratio
         # note that coarsened ntg values may exceed one when reference bulk rock volumes are from grid geometries
         fine_ntg_array = coarse_ntg_array = None
         for (part, info) in source_ntg.dict.items():
            if not copy_all_realizations and info[0] != realization:
               continue
            fine_ntg_array = array_box(other, part, box = box, uncache_other_arrays = uncache_other_arrays)
            coarse_ntg_array = coarsening_weighted_mean(coarsening,
                                                        fine_ntg_array,
                                                        fine_rv_array,
                                                        coarse_weight = coarse_rv_array,
                                                        zero_weight_result = 0.0)
            add_to_imported(self, coarse_ntg_array, 'coarsened from grid ' + str(other.support.uuid), info)

         if fine_ntg_array is None:
            fine_nrv_array = fine_rv_array
            coarse_nrv_array = coarse_rv_array
         else:
            fine_nrv_array = fine_rv_array * fine_ntg_array
            coarse_nrv_array = coarse_rv_array * coarse_ntg_array

         fine_poro_array = coarse_poro_array = None
         for (part, info) in source_poro.dict.items():
            if not copy_all_realizations and info[0] != realization:
               continue
            fine_poro_array = array_box(other, part, box = box, uncache_other_arrays = uncache_other_arrays)
            coarse_poro_array = coarsening_weighted_mean(coarsening,
                                                         fine_poro_array,
                                                         fine_nrv_array,
                                                         coarse_weight = coarse_nrv_array,
                                                         zero_weight_result = 0.0)
            add_to_imported(self, coarse_poro_array, 'coarsened from grid ' + str(other.support.uuid), info)

         # saturations
         fine_sat_array = coarse_sat_array = None
         fine_sat_weight = fine_nrv_array
         coarse_sat_weight = coarse_nrv_array
         if fine_poro_array is not None:
            fine_sat_weight *= fine_poro_array
            coarse_sat_weight *= coarse_poro_array
         for (part, info) in source_sat.dict.items():
            if not copy_all_realizations and info[0] != realization:
               continue
            fine_sat_array = array_box(other, part, box = box, uncache_other_arrays = uncache_other_arrays)
            coarse_sat_array = coarsening_weighted_mean(coarsening,
                                                        fine_sat_array,
                                                        fine_sat_weight,
                                                        coarse_weight = coarse_sat_weight,
                                                        zero_weight_result = 0.0)
            add_to_imported(self, coarse_sat_array, 'coarsened from grid ' + str(other.support.uuid), info)

         # permeabilities
         # todo: use more harmonic, arithmetic mean instead of just bulk rock volume weighted; consider ntg
         for (part, info) in source_perm.dict.items():
            if not copy_all_realizations and info[0] != realization:
               continue
            fine_perm_array = array_box(other, part, box = box, uncache_other_arrays = uncache_other_arrays)
            coarse_perm_array = coarsening_weighted_mean(coarsening,
                                                         fine_perm_array,
                                                         fine_nrv_array,
                                                         coarse_weight = coarse_nrv_array,
                                                         zero_weight_result = 0.0)
            add_to_imported(self, coarse_perm_array, 'coarsened from grid ' + str(other.support.uuid), info)

         # TODO: all other supported property kinds
         # default behaviour simply sample the first fine cell in the coarse cell box

      else:

         if realization is None:
            source_collection = other
         else:
            source_collection = selective_version_of_collection(other, realization = realization)

         for (part, info) in source_collection.dict.items():
            if not copy_all_realizations and info[0] != realization:
               continue

            const_value = info[20]
            if const_value is None:
               a = array_box(source_collection, part, box = box, uncache_other_arrays = uncache_other_arrays)
            else:
               a = None

            if refinement is not None and a is not None:  # simple resampling
               if info[6] != 'cells':
                  # todo: appropriate refinement of data for other indexable elements
                  continue
               # todo: dividing up of values when needed, eg. volumes, areas, lengths
               assert tuple(a.shape) == tuple(refinement.coarse_extent_kji)
               assert self.support is not None and tuple(self.support.extent_kji) == tuple(refinement.fine_extent_kji)
               k_ratio_vector = refinement.coarse_for_fine_axial_vector(0)
               a_refined_k = np.empty(
                  (refinement.fine_extent_kji[0], refinement.coarse_extent_kji[1], refinement.coarse_extent_kji[2]),
                  dtype = a.dtype)
               a_refined_k[:, :, :] = a[k_ratio_vector, :, :]
               j_ratio_vector = refinement.coarse_for_fine_axial_vector(1)
               a_refined_kj = np.empty(
                  (refinement.fine_extent_kji[0], refinement.fine_extent_kji[1], refinement.coarse_extent_kji[2]),
                  dtype = a.dtype)
               a_refined_kj[:, :, :] = a_refined_k[:, j_ratio_vector, :]
               i_ratio_vector = refinement.coarse_for_fine_axial_vector(2)
               a = np.empty(tuple(refinement.fine_extent_kji), dtype = a.dtype)
               a[:, :, :] = a_refined_kj[:, :, i_ratio_vector]

            self.add_cached_array_to_imported_list(
               a,
               'copied from grid ' + str(other.support.uuid),
               info[10],  # citation_title
               discrete = not info[4],
               indexable_element = 'cells',
               uom = info[15],
               time_index = info[12],
               null_value = None,  # todo: extract from other's xml
               property_kind = info[7],
               local_property_kind_uuid = info[17],
               facet_type = info[8],
               facet = info[9],
               realization = info[0],
               const_value = const_value)

   def import_nexus_property_to_cache(self,
                                      file_name,
                                      keyword,
                                      extent_kji = None,
                                      discrete = False,
                                      uom = None,
                                      time_index = None,
                                      null_value = None,
                                      property_kind = None,
                                      local_property_kind_uuid = None,
                                      facet_type = None,
                                      facet = None,
                                      realization = None,
                                      use_binary = True):
      """Reads a property array from an ascii (or pure binary) file, caches and adds to imported list (but not collection dict).

      arguments:
         file_name (string): the name of the file to read the array data from; should contain data for one array only, without
            the keyword
         keyword (string): the keyword to associate with the imported data, which will become the citation title
         extent_kji (optional, default None): if present, [nk, nj, ni] being the extent (shape) of the array to be imported;
            if None, the shape of the grid associated with this collection is used
         discrete (boolean, optional, default False): if True, integer data is imported, if False, float data
         uom (string, optional, default None): the resqml units of measure applicable to the data
         time_index (integer, optional, default None): if present, this array is for time varying data and this value is the
            index into a time series associated with this collection
         null_value (int or float, optional, default None): if present, this is used in the metadata to indicate that
            this value is to be interpreted as a null value wherever it appears in the data (this does not change the
            data during import)
         property_kind (string): resqml property kind, or None
         local_property_kind_uuid (uuid.UUID or string): uuid of local property kind, or None for standard property kind
         facet_type (string): resqml facet type, or None
         facet (string): resqml facet, or None
         realization (int): realization number, or None
         use_binary (boolean, optional, default True): if True, and an up-to-date pure binary version of the file exists,
            then the data is loaded from that instead of from the ascii file; if True but the binary version does not
            exist (or is older than the ascii file), the pure binary version is written as a side effect of the import;
            if False, the ascii file is used even if a pure binary equivalent exists, and no binary file is written

      note:
         this function only performs the first importation step of actually reading the array into memory; other steps
         must follow to include the array as a part in the resqml model and in this collection of properties (see doc
         string for add_cached_array_to_imported_list() for the sequence of steps)
      """

      log.debug(f'importing {keyword} array from file {file_name}')
      # note: code in resqml_import adds imported arrays to model and to dict
      if self.imported_list is None:
         self.imported_list = []
      if extent_kji is None:
         extent_kji = self.grid.extent_kji
      if discrete:
         data_type = 'integer'
      else:
         data_type = 'real'
      try:
         import_array = ld.load_array_from_file(file_name,
                                                extent_kji,
                                                data_type = data_type,
                                                comment_char = '!',
                                                data_free_of_comments = False,
                                                use_binary = use_binary)
      except Exception:
         log.exception('failed to import {} arrau from file {}'.format(keyword, file_name))
         return None

      self.add_cached_array_to_imported_list(import_array,
                                             file_name,
                                             keyword,
                                             discrete = discrete,
                                             uom = uom,
                                             time_index = time_index,
                                             null_value = null_value,
                                             property_kind = property_kind,
                                             local_property_kind_uuid = local_property_kind_uuid,
                                             facet_type = facet_type,
                                             facet = facet,
                                             realization = realization)
      return import_array

   def import_vdb_static_property_to_cache(self, vdbase, keyword, grid_name = 'ROOT', uom = None, realization = None):
      """Reads a vdb static property array, caches and adds to imported list (but not collection dict).

      arguments:
         vdbase: an object of class vdb.VDB, already initialised with the path of the vdb
         keyword (string): the Nexus keyword (or equivalent) of the static property to be loaded
         grid_name (string): the grid name as used in the vdb
         uom (string): The resqml unit of measure that applies to the data
         realization (optional, int): The realization number that this property belongs to; use None
            if not applicable

      returns:
         cached array containing the property data; the cached array is in an unpacked state
         (ie. can be directly indexed with [k0, j0, i0])

      note:
         when importing from a vdb (or other sources), use methods such as this to build up a list
         of imported arrays; then write hdf5 for the imported arrays; finally create xml for imported
         properties
      """

      log.info('importing vdb static property {} array'.format(keyword))
      keyword = keyword.upper()
      try:
         discrete = True
         dtype = None
         if keyword[0].upper() == 'I' or keyword in ['KID', 'UID', 'UNPACK', 'DAD']:
            # coerce to integer values (vdb stores integer data as reals!)
            dtype = 'int32'
         elif keyword in ['DEADCELL', 'LIVECELL']:
            dtype = 'bool'  # could use the default dtype of 64 bit integer
         else:
            dtype = 'float'  # convert to 64 bit; could omit but RESQML states 64 bit
            discrete = False
         import_array = vdbase.grid_static_property(grid_name, keyword, dtype = dtype)
         assert import_array is not None
      except Exception:
         log.exception(f'failed to import static property {keyword} from vdb')
         return None

      self.add_cached_array_to_imported_list(import_array,
                                             vdbase.path,
                                             keyword,
                                             discrete = discrete,
                                             uom = uom,
                                             time_index = None,
                                             null_value = None,
                                             realization = realization)
      return import_array

   def import_vdb_recurrent_property_to_cache(self,
                                              vdbase,
                                              timestep,
                                              keyword,
                                              grid_name = 'ROOT',
                                              time_index = None,
                                              uom = None,
                                              realization = None):
      """Reads a vdb recurrent property array for one timestep, caches and adds to imported list (but not collection dict).

      arguments:
         vdbase: an object of class vdb.VDB, already initialised with the path of the vdb
         timestep (int): The Nexus timestep number at which the property array was generated; NB. this is
            not necessarily the same as a resqml time index
         keyword (string): the Nexus keyword (or equivalent) of the recurrent property to be loaded
         grid_name (string): the grid name as used in the vdb
         time_index (int, optional): if present, used as the time index, otherwise timestep is used
         uom (string): The resqml unit of measure that applies to the data
         realization (optional, int): The realization number that this property belongs to; use None
            if not applicable

      returns:
         cached array containing the property data; the cached array is in an unpacked state
         (ie. can be directly indexed with [k0, j0, i0])

      notes:
         when importing from a vdb (or other sources), use methods such as this to build up a list
         of imported arrays; then write hdf5 for the imported arrays; finally create xml for imported
         properties
      """

      log.info('importing vdb recurrent property {0} array for timestep {1}'.format(keyword, str(timestep)))
      if time_index is None:
         time_index = timestep
      keyword = keyword.upper()
      try:
         import_array = vdbase.grid_recurrent_property_for_timestep(grid_name, keyword, timestep, dtype = 'float')
         assert import_array is not None
      except Exception:
         # could raise an exception (as for static properties)
         log.error(f'failed to import recurrent property {keyword} from vdb for timestep {timestep}')
         return None

      self.add_cached_array_to_imported_list(import_array,
                                             vdbase.path,
                                             keyword,
                                             discrete = False,
                                             uom = uom,
                                             time_index = time_index,
                                             null_value = None,
                                             realization = realization)
      return import_array

   def import_ab_property_to_cache(self,
                                   file_name,
                                   keyword,
                                   extent_kji = None,
                                   discrete = None,
                                   uom = None,
                                   time_index = None,
                                   null_value = None,
                                   property_kind = None,
                                   local_property_kind_uuid = None,
                                   facet_type = None,
                                   facet = None,
                                   realization = None):
      """Reads a property array from a pure binary file, caches and adds to imported list (but not collection dict).

      arguments:
         file_name (string): the name of the file to read the array data from; should contain data for one array only in
            'pure binary' format (as used by ab_* suite of utilities)
         keyword (string): the keyword to associate with the imported data, which will become the citation title
         extent_kji (optional, default None): if present, [nk, nj, ni] being the extent (shape) of the array to be imported;
            if None, the shape of the grid associated with this collection is used
         discrete (boolean, optional, default False): if True, integer data is imported, if False, float data
         uom (string, optional, default None): the resqml units of measure applicable to the data
         time_index (integer, optional, default None): if present, this array is for time varying data and this value is the
            index into a time series associated with this collection
         null_value (int or float, optional, default None): if present, this is used in the metadata to indicate that
            this value is to be interpreted as a null value wherever it appears in the data (this does not change the
            data during import)
         property_kind (string): resqml property kind, or None
         local_property_kind_uuid (uuid.UUID or string): uuid of local property kind, or None
         facet_type (string): resqml facet type, or None
         facet (string): resqml facet, or None
         realization (int): realization number, or None

      note:
         this function only performs the first importation step of actually reading the array into memory; other steps
         must follow to include the array as a part in the resqml model and in this collection of properties (see doc
         string for add_cached_array_to_imported_list() for the sequence of steps)
      """

      if self.imported_list is None:
         self.imported_list = []
      if extent_kji is None:
         extent_kji = self.grid.extent_kji
      assert file_name[-3:] in ['.db', '.fb', '.ib', '.lb',
                                '.bb'], 'file extension not in pure binary array expected set for: ' + file_name
      if discrete is None:
         discrete = (file_name[-3:] in ['.ib', '.lb', '.bb'])
      else:
         assert discrete == (file_name[-3:]
                             in ['.ib', '.lb',
                                 '.bb']), 'discrete argument is not consistent with file extension for: ' + file_name
      try:
         import_array = abt.load_array_from_ab_file(
            file_name, extent_kji, return_64_bit = False)  # todo: RESQML indicates 64 bit for everything
      except Exception:
         log.exception('failed to import property from pure binary file: ' + file_name)
         return None

      self.add_cached_array_to_imported_list(import_array,
                                             file_name,
                                             keyword,
                                             discrete = discrete,
                                             uom = uom,
                                             time_index = time_index,
                                             null_value = null_value,
                                             property_kind = property_kind,
                                             local_property_kind_uuid = local_property_kind_uuid,
                                             facet_type = facet_type,
                                             facet = facet,
                                             realization = realization)
      return import_array

   def decoarsen_imported_list(self, decoarsen_array = None, reactivate = True):
      """Decoarsen imported Nexus properties if needed.

      arguments:
         decoarsen_array (int array, optional): if present, the naturalised cell index of the coarsened host cell, for each fine cell;
            if None, the ICOARS keyword is searched for in the imported list and if not found KID data is used to derive the mapping
         reactivate (boolean, default True): if True, the parent grid will have decoarsened cells' inactive flag set to that of the
            host cell

      returns:
         a copy of the array used for decoarsening, if established, or None if no decoarsening array was identified

      notes:
         a return value of None indicates that no decoarsening occurred;
         coarsened values are redistributed quite naively, with coarse volumes being split equally between fine cells, similarly for
         length and area based properties; default used for most properties is simply to replicate the coarse value;
         the ICOARS array itself is left unchanged, which means the method should only be called once for an imported list;
         if no array is passed and no ICOARS array found, the KID values are inspected and the decoarsen array reverse engineered;
         the method must be called before the imported arrays are written to hdf5;
         reactivation only modifies the grid object attribute and does not write to hdf5, so the method should be called prior to
         writing the grid in this situation
      """
      # imported_list is list pf:
      # (0: uuid, 1: file_name, 2: keyword, 3: cached_name, 4: discrete, 5: uom, 6: time_index, 7: null_value, 8: min_value, 9: max_value,
      # 10: property_kind, 11: facet_type, 12: facet, 13: realization, 14: indexable_element, 15: count, 16: local_property_kind_uuid,
      # 17: const_value)

      skip_keywords = ['UID', 'ICOARS', 'KID', 'DAD']  # TODO: complete this list
      decoarsen_length_kinds = ['length', 'cell length', 'thickness', 'permeability thickness', 'permeability length']
      decoarsen_area_kinds = ['transmissibility']
      decoarsen_volume_kinds = ['volume', 'rock volume', 'pore volume', 'fluid volume']

      assert self.grid is not None

      kid_attr_name = None
      k_share = j_share = i_share = None

      if decoarsen_array is None:
         for import_item in self.imported_list:
            if (import_item[14] is None or import_item[14] == 'cells') and import_item[4] and hasattr(
                  self, import_item[3]):
               if import_item[2] == 'ICOARS':
                  decoarsen_array = self.__dict__[import_item[3]] - 1  # ICOARS values are one based
                  break
               if import_item[2] == 'KID':
                  kid_attr_name = import_item[3]

      if decoarsen_array is None and kid_attr_name is not None:
         kid = self.__dict__[kid_attr_name]
         kid_mask = (kid == -3)  # -3 indicates cell inactive due to coarsening
         assert kid_mask.shape == tuple(self.grid.extent_kji)
         if np.any(kid_mask):
            log.debug(f'{np.count_nonzero(kid_mask)} cells marked as requiring decoarsening in KID data')
            decoarsen_array = np.full(self.grid.extent_kji, -1, dtype = int)
            k_share = np.zeros(self.grid.extent_kji, dtype = int)
            j_share = np.zeros(self.grid.extent_kji, dtype = int)
            i_share = np.zeros(self.grid.extent_kji, dtype = int)
            natural = 0
            for k0 in range(self.grid.nk):
               for j0 in range(self.grid.nj):
                  for i0 in range(self.grid.ni):
                     #                     if decoarsen_array[k0, j0, i0] < 0:
                     if kid[k0, j0, i0] == 0:
                        #                        assert not kid_mask[k0, j0, i0]
                        ke = k0 + 1
                        while ke < self.grid.nk and kid_mask[ke, j0, i0]:
                           ke += 1
                        je = j0 + 1
                        while je < self.grid.nj and kid_mask[k0, je, i0]:
                           je += 1
                        ie = i0 + 1
                        while ie < self.grid.ni and kid_mask[k0, j0, ie]:
                           ie += 1
                        # todo: check for conflict and resolve
                        decoarsen_array[k0:ke, j0:je, i0:ie] = natural
                        k_share[k0:ke, j0:je, i0:ie] = ke - k0
                        j_share[k0:ke, j0:je, i0:ie] = je - j0
                        i_share[k0:ke, j0:je, i0:ie] = ie - i0
                     elif not kid_mask[k0, j0, i0]:  # inactive for reasons other than coarsening
                        decoarsen_array[k0, j0, i0] = natural
                        k_share[k0, j0, i0] = 1
                        j_share[k0, j0, i0] = 1
                        i_share[k0, j0, i0] = 1
                     natural += 1
            assert np.all(decoarsen_array >= 0)

      if decoarsen_array is None:
         return None

      cell_count = decoarsen_array.size
      host_count = len(np.unique(decoarsen_array))
      log.debug(f'{host_count} of {cell_count} are hosts; difference is {cell_count - host_count}')
      assert cell_count == self.grid.cell_count()
      if np.all(decoarsen_array.flatten() == np.arange(cell_count, dtype = int)):
         return None  # identity array

      if k_share is None:
         sharing_needed = False
         for import_item in self.imported_list:
            kind = import_item[10]
            if kind in decoarsen_volume_kinds or kind in decoarsen_area_kinds or kind in decoarsen_length_kinds:
               sharing_needed = True
               break
         if sharing_needed:
            k_share = np.zeros(self.grid.extent_kji, dtype = int)
            j_share = np.zeros(self.grid.extent_kji, dtype = int)
            i_share = np.zeros(self.grid.extent_kji, dtype = int)
            natural = 0
            for k0 in range(self.grid.nk):
               for j0 in range(self.grid.nj):
                  for i0 in range(self.grid.ni):
                     if k_share[k0, j0, i0] == 0:
                        ke = k0 + 1
                        while ke < self.grid.nk and decoarsen_array[ke, j0, i0] == natural:
                           ke += 1
                        je = j0 + 1
                        while je < self.grid.nj and decoarsen_array[k0, je, i0] == natural:
                           je += 1
                        ie = i0 + 1
                        while ie < self.grid.ni and decoarsen_array[k0, j0, ie] == natural:
                           ie += 1
                        k_share[k0:ke, j0:je, i0:ie] = ke - k0
                        j_share[k0:ke, j0:je, i0:ie] = je - j0
                        i_share[k0:ke, j0:je, i0:ie] = ie - i0
                     natural += 1

      if k_share is not None:
         assert np.all(k_share > 0) and np.all(j_share > 0) and np.all(i_share > 0)
         volume_share = (k_share * j_share * i_share).astype(float)
         k_share = k_share.astype(float)
         j_share = j_share.astype(float)
         i_share = i_share.astype(float)

      property_count = 0
      for import_item in self.imported_list:
         if import_item[3] is None or not hasattr(self, import_item[3]):
            continue  # todo: handle decoarsening of const arrays?
         if import_item[14] is not None and import_item[14] != 'cells':
            continue
         coarsened = self.__dict__[import_item[3]].flatten()
         assert coarsened.size == cell_count
         keyword = import_item[2]
         if keyword.upper() in skip_keywords:
            continue
         kind = import_item[10]
         if kind in decoarsen_volume_kinds:
            redistributed = coarsened[decoarsen_array] / volume_share
         elif kind in decoarsen_area_kinds:
            # only transmissibilty currently in this set of supported property kinds
            log.warning(
               f'decoarsening of transmissibility {keyword} skipped due to simple methods not yielding correct values')
         elif kind in decoarsen_length_kinds:
            facet_dir = import_item[12] if import_item[11] == 'direction' else None
            if kind in ['thickness', 'permeability thickness'] or (facet_dir == 'K'):
               redistributed = coarsened[decoarsen_array] / k_share
            elif facet_dir == 'J':
               redistributed = coarsened[decoarsen_array] / j_share
            elif facet_dir == 'I':
               redistributed = coarsened[decoarsen_array] / i_share
            else:
               log.warning(f'decoarsening of length property {keyword} skipped as direction not established')
         else:
            redistributed = coarsened[decoarsen_array]
         self.__dict__[import_item[3]] = redistributed.reshape(self.grid.extent_kji)
         property_count += 1

      if property_count:
         log.debug(f'{property_count} properties decoarsened')

      if reactivate and hasattr(self.grid, 'inactive'):
         log.debug('reactivating cells inactive due to coarsening')
         pre_count = np.count_nonzero(self.grid.inactive)
         self.grid.inactive = self.grid.inactive.flatten()[decoarsen_array].reshape(self.grid.extent_kji)
         post_count = np.count_nonzero(self.grid.inactive)
         log.debug(f'{pre_count - post_count} cells reactivated')

      return decoarsen_array

   def write_nexus_property(
         self,
         part,
         file_name,
         keyword = None,
         headers = True,
         append = False,
         columns = 20,
         decimals = 3,  # note: decimals only applicable to real numbers
         blank_line_after_i_block = True,
         blank_line_after_j_block = False,
         space_separated = False,  # default is tab separated
         use_binary = False,
         binary_only = False,
         nan_substitute_value = None):
      """Writes the property array to a file in a format suitable for including as nexus input.

      arguments:
         part (string): the part name for which the array is to be exported
         file_name (string): the path of the file to be created (any existing file will be overwritten)
         keyword (string, optional, default None): if not None, the Nexus keyword to be included in the
            ascii export file (otherwise data only is written, without a keyword)
         headers (boolean, optional, default True): if True, some header comments are included in the
            ascii export file, using a Nexus comment character
         append (boolean, optional, default False): if True, any existing file is appended to rather than
            overwritten
         columns (integer, optional, default 20): the maximum number of data items to be written per line
         decimals (integer, optional, default 3): the number of decimal places included in the values
            written to the ascii export file (ignored for integer data)
         blank_line_after_i_block (boolean, optional, default True): if True, a blank line is inserted
            after each I-block of data (ie. when the J index changes)
         blank_line_after_j_block (boolean, optional, default False): if True, a blank line is inserted
            after each J-block of data (ie. when the K index changes)
         space_separated (boolean, optional, default False): if True, a space is inserted between values;
            if False, a tab is used
         use_binary (boolean, optional, default False): if True, a pure binary copy of the array is
            written
         binary_only (boolean, optional, default False): if True, and if use_binary is True, then no
            ascii file is generated; if False (or if use_binary is False) then an ascii file is written
         nan_substitute_value (float, optional, default None): if a value is supplied, any not-a-number
            values are replaced with this value in the exported file (the cached property array remains
            unchanged); if None, then 'nan' or 'Nan' will appear in the ascii export file
      """

      array_ref = self.cached_part_array_ref(part)
      assert (array_ref is not None)
      extent_kji = array_ref.shape
      assert (len(extent_kji) == 3)
      wd.write_array_to_ascii_file(file_name,
                                   extent_kji,
                                   array_ref,
                                   headers = headers,
                                   keyword = keyword,
                                   columns = columns,
                                   data_type = rqet.simplified_data_type(array_ref.dtype),
                                   decimals = decimals,
                                   target_simulator = 'nexus',
                                   blank_line_after_i_block = blank_line_after_i_block,
                                   blank_line_after_j_block = blank_line_after_j_block,
                                   space_separated = space_separated,
                                   append = append,
                                   use_binary = use_binary,
                                   binary_only = binary_only,
                                   nan_substitute_value = nan_substitute_value)

   def write_nexus_property_generating_filename(
         self,
         part,
         directory,
         use_title_for_keyword = False,
         headers = True,
         columns = 20,
         decimals = 3,  # note: decimals only applicable to real numbers
         blank_line_after_i_block = True,
         blank_line_after_j_block = False,
         space_separated = False,  # default is tab separated
         use_binary = False,
         binary_only = False,
         nan_substitute_value = None):
      """Writes the property array to a file using a filename generated from the citation title etc.

      arguments:
         part (string): the part name for which the array is to be exported
         directory (string): the path of the diractory into which the file will be written
         use_title_for_keyword (boolean, optional, default False): if True, the citation title for the property part
            is used as a keyword in the ascii export file
         for other arguments, see the docstring for the write_nexus_property() function

      note:
         the generated filename consists of the citation title (with spaces replaced with underscores);
         the facet type and facet, if present;
         _t_ and the time_index, if the part has a time index
      """

      title = self.citation_title_for_part(part).replace(' ', '_')
      if use_title_for_keyword:
         keyword = title
      else:
         keyword = None
      fname = title
      facet_type = self.facet_type_for_part(part)
      if facet_type is not None:
         fname += '_' + facet_type.replace(' ', '_') + '_' + self.facet_for_part(part).replace(' ', '_')
      time_index = self.time_index_for_part(part)
      if time_index is not None:
         fname += '_t_' + str(time_index)
      # could add .dat extension
      self.write_nexus_property(part,
                                os.path.join(directory, fname),
                                keyword = keyword,
                                headers = headers,
                                append = False,
                                columns = columns,
                                decimals = decimals,
                                blank_line_after_i_block = blank_line_after_i_block,
                                blank_line_after_j_block = blank_line_after_j_block,
                                space_separated = space_separated,
                                use_binary = use_binary,
                                binary_only = binary_only,
                                nan_substitute_value = nan_substitute_value)

   def write_nexus_collection(self,
                              directory,
                              use_title_for_keyword = False,
                              headers = True,
                              columns = 20,
                              decimals = 3,
                              blank_line_after_i_block = True,
                              blank_line_after_j_block = False,
                              space_separated = False,
                              use_binary = False,
                              binary_only = False,
                              nan_substitute_value = None):
      """Writes a set of files, one for each part in the collection.

      arguments:
         directory (string): the path of the diractory into which the files will be written
         for other arguments, see the docstrings for the write_nexus_property_generating_filename() and
            write_nexus_property() functions

      note:
         the generated filenames are based on the citation titles etc., as for
         write_nexus_property_generating_filename()
      """

      for part in self.dict.keys():
         self.write_nexus_property_generating_filename(part,
                                                       directory,
                                                       use_title_for_keyword = use_title_for_keyword,
                                                       headers = headers,
                                                       columns = columns,
                                                       decimals = decimals,
                                                       blank_line_after_i_block = blank_line_after_i_block,
                                                       blank_line_after_j_block = blank_line_after_j_block,
                                                       space_separated = space_separated,
                                                       use_binary = use_binary,
                                                       binary_only = binary_only,
                                                       nan_substitute_value = nan_substitute_value)


class WellLogCollection(PropertyCollection):
   """Class for RESQML Property collection for a Wellbore Frame (ie well logs), inheriting from PropertyCollection."""

   def __init__(self, frame = None, property_set_root = None, realization = None):
      """Creates a new property collection related to a wellbore frame.

      arguments:
         frame (well.WellboreFrame object, optional): must be present unless creating a completely blank, empty collection.
            See :class:`resqpy.well.WellboreFrame`
         property_set_root (optional): if present, the collection is populated with the properties defined in the xml tree
            of the property set; frame must not be None when using this argument
         realization (integer, optional): if present, the single realisation (within an ensemble) that this collection is for;
            if None, then the collection is either covering a whole ensemble (individual properties can each be flagged with a
            realisation number), or is for properties that do not have multiple realizations

      returns:
         the new WellLogCollection object

      note:
         usually a wellbore frame should be passed, however a completely blank collection may be created prior to using
         collection inheritance methods to populate from another collection, in which case the frame can be lazily left
         as None here;
         for actual well logs, the realization argument will usually be None; for synthetic logs created from an ensemble
         it may be of use

      """

      super().__init__(support = frame, property_set_root = property_set_root, realization = realization)

   def add_log(self, title, data, unit, discrete = False, realization = None, write = True):
      """Add a well log to the collection, and optionally save to HDF / XML
      
      Note:
         If write=False, the data are not written to the model and are saved to be written later.
         To write the data, you can subsequently call::

            logs.write_hdf5_for_imported_list()
            logs.create_xml_for_imported_list_and_add_parts_to_model()

      Args:
         title (str): Name of log, typically the mnemonic
         data (array-like): log data to write. Must have same length as frame MDs
         unit (str): Unit of measure
         discrete (bool): by default False, i.e. continuous
         realization (int): If given, assign data to a realisation.
         write (bool): If True, write XML and HDF5.

      Returns:
         uuids: list of uuids of newly added properties. Only returned if write=True.

      """
      # Validate
      if self.support is None:
         raise ValueError('Supporting WellboreFrame not present')
      if len(data) != self.support.node_count:
         raise ValueError(f'Data mismatch: data length={len(data)}, but MD node count={self.support.node_count}')

      # Infer valid RESQML properties
      # TODO: Store orginal unit somewhere if it's not a valid RESQML unit
      uom = bwam.rq_uom(unit)
      property_kind, facet_type, facet = infer_property_kind(title, uom)

      # Add to the "import list"
      self.add_cached_array_to_imported_list(
         cached_array = np.array(data),
         source_info = '',
         # TODO: put the curve.descr somewhere
         keyword = title,
         discrete = discrete,
         uom = uom,
         property_kind = property_kind,
         facet_type = facet_type,
         facet = facet,
         realization = realization,
      )

      if write:
         self.write_hdf5_for_imported_list()
         return self.create_xml_for_imported_list_and_add_parts_to_model()
      else:
         return None

   def iter_logs(self):
      """ Generator that yields component Log objects.
      
      Yields:
         instances of :class:`resqpy.property.WellLog` .

      Example::

         for log in log_collection.logs():
            print(log.title)
      """

      return (WellLog(collection = self, uuid = uuid) for uuid in self.uuids())

   def to_df(self, include_units = False):
      """ Return pandas dataframe of log data

      Args:
         include_units (bool): include unit in column names
      """

      assert self.support is not None

      # Get MD values
      md_values = self.support.node_mds
      assert md_values.ndim == 1, 'measured depths not 1D numpy array'

      # Get logs
      data = {}
      for log in self.iter_logs():

         col_name = log.title
         if include_units and log.uom:
            col_name += f' ({log.uom})'

         values = log.values()
         if values.ndim > 1:
            raise NotImplementedError('Multidimensional logs not yet supported in pandas')

         data[col_name] = values

      df = pd.DataFrame(data = data, index = md_values)
      return df

   def to_las(self):
      """ Return a lasio.LASFile object, which can then be written to disk

      Example::

         las = collection.to_las()
         las.write('example_logs.las', version=2)

      """
      las = lasio.LASFile()

      las.well.WELL = str(self.support.wellbore_interpretation.title)
      las.well.DATE = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

      # todo: Get UWI from somewhere
      # las.well.UWI = uwi

      # Lookup depths from associated WellboreFrame and Trajectory
      md_values = self.support.node_mds
      md_unit = self.support.trajectory.md_uom

      # Measured depths should be first column in LAS file
      # todo: include datum information in description
      las.append_curve('MD', md_values, unit = md_unit)

      for well_log in self.iter_logs():
         name = well_log.title
         unit = well_log.uom
         values = well_log.values()
         if values.ndim > 1:
            raise NotImplementedError('Multidimensional logs not yet supported in pandas')
         assert len(values) > 0
         log.debug(f"Writing log {name} of length {len(values)} and shape {values.shape}")
         las.append_curve(name, values, unit = unit, descr = None)
      return las

   def set_wellbore_frame(self, frame):
      """Sets the supporting representation object for the property collection to be the given wellbore frame object.

      note:
         this method does not need to be called if the wellbore frame was identified at the time the collection
         was initialised
      """

      self.set_support(support = frame)


class WellLog:
   """ Thin wrapper class around RESQML properties for well logs """

   def __init__(self, collection, uuid):
      """ Create a well log from a part name """

      self.collection: PropertyCollection = collection
      self.model = collection.model
      self.uuid = uuid

      part = self.model.part_for_uuid(uuid)
      indexable = self.collection.indexable_for_part(part)
      if indexable != 'nodes':
         raise NotImplementedError('well frame related property does not have nodes as indexable element')

      #: Name of log
      self.title = self.model.citation_title_for_part(part)

      #: Unit of measure
      self.uom = self.collection.uom_for_part(part)

   def values(self):
      """ Return log data as numpy array

      Note:
         may return 2D numpy array with shape (num_depths, num_columns).
      """

      part = self.model.part_for_uuid(self.uuid)
      return self.collection.cached_part_array_ref(part)


class StringLookup(BaseResqpy):
   """Class catering for RESQML obj_StringLookupTable objects."""

   resqml_type = "StringTableLookup"

   def __init__(self,
                parent_model,
                root_node = None,
                uuid = None,
                int_to_str_dict = None,
                title = None,
                extra_metadata = None,
                originator = None):
      """Creates a new string lookup (RESQML obj_StringTableLookup) object.

      arguments:
         parent_model: the model to which this string lookup belongs
         root_node (optional): if present, the root xml node for the StringTableLookup from which this object is populated
         int_to_str_dict (optional): if present, a dictionary mapping from integers to strings, used to populate the lookup;
            ignored if root_node is present
         title (string, optional): if present, is used as the citation title for the object; ignored if root_node is not None

      returns:
         the new StringLookup object

      :meta common:
      """

      self.min_index = None
      self.max_index = None
      self.str_list = []
      self.str_dict = {}
      self.stored_as_list = False
      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata,
                       root_node = root_node)
      if uuid is None and root_node is None:
         self.load_from_dict(int_to_str_dict)

   def _load_from_xml(self):
      root_node = self.root
      for v_node in rqet.list_of_tag(root_node, 'Value'):
         key = rqet.find_tag_int(v_node, 'Key')
         value = rqet.find_tag_text(v_node, 'Value')
         assert key not in self.str_dict, 'key value ' + str(key) + ' occurs more than once in string lookup table xml'
         self.str_dict[key] = value
         if self.min_index is None or key < self.min_index:
            self.min_index = key
         if self.max_index is None or key > self.max_index:
            self.max_index = key

   def load_from_dict(self, int_to_str_dict):
      if int_to_str_dict is None:
         return
      assert len(int_to_str_dict), 'empty dictionary passed to string lookup initialisation'
      self.str_dict = int_to_str_dict.copy()
      self.min_index = min(self.str_dict.keys())
      self.max_index = max(self.str_dict.keys())
      self.set_list_from_dict_conditionally()

   def is_equivalent(self, other):
      """Returns True if this lookup is the same as other (apart from uuid); False otherwise."""

      if other is None:
         return False
      if self is other:
         return True
      if bu.matching_uuids(self.uuid, other.uuid):
         return True
      if self.title != other.title or self.min_index != other.min_index or self.max_index != other.max_index:
         return False
      return self.str_dict == other.str_dict

   def set_list_from_dict_conditionally(self):
      """Sets a list copy of the lookup table, which can be indexed directly, if it makes sense to do so."""

      self.str_list = []
      self.stored_as_list = False
      if self.min_index >= 0 and (self.max_index < 50 or 10 * len(self.str_dict) // self.max_index > 8):
         for key in range(self.max_index + 1):
            if key in self.str_dict:
               self.str_list.append(self.str_dict[key])
            else:
               self.str_list.append(None)
         self.stored_as_list = True

   def set_string(self, key, value):
      """Sets the string associated with a given integer key."""

      self.str_dict[key] = value
      limits_changed = False
      if self.min_index is None or value < self.min_index:
         self.min_index = value
         limits_changed = True
      if self.max_index is None or value > self.max_index:
         self.max_index = value
         limits_changed = True
      if self.stored_as_list:
         if limits_changed:
            self.set_list_from_dict_conditionally()
         else:
            self.str_list[key] = value

   def get_string(self, key):
      """Returns the string associated with the integer key, or None if not found.

      :meta common:
      """

      if key < self.min_index or key > self.max_index:
         return None
      if self.stored_as_list:
         return self.str_list[key]
      if key not in self.str_dict:
         return None
      return self.str_dict[key]

   def get_list(self):
      """Returns a list of values, sorted by key.

      :meta common:
      """

      if self.stored_as_list:
         return self.str_list
      return list(dict(sorted(list(self.str_dict.items()))).values())

   def length(self):
      """Returns the nominal length of the lookup table.

      :meta common:
      """

      if self.stored_as_list:
         return len(self.str_list)
      return len(self.str_dict)

   def get_index_for_string(self, string):
      """Returns the integer key for the given string (exact match required), or None if not found.

      :meta common:
      """

      if self.stored_as_list:
         try:
            index = self.str_list.index(string)
            return index
         except Exception:
            return None
      if string not in self.str_dict.values():
         return None
      for k, v in self.str_dict.items():
         if v == string:
            return k
      return None

   def create_xml(self, title = None, originator = None, add_as_part = True, reuse = True):
      """Creates an xml node for the string table lookup.

      arguments:
         title (string, optional): if present, overrides the object's title attribute to be used as citation title
         originator (string, optional): if present, used as the citation creator (otherwise login name is used)
         add_as_part (boolean, default True): if True, the property set is added to the model as a part

      :meta common:
      """

      if title:
         self.title = title

      if reuse and self.try_reuse():
         return self.root  # check for reusable (equivalent) object

      sl_node = super().create_xml(add_as_part = False, originator = originator)

      for k, v in self.str_dict.items():

         pair_node = rqet.SubElement(sl_node, ns['resqml2'] + 'Value')
         pair_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'StringLookup')
         pair_node.text = rqet.null_xml_text

         key_node = rqet.SubElement(pair_node, ns['resqml2'] + 'Key')
         key_node.set(ns['xsi'] + 'type', ns['xsd'] + 'integer')
         key_node.text = str(k)

         value_node = rqet.SubElement(pair_node, ns['resqml2'] + 'Value')
         value_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
         value_node.text = str(v)

      if add_as_part:
         self.model.add_part('obj_StringTableLookup', self.uuid, sl_node)

      return sl_node


class PropertyKind(BaseResqpy):
   """Class catering for RESQML bespoke PropertyKind objects."""

   resqml_type = "PropertyKind"

   def __init__(self,
                parent_model,
                root_node = None,
                uuid = None,
                title = None,
                is_abstract = False,
                example_uom = None,
                naming_system = 'urn:resqml:bp.com:resqpy',
                parent_property_kind = 'continuous',
                extra_metadata = None,
                originator = None):
      """Initialise a new bespoke property kind."""

      self.is_abstract = is_abstract
      self.naming_system = naming_system
      self.example_uom = example_uom
      self.parent_kind = parent_property_kind
      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata,
                       root_node = root_node)

   def _load_from_xml(self):
      root_node = self.root
      self.is_abstract = rqet.find_tag_bool(root_node, 'IsAbstract')
      self.naming_system = rqet.find_tag_text(root_node, 'NamingSystem')
      self.example_uom = rqet.find_tag_text(root_node, 'RepresentativeUom')
      ppk_node = rqet.find_tag(root_node, 'ParentPropertyKind')
      assert ppk_node is not None
      ppk_kind_node = rqet.find_tag(ppk_node, 'Kind')
      assert ppk_kind_node is not None, 'only standard property kinds supported as parent kind'
      self.parent_kind = ppk_kind_node.text

   def is_equivalent(self, other_pk, check_extra_metadata = True):
      """Returns True if this property kind is essentially the same as the other; False otherwise."""

      if other_pk is None:
         return False
      if self is other_pk:
         return True
      if bu.matching_uuids(self.uuid, other_pk.uuid):
         return True
      if (self.parent_kind != other_pk.parent_kind or self.title != other_pk.title or
          self.is_abstract != other_pk.is_abstract or self.naming_system != other_pk.naming_system):
         return False
      if (self.example_uom and other_pk.example_uom) and self.example_uom != other_pk.example_uom:
         return False
      if check_extra_metadata:
         if (self.extra_metadata or other_pk.extra_metadata) and self.extra_metadata != other_pk.extra_metadata:
            return False
      return True

   def create_xml(self, add_as_part = True, originator = None, reuse = True):
      """Create xml for this bespoke property kind."""

      if reuse and self.try_reuse():
         return self.root  # check for reusable (equivalent) object

      pk = super().create_xml(add_as_part = False, originator = originator)

      ns_node = rqet.SubElement(pk, ns['resqml2'] + 'NamingSystem')
      ns_node.set(ns['xsi'] + 'type', ns['xsd'] + 'anyURI')
      ns_node.text = str(self.naming_system)

      ia_node = rqet.SubElement(pk, ns['resqml2'] + 'IsAbstract')
      ia_node.set(ns['xsi'] + 'type', ns['xsd'] + 'boolean')
      ia_node.text = str(self.is_abstract).lower()

      # note: schema definition requires this field, even for discrete property kinds
      uom = self.example_uom
      if uom is None:
         uom = 'Euc'
      ru_node = rqet.SubElement(pk, ns['resqml2'] + 'RepresentativeUom')
      ru_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ResqmlUom')
      ru_node.text = str(uom)

      ppk_node = rqet.SubElement(pk, ns['resqml2'] + 'ParentPropertyKind')
      ppk_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'StandardPropertyKind')
      ppk_node.text = rqet.null_xml_text

      ppk_kind_node = rqet.SubElement(ppk_node, ns['resqml2'] + 'Kind')
      ppk_kind_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'ResqmlPropertyKind')
      ppk_kind_node.text = str(self.parent_kind)

      if add_as_part:
         self.model.add_part('obj_PropertyKind', self.uuid, pk)
         # no relationships at present, if local parent property kinds were to be supported then a rel. is needed there

      return pk


class WellIntervalProperty:
   """Thin wrapper class around interval properties for a Wellbore Frame or Blocked Wellbore (ie interval or cell well logss)."""

   def __init__(self, collection, part):
      """Create an interval log or blocked well log from a part name"""

      self.collection: PropertyCollection = collection
      self.model = collection.model
      self.part = part

      indexable = self.collection.indexable_for_part(part)
      assert indexable in ['cells', 'intervals'], 'expected cells or intervals as indexable element'

      self.name = self.model.citation_title_for_part(part)
      self.uom = self.collection.uom_for_part(part)

   def values(self):
      """Return interval log or blocked well log as numpy array"""

      return self.collection.cached_part_array_ref(self.part)


class WellIntervalPropertyCollection(PropertyCollection):
   """Class for RESQML property collection for a WellboreFrame for interval or blocked well logs, inheriting from PropertyCollection."""

   def __init__(self, frame = None, property_set_root = None, realization = None):
      """Creates a new property collection related to interval or blocked well logs and a wellbore frame."""

      super().__init__(support = frame, property_set_root = property_set_root, realization = realization)

   def logs(self):
      """Generator that yields component Interval log or Blocked well log objects"""
      return (WellIntervalProperty(collection = self, part = part) for part in self.parts())

   def to_pandas(self, include_units = False):
      cell_indices = [return_cell_indices(i, self.support.cell_indices) for i in self.support.cell_grid_link]
      data = {}
      for log in self.logs():
         col_name = log.name
         values = log.values()
         data[col_name] = values
      df = pd.DataFrame(data = data, index = cell_indices)
      return df


def same_property_kind(pk_a, pk_b):
   """Returns True if the two property kinds are the same, or pseudonyms."""

   if pk_a is None or pk_b is None:
      return False
   if pk_a == pk_b:
      return True
   if pk_a in ['permeability rock', 'rock permeability'] and pk_b in ['permeability rock', 'rock permeability']:
      return True
   return False


def create_transmisibility_multiplier_property_kind(model):
   """Create a local property kind 'transmisibility multiplier' for a given model

   argument:
      model: resqml model object

   returns:
      property kind uuid
   """
   log.debug("Making a new property kind 'Transmissibility multiplier'")
   tmult_kind = PropertyKind(parent_model = model,
                             title = 'transmissibility multiplier',
                             parent_property_kind = 'continuous')
   tmult_kind.create_xml()
   tmult_kind_uuid = tmult_kind.uuid
   model.store_epc()
   return tmult_kind_uuid


def property_kind_and_facet_from_keyword(keyword):
   """If keyword is recognised, returns equivalent resqml PropertyKind and Facet info.

   argument:
      keyword (string): Nexus grid property keyword

   returns:
      (property_kind, facet_type, facet) as defined in resqml standard; some or all may be None

   note:
      this function may now return the local property kind 'transmissibility multiplier'; calling code must ensure that
      the local property kind object is created if not already present
   """

   # note: this code doesn't cater for a property having more than one facet, eg. direction and phase

   def facet_info_for_dir_ch(dir_ch):
      facet_type = None
      facet = None
      if dir_ch in ['i', 'j', 'k', 'x', 'y', 'z']:
         facet_type = 'direction'
         if dir_ch in ['i', 'x']:
            facet = 'I'
         elif dir_ch in ['j', 'y']:
            facet = 'J'
         else:
            facet = 'K'
         # NB: resqml also allows for combinations, eg. IJ, IJK
      return (facet_type, facet)

   # note: 'what' facet_type for made-up uses might be better changed to the generic 'qualifier' facet type

   property_kind = None
   facet_type = None
   facet = None
   lk = keyword.lower()
   if lk in ['bv', 'brv']:  # bulk rock volume
      property_kind = 'rock volume'
      facet_type = 'netgross'
      facet = 'gross'  # todo: check this is the facet in xsd
   elif lk in ['pv', 'pvr', 'porv']:
      property_kind = 'pore volume'  # pore volume
   elif lk in ['mdep', 'depth', 'tops', 'mids']:
      property_kind = 'depth'  # depth (nexus) and tops mean top depth
      facet_type = 'what'  # this might need to be something different
      if lk in ['mdep', 'mids']:
         facet = 'cell centre'
      else:
         facet = 'cell top'
   elif lk in ['ntg', 'netgrs']:
      property_kind = 'net to gross ratio'  # net-to-gross
   elif lk in ['netv', 'nrv']:  # net volume
      property_kind = 'rock volume'
      facet_type = 'netgross'
      facet = 'net'
   elif lk in ['dzc', 'dzn', 'dz', 'dznet']:
      property_kind = 'thickness'  # or should these keywords use cell length in K direction?
      facet_type = 'netgross'
      if lk.startswith('dzn'):
         facet = 'net'
      else:
         facet = 'gross'
   elif lk in ['dxc', 'dyc', 'dx', 'dy']:
      property_kind = 'cell length'
      (facet_type, facet) = facet_info_for_dir_ch(lk[1])
   elif len(lk) > 2 and lk[0] == 'd' and lk[1] in 'xyz':
      property_kind = 'length'
      facet_type = 'direction'
      facet = lk[1].upper()  # keep as 'X', 'Y' or 'Z'
   elif lk in ['por', 'poro', 'porosity']:
      property_kind = 'porosity'  # porosity
   elif lk == 'kh':
      property_kind = 'permeability thickness'  # K.H (not horizontal permeability)
   elif lk[:4] == 'perm' or (len(lk) == 2 and lk[0] == 'k'):  # permeability
      property_kind = 'permeability rock'
      (facet_type, facet) = facet_info_for_dir_ch(lk[-1])
   elif lk[:5] == 'trans' or (len(lk) == 2 and lk[0] == 't'):  # transmissibility (for unit viscosity)
      property_kind = 'transmissibility'
      (facet_type, facet) = facet_info_for_dir_ch(lk[-1])
   elif lk in ['p', 'pressure']:
      property_kind = 'pressure'  # pressure; todo: phase pressures
   elif lk in ['sw', 'so', 'sg', 'satw', 'sato', 'satg', 'soil']:  # saturations
      property_kind = 'saturation'
      facet_type = 'what'  # todo: check use of 'what' for phase
      if lk in ['sw', 'satw', 'swat']:
         facet = 'water'
      elif lk in ['so', 'sato', 'soil']:
         facet = 'oil'
      elif lk in ['sg', 'satg', 'sgas']:
         facet = 'gas'
   elif lk in ['swl', 'swr', 'sgl', 'sgr', 'swro', 'sgro', 'sor', 'swu', 'sgu']:  # nexus saturation end points
      property_kind = 'saturation'
      facet_type = 'what'  # note: use of 'what' for phase is a guess
      if lk[1] == 'w':
         facet = 'water'
      elif lk[1] == 'g':
         facet = 'gas'
      elif lk[1] == 'o':
         facet = 'oil'
      if lk[-1] == 'l':
         facet += ' minimum'
      elif lk[-1] == 'u':
         facet += ' maximum'
      elif lk[2:] == 'ro':
         facet += ' residual to oil'
      elif lk[-1] == 'r':
         facet += ' residual'
      else:
         assert False, 'problem deciphering saturation end point keyword: ' + lk
#   elif lk == 'sal':    # todo: salinity; local property kind needed; various units possible in Nexus
   elif lk in ['wip', 'oip', 'gip', 'mobw', 'mobo', 'mobg', 'ocip']:  # todo: check these, especially ocip
      property_kind = 'fluid volume'
      facet_type = 'what'  # todo: check use of 'what' for phase
      if lk in ['wip', 'mobw']:
         facet = 'water'  # todo: add another facet indicating mobile volume
      elif lk in ['oip', 'mobo']:
         facet = 'oil'
      elif lk in ['gip', 'mobg']:
         facet = 'gas'
      elif lk == 'ocip':
         facet = 'oil condensate'  # todo: this seems unlikely: check
      if lk[:3] == 'mob':
         facet += ' (mobile)'
   elif lk in ['tmx', 'tmy', 'tmz', 'tmflx', 'tmfly', 'tmflz', 'multx', 'multy', 'multz']:
      property_kind = 'transmissibility multiplier'  # NB: resqpy local property kind
      facet_type = 'direction'
      _, facet = facet_info_for_dir_ch(lk[-1])
   elif lk in ['multbv', 'multpv']:
      property_kind = 'property multiplier'
      facet_type = 'what'  # here 'what' facet indicates property affected
      if lk == 'multbv':
         facet = 'rock volume'  # NB: making this up as I go along
      elif lk == 'multpv':
         facet = 'pore volume'
   elif lk == 'rs':
      property_kind = 'solution gas-oil ratio'
   elif lk == 'rv':
      property_kind = 'vapor oil-gas ratio'
   elif lk in ['temp', 'temperature']:
      property_kind = 'thermodynamic temperature'
   elif lk in ['dad', 'kid', 'unpack', 'deadcell', 'inactive']:
      property_kind = 'code'
      facet_type = 'what'
      # todo: kid can only be used as an inactive cell indication for the root grid
      if lk in ['kid', 'deadcell', 'inactive']:
         facet = 'inactive'  # standize on 'inactive' to indicate use as mask
      else:
         facet = lk  # note: use deadcell or unpack for inactive, if nothing better?
   elif lk == 'livecell' or lk.startswith('act'):
      property_kind = 'active'  # local property kind, see RESQML (2.0.1) usage guide, section 11.17


#      property_kind = 'code'
#      facet_type = 'what'
#      facet = 'active'
   elif lk[0] == 'i' or lk.startswith('reg') or lk.startswith('creg'):
      property_kind = 'region initialization'  # local property kind, see RESQML (2.0.1) usage guide, section 11.18
   elif lk == 'uid':
      property_kind = 'index'
      facet_type = 'what'
      facet = 'uid'
   return (property_kind, facet_type, facet)


def infer_property_kind(name, unit):
   """ Guess a valid property kind """

   # Currently unit is ignored

   valid_kinds = bwam.valid_property_kinds()

   if name in valid_kinds:
      kind = name
   else:
      # TODO: use an appropriate default
      kind = 'Unknown'

   # TODO: determine facet_type and facet somehow
   facet_type = None
   facet = None

   return kind, facet_type, facet


def guess_uom(property_kind, minimum, maximum, support, facet_type = None, facet = None):
   """Returns a guess at the units of measure for the given kind of property.

   arguments:
      property_kind (string): a valid resqml property kind, lowercase
      minimum: the minimum value in the data for which the units are being guessed
      maximum: the maximum value in the data for which the units are being guessed
      support: the grid.Grid or well.WellboreFrame object which the property data relates to
      facet_type (string, optional): a valid resqml facet type, lowercase, one of:
              'direction', 'what', 'netgross', 'qualifier', 'conditions', 'statistics'
      facet: (string, present if facet_type is present): the value relating to the facet_type,
               eg. 'I' for direction, or 'oil' for 'what'

   returns:
      a valid resqml unit of measure (uom) for the property_kind, or None

   note:
      the resqml standard allows a property to have any number of facets; however,
      this module currently only supports zero or one facet per property
   """

   def crs_m_or_ft(crs_node):  # NB. models not-so-rarely use metres for xy and feet for z
      if crs_node is None:
         return None
      xy_units = rqet.find_tag(crs_node, 'ProjectedUom').text.lower()
      z_units = rqet.find_tag(crs_node, 'VerticalUom').text.lower()
      if xy_units == 'm' and z_units == 'm':
         return 'm'
      if xy_units == 'ft' and z_units == 'ft':
         return 'ft'
      return None

   if support is None or not hasattr(support, 'extract_crs_root'):
      crs_node = None
   else:
      crs_node = support.extract_crs_root()
   from_crs = crs_m_or_ft(crs_node)

   if property_kind in ['rock volume', 'pore volume', 'volume']:
      if from_crs is None:
         return None
      if from_crs == 'ft' and property_kind == 'pore volume':
         return 'bbl'  # seems to be Nexus 'ENGLISH' uom for pv out
      return from_crs + '3'  # ie. m3 or ft3
   if property_kind == 'depth':
      if crs_node is None:
         return None
      return rqet.find_tag(crs_node, 'VerticalUom').text.lower()
   if property_kind == 'cell length':  # todo: pass in direction facet to pick up xy_units or z_units
      return from_crs
   if property_kind in ['net to gross ratio', 'porosity', 'saturation']:
      if maximum is not None and str(maximum) != 'unknown':
         max_real = float(maximum)
         if max_real > 1.0 and max_real <= 100.0:
            return '%'
         if max_real < 0.0 or max_real > 1.0:
            return None
      if from_crs == 'm':
         return 'm3/m3'
      if from_crs == 'ft':
         return 'ft3/ft3'
      return 'Euc'
   if property_kind == 'permeability rock' or property_kind == 'rock permeability':
      return 'mD'
   if property_kind == 'permeability thickness':
      z_units = rqet.find_tag(crs_node, 'VerticalUom').text.lower()
      if z_units == 'm':
         return 'mD.m'
      if z_units == 'ft':
         return 'mD.ft'
      return None
   if property_kind == 'permeability length':
      xy_units = rqet.find_tag(crs_node, 'ProjectedUom').text.lower()
      if xy_units == 'm':
         return 'mD.m'
      if xy_units == 'ft':
         return 'mD.ft'
      return None
   if property_kind == 'fluid volume':
      if from_crs == 'm':
         return 'm3'
      if from_crs == 'ft':
         if facet_type == 'what' and facet == 'gas':
            return '1000 ft3'  # todo: check units output by nexus for GIP
         else:
            return 'bbl'  # todo: check whether nexus uses 10^3 or 10^6 units
      return None
   if property_kind.endswith(
         'transmissibility'):  # note: RESQML QuantityClass only includes a unit-viscosity VolumePerTimePerPressureUom
      if from_crs == 'm':
         return 'm3.cP/(kPa.d)'  # NB: might actually be m3/(psi.d) or m3/(bar.d)
      if from_crs == 'ft':
         return 'bbl.cP/(psi.d)'  # gamble on barrels per day per psi; could be ft3/(psi.d)
      return None
   if property_kind == 'pressure':
      if from_crs == 'm':
         return 'kPa'  # NB: might actually be psi or bar
      if from_crs == 'ft':
         return 'psi'
      if maximum is not None:
         max_real = float(maximum)
         if max_real == 0.0:
            return None
         if max_real > 10000.0:
            return 'kPa'
         if max_real < 500.0:
            return 'bar'
         if max_real < 5000.0:
            return 'psi'
      return None
   if property_kind == 'solution gas-oil ratio':
      if from_crs == 'm':
         return 'm3/m3'  # NB: might actually be psi or bar
      if from_crs == 'ft':
         return '1000 ft3/bbl'
      return None
   if property_kind == 'vapor oil-gas ratio':
      if from_crs == 'm':
         return 'm3/m3'  # NB: might actually be psi or bar
      if from_crs == 'ft':
         return '0.001 bbl/ft3'
      return None
   if property_kind.endswith('multiplier'):
      return 'Euc'
   # todo: 'degC' or 'degF' for thermodynamic temperature
   return None


def selective_version_of_collection(
      collection,
      realization = None,
      support_uuid = None,
      grid = None,  # for backward compatibility
      uuid = None,
      continuous = None,
      count = None,
      indexable = None,
      property_kind = None,
      facet_type = None,
      facet = None,
      citation_title = None,
      time_series_uuid = None,
      time_index = None,
      uom = None,
      string_lookup_uuid = None,
      categorical = None):
   """Returns a new PropertyCollection with those parts which match all arguments that are not None.

   arguments:
      collection: an existing PropertyCollection from which a subset will be returned as a new object;
                  the existing collection might often be the 'main' collection holding all the properties
                  for a supporting representation (grid or wellbore frame)

   Other optional arguments:
   realization, support_uuid, grid, uuid, continuous, count, indexable, property_kind, facet_type, facet, citation_title,
   time_series_uuid, time_index, uom, string_lookup_uuid, categorical:

   for each of these arguments: if None, then all members of collection pass this filter;
   if not None then only those members with the given value pass this filter;
   finally, the filters for all the attributes must be passed for a given member
   to be included in the returned collection.

   returns:
      a new PropertyCollection containing those properties which match the filter parameters that are not None

   note:
      the grid keyword argument is maintained for backward compatibility: support_uuid argument takes precedence;
      the categorical boolean argument can be used to select only
      categorical (or non-categorical) properties, even though this is not explicitly held as a field in the
      internal dictionary
   """

   assert collection is not None
   view = PropertyCollection()
   if support_uuid is None and grid is not None:
      support_uuid = grid.uuid
   if support_uuid is not None:
      view.set_support(support_uuid = support_uuid)
   if realization is not None:
      view.set_realization(realization)
   view.inherit_parts_selectively_from_other_collection(collection,
                                                        realization = realization,
                                                        support_uuid = support_uuid,
                                                        uuid = uuid,
                                                        continuous = continuous,
                                                        count = count,
                                                        indexable = indexable,
                                                        property_kind = property_kind,
                                                        facet_type = facet_type,
                                                        facet = facet,
                                                        citation_title = citation_title,
                                                        time_series_uuid = time_series_uuid,
                                                        time_index = time_index,
                                                        uom = uom,
                                                        string_lookup_uuid = string_lookup_uuid,
                                                        categorical = categorical)
   return view


def property_over_time_series_from_collection(collection, example_part):
   """Returns a new PropertyCollection with parts like the example part, over all indices in its time series.

   arguments:
      collection: an existing PropertyCollection from which a subset will be returned as a new object;
                  the existing collection might often be the 'main' collection holding all the properties
                  for a grid
      example_part (string): the part name of an example member of collection (which has an associated time_series)

   returns:
      a new PropertyCollection containing those memners of collection which have the same property kind
      (and facet, if any) as the example part and which have the same associated time series
   """

   assert collection is not None and example_part is not None
   assert collection.part_in_collection(example_part)
   view = PropertyCollection()
   if collection.support_uuid is not None:
      view.set_support(support_uuid = collection.support_uuid)
   if collection.realization is not None:
      view.set_realization(collection.realization)
   view.inherit_similar_parts_for_time_series_from_other_collection(collection, example_part)
   return view


def property_collection_for_keyword(collection, keyword):
   """Returns a new PropertyCollection with parts that match the property kind and facet deduced for the keyword.

   arguments:
      collection: an existing PropertyCollection from which a subset will be returned as a new object;
                  the existing collection might often be the 'main' collection holding all the properties
                  for a supporting representation (grid or wellbore frame)
      keyword (string): a simulator keyword for which the property kind (and facet, if any) can be deduced

   returns:
      a new PropertyCollection containing those memners of collection which have the property kind
      (and facet, if any) as that deduced for the keyword

   note:
      this function is particularly relevant to grid property collections for simulation models;
      the handling of simulator keywords in this module is based on the main grid property keywords
      for Nexus; if the resqml dataset was generated from simulator data using this module then
      the result of this function should be reliable; resqml data sets from other sources might use facets
      if a different way, leading to an omission in the results of this function
   """

   assert collection is not None and keyword
   (property_kind, facet_type, facet) = property_kind_and_facet_from_keyword(keyword)
   if property_kind is None:
      log.warning('failed to deduce property kind for keyword: ' + keyword)
      return None
   return selective_version_of_collection(collection,
                                          property_kind = property_kind,
                                          facet_type = facet_type,
                                          facet = facet)


def reformat_column_edges_to_resqml_format(array):
   """Converts an array of shape (nj,ni,2,2) to shape (nj,ni,4) in RESQML edge ordering"""
   newarray = np.empty((array.shape[0], array.shape[1], 4))
   newarray[:, :, 0] = array[:, :, 1, 0]
   newarray[:, :, 1] = array[:, :, 0, 1]
   newarray[:, :, 2] = array[:, :, 1, 1]
   newarray[:, :, 3] = array[:, :, 0, 0]
   return newarray


def reformat_column_edges_from_resqml_format(array):
   """Converts an array of shape (nj,ni,4) in RESQML edge ordering to shape (nj,ni,2,2)"""
   newarray = np.empty((array.shape[0], array.shape[1], 2, 2))
   newarray[:, :, 0, 0] = array[:, :, 3]
   newarray[:, :, 0, 1] = array[:, :, 1]
   newarray[:, :, 1, 0] = array[:, :, 0]
   newarray[:, :, 1, 1] = array[:, :, 2]
   return newarray


# 'private' functions returning attribute name for cached version of property array
# I find the leading underscore so ugly, I can't bring myself to use it for 'private' functions, even though many people do


def _cache_name_for_uuid(uuid):
   """
      Returns the attribute name used for the cached copy of the property array for the given uuid.

      :meta private:
   """

   return 'c_' + bu.string_from_uuid(uuid)


def _cache_name(part):
   """
      Returns the attribute name used for the cached copy of the property array for the given part.

      :meta private:
   """

   if part is None:
      return None
   uuid = rqet.uuid_in_part_name(part)
   if uuid is None:
      return None
   return _cache_name_for_uuid(uuid)


def dtype_flavour(continuous, use_32_bit):
   """
      Returns the numpy elemental data type depending on the two boolean flags.

      :meta private:
   """

   if continuous:
      if use_32_bit:
         dtype = np.float32
      else:
         dtype = np.float64
   else:
      if use_32_bit:
         dtype = np.int32
      else:
         dtype = np.int64
   return dtype


def return_cell_indices(i, cell_indices):
   if i == -1:
      return np.nan
   else:
      return cell_indices[i]
