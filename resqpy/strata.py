"""strata.py: RESQML stratigraphy classes."""

version = '9th August 2021'

import logging

log = logging.getLogger(__name__)
log.debug('strata.py version ' + version)

import warnings

import resqpy.organize as rqo
import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu
from resqpy.olio.xml_namespaces import curly_namespace as ns
from resqpy.olio.base import BaseResqpy


class StratigraphicUnitFeature(BaseResqpy):
   """Class for RESQML Stratigraphic Unit Feature objects.

   RESQML documentation:

      A stratigraphic unit that can have a well-known (e.g., "Jurassic") chronostratigraphic top and
      chronostratigraphic bottom. These chronostratigraphic units have no associated interpretations or representations.

      BUSINESS RULE: The name must reference a well-known chronostratigraphic unit (such as "Jurassic"),
      for example, from the International Commission on Stratigraphy (http://www.stratigraphy.org).
   """

   resqml_type = "StratigraphicUnitFeature"

   def __init__(
         self,
         parent_model,
         uuid = None,
         # TODO: add chrono strata top and bottom uuid optional args
         title = None,
         originator = None,
         extra_metadata = None):
      """Initialises a stratigraphic unit feature object."""

      # the following two attributes are object references in the xsd
      # TODO: clarify with Energistics whether the references are to other StratigraphicUnitFeatures or what?
      self.top_unit = None
      self.bottom_unit = None

      super().__init__(model = parent_model,
                       uuid = uuid,
                       title = title,
                       originator = originator,
                       extra_metadata = extra_metadata)

      # TODO: if root is None, set chrono strata top and bottom if their uuids are present

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this feature is essentially the same as the other; otherwise False."""

      if not isinstance(other, StratigraphicUnitFeature):
         return False
      if self is other or bu.matching_uuids(self.uuid, other.uuid):
         return True
      return (self.title == other.title and ((not check_extra_metadata) or rqo.equivalent_extra_metadata(self, other)))

   def _load_from_xml(self):
      # TODO: load chrono strata top and bottom
      pass

   def create_xml(self, add_as_part = True, originator = None, reuse = True):
      """Creates xml for this stratigraphic unit feature."""

      if reuse and self.try_reuse():
         return self.root  # check for reusable (equivalent) object
      # create node with citation block
      suf = super().create_xml(add_as_part = False, originator = originator)

      if add_as_part:
         self.model.add_part('obj_StratigraphicUnitFeature', self.uuid, suf)

      return suf


class StratigraphicUnitInterpretation(BaseResqpy):
   """Class for RESQML Stratigraphic Unit Interpretation objects."""

   # TODO
   pass


class StratigraphicColumn(BaseResqpy):
   """Class for RESQML stratigraphic column objects.

   RESQML documentation:

      A global interpretation of the stratigraphy, which can be made up of
      several ranks of stratigraphic unit interpretations.

      All stratigraphic column rank interpretations that make up a stratigraphic column
      must be ordered by age.
   """

   resqml_type = "StratigraphicColumn"

   # todo: integrate with EarthModelInterpretation class which can optionally refer to a stratigraphic column

   def __init__(self, parent_model, uuid = None, rank_uuid_list = None, title = None, extra_metadata = None):
      """Initialises a Stratigraphic Column object.

      arguments:
         rank_uuid_list (list of uuid, optional): if not initialising for an existing stratigraphic column,
            the ranks can be established from this list of uuids for existing stratigraphic column ranks;
            ranks should be ordered from geologically oldest to youngest
      """

      self.ranks = []  # list of Stratigraphic Column Rank Interpretation objects

      super().__init__(model = parent_model, uuid = uuid, title = title, extra_metadata = extra_metadata)

      if self.root is None and rank_uuid_list:
         for rank_uuid in rank_uuid_list:
            rank = StratigraphicColumnRank(self.model, uuid = rank_uuid)
            self.add_rank(rank)

   def _load_from_xml(self):
      rank_node_list = rqet.list_of_tag(self.root, 'Ranks')
      assert rank_node_list is not None, 'no stratigraphic column ranks in xml for stratigraphic column'
      for rank_node in rank_node_list:
         rank = StratigraphicColumnRank(self.model, uuid = rqet.uuid_for_part_root(rank_node))
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
         ranks should be ordered from geologically oldest to youngest
      """
      assert rank is not None
      self.ranks.append(StratigraphicColumnRank(self.model, uuid = rank.uuid))

   def is_equivalent(self, other, check_extra_metadata = True):
      """Returns True if this interpretation is essentially the same as the other; otherwise False."""

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
      """Creates xml tree for a stratigraphic column object."""

      assert self.ranks, 'attempt to create xml for stratigraphic column without any contributing ranks'

      if reuse and self.try_reuse():
         return self.root

      sci = super().create_xml(add_as_part = False, originator = originator)

      assert sci is not None

      for rank in self.ranks:
         self.model.create_ref_node('InterpretedFeature',
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
   """Class for RESQML StratigraphicColumnRankInterpretation objects."""

   # TODO
   pass
