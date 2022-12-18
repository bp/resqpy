"""_stratigraphic_column.py: RESQML StratigraphicColumn class."""

# NB: in this module, the term 'unit' refers to a geological stratigraphic unit, i.e. a layer of rock, not a unit of measure

import logging

log = logging.getLogger(__name__)

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
import resqpy.strata
import resqpy.strata._strata_common as rqstc
import resqpy.strata._stratigraphic_column_rank as rqscr
from resqpy.olio.base import BaseResqpy


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
                rank = rqscr.StratigraphicColumnRank(self.model, uuid = rank_uuid)
                self.add_rank(rank)

    def _load_from_xml(self):
        """Loads class specific attributes from xml for an existing RESQML object; called from BaseResqpy."""
        rank_node_list = rqet.list_of_tag(self.root, 'Ranks')
        assert rank_node_list is not None, 'no stratigraphic column ranks in xml for stratigraphic column'
        for rank_node in rank_node_list:
            rank = rqscr.StratigraphicColumnRank(self.model, uuid = rqet.find_tag_text(rank_node, 'UUID'))
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
        self.ranks.sort(key = rqstc._index_attr)

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
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
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
