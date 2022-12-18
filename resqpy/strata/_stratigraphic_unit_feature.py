"""_stratigraphic_unit_feature.py: RESQML StratigraphicUnitFeature class."""

# NB: in this module, the term 'unit' refers to a geological stratigraphic unit, i.e. a layer of rock, not a unit of measure

import logging

log = logging.getLogger(__name__)

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
from resqpy.olio.base import BaseResqpy


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
        return (self.title == other.title and
                ((not check_extra_metadata) or rqo.equivalent_extra_metadata(self, other)))

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
                                                              self.model.root(uuid = self.bottom_unit_uuid),
                                                              'sourceObject')
                if self.top_unit_uuid is not None and not bu.matching_uuids(self.bottom_unit_uuid, self.top_unit_uuid):
                    self.model.create_reciprocal_relationship(suf, 'destinationObject',
                                                              self.model.root(uuid = self.top_unit_uuid),
                                                              'sourceObject')

        return suf
