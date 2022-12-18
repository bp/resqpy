"""RESQML StratigraphicColumnRankInterpretation class."""

# NB: in this module, the term 'unit' refers to a geological stratigraphic unit, i.e. a layer of rock, not a unit of measure
# RMS is a registered trademark of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.organize as rqo
import resqpy.strata
import resqpy.strata._strata_common as rqstc
import resqpy.strata._binary_contact_interpretation as rqsbc
import resqpy.strata._stratigraphic_unit_interpretation as rqsui
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns


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
        self.units = []  # ordered list of (index, stratagraphic unit interpretations)
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
            self.units.append((index, rqsui.StratigraphicUnitInterpretation(self.model, uuid = unit_uuid)))
        self._sort_units()
        self.contacts = []
        for contact_node in rqet.list_of_tag(root_node, 'ContactInterpretation'):
            self.contacts.append(rqsbc.BinaryContactInterpretation(self.model, existing_xml_node = contact_node))
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
            self.units.append((i, rqsui.StratigraphicUnitInterpretation(self.model, uuid = uuid)))

    def _sort_units(self):
        """Sorts units list, in situ, into increasing order of index values."""
        self.units.sort()

    def _sort_contacts(self):
        """Sorts contacts list, in situ, into increasing order of index values."""
        self.contacts.sort(key = rqstc._index_attr)

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
            contact = rqsbc.BinaryContactInterpretation(
                self.model,
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

        assert self.domain in rqstc.valid_domains, 'illegal domain value for stratigraphic column rank interpretation'
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
                            self.model.create_reciprocal_relationship(scri, 'destinationObject', horizon_root,
                                                                      'sourceObject')

        return scri
