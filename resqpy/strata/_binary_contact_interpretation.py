"""RESQML low level BinaryContactInterpretation class."""

# NB: in this module, the term 'unit' refers to a geological stratigraphic unit, i.e. a layer of rock, not a unit of measure

import logging

log = logging.getLogger(__name__)

from typing import Optional

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.strata
import resqpy.strata._strata_common as rqstc
from resqpy.olio.xml_namespaces import curly_namespace as ns


class BinaryContactInterpretation:
    """Internal class for contact between 2 geological entities; not a high level class but used by others."""

    def __init__(
            self,
            model,
            existing_xml_node = None,
            index = None,
            contact_relationship: Optional[str] = None,
            verb: Optional[str] = None,
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
            assert contact_relationship in rqstc.valid_contact_relationships
            assert verb in rqstc.valid_contact_verbs
            assert subject_uuid is not None and direct_object_uuid is not None
            if subject_contact_side is not None:
                assert subject_contact_side in rqstc.valid_contact_sides
            if subject_contact_mode is not None:
                assert subject_contact_mode in rqstc.valid_contact_modes
            if direct_object_contact_side is not None:
                assert direct_object_contact_side in rqstc.valid_contact_sides
            if direct_object_contact_mode is not None:
                assert direct_object_contact_mode in rqstc.valid_contact_modes
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
        assert self.contact_relationship in rqstc.valid_contact_relationships,  \
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
        assert self.verb in rqstc.valid_contact_verbs,  \
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
