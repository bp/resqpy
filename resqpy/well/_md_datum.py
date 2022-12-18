"""MdDatum class."""

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)

import warnings
import numpy as np

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
from resqpy.olio.base import BaseResqpy
from resqpy.olio.xml_namespaces import curly_namespace as ns

valid_md_reference_list = [
    "ground level", "kelly bushing", "mean sea level", "derrick floor", "casing flange", "arbitrary point",
    "crown valve", "rotary bushing", "rotary table", "sea floor", "lowest astronomical tide", "mean higher high water",
    "mean high water", "mean lower low water", "mean low water", "mean tide level", "kickoff point"
]

# todo: could require/maintain DeviationSurvey mds in same units as md datum object's crs vertical units?


class MdDatum(BaseResqpy):
    """Class for RESQML measured depth datum."""

    resqml_type = 'MdDatum'

    def __init__(self,
                 parent_model,
                 uuid = None,
                 crs_uuid = None,
                 location = None,
                 md_reference = 'mean sea level',
                 title = None,
                 originator = None,
                 extra_metadata = None):
        """Initialises a new MdDatum object.

        arguments:
           parent_model (model.Model object): the model which the new md datum belongs to
           uuid: If not None, load from existing object. Else, create new.
           crs_uuid (uuid.UUID): required if initialising from values
           location: (triple float): the x, y, z location of the new measured depth datum;
              ignored if uuid is not None
           md_reference (string): human readable resqml standard string indicating the real
              world nature of the datum, eg. 'kelly bushing'; the full list of options is
              available as the global variable valid_md_reference_list in this module;
              ignored if uuid is not None
           title (str, optional): the citation title to use for a new datum;
              ignored if uuid is not None
           originator (str, optional): the name of the person creating the datum, defaults to login id;
              ignored if uuid is not None
           extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the datum;
              ignored if uuid is not None

        returns:
           the newly instantiated measured depth datum object

        note:
           this function does not create an xml node for the md datum; call the create_xml() method afterwards
           if initialising from data other than an existing RESQML object
        """

        self.location = location
        self.md_reference = md_reference
        self.crs_uuid = crs_uuid

        super().__init__(model = parent_model,
                         uuid = uuid,
                         title = title,
                         originator = originator,
                         extra_metadata = extra_metadata)

        if self.root is None and (location is not None or md_reference):
            assert location is not None and md_reference
            assert md_reference in valid_md_reference_list
            assert len(location) == 3

    def _load_from_xml(self):
        md_datum_root = self.root
        assert md_datum_root is not None
        location_node = rqet.find_tag(md_datum_root, 'Location')
        self.location = (rqet.find_tag_float(location_node,
                                             'Coordinate1'), rqet.find_tag_float(location_node, 'Coordinate2'),
                         rqet.find_tag_float(location_node, 'Coordinate3'))
        self.md_reference = rqet.node_text(rqet.find_tag(md_datum_root, 'MdReference')).strip().lower()
        assert self.md_reference in valid_md_reference_list
        self.crs_uuid = self.extract_crs_uuid()

    # todo: the following function is almost identical to one in the grid module: it should be made common and put in model.py

    def extract_crs_uuid(self):
        """Returns uuid for coordinate reference system, as stored in reference node of this md datum's xml tree."""

        if self.crs_uuid is not None:
            return self.crs_uuid
        crs_root = rqet.find_tag(self.root, 'LocalCrs')
        uuid_str = rqet.find_tag(crs_root, 'UUID').text
        self.crs_uuid = bu.uuid_from_string(uuid_str)
        return self.crs_uuid

    def create_part(self):
        """Creates xml for this md datum object and adds to parent model as a part; returns root node for part."""

        # note: deprecated, call create_xml() directly
        assert self.root is None
        assert self.location is not None
        self.create_xml(add_as_part = True)

    def create_xml(self, add_as_part = True, add_relationships = True, title = None, originator = None):
        """Creates xml for a measured depth datum element; crs node must already exist; optionally adds as part.

        arguments:
           add_as_part (boolean, default True): if True, the newly created xml node is added as a part
              in the model
           add_relationships (boolean, default True): if True, a relationship xml part is created relating the
              new md datum part to the crs
           title (string): used as the citation Title text for the new md datum node
           originator (string, optional): the name of the human being who created the md datum part;
              default is to use the login name

        returns:
           the newly created measured depth datum xml node
        """

        md_reference = self.md_reference.lower()
        assert md_reference in valid_md_reference_list, 'invalid measured depth reference: ' + md_reference

        if title:
            self.title = title
        if not self.title:
            self.title = 'measured depth datum'

        crs_uuid = self.crs_uuid
        assert crs_uuid is not None

        datum = super().create_xml(add_as_part = False, originator = originator)

        self.model.create_solitary_point3d('Location', datum, self.location)

        md_ref = rqet.SubElement(datum, ns['resqml2'] + 'MdReference')
        md_ref.set(ns['xsi'] + 'type', ns['resqml2'] + 'MdReference')
        md_ref.text = md_reference

        self.model.create_crs_reference(crs_uuid = crs_uuid, root = datum)

        if add_as_part:
            self.model.add_part('obj_MdDatum', self.uuid, datum)
            if add_relationships:
                crs_root = self.model.root_for_uuid(self.crs_uuid)
                self.model.create_reciprocal_relationship(datum, 'destinationObject', crs_root, 'sourceObject')

        return datum

    def is_equivalent(self, other):
        """Implements equals operator, comparing metadata items deemed significant."""

        if not isinstance(other, self.__class__):
            return False
        if self.md_reference != other.md_reference or not np.allclose(self.location, other.location):
            return False
        return bu.matching_uuids(self.crs_uuid, other.crs_uuid)
