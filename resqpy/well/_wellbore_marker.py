"""_wellbore_marker.py: resqpy well module providing wellbore marker class"""

version = '6th December 2021'

# following should be kept in line with major.minor tag values in repository
citation_format = 'bp:resqpy:1.3'

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)
log.debug('_wellbore_marker.py version ' + version)
import getpass

import resqpy.olio.xml_et as rqet
from resqpy.olio.xml_namespaces import curly_namespace as ns
import resqpy.olio.uuid as bu
import resqpy.olio.time as time


class WellboreMarker():
    """Class to handle RESQML WellboreMarker objects.

    note:
       wellbore markers are not high level RESQML objects
    """

    resqml_type = 'WellboreMarker'

    def __init__(self,
                 parent_model,
                 parent_frame,
                 marker_index,
                 marker_node = None,
                 marker_type = None,
                 interpretation_uuid = None,
                 title = None,
                 originator = None,
                 extra_metadata = None):
        """Creates a new wellbore marker object and optionally loads it from xml.

        arguments:
           parent_model (model.Model object): the model which the new wellbore marker belongs to
           parent_frame (wellbore_marker_frame.WellboreMarkerFramer object): the wellbore marker frame to which the wellbore marker belongs
           marker_index (int): index of the wellbore marker in the parent WellboreMarkerFrame object
           marker_node (xml node, optional): if given, loads from xml. Else, creates new
           marker_type (str, optional): the type of geologic, fluid or contact feature
              e.g. "fault", "geobody", "horizon", "gas/oil/water down to", "gas/oil/water up to",
              "free water contact", "gas oil contact", "gas water contact", "water oil contact", "seal"
           interpretation_uuid (uuid.UUID or string, optional): uuid of the boundary feature Interpretation
              organizational object that the marker refers to.
              note: it is highly recommended that a related boundary feature interpretation is provided
           title (str, optional): the citation title to use for a new wellbore marker
              ignored if uuid is not None;
           originator (str, optional): the name of the person creating the wellbore marker, defaults to login id;
              ignored if uuid is not None
           extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the wellbore marker;
              ignored if uuid is not None

        returns:
           the newly created wellbore marker object
        """
        # verify that marker type is valid
        if marker_type is not None:
            assert marker_type in(["fault", "geobody", "horizon", "gas down to", "oil down to", "water down to",
                                      "gas up to", "oil up to", "water up to", "free water contact", "gas oil contact",
                                      "gas water contact", "water oil contact", "seal"]
                                    ) , "invalid marker type specified"

        self.model = parent_model
        self.wellbore_frame = parent_frame
        self.marker_index = marker_index
        self.uuid = None
        self.marker_type = marker_type
        self.interpretation_uuid = interpretation_uuid
        self.title = title
        self.originator = originator
        self.extra_metadata = extra_metadata
        if marker_node is not None:
            self._load_from_xml(marker_node = marker_node)
        if self.uuid is None:
            self.uuid = bu.new_uuid()
        assert self.uuid is not None

    def create_xml(self, parent_node, title = 'wellbore marker'):
        """Creates the xml tree for this wellbore marker."""

        assert self.uuid is not None
        wbm_node = self.model.new_obj_node('WellboreMarker', is_top_lvl_obj=False)
        wbm_node.set('uuid', str(self.uuid))

        # Citation block
        citation = self.__create_citation(root = wbm_node, title = title, originator = self.originator)

        # Add sub-elements to root node
        boundary_feature_dict = {'GeologicBoundaryKind': ['fault', 'geobody', 'horizon'],
                                 'FluidMarker': ['gas down to', 'gas up to', 'oil down to', 'oil up to', 'water down to',
                                                 'water up to'],
                                 'FluidContact': ['free water contact', 'gas oil contact', 'gas water contact', 'seal',
                                                  'water oil contact']
                                 }
        for k,v in boundary_feature_dict.items():
            if self.marker_type in v:
                boundary_kind = k

        wbm_gb_node = rqet.SubElement(wbm_node, ns['resqml2'] + boundary_kind)
        wbm_gb_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
        wbm_gb_node.text = str(self.marker_type)

        if self.interpretation_uuid is not None:
            interp_content_type = 'obj_' + self.marker_type.capitalize() + 'Interpretation'
            interp_root = self.model.root_for_uuid(uuid = self.interpretation_uuid)
            self.model.create_ref_node(flavour = 'Interpretation',
                                       title = rqet.find_tag(rqet.find_tag(interp_root, 'Citation'), 'Title').text,
                                       uuid = self.interpretation_uuid,
                                       content_type = interp_content_type,
                                       root=wbm_node)
        # Extra metadata
        if self.extra_metadata is None:
            self.extra_metadata = {}
        for key, value in self.extra_metadata.items():
            self.extra_metadata[str(key)] = str(value)
        if hasattr(self, 'extra_metadata') and self.extra_metadata:
            rqet.create_metadata_xml(node=wbm_node, extra_metadata=self.extra_metadata)

        if parent_node is not None:
            parent_node.append(wbm_node)
        return wbm_node

    def __create_citation(self, root = None, title = '', originator = None):
        """Creates a citation xml node and optionally appends as a child of root.

        arguments:
           root (optional): if not None, the newly created citation node is appended as
              a child to this node
           title (string): the citation title: a human readable string; this is the main point
              of having a citation node, so the argument should be used wisely
           originator (string, optional): the name of the human being who created the object
              which this citation is for; default is to use the login name

        returns:
           newly created citation xml node
        """

        if not title:
            title = '(no title)'

        citation = rqet.Element(ns['eml'] + 'Citation')
        citation.set(ns['xsi'] + 'type', ns['eml'] + 'Citation')
        citation.text = rqet.null_xml_text

        title_node = rqet.SubElement(citation, ns['eml'] + 'Title')
        title_node.set(ns['xsi'] + 'type', ns['eml'] + 'DescriptionString')
        title_node.text = title

        originator_node = rqet.SubElement(citation, ns['eml'] + 'Originator')
        if originator is None:
            try:
                originator = str(getpass.getuser())
            except Exception:
                originator = 'unknown'
        originator_node.set(ns['xsi'] + 'type', ns['eml'] + 'NameString')
        originator_node.text = originator

        creation_node = rqet.SubElement(citation, ns['eml'] + 'Creation')
        creation_node.set(ns['xsi'] + 'type', ns['xsd'] + 'dateTime')
        creation_node.text = time.now()

        format_node = rqet.SubElement(citation, ns['eml'] + 'Format')
        format_node.set(ns['xsi'] + 'type', ns['eml'] + 'DescriptionString')
        if rqet.pretend_to_be_fesapi:
            format_node.text = '[F2I-CONSULTING:fesapi]'
        else:
            format_node.text = citation_format

        if root is not None:
            root.append(citation)

        return citation

    def _load_from_xml(self, marker_node):
        """Load attributes from xml.

        This is invoked as part of the init method when an existing uuid is given.

        Returns:
           [bool]: True if successful
        """

        assert marker_node is not None
        # Load XML data
        uuid_str = rqet.find_tag_text(root = marker_node, tag_name = 'UUID')
        if uuid_str:
            self.uuid = bu.uuid_from_string(uuid_str)
        citation_tag = rqet.find_nested_tags(root = marker_node, tag_list=['Citation'])
        assert citation_tag is not None
        self.title = rqet.find_tag_text(root = citation_tag,
                                        tag_name = 'Title')
        self.originator = rqet.find_tag_text(root = citation_tag,
                                        tag_name = 'Originator')

        for boundary_feature_type in ['GeologicBoundaryKind', 'FluidMarker', 'FluidContact']:
            found_tag_text = rqet.find_tag_text(root = marker_node,
                                           tag_name = boundary_feature_type)
            if found_tag_text is not None:
                self.marker_type = found_tag_text
                break

        self.interpretation_uuid = rqet.find_nested_tags_text(root = marker_node, tag_list = ['Interpretation', 'UUID'])
        self.extra_metadata = rqet.load_metadata_from_xml(node = marker_node)

        return True




