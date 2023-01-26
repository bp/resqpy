"""WellboreMarker class."""

# Nexus is a registered trademark of the Halliburton Company
# RMS and ROXAR are registered trademarks of Roxar Software Solutions AS, an Emerson company

import logging

log = logging.getLogger(__name__)

import resqpy.olio.xml_et as rqet
import resqpy.olio.uuid as bu
from resqpy.olio.xml_namespaces import curly_namespace as ns


class WellboreMarker():
    """Class to handle RESQML WellboreMarker objects.

    note:
       wellbore markers are not high level RESQML objects
    """

    resqml_type = 'WellboreMarker'

    boundary_feature_dict = {
        'GeologicBoundaryKind': ['fault', 'geobody', 'horizon'],
        'FluidMarker': ['gas down to', 'gas up to', 'oil down to', 'oil up to', 'water down to', 'water up to'],
        'FluidContact': ['free water contact', 'gas oil contact', 'gas water contact', 'seal', 'water oil contact']
    }

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
        """Creates a new wellbore marker object and loads it from xml or populates it from arguments.

        arguments:
           parent_model (model.Model object): the model which the new wellbore marker belongs to
           parent_frame (wellbore_marker_frame.WellboreMarkerFramer object): the wellbore marker frame to which
              the wellbore marker belongs
           marker_index (int): index of the wellbore marker in the parent WellboreMarkerFrame object
           marker_node (xml node, optional): if given, loads from xml. Else, creates new
           marker_type (str, optional): the type of geologic, fluid or contact feature
              e.g. "fault", "geobody", "horizon ", "gas/oil/water down to", "gas/oil/water up to",
              "free water contact", "gas oil contact", "gas water contact", "water oil contact", "seal";
              ignored if marker_node is present
           interpretation_uuid (uuid.UUID or string, optional): uuid of the boundary feature interpretation
              organizational object that the marker refers to; ignored if marker_node is present
           title (str, optional): the citation title to use for a new wellbore marker; ignored if marker_node is present
           originator (str, optional): the name of the person creating the wellbore marker, defaults to login id;
              ignored if uuid is not None; ignored if marker_node is present
           extra_metadata (dict, optional): string key, value pairs to add as extra metadata for the wellbore marker;
              ignored if uuid is not None; ignored if marker_node is present

        returns:
           the newly created wellbore marker object

        note:
           it is highly recommended that a related boundary feature interpretation uuid is provided
        """
        # verify that marker type is valid
        if marker_type is not None:
            assert marker_type in ([
                "fault", "geobody", "horizon", "gas down to", "oil down to", "water down to", "gas up to", "oil up to",
                "water up to", "free water contact", "gas oil contact", "gas water contact", "water oil contact", "seal"
            ]), "invalid marker type specified"

        self.model = parent_model
        self.wellbore_frame = parent_frame
        self.uuid = None
        self.marker_index = marker_index
        self.marker_type = None
        self.interpretation_uuid = None
        self.title = None
        self.originator = None
        self.extra_metadata = None
        if marker_node is None:
            self.marker_type = marker_type
            self.interpretation_uuid = interpretation_uuid
            self.title = title
            self.originator = originator
            self.extra_metadata = extra_metadata
        else:
            self._load_from_xml(marker_node = marker_node)
        if self.uuid is None:
            self.uuid = bu.new_uuid()
        assert self.uuid is not None

    def create_xml(self, parent_node, title = 'wellbore marker'):
        """Creates the xml tree for this wellbore marker.

        arguments:
           parent_node (xml node): the root node of the WellboreMarkerFrame object to which the newly created node
               will be appended
           title (string, default "wellbore marker"): the citation title of the newly created node; only used if
               self.title is None

        returns:
           the newly created xml node
        """

        assert self.uuid is not None
        wbm_node = self.model.new_obj_node('WellboreMarker', is_top_lvl_obj = False)
        wbm_node.set('uuid', str(self.uuid))

        # Citation block
        if self.title:
            title = self.title
        citation = self.model.create_citation(root = wbm_node, title = title, originator = self.originator)

        # Add sub-elements to root node
        for k, v in WellboreMarker.boundary_feature_dict.items():
            if self.marker_type in v:
                boundary_kind = k
                break

        wbm_gb_node = rqet.SubElement(wbm_node, ns['resqml2'] + boundary_kind)
        # wbm_gb_node.set(ns['xsi'] + 'type', ns['xsd'] + 'string')
        wbm_gb_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'GeologicBoundaryKind')
        wbm_gb_node.text = str(self.marker_type)

        if self.interpretation_uuid is not None:
            interp_content_type = 'obj_' + self.marker_type.capitalize() + 'Interpretation'
            interp_root = self.model.root_for_uuid(uuid = self.interpretation_uuid)
            self.model.create_ref_node(flavour = 'Interpretation',
                                       title = rqet.find_tag(rqet.find_tag(interp_root, 'Citation'), 'Title').text,
                                       uuid = self.interpretation_uuid,
                                       content_type = interp_content_type,
                                       root = wbm_node)
        # Extra metadata
        if hasattr(self, 'extra_metadata') and self.extra_metadata:
            rqet.create_metadata_xml(node = wbm_node, extra_metadata = self.extra_metadata)

        if parent_node is not None:
            parent_node.append(wbm_node)

        return wbm_node

    def _load_from_xml(self, marker_node):
        """Load attributes from xml; invoked as part of the init method when an existing uuid is given.

        returns:
           bool: True if successful
        """

        assert marker_node is not None

        # Load XML data
        uuid_str = marker_node.attrib.get('uuid')
        if uuid_str:
            self.uuid = bu.uuid_from_string(uuid_str)
        citation_tag = rqet.find_nested_tags(root = marker_node, tag_list = ['Citation'])
        assert citation_tag is not None
        self.title = rqet.find_tag_text(root = citation_tag, tag_name = 'Title')
        self.originator = rqet.find_tag_text(root = citation_tag, tag_name = 'Originator')

        self.marker_type = None
        for boundary_feature_type in ['GeologicBoundaryKind', 'FluidMarker', 'FluidContact']:
            found_tag_text = rqet.find_tag_text(root = marker_node, tag_name = boundary_feature_type)
            if found_tag_text is not None:
                self.marker_type = found_tag_text
                break

        self.interpretation_uuid = bu.uuid_from_string(
            rqet.find_nested_tags_text(root = marker_node, tag_list = ['Interpretation', 'UUID']))

        self.extra_metadata = rqet.load_metadata_from_xml(node = marker_node)

        # patch to deduce a boundary kind from the interpretation object
        if self.marker_type is None and self.interpretation_uuid is not None:
            interp_type = self.model.type_of_uuid(self.interpretation_uuid, strip_obj = True)
            if interp_type == 'HorizonInterpretation':
                self.marker_type = 'horizon'
            elif interp_type == 'FaultInterpretation':
                self.marker_type = 'fault'
            elif interp_type == 'GeobodyBoundaryInterpretation':
                self.marker_type = 'geobody'

        assert self.marker_type is not None, f'geologic boundary tyoe not determined for marker {self.title}'

        return True
