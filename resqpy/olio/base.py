"""Base class for generic resqml objects."""

import logging

logger = logging.getLogger(__name__)

import warnings
from abc import ABCMeta, abstractmethod

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet


class BaseResqpy(metaclass = ABCMeta):
    """Base class for generic resqpy classes.

    Implements generic attributes such as uuid, root, part, title, originator.

    Implements generic magic methods, such as pretty printing and testing for
    equality.

    Example use::

        class AnotherResqpyObject(BaseResqpy):

            resqml_type = 'obj_anotherresqmlobjectrepresentation'
    """

    @property
    @abstractmethod
    def resqml_type(self):
        """Definition of which RESQML object the class represents.

        Subclasses must overwrite this abstract attribute.
        """
        raise NotImplementedError

    def __init__(self, model, uuid = None, title = None, originator = None, extra_metadata = None):
        """Load an existing resqml object, or create new.

        arguments:
            model (resqpy.model.Model): Parent model
            uuid (str, optional): Load from existing uuid (if given), else create new.
            title (str, optional): Citation title
            originator (str, optional): Creator of object. By default, uses user id.
        """
        self.model = model
        self.title = title  #: Citation title
        self.originator = originator  #: Creator of object. By default, user id.
        self.extra_metadata = {}
        if extra_metadata:
            self.extra_metadata = extra_metadata
            self._standardise_extra_metadata()  # has side effect of making a copy

        if uuid is None:
            self.uuid = bu.new_uuid()  #: Unique identifier
        else:
            self.uuid = uuid
            root_node = self.root
            citation_node = rqet.find_tag(root_node, 'Citation')
            if citation_node is not None:
                self.title = rqet.find_tag_text(citation_node, 'Title')
                self.originator = rqet.find_tag_text(citation_node, 'Originator')
            self.extra_metadata = rqet.load_metadata_from_xml(root_node)
            self._load_from_xml()

    # usually overridden by derived class, unless generic code above handles all attributes
    def _load_from_xml(self):
        pass

    # Define attributes self.part and self.root, using uuid as the primary key

    @property
    def part(self):
        """Standard part name corresponding to self.uuid."""

        # following caused trouble when resqml_type dynamically determined
        #       return rqet.part_name_for_object(self.resqml_type, self.uuid)
        return self.model.part_for_uuid(self.uuid)

    @property
    def root(self):
        """XML node corresponding to self.uuid."""

        return self.model.root_for_uuid(self.uuid)

    @property
    def citation_title(self):
        """Citation block title equivalent to self.title."""

        return self.title

    def try_reuse(self):
        """Look for an equivalent existing RESQML object and modify the uuid of this object if found.

        returns:
           boolean: True if an equivalent object was found, False if not

        note:
           by design this method may change this object's uuid as a side effect
        """

        assert self.uuid is not None
        if self.root is not None:
            return True
        uuid_list = self.model.uuids(obj_type = self.resqml_type)
        for other_uuid in uuid_list:
            if bu.matching_uuids(self.uuid, other_uuid):
                logger.debug(f'reusing existing xml for uuid {other_uuid}')
                return True
            try:
                other = self.__class__(self.model, uuid = other_uuid)
            except Exception:
                return False
            if self == other:
                logger.debug(f'reusing equivalent resqml object with uuid {other_uuid}')
                self.uuid = other_uuid  # NB: change of uuid for this object
                assert self.root is not None
                return True
        return False

    def create_xml(self, title = None, originator = None, extra_metadata = None, add_as_part = False):
        """Write citation block to XML.

        Note:

            `add_as_part` is False by default in this base method. Derived classes should typically
            extend this method to complete the XML representation, and then finally ensure the node
            is added as a part to the model.

        arguments:
            title (string): used as the citation Title text
            originator (string, optional): the name of the human being who created the deviation survey part;
                default is to use the login name
            extra_metadata (dict, optional): extra metadata items to be added
            add_as_part (boolean): if True, the newly created xml node is added as a part
                in the model

        returns:
            node: the newly created root node
        """

        assert self.uuid is not None

        # Create the root node
        node = self.model.new_obj_node(self.resqml_type)
        node.attrib['uuid'] = str(self.uuid)

        # Citation block
        if title:
            self.title = title
        if originator:
            self.originator = originator
        self.model.create_citation(root = node, title = self.title, originator = self.originator)

        # Extra metadata
        if extra_metadata:
            if not hasattr(self, 'extra_metadata'):
                self.extra_metadata = {}
            for key, value in extra_metadata.items():
                self.extra_metadata[str(key)] = str(value)
        if hasattr(self, 'extra_metadata') and self.extra_metadata:
            rqet.create_metadata_xml(node = node, extra_metadata = self.extra_metadata)

        if add_as_part:
            self.model.add_part(self.resqml_type, self.uuid, node)
            assert self.root is not None

        return node

    def append_extra_metadata(self, meta_dict):
        """Append a given dictionary of metadata to the existing metadata."""
        for key in meta_dict:
            self.extra_metadata[key] = meta_dict[key]
        self._standardise_extra_metadata()

    def _standardise_extra_metadata(self):
        if self.extra_metadata:
            em = {}
            for key, value in self.extra_metadata.items():
                em[str(key)] = str(value)
            self.extra_metadata = em

    # Generic magic methods

    def __eq__(self, other):
        """Implements equals operator; uses is_equivalent() otherwise compares class type and uuid."""
        if hasattr(self, 'is_equivalent'):
            return self.is_equivalent(other)
        if not isinstance(other, self.__class__):
            return False
        other_uuid = getattr(other, "uuid", None)
        return bu.matching_uuids(self.uuid, other_uuid)

    def __ne__(self, other):
        """Implements not equal operator."""
        return not self.__eq__(other)

    def __repr__(self):
        """String representation."""
        return f"{self.__class__.__name__}(uuid={self.uuid}, title={self.title})"

    def _repr_html_(self):
        """Return HTML for IPython / Jupyter representation."""

        keys_to_display = ('uuid', 'title', 'originator')
        html = f"<h3>{self.__class__.__name__}</h3>\n"
        for key in keys_to_display:
            html += f"<strong>{key}</strong>: {getattr(self, key)}<br>\n"
        return html
