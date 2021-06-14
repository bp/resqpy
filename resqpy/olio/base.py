"""Base class for generic resqml objects """

import logging
import warnings
from abc import ABCMeta, abstractmethod

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet


logger = logging.getLogger(__name__)


class BaseResqml(metaclass=ABCMeta):
    """Base class for generic RESQML objects
    
    Implements generic attributes such as uuid, root, part, title, originator.

    Implements generic magic methods, such as pretty printing and testing for
    equality.
    
    Example use::

        class AnotherResqmlObject(BaseResqml):
            
            _content_type = 'obj_anotherresqmlobjectrepresentation'

    """

    @property
    @abstractmethod
    def _content_type(self):
        """Definition of which RESQML object the class represents.
        
        Subclasses must overwrite this abstract attribute.
        """
        raise NotImplementedError

    def __init__(self, model, uuid=None, title=None, originator=None):
        """Load an existing resqml object, or create new.

        Args:
            model (resqpy.model.Model): Parent model
            uuid (str, optional): Load from existing uuid (if given), else create new.
            title (str, optional): Citation title
            originator (str, optional): Creator of object. By default, uses user id.
        """
        self.model = model
        self.title = title
        self.originator = originator

        if uuid is None:
            self.uuid = bu.new_uuid()
            logger.debug(f"Created new uuid for object {self}")
        else:
            self.uuid = uuid
            logger.debug(f"Loading existing object {self}")
            self.load_from_xml()
    
    # Define attributes self.part and self.root, using uuid as the primary key

    @property
    def part(self):
        """Part corresponding to self.uuid"""

        if self.uuid is None:
            raise ValueError('Cannot get part if uuid is None')
        return self.model.part_for_uuid(self.uuid)

    @property
    def root(self):
        """Node corresponding to self.uuid"""

        if self.uuid is None:
            raise ValueError('Cannot get root if uuid is None')
        return self.model.root_for_uuid(self.uuid)
    
    @root.setter
    def root(self, value):
        """Update self.uuid to match new root. Ensure root is added as a part"""

        new_uuid = rqet.uuid_for_part_root(value)
        if new_uuid is None:
            raise ValueError("Cannot set uuid to be None")
        self.uuid = new_uuid

    def load_from_xml(self):
        """Load citation block from XML"""

        # Citation block
        self.title = rqet.find_nested_tags_text(self.root, ['Citation', 'Title'])
        self.originator = rqet.find_nested_tags_text(self.root, ['Citation', 'Originator'])

    def create_xml(self, title=None, originator=None, ext_uuid=None, add_as_part=True):
        """Write citation block to XML
        
        Args:
            title (string): used as the citation Title text; should usually refer to the well name in a
                human readable way
            originator (string, optional): the name of the human being who created the deviation survey part;
                default is to use the login name
            add_as_part (boolean, default True): if True, the newly created xml node is added as a part
                in the model
    
        """

        assert self.uuid is not None

        if ext_uuid is None: ext_uuid = self.model.h5_uuid()

        # Create the root node
        node = self.model.new_obj_node(self._content_type)
        node.attrib['uuid'] = str(self.uuid)

        if add_as_part:
            self.model.add_part(self._content_type, self.uuid, node)
        # self.root = node

        assert self.root is not None

        # Citation block
        if title: self.title = title
        if originator: self.originator = originator
        self.model.create_citation(
            root=node, title=self.title, originator=self.originator
        )

    # Generic magic methods

    def __eq__(self, other):
        """Implements equals operator. By default, compare objects using uuid"""
        return self.uuid is not None and self.uuid == getattr(other, "uuid", None)

    def __ne__(self, other):
        """Implements not equal operator"""
        return not self.__eq__(other)

    def __repr__(self):
        """String representation"""
        return f"{self.__class__.__name__}(uuid={self.uuid}, title={self.title})"

    def _repr_html_(self):
        """IPython / Jupyter representation"""
        
        keys_to_display = ('uuid', 'title', 'originator')
        html =  f"<h3>{self.__class__.__name__}</h3>\n"
        for key in keys_to_display:
            html += f"<strong>{key}</strong>: {getattr(self, key)}<br>\n"
        return html

    # Include some aliases for root, but raise warnings if they are used
    # TODO: remove these aliases for self.node

    @property
    def root_node(self):
        warnings.warn("Attribute 'root_node' is deprecated. Use 'root'", DeprecationWarning)
        return self.root

    @property
    def node(self):
        warnings.warn("Attribute 'node' is deprecated. Sse 'root'", DeprecationWarning)
        return self.root
