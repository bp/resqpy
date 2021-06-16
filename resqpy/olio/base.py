"""Base class for generic resqml objects """

import logging
import warnings
from abc import ABCMeta, abstractmethod

import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet


logger = logging.getLogger(__name__)


class BaseResqpy(metaclass=ABCMeta):
    """Base class for generic resqpy classes
    
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

    def __init__(self, model, uuid=None, title=None, originator=None, root_node=None):
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

        if root_node is not None:
            warnings.warn("root_node parameter is deprecated, use uuid instead", DeprecationWarning)
            uuid = rqet.uuid_for_part_root(root_node)

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
        """Standard part name corresponding to self.uuid"""

        return rqet.part_name_for_object(self.resqml_type, self.uuid)

    @property
    def root(self):
        """XML node corresponding to self.uuid"""

        return self.model.root_for_uuid(self.uuid)

    def load_from_xml(self):
        """Load citation block from XML.
        
        Note: derived classes should extend this to load other XML and HDF attributes
        """

        # Citation block
        self.title = rqet.find_nested_tags_text(self.root, ['Citation', 'Title'])
        self.originator = rqet.find_nested_tags_text(self.root, ['Citation', 'Originator'])

    def create_xml(self, title=None, originator=None, add_as_part=False, reuse=False):
        """Write citation block to XML
        
        Note:

            `add_as_part` is False by default in this base method. Derived classes should typically
            extend this method to complete the XML representation, and then finally ensure the node
            is added as a part to the model.

            if `reuse` is True, a side effect of this method may be to modify the uuid of self;
            calling code should typically look for such a change and if detected, abandon any
            further work on building or adding the xml node (as it is already complete)

        Args:
            title (string): used as the citation Title text; should usually refer to the well name in a
                human readable way
            originator (string, optional): the name of the human being who created the deviation survey part;
                default is to use the login name
            add_as_part (boolean): if True, the newly created xml node is added as a part
                in the model
            reuse (boolean, default False): if True, the xml for other parts in the model of the same class
                is considered for reuse and if suitable the uuid of this object is modified

        Returns:
            node: the newly created root node, or reused root node as applicable
        """

        assert self.uuid is not None

        if reuse:
            if self.root is not None: return self.root
            uuid_list = self.model.uuids(obj_type = self.resqml_type)
            for other_uuid in uuid_list:
                other = self.__class__(self.model, uuid = other_uuid)
                if self == other:
                    logger.debug(f'reusing equivalent resqml object with uuid {other_uuid}')
                    self.uuid = other_uuid  # NB: change of uuid for this object
                    assert self.root is not None
                    return self.root

        # Create the root node
        node = self.model.new_obj_node(self.resqml_type)
        node.attrib['uuid'] = str(self.uuid)

        # Citation block
        if title: self.title = title
        if originator: self.originator = originator
        self.model.create_citation(
            root=node, title=self.title, originator=self.originator
        )

        if add_as_part:
            self.model.add_part(self.resqml_type, self.uuid, node)
            assert self.root is not None
        
        return node

    # Generic magic methods

    def __eq__(self, other):
        """Implements equals operator. Compares class type and uuid"""
        if hasattr(self, 'is_equivalent'): return self.is_equivalent(other)
        if not isinstance(other, self.__class__): return False
        other_uuid = getattr(other, "uuid", None)
        return bu.matching_uuids(self.uuid, other_uuid)

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
