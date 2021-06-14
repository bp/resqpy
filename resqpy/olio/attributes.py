""" Easy attributes for defining what should be saved to XML & HDF5 """

import logging
from abc import abstractmethod
from dataclasses import dataclass

import resqpy.olio.xml_et as rqet
import resqpy.olio.weights_and_measures as bwam
from resqpy.olio.xml_namespaces import curly_namespace as ns


logger = logging.getLogger(__name__)
    

class BaseAttribute:

    xml_types = dict(
        double           = dict(dtype=float, xml_ns='xsd'),
        positiveInteger  = dict(dtype=int,   xml_ns='xsd'),
        boolean          = dict(dtype=bool,  xml_ns='xsd'),
        LengthUom        = dict(dtype=str,   xml_ns='eml'),
        PlaneAngleUom    = dict(dtype=str,   xml_ns='eml'),
        DoubleHdf5Array  = dict(dtype=float, xml_ns='resqml2'),
    )

    def __init__(self, key, tag, xml_type, required=True, writable=True):
        """Definition of an attribute stored to disk
        
        Args:
            key (str): Name of attribute where value is stored in memory
            tag (str): Tag name in XML where value is stored to disk
            xtype (str): XML type, one of BaseAttribute.xml_types
            required (bool): If True, should always be given
            writable (bool): If False, do not try to write to disk, only load.
        """
        self.key = key
        self.tag = tag
        self.xml_type = xml_type
        self.required = required
        self.writable = writable
        
        # Lookup properties for the xml type
        xml_props = self.xml_types[xml_type]
        self.dtype = xml_props["dtype"]
        self.xml_ns = xml_props["xml_ns"]

    @abstractmethod
    def load(self, obj):
        raise NotImplementedError

    @abstractmethod
    def write_xml(self, obj):
        raise NotImplementedError


class XmlAttribute(BaseAttribute):
    """Definition of an attribute stored in XML """

    def load(self, obj):
        """Load the value from XML, set as attribute of obj
        
        Args:
            obj: python object for which to load. Must have attribute obj.root
        """

        logger.debug(f"Loading attribute {self} from XML")

        assert hasattr(obj, 'root')
        tag_list = self.tag.split('/')
        value = rqet.find_nested_tags_cast(obj.root, tag_list, dtype=self.dtype)
        if self.required and value is None:
            raise ValueError(f'Could not load required attribute {self}')
        setattr(obj, self.key, value)

    def write_xml(self, obj):
        """Write the object to XML"""

        if not self.writable:
            return

        if '/' in self.tag:
            raise NotImplementedError(
                "XmlAttribute cannot currently write nested attributes"
            )
        
        logger.debug(f"Writing attribute {self} to XML")

        node = obj.root
        assert node is not None

        value = getattr(obj, self.key)

        # Type-specific casting
        if self.xml_type == 'boolean':
            value = str(value).lower()
        elif self.xml_type == 'LengthUom':
            value = bwam.rq_length_unit(value)
        elif self.xml_type == 'PlaneAngleUom':
            if str(value).strip().lower().startswith('deg'):
                value = 'dega'
            else:
                value = 'rad'

        attr_node = rqet.SubElement(node, ns['resqml2'] + self.tag)
        attr_node.set(ns['xsi'] + 'type', ns[self.xml_ns] + self.xml_type)
        attr_node.text = str(value)


class HdfAttribute(BaseAttribute):
    """Definition of an attribute stord in HDF5 """

    def load(self, obj):
        """Load the array from HDF5, set as attribute of obj"""

        logger.debug(f"Loading attribute {self} from HDF")

        model = obj.model
        root = obj.root

        array_node = rqet.find_tag(root, self.tag, must_exist=self.required)
        assert rqet.node_type(array_node) in ['DoubleHdf5Array', 'IntegerHdf5Array', 'Point3dHdf5Array']

        h5_key_pair = model.h5_uuid_and_path_for_node(array_node, tag="Values")
        if h5_key_pair is None: return None
        return model.h5_array_element(h5_key_pair, index=None, cache_array=True,
            dtype=self.dtype, object=obj, array_attribute=self.key)

    def write_xml(self, obj):
        """Write the object to XML, set as attribute of obj"""
            
        if not self.writable:
            return

        logger.debug(f"Writing attribute {self} to HDF")

        node = obj.root
        model = obj.model
        obj_uuid = obj.uuid
        ext_uuid = model.h5_uuid()
        assert node is not None

        attr_node = rqet.SubElement(node, ns['resqml2'] + self.tag)
        attr_node.set(ns['xsi'] + 'type', ns[self.xml_ns] + self.xml_type)
        attr_node.text = rqet.null_xml_text

        attr_values_node = rqet.SubElement(attr_node, ns['resqml2'] + 'Values')
        attr_values_node.set(ns['xsi'] + 'type', ns['resqml2'] + 'Hdf5Dataset')
        attr_values_node.text = rqet.null_xml_text

        model.create_hdf5_dataset_ref(ext_uuid, obj_uuid, self.tag, root=attr_values_node)
