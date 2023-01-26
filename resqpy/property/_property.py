"""Containing resqml property class"""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import resqpy.property
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet
import resqpy.property.property_collection as rqp_pc
from resqpy.olio.base import BaseResqpy


class Property(BaseResqpy):
    """Class for an individual property object; uses a single element PropertyCollection behind the scenes."""

    @property
    def resqml_type(self):
        """Returns the RESQML object class string for this Property."""
        root_node = self.root
        if root_node is not None:
            return rqet.node_type(root_node, strip_obj = True)
        if (not hasattr(self, 'collection') or self.collection.number_of_parts() != 1 or self.is_continuous()):
            return 'ContinuousProperty'
        if self.collection.points_for_part(self.collection.parts()[0]):
            return 'PointsProperty'
        return 'CategoricalProperty' if self.is_categorical() else 'DiscreteProperty'

    def __init__(self, parent_model, uuid = None, title = None, support_uuid = None, extra_metadata = None):
        """Initialises a resqpy Property object, either for an existing RESQML property, or empty for support.

        arguments:
           parent_model (model.Model): the model to which the property belongs
           uuid (uuid.UUID, optional): required if initialising from an existing RESQML property object
           title (str, optional): the citation title to use for the property; ignored if uuid is present
           support_uuid (uuid.UUID, optional): identifies the supporting representation for the property;
              ignored if uuid is present
           extra_metadata (dict, optional): if present, the dictionary items are added as extra metadata;
              ignored if uuid is present

        returns:
           new resqpy Property object

        :meta common:
        """

        self.collection = rqp_pc.PropertyCollection()
        super().__init__(model = parent_model, uuid = uuid, title = title, extra_metadata = extra_metadata)
        if support_uuid is not None:
            if self.collection.support_uuid is None:
                self.collection.set_support(model = parent_model, support_uuid = support_uuid)
            else:
                assert bu.matching_uuids(support_uuid, self.collection.support_uuid)

    def _load_from_xml(self):
        """Populates this property object from xml; not usually called directly."""

        part = self.part
        assert part is not None
        if self.collection is None:
            self.collection = rqp_pc.PropertyCollection()
        if self.collection.model is None:
            self.collection.model = self.model
        self.collection.add_part_to_dict(part)
        # duplicate extra metadata, as standard attribute in BaseResqpy
        self.extra_metadata = self.collection.extra_metadata_for_part(part)
        self.collection.has_single_property_kind_flag = True
        self.collection.has_single_indexable_element_flag = True
        self.collection.has_multiple_realizations_flag = False
        assert self.collection.number_of_parts() == 1

    @classmethod
    def from_singleton_collection(cls, property_collection: rqp_pc.PropertyCollection):
        """Populates a new Property from a PropertyCollection containing just one part.

        arguments:
           property_collection (PropertyCollection): the singleton collection from which to populate this Property
        """

        # Validate
        assert property_collection.model is not None
        assert property_collection is not None
        assert property_collection.number_of_parts() == 1

        # Instantiate the object i.e. call the class __init__ method
        part = property_collection.parts()[0]
        prop = cls(parent_model = property_collection.model,
                   uuid = property_collection.uuid_for_part(part),
                   support_uuid = property_collection.support_uuid,
                   title = property_collection.citation_title_for_part(part),
                   extra_metadata = property_collection.extra_metadata_for_part(part))

        # Edit properties of collection attribute
        prop.collection.has_single_property_kind_flag = True
        prop.collection.has_single_indexable_element_flag = True
        prop.collection.has_multiple_realizations_flag = False

        return prop

    @classmethod
    def from_array(cls,
                   parent_model,
                   cached_array,
                   source_info,
                   keyword,
                   support_uuid,
                   property_kind = None,
                   local_property_kind_uuid = None,
                   indexable_element = None,
                   facet_type = None,
                   facet = None,
                   discrete = False,
                   uom = None,
                   null_value = None,
                   time_series_uuid = None,
                   time_index = None,
                   realization = None,
                   count = 1,
                   points = False,
                   const_value = None,
                   string_lookup_uuid = None,
                   find_local_property_kind = True,
                   expand_const_arrays = False,
                   dtype = None,
                   extra_metadata = {}):
        """Populates a new Property from a numpy array and metadata; NB. Writes data to hdf5 and adds part to model.

        arguments:
           parent_model (model.Model): the model to which the new property belongs
           cached_array: a numpy array to be made into a property; for a constant array set cached_array to None
              (and use const_value)
           source_info (string): typically the name of a file from which the array has been read but can be any
              information regarding the source of the data
           keyword (string): this will be used as the citation title for the property
           support_uuid (uuid): the uuid of the supporting representation
           property_kind (string): resqml property kind (required unless a local propery kind is identified with
              local_property_kind_uuid)
           local_property_kind_uuid (uuid.UUID or string, optional): uuid of local property kind, or None if a
              standard property kind applies or if the local property kind is identified by its name (see
              find_local_property_kind)
           indexable_element (string): the indexable element in the supporting representation; if None
              then the 'typical' indexable element for the class of supporting representation will be assumed
           facet_type (string, optional): resqml facet type, or None; resqpy only supports at most one facet per
              property though RESQML allows for multiple)
           facet (string, optional): resqml facet, or None; resqpy only supports at most one facet per property
              though RESQML allows for multiple)
           discrete (boolean, optional, default False): if True, the array should contain integer (or boolean)
              data, if False, float; set True for any discrete data including categorical
           uom (string, optional, default None): the resqml units of measure for the data; required for
              continuous (float) array
           null_value (int, optional, default None): if present, this is used in the metadata to indicate that
              the value is to be interpreted as a null value wherever it appears in the discrete data; not used
              for continuous data where NaN is always the null value
           time_series_uuid (optional): the uuid of the full or reduced time series that the time_index is for
           time_index (integer, optional, default None): if not None, the time index for the property (see also
              time_series_uuid)
           realization (int, optional): realization number, or None
           count (int, default 1): the number of values per indexable element; if greater than one then this
              must be the fastest cycling axis in the cached array, ie last index
           const_value (int, float or bool, optional): the value with which a constant array is filled;
              required if cached_array is None, must be None otherwise
           string_lookup_uuid (optional): if present, the uuid of the string table lookup which the categorical data
              relates to; if None, the property will not be configured as categorical
           find_local_property_kind (boolean, default True): if True, local property kind uuid need not be provided as
              long as the property_kind is set to match the title of the appropriate local property kind object
           expand_const_arrays (boolean, default False): if True, and a const_value is given, the array will be fully
              expanded and written to the hdf5 file; the xml will then not indicate that it is constant
           dtype (numpy dtype, optional): if present, the elemental data type to use when writing the array to hdf5
           extra_metadata (optional): if present, a dictionary of extra metadata to be added for the part

        returns:
           new Property object built from numpy array; the hdf5 data has been written, xml created and the part
           added to the model

        notes:
           this method writes to the hdf5 file and creates the xml node, which is added as a part to the model;
           calling code must still call the model's store_epc() method;
           this from_array() method is a convenience method calling self.collection.set_support(),
           self.prepare_import(), self.write_hdf5() and self.create_xml()

        :meta common:
        """
        # Validate
        assert parent_model is not None
        assert cached_array is not None or const_value is not None

        # Instantiate the object i.e. call the class __init__ method
        prop = cls(parent_model = parent_model,
                   title = keyword,
                   support_uuid = support_uuid,
                   extra_metadata = extra_metadata)

        # Prepare array data in collection attribute and add to model
        prop.collection.set_support(model = prop.model,
                                    support_uuid = support_uuid)  # this can be expensive; todo: optimise
        if prop.collection.model is None:
            prop.collection.model = prop.model
        prop.prepare_import(cached_array,
                            source_info,
                            keyword,
                            discrete = discrete,
                            uom = uom,
                            time_index = time_index,
                            null_value = null_value,
                            property_kind = property_kind,
                            local_property_kind_uuid = local_property_kind_uuid,
                            facet_type = facet_type,
                            facet = facet,
                            realization = realization,
                            indexable_element = indexable_element,
                            count = count,
                            points = points,
                            const_value = const_value)
        prop.write_hdf5(expand_const_arrays = expand_const_arrays, dtype = dtype)
        prop.create_xml(support_uuid = support_uuid,
                        time_series_uuid = time_series_uuid,
                        string_lookup_uuid = string_lookup_uuid,
                        property_kind_uuid = local_property_kind_uuid,
                        find_local_property_kind = find_local_property_kind,
                        expand_const_arrays = expand_const_arrays,
                        extra_metadata = extra_metadata)
        return prop

    def array_ref(self, dtype = None, masked = False, exclude_null = False):
        """Returns a (cached) numpy array containing the property values.

        arguments:
           dtype (str or type, optional): the required dtype of the array (usually float, int or bool)
           masked (boolean, default False): if True, a numpy masked array is returned, with the mask determined by
              the inactive cell mask in the case of a Grid property
           exclude_null (boolean, default False): if True and masked is True, elements whose value is the null value
              (NaN for floats) will be masked out

        returns:
           numpy array

        :meta common:
        """

        return self.collection.cached_part_array_ref(self.part,
                                                     dtype = dtype,
                                                     masked = masked,
                                                     exclude_null = exclude_null)

    def is_continuous(self):
        """Returns boolean indicating that the property contains continuous (ie. float) data.

        :meta common:
        """
        return self.collection.continuous_for_part(self.part)

    def is_points(self):
        """Returns boolean indicating that the property is a points property."""

        return self.collection.points_for_part(self.part)

    def is_categorical(self):
        """Returns boolean indicating that the property contains categorical data.

        :meta common:
        """
        return self.collection.part_is_categorical(self.part)

    def null_value(self):
        """Returns int being the null value for the (discrete) property."""
        return self.collection.null_value_for_part(self.part)

    def count(self):
        """Returns int being the number of values for each element (usually 1)."""
        return self.collection.count_for_part(self.part)

    def indexable_element(self):
        """Returns string being the indexable element for the property.

        :meta common:
        """
        return self.collection.indexable_for_part(self.part)

    def property_kind(self):
        """Returns string being the property kind for the property.

        :meta common:
        """
        return self.collection.property_kind_for_part(self.part)

    def local_property_kind_uuid(self):
        """Returns uuid of the local property kind for the property (if applicable, otherwise None)."""
        return self.collection.local_property_kind_uuid(self.part)

    def facet_type(self):
        """Returns string being the facet type for the property, or None if no facet present.

        note: resqpy currently supports at most one facet per property, though RESQML allows for multiple.

        :meta common:
        """
        return self.collection.facet_type_for_part(self.part)

    def facet(self):
        """Returns string being the facet value for the property, or None if no facet present.

        note: resqpy currently supports at most one facet per property, though RESQML allows for multiple.

        :meta common:
        """
        return self.collection.facet_for_part(self.part)

    def time_series_uuid(self):
        """Returns the uuid of the related time series (if applicable, otherwise None)."""
        return self.collection.time_series_uuid_for_part(self.part)

    def time_index(self):
        """Returns int being the index into the related time series (if applicable, otherwise None)."""
        return self.collection.time_index_for_part(self.part)

    def minimum_value(self):
        """Returns the minimum value for the property as stored in xml (float or int or None)."""
        return self.collection.minimum_value_for_part(self.part)

    def maximum_value(self):
        """Returns the maximum value for the property as stored in xml (float or int or None)."""
        return self.collection.maximum_value_for_part(self.part)

    def uom(self):
        """Returns string being the units of measure (for a continuous property, otherwise None).

        :meta common:
        """
        return self.collection.uom_for_part(self.part)

    def string_lookup_uuid(self):
        """Returns the uuid of the related string lookup table (for a categorical property, otherwise None)."""
        return self.collection.string_lookup_uuid_for_part(self.part)

    def string_lookup(self):
        """Returns a resqpy StringLookup table (for a categorical property, otherwise None)."""
        return self.collection.string_lookup_for_part(self.part)

    def constant_value(self):
        """For a constant property, returns the constant value (float or int or bool, or None if not constant)."""
        return self.collection.constant_value_for_part(self.part)

    def prepare_import(self,
                       cached_array,
                       source_info,
                       keyword,
                       discrete = False,
                       uom = None,
                       time_index = None,
                       null_value = None,
                       property_kind = None,
                       local_property_kind_uuid = None,
                       facet_type = None,
                       facet = None,
                       realization = None,
                       indexable_element = None,
                       count = 1,
                       const_value = None,
                       points = False):
        """Takes a numpy array and metadata and sets up a single array import list; not usually called directly.

        note:
           see the documentation for the convenience method from_array()
        """
        assert self.collection.number_of_parts() == 0
        assert not self.collection.imported_list
        self.collection.add_cached_array_to_imported_list(cached_array,
                                                          source_info,
                                                          keyword,
                                                          discrete = discrete,
                                                          uom = uom,
                                                          time_index = time_index,
                                                          null_value = null_value,
                                                          property_kind = property_kind,
                                                          local_property_kind_uuid = local_property_kind_uuid,
                                                          facet_type = facet_type,
                                                          facet = facet,
                                                          realization = realization,
                                                          indexable_element = indexable_element,
                                                          count = count,
                                                          const_value = const_value,
                                                          points = points)

    def write_hdf5(self, file_name = None, mode = 'a', expand_const_arrays = False, dtype = None):
        """Writes the array data to the hdf5 file; not usually called directly.

        arguments:
           file_name (str, optional): if present, the path of the hdf5 file to use; strongly recommended not to
              set this argument
           mode (str, default 'a'): 'a' or 'w' being the mode to open the hdf5 file in; strongly recommended to use 'a'
           expand_const_arrays (bool, default False): if True and the array is a constant array then a fully populated
              array is generated and stored (otherwise the constant value is held in xml and no hdf5 data is needed)
           dtype (numpy dtype, optional): if present, the elemental data type to use when writing the array to hdf5

        notes:
           see the documentation for the convenience method from_array()
        """
        if not self.collection.imported_list:
            log.warning('no imported Property array to write to hdf5')
            return
        self.collection.write_hdf5_for_imported_list(file_name = file_name,
                                                     mode = mode,
                                                     expand_const_arrays = expand_const_arrays,
                                                     dtype = dtype)

    def create_xml(self,
                   ext_uuid = None,
                   support_uuid = None,
                   time_series_uuid = None,
                   string_lookup_uuid = None,
                   property_kind_uuid = None,
                   find_local_property_kind = True,
                   expand_const_arrays = False,
                   extra_metadata = {}):
        """Creates an xml tree for the property and adds it as a part to the model; not usually called directly.

        note:
           see the documentation for the convenience method from_array()
           NB. this method has the deliberate side effect of modifying the uuid and title of self to match those
           of the property collection part!
        """
        if not self.collection.imported_list:
            log.warning('no imported Property array to create xml for')
            return
        self.collection.create_xml_for_imported_list_and_add_parts_to_model(
            ext_uuid = ext_uuid,
            support_uuid = support_uuid,
            time_series_uuid = time_series_uuid,
            selected_time_indices_list = None,
            string_lookup_uuid = string_lookup_uuid,
            property_kind_uuid = property_kind_uuid,
            find_local_property_kinds = find_local_property_kind,
            expand_const_arrays = expand_const_arrays,
            extra_metadata = extra_metadata)
        self.collection.has_single_property_kind_flag = True
        self.collection.has_single_uom_flag = True
        self.collection.has_single_indexable_element_flag = True
        self.collection.has_multiple_realizations_flag = False
        part = self.collection.parts()[0]
        assert part is not None
        self.uuid = self.collection.uuid_for_part(part)
        self.title = self.collection.citation_title_for_part(part)
        return self.root
