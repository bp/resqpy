"""Class handling set of RESQML properties using attribute syntax for properties and their metadata."""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import numpy as np
import numpy.ma as ma

import resqpy.property as rqp


class ApsProperty:
    """Class holding a single property with attribute style read access to metadata items."""

    # note: this class could be private to the AttributePropertySet class

    def __init__(self, aps, part):
        """Initialise a single property from a property set with attribute style read access to metadata items."""
        self.aps = aps
        self._part = part
        self.key = aps._key(part)

    # NB. the following are read-only attributes

    @property
    def part(self):
        """The part (string) identifier for this property."""
        return self._part

    @part.setter
    def part(self, value):
        self._part = value

    @property
    def node(self):
        """The xml root node for this property."""
        return self.aps.node_for_part(self.part)

    @property
    def uuid(self):
        """The uuid for this property."""
        return self.aps.uuid_for_part(self.part)

    @property
    def array_ref(self):
        """The cached numpy array of values for this property."""
        return self.aps.cached_part_array_ref(self.part)

    @property
    def values(self):
        """A copy of the numpy array of values for this property."""
        return self.array_ref.copy()

    @property
    def property_kind(self):
        """The property kind of this property."""
        return self.aps.property_kind_for_part(self.part)

    @property
    def facet_type(self):
        """The facet type for this property (may be None)."""
        return self.aps.facet_type_for_part(self.part)

    @property
    def facet(self):
        """The facet value for this property (may be None)."""
        return self.aps.facet_for_part(self.part)

    @property
    def indexable(self):
        """The indexable element for this property (synonymous with indexable_element)."""
        return self.aps.indexable_for_part(self.part)

    @property
    def indexable_element(self):
        """The indexable element for this property (synonymous with indexable)."""
        return self.indexable

    @property
    def is_continuous(self):
        """Boolean indicating whether this property is continuous."""
        return self.aps.continuous_for_part(self.part)

    @property
    def is_categorical(self):
        """Boolean indicating whether this property is categorical."""
        return self.aps.part_is_categorical(self.part)

    @property
    def is_discrete(self):
        """Boolean indicating whether this property is discrete (False for categorical properties)."""
        return not (self.is_continuous or self.is_categorical)

    @property
    def is_points(self):
        """Boolean indicating whether this is a points property."""
        return self.aps.points_for_part(self.part)

    @property
    def count(self):
        """The count (number of sub-elements per element, usually 1) for this property."""
        return self.aps.count_for_part(self.part)

    @property
    def uom(self):
        """The unit of measure for this property (will be None for discrete or categorical properties)."""
        return self.aps.uom_for_part(self.part)

    @property
    def null_value(self):
        """The null value for this property (will be None for continuous properties, for which NaN is always the null value)."""
        return self.aps.null_value_for_part(self.part)

    @property
    def realization(self):
        """The realisation number for this property (may be None)."""
        return self.aps.realization_for_part(self.part)

    @property
    def time_index(self):
        """The time index for this property (may be None)."""
        return self.aps.time_index_for_part(self.part)

    @property
    def title(self):
        """The citation title for this property (synonymous with citation_title)."""
        return self.aps.citation_title_for_part(self.part)

    @property
    def citation_title(self):
        """The citation title for this property (synonymous with title)."""
        return self.title

    @property
    def min_value(self):
        """The minimum value for this property, as stored in xml metadata."""
        return self.aps.minimum_value_for_part(self.part)

    @property
    def max_value(self):
        """The maximum value for this property, as stored in xml metadata."""
        return self.aps.maximum_value_for_part(self.part)

    @property
    def constant_value(self):
        """The constant value for this property, as stored in xml metadata (usually None)."""
        return self.aps.constant_value_for_part(self.part)

    @property
    def extra(self):
        """The extra metadata for this property (synonymous with extra_metadata)."""
        return self.aps.extra_metadata_for_part(self.part)

    @property
    def extra_metadata(self):
        """The extra metadata for this property (synonymous with extra)."""
        return self.extra

    @property
    def source(self):
        """The source extra metadata value for this property (or None)."""
        return self.aps.source_for_part(self.part)

    @property
    def support_uuid(self):
        """The uuid of the supporting representation for this property."""
        return self.aps.support_uuid_for_part(self.part)

    @property
    def string_lookup_uuid(self):
        """The uuid of the string lookup table for a categorical property (otherwise None)."""
        return self.aps.string_lookup_uuid_for_part(self.part)

    @property
    def time_series_uuid(self):
        """The uuid of the time series for this property (may be None)."""
        return self.aps.time_series_uuid_for_part(self.part)

    @property
    def local_property_kind_uuid(self):
        """The uuid of the local property kind for this property (may be None)."""
        return self.aps.local_property_kind_uuid(self.part)


class AttributePropertySet(rqp.PropertyCollection):
    """Class for set of RESQML properties for any supporting representation, using attribute syntax."""

    def __init__(self,
                 model = None,
                 support = None,
                 property_set_uuid = None,
                 realization = None,
                 key_mode = 'pk',
                 indexable = None,
                 multiple_handling = 'warn'):
        """Initialise an empty property set, optionally populate properties from a supporting representation.

        arguments:
           model (Model, optional): required if property_set_uuid is not None
           support (optional): a grid.Grid object, or a well.BlockedWell, or a well.WellboreFrame object which belongs to a
              resqpy.Model which includes associated properties; if this argument is given, and property_set_root is None,
              the properties in the support's parent model which are for this representation (ie. have this object as the
              supporting representation) are added to this collection as part of the initialisation
           property_set_uuid (optional): if present, the collection is populated with the properties defined in the xml tree
              of the property set
           realization (integer, optional): if present, the single realisation (within an ensemble) that this collection is for;
              if None, then the collection is either covering a whole ensemble (individual properties can each be flagged with a
              realisation number), or is for properties that do not have multiple realizations
           key_mode (str, default 'pk'): either 'pk' (for property kind) or 'title', identifying the basis of property attribute keys
           indexable (str, optional): if present and key_mode is 'pk', properties with indexable element other than this will
              have their indexable element included in their key
           multiple_handling (str, default 'warn'): either 'ignore', 'warn' ,or 'exception'; if 'warn' or 'ignore', and properties
              exist that generate the same key, then only the first is visible in the attribute property set (and a warning is given
              for each of the others in the case of 'warn'); if 'exception', a KeyError is raised if there are any duplicate keys

        note:
           at present, if the collection is being initialised from a property set, the support argument must also be specified;
           also for now, if not initialising from a property set, all properties related to the support are included, whether
           the relationship is supporting representation or some other relationship;

        :meta common:
        """

        assert key_mode in ['pk', 'title']
        assert property_set_uuid is None or model is not None
        assert support is None or model is None or support.model is model
        if property_set_uuid is None:
            property_set_root = None
        else:
            property_set_root = model.root_for_uuid(property_set_uuid)
        assert multiple_handling in ['ignore', 'warn', 'exception']

        super().__init__(support = support, property_set_root = property_set_root, realization = realization)
        self.key_mode = key_mode
        self.indexable_mode = indexable
        self.multiple_handling = multiple_handling
        self._make_attributes()

    def keys(self):
        """Iterator over property keys within the set."""
        for p in self.parts():
            yield self._key(p)

    def properties(self):
        """Iterator over ApsProperty members of the set."""
        for k in self.keys():
            yield getattr(self, k)

    def items(self):
        """Iterator over (key, ApsProperty) members of the set."""
        for k in self.keys():
            yield (k, getattr(self, k))

    def _key(self, part):
        """Returns the key (attribute name) for a given part."""
        return make_aps_key(self.key_mode,
                            property_kind = self.property_kind_for_part(part),
                            title = self.citation_title_for_part(part),
                            facet = self.facet_for_part(part),
                            time_index = self.time_index_for_part(part),
                            realization = self.realization_for_part(part),
                            indexable_mode = self.indexable_mode,
                            indexable = self.indexable_for_part(part))

    def _make_attributes(self):
        """Setup individual properties with attribute style read access to metadata."""
        for part in self.parts():
            key = self._key(part)
            if getattr(self, key, None) is not None:
                if self.multiple_handling == 'warn':
                    log.warning(f'duplicate key in AttributePropertySet; only first instance included: {key}')
                    continue
                if self.multiple_handling == 'ignore':
                    continue
                raise KeyError(f'duplicate key in attribute property set: {key}')
            aps_property = ApsProperty(self, part)
            setattr(self, key, aps_property)

    def __len__(self):
        """Returns the number of properties in the set."""
        return self.number_of_parts()


def make_aps_key(key_mode,
                 property_kind = None,
                 title = None,
                 facet = None,
                 time_index = None,
                 realization = None,
                 indexable_mode = None,
                 indexable = None):
    """Contructs the key (attribute name) for a property based on metadata items."""
    if key_mode == 'pk':
        assert property_kind is not None
        key = property_kind
        if indexable_mode is not None and indexable is not None and indexable != indexable_mode:
            key += f'_{indexable}'
        if facet is not None:
            key += f'_{facet}'
    else:
        assert title is not None
        key = title
    key = key.replace(' ', '_')
    if time_index is not None:
        key += f'_t{time_index}'
    if realization is not None:
        key += f'_r{realization}'
    return key
