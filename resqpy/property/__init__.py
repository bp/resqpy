"""Collections of properties for grids, wellbore frames, grid connection sets etc."""

__all__ = [
    'PropertyCollection', 'Property', 'WellLog', 'WellIntervalProperty', 'WellIntervalPropertyCollection',
    'WellLogCollection', 'StringLookup', 'PropertyKind', 'GridPropertyCollection',
    'property_over_time_series_from_collection', 'property_collection_for_keyword', 'infer_property_kind',
    'write_hdf5_and_create_xml_for_active_property', 'reformat_column_edges_to_resqml_format',
    'reformat_column_edges_from_resqml_format', 'same_property_kind', 'selective_version_of_collection',
    'supported_local_property_kind_list', 'supported_property_kind_list', 'supported_facet_type_list',
    'expected_facet_type_dict', 'create_transmisibility_multiplier_property_kind',
    'property_kind_and_facet_from_keyword', 'guess_uom', 'property_parts', 'property_part'
]

from .property_common import property_collection_for_keyword,  \
    property_over_time_series_from_collection,  \
    write_hdf5_and_create_xml_for_active_property,  \
    infer_property_kind,  \
    reformat_column_edges_to_resqml_format,  \
    reformat_column_edges_from_resqml_format,  \
    selective_version_of_collection,  \
    same_property_kind,  \
    supported_property_kind_list,  \
    supported_local_property_kind_list,  \
    supported_facet_type_list,  \
    expected_facet_type_dict,  \
    property_kind_and_facet_from_keyword,  \
    guess_uom,  \
    property_parts,  \
    property_part
from .property_kind import PropertyKind, create_transmisibility_multiplier_property_kind
from .string_lookup import StringLookup
from .property_collection import PropertyCollection
from .grid_property_collection import GridPropertyCollection
from ._property import Property
from .well_interval_property import WellIntervalProperty
from .well_interval_property_collection import WellIntervalPropertyCollection
from .well_log import WellLog
from .well_log_collection import WellLogCollection

# Set "module" attribute of all public objects to this path.
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
