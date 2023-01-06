"""Weights and measures valid units and unit conversion functions."""

__all__ = [
    'UOM_ALIASES', 'UOM_ALIAS_MAP', 'CASE_INSENSITIVE_UOMS', 'rq_uom', 'convert', 'valid_uoms', 'valid_quantities',
    'valid_property_kinds', 'nexus_uom_for_quantity', 'rq_uom_list', 'rq_length_unit', 'rq_time_unit', 'convert_times',
    'convert_lengths', 'convert_pressures', 'convert_volumes', 'convert_flow_rates', 'get_conversion_factors'
]

from .weights_and_measures import (UOM_ALIASES, UOM_ALIAS_MAP, CASE_INSENSITIVE_UOMS, rq_uom, convert, valid_uoms,
                                   valid_quantities, valid_property_kinds, rq_uom_list, rq_length_unit, rq_time_unit,
                                   convert_times, convert_lengths, convert_pressures, convert_volumes,
                                   convert_flow_rates, get_conversion_factors)
from .nexus_units import nexus_uom_for_quantity

# Set "module" attribute of all public objects to this path.
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
