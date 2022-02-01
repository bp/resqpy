"""Stratigraphy related classes and valid values."""

__all__ = [
    'BinaryContactInterpretation', 'GeologicUnitInterpretation', 'StratigraphicColumnRank', 'StratigraphicColumn',
    'StratigraphicUnitFeature', 'StratigraphicUnitInterpretation', 'valid_compositions', 'valid_implacements',
    'valid_domains', 'valid_deposition_modes', 'valid_ordering_criteria', 'valid_contact_relationships',
    'valid_contact_verbs', 'valid_contact_sides', 'valid_contact_modes'
]

from ._strata_common import valid_compositions, valid_implacements, valid_domains, valid_deposition_modes,  \
    valid_ordering_criteria, valid_contact_relationships, valid_contact_verbs, valid_contact_sides,  \
    valid_contact_modes
from ._binary_contact_interpretation import BinaryContactInterpretation
from ._geologic_unit_interpretation import GeologicUnitInterpretation
from ._stratigraphic_column_rank import StratigraphicColumnRank
from ._stratigraphic_column import StratigraphicColumn
from ._stratigraphic_unit_feature import StratigraphicUnitFeature
from ._stratigraphic_unit_interpretation import StratigraphicUnitInterpretation

# Set "module" attribute of all public objects to this path. Fixes issue #310
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
