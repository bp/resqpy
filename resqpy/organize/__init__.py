"""Organizational object classes: features and interpretations."""

__all__ = [
    # Classes
    "BoundaryFeature",
    "BoundaryFeatureInterpretation",
    "EarthModelInterpretation",
    "FaultInterpretation",
    "FluidBoundaryFeature",
    "FrontierFeature",
    "GenericInterpretation",
    "GeneticBoundaryFeature",
    "GeobodyBoundaryInterpretation",
    "GeobodyFeature",
    "GeobodyInterpretation",
    "GeologicUnitFeature",
    "HorizonInterpretation",
    "OrganizationFeature",
    "RockFluidUnitFeature",
    "TectonicBoundaryFeature",
    "WellboreFeature",
    "WellboreInterpretation",
    # Functions
    "equivalent_extra_metadata",
    "alias_for_attribute",
    "extract_has_occurred_during",
    "equivalent_chrono_pairs",
    "create_xml_has_occurred_during",
]

from resqpy.organize._utils import (equivalent_extra_metadata, alias_for_attribute, extract_has_occurred_during,
                                    equivalent_chrono_pairs, create_xml_has_occurred_during)
from .boundary_feature import BoundaryFeature
from .boundary_feature_interpretation import BoundaryFeatureInterpretation
from .earth_model_interpretation import EarthModelInterpretation
from .fault_interpretation import FaultInterpretation
from .fluid_boundary_feature import FluidBoundaryFeature
from .frontier_feature import FrontierFeature
from .generic_interpretation import GenericInterpretation
from .genetic_boundary_feature import GeneticBoundaryFeature
from .geobody_boundary_interpretation import GeobodyBoundaryInterpretation
from .geobody_feature import GeobodyFeature
from .geobody_interpretation import GeobodyInterpretation
from .geologic_unit_feature import GeologicUnitFeature
from .horizon_interpretation import HorizonInterpretation
from .organization_feature import OrganizationFeature
from .rock_fluid_unit_feature import RockFluidUnitFeature
from .tectonic_boundary_feature import TectonicBoundaryFeature
from .wellbore_feature import WellboreFeature
from .wellbore_interpretation import WellboreInterpretation

# Set "module" attribute of all public objects to this path.
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
