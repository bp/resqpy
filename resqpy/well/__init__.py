"""Classes relating to wells."""

__all__ = [
    'MdDatum', 'BlockedWell', 'Trajectory', 'DeviationSurvey', 'WellboreFrame', 'WellboreMarkerFrame', 'WellboreMarker',
    'add_wells_from_ascii_file', 'well_name', 'add_las_to_trajectory', 'add_blocked_wells_from_wellspec',
    'add_logs_from_cellio', 'lookup_from_cellio'
]

from ._md_datum import MdDatum
from ._wellbore_marker import WellboreMarker
from ._wellbore_marker_frame import WellboreMarkerFrame
from ._blocked_well import BlockedWell
from ._trajectory import Trajectory
from ._deviation_survey import DeviationSurvey
from ._wellbore_frame import WellboreFrame

from .well_object_funcs import add_wells_from_ascii_file, well_name, add_las_to_trajectory, \
    add_blocked_wells_from_wellspec, add_logs_from_cellio, lookup_from_cellio

# Set "module" attribute of all public objects to this path. Fixes issue #310
for _name in __all__:
    _obj = eval(_name)
    if hasattr(_obj, "__module__"):
        _obj.__module__ = __name__
