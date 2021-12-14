"""Class for a collection of well interval properties"""

version = '24th November 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import pandas as pd
from .property_collection import PropertyCollection
from .well_interval_property import WellIntervalProperty
from .property_common import return_cell_indices


class WellIntervalPropertyCollection(PropertyCollection):
    """Class for RESQML property collection for a WellboreFrame for interval or blocked well logs"""

    def __init__(self, frame = None, property_set_root = None, realization = None):
        """Creates a new property collection related to interval or blocked well logs and a wellbore frame."""

        super().__init__(support = frame, property_set_root = property_set_root, realization = realization)

    def logs(self):
        """Generator that yields component Interval log or Blocked well log objects."""

        return (WellIntervalProperty(collection = self, part = part) for part in self.parts())

    def to_pandas(self, include_units = False):
        """Returns a dataframe with a column for each well log included in the collection."""

        cell_indices = [return_cell_indices(i, self.support.cell_indices) for i in self.support.cell_grid_link]
        data = {}
        for log in self.logs():
            col_name = log.name
            values = log.values()
            data[col_name] = values
        df = pd.DataFrame(data = data, index = cell_indices)
        return df
