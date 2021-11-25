"""well_interval_property.py: class for wellintervalproperty, for resqml wellbore frame of blocked wellbore supports."""

version = '24th November 2021'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)
log.debug('property.py version ' + version)

from .property_collection import PropertyCollection


class WellIntervalProperty:
    """Thin wrapper class around interval properties for a Wellbore Frame or Blocked Wellbore.

    ie, interval or cell well logs.
    """

    def __init__(self, collection, part):
        """Create an interval log or blocked well log from a part name."""

        self.collection: PropertyCollection = collection
        self.model = collection.model
        self.part = part

        indexable = self.collection.indexable_for_part(part)
        assert indexable in ['cells', 'intervals'], 'expected cells or intervals as indexable element'

        self.name = self.model.citation_title_for_part(part)
        self.uom = self.collection.uom_for_part(part)

    def values(self):
        """Return interval log or blocked well log as numpy array."""

        return self.collection.cached_part_array_ref(self.part)
