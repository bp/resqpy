"""Class for a welllog, representing resqml properties for well logs"""

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)

import resqpy.property
import resqpy.property.property_collection as rqp_pc


class WellLog:
    """Thin wrapper class around RESQML properties for well logs."""

    def __init__(self, collection, uuid):
        """Create a well log from a part name."""

        self.collection: rqp_pc.PropertyCollection = collection
        self.model = collection.model
        self.uuid = uuid

        part = self.model.part_for_uuid(uuid)
        indexable = self.collection.indexable_for_part(part)
        if indexable != 'nodes':
            raise NotImplementedError('well frame related property does not have nodes as indexable element')

        #: Name of log
        self.title = self.model.citation_title_for_part(part)

        #: Unit of measure
        self.uom = self.collection.uom_for_part(part)

    def values(self):
        """Return log data as numpy array.

        Note:
           may return 2D numpy array with shape (num_depths, num_columns).
        """

        part = self.model.part_for_uuid(self.uuid)
        return self.collection.cached_part_array_ref(part)
