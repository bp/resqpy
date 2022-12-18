"""Class for RESQML Boundary Feature organizational objects."""

import resqpy.olio.uuid as bu
import resqpy.organize
import resqpy.organize._utils as ou
from resqpy.olio.base import BaseResqpy


class BoundaryFeature(BaseResqpy):
    """Class for RESQML Boundary Feature organizational objects."""

    resqml_type = "BoundaryFeature"
    feature_name = ou.alias_for_attribute("title")

    def __init__(self, parent_model, uuid = None, feature_name = None, extra_metadata = None):
        """Initialises a boundary feature organisational object."""

        super().__init__(model = parent_model, uuid = uuid, title = feature_name, extra_metadata = extra_metadata)

    def is_equivalent(self, other, check_extra_metadata = True):
        """Returns True if this feature is essentially the same as the other; otherwise False."""

        if other is None or not isinstance(other, BoundaryFeature):
            return False
        if self is other or bu.matching_uuids(self.uuid, other.uuid):
            return True
        if check_extra_metadata and not ou.equivalent_extra_metadata(self, other):
            return False
        return self.feature_name == other.feature_name

    def create_xml(self, add_as_part = True, originator = None, reuse = True):
        """Creates a geobody feature xml node from this geobody feature object."""
        if reuse and self.try_reuse():
            return self.root  # check for reusable (equivalent) object
        return super().create_xml(add_as_part = add_as_part, originator = originator)
