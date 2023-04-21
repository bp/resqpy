"""Class for RESQML Structural Organization Interpretation organizational objects."""

from resqpy.olio.base import BaseResqpy

valid_ordering_criteria = ('age', 'apparent depth', 'measured depth')

valid_contact_relationships = ("frontier feature to frontier feature", "genetic boundary to frontier feature",
                               "genetic boundary to genetic boundary", "genetic boundary to tectonic boundary",
                               "stratigraphic unit to frontier feature", "stratigraphic unit to stratigraphic unit",
                               "tectonic boundary to frontier feature", "tectonic boundary to genetic boundary",
                               "tectonic boundary to tectonic boundary")


class StructuralOrganizationInterpretation(BaseResqpy):
    """Class for RESQML Structural Organization Interpretation organizational objects."""

    resqml_type = 'StructuralOrganizationInterpretation'

    def __init__(self):
        """Initialise a structural organisation interpretation."""
        self.ordering_criterion = 'age'  #: one of 'age' (youngest to oldest!), 'apparent depth', or 'measured depth'
        self.fault_uuid_list = None  #: list of uuids of fault interpretation which intersect the structural object
        self.horizon_tuple_list = None  #: list of horizon interpretation uuids along with index and rank
        self.sides_uuid_list = None  #: list of uuids of interpretation objects for sides of structural object
        self.top_frontier_uuid_list = None  #: list of uuids of interpretation objects for top of structural object
        self.bottom_frontier_uuid_list = None  #: list of uuids of interpretation objects for base of structural object
        self.contact_interpretations = None  #: list of contact interpretations

    # TODO
