"""Custom exceptions used in resqpy."""


class InvalidUnitError(Exception):
    """Raised when a unit cannot be converted into a valid RESQML unit of measure."""
    pass


class IncompatibleUnitsError(Exception):
    """Raised when two units do not share compatible base units and dimensions."""
    pass
